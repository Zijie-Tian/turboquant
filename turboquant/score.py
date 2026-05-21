"""
TurboQuant score module — attention computation over compressed + exact segments.

Handles the read path:
  - Compute attention scores over compressed historical KV (via Triton or PyTorch fallback)
  - Compute attention scores over exact recent buffer (via standard matmul / SDPA)
  - Merge logits and weighted values from both segments

Design rule: compressed path is only invoked when history is large enough
to justify it (>= 16 tokens).
"""

from __future__ import annotations

import math
import logging
import os
import torch
import torch.nn.functional as F
from typing import Optional

from turboquant.store import FlatCache, CompressedKVStore
from turboquant.kv_cache import dequantize_values
from turboquant.quantizer import TurboQuantProd

logger = logging.getLogger("turboquant.score")

MIN_HISTORY_FOR_TQ = 16


def _use_triton_score(query: torch.Tensor) -> bool:
    """Experimental fast path gate for Triton compressed-key score kernels."""
    return (
        query.is_cuda
        and os.environ.get("TURBOQUANT_USE_TRITON_SCORE", "").lower()
        in ("1", "true", "yes", "on")
    )


def compute_hybrid_attention(
    query: torch.Tensor,
    store: CompressedKVStore,
    recent_k: Optional[torch.Tensor],
    recent_v: Optional[torch.Tensor],
    num_query_heads: int,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Compute attention output combining compressed history and exact recent buffer.

    Args:
        query: (num_tokens, num_query_heads, head_dim) — typically num_tokens=1 for decode
        store: compressed KV store with historical tokens
        recent_k: (recent_len, num_kv_heads, head_dim) or None
        recent_v: (recent_len, num_kv_heads, head_dim) or None
        num_query_heads: total query heads (for GQA expansion)
        scale: attention scale factor (default: 1/sqrt(head_dim))

    Returns:
        output: (num_tokens, num_query_heads, head_dim)
    """
    head_dim = store.head_dim
    num_kv_heads = store.num_kv_heads
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    flat = store.get_flat_cache()
    has_history = flat is not None and flat.num_tokens >= MIN_HISTORY_FOR_TQ
    has_recent = recent_k is not None and recent_k.shape[0] > 0

    if not has_history and not has_recent:
        return torch.zeros(
            query.shape[0], num_query_heads, head_dim,
            device=query.device, dtype=query.dtype,
        )

    gqa_ratio = num_query_heads // num_kv_heads

    if has_history and not has_recent:
        return _attend_compressed_only(
            query, flat, store.quantizer, gqa_ratio, num_kv_heads, scale
        )

    if not has_history and has_recent:
        return _attend_exact_only(
            query, recent_k, recent_v, gqa_ratio, num_kv_heads, scale
        )

    # Both segments present — merge via log-sum-exp trick
    return _attend_hybrid(
        query, flat, store.quantizer, recent_k, recent_v,
        gqa_ratio, num_kv_heads, head_dim, scale,
    )


def _attend_compressed_only(
    query: torch.Tensor,
    flat: FlatCache,
    quantizer: TurboQuantProd,
    gqa_ratio: int,
    num_kv_heads: int,
    scale: float,
) -> torch.Tensor:
    """Attention over compressed history only (PyTorch path)."""
    if _use_triton_score(query):
        triton_out = _attend_compressed_only_triton_score(
            query, flat, quantizer, gqa_ratio, num_kv_heads, scale
        )
        if triton_out is not None:
            return triton_out

    k_dequant = quantizer.dequantize(flat.prod_q)  # (H_kv, N, D)
    v_dequant = dequantize_values(flat.value_q, 32)

    return _matmul_attend(query, k_dequant, v_dequant, gqa_ratio, num_kv_heads, scale)


def _attend_exact_only(
    query: torch.Tensor,
    recent_k: torch.Tensor,
    recent_v: torch.Tensor,
    gqa_ratio: int,
    num_kv_heads: int,
    scale: float,
) -> torch.Tensor:
    """Attention over exact recent buffer only."""
    return _matmul_attend(
        query, recent_k.transpose(0, 1), recent_v.transpose(0, 1),
        gqa_ratio, num_kv_heads, scale,
    )


def _attend_hybrid(
    query: torch.Tensor,
    flat: FlatCache,
    quantizer: TurboQuantProd,
    recent_k: torch.Tensor,
    recent_v: torch.Tensor,
    gqa_ratio: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
) -> torch.Tensor:
    """Merge compressed history + exact recent via concatenated attention."""
    if _use_triton_score(query):
        triton_out = _attend_hybrid_triton_score(
            query, flat, quantizer, recent_k, recent_v,
            gqa_ratio, num_kv_heads, scale,
        )
        if triton_out is not None:
            return triton_out

    k_hist = quantizer.dequantize(flat.prod_q)  # (H_kv, N_hist, D)
    v_hist = dequantize_values(flat.value_q, 32)

    k_recent = recent_k.transpose(0, 1)   # (H_kv, N_recent, D)
    v_recent = recent_v.transpose(0, 1)

    k_all = torch.cat([k_hist.float(), k_recent.float()], dim=1)
    v_all = torch.cat([v_hist.float(), v_recent.float()], dim=1)

    return _matmul_attend(query, k_all, v_all, gqa_ratio, num_kv_heads, scale)


def _compressed_scores_triton_gqa(
    query: torch.Tensor,
    flat: FlatCache,
    quantizer: TurboQuantProd,
    gqa_ratio: int,
    num_kv_heads: int,
    scale: float,
) -> Optional[torch.Tensor]:
    """Return scaled compressed-history logits with the Triton GQA score kernel.

    Returns shape (T, Q_heads, N). Currently decode-only (T=1); callers fall
    back to the PyTorch path for prefill or unsupported layouts.
    """
    if query.shape[0] != 1:
        return None

    try:
        from turboquant.triton_kernels import turboquant_attention_score_gqa
    except Exception as exc:
        logger.debug("[TurboQuant] Triton score import failed; falling back: %s", exc)
        return None

    prod_q = flat.prod_q
    if prod_q.mse_indices.dim() != 3:
        return None
    if prod_q.mse_indices.shape[0] != num_kv_heads:
        return None

    raw_scores = turboquant_attention_score_gqa(
        query=query,
        quantized_key=prod_q,
        Pi=quantizer.mse_quantizer.Pi,
        S=quantizer.S,
        centroids=quantizer.mse_quantizer.centroids,
        mse_bits=prod_q.mse_bits,
        qjl_scale=quantizer.qjl_scale,
        gqa_ratio=gqa_ratio,
    )
    return raw_scores.unsqueeze(0) * scale


def _recent_scores(
    query: torch.Tensor,
    recent_k: torch.Tensor,
    gqa_ratio: int,
    num_kv_heads: int,
    scale: float,
) -> torch.Tensor:
    """Compute scaled exact recent logits without expanding KV heads."""
    T, Q, D = query.shape
    q = query.float().view(T, num_kv_heads, gqa_ratio, D)
    k = recent_k.transpose(0, 1).float()  # (H_kv, N_recent, D)
    scores = torch.einsum("thgd,hnd->thgn", q, k) * scale
    return scores.reshape(T, Q, recent_k.shape[0])


def _attend_from_grouped_logits(
    query: torch.Tensor,
    logits: torch.Tensor,
    values: torch.Tensor,
    gqa_ratio: int,
    num_kv_heads: int,
) -> torch.Tensor:
    """Softmax logits and apply grouped values without repeating KV tensors."""
    T, Q, D = query.shape
    weights = F.softmax(logits, dim=-1)
    w = weights.view(T, num_kv_heads, gqa_ratio, logits.shape[-1])
    out = torch.einsum("thgn,hnd->thgd", w.float(), values.float())
    return out.reshape(T, Q, D).to(query.dtype)


def _attend_compressed_only_triton_score(
    query: torch.Tensor,
    flat: FlatCache,
    quantizer: TurboQuantProd,
    gqa_ratio: int,
    num_kv_heads: int,
    scale: float,
) -> Optional[torch.Tensor]:
    scores = _compressed_scores_triton_gqa(
        query, flat, quantizer, gqa_ratio, num_kv_heads, scale
    )
    if scores is None:
        return None

    v_dequant = dequantize_values(flat.value_q, 32)
    return _attend_from_grouped_logits(
        query, scores, v_dequant, gqa_ratio, num_kv_heads
    )


def _attend_hybrid_triton_score(
    query: torch.Tensor,
    flat: FlatCache,
    quantizer: TurboQuantProd,
    recent_k: torch.Tensor,
    recent_v: torch.Tensor,
    gqa_ratio: int,
    num_kv_heads: int,
    scale: float,
) -> Optional[torch.Tensor]:
    hist_scores = _compressed_scores_triton_gqa(
        query, flat, quantizer, gqa_ratio, num_kv_heads, scale
    )
    if hist_scores is None:
        return None

    recent_scores = _recent_scores(query, recent_k, gqa_ratio, num_kv_heads, scale)
    logits = torch.cat([hist_scores, recent_scores], dim=-1)

    v_hist = dequantize_values(flat.value_q, 32)
    v_recent = recent_v.transpose(0, 1)
    values = torch.cat([v_hist.float(), v_recent.float()], dim=1)

    return _attend_from_grouped_logits(
        query, logits, values, gqa_ratio, num_kv_heads
    )


def _matmul_attend(
    query: torch.Tensor,
    kv_keys: torch.Tensor,
    kv_values: torch.Tensor,
    gqa_ratio: int,
    num_kv_heads: int,
    scale: float,
) -> torch.Tensor:
    """Standard matmul attention with GQA support.

    query: (T, Q_heads, D)
    kv_keys: (H_kv, N, D)
    kv_values: (H_kv, N, D)

    Returns: (T, Q_heads, D)
    """
    T, Q, D = query.shape
    H_kv = num_kv_heads
    if Q != H_kv * gqa_ratio:
        raise ValueError(
            f"Incompatible GQA shapes: Q={Q}, H_kv={H_kv}, gqa_ratio={gqa_ratio}"
        )

    # Avoid repeat_interleave(Q/H) on KV tensors to keep memory bounded at long context.
    # q: (T, Q, D) -> (H_kv, G, T, D)
    q = query.float().view(T, H_kv, gqa_ratio, D).permute(1, 2, 0, 3)
    k = kv_keys.float().unsqueeze(1)   # (H_kv, 1, N, D) broadcast over G
    v = kv_values.float().unsqueeze(1) # (H_kv, 1, N, D) broadcast over G

    # scores: (H_kv, G, T, N)
    scores = torch.einsum("hgtd,hgnd->hgtn", q, k) * scale
    weights = F.softmax(scores, dim=-1)
    out = torch.einsum("hgtn,hgnd->hgtd", weights, v)

    # Back to (T, Q, D)
    return out.permute(2, 0, 1, 3).reshape(T, Q, D).to(query.dtype)
