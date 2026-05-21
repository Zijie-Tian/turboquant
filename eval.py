#!/usr/bin/env python3
"""Configurable TurboQuant/vLLM evaluation entrypoint.

Examples:
    python eval.py --model /mnt/data/tzj/models/Llama-3.1-8B-Instruct \
        --mode tq --prompt "Explain KV cache quantization." --max-tokens 64

    python eval.py --model /path/to/model --mode baseline --input-file prompt.txt \
        --output-file result.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ["PYTHONPATH"] = (
    str(REPO_ROOT)
    if not os.environ.get("PYTHONPATH")
    else f"{REPO_ROOT}{os.pathsep}{os.environ['PYTHONPATH']}"
)

DEFAULT_MODEL = os.environ.get(
    "MODEL", "/mnt/data/tzj/models/Llama-3.1-8B-Instruct"
)


@dataclass
class EvalRequest:
    model: str
    mode: str
    prompt: str
    max_tokens: int
    temperature: float
    top_p: float
    max_model_len: int
    tensor_parallel_size: int
    gpu_memory_utilization: float
    dtype: str
    key_bits: int
    value_bits: int
    buffer_size: int
    initial_layers_count: int
    tq_mode: str
    use_triton_score: bool


@dataclass
class EvalResult:
    ok: bool
    request: EvalRequest
    answer: str = ""
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    load_seconds: float | None = None
    generate_seconds: float | None = None
    output_tokens_per_second: float | None = None
    total_tokens_per_second: float | None = None
    tq_install: Any = None
    tq_stats: Any = None
    freed_bytes: Any = None
    vram_before: Any = None
    vram_after_load: Any = None
    vram_after_generate: Any = None
    vram_after_free: Any = None
    expectation: dict[str, Any] | None = None
    error: str | None = None
    traceback: str | None = None


def read_prompt(args: argparse.Namespace) -> str:
    sources = [args.prompt is not None, args.input_file is not None, args.stdin]
    if sum(sources) != 1:
        raise ValueError("Provide exactly one of --prompt, --input-file, or --stdin")

    if args.prompt is not None:
        prompt = args.prompt
    elif args.input_file is not None:
        prompt = Path(args.input_file).read_text(encoding="utf-8")
    else:
        prompt = sys.stdin.read()

    prompt = prompt.rstrip("\n")
    if not prompt.strip():
        raise ValueError("Prompt is empty")
    return prompt


def gpu_memory() -> list[dict[str, Any]]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        rows = []
        for line in proc.stdout.strip().splitlines():
            if not line.strip():
                continue
            index, used, total, util = [part.strip() for part in line.split(",")]
            rows.append(
                {
                    "index": int(index),
                    "used_mb": int(used),
                    "total_mb": int(total),
                    "util_pct": int(util),
                }
            )
        return rows
    except Exception as exc:  # pragma: no cover - depends on system nvidia-smi
        return [{"error": repr(exc)}]


def get_executor(llm: Any) -> Any:
    engine = llm.llm_engine
    core = getattr(engine, "engine_core", engine)
    inner = getattr(core, "engine_core", core)
    return inner.model_executor


def collective_rpc(executor: Any, fn: Any) -> list[Any]:
    if hasattr(executor, "collective_rpc"):
        return executor.collective_rpc(fn)
    if hasattr(executor, "driver_worker"):
        return [fn(executor.driver_worker)]
    raise RuntimeError(f"Cannot find worker RPC API on executor {type(executor)!r}")


def check_expectation(answer: str, expected: str | None, contains: str | None) -> dict[str, Any] | None:
    if expected is None and contains is None:
        return None
    result: dict[str, Any] = {}
    if expected is not None:
        result["expected_answer"] = expected
        result["exact_match"] = answer.strip() == expected.strip()
    if contains is not None:
        result["expected_contains"] = contains
        result["contains"] = contains in answer
    result["passed"] = all(
        value for key, value in result.items() if key in {"exact_match", "contains"}
    )
    return result


def install_turboquant(executor: Any, args: argparse.Namespace) -> Any:
    key_bits = args.key_bits
    value_bits = args.value_bits
    buffer_size = args.buffer_size
    initial_layers_count = args.initial_layers_count
    tq_mode = args.tq_mode

    def _install(worker: Any) -> dict[str, Any]:
        from turboquant.vllm_attn_backend import (
            MODE_ACCUMULATE,
            MODE_ACTIVE,
            MODE_SHADOW,
            install_turboquant_hooks,
        )

        mode_map = {
            "active": MODE_ACTIVE,
            "accumulate": MODE_ACCUMULATE,
            "shadow": MODE_SHADOW,
        }
        states = install_turboquant_hooks(
            worker.model_runner,
            key_bits=key_bits,
            value_bits=value_bits,
            buffer_size=buffer_size,
            initial_layers_count=initial_layers_count,
            mode=mode_map[tq_mode],
        )
        return {
            "num_layers": len(states),
            "supports_hybrid": sum(
                1 for state in states.values() if getattr(state, "supports_hybrid", False)
            ),
            "head_dims": sorted({state.config.head_dim for state in states.values()}),
        }

    return collective_rpc(executor, _install)


def get_tq_stats(executor: Any) -> Any:
    def _stats(worker: Any) -> dict[str, Any]:
        from turboquant.integration.vllm import get_stats

        return get_stats(worker.model_runner)

    return collective_rpc(executor, _stats)


def free_tq_kv_cache(executor: Any) -> Any:
    def _free(worker: Any) -> int:
        from turboquant.vllm_attn_backend import free_kv_cache

        return free_kv_cache(worker.model_runner)

    return collective_rpc(executor, _free)


def run_eval(args: argparse.Namespace) -> EvalResult:
    prompt = read_prompt(args)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    if args.use_triton_score:
        os.environ["TURBOQUANT_USE_TRITON_SCORE"] = "1"
    else:
        os.environ.pop("TURBOQUANT_USE_TRITON_SCORE", None)

    request = EvalRequest(
        model=args.model,
        mode=args.mode,
        prompt=prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        key_bits=args.key_bits,
        value_bits=args.value_bits,
        buffer_size=args.buffer_size,
        initial_layers_count=args.initial_layers_count,
        tq_mode=args.tq_mode,
        use_triton_score=args.use_triton_score,
    )
    result = EvalResult(ok=False, request=request, vram_before=gpu_memory())

    try:
        import torch
        from vllm import LLM, SamplingParams

        load_start = time.perf_counter()
        llm = LLM(
            model=args.model,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            trust_remote_code=args.trust_remote_code,
            max_num_seqs=args.max_num_seqs,
            enforce_eager=args.enforce_eager,
        )
        result.load_seconds = round(time.perf_counter() - load_start, 4)
        result.vram_after_load = gpu_memory()

        executor = None
        if args.mode == "tq":
            executor = get_executor(llm)
            result.tq_install = install_turboquant(executor, args)

        tokenizer = llm.get_tokenizer()
        prompt_tokens = len(tokenizer.encode(prompt))

        sampling = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        generate_start = time.perf_counter()
        outputs = llm.generate([prompt], sampling)
        generate_seconds = time.perf_counter() - generate_start

        output = outputs[0].outputs[0]
        answer = output.text
        output_tokens = len(output.token_ids)

        result.ok = bool(answer)
        result.answer = answer
        result.prompt_tokens = prompt_tokens
        result.output_tokens = output_tokens
        result.generate_seconds = round(generate_seconds, 4)
        result.output_tokens_per_second = round(output_tokens / max(generate_seconds, 1e-9), 4)
        result.total_tokens_per_second = round(
            (prompt_tokens + output_tokens) / max(generate_seconds, 1e-9), 4
        )
        result.vram_after_generate = gpu_memory()
        result.expectation = check_expectation(
            answer, args.expected_answer, args.expect_contains
        )

        if args.mode == "tq" and executor is not None:
            result.tq_stats = get_tq_stats(executor)
            if args.free_kv_cache:
                result.freed_bytes = free_tq_kv_cache(executor)
                result.vram_after_free = gpu_memory()

        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        result.traceback = traceback.format_exc(limit=12)
        result.vram_after_generate = gpu_memory()

    return result


def result_to_dict(result: EvalResult) -> dict[str, Any]:
    return asdict(result)


def write_output(payload: str, output_file: str | None) -> None:
    if output_file:
        Path(output_file).write_text(payload, encoding="utf-8")
    else:
        print(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a configurable vLLM/TurboQuant eval prompt.")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--prompt", help="prompt text to evaluate")
    input_group.add_argument("--input-file", help="UTF-8 text file containing the prompt")
    input_group.add_argument("--stdin", action="store_true", help="read prompt from standard input")

    parser.add_argument("--model", default=DEFAULT_MODEL, help="model path or HF id")
    parser.add_argument("--mode", choices=("baseline", "tq"), default="tq")
    parser.add_argument("--output-file", help="write result to this file instead of stdout")
    parser.add_argument("--output-format", choices=("json", "text"), default="json")
    parser.add_argument("--expected-answer", help="optional exact-match expected answer")
    parser.add_argument("--expect-contains", help="optional substring expected in the answer")

    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", "--tp", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--key-bits", type=int, default=3)
    parser.add_argument("--value-bits", type=int, default=2)
    parser.add_argument("--buffer-size", type=int, default=64)
    parser.add_argument("--initial-layers-count", type=int, default=2)
    parser.add_argument("--tq-mode", choices=("active", "accumulate", "shadow"), default="active")
    parser.add_argument("--use-triton-score", action="store_true")
    parser.add_argument("--free-kv-cache", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_eval(args)

    if args.output_format == "text":
        payload = result.answer
    else:
        payload = json.dumps(result_to_dict(result), ensure_ascii=False, indent=2)
    write_output(payload, args.output_file)

    if result.expectation is not None and not result.expectation.get("passed", False):
        return 2
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
