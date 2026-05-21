#!/usr/bin/env python3
"""LongBench evaluation entrypoint for vLLM + TurboQuant.

This runner mirrors the prompt/chat-template behavior from LUTAttn while using
this repository's eval.py TurboQuant installation path.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ["PYTHONPATH"] = (
    str(REPO_ROOT)
    if not os.environ.get("PYTHONPATH")
    else f"{REPO_ROOT}{os.pathsep}{os.environ['PYTHONPATH']}"
)

from turboquant.longbench import (  # noqa: E402
    count_jsonl,
    format_longbench_prompt,
    load_env_file,
    load_json_config,
    load_longbench_dataset,
    normalize_output_model_name,
    output_path_for_run,
    post_process,
    remaining_after_resume,
    resolve_max_gen,
    resolve_max_model_len,
    resolve_model_path,
    select_samples,
)


def _eval_entry():
    return importlib.import_module("eval")


def parse_visible_gpus() -> list[int] | None:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not raw:
        return None
    visible: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if item.isdigit():
            visible.append(int(item))
    return visible or None


def reset_tq_state(executor: Any) -> list[dict[str, Any]]:
    """Reset per-worker TurboQuant compressed stores/ring buffers."""
    eval_entry = _eval_entry()

    def _reset(worker: Any) -> dict[str, Any]:
        model_runner = worker.model_runner
        layer_states = getattr(model_runner, "_tq_layer_states", None) or getattr(
            model_runner, "_tq_states", None
        )
        if not layer_states:
            return {"num_layers": 0, "reset": False}
        for state in layer_states.values():
            if hasattr(state, "reset"):
                state.reset()
            else:  # pragma: no cover - compatibility fallback for legacy states
                engine = getattr(state, "engine", None)
                store = getattr(state, "store", None)
                if engine is not None and hasattr(engine, "reset"):
                    engine.reset()
                elif store is not None and hasattr(store, "reset"):
                    store.reset()
        return {"num_layers": len(layer_states), "reset": True}

    return eval_entry.collective_rpc(executor, _reset)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate LongBench with vLLM/TurboQuant using LUTAttn-compatible prompts."
    )
    parser.add_argument("--model-name", "--model_name", default=os.environ.get("MODEL", "llama3.1-8b"))
    parser.add_argument("--model-path", "--model_path", default=os.environ.get("MODEL_PATH"))
    parser.add_argument("--mode", choices=("baseline", "tq"), default="tq")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")
    parser.add_argument("--max-samples", "--max_samples", type=int, default=-1)
    parser.add_argument("--max-model-len", "--max_model_len", type=int, default=None)
    parser.add_argument("--max-gen", "--max_gen", type=int, default=None)
    parser.add_argument("--prompt-token-reserve", type=int, default=256)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--overwrite", action="store_true", help="Remove existing dataset output before running")

    parser.add_argument("--tp-size", "--tp", dest="tp_size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--disable-chunked-prefill", action="store_true")
    parser.add_argument("--enable-triattention", action="store_true")

    parser.add_argument("--key-bits", type=int, default=3)
    parser.add_argument("--value-bits", type=int, default=2)
    parser.add_argument("--buffer-size", type=int, default=64)
    parser.add_argument("--initial-layers-count", type=int, default=2)
    parser.add_argument("--tq-mode", choices=("active", "accumulate", "shadow"), default="active")
    parser.add_argument("--use-triton-score", action="store_true")
    parser.add_argument("--free-kv-cache", action="store_true")

    parser.add_argument("--report-json", "--metadata-json", "--json-output", dest="report_json")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(args: argparse.Namespace, max_model_len: int, max_gen: int) -> None:
    if args.tp_size < 1:
        raise SystemExit("--tp-size must be >= 1")
    if args.max_samples == 0 or args.max_samples < -1:
        raise SystemExit("--max-samples must be -1 or a positive integer")
    if args.max_num_seqs < 1:
        raise SystemExit("--max-num-seqs must be >= 1")
    if args.mode == "tq" and args.max_num_seqs != 1:
        raise SystemExit("TurboQuant LongBench currently requires --max-num-seqs 1")
    if max_gen >= max_model_len:
        raise SystemExit(
            f"Dataset max generation ({max_gen}) must be smaller than max-model-len ({max_model_len})"
        )
    visible = parse_visible_gpus()
    if visible is not None and len(visible) < args.tp_size:
        raise SystemExit(f"Need at least tp-size={args.tp_size} visible GPUs; got {visible}")


def build_engine_kwargs(args: argparse.Namespace, model_path: str, max_model_len: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model_path,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": max_model_len,
        "tensor_parallel_size": args.tp_size,
        "trust_remote_code": args.trust_remote_code,
        "max_num_seqs": args.max_num_seqs,
        "enforce_eager": args.enforce_eager,
    }
    if args.seed is not None:
        kwargs["seed"] = args.seed
    if args.max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.disable_chunked_prefill:
        kwargs["enable_chunked_prefill"] = False
    return kwargs


def write_report(path: str | None, report: dict[str, Any]) -> None:
    if not path:
        return
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[report] wrote {report_path}")


def run_longbench(args: argparse.Namespace) -> int:
    load_env_file()
    if not args.enable_triattention:
        os.environ["ENABLE_TRIATTENTION"] = "0"
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.use_triton_score:
        os.environ["TURBOQUANT_USE_TRITON_SCORE"] = "1"
    else:
        os.environ.pop("TURBOQUANT_USE_TRITON_SCORE", None)

    model_path, model_path_source = resolve_model_path(args.model_name, args.model_path)
    max_model_len, max_model_len_source = resolve_max_model_len(args.model_name, args.max_model_len)
    max_gen, max_gen_source = resolve_max_gen(args.dataset, args.max_gen)
    validate_args(args, max_model_len=max_model_len, max_gen=max_gen)

    mode_tag = args.mode if args.output_tag is None else f"{args.mode}-{args.output_tag}"
    output_path = output_path_for_run(
        args.model_name,
        args.dataset,
        args.e,
        args.mode,
        output_root=args.output_root,
        tag=args.output_tag,
    )
    dataset_name = f"{args.dataset}_e" if args.e else args.dataset
    prompt_format = load_json_config("dataset2prompt.json")[args.dataset]
    engine_kwargs = build_engine_kwargs(args, model_path, max_model_len)

    report_base: dict[str, Any] = {
        "requested_backend": "vllm",
        "actual_backend": "vllm_turboquant" if args.mode == "tq" else "vllm",
        "mode": args.mode,
        "model_name": args.model_name,
        "normalized_model_name": normalize_output_model_name(args.model_name),
        "model_path": model_path,
        "model_path_source": model_path_source,
        "dataset": args.dataset,
        "dataset_name": dataset_name,
        "max_samples": args.max_samples,
        "max_model_len": max_model_len,
        "max_model_len_source": max_model_len_source,
        "max_gen": max_gen,
        "max_gen_source": max_gen_source,
        "prompt_token_reserve": args.prompt_token_reserve,
        "tp_size": args.tp_size,
        "visible_gpus": parse_visible_gpus(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "output_path": str(output_path),
        "output_model_tag": mode_tag,
        "engine_kwargs": engine_kwargs,
        "triattention_enabled": args.enable_triattention,
    }

    if args.dry_run:
        report = {"status": "dry_run", **report_base}
        print(json.dumps(report, indent=2, ensure_ascii=False))
        write_report(args.report_json, report)
        return 0

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import torch
    import vllm

    eval_entry = _eval_entry()
    data = select_samples(load_longbench_dataset(dataset_name, split="test"), args.max_samples)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and output_path.exists():
        output_path.unlink()
    data, completed = remaining_after_resume(data, output_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=("llama3" in args.model_name),
        trust_remote_code=args.trust_remote_code,
    )
    llm: LLM | None = None
    executor = None
    generated = 0
    sample_reports: list[dict[str, Any]] = []
    error: str | None = None
    tb: str | None = None
    started = time.time()
    vram_before = eval_entry.gpu_memory()
    vram_after_load = None
    vram_after_generate = None
    vram_after_free = None
    tq_install = None
    tq_stats = None
    freed_bytes = None

    try:
        llm = LLM(**engine_kwargs)
        vram_after_load = eval_entry.gpu_memory()
        if args.mode == "tq":
            executor = eval_entry.get_executor(llm)
            tq_install = eval_entry.install_turboquant(executor, args)

        for sample_idx, json_obj in enumerate(data):
            if executor is not None:
                reset_tq_state(executor)
            prompt = format_longbench_prompt(
                tokenizer=tokenizer,
                model_name=normalize_output_model_name(args.model_name),
                dataset=args.dataset,
                prompt_format=prompt_format,
                json_obj=json_obj,
                max_length=max_model_len,
                token_reserve=max(max_gen, args.prompt_token_reserve),
            )
            prompt_tokens = len(tokenizer.encode(prompt))
            sampling_params = SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=max_gen,
            )
            if args.dataset == "samsum":
                newline_tokens = tokenizer.encode("\n", add_special_tokens=False)
                stop_token_ids = [tokenizer.eos_token_id]
                if newline_tokens:
                    stop_token_ids.append(newline_tokens[-1])
                sampling_params = SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=max_gen,
                    stop_token_ids=stop_token_ids,
                )
            sample_start = time.perf_counter()
            outputs = llm.generate([prompt], sampling_params)
            elapsed = time.perf_counter() - sample_start
            output = outputs[0].outputs[0]
            pred = post_process(output.text, normalize_output_model_name(args.model_name))
            output_tokens = len(output.token_ids)
            with open(output_path, "a", encoding="utf-8") as f:
                json.dump(
                    {
                        "pred": pred,
                        "answers": json_obj["answers"],
                        "all_classes": json_obj["all_classes"],
                        "length": json_obj["length"],
                    },
                    f,
                    ensure_ascii=False,
                )
                f.write("\n")
            generated += 1
            sample_report: dict[str, Any] = {
                "sample_index": completed + sample_idx,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "generate_seconds": round(elapsed, 4),
            }
            if executor is not None:
                sample_report["tq_stats"] = eval_entry.get_tq_stats(executor)
            sample_reports.append(sample_report)

        vram_after_generate = eval_entry.gpu_memory()
        if executor is not None:
            tq_stats = eval_entry.get_tq_stats(executor)
            if args.free_kv_cache:
                freed_bytes = eval_entry.free_tq_kv_cache(executor)
                vram_after_free = eval_entry.gpu_memory()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        tb = traceback.format_exc(limit=12)
        print(f"[error] {error}", file=sys.stderr)
        vram_after_generate = eval_entry.gpu_memory()
    finally:
        if llm is not None:
            del llm
        gc.collect()
        if "torch" in locals() and torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = {
        "status": "ok" if error is None else "error",
        "error": error,
        "traceback": tb,
        **report_base,
        "vllm_path": getattr(vllm, "__file__", None),
        "completed_existing": completed,
        "generated_this_run": generated,
        "line_count": count_jsonl(output_path),
        "elapsed_sec": round(time.time() - started, 4),
        "tq_install": tq_install,
        "tq_stats": tq_stats,
        "freed_bytes": freed_bytes,
        "samples": sample_reports,
        "vram_before": vram_before,
        "vram_after_load": vram_after_load,
        "vram_after_generate": vram_after_generate,
        "vram_after_free": vram_after_free,
    }
    print("[report]")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    write_report(args.report_json, report)
    return 0 if error is None else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_longbench(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
