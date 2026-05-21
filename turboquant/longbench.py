"""LongBench helpers shared by TurboQuant evaluation entrypoints.

The prompt/chat-template logic intentionally mirrors /home/zijie/Code/LUTAttn
so generated LongBench prompts remain comparable across repos.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "longbench_config"
SKIP_CHAT_TEMPLATE_DATASETS = {
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "lcc",
    "repobench-p",
}


def load_env_file(env_file: str | os.PathLike[str] | None = None) -> None:
    """Load repo-local .env values without overriding existing environment."""
    path = Path(env_file) if env_file is not None else REPO_ROOT / ".env"
    if not path.exists():
        return
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"\'')
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception as exc:  # pragma: no cover - best-effort compatibility helper
        print(f"Warning: Failed to load .env file {path}: {exc}")


def data_root() -> Path:
    return Path(os.environ.get("DATA_ROOT", Path.home() / "data"))


def model_root() -> Path:
    return Path(os.environ.get("MODEL_ROOT", Path.home() / "models"))


def load_json_config(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_longbench_dataset(
    dataset_name: str,
    split: str = "test",
    data_root_override: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """Load a LongBench jsonl split from local disk.

    Mirrors LUTAttn's default layout: ${DATA_ROOT:-~/data}/LongBench/data/*.jsonl.
    """
    if split != "test":
        raise ValueError(f"LongBench only provides test split, got {split!r}")

    root = Path(data_root_override) if data_root_override is not None else data_root() / "LongBench"
    jsonl_path = root / "data" / f"{dataset_name}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {jsonl_path}. "
            f"Please ensure data.zip under {root} is extracted."
        )

    records: list[dict[str, Any]] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def normalize_output_model_name(model_name: str) -> str:
    if model_name == "llama2-7b-chat-4k":
        return "llama2-7b-chat"
    if model_name == "llama2-13b-chat-4k":
        return "llama2-13b-chat"
    if model_name == "mistral-7b-instruct":
        return "mistral-7b-instruct-v0.2"
    return model_name


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._=-" else "_" for ch in value)


def build_chat(tokenizer: Any, prompt: str, model_name: str) -> str:
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "glm-4" in model_name or "glm4" in model_name:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    elif "chatglm" in model_name:
        if getattr(tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template

        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2-7b-80k" in model_name:
        prompt = f"<|im_start|> {prompt}"
    elif "llama3.2" in model_name:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "mistral" in model_name and "instruct" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt


def post_process(response: str, model_name: str) -> str:
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def should_skip_chat_template(dataset: str) -> bool:
    return dataset in SKIP_CHAT_TEMPLATE_DATASETS


def _token_ids(tokenizer: Any, prompt: str, *, add_special_tokens: bool | None = None):
    kwargs: dict[str, Any] = {"truncation": False, "return_tensors": "pt"}
    if add_special_tokens is not None:
        kwargs["add_special_tokens"] = add_special_tokens
    return tokenizer(prompt, **kwargs).input_ids[0]


def format_longbench_prompt(
    tokenizer: Any,
    model_name: str,
    dataset: str,
    prompt_format: str,
    json_obj: dict[str, Any],
    max_length: int,
    token_reserve: int = 0,
) -> str:
    prompt = prompt_format.format(**json_obj)
    tokenized_prompt = _token_ids(tokenizer, prompt)
    if "chatglm3" in model_name:
        tokenized_prompt = _token_ids(tokenizer, prompt, add_special_tokens=False)
    effective_max_length = max(1, max_length - max(0, token_reserve))
    if len(tokenized_prompt) > effective_max_length:
        half = int(effective_max_length / 2)
        prompt = tokenizer.decode(
            tokenized_prompt[:half], skip_special_tokens=True
        ) + tokenizer.decode(
            tokenized_prompt[-(effective_max_length - half) :],
            skip_special_tokens=True,
        )
    if not should_skip_chat_template(dataset):
        prompt = build_chat(tokenizer, prompt, model_name)
    final_tokens = _token_ids(tokenizer, prompt)
    if len(final_tokens) > effective_max_length:
        half = int(effective_max_length / 2)
        prompt = tokenizer.decode(
            final_tokens[:half], skip_special_tokens=True
        ) + tokenizer.decode(
            final_tokens[-(effective_max_length - half) :],
            skip_special_tokens=True,
        )
    return prompt


def select_samples(data: list[dict[str, Any]], max_samples: int) -> list[dict[str, Any]]:
    if max_samples > 0:
        return data[: min(max_samples, len(data))]
    return data


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def remaining_after_resume(
    data: list[dict[str, Any]], out_path: Path
) -> tuple[list[dict[str, Any]], int]:
    completed = count_jsonl(out_path)
    if completed <= 0:
        return data, 0
    if completed >= len(data):
        print(f"[Skip] {out_path.stem}: all {completed} samples already done.")
        return [], completed
    print(f"[Resume] {out_path.stem}: skip {completed}, remaining {len(data) - completed}")
    return data[completed:], completed


def resolve_model_path(model_name: str, model_path: str | None = None) -> tuple[str, str]:
    if model_path:
        return model_path, "cli_override"
    if os.environ.get("MODEL_PATH"):
        return os.environ["MODEL_PATH"], "env_MODEL_PATH"
    if os.environ.get("VLLM_MODEL_PATH"):
        return os.environ["VLLM_MODEL_PATH"], "env_VLLM_MODEL_PATH"
    model2path = load_json_config("model2path.json")
    if model_name in model2path:
        path = str(model2path[model_name])
        # Keep common local root override without editing copied LUTAttn config.
        default_home_root = str(Path.home() / "models")
        configured_root = str(model_root())
        if configured_root != default_home_root and path.startswith(default_home_root):
            path = configured_root + path[len(default_home_root) :]
        return path, "longbench_config/model2path.json"
    raise SystemExit(
        f"Unknown model_name {model_name!r}; pass --model-path to use an explicit checkpoint."
    )


def resolve_max_model_len(model_name: str, max_model_len: int | None) -> tuple[int, str]:
    if max_model_len is not None:
        return max_model_len, "cli_override"
    model2maxlen = load_json_config("model2maxlen.json")
    if model_name in model2maxlen:
        return int(model2maxlen[model_name]), "longbench_config/model2maxlen.json"
    raise SystemExit(
        f"Unknown model_name {model_name!r}; pass --max-model-len with --model-path."
    )


def resolve_max_gen(dataset: str, max_gen: int | None) -> tuple[int, str]:
    if max_gen is not None:
        return max_gen, "cli_override"
    dataset2maxlen = load_json_config("dataset2maxlen.json")
    if dataset not in dataset2maxlen:
        raise SystemExit(f"Unknown LongBench dataset {dataset!r}")
    return int(dataset2maxlen[dataset]), "longbench_config/dataset2maxlen.json"


def output_path_for_run(
    model_name: str,
    dataset: str,
    longbench_e: bool,
    mode: str,
    output_root: str | os.PathLike[str] | None = None,
    tag: str | None = None,
) -> Path:
    root_env = "LONGBENCH_PRED_E_ROOT" if longbench_e else "LONGBENCH_PRED_ROOT"
    default_root = "pred_e" if longbench_e else "longbench_out/pred"
    root = Path(output_root or os.environ.get(root_env, default_root))
    output_model_name = normalize_output_model_name(model_name)
    suffixes = [mode]
    if tag:
        suffixes.append(tag)
    if suffixes:
        output_model_name = f"{output_model_name}-{'-'.join(safe_name(s) for s in suffixes)}"
    return root / output_model_name / f"{dataset}.jsonl"
