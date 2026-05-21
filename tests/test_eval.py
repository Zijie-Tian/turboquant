"""Lightweight tests for the configurable eval entrypoint."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("turboquant_eval_entry", ROOT / "eval.py")
eval_entry = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = eval_entry
SPEC.loader.exec_module(eval_entry)


def test_read_prompt_from_argument():
    args = argparse.Namespace(prompt="hello", input_file=None, stdin=False)
    assert eval_entry.read_prompt(args) == "hello"


def test_read_prompt_requires_single_source(tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("from file", encoding="utf-8")
    args = argparse.Namespace(prompt="hello", input_file=str(prompt_file), stdin=False)
    try:
        eval_entry.read_prompt(args)
    except ValueError as exc:
        assert "exactly one" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("read_prompt accepted multiple input sources")


def test_expectation_contains_and_exact_match():
    result = eval_entry.check_expectation("answer text", "answer text", "text")
    assert result == {
        "expected_answer": "answer text",
        "exact_match": True,
        "expected_contains": "text",
        "contains": True,
        "passed": True,
    }


def test_parser_accepts_eval_configuration():
    parser = eval_entry.build_parser()
    args = parser.parse_args([
        "--prompt",
        "Explain KV cache quantization.",
        "--mode",
        "tq",
        "--max-tokens",
        "16",
        "--key-bits",
        "3",
        "--use-triton-score",
    ])
    assert args.mode == "tq"
    assert args.max_tokens == 16
    assert args.key_bits == 3
    assert args.use_triton_score is True
