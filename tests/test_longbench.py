"""Tests for LongBench prompt parity, scheduler dry-run, and TQ reset."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

from turboquant.longbench import (
    SKIP_CHAT_TEMPLATE_DATASETS,
    build_chat,
    format_longbench_prompt,
    load_json_config,
    should_skip_chat_template,
)

ROOT = Path(__file__).resolve().parents[1]
LUTATTN = Path("/home/zijie/Code/LUTAttn")


class FakeTokenizer:
    chat_template = None
    eos_token_id = 2

    def __call__(self, prompt, **kwargs):
        ids = list(range(len(prompt.split())))

        class Batch:
            input_ids = [ids]

        return Batch()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"tok{i}" for i, _ in enumerate(ids))

    def encode(self, prompt, add_special_tokens=True):
        return list(range(len(prompt.split())))

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        suffix = "<GEN>" if add_generation_prompt else ""
        return f"<CHAT>{messages[0]['content']}{suffix}"

    def build_prompt(self, prompt):
        return f"<BUILT>{prompt}"

    def build_chat_input(self, prompt):
        return f"<CHATGLM3>{prompt}"


def test_copied_longbench_configs_match_lutattn():
    for name in [
        "dataset2prompt.json",
        "dataset2maxlen.json",
        "model2path.json",
        "model2maxlen.json",
    ]:
        ours = (ROOT / "longbench_config" / name).read_text(encoding="utf-8")
        theirs = (LUTATTN / "longbench_config" / name).read_text(encoding="utf-8")
        assert ours == theirs


def test_dataset_prompt_config_loads():
    prompts = load_json_config("dataset2prompt.json")
    assert "trec" in prompts
    assert "{context}" in prompts["hotpotqa"]


def test_build_chat_matches_lutattn_representative_branches():
    tok = FakeTokenizer()
    assert build_chat(tok, "hello", "llama2-7b") == "[INST]hello[/INST]"
    assert build_chat(tok, "hello", "mistral-7b-instruct") == "[INST]hello[/INST]"
    assert build_chat(tok, "hello", "llama2-7b-80k") == "<|im_start|> hello"
    assert build_chat(tok, "hello", "llama3.2-1b") == "<CHAT>hello<GEN>"
    assert build_chat(tok, "hello", "xgen-7b-8k").endswith(" ### Human: hello\n###")
    assert build_chat(tok, "hello", "internlm-7b-8k") == "<|User|>:hello<eoh>\n<|Bot|>:"


def test_chat_template_skip_list_is_lutattn_exact():
    expected = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
    assert SKIP_CHAT_TEMPLATE_DATASETS == expected
    for dataset in expected:
        assert should_skip_chat_template(dataset)
    assert not should_skip_chat_template("hotpotqa")


def test_format_prompt_middle_truncates_with_reserve():
    tok = FakeTokenizer()
    json_obj = {"context": " ".join(f"c{i}" for i in range(20)), "input": "question"}
    prompt = format_longbench_prompt(
        tokenizer=tok,
        model_name="llama3.1-8b",
        dataset="trec",
        prompt_format="{context} {input}",
        json_obj=json_obj,
        max_length=10,
        token_reserve=2,
    )
    assert len(tok.encode(prompt)) <= 8


def test_eval_longbench_dry_run_does_not_load_model(tmp_path):
    report = tmp_path / "report.json"
    proc = subprocess.run(
        [
            sys.executable,
            "eval_longbench.py",
            "--model-name",
            "llama3.1-8b",
            "--model-path",
            "/home/zijie/models/Llama-3.1-8B-Instruct",
            "--mode",
            "tq",
            "--dataset",
            "trec",
            "--max-samples",
            "1",
            "--max-model-len",
            "4096",
            "--max-gen",
            "4",
            "--tp-size",
            "2",
            "--report-json",
            str(report),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert '"status": "dry_run"' in proc.stdout
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "dry_run"
    assert payload["tp_size"] == 2


def test_run_longbench_dry_run_groups_by_tp_size(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "TP_SIZE": "2",
            "GPU_IDS_CSV": "0,1,2,3,4,5",
            "DATASETS_CSV": "trec,hotpotqa,passage_count",
            "MAX_SAMPLES": "1",
            "MAX_MODEL_LEN": "4096",
            "MAX_GEN": "4",
            "LOG_DIR": str(tmp_path / "logs"),
            "REPORT_DIR": str(tmp_path / "reports"),
            "OUTPUT_ROOT": str(tmp_path / "pred"),
        }
    )
    proc = subprocess.run(
        ["bash", "scripts/run_longbench.sh"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "GROUP[0]=0,1" in proc.stdout
    assert "GROUP[1]=2,3" in proc.stdout
    assert "GROUP[2]=4,5" in proc.stdout
    assert proc.stdout.count("--tp-size 2") == 3


def test_reset_tq_state_resets_all_worker_layers(monkeypatch):
    eval_longbench = importlib.import_module("eval_longbench")
    eval_mod = importlib.import_module("eval")

    class State:
        def __init__(self):
            self.count = 0

        def reset(self):
            self.count += 1

    class Runner:
        def __init__(self):
            self._tq_layer_states = {"a": State(), "b": State()}

    class Worker:
        def __init__(self):
            self.model_runner = Runner()

    workers = [Worker(), Worker()]

    def fake_collective_rpc(executor, fn):
        return [fn(worker) for worker in executor]

    monkeypatch.setattr(eval_mod, "collective_rpc", fake_collective_rpc)
    result = eval_longbench.reset_tq_state(workers)
    assert result == [
        {"num_layers": 2, "reset": True},
        {"num_layers": 2, "reset": True},
    ]
    assert [state.count for worker in workers for state in worker.model_runner._tq_layer_states.values()] == [1, 1, 1, 1]
