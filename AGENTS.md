# TurboQuant Agent Notes

This repository implements TurboQuant KV-cache compression plus a vLLM integration. Keep changes small and verify with the lightweight checks below before reporting completion.

## Preferred evaluation entrypoint

Use `eval.py` for single-prompt inference and quick quality/performance sanity checks. Do **not** use or recreate `smoke.py`; it has been replaced by `eval.py`.

### Basic TurboQuant inference

```bash
python eval.py \
  --model /mnt/data/tzj/models/Llama-3.1-8B-Instruct \
  --mode tq \
  --prompt "Explain KV cache quantization in one sentence." \
  --max-tokens 64
```

### Baseline inference

```bash
python eval.py \
  --model /mnt/data/tzj/models/Llama-3.1-8B-Instruct \
  --mode baseline \
  --prompt "Explain KV cache quantization in one sentence." \
  --max-tokens 64
```

### Read prompt from a file and save text output

```bash
python eval.py \
  --model /mnt/data/tzj/models/Llama-3.1-8B-Instruct \
  --mode baseline \
  --input-file prompt.txt \
  --output-format text \
  --output-file answer.txt
```

### Read prompt from stdin

```bash
printf '用一句中文解释 KV cache quantization。' | python eval.py \
  --model /mnt/data/tzj/models/Llama-3.1-8B-Instruct \
  --mode tq \
  --stdin \
  --max-tokens 64
```

### Enable experimental Triton score path

Use this when testing the GQA-aware compressed-history QK score kernel:

```bash
python eval.py \
  --model /mnt/data/tzj/models/Llama-3.1-8B-Instruct \
  --mode tq \
  --prompt "Summarize TurboQuant." \
  --max-tokens 96 \
  --use-triton-score
```

### Free hooked KV cache after generation

```bash
python eval.py \
  --model /mnt/data/tzj/models/Llama-3.1-8B-Instruct \
  --mode tq \
  --prompt "Explain KV cache compression." \
  --max-tokens 64 \
  --free-kv-cache
```

### Optional answer checks

```bash
python eval.py \
  --mode baseline \
  --prompt "Answer exactly: Paris" \
  --max-tokens 8 \
  --expect-contains "Paris"
```

`eval.py` exits with code `2` when an expectation is provided but not met.

## Common eval parameters

- `--mode {baseline,tq}`: run vanilla vLLM or install TurboQuant hooks.
- `--max-tokens`: output token budget.
- `--temperature`, `--top-p`: sampling controls.
- `--max-model-len`: vLLM maximum context length.
- `--gpu-memory-utilization`: vLLM memory budget.
- `--tp` / `--tensor-parallel-size`: tensor parallel size.
- `--key-bits`, `--value-bits`, `--buffer-size`: TurboQuant settings.
- `--output-format {json,text}`: structured metrics or answer-only output.

## LongBench workflow

Use `scripts/run_longbench.sh` as the preferred LongBench entrypoint. It wraps
`eval_longbench.py`, runs one vLLM process per dataset/mode task, and supports
bash-level data parallelism across GPU groups plus vLLM tensor parallelism inside
each group. Keep generated artifacts out of git: `longbench_out/` and `reports/`
are intentionally ignored.

### Configuration and prompt-template contract

- LongBench config JSON lives under `longbench_config/`:
  - `dataset2prompt.json`
  - `dataset2maxlen.json`
  - `model2path.json`
  - `model2maxlen.json`
- LongBench formatting lives in `turboquant/longbench.py`.
- The chat template and prompt handling must stay byte-for-byte compatible with
  `/home/zijie/Code/LUTAttn`; do not "simplify" the template unless the LUTAttn
  source changes too.
- `tests/test_longbench.py` checks representative prompt branches and compares
  copied config files against the LUTAttn copy when that repository is present.

### Default full TurboQuant run

The script is configured for a full TurboQuant LongBench run by default:

```bash
bash scripts/run_longbench.sh
```

Default behavior:

- `MODEL=llama3.1-8b`
- `MODEL_PATH=$HOME/models/Llama-3.1-8B-Instruct` when present; otherwise use
  `MODEL_PATH`/`VLLM_MODEL_PATH` or the `longbench_config/model2path.json` value.
- `EVAL_MODES_CSV=tq`
- `GPU_IDS_CSV=0,1,2,3,4,5`
- `TP_SIZE=2`, so the default schedule creates three disjoint TP=2 groups:
  `0,1`, `2,3`, and `4,5`.
- `DATASETS_CSV` unset means all 21 LongBench subsets are evaluated.
- `MAX_SAMPLES=-1` means full dataset; set `MAX_SAMPLES=1` only for smoke tests.
- `MAX_MODEL_LEN=32768`, `MAX_NUM_SEQS=1`, `GPU_MEMORY_UTILIZATION=0.65`.
- `MAX_GEN` unset means use per-dataset generation lengths from
  `longbench_config/dataset2maxlen.json`.
- `FREE_KV_CACHE=1` frees hooked KV state after each dataset generation.
- `SCORE_RESULTS=1` writes `result.json` after prediction files finish.

Default outputs:

- Predictions: `longbench_out/pred/llama3.1-8b-tq-full/*.jsonl`
- Scores: `longbench_out/pred/llama3.1-8b-tq-full/result.json`
- Per-task reports: `reports/longbench/full/*.json`
- Logs: `logs/longbench/full/*.log`

`longbench_out/` and `reports/` are ignored because they are local benchmark
artifacts. `logs/` is already ignored by the general `*.log` rule.

### Smoke test before full runs

Use a tiny smoke run to validate scheduling, imports, template formatting, vLLM
startup, and TurboQuant hooks before launching all subsets:

```bash
DRY_RUN=0 \
OVERWRITE=1 \
RUN_NAME=smoke \
OUTPUT_TAG=smoke \
TP_SIZE=2 \
GPU_IDS_CSV=0,1,2,3,4,5 \
DATASETS_CSV=trec,hotpotqa,passage_count \
MAX_SAMPLES=1 \
MAX_MODEL_LEN=4096 \
MAX_GEN=4 \
SCORE_RESULTS=0 \
bash scripts/run_longbench.sh
```

Expected smoke outputs:

- Predictions: `longbench_out/pred/llama3.1-8b-tq-smoke/{trec,hotpotqa,passage_count}.jsonl`
- Reports: `reports/longbench/smoke/*.json`
- Logs: `logs/longbench/smoke/*.log`

Each smoke JSONL should have one line when `MAX_SAMPLES=1`, and each report JSON
should contain `"status": "ok"`.

### Multi-GPU and TP controls

- Prefer `GPU_IDS_CSV` for simple contiguous scheduling. The script derives
  groups by slicing `GPU_IDS_CSV` into chunks of `TP_SIZE`.
  - Example: `GPU_IDS_CSV=0,1,2,3,4,5 TP_SIZE=2` creates 3 workers.
  - Example: `GPU_IDS_CSV=0,1,2,3 TP_SIZE=4` creates 1 worker.
- Use `GPU_GROUPS_CSV` when explicit non-contiguous groups are needed. It
  overrides `GPU_IDS_CSV`.
  - Example: `GPU_GROUPS_CSV='0,2;1,3' TP_SIZE=2`.
- Groups must be disjoint and each group must contain exactly `TP_SIZE` GPUs.
- Keep `MAX_NUM_SEQS=1` for TurboQuant LongBench until per-request TurboQuant
  state isolation is implemented.

### Resume, overwrite, and dry-run

- `OVERWRITE=0` resumes by default: existing dataset JSONL files are reused and
  only missing samples are generated.
- Set `OVERWRITE=1` for a fresh rerun of selected datasets.
- Set `DRY_RUN=1` to validate grouping, environment variables, and the exact
  commands without loading models.

Example dry-run:

```bash
DRY_RUN=1 \
TP_SIZE=2 \
GPU_IDS_CSV=0,1,2,3,4,5 \
DATASETS_CSV=trec,hotpotqa \
bash scripts/run_longbench.sh
```

### Single-dataset direct entrypoint

Use `eval_longbench.py` directly for debugging one subset:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
TOKENIZERS_PARALLELISM=false \
python eval_longbench.py \
  --model-name llama3.1-8b \
  --model-path /home/zijie/models/Llama-3.1-8B-Instruct \
  --mode tq \
  --dataset hotpotqa \
  --max-samples 1 \
  --max-model-len 4096 \
  --max-gen 4 \
  --tp-size 2 \
  --gpu-memory-utilization 0.65 \
  --max-num-seqs 1 \
  --output-root longbench_out/pred \
  --output-tag debug \
  --report-json reports/longbench/debug/hotpotqa.json \
  --free-kv-cache \
  --overwrite
```

The direct entrypoint writes predictions to
`longbench_out/pred/<model-name>-<mode>-<output-tag>/<dataset>.jsonl` and writes
metadata/status to the path passed via `--report-json`.

### Scoring existing predictions

The launcher scores automatically when `SCORE_RESULTS=1`. To score an existing
prediction directory manually:

```bash
python score_longbench.py \
  --model llama3.1-8b-tq-full \
  --output-root longbench_out/pred
```

This writes `longbench_out/pred/llama3.1-8b-tq-full/result.json`.

## 32k TP memory benchmark workflow

Use this flow to measure Llama 3.1 8B 32k-context peak memory under tensor parallelism.
Measure peak memory externally with `nvidia-smi`; `eval.py` reports before/load/generate/free
snapshots, but a polling wrapper is needed for true peak memory.

### Generate or refresh a 32k prompt

```bash
mkdir -p .omx/tmp
python - <<'PY'
from pathlib import Path
from transformers import AutoTokenizer

model = "/mnt/data/tzj/models/Llama-3.1-8B-Instruct"
out = Path(".omx/tmp/llama31_prompt_32k.txt")
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
seed = (
    "TurboQuant KV cache quantization benchmark. "
    "Explain the design, tradeoffs, and implementation details in Chinese. "
)
text = seed
while len(tok.encode(text, add_special_tokens=False)) < 32000:
    text += seed
ids = tok.encode(text, add_special_tokens=False)[:32000]
out.write_text(tok.decode(ids), encoding="utf-8")
print(out, len(tok.encode(out.read_text(), add_special_tokens=False)))
PY
```

### Check that both GPUs are idle

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
```

### Run TP=2 baseline and TurboQuant

Use `CUDA_VISIBLE_DEVICES=0,1` and `--tp 2`. Keep `--max-model-len 32768`,
`--max-num-seqs 1`, and a tiny `--max-tokens` value when the goal is memory rather
than throughput.

```bash
CUDA_VISIBLE_DEVICES=0,1 \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
TOKENIZERS_PARALLELISM=false \
python eval.py \
  --model /mnt/data/tzj/models/Llama-3.1-8B-Instruct \
  --mode baseline \
  --input-file .omx/tmp/llama31_prompt_32k.txt \
  --max-tokens 16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.65 \
  --tp 2 \
  --output-file .omx/tmp/peak32_tp2_baseline_eval.json \
  --output-format json
```

```bash
CUDA_VISIBLE_DEVICES=0,1 \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
TOKENIZERS_PARALLELISM=false \
python eval.py \
  --model /mnt/data/tzj/models/Llama-3.1-8B-Instruct \
  --mode tq \
  --input-file .omx/tmp/llama31_prompt_32k.txt \
  --max-tokens 16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.65 \
  --tp 2 \
  --free-kv-cache \
  --output-file .omx/tmp/peak32_tp2_tq_eval.json \
  --output-format json
```

### Interpretation notes

- Tensor parallelism does reduce per-GPU model-weight memory.
- vLLM peak memory may not drop with TP when `--gpu-memory-utilization` is high,
  because vLLM uses the freed memory to preallocate a larger KV-cache block pool.
- To expose the weight-memory reduction, repeat the run with a lower budget such as
  `--gpu-memory-utilization 0.40`.
- For TurboQuant, compare both `vram_after_generate` and `vram_after_free`; the
  peak can include coexistence of compressed KV and vLLM's original KV cache, while
  `--free-kv-cache` shows the post-release steady state.
- In the 32k TP=2 A100 run from this workspace, `--gpu-memory-utilization 0.65`
  peaked near `26.8 GiB/GPU` for baseline and `27.4 GiB/GPU` for TurboQuant, while
  TurboQuant dropped to about `9.7 GiB/GPU` after freeing the original vLLM KV cache.

## Lightweight verification

Run these after edits that affect eval, hooks, quantization, or kernels:

```bash
python eval.py --help
python -m pytest -q
PYTHONPYCACHEPREFIX=/tmp/tq_pycache python -m compileall -q eval.py benchmark.py proof.py tests turboquant
```

For a real inference sanity check, run a short prompt on the local Llama model:

```bash
CUDA_VISIBLE_DEVICES=0 \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
TOKENIZERS_PARALLELISM=false \
python eval.py \
  --model /mnt/data/tzj/models/Llama-3.1-8B-Instruct \
  --mode tq \
  --prompt "用一句中文解释 KV cache quantization。" \
  --max-tokens 24 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.60
```

## Current limitations

- `eval.py` remains a single-prompt evaluator; use `eval_longbench.py` or
  `scripts/run_longbench.sh` for LongBench.
- TurboQuant integration uses vLLM monkey-patching; treat benchmark numbers as experimental unless reproduced across multiple prompts.
- `--use-triton-score` only accelerates compressed-history QK score calculation; softmax and value aggregation are still outside that kernel path.
