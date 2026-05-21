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

- `eval.py` is a single-prompt evaluator, not a LongBench or batch dataset harness.
- TurboQuant integration uses vLLM monkey-patching; treat benchmark numbers as experimental unless reproduced across multiple prompts.
- `--use-triton-score` only accelerates compressed-history QK score calculation; softmax and value aggregation are still outside that kernel path.
