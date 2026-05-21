#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

usage() {
    cat <<'EOF'
Usage: [ENV=VALUE ...] bash scripts/run_longbench.sh

Run LongBench tasks with bash-level data parallelism and vLLM-internal tensor
parallelism. Each task calls eval_longbench.py and writes one JSONL output per
dataset plus a report JSON.

Common environment variables:
  MODEL=llama3.1-8b
  MODEL_PATH=/home/zijie/models/Llama-3.1-8B-Instruct
  EVAL_MODES_CSV=tq                 # tq or baseline; comma-separated
  GPU_IDS_CSV=0,1,2,3,4,5           # derive groups by TP_SIZE
  GPU_GROUPS_CSV='0,1;2,3;4,5'      # explicit groups override GPU_IDS_CSV
  TP_SIZE=2
  DATASETS_CSV=<unset>              # unset => all 21 LongBench datasets
  MAX_SAMPLES=-1                    # full dataset; use 1 only for smoke
  MAX_MODEL_LEN=32768
  MAX_GEN=<unset>                   # use LongBench dataset2maxlen defaults
  GPU_MEMORY_UTILIZATION=0.65
  RUN_NAME=full                     # separates logs/reports and output tag
  OUTPUT_ROOT=longbench_out/pred
  OUTPUT_TAG=full                   # output dir: llama3.1-8b-tq-full
  LOG_DIR=logs/longbench/full
  REPORT_DIR=reports/longbench/full
  OVERWRITE=0                       # resume by default; set 1 for fresh rerun
  DRY_RUN=0                         # set 1 to print commands without loading models
  SCORE_RESULTS=1                   # write result.json after all subsets finish
  FREE_KV_CACHE=1

Full run using all six GPUs as three TP=2 groups:
  bash scripts/run_longbench.sh

Smoke example:
  DRY_RUN=0 OVERWRITE=1 RUN_NAME=smoke OUTPUT_TAG=smoke TP_SIZE=2 GPU_IDS_CSV=0,1,2,3,4,5 \
  DATASETS_CSV=trec,hotpotqa,passage_count MAX_SAMPLES=1 MAX_MODEL_LEN=4096 MAX_GEN=4 SCORE_RESULTS=0 \
  bash scripts/run_longbench.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

error() {
    echo "ERROR: $*" >&2
    exit 2
}

warn() {
    echo "WARN: $*" >&2
}

is_truthy() {
    case "${1:-}" in
        1|true|TRUE|yes|YES|y|Y|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

split_csv() {
    local csv="$1"
    local -n out_ref="$2"
    out_ref=()
    local -a raw=()
    IFS=',' read -r -a raw <<< "$csv"
    local item
    for item in "${raw[@]}"; do
        item="$(trim "$item")"
        if [[ -n "$item" ]]; then
            out_ref+=("$item")
        fi
    done
}

join_csv() {
    local IFS=','
    printf '%s' "$*"
}

shell_quote_command() {
    local part
    for part in "$@"; do
        printf '%q ' "$part"
    done
}

safe_name() {
    local value="$1"
    value="${value//\//_}"
    value="${value// /_}"
    value="${value//,/+}"
    value="${value//;/+}"
    printf '%s' "$value"
}

require_positive_int() {
    local name="$1"
    local value="$2"
    if ! [[ "$value" =~ ^[0-9]+$ ]] || (( value < 1 )); then
        error "$name must be a positive integer; got '$value'"
    fi
}

require_all_or_positive_int() {
    local name="$1"
    local value="$2"
    if [[ "$value" == "-1" ]]; then
        return 0
    fi
    require_positive_int "$name" "$value"
}

require_float() {
    local name="$1"
    local value="$2"
    if ! [[ "$value" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        error "$name must be a non-negative number; got '$value'"
    fi
}

load_dotenv() {
    local env_file="${ENV_FILE:-${REPO_ROOT}/.env}"
    if [[ ! -f "$env_file" ]]; then
        return 0
    fi
    local line key value
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line//$'\r'/}"
        line="$(trim "$line")"
        if [[ -z "$line" || "${line:0:1}" == "#" ]]; then
            continue
        fi
        if [[ "$line" == export[[:space:]]* ]]; then
            line="$(trim "${line#export}")"
        fi
        if [[ "$line" != *=* ]]; then
            continue
        fi
        key="$(trim "${line%%=*}")"
        value="$(trim "${line#*=}")"
        if ! [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
            continue
        fi
        if [[ ( "$value" == \"*\" && "$value" == *\" ) || ( "$value" == \'*\' && "$value" == *\' ) ]]; then
            value="${value:1:${#value}-2}"
        fi
        if [[ -z "${!key+x}" ]]; then
            export "$key=$value"
        fi
    done < "$env_file"
}

load_dotenv

MODEL="${MODEL:-llama3.1-8b}"
MODEL_PATH="${MODEL_PATH:-${VLLM_MODEL_PATH:-}}"
if [[ -z "$MODEL_PATH" && "$MODEL" == "llama3.1-8b" && -d "${HOME}/models/Llama-3.1-8B-Instruct" ]]; then
    MODEL_PATH="${HOME}/models/Llama-3.1-8B-Instruct"
fi
EVAL_SCRIPT="${EVAL_SCRIPT:-eval_longbench.py}"
EVAL_MODES_CSV="${EVAL_MODES_CSV:-tq}"
TP_SIZE="${TP_SIZE:-2}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1,2,3,4,5}"
GPU_GROUPS_CSV="${GPU_GROUPS_CSV:-}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_GEN="${MAX_GEN:-}"
PROMPT_TOKEN_RESERVE="${PROMPT_TOKEN_RESERVE:-256}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
RUN_NAME="${RUN_NAME:-all}"
LOG_DIR="${LOG_DIR:-logs/longbench/${RUN_NAME}}"
REPORT_DIR="${REPORT_DIR:-reports/longbench/${RUN_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-longbench_out/pred}"
OUTPUT_TAG="${OUTPUT_TAG:-${RUN_NAME}}"
DRY_RUN="${DRY_RUN:-0}"
OVERWRITE="${OVERWRITE:-0}"
SCORE_RESULTS="${SCORE_RESULTS:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
FREE_KV_CACHE="${FREE_KV_CACHE:-1}"
ENABLE_TRIATTENTION="${ENABLE_TRIATTENTION:-0}"
VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}"

require_positive_int TP_SIZE "$TP_SIZE"
require_all_or_positive_int MAX_SAMPLES "$MAX_SAMPLES"
require_positive_int MAX_MODEL_LEN "$MAX_MODEL_LEN"
require_positive_int MAX_NUM_SEQS "$MAX_NUM_SEQS"
require_float GPU_MEMORY_UTILIZATION "$GPU_MEMORY_UTILIZATION"
if [[ -n "$MAX_GEN" ]]; then
    require_positive_int MAX_GEN "$MAX_GEN"
fi
if [[ -n "$MAX_NUM_BATCHED_TOKENS" ]]; then
    require_positive_int MAX_NUM_BATCHED_TOKENS "$MAX_NUM_BATCHED_TOKENS"
fi

if [[ ! -f "$EVAL_SCRIPT" ]]; then
    error "EVAL_SCRIPT '$EVAL_SCRIPT' not found"
fi

split_csv "$EVAL_MODES_CSV" EVAL_MODES
if (( ${#EVAL_MODES[@]} == 0 )); then
    error "At least one eval mode is required"
fi
for mode in "${EVAL_MODES[@]}"; do
    case "${mode,,}" in
        tq|baseline) ;;
        *) error "EVAL_MODES_CSV supports only tq or baseline; got '$mode'" ;;
    esac
done

if [[ -n "${DATASETS_CSV:-}" ]]; then
    split_csv "$DATASETS_CSV" DATASETS
else
    DATASETS=(
        "narrativeqa"
        "qasper"
        "multifieldqa_en"
        "multifieldqa_zh"
        "hotpotqa"
        "2wikimqa"
        "musique"
        "dureader"
        "gov_report"
        "qmsum"
        "multi_news"
        "vcsum"
        "trec"
        "triviaqa"
        "samsum"
        "lsht"
        "passage_count"
        "passage_retrieval_en"
        "passage_retrieval_zh"
        "lcc"
        "repobench-p"
    )
fi
if (( ${#DATASETS[@]} == 0 )); then
    error "At least one dataset is required"
fi

validate_disjoint_groups() {
    declare -A seen=()
    local group_idx gpu
    for group_idx in "${!GPU_GROUPS[@]}"; do
        split_csv "${GPU_GROUPS[$group_idx]}" group_items
        if (( ${#group_items[@]} != TP_SIZE )); then
            error "GPU group ${GPU_GROUPS[$group_idx]} has ${#group_items[@]} GPUs but TP_SIZE=$TP_SIZE"
        fi
        for gpu in "${group_items[@]}"; do
            if [[ -n "${seen[$gpu]:-}" ]]; then
                error "GPU group overlap: GPU '$gpu' appears in both group ${seen[$gpu]} and group $group_idx"
            fi
            seen[$gpu]="$group_idx"
        done
    done
}

declare -a GPU_GROUPS=()
if [[ -n "$GPU_GROUPS_CSV" ]]; then
    IFS=';' read -r -a raw_groups <<< "$GPU_GROUPS_CSV"
    for raw_group in "${raw_groups[@]}"; do
        raw_group="$(trim "$raw_group")"
        if [[ -n "$raw_group" ]]; then
            split_csv "$raw_group" group_items
            GPU_GROUPS+=("$(join_csv "${group_items[@]}")")
        fi
    done
    if (( ${#GPU_GROUPS[@]} == 0 )); then
        error "GPU_GROUPS_CSV was set but no valid groups were parsed"
    fi
else
    split_csv "$GPU_IDS_CSV" GPU_IDS
    if (( ${#GPU_IDS[@]} < TP_SIZE )); then
        error "Need at least TP_SIZE=$TP_SIZE GPUs in GPU_IDS_CSV; got ${#GPU_IDS[@]} (${GPU_IDS_CSV})"
    fi
    DERIVED_GROUPS=$(( ${#GPU_IDS[@]} / TP_SIZE ))
    UNUSED_GPUS=$(( ${#GPU_IDS[@]} % TP_SIZE ))
    if (( DERIVED_GROUPS < 1 )); then
        error "Unable to derive any DP group from GPU_IDS_CSV='${GPU_IDS_CSV}' and TP_SIZE=$TP_SIZE"
    fi
    if (( UNUSED_GPUS > 0 )); then
        warn "Ignoring ${UNUSED_GPUS} trailing GPU(s) because ${#GPU_IDS[@]} is not divisible by TP_SIZE=$TP_SIZE"
    fi
    for ((group_idx = 0; group_idx < DERIVED_GROUPS; group_idx++)); do
        group_items=()
        for ((offset = 0; offset < TP_SIZE; offset++)); do
            group_items+=("${GPU_IDS[$(( group_idx * TP_SIZE + offset ))]}")
        done
        GPU_GROUPS+=("$(join_csv "${group_items[@]}")")
    done
fi
validate_disjoint_groups

DP_SIZE=${#GPU_GROUPS[@]}
TOTAL_TASKS=$(( ${#EVAL_MODES[@]} * ${#DATASETS[@]} ))
mkdir -p "$LOG_DIR" "$REPORT_DIR"

build_eval_command() {
    local mode="$1"
    local dataset="$2"
    local report_path="$3"
    local -n cmd_ref="$4"
    cmd_ref=(
        python "$EVAL_SCRIPT"
        --model-name "$MODEL"
        --mode "${mode,,}"
        --dataset "$dataset"
        --max-samples "$MAX_SAMPLES"
        --max-model-len "$MAX_MODEL_LEN"
        --prompt-token-reserve "$PROMPT_TOKEN_RESERVE"
        --tp-size "$TP_SIZE"
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
        --max-num-seqs "$MAX_NUM_SEQS"
        --output-root "$OUTPUT_ROOT"
        --report-json "$report_path"
    )
    if [[ -n "$OUTPUT_TAG" ]]; then
        cmd_ref+=(--output-tag "$OUTPUT_TAG")
    fi
    if [[ -n "$MODEL_PATH" ]]; then
        cmd_ref+=(--model-path "$MODEL_PATH")
    fi
    if [[ -n "$MAX_GEN" ]]; then
        cmd_ref+=(--max-gen "$MAX_GEN")
    fi
    if [[ -n "$MAX_NUM_BATCHED_TOKENS" ]]; then
        cmd_ref+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
    fi
    if is_truthy "$TRUST_REMOTE_CODE"; then
        cmd_ref+=(--trust-remote-code)
    else
        cmd_ref+=(--no-trust-remote-code)
    fi
    if is_truthy "$OVERWRITE"; then
        cmd_ref+=(--overwrite)
    fi
    if is_truthy "$FREE_KV_CACHE"; then
        cmd_ref+=(--free-kv-cache)
    fi
    if is_truthy "$ENABLE_TRIATTENTION"; then
        cmd_ref+=(--enable-triattention)
    fi
    if [[ -n "$EXTRA_EVAL_ARGS" ]]; then
        read -r -a extra_args <<< "$EXTRA_EVAL_ARGS"
        cmd_ref+=("${extra_args[@]}")
    fi
}

print_config() {
    echo "[config] MODEL=$MODEL"
    echo "[config] MODEL_PATH=${MODEL_PATH:-<config-default>}"
    echo "[config] EVAL_SCRIPT=$EVAL_SCRIPT"
    echo "[config] EVAL_MODES=${EVAL_MODES[*]}"
    echo "[config] DATASETS=${DATASETS[*]}"
    echo "[config] TP_SIZE=$TP_SIZE"
    echo "[config] DP_SIZE=$DP_SIZE"
    echo "[config] MAX_SAMPLES=$MAX_SAMPLES"
    echo "[config] MAX_MODEL_LEN=$MAX_MODEL_LEN"
    echo "[config] MAX_GEN=${MAX_GEN:-<dataset-default>}"
    echo "[config] RUN_NAME=$RUN_NAME"
    echo "[config] OUTPUT_ROOT=$OUTPUT_ROOT"
    echo "[config] OUTPUT_TAG=${OUTPUT_TAG:-<none>}"
    echo "[config] LOG_DIR=$LOG_DIR"
    echo "[config] REPORT_DIR=$REPORT_DIR"
    echo "[config] DRY_RUN=$DRY_RUN OVERWRITE=$OVERWRITE SCORE_RESULTS=$SCORE_RESULTS"
    local idx
    for idx in "${!GPU_GROUPS[@]}"; do
        echo "[config] GROUP[$idx]=${GPU_GROUPS[$idx]}"
    done
}

run_one_task() {
    local group_idx="$1"
    local group="$2"
    local task_idx="$3"
    local mode="$4"
    local dataset="$5"
    local label="$(safe_name "${MODEL}_${mode}${OUTPUT_TAG:+_${OUTPUT_TAG}}_${dataset}_tp${TP_SIZE}_g${group_idx}")"
    local log_path="${LOG_DIR}/${label}.log"
    local report_path="${REPORT_DIR}/${label}.json"
    local -a cmd=()
    build_eval_command "$mode" "$dataset" "$report_path" cmd

    echo "[launch ${task_idx}/${TOTAL_TASKS}] group=${group_idx} CUDA_VISIBLE_DEVICES=${group} mode=${mode} dataset=${dataset} log=${log_path} report=${report_path}"
    if is_truthy "$DRY_RUN"; then
        printf '[dry-run] CUDA_VISIBLE_DEVICES=%q ' "$group"
        shell_quote_command "${cmd[@]}"
        printf '\n'
        return 0
    fi

    CUDA_VISIBLE_DEVICES="$group" \
    ENABLE_TRIATTENTION="$ENABLE_TRIATTENTION" \
    VLLM_ENABLE_V1_MULTIPROCESSING="$VLLM_ENABLE_V1_MULTIPROCESSING" \
    TOKENIZERS_PARALLELISM="$TOKENIZERS_PARALLELISM" \
    "${cmd[@]}" > "$log_path" 2>&1
}

run_group_worker() {
    local group_idx="$1"
    local group="${GPU_GROUPS[$group_idx]}"
    local task_idx=0
    local mode dataset
    for mode in "${EVAL_MODES[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            task_idx=$(( task_idx + 1 ))
            if (( (task_idx - 1) % DP_SIZE == group_idx )); then
                run_one_task "$group_idx" "$group" "$task_idx" "$mode" "$dataset"
            fi
        done
    done
}

cleanup() {
    echo "[cleanup] terminating worker processes" >&2
    for pid in "${WORKER_PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup INT TERM

print_config

if is_truthy "$DRY_RUN"; then
    echo "[dry-run] validated scheduler plan; no tasks will be executed"
    for group_idx in "${!GPU_GROUPS[@]}"; do
        run_group_worker "$group_idx"
    done
    exit 0
fi

echo "[run] starting ${DP_SIZE} DP worker(s) for ${TOTAL_TASKS} task(s)"
declare -a WORKER_PIDS=()
for group_idx in "${!GPU_GROUPS[@]}"; do
    run_group_worker "$group_idx" &
    WORKER_PIDS+=("$!")
done

failed=0
for pid in "${WORKER_PIDS[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done

if (( failed != 0 )); then
    echo "[run] one or more LongBench tasks failed; inspect ${LOG_DIR}" >&2
    exit 1
fi

echo "[run] all LongBench tasks completed"

if is_truthy "$SCORE_RESULTS"; then
    echo "[score] scoring generated LongBench outputs"
    shopt -s nullglob
    SCORE_GLOB="${SCORE_GLOB:-${OUTPUT_ROOT}/${MODEL}*${OUTPUT_TAG:+-${OUTPUT_TAG}}}"
    # shellcheck disable=SC2206
    score_dirs=( $SCORE_GLOB )
    for dir in "${score_dirs[@]}"; do
        model_dir="$(basename "$dir")"
        score_log="${LOG_DIR}/score_$(safe_name "$model_dir").log"
        echo "[score] ${model_dir} -> ${score_log}"
        python score_longbench.py --model "$model_dir" --output-root "$OUTPUT_ROOT" > "$score_log" 2>&1
    done
fi
