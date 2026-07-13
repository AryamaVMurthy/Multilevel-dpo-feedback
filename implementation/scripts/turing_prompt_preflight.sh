#!/bin/bash
# Compare direct and private two-pass thinking for the active-search cited-response protocol.
#SBATCH -p u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4096
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail

COMPONENT=turing_prompt_preflight
log_event() { local event="$1"; shift; printf 'event=%s timestamp=%s component=%s %s\n' "$event" "$(date -u +%FT%TZ)" "$COMPONENT" "$*"; }
fail() { log_event failure reason="$1" fallback_reason="${2:-none}" >&2; exit 2; }
allocated_gpu_count() { local raw="${SLURM_GPUS_ON_NODE:?SLURM_GPUS_ON_NODE is required}"; if [[ "$raw" =~ ^[0-9]+$ ]]; then printf '%s\n' "$raw"; elif [[ "$raw" =~ ^gpu:([0-9]+)$ ]]; then printf '%s\n' "${BASH_REMATCH[1]}"; else fail "unsupported SLURM_GPUS_ON_NODE format: $raw" gpu_count_parse_unsupported; fi; }

: "${PROJECT_DIR:?PROJECT_DIR must be supplied with --export}"
: "${DATA:?DATA must be a 32+ row train-dev JSONL}"
: "${MODEL:?MODEL must be supplied with --export}"
: "${MODEL_REVISION:?MODEL_REVISION must be supplied with --export}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT must be supplied with --export}"
: "${POLICY_HASH:?POLICY_HASH must be supplied with --export}"
: "${CONFIG:?CONFIG must be supplied with --export}"
: "${DATASET_SOURCE:?DATASET_SOURCE must be supplied with --export}"
: "${DATASET_REVISION:?DATASET_REVISION must be supplied with --export}"
: "${PROMPT_HASH:?PROMPT_HASH must be supplied with --export}"
: "${RETRIEVAL_HASH:?RETRIEVAL_HASH must be supplied with --export}"
: "${SOURCE_SCHEMA_HASH:?SOURCE_SCHEMA_HASH must be supplied with --export}"
: "${OPTIMIZATION_DECISION:?OPTIMIZATION_DECISION must be supplied with --export}"
: "${OPTIMIZATION_DECISION_SHA256:?OPTIMIZATION_DECISION_SHA256 must be supplied with --export}"
: "${STUDENT_THINKING_MODE:?STUDENT_THINKING_MODE must be explicit}"
: "${SCRATCHPAD_MAX_NEW_TOKENS:?SCRATCHPAD_MAX_NEW_TOKENS must be explicit}"
: "${QUERY_TEMPERATURE:?QUERY_TEMPERATURE must be explicit}"
: "${RESPONSE_TEMPERATURE:?RESPONSE_TEMPERATURE must be explicit}"
: "${TOP_P:?TOP_P must be explicit}"
: "${TOP_K:?TOP_K must be explicit}"
: "${BM25_K1:?BM25_K1 must be explicit}"
: "${BM25_B:?BM25_B must be explicit}"
: "${SLURM_NNODES:?SLURM_NNODES is required}"
[[ "$SLURM_NNODES" == "1" ]] || fail "prompt preflight requires exactly one node; got $SLURM_NNODES" multi_node_prompt_preflight
: "${SLURM_NTASKS:?SLURM_NTASKS is required}"
[[ "$SLURM_NTASKS" == "1" ]] || fail "prompt preflight requires exactly one task; got $SLURM_NTASKS" multi_task_prompt_preflight
ALLOCATED_GPU_COUNT="$(allocated_gpu_count)"
[[ "$ALLOCATED_GPU_COUNT" == "1" ]] || fail "prompt preflight requires exactly one allocated GPU; got $ALLOCATED_GPU_COUNT" prompt_preflight_gpu_count
PROBE_RUNNER="$PROJECT_DIR/scripts/turing_probe_runner.py"
DATA_SHA256="$(sha256sum "$DATA" | awk '{print $1}')"
CONFIG_SHA256="$(sha256sum "$CONFIG" | awk '{print $1}')"
COMMIT_HASH="$(git -C "$PROJECT_DIR" rev-parse HEAD)"
IFS=$'\t' read -r ATTENTION_IMPLEMENTATION QUERY_BATCH_SIZE RESPONSE_BATCH_SIZE QUERY_MAX_NEW_TOKENS RESPONSE_MAX_NEW_TOKENS FROZEN_THINKING_MODE FROZEN_SCRATCHPAD FROZEN_QUERY_TEMPERATURE FROZEN_RESPONSE_TEMPERATURE FROZEN_TOP_P FROZEN_TOP_K FROZEN_K1 FROZEN_B ATTENTION_FALLBACK_REASON VALIDATED_DECISION_SHA256 < <(
  "$PROBE_RUNNER" validate-decision --decision "$OPTIMIZATION_DECISION" --expected-sha256 "$OPTIMIZATION_DECISION_SHA256" --purpose generation --output-format generation-tsv \
    --commit-hash "$COMMIT_HASH" --config-sha256 "$CONFIG_SHA256" --model "$MODEL" --model-revision "$MODEL_REVISION" --dataset-source "$DATASET_SOURCE" --dataset-revision "$DATASET_REVISION" \
    --dataset-sha256 "$DATA_SHA256" --prompt-sha256 "$PROMPT_HASH" --retrieval-sha256 "$RETRIEVAL_HASH" --source-schema-sha256 "$SOURCE_SCHEMA_HASH" \
    --student-thinking-mode "$STUDENT_THINKING_MODE" --scratchpad-max-new-tokens "$SCRATCHPAD_MAX_NEW_TOKENS" --query-temperature "$QUERY_TEMPERATURE" \
    --response-temperature "$RESPONSE_TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K" --k1 "$BM25_K1" --b "$BM25_B"
) || fail "frozen optimization decision validation failed" optimization_decision_invalid

module load u22/cuda/12.4
cd "$PROJECT_DIR"
[[ -f pyproject.toml && -d src/text_feedback_dpo ]] || fail "PROJECT_DIR must contain pyproject.toml and src/text_feedback_dpo" invalid_project_root
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 UV_LINK_MODE=copy
export HF_HOME="${HF_CACHE_ROOT:-/scratch/$(hostname)/$USER/searchqa-dpo/hf}"
export HF_DATASETS_CACHE="$HF_HOME/datasets" HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" "$OUTPUT_ROOT" logs
nvidia-smi

for mode in direct two_pass; do
  uv run --frozen python -m text_feedback_dpo.cli generate-searchqa \
    --data "$DATA" --output "$OUTPUT_ROOT/$mode-predictions.jsonl" \
    --model "$MODEL" --model-revision "$MODEL_REVISION" --dataset-source "$DATASET_SOURCE" --dataset-revision "$DATASET_REVISION" --attention-implementation "$ATTENTION_IMPLEMENTATION" \
    --device cuda:0 --student-thinking-mode "$mode" --scratchpad-max-new-tokens "$SCRATCHPAD_MAX_NEW_TOKENS" \
    --query-batch-size "$QUERY_BATCH_SIZE" --response-batch-size "$RESPONSE_BATCH_SIZE" \
    --query-max-new-tokens "$QUERY_MAX_NEW_TOKENS" --response-max-new-tokens "$RESPONSE_MAX_NEW_TOKENS" \
    --context-budget 4096 --top-p 1.0 --top-k 8 --k1 1.2 --b 0.75 --policy-hash "$POLICY_HASH:$mode"
  uv run --frozen python -m text_feedback_dpo.cli preflight-quality \
    --data "$DATA" --predictions "$OUTPUT_ROOT/$mode-predictions.jsonl" \
    --output "$OUTPUT_ROOT/$mode-metrics.json" --samples "$OUTPUT_ROOT/$mode-samples.jsonl" \
    --split-name train-dev --sample-size 32 --seed 7 --protocol active-search
done

uv run --frozen python -m text_feedback_dpo.cli select-thinking-mode \
  --direct "$OUTPUT_ROOT/direct-metrics.json" --two-pass "$OUTPUT_ROOT/two_pass-metrics.json" \
  --output "$OUTPUT_ROOT/selected-thinking-mode.json"
