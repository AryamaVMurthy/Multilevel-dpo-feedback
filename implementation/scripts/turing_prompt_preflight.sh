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

: "${PROJECT_DIR:?PROJECT_DIR must be supplied with --export}"
: "${DATA:?DATA must be a 32+ row train-dev JSONL}"
: "${MODEL:?MODEL must be supplied with --export}"
: "${MODEL_REVISION:?MODEL_REVISION must be supplied with --export}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT must be supplied with --export}"
: "${POLICY_HASH:?POLICY_HASH must be supplied with --export}"

module load u22/cuda/12.4
cd "$PROJECT_DIR"
[[ -f pyproject.toml && -d src/text_feedback_dpo ]] || { echo "ERROR: PROJECT_DIR must contain pyproject.toml and src/text_feedback_dpo" >&2; exit 2; }
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 UV_LINK_MODE=copy
export HF_HOME="${HF_CACHE_ROOT:-/scratch/$(hostname)/$USER/searchqa-dpo/hf}"
export HF_DATASETS_CACHE="$HF_HOME/datasets" HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" "$OUTPUT_ROOT" logs
SCRATCHPAD_MAX_NEW_TOKENS="${SCRATCHPAD_MAX_NEW_TOKENS:-128}"
QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-4}"
RESPONSE_BATCH_SIZE="${RESPONSE_BATCH_SIZE:-4}"
QUERY_MAX_NEW_TOKENS="${QUERY_MAX_NEW_TOKENS:-32}"
RESPONSE_MAX_NEW_TOKENS="${RESPONSE_MAX_NEW_TOKENS:-256}"
nvidia-smi

for mode in direct two_pass; do
  uv run --frozen python -m text_feedback_dpo.cli generate-searchqa \
    --data "$DATA" --output "$OUTPUT_ROOT/$mode-predictions.jsonl" \
    --model "$MODEL" --model-revision "$MODEL_REVISION" --attention-implementation sdpa \
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
