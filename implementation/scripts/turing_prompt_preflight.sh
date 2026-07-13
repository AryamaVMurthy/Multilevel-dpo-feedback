#!/bin/bash
# Compare direct and private two-pass thinking on train-derived development data.
#SBATCH -p u22
#SBATCH -n 16
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
nvidia-smi

for mode in direct two_pass; do
  uv run --frozen python -m text_feedback_dpo.cli generate \
    --data "$DATA" --output "$OUTPUT_ROOT/$mode-predictions.jsonl" \
    --model "$MODEL" --model-revision "$MODEL_REVISION" --attention-implementation sdpa \
    --student-thinking-mode "$mode" --scratchpad-max-new-tokens 256 --max-new-tokens 32 \
    --batch-size 8 --temperature 0.0 --top-p 1.0 --policy-hash "$POLICY_HASH:$mode"
  uv run --frozen python -m text_feedback_dpo.cli preflight-quality \
    --data "$DATA" --predictions "$OUTPUT_ROOT/$mode-predictions.jsonl" \
    --output "$OUTPUT_ROOT/$mode-metrics.json" --samples "$OUTPUT_ROOT/$mode-samples.jsonl" \
    --split-name train-dev --sample-size 32 --seed 7
done

uv run --frozen python -m text_feedback_dpo.cli select-thinking-mode \
  --direct "$OUTPUT_ROOT/direct-metrics.json" --two-pass "$OUTPUT_ROOT/two_pass-metrics.json" \
  --output "$OUTPUT_ROOT/selected-thinking-mode.json"
