#!/bin/bash
# Submit with: sbatch -A <account> --export=ALL,PROJECT_DIR=...,DATA=...,OUTPUT=...,MODEL=... scripts/turing_generate.sh
#SBATCH -p u22
#SBATCH -n 16
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4096
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail

: "${TURING_ACCOUNT:?TURING_ACCOUNT must be supplied with --export}"
: "${PROJECT_DIR:?PROJECT_DIR must be supplied with --export}"
: "${DATA:?DATA must be supplied with --export}"
: "${OUTPUT:?OUTPUT must be supplied with --export}"
: "${MODEL:?MODEL must be supplied with --export}"
MODEL_REVISION="${MODEL_REVISION:-}"

module load u22/cuda/12.4
cd "$PROJECT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_CACHE_ROOT:-/scratch/$USER/searchqa-dpo/hf}"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" logs
ATTENTION_IMPLEMENTATION="${ATTENTION_IMPLEMENTATION:-sdpa}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-16}"
echo "<runtime component=generation attention_implementation=\"$ATTENTION_IMPLEMENTATION\" fallback_reason=\"${ATTENTION_FALLBACK_REASON:-none}\"/>"
nvidia-smi
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv -l 10 > "logs/gpu-${SLURM_JOB_ID}.csv" &
GPU_MONITOR_PID=$!
cleanup() {
  if kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then
    kill "$GPU_MONITOR_PID"
  fi
}
trap cleanup EXIT
GEN_ARGS=(--data "$DATA" --output "$OUTPUT" --model "$MODEL")
if [[ -n "$MODEL_REVISION" ]]; then
  GEN_ARGS+=(--model-revision "$MODEL_REVISION")
fi
uv run --frozen python -m text_feedback_dpo.cli generate "${GEN_ARGS[@]}" \
  --attention-implementation "$ATTENTION_IMPLEMENTATION" --batch-size "$GENERATION_BATCH_SIZE" \
  --max-new-tokens 512 --temperature 0.0 --top-p 1.0 \
  --policy-hash "${POLICY_HASH:?POLICY_HASH must be supplied with --export}"
