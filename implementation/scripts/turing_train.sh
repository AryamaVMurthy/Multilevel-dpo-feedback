#!/bin/bash
# Submit with: sbatch -A <account> --export=ALL,PROJECT_DIR=...,METHOD=dpo,... scripts/turing_train.sh
#SBATCH -p u22
#SBATCH -n 64
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=4096
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail

: "${TURING_ACCOUNT:?TURING_ACCOUNT must be supplied with --export}"
: "${PROJECT_DIR:?PROJECT_DIR must be supplied with --export}"
: "${METHOD:?METHOD must be sft, dpo, grpo, or dapo}"
: "${CONFIG:?CONFIG must be supplied with --export}"
: "${TRAIN:?TRAIN must be supplied with --export}"
: "${OUTPUT:?OUTPUT must be supplied with --export}"
: "${TRAIN_GPUS:?TRAIN_GPUS must be supplied with --export}"
if [[ "$TRAIN_GPUS" != "2" ]]; then
  echo "ERROR: this script requests exactly 2 GPUs; TRAIN_GPUS=$TRAIN_GPUS is unsupported" >&2
  exit 2
fi

case "$METHOD" in
  sft|dpo) : "${EVAL:?EVAL must be supplied for $METHOD}" ;;
  grpo|dapo) : ;;
  *) echo "ERROR: unsupported METHOD=$METHOD" >&2; exit 2 ;;
esac

module load u22/cuda/12.4
cd "$PROJECT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_CACHE_ROOT:-/scratch/$USER/searchqa-dpo/hf}"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" logs
if [[ "${CLEANUP_TRAIN_INPUTS:-false}" == "true" ]]; then
  export TFDPO_CLEANUP_TRAINING_INPUTS=1
  echo "<storage cleanup_training_inputs=enabled reason=derived_sft_inputs_are_rebuildable_from_raw_searchqa/>"
fi
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

ARGS=(--config "$CONFIG" --train "$TRAIN" --output "$OUTPUT" --deepspeed-config configs/deepspeed_zero3.json --save-steps 100 --eval-steps 100)
if [[ "$METHOD" == sft || "$METHOD" == dpo ]]; then
  ARGS+=(--eval "$EVAL")
fi
if [[ -n "${START_MODEL:-}" ]]; then
  ARGS+=(--model "$START_MODEL")
  if [[ -n "${START_REVISION:-}" ]]; then
    ARGS+=(--model-revision "$START_REVISION")
  fi
fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  ARGS+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
fi

torchrun --standalone --nproc_per_node="$TRAIN_GPUS" -m text_feedback_dpo.cli "train-$METHOD" "${ARGS[@]}"
