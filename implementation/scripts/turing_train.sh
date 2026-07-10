#!/bin/bash
#SBATCH -p u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4096
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

TRAIN_METHOD="${TRAIN_METHOD:?TRAIN_METHOD is required}"
DATA_PATH="${DATA_PATH:?DATA_PATH is required}"
MODEL_ID="${MODEL_ID:?MODEL_ID is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
TURING_ACCOUNT="${TURING_ACCOUNT:?TURING_ACCOUNT is required}"
HF_CACHE_DIR="${HF_CACHE_DIR:?HF_CACHE_DIR is required}"

module load u22/cuda/12.4
export PATH="$HOME/.local/bin:$PATH"

allocation_mismatch() {
  echo "ERROR: allocation_mismatch: $*" >&2
  exit 1
}
[[ "${SLURM_JOB_NUM_NODES:-}" == "1" ]] || allocation_mismatch "expected one node"
[[ "${SLURM_NNODES:-}" == "1" ]] || allocation_mismatch "expected one node"
if [[ -n "${SLURM_GPUS_ON_NODE:-}" && "${SLURM_GPUS_ON_NODE}" != "1" ]]; then
  allocation_mismatch "expected one GPU on node"
fi
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || allocation_mismatch "CUDA_VISIBLE_DEVICES is unset"

SCRATCH_DIR="/scratch/$USER/text-feedback-dpo/${SLURM_JOB_ID}"
mkdir -p "$SCRATCH_DIR" "/scratch/$USER/text-feedback-dpo/uv_cache" "$HF_CACHE_DIR"
export UV_CACHE_DIR="/scratch/$USER/text-feedback-dpo/uv_cache"
export UV_PROJECT_ENVIRONMENT="$SCRATCH_DIR/project_venv"
export UV_CONCURRENT_DOWNLOADS=1
export UV_CONCURRENT_BUILDS=1
export UV_CONCURRENT_INSTALLS=1
export UV_LINK_MODE=copy
export HF_HOME="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR"
export HF_DATASETS_CACHE="$HF_CACHE_DIR/datasets"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"

echo "job_id=${SLURM_JOB_ID}"
echo "method=$TRAIN_METHOD"
echo "data=$DATA_PATH"
echo "model_id=$MODEL_ID"
echo "output_dir=$OUTPUT_DIR"
nvidia-smi

uv run --frozen python -m text_feedback_dpo.cli train --method "$TRAIN_METHOD" \
  --data "$DATA_PATH" \
  --model-id "$MODEL_ID" \
  --output-dir "$OUTPUT_DIR" \
  --max-steps 1
