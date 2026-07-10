#!/bin/bash
# CPU-only manifest materialization; dataset archives must already be present on the node.
#SBATCH -p u22
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

CONFIG="${CONFIG:?CONFIG is required}"
SOURCE_PATH="${SOURCE_PATH:?SOURCE_PATH is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
TURING_ACCOUNT="${TURING_ACCOUNT:?TURING_ACCOUNT is required}"

module load u22/cuda/12.4
export PATH="$HOME/.local/bin:$PATH"
if [[ ! -d /scratch || ! -w /scratch ]]; then
  echo "ERROR: /scratch is not writable; refusing to place caches in home" >&2
  exit 1
fi
SCRATCH_DIR="/scratch/$USER/text-feedback-dpo/${SLURM_JOB_ID}"
mkdir -p "$SCRATCH_DIR" "$OUTPUT_DIR"
export UV_CACHE_DIR="$SCRATCH_DIR/uv_cache"
export UV_PROJECT_ENVIRONMENT="$SCRATCH_DIR/project_venv"
export HF_HOME="$SCRATCH_DIR/hf_cache"
export HF_DATASETS_CACHE="$SCRATCH_DIR/hf_datasets"
export TRANSFORMERS_CACHE="$SCRATCH_DIR/hf_cache"
export UV_CONCURRENT_DOWNLOADS=1
export UV_CONCURRENT_BUILDS=1
export UV_CONCURRENT_INSTALLS=1
export UV_LINK_MODE=copy
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src"

echo "job_id=${SLURM_JOB_ID} account=${TURING_ACCOUNT} host=$(hostname)"
echo "scratch_before=$(df -h /scratch | tail -1)"
cp "$CONFIG" "$SCRATCH_DIR/config.yaml"
uv run --no-project --with 'pyyaml==6.0.3' --with 'zstandard==0.25.0' \
  python -m text_feedback_dpo.cli materialize-dataset \
  --config "$SCRATCH_DIR/config.yaml" \
  --source-path "$SOURCE_PATH" \
  --output-dir "$OUTPUT_DIR"
echo "scratch_after=$(df -h /scratch | tail -1)"
