#!/bin/bash
# CPU-only construction of audited preference datasets from a completed collection.
#SBATCH -p u22
#SBATCH -n 2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

COLLECTION_PATH="${COLLECTION_PATH:?COLLECTION_PATH is required}"
DATASET_PATH="${DATASET_PATH:?DATASET_PATH is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
SEED="${SEED:?SEED is required}"
PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
TURING_ACCOUNT="${TURING_ACCOUNT:?TURING_ACCOUNT is required}"
RUNTIME_ROOT="${RUNTIME_ROOT:?RUNTIME_ROOT is required}"

module load u22/cuda/12.4
export PATH="$HOME/.local/bin:$PATH"
if [[ ! -d /scratch || ! -w /scratch ]]; then
  echo "ERROR: /scratch is not writable; refusing home cache fallback" >&2
  exit 1
fi
if [[ "$RUNTIME_ROOT" != /scratch/* ]]; then
  echo "ERROR: RUNTIME_ROOT must be node-local /scratch storage: $RUNTIME_ROOT" >&2
  exit 1
fi
export UV_CACHE_DIR="$RUNTIME_ROOT/uv_cache"
export UV_PROJECT_ENVIRONMENT="$RUNTIME_ROOT/project_venv"
if [[ ! -f "$UV_PROJECT_ENVIRONMENT/environment_verified.txt" ]]; then
  echo "ERROR: locked runtime verification is missing: $UV_PROJECT_ENVIRONMENT/environment_verified.txt" >&2
  exit 1
fi
export UV_LINK_MODE=hardlink
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src"

uv run --frozen --no-sync python -m text_feedback_dpo.cli build-preferences \
  --collection "$COLLECTION_PATH" \
  --dataset "$DATASET_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --seed "$SEED"
