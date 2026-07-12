#!/bin/bash
# CPU merge; every expected shard must carry a completion marker.
#SBATCH -p u22
#SBATCH -n 2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

CONFIG="${CONFIG:?CONFIG is required}"
DATASET_DIR="${DATASET_DIR:?DATASET_DIR is required}"
COLLECTION_DIR="${COLLECTION_DIR:?COLLECTION_DIR is required}"
EXPECTED_SHARDS="${EXPECTED_SHARDS:?EXPECTED_SHARDS is required}"
OUTPUT_PATH="${OUTPUT_PATH:?OUTPUT_PATH is required}"
PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
TURING_ACCOUNT="${TURING_ACCOUNT:?TURING_ACCOUNT is required}"
SOURCE_COMMIT="${SOURCE_COMMIT:?SOURCE_COMMIT is required}"
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
SCRATCH_DIR="/scratch/$USER/text-feedback-dpo/${SLURM_JOB_ID}"
mkdir -p "$SCRATCH_DIR"
export UV_CACHE_DIR="$RUNTIME_ROOT/uv_cache"
export UV_PROJECT_ENVIRONMENT="$RUNTIME_ROOT/project_venv"
if [[ ! -f "$UV_PROJECT_ENVIRONMENT/environment_verified.txt" ]]; then
  echo "ERROR: locked runtime verification is missing: $UV_PROJECT_ENVIRONMENT/environment_verified.txt" >&2
  exit 1
fi
export UV_LINK_MODE=hardlink
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src"
cp "$CONFIG" "$SCRATCH_DIR/config.yaml"
uv run --frozen --no-sync python -m text_feedback_dpo.cli merge-collection \
  --config "$SCRATCH_DIR/config.yaml" \
  --dataset-dir "$DATASET_DIR" \
  --collection-dir "$COLLECTION_DIR" \
  --expected-shards "$EXPECTED_SHARDS" \
  --output "$OUTPUT_PATH" \
  --source-commit "$SOURCE_COMMIT"
