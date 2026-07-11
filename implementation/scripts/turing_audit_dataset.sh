#!/bin/bash
# CPU-only integrity and protocol audit for an already materialized paper dataset.
#SBATCH -p u22
#SBATCH -n 2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

CONFIG="${CONFIG:?CONFIG is required}"
DATASET_DIR="${DATASET_DIR:?DATASET_DIR is required}"
AUDIT_OUTPUT="${AUDIT_OUTPUT:?AUDIT_OUTPUT is required}"
PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
TURING_ACCOUNT="${TURING_ACCOUNT:?TURING_ACCOUNT is required}"
RUNTIME_ROOT="${RUNTIME_ROOT:?RUNTIME_ROOT is required}"

module load u22/cuda/12.4
export PATH="$HOME/.local/bin:$PATH"
if [[ ! -d /scratch || ! -w /scratch ]]; then
  echo "ERROR: /scratch is not writable" >&2
  exit 1
fi
if [[ "$RUNTIME_ROOT" != /scratch/* || "$DATASET_DIR" != /scratch/* ]]; then
  echo "ERROR: runtime and dataset must use node-local /scratch" >&2
  exit 1
fi
if [[ -e "$AUDIT_OUTPUT" ]]; then
  echo "ERROR: refusing to overwrite dataset audit: $AUDIT_OUTPUT" >&2
  exit 1
fi
export UV_CACHE_DIR="$RUNTIME_ROOT/uv_cache"
export UV_PROJECT_ENVIRONMENT="$RUNTIME_ROOT/project_venv"
if [[ ! -f "$UV_PROJECT_ENVIRONMENT/environment_verified.txt" ]]; then
  echo "ERROR: locked runtime verification is missing: $UV_PROJECT_ENVIRONMENT/environment_verified.txt" >&2
  exit 1
fi
mkdir -p "$(dirname "$AUDIT_OUTPUT")"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src"

echo "job_id=${SLURM_JOB_ID} account=${TURING_ACCOUNT} host=$(hostname)"
uv run --frozen --no-sync python -m text_feedback_dpo.cli audit-dataset \
  --config "$CONFIG" \
  --dataset-dir "$DATASET_DIR" \
  --output-path "$AUDIT_OUTPUT"
