#!/bin/bash
# CPU-only immutable baseline evaluation freeze.
#SBATCH -p u22
#SBATCH -n 2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

CONFIG="${CONFIG:?CONFIG is required}"
DATASET_MANIFEST="${DATASET_MANIFEST:?DATASET_MANIFEST is required}"
SOURCE_COMMIT="${SOURCE_COMMIT:?SOURCE_COMMIT is required}"
FREEZE_OUTPUT="${FREEZE_OUTPUT:?FREEZE_OUTPUT is required}"
PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
TURING_ACCOUNT="${TURING_ACCOUNT:?TURING_ACCOUNT is required}"

module load u22/cuda/12.4
export PATH="$HOME/.local/bin:$PATH"
if [[ ! -d /scratch || ! -w /scratch ]]; then
  echo "ERROR: /scratch is not writable; refusing home cache fallback" >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: paper config does not exist: $CONFIG" >&2
  exit 1
fi
if [[ ! -f "$DATASET_MANIFEST" ]]; then
  echo "ERROR: dataset manifest does not exist: $DATASET_MANIFEST" >&2
  exit 1
fi
mkdir -p "$(dirname "$FREEZE_OUTPUT")"
export UV_CACHE_DIR="$HOME/tfdpo-runs/uv_cache"
export UV_PROJECT_ENVIRONMENT="$HOME/tfdpo-runs/project_venv"
export UV_LINK_MODE=hardlink
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src"

echo "job_id=${SLURM_JOB_ID} account=${TURING_ACCOUNT} host=$(hostname) source_commit=${SOURCE_COMMIT}"
uv run --frozen python -m text_feedback_dpo.cli freeze-baseline \
  --config "$CONFIG" \
  --dataset-manifest "$DATASET_MANIFEST" \
  --source-commit "$SOURCE_COMMIT" \
  --output "$FREEZE_OUTPUT"
