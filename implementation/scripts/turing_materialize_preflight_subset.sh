#!/bin/bash
# CPU-only deterministic baseline preflight subset materialization.
#SBATCH -p u22
#SBATCH -n 2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

SOURCE_PATH="${SOURCE_PATH:?SOURCE_PATH is required}"
DATASET_MANIFEST="${DATASET_MANIFEST:?DATASET_MANIFEST is required}"
OUTPUT_PATH="${OUTPUT_PATH:?OUTPUT_PATH is required}"
SUBSET_COUNT="${SUBSET_COUNT:?SUBSET_COUNT is required}"
SUBSET_SEED="${SUBSET_SEED:?SUBSET_SEED is required}"
PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
TURING_ACCOUNT="${TURING_ACCOUNT:?TURING_ACCOUNT is required}"

module load u22/cuda/12.4
export PATH="$HOME/.local/bin:$PATH"
if [[ ! -d /scratch || ! -w /scratch ]]; then
  echo "ERROR: /scratch is not writable; refusing home cache fallback" >&2
  exit 1
fi
if [[ ! -f "$SOURCE_PATH" ]]; then
  echo "ERROR: source dataset does not exist: $SOURCE_PATH" >&2
  exit 1
fi
if [[ ! -f "$DATASET_MANIFEST" ]]; then
  echo "ERROR: dataset manifest does not exist: $DATASET_MANIFEST" >&2
  exit 1
fi
mkdir -p "$(dirname "$OUTPUT_PATH")"
export UV_CACHE_DIR="$HOME/tfdpo-runs/uv_cache"
export UV_PROJECT_ENVIRONMENT="$HOME/tfdpo-runs/project_venv"
export UV_LINK_MODE=hardlink
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src"

echo "job_id=${SLURM_JOB_ID} account=${TURING_ACCOUNT} host=$(hostname)"
uv run --frozen python -m text_feedback_dpo.cli materialize-preflight-subset \
  --source-path "$SOURCE_PATH" \
  --output-path "$OUTPUT_PATH" \
  --count "$SUBSET_COUNT" \
  --seed "$SUBSET_SEED"

OUTPUT_MANIFEST="$(dirname "$OUTPUT_PATH")/manifest.json"
if [[ -e "$OUTPUT_MANIFEST" ]] && ! cmp -s "$DATASET_MANIFEST" "$OUTPUT_MANIFEST"; then
  echo "ERROR: preflight output directory has a conflicting manifest: $OUTPUT_MANIFEST" >&2
  exit 1
fi
if [[ ! -e "$OUTPUT_MANIFEST" ]]; then
  cp "$DATASET_MANIFEST" "$OUTPUT_MANIFEST"
fi
