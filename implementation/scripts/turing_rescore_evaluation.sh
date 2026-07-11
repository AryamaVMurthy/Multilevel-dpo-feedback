#!/bin/bash
# CPU-only deterministic rescore and manual audit of immutable checkpoint predictions.
#SBATCH -p u22
#SBATCH -n 2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

CONFIG="${CONFIG:?CONFIG is required}"
PREDICTIONS="${PREDICTIONS:?PREDICTIONS is required}"
EXAMPLES="${EXAMPLES:?EXAMPLES is required}"
LABELS="${LABELS:?LABELS is required}"
RESCORE_DIR="${RESCORE_DIR:?RESCORE_DIR is required}"
AUDIT_DIR="${AUDIT_DIR:?AUDIT_DIR is required}"
PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
SOURCE_COMMIT="${SOURCE_COMMIT:?SOURCE_COMMIT is required}"
TURING_ACCOUNT="${TURING_ACCOUNT:?TURING_ACCOUNT is required}"
RUNTIME_ROOT="${RUNTIME_ROOT:?RUNTIME_ROOT is required}"

module load u22/cuda/12.4
export PATH="$HOME/.local/bin:$PATH"
if [[ ! -d /scratch || ! -w /scratch ]]; then
  echo "ERROR: /scratch is not writable; refusing hidden cache fallback" >&2
  exit 1
fi
if [[ "$RUNTIME_ROOT" != /scratch/* ]]; then
  echo "ERROR: RUNTIME_ROOT must be node-local /scratch storage: $RUNTIME_ROOT" >&2
  exit 1
fi
for path in "$CONFIG" "$PREDICTIONS" "$EXAMPLES" "$LABELS"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: required rescore input does not exist: $path" >&2
    exit 1
  fi
done
if [[ -e "$RESCORE_DIR" || -e "$AUDIT_DIR" ]]; then
  echo "ERROR: refusing to overwrite rescore or audit output" >&2
  exit 1
fi

mkdir -p "$RUNTIME_ROOT"
export UV_CACHE_DIR="$RUNTIME_ROOT/uv_cache"
export UV_PROJECT_ENVIRONMENT="$RUNTIME_ROOT/project_venv"
if [[ ! -f "$UV_PROJECT_ENVIRONMENT/environment_verified.txt" ]]; then
  echo "ERROR: locked runtime verification is missing: $UV_PROJECT_ENVIRONMENT/environment_verified.txt" >&2
  exit 1
fi
export UV_LINK_MODE=hardlink
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src"

echo "job_id=${SLURM_JOB_ID} account=${TURING_ACCOUNT} host=$(hostname) source_commit=${SOURCE_COMMIT}"
uv run --frozen python -m text_feedback_dpo.cli rescore-evaluation \
  --predictions "$PREDICTIONS" \
  --examples "$EXAMPLES" \
  --output-dir "$RESCORE_DIR" \
  --source-commit "$SOURCE_COMMIT"
uv run --frozen python -m text_feedback_dpo.cli audit-evaluation \
  --config "$CONFIG" \
  --predictions "$RESCORE_DIR/predictions.jsonl" \
  --labels "$LABELS" \
  --output-dir "$AUDIT_DIR" \
  --minimum-labels 16
