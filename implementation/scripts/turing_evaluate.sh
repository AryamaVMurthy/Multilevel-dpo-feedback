#!/bin/bash
# Evaluate an explicitly selected protocol. New cited-reasoning runs must pass active-search.
#SBATCH -p u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4096
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail

COMPONENT=turing_evaluate
log_event() { local event="$1"; shift; printf 'event=%s timestamp=%s component=%s %s\n' "$event" "$(date -u +%FT%TZ)" "$COMPONENT" "$*"; }
fail() { log_event failure reason="$1" fallback_reason="${2:-none}" >&2; exit 2; }

: "${TURING_ACCOUNT:?TURING_ACCOUNT must be supplied with --export}"
: "${PROJECT_DIR:?PROJECT_DIR must be supplied with --export}"
: "${DATA:?DATA must be supplied with --export}"
: "${PREDICTIONS:?PREDICTIONS must be supplied with --export}"
: "${OUTPUT:?OUTPUT must be supplied with --export}"
: "${PROTOCOL:?PROTOCOL must be archival or active-search}"
[[ "$PROTOCOL" == "archival" || "$PROTOCOL" == "active-search" ]] || fail "unsupported PROTOCOL=$PROTOCOL" "protocol_validation"

cd "$PROJECT_DIR"
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_CACHE_ROOT:-/scratch/$USER/searchqa-dpo/hf}" HF_DATASETS_CACHE="$HF_HOME/datasets" HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" logs "$(dirname "$OUTPUT")"
log_event evaluation_start protocol="$PROTOCOL" data="$DATA" predictions="$PREDICTIONS" fallback_reason=none
uv run --frozen python -m text_feedback_dpo.cli evaluate --data "$DATA" --predictions "$PREDICTIONS" --output "$OUTPUT" --protocol "$PROTOCOL"
log_event evaluation_complete protocol="$PROTOCOL" artifact="$OUTPUT"
