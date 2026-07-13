#!/bin/bash
# Submit with: sbatch -A <account> --export=ALL,... scripts/turing_merge_predictions.sh
#SBATCH -p u22
#SBATCH -n 2
#SBATCH --mem-per-cpu=4096
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail

: "${TURING_ACCOUNT:?TURING_ACCOUNT must be supplied with --export}"
: "${PROJECT_DIR:?PROJECT_DIR must be supplied with --export}"
: "${SHARD_DIR:?SHARD_DIR must be supplied with --export}"
: "${SHARDS:?SHARDS must be supplied with --export}"
: "${OUTPUT:?OUTPUT must be supplied with --export}"
CLEANUP_INPUT_SHARDS="${CLEANUP_INPUT_SHARDS:-true}"

cd "$PROJECT_DIR"
mkdir -p "$(dirname "$OUTPUT")"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
uv run --frozen python -m text_feedback_dpo.cli merge-predictions \
  --shard-dir "$SHARD_DIR" --output "$OUTPUT" --shards "$SHARDS"
if [[ "$CLEANUP_INPUT_SHARDS" == "true" ]]; then
  for ((index=0; index<SHARDS; index++)); do
    rm -f -- "$SHARD_DIR/shard-${index}.jsonl" "$SHARD_DIR/predictions-${index}.jsonl"
  done
  if [[ -d "$SHARD_DIR" ]]; then
    rmdir "$SHARD_DIR"
  fi
  echo "<cleanup component=merge_predictions removed=duplicate_shard_inputs reason=verified_merged_output/>"
elif [[ "$CLEANUP_INPUT_SHARDS" != "false" ]]; then
  echo "ERROR: CLEANUP_INPUT_SHARDS must be true or false" >&2
  exit 2
fi
