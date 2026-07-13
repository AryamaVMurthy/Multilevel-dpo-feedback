#!/bin/bash
# Deterministically shard a JSONL dataset by ID for parallel GPU generation.
#SBATCH -p u22
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail

: "${PROJECT_DIR:?PROJECT_DIR must be supplied with --export}"
: "${INPUT:?INPUT must be supplied with --export}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be supplied with --export}"
: "${SHARDS:?SHARDS must be supplied with --export}"
cd "$PROJECT_DIR"
[[ -f pyproject.toml && -d src/text_feedback_dpo ]] || { echo "ERROR: invalid PROJECT_DIR" >&2; exit 2; }
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 UV_LINK_MODE=copy
uv run --frozen python -m text_feedback_dpo.cli shard-jsonl \
  --input "$INPUT" --output-dir "$OUTPUT_DIR" --shards "$SHARDS"
