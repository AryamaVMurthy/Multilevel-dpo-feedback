#!/bin/bash
# Submit with: sbatch -A <account> --export=ALL,... scripts/turing_evaluate.sh
#SBATCH -p u22
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4096
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail

: "${TURING_ACCOUNT:?TURING_ACCOUNT must be supplied with --export}"
: "${PROJECT_DIR:?PROJECT_DIR must be supplied with --export}"
: "${DATA:?DATA must be supplied with --export}"
: "${PREDICTIONS:?PREDICTIONS must be supplied with --export}"
: "${OUTPUT:?OUTPUT must be supplied with --export}"

cd "$PROJECT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_CACHE_ROOT:-/scratch/$USER/searchqa-dpo/hf}"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" logs
uv run --frozen python -m text_feedback_dpo.cli evaluate \
  --data "$DATA" --predictions "$PREDICTIONS" --output "$OUTPUT"
