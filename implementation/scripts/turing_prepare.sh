#!/bin/bash
# Submit with: sbatch -A <account> --export=ALL,PROJECT_DIR=...,SPLIT=...,OUTPUT=... scripts/turing_prepare.sh
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
: "${SPLIT:?SPLIT must be train, validation, or test}"
: "${OUTPUT:?OUTPUT must be supplied with --export}"
: "${TOKENIZER_MODEL:?TOKENIZER_MODEL must be supplied with --export}"
: "${TOKENIZER_REVISION:?TOKENIZER_REVISION must be supplied with --export}"
: "${SEARCHQA_REVISION:?SEARCHQA_REVISION must be supplied with --export}"

cd "$PROJECT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_CACHE_ROOT:-/scratch/$USER/searchqa-dpo/hf}"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" logs
uv run --frozen python -m text_feedback_dpo.cli prepare-searchqa \
  --source kyunghyuncho/search_qa --split "$SPLIT" --revision "$SEARCHQA_REVISION" \
  --tokenizer-model "$TOKENIZER_MODEL" --tokenizer-revision "$TOKENIZER_REVISION" --output "$OUTPUT" \
  --max-evidence-tokens 3000
