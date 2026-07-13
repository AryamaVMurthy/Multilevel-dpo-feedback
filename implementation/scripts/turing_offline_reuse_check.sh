#!/bin/bash
# Verify complete trajectory-cache reuse without loading student or teacher models.
# Submit with: sbatch -A <account> --export=ALL,... scripts/turing_offline_reuse_check.sh
#SBATCH -p u22
#SBATCH -n 2
#SBATCH --mem-per-cpu=4096
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail

: "${TURING_ACCOUNT:?TURING_ACCOUNT must be supplied with --export}"
: "${PROJECT_DIR:?PROJECT_DIR must be supplied with --export}"
: "${DATA:?DATA must be supplied with --export}"
: "${CACHE:?CACHE must be supplied with --export}"
: "${OUTPUT:?OUTPUT must be supplied with --export}"
: "${POLICY_HASH:?POLICY_HASH must be supplied with --export}"
: "${STUDENT_MODEL:?STUDENT_MODEL must be supplied with --export}"
: "${STUDENT_REVISION:?STUDENT_REVISION must be supplied with --export}"
: "${TEACHER_MODEL:?TEACHER_MODEL must be supplied with --export}"
: "${TEACHER_REVISION:?TEACHER_REVISION must be supplied with --export}"
: "${DATASET_REVISION:?DATASET_REVISION must be supplied with --export}"
: "${PROMPT_VERSION:?PROMPT_VERSION must be supplied with --export}"
: "${SEED:?SEED must be supplied with --export}"

cd "$PROJECT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_CACHE_ROOT:-/scratch/$USER/searchqa-dpo/hf}"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" logs "$(dirname "$OUTPUT")"

[[ -s "$CACHE" ]] || { echo "ERROR: complete trajectory cache is missing or empty: $CACHE" >&2; exit 2; }
[[ "$CACHE" != "$OUTPUT" ]] || { echo "ERROR: offline output must differ from the cache path" >&2; exit 2; }
rm -f "$OUTPUT"
uv run --frozen python -m text_feedback_dpo.cli collect \
  --data "$DATA" --output "$OUTPUT" \
  --student-model "$STUDENT_MODEL" \
  --student-revision "$STUDENT_REVISION" \
  --teacher-model "$TEACHER_MODEL" \
  --teacher-revision "$TEACHER_REVISION" \
  --dataset-revision "$DATASET_REVISION" --prompt-version "$PROMPT_VERSION" --seed "$SEED" \
  --teacher-quantization 4bit \
  --attention-implementation "${ATTENTION_IMPLEMENTATION:-sdpa}" \
  --student-device cuda:1 --teacher-device cuda:0 \
  --trajectory-cache "$CACHE" --policy-hash "$POLICY_HASH" \
  --max-interventions 4 --student-batch-size 1 --teacher-batch-size 1 --student-thinking-mode "${STUDENT_THINKING_MODE:-direct}" --teacher-thinking

cmp -s "$CACHE" "$OUTPUT" || {
  echo "ERROR: offline trajectory replay differs from the cache" >&2
  sha256sum "$CACHE" "$OUTPUT" >&2
  exit 2
}
echo "<offline_reuse cache=\"$CACHE\" output=\"$OUTPUT\" generation=\"skipped_complete_cache\" byte_identical=\"true\"/>"
