#!/bin/bash
# Build the final teacher-free comparison report after every comparison arm finishes.
# Submit with: sbatch -A <account> --dependency=afterok:<comparison-job> --export=ALL,... scripts/turing_finalize_report.sh
#SBATCH -p u22
#SBATCH -n 2
#SBATCH --mem-per-cpu=4096
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail

: "${TURING_ACCOUNT:?TURING_ACCOUNT must be supplied with --export}"
: "${PROJECT_DIR:?PROJECT_DIR must be supplied with --export}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT must be supplied with --export}"
: "${BASELINE_VALIDATION:?BASELINE_VALIDATION must be supplied with --export}"
: "${BASELINE_TEST:?BASELINE_TEST must be supplied with --export}"
: "${DPO_VALIDATION:?DPO_VALIDATION must be supplied with --export}"
: "${DPO_TEST:?DPO_TEST must be supplied with --export}"
: "${CONFIG:?CONFIG must be supplied with --export}"
: "${DATA_MANIFEST:?DATA_MANIFEST must be supplied with --export}"
: "${TRAIN_MANIFEST:?TRAIN_MANIFEST must be supplied with --export}"
: "${VALIDATION_MANIFEST:?VALIDATION_MANIFEST must be supplied with --export}"
: "${TEST_MANIFEST:?TEST_MANIFEST must be supplied with --export}"
: "${OFFLINE_REUSE_ARTIFACT:?OFFLINE_REUSE_ARTIFACT must be supplied with --export}"

cd "$PROJECT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_CACHE_ROOT:-/scratch/$USER/searchqa-dpo/hf}"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" logs "$OUTPUT_ROOT/final-report"

[[ -s "$BASELINE_VALIDATION" ]] || { echo "ERROR: missing baseline validation metrics: $BASELINE_VALIDATION" >&2; exit 2; }
[[ -s "$BASELINE_TEST" ]] || { echo "ERROR: missing baseline test metrics: $BASELINE_TEST" >&2; exit 2; }
[[ -s "$DPO_VALIDATION" ]] || { echo "ERROR: missing DPO validation metrics: $DPO_VALIDATION" >&2; exit 2; }
[[ -s "$DPO_TEST" ]] || { echo "ERROR: missing DPO test metrics: $DPO_TEST" >&2; exit 2; }
[[ -s "$OFFLINE_REUSE_ARTIFACT" ]] || { echo "ERROR: missing offline-reuse artifact: $OFFLINE_REUSE_ARTIFACT" >&2; exit 2; }

for manifest in "$DATA_MANIFEST" "$TRAIN_MANIFEST" "$VALIDATION_MANIFEST" "$TEST_MANIFEST"; do
  [[ -s "$manifest" ]] || { echo "ERROR: missing required manifest: $manifest" >&2; exit 2; }
done

REPORT_ROOT="$OUTPUT_ROOT/final-report"
RUN_ARGS=(
  --run "baseline-validation=$BASELINE_VALIDATION"
  --run "baseline-test=$BASELINE_TEST"
  --run "dpo-validation=$DPO_VALIDATION"
  --run "dpo-test=$DPO_TEST"
  --run "sft-validation=$OUTPUT_ROOT/sft-validation-metrics.json"
  --run "sft-test=$OUTPUT_ROOT/sft-test-metrics.json"
  --run "grpo-validation=$OUTPUT_ROOT/grpo-validation-metrics.json"
  --run "grpo-test=$OUTPUT_ROOT/grpo-test-metrics.json"
  --run "dapo-validation=$OUTPUT_ROOT/dapo-validation-metrics.json"
  --run "dapo-test=$OUTPUT_ROOT/dapo-test-metrics.json"
)
ARTIFACT_ARGS=(
  --artifact "$CONFIG"
  --artifact "$DATA_MANIFEST"
  --artifact "$TRAIN_MANIFEST"
  --artifact "$VALIDATION_MANIFEST"
  --artifact "$TEST_MANIFEST"
  --artifact "$OFFLINE_REUSE_ARTIFACT"
  --artifact "$BASELINE_VALIDATION"
  --artifact "$BASELINE_TEST"
  --artifact "$DPO_VALIDATION"
  --artifact "$DPO_TEST"
  --artifact "$OUTPUT_ROOT/comparison.json"
)
for method in sft grpo dapo; do
  for split in validation test; do
    ARTIFACT_ARGS+=(--artifact "$OUTPUT_ROOT/$method-$split-metrics.json")
  done
done

uv run --frozen python -m text_feedback_dpo.cli compare \
  "${RUN_ARGS[@]}" \
  --output "$REPORT_ROOT/comparison-with-baseline.json" \
  --html "$REPORT_ROOT/comparison-with-baseline.html" \
  "${ARTIFACT_ARGS[@]}"

echo "<report status=complete output=\"$REPORT_ROOT/comparison-with-baseline.html\" baseline=\"included\" dpo=\"included\" comparisons=\"sft,grpo,dapo\" offline_reuse=\"included\"/>"
