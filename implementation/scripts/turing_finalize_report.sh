#!/bin/bash
# Build the final comparison report after every arm and manifest gate finishes.
#SBATCH -p u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4096
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail

COMPONENT=turing_finalize_report
log_event() { local event="$1"; shift; printf 'event=%s timestamp=%s component=%s %s\n' "$event" "$(date -u +%FT%TZ)" "$COMPONENT" "$*"; }
fail() { log_event failure reason="$1" fallback_reason="${2:-none}" >&2; exit 2; }
hash_value() { printf '%s' "$1" | sha256sum | awk '{print $1}'; }
hash_path() { if [[ -f "$1" ]]; then sha256sum "$1" | awk '{print $1}'; else hash_value "$1"; fi; }
write_manifest() {
  local status="$1"; export MANIFEST_STATUS="$status" MANIFEST_ENDED_AT="$(date -u +%FT%TZ)"
  uv run --frozen python - "$RUN_MANIFEST" <<'PY'
import json, os, platform, socket, sys
manifest = {"status": os.environ["MANIFEST_STATUS"], "commit_hash": os.environ["COMMIT_HASH"], "config_hash": os.environ["CONFIG_HASH"], "model_hash": os.environ["MODEL_HASH"], "dataset_hash": os.environ["DATASET_HASH"], "prompt_hash": os.environ["PROMPT_HASH"], "retrieval_hash": os.environ["RETRIEVAL_HASH"], "source_schema_hash": os.environ["SOURCE_SCHEMA_HASH"], "node": socket.gethostname(), "platform": platform.platform(), "slurm_allocation": {k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_NNODES", "SLURM_NTASKS")}, "package_versions": os.environ["PACKAGE_VERSIONS"].split(";"), "gpu_telemetry": os.environ["GPU_TELEMETRY"], "timings": {"started_at": os.environ["MANIFEST_STARTED_AT"], "ended_at": os.environ["MANIFEST_ENDED_AT"]}, "artifact_paths": os.environ["ARTIFACT_PATHS"].split("|"), "fallback_reason": "none", "max_length": 4096}
with open(sys.argv[1], "w", encoding="utf-8") as handle: json.dump(manifest, handle, sort_keys=True, indent=2); handle.write("\n")
PY
}

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
: "${PROMPT_HASH:?PROMPT_HASH must be supplied with --export}"
: "${RETRIEVAL_HASH:?RETRIEVAL_HASH must be supplied with --export}"
: "${SOURCE_SCHEMA_HASH:?SOURCE_SCHEMA_HASH must be supplied with --export}"
: "${DATASET_HASH:?DATASET_HASH must be supplied with --export}"

cd "$PROJECT_DIR"
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_CACHE_ROOT:-/scratch/$USER/searchqa-dpo/hf}"
export HF_DATASETS_CACHE="$HF_HOME/datasets" HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" logs "$OUTPUT_ROOT/final-report"
for required in "$BASELINE_VALIDATION" "$BASELINE_TEST" "$DPO_VALIDATION" "$DPO_TEST" "$OFFLINE_REUSE_ARTIFACT" "$DATA_MANIFEST" "$TRAIN_MANIFEST" "$VALIDATION_MANIFEST" "$TEST_MANIFEST"; do [[ -s "$required" ]] || fail "required artifact is missing: $required" "missing_artifact"; done

COMMIT_HASH="$(git rev-parse HEAD)" CONFIG_HASH="$(hash_path "$CONFIG")" MODEL_HASH="${MODEL_HASH:-$(hash_value final-report)}" RUN_MANIFEST="${RUN_MANIFEST:-$OUTPUT_ROOT/final-report/run-manifest.json}" GPU_TELEMETRY="not_allocated" ARTIFACT_PATHS="$CONFIG|$DATA_MANIFEST|$TRAIN_MANIFEST|$VALIDATION_MANIFEST|$TEST_MANIFEST|$OFFLINE_REUSE_ARTIFACT|$OUTPUT_ROOT/comparison.json" MANIFEST_STARTED_AT="$(date -u +%FT%TZ)"
PACKAGE_VERSIONS="$(uv run --frozen python - <<'PY'
import importlib.metadata
print(";".join(f"{n}={importlib.metadata.version(n)}" for n in ("datasets", "transformers", "trl")))
PY
)"
export COMMIT_HASH CONFIG_HASH MODEL_HASH DATASET_HASH PROMPT_HASH RETRIEVAL_HASH SOURCE_SCHEMA_HASH RUN_MANIFEST GPU_TELEMETRY ARTIFACT_PATHS MANIFEST_STARTED_AT PACKAGE_VERSIONS
log_event report_start baseline=present dpo=present comparisons=sft,grpo,dapo offline_reuse=present max_length=4096
REPORT_ROOT="$OUTPUT_ROOT/final-report"
RUN_ARGS=(--run "baseline-validation=$BASELINE_VALIDATION" --run "baseline-test=$BASELINE_TEST" --run "dpo-validation=$DPO_VALIDATION" --run "dpo-test=$DPO_TEST" --run "sft-validation=$OUTPUT_ROOT/sft-validation-metrics.json" --run "sft-test=$OUTPUT_ROOT/sft-test-metrics.json" --run "grpo-validation=$OUTPUT_ROOT/grpo-validation-metrics.json" --run "grpo-test=$OUTPUT_ROOT/grpo-test-metrics.json" --run "dapo-validation=$OUTPUT_ROOT/dapo-validation-metrics.json" --run "dapo-test=$OUTPUT_ROOT/dapo-test-metrics.json")
ARTIFACT_ARGS=(--artifact "$CONFIG" --artifact "$DATA_MANIFEST" --artifact "$TRAIN_MANIFEST" --artifact "$VALIDATION_MANIFEST" --artifact "$TEST_MANIFEST" --artifact "$OFFLINE_REUSE_ARTIFACT" --artifact "$BASELINE_VALIDATION" --artifact "$BASELINE_TEST" --artifact "$DPO_VALIDATION" --artifact "$DPO_TEST" --artifact "$OUTPUT_ROOT/comparison.json")
for method in sft grpo dapo; do for split in validation test; do ARTIFACT_ARGS+=(--artifact "$OUTPUT_ROOT/$method-$split-metrics.json"); done; done
uv run --frozen python -m text_feedback_dpo.cli compare "${RUN_ARGS[@]}" --output "$REPORT_ROOT/comparison-with-baseline.json" --html "$REPORT_ROOT/comparison-with-baseline.html" "${ARTIFACT_ARGS[@]}"
log_event report_complete artifact="$REPORT_ROOT/comparison-with-baseline.html"
write_manifest complete
