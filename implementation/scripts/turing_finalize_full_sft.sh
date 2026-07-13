#!/bin/bash
# CPU-only, provenance-bound manifest finalization for a completed full-SFT run.
#SBATCH -p u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4096
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail

COMPONENT=turing_finalize_full_sft
log_event() { local event="$1"; shift; printf 'event=%s timestamp=%s component=%s %s\n' "$event" "$(date -u +%FT%TZ)" "$COMPONENT" "$*"; }
fail() { log_event failure reason="$1" fallback_reason="${2:-none}" >&2; exit 2; }
require_env() { local name="$1"; [[ -n "${!name:-}" ]] || fail "required environment variable is missing: $name" full_sft_finalize_contract_missing; }

for name in PROJECT_DIR EXPECTED_COMMIT CONFIG CONFIG_SHA256 TRAIN TRAIN_SHA256 EVAL EVAL_SHA256 \
  SPLIT_REPORT SPLIT_REPORT_SHA256 OUTPUT START_MODEL START_MODEL_SHA256 START_REVISION \
  DATASET_SOURCE DATASET_REVISION PROMPT_HASH RETRIEVAL_HASH SOURCE_SCHEMA_HASH LEARNING_RATE \
  EPOCHS SAVE_STEPS EVAL_STEPS GRADIENT_ACCUMULATION_STEPS EXPECTED_TRAIN_PAIRS \
  EXPECTED_EVAL_PAIRS GPU_TELEMETRY FINAL_MODEL_SHA256; do
  require_env "$name"
done
[[ "${SLURM_NNODES:?}" == 1 && "${SLURM_NTASKS:?}" == 1 ]] || fail "full SFT finalizer requires one node and one task" allocation_shape_invalid
[[ -f "$CONFIG" && -f "$TRAIN" && -f "$EVAL" && -f "$SPLIT_REPORT" ]] || fail "config, train, eval, or split report is missing" finalize_input_missing
[[ -f "$OUTPUT/final/model.safetensors" && -s "$GPU_TELEMETRY" ]] || fail "final model or GPU telemetry is missing" finalize_output_missing
[[ ! -e "$OUTPUT/run-manifest.json" ]] || fail "run manifest already exists; refusing overwrite" manifest_exists
[[ "$(git -C "$PROJECT_DIR" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || fail "remote commit differs from EXPECTED_COMMIT" commit_mismatch
[[ "$(sha256sum "$CONFIG" | awk '{print $1}')" == "$CONFIG_SHA256" ]] || fail "config artifact hash mismatch" config_hash_mismatch
[[ "$(sha256sum "$TRAIN" | awk '{print $1}')" == "$TRAIN_SHA256" ]] || fail "train artifact hash mismatch" train_hash_mismatch
[[ "$(sha256sum "$EVAL" | awk '{print $1}')" == "$EVAL_SHA256" ]] || fail "eval artifact hash mismatch" eval_hash_mismatch
[[ "$(sha256sum "$SPLIT_REPORT" | awk '{print $1}')" == "$SPLIT_REPORT_SHA256" ]] || fail "split report hash mismatch" split_report_hash_mismatch
[[ "$(sha256sum "$START_MODEL/model.safetensors" | awk '{print $1}')" == "$START_MODEL_SHA256" ]] || fail "start model hash mismatch" start_model_hash_mismatch
[[ "$(sha256sum "$OUTPUT/final/model.safetensors" | awk '{print $1}')" == "$FINAL_MODEL_SHA256" ]] || fail "final model hash mismatch" final_model_hash_mismatch

cd "$PROJECT_DIR"
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 UV_LINK_MODE=copy
log_event manifest_finalize_start output="$OUTPUT" final_model_sha256="$FINAL_MODEL_SHA256" fallback_reason=none
uv run --frozen python -m text_feedback_dpo.sft_manifest --output "$OUTPUT"
[[ -s "$OUTPUT/run-manifest.json" ]] || fail "manifest finalizer did not create a non-empty manifest" manifest_missing
log_event manifest_finalize_complete manifest="$OUTPUT/run-manifest.json" manifest_sha256="$(sha256sum "$OUTPUT/run-manifest.json" | awk '{print $1}')" fallback_reason=none
