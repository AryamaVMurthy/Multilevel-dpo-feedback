#!/bin/bash
# Gate DPO on the measured scale decision and a real save/resume checkpoint smoke.
# Submit this CPU job after ref-logprob preparation and scale freezing.
# It submits the GPU jobs only after every identity gate passes.
#SBATCH -p u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4096
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail

COMPONENT=turing_dpo_after_freeze
log_event() { local event="$1"; shift; printf 'event=%s timestamp=%s component=%s %s\n' "$event" "$(date -u +%FT%TZ)" "$COMPONENT" "$*"; }
fail() { log_event failure reason="$1" fallback_reason="${2:-none}" >&2; exit 2; }
hash_file() { sha256sum "$1" | awk '{print $1}'; }

required=(
  TURING_ACCOUNT PROJECT_DIR EXPECTED_COMMIT_HASH CONFIG TRAIN EVAL OUTPUT
  START_MODEL START_REVISION START_MODEL_SHA256 DATASET_SOURCE DATASET_REVISION
  PROMPT_HASH RETRIEVAL_HASH SOURCE_SCHEMA_HASH OPTIMIZATION_DECISION
  SCALE_DECISION REF_LOG_PROBS EVAL_REF_LOG_PROBS REFERENCE_CHECKPOINT_HASH PROMPT_CONTEXT_SCHEMA
  EFFECTIVE_BATCH_SIZE LEARNING_RATE EPOCHS SAVE_STEPS EVAL_STEPS INITIAL_MAX_STEPS RESUMED_MAX_STEPS
  SMOKE_ROOT SMOKE_MANIFEST RUN_MANIFEST
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || fail "$name must be supplied explicitly" dpo_continuation_contract_missing
done

module load u22/cuda/12.4
cd "$PROJECT_DIR"
[[ -f pyproject.toml && -d src/text_feedback_dpo ]] || fail "invalid PROJECT_DIR" invalid_project_root
CURRENT_COMMIT_HASH="$(git rev-parse HEAD)"
[[ "$CURRENT_COMMIT_HASH" == "$EXPECTED_COMMIT_HASH" ]] || fail "current commit=$CURRENT_COMMIT_HASH expected=$EXPECTED_COMMIT_HASH" commit_identity_mismatch
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
PROBE_RUNNER="$PROJECT_DIR/scripts/turing_probe_runner.py"
[[ -x "$PROBE_RUNNER" ]] || fail "probe runner is not executable: $PROBE_RUNNER" probe_runner_missing
run_probe_runner() { uv run --frozen python "$PROBE_RUNNER" "$@"; }

CONFIG_SHA256="$(hash_file "$CONFIG")"
DATASET_SHA256="$(hash_file "$TRAIN")"
EVAL_DATASET_SHA256="$(hash_file "$EVAL")"
[[ -f "$OPTIMIZATION_DECISION" ]] || fail "optimization decision is not available: $OPTIMIZATION_DECISION" optimization_decision_missing
[[ -f "$SCALE_DECISION" ]] || fail "scale decision is not available: $SCALE_DECISION" scale_decision_missing
OPTIMIZATION_DECISION_SHA256="$(hash_file "$OPTIMIZATION_DECISION")"
SCALE_DECISION_SHA256="$(hash_file "$SCALE_DECISION")"
SELECTED_TRAIN_GPUS="$(uv run --frozen python - "$SCALE_DECISION" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
selected = payload.get("selected_train_gpus")
if selected not in (4, 8):
    raise SystemExit(f"scale decision selected_train_gpus must be 4 or 8, got {selected!r}")
print(selected)
PY
)"
IFS=$'\t' read -r _validated_train_gpus _validated_scale_hash _scale_fallback < <(
  run_probe_runner validate-scale-decision --decision "$SCALE_DECISION" --expected-sha256 "$SCALE_DECISION_SHA256" \
    --train-gpus "$SELECTED_TRAIN_GPUS" --training-method dpo --commit-hash "$CURRENT_COMMIT_HASH" \
    --config-sha256 "$CONFIG_SHA256" --model "$START_MODEL" --model-revision "$START_REVISION" \
    --dataset-source "$DATASET_SOURCE" --dataset-revision "$DATASET_REVISION" --dataset-sha256 "$DATASET_SHA256" \
    --prompt-sha256 "$PROMPT_HASH" --retrieval-sha256 "$RETRIEVAL_HASH" --source-schema-sha256 "$SOURCE_SCHEMA_HASH"
)
[[ "$_validated_train_gpus" == "$SELECTED_TRAIN_GPUS" ]] || fail "validated scale selected $_validated_train_gpus GPUs, expected $SELECTED_TRAIN_GPUS" scale_validation_drift
[[ "$_validated_scale_hash" == "$SCALE_DECISION_SHA256" ]] || fail "validated scale hash drifted" scale_validation_hash_drift
log_event scale_gate_passed train_gpus="$SELECTED_TRAIN_GPUS" scale_decision_sha256="$SCALE_DECISION_SHA256" fallback_reason="$_scale_fallback"

SMOKE_EXPORT="ALL,TURING_ACCOUNT=$TURING_ACCOUNT,PROJECT_DIR=$PROJECT_DIR,EXPECTED_COMMIT_HASH=$EXPECTED_COMMIT_HASH,METHOD=dpo,CONFIG=$CONFIG,TRAIN=$TRAIN,EVAL=$EVAL,SMOKE_ROOT=$SMOKE_ROOT,SMOKE_MANIFEST=$SMOKE_MANIFEST,START_MODEL=$START_MODEL,START_REVISION=$START_REVISION,START_MODEL_SHA256=$START_MODEL_SHA256,DATASET_SOURCE=$DATASET_SOURCE,DATASET_REVISION=$DATASET_REVISION,PROMPT_HASH=$PROMPT_HASH,RETRIEVAL_HASH=$RETRIEVAL_HASH,SOURCE_SCHEMA_HASH=$SOURCE_SCHEMA_HASH,OPTIMIZATION_DECISION=$OPTIMIZATION_DECISION,OPTIMIZATION_DECISION_SHA256=$OPTIMIZATION_DECISION_SHA256,SCALE_DECISION=$SCALE_DECISION,SCALE_DECISION_SHA256=$SCALE_DECISION_SHA256,TRAIN_GPUS=$SELECTED_TRAIN_GPUS,INITIAL_MAX_STEPS=$INITIAL_MAX_STEPS,RESUMED_MAX_STEPS=$RESUMED_MAX_STEPS,LEARNING_RATE=$LEARNING_RATE,REF_LOG_PROBS=$REF_LOG_PROBS,EVAL_REF_LOG_PROBS=$EVAL_REF_LOG_PROBS,REFERENCE_CHECKPOINT_HASH=$REFERENCE_CHECKPOINT_HASH,PROMPT_CONTEXT_SCHEMA=$PROMPT_CONTEXT_SCHEMA"
log_event checkpoint_smoke_submit train_gpus="$SELECTED_TRAIN_GPUS" fallback_reason=none
sbatch --wait --account="$TURING_ACCOUNT" --nodes=1 --ntasks=1 --gres="gpu:$SELECTED_TRAIN_GPUS" --export="$SMOKE_EXPORT" "$PROJECT_DIR/scripts/turing_checkpoint_smoke.sh"
[[ -f "$SMOKE_MANIFEST" ]] || fail "checkpoint smoke did not produce manifest: $SMOKE_MANIFEST" checkpoint_smoke_manifest_missing
CHECKPOINT_SMOKE_MANIFEST_SHA256="$(hash_file "$SMOKE_MANIFEST")"
run_probe_runner validate-checkpoints --smoke-manifest "$SMOKE_MANIFEST" --expected-sha256 "$CHECKPOINT_SMOKE_MANIFEST_SHA256" \
  --commit-hash "$CURRENT_COMMIT_HASH" --config-sha256 "$CONFIG_SHA256" --model "$START_MODEL" --model-revision "$START_REVISION" \
  --dataset-source "$DATASET_SOURCE" --dataset-revision "$DATASET_REVISION" --dataset-sha256 "$DATASET_SHA256" \
  --eval-dataset-sha256 "$EVAL_DATASET_SHA256" --prompt-sha256 "$PROMPT_HASH" --retrieval-sha256 "$RETRIEVAL_HASH" \
  --source-schema-sha256 "$SOURCE_SCHEMA_HASH" --optimization-decision-sha256 "$OPTIMIZATION_DECISION_SHA256" \
  --scale-decision-sha256 "$SCALE_DECISION_SHA256" --method dpo >/dev/null
log_event checkpoint_smoke_gate_passed manifest="$SMOKE_MANIFEST" manifest_sha256="$CHECKPOINT_SMOKE_MANIFEST_SHA256" fallback_reason=none

TRAIN_EXPORT="ALL,TURING_ACCOUNT=$TURING_ACCOUNT,PROJECT_DIR=$PROJECT_DIR,METHOD=dpo,CONFIG=$CONFIG,TRAIN=$TRAIN,EVAL=$EVAL,OUTPUT=$OUTPUT,TRAIN_GPUS=$SELECTED_TRAIN_GPUS,EFFECTIVE_BATCH_SIZE=$EFFECTIVE_BATCH_SIZE,LEARNING_RATE=$LEARNING_RATE,EPOCHS=$EPOCHS,SAVE_STEPS=$SAVE_STEPS,EVAL_STEPS=$EVAL_STEPS,CHECKPOINT_SMOKE_MANIFEST=$SMOKE_MANIFEST,CHECKPOINT_SMOKE_MANIFEST_SHA256=$CHECKPOINT_SMOKE_MANIFEST_SHA256,START_MODEL=$START_MODEL,START_REVISION=$START_REVISION,START_MODEL_SHA256=$START_MODEL_SHA256,OPTIMIZATION_DECISION=$OPTIMIZATION_DECISION,OPTIMIZATION_DECISION_SHA256=$OPTIMIZATION_DECISION_SHA256,SCALE_DECISION=$SCALE_DECISION,SCALE_DECISION_SHA256=$SCALE_DECISION_SHA256,REF_LOG_PROBS=$REF_LOG_PROBS,EVAL_REF_LOG_PROBS=$EVAL_REF_LOG_PROBS,REFERENCE_CHECKPOINT_HASH=$REFERENCE_CHECKPOINT_HASH,PROMPT_CONTEXT_SCHEMA=$PROMPT_CONTEXT_SCHEMA,DATASET_SOURCE=$DATASET_SOURCE,DATASET_REVISION=$DATASET_REVISION,PROMPT_HASH=$PROMPT_HASH,RETRIEVAL_HASH=$RETRIEVAL_HASH,SOURCE_SCHEMA_HASH=$SOURCE_SCHEMA_HASH,RUN_MANIFEST=$RUN_MANIFEST"
log_event full_dpo_submit train_gpus="$SELECTED_TRAIN_GPUS" output="$OUTPUT" fallback_reason=none
sbatch --wait --account="$TURING_ACCOUNT" --nodes=1 --ntasks=1 --gres="gpu:$SELECTED_TRAIN_GPUS" --export="$TRAIN_EXPORT" "$PROJECT_DIR/scripts/turing_train.sh"
log_event full_dpo_complete output="$OUTPUT" fallback_reason=none
