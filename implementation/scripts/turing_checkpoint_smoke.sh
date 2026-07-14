#!/bin/bash
# Bounded real save/resume smoke. Requires Task 7 train CLI --max-steps support.
#SBATCH -p u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=4096
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail

COMPONENT=turing_checkpoint_smoke
log_event() { local event="$1"; shift; printf 'event=%s timestamp=%s component=%s %s\n' "$event" "$(date -u +%FT%TZ)" "$COMPONENT" "$*"; }
fail() { log_event failure reason="$1" fallback_reason="${2:-none}" >&2; exit 2; }
allocated_gpu_count() { local raw="${SLURM_GPUS_ON_NODE:?SLURM_GPUS_ON_NODE is required}"; if [[ "$raw" =~ ^[0-9]+$ ]]; then printf '%s\n' "$raw"; elif [[ "$raw" =~ ^gpu:([0-9]+)$ ]]; then printf '%s\n' "${BASH_REMATCH[1]}"; else fail "unsupported SLURM_GPUS_ON_NODE format: $raw" gpu_count_parse_unsupported; fi; }

: "${PROJECT_DIR:?PROJECT_DIR must be supplied}"
: "${EXPECTED_COMMIT_HASH:?EXPECTED_COMMIT_HASH must pin the source commit}"
: "${METHOD:?METHOD must be sft, dpo, grpo, or dapo}"
: "${CONFIG:?CONFIG must be supplied}"
: "${TRAIN:?TRAIN must be supplied}"
: "${EVAL:?EVAL must be the explicit evaluation dataset for checkpoint smoke}"
: "${SMOKE_ROOT:?SMOKE_ROOT must be a new output directory}"
: "${SMOKE_MANIFEST:?SMOKE_MANIFEST must be supplied}"
: "${START_MODEL:?START_MODEL must be supplied}"
: "${START_REVISION:?START_REVISION must be supplied}"
: "${START_MODEL_SHA256:?START_MODEL_SHA256 must hash START_MODEL/model.safetensors}"
: "${DATASET_SOURCE:?DATASET_SOURCE must be supplied}"
: "${DATASET_REVISION:?DATASET_REVISION must be supplied}"
: "${PROMPT_HASH:?PROMPT_HASH must be supplied}"
: "${RETRIEVAL_HASH:?RETRIEVAL_HASH must be supplied}"
: "${SOURCE_SCHEMA_HASH:?SOURCE_SCHEMA_HASH must be supplied}"
: "${OPTIMIZATION_DECISION:?OPTIMIZATION_DECISION must be supplied}"
: "${OPTIMIZATION_DECISION_SHA256:?OPTIMIZATION_DECISION_SHA256 must be supplied}"
: "${SCALE_DECISION:?SCALE_DECISION must be supplied}"
: "${SCALE_DECISION_SHA256:?SCALE_DECISION_SHA256 must be supplied}"
: "${TRAIN_GPUS:?TRAIN_GPUS must be supplied}"
: "${INITIAL_MAX_STEPS:?INITIAL_MAX_STEPS must be supplied}"
: "${RESUMED_MAX_STEPS:?RESUMED_MAX_STEPS must be supplied}"
: "${LEARNING_RATE:?LEARNING_RATE must be supplied}"
if [[ "$METHOD" == dpo ]]; then
  : "${REF_LOG_PROBS:?REF_LOG_PROBS must be supplied for DPO smoke}"
  : "${EVAL_REF_LOG_PROBS:?EVAL_REF_LOG_PROBS must be supplied for DPO smoke}"
  : "${REFERENCE_CHECKPOINT_HASH:?REFERENCE_CHECKPOINT_HASH must be supplied for DPO smoke}"
  : "${PROMPT_CONTEXT_SCHEMA:?PROMPT_CONTEXT_SCHEMA must be supplied for DPO smoke}"
fi
: "${SLURM_NNODES:?SLURM_NNODES is required}"
[[ "$SLURM_NNODES" == 1 ]] || fail "checkpoint smoke requires exactly one node" multi_node_checkpoint_smoke
: "${SLURM_NTASKS:?SLURM_NTASKS is required}"
[[ "$SLURM_NTASKS" == 1 ]] || fail "checkpoint smoke requires exactly one task" multi_task_checkpoint_smoke
ALLOCATED_GPU_COUNT="$(allocated_gpu_count)"
[[ "$TRAIN_GPUS" == 4 || "$TRAIN_GPUS" == 8 ]] || fail "checkpoint smoke TRAIN_GPUS must be 4 or 8" checkpoint_smoke_gpu_count
[[ "$ALLOCATED_GPU_COUNT" == "$TRAIN_GPUS" ]] || fail "checkpoint smoke allocation=$ALLOCATED_GPU_COUNT differs from TRAIN_GPUS=$TRAIN_GPUS" checkpoint_smoke_gpu_count
[[ "$INITIAL_MAX_STEPS" =~ ^[1-9][0-9]*$ && "$RESUMED_MAX_STEPS" =~ ^[1-9][0-9]*$ ]] || fail "checkpoint smoke max steps must be positive" max_steps_invalid
[[ "$RESUMED_MAX_STEPS" -gt "$INITIAL_MAX_STEPS" ]] || fail "resume max steps must exceed initial max steps" checkpoint_smoke_step_contract
[[ ! -e "$SMOKE_ROOT" ]] || fail "SMOKE_ROOT already exists; refusing stale checkpoint reuse: $SMOKE_ROOT" checkpoint_smoke_root_exists

module load u22/cuda/12.4
cd "$PROJECT_DIR"
CURRENT_COMMIT_HASH="$(git rev-parse HEAD)"
[[ "$CURRENT_COMMIT_HASH" == "$EXPECTED_COMMIT_HASH" ]] || fail "current commit=$CURRENT_COMMIT_HASH expected=$EXPECTED_COMMIT_HASH" checkpoint_smoke_commit_mismatch
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_TF32_CUBLAS_OVERRIDE=1
PROBE_RUNNER="$PROJECT_DIR/scripts/turing_probe_runner.py"
run_probe_runner() { uv run --frozen python "$PROBE_RUNNER" "$@"; }
CONFIG_SHA256="$(sha256sum "$CONFIG" | awk '{print $1}')"
DATASET_SHA256="$(sha256sum "$TRAIN" | awk '{print $1}')"
EVAL_DATASET_SHA256="$(sha256sum "$EVAL" | awk '{print $1}')"
IFS=$'\t' read -r ATTENTION_IMPLEMENTATION DECISION_MICROBATCH DECISION_GRADIENT_ACCUMULATION_STEPS DECISION_DATALOADER_WORKERS DECISION_EVAL_MICROBATCH DECISION_FULL_MAX_STEPS DECISION_MAX_LENGTH DECISION_GRADIENT_CHECKPOINTING DECISION_PACKING DECISION_PADDING_FREE DECISION_USE_LIGER DECISION_LEARNING_RATE DECISION_EPOCHS DECISION_SAVE_STEPS DECISION_EVAL_STEPS DECISION_NUM_GENERATIONS DECISION_GENERATION_BATCH_SIZE DECISION_MAX_COMPLETION_LENGTH ATTENTION_FALLBACK_REASON VALIDATED_DECISION_SHA256 < <(run_probe_runner validate-decision --decision "$OPTIMIZATION_DECISION" --expected-sha256 "$OPTIMIZATION_DECISION_SHA256" \
  --purpose training --training-method "$METHOD" --commit-hash "$CURRENT_COMMIT_HASH" --config-sha256 "$CONFIG_SHA256" --model "$START_MODEL" --model-revision "$START_REVISION" \
  --dataset-source "$DATASET_SOURCE" --dataset-revision "$DATASET_REVISION" --dataset-sha256 "$DATASET_SHA256" \
  --eval-dataset-sha256 "$EVAL_DATASET_SHA256" \
  --prompt-sha256 "$PROMPT_HASH" --retrieval-sha256 "$RETRIEVAL_HASH" --source-schema-sha256 "$SOURCE_SCHEMA_HASH" --output-format training-tsv) \
  || fail "frozen training decision validation failed" optimization_decision_invalid
awk -v expected="$LEARNING_RATE" -v frozen="$DECISION_LEARNING_RATE" 'BEGIN { exit !(expected == frozen) }' || fail "smoke learning rate differs from frozen decision" training_control_mismatch
run_probe_runner validate-scale-decision --decision "$SCALE_DECISION" --expected-sha256 "$SCALE_DECISION_SHA256" --train-gpus "$TRAIN_GPUS" --training-method "$METHOD" \
  --commit-hash "$CURRENT_COMMIT_HASH" --config-sha256 "$CONFIG_SHA256" --model "$START_MODEL" --model-revision "$START_REVISION" \
  --dataset-source "$DATASET_SOURCE" --dataset-revision "$DATASET_REVISION" --dataset-sha256 "$DATASET_SHA256" \
  --prompt-sha256 "$PROMPT_HASH" --retrieval-sha256 "$RETRIEVAL_HASH" --source-schema-sha256 "$SOURCE_SCHEMA_HASH" >/dev/null \
  || fail "frozen scaling decision validation failed" scale_decision_invalid

TRAIN_HELP="$(uv run --frozen python -m text_feedback_dpo.cli "train-$METHOD" --help)" || fail "cannot inspect Task 7 train-$METHOD CLI" task7_train_cli_help_failed
for required_flag in --eval --max-steps --max-length --dataloader-num-workers --per-device-train-batch-size --per-device-eval-batch-size --gradient-accumulation-steps --attention-implementation --gradient-checkpointing --packing --padding-free --use-liger-kernel; do
  [[ "$TRAIN_HELP" == *"$required_flag"* ]] || fail "Task 7 train-$METHOD CLI does not expose $required_flag; bounded smoke cannot run" task7_checkpoint_smoke_cli_missing
done

mkdir -p "$SMOKE_ROOT"
COMMON_ARGS=(
  --config "$CONFIG" --train "$TRAIN" --eval "$EVAL" --output "$SMOKE_ROOT" --model "$START_MODEL" --model-revision "$START_REVISION"
  --initial-checkpoint-sha256 "$START_MODEL_SHA256"
  --deepspeed-config configs/deepspeed_zero3.json --save-steps 1 --eval-steps 1
  --gradient-accumulation-steps "$DECISION_GRADIENT_ACCUMULATION_STEPS" --learning-rate "$DECISION_LEARNING_RATE"
  --per-device-train-batch-size "$DECISION_MICROBATCH" --per-device-eval-batch-size "$DECISION_EVAL_MICROBATCH"
  --dataloader-num-workers "$DECISION_DATALOADER_WORKERS" --max-length "$DECISION_MAX_LENGTH" --attention-implementation "$ATTENTION_IMPLEMENTATION"
  --num-generations "$DECISION_NUM_GENERATIONS" --generation-batch-size "$DECISION_GENERATION_BATCH_SIZE" --max-completion-length "$DECISION_MAX_COMPLETION_LENGTH"
)
if [[ "$DECISION_GRADIENT_CHECKPOINTING" == True ]]; then COMMON_ARGS+=(--gradient-checkpointing); else COMMON_ARGS+=(--no-gradient-checkpointing); fi
if [[ "$DECISION_PACKING" == True ]]; then COMMON_ARGS+=(--packing); else COMMON_ARGS+=(--no-packing); fi
if [[ "$DECISION_PADDING_FREE" == True ]]; then COMMON_ARGS+=(--padding-free); else COMMON_ARGS+=(--no-padding-free); fi
if [[ "$DECISION_USE_LIGER" == True ]]; then COMMON_ARGS+=(--use-liger-kernel); else COMMON_ARGS+=(--no-use-liger-kernel); fi
if [[ "$METHOD" == dpo ]]; then COMMON_ARGS+=(--ref-log-probs "$REF_LOG_PROBS" --eval-ref-log-probs "$EVAL_REF_LOG_PROBS" --reference-checkpoint-hash "$REFERENCE_CHECKPOINT_HASH" --prompt-context-schema "$PROMPT_CONTEXT_SCHEMA"); fi
log_event checkpoint_save_smoke_start max_steps="$INITIAL_MAX_STEPS" fallback_reason=none
uv run --frozen python -m torch.distributed.run --standalone --nproc_per_node="$ALLOCATED_GPU_COUNT" -m text_feedback_dpo.cli "train-$METHOD" "${COMMON_ARGS[@]}" --max-steps "$INITIAL_MAX_STEPS"
INITIAL_CHECKPOINT="$SMOKE_ROOT/checkpoint-$INITIAL_MAX_STEPS"
[[ -d "$INITIAL_CHECKPOINT" ]] || fail "initial checkpoint missing: $INITIAL_CHECKPOINT" checkpoint_save_missing

log_event checkpoint_resume_smoke_start max_steps="$RESUMED_MAX_STEPS" resume_from="$INITIAL_CHECKPOINT" fallback_reason=none
uv run --frozen python -m torch.distributed.run --standalone --nproc_per_node="$ALLOCATED_GPU_COUNT" -m text_feedback_dpo.cli "train-$METHOD" "${COMMON_ARGS[@]}" --max-steps "$RESUMED_MAX_STEPS" --resume-from-checkpoint "$INITIAL_CHECKPOINT"
RESUMED_CHECKPOINT="$SMOKE_ROOT/checkpoint-$RESUMED_MAX_STEPS"
[[ -d "$RESUMED_CHECKPOINT" ]] || fail "resumed checkpoint missing: $RESUMED_CHECKPOINT" checkpoint_resume_missing

run_probe_runner create-smoke-manifest --initial-checkpoint "$INITIAL_CHECKPOINT" --resumed-checkpoint "$RESUMED_CHECKPOINT" \
  --output "$SMOKE_MANIFEST" --commit-hash "$CURRENT_COMMIT_HASH" --config-sha256 "$CONFIG_SHA256" --model "$START_MODEL" \
  --model-revision "$START_REVISION" --dataset-source "$DATASET_SOURCE" --dataset-revision "$DATASET_REVISION" --dataset-sha256 "$DATASET_SHA256" --eval-dataset-sha256 "$EVAL_DATASET_SHA256" --prompt-sha256 "$PROMPT_HASH" \
  --retrieval-sha256 "$RETRIEVAL_HASH" --source-schema-sha256 "$SOURCE_SCHEMA_HASH" \
  --optimization-decision-sha256 "$OPTIMIZATION_DECISION_SHA256" --scale-decision-sha256 "$SCALE_DECISION_SHA256" --method "$METHOD"
SMOKE_MANIFEST_SHA256="$(sha256sum "$SMOKE_MANIFEST" | awk '{print $1}')"
log_event checkpoint_smoke_complete manifest="$SMOKE_MANIFEST" manifest_sha256="$SMOKE_MANIFEST_SHA256" fallback_reason=none
