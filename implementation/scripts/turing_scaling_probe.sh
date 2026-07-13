#!/bin/bash
# Submit twice with `--gres=gpu:4` and `--gres=gpu:8`; freeze only after both measured artifacts exist.
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

COMPONENT=turing_scaling_probe
log_event() { local event="$1"; shift; printf 'event=%s timestamp=%s component=%s %s\n' "$event" "$(date -u +%FT%TZ)" "$COMPONENT" "$*"; }
fail() { log_event failure reason="$1" fallback_reason="${2:-none}" >&2; exit 2; }
allocated_gpu_count() { local raw="${SLURM_GPUS_ON_NODE:?SLURM_GPUS_ON_NODE is required}"; if [[ "$raw" =~ ^[0-9]+$ ]]; then printf '%s\n' "$raw"; elif [[ "$raw" =~ ^gpu:([0-9]+)$ ]]; then printf '%s\n' "${BASH_REMATCH[1]}"; else fail "unsupported GPU allocation: $raw" gpu_count_parse_unsupported; fi; }

required=(PROJECT_DIR TRAIN_GPUS TRAINING_METHOD CONFIG TRAIN EVAL MODEL MODEL_REVISION DATASET_SOURCE DATASET_REVISION PROMPT_HASH RETRIEVAL_HASH SOURCE_SCHEMA_HASH DEEPSPEED_CONFIG OUTPUT_RESULT PROBE_MAX_STEPS TRAIN_MICROBATCH GRADIENT_ACCUMULATION_STEPS DATALOADER_NUM_WORKERS PER_DEVICE_EVAL_BATCH_SIZE NUM_GENERATIONS RL_GENERATION_BATCH_SIZE MAX_COMPLETION_LENGTH)
for name in "${required[@]}"; do [[ -n "${!name:-}" ]] || fail "$name must be supplied explicitly" scaling_probe_contract_missing; done
: "${SLURM_NNODES:?SLURM_NNODES is required}"; [[ "$SLURM_NNODES" == 1 ]] || fail "scaling probe requires one node" multi_node_probe_forbidden
: "${SLURM_NTASKS:?SLURM_NTASKS is required}"; [[ "$SLURM_NTASKS" == 1 ]] || fail "scaling probe requires one task" multi_task_probe_forbidden
[[ "$TRAIN_GPUS" == 4 || "$TRAIN_GPUS" == 8 ]] || fail "TRAIN_GPUS must be 4 or 8" scale_gpu_counts_invalid
ALLOCATED_GPU_COUNT="$(allocated_gpu_count)"
[[ "$ALLOCATED_GPU_COUNT" == "$TRAIN_GPUS" ]] || fail "allocation=$ALLOCATED_GPU_COUNT differs from TRAIN_GPUS=$TRAIN_GPUS" gpu_allocation_mismatch

module load u22/cuda/12.4
cd "$PROJECT_DIR"
[[ -f pyproject.toml && -x scripts/turing_probe_runner.py ]] || fail "invalid PROJECT_DIR" invalid_project_root
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
PROBE_RUNNER="$PROJECT_DIR/scripts/turing_probe_runner.py"
run_probe_runner() { uv run --frozen python "$PROBE_RUNNER" "$@"; }
mkdir -p "$(dirname "$OUTPUT_RESULT")" "${PROBE_OUTPUT_DIR:?PROBE_OUTPUT_DIR must be supplied}"
METHOD_ARGS=(--training-method "$TRAINING_METHOD")
if [[ "$TRAINING_METHOD" == dpo ]]; then
  METHOD_ARGS+=(--ref-log-probs "${REF_LOG_PROBS:?REF_LOG_PROBS is required for DPO scaling}" \
    --eval-ref-log-probs "${EVAL_REF_LOG_PROBS:?EVAL_REF_LOG_PROBS is required for DPO scaling}" \
    --reference-checkpoint-hash "${REFERENCE_CHECKPOINT_HASH:?REFERENCE_CHECKPOINT_HASH is required for DPO scaling}" \
    --prompt-context-schema "${PROMPT_CONTEXT_SCHEMA:?PROMPT_CONTEXT_SCHEMA is required for DPO scaling}")
fi

log_event scale_measure_start train_gpus="$TRAIN_GPUS" fallback_reason=none
run_probe_runner benchmark --probe-kind training --probe-name "a100-scale-$TRAIN_GPUS" --result "$OUTPUT_RESULT" \
  --commit-hash "$(git rev-parse HEAD)" --config "$CONFIG" --data "$TRAIN" --eval-data "$EVAL" --output-dir "$PROBE_OUTPUT_DIR" \
  --deepspeed-config "$DEEPSPEED_CONFIG" --model "$MODEL" --model-revision "$MODEL_REVISION" \
  --dataset-source "$DATASET_SOURCE" --dataset-revision "$DATASET_REVISION" --prompt-sha256 "$PROMPT_HASH" \
  --retrieval-sha256 "$RETRIEVAL_HASH" --source-schema-sha256 "$SOURCE_SCHEMA_HASH" --attention-implementation sdpa \
  --max-steps "$PROBE_MAX_STEPS" --max-length 4096 --train-microbatch "$TRAIN_MICROBATCH" \
  --per-device-eval-batch-size "$PER_DEVICE_EVAL_BATCH_SIZE" --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --dataloader-workers "$DATALOADER_NUM_WORKERS" --gradient-checkpointing \
  --num-generations "$NUM_GENERATIONS" --rl-generation-batch-size "$RL_GENERATION_BATCH_SIZE" --max-completion-length "$MAX_COMPLETION_LENGTH" "${METHOD_ARGS[@]}"
log_event scale_measure_complete result="$OUTPUT_RESULT" result_sha256="$(sha256sum "$OUTPUT_RESULT" | awk '{print $1}')" train_gpus="$TRAIN_GPUS" fallback_reason=none

if [[ "${FREEZE_SCALE_DECISION:-false}" == true ]]; then
  : "${FOUR_GPU_RESULT:?FOUR_GPU_RESULT must be supplied to freeze}"
  : "${EIGHT_GPU_RESULT:?EIGHT_GPU_RESULT must be supplied to freeze}"
  : "${SCALE_DECISION_OUTPUT:?SCALE_DECISION_OUTPUT must be supplied to freeze}"
  run_probe_runner freeze-scale-decision --result "$FOUR_GPU_RESULT" --result "$EIGHT_GPU_RESULT" --output "$SCALE_DECISION_OUTPUT"
  log_event scale_decision_frozen decision="$SCALE_DECISION_OUTPUT" decision_sha256="$(sha256sum "$SCALE_DECISION_OUTPUT" | awk '{print $1}')" fallback_reason=none
fi
