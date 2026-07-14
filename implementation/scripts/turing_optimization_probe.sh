#!/bin/bash
# Real measured optimization probes; never mutates primary launch settings.
#SBATCH -p u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4096
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail

COMPONENT=turing_optimization_probe
log_event() { local event="$1"; shift; printf 'event=%s timestamp=%s component=%s %s\n' "$event" "$(date -u +%FT%TZ)" "$COMPONENT" "$*"; }
fail() { log_event failure reason="$1" fallback_reason="${2:-none}" >&2; exit 2; }
allocated_gpu_count() { local raw="${SLURM_GPUS_ON_NODE:?SLURM_GPUS_ON_NODE is required}"; if [[ "$raw" =~ ^[0-9]+$ ]]; then printf '%s\n' "$raw"; elif [[ "$raw" =~ ^gpu:([0-9]+)$ ]]; then printf '%s\n' "${BASH_REMATCH[1]}"; else fail "unsupported SLURM_GPUS_ON_NODE format: $raw" gpu_count_parse_unsupported; fi; }

: "${TURING_ACCOUNT:?TURING_ACCOUNT must be supplied with --export}"
: "${PROJECT_DIR:?PROJECT_DIR must be supplied with --export}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT must be supplied with --export}"
: "${CONFIG:?CONFIG must be supplied with --export}"
: "${MODEL:?MODEL must be supplied with --export}"
: "${MODEL_REVISION:?MODEL_REVISION must be supplied with --export}"
: "${START_MODEL_SHA256:?START_MODEL_SHA256 must hash MODEL/model.safetensors}"
: "${DATA:?DATA must be supplied with --export}"
: "${TRAIN_EVAL:?TRAIN_EVAL must be supplied for measured training probes}"
: "${DEEPSPEED_CONFIG:?DEEPSPEED_CONFIG must be supplied for measured training probes}"
: "${PROBE_MAX_STEPS:?PROBE_MAX_STEPS must be supplied}"
: "${LAUNCH_MAX_STEPS:?LAUNCH_MAX_STEPS must be frozen explicitly}"
: "${LEARNING_RATE:?LEARNING_RATE must be frozen explicitly}"
: "${EPOCHS:?EPOCHS must be frozen explicitly}"
: "${SAVE_STEPS:?SAVE_STEPS must be frozen explicitly}"
: "${EVAL_STEPS:?EVAL_STEPS must be frozen explicitly}"
: "${NUM_GENERATIONS:?NUM_GENERATIONS must be frozen explicitly}"
: "${RL_GENERATION_BATCH_SIZE:?RL_GENERATION_BATCH_SIZE must be frozen explicitly}"
: "${MAX_COMPLETION_LENGTH:?MAX_COMPLETION_LENGTH must be frozen explicitly}"
: "${TRAINING_METHOD:?TRAINING_METHOD must be sft, dpo, grpo, or dapo}"
: "${DATASET_SOURCE:?DATASET_SOURCE must be supplied with --export}"
: "${DATASET_REVISION:?DATASET_REVISION must be supplied with --export}"
: "${PROMPT_HASH:?PROMPT_HASH must be supplied with --export}"
: "${RETRIEVAL_HASH:?RETRIEVAL_HASH must be supplied with --export}"
: "${SOURCE_SCHEMA_HASH:?SOURCE_SCHEMA_HASH must be supplied with --export}"
: "${QUERY_MAX_NEW_TOKENS:?QUERY_MAX_NEW_TOKENS must be supplied with --export}"
: "${RESPONSE_MAX_NEW_TOKENS:?RESPONSE_MAX_NEW_TOKENS must be supplied with --export}"
: "${STUDENT_THINKING_MODE:?STUDENT_THINKING_MODE must be supplied with --export}"
: "${SCRATCHPAD_MAX_NEW_TOKENS:?SCRATCHPAD_MAX_NEW_TOKENS must be supplied with --export}"
: "${QUERY_TEMPERATURE:?QUERY_TEMPERATURE must be supplied with --export}"
: "${RESPONSE_TEMPERATURE:?RESPONSE_TEMPERATURE must be supplied with --export}"
: "${TOP_P:?TOP_P must be supplied with --export}"
: "${TOP_K:?TOP_K must be supplied with --export}"
: "${BM25_K1:?BM25_K1 must be supplied with --export}"
: "${BM25_B:?BM25_B must be supplied with --export}"
: "${SLURM_NNODES:?SLURM_NNODES is required}"
[[ "$SLURM_NNODES" == "1" ]] || fail "optimization probe requires exactly one node; got $SLURM_NNODES" multi_node_probe_forbidden
: "${SLURM_NTASKS:?SLURM_NTASKS is required}"
[[ "$SLURM_NTASKS" == "1" ]] || fail "optimization probe requires exactly one task; got $SLURM_NTASKS" multi_task_probe_forbidden
ALLOCATED_GPU_COUNT="$(allocated_gpu_count)"
[[ "$ALLOCATED_GPU_COUNT" == "1" ]] || fail "optimization probe requires exactly one allocated GPU; got $ALLOCATED_GPU_COUNT" probe_gpu_count

DEFAULT_PROBE_RUNNER="$PROJECT_DIR/scripts/turing_probe_runner.py"
if [[ -n "${PROBE_RUNNER:-}" && "$PROBE_RUNNER" != "$DEFAULT_PROBE_RUNNER" && "${ALLOW_PROBE_RUNNER_OVERRIDE:-false}" != "true" ]]; then
  fail "PROBE_RUNNER override requires ALLOW_PROBE_RUNNER_OVERRIDE=true" probe_runner_override_not_authorized
fi
PROBE_RUNNER="${PROBE_RUNNER:-$DEFAULT_PROBE_RUNNER}"
[[ -x "$PROBE_RUNNER" ]] || fail "PROBE_RUNNER is not executable: $PROBE_RUNNER" probe_runner_missing
run_probe_runner() { uv run --frozen python "$PROBE_RUNNER" "$@"; }

module load u22/cuda/12.4
cd "$PROJECT_DIR"
[[ -f pyproject.toml && -d src/text_feedback_dpo ]] || fail "invalid PROJECT_DIR" invalid_project_root
mkdir -p "$OUTPUT_ROOT" logs

REPORT="$OUTPUT_ROOT/optimization-probe.jsonl"
BASELINE_RESULT="$OUTPUT_ROOT/baseline-sdpa.json"
DECISION="$OUTPUT_ROOT/generation-optimization-decision.json"
TRAIN_DECISION="$OUTPUT_ROOT/training-optimization-decision.json"
TRAIN_BASELINE_RESULT="$OUTPUT_ROOT/training-baseline-sdpa.json"
PROBE_SAMPLE_SIZE="${PROBE_SAMPLE_SIZE:-16}"
PROBE_WARMUP_REPEATS="${PROBE_WARMUP_REPEATS:-2}"
PROBE_MEASURED_REPEATS="${PROBE_MEASURED_REPEATS:-5}"
GENERATION_BATCH_SIZES="${GENERATION_BATCH_SIZES:-1 2 4 8 16}"
TRAIN_MICROBATCHES="${TRAIN_MICROBATCHES:-1 2 4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1 2 4 8}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-0 2 4 8}"
STATIC_CACHE="${STATIC_CACHE:-false true}"
COMPILE="${COMPILE:-false true}"
PACKING="${PACKING:-false true}"
PADDING_FREE="${PADDING_FREE:-false true}"
LIGER="${LIGER:-false true}"
: > "$REPORT"
GENERATION_CANDIDATES=()
TRAINING_CANDIDATES=()

COMMON_ARGS=(
  --commit-hash "$(git rev-parse HEAD)" --config "$CONFIG" --data "$DATA" --model "$MODEL" --model-revision "$MODEL_REVISION"
  --initial-checkpoint-sha256 "$START_MODEL_SHA256"
  --dataset-source "$DATASET_SOURCE" --dataset-revision "$DATASET_REVISION"
  --prompt-sha256 "$PROMPT_HASH" --retrieval-sha256 "$RETRIEVAL_HASH" --source-schema-sha256 "$SOURCE_SCHEMA_HASH"
  --sample-size "$PROBE_SAMPLE_SIZE" --warmup-repeats "$PROBE_WARMUP_REPEATS" --measured-repeats "$PROBE_MEASURED_REPEATS"
  --num-generations "$NUM_GENERATIONS" --rl-generation-batch-size "$RL_GENERATION_BATCH_SIZE" --max-completion-length "$MAX_COMPLETION_LENGTH"
  --training-method "$TRAINING_METHOD"
)
if [[ "$TRAINING_METHOD" == dpo ]]; then
  COMMON_ARGS+=(--ref-log-probs "${REF_LOG_PROBS:?REF_LOG_PROBS is required for DPO probes}" \
    --eval-ref-log-probs "${EVAL_REF_LOG_PROBS:?EVAL_REF_LOG_PROBS is required for DPO probes}" \
    --reference-checkpoint-hash "${REFERENCE_CHECKPOINT_HASH:?REFERENCE_CHECKPOINT_HASH is required for DPO probes}" \
    --prompt-context-schema "${PROMPT_CONTEXT_SCHEMA:?PROMPT_CONTEXT_SCHEMA is required for DPO probes}")
fi
run_probe() {
  local name="$1" result="$2"; shift 2
  log_event probe_start probe_name="$name" fallback_reason=none
  run_probe_runner benchmark --probe-name "$name" --result "$result" "${COMMON_ARGS[@]}" "$@"
  if run_probe_runner compare --baseline "$BASELINE_RESULT" --candidate "$result" >> "$REPORT"; then
    log_event probe_accepted probe_name="$name" fallback_reason=none
  else
    log_event probe_rejected probe_name="$name" fallback_reason=parity_compatibility_or_throughput_gate
  fi
  GENERATION_CANDIDATES+=(--candidate "$result")
}
run_training_probe() {
  local name="$1" result="$2"; shift 2
  log_event probe_start probe_name="$name" probe_kind=training fallback_reason=none
  run_probe_runner benchmark --probe-name "$name" --result "$result" "${COMMON_ARGS[@]}" --probe-kind training \
    --eval-data "$TRAIN_EVAL" --output-dir "$OUTPUT_ROOT/runs/$name" --deepspeed-config "$DEEPSPEED_CONFIG" --max-steps "$PROBE_MAX_STEPS" "$@"
  if run_probe_runner compare --baseline "$TRAIN_BASELINE_RESULT" --candidate "$result" >> "$REPORT"; then
    log_event probe_accepted probe_name="$name" fallback_reason=none
  else
    log_event probe_rejected probe_name="$name" fallback_reason=parity_compatibility_or_throughput_gate
  fi
  TRAINING_CANDIDATES+=(--candidate "$result")
}

# The measured SDPA baseline is always first and is the only implicit safe decision.
log_event probe_start probe_name=baseline-sdpa fallback_reason=none
run_probe_runner benchmark --probe-name baseline-sdpa --result "$BASELINE_RESULT" "${COMMON_ARGS[@]}" --attention-implementation sdpa --generation-batch-size 4
printf '{"probe_name":"baseline-sdpa","status":"baseline","accepted":true,"fallback_reason":"none"}\n' >> "$REPORT"

FA2_RESULT="$OUTPUT_ROOT/attention-flash_attention_2.json"
run_probe attention-flash_attention_2 "$FA2_RESULT" --attention-implementation flash_attention_2 --generation-batch-size 4
FA2_PARITY=false
if run_probe_runner compare --baseline "$BASELINE_RESULT" --candidate "$FA2_RESULT" >/dev/null; then FA2_PARITY=true; fi

for value in $GENERATION_BATCH_SIZES; do run_probe "generation-batch-$value" "$OUTPUT_ROOT/generation-batch-$value.json" --generation-batch-size "$value"; done
for value in $STATIC_CACHE; do args=(); [[ "$value" == true ]] && args+=(--static-cache); run_probe "static-cache-$value" "$OUTPUT_ROOT/static-cache-$value.json" "${args[@]}"; done
for value in $COMPILE; do args=(); [[ "$value" == true ]] && args+=(--compile); run_probe "compile-$value" "$OUTPUT_ROOT/compile-$value.json" "${args[@]}"; done
run_probe_runner benchmark --probe-name training-baseline-sdpa --result "$TRAIN_BASELINE_RESULT" "${COMMON_ARGS[@]}" --probe-kind training \
  --eval-data "$TRAIN_EVAL" --output-dir "$OUTPUT_ROOT/runs/training-baseline-sdpa" --deepspeed-config "$DEEPSPEED_CONFIG" --max-steps "$PROBE_MAX_STEPS" --attention-implementation sdpa
for value in $TRAIN_MICROBATCHES; do run_training_probe "train-microbatch-$value" "$OUTPUT_ROOT/train-microbatch-$value.json" --train-microbatch "$value"; done
for value in $GRADIENT_ACCUMULATION_STEPS; do run_training_probe "grad-accum-$value" "$OUTPUT_ROOT/grad-accum-$value.json" --gradient-accumulation-steps "$value"; done
for value in $DATALOADER_WORKERS; do run_training_probe "workers-$value" "$OUTPUT_ROOT/workers-$value.json" --dataloader-workers "$value"; done

if [[ "$FA2_PARITY" == true ]]; then
  for value in $PACKING; do args=(--attention-implementation flash_attention_2); [[ "$value" == true ]] && args+=(--packing); run_training_probe "packing-$value" "$OUTPUT_ROOT/packing-$value.json" "${args[@]}"; done
  for value in $PADDING_FREE; do args=(--attention-implementation flash_attention_2); [[ "$value" == true ]] && args+=(--padding-free); run_training_probe "padding-free-$value" "$OUTPUT_ROOT/padding-free-$value.json" "${args[@]}"; done
else
  printf '{"probe_name":"packing-padding-free","status":"rejected","accepted":false,"fallback_reason":"flash_attention_2_compatibility_or_parity_failed"}\n' >> "$REPORT"
fi

for value in $LIGER; do
  if [[ "$value" == true ]]; then
    result="$OUTPUT_ROOT/liger-true.json"
    run_training_probe liger-true "$result" --use-liger-kernel
  fi
done

run_probe_runner freeze-decision --purpose generation --baseline "$BASELINE_RESULT" "${GENERATION_CANDIDATES[@]}" --output "$DECISION" --launch-max-steps "$LAUNCH_MAX_STEPS" \
  --launch-learning-rate "$LEARNING_RATE" --launch-epochs "$EPOCHS" --launch-save-steps "$SAVE_STEPS" --launch-eval-steps "$EVAL_STEPS" \
  --query-max-new-tokens "$QUERY_MAX_NEW_TOKENS" --response-max-new-tokens "$RESPONSE_MAX_NEW_TOKENS" \
  --student-thinking-mode "$STUDENT_THINKING_MODE" --scratchpad-max-new-tokens "$SCRATCHPAD_MAX_NEW_TOKENS" \
  --query-temperature "$QUERY_TEMPERATURE" --response-temperature "$RESPONSE_TEMPERATURE" --top-p "$TOP_P" \
  --top-k "$TOP_K" --k1 "$BM25_K1" --b "$BM25_B"
run_probe_runner freeze-decision --purpose training --baseline "$TRAIN_BASELINE_RESULT" "${TRAINING_CANDIDATES[@]}" --output "$TRAIN_DECISION" --launch-max-steps "$LAUNCH_MAX_STEPS" \
  --launch-learning-rate "$LEARNING_RATE" --launch-epochs "$EPOCHS" --launch-save-steps "$SAVE_STEPS" --launch-eval-steps "$EVAL_STEPS" \
  --query-max-new-tokens "$QUERY_MAX_NEW_TOKENS" --response-max-new-tokens "$RESPONSE_MAX_NEW_TOKENS" \
  --student-thinking-mode "$STUDENT_THINKING_MODE" --scratchpad-max-new-tokens "$SCRATCHPAD_MAX_NEW_TOKENS" \
  --query-temperature "$QUERY_TEMPERATURE" --response-temperature "$RESPONSE_TEMPERATURE" --top-p "$TOP_P" \
  --top-k "$TOP_K" --k1 "$BM25_K1" --b "$BM25_B"
DECISION_SHA256="$(sha256sum "$DECISION" | awk '{print $1}')"
TRAIN_DECISION_SHA256="$(sha256sum "$TRAIN_DECISION" | awk '{print $1}')"
log_event probe_complete report="$REPORT" generation_decision="$DECISION" generation_decision_sha256="$DECISION_SHA256" training_decision="$TRAIN_DECISION" training_decision_sha256="$TRAIN_DECISION_SHA256" fallback_reason=none
