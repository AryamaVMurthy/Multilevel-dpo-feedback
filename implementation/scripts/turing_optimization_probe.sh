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
: "${DATA:?DATA must be supplied with --export}"
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

module load u22/cuda/12.4
cd "$PROJECT_DIR"
[[ -f pyproject.toml && -d src/text_feedback_dpo ]] || fail "invalid PROJECT_DIR" invalid_project_root
mkdir -p "$OUTPUT_ROOT" logs

REPORT="$OUTPUT_ROOT/optimization-probe.jsonl"
BASELINE_RESULT="$OUTPUT_ROOT/baseline-sdpa.json"
DECISION="$OUTPUT_ROOT/optimization-decision.json"
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
CANDIDATES=()

COMMON_ARGS=(
  --commit-hash "$(git rev-parse HEAD)" --config "$CONFIG" --data "$DATA" --model "$MODEL" --model-revision "$MODEL_REVISION"
  --dataset-source "$DATASET_SOURCE" --dataset-revision "$DATASET_REVISION"
  --prompt-sha256 "$PROMPT_HASH" --retrieval-sha256 "$RETRIEVAL_HASH" --source-schema-sha256 "$SOURCE_SCHEMA_HASH"
  --sample-size "$PROBE_SAMPLE_SIZE" --warmup-repeats "$PROBE_WARMUP_REPEATS" --measured-repeats "$PROBE_MEASURED_REPEATS"
)
run_probe() {
  local name="$1" result="$2"; shift 2
  log_event probe_start probe_name="$name" fallback_reason=none
  "$PROBE_RUNNER" benchmark --probe-name "$name" --result "$result" "${COMMON_ARGS[@]}" "$@"
  if "$PROBE_RUNNER" compare --baseline "$BASELINE_RESULT" --candidate "$result" >> "$REPORT"; then
    log_event probe_accepted probe_name="$name" fallback_reason=none
  else
    log_event probe_rejected probe_name="$name" fallback_reason=parity_compatibility_or_throughput_gate
  fi
  CANDIDATES+=(--candidate "$result")
}

# The measured SDPA baseline is always first and is the only implicit safe decision.
log_event probe_start probe_name=baseline-sdpa fallback_reason=none
"$PROBE_RUNNER" benchmark --probe-name baseline-sdpa --result "$BASELINE_RESULT" "${COMMON_ARGS[@]}" --attention-implementation sdpa --generation-batch-size 4
printf '{"probe_name":"baseline-sdpa","status":"baseline","accepted":true,"fallback_reason":"none"}\n' >> "$REPORT"

FA2_RESULT="$OUTPUT_ROOT/attention-flash_attention_2.json"
run_probe attention-flash_attention_2 "$FA2_RESULT" --attention-implementation flash_attention_2 --generation-batch-size 4
FA2_PARITY=false
if "$PROBE_RUNNER" compare --baseline "$BASELINE_RESULT" --candidate "$FA2_RESULT" >/dev/null; then FA2_PARITY=true; fi

for value in $GENERATION_BATCH_SIZES; do run_probe "generation-batch-$value" "$OUTPUT_ROOT/generation-batch-$value.json" --generation-batch-size "$value"; done
for value in $STATIC_CACHE; do args=(); [[ "$value" == true ]] && args+=(--static-cache); run_probe "static-cache-$value" "$OUTPUT_ROOT/static-cache-$value.json" "${args[@]}"; done
for value in $COMPILE; do args=(); [[ "$value" == true ]] && args+=(--compile); run_probe "compile-$value" "$OUTPUT_ROOT/compile-$value.json" "${args[@]}"; done
for value in $TRAIN_MICROBATCHES; do run_probe "train-microbatch-$value" "$OUTPUT_ROOT/train-microbatch-$value.json" --probe-kind training --train-microbatch "$value"; done
for value in $GRADIENT_ACCUMULATION_STEPS; do run_probe "grad-accum-$value" "$OUTPUT_ROOT/grad-accum-$value.json" --probe-kind training --gradient-accumulation-steps "$value"; done
for value in $DATALOADER_WORKERS; do run_probe "workers-$value" "$OUTPUT_ROOT/workers-$value.json" --probe-kind training --dataloader-workers "$value"; done

if [[ "$FA2_PARITY" == true ]]; then
  for value in $PACKING; do args=(--probe-kind training --attention-implementation flash_attention_2); [[ "$value" == true ]] && args+=(--packing); run_probe "packing-$value" "$OUTPUT_ROOT/packing-$value.json" "${args[@]}"; done
  for value in $PADDING_FREE; do args=(--probe-kind training --attention-implementation flash_attention_2); [[ "$value" == true ]] && args+=(--padding-free); run_probe "padding-free-$value" "$OUTPUT_ROOT/padding-free-$value.json" "${args[@]}"; done
else
  printf '{"probe_name":"packing-padding-free","status":"rejected","accepted":false,"fallback_reason":"flash_attention_2_compatibility_or_parity_failed"}\n' >> "$REPORT"
fi

for value in $LIGER; do
  if [[ "$value" == true ]]; then
    result="$OUTPUT_ROOT/liger-true.json"
    run_probe liger-true "$result" --probe-kind training --use-liger-kernel
  fi
done

"$PROBE_RUNNER" freeze-decision --baseline "$BASELINE_RESULT" "${CANDIDATES[@]}" --output "$DECISION" \
  --query-max-new-tokens "$QUERY_MAX_NEW_TOKENS" --response-max-new-tokens "$RESPONSE_MAX_NEW_TOKENS" \
  --student-thinking-mode "$STUDENT_THINKING_MODE" --scratchpad-max-new-tokens "$SCRATCHPAD_MAX_NEW_TOKENS" \
  --query-temperature "$QUERY_TEMPERATURE" --response-temperature "$RESPONSE_TEMPERATURE" --top-p "$TOP_P" \
  --top-k "$TOP_K" --k1 "$BM25_K1" --b "$BM25_B"
DECISION_SHA256="$(sha256sum "$DECISION" | awk '{print $1}')"
log_event probe_complete report="$REPORT" decision="$DECISION" decision_sha256="$DECISION_SHA256" fallback_reason=none
