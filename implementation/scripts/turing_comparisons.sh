#!/bin/bash
# Matched SFT/GRPO/DAPO training plus frozen base/DPO validation comparison on one four-GPU node.
# Untouched test evaluation is reserved for turing_finalize_report.sh after promotion.
#SBATCH -p u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=4096
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail

COMPONENT=turing_comparisons
log_event() { local event="$1"; shift; printf 'event=%s timestamp=%s component=%s %s\n' "$event" "$(date -u +%FT%TZ)" "$COMPONENT" "$*"; }
fail() { log_event failure reason="$1" fallback_reason="${2:-none}" >&2; exit 2; }
hash_file() { sha256sum "$1" | awk '{print $1}'; }
allocated_gpu_count() { local raw="${SLURM_GPUS_ON_NODE:?SLURM_GPUS_ON_NODE is required}"; if [[ "$raw" =~ ^[0-9]+$ ]]; then printf '%s\n' "$raw"; elif [[ "$raw" =~ ^gpu:([0-9]+)$ ]]; then printf '%s\n' "${BASH_REMATCH[1]}"; else fail "unsupported SLURM_GPUS_ON_NODE format: $raw" gpu_count_parse_unsupported; fi; }

required=(TURING_ACCOUNT PROJECT_DIR CONFIG BASE_MODEL BASE_REVISION DPO_MODEL DPO_OUTPUT_REVISION RL_START_MODEL RL_START_REVISION SFT_TRAIN SFT_EVAL RL_DATA RL_EVAL VAL_DATA OUTPUT_ROOT TRAIN_GPUS DATASET_SOURCE DATASET_REVISION PROMPT_VERSION PROMPT_HASH RETRIEVAL_HASH SOURCE_SCHEMA_HASH POLICY_HASH LEARNING_RATE EPOCHS SAVE_STEPS EVAL_STEPS
  SFT_TRAIN_DECISION SFT_TRAIN_DECISION_SHA256 GRPO_TRAIN_DECISION GRPO_TRAIN_DECISION_SHA256 DAPO_TRAIN_DECISION DAPO_TRAIN_DECISION_SHA256
  SFT_SCALE_DECISION SFT_SCALE_DECISION_SHA256 GRPO_SCALE_DECISION GRPO_SCALE_DECISION_SHA256 DAPO_SCALE_DECISION DAPO_SCALE_DECISION_SHA256
  BASE_GENERATION_DECISION BASE_GENERATION_DECISION_SHA256 DPO_GENERATION_DECISION DPO_GENERATION_DECISION_SHA256 SFT_GENERATION_DECISION SFT_GENERATION_DECISION_SHA256 GRPO_GENERATION_DECISION GRPO_GENERATION_DECISION_SHA256 DAPO_GENERATION_DECISION DAPO_GENERATION_DECISION_SHA256
  SFT_SMOKE_MANIFEST SFT_SMOKE_MANIFEST_SHA256 GRPO_SMOKE_MANIFEST GRPO_SMOKE_MANIFEST_SHA256 DAPO_SMOKE_MANIFEST DAPO_SMOKE_MANIFEST_SHA256
  SFT_OUTPUT_REVISION GRPO_OUTPUT_REVISION DAPO_OUTPUT_REVISION STUDENT_THINKING_MODE SCRATCHPAD_MAX_NEW_TOKENS QUERY_TEMPERATURE RESPONSE_TEMPERATURE TOP_P TOP_K BM25_K1 BM25_B)
for name in "${required[@]}"; do [[ -n "${!name:-}" ]] || fail "$name must be supplied explicitly" comparison_contract_missing; done
: "${SLURM_NNODES:?SLURM_NNODES is required}"; [[ "$SLURM_NNODES" == 1 ]] || fail "local torchrun requires one node; got $SLURM_NNODES" multi_node_training_forbidden
: "${SLURM_NTASKS:?SLURM_NTASKS is required}"; [[ "$SLURM_NTASKS" == 1 ]] || fail "local torchrun requires one task; got $SLURM_NTASKS" multi_task_training_forbidden
ALLOCATED_GPU_COUNT="$(allocated_gpu_count)"
[[ "$TRAIN_GPUS" == 4 || "$TRAIN_GPUS" == 8 ]] || fail "comparison TRAIN_GPUS must be frozen to 4 or 8" gpu_allocation_mismatch
[[ "$ALLOCATED_GPU_COUNT" == "$TRAIN_GPUS" ]] || fail "comparison allocation=$ALLOCATED_GPU_COUNT differs from TRAIN_GPUS=$TRAIN_GPUS" gpu_allocation_mismatch

module load u22/cuda/12.4
cd "$PROJECT_DIR"
[[ -f pyproject.toml && -d src/text_feedback_dpo ]] || fail "invalid PROJECT_DIR" invalid_project_root
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 UV_LINK_MODE=copy PYTORCH_TF32_CUBLAS_OVERRIDE=1
PROBE_RUNNER="$PROJECT_DIR/scripts/turing_probe_runner.py"; [[ -x "$PROBE_RUNNER" ]] || fail "probe runner missing" probe_runner_missing
run_probe_runner() { uv run --frozen python "$PROBE_RUNNER" "$@"; }
COMMIT_HASH="$(git rev-parse HEAD)"; CONFIG_HASH="$(hash_file "$CONFIG")"; mkdir -p "$OUTPUT_ROOT" logs
MODEL_HASH="$(printf '%s' "$BASE_MODEL@$BASE_REVISION|$DPO_MODEL@$DPO_OUTPUT_REVISION|$RL_START_MODEL@$RL_START_REVISION" | sha256sum | awk '{print $1}')"
DATASET_HASH="$(printf '%s' "$(hash_file "$SFT_TRAIN")|$(hash_file "$SFT_EVAL")|$(hash_file "$RL_DATA")|$(hash_file "$RL_EVAL")|$(hash_file "$VAL_DATA")" | sha256sum | awk '{print $1}')"
GPU_TELEMETRY="${GPU_TELEMETRY:-logs/gpu-${SLURM_JOB_ID}.csv}"; RUN_MANIFEST="${RUN_MANIFEST:-$OUTPUT_ROOT/run-manifest.json}"; MANIFEST_STARTED_AT="$(date -u +%FT%TZ)"
PACKAGE_VERSIONS="$(uv run --frozen python - <<'PY'
import importlib.metadata
print(";".join(f"{name}={importlib.metadata.version(name)}" for name in ("torch", "transformers", "trl", "deepspeed", "bitsandbytes")))
PY
)"
ARTIFACT_PATHS="$OUTPUT_ROOT|$CONFIG|$SFT_TRAIN|$SFT_EVAL|$RL_DATA|$RL_EVAL|$VAL_DATA|$SFT_TRAIN_DECISION|$GRPO_TRAIN_DECISION|$DAPO_TRAIN_DECISION|$BASE_GENERATION_DECISION|$DPO_GENERATION_DECISION|$SFT_GENERATION_DECISION|$GRPO_GENERATION_DECISION|$DAPO_GENERATION_DECISION|$SFT_SMOKE_MANIFEST|$GRPO_SMOKE_MANIFEST|$DAPO_SMOKE_MANIFEST"

validate_training() {
  local method="$1" decision="$2" decision_sha="$3" model="$4" revision="$5" data="$6" eval="$7" smoke="$8" smoke_sha="$9" scale="${10}" scale_sha="${11}"
  local data_sha eval_dataset_sha256 help
  data_sha="$(hash_file "$data")"
  eval_dataset_sha256="$(hash_file "$eval")"
  IFS=$'\t' read -r TRAIN_ATTENTION TRAIN_MICROBATCH TRAIN_ACCUM TRAIN_WORKERS TRAIN_EVAL_MICROBATCH TRAIN_MAX_STEPS TRAIN_MAX_LENGTH TRAIN_GRADIENT_CHECKPOINTING TRAIN_PACKING TRAIN_PADDING_FREE TRAIN_USE_LIGER TRAIN_LEARNING_RATE TRAIN_EPOCHS TRAIN_SAVE_STEPS TRAIN_EVAL_STEPS TRAIN_NUM_GENERATIONS TRAIN_GENERATION_BATCH_SIZE TRAIN_MAX_COMPLETION_LENGTH TRAIN_FALLBACK TRAIN_DECISION_SHA < <(
    run_probe_runner validate-decision --decision "$decision" --expected-sha256 "$decision_sha" --purpose training --output-format training-tsv \
      --training-method "$method" \
      --commit-hash "$COMMIT_HASH" --config-sha256 "$CONFIG_HASH" --model "$model" --model-revision "$revision" \
      --dataset-source "$DATASET_SOURCE" --dataset-revision "$DATASET_REVISION" --dataset-sha256 "$data_sha" \
      --eval-dataset-sha256 "$eval_dataset_sha256" \
      --prompt-sha256 "$PROMPT_HASH" --retrieval-sha256 "$RETRIEVAL_HASH" --source-schema-sha256 "$SOURCE_SCHEMA_HASH"
  ) || fail "$method training decision validation failed" optimization_decision_invalid
  awk -v expected="$LEARNING_RATE" -v frozen="$TRAIN_LEARNING_RATE" 'BEGIN { exit !(expected == frozen) }' || fail "$method learning rate differs from frozen decision" training_control_mismatch
  awk -v expected="$EPOCHS" -v frozen="$TRAIN_EPOCHS" 'BEGIN { exit !(expected == frozen) }' || fail "$method epochs differs from frozen decision" training_control_mismatch
  [[ "$SAVE_STEPS" == "$TRAIN_SAVE_STEPS" && "$EVAL_STEPS" == "$TRAIN_EVAL_STEPS" ]] || fail "$method cadence differs from frozen decision" training_control_mismatch
  run_probe_runner validate-scale-decision --decision "$scale" --expected-sha256 "$scale_sha" --train-gpus "$TRAIN_GPUS" --training-method "$method" \
    --commit-hash "$COMMIT_HASH" --config-sha256 "$CONFIG_HASH" --model "$model" --model-revision "$revision" \
    --dataset-source "$DATASET_SOURCE" --dataset-revision "$DATASET_REVISION" --dataset-sha256 "$data_sha" \
    --prompt-sha256 "$PROMPT_HASH" --retrieval-sha256 "$RETRIEVAL_HASH" --source-schema-sha256 "$SOURCE_SCHEMA_HASH" >/dev/null \
    || fail "$method scale decision validation failed" scale_decision_invalid
  run_probe_runner validate-checkpoints --smoke-manifest "$smoke" --expected-sha256 "$smoke_sha" \
    --commit-hash "$COMMIT_HASH" --config-sha256 "$CONFIG_HASH" --model "$model" --model-revision "$revision" \
    --dataset-source "$DATASET_SOURCE" --dataset-revision "$DATASET_REVISION" --dataset-sha256 "$data_sha" \
    --eval-dataset-sha256 "$eval_dataset_sha256" \
    --prompt-sha256 "$PROMPT_HASH" --retrieval-sha256 "$RETRIEVAL_HASH" --source-schema-sha256 "$SOURCE_SCHEMA_HASH" \
    --optimization-decision-sha256 "$decision_sha" --scale-decision-sha256 "$scale_sha" --method "$method" >/dev/null || fail "$method checkpoint smoke validation failed" checkpoint_smoke_invalid
  help="$(uv run --frozen python -m text_feedback_dpo.cli "train-$method" --help)" || fail "cannot inspect train-$method CLI" task7_train_cli_help_failed
  for flag in --eval --max-steps --max-length --dataloader-num-workers --per-device-train-batch-size --per-device-eval-batch-size --gradient-accumulation-steps --attention-implementation --gradient-checkpointing --packing --padding-free --use-liger-kernel; do [[ "$help" == *"$flag"* ]] || fail "Task 7 train-$method lacks $flag" task7_training_eval_cli_missing; done
}

validate_generation() {
  local method="$1" decision="$2" decision_sha="$3" model="$4" revision="$5" data_sha
  data_sha="$(hash_file "$VAL_DATA")"
  IFS=$'\t' read -r GEN_ATTENTION GEN_QUERY_BATCH GEN_RESPONSE_BATCH GEN_QUERY_TOKENS GEN_RESPONSE_TOKENS GEN_THINKING GEN_SCRATCH GEN_QUERY_TEMP GEN_RESPONSE_TEMP GEN_TOP_P GEN_TOP_K GEN_K1 GEN_B GEN_FALLBACK GEN_DECISION_SHA < <(
    run_probe_runner validate-decision --decision "$decision" --expected-sha256 "$decision_sha" --purpose generation --output-format generation-tsv \
      --commit-hash "$COMMIT_HASH" --config-sha256 "$CONFIG_HASH" --model "$model" --model-revision "$revision" \
      --dataset-source "$DATASET_SOURCE" --dataset-revision "$DATASET_REVISION" --dataset-sha256 "$data_sha" \
      --prompt-sha256 "$PROMPT_HASH" --retrieval-sha256 "$RETRIEVAL_HASH" --source-schema-sha256 "$SOURCE_SCHEMA_HASH" \
      --student-thinking-mode "$STUDENT_THINKING_MODE" --scratchpad-max-new-tokens "$SCRATCHPAD_MAX_NEW_TOKENS" \
      --query-temperature "$QUERY_TEMPERATURE" --response-temperature "$RESPONSE_TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K" --k1 "$BM25_K1" --b "$BM25_B"
  ) || fail "$method generation decision validation failed" generation_decision_invalid
}

SFT_MODEL="$OUTPUT_ROOT/sft/final"; GRPO_MODEL="$OUTPUT_ROOT/grpo/final"; DAPO_MODEL_TRAINED="$OUTPUT_ROOT/dapo/final"
validate_training sft "$SFT_TRAIN_DECISION" "$SFT_TRAIN_DECISION_SHA256" "$BASE_MODEL" "$BASE_REVISION" "$SFT_TRAIN" "$SFT_EVAL" "$SFT_SMOKE_MANIFEST" "$SFT_SMOKE_MANIFEST_SHA256" "$SFT_SCALE_DECISION" "$SFT_SCALE_DECISION_SHA256"
SFT_CONTROLS=("$TRAIN_MICROBATCH" "$TRAIN_ACCUM" "$TRAIN_WORKERS" "$TRAIN_EVAL_MICROBATCH" "$TRAIN_MAX_STEPS" "$TRAIN_MAX_LENGTH" "$TRAIN_ATTENTION" "$TRAIN_GRADIENT_CHECKPOINTING" "$TRAIN_PACKING" "$TRAIN_PADDING_FREE" "$TRAIN_USE_LIGER" "$TRAIN_LEARNING_RATE" "$TRAIN_EPOCHS" "$TRAIN_SAVE_STEPS" "$TRAIN_EVAL_STEPS" "$TRAIN_NUM_GENERATIONS" "$TRAIN_GENERATION_BATCH_SIZE" "$TRAIN_MAX_COMPLETION_LENGTH")
validate_training grpo "$GRPO_TRAIN_DECISION" "$GRPO_TRAIN_DECISION_SHA256" "$RL_START_MODEL" "$RL_START_REVISION" "$RL_DATA" "$RL_EVAL" "$GRPO_SMOKE_MANIFEST" "$GRPO_SMOKE_MANIFEST_SHA256" "$GRPO_SCALE_DECISION" "$GRPO_SCALE_DECISION_SHA256"
GRPO_CONTROLS=("$TRAIN_MICROBATCH" "$TRAIN_ACCUM" "$TRAIN_WORKERS" "$TRAIN_EVAL_MICROBATCH" "$TRAIN_MAX_STEPS" "$TRAIN_MAX_LENGTH" "$TRAIN_ATTENTION" "$TRAIN_GRADIENT_CHECKPOINTING" "$TRAIN_PACKING" "$TRAIN_PADDING_FREE" "$TRAIN_USE_LIGER" "$TRAIN_LEARNING_RATE" "$TRAIN_EPOCHS" "$TRAIN_SAVE_STEPS" "$TRAIN_EVAL_STEPS" "$TRAIN_NUM_GENERATIONS" "$TRAIN_GENERATION_BATCH_SIZE" "$TRAIN_MAX_COMPLETION_LENGTH")
validate_training dapo "$DAPO_TRAIN_DECISION" "$DAPO_TRAIN_DECISION_SHA256" "$RL_START_MODEL" "$RL_START_REVISION" "$RL_DATA" "$RL_EVAL" "$DAPO_SMOKE_MANIFEST" "$DAPO_SMOKE_MANIFEST_SHA256" "$DAPO_SCALE_DECISION" "$DAPO_SCALE_DECISION_SHA256"
DAPO_CONTROLS=("$TRAIN_MICROBATCH" "$TRAIN_ACCUM" "$TRAIN_WORKERS" "$TRAIN_EVAL_MICROBATCH" "$TRAIN_MAX_STEPS" "$TRAIN_MAX_LENGTH" "$TRAIN_ATTENTION" "$TRAIN_GRADIENT_CHECKPOINTING" "$TRAIN_PACKING" "$TRAIN_PADDING_FREE" "$TRAIN_USE_LIGER" "$TRAIN_LEARNING_RATE" "$TRAIN_EPOCHS" "$TRAIN_SAVE_STEPS" "$TRAIN_EVAL_STEPS" "$TRAIN_NUM_GENERATIONS" "$TRAIN_GENERATION_BATCH_SIZE" "$TRAIN_MAX_COMPLETION_LENGTH")
validate_generation base "$BASE_GENERATION_DECISION" "$BASE_GENERATION_DECISION_SHA256" "$BASE_MODEL" "$BASE_REVISION"
validate_generation dpo "$DPO_GENERATION_DECISION" "$DPO_GENERATION_DECISION_SHA256" "$DPO_MODEL" "$DPO_OUTPUT_REVISION"
validate_generation sft "$SFT_GENERATION_DECISION" "$SFT_GENERATION_DECISION_SHA256" "$SFT_MODEL" "$SFT_OUTPUT_REVISION"
validate_generation grpo "$GRPO_GENERATION_DECISION" "$GRPO_GENERATION_DECISION_SHA256" "$GRPO_MODEL" "$GRPO_OUTPUT_REVISION"
validate_generation dapo "$DAPO_GENERATION_DECISION" "$DAPO_GENERATION_DECISION_SHA256" "$DAPO_MODEL_TRAINED" "$DAPO_OUTPUT_REVISION"
log_event launch_contract_validated decisions="${BASE_GENERATION_DECISION_SHA256},${DPO_GENERATION_DECISION_SHA256},${SFT_TRAIN_DECISION_SHA256},${GRPO_TRAIN_DECISION_SHA256},${DAPO_TRAIN_DECISION_SHA256},${SFT_GENERATION_DECISION_SHA256},${GRPO_GENERATION_DECISION_SHA256},${DAPO_GENERATION_DECISION_SHA256}" fallback_reason=none
nvidia-smi
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv -l 10 > "$GPU_TELEMETRY" &
GPU_MONITOR_PID=$!
cleanup() { if kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then kill "$GPU_MONITOR_PID"; fi; }
trap cleanup EXIT

train_arm() {
  local method="$1" model="$2" revision="$3" train="$4" eval="$5" output="$6"; shift 6
  local -a controls=("$@"); local microbatch="${controls[0]}" accum="${controls[1]}" workers="${controls[2]}" eval_microbatch="${controls[3]}" max_steps="${controls[4]}" max_length="${controls[5]}" attention="${controls[6]}" learning_rate="${controls[11]}" epochs="${controls[12]}" save_steps="${controls[13]}" eval_steps="${controls[14]}" num_generations="${controls[15]}" generation_batch_size="${controls[16]}" max_completion_length="${controls[17]}"
  local -a booleans=(); [[ "${controls[7]}" == True ]] && booleans+=(--gradient-checkpointing) || booleans+=(--no-gradient-checkpointing); [[ "${controls[8]}" == True ]] && booleans+=(--packing) || booleans+=(--no-packing); [[ "${controls[9]}" == True ]] && booleans+=(--padding-free) || booleans+=(--no-padding-free); [[ "${controls[10]}" == True ]] && booleans+=(--use-liger-kernel) || booleans+=(--no-use-liger-kernel)
  uv run --frozen python -m torch.distributed.run --standalone --nproc_per_node="$ALLOCATED_GPU_COUNT" -m text_feedback_dpo.cli "train-$method" \
    --config "$CONFIG" --train "$train" --eval "$eval" --output "$output" --model "$model" --model-revision "$revision" \
    --deepspeed-config configs/deepspeed_zero3.json --save-steps "$save_steps" --eval-steps "$eval_steps" --gradient-accumulation-steps "$accum" \
    --learning-rate "$learning_rate" --epochs "$epochs" \
    --per-device-train-batch-size "$microbatch" --per-device-eval-batch-size "$eval_microbatch" --dataloader-num-workers "$workers" \
    --max-steps "$max_steps" --max-length "$max_length" --attention-implementation "$attention" \
    --num-generations "$num_generations" --generation-batch-size "$generation_batch_size" --max-completion-length "$max_completion_length" "${booleans[@]}"
}
train_arm sft "$BASE_MODEL" "$BASE_REVISION" "$SFT_TRAIN" "$SFT_EVAL" "$OUTPUT_ROOT/sft" "${SFT_CONTROLS[@]}"
train_arm grpo "$RL_START_MODEL" "$RL_START_REVISION" "$RL_DATA" "$RL_EVAL" "$OUTPUT_ROOT/grpo" "${GRPO_CONTROLS[@]}"
train_arm dapo "$RL_START_MODEL" "$RL_START_REVISION" "$RL_DATA" "$RL_EVAL" "$OUTPUT_ROOT/dapo" "${DAPO_CONTROLS[@]}"

for method in base dpo sft grpo dapo; do
  case "$method" in
    base) model="$BASE_MODEL"; revision="$BASE_REVISION"; decision="$BASE_GENERATION_DECISION"; decision_sha="$BASE_GENERATION_DECISION_SHA256" ;;
    dpo) model="$DPO_MODEL"; revision="$DPO_OUTPUT_REVISION"; decision="$DPO_GENERATION_DECISION"; decision_sha="$DPO_GENERATION_DECISION_SHA256" ;;
    sft) model="$SFT_MODEL"; revision="$SFT_OUTPUT_REVISION"; decision="$SFT_GENERATION_DECISION"; decision_sha="$SFT_GENERATION_DECISION_SHA256" ;;
    grpo) model="$GRPO_MODEL"; revision="$GRPO_OUTPUT_REVISION"; decision="$GRPO_GENERATION_DECISION"; decision_sha="$GRPO_GENERATION_DECISION_SHA256" ;;
    dapo) model="$DAPO_MODEL_TRAINED"; revision="$DAPO_OUTPUT_REVISION"; decision="$DAPO_GENERATION_DECISION"; decision_sha="$DAPO_GENERATION_DECISION_SHA256" ;;
  esac
  validate_generation "$method" "$decision" "$decision_sha" "$model" "$revision"
  uv run --frozen python -m text_feedback_dpo.cli generate-searchqa --data "$VAL_DATA" --output "$OUTPUT_ROOT/$method-validation-predictions.jsonl" \
    --model "$model" --model-revision "$revision" --dataset-source "$DATASET_SOURCE" --dataset-revision "$DATASET_REVISION" \
    --attention-implementation "$GEN_ATTENTION" --device cuda:0 --student-thinking-mode "$GEN_THINKING" --scratchpad-max-new-tokens "$GEN_SCRATCH" \
    --query-batch-size "$GEN_QUERY_BATCH" --response-batch-size "$GEN_RESPONSE_BATCH" --query-max-new-tokens "$GEN_QUERY_TOKENS" --response-max-new-tokens "$GEN_RESPONSE_TOKENS" \
    --query-temperature "$GEN_QUERY_TEMP" --response-temperature "$GEN_RESPONSE_TEMP" --top-p "$GEN_TOP_P" --top-k "$GEN_TOP_K" --k1 "$GEN_K1" --b "$GEN_B" \
    --context-budget 4096 --prompt-version "$PROMPT_VERSION" --policy-hash "$POLICY_HASH:$method:validation"
  uv run --frozen python -m text_feedback_dpo.cli evaluate --data "$VAL_DATA" --predictions "$OUTPUT_ROOT/$method-validation-predictions.jsonl" --output "$OUTPUT_ROOT/$method-validation-metrics.json" --protocol active-search
done
uv run --frozen python -m text_feedback_dpo.cli compare --run "base-validation=$OUTPUT_ROOT/base-validation-metrics.json" --run "dpo-validation=$OUTPUT_ROOT/dpo-validation-metrics.json" --run "sft-validation=$OUTPUT_ROOT/sft-validation-metrics.json" --run "grpo-validation=$OUTPUT_ROOT/grpo-validation-metrics.json" --run "dapo-validation=$OUTPUT_ROOT/dapo-validation-metrics.json" \
  --output "$OUTPUT_ROOT/comparison.json" --html "$OUTPUT_ROOT/comparison.html" --artifact "$OUTPUT_ROOT/base-validation-metrics.json" --artifact "$OUTPUT_ROOT/dpo-validation-metrics.json" --artifact "$OUTPUT_ROOT/sft-validation-metrics.json" --artifact "$OUTPUT_ROOT/grpo-validation-metrics.json" --artifact "$OUTPUT_ROOT/dapo-validation-metrics.json"
log_event comparisons_complete artifact="$OUTPUT_ROOT/comparison.json" protocol=active-search split=validation fallback_reason=none
MANIFEST_ENDED_AT="$(date -u +%FT%TZ)"
export COMMIT_HASH CONFIG_HASH MODEL_HASH DATASET_HASH PROMPT_HASH RETRIEVAL_HASH SOURCE_SCHEMA_HASH GPU_TELEMETRY PACKAGE_VERSIONS ARTIFACT_PATHS MANIFEST_STARTED_AT MANIFEST_ENDED_AT RUN_MANIFEST
export BASE_GENERATION_DECISION_SHA256 DPO_GENERATION_DECISION_SHA256 SFT_TRAIN_DECISION_SHA256 GRPO_TRAIN_DECISION_SHA256 DAPO_TRAIN_DECISION_SHA256 SFT_GENERATION_DECISION_SHA256 GRPO_GENERATION_DECISION_SHA256 DAPO_GENERATION_DECISION_SHA256
uv run --frozen python - "$RUN_MANIFEST" <<'PY'
import json, os, platform, socket, sys
manifest = {
    "status": "complete", "commit_hash": os.environ["COMMIT_HASH"], "config_hash": os.environ["CONFIG_HASH"],
    "model_hash": os.environ["MODEL_HASH"], "dataset_hash": os.environ["DATASET_HASH"], "prompt_hash": os.environ["PROMPT_HASH"],
    "retrieval_hash": os.environ["RETRIEVAL_HASH"], "source_schema_hash": os.environ["SOURCE_SCHEMA_HASH"], "node": socket.gethostname(),
    "platform": platform.platform(), "slurm_allocation": {key: os.environ.get(key) for key in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_NNODES", "SLURM_NTASKS", "SLURM_GPUS_ON_NODE")},
    "package_versions": os.environ["PACKAGE_VERSIONS"].split(";"), "gpu_telemetry": os.environ["GPU_TELEMETRY"],
    "timings": {"started_at": os.environ["MANIFEST_STARTED_AT"], "ended_at": os.environ["MANIFEST_ENDED_AT"]}, "artifact_paths": os.environ["ARTIFACT_PATHS"].split("|"), "fallback_reason": "none",
    "optimization_decision_hashes": {key: os.environ[key] for key in ("BASE_GENERATION_DECISION_SHA256", "DPO_GENERATION_DECISION_SHA256", "SFT_TRAIN_DECISION_SHA256", "GRPO_TRAIN_DECISION_SHA256", "DAPO_TRAIN_DECISION_SHA256", "SFT_GENERATION_DECISION_SHA256", "GRPO_GENERATION_DECISION_SHA256", "DAPO_GENERATION_DECISION_SHA256")},
}
with open(sys.argv[1], "w", encoding="utf-8") as handle: json.dump(manifest, handle, sort_keys=True, indent=2); handle.write("\n")
PY
