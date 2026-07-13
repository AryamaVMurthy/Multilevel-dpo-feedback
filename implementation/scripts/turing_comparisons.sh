#!/bin/bash
# Run post-DPO SFT/GRPO/DAPO comparisons from one frozen initialization.
#SBATCH -p u22
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=4096
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail

: "${TURING_ACCOUNT:?TURING_ACCOUNT must be supplied with --export}"
: "${PROJECT_DIR:?PROJECT_DIR must be supplied with --export}"
: "${CONFIG:?CONFIG must be supplied with --export}"
: "${BASE_MODEL:?BASE_MODEL must be supplied with --export}"
: "${BASE_REVISION:?BASE_REVISION must be supplied with --export}"
: "${SFT_TRAIN:?SFT_TRAIN must be supplied with --export}"
: "${SFT_EVAL:?SFT_EVAL must be supplied with --export}"
: "${RL_DATA:?RL_DATA must be supplied with --export}"
: "${VAL_DATA:?VAL_DATA must be supplied with --export}"
: "${TEST_DATA:?TEST_DATA must be supplied with --export}"
: "${DPO_METRICS:?DPO_METRICS must be supplied with --export}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT must be supplied with --export}"

module load u22/cuda/12.4
cd "$PROJECT_DIR"
[[ -f pyproject.toml && -d src/text_feedback_dpo ]] || { echo "ERROR: invalid PROJECT_DIR" >&2; exit 2; }
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 UV_LINK_MODE=copy
export HF_HOME="${HF_CACHE_ROOT:-/scratch/$(hostname)/$USER/searchqa-dpo/hf}"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" "$OUTPUT_ROOT" logs
ATTENTION_IMPLEMENTATION="${ATTENTION_IMPLEMENTATION:-sdpa}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-16}"
echo "<runtime component=comparisons attention_implementation=\"$ATTENTION_IMPLEMENTATION\" fallback_reason=\"${ATTENTION_FALLBACK_REASON:-none}\"/>"

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv -l 10 > "logs/gpu-${SLURM_JOB_ID}.csv" &
GPU_MONITOR_PID=$!
cleanup() {
  if kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then
    kill "$GPU_MONITOR_PID"
  fi
}
trap cleanup EXIT

if [[ -n "${EXISTING_SFT:-}" ]]; then
  SFT_MODEL="$EXISTING_SFT"
  START_MODEL="$EXISTING_SFT"
  START_REVISION=""
  echo "<comparison_sft source=existing_checkpoint model=\"$SFT_MODEL\"/>"
else
  uv run --frozen python -m torch.distributed.run --standalone --nproc_per_node=2 -m text_feedback_dpo.cli train-sft \
    --config "$CONFIG" --train "$SFT_TRAIN" --eval "$SFT_EVAL" \
    --model "$BASE_MODEL" --model-revision "$BASE_REVISION" \
    --output "$OUTPUT_ROOT/sft" --deepspeed-config configs/deepspeed_zero3.json \
    --save-steps 100 --eval-steps 100
  SFT_MODEL="$OUTPUT_ROOT/sft/final"
  START_MODEL="$SFT_MODEL"
  START_REVISION=""
fi

RL_ARGS=(--config "$CONFIG" --train "$RL_DATA" --output "$OUTPUT_ROOT/grpo" --model "$START_MODEL" --deepspeed-config configs/deepspeed_zero3.json --save-steps 100 --eval-steps 100)
if [[ -n "$START_REVISION" ]]; then RL_ARGS+=(--model-revision "$START_REVISION"); fi
uv run --frozen python -m torch.distributed.run --standalone --nproc_per_node=2 -m text_feedback_dpo.cli train-grpo "${RL_ARGS[@]}"

RL_ARGS=(--config "$CONFIG" --train "$RL_DATA" --output "$OUTPUT_ROOT/dapo" --model "$START_MODEL" --deepspeed-config configs/deepspeed_zero3.json --save-steps 100 --eval-steps 100)
if [[ -n "$START_REVISION" ]]; then RL_ARGS+=(--model-revision "$START_REVISION"); fi
uv run --frozen python -m torch.distributed.run --standalone --nproc_per_node=2 -m text_feedback_dpo.cli train-dapo "${RL_ARGS[@]}"

for method in sft grpo dapo; do
  MODEL="$OUTPUT_ROOT/$method/final"
  if [[ "$method" == "sft" ]]; then MODEL="$SFT_MODEL"; fi
  for split in validation test; do
    DATA="$VAL_DATA"
    if [[ "$split" == "test" ]]; then DATA="$TEST_DATA"; fi
    uv run --frozen python -m text_feedback_dpo.cli generate \
      --data "$DATA" --output "$OUTPUT_ROOT/$method-$split-predictions.jsonl" \
      --model "$MODEL" --attention-implementation "$ATTENTION_IMPLEMENTATION" \
      --batch-size "$GENERATION_BATCH_SIZE" --student-thinking-mode "${STUDENT_THINKING_MODE:-direct}" --max-new-tokens 32 --temperature 0.0 --top-p 1.0 \
      --policy-hash "${POLICY_HASH:?POLICY_HASH must be supplied with --export}:$method:$split"
    uv run --frozen python -m text_feedback_dpo.cli evaluate \
      --data "$DATA" --predictions "$OUTPUT_ROOT/$method-$split-predictions.jsonl" \
      --output "$OUTPUT_ROOT/$method-$split-metrics.json"
  done
done

uv run --frozen python -m text_feedback_dpo.cli compare \
  --run "dpo=$DPO_METRICS" \
  --run "sft-validation=$OUTPUT_ROOT/sft-validation-metrics.json" \
  --run "grpo-validation=$OUTPUT_ROOT/grpo-validation-metrics.json" \
  --run "dapo-validation=$OUTPUT_ROOT/dapo-validation-metrics.json" \
  --run "sft-test=$OUTPUT_ROOT/sft-test-metrics.json" \
  --run "grpo-test=$OUTPUT_ROOT/grpo-test-metrics.json" \
  --run "dapo-test=$OUTPUT_ROOT/dapo-test-metrics.json" \
  --output "$OUTPUT_ROOT/comparison.json" --html "$OUTPUT_ROOT/comparison.html" \
  --artifact "$DPO_METRICS" \
  --artifact "$OUTPUT_ROOT/sft-validation-metrics.json" \
  --artifact "$OUTPUT_ROOT/grpo-validation-metrics.json" \
  --artifact "$OUTPUT_ROOT/dapo-validation-metrics.json" \
  --artifact "$OUTPUT_ROOT/sft-test-metrics.json" \
  --artifact "$OUTPUT_ROOT/grpo-test-metrics.json" \
  --artifact "$OUTPUT_ROOT/dapo-test-metrics.json"
