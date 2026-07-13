#!/bin/bash
# Create balanced, trajectory-disjoint SFT train/eval pairs from verified targets.
#SBATCH -p u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4096
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail

COMPONENT=turing_split_paired_sft
log_event() { local event="$1"; shift; printf 'event=%s timestamp=%s component=%s %s\n' "$event" "$(date -u +%FT%TZ)" "$COMPONENT" "$*"; }
fail() { log_event failure reason="$1" fallback_reason="${2:-none}" >&2; exit 2; }
require_env() { local name="$1"; [[ -n "${!name:-}" ]] || fail "required environment variable is missing: $name" paired_split_contract_missing; }

require_env "PROJECT_DIR"
require_env "EXPECTED_COMMIT"
require_env "INPUT"
require_env "INPUT_SHA256"
require_env "TRAIN"
require_env "EVAL"
require_env "REPORT"
require_env "EVAL_PAIRS"
require_env "MIN_TRAIN_PAIRS"
require_env "SEED"
[[ "${SLURM_NNODES:?}" == 1 && "${SLURM_NTASKS:?}" == 1 ]] || fail "paired SFT split requires one node and one task" allocation_shape_invalid
[[ "$EVAL_PAIRS" =~ ^[1-9][0-9]*$ && "$MIN_TRAIN_PAIRS" =~ ^[1-9][0-9]*$ && "$SEED" =~ ^[0-9]+$ ]] || fail "paired SFT split numeric controls are invalid" paired_split_numeric_invalid
[[ -f "$INPUT" ]] || fail "verified SFT input is missing" sft_input_missing
[[ "$(sha256sum "$INPUT" | awk '{print $1}')" == "$INPUT_SHA256" ]] || fail "verified SFT input hash mismatch" sft_input_hash_mismatch
[[ "$(git -C "$PROJECT_DIR" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || fail "remote commit differs from EXPECTED_COMMIT" commit_mismatch
for output in "$TRAIN" "$EVAL" "$REPORT"; do
  [[ ! -e "$output" ]] || fail "paired SFT output already exists: $output" output_exists
done

cd "$PROJECT_DIR"
[[ -f pyproject.toml && -d src/text_feedback_dpo ]] || fail "PROJECT_DIR is not the implementation root" project_root_invalid
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 UV_LINK_MODE=copy
mkdir -p "$(dirname "$TRAIN")" "$(dirname "$EVAL")" "$(dirname "$REPORT")" logs

log_event paired_split_start input_sha256="$INPUT_SHA256" eval_pairs="$EVAL_PAIRS" min_train_pairs="$MIN_TRAIN_PAIRS" seed="$SEED" fallback_reason=none
uv run --frozen python -m text_feedback_dpo.cli split-paired-sft \
  --input "$INPUT" --train "$TRAIN" --eval "$EVAL" --report "$REPORT" \
  --eval-pairs "$EVAL_PAIRS" --min-train-pairs "$MIN_TRAIN_PAIRS" --seed "$SEED"
uv run --frozen python - "$TRAIN" "$EVAL" "$REPORT" <<'PY'
import json, sys
from pathlib import Path

train, evaluation, report_path = map(Path, sys.argv[1:])
report = json.loads(report_path.read_text(encoding="utf-8"))
if report.get("trajectory_overlap") != 0:
    raise SystemExit("paired SFT split reports trajectory leakage")
if report.get("train_rows") != sum(1 for line in train.open(encoding="utf-8") if line.strip()):
    raise SystemExit("paired SFT train row count mismatch")
if report.get("eval_rows") != sum(1 for line in evaluation.open(encoding="utf-8") if line.strip()):
    raise SystemExit("paired SFT eval row count mismatch")
if report["train_rows"] != 2 * report["train_pairs"] or report["eval_rows"] != 2 * report["eval_pairs"]:
    raise SystemExit("paired SFT output is not exactly task balanced")
print(json.dumps(report, sort_keys=True))
PY
log_event paired_split_complete train_sha256="$(sha256sum "$TRAIN" | awk '{print $1}')" eval_sha256="$(sha256sum "$EVAL" | awk '{print $1}')" report_sha256="$(sha256sum "$REPORT" | awk '{print $1}')" trajectory_overlap=0 fallback_reason=none
