#!/bin/bash
# Build canonically verified SFT rows from a commit- and hash-bound rollout artifact.
# Submit with: sbatch -A <account> --export=ALL,... scripts/turing_build_sft.sh
#SBATCH -p u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4096
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail

fail() { printf 'event=sft_build_failure reason=%q fallback_reason=%q\n' "$1" "${2:-none}" >&2; exit 2; }
for name in PROJECT_DIR EXPECTED_COMMIT CONFIG DATA DATA_SHA256 TRAJECTORIES \
  TRAJECTORIES_SHA256 OUTPUT REPORT MIN_COVERAGE MIN_ROWS; do
  [[ -n "${!name:-}" ]] || fail "required environment variable is missing: $name" sft_build_contract_missing
done
[[ "$(git -C "$PROJECT_DIR" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || fail "remote commit differs from EXPECTED_COMMIT" commit_mismatch
for path in "$CONFIG" "$DATA" "$TRAJECTORIES"; do
  [[ -f "$path" ]] || fail "required SFT input does not exist: $path" sft_build_input_missing
done
[[ "$(sha256sum "$DATA" | awk '{print $1}')" == "$DATA_SHA256" ]] || fail "canonical data hash mismatch" data_hash_mismatch
[[ "$(sha256sum "$TRAJECTORIES" | awk '{print $1}')" == "$TRAJECTORIES_SHA256" ]] || fail "trajectory hash mismatch" trajectory_hash_mismatch
[[ "$MIN_ROWS" =~ ^[1-9][0-9]*$ ]] || fail "MIN_ROWS must be a positive integer" sft_gate_invalid

cd "$PROJECT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 UV_LINK_MODE=copy
export HF_HOME="${HF_CACHE_ROOT:-/scratch/$(hostname)/$USER/searchqa-dpo/hf}"
export HF_DATASETS_CACHE="$HF_HOME/datasets" HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" "$(dirname "$OUTPUT")" "$(dirname "$REPORT")" logs
printf 'event=sft_build_start commit_hash=%q data_hash=%q trajectory_hash=%q min_coverage=%q min_rows=%q fallback_reason=none\n' \
  "$EXPECTED_COMMIT" "$DATA_SHA256" "$TRAJECTORIES_SHA256" "$MIN_COVERAGE" "$MIN_ROWS"
uv run --frozen python -m text_feedback_dpo.cli build-sft-data \
  --config "$CONFIG" \
  --data "$DATA" \
  --trajectories "$TRAJECTORIES" \
  --output "$OUTPUT" \
  --report "$REPORT" \
  --min-coverage "$MIN_COVERAGE" \
  --min-rows "$MIN_ROWS"
[[ -s "$OUTPUT" && -s "$REPORT" ]] || fail "SFT builder did not produce non-empty output and report" sft_build_output_missing
printf 'event=sft_build_complete output=%q report=%q fallback_reason=none\n' "$OUTPUT" "$REPORT"
