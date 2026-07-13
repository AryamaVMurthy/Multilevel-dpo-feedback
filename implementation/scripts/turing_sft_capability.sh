#!/bin/bash
# CPU-only canonical capability audit for generated SFT continuations.
#SBATCH -p u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4096
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail

COMPONENT=turing_sft_capability
log_event() { local event="$1"; shift; printf 'event=%s timestamp=%s component=%s %s\n' "$event" "$(date -u +%FT%TZ)" "$COMPONENT" "$*"; }
fail() { log_event failure reason="$1" fallback_reason="${2:-none}" >&2; exit 2; }
require_env() { local name="$1"; [[ -n "${!name:-}" ]] || fail "required environment variable is missing: $name" capability_contract_missing; }

require_env "PROJECT_DIR"
require_env "EXPECTED_COMMIT"
require_env "SFT_DATA"
require_env "SFT_DATA_HASH"
require_env "REPRODUCTION"
require_env "REPRODUCTION_HASH"
require_env "EXAMPLES"
require_env "EXAMPLES_HASH"
require_env "BOOTSTRAP"
require_env "BOOTSTRAP_HASH"
require_env "OUTPUT"
require_env "REPORT"
[[ "${SLURM_NNODES:?}" == 1 && "${SLURM_NTASKS:?}" == 1 ]] || fail "SFT capability audit requires one node and one task" allocation_shape_invalid
for path in "$SFT_DATA" "$REPRODUCTION" "$EXAMPLES" "$BOOTSTRAP"; do
  [[ -f "$path" ]] || fail "required capability artifact is missing: $path" capability_input_missing
done
[[ "$(git -C "$PROJECT_DIR" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || fail "remote commit differs from EXPECTED_COMMIT" commit_mismatch
[[ "$(sha256sum "$SFT_DATA" | awk '{print $1}')" == "$SFT_DATA_HASH" ]] || fail "SFT data hash mismatch" sft_data_hash_mismatch
[[ "$(sha256sum "$REPRODUCTION" | awk '{print $1}')" == "$REPRODUCTION_HASH" ]] || fail "reproduction hash mismatch" reproduction_hash_mismatch
[[ "$(sha256sum "$EXAMPLES" | awk '{print $1}')" == "$EXAMPLES_HASH" ]] || fail "examples hash mismatch" examples_hash_mismatch
[[ "$(sha256sum "$BOOTSTRAP" | awk '{print $1}')" == "$BOOTSTRAP_HASH" ]] || fail "bootstrap hash mismatch" bootstrap_hash_mismatch
[[ ! -e "$OUTPUT" && ! -e "$REPORT" ]] || fail "output or report already exists; refusing stale reuse" output_exists

cd "$PROJECT_DIR"
[[ -f pyproject.toml && -d src/text_feedback_dpo ]] || fail "PROJECT_DIR is not the implementation root" project_root_invalid
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 UV_LINK_MODE=copy
mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$REPORT")" logs

log_event capability_start sft_data_sha256="$SFT_DATA_HASH" reproduction_sha256="$REPRODUCTION_HASH" examples_sha256="$EXAMPLES_HASH" bootstrap_sha256="$BOOTSTRAP_HASH" fallback_reason=none
uv run --frozen python -m text_feedback_dpo.cli evaluate-sft-capability \
  --sft-data "$SFT_DATA" --sft-data-sha256 "$SFT_DATA_HASH" \
  --reproduction "$REPRODUCTION" --reproduction-sha256 "$REPRODUCTION_HASH" \
  --examples "$EXAMPLES" --examples-sha256 "$EXAMPLES_HASH" \
  --bootstrap "$BOOTSTRAP" --bootstrap-sha256 "$BOOTSTRAP_HASH" \
  --output "$OUTPUT" --report "$REPORT"
[[ -s "$OUTPUT" && -s "$REPORT" ]] || fail "capability audit did not create non-empty artifacts" capability_artifact_missing
log_event capability_complete output="$OUTPUT" report="$REPORT" output_sha256="$(sha256sum "$OUTPUT" | awk '{print $1}')" report_sha256="$(sha256sum "$REPORT" | awk '{print $1}')" fallback_reason=none
