#!/bin/bash
# Deterministically select and shard a train-only pool for parallel checkpoint rollouts.
#SBATCH -p u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8192
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail

COMPONENT=turing_prepare_sft_scale
log_event() { local event="$1"; shift; printf 'event=%s timestamp=%s component=%s %s\n' "$event" "$(date -u +%FT%TZ)" "$COMPONENT" "$*"; }
fail() { log_event failure reason="$1" fallback_reason="${2:-none}" >&2; exit 2; }
require_env() { local name="$1"; [[ -n "${!name:-}" ]] || fail "required environment variable is missing: $name" scale_prepare_contract_missing; }

require_env "PROJECT_DIR"
require_env "EXPECTED_COMMIT"
require_env "TRAIN"
require_env "TRAIN_SHA256"
require_env "OUTPUT_ROOT"
require_env "POOL_COUNT"
require_env "POOL_SEED"
require_env "SHARD_COUNT"
[[ "${SLURM_NNODES:?}" == 1 && "${SLURM_NTASKS:?}" == 1 ]] || fail "SFT scale preparation requires one node and one task" allocation_shape_invalid
[[ "$POOL_COUNT" =~ ^[1-9][0-9]*$ && "$POOL_SEED" =~ ^[0-9]+$ && "$SHARD_COUNT" =~ ^[1-9][0-9]*$ ]] || fail "pool count/seed/shard count are invalid" scale_prepare_numeric_invalid
[[ -f "$TRAIN" ]] || fail "train artifact is missing: $TRAIN" train_missing
[[ "$(sha256sum "$TRAIN" | awk '{print $1}')" == "$TRAIN_SHA256" ]] || fail "train artifact hash mismatch" train_hash_mismatch
[[ "$(git -C "$PROJECT_DIR" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || fail "remote commit differs from EXPECTED_COMMIT" commit_mismatch
[[ ! -e "$OUTPUT_ROOT" ]] || fail "OUTPUT_ROOT already exists; refusing stale reuse: $OUTPUT_ROOT" output_exists

cd "$PROJECT_DIR"
[[ -f pyproject.toml && -d src/text_feedback_dpo ]] || fail "PROJECT_DIR is not the implementation root" project_root_invalid
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 UV_LINK_MODE=copy
mkdir -p "$OUTPUT_ROOT"
POOL="$OUTPUT_ROOT/pool.jsonl"
SHARDS="$OUTPUT_ROOT/shards"

log_event selection_start train_sha256="$TRAIN_SHA256" pool_count="$POOL_COUNT" pool_seed="$POOL_SEED" fallback_reason=none
uv run --frozen python scripts/select_bootstrap_pool.py \
  --input "$TRAIN" --output "$POOL" --count "$POOL_COUNT" --seed "$POOL_SEED" \
  --expected-input-sha256 "$TRAIN_SHA256"
log_event sharding_start shard_count="$SHARD_COUNT" fallback_reason=none
uv run --frozen python -m text_feedback_dpo.cli shard-jsonl \
  --input "$POOL" --output-dir "$SHARDS" --shards "$SHARD_COUNT"

export POOL SHARDS
uv run --frozen python - <<'PY'
import hashlib, json, os
from pathlib import Path

def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

pool = Path(os.environ["POOL"])
shards = Path(os.environ["SHARDS"])
shard_manifest = json.loads((shards / "manifest.json").read_text(encoding="utf-8"))
if shard_manifest.get("rows") != int(os.environ["POOL_COUNT"]):
    raise SystemExit("shard manifest row count does not match POOL_COUNT")
if shard_manifest.get("shards") != int(os.environ["SHARD_COUNT"]):
    raise SystemExit("shard manifest shard count does not match SHARD_COUNT")
shard_files = []
for index in range(int(os.environ["SHARD_COUNT"])):
    path = shards / f"shard-{index}.jsonl"
    rows = sum(1 for line in path.open(encoding="utf-8") if line.strip())
    expected_rows = shard_manifest["shard_rows"][index]
    if rows != expected_rows:
        raise SystemExit(f"shard {index} row count mismatch: {rows} != {expected_rows}")
    shard_files.append({"index": index, "path": str(path), "rows": rows, "sha256": file_hash(path)})
manifest = {
    "status": "complete",
    "stage": "full_sft_train_only_pool_selection_and_sharding",
    "commit_hash": os.environ["EXPECTED_COMMIT"],
    "train": {"path": os.environ["TRAIN"], "sha256": os.environ["TRAIN_SHA256"]},
    "selection": {"count": int(os.environ["POOL_COUNT"]), "seed": int(os.environ["POOL_SEED"])},
    "pool": {"path": str(pool), "sha256": file_hash(pool)},
    "sharding": {"count": int(os.environ["SHARD_COUNT"]), "assignment": shard_manifest["assignment"], "files": shard_files},
    "fallback_reason": "none",
}
output = Path(os.environ["OUTPUT_ROOT"]) / "run-manifest.json"
output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(manifest, sort_keys=True))
PY
log_event scale_prepare_complete manifest="$OUTPUT_ROOT/run-manifest.json" fallback_reason=none
