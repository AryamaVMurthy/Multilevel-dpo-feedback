#!/bin/bash
# Validate and merge deterministic bootstrap shards after every rollout job succeeds.
#SBATCH -p u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8192
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail

COMPONENT=turing_merge_bootstrap
log_event() { local event="$1"; shift; printf 'event=%s timestamp=%s component=%s %s\n' "$event" "$(date -u +%FT%TZ)" "$COMPONENT" "$*"; }
fail() { log_event failure reason="$1" fallback_reason="${2:-none}" >&2; exit 2; }
require_env() { local name="$1"; [[ -n "${!name:-}" ]] || fail "required environment variable is missing: $name" bootstrap_merge_contract_missing; }

require_env "PROJECT_DIR"
require_env "EXPECTED_COMMIT"
require_env "PREPARE_MANIFEST"
require_env "PREPARE_MANIFEST_SHA256"
require_env "SHARD_DIR"
require_env "SHARD_COUNT"
require_env "OUTPUT"
require_env "MODEL"
require_env "MODEL_REVISION"
require_env "MODEL_ARTIFACT_SHA256"
require_env "POLICY_HASH"
require_env "PROMPT_VERSION"
require_env "EVALUATOR_VERSION"
require_env "SEEDS"
require_env "QUERY_BATCH_SIZE"
require_env "RESPONSE_BATCH_SIZE"
require_env "QUERY_MAX_NEW_TOKENS"
require_env "RESPONSE_MAX_NEW_TOKENS"
require_env "QUERY_MIN_NEW_TOKENS"
require_env "RESPONSE_MIN_NEW_TOKENS"
require_env "QUERY_TEMPERATURE"
require_env "RESPONSE_TEMPERATURE"
require_env "TOP_P"
[[ "${SLURM_NNODES:?}" == 1 && "${SLURM_NTASKS:?}" == 1 ]] || fail "bootstrap merge requires one node and one task" allocation_shape_invalid
[[ "$SHARD_COUNT" =~ ^[1-9][0-9]*$ ]] || fail "SHARD_COUNT must be positive" shard_count_invalid
[[ "$(git -C "$PROJECT_DIR" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || fail "remote commit differs from EXPECTED_COMMIT" commit_mismatch
[[ -f "$PREPARE_MANIFEST" ]] || fail "prepare manifest is missing" prepare_manifest_missing
[[ "$(sha256sum "$PREPARE_MANIFEST" | awk '{print $1}')" == "$PREPARE_MANIFEST_SHA256" ]] || fail "prepare manifest hash mismatch" prepare_manifest_hash_mismatch
[[ -d "$SHARD_DIR" ]] || fail "shard directory is missing" shard_dir_missing
[[ ! -e "$OUTPUT" && ! -e "${OUTPUT%.*}.manifest.json" ]] || fail "merged output or manifest already exists" output_exists

cd "$PROJECT_DIR"
[[ -f pyproject.toml && -d src/text_feedback_dpo ]] || fail "PROJECT_DIR is not the implementation root" project_root_invalid
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 UV_LINK_MODE=copy
read -r -a SEED_ARGS <<< "$SEEDS"
(( ${#SEED_ARGS[@]} > 0 )) || fail "SEEDS parsed to an empty list" seeds_empty

uv run --frozen python - <<'PY'
import hashlib, json, os
from pathlib import Path

def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

prepare = json.loads(Path(os.environ["PREPARE_MANIFEST"]).read_text(encoding="utf-8"))
shard_count = int(os.environ["SHARD_COUNT"])
if prepare.get("status") != "complete" or prepare.get("sharding", {}).get("count") != shard_count:
    raise SystemExit("prepare manifest status/shard count mismatch")
prepared = {int(item["index"]): item for item in prepare["sharding"]["files"]}
if set(prepared) != set(range(shard_count)):
    raise SystemExit("prepare manifest shard index parity mismatch")
seeds = [int(value) for value in os.environ["SEEDS"].split()]
generation = {
    "query_batch_size": int(os.environ["QUERY_BATCH_SIZE"]),
    "response_batch_size": int(os.environ["RESPONSE_BATCH_SIZE"]),
    "query_max_new_tokens": int(os.environ["QUERY_MAX_NEW_TOKENS"]),
    "response_max_new_tokens": int(os.environ["RESPONSE_MAX_NEW_TOKENS"]),
    "query_min_new_tokens": int(os.environ["QUERY_MIN_NEW_TOKENS"]),
    "response_min_new_tokens": int(os.environ["RESPONSE_MIN_NEW_TOKENS"]),
    "query_temperature": float(os.environ["QUERY_TEMPERATURE"]),
    "response_temperature": float(os.environ["RESPONSE_TEMPERATURE"]),
    "top_p": float(os.environ["TOP_P"]),
    "student_thinking_mode": "direct",
}
expected_model = {
    "identity": os.environ["MODEL"],
    "revision": os.environ["MODEL_REVISION"],
    "policy_hash": os.environ["POLICY_HASH"],
    "artifact_sha256": os.environ["MODEL_ARTIFACT_SHA256"],
}
root = Path(os.environ["SHARD_DIR"])
for index in range(shard_count):
    source = root / f"shard-{index}.jsonl"
    prediction = root / f"predictions-{index}.jsonl"
    manifest_path = root / f"predictions-{index}.manifest.json"
    for path in (source, prediction, manifest_path):
        if not path.is_file():
            raise SystemExit(f"missing required shard artifact: {path}")
    if file_hash(source) != prepared[index]["sha256"]:
        raise SystemExit(f"source shard {index} hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("command") != "bootstrap-rollouts":
        raise SystemExit(f"prediction shard {index} command mismatch")
    if manifest.get("rows") != prepared[index]["rows"] or manifest.get("candidates") != prepared[index]["rows"] * len(seeds):
        raise SystemExit(f"prediction shard {index} cardinality mismatch")
    if manifest.get("seeds") != seeds or manifest.get("dataset", {}).get("sha256") != prepared[index]["sha256"]:
        raise SystemExit(f"prediction shard {index} seed/data identity mismatch")
    if manifest.get("model") != expected_model:
        raise SystemExit(f"prediction shard {index} model identity mismatch")
    if manifest.get("prompt", {}).get("identity") != os.environ["PROMPT_VERSION"]:
        raise SystemExit(f"prediction shard {index} prompt identity mismatch")
    if manifest.get("evaluator_version") != os.environ["EVALUATOR_VERSION"]:
        raise SystemExit(f"prediction shard {index} evaluator identity mismatch")
    if manifest.get("generation") != generation:
        raise SystemExit(f"prediction shard {index} generation contract mismatch")
print("event=bootstrap_shard_manifests_validated shards=%d fallback_reason=none" % shard_count)
PY

log_event merge_start shards="$SHARD_COUNT" fallback_reason=none
uv run --frozen python -m text_feedback_dpo.cli merge-predictions \
  --shard-dir "$SHARD_DIR" --output "$OUTPUT" --shards "$SHARD_COUNT"

export MERGED_MANIFEST="${OUTPUT%.*}.manifest.json"
uv run --frozen python - <<'PY'
import hashlib, json, os
from pathlib import Path

def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

prepare = json.loads(Path(os.environ["PREPARE_MANIFEST"]).read_text(encoding="utf-8"))
root = Path(os.environ["SHARD_DIR"])
shards = []
for index in range(int(os.environ["SHARD_COUNT"])):
    prediction = root / f"predictions-{index}.jsonl"
    manifest = root / f"predictions-{index}.manifest.json"
    shards.append({
        "index": index,
        "prediction_path": str(prediction),
        "prediction_sha256": file_hash(prediction),
        "manifest_path": str(manifest),
        "manifest_sha256": file_hash(manifest),
    })
output = Path(os.environ["OUTPUT"])
manifest = {
    "status": "complete",
    "stage": "checkpoint_student_no_hint_bootstrap_merge",
    "commit_hash": os.environ["EXPECTED_COMMIT"],
    "prepare_manifest": {"path": os.environ["PREPARE_MANIFEST"], "sha256": os.environ["PREPARE_MANIFEST_SHA256"]},
    "pool": prepare["pool"],
    "model": {
        "identity": os.environ["MODEL"], "revision": os.environ["MODEL_REVISION"],
        "artifact_sha256": os.environ["MODEL_ARTIFACT_SHA256"], "policy_hash": os.environ["POLICY_HASH"],
    },
    "prompt_version": os.environ["PROMPT_VERSION"],
    "evaluator_version": os.environ["EVALUATOR_VERSION"],
    "seeds": [int(value) for value in os.environ["SEEDS"].split()],
    "shards": shards,
    "rows": prepare["selection"]["count"],
    "merged_path": str(output),
    "merged_sha256": file_hash(output),
    "fallback_reason": "none",
}
Path(os.environ["MERGED_MANIFEST"]).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(manifest, sort_keys=True))
PY
log_event merge_complete output="$OUTPUT" merged_sha256="$(sha256sum "$OUTPUT" | awk '{print $1}')" fallback_reason=none
