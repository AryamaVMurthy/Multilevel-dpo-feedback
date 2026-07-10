#!/bin/bash
# Download a pinned Hugging Face source inside Slurm and persist only the small raw source.
#SBATCH -p u22
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

DATASET_NAME="${DATASET_NAME:?DATASET_NAME is required}"
DATASET_CONFIG="${DATASET_CONFIG:?DATASET_CONFIG is required}"
DATASET_REVISION="${DATASET_REVISION:?DATASET_REVISION is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
EXPECTED_TRAIN="${EXPECTED_TRAIN:?EXPECTED_TRAIN is required}"
EXPECTED_TEST="${EXPECTED_TEST:?EXPECTED_TEST is required}"
TURING_ACCOUNT="${TURING_ACCOUNT:?TURING_ACCOUNT is required}"

module load u22/cuda/12.4
export PATH="$HOME/.local/bin:$PATH"
if [[ ! -d /scratch || ! -w /scratch ]]; then
  echo "ERROR: /scratch is not writable; refusing home cache fallback" >&2
  exit 1
fi
SCRATCH_DIR="/scratch/$USER/text-feedback-dpo/${SLURM_JOB_ID}"
mkdir -p "$SCRATCH_DIR" "$OUTPUT_DIR"
export UV_CACHE_DIR="$SCRATCH_DIR/uv_cache"
export UV_PROJECT_ENVIRONMENT="$SCRATCH_DIR/project_venv"
export HF_HOME="$SCRATCH_DIR/hf_cache"
export HF_DATASETS_CACHE="$SCRATCH_DIR/hf_datasets"
export TRANSFORMERS_CACHE="$SCRATCH_DIR/hf_cache"
export UV_CONCURRENT_DOWNLOADS=1
export UV_CONCURRENT_BUILDS=1
export UV_CONCURRENT_INSTALLS=1
export UV_LINK_MODE=copy
cd "$PROJECT_DIR"

uv run --frozen python - <<'PY'
import json
import os
from pathlib import Path

from datasets import load_dataset

name = os.environ["DATASET_NAME"]
config = os.environ["DATASET_CONFIG"]
revision = os.environ["DATASET_REVISION"]
output = Path(os.environ["OUTPUT_DIR"])
expected = {"train": int(os.environ["EXPECTED_TRAIN"]), "test": int(os.environ["EXPECTED_TEST"])}
dataset = load_dataset(name, config, revision=revision)
for split, expected_count in expected.items():
    if split not in dataset:
        raise RuntimeError(f"pinned dataset is missing split: {split}")
    rows = [dict(row) for row in dataset[split]]
    if len(rows) != expected_count:
        raise RuntimeError(f"{split} count mismatch: expected {expected_count}, observed {len(rows)}")
    target = output / f"{split}.jsonl"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite source artifact: {target}")
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
metadata = {
    "dataset": name,
    "config": config,
    "revision": revision,
    "counts": expected,
}
(output / "source_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
