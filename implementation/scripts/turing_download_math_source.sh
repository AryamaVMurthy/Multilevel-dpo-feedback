#!/bin/bash
# Download the pinned seven-subject official MATH snapshot under Slurm.
#SBATCH -p u22
#SBATCH -n 2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

CONFIG="${CONFIG:?CONFIG is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
TURING_ACCOUNT="${TURING_ACCOUNT:?TURING_ACCOUNT is required}"
export CONFIG OUTPUT_DIR

module load u22/cuda/12.4
export PATH="$HOME/.local/bin:$PATH"
if [[ ! -d /scratch || ! -w /scratch ]]; then
  echo "ERROR: /scratch is not writable; refusing home cache fallback" >&2
  exit 1
fi
if [[ -e "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "ERROR: refusing non-empty MATH source output directory: $OUTPUT_DIR" >&2
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

uv run --no-project --with 'datasets==5.0.0' --with 'pyyaml==6.0.3' python - <<'PY'
import json
import os
from pathlib import Path

import yaml
from datasets import load_dataset

config = yaml.safe_load(Path(os.environ["CONFIG"]).read_text(encoding="utf-8"))
dataset = config["dataset"]
if dataset["name"] != "math":
    raise RuntimeError("MATH downloader requires dataset.name=math")
if dataset["source"] != "EleutherAI/hendrycks_math":
    raise RuntimeError("MATH downloader requires the pinned official EleutherAI source")
subjects = dataset["subjects"]
expected = dataset["source_counts"]
output = Path(os.environ["OUTPUT_DIR"])
observed = {"train": 0, "test": 0}
for subject in subjects:
    source = load_dataset(dataset["source"], subject, revision=dataset["revision"])
    subject_output = output / subject
    subject_output.mkdir()
    for split in ("train", "test"):
        if split not in source:
            raise RuntimeError(f"{subject} is missing split: {split}")
        rows = [dict(row) for row in source[split]]
        observed[split] += len(rows)
        target = subject_output / f"{split}.jsonl"
        with target.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
if observed != {"train": int(expected["train"]), "test": int(expected["test"])}:
    raise RuntimeError(f"MATH count mismatch: expected={expected}, observed={observed}")
metadata = {
    "dataset": dataset["source"],
    "revision": dataset["revision"],
    "subjects": subjects,
    "counts": observed,
}
(output / "source_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
