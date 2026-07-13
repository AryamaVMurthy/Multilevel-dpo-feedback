#!/bin/bash
# Run on the login node only after invalid jobs are canceled.
set -euo pipefail

: "${INVALID_RUN_ROOT:?INVALID_RUN_ROOT must be supplied}"
: "${ARCHIVE_DIR:?ARCHIVE_DIR must be supplied}"
EXPECTED_INVALID_RUN_ROOT="$HOME/searchqa-dpo/20260713"
actual="$(realpath -m "$INVALID_RUN_ROOT")"
expected="$(realpath -m "$EXPECTED_INVALID_RUN_ROOT")"
[[ "$actual" == "$expected" ]] || { echo "ERROR: refusing cleanup outside $expected: $actual" >&2; exit 2; }
[[ -d "$actual" ]] || { echo "ERROR: invalid run root does not exist: $actual" >&2; exit 2; }
if squeue -h -u "$USER" | grep -q .; then
  echo "ERROR: refusing retirement while this user has active Slurm jobs" >&2
  exit 2
fi

mkdir -p "$ARCHIVE_DIR"
du -sh "$actual" > "$ARCHIVE_DIR/disk-before.txt"
find "$actual" -type f -print | sort > "$ARCHIVE_DIR/files-before.txt"
find "$actual" -type f \( -name '*manifest*.json' -o -name '*metrics*.json' -o -name '*.log' \) -exec cp --parents {} "$ARCHIVE_DIR" \;
find "$actual" -type f -name '*predictions*.jsonl' -exec sh -c 'for file do sed -n "1,3p" "$file"; done' sh {} + > "$ARCHIVE_DIR/representative-predictions.jsonl"

find "$actual" -type f \( -name '*predictions*.jsonl' -o -name '*trajectories*.jsonl' -o -name '*preferences*.jsonl' -o -name '*sft*.jsonl' \) -delete
find "$actual" -mindepth 1 -depth -type d -empty -delete
du -sh "$actual" > "$ARCHIVE_DIR/disk-after.txt"
