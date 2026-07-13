#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from text_feedback_dpo.audit import audit_trajectories, write_trajectory_audit
from text_feedback_dpo.io import iter_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonically validate and audit SearchQA trajectories")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--trajectories", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--sibling-seeds", required=True, type=int, nargs="+")
    args = parser.parse_args()

    result = audit_trajectories(
        list(iter_jsonl(args.data)),
        list(iter_jsonl(args.trajectories)),
        sibling_seeds=args.sibling_seeds,
    )
    paths = write_trajectory_audit(result, output_prefix=args.output_prefix)
    print(" ".join(f"{name}={path}" for name, path in sorted(paths.items())))


if __name__ == "__main__":
    main()
