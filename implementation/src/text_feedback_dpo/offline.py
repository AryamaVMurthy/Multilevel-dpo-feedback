from __future__ import annotations

import json
from pathlib import Path
from typing import Callable


def load_or_build_rollouts(*, examples: list[dict], cache_path: Path, policy_hash: str, generate: Callable[[dict], dict]) -> list[dict]:
    cached: dict[str, dict] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            example_id = row.get("example_id")
            if not example_id:
                raise ValueError("cached rollout missing example_id")
            if row.get("policy_hash") != policy_hash:
                raise ValueError(f"cached rollout policy_hash mismatch for {example_id}: expected {policy_hash}, got {row.get('policy_hash')}")
            if example_id in cached:
                raise ValueError(f"duplicate cached rollout: {example_id}")
            cached[example_id] = row
    rows = []
    for example in examples:
        example_id = str(example["id"])
        if example_id in cached:
            rows.append(cached[example_id])
            continue
        generated = generate(example)
        if not isinstance(generated, dict) or not generated.get("response"):
            raise ValueError(f"generator returned no response for {example_id}")
        rows.append({"example_id": example_id, "policy_hash": policy_hash, **generated})
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return rows


def load_or_build_trajectories(*, examples: list[dict], cache_path: Path, policy_hash: str, generate: Callable[[list[dict]], list[dict]]) -> list[dict]:
    cached: dict[str, dict] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("policy_hash") != policy_hash:
                raise ValueError(f"cached trajectory policy_hash mismatch for {row.get('id')}: expected {policy_hash}, got {row.get('policy_hash')}")
            if row.get("id") in cached:
                raise ValueError(f"duplicate cached trajectory: {row.get('id')}")
            cached[str(row["id"])] = row
    missing = [example for example in examples if str(example["id"]) not in cached]
    if missing:
        generated = generate(missing)
        if len(generated) != len(missing):
            raise ValueError(f"trajectory generator cardinality mismatch: expected {len(missing)}, got {len(generated)}")
        for row in generated:
            if str(row.get("id")) not in {str(example["id"]) for example in missing}:
                raise ValueError(f"trajectory generator returned unexpected id: {row.get('id')}")
            cached[str(row["id"])] = {"policy_hash": policy_hash, **row}
    rows = [cached[str(example["id"])] for example in examples]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return rows
