from __future__ import annotations

import json
from pathlib import Path


def load_named_metrics(specs: list[str]) -> dict[str, dict]:
    if not specs:
        raise ValueError("at least one named metrics file is required")
    result = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"comparison run must be NAME=METRICS_PATH: {spec}")
        name, raw_path = spec.split("=", 1)
        if not name or not raw_path:
            raise ValueError(f"comparison run must have non-empty name and path: {spec}")
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"metrics file does not exist for {name}: {path}")
        if name in result:
            raise ValueError(f"duplicate comparison run name: {name}")
        result[name] = json.loads(path.read_text(encoding="utf-8"))
    return result


def comparison_metrics(specs: list[str]) -> dict:
    runs = load_named_metrics(specs)
    return {"methods": runs, "method_count": len(runs), "selection_metric": "exact_match"}
