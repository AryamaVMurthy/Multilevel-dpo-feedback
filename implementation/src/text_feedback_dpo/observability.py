from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonlLogger:
    def __init__(self, path: Path, run_id: str) -> None:
        self.path = path
        self.run_id = run_id
        self.start_ns = time.monotonic_ns()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, event_name: str, **fields: Any) -> None:
        elapsed_ms = (time.monotonic_ns() - self.start_ns) // 1_000_000
        payload = {
            "event_name": event_name,
            "run_id": self.run_id,
            "elapsed_ms": elapsed_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": fields.pop("status", "ok"),
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def failure(self, *, stage: str, error_code: str, message: str, **fields: Any) -> None:
        self.event(
            "failure",
            status="error",
            stage=stage,
            error_code=error_code,
            message=message,
            **fields,
        )
