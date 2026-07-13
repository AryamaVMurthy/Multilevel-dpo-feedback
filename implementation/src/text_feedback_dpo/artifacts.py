from __future__ import annotations

import json
from pathlib import Path


def validate_artifacts(directory: Path) -> dict:
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("max_length") != 4096:
        raise ValueError("manifest max_length must be exactly 4096")
    for required in manifest.get("required_files", []):
        if not (directory / required).exists():
            raise ValueError(f"required artifact is missing: {required}")
    return {"valid": True, "manifest": manifest}
