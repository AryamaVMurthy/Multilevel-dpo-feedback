from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def _require_mapping(manifest: dict, field: str, required: tuple[str, ...]) -> dict:
    value = manifest.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"active manifest {field} must be a mapping")
    missing = [name for name in required if name not in value]
    if missing:
        raise ValueError(f"active manifest {field} is missing: {missing[0]}")
    return value


def _require_nonempty_strings(section_name: str, section: dict, fields: tuple[str, ...]) -> None:
    for field in fields:
        if not isinstance(section[field], str) or not section[field].strip():
            raise ValueError(f"active manifest {section_name}.{field} must be a non-empty string")


def _validate_sha256(section_name: str, value: object) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"active manifest {section_name}.sha256 must be a lowercase SHA-256")


def _validate_identity_hash(section_name: str, section: dict) -> None:
    declared = section["sha256"]
    _validate_sha256(section_name, declared)
    identity = {key: value for key, value in section.items() if key != "sha256"}
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != declared:
        raise ValueError(f"active manifest {section_name} identity hash mismatch")


def _parse_jsonl_objects(path: Path, *, require_id: bool = False) -> list[dict]:
    rows: list[dict] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"artifact JSONL contains a blank line: {path.name}:{line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"artifact contains invalid JSON: {path.name}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"artifact JSONL row must be a JSON object: {path.name}:{line_number}")
        row_id = row.get("id", row.get("example_id"))
        if require_id and (not isinstance(row_id, str) or not row_id.strip()):
            raise ValueError(f"active prediction JSONL row requires a non-empty id: {path.name}:{line_number}")
        if row_id is not None and (not isinstance(row_id, str) or not row_id.strip()):
            raise ValueError(f"artifact JSONL row identity must be a non-empty string: {path.name}:{line_number}")
        if row_id is not None and row_id in seen_ids:
            raise ValueError(f"artifact JSONL contains duplicate id: {row_id}")
        if row_id is not None:
            seen_ids.add(row_id)
        rows.append(row)
    return rows


def _validate_active_manifest(directory: Path, manifest: dict) -> None:
    model = _require_mapping(manifest, "model", ("identity", "revision", "policy_hash"))
    dataset = _require_mapping(manifest, "dataset", ("source", "revision", "sha256"))
    source_schema = _require_mapping(manifest, "source_schema", ("identity", "version", "sha256"))
    retrieval = _require_mapping(manifest, "retrieval", ("identity", "schema_version", "requested_top_k", "k1", "b"))
    prompt = _require_mapping(manifest, "prompt", ("identity", "sha256"))
    response = _require_mapping(manifest, "response", ("identity", "schema_version", "sha256"))
    generation = _require_mapping(manifest, "generation", ("context_budget", "query_max_new_tokens", "response_max_new_tokens"))
    _require_mapping(manifest, "timing", ("pipeline_wall_ms",))
    _require_nonempty_strings("model", model, ("identity", "revision", "policy_hash"))
    _require_nonempty_strings("dataset", dataset, ("source", "revision"))
    _validate_sha256("dataset", dataset["sha256"])
    _validate_identity_hash("source_schema", source_schema)
    _validate_identity_hash("prompt", prompt)
    _validate_identity_hash("response", response)
    if source_schema["identity"] != "searchqa.search_results.v1" or source_schema["version"] != 1:
        raise ValueError("active manifest source schema identity/version mismatch")
    if retrieval != {"identity": "fixed_bm25", "schema_version": 1, "requested_top_k": 8, "k1": 1.2, "b": 0.75}:
        raise ValueError("active manifest retrieval identity or frozen configuration mismatch")
    if prompt["identity"] != "fixed-retrieval-cited-v1":
        raise ValueError("active manifest prompt identity must be fixed-retrieval-cited-v1")
    if response["identity"] != "cited-response" or response["schema_version"] != 1:
        raise ValueError("active manifest response identity/schema mismatch")
    if generation["context_budget"] != 4096:
        raise ValueError("active manifest generation context_budget must be exactly 4096")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("active manifest artifacts must be a nonempty list")
    manifest_rows = manifest.get("rows")
    if isinstance(manifest_rows, bool) or not isinstance(manifest_rows, int) or manifest_rows <= 0:
        raise ValueError("active manifest rows must be a positive integer")
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            raise ValueError(f"active manifest artifact {index} must be a mapping")
        missing = [field for field in ("path", "format", "rows", "bytes", "sha256") if field not in artifact]
        if missing:
            raise ValueError(f"active manifest artifact {index} is missing: {missing[0]}")
        if artifact["format"] != "jsonl":
            raise ValueError(f"active manifest artifact {index} must use supported format jsonl")
        if isinstance(artifact["rows"], bool) or not isinstance(artifact["rows"], int) or artifact["rows"] != manifest_rows:
            raise ValueError(f"manifest rows does not match artifact row count declaration: {artifact['path']}")
        relative = Path(artifact["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"active manifest artifact {index} path must stay within the run directory")
        path = directory / relative
        if not path.is_file():
            raise ValueError(f"required artifact is missing: {artifact['path']}")
        if path.stat().st_size != artifact["bytes"]:
            raise ValueError(f"artifact bytes mismatch: {artifact['path']}")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != artifact["sha256"]:
            raise ValueError(f"artifact sha256 mismatch: {artifact['path']}")
        parsed_rows = _parse_jsonl_objects(path, require_id=True)
        row_count = len(parsed_rows)
        if row_count != artifact["rows"]:
            raise ValueError(f"artifact row count mismatch: {artifact['path']}")
        if row_count != manifest_rows:
            raise ValueError(f"manifest rows does not match artifact row count: {artifact['path']}")


def validate_artifacts(directory: Path) -> dict:
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("max_length") != 4096:
        raise ValueError("manifest max_length must be exactly 4096")
    required_files = manifest.get("required_files")
    if not isinstance(required_files, list) or not required_files or not all(isinstance(item, str) and item for item in required_files):
        raise ValueError("manifest required_files must be a nonempty list of paths")
    for required in required_files:
        required_path = directory / required
        if not required_path.exists():
            raise ValueError(f"required artifact is missing: {required}")
        if required_path.suffix == ".jsonl":
            _parse_jsonl_objects(required_path)
    if manifest.get("command") == "generate-searchqa":
        _validate_active_manifest(directory, manifest)
    return {"valid": True, "manifest": manifest}
