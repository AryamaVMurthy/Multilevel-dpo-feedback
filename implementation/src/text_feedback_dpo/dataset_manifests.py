from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any


def canonical_row_hash(row: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(row), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def materialize_preflight_subset(
    *,
    source_path: Any,
    output_path: Any,
    count: int,
    seed: int,
) -> dict[str, Any]:
    from pathlib import Path

    from text_feedback_dpo.io import append_jsonl_zst, read_jsonl_zst, write_json_atomic

    source = Path(source_path)
    output = Path(output_path)
    manifest_path = output.with_name(f"{output.name}.manifest.json")
    if count <= 0:
        raise ValueError("preflight subset count must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed <= 0:
        raise ValueError("preflight subset seed must be a positive integer")
    if output.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite preflight subset artifact: {output}")
    rows = read_jsonl_zst(source)
    if count > len(rows):
        raise ValueError(f"preflight subset count {count} exceeds source rows {len(rows)}")
    ids = [row.get("id") for row in rows]
    if any(not isinstance(row_id, str) or not row_id for row_id in ids):
        raise ValueError("preflight subset source contains a missing id")
    if len(set(ids)) != len(ids):
        raise ValueError("preflight subset source contains duplicate ids")
    ranked = sorted(
        range(len(rows)),
        key=lambda index: hashlib.sha256(f"{seed}:{ids[index]}".encode("utf-8")).hexdigest(),
    )
    selected_indices = set(ranked[:count])
    selected = [row for index, row in enumerate(rows) if index in selected_indices]
    for row in selected:
        append_jsonl_zst(output, row)
    selected_ids = [str(row["id"]) for row in selected]
    selection_sha256 = hashlib.sha256(
        json.dumps(selected_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema": "paper-preflight-subset-v1",
        "source_path": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "seed": seed,
        "count": count,
        "selected_ids": selected_ids,
        "selection_sha256": selection_sha256,
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def _normalized_question(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value).lower())
    return " ".join(text.split())


def _question_value(row: Mapping[str, Any]) -> object:
    return row.get("question", row.get("problem", ""))


def _annotate(row: Mapping[str, Any], *, source_split: str, source_index: int) -> dict[str, Any]:
    output = dict(row)
    output["source_split"] = source_split
    output["source_index"] = source_index
    output["source_key"] = f"{source_split}:{row.get('id', source_index)}"
    output["row_hash"] = canonical_row_hash(row)
    output["normalized_question"] = _normalized_question(_question_value(row))
    if not output["normalized_question"]:
        raise ValueError(f"{source_split}:{source_index} has an empty normalized question")
    return output


def split_gsm8k_train(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: int,
    validation_count: int,
) -> dict[str, list[dict[str, Any]]]:
    raw_rows = list(rows)
    if not raw_rows:
        raise ValueError("GSM8K source train must not be empty")
    if validation_count <= 0 or validation_count >= len(raw_rows):
        raise ValueError("GSM8K validation_count must be between one and source_count minus one")
    annotated = [_annotate(row, source_split="train", source_index=index) for index, row in enumerate(raw_rows)]
    ordered = sorted(
        annotated,
        key=lambda row: hashlib.sha256(f"{seed}:gsm8k:{row['row_hash']}".encode("utf-8")).hexdigest(),
    )
    validation_keys = {row["source_key"] for row in ordered[:validation_count]}
    train_rows = []
    validation_rows = []
    for row in sorted(annotated, key=lambda item: int(item["source_index"])):
        role = "validation" if row["source_key"] in validation_keys else "train"
        row = {**row, "dataset_role": role, "stratum": "gsm8k:all"}
        (validation_rows if role == "validation" else train_rows).append(row)
    return {"train": train_rows, "validation": validation_rows}


def split_gsm8k_validation_roles(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: int,
    tune_count: int,
) -> dict[str, list[dict[str, Any]]]:
    raw_rows = [dict(row) for row in rows]
    if not raw_rows or tune_count <= 0 or tune_count >= len(raw_rows):
        raise ValueError("GSM8K tune_count must be between one and validation_count minus one")
    if any(not row.get("source_key") or not row.get("row_hash") for row in raw_rows):
        raise ValueError("GSM8K validation rows must be annotated before nested splitting")
    ordered = sorted(
        raw_rows,
        key=lambda row: hashlib.sha256(f"{seed}:gsm8k:validation:{row['row_hash']}".encode("utf-8")).hexdigest(),
    )
    tune_keys = {row["source_key"] for row in ordered[:tune_count]}
    tune = [{**row, "dataset_role": "validation_tune"} for row in raw_rows if row["source_key"] in tune_keys]
    confirm = [{**row, "dataset_role": "validation_confirm"} for row in raw_rows if row["source_key"] not in tune_keys]
    if {row["source_key"] for row in tune} & {row["source_key"] for row in confirm}:
        raise RuntimeError("GSM8K nested validation roles overlap")
    return {"tune": tune, "confirm": confirm}


def _math_stratum(row: Mapping[str, Any]) -> str:
    subject = str(row.get("source_subject", "")).strip()
    level = row.get("difficulty_level")
    if not subject or isinstance(level, bool) or not isinstance(level, int):
        raise ValueError("MATH row is missing subject or difficulty level")
    return f"math:{subject}:level{level}"


def _hash_order(rows: Iterable[dict[str, Any]], *, seed: int, salt: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{seed}:{salt}:{row['row_hash']}".encode("utf-8")
        ).hexdigest(),
    )


def split_math_train(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: int,
    primary_levels: tuple[int, ...] = (4, 5),
) -> dict[str, list[dict[str, Any]]]:
    """Create deterministic 90/10 train/validation roles within MATH subject-level strata."""

    raw_rows = list(rows)
    if not raw_rows:
        raise ValueError("MATH source train must not be empty")
    if primary_levels != (4, 5):
        raise ValueError("MATH primary levels must be exactly (4, 5)")
    annotated = [_annotate(row, source_split="train", source_index=index) for index, row in enumerate(raw_rows)]
    primary = [row for row in annotated if int(row.get("difficulty_level", 0)) in primary_levels]
    if not primary:
        raise ValueError("MATH source train has no Levels 4-5 rows")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in primary:
        groups[_math_stratum(row)].append(row)
    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for stratum in sorted(groups):
        ordered = _hash_order(groups[stratum], seed=seed, salt=f"math:validation:{stratum}")
        if len(ordered) < 2:
            raise ValueError(f"MATH stratum {stratum} has fewer than two rows")
        validation_count = min(len(ordered) - 1, max(1, round(len(ordered) * 0.10)))
        validation_keys = {row["source_key"] for row in ordered[:validation_count]}
        for row in sorted(ordered, key=lambda item: int(item["source_index"])):
            role = "validation" if row["source_key"] in validation_keys else "train"
            item = {**row, "dataset_role": role, "stratum": stratum}
            (validation_rows if role == "validation" else train_rows).append(item)
    return {"train": train_rows, "validation": validation_rows}


def split_math_validation_roles(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Partition every primary MATH validation stratum into two-thirds tune / one-third confirm."""

    raw_rows = [dict(row) for row in rows]
    if not raw_rows:
        raise ValueError("MATH validation rows must not be empty")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        if row.get("dataset_role") != "validation" or not row.get("source_key") or not row.get("row_hash"):
            raise ValueError("MATH validation rows must be annotated validation rows")
        groups[_math_stratum(row)].append(row)
    tune: list[dict[str, Any]] = []
    confirm: list[dict[str, Any]] = []
    for stratum in sorted(groups):
        ordered = _hash_order(groups[stratum], seed=seed, salt=f"math:tune:{stratum}")
        if len(ordered) < 2:
            raise ValueError(f"MATH validation stratum {stratum} has fewer than two rows")
        tune_count = min(len(ordered) - 1, max(1, round(len(ordered) * (2 / 3))))
        tune_keys = {row["source_key"] for row in ordered[:tune_count]}
        for row in sorted(ordered, key=lambda item: int(item["source_index"])):
            item = {**row, "dataset_role": "validation_tune" if row["source_key"] in tune_keys else "validation_confirm"}
            (tune if item["dataset_role"] == "validation_tune" else confirm).append(item)
    return {"tune": tune, "confirm": confirm}


def _answer_text(row: Mapping[str, Any]) -> str:
    answers = row.get("answers")
    if isinstance(answers, list) and answers:
        return str(answers[0])
    return str(row.get("gold_answer", row.get("answer", "")))


def _context_text(row: Mapping[str, Any]) -> str:
    context = row.get("context")
    if context:
        return str(context)
    evidence = row.get("evidence")
    if isinstance(evidence, list):
        return " ".join(str(item) for item in evidence)
    return ""


def _bucket(value: int, values: list[int]) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    rank = sum(item < value for item in ordered)
    return min(3, 4 * rank // len(ordered))


def _stratify(rows: list[dict[str, Any]], source_split: str) -> dict[str, list[dict[str, Any]]]:
    answer_lengths = [len(_answer_text(row).split()) for row in rows]
    context_lengths = [len(_context_text(row).split()) for row in rows]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, answer_length, context_length in zip(rows, answer_lengths, context_lengths):
        stratum = f"{source_split}:answer{_bucket(answer_length, answer_lengths)}:context{_bucket(context_length, context_lengths)}"
        groups[stratum].append({**row, "stratum": stratum})
    return groups


def _ordered_group_rows(groups: dict[str, list[dict[str, Any]]], *, seed: int, salt: str) -> list[dict[str, Any]]:
    for stratum, group in groups.items():
        group.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}:{salt}:{stratum}:{row['row_hash']}".encode("utf-8")
            ).hexdigest()
        )
    result: list[dict[str, Any]] = []
    group_names = sorted(groups)
    position = 0
    while group_names:
        name = group_names[position % len(group_names)]
        result.append(groups[name].pop(0))
        if not groups[name]:
            group_names.remove(name)
            if group_names:
                position %= len(group_names)
        else:
            position += 1
    return result


def _sample_source(
    rows: Iterable[Mapping[str, Any]],
    *,
    source_split: str,
    main_count: int,
    auxiliary_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rows = list(rows)
    requested = main_count + auxiliary_count
    if requested > len(raw_rows):
        raise ValueError(
            f"{source_split} requested count {requested} exceeds source count {len(raw_rows)}"
        )
    annotated = [_annotate(row, source_split=source_split, source_index=index) for index, row in enumerate(raw_rows)]
    ordered = _ordered_group_rows(_stratify(annotated, source_split), seed=seed, salt=source_split)
    main = [{**row, "dataset_role": source_split} for row in ordered[:main_count]]
    auxiliary_role = f"hparam_{source_split}"
    auxiliary = [{**row, "dataset_role": auxiliary_role} for row in ordered[main_count:requested]]
    return main, auxiliary


def sample_searchqa8k(
    train: Iterable[Mapping[str, Any]],
    validation: Iterable[Mapping[str, Any]],
    test: Iterable[Mapping[str, Any]],
    *,
    seed: int,
    counts: Mapping[str, int],
    auxiliary_counts: Mapping[str, int],
) -> dict[str, list[dict[str, Any]]]:
    required_counts = {"train", "validation", "test"}
    if set(counts) != required_counts:
        raise ValueError("SearchQA counts must contain train, validation, and test")
    if set(auxiliary_counts) != {"train", "validation"}:
        raise ValueError("SearchQA auxiliary_counts must contain train and validation")
    train_rows, hparam_train = _sample_source(
        train,
        source_split="train",
        main_count=int(counts["train"]),
        auxiliary_count=int(auxiliary_counts["train"]),
        seed=seed,
    )
    validation_rows, hparam_validation = _sample_source(
        validation,
        source_split="validation",
        main_count=int(counts["validation"]),
        auxiliary_count=int(auxiliary_counts["validation"]),
        seed=seed,
    )
    test_rows, _ = _sample_source(
        test,
        source_split="test",
        main_count=int(counts["test"]),
        auxiliary_count=0,
        seed=seed,
    )
    result = {
        "train": train_rows,
        "validation": validation_rows,
        "test": test_rows,
        "hparam_train": hparam_train,
        "hparam_validation": hparam_validation,
    }
    validate_disjoint_splits([row for split_rows in result.values() for row in split_rows])
    return result


def validate_disjoint_splits(rows: Iterable[Mapping[str, Any]]) -> None:
    seen_source_keys: dict[str, str] = {}
    seen_questions: dict[str, str] = {}
    seen_hashes: dict[str, str] = {}
    for row in rows:
        source_key = str(row.get("source_key", ""))
        role = str(row.get("dataset_role", ""))
        normalized_question = str(row.get("normalized_question") or _normalized_question(_question_value(row)))
        row_hash = str(row.get("row_hash", ""))
        if not source_key:
            raise ValueError("manifest row is missing source_key")
        if not role:
            raise ValueError(f"manifest row {source_key} is missing dataset_role")
        if source_key in seen_source_keys:
            raise ValueError(f"source_key appears more than once: {source_key}")
        if normalized_question in seen_questions:
            raise ValueError(
                f"normalized question appears across dataset roles: {normalized_question}"
            )
        if row_hash and row_hash in seen_hashes:
            raise ValueError(f"row_hash appears more than once: {row_hash}")
        seen_source_keys[source_key] = role
        seen_questions[normalized_question] = role
        if row_hash:
            seen_hashes[row_hash] = role


def _write_compressed_jsonl(path: Any, rows: Iterable[Mapping[str, Any]]) -> None:
    try:
        import zstandard
    except ImportError as exc:
        raise ImportError("zstandard is required to write paper manifest artifacts") from exc
    compressor = zstandard.ZstdCompressor(level=3)
    with path.open("wb") as handle:
        with compressor.stream_writer(handle) as writer:
            for row in rows:
                line = json.dumps(dict(row), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
                writer.write((line + "\n").encode("utf-8"))


def write_manifest_bundle(
    output_dir: Any,
    splits: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    metadata: Mapping[str, Any],
    nested_roles: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    materialized = {role: [dict(row) for row in rows] for role, rows in splits.items()}
    validate_disjoint_splits([row for rows in materialized.values() for row in rows])
    roles = {role: len(rows) for role, rows in sorted(materialized.items())}
    row_hashes = {
        role: [str(row["row_hash"]) for row in rows]
        for role, rows in sorted(materialized.items())
    }
    nested_materialized = {
        role: [dict(row) for row in rows]
        for role, rows in sorted((nested_roles or {}).items())
    }
    if nested_materialized:
        if set(nested_materialized) != {"tune", "confirm"}:
            raise ValueError("nested_roles must contain exactly tune and confirm")
        if "validation" not in materialized:
            raise ValueError("nested_roles requires a primary validation role")
        validation_keys = {str(row["source_key"]) for row in materialized["validation"]}
        nested_keys = [str(row["source_key"]) for rows in nested_materialized.values() for row in rows]
        if len(nested_keys) != len(set(nested_keys)) or set(nested_keys) != validation_keys:
            raise ValueError("nested validation roles must partition primary validation rows")
    nested_counts = {role: len(rows) for role, rows in nested_materialized.items()}
    nested_hashes = {
        role: [str(row["row_hash"]) for row in rows]
        for role, rows in nested_materialized.items()
    }
    payload = {
        "metadata": dict(metadata),
        "roles": roles,
        "row_hashes": row_hashes,
        "nested_roles": nested_counts,
        "nested_row_hashes": nested_hashes,
    }
    manifest_path = output_path / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("content") != payload:
            raise ValueError("manifest.json already exists with different content")
        return existing
    for role, rows in materialized.items():
        artifact_path = output_path / f"{role}.jsonl.zst"
        if artifact_path.exists():
            raise ValueError(f"refusing to overwrite existing manifest artifact: {artifact_path}")
        _write_compressed_jsonl(artifact_path, rows)
    for role, rows in nested_materialized.items():
        artifact_path = output_path / f"validation_{role}.jsonl.zst"
        if artifact_path.exists():
            raise ValueError(f"refusing to overwrite existing nested manifest artifact: {artifact_path}")
        _write_compressed_jsonl(artifact_path, rows)
    manifest = {
        "schema": "paper-dataset-manifest-v1",
        "content": payload,
        "metadata": dict(metadata),
        "roles": roles,
        "row_hashes": row_hashes,
        "nested_roles": nested_counts,
        "nested_row_hashes": nested_hashes,
        "content_sha256": hashlib.sha256(
            json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _read_json_rows(path: Any) -> list[dict[str, Any]]:
    from pathlib import Path

    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        rows = []
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {source}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row at {source}:{line_number} must be an object")
            rows.append(row)
        return rows
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"JSON source must contain a list of objects: {source}")
    return list(value)


def _source_directory_sha256(path: Any) -> str:
    from pathlib import Path

    root = Path(path)
    digest = hashlib.sha256()
    for file in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(file.relative_to(root)).encode("utf-8"))
        digest.update(hashlib.sha256(file.read_bytes()).digest())
    return digest.hexdigest()


def materialize_paper_dataset(config: Any, source_path: Any, output_dir: Any) -> dict[str, Any]:
    from pathlib import Path

    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"paper dataset source does not exist: {source}")
    dataset = config.dataset
    nested_roles: dict[str, list[dict[str, Any]]] | None = None
    if dataset.name == "gsm8k":
        if not source.is_dir():
            raise ValueError("GSM8K materialization source must be a directory containing train and test JSON files")
        source_files = {file.stem.lower(): file for file in source.iterdir() if file.suffix.lower() in {".json", ".jsonl"}}
        if "train" not in source_files or "test" not in source_files:
            raise ValueError("GSM8K source directory must contain train.json/jsonl and test.json/jsonl")
        from text_feedback_dpo.benchmarks import convert_gsm8k_row

        raw_train = _read_json_rows(source_files["train"])
        raw_test = _read_json_rows(source_files["test"])
        if len(raw_train) != dataset.source_counts["train"] or len(raw_test) != dataset.source_counts["test"]:
            raise ValueError("GSM8K source counts do not match the frozen paper config")
        converted_train = [convert_gsm8k_row(row, index=index) for index, row in enumerate(raw_train)]
        converted_test = [convert_gsm8k_row(row, index=index) for index, row in enumerate(raw_test)]
        split_rows = split_gsm8k_train(
            converted_train,
            seed=dataset.seed,
            validation_count=dataset.splits["validation"],
        )
        split_rows["test"] = [
            {**_annotate(row, source_split="test", source_index=index), "dataset_role": "test", "stratum": "gsm8k:all"}
            for index, row in enumerate(converted_test)
        ]
        nested_roles = split_gsm8k_validation_roles(
            split_rows["validation"],
            seed=dataset.seed,
            tune_count=dataset.validation_roles["tune"],
        )
        metadata = {
            "dataset": dataset.name,
            "source": dataset.source,
            "revision": dataset.revision,
            "source_artifact_sha256": _source_directory_sha256(source),
            "seed": dataset.seed,
        }
    elif dataset.name == "searchqa8k":
        from text_feedback_dpo.searchqa import load_original_searchqa

        loaded = load_original_searchqa(source, expected_counts=dataset.source_counts)
        split_rows = sample_searchqa8k(
            loaded["splits"]["train"],
            loaded["splits"]["validation"],
            loaded["splits"]["test"],
            seed=dataset.seed,
            counts=dataset.splits,
            auxiliary_counts=dataset.auxiliary_hparam,
        )
        metadata = {
            "dataset": dataset.name,
            "source": dataset.source,
            "revision": dataset.revision,
            "source_artifact_sha256": loaded["artifact_sha256"],
            "seed": dataset.seed,
        }
    elif dataset.name == "math":
        if not source.is_dir():
            raise ValueError("MATH materialization source must be a directory of official subject subdirectories")
        from text_feedback_dpo.benchmarks import MATH_SUBJECTS, convert_math_row

        if tuple(dataset.subjects) != MATH_SUBJECTS:
            raise ValueError("MATH config subjects do not match the supported official snapshot")
        raw_train: list[dict[str, Any]] = []
        raw_test: list[dict[str, Any]] = []
        for subject in MATH_SUBJECTS:
            subject_dir = source / subject
            if not subject_dir.is_dir():
                raise ValueError(f"MATH source is missing subject directory: {subject}")
            source_files = {
                file.stem.lower(): file
                for file in subject_dir.iterdir()
                if file.suffix.lower() in {".json", ".jsonl"}
            }
            if set(source_files) != {"train", "test"}:
                raise ValueError(f"MATH subject {subject} must contain exactly train and test JSON/JSONL files")
            for index, row in enumerate(_read_json_rows(source_files["train"])):
                raw_train.append(convert_math_row(row, subject=subject, source_split="train", index=index))
            for index, row in enumerate(_read_json_rows(source_files["test"])):
                raw_test.append(convert_math_row(row, subject=subject, source_split="test", index=index))
        if len(raw_train) != dataset.source_counts["train"] or len(raw_test) != dataset.source_counts["test"]:
            raise ValueError("MATH source counts do not match the frozen paper config")
        split_rows = split_math_train(
            raw_train,
            seed=dataset.seed,
            primary_levels=dataset.primary_levels,
        )
        split_rows["test"] = [
            {
                **_annotate(row, source_split="test", source_index=index),
                "dataset_role": "test",
                "stratum": _math_stratum(row),
            }
            for index, row in enumerate(raw_test)
        ]
        nested_roles = split_math_validation_roles(split_rows["validation"], seed=dataset.seed)
        metadata = {
            "dataset": dataset.name,
            "source": dataset.source,
            "revision": dataset.revision,
            "source_artifact_sha256": _source_directory_sha256(source),
            "seed": dataset.seed,
            "subjects": list(dataset.subjects),
            "primary_levels": list(dataset.primary_levels),
            "train_fraction": dataset.train_fraction,
            "validation_tune_fraction": dataset.validation_tune_fraction,
        }
    else:
        raise ValueError(f"unsupported paper dataset: {dataset.name}")
    manifest = write_manifest_bundle(
        output_dir,
        split_rows,
        metadata=metadata,
        nested_roles=nested_roles,
    )
    return {"manifest": manifest, "output_dir": str(output_dir)}
