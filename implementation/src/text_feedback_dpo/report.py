from __future__ import annotations

import html
import json
import math
import re
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any


_SCALAR_METRIC_NAMES = {
    "answer_correct",
    "citation_coverage",
    "citation_precision",
    "citation_recall",
    "citation_validity",
    "correct",
    "duplicate_citation",
    "duplicate_citation_rate",
    "exact_match",
    "f1",
    "malformed_rate",
    "malformed_response",
    "parse_valid",
    "truncated",
    "truncation_known",
    "truncation_rate",
    "unsupported_source_rate",
    "valid_citation_rate",
}
_RECALL_METRIC_PATTERN = re.compile(r"(?:^|_)recall(?:@|_|$)")
_SECTION_TITLES = {
    "answer": "Answer accuracy",
    "citation": "Citation metrics",
    "length": "Length metrics",
    "other": "Other metrics",
    "retrieval": "Retrieval metrics",
    "structure": "Response structure metrics",
    "timing": "Timing metrics",
}


def _metric_path(path: tuple[str, ...]) -> str:
    return ".".join(path)


def _is_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _validate_metric_value(path: tuple[str, ...], value: object) -> None:
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"metric {_metric_path(path)} must be finite")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ValueError(f"metric {_metric_path(path)} has a non-string nested key")
            _validate_metric_value((*path, key), nested)
        return
    if isinstance(value, list):
        for index, nested in enumerate(value):
            _validate_metric_value((*path, str(index)), nested)
        return
    raise ValueError(f"metric {_metric_path(path)} has unsupported value type: {type(value).__name__}")


def _validate_numeric_tree(path: tuple[str, ...], value: object) -> None:
    if isinstance(value, Mapping):
        if not value:
            raise ValueError(f"metric {_metric_path(path)} must not be an empty mapping")
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ValueError(f"metric {_metric_path(path)} has a non-string nested key")
            _validate_numeric_tree((*path, key), nested)
        return
    if not _is_number(value):
        raise ValueError(f"metric {_metric_path(path)} must contain only finite numeric values")


def _is_scalar_metric(name: str) -> bool:
    lowered = name.lower()
    return name in _SCALAR_METRIC_NAMES or bool(_RECALL_METRIC_PATTERN.search(lowered)) or lowered in {
        "mrr",
        "mean_reciprocal_rank",
        "reciprocal_rank",
    }


def _validate_metric_shapes(metrics: Mapping[str, Any]) -> None:
    def walk(path: tuple[str, ...], value: object):
        yield path, value
        if isinstance(value, Mapping):
            for key, nested in value.items():
                yield from walk((*path, key), nested)

    for path, value in walk((), metrics):
        if not path:
            continue
        name = path[-1]
        if name == "report_type" and not isinstance(value, str):
            raise ValueError("metric report_type must be a string")
        if _is_scalar_metric(name) and not (
            _is_number(value) or isinstance(value, bool) or (name == "truncated" and value is None)
        ):
            raise ValueError(f"metric {_metric_path(path)} must be a finite scalar number or boolean")
        if name in {"answer_words", "reasoning_words"}:
            if isinstance(value, Mapping):
                _validate_numeric_tree(path, value)
            elif not _is_number(value):
                raise ValueError(f"metric {_metric_path(path)} must be a number or numeric summary mapping")
        if name in {"timing", "timings"}:
            _validate_numeric_tree(path, value)


def _flatten_metrics(value: object, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], object]]:
    if isinstance(value, Mapping):
        if not value:
            return [(path, value)] if path else []
        flattened: list[tuple[tuple[str, ...], object]] = []
        for key in sorted(value):
            flattened.extend(_flatten_metrics(value[key], (*path, key)))
        return flattened
    return [(path, value)]


def _section_for(path: tuple[str, ...]) -> str:
    name = path[-1].lower()
    joined = ".".join(path).lower()
    if name in {"exact_match", "f1", "answer_correct", "correct"}:
        return "answer"
    if name in {"parse_valid", "malformed_rate", "malformed_response", "error_code"}:
        return "structure"
    if (
        "citation" in name
        or "citation" in joined
        or name in {"unsupported_source_rate", "valid_citation_rate"}
        or ("support" in name and "answer" in name)
    ):
        return "citation"
    if "recall" in name or name in {"mrr", "mean_reciprocal_rank", "reciprocal_rank", "first_answer_rank"} or "retrieval" in joined:
        return "retrieval"
    if name in {"answer_words", "reasoning_words"} or "answer_words" in joined or "reasoning_words" in joined or "length" in name or "words" in name:
        return "length"
    if "truncat" in name or any(token in joined for token in ("timing", "latency", "duration", "throughput")):
        return "timing"
    return "other"


def _report_label(metrics: Mapping[str, Any]) -> str:
    report_type = metrics.get("report_type")
    if isinstance(report_type, str):
        normalized = report_type.casefold().replace("-", "_")
        if normalized in {"active_search", "cited_response", "cited_reasoning"}:
            return "Active-search metrics"
        if normalized in {"archival", "archival_short_answer", "short_answer"}:
            return "Archival short-answer baseline"
    leaf_names = {path[-1].casefold() for path, _value in _flatten_metrics(metrics) if path}
    if leaf_names & {"parse_valid", "malformed_response", "citation_precision", "valid_citation_rate"}:
        return "Active-search metrics"
    if "evidence_support" in leaf_names:
        return "Archival short-answer baseline"
    return "Metrics"


def _render_value(value: object) -> str:
    return html.escape(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False))


def _render_table(rows: Sequence[tuple[str, object]]) -> str:
    body = "".join(f"<tr><th>{html.escape(label)}</th><td>{_render_value(value)}</td></tr>" for label, value in rows)
    return f"<table><tr><th>Metric</th><th>Value</th></tr>{body}</table>"


def write_html_report(path: Path, metrics: dict, artifacts: list[str]) -> None:
    """Write a structured SearchQA report without dropping malformed metrics."""
    if not isinstance(path, Path):
        raise TypeError("report path must be a pathlib.Path")
    if not isinstance(metrics, Mapping):
        raise TypeError("metrics must be a mapping")
    if isinstance(artifacts, (str, bytes)) or not isinstance(artifacts, list) or not all(isinstance(item, str) for item in artifacts):
        raise TypeError("artifacts must be a list of strings")
    _validate_metric_value((), metrics)
    _validate_metric_shapes(metrics)

    section_rows: dict[str, list[tuple[str, object]]] = {section: [] for section in _SECTION_TITLES}
    for path_parts, value in _flatten_metrics(metrics):
        if not path_parts:
            continue
        section_rows[_section_for(path_parts)].append((_metric_path(path_parts), value))
    sections = [f"<h2>{html.escape(_report_label(metrics))}</h2>"]
    for section in ("answer", "structure", "retrieval", "citation", "length", "timing", "other"):
        rows = sorted(section_rows[section])
        if rows:
            sections.append(f"<h3>{html.escape(_SECTION_TITLES[section])}</h3>{_render_table(rows)}")
    links = "".join(f"<li><a href='{html.escape(item, quote=True)}'>{html.escape(item)}</a></li>" for item in artifacts)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(sections) + f"<h2>Artifacts</h2><ul>{links}</ul>"
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>SearchQA Research Report</title></head>"
        f"<body><h1>SearchQA Research Report</h1>{body}</body></html>",
        encoding="utf-8",
    )
