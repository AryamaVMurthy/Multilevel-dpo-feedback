from __future__ import annotations

import hashlib
import json
import io
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from text_feedback_dpo.searchqa import (
    SOURCE_SCHEMA,
    SOURCE_SCHEMA_VERSION,
    NoUsableSearchQASourcesError,
    materialize_row,
    pack_evidence,
)
from text_feedback_dpo.prompts import build_student_prompt


SFT_MAX_LENGTH = 4096


class SFTDataGateError(ValueError):
    """Raised when canonical Task 6 SFT coverage cannot satisfy a configured gate."""

    def __init__(self, message: str, report: dict):
        self.report = report
        super().__init__(message)


def dataset_fingerprint(rows: list[dict]) -> str:
    payload = "\n".join(json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_searchqa_split(source: str, split: str, *, revision: str, limit: int | None = None) -> list[dict]:
    rows, _ = load_searchqa_split_with_stats(source, split, revision=revision, limit=limit)
    return rows


def _new_stream_stats() -> dict:
    return {
        "source_rows": 0,
        "materialized_rows": 0,
        "dropped_rows": 0,
        "drop_reasons": {},
        "source_records": _empty_source_record_stats(),
    }


def stream_stats_report(stats: dict) -> dict:
    return {
        "source_schema": SOURCE_SCHEMA,
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "source_rows": stats["source_rows"],
        "materialized_rows": stats["materialized_rows"],
        "dropped_rows": stats["dropped_rows"],
        "drop_reasons": dict(stats["drop_reasons"]),
        "source_records": {
            **stats["source_records"],
            "drop_reasons": dict(stats["source_records"]["drop_reasons"]),
        },
    }


def _record_dropped_row(stats: dict, exc: NoUsableSearchQASourcesError) -> None:
    _accumulate_source_record_stats(stats["source_records"], exc.source_filter_stats)
    stats["dropped_rows"] += 1
    stats["drop_reasons"]["no_usable_evidence"] = stats["drop_reasons"].get("no_usable_evidence", 0) + 1


def stream_searchqa_split_with_stats(
    source: str,
    split: str,
    *,
    revision: str,
    limit: int | None = None,
) -> tuple[Iterable[dict], dict]:
    """Stream materialized SearchQA rows while retaining only compact load statistics."""
    if not source:
        raise ValueError("dataset source is required")

    stats = _new_stream_stats()

    def rows() -> Iterable[dict]:
        if source == "kyunghyuncho/search_qa":
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise ImportError("huggingface_hub is required for the official SearchQA archive") from exc
            filename = {"train": "data/train_test_val/train.zip", "validation": "data/train_test_val/val.zip", "test": "data/train_test_val/test.zip"}.get(split)
            if filename is None:
                raise ValueError("official SearchQA split must be train, validation, or test")
            archive_path = hf_hub_download(repo_id="kyunghyuncho/search_qa", filename=filename, repo_type="dataset", revision=revision)
            with zipfile.ZipFile(archive_path) as archive:
                for index, member in enumerate(sorted(name for name in archive.namelist() if name.endswith(".json"))):
                    if limit is not None and stats["materialized_rows"] >= limit:
                        break
                    stats["source_rows"] += 1
                    with archive.open(member) as handle:
                        raw = json.loads(io.TextIOWrapper(handle, encoding="utf-8").read())
                    try:
                        row = materialize_row(raw, split=split, index=index)
                    except NoUsableSearchQASourcesError as exc:
                        _record_dropped_row(stats, exc)
                        continue
                    _accumulate_source_record_stats(stats["source_records"], row["source_filter_stats"])
                    stats["materialized_rows"] += 1
                    yield row
        else:
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise ImportError("datasets is required for SearchQA materialization") from exc
            dataset = load_dataset(source, split=split, revision=revision)
            for index, raw in enumerate(dataset):
                if limit is not None and stats["materialized_rows"] >= limit:
                    break
                stats["source_rows"] += 1
                try:
                    row = materialize_row(raw, split=split, index=index)
                except NoUsableSearchQASourcesError as exc:
                    _record_dropped_row(stats, exc)
                    continue
                _accumulate_source_record_stats(stats["source_records"], row["source_filter_stats"])
                stats["materialized_rows"] += 1
                yield row
        if stats["materialized_rows"] == 0:
            raise ValueError(f"SearchQA split {split!r} produced zero rows")

    return rows(), stats


def load_searchqa_split_with_stats(source: str, split: str, *, revision: str, limit: int | None = None) -> tuple[list[dict], dict]:
    rows, stats = stream_searchqa_split_with_stats(source, split, revision=revision, limit=limit)
    materialized = list(rows)
    return materialized, stream_stats_report(stats)


def _load_official_searchqa_zip(split: str, revision: str, limit: int | None) -> tuple[list[dict], dict]:
    rows, stats = stream_searchqa_split_with_stats("kyunghyuncho/search_qa", split, revision=revision, limit=limit)
    materialized = list(rows)
    return materialized, stream_stats_report(stats)


def _empty_source_record_stats() -> dict:
    return {"input_records": 0, "usable_records": 0, "dropped_records": 0, "drop_reasons": {}}


def _accumulate_source_record_stats(total: dict, row_stats: dict | None) -> None:
    if row_stats is None:
        raise ValueError("SearchQA source filtering failed without source_filter_stats")
    required = {"input_records", "usable_records", "dropped_records", "drop_reasons"}
    if not isinstance(row_stats, dict) or set(row_stats) != required or not isinstance(row_stats["drop_reasons"], dict):
        raise ValueError("SearchQA source_filter_stats has an invalid schema")
    for field in ("input_records", "usable_records", "dropped_records"):
        value = row_stats[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"SearchQA source_filter_stats.{field} must be a nonnegative integer")
        total[field] += value
    if row_stats["input_records"] != row_stats["usable_records"] + row_stats["dropped_records"]:
        raise ValueError("SearchQA source_filter_stats counts do not balance")
    for reason, count in row_stats["drop_reasons"].items():
        if not isinstance(reason, str) or not reason or isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("SearchQA source_filter_stats.drop_reasons is invalid")
        total["drop_reasons"][reason] = total["drop_reasons"].get(reason, 0) + count


def _load_stats(
    source_rows: int,
    rows: list[dict],
    dropped_rows: int,
    drop_reasons: dict[str, int],
    source_record_stats: dict,
) -> dict:
    return {
        "source_schema": SOURCE_SCHEMA,
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "source_rows": source_rows,
        "materialized_rows": len(rows),
        "dropped_rows": dropped_rows,
        "drop_reasons": drop_reasons,
        "source_records": source_record_stats,
    }


def attach_evidence(rows: list[dict], *, max_evidence_tokens: int, token_count) -> list[dict]:
    if max_evidence_tokens <= 0:
        raise ValueError("max_evidence_tokens must be positive")
    output = []
    for row in rows:
        enriched = dict(row)
        enriched["packed_evidence"] = pack_evidence(row["snippets"], max_tokens=max_evidence_tokens, token_count=token_count)
        enriched["prompt"] = build_student_prompt(enriched, [])
        output.append(enriched)
    return output


def build_sft_rows(rows: list[dict]) -> list[dict]:
    del rows
    raise RuntimeError(
        "removed unsafe SFT path: gold-answer targets are forbidden; "
        "use build_sft_rows_from_trajectories"
    )


def build_sft_row(row: dict) -> dict:
    del row
    raise RuntimeError(
        "removed unsafe SFT path: gold-answer targets are forbidden; "
        "use build_sft_rows_from_trajectories"
    )


def _structured_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _record_exclusion(report: dict, example_id: str, task: str, reason: str) -> None:
    report["exclusions"].append({"id": example_id, "task": task, "reason": reason})
    report["exclusion_counts"][reason] = report["exclusion_counts"].get(reason, 0) + 1


def _candidate_base_reason(candidate: Mapping[str, object]) -> str | None:
    if candidate.get("provenance") != "student":
        return "teacher_provenance"
    if candidate.get("teacher_output") is True:
        return "teacher_target"
    if candidate.get("fabricated") is True:
        return "fabricated_target"
    if candidate.get("no_hint") is not True:
        return "hinted_prompt"
    if candidate.get("verified_no_hint_success") is not True:
        return "unverified_no_hint_success"
    if candidate.get("truncation") != {"query": False, "response": False}:
        return "truncated_target"
    score = candidate.get("cited_score")
    if not isinstance(score, Mapping) or score.get("correct") is not True:
        return "unverified_no_hint_success"
    return None


def _validate_query_candidate(candidate: Mapping[str, object], trajectory: Mapping[str, object]) -> str | None:
    reason = _candidate_base_reason(candidate)
    if reason:
        return reason
    prompt = candidate.get("query_prompt")
    trajectory_prompt = trajectory.get("query_prompt")
    if not isinstance(prompt, str) or not prompt.strip() or "Hints:" in prompt:
        return "hinted_prompt"
    if prompt != trajectory_prompt:
        return "query_prompt_identity_mismatch"
    if candidate.get("query_prompt_hash") != _structured_hash(prompt):
        return "query_prompt_hash_mismatch"
    raw_query = candidate.get("raw_query")
    if not isinstance(raw_query, str) or not raw_query.strip() or raw_query[0].isspace():
        return "invalid_student_query"
    return None


def _validate_response_candidate(candidate: Mapping[str, object]) -> str | None:
    reason = _candidate_base_reason(candidate)
    if reason:
        return reason
    prompt = candidate.get("response_prompt")
    if not isinstance(prompt, str) or not prompt.strip() or "Hints:" in prompt:
        return "hinted_prompt"
    if candidate.get("response_prompt_hash") != _structured_hash(prompt):
        return "response_prompt_hash_mismatch"
    ranked = candidate.get("canonical_ranked_search_results")
    if not isinstance(ranked, list) or not ranked:
        return "missing_canonical_retrieval_context"
    if candidate.get("retrieval_context_hash") != _structured_hash(ranked):
        return "retrieval_context_hash_mismatch"
    raw_response = candidate.get("raw_response")
    if not isinstance(raw_response, str) or not raw_response.strip() or raw_response[0].isspace():
        return "invalid_student_response"
    return None


def _completion(value: object, *, task: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{task} completion is empty")
    if value[0].isspace():
        raise ValueError(f"{task} completion has an ambiguous leading boundary")
    return f" {value}"


def _token_ids(tokenizer: object, text: str) -> Sequence:
    if callable(tokenizer):
        encoded = tokenizer(text=text, truncation=False).get("input_ids")
    elif hasattr(tokenizer, "encode"):
        encoded = tokenizer.encode(text, add_special_tokens=False)
    else:
        raise TypeError("Task 7 length validation requires a tokenizer with encode or call support")
    if isinstance(encoded, dict):
        encoded = encoded.get("input_ids")
    if not isinstance(encoded, Sequence) or isinstance(encoded, (str, bytes)):
        raise TypeError("tokenizer must return an input_ids sequence without truncation")
    return encoded


def _sft_combined_token_count(tokenizer: object, prompt: str, completion: str) -> int:
    eos = getattr(tokenizer, "eos_token", None)
    if not isinstance(eos, str) or not eos:
        raise ValueError("Task 7 SFT pinned tokenizer requires a non-empty eos_token")
    rendered_completion = completion if completion.endswith(eos) else completion + eos
    prompt_ids = _token_ids(tokenizer, prompt)
    combined_ids = _token_ids(tokenizer, prompt + rendered_completion)
    if combined_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Task 7 SFT tokenizer boundary mismatch between prompt and prompt+completion")
    return len(combined_ids)


def _select_task7_candidate(trajectory: Mapping[str, object]) -> Mapping[str, object] | None:
    siblings = trajectory.get("no_hint_siblings")
    candidates: list[Mapping[str, object]] = []
    if isinstance(siblings, list):
        candidates.extend(item for item in siblings if isinstance(item, Mapping))
    chosen = trajectory.get("chosen")
    if not candidates and isinstance(chosen, Mapping):
        candidates.append(chosen)
    if not candidates:
        return None
    verified = [item for item in candidates if item.get("verified_no_hint_success") is True]
    if not verified:
        return candidates[0]
    return sorted(verified, key=lambda item: (-float(item.get("future_sibling_gain", 1.0)), int(item.get("seed", 0))))[0]


def _canonical_validation_reason(candidate: Mapping[str, object], example: Mapping[str, object]) -> tuple[str, str] | None:
    """Re-run Task 6's canonical validator against dataset-owned truth.

    Trajectory flags and persisted hashes are provenance claims, not authority.  The
    validator recomputes prompts, retrieval, scores, and rendered responses from the
    example's question, gold answer, and source records.
    """
    from text_feedback_dpo.trajectories import TrajectoryError, validate_active_artifact

    try:
        validate_active_artifact(candidate, example=example, hints=[])
    except (TrajectoryError, ValueError, TypeError) as exc:
        return "canonical_artifact_validation_failed", str(exc)
    return None


def build_sft_rows_from_trajectories(
    trajectories: list[dict],
    *,
    examples: Mapping[str, Mapping[str, object]] | None = None,
    tokenizer: object | None = None,
    max_length: int = SFT_MAX_LENGTH,
    min_coverage: float = 0.0,
    min_rows: int = 0,
) -> tuple[list[dict], dict]:
    """Build Task 7 query/response rows from canonical, verified Task 6 artifacts.

    The function deliberately does not derive targets from gold answers, hints, or teacher
    output.  Every exclusion is recorded so a launcher can apply a visible coverage gate.
    """
    if examples is None:
        raise ValueError("Task 7 SFT requires dataset-owned examples for canonical artifact validation")
    if tokenizer is None:
        raise ValueError("Task 7 SFT requires a pinned tokenizer for no-truncation validation")
    if not isinstance(examples, Mapping):
        raise TypeError("Task 7 dataset examples must be a mapping keyed by trajectory id")
    if max_length != SFT_MAX_LENGTH:
        raise ValueError("Task 7 SFT max_length must remain exactly 4096")
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError("min_coverage must be between zero and one")
    if min_rows < 0:
        raise ValueError("min_rows must be nonnegative")
    report = {
        "input_trajectories": len(trajectories),
        "eligible_trajectories": 0,
        "query_rows": 0,
        "response_rows": 0,
        "query_coverage": 0.0,
        "response_coverage": 0.0,
        "exclusions": [],
        "exclusion_counts": {},
        "max_length": max_length,
        "length_validation": "pinned_tokenizer_no_truncation",
    }
    output: list[dict] = []
    for index, trajectory in enumerate(trajectories):
        example_id = str(trajectory.get("id", f"row-{index}")) if isinstance(trajectory, Mapping) else f"row-{index}"
        if not isinstance(trajectory, Mapping):
            for task in ("query", "response"):
                _record_exclusion(report, example_id, task, "trajectory_not_mapping")
            continue
        example = examples.get(example_id)
        if not isinstance(example, Mapping):
            for task in ("query", "response"):
                _record_exclusion(report, example_id, task, "missing_dataset_example")
            continue
        if trajectory.get("training_eligible") is not True:
            for task in ("query", "response"):
                _record_exclusion(report, example_id, task, "trajectory_not_training_eligible")
            continue
        candidate = _select_task7_candidate(trajectory)
        if candidate is None:
            for task in ("query", "response"):
                _record_exclusion(report, example_id, task, "missing_verified_no_hint_candidate")
            continue
        base_reason = _candidate_base_reason(candidate)
        if base_reason is not None:
            for task in ("query", "response"):
                _record_exclusion(report, example_id, task, base_reason)
            continue
        query_reason = _validate_query_candidate(candidate, trajectory)
        response_reason = _validate_response_candidate(candidate)
        if query_reason is not None or response_reason is not None:
            if query_reason is not None:
                _record_exclusion(report, example_id, "query", query_reason)
            if response_reason is not None:
                _record_exclusion(report, example_id, "response", response_reason)
            continue
        canonical_failure = _canonical_validation_reason(candidate, example)
        if canonical_failure is not None:
            reason, detail = canonical_failure
            for task in ("query", "response"):
                _record_exclusion(report, example_id, task, reason)
            report["exclusions"][-1]["detail"] = detail
            report["exclusions"][-2]["detail"] = detail
            continue
        if query_reason is None and response_reason is None:
            report["eligible_trajectories"] += 1
        for task, reason, source_field in (
            ("query", query_reason, "raw_query"),
            ("response", response_reason, "raw_response"),
        ):
            if reason is not None:
                _record_exclusion(report, example_id, task, reason)
                continue
            prompt_field = "query_prompt" if task == "query" else "response_prompt"
            try:
                completion = _completion(candidate[source_field], task=task)
            except ValueError:
                _record_exclusion(report, example_id, task, f"invalid_student_{task}")
                continue
            combined_tokens = _sft_combined_token_count(
                tokenizer, str(candidate[prompt_field]), completion,
            )
            if combined_tokens > max_length:
                _record_exclusion(report, example_id, task, "combined_token_length_exceeds_max_length")
                continue
            metadata = {
                "trajectory_id": example_id,
                "task": task,
                "provenance": "student",
                "no_hint": True,
                "verified_no_hint_success": True,
                "query_prompt_hash": candidate["query_prompt_hash"],
                "response_prompt_hash": candidate.get("response_prompt_hash"),
                "retrieval_context_hash": candidate.get("retrieval_context_hash"),
            }
            row = {"id": f"{example_id}::sft::{task}", "task": task, "prompt": candidate[prompt_field], "completion": completion, "metadata": metadata}
            if task == "response":
                row["visible_response"] = candidate["raw_response"]
            output.append(row)
            report[f"{task}_rows"] += 1
    denominator = max(1, report["input_trajectories"])
    report["query_coverage"] = report["query_rows"] / denominator
    report["response_coverage"] = report["response_rows"] / denominator
    if report["query_coverage"] < min_coverage or report["response_coverage"] < min_coverage or len(output) < min_rows:
        raise SFTDataGateError(
            "Task 7 SFT coverage gate failed; inspect exclusion_counts and remediation before relaunch",
            report,
        )
    return output, report


build_task7_sft_rows = build_sft_rows_from_trajectories


def select_balanced_sft_rows(
    rows: Iterable[Mapping[str, object]], *, per_task: int, seed: int,
) -> list[dict]:
    """Select equal query/response subsets without duplicating or inventing targets."""
    if isinstance(per_task, bool) or not isinstance(per_task, int) or per_task <= 0:
        raise ValueError("balanced SFT per_task must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("balanced SFT seed must be an integer")
    grouped: dict[str, list[Mapping[str, object]]] = {"query": [], "response": []}
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"balanced SFT row {index} must be an object")
        row_id = row.get("id")
        task = row.get("task")
        metadata = row.get("metadata")
        if not isinstance(row_id, str) or not row_id.strip() or row_id in seen:
            raise ValueError(f"balanced SFT row {index} requires a unique non-empty id")
        seen.add(row_id)
        if task not in grouped:
            raise ValueError(f"balanced SFT row {row_id} requires task=query or task=response")
        if (
            not isinstance(row.get("prompt"), str)
            or not str(row["prompt"]).strip()
            or not isinstance(row.get("completion"), str)
            or not str(row["completion"]).strip()
        ):
            raise ValueError(f"balanced SFT row {row_id} requires non-empty prompt and completion")
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("provenance") != "student"
            or metadata.get("no_hint") is not True
        ):
            raise ValueError(f"balanced SFT row {row_id} must be student-generated no-hint supervision")
        grouped[str(task)].append(row)
    for task, candidates in grouped.items():
        if len(candidates) < per_task:
            raise ValueError(f"balanced SFT requires {per_task} {task} rows but only {len(candidates)} are available")

    def key(row: Mapping[str, object]) -> tuple[str, str]:
        row_id = str(row["id"])
        digest = hashlib.sha256(f"{seed}\0{row_id}".encode()).hexdigest()
        return digest, row_id

    selected = {task: sorted(candidates, key=key)[:per_task] for task, candidates in grouped.items()}
    output: list[dict] = []
    for index in range(per_task):
        output.extend((dict(selected["query"][index]), dict(selected["response"][index])))
    return output


def split_paired_sft_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    eval_pairs: int,
    min_train_pairs: int,
    seed: int,
) -> tuple[list[dict], list[dict], dict]:
    """Create balanced, trajectory-disjoint train/eval SFT pairs without duplication."""
    for label, value in (("eval_pairs", eval_pairs), ("min_train_pairs", min_train_pairs)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"paired SFT {label} must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("paired SFT seed must be an integer")
    grouped: dict[str, dict[str, dict]] = {}
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"paired SFT row {index} must be an object")
        row_id = row.get("id")
        task = row.get("task")
        metadata = row.get("metadata")
        if not isinstance(row_id, str) or not row_id or row_id in seen_ids:
            raise ValueError(f"paired SFT row {index} requires a unique non-empty ID")
        seen_ids.add(row_id)
        if task not in {"query", "response"}:
            raise ValueError(f"paired SFT row {row_id} requires task=query or task=response")
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("provenance") != "student"
            or metadata.get("no_hint") is not True
        ):
            raise ValueError(f"paired SFT row {row_id} must be student-generated no-hint supervision")
        trajectory_id = metadata.get("trajectory_id")
        if not isinstance(trajectory_id, str) or not trajectory_id:
            raise ValueError(f"paired SFT row {row_id} requires a trajectory_id")
        if (
            not isinstance(row.get("prompt"), str)
            or not str(row["prompt"]).strip()
            or not isinstance(row.get("completion"), str)
            or not str(row["completion"]).strip()
        ):
            raise ValueError(f"paired SFT row {row_id} requires non-empty prompt/completion")
        tasks = grouped.setdefault(trajectory_id, {})
        if task in tasks:
            raise ValueError(f"paired SFT trajectory {trajectory_id} contains duplicate {task} rows")
        tasks[str(task)] = dict(row)

    paired = [trajectory_id for trajectory_id, tasks in grouped.items() if set(tasks) == {"query", "response"}]
    required_pairs = eval_pairs + min_train_pairs
    if len(paired) < required_pairs:
        raise ValueError(
            f"paired SFT requires at least {required_pairs} paired trajectories "
            f"({eval_pairs} eval + {min_train_pairs} train), got {len(paired)}"
        )

    def key(trajectory_id: str) -> tuple[str, str]:
        return hashlib.sha256(f"{seed}\0{trajectory_id}".encode()).hexdigest(), trajectory_id

    ordered = sorted(paired, key=key)
    eval_ids = ordered[:eval_pairs]
    train_ids = ordered[eval_pairs:]

    def flatten(trajectory_ids: Sequence[str]) -> list[dict]:
        output: list[dict] = []
        for trajectory_id in trajectory_ids:
            output.extend((grouped[trajectory_id]["query"], grouped[trajectory_id]["response"]))
        return output

    train = flatten(train_ids)
    evaluation = flatten(eval_ids)
    report = {
        "input_rows": len(seen_ids),
        "input_trajectories": len(grouped),
        "paired_trajectories": len(paired),
        "excluded_unpaired_trajectories": len(grouped) - len(paired),
        "train_pairs": len(train_ids),
        "train_rows": len(train),
        "eval_pairs": len(eval_ids),
        "eval_rows": len(evaluation),
        "seed": seed,
        "assignment": "lowest_sha256_seed_nul_trajectory_id_to_eval_v1",
        "train_trajectory_ids_sha256": hashlib.sha256(
            json.dumps(train_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "eval_trajectory_ids_sha256": hashlib.sha256(
            json.dumps(eval_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "trajectory_overlap": 0,
    }
    return train, evaluation, report


def split_balanced_dpo_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    train_per_task: int,
    eval_per_task: int,
    seed: int,
) -> tuple[list[dict], list[dict], dict]:
    """Select exact balanced DPO subsets whose examples never cross splits.

    Preference rows are selected without replacement.  Example IDs are assigned as
    whole groups before rows are selected, so a query and response preference from
    the same example can never leak between train and evaluation.
    """
    for label, value in (("train_per_task", train_per_task), ("eval_per_task", eval_per_task)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"balanced DPO {label} must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("balanced DPO seed must be an integer")

    grouped: dict[str, dict[str, list[dict]]] = {}
    validated_rows: list[dict] = []
    seen_ids: set[str] = set()
    task_counts = {"query": 0, "response": 0}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"balanced DPO row {index} must be an object")
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id.strip() or row_id in seen_ids:
            raise ValueError(f"balanced DPO row {index} requires a unique non-empty id")
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"balanced DPO row {row_id} requires metadata")
        pair_type = metadata.get("pair_type")
        if pair_type not in {"query", "response"}:
            raise ValueError(f"balanced DPO row {row_id} metadata.pair_type must be query or response")
        example_id = metadata.get("example_id")
        if not isinstance(example_id, str) or not example_id.strip():
            raise ValueError(f"balanced DPO row {row_id} metadata.example_id must be a non-empty string")
        copied = dict(row)
        validated_rows.append(copied)
        seen_ids.add(row_id)
        task_counts[pair_type] += 1
        grouped.setdefault(example_id, {"query": [], "response": []})[pair_type].append(copied)

    required_total = train_per_task + eval_per_task
    for task in ("query", "response"):
        if task_counts[task] < required_total:
            raise ValueError(
                f"balanced DPO requires {required_total} rows for each task "
                f"({task} has {task_counts[task]})"
            )

    def row_key(row: Mapping[str, object]) -> tuple[str, str]:
        row_id = str(row["id"])
        return hashlib.sha256(f"{seed}\0{row_id}".encode("utf-8")).hexdigest(), row_id

    def example_key(example_id: str) -> tuple[str, str]:
        return hashlib.sha256(f"{seed}\0{example_id}".encode("utf-8")).hexdigest(), example_id

    # Choose the eval examples with a deterministic bounded-capacity subset search.
    # The selected eval capacity must leave enough rows for both train tasks.
    ordered_examples = sorted(grouped, key=example_key)
    capacities = {
        example_id: (
            len(grouped[example_id]["query"]),
            len(grouped[example_id]["response"]),
        )
        for example_id in ordered_examples
    }
    lower = (eval_per_task, eval_per_task)
    upper = (task_counts["query"] - train_per_task, task_counts["response"] - train_per_task)
    suffix = [(0, 0)] * (len(ordered_examples) + 1)
    for index in range(len(ordered_examples) - 1, -1, -1):
        query_count, response_count = capacities[ordered_examples[index]]
        remaining_query, remaining_response = suffix[index + 1]
        suffix[index] = (query_count + remaining_query, response_count + remaining_response)

    states: dict[tuple[int, int], tuple[str, ...]] = {(0, 0): ()}
    for index, example_id in enumerate(ordered_examples):
        query_count, response_count = capacities[example_id]
        next_states: dict[tuple[int, int], tuple[str, ...]] = {}
        for (current_query, current_response), selected_ids in states.items():
            options = (
                (current_query + query_count, current_response + response_count, selected_ids + (example_id,)),
                (current_query, current_response, selected_ids),
            )
            for next_query, next_response, next_ids in options:
                if next_query > upper[0] or next_response > upper[1]:
                    continue
                remaining_query, remaining_response = suffix[index + 1]
                if next_query + remaining_query < lower[0] or next_response + remaining_response < lower[1]:
                    continue
                next_states.setdefault((next_query, next_response), next_ids)
        states = next_states

    valid_assignments = [
        selected_ids
        for (query_count, response_count), selected_ids in states.items()
        if lower[0] <= query_count <= upper[0] and lower[1] <= response_count <= upper[1]
    ]
    if not valid_assignments:
        raise ValueError(
            "balanced DPO cannot make train/eval example-disjoint selections with "
            f"{train_per_task} train and {eval_per_task} eval rows per task"
        )
    eval_example_ids = min(
        valid_assignments,
        key=lambda selected_ids: (
            hashlib.sha256(json.dumps(selected_ids, separators=(",", ":")).encode("utf-8")).hexdigest(),
            selected_ids,
        ),
    )
    eval_examples = set(eval_example_ids)

    def select_rows(example_ids: Iterable[str], task: str, count: int) -> list[dict]:
        candidates = [
            row
            for example_id in example_ids
            for row in grouped[example_id][task]
        ]
        candidates.sort(key=row_key)
        if len(candidates) < count:
            raise ValueError(
                "balanced DPO cannot make train/eval example-disjoint selections "
                f"with {count} {task} rows from the assigned examples"
            )
        return candidates[:count]

    eval_by_task = {
        task: select_rows(eval_examples, task, eval_per_task)
        for task in ("query", "response")
    }
    train_by_task = {
        task: select_rows((example_id for example_id in grouped if example_id not in eval_examples), task, train_per_task)
        for task in ("query", "response")
    }

    def interleave(by_task: Mapping[str, list[dict]]) -> list[dict]:
        output: list[dict] = []
        for index in range(max(len(by_task["query"]), len(by_task["response"]))):
            for task in ("query", "response"):
                if index < len(by_task[task]):
                    output.append(by_task[task][index])
        return output

    train = interleave(train_by_task)
    evaluation = interleave(eval_by_task)
    train_example_ids = {str(row["metadata"]["example_id"]) for row in train}
    eval_example_ids_selected = {str(row["metadata"]["example_id"]) for row in evaluation}
    example_overlap = train_example_ids & eval_example_ids_selected
    train_row_ids = {str(row["id"]) for row in train}
    eval_row_ids = {str(row["id"]) for row in evaluation}

    def hash_ids(ids: set[str]) -> str:
        return hashlib.sha256(
            json.dumps(sorted(ids), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    report = {
        "input_rows": len(validated_rows),
        "input_rows_sha256": dataset_fingerprint(sorted(validated_rows, key=lambda row: str(row["id"]))),
        "input_examples": len(grouped),
        "train_per_task": train_per_task,
        "eval_per_task": eval_per_task,
        "train_rows": len(train),
        "eval_rows": len(evaluation),
        "train_task_counts": {task: len(train_by_task[task]) for task in ("query", "response")},
        "eval_task_counts": {task: len(eval_by_task[task]) for task in ("query", "response")},
        "train_rows_sha256": dataset_fingerprint(train),
        "eval_rows_sha256": dataset_fingerprint(evaluation),
        "train_example_count": len(train_example_ids),
        "eval_example_count": len(eval_example_ids_selected),
        "train_example_ids_sha256": hash_ids(train_example_ids),
        "eval_example_ids_sha256": hash_ids(eval_example_ids_selected),
        "task_counts": {
            "train": {task: len(train_by_task[task]) for task in ("query", "response")},
            "eval": {task: len(eval_by_task[task]) for task in ("query", "response")},
        },
        "example_overlap": len(example_overlap),
        "example_overlap_count": len(example_overlap),
        "example_overlap_ids_sha256": hash_ids(example_overlap),
        "row_id_overlap_count": len(train_row_ids & eval_row_ids),
        "seed": seed,
        "selection_policy": "sha256(seed\\0,example_id)-bounded-capacity-dp-v1",
    }
    return train, evaluation, report


def build_sft_rows_from_bootstrap(
    bootstrap_rows: list[dict],
    *,
    examples: Mapping[str, Mapping[str, object]],
    tokenizer: object,
    max_length: int = SFT_MAX_LENGTH,
) -> tuple[list[dict], dict]:
    """Select query and response supervision independently from no-hint rollouts."""
    from text_feedback_dpo.bootstrap import validate_bootstrap_rows

    if max_length != SFT_MAX_LENGTH:
        raise ValueError("bootstrap SFT max_length must remain exactly 4096")
    ordered_examples: list[Mapping[str, object]] = []
    expected_seeds: list[int] | None = None
    for row in bootstrap_rows:
        example_id = row.get("id")
        example = examples.get(str(example_id))
        if not isinstance(example, Mapping):
            raise ValueError(f"bootstrap SFT is missing canonical example {example_id}")
        seeds = row.get("seeds")
        if not isinstance(seeds, list) or any(not isinstance(seed, int) for seed in seeds):
            raise ValueError(f"bootstrap SFT row {example_id} has invalid seeds")
        if expected_seeds is None:
            expected_seeds = seeds
        elif seeds != expected_seeds:
            raise ValueError("bootstrap SFT rows must share one exact seed list")
        ordered_examples.append(example)
    validated = validate_bootstrap_rows(
        bootstrap_rows,
        examples=ordered_examples,
        expected_seeds=expected_seeds or [],
    )
    report = {
        "input_examples": len(validated),
        "query_rows": 0,
        "response_rows": 0,
        "query_unique_examples": 0,
        "response_unique_examples": 0,
        "query_exclusion_counts": {},
        "response_exclusion_counts": {},
        "max_length": max_length,
    }
    output: list[dict] = []

    def exclude(task: str, reason: str) -> None:
        counts = report[f"{task}_exclusion_counts"]
        counts[reason] = counts.get(reason, 0) + 1

    for row in validated:
        query_candidates: list[tuple[int, int, Mapping[str, object]]] = []
        response_candidates: list[tuple[int, Mapping[str, object]]] = []
        for candidate in row["candidates"]:
            artifact = candidate["artifact"]
            seed = int(candidate["seed"])
            truncation = artifact["truncation"]
            first_rank = artifact["retrieval_metrics"]["first_answer_rank"]
            raw_query = artifact.get("raw_query")
            if truncation["query"] or not isinstance(raw_query, str) or not raw_query.strip():
                exclude("query", "query_invalid_or_truncated")
            elif first_rank is None:
                exclude("query", "query_answer_recall_missing")
            else:
                query_candidates.append((int(first_rank), seed, artifact))
            score = artifact["cited_score"]
            response_ok = (
                not truncation["query"]
                and not truncation["response"]
                and isinstance(score, Mapping)
                and score.get("correct") is True
                and score.get("parse_valid") is True
                and score.get("answer_correct") is True
                and score.get("lexical_cited_answer_support") == 1.0
            )
            if response_ok:
                response_candidates.append((seed, artifact))
            else:
                exclude("response", "response_not_verified_correct")

        selected = {
            "query": min(query_candidates, default=None, key=lambda item: (item[0], item[1])),
            "response": min(response_candidates, default=None, key=lambda item: item[0]),
        }
        for task, choice in selected.items():
            if choice is None:
                continue
            seed = choice[1] if task == "query" else choice[0]
            artifact = choice[2] if task == "query" else choice[1]
            prompt_field = "query_prompt" if task == "query" else "response_prompt"
            completion_field = "raw_query" if task == "query" else (
                "generated_response" if artifact.get("response_prefix") == "Answer: " else "raw_response"
            )
            completion = _completion(artifact[completion_field], task=task)
            if _sft_combined_token_count(tokenizer, str(artifact[prompt_field]), completion) > max_length:
                exclude(task, "combined_token_length_exceeds_max_length")
                continue
            sft_row = {
                "id": f"{row['id']}::sft::{task}",
                "task": task,
                "prompt": artifact[prompt_field],
                "completion": completion,
                "metadata": {
                    "trajectory_id": row["id"],
                    "seed": seed,
                    "provenance": "student",
                    "no_hint": True,
                    "verified_no_hint_success": True,
                    "query_prompt_hash": artifact["query_prompt_hash"],
                    "response_prompt_hash": artifact.get("response_prompt_hash"),
                    "retrieval_context_hash": artifact.get("retrieval_context_hash"),
                },
            }
            if task == "response":
                sft_row["visible_response"] = artifact["raw_response"]
            output.append(sft_row)
            report[f"{task}_rows"] += 1
            report[f"{task}_unique_examples"] += 1
    return output, report


def build_rl_rows_from_trajectories(
    trajectories: list[dict],
    *,
    examples: Mapping[str, Mapping[str, object]],
    tokenizer: object,
    max_length: int = SFT_MAX_LENGTH,
) -> tuple[list[dict], dict]:
    """Build task-tagged GRPO/DAPO rows from the same verified student artifacts.

    Query rows retain the complete fixed source corpus so the reward can rerun BM25 on
    the generated query.  Response rows retain the canonical retrieved context used by
    Task 6.  Both tasks therefore share provenance and split discipline with SFT/DPO.
    """
    sft_rows, report = build_sft_rows_from_trajectories(
        trajectories, examples=examples, tokenizer=tokenizer, max_length=max_length,
    )
    by_id = {str(trajectory["id"]): trajectory for trajectory in trajectories if isinstance(trajectory, Mapping) and "id" in trajectory}
    rows: list[dict] = []
    for sft_row in sft_rows:
        example_id = str(sft_row["metadata"]["trajectory_id"])
        trajectory = by_id[example_id]
        candidate = _select_task7_candidate(trajectory)
        if candidate is None:
            raise RuntimeError(f"Task 7 RL builder lost canonical candidate for {example_id}")
        example = examples[example_id]
        task = sft_row["task"]
        rows.append({
            "id": sft_row["id"].replace("::sft::", "::rl::"),
            "task": task,
            "prompt": sft_row["prompt"],
            "gold_answer": example["gold_answer"],
            "sources": json.loads(json.dumps(example["sources"], ensure_ascii=False, sort_keys=True)),
            "canonical_ranked_search_results": json.loads(json.dumps(candidate["canonical_ranked_search_results"], ensure_ascii=False, sort_keys=True)),
            "stored_query": candidate["raw_query"],
            "future_sibling_gain": candidate.get("future_sibling_gain"),
            "metadata": dict(sft_row["metadata"]),
        })
    report = dict(report)
    report["rl_rows"] = len(rows)
    return rows, report


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
