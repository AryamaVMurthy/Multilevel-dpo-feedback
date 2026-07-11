#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from text_feedback_dpo.answer_evaluation import evaluate_math_answer
from text_feedback_dpo.benchmarks import extract_math_boxed_answer
from text_feedback_dpo.concise_sweep import (
    PROFILES,
    build_sweep_prompt,
    promote_profiles,
    protocol_valid_correct,
    stratified_subset,
    summarize_records,
    validate_sweep_records,
    validate_screening_context,
)
from text_feedback_dpo.experiment_config import load_paper_experiment, validate_paper_experiment
from text_feedback_dpo.io import read_jsonl_zst, write_json_atomic
from text_feedback_dpo.models import generate_model_result


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_cuda_placement(model: object) -> None:
    device_map = getattr(model, "hf_device_map", None)
    if device_map is None:
        devices = {str(parameter.device) for parameter in model.parameters()}
    else:
        devices = {str(device) for device in device_map.values()}
    if not devices or any(device not in {"0", "cuda", "cuda:0"} for device in devices):
        raise RuntimeError(f"model is not entirely on the allocated GPU: devices={sorted(devices)}")
    floating_dtypes = {
        parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()
    }
    if floating_dtypes != {torch.bfloat16}:
        raise RuntimeError(f"model parameters are not exclusively BF16: {sorted(map(str, floating_dtypes))}")


def render_report(output_dir: Path, summaries: list[dict], promoted: list[str], *, stage: str) -> None:
    with (output_dir / "summary.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    rows = "".join(
        "<tr>" + "".join(f"<td>{value}</td>" for value in row.values()) + "</tr>"
        for row in summaries
    )
    headers = "".join(f"<th>{key}</th>" for key in summaries[0])
    html = f"""<!doctype html><meta charset='utf-8'><title>MATH decoding sweep</title>
<style>body{{font:14px system-ui;margin:32px;max-width:1200px}}table{{border-collapse:collapse}}th,td{{padding:7px 10px;border:1px solid #bbb;text-align:right}}th:first-child,td:first-child{{text-align:left}}</style>
<h1>MATH train-only decoding sweep: {stage}</h1><p>Promoted: {', '.join(promoted)}</p><table><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table>"""
    (output_dir / "report.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--dataset-audit", type=Path, required=True)
    parser.add_argument("--model-cache-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--stage", choices=("screening", "confirmation"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--promotion-manifest", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        raise ValueError("source-commit must be an immutable 40-character lowercase Git SHA")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing sweep output directory: {args.output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("allocated GPU does not support BF16; refusing dtype fallback")
    config = load_paper_experiment(args.config)
    validate_paper_experiment(config)
    if config.dataset.name != "math":
        raise ValueError("decoding sweep requires the frozen MATH experiment config")
    config_sha256 = file_sha256(args.config)
    dataset_manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    dataset_audit = json.loads(args.dataset_audit.read_text(encoding="utf-8"))
    cache_manifest = json.loads(args.model_cache_manifest.read_text(encoding="utf-8"))
    if dataset_audit.get("schema") != "paper-dataset-audit-v1" or dataset_audit.get("status") != "passed":
        raise ValueError("dataset audit is missing or not passed")
    if dataset_audit.get("manifest_sha256") != file_sha256(args.dataset_manifest):
        raise ValueError("dataset manifest hash does not match passed audit")
    if dataset_audit.get("manifest_content_sha256") != dataset_manifest.get("content_sha256"):
        raise ValueError("dataset content hash does not match passed audit")
    if dataset_audit.get("artifact_sha256", {}).get("train.jsonl.zst") != file_sha256(args.data_path):
        raise ValueError("train artifact hash does not match passed audit")
    if cache_manifest.get("schema") != "tfdpo-model-cache-v1":
        raise ValueError("model cache manifest schema mismatch")
    if cache_manifest.get("config_sha256") != config_sha256:
        raise ValueError("model cache config hash does not match sweep config")

    if args.stage == "screening":
        if args.promotion_manifest is not None:
            raise ValueError("screening must not receive a promotion manifest")
        count = 12
        max_new_tokens = 4096
        profiles = tuple(PROFILES)
        promote_count = 3
        excluded_ids: set[str] = set()
        promotion_sha256 = None
    else:
        if args.promotion_manifest is None:
            raise ValueError("confirmation requires the screening selection manifest")
        promotion = json.loads(args.promotion_manifest.read_text(encoding="utf-8"))
        if (
            promotion.get("schema") != "math-decoding-sweep-selection-v1"
            or promotion.get("stage") != "screening"
            or promotion.get("status") != "passed"
        ):
            raise ValueError("confirmation promotion manifest is not a screening selection")
        screening_manifest_path = args.promotion_manifest.parent / "manifest.json"
        if not screening_manifest_path.is_file():
            raise FileNotFoundError("screening sweep manifest is missing beside its selection")
        if promotion.get("sweep_manifest_sha256") != file_sha256(screening_manifest_path):
            raise ValueError("screening selection does not match its sweep manifest")
        screening_manifest = json.loads(screening_manifest_path.read_text(encoding="utf-8"))
        validate_screening_context(
            screening_manifest,
            config_sha256=config_sha256,
            dataset_manifest_sha256=file_sha256(args.dataset_manifest),
            dataset_audit_sha256=file_sha256(args.dataset_audit),
            model_cache_manifest_sha256=file_sha256(args.model_cache_manifest),
            model=config.models["student"],
        )
        promoted = promotion.get("promoted")
        if not isinstance(promoted, list) or len(promoted) != 3 or not all(name in PROFILES for name in promoted):
            raise ValueError("screening selection must promote exactly three known profiles")
        excluded = promotion.get("example_ids")
        if not isinstance(excluded, list) or len(excluded) != 12 or len(set(excluded)) != 12:
            raise ValueError("screening selection must bind exactly 12 unique example IDs")
        count = 32
        max_new_tokens = 8192
        profiles = tuple(promoted)
        promote_count = 1
        excluded_ids = {str(value) for value in excluded}
        promotion_sha256 = file_sha256(args.promotion_manifest)

    rows = read_jsonl_zst(args.data_path)
    rows = [row for row in rows if str(row.get("id")) not in excluded_ids]
    subset = stratified_subset(rows, count=count, seed=args.seed)
    example_ids = [str(row["id"]) for row in subset]
    if set(example_ids) & excluded_ids:
        raise RuntimeError("decoding confirmation overlaps screening examples")
    prompts = {str(row["id"]): build_sweep_prompt(row) for row in subset}

    student = config.models["student"]
    expected_identity = (student["id"], student["revision"])
    matches = [
        entry for entry in cache_manifest.get("models", [])
        if (entry.get("id"), entry.get("revision")) == expected_identity
    ]
    if len(matches) != 1:
        raise ValueError("model cache does not contain exactly one frozen student snapshot")
    snapshot = Path(str(matches[0].get("snapshot_path", "")))
    if not snapshot.is_dir():
        raise FileNotFoundError(f"frozen student snapshot is missing: {snapshot}")

    manifest = {
        "schema": "math-decoding-sweep-v1",
        "stage": args.stage,
        "source_commit": args.source_commit,
        "config_sha256": config_sha256,
        "dataset_manifest_sha256": file_sha256(args.dataset_manifest),
        "dataset_audit_sha256": file_sha256(args.dataset_audit),
        "train_artifact_sha256": file_sha256(args.data_path),
        "model_cache_manifest_sha256": file_sha256(args.model_cache_manifest),
        "model": student,
        "snapshot_path": str(snapshot),
        "seed": args.seed,
        "count": count,
        "max_new_tokens": max_new_tokens,
        "profiles": {name: {**PROFILES[name], "max_new_tokens": max_new_tokens} for name in profiles},
        "prompt_protocol": config.collection["prompt_protocol"],
        "prompt_sha256_by_id": {
            row_id: hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            for row_id, prompt in prompts.items()
        },
        "example_ids": example_ids,
        "excluded_example_ids": sorted(excluded_ids),
        "promotion_manifest_sha256": promotion_sha256,
        "device": torch.cuda.get_device_name(0),
    }
    args.output_dir.mkdir(parents=True)
    write_json_atomic(args.output_dir / "manifest.json", manifest)
    records_path = args.output_dir / "records.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    verify_cuda_placement(model)
    records = []
    for profile_name in profiles:
        generation = {**PROFILES[profile_name], "max_new_tokens": max_new_tokens}
        for example in subset:
            task_seed = int(
                hashlib.sha256(
                    f"{args.seed}:{profile_name}:{example['id']}".encode()
                ).hexdigest()[:8],
                16,
            )
            torch.manual_seed(task_seed)
            torch.cuda.manual_seed_all(task_seed)
            prompt = prompts[str(example["id"])]
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            result = generate_model_result(
                tokenizer=tokenizer,
                model=model,
                prompt=prompt,
                generation_kwargs=generation,
            )
            latency = time.perf_counter() - started
            if not result.text.strip() or result.generated_tokens is None or result.generated_tokens <= 0:
                raise RuntimeError(f"empty generation for {profile_name}/{example['id']}")
            try:
                extracted_answer, extraction_method = extract_math_boxed_answer(result.text)
            except ValueError:
                extracted_answer, extraction_method = result.text, "no_valid_box"
            evaluation = evaluate_math_answer(extracted_answer, example["gold_answer"])
            evaluable = not bool(evaluation["requires_model_judgment"])
            symbolic_correct = evaluable and bool(evaluation["correct"])
            record = {
                "profile": profile_name,
                "id": example["id"],
                "subject": example["source_subject"],
                "level": example["difficulty_level"],
                "gold_answer": example["gold_answer"],
                "extracted_answer": extracted_answer,
                "answer_extraction_method": extraction_method,
                "response": result.text,
                "evaluable": evaluable,
                "symbolic_correct": symbolic_correct,
                "correct": protocol_valid_correct(
                    symbolic_correct=symbolic_correct,
                    terminated=result.terminated,
                    truncated=result.truncated,
                ),
                "evaluation": evaluation,
                "prompt_tokens": result.prompt_tokens,
                "generated_tokens": result.generated_tokens,
                "finish_reason": result.finish_reason,
                "latency_seconds": latency,
                "tokens_per_second": result.generated_tokens / latency,
                "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
                "seed": task_seed,
                "source_commit": args.source_commit,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            }
            with records_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            records.append(record)
            print(json.dumps({key: record[key] for key in (
                "profile", "id", "correct", "evaluable", "generated_tokens",
                "finish_reason", "latency_seconds", "peak_gpu_memory_mib",
            )}, sort_keys=True), flush=True)

    validate_sweep_records(records, profiles=profiles, example_ids=example_ids)
    summaries = summarize_records(records)
    promoted = promote_profiles(summaries, count=promote_count)
    selection = {
        "schema": "math-decoding-sweep-selection-v1",
        "status": "passed",
        "stage": args.stage,
        "sweep_manifest_sha256": file_sha256(args.output_dir / "manifest.json"),
        "records_sha256": file_sha256(records_path),
        "example_ids": example_ids,
        "summaries": summaries,
        "promoted": promoted,
        "selected_profile": promoted[0] if args.stage == "confirmation" else None,
    }
    write_json_atomic(args.output_dir / "selection.json", selection)
    render_report(args.output_dir, summaries, promoted, stage=args.stage)
    print(json.dumps(selection, sort_keys=True))


if __name__ == "__main__":
    main()
