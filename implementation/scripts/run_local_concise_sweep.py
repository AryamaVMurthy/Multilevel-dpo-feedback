#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import torch
import yaml
from datasets import load_dataset

from text_feedback_dpo.answer_evaluation import evaluate_math_answer
from text_feedback_dpo.benchmarks import convert_math_row, extract_math_boxed_answer
from text_feedback_dpo.concise_sweep import PROFILES, promote_profiles, stratified_subset, summarize_records
from text_feedback_dpo.models import TransformersModelProvider


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_rows(config: dict) -> list[dict]:
    dataset = config["dataset"]
    rows = []
    for subject in dataset["subjects"]:
        source = load_dataset(dataset["source"], subject, split="train", revision=dataset["revision"])
        rows.extend(
            convert_math_row(dict(row), subject=subject, source_split="train", index=index)
            for index, row in enumerate(source)
        )
    if len(rows) != dataset["source_counts"]["train"]:
        raise RuntimeError(f"MATH train count mismatch: {len(rows)}")
    return rows


def verify_cuda_placement(model: object) -> None:
    device_map = getattr(model, "hf_device_map", None)
    if device_map is None:
        devices = {str(parameter.device) for parameter in model.parameters()}
    else:
        devices = {str(device) for device in device_map.values()}
    if not devices or any(device not in {"0", "cuda", "cuda:0"} for device in devices):
        raise RuntimeError(f"model is not entirely on laptop GPU: devices={sorted(devices)}")


def render_report(output_dir: Path, summaries: list[dict], promoted: list[str]) -> None:
    write_json(output_dir / "summary.json", {"summaries": summaries, "promoted": promoted})
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    rows = "".join(
        "<tr>" + "".join(f"<td>{value}</td>" for value in row.values()) + "</tr>"
        for row in summaries
    )
    headers = "".join(f"<th>{key}</th>" for key in summaries[0])
    html = f"""<!doctype html><meta charset='utf-8'><title>Concise generation sweep</title>
<style>body{{font:14px system-ui;margin:32px;max-width:1200px}}table{{border-collapse:collapse}}th,td{{padding:7px 10px;border:1px solid #bbb;text-align:right}}th:first-child,td:first-child{{text-align:left}}</style>
<h1>Local concise generation sweep</h1><p>Promoted: {', '.join(promoted)}</p><table><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table>"""
    (output_dir / "report.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--profiles", nargs="+", choices=sorted(PROFILES), default=sorted(PROFILES))
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--exclude-manifest", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    if args.output_dir.exists() and not args.output_dir.is_dir():
        raise RuntimeError(f"output path is not a directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    rows = load_rows(config)
    excluded_ids: set[str] = set()
    if args.exclude_manifest is not None:
        excluded = json.loads(args.exclude_manifest.read_text(encoding="utf-8"))
        excluded_ids = {str(value) for value in excluded.get("example_ids", [])}
        if not excluded_ids:
            raise RuntimeError("exclude manifest contains no example_ids")
        rows = [row for row in rows if row["id"] not in excluded_ids]
    subset = stratified_subset(rows, count=args.count, seed=args.seed)
    manifest = {
        "schema": "local-concise-sweep-v1",
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "model": config["models"]["student"],
        "seed": args.seed,
        "count": args.count,
        "max_new_tokens": args.max_new_tokens,
        "profiles": {name: {**PROFILES[name], "max_new_tokens": args.max_new_tokens} for name in args.profiles},
        "example_ids": [row["id"] for row in subset],
        "excluded_example_ids_sha256": hashlib.sha256(
            json.dumps(sorted(excluded_ids), separators=(",", ":")).encode()
        ).hexdigest() if excluded_ids else None,
        "device": torch.cuda.get_device_name(0),
    }
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError("existing sweep manifest differs; use a new output directory")
    write_json(manifest_path, manifest)
    records_path = args.output_dir / "records.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines()] if records_path.exists() else []
    completed = {(row["profile"], row["id"]) for row in records}
    provider = TransformersModelProvider(
        model_ids={"student": config["models"]["student"]["id"]},
        model_revisions={"student": config["models"]["student"]["revision"]},
    )
    model = provider._load("student")[1]
    verify_cuda_placement(model)
    for profile_name in args.profiles:
        generation = {**PROFILES[profile_name], "max_new_tokens": args.max_new_tokens}
        for example in subset:
            key = (profile_name, example["id"])
            if key in completed:
                continue
            task_seed = int(hashlib.sha256(f"{args.seed}:{profile_name}:{example['id']}".encode()).hexdigest()[:8], 16)
            torch.manual_seed(task_seed)
            torch.cuda.manual_seed_all(task_seed)
            prompt = (
                f"{example['problem']}\n\n"
                "Please reason step by step. End with a line of the exact form "
                "FINAL: \\boxed{answer}."
            )
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            result = provider.generate_result("student", prompt, **generation)
            latency = time.perf_counter() - started
            try:
                extracted_answer, extraction_method = extract_math_boxed_answer(result.text)
            except ValueError:
                extracted_answer, extraction_method = result.text, "no_valid_box"
            evaluation = evaluate_math_answer(extracted_answer, example["gold_answer"])
            record = {
                "profile": profile_name,
                "id": example["id"],
                "subject": example["source_subject"],
                "level": example["difficulty_level"],
                "gold_answer": example["gold_answer"],
                "extracted_answer": extracted_answer,
                "answer_extraction_method": extraction_method,
                "response": result.text,
                "correct": bool(evaluation["correct"]),
                "evaluation": evaluation,
                "prompt_tokens": result.prompt_tokens,
                "generated_tokens": result.generated_tokens,
                "finish_reason": result.finish_reason,
                "latency_seconds": latency,
                "tokens_per_second": result.generated_tokens / latency,
                "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
                "seed": task_seed,
            }
            with records_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
            records.append(record)
            print(json.dumps({key: record[key] for key in ("profile", "id", "correct", "generated_tokens", "finish_reason", "latency_seconds", "peak_gpu_memory_mib")}), flush=True)
    summaries = summarize_records(records)
    promoted = promote_profiles(summaries, count=min(3, len(summaries)))
    render_report(args.output_dir, summaries, promoted)
    print(json.dumps({"summaries": summaries, "promoted": promoted}, indent=2))


if __name__ == "__main__":
    main()
