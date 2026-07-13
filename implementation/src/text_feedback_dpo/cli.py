from __future__ import annotations

import argparse
import json
from pathlib import Path

from text_feedback_dpo.config import load_config
from text_feedback_dpo.dataset import attach_evidence, build_sft_row, load_searchqa_split_with_stats, write_jsonl
from text_feedback_dpo.scoring import score_searchqa


def iter_jsonl(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"JSONL input does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc


def read_jsonl(path: Path) -> list[dict]:
    rows = list(iter_jsonl(path))
    if not rows:
        raise ValueError(f"JSONL input is empty: {path}")
    return rows


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def cmd_prepare(args: argparse.Namespace) -> None:
    from text_feedback_dpo.runtime import load_tokenizer

    tokenizer = load_tokenizer(args.tokenizer_model, revision=args.tokenizer_revision)
    rows, load_stats = load_searchqa_split_with_stats(args.source, args.split, revision=args.revision, limit=args.limit)
    rows = attach_evidence(rows, max_evidence_tokens=args.max_evidence_tokens, token_count=lambda text: len(tokenizer.encode(text, add_special_tokens=False)))
    write_jsonl(rows, args.output)
    manifest = {"source": args.source, "split": args.split, "rows": len(rows), "max_length": 4096, "load_stats": load_stats, "required_files": [args.output.name]}
    write_json(args.output.with_suffix(".manifest.json"), manifest)
    write_json(args.output.parent / "manifest.json", manifest)


def cmd_shard_jsonl(args: argparse.Namespace) -> None:
    from text_feedback_dpo.sharding import shard_jsonl

    shard_jsonl(args.input, args.output_dir, shard_count=args.shards)


def cmd_merge_predictions(args: argparse.Namespace) -> None:
    from text_feedback_dpo.sharding import merge_prediction_shards

    result = merge_prediction_shards(args.shard_dir, args.output, shard_count=args.shards)
    write_json(args.output.with_suffix(".manifest.json"), result)


def cmd_probe_model(args: argparse.Namespace) -> None:
    from text_feedback_dpo.runtime import (
        extract_qwen_final_content,
        generate_batch,
        generate_student_batch,
        load_student,
        load_teacher,
        load_tokenizer,
        render_teacher_prompts,
    )

    tokenizer = load_tokenizer(args.model, revision=args.model_revision)
    if args.role == "student":
        model = load_student(args.model, revision=args.model_revision, attention_implementation=args.attention_implementation)
        prompt = "Evidence:\nAda Lovelace wrote the first algorithm.\n\nQuestion: Who wrote the first algorithm?\n\nAnswer:"
        result = generate_student_batch(
            model, tokenizer, [prompt], mode="direct", scratchpad_max_new_tokens=256,
            answer_max_new_tokens=32, temperature=0.0, top_p=1.0,
        )[0]
        write_json(args.output, {"role": "student", "model": args.model, "response": result.response, "truncated": result.truncated})
        return
    from text_feedback_dpo.feedback import parse_feedback
    from text_feedback_dpo.prompts import build_teacher_prompt

    model = load_teacher(
        args.model, revision=args.model_revision, quantization=args.teacher_quantization,
        attention_implementation=args.attention_implementation,
    )
    prompt = build_teacher_prompt(
        {"question": "Who wrote the first algorithm?", "packed_evidence": "Ada Lovelace wrote the first algorithm.", "gold_answer": "Ada Lovelace"},
        "Grace Hopper",
        [],
    )
    rendered = render_teacher_prompts(tokenizer, [prompt], enable_thinking=True)
    final = extract_qwen_final_content(generate_batch(model, tokenizer, rendered, max_new_tokens=96, temperature=0.0, top_p=1.0)[0])
    feedback = parse_feedback(final, gold_answer="Ada Lovelace")
    write_json(args.output, {"role": "teacher", "model": args.model, "quantization": args.teacher_quantization, "hint": feedback.hint, "native_thinking": True})


def cmd_build_sft(args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for row in iter_jsonl(args.data):
            handle.write(json.dumps(build_sft_row(row), sort_keys=True, ensure_ascii=False) + "\n")
            count += 1
    if count == 0:
        args.output.unlink(missing_ok=True)
        raise ValueError("SFT input produced zero rows")


def cmd_report(args: argparse.Namespace) -> None:
    from text_feedback_dpo.report import write_html_report

    write_html_report(args.output, json.loads(args.metrics.read_text(encoding="utf-8")), args.artifact)


def cmd_validate(args: argparse.Namespace) -> None:
    from text_feedback_dpo.artifacts import validate_artifacts

    write_json(args.output, validate_artifacts(args.directory))


def cmd_compare(args: argparse.Namespace) -> None:
    from text_feedback_dpo.comparison import comparison_metrics
    from text_feedback_dpo.report import write_html_report

    metrics = comparison_metrics(args.run)
    write_json(args.output, metrics)
    write_html_report(args.html, metrics, [args.output.name, *args.artifact])


def cmd_evaluate(args: argparse.Namespace) -> None:
    examples = {row["id"]: row for row in read_jsonl(args.data)}
    predictions = read_jsonl(args.predictions)
    results = []
    for prediction in predictions:
        example_id = prediction.get("id", prediction.get("example_id"))
        if example_id not in examples:
            raise ValueError(f"prediction has unknown example id: {example_id}")
        example = examples[example_id]
        result = score_searchqa(prediction["response"], example["gold_answer"], example["packed_evidence"])
        results.append({"id": example_id, **result})
    exact = sum(row["exact_match"] for row in results) / len(results)
    f1 = sum(row["f1"] for row in results) / len(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(results, args.output.with_suffix(".jsonl"))
    write_json(args.output, {"examples": len(results), "exact_match": exact, "f1": f1, "correct": sum(row["correct"] for row in results)})


def cmd_preflight_quality(args: argparse.Namespace) -> None:
    from text_feedback_dpo.preflight import assess_preflight, select_preflight_rows, summarize_response_quality

    if args.split_name != "train-dev":
        raise ValueError("prompt preflight may use only the explicit train-dev split")
    examples = read_jsonl(args.data)
    predictions = read_jsonl(args.predictions)
    metrics = summarize_response_quality(examples, predictions)
    metrics["split_name"] = args.split_name
    metrics["gate"] = assess_preflight(metrics)
    selected = select_preflight_rows(examples, sample_size=args.sample_size, seed=args.seed)
    predictions_by_id = {str(row["id"]): row for row in predictions}
    samples = [
        {
            "id": example["id"],
            "question": example["question"],
            "gold_answer": example["gold_answer"],
            "response": predictions_by_id[str(example["id"])]["response"],
            "truncated": predictions_by_id[str(example["id"])]["truncated"],
        }
        for example in selected
    ]
    write_jsonl(samples, args.samples)
    write_json(args.output, metrics)


def cmd_select_thinking_mode(args: argparse.Namespace) -> None:
    from text_feedback_dpo.preflight import select_thinking_mode

    summaries = {
        "direct": json.loads(args.direct.read_text(encoding="utf-8")),
        "two_pass": json.loads(args.two_pass.read_text(encoding="utf-8")),
    }
    write_json(args.output, select_thinking_mode(summaries))


def cmd_generate(args: argparse.Namespace) -> None:
    from text_feedback_dpo.prompts import build_student_prompt
    from text_feedback_dpo.runtime import generate_student_batch, load_student, load_tokenizer

    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rows = read_jsonl(args.data)
    tokenizer = load_tokenizer(args.model, revision=args.model_revision)
    model = load_student(args.model, revision=args.model_revision, attention_implementation=args.attention_implementation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            prompts = [row.get("prompt") or build_student_prompt(row, []) for row in batch]
            generations = generate_student_batch(
                model, tokenizer, prompts, mode=args.student_thinking_mode,
                scratchpad_max_new_tokens=args.scratchpad_max_new_tokens,
                answer_max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p,
            )
            for row, generation in zip(batch, generations, strict=True):
                output = {"id": row["id"], "response": generation.response, "truncated": generation.truncated, "thinking_mode": generation.mode, "policy_hash": args.policy_hash}
                if generation.scratchpad is not None:
                    output["private_scratchpad"] = generation.scratchpad
                    output["private_scratchpad_truncated"] = generation.scratchpad_truncated
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")
            handle.flush()


def cmd_build_preferences(args: argparse.Namespace) -> None:
    from text_feedback_dpo.preferences import build_preference_rows

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for trajectory in iter_jsonl(args.trajectories):
            for row in build_preference_rows(trajectory):
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
                count += 1
        handle.flush()
    if count == 0:
        args.output.unlink(missing_ok=True)
        raise ValueError("no valid preference rows were produced")


def cmd_collect(args: argparse.Namespace) -> None:
    from text_feedback_dpo.collection import collect_dataset_batchwise
    from text_feedback_dpo.offline import build_cache_manifest, load_or_build_trajectories
    from text_feedback_dpo.runtime import (
        extract_qwen_final_content,
        generate_batch,
        generate_student_batch,
        load_student,
        load_teacher,
        load_tokenizer,
        render_teacher_prompts,
    )

    examples = read_jsonl(args.data)
    teacher_reason = "trajectory_cache_reused_or_primary_teacher_loaded"
    def generate_trajectories(pending):
        student_tokenizer = load_tokenizer(args.student_model, revision=args.student_revision)
        student = load_student(args.student_model, revision=args.student_revision, attention_implementation=args.attention_implementation, device=args.student_device)
        teacher_tokenizer = load_tokenizer(args.teacher_model, revision=args.teacher_revision)
        teacher = load_teacher(args.teacher_model, revision=args.teacher_revision, quantization=args.teacher_quantization, attention_implementation=args.attention_implementation, device=args.teacher_device)
        probe = render_teacher_prompts(teacher_tokenizer, ['Return exactly: {"hint":"Recheck."}'], enable_thinking=args.teacher_thinking)
        generate_batch(teacher, teacher_tokenizer, probe, max_new_tokens=1, temperature=0.0, top_p=1.0)

        if args.student_batch_size <= 0 or args.teacher_batch_size <= 0:
            raise ValueError("student_batch_size and teacher_batch_size must be positive")

        def batched_generate(model, tokenizer, prompts, *, batch_size, **kwargs):
            outputs = []
            for start in range(0, len(prompts), batch_size):
                outputs.extend(generate_batch(model, tokenizer, prompts[start : start + batch_size], **kwargs))
            return outputs

        def student_batch(prompts, **kwargs):
            generations = []
            for start in range(0, len(prompts), args.student_batch_size):
                generations.extend(generate_student_batch(
                    student,
                    student_tokenizer,
                    prompts[start : start + args.student_batch_size],
                    mode=args.student_thinking_mode,
                    scratchpad_max_new_tokens=args.scratchpad_max_new_tokens,
                    answer_max_new_tokens=args.answer_max_new_tokens,
                    temperature=kwargs["temperature"],
                    top_p=kwargs["top_p"],
                ))
            return [generation.response for generation in generations]

        def teacher_batch(prompts, **kwargs):
            rendered = render_teacher_prompts(teacher_tokenizer, prompts, enable_thinking=args.teacher_thinking)
            raw = batched_generate(
                teacher, teacher_tokenizer, rendered,
                batch_size=args.teacher_batch_size,
                max_new_tokens=args.teacher_max_new_tokens,
                temperature=kwargs["temperature"], top_p=kwargs["top_p"],
            )
            return [extract_qwen_final_content(text) for text in raw]

        return collect_dataset_batchwise(examples=pending, student_generate_batch=student_batch, teacher_generate_batch=teacher_batch, max_interventions=args.max_interventions)

    cache_manifest = build_cache_manifest(
        student_model=args.student_model,
        student_revision=args.student_revision,
        teacher_model=args.teacher_model,
        teacher_revision=args.teacher_revision,
        dataset_revision=args.dataset_revision,
        prompt_version=args.prompt_version,
        student_thinking_mode=args.student_thinking_mode,
        teacher_thinking=args.teacher_thinking,
        decoding={
            "max_length": 4096,
            "answer_max_new_tokens": args.answer_max_new_tokens,
            "scratchpad_max_new_tokens": args.scratchpad_max_new_tokens,
            "student_temperature": 0.7,
            "student_top_p": 0.9,
            "teacher_max_new_tokens": args.teacher_max_new_tokens,
            "teacher_temperature": 0.0,
            "teacher_top_p": 1.0,
            "student_batch_size": args.student_batch_size,
            "teacher_batch_size": args.teacher_batch_size,
        },
        intervention_policy={"max_interventions": args.max_interventions, "max_hint_words": 24},
        seed=args.seed,
        policy_hash=args.policy_hash,
    )
    rows = load_or_build_trajectories(examples=examples, cache_path=args.trajectory_cache, cache_manifest=cache_manifest, generate=generate_trajectories)
    write_jsonl(rows, args.output)
    manifest = {"student_model": args.student_model, "teacher_model": args.teacher_model, "teacher_reason": teacher_reason, "student_thinking_mode": args.student_thinking_mode, "teacher_thinking": args.teacher_thinking, "max_length": 4096, "max_interventions": args.max_interventions, "required_files": [args.output.name]}
    write_json(args.output.with_suffix(".manifest.json"), manifest)
    write_json(args.output.parent / "manifest.json", manifest)


def cmd_train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    from text_feedback_dpo.trainers import run_dapo, run_dpo, run_grpo, run_sft

    model_id = args.model or config["student_model"]
    common = {"learning_rate": args.learning_rate, "epochs": args.epochs, "gradient_accumulation_steps": args.gradient_accumulation_steps, "deepspeed_config": args.deepspeed_config, "resume_from_checkpoint": args.resume_from_checkpoint, "save_steps": args.save_steps, "eval_steps": args.eval_steps, "dapo_enabled": args.method == "dapo", "attention_implementation": config["training"]["attention_implementation"], "model_revision": args.model_revision if args.model else config["student_revision"]}
    kwargs = {"model_id": model_id, "train_path": args.train, "output_dir": args.output, "config": common}
    if args.method in {"sft", "dpo"}:
        kwargs["eval_path"] = args.eval
    {"sft": run_sft, "dpo": run_dpo, "grpo": run_grpo, "dapo": run_dapo}[args.method](**kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tfdpo")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-searchqa")
    prepare.add_argument("--source", required=True)
    prepare.add_argument("--split", required=True)
    prepare.add_argument("--tokenizer-model", required=True)
    prepare.add_argument("--tokenizer-revision", required=True)
    prepare.add_argument("--revision", required=True)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--max-evidence-tokens", required=True, type=int)
    prepare.add_argument("--limit", type=int)
    prepare.set_defaults(func=cmd_prepare)
    shard = sub.add_parser("shard-jsonl")
    shard.add_argument("--input", required=True, type=Path)
    shard.add_argument("--output-dir", required=True, type=Path)
    shard.add_argument("--shards", required=True, type=int)
    shard.set_defaults(func=cmd_shard_jsonl)
    merge = sub.add_parser("merge-predictions")
    merge.add_argument("--shard-dir", required=True, type=Path)
    merge.add_argument("--output", required=True, type=Path)
    merge.add_argument("--shards", required=True, type=int)
    merge.set_defaults(func=cmd_merge_predictions)
    probe = sub.add_parser("probe-model")
    probe.add_argument("--role", choices=("student", "teacher"), required=True)
    probe.add_argument("--model", required=True)
    probe.add_argument("--model-revision", required=True)
    probe.add_argument("--teacher-quantization", choices=("4bit", "bf16"), required=True)
    probe.add_argument("--attention-implementation", choices=("sdpa", "flash_attention_2"), default="sdpa")
    probe.add_argument("--output", required=True, type=Path)
    probe.set_defaults(func=cmd_probe_model)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--data", required=True, type=Path)
    evaluate.add_argument("--predictions", required=True, type=Path)
    evaluate.add_argument("--output", required=True, type=Path)
    evaluate.set_defaults(func=cmd_evaluate)
    preflight_quality = sub.add_parser("preflight-quality")
    preflight_quality.add_argument("--data", required=True, type=Path)
    preflight_quality.add_argument("--predictions", required=True, type=Path)
    preflight_quality.add_argument("--output", required=True, type=Path)
    preflight_quality.add_argument("--samples", required=True, type=Path)
    preflight_quality.add_argument("--split-name", required=True)
    preflight_quality.add_argument("--sample-size", type=int, default=32)
    preflight_quality.add_argument("--seed", type=int, default=7)
    preflight_quality.set_defaults(func=cmd_preflight_quality)
    select_thinking = sub.add_parser("select-thinking-mode")
    select_thinking.add_argument("--direct", required=True, type=Path)
    select_thinking.add_argument("--two-pass", required=True, type=Path)
    select_thinking.add_argument("--output", required=True, type=Path)
    select_thinking.set_defaults(func=cmd_select_thinking_mode)
    generate = sub.add_parser("generate")
    generate.add_argument("--data", required=True, type=Path)
    generate.add_argument("--output", required=True, type=Path)
    generate.add_argument("--model", required=True)
    generate.add_argument("--model-revision")
    generate.add_argument("--attention-implementation", choices=("sdpa", "flash_attention_2"), required=True)
    generate.add_argument("--batch-size", type=int, default=4)
    generate.add_argument("--student-thinking-mode", choices=("direct", "two_pass"), default="direct")
    generate.add_argument("--scratchpad-max-new-tokens", type=int, default=256)
    generate.add_argument("--max-new-tokens", type=int, default=32)
    generate.add_argument("--temperature", type=float, default=0.0)
    generate.add_argument("--top-p", type=float, default=1.0)
    generate.add_argument("--policy-hash", default="unversioned-generate")
    generate.set_defaults(func=cmd_generate)
    preferences = sub.add_parser("build-preferences")
    preferences.add_argument("--trajectories", required=True, type=Path)
    preferences.add_argument("--output", required=True, type=Path)
    preferences.set_defaults(func=cmd_build_preferences)
    sft_data = sub.add_parser("build-sft-data")
    sft_data.add_argument("--data", required=True, type=Path)
    sft_data.add_argument("--output", required=True, type=Path)
    sft_data.set_defaults(func=cmd_build_sft)
    report = sub.add_parser("report")
    report.add_argument("--metrics", required=True, type=Path)
    report.add_argument("--output", required=True, type=Path)
    report.add_argument("--artifact", action="append", default=[])
    report.set_defaults(func=cmd_report)
    validate = sub.add_parser("validate-run")
    validate.add_argument("--directory", required=True, type=Path)
    validate.add_argument("--output", required=True, type=Path)
    validate.set_defaults(func=cmd_validate)
    compare = sub.add_parser("compare")
    compare.add_argument("--run", action="append", required=True)
    compare.add_argument("--output", required=True, type=Path)
    compare.add_argument("--html", required=True, type=Path)
    compare.add_argument("--artifact", action="append", default=[])
    compare.set_defaults(func=cmd_compare)
    collect = sub.add_parser("collect")
    collect.add_argument("--data", required=True, type=Path)
    collect.add_argument("--output", required=True, type=Path)
    collect.add_argument("--student-model", required=True)
    collect.add_argument("--teacher-model", required=True)
    collect.add_argument("--student-revision", required=True)
    collect.add_argument("--teacher-revision", required=True)
    collect.add_argument("--dataset-revision", required=True)
    collect.add_argument("--prompt-version", required=True)
    collect.add_argument("--seed", required=True, type=int)
    collect.add_argument("--teacher-quantization", choices=("4bit", "bf16"), required=True)
    collect.add_argument("--attention-implementation", choices=("sdpa", "flash_attention_2"), required=True)
    collect.add_argument("--student-device", required=True)
    collect.add_argument("--teacher-device", required=True)
    collect.add_argument("--max-interventions", type=int, default=4)
    collect.add_argument("--student-batch-size", type=int, default=32)
    collect.add_argument("--teacher-batch-size", type=int, default=8)
    collect.add_argument("--student-thinking-mode", choices=("direct", "two_pass"), default="direct")
    collect.add_argument("--scratchpad-max-new-tokens", type=int, default=256)
    collect.add_argument("--answer-max-new-tokens", type=int, default=32)
    collect.add_argument("--teacher-max-new-tokens", type=int, default=96)
    collect.add_argument("--teacher-thinking", action=argparse.BooleanOptionalAction, default=True)
    collect.add_argument("--trajectory-cache", required=True, type=Path)
    collect.add_argument("--policy-hash", required=True)
    collect.set_defaults(func=cmd_collect)
    for method in ("sft", "dpo", "grpo", "dapo"):
        train = sub.add_parser(f"train-{method}")
        train.add_argument("--config", required=True, type=Path)
        train.add_argument("--train", required=True, type=Path)
        if method in {"sft", "dpo"}:
            train.add_argument("--eval", required=True, type=Path)
        else:
            train.set_defaults(eval=None)
        train.add_argument("--output", required=True, type=Path)
        train.add_argument("--model")
        train.add_argument("--model-revision")
        train.add_argument("--learning-rate", type=float, default=1e-6)
        train.add_argument("--epochs", type=float, default=1.0)
        train.add_argument("--gradient-accumulation-steps", type=int, default=32)
        train.add_argument("--deepspeed-config", type=Path)
        train.add_argument("--resume-from-checkpoint")
        train.add_argument("--save-steps", type=int, default=100)
        train.add_argument("--eval-steps", type=int, default=100)
        train.set_defaults(func=cmd_train, method=method)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
