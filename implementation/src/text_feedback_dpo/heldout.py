from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any, Callable, Mapping

from text_feedback_dpo.evaluation import evaluate_native_examples
from text_feedback_dpo.io import write_json_atomic, write_jsonl
from text_feedback_dpo.prompts import build_native_student_prompt


def validate_adapter_compatibility(
    adapter_dir: Path,
    *,
    base_model_revision: str,
    lora_coverage_hash: str,
) -> dict[str, Any]:
    manifest_path = adapter_dir / "adapter_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"adapter manifest does not exist: {manifest_path}")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("adapter manifest must be a JSON object")
    for field, expected in (
        ("base_model_revision", base_model_revision),
        ("lora_coverage_hash", lora_coverage_hash),
    ):
        if value.get(field) != expected:
            raise ValueError(f"adapter {field} is incompatible with the requested checkpoint")
    return value


def _teacher_free_prompt(example: Mapping[str, Any]) -> str:
    prompt = build_native_student_prompt(
        problem=str(example["problem"]),
        domain=str(example["domain"]),
        evidence=example.get("evidence"),
    )
    lowered = prompt.casefold()
    forbidden = ("gold answer", "teacher guidance", "teacher-only", "privileged training")
    found = [token for token in forbidden if token in lowered]
    if found:
        raise ValueError(f"teacher-free checkpoint prompt contains forbidden context: {found[0]}")
    return prompt


def _seed_everything(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("evaluation seed must be an integer")
    random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_checkpoint(
    *,
    examples: list[dict[str, Any]],
    generate: Callable[[str], str],
    evaluator: Callable[[dict[str, Any], str], dict[str, Any]],
    output_dir: Path,
    checkpoint_kind: str,
    base_model_revision: str,
    seed: int,
    test: bool,
    freeze_manifest: Path | None = None,
    adapter_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if checkpoint_kind not in {"base", "adapter"}:
        raise ValueError("checkpoint_kind must be base or adapter")
    if not examples:
        raise ValueError("checkpoint evaluation examples must not be empty")
    if test:
        if freeze_manifest is None or not freeze_manifest.exists():
            raise FileNotFoundError("test evaluation requires an existing freeze manifest")
        freeze_value = json.loads(freeze_manifest.read_text(encoding="utf-8"))
        if not isinstance(freeze_value, dict) or freeze_value.get("schema") != "hyperparameter-freeze-v1":
            raise ValueError("freeze manifest does not have the required schema")
    marker = output_dir / "test_complete.json"
    if test and marker.exists():
        raise FileExistsError(f"refusing to repeat test evaluation; test marker exists: {marker}")
    if adapter_manifest is not None and checkpoint_kind != "adapter":
        raise ValueError("adapter_manifest is only valid for adapter checkpoints")
    _seed_everything(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    if predictions_path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint predictions: {predictions_path}")
    predictions: list[dict[str, Any]] = []
    result_by_id: dict[str, dict[str, Any]] = {}
    response_by_id: dict[str, str] = {}
    for example in examples:
        example_id = example.get("id")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError("checkpoint example is missing id")
        if example_id in response_by_id:
            raise ValueError(f"checkpoint examples contain duplicate id: {example_id}")
        prompt = _teacher_free_prompt(example)
        response = generate(prompt)
        if not isinstance(response, str) or not response.strip():
            raise ValueError(f"checkpoint generator returned empty response for {example_id}")
        result = evaluator(example, response)
        if not isinstance(result, dict):
            raise ValueError(f"checkpoint evaluator returned non-object result for {example_id}")
        response_by_id[example_id] = response
        result_by_id[example_id] = result
        predictions.append(
            {
                "id": example_id,
                "response": response,
                "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
                "teacher_free": True,
                "checkpoint_kind": checkpoint_kind,
                "base_model_revision": base_model_revision,
                "evaluator_result": result,
                "generated_tokens": len(response.split()),
            }
        )
    write_jsonl(predictions_path, predictions)

    def replay_evaluator(example: dict[str, Any], response: str) -> dict[str, Any]:
        if response != response_by_id[example["id"]]:
            raise ValueError(f"checkpoint response changed during summary for {example['id']}")
        return result_by_id[example["id"]]

    metrics = evaluate_native_examples(
        examples,
        [{"id": row["id"], "response": row["response"], "generated_tokens": row["generated_tokens"]} for row in predictions],
        evaluator=replay_evaluator,
    )
    summary = {
        "checkpoint_kind": checkpoint_kind,
        "base_model_revision": base_model_revision,
        "seed": seed,
        "test": test,
        "examples": len(examples),
        "metrics": metrics,
        "adapter_manifest": dict(adapter_manifest) if adapter_manifest is not None else None,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if test:
        write_json_atomic(
            marker,
            {
                "schema": "frozen-test-complete-v1",
                "freeze_manifest": str(freeze_manifest),
                "predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
                "examples": len(examples),
            },
        )
    return metrics


def build_transformers_checkpoint_generator(
    *,
    model_id: str,
    revision: str,
    generation_kwargs: Mapping[str, Any],
    adapter_dir: Path | None = None,
    adapter_base_revision: str | None = None,
    adapter_lora_coverage_hash: str | None = None,
) -> Callable[[str], str]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("torch and transformers are required for checkpoint evaluation") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for checkpoint evaluation; refusing CPU fallback")
    if adapter_dir is not None:
        if adapter_base_revision is None or adapter_lora_coverage_hash is None:
            raise ValueError("adapter loading requires base revision and LoRA coverage hash")
        validate_adapter_compatibility(
            adapter_dir,
            base_model_revision=adapter_base_revision,
            lora_coverage_hash=adapter_lora_coverage_hash,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if adapter_dir is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("peft is required to load an adapter checkpoint") from exc
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)

    def generate(prompt: str) -> str:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=True,
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        output_ids = model.generate(
            **encoded,
            max_new_tokens=int(generation_kwargs["max_new_tokens"]),
            do_sample=True,
            temperature=float(generation_kwargs["temperature"]),
            top_p=float(generation_kwargs["top_p"]),
            top_k=int(generation_kwargs["top_k"]),
        )
        generated = output_ids[0][encoded["input_ids"].shape[-1] :]
        return tokenizer.decode(generated, skip_special_tokens=True)

    return generate
