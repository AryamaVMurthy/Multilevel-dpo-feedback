from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import random
from typing import Any, Callable, Mapping

from text_feedback_dpo.hyperparameter_search import DpoCandidate, GrpoCandidate
from text_feedback_dpo.io import write_json_atomic
from text_feedback_dpo.lora_coverage import build_lora_config, discover_lora_coverage
from text_feedback_dpo.rewards import build_grpo_reward_function
from text_feedback_dpo.training import (
    build_chat_sft_rows,
    build_paper_dpo_config_kwargs,
    build_paper_grpo_config_kwargs,
    build_paper_sft_config_kwargs,
)


def _seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch
        from transformers import set_seed
    except ImportError as exc:
        raise ImportError("torch and transformers are required for paper training") from exc
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_student(config: Any) -> tuple[Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("torch and transformers are required for paper training") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for paper training; refusing CPU fallback")
    model_id = config.models["student"]["id"]
    revision = config.models["student"]["revision"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    return model, tokenizer


def _write_adapter_manifest(
    output_dir: Path,
    *,
    config: Any,
    method: str,
    seed: int,
    candidate: Any,
    coverage: Any,
) -> dict[str, Any]:
    manifest = {
        "schema": "paper-lora-adapter-v1",
        "student_model": config.models["student"]["id"],
        "base_model_revision": config.models["student"]["revision"],
        "method": method,
        "seed": seed,
        "candidate": {
            key: value
            for key, value in vars(candidate).items()
        },
        "lora": {
            "rank": config.lora.rank,
            "alpha": config.lora.alpha,
            "dropout": config.lora.dropout,
            "coverage_hash": coverage.coverage_hash,
            "target_modules": list(coverage.target_modules),
            "inventory": list(coverage.inventory),
            "estimated_lora_parameters": coverage.estimated_lora_parameters,
        },
    }
    write_json_atomic(output_dir / "adapter_manifest.json", manifest)
    return manifest


def train_paper_dpo(
    *,
    config: Any,
    method: str,
    pairs: list[dict[str, Any]],
    output_dir: Path,
    candidate: DpoCandidate,
    seed: int,
) -> dict[str, Any]:
    if method not in {"standard_dpo", "multilevel_dpo", "matched_dpo", "ld_dpo"}:
        raise ValueError("paper DPO method must be standard_dpo, multilevel_dpo, matched_dpo, or ld_dpo")
    if not pairs:
        raise ValueError("paper DPO training requires non-empty preference pairs")
    _seed(seed)
    try:
        from datasets import Dataset
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise ImportError("datasets and trl are required for paper DPO training") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    effective_batch = int(config.dpo_search.effective_global_batch)
    max_steps = max(1, (len(pairs) + effective_batch - 1) // effective_batch)
    loss_type = str(config.dpo_search.loss_type)
    ld_alpha: float | None = None
    if method == "ld_dpo":
        loss_type = "sigmoid"
        ld_alpha = candidate.ld_alpha
        if ld_alpha not in set(config.dpo_search.ld_alpha_values):
            raise ValueError("ld_dpo candidate ld_alpha is not in the frozen config")
    elif candidate.loss_type != loss_type or candidate.ld_alpha is not None:
        raise ValueError("primary DPO candidate does not match the frozen sigmoid_norm objective")
    model, tokenizer = _load_student(config)
    coverage = discover_lora_coverage(
        model,
        rank=config.lora.rank,
        excluded_components=config.lora.excluded_components,
    )
    lora_config = build_lora_config(
        model=model,
        rank=config.lora.rank,
        alpha=config.lora.alpha,
        dropout=config.lora.dropout,
        excluded_components=config.lora.excluded_components,
    )
    trainer_args = DPOConfig(
        **build_paper_dpo_config_kwargs(
            output_dir=output_dir,
            max_steps=max_steps,
            candidate=candidate,
            effective_global_batch=effective_batch,
            max_length=int(config.training["max_sequence_tokens"]),
            loss_type=loss_type,
            ld_alpha=ld_alpha,
        ),
        seed=seed,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=trainer_args,
        train_dataset=Dataset.from_list(pairs),
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    result = trainer.train()
    trainer.save_model(str(output_dir))
    adapter_manifest = _write_adapter_manifest(
        output_dir,
        config=config,
        method=method,
        seed=seed,
        candidate=candidate,
        coverage=coverage,
    )
    metrics = {
        "method": method,
        "seed": seed,
        "pairs": len(pairs),
        "max_steps": max_steps,
        "loss_type": loss_type,
        "ld_alpha": ld_alpha,
        "train_metrics": result.metrics,
        "history": trainer.state.log_history,
        "lora_coverage": {
            "hash": coverage.coverage_hash,
            "target_modules": list(coverage.target_modules),
            "total_parameters": coverage.total_parameters,
            "estimated_lora_parameters": coverage.estimated_lora_parameters,
        },
        "adapter_manifest": adapter_manifest,
    }
    (output_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def train_paper_sft(
    *,
    config: Any,
    method: str,
    rows: list[dict[str, Any]],
    output_dir: Path,
    candidate: DpoCandidate,
    seed: int,
) -> dict[str, Any]:
    if method not in {"response_sft", "on_policy_distillation"}:
        raise ValueError("paper SFT method must be response_sft or on_policy_distillation")
    if not rows:
        raise ValueError("paper SFT training requires non-empty prompt-completion rows")
    if candidate.loss_type != config.dpo_search.loss_type or candidate.ld_alpha is not None:
        raise ValueError("paper SFT baseline requires the frozen primary DPO optimizer candidate")
    _seed(seed)
    try:
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise ImportError("datasets and trl are required for paper SFT training") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    effective_batch = int(config.dpo_search.effective_global_batch)
    max_steps = max(1, (len(rows) + effective_batch - 1) // effective_batch)
    model, tokenizer = _load_student(config)
    coverage = discover_lora_coverage(
        model,
        rank=config.lora.rank,
        excluded_components=config.lora.excluded_components,
    )
    lora_config = build_lora_config(
        model=model,
        rank=config.lora.rank,
        alpha=config.lora.alpha,
        dropout=config.lora.dropout,
        excluded_components=config.lora.excluded_components,
    )
    trainer_args = SFTConfig(
        **build_paper_sft_config_kwargs(
            output_dir=output_dir,
            max_steps=max_steps,
            candidate=candidate,
            effective_global_batch=effective_batch,
            max_length=int(config.training["max_sequence_tokens"]),
        ),
        seed=seed,
    )
    trainer = SFTTrainer(
        model=model,
        args=trainer_args,
        train_dataset=Dataset.from_list(build_chat_sft_rows(rows)),
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    result = trainer.train()
    trainer.save_model(str(output_dir))
    adapter_manifest = _write_adapter_manifest(
        output_dir,
        config=config,
        method=method,
        seed=seed,
        candidate=candidate,
        coverage=coverage,
    )
    metrics = {
        "method": method,
        "seed": seed,
        "examples": len(rows),
        "max_steps": max_steps,
        "completion_only_loss": True,
        "optimizer_source": "frozen_standard_dpo_candidate",
        "train_metrics": result.metrics,
        "history": trainer.state.log_history,
        "lora_coverage": {
            "hash": coverage.coverage_hash,
            "target_modules": list(coverage.target_modules),
            "total_parameters": coverage.total_parameters,
            "estimated_lora_parameters": coverage.estimated_lora_parameters,
        },
        "adapter_manifest": adapter_manifest,
    }
    (output_dir / "train_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def train_paper_grpo(
    *,
    config: Any,
    method: str,
    examples: list[dict[str, Any]],
    output_dir: Path,
    candidate: GrpoCandidate,
    seed: int,
    evaluator: Callable[[dict[str, Any], str], Mapping[str, Any]],
) -> dict[str, Any]:
    if method not in {"grpo", "dapo_sensitivity"}:
        raise ValueError("paper GRPO method must be grpo or dapo_sensitivity")
    if not examples:
        raise ValueError("paper GRPO training requires non-empty examples")
    _seed(seed)
    try:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise ImportError("datasets and trl are required for paper GRPO training") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = _load_student(config)
    coverage = discover_lora_coverage(
        model,
        rank=config.lora.rank,
        excluded_components=config.lora.excluded_components,
    )
    lora_config = build_lora_config(
        model=model,
        rank=config.lora.rank,
        alpha=config.lora.alpha,
        dropout=config.lora.dropout,
        excluded_components=config.lora.excluded_components,
    )
    effective_candidate = candidate
    if method == "dapo_sensitivity":
        effective_candidate = replace(candidate, loss_type=config.grpo_search.sensitivity_loss_type)
    max_steps = max(1, len(examples) // int(config.grpo_search.num_generations))
    trainer_args = GRPOConfig(
        **build_paper_grpo_config_kwargs(
            output_dir=output_dir,
            max_steps=max_steps,
            candidate=effective_candidate,
            max_completion_length=config.generation.roles["student"].max_new_tokens,
            temperature=float(config.generation.roles["student"].temperature),
            top_p=float(config.generation.roles["student"].top_p),
            top_k=int(config.generation.roles["student"].top_k),
            min_p=float(config.generation.roles["student"].min_p),
            presence_penalty=float(config.generation.roles["student"].presence_penalty),
            repetition_penalty=float(config.generation.roles["student"].repetition_penalty),
        ),
        seed=seed,
    )
    examples_by_id = {str(row["id"]): dict(row) for row in examples}
    reward_func = build_grpo_reward_function(
        examples_by_id=examples_by_id,
        evaluator=evaluator,
        domain=str(examples[0]["domain"]),
        mask_truncated_completions=True,
        max_completion_tokens=config.generation.roles["student"].max_new_tokens,
    )
    train_rows = [
        {
            "prompt": row["prompt"],
            "example_id": str(row["id"]),
        }
        for row in examples
    ]
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=trainer_args,
        train_dataset=Dataset.from_list(train_rows),
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    result = trainer.train()
    trainer.save_model(str(output_dir))
    adapter_manifest = _write_adapter_manifest(
        output_dir,
        config=config,
        method=method,
        seed=seed,
        candidate=effective_candidate,
        coverage=coverage,
    )
    metrics = {
        "method": method,
        "seed": seed,
        "examples": len(examples),
        "max_steps": max_steps,
        "train_metrics": result.metrics,
        "history": trainer.state.log_history,
        "lora_coverage": {"hash": coverage.coverage_hash, "target_modules": list(coverage.target_modules)},
        "adapter_manifest": adapter_manifest,
    }
    (output_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics
