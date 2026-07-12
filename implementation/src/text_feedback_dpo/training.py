from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any


def build_standard_dpo_pairs(pairs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        pair_id = str(pair["id"])
        failed_attempt = int(pair.get("metadata", {}).get("failed_attempt", 0))
        previous = selected.get(pair_id)
        previous_attempt = (
            int(previous.get("metadata", {}).get("failed_attempt", 0))
            if previous is not None
            else None
        )
        if previous is None or failed_attempt < previous_attempt:
            selected[pair_id] = pair
    return list(selected.values())


def response_sft_weight(
    step: int,
    total_steps: int,
    *,
    initial: float = 1.0,
    final: float = 0.0,
) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if step < 0:
        raise ValueError("step must be non-negative")
    if not 0 <= final <= initial:
        raise ValueError("response SFT weights must satisfy 0 <= final <= initial")
    progress = min(1.0, step / total_steps)
    return initial + (final - initial) * progress


def dpo_loss(
    policy_chosen_logps: Any,
    policy_rejected_logps: Any,
    reference_chosen_logps: Any,
    reference_rejected_logps: Any,
    *,
    beta: float,
) -> Any:
    if beta <= 0:
        raise ValueError("beta must be positive")
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required for DPO loss") from exc
    policy_margin = policy_chosen_logps - policy_rejected_logps
    reference_margin = reference_chosen_logps - reference_rejected_logps
    return -torch.nn.functional.logsigmoid(beta * (policy_margin - reference_margin)).mean()


def load_training_rows(path: Any) -> list[dict[str, Any]]:
    from pathlib import Path

    from text_feedback_dpo.io import read_jsonl

    return read_jsonl(Path(path))


def build_distillation_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    output = []
    for row in rows:
        prompt = str(row.get("prompt", "")).strip()
        completion = str(row.get("completion", row.get("chosen", ""))).strip()
        if not prompt or not completion:
            raise ValueError("distillation row requires non-empty prompt and completion")
        output.append({"text": f"{prompt}\n{completion}"})
    if not output:
        raise ValueError("distillation dataset must not be empty")
    return output


def build_chat_sft_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        prompt = str(row.get("prompt", "")).strip()
        completion = str(row.get("completion", "")).strip()
        if not prompt or not completion:
            raise ValueError("SFT row requires non-empty prompt and completion")
        output.append(
            {
                "prompt": [{"role": "user", "content": prompt}],
                "completion": [{"role": "assistant", "content": completion}],
            }
        )
    if not output:
        raise ValueError("SFT dataset must not be empty")
    return output


def materialize_warmup_steps(total_updates: int, warmup_fraction: float) -> int:
    if isinstance(total_updates, bool) or not isinstance(total_updates, int) or total_updates <= 0:
        raise ValueError("total_updates must be a positive integer")
    if not 0 <= warmup_fraction <= 1:
        raise ValueError("warmup_fraction must be between 0 and 1")
    if warmup_fraction == 0:
        return 0
    return min(total_updates, max(1, math.ceil(total_updates * warmup_fraction)))


def build_optimizer_profile(
    *,
    learning_rate: float,
    weight_decay: float,
    warmup_fraction: float,
    total_updates: int,
    scheduler: str,
) -> dict[str, Any]:
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if scheduler not in {"linear", "cosine"}:
        raise ValueError("scheduler must be linear or cosine")
    return {
        "optim": "adamw_torch_fused",
        "learning_rate": learning_rate,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "weight_decay": weight_decay,
        "max_grad_norm": 1.0,
        "lr_scheduler_type": scheduler,
        "warmup_steps": materialize_warmup_steps(total_updates, warmup_fraction),
    }


def build_paper_dpo_config_kwargs(
    *,
    output_dir: Any,
    max_steps: int,
    candidate: Any,
    effective_global_batch: int,
    max_length: int,
    loss_type: str,
    ld_alpha: float | None,
) -> dict[str, Any]:
    if effective_global_batch <= 0:
        raise ValueError("effective_global_batch must be positive")
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if loss_type not in {"sigmoid_norm", "sigmoid"}:
        raise ValueError("paper DPO loss_type must be sigmoid_norm or sigmoid")
    if loss_type == "sigmoid_norm" and ld_alpha is not None:
        raise ValueError("sigmoid_norm DPO must not set ld_alpha")
    if loss_type == "sigmoid" and ld_alpha not in {0.25, 0.5, 0.75}:
        raise ValueError("length-desensitized DPO requires ld_alpha in {0.25, 0.5, 0.75}")
    profile = build_optimizer_profile(
        learning_rate=float(candidate.learning_rate),
        weight_decay=float(candidate.weight_decay),
        warmup_fraction=float(candidate.warmup_fraction),
        total_updates=max_steps,
        scheduler=str(candidate.scheduler),
    )
    result = {
        "output_dir": str(output_dir),
        "beta": float(candidate.beta),
        "max_length": max_length,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": effective_global_batch,
        "max_steps": max_steps,
        "logging_steps": 1,
        "report_to": [],
        "save_strategy": "no",
        "remove_unused_columns": False,
        "bf16": True,
        "loss_type": loss_type,
        **profile,
    }
    if ld_alpha is not None:
        result["ld_alpha"] = float(ld_alpha)
    return result


def build_paper_sft_config_kwargs(
    *,
    output_dir: Any,
    max_steps: int,
    candidate: Any,
    effective_global_batch: int,
    max_length: int,
) -> dict[str, Any]:
    if effective_global_batch <= 0:
        raise ValueError("effective_global_batch must be positive")
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    profile = build_optimizer_profile(
        learning_rate=float(candidate.learning_rate),
        weight_decay=float(candidate.weight_decay),
        warmup_fraction=float(candidate.warmup_fraction),
        total_updates=max_steps,
        scheduler=str(candidate.scheduler),
    )
    return {
        "output_dir": str(output_dir),
        "max_length": max_length,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": effective_global_batch,
        "max_steps": max_steps,
        "logging_steps": 1,
        "report_to": [],
        "save_strategy": "no",
        "bf16": True,
        "gradient_checkpointing": True,
        "use_cache": False,
        "completion_only_loss": True,
        **profile,
    }


def build_paper_grpo_config_kwargs(
    *,
    output_dir: Any,
    max_steps: int,
    candidate: Any,
    max_completion_length: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    presence_penalty: float,
    repetition_penalty: float,
) -> dict[str, Any]:
    if max_completion_length <= 0:
        raise ValueError("max_completion_length must be positive")
    if max_completion_length > 8192:
        raise ValueError("max_completion_length must not exceed 8192")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not 0 <= min_p <= 1:
        raise ValueError("min_p must be between 0 and 1")
    if not 0 <= presence_penalty <= 2:
        raise ValueError("presence_penalty must be between 0 and 2")
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be positive")
    profile = build_optimizer_profile(
        learning_rate=float(candidate.learning_rate),
        weight_decay=0.01,
        warmup_fraction=0.05,
        total_updates=max_steps,
        scheduler="cosine",
    )
    return {
        "output_dir": str(output_dir),
        "max_completion_length": max_completion_length,
        "num_generations": int(candidate.num_generations),
        "generation_batch_size": int(candidate.num_generations),
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "max_steps": max_steps,
        "logging_steps": 1,
        "report_to": [],
        "save_strategy": "no",
        "bf16": True,
        "beta": float(candidate.kl_beta),
        "epsilon": float(candidate.epsilon_low),
        "epsilon_high": float(candidate.epsilon_high),
        "num_iterations": int(candidate.num_iterations),
        "loss_type": str(candidate.loss_type),
        "scale_rewards": "group",
        "mask_truncated_completions": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "repetition_penalty": repetition_penalty,
        "chat_template_kwargs": {"enable_thinking": False},
        "use_vllm": True,
        "vllm_mode": "colocate",
        "vllm_gpu_memory_utilization": 0.25,
        "vllm_max_model_length": max_completion_length + 2048,
        "generation_kwargs": {"presence_penalty": presence_penalty},
        **profile,
    }


def _load_model_and_tokenizer(model_id: str) -> tuple[Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("torch and transformers are required for training") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training; refusing CPU fallback")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    return model, tokenizer


def _lora_config(model: Any, *, rank: int = 16, alpha: int = 32, dropout: float = 0.05) -> tuple[Any, Any]:
    from text_feedback_dpo.lora_coverage import build_lora_config, discover_lora_coverage

    coverage = discover_lora_coverage(
        model,
        rank=rank,
        excluded_components=("vision", "multimodal_projector", "embeddings", "output_head"),
    )
    return (
        build_lora_config(
            model=model,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            excluded_components=("vision", "multimodal_projector", "embeddings", "output_head"),
        ),
        coverage,
    )


def build_dpo_config_kwargs(*, output_dir: Any, max_steps: int) -> dict[str, Any]:
    """Build DPOConfig arguments compatible with the locked TRL v1 API."""

    return {
        "output_dir": str(output_dir),
        "beta": 0.1,
        "max_length": 1024,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 5e-6,
        "max_steps": max_steps,
        "logging_steps": 1,
        "report_to": [],
        "save_strategy": "no",
        "remove_unused_columns": False,
    }


def build_grpo_config_kwargs(*, output_dir: Any, max_steps: int) -> dict[str, Any]:
    """Build a GRPOConfig with a valid generation batch for two samples per prompt."""

    return {
        "output_dir": str(output_dir),
        "max_completion_length": 512,
        "num_generations": 2,
        "generation_batch_size": 2,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 5e-6,
        "max_steps": max_steps,
        "logging_steps": 1,
        "report_to": [],
        "save_strategy": "no",
    }


def run_dpo_training(
    *,
    model_id: str,
    pairs: list[dict[str, Any]],
    output_dir: Any,
    max_steps: int,
    baseline: bool,
) -> dict[str, Any]:
    if not pairs:
        raise ValueError("DPO dataset must contain at least one pair")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    rows = build_standard_dpo_pairs(pairs) if baseline else list(pairs)
    model, tokenizer = _load_model_and_tokenizer(model_id)
    lora_config, coverage = _lora_config(model)
    args = DPOConfig(**build_dpo_config_kwargs(output_dir=output_dir, max_steps=max_steps))
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    return {
        "method": "standard_dpo" if baseline else "native_multilevel_dpo",
        "model_id": model_id,
        "pairs": len(rows),
        "max_steps": max_steps,
        "lora_coverage_hash": coverage.coverage_hash,
        "lora_target_modules": list(coverage.target_modules),
        "train_metrics": train_result.metrics,
        "history": trainer.state.log_history,
    }


def run_distillation_training(
    *,
    model_id: str,
    rows: list[dict[str, Any]],
    output_dir: Any,
    max_steps: int,
) -> dict[str, Any]:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = _load_model_and_tokenizer(model_id)
    lora_config, coverage = _lora_config(model)
    args = SFTConfig(
        output_dir=str(output_dir),
        max_length=1024,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=5e-6,
        max_steps=max_steps,
        logging_steps=1,
        report_to=[],
        save_strategy="no",
        dataset_text_field="text",
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(build_distillation_rows(rows)),
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    return {
        "method": "on_policy_distillation",
        "model_id": model_id,
        "examples": len(rows),
        "max_steps": max_steps,
        "lora_coverage_hash": coverage.coverage_hash,
        "lora_target_modules": list(coverage.target_modules),
        "train_metrics": train_result.metrics,
        "history": trainer.state.log_history,
    }


def run_grpo_training(
    *,
    model_id: str,
    examples: list[dict[str, Any]],
    output_dir: Any,
    max_steps: int,
    evaluator: Any | None = None,
) -> dict[str, Any]:
    if not examples:
        raise ValueError("GRPO dataset must not be empty")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    from text_feedback_dpo.rewards import build_grpo_reward_function

    if evaluator is None:
        raise ValueError("GRPO training requires an explicit evaluator callback; refusing substring reward fallback")

    model, tokenizer = _load_model_and_tokenizer(model_id)
    lora_config, coverage = _lora_config(model)
    args = GRPOConfig(**build_grpo_config_kwargs(output_dir=output_dir, max_steps=max_steps))

    examples_by_id = {str(row["id"]): dict(row) for row in examples}
    if len(examples_by_id) != len(examples):
        raise ValueError("GRPO examples must have unique ids")
    domains = {str(row.get("domain")) for row in examples}
    if len(domains) != 1:
        raise ValueError("GRPO reward batch must contain exactly one domain")
    reward_func = build_grpo_reward_function(
        examples_by_id=examples_by_id,
        evaluator=evaluator,
        domain=next(iter(domains)),
        mask_truncated_completions=True,
        max_completion_tokens=2048,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=args,
        train_dataset=Dataset.from_list(
            [{"prompt": row["prompt"], "example_id": str(row["id"])} for row in examples]
        ),
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    return {
        "method": "standard_grpo",
        "model_id": model_id,
        "examples": len(examples),
        "max_steps": max_steps,
        "lora_coverage_hash": coverage.coverage_hash,
        "lora_target_modules": list(coverage.target_modules),
        "train_metrics": train_result.metrics,
        "history": trainer.state.log_history,
    }
