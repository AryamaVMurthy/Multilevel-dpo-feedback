from __future__ import annotations

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


def _lora_config() -> Any:
    try:
        from peft import LoraConfig
    except ImportError as exc:
        raise ImportError("peft is required for LoRA training") from exc
    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )


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
    args = DPOConfig(
        output_dir=str(output_dir),
        beta=0.1,
        max_length=1024,
        max_prompt_length=512,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=5e-6,
        max_steps=max_steps,
        logging_steps=1,
        report_to=[],
        save_strategy="no",
        remove_unused_columns=False,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer,
        peft_config=_lora_config(),
    )
    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    return {
        "method": "standard_dpo" if baseline else "native_multilevel_dpo",
        "model_id": model_id,
        "pairs": len(rows),
        "max_steps": max_steps,
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
        peft_config=_lora_config(),
    )
    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    return {
        "method": "on_policy_distillation",
        "model_id": model_id,
        "examples": len(rows),
        "max_steps": max_steps,
        "train_metrics": train_result.metrics,
        "history": trainer.state.log_history,
    }


def run_grpo_training(
    *,
    model_id: str,
    examples: list[dict[str, Any]],
    output_dir: Any,
    max_steps: int,
) -> dict[str, Any]:
    if not examples:
        raise ValueError("GRPO dataset must not be empty")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    model, tokenizer = _load_model_and_tokenizer(model_id)
    args = GRPOConfig(
        output_dir=str(output_dir),
        max_completion_length=512,
        num_generations=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=5e-6,
        max_steps=max_steps,
        logging_steps=1,
        report_to=[],
        save_strategy="no",
    )

    def reward_func(completions: list[Any], gold_answer: list[str], **_: Any) -> list[float]:
        rewards = []
        for completion, gold in zip(completions, gold_answer):
            if isinstance(completion, list):
                response = str(completion[-1].get("content", ""))
            else:
                response = str(completion)
            rewards.append(float(str(gold).strip().lower() in response.strip().lower()))
        return rewards

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=args,
        train_dataset=Dataset.from_list(
            [{"prompt": row["prompt"], "gold_answer": row["gold_answer"]} for row in examples]
        ),
        processing_class=tokenizer,
        peft_config=_lora_config(),
    )
    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    return {
        "method": "standard_grpo",
        "model_id": model_id,
        "examples": len(examples),
        "max_steps": max_steps,
        "train_metrics": train_result.metrics,
        "history": trainer.state.log_history,
    }
