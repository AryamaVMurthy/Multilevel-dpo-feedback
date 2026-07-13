from __future__ import annotations

from pathlib import Path
import os
import shutil

from text_feedback_dpo.training import build_method_config


def _load_dataset(path: Path):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("datasets is required for training") from exc
    if not path.exists():
        raise FileNotFoundError(f"training dataset does not exist: {path}")
    return load_dataset("json", data_files=str(path), split="train")


def _common_args(config: dict, output_dir: Path) -> dict:
    return {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": int(config.get("gradient_accumulation_steps", 32)),
        "learning_rate": float(config.get("learning_rate", 1e-6)),
        "num_train_epochs": float(config.get("epochs", 1.0)),
        "bf16": True,
        "gradient_checkpointing": True,
        "logging_steps": 10,
        "save_strategy": "steps",
        "save_steps": int(config.get("save_steps", 100)),
        "save_total_limit": 1,
        "save_only_model": True,
        "eval_strategy": "steps",
        "eval_steps": int(config.get("eval_steps", 100)),
        "deepspeed": str(config["deepspeed_config"]) if config.get("deepspeed_config") else None,
        "report_to": [],
        "remove_unused_columns": False,
    }


def _model_init_kwargs(config: dict) -> dict:
    return {"revision": config.get("model_revision"), "torch_dtype": "bfloat16", "attn_implementation": config.get("attention_implementation", "sdpa")}


def _tokenizer(config: dict, model_id: str):
    from text_feedback_dpo.runtime import load_tokenizer

    return load_tokenizer(model_id, revision=config.get("model_revision"))


def _save_final(trainer, output_dir: Path) -> None:
    for checkpoint in output_dir.glob("checkpoint-*"):
        if checkpoint.is_dir():
            shutil.rmtree(checkpoint)
    trainer.save_model(str(output_dir / "final"))


def _grpo_args(config: dict, output_dir: Path) -> dict:
    common = _common_args(config, output_dir)
    num_generations = int(config.get("num_generations", 4))
    if num_generations < 2:
        raise ValueError("num_generations must be at least 2 for GRPO/DAPO")
    requested_batch = int(config.get("generation_batch_size", config.get("gradient_accumulation_steps", 32)))
    generation_batch_size = max(num_generations, requested_batch)
    generation_batch_size += (-generation_batch_size) % num_generations
    common.update(num_generations=num_generations, generation_batch_size=generation_batch_size)
    return common


def run_sft(*, model_id: str, train_path: Path, eval_path: Path, output_dir: Path, config: dict) -> None:
    from trl import SFTConfig, SFTTrainer

    build_method_config("sft", max_length=4096)
    tokenizer = _tokenizer(config, model_id)
    train_dataset = _load_dataset(train_path)
    eval_dataset = _load_dataset(eval_path)
    args = SFTConfig(**_common_args(config, output_dir), model_init_kwargs=_model_init_kwargs(config), max_length=4096, dataset_text_field="text")
    trainer = SFTTrainer(model=model_id, processing_class=tokenizer, peft_config=None, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset)
    trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))
    if os.environ.get("TFDPO_CLEANUP_TRAINING_INPUTS") == "1":
        for input_path in (train_path, eval_path):
            input_path.unlink(missing_ok=True)
    _save_final(trainer, output_dir)


def run_dpo(*, model_id: str, train_path: Path, eval_path: Path, output_dir: Path, config: dict) -> None:
    from trl import DPOConfig, DPOTrainer

    build_method_config("dpo", max_length=4096)
    tokenizer = _tokenizer(config, model_id)
    train_dataset = _load_dataset(train_path)
    eval_dataset = _load_dataset(eval_path)
    args = DPOConfig(**_common_args(config, output_dir), model_init_kwargs=_model_init_kwargs(config), max_length=4096, loss_type=["sigmoid_norm"], precompute_ref_log_probs=True, beta=float(config.get("beta", 0.1)))
    trainer = DPOTrainer(model=model_id, processing_class=tokenizer, peft_config=None, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset)
    trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))
    _save_final(trainer, output_dir)


def run_grpo(*, model_id: str, train_path: Path, output_dir: Path, config: dict) -> None:
    from trl import GRPOConfig, GRPOTrainer

    build_method_config("grpo", max_length=4096)
    dataset = _load_dataset(train_path)

    def reward_func(completions, gold_answer, **_kwargs):
        from text_feedback_dpo.scoring import score_searchqa
        return [float(score_searchqa(completion, gold, "")["f1"]) for completion, gold in zip(completions, gold_answer, strict=True)]

    args = GRPOConfig(**_grpo_args(config, output_dir), model_init_kwargs=_model_init_kwargs(config), max_completion_length=512, loss_type="grpo")
    trainer = GRPOTrainer(model=model_id, reward_funcs=reward_func, peft_config=None, args=args, train_dataset=dataset, processing_class=_tokenizer(config, model_id))
    trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))
    _save_final(trainer, output_dir)


def run_dapo(*, model_id: str, train_path: Path, output_dir: Path, config: dict) -> None:
    if not config.get("dapo_enabled"):
        raise ValueError("DAPO requires explicit dapo_enabled=true after the primary DPO result is frozen")
    from trl import GRPOConfig, GRPOTrainer

    build_method_config("dapo", max_length=4096)
    dataset = _load_dataset(train_path)

    def reward_func(completions, gold_answer, **_kwargs):
        from text_feedback_dpo.scoring import score_searchqa
        return [float(score_searchqa(completion, gold, "")["f1"]) for completion, gold in zip(completions, gold_answer, strict=True)]

    args = GRPOConfig(
        **_grpo_args(config, output_dir),
        model_init_kwargs=_model_init_kwargs(config),
        max_completion_length=512,
        loss_type="dapo",
        epsilon_high=0.28,
        mask_truncated_completions=True,
    )
    trainer = GRPOTrainer(model=model_id, reward_funcs=reward_func, peft_config=None, args=args, train_dataset=dataset, processing_class=_tokenizer(config, model_id))
    trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))
    _save_final(trainer, output_dir)
