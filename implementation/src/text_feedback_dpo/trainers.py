from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

from text_feedback_dpo.training import (
    MAX_SEQUENCE_LENGTH,
    build_method_config,
    build_reference_manifest,
    dataset_identity_hash,
    load_precomputed_reference_log_probs,
    write_precomputed_reference_log_probs,
)


LOGGER = logging.getLogger(__name__)
MAX_CITED_RESPONSE_COMPLETION = 256

REWARD_COMPONENT_WEIGHTS = {
    "exact_answer": 2.0,
    "bounded_f1": 0.25,
    "retrieval_recall": 0.40,
    "retrieval_mrr": 0.20,
    "future_retrieval_proxy": 0.15,
    "valid_citations": 0.20,
    "lexical_support": 0.20,
    "concise_reasoning": 0.10,
    "malformed_penalty": 0.75,
    "fabricated_citation_penalty": 0.75,
    "truncation_penalty": 1.0,
    "verbosity_penalty": 0.25,
}


def _load_dataset(path: Path):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("datasets is required for training") from exc
    if not path.exists():
        raise FileNotFoundError(f"training dataset does not exist: {path}")
    return load_dataset("json", data_files=str(path), split="train")


def _common_args(config: dict, output_dir: Path) -> dict:
    if "max_steps" not in config:
        raise ValueError("training max_steps must be explicitly selected by launcher or config")
    max_steps = config["max_steps"]
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps == 0 or max_steps < -1:
        raise ValueError("max_steps must be -1 for an explicitly selected full run or positive for smoke")
    workers = int(config.get("dataloader_num_workers", 0))
    if workers < 0:
        raise ValueError("dataloader_num_workers must be nonnegative")
    args = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": int(config.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(config.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(config.get("gradient_accumulation_steps", 32)),
        "learning_rate": float(config.get("learning_rate", 1e-6)),
        "num_train_epochs": float(config.get("epochs", 1.0)),
        "max_steps": max_steps,
        "bf16": True,
        "tf32": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optim": "adamw_torch_fused",
        "dataloader_num_workers": workers,
        "dataloader_pin_memory": bool(config.get("dataloader_pin_memory", True)),
        "dataloader_persistent_workers": bool(config.get("dataloader_persistent_workers", workers > 0)),
        "ddp_find_unused_parameters": False,
        "include_num_input_tokens_seen": True,
        "logging_steps": int(config.get("logging_steps", 10)),
        "save_strategy": "steps",
        "save_steps": int(config.get("save_steps", 100)),
        "save_total_limit": 3,
        "save_only_model": False,
        "eval_strategy": "steps",
        "eval_steps": int(config.get("eval_steps", 100)),
        "deepspeed": str(config["deepspeed_config"]) if config.get("deepspeed_config") else None,
        "report_to": [],
        "remove_unused_columns": False,
    }
    if args["per_device_train_batch_size"] <= 0 or args["per_device_eval_batch_size"] <= 0 or args["gradient_accumulation_steps"] <= 0:
        raise ValueError("microbatch, eval batch, and gradient accumulation must be positive")
    return args


def _model_init_kwargs(config: dict) -> dict:
    return {
        "revision": config.get("model_revision"),
        "torch_dtype": "bfloat16",
        "attn_implementation": config.get("attention_implementation", "sdpa"),
    }


def _tokenizer(config: dict, model_id: str):
    from text_feedback_dpo.runtime import load_tokenizer

    return load_tokenizer(model_id, revision=config.get("model_revision"))


def _save_final(trainer, output_dir: Path) -> None:
    trainer.save_model(str(output_dir / "final"))


def _sft_args(config: dict, output_dir: Path) -> dict:
    args = {
        **_common_args(config, output_dir),
        "model_init_kwargs": _model_init_kwargs(config),
        "max_length": MAX_SEQUENCE_LENGTH,
        "completion_only_loss": True,
    }
    packing = bool(config.get("packing", False))
    padding_free = bool(config.get("padding_free", False))
    if padding_free and not packing:
        raise ValueError("padding_free requires packing=true under the installed TRL SFT contract")
    args.update(packing=packing, padding_free=padding_free)
    return args


def _dpo_args(config: dict, output_dir: Path) -> dict:
    if config.get("use_liger_kernel", False) is not False:
        raise ValueError("Liger is incompatible with DPO precompute_ref_log_probs and is explicitly disabled")
    return {
        **_common_args(config, output_dir),
        "model_init_kwargs": _model_init_kwargs(config),
        "max_length": MAX_SEQUENCE_LENGTH,
        "loss_type": ["sigmoid"],
        "precompute_ref_log_probs": True,
        "use_liger_kernel": False,
        "beta": float(config.get("beta", 0.1)),
    }


def _rl_args(config: dict, output_dir: Path, *, method: str) -> dict:
    if method not in {"grpo", "dapo"}:
        raise ValueError("RL method must be grpo or dapo")
    common = _common_args(config, output_dir)
    num_generations = int(config.get("num_generations", 4))
    if num_generations < 2:
        raise ValueError("num_generations must be at least 2 for GRPO/DAPO")
    requested_batch = int(config.get("generation_batch_size", config.get("gradient_accumulation_steps", 32)))
    generation_batch_size = max(num_generations, requested_batch)
    generation_batch_size += (-generation_batch_size) % num_generations
    eval_batch_size = int(config.get("per_device_eval_batch_size", num_generations))
    if eval_batch_size <= 0 or eval_batch_size % num_generations:
        raise ValueError("GRPO/DAPO per_device_eval_batch_size must be divisible by num_generations for evaluation")
    completion_budget = int(config.get("max_completion_length", config.get("response_max_new_tokens", MAX_CITED_RESPONSE_COMPLETION)))
    if completion_budget <= 32 or completion_budget > MAX_SEQUENCE_LENGTH:
        raise ValueError("RL max_completion_length must use the cited-response budget between 33 and 4096 tokens")
    common.update(
        num_generations=num_generations,
        generation_batch_size=generation_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        max_completion_length=completion_budget,
        loss_type="grpo" if method == "grpo" else "dapo",
        reward_weights=[REWARD_COMPONENT_WEIGHTS[name] for name in REWARD_COMPONENT_WEIGHTS],
    )
    if method == "dapo":
        common.update(epsilon=0.2, epsilon_high=0.28, mask_truncated_completions=True, beta=0.0)
    return common


def _token_count_without_truncation(tokenizer: object, prompt: str) -> int:
    if not callable(tokenizer):
        raise TypeError("RL prompt validation requires a callable pinned tokenizer")
    encoded = tokenizer(prompt, add_special_tokens=False, truncation=False)
    if not isinstance(encoded, Mapping) or not isinstance(encoded.get("input_ids"), Sequence):
        raise TypeError("pinned tokenizer must return input_ids without truncation")
    return len(encoded["input_ids"])


def validate_rl_prompt_budget(dataset: Sequence[Mapping[str, object]], tokenizer: object, max_completion_length: int) -> Sequence[Mapping[str, object]]:
    if max_completion_length <= 0 or max_completion_length > MAX_SEQUENCE_LENGTH:
        raise ValueError("RL completion budget must be within the 4096-token context")
    prompt_budget = MAX_SEQUENCE_LENGTH - max_completion_length
    for index, row in enumerate(dataset):
        if not isinstance(row, Mapping) or not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError(f"RL row {index} requires a non-empty prompt")
        if row.get("task") not in {"query", "response"}:
            raise ValueError(f"RL row {index} requires task=query or task=response")
        count = _token_count_without_truncation(tokenizer, row["prompt"])
        if count > prompt_budget:
            raise ValueError(f"RL prompt token budget exceeded at row {index}: {count}>{prompt_budget}; truncation is forbidden")
    return dataset


def _token_ids_without_truncation(tokenizer: object, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False, truncation=False)
    if not isinstance(encoded, Mapping) or not isinstance(encoded.get("input_ids"), Sequence):
        raise TypeError("pinned tokenizer must return input_ids without truncation")
    ids = encoded["input_ids"]
    if any(isinstance(token, bool) or not isinstance(token, int) for token in ids):
        raise TypeError("tokenizer input_ids must be integer token IDs")
    return list(ids)


def precompute_reference_log_probs(
    *,
    model: object,
    tokenizer: object,
    rows: Sequence[Mapping[str, object]],
    output_path: Path,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    """Compute real reference log probabilities and persist a strict artifact.

    This deliberately requires a loaded model and performs a forward pass for every
    chosen/rejected completion.  There is no placeholder or deterministic substitute
    for missing GPU/model state.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required for DPO reference-log-probability precompute") from exc
    if not hasattr(model, "parameters") or not hasattr(model, "__call__"):
        raise TypeError("reference precompute requires a loaded causal language model")
    try:
        device = next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("reference model has no parameters; refusing fake reference log probabilities") from exc
    model.eval()
    computed: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"reference precompute row {index} must be a mapping")
        prompt = row.get("prompt")
        chosen = row.get("chosen")
        rejected = row.get("rejected")
        if not all(isinstance(value, str) and value for value in (prompt, chosen, rejected)):
            raise ValueError(f"reference precompute row {index} requires prompt/chosen/rejected text")
        prompt_ids = _token_ids_without_truncation(tokenizer, prompt)
        if not prompt_ids:
            raise ValueError(f"reference precompute row {index} prompt tokenization is empty")
        values: dict[str, float] = {}
        for label, completion in (("chosen", chosen), ("rejected", rejected)):
            input_ids = _token_ids_without_truncation(tokenizer, prompt + completion)
            if len(input_ids) > MAX_SEQUENCE_LENGTH:
                raise ValueError(f"reference precompute row {index} {label} exceeds max_length=4096; truncation is forbidden")
            if len(input_ids) <= len(prompt_ids):
                raise ValueError(f"reference precompute row {index} {label} has no completion tokens")
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_tensor)
            with torch.inference_mode():
                outputs = model(input_ids=input_tensor, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits[:, :-1, :]
            labels = input_tensor[:, 1:]
            token_logps = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            completion_logps = token_logps[:, len(prompt_ids) - 1 :]
            values[label] = float(completion_logps.sum().item())
        computed.append({**dict(row), "ref_chosen_logps": values["chosen"], "ref_rejected_logps": values["rejected"]})
    return write_precomputed_reference_log_probs(output_path, computed, manifest)


def _ranked_sources(value: object) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def evaluate_reward_components(
    completion: str,
    gold_answer: str,
    ranked_sources: Sequence[Mapping[str, object]] | None,
    *,
    task: str = "response",
    sources: Sequence[Mapping[str, object]] | None = None,
    truncated: bool = False,
) -> dict[str, object]:
    from text_feedback_dpo.batch_generation import canonical_cited_score, parse_search_query
    from text_feedback_dpo.retrieval import FixedBM25Retriever, retrieval_metrics

    if task not in {"query", "response"}:
        raise ValueError(f"unknown RL task: {task}")
    if not isinstance(completion, str):
        raise TypeError("RL completion must be text")
    components = {name: 0.0 for name in REWARD_COMPONENT_WEIGHTS}
    source_records = _ranked_sources(ranked_sources)
    malformed = False
    fabricated = False
    if truncated:
        components["truncation_penalty"] = -1.0
    if task == "query":
        try:
            query = parse_search_query(completion)
            corpus = _ranked_sources(sources)
            if not corpus:
                raise ValueError("query reward requires complete source records")
            ranked = FixedBM25Retriever(corpus).search(query, top_k=8)
            metrics = retrieval_metrics(ranked, gold_answer)
            components["retrieval_recall"] = float(metrics["recall@8"])
            components["retrieval_mrr"] = float(metrics["mrr"])
            components["future_retrieval_proxy"] = float(metrics["recall@8"])
            components["verbosity_penalty"] = -max(0.0, len(query.split()) - 16) / 16.0
        except (TypeError, ValueError):
            malformed = True
            components["verbosity_penalty"] = -max(0.0, len(completion.split()) - 16) / 16.0
    else:
        if not source_records:
            raise ValueError("response reward requires canonical retrieved source records")
        score = canonical_cited_score(completion, gold_answer, source_records, truncated=truncated)
        metrics = retrieval_metrics(source_records, gold_answer)
        components["exact_answer"] = float(score.get("answer_correct", False) and score.get("parse_valid", False) and not truncated)
        components["bounded_f1"] = min(0.25, float(score.get("f1", 0.0)) * 0.25)
        components["retrieval_recall"] = float(metrics["recall@8"])
        components["retrieval_mrr"] = float(metrics["mrr"])
        components["valid_citations"] = float(score.get("valid_citation_rate", 0.0))
        components["lexical_support"] = float(score.get("lexical_cited_answer_support", 0.0))
        reasoning_words = int(score.get("reasoning_words", 0))
        if score.get("parse_valid") and not truncated:
            components["concise_reasoning"] = max(0.0, 1.0 - max(0, reasoning_words - 32) / 64.0)
        components["verbosity_penalty"] = -max(0.0, reasoning_words - 32) / 64.0
        malformed = not bool(score.get("parse_valid")) or bool(score.get("malformed_response"))
        fabricated = score.get("error_code") in {"unknown_citation", "invalid_sources", "citation_mismatch", "duplicate_citation"}
    if malformed:
        components["malformed_penalty"] = -1.0
    if fabricated:
        components["fabricated_citation_penalty"] = -1.0
    weighted_total = sum(REWARD_COMPONENT_WEIGHTS[name] * value for name, value in components.items())
    result = {"task": task, "components": components, "weighted_total": weighted_total}
    LOGGER.debug("task7_reward", extra={"task": task, "reward_components": components, "weighted_total": weighted_total})
    return result


def _batch_value(value: object, index: int, size: int, *, name: str) -> object:
    if isinstance(value, list):
        if len(value) != size:
            raise ValueError(f"reward column {name} cardinality mismatch")
        return value[index]
    if size != 1:
        raise ValueError(f"reward column {name} must be batched")
    return value


def build_component_reward_functions() -> list:
    functions = []
    for component_name in REWARD_COMPONENT_WEIGHTS:
        def reward_func(completions, _component_name=component_name, **kwargs):
            if not isinstance(completions, list):
                raise TypeError("GRPO reward completions must be a list")
            values = []
            for index, completion in enumerate(completions):
                task = _batch_value(kwargs.get("task"), index, len(completions), name="task")
                gold = _batch_value(kwargs.get("gold_answer"), index, len(completions), name="gold_answer")
                sources = _batch_value(kwargs.get("sources"), index, len(completions), name="sources")
                ranked = _batch_value(kwargs.get("canonical_ranked_search_results"), index, len(completions), name="canonical_ranked_search_results")
                truncated = _batch_value(kwargs.get("truncated", False), index, len(completions), name="truncated")
                audit = evaluate_reward_components(
                    completion, gold, ranked, task=task, sources=sources, truncated=truncated,
                )
                values.append(float(audit["components"][_component_name]))
            return values
        reward_func.__name__ = f"{component_name}_reward" if not component_name.endswith("penalty") else component_name
        functions.append(reward_func)
    return functions


def searchqa_rl_reward(completion: str, gold_answer: str, packed_evidence: str = "") -> float:
    """Compatibility scalar for archival callers; trainers use component rewards."""
    from text_feedback_dpo.scoring import score_searchqa

    score = score_searchqa(completion, gold_answer, packed_evidence)
    return 0.9 * float(score["exact_match"]) + 0.1 * float(score["f1"])


class PersistedReferenceDPOTrainerMixin:
    def _precompute_ref_logps(self, dataset, name: str, batch_size: int):
        required = {"ref_chosen_logps", "ref_rejected_logps"}
        if required <= set(dataset.column_names):
            return dataset
        return super()._precompute_ref_logps(dataset, name, batch_size)


def _load_rl_dataset(path: Path, tokenizer: object, completion_budget: int):
    dataset = _load_dataset(path)
    validate_rl_prompt_budget(dataset, tokenizer, completion_budget)
    return dataset


def run_sft(*, model_id: str, train_path: Path, eval_path: Path, output_dir: Path, config: dict) -> None:
    from trl import SFTConfig, SFTTrainer

    build_method_config("sft", max_length=MAX_SEQUENCE_LENGTH, max_steps=config["max_steps"])
    tokenizer = _tokenizer(config, model_id)
    train_dataset = _load_dataset(train_path)
    eval_dataset = _load_dataset(eval_path)
    args = SFTConfig(**_sft_args(config, output_dir))
    trainer = SFTTrainer(model=model_id, processing_class=tokenizer, peft_config=None, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset)
    trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))
    _save_final(trainer, output_dir)


def run_dpo(*, model_id: str, train_path: Path, eval_path: Path, output_dir: Path, config: dict) -> None:
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    build_method_config("dpo", max_length=MAX_SEQUENCE_LENGTH, max_steps=config["max_steps"])
    tokenizer = _tokenizer(config, model_id)
    if config.get("precomputed_ref_log_probs_path"):
        rows = _load_dataset(Path(config["precomputed_ref_log_probs_path"]))
        raw_rows = [dict(row) for row in rows]
        expected = build_reference_manifest(
            model=model_id, model_revision=config["model_revision"],
            reference_checkpoint_hash=config["reference_checkpoint_hash"],
            tokenizer=config.get("tokenizer_model", model_id), tokenizer_revision=config.get("tokenizer_revision", config["model_revision"]),
            data_hash=dataset_identity_hash(raw_rows), prompt_context_schema=config["prompt_context_schema"], max_length=MAX_SEQUENCE_LENGTH,
        )
        persisted_rows = load_precomputed_reference_log_probs(Path(config["precomputed_ref_log_probs_path"]), expected)
        train_dataset = Dataset.from_list(persisted_rows)
    else:
        train_dataset = _load_dataset(train_path)
    eval_dataset = _load_dataset(eval_path)
    args = DPOConfig(**_dpo_args(config, output_dir))
    trainer_class = DPOTrainer
    if config.get("precomputed_ref_log_probs_path"):
        trainer_class = type("PersistedReferenceDPOTrainer", (PersistedReferenceDPOTrainerMixin, DPOTrainer), {})
    trainer = trainer_class(model=model_id, processing_class=tokenizer, peft_config=None, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset)
    trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))
    _save_final(trainer, output_dir)


def _run_rl(*, method: str, model_id: str, train_path: Path, eval_path: Path, output_dir: Path, config: dict) -> None:
    from trl import GRPOConfig, GRPOTrainer

    if method == "dapo" and not config.get("dapo_enabled"):
        raise ValueError("DAPO requires explicit dapo_enabled=true after the primary DPO result is frozen")
    tokenizer = _tokenizer(config, model_id)
    args_dict = _rl_args(config, output_dir, method=method)
    train_dataset = _load_rl_dataset(train_path, tokenizer, args_dict["max_completion_length"])
    eval_dataset = _load_rl_dataset(eval_path, tokenizer, args_dict["max_completion_length"])
    args = GRPOConfig(**args_dict)
    trainer = GRPOTrainer(
        model=model_id,
        reward_funcs=build_component_reward_functions(),
        peft_config=None,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))
    _save_final(trainer, output_dir)


def run_grpo(*, model_id: str, train_path: Path, eval_path: Path, output_dir: Path, config: dict) -> None:
    _run_rl(method="grpo", model_id=model_id, train_path=train_path, eval_path=eval_path, output_dir=output_dir, config=config)


def run_dapo(*, model_id: str, train_path: Path, eval_path: Path, output_dir: Path, config: dict) -> None:
    _run_rl(method="dapo", model_id=model_id, train_path=train_path, eval_path=eval_path, output_dir=output_dir, config=config)
