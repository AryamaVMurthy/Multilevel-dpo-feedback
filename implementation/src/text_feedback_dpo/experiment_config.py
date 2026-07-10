from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    config: str
    source: str
    revision: str
    source_counts: dict[str, int]
    splits: dict[str, int]
    validation_roles: dict[str, int]
    auxiliary_hparam: dict[str, int]
    seed: int


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float
    top_p: float
    top_k: int
    presence_penalty: float
    max_completion_tokens: int


@dataclass(frozen=True)
class LoraConfig:
    rank: int
    alpha: int
    dropout: float
    dtype: str
    quantization: str
    target_policy: str
    excluded_components: tuple[str, ...]


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    adam_betas: tuple[float, float]
    adam_epsilon: float
    weight_decay: float
    max_grad_norm: float
    scheduler: str
    warmup_fraction: float


@dataclass(frozen=True)
class DpoSearchConfig:
    learning_rates: tuple[float, ...]
    betas: tuple[float, ...]
    weight_decays: tuple[float, ...]
    warmup_fractions: tuple[float, ...]
    schedulers: tuple[str, ...]
    max_epochs: float
    effective_global_batch: int
    promote_counts: tuple[int, ...]
    tuning_seeds: tuple[int, ...]


@dataclass(frozen=True)
class GrpoSearchConfig:
    learning_rates: tuple[float, ...]
    kl_betas: tuple[float, ...]
    epsilon: float
    num_iterations: int
    num_generations: int
    loss_type: str
    sensitivity_loss_type: str
    scale_rewards: str
    mask_truncated_completions: bool
    promote_counts: tuple[int, ...]
    tuning_seeds: tuple[int, ...]


@dataclass(frozen=True)
class PaperExperimentConfig:
    schema_version: int
    experiment_id: str
    dataset: DatasetConfig
    models: dict[str, dict[str, str]]
    generation: GenerationConfig
    collection: dict[str, int | str]
    lora: LoraConfig
    optimizer: OptimizerConfig
    dpo_search: DpoSearchConfig
    grpo_search: GrpoSearchConfig
    training: dict[str, Any]
    slurm: dict[str, Any]
    require_freeze_manifest_for_test: bool


def _mapping(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping")
    return value


def _strict_keys(value: Mapping[str, Any], path: str, required: set[str]) -> None:
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"{path}.{missing[0]} is required")
    unknown = sorted(set(value) - required)
    if unknown:
        raise ValueError(f"{path}.{unknown[0]} is unknown")


def _positive_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _nonnegative_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return value


def _number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    return float(value)


def _number_tuple(value: object, path: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must be a non-empty list")
    return tuple(_number(item, f"{path}[{index}]") for index, item in enumerate(value))


def _int_tuple(value: object, path: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must be a non-empty list")
    return tuple(_positive_int(item, f"{path}[{index}]") for index, item in enumerate(value))


def _string_tuple(value: object, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{path} must be a non-empty string list")
    return tuple(value)


def _count_mapping(value: object, path: str, required: set[str]) -> dict[str, int]:
    mapping = _mapping(value, path)
    _strict_keys(mapping, path, required)
    return {key: _nonnegative_int(mapping[key], f"{path}.{key}") for key in sorted(required)}


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("pyyaml is required to load paper experiment configs") from exc
    if not path.is_file():
        raise FileNotFoundError(f"paper experiment config does not exist: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"paper experiment config must be a mapping: {path}")
    return value


def _parse_dataset(value: object) -> DatasetConfig:
    path = "dataset"
    mapping = _mapping(value, path)
    required = {
        "name",
        "config",
        "source",
        "revision",
        "source_counts",
        "splits",
        "validation_roles",
        "auxiliary_hparam",
        "seed",
    }
    _strict_keys(mapping, path, required)
    revision = mapping["revision"]
    if not isinstance(revision, str) or not _SHA_PATTERN.fullmatch(revision):
        raise ValueError("dataset.revision must be an immutable 40-character commit SHA")
    source_counts = _count_mapping(
        mapping["source_counts"], "dataset.source_counts", {"train", "validation", "test"}
    )
    splits = _count_mapping(mapping["splits"], "dataset.splits", {"train", "validation", "test"})
    validation_roles = _mapping(mapping["validation_roles"], "dataset.validation_roles")
    auxiliary_hparam = _mapping(mapping["auxiliary_hparam"], "dataset.auxiliary_hparam")
    name = str(mapping["name"])
    if name == "gsm8k":
        validation_roles = _count_mapping(
            validation_roles, "dataset.validation_roles", {"tune", "confirm"}
        )
        if auxiliary_hparam:
            raise ValueError("dataset.auxiliary_hparam must be empty for GSM8K")
        if splits["train"] + splits["validation"] != source_counts["train"]:
            raise ValueError("dataset.splits train and validation must partition GSM8K source train")
        if splits["test"] != source_counts["test"]:
            raise ValueError("dataset.splits.test must equal the GSM8K source test count")
        if sum(validation_roles.values()) != splits["validation"]:
            raise ValueError("dataset.validation_roles must partition dataset.splits.validation")
    elif name == "searchqa8k":
        if validation_roles:
            raise ValueError("dataset.validation_roles must be empty for SearchQA-8K")
        auxiliary_hparam = _count_mapping(
            auxiliary_hparam, "dataset.auxiliary_hparam", {"train", "validation"}
        )
        for split in ("train", "validation", "test"):
            if splits[split] > source_counts[split]:
                raise ValueError(f"dataset.splits.{split} exceeds its official source split")
        for split in ("train", "validation"):
            unused = source_counts[split] - splits[split]
            if auxiliary_hparam[split] > unused:
                raise ValueError(
                    f"dataset.auxiliary_hparam.{split} exceeds unused official {split} rows"
                )
    else:
        raise ValueError("dataset.name must be gsm8k or searchqa8k")
    return DatasetConfig(
        name=name,
        config=str(mapping["config"]),
        source=str(mapping["source"]),
        revision=revision,
        source_counts=source_counts,
        splits=splits,
        validation_roles=dict(validation_roles),
        auxiliary_hparam=dict(auxiliary_hparam),
        seed=_positive_int(mapping["seed"], "dataset.seed"),
    )


def _parse_models(value: object) -> dict[str, dict[str, str]]:
    mapping = _mapping(value, "models")
    _strict_keys(mapping, "models", {"student", "teacher", "evaluator"})
    parsed: dict[str, dict[str, str]] = {}
    for role in ("student", "teacher", "evaluator"):
        role_value = _mapping(mapping[role], f"models.{role}")
        _strict_keys(role_value, f"models.{role}", {"id", "revision"})
        revision = role_value["revision"]
        if not isinstance(revision, str) or not _SHA_PATTERN.fullmatch(revision):
            raise ValueError(f"models.{role}.revision must be an immutable 40-character commit SHA")
        parsed[role] = {"id": str(role_value["id"]), "revision": revision}
    return parsed


def _parse_generation(value: object) -> GenerationConfig:
    mapping = _mapping(value, "generation")
    required = {"temperature", "top_p", "top_k", "presence_penalty", "max_completion_tokens"}
    _strict_keys(mapping, "generation", required)
    max_tokens = _positive_int(mapping["max_completion_tokens"], "generation.max_completion_tokens")
    if max_tokens != 2048:
        raise ValueError("generation.max_completion_tokens must be exactly 2048")
    config = GenerationConfig(
        temperature=_number(mapping["temperature"], "generation.temperature"),
        top_p=_number(mapping["top_p"], "generation.top_p"),
        top_k=_positive_int(mapping["top_k"], "generation.top_k"),
        presence_penalty=_number(mapping["presence_penalty"], "generation.presence_penalty"),
        max_completion_tokens=max_tokens,
    )
    if (config.temperature, config.top_p, config.top_k, config.presence_penalty) != (1.0, 0.95, 20, 1.5):
        raise ValueError("generation must use temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5")
    return config


def _parse_lora(value: object) -> LoraConfig:
    mapping = _mapping(value, "lora")
    required = {"rank", "alpha", "dropout", "dtype", "quantization", "target_policy", "excluded_components"}
    _strict_keys(mapping, "lora", required)
    config = LoraConfig(
        rank=_positive_int(mapping["rank"], "lora.rank"),
        alpha=_positive_int(mapping["alpha"], "lora.alpha"),
        dropout=_number(mapping["dropout"], "lora.dropout"),
        dtype=str(mapping["dtype"]),
        quantization=str(mapping["quantization"]),
        target_policy=str(mapping["target_policy"]),
        excluded_components=_string_tuple(mapping["excluded_components"], "lora.excluded_components"),
    )
    if (config.rank, config.alpha, config.dropout) != (16, 32, 0.05):
        raise ValueError("lora must use rank=16, alpha=32, dropout=0.05")
    if config.dtype != "bfloat16" or config.quantization != "none":
        raise ValueError("lora must use bfloat16 with no quantization")
    if config.target_policy != "qwen35_text_linear":
        raise ValueError("lora.target_policy must be qwen35_text_linear")
    return config


def _parse_optimizer(value: object) -> OptimizerConfig:
    mapping = _mapping(value, "optimizer")
    if "warmup_ratio" in mapping:
        raise ValueError("optimizer.warmup_ratio is deprecated; use explicit warmup_fraction")
    required = {
        "name",
        "adam_beta1",
        "adam_beta2",
        "adam_epsilon",
        "weight_decay",
        "max_grad_norm",
        "scheduler",
        "warmup_fraction",
    }
    _strict_keys(mapping, "optimizer", required)
    config = OptimizerConfig(
        name=str(mapping["name"]),
        adam_betas=(
            _number(mapping["adam_beta1"], "optimizer.adam_beta1"),
            _number(mapping["adam_beta2"], "optimizer.adam_beta2"),
        ),
        adam_epsilon=_number(mapping["adam_epsilon"], "optimizer.adam_epsilon"),
        weight_decay=_number(mapping["weight_decay"], "optimizer.weight_decay"),
        max_grad_norm=_number(mapping["max_grad_norm"], "optimizer.max_grad_norm"),
        scheduler=str(mapping["scheduler"]),
        warmup_fraction=_number(mapping["warmup_fraction"], "optimizer.warmup_fraction"),
    )
    approved = ("adamw_torch_fused", (0.9, 0.999), 1e-8, 0.01, 1.0, "cosine", 0.05)
    actual = (
        config.name,
        config.adam_betas,
        config.adam_epsilon,
        config.weight_decay,
        config.max_grad_norm,
        config.scheduler,
        config.warmup_fraction,
    )
    if actual != approved:
        raise ValueError("optimizer does not match the approved fixed foundation")
    return config


def _parse_dpo_search(value: object) -> DpoSearchConfig:
    mapping = _mapping(value, "dpo_search")
    required = {
        "learning_rates",
        "betas",
        "weight_decays",
        "warmup_fractions",
        "schedulers",
        "max_epochs",
        "effective_global_batch",
        "promote_counts",
        "tuning_seeds",
    }
    _strict_keys(mapping, "dpo_search", required)
    config = DpoSearchConfig(
        learning_rates=_number_tuple(mapping["learning_rates"], "dpo_search.learning_rates"),
        betas=_number_tuple(mapping["betas"], "dpo_search.betas"),
        weight_decays=_number_tuple(mapping["weight_decays"], "dpo_search.weight_decays"),
        warmup_fractions=_number_tuple(mapping["warmup_fractions"], "dpo_search.warmup_fractions"),
        schedulers=_string_tuple(mapping["schedulers"], "dpo_search.schedulers"),
        max_epochs=_number(mapping["max_epochs"], "dpo_search.max_epochs"),
        effective_global_batch=_positive_int(mapping["effective_global_batch"], "dpo_search.effective_global_batch"),
        promote_counts=_int_tuple(mapping["promote_counts"], "dpo_search.promote_counts"),
        tuning_seeds=_int_tuple(mapping["tuning_seeds"], "dpo_search.tuning_seeds"),
    )
    if config.learning_rates != (2e-6, 5e-6, 1e-5) or config.betas != (0.05, 0.1, 0.3, 0.5):
        raise ValueError("dpo_search must contain the approved learning-rate and beta matrix")
    return config


def _parse_grpo_search(value: object) -> GrpoSearchConfig:
    mapping = _mapping(value, "grpo_search")
    required = {
        "learning_rates",
        "kl_betas",
        "epsilon",
        "num_iterations",
        "num_generations",
        "loss_type",
        "sensitivity_loss_type",
        "scale_rewards",
        "mask_truncated_completions",
        "promote_counts",
        "tuning_seeds",
    }
    _strict_keys(mapping, "grpo_search", required)
    config = GrpoSearchConfig(
        learning_rates=_number_tuple(mapping["learning_rates"], "grpo_search.learning_rates"),
        kl_betas=_number_tuple(mapping["kl_betas"], "grpo_search.kl_betas"),
        epsilon=_number(mapping["epsilon"], "grpo_search.epsilon"),
        num_iterations=_positive_int(mapping["num_iterations"], "grpo_search.num_iterations"),
        num_generations=_positive_int(mapping["num_generations"], "grpo_search.num_generations"),
        loss_type=str(mapping["loss_type"]),
        sensitivity_loss_type=str(mapping["sensitivity_loss_type"]),
        scale_rewards=str(mapping["scale_rewards"]),
        mask_truncated_completions=mapping["mask_truncated_completions"],
        promote_counts=_int_tuple(mapping["promote_counts"], "grpo_search.promote_counts"),
        tuning_seeds=_int_tuple(mapping["tuning_seeds"], "grpo_search.tuning_seeds"),
    )
    if not isinstance(config.mask_truncated_completions, bool):
        raise ValueError("grpo_search.mask_truncated_completions must be boolean")
    if config.learning_rates != (2e-6, 5e-6, 1e-5) or config.kl_betas != (0.0, 0.001, 0.01, 0.04):
        raise ValueError("grpo_search must contain the approved learning-rate and KL-beta matrix")
    expected = (0.2, 1, 4, "grpo", "dapo", "group", True)
    actual = (
        config.epsilon,
        config.num_iterations,
        config.num_generations,
        config.loss_type,
        config.sensitivity_loss_type,
        config.scale_rewards,
        config.mask_truncated_completions,
    )
    if actual != expected:
        raise ValueError("grpo_search does not match the approved original-GRPO protocol")
    return config


def load_paper_experiment(path: Path) -> PaperExperimentConfig:
    value = _load_yaml(path)
    required = {
        "schema_version",
        "experiment_id",
        "dataset",
        "models",
        "generation",
        "collection",
        "lora",
        "optimizer",
        "dpo_search",
        "grpo_search",
        "training",
        "slurm",
        "require_freeze_manifest_for_test",
    }
    _strict_keys(value, "config", required)
    collection = _mapping(value["collection"], "collection")
    _strict_keys(
        collection,
        "collection",
        {"max_guidance_steps", "max_guidance_regenerations", "shard_size", "artifact_schema"},
    )
    for field in ("max_guidance_steps", "max_guidance_regenerations", "shard_size"):
        _positive_int(collection[field], f"collection.{field}")
    training = _mapping(value["training"], "training")
    _strict_keys(training, "training", {"final_seeds", "max_epochs", "checkpoint_fractions"})
    _int_tuple(training["final_seeds"], "training.final_seeds")
    if len(training["final_seeds"]) != 3:
        raise ValueError("training.final_seeds must contain exactly three seeds")
    _number(training["max_epochs"], "training.max_epochs")
    _number_tuple(training["checkpoint_fractions"], "training.checkpoint_fractions")
    slurm = _mapping(value["slurm"], "slurm")
    _strict_keys(slurm, "slurm", {"account", "partition", "gpus_per_job", "max_walltime", "memory_limit_fraction"})
    if not str(slurm["account"]) or not str(slurm["partition"]):
        raise ValueError("slurm.account and slurm.partition must be non-empty")
    _positive_int(slurm["gpus_per_job"], "slurm.gpus_per_job")
    freeze_required = value["require_freeze_manifest_for_test"]
    if freeze_required is not True:
        raise ValueError("require_freeze_manifest_for_test must be true")
    return PaperExperimentConfig(
        schema_version=_positive_int(value["schema_version"], "schema_version"),
        experiment_id=str(value["experiment_id"]),
        dataset=_parse_dataset(value["dataset"]),
        models=_parse_models(value["models"]),
        generation=_parse_generation(value["generation"]),
        collection=dict(collection),
        lora=_parse_lora(value["lora"]),
        optimizer=_parse_optimizer(value["optimizer"]),
        dpo_search=_parse_dpo_search(value["dpo_search"]),
        grpo_search=_parse_grpo_search(value["grpo_search"]),
        training=dict(training),
        slurm=dict(slurm),
        require_freeze_manifest_for_test=freeze_required,
    )


def validate_paper_experiment(config: PaperExperimentConfig) -> None:
    if config.schema_version != 1:
        raise ValueError("schema_version must be 1")
    if not config.experiment_id.strip():
        raise ValueError("experiment_id must be non-empty")
