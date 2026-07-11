from __future__ import annotations

from dataclasses import dataclass, field
import re


@dataclass(frozen=True)
class ModelGeneration:
    text: str
    prompt_tokens: int | None
    generated_tokens: int | None
    terminated: bool | None
    truncated: bool | None
    finish_reason: str


def normalize_model_generation(value: str | ModelGeneration) -> ModelGeneration:
    if isinstance(value, ModelGeneration):
        return value
    if isinstance(value, str):
        return ModelGeneration(
            text=value,
            prompt_tokens=None,
            generated_tokens=None,
            terminated=None,
            truncated=None,
            finish_reason="unavailable",
        )
    raise TypeError("model generation must be text or ModelGeneration")


class PresencePenaltyLogitsProcessor:
    """Subtract a fixed penalty from tokens already present in each sequence."""

    def __init__(self, penalty: float) -> None:
        if penalty < 0:
            raise ValueError("presence_penalty must be non-negative")
        self.penalty = penalty

    def __call__(self, input_ids: object, scores: object) -> object:
        if self.penalty == 0:
            return scores
        for batch_idx, sequence in enumerate(input_ids):
            unique_token_ids = set(sequence.tolist() if hasattr(sequence, "tolist") else sequence)
            if unique_token_ids:
                scores[batch_idx, list(unique_token_ids)] -= self.penalty
        return scores


_FINAL_ANSWER_PATTERN = re.compile(r"(?m)^FINAL:\s*\\boxed\{")


def complete_final_answer_end(text: str) -> int | None:
    """Return the end offset of a balanced FINAL boxed answer, if present."""

    match = _FINAL_ANSWER_PATTERN.search(text)
    if match is None:
        return None
    depth = 1
    for index in range(match.end(), len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                if not text[match.end() : index].strip():
                    return None
                return index + 1
    return None


class FinalAnswerStoppingCriteria:
    """Stop a single-example generation after a complete FINAL boxed answer."""

    def __init__(self, *, tokenizer: object, prompt_tokens: int) -> None:
        self.tokenizer = tokenizer
        self.prompt_tokens = prompt_tokens

    def __call__(self, input_ids: object, _scores: object, **_kwargs: object) -> object:
        if len(input_ids) != 1:
            raise ValueError("final-answer stopping requires single-example generation")
        generated = input_ids[0][self.prompt_tokens :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is required for final-answer stopping") from exc
        return torch.tensor(
            [complete_final_answer_end(text) is not None],
            dtype=torch.bool,
            device=getattr(input_ids, "device", None),
        )


def generate_model_result(
    *,
    tokenizer: object,
    model: object,
    prompt: str,
    generation_kwargs: dict[str, object],
) -> ModelGeneration:
    try:
        from transformers import LogitsProcessorList
    except ImportError as exc:
        raise ImportError("transformers is required for model generation") from exc

    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError(
            "model tokenizer does not support apply_chat_template; refusing raw-text generation "
            "because Qwen3 requires an explicit chat generation turn"
        )
    template_options = {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    if "enable_thinking" in generation_kwargs:
        enable_thinking = generation_kwargs["enable_thinking"]
        if not isinstance(enable_thinking, bool):
            raise ValueError("enable_thinking must be boolean when provided")
        template_options["enable_thinking"] = enable_thinking
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        **template_options,
    )
    if hasattr(model, "device"):
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
    max_new_tokens = int(generation_kwargs.get("max_new_tokens", 128))
    stop_after_final_answer = generation_kwargs.get("stop_after_final_answer", False)
    if not isinstance(stop_after_final_answer, bool):
        raise ValueError("stop_after_final_answer must be boolean when provided")
    temperature = float(generation_kwargs.get("temperature", 0.0))
    do_sample_value = generation_kwargs.get("do_sample", temperature > 0.0)
    if not isinstance(do_sample_value, bool):
        raise ValueError("do_sample must be boolean when provided")
    model_generation_kwargs: dict[str, object] = {
        **encoded,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample_value,
    }
    if do_sample_value:
        if temperature <= 0:
            raise ValueError("sampled generation requires a positive temperature")
        presence_penalty = float(generation_kwargs.get("presence_penalty", 0.0))
        logits_processor = LogitsProcessorList()
        if presence_penalty:
            logits_processor.append(PresencePenaltyLogitsProcessor(presence_penalty))
        model_generation_kwargs.update(
            temperature=temperature,
            top_p=float(generation_kwargs.get("top_p", 1.0)),
            top_k=int(generation_kwargs.get("top_k", 50)),
            min_p=float(generation_kwargs.get("min_p", 0.0)),
            logits_processor=logits_processor,
        )
    if stop_after_final_answer:
        from transformers import StoppingCriteriaList

        model_generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
            [FinalAnswerStoppingCriteria(tokenizer=tokenizer, prompt_tokens=int(encoded["input_ids"].shape[-1]))]
        )
    repetition_penalty = float(generation_kwargs.get("repetition_penalty", 1.0))
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be positive")
    if repetition_penalty != 1.0:
        model_generation_kwargs["repetition_penalty"] = repetition_penalty
    output_ids = model.generate(**model_generation_kwargs)
    prompt_tokens = int(encoded["input_ids"].shape[-1])
    generated = output_ids[0][prompt_tokens:]
    generated_tokens = len(generated)
    eos_value = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos_value is None:
        eos_value = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_value, int):
        eos_ids = {eos_value}
    elif isinstance(eos_value, (list, tuple, set)):
        eos_ids = {int(value) for value in eos_value}
    else:
        eos_ids = set()
    last_token: int | None = None
    if generated_tokens:
        raw_last = generated[-1]
        last_token = int(raw_last.item()) if hasattr(raw_last, "item") else int(raw_last)
    eos_terminated = last_token in eos_ids if last_token is not None else False
    decoded = tokenizer.decode(generated, skip_special_tokens=True)
    final_answer_end = complete_final_answer_end(decoded) if stop_after_final_answer else None
    final_answer_terminated = final_answer_end is not None
    terminated = eos_terminated or final_answer_terminated
    truncated = generated_tokens >= max_new_tokens and not terminated
    finish_reason = (
        "eos"
        if eos_terminated
        else ("final_answer" if final_answer_terminated else ("length" if truncated else "other"))
    )
    return ModelGeneration(
        text=decoded,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        terminated=terminated,
        truncated=truncated,
        finish_reason=finish_reason,
    )


class ModelProvider:
    def generate(self, role: str, prompt: str, **generation_kwargs: object) -> str:
        raise NotImplementedError

    def generate_result(self, role: str, prompt: str, **generation_kwargs: object) -> ModelGeneration:
        return ModelGeneration(
            text=self.generate(role, prompt, **generation_kwargs),
            prompt_tokens=None,
            generated_tokens=None,
            terminated=None,
            truncated=None,
            finish_reason="unavailable",
        )


@dataclass
class FakeModelProvider(ModelProvider):
    outputs: dict[str, str]
    prompts: list[tuple[str, str]] = field(default_factory=list)

    def generate(self, role: str, prompt: str, **generation_kwargs: object) -> str:
        self.prompts.append((role, prompt))
        if role not in self.outputs:
            raise ValueError(f"missing fake output for role: {role}")
        return self.outputs[role]


class TransformersModelProvider(ModelProvider):
    def __init__(
        self,
        *,
        model_ids: dict[str, str],
        model_revisions: dict[str, str] | None = None,
        allow_cpu_for_unit_tests: bool = False,
    ) -> None:
        self.model_ids = model_ids
        self.model_revisions = model_revisions or {}
        self.allow_cpu_for_unit_tests = allow_cpu_for_unit_tests
        self._loaded: dict[str, tuple[object, object]] = {}

    def _load(self, role: str) -> tuple[object, object]:
        if role not in self.model_ids:
            raise ValueError(f"missing model id for role: {role}")

        model_id = self.model_ids[role]
        if role in self._loaded:
            return self._loaded[role]
        if model_id in self._loaded:
            return self._loaded[model_id]

        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is required for TransformersModelProvider") from exc
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("transformers is required for TransformersModelProvider") from exc

        if not self.allow_cpu_for_unit_tests and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for model generation; refusing hidden CPU fallback")

        revision = self.model_revisions.get(role) or self.model_revisions.get(model_id)
        tokenizer_kwargs = {"trust_remote_code": True}
        model_kwargs = {"device_map": "auto" if torch.cuda.is_available() else None, "torch_dtype": "auto", "trust_remote_code": True}
        if revision:
            tokenizer_kwargs["revision"] = revision
            model_kwargs["revision"] = revision
        tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs,
        )
        self._loaded[model_id] = (tokenizer, model)
        return self._loaded[model_id]

    def generate(self, role: str, prompt: str, **generation_kwargs: object) -> str:
        return self.generate_result(role, prompt, **generation_kwargs).text

    def generate_result(self, role: str, prompt: str, **generation_kwargs: object) -> ModelGeneration:
        tokenizer, model = self._load(role)
        return generate_model_result(
            tokenizer=tokenizer,
            model=model,
            prompt=prompt,
            generation_kwargs=dict(generation_kwargs),
        )
