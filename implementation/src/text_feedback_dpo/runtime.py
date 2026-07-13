from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


TOTAL_CONTEXT_TOKENS = 4096
RETRY_CAP_BUCKET_TOKENS = 256
PRIMARY_TEACHER_MODEL = "Qwen/Qwen3-32B"
FALLBACK_TEACHER_MODEL = "Qwen/Qwen3-14B"


class RuntimeErrorExplicit(RuntimeError):
    """Runtime failure that must never degrade to an implicit fallback."""


@dataclass(frozen=True)
class StudentGeneration:
    response: str
    scratchpad: str | None
    mode: str
    truncated: bool
    scratchpad_truncated: bool | None


@dataclass(frozen=True)
class GeneratedText:
    text: str
    truncated: bool


def bounded_teacher_outputs(
    prompts: list[str],
    *,
    prompt_token_counts: list[int],
    primary_max_new_tokens: int,
    retry_max_new_tokens: int,
    generate: Callable[..., list[str]],
    token_count: Callable[[str], int],
    validate_output: Callable[[str], bool] | None = None,
    validate_outputs: list[Callable[[str], bool]] | None = None,
) -> tuple[list[str], dict[str, object]]:
    """Retry outputs that violate the thinking or final-content contract."""
    if not prompts or len(prompt_token_counts) != len(prompts):
        raise ValueError("teacher retry requires nonempty prompt/token-count parity")
    if validate_output is not None and validate_outputs is not None:
        raise ValueError("teacher retry accepts either one validator or per-row validators, not both")
    if validate_outputs is not None and len(validate_outputs) != len(prompts):
        raise ValueError("teacher retry requires per-row validator parity")
    if not 0 < primary_max_new_tokens < retry_max_new_tokens < TOTAL_CONTEXT_TOKENS:
        raise ValueError("teacher retry caps must satisfy 0 < primary < retry < 4096")
    raw = generate(prompts, max_new_tokens=primary_max_new_tokens)
    if not isinstance(raw, list) or len(raw) != len(prompts) or any(not isinstance(item, str) for item in raw):
        raise RuntimeErrorExplicit("primary teacher generation cardinality/type mismatch")
    primary_output_token_counts = [token_count(item) for item in raw]
    final: list[str | None] = []
    malformed: list[int] = []
    invalid_content: list[int] = []

    def extract_valid_content(text: str, index: int) -> str:
        content = extract_qwen_final_content(text)
        validator = validate_outputs[index] if validate_outputs is not None else validate_output
        if validator is not None and validator(content) is not True:
            raise RuntimeErrorExplicit("teacher final content failed the output validator")
        return content

    for index, text in enumerate(raw):
        try:
            final.append(extract_valid_content(text, index))
        except RuntimeErrorExplicit:
            try:
                extract_qwen_final_content(text)
            except RuntimeErrorExplicit:
                malformed.append(index)
            else:
                invalid_content.append(index)
            final.append(None)
    report: dict[str, object] = {
        "primary_max_new_tokens": primary_max_new_tokens,
        "primary_output_token_counts": primary_output_token_counts,
        "primary_malformed_indices": malformed,
        "malformed_thinking_indices": malformed,
        "primary_non_exhausted_malformed_indices": [
            index for index in malformed if primary_output_token_counts[index] < primary_max_new_tokens
        ],
        "primary_invalid_content_indices": invalid_content,
        "retry_max_new_tokens": retry_max_new_tokens,
        "retry_indices": sorted(set(malformed + invalid_content)),
        "retry_reason": (
            "teacher_thinking_or_feedback_contract"
            if malformed and invalid_content
            else "teacher_thinking_budget_exhausted"
            if malformed
            else "teacher_feedback_contract"
            if invalid_content
            else "none"
        ),
        "retry_output_token_counts": [],
    }
    retry_indices = sorted(set(malformed + invalid_content))
    if retry_indices:
        def retry_cap(prompt_count: int) -> int:
            available = TOTAL_CONTEXT_TOKENS - prompt_count
            if available <= primary_max_new_tokens:
                raise RuntimeErrorExplicit(
                    "teacher retry prompt budget leaves no larger legal retry: "
                    f"prompt_tokens={prompt_count} available_output_tokens={available} "
                    f"primary_max_new_tokens={primary_max_new_tokens}"
                )
            cap = min(retry_max_new_tokens, available)
            bucketed = (cap // RETRY_CAP_BUCKET_TOKENS) * RETRY_CAP_BUCKET_TOKENS
            return bucketed if bucketed > primary_max_new_tokens else cap

        retry_caps_by_index = {index: retry_cap(prompt_token_counts[index]) for index in retry_indices}
        report["retry_output_caps"] = [retry_caps_by_index[index] for index in retry_indices]
        retry_output_counts_by_index: dict[int, int] = {}
        retry_malformed: list[int] = []
        retry_invalid_content: list[int] = []
        retry_groups: dict[int, list[int]] = {}
        for index in retry_indices:
            retry_groups.setdefault(retry_caps_by_index[index], []).append(index)
        for cap, group in sorted(retry_groups.items()):
            retry_raw = generate([prompts[index] for index in group], max_new_tokens=cap)
            if not isinstance(retry_raw, list) or len(retry_raw) != len(group) or any(
                not isinstance(item, str) for item in retry_raw
            ):
                raise RuntimeErrorExplicit("retry teacher generation cardinality/type mismatch")
            for original_index, text in zip(group, retry_raw, strict=True):
                retry_output_counts_by_index[original_index] = token_count(text)
                try:
                    final[original_index] = extract_valid_content(text, original_index)
                except RuntimeErrorExplicit:
                    try:
                        extract_qwen_final_content(text)
                    except RuntimeErrorExplicit:
                        retry_malformed.append(original_index)
                    else:
                        retry_invalid_content.append(original_index)
        report["retry_output_token_counts"] = [retry_output_counts_by_index[index] for index in retry_indices]
        report["retry_malformed_indices"] = sorted(retry_malformed)
        report["retry_invalid_content_indices"] = sorted(retry_invalid_content)
        if retry_malformed or retry_invalid_content:
            original_indices = sorted(retry_malformed + retry_invalid_content)
            raise RuntimeErrorExplicit(
                "teacher thinking retry exhausted or remained malformed at original indices: "
                + ",".join(str(index) for index in original_indices)
            )
    if any(item is None for item in final):
        raise RuntimeErrorExplicit("teacher retry did not populate every final output")
    return [str(item) for item in final], report


def validate_teacher_identity(
    model_id: str,
    *,
    revision: str | None,
    quantization: str,
    fallback_reason: str | None,
) -> str:
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("teacher requires a pinned revision")
    if quantization != "4bit":
        raise ValueError("Qwen3 teacher inference requires 4bit quantization")
    if model_id == PRIMARY_TEACHER_MODEL:
        if fallback_reason is not None:
            raise ValueError("primary Qwen3-32B teacher must not declare a fallback reason")
        return "primary_qwen3_32b_4bit"
    if model_id == FALLBACK_TEACHER_MODEL:
        if not isinstance(fallback_reason, str) or not fallback_reason.strip():
            raise ValueError("Qwen3-14B teacher requires an explicit documented fallback reason")
        return "fallback_qwen3_14b_4bit"
    raise ValueError("teacher must be Qwen/Qwen3-32B primary or Qwen/Qwen3-14B explicit fallback")


def set_generation_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("generation seed must be a nonnegative integer")
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required to seed model generation") from exc
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokenizer(model_id: str, *, revision: str | None = None):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers is required for model runtime") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision or None, trust_remote_code=False)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise RuntimeErrorExplicit(f"tokenizer for {model_id} has neither pad_token nor eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_student(model_id: str, *, revision: str | None = None, attention_implementation: str = "sdpa", device: str = "cuda:0"):
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError("torch and transformers are required for student runtime") from exc
    if not torch.cuda.is_available():
        raise RuntimeErrorExplicit("CUDA is required for full-finetuning student runtime")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision or None,
        device_map={"": device},
        torch_dtype=torch.bfloat16,
        attn_implementation=attention_implementation,
        trust_remote_code=False,
    )
    model.config.use_cache = True
    return model


def load_teacher(
    model_id: str,
    *,
    revision: str | None = None,
    quantization: str,
    fallback_reason: str | None = None,
    attention_implementation: str = "sdpa",
    device: str = "cuda:0",
):
    validate_teacher_identity(
        model_id,
        revision=revision,
        quantization=quantization,
        fallback_reason=fallback_reason,
    )
    try:
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError("torch and transformers are required for teacher runtime") from exc
    if not torch.cuda.is_available():
        raise RuntimeErrorExplicit("CUDA is required for teacher inference")
    kwargs: dict[str, Any] = {
        "device_map": {"": device},
        "trust_remote_code": False,
        "attn_implementation": attention_implementation,
    }
    kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    return AutoModelForCausalLM.from_pretrained(model_id, revision=revision, **kwargs)


def decode_generated_records(tokenizer, output_ids, *, input_length: int, max_new_tokens: int) -> list[GeneratedText]:
    records = []
    for output in output_ids:
        token_ids = output.tolist() if hasattr(output, "tolist") else list(output)
        generated = token_ids[input_length:]
        eos_index = generated.index(tokenizer.eos_token_id) if tokenizer.eos_token_id in generated else None
        if eos_index is not None:
            content_ids = generated[:eos_index]
            truncated = False
        else:
            pad_index = generated.index(tokenizer.pad_token_id) if tokenizer.pad_token_id in generated else None
            content_ids = generated[:pad_index] if pad_index is not None else generated
            truncated = pad_index is None and len(generated) >= max_new_tokens
        text = tokenizer.decode(content_ids, skip_special_tokens=True).strip()
        records.append(GeneratedText(text=text, truncated=truncated))
    return records


def _field(value: Any, name: str) -> Any:
    if hasattr(value, name):
        return getattr(value, name)
    if isinstance(value, dict):
        return value[name]
    return None


def _sequence_length(value: Any) -> int | None:
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[-1])
    if isinstance(value, list):
        if value and isinstance(value[0], list):
            return len(value[0])
        return len(value)
    return None


def _attention_lengths(encoded: Any) -> list[int]:
    attention = _field(encoded, "attention_mask")
    if attention is None:
        length = _sequence_length(_field(encoded, "input_ids"))
        if length is None:
            raise RuntimeErrorExplicit("tokenizer output has no inspectable input length")
        batch_size = getattr(_field(encoded, "input_ids"), "shape", (1,))[0]
        return [length] * int(batch_size)
    if hasattr(attention, "sum"):
        summed = attention.sum(dim=1)
        if hasattr(summed, "tolist"):
            return [int(value) for value in summed.tolist()]
    if isinstance(attention, list):
        return [sum(int(token) for token in row) if isinstance(row, list) else int(row) for row in attention]
    raise RuntimeErrorExplicit("tokenizer attention mask has no inspectable input lengths")


def generate_batch_records(
    model,
    tokenizer,
    prompts: list[str],
    *,
    max_new_tokens: int,
    min_new_tokens: int = 0,
    temperature: float,
    top_p: float,
    context_budget: int = TOTAL_CONTEXT_TOKENS,
) -> list[GeneratedText]:
    if not prompts:
        return []
    if context_budget != TOTAL_CONTEXT_TOKENS:
        raise ValueError(f"context_budget must be exactly {TOTAL_CONTEXT_TOKENS}")
    if max_new_tokens <= 0 or max_new_tokens > 4096:
        raise ValueError("max_new_tokens must be between 1 and 4096")
    if min_new_tokens < 0 or min_new_tokens > max_new_tokens:
        raise ValueError("min_new_tokens must be between 0 and max_new_tokens")
    max_input_tokens = TOTAL_CONTEXT_TOKENS - max_new_tokens
    if max_input_tokens <= 0:
        raise ValueError("max_new_tokens leaves no room for an input within the 4096-token limit")
    try:
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    except Exception as exc:
        raise RuntimeErrorExplicit(
            "tokenizer failed while truncation was disabled; inspect prompt length and tokenizer configuration"
        ) from exc
    input_lengths = _attention_lengths(encoded)
    if len(input_lengths) != len(prompts):
        raise RuntimeErrorExplicit(
            f"tokenizer input cardinality mismatch: expected {len(prompts)}, got {len(input_lengths)}"
        )
    over_budget = [index for index, length in enumerate(input_lengths) if length > max_input_tokens]
    if over_budget:
        raise RuntimeErrorExplicit(
            f"input truncation refused: prompt index {over_budget[0]} has {input_lengths[over_budget[0]]} tokens, "
            f"but only {max_input_tokens} fit within the explicit {TOTAL_CONTEXT_TOKENS}-token total budget"
        )
    encoded = encoded.to(model.device)
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": min_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        kwargs.update(temperature=temperature, top_p=top_p)
    output_ids = model.generate(**encoded, **kwargs)
    input_length = encoded.input_ids.shape[1]
    return decode_generated_records(tokenizer, output_ids, input_length=input_length, max_new_tokens=max_new_tokens)


def generate_batch(model, tokenizer, prompts: list[str], *, max_new_tokens: int, temperature: float, top_p: float) -> list[str]:
    return [
        record.text
        for record in generate_batch_records(
            model, tokenizer, prompts, max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p,
        )
    ]


def render_teacher_prompts(tokenizer, prompts: list[str], *, enable_thinking: bool = True) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        for prompt in prompts
    ]


def extract_qwen_final_content(text: str) -> str:
    stripped = text.strip()
    has_open = "<think>" in stripped
    has_close = "</think>" in stripped
    if has_open != has_close:
        raise RuntimeErrorExplicit("unterminated or malformed Qwen thinking block")
    if has_close:
        stripped = stripped.rsplit("</think>", maxsplit=1)[1].strip()
    if not stripped:
        raise RuntimeErrorExplicit("teacher generation has no final content after private thinking")
    return stripped


def generate_student_batch(
    model,
    tokenizer,
    prompts: list[str],
    *,
    mode: str,
    scratchpad_max_new_tokens: int,
    answer_max_new_tokens: int,
    temperature: float,
    top_p: float,
    generation_fn: Callable[..., list[GeneratedText]] | None = None,
    visible_instruction: str | None = None,
    scratchpad_instruction: str | None = None,
) -> list[StudentGeneration]:
    if mode not in {"direct", "two_pass"}:
        raise ValueError("student thinking mode must be direct or two_pass")
    generate = generation_fn or generate_batch_records
    if mode == "direct":
        responses = generate(
            model, tokenizer, prompts, max_new_tokens=answer_max_new_tokens,
            temperature=temperature, top_p=top_p,
        )
        if len(responses) != len(prompts):
            raise RuntimeErrorExplicit("student answer batch cardinality mismatch")
        return [
            StudentGeneration(response=response.text, scratchpad=None, mode=mode, truncated=response.truncated, scratchpad_truncated=None)
            for response in responses
        ]
    if scratchpad_max_new_tokens <= 0:
        raise ValueError("scratchpad_max_new_tokens must be positive in two_pass mode")
    private_instruction = scratchpad_instruction or "reason from the evidence before answering"
    scratchpad_prompts = [
        f"{prompt}\n\nPrivate scratchpad: {private_instruction} Use plain prose without XML, JSON, tags, or code fences. This text will not be scored."
        for prompt in prompts
    ]
    scratchpad_records = generate(
        model, tokenizer, scratchpad_prompts, max_new_tokens=scratchpad_max_new_tokens,
        temperature=temperature, top_p=top_p,
    )
    if len(scratchpad_records) != len(prompts):
        raise RuntimeErrorExplicit("student scratchpad batch cardinality mismatch")
    scratchpads = [record.text for record in scratchpad_records]
    instruction = visible_instruction or "Return only a noun-phrase answer in plain text with no explanation, using at most 8 words. Never restate the clue; if uncertain, give the best short guess. Do not use XML, JSON, tags, code fences, or labels.\nAnswer:"
    answer_prompts = [
        f"{prompt}\n\nPrivate scratchpad (do not repeat or imitate its formatting):\n{scratchpad}\n\n{instruction}"
        for prompt, scratchpad in zip(prompts, scratchpads, strict=True)
    ]
    response_records = generate(
        model, tokenizer, answer_prompts, max_new_tokens=answer_max_new_tokens,
        temperature=temperature, top_p=top_p,
    )
    if len(response_records) != len(prompts):
        raise RuntimeErrorExplicit("student answer batch cardinality mismatch")
    return [
        StudentGeneration(
            response=response.text,
            scratchpad=scratchpad.text,
            mode=mode,
            truncated=response.truncated,
            scratchpad_truncated=scratchpad.truncated,
        )
        for response, scratchpad in zip(response_records, scratchpad_records, strict=True)
    ]
