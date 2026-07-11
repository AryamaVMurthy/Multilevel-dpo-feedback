import unittest
from types import ModuleType
from unittest import mock

import torch

from text_feedback_dpo.models import (
    FakeModelProvider,
    FinalAnswerStoppingCriteria,
    ModelGeneration,
    PresencePenaltyLogitsProcessor,
    TransformersModelProvider,
    complete_final_answer_end,
)


class FakeTensor:
    @property
    def shape(self):
        return (1, 2)

    def to(self, _device):
        return self


class FakeEncoded(dict):
    def to(self, _device):
        return self


class FakeTokenizer:
    def __init__(self):
        self.chat_template_calls = []
        self.eos_token_id = 3
        self.decoded_text = "decoded"

    def __call__(self, _prompt, return_tensors):
        return FakeEncoded({"input_ids": FakeTensor()})

    def apply_chat_template(self, messages, **kwargs):
        self.chat_template_calls.append((messages, kwargs))
        return FakeEncoded({"input_ids": FakeTensor()})

    def decode(self, _tokens, skip_special_tokens):
        return self.decoded_text


class FakeModel:
    device = "cuda:0"

    def __init__(self):
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [[1, 2, 3]]


class FakeLogitsProcessorList(list):
    pass


class FakeStoppingCriteriaList(list):
    pass


class ModelProviderTest(unittest.TestCase):
    def test_balanced_final_answer_detection_supports_nested_latex(self):
        text = "work\nFINAL: \\boxed{\\frac{1}{2}}"
        self.assertEqual(complete_final_answer_end(text), len(text))
        self.assertIsNone(complete_final_answer_end("FINAL: \\boxed{\\frac{1}{2}"))
        self.assertIsNone(complete_final_answer_end("FINAL: \\boxed{}"))
        self.assertIsNone(complete_final_answer_end("provisional \\boxed{4}"))

    def test_final_answer_stopping_criterion_returns_one_boolean_per_sequence(self):
        tokenizer = mock.Mock()
        tokenizer.decode.return_value = "Reason.\nFINAL: \\boxed{4}"
        criterion = FinalAnswerStoppingCriteria(tokenizer=tokenizer, prompt_tokens=2)

        result = criterion(torch.tensor([[10, 11, 12]]), None)

        self.assertEqual(result.dtype, torch.bool)
        self.assertEqual(tuple(result.shape), (1,))
        self.assertTrue(result.item())

    def test_fake_model_provider_returns_configured_text(self):
        provider = FakeModelProvider({"student": "hello"})
        self.assertEqual(provider.generate("student", "prompt"), "hello")

    def test_fake_model_provider_requires_known_role(self):
        provider = FakeModelProvider({"student": "hello"})
        with self.assertRaisesRegex(ValueError, "missing fake output"):
            provider.generate("teacher", "prompt")

    def test_transformers_provider_requires_cuda_unless_explicitly_allowed(self):
        with mock.patch.dict("sys.modules", {"torch": None}):
            provider = TransformersModelProvider(
                model_ids={"student": "Qwen/Qwen3.5-2B"},
                allow_cpu_for_unit_tests=False,
            )
            with self.assertRaisesRegex(ImportError, "torch"):
                provider.generate("student", "prompt")

    def test_transformers_provider_forwards_sampling_settings_and_presence_penalty(self):
        provider = TransformersModelProvider(
            model_ids={"student": "Qwen/Qwen3.5-2B"},
            allow_cpu_for_unit_tests=True,
        )
        fake_model = FakeModel()
        provider._loaded["student"] = (FakeTokenizer(), fake_model)

        fake_transformers = ModuleType("transformers")
        fake_transformers.LogitsProcessorList = FakeLogitsProcessorList
        fake_transformers.StoppingCriteriaList = FakeStoppingCriteriaList
        with mock.patch.dict("sys.modules", {"transformers": fake_transformers}):
            output = provider.generate(
                "student",
                "prompt",
                max_new_tokens=11,
                temperature=1.0,
                top_p=0.95,
                top_k=20,
                min_p=0.05,
                presence_penalty=1.5,
                repetition_penalty=1.05,
                stop_after_final_answer=True,
            )

        self.assertEqual(output, "decoded")
        self.assertEqual(fake_model.generate_kwargs["max_new_tokens"], 11)
        self.assertTrue(fake_model.generate_kwargs["do_sample"])
        self.assertEqual(fake_model.generate_kwargs["temperature"], 1.0)
        self.assertEqual(fake_model.generate_kwargs["top_p"], 0.95)
        self.assertEqual(fake_model.generate_kwargs["top_k"], 20)
        self.assertEqual(fake_model.generate_kwargs["min_p"], 0.05)
        self.assertEqual(fake_model.generate_kwargs["repetition_penalty"], 1.05)
        self.assertEqual(len(fake_model.generate_kwargs["stopping_criteria"]), 1)
        processors = fake_model.generate_kwargs["logits_processor"]
        self.assertEqual(len(processors), 1)
        self.assertIsInstance(processors[0], PresencePenaltyLogitsProcessor)
        self.assertEqual(processors[0].penalty, 1.5)

    def test_transformers_provider_returns_exact_generation_metadata(self):
        provider = TransformersModelProvider(
            model_ids={"student": "Qwen/Qwen3.5-2B"},
            allow_cpu_for_unit_tests=True,
        )
        provider._loaded["student"] = (FakeTokenizer(), FakeModel())
        fake_transformers = ModuleType("transformers")
        fake_transformers.LogitsProcessorList = FakeLogitsProcessorList

        with mock.patch.dict("sys.modules", {"transformers": fake_transformers}):
            result = provider.generate_result(
                "student",
                "prompt",
                max_new_tokens=4,
                do_sample=False,
                enable_thinking=True,
            )

        self.assertIsInstance(result, ModelGeneration)
        self.assertEqual(result.text, "decoded")
        self.assertEqual(result.prompt_tokens, 2)
        self.assertEqual(result.generated_tokens, 1)
        self.assertTrue(result.terminated)
        self.assertFalse(result.truncated)
        self.assertEqual(result.finish_reason, "eos")

    def test_length_limited_generation_is_explicitly_truncated(self):
        provider = TransformersModelProvider(
            model_ids={"student": "Qwen/Qwen3.5-2B"},
            allow_cpu_for_unit_tests=True,
        )
        tokenizer = FakeTokenizer()
        tokenizer.eos_token_id = 99
        model = FakeModel()
        provider._loaded["student"] = (tokenizer, model)
        fake_transformers = ModuleType("transformers")
        fake_transformers.LogitsProcessorList = FakeLogitsProcessorList

        with mock.patch.dict("sys.modules", {"transformers": fake_transformers}):
            result = provider.generate_result(
                "student",
                "prompt",
                max_new_tokens=1,
                do_sample=False,
            )

        self.assertFalse(result.terminated)
        self.assertTrue(result.truncated)
        self.assertEqual(result.finish_reason, "length")

    def test_complete_final_box_is_a_valid_generation_stop(self):
        provider = TransformersModelProvider(
            model_ids={"student": "Qwen/Qwen3.5-2B"},
            allow_cpu_for_unit_tests=True,
        )
        tokenizer = FakeTokenizer()
        tokenizer.eos_token_id = 99
        tokenizer.decoded_text = "Short derivation.\nFINAL: \\boxed{\\frac{1}{2}}"
        model = FakeModel()
        provider._loaded["student"] = (tokenizer, model)
        fake_transformers = ModuleType("transformers")
        fake_transformers.LogitsProcessorList = FakeLogitsProcessorList
        fake_transformers.StoppingCriteriaList = FakeStoppingCriteriaList

        with mock.patch.dict("sys.modules", {"transformers": fake_transformers}):
            result = provider.generate_result(
                "student",
                "Solve.",
                max_new_tokens=16384,
                do_sample=False,
                enable_thinking=False,
                stop_after_final_answer=True,
            )

        self.assertTrue(result.terminated)
        self.assertFalse(result.truncated)
        self.assertEqual(result.finish_reason, "final_answer")
        criteria = model.generate_kwargs["stopping_criteria"]
        self.assertIsInstance(criteria, FakeStoppingCriteriaList)
        self.assertIsInstance(criteria[0], FinalAnswerStoppingCriteria)
        self.assertTrue(criteria[0]([[1, 2, 3]], None))

    def test_greedy_generation_omits_sampling_only_kwargs(self):
        provider = TransformersModelProvider(
            model_ids={"evaluator": "Qwen/Qwen3.5-9B"},
            allow_cpu_for_unit_tests=True,
        )
        model = FakeModel()
        provider._loaded["evaluator"] = (FakeTokenizer(), model)
        fake_transformers = ModuleType("transformers")
        fake_transformers.LogitsProcessorList = FakeLogitsProcessorList

        with mock.patch.dict("sys.modules", {"transformers": fake_transformers}):
            provider.generate_result(
                "evaluator",
                "Return JSON.",
                max_new_tokens=16,
                do_sample=False,
                enable_thinking=False,
            )

        self.assertFalse(model.generate_kwargs["do_sample"])
        self.assertNotIn("temperature", model.generate_kwargs)
        self.assertNotIn("top_p", model.generate_kwargs)
        self.assertNotIn("top_k", model.generate_kwargs)
        self.assertNotIn("logits_processor", model.generate_kwargs)

    def test_transformers_provider_uses_a_chat_generation_turn(self):
        provider = TransformersModelProvider(
            model_ids={"student": "Qwen/Qwen3.5-2B"},
            allow_cpu_for_unit_tests=True,
        )
        tokenizer = FakeTokenizer()
        provider._loaded["student"] = (tokenizer, FakeModel())
        fake_transformers = ModuleType("transformers")
        fake_transformers.LogitsProcessorList = FakeLogitsProcessorList

        with mock.patch.dict("sys.modules", {"transformers": fake_transformers}):
            provider.generate("student", "Solve 2 + 2.", max_new_tokens=1)

        self.assertEqual(tokenizer.chat_template_calls[0][0], [{"role": "user", "content": "Solve 2 + 2."}])
        self.assertTrue(tokenizer.chat_template_calls[0][1]["add_generation_prompt"])
        self.assertTrue(tokenizer.chat_template_calls[0][1]["tokenize"])
        self.assertTrue(tokenizer.chat_template_calls[0][1]["return_dict"])

    def test_transformers_provider_forwards_local_qwen_thinking_flag(self):
        provider = TransformersModelProvider(
            model_ids={"evaluator": "Qwen/Qwen3.5-9B"},
            allow_cpu_for_unit_tests=True,
        )
        tokenizer = FakeTokenizer()
        provider._loaded["evaluator"] = (tokenizer, FakeModel())
        fake_transformers = ModuleType("transformers")
        fake_transformers.LogitsProcessorList = FakeLogitsProcessorList

        with mock.patch.dict("sys.modules", {"transformers": fake_transformers}):
            provider.generate(
                "evaluator",
                "Return JSON.",
                max_new_tokens=16,
                enable_thinking=False,
            )

        self.assertEqual(
            tokenizer.chat_template_calls[0][1]["enable_thinking"],
            False,
        )

    def test_cached_model_load_does_not_import_torch(self):
        provider = TransformersModelProvider(
            model_ids={"student": "Qwen/Qwen3.5-2B"},
            allow_cpu_for_unit_tests=True,
        )
        provider._loaded["student"] = (FakeTokenizer(), FakeModel())

        with mock.patch.dict("sys.modules", {"torch": None}):
            self.assertIsInstance(provider._load("student")[1], FakeModel)


if __name__ == "__main__":
    unittest.main()
