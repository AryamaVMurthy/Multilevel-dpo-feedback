import unittest
from unittest import mock

from text_feedback_dpo.models import FakeModelProvider, PresencePenaltyLogitsProcessor, TransformersModelProvider


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
    def __call__(self, _prompt, return_tensors):
        return FakeEncoded({"input_ids": FakeTensor()})

    def decode(self, _tokens, skip_special_tokens):
        return "decoded"


class FakeModel:
    device = "cuda:0"

    def __init__(self):
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [[1, 2, 3]]


class ModelProviderTest(unittest.TestCase):
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

        output = provider.generate(
            "student",
            "prompt",
            max_new_tokens=11,
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            presence_penalty=1.5,
        )

        self.assertEqual(output, "decoded")
        self.assertEqual(fake_model.generate_kwargs["max_new_tokens"], 11)
        self.assertTrue(fake_model.generate_kwargs["do_sample"])
        self.assertEqual(fake_model.generate_kwargs["temperature"], 1.0)
        self.assertEqual(fake_model.generate_kwargs["top_p"], 0.95)
        self.assertEqual(fake_model.generate_kwargs["top_k"], 20)
        processors = fake_model.generate_kwargs["logits_processor"]
        self.assertEqual(len(processors), 1)
        self.assertIsInstance(processors[0], PresencePenaltyLogitsProcessor)
        self.assertEqual(processors[0].penalty, 1.5)


if __name__ == "__main__":
    unittest.main()
