import unittest
from unittest import mock

from text_feedback_dpo.models import FakeModelProvider, TransformersModelProvider


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


if __name__ == "__main__":
    unittest.main()
