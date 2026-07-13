import unittest

from text_feedback_dpo.batch_generation import generate_batch


class BatchGenerationTest(unittest.TestCase):
    def test_batch_generation_calls_provider_once_and_preserves_order(self):
        calls = []

        def provider(prompts, **kwargs):
            calls.append((prompts, kwargs))
            return prompts

        outputs = generate_batch(provider, ["a", "b"], max_new_tokens=32)
        self.assertEqual(len(calls), 1)
        self.assertEqual(outputs[0]["response"], "a")
        self.assertEqual(outputs[1]["response"], "b")

    def test_batch_generation_rejects_wrong_output_cardinality(self):
        with self.assertRaisesRegex(ValueError, "cardinality"):
            generate_batch(lambda prompts, **kwargs: ["one"], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
