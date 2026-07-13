import unittest

from text_feedback_dpo.batch_generation import generate_batch


class BatchGenerationTest(unittest.TestCase):
    def test_batch_generation_calls_provider_once_and_preserves_order(self):
        calls = []

        def provider(prompts, **kwargs):
            calls.append((prompts, kwargs))
            return [f"<response><answer>{prompt}</answer><evidence>e</evidence></response>" for prompt in prompts]

        outputs = generate_batch(provider, ["a", "b"], max_new_tokens=32)
        self.assertEqual(len(calls), 1)
        self.assertEqual(outputs[0]["response"].split("<answer>")[1].split("</answer>")[0], "a")
        self.assertEqual(outputs[1]["response"].split("<answer>")[1].split("</answer>")[0], "b")

    def test_batch_generation_rejects_wrong_output_cardinality(self):
        with self.assertRaisesRegex(ValueError, "cardinality"):
            generate_batch(lambda prompts, **kwargs: ["one"], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
