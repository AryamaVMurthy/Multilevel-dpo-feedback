import unittest

from text_feedback_dpo.preference_data import build_preference_datasets


class PreferenceDataTest(unittest.TestCase):
    def setUp(self):
        self.examples = [
            {"id": "m1", "domain": "math", "problem": "Compute the value.", "gold_answer": "4"},
            {"id": "m2", "domain": "math", "problem": "Compute another value.", "gold_answer": "7"},
            {"id": "m3", "domain": "math", "problem": "An unresolved value.", "gold_answer": "9"},
        ]
        self.attempts = [
            {"id": "m1", "attempt": 0, "prompt": "retry prompt should not be used", "response": "wrong-0", "result": {"correct": False}},
            {"id": "m1", "attempt": 1, "prompt": "retry prompt should not be used", "response": "wrong-1", "result": {"correct": False}},
            {"id": "m1", "attempt": 2, "prompt": "retry prompt should not be used", "response": "right-1", "result": {"correct": True}},
            {"id": "m2", "attempt": 0, "prompt": "base prompt", "response": "wrong-2", "result": {"correct": False}},
            {"id": "m2", "attempt": 1, "prompt": "retry prompt", "response": "right-2", "result": {"correct": True}},
            {"id": "m3", "attempt": 0, "prompt": "base prompt", "response": "wrong-3", "result": {"correct": False}},
        ]

    def test_standard_and_multilevel_use_first_correct_and_original_prompt(self):
        result = build_preference_datasets(
            attempts=self.attempts,
            examples=self.examples,
            seed=17,
            base_prompt_builder=lambda example: f"Solve only this original problem: {example['problem']}",
        )

        self.assertEqual(len(result["standard"]), 2)
        self.assertEqual(len(result["multilevel"]), 3)
        self.assertEqual(len(result["matched"]), len(result["standard"]))
        self.assertEqual({row["metadata"]["failed_attempt"] for row in result["standard"]}, {0})
        self.assertEqual(
            {row["metadata"]["failed_attempt"] for row in result["multilevel"]},
            {0, 1},
        )
        self.assertTrue(all("retry prompt" not in row["prompt"] for row in result["multilevel"]))
        self.assertEqual(result["metrics"]["unresolved_groups"], 1)
        self.assertEqual(result["metrics"]["standard_pairs"], 2)
        self.assertEqual(result["metrics"]["multilevel_pairs"], 3)

    def test_matched_sampling_is_deterministic_and_has_auditable_metadata(self):
        first = build_preference_datasets(
            attempts=self.attempts,
            examples=self.examples,
            seed=29,
            base_prompt_builder=lambda example: f"Original: {example['problem']}",
        )
        second = build_preference_datasets(
            attempts=self.attempts,
            examples=self.examples,
            seed=29,
            base_prompt_builder=lambda example: f"Original: {example['problem']}",
        )

        self.assertEqual(first["matched"], second["matched"])
        self.assertTrue(all(row["metadata"]["matched"] for row in first["matched"]))
        self.assertTrue(all(row["metadata"]["prompt_hash"] for row in first["standard"]))
        self.assertTrue(all(row["metadata"]["chosen_hash"] for row in first["multilevel"]))


if __name__ == "__main__":
    unittest.main()
