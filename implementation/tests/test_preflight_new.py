import unittest

from text_feedback_dpo.preflight import assess_preflight, select_preflight_rows, select_thinking_mode, summarize_response_quality


class PreflightQualityTest(unittest.TestCase):
    def setUp(self):
        self.examples = [
            {"id": "1", "question": "Who?", "gold_answer": "Ada", "packed_evidence": "Ada wrote it."},
            {"id": "2", "question": "Where?", "gold_answer": "California", "packed_evidence": "The valley is in California."},
        ]

    def test_train_dev_selection_is_deterministic_and_order_independent(self):
        first = select_preflight_rows(self.examples, sample_size=1, seed=7)
        second = select_preflight_rows(list(reversed(self.examples)), sample_size=1, seed=7)
        self.assertEqual(first, second)

    def test_quality_summary_reports_plain_answer_failures_and_lengths(self):
        metrics = summarize_response_quality(
            self.examples,
            [
                {"id": "1", "response": "Ada", "truncated": False},
                {"id": "2", "response": "<answer></answer>", "truncated": True},
            ],
        )
        self.assertEqual(metrics["examples"], 2)
        self.assertEqual(metrics["exact_match"], 0.5)
        self.assertEqual(metrics["nonempty_rate"], 1.0)
        self.assertEqual(metrics["truncation_rate"], 0.5)
        self.assertEqual(metrics["markup_rate"], 0.5)
        self.assertEqual(metrics["verbose_rate"], 0.0)
        self.assertEqual(metrics["answer_words"]["max"], 2)

    def test_missing_truncation_metadata_fails_instead_of_guessing(self):
        with self.assertRaisesRegex(ValueError, "truncated"):
            summarize_response_quality(self.examples, [{"id": "1", "response": "Ada"}, {"id": "2", "response": "California"}])

    def test_gate_rejects_structurally_broken_outputs(self):
        decision = assess_preflight({"nonempty_rate": 0.5, "copying_rate": 0.0, "truncation_rate": 0.0, "markup_rate": 0.1, "verbose_rate": 0.1})
        self.assertFalse(decision["promote"])
        self.assertIn("nonempty_rate", decision["failures"])
        self.assertIn("markup_rate", decision["failures"])
        self.assertIn("verbose_rate", decision["failures"])

    def test_thinking_mode_selection_uses_accuracy_after_quality_gate(self):
        direct = {"exact_match": 0.25, "f1": 0.30, "nonempty_rate": 1.0, "copying_rate": 0.0, "truncation_rate": 0.0, "markup_rate": 0.0, "verbose_rate": 0.0}
        two_pass = {"exact_match": 0.35, "f1": 0.40, "nonempty_rate": 1.0, "copying_rate": 0.0, "truncation_rate": 0.0, "markup_rate": 0.0, "verbose_rate": 0.0}
        self.assertEqual(select_thinking_mode({"direct": direct, "two_pass": two_pass})["selected"], "two_pass")


if __name__ == "__main__":
    unittest.main()
