import unittest

from text_feedback_dpo.evaluation import evaluate_examples


VALID_MATH = "<plan>x</plan><reflect>Verification: 2 + 2 = 4</reflect><final>4</final>"
VALID_SEARCH = (
    "<plan>x</plan><reflect>Verification: evidence supports Ada Lovelace</reflect>"
    "<final>Ada Lovelace</final>"
)


class EvaluationTest(unittest.TestCase):
    def test_evaluator_reports_math_and_searchqa_metrics(self):
        examples = [
            {"id": "m1", "domain": "math", "gold_answer": "4"},
            {
                "id": "s1",
                "domain": "search_qa",
                "gold_answer": "Ada Lovelace",
                "answer_type": "person",
                "evidence": ["Ada Lovelace wrote notes on the Analytical Engine."],
            },
        ]
        rollouts = [
            {"id": "m1", "rollout": VALID_MATH, "generated_tokens": 12},
            {"id": "s1", "rollout": VALID_SEARCH, "generated_tokens": 15, "answer_type": "person"},
        ]

        result = evaluate_examples(examples, rollouts)

        self.assertEqual(result["common"]["final_answer_accuracy"], 1.0)
        self.assertEqual(result["common"]["format_valid_rate"], 1.0)
        self.assertEqual(result["common"]["average_generated_tokens"], 13.5)
        self.assertEqual(result["math"]["exact_accuracy"], 1.0)
        self.assertEqual(result["search_qa"]["exact_match"], 1.0)
        self.assertEqual(result["search_qa"]["token_f1"], 1.0)
        self.assertEqual(result["search_qa"]["answer_type_accuracy"], 1.0)
        self.assertEqual(result["search_qa"]["evidence_support_rate"], 1.0)

    def test_searchqa_requires_controlled_evidence_and_answer_type(self):
        examples = [{"id": "s1", "domain": "search_qa", "gold_answer": "Ada Lovelace"}]
        rollouts = [{"id": "s1", "rollout": VALID_SEARCH, "answer_type": "person"}]

        with self.assertRaisesRegex(ValueError, "controlled evidence"):
            evaluate_examples(examples, rollouts)


if __name__ == "__main__":
    unittest.main()
