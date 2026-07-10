import unittest

from text_feedback_dpo.answer_evaluation import evaluate_gsm8k_answer, evaluate_searchqa_answer


class AnswerEvaluationTest(unittest.TestCase):
    def test_gsm8k_accepts_numeric_equivalents_with_currency_commas_and_decimals(self):
        for prediction, gold in (("$1,000.00", "1000"), ("-2.50", "-2.5"), ("The final value is 4", "4") ):
            result = evaluate_gsm8k_answer(prediction, gold)
            self.assertTrue(result["correct"], result)
            self.assertTrue(result["numeric_exact_match"])

    def test_gsm8k_rejects_missing_and_ambiguous_answers(self):
        missing = evaluate_gsm8k_answer("No numeric answer was provided.", "4")
        ambiguous = evaluate_gsm8k_answer("4 or 5", "4")

        self.assertFalse(missing["correct"])
        self.assertEqual(missing["error_code"], "missing_numeric_answer")
        self.assertTrue(ambiguous["ambiguous"])
        self.assertTrue(ambiguous["requires_model_judgment"])

    def test_searchqa_uses_aliases_f1_type_and_evidence(self):
        result = evaluate_searchqa_answer(
            "The New York Times",
            gold_answer="New York Times",
            answer_aliases=["New York Times", "The New York Times"],
            expected_answer_type="organization",
            actual_answer_type="organization",
            evidence=["The New York Times published the report."],
        )

        self.assertTrue(result["correct"])
        self.assertTrue(result["exact_match"])
        self.assertEqual(result["token_f1"], 1.0)
        self.assertTrue(result["answer_type_correct"])
        self.assertTrue(result["evidence_supported"])

    def test_searchqa_routes_ambiguous_and_unsupported_answers_explicitly(self):
        ambiguous = evaluate_searchqa_answer(
            "A or B",
            gold_answer="A",
            answer_aliases=["A", "B"],
            expected_answer_type="unknown",
            actual_answer_type="unknown",
            evidence=["The source mentions A."],
        )
        unsupported = evaluate_searchqa_answer(
            "Completely unrelated",
            gold_answer="A",
            answer_aliases=["A"],
            expected_answer_type="unknown",
            actual_answer_type="unknown",
            evidence=["The source mentions A."],
        )

        self.assertTrue(ambiguous["ambiguous"])
        self.assertTrue(ambiguous["requires_model_judgment"])
        self.assertFalse(unsupported["correct"])
        self.assertFalse(unsupported["evidence_supported"])


if __name__ == "__main__":
    unittest.main()
