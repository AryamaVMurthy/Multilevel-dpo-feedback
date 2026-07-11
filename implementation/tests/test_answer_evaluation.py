import unittest

from text_feedback_dpo.answer_evaluation import (
    evaluate_domain_answer,
    evaluate_gsm8k_answer,
    evaluate_math_answer,
    evaluate_searchqa_answer,
)


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

    def test_math_accepts_rational_decimal_and_simple_algebraic_equivalence(self):
        for prediction, gold in (
            ("\\frac{1}{2}", "0.5"),
            ("45/1024", "\\dfrac{45}{1024}"),
            ("9000000", "9,\\!000,\\!000"),
            ("\\frac{1}{1+\\frac{1}{2}}", "2/3"),
            ("\\boxed{(x+1)^2}", "x^2+2*x+1"),
            ("2*(a+b)", "2*a+2*b"),
        ):
            result = evaluate_math_answer(prediction, gold)
            self.assertTrue(result["correct"], result)
            self.assertEqual(result["evaluator_source"], "deterministic_math")

    def test_math_nested_fraction_exponent_is_deterministically_incorrect(self):
        result = evaluate_math_answer("\\sqrt{3}-3", "\\frac{1}{2^{98}}")

        self.assertFalse(result["correct"], result)
        self.assertFalse(result["requires_model_judgment"], result)
        self.assertIsNone(result["error_code"])

    def test_math_accepts_set_interval_and_unit_equivalence(self):
        cases = (
            ("\\{2, \\frac{1}{2}\\}", "{0.5, 2}"),
            ("(0, \\frac{1}{2}]", "(0, 0.5]"),
            ("200 \\text{minutes}", "200 minutes"),
        )
        for prediction, gold in cases:
            result = evaluate_math_answer(prediction, gold)
            self.assertTrue(result["correct"], result)

    def test_math_routes_ambiguous_or_unsupported_forms_to_model_judgment(self):
        for prediction, gold in (("1 or 2", "1"), ("a proof", "1"), ("2 cm", "2 meters")):
            result = evaluate_math_answer(prediction, gold)
            self.assertFalse(result["correct"])
            self.assertTrue(result["requires_model_judgment"])

    def test_domain_dispatch_uses_math_evaluator_only_for_official_math_rows(self):
        result = evaluate_domain_answer(
            domain="math",
            prediction="\\frac{1}{2}",
            example={"source": "EleutherAI/hendrycks_math", "gold_answer": "0.5"},
        )
        self.assertTrue(result["correct"])
        self.assertEqual(result["evaluator_source"], "deterministic_math")


if __name__ == "__main__":
    unittest.main()
