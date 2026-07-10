import unittest

from text_feedback_dpo.guidance_policy import (
    validate_accumulated_guidance,
    validate_guidance_surface,
)


class GuidancePolicyTest(unittest.TestCase):
    def setUp(self):
        self.problem = "A question about a set of quantities and their relationship."
        self.gold = "42"
        self.evidence = ["The source states the requested relationship clearly."]

    def test_accepts_a_slight_abstract_hint(self):
        result = validate_guidance_surface(
            "Recheck how the quantities relate before performing the final calculation.",
            problem=self.problem,
            gold_answer=self.gold,
            evidence=self.evidence,
        )

        self.assertTrue(result["valid"])
        self.assertEqual(result["word_count"], 10)
        self.assertEqual(result["sentence_count"], 1)

    def test_rejects_short_hint_and_digits(self):
        for hint, reason in (("Recheck the calculation.", "word_count"), ("Recheck step 2 before answering this.", "digits")):
            result = validate_guidance_surface(
                hint,
                problem=self.problem,
                gold_answer=self.gold,
                evidence=self.evidence,
            )
            self.assertFalse(result["valid"])
            self.assertIn(reason, result["reasons"])

    def test_rejects_equations_operations_proper_nouns_and_copied_evidence(self):
        cases = (
            ("Recheck whether to multiply the quantities before the final response.", "explicit_operation"),
            ("Recheck Einstein's relation before providing the final response now.", "proper_noun"),
            ("The source states the requested relationship clearly before answering now.", "copied_evidence"),
        )
        for hint, reason in cases:
            result = validate_guidance_surface(
                hint,
                problem=self.problem,
                gold_answer=self.gold,
                evidence=self.evidence,
            )
            self.assertFalse(result["valid"])
            self.assertIn(reason, result["reasons"])

    def test_accumulated_guidance_is_validated_as_a_single_context(self):
        result = validate_accumulated_guidance(
            [
                "Recheck the relation between the quantities before answering fully.",
                "Confirm that the response matches the requested kind of answer carefully.",
            ],
            problem=self.problem,
            gold_answer=self.gold,
            evidence=self.evidence,
        )

        self.assertTrue(result["valid"])
        self.assertEqual(result["hint_count"], 2)
        self.assertIn("accumulated", result)

    def test_surface_invalid_hint_cannot_be_marked_valid_by_accumulation(self):
        result = validate_accumulated_guidance(
            ["Recheck step 2 before answering this."],
            problem=self.problem,
            gold_answer=self.gold,
            evidence=self.evidence,
        )

        self.assertFalse(result["valid"])
        self.assertIn("digits", result["reasons"])


if __name__ == "__main__":
    unittest.main()
