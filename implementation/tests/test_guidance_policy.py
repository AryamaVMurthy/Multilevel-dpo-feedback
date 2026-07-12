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

    def test_accepts_free_two_sentence_hint_within_teacher_budget(self):
        result = validate_guidance_surface(
            "The first reduction changes the constraint. Recheck that relation before substituting values.",
            problem=self.problem,
            gold_answer=self.gold,
            evidence=self.evidence,
        )

        self.assertTrue(result["valid"], result)
        self.assertEqual(result["sentence_count"], 2)
        self.assertLessEqual(result["word_count"], 40)

    def test_rejects_only_too_short_or_direct_answer_surface_leaks(self):
        for hint, reason in (
            ("Recheck the calculation.", "word_count"),
            ("The correct result is 42 after checking.", "answer_disclosure"),
        ):
            result = validate_guidance_surface(
                hint,
                problem=self.problem,
                gold_answer=self.gold,
                evidence=self.evidence,
            )
            self.assertFalse(result["valid"])
            self.assertIn(reason, result["reasons"])

    def test_allows_operations_quantities_and_names_when_the_answer_is_not_disclosed(self):
        for hint in (
            "Recheck whether to multiply the quantities before the final response.",
            "Reconsider whether Samantha's name length uses the requested component.",
            "Compare the two daily portions before calculating the weekly total.",
        ):
            result = validate_guidance_surface(
                hint,
                problem=self.problem,
                gold_answer=self.gold,
                evidence=self.evidence,
            )
            self.assertTrue(result["valid"], result)

    def test_rejects_a_long_copied_evidence_phrase(self):
        result = validate_guidance_surface(
            "The source states the requested relationship clearly before answering now.",
            problem=self.problem,
            gold_answer=self.gold,
            evidence=self.evidence,
        )
        self.assertFalse(result["valid"])
        self.assertIn("copied_evidence", result["reasons"])

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
            ["The correct result is 42 after checking."],
            problem=self.problem,
            gold_answer=self.gold,
            evidence=self.evidence,
        )

        self.assertFalse(result["valid"])
        self.assertIn("answer_disclosure", result["reasons"])


if __name__ == "__main__":
    unittest.main()
