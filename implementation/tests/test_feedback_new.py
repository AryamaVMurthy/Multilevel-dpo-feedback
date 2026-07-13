import unittest

from text_feedback_dpo.feedback import FeedbackFormatError, parse_feedback
from text_feedback_dpo.prompts import build_teacher_prompt


class FeedbackContractTest(unittest.TestCase):
    def test_accepts_exactly_one_short_answer_free_json_hint(self):
        feedback = parse_feedback('{"hint":"Look for the person directly associated with the algorithm."}', gold_answer="Ada Lovelace")
        self.assertEqual(feedback.hint, "Look for the person directly associated with the algorithm.")

    def test_rejects_extra_fields_long_hints_and_gold_leakage(self):
        with self.assertRaisesRegex(FeedbackFormatError, "exactly"):
            parse_feedback('{"hint":"Recheck the person.","critique":"Wrong entity"}', gold_answer="Ada Lovelace")
        with self.assertRaisesRegex(FeedbackFormatError, "24 words"):
            parse_feedback('{"hint":"' + "word " * 25 + '"}', gold_answer="Ada Lovelace")
        with self.assertRaisesRegex(FeedbackFormatError, "gold answer"):
            parse_feedback('{"hint":"The relevant person is Ada-Lovelace."}', gold_answer="Ada Lovelace")

    def test_teacher_prompt_is_plain_and_controls_escalation_without_xml(self):
        prompt = build_teacher_prompt(
            {"question": "Who?", "packed_evidence": "Ada evidence", "gold_answer": "Ada"},
            "Grace",
            [{"hint": "Look for the writer.", "level": 1}],
        )
        self.assertIn("Escalation level: 2", prompt)
        self.assertIn('Return exactly one JSON object: {"hint":"..."}', prompt)
        self.assertIn("Grace", prompt)
        self.assertNotIn("<feedback>", prompt)
        self.assertNotIn("<teacher_task>", prompt)


if __name__ == "__main__":
    unittest.main()
