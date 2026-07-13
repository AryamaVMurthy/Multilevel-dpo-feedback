import unittest

from text_feedback_dpo.feedback import FeedbackFormatError, diagnose_attempt, parse_feedback
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
        with self.assertRaisesRegex(FeedbackFormatError, "gold answer token"):
            parse_feedback('{"hint":"Focus specifically on Lovelace."}', gold_answer="Ada Lovelace")
        with self.assertRaisesRegex(FeedbackFormatError, "meaningful"):
            parse_feedback('{"hint":"Recheck the entity."}', gold_answer="!!!")
        self.assertEqual(
            parse_feedback('{"hint":"Check who is directly associated."}', gold_answer="The Who").hint,
            "Check who is directly associated.",
        )

    def test_rejects_duplicate_json_keys(self):
        with self.assertRaisesRegex(FeedbackFormatError, "duplicate JSON key: hint"):
            parse_feedback('{"hint":"First direction.","hint":"Second direction."}', gold_answer="Ada Lovelace")

    def test_teacher_prompt_is_plain_and_controls_escalation_without_xml(self):
        prompt = build_teacher_prompt(
            {
                "question": "Who?",
                "gold_answer": "Ada",
                "sources": [{"source_id": "S001", "title": "Ada", "url": "https://example.test/ada", "snippet": "Ada evidence"}],
            },
            "Grace",
            [{"hint": "Look for the writer.", "level": 1}],
            raw_query="writer",
            retrieved_sources=[{"source_id": "S001", "title": "Ada", "url": "https://example.test/ada", "snippet": "Ada evidence"}],
            diagnostics={"responsible_region": "answer", "error_code": "answer_mismatch"},
        )
        self.assertIn('"escalation_level": 2', prompt)
        self.assertIn('Return exactly one strict JSON object with exactly this shape: {"hint":"..."}', prompt)
        self.assertIn("Grace", prompt)
        self.assertNotIn("<feedback>", prompt)
        self.assertNotIn("<teacher_task>", prompt)

    def test_diagnostics_select_the_earliest_failure_region_and_label_support_as_a_proxy(self):
        base = {
            "raw_query": "",
            "ranked_search_results": [],
            "raw_response": None,
            "truncation": {"query": False, "response": False},
            "cited_score": None,
            "error_code": "query_invalid_format",
        }
        self.assertEqual(diagnose_attempt(base)["responsible_region"], "query/retrieval")

        malformed = {**base, "raw_query": "writer", "ranked_search_results": [{"source_id": "S001"}], "error_code": "line_count", "cited_score": {"parse_valid": False}}
        self.assertEqual(diagnose_attempt(malformed)["responsible_region"], "response grammar/truncation")

        wrong_answer = {**malformed, "raw_response": "Answer: Grace", "error_code": None, "cited_score": {"parse_valid": True, "answer_correct": False, "lexical_cited_answer_support": 1.0}}
        self.assertEqual(diagnose_attempt(wrong_answer)["responsible_region"], "answer")

        unsupported = {**wrong_answer, "cited_score": {"parse_valid": True, "answer_correct": True, "lexical_cited_answer_support": 0.0, "citation_precision": 0.0}}
        diagnostic = diagnose_attempt(unsupported)
        self.assertEqual(diagnostic["responsible_region"], "lexical support proxy/citation selection")
        self.assertTrue(diagnostic["lexical_support_is_proxy"])


if __name__ == "__main__":
    unittest.main()
