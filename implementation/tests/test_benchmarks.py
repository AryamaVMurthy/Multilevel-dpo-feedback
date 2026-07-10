import unittest

from text_feedback_dpo.benchmarks import (
    convert_gsm8k_row,
    convert_math_row,
    convert_searchqa_row,
    extract_math_boxed_answer,
)


class BenchmarkConversionTest(unittest.TestCase):
    def test_gsm8k_conversion_extracts_answer_after_marker(self):
        row = convert_gsm8k_row(
            {"question": "What is 2 + 2?", "answer": "Work.\n#### 4"},
            index=3,
        )
        self.assertEqual(row["id"], "gsm8k-3")
        self.assertEqual(row["domain"], "math")
        self.assertEqual(row["gold_answer"], "4")

    def test_math_conversion_uses_last_balanced_boxed_answer_and_keeps_provenance(self):
        row = convert_math_row(
            {
                "problem": "Find the value.",
                "solution": "First \\boxed{wrong}. Therefore \\boxed{\\frac{3}{4}}.",
                "level": 4,
                "type": "algebra",
            },
            subject="algebra",
            source_split="train",
            index=9,
        )
        self.assertEqual(row["id"], "math-algebra-train-9")
        self.assertEqual(row["gold_answer"], "\\frac{3}{4}")
        self.assertEqual(row["difficulty_level"], 4)
        self.assertEqual(row["gold_answer_extraction"]["method"], "last_balanced_boxed")

    def test_math_boxed_extraction_refuses_missing_or_unbalanced_answers(self):
        with self.assertRaisesRegex(ValueError, "no boxed"):
            extract_math_boxed_answer("No final answer.")
        with self.assertRaisesRegex(ValueError, "unbalanced"):
            extract_math_boxed_answer("\\boxed{1")

    def test_searchqa_conversion_preserves_context_as_controlled_evidence(self):
        row = convert_searchqa_row(
            {
                "question": "Who wrote Hamlet?",
                "answers": ["William Shakespeare"],
                "context": "Hamlet was written by William Shakespeare.",
            },
            index=4,
        )
        self.assertEqual(row["id"], "searchqa-4")
        self.assertEqual(row["gold_answer"], "William Shakespeare")
        self.assertEqual(row["evidence"], ["Hamlet was written by William Shakespeare."])


if __name__ == "__main__":
    unittest.main()
