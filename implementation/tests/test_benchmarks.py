import unittest

from text_feedback_dpo.benchmarks import (
    convert_gsm8k_row,
    convert_searchqa_row,
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
