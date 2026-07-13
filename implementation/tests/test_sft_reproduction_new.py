import unittest

from text_feedback_dpo.monitoring import build_sft_reproduction_report
from text_feedback_dpo.runtime import GeneratedText


class SFTReproductionTest(unittest.TestCase):
    def test_report_counts_exact_empty_truncated_and_tasks_without_repair(self):
        rows = [
            {"id": "q1", "task": "query", "completion": " alpha query"},
            {"id": "r1", "task": "response", "completion": " answer\nReasoning: evidence [S001].\nSources: S001"},
        ]
        generated = {
            "q1": GeneratedText("alpha query", False),
            "r1": GeneratedText("", True),
        }
        records, summary = build_sft_reproduction_report(rows, generated)
        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["exact"], 1)
        self.assertEqual(summary["exact_rate"], 0.5)
        self.assertEqual(summary["empty"], 1)
        self.assertEqual(summary["truncated"], 1)
        self.assertEqual(summary["tasks"]["query"]["exact_rate"], 1.0)
        self.assertEqual(summary["tasks"]["response"]["exact_rate"], 0.0)
        self.assertEqual(records[1]["reference"], rows[1]["completion"].strip())
        self.assertEqual(records[1]["generated"], "")

    def test_report_rejects_missing_duplicate_or_unexpected_generation_ids(self):
        row = {"id": "q1", "task": "query", "completion": " query"}
        with self.assertRaisesRegex(ValueError, "generation ID parity"):
            build_sft_reproduction_report([row], {})
        with self.assertRaisesRegex(ValueError, "generation ID parity"):
            build_sft_reproduction_report([row], {"q1": GeneratedText("query", False), "extra": GeneratedText("x", False)})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_sft_reproduction_report([row, row], {"q1": GeneratedText("query", False)})


if __name__ == "__main__":
    unittest.main()
