import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from text_feedback_dpo.report import write_html_report


class ReportContractTest(unittest.TestCase):
    def test_active_search_report_renders_answer_structure_retrieval_citation_lengths_and_runtime(self):
        metrics = {
            "report_type": "active_search",
            "exact_match": 0.75,
            "f1": 0.6,
            "parse_valid": 0.8,
            "malformed_rate": 0.2,
            "recall@1": 0.4,
            "mrr": 0.7,
            "citation_validity": 0.9,
            "valid_citation_rate": 0.9,
            "citation_precision": 0.8,
            "citation_coverage": 0.85,
            "lexical_cited_answer_support": 0.75,
            "unsupported_source_rate": 0.1,
            "duplicate_citation_rate": 0.02,
            "malformed_response": 0.2,
            "answer_words": {"mean": 2.0, "max": 4},
            "reasoning_words": {"mean": 8.0, "max": 16},
            "truncation_rate": 0.05,
            "truncated": 0.05,
            "timings": {"query_ms": 12.0, "search_ms": 3.0, "response_ms": 18.0},
            "custom_metric": "must remain visible",
        }
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.html"
            write_html_report(path, metrics, ["metrics.json"])
            rendered = path.read_text(encoding="utf-8")

        for text in (
            "Active-search metrics",
            "Answer accuracy",
            "exact_match",
            "parse_valid",
            "malformed_rate",
            "Retrieval metrics",
            "recall@1",
            "mrr",
            "Citation metrics",
            "citation_validity",
            "valid_citation_rate",
            "citation_precision",
            "citation_coverage",
            "lexical_cited_answer_support",
            "unsupported_source_rate",
            "duplicate_citation_rate",
            "malformed_response",
            "Length metrics",
            "answer_words",
            "reasoning_words",
            "truncation_rate",
            "truncated",
            "Timing metrics",
            "query_ms",
            "custom_metric",
            "metrics.json",
        ):
            with self.subTest(text=text):
                self.assertIn(text, rendered)

    def test_archival_report_is_explicitly_labeled(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "archival.html"
            write_html_report(
                path,
                {"report_type": "archival_short_answer", "exact_match": 0.5, "evidence_support": 0.5},
                [],
            )
            rendered = path.read_text(encoding="utf-8")
        self.assertIn("Archival short-answer baseline", rendered)
        self.assertIn("evidence_support", rendered)

    def test_malformed_metric_structure_fails_explicitly(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.html"
            with self.assertRaisesRegex(ValueError, "exact_match"):
                write_html_report(path, {"exact_match": {"mean": 0.5}}, [])


if __name__ == "__main__":
    unittest.main()
