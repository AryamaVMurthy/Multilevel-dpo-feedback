import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from text_feedback_dpo.report import write_html_report


class ReportTest(unittest.TestCase):
    def test_report_contains_metric_table_and_attempt_chart(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.html"
            write_html_report(
                path,
                {"run_id": "r1", "success_by_attempt": {"0": 1, "1": 2}},
                training_history=[{"step": 1, "loss": 0.8}, {"step": 2, "loss": 0.4}],
            )
            html = path.read_text(encoding="utf-8")
            self.assertIn("<table>", html)
            self.assertIn("<svg", html)
            self.assertIn("success_by_attempt", html)
            self.assertIn("training_loss", html)


if __name__ == "__main__":
    unittest.main()
