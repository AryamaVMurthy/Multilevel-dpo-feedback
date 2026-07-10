import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from text_feedback_dpo.report import write_comparison_report, write_html_report


class ReportTest(unittest.TestCase):
    def test_comparison_report_contains_method_table_and_loss_chart(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "comparison.html"
            write_comparison_report(
                path,
                [
                    {"method": "standard_dpo", "train_loss": 0.69, "runtime": 2.1},
                    {"method": "grpo", "train_loss": 0.0, "runtime": 15.0},
                ],
            )
            html = path.read_text(encoding="utf-8")
            self.assertIn("standard_dpo", html)
            self.assertIn("grpo", html)
            self.assertIn("comparison_train_loss", html)
            self.assertIn("<svg", html)

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
