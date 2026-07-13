import unittest

from text_feedback_dpo.cli import build_parser


class CompareCLITest(unittest.TestCase):
    def test_compare_accepts_named_metric_runs(self):
        args = build_parser().parse_args(["compare", "--run", "dpo=dpo.json", "--run", "grpo=grpo.json", "--output", "comparison.json", "--html", "comparison.html"])
        self.assertEqual(args.run, ["dpo=dpo.json", "grpo=grpo.json"])


if __name__ == "__main__":
    unittest.main()
