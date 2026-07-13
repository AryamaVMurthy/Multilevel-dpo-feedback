import unittest

from text_feedback_dpo.cli import build_parser


class CLITest(unittest.TestCase):
    def test_exposes_only_searchqa_training_commands(self):
        parser = build_parser()
        for command in ("prepare-searchqa", "collect", "build-preferences", "build-sft-data", "generate", "evaluate", "report", "validate-run", "train-sft", "train-dpo", "train-grpo", "train-dapo"):
            parsed = parser.parse_args([command] + self._required_args(command))
            self.assertEqual(parsed.command, command)

    @staticmethod
    def _required_args(command):
        if command == "prepare-searchqa":
            return ["--source", "source", "--split", "train", "--tokenizer-model", "model", "--tokenizer-revision", "tok-rev", "--revision", "data-rev", "--output", "x.jsonl", "--max-evidence-tokens", "100"]
        if command == "collect":
            return ["--data", "x.jsonl", "--output", "y.jsonl", "--student-model", "student", "--teacher-model", "teacher", "--student-revision", "student-rev", "--teacher-revision", "teacher-rev", "--teacher-fallback-revision", "fallback-rev", "--teacher-quantization", "4bit", "--attention-implementation", "sdpa", "--student-device", "cuda:1", "--teacher-device", "cuda:0", "--trajectory-cache", "cache.jsonl", "--policy-hash", "p1"]
        if command == "build-preferences":
            return ["--trajectories", "x.jsonl", "--output", "y.jsonl"]
        if command == "build-sft-data":
            return ["--data", "x.jsonl", "--output", "y.jsonl"]
        if command == "evaluate":
            return ["--data", "x.jsonl", "--predictions", "y.jsonl", "--output", "z.json"]
        if command == "generate":
            return ["--data", "x.jsonl", "--output", "y.jsonl", "--model", "model", "--attention-implementation", "sdpa"]
        if command == "report":
            return ["--metrics", "x.json", "--output", "y.html"]
        if command == "validate-run":
            return ["--directory", "run", "--output", "y.json"]
        return ["--config", "config.yaml", "--train", "train.jsonl", "--output", "out"] + (["--eval", "eval.jsonl"] if command in {"train-sft", "train-dpo"} else [])


if __name__ == "__main__":
    unittest.main()
