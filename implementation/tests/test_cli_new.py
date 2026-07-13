import unittest

from text_feedback_dpo.cli import build_parser


class CLITest(unittest.TestCase):
    def test_exposes_only_searchqa_training_commands(self):
        parser = build_parser()
        for command in ("prepare-searchqa", "probe-model", "collect", "build-preferences", "build-sft-data", "generate", "evaluate", "preflight-quality", "select-thinking-mode", "report", "validate-run", "train-sft", "train-dpo", "train-grpo", "train-dapo"):
            parsed = parser.parse_args([command] + self._required_args(command))
            self.assertEqual(parsed.command, command)

    def test_generation_and_collection_expose_explicit_thinking_controls(self):
        parser = build_parser()
        generated = parser.parse_args(["generate"] + self._required_args("generate"))
        self.assertEqual(generated.student_thinking_mode, "direct")
        self.assertEqual(generated.scratchpad_max_new_tokens, 256)
        self.assertEqual(generated.max_new_tokens, 32)
        collected = parser.parse_args(["collect"] + self._required_args("collect"))
        self.assertEqual(collected.student_thinking_mode, "direct")
        self.assertTrue(collected.teacher_thinking)
        self.assertEqual(collected.answer_max_new_tokens, 32)
        self.assertEqual(collected.teacher_max_new_tokens, 96)
        self.assertEqual(collected.student_batch_size, 32)
        self.assertEqual(collected.teacher_batch_size, 8)
        self.assertEqual(collected.dataset_revision, "data-rev")
        self.assertEqual(collected.prompt_version, "plain-v2")
        self.assertEqual(collected.seed, 7)

    @staticmethod
    def _required_args(command):
        if command == "prepare-searchqa":
            return ["--source", "source", "--split", "train", "--tokenizer-model", "model", "--tokenizer-revision", "tok-rev", "--revision", "data-rev", "--output", "x.jsonl", "--max-evidence-tokens", "100"]
        if command == "probe-model":
            return ["--role", "teacher", "--model", "model", "--model-revision", "rev", "--teacher-quantization", "4bit", "--output", "probe.json"]
        if command == "collect":
            return ["--data", "x.jsonl", "--output", "y.jsonl", "--student-model", "student", "--teacher-model", "teacher", "--student-revision", "student-rev", "--teacher-revision", "teacher-rev", "--dataset-revision", "data-rev", "--prompt-version", "plain-v2", "--seed", "7", "--teacher-quantization", "4bit", "--attention-implementation", "sdpa", "--student-device", "cuda:1", "--teacher-device", "cuda:0", "--trajectory-cache", "cache.jsonl", "--policy-hash", "p1"]
        if command == "build-preferences":
            return ["--trajectories", "x.jsonl", "--output", "y.jsonl"]
        if command == "build-sft-data":
            return ["--data", "x.jsonl", "--output", "y.jsonl"]
        if command == "evaluate":
            return ["--data", "x.jsonl", "--predictions", "y.jsonl", "--output", "z.json"]
        if command == "preflight-quality":
            return ["--data", "x.jsonl", "--predictions", "y.jsonl", "--output", "z.json", "--samples", "samples.jsonl", "--split-name", "train-dev"]
        if command == "select-thinking-mode":
            return ["--direct", "direct.json", "--two-pass", "two.json", "--output", "choice.json"]
        if command == "generate":
            return ["--data", "x.jsonl", "--output", "y.jsonl", "--model", "model", "--attention-implementation", "sdpa"]
        if command == "report":
            return ["--metrics", "x.json", "--output", "y.html"]
        if command == "validate-run":
            return ["--directory", "run", "--output", "y.json"]
        return ["--config", "config.yaml", "--train", "train.jsonl", "--output", "out"] + (["--eval", "eval.jsonl"] if command in {"train-sft", "train-dpo"} else [])


if __name__ == "__main__":
    unittest.main()
