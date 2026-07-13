import json
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

from text_feedback_dpo.cli import build_parser


class Task7CLIExecutionTest(unittest.TestCase):
    def test_every_training_command_forwards_bounded_hardware_eval_and_resume_controls(self):
        parser = build_parser()
        config = {
            "student_model": "Qwen/Qwen3-4B-Base",
            "student_revision": "906bfd4b4dc7f14ee4320094d8b41684abff8539",
            "training": {},
        }
        with TemporaryDirectory() as directory:
            schema = Path(directory) / "schema.json"
            schema.write_text(json.dumps({"schema": 1}), encoding="utf-8")
            for method in ("sft", "dpo", "grpo", "dapo"):
                argv = [
                    f"train-{method}", "--config", "config.yaml", "--train", "train.jsonl",
                    "--eval", "eval.jsonl", "--output", "out", "--max-steps", "7",
                    "--max-length", "4096", "--per-device-train-batch-size", "2",
                    "--per-device-eval-batch-size", "4", "--dataloader-num-workers", "3",
                    "--gradient-accumulation-steps", "8", "--attention-implementation", "sdpa",
                    "--deepspeed-config", "zero3.json", "--resume-from-checkpoint", "checkpoint-4",
                    "--save-steps", "2", "--eval-steps", "2",
                ]
                if method == "dpo":
                    argv += [
                        "--ref-log-probs", "train.refs.jsonl", "--eval-ref-log-probs", "eval.refs.jsonl",
                        "--reference-checkpoint-hash", "a" * 64, "--prompt-context-schema", str(schema),
                    ]
                args = parser.parse_args(argv)
                target = f"text_feedback_dpo.trainers.run_{method}"
                with patch("text_feedback_dpo.cli.load_config", return_value=config), patch(
                    "text_feedback_dpo.training.validate_student_model_selection",
                    return_value=(config["student_model"], config["student_revision"]),
                ) as validate, patch(target) as run:
                    args.func(args)
                validate.assert_called_once_with(config, requested_model=None, requested_revision=None)
                kwargs = run.call_args.kwargs
                self.assertEqual(kwargs["eval_path"], Path("eval.jsonl"))
                forwarded = kwargs["config"]
                self.assertEqual(forwarded["max_steps"], 7)
                self.assertEqual(forwarded["per_device_train_batch_size"], 2)
                self.assertEqual(forwarded["per_device_eval_batch_size"], 4)
                self.assertEqual(forwarded["dataloader_num_workers"], 3)
                self.assertEqual(forwarded["gradient_accumulation_steps"], 8)
                self.assertEqual(forwarded["resume_from_checkpoint"], "checkpoint-4")
                self.assertEqual(forwarded["model_revision"], config["student_revision"])

    def test_training_rejects_any_non_4096_combined_length(self):
        args = build_parser().parse_args([
            "train-sft", "--config", "config.yaml", "--train", "train.jsonl", "--eval", "eval.jsonl",
            "--output", "out", "--max-steps", "1", "--max-length", "2048",
            "--per-device-train-batch-size", "1", "--per-device-eval-batch-size", "1",
            "--dataloader-num-workers", "0", "--gradient-accumulation-steps", "1",
            "--attention-implementation", "sdpa",
        ])
        with patch("text_feedback_dpo.cli.load_config", return_value={}):
            with self.assertRaisesRegex(ValueError, "exactly 4096"):
                args.func(args)

    def test_unsupported_packing_and_liger_controls_fail_explicitly(self):
        base = [
            "--config", "config.yaml", "--train", "train.jsonl", "--eval", "eval.jsonl",
            "--output", "out", "--max-steps", "1", "--max-length", "4096",
            "--per-device-train-batch-size", "1", "--per-device-eval-batch-size", "4",
            "--dataloader-num-workers", "0", "--gradient-accumulation-steps", "1",
            "--attention-implementation", "sdpa",
        ]
        for method, control, error in (
            ("sft", "--use-liger-kernel", "Liger"),
            ("grpo", "--packing", "only for SFT"),
        ):
            args = build_parser().parse_args([f"train-{method}", *base, control])
            with self.subTest(method=method), patch("text_feedback_dpo.cli.load_config", return_value={}):
                with self.assertRaisesRegex(ValueError, error):
                    args.func(args)


if __name__ == "__main__":
    unittest.main()
