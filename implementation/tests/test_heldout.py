import hashlib
import json
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from text_feedback_dpo.cli import run_evaluate_paper, run_freeze_baseline
from text_feedback_dpo.heldout import (
    build_baseline_evaluation_freeze,
    evaluate_checkpoint,
    merge_checkpoint_evaluations,
    validate_adapter_compatibility,
)
from text_feedback_dpo.models import ModelGeneration


class HeldoutTest(unittest.TestCase):
    def _evaluator(self, example, response):
        correct = response == "right"
        return {
            "correct": correct,
            "answer": "4" if correct else "5",
            "confidence": 0.9,
            "requires_model_judgment": False,
            "deterministic": {"numeric_exact_match": correct, "correct": correct},
        }

    def test_base_checkpoint_is_teacher_free_and_persists_raw_predictions(self):
        examples = [{"id": "m1", "domain": "math", "problem": "Compute this value.", "gold_answer": "4"}]
        prompts = []
        with TemporaryDirectory() as tmp:
            result = evaluate_checkpoint(
                examples=examples,
                generate=lambda prompt: (prompts.append(prompt) or "right"),
                evaluator=self._evaluator,
                output_dir=Path(tmp),
                checkpoint_kind="base",
                base_model_revision="base-rev",
                seed=17,
                test=False,
            )
            self.assertEqual(result["common"]["final_answer_accuracy"], 1.0)
            self.assertNotIn("gold answer", prompts[0].lower())
            predictions = json.loads((Path(tmp) / "predictions.jsonl").read_text().splitlines()[0])
            self.assertEqual(predictions["response"], "right")
            self.assertTrue(predictions["teacher_free"])

    def test_exact_generation_metadata_is_required_and_truncation_overrides_correctness(self):
        examples = [{"id": "m1", "domain": "math", "problem": "Compute this value.", "gold_answer": "4"}]
        generation = ModelGeneration(
            text="right",
            prompt_tokens=13,
            generated_tokens=8192,
            terminated=False,
            truncated=True,
            finish_reason="length",
        )
        with TemporaryDirectory() as tmp:
            metrics = evaluate_checkpoint(
                examples=examples,
                generate=lambda _prompt: generation,
                evaluator=self._evaluator,
                output_dir=Path(tmp),
                checkpoint_kind="base",
                base_model_revision="base-rev",
                seed=17,
                test=False,
                require_generation_metadata=True,
            )
            prediction = json.loads((Path(tmp) / "predictions.jsonl").read_text().splitlines()[0])
            self.assertEqual(prediction["prompt_tokens"], 13)
            self.assertEqual(prediction["generated_tokens"], 8192)
            self.assertEqual(prediction["finish_reason"], "length")
            self.assertTrue(prediction["truncated"])
            self.assertGreaterEqual(prediction["generation_latency_ms"], 0)
            self.assertGreaterEqual(prediction["evaluation_latency_ms"], 0)
            self.assertFalse(prediction["evaluator_result"]["correct"])
            self.assertEqual(metrics["common"]["truncation_rate"], 1.0)
            self.assertEqual(json.loads((Path(tmp) / "failures.jsonl").read_text() or "[]"), [])

    def test_final_answer_stop_is_valid_termination_metadata(self):
        examples = [{"id": "m1", "domain": "math", "problem": "Compute.", "gold_answer": "4"}]
        generation = ModelGeneration(
            text="FINAL: \\boxed{4}",
            prompt_tokens=11,
            generated_tokens=7,
            terminated=True,
            truncated=False,
            finish_reason="final_answer",
        )
        with TemporaryDirectory() as tmp:
            metrics = evaluate_checkpoint(
                examples=examples,
                generate=lambda _prompt: generation,
                evaluator=self._evaluator,
                output_dir=Path(tmp),
                checkpoint_kind="base",
                base_model_revision="base-rev",
                seed=17,
                test=False,
                require_generation_metadata=True,
            )
            prediction = json.loads((Path(tmp) / "predictions.jsonl").read_text().splitlines()[0])
            self.assertEqual(prediction["finish_reason"], "final_answer")
            self.assertTrue(prediction["terminated"])
            self.assertFalse(prediction["truncated"])
            self.assertEqual(metrics["common"]["finish_reason_counts"], {"final_answer": 1})

    def test_evaluation_failure_is_persisted_before_the_shard_stops(self):
        examples = [{"id": "m1", "domain": "math", "problem": "Compute.", "gold_answer": "4"}]
        with TemporaryDirectory() as tmp:
            output = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "model failed"):
                evaluate_checkpoint(
                    examples=examples,
                    generate=lambda _prompt: (_ for _ in ()).throw(RuntimeError("model failed")),
                    evaluator=self._evaluator,
                    output_dir=output,
                    checkpoint_kind="base",
                    base_model_revision="base-rev",
                    seed=17,
                    test=False,
                )
            failure = json.loads((output / "failures.jsonl").read_text().splitlines()[0])
            self.assertEqual(failure["id"], "m1")
            self.assertEqual(failure["stage"], "generation")
            self.assertEqual(failure["error_type"], "RuntimeError")

    def test_invalid_generation_metadata_is_recorded_as_a_failure(self):
        examples = [{"id": "m1", "domain": "math", "problem": "Compute.", "gold_answer": "4"}]
        with TemporaryDirectory() as tmp:
            output = Path(tmp)
            with self.assertRaisesRegex(ValueError, "metadata is unavailable"):
                evaluate_checkpoint(
                    examples=examples,
                    generate=lambda _prompt: "right",
                    evaluator=self._evaluator,
                    output_dir=output,
                    checkpoint_kind="base",
                    base_model_revision="base-rev",
                    seed=17,
                    test=False,
                    require_generation_metadata=True,
                )
            failure = json.loads((output / "failures.jsonl").read_text().splitlines()[0])
            self.assertEqual(failure["stage"], "generation_validation")

    def test_baseline_freeze_binds_model_evaluator_dataset_protocol_and_source(self):
        freeze = build_baseline_evaluation_freeze(
            experiment_id="paper-gsm",
            source_commit="a" * 40,
            config_sha256="b" * 64,
            dataset_manifest_sha256="c" * 64,
            student_model={"id": "student", "revision": "d" * 40},
            evaluator_model={"id": "evaluator", "revision": "e" * 40},
            prompt_protocol="qwen-nonthinking-final-r2",
            student_generation={"temperature": 1.0, "max_new_tokens": 16384},
            evaluator_generation={"do_sample": False, "max_new_tokens": 256},
            generation_seed=20260710,
        )
        self.assertEqual(freeze["schema"], "baseline-evaluation-freeze-v1")
        self.assertEqual(freeze["source_commit"], "a" * 40)
        self.assertEqual(freeze["checkpoint"]["revision"], "d" * 40)
        self.assertEqual(freeze["evaluator"]["revision"], "e" * 40)
        self.assertEqual(freeze["splits"], ["validation", "test"])

    def test_cli_freezes_and_evaluates_the_base_checkpoint_without_an_adapter(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "data"
            dataset_dir.mkdir()
            manifest = dataset_dir / "manifest.json"
            manifest.write_text('{"content_sha256":"dataset-content"}\n', encoding="utf-8")
            data = dataset_dir / "validation.jsonl"
            data.write_text(
                json.dumps({"id": "m1", "domain": "math", "problem": "Compute.", "gold_answer": "4"}) + "\n",
                encoding="utf-8",
            )
            freeze_path = root / "baseline-freeze.json"
            freeze = run_freeze_baseline(
                config_path=Path("configs/paper/gsm8k.yaml"),
                dataset_manifest_path=manifest,
                source_commit="a" * 40,
                output_path=freeze_path,
            )
            self.assertEqual(freeze["schema"], "baseline-evaluation-freeze-v1")

            with mock.patch(
                "text_feedback_dpo.cli.build_transformers_checkpoint_generator",
                return_value=lambda _prompt: "right",
            ) as build_generator, mock.patch(
                "text_feedback_dpo.cli._paper_evaluator",
                return_value=self._evaluator,
            ), mock.patch(
                "text_feedback_dpo.cli.evaluate_checkpoint",
                return_value={"common": {"final_answer_accuracy": 1.0}},
            ) as evaluate:
                result = run_evaluate_paper(
                    config_path=Path("configs/paper/gsm8k.yaml"),
                    checkpoint=None,
                    checkpoint_kind="base",
                    data_path=data,
                    split="validation",
                    output_dir=root / "baseline-validation",
                    freeze_manifest=freeze_path,
                    source_commit="a" * 40,
                    shard_index=0,
                    num_shards=1,
                )

            self.assertEqual(result["common"]["final_answer_accuracy"], 1.0)
            self.assertIsNone(build_generator.call_args.kwargs["adapter_dir"])
            self.assertEqual(evaluate.call_args.kwargs["checkpoint_kind"], "base")
            self.assertTrue(evaluate.call_args.kwargs["require_generation_metadata"])
            self.assertEqual(evaluate.call_args.kwargs["shard_index"], 0)
            self.assertEqual(evaluate.call_args.kwargs["num_shards"], 1)

    def test_test_evaluation_requires_freeze_and_cannot_repeat(self):
        examples = [{"id": "m1", "domain": "math", "problem": "Compute this value.", "gold_answer": "4"}]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            freeze = root / "freeze.json"
            freeze.write_text(json.dumps({"schema": "hyperparameter-freeze-v1"}), encoding="utf-8")
            evaluate_checkpoint(
                examples=examples,
                generate=lambda _prompt: "right",
                evaluator=self._evaluator,
                output_dir=root / "test",
                checkpoint_kind="adapter",
                base_model_revision="base-rev",
                seed=17,
                test=True,
                freeze_manifest=freeze,
            )
            with self.assertRaisesRegex(FileExistsError, "test marker"):
                evaluate_checkpoint(
                    examples=examples,
                    generate=lambda _prompt: "right",
                    evaluator=self._evaluator,
                    output_dir=root / "test",
                    checkpoint_kind="adapter",
                    base_model_revision="base-rev",
                    seed=17,
                    test=True,
                    freeze_manifest=freeze,
                )

    def test_completed_evaluation_shards_merge_with_exact_coverage(self):
        examples = [
            {"id": "m1", "domain": "math", "problem": "First.", "gold_answer": "4"},
            {"id": "m2", "domain": "math", "problem": "Second.", "gold_answer": "4"},
        ]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, example in enumerate(examples):
                evaluate_checkpoint(
                    examples=[example],
                    generate=lambda _prompt: ModelGeneration("right", 10, 5, True, False, "eos"),
                    evaluator=self._evaluator,
                    output_dir=root / f"shard-{index:04d}",
                    checkpoint_kind="base",
                    base_model_revision="base-rev",
                    seed=17,
                    test=False,
                    shard_index=index,
                    num_shards=2,
                    require_generation_metadata=True,
                )
                marker = json.loads(
                    (root / f"shard-{index:04d}" / "evaluation_complete.json").read_text()
                )
                self.assertRegex(marker["failures_sha256"], r"^[0-9a-f]{64}$")
            metrics = merge_checkpoint_evaluations(
                examples=examples,
                shard_root=root,
                expected_shards=2,
                output_dir=root / "merged",
                checkpoint_kind="base",
                base_model_revision="base-rev",
                seed=17,
                test=False,
                freeze_manifest=None,
            )
            self.assertEqual(metrics["common"]["examples"], 2)
            self.assertEqual(metrics["common"]["final_answer_accuracy"], 1.0)
            self.assertEqual(metrics["common"]["truncation_rate"], 0.0)
            self.assertTrue((root / "merged" / "evaluation_complete.json").exists())

    def test_per_example_generation_is_invariant_to_shard_count(self):
        examples = [
            {"id": f"m{index}", "domain": "math", "problem": f"Problem {index}.", "gold_answer": "4"}
            for index in range(4)
        ]

        def random_generation(_prompt):
            return f"sample-{random.random():.12f}"

        def evaluator(_example, _response):
            return {
                "correct": False,
                "answer": "5",
                "confidence": 0.9,
                "requires_model_judgment": False,
                "deterministic": {"numeric_exact_match": False, "correct": False},
            }
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluate_checkpoint(
                examples=examples,
                generate=random_generation,
                evaluator=evaluator,
                output_dir=root / "full",
                checkpoint_kind="base",
                base_model_revision="base-rev",
                seed=20260710,
                test=False,
            )
            full = {row["id"]: row for row in map(json.loads, (root / "full" / "predictions.jsonl").read_text().splitlines())}
            sharded = {}
            for shard_index in range(2):
                shard_examples = examples[shard_index * 2 : (shard_index + 1) * 2]
                evaluate_checkpoint(
                    examples=shard_examples,
                    generate=random_generation,
                    evaluator=evaluator,
                    output_dir=root / f"shard-{shard_index}",
                    checkpoint_kind="base",
                    base_model_revision="base-rev",
                    seed=20260710,
                    test=False,
                    shard_index=shard_index,
                    num_shards=2,
                )
                sharded.update(
                    {
                        row["id"]: row
                        for row in map(
                            json.loads,
                            (root / f"shard-{shard_index}" / "predictions.jsonl").read_text().splitlines(),
                        )
                    }
                )
            self.assertEqual(
                {row_id: row["response"] for row_id, row in full.items()},
                {row_id: row["response"] for row_id, row in sharded.items()},
            )
            self.assertEqual(
                {row_id: row["generation_seed"] for row_id, row in full.items()},
                {row_id: row["generation_seed"] for row_id, row in sharded.items()},
            )

    def test_merge_rejects_a_non_teacher_free_prediction_even_with_a_matching_file_hash(self):
        example = {"id": "m1", "domain": "math", "problem": "Compute.", "gold_answer": "4"}
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = root / "shard-0000"
            evaluate_checkpoint(
                examples=[example],
                generate=lambda _prompt: ModelGeneration("right", 10, 5, True, False, "eos"),
                evaluator=self._evaluator,
                output_dir=shard,
                checkpoint_kind="base",
                base_model_revision="base-rev",
                seed=17,
                test=False,
                shard_index=0,
                num_shards=1,
                require_generation_metadata=True,
            )
            predictions_path = shard / "predictions.jsonl"
            prediction = json.loads(predictions_path.read_text().strip())
            prediction["teacher_free"] = False
            predictions_path.write_text(json.dumps(prediction, sort_keys=True) + "\n", encoding="utf-8")
            marker_path = shard / "evaluation_complete.json"
            marker = json.loads(marker_path.read_text())
            marker["predictions_sha256"] = hashlib.sha256(predictions_path.read_bytes()).hexdigest()
            marker_path.write_text(json.dumps(marker), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "teacher-free"):
                merge_checkpoint_evaluations(
                    examples=[example],
                    shard_root=root,
                    expected_shards=1,
                    output_dir=root / "merged",
                    checkpoint_kind="base",
                    base_model_revision="base-rev",
                    seed=17,
                    test=False,
                    freeze_manifest=None,
                )

    def test_adapter_manifest_mismatch_fails_explicitly(self):
        with TemporaryDirectory() as tmp:
            adapter = Path(tmp)
            (adapter / "adapter_manifest.json").write_text(
                json.dumps(
                    {
                        "base_model_revision": "base-rev",
                        "lora_coverage_hash": "coverage-a",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "base_model_revision"):
                validate_adapter_compatibility(
                    adapter,
                    base_model_revision="other-rev",
                    lora_coverage_hash="coverage-a",
                )


if __name__ == "__main__":
    unittest.main()
