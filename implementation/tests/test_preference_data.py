import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from text_feedback_dpo.cli import run_build_preferences
from text_feedback_dpo.preference_data import build_preference_datasets
from text_feedback_dpo.io import append_jsonl_zst, read_jsonl


class PreferenceDataTest(unittest.TestCase):
    def setUp(self):
        self.examples = [
            {"id": "m1", "domain": "math", "problem": "Compute the value.", "gold_answer": "4"},
            {"id": "m2", "domain": "math", "problem": "Compute another value.", "gold_answer": "7"},
            {"id": "m3", "domain": "math", "problem": "An unresolved value.", "gold_answer": "9"},
            {"id": "m4", "domain": "math", "problem": "A first-attempt success.", "gold_answer": "3"},
        ]
        self.attempts = [
            {"id": "m1", "attempt": 0, "prompt": "retry prompt should not be used", "response": "wrong-0", "result": {"correct": False}},
            {"id": "m1", "attempt": 1, "prompt": "retry prompt should not be used", "response": "wrong-1", "result": {"correct": False}},
            {"id": "m1", "attempt": 2, "prompt": "retry prompt should not be used", "response": "right-1", "result": {"correct": True}},
            {"id": "m2", "attempt": 0, "prompt": "base prompt", "response": "wrong-2", "result": {"correct": False}},
            {"id": "m2", "attempt": 1, "prompt": "retry prompt", "response": "right-2", "result": {"correct": True}},
            {"id": "m3", "attempt": 0, "prompt": "base prompt", "response": "wrong-3", "result": {"correct": False}},
            {"id": "m4", "attempt": 0, "prompt": "base prompt", "response": "right-4", "result": {"correct": True}},
        ]

    def test_standard_and_multilevel_use_first_correct_and_original_prompt(self):
        result = build_preference_datasets(
            attempts=self.attempts,
            examples=self.examples,
            seed=17,
            base_prompt_builder=lambda example: f"Solve only this original problem: {example['problem']}",
        )

        self.assertEqual(len(result["standard"]), 2)
        self.assertEqual(len(result["multilevel"]), 3)
        self.assertEqual(len(result["matched"]), len(result["standard"]))
        self.assertEqual({row["metadata"]["failed_attempt"] for row in result["standard"]}, {0})
        self.assertEqual(
            {row["metadata"]["failed_attempt"] for row in result["multilevel"]},
            {0, 1},
        )
        self.assertTrue(all("retry prompt" not in row["prompt"] for row in result["multilevel"]))
        self.assertEqual(result["metrics"]["unresolved_groups"], 1)
        self.assertEqual(result["metrics"]["standard_pairs"], 2)
        self.assertEqual(result["metrics"]["multilevel_pairs"], 3)
        self.assertEqual(len(result["response_sft"]), 3)
        self.assertEqual({row["group_id"] for row in result["response_sft"]}, {"m1", "m2", "m4"})
        self.assertTrue(all("retry prompt" not in row["prompt"] for row in result["response_sft"]))
        self.assertEqual([row["group_id"] for row in result["unresolved"]], ["m3"])
        self.assertEqual(result["unresolved"][0]["attempts"][0]["response"], "wrong-3")
        self.assertEqual(result["metrics"]["response_sft_rows"], 3)

    def test_matched_sampling_is_deterministic_and_has_auditable_metadata(self):
        first = build_preference_datasets(
            attempts=self.attempts,
            examples=self.examples,
            seed=29,
            base_prompt_builder=lambda example: f"Original: {example['problem']}",
        )
        second = build_preference_datasets(
            attempts=self.attempts,
            examples=self.examples,
            seed=29,
            base_prompt_builder=lambda example: f"Original: {example['problem']}",
        )

        self.assertEqual(first["matched"], second["matched"])
        self.assertTrue(all(row["metadata"]["matched"] for row in first["matched"]))
        self.assertTrue(all(row["metadata"]["prompt_hash"] for row in first["standard"]))
        self.assertTrue(all(row["metadata"]["chosen_hash"] for row in first["multilevel"]))

    def test_preference_export_retains_sft_and_unresolved_artifacts_with_hashes(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "train.jsonl.zst"
            collection = root / "collection.jsonl.zst"
            for example in self.examples:
                append_jsonl_zst(dataset, example)
                append_jsonl_zst(
                    collection,
                    {
                        "id": example["id"],
                        "attempts": [
                            {key: value for key, value in attempt.items() if key != "id"}
                            for attempt in self.attempts
                            if attempt["id"] == example["id"]
                        ],
                    },
                )

            metrics = run_build_preferences(
                collection_path=collection,
                dataset_path=dataset,
                output_dir=root / "preferences",
                seed=17,
            )

            output = root / "preferences"
            self.assertEqual(len(read_jsonl(output / "response_sft.jsonl")), 3)
            self.assertEqual(len(read_jsonl(output / "unresolved.jsonl")), 1)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], "paper-preference-v2")
            self.assertEqual(set(manifest["artifacts"]), {
                "matched", "multilevel", "response_sft", "standard", "unresolved"
            })
            self.assertTrue(all(len(value["sha256"]) == 64 for value in manifest["artifacts"].values()))
            self.assertEqual(metrics["response_sft_rows"], 3)

    def test_build_preferences_rejects_dataset_directory_with_actionable_error(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "math-materialized-v1"
            dataset_dir.mkdir()
            collection = root / "collection.jsonl.zst"
            append_jsonl_zst(
                collection,
                {"id": "m1", "attempts": self.attempts[:3]},
            )

            with self.assertRaisesRegex(
                ValueError,
                r"dataset_path must be a JSONL file; pass the concrete .*train\.jsonl\.zst path",
            ):
                run_build_preferences(
                    collection_path=collection,
                    dataset_path=dataset_dir,
                    output_dir=root / "preferences",
                    seed=17,
                )


if __name__ == "__main__":
    unittest.main()
