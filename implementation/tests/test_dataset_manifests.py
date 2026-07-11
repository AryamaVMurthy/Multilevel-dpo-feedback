import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from text_feedback_dpo.dataset_manifests import (
    canonical_row_hash,
    exclude_math_train_test_overlaps,
    materialize_paper_dataset,
    materialize_preflight_subset,
    sample_searchqa8k,
    split_math_train,
    split_math_validation_roles,
    split_gsm8k_validation_roles,
    split_gsm8k_train,
    validate_disjoint_splits,
    write_manifest_bundle,
)
from text_feedback_dpo.cli import run_materialize_preflight_subset
from text_feedback_dpo.io import append_jsonl_zst, read_jsonl_zst


class DatasetManifestTest(unittest.TestCase):
    def _gsm_rows(self, count: int) -> list[dict]:
        return [
            {"question": f"Question {index}", "answer": f"work\n#### {index}"}
            for index in range(count)
        ]

    def _search_rows(self, prefix: str, count: int) -> list[dict]:
        return [
            {
                "id": f"{prefix}-{index}",
                "question": f"Question {prefix} {index}",
                "answers": [f"Answer {index}"],
                "context": f"Context for answer {index}. " * (index % 3 + 1),
            }
            for index in range(count)
        ]

    def _math_rows(self) -> list[dict]:
        rows = []
        for subject in ("algebra", "geometry"):
            for level in (4, 5):
                for index in range(10):
                    rows.append(
                        {
                            "id": f"math-{subject}-{level}-{index}",
                            "problem": f"Problem {subject} {level} {index}",
                            "gold_answer": str(index),
                            "source_subject": subject,
                            "difficulty_level": level,
                        }
                    )
        rows.append(
            {
                "id": "math-algebra-3-0",
                "problem": "Lower level problem",
                "gold_answer": "0",
                "source_subject": "algebra",
                "difficulty_level": 3,
            }
        )
        rows.append(
            {
                "id": "math-geometry-unknown-0",
                "problem": "Unknown difficulty problem",
                "gold_answer": "0",
                "source_subject": "geometry",
                "difficulty_level": None,
            }
        )
        return rows

    def test_canonical_hash_is_order_independent_and_changes_with_content(self):
        left = canonical_row_hash({"b": 2, "a": [1, 2]})
        right = canonical_row_hash({"a": [1, 2], "b": 2})
        changed = canonical_row_hash({"a": [1, 3], "b": 2})

        self.assertEqual(left, right)
        self.assertNotEqual(left, changed)
        self.assertEqual(len(left), 64)

    def test_gsm_split_is_deterministic_and_disjoint(self):
        rows = self._gsm_rows(10)
        first = split_gsm8k_train(rows, seed=7, validation_count=3)
        second = split_gsm8k_train(rows, seed=7, validation_count=3)

        self.assertEqual(first, second)
        self.assertEqual(len(first["train"]), 7)
        self.assertEqual(len(first["validation"]), 3)
        self.assertEqual({row["source_key"] for row in first["train"]} & {row["source_key"] for row in first["validation"]}, set())
        self.assertTrue(all(row["dataset_role"] in {"train", "validation"} for rows in first.values() for row in rows))

    def test_searchqa_sampling_is_stratified_and_auxiliary_rows_are_disjoint(self):
        result = sample_searchqa8k(
            self._search_rows("train", 12),
            self._search_rows("validation", 8),
            self._search_rows("test", 6),
            seed=7,
            counts={"train": 5, "validation": 3, "test": 2},
            auxiliary_counts={"train": 2, "validation": 1},
        )

        self.assertEqual({key: len(value) for key, value in result.items()}, {
            "train": 5,
            "validation": 3,
            "test": 2,
            "hparam_train": 2,
            "hparam_validation": 1,
        })
        validate_disjoint_splits([row for rows in result.values() for row in rows])
        self.assertTrue(all(row["stratum"] for rows in result.values() for row in rows))

    def test_duplicate_source_key_or_normalized_question_fails(self):
        rows = [
            {"source_key": "train:1", "question": "What is X?", "dataset_role": "train"},
            {"source_key": "validation:2", "question": " what is x ? ", "dataset_role": "validation"},
        ]

        with self.assertRaisesRegex(ValueError, "normalized question"):
            validate_disjoint_splits(rows)

    def test_same_role_normalized_question_duplicate_is_preserved(self):
        rows = [
            {"source_key": "train:1", "question": "Same problem", "dataset_role": "train", "row_hash": "a"},
            {"source_key": "train:2", "question": " same problem ", "dataset_role": "train", "row_hash": "b"},
        ]
        validate_disjoint_splits(rows)

    def test_math_normalization_preserves_relational_operator_distinctions(self):
        rows = [
            {"source_key": "train:1", "problem": "Find x when k > 0.", "dataset_role": "train"},
            {"source_key": "train:2", "problem": "Find x when k < 0.", "dataset_role": "train"},
        ]
        validate_disjoint_splits(rows)

    def test_unexpected_source_key_duplicate_fails(self):
        rows = [
            {"source_key": "train:1", "question": "A", "dataset_role": "train"},
            {"source_key": "train:1", "question": "B", "dataset_role": "train"},
        ]

        with self.assertRaisesRegex(ValueError, "source_key"):
            validate_disjoint_splits(rows)

    def test_sampling_rejects_counts_larger_than_source(self):
        with self.assertRaisesRegex(ValueError, "requested count"):
            sample_searchqa8k(
                self._search_rows("train", 2),
                self._search_rows("validation", 2),
                self._search_rows("test", 2),
                seed=7,
                counts={"train": 3, "validation": 1, "test": 1},
                auxiliary_counts={"train": 0, "validation": 0},
            )

    def test_manifest_bundle_writes_compressed_roles_and_metadata(self):
        rows = split_gsm8k_train(self._gsm_rows(4), seed=7, validation_count=1)

        with TemporaryDirectory() as tmp:
            manifest = write_manifest_bundle(
                Path(tmp),
                rows,
                metadata={"dataset": "gsm8k", "revision": "abc"},
            )

            self.assertEqual(manifest["roles"], {"train": 3, "validation": 1})
            self.assertTrue((Path(tmp) / "manifest.json").is_file())
            self.assertTrue((Path(tmp) / "train.jsonl.zst").is_file())
            self.assertTrue((Path(tmp) / "validation.jsonl.zst").is_file())
            self.assertEqual(manifest["metadata"]["dataset"], "gsm8k")

    def test_gsm_validation_roles_partition_validation_and_are_written_as_nested_artifacts(self):
        rows = split_gsm8k_train(self._gsm_rows(6), seed=7, validation_count=3)
        roles = split_gsm8k_validation_roles(rows["validation"], seed=7, tune_count=2)
        self.assertEqual({key: len(value) for key, value in roles.items()}, {"tune": 2, "confirm": 1})
        self.assertEqual(
            {row["source_key"] for value in roles.values() for row in value},
            {row["source_key"] for row in rows["validation"]},
        )

        with TemporaryDirectory() as tmp:
            manifest = write_manifest_bundle(
                Path(tmp),
                rows,
                metadata={"dataset": "gsm8k", "revision": "abc"},
                nested_roles=roles,
            )
            self.assertEqual(manifest["nested_roles"], {"confirm": 1, "tune": 2})
            self.assertTrue((Path(tmp) / "validation_tune.jsonl.zst").is_file())
            self.assertTrue((Path(tmp) / "validation_confirm.jsonl.zst").is_file())

    def test_math_level45_split_is_stratified_deterministic_and_disjoint(self):
        first = split_math_train(self._math_rows(), seed=7)
        second = split_math_train(self._math_rows(), seed=7)

        self.assertEqual(first, second)
        self.assertEqual(len(first["train"]), 36)
        self.assertEqual(len(first["validation"]), 4)
        self.assertTrue(all(row["difficulty_level"] in {4, 5} for rows in first.values() for row in rows))
        self.assertEqual(
            {row["stratum"] for row in first["validation"]},
            {"math:algebra:level4", "math:algebra:level5", "math:geometry:level4", "math:geometry:level5"},
        )
        validate_disjoint_splits([row for rows in first.values() for row in rows])

    def test_math_nested_roles_partition_each_validation_stratum(self):
        rows = []
        for subject in ("algebra", "geometry"):
            for level in (4, 5):
                for index in range(30):
                    rows.append(
                        {
                            "id": f"math-{subject}-{level}-{index}",
                            "problem": f"Problem {subject} {level} {index}",
                            "gold_answer": str(index),
                            "source_subject": subject,
                            "difficulty_level": level,
                        }
                    )
        split = split_math_train(rows, seed=7)
        roles = split_math_validation_roles(split["validation"], seed=7)
        self.assertEqual(
            {row["source_key"] for rows in roles.values() for row in rows},
            {row["source_key"] for row in split["validation"]},
        )
        self.assertTrue(all(row["dataset_role"] == "validation_tune" for row in roles["tune"]))
        self.assertTrue(all(row["dataset_role"] == "validation_confirm" for row in roles["confirm"]))

    def test_math_train_test_overlap_is_quarantined_with_hash_evidence(self):
        train = [
            {"id": "train-1", "problem": "Find x + y."},
            {"id": "train-2", "problem": "Keep this problem."},
        ]
        test = [{"id": "test-1", "problem": " find x+y "}]
        kept, excluded = exclude_math_train_test_overlaps(train, test)
        self.assertEqual([row["id"] for row in kept], ["train-2"])
        self.assertEqual(excluded[0]["id"], "train-1")
        self.assertEqual(excluded[0]["reason"], "normalized_problem_present_in_official_test")
        self.assertEqual(len(excluded[0]["row_hash"]), 64)

    def test_math_materialization_requires_all_subjects_and_derives_primary_roles(self):
        subjects = (
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "prealgebra",
            "precalculus",
        )
        dataset = SimpleNamespace(
            name="math",
            source="EleutherAI/hendrycks_math",
            revision="a" * 40,
            source_counts={"train": 280, "validation": 0, "test": 7},
            seed=7,
            subjects=subjects,
            primary_levels=(4, 5),
            train_fraction=0.9,
            validation_tune_fraction=2 / 3,
        )
        config = SimpleNamespace(dataset=dataset)
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            for subject in subjects:
                subject_root = root / subject
                subject_root.mkdir(parents=True)
                train_rows = [
                    {
                        "problem": f"{subject} level {level} problem {index}",
                        "solution": f"Reasoning. \\boxed{{{index}}}",
                        "level": level,
                        "type": subject,
                    }
                    for level in (4, 5)
                    for index in range(20)
                ]
                test_rows = [
                    {
                        "problem": f"{subject} test problem",
                        "solution": "Reasoning. \\boxed{test}",
                        "level": 4,
                        "type": subject,
                    }
                ]
                (subject_root / "train.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in train_rows) + "\n",
                    encoding="utf-8",
                )
                (subject_root / "test.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in test_rows) + "\n",
                    encoding="utf-8",
                )
            with mock.patch("text_feedback_dpo.dataset_manifests.write_manifest_bundle", return_value={"ok": True}) as write:
                result = materialize_paper_dataset(config, root, Path(tmp) / "output")

        self.assertEqual(result["manifest"], {"ok": True})
        split_rows = write.call_args.args[1]
        nested_roles = write.call_args.kwargs["nested_roles"]
        self.assertEqual({key: len(value) for key, value in split_rows.items()}, {"train": 252, "validation": 28, "test": 7})
        self.assertEqual({key: len(value) for key, value in nested_roles.items()}, {"tune": 14, "confirm": 14})
        self.assertTrue(all(row["difficulty_level"] in {4, 5} for row in split_rows["train"]))
        self.assertTrue(all(row["dataset_role"] == "test" for row in split_rows["test"]))
        self.assertEqual(write.call_args.kwargs["metadata"]["subjects"], list(subjects))

    def test_preflight_subset_is_hash_selected_reproducible_and_auditable(self):
        rows = [{"id": f"row-{index}", "problem": f"Problem {index}"} for index in range(20)]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "validation.jsonl.zst"
            for row in rows:
                append_jsonl_zst(source, row)
            first = materialize_preflight_subset(
                source_path=source,
                output_path=root / "preflight-a.jsonl.zst",
                count=5,
                seed=20260710,
            )
            second = materialize_preflight_subset(
                source_path=source,
                output_path=root / "preflight-b.jsonl.zst",
                count=5,
                seed=20260710,
            )
            first_rows = read_jsonl_zst(root / "preflight-a.jsonl.zst")
            second_rows = read_jsonl_zst(root / "preflight-b.jsonl.zst")
            self.assertEqual(first_rows, second_rows)
            self.assertEqual(first["selected_ids"], [row["id"] for row in first_rows])
            self.assertEqual(first["selection_sha256"], second["selection_sha256"])
            self.assertEqual(first["count"], 5)
            self.assertTrue((root / "preflight-a.jsonl.zst.manifest.json").exists())

    def test_preflight_subset_cli_runner_returns_manifest(self):
        rows = [{"id": f"row-{index}", "problem": f"Problem {index}"} for index in range(6)]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "validation.jsonl.zst"
            output = root / "validation_preflight.jsonl.zst"
            for row in rows:
                append_jsonl_zst(source, row)

            manifest = run_materialize_preflight_subset(
                source_path=source,
                output_path=output,
                count=3,
                seed=20260710,
            )

            self.assertEqual(manifest["schema"], "paper-preflight-subset-v1")
            self.assertEqual(manifest["count"], 3)
            self.assertEqual(len(read_jsonl_zst(output)), 3)


if __name__ == "__main__":
    unittest.main()
