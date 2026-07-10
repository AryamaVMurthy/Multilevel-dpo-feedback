import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from text_feedback_dpo.dataset_manifests import (
    canonical_row_hash,
    sample_searchqa8k,
    split_gsm8k_validation_roles,
    split_gsm8k_train,
    validate_disjoint_splits,
    write_manifest_bundle,
)


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


if __name__ == "__main__":
    unittest.main()
