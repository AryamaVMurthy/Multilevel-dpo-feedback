import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from text_feedback_dpo.io import append_jsonl_zst, read_jsonl_zst
from text_feedback_dpo.sharding import (
    complete_shard,
    merge_completed_shards,
    shard_rows,
    write_progress,
)


class ShardingTest(unittest.TestCase):
    def test_membership_is_deterministic_complete_and_non_overlapping(self):
        rows = [{"id": f"row-{index}", "value": index} for index in range(10)]
        shards = [shard_rows(rows, shard_index=index, num_shards=3) for index in range(3)]

        self.assertEqual([len(shard) for shard in shards], [4, 3, 3])
        self.assertEqual([row["id"] for shard in shards for row in shard], [row["id"] for row in rows])
        self.assertEqual(shards[0], shard_rows(rows, shard_index=0, num_shards=3))
        self.assertEqual(len({row["id"] for shard in shards for row in shard}), len(rows))

    def test_progress_resumes_after_last_fsynced_example_and_rejects_config_change(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.json"
            write_progress(
                path,
                config_hash="config-a",
                dataset_manifest_hash="dataset-a",
                protocol_hash="protocol-a",
                shard_index=1,
                num_shards=3,
                last_completed_local_index=4,
                records_written=5,
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["last_completed_local_index"], 4)
            self.assertEqual(payload["records_written"], 5)
            with self.assertRaisesRegex(ValueError, "config hash"):
                write_progress(
                    path,
                    config_hash="config-b",
                    dataset_manifest_hash="dataset-a",
                    protocol_hash="protocol-a",
                    shard_index=1,
                    num_shards=3,
                    last_completed_local_index=5,
                    records_written=6,
                )

    def test_completed_shard_is_atomic_and_merge_refuses_missing_or_mismatched_shards(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(2):
                shard_dir = root / f"shard-{index:04d}"
                shard_dir.mkdir()
                records = [{"id": f"row-{index}", "attempt": 0}]
                records_path = shard_dir / "records.jsonl.zst"
                for record in records:
                    append_jsonl_zst(records_path, record)
                write_progress(
                    shard_dir / "progress.json",
                    config_hash="config-a",
                    dataset_manifest_hash="dataset-a",
                    protocol_hash="protocol-a",
                    shard_index=index,
                    num_shards=2,
                    last_completed_local_index=0,
                    records_written=1,
                )
                complete_shard(
                    shard_dir,
                    config_hash="config-a",
                    dataset_manifest_hash="dataset-a",
                    protocol_hash="protocol-a",
                    shard_index=index,
                    num_shards=2,
                    expected_records=1,
                )
                self.assertEqual(read_jsonl_zst(records_path), records)

            merged = merge_completed_shards(
                root,
                expected_shards=2,
                config_hash="config-a",
                dataset_manifest_hash="dataset-a",
                protocol_hash="protocol-a",
            )
            self.assertEqual([row["id"] for row in merged], ["row-0", "row-1"])
            (root / "shard-0001" / "complete.json").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "completion marker"):
                merge_completed_shards(
                    root,
                    expected_shards=2,
                    config_hash="config-a",
                    dataset_manifest_hash="dataset-a",
                    protocol_hash="protocol-a",
                )

    def test_resume_rejects_a_protocol_or_source_change(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.json"
            write_progress(
                path,
                config_hash="config-a",
                dataset_manifest_hash="dataset-a",
                protocol_hash="protocol-commit-a",
                shard_index=0,
                num_shards=1,
                last_completed_local_index=0,
                records_written=1,
            )

            with self.assertRaisesRegex(ValueError, "protocol hash"):
                write_progress(
                    path,
                    config_hash="config-a",
                    dataset_manifest_hash="dataset-a",
                    protocol_hash="protocol-commit-b",
                    shard_index=0,
                    num_shards=1,
                    last_completed_local_index=1,
                    records_written=2,
                )


if __name__ == "__main__":
    unittest.main()
