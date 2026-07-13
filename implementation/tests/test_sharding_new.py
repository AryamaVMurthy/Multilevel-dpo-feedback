import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from text_feedback_dpo.sharding import merge_prediction_shards, shard_jsonl


class ShardingTest(unittest.TestCase):
    def test_sharding_is_deterministic_and_preserves_every_id_once(self):
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "data.jsonl"
            source.write_text("".join(json.dumps({"id": str(i), "value": i}) + "\n" for i in range(10)), encoding="utf-8")
            output = Path(tmp) / "shards"
            manifest = shard_jsonl(source, output, shard_count=4)
            ids = []
            for path in sorted(output.glob("shard-*.jsonl")):
                ids.extend(json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines())
            self.assertEqual(sorted(ids), [str(i) for i in range(10)])
            self.assertEqual(manifest["rows"], 10)
            self.assertEqual(sum(manifest["shard_rows"]), 10)

    def test_merge_requires_exact_ordered_id_parity(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "shard-0.jsonl").write_text('{"id":"a"}\n{"id":"b"}\n', encoding="utf-8")
            (root / "predictions-0.jsonl").write_text('{"id":"a","response":"A"}\n{"id":"wrong","response":"B"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ID mismatch"):
                merge_prediction_shards(root, root / "merged.jsonl", shard_count=1)

    def test_merge_publishes_all_predictions_after_validation(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, example_id in enumerate(("a", "b")):
                (root / f"shard-{index}.jsonl").write_text(json.dumps({"id": example_id}) + "\n", encoding="utf-8")
                (root / f"predictions-{index}.jsonl").write_text(json.dumps({"id": example_id, "response": example_id.upper()}) + "\n", encoding="utf-8")
            result = merge_prediction_shards(root, root / "merged.jsonl", shard_count=2)
            self.assertEqual(result["rows"], 2)
            self.assertEqual([json.loads(line)["id"] for line in (root / "merged.jsonl").read_text(encoding="utf-8").splitlines()], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
