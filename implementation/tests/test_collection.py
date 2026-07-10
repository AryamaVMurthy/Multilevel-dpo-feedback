import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from text_feedback_dpo.collection import collect_paper_shard, merge_paper_collection
from text_feedback_dpo.experiment_config import load_paper_experiment
from text_feedback_dpo.io import read_jsonl_zst, write_json_atomic
from text_feedback_dpo.models import ModelProvider


class ScriptedProvider(ModelProvider):
    def __init__(self):
        self.outputs = {
            "student": iter(["wrong one", "right one", "right two"]),
            "teacher": iter([
                "Recheck how the quantities relate before answering fully.",
            ]),
        }

    def generate(self, role, _prompt, **_kwargs):
        return next(self.outputs[role])


class CollectionTest(unittest.TestCase):
    def test_collection_writes_complete_compressed_records_and_merges(self):
        repo = Path(__file__).resolve().parents[1]
        config_path = repo / "configs" / "paper" / "gsm8k.yaml"
        config = load_paper_experiment(config_path)
        examples = [
            {"id": "m1", "domain": "math", "problem": "Compute one.", "gold_answer": "4"},
            {"id": "m2", "domain": "math", "problem": "Compute two.", "gold_answer": "7"},
        ]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "data"
            dataset_dir.mkdir()
            write_json_atomic(
                dataset_dir / "manifest.json",
                {"content_sha256": hashlib.sha256(b"fixture").hexdigest()},
            )
            output_dir = root / "collection"
            provider = ScriptedProvider()

            result = collect_paper_shard(
                config=config,
                config_path=config_path,
                examples=examples,
                dataset_dir=dataset_dir,
                output_root=output_dir,
                split="train",
                shard_index=0,
                num_shards=1,
                model_provider=provider,
                evaluator=lambda _example, response: {
                    "correct": response.startswith("right"),
                    "confidence": 0.9,
                    "reason": "fixture",
                },
                guidance_guard=lambda *_args: {"safe": True, "confidence": 0.9, "reason": "fixture"},
            )

            self.assertEqual(result["records"], 2)
            records = read_jsonl_zst(output_dir / "shard-0000" / "records.jsonl.zst")
            self.assertEqual([record["id"] for record in records], ["m1", "m2"])
            self.assertTrue((output_dir / "shard-0000" / "complete.json").exists())
            self.assertEqual(len(records[0]["attempts"]), 2)
            self.assertNotIn("prompt", records[0]["attempts"][0])

            merged = merge_paper_collection(
                config_path=config_path,
                dataset_dir=dataset_dir,
                collection_dir=output_dir,
                expected_shards=1,
                output_path=root / "merged.jsonl.zst",
            )
            self.assertEqual(merged["records"], 2)
            self.assertEqual(len(read_jsonl_zst(root / "merged.jsonl.zst")), 2)


if __name__ == "__main__":
    unittest.main()
