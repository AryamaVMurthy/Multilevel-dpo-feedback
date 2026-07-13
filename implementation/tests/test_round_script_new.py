import unittest
from pathlib import Path


class RoundScriptTest(unittest.TestCase):
    def test_primary_round_script_has_ordered_fail_fast_stages(self):
        text = Path("scripts/turing_primary_round.sh").read_text(encoding="utf-8")
        self.assertIn("set -euo pipefail", text)
        for stage in ("collect", "build-preferences", "train-dpo", "generate", "evaluate", "report"):
            self.assertIn(stage, text)
        self.assertNotIn("|| true", text)

    def test_primary_round_separates_two_gpu_collection_from_four_gpu_training(self):
        text = Path("scripts/turing_primary_round.sh").read_text(encoding="utf-8")
        self.assertIn("COLLECTION_SCRIPT", text)
        self.assertIn("sbatch", text)
        self.assertIn("--gres=gpu:2", text)
        self.assertIn("TRAIN_GPUS=4", text)
        self.assertIn("--nodes=1", text)
        self.assertIn("SHARD_INDEX", text)
        self.assertIn("MERGE_ID", text)
        self.assertIn("checkpoint", text.lower())
        self.assertIn("resume", text.lower())

    def test_round_records_hashes_and_artifact_paths(self):
        text = Path("scripts/turing_primary_round.sh").read_text(encoding="utf-8")
        for value in ("CONFIG_HASH", "MODEL_HASH", "DATASET_HASH", "PROMPT_HASH", "RETRIEVAL_HASH", "SOURCE_SCHEMA_HASH", "RUN_MANIFEST"):
            self.assertIn(value, text)


if __name__ == "__main__":
    unittest.main()
