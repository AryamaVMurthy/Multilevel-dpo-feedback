import unittest
from pathlib import Path


class RoundScriptTest(unittest.TestCase):
    def test_primary_round_script_has_ordered_fail_fast_stages(self):
        text = Path("scripts/turing_primary_round.sh").read_text(encoding="utf-8")
        self.assertIn("set -euo pipefail", text)
        for stage in ("collect", "build-preferences", "train-dpo", "generate", "evaluate", "report"):
            self.assertIn(stage, text)
        self.assertNotIn("|| true", text)


if __name__ == "__main__":
    unittest.main()
