import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from text_feedback_dpo.offline import load_or_build_rollouts, load_or_build_trajectories


class OfflineReuseTest(unittest.TestCase):
    def test_reuses_matching_cached_rollouts_without_generation(self):
        with TemporaryDirectory() as tmp:
            cache = Path(tmp) / "rollouts.jsonl"
            cache.write_text(json.dumps({"example_id": "1", "policy_hash": "p1", "response": "x"}) + "\n", encoding="utf-8")
            calls = []

            rows = load_or_build_rollouts(
                examples=[{"id": "1"}],
                cache_path=cache,
                policy_hash="p1",
                generate=lambda _example: calls.append(True),
            )
            self.assertEqual(rows[0]["response"], "x")
            self.assertEqual(calls, [])

    def test_cache_mismatch_fails_instead_of_using_stale_response(self):
        with TemporaryDirectory() as tmp:
            cache = Path(tmp) / "rollouts.jsonl"
            cache.write_text(json.dumps({"example_id": "1", "policy_hash": "old", "response": "x"}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "policy_hash"):
                load_or_build_rollouts(
                    examples=[{"id": "1"}],
                    cache_path=cache,
                    policy_hash="new",
                    generate=lambda _example: {"response": "y"},
                )

    def test_reuses_complete_trajectory_cache_by_policy_hash(self):
        with TemporaryDirectory() as tmp:
            cache = Path(tmp) / "trajectories.jsonl"
            cache.write_text(json.dumps({"id": "1", "policy_hash": "p1", "resolved": True}) + "\n", encoding="utf-8")
            calls = []
            rows = load_or_build_trajectories(
                examples=[{"id": "1"}],
                cache_path=cache,
                policy_hash="p1",
                generate=lambda missing: calls.append(missing),
            )
            self.assertTrue(rows[0]["resolved"])
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
