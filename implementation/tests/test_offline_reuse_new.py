import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from text_feedback_dpo.offline import build_cache_manifest, load_or_build_rollouts, load_or_build_trajectories


def manifest(**overrides):
    values = {
        "student_model": "Qwen/Qwen3-4B-Base",
        "student_revision": "student-rev",
        "teacher_model": "Qwen/Qwen3-32B",
        "teacher_revision": "teacher-rev",
        "dataset_revision": "data-rev",
        "prompt_version": "plain-v2",
        "student_thinking_mode": "direct",
        "teacher_thinking": True,
        "decoding": {"answer_max_new_tokens": 32, "temperature": 0.7, "top_p": 0.9},
        "intervention_policy": {"max_interventions": 4, "max_hint_words": 24},
        "seed": 7,
        "policy_hash": "checkpoint-hash",
        "dataset_schema": "searchqa.search_results.v1",
        "source_schema_version": 1,
        "source_schema_hash": "source-schema-hash",
        "retrieval_config": {"backend": "fixed_bm25", "top_k": 8, "k1": 1.2, "b": 0.75},
        "retrieval_hash": "retrieval-hash",
        "response_schema_version": 1,
        "evaluator_version": "evaluator-v1",
        "policy_version": "policy-v1",
        "sibling_seeds": [101, 102],
    }
    values.update(overrides)
    return build_cache_manifest(**values)


class OfflineReuseTest(unittest.TestCase):
    def test_reuses_matching_cached_rollouts_without_generation(self):
        with TemporaryDirectory() as tmp:
            cache = Path(tmp) / "rollouts.jsonl"
            identity = manifest()
            cache.write_text(json.dumps({"example_id": "1", "row_identity": "row-identity", "cache_hash": identity["cache_hash"], "response": "x"}) + "\n", encoding="utf-8")
            cache.with_suffix(".manifest.json").write_text(json.dumps(identity), encoding="utf-8")
            calls = []

            rows = load_or_build_rollouts(
                examples=[{"id": "1", "row_identity": "row-identity"}],
                cache_path=cache,
                cache_manifest=identity,
                generate=lambda _example: calls.append(True),
            )
            self.assertEqual(rows[0]["response"], "x")
            self.assertEqual(calls, [])

    def test_cache_mismatch_fails_instead_of_using_stale_response(self):
        with TemporaryDirectory() as tmp:
            cache = Path(tmp) / "rollouts.jsonl"
            old = manifest(prompt_version="old")
            cache.write_text(json.dumps({"example_id": "1", "cache_hash": old["cache_hash"], "response": "x"}) + "\n", encoding="utf-8")
            cache.with_suffix(".manifest.json").write_text(json.dumps(old), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "policy_hash"):
                load_or_build_rollouts(
                    examples=[{"id": "1"}],
                    cache_path=cache,
                    cache_manifest=manifest(policy_hash="new"),
                    generate=lambda _example: {"response": "y"},
                )

    def test_reuses_complete_trajectory_cache_by_policy_hash(self):
        with TemporaryDirectory() as tmp:
            cache = Path(tmp) / "trajectories.jsonl"
            identity = manifest()
            cache.write_text(json.dumps({"id": "1", "row_identity": "row-identity", "cache_hash": identity["cache_hash"], "resolved": True}) + "\n", encoding="utf-8")
            cache.with_suffix(".manifest.json").write_text(json.dumps(identity), encoding="utf-8")
            calls = []
            rows = load_or_build_trajectories(
                examples=[{"id": "1", "row_identity": "row-identity"}],
                cache_path=cache,
                cache_manifest=identity,
                generate=lambda missing: calls.append(missing),
            )
            self.assertTrue(rows[0]["resolved"])
            self.assertEqual(calls, [])

    def test_every_causal_cache_field_changes_the_hash(self):
        baseline = manifest()["cache_hash"]
        changes = [
            {"student_revision": "student-rev-2"},
            {"teacher_revision": "teacher-rev-2"},
            {"dataset_revision": "data-rev-2"},
            {"prompt_version": "plain-v3"},
            {"student_thinking_mode": "two_pass"},
            {"teacher_thinking": False},
            {"decoding": {"answer_max_new_tokens": 31}},
            {"intervention_policy": {"max_interventions": 3}},
            {"seed": 8},
            {"policy_hash": "checkpoint-hash-2"},
            {"dataset_schema": "other.schema"},
            {"source_schema_version": 2},
            {"retrieval_hash": "retrieval-hash-2"},
            {"response_schema_version": 2},
            {"evaluator_version": "evaluator-v2"},
            {"policy_version": "policy-v2"},
            {"sibling_seeds": [103, 104]},
        ]
        for change in changes:
            with self.subTest(change=change):
                self.assertNotEqual(manifest(**change)["cache_hash"], baseline)


if __name__ == "__main__":
    unittest.main()
