import json
import hashlib
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

from text_feedback_dpo.offline import (
    build_cache_manifest,
    load_or_build_rollouts,
    load_or_build_trajectories,
    student_policy_identity,
)
from text_feedback_dpo.batch_generation import SCAFFOLD_PROMPT_VERSION, run_fixed_retrieval_pipeline
from text_feedback_dpo.runtime import GeneratedText
from text_feedback_dpo.feedback import diagnose_attempt
from text_feedback_dpo.prompts import prompt_builder_identity


def _hash(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def manifest(**overrides):
    values = {
        "student_model": "Qwen/Qwen3-4B-Base",
        "student_revision": "student-rev",
        "teacher_model": "Qwen/Qwen3-32B",
        "teacher_revision": "teacher-rev",
        "teacher_identity": "primary_qwen3_32b_4bit",
        "teacher_quantization": "4bit",
        "teacher_fallback_reason": None,
        "dataset_revision": "data-rev",
        "dataset_hash": "a" * 64,
        "prompt_version": "fixed-retrieval-cited-v1",
        "prompt_hash": _hash({
            "identity": "fixed-retrieval-cited-v1", "builders": prompt_builder_identity(),
        }),
        "student_thinking_mode": "direct",
        "teacher_thinking": True,
        "decoding": {"answer_max_new_tokens": 32, "temperature": 0.7, "top_p": 0.9},
        "intervention_policy": {"max_interventions": 4, "max_hint_words": 24},
        "seed": 7,
        "policy_hash": student_policy_identity(
            student_model="Qwen/Qwen3-4B-Base",
            student_revision="student-rev",
            policy_version="policy-v1",
        )["sha256"],
        "dataset_schema": "searchqa.search_results.v1",
        "source_schema_version": 1,
        "source_schema_hash": _hash({"identity": "searchqa.search_results.v1", "version": 1}),
        "retrieval_config": {"identity": "fixed_bm25", "schema_version": 1, "requested_top_k": 8, "k1": 1.2, "b": 0.75},
        "retrieval_hash": _hash({"identity": "fixed_bm25", "schema_version": 1, "requested_top_k": 8, "k1": 1.2, "b": 0.75}),
        "response_schema_version": 1,
        "response_schema_hash": _hash({"identity": "cited-response", "schema_version": 1}),
        "evaluator_version": "cited-response-evaluator-v2-standalone-article-answer",
        "evaluator_hash": _hash({"identity": "cited-response-evaluator-v2-standalone-article-answer"}),
        "policy_version": "policy-v1",
        "sibling_seeds": [101, 102],
        "sibling_count": 2,
    }
    values.update(overrides)
    if "policy_hash" not in overrides:
        values["policy_hash"] = student_policy_identity(
            student_model=values["student_model"],
            student_revision=values["student_revision"],
            policy_version=values["policy_version"],
        )["sha256"]
    return build_cache_manifest(**values)


def active_example():
    return {
        "id": "1", "question": "Who?", "gold_answer": "Ada",
        "sources": [{"source_id": "S001", "original_rank": 1, "title": "Ada", "url": "https://example.test/ada", "snippet": "Ada wrote it."}],
    }


def active_trajectory():
    example = active_example()
    artifact = run_fixed_retrieval_pipeline(
        [example], query_generate_batch=lambda _prompts: [GeneratedText("Ada", False)],
        response_generate_batch=lambda _prompts: [GeneratedText("Answer: Ada\nReasoning: Source says Ada [S001].\nSources: S001", False)],
        policy_hash="b" * 64,
    )[0]
    diagnostics = diagnose_attempt(artifact)
    attempt_supervision = {
        "response": artifact["raw_response"], "correct": True,
        "score": artifact["cited_score"], "diagnostics": diagnostics,
        "responsible_region": None, "no_hint": True, "provenance": "student",
    }
    return {
        "id": "1", "prompt": artifact["query_prompt"], "query_prompt": artifact["query_prompt"],
        "query_prompt_hash": artifact["query_prompt_hash"], "example_identity": _hash(example),
        "attempts": [{"attempt_index": 0, "artifact": artifact, **attempt_supervision,
                      "supervision_hash": _hash(attempt_supervision)}],
        "interventions": [], "ranked_interventions": [], "chosen": artifact, "resolved": True,
        "no_hint_siblings": [], "sibling_verification": {"status": "not_required", "eligible": True},
        "training_eligible": True, "sft_eligible": True, "preference_eligible": False,
        "preference_rows": [], "policy_hash": artifact["policy_hash"],
        "response_prompt_hash": artifact["response_prompt_hash"], "evaluator_version": artifact["evaluator_version"],
    }
class OfflineReuseTest(unittest.TestCase):
    def test_cache_manifest_accepts_the_selected_scaffold_prompt_identity(self):
        scaffold_hash = _hash({
            "identity": SCAFFOLD_PROMPT_VERSION, "builders": prompt_builder_identity(),
        })
        result = manifest(prompt_version=SCAFFOLD_PROMPT_VERSION, prompt_hash=scaffold_hash)
        self.assertEqual(result["prompt_version"], SCAFFOLD_PROMPT_VERSION)
        self.assertEqual(result["prompt_hash"], scaffold_hash)

    def test_student_policy_identity_is_canonical_and_weight_bound(self):
        first = student_policy_identity(
            student_model="Qwen/Qwen3-4B-Base",
            student_revision="student-rev",
            policy_version="raw-base-v1",
        )
        repeated = student_policy_identity(
            student_model="Qwen/Qwen3-4B-Base",
            student_revision="student-rev",
            policy_version="raw-base-v1",
        )
        changed = student_policy_identity(
            student_model="Qwen/Qwen3-4B-Base",
            student_revision="sft-checkpoint-sha256",
            policy_version="sft-v1",
        )

        self.assertEqual(first, repeated)
        self.assertRegex(first["sha256"], r"^[0-9a-f]{64}$")
        self.assertNotEqual(first["sha256"], changed["sha256"])
        self.assertEqual(first["identity"], "student-policy-v1")

    def test_cache_manifest_rejects_policy_hash_not_bound_to_student_weights(self):
        with self.assertRaisesRegex(ValueError, "policy_hash identity mismatch"):
            manifest(policy_hash="b" * 64)

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
            old = manifest(policy_version="old")
            cache.write_text(json.dumps({"example_id": "1", "cache_hash": old["cache_hash"], "response": "x"}) + "\n", encoding="utf-8")
            cache.with_suffix(".manifest.json").write_text(json.dumps(old), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "policy_hash"):
                load_or_build_rollouts(
                    examples=[{"id": "1"}],
                    cache_path=cache,
                    cache_manifest=manifest(policy_hash="c" * 64),
                    generate=lambda _example: {"response": "y"},
                )

    def test_reuses_complete_trajectory_cache_by_policy_hash(self):
        with TemporaryDirectory() as tmp:
            cache = Path(tmp) / "trajectories.jsonl"
            identity = manifest()
            load_or_build_trajectories(
                examples=[active_example()], cache_path=cache, cache_manifest=identity,
                generate=lambda _missing: [active_trajectory()],
            )
            calls = []
            rows = load_or_build_trajectories(
                examples=[active_example()],
                cache_path=cache,
                cache_manifest=identity,
                generate=lambda missing: calls.append(missing),
            )
            self.assertTrue(rows[0]["resolved"])
            self.assertEqual(calls, [])

    def test_cached_trajectory_reuse_rejects_tampered_artifacts(self):
        with TemporaryDirectory() as tmp:
            cache = Path(tmp) / "trajectories.jsonl"
            identity = manifest()
            load_or_build_trajectories(
                examples=[active_example()], cache_path=cache, cache_manifest=identity,
                generate=lambda _missing: [active_trajectory()],
            )
            cached = json.loads(cache.read_text(encoding="utf-8"))
            cached["attempts"][0]["artifact"]["raw_response"] = "Answer: forged\nReasoning: Forged [S001].\nSources: S001"
            cached["chosen"] = cached["attempts"][0]["artifact"]
            cache.write_text(json.dumps(cached) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cited_score|parsed_response"):
                load_or_build_trajectories(
                    examples=[active_example()], cache_path=cache, cache_manifest=identity,
                    generate=lambda _missing: self.fail("tampered cache must not regenerate silently"),
                )

    def test_cached_trajectory_reuse_rejects_tampered_trajectory_metadata(self):
        for mutate, message in (
            (lambda row: row.__setitem__("query_prompt_hash", "f" * 64), "query_prompt_hash"),
            (lambda row: row.__setitem__("sibling_verification", {"status": "verified"}), "sibling_verification"),
            (lambda row: row.__setitem__("ranked_interventions", [{"efficiency_score": 1.0}]), "ranked_interventions"),
        ):
            with self.subTest(message=message), TemporaryDirectory() as tmp:
                cache = Path(tmp) / "trajectories.jsonl"
                identity = manifest()
                load_or_build_trajectories(
                    examples=[active_example()], cache_path=cache, cache_manifest=identity,
                    generate=lambda _missing: [active_trajectory()],
                )
                cached = json.loads(cache.read_text(encoding="utf-8"))
                mutate(cached)
                cache.write_text(json.dumps(cached) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_or_build_trajectories(
                        examples=[active_example()], cache_path=cache, cache_manifest=identity,
                        generate=lambda _missing: self.fail("tampered cache must not regenerate silently"),
                    )

    def test_new_trajectory_is_fully_revalidated_before_cache_write(self):
        with TemporaryDirectory() as tmp:
            cache = Path(tmp) / "trajectories.jsonl"
            generated = active_trajectory()
            generated["query_prompt_hash"] = "f" * 64

            with self.assertRaisesRegex(ValueError, "query_prompt_hash"):
                load_or_build_trajectories(
                    examples=[active_example()], cache_path=cache, cache_manifest=manifest(),
                    generate=lambda _missing: [generated],
                )
            self.assertFalse(cache.exists())

    def test_new_trajectory_cannot_override_cache_wrapper_identity(self):
        for field in ("cache_hash", "row_identity"):
            with self.subTest(field=field), TemporaryDirectory() as tmp:
                cache = Path(tmp) / "trajectories.jsonl"
                generated = active_trajectory()
                generated[field] = "forged"
                with self.assertRaisesRegex(ValueError, field):
                    load_or_build_trajectories(
                        examples=[active_example()], cache_path=cache, cache_manifest=manifest(),
                        generate=lambda _missing: [generated],
                    )
                self.assertFalse(cache.exists())

    def test_trajectory_cache_validates_sources_before_generator(self):
        with TemporaryDirectory() as tmp:
            cache = Path(tmp) / "trajectories.jsonl"
            malformed = active_example()
            malformed["sources"][0]["original_rank"] = True
            calls = []
            with self.assertRaisesRegex(ValueError, "original_rank"):
                load_or_build_trajectories(
                    examples=[malformed], cache_path=cache, cache_manifest=manifest(),
                    generate=lambda missing: calls.append(missing),
                )
            self.assertEqual(calls, [])
            self.assertFalse(cache.exists())

    def test_cache_loaders_reject_blank_jsonl_records(self):
        with TemporaryDirectory() as tmp:
            cache = Path(tmp) / "trajectories.jsonl"
            identity = manifest()
            load_or_build_trajectories(
                examples=[active_example()], cache_path=cache, cache_manifest=identity,
                generate=lambda _missing: [active_trajectory()],
            )
            cache.write_text(cache.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "blank"):
                load_or_build_trajectories(
                    examples=[active_example()], cache_path=cache, cache_manifest=identity,
                    generate=lambda _missing: self.fail("blank cache must not regenerate"),
                )

        with TemporaryDirectory() as tmp:
            cache = Path(tmp) / "rollouts.jsonl"
            identity = manifest()
            load_or_build_rollouts(
                examples=[{"id": "1"}], cache_path=cache, cache_manifest=identity,
                generate=lambda _example: {"response": "x"},
            )
            cache.write_text(cache.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "blank"):
                load_or_build_rollouts(
                    examples=[{"id": "1"}], cache_path=cache, cache_manifest=identity,
                    generate=lambda _example: self.fail("blank cache must not regenerate"),
                )

    def test_cache_manifest_rejects_invalid_teacher_seed_and_hash_semantics(self):
        for overrides, message in (
            ({"teacher_model": "Qwen/Qwen3-14B"}, "teacher"),
            ({"teacher_quantization": "bf16"}, "4bit"),
            ({"teacher_fallback_reason": "unexpected"}, "fallback"),
            ({"sibling_seeds": [101, 101]}, "unique"),
            ({"sibling_seeds": [101, -1]}, "nonnegative"),
            ({"dataset_hash": "not-a-hash"}, "SHA-256"),
            ({"student_revision": ""}, "student_revision"),
        ):
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, message):
                manifest(**overrides)

    def test_every_causal_cache_field_changes_the_hash(self):
        baseline = manifest()["cache_hash"]
        changes = [
            {"student_revision": "student-rev-2"},
            {"teacher_revision": "teacher-rev-2"},
            {"teacher_identity": "fallback_qwen3_14b_4bit", "teacher_model": "Qwen/Qwen3-14B", "teacher_fallback_reason": "32B unavailable on allocated GPU memory"},
            {"dataset_revision": "data-rev-2"},
            {"dataset_hash": "c" * 64},
            {"student_thinking_mode": "two_pass"},
            {"teacher_thinking": False},
            {"decoding": {"answer_max_new_tokens": 31}},
            {"intervention_policy": {"max_interventions": 3}},
            {"seed": 8},
            {"policy_version": "policy-v2"},
            {"sibling_seeds": [103, 104]},
        ]
        for change in changes:
            with self.subTest(change=change):
                self.assertNotEqual(manifest(**change)["cache_hash"], baseline)

    def test_prompt_builder_implementation_identity_invalidates_cache_without_version_bump(self):
        baseline = manifest()
        changed_builders = {**prompt_builder_identity(), "query": {"implementation_sha256": "f" * 64}}
        with patch(
            "text_feedback_dpo.prompts.prompt_builder_identity", return_value=changed_builders
        ):
            changed = manifest(prompt_hash=_hash({
                "identity": "fixed-retrieval-cited-v1", "builders": changed_builders,
            }))
        self.assertEqual(changed["prompt_version"], baseline["prompt_version"])
        self.assertNotEqual(changed["prompt_builders"], baseline["prompt_builders"])
        self.assertNotEqual(changed["cache_hash"], baseline["cache_hash"])


if __name__ == "__main__":
    unittest.main()
