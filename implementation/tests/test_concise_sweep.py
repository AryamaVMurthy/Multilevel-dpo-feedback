import unittest

from text_feedback_dpo.concise_sweep import (
    PROFILES,
    build_sweep_prompt,
    build_decoding_freeze,
    promote_profiles,
    protocol_valid_correct,
    stratified_subset,
    summarize_records,
    validate_sweep_records,
    validate_screening_context,
)
from text_feedback_dpo.prompts import build_native_student_prompt


class ConciseSweepTest(unittest.TestCase):
    def test_decoding_freeze_requires_selected_profile_to_match_frozen_generation(self):
        screening_manifest = {
            "schema": "math-decoding-sweep-v1",
            "stage": "screening",
            "count": 12,
            "max_new_tokens": 4096,
            "prompt_protocol": "qwen3-nonthinking-final-r1",
            "model": {"id": "student", "revision": "r"},
            "dataset_manifest_sha256": "d",
            "dataset_audit_sha256": "a",
            "model_cache_manifest_sha256": "m",
            "config_sha256": "old",
        }
        screening_selection = {
            "schema": "math-decoding-sweep-selection-v1",
            "status": "passed",
            "stage": "screening",
            "promoted": ["presence-1", "presence-0", "presence-1.5"],
            "example_ids": [f"s{index}" for index in range(12)],
        }
        confirmation_manifest = {
            **screening_manifest,
            "stage": "confirmation",
            "count": 32,
            "max_new_tokens": 8192,
            "profiles": {name: {} for name in screening_selection["promoted"]},
            "example_ids": [f"c{index}" for index in range(32)],
        }
        confirmation_selection = {
            "schema": "math-decoding-sweep-selection-v1",
            "status": "passed",
            "stage": "confirmation",
            "promoted": ["presence-1"],
            "selected_profile": "presence-1",
            "example_ids": confirmation_manifest["example_ids"],
        }
        student_generation = {**PROFILES["presence-1"], "max_new_tokens": 8192}

        freeze = build_decoding_freeze(
            screening_manifest=screening_manifest,
            screening_selection=screening_selection,
            confirmation_manifest=confirmation_manifest,
            confirmation_selection=confirmation_selection,
            student_generation=student_generation,
            frozen_config_sha256="f" * 64,
            source_commit="e" * 40,
            screening_selection_sha256="1" * 64,
            confirmation_selection_sha256="2" * 64,
        )

        self.assertEqual(freeze["schema"], "math-decoding-freeze-v1")
        self.assertEqual(freeze["selected_profile"], "presence-1")
        with self.assertRaisesRegex(ValueError, "frozen student generation"):
            build_decoding_freeze(
                screening_manifest=screening_manifest,
                screening_selection=screening_selection,
                confirmation_manifest=confirmation_manifest,
                confirmation_selection=confirmation_selection,
                student_generation={**student_generation, "presence_penalty": 0.0},
                frozen_config_sha256="f" * 64,
                source_commit="e" * 40,
                screening_selection_sha256="1" * 64,
                confirmation_selection_sha256="2" * 64,
            )

    def test_sweep_prompt_is_byte_identical_to_frozen_baseline_prompt(self):
        example = {
            "id": "m1",
            "domain": "math",
            "problem": "Compute 2 + 2.",
        }
        self.assertEqual(
            build_sweep_prompt(example),
            build_native_student_prompt(problem=example["problem"], domain="math"),
        )

    def test_profiles_freeze_official_controls_and_efficiency_candidates(self):
        self.assertEqual(tuple(PROFILES), ("presence-0", "presence-0.5", "presence-1", "presence-1.5", "presence-2"))
        self.assertEqual({profile["presence_penalty"] for profile in PROFILES.values()}, {0.0, 0.5, 1.0, 1.5, 2.0})
        for profile in PROFILES.values():
            self.assertEqual(
                (profile["temperature"], profile["top_p"], profile["top_k"], profile["min_p"], profile["repetition_penalty"]),
                (0.7, 0.8, 20, 0.0, 1.0),
            )
        self.assertTrue(all(profile["stop_after_final_answer"] for profile in PROFILES.values()))

    def test_stratified_subset_is_deterministic_and_uses_only_level45_train_rows(self):
        rows = [
            {"id": f"{subject}-{level}-{index}", "source_subject": subject, "difficulty_level": level}
            for subject in ("algebra", "geometry")
            for level in (3, 4, 5)
            for index in range(5)
        ]
        first = stratified_subset(rows, count=8, seed=7)
        second = stratified_subset(list(reversed(rows)), count=8, seed=7)
        self.assertEqual([row["id"] for row in first], [row["id"] for row in second])
        self.assertEqual(len(first), 8)
        self.assertTrue(all(row["difficulty_level"] in {4, 5} for row in first))

    def test_promotion_uses_frozen_accuracy_truncation_efficiency_length_latency_order(self):
        summaries = [
            {"profile": "a", "correct": 10, "truncated": 1, "correct_per_million_tokens": 900, "median_tokens": 100, "mean_latency": 1.0},
            {"profile": "b", "correct": 10, "truncated": 0, "correct_per_million_tokens": 700, "median_tokens": 300, "mean_latency": 3.0},
            {"profile": "c", "correct": 10, "truncated": 0, "correct_per_million_tokens": 800, "median_tokens": 500, "mean_latency": 2.0},
            {"profile": "d", "correct": 9, "truncated": 0, "correct_per_million_tokens": 1000, "median_tokens": 50, "mean_latency": 0.5},
        ]
        self.assertEqual(promote_profiles(summaries, count=3), ["c", "b", "a"])

    def test_protocol_valid_correct_rejects_truncation_and_missing_termination(self):
        self.assertTrue(protocol_valid_correct(symbolic_correct=True, terminated=True, truncated=False))
        self.assertFalse(protocol_valid_correct(symbolic_correct=True, terminated=False, truncated=True))
        self.assertFalse(protocol_valid_correct(symbolic_correct=True, terminated=None, truncated=None))

    def test_sweep_completeness_requires_every_profile_example_pair_once(self):
        records = [
            {
                "profile": profile,
                "id": row_id,
                "correct": profile == "a",
                "generated_tokens": 10,
                "latency_seconds": 1.0,
                "finish_reason": "final_answer",
                "evaluable": True,
            }
            for profile in ("a", "b")
            for row_id in ("x", "y")
        ]
        validate_sweep_records(records, profiles=("a", "b"), example_ids=("x", "y"))
        summary = summarize_records(records)
        self.assertEqual(summary[0]["truncation_rate"], 0.0)
        self.assertEqual(summary[0]["unevaluable"], 0)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_sweep_records(records[:-1], profiles=("a", "b"), example_ids=("x", "y"))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_sweep_records(records + [records[0]], profiles=("a", "b"), example_ids=("x", "y"))

    def test_confirmation_requires_exact_screening_context_binding(self):
        manifest = {
            "schema": "math-decoding-sweep-v1",
            "stage": "screening",
            "config_sha256": "a",
            "dataset_manifest_sha256": "b",
            "dataset_audit_sha256": "c",
            "model_cache_manifest_sha256": "d",
            "model": {"id": "student", "revision": "rev"},
        }
        validate_screening_context(
            manifest,
            config_sha256="a",
            dataset_manifest_sha256="b",
            dataset_audit_sha256="c",
            model_cache_manifest_sha256="d",
            model={"id": "student", "revision": "rev"},
        )
        with self.assertRaisesRegex(ValueError, "config_sha256"):
            validate_screening_context(
                manifest,
                config_sha256="changed",
                dataset_manifest_sha256="b",
                dataset_audit_sha256="c",
                model_cache_manifest_sha256="d",
                model={"id": "student", "revision": "rev"},
            )


if __name__ == "__main__":
    unittest.main()
