import unittest

from text_feedback_dpo.concise_sweep import (
    PROFILES,
    promote_profiles,
    protocol_valid_correct,
    stratified_subset,
    summarize_records,
    validate_sweep_records,
)


class ConciseSweepTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
