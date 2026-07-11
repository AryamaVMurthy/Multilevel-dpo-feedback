import unittest

from text_feedback_dpo.concise_sweep import PROFILES, promote_profiles, stratified_subset


class ConciseSweepTest(unittest.TestCase):
    def test_profiles_freeze_official_controls_and_efficiency_candidates(self):
        self.assertEqual(len(PROFILES), 6)
        self.assertEqual(PROFILES["qwen-reasoning"]["top_k"], 40)
        self.assertEqual(PROFILES["qwen-general"]["temperature"], 0.7)
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

    def test_promotion_keeps_accuracy_within_one_correct_then_minimizes_tokens(self):
        summaries = [
            {"profile": "a", "correct": 10, "median_tokens": 3000, "truncated": 0, "mean_latency": 3.0},
            {"profile": "b", "correct": 9, "median_tokens": 1000, "truncated": 0, "mean_latency": 2.0},
            {"profile": "c", "correct": 8, "median_tokens": 500, "truncated": 0, "mean_latency": 1.0},
            {"profile": "d", "correct": 10, "median_tokens": 2000, "truncated": 1, "mean_latency": 2.0},
        ]
        self.assertEqual(promote_profiles(summaries, count=3), ["b", "d", "a"])


if __name__ == "__main__":
    unittest.main()
