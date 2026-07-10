import unittest

from text_feedback_dpo.hyperparameter_search import (
    build_dpo_candidates,
    build_grpo_candidates,
    create_search_ledger,
    freeze_selection,
    promote_stage,
    register_observation,
)


class HyperparameterSearchTest(unittest.TestCase):
    def test_candidate_matrices_are_exact(self):
        dpo = build_dpo_candidates(
            learning_rates=(2e-6, 5e-6, 1e-5),
            betas=(0.05, 0.1, 0.3, 0.5),
            weight_decay=0.01,
            warmup_fraction=0.05,
            scheduler="cosine",
        )
        grpo = build_grpo_candidates(
            learning_rates=(2e-6, 5e-6, 1e-5),
            kl_betas=(0.0, 0.001, 0.01, 0.04),
        )
        self.assertEqual(len(dpo), 12)
        self.assertEqual(len(grpo), 12)
        self.assertEqual(len({candidate.candidate_id for candidate in dpo}), 12)
        self.assertEqual(len({candidate.candidate_id for candidate in grpo}), 12)
        self.assertEqual(dpo[0].scheduler, "cosine")
        self.assertEqual(dpo[0].loss_type, "sigmoid_norm")
        self.assertIsNone(dpo[0].ld_alpha)
        self.assertEqual(grpo[-1].kl_beta, 0.04)

    def test_length_desensitized_candidates_are_labeled_and_reject_invalid_alpha(self):
        candidates = build_dpo_candidates(
            learning_rates=(5e-6,),
            betas=(0.1,),
            weight_decay=0.01,
            warmup_fraction=0.05,
            scheduler="cosine",
            loss_type="sigmoid",
            ld_alpha=0.5,
        )
        self.assertEqual(candidates[0].loss_type, "sigmoid")
        self.assertEqual(candidates[0].ld_alpha, 0.5)
        with self.assertRaisesRegex(ValueError, "ld_alpha"):
            build_dpo_candidates(
                learning_rates=(5e-6,), betas=(0.1,), weight_decay=0.01,
                warmup_fraction=0.05, scheduler="cosine", loss_type="sigmoid", ld_alpha=1.0,
            )

    def test_successive_halving_rejects_invalid_runs_and_freezes_without_overwrite(self):
        candidates = build_dpo_candidates(
            learning_rates=(2e-6, 5e-6),
            betas=(0.1, 0.3),
            weight_decay=0.01,
            warmup_fraction=0.05,
            scheduler="cosine",
        )
        ledger = create_search_ledger(
            method="standard_dpo",
            candidates=candidates,
            promote_counts=(2, 1),
            dataset_manifest_hash="dataset",
            seed=11,
        )
        for index, candidate in enumerate(candidates):
            register_observation(
                ledger,
                candidate_id=candidate.candidate_id,
                stage=0,
                status="valid" if index != 3 else "invalid",
                metrics={"selection_metric": 0.9 - index * 0.1, "gpu_hours": 1.0},
                artifact_hash=f"artifact-{index}",
                failure_reason="out of memory" if index == 3 else None,
            )
        promoted = promote_stage(ledger, stage=0)
        self.assertEqual(len(promoted), 2)
        selected = promoted[0]
        for stage_index, candidate_id in enumerate(promoted):
            register_observation(
                ledger,
                candidate_id=candidate_id,
                stage=1,
                status="valid",
                metrics={"selection_metric": 0.95 - stage_index * 0.1, "gpu_hours": 2.0},
                artifact_hash=f"final-artifact-{stage_index}",
            )
        promote_stage(ledger, stage=1)
        frozen = freeze_selection(ledger, candidate_id=selected, stage=1)
        self.assertEqual(frozen["candidate_id"], selected)
        with self.assertRaisesRegex(RuntimeError, "frozen"):
            register_observation(
                ledger,
                candidate_id=selected,
                stage=1,
                status="valid",
                metrics={"selection_metric": 0.99, "gpu_hours": 2.0},
                artifact_hash="other",
            )


if __name__ == "__main__":
    unittest.main()
