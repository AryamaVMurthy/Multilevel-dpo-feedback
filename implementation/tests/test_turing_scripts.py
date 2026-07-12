import unittest
from pathlib import Path


class TuringScriptTest(unittest.TestCase):
    def test_paper_scripts_use_fail_fast_slurm_and_explicit_artifacts(self):
        for name in (
            "turing_setup_environment.sh",
            "turing_stage_model_cache.sh",
            "turing_verify_model_cache_lookup.sh",
            "turing_download_dataset_source.sh",
            "turing_download_math_source.sh",
            "turing_materialize_dataset.sh",
            "turing_audit_dataset.sh",
            "turing_materialize_preflight_subset.sh",
            "turing_locate_dataset_row.sh",
            "turing_decoding_sweep.sh",
            "turing_freeze_decoding.sh",
            "turing_freeze_baseline.sh",
            "turing_collect_array.sh",
            "turing_merge_collection.sh",
            "turing_tune_paper.sh",
            "turing_train_paper.sh",
            "turing_evaluate_paper.sh",
            "turing_merge_evaluations.sh",
            "turing_setup_grpo_environment.sh",
        ):
            text = Path("scripts" / Path(name)).read_text(encoding="utf-8")
            self.assertIn("set -euo pipefail", text, name)
            self.assertIn("TURING_ACCOUNT:?TURING_ACCOUNT is required", text, name)
            self.assertIn("PROJECT_DIR:?PROJECT_DIR is required", text, name)
            self.assertIn("module load u22/cuda/12.4", text, name)
            self.assertNotIn("|| true", text, name)
            self.assertNotIn("/home/$USER/.cache", text, name)
        setup = Path("scripts/turing_setup_environment.sh").read_text(encoding="utf-8")
        self.assertIn("SHARED_UV_CACHE:?SHARED_UV_CACHE is required", setup)
        self.assertIn('"$SHARED_UV_CACHE" != /scratch/*', setup)
        self.assertIn('"$SHARED_PROJECT_ENV" != /scratch/*', setup)
        self.assertIn("uv sync --frozen", setup)
        self.assertIn("environment_verified.txt.tmp", setup)
        self.assertIn('mv "$VERIFY_TMP" "$SHARED_PROJECT_ENV/environment_verified.txt"', setup)
        math_download = Path("scripts/turing_download_math_source.sh").read_text(encoding="utf-8")
        self.assertIn("CONFIG:?CONFIG is required", math_download)
        self.assertIn("EleutherAI/hendrycks_math", math_download)
        self.assertIn("dataset[\"subjects\"]", math_download)
        self.assertIn("dataset[\"revision\"]", math_download)
        self.assertIn("refusing non-empty MATH source output directory", math_download)
        self.assertIn("RUNTIME_ROOT:?RUNTIME_ROOT is required", math_download)
        self.assertIn("validate-paper-config", math_download)
        self.assertIn("uv run --frozen --no-sync python", math_download)
        stage_cache = Path("scripts/turing_stage_model_cache.sh").read_text(encoding="utf-8")
        self.assertIn("CONFIG:?CONFIG is required", stage_cache)
        self.assertIn("MODEL_CACHE_DIR:?MODEL_CACHE_DIR is required", stage_cache)
        self.assertIn("SOURCE_COMMIT:?SOURCE_COMMIT is required", stage_cache)
        self.assertIn('"$MODEL_CACHE_DIR" != /scratch/*', stage_cache)
        self.assertIn("snapshot_download", stage_cache)
        self.assertIn("tfdpo-model-cache-manifest.json", stage_cache)
        self.assertIn("validate-paper-config", stage_cache)
        verify_cache = Path("scripts/turing_verify_model_cache_lookup.sh").read_text(encoding="utf-8")
        self.assertIn('HF_HUB_CACHE="$MODEL_CACHE_DIR"', verify_cache)
        self.assertIn("local_files_only=True", verify_cache)
        self.assertIn("CACHE_LOOKUP_OUTPUT:?CACHE_LOOKUP_OUTPUT is required", verify_cache)
        self.assertIn("uv run --frozen --no-sync", verify_cache)
        preflight = Path("scripts/turing_materialize_preflight_subset.sh").read_text(encoding="utf-8")
        self.assertIn("materialize-preflight-subset", preflight)
        self.assertIn("SOURCE_PATH:?SOURCE_PATH is required", preflight)
        self.assertIn("DATASET_MANIFEST:?DATASET_MANIFEST is required", preflight)
        self.assertIn("OUTPUT_PATH:?OUTPUT_PATH is required", preflight)
        self.assertIn("SUBSET_COUNT:?SUBSET_COUNT is required", preflight)
        self.assertIn("SUBSET_SEED:?SUBSET_SEED is required", preflight)
        self.assertIn("SELECTION_POLICY:?SELECTION_POLICY is required", preflight)
        self.assertIn('--selection-policy "$SELECTION_POLICY"', preflight)
        self.assertIn('OUTPUT_MANIFEST="$(dirname "$OUTPUT_PATH")/manifest.json"', preflight)
        self.assertIn('cmp -s "$DATASET_MANIFEST" "$OUTPUT_MANIFEST"', preflight)
        locate = Path("scripts/turing_locate_dataset_row.sh").read_text(encoding="utf-8")
        self.assertIn("DATA_PATH:?DATA_PATH is required", locate)
        self.assertIn("ROW_ID:?ROW_ID is required", locate)
        self.assertIn("LOOKUP_OUTPUT:?LOOKUP_OUTPUT is required", locate)
        self.assertIn("paper-dataset-row-lookup-v1", locate)
        self.assertIn("uv run --frozen --no-sync", locate)
        audit = Path("scripts/turing_audit_dataset.sh").read_text(encoding="utf-8")
        self.assertIn("audit-dataset", audit)
        self.assertIn("DATASET_DIR:?DATASET_DIR is required", audit)
        self.assertIn("AUDIT_OUTPUT:?AUDIT_OUTPUT is required", audit)
        self.assertIn('"$DATASET_DIR" != /scratch/*', audit)
        self.assertIn("uv run --frozen --no-sync python", audit)
        decoding = Path("scripts/turing_decoding_sweep.sh").read_text(encoding="utf-8")
        self.assertIn("SWEEP_STAGE:?SWEEP_STAGE is required", decoding)
        self.assertIn("DATA_PATH:?DATA_PATH is required", decoding)
        self.assertIn("DATASET_AUDIT:?DATASET_AUDIT is required", decoding)
        self.assertIn("MODEL_CACHE_DIR:?MODEL_CACHE_DIR is required", decoding)
        self.assertIn("run_local_concise_sweep.py", decoding)
        self.assertIn("nvidia-smi", decoding)
        freeze_decoding = Path("scripts/turing_freeze_decoding.sh").read_text(encoding="utf-8")
        self.assertIn("freeze-decoding", freeze_decoding)
        self.assertIn("MODEL_CACHE_DIR:?MODEL_CACHE_DIR is required", freeze_decoding)
        self.assertIn("CACHE_PRESERVE_OUTPUT:?CACHE_PRESERVE_OUTPUT is required", freeze_decoding)
        self.assertIn("DECODING_FREEZE_OUTPUT:?DECODING_FREEZE_OUTPUT is required", freeze_decoding)
        self.assertIn("model-cache selection hash mismatch", freeze_decoding)
        self.assertIn("uv run --frozen --no-sync", freeze_decoding)
        freeze_baseline = Path("scripts/turing_freeze_baseline.sh").read_text(encoding="utf-8")
        self.assertIn("freeze-baseline", freeze_baseline)
        self.assertIn("DATASET_MANIFEST:?DATASET_MANIFEST is required", freeze_baseline)
        self.assertIn("SOURCE_COMMIT:?SOURCE_COMMIT is required", freeze_baseline)
        self.assertIn("FREEZE_OUTPUT:?FREEZE_OUTPUT is required", freeze_baseline)
        self.assertIn("uv run --frozen --no-sync", freeze_baseline)
        merge_collection = Path("scripts/turing_merge_collection.sh").read_text(encoding="utf-8")
        self.assertIn("uv run --frozen --no-sync", merge_collection)
        rescore = Path("scripts/turing_rescore_evaluation.sh").read_text(encoding="utf-8")
        self.assertEqual(rescore.count("uv run --frozen --no-sync"), 2)
        merge_evaluations = Path("scripts/turing_merge_evaluations.sh").read_text(encoding="utf-8")
        self.assertIn("uv run --frozen --no-sync", merge_evaluations)
        for name in (
            "turing_collect_array.sh",
            "turing_tune_paper.sh",
            "turing_train_paper.sh",
            "turing_evaluate_paper.sh",
        ):
            text = Path("scripts" / Path(name)).read_text(encoding="utf-8")
            self.assertIn("MODEL_CACHE_DIR:?MODEL_CACHE_DIR is required", text, name)
            self.assertIn("nvidia-smi", text, name)
            self.assertIn("HF_HUB_OFFLINE=1", text, name)
            self.assertIn('HF_HUB_CACHE="$MODEL_CACHE_DIR"', text, name)
            self.assertIn("TRANSFORMERS_OFFLINE=1", text, name)
            self.assertIn("--frozen --no-sync", text, name)

    def test_paper_scripts_have_role_specific_commands_and_cleanup_traps(self):
        collect = Path("scripts/turing_collect_array.sh").read_text(encoding="utf-8")
        self.assertIn("SLURM_ARRAY_TASK_ID", collect)
        self.assertIn("collect-shard", collect)
        self.assertIn("SOURCE_COMMIT:?SOURCE_COMMIT is required", collect)
        self.assertIn('source-commit "$SOURCE_COMMIT"', collect)
        merge = Path("scripts/turing_merge_collection.sh").read_text(encoding="utf-8")
        self.assertIn("merge-collection", merge)
        self.assertIn("SOURCE_COMMIT:?SOURCE_COMMIT is required", merge)
        self.assertIn('source-commit "$SOURCE_COMMIT"', merge)
        for name in ("turing_tune_paper.sh", "turing_train_paper.sh", "turing_evaluate_paper.sh"):
            text = Path("scripts" / Path(name)).read_text(encoding="utf-8")
            self.assertIn("trap cleanup EXIT", text, name)
        evaluate = Path("scripts/turing_evaluate_paper.sh").read_text(encoding="utf-8")
        self.assertIn("CHECKPOINT_KIND:?CHECKPOINT_KIND is required", evaluate)
        self.assertIn("SOURCE_COMMIT:?SOURCE_COMMIT is required", evaluate)
        self.assertIn('--checkpoint-kind "$CHECKPOINT_KIND"', evaluate)
        self.assertIn('--source-commit "$SOURCE_COMMIT"', evaluate)
        self.assertIn("SLURM_ARRAY_TASK_ID", evaluate)
        self.assertIn("NUM_SHARDS:?NUM_SHARDS is required", evaluate)
        self.assertIn('--shard-index "$SHARD_INDEX"', evaluate)
        self.assertIn('--num-shards "$NUM_SHARDS"', evaluate)
        merge_evaluations = Path("scripts/turing_merge_evaluations.sh").read_text(encoding="utf-8")
        self.assertIn("merge-evaluations", merge_evaluations)
        self.assertIn("EXPECTED_SHARDS:?EXPECTED_SHARDS is required", merge_evaluations)
        self.assertIn("#SBATCH -n 2", merge_evaluations)
        self.assertIn("#SBATCH -n 2", merge)
        for name in (
            "turing_collect_array.sh",
            "turing_tune_paper.sh",
            "turing_train_paper.sh",
            "turing_evaluate_paper.sh",
        ):
            self.assertIn("--format=csv -l 1", (Path("scripts") / name).read_text(encoding="utf-8"))
        grpo_setup = Path("scripts/turing_setup_grpo_environment.sh").read_text(encoding="utf-8")
        self.assertIn("GRPO_ENVIRONMENT:?GRPO_ENVIRONMENT is required", grpo_setup)
        self.assertIn("environments/grpo", grpo_setup)
        self.assertIn("uv sync --project", grpo_setup)
        self.assertIn("environment_verified.txt.tmp", grpo_setup)
        for name in ("turing_tune_paper.sh", "turing_train_paper.sh"):
            text = (Path("scripts") / name).read_text(encoding="utf-8")
            self.assertIn("GRPO_ENVIRONMENT:?GRPO_ENVIRONMENT is required", text)
            self.assertIn('uv run --project "$PROJECT_DIR/environments/grpo" --frozen --no-sync', text)

    def test_paper_runtime_environments_never_fall_back_to_home_storage(self):
        for name in (
            "turing_stage_model_cache.sh",
            "turing_download_dataset_source.sh",
            "turing_download_math_source.sh",
            "turing_materialize_dataset.sh",
            "turing_audit_dataset.sh",
            "turing_materialize_preflight_subset.sh",
            "turing_decoding_sweep.sh",
            "turing_freeze_decoding.sh",
            "turing_freeze_baseline.sh",
            "turing_collect_array.sh",
            "turing_tune_paper.sh",
            "turing_train_paper.sh",
            "turing_evaluate_paper.sh",
            "turing_merge_evaluations.sh",
            "turing_rescore_evaluation.sh",
        ):
            text = (Path("scripts") / name).read_text(encoding="utf-8")
            self.assertNotIn("$HOME/tfdpo-runs", text, name)
            self.assertIn("RUNTIME_ROOT:?RUNTIME_ROOT is required", text, name)
            self.assertIn('"$RUNTIME_ROOT" != /scratch/*', text, name)
            self.assertIn("environment_verified.txt", text, name)
    def test_model_load_smoke_script_has_required_gpu_checks(self):
        text = Path("scripts/turing_model_load_smoke.sh").read_text(encoding="utf-8")
        self.assertIn("#SBATCH -p u22", text)
        self.assertIn("#SBATCH --gres=gpu:1", text)
        self.assertIn("set -euo pipefail", text)
        self.assertIn("module load u22/cuda/12.4", text)
        self.assertIn("CONFIG:?CONFIG is required", text)
        self.assertIn("MODEL_CACHE_DIR:?MODEL_CACHE_DIR is required", text)
        self.assertIn("PREFLIGHT_OUTPUT:?PREFLIGHT_OUTPUT is required", text)
        self.assertIn("SOURCE_COMMIT:?SOURCE_COMMIT is required", text)
        self.assertIn("RUNTIME_ROOT:?RUNTIME_ROOT is required", text)
        self.assertIn('"$MODEL_CACHE_DIR" != /scratch/*', text)
        self.assertIn("environment_verified.txt", text)
        self.assertIn("tfdpo-model-cache-manifest.json", text)
        self.assertIn("HF_HUB_OFFLINE=1", text)
        self.assertIn("TRANSFORMERS_OFFLINE=1", text)
        self.assertIn("uv run --frozen --no-sync python", text)
        self.assertIn("Qwen/Qwen3-4B", text)
        self.assertIn("Qwen/Qwen3-8B", text)
        self.assertIn("discover_lora_coverage", text)
        self.assertIn("len(coverage.target_modules) != 252", text)
        self.assertIn("enable_thinking=False", text)
        self.assertIn("torch.bfloat16", text)
        self.assertIn("local_files_only=True", text)
        self.assertIn("torch.cuda.is_available()", text)
        self.assertIn("torch.cuda.is_bf16_supported()", text)
        self.assertIn("paper-model-preflight-v1", text)
        self.assertNotIn("|| true", text)

if __name__ == "__main__":
    unittest.main()
