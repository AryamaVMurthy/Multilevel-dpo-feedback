import unittest
from pathlib import Path


class TuringScriptTest(unittest.TestCase):
    def test_paper_scripts_use_fail_fast_slurm_and_explicit_artifacts(self):
        for name in (
            "turing_setup_environment.sh",
            "turing_download_dataset_source.sh",
            "turing_materialize_dataset.sh",
            "turing_materialize_preflight_subset.sh",
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
        self.assertIn("uv sync --frozen", setup)
        preflight = Path("scripts/turing_materialize_preflight_subset.sh").read_text(encoding="utf-8")
        self.assertIn("materialize-preflight-subset", preflight)
        self.assertIn("SOURCE_PATH:?SOURCE_PATH is required", preflight)
        self.assertIn("OUTPUT_PATH:?OUTPUT_PATH is required", preflight)
        self.assertIn("SUBSET_COUNT:?SUBSET_COUNT is required", preflight)
        self.assertIn("SUBSET_SEED:?SUBSET_SEED is required", preflight)
        freeze_baseline = Path("scripts/turing_freeze_baseline.sh").read_text(encoding="utf-8")
        self.assertIn("freeze-baseline", freeze_baseline)
        self.assertIn("DATASET_MANIFEST:?DATASET_MANIFEST is required", freeze_baseline)
        self.assertIn("SOURCE_COMMIT:?SOURCE_COMMIT is required", freeze_baseline)
        self.assertIn("FREEZE_OUTPUT:?FREEZE_OUTPUT is required", freeze_baseline)
        for name in (
            "turing_collect_array.sh",
            "turing_tune_paper.sh",
            "turing_train_paper.sh",
            "turing_evaluate_paper.sh",
        ):
            self.assertIn("MODEL_CACHE_DIR:?MODEL_CACHE_DIR is required", Path("scripts" / Path(name)).read_text(encoding="utf-8"), name)
            self.assertIn("nvidia-smi", Path("scripts" / Path(name)).read_text(encoding="utf-8"), name)

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
        grpo_setup = Path("scripts/turing_setup_grpo_environment.sh").read_text(encoding="utf-8")
        self.assertIn("GRPO_ENVIRONMENT:?GRPO_ENVIRONMENT is required", grpo_setup)
        self.assertIn("environments/grpo", grpo_setup)
        self.assertIn("uv sync --project", grpo_setup)
        for name in ("turing_tune_paper.sh", "turing_train_paper.sh"):
            text = (Path("scripts") / name).read_text(encoding="utf-8")
            self.assertIn("GRPO_ENVIRONMENT:?GRPO_ENVIRONMENT is required", text)
            self.assertIn('uv run --project "$PROJECT_DIR/environments/grpo" --frozen --no-sync', text)
    def test_model_load_smoke_script_has_required_gpu_checks(self):
        text = Path("scripts/turing_model_load_smoke.sh").read_text(encoding="utf-8")
        self.assertIn("#SBATCH -p u22", text)
        self.assertIn("#SBATCH --gres=gpu:1", text)
        self.assertIn("set -euo pipefail", text)
        self.assertIn("module load u22/cuda/12.4", text)
        self.assertIn("Qwen/Qwen3.5-2B", text)
        self.assertIn("torch.cuda.is_available()", text)
        self.assertNotIn("|| true", text)

    def test_basic_pair_generation_script_does_not_train(self):
        text = Path("scripts/turing_basic_pair_generation.sh").read_text(encoding="utf-8")
        self.assertIn("${CONFIG:?CONFIG is required}", text)
        self.assertIn("${TURING_ACCOUNT:?TURING_ACCOUNT is required}", text)
        self.assertIn("${HF_CACHE_DIR:?HF_CACHE_DIR is required}", text)
        self.assertIn("#SBATCH -p u22", text)
        self.assertIn("#SBATCH --nodes=1", text)
        self.assertIn("#SBATCH --ntasks=1", text)
        self.assertIn("#SBATCH --cpus-per-task=16", text)
        self.assertIn("#SBATCH --gres=gpu:1", text)
        self.assertIn("#SBATCH --time=01:00:00", text)
        self.assertIn('export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"', text)
        self.assertIn("nvidia-smi --query-gpu", text)
        self.assertIn('${PIPELINE_COMMAND:?PIPELINE_COMMAND is required}', text)
        self.assertIn('native-pipeline', text)
        self.assertIn("output_dir", text)
        self.assertIn("hf_cache_dir", text)
        self.assertIn('UV_CACHE_DIR="/scratch/$USER/text-feedback-dpo/uv_cache"', text)
        self.assertIn("SLURM_JOB_NUM_NODES", text)
        self.assertIn("SLURM_GPUS_ON_NODE", text)
        self.assertIn("allocation_mismatch", text)
        self.assertNotIn("runs/qwen35-basic-smoke/gpu-", text)
        self.assertNotIn("DPOTrainer", text)
        self.assertNotIn("GRPOTrainer", text)
        self.assertNotIn("train-dpo", text)
        self.assertNotIn("train-grpo", text)

    def test_training_script_requires_explicit_method_and_one_gpu(self):
        text = Path("scripts/turing_train.sh").read_text(encoding="utf-8")
        self.assertIn('${TRAIN_METHOD:?TRAIN_METHOD is required}', text)
        self.assertIn('${DATA_PATH:?DATA_PATH is required}', text)
        self.assertIn("#SBATCH --nodes=1", text)
        self.assertIn("#SBATCH --gres=gpu:1", text)
        self.assertIn("train --method", text)


if __name__ == "__main__":
    unittest.main()
