import unittest
from pathlib import Path


class TuringScriptTest(unittest.TestCase):
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
        self.assertIn("#SBATCH --gres=gpu:1", text)
        self.assertIn("#SBATCH --time=00:20:00", text)
        self.assertIn('export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"', text)
        self.assertIn("nvidia-smi --query-gpu", text)
        self.assertIn("generate-pipeline", text)
        self.assertIn("output_dir", text)
        self.assertIn("hf_cache_dir", text)
        self.assertNotIn("runs/qwen35-basic-smoke/gpu-", text)
        self.assertNotIn("DPOTrainer", text)
        self.assertNotIn("GRPOTrainer", text)
        self.assertNotIn("train-dpo", text)
        self.assertNotIn("train-grpo", text)


if __name__ == "__main__":
    unittest.main()
