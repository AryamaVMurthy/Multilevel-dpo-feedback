import re
import unittest
from pathlib import Path


class TuringOptimizationProbeTest(unittest.TestCase):
    def test_probe_covers_required_dimensions_and_measurements(self):
        text = Path("scripts/turing_optimization_probe.sh").read_text(encoding="utf-8")
        for dimension in (
            "sdpa",
            "flash_attention_2",
            "GENERATION_BATCH_SIZES",
            "TRAIN_MICROBATCHES",
            "GRADIENT_ACCUMULATION_STEPS",
            "DATALOADER_WORKERS",
            "STATIC_CACHE",
            "COMPILE",
            "PACKING",
        ):
            self.assertIn(dimension, text)
        for measurement in (
            "examples_per_second",
            "tokens_per_second",
            "peak_gpu_memory_mb",
            "gpu_utilization",
            "output_hash",
            "fallback_reason",
        ):
            self.assertIn(measurement, text)

    def test_probe_rejects_mismatched_hash_or_non_improving_throughput_and_never_exports_settings(self):
        text = Path("scripts/turing_optimization_probe.sh").read_text(encoding="utf-8")
        self.assertRegex(text, re.compile(r"output_hash.*baseline_output_hash|baseline_output_hash.*output_hash", re.DOTALL))
        self.assertIn("rejected", text)
        self.assertIn("throughput", text)
        self.assertNotIn("export ATTENTION_IMPLEMENTATION=", text)
        self.assertNotIn("export GENERATION_BATCH_SIZE=", text)

    def test_probe_requires_an_explicit_runner_contract(self):
        text = Path("scripts/turing_optimization_probe.sh").read_text(encoding="utf-8")
        self.assertIn('PROBE_RUNNER:?PROBE_RUNNER', text)
        self.assertIn("PROBE_RESULT", text)
        self.assertIn("set -euo pipefail", text)

    def test_probe_includes_liger_rejection_and_flash_padding_free_packing(self):
        text = Path("scripts/turing_optimization_probe.sh").read_text(encoding="utf-8")
        self.assertIn("liger", text.lower())
        self.assertIn("use_liger_kernel", text)
        self.assertIn("precompute_ref_log_probs", text)
        self.assertIn("padding_free", text)
        self.assertIn("flash_attention_2", text)
        self.assertIn("packing", text.lower())

    def test_probe_artifacts_record_installed_package_versions(self):
        text = Path("scripts/turing_optimization_probe.sh").read_text(encoding="utf-8")
        self.assertIn("PACKAGE_VERSIONS", text)
        self.assertIn("package_versions", text)
        self.assertIn("torch", text)
        self.assertIn("transformers", text)
        self.assertIn("trl", text)
        self.assertIn("deepspeed", text)

    def test_primary_dpo_scripts_never_auto_enable_liger(self):
        for name in ("turing_train.sh", "turing_primary_round.sh", "turing_comparisons.sh"):
            text = Path("scripts", name).read_text(encoding="utf-8").lower()
            self.assertNotIn("use_liger_kernel=true", text, name)
            self.assertNotIn("liger=true", text, name)


if __name__ == "__main__":
    unittest.main()
