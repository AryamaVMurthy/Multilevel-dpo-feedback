import unittest
import re
from pathlib import Path

from text_feedback_dpo.config import load_config


class TuringRuntimeTest(unittest.TestCase):
    OWNED_SCRIPTS = (
        "turing_prepare.sh",
        "turing_generate.sh",
        "turing_collect.sh",
        "turing_train.sh",
        "turing_preflight.sh",
        "turing_primary_round.sh",
        "turing_comparisons.sh",
        "turing_finalize_report.sh",
    )

    def test_primary_config_is_searchqa_4b_full_finetune_with_quantized_teacher_candidates(self):
        config = load_config(Path("configs/searchqa.yaml"))
        self.assertEqual(config["student_model"], "Qwen/Qwen3-4B-Base")
        self.assertEqual(config["teacher_model"], "Qwen/Qwen3-32B")
        self.assertEqual(config["training"]["teacher_fallback_model"], "Qwen/Qwen3-14B")
        self.assertEqual(config["training"]["teacher_quantization"], "4bit")
        self.assertTrue(config["training"]["full_finetuning"])
        self.assertEqual(config["teacher_generation"]["max_new_tokens"], 512)

    def test_turing_scripts_fail_fast_and_use_no_hidden_fallback(self):
        for path in Path("scripts").glob("*.sh"):
            text = path.read_text(encoding="utf-8")
            self.assertIn("set -euo pipefail", text)
            self.assertNotIn("|| true", text)
            if "#SBATCH" in text:
                self.assertNotIn("#SBATCH -n ", text)

    def test_generation_is_plain_short_answer_with_explicit_thinking_mode(self):
        text = Path("scripts/turing_generate.sh").read_text(encoding="utf-8")
        self.assertIn("generate-searchqa", text)
        self.assertNotIn("active-generate", text)
        self.assertIn("QUERY_BATCH_SIZE", text)
        self.assertIn("RESPONSE_BATCH_SIZE", text)
        self.assertIn("QUERY_MAX_NEW_TOKENS", text)
        self.assertIn("RESPONSE_MAX_NEW_TOKENS", text)
        self.assertIn('--context-budget 4096', text)
        self.assertIn("STUDENT_THINKING_MODE", text)
        self.assertIn('${STUDENT_THINKING_MODE:?', text)
        self.assertNotIn('STUDENT_THINKING_MODE="${STUDENT_THINKING_MODE:-direct}"', text)
        for setting in ("SCRATCHPAD_MAX_NEW_TOKENS", "QUERY_TEMPERATURE", "RESPONSE_TEMPERATURE", "TOP_P"):
            self.assertIn(f'${{{setting}:?', text)
        self.assertIn("SCRATCHPAD_MAX_NEW_TOKENS", text)
        self.assertNotIn("--max-new-tokens 512", text)

    def test_collection_uses_one_explicit_teacher_identity_and_complete_cache_key(self):
        text = Path("scripts/turing_collect.sh").read_text(encoding="utf-8")
        for value in ("STUDENT_REVISION", "DATASET_REVISION", "PROMPT_VERSION", "SEED", "--teacher-thinking"):
            self.assertIn(value, text)
        self.assertNotIn("TEACHER_FALLBACK", text)
        self.assertIn("STUDENT_BATCH_SIZE", text)
        self.assertIn("TEACHER_BATCH_SIZE", text)
        self.assertIn("TEACHER_MAX_NEW_TOKENS", text)
        self.assertIn('--teacher-max-new-tokens "$TEACHER_MAX_NEW_TOKENS"', text)
        self.assertNotIn("GENERATION_BATCH_SIZE", text)

    def test_gpu_scripts_use_conservative_uv_and_verify_source_root(self):
        for name in ("turing_generate.sh", "turing_collect.sh", "turing_train.sh", "turing_preflight.sh"):
            text = Path("scripts", name).read_text(encoding="utf-8")
            self.assertIn("UV_CONCURRENT_DOWNLOADS=1", text)
            self.assertIn("src/text_feedback_dpo", text)

    def test_retirement_script_is_scoped_and_archive_first(self):
        text = Path("scripts/turing_retire_invalid_run.sh").read_text(encoding="utf-8")
        self.assertIn("EXPECTED_INVALID_RUN_ROOT", text)
        self.assertIn("realpath", text)
        self.assertIn("ARCHIVE_DIR", text)
        self.assertIn('$actual/baseline/shards', text)
        self.assertNotIn("rm -rf", text)

    def test_prompt_preflight_compares_direct_and_two_pass(self):
        text = Path("scripts/turing_prompt_preflight.sh").read_text(encoding="utf-8")
        self.assertIn("direct", text)
        self.assertIn("two_pass", text)
        self.assertIn("preflight-quality", text)
        self.assertIn("select-thinking-mode", text)
        self.assertIn("SCRATCHPAD_MAX_NEW_TOKENS", text)

    def test_model_preflight_actually_runs_role_specific_probe(self):
        text = Path("scripts/turing_preflight.sh").read_text(encoding="utf-8")
        self.assertIn("MODEL_ROLE", text)
        self.assertIn("probe-model", text)
        self.assertIn("--teacher-quantization 4bit", text)
        self.assertIn("--teacher-max-new-tokens 512", text)

    def test_training_uses_configurable_multi_gpu_with_fixed_effective_batch(self):
        text = Path("scripts/turing_train.sh").read_text(encoding="utf-8")
        self.assertIn('TRAIN_GPUS" != "4"', text)
        self.assertIn("SLURM_NNODES", text)
        self.assertIn('SLURM_NNODES" != "1"', text)
        self.assertIn("SLURM_GPUS_ON_NODE", text)
        self.assertIn("ALLOCATED_GPU_COUNT", text)
        self.assertIn("EFFECTIVE_BATCH_SIZE", text)
        self.assertIn("GRADIENT_ACCUMULATION_STEPS", text)
        self.assertIn('--gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"', text)
        self.assertIn('--nproc_per_node="$ALLOCATED_GPU_COUNT"', text)
        self.assertIn("uv run --frozen python -m torch.distributed.run", text)
        for setting in ("LEARNING_RATE", "EPOCHS", "SAVE_STEPS", "EVAL_STEPS"):
            self.assertIn(f'${{{setting}:?', text)
        self.assertIn('--learning-rate "$LEARNING_RATE"', text)
        self.assertIn('--epochs "$EPOCHS"', text)

    def test_owned_scripts_have_single_node_headers_and_no_xml_status_lines(self):
        for name in self.OWNED_SCRIPTS:
            text = Path("scripts", name).read_text(encoding="utf-8")
            self.assertIn("#SBATCH --nodes=1", text, name)
            self.assertNotRegex(text, re.compile(r"(?:echo|printf)\s+['\"]<[^\n]*>", re.MULTILINE), name)
            self.assertNotIn("<runtime", text, name)
            self.assertNotIn("<report", text, name)

    def test_collection_is_exactly_two_gpu_single_allocation_with_deterministic_shard_identity(self):
        text = Path("scripts/turing_collect.sh").read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:2", text)
        self.assertIn('SLURM_NNODES" != "1"', text)
        self.assertIn('SLURM_NTASKS" != "1"', text)
        self.assertIn("ALLOCATED_GPU_COUNT", text)
        self.assertIn('ALLOCATED_GPU_COUNT" != "2"', text)
        for value in ("SHARD_INDEX", "SHARD_COUNT", "SHARD_SEED", "MERGE_ID"):
            self.assertIn(value, text)
        self.assertIn("cuda:0", text)
        self.assertIn("cuda:1", text)
        self.assertIn("--teacher-batch-size", text)
        self.assertIn("--student-batch-size", text)
        self.assertIn("--teacher-max-new-tokens", text)
        self.assertIn("--answer-max-new-tokens", text)
        self.assertIn("--scratchpad-max-new-tokens", text)

    def test_scripts_record_required_manifest_identity_and_observability(self):
        for name in self.OWNED_SCRIPTS:
            text = Path("scripts", name).read_text(encoding="utf-8")
            for field in (
                "commit_hash",
                "config_hash",
                "model_hash",
                "dataset_hash",
                "prompt_hash",
                "retrieval_hash",
                "source_schema_hash",
                "SLURM_JOB_NODELIST",
                "GPU_TELEMETRY",
                "artifact_paths",
                "package_versions",
            ):
                self.assertIn(field, text, f"{field} missing from {name}")

    def test_training_preserves_full_finetune_and_smoke_gates(self):
        text = Path("scripts/turing_train.sh").read_text(encoding="utf-8")
        self.assertNotIn("CHECKPOINT_SMOKE_COMMAND", text)
        self.assertNotIn("RESUME_SMOKE_COMMAND", text)
        self.assertIn("validate-checkpoints", text)
        self.assertIn("CHECKPOINT_SMOKE_MANIFEST", text)
        self.assertIn("CHECKPOINT_SMOKE_MANIFEST_SHA256", text)
        self.assertIn("bf16", text.lower())
        self.assertIn("TF32", text)
        self.assertIn("zero3", text.lower())
        self.assertIn("fused", text.lower())
        self.assertIn("non-reentrant", text.lower())

    def test_generation_and_training_consume_a_hashed_frozen_optimization_decision(self):
        for name in ("turing_generate.sh", "turing_train.sh", "turing_comparisons.sh"):
            text = Path("scripts", name).read_text(encoding="utf-8")
            self.assertIn("OPTIMIZATION_DECISION", text, name)
            self.assertIn("OPTIMIZATION_DECISION_SHA256", text, name)
            self.assertIn("validate-decision", text, name)
            self.assertNotIn('ATTENTION_IMPLEMENTATION="${ATTENTION_IMPLEMENTATION:-', text, name)
        comparisons = Path("scripts/turing_comparisons.sh").read_text(encoding="utf-8")
        self.assertNotIn("CHECKPOINT_SMOKE_COMMAND", comparisons)
        self.assertNotIn("RESUME_SMOKE_COMMAND", comparisons)
        self.assertIn("validate-checkpoints", comparisons)

    def test_no_unsafe_hard_coded_two_process_training_remains(self):
        for name in ("turing_train.sh", "turing_primary_round.sh", "turing_comparisons.sh"):
            text = Path("scripts", name).read_text(encoding="utf-8")
            self.assertNotRegex(text, re.compile(r"--nproc_per_node(?:=|\s+)2\b"), name)

    def test_new_research_path_uses_explicit_active_search_protocol(self):
        evaluate = Path("scripts/turing_evaluate.sh").read_text(encoding="utf-8")
        prompt = Path("scripts/turing_prompt_preflight.sh").read_text(encoding="utf-8")
        comparisons = Path("scripts/turing_comparisons.sh").read_text(encoding="utf-8")
        primary = Path("scripts/turing_primary_round.sh").read_text(encoding="utf-8")
        self.assertIn('PROTOCOL:?PROTOCOL', evaluate)
        self.assertIn('--protocol "$PROTOCOL"', evaluate)
        self.assertIn("generate-searchqa", prompt)
        self.assertIn("--protocol active-search", prompt)
        self.assertIn("generate-searchqa", comparisons)
        self.assertIn('--protocol active-search', comparisons)
        self.assertIn('--protocol active-search', primary)
        for text, name in ((prompt, "prompt"), (comparisons, "comparisons"), (primary, "primary")):
            self.assertNotRegex(text, re.compile(r"cli generate(?:\s|\\)"), name)

    def test_merge_and_offline_scripts_are_structured_and_protocol_scoped(self):
        for name in ("turing_merge_predictions.sh", "turing_offline_reuse_check.sh"):
            text = Path("scripts", name).read_text(encoding="utf-8")
            self.assertIn("#SBATCH --nodes=1", text, name)
            self.assertNotRegex(text, re.compile(r"(?:echo|printf)\s+['\"]<[^\n]*>", re.MULTILINE), name)
            self.assertIn("PROTOCOL", text, name)
            self.assertIn("fallback_reason", text, name)

    def test_protocol_audit_scripts_have_single_node_headers(self):
        for name in ("turing_evaluate.sh", "turing_prompt_preflight.sh"):
            text = Path("scripts", name).read_text(encoding="utf-8")
            self.assertIn("#SBATCH --nodes=1", text, name)


if __name__ == "__main__":
    unittest.main()
