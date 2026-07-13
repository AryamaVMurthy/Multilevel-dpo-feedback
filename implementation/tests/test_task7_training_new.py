import hashlib
import json
import unittest
from types import SimpleNamespace
from dataclasses import fields
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from trl import DPOConfig, GRPOConfig, SFTConfig

from text_feedback_dpo.trainers import (
    REWARD_COMPONENT_WEIGHTS,
    _dpo_args,
    _rl_args,
    _sft_args,
    build_component_reward_functions,
    evaluate_reward_components,
    precompute_reference_log_probs,
    validate_prompt_completion_lengths,
    validate_rl_prompt_budget,
)
from text_feedback_dpo.dataset import build_rl_rows_from_trajectories
from text_feedback_dpo.training import (
    build_reference_manifest,
    validate_precomputed_reference_manifest,
    validate_student_model_selection,
    dataset_identity_hash,
    load_precomputed_reference_log_probs,
    write_precomputed_reference_log_probs,
)


def _example():
    return {
        "id": "q1", "question": "Who wrote the first algorithm?", "gold_answer": "Ada Lovelace",
        "sources": [{"source_id": "S001", "original_rank": 1, "title": "Ada", "url": "https://example.test/ada", "snippet": "Ada Lovelace wrote the first algorithm."}],
    }


def _task7_candidate():
    from text_feedback_dpo.batch_generation import run_fixed_retrieval_pipeline
    from text_feedback_dpo.runtime import GeneratedText
    artifact = run_fixed_retrieval_pipeline(
        [_example()],
        query_generate_batch=lambda prompts: [GeneratedText("Ada algorithm", False) for _ in prompts],
        response_generate_batch=lambda prompts: [GeneratedText("Answer: Ada Lovelace\nReasoning: The source identifies Ada Lovelace [S001].\nSources: S001", False) for _ in prompts],
        policy_hash="policy-v1",
    )[0]
    artifact["verified_no_hint_success"] = True
    artifact["future_sibling_gain"] = 1.0
    return artifact


class _Tokenizer:
    eos_token = "<eos>"
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(1, len(text.split()) + 1))}

    def encode(self, text, add_special_tokens=False):
        return self(text)["input_ids"]


class Task7TrainingTest(unittest.TestCase):
    def test_grpo_constructor_wiring_passes_eval_dataset_all_rewards_and_no_peft(self):
        from text_feedback_dpo.trainers import run_grpo

        class Tokenizer:
            eos_token_id = 2
            pad_token_id = 0

            def __call__(self, text, **kwargs):
                return {"input_ids": [1]}

        train = [{"id": "q::rl::query", "task": "query", "prompt": "query", "gold_answer": "Ada", "sources": [], "canonical_ranked_search_results": []}]
        evaluation = [{"id": "q::rl::response", "task": "response", "prompt": "response", "gold_answer": "Ada", "sources": [], "canonical_ranked_search_results": []}]
        with patch("text_feedback_dpo.trainers._tokenizer", return_value=Tokenizer()), patch(
            "text_feedback_dpo.trainers._load_dataset", side_effect=[train, evaluation]
        ), patch("text_feedback_dpo.trainers.require_bf16_hardware"), patch(
            "trl.GRPOTrainer"
        ) as trainer_class:
            run_grpo(
                model_id="Qwen/Qwen3-4B-Base", train_path=Path("train.jsonl"), eval_path=Path("eval.jsonl"),
                output_dir=Path("out"), config={"max_steps": 1, "model_revision": "rev"},
            )
        kwargs = trainer_class.call_args.kwargs
        self.assertIs(kwargs["peft_config"], None)
        self.assertIs(kwargs["eval_dataset"], evaluation)
        self.assertEqual(len(kwargs["reward_funcs"]), len(REWARD_COMPONENT_WEIGHTS))

    def test_all_trainer_constructors_receive_final_bf16_model_init_kwargs_and_no_peft(self):
        import torch

        from text_feedback_dpo.trainers import run_dapo, run_dpo, run_grpo, run_sft

        captured = []

        class Trainer:
            def __init__(self, **kwargs):
                captured.append(kwargs)

            def train(self, **kwargs):
                self.train_kwargs = kwargs

            def save_model(self, path):
                self.saved_path = path

        sft_rows = [{"id": "s1", "prompt": "P", "completion": " C"}]
        dpo_rows = [{"id": "d1", "prompt": "P", "chosen": " C", "rejected": " R"}]
        persisted = [{**dpo_rows[0], "ref_chosen_logps": -1.0, "ref_rejected_logps": -2.0}]
        rl_rows = [{
            "id": "r1", "task": "query", "prompt": "P", "gold_answer": "Ada",
            "sources": _example()["sources"], "canonical_ranked_search_results": [],
            "stored_query": "Ada algorithm",
        }]
        common = {"max_steps": 1, "model_revision": "pinned-revision", "per_device_eval_batch_size": 4}

        with patch("text_feedback_dpo.trainers.require_bf16_hardware"), patch(
            "text_feedback_dpo.trainers._tokenizer", return_value=_Tokenizer()
        ), patch("trl.SFTTrainer", Trainer), patch(
            "text_feedback_dpo.trainers._load_dataset", side_effect=[sft_rows, sft_rows]
        ):
            run_sft(
                model_id="Qwen/Qwen3-4B-Base", train_path=Path("sft-train.jsonl"),
                eval_path=Path("sft-eval.jsonl"), output_dir=Path("sft-out"), config=common,
            )

        dpo_config = {
            **common,
            "precomputed_ref_log_probs_path": "train.refs.jsonl",
            "precomputed_eval_ref_log_probs_path": "eval.refs.jsonl",
            "reference_checkpoint_hash": "a" * 64,
            "prompt_context_schema": {"schema": 1},
        }
        with patch("text_feedback_dpo.trainers.require_bf16_hardware"), patch(
            "text_feedback_dpo.trainers._tokenizer", return_value=_Tokenizer()
        ), patch("text_feedback_dpo.trainers._load_dataset", side_effect=[dpo_rows, dpo_rows]), patch(
            "text_feedback_dpo.trainers.load_precomputed_reference_log_probs",
            side_effect=[persisted, persisted],
        ), patch("datasets.Dataset.from_list", side_effect=lambda rows: rows), patch("trl.DPOTrainer", Trainer):
            run_dpo(
                model_id="Qwen/Qwen3-4B-Base", train_path=Path("dpo-train.jsonl"),
                eval_path=Path("dpo-eval.jsonl"), output_dir=Path("dpo-out"), config=dpo_config,
            )

        for method, runner in (("grpo", run_grpo), ("dapo", run_dapo)):
            with patch("text_feedback_dpo.trainers.require_bf16_hardware"), patch(
                "text_feedback_dpo.trainers._tokenizer", return_value=_Tokenizer()
            ), patch("text_feedback_dpo.trainers._load_dataset", side_effect=[rl_rows, rl_rows]), patch(
                "trl.GRPOTrainer", Trainer,
            ):
                runner(
                    model_id="Qwen/Qwen3-4B-Base", train_path=Path(f"{method}-train.jsonl"),
                    eval_path=Path(f"{method}-eval.jsonl"), output_dir=Path(f"{method}-out"),
                    config={**common, "dapo_enabled": method == "dapo"},
                )

        self.assertEqual(len(captured), 4)
        for constructor in captured:
            model_kwargs = constructor["args"].model_init_kwargs
            self.assertEqual(model_kwargs["dtype"], torch.bfloat16)
            self.assertNotIn("torch_dtype", model_kwargs)
            self.assertTrue(constructor["args"].bf16)
            self.assertFalse(constructor["args"].fp16)
            self.assertIsNone(constructor["peft_config"])

    def test_bf16_runtime_gate_rejects_cpu_and_unsupported_cuda_without_fallback(self):
        from text_feedback_dpo.trainers import require_bf16_hardware

        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA hardware"):
                require_bf16_hardware()
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.is_bf16_supported", return_value=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "does not support BF16"):
                require_bf16_hardware()

    def test_rl_rows_include_query_and_response_tasks_with_dataset_context(self):
        candidate = _task7_candidate()
        rows, report = build_rl_rows_from_trajectories(
            [{"id": "q1", "training_eligible": True, "query_prompt": candidate["query_prompt"], "query_prompt_hash": candidate["query_prompt_hash"], "no_hint_siblings": [candidate]}],
            examples={"q1": _example()},
            tokenizer=_Tokenizer(),
        )
        self.assertEqual({row["task"] for row in rows}, {"query", "response"})
        self.assertEqual({row["gold_answer"] for row in rows}, {"Ada Lovelace"})
        self.assertEqual(report["query_rows"], 1)
        self.assertEqual(report["response_rows"], 1)

    def test_configs_match_installed_trl_fields_and_rl_has_no_legacy_prompt_length(self):
        for config_cls, args in ((SFTConfig, _sft_args({"max_steps": 1}, "out")), (DPOConfig, _dpo_args({"max_steps": 1}, "out"))):
            names = {field.name for field in fields(config_cls)}
            self.assertTrue(set(args) <= names)
            self.assertEqual(args["max_length"], 4096)
        rl = _rl_args({"max_steps": 1}, "out", method="grpo")
        names = {field.name for field in fields(GRPOConfig)}
        self.assertTrue(set(rl) <= names)
        self.assertNotIn("max_length", rl)
        self.assertNotIn("max_prompt_length", rl)
        self.assertEqual(rl["loss_type"], "grpo")
        self.assertGreater(rl["max_completion_length"], 32)

    def test_dapo_parameters_and_liger_request_are_explicit(self):
        dapo = _rl_args({"max_steps": 1}, "out", method="dapo")
        self.assertEqual(dapo["loss_type"], "dapo")
        self.assertEqual(dapo["epsilon"], 0.2)
        self.assertEqual(dapo["epsilon_high"], 0.28)
        self.assertTrue(dapo["mask_truncated_completions"])
        self.assertEqual(dapo["beta"], 0.0)
        self.assertFalse(_dpo_args({"max_steps": 1}, "out")["use_liger_kernel"])
        with self.assertRaisesRegex(ValueError, "Liger"):
            _dpo_args({"max_steps": 1, "use_liger_kernel": True}, "out")

    def test_dpo_ref_artifacts_are_mandatory_with_no_live_fallback(self):
        from text_feedback_dpo.trainers import run_dpo

        with patch("text_feedback_dpo.trainers._tokenizer", return_value=_Tokenizer()):
            with self.assertRaisesRegex(ValueError, "both train and eval"):
                run_dpo(
                    model_id="Qwen/Qwen3-4B-Base", train_path=Path("train.jsonl"),
                    eval_path=Path("eval.jsonl"), output_dir=Path("out"),
                    config={"max_steps": 1, "model_revision": "rev"},
                )

    def test_rl_prompt_budget_rejects_overlong_prompts_without_truncation(self):
        class Tokenizer:
            def __call__(self, text, **kwargs):
                self.kwargs = kwargs
                return {"input_ids": list(range(len(text.split())))}

        tokenizer = Tokenizer()
        with self.assertRaisesRegex(ValueError, "prompt token budget"):
            validate_rl_prompt_budget([{"prompt": "one two three four", "task": "query", "gold_answer": "x"}], tokenizer, 4094)
        validate_rl_prompt_budget([{"prompt": "one", "task": "query", "gold_answer": "x"}], tokenizer, 4094)
        self.assertFalse(tokenizer.kwargs.get("truncation", True))

    def test_dpo_boundary_merge_is_rejected_before_training_or_precompute(self):
        class BoundaryMergingTokenizer(_Tokenizer):
            def __call__(self, text, **kwargs):
                if text == "prompt":
                    return {"input_ids": [1]}
                return {"input_ids": [9, 2]}

        with self.assertRaisesRegex(ValueError, "boundary mismatch"):
            validate_prompt_completion_lengths(
                [{"prompt": "prompt", "chosen": " chosen", "rejected": " rejected"}],
                BoundaryMergingTokenizer(), method="dpo",
            )
        with self.assertRaisesRegex(ValueError, "boundary mismatch"):
            validate_prompt_completion_lengths(
                [{"prompt": "prompt", "completion": " completion"}],
                BoundaryMergingTokenizer(), method="sft",
            )

    def test_reward_components_use_strict_evaluator_and_verbosity_is_never_positive(self):
        candidate = _task7_candidate()
        ranked = candidate["canonical_ranked_search_results"]
        response = "Answer: Ada Lovelace\nReasoning: The source identifies Ada Lovelace [S001].\nSources: S001"
        score = evaluate_reward_components(
            response, "Ada Lovelace", ranked, sources=_example()["sources"], stored_query=candidate["raw_query"],
        )
        self.assertEqual(score["components"]["exact_answer"], 1.0)
        self.assertIn("weighted_total", score)
        self.assertLessEqual(score["components"]["verbosity_penalty"], 0.0)
        forged = [dict(ranked[0], title="forged dataset context")]
        with self.assertRaisesRegex(ValueError, "retrieval mismatch"):
            evaluate_reward_components(
                response, "Ada Lovelace", forged,
                sources=_example()["sources"], stored_query=candidate["raw_query"],
            )
        malformed = evaluate_reward_components(
            "Answer: Ada Lovelace\nReasoning: forged [S999].\nSources: S999", "Ada Lovelace", ranked,
            sources=_example()["sources"], stored_query=candidate["raw_query"],
        )
        self.assertLess(malformed["weighted_total"], score["weighted_total"])
        funcs = build_component_reward_functions(_Tokenizer())
        self.assertEqual({func.__name__ for func in funcs}, {
            "exact_answer_reward", "bounded_f1_reward", "retrieval_recall_reward", "retrieval_mrr_reward", "future_retrieval_proxy_reward",
            "valid_citations_reward", "lexical_support_reward", "concise_reasoning_reward",
            "malformed_penalty", "fabricated_citation_penalty", "truncation_penalty", "verbosity_penalty",
        })

    def test_reward_functions_branch_on_task_and_penalize_fabrication_and_truncation(self):
        candidate = _task7_candidate()
        ranked = candidate["canonical_ranked_search_results"]
        funcs = {func.__name__: func for func in build_component_reward_functions(_Tokenizer())}
        values = funcs["retrieval_recall_reward"](
            ["Ada algorithm", "Answer: Ada Lovelace\nReasoning: The source identifies Ada Lovelace [S001].\nSources: S001"],
            task=["query", "response"], gold_answer=["Ada Lovelace", "Ada Lovelace"],
            sources=[[_example()["sources"][0]], [_example()["sources"][0]]],
            canonical_ranked_search_results=[ranked, ranked],
            stored_query=[candidate["raw_query"], candidate["raw_query"]], completion_ids=[[1, 2], [1, 2]],
        )
        self.assertGreater(values[0], 0.0)
        self.assertGreater(values[1], 0.0)
        fabricated = funcs["fabricated_citation_penalty"](
            ["Answer: Ada Lovelace\nReasoning: forged [S999].\nSources: S999"], task=["response"],
            gold_answer=["Ada Lovelace"], sources=[[_example()["sources"][0]]],
            canonical_ranked_search_results=[ranked], stored_query=[candidate["raw_query"]], completion_ids=[[1, 2]],
        )
        truncated = funcs["truncation_penalty"](
            ["Answer: Ada Lovelace"], task=["response"], gold_answer=["Ada Lovelace"],
            sources=[[_example()["sources"][0]]], canonical_ranked_search_results=[ranked], truncated=[True],
            stored_query=[candidate["raw_query"]], completion_ids=[[1, 9]],
        )
        self.assertLess(fabricated[0], 0.0)
        self.assertLess(truncated[0], 0.0)
        self.assertLessEqual(funcs["verbosity_penalty"](
            ["x"], task=["query"], gold_answer=["Ada"], sources=[[_example()["sources"][0]]],
            canonical_ranked_search_results=[ranked], stored_query=[candidate["raw_query"]], completion_ids=[[2]],
        )[0], 0.0)

    def test_reference_manifest_requires_exact_identity_match(self):
        manifest = build_reference_manifest(
            model="Qwen/Qwen3-4B-Base", model_revision="model-rev", reference_checkpoint_hash="a" * 64,
            tokenizer="Qwen/Qwen3-4B-Base", tokenizer_revision="tok-rev", data_hash="b" * 64,
            prompt_context_schema={"prompt": "fixed", "response": "cited", "schema": 1}, max_length=4096,
        )
        validate_precomputed_reference_manifest(manifest, manifest)
        changed = dict(manifest)
        changed["model"] = dict(manifest["model"])
        changed["model"]["revision"] = "different"
        with self.assertRaisesRegex(ValueError, "manifest mismatch"):
            validate_precomputed_reference_manifest(changed, manifest)

    def test_persisted_reference_logprob_artifact_is_strict_finite_and_reusable(self):
        rows = [{"id": "r1", "prompt": "p", "chosen": " c", "rejected": " r"}]
        identity = {
            "model": "Qwen/Qwen3-4B-Base", "model_revision": "model-rev",
            "reference_checkpoint_hash": "a" * 64, "tokenizer": "Qwen/Qwen3-4B-Base",
            "tokenizer_revision": "tok-rev", "prompt_context_schema": {"schema": 1}, "max_length": 4096,
        }
        manifest = build_reference_manifest(data_hash=dataset_identity_hash(rows), **identity)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "ref.jsonl"
            write_precomputed_reference_log_probs(path, [{**rows[0], "ref_chosen_logps": -1.0, "ref_rejected_logps": -2.0}], manifest)
            loaded = load_precomputed_reference_log_probs(path, manifest)
            self.assertEqual(loaded[0]["ref_chosen_logps"], -1.0)
            forged = [{**rows[0], "ref_chosen_logps": float("nan"), "ref_rejected_logps": -2.0}]
            with self.assertRaisesRegex(ValueError, "finite"):
                write_precomputed_reference_log_probs(path, forged, manifest)
            mismatch = dict(manifest)
            mismatch["data_hash"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "manifest mismatch"):
                load_precomputed_reference_log_probs(path, mismatch)

    def test_precompute_writes_real_finite_rows_with_exact_data_identity(self):
        import torch

        class CharacterTokenizer:
            eos_token = "!"

            def __call__(self, text, **kwargs):
                return {"input_ids": [ord(character) % 31 + 1 for character in text]}

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))

            def forward(self, input_ids, attention_mask, use_cache):
                batch, length = input_ids.shape
                return SimpleNamespace(logits=torch.zeros(batch, length, 64, device=input_ids.device))

        rows = [{"id": "r1", "prompt": "P", "chosen": " C", "rejected": " R"}]
        manifest = build_reference_manifest(
            model="Qwen/Qwen3-4B-Base", model_revision="model-rev",
            reference_checkpoint_hash="a" * 64, tokenizer="Qwen/Qwen3-4B-Base",
            tokenizer_revision="model-rev", data_hash=dataset_identity_hash(rows),
            prompt_context_schema={"schema": 1}, max_length=4096,
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "refs.jsonl"
            result = precompute_reference_log_probs(
                model=Model(), tokenizer=CharacterTokenizer(), rows=rows,
                output_path=path, manifest=manifest,
            )
            loaded = load_precomputed_reference_log_probs(path, manifest)
        self.assertEqual(result["rows"], 1)
        self.assertEqual(dataset_identity_hash(loaded), dataset_identity_hash(rows))
        self.assertTrue(all(isinstance(loaded[0][key], float) for key in ("ref_chosen_logps", "ref_rejected_logps")))

    def test_model_override_and_fallback_require_pinned_or_authorized_oom(self):
        config = {"student_model": "Qwen/Qwen3-4B-Base", "student_revision": "906bfd4b4dc7f14ee4320094d8b41684abff8539", "training": {}}
        self.assertEqual(validate_student_model_selection(config), ("Qwen/Qwen3-4B-Base", config["student_revision"]))
        with self.assertRaisesRegex(ValueError, "model override"):
            validate_student_model_selection(config, requested_model="other", requested_revision="other-rev")

    def test_full_finetune_may_start_from_exact_hash_bound_local_4b_checkpoint(self):
        config = {
            "student_model": "Qwen/Qwen3-4B-Base",
            "student_revision": "906bfd4b4dc7f14ee4320094d8b41684abff8539",
            "training": {},
        }
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            model_file = checkpoint / "model.safetensors"
            model_file.write_bytes(b"full 4b checkpoint weights")
            expected = hashlib.sha256(model_file.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "initial checkpoint SHA-256"):
                validate_student_model_selection(
                    config, requested_model=str(checkpoint), requested_revision=config["student_revision"]
                )
            with self.assertRaisesRegex(ValueError, "checkpoint hash mismatch"):
                validate_student_model_selection(
                    config, requested_model=str(checkpoint), requested_revision=config["student_revision"],
                    initial_checkpoint_sha256="0" * 64,
                )
            self.assertEqual(
                validate_student_model_selection(
                    config, requested_model=str(checkpoint), requested_revision=config["student_revision"],
                    initial_checkpoint_sha256=expected,
                ),
                (str(checkpoint), config["student_revision"]),
            )

    def test_fallback_requires_persisted_intended_config_cuda_oom_evidence(self):
        with TemporaryDirectory() as directory:
            evidence = Path(directory) / "oom.json"
            config_hash = "c" * 64
            config = {
                "student_model": "Qwen/Qwen3-4B-Base",
                "student_revision": "906bfd4b4dc7f14ee4320094d8b41684abff8539",
                "training": {
                    "intended_config_hash": config_hash,
                    "student_fallback_model": "Qwen/Qwen3-1.7B-Base",
                    "student_fallback_revision": "fallback-rev",
                    "student_fallback_oom_artifact": str(evidence),
                },
            }
            evidence.write_text(json.dumps({
                "status": "failed", "error_type": "cuda_oom", "authorized_fallback": True,
                "intended_model": config["student_model"], "intended_revision": config["student_revision"],
                "intended_config_hash": config_hash,
            }), encoding="utf-8")
            self.assertEqual(validate_student_model_selection(config, requested_model="Qwen/Qwen3-1.7B-Base", requested_revision="fallback-rev"), ("Qwen/Qwen3-1.7B-Base", "fallback-rev"))
            evidence.write_text(evidence.read_text(encoding="utf-8").replace("cuda_oom", "runtime_error"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "CUDA OOM"):
                validate_student_model_selection(config, requested_model="Qwen/Qwen3-1.7B-Base", requested_revision="fallback-rev")


if __name__ == "__main__":
    unittest.main()
