import copy
import unittest
from pathlib import Path

from text_feedback_dpo.batch_generation import run_fixed_retrieval_pipeline
from text_feedback_dpo.bootstrap import (
    collect_bootstrap_rollouts,
    select_bootstrap_pool,
    validate_bootstrap_rows,
)
from text_feedback_dpo.cli import build_parser
from text_feedback_dpo.dataset import build_sft_rows_from_bootstrap
from text_feedback_dpo.runtime import GeneratedText


def _example(example_id="q1") -> dict:
    return {
        "id": example_id,
        "question": "Who wrote the first algorithm?",
        "gold_answer": "Ada Lovelace",
        "sources": [{
            "source_id": "S001", "original_rank": 1, "title": "History",
            "url": "https://example.test/history", "snippet": "Ada Lovelace wrote the first algorithm.",
        }],
    }


def _artifact(example: dict, *, policy_hash="policy-v1") -> dict:
    return run_fixed_retrieval_pipeline(
        [example],
        query_generate_batch=lambda _prompts: [GeneratedText("first algorithm author", False)],
        response_generate_batch=lambda _prompts: [GeneratedText(
            "Answer: Ada Lovelace\nReasoning: The source identifies Ada Lovelace [S001].\nSources: S001",
            False,
        )],
        policy_hash=policy_hash,
    )[0]


class _Tokenizer:
    eos_token = "<eos>"

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))


class BootstrapRolloutsTest(unittest.TestCase):
    def test_query_sft_is_independent_when_response_is_invalid(self):
        example = _example()
        malformed = run_fixed_retrieval_pipeline(
            [example],
            query_generate_batch=lambda _prompts: [GeneratedText("first algorithm author", False)],
            response_generate_batch=lambda _prompts: [GeneratedText("Ada Lovelace", False)],
            policy_hash="policy-v1",
        )[0]
        bootstrap = collect_bootstrap_rollouts(
            [example], seeds=(11,), generate_seed_batch=lambda _batch, **_kwargs: [malformed]
        )
        rows, report = build_sft_rows_from_bootstrap(
            bootstrap, examples={"q1": example}, tokenizer=_Tokenizer()
        )
        self.assertEqual([row["task"] for row in rows], ["query"])
        self.assertEqual(report["query_unique_examples"], 1)
        self.assertEqual(report["response_unique_examples"], 0)
        self.assertEqual(report["response_exclusion_counts"]["response_not_verified_correct"], 1)

    def test_pool_selection_is_exact_deterministic_and_input_order_independent(self):
        examples = [_example(f"q{index}") for index in range(20)]
        selected = select_bootstrap_pool(examples, count=7, seed=20260713)
        reversed_selected = select_bootstrap_pool(reversed(examples), count=7, seed=20260713)
        self.assertEqual([row["id"] for row in selected], [row["id"] for row in reversed_selected])
        self.assertEqual(len(selected), 7)
        with self.assertRaisesRegex(ValueError, "requested 21.*only 20"):
            select_bootstrap_pool(examples, count=21, seed=20260713)

    def test_turing_launcher_is_single_gpu_commit_and_hash_bound(self):
        script = (Path(__file__).parents[1] / "scripts" / "turing_bootstrap_rollouts.sh").read_text()
        self.assertIn("#SBATCH --gres=gpu:1", script)
        self.assertIn("EXPECTED_COMMIT", script)
        self.assertIn("DATA_SHA256", script)
        self.assertIn("fallback_reason=none", script)
        self.assertIn("bootstrap-rollouts", script)

    def test_cli_requires_explicit_seed_list_and_pins_direct_mode(self):
        args = build_parser().parse_args([
            "bootstrap-rollouts", "--data", "train.jsonl", "--output", "rollouts.jsonl",
            "--model", "Qwen/Qwen3-4B-Base", "--model-revision", "rev",
            "--dataset-source", "kyunghyuncho/search_qa", "--dataset-revision", "data-rev",
            "--attention-implementation", "sdpa", "--policy-hash", "a" * 64,
            "--seeds", "11", "12", "13",
        ])
        self.assertEqual(args.seeds, [11, 12, 13])
        self.assertEqual(args.context_budget, 4096)
        self.assertEqual(args.func.__name__, "cmd_bootstrap_rollouts")

    def test_expands_each_seed_deterministically_and_records_no_teacher_provenance(self):
        examples = [_example("q1"), _example("q2")]
        calls = []

        def generate(batch, *, seed):
            calls.append((seed, [row["id"] for row in batch]))
            return [_artifact(row) for row in batch]

        rows = collect_bootstrap_rollouts(examples, seeds=(11, 12, 13), generate_seed_batch=generate)
        self.assertEqual(calls, [(11, ["q1", "q2"]), (12, ["q1", "q2"]), (13, ["q1", "q2"])])
        self.assertEqual([row["id"] for row in rows], ["q1", "q2"])
        self.assertEqual([item["seed"] for item in rows[0]["candidates"]], [11, 12, 13])
        self.assertEqual(sum(len(row["candidates"]) for row in rows), 6)
        for row in rows:
            for candidate in row["candidates"]:
                self.assertEqual(candidate["provenance"], "student")
                self.assertTrue(candidate["no_hint"])
                self.assertNotIn("teacher", candidate)

    def test_rejects_duplicates_cardinality_tampering_and_noncanonical_artifacts(self):
        with self.assertRaisesRegex(ValueError, "duplicate example"):
            collect_bootstrap_rollouts([_example(), _example()], seeds=(1,), generate_seed_batch=lambda *_args, **_kwargs: [])
        with self.assertRaisesRegex(ValueError, "unique"):
            collect_bootstrap_rollouts([_example()], seeds=(1, 1), generate_seed_batch=lambda *_args, **_kwargs: [])
        with self.assertRaisesRegex(ValueError, "cardinality"):
            collect_bootstrap_rollouts([_example()], seeds=(1,), generate_seed_batch=lambda *_args, **_kwargs: [])

        rows = collect_bootstrap_rollouts(
            [_example()], seeds=(1,), generate_seed_batch=lambda batch, **_kwargs: [_artifact(batch[0])]
        )
        forged = copy.deepcopy(rows)
        forged[0]["candidates"][0]["artifact"]["raw_response"] = "forged"
        with self.assertRaisesRegex(ValueError, "cited_score|parsed_response"):
            validate_bootstrap_rows(forged, examples=[_example()], expected_seeds=(1,))


if __name__ == "__main__":
    unittest.main()
