import unittest

from text_feedback_dpo.batch_generation import run_fixed_retrieval_pipeline
from text_feedback_dpo.bootstrap import collect_bootstrap_rollouts
from text_feedback_dpo.dataset import build_sft_rows_from_bootstrap
from text_feedback_dpo.monitoring import build_sft_capability_report, build_sft_reproduction_report
from text_feedback_dpo.runtime import GeneratedText


def _example() -> dict:
    return {
        "id": "q1",
        "question": "Who wrote the first algorithm?",
        "gold_answer": "Ada Lovelace",
        "sources": [{
            "source_id": "S001", "original_rank": 1, "title": "History",
            "url": "https://example.test/history", "snippet": "Ada Lovelace wrote the first algorithm.",
        }],
    }


class _Tokenizer:
    eos_token = "<eos>"

    @staticmethod
    def encode(text, add_special_tokens=False):
        return list(range(len(text.split())))


class SFTReproductionTest(unittest.TestCase):
    def test_report_counts_exact_empty_truncated_and_tasks_without_repair(self):
        rows = [
            {"id": "q1", "task": "query", "completion": " alpha query"},
            {"id": "r1", "task": "response", "completion": " answer\nReasoning: evidence [S001].\nSources: S001"},
        ]
        generated = {
            "q1": GeneratedText("alpha query", False),
            "r1": GeneratedText("", True),
        }
        records, summary = build_sft_reproduction_report(rows, generated)
        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["exact"], 1)
        self.assertEqual(summary["exact_rate"], 0.5)
        self.assertEqual(summary["empty"], 1)
        self.assertEqual(summary["truncated"], 1)
        self.assertEqual(summary["tasks"]["query"]["exact_rate"], 1.0)
        self.assertEqual(summary["tasks"]["response"]["exact_rate"], 0.0)
        self.assertEqual(records[1]["reference"], rows[1]["completion"].strip())
        self.assertEqual(records[1]["generated"], "")

    def test_report_rejects_missing_duplicate_or_unexpected_generation_ids(self):
        row = {"id": "q1", "task": "query", "completion": " query"}
        with self.assertRaisesRegex(ValueError, "generation ID parity"):
            build_sft_reproduction_report([row], {})
        with self.assertRaisesRegex(ValueError, "generation ID parity"):
            build_sft_reproduction_report([row], {"q1": GeneratedText("query", False), "extra": GeneratedText("x", False)})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_sft_reproduction_report([row, row], {"q1": GeneratedText("query", False)})

    def test_capability_report_canonically_revalidates_paraphrases_instead_of_requiring_exact_text(self):
        example = _example()
        artifact = run_fixed_retrieval_pipeline(
            [example],
            query_generate_batch=lambda _prompts: [GeneratedText("first algorithm author", False)],
            response_generate_batch=lambda _prompts: [GeneratedText(
                "Answer: Ada Lovelace\nReasoning: The source names Ada Lovelace [S001].\nSources: S001",
                False,
            )],
            policy_hash="policy-v1",
        )[0]
        bootstrap = collect_bootstrap_rollouts(
            [example], seeds=(11,), generate_seed_batch=lambda _batch, **_kwargs: [artifact]
        )
        sft_rows, _report = build_sft_rows_from_bootstrap(
            bootstrap, examples={"q1": example}, tokenizer=_Tokenizer()
        )
        generated = []
        for row in sft_rows:
            text = (
                "algorithm author Ada Lovelace"
                if row["task"] == "query"
                else "Answer: Ada Lovelace\nReasoning: Evidence identifies her [S001].\nSources: S001"
            )
            generated.append({
                "id": row["id"], "task": row["task"], "reference": row["completion"].strip(),
                "generated": text, "exact": False, "empty": False, "truncated": False,
            })
        records, summary = build_sft_capability_report(
            sft_rows,
            generated,
            examples_by_id={"q1": example},
            bootstrap_by_id={"q1": bootstrap[0]},
        )
        self.assertEqual(summary["tasks"]["query"]["retrieval_recall@8"], 1)
        self.assertEqual(summary["tasks"]["response"]["parse_valid"], 1)
        self.assertEqual(summary["tasks"]["response"]["answer_correct"], 1)
        self.assertEqual(summary["tasks"]["response"]["strict_sft_eligible"], 1)
        self.assertFalse(any(record["exact"] for record in records))

    def test_capability_report_rejects_lineage_mismatch(self):
        row = {
            "id": "q1::sft::query", "task": "query", "prompt": "prompt", "completion": "query",
            "metadata": {"trajectory_id": "q1", "seed": 11, "provenance": "student", "no_hint": True},
        }
        generated = [{
            "id": row["id"], "task": "query", "reference": "query", "generated": "query",
            "exact": True, "empty": False, "truncated": False,
        }]
        with self.assertRaisesRegex(ValueError, "bootstrap lineage"):
            build_sft_capability_report(
                [row], generated, examples_by_id={"q1": _example()}, bootstrap_by_id={}
            )


if __name__ == "__main__":
    unittest.main()
