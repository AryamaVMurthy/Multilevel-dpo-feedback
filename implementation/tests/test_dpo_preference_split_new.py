import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from text_feedback_dpo.cli import build_parser
from text_feedback_dpo import dataset


def _preference_row(row_id: str, pair_type: str, example_id: str) -> dict:
    return {
        "id": row_id,
        "prompt": f"prompt-{example_id}-{pair_type}",
        "chosen": " chosen",
        "rejected": " rejected",
        "metadata": {"pair_type": pair_type, "example_id": example_id},
    }


def _balanced_rows(example_ids: list[str]) -> list[dict]:
    return [
        _preference_row(f"{example_id}-{pair_type}", pair_type, example_id)
        for example_id in example_ids
        for pair_type in ("query", "response")
    ]


class DPOPreferenceSplitTest(unittest.TestCase):
    def test_split_is_deterministic_balanced_example_disjoint_and_preserves_rows(self):
        rows = _balanced_rows([f"example-{index}" for index in range(6)])
        train, evaluation, report = dataset.split_balanced_dpo_rows(
            rows, train_per_task=2, eval_per_task=1, seed=20260714,
        )
        reversed_train, reversed_eval, reversed_report = dataset.split_balanced_dpo_rows(
            list(reversed(rows)), train_per_task=2, eval_per_task=1, seed=20260714,
        )

        self.assertEqual((train, evaluation, report), (reversed_train, reversed_eval, reversed_report))
        self.assertTrue(all(isinstance(row, dict) and row in rows for row in train + evaluation))
        self.assertEqual(len({row["id"] for row in train + evaluation}), len(train) + len(evaluation))
        self.assertEqual({row["metadata"]["pair_type"] for row in train}, {"query", "response"})
        self.assertEqual({row["metadata"]["pair_type"] for row in evaluation}, {"query", "response"})
        self.assertEqual(len(train), 4)
        self.assertEqual(len(evaluation), 2)
        self.assertEqual(
            {row["metadata"]["pair_type"] for row in train}, {"query", "response"},
        )
        train_examples = {row["metadata"]["example_id"] for row in train}
        eval_examples = {row["metadata"]["example_id"] for row in evaluation}
        self.assertFalse(train_examples & eval_examples)
        self.assertEqual(report["train_task_counts"], {"query": 2, "response": 2})
        self.assertEqual(report["eval_task_counts"], {"query": 1, "response": 1})
        self.assertEqual(report["example_overlap"], 0)
        self.assertEqual(report["example_overlap_count"], 0)
        self.assertEqual(report["row_id_overlap_count"], 0)
        self.assertEqual(report["train_rows_sha256"], reversed_report["train_rows_sha256"])

    def test_split_fails_fast_for_malformed_duplicate_insufficient_and_non_disjoint_inputs(self):
        valid = _balanced_rows(["a", "b", "c"])
        cases = [
            (valid[:1] + [{**valid[1], "id": valid[0]["id"]}], "unique non-empty id"),
            ([{**valid[0], "metadata": {"pair_type": "other", "example_id": "a"}}], "pair_type"),
            ([{**valid[0], "metadata": {"pair_type": "query"}}], "example_id"),
            (valid[:2], "requires 2 rows for each task"),
            (
                [
                    _preference_row("a-q", "query", "a"),
                    _preference_row("b-q", "query", "b"),
                    _preference_row("a-r1", "response", "a"),
                    _preference_row("a-r2", "response", "a"),
                ],
                "example-disjoint",
            ),
        ]
        for rows, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    dataset.split_balanced_dpo_rows(rows, train_per_task=1, eval_per_task=1, seed=7)

    def test_cli_exposes_balanced_dpo_split_contract(self):
        args = build_parser().parse_args([
            "split-balanced-dpo", "--input", "preferences.jsonl", "--train", "train.jsonl",
            "--eval", "eval.jsonl", "--report", "split.json", "--train-per-task", "32",
            "--eval-per-task", "8", "--seed", "20260714",
        ])
        self.assertEqual(args.train_per_task, 32)
        self.assertEqual(args.eval_per_task, 8)
        self.assertEqual(args.seed, 20260714)
        self.assertEqual(args.func.__name__, "cmd_split_balanced_dpo")

    def test_cli_writes_jsonl_and_hash_bound_report(self):
        from text_feedback_dpo.cli import cmd_split_balanced_dpo

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "preferences.jsonl"
            train_path = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            report_path = root / "report.json"
            rows = _balanced_rows(["a", "b", "c"])
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            args = build_parser().parse_args([
                "split-balanced-dpo", "--input", str(source), "--train", str(train_path),
                "--eval", str(eval_path), "--report", str(report_path), "--train-per-task", "1",
                "--eval-per-task", "1", "--seed", "3",
            ])
            cmd_split_balanced_dpo(args)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["train_rows"], 2)
            self.assertEqual(report["eval_rows"], 2)
            self.assertEqual(report["example_overlap_count"], 0)
            self.assertEqual(len(train_path.read_text(encoding="utf-8").splitlines()), 2)
            self.assertEqual(len(eval_path.read_text(encoding="utf-8").splitlines()), 2)
            self.assertEqual(len(report["train_file_sha256"]), 64)
            self.assertEqual(len(report["eval_file_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
