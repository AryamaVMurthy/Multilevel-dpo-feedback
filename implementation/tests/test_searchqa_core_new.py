import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from text_feedback_dpo.config import load_config
from text_feedback_dpo.preferences import build_preference_rows
from text_feedback_dpo.prompts import build_student_prompt
from text_feedback_dpo.scoring import score_searchqa
from text_feedback_dpo.trajectories import collect_trajectory


class SearchQACoreContractTest(unittest.TestCase):
    def test_searchqa_scoring_reports_exact_match_f1_and_evidence_support(self):
        result = score_searchqa(
            response="Ada Lovelace",
            gold_answer="Ada Lovelace",
            packed_evidence="Ada Lovelace wrote the first algorithm.",
        )
        self.assertEqual(result["exact_match"], 1.0)
        self.assertEqual(result["f1"], 1.0)
        self.assertEqual(result["evidence_support"], 1.0)
        self.assertTrue(result["correct"])

    def test_searchqa_scoring_does_not_accept_unrelated_answer(self):
        result = score_searchqa(
            response="Grace Hopper",
            gold_answer="Ada Lovelace",
            packed_evidence="Ada Lovelace wrote the first algorithm.",
        )
        self.assertEqual(result["exact_match"], 0.0)
        self.assertEqual(result["f1"], 0.0)
        self.assertFalse(result["correct"])

    def test_trajectory_stops_at_first_correct_without_teacher_written_answer(self):
        outputs = iter([
            "Grace Hopper",
            "Ada Lovelace",
        ])

        def student(_prompt, _attempt):
            return next(outputs)

        def teacher(_request):
            return '{"hint":"Recheck the person associated with the algorithm."}'

        trajectory = collect_trajectory(
            example={
                "id": "sq-1",
                "question": "Who wrote the first algorithm?",
                "gold_answer": "Ada Lovelace",
                "packed_evidence": "Ada Lovelace wrote the first algorithm.",
            },
            student_generate=student,
            teacher_generate=teacher,
            max_interventions=4,
        )
        self.assertTrue(trajectory["resolved"])
        self.assertEqual(len(trajectory["attempts"]), 2)
        self.assertEqual(trajectory["chosen"], "Ada Lovelace")
        self.assertEqual(len(trajectory["interventions"]), 1)
        self.assertNotIn("Ada Lovelace", trajectory["interventions"][0]["hint"])
        self.assertEqual(trajectory["interventions"][0]["level"], 1)

    def test_preference_builder_excludes_hints_from_prompt_and_keeps_all_failures(self):
        trajectory = {
            "id": "sq-1",
            "resolved": True,
            "prompt": "Question: Who wrote the first algorithm?\nEvidence: snippet",
            "chosen": "Ada Lovelace",
            "attempts": [
                {"response": "Grace Hopper", "correct": False, "attempt_index": 0},
                {"response": "Charles Babbage", "correct": False, "attempt_index": 1},
                {"response": "Ada Lovelace", "correct": True, "attempt_index": 2},
            ],
            "interventions": [{"hint": "Recheck the person associated with the algorithm."}],
        }
        rows = build_preference_rows(trajectory)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["rejected"], "Grace Hopper")
        self.assertEqual(rows[1]["rejected"], "Charles Babbage")
        for row in rows:
            self.assertNotIn("Recheck", row["prompt"])
            self.assertNotIn("Ada Lovelace", row["prompt"])

    def test_config_requires_searchqa_and_full_finetuning(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(
                json.dumps(
                    {
                        "run_id": "searchqa-test",
                        "student_model": "Qwen/Qwen3-4B-Base",
                        "teacher_model": "Qwen/Qwen3-14B",
                        "student_revision": "student-rev",
                        "teacher_revision": "teacher-rev",
                        "dataset": {"name": "searchqa", "source": "kyunghyuncho/search_qa", "revision": "data-rev", "max_length": 4096},
                        "training": {"full_finetuning": True, "method": "dpo"},
                        "slurm": {"partition": "u22", "gpus": 2},
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config["dataset"]["name"], "searchqa")
            self.assertTrue(config["training"]["full_finetuning"])

    def test_prompt_uses_plain_answer_contract_without_markup(self):
        prompt = build_student_prompt(
            {"question": "Who <wrote> it?", "packed_evidence": "A & B", "gold_answer": "Ada"},
            [],
        )
        self.assertIn("Who <wrote> it?", prompt)
        self.assertIn("A & B", prompt)
        self.assertTrue(prompt.endswith("Answer:"))
        self.assertIn("Think through the evidence", prompt)
        self.assertIn("plain text only", prompt)
        self.assertIn("Do not use XML", prompt)
        self.assertIn("at most 8 words", prompt)
        self.assertNotIn("<response>", prompt)
        self.assertNotIn("<student_task>", prompt)


if __name__ == "__main__":
    unittest.main()
