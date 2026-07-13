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
            response="<response><answer>Ada Lovelace</answer><evidence>Ada Lovelace wrote the first algorithm.</evidence></response>",
            gold_answer="Ada Lovelace",
            packed_evidence="Ada Lovelace wrote the first algorithm.",
        )
        self.assertEqual(result["exact_match"], 1.0)
        self.assertEqual(result["f1"], 1.0)
        self.assertEqual(result["evidence_support"], 1.0)
        self.assertTrue(result["correct"])

    def test_searchqa_scoring_does_not_accept_unrelated_answer(self):
        result = score_searchqa(
            response="<response><answer>Grace Hopper</answer><evidence>No supporting evidence.</evidence></response>",
            gold_answer="Ada Lovelace",
            packed_evidence="Ada Lovelace wrote the first algorithm.",
        )
        self.assertEqual(result["exact_match"], 0.0)
        self.assertEqual(result["f1"], 0.0)
        self.assertFalse(result["correct"])

    def test_trajectory_stops_at_first_correct_without_teacher_written_answer(self):
        outputs = iter([
            "<response><answer>Grace Hopper</answer><evidence>Unsupported.</evidence></response>",
            "<response><answer>Ada Lovelace</answer><evidence>Ada Lovelace wrote the first algorithm.</evidence></response>",
        ])

        def student(_prompt, _attempt):
            return next(outputs)

        def teacher(_request):
            return "<feedback><error_span>Grace Hopper</error_span><hint>Recheck the person associated with the algorithm.</hint><scope>entity</scope></feedback>"

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
        self.assertIn("<answer>Ada Lovelace</answer>", trajectory["chosen"])
        self.assertEqual(len(trajectory["interventions"]), 1)
        self.assertNotIn("Ada Lovelace", trajectory["interventions"][0]["hint"])

    def test_preference_builder_excludes_hints_from_prompt_and_keeps_all_failures(self):
        trajectory = {
            "id": "sq-1",
            "resolved": True,
            "prompt": "Question: Who wrote the first algorithm?\nEvidence: snippet",
            "chosen": "<response><answer>Ada Lovelace</answer><evidence>Supported.</evidence></response>",
            "attempts": [
                {"response": "<response><answer>Grace Hopper</answer><evidence>Unsupported.</evidence></response>", "correct": False, "attempt_index": 0},
                {"response": "<response><answer>Charles Babbage</answer><evidence>Unsupported.</evidence></response>", "correct": False, "attempt_index": 1},
                {"response": "<response><answer>Ada Lovelace</answer><evidence>Supported.</evidence></response>", "correct": True, "attempt_index": 2},
            ],
            "interventions": [{"hint": "Recheck the person associated with the algorithm."}],
        }
        rows = build_preference_rows(trajectory)
        self.assertEqual(len(rows), 2)
        self.assertIn("<answer>Grace Hopper</answer>", rows[0]["rejected"])
        self.assertIn("<answer>Charles Babbage</answer>", rows[1]["rejected"])
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

    def test_prompts_escape_user_content_and_use_only_current_xml_contract(self):
        prompt = build_student_prompt(
            {"question": "Who <wrote> it?", "packed_evidence": "A & B", "gold_answer": "Ada"},
            [],
        )
        self.assertIn("Who &lt;wrote&gt; it?", prompt)
        self.assertIn("A &amp; B", prompt)
        self.assertIn("<response>", prompt)
        self.assertNotIn("<teacher_task>", prompt)


if __name__ == "__main__":
    unittest.main()
