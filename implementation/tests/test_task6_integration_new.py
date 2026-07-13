import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from text_feedback_dpo.batch_generation import run_fixed_retrieval_pipeline
from text_feedback_dpo.cli import build_parser
from text_feedback_dpo.collection import collect_dataset_batchwise
from text_feedback_dpo.preferences import build_query_preference_rows, build_response_preference_rows
from text_feedback_dpo.runtime import GeneratedText
from text_feedback_dpo.trajectories import TrajectoryError, validate_active_artifact


EVALUATOR_VERSION = "cited-response-evaluator-v1"

def source_records() -> list[dict]:
    return [
        {
            "source_id": f"S{index:03d}",
            "original_rank": index,
            "title": f"Ada source {index}",
            "url": f"https://example.test/ada/{index}",
            "snippet": f"Ada Lovelace wrote the first algorithm source {index}.",
            "related_links": None,
        }
        for index in range(1, 9)
    ]


def example() -> dict:
    return {
        "id": "q1",
        "question": "Who wrote the first algorithm?",
        "gold_answer": "Ada Lovelace",
        "sources": source_records(),
    }


def active_artifact(*, response: str, query: str = "Ada algorithm", hints: tuple[str, ...] = ()) -> dict:
    return run_fixed_retrieval_pipeline(
        [example()],
        query_generate_batch=lambda _prompts: [GeneratedText(query, False)],
        response_generate_batch=lambda _prompts: [GeneratedText(response, False)],
        policy_hash="policy-v1",
        hints_by_id={"q1": list(hints)},
        evaluator_version=EVALUATOR_VERSION,
    )[0]


def correct_response() -> str:
    return "Answer: Ada Lovelace\nReasoning: The source identifies Ada Lovelace [S001].\nSources: S001"


def wrong_response() -> str:
    return "Answer: Grace Hopper\nReasoning: The source identifies another person [S001].\nSources: S001"


class Task6ArtifactValidationTest(unittest.TestCase):
    def test_requires_explicit_provenance_and_recomputes_all_context_hashes(self):
        artifact = active_artifact(response=correct_response())
        validated = validate_active_artifact(artifact, example=example(), hints=[])
        self.assertEqual(validated["provenance"], "student")

        missing = copy.deepcopy(artifact)
        missing.pop("provenance")
        with self.assertRaisesRegex(TrajectoryError, "provenance"):
            validate_active_artifact(missing, example=example(), hints=[])

        forged_prompt = copy.deepcopy(artifact)
        forged_prompt["query_prompt_hash"] = "forged"
        with self.assertRaisesRegex(TrajectoryError, "query_prompt_hash"):
            validate_active_artifact(forged_prompt, example=example(), hints=[])

        forged_retrieval = copy.deepcopy(artifact)
        forged_retrieval["ranked_search_results"][0]["corpus_hash"] = "forged"
        with self.assertRaisesRegex(TrajectoryError, "ranked retrieval"):
            validate_active_artifact(forged_retrieval, example=example(), hints=[])

    def test_rejects_xml_or_angle_markup_even_when_caller_claims_success(self):
        artifact = active_artifact(response=correct_response())
        artifact["raw_response"] = "Answer: Ada Lovelace\nReasoning: <claim> [S001].\nSources: S001"
        with self.assertRaisesRegex(TrajectoryError, "angle markup"):
            validate_active_artifact(artifact, example=example(), hints=[])

    def test_recomputes_parse_render_error_fields_and_query_stage_nulls(self):
        artifact = active_artifact(response=correct_response())
        for field, forged in (
            ("parsed_response", {"answer": "forged", "reasoning": "forged", "source_ids": ["S001"]}),
            ("rendered_visible_response", "forged"),
            ("error_code", "forged_error"),
        ):
            with self.subTest(field=field):
                changed = copy.deepcopy(artifact)
                changed[field] = forged
                with self.assertRaisesRegex(TrajectoryError, field):
                    validate_active_artifact(changed, example=example(), hints=[])

        query_failure = active_artifact(response=correct_response(), query="Ada <xml")
        for field, forged in (
            ("parsed_response", {"answer": "forged"}),
            ("rendered_visible_response", "forged"),
            ("error_code", None),
        ):
            with self.subTest(query_failure_field=field):
                changed = copy.deepcopy(query_failure)
                changed[field] = forged
                with self.assertRaisesRegex(TrajectoryError, field):
                    validate_active_artifact(changed, example=example(), hints=[])

        forged_query_score = copy.deepcopy(query_failure)
        forged_query_score["cited_score"]["correct"] = True
        with self.assertRaisesRegex(TrajectoryError, "cited_score"):
            validate_active_artifact(forged_query_score, example=example(), hints=[])

        forged_response_score = copy.deepcopy(artifact)
        forged_response_score["cited_score"]["citation_recall"] = 0.25
        with self.assertRaisesRegex(TrajectoryError, "cited_score"):
            validate_active_artifact(forged_response_score, example=example(), hints=[])


class Task6PreferenceContextTest(unittest.TestCase):
    @staticmethod
    def _candidate(artifact: dict, *, gain: float, verified: bool, seed: int) -> dict:
        return {
            **artifact,
            "future_sibling_gain": gain,
            "verified_no_hint_success": verified,
            "seed": seed,
        }

    def test_response_pairs_require_identical_prompt_bytes_and_canonical_retrieval(self):
        chosen = self._candidate(active_artifact(response=correct_response()), gain=1.0, verified=True, seed=101)
        rejected = self._candidate(active_artifact(response=wrong_response()), gain=0.0, verified=False, seed=102)
        trajectory = {
            "id": "q1", "training_eligible": True, "query_prompt": chosen["query_prompt"],
            "query_prompt_hash": chosen["query_prompt_hash"], "policy_hash": "policy-v1",
            "no_hint_siblings": [chosen, rejected],
        }
        rows = build_response_preference_rows(trajectory)
        self.assertEqual(rows[0]["metadata"]["canonical_context"]["response_prompt"], chosen["response_prompt"])
        self.assertEqual(rows[0]["metadata"]["canonical_context"]["ranked_search_results"], chosen["canonical_ranked_search_results"])

        changed_prompt = copy.deepcopy(rejected)
        changed_prompt["response_prompt"] += " "
        with self.assertRaisesRegex(ValueError, "prompt bytes"):
            build_response_preference_rows({**trajectory, "no_hint_siblings": [chosen, changed_prompt]})

        changed_records = copy.deepcopy(rejected)
        changed_records["canonical_ranked_search_results"][0]["title"] = "Different bytes"
        with self.assertRaisesRegex(ValueError, "retrieval records"):
            build_response_preference_rows({**trajectory, "no_hint_siblings": [chosen, changed_records]})

    def test_query_pairs_require_identical_no_hint_prompt_bytes(self):
        chosen = self._candidate(active_artifact(response=correct_response(), query="Ada algorithm"), gain=1.0, verified=True, seed=101)
        rejected = self._candidate(active_artifact(response=wrong_response(), query="algorithm author"), gain=0.0, verified=False, seed=102)
        trajectory = {
            "id": "q1", "training_eligible": True, "query_prompt": chosen["query_prompt"],
            "query_prompt_hash": chosen["query_prompt_hash"], "policy_hash": "policy-v1",
            "no_hint_siblings": [chosen, rejected],
        }
        rows = build_query_preference_rows(trajectory)
        self.assertEqual(rows[0]["metadata"]["canonical_context"]["query_prompt"], chosen["query_prompt"])
        rejected["query_prompt"] += " "
        with self.assertRaisesRegex(ValueError, "query prompt bytes"):
            build_query_preference_rows(trajectory)

    def test_response_pairs_reject_success_without_verified_sibling_status(self):
        chosen = self._candidate(active_artifact(response=correct_response()), gain=1.0, verified=True, seed=101)
        unverified = self._candidate(active_artifact(response=correct_response()), gain=0.0, verified=False, seed=102)
        trajectory = {
            "id": "q1", "training_eligible": True, "query_prompt": chosen["query_prompt"],
            "query_prompt_hash": chosen["query_prompt_hash"], "policy_hash": "policy-v1",
            "no_hint_siblings": [chosen, unverified],
        }
        with self.assertRaisesRegex(ValueError, "unverified success"):
            build_response_preference_rows(trajectory)

    def test_equal_gain_identical_completion_is_counted_before_gain_filter(self):
        chosen = self._candidate(active_artifact(response=correct_response()), gain=1.0, verified=True, seed=101)
        duplicate = self._candidate(active_artifact(response=correct_response()), gain=1.0, verified=True, seed=102)
        rejected = self._candidate(active_artifact(response=wrong_response()), gain=0.0, verified=False, seed=103)
        trajectory = {
            "id": "q1", "training_eligible": True, "preference_eligible": True,
            "query_prompt": chosen["query_prompt"], "query_prompt_hash": chosen["query_prompt_hash"],
            "policy_hash": "policy-v1", "no_hint_siblings": [chosen, duplicate, rejected],
            "preference_exclusions": [], "preference_exclusion_counts": {},
        }
        rows = build_response_preference_rows(trajectory)
        self.assertEqual(len(rows), 1)
        self.assertEqual(trajectory["preference_exclusion_counts"], {"identical_response_completion": 1})
        self.assertEqual(trajectory["preference_exclusions"][0]["rejected_seed"], 102)


class Task6CollectionAndCliTest(unittest.TestCase):
    def test_build_preferences_requires_dataset_and_revalidates_siblings(self):
        def student(requests, **_kwargs):
            return [active_artifact(
                response=correct_response() if request["hints"] else wrong_response(),
                hints=tuple(request["hints"]),
            ) for request in requests]

        def siblings(requests, **_kwargs):
            return [active_artifact(
                response=correct_response() if request["seed"] == 101 else wrong_response()
            ) for request in requests]

        trajectory = collect_dataset_batchwise(
            examples=[example()], student_generate_batch=student,
            teacher_generate_batch=lambda _prompts, **_kwargs: ['{"hint":"Focus on the associated person."}'],
            max_interventions=1, sibling_generate_batch=siblings,
            sibling_seeds=(101, 102), student_seed=7,
        )[0]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.jsonl"
            trajectories = root / "trajectories.jsonl"
            output = root / "preferences.jsonl"
            data.write_text(json.dumps(example()) + "\n", encoding="utf-8")
            trajectories.write_text(json.dumps(trajectory) + "\n", encoding="utf-8")
            args = build_parser().parse_args([
                "build-preferences", "--data", str(data),
                "--trajectories", str(trajectories), "--output", str(output),
            ])
            args.func(args)
            self.assertTrue(output.read_text(encoding="utf-8").strip())

            forged = copy.deepcopy(trajectory)
            forged["no_hint_siblings"][1]["raw_response"] = correct_response()
            forged["no_hint_siblings"][1]["verified_no_hint_success"] = True
            forged["no_hint_siblings"][1]["future_sibling_gain"] = 1.0
            trajectories.write_text(json.dumps(forged) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cited_score|parsed_response"):
                args.func(args)

            forged = copy.deepcopy(trajectory)
            forged["no_hint_siblings"][1]["ranked_search_results"][0]["title"] = "forged"
            forged["no_hint_siblings"][1]["canonical_ranked_search_results"] = copy.deepcopy(
                forged["no_hint_siblings"][1]["ranked_search_results"]
            )
            with self.assertRaisesRegex(ValueError, "ranked retrieval"):
                trajectories.write_text(json.dumps(forged) + "\n", encoding="utf-8")
                args.func(args)

            forged = copy.deepcopy(trajectory)
            forged["sibling_verification"]["success_count"] = 2
            trajectories.write_text(json.dumps(forged) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sibling_verification"):
                args.func(args)

    def test_hinted_success_without_sibling_generator_is_explicitly_ineligible(self):
        calls = []

        def student(requests, **_kwargs):
            calls.append(requests)
            hints = tuple(requests[0]["hints"])
            return [active_artifact(response=correct_response() if hints else wrong_response(), hints=hints)]

        rows = collect_dataset_batchwise(
            examples=[example()], student_generate_batch=student,
            teacher_generate_batch=lambda _prompts, **_kwargs: ['{"hint":"Focus on the associated person."}'],
            max_interventions=1, sibling_generate_batch=None, sibling_seeds=(101, 102), student_seed=7,
        )
        self.assertTrue(rows[0]["resolved"])
        self.assertFalse(rows[0]["training_eligible"])
        self.assertEqual(rows[0]["sibling_verification"]["status"], "missing_sibling_generator")
        self.assertEqual(rows[0]["ranked_interventions"], [])

    def test_cmd_collect_runs_active_pipeline_siblings_and_complete_manifest(self):
        class Tokenizer:
            def apply_chat_template(self, _messages, **_kwargs):
                return "teacher prompt"

        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.jsonl"
            output = root / "trajectories.jsonl"
            cache = root / "cache.jsonl"
            second = copy.deepcopy(example())
            second["id"] = "q2"
            data.write_text(json.dumps(example()) + "\n" + json.dumps(second) + "\n", encoding="utf-8")
            args = build_parser().parse_args([
                "collect", "--data", str(data), "--output", str(output),
                "--student-model", "Qwen/Qwen3-4B-Base", "--student-revision", "student-rev",
                "--teacher-model", "Qwen/Qwen3-32B", "--teacher-revision", "teacher-rev",
                "--dataset-revision", "data-rev", "--prompt-version", "fixed-retrieval-cited-v1",
                "--policy-version", "sft-v1", "--policy-hash", "a" * 64, "--seed", "7",
                "--sibling-count", "2", "--sibling-seeds", "101", "102",
                "--teacher-quantization", "4bit", "--attention-implementation", "sdpa",
                "--student-device", "cuda:1", "--teacher-device", "cuda:0",
                "--trajectory-cache", str(cache), "--max-interventions", "1",
            ])
            current_seed = {"value": None}
            teacher_calls = []
            student_stage_calls = []

            def set_seed(seed):
                current_seed["value"] = seed

            def student_records(_model, _tokenizer, prompts, **_kwargs):
                student_stage_calls.append((current_seed["value"], len(prompts), prompts[0].endswith("Search query:")))
                if prompts[0].endswith("Search query:"):
                    return [GeneratedText("Ada algorithm", False) for _ in prompts]
                success = "Hints:" in prompts[0] or current_seed["value"] == 101
                response = correct_response() if success else wrong_response()
                return [GeneratedText(response, False) for _ in prompts]

            def teacher_generate(_model, _tokenizer, prompts, **kwargs):
                teacher_calls.append((prompts, kwargs))
                if kwargs["max_new_tokens"] == 1:
                    return ["probe" for _ in prompts]
                return ['{"hint":"Focus on the associated person."}' for _ in prompts]

            with (
                patch("text_feedback_dpo.runtime.load_tokenizer", return_value=Tokenizer()),
                patch("text_feedback_dpo.runtime.load_student", return_value=object()),
                patch("text_feedback_dpo.runtime.load_teacher", return_value=object()),
                patch("text_feedback_dpo.runtime.generate_batch_records", side_effect=student_records),
                patch("text_feedback_dpo.runtime.generate_batch", side_effect=teacher_generate),
                patch("text_feedback_dpo.runtime.set_generation_seed", side_effect=set_seed),
            ):
                args.func(args)

            output_rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["id"] for row in output_rows], ["q1", "q2"])
            row = output_rows[0]
            self.assertTrue(row["training_eligible"])
            self.assertTrue(row["sft_eligible"])
            self.assertTrue(row["preference_eligible"])
            self.assertEqual(row["sibling_verification"]["seeds"], [101, 102])
            self.assertEqual(row["sibling_verification"]["sibling_count"], 2)
            self.assertEqual(len(row["ranked_interventions"]), 1)
            self.assertTrue(any(item["metadata"]["pair_type"] == "response" for item in row["preference_rows"]))
            self.assertEqual(row["preference_exclusion_counts"], {"identical_query_completion": 1})
            self.assertEqual(row["preference_exclusions"][0]["reason"], "identical_query_completion")
            self.assertEqual(len(teacher_calls), 1)
            self.assertEqual(row["attempts"][0]["artifact"]["provenance"], "student")
            self.assertTrue(row["attempts"][0]["artifact"]["ranked_search_results"])
            manifest = json.loads(cache.with_suffix(".manifest.json").read_text(encoding="utf-8"))
            for key in (
                "dataset_schema", "source_schema_hash", "retrieval_config", "retrieval_hash",
                "response_schema_version", "evaluator_version", "policy_version", "sibling_seeds",
            ):
                self.assertIn(key, manifest)
            self.assertEqual(manifest["sibling_seeds"], [101, 102])
            sibling_query_calls = [call for call in student_stage_calls if call[0] in {101, 102} and call[2]]
            self.assertEqual(sibling_query_calls, [(101, 2, True), (102, 2, True)])


if __name__ == "__main__":
    unittest.main()
