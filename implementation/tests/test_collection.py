import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from text_feedback_dpo.collection import collect_paper_shard, merge_paper_collection, paper_generation_kwargs
from text_feedback_dpo.evaluators import ModelOutputParseError
from text_feedback_dpo.experiment_config import load_paper_experiment
from text_feedback_dpo.io import read_jsonl_zst, write_json_atomic
from text_feedback_dpo.models import ModelGeneration, ModelProvider


class ScriptedProvider(ModelProvider):
    def __init__(self):
        self.prompts = []
        self.outputs = {
            "student": iter(["wrong one", "right one", "right two"]),
            "teacher": iter([
                "<student_feedback>Recheck how the quantities relate before answering fully.</student_feedback>",
            ]),
        }

    def generate(self, role, _prompt, **_kwargs):
        return next(self.outputs[role])

    def generate_result(self, role, prompt, **kwargs):
        self.prompts.append({"role": role, "prompt": prompt})
        text = self.generate(role, prompt, **kwargs)
        return ModelGeneration(
            text=text,
            prompt_tokens=7,
            generated_tokens=3,
            terminated=True,
            truncated=False,
            finish_reason="eos",
        )


class CollectionTest(unittest.TestCase):
    def test_paper_generation_kwargs_are_explicit_and_role_specific(self):
        repo = Path(__file__).resolve().parents[1]
        config = load_paper_experiment(repo / "configs" / "paper" / "math.yaml")

        student = paper_generation_kwargs(config, role="student")
        self.assertEqual(
            student,
            {
                "enable_thinking": False,
                "do_sample": True,
                "max_new_tokens": 8192,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "min_p": 0.0,
                "presence_penalty": 1.5,
                "repetition_penalty": 1.0,
                "stop_after_final_answer": True,
            },
        )
        self.assertEqual(
            paper_generation_kwargs(config, role="teacher"),
            {"enable_thinking": False, "do_sample": False, "max_new_tokens": 64},
        )
        self.assertEqual(
            paper_generation_kwargs(config, role="guidance_guard"),
            {"enable_thinking": False, "do_sample": False, "max_new_tokens": 8},
        )

    def test_collection_writes_complete_compressed_records_and_merges(self):
        repo = Path(__file__).resolve().parents[1]
        config_path = repo / "configs" / "paper" / "gsm8k.yaml"
        config = load_paper_experiment(config_path)
        examples = [
            {"id": "m1", "domain": "math", "problem": "Compute one.", "gold_answer": "4"},
            {"id": "m2", "domain": "math", "problem": "Compute two.", "gold_answer": "7"},
        ]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "data"
            dataset_dir.mkdir()
            write_json_atomic(
                dataset_dir / "manifest.json",
                {"content_sha256": hashlib.sha256(b"fixture").hexdigest()},
            )
            output_dir = root / "collection"
            provider = ScriptedProvider()

            result = collect_paper_shard(
                config=config,
                config_path=config_path,
                examples=examples,
                dataset_dir=dataset_dir,
                output_root=output_dir,
                split="train",
                shard_index=0,
                num_shards=1,
                source_commit="a" * 40,
                model_provider=provider,
                evaluator=lambda _example, response: {
                    "correct": response.startswith("right"),
                    "confidence": 0.9,
                    "reason": "fixture",
                },
                guidance_guard=lambda *_args: {"safe": True, "confidence": 0.9, "reason": "fixture"},
                guidance_critic=lambda *_args: {"valid": True, "reason": "fixture"},
            )

            self.assertEqual(result["records"], 2)
            records = read_jsonl_zst(output_dir / "shard-0000" / "records.jsonl.zst")
            self.assertEqual([record["id"] for record in records], ["m1", "m2"])
            self.assertTrue((output_dir / "shard-0000" / "complete.json").exists())
            self.assertEqual(len(records[0]["attempts"]), 2)
            self.assertNotIn("prompt", records[0]["attempts"][0])
            self.assertEqual(records[0]["attempts"][0]["generation"]["generated_tokens"], 3)
            self.assertEqual(records[0]["generation_events"][0]["finish_reason"], "eos")
            protocol = json.loads((output_dir / "shard-0000" / "protocol.json").read_text())
            self.assertEqual(protocol["source_commit"], "a" * 40)
            self.assertEqual(protocol["artifact_schema"], "paper-v3")
            self.assertEqual(protocol["feedback_policy"], "hint_only")
            self.assertIn("guidance_critic", protocol["role_generation"])
            retry_prompts = [
                item["prompt"]
                for item in provider.prompts
                if item["role"] == "student" and "General problem-solving advice" in item["prompt"]
            ]
            self.assertEqual(len(retry_prompts), 1)
            self.assertIn("General problem-solving advice", retry_prompts[0])
            self.assertNotIn("Teacher guidance", retry_prompts[0])
            self.assertNotIn("Solve again", retry_prompts[0])
            self.assertNotIn("<student_feedback>", retry_prompts[0])
            self.assertEqual(
                records[0]["guidance"][0]["parsed_student_feedback"],
                "Recheck how the quantities relate before answering fully.",
            )
            completion = json.loads((output_dir / "shard-0000" / "complete.json").read_text())
            self.assertEqual(completion["protocol_hash"], protocol["protocol_hash"])

            merged = merge_paper_collection(
                config_path=config_path,
                dataset_dir=dataset_dir,
                collection_dir=output_dir,
                expected_shards=1,
                output_path=root / "merged.jsonl.zst",
                source_commit="a" * 40,
            )
            self.assertEqual(merged["records"], 2)
            self.assertEqual(len(read_jsonl_zst(root / "merged.jsonl.zst")), 2)

    def test_model_output_failure_is_persisted_before_collection_stops(self):
        repo = Path(__file__).resolve().parents[1]
        config_path = repo / "configs" / "paper" / "gsm8k.yaml"
        config = load_paper_experiment(config_path)
        examples = [{"id": "m1", "domain": "math", "problem": "Compute.", "gold_answer": "4"}]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "data"
            dataset_dir.mkdir()
            write_json_atomic(dataset_dir / "manifest.json", {"content_sha256": "fixture"})

            def broken_evaluator(_example, _response):
                raise ModelOutputParseError(role="evaluator", raw="not-json", message="invalid JSON")

            with self.assertRaises(ModelOutputParseError):
                collect_paper_shard(
                    config=config,
                    config_path=config_path,
                    examples=examples,
                    dataset_dir=dataset_dir,
                    output_root=root / "collection",
                    split="train",
                    shard_index=0,
                    num_shards=1,
                    source_commit="a" * 40,
                    model_provider=ScriptedProvider(),
                    evaluator=broken_evaluator,
                    guidance_guard=lambda *_args: {"safe": True},
                )
            failure = json.loads((root / "collection" / "shard-0000" / "model_failures.jsonl").read_text().splitlines()[0])
            self.assertEqual(failure["raw_output"], "not-json")

            with self.assertRaisesRegex(ValueError, "protocol"):
                collect_paper_shard(
                    config=config,
                    config_path=config_path,
                    examples=examples,
                    dataset_dir=dataset_dir,
                    output_root=root / "collection",
                    split="train",
                    shard_index=0,
                    num_shards=1,
                    source_commit="b" * 40,
                    model_provider=ScriptedProvider(),
                    evaluator=broken_evaluator,
                    guidance_guard=lambda *_args: {"safe": True},
                    guidance_critic=lambda *_args: {"valid": True},
                )


if __name__ == "__main__":
    unittest.main()
