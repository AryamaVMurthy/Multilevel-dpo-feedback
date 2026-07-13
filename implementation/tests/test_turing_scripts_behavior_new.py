import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PROBE_RUNNER = SCRIPTS / "turing_probe_runner.py"


def valid_probe_result(*, throughput: float = 10.0, compile_: bool = False, gpu_count: int = 1) -> dict:
    token_ids = [[1, 2, 3]]
    decoded_outputs = ["decoded"]
    return {
        "schema_version": 1,
        "probe_name": "probe",
        "status": "ok",
        "fallback_reason": "none",
        "output_hash": hashlib.sha256(json.dumps(token_ids, separators=(",", ":"), sort_keys=True).encode()).hexdigest(),
        "decoded_output_hash": hashlib.sha256(json.dumps(decoded_outputs, separators=(",", ":"), sort_keys=True).encode()).hexdigest(),
        "output_token_ids": token_ids,
        "decoded_outputs": decoded_outputs,
        "examples_per_second": throughput / 2,
        "tokens_per_second": throughput,
        "peak_gpu_memory_mb": 1024.0,
        "gpu_utilization": {
            "sample_count": 30,
            "monitor_interval_seconds": 0.2,
            "measured_duration_seconds": 6.0,
            "coverage_ratio": 1.0,
            "monitored_uuids": [f"GPU-{index:032x}" for index in range(gpu_count)],
            "query_errors": [],
            "utilization_mean_percent": 70.0,
            "utilization_peak_percent": 90.0,
            "nvidia_smi_peak_memory_mb": 1100.0,
            "power_peak_watts": 200.0,
            "temperature_peak_c": 70.0,
        },
        "gpu_hardware": {
            "count": gpu_count,
            "devices": [{
                "index": index,
                "name": "nvidia a100-sxm4-80gb",
                "uuid": f"GPU-{index:032x}",
                "total_memory_bytes": 85899345920,
                "free_memory_bytes": 81604378624,
                "compute_capability": "8.0",
            } for index in range(gpu_count)],
        },
        "package_versions": {
            "torch": "2.13.0+cu126",
            "transformers": "5.13.0",
            "trl": "1.8.0",
            "deepspeed": "0.19.2",
            "bitsandbytes": "0.49.2",
            "flash-attn": "missing",
            "liger-kernel": "missing",
        },
        "config": {
            "probe_kind": "generation",
            "attention_implementation": "sdpa",
            "generation_batch_size": 4,
            "static_cache": False,
            "compile": compile_,
            "train_microbatch": 1,
            "gradient_accumulation_steps": 32,
            "dataloader_workers": 0,
            "packing": False,
            "padding_free": False,
            "use_liger_kernel": False,
            "per_device_eval_batch_size": 1,
            "max_steps": 4,
            "max_length": 4096,
            "gradient_checkpointing": True,
            "num_generations": 8,
            "rl_generation_batch_size": 8,
            "max_completion_length": 512,
            "training_method": "sft",
        },
        "identities": {
            "commit_hash": "commit",
            "config_sha256": "config",
            "model": "model",
            "model_revision": "revision",
            "dataset_source": "source",
            "dataset_revision": "dataset-revision",
            "dataset_sha256": "dataset",
            "prompt_sha256": "prompt",
            "retrieval_sha256": "retrieval",
            "source_schema_sha256": "schema",
        },
    }


def valid_training_probe(*, throughput: float, gpu_count: int, loss: float = 1.0) -> dict:
    result = valid_probe_result(throughput=throughput, gpu_count=gpu_count)
    result["config"]["probe_kind"] = "training"
    result["finite_metrics"] = {"train_loss": loss, "eval_loss": loss + 0.1}
    result["global_examples_per_second"] = throughput / 2
    result["global_tokens_per_second"] = throughput
    result["correctness_hash"] = "c" * 64
    return result


def freeze_probe(tmp_path: Path, baseline_data: dict, candidate_data: dict | None = None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(baseline_data), encoding="utf-8")
    command = [
        sys.executable, str(PROBE_RUNNER), "freeze-decision", "--baseline", str(baseline),
        "--output", str(tmp_path / "decision.json"), "--launch-max-steps", "100", "--launch-learning-rate", "1e-6",
        "--launch-epochs", "1", "--launch-save-steps", "10", "--launch-eval-steps", "10", "--query-max-new-tokens", "32",
        "--response-max-new-tokens", "256", "--student-thinking-mode", "direct",
        "--scratchpad-max-new-tokens", "128", "--query-temperature", "0",
        "--response-temperature", "0", "--top-p", "1", "--top-k", "8",
        "--k1", "1.2", "--b", "0.75",
    ]
    if candidate_data is not None:
        candidate = tmp_path / "candidate.json"
        candidate.write_text(json.dumps(candidate_data), encoding="utf-8")
        command.extend(("--candidate", str(candidate)))
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)


def write_realistic_checkpoint(path: Path, step: int) -> None:
    path.mkdir(parents=True)
    (path / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "max_steps": step + 1, "log_history": [{"step": step, "loss": 1.0}]}),
        encoding="utf-8",
    )
    (path / "config.json").write_text(json.dumps({
        "model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"], "hidden_size": 8,
        "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 2,
        "head_dim": 4, "intermediate_size": 16, "vocab_size": 16,
    }), encoding="utf-8")
    torch.save({
        "model.embed_tokens.weight": torch.ones(16, 8),
        "model.layers.0.self_attn.q_proj.weight": torch.ones(8, 8),
        "model.layers.0.self_attn.k_proj.weight": torch.ones(8, 8),
        "model.layers.0.self_attn.v_proj.weight": torch.ones(8, 8),
        "model.layers.0.self_attn.o_proj.weight": torch.ones(8, 8),
        "model.layers.0.mlp.gate_proj.weight": torch.ones(16, 8),
        "model.layers.0.mlp.up_proj.weight": torch.ones(16, 8),
        "model.layers.0.mlp.down_proj.weight": torch.ones(8, 16),
        "model.layers.1.self_attn.q_proj.weight": torch.ones(8, 8),
        "model.layers.1.self_attn.k_proj.weight": torch.ones(8, 8),
        "model.layers.1.self_attn.v_proj.weight": torch.ones(8, 8),
        "model.layers.1.self_attn.o_proj.weight": torch.ones(8, 8),
        "model.layers.1.mlp.gate_proj.weight": torch.ones(16, 8),
        "model.layers.1.mlp.up_proj.weight": torch.ones(16, 8),
        "model.layers.1.mlp.down_proj.weight": torch.ones(8, 16),
        "model.norm.weight": torch.ones(8),
        "lm_head.weight": torch.ones(16, 8),
    }, path / "pytorch_model.bin")
    torch.save({
        "state": {0: {"step": torch.tensor(step), "exp_avg": torch.ones(4), "exp_avg_sq": torch.ones(4)}},
        "param_groups": [{"params": [0], "lr": 1e-6, "step": step}],
    }, path / "optimizer.pt")
    torch.save({"last_epoch": step, "_step_count": step + 1}, path / "scheduler.pt")
    torch.save({
        "python": random.getstate(), "numpy": np.random.get_state(), "cpu": torch.arange(16, dtype=torch.uint8),
        "cuda": [torch.arange(16, dtype=torch.uint8)],
    }, path / "rng_state.pth")


def write_identity_decision(path: Path, identities: dict[str, str], training_method: str = "sft") -> str:
    write_decision(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["identities"] = identities
    payload["selected"]["training"]["method"] = training_method
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_scale_decision(path: Path, identities: dict[str, str], train_gpus: int = 4, training_method: str = "sft") -> str:
    measured = valid_training_probe(throughput=100.0, gpu_count=train_gpus)
    measured["config"]["training_method"] = training_method
    payload = {
        "schema_version": 1, "status": "frozen", "decision_kind": "gpu_scaling", "fallback_reason": "none",
        "compared_gpu_counts": [4, 8], "selected_train_gpus": train_gpus, "identities": identities,
        "package_versions": measured["package_versions"], "hardware_profile": {
            "name": "nvidia a100-sxm4-80gb", "total_memory_bytes": 85899345920, "compute_capability": "8.0",
        },
        "selected_gpu_hardware": measured["gpu_hardware"], "selected_global_examples_per_second": 50.0,
        "selected_global_tokens_per_second": 100.0, "training_controls": measured["config"],
        "results": [{"gpu_count": 4}, {"gpu_count": 8}],
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_smoke_manifest(tmp_path: Path, name: str, identities: dict[str, str], decision_sha: str, method: str, scale_sha: str = "scale") -> tuple[Path, str]:
    initial = tmp_path / f"{name}-initial" / "checkpoint-1"
    resumed = tmp_path / f"{name}-resumed" / "checkpoint-2"
    write_realistic_checkpoint(initial, 1)
    write_realistic_checkpoint(resumed, 2)
    manifest = tmp_path / f"{name}-smoke.json"
    result = subprocess.run(
        [sys.executable, str(PROBE_RUNNER), "create-smoke-manifest", "--initial-checkpoint", str(initial),
         "--resumed-checkpoint", str(resumed), "--output", str(manifest), "--commit-hash", identities["commit_hash"],
         "--config-sha256", identities["config_sha256"], "--model", identities["model"],
         "--model-revision", identities["model_revision"], "--dataset-source", identities["dataset_source"],
         "--dataset-revision", identities["dataset_revision"], "--dataset-sha256", identities["dataset_sha256"],
         "--eval-dataset-sha256", identities.get("eval_dataset_sha256", "not-applicable"),
         "--prompt-sha256", identities["prompt_sha256"], "--retrieval-sha256", identities["retrieval_sha256"],
         "--source-schema-sha256", identities["source_schema_sha256"], "--optimization-decision-sha256", decision_sha,
         "--scale-decision-sha256", scale_sha,
         "--method", method],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    return manifest, hashlib.sha256(manifest.read_bytes()).hexdigest()


def run_script(name: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPTS / name)],
        cwd=ROOT,
        env={**os.environ, **env},
        text=True,
        capture_output=True,
        check=False,
    )


def required_generate_env(tmp_path: Path) -> dict[str, str]:
    data = tmp_path / "shard.jsonl"
    data.write_text('{"id":"one","question":"q","search_results":[]}\n', encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("training: {}\n", encoding="utf-8")
    return {
        "TURING_ACCOUNT": "test",
        "PROJECT_DIR": str(ROOT),
        "DATA": str(data),
        "OUTPUT": str(tmp_path / "predictions.jsonl"),
        "MODEL": "model",
        "MODEL_REVISION": "revision",
        "DATASET_SOURCE": "kyunghyuncho/search_qa",
        "DATASET_REVISION": "dataset-revision",
        "CONFIG": str(config),
        "PROMPT_VERSION": "fixed-retrieval-cited-v1",
        "PROMPT_HASH": "prompt",
        "RETRIEVAL_HASH": "retrieval",
        "SOURCE_SCHEMA_HASH": "schema",
        "POLICY_HASH": "policy",
        "STUDENT_THINKING_MODE": "direct",
        "SCRATCHPAD_MAX_NEW_TOKENS": "128",
        "QUERY_TEMPERATURE": "0.0",
        "RESPONSE_TEMPERATURE": "0.0",
        "TOP_P": "1.0",
        "TOP_K": "8",
        "BM25_K1": "1.2",
        "BM25_B": "0.75",
        "SHARD_INDEX": "0",
        "SHARD_COUNT": "1",
        "SHARD_INPUT_SHA256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "MERGE_ID": "eval-shards-v1",
        "SLURM_JOB_ID": "123",
        "SLURM_NNODES": "1",
        "SLURM_NTASKS": "1",
        "SLURM_GPUS_ON_NODE": "1",
    }


def write_decision(path: Path) -> str:
    decision = {
        "schema_version": 1,
        "status": "frozen",
        "fallback_reason": "sdpa_baseline_selected",
        "selected": {
            "attention_implementation": "sdpa",
            "generation": {
                "query_batch_size": 4,
                "response_batch_size": 4,
                "query_max_new_tokens": 32,
                "response_max_new_tokens": 256,
                "static_cache": False,
                "compile": False,
                "student_thinking_mode": "direct",
                "scratchpad_max_new_tokens": 128,
                "query_temperature": 0.0,
                "response_temperature": 0.0,
                "top_p": 1.0,
                "top_k": 8,
                "k1": 1.2,
                "b": 0.75,
            },
            "training": {
                "microbatch": 1,
                "gradient_accumulation_steps": 32,
                "dataloader_workers": 0,
                "per_device_eval_batch_size": 1,
                "max_steps": 4,
                "max_length": 4096,
                "gradient_checkpointing": True,
                "learning_rate": 1e-6,
                "epochs": 1.0,
                "save_steps": 10,
                "eval_steps": 10,
                "num_generations": 8,
                "generation_batch_size": 8,
                "max_completion_length": 512,
                "method": "sft",
                "packing": False,
                "padding_free": False,
                "use_liger_kernel": False,
            },
        },
        "identities": {
            "commit_hash": "commit",
            "config_sha256": "config",
            "model": "model",
            "model_revision": "revision",
            "dataset_source": "kyunghyuncho/search_qa",
            "dataset_revision": "dataset-revision",
            "dataset_sha256": "dataset",
            "prompt_sha256": "prompt",
            "retrieval_sha256": "retrieval",
            "source_schema_sha256": "schema",
        },
        "baseline_result_sha256": "a" * 64,
        "selected_result_sha256": "a" * 64,
        "package_versions": valid_probe_result()["package_versions"],
        "gpu_hardware": valid_probe_result()["gpu_hardware"],
    }
    path.write_text(json.dumps(decision, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generation_rejects_two_nodes_before_launch(tmp_path: Path):
    env = required_generate_env(tmp_path)
    env["SLURM_NNODES"] = "2"
    result = run_script("turing_generate.sh", env)
    assert result.returncode == 2
    assert "requires exactly one node" in result.stderr
    assert "multi_node_generation_forbidden" in result.stderr


def test_generation_rejects_non_exact_single_gpu(tmp_path: Path):
    env = required_generate_env(tmp_path)
    env["SLURM_GPUS_ON_NODE"] = "2"
    result = run_script("turing_generate.sh", env)
    assert result.returncode == 2
    assert "requires exactly one allocated GPU" in result.stderr


def test_generation_rejects_shard_input_identity_mismatch(tmp_path: Path):
    env = required_generate_env(tmp_path)
    env["SHARD_INPUT_SHA256"] = "0" * 64
    result = run_script("turing_generate.sh", env)
    assert result.returncode == 2
    assert "SHARD_INPUT_SHA256" in result.stderr
    assert "shard_input_identity_mismatch" in result.stderr


def test_probe_runner_validates_frozen_decision_hash_and_emits_generation_contract(tmp_path: Path):
    decision = tmp_path / "optimization-decision.json"
    digest = write_decision(decision)
    result = subprocess.run(
        [
            sys.executable,
            str(PROBE_RUNNER),
            "validate-decision",
            "--decision",
            str(decision),
            "--expected-sha256",
            digest,
            "--purpose",
            "generation",
            "--commit-hash", "commit",
            "--config-sha256", "config",
            "--model", "model",
            "--model-revision", "revision",
            "--dataset-source", "kyunghyuncho/search_qa",
            "--dataset-revision", "dataset-revision",
            "--dataset-sha256", "dataset",
            "--prompt-sha256", "prompt",
            "--retrieval-sha256", "retrieval",
            "--source-schema-sha256", "schema",
            "--student-thinking-mode", "direct", "--scratchpad-max-new-tokens", "128",
            "--query-temperature", "0", "--response-temperature", "0", "--top-p", "1",
            "--top-k", "8", "--k1", "1.2", "--b", "0.75",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    contract = json.loads(result.stdout)
    assert contract["status"] == "validated"
    assert contract["decision_sha256"] == digest
    assert contract["attention_implementation"] == "sdpa"
    assert contract["query_batch_size"] == 4
    assert contract["fallback_reason"] == "sdpa_baseline_selected"


def test_probe_runner_rejects_cross_model_or_dataset_decision_reuse(tmp_path: Path):
    decision = tmp_path / "optimization-decision.json"
    digest = write_decision(decision)
    common = [
        sys.executable, str(PROBE_RUNNER), "validate-decision", "--decision", str(decision),
        "--expected-sha256", digest, "--purpose", "generation", "--commit-hash", "commit", "--config-sha256", "config",
        "--model-revision", "revision", "--dataset-source", "kyunghyuncho/search_qa",
        "--dataset-revision", "dataset-revision", "--prompt-sha256", "prompt",
        "--retrieval-sha256", "retrieval", "--source-schema-sha256", "schema",
        "--student-thinking-mode", "direct", "--scratchpad-max-new-tokens", "128",
        "--query-temperature", "0", "--response-temperature", "0", "--top-p", "1",
        "--top-k", "8", "--k1", "1.2", "--b", "0.75",
    ]
    for mismatched_args in (("--model", "other-model", "--dataset-sha256", "dataset"), ("--model", "model", "--dataset-sha256", "other-dataset")):
        result = subprocess.run([*common, *mismatched_args], cwd=ROOT, text=True, capture_output=True, check=False)
        assert result.returncode == 2
        assert json.loads(result.stderr)["fallback_reason"] == "decision_identity_mismatch"


def test_freeze_excludes_faster_candidate_that_launch_validator_rejects(tmp_path: Path):
    baseline = tmp_path / "baseline.json"
    unsupported = tmp_path / "unsupported.json"
    baseline.write_text(json.dumps(valid_probe_result()), encoding="utf-8")
    unsupported.write_text(json.dumps(valid_probe_result(throughput=100.0, compile_=True)), encoding="utf-8")
    decision = tmp_path / "decision.json"
    result = subprocess.run(
        [sys.executable, str(PROBE_RUNNER), "freeze-decision", "--baseline", str(baseline), "--candidate", str(unsupported),
         "--output", str(decision), "--launch-max-steps", "100", "--launch-learning-rate", "1e-6", "--launch-epochs", "1",
         "--launch-save-steps", "10", "--launch-eval-steps", "10", "--query-max-new-tokens", "32", "--response-max-new-tokens", "256",
         "--student-thinking-mode", "direct", "--scratchpad-max-new-tokens", "128", "--query-temperature", "0",
         "--response-temperature", "0", "--top-p", "1", "--top-k", "8", "--k1", "1.2", "--b", "0.75"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    frozen = json.loads(decision.read_text(encoding="utf-8"))
    assert frozen["selected_result"] == str(baseline.resolve())
    assert frozen["rejected_candidates"][0]["fallback_reason"] == "launch_unsupported_compile"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("output_hash", None),
        ("decoded_output_hash", "fake"),
        ("examples_per_second", 0),
        ("tokens_per_second", float("nan")),
        ("peak_gpu_memory_mb", -1),
        ("gpu_utilization", {}),
        ("package_versions", {}),
        ("config", {}),
        ("identities", {}),
        ("output_token_ids", [[9, 9]]),
        ("gpu_hardware", {}),
        ("gpu_utilization", {
            "sample_count": 1, "utilization_mean_percent": 0.0, "utilization_peak_percent": 1.0,
            "nvidia_smi_peak_memory_mb": 1.0, "power_peak_watts": 1.0, "temperature_peak_c": 1.0,
        }),
    ],
)
def test_freeze_rejects_incomplete_or_fake_baseline_probe_artifacts(tmp_path: Path, field: str, value):
    baseline = valid_probe_result()
    if value is None:
        baseline.pop(field)
    else:
        baseline[field] = value
    result = freeze_probe(tmp_path, baseline)
    assert result.returncode == 2
    assert json.loads(result.stderr)["fallback_reason"] == "baseline_probe_invalid"


@pytest.mark.parametrize("mismatch", ("identities", "package_versions", "config_sha256", "gpu_hardware"))
def test_freeze_never_accepts_candidate_with_baseline_parity_mismatch(tmp_path: Path, mismatch: str):
    baseline = valid_probe_result()
    candidate = valid_probe_result(throughput=20.0)
    if mismatch == "identities":
        candidate["identities"]["model"] = "other-model"
    elif mismatch == "package_versions":
        candidate["package_versions"]["torch"] = "other-version"
    else:
        if mismatch == "config_sha256":
            candidate["identities"]["config_sha256"] = "other-config"
        else:
            candidate["gpu_hardware"]["devices"][0]["uuid"] = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
            candidate["gpu_utilization"]["monitored_uuids"] = ["GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"]
    result = freeze_probe(tmp_path, baseline, candidate)
    assert result.returncode == 0, result.stderr
    decision = json.loads((tmp_path / "decision.json").read_text(encoding="utf-8"))
    assert decision["selected_result"] == str((tmp_path / "baseline.json").resolve())
    assert "parity_mismatch" in decision["rejected_candidates"][0]["fallback_reason"]


def test_ordinary_probe_rejects_one_sample_or_uuid_coverage_mismatch(tmp_path: Path):
    baseline = valid_probe_result()
    candidate = valid_probe_result(throughput=20.0)
    candidate["gpu_utilization"]["sample_count"] = 1
    candidate["gpu_utilization"]["coverage_ratio"] = 0.03
    result = freeze_probe(tmp_path, baseline, candidate)
    assert result.returncode == 0, result.stderr
    frozen = json.loads((tmp_path / "decision.json").read_text(encoding="utf-8"))
    assert frozen["rejected_candidates"][0]["fallback_reason"] == "candidate_probe_gpu_telemetry_invalid"

    candidate = valid_probe_result(throughput=20.0)
    candidate["gpu_utilization"]["monitored_uuids"] = ["GPU-not-allocated"]
    result = freeze_probe(tmp_path / "uuid", baseline, candidate)
    assert result.returncode == 0, result.stderr
    frozen = json.loads((tmp_path / "uuid/decision.json").read_text(encoding="utf-8"))
    assert frozen["rejected_candidates"][0]["fallback_reason"] == "candidate_probe_gpu_telemetry_invalid"


@pytest.mark.parametrize("package", ("torch", "transformers", "trl", "deepspeed", "bitsandbytes"))
def test_probe_rejects_missing_required_package(tmp_path: Path, package: str):
    baseline = valid_probe_result()
    baseline["package_versions"][package] = "missing"
    result = freeze_probe(tmp_path, baseline)
    assert result.returncode == 2
    assert json.loads(result.stderr)["fallback_reason"] == "baseline_probe_invalid"


def test_optional_packages_may_be_missing_only_when_feature_disabled(tmp_path: Path):
    baseline = valid_probe_result()
    baseline["config"]["attention_implementation"] = "flash_attention_2"
    result = freeze_probe(tmp_path, baseline)
    assert result.returncode == 2
    assert "package" in result.stderr

    baseline = valid_probe_result()
    baseline["config"]["use_liger_kernel"] = True
    result = freeze_probe(tmp_path / "liger", baseline)
    assert result.returncode == 2
    assert "package" in result.stderr


def test_scale_decision_intentionally_compares_four_and_eight_identical_a100s(tmp_path: Path):
    four = tmp_path / "four.json"
    eight = tmp_path / "eight.json"
    four.write_text(json.dumps(valid_training_probe(throughput=100.0, gpu_count=4)), encoding="utf-8")
    eight.write_text(json.dumps(valid_training_probe(throughput=90.0, gpu_count=8)), encoding="utf-8")
    decision = tmp_path / "scale.json"
    result = subprocess.run([
        sys.executable, str(PROBE_RUNNER), "freeze-scale-decision", "--result", str(four),
        "--result", str(eight), "--output", str(decision), "--loss-relative-tolerance", "0.01",
    ], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    payload = json.loads(decision.read_text(encoding="utf-8"))
    assert payload["selected_train_gpus"] == 4
    assert payload["selected_global_tokens_per_second"] == 100.0
    assert payload["compared_gpu_counts"] == [4, 8]


def test_scale_decision_rejects_non_a100_or_non_count_hardware_drift(tmp_path: Path):
    four = valid_training_probe(throughput=100.0, gpu_count=4)
    eight = valid_training_probe(throughput=120.0, gpu_count=8)
    for device in eight["gpu_hardware"]["devices"]:
        device["total_memory_bytes"] -= 1
    paths = []
    for name, payload in (("four", four), ("eight", eight)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)
    result = subprocess.run([
        sys.executable, str(PROBE_RUNNER), "freeze-scale-decision", "--result", str(paths[0]),
        "--result", str(paths[1]), "--output", str(tmp_path / "scale.json"),
    ], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 2
    assert json.loads(result.stderr)["fallback_reason"] == "scale_hardware_profile_mismatch"


def test_collection_decision_freezes_largest_vram_teacher_and_distinct_student(tmp_path: Path):
    hardware = valid_probe_result(gpu_count=2)
    hardware["gpu_hardware"]["devices"][1]["free_memory_bytes"] += 1024
    result_path = tmp_path / "hardware.json"
    result_path.write_text(json.dumps(hardware), encoding="utf-8")
    decision = tmp_path / "collection.json"
    freeze = subprocess.run([
        sys.executable, str(PROBE_RUNNER), "freeze-collection-decision", "--hardware-result", str(result_path),
        "--output", str(decision), "--teacher-model", "teacher", "--teacher-revision", "teacher-rev",
        "--student-model", "student", "--student-revision", "student-rev", "--teacher-device-index", "1",
        "--student-device-index", "0",
    ], cwd=ROOT, text=True, capture_output=True, check=False)
    assert freeze.returncode == 0, freeze.stderr
    digest = hashlib.sha256(decision.read_bytes()).hexdigest()
    validate = subprocess.run([
        sys.executable, str(PROBE_RUNNER), "validate-collection-decision", "--decision", str(decision),
        "--expected-sha256", digest, "--teacher-model", "teacher", "--teacher-revision", "teacher-rev",
        "--student-model", "student", "--student-revision", "student-rev", "--allocated-gpus", "2",
        "--current-hardware", str(result_path),
    ], cwd=ROOT, text=True, capture_output=True, check=False)
    assert validate.returncode == 0, validate.stderr
    assert validate.stdout.split("\t")[:2] == ["cuda:1", "cuda:0"]


def test_collection_decision_accepts_hardware_probe_without_generation_artifact_fields(tmp_path: Path):
    probe = valid_probe_result(gpu_count=2)
    hardware = {
        "schema_version": probe["schema_version"],
        "status": probe["status"],
        "fallback_reason": probe["fallback_reason"],
        "gpu_hardware": probe["gpu_hardware"],
        "package_versions": probe["package_versions"],
    }
    result_path = tmp_path / "hardware.json"
    result_path.write_text(json.dumps(hardware), encoding="utf-8")
    decision = tmp_path / "collection.json"
    freeze = subprocess.run([
        sys.executable, str(PROBE_RUNNER), "freeze-collection-decision", "--hardware-result", str(result_path),
        "--output", str(decision), "--teacher-model", "teacher", "--teacher-revision", "teacher-rev",
        "--student-model", "student", "--student-revision", "student-rev", "--teacher-device-index", "0",
        "--student-device-index", "1",
    ], cwd=ROOT, text=True, capture_output=True, check=False)
    assert freeze.returncode == 0, freeze.stderr
    payload = json.loads(decision.read_text(encoding="utf-8"))
    assert payload["decision_kind"] == "collection_devices"
    assert payload["teacher"]["device_index"] == 0


def test_collection_decision_rejects_non_largest_or_same_device(tmp_path: Path):
    hardware = valid_probe_result(gpu_count=2)
    hardware["gpu_hardware"]["devices"][1]["free_memory_bytes"] -= 1024
    result_path = tmp_path / "hardware.json"
    result_path.write_text(json.dumps(hardware), encoding="utf-8")
    common = [
        sys.executable, str(PROBE_RUNNER), "freeze-collection-decision", "--hardware-result", str(result_path),
        "--output", str(tmp_path / "collection.json"), "--teacher-model", "teacher", "--teacher-revision", "rev",
        "--student-model", "student", "--student-revision", "rev",
    ]
    non_largest = subprocess.run([*common, "--teacher-device-index", "1", "--student-device-index", "0"], cwd=ROOT, text=True, capture_output=True, check=False)
    assert non_largest.returncode == 2
    assert json.loads(non_largest.stderr)["fallback_reason"] == "teacher_not_largest_fit_device"
    same = subprocess.run([*common, "--teacher-device-index", "0", "--student-device-index", "0"], cwd=ROOT, text=True, capture_output=True, check=False)
    assert same.returncode == 2
    assert json.loads(same.stderr)["fallback_reason"] == "collection_device_collision"


def test_checkpoint_rejects_exact_padding_weight_attack(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint-1"
    write_realistic_checkpoint(checkpoint, 1)
    torch.save({"padding.weight": torch.ones(4096, 4096)}, checkpoint / "pytorch_model.bin")
    result = subprocess.run([
        sys.executable, str(PROBE_RUNNER), "create-smoke-manifest", "--initial-checkpoint", str(checkpoint),
        "--resumed-checkpoint", str(checkpoint), "--output", str(tmp_path / "manifest.json"),
        "--commit-hash", "commit", "--config-sha256", "config", "--model", "model",
        "--model-revision", "revision", "--dataset-source", "source", "--dataset-revision", "revision",
        "--dataset-sha256", "data", "--prompt-sha256", "prompt", "--retrieval-sha256", "retrieval",
        "--source-schema-sha256", "schema", "--optimization-decision-sha256", "decision", "--scale-decision-sha256", "scale", "--method", "sft",
    ], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 2
    assert json.loads(result.stderr)["fallback_reason"] in {
        "checkpoint_model_semantics_invalid",
        "checkpoint_model_layer_set_invalid",
    }


def checkpoint_manifest_command(initial: Path, resumed: Path, output: Path) -> list[str]:
    return [
        sys.executable, str(PROBE_RUNNER), "create-smoke-manifest", "--initial-checkpoint", str(initial),
        "--resumed-checkpoint", str(resumed), "--output", str(output), "--commit-hash", "commit",
        "--config-sha256", "config", "--model", "model", "--model-revision", "revision",
        "--dataset-source", "source", "--dataset-revision", "revision", "--dataset-sha256", "data",
        "--prompt-sha256", "prompt", "--retrieval-sha256", "retrieval", "--source-schema-sha256", "schema",
        "--optimization-decision-sha256", "decision", "--scale-decision-sha256", "scale", "--method", "sft",
    ]


@pytest.mark.parametrize("attack", ("wrong_shape", "nonexistent_layer"))
def test_checkpoint_rejects_qwen3_architecture_shape_and_layer_attacks(tmp_path: Path, attack: str):
    initial = tmp_path / "checkpoint-1"
    resumed = tmp_path / "checkpoint-2"
    write_realistic_checkpoint(initial, 1)
    write_realistic_checkpoint(resumed, 2)
    state = torch.load(initial / "pytorch_model.bin", map_location="cpu", weights_only=True)
    if attack == "wrong_shape":
        state["model.layers.0.self_attn.q_proj.weight"] = torch.ones(7, 8)
    else:
        state["model.layers.999.self_attn.q_proj.weight"] = torch.ones(8, 8)
    torch.save(state, initial / "pytorch_model.bin")
    result = subprocess.run(checkpoint_manifest_command(initial, resumed, tmp_path / "manifest.json"), cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 2
    assert json.loads(result.stderr)["fallback_reason"] in {"checkpoint_model_shape_invalid", "checkpoint_model_layer_out_of_range"}


def test_training_probe_rejects_nonpositive_max_steps_before_cuda_or_subprocess(tmp_path: Path):
    data = tmp_path / "data.jsonl"
    data.write_text('{"question":"q"}\n', encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    result = subprocess.run([
        sys.executable, str(PROBE_RUNNER), "benchmark", "--result", str(tmp_path / "result.json"),
        "--commit-hash", "commit", "--probe-name", "bad-steps", "--config", str(config), "--data", str(data),
        "--eval-data", str(data), "--output-dir", str(tmp_path / "run"), "--deepspeed-config", str(config),
        "--model", "model", "--model-revision", "revision", "--dataset-source", "source", "--dataset-revision", "revision",
        "--prompt-sha256", "prompt", "--retrieval-sha256", "retrieval", "--source-schema-sha256", "schema",
        "--probe-kind", "training", "--training-method", "sft", "--max-steps", "0", "--num-generations", "1",
        "--rl-generation-batch-size", "1", "--max-completion-length", "8",
    ], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 2
    assert json.loads(result.stderr)["fallback_reason"] == "max_steps_invalid"


def test_prompt_preflight_requires_exact_sample_before_generation(tmp_path: Path):
    data = tmp_path / "31.jsonl"
    data.write_text("".join(json.dumps({"question": f"q-{index}"}) + "\n" for index in range(31)), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("training: {}\n", encoding="utf-8")
    decision = tmp_path / "decision.json"
    identities = {
        "commit_hash": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(), "model": "model", "model_revision": "revision",
        "dataset_source": "source", "dataset_revision": "revision", "dataset_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "prompt_sha256": "prompt", "retrieval_sha256": "retrieval", "source_schema_sha256": "schema",
    }
    write_identity_decision(decision, identities)
    decision_sha = hashlib.sha256(decision.read_bytes()).hexdigest()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "module").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    (fake_bin / "uv").write_text("#!/bin/bash\necho generate-called >&2\nexit 99\n", encoding="utf-8")
    for item in fake_bin.iterdir():
        item.chmod(0o755)
    env = {
        "PATH": f"{fake_bin}:{os.environ['PATH']}", "TURING_ACCOUNT": "account", "PROJECT_DIR": str(ROOT), "DATA": str(data),
        "MODEL": "model", "MODEL_REVISION": "revision", "OUTPUT_ROOT": str(tmp_path / "out"), "POLICY_HASH": "policy",
        "CONFIG": str(config), "DATASET_SOURCE": "source", "DATASET_REVISION": "revision", "PROMPT_HASH": "prompt",
        "RETRIEVAL_HASH": "retrieval", "SOURCE_SCHEMA_HASH": "schema", "OPTIMIZATION_DECISION": str(decision),
        "OPTIMIZATION_DECISION_SHA256": decision_sha, "STUDENT_THINKING_MODE": "direct", "SCRATCHPAD_MAX_NEW_TOKENS": "8",
        "QUERY_TEMPERATURE": "0", "RESPONSE_TEMPERATURE": "0", "TOP_P": "1", "TOP_K": "8", "BM25_K1": "1.2", "BM25_B": "0.75",
        "SLURM_NNODES": "1", "SLURM_NTASKS": "1", "SLURM_GPUS_ON_NODE": "1",
    }
    result = subprocess.run(["bash", str(SCRIPTS / "turing_prompt_preflight.sh")], cwd=ROOT, env={**os.environ, **env}, text=True, capture_output=True, check=False)
    assert result.returncode == 2
    assert "prompt_preflight_sample_too_small" in result.stderr
    assert "generate-called" not in result.stderr


def test_prompt_preflight_and_collection_contracts_are_explicit():
    prompt = (SCRIPTS / "turing_prompt_preflight.sh").read_text(encoding="utf-8")
    for field in ("query-temperature", "response-temperature", "top-p", "top-k", "k1", "b"):
        assert f"--{field} \"$FROZEN_" in prompt
    assert '"row_count":32' in prompt
    assert 'ALLOCATED_GPU_COUNT" != "2"' in (SCRIPTS / "turing_collect.sh").read_text(encoding="utf-8")
    collection = (SCRIPTS / "turing_collect.sh").read_text(encoding="utf-8")
    assert "probe-hardware" in collection
    assert "current-hardware" in collection
    assert "uuid" in collection


def test_generation_baseline_refresh_is_a_tracked_fail_fast_slurm_entrypoint():
    script_path = SCRIPTS / "turing_generation_baseline.sh"
    assert script_path.exists()
    script = script_path.read_text(encoding="utf-8")
    assert "set -euo pipefail" in script
    assert " benchmark " in script
    assert " freeze-decision " in script
    assert "--purpose generation" in script
    assert "--commit-hash \"$(git rev-parse HEAD)\"" in script
    assert "fallback_reason" in script


def test_collection_rejects_non_sha_policy_hash_before_allocation_checks(tmp_path: Path):
    required = {
        "TURING_ACCOUNT": "account", "PROJECT_DIR": str(ROOT), "DATA": str(tmp_path / "data.jsonl"),
        "OUTPUT": str(tmp_path / "out.jsonl"), "STUDENT_MODEL": "student", "STUDENT_REVISION": "student-rev",
        "TEACHER_MODEL": "teacher", "TEACHER_REVISION": "teacher-rev", "DATASET_REVISION": "data-rev",
        "DATASET_SOURCE": "source", "PROMPT_VERSION": "prompt-v1", "PROMPT_HASH": "a" * 64,
        "RETRIEVAL_HASH": "b" * 64, "SOURCE_SCHEMA_HASH": "c" * 64, "SEED": "7", "SHARD_INDEX": "0",
        "SHARD_COUNT": "1", "SHARD_SEED": "7", "MERGE_ID": "merge-v1", "SHARD_INPUT_SHA256": "d" * 64,
        "TRAJECTORY_CACHE": str(tmp_path / "cache.jsonl"), "POLICY_HASH": "student-searchqa-v1",
        "POLICY_VERSION": "student-searchqa-v1", "OPTIMIZATION_DECISION": str(tmp_path / "generation.json"),
        "OPTIMIZATION_DECISION_SHA256": "e" * 64, "COLLECTION_DECISION": str(tmp_path / "collection.json"),
        "COLLECTION_DECISION_SHA256": "f" * 64, "TEACHER_BATCH_SIZE": "1", "TEACHER_MAX_NEW_TOKENS": "32",
        "TEACHER_TEMPERATURE": "0", "TEACHER_TOP_P": "1", "TEACHER_THINKING": "true",
        "TEACHER_QUANTIZATION": "4bit", "TEACHER_FALLBACK_REASON": "none", "MAX_INTERVENTIONS": "1",
        "SIBLING_COUNT": "1", "SIBLING_SEEDS": "11",
    }
    result = run_script("turing_collect.sh", required)

    assert result.returncode == 2
    assert "policy_hash_invalid" in result.stderr
    assert "SLURM_NNODES" not in result.stderr


def test_generation_missing_explicit_thinking_mode_fails_before_launch(tmp_path: Path):
    env = required_generate_env(tmp_path)
    env.pop("STUDENT_THINKING_MODE")
    result = run_script("turing_generate.sh", env)
    assert result.returncode != 0
    assert "STUDENT_THINKING_MODE" in result.stderr


def test_probe_runner_rejects_decision_hash_mismatch(tmp_path: Path):
    decision = tmp_path / "optimization-decision.json"
    write_decision(decision)
    result = subprocess.run(
        [
            sys.executable,
            str(PROBE_RUNNER),
            "validate-decision",
            "--decision",
            str(decision),
            "--expected-sha256",
            "0" * 64,
                "--purpose",
                "training",
                "--commit-hash", "commit",
            "--config-sha256", "config", "--model", "model", "--model-revision", "revision",
            "--dataset-source", "kyunghyuncho/search_qa", "--dataset-revision", "dataset-revision",
            "--dataset-sha256", "dataset", "--prompt-sha256", "prompt", "--retrieval-sha256", "retrieval",
            "--source-schema-sha256", "schema",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    error = json.loads(result.stderr)
    assert error["status"] == "rejected"
    assert error["fallback_reason"] == "decision_hash_mismatch"


def test_fake_step_only_checkpoints_are_rejected(tmp_path: Path):
    initial = tmp_path / "checkpoint-10"
    resumed = tmp_path / "checkpoint-20"
    initial.mkdir()
    resumed.mkdir()
    (initial / "trainer_state.json").write_text('{"global_step":10}\n', encoding="utf-8")
    (resumed / "trainer_state.json").write_text('{"global_step":20}\n', encoding="utf-8")
    output = tmp_path / "smoke-manifest.json"
    result = subprocess.run(
        [
            sys.executable,
            str(PROBE_RUNNER),
            "create-smoke-manifest",
            "--initial-checkpoint",
            str(initial),
            "--resumed-checkpoint",
            str(resumed),
            "--output", str(output), "--commit-hash", "commit", "--config-sha256", "config",
            "--model", "model", "--model-revision", "revision", "--dataset-source", "source",
            "--dataset-revision", "dataset-revision", "--dataset-sha256", "dataset",
            "--prompt-sha256", "prompt", "--retrieval-sha256", "retrieval", "--source-schema-sha256", "schema",
            "--optimization-decision-sha256", "decision", "--scale-decision-sha256", "scale", "--method", "dpo",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert json.loads(result.stderr)["fallback_reason"] == "checkpoint_model_config_missing"


def test_checkpoint_smoke_script_has_real_bounded_save_resume_contract():
    text = (SCRIPTS / "turing_checkpoint_smoke.sh").read_text(encoding="utf-8")
    assert "--max-steps" in text
    assert "--dataloader-num-workers" in text
    assert "--per-device-train-batch-size" in text
    assert "task7_checkpoint_smoke_cli_missing" in text
    assert "fallback_reason" in text
    assert text.count("torch.distributed.run") >= 2
    assert "--resume-from-checkpoint" in text
    assert "create-smoke-manifest" in text


def test_smoke_manifest_requires_checkpoint_lineage_and_matching_identity(tmp_path: Path):
    initial = tmp_path / "checkpoint-1"
    resumed = tmp_path / "checkpoint-2"
    for checkpoint, step in ((initial, 1), (resumed, 2)):
        write_realistic_checkpoint(checkpoint, step)
    manifest = tmp_path / "smoke.json"
    identities = [
        "--commit-hash", "commit", "--config-sha256", "config", "--model", "model",
        "--model-revision", "revision", "--dataset-source", "source", "--dataset-revision", "dataset-revision",
        "--dataset-sha256", "dataset", "--prompt-sha256", "prompt",
        "--retrieval-sha256", "retrieval", "--source-schema-sha256", "schema",
        "--optimization-decision-sha256", "decision", "--scale-decision-sha256", "scale", "--method", "dpo",
    ]
    create = subprocess.run(
        [sys.executable, str(PROBE_RUNNER), "create-smoke-manifest", "--initial-checkpoint", str(initial),
         "--resumed-checkpoint", str(resumed), "--output", str(manifest), *identities],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert create.returncode == 0, create.stderr
    created_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    assert created_manifest["package_versions"]["torch"]
    assert len(created_manifest["training_metrics_lineage_sha256"]) == 64
    assert len(created_manifest["initial_checkpoint"]["training_metrics_sha256"]) == 64
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    valid = subprocess.run(
        [sys.executable, str(PROBE_RUNNER), "validate-checkpoints", "--smoke-manifest", str(manifest),
         "--expected-sha256", digest, *identities], cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert valid.returncode == 0, valid.stderr
    wrong_identities = identities.copy()
    wrong_identities[wrong_identities.index("config")]= "other-config"
    invalid = subprocess.run(
        [sys.executable, str(PROBE_RUNNER), "validate-checkpoints", "--smoke-manifest", str(manifest),
         "--expected-sha256", digest, *wrong_identities], cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert invalid.returncode == 2
    assert json.loads(invalid.stderr)["fallback_reason"] == "smoke_identity_mismatch"


def test_smoke_validation_rejects_package_identity_tampering(tmp_path: Path):
    identities = {
        "commit_hash": "commit", "config_sha256": "config", "model": "model", "model_revision": "revision",
        "dataset_source": "source", "dataset_revision": "dataset-revision", "dataset_sha256": "dataset",
        "prompt_sha256": "prompt", "retrieval_sha256": "retrieval", "source_schema_sha256": "schema",
    }
    manifest, _ = write_smoke_manifest(tmp_path, "package", identities, "decision", "dpo")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["package_versions"]["torch"] = "forged"
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(PROBE_RUNNER), "validate-checkpoints", "--smoke-manifest", str(manifest),
         "--expected-sha256", hashlib.sha256(manifest.read_bytes()).hexdigest(), "--commit-hash", "commit",
         "--config-sha256", "config", "--model", "model", "--model-revision", "revision",
         "--dataset-source", "source", "--dataset-revision", "dataset-revision", "--dataset-sha256", "dataset",
         "--eval-dataset-sha256", "not-applicable",
         "--prompt-sha256", "prompt", "--retrieval-sha256", "retrieval", "--source-schema-sha256", "schema",
         "--optimization-decision-sha256", "decision", "--scale-decision-sha256", "scale", "--method", "dpo"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 2
    assert json.loads(result.stderr)["fallback_reason"] == "smoke_package_identity_mismatch"


def test_checkpoint_gate_rejects_large_padded_arbitrary_mappings(tmp_path: Path):
    for root, step in ((tmp_path / "checkpoint-1", 1), (tmp_path / "checkpoint-2", 2)):
        root.mkdir()
        (root / "config.json").write_text(json.dumps({
            "model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"], "hidden_size": 8,
            "num_hidden_layers": 2, "num_attention_heads": 2, "vocab_size": 16,
        }), encoding="utf-8")
        (root / "trainer_state.json").write_text(
            json.dumps({"global_step": step, "max_steps": step + 1, "log_history": [{"step": step, "loss": 1.0}]}),
            encoding="utf-8",
        )
        torch.save({"model.embed_tokens.weight": "x" * 8192}, root / "pytorch_model.bin")
        torch.save({"state": {0: {"padding": "x" * 8192}}, "param_groups": [{"params": [0]}]}, root / "optimizer.pt")
        torch.save({"padding": "x" * 8192}, root / "scheduler.pt")
        torch.save({"padding": "x" * 8192}, root / "rng_state.pth")
    result = subprocess.run(
        [sys.executable, str(PROBE_RUNNER), "create-smoke-manifest", "--initial-checkpoint", str(tmp_path / "checkpoint-1"),
         "--resumed-checkpoint", str(tmp_path / "checkpoint-2"), "--output", str(tmp_path / "smoke.json"),
         "--commit-hash", "commit", "--config-sha256", "config", "--model", "model", "--model-revision", "revision",
         "--dataset-source", "source", "--dataset-revision", "dataset-revision", "--dataset-sha256", "dataset",
         "--prompt-sha256", "prompt", "--retrieval-sha256", "retrieval", "--source-schema-sha256", "schema",
         "--optimization-decision-sha256", "decision", "--scale-decision-sha256", "scale", "--method", "dpo"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 2
    assert json.loads(result.stderr)["fallback_reason"] in {
        "checkpoint_model_semantics_invalid",
        "checkpoint_model_config_invalid",
    }


def test_checkpoint_gate_rejects_tiny_named_placeholders(tmp_path: Path):
    initial = tmp_path / "checkpoint-1"
    resumed = tmp_path / "checkpoint-2"
    for checkpoint, step in ((initial, 1), (resumed, 2)):
        checkpoint.mkdir()
        (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": step}), encoding="utf-8")
        for name in ("pytorch_model.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
            (checkpoint / name).write_bytes(b"placeholder")
    result = subprocess.run(
        [sys.executable, str(PROBE_RUNNER), "create-smoke-manifest", "--initial-checkpoint", str(initial),
         "--resumed-checkpoint", str(resumed), "--output", str(tmp_path / "smoke.json"),
         "--commit-hash", "commit", "--config-sha256", "config", "--model", "model",
         "--model-revision", "revision", "--dataset-source", "source", "--dataset-revision", "dataset-revision",
         "--dataset-sha256", "dataset", "--prompt-sha256", "prompt", "--retrieval-sha256", "retrieval",
         "--source-schema-sha256", "schema", "--optimization-decision-sha256", "decision", "--scale-decision-sha256", "scale", "--method", "dpo"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 2
    assert json.loads(result.stderr)["fallback_reason"] in {"trainer_state_invalid", "checkpoint_state_too_small", "checkpoint_model_config_missing"}


def test_owned_checkpoint_gate_rejects_no_progress(tmp_path: Path):
    initial = tmp_path / "checkpoint-10"
    resumed = tmp_path / "resumed" / "checkpoint-10"
    write_realistic_checkpoint(initial, 10)
    write_realistic_checkpoint(resumed, 10)
    result = subprocess.run(
        [
            sys.executable,
            str(PROBE_RUNNER),
            "create-smoke-manifest",
            "--initial-checkpoint",
            str(initial),
            "--resumed-checkpoint",
            str(resumed),
            "--output", str(tmp_path / "gate.json"), "--commit-hash", "commit", "--config-sha256", "config",
            "--model", "model", "--model-revision", "revision", "--dataset-source", "source",
            "--dataset-revision", "dataset-revision", "--dataset-sha256", "dataset",
            "--prompt-sha256", "prompt", "--retrieval-sha256", "retrieval", "--source-schema-sha256", "schema",
            "--optimization-decision-sha256", "decision", "--scale-decision-sha256", "scale", "--method", "dpo",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert json.loads(result.stderr)["fallback_reason"] == "resume_step_not_advanced"


def test_comparisons_reaches_launch_only_after_all_decisions_and_smokes_validate(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text("training: {}\n", encoding="utf-8")
    sft_train = tmp_path / "sft.jsonl"
    sft_train.write_text('{"id":"sft"}\n', encoding="utf-8")
    sft_eval = tmp_path / "sft-eval.jsonl"
    sft_eval.write_text('{"id":"eval"}\n', encoding="utf-8")
    rl_data = tmp_path / "rl.jsonl"
    rl_data.write_text('{"id":"rl"}\n', encoding="utf-8")
    rl_eval = tmp_path / "rl-eval.jsonl"
    rl_eval.write_text('{"id":"rl-eval"}\n', encoding="utf-8")
    val_data = tmp_path / "val.jsonl"
    val_data.write_text('{"id":"val"}\n', encoding="utf-8")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    common = {"commit_hash": commit, "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
              "dataset_source": "source", "dataset_revision": "dataset-revision", "prompt_sha256": "prompt",
              "retrieval_sha256": "retrieval", "source_schema_sha256": "schema"}
    output_root = tmp_path / "output"
    specs = {
        "SFT_TRAIN": ("base", "base-revision", sft_train, sft_eval, "sft"),
        "GRPO_TRAIN": ("rl-start", "rl-revision", rl_data, rl_eval, "grpo"),
        "DAPO_TRAIN": ("rl-start", "rl-revision", rl_data, rl_eval, "dapo"),
        "SFT_GENERATION": (str(output_root / "sft/final"), "sft-output-revision", val_data, None, None),
        "GRPO_GENERATION": (str(output_root / "grpo/final"), "grpo-output-revision", val_data, None, None),
        "DAPO_GENERATION": (str(output_root / "dapo/final"), "dapo-output-revision", val_data, None, None),
    }
    env: dict[str, str] = {}
    for name, (model, revision, data, eval_data, method) in specs.items():
        identities = {**common, "model": model, "model_revision": revision, "dataset_sha256": hashlib.sha256(data.read_bytes()).hexdigest()}
        if eval_data is not None:
            identities["eval_dataset_sha256"] = hashlib.sha256(eval_data.read_bytes()).hexdigest()
        decision = tmp_path / f"{name.lower()}-decision.json"
        decision_sha = write_identity_decision(decision, identities, method or "sft")
        env[f"{name}_DECISION"] = str(decision)
        env[f"{name}_DECISION_SHA256"] = decision_sha
        if method:
            scale = tmp_path / f"{name.lower()}-scale.json"
            scale_sha = write_scale_decision(scale, identities, training_method=method)
            env[f"{name.split('_')[0]}_SCALE_DECISION"] = str(scale)
            env[f"{name.split('_')[0]}_SCALE_DECISION_SHA256"] = scale_sha
            manifest, manifest_sha = write_smoke_manifest(tmp_path, name.lower(), identities, decision_sha, method, scale_sha)
            env[f"{name.split('_')[0]}_SMOKE_MANIFEST"] = str(manifest)
            env[f"{name.split('_')[0]}_SMOKE_MANIFEST_SHA256"] = manifest_sha
    fake_bin = tmp_path / "home/.local/bin"
    fake_bin.mkdir(parents=True)
    (fake_bin / "module").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    (fake_bin / "nvidia-smi").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    (fake_bin / "git").write_text(f"#!/bin/bash\n[[ \"${{1:-}}\" == rev-parse ]] && printf '%s\\n' '{commit}' && exit 0\nexec /usr/bin/git \"$@\"\n", encoding="utf-8")
    (fake_bin / "uv").write_text(
        "#!/bin/bash\ncase \" $* \" in *\" --help \"*) echo '--eval --max-steps --max-length --dataloader-num-workers --per-device-train-batch-size --per-device-eval-batch-size --gradient-accumulation-steps --attention-implementation --gradient-checkpointing --packing --padding-free --use-liger-kernel'; exit 0;; "
        "*\" torch.distributed.run \"*) echo mocked-training-launch >&2; exit 91;; esac\n"
        f"if [[ \"${{1:-}}\" == run && \"${{2:-}}\" == --frozen && \"${{3:-}}\" == python && \"${{4:-}}\" == \"{PROBE_RUNNER}\" ]]; then exec \"{sys.executable}\" \"${{@:4}}\"; fi\n"
        "echo 'torch=x;transformers=x;trl=x;deepspeed=x;bitsandbytes=x'\n",
        encoding="utf-8",
    )
    for item in fake_bin.iterdir():
        item.chmod(0o755)
    env.update({
        "HOME": str(tmp_path / "home"), "PATH": f"{fake_bin}:{os.environ['PATH']}", "HF_CACHE_ROOT": str(tmp_path / "hf"), "TURING_ACCOUNT": "account",
        "PROJECT_DIR": str(ROOT), "CONFIG": str(config), "BASE_MODEL": "base", "BASE_REVISION": "base-revision",
        "RL_START_MODEL": "rl-start", "RL_START_REVISION": "rl-revision", "SFT_TRAIN": str(sft_train),
        "SFT_EVAL": str(sft_eval), "RL_DATA": str(rl_data), "RL_EVAL": str(rl_eval), "VAL_DATA": str(val_data), "OUTPUT_ROOT": str(output_root),
        "TRAIN_GPUS": "4", "DATASET_SOURCE": "source", "DATASET_REVISION": "dataset-revision", "PROMPT_HASH": "prompt",
        "RETRIEVAL_HASH": "retrieval", "SOURCE_SCHEMA_HASH": "schema", "POLICY_HASH": "policy", "LEARNING_RATE": "1e-6",
        "EPOCHS": "1", "SAVE_STEPS": "10", "EVAL_STEPS": "10", "SFT_OUTPUT_REVISION": "sft-output-revision",
        "GRPO_OUTPUT_REVISION": "grpo-output-revision", "DAPO_OUTPUT_REVISION": "dapo-output-revision",
        "STUDENT_THINKING_MODE": "direct", "SCRATCHPAD_MAX_NEW_TOKENS": "128", "QUERY_TEMPERATURE": "0",
        "RESPONSE_TEMPERATURE": "0", "TOP_P": "1", "TOP_K": "8", "BM25_K1": "1.2", "BM25_B": "0.75",
        "SLURM_NNODES": "1", "SLURM_NTASKS": "1", "SLURM_GPUS_ON_NODE": "4", "SLURM_JOB_ID": "123",
    })
    result = run_script("turing_comparisons.sh", env)
    assert result.returncode == 91, result.stderr
    assert "event=launch_contract_validated" in result.stdout
    assert "mocked-training-launch" in result.stderr


def test_build_preferences_fails_without_explicit_canonical_data(tmp_path: Path):
    result = run_script("turing_build_preferences.sh", {
        "TURING_ACCOUNT": "account", "PROJECT_DIR": str(ROOT), "TRAJECTORIES": str(tmp_path / "trajectories.jsonl"),
        "OUTPUT": str(tmp_path / "preferences.jsonl"),
    })
    assert result.returncode != 0
    assert "PREFERENCE_DATA" in result.stderr
