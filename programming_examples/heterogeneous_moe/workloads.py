# SPDX-License-Identifier: MIT

from __future__ import annotations

import copy
from typing import Any

from case_runner import contains_npu
from kernels import KernelConfig
from reference import (
    DEFAULT_INPUT_SCALE,
    DEFAULT_ROUTING_PROFILE,
    DEFAULT_WEIGHT_SCALE,
    random_inputs,
    random_weights,
    router_logits,
    topk_weights,
)

SHAPE_SWEEP = [
    {"name": "small", "batch_tokens": 4, "hidden_size": 16, "ffn_size": 32, "scale": 0.5},
    {"name": "smallplus", "batch_tokens": 4, "hidden_size": 24, "ffn_size": 48, "scale": 0.375},
    {"name": "medium", "batch_tokens": 8, "hidden_size": 32, "ffn_size": 64, "scale": 0.25},
    {"name": "midlarge", "batch_tokens": 8, "hidden_size": 40, "ffn_size": 80, "scale": 0.1875},
    {"name": "large", "batch_tokens": 8, "hidden_size": 48, "ffn_size": 96, "scale": 0.125},
]

MODEL_PRESETS = [
    {
        "name": "lfm2_8b_a1b",
        "model_id": "LiquidAI/LFM2-8B-A1B",
        "model_class": "small_modern",
        "batch_tokens": 4,
        "hidden_size": 2048,
        "ffn_size": 1792,
        "scale": 0.015625,
        "weight_storage": "bf16",
        "compute_dtype": "bf16",
        "num_experts": 32,
        "active_experts": 4,
    },
    {
        "name": "granite4_h_tiny_7b",
        "model_id": "ibm-granite/granite-4.0-h-tiny",
        "model_class": "small_modern",
        "batch_tokens": 4,
        "hidden_size": 1536,
        "ffn_size": 512,
        "scale": 0.015625,
        "weight_storage": "bf16",
        "compute_dtype": "bf16",
        "num_experts": 64,
        "active_experts": 6,
    },
    {
        "name": "gemma4_26b_a4b_qbf16",
        "model_id": "google/gemma-4-26B-A4B-it",
        "model_class": "frontier_modern",
        "batch_tokens": 4,
        "hidden_size": 2816,
        "ffn_size": 704,
        "scale": 0.011048543456039806,
        "weight_storage": "quantized",
        "compute_dtype": "bf16",
        "num_experts": 128,
        "active_experts": 8,
    },
    {
        "name": "qwen36_35b_a3b_qbf16",
        "model_id": "Qwen/Qwen3.6-35B-A3B",
        "model_class": "frontier_modern",
        "batch_tokens": 4,
        "hidden_size": 2048,
        "ffn_size": 512,
        "scale": 0.015625,
        "weight_storage": "quantized",
        "compute_dtype": "bf16",
        "num_experts": 256,
        "active_experts": 8,
        "shared_expert_ffn_size": 512,
    },
]

CONTEXT_LENGTHS = [64, 128, 256, 512, 1024, 2048]
ROUTING_PROFILES = ["balanced", "expert0_hot", "expert1_hot", "alternating"]
ROUTING_CASE_NAMES = [
    "cpu_top2",
    "gpu_top2",
    "npu_top2",
    "cpu_router_gpu_experts_cpu_agg_top2",
    "cpu_router_npu_experts_cpu_agg_top2",
    "cpu_router_npu_gpu_experts_cpu_agg_top2",
]


def case_map(matrix: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {case["name"]: dict(case) for case in matrix["cases"]}


def select_cases(matrix: dict[str, Any], names: list[str] | None = None) -> list[dict[str, Any]]:
    if names is None:
        return [dict(case) for case in matrix["cases"]]
    cases = case_map(matrix)
    return [dict(cases[name]) for name in names]


def required_backends(cases: list[dict[str, Any]], allow_npu: bool) -> set[str]:
    backends: set[str] = set()
    for case in cases:
        stage_backends = {
            "router": case["router_backend"],
            "expert0": case["expert0_backend"],
            "expert1": case["expert1_backend"],
            "aggregation": case["aggregation_backend"],
        }
        if contains_npu(stage_backends) and not allow_npu:
            continue
        for backend in stage_backends.values():
            if backend == "cpu":
                continue
            if backend == "npu" and not allow_npu:
                continue
            backends.add(backend)
    return backends


def routed_tokens_for_context(preset: dict[str, Any], context_length: int) -> int:
    active_experts = int(preset["active_experts"])
    num_experts = int(preset["num_experts"])
    return max(1, (context_length * active_experts + num_experts - 1) // num_experts)


def routing_stats(manifest: dict[str, Any]) -> dict[str, Any]:
    workload = manifest.get("workload", {})
    cfg = KernelConfig(
        batch_tokens=int(workload.get("routed_tokens", manifest["model"]["batch_tokens"])),
        hidden_size=manifest["model"]["hidden_size"],
        ffn_size=manifest["model"]["ffn_size"],
        dtype=manifest["model"]["dtype"],
    )
    routing_profile = workload.get("routing_profile", DEFAULT_ROUTING_PROFILE)
    input_scale = float(manifest.get("inputs", {}).get("scale", DEFAULT_INPUT_SCALE))
    weight_scale = float(manifest.get("weights", {}).get("scale", DEFAULT_WEIGHT_SCALE))
    inputs = random_inputs(cfg, manifest["inputs"]["seed"], scale=input_scale, routing_profile=routing_profile)
    weights = random_weights(cfg, manifest["weights"]["seed"], scale=weight_scale, routing_profile=routing_profile)
    logits = router_logits(inputs, weights.router, cfg.dtype)
    top1 = topk_weights(logits, "top1", cfg.dtype)
    top2 = topk_weights(logits, "top2", cfg.dtype)
    top1_counts = top1.astype("float32").sum(axis=0)
    top2_mass = top2.astype("float32").sum(axis=0)
    return {
        "top1_token_counts": {"expert0": int(round(float(top1_counts[0]))), "expert1": int(round(float(top1_counts[1])))},
        "top2_probability_mass": {"expert0": float(top2_mass[0]), "expert1": float(top2_mass[1])},
    }


def clone_manifest(base_manifest: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(base_manifest)


def _set_generated_paths(manifest: dict[str, Any], artifact_root: str, source_root: str) -> None:
    manifest["paths"]["artifacts"] = artifact_root
    manifest["paths"]["generated_air_sources"] = source_root


def shape_workloads(base_manifest: dict[str, Any], matrix: dict[str, Any]) -> list[dict[str, Any]]:
    cases = select_cases(matrix)
    workloads = []
    for shape in SHAPE_SWEEP:
        manifest = clone_manifest(base_manifest)
        manifest["model"]["batch_tokens"] = shape["batch_tokens"]
        manifest["model"]["hidden_size"] = shape["hidden_size"]
        manifest["model"]["ffn_size"] = shape["ffn_size"]
        manifest["inputs"]["scale"] = shape["scale"]
        manifest["weights"]["scale"] = shape["scale"]
        manifest.setdefault("workload", {})["routing_profile"] = DEFAULT_ROUTING_PROFILE
        workload_name = f"{shape['name']}_{shape['batch_tokens']}x{shape['hidden_size']}x{shape['ffn_size']}"
        root = f"artifacts/workloads/shape_sweep/{workload_name}"
        _set_generated_paths(manifest, f"{root}/compiled", f"{root}/air_sources")
        workloads.append(
            {
                "suite": "shape_sweep",
                "name": workload_name,
                "manifest": manifest,
                "cases": cases,
            }
        )
    return workloads


def routing_workloads(base_manifest: dict[str, Any], matrix: dict[str, Any]) -> list[dict[str, Any]]:
    cases = select_cases(matrix, ROUTING_CASE_NAMES)
    workloads = []
    shared_artifact_root = (
        f"artifacts/workloads/routing_sweep/shared_"
        f"{base_manifest['model']['batch_tokens']}x{base_manifest['model']['hidden_size']}x{base_manifest['model']['ffn_size']}"
    )
    for profile in ROUTING_PROFILES:
        manifest = clone_manifest(base_manifest)
        manifest.setdefault("workload", {})["routing_profile"] = profile
        workload_name = f"{profile}_{manifest['model']['batch_tokens']}x{manifest['model']['hidden_size']}x{manifest['model']['ffn_size']}"
        _set_generated_paths(manifest, f"{shared_artifact_root}/compiled", f"{shared_artifact_root}/air_sources")
        workloads.append(
            {
                "suite": "routing_sweep",
                "name": workload_name,
                "manifest": manifest,
                "cases": cases,
            }
        )
    return workloads


def model_preset_workloads(base_manifest: dict[str, Any], matrix: dict[str, Any]) -> list[dict[str, Any]]:
    cases = select_cases(matrix)
    workloads = []
    for preset in MODEL_PRESETS:
        artifact_name = f"{preset['name']}_{preset['batch_tokens']}x{preset['hidden_size']}x{preset['ffn_size']}"
        for context_length in CONTEXT_LENGTHS:
            manifest = clone_manifest(base_manifest)
            manifest["model"]["batch_tokens"] = preset["batch_tokens"]
            manifest["model"]["hidden_size"] = preset["hidden_size"]
            manifest["model"]["ffn_size"] = preset["ffn_size"]
            manifest["model"]["dtype"] = preset["compute_dtype"]
            manifest["inputs"]["scale"] = preset["scale"]
            manifest["weights"]["scale"] = preset["scale"]
            workload = manifest.setdefault("workload", {})
            workload["routing_profile"] = DEFAULT_ROUTING_PROFILE
            workload["model_id"] = preset["model_id"]
            workload["model_class"] = preset["model_class"]
            workload["weight_storage"] = preset["weight_storage"]
            workload["compute_dtype"] = preset["compute_dtype"]
            workload["num_experts"] = preset["num_experts"]
            workload["active_experts"] = preset["active_experts"]
            workload["context_length"] = context_length
            workload["routed_tokens"] = routed_tokens_for_context(preset, context_length)
            if "shared_expert_ffn_size" in preset:
                workload["shared_expert_ffn_size"] = preset["shared_expert_ffn_size"]
            workload_name = (
                f"{preset['name']}_ctx{context_length}_rt{workload['routed_tokens']}_"
                f"{preset['batch_tokens']}x{preset['hidden_size']}x{preset['ffn_size']}"
            )
            root = f"artifacts/workloads/model_presets/{artifact_name}"
            _set_generated_paths(manifest, f"{root}/compiled", f"{root}/air_sources")
            workloads.append(
                {
                    "suite": "model_presets",
                    "name": workload_name,
                    "manifest": manifest,
                    "cases": cases,
                    "model_preset": dict(preset),
                }
            )
    return workloads


def suite_workloads(
    suite_names: list[str],
    base_manifest: dict[str, Any],
    matrix: dict[str, Any],
) -> list[dict[str, Any]]:
    workloads: list[dict[str, Any]] = []
    if "shape_sweep" in suite_names:
        workloads.extend(shape_workloads(base_manifest, matrix))
    if "routing_sweep" in suite_names:
        workloads.extend(routing_workloads(base_manifest, matrix))
    if "model_presets" in suite_names:
        workloads.extend(model_preset_workloads(base_manifest, matrix))
    return workloads
