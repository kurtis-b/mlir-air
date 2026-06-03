#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gemma3 real-model NPU execution wiring manifest.

This module does not execute kernels. It turns the real-shape preflight data
into an explicit per-layer stage contract so the model runner can be wired
incrementally without hiding host fallbacks or paper-parity blockers.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from gemma3_artifacts import MODEL_SPECS
from gemma3_kernel_parity import kernel_parity_targets
from gemma3_nonlinears import nonlinear_registry
from gemma3_npu_preflight import Gemma3NPUPreflightPlan, ProjectionPlan, build_preflight_plan
from gemma3_static_preload import has_full_xrt_preload_evidence
from gemma3_xrt_runner import has_paper_shape_bo_allocation_evidence


TEXT_STAGE_TEMPLATE = (
    ("prefill", "pre_attention_norm", "host-fallback", "rms_norm", "weighted_rms_norm candidate"),
    ("prefill", "qkv_projection", "npu-candidate", "q4nx+bf16_mm", "Q4NX dequant plus BF16 MM"),
    ("prefill", "rope", "host-fallback", "rope", "Llama half-split source candidate"),
    ("prefill", "qk_norm", "host-fallback", "qk_norm", "weighted_rms_norm per-head candidate"),
    ("prefill", "kv_cache_append", "host-runtime", "host", "append K/V tensors after projection"),
    ("prefill", "attention", "npu-candidate", "flowqkv", "causal local/global FlowQKV"),
    ("prefill", "output_projection", "npu-candidate", "q4nx+bf16_mm", "Q4NX dequant plus BF16 MM"),
    ("prefill", "attention_residual", "host-fallback", "residual_add", "residual add after attention"),
    ("prefill", "post_attention_norm", "host-fallback", "rms_norm", "weighted_rms_norm candidate"),
    ("prefill", "mlp_gate_up_projection", "npu-candidate", "q4nx+bf16_mm", "Q4NX dequant plus BF16 MM"),
    ("prefill", "mlp_activation", "host-fallback", "mlp_activation", "standalone GeGLU hardware-smoke only"),
    ("prefill", "mlp_down_projection", "npu-candidate", "q4nx+bf16_mm", "Q4NX dequant plus BF16 MM"),
    ("prefill", "mlp_residual", "host-fallback", "residual_add", "residual add after MLP"),
    ("decode", "pre_attention_norm", "host-fallback", "rms_norm", "weighted_rms_norm candidate"),
    ("decode", "qkv_projection", "npu-candidate", "fused_dqp", "FusedDQP decode projection"),
    ("decode", "rope", "host-fallback", "rope", "Llama half-split source candidate"),
    ("decode", "qk_norm", "host-fallback", "qk_norm", "weighted_rms_norm per-head candidate"),
    ("decode", "kv_cache_append", "host-runtime", "host", "append one K/V entry"),
    ("decode", "attention", "npu-candidate", "flowkv", "Q_CHUNK=1 FlowKV"),
    ("decode", "output_projection", "npu-candidate", "fused_dqp", "FusedDQP decode projection"),
    ("decode", "attention_residual", "host-fallback", "residual_add", "residual add after attention"),
    ("decode", "post_attention_norm", "host-fallback", "rms_norm", "weighted_rms_norm candidate"),
    ("decode", "mlp_gate_up_projection", "npu-candidate", "fused_dqp", "FusedDQP decode projection"),
    ("decode", "mlp_activation", "host-fallback", "mlp_activation", "standalone GeGLU hardware-smoke only"),
    ("decode", "mlp_down_projection", "npu-candidate", "fused_dqp", "FusedDQP decode projection"),
    ("decode", "mlp_residual", "host-fallback", "residual_add", "residual add after MLP"),
)


@dataclass(frozen=True)
class Gemma3NPUStage:
    phase: str
    layer_index: int
    stage_index: int
    role: str
    backend: str
    kernel: str
    route: str
    attention_kind: str
    window_len: int
    tensor_contract: str
    status: str
    blockers: tuple[str, ...]

    def format(self) -> str:
        blockers = ",".join(self.blockers) if self.blockers else "none"
        return (
            f"stage {self.phase}:L{self.layer_index}:{self.stage_index}:{self.role} "
            f"backend={self.backend} kernel={self.kernel} route={self.route} "
            f"attention={self.attention_kind} window={self.window_len} "
            f"status={self.status} blockers={blockers}"
        )


@dataclass(frozen=True)
class Gemma3NPUWiringPlan:
    model_variant: str
    status: str
    layers: int
    hidden_size: int
    head_dim: int
    attention_pattern: str
    stage_count: int
    npu_candidate_count: int
    host_fallback_count: int
    host_runtime_count: int
    blockers: tuple[str, ...]
    stages: tuple[Gemma3NPUStage, ...]

    def format(self, *, include_stages: bool = False) -> str:
        blockers = ",".join(self.blockers) if self.blockers else "none"
        lines = [
            f"npu_wiring model={self.model_variant} status={self.status} "
            f"layers={self.layers} hidden={self.hidden_size} head_dim={self.head_dim} "
            f"stages={self.stage_count} npu_candidates={self.npu_candidate_count} "
            f"host_fallbacks={self.host_fallback_count} host_runtime={self.host_runtime_count} "
            f"pattern={self.attention_pattern} blockers={blockers}"
        ]
        if include_stages:
            lines.extend(stage.format() for stage in self.stages)
        return "\n".join(lines)

    def to_json_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["stages"] = [asdict(stage) for stage in self.stages]
        return data


def _kernel_route_lookup() -> dict[str, str]:
    route = {}
    preferred = {
        "q4nx": "q4nx_smoke_8x4",
        "bf16_mm": "bf16_mm_8x4",
        "flowqkv": "flowqkv_paper",
        "flowkv": "flowkv_paper",
        "fused_dqp": "fused_dqp_paper",
    }
    for target in kernel_parity_targets():
        for kernel, name in preferred.items():
            if target.name == name:
                route[kernel] = f"{target.herd_shape}/{target.output_mode}/{target.schedule_mode}"
    return route


def _nonlinear_status_lookup() -> dict[str, str]:
    return {spec.operation: spec.hardware_status for spec in nonlinear_registry()}


def _attention_kind(layer_index: int) -> str:
    return "global_full" if layer_index % 6 == 5 else "local_swa"


def _stage_route(kernel: str, routes: dict[str, str]) -> str:
    if "+" in kernel:
        pieces = [_stage_route(piece, routes) for piece in kernel.split("+")]
        return "+".join(pieces)
    if kernel == "mlp_activation":
        return "geglu/standalone-elf-smoke"
    return routes.get(kernel, "host")


def _stage_backend(backend: str, kernel: str, nonlinear_status: dict[str, str]) -> str:
    if (
        backend == "host-fallback"
        and kernel == "mlp_activation"
        and nonlinear_status.get(kernel, "").startswith("hardware-smoke-pass")
    ):
        return "npu-candidate"
    return backend


def _stage_status(
    backend: str,
    kernel: str,
    nonlinear_status: dict[str, str],
) -> tuple[str, tuple[str, ...]]:
    launch_blockers = ("model-kernel-launch-not-wired", "paper-shape-hardware-rerun-required")
    if backend == "npu-candidate":
        if (
            kernel == "mlp_activation"
            and nonlinear_status.get(kernel, "").startswith("hardware-smoke-pass")
        ):
            return "standalone-hardware-smoke-model-candidate", launch_blockers
        return "candidate-only", launch_blockers
    if backend == "host-runtime":
        return "host-runtime-contract", ()
    if kernel in nonlinear_status and nonlinear_status[kernel].startswith("hardware-smoke-pass"):
        return "standalone-hardware-smoke-host-fallback", ("model-stage-not-promoted",)
    return "host-fallback", ("model-stage-not-promoted",)


def build_wiring_plan_from_preflight(
    preflight: Gemma3NPUPreflightPlan,
    *,
    use_static_preload_evidence: bool = False,
    use_bo_allocation_evidence: bool = False,
) -> Gemma3NPUWiringPlan:
    if preflight.layers is None or preflight.hidden_size is None or preflight.head_dim is None:
        raise ValueError("preflight plan must include layer count, hidden size, and head dim")
    routes = _kernel_route_lookup()
    nonlinear_status = _nonlinear_status_lookup()
    window_len = int(preflight.sliding_window or 0)
    stages: list[Gemma3NPUStage] = []
    for layer_index in range(int(preflight.layers)):
        attention_kind = _attention_kind(layer_index)
        layer_window = 0 if attention_kind == "global_full" else window_len
        for stage_index, (phase, role, backend, kernel, contract) in enumerate(TEXT_STAGE_TEMPLATE):
            stage_backend = _stage_backend(backend, kernel, nonlinear_status)
            status, blockers = _stage_status(stage_backend, kernel, nonlinear_status)
            stages.append(
                Gemma3NPUStage(
                    phase=phase,
                    layer_index=layer_index,
                    stage_index=stage_index,
                    role=role,
                    backend=stage_backend,
                    kernel=kernel,
                    route=_stage_route(kernel, routes) if stage_backend != "host-fallback" else "host",
                    attention_kind=attention_kind,
                    window_len=layer_window,
                    tensor_contract=contract,
                    status=status,
                    blockers=blockers,
                )
            )

    blockers = [
        "model-kernel-launch-not-wired",
        "model-kernel-argument-binding-not-validated",
    ]
    if not (
        use_static_preload_evidence
        and has_full_xrt_preload_evidence(preflight.model_variant)
    ):
        blockers.append("full-static-weight-bo-preload-not-validated")
    if not (
        use_bo_allocation_evidence
        and has_paper_shape_bo_allocation_evidence(preflight.model_variant)
    ):
        blockers.append("paper-shape-bo-allocation-not-validated")
    blockers.extend(
        [
            "nonlinear-model-stage-promotion-incomplete",
            "paper-shape-hardware-rerun-required",
        ]
    )
    if preflight.model_variant.endswith("vision"):
        blockers.append("vision-npu-path-not-implemented")

    return Gemma3NPUWiringPlan(
        model_variant=preflight.model_variant,
        status="BLOCKED",
        layers=int(preflight.layers),
        hidden_size=int(preflight.hidden_size),
        head_dim=int(preflight.head_dim),
        attention_pattern=preflight.attention_pattern,
        stage_count=len(stages),
        npu_candidate_count=sum(stage.backend == "npu-candidate" for stage in stages),
        host_fallback_count=sum(stage.backend == "host-fallback" for stage in stages),
        host_runtime_count=sum(stage.backend == "host-runtime" for stage in stages),
        blockers=tuple(dict.fromkeys(blockers)),
        stages=tuple(stages),
    )


def build_wiring_plan(
    model_variant: str,
    *,
    weights_dir: Path | None = None,
) -> Gemma3NPUWiringPlan:
    return build_wiring_plan_from_preflight(
        build_preflight_plan(model_variant, weights_dir=weights_dir),
        use_static_preload_evidence=True,
        use_bo_allocation_evidence=True,
    )


def _fake_preflight() -> Gemma3NPUPreflightPlan:
    projection = ProjectionPlan(
        family="q_proj",
        shape=(1024, 1152),
        padded_shape=(1024, 1280),
        row_blocks=32,
        col_blocks=5,
        requires_padding=True,
        max_abs_error=0.1,
        mean_abs_error=0.01,
    )
    return Gemma3NPUPreflightPlan(
        model_variant="gemma3-1b",
        status="READY_FOR_NPU_WIRING",
        blocker="npu-model-execution-not-implemented",
        layers=6,
        hidden_size=1152,
        intermediate_size=6912,
        head_dim=256,
        num_attention_heads=4,
        num_key_value_heads=1,
        sliding_window=512,
        attention_pattern="5-local-1-global",
        projections=(projection,),
    )


def _self_test() -> None:
    plan = build_wiring_plan_from_preflight(_fake_preflight())
    if plan.stage_count != 6 * len(TEXT_STAGE_TEMPLATE):
        raise AssertionError(plan.stage_count)
    if plan.npu_candidate_count != 6 * 12:
        raise AssertionError(plan.npu_candidate_count)
    if plan.host_fallback_count != 6 * 12:
        raise AssertionError(plan.host_fallback_count)
    global_attention = [stage for stage in plan.stages if stage.layer_index == 5 and stage.role == "attention"]
    if not global_attention or any(stage.attention_kind != "global_full" or stage.window_len != 0 for stage in global_attention):
        raise AssertionError(global_attention)
    print(plan.format())
    for stage in plan.stages[:8]:
        print(stage.format())
    for stage in [stage for stage in plan.stages if stage.role == "mlp_activation"][:2]:
        print(stage.format())
    for stage in global_attention:
        print(stage.format())
    print("GEMMA3_NPU_WIRING_SELF_TEST: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma3 NPU model execution wiring manifest")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model-variant", choices=sorted(MODEL_SPECS), default="gemma3-1b")
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--include-stages", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return 0
    plan = build_wiring_plan(args.model_variant, weights_dir=args.weights_dir)
    if args.json:
        print(json.dumps(plan.to_json_dict(), indent=2, sort_keys=True))
    else:
        print(plan.format(include_stages=args.include_stages))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
