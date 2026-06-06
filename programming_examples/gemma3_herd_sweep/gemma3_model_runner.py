#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gemma3 real-model runner launch plan.

This composes the real-shape preflight, static-weight plan, BO plan, capped XRT
allocation smoke contract, and per-layer NPU wiring into one launch-order
manifest. It still does not execute model kernels; the remaining blocker after
this module is kernel launch and validation, not the absence of a runner plan.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from gemma3_artifacts import MODEL_SPECS, model_spec
from gemma3_bo_plan import KV_STRATEGIES, Gemma3BOPlan, Gemma3BORecord, build_bo_plan_from_preflight
from gemma3_argument_binding import Gemma3KernelArgumentBindingPlan, build_argument_binding_plan_from_components
from gemma3_buffer_binding import Gemma3BufferBindingPlan, build_buffer_binding_plan_from_components
from gemma3_npu_preflight import Gemma3NPUPreflightPlan, ProjectionPlan, build_preflight_plan
from gemma3_npu_wiring import (
    LOGITS_SAMPLING_BLOCKER,
    LOGITS_SAMPLING_HOST_DIAGNOSTIC_BLOCKER,
    MODEL_FULL_1B_LOOP_BLOCKER,
    MODEL_FULL_LAYER_BLOCKER,
    MODEL_FULL_QKV_SUBSTEP_BLOCKER,
    MODEL_KERNEL_LAUNCH_BLOCKER,
    MODEL_SUBSTEP_SEQUENCE_BLOCKER,
    NPU_ATTENTION_REDUCTION_BLOCKER,
    NPU_PREFILL_KV_CACHE_BLOCKER,
    PREFILL_1K_NPU_BLOCKER,
    PREFILL_PRODUCED_KV_CACHE_BLOCKER,
    PRODUCTION_STATIC_BO_BLOCKER,
    Gemma3NPUWiringPlan,
    build_wiring_plan_from_preflight,
)
from gemma3_norm_weight_plan import Gemma3NormWeightPlan, build_norm_weight_plan
from gemma3_weight_plan import (
    Gemma3ProjectionWeightRecord,
    Gemma3StaticWeightPlan,
    Gemma3WeightFamilySummary,
    build_weight_plan,
)
from gemma3_static_preload import has_full_xrt_preload_evidence
from gemma3_xrt_runner import (
    Gemma3XRTRunnerReport,
    allocate_smoke,
    dry_run_allocation_plan,
    has_paper_shape_bo_allocation_evidence,
)


MODEL_RUNNER_BLOCKERS = (
    "model-kernel-launch-not-wired",
    "full-static-weight-bo-preload-not-validated",
    "paper-shape-bo-allocation-not-validated",
    "nonlinear-model-stage-promotion-incomplete",
    "paper-shape-hardware-rerun-required",
)
MODEL_RUNNER_SPECIFIC_WIRING_BLOCKERS = (
    PREFILL_1K_NPU_BLOCKER,
    PREFILL_PRODUCED_KV_CACHE_BLOCKER,
    NPU_PREFILL_KV_CACHE_BLOCKER,
    NPU_ATTENTION_REDUCTION_BLOCKER,
    LOGITS_SAMPLING_BLOCKER,
    LOGITS_SAMPLING_HOST_DIAGNOSTIC_BLOCKER,
    PRODUCTION_STATIC_BO_BLOCKER,
)


@dataclass(frozen=True)
class Gemma3RunnerStep:
    step_index: int
    phase: str
    layer_index: int | None
    role: str
    action: str
    backend: str
    kernel: str
    route: str
    bytes: int
    status: str
    blockers: tuple[str, ...]

    def format(self) -> str:
        layer = "model" if self.layer_index is None else f"L{self.layer_index}"
        blockers = ",".join(self.blockers) if self.blockers else "none"
        return (
            f"runner_step {self.step_index} phase={self.phase} layer={layer} "
            f"role={self.role} action={self.action} backend={self.backend} "
            f"kernel={self.kernel} route={self.route} bytes={self.bytes} "
            f"status={self.status} blockers={blockers}"
        )


@dataclass(frozen=True)
class Gemma3ModelRunnerPlan:
    model_variant: str
    status: str
    layers: int
    prompt_len: int
    decode_context: int
    step_count: int
    bo_allocation_status: str
    bo_requested_bytes: int
    bo_allocated_bytes: int
    bo_allocation_count: int
    bo_skipped_count: int
    static_preload_tensor_count: int
    buffer_binding_status: str
    buffer_binding_count: int
    virtual_buffer_count: int
    kernel_argument_binding_status: str
    kernel_argument_binding_count: int
    kernel_argument_binding_blocker_count: int
    kernel_launch_count: int
    host_fallback_count: int
    host_runtime_count: int
    blockers: tuple[str, ...]
    steps: tuple[Gemma3RunnerStep, ...]

    def format(self, *, include_steps: bool = False) -> str:
        blockers = ",".join(self.blockers) if self.blockers else "none"
        lines = [
            f"model_runner model={self.model_variant} status={self.status} "
            f"layers={self.layers} prompt_len={self.prompt_len} "
            f"decode_context={self.decode_context} steps={self.step_count} "
            f"bo_status={self.bo_allocation_status} bo_requested={self.bo_requested_bytes} "
            f"bo_allocated={self.bo_allocated_bytes} bo_allocations={self.bo_allocation_count} "
            f"bo_skipped={self.bo_skipped_count} static_preload_tensors={self.static_preload_tensor_count} "
            f"buffer_status={self.buffer_binding_status} buffer_bindings={self.buffer_binding_count} "
            f"virtual_buffers={self.virtual_buffer_count} argument_binding_status={self.kernel_argument_binding_status} "
            f"argument_bindings={self.kernel_argument_binding_count} argument_binding_blockers={self.kernel_argument_binding_blocker_count} "
            f"kernel_launches={self.kernel_launch_count} host_fallbacks={self.host_fallback_count} "
            f"host_runtime={self.host_runtime_count} blockers={blockers}"
        ]
        if include_steps:
            lines.extend(step.format() for step in self.steps)
        return "\n".join(lines)

    def to_json_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["steps"] = [asdict(step) for step in self.steps]
        return data


def _dedupe(items: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item for item in items if item))


def _bo_steps(report: Gemma3XRTRunnerReport) -> list[Gemma3RunnerStep]:
    steps = []
    for result in report.results:
        blockers = ()
        if result.status in ("SKIPPED_TOTAL_CAP", "ALLOCATED_TRUNCATED", "PLANNED_TRUNCATED"):
            blockers = ("paper-shape-bo-allocation-not-validated",)
        steps.append(
            Gemma3RunnerStep(
                step_index=0,
                phase="setup",
                layer_index=None,
                role=f"allocate_bo:{result.name}",
                action="allocate_bo" if report.status != "DRY_RUN" else "plan_bo",
                backend="xrt" if report.status != "DRY_RUN" else "dry-run",
                kernel="host",
                route="host",
                bytes=result.allocated_bytes,
                status=result.status,
                blockers=blockers,
            )
        )
    return steps


def _static_preload_steps(
    weight_plan: Gemma3StaticWeightPlan,
    *,
    max_static_tensors: int,
) -> list[Gemma3RunnerStep]:
    steps = []
    for record in weight_plan.records[:max_static_tensors]:
        steps.append(
            Gemma3RunnerStep(
                step_index=0,
                phase="setup",
                layer_index=record.layer_index,
                role=f"static_preload:{record.family}",
                action="serialize_static_q4nx",
                backend="host-to-xrt",
                kernel="q4nx_static_weight_pack",
                route="host",
                bytes=record.static_bo_bytes,
                status="STATIC_PRELOAD_PLANNED",
                blockers=("full-static-weight-bo-preload-not-validated",),
            )
        )
    return steps


def _wiring_steps(wiring: Gemma3NPUWiringPlan) -> list[Gemma3RunnerStep]:
    steps = []
    for stage in wiring.stages:
        if stage.backend == "npu-candidate":
            action = "launch_kernel"
        elif stage.backend == "host-runtime":
            action = "bind_runtime_buffer"
        else:
            action = "run_host_fallback"
        steps.append(
            Gemma3RunnerStep(
                step_index=0,
                phase=stage.phase,
                layer_index=stage.layer_index,
                role=stage.role,
                action=action,
                backend=stage.backend,
                kernel=stage.kernel,
                route=stage.route,
                bytes=0,
                status=stage.status,
                blockers=stage.blockers,
            )
        )
    return steps


def build_model_runner_plan_from_components(
    *,
    model_variant: str,
    bo_plan: Gemma3BOPlan,
    weight_plan: Gemma3StaticWeightPlan,
    wiring: Gemma3NPUWiringPlan,
    buffer_binding_plan: Gemma3BufferBindingPlan,
    argument_binding_plan: Gemma3KernelArgumentBindingPlan,
    bo_report: Gemma3XRTRunnerReport,
    max_static_tensors: int = 4,
    static_preload_validated: bool = False,
    bo_allocation_validated: bool = False,
) -> Gemma3ModelRunnerPlan:
    if max_static_tensors <= 0:
        raise ValueError("max_static_tensors must be positive")
    steps = (
        _bo_steps(bo_report)
        + _static_preload_steps(weight_plan, max_static_tensors=max_static_tensors)
        + _wiring_steps(wiring)
    )
    numbered_steps = [
        Gemma3RunnerStep(
            step_index=index,
            phase=step.phase,
            layer_index=step.layer_index,
            role=step.role,
            action=step.action,
            backend=step.backend,
            kernel=step.kernel,
            route=step.route,
            bytes=step.bytes,
            status=step.status,
            blockers=step.blockers,
        )
        for index, step in enumerate(steps)
    ]
    blockers = list(MODEL_RUNNER_BLOCKERS)
    specific_wiring_blockers = [
        blocker for blocker in MODEL_RUNNER_SPECIFIC_WIRING_BLOCKERS if blocker in wiring.blockers
    ]
    if specific_wiring_blockers:
        blockers = [
            *specific_wiring_blockers,
            *(
                blocker
                for blocker in blockers
                if blocker not in (MODEL_KERNEL_LAUNCH_BLOCKER, "paper-shape-hardware-rerun-required")
            ),
        ]
    elif MODEL_FULL_1B_LOOP_BLOCKER in wiring.blockers:
        blockers = [
            MODEL_FULL_1B_LOOP_BLOCKER
            if blocker == MODEL_KERNEL_LAUNCH_BLOCKER
            else blocker
            for blocker in blockers
        ]
    elif MODEL_FULL_LAYER_BLOCKER in wiring.blockers:
        blockers = [
            MODEL_FULL_LAYER_BLOCKER
            if blocker == MODEL_KERNEL_LAUNCH_BLOCKER
            else blocker
            for blocker in blockers
        ]
    elif MODEL_FULL_QKV_SUBSTEP_BLOCKER in wiring.blockers:
        blockers = [
            MODEL_FULL_QKV_SUBSTEP_BLOCKER
            if blocker == MODEL_KERNEL_LAUNCH_BLOCKER
            else blocker
            for blocker in blockers
        ]
    elif MODEL_SUBSTEP_SEQUENCE_BLOCKER in wiring.blockers:
        blockers = [
            MODEL_SUBSTEP_SEQUENCE_BLOCKER
            if blocker == MODEL_KERNEL_LAUNCH_BLOCKER
            else blocker
            for blocker in blockers
        ]
    if static_preload_validated:
        blockers = [
            blocker
            for blocker in blockers
            if blocker != "full-static-weight-bo-preload-not-validated"
        ]
    if bo_allocation_validated:
        blockers = [
            blocker
            for blocker in blockers
            if blocker != "paper-shape-bo-allocation-not-validated"
        ]
    if "nonlinear-model-stage-promotion-incomplete" not in wiring.blockers:
        blockers = [
            blocker
            for blocker in blockers
            if blocker != "nonlinear-model-stage-promotion-incomplete"
        ]
    if argument_binding_plan.blockers:
        blockers.extend(argument_binding_plan.blockers)
    if model_variant.endswith("vision"):
        blockers.append("vision-npu-path-not-implemented")
    return Gemma3ModelRunnerPlan(
        model_variant=model_variant,
        status="BLOCKED",
        layers=wiring.layers,
        prompt_len=bo_plan.prompt_len,
        decode_context=bo_plan.decode_context,
        step_count=len(numbered_steps),
        bo_allocation_status=bo_report.status,
        bo_requested_bytes=bo_report.requested_bytes,
        bo_allocated_bytes=bo_report.allocated_bytes,
        bo_allocation_count=bo_report.allocation_count,
        bo_skipped_count=bo_report.skipped_count,
        static_preload_tensor_count=min(max_static_tensors, len(weight_plan.records)),
        buffer_binding_status=buffer_binding_plan.status,
        buffer_binding_count=buffer_binding_plan.binding_count,
        virtual_buffer_count=buffer_binding_plan.virtual_buffer_count,
        kernel_argument_binding_status=argument_binding_plan.status,
        kernel_argument_binding_count=argument_binding_plan.argument_binding_count,
        kernel_argument_binding_blocker_count=argument_binding_plan.missing_argument_count,
        kernel_launch_count=wiring.npu_candidate_count,
        host_fallback_count=wiring.host_fallback_count,
        host_runtime_count=wiring.host_runtime_count,
        blockers=_dedupe(blockers),
        steps=tuple(numbered_steps),
    )


def build_model_runner_plan(
    model_variant: str,
    *,
    weights_dir: Path | None = None,
    prompt_len: int | None = None,
    decode_context: int | None = None,
    kv_strategy: str = "benchmark-cell",
    max_static_tensors: int = 4,
    max_total_bytes: int = 64 * 1024 * 1024,
    max_bo_bytes: int = 8 * 1024 * 1024,
    allocate_bo_smoke: bool = False,
    device_index: int = 0,
) -> Gemma3ModelRunnerPlan:
    spec = model_spec(model_variant)
    prompt_len = prompt_len or spec.prefill_lengths[0]
    decode_context = decode_context or spec.max_decode_context
    preflight = build_preflight_plan(model_variant, weights_dir=weights_dir)
    weight_plan = build_weight_plan(model_variant, weights_dir=weights_dir)
    norm_weight_plan = build_norm_weight_plan(model_variant, weights_dir=weights_dir)
    bo_plan = build_bo_plan_from_preflight(
        preflight,
        weight_plan,
        norm_weight_plan,
        prompt_len=prompt_len,
        decode_context=decode_context,
        kv_strategy=kv_strategy,
    )
    static_preload_validated = has_full_xrt_preload_evidence(model_variant)
    bo_allocation_validated = has_paper_shape_bo_allocation_evidence(model_variant)
    wiring = build_wiring_plan_from_preflight(
        preflight,
        use_static_preload_evidence=static_preload_validated,
        use_bo_allocation_evidence=bo_allocation_validated,
        use_first_kernel_launch_evidence=True,
        use_decode_q_projection_substep_evidence=True,
        use_decode_qkv_substep_evidence=True,
        use_decode_full_layer_evidence=True,
        use_decode_loop_tiled_stats_evidence=True,
    )
    buffer_binding_plan = build_buffer_binding_plan_from_components(
        model_variant=model_variant,
        bo_plan=bo_plan,
        weight_plan=weight_plan,
        wiring=wiring,
    )
    argument_binding_plan = build_argument_binding_plan_from_components(
        model_variant=model_variant,
        preflight=preflight,
        bo_plan=bo_plan,
        wiring=wiring,
        buffer_binding_plan=buffer_binding_plan,
    )
    if allocate_bo_smoke:
        bo_report = allocate_smoke(
            bo_plan,
            device_index=device_index,
            max_total_bytes=max_total_bytes,
            max_bo_bytes=max_bo_bytes,
            preload_static=True,
            preload_dynamic=False,
        )
    else:
        bo_report = dry_run_allocation_plan(
            bo_plan,
            max_total_bytes=max_total_bytes,
            max_bo_bytes=max_bo_bytes,
        )
    return build_model_runner_plan_from_components(
        model_variant=model_variant,
        bo_plan=bo_plan,
        weight_plan=weight_plan,
        wiring=wiring,
        buffer_binding_plan=buffer_binding_plan,
        argument_binding_plan=argument_binding_plan,
        bo_report=bo_report,
        max_static_tensors=max_static_tensors,
        static_preload_validated=static_preload_validated,
        bo_allocation_validated=bo_allocation_validated,
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
        layers=2,
        hidden_size=1152,
        intermediate_size=6912,
        head_dim=256,
        num_attention_heads=4,
        num_key_value_heads=1,
        sliding_window=512,
        attention_pattern="5-local-1-global",
        projections=(projection,),
    )


def _fake_norm_weight_plan() -> Gemma3NormWeightPlan:
    return Gemma3NormWeightPlan(
        model_variant="gemma3-1b",
        status="READY_FOR_NORM_WEIGHT_PRELOAD",
        layers=2,
        tensor_count=12,
        static_bo_bytes=20480,
        families=(),
        records=(),
        blockers=(),
    )


def _fake_weight_plan() -> Gemma3StaticWeightPlan:
    records = (
        Gemma3ProjectionWeightRecord(0, "q_proj", "model.layers.0.self_attn.q_proj.weight", (1024, 1152), (1024, 1280), 32, 5, 655360, 81920, 81920, 819200, True),
        Gemma3ProjectionWeightRecord(0, "k_proj", "model.layers.0.self_attn.k_proj.weight", (256, 1152), (256, 1280), 8, 5, 163840, 20480, 20480, 204800, True),
        Gemma3ProjectionWeightRecord(0, "v_proj", "model.layers.0.self_attn.v_proj.weight", (256, 1152), (256, 1280), 8, 5, 163840, 20480, 20480, 204800, True),
    )
    families = (
        Gemma3WeightFamilySummary("q_proj", 1, 819200, 1),
        Gemma3WeightFamilySummary("k_proj", 1, 204800, 1),
        Gemma3WeightFamilySummary("v_proj", 1, 204800, 1),
    )
    return Gemma3StaticWeightPlan(
        model_variant="gemma3-1b",
        status="READY_FOR_STATIC_BO_PRELOAD",
        layers=2,
        tensor_count=len(records),
        static_bo_bytes=sum(record.static_bo_bytes for record in records),
        families=families,
        records=records,
        blockers=(),
    )


def _self_test() -> None:
    preflight = _fake_preflight()
    weight_plan = _fake_weight_plan()
    bo_plan = Gemma3BOPlan(
        model_variant="gemma3-1b",
        status="READY_FOR_BO_ALLOCATION",
        layers=2,
        prompt_len=16,
        decode_context=16,
        kv_strategy="monolithic",
        kv_record_count=1,
        max_bo_bytes=4096,
        total_bytes=10240,
        dynamic_bytes=6144,
        static_bytes=4096,
        records=(
            Gemma3BORecord("static_projection_weights", "model", "q4nx", (4096,), 4096, True, "fixture"),
            Gemma3BORecord("layer_input", "layer", "bf16", (16, 64), 2048, False, "fixture"),
            Gemma3BORecord("kv_cache_k", "model", "bf16", (2, 16, 1, 64), 4096, False, "fixture"),
        ),
        blockers=("paper-shape-bo-allocation-not-validated",),
    )
    binding_bo_plan = build_bo_plan_from_preflight(preflight, weight_plan, _fake_norm_weight_plan(), prompt_len=16, decode_context=16)
    buffer_binding_plan = build_buffer_binding_plan_from_components(
        model_variant="gemma3-1b",
        bo_plan=binding_bo_plan,
        weight_plan=weight_plan,
        wiring=build_wiring_plan_from_preflight(preflight),
    )
    argument_binding_plan = build_argument_binding_plan_from_components(
        model_variant="gemma3-1b",
        preflight=preflight,
        bo_plan=binding_bo_plan,
        wiring=build_wiring_plan_from_preflight(preflight),
        buffer_binding_plan=buffer_binding_plan,
    )
    bo_report = dry_run_allocation_plan(bo_plan, max_total_bytes=6144, max_bo_bytes=4096)
    plan = build_model_runner_plan_from_components(
        model_variant="gemma3-1b",
        bo_plan=bo_plan,
        weight_plan=weight_plan,
        wiring=build_wiring_plan_from_preflight(preflight),
        buffer_binding_plan=buffer_binding_plan,
        argument_binding_plan=argument_binding_plan,
        bo_report=bo_report,
        max_static_tensors=2,
    )
    if plan.step_count != 65:
        raise AssertionError(plan.step_count)
    if plan.bo_allocation_count != 2 or plan.bo_skipped_count != 1:
        raise AssertionError((plan.bo_allocation_count, plan.bo_skipped_count))
    if plan.static_preload_tensor_count != 2:
        raise AssertionError(plan.static_preload_tensor_count)
    if plan.buffer_binding_status != "READY_FOR_MODEL_RUNNER" or plan.buffer_binding_count != 60:
        raise AssertionError(plan)
    if plan.kernel_argument_binding_status != "READY_FOR_KERNEL_LAUNCH" or plan.kernel_argument_binding_count != 56:
        raise AssertionError((plan.kernel_argument_binding_status, plan.kernel_argument_binding_count))
    if plan.kernel_argument_binding_blocker_count != 0:
        raise AssertionError(plan.kernel_argument_binding_blocker_count)
    if plan.kernel_launch_count != 56 or plan.host_fallback_count != 0 or plan.host_runtime_count != 4:
        raise AssertionError(plan)
    print(plan.format(include_steps=True))
    print("GEMMA3_MODEL_RUNNER_SELF_TEST: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma3 real-model runner launch plan")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model-variant", choices=sorted(MODEL_SPECS), default="gemma3-1b")
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--prompt-len", type=int)
    parser.add_argument("--decode-context", type=int)
    parser.add_argument("--kv-strategy", choices=KV_STRATEGIES, default="benchmark-cell")
    parser.add_argument("--max-static-tensors", type=int, default=4)
    parser.add_argument("--max-total-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-bo-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--allocate-bo-smoke", action="store_true")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--include-steps", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return 0
    plan = build_model_runner_plan(
        args.model_variant,
        weights_dir=args.weights_dir,
        prompt_len=args.prompt_len,
        decode_context=args.decode_context,
        kv_strategy=args.kv_strategy,
        max_static_tensors=args.max_static_tensors,
        max_total_bytes=args.max_total_bytes,
        max_bo_bytes=args.max_bo_bytes,
        allocate_bo_smoke=args.allocate_bo_smoke,
        device_index=args.device_index,
    )
    if args.json:
        print(json.dumps(plan.to_json_dict(), indent=2, sort_keys=True))
    else:
        print(plan.format(include_steps=args.include_steps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
