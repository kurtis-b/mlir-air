#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Production-shaped Gemma3 NPU inference runtime shell.

This module owns the public Gemma3 NPU runtime entrypoints. It prepares real
model metadata, Q4NX manifests, static-weight plans, BO plans, wiring, and
kernel argument bindings before timed model inference. Runtime entrypoints then
report measured NPU work together with the remaining paper blockers instead of
hiding diagnostic cache or host-fallback paths.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Sequence

from gemma3.core.artifacts import MODEL_SPECS, discover_model_artifacts, model_spec
from gemma3.core.quantized_weights import ensure_q4nx_cache, manifest_path_for, manifest_sha256
from gemma3.npu.argument_binding import (
    Gemma3KernelArgumentBindingPlan,
    build_argument_binding_plan_from_components,
)
from gemma3.npu.bo_plan import KV_STRATEGIES, Gemma3BOPlan, build_bo_plan_from_preflight
from gemma3.npu.buffer_binding import (
    Gemma3BufferBindingPlan,
    build_buffer_binding_plan_from_components,
)
from gemma3.npu.model_runner import Gemma3ModelRunnerPlan, build_model_runner_plan_from_components
from gemma3.npu.norm_weight_plan import Gemma3NormWeightPlan, build_norm_weight_plan
from gemma3.npu.preflight import Gemma3NPUPreflightPlan, build_preflight_plan
from gemma3.npu.static_preload import has_full_xrt_preload_evidence
from gemma3.npu.weight_plan import Gemma3StaticWeightPlan, build_weight_plan
from gemma3.npu.wiring import (
    LOGITS_SAMPLING_BLOCKER,
    LOGITS_SAMPLING_HOST_DIAGNOSTIC_BLOCKER,
    NPU_ATTENTION_REDUCTION_BLOCKER,
    NPU_PREFILL_KV_CACHE_BLOCKER,
    PREFILL_1K_NPU_BLOCKER,
    PREFILL_PRODUCED_KV_CACHE_BLOCKER,
    PRODUCTION_STATIC_BO_BLOCKER,
    Gemma3NPUWiringPlan,
    build_wiring_plan_from_preflight,
)
from gemma3.npu.xrt_runner import dry_run_allocation_plan, has_paper_shape_bo_allocation_evidence


RUNTIME_SETUP_VERSION = 1


@dataclass(frozen=True)
class Gemma3RuntimeOwnershipRecord:
    name: str
    phase: str
    owner: str
    timed_window: bool
    status: str
    blockers: tuple[str, ...]

    def format(self) -> str:
        blockers = ",".join(self.blockers) if self.blockers else "none"
        timed = "timed" if self.timed_window else "setup"
        return (
            f"ownership name={self.name} phase={self.phase} owner={self.owner} "
            f"window={timed} status={self.status} blockers={blockers}"
        )


@dataclass(frozen=True)
class Gemma3RuntimeSetupRecord:
    schema_version: int
    model_variant: str
    status: str
    weights_dir: str | None
    config_path: str | None
    tokenizer_path: str | None
    prompt_len: int
    decode_context: int
    layers: int | None
    artifact_status: str
    quantized_weights_status: str
    q4nx_manifest: str | None
    q4nx_manifest_sha256: str | None
    projection_weight_source: str
    preflight_status: str | None
    weight_plan_status: str | None
    norm_weight_plan_status: str | None
    bo_plan_status: str | None
    bo_record_count: int
    bo_total_bytes: int
    static_input_keys: tuple[str, ...]
    persistent_output_keys: tuple[str, ...]
    virtual_output_count: int
    readback_policy: str
    buffer_binding_status: str | None
    buffer_binding_count: int
    argument_binding_status: str | None
    argument_binding_count: int
    argument_count: int
    argument_binding_blocker_count: int
    kernel_launch_count: int
    host_fallback_count: int
    host_runtime_count: int
    model_runner_status: str | None
    model_runner_step_count: int
    blockers: tuple[str, ...]
    operation_ownership: tuple[Gemma3RuntimeOwnershipRecord, ...]
    elapsed_setup_seconds: float
    error: str | None = None

    @property
    def ready_for_entrypoints(self) -> bool:
        return (
            self.artifact_status == "READY"
            and self.argument_binding_status == "READY_FOR_KERNEL_LAUNCH"
            and self.argument_binding_blocker_count == 0
        )

    def format(self, *, include_ownership: bool = False) -> str:
        blockers = ",".join(self.blockers) if self.blockers else "none"
        static_inputs = ",".join(self.static_input_keys) if self.static_input_keys else "none"
        outputs = ",".join(self.persistent_output_keys) if self.persistent_output_keys else "none"
        lines = [
            f"runtime model={self.model_variant} status={self.status} "
            f"layers={self.layers} prompt_len={self.prompt_len} "
            f"decode_context={self.decode_context} artifacts={self.artifact_status} "
            f"quantized_weights={self.quantized_weights_status} "
            f"projection_weight_source={self.projection_weight_source} "
            f"bo_records={self.bo_record_count} bo_total_bytes={self.bo_total_bytes} "
            f"static_inputs={static_inputs} persistent_outputs={outputs} "
            f"virtual_outputs={self.virtual_output_count} readback_policy={self.readback_policy} "
            f"buffer_status={self.buffer_binding_status} buffer_bindings={self.buffer_binding_count} "
            f"argument_binding_status={self.argument_binding_status} "
            f"argument_bindings={self.argument_binding_count} args={self.argument_count} "
            f"argument_binding_blockers={self.argument_binding_blocker_count} "
            f"kernel_launches={self.kernel_launch_count} host_fallbacks={self.host_fallback_count} "
            f"host_runtime={self.host_runtime_count} blockers={blockers}"
        ]
        if self.error:
            lines.append(f"runtime_error {self.error}")
        if include_ownership:
            lines.extend(record.format() for record in self.operation_ownership)
        return "\n".join(lines)

    def to_json_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["operation_ownership"] = [
            asdict(record) for record in self.operation_ownership
        ]
        data["ready_for_entrypoints"] = self.ready_for_entrypoints
        return data


@dataclass(frozen=True)
class Gemma3RuntimeExecutionResult:
    schema_version: int
    entrypoint: str
    model_variant: str
    status: str
    prompt_len: int
    decode_context: int
    decode_tokens: int
    generated_token_ids: tuple[int, ...]
    local_value: float | None
    unit: str | None
    blockers: tuple[str, ...]
    operation_ownership: tuple[Gemma3RuntimeOwnershipRecord, ...]
    elapsed_seconds: float | None = None
    setup_status: str | None = None
    q4nx_manifest: str | None = None
    q4nx_manifest_sha256: str | None = None
    static_input_keys: tuple[str, ...] = ()
    persistent_output_keys: tuple[str, ...] = ()
    readback_policy: str | None = None
    argument_binding_status: str | None = None
    argument_binding_count: int = 0
    argument_binding_blocker_count: int = 0
    kernel_launch_count: int = 0
    host_fallback_count: int = 0
    host_runtime_count: int = 0
    npu_decode_loop: dict[str, object] | None = None
    error: str | None = None

    def format(self, *, include_ownership: bool = False) -> str:
        blockers = ",".join(self.blockers) if self.blockers else "none"
        loop = self.npu_decode_loop or {}
        loop_status = loop.get("status", "none")
        lines = [
            f"runtime_result entrypoint={self.entrypoint} model={self.model_variant} "
            f"status={self.status} prompt_len={self.prompt_len} "
            f"decode_context={self.decode_context} decode_tokens={self.decode_tokens} "
            f"local={self.local_value} unit={self.unit} "
            f"kernel_launches={self.kernel_launch_count} host_fallbacks={self.host_fallback_count} "
            f"host_runtime={self.host_runtime_count} decode_loop_status={loop_status} "
            f"blockers={blockers}"
        ]
        if self.error:
            lines.append(f"runtime_error {self.error}")
        if include_ownership:
            lines.extend(record.format() for record in self.operation_ownership)
        return "\n".join(lines)

    def to_json_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["operation_ownership"] = [
            asdict(record) for record in self.operation_ownership
        ]
        return data


@dataclass(frozen=True)
class Gemma3RuntimeSession:
    setup: Gemma3RuntimeSetupRecord
    preflight: Gemma3NPUPreflightPlan | None = None
    weight_plan: Gemma3StaticWeightPlan | None = None
    norm_weight_plan: Gemma3NormWeightPlan | None = None
    bo_plan: Gemma3BOPlan | None = None
    wiring: Gemma3NPUWiringPlan | None = None
    buffer_binding_plan: Gemma3BufferBindingPlan | None = None
    argument_binding_plan: Gemma3KernelArgumentBindingPlan | None = None
    model_runner_plan: Gemma3ModelRunnerPlan | None = None


def _dedupe(items: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item for item in items if item))


def _artifact_blockers(inventory: Any) -> tuple[str, ...]:
    blockers: list[str] = []
    if not inventory.has_weight_files:
        blockers.append("missing-safetensors")
    if not inventory.config_exists:
        blockers.append("missing-config-json")
    if not inventory.tokenizer_exists:
        blockers.append("missing-tokenizer")
    if inventory.has_vision and not inventory.processor_exists:
        blockers.append("missing-processor")
    if not inventory.optional_packages.get("safetensors", False):
        blockers.append("missing-python-safetensors")
    if not any(
        inventory.optional_packages.get(pkg, False)
        for pkg in ("tokenizers", "sentencepiece", "transformers")
    ):
        blockers.append("missing-python-tokenizer-package")
    return _dedupe(blockers)


def _binding_keys(buffer_binding_plan: Gemma3BufferBindingPlan) -> tuple[tuple[str, ...], tuple[str, ...], int]:
    bo_keys: set[str] = set()
    static_keys: set[str] = set()
    persistent_outputs: set[str] = set()
    virtual_outputs: set[str] = set()
    for binding in buffer_binding_plan.bindings:
        static_keys.update(binding.static_weight_bos)
        bo_keys.update(binding.inputs)
        bo_keys.update(binding.outputs)
        bo_keys.update(binding.mutable_buffers)
        for key in binding.outputs:
            if key in binding.virtual_buffers:
                virtual_outputs.add(key)
            elif key.startswith("kv_cache_") or key in bo_keys:
                persistent_outputs.add(key)
    return tuple(sorted(static_keys)), tuple(sorted(persistent_outputs)), len(virtual_outputs)


def _blocked_ownership(
    *,
    name: str,
    phase: str,
    blockers: Iterable[str],
) -> Gemma3RuntimeOwnershipRecord:
    return Gemma3RuntimeOwnershipRecord(
        name=name,
        phase=phase,
        owner="missing",
        timed_window=True,
        status="blocked",
        blockers=_dedupe(blockers),
    )


def _ownership_or_ready(
    *,
    name: str,
    phase: str,
    blockers: Iterable[str],
    ready_owner: str = "npu",
    ready_status: str = "planned",
) -> Gemma3RuntimeOwnershipRecord:
    blocker_tuple = _dedupe(blockers)
    if blocker_tuple:
        return _blocked_ownership(name=name, phase=phase, blockers=blocker_tuple)
    return Gemma3RuntimeOwnershipRecord(
        name=name,
        phase=phase,
        owner=ready_owner,
        timed_window=True,
        status=ready_status,
        blockers=(),
    )


def _operation_ownership(plan: Gemma3ModelRunnerPlan | None) -> tuple[Gemma3RuntimeOwnershipRecord, ...]:
    if plan is None:
        return (
            _blocked_ownership(
                name="runtime_plan",
                phase="setup",
                blockers=("model-runner-plan-missing",),
            ),
        )
    blockers = set(plan.blockers)
    records = [
        Gemma3RuntimeOwnershipRecord(
            name="static_projection_preload",
            phase="setup",
            owner="host-fallback",
            timed_window=False,
            status="planned-outside-timed-window",
            blockers=tuple(
                blocker
                for blocker in plan.blockers
                if blocker == "full-static-weight-bo-preload-not-validated"
            ),
        ),
        Gemma3RuntimeOwnershipRecord(
            name="argument_binding_validation",
            phase="setup",
            owner="host-fallback",
            timed_window=False,
            status="validated" if plan.kernel_argument_binding_blocker_count == 0 else "blocked",
            blockers=tuple(
                blocker
                for blocker in plan.blockers
                if blocker == "model-kernel-argument-binding-not-validated"
            ),
        ),
        Gemma3RuntimeOwnershipRecord(
            name="planned_layer_kernels",
            phase="prefill+decode",
            owner="npu",
            timed_window=True,
            status="planned-not-launched",
            blockers=tuple(
                blocker
                for blocker in plan.blockers
                if blocker
                in (
                    "model-kernel-launch-not-wired",
                    "model-substep-sequence-not-wired",
                    "model-full-qkv-substep-not-wired",
                    "model-full-layer-not-wired",
                    "full-1b-loop-not-wired",
                    "production-contiguous-static-weight-bo-not-used-by-fused-dqp-route",
                )
            ),
        ),
        _ownership_or_ready(
            name="prefill_kv_producer",
            phase="prefill",
            blockers=(
                blocker
                for blocker in (
                    PREFILL_1K_NPU_BLOCKER,
                    PREFILL_PRODUCED_KV_CACHE_BLOCKER,
                    NPU_PREFILL_KV_CACHE_BLOCKER,
                )
                if blocker in blockers
            ),
        ),
        _ownership_or_ready(
            name="attention_stat_reduction",
            phase="decode",
            blockers=(
                (NPU_ATTENTION_REDUCTION_BLOCKER,)
                if NPU_ATTENTION_REDUCTION_BLOCKER in blockers
                else ()
            ),
        ),
        _ownership_or_ready(
            name="logits_sampling",
            phase="decode",
            blockers=(
                blocker
                for blocker in (LOGITS_SAMPLING_BLOCKER, LOGITS_SAMPLING_HOST_DIAGNOSTIC_BLOCKER)
                if blocker in blockers
            ),
            ready_owner="host-fallback",
            ready_status="timed-host-diagnostic-or-not-required",
        ),
    ]
    return tuple(records)


def _setup_record_from_components(
    *,
    model_variant: str,
    weights_dir: str | None,
    config_path: str | None,
    tokenizer_path: str | None,
    prompt_len: int,
    decode_context: int,
    artifact_status: str,
    quantized_weights_status: str,
    q4nx_manifest: str | None,
    q4nx_manifest_sha256: str | None,
    projection_weight_source: str,
    preflight: Gemma3NPUPreflightPlan | None,
    weight_plan: Gemma3StaticWeightPlan | None,
    norm_weight_plan: Gemma3NormWeightPlan | None,
    bo_plan: Gemma3BOPlan | None,
    buffer_binding_plan: Gemma3BufferBindingPlan | None,
    argument_binding_plan: Gemma3KernelArgumentBindingPlan | None,
    model_runner_plan: Gemma3ModelRunnerPlan | None,
    blockers: Iterable[str],
    elapsed_setup_seconds: float,
    error: str | None = None,
) -> Gemma3RuntimeSetupRecord:
    static_input_keys: tuple[str, ...] = ()
    persistent_output_keys: tuple[str, ...] = ()
    virtual_output_count = 0
    if buffer_binding_plan is not None:
        static_input_keys, persistent_output_keys, virtual_output_count = _binding_keys(buffer_binding_plan)
    all_blockers = _dedupe(
        list(blockers)
        + list(model_runner_plan.blockers if model_runner_plan is not None else ())
        + list(argument_binding_plan.blockers if argument_binding_plan is not None else ())
    )
    argument_status = argument_binding_plan.status if argument_binding_plan is not None else None
    argument_blockers = (
        argument_binding_plan.missing_argument_count
        if argument_binding_plan is not None
        else 0
    )
    if artifact_status != "READY":
        status = "BLOCKED_REAL_ARTIFACTS"
    elif error is not None:
        status = "SETUP_BLOCKED"
    elif argument_status == "READY_FOR_KERNEL_LAUNCH" and argument_blockers == 0:
        status = "SETUP_READY_EXECUTION_BLOCKED" if all_blockers else "READY"
    else:
        status = "SETUP_BLOCKED"
    return Gemma3RuntimeSetupRecord(
        schema_version=RUNTIME_SETUP_VERSION,
        model_variant=model_variant,
        status=status,
        weights_dir=weights_dir,
        config_path=config_path,
        tokenizer_path=tokenizer_path,
        prompt_len=prompt_len,
        decode_context=decode_context,
        layers=preflight.layers if preflight is not None else None,
        artifact_status=artifact_status,
        quantized_weights_status=quantized_weights_status,
        q4nx_manifest=q4nx_manifest,
        q4nx_manifest_sha256=q4nx_manifest_sha256,
        projection_weight_source=projection_weight_source,
        preflight_status=preflight.status if preflight is not None else None,
        weight_plan_status=weight_plan.status if weight_plan is not None else None,
        norm_weight_plan_status=norm_weight_plan.status if norm_weight_plan is not None else None,
        bo_plan_status=bo_plan.status if bo_plan is not None else None,
        bo_record_count=len(bo_plan.records) if bo_plan is not None else 0,
        bo_total_bytes=bo_plan.total_bytes if bo_plan is not None else 0,
        static_input_keys=static_input_keys,
        persistent_output_keys=persistent_output_keys,
        virtual_output_count=virtual_output_count,
        readback_policy="explicit-output-readback-only",
        buffer_binding_status=buffer_binding_plan.status if buffer_binding_plan is not None else None,
        buffer_binding_count=buffer_binding_plan.binding_count if buffer_binding_plan is not None else 0,
        argument_binding_status=argument_status,
        argument_binding_count=(
            argument_binding_plan.argument_binding_count
            if argument_binding_plan is not None
            else 0
        ),
        argument_count=argument_binding_plan.argument_count if argument_binding_plan is not None else 0,
        argument_binding_blocker_count=argument_blockers,
        kernel_launch_count=model_runner_plan.kernel_launch_count if model_runner_plan is not None else 0,
        host_fallback_count=model_runner_plan.host_fallback_count if model_runner_plan is not None else 0,
        host_runtime_count=model_runner_plan.host_runtime_count if model_runner_plan is not None else 0,
        model_runner_status=model_runner_plan.status if model_runner_plan is not None else None,
        model_runner_step_count=model_runner_plan.step_count if model_runner_plan is not None else 0,
        blockers=all_blockers,
        operation_ownership=_operation_ownership(model_runner_plan),
        elapsed_setup_seconds=elapsed_setup_seconds,
        error=error,
    )


def prepare_runtime(
    *,
    model_variant: str = "gemma3-1b",
    prompt_len: int = 1024,
    decode_context: int = 1024,
    weights_dir: Path | None = None,
    tokenizer: Path | None = None,
    kv_strategy: str = "benchmark-cell",
    max_static_tensors: int = 4,
    max_total_bytes: int = 64 * 1024 * 1024,
    max_bo_bytes: int = 8 * 1024 * 1024,
    quantized_weights: str = "required",
    quantized_weights_dir: Path | None = None,
    force_quantized_weights: bool = False,
) -> Gemma3RuntimeSession:
    """Prepare Gemma3 NPU runtime state without timed model inference."""
    if kv_strategy not in KV_STRATEGIES:
        raise ValueError(f"kv_strategy must be one of: {', '.join(KV_STRATEGIES)}")
    if quantized_weights not in ("required", "off"):
        raise ValueError("quantized_weights must be required or off")
    spec = model_spec(model_variant)
    spec.validate_sequence_length(prompt_len, phase="prefill")
    spec.validate_sequence_length(decode_context, phase="decode")
    start = perf_counter()
    inventory = discover_model_artifacts(
        model_variant,
        weights_dir=weights_dir,
        tokenizer=tokenizer,
    )
    artifact_status = "READY" if inventory.can_load_real_artifacts else "BLOCKED"
    if artifact_status != "READY":
        setup = _setup_record_from_components(
            model_variant=model_variant,
            weights_dir=inventory.weights_dir,
            config_path=inventory.config_path,
            tokenizer_path=inventory.tokenizer_path,
            prompt_len=prompt_len,
            decode_context=decode_context,
            artifact_status=artifact_status,
            quantized_weights_status="blocked",
            q4nx_manifest=None,
            q4nx_manifest_sha256=None,
            projection_weight_source="q4nx" if quantized_weights == "required" else "bf16-safetensors",
            preflight=None,
            weight_plan=None,
            norm_weight_plan=None,
            bo_plan=None,
            buffer_binding_plan=None,
            argument_binding_plan=None,
            model_runner_plan=None,
            blockers=_artifact_blockers(inventory),
            elapsed_setup_seconds=perf_counter() - start,
        )
        return Gemma3RuntimeSession(setup=setup)

    q4nx_manifest = None
    q4nx_manifest_hash = None
    quantized_weights_status = "off"
    projection_weight_source = "bf16-safetensors"
    try:
        if quantized_weights == "required":
            manifest = ensure_q4nx_cache(
                model_variant,
                weights_dir=weights_dir,
                quantized_weights_dir=quantized_weights_dir,
                force=force_quantized_weights,
            )
            q4nx_manifest_path = manifest_path_for(Path(manifest.quantized_weights_dir))
            q4nx_manifest = str(q4nx_manifest_path)
            q4nx_manifest_hash = manifest_sha256(q4nx_manifest_path)
            quantized_weights_status = manifest.status
            projection_weight_source = "q4nx"
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
        bo_report = dry_run_allocation_plan(
            bo_plan,
            max_total_bytes=max_total_bytes,
            max_bo_bytes=max_bo_bytes,
        )
        model_runner_plan = build_model_runner_plan_from_components(
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
            quantized_weights_status=quantized_weights_status,
            q4nx_manifest=q4nx_manifest,
            q4nx_manifest_sha256=q4nx_manifest_hash,
            projection_weight_source=projection_weight_source,
        )
    except Exception as exc:
        setup = _setup_record_from_components(
            model_variant=model_variant,
            weights_dir=inventory.weights_dir,
            config_path=inventory.config_path,
            tokenizer_path=inventory.tokenizer_path,
            prompt_len=prompt_len,
            decode_context=decode_context,
            artifact_status=artifact_status,
            quantized_weights_status=quantized_weights_status,
            q4nx_manifest=q4nx_manifest,
            q4nx_manifest_sha256=q4nx_manifest_hash,
            projection_weight_source=projection_weight_source,
            preflight=None,
            weight_plan=None,
            norm_weight_plan=None,
            bo_plan=None,
            buffer_binding_plan=None,
            argument_binding_plan=None,
            model_runner_plan=None,
            blockers=("npu-runtime-setup-failed",),
            elapsed_setup_seconds=perf_counter() - start,
            error=str(exc),
        )
        return Gemma3RuntimeSession(setup=setup)

    setup = _setup_record_from_components(
        model_variant=model_variant,
        weights_dir=inventory.weights_dir,
        config_path=inventory.config_path,
        tokenizer_path=inventory.tokenizer_path,
        prompt_len=prompt_len,
        decode_context=decode_context,
        artifact_status=artifact_status,
        quantized_weights_status=quantized_weights_status,
        q4nx_manifest=q4nx_manifest,
        q4nx_manifest_sha256=q4nx_manifest_hash,
        projection_weight_source=projection_weight_source,
        preflight=preflight,
        weight_plan=weight_plan,
        norm_weight_plan=norm_weight_plan,
        bo_plan=bo_plan,
        buffer_binding_plan=buffer_binding_plan,
        argument_binding_plan=argument_binding_plan,
        model_runner_plan=model_runner_plan,
        blockers=(),
        elapsed_setup_seconds=perf_counter() - start,
    )
    return Gemma3RuntimeSession(
        setup=setup,
        preflight=preflight,
        weight_plan=weight_plan,
        norm_weight_plan=norm_weight_plan,
        bo_plan=bo_plan,
        wiring=wiring,
        buffer_binding_plan=buffer_binding_plan,
        argument_binding_plan=argument_binding_plan,
        model_runner_plan=model_runner_plan,
    )


def _execution_setup_fields(session: Gemma3RuntimeSession) -> dict[str, object]:
    setup = session.setup
    return {
        "setup_status": setup.status,
        "q4nx_manifest": setup.q4nx_manifest,
        "q4nx_manifest_sha256": setup.q4nx_manifest_sha256,
        "static_input_keys": setup.static_input_keys,
        "persistent_output_keys": setup.persistent_output_keys,
        "readback_policy": setup.readback_policy,
        "argument_binding_status": setup.argument_binding_status,
        "argument_binding_count": setup.argument_binding_count,
        "argument_binding_blocker_count": setup.argument_binding_blocker_count,
    }


def _setup_ownership(session: Gemma3RuntimeSession) -> tuple[Gemma3RuntimeOwnershipRecord, ...]:
    return tuple(record for record in session.setup.operation_ownership if not record.timed_window)


def _decode_probe_blockers(session: Gemma3RuntimeSession, probe: Any) -> tuple[str, ...]:
    blockers: list[str] = list(session.setup.blockers)
    blockers.extend(getattr(probe, "blockers", ()))
    gap_map = {
        "prefill-kv-cache-not-constructed": NPU_PREFILL_KV_CACHE_BLOCKER,
        "prefill-produced-kv-cache-not-wired": PREFILL_PRODUCED_KV_CACHE_BLOCKER,
        "npu-prefill-kv-cache-not-wired": NPU_PREFILL_KV_CACHE_BLOCKER,
        "paper-1k-kv-attention-not-wired": "paper-1k-kv-attention-not-wired",
        "paper-1k-kv-attention-npu-reduction-not-wired": NPU_ATTENTION_REDUCTION_BLOCKER,
        "paper-1k-kv-attention-production-cache-and-npu-reduction-not-wired": NPU_ATTENTION_REDUCTION_BLOCKER,
        "logits-sampling-not-wired": LOGITS_SAMPLING_BLOCKER,
        "logits-sampling-host-diagnostic-only": LOGITS_SAMPLING_HOST_DIAGNOSTIC_BLOCKER,
        "production-contiguous-static-weight-bo-not-used-by-fused-dqp-route": PRODUCTION_STATIC_BO_BLOCKER,
    }
    for gap in getattr(probe, "remaining_paper_gaps", ()):  # keep unrecognized gaps visible.
        blockers.append(gap_map.get(gap, gap))
    return _dedupe(blockers)


def _probe_host_fallback_count(probe: Any) -> int:
    count = len(getattr(probe, "host_fallbacks", ()) or ())
    if getattr(probe, "attention_host_reduction", False):
        count += 1
    logits = getattr(probe, "logits_evidence", None)
    if isinstance(logits, dict) and str(logits.get("backend", "")).startswith("torch-cpu"):
        count += 1
    return count


def _decode_probe_ownership(
    session: Gemma3RuntimeSession,
    probe: Any,
) -> tuple[Gemma3RuntimeOwnershipRecord, ...]:
    records: list[Gemma3RuntimeOwnershipRecord] = list(_setup_ownership(session))
    probe_blockers = _dedupe(getattr(probe, "blockers", ()))
    records.append(
        Gemma3RuntimeOwnershipRecord(
            name="decode_layer_loop",
            phase="decode",
            owner="npu",
            timed_window=True,
            status="launched-pass" if not probe_blockers else "blocked",
            blockers=probe_blockers,
        )
    )
    cache_contract = str(getattr(probe, "attention_cache_contract", ""))
    if cache_contract == "single-current-token-kv":
        records.append(
            Gemma3RuntimeOwnershipRecord(
                name="decode_prefill_kv_cache",
                phase="decode",
                owner="missing",
                timed_window=True,
                status="single-token-diagnostic-cache",
                blockers=(NPU_PREFILL_KV_CACHE_BLOCKER, "paper-1k-kv-attention-not-wired"),
            )
        )
        records.append(
            Gemma3RuntimeOwnershipRecord(
                name="attention_stat_reduction",
                phase="decode",
                owner="missing",
                timed_window=True,
                status="not-run-for-single-token-attention",
                blockers=(NPU_ATTENTION_REDUCTION_BLOCKER,),
            )
        )
    else:
        if cache_contract == "synthetic-prefill-kv-cache":
            cache_owner = "host-fallback"
            cache_status = "diagnostic-synthetic-cache"
            cache_blockers = (PREFILL_PRODUCED_KV_CACHE_BLOCKER,)
        elif cache_contract == "host-hf-prefill-kv-cache":
            cache_owner = "host-fallback"
            cache_status = "diagnostic-host-hf-cache"
            cache_blockers = (NPU_PREFILL_KV_CACHE_BLOCKER,)
        else:
            cache_owner = "npu"
            cache_status = "provided"
            cache_blockers = ()
        records.append(
            Gemma3RuntimeOwnershipRecord(
                name="decode_prefill_kv_cache",
                phase="decode",
                owner=cache_owner,
                timed_window=False,
                status=cache_status,
                blockers=cache_blockers,
            )
        )
        reduction_host = bool(getattr(probe, "attention_host_reduction", False))
        records.append(
            Gemma3RuntimeOwnershipRecord(
                name="attention_stat_reduction",
                phase="decode",
                owner="host-fallback" if reduction_host else "npu",
                timed_window=True,
                status="timed-host-reduction" if reduction_host else "not-required-or-npu",
                blockers=(NPU_ATTENTION_REDUCTION_BLOCKER,) if reduction_host else (),
            )
        )
    logits = getattr(probe, "logits_evidence", None)
    if isinstance(logits, dict):
        timing = str(logits.get("timing_window", "host"))
        records.append(
            Gemma3RuntimeOwnershipRecord(
                name="logits_sampling",
                phase="decode",
                owner="host-fallback",
                timed_window=timing == "included-in-measured-loop-wall",
                status=timing,
                blockers=(
                    ()
                    if timing == "included-in-measured-loop-wall"
                    else (LOGITS_SAMPLING_HOST_DIAGNOSTIC_BLOCKER,)
                ),
            )
        )
    else:
        records.append(
            Gemma3RuntimeOwnershipRecord(
                name="logits_sampling",
                phase="decode",
                owner="missing",
                timed_window=True,
                status="not-wired",
                blockers=(LOGITS_SAMPLING_BLOCKER,),
            )
        )
    return tuple(records)


def _blocked_execution_result(
    session: Gemma3RuntimeSession,
    *,
    entrypoint: str,
    decode_tokens: int,
    blockers: Iterable[str],
    operation_ownership: tuple[Gemma3RuntimeOwnershipRecord, ...] | None = None,
    unit: str | None = None,
    error: str | None = None,
) -> Gemma3RuntimeExecutionResult:
    return Gemma3RuntimeExecutionResult(
        schema_version=RUNTIME_SETUP_VERSION,
        entrypoint=entrypoint,
        model_variant=session.setup.model_variant,
        status="BLOCKED",
        prompt_len=session.setup.prompt_len,
        decode_context=session.setup.decode_context,
        decode_tokens=decode_tokens,
        generated_token_ids=(),
        local_value=None,
        unit=unit,
        blockers=_dedupe(blockers),
        operation_ownership=(
            operation_ownership
            if operation_ownership is not None
            else session.setup.operation_ownership
        ),
        elapsed_seconds=None,
        kernel_launch_count=0,
        host_fallback_count=0,
        host_runtime_count=0,
        error=error,
        **_execution_setup_fields(session),
    )


def run_npu_prefill(
    session: Gemma3RuntimeSession,
    token_ids: Sequence[int],
) -> Gemma3RuntimeExecutionResult:
    if len(token_ids) != session.setup.prompt_len:
        raise ValueError(
            f"token_ids length {len(token_ids)} does not match prompt_len {session.setup.prompt_len}"
        )
    blockers = tuple(
        blocker
        for blocker in session.setup.blockers
        if blocker
        in (
            PREFILL_1K_NPU_BLOCKER,
            PREFILL_PRODUCED_KV_CACHE_BLOCKER,
            NPU_PREFILL_KV_CACHE_BLOCKER,
            "model-kernel-launch-not-wired",
            "full-static-weight-bo-preload-not-validated",
        )
    )
    if not blockers:
        blockers = ("prefill-runtime-launch-not-implemented",)
    ownership = tuple(
        record
        for record in session.setup.operation_ownership
        if record.phase in ("setup", "prefill", "prefill+decode")
    )
    return _blocked_execution_result(
        session,
        entrypoint="run_npu_prefill",
        decode_tokens=0,
        blockers=blockers,
        operation_ownership=ownership,
        unit="seconds",
    )


def generate(
    session: Gemma3RuntimeSession,
    prompt_or_token_ids: str | Sequence[int],
    *,
    decode_tokens: int,
    run_hardware: bool = True,
) -> Gemma3RuntimeExecutionResult:
    if decode_tokens < 0:
        raise ValueError("decode_tokens must be non-negative")
    if isinstance(prompt_or_token_ids, str):
        prompt_len = session.setup.prompt_len
    else:
        prompt_len = len(prompt_or_token_ids)
    if prompt_len != session.setup.prompt_len:
        raise ValueError(
            f"prompt length {prompt_len} does not match session prompt_len {session.setup.prompt_len}"
        )
    unit = "tokens_per_second" if decode_tokens else "seconds"
    if not session.setup.ready_for_entrypoints:
        return _blocked_execution_result(
            session,
            entrypoint="generate",
            decode_tokens=decode_tokens,
            blockers=session.setup.blockers or ("generate-runtime-setup-blocked",),
            unit=unit,
        )
    if decode_tokens == 0:
        return Gemma3RuntimeExecutionResult(
            schema_version=RUNTIME_SETUP_VERSION,
            entrypoint="generate",
            model_variant=session.setup.model_variant,
            status="READY_NO_DECODE_TOKENS",
            prompt_len=session.setup.prompt_len,
            decode_context=session.setup.decode_context,
            decode_tokens=0,
            generated_token_ids=(),
            local_value=None,
            unit=unit,
            blockers=session.setup.blockers,
            operation_ownership=session.setup.operation_ownership,
            elapsed_seconds=0.0,
            kernel_launch_count=0,
            host_fallback_count=0,
            host_runtime_count=0,
            **_execution_setup_fields(session),
        )
    if session.setup.model_variant != "gemma3-1b":
        return _blocked_execution_result(
            session,
            entrypoint="generate",
            decode_tokens=decode_tokens,
            blockers=("decode-runtime-gemma3-1b-only", *session.setup.blockers),
            unit=unit,
        )
    if not run_hardware:
        return _blocked_execution_result(
            session,
            entrypoint="generate",
            decode_tokens=decode_tokens,
            blockers=("generate-runtime-launch-not-run", *session.setup.blockers),
            unit=unit,
        )

    start = perf_counter()
    try:
        from gemma3.probes.decode_loop import run_decode_loop_runtime

        quantized_weights_dir = (
            Path(session.setup.q4nx_manifest).parent
            if session.setup.q4nx_manifest is not None
            else None
        )
        probe = run_decode_loop_runtime(
            model_variant=session.setup.model_variant,
            weights_dir=(Path(session.setup.weights_dir) if session.setup.weights_dir else None),
            layers=int(session.setup.layers or 26),
            decode_tokens=decode_tokens,
            prompt_context_length=session.setup.decode_context,
            warmup_layers=1,
            stitched=True,
            attention_mode="single-token",
            attention_cache_mode="repeated-current-token",
            logits_mode="none",
            logits_timing="excluded",
            quantized_weights_dir=quantized_weights_dir,
            force_quantized_weights=False,
            power_sample=False,
        )
    except Exception as exc:
        return _blocked_execution_result(
            session,
            entrypoint="generate",
            decode_tokens=decode_tokens,
            blockers=("decode-runtime-launch-failed", *session.setup.blockers),
            unit=unit,
            error=str(exc),
        )

    blockers = _decode_probe_blockers(session, probe)
    ownership = _decode_probe_ownership(session, probe)
    status = (
        "DECODE_RUNTIME_PASS_WITH_BLOCKERS"
        if getattr(probe, "status", "") == "DECODE_LOOP_DIAGNOSTIC_PASS"
        else "DECODE_RUNTIME_BLOCKED"
    )
    logits = getattr(probe, "logits_evidence", None)
    generated_token_ids = (
        (int(logits["sampled_token_id"]),)
        if isinstance(logits, dict) and "sampled_token_id" in logits
        else ()
    )
    elapsed = getattr(probe, "elapsed_seconds", None) or (perf_counter() - start)
    return Gemma3RuntimeExecutionResult(
        schema_version=RUNTIME_SETUP_VERSION,
        entrypoint="generate",
        model_variant=session.setup.model_variant,
        status=status,
        prompt_len=session.setup.prompt_len,
        decode_context=session.setup.decode_context,
        decode_tokens=decode_tokens,
        generated_token_ids=generated_token_ids,
        local_value=getattr(probe, "diagnostic_decode_tps_loop_wall", None),
        unit=unit,
        blockers=blockers,
        operation_ownership=ownership,
        elapsed_seconds=elapsed,
        kernel_launch_count=int(getattr(probe, "timed_kernel_count", 0) or 0),
        host_fallback_count=_probe_host_fallback_count(probe),
        host_runtime_count=0,
        npu_decode_loop=probe.to_json_dict(),
        **_execution_setup_fields(session),
    )


def _fixture_session() -> Gemma3RuntimeSession:
    from gemma3.npu.model_runner import _fake_norm_weight_plan, _fake_preflight, _fake_weight_plan

    preflight = _fake_preflight()
    weight_plan = _fake_weight_plan()
    norm_weight_plan = _fake_norm_weight_plan()
    bo_plan = build_bo_plan_from_preflight(
        preflight,
        weight_plan,
        norm_weight_plan,
        prompt_len=16,
        decode_context=16,
    )
    wiring = build_wiring_plan_from_preflight(preflight)
    buffer_binding_plan = build_buffer_binding_plan_from_components(
        model_variant="gemma3-1b",
        bo_plan=bo_plan,
        weight_plan=weight_plan,
        wiring=wiring,
    )
    argument_binding_plan = build_argument_binding_plan_from_components(
        model_variant="gemma3-1b",
        preflight=preflight,
        bo_plan=bo_plan,
        wiring=wiring,
        buffer_binding_plan=buffer_binding_plan,
    )
    bo_report = dry_run_allocation_plan(
        bo_plan,
        max_total_bytes=16 * 1024 * 1024,
        max_bo_bytes=8 * 1024 * 1024,
    )
    model_runner_plan = build_model_runner_plan_from_components(
        model_variant="gemma3-1b",
        bo_plan=bo_plan,
        weight_plan=weight_plan,
        wiring=wiring,
        buffer_binding_plan=buffer_binding_plan,
        argument_binding_plan=argument_binding_plan,
        bo_report=bo_report,
        max_static_tensors=2,
    )
    setup = _setup_record_from_components(
        model_variant="gemma3-1b",
        weights_dir="/tmp/gemma3-fixture",
        config_path="/tmp/gemma3-fixture/config.json",
        tokenizer_path="/tmp/gemma3-fixture/tokenizer.json",
        prompt_len=16,
        decode_context=16,
        artifact_status="READY",
        quantized_weights_status="not-checked",
        q4nx_manifest=None,
        q4nx_manifest_sha256=None,
        projection_weight_source="q4nx",
        preflight=preflight,
        weight_plan=weight_plan,
        norm_weight_plan=norm_weight_plan,
        bo_plan=bo_plan,
        buffer_binding_plan=buffer_binding_plan,
        argument_binding_plan=argument_binding_plan,
        model_runner_plan=model_runner_plan,
        blockers=(),
        elapsed_setup_seconds=0.0,
    )
    return Gemma3RuntimeSession(
        setup=setup,
        preflight=preflight,
        weight_plan=weight_plan,
        norm_weight_plan=norm_weight_plan,
        bo_plan=bo_plan,
        wiring=wiring,
        buffer_binding_plan=buffer_binding_plan,
        argument_binding_plan=argument_binding_plan,
        model_runner_plan=model_runner_plan,
    )


def _self_test() -> None:
    session = _fixture_session()
    setup = session.setup
    if setup.status != "SETUP_READY_EXECUTION_BLOCKED":
        raise AssertionError(setup.status)
    if setup.argument_binding_status != "READY_FOR_KERNEL_LAUNCH":
        raise AssertionError(setup.argument_binding_status)
    if setup.argument_binding_blocker_count != 0:
        raise AssertionError(setup.argument_binding_blocker_count)
    if setup.argument_binding_count != 56:
        raise AssertionError(setup.argument_binding_count)
    if "static_projection_weights" not in setup.static_input_keys:
        raise AssertionError(setup.static_input_keys)
    if not any(record.owner == "host-fallback" for record in setup.operation_ownership):
        raise AssertionError(setup.operation_ownership)
    prefill = run_npu_prefill(session, [1] * setup.prompt_len)
    if prefill.status != "BLOCKED" or not prefill.blockers:
        raise AssertionError(prefill)
    generated = generate(session, [1] * setup.prompt_len, decode_tokens=1, run_hardware=False)
    if generated.status != "BLOCKED" or generated.generated_token_ids:
        raise AssertionError(generated)
    if generated.kernel_launch_count != 0 or generated.setup_status != setup.status:
        raise AssertionError(generated)
    print(setup.format(include_ownership=True))
    print(prefill.format(include_ownership=True))
    print(generated.format(include_ownership=True))
    print("GEMMA3_NPU_RUNTIME_SELF_TEST: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma3 NPU inference runtime shell")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--prepare-runtime", action="store_true")
    mode.add_argument("--run-npu-prefill", action="store_true")
    mode.add_argument("--generate", action="store_true")
    parser.add_argument("--model-variant", choices=sorted(MODEL_SPECS), default="gemma3-1b")
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--prompt-len", type=int, default=1024)
    parser.add_argument("--decode-context", type=int, default=1024)
    parser.add_argument("--decode-tokens", type=int, default=1)
    parser.add_argument("--kv-strategy", choices=KV_STRATEGIES, default="benchmark-cell")
    parser.add_argument("--max-static-tensors", type=int, default=4)
    parser.add_argument("--max-total-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-bo-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--quantized-weights", choices=["required", "off"], default="required")
    parser.add_argument("--quantized-weights-dir", type=Path)
    parser.add_argument("--force-quantized-weights", action="store_true")
    parser.add_argument("--include-ownership", action="store_true")
    parser.add_argument("--no-run-hardware", action="store_true", help="prepare and format a blocked runtime result without launching NPU kernels")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return 0

    session = prepare_runtime(
        model_variant=args.model_variant,
        prompt_len=args.prompt_len,
        decode_context=args.decode_context,
        weights_dir=args.weights_dir,
        tokenizer=args.tokenizer,
        kv_strategy=args.kv_strategy,
        max_static_tensors=args.max_static_tensors,
        max_total_bytes=args.max_total_bytes,
        max_bo_bytes=args.max_bo_bytes,
        quantized_weights=args.quantized_weights,
        quantized_weights_dir=args.quantized_weights_dir,
        force_quantized_weights=args.force_quantized_weights,
    )
    result: Gemma3RuntimeSetupRecord | Gemma3RuntimeExecutionResult
    if args.run_npu_prefill:
        result = run_npu_prefill(session, [1] * args.prompt_len)
    elif args.generate:
        result = generate(
            session,
            [1] * args.prompt_len,
            decode_tokens=args.decode_tokens,
            run_hardware=not args.no_run_hardware,
        )
    else:
        result = session.setup

    if args.json:
        print(json.dumps(result.to_json_dict(), indent=2, sort_keys=True))
    else:
        print(result.format(include_ownership=args.include_ownership))
    if args.result_json:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(
            json.dumps(result.to_json_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"GEMMA3_NPU_RUNTIME_JSON: {args.result_json}")
    if isinstance(result, Gemma3RuntimeSetupRecord):
        print("GEMMA3_NPU_RUNTIME_PREPARE: ready" if result.ready_for_entrypoints else "GEMMA3_NPU_RUNTIME_PREPARE: blocked")
    else:
        print(f"GEMMA3_NPU_RUNTIME_{result.entrypoint.upper()}: {result.status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
