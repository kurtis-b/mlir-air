#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gemma3 model-loop runtime buffer binding plan.

The binding plan assigns persistent BO keys and virtual intermediate keys to
each per-layer prefill/decode stage. It does not bind compiled kernel argument
orders or launch kernels; that remains a separate validation blocker.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from gemma3_artifacts import MODEL_SPECS, model_spec
from gemma3_bo_plan import Gemma3BOPlan, build_bo_plan_from_preflight
from gemma3_npu_preflight import Gemma3NPUPreflightPlan, ProjectionPlan, build_preflight_plan
from gemma3_npu_wiring import Gemma3NPUStage, Gemma3NPUWiringPlan, build_wiring_plan_from_preflight
from gemma3_weight_plan import (
    Gemma3ProjectionWeightRecord,
    Gemma3StaticWeightPlan,
    Gemma3WeightFamilySummary,
    build_weight_plan,
)


PREFILL_STATE = "layer_input"
DECODE_STATE = "decode_token_state"


@dataclass(frozen=True)
class Gemma3BufferBinding:
    phase: str
    layer_index: int
    role: str
    backend: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    static_weight_families: tuple[str, ...]
    mutable_buffers: tuple[str, ...]
    virtual_buffers: tuple[str, ...]
    status: str
    blockers: tuple[str, ...]

    def format(self) -> str:
        inputs = ",".join(self.inputs) if self.inputs else "none"
        outputs = ",".join(self.outputs) if self.outputs else "none"
        weights = ",".join(self.static_weight_families) if self.static_weight_families else "none"
        mutable = ",".join(self.mutable_buffers) if self.mutable_buffers else "none"
        virtual = ",".join(self.virtual_buffers) if self.virtual_buffers else "none"
        blockers = ",".join(self.blockers) if self.blockers else "none"
        return (
            f"buffer_binding {self.phase}:L{self.layer_index}:{self.role} "
            f"backend={self.backend} inputs={inputs} outputs={outputs} "
            f"weights={weights} mutable={mutable} virtual={virtual} "
            f"status={self.status} blockers={blockers}"
        )


@dataclass(frozen=True)
class Gemma3BufferBindingPlan:
    model_variant: str
    status: str
    layers: int
    binding_count: int
    persistent_bo_count: int
    virtual_buffer_count: int
    static_weight_family_count: int
    missing_bo_keys: tuple[str, ...]
    blockers: tuple[str, ...]
    bindings: tuple[Gemma3BufferBinding, ...]

    def format(self, *, include_bindings: bool = False) -> str:
        missing = ",".join(self.missing_bo_keys) if self.missing_bo_keys else "none"
        blockers = ",".join(self.blockers) if self.blockers else "none"
        lines = [
            f"buffer_plan model={self.model_variant} status={self.status} "
            f"layers={self.layers} bindings={self.binding_count} "
            f"persistent_bos={self.persistent_bo_count} virtual_buffers={self.virtual_buffer_count} "
            f"static_weight_families={self.static_weight_family_count} "
            f"missing_bos={missing} blockers={blockers}"
        ]
        if include_bindings:
            lines.extend(binding.format() for binding in self.bindings)
        return "\n".join(lines)

    def to_json_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["bindings"] = [asdict(binding) for binding in self.bindings]
        return data


def _state_key(phase: str) -> str:
    return PREFILL_STATE if phase == "prefill" else DECODE_STATE


def _prefix(stage: Gemma3NPUStage) -> str:
    return f"{stage.phase}_L{stage.layer_index}_{stage.role}"


def _role_contract(stage: Gemma3NPUStage) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    state = _state_key(stage.phase)
    q = f"{stage.phase}_q"
    k = f"{stage.phase}_k"
    v = f"{stage.phase}_v"
    attention_out = f"{stage.phase}_attention_out"
    norm = f"{stage.phase}_L{stage.layer_index}_pre_attention_norm"
    post_norm = f"{stage.phase}_L{stage.layer_index}_post_attention_norm"
    activation = f"{stage.phase}_L{stage.layer_index}_mlp_activation"
    attn_proj = f"{stage.phase}_L{stage.layer_index}_attention_projection"

    if stage.role == "pre_attention_norm":
        return (state,), (norm,), (), ()
    if stage.role == "qkv_projection":
        return (norm,), (q, k, v), ("q_proj", "k_proj", "v_proj"), ()
    if stage.role == "rope":
        return (q, k), (q, k), (), ()
    if stage.role == "qk_norm":
        return (q, k), (q, k), (), ()
    if stage.role == "kv_cache_append":
        return (k, v), ("kv_cache_k", "kv_cache_v"), (), ("kv_cache_k", "kv_cache_v")
    if stage.role == "attention":
        return (q, "kv_cache_k", "kv_cache_v"), (attention_out,), (), ()
    if stage.role == "output_projection":
        return (attention_out,), (attn_proj,), ("o_proj",), ()
    if stage.role == "attention_residual":
        return (state, attn_proj), (state,), (), ()
    if stage.role == "post_attention_norm":
        return (state,), (post_norm,), (), ()
    if stage.role == "mlp_gate_up_projection":
        return (post_norm,), ("mlp_gate", "mlp_up"), ("gate_proj", "up_proj"), ()
    if stage.role == "mlp_activation":
        return ("mlp_gate", "mlp_up"), (activation,), (), ()
    if stage.role == "mlp_down_projection":
        return (activation,), ("mlp_down",), ("down_proj",), ()
    if stage.role == "mlp_residual":
        return (state, "mlp_down"), (state,), (), ()
    raise ValueError(f"unhandled Gemma3 stage role: {stage.role}")


def _stage_binding(stage: Gemma3NPUStage, bo_keys: set[str]) -> Gemma3BufferBinding:
    inputs, outputs, weights, mutable = _role_contract(stage)
    virtual = tuple(
        dict.fromkeys(
            key
            for key in inputs + outputs
            if key not in bo_keys and not key.startswith("kv_cache_")
        )
    )
    blockers = ()
    status = "BUFFER_BINDING_PLANNED"
    if stage.backend == "npu-candidate":
        blockers = ("model-kernel-argument-binding-not-validated",)
        status = "KERNEL_ARGUMENT_BINDING_PLANNED"
    return Gemma3BufferBinding(
        phase=stage.phase,
        layer_index=stage.layer_index,
        role=stage.role,
        backend=stage.backend,
        inputs=inputs,
        outputs=outputs,
        static_weight_families=weights,
        mutable_buffers=mutable,
        virtual_buffers=virtual,
        status=status,
        blockers=blockers,
    )


def build_buffer_binding_plan_from_components(
    *,
    model_variant: str,
    bo_plan: Gemma3BOPlan,
    weight_plan: Gemma3StaticWeightPlan,
    wiring: Gemma3NPUWiringPlan,
) -> Gemma3BufferBindingPlan:
    bo_keys = {record.name for record in bo_plan.records}
    bindings = tuple(_stage_binding(stage, bo_keys) for stage in wiring.stages)
    referenced_bo_keys = {
        key
        for binding in bindings
        for key in binding.inputs + binding.outputs + binding.mutable_buffers
        if key in bo_keys or key.startswith("kv_cache_")
    }
    missing = tuple(sorted(key for key in referenced_bo_keys if key not in bo_keys))
    virtual = tuple(
        sorted(
            {
                key
                for binding in bindings
                for key in binding.virtual_buffers
            }
        )
    )
    blockers = []
    if missing:
        blockers.append("missing-runtime-bo-binding")
    if any(binding.blockers for binding in bindings):
        blockers.append("model-kernel-argument-binding-not-validated")
    return Gemma3BufferBindingPlan(
        model_variant=model_variant,
        status="READY_FOR_MODEL_RUNNER" if not missing else "BLOCKED",
        layers=wiring.layers,
        binding_count=len(bindings),
        persistent_bo_count=len(bo_keys),
        virtual_buffer_count=len(virtual),
        static_weight_family_count=len({family.family for family in weight_plan.families}),
        missing_bo_keys=missing,
        blockers=tuple(dict.fromkeys(blockers)),
        bindings=bindings,
    )


def build_buffer_binding_plan(
    model_variant: str,
    *,
    weights_dir: Path | None = None,
    prompt_len: int | None = None,
    decode_context: int | None = None,
) -> Gemma3BufferBindingPlan:
    spec = model_spec(model_variant)
    prompt_len = prompt_len or spec.prefill_lengths[0]
    decode_context = decode_context or spec.max_decode_context
    preflight = build_preflight_plan(model_variant, weights_dir=weights_dir)
    weight_plan = build_weight_plan(model_variant, weights_dir=weights_dir)
    bo_plan = build_bo_plan_from_preflight(
        preflight,
        weight_plan,
        prompt_len=prompt_len,
        decode_context=decode_context,
    )
    return build_buffer_binding_plan_from_components(
        model_variant=model_variant,
        bo_plan=bo_plan,
        weight_plan=weight_plan,
        wiring=build_wiring_plan_from_preflight(preflight),
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


def _fake_weight_plan() -> Gemma3StaticWeightPlan:
    records = (
        Gemma3ProjectionWeightRecord(0, "q_proj", "fixture.q", (1024, 1152), (1024, 1280), 32, 5, 655360, 81920, 81920, 819200, True),
        Gemma3ProjectionWeightRecord(0, "k_proj", "fixture.k", (256, 1152), (256, 1280), 8, 5, 163840, 20480, 20480, 204800, True),
        Gemma3ProjectionWeightRecord(0, "v_proj", "fixture.v", (256, 1152), (256, 1280), 8, 5, 163840, 20480, 20480, 204800, True),
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
    bo_plan = build_bo_plan_from_preflight(preflight, weight_plan, prompt_len=16, decode_context=16)
    plan = build_buffer_binding_plan_from_components(
        model_variant="gemma3-1b",
        bo_plan=bo_plan,
        weight_plan=weight_plan,
        wiring=build_wiring_plan_from_preflight(preflight),
    )
    if plan.binding_count != 52:
        raise AssertionError(plan.binding_count)
    if plan.missing_bo_keys:
        raise AssertionError(plan.missing_bo_keys)
    if plan.static_weight_family_count != 3:
        raise AssertionError(plan.static_weight_family_count)
    if "model-kernel-argument-binding-not-validated" not in plan.blockers:
        raise AssertionError(plan.blockers)
    print(plan.format(include_bindings=True))
    print("GEMMA3_BUFFER_BINDING_SELF_TEST: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma3 model-loop buffer binding plan")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model-variant", choices=sorted(MODEL_SPECS), default="gemma3-1b")
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--prompt-len", type=int)
    parser.add_argument("--decode-context", type=int)
    parser.add_argument("--include-bindings", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return 0
    plan = build_buffer_binding_plan(
        args.model_variant,
        weights_dir=args.weights_dir,
        prompt_len=args.prompt_len,
        decode_context=args.decode_context,
    )
    if args.json:
        print(json.dumps(plan.to_json_dict(), indent=2, sort_keys=True))
    else:
        print(plan.format(include_bindings=args.include_bindings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
