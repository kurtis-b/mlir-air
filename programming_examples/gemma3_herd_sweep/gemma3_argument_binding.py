#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gemma3 model-loop kernel argument binding validation.

This module turns the runtime buffer-binding manifest into deterministic
per-kernel positional argument layouts. It does not launch kernels and does not
claim compiled-kernel ABI parity; it validates that every NPU candidate has a
stable argument order, storage class, shape, dtype, and backing BO or virtual
buffer contract before launch wiring starts.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re

from gemma3_artifacts import MODEL_SPECS, model_spec
from gemma3_bo_plan import KV_STRATEGIES, Gemma3BOPlan, Gemma3BORecord, build_bo_plan_from_preflight
from gemma3_buffer_binding import Gemma3BufferBinding, Gemma3BufferBindingPlan, build_buffer_binding_plan_from_components
from gemma3_npu_preflight import Gemma3NPUPreflightPlan, ProjectionPlan, build_preflight_plan
from gemma3_npu_wiring import Gemma3NPUWiringPlan, build_wiring_plan_from_preflight
from gemma3_norm_weight_plan import Gemma3NormWeightPlan, build_norm_weight_plan, build_norm_weight_plan_from_shapes
from gemma3_weight_plan import Gemma3StaticWeightPlan, build_weight_plan, build_weight_plan_from_shapes

BF16_BYTES = 2
_LAYER_VIRTUAL_RE = re.compile(r"^(?P<phase>prefill|decode)_L(?P<layer>\d+)_(?P<name>.+)$")


@dataclass(frozen=True)
class Gemma3KernelArgument:
    arg_index: int
    key: str
    direction: str
    storage: str
    dtype: str
    shape: tuple[int, ...]
    bytes: int
    status: str
    blockers: tuple[str, ...]

    def format(self) -> str:
        shape = "x".join(str(dim) for dim in self.shape) if self.shape else "scalar"
        blockers = ",".join(self.blockers) if self.blockers else "none"
        return (
            f"arg {self.arg_index} key={self.key} direction={self.direction} "
            f"storage={self.storage} dtype={self.dtype} shape={shape} "
            f"bytes={self.bytes} status={self.status} blockers={blockers}"
        )


@dataclass(frozen=True)
class Gemma3KernelArgumentBinding:
    phase: str
    layer_index: int
    role: str
    kernel: str
    route: str
    argument_count: int
    status: str
    blockers: tuple[str, ...]
    arguments: tuple[Gemma3KernelArgument, ...]

    def format(self, *, include_arguments: bool = False) -> str:
        blockers = ",".join(self.blockers) if self.blockers else "none"
        lines = [
            f"argument_binding {self.phase}:L{self.layer_index}:{self.role} "
            f"kernel={self.kernel} route={self.route} args={self.argument_count} "
            f"status={self.status} blockers={blockers}"
        ]
        if include_arguments:
            lines.extend(argument.format() for argument in self.arguments)
        return "\n".join(lines)


@dataclass(frozen=True)
class Gemma3KernelArgumentBindingPlan:
    model_variant: str
    status: str
    layers: int
    prompt_len: int
    decode_context: int
    npu_candidate_count: int
    argument_binding_count: int
    argument_count: int
    missing_argument_count: int
    blockers: tuple[str, ...]
    bindings: tuple[Gemma3KernelArgumentBinding, ...]

    def format(self, *, include_bindings: bool = False, include_arguments: bool = False) -> str:
        blockers = ",".join(self.blockers) if self.blockers else "none"
        lines = [
            f"argument_plan model={self.model_variant} status={self.status} "
            f"layers={self.layers} prompt_len={self.prompt_len} "
            f"decode_context={self.decode_context} npu_candidates={self.npu_candidate_count} "
            f"bindings={self.argument_binding_count} args={self.argument_count} "
            f"missing_args={self.missing_argument_count} blockers={blockers}"
        ]
        if include_bindings:
            lines.extend(binding.format(include_arguments=include_arguments) for binding in self.bindings)
        return "\n".join(lines)

    def to_json_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["bindings"] = [asdict(binding) for binding in self.bindings]
        return data


def _bf16_bytes(shape: tuple[int, ...]) -> int:
    count = 1
    for dim in shape:
        count *= int(dim)
    return count * BF16_BYTES


def _argument_direction(key: str, binding: Gemma3BufferBinding) -> str:
    if key in binding.static_weight_bos:
        return "static"
    if key in binding.mutable_buffers or (key in binding.inputs and key in binding.outputs):
        return "inout"
    if key in binding.inputs:
        return "input"
    if key in binding.outputs:
        return "output"
    return "unknown"


def _ordered_argument_keys(binding: Gemma3BufferBinding) -> tuple[str, ...]:
    keys: list[str] = []
    for group in (binding.inputs, binding.static_weight_bos, binding.mutable_buffers, binding.outputs):
        for key in group:
            if key not in keys:
                keys.append(key)
    return tuple(keys)


def _bo_map(bo_plan: Gemma3BOPlan) -> dict[str, Gemma3BORecord]:
    return {record.name: record for record in bo_plan.records}


def _virtual_shape(key: str, bo_plan: Gemma3BOPlan, preflight: Gemma3NPUPreflightPlan) -> tuple[int, ...] | None:
    if preflight.hidden_size is None or preflight.intermediate_size is None:
        return None
    hidden = int(preflight.hidden_size)
    intermediate = int(preflight.intermediate_size)
    match = _LAYER_VIRTUAL_RE.match(key)
    if not match:
        return None
    phase = match.group("phase")
    name = match.group("name")
    tokens = bo_plan.prompt_len if phase == "prefill" else 1
    if name in {
        "pre_attention_norm",
        "attention_projection",
        "post_attention_norm",
        "pre_feedforward_norm",
        "post_feedforward_norm",
    }:
        return (tokens, hidden)
    if name == "mlp_activation":
        return (tokens, intermediate)
    return None


def _argument_for_key(
    *,
    arg_index: int,
    key: str,
    direction: str,
    binding: Gemma3BufferBinding,
    bo_records: dict[str, Gemma3BORecord],
    bo_plan: Gemma3BOPlan,
    preflight: Gemma3NPUPreflightPlan,
) -> Gemma3KernelArgument:
    blockers: list[str] = []
    if key in bo_records:
        record = bo_records[key]
        storage = "persistent-bo"
        dtype = record.dtype
        shape = tuple(int(dim) for dim in record.shape)
        byte_count = int(record.bytes)
    elif key in binding.virtual_buffers:
        storage = "virtual-buffer"
        dtype = "bf16"
        shape_value = _virtual_shape(key, bo_plan, preflight)
        if shape_value is None:
            shape = ()
            byte_count = 0
            blockers.append("missing-virtual-buffer-shape")
        else:
            shape = shape_value
            byte_count = _bf16_bytes(shape)
    else:
        storage = "unresolved"
        dtype = "unknown"
        shape = ()
        byte_count = 0
        blockers.append("missing-argument-storage")
    if direction == "unknown":
        blockers.append("missing-argument-direction")
    return Gemma3KernelArgument(
        arg_index=arg_index,
        key=key,
        direction=direction,
        storage=storage,
        dtype=dtype,
        shape=shape,
        bytes=byte_count,
        status="ARGUMENT_BOUND" if not blockers else "ARGUMENT_BINDING_BLOCKED",
        blockers=tuple(dict.fromkeys(blockers)),
    )


def build_argument_binding_plan_from_components(
    *,
    model_variant: str,
    preflight: Gemma3NPUPreflightPlan,
    bo_plan: Gemma3BOPlan,
    wiring: Gemma3NPUWiringPlan,
    buffer_binding_plan: Gemma3BufferBindingPlan,
) -> Gemma3KernelArgumentBindingPlan:
    bo_records = _bo_map(bo_plan)
    stage_by_key = {
        (stage.phase, stage.layer_index, stage.role): stage
        for stage in wiring.stages
    }
    bindings: list[Gemma3KernelArgumentBinding] = []
    for binding in buffer_binding_plan.bindings:
        if binding.backend != "npu-candidate":
            continue
        stage = stage_by_key[(binding.phase, binding.layer_index, binding.role)]
        args = tuple(
            _argument_for_key(
                arg_index=index,
                key=key,
                direction=_argument_direction(key, binding),
                binding=binding,
                bo_records=bo_records,
                bo_plan=bo_plan,
                preflight=preflight,
            )
            for index, key in enumerate(_ordered_argument_keys(binding))
        )
        blockers = tuple(dict.fromkeys(blocker for arg in args for blocker in arg.blockers))
        bindings.append(
            Gemma3KernelArgumentBinding(
                phase=binding.phase,
                layer_index=binding.layer_index,
                role=binding.role,
                kernel=stage.kernel,
                route=stage.route,
                argument_count=len(args),
                status="ARGUMENT_BINDING_VALIDATED" if not blockers else "ARGUMENT_BINDING_BLOCKED",
                blockers=blockers,
                arguments=args,
            )
        )
    missing = sum(arg.status != "ARGUMENT_BOUND" for binding in bindings for arg in binding.arguments)
    blockers = ("model-kernel-argument-binding-not-validated",) if missing else ()
    return Gemma3KernelArgumentBindingPlan(
        model_variant=model_variant,
        status="READY_FOR_KERNEL_LAUNCH" if not blockers else "BLOCKED",
        layers=wiring.layers,
        prompt_len=bo_plan.prompt_len,
        decode_context=bo_plan.decode_context,
        npu_candidate_count=wiring.npu_candidate_count,
        argument_binding_count=len(bindings),
        argument_count=sum(binding.argument_count for binding in bindings),
        missing_argument_count=missing,
        blockers=blockers,
        bindings=tuple(bindings),
    )


def build_argument_binding_plan(
    model_variant: str,
    *,
    weights_dir: Path | None = None,
    prompt_len: int | None = None,
    decode_context: int | None = None,
    kv_strategy: str = "benchmark-cell",
) -> Gemma3KernelArgumentBindingPlan:
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
    wiring = build_wiring_plan_from_preflight(preflight)
    buffer_binding_plan = build_buffer_binding_plan_from_components(
        model_variant=model_variant,
        bo_plan=bo_plan,
        weight_plan=weight_plan,
        wiring=wiring,
    )
    return build_argument_binding_plan_from_components(
        model_variant=model_variant,
        preflight=preflight,
        bo_plan=bo_plan,
        wiring=wiring,
        buffer_binding_plan=buffer_binding_plan,
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
    shapes = {}
    for layer in range(2):
        shapes.update(
            {
                f"model.layers.{layer}.self_attn.q_proj.weight": (1024, 1152),
                f"model.layers.{layer}.self_attn.k_proj.weight": (256, 1152),
                f"model.layers.{layer}.self_attn.v_proj.weight": (256, 1152),
                f"model.layers.{layer}.self_attn.o_proj.weight": (1152, 1024),
                f"model.layers.{layer}.mlp.gate_proj.weight": (6912, 1152),
                f"model.layers.{layer}.mlp.up_proj.weight": (6912, 1152),
                f"model.layers.{layer}.mlp.down_proj.weight": (1152, 6912),
            }
        )
    return build_weight_plan_from_shapes("gemma3-1b", shapes)


def _fake_norm_weight_plan() -> Gemma3NormWeightPlan:
    shapes = {}
    for layer in range(2):
        shapes.update(
            {
                f"model.layers.{layer}.input_layernorm.weight": (1152,),
                f"model.layers.{layer}.post_attention_layernorm.weight": (1152,),
                f"model.layers.{layer}.pre_feedforward_layernorm.weight": (1152,),
                f"model.layers.{layer}.post_feedforward_layernorm.weight": (1152,),
                f"model.layers.{layer}.self_attn.q_norm.weight": (256,),
                f"model.layers.{layer}.self_attn.k_norm.weight": (256,),
            }
        )
    return build_norm_weight_plan_from_shapes("gemma3-1b", shapes)


def _self_test() -> None:
    preflight = _fake_preflight()
    weight_plan = _fake_weight_plan()
    bo_plan = build_bo_plan_from_preflight(
        preflight,
        weight_plan,
        _fake_norm_weight_plan(),
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
    plan = build_argument_binding_plan_from_components(
        model_variant="gemma3-1b",
        preflight=preflight,
        bo_plan=bo_plan,
        wiring=wiring,
        buffer_binding_plan=buffer_binding_plan,
    )
    if plan.status != "READY_FOR_KERNEL_LAUNCH" or plan.argument_binding_count != 52:
        raise AssertionError(plan)
    if plan.argument_count != 164:
        raise AssertionError(plan)
    if plan.missing_argument_count != 0 or plan.blockers:
        raise AssertionError(plan)
    first = plan.bindings[0]
    if first.role != "pre_attention_norm" or first.argument_count != 3:
        raise AssertionError(first)
    qkv = next(binding for binding in plan.bindings if binding.role == "qkv_projection" and binding.phase == "prefill")
    if qkv.argument_count != 5:
        raise AssertionError(qkv)
    print(plan.format(include_bindings=True, include_arguments=True))
    print("GEMMA3_ARGUMENT_BINDING_SELF_TEST: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma3 model-loop kernel argument binding validation")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model-variant", choices=sorted(MODEL_SPECS), default="gemma3-1b")
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--prompt-len", type=int)
    parser.add_argument("--decode-context", type=int)
    parser.add_argument("--kv-strategy", choices=KV_STRATEGIES, default="benchmark-cell")
    parser.add_argument("--include-bindings", action="store_true")
    parser.add_argument("--include-arguments", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return 0
    plan = build_argument_binding_plan(
        args.model_variant,
        weights_dir=args.weights_dir,
        prompt_len=args.prompt_len,
        decode_context=args.decode_context,
        kv_strategy=args.kv_strategy,
    )
    if args.json:
        print(json.dumps(plan.to_json_dict(), indent=2, sort_keys=True))
    else:
        print(plan.format(include_bindings=args.include_bindings, include_arguments=args.include_arguments))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
