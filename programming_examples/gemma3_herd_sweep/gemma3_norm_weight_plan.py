#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gemma3 norm-weight static input plan for nonlinear NPU promotion.

This records the BF16 vector weights needed before RMSNorm and QK-Norm can move
from host fallback to model-loop NPU launch candidates. It deliberately stays
separate from the projection Q4NX static BO plan until norm preload and binding
are validated.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Iterable

from gemma3_artifacts import MODEL_SPECS, load_real_model_artifacts


BF16_BYTES = 2
_NORM_WEIGHT_RE = re.compile(
    r"(?:^|language_model\.)model\.layers\.(?P<layer>\d+)\."
    r"(?:(?:self_attn\.(?P<attention_family>q_norm|k_norm))|"
    r"(?P<layer_family>input_layernorm|post_attention_layernorm|pre_feedforward_layernorm|post_feedforward_layernorm))"
    r"\.weight$"
)
NORM_WEIGHT_FAMILIES = (
    "input_layernorm",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm",
    "q_norm",
    "k_norm",
)


@dataclass(frozen=True)
class Gemma3NormWeightRecord:
    layer_index: int
    family: str
    tensor_key: str
    shape: tuple[int, ...]
    static_bo_bytes: int

    def format(self) -> str:
        shape = "x".join(str(dim) for dim in self.shape) if self.shape else "scalar"
        return (
            f"norm_weight L{self.layer_index} family={self.family} "
            f"shape={shape} bo_bytes={self.static_bo_bytes}"
        )


@dataclass(frozen=True)
class Gemma3NormWeightFamilySummary:
    family: str
    tensor_count: int
    static_bo_bytes: int

    def format(self) -> str:
        return (
            f"norm_weight_family {self.family} tensors={self.tensor_count} "
            f"bo_bytes={self.static_bo_bytes}"
        )


@dataclass(frozen=True)
class Gemma3NormWeightPlan:
    model_variant: str
    status: str
    layers: int
    tensor_count: int
    static_bo_bytes: int
    families: tuple[Gemma3NormWeightFamilySummary, ...]
    records: tuple[Gemma3NormWeightRecord, ...]
    blockers: tuple[str, ...]

    def format(self, *, include_records: bool = False) -> str:
        blockers = ",".join(self.blockers) if self.blockers else "none"
        lines = [
            f"norm_weight_plan model={self.model_variant} status={self.status} "
            f"layers={self.layers} tensors={self.tensor_count} "
            f"bo_bytes={self.static_bo_bytes} blockers={blockers}"
        ]
        lines.extend(family.format() for family in self.families)
        if include_records:
            lines.extend(record.format() for record in self.records)
        return "\n".join(lines)

    def to_json_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["families"] = [asdict(family) for family in self.families]
        data["records"] = [asdict(record) for record in self.records]
        return data


def _bf16_bytes(shape: tuple[int, ...]) -> int:
    count = 1
    for dim in shape:
        count *= int(dim)
    return count * BF16_BYTES


def build_norm_weight_plan_from_shapes(model_variant: str, shapes: dict[str, tuple[int, ...]]) -> Gemma3NormWeightPlan:
    records: list[Gemma3NormWeightRecord] = []
    blockers: list[str] = []
    for key, shape in sorted(shapes.items()):
        match = _NORM_WEIGHT_RE.search(key)
        if not match:
            continue
        if len(shape) != 1:
            blockers.append(f"non-vector-norm-weight:{key}")
            continue
        family = match.group("attention_family") or match.group("layer_family")
        records.append(
            Gemma3NormWeightRecord(
                layer_index=int(match.group("layer")),
                family=str(family),
                tensor_key=key,
                shape=tuple(int(dim) for dim in shape),
                static_bo_bytes=_bf16_bytes(tuple(int(dim) for dim in shape)),
            )
        )
    if not records:
        blockers.append("no-norm-weight-tensors")
    layers = max((record.layer_index for record in records), default=-1) + 1
    families = []
    for family in NORM_WEIGHT_FAMILIES:
        family_records = [record for record in records if record.family == family]
        if family_records:
            families.append(
                Gemma3NormWeightFamilySummary(
                    family=family,
                    tensor_count=len(family_records),
                    static_bo_bytes=sum(record.static_bo_bytes for record in family_records),
                )
            )
    status = "READY_FOR_NORM_WEIGHT_PRELOAD" if not blockers else "BLOCKED"
    return Gemma3NormWeightPlan(
        model_variant=model_variant,
        status=status,
        layers=layers,
        tensor_count=len(records),
        static_bo_bytes=sum(record.static_bo_bytes for record in records),
        families=tuple(families),
        records=tuple(records),
        blockers=tuple(dict.fromkeys(blockers)),
    )


def _shape_from_slice(handle: object, key: str) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in handle.get_slice(key).get_shape())
    except Exception:
        return tuple(int(dim) for dim in handle.get_tensor(key).shape)


def _safetensor_norm_shapes(paths: Iterable[str]) -> dict[str, tuple[int, ...]]:
    try:
        from safetensors import safe_open
    except Exception as exc:
        raise RuntimeError("python:safetensors is required for norm weight planning") from exc
    shapes: dict[str, tuple[int, ...]] = {}
    for filename in paths:
        with safe_open(filename, framework="np") as handle:
            for key in handle.keys():
                if _NORM_WEIGHT_RE.search(key):
                    shapes[key] = _shape_from_slice(handle, key)
    return shapes


def build_norm_weight_plan(
    model_variant: str,
    *,
    weights_dir: Path | None = None,
) -> Gemma3NormWeightPlan:
    inventory = load_real_model_artifacts(model_variant, weights_dir=weights_dir, strict=True)
    return build_norm_weight_plan_from_shapes(
        model_variant,
        _safetensor_norm_shapes(inventory.safetensors),
    )


def _self_test() -> None:
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
    plan = build_norm_weight_plan_from_shapes("gemma3-1b", shapes)
    if plan.status != "READY_FOR_NORM_WEIGHT_PRELOAD" or plan.tensor_count != 12 or plan.layers != 2:
        raise AssertionError(plan)
    q_norm = next(record for record in plan.records if record.family == "q_norm")
    if q_norm.static_bo_bytes != 512:
        raise AssertionError(q_norm)
    print(plan.format(include_records=True))
    print("GEMMA3_NORM_WEIGHT_PLAN_SELF_TEST: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma3 norm-weight static input plan")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model-variant", choices=sorted(MODEL_SPECS), default="gemma3-1b")
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--include-records", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return 0
    plan = build_norm_weight_plan(args.model_variant, weights_dir=args.weights_dir)
    if args.json:
        print(json.dumps(plan.to_json_dict(), indent=2, sort_keys=True))
    else:
        print(plan.format(include_records=args.include_records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
