#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gemma3 text-stack static projection weight plan for future NPU execution."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Iterable

from gemma3.core.common import Q4NX_COLS, Q4NX_ROWS
from gemma3.core.artifacts import MODEL_SPECS, Q4NX_PROJECTION_FAMILIES, load_real_model_artifacts


_TEXT_PROJECTION_RE = re.compile(
    r"(?:^|language_model\.)model\.layers\.(?P<layer>\d+)\..*\."
    r"(?P<family>q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$"
)


@dataclass(frozen=True)
class Gemma3ProjectionWeightRecord:
    layer_index: int
    family: str
    tensor_key: str
    shape: tuple[int, int]
    padded_shape: tuple[int, int]
    row_blocks: int
    col_blocks: int
    packed_weight_bytes: int
    scale_bytes: int
    min_bytes: int
    static_bo_bytes: int
    requires_padding: bool

    def format(self) -> str:
        shape = f"{self.shape[0]}x{self.shape[1]}"
        padded = f"{self.padded_shape[0]}x{self.padded_shape[1]}"
        return (
            f"weight L{self.layer_index} family={self.family} shape={shape} "
            f"padded={padded} row_blocks={self.row_blocks} col_blocks={self.col_blocks} "
            f"bo_bytes={self.static_bo_bytes} padding={self.requires_padding}"
        )


@dataclass(frozen=True)
class Gemma3WeightFamilySummary:
    family: str
    tensor_count: int
    static_bo_bytes: int
    padded_tensor_count: int

    def format(self) -> str:
        return (
            f"weight_family {self.family} tensors={self.tensor_count} "
            f"bo_bytes={self.static_bo_bytes} padded_tensors={self.padded_tensor_count}"
        )


@dataclass(frozen=True)
class Gemma3StaticWeightPlan:
    model_variant: str
    status: str
    layers: int
    tensor_count: int
    static_bo_bytes: int
    families: tuple[Gemma3WeightFamilySummary, ...]
    records: tuple[Gemma3ProjectionWeightRecord, ...]
    blockers: tuple[str, ...]

    def format(self, *, include_records: bool = False) -> str:
        blockers = ",".join(self.blockers) if self.blockers else "none"
        lines = [
            f"weight_plan model={self.model_variant} scope=text status={self.status} "
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


def _ceil_to(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _projection_record(layer_index: int, family: str, tensor_key: str, shape: tuple[int, int]) -> Gemma3ProjectionWeightRecord:
    rows, cols = shape
    padded_rows = _ceil_to(rows, Q4NX_ROWS)
    padded_cols = _ceil_to(cols, Q4NX_COLS)
    row_blocks = padded_rows // Q4NX_ROWS
    col_blocks = padded_cols // Q4NX_COLS
    packed_weight_bytes = padded_rows * padded_cols // 2
    # One BF16 scale vector and one BF16 min/offset vector per 32-row block.
    scale_bytes = row_blocks * padded_cols * 2
    min_bytes = row_blocks * padded_cols * 2
    return Gemma3ProjectionWeightRecord(
        layer_index=layer_index,
        family=family,
        tensor_key=tensor_key,
        shape=(rows, cols),
        padded_shape=(padded_rows, padded_cols),
        row_blocks=row_blocks,
        col_blocks=col_blocks,
        packed_weight_bytes=packed_weight_bytes,
        scale_bytes=scale_bytes,
        min_bytes=min_bytes,
        static_bo_bytes=packed_weight_bytes + scale_bytes + min_bytes,
        requires_padding=padded_rows != rows or padded_cols != cols,
    )


def build_weight_plan_from_shapes(model_variant: str, shapes: dict[str, tuple[int, ...]]) -> Gemma3StaticWeightPlan:
    records: list[Gemma3ProjectionWeightRecord] = []
    blockers: list[str] = []
    for key, shape in sorted(shapes.items()):
        match = _TEXT_PROJECTION_RE.search(key)
        if not match:
            continue
        if len(shape) != 2:
            blockers.append(f"non-matrix-projection:{key}")
            continue
        records.append(
            _projection_record(
                int(match.group("layer")),
                match.group("family"),
                key,
                (int(shape[0]), int(shape[1])),
            )
        )
    if not records:
        blockers.append("no-projection-weight-tensors")
    layers = max((record.layer_index for record in records), default=-1) + 1
    families = []
    for family in Q4NX_PROJECTION_FAMILIES:
        family_records = [record for record in records if record.family == family]
        if family_records:
            families.append(
                Gemma3WeightFamilySummary(
                    family=family,
                    tensor_count=len(family_records),
                    static_bo_bytes=sum(record.static_bo_bytes for record in family_records),
                    padded_tensor_count=sum(record.requires_padding for record in family_records),
                )
            )
    status = "READY_FOR_STATIC_BO_PRELOAD" if not blockers else "BLOCKED"
    return Gemma3StaticWeightPlan(
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


def _safetensor_projection_shapes(paths: Iterable[str]) -> dict[str, tuple[int, ...]]:
    try:
        from safetensors import safe_open
    except Exception as exc:
        raise RuntimeError("python:safetensors is required for weight planning") from exc
    shapes: dict[str, tuple[int, ...]] = {}
    for filename in paths:
        with safe_open(filename, framework="np") as handle:
            for key in handle.keys():
                if _TEXT_PROJECTION_RE.search(key):
                    shapes[key] = _shape_from_slice(handle, key)
    return shapes


def build_weight_plan(
    model_variant: str,
    *,
    weights_dir: Path | None = None,
) -> Gemma3StaticWeightPlan:
    inventory = load_real_model_artifacts(model_variant, weights_dir=weights_dir, strict=True)
    return build_weight_plan_from_shapes(
        model_variant,
        _safetensor_projection_shapes(inventory.safetensors),
    )


def _self_test() -> None:
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
    plan = build_weight_plan_from_shapes("gemma3-1b", shapes)
    if plan.status != "READY_FOR_STATIC_BO_PRELOAD" or plan.tensor_count != 14 or plan.layers != 2:
        raise AssertionError(plan)
    q_proj = next(record for record in plan.records if record.family == "q_proj")
    if q_proj.padded_shape != (1024, 1280) or not q_proj.requires_padding:
        raise AssertionError(q_proj)
    print(plan.format(include_records=True))
    print("GEMMA3_WEIGHT_PLAN_SELF_TEST: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma3 static projection weight plan")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model-variant", choices=sorted(MODEL_SPECS), default="gemma3-1b")
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--include-records", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return 0
    plan = build_weight_plan(args.model_variant, weights_dir=args.weights_dir)
    if args.json:
        print(json.dumps(plan.to_json_dict(), indent=2, sort_keys=True))
    else:
        print(plan.format(include_records=args.include_records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
