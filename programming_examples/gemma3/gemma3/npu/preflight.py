#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Real Gemma3 NPU model-execution preflight.

This module derives real model projection padding and attention metadata before
wiring full NPU execution. It is compile/runtime neutral and imports no AIR.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gemma3.core.common import Q4NX_COLS, Q4NX_ROWS
from gemma3.core.artifacts import MODEL_SPECS, load_config_summary, q4nx_roundtrip_report, load_real_model_artifacts


@dataclass(frozen=True)
class ProjectionPlan:
    family: str
    shape: tuple[int, int]
    padded_shape: tuple[int, int]
    row_blocks: int
    col_blocks: int
    requires_padding: bool
    max_abs_error: float
    mean_abs_error: float

    def format(self) -> str:
        shape = f"{self.shape[0]}x{self.shape[1]}"
        padded = f"{self.padded_shape[0]}x{self.padded_shape[1]}"
        return (
            f"projection family={self.family} shape={shape} padded={padded} "
            f"row_blocks={self.row_blocks} col_blocks={self.col_blocks} "
            f"padding={self.requires_padding} max_abs_error={self.max_abs_error:.6f} "
            f"mean_abs_error={self.mean_abs_error:.6f}"
        )


@dataclass(frozen=True)
class Gemma3NPUPreflightPlan:
    model_variant: str
    status: str
    blocker: str
    layers: int | None
    hidden_size: int | None
    intermediate_size: int | None
    head_dim: int | None
    num_attention_heads: int | None
    num_key_value_heads: int | None
    sliding_window: int | None
    attention_pattern: str
    projections: tuple[ProjectionPlan, ...]

    def format(self) -> str:
        lines = [
            f"npu_preflight model={self.model_variant} status={self.status} "
            f"blocker={self.blocker} layers={self.layers} hidden={self.hidden_size} "
            f"heads={self.num_attention_heads} kv_heads={self.num_key_value_heads} "
            f"head_dim={self.head_dim} sliding_window={self.sliding_window} "
            f"pattern={self.attention_pattern}"
        ]
        lines.extend(projection.format() for projection in self.projections)
        return "\n".join(lines)


def _ceil_to(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _projection_plan(record: dict[str, Any]) -> ProjectionPlan:
    rows, cols = (int(v) for v in record["shape"])
    padded_rows = _ceil_to(rows, Q4NX_ROWS)
    padded_cols = _ceil_to(cols, Q4NX_COLS)
    return ProjectionPlan(
        family=str(record["family"]),
        shape=(rows, cols),
        padded_shape=(padded_rows, padded_cols),
        row_blocks=padded_rows // Q4NX_ROWS,
        col_blocks=padded_cols // Q4NX_COLS,
        requires_padding=padded_rows != rows or padded_cols != cols,
        max_abs_error=float(record.get("max_abs_error", 0.0)),
        mean_abs_error=float(record.get("mean_abs_error", 0.0)),
    )


def _infer_heads(summary: dict[str, Any], projections: tuple[ProjectionPlan, ...]) -> tuple[int | None, int | None, int | None]:
    head_dim = summary.get("head_dim") or 256
    q_proj = next((item for item in projections if item.family == "q_proj"), None)
    k_proj = next((item for item in projections if item.family == "k_proj"), None)
    heads = summary.get("num_attention_heads")
    kv_heads = summary.get("num_key_value_heads")
    if heads is None and q_proj is not None and head_dim:
        heads = q_proj.shape[0] // int(head_dim)
    if kv_heads is None and k_proj is not None and head_dim:
        kv_heads = k_proj.shape[0] // int(head_dim)
    return int(head_dim) if head_dim else None, int(heads) if heads else None, int(kv_heads) if kv_heads else None


def build_preflight_plan(
    model_variant: str,
    *,
    weights_dir: Path | None = None,
) -> Gemma3NPUPreflightPlan:
    inventory = load_real_model_artifacts(model_variant, weights_dir=weights_dir, strict=True)
    summary = load_config_summary(inventory)
    roundtrip = q4nx_roundtrip_report(model_variant, weights_dir=weights_dir)
    projections = tuple(
        _projection_plan(record) for record in roundtrip if record.get("status") == "PASS"
    )
    head_dim, heads, kv_heads = _infer_heads(summary, projections)
    pattern = "5-local-1-global" if summary.get("sliding_window") else "global-or-config-dependent"
    return Gemma3NPUPreflightPlan(
        model_variant=model_variant,
        status="READY_FOR_NPU_WIRING",
        blocker="npu-model-execution-not-implemented",
        layers=summary.get("num_hidden_layers"),
        hidden_size=summary.get("hidden_size"),
        intermediate_size=summary.get("intermediate_size"),
        head_dim=head_dim,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        sliding_window=summary.get("sliding_window"),
        attention_pattern=pattern,
        projections=projections,
    )


def _self_test() -> None:
    records = [
        {
            "family": "q_proj",
            "shape": (1024, 1152),
            "max_abs_error": 0.1,
            "mean_abs_error": 0.01,
        },
        {
            "family": "o_proj",
            "shape": (1152, 1024),
            "max_abs_error": 0.2,
            "mean_abs_error": 0.02,
        },
    ]
    plans = tuple(_projection_plan(record) for record in records)
    if plans[0].padded_shape != (1024, 1280) or not plans[0].requires_padding:
        raise AssertionError(plans[0])
    if plans[1].padded_shape != (1152, 1024) or plans[1].requires_padding:
        raise AssertionError(plans[1])
    print(plans[0].format())
    print(plans[1].format())
    print("GEMMA3_NPU_PREFLIGHT_SELF_TEST: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma3 real NPU execution preflight")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model-variant", choices=sorted(MODEL_SPECS), default="gemma3-1b")
    parser.add_argument("--weights-dir", type=Path)
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return 0
    plan = build_preflight_plan(args.model_variant, weights_dir=args.weights_dir)
    print(plan.format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
