#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gemma3 real Q4NX static-weight preload smoke.

This serializes selected real projection tensors into the Q4NX packed
weight/scale/min byte contract and can write the serialized stream into XRT BOs.
It is a preload smoke only; it does not prove that a downstream kernel consumes
the stream layout until model-runner binding is implemented.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np
from ml_dtypes import bfloat16

from common import Q4NX_COLS, Q4NX_ROWS, pack_int4_low_first
from gemma3_artifacts import MODEL_SPECS, load_real_model_artifacts
from gemma3_weight_plan import Gemma3ProjectionWeightRecord, build_weight_plan


@dataclass(frozen=True)
class Gemma3StaticPreloadRecord:
    layer_index: int
    family: str
    tensor_key: str
    requested_bytes: int
    serialized_bytes: int
    xrt_written_bytes: int
    checksum: str
    status: str

    def format(self) -> str:
        return (
            f"static_preload L{self.layer_index} family={self.family} "
            f"requested={self.requested_bytes} serialized={self.serialized_bytes} "
            f"xrt_written={self.xrt_written_bytes} status={self.status} "
            f"sha256={self.checksum[:16]}"
        )


@dataclass(frozen=True)
class Gemma3StaticPreloadReport:
    model_variant: str
    status: str
    tensor_count: int
    requested_bytes: int
    serialized_bytes: int
    xrt_written_bytes: int
    elapsed_seconds: float
    blockers: tuple[str, ...]
    records: tuple[Gemma3StaticPreloadRecord, ...]

    def format(self, *, include_records: bool = False) -> str:
        blockers = ",".join(self.blockers) if self.blockers else "none"
        lines = [
            f"static_preload model={self.model_variant} status={self.status} "
            f"tensors={self.tensor_count} requested={self.requested_bytes} "
            f"serialized={self.serialized_bytes} xrt_written={self.xrt_written_bytes} "
            f"blockers={blockers}"
        ]
        if include_records:
            lines.extend(record.format() for record in self.records)
        return "\n".join(lines)

    def to_json_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["records"] = [asdict(record) for record in self.records]
        return data


def _ceil_to(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def serialize_q4nx_matrix(matrix: np.ndarray) -> bytes:
    source = np.asarray(matrix, dtype=np.float32)
    padded_rows = _ceil_to(source.shape[0], Q4NX_ROWS)
    padded_cols = _ceil_to(source.shape[1], Q4NX_COLS)
    padded = np.zeros((padded_rows, padded_cols), dtype=np.float32)
    padded[: source.shape[0], : source.shape[1]] = source

    chunks: list[bytes] = []
    for row_base in range(0, padded_rows, Q4NX_ROWS):
        block = padded[row_base : row_base + Q4NX_ROWS, :]
        col_min = block.min(axis=0)
        col_max = block.max(axis=0)
        scale = (col_max - col_min) / 15.0
        scale = np.where(scale == 0.0, 1.0, scale)
        q = np.rint((block - col_min[None, :]) / scale[None, :])
        q = np.clip(q, 0, 15).astype(np.uint8)
        chunks.append(pack_int4_low_first(q).tobytes())
        chunks.append(scale.astype(bfloat16).tobytes())
        chunks.append(col_min.astype(bfloat16).tobytes())
    return b"".join(chunks)


def _load_matrix(paths: Iterable[str], key: str) -> np.ndarray:
    try:
        from safetensors import safe_open
    except Exception as exc:
        raise RuntimeError("python:safetensors is required for static preload") from exc
    try:
        import torch  # noqa: F401
    except Exception as exc:
        raise RuntimeError("python:torch is required for BF16 safetensor loading") from exc
    for filename in paths:
        with safe_open(filename, framework="pt", device="cpu") as handle:
            if key in handle.keys():
                return handle.get_tensor(key).float().cpu().numpy()
    raise RuntimeError(f"tensor key not found in safetensors: {key}")


def _select_records(
    records: tuple[Gemma3ProjectionWeightRecord, ...],
    *,
    layer_index: int | None,
    family: str | None,
    max_tensors: int,
) -> tuple[Gemma3ProjectionWeightRecord, ...]:
    selected = []
    for record in records:
        if layer_index is not None and record.layer_index != layer_index:
            continue
        if family is not None and record.family != family:
            continue
        selected.append(record)
        if len(selected) >= max_tensors:
            break
    return tuple(selected)


def build_static_preload_report(
    model_variant: str,
    *,
    weights_dir: Path | None = None,
    layer_index: int | None = 0,
    family: str | None = "q_proj",
    max_tensors: int = 1,
    xrt_smoke: bool = False,
    device_index: int = 0,
) -> Gemma3StaticPreloadReport:
    if max_tensors <= 0:
        raise ValueError("max_tensors must be positive")
    start = perf_counter()
    inventory = load_real_model_artifacts(model_variant, weights_dir=weights_dir, strict=True)
    plan = build_weight_plan(model_variant, weights_dir=weights_dir)
    selected = _select_records(plan.records, layer_index=layer_index, family=family, max_tensors=max_tensors)
    if not selected:
        raise RuntimeError("no static projection records matched the preload filter")

    xrt = None
    device = None
    if xrt_smoke:
        try:
            import pyxrt as xrt_module
        except Exception as exc:
            raise RuntimeError("python:pyxrt is required for XRT static preload") from exc
        xrt = xrt_module
        device = xrt.device(device_index)

    records: list[Gemma3StaticPreloadRecord] = []
    for record in selected:
        payload = serialize_q4nx_matrix(_load_matrix(inventory.safetensors, record.tensor_key))
        if len(payload) != record.static_bo_bytes:
            raise RuntimeError(
                f"serialized size mismatch for {record.tensor_key}: "
                f"got {len(payload)}, expected {record.static_bo_bytes}"
            )
        written = 0
        status = "SERIALIZED"
        if xrt_smoke:
            bo = xrt.bo(device, len(payload), xrt.bo.host_only, 0)
            bo.write(payload, 0)
            bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            written = len(payload)
            status = "XRT_PRELOAD_SMOKE_PASS"
        records.append(
            Gemma3StaticPreloadRecord(
                layer_index=record.layer_index,
                family=record.family,
                tensor_key=record.tensor_key,
                requested_bytes=record.static_bo_bytes,
                serialized_bytes=len(payload),
                xrt_written_bytes=written,
                checksum=hashlib.sha256(payload).hexdigest(),
                status=status,
            )
        )
    blockers = ("full-static-weight-bo-preload-not-validated",)
    return Gemma3StaticPreloadReport(
        model_variant=model_variant,
        status="XRT_PRELOAD_SMOKE_PASS" if xrt_smoke else "SERIALIZED",
        tensor_count=len(records),
        requested_bytes=sum(record.requested_bytes for record in records),
        serialized_bytes=sum(record.serialized_bytes for record in records),
        xrt_written_bytes=sum(record.xrt_written_bytes for record in records),
        elapsed_seconds=perf_counter() - start,
        blockers=blockers,
        records=tuple(records),
    )


def _self_test() -> None:
    matrix = np.arange(Q4NX_ROWS * Q4NX_COLS, dtype=np.float32).reshape(Q4NX_ROWS, Q4NX_COLS) / 100.0
    payload = serialize_q4nx_matrix(matrix)
    expected = Q4NX_ROWS * Q4NX_COLS // 2 + Q4NX_COLS * 2 + Q4NX_COLS * 2
    if len(payload) != expected:
        raise AssertionError((len(payload), expected))
    checksum = hashlib.sha256(payload).hexdigest()
    record = Gemma3StaticPreloadRecord(0, "q_proj", "fixture", expected, len(payload), 0, checksum, "SERIALIZED")
    report = Gemma3StaticPreloadReport(
        "gemma3-1b",
        "SERIALIZED",
        1,
        expected,
        len(payload),
        0,
        0.0,
        ("full-static-weight-bo-preload-not-validated",),
        (record,),
    )
    print(report.format(include_records=True))
    print("GEMMA3_STATIC_PRELOAD_SELF_TEST: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma3 real Q4NX static-weight preload smoke")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model-variant", choices=sorted(MODEL_SPECS), default="gemma3-1b")
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--family", choices=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"), default="q_proj")
    parser.add_argument("--max-tensors", type=int, default=1)
    parser.add_argument("--xrt-smoke", action="store_true")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--include-records", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return 0
    report = build_static_preload_report(
        args.model_variant,
        weights_dir=args.weights_dir,
        layer_index=args.layer_index,
        family=args.family,
        max_tensors=args.max_tensors,
        xrt_smoke=args.xrt_smoke,
        device_index=args.device_index,
    )
    print(report.format(include_records=args.include_records))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8")
        print(f"GEMMA3_STATIC_PRELOAD_JSON: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
