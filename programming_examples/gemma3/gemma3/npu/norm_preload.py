#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gemma3 BF16 norm-weight XRT preload smoke.

This preloads the small RMSNorm/QK-Norm vector weights identified by
`gemma3.npu.norm_weight_plan`. It is preload evidence only; model kernel argument
binding and launch validation remain separate blockers.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from ml_dtypes import bfloat16

from gemma3.core.artifacts import MODEL_SPECS, load_real_model_artifacts
from gemma3.npu.norm_weight_plan import Gemma3NormWeightRecord, build_norm_weight_plan
from gemma3.paths import RESULTS_DIR


DEFAULT_NORM_PRELOAD_EVIDENCE = RESULTS_DIR / "gemma3_norm_preload_evidence.json"


def load_norm_preload_evidence(path: Path | None = None) -> list[dict[str, object]]:
    path = path or DEFAULT_NORM_PRELOAD_EVIDENCE
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return list(data["results"])
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return list(data)
    raise ValueError(f"unsupported norm preload evidence format: {path}")


def has_full_norm_xrt_preload_evidence(model_variant: str, path: Path | None = None) -> bool:
    for item in load_norm_preload_evidence(path):
        if item.get("model_variant") != model_variant:
            continue
        if item.get("status") != "FULL_NORM_XRT_PRELOAD_PASS" or not item.get("full_model"):
            continue
        if item.get("allocation_mode") != "contiguous-norm-bo":
            continue
        tensor_count = int(item.get("tensor_count", -1))
        planned_count = int(item.get("planned_tensor_count", -2))
        requested = int(item.get("requested_bytes", -1))
        serialized = int(item.get("serialized_bytes", -2))
        written = int(item.get("xrt_written_bytes", -3))
        if tensor_count == planned_count and requested == serialized == written and not item.get("blockers", []):
            return True
    return False


@dataclass(frozen=True)
class Gemma3NormPreloadRecord:
    layer_index: int
    family: str
    tensor_key: str
    requested_bytes: int
    serialized_bytes: int
    xrt_written_bytes: int
    bo_offset: int
    checksum: str
    status: str

    def format(self) -> str:
        return (
            f"norm_preload L{self.layer_index} family={self.family} "
            f"requested={self.requested_bytes} serialized={self.serialized_bytes} "
            f"xrt_written={self.xrt_written_bytes} offset={self.bo_offset} "
            f"status={self.status} sha256={self.checksum[:16]}"
        )


@dataclass(frozen=True)
class Gemma3NormPreloadReport:
    model_variant: str
    status: str
    full_model: bool
    planned_tensor_count: int
    allocation_mode: str
    tensor_count: int
    requested_bytes: int
    serialized_bytes: int
    xrt_written_bytes: int
    elapsed_seconds: float
    blockers: tuple[str, ...]
    records: tuple[Gemma3NormPreloadRecord, ...]

    def format(self, *, include_records: bool = False) -> str:
        blockers = ",".join(self.blockers) if self.blockers else "none"
        lines = [
            f"norm_preload model={self.model_variant} status={self.status} "
            f"full_model={self.full_model} planned_tensors={self.planned_tensor_count} "
            f"allocation_mode={self.allocation_mode} tensors={self.tensor_count} "
            f"requested={self.requested_bytes} serialized={self.serialized_bytes} "
            f"xrt_written={self.xrt_written_bytes} blockers={blockers}"
        ]
        if include_records:
            lines.extend(record.format() for record in self.records)
        return "\n".join(lines)

    def to_json_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["records"] = [asdict(record) for record in self.records]
        return data


def serialize_bf16_vector(vector: np.ndarray) -> bytes:
    source = np.asarray(vector, dtype=np.float32).reshape(-1)
    return source.astype(bfloat16).tobytes()


def _load_vector(paths: tuple[str, ...], key: str) -> np.ndarray:
    try:
        from safetensors import safe_open
    except Exception as exc:
        raise RuntimeError("python:safetensors is required for norm preload") from exc
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
    records: tuple[Gemma3NormWeightRecord, ...],
    *,
    layer_index: int | None,
    family: str | None,
    max_tensors: int,
) -> tuple[Gemma3NormWeightRecord, ...]:
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


def build_norm_preload_report(
    model_variant: str,
    *,
    weights_dir: Path | None = None,
    layer_index: int | None = 0,
    family: str | None = "input_layernorm",
    max_tensors: int = 1,
    full_model: bool = False,
    contiguous_bo: bool = False,
    xrt_smoke: bool = False,
    device_index: int = 0,
) -> Gemma3NormPreloadReport:
    if max_tensors <= 0:
        raise ValueError("max_tensors must be positive")
    start = perf_counter()
    inventory = load_real_model_artifacts(model_variant, weights_dir=weights_dir, strict=True)
    plan = build_norm_weight_plan(model_variant, weights_dir=weights_dir)
    if full_model:
        layer_index = None
        family = None
        max_tensors = plan.tensor_count
    selected = _select_records(plan.records, layer_index=layer_index, family=family, max_tensors=max_tensors)
    if not selected:
        raise RuntimeError("no norm weight records matched the preload filter")

    xrt = None
    device = None
    if xrt_smoke:
        try:
            import pyxrt as xrt_module
        except Exception as exc:
            raise RuntimeError("python:pyxrt is required for XRT norm preload") from exc
        xrt = xrt_module
        device = xrt.device(device_index)

    contiguous_norm_bo = None
    if xrt_smoke and contiguous_bo:
        contiguous_norm_bo = xrt.bo(device, plan.static_bo_bytes, xrt.bo.host_only, 0)

    offset = 0
    records: list[Gemma3NormPreloadRecord] = []
    for record in selected:
        payload = serialize_bf16_vector(_load_vector(tuple(inventory.safetensors), record.tensor_key))
        if len(payload) != record.static_bo_bytes:
            raise RuntimeError(
                f"serialized size mismatch for {record.tensor_key}: "
                f"got {len(payload)}, expected {record.static_bo_bytes}"
            )
        written = 0
        status = "SERIALIZED"
        bo_offset = offset
        if xrt_smoke and contiguous_norm_bo is not None:
            contiguous_norm_bo.write(payload, bo_offset)
            written = len(payload)
            status = "XRT_NORM_PRELOAD_SMOKE_PASS"
        elif xrt_smoke:
            bo = xrt.bo(device, len(payload), xrt.bo.host_only, 0)
            bo.write(payload, 0)
            bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            written = len(payload)
            status = "XRT_NORM_PRELOAD_SMOKE_PASS"
        records.append(
            Gemma3NormPreloadRecord(
                layer_index=record.layer_index,
                family=record.family,
                tensor_key=record.tensor_key,
                requested_bytes=record.static_bo_bytes,
                serialized_bytes=len(payload),
                xrt_written_bytes=written,
                bo_offset=bo_offset,
                checksum=hashlib.sha256(payload).hexdigest(),
                status=status,
            )
        )
        offset += len(payload)
    if contiguous_norm_bo is not None:
        contiguous_norm_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    allocation_mode = "contiguous-norm-bo" if contiguous_bo else "per-tensor-bo" if xrt_smoke else "serialized-only"
    complete = (
        full_model
        and len(records) == plan.tensor_count
        and sum(record.serialized_bytes for record in records) == plan.static_bo_bytes
    )
    if complete and xrt_smoke and contiguous_bo and sum(record.xrt_written_bytes for record in records) == plan.static_bo_bytes:
        status = "FULL_NORM_XRT_PRELOAD_PASS"
        blockers: tuple[str, ...] = ()
    elif complete:
        status = "FULL_NORM_SERIALIZED_PASS"
        blockers = ("full-norm-weight-xrt-preload-not-validated",)
    else:
        status = "XRT_NORM_PRELOAD_SMOKE_PASS" if xrt_smoke else "SERIALIZED"
        blockers = ("full-norm-weight-bo-preload-not-validated",)
    return Gemma3NormPreloadReport(
        model_variant=model_variant,
        status=status,
        full_model=full_model,
        planned_tensor_count=plan.tensor_count,
        allocation_mode=allocation_mode,
        tensor_count=len(records),
        requested_bytes=sum(record.requested_bytes for record in records),
        serialized_bytes=sum(record.serialized_bytes for record in records),
        xrt_written_bytes=sum(record.xrt_written_bytes for record in records),
        elapsed_seconds=perf_counter() - start,
        blockers=blockers,
        records=tuple(records),
    )


def _self_test() -> None:
    payload = serialize_bf16_vector(np.arange(8, dtype=np.float32))
    if len(payload) != 16:
        raise AssertionError(len(payload))
    checksum = hashlib.sha256(payload).hexdigest()
    record = Gemma3NormPreloadRecord(0, "q_norm", "fixture", 16, 16, 0, 0, checksum, "SERIALIZED")
    report = Gemma3NormPreloadReport(
        "gemma3-1b",
        "SERIALIZED",
        False,
        1,
        "serialized-only",
        1,
        16,
        16,
        0,
        0.0,
        ("full-norm-weight-bo-preload-not-validated",),
        (record,),
    )
    if has_full_norm_xrt_preload_evidence("gemma3-missing-fixture", path=Path("/tmp/gemma3_missing_norm_preload_evidence.json")):
        raise AssertionError("unexpected missing evidence pass")
    full_report = Gemma3NormPreloadReport(
        "gemma3-1b",
        "FULL_NORM_XRT_PRELOAD_PASS",
        True,
        1,
        "contiguous-norm-bo",
        1,
        16,
        16,
        16,
        0.0,
        (),
        (record,),
    )
    tmp = Path("/tmp/gemma3_norm_preload_self_test_evidence.json")
    tmp.write_text(json.dumps({"results": [full_report.to_json_dict()]}, sort_keys=True), encoding="utf-8")
    if not has_full_norm_xrt_preload_evidence("gemma3-1b", path=tmp):
        raise AssertionError("expected full norm preload evidence pass")
    print(report.format(include_records=True))
    print("GEMMA3_NORM_PRELOAD_EVIDENCE_SELF_TEST: PASS")
    print("GEMMA3_NORM_PRELOAD_SELF_TEST: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma3 BF16 norm-weight preload smoke")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model-variant", choices=sorted(MODEL_SPECS), default="gemma3-1b")
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument(
        "--family",
        choices=("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm", "q_norm", "k_norm"),
        default="input_layernorm",
    )
    parser.add_argument("--max-tensors", type=int, default=1)
    parser.add_argument("--full-model", action="store_true")
    parser.add_argument("--contiguous-bo", action="store_true")
    parser.add_argument("--xrt-smoke", action="store_true")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--include-records", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return 0
    report = build_norm_preload_report(
        args.model_variant,
        weights_dir=args.weights_dir,
        layer_index=args.layer_index,
        family=args.family,
        max_tensors=args.max_tensors,
        full_model=args.full_model,
        contiguous_bo=args.contiguous_bo,
        xrt_smoke=args.xrt_smoke,
        device_index=args.device_index,
    )
    print(report.format(include_records=args.include_records))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8")
        print(f"GEMMA3_NORM_PRELOAD_JSON: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
