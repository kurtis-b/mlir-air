#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gemma3 XRT BO allocation and preload smoke runner.

This is a runtime resource smoke, not model execution. It allocates BOs from the
Gemma3 BO plan under explicit byte caps and writes deterministic preload bytes
so future model-runner work can distinguish allocation, preload, kernel launch,
and validation failures.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from gemma3.core.artifacts import MODEL_SPECS, model_spec
from gemma3.npu.bo_plan import KV_STRATEGIES, Gemma3BOPlan, Gemma3BORecord, build_bo_plan
from gemma3.paths import RESULTS_DIR


DEFAULT_BO_ALLOCATION_EVIDENCE = RESULTS_DIR / "gemma3_bo_allocation_evidence.json"


def load_bo_allocation_evidence(path: Path | None = None) -> list[dict[str, object]]:
    path = path or DEFAULT_BO_ALLOCATION_EVIDENCE
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "model_variant" in data:
        return [data]
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return list(data["results"])
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return list(data)
    raise ValueError(f"unsupported BO allocation evidence format: {path}")


def has_paper_shape_bo_allocation_evidence(model_variant: str, path: Path | None = None) -> bool:
    spec = model_spec(model_variant)
    prompt_len = max(spec.prefill_lengths)
    decode_context = spec.max_decode_context
    try:
        expected_plan = build_bo_plan(
            model_variant,
            prompt_len=prompt_len,
            decode_context=decode_context,
        )
    except Exception:
        return False
    for item in load_bo_allocation_evidence(path):
        if item.get("model_variant") != model_variant:
            continue
        if item.get("status") != "FULL_ALLOCATE_PASS":
            continue
        if int(item.get("prompt_len", -1)) != prompt_len:
            continue
        if int(item.get("decode_context", -1)) != decode_context:
            continue
        requested = int(item.get("requested_bytes", -1))
        allocated = int(item.get("allocated_bytes", -2))
        if requested != allocated:
            continue
        if requested != expected_plan.total_bytes:
            continue
        if item.get("kv_strategy") != expected_plan.kv_strategy:
            continue
        if int(item.get("skipped_count", -1)) != 0:
            continue
        if item.get("blockers", []):
            continue
        return True
    return False


@dataclass(frozen=True)
class Gemma3XRTBOResult:
    name: str
    requested_bytes: int
    allocated_bytes: int
    status: str
    static: bool
    preload: str
    notes: str

    def format(self) -> str:
        return (
            f"xrt_bo name={self.name} requested={self.requested_bytes} "
            f"allocated={self.allocated_bytes} status={self.status} "
            f"static={self.static} preload={self.preload} notes={self.notes}"
        )


@dataclass(frozen=True)
class Gemma3XRTRunnerReport:
    model_variant: str
    status: str
    device_index: int
    prompt_len: int
    decode_context: int
    kv_strategy: str
    max_total_bytes: int
    max_bo_bytes: int
    requested_bytes: int
    allocated_bytes: int
    allocation_count: int
    skipped_count: int
    elapsed_seconds: float
    blockers: tuple[str, ...]
    results: tuple[Gemma3XRTBOResult, ...]

    def format(self, *, include_records: bool = False) -> str:
        blockers = ",".join(self.blockers) if self.blockers else "none"
        lines = [
            f"xrt_runner model={self.model_variant} status={self.status} "
            f"device={self.device_index} prompt_len={self.prompt_len} "
            f"decode_context={self.decode_context} kv_strategy={self.kv_strategy} "
            f"requested={self.requested_bytes} "
            f"allocated={self.allocated_bytes} allocations={self.allocation_count} "
            f"skipped={self.skipped_count} blockers={blockers}"
        ]
        if include_records:
            lines.extend(result.format() for result in self.results)
        return "\n".join(lines)

    def to_json_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["results"] = [asdict(result) for result in self.results]
        return data


def _preload_pattern(name: str, size: int) -> bytearray:
    seed = sum(ord(ch) for ch in name) & 0xFF
    return bytearray((seed + idx) & 0xFF for idx in range(size))


def _planned_result(record: Gemma3BORecord, *, max_bo_bytes: int, remaining_bytes: int) -> Gemma3XRTBOResult:
    if remaining_bytes <= 0:
        return Gemma3XRTBOResult(record.name, record.bytes, 0, "SKIPPED_TOTAL_CAP", record.static, "none", "total byte cap reached")
    allocated = min(record.bytes, max_bo_bytes, remaining_bytes)
    status = "PLANNED_FULL" if allocated == record.bytes else "PLANNED_TRUNCATED"
    return Gemma3XRTBOResult(record.name, record.bytes, allocated, status, record.static, "planned", "dry-run")


def dry_run_allocation_plan(
    plan: Gemma3BOPlan,
    *,
    max_total_bytes: int,
    max_bo_bytes: int,
) -> Gemma3XRTRunnerReport:
    allocated_total = 0
    results: list[Gemma3XRTBOResult] = []
    for record in plan.records:
        result = _planned_result(record, max_bo_bytes=max_bo_bytes, remaining_bytes=max_total_bytes - allocated_total)
        allocated_total += result.allocated_bytes
        results.append(result)
    return Gemma3XRTRunnerReport(
        model_variant=plan.model_variant,
        status="DRY_RUN",
        device_index=-1,
        prompt_len=plan.prompt_len,
        decode_context=plan.decode_context,
        kv_strategy=plan.kv_strategy,
        max_total_bytes=max_total_bytes,
        max_bo_bytes=max_bo_bytes,
        requested_bytes=plan.total_bytes,
        allocated_bytes=allocated_total,
        allocation_count=sum(result.allocated_bytes > 0 for result in results),
        skipped_count=sum(result.allocated_bytes == 0 for result in results),
        elapsed_seconds=0.0,
        blockers=("xrt-hardware-not-touched",),
        results=tuple(results),
    )


def allocate_smoke(
    plan: Gemma3BOPlan,
    *,
    device_index: int = 0,
    max_total_bytes: int,
    max_bo_bytes: int,
    preload_static: bool = True,
    preload_dynamic: bool = False,
) -> Gemma3XRTRunnerReport:
    try:
        import pyxrt as xrt
    except Exception as exc:
        raise RuntimeError("python:pyxrt is required for XRT BO allocation") from exc

    start = perf_counter()
    device = xrt.device(device_index)
    allocated_total = 0
    bos: list[Any] = []
    results: list[Gemma3XRTBOResult] = []
    for index, record in enumerate(plan.records):
        remaining = max_total_bytes - allocated_total
        if remaining <= 0:
            results.append(Gemma3XRTBOResult(record.name, record.bytes, 0, "SKIPPED_TOTAL_CAP", record.static, "none", "total byte cap reached"))
            continue
        alloc_bytes = min(record.bytes, max_bo_bytes, remaining)
        status = "ALLOCATED_FULL" if alloc_bytes == record.bytes else "ALLOCATED_TRUNCATED"
        try:
            bo = xrt.bo(device, alloc_bytes, xrt.bo.host_only, 0)
        except Exception as exc:
            results.append(
                Gemma3XRTBOResult(
                    record.name,
                    record.bytes,
                    0,
                    "ALLOCATE_FAILED",
                    record.static,
                    "none",
                    f"{type(exc).__name__}: {exc}",
                )
            )
            for skipped in plan.records[index + 1 :]:
                results.append(
                    Gemma3XRTBOResult(
                        skipped.name,
                        skipped.bytes,
                        0,
                        "SKIPPED_AFTER_FAILURE",
                        skipped.static,
                        "none",
                        "prior BO allocation failed",
                    )
                )
            break
        preload = "none"
        if record.static and preload_static or (not record.static and preload_dynamic):
            bo.write(_preload_pattern(record.name, alloc_bytes), 0)
            bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            preload = "deterministic-pattern"
        bos.append(bo)
        allocated_total += alloc_bytes
        results.append(
            Gemma3XRTBOResult(
                record.name,
                record.bytes,
                alloc_bytes,
                status,
                record.static,
                preload,
                "smoke allocation under explicit cap",
            )
        )
    elapsed = perf_counter() - start
    full_allocation = (
        allocated_total == plan.total_bytes
        and all(result.status == "ALLOCATED_FULL" for result in results)
        and not any(result.allocated_bytes == 0 for result in results)
    )
    allocation_failed = any(result.status == "ALLOCATE_FAILED" for result in results)
    if full_allocation:
        status = "FULL_ALLOCATE_PASS"
        blockers: tuple[str, ...] = ()
    else:
        status = "ALLOCATE_SMOKE_FAIL" if allocation_failed else "ALLOCATE_SMOKE_PASS"
        blocker_list = ["paper-shape-bo-allocation-not-validated"]
        if allocation_failed:
            blocker_list.append("xrt-bo-allocation-failed")
        if any(result.status == "ALLOCATED_TRUNCATED" for result in results):
            blocker_list.append("allocation-cap-truncated")
        blockers = tuple(blocker_list)
    return Gemma3XRTRunnerReport(
        model_variant=plan.model_variant,
        status=status,
        device_index=device_index,
        prompt_len=plan.prompt_len,
        decode_context=plan.decode_context,
        kv_strategy=plan.kv_strategy,
        max_total_bytes=max_total_bytes,
        max_bo_bytes=max_bo_bytes,
        requested_bytes=plan.total_bytes,
        allocated_bytes=allocated_total,
        allocation_count=sum(result.allocated_bytes > 0 for result in results),
        skipped_count=sum(result.allocated_bytes == 0 for result in results),
        elapsed_seconds=elapsed,
        blockers=blockers,
        results=tuple(results),
    )


def _self_test() -> None:
    records = (
        Gemma3BORecord("static_projection_weights", "model", "q4nx", (4096,), 4096, True, "fixture"),
        Gemma3BORecord("layer_input", "layer", "bf16", (16, 64), 2048, False, "fixture"),
        Gemma3BORecord("kv_cache_k", "model", "bf16", (2, 16, 1, 64), 4096, False, "fixture"),
    )
    plan = Gemma3BOPlan(
        model_variant="gemma3-1b",
        status="READY_FOR_BO_ALLOCATION",
        layers=2,
        prompt_len=16,
        decode_context=16,
        kv_strategy="monolithic",
        kv_record_count=1,
        max_bo_bytes=max(record.bytes for record in records),
        total_bytes=sum(record.bytes for record in records),
        dynamic_bytes=sum(record.bytes for record in records if not record.static),
        static_bytes=sum(record.bytes for record in records if record.static),
        records=records,
        blockers=("xrt-bo-allocation-not-wired",),
    )
    report = dry_run_allocation_plan(plan, max_total_bytes=6144, max_bo_bytes=4096)
    if report.allocation_count != 2 or report.skipped_count != 1:
        raise AssertionError(report)
    if has_paper_shape_bo_allocation_evidence("gemma3-1b", path=Path("/tmp/gemma3_missing_bo_evidence.json")):
        raise AssertionError("unexpected missing BO evidence pass")
    print(report.format(include_records=True))
    print("GEMMA3_XRT_RUNNER_SELF_TEST: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma3 XRT BO allocation smoke runner")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allocate-smoke", action="store_true")
    parser.add_argument("--model-variant", choices=sorted(MODEL_SPECS), default="gemma3-1b")
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--prompt-len", type=int)
    parser.add_argument("--decode-context", type=int)
    parser.add_argument("--kv-strategy", choices=KV_STRATEGIES, default="benchmark-cell")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--max-total-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-bo-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--no-preload-static", action="store_true")
    parser.add_argument("--preload-dynamic", action="store_true")
    parser.add_argument("--include-records", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return 0
    plan = build_bo_plan(
        args.model_variant,
        weights_dir=args.weights_dir,
        prompt_len=args.prompt_len,
        decode_context=args.decode_context,
        kv_strategy=args.kv_strategy,
    )
    if args.allocate_smoke:
        report = allocate_smoke(
            plan,
            device_index=args.device_index,
            max_total_bytes=args.max_total_bytes,
            max_bo_bytes=args.max_bo_bytes,
            preload_static=not args.no_preload_static,
            preload_dynamic=args.preload_dynamic,
        )
    elif args.dry_run:
        report = dry_run_allocation_plan(
            plan,
            max_total_bytes=args.max_total_bytes,
            max_bo_bytes=args.max_bo_bytes,
        )
    else:
        parser.error("one of --self-test, --dry-run, or --allocate-smoke is required")
    print(report.format(include_records=args.include_records))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8")
        print(f"GEMMA3_XRT_RUNNER_JSON: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
