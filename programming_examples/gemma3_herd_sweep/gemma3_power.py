#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gemma3 power telemetry and TPS-per-watt contracts."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

RAILS = ("cpu", "gpu", "npu", "total")
MISSING_POWER_FIELD = "MISSING_POWER_FIELD"


@dataclass(frozen=True)
class Gemma3PowerSnapshot:
    schema_version: int
    sampling_backend: str
    aligned_with_timed_window: bool
    watts: dict[str, float | None]
    field_status: dict[str, str]
    run_id: str | None
    notes: tuple[str, ...]

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


def _available_backend() -> tuple[str, list[str]]:
    notes: list[str] = []
    if shutil.which("xrt-smi"):
        notes.append("xrt-smi is available, but rail parsing is not implemented")
        return "xrt-smi-unimplemented", notes
    rapl = Path("/sys/class/powercap")
    if rapl.exists():
        notes.append("RAPL path exists, but Gemma3 timed-window integration is not implemented")
        return "rapl-unimplemented", notes
    notes.append("no supported power telemetry backend found")
    return "missing", notes


def capture_power_snapshot(*, sample: bool, run_id: str | None = None) -> Gemma3PowerSnapshot:
    backend, notes = _available_backend()
    if not sample:
        notes.append("power sampling was not requested")
    watts = {rail: None for rail in RAILS}
    status = {rail: MISSING_POWER_FIELD for rail in RAILS}
    return Gemma3PowerSnapshot(
        schema_version=1,
        sampling_backend=backend,
        aligned_with_timed_window=False,
        watts=watts,
        field_status=status,
        run_id=run_id,
        notes=tuple(notes),
    )


def tps_per_watt(throughput_tps: float | None, watts: float | None) -> tuple[float | None, str]:
    if throughput_tps is None:
        return None, "MISSING_THROUGHPUT_FIELD"
    if watts is None:
        return None, MISSING_POWER_FIELD
    if watts <= 0.0:
        return None, "INVALID_POWER_FIELD"
    return throughput_tps / watts, "PASS"


def compare_tps_per_watt(
    *,
    npu_tps: float | None,
    baseline_tps: float | None,
    npu_watts: float | None,
    baseline_watts: float | None,
) -> dict[str, object]:
    npu_value, npu_status = tps_per_watt(npu_tps, npu_watts)
    baseline_value, baseline_status = tps_per_watt(baseline_tps, baseline_watts)
    if npu_value is None or baseline_value is None:
        return {
            "npu_tps_per_watt": npu_value,
            "baseline_tps_per_watt": baseline_value,
            "speedup": None,
            "classification": "MISSING_POWER_FIELD",
            "npu_status": npu_status,
            "baseline_status": baseline_status,
        }
    return {
        "npu_tps_per_watt": npu_value,
        "baseline_tps_per_watt": baseline_value,
        "speedup": npu_value / baseline_value,
        "classification": "PASS",
        "npu_status": npu_status,
        "baseline_status": baseline_status,
    }


def format_snapshot(snapshot: Gemma3PowerSnapshot) -> str:
    rail_status = ",".join(f"{rail}={snapshot.field_status[rail]}" for rail in RAILS)
    return (
        f"GEMMA3_POWER_SNAPSHOT backend={snapshot.sampling_backend} "
        f"aligned={snapshot.aligned_with_timed_window} {rail_status}"
    )


def _self_test() -> None:
    snapshot = capture_power_snapshot(sample=True, run_id="self-test")
    for rail in RAILS:
        if snapshot.watts[rail] is not None:
            raise AssertionError(f"unexpected watt value for {rail}: {snapshot.watts[rail]}")
        if snapshot.field_status[rail] != MISSING_POWER_FIELD:
            raise AssertionError(f"unexpected status for {rail}: {snapshot.field_status[rail]}")
    missing = compare_tps_per_watt(
        npu_tps=41.1,
        baseline_tps=10.0,
        npu_watts=None,
        baseline_watts=5.0,
    )
    if missing["classification"] != MISSING_POWER_FIELD:
        raise AssertionError(missing)
    present = compare_tps_per_watt(
        npu_tps=40.0,
        baseline_tps=10.0,
        npu_watts=2.0,
        baseline_watts=5.0,
    )
    if present["classification"] != "PASS" or present["speedup"] != 10.0:
        raise AssertionError(present)
    print(format_snapshot(snapshot))
    print("GEMMA3_POWER_CONTRACT_SELF_TEST: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma3 power telemetry contract")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--run-id")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return 0
    snapshot = capture_power_snapshot(sample=args.sample, run_id=args.run_id)
    if args.json:
        print(json.dumps(snapshot.to_json_dict(), indent=2, sort_keys=True))
    else:
        print(format_snapshot(snapshot))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
