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


@dataclass(frozen=True)
class Gemma3PowerWindow:
    schema_version: int
    sampling_backend: str
    sample_requested: bool
    start_readings: dict[str, int]
    max_energy_range_uj: dict[str, int | None]
    run_id: str | None
    notes: tuple[str, ...]


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


def _read_int(path: Path) -> tuple[int | None, str | None]:
    try:
        return int(path.read_text(encoding="utf-8").strip()), None
    except PermissionError:
        return None, f"permission denied reading {path}"
    except FileNotFoundError:
        return None, f"missing {path}"
    except Exception as exc:
        return None, f"failed reading {path}: {exc}"


def _read_rapl_energy() -> tuple[dict[str, int], dict[str, int | None], list[str]]:
    notes: list[str] = []
    readings: dict[str, int] = {}
    ranges: dict[str, int | None] = {}
    powercap = Path("/sys/class/powercap")
    if not powercap.exists():
        notes.append("RAPL powercap path is missing")
        return readings, ranges, notes
    candidates = sorted(
        path
        for path in powercap.glob("intel-rapl:*")
        if (path / "energy_uj").exists()
    )
    if not candidates:
        notes.append("RAPL energy counters were not found")
        return readings, ranges, notes
    for path in candidates:
        name_path = path / "name"
        name = path.name
        try:
            name = name_path.read_text(encoding="utf-8").strip() or path.name
        except Exception:
            pass
        rail = "cpu" if name.startswith(("package", "core")) else name
        energy, error = _read_int(path / "energy_uj")
        if error:
            notes.append(error)
            continue
        if energy is None:
            continue
        max_range, range_error = _read_int(path / "max_energy_range_uj")
        if range_error and "permission denied" not in range_error:
            notes.append(range_error)
        if rail in readings:
            notes.append(f"duplicate RAPL rail {rail} from {path}; keeping first reading")
            continue
        readings[rail] = energy
        ranges[rail] = max_range
    if not readings:
        notes.append("no readable RAPL energy counters")
    return readings, ranges, notes


def begin_power_window(*, sample: bool, run_id: str | None = None) -> Gemma3PowerWindow:
    if not sample:
        return Gemma3PowerWindow(
            schema_version=1,
            sampling_backend="not-requested",
            sample_requested=False,
            start_readings={},
            max_energy_range_uj={},
            run_id=run_id,
            notes=("power sampling was not requested",),
        )
    readings, ranges, notes = _read_rapl_energy()
    backend = "rapl" if readings else "rapl-unavailable"
    return Gemma3PowerWindow(
        schema_version=1,
        sampling_backend=backend,
        sample_requested=True,
        start_readings=readings,
        max_energy_range_uj=ranges,
        run_id=run_id,
        notes=tuple(notes),
    )


def finish_power_window(window: Gemma3PowerWindow, *, elapsed_seconds: float) -> Gemma3PowerSnapshot:
    watts = {rail: None for rail in RAILS}
    status = {rail: MISSING_POWER_FIELD for rail in RAILS}
    notes = list(window.notes)
    aligned = False
    backend = window.sampling_backend
    if not window.sample_requested:
        return Gemma3PowerSnapshot(
            schema_version=1,
            sampling_backend=backend,
            aligned_with_timed_window=False,
            watts=watts,
            field_status=status,
            run_id=window.run_id,
            notes=tuple(notes),
        )
    if elapsed_seconds <= 0.0:
        notes.append("elapsed_seconds must be positive for power calculation")
    elif window.start_readings:
        end_readings, _ranges, end_notes = _read_rapl_energy()
        notes.extend(end_notes)
        for rail, start_uj in window.start_readings.items():
            if rail not in end_readings:
                notes.append(f"missing ending RAPL reading for {rail}")
                continue
            end_uj = end_readings[rail]
            delta_uj = end_uj - start_uj
            max_range = window.max_energy_range_uj.get(rail)
            if delta_uj < 0 and max_range:
                delta_uj += max_range
            if delta_uj < 0:
                notes.append(f"invalid negative RAPL energy delta for {rail}")
                continue
            mapped_rail = "cpu" if rail not in RAILS else rail
            watts[mapped_rail] = (delta_uj / 1_000_000.0) / elapsed_seconds
            status[mapped_rail] = "PASS"
        if watts["cpu"] is not None and watts["total"] is None:
            watts["total"] = watts["cpu"]
            status["total"] = "PASS"
            notes.append("RAPL CPU package power mapped to total because no separate total rail is available")
        aligned = any(value is not None for value in watts.values())
    if not aligned and not notes:
        notes.append("power sampling requested but no readable timed-window telemetry was available")
    return Gemma3PowerSnapshot(
        schema_version=1,
        sampling_backend=backend,
        aligned_with_timed_window=aligned,
        watts=watts,
        field_status=status,
        run_id=window.run_id,
        notes=tuple(notes),
    )


def capture_power_snapshot(*, sample: bool, run_id: str | None = None) -> Gemma3PowerSnapshot:
    backend, notes = _available_backend()
    if not sample:
        notes.append("power sampling was not requested")
    if sample:
        readings, _ranges, rapl_notes = _read_rapl_energy()
        if readings:
            backend = "rapl-window-required"
            notes.append("RAPL counters are readable, but a timed window is required for watts")
        else:
            notes.extend(rapl_notes)
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
