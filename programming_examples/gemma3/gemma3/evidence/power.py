#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gemma3 power telemetry and TPS-per-watt contracts."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any

RAILS = ("cpu", "gpu", "npu", "total")
MISSING_POWER_FIELD = "MISSING_POWER_FIELD"
_FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


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


@dataclass
class Gemma3PowerWindow:
    schema_version: int
    sampling_backend: str
    sample_requested: bool
    target_backend: str | None
    baseline_pkg_watts: float | None
    run_id: str | None
    rapl_start_uj: int | None = None
    rapl_max_energy_range_uj: int | None = None
    pkg_sampler_enabled: bool = False
    notes: list[str] = field(default_factory=list)
    pkg_samples: list[float] = field(default_factory=list)
    gpu_samples: list[float] = field(default_factory=list)
    stop_event: threading.Event | None = None
    sampler_thread: threading.Thread | None = None


def _dedupe(items: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item for item in items if item))


def _run(args: list[str], *, timeout: float = 2.0) -> tuple[str, str, int | None]:
    try:
        proc = subprocess.run(
            args,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return "", "not found", None
    except subprocess.TimeoutExpired as exc:
        return exc.stdout or "", exc.stderr or "timeout", None
    return proc.stdout.strip(), proc.stderr.strip(), proc.returncode


def _last_float(text: str) -> float | None:
    matches = _FLOAT_RE.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def _parse_pkg_watts(text: str) -> float | None:
    for line in text.splitlines():
        if "PkgWatt" in line:
            value = _last_float(line)
            if value is not None:
                return value
    return _last_float(text)


def _parse_rocm_smi_watts(text: str) -> float | None:
    for line in text.splitlines():
        if "Power" in line and "W" in line:
            value = _last_float(line)
            if value is not None:
                return value
    return None


def _find_rapl_package_dir() -> Path | None:
    powercap = Path("/sys/class/powercap")
    if not powercap.exists():
        return None
    candidates = sorted(powercap.glob("intel-rapl:*"))
    package_candidates: list[Path] = []
    for path in candidates:
        if not (path / "energy_uj").exists():
            continue
        name = ""
        try:
            name = (path / "name").read_text(encoding="utf-8").strip()
        except Exception:
            pass
        if name.startswith("package"):
            package_candidates.append(path)
    if package_candidates:
        return package_candidates[0]
    fallback = powercap / "intel-rapl:0"
    return fallback if (fallback / "energy_uj").exists() else None


def _read_rapl_package_energy() -> tuple[int | None, int | None, str | None]:
    package_dir = _find_rapl_package_dir()
    if package_dir is None:
        return None, None, "RAPL package energy counter was not found"
    energy, error = _read_int(package_dir / "energy_uj")
    if error or energy is None:
        return None, None, error or f"failed reading {package_dir / 'energy_uj'}"
    max_range, range_error = _read_int(package_dir / "max_energy_range_uj")
    if range_error and "permission denied" not in range_error:
        return energy, None, range_error
    return energy, max_range, None


def _rapl_delta_watts(start_uj: int, end_uj: int, max_range_uj: int | None, elapsed_seconds: float) -> tuple[float | None, str | None]:
    if elapsed_seconds <= 0.0:
        return None, "elapsed_seconds must be positive for RAPL package power calculation"
    delta_uj = end_uj - start_uj
    if delta_uj < 0 and max_range_uj:
        delta_uj += max_range_uj
    if delta_uj < 0:
        return None, "invalid negative RAPL package energy delta"
    return (delta_uj / 1_000_000.0) / elapsed_seconds, None


def _sample_rapl_pkg_watts(interval_seconds: float = 0.1) -> tuple[float | None, str | None, str]:
    start, max_range, start_error = _read_rapl_package_energy()
    if start_error or start is None:
        return None, start_error or "RAPL package start reading unavailable", "rapl-sysfs"
    time.sleep(max(interval_seconds, 0.0))
    end, _end_range, end_error = _read_rapl_package_energy()
    if end_error or end is None:
        return None, end_error or "RAPL package end reading unavailable", "rapl-sysfs"
    value, delta_error = _rapl_delta_watts(start, end, max_range, max(interval_seconds, 1e-9))
    if delta_error or value is None:
        return None, delta_error or "RAPL package delta unavailable", "rapl-sysfs"
    return value, None, "rapl-sysfs"


def _sample_turbostat_pkgwatt() -> tuple[float | None, str | None, str]:
    helper = os.environ.get("GEMMA3_TURBOSTAT_PKGWATT") or shutil.which("turbostat_pkgwatt")
    if helper:
        stdout, stderr, returncode = _run([helper], timeout=3.0)
        value = _parse_pkg_watts(stdout or stderr)
        if returncode == 0 and value is not None:
            return value, None, "turbostat_pkgwatt"
        return None, f"turbostat_pkgwatt failed: {stderr or stdout or returncode}", "turbostat_pkgwatt"

    turbostat = shutil.which("turbostat")
    if not turbostat:
        return None, "turbostat_pkgwatt and turbostat are not available", "missing"
    stdout, stderr, returncode = _run(
        [turbostat, "--quiet", "--show", "PkgWatt", "--num_iterations", "1", "--interval", "0.1"],
        timeout=4.0,
    )
    value = _parse_pkg_watts(stdout)
    if returncode == 0 and value is not None:
        return value, None, "turbostat"
    return None, f"turbostat PkgWatt failed: {stderr or stdout or returncode}", "turbostat"


def _sample_rocm_smi_watts() -> tuple[float | None, str | None, str]:
    rocm_smi = shutil.which("rocm-smi")
    if rocm_smi:
        stdout, stderr, returncode = _run([rocm_smi, "--showpower"], timeout=4.0)
        value = _parse_rocm_smi_watts(stdout or stderr)
        if returncode == 0 and value is not None:
            return value, None, "rocm-smi"
        return None, f"rocm-smi --showpower failed: {stderr or stdout or returncode}", "rocm-smi"
    amd_smi = shutil.which("amd-smi")
    if amd_smi:
        stdout, stderr, returncode = _run([amd_smi, "metric", "-p"], timeout=4.0)
        value = _parse_rocm_smi_watts(stdout or stderr)
        if returncode == 0 and value is not None:
            return value, None, "amd-smi"
        return None, f"amd-smi metric -p failed: {stderr or stdout or returncode}", "amd-smi"
    return None, "rocm-smi and amd-smi are not available", "missing"


def _wants_pkg_sampler(target_backend: str | None) -> bool:
    return target_backend in (None, "cpu", "npu")


def _wants_gpu_sampler(target_backend: str | None) -> bool:
    return target_backend in ("igpu", "gpu")


def _available_backend() -> tuple[str, list[str]]:
    notes: list[str] = []
    backends: list[str] = []
    if _find_rapl_package_dir() is not None:
        backends.append("rapl-sysfs")
    if os.environ.get("GEMMA3_TURBOSTAT_PKGWATT") or shutil.which("turbostat_pkgwatt"):
        backends.append("turbostat_pkgwatt")
    elif shutil.which("turbostat"):
        backends.append("turbostat")
        notes.append("raw turbostat is available as a package-watt fallback, but may require matching linux-tools and privileges")
    if shutil.which("rocm-smi"):
        backends.append("rocm-smi")
    elif shutil.which("amd-smi"):
        backends.append("amd-smi")
    if shutil.which("xrt-smi"):
        notes.append("xrt-smi is available, but direct NPU rail parsing reports N/A on this host")
    if not backends:
        notes.append("no supported power telemetry backend found")
        return "missing", notes
    return "+".join(backends), notes


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


def _sampler_loop(window: Gemma3PowerWindow, sample_interval_seconds: float) -> None:
    assert window.stop_event is not None
    while not window.stop_event.is_set():
        if window.pkg_sampler_enabled:
            value, note, _backend = _sample_turbostat_pkgwatt()
            if value is not None:
                window.pkg_samples.append(value)
            if note:
                window.notes.append(note)
        if _wants_gpu_sampler(window.target_backend):
            value, note, _backend = _sample_rocm_smi_watts()
            if value is not None:
                window.gpu_samples.append(value)
            if note:
                window.notes.append(note)
        window.stop_event.wait(sample_interval_seconds)


def begin_power_window(
    *,
    sample: bool,
    run_id: str | None = None,
    target_backend: str | None = None,
    sample_interval_seconds: float = 0.25,
) -> Gemma3PowerWindow:
    if not sample:
        return Gemma3PowerWindow(
            schema_version=1,
            sampling_backend="not-requested",
            sample_requested=False,
            target_backend=target_backend,
            baseline_pkg_watts=None,
            run_id=run_id,
            notes=["power sampling was not requested"],
        )
    backend, notes = _available_backend()
    baseline_pkg_watts = None
    rapl_start_uj = None
    rapl_max_energy_range_uj = None
    pkg_sampler_enabled = False
    if _wants_pkg_sampler(target_backend):
        baseline_pkg_watts, baseline_note, baseline_backend = _sample_rapl_pkg_watts()
        if baseline_backend not in backend:
            backend = "+".join(item for item in (backend, baseline_backend) if item and item != "missing") or baseline_backend
        if baseline_note:
            notes.append("quiescent direct RAPL package power sample unavailable: " + baseline_note)
            baseline_pkg_watts, fallback_note, fallback_backend = _sample_turbostat_pkgwatt()
            if fallback_backend not in backend:
                backend = "+".join(item for item in (backend, fallback_backend) if item and item != "missing") or fallback_backend
            if fallback_note:
                notes.append("quiescent fallback package power sample unavailable: " + fallback_note)
        rapl_start_uj, rapl_max_energy_range_uj, rapl_start_error = _read_rapl_package_energy()
        if rapl_start_error or rapl_start_uj is None:
            notes.append("timed direct RAPL package start unavailable: " + (rapl_start_error or "missing start reading"))
            pkg_sampler_enabled = True
    window = Gemma3PowerWindow(
        schema_version=1,
        sampling_backend=backend,
        sample_requested=True,
        target_backend=target_backend,
        baseline_pkg_watts=baseline_pkg_watts,
        run_id=run_id,
        rapl_start_uj=rapl_start_uj,
        rapl_max_energy_range_uj=rapl_max_energy_range_uj,
        pkg_sampler_enabled=pkg_sampler_enabled,
        notes=list(notes),
        stop_event=threading.Event(),
    )
    if pkg_sampler_enabled or _wants_gpu_sampler(target_backend):
        window.sampler_thread = threading.Thread(
            target=_sampler_loop,
            args=(window, sample_interval_seconds),
            daemon=True,
        )
        window.sampler_thread.start()
    return window


def _set_watt(watts: dict[str, float | None], status: dict[str, str], rail: str, value: float, field_status: str) -> None:
    watts[rail] = value
    status[rail] = field_status


def finish_power_window(window: Gemma3PowerWindow, *, elapsed_seconds: float) -> Gemma3PowerSnapshot:
    watts = {rail: None for rail in RAILS}
    status = {rail: MISSING_POWER_FIELD for rail in RAILS}
    notes = list(window.notes)
    if window.stop_event is not None:
        window.stop_event.set()
    if window.sampler_thread is not None:
        window.sampler_thread.join(timeout=2.0)
    aligned = False
    backend = window.sampling_backend
    if not window.sample_requested:
        return Gemma3PowerSnapshot(1, backend, False, watts, status, window.run_id, tuple(notes))
    if elapsed_seconds <= 0.0:
        notes.append("elapsed_seconds must be positive for power calculation")
    pkg_avg = fmean(window.pkg_samples) if window.pkg_samples else None
    pkg_status = "PKGWATT_PACKAGE"
    pkg_note = "package watts use turbostat package-watt fallback"
    if window.rapl_start_uj is not None:
        rapl_end_uj, _rapl_end_range, rapl_end_error = _read_rapl_package_energy()
        if rapl_end_error or rapl_end_uj is None:
            notes.append("timed direct RAPL package end unavailable: " + (rapl_end_error or "missing end reading"))
        else:
            rapl_pkg_avg, rapl_delta_error = _rapl_delta_watts(
                window.rapl_start_uj,
                rapl_end_uj,
                window.rapl_max_energy_range_uj,
                elapsed_seconds,
            )
            if rapl_delta_error or rapl_pkg_avg is None:
                notes.append(rapl_delta_error or "direct RAPL package delta unavailable")
            else:
                pkg_avg = rapl_pkg_avg
                pkg_status = "RAPL_SYSFS_PACKAGE"
                pkg_note = "package watts use direct RAPL sysfs energy_uj delta over the timed window"
    gpu_avg = fmean(window.gpu_samples) if window.gpu_samples else None
    if window.target_backend == "cpu" and pkg_avg is not None:
        _set_watt(watts, status, "cpu", pkg_avg, pkg_status)
        _set_watt(watts, status, "total", pkg_avg, pkg_status)
        notes.append("CPU rail uses package watts; package power is treated as total for this CPU baseline")
        notes.append(pkg_note)
    elif window.target_backend == "npu" and pkg_avg is not None:
        _set_watt(watts, status, "total", pkg_avg, pkg_status)
        if window.baseline_pkg_watts is not None:
            pseudo_npu = max(pkg_avg - window.baseline_pkg_watts, 0.0)
            pseudo_status = "PSEUDO_RAPL_SYSFS_DELTA" if pkg_status == "RAPL_SYSFS_PACKAGE" else "PSEUDO_PKGWATT_DELTA"
            _set_watt(watts, status, "npu", pseudo_npu, pseudo_status)
            notes.append(
                "NPU rail is pseudo power: average package watts during timed window minus quiescent package watts before the run"
            )
            notes.append(f"quiescent_pkg_watts={window.baseline_pkg_watts:.6f}")
            notes.append(pkg_note)
        else:
            notes.append("pseudo-NPU power requires a quiescent package-watt sample before the run")
    elif window.target_backend in ("igpu", "gpu") and gpu_avg is not None:
        _set_watt(watts, status, "gpu", gpu_avg, "ROCM_SMI_SOCKET_GRAPHICS")
        notes.append("iGPU rail uses ROCm SMI socket graphics package power sampled during the timed window")
        if pkg_avg is not None:
            _set_watt(watts, status, "total", pkg_avg, pkg_status)
    elif pkg_avg is not None:
        _set_watt(watts, status, "total", pkg_avg, "PKGWATT_PACKAGE")
        notes.append("package-watt samples were captured, but no target backend mapped them to a specific rail")
    if gpu_avg is not None and window.target_backend not in ("igpu", "gpu"):
        notes.append("ROCm SMI iGPU samples were ignored because the timed backend is not iGPU")
    aligned = any(value is not None for value in watts.values())
    if not aligned and not notes:
        notes.append("power sampling requested but no readable timed-window telemetry was available")
    if pkg_avg is None and _wants_pkg_sampler(window.target_backend):
        notes.append("no timed-window package power data was captured")
    if not window.gpu_samples and _wants_gpu_sampler(window.target_backend):
        notes.append("no timed-window ROCm SMI samples were captured")
    return Gemma3PowerSnapshot(
        schema_version=1,
        sampling_backend=backend,
        aligned_with_timed_window=aligned,
        watts=watts,
        field_status=status,
        run_id=window.run_id,
        notes=_dedupe(notes),
    )


def capture_power_snapshot(*, sample: bool, run_id: str | None = None) -> Gemma3PowerSnapshot:
    backend, notes = _available_backend()
    if not sample:
        notes.append("power sampling was not requested")
    else:
        notes.append("instantaneous snapshots do not report watts; use begin_power_window/finish_power_window around timed inference")
        rapl_energy, _rapl_range, rapl_error = _read_rapl_package_energy()
        if rapl_energy is not None:
            notes.append("direct RAPL sysfs package counter is readable; timed-window integration will use energy_uj deltas")
        else:
            notes.append("direct RAPL sysfs package counter is unavailable: " + (rapl_error or "missing counter"))
            readings, _ranges, rapl_notes = _read_rapl_energy()
            if not readings:
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
        notes=_dedupe(notes),
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
    if _parse_pkg_watts("PkgWatt\n12.5") != 12.5:
        raise AssertionError("PkgWatt parser failed")
    if _parse_rocm_smi_watts("Current Socket Graphics Package Power (W): 9.029") != 9.029:
        raise AssertionError("ROCm SMI parser failed")
    watts, err = _rapl_delta_watts(100, 1_100_100, None, 1.1)
    if err or watts != 1.0:
        raise AssertionError((watts, err))
    window = Gemma3PowerWindow(
        schema_version=1,
        sampling_backend="fixture",
        sample_requested=True,
        target_backend="npu",
        baseline_pkg_watts=4.0,
        run_id="self-test",
        pkg_samples=[5.0, 7.0],
    )
    pseudo = finish_power_window(window, elapsed_seconds=1.0)
    if pseudo.watts["npu"] != 2.0 or pseudo.field_status["npu"] != "PSEUDO_PKGWATT_DELTA":
        raise AssertionError(pseudo)
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
    parser.add_argument("--backend", choices=["cpu", "igpu", "npu"])
    parser.add_argument("--window-seconds", type=float)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--run-id")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return 0
    if args.window_seconds is not None:
        window = begin_power_window(sample=args.sample, run_id=args.run_id, target_backend=args.backend)
        if args.sample:
            time.sleep(max(args.window_seconds, 0.0))
        snapshot = finish_power_window(window, elapsed_seconds=max(args.window_seconds or 0.0, 1e-9))
    else:
        snapshot = capture_power_snapshot(sample=args.sample, run_id=args.run_id)
    if args.json:
        print(json.dumps(snapshot.to_json_dict(), indent=2, sort_keys=True))
    else:
        print(format_snapshot(snapshot))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
