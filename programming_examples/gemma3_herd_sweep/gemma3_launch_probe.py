#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gemma3 first-kernel launch probe.

This diagnostic wrapper records the first staged proof point for the real Gemma3
model runner: a promoted RMSNorm stage using the Gemma3 1B prefill shape. It is
not an end-to-end model launch and it is not a TTFT/TPS measurement. Hardware is
only touched when --run-hardware is passed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

from gemma3_artifacts import MODEL_SPECS

DEFAULT_MODEL = "gemma3-1b"
DEFAULT_PHASE = "prefill"
DEFAULT_LAYER = 0
DEFAULT_ROLE = "pre_attention_norm"
DEFAULT_KERNEL = "rms_norm"
DEFAULT_ROUTE = "weighted_rms_norm/standalone-elf-smoke"
DEFAULT_M = 1024
DEFAULT_N = 1152
DEFAULT_OUTPUT_FORMAT = "elf"
DEFAULT_HERD_X = 1
DEFAULT_THRESHOLD = 0.99
_CORRELATION_RE = re.compile(r"Output\s+0\s+correlation:\s+(?P<value>[-+0-9.eE]+)\s+\(threshold:\s+(?P<threshold>[-+0-9.eE]+)\)")


@dataclass(frozen=True)
class Gemma3KernelLaunchProbeResult:
    schema_version: int
    model_variant: str
    status: str
    phase: str
    layer_index: int
    role: str
    kernel: str
    route: str
    shape: tuple[int, int]
    output_format: str
    herd_x: int
    command: tuple[str, ...]
    returncode: int | None
    elapsed_seconds: float | None
    correlation: float | None
    threshold: float
    blockers: tuple[str, ...]
    git_commit: str | None
    dirty_worktree: bool | None
    stdout_tail: tuple[str, ...]
    stderr_tail: tuple[str, ...]

    def format(self) -> str:
        blockers = ",".join(self.blockers) if self.blockers else "none"
        correlation = "n/a" if self.correlation is None else f"{self.correlation:.6f}"
        shape = f"{self.shape[0]}x{self.shape[1]}"
        return (
            f"launch_probe model={self.model_variant} status={self.status} "
            f"phase={self.phase} layer=L{self.layer_index} role={self.role} "
            f"kernel={self.kernel} route={self.route} shape={shape} "
            f"output_format={self.output_format} herd_x={self.herd_x} "
            f"correlation={correlation} threshold={self.threshold:g} blockers={blockers}"
        )

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _tail(text: str, limit: int = 40) -> tuple[str, ...]:
    lines = text.splitlines()
    return tuple(lines[-limit:])


def _parse_correlation(stdout: str) -> tuple[float | None, float | None]:
    match = _CORRELATION_RE.search(stdout)
    if not match:
        return None, None
    return float(match.group("value")), float(match.group("threshold"))


def _git_info(repo: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        ).stdout
    except Exception:
        return None, None
    return commit or None, bool(status.strip())


def _prepend_path(existing: str | None, paths: list[Path | str]) -> str:
    parts = [str(path) for path in paths if str(path)]
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def _probe_env(repo: Path) -> dict[str, str]:
    env = dict(os.environ)
    py_paths = [
        repo / "build-xrt" / "python",
        repo / "sandbox" / "lib" / "python3.12" / "site-packages" / "mlir_aie" / "python",
        "/opt/xilinx/xrt/python",
        "/usr/lib/python",
    ]
    path_entries = [
        repo / "build-xrt" / "bin",
        repo / "install-xrt" / "bin",
        repo / "sandbox" / "bin",
        "/opt/xilinx/xrt/bin",
    ]
    env["PYTHONPATH"] = _prepend_path(env.get("PYTHONPATH"), py_paths)
    env["PATH"] = _prepend_path(env.get("PATH"), path_entries)
    env.setdefault(
        "PEANO_INSTALL_DIR",
        str(repo / "sandbox" / "lib" / "python3.12" / "site-packages" / "llvm-aie"),
    )
    if Path("/opt/xilinx/xrt").exists():
        env.setdefault("XILINX_XRT", "/opt/xilinx/xrt")
        env["LD_LIBRARY_PATH"] = _prepend_path(env.get("LD_LIBRARY_PATH"), ["/opt/xilinx/xrt/lib"])
    return env


def _build_command(*, repo: Path, m: int, n: int, herd_x: int, output_format: str) -> tuple[str, ...]:
    return (
        sys.executable,
        str(repo / "programming_examples" / "weighted_rms_norm" / "weighted_rms_norm.py"),
        "--M",
        str(m),
        "--N",
        str(n),
        "--herd-x",
        str(herd_x),
        "--output-format",
        output_format,
        "--compile-mode",
        "compile-and-run",
    )


def _result_from_completed_process(
    *,
    model_variant: str,
    command: tuple[str, ...],
    returncode: int,
    elapsed_seconds: float,
    stdout: str,
    stderr: str,
    m: int,
    n: int,
    herd_x: int,
    output_format: str,
    git_commit: str | None,
    dirty_worktree: bool | None,
) -> Gemma3KernelLaunchProbeResult:
    correlation, parsed_threshold = _parse_correlation(stdout)
    threshold = parsed_threshold if parsed_threshold is not None else DEFAULT_THRESHOLD
    blockers: list[str] = []
    if returncode != 0:
        blockers.append("first-kernel-launch-failed")
    if correlation is None:
        blockers.append("first-kernel-launch-correlation-missing")
    elif correlation < threshold:
        blockers.append("first-kernel-launch-correlation-low")
    status = "FIRST_KERNEL_LAUNCH_PASS" if not blockers else "FIRST_KERNEL_LAUNCH_BLOCKED"
    return Gemma3KernelLaunchProbeResult(
        schema_version=1,
        model_variant=model_variant,
        status=status,
        phase=DEFAULT_PHASE,
        layer_index=DEFAULT_LAYER,
        role=DEFAULT_ROLE,
        kernel=DEFAULT_KERNEL,
        route=DEFAULT_ROUTE,
        shape=(m, n),
        output_format=output_format,
        herd_x=herd_x,
        command=command,
        returncode=returncode,
        elapsed_seconds=elapsed_seconds,
        correlation=correlation,
        threshold=threshold,
        blockers=tuple(dict.fromkeys(blockers)),
        git_commit=git_commit,
        dirty_worktree=dirty_worktree,
        stdout_tail=_tail(stdout),
        stderr_tail=_tail(stderr),
    )


def run_rmsnorm_launch_probe(
    *,
    model_variant: str,
    m: int,
    n: int,
    herd_x: int,
    output_format: str,
) -> Gemma3KernelLaunchProbeResult:
    repo = _repo_root()
    command = _build_command(repo=repo, m=m, n=n, herd_x=herd_x, output_format=output_format)
    git_commit, dirty = _git_info(repo)
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="gemma3-launch-probe-") as tmpdir:
        proc = subprocess.run(
            command,
            cwd=tmpdir,
            env=_probe_env(repo),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    elapsed = time.perf_counter() - start
    return _result_from_completed_process(
        model_variant=model_variant,
        command=command,
        returncode=proc.returncode,
        elapsed_seconds=elapsed,
        stdout=proc.stdout,
        stderr=proc.stderr,
        m=m,
        n=n,
        herd_x=herd_x,
        output_format=output_format,
        git_commit=git_commit,
        dirty_worktree=dirty,
    )


def _self_test() -> None:
    fake_stdout = """Weighted RMSNorm: M=1024, N=1152, herd=[1,1]
Output 0 correlation: 0.999984 (threshold: 0.99)
PASS!
"""
    result = _result_from_completed_process(
        model_variant=DEFAULT_MODEL,
        command=("python3", "weighted_rms_norm.py"),
        returncode=0,
        elapsed_seconds=0.125,
        stdout=fake_stdout,
        stderr="",
        m=DEFAULT_M,
        n=DEFAULT_N,
        herd_x=DEFAULT_HERD_X,
        output_format=DEFAULT_OUTPUT_FORMAT,
        git_commit="fixture",
        dirty_worktree=False,
    )
    if result.status != "FIRST_KERNEL_LAUNCH_PASS":
        raise AssertionError(result)
    if result.correlation != 0.999984 or result.threshold != 0.99:
        raise AssertionError((result.correlation, result.threshold))
    print(result.format())
    print("GEMMA3_LAUNCH_PROBE_SELF_TEST: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma3 first-kernel launch probe")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--run-hardware", action="store_true")
    parser.add_argument("--model-variant", choices=sorted(MODEL_SPECS), default=DEFAULT_MODEL)
    parser.add_argument("--m", type=int, default=DEFAULT_M)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--herd-x", type=int, default=DEFAULT_HERD_X)
    parser.add_argument("--output-format", choices=("elf", "xclbin"), default=DEFAULT_OUTPUT_FORMAT)
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return 0
    if not args.run_hardware:
        raise SystemExit("pass --run-hardware to touch the NPU; --self-test is hardware-free")
    result = run_rmsnorm_launch_probe(
        model_variant=args.model_variant,
        m=args.m,
        n=args.n,
        herd_x=args.herd_x,
        output_format=args.output_format,
    )
    print(result.format())
    if args.result_json:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(json.dumps(result.to_json_dict(), indent=2, sort_keys=True) + "\n")
        print(f"GEMMA3_LAUNCH_PROBE_JSON: {args.result_json}")
    return 0 if result.status == "FIRST_KERNEL_LAUNCH_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
