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

from gemma3_artifacts import MODEL_SPECS, default_weights_dir

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
DEFAULT_TENSOR_KEY = "model.layers.0.input_layernorm.weight"
DEFAULT_INPUT_DISTRIBUTION = "bounded-uniform-seed0"
PROBE_KINDS = ("model-rmsnorm", "standalone-rmsnorm")
_CORRELATION_RE = re.compile(r"Output\s+0\s+correlation:\s+(?P<value>[-+0-9.eE]+)\s+\(threshold:\s+(?P<threshold>[-+0-9.eE]+)\)")


@dataclass(frozen=True)
class Gemma3KernelLaunchProbeResult:
    schema_version: int
    model_variant: str
    status: str
    probe_kind: str
    phase: str
    layer_index: int
    role: str
    kernel: str
    route: str
    shape: tuple[int, int]
    output_format: str
    herd_x: int
    tensor_key: str | None
    input_distribution: str | None
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
        tensor = self.tensor_key or "none"
        distribution = self.input_distribution or "none"
        return (
            f"launch_probe model={self.model_variant} status={self.status} "
            f"probe={self.probe_kind} phase={self.phase} layer=L{self.layer_index} role={self.role} "
            f"kernel={self.kernel} route={self.route} shape={shape} "
            f"output_format={self.output_format} herd_x={self.herd_x} "
            f"tensor={tensor} input={distribution} correlation={correlation} "
            f"threshold={self.threshold:g} blockers={blockers}"
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
        repo / "programming_examples" / "weighted_rms_norm",
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


def _resolve_weights_dir(model_variant: str, weights_dir: Path | None) -> Path:
    return (weights_dir or default_weights_dir(model_variant)).expanduser()


def _build_standalone_command(*, repo: Path, m: int, n: int, herd_x: int, output_format: str) -> tuple[str, ...]:
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


def _build_model_worker_command(
    *,
    repo: Path,
    model_variant: str,
    weights_dir: Path,
    m: int,
    n: int,
    herd_x: int,
    output_format: str,
    tensor_key: str,
) -> tuple[str, ...]:
    return (
        sys.executable,
        str(Path(__file__).resolve()),
        "--_model-rmsnorm-worker",
        "--model-variant",
        model_variant,
        "--weights-dir",
        str(weights_dir),
        "--m",
        str(m),
        "--n",
        str(n),
        "--herd-x",
        str(herd_x),
        "--output-format",
        output_format,
        "--tensor-key",
        tensor_key,
    )


def _result_from_completed_process(
    *,
    model_variant: str,
    probe_kind: str,
    tensor_key: str | None,
    input_distribution: str | None,
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
        probe_kind=probe_kind,
        phase=DEFAULT_PHASE,
        layer_index=DEFAULT_LAYER,
        role=DEFAULT_ROLE,
        kernel=DEFAULT_KERNEL,
        route=DEFAULT_ROUTE,
        shape=(m, n),
        output_format=output_format,
        herd_x=herd_x,
        tensor_key=tensor_key,
        input_distribution=input_distribution,
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


def _run_probe_command(
    *,
    model_variant: str,
    probe_kind: str,
    tensor_key: str | None,
    input_distribution: str | None,
    command: tuple[str, ...],
    m: int,
    n: int,
    herd_x: int,
    output_format: str,
) -> Gemma3KernelLaunchProbeResult:
    repo = _repo_root()
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
        probe_kind=probe_kind,
        tensor_key=tensor_key,
        input_distribution=input_distribution,
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


def run_rmsnorm_launch_probe(
    *,
    model_variant: str,
    weights_dir: Path | None,
    probe_kind: str,
    m: int,
    n: int,
    herd_x: int,
    output_format: str,
    tensor_key: str,
) -> Gemma3KernelLaunchProbeResult:
    repo = _repo_root()
    if probe_kind == "standalone-rmsnorm":
        command = _build_standalone_command(repo=repo, m=m, n=n, herd_x=herd_x, output_format=output_format)
        return _run_probe_command(
            model_variant=model_variant,
            probe_kind=probe_kind,
            tensor_key=None,
            input_distribution="standalone-random-seed0",
            command=command,
            m=m,
            n=n,
            herd_x=herd_x,
            output_format=output_format,
        )
    if probe_kind == "model-rmsnorm":
        resolved_weights = _resolve_weights_dir(model_variant, weights_dir)
        command = _build_model_worker_command(
            repo=repo,
            model_variant=model_variant,
            weights_dir=resolved_weights,
            m=m,
            n=n,
            herd_x=herd_x,
            output_format=output_format,
            tensor_key=tensor_key,
        )
        return _run_probe_command(
            model_variant=model_variant,
            probe_kind=probe_kind,
            tensor_key=tensor_key,
            input_distribution=DEFAULT_INPUT_DISTRIBUTION,
            command=command,
            m=m,
            n=n,
            herd_x=herd_x,
            output_format=output_format,
        )
    raise ValueError(f"unsupported probe_kind: {probe_kind}")


def _load_safetensor_vector(weights_dir: Path, tensor_key: str):
    try:
        from safetensors import safe_open
    except Exception as exc:
        raise RuntimeError("python:safetensors is required for model-rmsnorm launch probe") from exc
    for path in sorted(weights_dir.glob("*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if tensor_key in handle.keys():
                return handle.get_tensor(tensor_key).float().cpu().numpy()
    raise RuntimeError(f"tensor key not found in {weights_dir}: {tensor_key}")


def _model_rmsnorm_worker(args: argparse.Namespace) -> int:
    import numpy as np
    from ml_dtypes import bfloat16
    from air.backend.xrt_runner import XRTRunner
    from weighted_rms_norm import build_module, rms_norm_reference

    weights_dir = _resolve_weights_dir(args.model_variant, args.weights_dir)
    weight = _load_safetensor_vector(weights_dir, args.tensor_key).astype(bfloat16)
    if weight.shape != (args.n,):
        raise RuntimeError(f"expected norm vector shape ({args.n},), got {weight.shape}")
    np.random.seed(0)
    x_input = np.random.rand(args.m, args.n).astype(bfloat16)
    y_expected = rms_norm_reference(x_input, weight)
    mlir_module = build_module(args.m, args.n, bfloat16, 16, herd_x=args.herd_x)
    runner = XRTRunner(
        verbose=False,
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name="weighted_rms_norm",
        runtime_loop_tiling_sizes=[4, 4],
    )
    return runner.run_test(
        mlir_module,
        inputs=[x_input, weight],
        expected_outputs=[y_expected],
        rtol=5e-2,
        atol=5e-1,
        min_correlation=DEFAULT_THRESHOLD,
    )


def _self_test() -> None:
    fake_stdout = """Weighted RMSNorm: M=1024, N=1152, herd=[1,1]
Output 0 correlation: 0.999984 (threshold: 0.99)
PASS!
"""
    result = _result_from_completed_process(
        model_variant=DEFAULT_MODEL,
        probe_kind="model-rmsnorm",
        tensor_key=DEFAULT_TENSOR_KEY,
        input_distribution=DEFAULT_INPUT_DISTRIBUTION,
        command=("python3", "gemma3_launch_probe.py"),
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
    parser.add_argument("--_model-rmsnorm-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model-variant", choices=sorted(MODEL_SPECS), default=DEFAULT_MODEL)
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--probe-kind", choices=PROBE_KINDS, default="model-rmsnorm")
    parser.add_argument("--m", type=int, default=DEFAULT_M)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--herd-x", type=int, default=DEFAULT_HERD_X)
    parser.add_argument("--output-format", choices=("elf", "xclbin"), default=DEFAULT_OUTPUT_FORMAT)
    parser.add_argument("--tensor-key", default=DEFAULT_TENSOR_KEY)
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()

    if args._model_rmsnorm_worker:
        return _model_rmsnorm_worker(args)
    if args.self_test:
        _self_test()
        return 0
    if not args.run_hardware:
        raise SystemExit("pass --run-hardware to touch the NPU; --self-test is hardware-free")
    result = run_rmsnorm_launch_probe(
        model_variant=args.model_variant,
        weights_dir=args.weights_dir,
        probe_kind=args.probe_kind,
        m=args.m,
        n=args.n,
        herd_x=args.herd_x,
        output_format=args.output_format,
        tensor_key=args.tensor_key,
    )
    print(result.format())
    if args.result_json:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(json.dumps(result.to_json_dict(), indent=2, sort_keys=True) + "\n")
        print(f"GEMMA3_LAUNCH_PROBE_JSON: {args.result_json}")
    return 0 if result.status == "FIRST_KERNEL_LAUNCH_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
