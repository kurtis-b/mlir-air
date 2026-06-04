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
MODEL_RMSNORM_ARGUMENT_KEYS = (
    "layer_input",
    "static_norm_weights",
    "prefill_L0_pre_attention_norm",
)
_CORRELATION_RE = re.compile(r"Output\s+0\s+correlation:\s+(?P<value>[-+0-9.eE]+)\s+\(threshold:\s+(?P<threshold>[-+0-9.eE]+)\)")


@dataclass(frozen=True)
class Gemma3LaunchArgumentBindingEvidence:
    status: str
    argument_count: int | None
    argument_keys: tuple[str, ...]
    argument_directions: tuple[str, ...]
    argument_storage: tuple[str, ...]
    argument_shapes: tuple[str, ...]
    static_norm_tensor_key: str | None
    static_norm_tensor_offset_bytes: int | None
    static_norm_tensor_bytes: int | None
    static_norm_bo_bytes: int | None
    blockers: tuple[str, ...]


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
    argument_binding: Gemma3LaunchArgumentBindingEvidence | None
    remaining_model_runner_gaps: tuple[str, ...]
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
        if self.argument_binding is None:
            arg_binding = "none"
            arg_count = "n/a"
            arg_keys = "none"
            static_norm = "none"
        else:
            arg_binding = self.argument_binding.status
            arg_count = (
                "n/a"
                if self.argument_binding.argument_count is None
                else str(self.argument_binding.argument_count)
            )
            arg_keys = "|".join(self.argument_binding.argument_keys) or "none"
            if self.argument_binding.static_norm_tensor_key is None:
                static_norm = "none"
            else:
                static_norm = (
                    f"{self.argument_binding.static_norm_tensor_key}"
                    f"@{self.argument_binding.static_norm_tensor_offset_bytes}"
                    f"+{self.argument_binding.static_norm_tensor_bytes}"
                    f"/bo={self.argument_binding.static_norm_bo_bytes}"
                )
        gaps = ",".join(self.remaining_model_runner_gaps) if self.remaining_model_runner_gaps else "none"
        return (
            f"launch_probe model={self.model_variant} status={self.status} "
            f"probe={self.probe_kind} phase={self.phase} layer=L{self.layer_index} role={self.role} "
            f"kernel={self.kernel} route={self.route} shape={shape} "
            f"output_format={self.output_format} herd_x={self.herd_x} "
            f"tensor={tensor} input={distribution} correlation={correlation} "
            f"threshold={self.threshold:g} arg_binding={arg_binding} "
            f"arg_count={arg_count} arg_keys={arg_keys} static_norm={static_norm} "
            f"model_runner_gaps={gaps} blockers={blockers}"
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


def _shape_text(shape: tuple[int, ...]) -> str:
    return "x".join(str(dim) for dim in shape) if shape else "scalar"


def _norm_tensor_subspan(
    *,
    model_variant: str,
    weights_dir: Path,
    tensor_key: str,
) -> tuple[int | None, int | None, int | None, tuple[str, ...]]:
    try:
        from gemma3_norm_weight_plan import build_norm_weight_plan
    except Exception as exc:
        return None, None, None, (f"norm-weight-plan-import-failed:{exc}",)
    try:
        plan = build_norm_weight_plan(model_variant, weights_dir=weights_dir)
    except Exception as exc:
        return None, None, None, (f"norm-weight-plan-failed:{exc}",)
    offset = 0
    for record in plan.records:
        if record.tensor_key == tensor_key:
            return offset, record.static_bo_bytes, plan.static_bo_bytes, ()
        offset += record.static_bo_bytes
    return None, None, plan.static_bo_bytes, ("first-stage-static-norm-tensor-not-planned",)


def _first_stage_argument_binding_evidence(
    *,
    model_variant: str,
    weights_dir: Path,
    m: int,
    n: int,
    tensor_key: str,
) -> Gemma3LaunchArgumentBindingEvidence:
    blockers: list[str] = []
    binding = None
    try:
        from gemma3_argument_binding import build_argument_binding_plan
    except Exception as exc:
        return Gemma3LaunchArgumentBindingEvidence(
            status="ARGUMENT_BINDING_BLOCKED",
            argument_count=None,
            argument_keys=(),
            argument_directions=(),
            argument_storage=(),
            argument_shapes=(),
            static_norm_tensor_key=tensor_key,
            static_norm_tensor_offset_bytes=None,
            static_norm_tensor_bytes=None,
            static_norm_bo_bytes=None,
            blockers=(f"argument-binding-import-failed:{exc}",),
        )
    try:
        decode_context = MODEL_SPECS[model_variant].max_decode_context
        plan = build_argument_binding_plan(
            model_variant,
            weights_dir=weights_dir,
            prompt_len=m,
            decode_context=decode_context,
        )
        for candidate in plan.bindings:
            if (
                candidate.phase == DEFAULT_PHASE
                and candidate.layer_index == DEFAULT_LAYER
                and candidate.role == DEFAULT_ROLE
            ):
                binding = candidate
                break
        if plan.status != "READY_FOR_KERNEL_LAUNCH":
            blockers.extend(plan.blockers or ("argument-binding-plan-blocked",))
    except Exception as exc:
        return Gemma3LaunchArgumentBindingEvidence(
            status="ARGUMENT_BINDING_BLOCKED",
            argument_count=None,
            argument_keys=(),
            argument_directions=(),
            argument_storage=(),
            argument_shapes=(),
            static_norm_tensor_key=tensor_key,
            static_norm_tensor_offset_bytes=None,
            static_norm_tensor_bytes=None,
            static_norm_bo_bytes=None,
            blockers=(f"argument-binding-plan-failed:{exc}",),
        )
    if binding is None:
        return Gemma3LaunchArgumentBindingEvidence(
            status="ARGUMENT_BINDING_BLOCKED",
            argument_count=None,
            argument_keys=(),
            argument_directions=(),
            argument_storage=(),
            argument_shapes=(),
            static_norm_tensor_key=tensor_key,
            static_norm_tensor_offset_bytes=None,
            static_norm_tensor_bytes=None,
            static_norm_bo_bytes=None,
            blockers=("first-stage-argument-binding-missing",),
        )

    args = binding.arguments
    keys = tuple(argument.key for argument in args)
    directions = tuple(argument.direction for argument in args)
    storage = tuple(argument.storage for argument in args)
    shapes = tuple(_shape_text(argument.shape) for argument in args)
    if binding.status != "ARGUMENT_BINDING_VALIDATED":
        blockers.extend(binding.blockers or ("first-stage-argument-binding-blocked",))
    if binding.kernel != DEFAULT_KERNEL or binding.route != DEFAULT_ROUTE:
        blockers.append("first-stage-kernel-route-mismatch")
    if keys != MODEL_RMSNORM_ARGUMENT_KEYS:
        blockers.append("first-stage-argument-key-order-mismatch")
    if binding.argument_count != len(MODEL_RMSNORM_ARGUMENT_KEYS):
        blockers.append("first-stage-argument-count-mismatch")
    if len(args) >= 3:
        expected_shapes = ((m, n), args[1].shape, (m, n))
        if tuple(argument.shape for argument in args) != expected_shapes:
            blockers.append("first-stage-argument-shape-mismatch")
        if args[0].dtype != "bf16" or args[1].dtype != "bf16" or args[2].dtype != "bf16":
            blockers.append("first-stage-argument-dtype-mismatch")
        if args[0].direction != "input" or args[1].direction != "static" or args[2].direction != "output":
            blockers.append("first-stage-argument-direction-mismatch")
    offset, tensor_bytes, bo_bytes, subspan_blockers = _norm_tensor_subspan(
        model_variant=model_variant,
        weights_dir=weights_dir,
        tensor_key=tensor_key,
    )
    blockers.extend(subspan_blockers)
    if tensor_bytes != n * 2:
        blockers.append("first-stage-static-norm-size-mismatch")
    if len(args) >= 2 and bo_bytes is not None and args[1].bytes != bo_bytes:
        blockers.append("first-stage-static-norm-bo-size-mismatch")
    if offset not in (0, None):
        blockers.append("first-stage-static-norm-suboffset-not-zero")
    blockers = list(dict.fromkeys(blockers))
    return Gemma3LaunchArgumentBindingEvidence(
        status="ARGUMENT_BINDING_VALIDATED" if not blockers else "ARGUMENT_BINDING_BLOCKED",
        argument_count=binding.argument_count,
        argument_keys=keys,
        argument_directions=directions,
        argument_storage=storage,
        argument_shapes=shapes,
        static_norm_tensor_key=tensor_key,
        static_norm_tensor_offset_bytes=offset,
        static_norm_tensor_bytes=tensor_bytes,
        static_norm_bo_bytes=bo_bytes,
        blockers=tuple(blockers),
    )


def _fixture_argument_binding_evidence() -> Gemma3LaunchArgumentBindingEvidence:
    return Gemma3LaunchArgumentBindingEvidence(
        status="ARGUMENT_BINDING_VALIDATED",
        argument_count=3,
        argument_keys=MODEL_RMSNORM_ARGUMENT_KEYS,
        argument_directions=("input", "static", "output"),
        argument_storage=("persistent-bo", "persistent-bo", "virtual-buffer"),
        argument_shapes=("1024x1152", "133120", "1024x1152"),
        static_norm_tensor_key=DEFAULT_TENSOR_KEY,
        static_norm_tensor_offset_bytes=0,
        static_norm_tensor_bytes=2304,
        static_norm_bo_bytes=266240,
        blockers=(),
    )


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
    use_full_static_norm_bo: bool,
) -> tuple[str, ...]:
    command = [
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
    ]
    if use_full_static_norm_bo:
        command.append("--use-full-static-norm-bo")
    return tuple(command)


def _result_from_completed_process(
    *,
    model_variant: str,
    probe_kind: str,
    tensor_key: str | None,
    input_distribution: str | None,
    argument_binding: Gemma3LaunchArgumentBindingEvidence | None,
    remaining_model_runner_gaps: tuple[str, ...],
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
    if argument_binding is not None:
        blockers.extend(argument_binding.blockers)
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
        argument_binding=argument_binding,
        remaining_model_runner_gaps=remaining_model_runner_gaps,
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
    argument_binding: Gemma3LaunchArgumentBindingEvidence | None,
    command: tuple[str, ...],
    m: int,
    n: int,
    herd_x: int,
    output_format: str,
    remaining_model_runner_gaps: tuple[str, ...] = (),
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
        argument_binding=argument_binding,
        remaining_model_runner_gaps=remaining_model_runner_gaps,
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
            argument_binding=None,
            command=command,
            m=m,
            n=n,
            herd_x=herd_x,
            output_format=output_format,
        )
    if probe_kind == "model-rmsnorm":
        resolved_weights = _resolve_weights_dir(model_variant, weights_dir)
        argument_binding = _first_stage_argument_binding_evidence(
            model_variant=model_variant,
            weights_dir=resolved_weights,
            m=m,
            n=n,
            tensor_key=tensor_key,
        )
        command = _build_model_worker_command(
            repo=repo,
            model_variant=model_variant,
            weights_dir=resolved_weights,
            m=m,
            n=n,
            herd_x=herd_x,
            output_format=output_format,
            tensor_key=tensor_key,
            use_full_static_norm_bo=True,
        )
        return _run_probe_command(
            model_variant=model_variant,
            probe_kind=probe_kind,
            tensor_key=tensor_key,
            input_distribution=DEFAULT_INPUT_DISTRIBUTION,
            argument_binding=argument_binding,
            remaining_model_runner_gaps=("runner-owned-bo-launch-not-wired",),
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


def _load_static_norm_payload(weights_dir: Path, model_variant: str, tensor_key: str):
    import numpy as np
    from ml_dtypes import bfloat16
    from gemma3_norm_weight_plan import build_norm_weight_plan

    plan = build_norm_weight_plan(model_variant, weights_dir=weights_dir)
    vectors = []
    tensor_offset = 0
    selected = None
    selected_offset = None
    for record in plan.records:
        vector = _load_safetensor_vector(weights_dir, record.tensor_key).astype(bfloat16).reshape(-1)
        if vector.nbytes != record.static_bo_bytes:
            raise RuntimeError(
                f"norm vector size mismatch for {record.tensor_key}: "
                f"got {vector.nbytes}, expected {record.static_bo_bytes}"
            )
        if record.tensor_key == tensor_key:
            selected = vector
            selected_offset = tensor_offset
        vectors.append(vector)
        tensor_offset += vector.nbytes
    if selected is None or selected_offset is None:
        raise RuntimeError(f"tensor key not found in norm-weight plan: {tensor_key}")
    static_norm_weights = np.concatenate(vectors).astype(bfloat16)
    if static_norm_weights.nbytes != plan.static_bo_bytes:
        raise RuntimeError(
            f"static norm payload size mismatch: got {static_norm_weights.nbytes}, "
            f"expected {plan.static_bo_bytes}"
        )
    return static_norm_weights, selected, selected_offset


def _model_rmsnorm_worker(args: argparse.Namespace) -> int:
    import numpy as np
    from ml_dtypes import bfloat16
    from air.backend.xrt_runner import XRTRunner
    from weighted_rms_norm import build_module, rms_norm_reference

    weights_dir = _resolve_weights_dir(args.model_variant, args.weights_dir)
    if args.use_full_static_norm_bo:
        norm_arg, weight, tensor_offset = _load_static_norm_payload(weights_dir, args.model_variant, args.tensor_key)
        if tensor_offset != 0:
            raise RuntimeError(
                f"weighted_rms_norm can only consume a base-addressed norm vector; "
                f"{args.tensor_key} starts at byte offset {tensor_offset}"
            )
    else:
        weight = _load_safetensor_vector(weights_dir, args.tensor_key).astype(bfloat16)
        norm_arg = weight
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
        inputs=[x_input, norm_arg],
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
        argument_binding=_fixture_argument_binding_evidence(),
        remaining_model_runner_gaps=("runner-owned-bo-launch-not-wired",),
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
    parser.add_argument("--use-full-static-norm-bo", action="store_true", help=argparse.SUPPRESS)
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
