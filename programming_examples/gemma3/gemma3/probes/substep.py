#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gemma3 real-weight decode substep launch probe.

This diagnostic proves the first multi-kernel model substep that is narrower
than a full layer: Gemma3 1B layer-0 decode RMSNorm followed by q_proj through
the FusedDQP paper kernel. It is correctness evidence for staged runner wiring,
not a TTFT/TPS or power measurement. Hardware is only touched with
--run-hardware.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from gemma3.core.artifacts import MODEL_SPECS, default_weights_dir
from gemma3.paths import EXAMPLE_ROOT, REPO_ROOT, RESULTS_DIR


DEFAULT_MODEL = "gemma3-1b"
DEFAULT_PHASE = "decode"
DEFAULT_LAYER = 0
DEFAULT_SEQUENCE_KIND = "decode-rmsnorm-qproj"
DEFAULT_OUTPUT_FORMAT = "elf"
DEFAULT_THRESHOLD = 0.99
DEFAULT_INPUT_DISTRIBUTION = "bounded-uniform-seed0"
DEFAULT_NORM_TENSOR_KEY = "model.layers.0.input_layernorm.weight"
DEFAULT_Q_PROJ_TENSOR_KEY = "model.layers.0.self_attn.q_proj.weight"
DEFAULT_SUBSTEP_PROBE_EVIDENCE = (
    RESULTS_DIR / "gemma3_1b_decode_rmsnorm_qproj_substep_probe.json"
)


@dataclass(frozen=True)
class Gemma3SubstepProbeResult:
    schema_version: int
    model_variant: str
    status: str
    sequence_kind: str
    phase: str
    layer_index: int
    stages: tuple[str, ...]
    input_shape: tuple[int, ...]
    norm_shape: tuple[int, ...]
    activation_shape: tuple[int, ...]
    q_projection_shape: tuple[int, ...]
    output_format: str
    bo_binding_mode: str
    norm_tensor_key: str
    static_norm_tensor_offset_bytes: int | None
    static_norm_bo_bytes: int | None
    q_projection_tensor_key: str
    projection_weight_layout: str
    input_distribution: str
    rms_correlation: float | None
    q_projection_correlation: float | None
    dense_q_projection_correlation: float | None
    threshold: float
    remaining_model_runner_gaps: tuple[str, ...]
    command: tuple[str, ...]
    returncode: int | None
    elapsed_seconds: float | None
    blockers: tuple[str, ...]
    git_commit: str | None
    dirty_worktree: bool | None
    stdout_tail: tuple[str, ...]
    stderr_tail: tuple[str, ...]

    def format(self) -> str:
        blockers = ",".join(self.blockers) if self.blockers else "none"
        gaps = (
            ",".join(self.remaining_model_runner_gaps)
            if self.remaining_model_runner_gaps
            else "none"
        )
        stages = "|".join(self.stages) if self.stages else "none"
        rms = "n/a" if self.rms_correlation is None else f"{self.rms_correlation:.6f}"
        q_proj = (
            "n/a"
            if self.q_projection_correlation is None
            else f"{self.q_projection_correlation:.6f}"
        )
        dense = (
            "n/a"
            if self.dense_q_projection_correlation is None
            else f"{self.dense_q_projection_correlation:.6f}"
        )
        return (
            f"substep_probe model={self.model_variant} status={self.status} "
            f"sequence={self.sequence_kind} phase={self.phase} layer=L{self.layer_index} "
            f"stages={stages} input={_shape_text(self.input_shape)} "
            f"norm={_shape_text(self.norm_shape)} activation={_shape_text(self.activation_shape)} "
            f"q_projection={_shape_text(self.q_projection_shape)} "
            f"output_format={self.output_format} bo_binding={self.bo_binding_mode} "
            f"norm_tensor={self.norm_tensor_key}@{self.static_norm_tensor_offset_bytes}/bo={self.static_norm_bo_bytes} "
            f"q_tensor={self.q_projection_tensor_key} weight_layout={self.projection_weight_layout} "
            f"input_distribution={self.input_distribution} rms_correlation={rms} "
            f"q_projection_correlation={q_proj} dense_q_projection_correlation={dense} "
            f"threshold={self.threshold:g} model_runner_gaps={gaps} blockers={blockers}"
        )

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


def _repo_root() -> Path:
    return REPO_ROOT


def _shape_text(shape: tuple[int, ...]) -> str:
    return "x".join(str(dim) for dim in shape) if shape else "scalar"


def _tail(text: str, limit: int = 40) -> tuple[str, ...]:
    return tuple(text.splitlines()[-limit:])


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
        EXAMPLE_ROOT,
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


def _activate_probe_env() -> None:
    env = _probe_env(_repo_root())
    for key in ("PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "XILINX_XRT", "PEANO_INSTALL_DIR"):
        if key in env:
            os.environ[key] = env[key]
    for entry in reversed(env.get("PYTHONPATH", "").split(os.pathsep)):
        if entry and entry not in sys.path:
            sys.path.insert(0, entry)


def _resolve_weights_dir(model_variant: str, weights_dir: Path | None) -> Path:
    return (weights_dir or default_weights_dir(model_variant)).expanduser()


def _correlation(actual, expected) -> float:
    import numpy as np
    from ml_dtypes import bfloat16

    actual_flat = actual.reshape(-1)
    expected_flat = expected.reshape(-1)
    if actual.dtype == bfloat16:
        actual_flat = actual_flat.astype(np.float64)
    if expected.dtype == bfloat16:
        expected_flat = expected_flat.astype(np.float64)
    return float(np.corrcoef(actual_flat, expected_flat)[0, 1])


def _is_decode_q_projection_substep_evidence(data: object, *, model_variant: str) -> bool:
    if not isinstance(data, dict):
        return False
    return (
        data.get("schema_version") == 1
        and data.get("model_variant") == model_variant
        and data.get("status") == "SUBSTEP_SEQUENCE_PASS"
        and data.get("sequence_kind") == DEFAULT_SEQUENCE_KIND
        and data.get("phase") == DEFAULT_PHASE
        and data.get("layer_index") == DEFAULT_LAYER
        and data.get("output_format") == DEFAULT_OUTPUT_FORMAT
        and data.get("bo_binding_mode") == "runner-owned-persistent-bo"
        and data.get("q_projection_tensor_key") == DEFAULT_Q_PROJ_TENSOR_KEY
        and not data.get("blockers")
        and "full-qkv-substep-not-wired"
        in tuple(data.get("remaining_model_runner_gaps", ()))
    )


def has_decode_q_projection_substep_evidence(
    model_variant: str,
    path: Path | None = None,
) -> bool:
    evidence_path = path or DEFAULT_SUBSTEP_PROBE_EVIDENCE
    try:
        data = json.loads(evidence_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return _is_decode_q_projection_substep_evidence(data, model_variant=model_variant)


def _load_safetensor_array(weights_dir: Path, tensor_key: str):
    try:
        from safetensors import safe_open
    except Exception as exc:
        raise RuntimeError("python:safetensors is required for Gemma3 substep probe") from exc
    for path in sorted(weights_dir.glob("*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if tensor_key in handle.keys():
                return handle.get_tensor(tensor_key).float().cpu().numpy()
    raise RuntimeError(f"tensor key not found in {weights_dir}: {tensor_key}")


def _load_static_norm_payload(weights_dir: Path, model_variant: str, tensor_key: str):
    import numpy as np
    from ml_dtypes import bfloat16
    from gemma3.npu.norm_weight_plan import build_norm_weight_plan

    plan = build_norm_weight_plan(model_variant, weights_dir=weights_dir)
    vectors = []
    tensor_offset = 0
    selected = None
    selected_offset = None
    for record in plan.records:
        vector = _load_safetensor_array(weights_dir, record.tensor_key).astype(bfloat16).reshape(-1)
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


def _q_projection_tensor_key(model_variant: str, weights_dir: Path) -> str:
    from gemma3.npu.weight_plan import build_weight_plan

    plan = build_weight_plan(model_variant, weights_dir=weights_dir)
    for record in plan.records:
        if record.layer_index == DEFAULT_LAYER and record.family == "q_proj":
            return record.tensor_key
    raise RuntimeError(f"q_proj layer {DEFAULT_LAYER} not found in {weights_dir}")


def _ceil_to(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _repack_q_proj_for_fused_dqp(weight) -> tuple[object, object, object, object]:
    import numpy as np
    from ml_dtypes import bfloat16
    from gemma3.core.common import Q4NX_COLS, Q4NX_ROWS, pack_int4_low_first

    rows, cols = weight.shape
    padded_rows = _ceil_to(int(rows), Q4NX_ROWS)
    padded_cols = _ceil_to(int(cols), Q4NX_COLS)
    row_blocks = padded_rows // Q4NX_ROWS
    col_blocks = padded_cols // Q4NX_COLS
    padded = np.zeros((padded_rows, padded_cols), dtype=np.float32)
    padded[:rows, :cols] = weight.astype(np.float32)

    packed = np.empty(
        (row_blocks, col_blocks, Q4NX_ROWS * Q4NX_COLS // 2),
        dtype=np.int8,
    )
    scale = np.empty((row_blocks, col_blocks, Q4NX_COLS), dtype=bfloat16)
    min_offset = np.empty((row_blocks, col_blocks, Q4NX_COLS), dtype=bfloat16)
    for rb in range(row_blocks):
        r0 = rb * Q4NX_ROWS
        r1 = r0 + Q4NX_ROWS
        for cb in range(col_blocks):
            c0 = cb * Q4NX_COLS
            c1 = c0 + Q4NX_COLS
            block = padded[r0:r1, c0:c1]
            mn = block.min(axis=0)
            mx = block.max(axis=0)
            sc = (mx - mn) / 15.0
            quant_scale = np.where(sc == 0.0, 1.0, sc)
            q = np.rint((block - mn[None, :]) / quant_scale[None, :])
            q = np.clip(q, 0, 15).astype(np.uint8)
            packed[rb, cb] = pack_int4_low_first(q).view(np.int8)
            scale[rb, cb] = sc.astype(bfloat16)
            min_offset[rb, cb] = mn.astype(bfloat16)
    return packed, scale, min_offset, padded


def _write_bo_arg(xrt, bo, array) -> None:
    from ml_dtypes import bfloat16

    payload = array.view("int16") if array.dtype == bfloat16 else array
    bo.write(payload, 0)
    bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)


def _run_elf_with_runner_bos(
    *,
    mlir_module,
    backend_options: dict[str, object],
    inputs: list[object],
    output_shape: tuple[int, ...],
    output_dtype,
    timed_kernel_seconds: list[float] | None = None,
    power_meter=None,
):
    import numpy as np
    from air.backend.xrt import XRTBackend
    from filelock import FileLock

    try:
        import pyxrt as xrt
    except Exception as exc:
        raise RuntimeError("python:pyxrt is required for Gemma3 substep probe") from exc
    if backend_options.get("output_format") != "elf":
        raise RuntimeError("Gemma3 substep probe currently requires ELF output")

    backend = XRTBackend(**backend_options)
    artifact = backend.compile(mlir_module)
    with FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        device = xrt.device(0)
        elf = xrt.elf(artifact.output_binary)
        context = xrt.hw_context(device, elf)
        kernel = xrt.ext.kernel(context, artifact.kernel)
        y_out = np.zeros(output_shape, dtype=output_dtype)
        arrays = [*inputs, y_out]
        sizes = [array.size * array.itemsize for array in arrays]
        bos = [xrt.ext.bo(device, size) for size in sizes]
        for bo, array in zip(bos, arrays):
            _write_bo_arg(xrt, bo, array)
        run = xrt.run(kernel)
        for index, bo in enumerate(bos):
            run.set_arg(index, bo)
        if power_meter is not None:
            power_meter.begin_segment()
        timed_start = time.perf_counter()
        run.start()
        run.wait2()
        timed_elapsed = time.perf_counter() - timed_start
        if timed_kernel_seconds is not None:
            timed_kernel_seconds.append(timed_elapsed)
        if power_meter is not None:
            power_meter.end_segment(timed_elapsed)
        bos[-1].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        actual = bos[-1].read(sizes[-1], 0).view(output_dtype).reshape(output_shape)
    backend.unload()
    return actual


def _run_hardware_sequence(args: argparse.Namespace) -> Gemma3SubstepProbeResult:
    _activate_probe_env()
    import numpy as np
    from ml_dtypes import bfloat16
    from gemma3.core.common import fused_dqp_paper_reference
    from gemma3.kernels.fused_dqp import _pack_l3_inputs, build_paper_module
    from weighted_rms_norm import build_module as build_rms_module
    from weighted_rms_norm import rms_norm_reference

    repo = _repo_root()
    git_commit, dirty = _git_info(repo)
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    blockers: list[str] = []
    start = time.perf_counter()

    try:
        weights_dir = _resolve_weights_dir(args.model_variant, args.weights_dir)
        norm_payload, norm_weight, norm_offset = _load_static_norm_payload(
            weights_dir,
            args.model_variant,
            args.norm_tensor_key,
        )
        q_tensor_key = args.q_projection_tensor_key or _q_projection_tensor_key(
            args.model_variant,
            weights_dir,
        )
        q_weight = _load_safetensor_array(weights_dir, q_tensor_key)
        if q_weight.shape != (1024, 1152):
            raise RuntimeError(f"expected q_proj shape (1024, 1152), got {q_weight.shape}")

        rng = np.random.default_rng(0)
        x_input = rng.uniform(-0.5, 0.5, size=(1, 1152)).astype(bfloat16)
        rms_expected = rms_norm_reference(x_input, norm_weight)
        rms_module = build_rms_module(1, 1152, bfloat16, 16, herd_x=1)
        rms_actual = _run_elf_with_runner_bos(
            mlir_module=rms_module,
            backend_options=dict(
                verbose=False,
                omit_while_true_loop=False,
                output_format=DEFAULT_OUTPUT_FORMAT,
                instance_name="weighted_rms_norm",
                runtime_loop_tiling_sizes=[4, 4],
            ),
            inputs=[x_input, norm_payload],
            output_shape=rms_expected.shape,
            output_dtype=bfloat16,
        )
        rms_correlation = _correlation(rms_actual, rms_expected)
        stdout_lines.append(f"RMSNorm correlation: {rms_correlation:.6f}")
        if rms_correlation < DEFAULT_THRESHOLD:
            blockers.append("decode-rmsnorm-correlation-low")

        packed, scale, min_offset, padded_weight = _repack_q_proj_for_fused_dqp(q_weight)
        activation_padded = np.zeros((5 * 256,), dtype=bfloat16)
        activation_padded[:1152] = rms_actual.reshape(-1)
        activation = activation_padded.reshape(5, 256)
        q_expected = fused_dqp_paper_reference(
            packed,
            scale,
            min_offset,
            activation,
            32,
            256,
        )
        object_file = EXAMPLE_ROOT / "build_peano" / "fused_dqp.o"
        if not object_file.exists():
            raise RuntimeError(f"missing FusedDQP object file: {object_file}")
        q_module = build_paper_module(
            32,
            256,
            "fused_dqp_accum_block_opt",
            str(object_file),
            32,
            1,
            2,
            4,
            "direct",
        )
        q_accum = np.zeros(q_expected.shape, dtype=np.float32)
        for col_block in range(5):
            cb_slice = slice(col_block, col_block + 1)
            params = np.empty((32, 1, 512), dtype=bfloat16)
            params[..., :256] = scale[:, cb_slice, :]
            params[..., 256:] = min_offset[:, cb_slice, :]
            packed_l3 = _pack_l3_inputs(packed[:, cb_slice, :], params).reshape(8, 4, 1, -1)
            q_partial = _run_elf_with_runner_bos(
                mlir_module=q_module,
                backend_options=dict(
                    verbose=False,
                    omit_pingpong=True,
                    output_format=DEFAULT_OUTPUT_FORMAT,
                    instance_name="fused_dqp_paper",
                    target_device="npu2",
                    runtime_loop_tiling_sizes=[1, 1],
                    use_lock_race_condition_fix=True,
                ),
                inputs=[packed_l3, activation[cb_slice, :]],
                output_shape=q_expected.shape,
                output_dtype=bfloat16,
            )
            q_accum += q_partial.astype(np.float32)
        q_actual = q_accum.astype(bfloat16)
        q_projection_correlation = _correlation(q_actual, q_expected)
        stdout_lines.append(f"Q projection correlation: {q_projection_correlation:.6f}")
        if q_projection_correlation < DEFAULT_THRESHOLD:
            blockers.append("decode-q-projection-correlation-low")

        dense_expected = (
            padded_weight.astype(np.float32) @ activation_padded.astype(np.float32)
        ).astype(bfloat16)
        dense_q_projection_correlation = _correlation(q_actual.reshape(-1), dense_expected)
        stdout_lines.append(
            f"Dense q projection correlation: {dense_q_projection_correlation:.6f}"
        )
        returncode = 0 if not blockers else 1
    except Exception as exc:
        blockers.append(f"decode-substep-probe-failed:{exc}")
        rms_correlation = None
        q_projection_correlation = None
        dense_q_projection_correlation = None
        norm_offset = None
        norm_payload = None
        q_tensor_key = args.q_projection_tensor_key or DEFAULT_Q_PROJ_TENSOR_KEY
        returncode = 1
        stderr_lines.append(str(exc))

    elapsed = time.perf_counter() - start
    status = "SUBSTEP_SEQUENCE_PASS" if not blockers else "SUBSTEP_SEQUENCE_BLOCKED"
    return Gemma3SubstepProbeResult(
        schema_version=1,
        model_variant=args.model_variant,
        status=status,
        sequence_kind=DEFAULT_SEQUENCE_KIND,
        phase=DEFAULT_PHASE,
        layer_index=DEFAULT_LAYER,
        stages=(
            "decode:L0:pre_attention_norm",
            "decode:L0:q_projection",
        ),
        input_shape=(1, 1152),
        norm_shape=(1, 1152),
        activation_shape=(5, 256),
        q_projection_shape=(1024,),
        output_format=DEFAULT_OUTPUT_FORMAT,
        bo_binding_mode="runner-owned-persistent-bo",
        norm_tensor_key=args.norm_tensor_key,
        static_norm_tensor_offset_bytes=norm_offset,
        static_norm_bo_bytes=None if norm_payload is None else int(norm_payload.nbytes),
        q_projection_tensor_key=q_tensor_key,
        projection_weight_layout="fused-dqp-paper-repacked-q_proj-colblock-loop",
        input_distribution=DEFAULT_INPUT_DISTRIBUTION,
        rms_correlation=rms_correlation,
        q_projection_correlation=q_projection_correlation,
        dense_q_projection_correlation=dense_q_projection_correlation,
        threshold=DEFAULT_THRESHOLD,
        remaining_model_runner_gaps=("full-qkv-substep-not-wired", "full-layer-not-wired"),
        command=tuple(sys.argv),
        returncode=returncode,
        elapsed_seconds=elapsed,
        blockers=tuple(dict.fromkeys(blockers)),
        git_commit=git_commit,
        dirty_worktree=dirty,
        stdout_tail=_tail("\n".join(stdout_lines)),
        stderr_tail=_tail("\n".join(stderr_lines)),
    )


def _self_test() -> None:
    result = Gemma3SubstepProbeResult(
        schema_version=1,
        model_variant=DEFAULT_MODEL,
        status="SUBSTEP_SEQUENCE_PASS",
        sequence_kind=DEFAULT_SEQUENCE_KIND,
        phase=DEFAULT_PHASE,
        layer_index=DEFAULT_LAYER,
        stages=("decode:L0:pre_attention_norm", "decode:L0:q_projection"),
        input_shape=(1, 1152),
        norm_shape=(1, 1152),
        activation_shape=(5, 256),
        q_projection_shape=(1024,),
        output_format=DEFAULT_OUTPUT_FORMAT,
        bo_binding_mode="runner-owned-persistent-bo",
        norm_tensor_key=DEFAULT_NORM_TENSOR_KEY,
        static_norm_tensor_offset_bytes=0,
        static_norm_bo_bytes=266240,
        q_projection_tensor_key=DEFAULT_Q_PROJ_TENSOR_KEY,
        projection_weight_layout="fused-dqp-paper-repacked-q_proj-colblock-loop",
        input_distribution=DEFAULT_INPUT_DISTRIBUTION,
        rms_correlation=0.999983,
        q_projection_correlation=0.999991,
        dense_q_projection_correlation=0.997500,
        threshold=DEFAULT_THRESHOLD,
        remaining_model_runner_gaps=("full-qkv-substep-not-wired", "full-layer-not-wired"),
        command=("python3", "gemma3.probes.substep", "--self-test"),
        returncode=0,
        elapsed_seconds=0.125,
        blockers=(),
        git_commit="fixture",
        dirty_worktree=False,
        stdout_tail=("RMSNorm correlation: 0.999983", "Q projection correlation: 0.999991"),
        stderr_tail=(),
    )
    if result.status != "SUBSTEP_SEQUENCE_PASS":
        raise AssertionError(result)
    if not _is_decode_q_projection_substep_evidence(result.to_json_dict(), model_variant=DEFAULT_MODEL):
        raise AssertionError(result.to_json_dict())
    stale = dict(result.to_json_dict())
    stale["bo_binding_mode"] = "xrt-runner-transient-bo"
    if _is_decode_q_projection_substep_evidence(stale, model_variant=DEFAULT_MODEL):
        raise AssertionError(stale)
    print(result.format())
    print("GEMMA3_SUBSTEP_PROBE_SELF_TEST: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma3 decode RMSNorm/q_proj substep probe")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--run-hardware", action="store_true")
    parser.add_argument("--model-variant", choices=sorted(MODEL_SPECS), default=DEFAULT_MODEL)
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--norm-tensor-key", default=DEFAULT_NORM_TENSOR_KEY)
    parser.add_argument("--q-projection-tensor-key")
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return 0
    if not args.run_hardware:
        raise SystemExit("pass --run-hardware to touch the NPU; --self-test is hardware-free")
    result = _run_hardware_sequence(args)
    print(result.format())
    if args.result_json:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(json.dumps(result.to_json_dict(), indent=2, sort_keys=True) + "\n")
        print(f"GEMMA3_SUBSTEP_PROBE_JSON: {args.result_json}")
    return 0 if result.status == "SUBSTEP_SEQUENCE_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
