#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gemma3 real-weight staged decode full-layer probe.

This diagnostic wires one Gemma3 1B decode layer pass with real weights. The
projection stages launch on the NPU through the FusedDQP column-block route;
RMSNorm/QK-Norm, RoPE, single-token FlowQKV attention, GeGLU, and residual
adds launch through Gemma standalone wrappers. This is full-layer staged
correctness evidence, not a full model loop, TTFT/TPS timing, pseudo-NPU power,
or a paper-parity result.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time

from gemma3.core.artifacts import MODEL_SPECS
from gemma3.paths import AIE_KERNELS_DIR, EXAMPLE_ROOT, RESULTS_DIR
from gemma3.evidence.power import MISSING_POWER_FIELD
from gemma3.probes.qkv_substep import _projection_backend_options
from gemma3.probes.substep import (
    DEFAULT_INPUT_DISTRIBUTION,
    DEFAULT_LAYER,
    DEFAULT_MODEL,
    DEFAULT_NORM_TENSOR_KEY,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_PHASE,
    DEFAULT_THRESHOLD,
    _activate_probe_env,
    _correlation,
    _git_info,
    _load_safetensor_array,
    _load_static_norm_payload,
    _repo_root,
    _resolve_weights_dir,
    _run_elf_with_runner_bos,
    _shape_text,
    _write_bo_arg,
    _tail,
)

DEFAULT_SEQUENCE_KIND = "decode-full-layer-staged"
DEFAULT_FULL_LAYER_PROBE_EVIDENCE = (
    RESULTS_DIR / "gemma3_1b_decode_full_layer_probe.json"
)
FULL_LAYER_PROJECTION_FAMILIES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
PROJECTION_SHAPES = {
    "q_proj": (1024, 1152),
    "k_proj": (256, 1152),
    "v_proj": (256, 1152),
    "o_proj": (1152, 1024),
    "gate_proj": (6912, 1152),
    "up_proj": (6912, 1152),
    "down_proj": (1152, 6912),
}
PROJECTION_OUTPUT_SHAPES = {
    "q_proj": (1024,),
    "k_proj": (256,),
    "v_proj": (256,),
    "o_proj": (1152,),
    "gate_proj": (6912,),
    "up_proj": (6912,),
    "down_proj": (1152,),
}
def _norm_tensor_keys(layer_index: int) -> dict[str, str]:
    return {
        "input_layernorm": f"model.layers.{layer_index}.input_layernorm.weight",
        "q_norm": f"model.layers.{layer_index}.self_attn.q_norm.weight",
        "k_norm": f"model.layers.{layer_index}.self_attn.k_norm.weight",
        "post_attention_layernorm": f"model.layers.{layer_index}.post_attention_layernorm.weight",
        "pre_feedforward_layernorm": f"model.layers.{layer_index}.pre_feedforward_layernorm.weight",
        "post_feedforward_layernorm": f"model.layers.{layer_index}.post_feedforward_layernorm.weight",
    }


NORM_TENSOR_KEYS = _norm_tensor_keys(DEFAULT_LAYER)


@dataclass(frozen=True)
class ProjectionEvidence:
    family: str
    tensor_key: str
    shape: tuple[int, int]
    padded_shape: tuple[int, int]
    row_blocks: int
    col_blocks: int
    projection_correlation: float | None
    dense_projection_correlation: float | None


@dataclass(frozen=True)
class Gemma3FullLayerProbeResult:
    schema_version: int
    model_variant: str
    status: str
    sequence_kind: str
    phase: str
    layer_index: int
    stages: tuple[str, ...]
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    output_format: str
    bo_binding_mode: str
    runner_reuse_mode: str
    norm_tensor_key: str
    static_norm_argument_mode: str
    static_norm_tensor_offset_bytes: int | None
    static_norm_bo_bytes: int | None
    static_norm_argument_bytes: int | None
    norm_tensor_keys: dict[str, str]
    projection_tensor_keys: dict[str, str]
    projection_weight_layout: str
    host_fallbacks: tuple[str, ...]
    input_distribution: str
    rms_correlation: float | None
    projection_evidence: tuple[ProjectionEvidence, ...]
    attention_correlation: float | None
    rope_q_correlation: float | None
    rope_k_correlation: float | None
    q_norm_correlation: float | None
    k_norm_correlation: float | None
    post_attention_norm_correlation: float | None
    pre_feedforward_norm_correlation: float | None
    post_feedforward_norm_correlation: float | None
    attention_residual_correlation: float | None
    mlp_residual_correlation: float | None
    mlp_activation_correlation: float | None
    final_output_correlation: float | None
    dense_final_output_correlation: float | None
    timed_kernel_count: int
    timed_kernel_seconds: float | None
    timed_kernel_mean_seconds: float | None
    diagnostic_layer_passes_per_second: float | None
    estimated_26_layer_decode_tps_kernel_only: float | None
    timing_window: str
    timing_notes: tuple[str, ...]
    power_snapshot: dict[str, object] | None
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

    @staticmethod
    def _corr_text(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.6f}"

    def format(self) -> str:
        blockers = ",".join(self.blockers) if self.blockers else "none"
        gaps = (
            ",".join(self.remaining_model_runner_gaps)
            if self.remaining_model_runner_gaps
            else "none"
        )
        stages = "|".join(self.stages) if self.stages else "none"
        host_fallbacks = "|".join(self.host_fallbacks) if self.host_fallbacks else "none"
        projection_corrs = "|".join(
            f"{item.family}:{self._corr_text(item.projection_correlation)}"
            for item in self.projection_evidence
        )
        dense_corrs = "|".join(
            f"{item.family}:{self._corr_text(item.dense_projection_correlation)}"
            for item in self.projection_evidence
        )
        return (
            f"full_layer_probe model={self.model_variant} status={self.status} "
            f"sequence={self.sequence_kind} phase={self.phase} layer=L{self.layer_index} "
            f"stages={stages} input={_shape_text(self.input_shape)} "
            f"output={_shape_text(self.output_shape)} output_format={self.output_format} "
            f"bo_binding={self.bo_binding_mode} runner_reuse={self.runner_reuse_mode} "
            f"norm_arg={self.static_norm_argument_mode} "
            f"norm_tensor={self.norm_tensor_key}@{self.static_norm_tensor_offset_bytes}/"
            f"bo={self.static_norm_bo_bytes}/arg={self.static_norm_argument_bytes} "
            f"weight_layout={self.projection_weight_layout} "
            f"host_fallbacks={host_fallbacks} input_distribution={self.input_distribution} "
            f"rms_correlation={self._corr_text(self.rms_correlation)} "
            f"projection_correlations={projection_corrs} "
            f"dense_projection_correlations={dense_corrs} "
            f"attention_correlation={self._corr_text(self.attention_correlation)} "
            f"norm_correlations=q:{self._corr_text(self.q_norm_correlation)}|"
            f"k:{self._corr_text(self.k_norm_correlation)}|"
            f"post_attention:{self._corr_text(self.post_attention_norm_correlation)}|"
            f"pre_ff:{self._corr_text(self.pre_feedforward_norm_correlation)}|"
            f"post_ff:{self._corr_text(self.post_feedforward_norm_correlation)} "
            f"rope_correlations=q:{self._corr_text(self.rope_q_correlation)}|"
            f"k:{self._corr_text(self.rope_k_correlation)} "
            f"residual_correlations=attention:{self._corr_text(self.attention_residual_correlation)}|"
            f"mlp:{self._corr_text(self.mlp_residual_correlation)} "
            f"mlp_activation_correlation={self._corr_text(self.mlp_activation_correlation)} "
            f"final_output_correlation={self._corr_text(self.final_output_correlation)} "
            f"dense_final_output_correlation={self._corr_text(self.dense_final_output_correlation)} "
            f"timed_kernel_count={self.timed_kernel_count} "
            f"timed_kernel_seconds={self._corr_text(self.timed_kernel_seconds)} "
            f"diagnostic_layer_passes_per_second={self._corr_text(self.diagnostic_layer_passes_per_second)} "
            f"estimated_26_layer_decode_tps_kernel_only={self._corr_text(self.estimated_26_layer_decode_tps_kernel_only)} "
            f"timing_window={self.timing_window} "
            f"threshold={self.threshold:g} model_runner_gaps={gaps} blockers={blockers}"
        )

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


def _is_decode_full_layer_evidence(data: object, *, model_variant: str) -> bool:
    if not isinstance(data, dict):
        return False
    return (
        data.get("schema_version") == 1
        and data.get("model_variant") == model_variant
        and data.get("status") == "FULL_LAYER_SEQUENCE_PASS"
        and data.get("sequence_kind") == DEFAULT_SEQUENCE_KIND
        and data.get("phase") == DEFAULT_PHASE
        and data.get("layer_index") == DEFAULT_LAYER
        and data.get("output_format") == DEFAULT_OUTPUT_FORMAT
        and data.get("bo_binding_mode") == "runner-owned-persistent-bo"
        and data.get("static_norm_argument_mode") in {
            "selected-vector",
            "contiguous-payload",
        }
        and not data.get("blockers")
        and "full-1b-loop-not-wired" in tuple(data.get("remaining_model_runner_gaps", ()))
    )


def has_decode_full_layer_evidence(
    model_variant: str,
    path: Path | None = None,
) -> bool:
    evidence_path = path or DEFAULT_FULL_LAYER_PROBE_EVIDENCE
    try:
        data = json.loads(evidence_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return _is_decode_full_layer_evidence(data, model_variant=model_variant)


def decode_full_layer_host_fallbacks(
    model_variant: str,
    path: Path | None = None,
) -> tuple[str, ...] | None:
    evidence_path = path or DEFAULT_FULL_LAYER_PROBE_EVIDENCE
    try:
        data = json.loads(evidence_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not _is_decode_full_layer_evidence(data, model_variant=model_variant):
        return None
    return tuple(str(item) for item in data.get("host_fallbacks", ()))


def has_decode_full_layer_without_host_fallback_evidence(
    model_variant: str,
    path: Path | None = None,
) -> bool:
    return decode_full_layer_host_fallbacks(model_variant, path=path) == ()


def _ceil_to(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


@dataclass(frozen=True)
class _PreparedStaticArg:
    shape: tuple[int, ...]
    dtype: object
    nbytes: int


def _prepared_static_arg(shape: tuple[int, ...], dtype, nbytes: int) -> _PreparedStaticArg:
    return _PreparedStaticArg(
        shape=tuple(int(dim) for dim in shape),
        dtype=dtype,
        nbytes=int(nbytes),
    )


def _prepared_static_arg_like(array) -> _PreparedStaticArg:
    return _prepared_static_arg(array.shape, array.dtype, int(array.size * array.itemsize))


class _ReusableElfRunner:
    def __init__(
        self,
        cache,
        *,
        mlir_module,
        backend_options: dict[str, object],
        output_shape: tuple[int, ...],
        output_dtype,
    ) -> None:
        import numpy as np
        from air.backend.xrt import XRTBackend

        self.cache = cache
        self.output_shape = tuple(output_shape)
        self.output_dtype = output_dtype
        self.backend = XRTBackend(**backend_options)
        self.artifact = self.backend.compile(mlir_module)
        self.elf = cache.xrt.elf(self.artifact.output_binary)
        self.context = cache.xrt.hw_context(cache.device, self.elf)
        self.kernel = cache.xrt.ext.kernel(self.context, self.artifact.kernel)
        self.bos = None
        self.sizes: list[int] | None = None
        self.static_keys: list[object | None] | None = None
        self.bo_sets: dict[tuple[object, ...], dict[str, object]] = {}
        self._np = np

    def _arrays_and_sizes(self, inputs: list[object]):
        y_out = self._np.zeros(self.output_shape, dtype=self.output_dtype)
        arrays = [*inputs, y_out]
        sizes = [
            array.nbytes if isinstance(array, _PreparedStaticArg) else array.size * array.itemsize
            for array in arrays
        ]
        return arrays, sizes

    def _state_for(self, *, arrays: list[object], sizes: list[int], bo_set_key: tuple[object, ...] | None):
        if bo_set_key is None:
            if self.bos is None:
                self.sizes = sizes
                self.static_keys = [None] * len(arrays)
                self.bos = [self.cache.xrt.ext.bo(self.cache.device, size) for size in sizes]
            elif self.sizes != sizes:
                raise RuntimeError(f"reused ELF runner size mismatch: expected {self.sizes}, got {sizes}")
            return self.bos, self.static_keys

        state = self.bo_sets.get(bo_set_key)
        if state is None:
            state = {
                "sizes": sizes,
                "static_keys": [None] * len(arrays),
                "bos": [self.cache.xrt.ext.bo(self.cache.device, size) for size in sizes],
            }
            self.bo_sets[bo_set_key] = state
        elif state["sizes"] != sizes:
            raise RuntimeError(
                f"reused ELF runner BO-set size mismatch for {bo_set_key}: "
                f"expected {state['sizes']}, got {sizes}"
            )
        return state["bos"], state["static_keys"]

    def _write_args(
        self,
        *,
        bos,
        arrays: list[object],
        static_keys: list[object | None],
        requested_static_keys: list[object | None],
        write_dynamic: bool,
    ) -> None:
        for index, (bo, array) in enumerate(zip(bos, arrays)):
            requested_key = requested_static_keys[index]
            if requested_key is None and not write_dynamic:
                continue
            if requested_key is not None and static_keys[index] == requested_key:
                continue
            if isinstance(array, _PreparedStaticArg):
                raise RuntimeError(
                    "prepared static placeholder reached a BO write; "
                    "preload the matching static input before timed execution"
                )
            _write_bo_arg(self.cache.xrt, bo, array)
            if requested_key is not None:
                static_keys[index] = requested_key

    def prepare(
        self,
        *,
        inputs: list[object],
        bo_set_key: tuple[object, ...],
        static_input_keys: list[object | None],
    ) -> None:
        if len(static_input_keys) != len(inputs):
            raise RuntimeError(
                f"static_input_keys length mismatch: expected {len(inputs)}, got {len(static_input_keys)}"
            )
        arrays, sizes = self._arrays_and_sizes(inputs)
        bos, static_keys = self._state_for(arrays=arrays, sizes=sizes, bo_set_key=bo_set_key)
        self._write_args(
            bos=bos,
            arrays=arrays,
            static_keys=static_keys,
            requested_static_keys=[*static_input_keys, None],
            write_dynamic=False,
        )

    def run(
        self,
        *,
        inputs: list[object],
        timed_kernel_seconds: list[float] | None,
        power_meter,
        bo_set_key: tuple[object, ...] | None = None,
        static_input_keys: list[object | None] | None = None,
    ):
        arrays, sizes = self._arrays_and_sizes(inputs)
        bos, static_keys = self._state_for(arrays=arrays, sizes=sizes, bo_set_key=bo_set_key)
        if static_input_keys is None:
            static_input_keys = [None] * len(inputs)
        if len(static_input_keys) != len(inputs):
            raise RuntimeError(
                f"static_input_keys length mismatch: expected {len(inputs)}, got {len(static_input_keys)}"
            )
        self._write_args(
            bos=bos,
            arrays=arrays,
            static_keys=static_keys,
            requested_static_keys=[*static_input_keys, None],
            write_dynamic=True,
        )
        run = self.cache.xrt.run(self.kernel)
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
        bos[-1].sync(self.cache.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        return bos[-1].read(sizes[-1], 0).view(self.output_dtype).reshape(self.output_shape)

    def close(self) -> None:
        self.bo_sets.clear()
        self.backend.unload()


class _ReusableElfRunnerCache:
    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self.runners: dict[tuple[object, ...], _ReusableElfRunner] = {}
        self.lock = None
        self.xrt = None
        self.device = None

    def __enter__(self):
        if not self.enabled:
            return self
        import os
        import tempfile
        from filelock import FileLock

        try:
            import pyxrt as xrt
        except Exception as exc:
            raise RuntimeError("python:pyxrt is required for Gemma3 reusable ELF runner") from exc
        self.xrt = xrt
        self.lock = FileLock(os.path.join(tempfile.gettempdir(), "npu.lock"))
        self.lock.acquire()
        self.device = xrt.device(0)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for runner in self.runners.values():
            runner.close()
        self.runners.clear()
        if self.lock is not None:
            self.lock.release()

    def _runner(
        self,
        *,
        key: tuple[object, ...],
        mlir_module,
        backend_options: dict[str, object],
        output_shape: tuple[int, ...],
        output_dtype,
    ) -> _ReusableElfRunner:
        runner_key = (*key, tuple(output_shape), str(output_dtype))
        runner = self.runners.get(runner_key)
        if runner is None:
            runner = _ReusableElfRunner(
                self,
                mlir_module=mlir_module,
                backend_options=backend_options,
                output_shape=output_shape,
                output_dtype=output_dtype,
            )
            self.runners[runner_key] = runner
        return runner

    def prepare(
        self,
        *,
        key: tuple[object, ...],
        mlir_module,
        backend_options: dict[str, object],
        inputs: list[object],
        output_shape: tuple[int, ...],
        output_dtype,
        bo_set_key: tuple[object, ...],
        static_input_keys: list[object | None],
    ) -> None:
        if not self.enabled:
            return
        runner = self._runner(
            key=key,
            mlir_module=mlir_module,
            backend_options=backend_options,
            output_shape=output_shape,
            output_dtype=output_dtype,
        )
        runner.prepare(
            inputs=inputs,
            bo_set_key=bo_set_key,
            static_input_keys=static_input_keys,
        )

    def run(
        self,
        *,
        key: tuple[object, ...],
        mlir_module,
        backend_options: dict[str, object],
        inputs: list[object],
        output_shape: tuple[int, ...],
        output_dtype,
        timed_kernel_seconds: list[float] | None = None,
        power_meter=None,
        bo_set_key: tuple[object, ...] | None = None,
        static_input_keys: list[object | None] | None = None,
    ):
        if not self.enabled:
            return _run_elf_with_runner_bos(
                mlir_module=mlir_module,
                backend_options=backend_options,
                inputs=inputs,
                output_shape=output_shape,
                output_dtype=output_dtype,
                timed_kernel_seconds=timed_kernel_seconds,
                power_meter=power_meter,
            )
        runner = self._runner(
            key=key,
            mlir_module=mlir_module,
            backend_options=backend_options,
            output_shape=output_shape,
            output_dtype=output_dtype,
        )
        return runner.run(
            inputs=inputs,
            timed_kernel_seconds=timed_kernel_seconds,
            power_meter=power_meter,
            bo_set_key=bo_set_key,
            static_input_keys=static_input_keys,
        )


class _SegmentedRAPLPowerMeter:
    def __init__(self, *, sample: bool, run_id: str | None) -> None:
        self.sample = sample
        self.run_id = run_id
        self.notes: list[str] = []
        self.baseline_pkg_watts: float | None = None
        self.energy_delta_uj = 0
        self.elapsed_seconds = 0.0
        self.max_range_uj: int | None = None
        self._segment_start_uj: int | None = None
        self._segment_max_range_uj: int | None = None
        self.sampling_backend = "not-requested"
        if not sample:
            self.notes.append("power sampling was not requested")
            return
        self.sampling_backend = "rapl-sysfs-segmented"
        try:
            from gemma3.evidence.power import _sample_rapl_pkg_watts

            baseline, note, backend = _sample_rapl_pkg_watts()
        except Exception as exc:
            baseline, note, backend = None, str(exc), "rapl-sysfs"
        if backend:
            self.sampling_backend = backend + "+segmented"
        if baseline is None:
            self.notes.append("quiescent direct RAPL package power sample unavailable: " + (note or "unknown"))
        else:
            self.baseline_pkg_watts = float(baseline)
            self.notes.append(f"quiescent_pkg_watts={self.baseline_pkg_watts:.6f}")

    def begin_segment(self) -> None:
        if not self.sample:
            return
        try:
            from gemma3.evidence.power import _read_rapl_package_energy

            start, max_range, error = _read_rapl_package_energy()
        except Exception as exc:
            start, max_range, error = None, None, str(exc)
        if error or start is None:
            self.notes.append("timed direct RAPL package segment start unavailable: " + (error or "missing start reading"))
            self._segment_start_uj = None
            self._segment_max_range_uj = None
            return
        self._segment_start_uj = int(start)
        self._segment_max_range_uj = max_range
        if max_range is not None:
            self.max_range_uj = max_range

    def end_segment(self, elapsed_seconds: float) -> None:
        if not self.sample or self._segment_start_uj is None:
            return
        try:
            from gemma3.evidence.power import _read_rapl_package_energy

            end, _max_range, error = _read_rapl_package_energy()
        except Exception as exc:
            end, error = None, str(exc)
        if error or end is None:
            self.notes.append("timed direct RAPL package segment end unavailable: " + (error or "missing end reading"))
            return
        delta = int(end) - self._segment_start_uj
        max_range = self._segment_max_range_uj or self.max_range_uj
        if delta < 0 and max_range:
            delta += int(max_range)
        if delta < 0:
            self.notes.append("invalid negative RAPL package energy delta in timed segment")
            return
        self.energy_delta_uj += delta
        self.elapsed_seconds += max(float(elapsed_seconds), 0.0)

    def snapshot(self) -> dict[str, object] | None:
        if not self.sample:
            return None
        watts: dict[str, float | None] = {rail: None for rail in ("cpu", "gpu", "npu", "total")}
        status = {rail: MISSING_POWER_FIELD for rail in ("cpu", "gpu", "npu", "total")}
        notes = list(dict.fromkeys(self.notes))
        aligned = False
        if self.elapsed_seconds > 0.0 and self.energy_delta_uj > 0:
            pkg_avg = (self.energy_delta_uj / 1_000_000.0) / self.elapsed_seconds
            watts["total"] = pkg_avg
            status["total"] = "RAPL_SYSFS_PACKAGE_SEGMENTED"
            aligned = True
            notes.append("package watts use direct RAPL sysfs energy_uj deltas summed over NPU run.start/wait2 segments")
            if self.baseline_pkg_watts is not None:
                watts["npu"] = max(pkg_avg - self.baseline_pkg_watts, 0.0)
                status["npu"] = "PSEUDO_RAPL_SYSFS_DELTA_SEGMENTED"
                notes.append("NPU rail is pseudo power: segmented package watts minus quiescent package watts before the run")
            else:
                notes.append("pseudo-NPU power requires a quiescent package-watt sample before the run")
        else:
            notes.append("no readable segmented RAPL package energy was captured")
        return {
            "schema_version": 1,
            "sampling_backend": self.sampling_backend,
            "aligned_with_timed_window": aligned,
            "watts": watts,
            "field_status": status,
            "run_id": self.run_id,
            "notes": notes,
            "segmented_elapsed_seconds": self.elapsed_seconds,
            "segmented_energy_delta_uj": self.energy_delta_uj,
        }


def _projection_tensor_keys(model_variant: str, weights_dir: Path, layer_index: int) -> dict[str, str]:
    from gemma3.npu.weight_plan import build_weight_plan

    plan = build_weight_plan(model_variant, weights_dir=weights_dir)
    keys = {}
    for record in plan.records:
        if record.layer_index == layer_index and record.family in FULL_LAYER_PROJECTION_FAMILIES:
            keys[record.family] = record.tensor_key
    missing = [family for family in FULL_LAYER_PROJECTION_FAMILIES if family not in keys]
    if missing:
        raise RuntimeError(f"missing layer-{layer_index} projection tensors: {missing}")
    return keys


def _repack_matrix_for_fused_dqp(weight, *, row_block_multiple: int = 8):
    import numpy as np
    from ml_dtypes import bfloat16
    from gemma3.core.common import Q4NX_COLS, Q4NX_ROWS, pack_int4_low_first

    rows, cols = weight.shape
    padded_rows = _ceil_to(_ceil_to(int(rows), Q4NX_ROWS) // Q4NX_ROWS, row_block_multiple) * Q4NX_ROWS
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


def _run_projection(
    *,
    family: str,
    weight,
    activation,
    object_file: Path,
    runner_cache: _ReusableElfRunnerCache,
    timed_kernel_seconds: list[float] | None = None,
    power_meter=None,
):
    import numpy as np
    from ml_dtypes import bfloat16
    from gemma3.core.common import fused_dqp_paper_reference
    from gemma3.kernels.fused_dqp import _pack_l3_inputs, build_paper_module

    expected_shape = PROJECTION_SHAPES[family]
    if tuple(weight.shape) != expected_shape:
        raise RuntimeError(f"expected {family} shape {expected_shape}, got {weight.shape}")
    out_dim, in_dim = expected_shape
    packed, scale, min_offset, padded_weight = _repack_matrix_for_fused_dqp(weight)
    row_blocks, col_blocks = packed.shape[:2]
    activation_padded = np.zeros((col_blocks * 256,), dtype=bfloat16)
    activation_padded[:in_dim] = activation.reshape(-1).astype(bfloat16)
    activation_blocks = activation_padded.reshape(col_blocks, 256)
    module = build_paper_module(
        32,
        256,
        "fused_dqp_accum_block_opt",
        str(object_file),
        row_blocks,
        1,
        2,
        4,
        "direct",
    )
    expected = fused_dqp_paper_reference(
        packed,
        scale,
        min_offset,
        activation_blocks,
        32,
        256,
    )
    accum = np.zeros(expected.shape, dtype=np.float32)
    for col_block in range(col_blocks):
        cb_slice = slice(col_block, col_block + 1)
        params = np.empty((row_blocks, 1, 512), dtype=bfloat16)
        params[..., :256] = scale[:, cb_slice, :]
        params[..., 256:] = min_offset[:, cb_slice, :]
        packed_l3 = _pack_l3_inputs(packed[:, cb_slice, :], params).reshape(
            row_blocks // 4,
            4,
            1,
            -1,
        )
        partial = runner_cache.run(
            key=("fused_dqp_accum_block_opt", int(row_blocks)),
            mlir_module=module,
            backend_options=_projection_backend_options(),
            inputs=[packed_l3, activation_blocks[cb_slice, :]],
            output_shape=expected.shape,
            output_dtype=bfloat16,
            timed_kernel_seconds=timed_kernel_seconds,
            power_meter=power_meter,
        )
        accum += partial.astype(np.float32)
    actual_padded = accum.astype(bfloat16).reshape(-1)
    expected_padded = expected.reshape(-1)
    actual = actual_padded[:out_dim].astype(bfloat16)
    expected_vec = expected_padded[:out_dim].astype(bfloat16)
    dense_expected = (weight.astype(np.float32) @ activation.reshape(-1).astype(np.float32)).astype(bfloat16)
    projection_corr = _correlation(actual, expected_vec)
    dense_corr = _correlation(actual, dense_expected)
    evidence = ProjectionEvidence(
        family=family,
        tensor_key="",
        shape=expected_shape,
        padded_shape=tuple(int(dim) for dim in padded_weight.shape),
        row_blocks=int(row_blocks),
        col_blocks=int(col_blocks),
        projection_correlation=projection_corr,
        dense_projection_correlation=dense_corr,
    )
    return actual, expected_vec, dense_expected, evidence


def _rms_host(x, weight, eps: float = 1e-5):
    import numpy as np
    from ml_dtypes import bfloat16

    xf = x.astype(np.float32)
    wf = weight.astype(np.float32)
    rms = np.sqrt(np.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return ((xf / rms) * wf).astype(bfloat16)


def _gelu_tanh(x):
    import numpy as np
    from ml_dtypes import bfloat16

    xf = x.astype(np.float32)
    inner = 0.7978845608 * (xf + 0.044715 * xf * xf * xf)
    return (0.5 * xf * (1.0 + np.tanh(inner))).astype(bfloat16)


def _geglu(gate, up):
    from ml_dtypes import bfloat16

    return (_gelu_tanh(gate).astype("float32") * up.astype("float32")).astype(bfloat16)


def _rope_module_helpers():
    from gemma3.kernels.rope_halfsplit import build_module, compile_rope_kernel, rope_halfsplit_reference

    return build_module, compile_rope_kernel, rope_halfsplit_reference


def _dataflow_dir() -> Path:
    return AIE_KERNELS_DIR


def _aie_api_include() -> Path:
    candidates = []
    mlir_aie = os.environ.get("MLIR_AIE_INSTALL_DIR")
    if mlir_aie:
        base = Path(mlir_aie)
        candidates.extend([
            base / "include",
            base / "lib/python3.12/site-packages/mlir_aie/include",
        ])
    candidates.append(_repo_root() / "sandbox/lib/python3.12/site-packages/mlir_aie/include")
    for candidate in candidates:
        if (candidate / "aie_api").is_dir():
            return candidate
    raise RuntimeError("could not locate aie_api include directory")


def _compile_flowqkv_single_token_kernel(object_file: Path) -> None:
    peano = os.environ.get("PEANO_INSTALL_DIR")
    if not peano:
        raise RuntimeError("PEANO_INSTALL_DIR is required to compile flow_attention.cc")
    clangxx = Path(peano) / "bin/clang++"
    if not clangxx.exists():
        raise RuntimeError(f"missing Peano clang++: {clangxx}")
    src = _dataflow_dir() / "flow_attention.cc"
    object_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(clangxx),
        "-O2",
        "-std=c++20",
        "--target=aie2p-none-unknown-elf",
        "-Wno-parentheses",
        "-Wno-attributes",
        "-Wno-macro-redefined",
        "-Wno-empty-body",
        "-DNDEBUG",
        "-I",
        str(_aie_api_include()),
        "-DQ_CHUNK=4",
        "-DKV_LEN=1",
        "-DHEAD_DIM=256",
        "-DQUERY_BASE=0",
        "-DWINDOW_LEN=0",
        "-DCAUSAL=1",
        "-c",
        str(src),
        "-o",
        str(object_file),
    ]
    subprocess.run(cmd, check=True)


def _flowqkv_module_helpers():
    from gemma3.kernels.flow_common import build_flow_module

    return build_flow_module


def _identity_rope_lut(rows: int, head_dim: int, dtype):
    import numpy as np

    half = head_dim // 2
    row = np.concatenate(
        [
            np.ones(half, dtype=np.float32),
            np.zeros(half, dtype=np.float32),
        ]
    )
    return np.tile(row, (rows, 1)).astype(dtype)


def _run_rms_stage(
    *,
    name: str,
    x,
    weight,
    runner_cache: _ReusableElfRunnerCache,
    timed_kernel_seconds: list[float] | None = None,
    power_meter=None,
):
    from ml_dtypes import bfloat16
    from weighted_rms_norm import build_module as build_rms_module
    from weighted_rms_norm import rms_norm_reference

    x_2d = x.reshape((-1, int(weight.size))).astype(bfloat16)
    rows, width = (int(dim) for dim in x_2d.shape)
    module = build_rms_module(rows, width, bfloat16, 16, herd_x=1)
    actual = runner_cache.run(
        key=("weighted_rms_norm", name, rows, width),
        mlir_module=module,
        backend_options=dict(
            verbose=False,
            omit_while_true_loop=False,
            output_format=DEFAULT_OUTPUT_FORMAT,
            instance_name="weighted_rms_norm",
            runtime_loop_tiling_sizes=[4, 4],
        ),
        inputs=[x_2d, weight.reshape(-1).astype(bfloat16)],
        output_shape=x_2d.shape,
        output_dtype=bfloat16,
        timed_kernel_seconds=timed_kernel_seconds,
        power_meter=power_meter,
    )
    expected = rms_norm_reference(x_2d, weight.reshape(-1).astype(bfloat16))
    return actual.reshape(x.shape).astype(bfloat16), expected.reshape(x.shape).astype(bfloat16)


def _run_rope_stage(
    *,
    name: str,
    x,
    object_file: Path,
    runner_cache: _ReusableElfRunnerCache,
    timed_kernel_seconds: list[float] | None = None,
    power_meter=None,
):
    from ml_dtypes import bfloat16

    build_module, compile_rope_kernel, rope_halfsplit_reference = _rope_module_helpers()
    rows, head_dim = (int(dim) for dim in x.shape)
    if not object_file.exists():
        compile_rope_kernel(object_file)
    herd_x = 4 if rows % 4 == 0 else 1
    lut = _identity_rope_lut(rows, head_dim, bfloat16)
    module = build_module(rows, head_dim, bfloat16, herd_x, str(object_file))
    actual = runner_cache.run(
        key=("rope_halfsplit", name, rows, head_dim, herd_x),
        mlir_module=module,
        backend_options=dict(
            verbose=False,
            omit_while_true_loop=False,
            output_format=DEFAULT_OUTPUT_FORMAT,
            instance_name="gemma3_rope_halfsplit",
            runtime_loop_tiling_sizes=[4, 4],
        ),
        inputs=[x.astype(bfloat16), lut.reshape(-1)],
        output_shape=tuple(x.shape),
        output_dtype=bfloat16,
        timed_kernel_seconds=timed_kernel_seconds,
        power_meter=power_meter,
    )
    expected = rope_halfsplit_reference(x, lut)
    return actual.astype(bfloat16), expected.astype(bfloat16)


def _residual_module_helpers():
    from gemma3.kernels.residual_add import build_module, residual_add_reference

    return build_module, residual_add_reference


def _run_residual_stage(
    *,
    name: str,
    lhs,
    rhs,
    runner_cache: _ReusableElfRunnerCache,
    timed_kernel_seconds: list[float] | None = None,
    power_meter=None,
):
    from ml_dtypes import bfloat16

    build_module, residual_add_reference = _residual_module_helpers()
    lhs_flat = lhs.reshape(-1).astype(bfloat16)
    rhs_flat = rhs.reshape(-1).astype(bfloat16)
    n = int(lhs_flat.size)
    tile_n = 288 if n % (288 * 2) == 0 else max(16, n // 2)
    module = build_module(n, tile_n, bfloat16, 16)
    actual = runner_cache.run(
        key=("residual_add", name, n, tile_n),
        mlir_module=module,
        backend_options=dict(
            verbose=False,
            omit_while_true_loop=False,
            output_format=DEFAULT_OUTPUT_FORMAT,
            instance_name="gemma3_residual_add",
            runtime_loop_tiling_sizes=[4, 4],
        ),
        inputs=[lhs_flat, rhs_flat],
        output_shape=(n,),
        output_dtype=bfloat16,
        timed_kernel_seconds=timed_kernel_seconds,
        power_meter=power_meter,
    )
    expected = residual_add_reference(lhs_flat, rhs_flat)
    return actual.reshape(lhs.shape).astype(bfloat16), expected.reshape(lhs.shape).astype(bfloat16)


def _geglu_module_helpers():
    from gemma3.kernels.geglu import build_module, geglu_reference

    return build_module, geglu_reference


def _run_geglu_stage(
    *,
    name: str,
    gate,
    up,
    runner_cache: _ReusableElfRunnerCache,
    timed_kernel_seconds: list[float] | None = None,
    power_meter=None,
):
    from ml_dtypes import bfloat16

    build_module, geglu_reference = _geglu_module_helpers()
    gate_flat = gate.reshape(-1).astype(bfloat16)
    up_flat = up.reshape(-1).astype(bfloat16)
    n = int(gate_flat.size)
    tile_n = 288 if n % (288 * 2) == 0 else max(16, n // 2)
    module = build_module(n, tile_n, bfloat16, 16)
    actual = runner_cache.run(
        key=("geglu", name, n, tile_n),
        mlir_module=module,
        backend_options=dict(
            verbose=False,
            omit_while_true_loop=False,
            output_format=DEFAULT_OUTPUT_FORMAT,
            instance_name="gemma3_geglu",
            runtime_loop_tiling_sizes=[4, 4],
        ),
        inputs=[gate_flat, up_flat],
        output_shape=(n,),
        output_dtype=bfloat16,
        timed_kernel_seconds=timed_kernel_seconds,
        power_meter=power_meter,
    )
    expected = geglu_reference(gate_flat, up_flat)
    return actual.reshape(gate.shape).astype(bfloat16), expected.reshape(gate.shape).astype(bfloat16)


def _run_single_token_attention_stage(
    *,
    q,
    k,
    v,
    object_file: Path,
    runner_cache: _ReusableElfRunnerCache,
    timed_kernel_seconds: list[float] | None = None,
    power_meter=None,
):
    import numpy as np
    from ml_dtypes import bfloat16

    build_flow_module = _flowqkv_module_helpers()
    if not object_file.exists():
        _compile_flowqkv_single_token_kernel(object_file)
    q_in = q.reshape(1, 4, 256).astype(bfloat16)
    k_in = k.reshape(1, 1, 256).astype(bfloat16)
    v_in = v.reshape(1, 1, 256).astype(bfloat16)
    module = build_flow_module(
        4,
        1,
        256,
        "flowqkv_chunk_bf16",
        str(object_file),
        "flowqkv_single_token",
        1,
        1,
        1,
    )
    actual = runner_cache.run(
        key=("flowqkv_single_token", 4, 1, 256),
        mlir_module=module,
        backend_options=dict(
            verbose=False,
            omit_pingpong=True,
            output_format=DEFAULT_OUTPUT_FORMAT,
            instance_name="flow_attention",
            target_device="npu2",
            runtime_loop_tiling_sizes=[1, 1],
        ),
        inputs=[q_in, k_in, v_in],
        output_shape=(1, 4, 256),
        output_dtype=bfloat16,
        timed_kernel_seconds=timed_kernel_seconds,
        power_meter=power_meter,
    )
    expected = np.tile(v.reshape(1, 256), (4, 1)).reshape(1, 4, 256).astype(bfloat16)
    return actual.reshape(1024).astype(bfloat16), expected.reshape(1024).astype(bfloat16)


def _run_hardware_sequence(args: argparse.Namespace) -> Gemma3FullLayerProbeResult:
    _activate_probe_env()
    import numpy as np
    from ml_dtypes import bfloat16
    from weighted_rms_norm import build_module as build_rms_module
    from weighted_rms_norm import rms_norm_reference

    repo = _repo_root()
    git_commit, dirty = _git_info(repo)
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    blockers: list[str] = []
    projection_keys: dict[str, str] = {}
    layer_index = int(getattr(args, "layer_index", DEFAULT_LAYER))
    norm_tensor_keys = _norm_tensor_keys(layer_index)
    projection_evidence: list[ProjectionEvidence] = []
    timed_kernel_samples: list[float] = []
    runner_reuse_enabled = not bool(getattr(args, "no_reuse_elf", False))
    runner_cache: _ReusableElfRunnerCache | None = None
    power_meter = _SegmentedRAPLPowerMeter(
        sample=bool(getattr(args, "power_sample", False)),
        run_id="gemma3_1b_decode_full_layer_probe",
    )
    start = time.perf_counter()

    try:
        weights_dir = _resolve_weights_dir(args.model_variant, args.weights_dir)
        norm_tensor_key = args.norm_tensor_key or norm_tensor_keys["input_layernorm"]
        norm_payload, input_norm_weight, norm_offset = _load_static_norm_payload(
            weights_dir,
            args.model_variant,
            norm_tensor_key,
        )
        norm_argument_mode = args.norm_argument_mode
        if norm_argument_mode == "auto":
            norm_argument_mode = "selected-vector"
        if norm_argument_mode == "selected-vector":
            norm_argument = input_norm_weight
        elif norm_argument_mode == "contiguous-payload":
            if norm_offset != 0:
                raise RuntimeError(
                    "contiguous norm payload has no offset ABI for this RMSNorm probe: "
                    f"{norm_tensor_key} starts at byte offset {norm_offset}; "
                    "use selected-vector or add explicit offset/sub-BO plumbing"
                )
            norm_argument = norm_payload
        else:
            raise RuntimeError(f"unsupported norm argument mode: {norm_argument_mode}")
        norm_weights = {
            name: _load_safetensor_array(weights_dir, key).astype(bfloat16).reshape(-1)
            for name, key in norm_tensor_keys.items()
        }
        projection_keys = _projection_tensor_keys(args.model_variant, weights_dir, layer_index)
        object_file = EXAMPLE_ROOT / "build_peano" / "fused_dqp.o"
        if not object_file.exists():
            raise RuntimeError(f"missing FusedDQP object file: {object_file}")
        rope_object_file = EXAMPLE_ROOT / "build_peano" / "rope_halfsplit.o"
        flowqkv_object_file = EXAMPLE_ROOT / "build_peano" / "flowqkv_single_token_q4_kv1_d256.o"

        rng = np.random.default_rng(0)
        x_input = rng.uniform(-0.5, 0.5, size=(1, 1152)).astype(bfloat16)
        norm_expected = rms_norm_reference(x_input, input_norm_weight)
        rms_module = build_rms_module(1, 1152, bfloat16, 16, herd_x=1)
        runner_cache = _ReusableElfRunnerCache(enabled=runner_reuse_enabled)
        runner_cache.__enter__()
        norm_actual = runner_cache.run(
            key=("weighted_rms_norm", 1, 1152),
            mlir_module=rms_module,
            backend_options=dict(
                verbose=False,
                omit_while_true_loop=False,
                output_format=DEFAULT_OUTPUT_FORMAT,
                instance_name="weighted_rms_norm",
                runtime_loop_tiling_sizes=[4, 4],
            ),
            inputs=[x_input, norm_argument],
            output_shape=norm_expected.shape,
            output_dtype=bfloat16,
            timed_kernel_seconds=timed_kernel_samples,
            power_meter=power_meter,
        )
        rms_correlation = _correlation(norm_actual, norm_expected)
        stdout_lines.append(f"RMSNorm correlation: {rms_correlation:.6f}")
        if rms_correlation < DEFAULT_THRESHOLD:
            blockers.append("decode-rmsnorm-correlation-low")

        actual: dict[str, object] = {"input_norm": norm_actual.reshape(-1)}
        expected: dict[str, object] = {"input_norm": norm_expected.reshape(-1)}
        dense: dict[str, object] = {}
        for family in ("q_proj", "k_proj", "v_proj"):
            weight = _load_safetensor_array(weights_dir, projection_keys[family])
            actual_vec, expected_vec, dense_vec, evidence = _run_projection(
                family=family,
                weight=weight,
                activation=actual["input_norm"],
                object_file=object_file,
                runner_cache=runner_cache,
                timed_kernel_seconds=timed_kernel_samples,
                power_meter=power_meter,
            )
            projection_evidence.append(
                ProjectionEvidence(
                    family=evidence.family,
                    tensor_key=projection_keys[family],
                    shape=evidence.shape,
                    padded_shape=evidence.padded_shape,
                    row_blocks=evidence.row_blocks,
                    col_blocks=evidence.col_blocks,
                    projection_correlation=evidence.projection_correlation,
                    dense_projection_correlation=evidence.dense_projection_correlation,
                )
            )
            actual[family] = actual_vec
            expected[family] = expected_vec
            dense[family] = dense_vec
            stdout_lines.append(f"{family} correlation: {evidence.projection_correlation:.6f}")
            if evidence.projection_correlation is None or evidence.projection_correlation < DEFAULT_THRESHOLD:
                blockers.append(f"decode-{family}-correlation-low")

        q_actual = actual["q_proj"].reshape(4, 256)
        k_actual = actual["k_proj"].reshape(1, 256)
        v_actual = actual["v_proj"].reshape(1, 256)
        q_expected = expected["q_proj"].reshape(4, 256)
        k_expected = expected["k_proj"].reshape(1, 256)
        v_expected = expected["v_proj"].reshape(1, 256)

        # current_pos=0 makes half-split RoPE an identity, but launch the
        # validated Gemma RoPE wrapper here to prove model-stage integration.
        qn_actual, qn_reference = _run_rms_stage(
            name="q_norm",
            x=q_actual,
            weight=norm_weights["q_norm"],
            runner_cache=runner_cache,
            timed_kernel_seconds=timed_kernel_samples,
            power_meter=power_meter,
        )
        kn_actual, kn_reference = _run_rms_stage(
            name="k_norm",
            x=k_actual,
            weight=norm_weights["k_norm"],
            runner_cache=runner_cache,
            timed_kernel_seconds=timed_kernel_samples,
            power_meter=power_meter,
        )
        qn_expected = _rms_host(q_expected, norm_weights["q_norm"])
        kn_expected = _rms_host(k_expected, norm_weights["k_norm"])
        q_norm_correlation = _correlation(qn_actual, qn_expected)
        k_norm_correlation = _correlation(kn_actual, kn_expected)
        if _correlation(qn_actual, qn_reference) < DEFAULT_THRESHOLD:
            blockers.append("decode-q-norm-correlation-low")
        if _correlation(kn_actual, kn_reference) < DEFAULT_THRESHOLD:
            blockers.append("decode-k-norm-correlation-low")
        q_rope_actual, q_rope_expected = _run_rope_stage(
            name="q",
            x=qn_actual,
            object_file=rope_object_file,
            runner_cache=runner_cache,
            timed_kernel_seconds=timed_kernel_samples,
            power_meter=power_meter,
        )
        k_rope_actual, k_rope_expected = _run_rope_stage(
            name="k",
            x=kn_actual,
            object_file=rope_object_file,
            runner_cache=runner_cache,
            timed_kernel_seconds=timed_kernel_samples,
            power_meter=power_meter,
        )
        rope_q_correlation = _correlation(q_rope_actual, q_rope_expected)
        rope_k_correlation = _correlation(k_rope_actual, k_rope_expected)
        stdout_lines.append(f"rope q correlation: {rope_q_correlation:.6f}")
        stdout_lines.append(f"rope k correlation: {rope_k_correlation:.6f}")
        if rope_q_correlation < DEFAULT_THRESHOLD:
            blockers.append("decode-rope-q-correlation-low")
        if rope_k_correlation < DEFAULT_THRESHOLD:
            blockers.append("decode-rope-k-correlation-low")
        attention_actual, attention_reference = _run_single_token_attention_stage(
            q=q_rope_actual,
            k=k_rope_actual,
            v=v_actual,
            object_file=flowqkv_object_file,
            runner_cache=runner_cache,
            timed_kernel_seconds=timed_kernel_samples,
            power_meter=power_meter,
        )
        attention_expected = np.tile(v_expected.reshape(1, 256), (4, 1)).reshape(1024).astype(bfloat16)
        attention_correlation = _correlation(attention_actual, attention_expected)
        if _correlation(attention_actual, attention_reference) < DEFAULT_THRESHOLD:
            blockers.append("decode-single-token-attention-correlation-low")

        o_weight = _load_safetensor_array(weights_dir, projection_keys["o_proj"])
        o_actual, o_expected, o_dense, evidence = _run_projection(
            family="o_proj",
            weight=o_weight,
            activation=attention_actual,
            object_file=object_file,
            runner_cache=runner_cache,
            timed_kernel_seconds=timed_kernel_samples,
            power_meter=power_meter,
        )
        projection_evidence.append(
            ProjectionEvidence(evidence.family, projection_keys["o_proj"], evidence.shape, evidence.padded_shape, evidence.row_blocks, evidence.col_blocks, evidence.projection_correlation, evidence.dense_projection_correlation)
        )
        stdout_lines.append(f"o_proj correlation: {evidence.projection_correlation:.6f}")
        if evidence.projection_correlation is None or evidence.projection_correlation < DEFAULT_THRESHOLD:
            blockers.append("decode-o_proj-correlation-low")

        post_attention_actual, post_attention_reference = _run_rms_stage(
            name="post_attention_norm",
            x=o_actual.reshape(1, 1152),
            weight=norm_weights["post_attention_layernorm"],
            runner_cache=runner_cache,
            timed_kernel_seconds=timed_kernel_samples,
            power_meter=power_meter,
        )
        post_attention_expected = _rms_host(o_expected.reshape(1, 1152), norm_weights["post_attention_layernorm"])
        post_attention_norm_correlation = _correlation(post_attention_actual, post_attention_expected)
        if _correlation(post_attention_actual, post_attention_reference) < DEFAULT_THRESHOLD:
            blockers.append("decode-post-attention-norm-correlation-low")
        residual_actual, attention_residual_reference = _run_residual_stage(
            name="attention",
            lhs=x_input.reshape(-1),
            rhs=post_attention_actual.reshape(-1),
            runner_cache=runner_cache,
            timed_kernel_seconds=timed_kernel_samples,
            power_meter=power_meter,
        )
        residual_actual = residual_actual.reshape(1, 1152)
        residual_expected = (x_input.astype(np.float32) + post_attention_expected.astype(np.float32)).astype(bfloat16)
        attention_residual_correlation = _correlation(residual_actual, residual_expected)
        if _correlation(residual_actual, attention_residual_reference.reshape(1, 1152)) < DEFAULT_THRESHOLD:
            blockers.append("decode-attention-residual-add-correlation-low")

        pre_ff_actual, pre_ff_reference = _run_rms_stage(
            name="pre_feedforward_norm",
            x=residual_actual,
            weight=norm_weights["pre_feedforward_layernorm"],
            runner_cache=runner_cache,
            timed_kernel_seconds=timed_kernel_samples,
            power_meter=power_meter,
        )
        pre_ff_expected = _rms_host(residual_expected, norm_weights["pre_feedforward_layernorm"])
        pre_feedforward_norm_correlation = _correlation(pre_ff_actual, pre_ff_expected)
        if _correlation(pre_ff_actual, pre_ff_reference) < DEFAULT_THRESHOLD:
            blockers.append("decode-pre-feedforward-norm-correlation-low")
        gate_actual, gate_expected, _, gate_evidence = _run_projection(
            family="gate_proj",
            weight=_load_safetensor_array(weights_dir, projection_keys["gate_proj"]),
            activation=pre_ff_actual.reshape(-1),
            object_file=object_file,
            runner_cache=runner_cache,
            timed_kernel_seconds=timed_kernel_samples,
            power_meter=power_meter,
        )
        up_actual, up_expected, _, up_evidence = _run_projection(
            family="up_proj",
            weight=_load_safetensor_array(weights_dir, projection_keys["up_proj"]),
            activation=pre_ff_actual.reshape(-1),
            object_file=object_file,
            runner_cache=runner_cache,
            timed_kernel_seconds=timed_kernel_samples,
            power_meter=power_meter,
        )
        for fam, ev in (("gate_proj", gate_evidence), ("up_proj", up_evidence)):
            projection_evidence.append(
                ProjectionEvidence(ev.family, projection_keys[fam], ev.shape, ev.padded_shape, ev.row_blocks, ev.col_blocks, ev.projection_correlation, ev.dense_projection_correlation)
            )
            stdout_lines.append(f"{fam} correlation: {ev.projection_correlation:.6f}")
            if ev.projection_correlation is None or ev.projection_correlation < DEFAULT_THRESHOLD:
                blockers.append(f"decode-{fam}-correlation-low")
        mlp_actual, mlp_reference = _run_geglu_stage(
            name="mlp_activation",
            gate=gate_actual,
            up=up_actual,
            runner_cache=runner_cache,
            timed_kernel_seconds=timed_kernel_samples,
            power_meter=power_meter,
        )
        mlp_expected = _geglu(gate_expected, up_expected)
        mlp_activation_correlation = _correlation(mlp_actual, mlp_expected)
        if _correlation(mlp_actual, mlp_reference) < DEFAULT_THRESHOLD:
            blockers.append("decode-mlp-activation-correlation-low")

        down_actual, down_expected, _, down_evidence = _run_projection(
            family="down_proj",
            weight=_load_safetensor_array(weights_dir, projection_keys["down_proj"]),
            activation=mlp_actual,
            object_file=object_file,
            runner_cache=runner_cache,
            timed_kernel_seconds=timed_kernel_samples,
            power_meter=power_meter,
        )
        projection_evidence.append(
            ProjectionEvidence(down_evidence.family, projection_keys["down_proj"], down_evidence.shape, down_evidence.padded_shape, down_evidence.row_blocks, down_evidence.col_blocks, down_evidence.projection_correlation, down_evidence.dense_projection_correlation)
        )
        stdout_lines.append(f"down_proj correlation: {down_evidence.projection_correlation:.6f}")
        if down_evidence.projection_correlation is None or down_evidence.projection_correlation < DEFAULT_THRESHOLD:
            blockers.append("decode-down_proj-correlation-low")

        post_ff_actual, post_ff_reference = _run_rms_stage(
            name="post_feedforward_norm",
            x=down_actual.reshape(1, 1152),
            weight=norm_weights["post_feedforward_layernorm"],
            runner_cache=runner_cache,
            timed_kernel_seconds=timed_kernel_samples,
            power_meter=power_meter,
        )
        post_ff_expected = _rms_host(down_expected.reshape(1, 1152), norm_weights["post_feedforward_layernorm"])
        post_feedforward_norm_correlation = _correlation(post_ff_actual, post_ff_expected)
        if _correlation(post_ff_actual, post_ff_reference) < DEFAULT_THRESHOLD:
            blockers.append("decode-post-feedforward-norm-correlation-low")
        output_actual_arr, mlp_residual_reference = _run_residual_stage(
            name="mlp",
            lhs=residual_actual.reshape(-1),
            rhs=post_ff_actual.reshape(-1),
            runner_cache=runner_cache,
            timed_kernel_seconds=timed_kernel_samples,
            power_meter=power_meter,
        )
        output_actual = output_actual_arr.reshape(-1)
        output_expected = (residual_expected.astype(np.float32) + post_ff_expected.astype(np.float32)).astype(bfloat16).reshape(-1)
        mlp_residual_correlation = _correlation(output_actual, output_expected)
        if _correlation(output_actual, mlp_residual_reference.reshape(-1)) < DEFAULT_THRESHOLD:
            blockers.append("decode-mlp-residual-add-correlation-low")
        final_output_correlation = _correlation(output_actual, output_expected)
        dense_final_output_correlation = None
        stdout_lines.append(f"final output correlation: {final_output_correlation:.6f}")
        if final_output_correlation < DEFAULT_THRESHOLD:
            blockers.append("decode-full-layer-output-correlation-low")
        returncode = 0 if not blockers else 1
    except Exception as exc:
        blockers.append(f"decode-full-layer-probe-failed:{exc}")
        rms_correlation = None
        attention_correlation = None
        q_norm_correlation = None
        k_norm_correlation = None
        post_attention_norm_correlation = None
        pre_feedforward_norm_correlation = None
        post_feedforward_norm_correlation = None
        rope_q_correlation = None
        rope_k_correlation = None
        attention_residual_correlation = None
        mlp_residual_correlation = None
        mlp_activation_correlation = None
        final_output_correlation = None
        dense_final_output_correlation = None
        norm_offset = None
        norm_payload = None
        norm_tensor_key = ""
        norm_argument_mode = getattr(args, "norm_argument_mode", "selected-vector")
        norm_argument = None
        returncode = 1
        stderr_lines.append(str(exc))
    finally:
        if runner_cache is not None:
            runner_cache.__exit__(None, None, None)

    elapsed = time.perf_counter() - start
    timed_total = sum(timed_kernel_samples) if timed_kernel_samples else None
    timed_mean = (timed_total / len(timed_kernel_samples)) if timed_total is not None and timed_kernel_samples else None
    diagnostic_layer_tps = (1.0 / timed_total) if timed_total and timed_total > 0.0 else None
    estimated_decode_tps = (1.0 / (timed_total * 26.0)) if timed_total and timed_total > 0.0 else None
    status = "FULL_LAYER_SEQUENCE_PASS" if not blockers else "FULL_LAYER_SEQUENCE_BLOCKED"
    return Gemma3FullLayerProbeResult(
        schema_version=1,
        model_variant=args.model_variant,
        status=status,
        sequence_kind=DEFAULT_SEQUENCE_KIND,
        phase=DEFAULT_PHASE,
        layer_index=layer_index,
        stages=(
            f"decode:L{layer_index}:pre_attention_norm",
            f"decode:L{layer_index}:qkv_projection",
            f"decode:L{layer_index}:qk_norm",
            f"decode:L{layer_index}:rope_halfsplit_npu_identity_pos0",
            f"decode:L{layer_index}:single_token_attention_flowqkv_npu",
            f"decode:L{layer_index}:output_projection",
            f"decode:L{layer_index}:post_attention_norm",
            f"decode:L{layer_index}:attention_residual",
            f"decode:L{layer_index}:pre_feedforward_norm",
            f"decode:L{layer_index}:mlp_gate_up_projection",
            f"decode:L{layer_index}:mlp_activation",
            f"decode:L{layer_index}:mlp_down_projection",
            f"decode:L{layer_index}:post_feedforward_norm",
            f"decode:L{layer_index}:mlp_residual",
        ),
        input_shape=(1, 1152),
        output_shape=(1152,),
        output_format=DEFAULT_OUTPUT_FORMAT,
        bo_binding_mode="runner-owned-persistent-bo",
        runner_reuse_mode=("reused-elf-persistent-bo" if runner_reuse_enabled else "per-launch-compile-load"),
        norm_tensor_key=norm_tensor_key,
        static_norm_argument_mode=norm_argument_mode,
        static_norm_tensor_offset_bytes=norm_offset,
        static_norm_bo_bytes=None if norm_payload is None else int(norm_payload.nbytes),
        static_norm_argument_bytes=None if norm_argument is None else int(norm_argument.nbytes),
        norm_tensor_keys=norm_tensor_keys,
        projection_tensor_keys=projection_keys,
        projection_weight_layout="fused-dqp-paper-repacked-full-layer-colblock-loop",
        host_fallbacks=(),
        input_distribution=DEFAULT_INPUT_DISTRIBUTION,
        rms_correlation=rms_correlation,
        projection_evidence=tuple(projection_evidence),
        attention_correlation=attention_correlation,
        q_norm_correlation=q_norm_correlation,
        k_norm_correlation=k_norm_correlation,
        post_attention_norm_correlation=post_attention_norm_correlation,
        pre_feedforward_norm_correlation=pre_feedforward_norm_correlation,
        post_feedforward_norm_correlation=post_feedforward_norm_correlation,
        rope_q_correlation=rope_q_correlation,
        rope_k_correlation=rope_k_correlation,
        attention_residual_correlation=attention_residual_correlation,
        mlp_residual_correlation=mlp_residual_correlation,
        mlp_activation_correlation=mlp_activation_correlation,
        final_output_correlation=final_output_correlation,
        dense_final_output_correlation=dense_final_output_correlation,
        timed_kernel_count=len(timed_kernel_samples),
        timed_kernel_seconds=timed_total,
        timed_kernel_mean_seconds=timed_mean,
        diagnostic_layer_passes_per_second=diagnostic_layer_tps,
        estimated_26_layer_decode_tps_kernel_only=estimated_decode_tps,
        timing_window="segmented-run-start-wait2-only",
        timing_notes=(
            "timed_kernel_seconds sums only pyxrt run.start()/wait2() calls for this staged layer",
            "compile, ELF load, BO allocation, BO writes/preload, argument binding, output sync/readback, and host fallback compute are excluded",
            "reused ELF mode compiles, loads, allocates BOs, and binds kernel arguments before the timed launch segments",
            "RMSNorm uses a preselected BF16 norm-vector argument because the current two-argument RMSNorm ABI has no static-norm BO offset parameter",
            "estimated_26_layer_decode_tps_kernel_only is an extrapolation from one layer and is not a measured full-model decode TPS",
        ),
        power_snapshot=power_meter.snapshot(),
        threshold=DEFAULT_THRESHOLD,
        remaining_model_runner_gaps=("full-1b-loop-not-wired",),
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
    projection_evidence = tuple(
        ProjectionEvidence(
            family=family,
            tensor_key=f"model.layers.0.fixture.{family}.weight",
            shape=PROJECTION_SHAPES[family],
            padded_shape=PROJECTION_SHAPES[family],
            row_blocks=8,
            col_blocks=1,
            projection_correlation=1.0,
            dense_projection_correlation=0.995,
        )
        for family in FULL_LAYER_PROJECTION_FAMILIES
    )
    result = Gemma3FullLayerProbeResult(
        schema_version=1,
        model_variant=DEFAULT_MODEL,
        status="FULL_LAYER_SEQUENCE_PASS",
        sequence_kind=DEFAULT_SEQUENCE_KIND,
        phase=DEFAULT_PHASE,
        layer_index=DEFAULT_LAYER,
        stages=("decode:L0:pre_attention_norm", "decode:L0:qkv_projection", "decode:L0:mlp_residual"),
        input_shape=(1, 1152),
        output_shape=(1152,),
        output_format=DEFAULT_OUTPUT_FORMAT,
        bo_binding_mode="runner-owned-persistent-bo",
        runner_reuse_mode="reused-elf-persistent-bo",
        norm_tensor_key=DEFAULT_NORM_TENSOR_KEY,
        static_norm_argument_mode="selected-vector",
        static_norm_tensor_offset_bytes=0,
        static_norm_bo_bytes=266240,
        static_norm_argument_bytes=2304,
        norm_tensor_keys=NORM_TENSOR_KEYS,
        projection_tensor_keys={family: f"model.layers.0.fixture.{family}.weight" for family in FULL_LAYER_PROJECTION_FAMILIES},
        projection_weight_layout="fused-dqp-paper-repacked-full-layer-colblock-loop",
        host_fallbacks=(),
        input_distribution=DEFAULT_INPUT_DISTRIBUTION,
        rms_correlation=0.999991,
        projection_evidence=projection_evidence,
        attention_correlation=1.0,
        q_norm_correlation=1.0,
        k_norm_correlation=1.0,
        post_attention_norm_correlation=1.0,
        pre_feedforward_norm_correlation=1.0,
        post_feedforward_norm_correlation=1.0,
        rope_q_correlation=1.0,
        rope_k_correlation=1.0,
        attention_residual_correlation=1.0,
        mlp_residual_correlation=1.0,
        mlp_activation_correlation=1.0,
        final_output_correlation=0.999998,
        dense_final_output_correlation=None,
        timed_kernel_count=len(projection_evidence) + 1,
        timed_kernel_seconds=0.123,
        timed_kernel_mean_seconds=0.123 / float(len(projection_evidence) + 1),
        diagnostic_layer_passes_per_second=1.0 / 0.123,
        estimated_26_layer_decode_tps_kernel_only=1.0 / (0.123 * 26.0),
        timing_window="segmented-run-start-wait2-only",
        timing_notes=("fixture",),
        power_snapshot=None,
        threshold=DEFAULT_THRESHOLD,
        remaining_model_runner_gaps=("full-1b-loop-not-wired",),
        command=("python3", "gemma3.probes.full_layer", "--self-test"),
        returncode=0,
        elapsed_seconds=0.125,
        blockers=(),
        git_commit="fixture",
        dirty_worktree=False,
        stdout_tail=("final output correlation: 0.999998",),
        stderr_tail=(),
    )
    if result.status != "FULL_LAYER_SEQUENCE_PASS":
        raise AssertionError(result)
    if not _is_decode_full_layer_evidence(result.to_json_dict(), model_variant=DEFAULT_MODEL):
        raise AssertionError(result.to_json_dict())
    stale = dict(result.to_json_dict())
    stale["remaining_model_runner_gaps"] = ["full-layer-not-wired"]
    if _is_decode_full_layer_evidence(stale, model_variant=DEFAULT_MODEL):
        raise AssertionError(stale)
    print(result.format())
    print("GEMMA3_FULL_LAYER_PROBE_SELF_TEST: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma3 staged decode full-layer probe")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--run-hardware", action="store_true")
    parser.add_argument("--model-variant", choices=sorted(MODEL_SPECS), default=DEFAULT_MODEL)
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--layer-index", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--norm-tensor-key")
    parser.add_argument(
        "--norm-argument-mode",
        choices=("auto", "selected-vector", "contiguous-payload"),
        default="selected-vector",
        help="RMSNorm static weight argument mode; selected-vector is the model-loop-compatible offset workaround",
    )
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--power-sample", action="store_true")
    parser.add_argument("--no-reuse-elf", action="store_true", help="diagnostic fallback: compile/load every launch")
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
        print(f"GEMMA3_FULL_LAYER_PROBE_JSON: {args.result_json}")
    return 0 if result.status == "FULL_LAYER_SEQUENCE_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
