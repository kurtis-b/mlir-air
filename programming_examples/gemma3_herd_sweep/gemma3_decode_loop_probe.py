#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gemma3 real-weight staged decode loop probe.

This diagnostic runs the existing staged full-layer decode route repeatedly
across Gemma3 1B layers with one reusable ELF runner cache. It is not a paper
TTFT/TPS result: attention is still the single-token staged NPU path, logits and
sampling are absent, and CPU reference checks remain inside the diagnostic loop.
The default diagnostic preloads packed projection inputs into runner-owned BO
sets before timing; dynamic mode can still rewrite them per launch for
comparison. The result records exactly which timing windows are measured.
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
from typing import Any

from gemma3_artifacts import MODEL_SPECS
from gemma3_full_layer_probe import (
    DEFAULT_FULL_LAYER_PROBE_EVIDENCE,
    FULL_LAYER_PROJECTION_FAMILIES,
    PROJECTION_SHAPES,
    ProjectionEvidence,
    _ReusableElfRunnerCache,
    _SegmentedRAPLPowerMeter,
    _aie_api_include,
    _ceil_to,
    _dataflow_dir,
    _prepared_static_arg,
    _geglu,
    _norm_tensor_keys,
    _projection_tensor_keys,
    _rms_host,
    _run_geglu_stage,
    _run_residual_stage,
    _run_rms_stage,
    _run_rope_stage,
    _run_single_token_attention_stage,
)
from gemma3_power import begin_power_window, finish_power_window
from gemma3_qkv_substep_probe import _projection_backend_options
from gemma3_substep_probe import (
    DEFAULT_INPUT_DISTRIBUTION,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_PHASE,
    DEFAULT_THRESHOLD,
    _activate_probe_env,
    _correlation,
    _git_info,
    _repo_root,
    _resolve_weights_dir,
    _shape_text,
    _tail,
    _write_bo_arg,
)

DEFAULT_SEQUENCE_KIND = "decode-loop-staged-full-1b"
DEFAULT_LOOP_PROBE_EVIDENCE = (
    Path(__file__).with_name("results") / "gemma3_1b_decode_loop_probe.json"
)
DEFAULT_TILED_LOOP_PROBE_EVIDENCE = (
    Path(__file__).with_name("results") / "gemma3_1b_decode_loop_tiled_stats_probe.json"
)
PAPER_DECODE_TPS_1K = 41.1
DEFAULT_INGRESS_MODE = "staged"
DEFAULT_ATTENTION_MODE = "single-token"
DEFAULT_ATTENTION_CACHE_MODE = "repeated-current-token"
DEFAULT_TILED_ATTENTION_KV_TILE = 32
DEFAULT_TILED_ATTENTION_HOST_BATCH_TILES = 2
STITCHED_INGRESS_BO_ALIASES = {3: 2}
_INGRESS_PACK_CACHE: dict[tuple[object, ...], Any] = {}


@dataclass(frozen=True)
class LayerLoopEvidence:
    layer_index: int
    norm_tensor_key: str
    static_norm_tensor_offset_bytes: int
    static_norm_argument_bytes: int
    rms_correlation: float | None
    final_output_correlation: float | None
    projection_evidence: tuple[ProjectionEvidence, ...]
    timed_kernel_count: int
    timed_kernel_seconds: float | None

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Gemma3DecodeLoopProbeResult:
    schema_version: int
    model_variant: str
    status: str
    sequence_kind: str
    phase: str
    layer_count: int
    decode_tokens: int
    prompt_context_length: int
    output_format: str
    runner_reuse_mode: str
    ingress_mode: str
    norm_argument_mode: str
    static_projection_argument_mode: str
    static_projection_bo_set_count: int
    static_ingress_bo_set_count: int
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    input_distribution: str
    reference_check_mode: str
    reference_check_layer_count: int
    reference_check_seconds: float | None
    attention_mode: str
    attention_cache_contract: str
    attention_kv_tile: int | None
    attention_host_batch_tiles: int | None
    attention_host_batch_count: int | None
    attention_host_reduction: bool
    host_fallbacks: tuple[str, ...]
    timed_kernel_count: int
    timed_kernel_seconds: float | None
    timed_kernel_mean_seconds: float | None
    measured_loop_seconds: float | None
    diagnostic_decode_tps_loop_wall: float | None
    diagnostic_decode_tps_kernel_only: float | None
    paper_decode_tps_1k: float
    loop_wall_delta_pct_vs_paper_decode_tps_1k: float | None
    kernel_only_delta_pct_vs_paper_decode_tps_1k: float | None
    timing_window: str
    timing_notes: tuple[str, ...]
    power_snapshot: dict[str, object] | None
    segmented_kernel_power_snapshot: dict[str, object] | None
    layer_evidence: tuple[LayerLoopEvidence, ...]
    remaining_paper_gaps: tuple[str, ...]
    command: tuple[str, ...]
    returncode: int | None
    elapsed_seconds: float | None
    blockers: tuple[str, ...]
    git_commit: str | None
    dirty_worktree: bool | None
    stdout_tail: tuple[str, ...]
    stderr_tail: tuple[str, ...]

    @staticmethod
    def _value_text(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.6f}"

    def format(self) -> str:
        blockers = ",".join(self.blockers) if self.blockers else "none"
        gaps = ",".join(self.remaining_paper_gaps) if self.remaining_paper_gaps else "none"
        return (
            f"decode_loop_probe model={self.model_variant} status={self.status} "
            f"sequence={self.sequence_kind} phase={self.phase} layers={self.layer_count} "
            f"decode_tokens={self.decode_tokens} context={self.prompt_context_length} "
            f"input={_shape_text(self.input_shape)} output={_shape_text(self.output_shape)} "
            f"output_format={self.output_format} runner_reuse={self.runner_reuse_mode} "
            f"ingress={self.ingress_mode} "
            f"norm_arg={self.norm_argument_mode} static_projection_arg={self.static_projection_argument_mode} "
            f"static_projection_bo_sets={self.static_projection_bo_set_count} "
            f"static_ingress_bo_sets={self.static_ingress_bo_set_count} "
            f"attention_mode={self.attention_mode} attention_cache={self.attention_cache_contract} "
            f"timed_kernel_count={self.timed_kernel_count} "
            f"timed_kernel_seconds={self._value_text(self.timed_kernel_seconds)} "
            f"measured_loop_seconds={self._value_text(self.measured_loop_seconds)} "
            f"diagnostic_decode_tps_loop_wall={self._value_text(self.diagnostic_decode_tps_loop_wall)} "
            f"diagnostic_decode_tps_kernel_only={self._value_text(self.diagnostic_decode_tps_kernel_only)} "
            f"paper_decode_tps_1k={self.paper_decode_tps_1k:g} "
            f"threshold={DEFAULT_THRESHOLD:g} paper_gaps={gaps} blockers={blockers}"
        )

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _NormPlan:
    tensor_key: str
    weight: Any
    offset_bytes: int
    static_bo_bytes: int
    argument_bytes: int
    norm_weights: dict[str, Any]


@dataclass(frozen=True)
class _PackedProjectionPlan:
    family: str
    tensor_key: str
    shape: tuple[int, int]
    padded_shape: tuple[int, int]
    row_blocks: int
    col_blocks: int
    packed: Any
    scale: Any
    min_offset: Any
    mlir_module: Any




def has_decode_loop_tiled_stats_evidence(
    model_variant: str,
    path: Path | None = None,
) -> bool:
    evidence_path = path or DEFAULT_TILED_LOOP_PROBE_EVIDENCE
    try:
        data = json.loads(evidence_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        data.get("schema_version") == 2
        and data.get("model_variant") == model_variant
        and data.get("status") == "DECODE_LOOP_DIAGNOSTIC_PASS"
        and data.get("sequence_kind") == DEFAULT_SEQUENCE_KIND
        and data.get("phase") == DEFAULT_PHASE
        and data.get("layer_count") == 26
        and data.get("decode_tokens") == 1
        and data.get("prompt_context_length") == 1024
        and data.get("attention_mode") == "tiled-stats-1k"
        and data.get("attention_cache_contract") == "synthetic-prefill-kv-cache"
        and data.get("attention_host_batch_count") == 16
        and data.get("attention_host_reduction") is True
        and data.get("output_format") == DEFAULT_OUTPUT_FORMAT
        and data.get("runner_reuse_mode") == "reused-elf-persistent-bo"
        and not data.get("host_fallbacks")
        and not data.get("blockers")
        and data.get("dirty_worktree") is False
        and "prefill-produced-kv-cache-not-wired" in tuple(data.get("remaining_paper_gaps", ()))
        and "paper-1k-kv-attention-npu-reduction-not-wired"
        in tuple(data.get("remaining_paper_gaps", ()))
    )

def _delta_pct(local_tps: float | None) -> float | None:
    if local_tps is None:
        return None
    return abs(float(local_tps) - PAPER_DECODE_TPS_1K) / PAPER_DECODE_TPS_1K * 100.0


def _load_weight_array(weights_dir: Path, tensor_key: str):
    from gemma3_stitched_decode import _load_safetensor_array_np

    return _load_safetensor_array_np(weights_dir, tensor_key)


def _load_static_norm_payload_np(weights_dir: Path, model_variant: str, tensor_key: str):
    import numpy as np
    from gemma3_norm_weight_plan import build_norm_weight_plan
    from ml_dtypes import bfloat16

    plan = build_norm_weight_plan(model_variant, weights_dir=weights_dir)
    vectors = []
    tensor_offset = 0
    selected = None
    selected_offset = None
    for record in plan.records:
        vector = _load_weight_array(weights_dir, record.tensor_key).astype(bfloat16).reshape(-1)
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


def _repack_projection(weight, *, row_block_multiple: int = 8):
    import numpy as np
    from ml_dtypes import bfloat16
    from common import Q4NX_COLS, Q4NX_ROWS, pack_int4_low_first

    rows, cols = weight.shape
    padded_rows = _ceil_to(_ceil_to(int(rows), Q4NX_ROWS) // Q4NX_ROWS, row_block_multiple) * Q4NX_ROWS
    padded_cols = _ceil_to(int(cols), Q4NX_COLS)
    row_blocks = padded_rows // Q4NX_ROWS
    col_blocks = padded_cols // Q4NX_COLS
    padded = np.zeros((padded_rows, padded_cols), dtype=np.float32)
    padded[:rows, :cols] = weight.astype(np.float32)
    packed = np.empty((row_blocks, col_blocks, Q4NX_ROWS * Q4NX_COLS // 2), dtype=np.int8)
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
    return packed, scale, min_offset, (padded_rows, padded_cols)


def _prepare_layer_plans(model_variant: str, weights_dir: Path, layers: int):
    from fused_dqp import build_paper_module
    from ml_dtypes import bfloat16

    object_file = Path(__file__).with_name("build_peano") / "fused_dqp.o"
    if not object_file.exists():
        raise RuntimeError(f"missing FusedDQP object file: {object_file}")
    norm_plans: dict[int, _NormPlan] = {}
    projection_plans: dict[int, dict[str, _PackedProjectionPlan]] = {}
    module_cache: dict[int, Any] = {}
    for layer_index in range(layers):
        norm_keys = _norm_tensor_keys(layer_index)
        tensor_key = norm_keys["input_layernorm"]
        norm_payload, norm_weight, norm_offset = _load_static_norm_payload_np(weights_dir, model_variant, tensor_key)
        norm_weights = {
            name: _load_weight_array(weights_dir, key).astype(bfloat16).reshape(-1)
            for name, key in norm_keys.items()
        }
        norm_plans[layer_index] = _NormPlan(
            tensor_key=tensor_key,
            weight=norm_weight,
            offset_bytes=int(norm_offset),
            static_bo_bytes=int(norm_payload.nbytes),
            argument_bytes=int(norm_weight.nbytes),
            norm_weights=norm_weights,
        )
        keys = _projection_tensor_keys(model_variant, weights_dir, layer_index)
        layer_projection_plans: dict[str, _PackedProjectionPlan] = {}
        for family in FULL_LAYER_PROJECTION_FAMILIES:
            weight = _load_weight_array(weights_dir, keys[family])
            expected_shape = PROJECTION_SHAPES[family]
            if tuple(weight.shape) != expected_shape:
                raise RuntimeError(f"expected {family} shape {expected_shape}, got {weight.shape}")
            packed, scale, min_offset, padded_shape = _repack_projection(weight)
            row_blocks = int(packed.shape[0])
            if row_blocks not in module_cache:
                module_cache[row_blocks] = build_paper_module(
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
            layer_projection_plans[family] = _PackedProjectionPlan(
                family=family,
                tensor_key=keys[family],
                shape=expected_shape,
                padded_shape=tuple(int(dim) for dim in padded_shape),
                row_blocks=row_blocks,
                col_blocks=int(packed.shape[1]),
                packed=packed,
                scale=scale,
                min_offset=min_offset,
                mlir_module=module_cache[row_blocks],
            )
        projection_plans[layer_index] = layer_projection_plans
    return norm_plans, projection_plans


def _projection_bo_set_key(plan: _PackedProjectionPlan, col_block: int) -> tuple[object, ...]:
    return ("fused-dqp-static-projection", plan.tensor_key, int(col_block))


def _projection_static_input_key(plan: _PackedProjectionPlan, col_block: int) -> tuple[object, ...]:
    return ("packed-l3", plan.tensor_key, int(col_block))


def _packed_l3_for_col_block(plan: _PackedProjectionPlan, col_block: int):
    import numpy as np
    from ml_dtypes import bfloat16
    from fused_dqp import _pack_l3_inputs

    cb_slice = slice(col_block, col_block + 1)
    params = np.empty((plan.row_blocks, 1, 512), dtype=bfloat16)
    params[..., :256] = plan.scale[:, cb_slice, :]
    params[..., 256:] = plan.min_offset[:, cb_slice, :]
    return _pack_l3_inputs(plan.packed[:, cb_slice, :], params).reshape(
        plan.row_blocks // 4,
        4,
        1,
        -1,
    )


def _packed_l3_static_placeholder(plan: _PackedProjectionPlan):
    import numpy as np

    block_bytes = int(plan.packed.shape[-1]) + 512 * np.dtype(plan.scale.dtype).itemsize
    shape = (plan.row_blocks // 4, 4, 1, block_bytes)
    return _prepared_static_arg(shape, np.dtype(np.int8), int(np.prod(shape)))


def _full_packed_l3_for_ingress(plan: _PackedProjectionPlan):
    import numpy as np
    from fused_dqp import _pack_l3_inputs

    key = (plan.tensor_key, int(plan.row_blocks), int(plan.col_blocks), tuple(plan.packed.shape), tuple(plan.scale.shape))
    cached = _INGRESS_PACK_CACHE.get(key)
    if cached is not None:
        return cached
    params = np.empty((plan.row_blocks, plan.col_blocks, 512), dtype=plan.scale.dtype)
    params[..., :256] = plan.scale
    params[..., 256:] = plan.min_offset
    cached = _pack_l3_inputs(plan.packed, params).reshape(
        plan.row_blocks // 4,
        4,
        plan.col_blocks,
        -1,
    )
    _INGRESS_PACK_CACHE[key] = cached
    return cached


def _stitched_ingress_bo_set_key(layer_index: int) -> tuple[object, ...]:
    return ("stitched-ingress", int(layer_index))


def _stitched_ingress_static_input_keys(
    *,
    layer_index: int,
    norm_plan: _NormPlan,
    projection_plans: dict[str, _PackedProjectionPlan],
) -> list[object | None]:
    keys: list[object | None] = [None] * 18
    keys[1] = ("stitched-ingress-static", norm_plan.tensor_key)
    keys[2] = ("stitched-ingress-zero-tail", int(layer_index), "activation")
    keys[4] = ("stitched-ingress-static", projection_plans["q_proj"].tensor_key, "packed-l3-full")
    keys[5] = ("stitched-ingress-static", projection_plans["k_proj"].tensor_key, "packed-l3-full")
    keys[6] = ("stitched-ingress-static", projection_plans["v_proj"].tensor_key, "packed-l3-full")
    keys[10] = ("stitched-ingress-static", f"model.layers.{int(layer_index)}.self_attn.q_norm.weight")
    keys[11] = ("stitched-ingress-static", f"model.layers.{int(layer_index)}.self_attn.k_norm.weight")
    keys[14] = ("stitched-ingress-static", "identity-rope", 4, 256)
    keys[15] = ("stitched-ingress-static", "identity-rope", 1, 256)
    for index, name in (
        (7, "q_proj"),
        (8, "k_proj"),
        (9, "v_proj"),
        (12, "q_norm"),
        (13, "k_norm"),
        (16, "q_rope"),
        (17, "k_rope"),
    ):
        keys[index] = ("stitched-ingress-zero-output", int(layer_index), name)
    return keys


def _stitched_ingress_arrays(
    *,
    x_input,
    norm_plan: _NormPlan,
    projection_plans: dict[str, _PackedProjectionPlan],
):
    import numpy as np
    from gemma3_stitched_decode import _identity_rope_lut
    from ml_dtypes import bfloat16

    activation_storage = np.zeros((5, 256), dtype=bfloat16)
    return [
        x_input.reshape(1, 1152).astype(bfloat16),
        norm_plan.weight.reshape(-1).astype(bfloat16),
        activation_storage,
        activation_storage,
        _full_packed_l3_for_ingress(projection_plans["q_proj"]),
        _full_packed_l3_for_ingress(projection_plans["k_proj"]),
        _full_packed_l3_for_ingress(projection_plans["v_proj"]),
        np.zeros((32, 32), dtype=bfloat16),
        np.zeros((8, 32), dtype=bfloat16),
        np.zeros((8, 32), dtype=bfloat16),
        norm_plan.norm_weights["q_norm"].reshape(-1).astype(bfloat16),
        norm_plan.norm_weights["k_norm"].reshape(-1).astype(bfloat16),
        np.zeros((4, 256), dtype=bfloat16),
        np.zeros((1, 256), dtype=bfloat16),
        _identity_rope_lut(4, 256, bfloat16).reshape(-1),
        _identity_rope_lut(1, 256, bfloat16).reshape(-1),
        np.zeros((4, 256), dtype=bfloat16),
        np.zeros((1, 256), dtype=bfloat16),
    ]


def _stitched_ingress_readback(dtype) -> dict[str, tuple[int, tuple[int, ...], object]]:
    return {
        "input_norm": (2, (1, 1152), dtype),
        "activation": (3, (5, 256), dtype),
        "q_proj": (7, (32, 32), dtype),
        "k_proj": (8, (8, 32), dtype),
        "v_proj": (9, (8, 32), dtype),
        "q_norm": (12, (4, 256), dtype),
        "k_norm": (13, (1, 256), dtype),
        "q_rope": (16, (4, 256), dtype),
        "k_rope": (17, (1, 256), dtype),
    }


class _ReusableMultiOutputElfRunner:
    def __init__(self, cache, *, mlir_module, backend_options: dict[str, object]) -> None:
        import numpy as np
        from air.backend.xrt import XRTBackend

        self.cache = cache
        self.backend = XRTBackend(**backend_options)
        self.artifact = self.backend.compile(mlir_module)
        self.elf = cache.xrt.elf(self.artifact.output_binary)
        self.context = cache.xrt.hw_context(cache.device, self.elf)
        self.kernel = cache.xrt.ext.kernel(self.context, self.artifact.kernel)
        self.state: dict[str, object] | None = None
        self.bo_sets: dict[tuple[object, ...], dict[str, object]] = {}
        self._np = np

    @staticmethod
    def _array_nbytes(array) -> int:
        if hasattr(array, "nbytes"):
            return int(array.nbytes)
        if hasattr(array, "size") and hasattr(array, "itemsize"):
            return int(array.size * array.itemsize)
        raise RuntimeError(f"unsupported argument placeholder: {type(array)!r}")

    def _allocate_bos(self, sizes: list[int], bo_aliases: dict[int, int]):
        bos = []
        for index, size in enumerate(sizes):
            alias = bo_aliases.get(index)
            if alias is None:
                bos.append(self.cache.xrt.ext.bo(self.cache.device, size))
                continue
            if alias >= len(bos):
                raise RuntimeError(f"BO alias {index}->{alias} targets an unallocated argument")
            if sizes[alias] < size:
                raise RuntimeError(f"BO alias {index}->{alias} target is smaller than alias view")
            bos.append(bos[alias])
        return bos

    def _state_for(
        self,
        *,
        arrays: list[object],
        bo_aliases: dict[int, int],
        bo_set_key: tuple[object, ...] | None,
    ):
        sizes = [self._array_nbytes(array) for array in arrays]
        alias_key = dict(bo_aliases)
        if bo_set_key is None:
            if self.state is None:
                self.state = {
                    "sizes": sizes,
                    "static_keys": [None] * len(arrays),
                    "bo_aliases": alias_key,
                    "bos": self._allocate_bos(sizes, bo_aliases),
                }
            elif self.state["sizes"] != sizes or self.state["bo_aliases"] != alias_key:
                raise RuntimeError("reused multi-output ELF runner argument layout changed")
            return self.state["bos"], self.state["static_keys"]

        state = self.bo_sets.get(bo_set_key)
        if state is None:
            state = {
                "sizes": sizes,
                "static_keys": [None] * len(arrays),
                "bo_aliases": alias_key,
                "bos": self._allocate_bos(sizes, bo_aliases),
            }
            self.bo_sets[bo_set_key] = state
        elif state["sizes"] != sizes or state["bo_aliases"] != alias_key:
            raise RuntimeError(
                f"reused multi-output ELF runner BO-set layout mismatch for {bo_set_key}"
            )
        return state["bos"], state["static_keys"]

    def _write_args(
        self,
        *,
        bos,
        arrays: list[object],
        static_keys: list[object | None],
        requested_static_keys: list[object | None],
        bo_aliases: dict[int, int],
        write_dynamic: bool,
    ) -> None:
        for index, (bo, array) in enumerate(zip(bos, arrays)):
            if index in bo_aliases:
                continue
            requested_key = requested_static_keys[index]
            if requested_key is None and not write_dynamic:
                continue
            if requested_key is not None and static_keys[index] == requested_key:
                continue
            if not hasattr(array, "dtype"):
                raise RuntimeError(
                    "prepared static placeholder reached a multi-output BO write; "
                    "preload the matching static input before timed execution"
                )
            _write_bo_arg(self.cache.xrt, bo, array)
            if requested_key is not None:
                static_keys[index] = requested_key

    def prepare(
        self,
        *,
        arrays: list[object],
        bo_set_key: tuple[object, ...],
        static_input_keys: list[object | None],
        bo_aliases: dict[int, int] | None = None,
    ) -> None:
        if len(static_input_keys) != len(arrays):
            raise RuntimeError(
                f"static_input_keys length mismatch: expected {len(arrays)}, got {len(static_input_keys)}"
            )
        bo_aliases = bo_aliases or {}
        bos, static_keys = self._state_for(arrays=arrays, bo_aliases=bo_aliases, bo_set_key=bo_set_key)
        self._write_args(
            bos=bos,
            arrays=arrays,
            static_keys=static_keys,
            requested_static_keys=static_input_keys,
            bo_aliases=bo_aliases,
            write_dynamic=False,
        )

    def run(
        self,
        *,
        arrays: list[object],
        readback: dict[str, tuple[int, tuple[int, ...], object]],
        timed_kernel_seconds: list[float] | None,
        power_meter,
        bo_set_key: tuple[object, ...] | None = None,
        static_input_keys: list[object | None] | None = None,
        bo_aliases: dict[int, int] | None = None,
    ) -> dict[str, object]:
        bo_aliases = bo_aliases or {}
        bos, static_keys = self._state_for(arrays=arrays, bo_aliases=bo_aliases, bo_set_key=bo_set_key)
        if static_input_keys is None:
            static_input_keys = [None] * len(arrays)
        if len(static_input_keys) != len(arrays):
            raise RuntimeError(
                f"static_input_keys length mismatch: expected {len(arrays)}, got {len(static_input_keys)}"
            )
        self._write_args(
            bos=bos,
            arrays=arrays,
            static_keys=static_keys,
            requested_static_keys=static_input_keys,
            bo_aliases=bo_aliases,
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
        outputs = {}
        for name, (index, shape, dtype) in readback.items():
            bos[index].sync(self.cache.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            read_size = int(self._np.prod(shape)) * self._np.dtype(dtype).itemsize
            outputs[name] = bos[index].read(read_size, 0).view(dtype).reshape(shape)
        return outputs

    def close(self) -> None:
        self.bo_sets.clear()
        self.backend.unload()


def _stitched_ingress_runner(
    runner_cache: _ReusableElfRunnerCache,
    *,
    fused_dqp_object_file: Path,
    rope_object_file: Path,
) -> _ReusableMultiOutputElfRunner:
    if not runner_cache.enabled:
        raise RuntimeError("--ingress-mode=stitched requires reusable ELF runners")
    from gemma3_stitched_decode import DEFAULT_INGRESS_FUNCTION_NAME, build_ingress_module, _stitched_backend_options

    key = ("stitched-ingress-multi-output", str(fused_dqp_object_file), str(rope_object_file))
    runner = runner_cache.runners.get(key)
    if runner is None:
        runner = _ReusableMultiOutputElfRunner(
            runner_cache,
            mlir_module=build_ingress_module(
                object_file=fused_dqp_object_file,
                rope_object_file=rope_object_file,
            ),
            backend_options=_stitched_backend_options(DEFAULT_INGRESS_FUNCTION_NAME),
        )
        runner_cache.runners[key] = runner
    return runner


def _preload_static_stitched_ingress_bo_sets(
    *,
    norm_plans: dict[int, _NormPlan],
    projection_plans: dict[int, dict[str, _PackedProjectionPlan]],
    runner_cache: _ReusableElfRunnerCache,
    fused_dqp_object_file: Path,
    rope_object_file: Path,
) -> int:
    import numpy as np
    from ml_dtypes import bfloat16

    runner = _stitched_ingress_runner(
        runner_cache,
        fused_dqp_object_file=fused_dqp_object_file,
        rope_object_file=rope_object_file,
    )
    count = 0
    zero_hidden = np.zeros((1, 1152), dtype=bfloat16)
    for layer_index in sorted(projection_plans):
        arrays = _stitched_ingress_arrays(
            x_input=zero_hidden,
            norm_plan=norm_plans[layer_index],
            projection_plans=projection_plans[layer_index],
        )
        runner.prepare(
            arrays=arrays,
            bo_set_key=_stitched_ingress_bo_set_key(layer_index),
            static_input_keys=_stitched_ingress_static_input_keys(
                layer_index=layer_index,
                norm_plan=norm_plans[layer_index],
                projection_plans=projection_plans[layer_index],
            ),
            bo_aliases=STITCHED_INGRESS_BO_ALIASES,
        )
        count += 1
    return count


def _projection_output_shape(plan: _PackedProjectionPlan) -> tuple[int, int]:
    return (int(plan.row_blocks), 32)


def _compile_flowqkv_tiled_stats_kernel(
    object_file: Path,
    *,
    q_chunk: int = 4,
    kv_tile: int = DEFAULT_TILED_ATTENTION_KV_TILE,
    head_dim: int = 256,
) -> None:
    peano = os.environ.get("PEANO_INSTALL_DIR")
    if not peano:
        raise RuntimeError("PEANO_INSTALL_DIR is required to compile flow_attention_stats.cc")
    clangxx = Path(peano) / "bin/clang++"
    if not clangxx.exists():
        raise RuntimeError(f"missing Peano clang++: {clangxx}")
    src = _dataflow_dir() / "flow_attention_stats.cc"
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
        f"-DQ_CHUNK={int(q_chunk)}",
        f"-DKV_TILE={int(kv_tile)}",
        f"-DHEAD_DIM={int(head_dim)}",
        "-c",
        str(src),
        "-o",
        str(object_file),
    ]
    subprocess.run(cmd, check=True)


def _flowqkv_tiled_stats_helpers():
    dataflow_dir = _dataflow_dir()
    if str(dataflow_dir) not in sys.path:
        sys.path.insert(0, str(dataflow_dir))
    module_path = dataflow_dir / "flowqkv_tiled_stats.py"
    spec = importlib.util.spec_from_file_location("gemma3_dataflow_flowqkv_tiled_stats", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load flowqkv_tiled_stats.py from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_tiled_stats_module, module.merge_tiled_stats


def _synthetic_prefill_kv_cache(
    *,
    layer_index: int,
    prompt_context_length: int,
):
    import numpy as np
    from ml_dtypes import bfloat16

    rng = np.random.default_rng(4096 + int(layer_index))
    val_range = 0.35
    k_full = rng.uniform(-val_range, val_range, (prompt_context_length, 256)).astype(bfloat16)
    v_full = rng.uniform(-val_range, val_range, (prompt_context_length, 256)).astype(bfloat16)
    return k_full, v_full


def _run_tiled_stats_attention_stage(
    *,
    layer_index: int,
    q,
    k,
    v,
    object_file: Path,
    runner_cache: _ReusableElfRunnerCache,
    prompt_context_length: int,
    kv_tile: int,
    host_batch_tiles: int,
    attention_cache_mode: str,
    timed_kernel_seconds: list[float] | None = None,
    power_meter=None,
):
    import numpy as np
    from ml_dtypes import bfloat16

    dataflow_dir = _dataflow_dir()
    if str(dataflow_dir) not in sys.path:
        sys.path.insert(0, str(dataflow_dir))
    from common import attention_reference

    if prompt_context_length % kv_tile != 0:
        raise RuntimeError("--prompt-context-length must be divisible by tiled attention kv tile")
    if host_batch_tiles <= 0:
        raise RuntimeError("--tiled-attention-host-batch-tiles must be positive")
    tile_count = prompt_context_length // kv_tile
    if tile_count % host_batch_tiles != 0:
        raise RuntimeError("tiled attention tile count must be divisible by host batch tiles")
    if not object_file.exists():
        _compile_flowqkv_tiled_stats_kernel(object_file, q_chunk=4, kv_tile=kv_tile, head_dim=256)
    build_tiled_stats_module, merge_tiled_stats = _flowqkv_tiled_stats_helpers()

    q_single = q.reshape(4, 256).astype(bfloat16)
    if attention_cache_mode == "repeated-current-token":
        k_full = np.broadcast_to(k.reshape(1, 256).astype(bfloat16), (prompt_context_length, 256)).copy()
        v_full = np.broadcast_to(v.reshape(1, 256).astype(bfloat16), (prompt_context_length, 256)).copy()
    elif attention_cache_mode == "synthetic-prefill":
        k_full, v_full = _synthetic_prefill_kv_cache(
            layer_index=layer_index,
            prompt_context_length=prompt_context_length,
        )
    else:
        raise RuntimeError(f"unsupported attention cache mode: {attention_cache_mode}")
    q_tiles = np.broadcast_to(q_single, (tile_count, 4, 256)).copy()
    k_tiles = k_full.reshape(tile_count, kv_tile, 256).copy()
    v_tiles = v_full.reshape(tile_count, kv_tile, 256).copy()
    module = build_tiled_stats_module(
        4,
        kv_tile,
        256,
        "flowqkv_tile_stats_bf16",
        str(object_file),
        host_batch_tiles,
        1,
        host_batch_tiles,
        "direct",
    )
    stats = np.zeros((tile_count, 4, 258), dtype=np.float32)
    for batch_start in range(0, tile_count, host_batch_tiles):
        batch_end = batch_start + host_batch_tiles
        batch_stats = runner_cache.run(
            key=("flowqkv_tiled_stats", 4, kv_tile, 256, host_batch_tiles, "direct"),
            mlir_module=module,
            backend_options=dict(
                verbose=False,
                omit_pingpong=True,
                output_format=DEFAULT_OUTPUT_FORMAT,
                instance_name="flowqkv_tiled_stats",
                target_device="npu2",
                runtime_loop_tiling_sizes=[1, 1],
            ),
            inputs=[
                q_tiles[batch_start:batch_end],
                k_tiles[batch_start:batch_end],
                v_tiles[batch_start:batch_end],
            ],
            output_shape=(host_batch_tiles, 4, 258),
            output_dtype=np.float32,
            timed_kernel_seconds=timed_kernel_seconds,
            power_meter=power_meter,
        )
        stats[batch_start:batch_end] = batch_stats.reshape(host_batch_tiles, 4, 258)

    actual = merge_tiled_stats(stats).reshape(1024).astype(bfloat16)
    expected = attention_reference(q_single, k_full, v_full).reshape(1024).astype(bfloat16)
    return actual, expected


def _run_attention_stage(
    *,
    mode: str,
    layer_index: int,
    q,
    k,
    v,
    single_token_object_file: Path,
    tiled_stats_object_file: Path,
    runner_cache: _ReusableElfRunnerCache,
    prompt_context_length: int,
    tiled_attention_kv_tile: int,
    tiled_attention_host_batch_tiles: int,
    attention_cache_mode: str,
    timed_kernel_seconds: list[float] | None = None,
    power_meter=None,
):
    if mode == "single-token":
        return _run_single_token_attention_stage(
            q=q,
            k=k,
            v=v,
            object_file=single_token_object_file,
            runner_cache=runner_cache,
            timed_kernel_seconds=timed_kernel_seconds,
            power_meter=power_meter,
        )
    if mode == "tiled-stats-1k":
        return _run_tiled_stats_attention_stage(
            layer_index=layer_index,
            q=q,
            k=k,
            v=v,
            object_file=tiled_stats_object_file,
            runner_cache=runner_cache,
            prompt_context_length=prompt_context_length,
            kv_tile=tiled_attention_kv_tile,
            host_batch_tiles=tiled_attention_host_batch_tiles,
            attention_cache_mode=attention_cache_mode,
            timed_kernel_seconds=timed_kernel_seconds,
            power_meter=power_meter,
        )
    raise RuntimeError(f"unsupported attention mode: {mode}")



def _preload_static_projection_bo_sets(
    projection_plans: dict[int, dict[str, _PackedProjectionPlan]],
    runner_cache: _ReusableElfRunnerCache,
    *,
    families: tuple[str, ...] | None = None,
) -> int:
    import numpy as np
    from ml_dtypes import bfloat16

    count = 0
    selected_families = set(families) if families is not None else None
    zero_activation = np.zeros((1, 256), dtype=bfloat16)
    for layer_plans in projection_plans.values():
        for family, plan in layer_plans.items():
            if selected_families is not None and family not in selected_families:
                continue
            output_shape = _projection_output_shape(plan)
            for col_block in range(plan.col_blocks):
                runner_cache.prepare(
                    key=("fused_dqp_accum_block_opt", int(plan.row_blocks)),
                    mlir_module=plan.mlir_module,
                    backend_options=_projection_backend_options(),
                    inputs=[_packed_l3_for_col_block(plan, col_block), zero_activation],
                    output_shape=output_shape,
                    output_dtype=bfloat16,
                    bo_set_key=_projection_bo_set_key(plan, col_block),
                    static_input_keys=[_projection_static_input_key(plan, col_block), None],
                )
                count += 1
    return count


def _run_stitched_ingress_stage(
    *,
    layer_index: int,
    x_input,
    norm_plan: _NormPlan,
    projection_plans: dict[str, _PackedProjectionPlan],
    runner_cache: _ReusableElfRunnerCache,
    timed_kernel_seconds: list[float] | None,
    power_meter,
    fused_dqp_object_file: Path,
    rope_object_file: Path,
    check_references: bool,
) -> tuple[dict[str, Any], dict[str, Any], list[ProjectionEvidence], list[str], float | None]:
    import numpy as np
    from common import fused_dqp_paper_reference
    from ml_dtypes import bfloat16

    dataflow_dir = _dataflow_dir()
    if str(dataflow_dir) not in sys.path:
        sys.path.insert(0, str(dataflow_dir))
    from rope_halfsplit import rope_halfsplit_reference

    arrays = _stitched_ingress_arrays(
        x_input=x_input,
        norm_plan=norm_plan,
        projection_plans=projection_plans,
    )
    runner = _stitched_ingress_runner(
        runner_cache,
        fused_dqp_object_file=fused_dqp_object_file,
        rope_object_file=rope_object_file,
    )
    actual = runner.run(
        arrays=arrays,
        readback=_stitched_ingress_readback(bfloat16),
        timed_kernel_seconds=timed_kernel_seconds,
        power_meter=power_meter,
        bo_set_key=_stitched_ingress_bo_set_key(layer_index),
        static_input_keys=_stitched_ingress_static_input_keys(
            layer_index=layer_index,
            norm_plan=norm_plan,
            projection_plans=projection_plans,
        ),
        bo_aliases=STITCHED_INGRESS_BO_ALIASES,
    )
    if not check_references:
        projection_evidence = [
            ProjectionEvidence(
                family=family,
                tensor_key=projection_plans[family].tensor_key,
                shape=projection_plans[family].shape,
                padded_shape=projection_plans[family].padded_shape,
                row_blocks=projection_plans[family].row_blocks,
                col_blocks=projection_plans[family].col_blocks,
                projection_correlation=None,
                dense_projection_correlation=None,
            )
            for family in ("q_proj", "k_proj", "v_proj")
        ]
        return actual, dict(actual), projection_evidence, [], None

    activation_expected = _rms_host(x_input.reshape(1, 1152), norm_plan.weight).reshape(-1)
    activation = np.zeros((5, 256), dtype=bfloat16)
    activation.reshape(-1)[:1152] = activation_expected
    q_expected = fused_dqp_paper_reference(
        projection_plans["q_proj"].packed,
        projection_plans["q_proj"].scale,
        projection_plans["q_proj"].min_offset,
        activation,
        32,
        256,
    ).reshape(32, 32)
    k_expected = fused_dqp_paper_reference(
        projection_plans["k_proj"].packed,
        projection_plans["k_proj"].scale,
        projection_plans["k_proj"].min_offset,
        activation,
        32,
        256,
    ).reshape(8, 32)
    v_expected = fused_dqp_paper_reference(
        projection_plans["v_proj"].packed,
        projection_plans["v_proj"].scale,
        projection_plans["v_proj"].min_offset,
        activation,
        32,
        256,
    ).reshape(8, 32)
    q_norm_expected = _rms_host(q_expected.reshape(4, 256), norm_plan.norm_weights["q_norm"])
    k_norm_expected = _rms_host(k_expected.reshape(1, 256), norm_plan.norm_weights["k_norm"])
    q_rope_expected = rope_halfsplit_reference(q_norm_expected, arrays[14].reshape(4, 256))
    k_rope_expected = rope_halfsplit_reference(k_norm_expected, arrays[15].reshape(1, 256))
    expected = {
        "input_norm": activation_expected.reshape(1, 1152),
        "activation": activation,
        "q_proj": q_expected,
        "k_proj": k_expected,
        "v_proj": v_expected,
        "q_norm": q_norm_expected,
        "k_norm": k_norm_expected,
        "q_rope": q_rope_expected,
        "k_rope": k_rope_expected,
    }
    blockers: list[str] = []
    for name, expected_value in expected.items():
        corr = _correlation(actual[name], expected_value)
        if corr < DEFAULT_THRESHOLD:
            blockers.append(f"L{layer_index}:stitched-ingress-{name}-correlation-low")
    projection_evidence = []
    for family, name in (("q_proj", "q_proj"), ("k_proj", "k_proj"), ("v_proj", "v_proj")):
        plan = projection_plans[family]
        corr = _correlation(actual[name], expected[name])
        projection_evidence.append(
            ProjectionEvidence(
                family=plan.family,
                tensor_key=plan.tensor_key,
                shape=plan.shape,
                padded_shape=plan.padded_shape,
                row_blocks=plan.row_blocks,
                col_blocks=plan.col_blocks,
                projection_correlation=corr,
                dense_projection_correlation=None,
            )
        )
    rms_corr = _correlation(actual["input_norm"], expected["input_norm"])
    return actual, expected, projection_evidence, blockers, rms_corr


def _run_packed_projection(
    plan: _PackedProjectionPlan,
    activation,
    runner_cache: _ReusableElfRunnerCache,
    timed_kernel_seconds: list[float] | None,
    power_meter,
    static_projection_argument_mode: str,
    check_references: bool = True,
):
    import numpy as np
    from ml_dtypes import bfloat16
    from common import fused_dqp_paper_reference

    out_dim, in_dim = plan.shape
    activation_padded = np.zeros((plan.col_blocks * 256,), dtype=bfloat16)
    activation_padded[:in_dim] = activation.reshape(-1).astype(bfloat16)
    activation_blocks = activation_padded.reshape(plan.col_blocks, 256)
    output_shape = _projection_output_shape(plan)
    expected = None
    if check_references:
        expected = fused_dqp_paper_reference(plan.packed, plan.scale, plan.min_offset, activation_blocks, 32, 256)
        if tuple(expected.shape) != output_shape:
            raise RuntimeError(f"projection output shape mismatch: expected {output_shape}, got {expected.shape}")
    accum = np.zeros(output_shape, dtype=np.float32)
    for col_block in range(plan.col_blocks):
        cb_slice = slice(col_block, col_block + 1)
        static_mode = static_projection_argument_mode == "preloaded-runner-bo-set"
        static_l3 = _packed_l3_static_placeholder(plan) if static_mode else _packed_l3_for_col_block(plan, col_block)
        partial = runner_cache.run(
            key=("fused_dqp_accum_block_opt", int(plan.row_blocks)),
            mlir_module=plan.mlir_module,
            backend_options=_projection_backend_options(),
            inputs=[static_l3, activation_blocks[cb_slice, :]],
            output_shape=output_shape,
            output_dtype=bfloat16,
            timed_kernel_seconds=timed_kernel_seconds,
            power_meter=power_meter,
            bo_set_key=_projection_bo_set_key(plan, col_block) if static_mode else None,
            static_input_keys=[_projection_static_input_key(plan, col_block), None] if static_mode else None,
        )
        accum += partial.astype(np.float32)
    actual = accum.astype(bfloat16).reshape(-1)[:out_dim]
    expected_vec = actual if expected is None else expected.reshape(-1)[:out_dim].astype(bfloat16)
    evidence = ProjectionEvidence(
        family=plan.family,
        tensor_key=plan.tensor_key,
        shape=plan.shape,
        padded_shape=plan.padded_shape,
        row_blocks=plan.row_blocks,
        col_blocks=plan.col_blocks,
        projection_correlation=(None if expected is None else _correlation(actual, expected_vec)),
        dense_projection_correlation=None,
    )
    return actual, expected_vec, evidence


def _run_one_layer(
    *,
    layer_index: int,
    x_input,
    norm_plan: _NormPlan,
    projection_plans: dict[str, _PackedProjectionPlan],
    runner_cache: _ReusableElfRunnerCache,
    timed_kernel_seconds: list[float] | None,
    power_meter,
    static_projection_argument_mode: str,
    ingress_mode: str,
    fused_dqp_object_file: Path,
    rope_object_file: Path,
    flowqkv_object_file: Path,
    tiled_stats_object_file: Path,
    attention_mode: str,
    attention_cache_mode: str,
    prompt_context_length: int,
    tiled_attention_kv_tile: int,
    tiled_attention_host_batch_tiles: int,
    check_references: bool = True,
) -> tuple[Any, LayerLoopEvidence, list[str]]:
    from ml_dtypes import bfloat16
    import numpy as np

    blockers: list[str] = []
    start_count = len(timed_kernel_seconds) if timed_kernel_seconds is not None else 0
    start_seconds = sum(timed_kernel_seconds) if timed_kernel_seconds is not None else 0.0
    actual: dict[str, Any]
    expected: dict[str, Any]
    projection_evidence: list[ProjectionEvidence]
    if ingress_mode == "stitched":
        actual, expected, projection_evidence, ingress_blockers, rms_corr = _run_stitched_ingress_stage(
            layer_index=layer_index,
            x_input=x_input,
            norm_plan=norm_plan,
            projection_plans=projection_plans,
            runner_cache=runner_cache,
            timed_kernel_seconds=timed_kernel_seconds,
            power_meter=power_meter,
            fused_dqp_object_file=fused_dqp_object_file,
            rope_object_file=rope_object_file,
            check_references=check_references,
        )
        blockers.extend(ingress_blockers)
    elif ingress_mode == "staged":
        from weighted_rms_norm import build_module as build_rms_module
        from weighted_rms_norm import rms_norm_reference

        norm_expected = rms_norm_reference(x_input, norm_plan.weight) if check_references else x_input
        rms_module = build_rms_module(1, 1152, bfloat16, 16, herd_x=1)
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
            inputs=[x_input, norm_plan.weight],
            output_shape=norm_expected.shape,
            output_dtype=bfloat16,
            timed_kernel_seconds=timed_kernel_seconds,
            power_meter=power_meter,
        )
        rms_corr = _correlation(norm_actual, norm_expected) if check_references else None
        if check_references and rms_corr < DEFAULT_THRESHOLD:
            blockers.append(f"L{layer_index}:rmsnorm-correlation-low")

        actual = {"input_norm": norm_actual.reshape(-1)}
        expected = {"input_norm": norm_expected.reshape(-1) if check_references else actual["input_norm"]}
        projection_evidence = []
        for family in ("q_proj", "k_proj", "v_proj"):
            actual_vec, expected_vec, evidence = _run_packed_projection(
                projection_plans[family],
                actual["input_norm"],
                runner_cache,
                timed_kernel_seconds,
                power_meter,
                static_projection_argument_mode,
                check_references=check_references,
            )
            projection_evidence.append(evidence)
            actual[family] = actual_vec
            expected[family] = expected_vec
            if check_references and (evidence.projection_correlation is None or evidence.projection_correlation < DEFAULT_THRESHOLD):
                blockers.append(f"L{layer_index}:{family}-correlation-low")
    else:
        raise RuntimeError(f"unsupported ingress mode: {ingress_mode}")

    q_actual = actual["q_proj"].reshape(4, 256)
    k_actual = actual["k_proj"].reshape(1, 256)
    v_actual = actual["v_proj"].reshape(1, 256)
    q_expected = expected["q_proj"].reshape(4, 256)
    k_expected = expected["k_proj"].reshape(1, 256)
    v_expected = expected["v_proj"].reshape(1, 256)

    if ingress_mode == "stitched":
        qn_actual = actual["q_norm"].reshape(4, 256)
        kn_actual = actual["k_norm"].reshape(1, 256)
        qn_expected = expected["q_norm"].reshape(4, 256)
        kn_expected = expected["k_norm"].reshape(1, 256)
        q_rope_actual = actual["q_rope"].reshape(4, 256)
        k_rope_actual = actual["k_rope"].reshape(1, 256)
        q_rope_expected = expected["q_rope"].reshape(4, 256)
        k_rope_expected = expected["k_rope"].reshape(1, 256)
    else:
        qn_actual, qn_reference = _run_rms_stage(
            name="q_norm",
            x=q_actual,
            weight=norm_plan.norm_weights["q_norm"],
            runner_cache=runner_cache,
            timed_kernel_seconds=timed_kernel_seconds,
            power_meter=power_meter,
        )
        kn_actual, kn_reference = _run_rms_stage(
            name="k_norm",
            x=k_actual,
            weight=norm_plan.norm_weights["k_norm"],
            runner_cache=runner_cache,
            timed_kernel_seconds=timed_kernel_seconds,
            power_meter=power_meter,
        )
        qn_expected = _rms_host(q_expected, norm_plan.norm_weights["q_norm"]) if check_references else qn_actual
        kn_expected = _rms_host(k_expected, norm_plan.norm_weights["k_norm"]) if check_references else kn_actual
        if check_references and (_correlation(qn_actual, qn_reference) < DEFAULT_THRESHOLD or _correlation(qn_actual, qn_expected) < DEFAULT_THRESHOLD):
            blockers.append(f"L{layer_index}:q-norm-correlation-low")
        if check_references and (_correlation(kn_actual, kn_reference) < DEFAULT_THRESHOLD or _correlation(kn_actual, kn_expected) < DEFAULT_THRESHOLD):
            blockers.append(f"L{layer_index}:k-norm-correlation-low")

        q_rope_actual, q_rope_expected = _run_rope_stage(
            name="q",
            x=qn_actual,
            object_file=rope_object_file,
            runner_cache=runner_cache,
            timed_kernel_seconds=timed_kernel_seconds,
            power_meter=power_meter,
        )
        k_rope_actual, k_rope_expected = _run_rope_stage(
            name="k",
            x=kn_actual,
            object_file=rope_object_file,
            runner_cache=runner_cache,
            timed_kernel_seconds=timed_kernel_seconds,
            power_meter=power_meter,
        )
        if check_references and _correlation(q_rope_actual, q_rope_expected) < DEFAULT_THRESHOLD:
            blockers.append(f"L{layer_index}:rope-q-correlation-low")
        if check_references and _correlation(k_rope_actual, k_rope_expected) < DEFAULT_THRESHOLD:
            blockers.append(f"L{layer_index}:rope-k-correlation-low")

    attention_actual, attention_reference = _run_attention_stage(
        mode=attention_mode,
        layer_index=layer_index,
        q=q_rope_actual,
        k=k_rope_actual,
        v=v_actual,
        single_token_object_file=flowqkv_object_file,
        tiled_stats_object_file=tiled_stats_object_file,
        runner_cache=runner_cache,
        prompt_context_length=prompt_context_length,
        tiled_attention_kv_tile=tiled_attention_kv_tile,
        tiled_attention_host_batch_tiles=tiled_attention_host_batch_tiles,
        attention_cache_mode=attention_cache_mode,
        timed_kernel_seconds=timed_kernel_seconds,
        power_meter=power_meter,
    )
    attention_expected = attention_reference if check_references else attention_actual
    if check_references and (_correlation(attention_actual, attention_reference) < DEFAULT_THRESHOLD or _correlation(attention_actual, attention_expected) < DEFAULT_THRESHOLD):
        blockers.append(f"L{layer_index}:{attention_mode}-attention-correlation-low")

    o_actual, o_expected, evidence = _run_packed_projection(
        projection_plans["o_proj"], attention_actual, runner_cache, timed_kernel_seconds, power_meter, static_projection_argument_mode, check_references=check_references
    )
    projection_evidence.append(evidence)
    if check_references and (evidence.projection_correlation is None or evidence.projection_correlation < DEFAULT_THRESHOLD):
        blockers.append(f"L{layer_index}:o_proj-correlation-low")

    post_attention_actual, post_attention_reference = _run_rms_stage(
        name="post_attention_norm",
        x=o_actual.reshape(1, 1152),
        weight=norm_plan.norm_weights["post_attention_layernorm"],
        runner_cache=runner_cache,
        timed_kernel_seconds=timed_kernel_seconds,
        power_meter=power_meter,
    )
    post_attention_expected = (
        _rms_host(o_expected.reshape(1, 1152), norm_plan.norm_weights["post_attention_layernorm"])
        if check_references
        else post_attention_actual
    )
    if check_references and (_correlation(post_attention_actual, post_attention_reference) < DEFAULT_THRESHOLD or _correlation(post_attention_actual, post_attention_expected) < DEFAULT_THRESHOLD):
        blockers.append(f"L{layer_index}:post-attention-norm-correlation-low")

    residual_actual, residual_reference = _run_residual_stage(
        name="attention",
        lhs=x_input.reshape(-1),
        rhs=post_attention_actual.reshape(-1),
        runner_cache=runner_cache,
        timed_kernel_seconds=timed_kernel_seconds,
        power_meter=power_meter,
    )
    residual_actual = residual_actual.reshape(1, 1152)
    residual_expected = (
        (x_input.astype(np.float32) + post_attention_expected.astype(np.float32)).astype(bfloat16)
        if check_references
        else residual_actual
    )
    if check_references and (_correlation(residual_actual, residual_reference.reshape(1, 1152)) < DEFAULT_THRESHOLD or _correlation(residual_actual, residual_expected) < DEFAULT_THRESHOLD):
        blockers.append(f"L{layer_index}:attention-residual-correlation-low")

    pre_ff_actual, pre_ff_reference = _run_rms_stage(
        name="pre_feedforward_norm",
        x=residual_actual,
        weight=norm_plan.norm_weights["pre_feedforward_layernorm"],
        runner_cache=runner_cache,
        timed_kernel_seconds=timed_kernel_seconds,
        power_meter=power_meter,
    )
    pre_ff_expected = (
        _rms_host(residual_expected, norm_plan.norm_weights["pre_feedforward_layernorm"])
        if check_references
        else pre_ff_actual
    )
    if check_references and (_correlation(pre_ff_actual, pre_ff_reference) < DEFAULT_THRESHOLD or _correlation(pre_ff_actual, pre_ff_expected) < DEFAULT_THRESHOLD):
        blockers.append(f"L{layer_index}:pre-feedforward-norm-correlation-low")

    gate_actual, gate_expected, gate_evidence = _run_packed_projection(
        projection_plans["gate_proj"], pre_ff_actual.reshape(-1), runner_cache, timed_kernel_seconds, power_meter, static_projection_argument_mode, check_references=check_references
    )
    up_actual, up_expected, up_evidence = _run_packed_projection(
        projection_plans["up_proj"], pre_ff_actual.reshape(-1), runner_cache, timed_kernel_seconds, power_meter, static_projection_argument_mode, check_references=check_references
    )
    for family, evidence in (("gate_proj", gate_evidence), ("up_proj", up_evidence)):
        projection_evidence.append(evidence)
        if check_references and (evidence.projection_correlation is None or evidence.projection_correlation < DEFAULT_THRESHOLD):
            blockers.append(f"L{layer_index}:{family}-correlation-low")

    mlp_actual, mlp_reference = _run_geglu_stage(
        name="mlp_activation",
        gate=gate_actual,
        up=up_actual,
        runner_cache=runner_cache,
        timed_kernel_seconds=timed_kernel_seconds,
        power_meter=power_meter,
    )
    mlp_expected = _geglu(gate_expected, up_expected) if check_references else mlp_actual
    if check_references and (_correlation(mlp_actual, mlp_reference) < DEFAULT_THRESHOLD or _correlation(mlp_actual, mlp_expected) < DEFAULT_THRESHOLD):
        blockers.append(f"L{layer_index}:mlp-activation-correlation-low")

    down_actual, down_expected, down_evidence = _run_packed_projection(
        projection_plans["down_proj"], mlp_actual, runner_cache, timed_kernel_seconds, power_meter, static_projection_argument_mode, check_references=check_references
    )
    projection_evidence.append(down_evidence)
    if check_references and (down_evidence.projection_correlation is None or down_evidence.projection_correlation < DEFAULT_THRESHOLD):
        blockers.append(f"L{layer_index}:down_proj-correlation-low")

    post_ff_actual, post_ff_reference = _run_rms_stage(
        name="post_feedforward_norm",
        x=down_actual.reshape(1, 1152),
        weight=norm_plan.norm_weights["post_feedforward_layernorm"],
        runner_cache=runner_cache,
        timed_kernel_seconds=timed_kernel_seconds,
        power_meter=power_meter,
    )
    post_ff_expected = (
        _rms_host(down_expected.reshape(1, 1152), norm_plan.norm_weights["post_feedforward_layernorm"])
        if check_references
        else post_ff_actual
    )
    if check_references and (_correlation(post_ff_actual, post_ff_reference) < DEFAULT_THRESHOLD or _correlation(post_ff_actual, post_ff_expected) < DEFAULT_THRESHOLD):
        blockers.append(f"L{layer_index}:post-feedforward-norm-correlation-low")

    output_actual_arr, output_reference = _run_residual_stage(
        name="mlp",
        lhs=residual_actual.reshape(-1),
        rhs=post_ff_actual.reshape(-1),
        runner_cache=runner_cache,
        timed_kernel_seconds=timed_kernel_seconds,
        power_meter=power_meter,
    )
    output_actual = output_actual_arr.reshape(1, 1152)
    output_expected = (
        (residual_expected.astype(np.float32) + post_ff_expected.astype(np.float32)).astype(bfloat16)
        if check_references
        else output_actual
    )
    if check_references and _correlation(output_actual, output_reference.reshape(1, 1152)) < DEFAULT_THRESHOLD:
        blockers.append(f"L{layer_index}:mlp-residual-correlation-low")
    final_corr = _correlation(output_actual.reshape(-1), output_expected.reshape(-1)) if check_references else None
    if check_references and final_corr < DEFAULT_THRESHOLD:
        blockers.append(f"L{layer_index}:final-output-correlation-low")
    end_count = len(timed_kernel_seconds) if timed_kernel_seconds is not None else start_count
    end_seconds = sum(timed_kernel_seconds) if timed_kernel_seconds is not None else start_seconds
    evidence = LayerLoopEvidence(
        layer_index=layer_index,
        norm_tensor_key=norm_plan.tensor_key,
        static_norm_tensor_offset_bytes=norm_plan.offset_bytes,
        static_norm_argument_bytes=norm_plan.argument_bytes,
        rms_correlation=rms_corr,
        final_output_correlation=final_corr,
        projection_evidence=tuple(projection_evidence),
        timed_kernel_count=end_count - start_count,
        timed_kernel_seconds=end_seconds - start_seconds,
    )
    return output_actual.reshape(1, 1152), evidence, blockers


def _run_hardware_sequence(args: argparse.Namespace) -> Gemma3DecodeLoopProbeResult:
    _activate_probe_env()
    import numpy as np
    from ml_dtypes import bfloat16

    repo = _repo_root()
    git_commit, dirty = _git_info(repo)
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    blockers: list[str] = []
    layer_evidence: list[LayerLoopEvidence] = []
    timed_kernel_samples: list[float] = []
    runner_cache: _ReusableElfRunnerCache | None = None
    segment_power = _SegmentedRAPLPowerMeter(
        sample=bool(args.power_sample),
        run_id="gemma3_1b_decode_loop_probe_kernel_segments",
    )
    start = time.perf_counter()
    loop_elapsed: float | None = None
    power_snapshot: dict[str, object] | None = None
    static_projection_argument_mode = "preloaded-runner-bo-set"
    static_projection_bo_set_count = 0
    static_ingress_bo_set_count = 0
    reference_check_layer_count = 0
    reference_check_seconds: float | None = None
    returncode = 1
    try:
        if args.model_variant != DEFAULT_MODEL:
            raise RuntimeError("decode loop probe currently supports gemma3-1b only")
        if args.layers <= 0:
            raise RuntimeError("--layers must be positive")
        if args.ingress_mode == "stitched" and args.no_reuse_elf:
            raise RuntimeError("--ingress-mode=stitched requires reusable ELF runners")
        if args.attention_mode == "tiled-stats-1k" and args.prompt_context_length != 1024:
            raise RuntimeError("tiled-stats-1k attention mode currently requires --prompt-context-length=1024")
        weights_dir = _resolve_weights_dir(args.model_variant, args.weights_dir)
        norm_plans, projection_plans = _prepare_layer_plans(args.model_variant, weights_dir, args.layers)
        peano_build_dir = Path(__file__).with_name("build_peano")
        fused_dqp_object_file = peano_build_dir / "fused_dqp.o"
        rope_object_file = peano_build_dir / "rope_halfsplit.o"
        flowqkv_object_file = peano_build_dir / "flowqkv_single_token_q4_kv1_d256.o"
        tiled_stats_object_file = peano_build_dir / "flowqkv_tiled_stats_q4_kv32_d256.o"
        runner_cache = _ReusableElfRunnerCache(enabled=not args.no_reuse_elf)
        runner_cache.__enter__()
        static_projection_argument_mode = (
            "per-launch-write"
            if args.dynamic_static_weight_writes or args.no_reuse_elf
            else "preloaded-runner-bo-set"
        )
        static_projection_bo_set_count = 0
        static_ingress_bo_set_count = 0
        if static_projection_argument_mode == "preloaded-runner-bo-set":
            projection_preload_families = (
                None
                if args.ingress_mode == "staged"
                else ("o_proj", "gate_proj", "up_proj", "down_proj")
            )
            static_projection_bo_set_count = _preload_static_projection_bo_sets(
                projection_plans,
                runner_cache,
                families=projection_preload_families,
            )
        if args.ingress_mode == "stitched":
            static_ingress_bo_set_count = _preload_static_stitched_ingress_bo_sets(
                norm_plans=norm_plans,
                projection_plans=projection_plans,
                runner_cache=runner_cache,
                fused_dqp_object_file=fused_dqp_object_file,
                rope_object_file=rope_object_file,
            )
        rng = np.random.default_rng(0)
        warmup_input = rng.uniform(-0.5, 0.5, size=(1, 1152)).astype(bfloat16)
        for layer_index in range(min(args.warmup_layers, args.layers)):
            warmup_input, _evidence, warmup_blockers = _run_one_layer(
                layer_index=layer_index,
                x_input=warmup_input,
                norm_plan=norm_plans[layer_index],
                projection_plans=projection_plans[layer_index],
                runner_cache=runner_cache,
                timed_kernel_seconds=None,
                power_meter=None,
                static_projection_argument_mode=static_projection_argument_mode,
                ingress_mode=args.ingress_mode,
                fused_dqp_object_file=fused_dqp_object_file,
                rope_object_file=rope_object_file,
                flowqkv_object_file=flowqkv_object_file,
                tiled_stats_object_file=tiled_stats_object_file,
                attention_mode=args.attention_mode,
                attention_cache_mode=args.attention_cache_mode,
                prompt_context_length=args.prompt_context_length,
                tiled_attention_kv_tile=args.tiled_attention_kv_tile,
                tiled_attention_host_batch_tiles=args.tiled_attention_host_batch_tiles,
            )
            if warmup_blockers:
                blockers.extend(f"warmup:{item}" for item in warmup_blockers)
        reference_input = rng.uniform(-0.5, 0.5, size=(1, 1152)).astype(bfloat16)
        reference_start = time.perf_counter()
        for layer_index in range(args.layers):
            reference_input, evidence, reference_blockers = _run_one_layer(
                layer_index=layer_index,
                x_input=reference_input,
                norm_plan=norm_plans[layer_index],
                projection_plans=projection_plans[layer_index],
                runner_cache=runner_cache,
                timed_kernel_seconds=None,
                power_meter=None,
                static_projection_argument_mode=static_projection_argument_mode,
                ingress_mode=args.ingress_mode,
                fused_dqp_object_file=fused_dqp_object_file,
                rope_object_file=rope_object_file,
                flowqkv_object_file=flowqkv_object_file,
                tiled_stats_object_file=tiled_stats_object_file,
                attention_mode=args.attention_mode,
                attention_cache_mode=args.attention_cache_mode,
                prompt_context_length=args.prompt_context_length,
                tiled_attention_kv_tile=args.tiled_attention_kv_tile,
                tiled_attention_host_batch_tiles=args.tiled_attention_host_batch_tiles,
                check_references=True,
            )
            layer_evidence.append(evidence)
            reference_check_layer_count += 1
            blockers.extend(reference_blockers)
        reference_check_seconds = time.perf_counter() - reference_start
        x_input = rng.uniform(-0.5, 0.5, size=(1, 1152)).astype(bfloat16)
        power_window = begin_power_window(
            sample=bool(args.power_sample),
            run_id="gemma3_1b_decode_loop_probe",
            target_backend="npu",
        )
        loop_start = time.perf_counter()
        for token_index in range(args.decode_tokens):
            for layer_index in range(args.layers):
                x_input, _timing_evidence, layer_blockers = _run_one_layer(
                    layer_index=layer_index,
                    x_input=x_input,
                    norm_plan=norm_plans[layer_index],
                    projection_plans=projection_plans[layer_index],
                    runner_cache=runner_cache,
                    timed_kernel_seconds=timed_kernel_samples,
                    power_meter=segment_power,
                    static_projection_argument_mode=static_projection_argument_mode,
                    ingress_mode=args.ingress_mode,
                    fused_dqp_object_file=fused_dqp_object_file,
                    rope_object_file=rope_object_file,
                    flowqkv_object_file=flowqkv_object_file,
                    tiled_stats_object_file=tiled_stats_object_file,
                    attention_mode=args.attention_mode,
                    attention_cache_mode=args.attention_cache_mode,
                    prompt_context_length=args.prompt_context_length,
                    tiled_attention_kv_tile=args.tiled_attention_kv_tile,
                    tiled_attention_host_batch_tiles=args.tiled_attention_host_batch_tiles,
                    check_references=False,
                )
                blockers.extend(layer_blockers)
            stdout_lines.append(f"decode token {token_index}: final checksum={float(x_input.astype(np.float32).sum()):.6f}")
        loop_elapsed = time.perf_counter() - loop_start
        power_snapshot = finish_power_window(power_window, elapsed_seconds=loop_elapsed).to_json_dict()
        returncode = 0 if not blockers else 1
    except Exception as exc:
        blockers.append(f"decode-loop-probe-failed:{exc}")
        stderr_lines.append(str(exc))
    finally:
        if runner_cache is not None:
            runner_cache.__exit__(None, None, None)

    elapsed = time.perf_counter() - start
    timed_total = sum(timed_kernel_samples) if timed_kernel_samples else None
    timed_mean = (timed_total / len(timed_kernel_samples)) if timed_total is not None and timed_kernel_samples else None
    loop_tps = (args.decode_tokens / loop_elapsed) if loop_elapsed and loop_elapsed > 0.0 else None
    kernel_tps = (args.decode_tokens / timed_total) if timed_total and timed_total > 0.0 else None
    status = "DECODE_LOOP_DIAGNOSTIC_PASS" if not blockers else "DECODE_LOOP_DIAGNOSTIC_BLOCKED"
    return Gemma3DecodeLoopProbeResult(
        schema_version=2,
        model_variant=args.model_variant,
        status=status,
        sequence_kind=DEFAULT_SEQUENCE_KIND,
        phase=DEFAULT_PHASE,
        layer_count=args.layers,
        decode_tokens=args.decode_tokens,
        prompt_context_length=args.prompt_context_length,
        output_format=DEFAULT_OUTPUT_FORMAT,
        runner_reuse_mode=("reused-elf-persistent-bo" if not args.no_reuse_elf else "per-launch-compile-load"),
        ingress_mode=args.ingress_mode,
        norm_argument_mode="selected-vector",
        static_projection_argument_mode=static_projection_argument_mode,
        static_projection_bo_set_count=static_projection_bo_set_count,
        static_ingress_bo_set_count=static_ingress_bo_set_count,
        input_shape=(1, 1152),
        output_shape=(1, 1152),
        input_distribution=DEFAULT_INPUT_DISTRIBUTION,
        reference_check_mode="all-layers-before-timing",
        reference_check_layer_count=reference_check_layer_count,
        reference_check_seconds=reference_check_seconds,
        attention_mode=args.attention_mode,
        attention_cache_contract=(
            (
                "synthetic-prefill-kv-cache"
                if args.attention_cache_mode == "synthetic-prefill"
                else "synthetic-repeated-current-token-kv-cache"
            )
            if args.attention_mode == "tiled-stats-1k"
            else "single-current-token-kv"
        ),
        attention_kv_tile=(args.tiled_attention_kv_tile if args.attention_mode == "tiled-stats-1k" else None),
        attention_host_batch_tiles=(
            args.tiled_attention_host_batch_tiles if args.attention_mode == "tiled-stats-1k" else None
        ),
        attention_host_batch_count=(
            (args.prompt_context_length // args.tiled_attention_kv_tile) // args.tiled_attention_host_batch_tiles
            if args.attention_mode == "tiled-stats-1k"
            else None
        ),
        attention_host_reduction=(args.attention_mode == "tiled-stats-1k"),
        host_fallbacks=(),
        timed_kernel_count=len(timed_kernel_samples),
        timed_kernel_seconds=timed_total,
        timed_kernel_mean_seconds=timed_mean,
        measured_loop_seconds=loop_elapsed,
        diagnostic_decode_tps_loop_wall=loop_tps,
        diagnostic_decode_tps_kernel_only=kernel_tps,
        paper_decode_tps_1k=PAPER_DECODE_TPS_1K,
        loop_wall_delta_pct_vs_paper_decode_tps_1k=_delta_pct(loop_tps),
        kernel_only_delta_pct_vs_paper_decode_tps_1k=_delta_pct(kernel_tps),
        timing_window="post-warmup-loop-wall-and-segmented-run-start-wait2",
        timing_notes=(
            "warmup layers compile/load ELF runners and allocate runner-owned BOs before measured loop timing",
            "preloaded-runner-bo-set mode allocates BO sets and writes packed projection static inputs before measured loop timing",
            (
                "stitched ingress mode preloads one aliased RMSNorm/QKV/QK-norm/RoPE BO set per layer before measured loop timing"
                if args.ingress_mode == "stitched"
                else "staged ingress mode runs RMSNorm, q/k/v projections, Q/K norm, and RoPE as separate launches"
            ),
            "reference_check_seconds validates all layers before measured loop timing and is excluded from measured_loop_seconds",
            "measured_loop_seconds starts after warmup/reference checks and includes current implementation dynamic BO writes and output sync/readback",
            "timed_kernel_seconds sums only pyxrt run.start()/wait2() calls and excludes compile, ELF load, BO allocation, BO writes, sync/readback, and CPU reference/correlation checks",
            (
                "attention is the staged single-token FlowQKV NPU path, not paper 1k KV-cache attention"
                if args.attention_mode == "single-token"
                else (
                    "attention is host-batched 1k tiled-stat FlowQKV NPU launches with host-side softmax-stat reduction over a synthetic prefill-shaped KV cache"
                    if args.attention_cache_mode == "synthetic-prefill"
                    else "attention is host-batched 1k tiled-stat FlowQKV NPU launches with host-side softmax-stat reduction over a synthetic repeated current-token KV cache"
                )
            ),
            "logits and sampling are not wired, so this is a diagnostic decode-loop throughput, not an official paper decode TPS cell",
        ),
        power_snapshot=power_snapshot,
        segmented_kernel_power_snapshot=segment_power.snapshot(),
        layer_evidence=tuple(layer_evidence),
        remaining_paper_gaps=(
            "prefill-kv-cache-not-constructed",
            *(
                ("paper-1k-kv-attention-not-wired",)
                if args.attention_mode == "single-token"
                else (
                    (
                        "prefill-produced-kv-cache-not-wired",
                        "paper-1k-kv-attention-npu-reduction-not-wired",
                    )
                    if args.attention_cache_mode == "synthetic-prefill"
                    else ("paper-1k-kv-attention-production-cache-and-npu-reduction-not-wired",)
                )
            ),
            "logits-sampling-not-wired",
            *(
                ("decode-layer-after-stitched-ingress-still-staged",)
                if args.ingress_mode == "stitched"
                else ()
            ),
            "production-contiguous-static-weight-bo-not-used-by-fused-dqp-route",
        ),
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
    evidence = LayerLoopEvidence(
        layer_index=0,
        norm_tensor_key="model.layers.0.input_layernorm.weight",
        static_norm_tensor_offset_bytes=0,
        static_norm_argument_bytes=2304,
        rms_correlation=0.999991,
        final_output_correlation=1.0,
        projection_evidence=(),
        timed_kernel_count=57,
        timed_kernel_seconds=0.15,
    )
    result = Gemma3DecodeLoopProbeResult(
        schema_version=2,
        model_variant=DEFAULT_MODEL,
        status="DECODE_LOOP_DIAGNOSTIC_PASS",
        sequence_kind=DEFAULT_SEQUENCE_KIND,
        phase=DEFAULT_PHASE,
        layer_count=26,
        decode_tokens=1,
        prompt_context_length=1024,
        output_format=DEFAULT_OUTPUT_FORMAT,
        runner_reuse_mode="reused-elf-persistent-bo",
        ingress_mode=DEFAULT_INGRESS_MODE,
        norm_argument_mode="selected-vector",
        static_projection_argument_mode="preloaded-runner-bo-set",
        static_projection_bo_set_count=1456,
        static_ingress_bo_set_count=0,
        input_shape=(1, 1152),
        output_shape=(1, 1152),
        input_distribution=DEFAULT_INPUT_DISTRIBUTION,
        reference_check_mode="all-layers-before-timing",
        reference_check_layer_count=26,
        reference_check_seconds=1.0,
        attention_mode=DEFAULT_ATTENTION_MODE,
        attention_cache_contract="single-current-token-kv",
        attention_kv_tile=None,
        attention_host_batch_tiles=None,
        attention_host_batch_count=None,
        attention_host_reduction=False,
        host_fallbacks=(),
        timed_kernel_count=1482,
        timed_kernel_seconds=3.9,
        timed_kernel_mean_seconds=3.9 / 1482.0,
        measured_loop_seconds=5.0,
        diagnostic_decode_tps_loop_wall=0.2,
        diagnostic_decode_tps_kernel_only=1.0 / 3.9,
        paper_decode_tps_1k=PAPER_DECODE_TPS_1K,
        loop_wall_delta_pct_vs_paper_decode_tps_1k=_delta_pct(0.2),
        kernel_only_delta_pct_vs_paper_decode_tps_1k=_delta_pct(1.0 / 3.9),
        timing_window="fixture",
        timing_notes=("fixture",),
        power_snapshot=None,
        segmented_kernel_power_snapshot=None,
        layer_evidence=(evidence,),
        remaining_paper_gaps=("paper-1k-kv-attention-not-wired",),
        command=("python3", "gemma3_decode_loop_probe.py", "--self-test"),
        returncode=0,
        elapsed_seconds=5.5,
        blockers=(),
        git_commit="fixture",
        dirty_worktree=False,
        stdout_tail=(),
        stderr_tail=(),
    )
    if result.status != "DECODE_LOOP_DIAGNOSTIC_PASS":
        raise AssertionError(result)
    print(result.format())
    print("GEMMA3_DECODE_LOOP_PROBE_SELF_TEST: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma3 staged decode loop probe")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--run-hardware", action="store_true")
    parser.add_argument("--model-variant", choices=sorted(MODEL_SPECS), default=DEFAULT_MODEL)
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--layers", type=int, default=26)
    parser.add_argument("--decode-tokens", type=int, default=1)
    parser.add_argument("--prompt-context-length", type=int, default=1024)
    parser.add_argument("--warmup-layers", type=int, default=1)
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--power-sample", action="store_true")
    parser.add_argument("--ingress-mode", choices=["staged", "stitched"], default=DEFAULT_INGRESS_MODE)
    parser.add_argument("--attention-mode", choices=["single-token", "tiled-stats-1k"], default=DEFAULT_ATTENTION_MODE)
    parser.add_argument("--attention-cache-mode", choices=["repeated-current-token", "synthetic-prefill"], default=DEFAULT_ATTENTION_CACHE_MODE)
    parser.add_argument("--tiled-attention-kv-tile", type=int, default=DEFAULT_TILED_ATTENTION_KV_TILE)
    parser.add_argument("--tiled-attention-host-batch-tiles", type=int, default=DEFAULT_TILED_ATTENTION_HOST_BATCH_TILES)
    parser.add_argument("--dynamic-static-weight-writes", action="store_true", help="diagnostic fallback: write packed projection static inputs inside each launch")
    parser.add_argument("--no-reuse-elf", action="store_true")
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
        args.result_json.write_text(json.dumps(result.to_json_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"GEMMA3_DECODE_LOOP_PROBE_JSON: {args.result_json}")
    return 0 if result.status == "DECODE_LOOP_DIAGNOSTIC_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
