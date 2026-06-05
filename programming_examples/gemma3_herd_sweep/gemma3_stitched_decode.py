#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gemma3 stitched-ELF decode track.

This file moves Gemma3 model bring-up away from one-operator diagnostic launches
and toward Llama32-style stitched ELF subgraphs. The implemented decode ingress
slice now stitches the full front half of a decode layer:

  RMSNorm -> Q/K/V projections -> Q/K Norm -> RoPE

The slice uses one padded RMSNorm launch, three full-column-block FusedDQP
projection launches, two projection-output view plus Q/K RMSNorm launches, and
two RoPE launches. It is compile-only evidence until the stitched ELF is run on
hardware against real layer tensors.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time

from gemma3_stitching import StitchSpec, stitch_module_text


DEFAULT_MODEL = "gemma3-1b"
DEFAULT_FUNCTION_NAME = "gemma3_decode_qkv_projection_core"
DEFAULT_RMS_QKV_FUNCTION_NAME = "gemma3_decode_rms_qkv_projection_core"
DEFAULT_INGRESS_FUNCTION_NAME = "gemma3_decode_ingress_rms_qkv_qknorm_rope"
DEFAULT_ATTENTION_O_FUNCTION_NAME = "gemma3_decode_attention_o_projection"
DEFAULT_POST_ATTENTION_RESIDUAL_FUNCTION_NAME = "gemma3_decode_post_attention_residual"
DEFAULT_FFN_GATE_UP_FUNCTION_NAME = "gemma3_decode_ffn_gate_up"
DEFAULT_GEGLU_DOWN_FUNCTION_NAME = "gemma3_decode_geglu_down"
DEFAULT_OUTPUT_FORMAT = "elf"
DEFAULT_FUSED_DQP_OBJECT = Path(__file__).with_name("build_peano") / "fused_dqp.o"
DEFAULT_ROPE_OBJECT = Path(__file__).with_name("build_peano") / "rope_halfsplit.o"
DEFAULT_FLOWQKV_OBJECT = Path(__file__).with_name("build_peano") / "flowqkv_single_token_q4_kv1_d256.o"
DEFAULT_THRESHOLD = 0.99
DEFAULT_LAYER = 0
DEFAULT_INGRESS_RESULT_JSON = Path(__file__).with_name("results") / "gemma3_1b_stitched_decode_ingress_probe.json"
DEFAULT_ATTENTION_O_RESULT_JSON = Path(__file__).with_name("results") / "gemma3_1b_stitched_attention_o_probe.json"
DEFAULT_POST_ATTENTION_RESIDUAL_RESULT_JSON = Path(__file__).with_name("results") / "gemma3_1b_stitched_post_attention_residual_probe.json"
DEFAULT_FFN_GATE_UP_RESULT_JSON = Path(__file__).with_name("results") / "gemma3_1b_stitched_ffn_gate_up_probe.json"
DEFAULT_GEGLU_DOWN_RESULT_JSON = Path(__file__).with_name("results") / "gemma3_1b_stitched_geglu_down_probe.json"

PROJECTION_CORE_ARG_TYPES = (
    "memref<8x4x5x5120xi8>",  # q packed Q4NX blocks, scale, min
    "memref<2x4x5x5120xi8>",  # k packed Q4NX blocks, scale, min
    "memref<2x4x5x5120xi8>",  # v packed Q4NX blocks, scale, min
    "memref<5x256xbf16>",  # padded normalized activation
    "memref<32x32xbf16>",  # q projection output, contiguous 1024 values
    "memref<8x32xbf16>",  # k projection output, contiguous 256 values
    "memref<8x32xbf16>",  # v projection output, contiguous 256 values
)

RMS_QKV_ARG_TYPES = (
    "memref<1x1152xbf16>",  # hidden input
    "memref<1152xbf16>",  # RMSNorm weight
    "memref<1x1152xbf16>",  # RMSNorm output view of the padded activation BO
    "memref<5x256xbf16>",  # padded activation alias, same BO as arg2 at runtime
    "memref<8x4x5x5120xi8>",  # q packed Q4NX blocks, scale, min
    "memref<2x4x5x5120xi8>",  # k packed Q4NX blocks, scale, min
    "memref<2x4x5x5120xi8>",  # v packed Q4NX blocks, scale, min
    "memref<32x32xbf16>",  # q projection output, contiguous 1024 values
    "memref<8x32xbf16>",  # k projection output, contiguous 256 values
    "memref<8x32xbf16>",  # v projection output, contiguous 256 values
)

INGRESS_ARG_TYPES = (
    "memref<1x1152xbf16>",  # hidden input
    "memref<1152xbf16>",  # input RMSNorm weight
    "memref<1x1152xbf16>",  # RMSNorm output view of the padded activation BO
    "memref<5x256xbf16>",  # padded activation alias, same BO as arg2 at runtime
    "memref<8x4x5x5120xi8>",  # q packed Q4NX blocks, scale, min
    "memref<2x4x5x5120xi8>",  # k packed Q4NX blocks, scale, min
    "memref<2x4x5x5120xi8>",  # v packed Q4NX blocks, scale, min
    "memref<32x32xbf16>",  # q projection output, contiguous 1024 values
    "memref<8x32xbf16>",  # k projection output, contiguous 256 values
    "memref<8x32xbf16>",  # v projection output, contiguous 256 values
    "memref<256xbf16>",  # q norm weight
    "memref<256xbf16>",  # k norm weight
    "memref<4x256xbf16>",  # q norm output
    "memref<1x256xbf16>",  # k norm output
    "memref<1024xbf16>",  # q RoPE LUT
    "memref<256xbf16>",  # k RoPE LUT
    "memref<4x256xbf16>",  # q RoPE output
    "memref<1x256xbf16>",  # k RoPE output
)

ATTENTION_O_ARG_TYPES = (
    "memref<1x4x256xbf16>",  # Q for single-token FlowQKV
    "memref<1x1x256xbf16>",  # K for single-token FlowQKV
    "memref<1x1x256xbf16>",  # V for single-token FlowQKV
    "memref<1x4x256xbf16>",  # attention output view
    "memref<4x256xbf16>",  # O-projection activation alias, same BO as arg3
    "memref<10x4x4x5120xi8>",  # O packed Q4NX blocks, scale, min
    "memref<40x32xbf16>",  # O projection output, contiguous 1280 padded values
)

POST_ATTENTION_RESIDUAL_ARG_TYPES = (
    "memref<1x1152xbf16>",  # O-projection output entering post-attention RMSNorm
    "memref<1152xbf16>",  # post-attention RMSNorm weight
    "memref<1x1152xbf16>",  # post-attention RMSNorm output
    "memref<1152xbf16>",  # residual lhs, the original layer input
    "memref<1152xbf16>",  # residual rhs alias, same BO as arg2 at runtime
    "memref<1152xbf16>",  # attention residual output
)

FFN_GATE_UP_ARG_TYPES = (
    "memref<1x1152xbf16>",  # attention residual entering pre-FF RMSNorm
    "memref<1152xbf16>",  # pre-feedforward RMSNorm weight
    "memref<1x1152xbf16>",  # pre-feedforward RMSNorm output view
    "memref<5x256xbf16>",  # padded activation alias, same BO as arg2 at runtime
    "memref<54x4x5x5120xi8>",  # gate packed Q4NX blocks, scale, min
    "memref<54x4x5x5120xi8>",  # up packed Q4NX blocks, scale, min
    "memref<216x32xbf16>",  # gate projection output, contiguous 6912 values
    "memref<216x32xbf16>",  # up projection output, contiguous 6912 values
)

GEGLU_DOWN_ARG_TYPES = (
    "memref<6912xbf16>",  # gate projection vector
    "memref<6912xbf16>",  # up projection vector
    "memref<6912xbf16>",  # GeGLU output
    "memref<27x256xbf16>",  # down activation alias, same BO as arg2
    "memref<20x2x27x5120xi8>",  # down packed Q4NX blocks, scale, min
    "memref<40x32xbf16>",  # down projection output, contiguous 1280 padded values
)


@dataclass(frozen=True)
class DecodeStitchStage:
    name: str
    status: str
    input_contract: str
    output_contract: str
    notes: str


@dataclass(frozen=True)
class DecodeStitchPlan:
    model_variant: str
    target_subgraph: str
    implemented_slice: str
    implemented_launches: int
    target_launches: int
    timing_policy: str
    stages: tuple[DecodeStitchStage, ...]
    remaining_bridges: tuple[str, ...]

    @property
    def status(self) -> str:
        if self.remaining_bridges:
            return "STITCHED_DECODE_TRACK_STARTED"
        return "STITCHED_DECODE_INGRESS_READY"

    def format(self) -> str:
        bridges = ",".join(self.remaining_bridges) if self.remaining_bridges else "none"
        stage_text = "|".join(f"{stage.name}:{stage.status}" for stage in self.stages)
        return (
            f"stitched_decode_plan model={self.model_variant} status={self.status} "
            f"target={self.target_subgraph} implemented_slice={self.implemented_slice} "
            f"implemented_launches={self.implemented_launches} target_launches={self.target_launches} "
            f"timing_policy={self.timing_policy} stages={stage_text} remaining_bridges={bridges}"
        )


@dataclass(frozen=True)
class StitchedIngressCorrelation:
    name: str
    shape: tuple[int, ...]
    correlation: float | None


@dataclass(frozen=True)
class StitchedIngressProbeResult:
    schema_version: int
    model_variant: str
    status: str
    sequence_kind: str
    layer_index: int
    function_name: str
    output_format: str
    launch_count: int
    argument_count: int
    output_correlations: tuple[StitchedIngressCorrelation, ...]
    dense_projection_correlations: dict[str, float | None]
    timed_kernel_count: int
    timed_kernel_seconds: float | None
    timing_window: str
    timing_notes: tuple[str, ...]
    threshold: float
    remaining_model_runner_gaps: tuple[str, ...]
    blockers: tuple[str, ...]
    command: tuple[str, ...]
    returncode: int | None
    elapsed_seconds: float | None
    git_commit: str | None
    dirty_worktree: bool | None
    stdout_tail: tuple[str, ...]
    stderr_tail: tuple[str, ...]

    @staticmethod
    def _corr_text(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.6f}"

    def format(self) -> str:
        blockers = ",".join(self.blockers) if self.blockers else "none"
        gaps = ",".join(self.remaining_model_runner_gaps) if self.remaining_model_runner_gaps else "none"
        corr_text = "|".join(
            f"{item.name}:{self._corr_text(item.correlation)}"
            for item in self.output_correlations
        )
        dense_text = "|".join(
            f"{name}:{self._corr_text(value)}"
            for name, value in sorted(self.dense_projection_correlations.items())
        )
        return (
            f"stitched_decode_ingress_probe model={self.model_variant} status={self.status} "
            f"sequence={self.sequence_kind} layer=L{self.layer_index} "
            f"function={self.function_name} output_format={self.output_format} "
            f"launches={self.launch_count} args={self.argument_count} "
            f"output_correlations={corr_text} "
            f"dense_projection_correlations={dense_text} "
            f"timed_kernel_count={self.timed_kernel_count} "
            f"timed_kernel_seconds={self._corr_text(self.timed_kernel_seconds)} "
            f"timing_window={self.timing_window} threshold={self.threshold:g} "
            f"model_runner_gaps={gaps} blockers={blockers}"
        )

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)



@dataclass(frozen=True)
class StitchedAttentionOProbeResult:
    schema_version: int
    model_variant: str
    status: str
    sequence_kind: str
    layer_index: int
    function_name: str
    output_format: str
    launch_count: int
    argument_count: int
    output_correlations: tuple[StitchedIngressCorrelation, ...]
    dense_o_projection_correlation: float | None
    timed_kernel_count: int
    timed_kernel_seconds: float | None
    timing_window: str
    timing_notes: tuple[str, ...]
    threshold: float
    remaining_model_runner_gaps: tuple[str, ...]
    blockers: tuple[str, ...]
    command: tuple[str, ...]
    returncode: int | None
    elapsed_seconds: float | None
    git_commit: str | None
    dirty_worktree: bool | None
    stdout_tail: tuple[str, ...]
    stderr_tail: tuple[str, ...]

    @staticmethod
    def _corr_text(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.6f}"

    def format(self) -> str:
        blockers = ",".join(self.blockers) if self.blockers else "none"
        gaps = ",".join(self.remaining_model_runner_gaps) if self.remaining_model_runner_gaps else "none"
        corr_text = "|".join(
            f"{item.name}:{self._corr_text(item.correlation)}"
            for item in self.output_correlations
        )
        return (
            f"stitched_attention_o_probe model={self.model_variant} status={self.status} "
            f"sequence={self.sequence_kind} layer=L{self.layer_index} "
            f"function={self.function_name} output_format={self.output_format} "
            f"launches={self.launch_count} args={self.argument_count} "
            f"output_correlations={corr_text} "
            f"dense_o_projection_correlation={self._corr_text(self.dense_o_projection_correlation)} "
            f"timed_kernel_count={self.timed_kernel_count} "
            f"timed_kernel_seconds={self._corr_text(self.timed_kernel_seconds)} "
            f"timing_window={self.timing_window} threshold={self.threshold:g} "
            f"model_runner_gaps={gaps} blockers={blockers}"
        )

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StitchedPostAttentionResidualProbeResult:
    schema_version: int
    model_variant: str
    status: str
    sequence_kind: str
    layer_index: int
    function_name: str
    output_format: str
    launch_count: int
    argument_count: int
    output_correlations: tuple[StitchedIngressCorrelation, ...]
    timed_kernel_count: int
    timed_kernel_seconds: float | None
    timing_window: str
    timing_notes: tuple[str, ...]
    threshold: float
    remaining_model_runner_gaps: tuple[str, ...]
    blockers: tuple[str, ...]
    command: tuple[str, ...]
    returncode: int | None
    elapsed_seconds: float | None
    git_commit: str | None
    dirty_worktree: bool | None
    stdout_tail: tuple[str, ...]
    stderr_tail: tuple[str, ...]

    @staticmethod
    def _corr_text(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.6f}"

    def format(self) -> str:
        blockers = ",".join(self.blockers) if self.blockers else "none"
        gaps = ",".join(self.remaining_model_runner_gaps) if self.remaining_model_runner_gaps else "none"
        corr_text = "|".join(
            f"{item.name}:{self._corr_text(item.correlation)}"
            for item in self.output_correlations
        )
        return (
            f"stitched_post_attention_residual_probe model={self.model_variant} status={self.status} "
            f"sequence={self.sequence_kind} layer=L{self.layer_index} "
            f"function={self.function_name} output_format={self.output_format} "
            f"launches={self.launch_count} args={self.argument_count} "
            f"output_correlations={corr_text} "
            f"timed_kernel_count={self.timed_kernel_count} "
            f"timed_kernel_seconds={self._corr_text(self.timed_kernel_seconds)} "
            f"timing_window={self.timing_window} threshold={self.threshold:g} "
            f"model_runner_gaps={gaps} blockers={blockers}"
        )

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StitchedFFNGateUpProbeResult:
    schema_version: int
    model_variant: str
    status: str
    sequence_kind: str
    layer_index: int
    function_name: str
    output_format: str
    launch_count: int
    argument_count: int
    output_correlations: tuple[StitchedIngressCorrelation, ...]
    dense_projection_correlations: dict[str, float | None]
    timed_kernel_count: int
    timed_kernel_seconds: float | None
    timing_window: str
    timing_notes: tuple[str, ...]
    threshold: float
    remaining_model_runner_gaps: tuple[str, ...]
    blockers: tuple[str, ...]
    command: tuple[str, ...]
    returncode: int | None
    elapsed_seconds: float | None
    git_commit: str | None
    dirty_worktree: bool | None
    stdout_tail: tuple[str, ...]
    stderr_tail: tuple[str, ...]

    @staticmethod
    def _corr_text(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.6f}"

    def format(self) -> str:
        blockers = ",".join(self.blockers) if self.blockers else "none"
        gaps = ",".join(self.remaining_model_runner_gaps) if self.remaining_model_runner_gaps else "none"
        corr_text = "|".join(
            f"{item.name}:{self._corr_text(item.correlation)}"
            for item in self.output_correlations
        )
        dense_text = "|".join(
            f"{name}:{self._corr_text(value)}"
            for name, value in sorted(self.dense_projection_correlations.items())
        )
        return (
            f"stitched_ffn_gate_up_probe model={self.model_variant} status={self.status} "
            f"sequence={self.sequence_kind} layer=L{self.layer_index} "
            f"function={self.function_name} output_format={self.output_format} "
            f"launches={self.launch_count} args={self.argument_count} "
            f"output_correlations={corr_text} "
            f"dense_projection_correlations={dense_text} "
            f"timed_kernel_count={self.timed_kernel_count} "
            f"timed_kernel_seconds={self._corr_text(self.timed_kernel_seconds)} "
            f"timing_window={self.timing_window} threshold={self.threshold:g} "
            f"model_runner_gaps={gaps} blockers={blockers}"
        )

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StitchedGeGLUDownProbeResult:
    schema_version: int
    model_variant: str
    status: str
    sequence_kind: str
    layer_index: int
    function_name: str
    output_format: str
    launch_count: int
    argument_count: int
    output_correlations: tuple[StitchedIngressCorrelation, ...]
    dense_down_projection_correlation: float | None
    timed_kernel_count: int
    timed_kernel_seconds: float | None
    timing_window: str
    timing_notes: tuple[str, ...]
    threshold: float
    remaining_model_runner_gaps: tuple[str, ...]
    blockers: tuple[str, ...]
    command: tuple[str, ...]
    returncode: int | None
    elapsed_seconds: float | None
    git_commit: str | None
    dirty_worktree: bool | None
    stdout_tail: tuple[str, ...]
    stderr_tail: tuple[str, ...]

    @staticmethod
    def _corr_text(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.6f}"

    def format(self) -> str:
        blockers = ",".join(self.blockers) if self.blockers else "none"
        gaps = ",".join(self.remaining_model_runner_gaps) if self.remaining_model_runner_gaps else "none"
        corr_text = "|".join(
            f"{item.name}:{self._corr_text(item.correlation)}"
            for item in self.output_correlations
        )
        return (
            f"stitched_geglu_down_probe model={self.model_variant} status={self.status} "
            f"sequence={self.sequence_kind} layer=L{self.layer_index} "
            f"function={self.function_name} output_format={self.output_format} "
            f"launches={self.launch_count} args={self.argument_count} "
            f"output_correlations={corr_text} "
            f"dense_down_projection_correlation={self._corr_text(self.dense_down_projection_correlation)} "
            f"timed_kernel_count={self.timed_kernel_count} "
            f"timed_kernel_seconds={self._corr_text(self.timed_kernel_seconds)} "
            f"timing_window={self.timing_window} threshold={self.threshold:g} "
            f"model_runner_gaps={gaps} blockers={blockers}"
        )

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


def build_decode_ingress_plan(model_variant: str = DEFAULT_MODEL) -> DecodeStitchPlan:
    stages = (
        DecodeStitchStage(
            "rmsnorm_activation_alias",
            "implemented-hardware-pass-alias",
            "layer_input:1x1152 + norm_weight:1152",
            "rms_out:1x1152 over activation_padded:5x256 BO",
            "Uses the proven weighted RMSNorm kernel and aliases its output BO as the padded FusedDQP activation view.",
        ),
        DecodeStitchStage(
            "qkv_projection_core",
            "implemented-hardware-pass-stitch",
            "activation_padded:5x256 + q/k/v packed static BOs",
            "q:32x32 k:8x32 v:8x32",
            "Uses full-col-block FusedDQP so host col-block accumulation is not in the timed path.",
        ),
        DecodeStitchStage(
            "projection_qk_views",
            "implemented-hardware-pass-bridge",
            "q:32x32 k:8x32 + q/k norm weights",
            "q_norm:4x256 k_norm:1x256",
            "Zero-copy collapse/expand view is fused into Q/K weighted RMSNorm.",
        ),
        DecodeStitchStage(
            "qk_norm_rope",
            "implemented-hardware-pass-stitch",
            "q_norm:4x256 k_norm:1x256 + q/k RoPE LUTs",
            "q_rope:4x256 k_rope:1x256",
            "Uses existing Gemma half-split RoPE wrapper after Q/K norm.",
        ),
    )
    return DecodeStitchPlan(
        model_variant=model_variant,
        target_subgraph="decode-ingress-rmsnorm-qkv-qknorm-rope",
        implemented_slice=DEFAULT_INGRESS_FUNCTION_NAME,
        implemented_launches=8,
        target_launches=8,
        timing_policy="compile-load-bo-preload-arg-binding-outside-timed-region",
        stages=stages,
        remaining_bridges=(),
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _activate_builder_paths() -> None:
    root = _repo_root()
    dataflow = str(root / "programming_examples/gemma3_dataflow_kernels")
    herd_sweep = str(root / "programming_examples/gemma3_herd_sweep")
    weighted = str(root / "programming_examples/weighted_rms_norm")
    for path in (dataflow, herd_sweep, weighted):
        while path in sys.path:
            sys.path.remove(path)
    sys.path.insert(0, weighted)
    sys.path.insert(0, dataflow)
    sys.path.insert(0, herd_sweep)


def _dataflow_flow_module_builder():
    module_path = _repo_root() / "programming_examples/gemma3_dataflow_kernels/flow_common.py"
    spec = importlib.util.spec_from_file_location("gemma3_stitched_dataflow_flow_common", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load flow_common.py from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_flow_module


def _projection_irs(object_file: Path) -> tuple[str, str, str]:
    _activate_builder_paths()
    from fused_dqp import build_paper_module

    q_ir = str(
        build_paper_module(
            32,
            256,
            "fused_dqp_accum_block_opt",
            str(object_file),
            32,
            5,
            2,
            4,
            "l2-gather",
        )
    )
    k_ir = str(
        build_paper_module(
            32,
            256,
            "fused_dqp_accum_block_opt",
            str(object_file),
            8,
            5,
            2,
            4,
            "l2-gather",
        )
    )
    v_ir = str(
        build_paper_module(
            32,
            256,
            "fused_dqp_accum_block_opt",
            str(object_file),
            8,
            5,
            2,
            4,
            "l2-gather",
        )
    )
    return q_ir, k_ir, v_ir


def build_projection_core_text(
    *,
    object_file: Path = DEFAULT_FUSED_DQP_OBJECT,
    function_name: str = DEFAULT_FUNCTION_NAME,
) -> str:
    """Build combined MLIR text for the standalone Q/K/V projection core."""
    q_ir, k_ir, v_ir = _projection_irs(object_file)
    return stitch_module_text(
        function_name=function_name,
        arg_types=PROJECTION_CORE_ARG_TYPES,
        specs=(
            StitchSpec(q_ir, "q", {0: 0, 1: 3, 2: 4}),
            StitchSpec(k_ir, "k", {0: 1, 1: 3, 2: 5}),
            StitchSpec(v_ir, "v", {0: 2, 1: 3, 2: 6}),
        ),
    )


def build_rms_qkv_text(
    *,
    object_file: Path = DEFAULT_FUSED_DQP_OBJECT,
    function_name: str = DEFAULT_RMS_QKV_FUNCTION_NAME,
) -> str:
    """Build combined MLIR text for RMSNorm alias plus Q/K/V projections."""
    _activate_builder_paths()
    from ml_dtypes import bfloat16
    from weighted_rms_norm import build_module as build_weighted_rms

    rms_ir = str(build_weighted_rms(1, 1152, bfloat16, 16, herd_x=1))
    q_ir, k_ir, v_ir = _projection_irs(object_file)
    return stitch_module_text(
        function_name=function_name,
        arg_types=RMS_QKV_ARG_TYPES,
        specs=(
            StitchSpec(rms_ir, "r", {0: 0, 1: 1, 2: 2}, wrap_bare_herds=True),
            StitchSpec(q_ir, "q", {0: 4, 1: 3, 2: 7}),
            StitchSpec(k_ir, "k", {0: 5, 1: 3, 2: 8}),
            StitchSpec(v_ir, "v", {0: 6, 1: 3, 2: 9}),
        ),
    )


def build_ingress_text(
    *,
    object_file: Path = DEFAULT_FUSED_DQP_OBJECT,
    rope_object_file: Path = DEFAULT_ROPE_OBJECT,
    function_name: str = DEFAULT_INGRESS_FUNCTION_NAME,
) -> str:
    """Build full decode ingress MLIR text: RMSNorm, Q/K/V, Q/K norm, RoPE."""
    _activate_builder_paths()
    from gemma3_projection_qk_norm import build_module as build_qk_norm
    from ml_dtypes import bfloat16
    from rope_halfsplit import build_module as build_rope
    from weighted_rms_norm import build_module as build_weighted_rms

    rms_ir = str(build_weighted_rms(1, 1152, bfloat16, 16, herd_x=1))
    q_ir, k_ir, v_ir = _projection_irs(object_file)
    q_norm_ir = str(build_qk_norm(32, 32, 4, 256))
    k_norm_ir = str(build_qk_norm(8, 32, 1, 256))
    q_rope_ir = str(build_rope(4, 256, bfloat16, 4, str(rope_object_file)))
    k_rope_ir = str(build_rope(1, 256, bfloat16, 1, str(rope_object_file)))
    return stitch_module_text(
        function_name=function_name,
        arg_types=INGRESS_ARG_TYPES,
        specs=(
            StitchSpec(rms_ir, "r", {0: 0, 1: 1, 2: 2}, wrap_bare_herds=True),
            StitchSpec(q_ir, "q", {0: 4, 1: 3, 2: 7}),
            StitchSpec(k_ir, "k", {0: 5, 1: 3, 2: 8}),
            StitchSpec(v_ir, "v", {0: 6, 1: 3, 2: 9}),
            StitchSpec(q_norm_ir, "qn", {0: 7, 1: 10, 2: 12}),
            StitchSpec(k_norm_ir, "kn", {0: 8, 1: 11, 2: 13}),
            StitchSpec(q_rope_ir, "rq", {0: 12, 1: 14, 2: 16}),
            StitchSpec(k_rope_ir, "rk", {0: 13, 1: 15, 2: 17}),
        ),
    )



def build_attention_o_text(
    *,
    object_file: Path = DEFAULT_FUSED_DQP_OBJECT,
    flowqkv_object_file: Path = DEFAULT_FLOWQKV_OBJECT,
    function_name: str = DEFAULT_ATTENTION_O_FUNCTION_NAME,
) -> str:
    """Build stitched single-token attention plus O projection MLIR text."""
    _activate_builder_paths()
    from fused_dqp import build_paper_module

    build_flow_module = _dataflow_flow_module_builder()
    flow_ir = str(
        build_flow_module(
            4,
            1,
            256,
            "flowqkv_chunk_bf16",
            str(flowqkv_object_file),
            "flowqkv_single_token",
            1,
            1,
            1,
        )
    )
    o_ir = str(
        build_paper_module(
            32,
            256,
            "fused_dqp_accum_block_opt",
            str(object_file),
            40,
            4,
            2,
            4,
            "l2-gather",
        )
    )
    return stitch_module_text(
        function_name=function_name,
        arg_types=ATTENTION_O_ARG_TYPES,
        specs=(
            StitchSpec(flow_ir, "att", {0: 0, 1: 1, 2: 2, 3: 3}),
            StitchSpec(o_ir, "o", {0: 5, 1: 4, 2: 6}),
        ),
    )


def build_ffn_gate_up_text(
    *,
    object_file: Path = DEFAULT_FUSED_DQP_OBJECT,
    function_name: str = DEFAULT_FFN_GATE_UP_FUNCTION_NAME,
) -> str:
    """Build stitched pre-FF RMSNorm plus gate/up projection MLIR text."""
    _activate_builder_paths()
    from fused_dqp import build_paper_module
    from ml_dtypes import bfloat16
    from weighted_rms_norm import build_module as build_weighted_rms

    rms_ir = str(build_weighted_rms(1, 1152, bfloat16, 16, herd_x=1))
    gate_ir = str(
        build_paper_module(
            32,
            256,
            "fused_dqp_accum_block_opt",
            str(object_file),
            216,
            5,
            2,
            4,
            "l2-gather",
        )
    )
    up_ir = str(
        build_paper_module(
            32,
            256,
            "fused_dqp_accum_block_opt",
            str(object_file),
            216,
            5,
            2,
            4,
            "l2-gather",
        )
    )
    return stitch_module_text(
        function_name=function_name,
        arg_types=FFN_GATE_UP_ARG_TYPES,
        specs=(
            StitchSpec(rms_ir, "ffn", {0: 0, 1: 1, 2: 2}, wrap_bare_herds=True),
            StitchSpec(gate_ir, "gate", {0: 4, 1: 3, 2: 6}),
            StitchSpec(up_ir, "up", {0: 5, 1: 3, 2: 7}),
        ),
    )


def build_geglu_down_text(
    *,
    object_file: Path = DEFAULT_FUSED_DQP_OBJECT,
    function_name: str = DEFAULT_GEGLU_DOWN_FUNCTION_NAME,
) -> str:
    """Build stitched GeGLU plus down projection MLIR text."""
    _activate_builder_paths()
    from fused_dqp import build_paper_module
    from geglu import build_module as build_geglu
    from ml_dtypes import bfloat16

    geglu_ir = str(build_geglu(6912, 288, bfloat16, 16))
    down_ir = str(
        build_paper_module(
            32,
            256,
            "fused_dqp_accum_block_opt",
            str(object_file),
            40,
            27,
            1,
            2,
            "l2-gather",
            stream_l1_col_blocks=True,
            l1_col_block_chunk=9,
        )
    )
    return stitch_module_text(
        function_name=function_name,
        arg_types=GEGLU_DOWN_ARG_TYPES,
        specs=(
            StitchSpec(geglu_ir, "geglu", {0: 0, 1: 1, 2: 2}, wrap_bare_herds=True),
            StitchSpec(down_ir, "down", {0: 4, 1: 3, 2: 5}),
        ),
    )


def build_post_attention_residual_text(
    *,
    function_name: str = DEFAULT_POST_ATTENTION_RESIDUAL_FUNCTION_NAME,
) -> str:
    """Build stitched post-attention RMSNorm plus attention residual MLIR text."""
    _activate_builder_paths()
    from ml_dtypes import bfloat16
    from residual_add import build_module as build_residual_add
    from weighted_rms_norm import build_module as build_weighted_rms

    rms_ir = str(build_weighted_rms(1, 1152, bfloat16, 16, herd_x=1))
    residual_ir = str(build_residual_add(1152, 288, bfloat16, 16))
    return stitch_module_text(
        function_name=function_name,
        arg_types=POST_ATTENTION_RESIDUAL_ARG_TYPES,
        specs=(
            StitchSpec(rms_ir, "pan", {0: 0, 1: 1, 2: 2}, wrap_bare_herds=True),
            StitchSpec(residual_ir, "par", {0: 3, 1: 4, 2: 5}, wrap_bare_herds=True),
        ),
    )


def _parse_module(text: str):
    from air.ir import Context, Module

    with Context() as ctx:
        return Module.parse(text, ctx)


def build_projection_core_module(*, object_file: Path = DEFAULT_FUSED_DQP_OBJECT):
    """Build and parse the stitched Q/K/V projection-core MLIR module."""
    return _parse_module(build_projection_core_text(object_file=object_file))


def build_rms_qkv_module(*, object_file: Path = DEFAULT_FUSED_DQP_OBJECT):
    """Build and parse the stitched RMSNorm-alias plus Q/K/V MLIR module."""
    return _parse_module(build_rms_qkv_text(object_file=object_file))


def build_ingress_module(
    *,
    object_file: Path = DEFAULT_FUSED_DQP_OBJECT,
    rope_object_file: Path = DEFAULT_ROPE_OBJECT,
):
    """Build and parse the full stitched decode ingress MLIR module."""
    return _parse_module(build_ingress_text(object_file=object_file, rope_object_file=rope_object_file))


def build_attention_o_module(
    *,
    object_file: Path = DEFAULT_FUSED_DQP_OBJECT,
    flowqkv_object_file: Path = DEFAULT_FLOWQKV_OBJECT,
):
    """Build and parse the stitched attention plus O-projection MLIR module."""
    return _parse_module(build_attention_o_text(object_file=object_file, flowqkv_object_file=flowqkv_object_file))


def build_ffn_gate_up_module(*, object_file: Path = DEFAULT_FUSED_DQP_OBJECT):
    """Build and parse the stitched pre-FF norm plus gate/up MLIR module."""
    return _parse_module(build_ffn_gate_up_text(object_file=object_file))


def build_geglu_down_module(*, object_file: Path = DEFAULT_FUSED_DQP_OBJECT):
    """Build and parse the stitched GeGLU plus down projection MLIR module."""
    return _parse_module(build_geglu_down_text(object_file=object_file))


def build_post_attention_residual_module():
    """Build and parse the stitched post-attention norm plus residual MLIR module."""
    return _parse_module(build_post_attention_residual_text())


def _stitched_backend_options(instance_name: str) -> dict[str, object]:
    return dict(
        verbose=False,
        omit_pingpong=True,
        output_format=DEFAULT_OUTPUT_FORMAT,
        instance_name=instance_name,
        target_device="npu2",
        runtime_loop_tiling_sizes=[1, 1],
        use_lock_race_condition_fix=True,
    )


def _compile_module(module, *, instance_name: str, output_binary_name: str):
    from air.backend.xrt import XRTBackend

    backend = XRTBackend(**_stitched_backend_options(instance_name))
    artifact = backend.compile(module, output_binary_name=output_binary_name)
    backend.unload()
    return artifact


def compile_projection_core(
    *,
    object_file: Path = DEFAULT_FUSED_DQP_OBJECT,
    output_binary_name: str = DEFAULT_FUNCTION_NAME,
):
    """Compile the projection core as an ELF artifact. Does not run hardware."""
    return _compile_module(
        build_projection_core_module(object_file=object_file),
        instance_name=DEFAULT_FUNCTION_NAME,
        output_binary_name=output_binary_name,
    )


def compile_rms_qkv(
    *,
    object_file: Path = DEFAULT_FUSED_DQP_OBJECT,
    output_binary_name: str = DEFAULT_RMS_QKV_FUNCTION_NAME,
):
    """Compile the RMSNorm-padding plus Q/K/V stitched slice as an ELF artifact."""
    return _compile_module(
        build_rms_qkv_module(object_file=object_file),
        instance_name=DEFAULT_RMS_QKV_FUNCTION_NAME,
        output_binary_name=output_binary_name,
    )


def compile_ingress(
    *,
    object_file: Path = DEFAULT_FUSED_DQP_OBJECT,
    rope_object_file: Path = DEFAULT_ROPE_OBJECT,
    output_binary_name: str = DEFAULT_INGRESS_FUNCTION_NAME,
):
    """Compile the full decode ingress stitched slice as an ELF artifact."""
    return _compile_module(
        build_ingress_module(object_file=object_file, rope_object_file=rope_object_file),
        instance_name=DEFAULT_INGRESS_FUNCTION_NAME,
        output_binary_name=output_binary_name,
    )


def compile_attention_o(
    *,
    object_file: Path = DEFAULT_FUSED_DQP_OBJECT,
    flowqkv_object_file: Path = DEFAULT_FLOWQKV_OBJECT,
    output_binary_name: str = DEFAULT_ATTENTION_O_FUNCTION_NAME,
):
    """Compile the stitched attention plus O-projection slice as an ELF artifact."""
    return _compile_module(
        build_attention_o_module(object_file=object_file, flowqkv_object_file=flowqkv_object_file),
        instance_name=DEFAULT_ATTENTION_O_FUNCTION_NAME,
        output_binary_name=output_binary_name,
    )


def compile_ffn_gate_up(
    *,
    object_file: Path = DEFAULT_FUSED_DQP_OBJECT,
    output_binary_name: str = DEFAULT_FFN_GATE_UP_FUNCTION_NAME,
):
    """Compile the stitched pre-FF RMSNorm plus gate/up slice as an ELF."""
    return _compile_module(
        build_ffn_gate_up_module(object_file=object_file),
        instance_name=DEFAULT_FFN_GATE_UP_FUNCTION_NAME,
        output_binary_name=output_binary_name,
    )


def compile_geglu_down(
    *,
    object_file: Path = DEFAULT_FUSED_DQP_OBJECT,
    output_binary_name: str = DEFAULT_GEGLU_DOWN_FUNCTION_NAME,
):
    """Compile the stitched GeGLU plus down projection slice as an ELF."""
    return _compile_module(
        build_geglu_down_module(object_file=object_file),
        instance_name=DEFAULT_GEGLU_DOWN_FUNCTION_NAME,
        output_binary_name=output_binary_name,
    )


def compile_post_attention_residual(
    *,
    output_binary_name: str = DEFAULT_POST_ATTENTION_RESIDUAL_FUNCTION_NAME,
):
    """Compile the stitched post-attention RMSNorm plus residual slice as an ELF."""
    return _compile_module(
        build_post_attention_residual_module(),
        instance_name=DEFAULT_POST_ATTENTION_RESIDUAL_FUNCTION_NAME,
        output_binary_name=output_binary_name,
    )


def _tail(lines: list[str], limit: int = 20) -> tuple[str, ...]:
    return tuple(lines[-limit:])


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


def _write_bo_arg(xrt, bo, array) -> None:
    from ml_dtypes import bfloat16

    payload = array.view("int16") if array.dtype == bfloat16 else array
    bo.write(payload, 0)
    bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)


def _git_info(repo: Path) -> tuple[str | None, bool | None]:
    import subprocess

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


def _run_multi_output_elf(
    *,
    mlir_module,
    backend_options: dict[str, object],
    arrays: list[object],
    readback: dict[str, tuple[int, tuple[int, ...], object]],
    timed_kernel_seconds: list[float] | None = None,
    bo_aliases: dict[int, int] | None = None,
) -> dict[str, object]:
    from air.backend.xrt import XRTBackend
    from filelock import FileLock

    try:
        import pyxrt as xrt
    except Exception as exc:
        raise RuntimeError("python:pyxrt is required for Gemma3 stitched ingress probe") from exc
    if backend_options.get("output_format") != "elf":
        raise RuntimeError("Gemma3 stitched ingress probe currently requires ELF output")

    backend = XRTBackend(**backend_options)
    artifact = backend.compile(mlir_module)
    try:
        with FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
            device = xrt.device(0)
            elf = xrt.elf(artifact.output_binary)
            context = xrt.hw_context(device, elf)
            kernel = xrt.ext.kernel(context, artifact.kernel)
            sizes = [array.size * array.itemsize for array in arrays]
            bo_aliases = bo_aliases or {}
            bos = []
            for index, size in enumerate(sizes):
                alias = bo_aliases.get(index)
                if alias is None:
                    bos.append(xrt.ext.bo(device, size))
                    continue
                if alias >= len(bos):
                    raise RuntimeError(f"BO alias {index}->{alias} targets an unallocated argument")
                if sizes[alias] < size:
                    raise RuntimeError(f"BO alias {index}->{alias} target is smaller than alias view")
                bos.append(bos[alias])
            for index, (bo, array) in enumerate(zip(bos, arrays)):
                if index in bo_aliases:
                    continue
                _write_bo_arg(xrt, bo, array)
            run = xrt.run(kernel)
            for index, bo in enumerate(bos):
                run.set_arg(index, bo)
            timed_start = time.perf_counter()
            run.start()
            run.wait2()
            timed_elapsed = time.perf_counter() - timed_start
            if timed_kernel_seconds is not None:
                timed_kernel_seconds.append(timed_elapsed)
            outputs = {}
            for name, (index, shape, dtype) in readback.items():
                bos[index].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
                read_size = int(__import__("numpy").prod(shape)) * __import__("numpy").dtype(dtype).itemsize
                outputs[name] = bos[index].read(read_size, 0).view(dtype).reshape(shape)
            return outputs
    finally:
        backend.unload()


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


def _rms_host(x, weight, eps: float = 1e-5):
    import numpy as np
    from ml_dtypes import bfloat16

    xf = x.astype(np.float32)
    wf = weight.astype(np.float32)
    rms = np.sqrt(np.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return ((xf / rms) * wf).astype(bfloat16)


def _load_safetensor_array_np(weights_dir: Path, tensor_key: str):
    try:
        from safetensors import safe_open
    except Exception as exc:
        raise RuntimeError("python:safetensors is required for Gemma3 stitched ingress probe") from exc
    for path in sorted(weights_dir.glob("*.safetensors")):
        with safe_open(str(path), framework="np") as handle:
            if tensor_key in handle.keys():
                return handle.get_tensor(tensor_key)
    raise RuntimeError(f"tensor key not found in safetensors: {tensor_key}")


def _pack_projection_for_herd_cols(weight, *, herd_cols: int):
    import numpy as np
    from fused_dqp import _pack_l3_inputs
    from gemma3_full_layer_probe import _repack_matrix_for_fused_dqp

    packed, scale, min_offset, padded = _repack_matrix_for_fused_dqp(weight)
    params = np.empty((*scale.shape[:-1], 512), dtype=scale.dtype)
    params[..., :256] = scale
    params[..., 256:] = min_offset
    row_blocks, col_blocks = packed.shape[:2]
    if row_blocks % herd_cols != 0:
        raise RuntimeError(f"row_blocks={row_blocks} is not divisible by herd_cols={herd_cols}")
    packed_l3 = _pack_l3_inputs(packed, params).reshape(row_blocks // herd_cols, herd_cols, col_blocks, -1)
    return packed_l3, packed, scale, min_offset, padded


def _pack_projection_for_ingress(weight):
    import numpy as np
    from fused_dqp import _pack_l3_inputs
    from gemma3_full_layer_probe import _repack_matrix_for_fused_dqp

    packed, scale, min_offset, padded = _repack_matrix_for_fused_dqp(weight)
    params = np.empty((*scale.shape[:-1], 512), dtype=scale.dtype)
    params[..., :256] = scale
    params[..., 256:] = min_offset
    row_blocks, col_blocks = packed.shape[:2]
    packed_l3 = _pack_l3_inputs(packed, params).reshape(row_blocks // 4, 4, col_blocks, -1)
    return packed_l3, packed, scale, min_offset, padded


def _run_ingress_hardware(args: argparse.Namespace) -> StitchedIngressProbeResult:
    _activate_builder_paths()
    from common import fused_dqp_paper_reference
    from gemma3_artifacts import default_weights_dir
    from gemma3_full_layer_probe import _projection_tensor_keys
    from ml_dtypes import bfloat16
    import numpy as np

    repo = _repo_root()
    git_commit, dirty = _git_info(repo)
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    blockers: list[str] = []
    start = time.perf_counter()
    layer_index = int(args.layer_index)
    weights_dir = (args.weights_dir or default_weights_dir(args.model_variant)).expanduser()
    timed_kernel_seconds: list[float] = []

    try:
        norm_keys = {
            "input": f"model.layers.{layer_index}.input_layernorm.weight",
            "q": f"model.layers.{layer_index}.self_attn.q_norm.weight",
            "k": f"model.layers.{layer_index}.self_attn.k_norm.weight",
        }
        input_norm_weight = _load_safetensor_array_np(weights_dir, norm_keys["input"]).astype(bfloat16).reshape(-1)
        q_norm_weight = _load_safetensor_array_np(weights_dir, norm_keys["q"]).astype(bfloat16).reshape(-1)
        k_norm_weight = _load_safetensor_array_np(weights_dir, norm_keys["k"]).astype(bfloat16).reshape(-1)
        projection_keys = _projection_tensor_keys(args.model_variant, weights_dir, layer_index)
        q_weight = _load_safetensor_array_np(weights_dir, projection_keys["q_proj"])
        k_weight = _load_safetensor_array_np(weights_dir, projection_keys["k_proj"])
        v_weight = _load_safetensor_array_np(weights_dir, projection_keys["v_proj"])

        rng = np.random.default_rng(args.seed)
        hidden = rng.uniform(-0.5, 0.5, size=(1, 1152)).astype(bfloat16)
        activation = np.zeros((5, 256), dtype=bfloat16)
        activation_expected = _rms_host(hidden, input_norm_weight).reshape(-1)
        activation.reshape(-1)[:1152] = activation_expected

        q_pack, q_packed, q_scale, q_min, _q_padded = _pack_projection_for_ingress(q_weight)
        k_pack, k_packed, k_scale, k_min, _k_padded = _pack_projection_for_ingress(k_weight)
        v_pack, v_packed, v_scale, v_min, _v_padded = _pack_projection_for_ingress(v_weight)

        q_expected = fused_dqp_paper_reference(q_packed, q_scale, q_min, activation, 32, 256).reshape(32, 32)
        k_expected = fused_dqp_paper_reference(k_packed, k_scale, k_min, activation, 32, 256).reshape(8, 32)
        v_expected = fused_dqp_paper_reference(v_packed, v_scale, v_min, activation, 32, 256).reshape(8, 32)
        q_norm_expected = _rms_host(q_expected.reshape(4, 256), q_norm_weight)
        k_norm_expected = _rms_host(k_expected.reshape(1, 256), k_norm_weight)

        dataflow = str(_repo_root() / "programming_examples/gemma3_dataflow_kernels")
        if dataflow not in sys.path:
            sys.path.insert(0, dataflow)
        from rope_halfsplit import rope_halfsplit_reference

        q_lut = _identity_rope_lut(4, 256, bfloat16)
        k_lut = _identity_rope_lut(1, 256, bfloat16)
        q_rope_expected = rope_halfsplit_reference(q_norm_expected, q_lut)
        k_rope_expected = rope_halfsplit_reference(k_norm_expected, k_lut)

        activation_storage = np.zeros((5, 256), dtype=bfloat16)
        arrays = [
            hidden,
            input_norm_weight,
            activation_storage,
            activation_storage,
            q_pack,
            k_pack,
            v_pack,
            np.zeros((32, 32), dtype=bfloat16),
            np.zeros((8, 32), dtype=bfloat16),
            np.zeros((8, 32), dtype=bfloat16),
            q_norm_weight,
            k_norm_weight,
            np.zeros((4, 256), dtype=bfloat16),
            np.zeros((1, 256), dtype=bfloat16),
            q_lut.reshape(-1),
            k_lut.reshape(-1),
            np.zeros((4, 256), dtype=bfloat16),
            np.zeros((1, 256), dtype=bfloat16),
        ]
        readback = {
            "input_norm": (2, (1, 1152), bfloat16),
            "activation": (3, (5, 256), bfloat16),
            "q_proj": (7, (32, 32), bfloat16),
            "k_proj": (8, (8, 32), bfloat16),
            "v_proj": (9, (8, 32), bfloat16),
            "q_norm": (12, (4, 256), bfloat16),
            "k_norm": (13, (1, 256), bfloat16),
            "q_rope": (16, (4, 256), bfloat16),
            "k_rope": (17, (1, 256), bfloat16),
        }
        actual = _run_multi_output_elf(
            mlir_module=build_ingress_module(object_file=args.object_file, rope_object_file=args.rope_object_file),
            backend_options=_stitched_backend_options(DEFAULT_INGRESS_FUNCTION_NAME),
            arrays=arrays,
            readback=readback,
            timed_kernel_seconds=timed_kernel_seconds,
            bo_aliases={3: 2},
        )
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
        correlations = []
        for name, expected_value in expected.items():
            corr = _correlation(actual[name], expected_value)
            correlations.append(
                StitchedIngressCorrelation(
                    name=name,
                    shape=tuple(int(dim) for dim in expected_value.shape),
                    correlation=corr,
                )
            )
            stdout_lines.append(f"{name} correlation: {corr:.6f}")
            if corr < args.threshold:
                blockers.append(f"{name}-correlation-low")

        dense_projection_correlations = {
            "q_proj": _correlation(
                actual["q_proj"].reshape(-1),
                (q_weight.astype(np.float32) @ activation_expected.astype(np.float32)).astype(bfloat16),
            ),
            "k_proj": _correlation(
                actual["k_proj"].reshape(-1),
                (k_weight.astype(np.float32) @ activation_expected.astype(np.float32)).astype(bfloat16),
            ),
            "v_proj": _correlation(
                actual["v_proj"].reshape(-1),
                (v_weight.astype(np.float32) @ activation_expected.astype(np.float32)).astype(bfloat16),
            ),
        }
    except Exception as exc:
        blockers.append(type(exc).__name__)
        stderr_lines.append(str(exc))
        correlations = ()
        dense_projection_correlations = {}

    status = "STITCHED_INGRESS_PASS" if not blockers else "STITCHED_INGRESS_BLOCKED"
    elapsed = time.perf_counter() - start
    timed_sum = float(sum(timed_kernel_seconds)) if timed_kernel_seconds else None
    return StitchedIngressProbeResult(
        schema_version=1,
        model_variant=args.model_variant,
        status=status,
        sequence_kind="decode-ingress-stitched",
        layer_index=layer_index,
        function_name=DEFAULT_INGRESS_FUNCTION_NAME,
        output_format=DEFAULT_OUTPUT_FORMAT,
        launch_count=8,
        argument_count=len(INGRESS_ARG_TYPES),
        output_correlations=tuple(correlations),
        dense_projection_correlations=dense_projection_correlations,
        timed_kernel_count=len(timed_kernel_seconds),
        timed_kernel_seconds=timed_sum,
        timing_window="single-stitched-elf-run-start-wait2-only-diagnostic",
        timing_notes=(
            "compile, ELF load, BO creation, BO writes, and argument binding occur before the timed run.start/wait2 window",
            "single-ingress diagnostic timing is not a TTFT/TPS or paper-parity result",
        ),
        threshold=float(args.threshold),
        remaining_model_runner_gaps=(
            "replace-staged-decode-ingress-with-stitched-elf",
            "stitch-rest-of-decode-layer",
            "prefill-produced-kv-cache-not-wired",
            "logits-sampling-not-wired",
        ),
        blockers=tuple(dict.fromkeys(blockers)),
        command=tuple(sys.argv),
        returncode=0 if not blockers else 1,
        elapsed_seconds=elapsed,
        git_commit=git_commit,
        dirty_worktree=dirty,
        stdout_tail=_tail(stdout_lines),
        stderr_tail=_tail(stderr_lines),
    )


def _run_attention_o_hardware(args: argparse.Namespace) -> StitchedAttentionOProbeResult:
    _activate_builder_paths()
    from common import attention_reference, fused_dqp_paper_reference
    from gemma3_artifacts import default_weights_dir
    from gemma3_full_layer_probe import _compile_flowqkv_single_token_kernel, _projection_tensor_keys
    from ml_dtypes import bfloat16
    import numpy as np

    repo = _repo_root()
    git_commit, dirty = _git_info(repo)
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    blockers: list[str] = []
    start = time.perf_counter()
    layer_index = int(args.layer_index)
    weights_dir = (args.weights_dir or default_weights_dir(args.model_variant)).expanduser()
    timed_kernel_seconds: list[float] = []
    correlations: list[StitchedIngressCorrelation] = []
    dense_o_correlation: float | None = None

    try:
        if not args.flowqkv_object_file.exists():
            _compile_flowqkv_single_token_kernel(args.flowqkv_object_file)
        projection_keys = _projection_tensor_keys(args.model_variant, weights_dir, layer_index)
        o_weight = _load_safetensor_array_np(weights_dir, projection_keys["o_proj"])
        o_pack, o_packed, o_scale, o_min, _o_padded = _pack_projection_for_ingress(o_weight)

        rng = np.random.default_rng(args.seed)
        val_range = 0.35
        q = rng.uniform(-val_range, val_range, (1, 4, 256)).astype(bfloat16)
        k = rng.uniform(-val_range, val_range, (1, 1, 256)).astype(bfloat16)
        v = rng.uniform(-val_range, val_range, (1, 1, 256)).astype(bfloat16)
        attention_expected = attention_reference(q.reshape(4, 256), k.reshape(1, 256), v.reshape(1, 256)).reshape(1, 4, 256).astype(bfloat16)
        o_expected = fused_dqp_paper_reference(
            o_packed,
            o_scale,
            o_min,
            attention_expected.reshape(4, 256),
            32,
            256,
        ).reshape(40, 32)

        attention_storage = np.zeros((1, 4, 256), dtype=bfloat16)
        arrays = [
            q,
            k,
            v,
            attention_storage,
            attention_storage.reshape(4, 256),
            o_pack,
            np.zeros((40, 32), dtype=bfloat16),
        ]
        actual = _run_multi_output_elf(
            mlir_module=build_attention_o_module(
                object_file=args.object_file,
                flowqkv_object_file=args.flowqkv_object_file,
            ),
            backend_options=_stitched_backend_options(DEFAULT_ATTENTION_O_FUNCTION_NAME),
            arrays=arrays,
            readback={
                "attention": (3, (1, 4, 256), bfloat16),
                "o_proj": (6, (40, 32), bfloat16),
            },
            timed_kernel_seconds=timed_kernel_seconds,
            bo_aliases={4: 3},
        )
        expected = {
            "attention": attention_expected,
            "o_proj": o_expected,
        }
        for name, expected_value in expected.items():
            corr = _correlation(actual[name], expected_value)
            correlations.append(
                StitchedIngressCorrelation(
                    name=name,
                    shape=tuple(int(dim) for dim in expected_value.shape),
                    correlation=corr,
                )
            )
            stdout_lines.append(f"{name} correlation: {corr:.6f}")
            if corr < args.threshold:
                blockers.append(f"{name}-correlation-low")
        dense_o_correlation = _correlation(
            actual["o_proj"].reshape(-1)[:1152],
            (o_weight.astype(np.float32) @ attention_expected.reshape(-1).astype(np.float32)).astype(bfloat16),
        )
    except Exception as exc:
        blockers.append(type(exc).__name__)
        stderr_lines.append(str(exc))

    status = "STITCHED_ATTENTION_O_PASS" if not blockers else "STITCHED_ATTENTION_O_BLOCKED"
    elapsed = time.perf_counter() - start
    timed_sum = float(sum(timed_kernel_seconds)) if timed_kernel_seconds else None
    return StitchedAttentionOProbeResult(
        schema_version=1,
        model_variant=args.model_variant,
        status=status,
        sequence_kind="decode-attention-o-stitched",
        layer_index=layer_index,
        function_name=DEFAULT_ATTENTION_O_FUNCTION_NAME,
        output_format=DEFAULT_OUTPUT_FORMAT,
        launch_count=2,
        argument_count=len(ATTENTION_O_ARG_TYPES),
        output_correlations=tuple(correlations),
        dense_o_projection_correlation=dense_o_correlation,
        timed_kernel_count=len(timed_kernel_seconds),
        timed_kernel_seconds=timed_sum,
        timing_window="single-stitched-elf-run-start-wait2-only-diagnostic",
        timing_notes=(
            "compile, ELF load, BO creation, BO writes, and argument binding occur before the timed run.start/wait2 window",
            "attention output and O-projection activation are aliased BO views: 1x4x256 and 4x256",
            "single attention/O diagnostic timing is not a TTFT/TPS or paper-parity result",
        ),
        threshold=float(args.threshold),
        remaining_model_runner_gaps=(
            "integrate-attention-o-stitched-slice-into-decode-loop",
            "stitch-remaining-ffn-layer-tail",
            "prefill-produced-kv-cache-not-wired",
            "logits-sampling-not-wired",
        ),
        blockers=tuple(dict.fromkeys(blockers)),
        command=tuple(sys.argv),
        returncode=0 if not blockers else 1,
        elapsed_seconds=elapsed,
        git_commit=git_commit,
        dirty_worktree=dirty,
        stdout_tail=_tail(stdout_lines),
        stderr_tail=_tail(stderr_lines),
    )


def _run_post_attention_residual_hardware(args: argparse.Namespace) -> StitchedPostAttentionResidualProbeResult:
    _activate_builder_paths()
    from gemma3_artifacts import default_weights_dir
    from ml_dtypes import bfloat16
    import numpy as np

    repo = _repo_root()
    git_commit, dirty = _git_info(repo)
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    blockers: list[str] = []
    start = time.perf_counter()
    layer_index = int(args.layer_index)
    weights_dir = (args.weights_dir or default_weights_dir(args.model_variant)).expanduser()
    timed_kernel_seconds: list[float] = []
    correlations: list[StitchedIngressCorrelation] = []

    try:
        norm_key = f"model.layers.{layer_index}.post_attention_layernorm.weight"
        norm_weight = _load_safetensor_array_np(weights_dir, norm_key).astype(bfloat16).reshape(-1)

        rng = np.random.default_rng(args.seed)
        o_input = rng.uniform(-0.45, 0.45, size=(1, 1152)).astype(bfloat16)
        residual_lhs = rng.uniform(-0.55, 0.55, size=(1152,)).astype(bfloat16)
        norm_expected = _rms_host(o_input, norm_weight)
        residual_expected = (
            residual_lhs.astype(np.float32) + norm_expected.reshape(-1).astype(np.float32)
        ).astype(bfloat16)

        norm_storage = np.zeros((1, 1152), dtype=bfloat16)
        arrays = [
            o_input,
            norm_weight,
            norm_storage,
            residual_lhs,
            norm_storage.reshape(-1),
            np.zeros((1152,), dtype=bfloat16),
        ]
        actual = _run_multi_output_elf(
            mlir_module=build_post_attention_residual_module(),
            backend_options=_stitched_backend_options(DEFAULT_POST_ATTENTION_RESIDUAL_FUNCTION_NAME),
            arrays=arrays,
            readback={
                "post_attention_norm": (2, (1, 1152), bfloat16),
                "attention_residual": (5, (1152,), bfloat16),
            },
            timed_kernel_seconds=timed_kernel_seconds,
            bo_aliases={4: 2},
        )
        expected = {
            "post_attention_norm": norm_expected,
            "attention_residual": residual_expected,
        }
        for name, expected_value in expected.items():
            corr = _correlation(actual[name], expected_value)
            correlations.append(
                StitchedIngressCorrelation(
                    name=name,
                    shape=tuple(int(dim) for dim in expected_value.shape),
                    correlation=corr,
                )
            )
            stdout_lines.append(f"{name} correlation: {corr:.6f}")
            if corr < args.threshold:
                blockers.append(f"{name}-correlation-low")
    except Exception as exc:
        blockers.append(type(exc).__name__)
        stderr_lines.append(str(exc))

    status = "STITCHED_POST_ATTENTION_RESIDUAL_PASS" if not blockers else "STITCHED_POST_ATTENTION_RESIDUAL_BLOCKED"
    elapsed = time.perf_counter() - start
    timed_sum = float(sum(timed_kernel_seconds)) if timed_kernel_seconds else None
    return StitchedPostAttentionResidualProbeResult(
        schema_version=1,
        model_variant=args.model_variant,
        status=status,
        sequence_kind="decode-post-attention-residual-stitched",
        layer_index=layer_index,
        function_name=DEFAULT_POST_ATTENTION_RESIDUAL_FUNCTION_NAME,
        output_format=DEFAULT_OUTPUT_FORMAT,
        launch_count=2,
        argument_count=len(POST_ATTENTION_RESIDUAL_ARG_TYPES),
        output_correlations=tuple(correlations),
        timed_kernel_count=len(timed_kernel_seconds),
        timed_kernel_seconds=timed_sum,
        timing_window="single-stitched-elf-run-start-wait2-only-diagnostic",
        timing_notes=(
            "compile, ELF load, BO creation, BO writes, and argument binding occur before the timed run.start/wait2 window",
            "post-attention RMSNorm output and residual RHS are aliased BO views: 1x1152 and 1152",
            "single post-attention residual diagnostic timing is not a TTFT/TPS or paper-parity result",
        ),
        threshold=float(args.threshold),
        remaining_model_runner_gaps=(
            "integrate-post-attention-residual-stitched-slice-into-decode-loop",
            "stitch-remaining-ffn-layer-tail",
            "prefill-produced-kv-cache-not-wired",
            "logits-sampling-not-wired",
        ),
        blockers=tuple(dict.fromkeys(blockers)),
        command=tuple(sys.argv),
        returncode=0 if not blockers else 1,
        elapsed_seconds=elapsed,
        git_commit=git_commit,
        dirty_worktree=dirty,
        stdout_tail=_tail(stdout_lines),
        stderr_tail=_tail(stderr_lines),
    )


def _run_ffn_gate_up_hardware(args: argparse.Namespace) -> StitchedFFNGateUpProbeResult:
    _activate_builder_paths()
    from common import fused_dqp_paper_reference
    from gemma3_artifacts import default_weights_dir
    from gemma3_full_layer_probe import _projection_tensor_keys
    from ml_dtypes import bfloat16
    import numpy as np

    repo = _repo_root()
    git_commit, dirty = _git_info(repo)
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    blockers: list[str] = []
    start = time.perf_counter()
    layer_index = int(args.layer_index)
    weights_dir = (args.weights_dir or default_weights_dir(args.model_variant)).expanduser()
    timed_kernel_seconds: list[float] = []
    correlations: list[StitchedIngressCorrelation] = []
    dense_projection_correlations: dict[str, float | None] = {}

    try:
        norm_key = f"model.layers.{layer_index}.pre_feedforward_layernorm.weight"
        norm_weight = _load_safetensor_array_np(weights_dir, norm_key).astype(bfloat16).reshape(-1)
        projection_keys = _projection_tensor_keys(args.model_variant, weights_dir, layer_index)
        gate_weight = _load_safetensor_array_np(weights_dir, projection_keys["gate_proj"])
        up_weight = _load_safetensor_array_np(weights_dir, projection_keys["up_proj"])
        gate_pack, gate_packed, gate_scale, gate_min, _gate_padded = _pack_projection_for_ingress(gate_weight)
        up_pack, up_packed, up_scale, up_min, _up_padded = _pack_projection_for_ingress(up_weight)

        rng = np.random.default_rng(args.seed)
        residual = rng.uniform(-0.45, 0.45, size=(1, 1152)).astype(bfloat16)
        activation = np.zeros((5, 256), dtype=bfloat16)
        activation_expected = _rms_host(residual, norm_weight).reshape(-1)
        activation.reshape(-1)[:1152] = activation_expected

        gate_expected = fused_dqp_paper_reference(
            gate_packed,
            gate_scale,
            gate_min,
            activation,
            32,
            256,
        ).reshape(216, 32)
        up_expected = fused_dqp_paper_reference(
            up_packed,
            up_scale,
            up_min,
            activation,
            32,
            256,
        ).reshape(216, 32)

        activation_storage = np.zeros((5, 256), dtype=bfloat16)
        arrays = [
            residual,
            norm_weight,
            activation_storage,
            activation_storage,
            gate_pack,
            up_pack,
            np.zeros((216, 32), dtype=bfloat16),
            np.zeros((216, 32), dtype=bfloat16),
        ]
        actual = _run_multi_output_elf(
            mlir_module=build_ffn_gate_up_module(object_file=args.object_file),
            backend_options=_stitched_backend_options(DEFAULT_FFN_GATE_UP_FUNCTION_NAME),
            arrays=arrays,
            readback={
                "pre_feedforward_norm": (2, (1, 1152), bfloat16),
                "activation": (3, (5, 256), bfloat16),
                "gate_proj": (6, (216, 32), bfloat16),
                "up_proj": (7, (216, 32), bfloat16),
            },
            timed_kernel_seconds=timed_kernel_seconds,
            bo_aliases={3: 2},
        )
        expected = {
            "pre_feedforward_norm": activation_expected.reshape(1, 1152),
            "activation": activation,
            "gate_proj": gate_expected,
            "up_proj": up_expected,
        }
        for name, expected_value in expected.items():
            corr = _correlation(actual[name], expected_value)
            correlations.append(
                StitchedIngressCorrelation(
                    name=name,
                    shape=tuple(int(dim) for dim in expected_value.shape),
                    correlation=corr,
                )
            )
            stdout_lines.append(f"{name} correlation: {corr:.6f}")
            if corr < args.threshold:
                blockers.append(f"{name}-correlation-low")
        dense_projection_correlations = {
            "gate_proj": _correlation(
                actual["gate_proj"].reshape(-1)[:6912],
                (gate_weight.astype(np.float32) @ activation_expected.astype(np.float32)).astype(bfloat16),
            ),
            "up_proj": _correlation(
                actual["up_proj"].reshape(-1)[:6912],
                (up_weight.astype(np.float32) @ activation_expected.astype(np.float32)).astype(bfloat16),
            ),
        }
    except Exception as exc:
        blockers.append(type(exc).__name__)
        stderr_lines.append(str(exc))

    status = "STITCHED_FFN_GATE_UP_PASS" if not blockers else "STITCHED_FFN_GATE_UP_BLOCKED"
    elapsed = time.perf_counter() - start
    timed_sum = float(sum(timed_kernel_seconds)) if timed_kernel_seconds else None
    return StitchedFFNGateUpProbeResult(
        schema_version=1,
        model_variant=args.model_variant,
        status=status,
        sequence_kind="decode-ffn-gate-up-stitched",
        layer_index=layer_index,
        function_name=DEFAULT_FFN_GATE_UP_FUNCTION_NAME,
        output_format=DEFAULT_OUTPUT_FORMAT,
        launch_count=3,
        argument_count=len(FFN_GATE_UP_ARG_TYPES),
        output_correlations=tuple(correlations),
        dense_projection_correlations=dense_projection_correlations,
        timed_kernel_count=len(timed_kernel_seconds),
        timed_kernel_seconds=timed_sum,
        timing_window="single-stitched-elf-run-start-wait2-only-diagnostic",
        timing_notes=(
            "compile, ELF load, BO creation, BO writes, and argument binding occur before the timed run.start/wait2 window",
            "pre-feedforward RMSNorm output and padded gate/up activation are aliased BO views: 1x1152 and 5x256",
            "single FFN gate/up diagnostic timing is not a TTFT/TPS or paper-parity result",
        ),
        threshold=float(args.threshold),
        remaining_model_runner_gaps=(
            "integrate-ffn-gate-up-stitched-slice-into-decode-loop",
            "stitch-geglu-down-postff-final-residual-tail",
            "prefill-produced-kv-cache-not-wired",
            "logits-sampling-not-wired",
        ),
        blockers=tuple(dict.fromkeys(blockers)),
        command=tuple(sys.argv),
        returncode=0 if not blockers else 1,
        elapsed_seconds=elapsed,
        git_commit=git_commit,
        dirty_worktree=dirty,
        stdout_tail=_tail(stdout_lines),
        stderr_tail=_tail(stderr_lines),
    )


def _run_geglu_down_hardware(args: argparse.Namespace) -> StitchedGeGLUDownProbeResult:
    _activate_builder_paths()
    from common import fused_dqp_paper_reference
    from gemma3_artifacts import default_weights_dir
    from gemma3_full_layer_probe import _projection_tensor_keys
    from geglu import geglu_reference
    from ml_dtypes import bfloat16
    import numpy as np

    repo = _repo_root()
    git_commit, dirty = _git_info(repo)
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    blockers: list[str] = []
    start = time.perf_counter()
    layer_index = int(args.layer_index)
    weights_dir = (args.weights_dir or default_weights_dir(args.model_variant)).expanduser()
    timed_kernel_seconds: list[float] = []
    correlations: list[StitchedIngressCorrelation] = []
    dense_down_correlation: float | None = None

    try:
        projection_keys = _projection_tensor_keys(args.model_variant, weights_dir, layer_index)
        down_weight = _load_safetensor_array_np(weights_dir, projection_keys["down_proj"])
        down_pack, down_packed, down_scale, down_min, _down_padded = _pack_projection_for_herd_cols(down_weight, herd_cols=2)

        rng = np.random.default_rng(args.seed)
        gate = rng.uniform(-0.45, 0.45, size=(6912,)).astype(bfloat16)
        up = rng.uniform(-0.45, 0.45, size=(6912,)).astype(bfloat16)
        geglu_expected = geglu_reference(gate, up)
        down_expected = fused_dqp_paper_reference(
            down_packed,
            down_scale,
            down_min,
            geglu_expected.reshape(27, 256),
            32,
            256,
        ).reshape(40, 32)

        geglu_storage = np.zeros((6912,), dtype=bfloat16)
        arrays = [
            gate,
            up,
            geglu_storage,
            geglu_storage.reshape(27, 256),
            down_pack,
            np.zeros((40, 32), dtype=bfloat16),
        ]
        actual = _run_multi_output_elf(
            mlir_module=build_geglu_down_module(object_file=args.object_file),
            backend_options=_stitched_backend_options(DEFAULT_GEGLU_DOWN_FUNCTION_NAME),
            arrays=arrays,
            readback={
                "geglu": (2, (6912,), bfloat16),
                "down_proj": (5, (40, 32), bfloat16),
            },
            timed_kernel_seconds=timed_kernel_seconds,
            bo_aliases={3: 2},
        )
        expected = {
            "geglu": geglu_expected,
            "down_proj": down_expected,
        }
        for name, expected_value in expected.items():
            corr = _correlation(actual[name], expected_value)
            correlations.append(
                StitchedIngressCorrelation(
                    name=name,
                    shape=tuple(int(dim) for dim in expected_value.shape),
                    correlation=corr,
                )
            )
            stdout_lines.append(f"{name} correlation: {corr:.6f}")
            if corr < args.threshold:
                blockers.append(f"{name}-correlation-low")
        dense_down_correlation = _correlation(
            actual["down_proj"].reshape(-1)[:1152],
            (down_weight.astype(np.float32) @ geglu_expected.astype(np.float32)).astype(bfloat16),
        )
    except Exception as exc:
        blockers.append(type(exc).__name__)
        stderr_lines.append(str(exc))

    status = "STITCHED_GEGLU_DOWN_PASS" if not blockers else "STITCHED_GEGLU_DOWN_BLOCKED"
    elapsed = time.perf_counter() - start
    timed_sum = float(sum(timed_kernel_seconds)) if timed_kernel_seconds else None
    return StitchedGeGLUDownProbeResult(
        schema_version=1,
        model_variant=args.model_variant,
        status=status,
        sequence_kind="decode-geglu-down-stitched",
        layer_index=layer_index,
        function_name=DEFAULT_GEGLU_DOWN_FUNCTION_NAME,
        output_format=DEFAULT_OUTPUT_FORMAT,
        launch_count=2,
        argument_count=len(GEGLU_DOWN_ARG_TYPES),
        output_correlations=tuple(correlations),
        dense_down_projection_correlation=dense_down_correlation,
        timed_kernel_count=len(timed_kernel_seconds),
        timed_kernel_seconds=timed_sum,
        timing_window="single-stitched-elf-run-start-wait2-only-diagnostic",
        timing_notes=(
            "compile, ELF load, BO creation, BO writes, and argument binding occur before the timed run.start/wait2 window",
            "GeGLU output and down-projection activation are aliased BO views: 6912 and 27x256",
            "single GeGLU/down diagnostic timing is not a TTFT/TPS or paper-parity result",
        ),
        threshold=float(args.threshold),
        remaining_model_runner_gaps=(
            "integrate-geglu-down-stitched-slice-into-decode-loop",
            "stitch-postff-final-residual-tail",
            "prefill-produced-kv-cache-not-wired",
            "logits-sampling-not-wired",
        ),
        blockers=tuple(dict.fromkeys(blockers)),
        command=tuple(sys.argv),
        returncode=0 if not blockers else 1,
        elapsed_seconds=elapsed,
        git_commit=git_commit,
        dirty_worktree=dirty,
        stdout_tail=_tail(stdout_lines),
        stderr_tail=_tail(stderr_lines),
    )


def _projection_core_text_self_test() -> None:
    a_ir = """module {
  func.func private @fused_dqp_accum_block_opt(memref<4xbf16>) attributes {link_with = "fused_dqp.o", llvm.emit_c_interface}
  func.func @fused_dqp_paper(%arg0: memref<4xbf16>, %arg1: memref<4xbf16>, %arg2: memref<4xbf16>) {
    air.launch () in () args(%arg3=%arg0, %arg4=%arg1, %arg5=%arg2) : memref<4xbf16>, memref<4xbf16>, memref<4xbf16> {
      call @fused_dqp_accum_block_opt(%arg3) : (memref<4xbf16>) -> ()
    }
    return
  }
}
"""
    text = stitch_module_text(
        function_name=DEFAULT_FUNCTION_NAME,
        arg_types=("memref<4xbf16>",) * 7,
        specs=(
            StitchSpec(a_ir, "q", {0: 0, 1: 3, 2: 4}),
            StitchSpec(a_ir, "k", {0: 1, 1: 3, 2: 5}),
            StitchSpec(a_ir, "v", {0: 2, 1: 3, 2: 6}),
        ),
    )
    if text.count("air.launch") != 3:
        raise AssertionError("expected three stitched projection launches")
    if text.count("func.func private @fused_dqp_accum_block_opt") != 1:
        raise AssertionError("expected deduped FusedDQP private declaration")
    if "%q_arg0" in text or "%k_arg0" in text or "%v_arg0" in text:
        raise AssertionError("projection core launch args were not remapped")


def _self_test() -> None:
    _projection_core_text_self_test()
    plan = build_decode_ingress_plan()
    if plan.implemented_launches != 8 or plan.target_launches != 8:
        raise AssertionError("unexpected stitched decode launch counts")
    if plan.remaining_bridges:
        raise AssertionError("full decode ingress should not have remaining bridge blockers")
    print(plan.format())


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma3 stitched decode ELF track")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--print-projection-core-mlir", action="store_true")
    parser.add_argument("--print-rms-qkv-mlir", action="store_true")
    parser.add_argument("--print-ingress-mlir", action="store_true")
    parser.add_argument("--print-attention-o-mlir", action="store_true")
    parser.add_argument("--print-post-attention-residual-mlir", action="store_true")
    parser.add_argument("--print-ffn-gate-up-mlir", action="store_true")
    parser.add_argument("--print-geglu-down-mlir", action="store_true")
    parser.add_argument("--parse-projection-core", action="store_true")
    parser.add_argument("--parse-rms-qkv", action="store_true")
    parser.add_argument("--parse-ingress", action="store_true")
    parser.add_argument("--parse-attention-o", action="store_true")
    parser.add_argument("--parse-post-attention-residual", action="store_true")
    parser.add_argument("--parse-ffn-gate-up", action="store_true")
    parser.add_argument("--parse-geglu-down", action="store_true")
    parser.add_argument("--compile-projection-core", action="store_true")
    parser.add_argument("--compile-rms-qkv", action="store_true")
    parser.add_argument("--compile-ingress", action="store_true")
    parser.add_argument("--compile-attention-o", action="store_true")
    parser.add_argument("--compile-post-attention-residual", action="store_true")
    parser.add_argument("--compile-ffn-gate-up", action="store_true")
    parser.add_argument("--compile-geglu-down", action="store_true")
    parser.add_argument("--run-ingress-hardware", action="store_true")
    parser.add_argument("--run-attention-o-hardware", action="store_true")
    parser.add_argument("--run-post-attention-residual-hardware", action="store_true")
    parser.add_argument("--run-ffn-gate-up-hardware", action="store_true")
    parser.add_argument("--run-geglu-down-hardware", action="store_true")
    parser.add_argument("--object-file", type=Path, default=DEFAULT_FUSED_DQP_OBJECT)
    parser.add_argument("--rope-object-file", type=Path, default=DEFAULT_ROPE_OBJECT)
    parser.add_argument("--flowqkv-object-file", type=Path, default=DEFAULT_FLOWQKV_OBJECT)
    parser.add_argument("--model-variant", default=DEFAULT_MODEL)
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--layer-index", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return
    if args.print_plan:
        print(build_decode_ingress_plan().format())
        return
    if args.print_projection_core_mlir:
        print(build_projection_core_text(object_file=args.object_file))
        return
    if args.print_rms_qkv_mlir:
        print(build_rms_qkv_text(object_file=args.object_file))
        return
    if args.print_ingress_mlir:
        print(build_ingress_text(object_file=args.object_file, rope_object_file=args.rope_object_file))
        return
    if args.print_attention_o_mlir:
        print(build_attention_o_text(object_file=args.object_file, flowqkv_object_file=args.flowqkv_object_file))
        return
    if args.print_post_attention_residual_mlir:
        print(build_post_attention_residual_text())
        return
    if args.print_ffn_gate_up_mlir:
        print(build_ffn_gate_up_text(object_file=args.object_file))
        return
    if args.print_geglu_down_mlir:
        print(build_geglu_down_text(object_file=args.object_file))
        return
    if args.parse_projection_core:
        build_projection_core_module(object_file=args.object_file)
        print(
            f"stitched_decode_projection_core status=PARSE_PASS function={DEFAULT_FUNCTION_NAME} "
            f"launches=3 args={len(PROJECTION_CORE_ARG_TYPES)} output_format={DEFAULT_OUTPUT_FORMAT}"
        )
        return
    if args.parse_rms_qkv:
        build_rms_qkv_module(object_file=args.object_file)
        print(
            f"stitched_decode_rms_qkv status=PARSE_PASS function={DEFAULT_RMS_QKV_FUNCTION_NAME} "
            f"launches=4 args={len(RMS_QKV_ARG_TYPES)} output_format={DEFAULT_OUTPUT_FORMAT}"
        )
        return
    if args.parse_ingress:
        build_ingress_module(object_file=args.object_file, rope_object_file=args.rope_object_file)
        print(
            f"stitched_decode_ingress status=PARSE_PASS function={DEFAULT_INGRESS_FUNCTION_NAME} "
            f"launches=8 args={len(INGRESS_ARG_TYPES)} output_format={DEFAULT_OUTPUT_FORMAT}"
        )
        return
    if args.parse_attention_o:
        build_attention_o_module(object_file=args.object_file, flowqkv_object_file=args.flowqkv_object_file)
        print(
            f"stitched_attention_o status=PARSE_PASS function={DEFAULT_ATTENTION_O_FUNCTION_NAME} "
            f"launches=2 args={len(ATTENTION_O_ARG_TYPES)} output_format={DEFAULT_OUTPUT_FORMAT}"
        )
        return
    if args.parse_post_attention_residual:
        build_post_attention_residual_module()
        print(
            f"stitched_post_attention_residual status=PARSE_PASS function={DEFAULT_POST_ATTENTION_RESIDUAL_FUNCTION_NAME} "
            f"launches=2 args={len(POST_ATTENTION_RESIDUAL_ARG_TYPES)} output_format={DEFAULT_OUTPUT_FORMAT}"
        )
        return
    if args.parse_ffn_gate_up:
        build_ffn_gate_up_module(object_file=args.object_file)
        print(
            f"stitched_ffn_gate_up status=PARSE_PASS function={DEFAULT_FFN_GATE_UP_FUNCTION_NAME} "
            f"launches=3 args={len(FFN_GATE_UP_ARG_TYPES)} output_format={DEFAULT_OUTPUT_FORMAT}"
        )
        return
    if args.parse_geglu_down:
        build_geglu_down_module(object_file=args.object_file)
        print(
            f"stitched_geglu_down status=PARSE_PASS function={DEFAULT_GEGLU_DOWN_FUNCTION_NAME} "
            f"launches=2 args={len(GEGLU_DOWN_ARG_TYPES)} output_format={DEFAULT_OUTPUT_FORMAT}"
        )
        return
    if args.compile_projection_core:
        artifact = compile_projection_core(object_file=args.object_file)
        print(
            f"stitched_decode_projection_core status=COMPILE_PASS function={DEFAULT_FUNCTION_NAME} "
            f"output={artifact.output_binary}"
        )
        return
    if args.compile_rms_qkv:
        artifact = compile_rms_qkv(object_file=args.object_file)
        print(
            f"stitched_decode_rms_qkv status=COMPILE_PASS function={DEFAULT_RMS_QKV_FUNCTION_NAME} "
            f"output={artifact.output_binary}"
        )
        return
    if args.compile_ingress:
        artifact = compile_ingress(object_file=args.object_file, rope_object_file=args.rope_object_file)
        print(
            f"stitched_decode_ingress status=COMPILE_PASS function={DEFAULT_INGRESS_FUNCTION_NAME} "
            f"output={artifact.output_binary}"
        )
        return
    if args.compile_attention_o:
        artifact = compile_attention_o(object_file=args.object_file, flowqkv_object_file=args.flowqkv_object_file)
        print(
            f"stitched_attention_o status=COMPILE_PASS function={DEFAULT_ATTENTION_O_FUNCTION_NAME} "
            f"output={artifact.output_binary}"
        )
        return
    if args.compile_post_attention_residual:
        artifact = compile_post_attention_residual()
        print(
            f"stitched_post_attention_residual status=COMPILE_PASS function={DEFAULT_POST_ATTENTION_RESIDUAL_FUNCTION_NAME} "
            f"output={artifact.output_binary}"
        )
        return
    if args.compile_ffn_gate_up:
        artifact = compile_ffn_gate_up(object_file=args.object_file)
        print(
            f"stitched_ffn_gate_up status=COMPILE_PASS function={DEFAULT_FFN_GATE_UP_FUNCTION_NAME} "
            f"output={artifact.output_binary}"
        )
        return
    if args.compile_geglu_down:
        artifact = compile_geglu_down(object_file=args.object_file)
        print(
            f"stitched_geglu_down status=COMPILE_PASS function={DEFAULT_GEGLU_DOWN_FUNCTION_NAME} "
            f"output={artifact.output_binary}"
        )
        return
    if args.run_ingress_hardware:
        result = _run_ingress_hardware(args)
        print(result.format())
        if args.result_json:
            args.result_json.parent.mkdir(parents=True, exist_ok=True)
            args.result_json.write_text(json.dumps(result.to_json_dict(), indent=2, sort_keys=True) + "\n")
            print(f"GEMMA3_STITCHED_INGRESS_JSON: {args.result_json}")
        raise SystemExit(0 if result.status == "STITCHED_INGRESS_PASS" else 1)
    if args.run_attention_o_hardware:
        result = _run_attention_o_hardware(args)
        print(result.format())
        if args.result_json:
            args.result_json.parent.mkdir(parents=True, exist_ok=True)
            args.result_json.write_text(json.dumps(result.to_json_dict(), indent=2, sort_keys=True) + "\n")
            print(f"GEMMA3_STITCHED_ATTENTION_O_JSON: {args.result_json}")
        raise SystemExit(0 if result.status == "STITCHED_ATTENTION_O_PASS" else 1)
    if args.run_post_attention_residual_hardware:
        result = _run_post_attention_residual_hardware(args)
        print(result.format())
        if args.result_json:
            args.result_json.parent.mkdir(parents=True, exist_ok=True)
            args.result_json.write_text(json.dumps(result.to_json_dict(), indent=2, sort_keys=True) + "\n")
            print(f"GEMMA3_STITCHED_POST_ATTENTION_RESIDUAL_JSON: {args.result_json}")
        raise SystemExit(0 if result.status == "STITCHED_POST_ATTENTION_RESIDUAL_PASS" else 1)
    if args.run_ffn_gate_up_hardware:
        result = _run_ffn_gate_up_hardware(args)
        print(result.format())
        if args.result_json:
            args.result_json.parent.mkdir(parents=True, exist_ok=True)
            args.result_json.write_text(json.dumps(result.to_json_dict(), indent=2, sort_keys=True) + "\n")
            print(f"GEMMA3_STITCHED_FFN_GATE_UP_JSON: {args.result_json}")
        raise SystemExit(0 if result.status == "STITCHED_FFN_GATE_UP_PASS" else 1)
    if args.run_geglu_down_hardware:
        result = _run_geglu_down_hardware(args)
        print(result.format())
        if args.result_json:
            args.result_json.parent.mkdir(parents=True, exist_ok=True)
            args.result_json.write_text(json.dumps(result.to_json_dict(), indent=2, sort_keys=True) + "\n")
            print(f"GEMMA3_STITCHED_GEGLU_DOWN_JSON: {args.result_json}")
        raise SystemExit(0 if result.status == "STITCHED_GEGLU_DOWN_PASS" else 1)
    parser.error(
        "pass --self-test, --print-plan, --print-projection-core-mlir, "
        "--print-rms-qkv-mlir, --print-ingress-mlir, --print-attention-o-mlir, "
        "--print-post-attention-residual-mlir, --print-ffn-gate-up-mlir, "
        "--print-geglu-down-mlir, "
        "--parse-projection-core, --parse-rms-qkv, --parse-ingress, "
        "--parse-attention-o, --parse-post-attention-residual, --parse-ffn-gate-up, "
        "--parse-geglu-down, "
        "--compile-projection-core, --compile-rms-qkv, "
        "--compile-ingress, --compile-attention-o, --compile-post-attention-residual, "
        "--compile-ffn-gate-up, --compile-geglu-down, "
        "--run-ingress-hardware, "
        "--run-attention-o-hardware, --run-post-attention-residual-hardware, "
        "--run-ffn-gate-up-hardware, or --run-geglu-down-hardware"
    )


if __name__ == "__main__":
    main()
