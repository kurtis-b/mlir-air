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
from dataclasses import dataclass
from pathlib import Path
import sys

from gemma3_stitching import StitchSpec, stitch_module_text


DEFAULT_MODEL = "gemma3-1b"
DEFAULT_FUNCTION_NAME = "gemma3_decode_qkv_projection_core"
DEFAULT_RMS_QKV_FUNCTION_NAME = "gemma3_decode_rms_qkv_projection_core"
DEFAULT_INGRESS_FUNCTION_NAME = "gemma3_decode_ingress_rms_qkv_qknorm_rope"
DEFAULT_OUTPUT_FORMAT = "elf"
DEFAULT_FUSED_DQP_OBJECT = Path(__file__).with_name("build_peano") / "fused_dqp.o"
DEFAULT_ROPE_OBJECT = Path(__file__).with_name("build_peano") / "rope_halfsplit.o"

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
    "memref<5x256xbf16>",  # padded normalized activation
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
    "memref<5x256xbf16>",  # padded normalized activation
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


def build_decode_ingress_plan(model_variant: str = DEFAULT_MODEL) -> DecodeStitchPlan:
    stages = (
        DecodeStitchStage(
            "rmsnorm_pad_activation",
            "implemented-compile-pass-bridge",
            "layer_input:1x1152 + norm_weight:1152",
            "activation_padded:5x256",
            "Padded RMSNorm bridge removes host activation packing before FusedDQP.",
        ),
        DecodeStitchStage(
            "qkv_projection_core",
            "implemented-compile-pass-stitch",
            "activation_padded:5x256 + q/k/v packed static BOs",
            "q:32x32 k:8x32 v:8x32",
            "Uses full-col-block FusedDQP so host col-block accumulation is not in the timed path.",
        ),
        DecodeStitchStage(
            "projection_qk_views",
            "implemented-compile-pass-bridge",
            "q:32x32 k:8x32 + q/k norm weights",
            "q_norm:4x256 k_norm:1x256",
            "Zero-copy collapse/expand view is fused into Q/K weighted RMSNorm.",
        ),
        DecodeStitchStage(
            "qk_norm_rope",
            "implemented-compile-pass-stitch",
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
    for path in (dataflow, herd_sweep):
        while path in sys.path:
            sys.path.remove(path)
    sys.path.insert(0, dataflow)
    sys.path.insert(0, herd_sweep)


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
            "direct",
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
            "direct",
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
            "direct",
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
    """Build combined MLIR text for RMSNorm-padding plus Q/K/V projections."""
    _activate_builder_paths()
    from gemma3_padded_rms_norm import build_module as build_padded_rms

    rms_ir = str(build_padded_rms())
    q_ir, k_ir, v_ir = _projection_irs(object_file)
    return stitch_module_text(
        function_name=function_name,
        arg_types=RMS_QKV_ARG_TYPES,
        specs=(
            StitchSpec(rms_ir, "r", {0: 0, 1: 1, 2: 2}),
            StitchSpec(q_ir, "q", {0: 3, 1: 2, 2: 6}),
            StitchSpec(k_ir, "k", {0: 4, 1: 2, 2: 7}),
            StitchSpec(v_ir, "v", {0: 5, 1: 2, 2: 8}),
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
    from gemma3_padded_rms_norm import build_module as build_padded_rms
    from gemma3_projection_qk_norm import build_module as build_qk_norm
    from ml_dtypes import bfloat16
    from rope_halfsplit import build_module as build_rope

    rms_ir = str(build_padded_rms())
    q_ir, k_ir, v_ir = _projection_irs(object_file)
    q_norm_ir = str(build_qk_norm(32, 32, 4, 256))
    k_norm_ir = str(build_qk_norm(8, 32, 1, 256))
    q_rope_ir = str(build_rope(4, 256, bfloat16, 4, str(rope_object_file)))
    k_rope_ir = str(build_rope(1, 256, bfloat16, 1, str(rope_object_file)))
    return stitch_module_text(
        function_name=function_name,
        arg_types=INGRESS_ARG_TYPES,
        specs=(
            StitchSpec(rms_ir, "r", {0: 0, 1: 1, 2: 2}),
            StitchSpec(q_ir, "q", {0: 3, 1: 2, 2: 6}),
            StitchSpec(k_ir, "k", {0: 4, 1: 2, 2: 7}),
            StitchSpec(v_ir, "v", {0: 5, 1: 2, 2: 8}),
            StitchSpec(q_norm_ir, "qn", {0: 6, 1: 9, 2: 11}),
            StitchSpec(k_norm_ir, "kn", {0: 7, 1: 10, 2: 12}),
            StitchSpec(q_rope_ir, "rq", {0: 11, 1: 13, 2: 15}),
            StitchSpec(k_rope_ir, "rk", {0: 12, 1: 14, 2: 16}),
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
    """Build and parse the stitched RMSNorm-padding plus Q/K/V MLIR module."""
    return _parse_module(build_rms_qkv_text(object_file=object_file))


def build_ingress_module(
    *,
    object_file: Path = DEFAULT_FUSED_DQP_OBJECT,
    rope_object_file: Path = DEFAULT_ROPE_OBJECT,
):
    """Build and parse the full stitched decode ingress MLIR module."""
    return _parse_module(build_ingress_text(object_file=object_file, rope_object_file=rope_object_file))


def _compile_module(module, *, instance_name: str, output_binary_name: str):
    from air.backend.xrt import XRTBackend

    backend = XRTBackend(
        verbose=False,
        omit_while_true_loop=False,
        output_format=DEFAULT_OUTPUT_FORMAT,
        instance_name=instance_name,
        target_device="npu2",
        runtime_loop_tiling_sizes=[4, 4],
    )
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
    parser.add_argument("--parse-projection-core", action="store_true")
    parser.add_argument("--parse-rms-qkv", action="store_true")
    parser.add_argument("--parse-ingress", action="store_true")
    parser.add_argument("--compile-projection-core", action="store_true")
    parser.add_argument("--compile-rms-qkv", action="store_true")
    parser.add_argument("--compile-ingress", action="store_true")
    parser.add_argument("--object-file", type=Path, default=DEFAULT_FUSED_DQP_OBJECT)
    parser.add_argument("--rope-object-file", type=Path, default=DEFAULT_ROPE_OBJECT)
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
    parser.error(
        "pass --self-test, --print-plan, --print-projection-core-mlir, "
        "--print-rms-qkv-mlir, --print-ingress-mlir, --parse-projection-core, "
        "--parse-rms-qkv, --parse-ingress, --compile-projection-core, "
        "--compile-rms-qkv, or --compile-ingress"
    )


if __name__ == "__main__":
    main()
