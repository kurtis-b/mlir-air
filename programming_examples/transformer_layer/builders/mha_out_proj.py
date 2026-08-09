# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fused multi-head attention plus output projection, with optional causal mask.

CONTRACT
    ``build_mha_out_proj_module(...)`` returns an ``air.ir.Module`` with one
    function ``mha_out_proj`` computing

        y = softmax(Q K^T / sqrt(d) [+ causal mask]) V  @  W_o

    over a SEQ-FIRST layout: ``q`` is ``[seq_len, num_heads * head_dim]``,
    ``k`` and ``v`` are ``[seq_len, num_kv_heads * head_dim]``, ``w_o`` is
    ``[emb_dim, emb_dim]`` with ``emb_dim = num_heads * head_dim``, and ``y`` is
    ``[seq_len, emb_dim]``.

    The signature is built by ``mha_out_proj_arg_layout`` rather than written
    out, because whether there is an f32 GEMM scratch depends on which method
    the registry picks for the projection. Call
    ``mha_out_proj_device_inputs`` to build the host-side argument list; do not
    hand-order it. Every argument the host fills precedes the single argument it
    reads back (``y``), which is the layout ``XRTRunner.run_test`` assumes when
    it appends output placeholders after ``inputs``.

THE THREE SEAMS (porting convention 5)
    iron's ``mha_out_proj/design.py`` is 1350 lines against ``aie.iron``
    ObjectFifo / Worker / Runtime. Phase A split ``encoder.cc`` into three files
    by role; this splits the same way:

        builders/mha_attention.py   attention staging  -- the FlashAttention
                                    design point, its kernel flags, its oracle
        builders/o_proj.py          O-projection staging -- registry lookup,
                                    the GEMM sub-kernel, its oracle
        builders/mha_out_proj.py    the entry layer -- signature, wiring,
                                    the composed oracle (this file)

    Neither staging module imports the other. This one imports both and owns
    nothing but the composition.

WHAT IS FUSED, AND WHAT THAT BUYS
    One ELF, one dispatch: the attention launches and the projection launches
    are stitched into a single ``func.func``, and ``attn_out`` is an ordinary
    DDR buffer both halves address. There is no host-side step between them and
    no repack -- the seq-first FlashAttention variant writes exactly the
    ``[seq_len, emb_dim]`` the GEMM reads. See ``mha_attention.py`` on why the
    heads-first variant would need a transpose here for no numerical gain.

ACCURACY, AND WHY IT IS LOOSER THAN A GEMM
    FlashAttention chains two BFP16-emulated MMAs plus a bf16 online softmax,
    giving ``mean_rel_L1`` around 3.9e-2 on its own -- about 4x the GEMM tier.
    The registry holds it at ``rtol = 1.6e-2`` with ``atol = 1e-1``
    (``kernel_registry/details/FlashAttention_bf16.md``), and ``1e-1`` is the
    loosest ``atol`` anywhere in the registry. This operator inherits that
    datapath and adds a projection on top, so it cannot be tighter than
    FlashAttention alone. If it ever needs MORE than ``atol = 1e-1`` to pass,
    the answer is a defect report, not a wider tolerance.

FOOTGUNS
    - ``attn_out`` is a real DDR buffer in the signature, not an internal
      allocation. It is handed to the device zero-filled and is NOT compared
      against anything -- the gate is on ``y``. Reading it back is legitimate
      for debugging, but it is staging, not an output. Same disposition as
      ``ffn.py``'s ``h`` and ``a``.
    - The device stages ``attn_out`` in bf16 between the halves. The reference
      below is FP32 end to end and does NOT reproduce that rounding, and that
      gap is a real term in the measured error. Rounding it in the reference
      would make the check agree with the device for the wrong reason.
    - Both external objects must be in the working directory when aiecc links:
      ``attn_npu2.o`` (built with tile-shape ``-D`` flags that MUST match this
      design, or the kernel hangs) and the resolved method's ``mm_*.o``.
      ``compile_mha_out_proj_kernels`` builds both.
    - The backend options are FlashAttention's, not the GEMM's:
      ``omit_pingpong="all"`` and ``runtime_loop_tiling_sizes=[1, 1]``, which is
      what the shipped llama-3.2-1B prefill uses for this design. They are
      exported as ``MHA_OUT_PROJ_RUNNER_KWARGS`` rather than left to the caller,
      because the attention half does not place without them and the failure is
      a placement error a long way from its cause.
"""

import os
import sys

import numpy as np

_PROJ_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
for _p in (_PROJ_ROOT, os.path.join(_PROJ_ROOT, "llms")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared.infra.stitching import FuncArg, KernelSlice, stitch_elf  # noqa: E402

from builders.mha_attention import (  # noqa: E402
    ATTENTION_EXTERN_SYMS,
    attention_config,
    build_attention_ir,
    chunked_attention_reference,
    compile_attention_kernel,
    describe_attention_config,
)
from builders.o_proj import (  # noqa: E402
    build_o_proj_ir,
    compile_o_proj_kernel,
    describe_o_proj_spec,
    o_proj_extern_syms,
    o_proj_gemm_spec,
    o_proj_reference,
)

# Backend settings this design needs. FlashAttention's, because the attention
# half is the constrained one:
#   runtime_loop_tiling_sizes [1, 1]: LOAD-BEARING, and measured. At [2, 2] this
#     design compiles and then HANGS on hardware -- ERT_CMD_STATE_TIMEOUT, 3/3
#     at 4096x768 twelve heads non-causal, against 3/3 clean passes at [1, 1]
#     (0 / 3,145,728 mismatches, mean_rel_L1 5.3348e-02, atol_required
#     8.7061e-03 against the row's atol 2.5e-02). The launch grid is 2-D (Q
#     iterations x head groups) and each iteration's shim BDs are already one
#     transfer deep; [2, 2] is the BD-ID recycling the standalone GEMM-only
#     operators want, and it re-tiles the attention runtime loop as well.
#     `[2026-08-08]` Do not "simplify" this to match the GEMM preset on the
#     grounds that the lowered IR is identical between the two. It is identical
#     -- aircc's aie.air.mlir matches op-for-op, because air-opt-shim-dma-bds
#     early-exits with no shim-level scf.for to tile -- and the setting is
#     decisive anyway. That inference was drawn, written down, and refuted by
#     the run above. See agents/probes/probe_backend_preset_hardware.py.
#   omit_pingpong "all": NOT load-bearing at this shape, kept as shipped. The
#     comment here used to say ping-pong "does not fit L1 and the design fails
#     to place"; measured, it places, runs, and returns statistics byte-identical
#     to ping-pong off. It may still matter at a shape this has not been run at
#     -- causal, or a different head count -- so it stays, but it is not the
#     reason one ELF cannot serve both halves. The tiling is.
#   output_format "elf": multi-segment design -- one aie.device per air.launch.
#     The xclbin path names a single instruction blob on the aircc command line,
#     so a second segment collides on it.
#   omit_while_true_loop False: shared-buffer mode requires the loop.
MHA_OUT_PROJ_RUNNER_KWARGS = {
    "omit_pingpong": "all",
    "runtime_loop_tiling_sizes": [1, 1],
    "output_format": "elf",
    "omit_while_true_loop": False,
}


def mha_out_proj_config(
    seq_len,
    head_dim,
    num_heads,
    num_kv_heads=None,
    causal=False,
    parallel_seq=256,
    parallel_heads=2,
    kv_seq_tile=None,
    cascade_stages=4,
    num_q_tiles=None,
    o_proj_acc_depth=None,
    gemm_spec_fn=None,
):
    """Resolve both halves' configuration without building anything.

    Returns ``(attn_cfg, gemm_spec, gemm_source)``. Split out from the builder
    so ``opcheck.py`` can compile the external kernels and record the resolved
    spec in the results artifact without building the module twice.
    """
    attn_cfg = attention_config(
        seq_len,
        head_dim,
        num_heads,
        num_kv_heads=num_kv_heads,
        causal=causal,
        parallel_seq=parallel_seq,
        parallel_heads=parallel_heads,
        kv_seq_tile=kv_seq_tile,
        cascade_stages=cascade_stages,
        num_q_tiles=num_q_tiles,
    )
    gemm_spec, gemm_source = o_proj_gemm_spec(
        seq_len,
        attn_cfg["emb_dim"],
        o_proj_acc_depth=o_proj_acc_depth,
        gemm_spec_fn=gemm_spec_fn,
    )
    return attn_cfg, gemm_spec, gemm_source


def compile_mha_out_proj_kernels(attn_cfg, gemm_spec):
    """Build both external objects into the CWD, where aiecc's linker looks."""
    compile_attention_kernel(attn_cfg)
    compile_o_proj_kernel(gemm_spec)


def mha_out_proj_arg_layout(attn_cfg, gemm_spec):
    """The combined function's signature and the index of each named buffer.

    Returns ``(args, idx)`` where ``args`` is the ordered ``list[FuncArg]`` and
    ``idx`` maps a name (``q``, ``k``, ``v``, ``attn_out``, ``w_o``, ``y_f32``,
    ``y``) to its position. ``y_f32`` is present only for a method that needs an
    f32 scratch, which is why this is computed rather than written out -- the
    same bug class ``alloc_gemm_scratch`` exists to kill.

    ``y`` is last, and it is the only buffer the host reads back.
    """
    seq_len = attn_cfg["seq_len"]
    emb_dim = attn_cfg["emb_dim"]
    kv_emb_dim = attn_cfg["kv_emb_dim"]

    args, idx = [], {}

    def add(name, type_str):
        idx[name] = len(args)
        args.append(FuncArg(f"%arg{len(args)}", type_str))

    add("q", f"memref<{seq_len}x{emb_dim}xbf16>")
    add("k", f"memref<{seq_len}x{kv_emb_dim}xbf16>")
    add("v", f"memref<{seq_len}x{kv_emb_dim}xbf16>")
    add("attn_out", f"memref<{seq_len}x{emb_dim}xbf16>")
    add("w_o", f"memref<{emb_dim}x{emb_dim}xbf16>")
    if gemm_spec["needs_f32_scratch"]:
        add("y_f32", f"memref<{seq_len}x{emb_dim}xf32>")
    add("y", f"memref<{seq_len}x{emb_dim}xbf16>")
    return args, idx


def mha_out_proj_device_inputs(q, k, v, w_o, attn_cfg, gemm_spec):
    """Every argument the host fills, in signature order, ``y`` excluded.

    The staging buffer is zero-filled. Pass the result straight to
    ``XRTRunner.run_test(inputs=...)`` -- keeping this beside
    ``mha_out_proj_arg_layout`` is what stops a caller from getting the order
    right today and wrong the next time the registry's method changes.
    """
    seq_len = attn_cfg["seq_len"]
    emb_dim = attn_cfg["emb_dim"]
    inputs = [
        q,
        k,
        v,
        np.zeros((seq_len, emb_dim), dtype=q.dtype),  # attn_out staging
        w_o,
    ]
    if gemm_spec["needs_f32_scratch"]:
        inputs.append(np.zeros((seq_len, emb_dim), dtype=np.float32))
    return inputs


def build_mha_out_proj_module(
    seq_len,
    head_dim,
    num_heads,
    num_kv_heads=None,
    causal=False,
    parallel_seq=256,
    parallel_heads=2,
    kv_seq_tile=None,
    cascade_stages=4,
    num_q_tiles=None,
    o_proj_acc_depth=None,
    o_herd_m=None,
    o_herd_n=None,
    gemm_spec_fn=None,
):
    """Build the fused attention + output-projection ELF.

    Args:
        seq_len: Q and K/V sequence length (self-attention, so one value).
        head_dim: per-head K/V dimension. 64 throughout this operator's case
            matrix; see ``mha_attention.py`` on why 128 is rejected here.
        num_heads: query heads. ``num_heads * head_dim`` is ``emb_dim``, and
            ``(seq_len, emb_dim, emb_dim)`` must be in the GEMM registry or
            ``gemm_spec_fn`` must supply it.
        num_kv_heads: key/value heads for GQA. ``None`` means MHA.
        causal: autoregressive masking. Requires ``q_seq_tile == kv_seq_tile``.
        parallel_seq: iron's name for the Q rows one launch iteration covers.
        parallel_heads: iron's name for the heads one segment instance covers.
        kv_seq_tile, cascade_stages, num_q_tiles: the remaining FlashAttention
            knobs; see ``mha_attention.attention_config``.
        o_proj_acc_depth: iron's name for the projection's memory-tile
            accumulation depth, i.e. the GEMM's ``tile_k_l2``. ``None`` (the
            default, and the normal setting) reads it from ``kernel_registry``.
            A value here is an explicit override, printed and recorded.
        o_herd_m, o_herd_n: projection GEMM herd. ``None`` (the normal setting)
            uses the herd the resolved row's tiles were measured at; a value
            here is an explicit override.
        gemm_spec_fn: ``(m, k, n) -> spec`` escape hatch for a shape the
            registry has not measured, the same hook
            ``rms_qkv_qknorm_rope_multi.py`` ships. Leave ``None`` in normal
            use.

    Returns:
        air.ir.Module with one function ``mha_out_proj``.
    """
    attn_cfg, gemm_spec, gemm_source = mha_out_proj_config(
        seq_len,
        head_dim,
        num_heads,
        num_kv_heads=num_kv_heads,
        causal=causal,
        parallel_seq=parallel_seq,
        parallel_heads=parallel_heads,
        kv_seq_tile=kv_seq_tile,
        cascade_stages=cascade_stages,
        num_q_tiles=num_q_tiles,
        o_proj_acc_depth=o_proj_acc_depth,
        gemm_spec_fn=gemm_spec_fn,
    )
    emb_dim = attn_cfg["emb_dim"]

    args, idx = mha_out_proj_arg_layout(attn_cfg, gemm_spec)

    print(f"  [1/2] FlashAttention {describe_attention_config(attn_cfg)}...")
    attn_ir = build_attention_ir(attn_cfg)

    print(
        f"  [2/2] O-projection GEMM "
        f"{describe_o_proj_spec(seq_len, emb_dim, gemm_spec, gemm_source, o_herd_m, o_herd_n)}..."
    )
    o_ir = build_o_proj_ir(seq_len, emb_dim, gemm_spec, o_herd_m, o_herd_n)

    # fused-cast is {A, B, C-f32-scratch, D-bf16-out} over two launches; drain
    # is {A, B, C-bf16-out} over one. Same branch ffn.py::_gemm_slice takes.
    if gemm_spec["needs_f32_scratch"]:
        o_arg_map = {0: idx["attn_out"], 1: idx["w_o"], 2: idx["y_f32"], 3: idx["y"]}
    else:
        o_arg_map = {0: idx["attn_out"], 1: idx["w_o"], 2: idx["y"]}

    slices = [
        KernelSlice(
            attn_ir,
            "at",
            {0: idx["q"], 1: idx["k"], 2: idx["v"], 3: idx["attn_out"]},
            extern_syms=set(ATTENTION_EXTERN_SYMS),
        ),
        KernelSlice(o_ir, "op", o_arg_map, extern_syms=o_proj_extern_syms(gemm_spec)),
    ]

    module = stitch_elf(
        "mha_out_proj",
        args,
        slices,
        debug_dump_path="/tmp/debug_mha_out_proj.mlir",
    )
    print(f"  Module: {len(str(module).splitlines())} lines, parsed OK")
    return module


def mha_out_proj_reference(
    q,
    k,
    v,
    w_o,
    num_heads,
    num_kv_heads=None,
    causal=False,
    query_block_size=256,
):
    """FP32 reference for the fused operator, rounded once at the end.

    Chunked FP32 attention at EVERY sequence length (see
    ``mha_attention.chunked_attention_reference`` for why iron's
    ``seq_len >= 16384`` branch is dropped), then the projection in f32, then a
    single rounding to ``q``'s dtype.

    FP32 end to end: the device stages the attention output in bf16 between the
    two halves and this does not, so the difference between them is MEASURED
    rather than defined away. That is the same rule ``ffn.py`` follows for its
    intermediate activation and ``layer_norm.py`` for one-pass variance.

    Returns ``[seq_len, num_heads * head_dim]`` in ``q``'s dtype.
    """
    attn = chunked_attention_reference(
        q,
        k,
        v,
        num_heads,
        num_kv_heads=num_kv_heads,
        causal=causal,
        query_block_size=query_block_size,
    )
    return o_proj_reference(attn, w_o).astype(q.dtype)
