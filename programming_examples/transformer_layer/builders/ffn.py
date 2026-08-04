# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Staged feed-forward network: up-projection, GeLU, down-projection.

CONTRACT
    ``build_ffn_module(seq_len, emb_dim, ffn_dim, ...)`` returns an
    ``air.ir.Module`` with one function ``ffn`` computing

        y = gelu(x @ w_up) @ w_down

    where ``x`` is ``[seq_len, emb_dim]``, ``w_up`` is ``[emb_dim, ffn_dim]``
    and ``w_down`` is ``[ffn_dim, emb_dim]``. The signature is built by
    ``ffn_arg_layout`` rather than written out here, because how many memref
    arguments there are depends on which method the registry picks for each of
    the two GEMMs. Call ``ffn_device_inputs`` to build the host-side argument
    list; do not hand-order it.

    Every argument the host fills precedes the single argument it reads back
    (``y``), which is the layout ``XRTRunner.run_test`` assumes when it appends
    output placeholders after ``inputs``.

THE STAGING SEAM (porting convention 5)
    iron's ``ffn/design.py`` is 1096 lines carrying all three stages plus its
    own activation. The seam it already has internally is the boundary between
    the two projections and the activation between them, so that is where this
    splits: the activation stage is ``builders/gelu.py``, and what remains here
    is the composition -- resolve two registry specs, build two GEMMs, stitch
    five launches. Neither half needs to know the other's internals.

WHERE ``down_proj_depth`` WENT
    iron exposes a ``down_proj_depth`` knob for how deep the down-projection
    accumulates in the memory tile before draining. In AIR that quantity is the
    down-projection GEMM's ``tile_k_l2``, and it is measured per shape and
    stored in ``kernel_registry``. So it is not a parameter here: it is read
    from the registry with the rest of the tiling, per porting convention 9.
    ``build_ffn_module`` prints the resolved value so it is visible in a log.

FOOTGUNS
    - ``h`` and ``a`` (the up-projection output and its activation) are real DDR
      buffers in the signature, not internal allocations. They are handed to the
      device zero-filled and are NOT compared against anything -- the gate is on
      ``y``. Reading them back is legitimate for debugging but they are staging,
      not outputs.
    - The device stages ``h`` in bf16 between the two GEMMs, and the activation
      carries bf16 intermediates on top of that. The reference below is FP32
      end to end and reproduces NEITHER, deliberately. That gap is the dominant
      term in this operator's absolute error -- an order of magnitude above a
      single GEMM's -- and it is what ``atol`` is sized against in
      ``opcheck.py``. Rounding ``h`` in the reference would make the check agree
      with the device for the wrong reason.
    - Both GEMM objects and the GeLU object must be in the working directory
      when aiecc links: the registry method's ``mm_*.o`` and ``encoder_ffn.o``.
      ``encoder_ffn.o`` is encoder.cc's FFN half ONLY; the addnorm half would
      collide with ``addnorm_ffn.o`` on ``ffn_gelu_bf16``.
    - The two GEMMs may resolve to different methods. Their symbol suffixes and
      ``mm_*.o`` names differ accordingly, which is what lets both co-link.
"""

import os
import sys

import numpy as np
from ml_dtypes import bfloat16

_PROJ_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
for _p in (_PROJ_ROOT, os.path.join(_PROJ_ROOT, "llms")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared.builders.gemm_builder import (  # noqa: E402
    _build_gemm_module,
    gemm_registry_config,
)
from shared.infra.stitching import FuncArg, KernelSlice, stitch_elf  # noqa: E402

from builders.gelu import (  # noqa: E402
    GELU_SYMBOL,
    build_gelu_module,
    gelu_tanh_reference,
)


def ffn_gemm_specs(seq_len, emb_dim, ffn_dim, gemm_spec_fn=None):
    """Resolve both projections' methods and tiles, registry-first.

    Returns ``(up_spec, down_spec, source)`` with ``source`` either
    ``"registry"`` or ``"injected"``. Split out from the builder so
    ``opcheck.py`` can record what was actually used without building twice.
    """
    if gemm_spec_fn is not None:
        return (
            gemm_spec_fn(seq_len, emb_dim, ffn_dim),
            gemm_spec_fn(seq_len, ffn_dim, emb_dim),
            "injected",
        )
    return (
        gemm_registry_config(seq_len, emb_dim, ffn_dim, "bf16", "high"),
        gemm_registry_config(seq_len, ffn_dim, emb_dim, "bf16", "high"),
        "registry",
    )


def ffn_arg_layout(seq_len, emb_dim, ffn_dim, up_spec, down_spec):
    """The combined function's signature and the index of each named buffer.

    Returns ``(args, idx)`` where ``args`` is the ordered ``list[FuncArg]`` and
    ``idx`` maps a name (``x``, ``w_up``, ``h_f32``, ``h``, ``a``, ``w_down``,
    ``y_f32``, ``y``) to its position. The f32 scratch entries are present only
    for a method that needs one, which is why this is computed rather than
    written out -- the same bug class ``alloc_gemm_scratch`` exists to kill.

    ``y`` is last, and it is the only buffer the host reads back.
    """
    args, idx = [], {}

    def add(name, type_str):
        idx[name] = len(args)
        args.append(FuncArg(f"%arg{len(args)}", type_str))

    add("x", f"memref<{seq_len}x{emb_dim}xbf16>")
    add("w_up", f"memref<{emb_dim}x{ffn_dim}xbf16>")
    if up_spec["needs_f32_scratch"]:
        add("h_f32", f"memref<{seq_len}x{ffn_dim}xf32>")
    add("h", f"memref<{seq_len}x{ffn_dim}xbf16>")
    add("a", f"memref<{seq_len}x{ffn_dim}xbf16>")
    add("w_down", f"memref<{ffn_dim}x{emb_dim}xbf16>")
    if down_spec["needs_f32_scratch"]:
        add("y_f32", f"memref<{seq_len}x{emb_dim}xf32>")
    add("y", f"memref<{seq_len}x{emb_dim}xbf16>")
    return args, idx


def ffn_device_inputs(x, w_up, w_down, up_spec, down_spec):
    """Every argument the host fills, in signature order, ``y`` excluded.

    Staging buffers are zero-filled. Pass the result straight to
    ``XRTRunner.run_test(inputs=...)`` -- keeping this beside
    ``ffn_arg_layout`` is what stops a caller from getting the order right today
    and wrong the next time a registry method changes.
    """
    seq_len, emb_dim = x.shape
    ffn_dim = w_up.shape[1]
    inputs = [x, w_up]
    if up_spec["needs_f32_scratch"]:
        inputs.append(np.zeros((seq_len, ffn_dim), dtype=np.float32))
    inputs.append(np.zeros((seq_len, ffn_dim), dtype=x.dtype))  # h
    inputs.append(np.zeros((seq_len, ffn_dim), dtype=x.dtype))  # a
    inputs.append(w_down)
    if down_spec["needs_f32_scratch"]:
        inputs.append(np.zeros((seq_len, emb_dim), dtype=np.float32))
    return inputs


def _gemm_slice(spec, ir, prefix, in_idx, w_idx, scratch_idx, out_idx):
    """One GEMM's ``KernelSlice``, wired for its method's operand count.

    fused-cast is ``{in, w, C-f32-scratch, bf16-out}`` over two launches; drain
    is ``{in, w, bf16-out}`` over one. The extern set carries the method's
    suffixed ``mm.o`` symbols so both variants can co-link.
    """
    sfx = spec["sym_suffix"]
    externs = {
        "@matmul_bf16",
        "@op_has_no_registered_library_name" + sfx,
        "@zero_f32_mn" + sfx,
        "@f32_to_bf16_mn" + sfx,
    }
    if spec["needs_f32_scratch"]:
        arg_map = {0: in_idx, 1: w_idx, 2: scratch_idx, 3: out_idx}
    else:
        arg_map = {0: in_idx, 1: w_idx, 2: out_idx}
    return KernelSlice(ir, prefix, arg_map, extern_syms=externs)


def build_ffn_module(
    seq_len,
    emb_dim,
    ffn_dim,
    herd_m=8,
    herd_n=4,
    gelu_herd_x=8,
    gelu_tile_n=None,
    gemm_spec_fn=None,
):
    """Build the staged FFN: up-projection, GeLU, down-projection.

    Args:
        seq_len, emb_dim, ffn_dim: activation and weight shapes. Both
            ``(seq_len, emb_dim, ffn_dim)`` and ``(seq_len, ffn_dim, emb_dim)``
            must be in the GEMM registry, or ``gemm_spec_fn`` must supply them.
        herd_m, herd_n: GEMM herd, as the registry tiles assume.
        gelu_herd_x: AIE columns for the activation launch.
        gelu_tile_n: elements per ``ffn_gelu_bf16`` call; defaults to one row.
        gemm_spec_fn: ``(m, k, n) -> spec`` escape hatch for an unmeasured
            shape, the same hook ``rms_qkv_qknorm_rope_multi.py`` ships. An
            injected spec is recorded in the results artifact. Leave ``None`` in
            normal use.

    Returns:
        air.ir.Module with one function ``ffn``.
    """
    up_spec, down_spec = ffn_gemm_specs(seq_len, emb_dim, ffn_dim, gemm_spec_fn)[:2]
    source = "injected" if gemm_spec_fn is not None else "registry"

    args, idx = ffn_arg_layout(seq_len, emb_dim, ffn_dim, up_spec, down_spec)

    def _describe(label, spec, m, k, n):
        print(
            f"  {label} {m}x{k}x{n} ({spec['method']}, {source}, "
            f"tile_m={spec['tile_m']} tile_k_l2={spec['tile_k_l2']} "
            f"tile_k_l1={spec['tile_k_l1']} tile_n={spec['tile_n']})..."
        )

    _describe("[1/3] up-projection", up_spec, seq_len, emb_dim, ffn_dim)
    up_ir = str(
        _build_gemm_module(
            seq_len,
            emb_dim,
            ffn_dim,
            up_spec["tile_m"],
            up_spec["tile_k_l2"],
            up_spec["tile_k_l1"],
            up_spec["tile_n"],
            herd_m,
            herd_n,
            **up_spec["build_kwargs"],
        )
    )

    print(f"  [2/3] GeLU {seq_len}x{ffn_dim} (tanh approximation)...")
    gelu_ir = str(
        build_gelu_module(
            seq_len, ffn_dim, bfloat16, herd_x=gelu_herd_x, tile_n=gelu_tile_n
        )
    )

    # The memory-tile accumulation depth iron calls down_proj_depth.
    _describe("[3/3] down-projection", down_spec, seq_len, ffn_dim, emb_dim)
    down_ir = str(
        _build_gemm_module(
            seq_len,
            ffn_dim,
            emb_dim,
            down_spec["tile_m"],
            down_spec["tile_k_l2"],
            down_spec["tile_k_l1"],
            down_spec["tile_n"],
            herd_m,
            herd_n,
            **down_spec["build_kwargs"],
        )
    )

    slices = [
        _gemm_slice(
            up_spec, up_ir, "u", idx["x"], idx["w_up"], idx.get("h_f32"), idx["h"]
        ),
        KernelSlice(
            gelu_ir,
            "gl",
            {0: idx["h"], 1: idx["a"]},
            extern_syms={"@" + GELU_SYMBOL},
        ),
        _gemm_slice(
            down_spec,
            down_ir,
            "d",
            idx["a"],
            idx["w_down"],
            idx.get("y_f32"),
            idx["y"],
        ),
    ]

    module = stitch_elf(
        "ffn",
        args,
        slices,
        debug_dump_path="/tmp/debug_ffn.mlir",
    )
    print(f"  Module: {len(str(module).splitlines())} lines, parsed OK")
    return module


def ffn_reference(x, w_up, w_down):
    """FP32 reference for ``gelu(x @ w_up) @ w_down``, rounded once at the end.

    FP32 end to end, with the tanh-approximation GeLU (see ``gelu.py``). The
    device stages the intermediate activation in bf16 twice over -- once as the
    up-projection's cast, once inside the activation kernel -- and this does
    not, so the difference between them is measured rather than defined away.
    That is the same rule ``layer_norm.py`` follows for one-pass variance.
    """
    h = x.astype(np.float32) @ w_up.astype(np.float32)
    a = gelu_tanh_reference(h.astype(np.float32))
    return (a.astype(np.float32) @ w_down.astype(np.float32)).astype(x.dtype)
