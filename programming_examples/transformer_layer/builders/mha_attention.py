# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Attention staging for ``mha_out_proj``: the FlashAttention half.

CONTRACT
    ``attention_config(...)`` validates one FlashAttention design point and
    returns the knob set as a plain dict. ``build_attention_ir(cfg)`` turns that
    into the sub-kernel MLIR text the composition stitches, and
    ``compile_attention_kernel(cfg)`` builds the ``attn_npu2.o`` those launches
    link against. ``chunked_attention_reference(...)`` is the FP32 oracle.

    Nothing here builds a device design of its own. The design is
    ``flash_attention/kernel_fusion_based/attn_npu2_seqfirst.py``, imported and
    parameterised -- Phase C3 composes that verified object rather than
    re-deriving it, and Goal 1 will later change FlashAttention's behaviour, so
    a second copy here would be a second thing to keep in step.

WHY SEQ-FIRST AND NOT HEADS-FIRST
    Both harnesses drive the same ``attn_npu2.o`` and are verified bit-identical
    (``kernel_registry/details/FlashAttention_bf16.md``). They differ only in the
    L3 layout: heads-first writes ``[heads * dv_chunks, lq, dv_tile]``, seq-first
    writes ``[lq, num_heads * dv]``.

    ``[lq, num_heads * dv]`` IS the O-projection's ``A`` operand -- the same
    bytes, no repack. Composing the heads-first variant instead would need a
    transpose launch between the two halves, for no numerical gain. This is the
    ``opt-layout-alignment`` argument, applied at build time rather than as a
    later optimisation.

THE KNOB NAMES, AND WHAT THEY WERE CALLED IN iron
    iron's ``mha_out_proj`` exposes ``parallel_seq`` / ``q_seq_tile`` /
    ``kv_seq_tile`` / ``emb_tile`` / ``parallel_heads``. The AIR design's names
    for the same quantities:

        parallel_seq   -> lqp                  Q rows per launch iteration
        q_seq_tile     -> lqp // num_q_tiles   Q rows resident in one core
        kv_seq_tile    -> lkp                  K/V rows per chunk, and the
                                               dk/dv TILE the kernel is built for
        emb_tile       -> dk == dv == lkp      (shared-buffer mode)
        parallel_heads -> num_heads_per_unroll heads per segment instance

    ``cascade_stages`` (AIR ``num_cascade_stages``) has no iron counterpart; it
    is the online-softmax merge depth, and it is the herd's row count.

FOOTGUNS
    - The ``-D`` flags baked into ``attn_npu2.o`` are PER TILE, not per launch:
      ``lqp`` is ``q_seq_tile``, and ``dk``/``dv`` are the ``lkp``-sized TILE
      while ``dk_full``/``dv_full`` are the full head dimension. They must match
      the L1 buffer shapes this design emits or the kernel does not fail -- it
      HANGS, as ``ERT_CMD_STATE_TIMEOUT``. ``compile_attention_kernel`` derives
      them from the same config the builder uses so they cannot drift apart, and
      passes ``force=True`` because a shared working directory may hold an
      ``attn_npu2.o`` built for another shape.
    - ``copy_O_tile_rows`` (behind ``-DCAUSAL_ROW_HELPERS``) is NUMERICALLY A
      NO-OP: it reads every element of the O tile and writes it straight back.
      That is the point. It exists so a KV block entirely above the causal
      diagonal -- which runs no matmul -- still completes the consuming DMA's
      buffer descriptor. Deleting it as dead code hangs the design with
      ``ERT_CMD_STATE_TIMEOUT``. See ``attn_npu2.cc`` for the original note.
      This composition builds the helpers into the object for every causal
      variant. It does not call them from the MLIR, because the masking path it
      composes -- ``apply_causal_mask`` -- fills a wholly-masked score tile with
      -inf and lets the matmul run anyway, so no O tile is ever left untouched.
      The helpers are the entry points a block-skipping variant needs; keeping
      them linked is what lets one be added without re-deriving the flag set.
    - Causal masking pins ``q_seq_tile == kv_seq_tile``. The device compares
      BLOCK indices, not element positions, so a Q tile whose row span does not
      line up with a K chunk's would mask the wrong triangle -- silently, with a
      correctly-shaped result. ``attention_config`` asserts it.
    - The reference below is FP32 and chunked at EVERY sequence length.
      ``iron/operators/mha_out_proj/reference.py`` computes bf16 SDPA below
      ``seq_len 16384`` and switches to FP32 chunked attention at and above it,
      so its own precision changes across the ladder and a tolerance calibrated
      at one end means something different at the other. The branch is dropped
      here deliberately; do not restore it.
    - ``head_dim = 128`` FlashAttention has been flaky (hang or NaN) on some
      NPU2 setups -- the registry rows record it. This operator's case matrix is
      ``head_dim = 64`` throughout, so ``attention_config`` rejects anything
      that would put it on the ``dv_chunks > 1`` path rather than letting a
      mis-shaped call find that out on hardware.
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

# Every symbol attn_npu2.o exports that the design may call. Passed to
# `stitch_elf` as this slice's `extern_syms` so they survive the per-slice
# prefix rename and still resolve against the object at link time. Listing a
# symbol the design does not emit is harmless -- the set only suppresses
# renaming -- and listing one too FEW renames a call away from its declaration,
# which fails to link rather than failing quietly.
ATTENTION_EXTERN_SYMS = frozenset(
    {
        "@zero_fill_g_bf16",
        "@zero_fill_gp_bf16",
        "@zero_fill_sp_bf16",
        "@neg_inf_fill_up_bf16",
        "@matmul_a_b_bf16",
        "@matmul_g_b_bf16",
        "@fused_softmax",
        "@maximum_up_u_bf16",
        "@exp_up_minus_u",
        "@mul_r_gp",
        "@accum_sp_r_s",
        "@vector_copy_32elems",
        "@copy_tile",
        "@div_gp_sp",
        "@add_gp_g",
        "@apply_causal_mask",
        # Causal row helpers (-DCAUSAL_ROW_HELPERS). Linked into the object for
        # every causal variant; see the module footguns for why the composed
        # masking path does not call them and why they stay anyway.
        "@copy_O_tile_rows",
        "@store_row_value",
        "@copy_row_values",
    }
)

# NPU2's physical array. Columns are consumed by num_q_tiles * parallel_heads
# and rows by cascade_stages; the registry's FlashAttention sweep found only 2
# of 8 candidate 32-tile configs place at all, so these are hard limits, not
# guidance.
_NPU2_COLUMNS = 8
_NPU2_ROWS = 4


def attention_config(
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
):
    """Validate one FlashAttention design point and return its knobs.

    Args:
        seq_len: Q and K/V sequence length. This operator is self-attention, so
            they are the same; causal masking requires it regardless.
        head_dim: per-head K and V dimension. Must equal ``kv_seq_tile``
            (shared-buffer mode) -- see the module footguns on ``head_dim=128``.
        num_heads: query heads. ``num_heads * head_dim`` is the model width.
        num_kv_heads: key/value heads for GQA. ``None`` means MHA.
        causal: autoregressive masking.
        parallel_seq: iron's name for ``lqp``, the Q rows one launch iteration
            covers.
        parallel_heads: iron's name for ``num_heads_per_unroll``, the heads one
            segment instance covers. Also the physical-column multiplier.
        kv_seq_tile: iron's ``lkp``. Defaults to ``head_dim``.
        cascade_stages: online-softmax merge depth; the herd's row count.
        num_q_tiles: Q tiles per launch iteration; the herd's column count.
            Defaults to ``parallel_seq // kv_seq_tile``, which is the only value
            causal masking permits and the one every registry row uses.

    Returns:
        dict of the validated knobs, plus the derived ``emb_dim`` and
        ``gqa_group_size``. Pass it to ``build_attention_ir`` and
        ``compile_attention_kernel`` -- both read the same dict, which is what
        keeps the kernel's ``-D`` flags and the design's L1 shapes in step.
    """
    if kv_seq_tile is None:
        kv_seq_tile = head_dim
    if num_kv_heads is None:
        num_kv_heads = num_heads
    if num_q_tiles is None:
        num_q_tiles = parallel_seq // kv_seq_tile

    if head_dim != kv_seq_tile:
        raise ValueError(
            f"head_dim ({head_dim}) must equal kv_seq_tile ({kv_seq_tile}). "
            f"A larger head_dim puts the design on the dv_chunks > 1 path, "
            f"which the seq-first harness does not support and which the "
            f"registry records as flaky (hang or NaN) on some NPU2 setups. "
            f"This operator's case matrix is head_dim = 64 throughout."
        )
    if seq_len % parallel_seq:
        raise ValueError(
            f"seq_len ({seq_len}) must be divisible by parallel_seq "
            f"({parallel_seq})"
        )
    if parallel_seq % num_q_tiles:
        raise ValueError(
            f"parallel_seq ({parallel_seq}) must be divisible by num_q_tiles "
            f"({num_q_tiles})"
        )
    if seq_len % (kv_seq_tile * cascade_stages):
        raise ValueError(
            f"seq_len ({seq_len}) must be divisible by kv_seq_tile * "
            f"cascade_stages ({kv_seq_tile * cascade_stages})"
        )
    if num_heads % parallel_heads:
        raise ValueError(
            f"num_heads ({num_heads}) must be divisible by parallel_heads "
            f"({parallel_heads})"
        )
    if num_heads % num_kv_heads:
        raise ValueError(
            f"num_heads ({num_heads}) must be divisible by num_kv_heads "
            f"({num_kv_heads})"
        )

    columns = num_q_tiles * parallel_heads
    if columns > _NPU2_COLUMNS:
        raise ValueError(
            f"num_q_tiles * parallel_heads = {columns} exceeds NPU2's "
            f"{_NPU2_COLUMNS} columns"
        )
    if cascade_stages > _NPU2_ROWS:
        raise ValueError(
            f"cascade_stages ({cascade_stages}) exceeds NPU2's {_NPU2_ROWS} rows"
        )

    q_seq_tile = parallel_seq // num_q_tiles
    if causal and q_seq_tile != kv_seq_tile:
        raise ValueError(
            f"causal masking requires q_seq_tile == kv_seq_tile, got "
            f"q_seq_tile={q_seq_tile}, kv_seq_tile={kv_seq_tile}. The device "
            f"masks by BLOCK index, so misaligned tiles mask the wrong triangle "
            f"and return a correctly-shaped wrong answer."
        )

    return {
        "seq_len": seq_len,
        "head_dim": head_dim,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "causal": causal,
        "parallel_seq": parallel_seq,
        "parallel_heads": parallel_heads,
        "kv_seq_tile": kv_seq_tile,
        "cascade_stages": cascade_stages,
        "num_q_tiles": num_q_tiles,
        "q_seq_tile": q_seq_tile,
        "emb_dim": num_heads * head_dim,
        "kv_emb_dim": num_kv_heads * head_dim,
        "gqa_group_size": num_heads // num_kv_heads,
    }


def describe_attention_config(cfg):
    """One-line summary of a config, for the run log."""
    return (
        f"{cfg['seq_len']}x{cfg['seq_len']} head_dim={cfg['head_dim']} "
        f"heads={cfg['num_heads']}q/{cfg['num_kv_heads']}kv "
        f"causal={cfg['causal']} parallel_seq={cfg['parallel_seq']} "
        f"parallel_heads={cfg['parallel_heads']} "
        f"num_q_tiles={cfg['num_q_tiles']} cascade_stages={cfg['cascade_stages']}"
    )


def compile_attention_kernel(cfg, force=True):
    """Build ``attn_npu2.o`` with the tile shapes ``cfg`` implies.

    The causal row helpers are always requested. They cost three symbols in the
    object and nothing at run time, and building them conditionally would make
    the object's contents depend on a flag the linker cannot see -- exactly the
    drift the ``force=True`` rebuild exists to avoid.
    """
    import shared.infra.external_kernels as ek

    ek.compile_attn_npu2(
        head_dim=cfg["head_dim"],
        lkp=cfg["kv_seq_tile"],
        lqp_tile=cfg["q_seq_tile"],
        force=force,
        causal_row_helpers=True,
    )


def build_attention_ir(cfg):
    """The FlashAttention sub-kernel's MLIR text, seq-first.

    Four launch operands, in order: ``q [seq, num_heads*head_dim]``,
    ``k [seq, num_kv_heads*head_dim]``, ``v [seq, num_kv_heads*head_dim]``,
    ``out [seq, num_heads*head_dim]``.
    """
    from flash_attention.kernel_fusion_based.attn_npu2_seqfirst import (
        build_module as build_attention_module,
    )

    return str(
        build_attention_module(
            lk=cfg["seq_len"],
            lkp=cfg["kv_seq_tile"],
            lq=cfg["seq_len"],
            lqp=cfg["parallel_seq"],
            dk=cfg["head_dim"],
            dv=cfg["head_dim"],
            num_q_tiles=cfg["num_q_tiles"],
            num_cascade_stages=cfg["cascade_stages"],
            num_heads=cfg["num_heads"],
            num_kv_heads=cfg["num_kv_heads"],
            causal=cfg["causal"],
            num_heads_per_unroll=cfg["parallel_heads"],
        )
    )


def chunked_attention_reference(
    q,
    k,
    v,
    num_heads,
    num_kv_heads=None,
    causal=False,
    query_block_size=256,
):
    """FP32 chunked scaled-dot-product attention over a seq-first layout.

    Ports the shape of ``iron``'s ``_chunked_attention`` -- the query-block
    loop, the mask built from ``q_positions``/``kv_positions``, and
    ``scale = 1/sqrt(head_dim)`` -- and DROPS its sequence-length branch. iron
    runs bf16 SDPA below ``seq_len 16384`` and FP32 chunked attention at and
    above it, so the oracle's own precision changes across the ladder; a
    tolerance calibrated at one end then means something different at the other.
    Chunking is arithmetically exact here, so applying it everywhere costs
    nothing and removes the discontinuity.

    Args:
        q: ``[seq, num_heads * head_dim]``.
        k, v: ``[seq, num_kv_heads * head_dim]``.
        num_heads, num_kv_heads: head counts; ``None`` means MHA.
        causal: mask ``kv_position > q_position``.
        query_block_size: rows per block. Arithmetically irrelevant; it bounds
            the score matrix to ``query_block_size x seq``.

    Returns:
        ``[seq, num_heads * head_dim]`` FLOAT32 -- not rounded to bf16. The
        caller owns the rounding, because the O-projection downstream consumes
        this in f32 and rounding here would model a staging step the reference
        deliberately does not reproduce (see ``mha_out_proj.py``).
    """
    if num_kv_heads is None:
        num_kv_heads = num_heads
    seq_len = q.shape[0]
    head_dim = q.shape[1] // num_heads
    gqa_group_size = num_heads // num_kv_heads
    scale = 1.0 / np.sqrt(head_dim)

    out = np.zeros((seq_len, num_heads * head_dim), dtype=np.float32)
    for h in range(num_heads):
        kv_h = h // gqa_group_size
        qf = q[:, h * head_dim : (h + 1) * head_dim].astype(np.float32)
        kf = k[:, kv_h * head_dim : (kv_h + 1) * head_dim].astype(np.float32)
        vf = v[:, kv_h * head_dim : (kv_h + 1) * head_dim].astype(np.float32)
        kv_positions = np.arange(seq_len)[None, :]
        for start in range(0, seq_len, query_block_size):
            end = min(start + query_block_size, seq_len)
            scores = (qf[start:end] @ kf.T) * scale
            if causal:
                q_positions = np.arange(start, end)[:, None]
                scores = np.where(kv_positions > q_positions, -np.inf, scores)
            # Subtract the row max before exponentiating, as torch.softmax
            # does. Under causal masking every row keeps its diagonal element,
            # so the max is always finite and no row is wholly -inf.
            scores -= scores.max(axis=-1, keepdims=True)
            probs = np.exp(scores)
            probs /= probs.sum(axis=-1, keepdims=True)
            out[start:end, h * head_dim : (h + 1) * head_dim] = probs @ vf
    return out
