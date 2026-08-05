# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""O-projection staging for ``mha_out_proj``: the output GEMM half.

CONTRACT
    ``o_proj_gemm_spec(seq_len, emb_dim, ...)`` resolves the method and tiles
    for ``y = attn_out @ w_o``, an ``[S, E] @ [E, E]`` GEMM, registry-first.
    ``build_o_proj_ir(...)`` returns that GEMM's sub-kernel MLIR text and
    ``o_proj_extern_syms(spec)`` the symbols its launches link against.
    ``o_proj_reference(...)`` is the FP32 oracle for this half alone.

    Split out from the composition per porting convention 5. This half knows
    the registry and the GEMM builder; the attention half knows the
    FlashAttention design; neither needs the other's internals, and the
    composition in ``mha_out_proj.py`` wires them.

WHAT ``o_proj_acc_depth`` IS
    iron exposes an ``o_proj_acc_depth`` knob for how deep the O-projection
    accumulates in the memory tile before draining. In AIR that quantity is the
    GEMM's ``tile_k_l2``, and it is MEASURED PER SHAPE and stored in
    ``kernel_registry`` -- porting convention 9 puts the registry, not a
    constant and not a caller, in charge of it.

    So the parameter here defaults to ``None``, meaning "whatever the registry
    measured". A caller may still pass a value; that is an explicit override,
    it is recorded in the results artifact alongside the registry's own, and it
    is printed. What it is not is a silent hand-copied tile config, which is the
    drift bug ``registry_lookup.py``'s docstring exists to describe.

WHY THIS DOES NOT REQUIRE A PARTICULAR METHOD
    ``qkv_proj`` rejects the ``drain`` method because its three-way C split
    rides the separate cast launch that only ``fused-cast`` has. This operator
    has no such structural requirement -- it consumes one ``A``, one ``B`` and
    writes one bf16 ``C`` -- so both methods are wired, and the arg layout
    grows an f32 scratch only when the resolved method needs one. That is why
    ``mha_out_proj_arg_layout`` is computed rather than written out.

FOOTGUNS
    - The GEMM's ``A`` operand is the attention half's OUTPUT buffer, in DDR, in
      bf16. The reference in ``mha_out_proj.py`` is FP32 end to end and does not
      reproduce that rounding, deliberately -- the same stance ``ffn.py`` takes
      for its ``h``. The gap is measured rather than defined away, and it is
      part of what ``atol`` is sized against.
    - The resolved method's ``mm_*.o`` (``mm_m32n128.o`` for drain, ``mm_m64n128.o``
      for fused-cast) must be in the working directory when aiecc links.
      ``compile_o_proj_kernel`` builds it there.
    - ``herd_m`` / ``herd_n`` default to the herd the registry tiles were
      measured against -- 8x4 for most rows, smaller for a row whose ``M`` is
      too short to hold 8 of its method's ``tile_m``. Changing them invalidates
      the tiling without invalidating the lookup, so leave them alone unless
      the registry entry moves too.
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


def o_proj_gemm_spec(seq_len, emb_dim, o_proj_acc_depth=None, gemm_spec_fn=None):
    """Resolve the O-projection GEMM's method and tiles, registry-first.

    Returns ``(spec, source)``. ``source`` is ``"registry"``, ``"injected"``
    (``gemm_spec_fn`` supplied the whole spec for an unmeasured shape), or
    ``"registry+acc_depth_override"`` (registry spec with ``tile_k_l2``
    replaced). Split out from the builder so the caller can record what was
    actually used without building the module twice.
    """
    from builders.gemm_spec import resolve_gemm_spec

    if gemm_spec_fn is not None:
        spec = dict(gemm_spec_fn(seq_len, emb_dim, emb_dim))
        source = "injected"
    else:
        spec = dict(resolve_gemm_spec(seq_len, emb_dim, emb_dim, "bf16", "high"))
        source = "registry"

    if o_proj_acc_depth is not None and o_proj_acc_depth != spec["tile_k_l2"]:
        spec["registry_tile_k_l2"] = spec["tile_k_l2"]
        spec["tile_k_l2"] = o_proj_acc_depth
        source = f"{source}+acc_depth_override"
    return spec, source


def o_proj_extern_syms(spec):
    """Symbols the GEMM's launches link against, which must not be renamed.

    Both methods are covered: the suffixed ``mm_*.o`` entry points plus the
    unsuffixed ``@matmul_bf16`` the drain path emits. Naming a symbol the
    resolved method does not emit is harmless; the set only suppresses the
    per-slice prefix rename.
    """
    sfx = spec["sym_suffix"]
    return {
        "@matmul_bf16",
        "@op_has_no_registered_library_name" + sfx,
        "@zero_f32_mn" + sfx,
        "@f32_to_bf16_mn" + sfx,
    }


def compile_o_proj_kernel(spec):
    """Build the ``mm_*.o`` the resolved method names, in the CWD."""
    import shared.infra.external_kernels as ek

    ek.compile_gemm_mm(
        tile_m=spec["tile_m"],
        tile_n=spec["tile_n"],
        tile_k_l1=spec["tile_k_l1"],
        sym_suffix=spec["sym_suffix"],
        out_name=spec["obj"],
    )


def describe_o_proj_spec(seq_len, emb_dim, spec, source, herd_m=None, herd_n=None):
    """One-line summary of a resolved spec, for the run log."""
    from builders.gemm_spec import describe_herd

    depth = f"tile_k_l2={spec['tile_k_l2']}"
    if "registry_tile_k_l2" in spec:
        depth += f" (registry {spec['registry_tile_k_l2']})"
    return (
        f"{seq_len}x{emb_dim}x{emb_dim} ({spec['method']}, {source}, "
        f"tile_m={spec['tile_m']} {depth} tile_k_l1={spec['tile_k_l1']} "
        f"tile_n={spec['tile_n']} herd={describe_herd(spec, herd_m, herd_n)})"
    )


def build_o_proj_ir(seq_len, emb_dim, spec, herd_m=None, herd_n=None):
    """The O-projection GEMM's sub-kernel MLIR text.

    Launch operands, in order: ``A [seq_len, emb_dim]``,
    ``B [emb_dim, emb_dim]``, then either ``C-f32-scratch`` and ``D-bf16-out``
    (fused-cast, two launches) or ``C-bf16-out`` (drain, one launch). The
    composition reads ``spec["needs_f32_scratch"]`` to wire them; it is the same
    branch ``ffn.py::_gemm_slice`` takes.

    ``herd_m`` / ``herd_n`` of ``None`` take the herd the resolved row was
    measured at, which is the only setting that builds at every point on the
    sequence ladder.
    """
    from shared.builders.gemm_builder import _build_gemm_module

    from builders.gemm_spec import spec_herd

    return str(
        _build_gemm_module(
            seq_len,
            emb_dim,
            emb_dim,
            spec["tile_m"],
            spec["tile_k_l2"],
            spec["tile_k_l1"],
            spec["tile_n"],
            *spec_herd(spec, herd_m, herd_n),
            **spec["build_kwargs"],
        )
    )


def o_proj_reference(attn_out, w_o):
    """FP32 reference for ``attn_out @ w_o``, unrounded.

    Takes and returns float32. The device accumulates the whole ``emb_dim``
    reduction in f32 and rounds once in its epilogue; the caller applies that
    single rounding, so this half stays composable with the attention half
    without an intermediate bf16 step neither reference models.
    """
    return attn_out.astype(np.float32) @ w_o.astype(np.float32)
