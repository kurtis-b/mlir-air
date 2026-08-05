# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GEMM module builder for NPU2 BF16 matrix multiplication.

Thin llama-side adapter over the contract-split example builders in
`matrix_multiplication/bf16_in_{fp32,bf16}_out/run.py`. The direct-codegen
transform lives there (a single definition per dtype, reused via
`build_module_lowered`); this file no longer keeps its own copy.

THE VARIANT NAME IS A FUNCTION OF (tile_m, tile_n), NOT OF THE METHOD
    `gemm_variant_names` below is the single authority for both the MLIR symbol
    suffix and the `mm_*.o` filename of an external GEMM, and it derives them
    from BOTH tile dimensions. It has to, because `mm_aie2p.cc` bakes tile_m AND
    tile_n in as `-DDIM_M` / `-DDIM_N`, and the private FuncOps the GEMM builder
    declares (`f32_to_bf16_mn<suffix>`, `zero_f32_mn<suffix>`,
    `op_has_no_registered_library_name<suffix>`) carry operand memref types that
    are functions of tile_n. Two GEMMs of the same method at different tile_n
    are therefore two DIFFERENT micro-kernels with two DIFFERENT signatures.

    Naming them from the method alone -- which is what this file did until Phase
    E1 -- collided in two places, one loud and one silent:

      - `stitch_elf` collects each slice's private declarations into one set()
        and re-parses. Same symbol, different memref types -> `redefinition of
        symbol named ...`, and `build_ffn_module` could not build at any
        sequence length where both projections resolved to the same method.
      - `compile_gemm_mm` writes its object named from the method while `-DDIM_N`
        comes from tile_n, so two such GEMMs wrote the SAME FILE and aiecc linked
        whichever landed last. That one does not fail. Phase D2 measured what it
        does instead: the FFN's up-projection got the o-projection's 96-wide
        micro-kernel and returned exactly zero for 32 of every 128 output
        columns, which the GeLU passed through and the down-projection's
        3072-deep reduction smeared over the whole FFN output.

    So `tile_n` is now a required argument of `gemm_method_spec`, and it is
    required rather than defaulted on purpose: a caller that does not know its
    tile_n cannot know which object to link, and a default would reintroduce
    exactly the silent collision above.

    Consequence worth knowing before you see it: `tile_n` takes four values
    across the registry ({32, 64, 96, 128}), so up to eight `mm_*.o` objects are
    minted where there used to be two. Each is smaller and they compile
    independently; it is more compiles, not more work per compile.
"""

from ml_dtypes import bfloat16

# External-bf16 high-precision methods (both = f32 accumulate + single epilogue
# cast = 9.3e-3). They differ only in HOW the cast is done, which fixes tile_m:
#   - fused-cast: external GEMM (f32 scratch) + separate cast launch, tile_m=64.
#                 Faster on large shapes (M*K*N >= 4e9).
#   - drain:      in-GEMM drain-herd cast, single launch, tile_m=32 (L1 ceiling).
#                 Better on small/thin shapes.
# THIS TABLE IS AUTHORITATIVE for method -> tile_m. `sweep/sweep_families.py`
# duplicates it for planning purposes and cross-references this line; if the two
# ever disagree the sweep plans configurations the builders cannot build.
_METHOD_TILE_M = {"fused-cast": 64, "drain": 32}

# Per-method module structure. tile_m is above; everything here is about how the
# epilogue cast is wired, which is what makes the two methods non-interchangeable
# at the module level (see gemm_spec.py in the transformer-layer builders).
_METHOD_SHAPE = {
    # n_launches: launches this GEMM contributes to a stitched func.
    # scratch:    needs one extra f32 C-scratch func arg.
    # build_flag: the _build_gemm_module keyword that selects this path.
    "fused-cast": {"n_launches": 2, "scratch": True, "flag": "external_fused_cast"},
    "drain": {"n_launches": 1, "scratch": False, "flag": "external_bf16_out"},
}


def gemm_variant_names(tile_m, tile_n):
    """``(sym_suffix, obj)`` for one external-GEMM micro-kernel variant.

    ``gemm_variant_names(32, 128) -> ("_m32n128", "mm_m32n128.o")``.

    The ONE place either name is spelled. Every builder gets them from a spec
    that came from here, and every caller of ``compile_gemm_mm`` should get them
    from here too (``external_kernels.compile_gemm_mm_variant`` does it for you)
    -- an object compiled under a name the module does not reference is an
    unresolved symbol at link time, and an object compiled under a name the
    module DOES reference but at the wrong ``-DDIM_N`` is silent wrong numbers.
    """
    stem = f"_m{tile_m}n{tile_n}"
    return stem, f"mm{stem}.o"


def with_tile_n(spec, tile_n):
    """A copy of `spec` retiled to `tile_n`, with its variant names RE-MINTED.

    Call this instead of assigning `spec["tile_n"] = ...`. A caller that
    overrides tile_n after the fact -- and several do, because the registry's
    tile_n for a narrow N can be numerically wrong and the builder pads N out to
    admit a wider one -- is choosing a DIFFERENT micro-kernel, not retuning the
    same one. Before Phase E1 the assignment was harmless because the object
    name did not depend on tile_n; now it is the difference between linking
    `mm_m32n128.o` and linking `mm_m32n32.o`, and the plain assignment would
    leave the spec asking for an object nobody builds.

    `sym_suffix`, `obj` and `build_kwargs` all move together, which is the point:
    they are three views of one variant identity and no caller should be able to
    move one without the others.
    """
    retiled = dict(spec)
    retiled["tile_n"] = tile_n
    sym_suffix, obj = gemm_variant_names(spec["tile_m"], tile_n)
    retiled["sym_suffix"] = sym_suffix
    retiled["obj"] = obj
    build_kwargs = dict(spec.get("build_kwargs", {}))
    if "sym_suffix" in build_kwargs:
        build_kwargs["sym_suffix"] = sym_suffix
    if "link_with_name" in build_kwargs:
        build_kwargs["link_with_name"] = obj
    retiled["build_kwargs"] = build_kwargs
    return retiled


def gemm_registry_config(m, k, n, output_dtype="bf16", precision="high"):
    """Full per-shape build recipe from the registry: the chosen method's spec
    (build_kwargs / suffix / launches) MERGED with the registry tile sizes. This is
    the single entry point llama builders use so tiles + method are never hardcoded.

    Returns the gemm_method_spec dict plus:
      tile_k_l2, tile_k_l1, tile_n : from the registry JSON (tile_m comes from the
                                     method spec — drain=32 / fused-cast=64)
      method                       : the registry-selected method name
    """
    from kernel_registry.registry_lookup import gemm_config

    cfg = gemm_config(m, k, n, output_dtype, precision)
    return _spec_with_tiles(cfg["method"], cfg["tile"])


def _spec_with_tiles(method, tile):
    """Merge a method's build spec with the registry tile (a named dict
    {tile_m, tile_k_l2, tile_k_l1, tile_n}). tile_m is dictated by the method
    (drain=32 / fused=64) and matches spec['tile_m'] (asserted for safety).

    The registry's tile_n goes IN to `gemm_method_spec`, because it is half of
    the variant's name, and is then re-set below for the callers that read it
    back off the merged spec.
    """
    spec = dict(gemm_method_spec(method, tile["tile_n"]))
    assert (
        tile["tile_m"] == spec["tile_m"]
    ), f"registry tile_m={tile['tile_m']} != method '{method}' tile_m={spec['tile_m']}"
    spec["method"] = method
    spec["tile_k_l2"] = tile["tile_k_l2"]
    spec["tile_k_l1"] = tile["tile_k_l1"]
    spec["tile_n"] = tile["tile_n"]
    return spec


def gemm_method_spec(method, tile_n):
    """Reusable per-GEMM method primitive for ELF-merged kernels. Returns a dict
    describing how to build + stitch ONE GEMM by the chosen method at this
    tile_n, so any GEMM in any merged ELF can independently pick drain vs
    fused-cast (they are two implementations of the same bf16-in/bf16-out
    high-precision GEMM) AND any tile_n the registry measured for it:

      tile_m         : the forced tile_m (drain=32, fused-cast=64)
      tile_n         : echoed back, because it is half of the variant's name
      n_launches     : launches this GEMM contributes to the stitched func (drain=1,
                       fused-cast=2 — the GEMM launch + the cast launch)
      needs_f32_scratch : fused-cast needs one extra f32 C-scratch func arg
      sym_suffix / obj  : symbol suffix + mm.o filename for co-linking
      build_kwargs   : kwargs for _build_gemm_module (minus m,k,n,tiles,herd)

    `tile_n` is REQUIRED. See the module docstring: the object's `-DDIM_N` and
    the declared operand memref types both depend on it, so a variant named
    without it either fails to parse (two same-method GEMMs in one ELF) or links
    the wrong micro-kernel and returns zeros for part of the output tile.

    Only the two high-precision external methods are here. `direct` raises, and
    that asymmetry is deliberate and pre-existing: a caller that wants direct
    codegen synthesizes its own spec (see `qwen25_0_5b_prefill.py`), because
    there is no external object to name.
    """
    if method not in _METHOD_SHAPE:
        raise ValueError(f"unknown gemm method: {method!r}")
    tile_m = _METHOD_TILE_M[method]
    shape = _METHOD_SHAPE[method]
    sym_suffix, obj = gemm_variant_names(tile_m, tile_n)
    return {
        "tile_m": tile_m,
        "tile_n": tile_n,
        "n_launches": shape["n_launches"],
        "needs_f32_scratch": shape["scratch"],
        "sym_suffix": sym_suffix,
        "obj": obj,
        "build_kwargs": {
            shape["flag"]: True,
            "sym_suffix": sym_suffix,
            "link_with_name": obj,
        },
    }


def _build_gemm_module(
    m,
    k,
    n,
    tile_m,
    tile_k_l2,
    tile_k_l1,
    tile_n,
    herd_m=8,
    herd_n=4,
    external_fused_cast=False,
    external_bf16_out=False,
    sym_suffix="",
    link_with_name="mm.o",
):
    """Build a high-precision BF16-in/BF16-out GEMM via the external mm.o microkernel.

    Two methods (both = f32 accumulate + single epilogue cast = GPU-standard 9.3e-3;
    the registry picks which per shape, see gemm_registry_config):
    - external_fused_cast=True: external GEMM writes an f32 C scratch (full tile_m=64)
      then a SEPARATE on-chip cast launch → `@gemm_cast_bf16`, 2 launches, 4 args
      (A, B, C-f32-scratch, D-bf16-out). Faster on large shapes (M*K*N>=4e9).
    - external_bf16_out=True: in-GEMM drain-herd cast, 1 launch, tile_m=32 (the
      tile_m=64 drain overflows L1). Better on small/thin shapes.

    sym_suffix / link_with_name disambiguate the mm.o variant -- per (tile_m,
    tile_n), e.g. `_m64n128` / `mm_m64n128.o`, see `gemm_variant_names` -- so any
    mix of methods and tile_n can co-link in one fused ELF. The bare `mm.o`
    default is for a single-variant ELF that names its own object.
    """
    if external_fused_cast:
        from matrix_multiplication.bf16_in_bf16_out.run import build_module_gemm_cast

        return build_module_gemm_cast(
            m,
            k,
            n,
            tile_m,
            tile_k_l2,
            tile_k_l1,
            tile_n,
            herd_m,
            herd_n,
            arch="aie2p",
            sym_suffix=sym_suffix,
            link_with_name=link_with_name,
        )

    if external_bf16_out:
        from matrix_multiplication.bf16_in_bf16_out.run import (
            build_module as build_gemm_bf16_ext,
        )

        return build_gemm_bf16_ext(
            m,
            k,
            n,
            tile_m,
            tile_k_l2,
            tile_k_l1,
            tile_n,
            herd_m,
            herd_n,
            bfloat16,
            bfloat16,  # bf16 output: f32 accumulator + single drain cast
            arch="aie2p",
            emit_external_call=True,
            sym_suffix=sym_suffix,
            link_with_name=link_with_name,
        )

    raise ValueError(
        "_build_gemm_module: must set external_fused_cast=True or external_bf16_out=True "
        "(llama GEMM is always the external high-precision path; tiles+method come from "
        "the registry via gemm_registry_config)."
    )
