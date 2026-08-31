# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GEMM module builder for NPU2 BF16 matrix multiplication.

Thin llama-side adapter over the contract-split example builders in
`matrix_multiplication/bf16_in_{fp32,bf16}_out/run.py`. The direct-codegen
transform lives there (a single definition per dtype, reused via
`build_module_lowered`); this file no longer keeps its own copy.

THE VARIANT NAME IS A FUNCTION OF (tile_m, tile_n), NOT OF THE METHOD ALONE
    `gemm_variant_names` below is the single authority for both the MLIR symbol
    suffix and the `mm_*.o` filename of an external GEMM, and it derives them
    from BOTH tile dimensions. It has to, because `mm_aie2p.cc` bakes tile_m AND
    tile_n in as `-DDIM_M` / `-DDIM_N`, and the private FuncOps the GEMM builder
    declares (`f32_to_bf16_mn<suffix>`, `zero_f32_mn<suffix>`,
    `op_has_no_registered_library_name<suffix>`) carry operand memref types that
    are functions of tile_n. Two GEMMs of the same method at different tile_n
    are therefore two DIFFERENT micro-kernels with two DIFFERENT signatures.

    Naming them from the method alone collided in two places, one loud and one
    silent:

      - `stitch_elf` collects each slice's private declarations into one set()
        and re-parses. Same symbol, different memref types -> `redefinition of
        symbol named ...` at any shape where two same-method GEMMs co-link.
      - `compile_gemm_mm` writes its object named from the method while
        `-DDIM_N` comes from tile_n, so two such GEMMs wrote the SAME FILE and
        aiecc linked whichever landed last. That one does not fail: the study
        (phase D2) measured an FFN up-projection linked against a 96-wide
        micro-kernel returning exactly zero for 32 of every 128 output columns.

    So the registry path (`gemm_registry_config` -> `_spec_with_tiles`) now
    mints `sym_suffix` / `obj` per (tile_m, tile_n), and `with_tile_n` re-mints
    a spec whose tile_n a caller overrides. One deliberate divergence from the
    study branch: at tile_n=128 the minted names are the bare method-era ones
    (`_m32` / `mm_m32.o`), because every shipped ELF and the cache staging list
    link those exact names for objects that were all compiled at DIM_N=128 --
    see `gemm_variant_names`. `disambiguate_by_tile_n` below predates this
    minting (SmolVLA's `_m32_n80`-style names) and is kept for those callers;
    unifying the two spellings is follow-up work.
"""

from ml_dtypes import bfloat16

# External-bf16 high-precision methods (both = f32 accumulate + single epilogue
# cast = 9.3e-3). They differ only in HOW the cast is done, which fixes tile_m:
#   - fused-cast: external GEMM (f32 scratch) + separate cast launch, tile_m=64.
#                 Faster on large shapes (M*K*N >= 4e9).
#   - drain:      in-GEMM drain-herd cast, single launch, tile_m=32 (L1 ceiling).
#                 Better on small/thin shapes.
# THIS TABLE IS AUTHORITATIVE for method -> tile_m.
_METHOD_TILE_M = {"fused-cast": 64, "drain": 32}

# Per-method module structure. tile_m is above; everything here is about how the
# epilogue cast is wired, which is what makes the two methods
# non-interchangeable at the module level.
_METHOD_SHAPE = {
    # n_launches: launches this GEMM contributes to a stitched func.
    # scratch:    needs one extra f32 C-scratch func arg.
    # flag:       the _build_gemm_module keyword that selects this path.
    "fused-cast": {"n_launches": 2, "scratch": True, "flag": "external_fused_cast"},
    "drain": {"n_launches": 1, "scratch": False, "flag": "external_bf16_out"},
}


# Bias-on-the-weight-stream. An AIE2P core tile has only 2 inbound DMA
# channels and A/B already hold both, so a per-channel bias cannot arrive on a
# DMA of its own. Instead B is repacked with an 8-row pad block after every L1
# sub-chunk of tile_k_l1 weight rows; only the FINAL sub-chunk's pad carries the
# bias (replicated down its rows), the rest are zero. The mmul skips pad rows
# via DIM_K_PAD, and the drain herd folds the bias into the epilogue cast it
# already performs. 8 = the mmul k granularity, the smallest legal block.
BIAS_PAD_ROWS = 8


def packed_k(k, tile_k_l1):
    """Row count of the bias-packed B for a (k, tile_k_l1) GEMM."""
    return (k // tile_k_l1) * (tile_k_l1 + BIAS_PAD_ROWS)


def repack_gemm_b_with_bias(w, bias, tile_k_l1):
    """(K,N) weights + (N,) bias -> packed_k(K,tile_k_l1) x N, bias in the last pad.

    Mirrors the access pattern build_module emits for b_pad_rows=BIAS_PAD_ROWS.
    """
    import numpy as np

    k, n = w.shape
    tk1p = tile_k_l1 + BIAS_PAD_ROWS
    nsub = k // tile_k_l1
    out = np.zeros((nsub * tk1p, n), dtype=w.dtype)
    for j in range(nsub):
        out[j * tk1p : j * tk1p + tile_k_l1] = w[j * tile_k_l1 : (j + 1) * tile_k_l1]
    last = (nsub - 1) * tk1p + tile_k_l1
    out[last : last + BIAS_PAD_ROWS] = bias[None, :].astype(w.dtype)
    return out


def gemm_variant_names(tile_m, tile_n):
    """``(sym_suffix, obj)`` for one external-GEMM micro-kernel variant.

    ``gemm_variant_names(32, 64) -> ("_m32n64", "mm_m32n64.o")``; at the legacy
    width the bare method-era names survive:
    ``gemm_variant_names(32, 128) -> ("_m32", "mm_m32.o")``.

    The ONE place either name is spelled. Every builder gets them from a spec
    that came from here, and every caller of ``compile_gemm_mm`` should get them
    from here too (``external_kernels.compile_gemm_mm_variant`` does it for you)
    -- an object compiled under a name the module does not reference is an
    unresolved symbol at link time, and an object compiled under a name the
    module DOES reference but at the wrong ``-DDIM_N`` is silent wrong numbers.

    Why tile_n=128 keeps the bare names: every shipped bf16 ELF and
    `cache.prepare_air_project`'s staging list link `_m32` / `_m64`, and every
    one of those objects was compiled at DIM_N=128. The bare name therefore IS
    the n128 variant; keeping it as that variant's name leaves every existing
    artifact name and the staging list untouched, and the mapping stays
    bijective (bare <=> tile_n=128, `n{tile_n}`-suffixed <=> anything else).
    """
    stem = f"_m{tile_m}" if tile_n == 128 else f"_m{tile_m}n{tile_n}"
    return stem, f"mm{stem}.o"


def with_tile_n(spec, tile_n):
    """A copy of `spec` retiled to `tile_n`, with its variant names RE-MINTED.

    Call this instead of assigning `spec["tile_n"] = ...`. A caller that
    overrides tile_n after the fact -- and several do, because the registry's
    tile_n for a narrow N can be numerically wrong and the builder pads N out
    to admit a wider one -- is choosing a DIFFERENT micro-kernel, not retuning
    the same one. The plain assignment leaves `sym_suffix` / `obj` naming the
    OLD variant: a spec asking for an object nobody builds, or worse, one
    somebody else built at a different -DDIM_N.

    `sym_suffix`, `obj` and `build_kwargs` all move together, which is the
    point: they are three views of one variant identity and no caller should
    be able to move one without the others.
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


def gemm_registry_config(m, k, n, output_dtype="bf16", precision="high", method=None):
    """Full per-shape build recipe from the registry: the chosen method's spec
    (build_kwargs / suffix / launches) MERGED with the registry tile sizes. This is
    the single entry point llama builders use so tiles + method are never hardcoded.

    `method` FORCES a registry method (its MEASURED tiles for this shape, via
    `gemm_config_method`) instead of the tier's best -- for a caller whose
    cascade supports one form only, or an A/B comparison. A forced method is a
    deviation from the plan and must be recorded as one by the caller.

    Returns the gemm_method_spec dict plus:
      tile_k_l2, tile_k_l1, tile_n : from the registry JSON (tile_m comes from the
                                     method spec — drain=32 / fused-cast=64)
      method                       : the registry-selected (or forced) method name
    """
    from kernel_registry.registry_lookup import gemm_config, gemm_config_method

    if method is not None:
        cfg = gemm_config_method(m, k, n, output_dtype, method, precision)
    else:
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


def gemm_method_spec(method, tile_n=None):
    """Reusable per-GEMM method primitive for ELF-merged kernels. Returns a dict
    describing how to build + stitch ONE GEMM by the chosen method at this
    tile_n, so any GEMM in any merged ELF can independently pick drain vs
    fused-cast (they are two implementations of the same bf16-in/bf16-out
    high-precision GEMM) AND any tile_n the registry measured for it:

      tile_m         : the forced tile_m (drain=32, fused-cast=64)
      tile_n         : echoed back when given (it is half of the variant's name)
      n_launches     : launches this GEMM contributes to the stitched func (drain=1,
                       fused-cast=2 — the GEMM launch + the cast launch)
      needs_f32_scratch : fused-cast needs one extra f32 C-scratch func arg
      sym_suffix / obj  : symbol suffix + mm.o filename for co-linking, minted
                       per (tile_m, tile_n) by `gemm_variant_names`
      build_kwargs   : kwargs for _build_gemm_module (minus m,k,n,tiles,herd)

    `tile_n=None` (back-compat) mints the bare method-era names, which are the
    tile_n=128 variant's names -- correct exactly for the remaining direct
    callers that compile their own object at DIM_N=128. Callers that know
    their tile_n must pass it (the registry path does); making it required
    waits until the stragglers thread theirs through.
    """
    if method not in _METHOD_SHAPE:
        raise ValueError(f"unknown gemm method: {method!r}")
    tile_m = _METHOD_TILE_M[method]
    shape = _METHOD_SHAPE[method]
    sym_suffix, obj = gemm_variant_names(tile_m, 128 if tile_n is None else tile_n)
    spec = {
        "tile_m": tile_m,
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
    if tile_n is not None:
        spec["tile_n"] = tile_n
    return spec


def disambiguate_by_tile_n(specs):
    """Fix up a list of gemm_registry_config() specs that will be co-linked into
    ONE fused ELF, so GEMMs sharing a method but resolving to DIFFERENT tile_n
    don't collide.

    Since the per-(tile_m, tile_n) minting in `gemm_variant_names` landed,
    specs arrive here already distinct across tile_n; this rewrite now exists
    to keep the SmolVLA builders' historical `_m32_n80`-style names stable for
    mixed-tile_n sets (its hand-mirrored helper spells them too). Unifying the
    two spellings on gemm_variant_names is deliberate follow-up work.

    Why this is needed: `compile_gemm_mm` bakes DIM_N=tile_n as a compile-time
    C macro into the external mm.o object, and gemm_method_spec()'s sym_suffix
    (_m32 / _m64) is keyed ONLY on method, not on tile_n. Every existing
    fused-ELF caller (llama32_1b, rms_gemms_rope for all current models, etc.)
    happens to have uniform tile_n per method across its GEMMs, so this never
    mattered before. SmolVLA's o_ffn ELF is the first case where two "drain"
    GEMMs in the SAME ELF resolve to different tile_n (e.g. O/Down -> 80,
    Gate/Up -> 128): sharing "_m32"/mm_m32.o for both means the second one's
    call sites disagree with the first's compiled object -> MLIR verifier
    rejects the stitched module (operand shape mismatch on the shared cast
    symbol, since its tile shape is derived from tile_m*tile_n).

    For each method, if all given specs using it share one tile_n, they are
    left untouched (identical suffix/obj as gemm_method_spec, so no existing
    caller's compiled artifacts need to change name). Only when a method has
    >1 distinct tile_n among the given specs are those specs' sym_suffix/obj/
    build_kwargs rewritten to also key off tile_n, so each distinct (method,
    tile_n) pair gets its own non-colliding symbol suffix + mm.o name.

    Returns a NEW list of spec dicts (same order/length as `specs`); inputs
    are not mutated. Callers must compile_gemm_mm(...) using the RETURNED
    specs' tile_m/tile_n/tile_k_l1/sym_suffix/obj (not the pre-disambiguation
    ones) so the compiled objects match what the stitched IR expects.
    """
    tile_n_by_method = {}
    for s in specs:
        tile_n_by_method.setdefault(s["method"], set()).add(s["tile_n"])

    out = []
    for s in specs:
        if len(tile_n_by_method[s["method"]]) > 1:
            s = dict(s)
            tag = "m32" if s["method"] == "drain" else "m64"
            suffix = f"_{tag}_n{s['tile_n']}"
            obj = f"mm_{tag}_n{s['tile_n']}.o"
            s["sym_suffix"] = suffix
            s["obj"] = obj
            s["build_kwargs"] = dict(s["build_kwargs"])
            s["build_kwargs"]["sym_suffix"] = suffix
            s["build_kwargs"]["link_with_name"] = obj
        out.append(s)
    return out


def o_ffn_gemm_layout(
    seq_len, emb_dim, hidden_dim, q_dim=None, method=None, base_arg_count=15
):
    """The O+FFN cascade's per-GEMM registry specs and f32-scratch tail,
    derived WITHOUT building anything.

    The single owner of the cascade's GEMM identity: `o_ffn_multi._build_o_ffn`
    builds from it, and driver-side arg planning mirrors it
    (`llama32_1b_prefill._o_ffn_scratch_plan`). Air-free on purpose (registry
    JSON + `alloc_gemm_scratch` only) so host tests can pin it.

    `q_dim` is the O GEMM's inner dim (decoupled head: K = q_dim != emb_dim);
    None means the square llama form. `method` forces every GEMM's registry
    method (test/A-B only; a forced method is a plan deviation recorded by the
    caller). Returns {"o", "gate_up", "down": spec, "scratch_args",
    "scratch_for" (order O/gate/up/down), "launches" (GEMM + 4 non-GEMM)}.
    """
    from shared.infra.stitching import alloc_gemm_scratch

    if q_dim is None:
        q_dim = emb_dim
    o_spec = gemm_registry_config(
        seq_len, q_dim, emb_dim, "bf16", "high", method=method
    )
    g_spec = gemm_registry_config(
        seq_len, emb_dim, hidden_dim, "bf16", "high", method=method
    )
    d_spec = gemm_registry_config(
        seq_len, hidden_dim, emb_dim, "bf16", "high", method=method
    )
    scratch_args, scratch_for = alloc_gemm_scratch(
        [
            (o_spec, seq_len, emb_dim),
            (g_spec, seq_len, hidden_dim),
            (g_spec, seq_len, hidden_dim),
            (d_spec, seq_len, emb_dim),
        ],
        base_arg_count=base_arg_count,
    )
    launches = (
        o_spec["n_launches"] + 2 * g_spec["n_launches"] + d_spec["n_launches"] + 4
    )  # + res add, RMSNorm, SwiGLU, FFN add
    return {
        "o": o_spec,
        "gate_up": g_spec,
        "down": d_spec,
        "scratch_args": scratch_args,
        "scratch_for": scratch_for,
        "launches": launches,
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
    b_pad_rows=0,
    epilogue_gelu=False,
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
    tile_n), e.g. `_m64` / `mm_m64.o` at the legacy tile_n=128 and `_m64n64` /
    `mm_m64n64.o` otherwise, see `gemm_variant_names` -- so any mix of methods
    and tile_n can co-link in one fused ELF. The bare `mm.o` default is for a
    single-variant ELF that names its own object.
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
            b_pad_rows=b_pad_rows,
            epilogue_gelu=epilogue_gelu,
        )

    raise ValueError(
        "_build_gemm_module: must set external_fused_cast=True or external_bf16_out=True "
        "(llama GEMM is always the external high-precision path; tiles+method come from "
        "the registry via gemm_registry_config)."
    )
