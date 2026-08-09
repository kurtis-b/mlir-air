# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One ``encoder_bert`` transformer layer, assembled from the Phase C operators.

CONTRACT
    ``block_config(...)`` resolves every operator's configuration without
    building anything. ``compile_block_artifacts(cache, cfg)`` puts four ELFs in
    a ``KernelCache``. ``run_block(cache, cfg, inputs)`` dispatches them through
    ``KernelCache.run_sequence`` and returns

        (boundaries, vector_rows)

    where ``boundaries`` is a dict holding the DEVICE's value at every operator
    boundary -- the same ten names ``pattern/reference.py`` produces for the
    golden model -- and ``vector_rows`` is one ``DispatchVector`` row per
    sequence, for the results artifact.

    ``inputs`` is the ordered list in ``BLOCK_INPUT_NAMES``. It is a LIST and not
    a dict because ``opcheck.py``'s fault injection perturbs
    ``prepared["inputs"][i]``; the order below is therefore part of the contract
    between this module and ``opcheck_specs.py``.

    Five operator launches, in order::

        1  qkv_proj      x                     -> q, k, v
        2  mha_out_proj  q, k, v, w_o          -> attn_context, attn_out
        3  addnorm       attn_out, x,   ln1_w  -> hidden        (PRE-ADD)
        4  ffn           hidden, w_up, w_down  -> ffn_up, ffn_gelu, ffn_out
        5  addnorm       ffn_out, hidden, ln2_w-> output        (PRE-ADD)

    ``layer_norm``, ``elementwise_add`` and ``causal_mask`` are NOT on this path:
    the residual add lives inside the pre-add ``addnorm`` and ``encoder_bert`` is
    bidirectional. They are validated at 768 by D1 because Phase E's
    finer-grained modes decompose down to them, not because this block calls
    them.

WHY THE LAYER IS FOUR DISPATCH SEQUENCES AND NOT ONE
    Because ``addnorm`` cannot be dispatched over 4096 rows, and that is a
    property of the compiler's packet-DMA lowering, not of this file. Three
    L3->L1 streams per tile force packet multiplexing on the shim, and on a
    multi-column herd the multiplexed put loops cannot be fused into feed
    order (``air-fuse-packet-put-loops`` only matches sibling ``scf.for``
    loops, not the ``scf.parallel``-wrapped form a herd produces), so any
    multi-trip row loop miscompiles -- phase J1 measured the candidate
    collapse, 64 trips at 4096x768 over 8 columns, at 3.13M of 3.15M elements
    wrong, and every builder-side alternative (one-column herd, hoisted
    weight, weight staged through L2) hit a measured wall of its own; see
    ``builders/addnorm.py``'s multi-trip section for the full matrix. So
    ``builders/addnorm.py`` requires one kernel call per tile on multi-column
    herds and the row count is capped by what fits L1:
    ``addnorm_max_rows(768, pre_add=True)`` is 104. The layer's 4096 rows are
    therefore ROW-BLOCKED into ``seq_len // norm_rows`` dispatches of the very
    shape D1 validated (64 x 768, pre-add), which is the strongest form the
    constraint allows -- the block runs the operator that was measured, not a
    wider one that was not.

    The consequence is a host restage, because a sequence's buffers are whole
    BOs: ``run.set_arg`` takes a buffer, never a buffer and an offset, so an
    operator that consumes one 4096-row tensor in 64 pieces needs 64 buffers and
    the tensor has to be cut into them. The four sequences are:

        A  qkv_proj + mha_out_proj   fully device-resident: q/k/v are produced
                                     by step 1 and consumed by step 2 without
                                     touching the host. This is the sequence
                                     that actually exercises cross-artifact
                                     chaining under the Phase B allocator.
        B  64 x addnorm              ln1, row-blocked
        C  ffn
        D  64 x addnorm              ln2, row-blocked

    A single sequence would be preferable and is not available. Do not "fix" it
    by raising ``rows`` past ``addnorm_max_rows``: the builder raises, and the
    reason it raises is that the two-trip form MISCOMPILES rather than failing.

FOOTGUNS
    - ``run_sequence`` returns ZERO-COPY VIEWS into pool memory. Every boundary
      here is copied out with ``np.array(..., copy=True)`` before the next
      sequence runs. Skipping that does not fail; it silently compares the value
      a later launch left in the same slot.
    - The four artifacts must share an ABI, and ``run_sequence`` refuses a mixed
      one. Everything here is built ``output_format="elf"`` -- including
      ``addnorm``, which Phase C ran as an xclbin. That is not a free change of
      packaging: a multi-``air.launch`` design cannot be an xclbin at all
      (``air.insts.bin produced duplicate output path``), so ELF is forced by
      ``qkv_proj``, ``ffn`` and ``mha_out_proj``, and ``addnorm`` follows them.
    - ``instance_name`` must equal the emitted ``func.func`` name for each
      artifact. A mismatch does not fail to load; it times out with
      ``ERT_CMD_STATE_TIMEOUT`` a long way from the cause.
    - The per-operator backend settings are NOT interchangeable, and
      ``[2026-08-08]`` that is measured rather than inferred. ``mha_out_proj``
      needs ``runtime_loop_tiling_sizes=[1, 1]``; the GEMM-backed pair is built
      at ``[2, 2]`` for BD-ID recycling. Giving ``mha_out_proj`` the ``[2, 2]``
      setting makes it compile and then HANG — ``ERT_CMD_STATE_TIMEOUT``, 3/3
      at 4096, against 3/3 clean passes at ``[1, 1]``. The other half of the
      old claim does not hold: ``omit_pingpong="all"`` is not load-bearing at
      this shape. See ``agents/probes/probe_backend_preset_hardware.py``.
    - Each artifact's external objects are built IMMEDIATELY BEFORE ITS OWN ELF,
      never all up front. ``compile_gemm_mm`` names its output from the method
      alone while baking ``tile_n`` into it, so two operators of the same method
      at different ``tile_n`` overwrite each other's object and aiecc links
      whichever was written last. That cost this phase a run: see
      ``compile_block_artifacts``, which has the measurement.
    - ``attn_npu2.o`` survives that ordering only because
      ``compile_attention_kernel`` forces a rebuild. ``KernelCache`` runs
      ``compile_all_external_kernels`` on EVERY compile, which builds a generic
      ``head_dim=64`` attention object if none exists and skips one that does;
      the forced rebuild in ``compile_mha_kernels`` is what puts this design's
      tile flags back. Drop the force and the attention kernel hangs.
    - The weights are declared ``static=True`` with a ``content_key``, so they
      land in the content-keyed pool and are uploaded once. ``x`` is NOT static:
      it is an activation, and the row bands of it that sequence B reads are
      distinct buffers with distinct contents.
    - A cached ELF is reused only when its FINGERPRINT matches. An artifact name
      carries the shape and nothing else; it does not carry the registry row the
      tiles came from or the builder that emitted the IR, and reusing a binary
      on the strength of its name alone runs the gate against code that is no
      longer here. ``builders/block_cache.py`` owns that decision.
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

import shared.infra.external_kernels as ek  # noqa: E402
from shared.infra.bo_pool import BufferSpec, DispatchStep, content_key  # noqa: E402

from builders.addnorm import (  # noqa: E402
    addnorm_max_rows,
    build_addnorm_module,
    compile_addnorm_kernel,
)
from builders.block_cache import (  # noqa: E402
    block_artifact_fingerprint,
    load_fingerprints,
    save_fingerprints,
)
from builders.ffn import (  # noqa: E402
    build_ffn_module,
    ffn_arg_layout,
    ffn_gemm_specs,
)
from builders.mha_out_proj import (  # noqa: E402
    MHA_OUT_PROJ_RUNNER_KWARGS,
    build_mha_out_proj_module,
    compile_mha_out_proj_kernels,
    mha_out_proj_arg_layout,
    mha_out_proj_config,
)
from builders.qkv_proj import build_qkv_proj_module, qkv_gemm_spec  # noqa: E402

#: The host tensors the layer takes, in the order ``opcheck.py`` indexes them
#: for fault injection. See the module docstring.
BLOCK_INPUT_NAMES = (
    "x",
    "w_qkv",
    "w_o",
    "ln1_weight",
    "w_up",
    "w_down",
    "ln2_weight",
)

#: Every boundary ``run_block`` reads back, in execution order. Matches
#: ``pattern/reference.py::ENCODER_BOUNDARIES`` name for name; the per-boundary
#: comparison pairs them up by name rather than by position, so a reordering
#: here is caught rather than silently comparing the wrong pair.
BLOCK_BOUNDARIES = (
    "q",
    "k",
    "v",
    "attn_context",
    "attn_out",
    "hidden",
    "ffn_up",
    "ffn_gelu",
    "ffn_out",
    "output",
)

#: AIE columns each row-blocked ``addnorm`` dispatch runs on. The operator's
#: herd is ``herd_x x 1``; see builders/addnorm.py.
NORM_HERD_X = 8

#: Fraction of ``addnorm_max_rows`` the row block is allowed to reach.
#: ``addnorm_max_rows`` counts ALLOCATIONS and ignores aircc's ping-pong copies
#: of the DMA-fed buffers, so a shape sitting at the cap can still fail to
#: place. At ``emb_dim = 768`` the cap is 104 and this picks 64 -- exactly the
#: shape D1 measured, which is the point: the block dispatches the operator that
#: was validated rather than a wider one that was not.
NORM_ROW_MARGIN = 0.75

#: Backend settings for the two GEMM-backed artifacts. ``runtime_loop_tiling_
#: sizes`` is BD-ID recycling for a fused-cast herd; ``output_format`` is forced
#: by the multi-launch structure. Both are the settings D1 validated these two
#: operators at, unchanged.
_GEMM_BACKEND = {
    "runtime_loop_tiling_sizes": [2, 2],
    "output_format": "elf",
    "omit_while_true_loop": False,
}

#: ``addnorm`` is single-launch and needs none of the GEMM settings, but it does
#: need the ELF ABI so it can share a dispatch sequence with the rest.
_ADDNORM_BACKEND = {
    "output_format": "elf",
    "omit_while_true_loop": False,
}


def norm_rows(seq_len, emb_dim, herd_x=NORM_HERD_X):
    """Rows per ``addnorm`` dispatch: the largest legal block of ``seq_len``.

    Legal means all four at once: a multiple of ``herd_x`` (one kernel call per
    tile), a divisor of ``seq_len`` (the bands tile the activation exactly), at
    most ``NORM_ROW_MARGIN`` of the L1 cap, and non-zero.

    DERIVED, never carried over from another width. The cap moves with
    ``emb_dim`` -- 104 rows at 768, 80 at 1024 -- and a row count that happened
    to fit at one width is a placement failure at the next.
    """
    cap = int(addnorm_max_rows(emb_dim, herd_x=herd_x, pre_add=True) * NORM_ROW_MARGIN)
    legal = [
        rows
        for rows in range(herd_x, cap + 1, herd_x)
        if seq_len % rows == 0 and rows <= cap
    ]
    if not legal:
        raise ValueError(
            f"no legal addnorm row block for seq_len={seq_len} emb_dim={emb_dim}: "
            f"needs a divisor of seq_len that is a multiple of {herd_x} and at "
            f"most {cap} rows (L1 cap "
            f"{addnorm_max_rows(emb_dim, herd_x=herd_x, pre_add=True)} at "
            f"{NORM_ROW_MARGIN:.0%} margin). Raise herd_x or pick a sequence "
            f"length with a usable divisor; do NOT raise the row count past the "
            f"cap, which miscompiles rather than failing."
        )
    return max(legal)


def block_config(seq_len, emb_dim, ffn_dim, num_heads, head_dim):
    """Resolve every operator's configuration without building anything.

    Split from the builders the same way ``mha_out_proj_config`` and
    ``ffn_gemm_specs`` are, so ``opcheck.py`` can record what was resolved
    without building the modules twice.

    Raises if ``num_heads * head_dim != emb_dim``: the fused QKV weight, the
    attention layout and the output projection all assume it, and none of them
    would report the mismatch as a mismatch.
    """
    if num_heads * head_dim != emb_dim:
        raise ValueError(
            f"num_heads * head_dim ({num_heads} * {head_dim}) must equal emb_dim "
            f"({emb_dim}); the seq-first attention layout and the fused QKV "
            f"weight both assume it"
        )

    qkv_spec, qkv_source = qkv_gemm_spec(seq_len, emb_dim)
    up_spec, down_spec, ffn_source = ffn_gemm_specs(seq_len, emb_dim, ffn_dim)
    attn_cfg, o_spec, o_source = mha_out_proj_config(
        seq_len, head_dim, num_heads, causal=False
    )
    rows = norm_rows(seq_len, emb_dim)

    _, mha_idx = mha_out_proj_arg_layout(attn_cfg, o_spec)
    _, ffn_idx = ffn_arg_layout(seq_len, emb_dim, ffn_dim, up_spec, down_spec)

    suffix = f"{seq_len}x{emb_dim}"
    artifacts = {
        "qkv_proj": f"blk_qkv_proj_{suffix}",
        "mha_out_proj": f"blk_mha_out_proj_{suffix}x{num_heads}h",
        "ffn": f"blk_ffn_{suffix}x{ffn_dim}",
        "addnorm": f"blk_addnorm_{rows}x{emb_dim}_pre_add",
    }
    return {
        "seq_len": seq_len,
        "emb_dim": emb_dim,
        "ffn_dim": ffn_dim,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "qkv_spec": qkv_spec,
        "qkv_source": qkv_source,
        "ffn_up_spec": up_spec,
        "ffn_down_spec": down_spec,
        "ffn_source": ffn_source,
        "attn_cfg": attn_cfg,
        "o_proj_spec": o_spec,
        "o_proj_source": o_source,
        "norm_rows": rows,
        "norm_blocks": seq_len // rows,
        "mha_idx": mha_idx,
        "ffn_idx": ffn_idx,
        "artifacts": artifacts,
        "backend_kwargs": {
            artifacts["qkv_proj"]: dict(_GEMM_BACKEND, instance_name="qkv_proj"),
            artifacts["mha_out_proj"]: dict(
                MHA_OUT_PROJ_RUNNER_KWARGS, instance_name="mha_out_proj"
            ),
            artifacts["ffn"]: dict(_GEMM_BACKEND, instance_name="ffn"),
            artifacts["addnorm"]: dict(_ADDNORM_BACKEND, instance_name="addnorm"),
        },
    }


def _compile_gemm_object(spec):
    """One GEMM micro-kernel object, at the tile shape its spec names."""
    ek.compile_gemm_mm(
        tile_m=spec["tile_m"],
        tile_n=spec["tile_n"],
        tile_k_l1=spec["tile_k_l1"],
        sym_suffix=spec["sym_suffix"],
        out_name=spec["obj"],
    )


def compile_qkv_proj_kernels(cfg):
    """External objects the QKV projection ELF links."""
    _compile_gemm_object(cfg["qkv_spec"])


def compile_ffn_kernels(cfg):
    """External objects the FFN ELF links.

    Both projections' micro-kernels and encoder.cc's FFN half. The addnorm half
    is deliberately not built into this object: it would collide with
    addnorm_ffn.o on ``ffn_gelu_bf16``, and this ELF needs neither.
    """
    _compile_gemm_object(cfg["ffn_up_spec"])
    _compile_gemm_object(cfg["ffn_down_spec"])
    ek.compile_encoder(build_ffn=True, build_addnorm=False, out_name="encoder_ffn.o")


def compile_addnorm_kernels(cfg):
    """External object the addnorm ELF links: the pre-add half only."""
    compile_addnorm_kernel(pre_add=True)


def compile_mha_kernels(cfg):
    """External objects the fused attention + projection ELF links.

    ``attn_npu2.o`` at THIS design's tile shapes -- the flags are derived from
    the same config the builder uses, and a mismatch does not fail to link, it
    HANGS -- plus the o-projection's micro-kernel.
    """
    compile_mha_out_proj_kernels(cfg["attn_cfg"], cfg["o_proj_spec"])


#: Per-artifact (external objects, module builder). The ORDER of the pairs does
#: not matter; the PAIRING does. See ``compile_block_artifacts``.
_ARTIFACT_BUILD = {
    "qkv_proj": (
        compile_qkv_proj_kernels,
        lambda cfg: build_qkv_proj_module(cfg["seq_len"], cfg["emb_dim"]),
    ),
    "mha_out_proj": (
        compile_mha_kernels,
        lambda cfg: build_mha_out_proj_module(
            cfg["seq_len"], cfg["head_dim"], cfg["num_heads"], causal=False
        ),
    ),
    "ffn": (
        compile_ffn_kernels,
        lambda cfg: build_ffn_module(cfg["seq_len"], cfg["emb_dim"], cfg["ffn_dim"]),
    ),
    "addnorm": (
        compile_addnorm_kernels,
        lambda cfg: build_addnorm_module(
            cfg["norm_rows"], cfg["emb_dim"], bfloat16, herd_x=NORM_HERD_X, pre_add=True
        ),
    ),
}


def compile_block_artifacts(cache, cfg, run_only=False):
    """Compile the four ELFs into ``cache``, reusing only exact matches.

    EVERY EXTERNAL OBJECT FIRST, THEN EVERY ELF. The order used to be the other
    way round -- each artifact's objects built immediately before its own ELF --
    and that interleaving was a correctness requirement, not tidiness. It is
    HISTORICAL now, and the history is worth keeping because the failure it
    worked around was silent:

        ``compile_gemm_mm`` named its output from the GEMM METHOD ALONE
        (``mm_m32.o`` for drain, ``mm_m64.o`` for fused-cast) while its
        ``DIM_N`` was baked in from ``tile_n``. Two operators of the same method
        at different ``tile_n`` therefore wrote the SAME FILE with different
        contents, and aiecc linked whichever was written last.

        In this block that was not hypothetical: the FFN's up-projection is
        drain at ``tile_n = 128`` and the o-projection is drain at
        ``tile_n = 96``. Building every object up front and then every ELF gave
        the FFN a 96-wide micro-kernel for its 128-wide tile, and it did not
        fail -- it returned exactly zero for 32 of every 128 output columns, 25%
        of the up-projection, which the GeLU then passed through and the
        down-projection's 3072-deep reduction smeared over every element of the
        FFN output. The per-boundary stage list is what localized it to
        ``ffn_up``; the layer output alone said only that 54% of the layer was
        wrong.

        D1 never met this because each operator ran in its own ``opcheck.py``
        invocation and no single operator holds two same-method GEMMs at
        different ``tile_n``. Any caller that built several of these operators
        together did.

    Phase E1 fixed the root cause instead: ``gemm_builder.gemm_variant_names``
    mints both the object name and the symbol suffix per ``(tile_m, tile_n)``, so
    those two GEMMs are now ``mm_m32n128.o`` and ``mm_m32n96.o`` and cannot
    overwrite each other. The interleaving is therefore removed -- but note what
    still holds and what does not. It is safe to build all the objects first
    because the NAMES are distinct; it would not be safe again if any caller went
    back to naming an object by anything coarser than its full tile shape. That
    is the invariant, not the ordering.

    ``run_only`` REUSES A CACHED ELF ONLY WHERE ITS FINGERPRINT STILL MATCHES,
    and the name is not the fingerprint -- ``builders/block_cache.py`` has the
    argument for why, and what the digest covers. So every artifact's module is
    built on every call and hashed before the decision is taken. Building all
    four costs about 0.1s against the minutes of ELF compilation the answer
    gates. A miss recompiles that artifact and only that one, and only a miss
    rebuilds its external objects.
    """
    names = cfg["artifacts"]
    have_manifest = bool(run_only and cache.load_manifest())
    recorded = load_fingerprints(cache) if have_manifest else {}

    fingerprints = {}
    reused = []
    modules = {}
    stale = []
    for key, (_, build_module) in _ARTIFACT_BUILD.items():
        name = names[key]
        module = build_module(cfg)
        fingerprints[name] = block_artifact_fingerprint(cfg, key, module)
        if name in cache.artifacts and recorded.get(name) == fingerprints[name]:
            reused.append(name)
            continue
        modules[key] = module
        stale.append(key)

    for key in stale:
        print(f"== external objects for {names[key]} ==")
        _ARTIFACT_BUILD[key][0](cfg)
    for key in stale:
        name = names[key]
        print(f"== compiling {name} ==")
        cache.compile_and_cache(name, modules[key], cfg["backend_kwargs"][name])
    if reused:
        print(f"  reusing {len(reused)} cached block artifacts: {', '.join(reused)}")
    cache._save_manifest()
    save_fingerprints(cache, fingerprints)


def _spec(name, array, static=False, host_output=False):
    """One ``BufferSpec``, sized and keyed from the array that backs it."""
    return BufferSpec(
        name=name,
        nbytes=array.size * array.itemsize,
        static=static,
        host_output=host_output,
        content_key=content_key(array) if static else None,
    )


def _ordered_args(idx, rename):
    """A step's argument names in signature order, from a layout's index map.

    ``idx`` is ``{layout_name: position}`` as the builder's ``*_arg_layout``
    returns it; ``rename`` maps those layout names onto this block's buffer
    names. Reading the order back out of the layout rather than writing it down
    here is what keeps this correct when a registry method change adds or drops
    an f32 scratch.
    """
    return tuple(rename[n] for n, _ in sorted(idx.items(), key=lambda kv: kv[1]))


def _write_indices(idx, layout_names):
    """Positions of the arguments the kernel writes, for ``DispatchStep``."""
    return tuple(sorted(idx[n] for n in layout_names if n in idx))


def _copy_out(results, names):
    """Copy every named result out of pool memory (rule H1).

    ``run_sequence``'s results are zero-copy views into the pool, and the next
    sequence reuses those slots. A boundary read after the fact without this is
    whatever the next launch happened to write there.
    """
    return {name: np.array(results[name], copy=True) for name in names}


def _sequence_a(cache, cfg, x, w_qkv, w_o):
    """qkv_proj then mha_out_proj, fully device-resident between them."""
    seq_len, emb_dim = cfg["seq_len"], cfg["emb_dim"]
    names = cfg["artifacts"]
    n_total = 3 * emb_dim

    def zeros_act():
        return np.zeros((seq_len, emb_dim), dtype=bfloat16)

    arrays = {
        "x": x,
        "w_qkv": w_qkv,
        "w_o": w_o,
        "qkv_f32": np.zeros((seq_len, n_total), dtype=np.float32),
        "q": zeros_act(),
        "k": zeros_act(),
        "v": zeros_act(),
        "attn_context": zeros_act(),
        "attn_out": zeros_act(),
    }
    rename = {
        "q": "q",
        "k": "k",
        "v": "v",
        "attn_out": "attn_context",
        "w_o": "w_o",
        "y_f32": "attn_f32",
        "y": "attn_out",
    }
    idx = cfg["mha_idx"]
    if "y_f32" in idx:
        arrays["attn_f32"] = np.zeros((seq_len, emb_dim), dtype=np.float32)

    steps = [
        DispatchStep(
            names["qkv_proj"],
            ("x", "w_qkv", "qkv_f32", "q", "k", "v"),
            writes=(2, 3, 4, 5),
        ),
        DispatchStep(
            names["mha_out_proj"],
            _ordered_args(idx, rename),
            writes=_write_indices(idx, ("attn_out", "y_f32", "y")),
        ),
    ]
    outputs = ("q", "k", "v", "attn_context", "attn_out")
    specs = {
        name: _spec(
            name,
            array,
            static=name in ("w_qkv", "w_o"),
            host_output=name in outputs,
        )
        for name, array in arrays.items()
    }
    # The f32 scratches are handed to the device zero-filled, exactly as D1's
    # per-operator runs hand them to XRTRunner. They only appear at written
    # positions, so `default_host_writes` would call them produced and skip the
    # upload; naming them here keeps the block on the configuration the operator
    # gate measured rather than on one that merely ought to be equivalent.
    host_writes = {"x", "w_qkv", "w_o", "qkv_f32"} & set(arrays)
    host_writes |= {"attn_f32"} & set(arrays)
    results, vector = cache.run_sequence(
        steps, specs, cfg["backend_kwargs"], arrays, host_writes=host_writes
    )
    return _copy_out(results, outputs), vector


def _sequence_norm(cache, cfg, label, x_full, residual_full, weight):
    """One normalization point, row-blocked into ``norm_blocks`` dispatches.

    ``x_full`` and ``residual_full`` are whole ``[seq_len, emb_dim]`` tensors;
    they are cut into bands here because a dispatch argument is a whole BO. The
    weight is one buffer shared by every band and declared static, so it is
    uploaded once rather than ``norm_blocks`` times.
    """
    rows, blocks = cfg["norm_rows"], cfg["norm_blocks"]
    name = cfg["artifacts"]["addnorm"]
    weight_name = f"{label}_weight"

    arrays = {weight_name: weight}
    steps = []
    out_names = []
    for i in range(blocks):
        lo, hi = i * rows, (i + 1) * rows
        x_name, r_name, o_name = (
            f"{label}_x{i}",
            f"{label}_r{i}",
            f"{label}_out{i}",
        )
        # Contiguous copies, not views: a BufferSpec is sized from the array and
        # the host write reads it flat, so a non-contiguous slice would upload
        # the wrong bytes without complaining.
        arrays[x_name] = np.ascontiguousarray(x_full[lo:hi])
        arrays[r_name] = np.ascontiguousarray(residual_full[lo:hi])
        arrays[o_name] = np.zeros((rows, cfg["emb_dim"]), dtype=bfloat16)
        steps.append(
            DispatchStep(name, (x_name, r_name, weight_name, o_name), writes=(3,))
        )
        out_names.append(o_name)

    specs = {
        n: _spec(
            n,
            a,
            static=n == weight_name,
            host_output=n in out_names,
        )
        for n, a in arrays.items()
    }
    results, vector = cache.run_sequence(steps, specs, cfg["backend_kwargs"], arrays)
    return np.concatenate([np.array(results[n], copy=True) for n in out_names]), vector


def _sequence_ffn(cache, cfg, hidden, w_up, w_down):
    """The staged FFN, with its two interior boundaries read back."""
    seq_len, emb_dim, ffn_dim = cfg["seq_len"], cfg["emb_dim"], cfg["ffn_dim"]
    idx = cfg["ffn_idx"]
    rename = {
        "x": "hidden",
        "w_up": "w_up",
        "h_f32": "ffn_up_f32",
        "h": "ffn_up",
        "a": "ffn_gelu",
        "w_down": "w_down",
        "y_f32": "ffn_out_f32",
        "y": "ffn_out",
    }
    arrays = {
        "hidden": hidden,
        "w_up": w_up,
        "w_down": w_down,
        "ffn_up": np.zeros((seq_len, ffn_dim), dtype=bfloat16),
        "ffn_gelu": np.zeros((seq_len, ffn_dim), dtype=bfloat16),
        "ffn_out": np.zeros((seq_len, emb_dim), dtype=bfloat16),
    }
    if "h_f32" in idx:
        arrays["ffn_up_f32"] = np.zeros((seq_len, ffn_dim), dtype=np.float32)
    if "y_f32" in idx:
        arrays["ffn_out_f32"] = np.zeros((seq_len, emb_dim), dtype=np.float32)

    steps = [
        DispatchStep(
            cfg["artifacts"]["ffn"],
            _ordered_args(idx, rename),
            writes=_write_indices(idx, ("h_f32", "h", "a", "y_f32", "y")),
        )
    ]
    outputs = ("ffn_up", "ffn_gelu", "ffn_out")
    specs = {
        n: _spec(n, a, static=n in ("w_up", "w_down"), host_output=n in outputs)
        for n, a in arrays.items()
    }
    host_writes = {"hidden", "w_up", "w_down"} | (
        {"ffn_up_f32", "ffn_out_f32"} & set(arrays)
    )
    results, vector = cache.run_sequence(
        steps, specs, cfg["backend_kwargs"], arrays, host_writes=host_writes
    )
    return _copy_out(results, outputs), vector


def run_block(cache, cfg, inputs):
    """Dispatch the whole layer and return every device boundary.

    Args:
        cache: a ``KernelCache`` holding the four artifacts.
        cfg: ``block_config(...)``.
        inputs: the tensors named by ``BLOCK_INPUT_NAMES``, in that order. This
            is the list ``opcheck.py`` perturbs under ``--fault-inject``, so it
            is taken positionally and unpacked here rather than by name.

    Returns:
        ``(boundaries, vector_rows)``. ``boundaries`` holds a COPY of each of
        ``BLOCK_BOUNDARIES``; nothing in it aliases pool memory.
    """
    x, w_qkv, w_o, ln1_weight, w_up, w_down, ln2_weight = inputs

    print("  [1/4] qkv_proj + mha_out_proj (one sequence, device-resident q/k/v)")
    attn, vec_a = _sequence_a(cache, cfg, x, w_qkv, w_o)

    print(f"  [2/4] addnorm ln1 x{cfg['norm_blocks']} (pre-add)")
    hidden, vec_b = _sequence_norm(cache, cfg, "ln1", attn["attn_out"], x, ln1_weight)

    print("  [3/4] ffn")
    ffn, vec_c = _sequence_ffn(cache, cfg, hidden, w_up, w_down)

    print(f"  [4/4] addnorm ln2 x{cfg['norm_blocks']} (pre-add)")
    output, vec_d = _sequence_norm(
        cache, cfg, "ln2", ffn["ffn_out"], hidden, ln2_weight
    )

    boundaries = dict(attn)
    boundaries["hidden"] = hidden
    boundaries.update(ffn)
    boundaries["output"] = output
    missing = [n for n in BLOCK_BOUNDARIES if n not in boundaries]
    if missing:
        raise AssertionError(f"run_block produced no value for {missing}")
    return boundaries, [v.as_row() for v in (vec_a, vec_b, vec_c, vec_d)]


def describe_block(cfg):
    """One line per operator, naming what the registry resolved for it."""
    print(
        f"  block {cfg['seq_len']}x{cfg['emb_dim']} ffn {cfg['ffn_dim']} "
        f"{cfg['num_heads']}h x {cfg['head_dim']} (encoder_bert, non-causal)"
    )
    print(
        f"    qkv_proj  {cfg['qkv_spec']['method']} ({cfg['qkv_source']}), "
        f"ffn up {cfg['ffn_up_spec']['method']} / down "
        f"{cfg['ffn_down_spec']['method']} ({cfg['ffn_source']}), "
        f"o_proj {cfg['o_proj_spec']['method']} ({cfg['o_proj_source']})"
    )
    print(
        f"    addnorm pre-add {cfg['norm_rows']}x{cfg['emb_dim']} "
        f"x{cfg['norm_blocks']} dispatches "
        f"(L1 cap "
        f"{addnorm_max_rows(cfg['emb_dim'], herd_x=NORM_HERD_X, pre_add=True)})"
    )
