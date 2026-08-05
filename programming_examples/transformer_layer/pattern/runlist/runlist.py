# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``runlist`` — the layer decomposed into small operators over two runlists.

CONTRACT
    ``prepare_runlist(shape, seed=...)`` is this mode's entry in the ``SPECS``
    catalogue: the D2 layer prepared for ``opcheck.py``'s ``dispatch`` seam.
    The layer is decomposed into THIRTEEN single-operator dispatch steps over
    two ``KernelCache.run_sequence`` calls, each forced to one runlist with
    ``require_single_submission=True``::

        runlist 1 (3 entries)   q_proj  k_proj  v_proj
        -- host --              blocked attention (shared with offload)
        runlist 2 (10 entries)  output_proj  add  layer_norm  mul(gamma)
                                up_proj  gelu  down_proj  add  layer_norm
                                mul(gamma)

    Two recorded ``DispatchVector`` rows; the driver-summed totals are 2
    submissions over 13 runlist entries. Intermediates between device steps
    inside a runlist stay DEVICE-RESIDENT (``attn_out`` feeds the residual add
    without touching the host; ``hidden`` feeds both the up-projection and the
    second residual add) — that chaining is what this mode exists to measure.

WHY TWO RUNLISTS AND NOT ONE
    Attention. Its two GEMMs (``4096 x 64 x 4096`` and ``4096 x 4096 x 64``)
    resolve in no registry and cannot be swept (08c has the derivation), so
    attention is host torch through ``pattern/blocked_attention.py`` — the SAME
    implementation and query blocking ``offload`` uses, per 08d work item 4.
    Everything before it and everything after it is one runlist each; the host
    boundary in the middle is a measurement, not a shortcut, and it is why
    iron's ``k_transpose`` entry has no counterpart here: its consumer (the
    on-device scores GEMM) is the thing this hardware cannot dispatch, and a
    device transpose feeding a host ``@`` that re-layouts anyway would measure
    nothing. The ``transpose`` operator is validated standalone instead.

THE ENTRY COUNT IS A MEASUREMENT, AND IT LANDS *BELOW* ``coarse``'S
    13 entries against ``coarse``'s 131. Not because this mode is coarser —
    it dispatches ten distinct operator kernels where ``coarse`` dispatches
    five — but because 128 of ``coarse``'s 131 entries are ``addnorm``'s row
    blocking (one kernel call per tile caps it at 64 rows per dispatch at
    width 768), while every operator THIS mode decomposes to streams its rows
    inside one launch: ``elementwise_add``, ``layer_norm``, ``elementwise_mul``
    and ``gelu`` all walk 4096 rows in a single entry. The fine-grained
    decomposition therefore has FEWER runlist entries than the coarse one at
    ``baseline_768`` — 08d anticipates exactly this ("a decomposition that
    folds normalization back into a fused kernel can easily come out below
    it") and prescribes reporting the number rather than inflating the
    decomposition, e.g. by row-blocking operators that do not need it. The
    ``runlist_entries > coarse`` ordinal clause fails on this hardware and
    that is a finding about the measurement model, recorded in this mode's
    README.

FOOTGUNS
    - ``RUNLIST_CACHE_DIR`` is this mode's OWN ELF cache, in
      ``transformer_layer/.gitignore`` AND the Makefile ``clean`` target.
      ``KernelCache`` picks the directory by NAME, so two modes sharing one
      can trade ELFs whose fingerprints happen to agree.
    - NO GEMM ELF EXECUTES TWICE, and that is a measurement, not caution.
      q/k/v/output_proj are one module compiled FOUR times to four artifacts,
      each with its own ``hw_context``. The first bring-up shared one ELF for
      q/k/v inside one runlist, and executions two and three returned the
      exact corruption signature ``offload`` measured across submissions —
      k at 3.539e-1 and v at 3.561e-1 mean_rel_L1 against q's clean 9.3e-3,
      offload's measured mode being 3.56e-1 — so the reused-context failure
      holds INSIDE a single runlist too, and context eviction (offload's fix)
      is unavailable there: entries of one artifact share its context by
      construction. Four compiles of one module per clean cache is the cost.
      The ``add``/``layer_norm``/``mul`` ELFs DO execute twice inside runlist
      2: no runtime loop tiling, same class as the block's 64-fold re-executed
      ``addnorm`` ELF, and the same first bring-up measured their second
      executions clean (the ffn/output stages sat at their expected error
      levels while k/v were corrupt).
    - The LayerNorm gamma is applied by ``elementwise_mul`` against a
      HOST-MATERIALIZED ``[seq, emb]`` broadcast of the ``[emb]`` weight
      (``broadcast_row_weight``), declared static and content-keyed. 6 MB per
      norm point instead of 1.5 KB is the honest cost of decomposing to a
      two-tensor multiply; the builder's docstring records why there is no
      broadcast form.
    - The mode computes; the oracle checks. Attention is torch
      (``blocked_attention``), every device operator is its own kernel; the
      per-boundary references come from the numpy oracles behind
      ``pattern/reference.py``. Nothing here may import
      ``addnorm_pre_add_reference``, ``gelu_tanh_reference`` or any other
      function that computes a boundary.
    - The dispatch vectors are recorded on the fault-injected path too. The
      driver requires the fault artifact's summed totals to EQUAL the clean
      run's; anything conditional on the injected flag fails that.
    - ``execution_mode`` comes from ``pattern.EXECUTION_MODE_CSV``. Do not
      inline the string.
"""

import os
import sys

import numpy as np
from ml_dtypes import bfloat16

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE_ROOT = os.path.dirname(os.path.dirname(_HERE))  # transformer_layer/
_PROJ_ROOT = os.path.dirname(_EXAMPLE_ROOT)  # programming_examples/
for _p in (_PROJ_ROOT, os.path.join(_PROJ_ROOT, "llms"), _EXAMPLE_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import shared.infra.external_kernels as ek  # noqa: E402
from shared.infra.bo_pool import BufferSpec, DispatchStep, content_key  # noqa: E402

from builders.block_cache import (  # noqa: E402
    block_artifact_fingerprint,
    load_fingerprints,
    save_fingerprints,
)
from builders.elementwise_add import build_elementwise_add_module  # noqa: E402
from builders.elementwise_mul import (  # noqa: E402
    broadcast_row_weight,
    build_elementwise_mul_module,
)
from builders.gelu import build_gelu_module  # noqa: E402
from builders.gemm_spec import resolve_gemm_spec, spec_herd  # noqa: E402
from builders.layer_norm import build_layer_norm_module  # noqa: E402
from opcheck_layer import BLOCK_STAGE_ATOL, print_dispatch_totals  # noqa: E402
from opcheck_prepare import _spec_digest  # noqa: E402
from pattern import EXECUTION_MODE_CSV  # noqa: E402
from pattern.blocked_attention import (  # noqa: E402
    blocked_attention,
    resolve_query_block_size,
    round_bf16,
)
from pattern.reference import (  # noqa: E402
    ENCODER_BOUNDARIES,
    generate_golden_reference,
)

#: This mode's ELF cache, relative to the working directory. Its OWN — see the
#: module footguns.
RUNLIST_CACHE_DIR = "runlist_cache"

#: The host tensors the layer takes, in the order ``opcheck.py`` indexes them
#: for fault injection. Identical to offload's: q/k/v weights SEPARATE, because
#: this mode dispatches three projections rather than one fused ``w_qkv``.
RUNLIST_INPUT_NAMES = (
    "x",
    "w_q",
    "w_k",
    "w_v",
    "w_o",
    "ln1_weight",
    "w_up",
    "w_down",
    "ln2_weight",
)

#: Recorded in the artifact: the attention boundary is host torch f32 through
#: the shared query-blocked implementation, NOT a device dispatch.
ATTENTION_PATH = "host_torch_fp32_blocked"

#: The single func.func each GEMM method's module emits — what instance_name
#: must equal, or the dispatch times out a long way from the cause.
_METHOD_FUNC = {"drain": "matmul_bf16", "fused-cast": "gemm_cast_bf16"}

#: Backend settings for the GEMM artifacts: BD-ID recycling and the ELF ABI,
#: the settings D1 validated the GEMM-backed operators at, unchanged.
_GEMM_BACKEND = {
    "runtime_loop_tiling_sizes": [2, 2],
    "output_format": "elf",
    "omit_while_true_loop": False,
}

#: The streaming single-launch operators need none of the GEMM settings, but
#: they do need the ELF ABI to share a runlist with the GEMMs.
_SMALL_BACKEND = {
    "output_format": "elf",
    "omit_while_true_loop": False,
}

#: instance_name per small artifact key — each is the emitted func.func name,
#: asserted at build time in ``compile_runlist_artifacts``.
_SMALL_FUNC = {
    "add": "eltwise_add_2d",
    "ln": "layer_norm_multi_row",
    "mul": "eltwise_mul_2d",
    "gelu": "ffn_gelu_2d",
}


def runlist_config(seq_len, emb_dim, ffn_dim, num_heads, head_dim):
    """Resolve every operator's configuration without building anything.

    Three GEMM shapes serve six of the thirteen entries, but the four
    ``[seq, emb] @ [emb, emb]`` projections each get their OWN compiled ELF of
    the same module: a runtime-tiled GEMM ELF returns wrong numbers from its
    second execution in one ``hw_context`` onward, and this mode MEASURED that
    the corruption holds inside a single runlist too (see the module
    footguns), so no GEMM ELF may appear twice in the dispatch. Four compiles
    of one module is the price of thirteen entries over two submissions; the
    streaming operators are re-execution clean and keyed by their L3 shape.
    Raises (via the registry) on an unmeasured GEMM shape, and on
    ``num_heads * head_dim != emb_dim``.
    """
    if num_heads * head_dim != emb_dim:
        raise ValueError(
            f"num_heads * head_dim ({num_heads} * {head_dim}) must equal emb_dim "
            f"({emb_dim}); the head reshape around host attention assumes it"
        )

    specs = {
        "proj": (
            resolve_gemm_spec(seq_len, emb_dim, emb_dim),
            (seq_len, emb_dim, emb_dim),
        ),
        "up": (
            resolve_gemm_spec(seq_len, emb_dim, ffn_dim),
            (seq_len, emb_dim, ffn_dim),
        ),
        "down": (
            resolve_gemm_spec(seq_len, ffn_dim, emb_dim),
            (seq_len, ffn_dim, emb_dim),
        ),
    }
    # GEMM entry -> spec key. Four distinct proj ELFs on purpose; see above.
    gemms = {
        "q_proj": "proj",
        "k_proj": "proj",
        "v_proj": "proj",
        "o_proj": "proj",
        "up": "up",
        "down": "down",
    }
    artifacts = {}
    for gemm_key, spec_key in gemms.items():
        _, (m, k, n) = specs[spec_key]
        artifacts[gemm_key] = f"rl_gemm_{gemm_key}_{m}x{k}x{n}"
    artifacts.update(
        {
            "add": f"rl_add_{seq_len}x{emb_dim}",
            "ln": f"rl_ln_{seq_len}x{emb_dim}",
            "mul": f"rl_mul_{seq_len}x{emb_dim}",
            "gelu": f"rl_gelu_{seq_len}x{ffn_dim}",
        }
    )
    backend_kwargs = {
        artifacts[gemm_key]: dict(
            _GEMM_BACKEND, instance_name=_METHOD_FUNC[specs[spec_key][0]["method"]]
        )
        for gemm_key, spec_key in gemms.items()
    }
    backend_kwargs.update(
        {
            artifacts[key]: dict(_SMALL_BACKEND, instance_name=func)
            for key, func in _SMALL_FUNC.items()
        }
    )
    return {
        "seq_len": seq_len,
        "emb_dim": emb_dim,
        "ffn_dim": ffn_dim,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "specs": specs,
        "gemms": gemms,
        "artifacts": artifacts,
        "backend_kwargs": backend_kwargs,
        "query_block_size": resolve_query_block_size(seq_len, num_heads),
    }


def describe_runlist(cfg):
    """One line per resolved decision, for the run log and the lit gate."""
    print(
        f"  runlist {cfg['seq_len']}x{cfg['emb_dim']} ffn {cfg['ffn_dim']} "
        f"{cfg['num_heads']}h x {cfg['head_dim']} (encoder_bert, non-causal, "
        f"13 fine-grained entries over 2 runlists, attention in host torch)"
    )
    parts = []
    for key in ("proj", "up", "down"):
        spec, (m, k, n) = cfg["specs"][key]
        parts.append(f"{key} {m}x{k}x{n} {spec['method']} (registry)")
    print("    " + ", ".join(parts))
    blocks = cfg["seq_len"] // cfg["query_block_size"]
    print(
        f"    attention host torch fp32, query block {cfg['query_block_size']} "
        f"x{blocks} block(s)"
    )


def _build_runlist_module(cfg, key):
    """One artifact's module: a registry GEMM or a streaming operator."""
    seq_len, emb_dim = cfg["seq_len"], cfg["emb_dim"]
    ffn_dim = cfg["ffn_dim"]
    if key in cfg["gemms"]:
        from shared.builders.gemm_builder import _build_gemm_module

        spec, (m, k, n) = cfg["specs"][cfg["gemms"][key]]
        return _build_gemm_module(
            m,
            k,
            n,
            spec["tile_m"],
            spec["tile_k_l2"],
            spec["tile_k_l1"],
            spec["tile_n"],
            *spec_herd(spec),
            **spec["build_kwargs"],
        )
    if key == "add":
        return build_elementwise_add_module(seq_len, emb_dim, bfloat16)
    if key == "ln":
        return build_layer_norm_module(seq_len, emb_dim, bfloat16)
    if key == "mul":
        return build_elementwise_mul_module(seq_len, emb_dim, bfloat16)
    if key == "gelu":
        return build_gelu_module(seq_len, ffn_dim, bfloat16)
    raise KeyError(f"unknown runlist artifact key {key!r}")


def compile_runlist_artifacts(cache, cfg, run_only=False):
    """Compile the ten ELFs into ``cache``, reusing only exact matches.

    Same shape as ``compile_block_artifacts`` and reusing its fingerprint
    machinery verbatim: every module is built and hashed on every call, a
    cached ELF is reused only when its recorded fingerprint matches, and only
    a miss rebuilds — including that artifact's external objects, whose names
    carry their tile shapes so variants cannot overwrite each other.
    """
    names = cfg["artifacts"]
    have_manifest = bool(run_only and cache.load_manifest())
    recorded = load_fingerprints(cache) if have_manifest else {}

    fingerprints = {}
    reused = []
    modules = {}
    stale = []
    for key in names:
        name = names[key]
        module = _build_runlist_module(cfg, key)
        if key in _SMALL_FUNC and _SMALL_FUNC[key] not in str(module):
            raise AssertionError(
                f"{name}: emitted module does not define {_SMALL_FUNC[key]}; "
                "instance_name would not match and the dispatch would time out"
            )
        fingerprints[name] = block_artifact_fingerprint(cfg, key, module)
        if name in cache.artifacts and recorded.get(name) == fingerprints[name]:
            reused.append(name)
            continue
        modules[key] = module
        stale.append(key)

    for key in stale:
        print(f"== external objects for {names[key]} ==")
        if key in cfg["gemms"]:
            spec, _ = cfg["specs"][cfg["gemms"][key]]
            ek.compile_gemm_mm(
                tile_m=spec["tile_m"],
                tile_n=spec["tile_n"],
                tile_k_l1=spec["tile_k_l1"],
                sym_suffix=spec["sym_suffix"],
                out_name=spec["obj"],
            )
        elif key == "ln":
            ek.compile_layer_norm()
        elif key == "gelu":
            # encoder.cc's FFN half only, same object name the ffn operator
            # links; both builds are identical so sharing the file is safe.
            ek.compile_encoder(
                build_ffn=True, build_addnorm=False, out_name="encoder_ffn.o"
            )
        # "add" and "mul" are direct vector codegen: nothing to compile.
    for key in stale:
        name = names[key]
        print(f"== compiling {name} ==")
        cache.compile_and_cache(name, modules[key], cfg["backend_kwargs"][name])
    if reused:
        print(f"  reusing {len(reused)} cached runlist artifacts: {', '.join(reused)}")
    cache._save_manifest()
    save_fingerprints(cache, fingerprints)


def _spec_buf(name, array, static=False, host_output=False):
    """One ``BufferSpec``, sized and keyed from the array that backs it."""
    return BufferSpec(
        name=name,
        nbytes=array.size * array.itemsize,
        static=static,
        host_output=host_output,
        content_key=content_key(array) if static else None,
    )


def _gemm_step(cfg, key, a_name, b_name, out_name, arrays):
    """One GEMM entry's ``DispatchStep`` (+ its f32 scratch, if the method
    stages one), appending any scratch array to ``arrays``.

    Returns ``(step, scratch_name_or_None)``. The scratch appears only at a
    written position, so the CALLER must name it in ``host_writes`` — exactly
    the fused-cast contract every D1 run used. ``key`` is a GEMM entry key
    (``cfg["gemms"]``), so the four projections resolve one shared spec but
    four distinct artifacts — the no-re-execution rule.
    """
    spec, (m, k, n) = cfg["specs"][cfg["gemms"][key]]
    artifact = cfg["artifacts"][key]
    if spec["needs_f32_scratch"]:
        scratch = f"{out_name}_f32"
        arrays[scratch] = np.zeros((m, n), dtype=np.float32)
        step = DispatchStep(
            artifact, (a_name, b_name, scratch, out_name), writes=(2, 3)
        )
        return step, scratch
    return DispatchStep(artifact, (a_name, b_name, out_name), writes=(2,)), None


def prepare_runlist(shape, seed=42):
    """The ``runlist`` mode's ``SPECS`` preparer: the D2 layer, fine-grained.

    Same golden model, same per-boundary comparisons at ``BLOCK_STAGE_ATOL``,
    same injection target (``ln1_weight`` — the measured choice; here it feeds
    the first gamma multiply, scaling one column of ``hidden`` and cascading
    through both residual paths exactly as in the block). What differs is the
    execution boundary: thirteen single-operator entries over two runlists,
    with host torch attention between them.
    """
    seq_len, emb_dim = shape["seq_len"], shape["emb_dim"]
    ffn_dim, num_heads = shape["ffn_dim"], shape["num_heads"]
    head_dim = shape["head_dim"]

    cfg = runlist_config(seq_len, emb_dim, ffn_dim, num_heads, head_dim)
    describe_runlist(cfg)

    golden = generate_golden_reference(
        seq_len, emb_dim, ffn_dim, num_heads, seed=seed, workload_variant="encoder_bert"
    )
    weights = golden["weights"]
    reference = golden["boundaries"]
    # The order is RUNLIST_INPUT_NAMES; `inject` below indexes into it.
    inputs = [
        golden["input"],
        weights["q_weight"],
        weights["k_weight"],
        weights["v_weight"],
        weights["attn_output_weight"],
        weights["ln1_weight"],
        weights["ffn_up_weight"],
        weights["ffn_down_weight"],
        weights["ln2_weight"],
    ]

    from shared.infra.cache import KernelCache, Profiler

    cache = KernelCache(
        cache_dir=RUNLIST_CACHE_DIR, verbose=False, profiler=Profiler(enabled=False)
    )
    compile_runlist_artifacts(cache, cfg, run_only=True)

    names = cfg["artifacts"]

    def _run_projections(x, w_q, w_k, w_v):
        """Runlist 1: q/k/v projections, three entries in one submission."""
        arrays = {"x": x, "w_q": w_q, "w_k": w_k, "w_v": w_v}
        steps = []
        scratches = set()
        for gemm_key, out_name, w_name in (
            ("q_proj", "q", "w_q"),
            ("k_proj", "k", "w_k"),
            ("v_proj", "v", "w_v"),
        ):
            arrays[out_name] = np.zeros((seq_len, emb_dim), dtype=bfloat16)
            step, scratch = _gemm_step(cfg, gemm_key, "x", w_name, out_name, arrays)
            steps.append(step)
            if scratch:
                scratches.add(scratch)
        outputs = ("q", "k", "v")
        specs = {
            name: _spec_buf(
                name,
                arr,
                static=name in ("w_q", "w_k", "w_v"),
                host_output=name in outputs,
            )
            for name, arr in arrays.items()
        }
        host_writes = {"x", "w_q", "w_k", "w_v"} | scratches
        results, vector = cache.run_sequence(
            steps,
            specs,
            cfg["backend_kwargs"],
            arrays,
            host_writes=host_writes,
            require_single_submission=True,
        )
        out = {n: np.array(results[n], copy=True) for n in outputs}
        return out, vector

    def _run_post_attention(ctx, x, w_o, gamma1, w_up, w_down, gamma2):
        """Runlist 2: output_proj through the second gamma, ten entries."""
        act = lambda: np.zeros((seq_len, emb_dim), dtype=bfloat16)  # noqa: E731
        wide = lambda: np.zeros((seq_len, ffn_dim), dtype=bfloat16)  # noqa: E731
        arrays = {
            "ctx": ctx,
            "x": x,
            "w_o": w_o,
            "gamma1": gamma1,
            "w_up": w_up,
            "w_down": w_down,
            "gamma2": gamma2,
            "attn_out": act(),
            "add1": act(),
            "ln1n": act(),
            "hidden": act(),
            "ffn_up": wide(),
            "ffn_gelu": wide(),
            "ffn_out": act(),
            "add2": act(),
            "ln2n": act(),
            "output": act(),
        }
        steps = []
        scratches = set()

        def gemm(key, a, b, out):
            step, scratch = _gemm_step(cfg, key, a, b, out, arrays)
            steps.append(step)
            if scratch:
                scratches.add(scratch)

        gemm("o_proj", "ctx", "w_o", "attn_out")
        steps.append(DispatchStep(names["add"], ("attn_out", "x", "add1"), writes=(2,)))
        steps.append(DispatchStep(names["ln"], ("add1", "ln1n"), writes=(1,)))
        steps.append(
            DispatchStep(names["mul"], ("ln1n", "gamma1", "hidden"), writes=(2,))
        )
        gemm("up", "hidden", "w_up", "ffn_up")
        steps.append(DispatchStep(names["gelu"], ("ffn_up", "ffn_gelu"), writes=(1,)))
        gemm("down", "ffn_gelu", "w_down", "ffn_out")
        steps.append(
            DispatchStep(names["add"], ("ffn_out", "hidden", "add2"), writes=(2,))
        )
        steps.append(DispatchStep(names["ln"], ("add2", "ln2n"), writes=(1,)))
        steps.append(
            DispatchStep(names["mul"], ("ln2n", "gamma2", "output"), writes=(2,))
        )

        # The boundaries the artifact compares; the decomposition's own
        # interiors (add1, ln1n, add2, ln2n) stay device-resident and are
        # covered collectively by `hidden` and `output`.
        outputs = ("attn_out", "hidden", "ffn_up", "ffn_gelu", "ffn_out", "output")
        statics = {"w_o", "gamma1", "w_up", "w_down", "gamma2"}
        specs = {
            name: _spec_buf(
                name, arr, static=name in statics, host_output=name in outputs
            )
            for name, arr in arrays.items()
        }
        host_writes = {"ctx", "x"} | statics | scratches
        results, vector = cache.run_sequence(
            steps,
            specs,
            cfg["backend_kwargs"],
            arrays,
            host_writes=host_writes,
            require_single_submission=True,
        )
        out = {n: np.array(results[n], copy=True) for n in outputs}
        return out, vector

    def dispatch(device_inputs, stage_stats):
        x, w_q, w_k, w_v, w_o, ln1_weight, w_up, w_down, ln2_weight = device_inputs

        print("  [runlist 1/2] q_proj + k_proj + v_proj (3 entries, one submission)")
        proj, vec_a = _run_projections(x, w_q, w_k, w_v)

        print(f"  [host] blocked attention, query block {cfg['query_block_size']}")
        attn_context = blocked_attention(
            proj["q"],
            proj["k"],
            proj["v"],
            num_heads,
            causal=False,
            query_block_size=cfg["query_block_size"],
        )

        print(
            "  [runlist 2/2] output_proj + add + layer_norm + mul + up_proj + "
            "gelu + down_proj + add + layer_norm + mul (10 entries, one submission)"
        )
        post, vec_b = _run_post_attention(
            round_bf16(attn_context),
            x,
            w_o,
            round_bf16(broadcast_row_weight(ln1_weight, seq_len)),
            w_up,
            w_down,
            round_bf16(broadcast_row_weight(ln2_weight, seq_len)),
        )

        boundaries = dict(post)
        boundaries.update({"q": proj["q"], "k": proj["k"], "v": proj["v"]})
        boundaries["attn_context"] = attn_context

        vector_rows = [vec_a.as_row(), vec_b.as_row()]
        stages = []
        for name in ENCODER_BOUNDARIES:
            atol = BLOCK_STAGE_ATOL[name]
            stats = stage_stats(boundaries[name], reference[name], atol=atol)
            stages.append(dict(stats, name=name, atol=atol))
            print(
                f"  [stage] {name:13s} {stats['n_elements']:>9d} elements  "
                f"mismatch {stats['n_mismatch']:>7d}  "
                f"mean_rel_L1 {stats['mean_rel_L1']:.3e}  "
                f"atol_required {stats['atol_required']:.3e} (atol {atol:.1e})"
            )
        clean = sum(1 for s in stages if s["n_mismatch"] == 0)
        print(f"[runlist] stages: {clean}/{len(stages)} clean")
        # On the fault path too — the FAULT half of the lit recipe pins the
        # printed totals to the same literals as the clean half.
        print_dispatch_totals("runlist", vector_rows)
        return [boundaries["output"]], {
            "stages": stages,
            "stages_passed": clean == len(stages),
            "dispatch_vectors": vector_rows,
        }

    record_extra = {
        "variant": "encoder_bert",
        "causal": False,
        "golden_seed": seed,
        "execution_mode": EXECUTION_MODE_CSV["runlist"],
        "attention_path": ATTENTION_PATH,
        "query_block_size": cfg["query_block_size"],
        "gemm_spec_source": "registry",
        "gemm_spec_proj": _spec_digest(cfg["specs"]["proj"][0]),
        "gemm_spec_ffn_up": _spec_digest(cfg["specs"]["up"][0]),
        "gemm_spec_ffn_down": _spec_digest(cfg["specs"]["down"][0]),
    }
    return {
        "inputs": inputs,
        # ln1_weight, index 5. Same measured target as the block's; see
        # opcheck_layer.py for the numbers.
        "inject": (RUNLIST_INPUT_NAMES.index("ln1_weight"), (0,)),
        "expected": [reference["output"]],
        "dispatch": dispatch,
        "record_extra": record_extra,
    }
