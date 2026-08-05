# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``runlist`` — the layer decomposed into single-operator entries over five runlists.

CONTRACT
    ``prepare_runlist(shape, seed=...)`` is this mode's entry in the ``SPECS``
    catalogue: the D2 layer prepared for ``opcheck.py``'s ``dispatch`` seam.
    The mode holds ``coarse``'s dispatch schedule FIXED and refines every one
    of its dispatch units into single-operator kernels, each
    ``KernelCache.run_sequence`` call forced to one runlist with
    ``require_single_submission=True``::

        coarse's unit                        this mode's refinement
        ------------------------------       --------------------------------
        qkv_proj (fused)              ->     runlist 1: q_proj  k_proj  v_proj
        mha_out_proj (fused)          ->     host blocked attention (shared
                                             with offload), then
                                             runlist 2: output_proj
        64 x addnorm ln1 (pre-add)    ->     runlist 3: 64 x (add  ln  mul)
        ffn (fused up+gelu+down)      ->     runlist 4: up_proj  gelu  down_proj
        64 x addnorm ln2 (pre-add)    ->     runlist 5: 64 x (add  ln  mul)

    Five recorded ``DispatchVector`` rows; the driver-summed totals are 5
    submissions over ``7 + 6 * norm_blocks`` runlist entries — 391 at the gate
    configuration, against ``coarse``'s 131. Every coarse dispatch unit maps
    onto one or more finer units, so ``runlist_entries > coarse`` holds BY
    CONSTRUCTION, which is what the mode's ordinal claim ("the fine-grained
    point of the taxonomy") means. Intermediates inside a runlist stay
    DEVICE-RESIDENT: q/k/v never touch the host before attention reads them
    back, each band's residual sum feeds its LayerNorm and gamma multiply on
    device, and ``ffn_up``/``ffn_gelu`` chain into the down projection.

WHY THE NORM CHAINS ARE ROW-BANDED WHEN THE KERNELS COULD STREAM
    The decomposed ``elementwise_add``/``layer_norm``/``elementwise_mul``
    kernels have no L1 cap forcing 64-row dispatches — each can walk all 4096
    rows in one launch, and the first structure tried did exactly that: 13
    entries over 2 runlists, which landed BELOW coarse's 131 and failed the
    one ordinal clause this mode owns. That structure changed TWO variables at
    once — operator granularity AND the dispatch schedule — and at the
    normalization points it was 64x COARSER-grained than ``coarse`` itself
    (one streaming launch against 64 banded dispatches), so its entry count
    measured the schedule change, not the decomposition. This structure holds
    the schedule fixed at coarse's own — the band size is IMPORTED from
    ``builders.block.norm_rows``, the L1-derived cap coarse measured, never
    re-derived or tuned here — so the two modes differ in exactly one
    variable, operator granularity, and the entry comparison measures it.

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
      The band ``add``/``ln``/``mul`` ELFs execute 64 times per chain and 128
      times per layer in one context each: no runtime loop tiling, the same
      class as the block's 64-fold re-executed ``addnorm`` ELF, measured clean
      at every stage boundary.
    - The LayerNorm gamma is applied by ``elementwise_mul`` against ONE
      host-materialized ``[norm_rows, emb]`` broadcast of the ``[emb]`` weight
      (``broadcast_row_weight``), declared static and content-keyed, shared by
      all 64 band multiplies — every band multiplies by the same rows. Under
      fault injection the content key changes and the perturbed broadcast is
      re-uploaded; nothing special-cases the injected path.
    - The mode computes; the oracle checks. Attention is torch
      (``blocked_attention``), every device operator is its own kernel; the
      per-boundary references come from the numpy oracles behind
      ``pattern/reference.py``. Nothing here may import
      ``addnorm_pre_add_reference``, ``gelu_tanh_reference`` or any other
      function that computes a boundary.
    - The dispatch vectors are recorded on the fault-injected path too. The
      driver requires the fault artifact's summed totals to EQUAL the clean
      run's; anything conditional on the injected flag fails that.
    - Band inputs are CONTIGUOUS COPIES, never views: a ``BufferSpec`` is
      sized from the array and the host write reads it flat, so a
      non-contiguous slice would upload the wrong bytes without complaining.
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

from builders.block import norm_rows  # noqa: E402
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
#: the shared query-blocked implementation, NOT a device dispatch. Its two
#: GEMMs (``4096 x 64 x 4096`` and ``4096 x 4096 x 64``) resolve in no
#: registry and the C4 sweep cannot stage them (08c has the derivation), so
#: this seam — the one submission boundary a whole-BO argument does not
#: explain — is forced, and it is why iron's ``k_transpose`` entry has no
#: counterpart here: its consumer is the GEMM this hardware cannot dispatch.
#: The ``transpose`` operator is validated standalone instead.
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

#: The single-launch operators need none of the GEMM settings, but they do
#: need the ELF ABI to share a runlist with the GEMMs.
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

    Three GEMM shapes serve six of the entries, but the four
    ``[seq, emb] @ [emb, emb]`` projections each get their OWN compiled ELF of
    the same module: a runtime-tiled GEMM ELF returns wrong numbers from its
    second execution in one ``hw_context`` onward, and this mode MEASURED that
    the corruption holds inside a single runlist too (see the module
    footguns), so no GEMM ELF may appear twice in the dispatch. The band
    ``add``/``ln``/``mul`` operators are re-execution clean and built at
    ``coarse``'s row granularity — ``builders.block.norm_rows``, the L1 cap
    the fused ``addnorm`` measured, imported so the two modes share one
    schedule by construction. Raises (via the registry) on an unmeasured GEMM
    shape, on ``num_heads * head_dim != emb_dim``, and (via the builders) on
    a band shape the small operators cannot tile.
    """
    if num_heads * head_dim != emb_dim:
        raise ValueError(
            f"num_heads * head_dim ({num_heads} * {head_dim}) must equal emb_dim "
            f"({emb_dim}); the head reshape around host attention assumes it"
        )
    rows = norm_rows(seq_len, emb_dim)

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
            "add": f"rl_add_{rows}x{emb_dim}",
            "ln": f"rl_ln_{rows}x{emb_dim}",
            "mul": f"rl_mul_{rows}x{emb_dim}",
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
        "norm_rows": rows,
        "norm_blocks": seq_len // rows,
        "specs": specs,
        "gemms": gemms,
        "artifacts": artifacts,
        "backend_kwargs": backend_kwargs,
        "query_block_size": resolve_query_block_size(seq_len, num_heads),
    }


def runlist_entry_count(cfg):
    """Total runlist entries the decomposition dispatches, derived not counted:
    q/k/v (3) + output_proj (1) + up/gelu/down (3) + two norm chains of
    ``3 * norm_blocks`` band entries each."""
    return 7 + 6 * cfg["norm_blocks"]


def describe_runlist(cfg):
    """One line per resolved decision, for the run log and the lit gate."""
    print(
        f"  runlist {cfg['seq_len']}x{cfg['emb_dim']} ffn {cfg['ffn_dim']} "
        f"{cfg['num_heads']}h x {cfg['head_dim']} (encoder_bert, non-causal, "
        f"{runlist_entry_count(cfg)} fine-grained entries over 5 runlists, "
        f"attention in host torch)"
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
    print(
        f"    norm chains banded at {cfg['norm_rows']} rows x"
        f"{cfg['norm_blocks']} (builders.block.norm_rows — coarse's schedule)"
    )


def _build_runlist_module(cfg, key):
    """One artifact's module: a registry GEMM, a band operator, or gelu."""
    emb_dim, ffn_dim = cfg["emb_dim"], cfg["ffn_dim"]
    rows = cfg["norm_rows"]
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
        return build_elementwise_add_module(rows, emb_dim, bfloat16)
    if key == "ln":
        return build_layer_norm_module(rows, emb_dim, bfloat16)
    if key == "mul":
        return build_elementwise_mul_module(rows, emb_dim, bfloat16)
    if key == "gelu":
        return build_gelu_module(cfg["seq_len"], ffn_dim, bfloat16)
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
    execution boundary: coarse's schedule refined into single-operator
    entries, five runlists with host torch attention after the first.
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

    def _run_o_proj(ctx, w_o):
        """Runlist 2: the output projection, one entry."""
        arrays = {"ctx": ctx, "w_o": w_o}
        arrays["attn_out"] = np.zeros((seq_len, emb_dim), dtype=bfloat16)
        step, scratch = _gemm_step(cfg, "o_proj", "ctx", "w_o", "attn_out", arrays)
        specs = {
            name: _spec_buf(
                name, arr, static=name == "w_o", host_output=name == "attn_out"
            )
            for name, arr in arrays.items()
        }
        host_writes = {"ctx", "w_o"} | ({scratch} if scratch else set())
        results, vector = cache.run_sequence(
            [step],
            specs,
            cfg["backend_kwargs"],
            arrays,
            host_writes=host_writes,
            require_single_submission=True,
        )
        return np.array(results["attn_out"], copy=True), vector

    def _run_norm_chain(label, x_full, residual_full, gamma_band):
        """Runlists 3 and 5: one normalization point as ``norm_blocks`` bands
        of add -> LayerNorm -> gamma multiply, ``3 * norm_blocks`` entries in
        one submission.

        ``x_full`` and ``residual_full`` are whole ``[seq_len, emb_dim]``
        tensors cut into bands here, because a dispatch argument is a whole
        BO. Within a band the residual sum and the normalized rows stay
        device-resident; ``gamma_band`` is ONE static buffer shared by every
        band's multiply.
        """
        rows, blocks = cfg["norm_rows"], cfg["norm_blocks"]
        gamma_name = f"{label}_gamma"
        arrays = {gamma_name: gamma_band}
        steps = []
        out_names = []
        for i in range(blocks):
            lo, hi = i * rows, (i + 1) * rows
            x_name, r_name = f"{label}_x{i}", f"{label}_r{i}"
            s_name, n_name = f"{label}_sum{i}", f"{label}_norm{i}"
            o_name = f"{label}_out{i}"
            # Contiguous copies, not views — see the module footguns.
            arrays[x_name] = np.ascontiguousarray(x_full[lo:hi])
            arrays[r_name] = np.ascontiguousarray(residual_full[lo:hi])
            arrays[s_name] = np.zeros((rows, emb_dim), dtype=bfloat16)
            arrays[n_name] = np.zeros((rows, emb_dim), dtype=bfloat16)
            arrays[o_name] = np.zeros((rows, emb_dim), dtype=bfloat16)
            steps.append(
                DispatchStep(names["add"], (x_name, r_name, s_name), writes=(2,))
            )
            steps.append(DispatchStep(names["ln"], (s_name, n_name), writes=(1,)))
            steps.append(
                DispatchStep(names["mul"], (n_name, gamma_name, o_name), writes=(2,))
            )
            out_names.append(o_name)
        specs = {
            name: _spec_buf(
                name, arr, static=name == gamma_name, host_output=name in out_names
            )
            for name, arr in arrays.items()
        }
        host_writes = {gamma_name}
        for i in range(blocks):
            host_writes |= {f"{label}_x{i}", f"{label}_r{i}"}
        results, vector = cache.run_sequence(
            steps,
            specs,
            cfg["backend_kwargs"],
            arrays,
            host_writes=host_writes,
            require_single_submission=True,
        )
        out = np.concatenate([np.array(results[n], copy=True) for n in out_names])
        return out, vector

    def _run_ffn(hidden, w_up, w_down):
        """Runlist 4: up_proj, GeLU, down_proj — three entries, the
        interiors device-resident."""
        arrays = {
            "hidden": hidden,
            "w_up": w_up,
            "w_down": w_down,
            "ffn_up": np.zeros((seq_len, ffn_dim), dtype=bfloat16),
            "ffn_gelu": np.zeros((seq_len, ffn_dim), dtype=bfloat16),
            "ffn_out": np.zeros((seq_len, emb_dim), dtype=bfloat16),
        }
        steps = []
        scratches = set()
        step, scratch = _gemm_step(cfg, "up", "hidden", "w_up", "ffn_up", arrays)
        steps.append(step)
        if scratch:
            scratches.add(scratch)
        steps.append(DispatchStep(names["gelu"], ("ffn_up", "ffn_gelu"), writes=(1,)))
        step, scratch = _gemm_step(cfg, "down", "ffn_gelu", "w_down", "ffn_out", arrays)
        steps.append(step)
        if scratch:
            scratches.add(scratch)
        outputs = ("ffn_up", "ffn_gelu", "ffn_out")
        specs = {
            name: _spec_buf(
                name,
                arr,
                static=name in ("w_up", "w_down"),
                host_output=name in outputs,
            )
            for name, arr in arrays.items()
        }
        host_writes = {"hidden", "w_up", "w_down"} | scratches
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
        blocks = cfg["norm_blocks"]

        print("  [runlist 1/5] q_proj + k_proj + v_proj (3 entries, one submission)")
        proj, vec_1 = _run_projections(x, w_q, w_k, w_v)

        print(f"  [host] blocked attention, query block {cfg['query_block_size']}")
        attn_context = blocked_attention(
            proj["q"],
            proj["k"],
            proj["v"],
            num_heads,
            causal=False,
            query_block_size=cfg["query_block_size"],
        )

        print("  [runlist 2/5] output_proj (1 entry, one submission)")
        attn_out, vec_2 = _run_o_proj(round_bf16(attn_context), w_o)

        print(
            f"  [runlist 3/5] {blocks} x (add + layer_norm + mul) ln1 "
            f"({3 * blocks} entries, one submission)"
        )
        gamma1 = round_bf16(broadcast_row_weight(ln1_weight, cfg["norm_rows"]))
        hidden, vec_3 = _run_norm_chain("ln1", attn_out, x, gamma1)

        print("  [runlist 4/5] up_proj + gelu + down_proj (3 entries, one submission)")
        ffn, vec_4 = _run_ffn(hidden, w_up, w_down)

        print(
            f"  [runlist 5/5] {blocks} x (add + layer_norm + mul) ln2 "
            f"({3 * blocks} entries, one submission)"
        )
        gamma2 = round_bf16(broadcast_row_weight(ln2_weight, cfg["norm_rows"]))
        output, vec_5 = _run_norm_chain("ln2", ffn["ffn_out"], hidden, gamma2)

        boundaries = dict(ffn)
        boundaries.update({"q": proj["q"], "k": proj["k"], "v": proj["v"]})
        boundaries["attn_context"] = attn_context
        boundaries["attn_out"] = attn_out
        boundaries["hidden"] = hidden
        boundaries["output"] = output

        vector_rows = [v.as_row() for v in (vec_1, vec_2, vec_3, vec_4, vec_5)]
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
        "norm_rows": cfg["norm_rows"],
        "norm_blocks": cfg["norm_blocks"],
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
