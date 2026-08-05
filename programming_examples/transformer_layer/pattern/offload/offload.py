# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``offload`` — the host owns the layer and sends one GEMM at a time to the device.

CONTRACT
    ``prepare_offload(shape, seed=...)`` is this mode's entry in the ``SPECS``
    catalogue: the D2 layer prepared for ``opcheck.py``'s ``dispatch`` seam.
    The host holds every intermediate and dispatches exactly SIX registry
    GEMMs, each as a one-step ``KernelCache.run_sequence`` call::

        q_proj  k_proj  v_proj  output_proj  up_proj  down_proj

    Six calls, six recorded ``DispatchVector`` rows, each with one host
    submission holding one runlist entry — summed by the driver that is six
    submissions and six entries, which is what "aggregates nothing" means.
    Everything else — attention, softmax, both normalization points, the GeLU,
    reshapes, residuals — is host torch. This is the host-mediated extreme of
    the Phase E taxonomy: the most submissions, the most sync boundaries, no
    aggregation at all.

SIX GEMMS, NOT EIGHT, AND WHY THAT IS A DECISION RATHER THAN A SHORTCUT
    iron's offload dispatches eight, including the two attention GEMMs. On
    this device those two cannot be registry GEMMs: at the gate configuration
    they are ``4096 x 64 x 4096`` (attn_scores) and ``4096 x 4096 x 64``
    (attn_output), and no ``K = 64`` or ``N = 64`` bf16-out row exists or can
    be swept (08c has the derivation). So ATTENTION STAYS IN HOST TORCH —
    ``pattern/blocked_attention.py``, shared with ``runlist`` — and this mode
    is a HYBRID boundary, not a pure per-GEMM device implementation. The
    artifact records ``attention_path`` so the mode's own record says which
    boundary it actually drew.

THE RULE THAT DECIDES WHETHER THIS MODE MEASURES ANYTHING
    The mode computes; the oracle checks. They may not share arithmetic. This
    module does more host math than any other mode, and every piece of it is
    written against torch — ``F.layer_norm``, ``F.gelu(approximate="tanh")``,
    ``blocked_attention``'s torch softmax — while the oracle's boundaries come
    from the numpy operator references behind ``pattern/reference.py``. What
    this module imports from the reference: the golden draws, the boundary
    NAMES. What it must never import: ``addnorm_pre_add_reference``,
    ``gelu_tanh_reference``, ``chunked_attention_reference`` or any other
    function that computes a boundary — a stage whose "actual" and "expected"
    are the same call compares a value against itself.

FOOTGUNS
    - ``OFFLOAD_CACHE_DIR`` is this mode's OWN ELF cache, in
      ``transformer_layer/.gitignore`` AND the Makefile ``clean`` target.
      ``KernelCache`` picks the directory by NAME, so two modes sharing one
      can trade ELFs whose fingerprints happen to agree — numerically valid
      output attributed to the wrong execution boundary.
    - The six dispatches are six SEPARATE ``run_sequence`` calls on purpose.
      Under the ELF ABI ``run_sequence`` merges every step it is given into
      ONE submission, so batching the six steps into one call would record
      one submission over six entries — which is ``coarse``, not this mode.
    - q/k/v/output_proj share ONE compiled ELF (they are the same
      ``4096x768x768`` module); up and down get their own. Sharing the binary
      is not aggregation — each dispatch is still its own submission — and
      compiling three ELFs instead of six is real minutes on every gate.
    - Weights are NOT declared static. Six weight uploads per layer is the
      mode being itself; do not optimize it.
    - EVERY DISPATCH RUNS IN A FRESH ``hw_context``. Re-executing one of
      these GEMM ELFs in a reused context returns wrong numbers from the
      second execution onward — measured, with the stale-input and
      stale-output explanations ruled out; see ``_evict_context``. Nothing
      else in the example re-executes a GEMM ELF across submissions, so this
      mode is where the failure was found. If a later mode re-executes one
      inside a single runlist, measure before assuming either behaviour.
    - The dispatch vectors are recorded on the fault-injected path too. The
      driver requires the fault artifact's summed totals to EQUAL the clean
      run's; anything conditional on the injected flag fails that.
    - ``execution_mode`` comes from ``pattern.EXECUTION_MODE_CSV`` — for this
      mode the code name and the CSV value coincide, but the mapping still
      has exactly one home. Do not inline the string.
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
from shared.infra.bo_pool import BufferSpec, DispatchStep  # noqa: E402

from builders.block_cache import (  # noqa: E402
    block_artifact_fingerprint,
    load_fingerprints,
    save_fingerprints,
)
from builders.gemm_spec import resolve_gemm_spec, spec_herd  # noqa: E402
from opcheck_prepare import (  # noqa: E402
    BLOCK_STAGE_ATOL,
    _spec_digest,
    dispatch_vector_totals,
)
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
OFFLOAD_CACHE_DIR = "offload_cache"

#: The six device GEMMs, in dispatch order. The order is the layer's dataflow
#: and each entry becomes one recorded DispatchVector row.
OFFLOAD_GEMMS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "output_proj",
    "up_proj",
    "down_proj",
)

#: The host tensors the layer takes, in the order ``opcheck.py`` indexes them
#: for fault injection. The q/k/v weights are SEPARATE here — this mode
#: dispatches three projections, not one fused ``w_qkv``.
OFFLOAD_INPUT_NAMES = (
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

#: Recorded in the artifact: this mode's attention boundary is host torch f32
#: through the shared query-blocked implementation, NOT a device dispatch.
ATTENTION_PATH = "host_torch_fp32_blocked"

#: The single func.func each GEMM method's module emits, which is what
#: ``instance_name`` must equal — a mismatch does not fail to load, it times
#: out with ERT_CMD_STATE_TIMEOUT a long way from the cause.
_METHOD_FUNC = {"drain": "matmul_bf16", "fused-cast": "gemm_cast_bf16"}

#: Backend settings for every offload artifact: BD-ID recycling for the GEMM
#: herds, and the ELF ABI so the fused-cast module's two launches package at
#: all. The settings D1 validated the GEMM-backed operators at, unchanged.
_GEMM_BACKEND = {
    "runtime_loop_tiling_sizes": [2, 2],
    "output_format": "elf",
    "omit_while_true_loop": False,
}


def offload_config(seq_len, emb_dim, ffn_dim, num_heads, head_dim):
    """Resolve every GEMM's registry spec and this mode's artifact map.

    Three distinct GEMM shapes serve the six dispatches: q/k/v/output_proj are
    all ``[seq, emb] @ [emb, emb]``, so they share one compiled module. Raises
    (via the registry) on an unmeasured shape rather than guessing, and on
    ``num_heads * head_dim != emb_dim``, which nothing downstream would report
    as a mismatch.
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
    artifacts = {key: f"off_gemm_{m}x{k}x{n}" for key, (_, (m, k, n)) in specs.items()}
    backend_kwargs = {
        artifacts[key]: dict(_GEMM_BACKEND, instance_name=_METHOD_FUNC[spec["method"]])
        for key, (spec, _) in specs.items()
    }
    gemms = {
        "q_proj": "proj",
        "k_proj": "proj",
        "v_proj": "proj",
        "output_proj": "proj",
        "up_proj": "up",
        "down_proj": "down",
    }
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


def describe_offload(cfg):
    """One line per resolved decision, for the run log and the lit gate."""
    print(
        f"  offload {cfg['seq_len']}x{cfg['emb_dim']} ffn {cfg['ffn_dim']} "
        f"{cfg['num_heads']}h x {cfg['head_dim']} (encoder_bert, non-causal, "
        f"hybrid boundary: 6 device GEMMs, attention in host torch)"
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


def _build_offload_module(cfg, key):
    """One plain GEMM module at its registry spec. Method picks the func name."""
    from shared.builders.gemm_builder import _build_gemm_module

    spec, (m, k, n) = cfg["specs"][key]
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


def compile_offload_artifacts(cache, cfg, run_only=False):
    """Compile the three GEMM ELFs into ``cache``, reusing only exact matches.

    Same shape as ``builders/block.py::compile_block_artifacts`` and reusing
    its fingerprint machinery verbatim: every module is built and hashed on
    every call (about 0.1s against minutes of ELF compilation), a cached ELF
    is reused only when its recorded fingerprint matches, and only a miss
    rebuilds — including that artifact's external ``mm_*.o``, whose name is a
    function of ``(tile_m, tile_n)`` so the three variants cannot overwrite
    each other.
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
        module = _build_offload_module(cfg, key)
        fingerprints[name] = block_artifact_fingerprint(cfg, key, module)
        if name in cache.artifacts and recorded.get(name) == fingerprints[name]:
            reused.append(name)
            continue
        modules[key] = module
        stale.append(key)

    for key in stale:
        spec, _ = cfg["specs"][key]
        print(f"== external objects for {names[key]} ==")
        ek.compile_gemm_mm(
            tile_m=spec["tile_m"],
            tile_n=spec["tile_n"],
            tile_k_l1=spec["tile_k_l1"],
            sym_suffix=spec["sym_suffix"],
            out_name=spec["obj"],
        )
    for key in stale:
        name = names[key]
        print(f"== compiling {name} ==")
        cache.compile_and_cache(name, modules[key], cfg["backend_kwargs"][name])
    if reused:
        print(f"  reusing {len(reused)} cached offload artifacts: {', '.join(reused)}")
    cache._save_manifest()
    save_fingerprints(cache, fingerprints)


def _evict_context(cache, artifact):
    """Drop ``artifact``'s cached ``hw_context`` (and every pool) before a dispatch.

    MEASURED, NOT DEFENSIVE: re-executing one of these runtime-tiled GEMM
    ELFs in a reused ``hw_context`` returns wrong numbers from the SECOND
    execution onward — mean_rel_L1 3.56e-1 against the same run's own 9.6e-3
    on a fresh context, with the same inputs, uniformly across rows and
    columns, at roughly one third of the reduction lost. Stale-input and
    stale-output hypotheses were ruled out directly (the wrong output matches
    neither the previous weights' product nor the previous result); the
    corruption is device-side state the ELF leaves behind. The block gate
    never sees this because each of its GEMM ELFs executes exactly once per
    process, and its re-executed addnorm ELF (no runtime loop tiling) re-runs
    clean.

    So this mode reloads the context per dispatch. The pools go with it: their
    BOs were allocated against the evicted backend's device wrapper, and a
    fresh per-dispatch pool keeps every buffer's provenance one dispatch wide
    — which is also this mode's semantics, since nothing may stay device
    resident between GEMMs. The dispatch-vector counts are unchanged (nothing
    is static, so every call already uploads all of its inputs).
    """
    loaded = cache._loaded.pop(artifact, None)
    if loaded is not None:
        loaded[0].unload()
    cache._pools.clear()


def _dispatch_gemm(cache, cfg, op_name, a, b):
    """One GEMM on the device, as ONE one-step ``run_sequence`` call.

    Returns ``(c, vector_row)`` where ``c`` is a COPY of the bf16 output
    (``run_sequence`` returns zero-copy views into pool memory, and the next
    dispatch reuses the slot) and ``vector_row`` is the call's
    ``DispatchVector.as_row()`` — one submission, one entry, recorded by the
    shared implementation and never hand-built here. Each dispatch runs in a
    FRESH ``hw_context`` — see ``_evict_context`` for the measurement that
    forces that.
    """
    key = cfg["gemms"][op_name]
    spec, (m, k, n) = cfg["specs"][key]
    artifact = cfg["artifacts"][key]
    _evict_context(cache, artifact)

    arrays = {
        "a": np.ascontiguousarray(a),
        "b": np.ascontiguousarray(b),
        "c": np.zeros((m, n), dtype=bfloat16),
    }
    if spec["needs_f32_scratch"]:
        # The fused-cast module's f32 C scratch: an input slot the device
        # writes, uploaded zero-filled exactly as every D1 run handed it to
        # XRTRunner. It appears only at a written position, so it must be
        # named in host_writes or the upload is skipped.
        arrays["c_f32"] = np.zeros((m, n), dtype=np.float32)
        args = ("a", "b", "c_f32", "c")
        writes = (2, 3)
        host_writes = {"a", "b", "c_f32"}
    else:
        args = ("a", "b", "c")
        writes = (2,)
        host_writes = {"a", "b"}

    specs = {
        name: BufferSpec(
            name=name,
            nbytes=arr.size * arr.itemsize,
            static=False,
            host_output=name == "c",
            content_key=None,
        )
        for name, arr in arrays.items()
    }
    steps = [DispatchStep(artifact, args, writes=writes)]
    results, vector = cache.run_sequence(
        steps, specs, cfg["backend_kwargs"], arrays, host_writes=host_writes
    )
    return np.array(results["c"], copy=True), vector.as_row()


def _torch_f32(array):
    """bf16 (or f32) numpy -> torch float32, bridging bf16 through an int16 view."""
    import torch

    array = np.ascontiguousarray(array)
    if array.dtype == bfloat16:
        return (
            torch.from_numpy(array.view(np.int16))
            .view(torch.bfloat16)
            .to(torch.float32)
        )
    return torch.from_numpy(array.astype(np.float32, copy=False))


def _host_addnorm(x, residual, weight):
    """Host pre-add addnorm: ``LayerNorm(x + residual) * weight``, f32, to bf16.

    torch's ``F.layer_norm`` with the gamma passed in and a zero bias — the
    weight multiplies the normalized value, which IS the pre-add form the
    encoder uses at both normalization points. Independent of the numpy
    oracle in ``builders/addnorm.py`` by construction (torch vs numpy), same
    ``eps = 1e-5``. Returns bf16 because the boundary feeds a device GEMM.
    """
    import torch

    weight_f32 = _torch_f32(weight)
    normed = torch.nn.functional.layer_norm(
        _torch_f32(x) + _torch_f32(residual),
        (weight_f32.shape[-1],),
        weight=weight_f32,
        bias=torch.zeros_like(weight_f32),
        eps=1e-5,
    )
    return round_bf16(normed.numpy())


def _host_gelu(x):
    """Host tanh-approximation GeLU in f32, unrounded.

    ``approximate="tanh"`` because the layer's activation IS the tanh form
    (the golden model pins it by identity); torch's implementation keeps this
    independent of the numpy ``gelu_tanh_reference`` the oracle uses. Returns
    f32 — the boundary the reference leaves unrounded — and the caller rounds
    to bf16 only where the tensor actually feeds the device.
    """
    import torch

    return torch.nn.functional.gelu(_torch_f32(x), approximate="tanh").numpy()


def prepare_offload(shape, seed=42):
    """The ``offload`` mode's ``SPECS`` preparer: the D2 layer, host-mediated.

    Same golden model, same per-boundary comparisons at ``BLOCK_STAGE_ATOL``,
    same injection target (``ln1_weight``, chosen by the measurement in
    ``opcheck_prepare.py``'s block section — every attention-side candidate
    puts ZERO elements outside the band, and this mode's attention is host
    f32, which damps them further, not less). What differs is the execution
    boundary: six one-step device GEMM dispatches, host torch for everything
    between them.
    """
    seq_len, emb_dim = shape["seq_len"], shape["emb_dim"]
    ffn_dim, num_heads = shape["ffn_dim"], shape["num_heads"]
    head_dim = shape["head_dim"]

    cfg = offload_config(seq_len, emb_dim, ffn_dim, num_heads, head_dim)
    describe_offload(cfg)

    golden = generate_golden_reference(
        seq_len, emb_dim, ffn_dim, num_heads, seed=seed, workload_variant="encoder_bert"
    )
    weights = golden["weights"]
    reference = golden["boundaries"]
    # The order is OFFLOAD_INPUT_NAMES; `inject` below indexes into it. The
    # q/k/v weights stay separate — this mode dispatches three projections.
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
        cache_dir=OFFLOAD_CACHE_DIR, verbose=False, profiler=Profiler(enabled=False)
    )
    compile_offload_artifacts(cache, cfg, run_only=True)

    def dispatch(device_inputs, stage_stats):
        x, w_q, w_k, w_v, w_o, ln1_weight, w_up, w_down, ln2_weight = device_inputs

        vector_rows = []

        def gemm(op_name, a, b):
            out, row = _dispatch_gemm(cache, cfg, op_name, a, b)
            vector_rows.append(row)
            return out

        print("  [1/6] q_proj + [2/6] k_proj + [3/6] v_proj (one dispatch each)")
        q = gemm("q_proj", x, w_q)
        k = gemm("k_proj", x, w_k)
        v = gemm("v_proj", x, w_v)
        print(f"  [host] blocked attention, query block {cfg['query_block_size']}")
        attn_context = blocked_attention(
            q,
            k,
            v,
            num_heads,
            causal=False,
            query_block_size=cfg["query_block_size"],
        )
        print("  [4/6] output_proj")
        attn_out = gemm("output_proj", round_bf16(attn_context), w_o)
        hidden = _host_addnorm(attn_out, x, ln1_weight)
        print("  [5/6] up_proj")
        ffn_up = gemm("up_proj", hidden, w_up)
        ffn_gelu = _host_gelu(ffn_up)
        print("  [6/6] down_proj")
        ffn_out = gemm("down_proj", round_bf16(ffn_gelu), w_down)
        output = _host_addnorm(ffn_out, hidden, ln2_weight)

        boundaries = {
            "q": q,
            "k": k,
            "v": v,
            "attn_context": attn_context,
            "attn_out": attn_out,
            "hidden": hidden,
            "ffn_up": ffn_up,
            "ffn_gelu": ffn_gelu,
            "ffn_out": ffn_out,
            "output": output,
        }
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
        print(f"[offload] stages: {clean}/{len(stages)} clean")
        # On the fault path too — the FAULT half of the lit recipe matches the
        # two lines below against the SAME literals as the clean half, so
        # instrumentation conditional on the injected flag, a malformed row
        # (`dispatch_vector_totals` raises, failing the run) or drifted totals
        # fail in the suite before the driver's comparison sees them.
        print(f"[offload] recorded {len(vector_rows)} dispatch vectors")
        totals = dispatch_vector_totals(vector_rows)
        print(
            f"[offload] dispatch totals: "
            f"submissions {totals['host_submissions']} "
            f"entries {totals['runlist_entries']} "
            f"air {totals['air_launches']} "
            f"herd {totals['herd_launches']} "
            f"sync {totals['sync_boundaries']} "
            f"bytes {totals['bytes_transferred']}"
        )
        return [boundaries["output"]], {
            "stages": stages,
            "stages_passed": clean == len(stages),
            "dispatch_vectors": vector_rows,
        }

    record_extra = {
        "variant": "encoder_bert",
        "causal": False,
        "golden_seed": seed,
        "execution_mode": EXECUTION_MODE_CSV["offload"],
        "attention_path": ATTENTION_PATH,
        "query_block_size": cfg["query_block_size"],
        "gemm_spec_source": "registry",
        "gemm_spec_proj": _spec_digest(cfg["specs"]["proj"][0]),
        "gemm_spec_ffn_up": _spec_digest(cfg["specs"]["up"][0]),
        "gemm_spec_ffn_down": _spec_digest(cfg["specs"]["down"][0]),
    }
    return {
        "inputs": inputs,
        "expected": [reference["output"]],
        # ln1_weight, index 5. See the docstring for why the block's measured
        # choice carries over to this mode.
        "inject": (OFFLOAD_INPUT_NAMES.index("ln1_weight"), (0,)),
        "dispatch": dispatch,
        "record_extra": record_extra,
    }
