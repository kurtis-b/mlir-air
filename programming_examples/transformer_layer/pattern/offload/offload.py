# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``offload`` — every LINEAR operator on the device, every NON-LINEAR one on the host.

CONTRACT
    ``prepare_offload(shape, seed=...)`` is this mode's entry in the ``SPECS``
    catalogue: the D2 layer prepared for ``opcheck.py``'s ``dispatch`` seam.
    The host holds every intermediate and dispatches EIGHT linear operators,
    each as a one-step ``KernelCache.run_sequence`` call::

        q_proj  k_proj  v_proj  output_proj  up_proj  down_proj
        attn_scores x num_heads     attn_output x num_heads

    Six projections plus two attention matmuls per head: at the gate
    configuration's twelve heads that is 6 + 24 = 30 calls, 30 recorded
    ``DispatchVector`` rows, each with one host submission holding one runlist
    entry. Summed by the driver that is thirty submissions over thirty
    entries, which is what "aggregates nothing" means.

    What stays on the host is exactly the NON-LINEAR set: the softmax between
    the two attention matmuls, both normalization points, and the GeLU. Plus
    layout — head slicing, the K transpose, residual adds — which is data
    movement rather than arithmetic and is not what the partition is about.

WHAT THIS MODE ISOLATES, AND WHY IT IS NOT "THE HOST-MEDIATED EXTREME"
    `[2026-08-08]` The taxonomy was corrected by the study's author: the four
    modes span RECONFIGURATION COST AGAINST DRAM TRAFFIC, not "who sequences
    the work". ``offload`` is the mode that MINIMIZES reconfiguration — one
    xclbin, N instruction streams, one per GEMM shape, which is what iron
    implements. Its host/device split is decided by LINEARITY, not by which
    GEMMs happen to resolve in the registry. See 03 §The taxonomy.

    `[2026-08-09]` BOTH HALVES ARE BUILT. This module was the linearity half
    only; the N-streams-under-one-xclbin half landed in ``93e15a64`` and is
    reached through ``SHARED_XCLBIN`` below — five shapes chained into one
    xclbin, ``context_loads 1`` against the ELF path's 30, with the dispatch
    vector identical between them because the mode makes one ``run_sequence``
    call per GEMM either way. Both paths are gated by
    ``run_npu2_offload_peano.lit``, which pins the reconfiguration counters on
    each.

    Two limits travel with it, and neither is a to-do hidden in a docstring:

    - **The ELF path is still the DEFAULT**, so an unconfigured run pays a
      reconfiguration per dispatch and the sentence above describes it. The
      shared path is opt-in because it trades ~20% of best-case latency at 512
      for a ~20x reduction in run-to-run spread, and because flipping the
      default re-dates every recorded ``offload`` number (doc 29 §What this
      does not do).
    - **The shared path's 4096 wall is down at COMPILE time, pending the
      install rebuild.** At 4096 the down-projection is a two-launch
      ``fused-cast``; the backend's fixed ``air.insts.bin`` name used to
      collide with itself and bounded the chain to 1024. `[2026-08-10]`
      ``XRTBackend.compile`` now packages a multi-launch xclbin module as its
      "main" orchestration device -- one kernel, one instruction stream with
      per-launch ``load_pdi`` embedded, per-launch PDIs merged kernel-less
      into the partition -- and the five-shape chain builds at 4096 (doc 29
      §The 4096 wall, dated close-out; fixture
      ``test/xrt/56_multi_launch_xclbin_compile``). The change lives in
      ``python/air/backend/xrt.py``, so the RUNTIME sees it only after the
      coordinated ``install-xrt`` rebuild, and no multi-launch xclbin has yet
      DISPATCHED on hardware -- the gate stays at 1024 until that phase.

THE TWO ATTENTION MATMULS ARE LINEAR, SO THEY ARE ON THE DEVICE
    `[2026-08-08]` They resolve in no registry — ``sweep_families.py`` derives
    K and N from ``FAMILY_HIDDEN x ROLE_KN_MULTIPLES`` with a minimum hidden of
    512, so no ``--family`` can put a 64 in the K or N slot. That is a
    CATALOGUE constraint and not a hardware one: both shapes are measured
    passing on hardware at every rung of the study's ladder, by all three GEMM
    methods for ``attn_output``. The tiles below come from those measurements
    and are injected through the ``gemm_spec_fn`` escape hatch every builder
    here ships, recorded as ``gemm_spec_source: injected``. Nothing about this
    requires a registry write.

    This mode used to keep attention in host torch via
    ``pattern/blocked_attention.py`` and describe itself as a hybrid boundary.
    That was the superseded taxonomy plus the catalogue constraint read as a
    hardware one. Both are corrected; the artifact still records
    ``attention_path`` so the mode's own record says which boundary it drew.

THE DRAM COST OF THIS PARTITION IS THE RESULT, NOT A REGRESSION
    A host softmax between two device matmuls means the full ``[seq, seq]``
    score matrix crosses DRAM twice per head — at the gate configuration that
    is ~64 MiB per head, ~792 MiB per layer, against ~140 MB for the whole of
    the previous six-GEMM form. This mode is therefore MUCH slower at 4096
    than the implementation it replaces. That is what "all linear on the NPU,
    all non-linear on the host" costs when the non-linear operator sits
    between two matmuls, and pricing it is the point of the mode. Do not read
    the latency as a defect and do not "fix" it by moving softmax to the
    device — that is ``runlist``.

THE RULE THAT DECIDES WHETHER THIS MODE MEASURES ANYTHING
    The mode computes; the oracle checks. They may not share arithmetic. This
    module still does the layer's whole non-linear half on the host, and every
    piece of it is written against torch — ``F.layer_norm``,
    ``F.gelu(approximate="tanh")``, ``torch.softmax`` in
    ``_host_softmax_bf16`` — while the oracle's boundaries come
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
from opcheck_layer import (  # noqa: E402
    BLOCK_STAGE_ATOL,
    print_dispatch_totals,
)
from opcheck_prepare import _spec_digest  # noqa: E402
from pattern import EXECUTION_MODE_CSV  # noqa: E402
from pattern.blocked_attention import round_bf16  # noqa: E402
from pattern.reference import (  # noqa: E402
    ENCODER_BOUNDARIES,
    generate_golden_reference,
)

#: This mode's ELF cache, relative to the working directory. Its OWN — see the
#: module footguns.
#:
#: `[2026-08-09]` The shared-xclbin path gets a SEPARATE directory, for the same
#: reason two modes never share one: ``KernelCache`` keys the directory by name,
#: and these two builds produce artifacts with identical NAMES
#: (``off_gemm_1024x768x768`` either way) over different ABIs. Pointed at one
#: directory they would trade artifacts whose fingerprints happen to agree, and
#: the result is numerically valid output attributed to the wrong execution
#: boundary — which here would mean crediting a 30-reconfiguration run to the
#: one-reconfiguration claim.
OFFLOAD_CACHE_DIR = (
    "offload_shared_cache"
    if os.environ.get("AIR_OFFLOAD_SHARED_XCLBIN", "") == "1"
    else "offload_cache"
)

#: The six projection GEMMs, in dispatch order. The order is the layer's
#: dataflow and each entry becomes one recorded DispatchVector row. The two
#: attention matmuls are dispatched per head and are not in this tuple; see
#: ATTENTION_GEMM_TILES.
OFFLOAD_GEMMS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "output_proj",
    "up_proj",
    "down_proj",
)

# ---------------------------------------------------------------------------
# THE TWO ATTENTION GEMMS: MEASURED TILES, INJECTED
#
# `[2026-08-08]` Neither shape resolves in the registry and neither can be swept
# into it (see the module docstring). These tiles are not guesses and not
# defaults -- each is the configuration that PASSED on real NPU2 at 0% allowed
# mismatch over the full output, against the same `XRTRunner._check_outputs`
# comparison every registry row was sized with:
#
#   attn_scores  4096x64x4096   drain  tm=32 tk2=64  tk1=32 tn=128  herd 8x4
#                               0 / 16,777,216 mismatches, mean_rel_L1 9.386e-3,
#                               2901.4 us, 740.1 GFLOP/s.
#                               tk2=64 is FORCED: K=64 admits no other L2 tile.
#
#   attn_output  4096x4096x64   drain  tm=32 tk2=256 tk1=32 tn=16   herd 8x4
#                               0 / 262,144 mismatches, mean_rel_L1 9.417e-3,
#                               abs_err_max 7.324e-4 against atol 2.121e-3
#                               (3.46x margin), 1179.3 us, 1820.9 GFLOP/s.
#
# `attn_output` also passes by `direct` (915.8 us, faster) and by `fused-cast`.
# `drain` is chosen for ACCURACY, not speed: direct measures mean_rel_L1
# 1.542e-2 against drain's 9.417e-3, and this layer's tolerance has no headroom
# to spend -- `output` sits at the hard 1e-1 ceiling at a 1.35x margin.
#
# Two failure clusters bound the space, both fully characterised, so a future
# reader does not re-search it: herd_n=1 at N=64 HANGS (returns the
# host-written buffer at a flat 6,144,000 us per iteration), and tile_n=8 fails
# the microkernel's own `static_assert(n % (2 * t) == 0)` for drain/fused-cast
# while `direct` builds and returns garbage.
# ---------------------------------------------------------------------------
ATTENTION_GEMM_TILES = {
    "attn_scores": {
        "method": "drain",
        "tile": {"tile_m": 32, "tile_k_l2": 64, "tile_k_l1": 32, "tile_n": 128},
        "herd": (8, 4),
    },
    "attn_output": {
        "method": "drain",
        "tile": {"tile_m": 32, "tile_k_l2": 256, "tile_k_l1": 32, "tile_n": 16},
        "herd": (8, 4),
    },
}


def attention_gemm_spec(role):
    """The injected build recipe for one attention matmul.

    Built through ``llms/shared``'s own ``_spec_with_tiles`` rather than as a
    hand-written dict, so an injected spec carries exactly the fields a
    registry-resolved one does — ``sym_suffix`` and ``obj`` derived from
    ``(tile_m, tile_n)``, ``needs_f32_scratch``, ``build_kwargs`` — and cannot
    drift from what ``_build_gemm_module`` and ``ek.compile_gemm_mm`` expect.
    The herd is set explicitly because an injected spec carries none and
    ``spec_herd`` would otherwise fall back to 8x4 by coincidence rather than
    by measurement (``builders/gemm_spec.py`` footguns).
    """
    from shared.builders.gemm_builder import _spec_with_tiles

    entry = ATTENTION_GEMM_TILES[role]
    spec = dict(_spec_with_tiles(entry["method"], entry["tile"]))
    spec["herd_m"], spec["herd_n"] = entry["herd"]
    return spec

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

#: Recorded in the artifact: both attention matmuls are device dispatches and
#: only the softmax between them is host torch. `[2026-08-08]` This value was
#: ``host_torch_fp32_blocked`` until the taxonomy correction; a results tree
#: mixing the two is mixing two different modes, and the field is what says so.
#: The sequence ladder's slopes split on exactly this covariate.
ATTENTION_PATH = "device_gemm_host_softmax"

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

#: `[2026-08-09]` The same settings on the **xclbin** ABI, which is what N
#: instruction streams under one xclbin requires. `[2,2]` is kept: the tiling is
#: the mode's validated one, and context reuse -- the thing N streams turns on --
#: is clean at `[2,2]` under this ABI (it is `elf` + `[2,2]` that corrupts, and
#: only that; `agents/probes/probe_context_reuse.py`).
_GEMM_BACKEND_SHARED = dict(_GEMM_BACKEND, output_format="xclbin")

#: Base DPU kernel id for the shared xclbin. Each stream gets `_KERNEL_ID_BASE +
#: i`, because the merged `AIE_PARTITION` routes a kernel to its PDI -- its array
#: configuration -- by `dpu_kernel_ids`, and EVERY AIR compile defaults to 0x901.
#: Two PDIs both claiming 0x901 are indistinguishable to the runtime and the
#: second kernel runs against the first's configuration: measured as
#: `ERT_CMD_STATE_TIMEOUT` at one shape and garbage at `mean_rel_L1` 1.41 with no
#: error raised at another (`agents/probes/probe_one_xclbin_n_streams.py`).
_KERNEL_ID_BASE = 0x901

#: Whether the mode packages its GEMMs into ONE xclbin. Off by default: the
#: shared path is newer than the gated ELF one, and this keeps the mode's
#: recorded provenance switchable in one place rather than by editing kwargs.
#: `AIR_OFFLOAD_SHARED_XCLBIN=1` turns it on.
SHARED_XCLBIN = os.environ.get("AIR_OFFLOAD_SHARED_XCLBIN", "") == "1"


def _check_no_object_collision(specs):
    """Raise if two GEMMs share an ``mm_*.o`` name without sharing its contents.

    ``ek.compile_gemm_mm`` names the object from ``(tile_m, tile_n)`` alone
    while ``tile_k_l1`` is baked in as a compile flag, so two GEMMs that agree
    on the first two and differ on the third write the SAME file with DIFFERENT
    micro-kernels — and whichever compiles last wins. D2 found exactly this
    between the FFN's up-projection and the o-projection; it does not fail to
    build, it silently returns the other GEMM's arithmetic.

    Today ``up`` and ``attn_scores`` legitimately share ``mm_m32n128.o``: both
    are drain at tile_m 32 / tile_n 128 / tile_k_l1 32, so the object really is
    the same one and compiling it twice is a no-op. That is a coincidence of
    the current tiles, not a property. Retuning either shape's ``tile_k_l1``
    would reintroduce the collision, and this turns that into an immediate
    raise naming both GEMMs instead of a wrong number 4096 rows later.
    """
    seen = {}
    for key, (spec, _) in specs.items():
        signature = (spec["tile_m"], spec["tile_n"], spec["tile_k_l1"])
        previous_key, previous_signature = seen.get(spec["obj"], (None, None))
        if previous_key is not None and previous_signature != signature:
            raise ValueError(
                f"GEMM object name collision: '{key}' and '{previous_key}' both "
                f"compile to {spec['obj']} but from different micro-kernels "
                f"({signature} vs {previous_signature}). compile_gemm_mm names "
                f"the object from (tile_m, tile_n) only, so one would silently "
                f"overwrite the other. Give one of them a different tile_n, or "
                f"fix the naming in llms/shared/builders/gemm_builder.py."
            )
        seen[spec["obj"]] = (key, signature)


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
        # Per HEAD, not per layer: Q_h @ K_h^T is [seq, head_dim] @ [head_dim,
        # seq], and P_h @ V_h is [seq, seq] @ [seq, head_dim]. One compiled
        # module each, dispatched num_heads times.
        "attn_scores": (
            attention_gemm_spec("attn_scores"),
            (seq_len, head_dim, seq_len),
        ),
        "attn_output": (
            attention_gemm_spec("attn_output"),
            (seq_len, seq_len, head_dim),
        ),
    }
    _check_no_object_collision(specs)
    artifacts = {key: f"off_gemm_{m}x{k}x{n}" for key, (_, (m, k, n)) in specs.items()}
    if SHARED_XCLBIN:
        # One xclbin over every shape, so the array is configured ONCE. Both
        # identifiers must be distinct per stream and neither defaults usefully:
        # `instance_name` is matched by SUBSTRING when the kernel is looked up,
        # and `kernel_id` selects the PDI. Note the shipped ELF path gives every
        # `drain` GEMM the same `instance_name` (`matmul_bf16`) -- harmless when
        # each carries its own xclbin, fatal when they share one.
        backend_kwargs = {
            artifacts[key]: dict(
                _GEMM_BACKEND_SHARED,
                # THREE identifiers, all three required and none defaulting
                # usefully. `kernel_name` is the EMBEDDED_METADATA entry and
                # xclbinutil REFUSES a duplicate ("Kernel name already exists in
                # the EMBEDDED_METADATA section: 'MLIR_AIE'") -- the one failure
                # of the three that is loud. `instance_name` is what the kernel
                # lookup substring-matches. `kernel_id` selects the PDI.
                kernel_name=f"{_METHOD_FUNC[spec['method']]}_{key}",
                instance_name=f"{_METHOD_FUNC[spec['method']]}_{key}",
                kernel_id=hex(_KERNEL_ID_BASE + i),
            )
            for i, (key, (spec, _)) in enumerate(specs.items())
        }
    else:
        backend_kwargs = {
            artifacts[key]: dict(
                _GEMM_BACKEND, instance_name=_METHOD_FUNC[spec["method"]]
            )
            for key, (spec, _) in specs.items()
        }
    gemms = {
        "q_proj": "proj",
        "k_proj": "proj",
        "v_proj": "proj",
        "output_proj": "proj",
        "up_proj": "up",
        "down_proj": "down",
        "attn_scores": "attn_scores",
        "attn_output": "attn_output",
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
        # 6 projections + 2 attention matmuls per head. Derived here rather
        # than written as a literal so the gate's count and the run's count
        # cannot disagree at a shape with a different head count.
        "n_dispatches": len(OFFLOAD_GEMMS) + 2 * num_heads,
        # Resolved into the config rather than read from the environment at each
        # use, so one run cannot be half shared and half not.
        "shared_xclbin": SHARED_XCLBIN,
    }


def describe_offload(cfg):
    """One line per resolved decision, for the run log and the lit gate."""
    print(
        f"  offload {cfg['seq_len']}x{cfg['emb_dim']} ffn {cfg['ffn_dim']} "
        f"{cfg['num_heads']}h x {cfg['head_dim']} (encoder_bert, non-causal, "
        f"linear boundary: 8 linear operators over 5 shapes, "
        f"{cfg['n_dispatches']} dispatches, attention on device)"
    )
    parts = []
    for key in ("proj", "up", "down"):
        spec, (m, k, n) = cfg["specs"][key]
        parts.append(f"{key} {m}x{k}x{n} {spec['method']} (registry)")
    print("    " + ", ".join(parts))
    # The injected tiles, printed in full: this is the only place a reader sees
    # that these two shapes came from a measurement rather than the registry,
    # and the gate pins the line so a silent retune fails there.
    parts = []
    for key in ("attn_scores", "attn_output"):
        spec, (m, k, n) = cfg["specs"][key]
        parts.append(
            f"{key} {m}x{k}x{n} {spec['method']} tk2={spec['tile_k_l2']} "
            f"tk1={spec['tile_k_l1']} tn={spec['tile_n']} "
            f"herd {spec['herd_m']}x{spec['herd_n']} (injected)"
        )
    print("    " + ", ".join(parts))
    print(
        f"    attention on device: 2 GEMMs x {cfg['num_heads']} heads = "
        f"{2 * cfg['num_heads']} dispatches, host keeps scale + softmax"
    )
    # THE MODE'S OWN AXIS, printed so a gate can pin it. Reconfiguration is not
    # one of the six dispatch-vector fields and cannot be: the vector is
    # identical between these two paths, because the mode makes one run_sequence
    # call per GEMM either way. What changes is how many times the array is
    # configured to serve them.
    if cfg["shared_xclbin"]:
        print(
            f"    reconfiguration: ONE xclbin over {len(cfg['artifacts'])} shapes, "
            f"{cfg['n_dispatches']} dispatches share 1 hw_context "
            f"(N instruction streams, array configured once)"
        )
    else:
        print(
            f"    reconfiguration: {len(cfg['artifacts'])} xclbins, "
            f"{cfg['n_dispatches']} hw_context loads for {cfg['n_dispatches']} "
            f"dispatches (one per dispatch; see _evict_context)"
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

    if SHARED_XCLBIN and stale:
        # A chain cannot be built piecemeal: only the LAST xclbin holds every
        # kernel, so a partial rebuild would leave the reused artifacts pointing
        # at a file without theirs. Rebuild the whole chain whenever any member
        # is stale -- which is also why `reused` is recomputed as empty here.
        stale, reused = list(names), []
        modules = {key: _build_offload_module(cfg, key) for key in stale}

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

    if SHARED_XCLBIN:
        if stale:
            print(f"== compiling {len(stale)} kernels into ONE xclbin ==")
            cache.compile_shared_xclbin(
                [(names[key], modules[key], cfg["backend_kwargs"][names[key]])
                 for key in stale]
            )
    else:
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
    shared implementation and never hand-built here.

    **Whether this dispatch reconfigures the array is the mode's whole point.**
    On the shipped ELF path each dispatch runs in a FRESH ``hw_context`` — see
    ``_evict_context`` for the measurement that forces it — so the layer pays 30
    reconfigurations, which is the MAXIMUM rather than the minimum the mode is
    defined to isolate. Under ``SHARED_XCLBIN`` every GEMM lives in one xclbin,
    the context is loaded once and never evicted, and moving between shapes costs
    an instruction swap. The dispatch VECTOR is identical either way — this mode
    makes one ``run_sequence`` call per GEMM regardless, so submissions stay at
    30 — which is what makes the existing gate a correctness check on the change
    rather than a measurement of it. The reconfiguration count is recorded
    separately by ``describe_offload``.
    """
    key = cfg["gemms"][op_name]
    spec, (m, k, n) = cfg["specs"][key]
    artifact = cfg["artifacts"][key]
    if not SHARED_XCLBIN:
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


def _host_softmax_bf16(scores, scale):
    """Scale, then row-softmax, in f32 on the host; rounded to bf16 for the device.

    The one NON-LINEAR operator inside attention, and therefore the whole
    reason the ``[seq, seq]`` score matrix crosses DRAM twice per head. torch
    rather than numpy, per the oracle independence rule in the module
    docstring: the CHECK path's softmax is
    ``builders/mha_attention.py::chunked_attention_reference``, in numpy.

    The ``1/sqrt(head_dim)`` scale is folded in here rather than into Q
    upstream. Either is correct and neither is free of rounding, but doing it
    on the already-materialized f32 scores keeps the device GEMM's operands
    exactly the projections the previous dispatch produced, so a stage
    comparison on ``q`` is comparing the tensor the next GEMM actually read.
    """
    import torch

    probs = torch.softmax(_torch_f32(scores) * scale, dim=-1)
    return round_bf16(probs.numpy())


def _device_attention(cache, cfg, gemm, q, k, v):
    """Non-causal multi-head attention with BOTH matmuls on the device.

    One head at a time, because the two shapes are per-head: ``[seq, head_dim]
    @ [head_dim, seq]`` and ``[seq, seq] @ [seq, head_dim]``. Two dispatches
    per head, each its own submission, which is the mode being itself — see
    the ``run_sequence`` footgun on why they are not batched.

    ``q``/``k``/``v`` are ``[seq, num_heads * head_dim]`` bf16, head ``h`` in
    columns ``h*head_dim : (h+1)*head_dim`` — the layout the projection GEMMs
    produce and the one the golden reference indexes. Returns the context in
    that same layout as float32.

    The head slice and the K transpose are host LAYOUT, not host arithmetic:
    the partition this mode implements is linear-on-device / non-linear-on-host,
    and a transpose is neither. It is a real host cost and it is timed as one.
    """
    import math

    seq_len, num_heads = cfg["seq_len"], cfg["num_heads"]
    head_dim = cfg["head_dim"]
    scale = 1.0 / math.sqrt(head_dim)

    context = np.empty((seq_len, num_heads * head_dim), dtype=bfloat16)
    for head in range(num_heads):
        columns = slice(head * head_dim, (head + 1) * head_dim)
        # Contiguous copies: the device operands are plain row-major memrefs,
        # and a numpy view with a column stride is not one.
        with cache.profiler.time_cpu("attention_layout"):
            q_head = np.ascontiguousarray(q[:, columns])
            k_head_t = np.ascontiguousarray(k[:, columns].T)
            v_head = np.ascontiguousarray(v[:, columns])

        scores = gemm("attn_scores", q_head, k_head_t)
        with cache.profiler.time_cpu("softmax"):
            probs = _host_softmax_bf16(scores, scale)
        context[:, columns] = gemm("attn_output", probs, v_head)

    return context.astype(np.float32)


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
        cache_dir=OFFLOAD_CACHE_DIR, verbose=False, profiler=Profiler(enabled=True)
    )
    compile_offload_artifacts(cache, cfg, run_only=True)

    def dispatch(device_inputs, stage_stats):
        cache.profiler.cpu_times.clear()
        x, w_q, w_k, w_v, w_o, ln1_weight, w_up, w_down, ln2_weight = device_inputs

        vector_rows = []

        def gemm(op_name, a, b):
            out, row = _dispatch_gemm(cache, cfg, op_name, a, b)
            vector_rows.append(row)
            return out

        print("  [1/8] q_proj + [2/8] k_proj + [3/8] v_proj (one dispatch each)")
        q = gemm("q_proj", x, w_q)
        k = gemm("k_proj", x, w_k)
        v = gemm("v_proj", x, w_v)
        print(
            f"  [4/8+5/8] attn_scores + attn_output on device, "
            f"{num_heads} heads, host softmax between them"
        )
        # Softmax and layout are timed SEPARATELY, accumulating per key across
        # the per-head calls: softmax is the non-linear operator this mode
        # deliberately leaves on the host, the head slicing and K transpose are
        # data movement the partition says nothing about. Summing them into one
        # "attention" figure would report the partition's cost as larger than
        # it is.
        attn_context = _device_attention(cache, cfg, gemm, q, k, v)
        print("  [6/8] output_proj")
        attn_out = gemm("output_proj", round_bf16(attn_context), w_o)
        with cache.profiler.time_cpu("ln1"):
            hidden = _host_addnorm(attn_out, x, ln1_weight)
        print("  [7/8] up_proj")
        ffn_up = gemm("up_proj", hidden, w_up)
        with cache.profiler.time_cpu("gelu"):
            ffn_gelu = _host_gelu(ffn_up)
        print("  [8/8] down_proj")
        ffn_out = gemm("down_proj", round_bf16(ffn_gelu), w_down)
        with cache.profiler.time_cpu("ln2"):
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
        # two printed lines against the SAME literals as the clean half, so
        # instrumentation conditional on the injected flag, a malformed row
        # (the shared validation raises, failing the run) or drifted totals
        # fail in the suite before the driver's comparison sees them.
        print_dispatch_totals("offload", vector_rows)
        # THE MODE'S OWN AXIS, MEASURED. The six-field vector above cannot carry
        # it -- it is identical on both paths, because the mode makes one
        # `run_sequence` call per GEMM whether the array is configured once or
        # thirty times. This counts the configurations themselves, so the claim
        # is read off a counter rather than asserted from the build settings.
        loads, attaches = cache.reconfiguration_counts()
        print(
            f"[offload] reconfiguration: context_loads {loads} "
            f"kernel_attaches {attaches} over {cfg['n_dispatches']} dispatches"
        )
        return [boundaries["output"]], {
            "context_loads": loads,
            "kernel_attaches": attaches,
            # The two schema v1 columns that separate this mode's two packaging
            # paths in a results row. Without them a shared-path CSV is
            # byte-identical to an ELF-path one, and a walk comparing the two
            # would be comparing two files nothing distinguishes.
            #
            # COUNTED off the artifacts this run actually loaded, not inferred
            # from `shared_xclbin`, and counted as what the column literally
            # says -- xclbins. The ELF path loads `.elf` artifacts through
            # `xrt.elf()` and loads NO xclbin at all, so its honest value here
            # is **zero**, not five: five is the number of distinct BINARIES,
            # which is a different quantity and not what this column means.
            # 0 against 1 still separates the two paths, and both values are
            # true.
            #
            # This column is NOT the reconfiguration count. That is 30 against
            # 1, it is the mode's actual axis, and schema v1 has nowhere to put
            # it -- adding one is a version bump. `reconfiguration_counts()`
            # and the gated `reconfiguration:` line carry it meanwhile.
            "unique_xclbins": sum(
                1
                for binary in {
                    cache.artifacts[name].output_binary
                    for name in cfg["artifacts"].values()
                    if name in cache.artifacts
                }
                if str(binary).endswith(".xclbin")
            ),
            # Per LAYER, which is the unit every other count in this row uses.
            "n_dispatches": cfg["n_dispatches"],
            "stages": stages,
            "stages_passed": clean == len(stages),
            "dispatch_vectors": vector_rows,
            # `[2026-08-08]` The latency decomposition convention 10 asks for.
            # device_ms and sync_ms were already measured by dispatch.py and
            # thrown away; host_cpu_ms is this mode's own torch work, timed
            # through Profiler.time_cpu. Together they say how much of a mode's
            # latency is the NPU and how much is the host running attention --
            # the confound every cross-mode comparison in this study carries.
            "device_ms": sum(
                float(r.get("device_submission_ms", 0.0)) for r in vector_rows
            ),
            "sync_ms": sum(float(r.get("host_sync_ms", 0.0)) for r in vector_rows),
            "host_cpu_ms": {
                k: sum(v) * 1000.0 for k, v in cache.profiler.cpu_times.items()
            },
        }

    record_extra = {
        "variant": "encoder_bert",
        "causal": False,
        "golden_seed": seed,
        "execution_mode": EXECUTION_MODE_CSV["offload"],
        "attention_path": ATTENTION_PATH,
        "n_dispatches": cfg["n_dispatches"],
        # MIXED, and the field says so rather than rounding to one word: the
        # three projection shapes resolve in the registry, the two attention
        # shapes are injected measured tiles that resolve in none. The per-spec
        # digests below are what a reader checks that against; the phase allows
        # injection precisely on the condition that it is not silent.
        "gemm_spec_source": "registry+injected",
        "gemm_spec_proj": _spec_digest(cfg["specs"]["proj"][0]),
        "gemm_spec_ffn_up": _spec_digest(cfg["specs"]["up"][0]),
        "gemm_spec_ffn_down": _spec_digest(cfg["specs"]["down"][0]),
        "gemm_spec_attn_scores": _spec_digest(cfg["specs"]["attn_scores"][0]),
        "gemm_spec_attn_output": _spec_digest(cfg["specs"]["attn_output"][0]),
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
