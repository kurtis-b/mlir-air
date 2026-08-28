# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``runlist`` — every operator its own kernel, on the device, nothing on the host.

CONTRACT
    ``prepare_runlist(shape, seed=...)`` is this mode's entry in the ``SPECS``
    catalogue: the D2 layer prepared for ``opcheck.py``'s ``dispatch`` seam.
    Every operator in the encoder layer is dispatched individually, on the
    device, each ``KernelCache.run_sequence`` call forced to one runlist with
    ``require_single_submission=True``::

        coarse's unit                        this mode's refinement
        ------------------------------       --------------------------------
        qkv_proj (fused)              ->     runlist 1: q_proj  k_proj  v_proj
        mha_out_proj (fused)          ->     runlists 2..13, ONE PER HEAD:
                                               attn_scores  softmax  attn_output
                                             then runlist 14: output_proj
        64 x addnorm ln1 (pre-add)    ->     runlist 15: 64 x (add  ln  mul)
        ffn (fused up+gelu+down)      ->     runlist 16: up_proj gelu down_proj
        64 x addnorm ln2 (pre-add)    ->     runlist 17: 64 x (add  ln  mul)

    17 recorded ``DispatchVector`` rows; the driver-summed totals are 17
    submissions over ``7 + 3 * num_heads + 6 * norm_blocks`` runlist entries —
    427 at the gate configuration, against ``coarse``'s 131. Every coarse
    dispatch unit maps onto one or more finer units, so
    ``runlist_entries > coarse`` holds BY CONSTRUCTION, which is what the
    mode's ordinal claim ("the fine-grained point of the taxonomy") means.
    Intermediates inside a runlist stay DEVICE-RESIDENT: q/k/v never touch the
    host before attention reads them, each head's score and probability
    matrices never leave the array, each band's residual sum feeds its
    LayerNorm and gamma multiply on device, and ``ffn_up``/``ffn_gelu`` chain
    into the down projection.

`[2026-08-09]` WHAT THE REBUILD CHANGED, AND THE ONE NUMBER TO READ
    Attention used to run in host torch through ``blocked_attention``, which
    made this mode price host torch rather than reconfiguration — 24.15 ms at
    1024, 47.8% of its total. Both matmuls are linear and now dispatch with
    the tiles measured in ``pattern/offload/offload.py`` (imported, not
    copied); the softmax between them is ``builders/softmax.py``.

    The consequence worth reading is **bytes**: 190,513,152 here against
    ``offload``'s 970,457,088 for the same layer, a 5.1x difference produced
    entirely by where the softmax runs. ``offload`` keeps it on the host, so
    every head's ``[seq, seq]`` score matrix crosses DRAM twice; here it stays
    resident inside the head's runlist and only ``q_h``/``k_h_t``/``v_h`` in
    and ``ctx_h`` out cross. That is the reconfiguration-against-DRAM-traffic
    axis the corrected taxonomy is about, measured on two modes that differ in
    exactly that.

ONE SUBMISSION PER HEAD, AND WHY IT IS NOT A SCHEDULE CHOICE
    Twelve heads in one runlist would need every score and probability matrix
    live simultaneously — ~800 MiB at the gate configuration against ~70 MiB
    per head. It is a memory bound. It does not touch the ENTRY count, which
    is what the mode's granularity claim is about, and
    ``runlist_submission_count`` derives it rather than hardcoding 17.

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
    - The two attention GEMM ELFs get ONE artifact each rather than one per
      head, and that is consistent with the no-re-execution rule above rather
      than an exception to it. The rule is about re-executing a runtime-tiled
      GEMM ELF in a REUSED ``hw_context``; each per-head submission dispatches
      each of them exactly once, and ``evict_attention_contexts`` drops their
      contexts between heads, which is available here precisely because the
      boundary is BETWEEN submissions. The softmax artifact is not evicted:
      no runtime loop tiling, the same re-execution-clean class as the band
      ``add``/``ln``/``mul`` ELFs.
    - The mode computes; the oracle checks. Every operator is its own device
      kernel and nothing runs on the host; the per-boundary references come
      from the numpy oracles behind ``pattern/reference.py``. Nothing here may
      import ``addnorm_pre_add_reference``, ``gelu_tanh_reference`` or any
      other function that computes a boundary. ``round_bf16`` is imported from
      ``pattern/blocked_attention.py`` and is a plain rounding helper — the
      module name is historical, and importing that SYMBOL does not make this
      a host-attention mode (``study/test_attention_path.py`` checks the symbol
      and not the module for exactly this reason).
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
from shared.infra.bo_pool import BufferSpec, DispatchStep, content_key_once  # noqa: E402

from builders.block import norm_rows  # noqa: E402
from builders.block_cache import (  # noqa: E402
    block_artifact_fingerprint,
    load_fingerprints,
    save_fingerprints,
)
from builders.elementwise_add import (  # noqa: E402
    build_elementwise_add_module,
    causal_mask_bias,
)
from builders.elementwise_mul import (  # noqa: E402
    broadcast_row_weight,
    build_elementwise_mul_module,
)
from builders.gelu import build_gelu_module  # noqa: E402
from builders.gemm_spec import resolve_gemm_spec, spec_herd  # noqa: E402
from builders.layer_norm import build_layer_norm_module  # noqa: E402
from builders.softmax import build_softmax_module  # noqa: E402
from builders.softmax import derive_rows_per_call as derive_softmax_rows_per_call  # noqa: E402
from opcheck_layer import (  # noqa: E402
    BLOCK_STAGE_ATOL,
    decoder_stage_atol,
    fault_delta_hook,
    print_dispatch_totals,
    reconfiguration_delta,
)
from opcheck_prepare import _spec_digest  # noqa: E402
from pattern import EXECUTION_MODE_CSV  # noqa: E402
from pattern.blocked_attention import round_bf16  # noqa: E402

# The measured attention tiles live in ONE place, offload's module, and are
# imported rather than copied. They are a measurement (see that module's
# ATTENTION_GEMM_TILES comment for the checkpoints), so two modes carrying two
# copies is two things to keep in step and one of them silently going stale.
from pattern.offload.offload import (  # noqa: E402
    _check_no_object_collision,
    attention_gemm_spec,
)
from pattern.reference import (  # noqa: E402
    DECODER_BOUNDARIES,
    ENCODER_BOUNDARIES,
    generate_golden_reference,
    layer_inputs,
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

#: Recorded in the artifact: `[2026-08-09]` the whole attention interior is on
#: the device — both matmuls AND the softmax between them — so this mode now
#: runs NOTHING on the host, which is what the corrected taxonomy defines it as.
#: It was ``host_torch_fp32_blocked`` until this rebuild; a results tree mixing
#: the two values is mixing two different modes.
ATTENTION_PATH = "device_all"

#: instance_name for the softmax artifact — the func.func ``builders/softmax.py``
#: emits. The small-operator backend applies: no runtime loop tiling, so unlike
#: the GEMM ELFs it is re-execution clean and one artifact serves every head.
_SOFTMAX_FUNC = "softmax_rows"

#: Rows of the score matrix resident in L1 per softmax kernel call. Three
#: ``[rows_per_call, cols]`` bf16 buffers live there and cols IS the sequence
#: length for a score matrix, so this is bounded by the row width: at 4096 it
#: needs 48 KiB of a 64 KiB L1 and 4 would need 96 KiB. The row loop is inside
#: the herd body, so this bounds L1 and NOT the entry count — one softmax is
#: one runlist entry however many trips it walks.
#:
#: `[2026-08-20]` This is now a CEILING, not the value: the first ``full``
#: profile (devq 427) took `runlist` to 8192, where 2 rows are 96 KiB, and
#: aircc failed with an empty message after 88 s. ``builders.softmax.
#: derive_rows_per_call`` picks the largest legal value at or below this, so
#: every length where 2 fits (<= 4096) emits byte-identical IR, 8192 gets 1,
#: and 16384 -- where even one 16384-wide row is 96 KiB -- refuses by name at
#: prepare time instead of 468 s into a compile.
SOFTMAX_ROWS_PER_CALL = 2

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
    # `[2026-08-09]` softmax belongs in this family and not with the GEMMs: it
    # takes no runtime loop tiling, which is what makes an ELF re-execution
    # clean, so ONE artifact serves all twelve heads. Being here also gets it
    # the emitted-symbol assertion in compile_runlist_artifacts.
    "softmax": "softmax_rows",
    # Decoder only (`runlist_config(causal=True)` adds the artifact): the
    # [seq, seq] mask-tensor add between the score GEMM and the softmax --
    # the same builder as `add`, at score shape, with the validated
    # `causal_mask` op's semantics (builders/elementwise_add.py).
    "causal_mask": "eltwise_add_2d",
}


def runlist_config(seq_len, emb_dim, ffn_dim, num_heads, head_dim, causal=False):
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
    # `[2026-08-09]` The attention interior, per head. Both matmuls are LINEAR
    # and belong on the device under the corrected taxonomy; the tiles are
    # imported from offload, where the measurement lives.
    specs["attn_scores"] = (
        attention_gemm_spec("attn_scores"),
        (seq_len, head_dim, seq_len),
    )
    specs["attn_output"] = (
        attention_gemm_spec("attn_output"),
        (seq_len, seq_len, head_dim),
    )

    # GEMM entry -> spec key. Four distinct proj ELFs on purpose; see above.
    #
    # The two attention entries get ONE artifact each, not one per head, and
    # that is consistent with the no-re-execution rule rather than an exception
    # to it: the rule is about re-executing a runtime-tiled GEMM ELF in a REUSED
    # hw_context. Each per-head submission dispatches each of them exactly once,
    # and their contexts are evicted between heads exactly as `offload` evicts
    # between its dispatches (`evict_attention_contexts`). Twelve artifacts
    # apiece would also work and would cost 24 large ELF compiles per clean
    # cache for nothing.
    gemms = {
        "q_proj": "proj",
        "k_proj": "proj",
        "v_proj": "proj",
        "o_proj": "proj",
        "up": "up",
        "down": "down",
        "attn_scores": "attn_scores",
        "attn_output": "attn_output",
    }
    # Same guard offload carries, for the same reason: compile_gemm_mm names
    # its object from (tile_m, tile_n) alone while tile_k_l1 is a compile flag,
    # so two GEMMs agreeing on the first two and differing on the third write
    # one file with two micro-kernels. `up` and `attn_scores` legitimately
    # share mm_m32n128.o today because all three agree; retuning either would
    # not.
    _check_no_object_collision(specs)
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
            "softmax": f"rl_softmax_{seq_len}x{seq_len}",
        }
    )
    if causal:
        # The decoder-only keys are added CONDITIONALLY, same argument as
        # builders/block.py: the fingerprint hashes the whole cfg, and a new
        # key on the encoder path would invalidate every cached encoder ELF
        # for a config that resolved identically.
        artifacts["causal_mask"] = f"rl_causal_mask_{seq_len}x{seq_len}"
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
            if key in artifacts
        }
    )
    cfg = {
        "seq_len": seq_len,
        "emb_dim": emb_dim,
        "ffn_dim": ffn_dim,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "norm_rows": rows,
        "norm_blocks": seq_len // rows,
        # Derived once here -- the artifact builder and the record both read
        # it -- so a width where nothing fits refuses at CONFIG time, by name.
        "softmax_rows_per_call": derive_softmax_rows_per_call(
            seq_len, seq_len, bfloat16, ceiling=SOFTMAX_ROWS_PER_CALL
        ),
        "specs": specs,
        "gemms": gemms,
        "artifacts": artifacts,
        "backend_kwargs": backend_kwargs,
    }
    if causal:
        cfg["causal"] = True
    return cfg


def runlist_entry_count(cfg):
    """Total runlist entries the decomposition dispatches, derived not counted:
    q/k/v (3) + per head (attn_scores, softmax, attn_output) + output_proj (1)
    + up/gelu/down (3) + two norm chains of ``3 * norm_blocks`` band entries
    each. The DECODER adds one mask-add entry per head and two bare
    residual-add runs of ``norm_blocks`` band entries each."""
    base = 7 + 3 * cfg["num_heads"] + 6 * cfg["norm_blocks"]
    if cfg.get("causal"):
        base += cfg["num_heads"] + 2 * cfg["norm_blocks"]
    return base


def runlist_submission_count(cfg):
    """Submissions: qkv, one PER HEAD, o_proj, ln1, ffn, ln2.

    The per-head split is a MEMORY bound, not a taste. One submission holding
    every head's attention would need all of its buffers live at once: at the
    gate configuration that is 12 score matrices and 12 probability matrices of
    [4096, 4096] bf16, ~800 MiB before counting anything else. Per head it is
    ~70 MiB. The mode's granularity claim is about ENTRIES, which the split
    does not change.

    The DECODER's two bare residual-add runs are two more submissions (the
    mask add rides inside each head's existing one).
    """
    return (7 if cfg.get("causal") else 5) + cfg["num_heads"]


def describe_runlist(cfg):
    """One line per resolved decision, for the run log and the lit gate."""
    variant = (
        "decoder_gpt2, causal" if cfg.get("causal") else "encoder_bert, non-causal"
    )
    print(
        f"  runlist {cfg['seq_len']}x{cfg['emb_dim']} ffn {cfg['ffn_dim']} "
        f"{cfg['num_heads']}h x {cfg['head_dim']} ({variant}, "
        f"{runlist_entry_count(cfg)} fine-grained entries over "
        f"{runlist_submission_count(cfg)} runlists, nothing on the host)"
    )
    parts = []
    for key in ("proj", "up", "down"):
        spec, (m, k, n) = cfg["specs"][key]
        parts.append(f"{key} {m}x{k}x{n} {spec['method']} (registry)")
    print("    " + ", ".join(parts))
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
        f"    attention on device: {cfg['num_heads']} x "
        f"(attn_scores + softmax + attn_output), "
        f"softmax {cfg['seq_len']}x{cfg['seq_len']} rows_per_call "
        f"{SOFTMAX_ROWS_PER_CALL}"
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
    if key == "causal_mask":
        # Score shape, whole [seq, seq] in one launch; the causal_mask keyword
        # asserts the square shape and records that `b` is the static mask.
        return build_elementwise_add_module(
            cfg["seq_len"], cfg["seq_len"], bfloat16, causal_mask=True
        )
    if key == "ln":
        return build_layer_norm_module(rows, emb_dim, bfloat16)
    if key == "mul":
        return build_elementwise_mul_module(rows, emb_dim, bfloat16)
    if key == "gelu":
        return build_gelu_module(cfg["seq_len"], ffn_dim, bfloat16)
    if key == "softmax":
        # Square: a score matrix is [seq, seq], so the row width IS the
        # sequence length and rows_per_call is bounded by it, not by the
        # hidden size every other row-wise operator here works at.
        return build_softmax_module(
            cfg["seq_len"],
            cfg["seq_len"],
            bfloat16,
            rows_per_call=cfg["softmax_rows_per_call"],
        )
    raise KeyError(f"unknown runlist artifact key {key!r}")


def compile_runlist_artifacts(cache, cfg, run_only=False, keys=None):
    """Compile the ten ELFs into ``cache``, reusing only exact matches.

    Same shape as ``compile_block_artifacts`` and reusing its fingerprint
    machinery verbatim: every module is built and hashed on every call, a
    cached ELF is reused only when its recorded fingerprint matches, and only
    a miss rebuilds — including that artifact's external objects, whose names
    carry their tile shapes so variants cannot overwrite each other.

    ``keys`` narrows the build to a SUBSET, and defaults to all of them — this
    mode dispatches every one. A caller composing this half with another mode's
    (``pattern/coarse/cells.py``) dispatches only some, and the rest would be
    large ELFs nothing runs.
    """
    names = cfg["artifacts"]
    have_manifest = bool(run_only and cache.load_manifest())
    recorded = load_fingerprints(cache) if have_manifest else {}

    fingerprints = {}
    reused = []
    modules = {}
    stale = []
    for key in names:
        if keys is not None and key not in keys:
            continue
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
        elif key == "softmax":
            # The STREAMING family behind -DSOFTMAX_STREAMING, not the
            # single-shot softmax_bf16 in the same file; see builders/softmax.py
            # for why. vec_len must match the builder's SM_VEC_LEN.
            ek.compile_softmax_streaming(vec_len=64)
        # "add", "mul" and "causal_mask" are direct vector codegen: nothing
        # to compile.
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
        content_key=content_key_once(array) if static else None,
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


def run_projections(cache, cfg, x, w_q, w_k, w_v):
    """Runlist 1: q/k/v projections, three entries in one submission."""
    seq_len, emb_dim = cfg["seq_len"], cfg["emb_dim"]
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


def evict_attention_contexts(cache, cfg):
    """Drop the two attention GEMM artifacts' ``hw_context``s (and the pools).

    MEASURED, NOT DEFENSIVE, and it is the same measurement this module's
    no-re-execution footgun records: a runtime-tiled GEMM ELF returns wrong
    numbers from its SECOND execution in one context onward. The per-head
    submissions dispatch each attention GEMM once, so nothing re-executes
    INSIDE a runlist -- what has to be broken is reuse ACROSS the twelve
    submissions, which is exactly what ``offload._evict_context`` does
    between its dispatches. Evicting is available here precisely because
    the boundary is between submissions rather than inside one.

    The softmax artifact is deliberately NOT evicted: no runtime loop
    tiling, same class as the band add/ln/mul ELFs that already re-execute
    128 times per layer clean.

    The pool eviction is TARGETED, not wholesale. Only the pools whose
    sequences involve the two attention artifacts are dropped -- the reuse
    being broken is theirs, and the measured footgun is context state, not
    BO state (pooled ELF-ABI BOs are allocated device-level,
    ``xrt.ext.bo``, and survive a context unload). The first version did
    ``cache._pools.clear()``, which also destroyed the content-keyed
    static-weight pools and made every runlist-front layer run re-upload
    ~14 MB of weights at 4096 -- measured as the zero warm-vs-cold byte
    drop in doc 30, and never a safety property.
    """
    names = {cfg["artifacts"]["attn_scores"], cfg["artifacts"]["attn_output"]}
    for name in names:
        loaded = cache._loaded.pop(name, None)
        if loaded is not None:
            loaded[0].unload()
    cache.evict_pools_for(names)


def run_attention_head(cache, cfg, head, q, k, v, mask=None):
    """One head's whole attention interior, three entries in one submission.

    ``attn_scores`` (Q_h @ K_h^T), ``softmax`` over the score matrix, then
    ``attn_output`` (P_h @ V_h). Nothing crosses to the host between them:
    the score matrix and the probability matrix are device-resident inside
    the runlist, which is the property this mode exists to have and the one
    the host-softmax ``offload`` partition deliberately gives up.

    ``mask`` is the decoder's [seq, seq] additive causal mask
    (``causal_mask_bias``: 0 on and below the diagonal, -10000 above, bf16 --
    NOT -inf, which turns the bf16 add into NaN). It becomes a FOURTH entry
    between the score GEMM and the softmax, the validated ``causal_mask`` op
    at score shape; the softmax then reads the masked scores. One static
    content-keyed buffer serves all heads and all dispatches. The q slices
    arrive pre-scaled, so the masked tensor is scale*QK^T - 10000 above the
    diagonal, which the plain streaming softmax drives to exactly zero.

    ONE HEAD PER SUBMISSION IS A MEMORY BOUND. See
    ``runlist_submission_count``: all twelve heads in one runlist would
    need every score and probability matrix live at once, ~800 MiB at the
    gate configuration against ~70 MiB per head.

    The head slice and the K transpose are contiguous host COPIES, per the
    module's band-inputs footgun -- a strided view would upload the wrong
    bytes without complaining. They are layout, not arithmetic; the mode
    still computes nothing.
    """
    seq_len, head_dim = cfg["seq_len"], cfg["head_dim"]
    names = cfg["artifacts"]
    columns = slice(head * head_dim, (head + 1) * head_dim)
    # `[2026-08-19]` THE ATTENTION SCALE, APPLIED WHERE builders/softmax.py
    # SAYS IT MUST BE. The device softmax is the plain streaming family
    # (SM_LOG2E only), so 1/sqrt(head_dim) has to be applied upstream -- the
    # documented offload precedent -- and from the runlist front's landing
    # until today NOTHING applied it: the interior computed
    # softmax(QK^T) and every per-boundary gate passed anyway, because the
    # golden model keeps the deviation inside attn_context's 1e-3 absolute
    # ceiling while the RELATIVE error sat at 3.7-9.8e-2 (25x the other
    # modes) in every recorded log. Settled on device (devq 363): a crafted
    # q/k makes the two hypotheses' contexts diverge, and the device output
    # correlated 0.9993 with the UNSCALED form against 0.45 with the scaled
    # one. Folded into the q head slice, one multiply per head: head_dim is
    # 64 everywhere in the matrix, 1/8 is a power of two, so the bf16
    # multiply is an exponent shift -- exact, no added rounding.
    arrays = {
        "q_h": np.ascontiguousarray(
            (q[:, columns].astype(np.float32) * (1.0 / np.sqrt(head_dim))).astype(
                bfloat16
            )
        ),
        "k_h_t": np.ascontiguousarray(k[:, columns].T),
        "v_h": np.ascontiguousarray(v[:, columns]),
        "scores": np.zeros((seq_len, seq_len), dtype=bfloat16),
        "probs": np.zeros((seq_len, seq_len), dtype=bfloat16),
        "ctx_h": np.zeros((seq_len, head_dim), dtype=bfloat16),
    }
    steps = []
    scratches = set()
    step, scratch = _gemm_step(cfg, "attn_scores", "q_h", "k_h_t", "scores", arrays)
    steps.append(step)
    if scratch:
        scratches.add(scratch)
    softmax_in = "scores"
    if mask is not None:
        arrays["mask"] = mask
        arrays["masked"] = np.zeros((seq_len, seq_len), dtype=bfloat16)
        steps.append(
            DispatchStep(names["causal_mask"], ("scores", "mask", "masked"), writes=(2,))
        )
        softmax_in = "masked"
    steps.append(DispatchStep(names["softmax"], (softmax_in, "probs"), writes=(1,)))
    step, scratch = _gemm_step(cfg, "attn_output", "probs", "v_h", "ctx_h", arrays)
    steps.append(step)
    if scratch:
        scratches.add(scratch)
    specs = {
        name: _spec_buf(
            name, arr, static=name == "mask", host_output=name == "ctx_h"
        )
        for name, arr in arrays.items()
    }
    host_writes = {"q_h", "k_h_t", "v_h"} | scratches
    if mask is not None:
        # A static buffer still needs the host write, exactly as the norm
        # chain's gamma does -- left out, the device reads ZEROS and the mask
        # add is silently `scores + 0`: the first decoder walk (devq 365)
        # failed from attn_context onward with exactly that signature, and the
        # step probe (devq 366) read the unmasked scores back out of `masked`.
        host_writes.add("mask")
    evict_attention_contexts(cache, cfg)
    results, vector = cache.run_sequence(
        steps,
        specs,
        cfg["backend_kwargs"],
        arrays,
        host_writes=host_writes,
        require_single_submission=True,
    )
    return np.array(results["ctx_h"], copy=True), vector


def run_attention_interior(cache, cfg, q, k, v, mask=None):
    """Every head's attention interior, one submission each, columns reassembled.

    The loop is here rather than in a mode's ``dispatch`` because it is the
    ``runlist`` FRONT: `28-coarse-blend-space.md`'s C3 cell pairs exactly this
    with a banded tail, and a second copy of the reassembly is a second place
    for the column slicing to drift. Progress narration stays with the caller —
    a mode numbers its own submissions.
    """
    seq_len, emb_dim = cfg["seq_len"], cfg["emb_dim"]
    head_dim = cfg["head_dim"]
    attn_context = np.empty((seq_len, emb_dim), dtype=bfloat16)
    vectors = []
    for head in range(cfg["num_heads"]):
        columns = slice(head * head_dim, (head + 1) * head_dim)
        ctx_h, vector = run_attention_head(cache, cfg, head, q, k, v, mask=mask)
        attn_context[:, columns] = ctx_h
        vectors.append(vector)
    return attn_context, vectors


def run_o_proj(cache, cfg, ctx, w_o):
    """Runlist 2: the output projection, one entry."""
    seq_len, emb_dim = cfg["seq_len"], cfg["emb_dim"]
    arrays = {"ctx": ctx, "w_o": w_o}
    arrays["attn_out"] = np.zeros((seq_len, emb_dim), dtype=bfloat16)
    step, scratch = _gemm_step(cfg, "o_proj", "ctx", "w_o", "attn_out", arrays)
    specs = {
        name: _spec_buf(name, arr, static=name == "w_o", host_output=name == "attn_out")
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


def run_norm_chain(cache, cfg, label, x_full, residual_full, gamma_band):
    """Runlists 3 and 5: one normalization point as ``norm_blocks`` bands
    of add -> LayerNorm -> gamma multiply, ``3 * norm_blocks`` entries in
    one submission.

    ``x_full`` and ``residual_full`` are whole ``[seq_len, emb_dim]``
    tensors cut into bands here, because a dispatch argument is a whole
    BO. Within a band the residual sum and the normalized rows stay
    device-resident; ``gamma_band`` is ONE static buffer shared by every
    band's multiply.
    """
    emb_dim = cfg["emb_dim"]
    names = cfg["artifacts"]
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
        steps.append(DispatchStep(names["add"], (x_name, r_name, s_name), writes=(2,)))
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


def run_add_bands(cache, cfg, label, a_full, b_full):
    """One bare elementwise add over ``[seq_len, emb_dim]``, banded.

    The DECODER's raw residual stream: ``norm_blocks`` band entries of the
    same ``add`` artifact the norm chains dispatch, in one submission, with
    no layer_norm or gamma multiply after them -- the decoder's residual is
    the unnormalized sum.
    """
    emb_dim = cfg["emb_dim"]
    names = cfg["artifacts"]
    rows, blocks = cfg["norm_rows"], cfg["norm_blocks"]
    arrays = {}
    steps = []
    out_names = []
    for i in range(blocks):
        lo, hi = i * rows, (i + 1) * rows
        a_name, b_name, o_name = f"{label}_a{i}", f"{label}_b{i}", f"{label}_out{i}"
        # Contiguous copies, not views — see the module footguns.
        arrays[a_name] = np.ascontiguousarray(a_full[lo:hi])
        arrays[b_name] = np.ascontiguousarray(b_full[lo:hi])
        arrays[o_name] = np.zeros((rows, emb_dim), dtype=bfloat16)
        steps.append(DispatchStep(names["add"], (a_name, b_name, o_name), writes=(2,)))
        out_names.append(o_name)
    specs = {
        name: _spec_buf(name, arr, host_output=name in out_names)
        for name, arr in arrays.items()
    }
    host_writes = set()
    for i in range(blocks):
        host_writes |= {f"{label}_a{i}", f"{label}_b{i}"}
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


def run_ffn(cache, cfg, hidden, w_up, w_down):
    """Runlist 4: up_proj, GeLU, down_proj — three entries, the
    interiors device-resident."""
    seq_len, emb_dim = cfg["seq_len"], cfg["emb_dim"]
    ffn_dim = cfg["ffn_dim"]
    names = cfg["artifacts"]
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


def prepare_runlist(shape, seed=42, weights=None):
    """The ``runlist`` mode's ``SPECS`` preparer: the D2 layer, fine-grained.

    Same golden model, same per-boundary comparisons at ``BLOCK_STAGE_ATOL``,
    same injection target (``ln1_weight`` — the measured choice; here it feeds
    the first gamma multiply, scaling one column of ``hidden`` and cascading
    through both residual paths exactly as in the block). What differs is the
    execution boundary: every operator its own device kernel, nothing on the
    host, over ``runlist_submission_count`` runlists — 17 at the gate
    configuration, twelve of them the per-head attention interior.

    ``weights`` is doc 58 phase M1's injection seam. ``None`` -- every pre-M1
    caller -- draws the weight set exactly as before; a dict replaces it,
    leaving the input activation, the draw order, the oracle and the injection
    target where they are.
    """
    seq_len, emb_dim = shape["seq_len"], shape["emb_dim"]
    ffn_dim, num_heads = shape["ffn_dim"], shape["num_heads"]
    head_dim = shape["head_dim"]
    # The variant rides in the shape dict (run_mode._shape_for stamps it).
    # The decoder's one new DEVICE piece is the [seq, seq] mask-tensor add
    # between the score GEMM and the softmax -- the validated `causal_mask`
    # op at score shape; the pre-norms and raw residual adds reuse the band
    # artifacts the encoder's norm chains already dispatch.
    variant = shape.get("workload_variant", "encoder_bert")
    causal = variant == "decoder_gpt2"
    boundary_names = DECODER_BOUNDARIES if causal else ENCODER_BOUNDARIES
    stage_atol = decoder_stage_atol(emb_dim) if causal else BLOCK_STAGE_ATOL

    cfg = runlist_config(seq_len, emb_dim, ffn_dim, num_heads, head_dim, causal=causal)
    describe_runlist(cfg)

    golden_kwargs = dict(
        seq_len=seq_len,
        hidden_size=emb_dim,
        intermediate_size=ffn_dim,
        num_heads=num_heads,
        seed=seed,
        workload_variant=variant,
    )
    golden = generate_golden_reference(**golden_kwargs, weights=weights)
    reference = golden["boundaries"]
    # The order is RUNLIST_INPUT_NAMES, read back OUT of the tuple rather than
    # transcribed beside it; `inject` below indexes into the same tuple.
    inputs = layer_inputs(golden, RUNLIST_INPUT_NAMES)

    from shared.infra.cache import KernelCache, Profiler

    cache = KernelCache(
        cache_dir=RUNLIST_CACHE_DIR, verbose=False, profiler=Profiler(enabled=True)
    )
    compile_runlist_artifacts(cache, cfg, run_only=True)

    def dispatch(device_inputs, stage_stats, forward_done=None):
        cache.profiler.cpu_times.clear()
        reconfig_baseline = cache.reconfiguration_counts()
        x, w_q, w_k, w_v, w_o, ln1_weight, w_up, w_down, ln2_weight = device_inputs
        blocks = cfg["norm_blocks"]

        total = runlist_submission_count(cfg)
        gamma1 = round_bf16(broadcast_row_weight(ln1_weight, cfg["norm_rows"]))
        gamma2 = round_bf16(broadcast_row_weight(ln2_weight, cfg["norm_rows"]))
        boundaries = {}
        if causal:
            zeros = np.zeros_like(x)
            mask = causal_mask_bias(cfg["seq_len"], bfloat16)
            print(
                f"  [runlist 1/{total}] {blocks} x (add + layer_norm + mul) "
                f"ln1 pre-norm, zero residual ({3 * blocks} entries)"
            )
            ln_in, vec_0 = run_norm_chain(cache, cfg, "ln1", x, zeros, gamma1)
            boundaries["ln_in"] = ln_in
            print(f"  [runlist 2/{total}] q_proj + k_proj + v_proj (3 entries)")
            proj, vec_1 = run_projections(cache, cfg, ln_in, w_q, w_k, w_v)
            print(
                f"  [runlist 3..{2 + num_heads}/{total}] attention on device: "
                f"{num_heads} x (attn_scores + causal_mask + softmax + "
                f"attn_output), 4 entries each"
            )
            attn_context, attn_vectors = run_attention_interior(
                cache, cfg, proj["q"], proj["k"], proj["v"], mask=mask
            )
            print(f"  [runlist {3 + num_heads}/{total}] output_proj (1 entry)")
            attn_out, vec_2 = run_o_proj(cache, cfg, round_bf16(attn_context), w_o)
            print(
                f"  [runlist {4 + num_heads}/{total}] {blocks} x add "
                f"(residual = attn_out + x, {blocks} entries)"
            )
            residual, vec_r1 = run_add_bands(cache, cfg, "res1", attn_out, x)
            boundaries["residual"] = residual
            print(
                f"  [runlist {5 + num_heads}/{total}] {blocks} x "
                f"(add + layer_norm + mul) ln2 pre-norm, zero residual "
                f"({3 * blocks} entries)"
            )
            ffn_in, vec_3 = run_norm_chain(cache, cfg, "ln2", residual, zeros, gamma2)
            boundaries["ffn_in"] = ffn_in
            print(
                f"  [runlist {6 + num_heads}/{total}] up_proj + gelu + "
                f"down_proj (3 entries)"
            )
            ffn, vec_4 = run_ffn(cache, cfg, ffn_in, w_up, w_down)
            print(
                f"  [runlist {7 + num_heads}/{total}] {blocks} x add "
                f"(output = ffn_out + residual, {blocks} entries)"
            )
            output, vec_5 = run_add_bands(cache, cfg, "res2", ffn["ffn_out"], residual)
            vecs = [vec_0, vec_1] + attn_vectors + [vec_2, vec_r1, vec_3, vec_4, vec_5]
        else:
            print(f"  [runlist 1/{total}] q_proj + k_proj + v_proj (3 entries)")
            proj, vec_1 = run_projections(cache, cfg, x, w_q, w_k, w_v)

            print(
                f"  [runlist 2..{1 + num_heads}/{total}] attention on device: "
                f"{num_heads} x (attn_scores + softmax + attn_output), "
                f"3 entries each"
            )
            attn_context, attn_vectors = run_attention_interior(
                cache, cfg, proj["q"], proj["k"], proj["v"]
            )

            print(f"  [runlist {2 + num_heads}/{total}] output_proj (1 entry)")
            attn_out, vec_2 = run_o_proj(cache, cfg, round_bf16(attn_context), w_o)

            print(
                f"  [runlist {3 + num_heads}/{total}] {blocks} x "
                f"(add + layer_norm + mul) ln1 ({3 * blocks} entries)"
            )
            hidden, vec_3 = run_norm_chain(cache, cfg, "ln1", attn_out, x, gamma1)

            print(
                f"  [runlist {4 + num_heads}/{total}] up_proj + gelu + down_proj "
                f"(3 entries)"
            )
            ffn, vec_4 = run_ffn(cache, cfg, hidden, w_up, w_down)

            print(
                f"  [runlist {5 + num_heads}/{total}] {blocks} x "
                f"(add + layer_norm + mul) ln2 ({3 * blocks} entries)"
            )
            output, vec_5 = run_norm_chain(
                cache, cfg, "ln2", ffn["ffn_out"], hidden, gamma2
            )
            boundaries["hidden"] = hidden
            vecs = [vec_1] + attn_vectors + [vec_2, vec_3, vec_4, vec_5]

        boundaries.update(ffn)
        boundaries.update({"q": proj["q"], "k": proj["k"], "v": proj["v"]})
        boundaries["attn_out"] = attn_out
        boundaries["output"] = output
        # The forward is DONE here: every boundary is a host array (attn_context
        # as the bf16 the device produced). The study's clock stops at this
        # instant (operator rule, 2026-08-22); everything below -- the f32
        # widening that exists only for the comparison, and the comparison --
        # is verification and runs outside it.
        if forward_done is not None:
            forward_done()
        # bf16 straight from the device, widened for the comparison. The other
        # modes' attn_context is f32 because a host implementation produced it;
        # here the widening is exact and adds nothing.
        boundaries["attn_context"] = attn_context.astype(np.float32)

        vector_rows = [v.as_row() for v in vecs]
        stages = []
        for name in boundary_names:
            atol = stage_atol[name]
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
            # What THIS dispatch loaded and attached (schema v2's
            # reconfiguration columns). Steady state here is the per-head
            # attention reloads -- `evict_attention_contexts` drops both
            # attention GEMM contexts before every head, so each head pays two
            # loads -- while every other artifact's context stands for the
            # process. That reload count is this mode's real reconfiguration
            # cost, the thing its bytes advantage over `offload` is traded
            # against.
            **reconfiguration_delta(cache, reconfig_baseline),
        }

    record_extra = {
        "variant": variant,
        "causal": causal,
        "golden_seed": seed,
        # Which of the two weight sources this run used (doc 58 M1). Recorded
        # rather than inferred: a results tree that mixes generated and injected
        # weights is mixing two experiments, and the seed alone no longer
        # identifies the tensors once injection exists.
        "weight_source": "injected" if weights is not None else "generated",
        "execution_mode": EXECUTION_MODE_CSV["runlist"],
        "attention_path": ATTENTION_PATH,
        "norm_rows": cfg["norm_rows"],
        "norm_blocks": cfg["norm_blocks"],
        "runlist_entries": runlist_entry_count(cfg),
        "runlist_submissions": runlist_submission_count(cfg),
        "softmax_rows_per_call": cfg["softmax_rows_per_call"],
        # MIXED: the three projection shapes resolve in the registry, the two
        # attention shapes are injected measured tiles that resolve in none.
        "gemm_spec_source": "registry+injected",
        "gemm_spec_proj": _spec_digest(cfg["specs"]["proj"][0]),
        "gemm_spec_ffn_up": _spec_digest(cfg["specs"]["up"][0]),
        "gemm_spec_ffn_down": _spec_digest(cfg["specs"]["down"][0]),
        "gemm_spec_attn_scores": _spec_digest(cfg["specs"]["attn_scores"][0]),
        "gemm_spec_attn_output": _spec_digest(cfg["specs"]["attn_output"][0]),
    }
    prepared = {
        "inputs": inputs,
        # ln1_weight, index 5. Same measured target as the block's; see
        # opcheck_layer.py for the numbers.
        "inject": (RUNLIST_INPUT_NAMES.index("ln1_weight"), (0,)),
        "expected": [reference["output"]],
        "dispatch": dispatch,
        "record_extra": record_extra,
    }
    # Injected weights only: the delta is DERIVED from the response, because
    # `opcheck.py`'s constant is calibrated against the generated scale and a
    # real weight set moves that calibration. Deferred -- resolved only when
    # `opcheck.py` is actually injecting. `opcheck_layer.py`'s section header
    # carries the measurement, including the decoder case where the shipped 2.0
    # stops discriminating entirely.
    hook = fault_delta_hook(
        golden_kwargs,
        weights,
        "ln1_weight",
        (0,),
        reference["output"],
        stage_atol["output"],
    )
    if hook is not None:
        prepared["fault_delta"] = hook
    if causal:
        # Same seam as opcheck_layer's: the decoder's whole-layer comparison
        # runs at its own output boundary's atol, not the spec row's
        # encoder-measured one. run_mode prefers this key when present.
        prepared["atol"] = stage_atol["output"]
    return prepared
