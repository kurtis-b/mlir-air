# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Backend kwarg presets for the kernels used by the LLAMA-3 example.

These dicts are passed to `cache.load_and_run(...)` (and equivalent helpers)
as the per-kernel `backend_kwargs`. Centralized here so callers don't
re-build identical dicts on every invocation, and so prefill / decode /
inference share the same canonical values.
"""

# ---------------------------------------------------------------------------
# Generic / shared
# ---------------------------------------------------------------------------

SIMPLE_BACKEND = {"omit_while_true_loop": False}

# ---------------------------------------------------------------------------
# Prefill (multi-launch ELFs)
# ---------------------------------------------------------------------------

RMS_GEMMS_ROPE_BACKEND = {
    "omit_while_true_loop": False,
    "output_format": "elf",
    "instance_name": "rms_gemms_rope",
}

O_FFN_BACKEND = {
    "omit_while_true_loop": False,
    "output_format": "elf",
    "instance_name": "o_ffn",
}

# ---------------------------------------------------------------------------
# Decode (GEMV multi-launch ELFs)
# ---------------------------------------------------------------------------
#
# K=2048 GEMV uses ping-pong (L1 fits both buffers).
# `[2026-08-27]` queue item 28: the explicit `use_lock_race_condition_fix: False`
# is GONE from this preset. It was a legacy default rather than a considered
# decline, and it read as one: `o_gemv_ffn` inherits this dict and its stage-2
# herd is 8 x 4, so the preset was telling the compile chokepoint "no fix" for a
# multi-row herd. Absent means "the chokepoint decides", which is what every
# caller of this preset actually wants -- single-row kernels get nothing (the
# XRTBackend default is False anyway) and multi-row ones get the fix.
GEMV_K2048_BACKEND = {
    "omit_while_true_loop": False,
    "omit_pingpong": "",
    "runtime_loop_tiling_sizes": [16, 16],
}

RGR_BACKEND = {
    "output_format": "elf",
    "instance_name": "rms_gemv_rope",
    **GEMV_K2048_BACKEND,
}

# OGF includes the K=8192 down-projection GEMV; ping-pong off because
# L1 is too tight to hold both buffers for that K.
OGF_BACKEND = {
    "output_format": "elf",
    "instance_name": "o_gemv_ffn",
    "omit_pingpong": "all",
    **{k: v for k, v in GEMV_K2048_BACKEND.items() if k != "omit_pingpong"},
}

LM_GEMV_BACKEND = {
    "output_format": "elf",
    "instance_name": "lm_head_gemv",
    **GEMV_K2048_BACKEND,
}

# ---------------------------------------------------------------------------
# Decode (int4-AWQ ELFs). Inherits ping-pong + runtime_loop_tiling from
# GEMV_K2048_BACKEND; dropping ping-pong regressed e2e 12.4 -> 7.8 tok/s.
# ---------------------------------------------------------------------------

RGR_INT4_BACKEND = {
    "output_format": "elf",
    "instance_name": "rms_qkv_int4_rope",
    "stack_size": 4096,
    **GEMV_K2048_BACKEND,
}

OGF_INT4_BACKEND = {
    "output_format": "elf",
    "instance_name": "o_gemv_ffn_int4",
    "stack_size": 4096,
    **GEMV_K2048_BACKEND,
}


# ---------------------------------------------------------------------------
# Herd rows (queue item 28)
# ---------------------------------------------------------------------------
#
# `[2026-08-27]` THERE IS NO `with_herd_rows` HELPER AND NO PRESET SETS
# `use_lock_race_condition_fix`, deliberately.
#
# It existed, and it was a second way to acquire the flag: it injected from a
# ROW COUNT, so any caller that knew it was building a multi-row herd could turn
# it on. Round 5 claimed the builder-emitted mark was the only trigger while
# this was still here, which made the claim false -- and over-broad injection is
# exactly what faulted the device on the study's QKV split-cast form (devq
# 812/813). One trigger, in one place: `matvec.py` marks the herd it builds at
# `herd_rows > 1`, and `KernelCache.compile_and_cache` supplies the flag for
# that mark and for nothing else.
#
# The presets also no longer write `use_lock_race_condition_fix: False`. That
# was a legacy default rather than a decision, and an explicit False on a marked
# herd is refused rather than overridden -- see
# `dispatch.ensure_lock_fix_for_marked_herds`.
