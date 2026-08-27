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
GEMV_K2048_BACKEND = {
    "omit_while_true_loop": False,
    "omit_pingpong": "",
    "runtime_loop_tiling_sizes": [16, 16],
    "use_lock_race_condition_fix": False,
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


def with_herd_rows(backend, herd_rows):
    """`backend` with the lock-race fix set iff `herd_rows > 1`.

    `[2026-08-26]` queue item 28. A herd occupying more than one core row makes
    each column's L2 input tile single-writer / multi-reader, and the shipped
    decode preset then HANGS on device with `ERT_CMD_STATE_TIMEOUT`. Item 27
    section 6.1 bisected it over three knobs x two row counts (devq 673/674):
    `--use-lock-race-condition-fix` is the ONLY knob that unblocks it (ping-pong
    and `runtime_loop_tiling_sizes` are irrelevant), and it costs +0.8 % at
    8 cores. v2 also works and costs +12 %, so v1 is the one used.

    This helper is the EASY half of the coupling: one call derives the flag from
    the row count, so a call site cannot set one and forget the other. The HARD
    half is `dispatch.check_herd_rows_lock_fix`, which reads the geometry off the
    compiled module inside `KernelCache.compile_and_cache` and **fails closed** --
    a multi-row herd is refused unless the kernel is on
    `dispatch.HERD_ROWS_MEASURED_GREEN`, and geometry it cannot decode counts as
    multi-row. That one catches a builder or a caller this helper never touched,
    including the same builder reached through a renamed micro-kernel object.

    `herd_rows` may be an int or a per-partition sequence (the LM head's mixed
    4/2 form); the maximum decides.
    """
    rows = max(herd_rows) if hasattr(herd_rows, "__iter__") else int(herd_rows)
    if rows <= 1:
        return dict(backend)
    return {**backend, "use_lock_race_condition_fix": True}
