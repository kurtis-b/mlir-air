# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verify adapter for the bf16 Llama-3.2-3B example.

Wraps the production `llama32_3b_inference` driver (which itself reuses the
config-driven llama32_1b runtime verbatim) into a Runner that satisfies
`verify/runners/base.Runner`. The shared verify framework imports this module
via `--runner=llama32_3b.verify_adapter` and calls `build_runner`.

Nothing here is reachable from production code; it exists only so the verify
gate exercises the exact same NPU code path that `make run` uses.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

# Ensure llms/, this dir, llms/verify/, and the sibling llama32_1b/ are importable.
_THIS_DIR = Path(__file__).resolve().parent
_LLMS_DIR = _THIS_DIR.parent
_VERIFY = _LLMS_DIR / "verify"
_LLAMA1B = _LLMS_DIR / "llama32_1b"
for _p in (str(_LLMS_DIR), str(_VERIFY), str(_LLAMA1B), str(_THIS_DIR)):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from shared.infra.cache import KernelCache  # noqa: E402
from llama32_3b_prefill import (  # noqa: E402
    compile_all_kernels as compile_prefill_kernels,
    run_transformer_block as run_prefill_block,
)
from llama32_3b_decode import compile_decode_kernels  # noqa: E402
from llama32_3b_inference import (  # noqa: E402
    prepare_runtime,
    run_npu_prefill,
    run_npu_decode_step,
)
from llama32_3b_weights import (  # noqa: E402
    LlamaConfig,
    load_weights,
    generate_rope_lut,
)
from llama32_3b_cpu_helpers import rms_norm  # noqa: E402
from runners._records import DecodeStepRecord, PrefillRecord  # noqa: E402
from runners.bf16_npu_runner import Bf16NpuRunner  # noqa: E402

# CLI --model choice -> HF id.
MODEL_CHOICES = {
    "base": "meta-llama/Llama-3.2-3B",
    "instruct": "meta-llama/Llama-3.2-3B-Instruct",
}
DEFAULT_MODEL = "instruct"


def resolve_model(model_choice_or_id: str) -> str:
    return MODEL_CHOICES.get(model_choice_or_id, model_choice_or_id)


def hf_reference(npu_model_name: str) -> str:
    """HF reference checkpoint: bf16 baseline is its own reference (NPU bf16 vs
    HF bf16 on the same checkpoint)."""
    return npu_model_name


def build_config():
    return LlamaConfig()


def build_runner(
    model_name: str,
    config,
    max_seq: int,
    tokenizer,
    *,
    npu_attn: bool = True,
    lite_mode: bool = False,
):
    """Load bf16 weights, compile NPU kernels, return a `NpuRunner`."""
    weights = load_weights(model_name, config=config)
    return NpuRunner(
        weights=weights,
        config=config,
        max_seq=max_seq,
        tokenizer=tokenizer,
        npu_attn=npu_attn,
        lite_mode=lite_mode,
    )


class NpuRunner(Bf16NpuRunner):
    """Adapter over the bf16 production NPU prefill + decode functions."""

    this_dir = _THIS_DIR
    eps = 1e-5  # llama/SmolLM2 rms_norm default, now explicit
    generate_rope_lut = staticmethod(generate_rope_lut)
    compile_prefill_kernels = staticmethod(compile_prefill_kernels)
    compile_decode_kernels = staticmethod(compile_decode_kernels)
    prepare_runtime = staticmethod(prepare_runtime)
    run_npu_prefill = staticmethod(run_npu_prefill)
    run_prefill_block = staticmethod(run_prefill_block)
    run_npu_decode_step = staticmethod(run_npu_decode_step)
    rms_norm = staticmethod(rms_norm)
