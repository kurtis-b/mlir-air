# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verify adapter for the bf16 SmolLM2-1.7B example.

Wraps the production `llama32_1b_inference` driver into a Runner that
satisfies `verify/runners/base.Runner`. The shared verify framework
(see `programming_examples/llms/verify/verify_runner.py`) imports this
module via `--runner=smollm2_1_7b.verify_adapter` and calls `build_runner`.

Nothing here is reachable from production code; it exists only so the
verify gate can exercise the exact same NPU code path that `make run`
uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

# Ensure llms/, this dir, llms/verify/, and the llama32_1b reference (whose
# production prefill/decode/inference we inherit bit-for-bit) are importable.
_THIS_DIR = Path(__file__).resolve().parent
_LLMS_DIR = _THIS_DIR.parent
_VERIFY = _LLMS_DIR / "verify"
_LLAMA_REF = _LLMS_DIR / "llama32_1b"
for _p in (str(_LLMS_DIR), str(_VERIFY), str(_LLAMA_REF), str(_THIS_DIR)):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from shared.infra.cache import KernelCache  # noqa: E402

# SmolLM2 is pure MHA (kv_dim == emb_dim). The shared llama32_1b_prefill
# run_transformer_block / preload_prefill_weights are registry-driven and
# MHA-safe (per-shape gemm_registry_config supplies the fused-cast f32 C-scratch
# args), so they're used directly — no fork or monkeypatch.
from llama32_1b_prefill import (  # noqa: E402
    compile_all_kernels as compile_prefill_kernels,
    run_transformer_block as run_prefill_block,
)

from llama32_1b_decode import compile_decode_kernels  # noqa: E402
from llama32_1b_inference import (  # noqa: E402
    prepare_runtime,
    run_npu_prefill,
    run_npu_decode_step,
)
from smollm2_1_7b_weights import (  # noqa: E402
    LlamaConfig,
    load_weights,
    generate_rope_lut,
)
from smollm2_1_7b_cpu_helpers import rms_norm  # noqa: E402
from runners._records import DecodeStepRecord, PrefillRecord  # noqa: E402
from runners.bf16_npu_runner import Bf16NpuRunner  # noqa: E402

# CLI --model choice -> HF id. Both Llamas use the same architecture; only
# the weights and chat template differ.
MODEL_CHOICES = {
    "base": "HuggingFaceTB/SmolLM2-1.7B",
    "instruct": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
}
DEFAULT_MODEL = "base"


def resolve_model(model_choice_or_id: str) -> str:
    """`--model` accepts either a `MODEL_CHOICES` key (base/instruct) or a
    raw HF model id / local path. Return the HF id."""
    return MODEL_CHOICES.get(model_choice_or_id, model_choice_or_id)


def hf_reference(npu_model_name: str) -> str:
    """HF reference checkpoint for `npu_model_name`. bf16 baseline is its
    own reference (verifies NPU bf16 vs HF bf16 on the same checkpoint)."""
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
