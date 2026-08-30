# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Verify adapter for the int4 (GGUF q4_0) SmolLM2-1.7B example.

Wraps the production driver into a Runner for the shared verify framework.
Mirrors `llama32_1b_int4/verify_adapter.py` with two deliberate differences:

* Weights come from the GGUF q4_0 checkpoint through
  `smollm2_1_7b_int4_weights.load_weights_gguf_q4_0` (gs=32 packed BOs,
  prefill dequant taken from the q4_0 payloads).
* The HF reference is the PLAIN bf16 `HuggingFaceTB/SmolLM2-1.7B-Instruct`
  model — no dequant patching (`build_hf_model` deliberately absent, so the
  framework loads the reference itself). The NPU-vs-HF delta therefore
  legitimately INCLUDES q4_0 quantization error: this gate checks the
  quantized model against the full-precision one, which is the deployment's
  actual claim. The AWQ example patches dequant weights into the reference
  to isolate NPU drift; here the quantization is the thing being shipped.

The GGUF path resolves through `smollm2_1_7b_int4_weights.resolve_gguf_path`:
`$SMOLLM2_GGUF` (the Makefile exports its `GGUF=`), else the hub file via the
HF cache.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_THIS_DIR = Path(__file__).resolve().parent
_LLMS_DIR = _THIS_DIR.parent
_LLAMA_BF16 = _LLMS_DIR / "llama32_1b"
_LLAMA_INT4 = _LLMS_DIR / "llama32_1b_int4"
_SMOLLM2_BF16 = _LLMS_DIR / "smollm2_1_7b"
_VERIFY = _LLMS_DIR / "verify"
for _p in (
    str(_LLMS_DIR),
    str(_SMOLLM2_BF16),
    str(_LLAMA_BF16),
    str(_LLAMA_INT4),
    str(_VERIFY),
    str(_THIS_DIR),
):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from shared.infra.cache import KernelCache  # noqa: E402
from smollm2_1_7b_weights import LlamaConfig, generate_rope_lut  # noqa: E402
from smollm2_1_7b_int4_weights import (  # noqa: E402
    HF_MODEL,
    load_weights_gguf_q4_0,
    resolve_gguf_path,
)
from smollm2_1_7b_int4_decode import compile_decode_kernels  # noqa: E402
from llama32_1b_prefill import (  # noqa: E402
    compile_all_kernels as compile_prefill_kernels,
)
from llama32_1b_int4_inference import (  # noqa: E402
    _multi_launch_dir,
    prepare_runtime,
    run_npu_decode_step,
    run_npu_prefill,
)
from runners._records import DecodeStepRecord, PrefillRecord  # noqa: E402

MODEL_CHOICES = {"instruct": HF_MODEL}
DEFAULT_MODEL = "instruct"


def resolve_model(model_choice_or_id: str) -> str:
    return MODEL_CHOICES.get(model_choice_or_id, model_choice_or_id)


def hf_reference(npu_model_name: str) -> str:
    """The PLAIN bf16 checkpoint — see the module docstring: the gate is the
    quantized model against full precision, so the reference is unpatched."""
    return HF_MODEL


def build_config():
    return LlamaConfig()


# The framework calls build_runner once per verify run; cache the loaded
# weights in case a future caller builds twice in one process.
_WEIGHTS_CACHE: dict = {}


def _get_or_load_weights(config):
    gguf = resolve_gguf_path()
    key = (gguf, id(config))
    if key not in _WEIGHTS_CACHE:
        _WEIGHTS_CACHE[key] = load_weights_gguf_q4_0(gguf, HF_MODEL, config=config)
    return _WEIGHTS_CACHE[key]


def build_runner(
    model_name: str,
    config,
    max_seq: int,
    tokenizer,
    *,
    npu_attn: bool = True,
    lite_mode: bool = False,
):
    """Load GGUF q4_0 weights, compile NPU kernels (bf16 prefill + gs=32
    int4 decode), return an `Int4NpuRunner`."""
    weights = _get_or_load_weights(config)
    return Int4NpuRunner(
        weights=weights,
        config=config,
        max_seq=max_seq,
        tokenizer=tokenizer,
        npu_attn=npu_attn,
        lite_mode=lite_mode,
    )


class Int4NpuRunner:
    """Adapter over the int4 production NPU prefill + decode functions.

    Prefill is NPU bf16 on the q4_0-dequantized weights; decode is NPU int4
    at gs=32. Same runner shape as the AWQ example's.
    """

    name = "npu_int4_gguf"

    def __init__(
        self,
        weights,
        config,
        max_seq: int,
        tokenizer,
        npu_attn: bool = True,
        lite_mode: bool = False,
    ):
        self.weights = weights
        self.config = config
        self.max_seq = max_seq
        self.npu_attn = npu_attn
        self.cpu_attn = not npu_attn
        self.lite_mode = lite_mode
        self._tokenizer = tokenizer

        self.rope_lut_bf16 = generate_rope_lut(config=config, seq_len=max_seq).astype(
            bfloat16
        )

        # Reuse this model's build_peano kernel caches so verify and run
        # share one per-model cache; absolute paths keep it per-model. The
        # prefill compile resolves `multi_launch_builder` to the bf16
        # reference's package, the decode compile to the int4 reference's —
        # THIS directory deliberately ships no namesake package of its own.
        _cache_root = _THIS_DIR / "build_peano"
        self.prefill_cache = KernelCache(
            str(_cache_root / "prefill_kernel_cache"), verbose=False
        )
        with _multi_launch_dir(str(_LLAMA_BF16)):
            compile_prefill_kernels(
                self.prefill_cache, config, seq_len=max_seq, cpu_attn=self.cpu_attn
            )
        self.decode_cache = KernelCache(
            str(_cache_root / "decode_kernel_cache"), verbose=False
        )
        with _multi_launch_dir(str(_LLAMA_INT4)):
            compile_decode_kernels(self.decode_cache, config)

        prepare_runtime(
            self.prefill_cache,
            self.decode_cache,
            weights,
            config,
            max_seq,
            self.rope_lut_bf16,
        )

        self.k_cache = None
        self.v_cache = None

    def prefill(self, prompt_tokens: np.ndarray) -> PrefillRecord:
        eos = self._tokenizer.eos_token_id
        if len(prompt_tokens) < self.max_seq:
            padded = list(prompt_tokens) + [eos] * (self.max_seq - len(prompt_tokens))
        else:
            padded = list(prompt_tokens)[: self.max_seq]
        prefill_token, logits_row, k_cache, v_cache, prompt_len = run_npu_prefill(
            padded,
            self.weights,
            self.config,
            self.prefill_cache,
            self.decode_cache,
            self.rope_lut_bf16,
            self.max_seq,
            tokenizer=self._tokenizer,
            cpu_attn=self.cpu_attn,
            quiet=True,
            # Exact pre-padding length: SmolLM2's EOS (<|im_end|>) can appear
            # inside a templated prompt, so the driver's non-EOS-count
            # fallback would read the logits at the wrong row.
            prompt_len=min(len(prompt_tokens), self.max_seq),
        )
        self.k_cache = k_cache
        self.v_cache = v_cache

        # Lite mode is the verified path; the int4 prefill block does not
        # expose per-layer intermediates for diagnosis (same as the AWQ
        # example).
        empty = np.empty((0,), dtype=np.float32)
        layer_intermediates = (
            [] if self.lite_mode else [{} for _ in range(self.config.n_layers)]
        )
        return PrefillRecord(
            layer_intermediates=layer_intermediates,
            final_hidden_normed=empty,
            logits_at_pred=logits_row,
            top1_token=prefill_token,
        )

    def decode_step(self, input_token: int, current_pos: int) -> DecodeStepRecord:
        x = self.weights.embed_table[input_token].astype(bfloat16)
        next_token, logits = run_npu_decode_step(
            x,
            self.weights,
            self.config,
            self.decode_cache,
            self.rope_lut_bf16,
            self.k_cache,
            self.v_cache,
            current_pos,
        )
        return DecodeStepRecord(
            lm_head_logits=logits,
            top1_token=next_token,
        )
