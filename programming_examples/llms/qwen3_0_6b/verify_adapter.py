# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verify adapter for the bf16 Qwen3-0.6B example.

Wraps the production `qwen3_0_6b_inference` driver into a Runner that
satisfies `verify/runners/base.Runner`. The shared verify framework
imports this via `--runner=qwen3_0_6b.verify_adapter`.

Qwen3-0.6B has no separate "-Instruct" repo (Qwen3 unifies base and
instruct), so both MODEL_CHOICES keys map to "Qwen/Qwen3-0.6B".
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_THIS_DIR = Path(__file__).resolve().parent
_LLMS_DIR = _THIS_DIR.parent
_VERIFY = _LLMS_DIR / "verify"
for _p in (str(_LLMS_DIR), str(_VERIFY), str(_THIS_DIR)):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from shared.infra.cache import KernelCache  # noqa: E402
from qwen3_0_6b_prefill import (  # noqa: E402
    compile_all_kernels as compile_prefill_kernels,
    run_transformer_block_qwen3 as run_prefill_block,
)
from qwen3_0_6b_decode import compile_decode_kernels  # noqa: E402
from qwen3_0_6b_inference import (  # noqa: E402
    prepare_runtime,
    run_npu_prefill,
    run_npu_decode_step,
)
from qwen3_0_6b_weights import (  # noqa: E402
    LlamaConfig,
    load_weights,
    generate_rope_lut,
)
from qwen3_0_6b_cpu_helpers import rms_norm  # noqa: E402
from runners._records import DecodeStepRecord, PrefillRecord  # noqa: E402

# Qwen3 unifies base + instruct in one checkpoint.
MODEL_CHOICES = {
    "base": "Qwen/Qwen3-0.6B",
    "instruct": "Qwen/Qwen3-0.6B",
}
DEFAULT_MODEL = "instruct"

EPS = 1e-6


def resolve_model(model_choice_or_id: str) -> str:
    return MODEL_CHOICES.get(model_choice_or_id, model_choice_or_id)


def hf_reference(npu_model_name: str) -> str:
    """bf16 baseline is its own reference (NPU bf16 vs HF bf16)."""
    return npu_model_name


def build_config():
    return LlamaConfig()


def build_runner(
    model_name, config, max_seq, tokenizer, *, npu_attn=True, lite_mode=False
):
    weights = load_weights(model_name, config=config)
    return NpuRunner(
        weights=weights,
        config=config,
        max_seq=max_seq,
        tokenizer=tokenizer,
        npu_attn=npu_attn,
        lite_mode=lite_mode,
    )


class NpuRunner:
    """Adapter over the bf16 production NPU prefill + decode functions."""

    name = "npu_bf16"

    def __init__(
        self, weights, config, max_seq, tokenizer, npu_attn=True, lite_mode=False
    ):
        self.weights = weights
        self.config = config
        self.max_seq = max_seq
        self.npu_attn = npu_attn
        self.cpu_attn = (
            not npu_attn
        )  # Qwen3 prefill attention is NPU FlashAttention by default (CPU only if npu_attn is explicitly disabled).
        self.lite_mode = lite_mode
        self._tokenizer = tokenizer

        self.rope_lut_bf16 = generate_rope_lut(config=config, seq_len=max_seq).astype(
            bfloat16
        )

        # `[2026-08-23]` THE TIMED ARTIFACT SET -- see llama32_1b/verify_adapter.py
        # for the contract: LLMS_VERIFY_PREFILL_CACHE / LLMS_VERIFY_DECODE_CACHE
        # are LOADED, never compiled; LLMS_VERIFY_PREFILL_M is the pad target;
        # unset, production compiles into build_peano and pads to max_seq.
        _prefill_dir = os.environ.get("LLMS_VERIFY_PREFILL_CACHE")
        _decode_dir = os.environ.get("LLMS_VERIFY_DECODE_CACHE")
        self.prefill_M = int(os.environ.get("LLMS_VERIFY_PREFILL_M") or max_seq)
        if self.prefill_M > max_seq:
            raise ValueError(f"LLMS_VERIFY_PREFILL_M {self.prefill_M} exceeds max_seq {max_seq}")
        _cache_root = _THIS_DIR / "build_peano"
        self.prefill_cache = KernelCache(
            _prefill_dir or str(_cache_root / "prefill_kernel_cache"), verbose=False
        )
        if _prefill_dir:
            if not self.prefill_cache.load_manifest():
                raise RuntimeError(f"LLMS_VERIFY_PREFILL_CACHE {_prefill_dir}: no loadable manifest")
            # the QKV + o_ffn scratch-arg layouts, which only compiling used to
            # set. A FORCED artifact set (compile.json artifact_deviation, the
            # gemm_method= override) restores the DEVIATION's o_ffn layout: the
            # ELF's arg count is the deviation's, and the registry layout would
            # leave a forced fused-cast ELF's f32 scratch args unbound (item 14;
            # observed live as a nondeterministic token-gate FAIL, devq 583).
            import json as _json

            from qwen3_0_6b_prefill import restore_scratch_layout

            _kwargs = {}
            _note = Path(_prefill_dir) / "compile.json"
            if _note.is_file():
                _dev = _json.loads(_note.read_text(encoding="utf-8")).get("artifact_deviation") or {}
                if _dev.get("o_ffn_gemm_method"):
                    _kwargs["o_ffn_gemm_method"] = _dev["o_ffn_gemm_method"]
            restore_scratch_layout(config, self.prefill_M, **_kwargs)
        else:
            compile_prefill_kernels(
                self.prefill_cache, config, seq_len=max_seq, cpu_attn=self.cpu_attn
            )
        self.decode_cache = KernelCache(
            _decode_dir or str(_cache_root / "decode_kernel_cache"), verbose=False
        )
        if _decode_dir:
            if not self.decode_cache.load_manifest():
                raise RuntimeError(f"LLMS_VERIFY_DECODE_CACHE {_decode_dir}: no loadable manifest")
        else:
            compile_decode_kernels(self.decode_cache, config)
        self.loaded_artifacts = {
            name: art.output_binary
            for cache in (self.prefill_cache, self.decode_cache)
            for name, art in cache.artifacts.items()
        }
        print(f"[verify-adapter] artifact set: prefill_M={self.prefill_M} max_seq={max_seq} "
              f"{'LOADED' if _prefill_dir or _decode_dir else 'compiled'}: "
              f"{sorted(self.loaded_artifacts.values())}")

        prepare_runtime(
            self.prefill_cache,
            self.decode_cache,
            weights,
            config,
            self.prefill_M,
            self.rope_lut_bf16,
        )

        self.k_cache = None
        self.v_cache = None

    def prefill(self, prompt_tokens: np.ndarray) -> PrefillRecord:
        eos = self._tokenizer.eos_token_id
        if len(prompt_tokens) < self.prefill_M:
            padded = list(prompt_tokens) + [eos] * (self.prefill_M - len(prompt_tokens))
        else:
            padded = list(prompt_tokens)[: self.prefill_M]
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
            profile=False,
            quiet=True,
        )
        self.k_cache = k_cache
        self.v_cache = v_cache

        if self.lite_mode:
            empty = np.empty((0,), dtype=np.float32)
            return PrefillRecord(
                layer_intermediates=[],
                final_hidden_normed=empty,
                logits_at_pred=logits_row,
                top1_token=prefill_token,
            )

        # Diagnosis-only: re-run the prefill layer loop capturing per-layer
        # ffn_out + final post-norm hidden state.
        cfg = self.config
        if len(prompt_tokens) < self.prefill_M:
            pad = np.zeros(self.prefill_M - len(prompt_tokens), dtype=prompt_tokens.dtype)
            padded_diag = np.concatenate([prompt_tokens, pad])
        else:
            padded_diag = prompt_tokens[: self.prefill_M]
        x = self.weights.embed_table[padded_diag].astype(np.float32).astype(bfloat16)
        layer_intermediates = []
        for li in range(cfg.n_layers):
            x, ints = run_prefill_block(
                x,
                self.weights.layers[li],
                self.rope_lut_bf16,
                cfg,
                self.prefill_cache,
                layer_idx=li,
                cpu_attn=self.cpu_attn,
                verbose=False,
            )
            fo_full = np.asarray(ints["ffn_out"])
            layer_intermediates.append({"ffn_out": fo_full[:prompt_len]})

        x_full_f32 = np.asarray(x, dtype=np.float32)[:prompt_len]
        x_full_normed = rms_norm(x_full_f32, self.weights.final_norm, eps=EPS)

        return PrefillRecord(
            layer_intermediates=layer_intermediates,
            final_hidden_normed=x_full_normed.astype(np.float32),
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
        return DecodeStepRecord(lm_head_logits=logits, top1_token=next_token)
