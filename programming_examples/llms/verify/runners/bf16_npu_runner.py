# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared bf16 `NpuRunner` for the `llms/` verify adapters.

Eight bf16 adapters carried this class as **two** textually distinct bodies
(hashes `5330e096`: llama32_1b, llama32_3b, smollm2_1_7b; `e4afbe4a`:
qwen25_0_5b/1_5b/3b, qwen3_1_7b/4b) whose only semantic difference was the
RMSNorm epsilon — see `eps` below. Everything else that varied was per-model
functions, which each subclass now supplies as class attributes.

A subclass is the model's prefill/decode API plus its eps:

    class NpuRunner(Bf16NpuRunner):
        this_dir = _THIS_DIR
        eps = 1e-5
        generate_rope_lut      = staticmethod(generate_rope_lut)
        compile_prefill_kernels = staticmethod(compile_prefill_kernels)
        compile_decode_kernels  = staticmethod(compile_decode_kernels)
        prepare_runtime        = staticmethod(prepare_runtime)
        run_npu_prefill        = staticmethod(run_npu_prefill)
        run_prefill_block      = staticmethod(run_prefill_block)
        run_npu_decode_step    = staticmethod(run_npu_decode_step)
        rms_norm               = staticmethod(rms_norm)

`eps` is REQUIRED and always passed explicitly to `rms_norm`. It is the one
place the two bodies genuinely disagreed: qwen passed `eps=EPS` (1e-6) while
llama/smollm2 passed nothing and relied on their own helper's default of 1e-5.
A shared base that omitted it would silently inherit whichever default the
shared import happened to carry, and be wrong for one family in a *verify*
path. Making it explicit preserves both behaviours and removes the footgun.

`name` stays a plain class attribute so a subclass may override it with a
property: qwen3_0_6b reports the decode precision actually selected, so a
report can never say `npu_bf16` while the int4 cascade dispatches.
"""

from __future__ import annotations

import numpy as np
from ml_dtypes import bfloat16

from shared.infra.cache import KernelCache
from runners._records import DecodeStepRecord, PrefillRecord


class Bf16NpuRunner:
    """Adapter over a model's bf16 production NPU prefill + decode functions."""

    name = "npu_bf16"

    # --- supplied by the subclass -------------------------------------------
    this_dir = None
    eps = None
    generate_rope_lut = None
    compile_prefill_kernels = None
    compile_decode_kernels = None
    prepare_runtime = None
    run_npu_prefill = None
    run_prefill_block = None
    run_npu_decode_step = None
    rms_norm = None
    # ------------------------------------------------------------------------

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

        self.rope_lut_bf16 = self.generate_rope_lut(
            config=config, seq_len=max_seq
        ).astype(bfloat16)

        # Reuse this model's build_peano kernel cache -- the same dirs `make run`
        # / `make profile` compile into -- so verify/diagnosis and run/profile
        # share one per-model cache (no recompile between commands). The path is
        # absolute (anchored to the subclass's this_dir) so it stays per-model and
        # never contaminates another model's cache regardless of CWD.
        _cache_root = self.this_dir / "build_peano"
        self.prefill_cache = KernelCache(
            str(_cache_root / "prefill_kernel_cache"), verbose=False
        )
        self.compile_prefill_kernels(
            self.prefill_cache,
            config,
            seq_len=max_seq,
            cpu_attn=self.cpu_attn,
        )
        self.decode_cache = KernelCache(
            str(_cache_root / "decode_kernel_cache"), verbose=False
        )
        self.compile_decode_kernels(self.decode_cache, config)

        self.prepare_runtime(
            self.prefill_cache,
            self.decode_cache,
            weights,
            config,
            max_seq,
            self.rope_lut_bf16,
        )

        # Repopulated by prefill(); read by decode_step() within the same
        # verify run.
        self.k_cache = None
        self.v_cache = None

    def prefill(self, prompt_tokens: np.ndarray) -> PrefillRecord:
        # Mirror production's eos-pad-to-max_seq before run_npu_prefill so
        # the verify path hits the same kernel shape make run does.
        eos = self._tokenizer.eos_token_id
        if len(prompt_tokens) < self.max_seq:
            padded = list(prompt_tokens) + [eos] * (self.max_seq - len(prompt_tokens))
        else:
            padded = list(prompt_tokens)[: self.max_seq]
        prefill_token, logits_row, k_cache, v_cache, prompt_len = self.run_npu_prefill(
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

        # Diagnosis-only: re-run the prefill layer loop to capture per-layer
        # ffn_out + final post-norm hidden state. ~3-5s extra; diagnosis is
        # single-prompt so the overhead doesn't matter.
        cfg = self.config
        if len(prompt_tokens) < self.max_seq:
            pad = np.zeros(self.max_seq - len(prompt_tokens), dtype=prompt_tokens.dtype)
            padded_diag = np.concatenate([prompt_tokens, pad])
        else:
            padded_diag = prompt_tokens[: self.max_seq]
        embed = self.weights.embed_table[padded_diag].astype(np.float32)
        x = embed.astype(bfloat16)
        layer_intermediates: list[dict[str, np.ndarray]] = []
        for li in range(cfg.n_layers):
            x, ints = self.run_prefill_block(
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
        x_full_normed = self.rms_norm(x_full_f32, self.weights.final_norm, eps=self.eps)

        return PrefillRecord(
            layer_intermediates=layer_intermediates,
            final_hidden_normed=x_full_normed.astype(np.float32),
            logits_at_pred=logits_row,
            top1_token=prefill_token,
        )

    def decode_step(self, input_token: int, current_pos: int) -> DecodeStepRecord:
        x = self.weights.embed_table[input_token].astype(bfloat16)
        next_token, logits = self.run_npu_decode_step(
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
