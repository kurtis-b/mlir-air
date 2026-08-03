# 11 — Goal 1: SOTA Models via Sliding-Window Attention

Unlock the architecture class that every current deployment gate rejects: sliding-window /
local-global attention, as used by Gemma 3 and Mistral-style models.

`[Codex]` **This is a new model deployment plus attention and KV-cache infrastructure work, not
a mask tweak.** Scope it accordingly.

## Why this axis

MLIR-AIR ships ten decoder-only LLMs, all full-causal-attention, all ≤4B parameters:
Llama-3.2-1B/3B, SmolLM2-1.7B, Qwen2.5-0.5B/1.5B/3B, Qwen3-0.6B/1.7B/4B. The architectural axes
they exercise are attention norm/bias, head dimension, and hidden size — not attention *span*.

Sliding-window is the cheapest next capability because there is in-flight work to build on, and
because it is the gating architecture for a large share of current open-weight models.

## Prior art: the `exper/gemma3-dataflow` branch

Roughly 15 commits of Gemma 3 NPU bring-up, using "runlist" terminology already, plus `q4nx`
quantized decode projections.

**But note its shape.** It is a **182-file parallel tree** at `programming_examples/gemma3/`
with:

- its own `gemma3/core/` Python package (`artifacts.py`, `common.py`, `config.py`, …)
- its own `aie_kernels/` — `q4nx.cc`, `q4nx_opt.cc`, `flow_attention.cc`,
  `flow_attention_opt.cc`, `flow_attention_stats.cc`, `flow_attention_stats_merge.cc`,
  `fused_dqp.cc`, `fused_dqp_opt.cc`
- its own `docs/` (`kernels.md`, `npu_runtime_loop.md`, `results.md`)
- `data/paper_targets.json`

It does **not** follow the `programming_examples/llms/<model>/` per-model contract
(`<model>_{weights,cpu_helpers,prefill,decode,inference}.py`, `verify_adapter.py`, `Makefile`,
`ARCHITECTURE.md`, three `.lit` files).

**Reconciling it into `llms/` versus keeping it parallel is the first decision of this goal.**
Reconciling gets it the shared verify subsystem, the perf pipeline, and the nightly CI for free;
keeping it parallel avoids a large refactor of working code. Decide explicitly rather than by
default.

## The architecture gates

`[Codex]` **Update every gate, not just one.** There are at least two, in two mirrored trees:

1. `deploy-new-llm/SKILL.md` Step 2 rejects a model when `sliding_window` is set in the config
   **and** `use_sliding_window=true`.
2. `phase-0-build-cpu-reference/SKILL.md` Step 2 independently enforces an **allowlist** of
   architectures (`["LlamaForCausalLM", "MistralForCausalLM", ...]`), so `Gemma3ForCausalLM`
   must be added there regardless of the sliding-window flag.

Both `.claude/skills/` and the `.codex/skills/` mirror must change. The skills are byte-near
identical copies; keep them in sync.

While editing them, fix the pre-existing staleness noted in
[13-verification-and-acceptance.md](13-verification-and-acceptance.md): all 15 `SKILL.md` files
still reference `programming_examples/llms/llama_kernel_builder/`, renamed to `llms/shared/` in
commit `2f20c2fa`. `deploy-new-llm` Step 3 literally `test -d`'s that path and always reports
MISSING.

## The real work: attention and KV cache

`[Codex]` The current implementations assume full causal attention in several places.

**Prefill FlashAttention** — `flash_attention/kernel_fusion_based/attn_npu2.py` assumes full
causal dimensions in several paths. Windowing needs:

- banded masking, not just triangular
- absolute-position handling and window offsets
- correctness at the window boundary, including the first window

**Decode attention** — `attention_decode/attn_decode_npu2.py` uses a growing sequence-length KV
layout. Windowed decode needs:

- eviction or ring-buffer indexing so the cache stops growing at the window size
- RoPE position correctness under that indexing — positions are absolute, cache slots are not
- equivalence with the prefill path under the same windowing

**Layer configuration** — alternating local/global layers as a first-class model axis, not a
per-model special case. Gemma 3 interleaves them on a fixed pattern.

## Work items

1. Decide the landing shape for `exper/gemma3-dataflow` (reconcile into `llms/` versus keep
   parallel), then land it on `main`.
2. Update both architecture gates in both skill trees; fix the stale
   `llama_kernel_builder` references while there.
3. Add banded/windowed masking with absolute-position and window-offset handling to prefill
   FlashAttention.
4. Add windowed KV cache indexing (eviction or ring buffer) to decode attention, with RoPE
   position correctness.
5. Make alternating local/global layer configuration a first-class axis.
6. Register windowed-FA shapes in `kernel_registry/details/FlashAttention_bf16.md`.
7. Add Gemma-3-1B/4B to `programming_examples/llms/hf_models.txt` — it feeds
   `.github/workflows/downloadLLMWeights.yml`, which pre-seeds the perf runner's HF cache so the
   nightly can run with `HF_HUB_OFFLINE=1`.
8. Measure the new model across all four execution modes via the Phase F harness.

## Gate

`make verify` exits 0 — **with prompts that cross the window boundary** and exercise both local
and global layers.

`[Codex]` This qualification is the whole point. The existing gate is a top-5 token-set
inclusion check over 32 decoded tokens from short prompts; a short prompt never reaches the
window edge, so a pass would prove nothing about windowing. Add long-prompt fixtures to
`verify/prompts/` specifically for this.

## Risks

- The gemma3 branch's parallel structure means either a large reconciliation refactor or a
  permanent second convention in `programming_examples/`.
- Windowed decode interacts with the KV-cache layout that Phase B's BO pooling also touches.
  Sequence these carefully.
- `mha_out_proj` (Phase C) is built on the current FlashAttention behaviour; changing that
  behaviour here may require revisiting it.
