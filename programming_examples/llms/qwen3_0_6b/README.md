# Qwen3-0.6B Inference on AMD NPU2 (MLIR-AIR)

End-to-end Qwen3-0.6B (0.6B parameter) inference running on AMD NPU2
(AIE2P) hardware via MLIR-AIR. Supports both prefill (seq_len=2048) and
autoregressive decode with a KV cache. Built kernel-first on the shared LLM
infrastructure (`../shared/`, `../verify/`) and the `../llama32_1b/` reference
exemplar; the Qwen3-specific deltas (per-head QK-norm, head_dim=128, decoupled
q/kv dims, eps=1e-6, vocab=151936) are handled inside the per-layer block
runners and fused onto the NPU.

## Decode precision: int4 by default `[2026-08-26]`

**The decode O+FFN cascade runs int4 (`w4_decode`) by default.** Prefill, the decode QKV
stage and the LM head stay bf16 under both settings; `QWEN3_W4_DECODE=0` selects the bf16
decode for A/B, and that env is the only way to name the axis (doc 56 H2b, queue items 18
and 24).

**The default therefore carries quantization error.** The O, gate, up and down matrices are
round-to-nearest asymmetric uint4 at group size 128, fake-quantized from the bf16 checkpoint
by `w4_decode_pack.py`; prefill and the verify oracle then compute on the DEQUANTIZED copy, so
prefill, decode and the reference are numerically one model. What that costs in tokens is
measured, not assumed — see Verification below. To get bf16 back:
`QWEN3_W4_DECODE=0 make run` (likewise `profile`, `verify`, `compile`). A build tree compiled
before the flip is refused by name, with the recompile command in the message.

## Performance

Two clocks, never mixed (operator rule 2026-08-22). This table is `make profile`'s.

Measured on NPU2 (AIE2P), `make profile N_TOKENS=32`, 2026-08-26, **Turbo recorded**,
one session (devq 699 for the int4 arm, an A-B-A of four 10-launch-head rounds against
three 3-launch-head ones; devq 707 for the bf16 arm; devq 663 for the pre-item-28 numbers).

| Phase | Measured | Notes |
|-------|----------|-------|
| Prefill / TTFT (2048 tokens) | **1.48–1.50 s** | unchanged by the precision — prefill is bf16 GEMMs under both plans (bf16 arm 1.49 s); head_dim=128 → host head-first FA seq↔head transpose included in wall |
| Decode / TPOT, **int4 default** | **18.08–18.28 tok/s** (54.7–55.3 ms/token) | 28 layers; per token: 28 × (`rms_qkv_qknorm_rope_gemv2` 0.49 + `o_gemv_ffn_int4` ~1.07 + host attention 0.11 ms) + `lm_head_gemv` **6.48 ms** |
| Decode / TPOT, bf16 arm (`QWEN3_W4_DECODE=0`) | 14.03–14.23 tok/s (70.3–71.3 ms/token) | the same `lm_head_gemv` (6.47–6.50 ms). Its own before/after is NOT controlled — `o_gemv_ffn.elf` was recompiled between the two measurements — so read the head, not the token; 13.7–13.9 on 08-26 before item 28, 13.0 on 08-21, 11.7 in 2026-06 |

`[2026-08-26]` **queue item 28: the LM head covers the vocab in 3 launches, not 10.**
The activation broadcast's BD repeat cap scales with the herd's core-row count, so at 4 rows a
partition carries 65536 rows instead of 16384 and the head is `2 × 65536 + 20864`. **A-B-A in
one session: `lm_head_gemv` 7.58 → 6.48 ms (−14.5 %), decode 17.68 → 18.10 tok/s (+2.4 %),
56.58 → 55.25 ms/token** (devq 699; `make verify` PASS on both precisions, devq 703;
`make check-lm-head-reexec` 7/7 at an odd LOAD_PDI count). The **rows themselves** are worth
−0.4 % (devq 688) — at 41.1 GB/s end to end this head was already above item 27's harness at
the same 8-core geometry, so there was little rate to buy — so the win is the
seven launch boundaries the bigger partitions remove — though only part of that is separable:
holding rows at 2 and taking launches 10 → 5 is a clean control (147 µs per boundary), while the
5 → 3 step moves rows too and is not decomposed. `QWEN3_LM_HERD_ROWS=1` is the A/B arm and
reproduces the pre-item-28 head byte for byte. Evidence:
`transformer_layer/results/item28-land-herd-rows-20260826/`.

**+27 % on decode's median (17.46 vs 13.73 tok/s), at an unchanged launch structure** — the
int4 cascade is the same 3 launches / 1 submission as the bf16 one, so the whole gain is
weight bytes. The study runner's clock (a DIFFERENT clock — do not put its numbers in the
table above) reads 66.4 ms/token at ctx 512 against bf16's 80.3 in the same session; see doc
56 §4's H2b block, queue item 24.

## Model Config

28 layers, emb_dim=1024, n_heads=16, head_dim=128, n_kv_heads=8 (GQA group=2),
q_dim=2048, kv_dim=1024, hidden_dim=3072, vocab_size=151936, BF16,
rope_theta=1000000, eps=1e-6, tied embeddings (lm_head = embed_tokens),
**per-head QK-norm** (RMSNorm over head_dim before RoPE — the key Qwen3 delta).

## Prerequisites

1. **MLIR-AIR base environment** — AMD NPU2 hardware, Peano compiler, the
   project's standard env: `source utils/env_setup.sh ...`

2. **Extra Python packages** (on top of the base):
   ```bash
   pip install -r requirements.txt
   ```
   Installs `safetensors`, `huggingface_hub`, `transformers`, and `torch`
   (used by `make verify` for the HuggingFace bf16 reference comparison).

3. **HuggingFace model access** (one-time):
   - Qwen3-0.6B is openly licensed: https://huggingface.co/Qwen/Qwen3-0.6B
   - Weights are auto-downloaded on the first `make run` and cached under
     `~/.cache/huggingface/hub/`.

## Quick Start

```bash
# One-time: compile all kernels (cached to disk)
make compile

# Run inference (instruct model by default; up to 1000 tokens, stops early on EOT)
make run

# Custom prompt / token budget
make run PROMPT="How does photosynthesis work?" N_TOKENS=64

# Run with profiling breakdown (per-kernel + per-phase)
make profile

# Top-k token-level correctness gate (NPU bf16 vs HF transformers bf16,
# 2 prompts × 32 greedy tokens, k=5) — the production-readiness gate
make verify

# Per-layer cosine diagnosis lens (informational, single prompt)
make diagnosis
```

## Verification — and what the gate does and does not prove

`make verify` is the PASS/FAIL gate: it greedily decodes 32 tokens on the NPU and on HF
transformers (bf16) for each prompt and checks, at the first position where the two disagree,
that each side's token is in the other's top-5 set.

Because the default is quantized, that is **two** questions, and this model gates both. The
standing test `run_npu2_verify.lit` runs three arms, both precisions pinned explicitly:

| Arm | Command | What it proves | Oracle |
|---|---|---|---|
| 1 | `QWEN3_W4_DECODE=0 make verify` | the bf16 A/B path still works | plain HF bf16 |
| 2 | `make verify` (the default) | **NPU drift**: the int4 cascade computes the quantized model correctly | HF patched with the same dequantized O+FFN weights |
| 3 | `make verify-quant-bar` | **the quantization bar**: the int4 default's tokens are still top-5-included in the REAL checkpoint's | plain HF bf16, unpatched |

Arm 2 is the one `make verify` runs, and by construction it **cannot see quantization error** —
both sides carry it. That is deliberate: it isolates NPU drift. Arm 3 is the other half. It
re-judges the same NPU capture against the unpatched checkpoint at the same k=5 / 32-token bar
every bf16 default in this tree meets, and it needs no new flag (its capture phase runs at
`QWEN3_W4_DECODE=1`, its compare phase at `0`, and it is the `0` that makes the oracle plain).

Current status: **PASS on all three arms, 2/2 prompts each, reproduced identically across two independent lit runs** (devq 652 and 662). The quantization bar's own reading, which is the number worth knowing: against the unpatched bf16 checkpoint the int4 default's greedy generation first differs at **step 16** on one prompt and **step 2** on the other, and at both points each side's token is inside the other's top-5 (ranks 2/2 and 3/2). That is what RTN int4 on the O+FFN mass costs at 32 tokens — stated, not implied.

**What none of the three prove**: per-element numerics inside the int4 cascade. The token-set
criterion stops at the first divergence, so a numerical regression that preserves the top-5
sets passes. That bound is queue item 18's per-stage SHIP gate
(`transformer_layer/results/item18-h2b-20260826/reexec_w4_qwen_gate.py`, devq 621) — each
stage read back and checked at its own kernel family's published `rtol`/`atol`. It is a
release-time gate for a change to the int4 cascade builder or `mv_int4_bf16.cc` (~8 min, a
bespoke harness), not a per-commit one.

## Key Files

| File | Purpose |
|------|---------|
| `qwen3_0_6b_inference.py` | Unified prefill + decode driver (`prepare_runtime` does all one-time init outside the timed region) |
| `qwen3_0_6b_prefill.py` | Prefill kernel builders + `run_transformer_block_qwen3` + `preload_prefill_weights` |
| `qwen3_0_6b_decode.py` | Decode kernel builders + `run_decode_block` (KV cache) |
| `qwen3_0_6b_weights.py` | Weight loading from HuggingFace safetensors (incl. q_norm/k_norm, tied lm_head) |
| `qwen3_0_6b_cpu_helpers.py` | NumPy helpers shared by production + verify: `rms_norm`, `qk_norm_per_head` (Qwen3 delta), `attention_reference`, `softmax` |
| `verify_adapter.py` | Hooks this model's prefill/decode into the shared `../verify/` subsystem |
| `w4_decode_pack.py` | The int4 decode path's ONE owner: the `QWEN3_W4_DECODE` flag and its default, the RTN uint4 gs=128 quantization + packing, the dequantized-copy substitution, and the `quant_*` contract |
| `Makefile` | compile / run / profile / chat / verify / verify-full / verify-quant-bar / diagnosis / clean / clean-build |
| `ARCHITECTURE.md` | Per-layer kernel sequence, NPU/CPU mapping, runtime flow, deltas |
