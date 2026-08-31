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

Measured on NPU2 (AIE2P), `make profile N_TOKENS=32`, 2026-06-28.

| Phase | Measured | Notes |
|-------|----------|-------|
| Prefill / TTFT (2048 tokens) | **1.52 s wall** | head_dim=128 → host head-first FA seq↔head transpose included in wall; NPU-kernel time is lower (~1.29 s) |
| Decode / TPOT (steady-state) | **11.7 tok/s** | 28 layers, NPU-compute-bound; only cheap single-token glue stays on host |

The table above is the 2026-06 measurement of the bf16 decode path and predates the int4
decode default. The int4-default and bf16-arm re-measurements live in the pre-port study
record (tag `pre-port-20260829`, devq 699/707) and are not restated here; this tree's numbers
come from `make profile` (Turbo pmode verified first).

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

# Top-k token-level correctness gate. Default = W4 decode vs the dequant-
# PATCHED HF oracle: NPU drift only, NOT quantization error — that bar is
# `make verify-quant-bar` (2 prompts × 32 greedy tokens, k=5)
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

On the pre-port study tree all three arms passed 2/2 prompts across two independent lit runs
(tag `pre-port-20260829`, devq 652 and 662; the quantization bar's first divergences were at
steps 16 and 2, mutual ranks ≤ 3). Status on THIS tree is established by re-running
`run_npu2_verify.lit`.

**What none of the three prove**: per-element numerics inside the int4 cascade. The token-set
criterion stops at the first divergence, so a numerical regression that preserves the top-5
sets passes. That bound is queue item 18's per-stage SHIP gate (`reexec_w4_qwen_gate.py`,
pre-port study tree, devq 621) — each stage read back and checked at its own kernel family's
published `rtol`/`atol`; a release-time gate for a change to the int4 cascade builder or
`mv_int4_bf16.cc`, not a per-commit one.

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
