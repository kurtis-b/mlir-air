# Qwen3-0.6B BF16 Inference — Architecture

Companion to [README.md](README.md). This doc describes the per-layer kernel
chain (which ELF runs each op, what is fused inside it, what stays on CPU and
why) and the runtime flow. Built kernel-first on the shared LLM infra
(`../shared/`, `../verify/`) and the `../llama32_1b/` exemplar.

## Model Config

28 layers, emb_dim=1024, n_heads=16, head_dim=128, n_kv_heads=8 (GQA group=2),
q_dim=2048, kv_dim=1024, hidden_dim=3072, vocab_size=151936, BF16,
rope_theta=1000000, eps=1e-6, tied embeddings, **per-head QK-norm** (no bias).

Topology: **Qwen3, O+FFN fused** (small aligned hidden=3072). head_dim=128 →
head-first FlashAttention.

## Per-Layer Kernel Sequence

**Prefill — 3 NPU ELFs/layer:**

```
x ─[NPU elf:rms_qkv_qknorm_rope]   FUSED, 1 ELF, 9 launches (8 ops + the Q GEMM's fused-cast launch)
      { RMSNorm + Q/K/V GEMM + QK-norm(Q) + QK-norm(K) + RoPE-Q + RoPE-K }
      QK-norm = per-head RMSNorm over head_dim=128, eps=1e-6
      → q_roped[seq,2048], k_roped[seq,1024], v[seq,1024]
  ─[NPU elf:flash_attn]   npu_fa_headfirst (head-first, hd=128)
      (HOST) seq→head transpose → NPU FA → (HOST) head→seq transpose → attn_out[seq,2048]
  ─[NPU elf:o_ffn_qwen]   FUSED, 1 ELF
      { O GEMM + Add + RMSNorm + Gate + Up + SwiGLU + Down + Add } → layer_out[seq,1024]
once: (HOST) final RMSNorm → [NPU elf:lm_head_gemv] (10 partitions: 9 × 16384 + 4480 = vocab 151936, m_input 8; was 19 × 8192 before 2026-08-21)
```

**Decode — 2 NPU ELFs/layer (+ lm_head once/token).** `[2026-08-26]` The
O+FFN ELF is the **int4 cascade by default** (doc 56 H2b, queue item 24);
everything else in this sequence is bf16 under both precisions:

```
x ─[NPU elf:rms_qkv_qknorm_rope_gemv2]   FUSED, 1 ELF, 2 launches (8 before 2026-08-21, 4 before 2026-08-23)
      (RMSNorm launch writes the head of the packed B = [normed | RoPE LUT | q_norm | k_norm];
       the GEMV launch is the head-aligned mv_heads.cc kernel: each column owns 4 whole heads,
       accumulates a head over 16 iterations in L1 and runs QK-norm + RoPE in-core on the last
       chunk; chunk TAG / Q-K-V KIND ride in a 64-element row padding of the static weight,
       the head goes out through per-iteration slots the host gathers. QWEN3_RMS_QKV_LAUNCHES=4
       keeps the form below for A/B. Stage 0.62 -> 0.49 ms/layer, 13.0 -> 13.4-13.5 tok/s.)
x ─[NPU elf:rms_qkv_qknorm_rope_gemv4]   FUSED, 1 ELF, 4 launches (was 8 before 2026-08-21)
      { RMSNorm | ONE GEMV over [wq;wk;wv] | ONE per-row-weighted QK-norm over Q|K | ONE RoPE over Q|K }
      (RoPE LUT per-position, tiled over the 24 Q|K rows, NOT a static BO; each air.launch
       boundary costs ~107 us, so 8 -> 4 took the stage 1.03 -> 0.62 ms/layer, bit-identical)
  (HOST) KV-cache write → (HOST) decode_attention_cpu (single-token GQA over KV cache)
  ─[NPU elf:o_gemv_ffn_int4]   FUSED cascade, 1 ELF, 3 launches -- THE DEFAULT since 2026-08-26
      { O GEMV + Add + RMSNorm + Gate/Up cascade + SwiGLU + Down } → layer_out[1024]
      (int4: wo / gate|up (nibble-row-interleaved) / wdown are RTN asym uint4 gs=128, packed by
       `w4_decode_pack.py`, dequanted `(q - z) * s` IN-KERNEL by mv_int4_bf16.cc at DIM_K=1024
       (K_div 2/1/3). SAME launch structure and same 15-arg ABI as the bf16 form below -- the
       whole saving is weight bytes, 6.04 MB packed vs 24.1 MB bf16 per layer. QWEN3_W4_DECODE=0
       selects `o_gemv_ffn`, the bf16 cascade, for A/B; the QKV stage and the LM head are bf16
       under BOTH (priced negatives, doc 56 H2b).)
once: (HOST) embed/final RMSNorm → [NPU elf:lm_head_gemv] (10 partitions, 9 × 16384 + 4480)
```

The fused 2-ELF decode cascade is reachable here because emb=1024 (< 2560) and
hidden=3072 is divisible by 512 — the two limits that wall the bigger models off
this lean form (see NPU vs CPU below).

## NPU vs CPU Mapping

**On NPU (all heavy compute):** every GEMM / GEMV (Q/K/V, O, Gate, Up, Down,
LM-head), RMSNorm, **per-head QK-norm**, RoPE, prefill FlashAttention, SwiGLU
(fused inside `o_ffn_qwen` / `o_gemv_ffn`). Prefill folds Q/K/V + QK-norm + RoPE
into ONE ELF, and O + residual + FFN-norm + Gate + Up + SwiGLU + Down into a
second; attention is the third.

**On CPU (cheap glue + one transpose, evidence-backed):**
- **Head-first FA seq↔head transpose** (prefill, hd=128): the BF16 DMA stride-1
  requirement (sub-32b types) + the seq-first `dk_chunks>1` upstream FA bug block
  an on-device transpose. Would need an upstream FA fix.
- **Decode attention** (single token): NPU FA launch overhead > compute for one
  query row.
- **KV-cache write / embed lookup / final RMSNorm**: single-row dispatch >
  compute.

## Runtime Flow

```
build_session → prepare_runtime()   ← one-time, OUTSIDE timed region
  · load weights, transpose decode GEMV weights, tag per-layer index
  · preload_prefill_weights (warm-up XRT call per prefill ELF → static weight BOs)
  · preload decode + LM-head BOs
  ↓
run_once():  prefill (28 layers × 3 ELFs + final RMSNorm + LM-head)   ← TTFT clock
  ↓
generate() decode loop:  per token 28 layers × 2 ELFs + LM-head GEMV  ← TPOT clock
```

`static_input_indices` + per-layer `bo_key` make the timed kernels skip every
weight host→device write (all weights land in `prepare_runtime`). The RoPE LUT
is position-dependent → deliberately NON-static.

## Key Design Patterns / Deltas

- **Per-head QK-norm fused into `rms_qkv_qknorm_rope`** — the Qwen3 delta. A
  nonlinear per-head RMSNorm (over head_dim=128, eps=1e-6) sits between the Q/K/V
  GEMM and RoPE; it is an NPU slice fused into the attention-input ELF.
- **head_dim=128 → head-first FlashAttention** + host seq↔head transposes
  (`../shared/infra/fa_headfirst.py`).
- **Decoupled q/kv dims** (q_dim=2048 ≠ emb=1024, kv_dim=1024) → non-square O
  projection (2048→1024).
- **O+FFN fused into one ELF** (hidden=3072 small + aligned).
- **Decode O+FFN fully on NPU** via the fused O+FFN cascade (only with
  qwen3_1_7b does decode reach the lean 2-ELF form). `[2026-08-26]` that
  cascade is **int4 by default** (`o_gemv_ffn_int4`; `QWEN3_W4_DECODE=0`
  selects the bf16 `o_gemv_ffn`), which is where this model's decode
  throughput comes from -- same 3 launches, a quarter of the weight bytes.
- **Multi-launch ELF + text-based MLIR stitching** (shared infra): multiple
  `air.launch` ops → one `xrt.run()`; intermediates flow through DDR with no CPU
  round-trip. Half-split RoPE LUT `[cos..., sin...]` matches HF's
  `(d[i], d[i+head_dim/2])` rotation convention.
