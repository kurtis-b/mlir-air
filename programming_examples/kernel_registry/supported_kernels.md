<!---//===- supported_kernels.md ------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//-->

# Supported Kernels Registry — LLM Deployment on NPU2

High-level index of the leaf kernels validated for decoder-only LLM deployment on AMD NPU2 (Strix, AIE2P): which kernels are covered, which shapes have been tested, and the best measured performance. Per-kernel detail (datapath, tunable parameters, tolerances, how to reproduce) lives in `details/<KERNEL>.md`.

This is **documentation, not executable code** — it records results produced by the `programming_examples/` kernels, run on real NPU2. See [`README.md`](README.md) for scope and methodology.

**Status legend**: ✅ verified on real NPU2, accuracy in line with the bf16 standard · ⚠️ verified on real NPU2 but with a documented precision/coverage caveat · ❌ broken/missing

> **Scope**: currently **GEMM**, **GEMV**, **RMSNorm**, **FlashAttention**, **Element-wise Add**, **SiLU-and-Mul**, **RoPE**, **LayerNorm**, **AddNorm**, **QKV Projection**, **FFN** and **MHA + Output Projection** — the registry is built up one verified kernel at a time. The core LLM leaf kernels are now covered; see [`README.md`](README.md) for the roadmap.
>
> The last five arrive with the transformer block of the execution studies and are **correctness-only entries so far**: their throughput columns read `—` because Phase C gates numerics and nothing else. An entry here never carries an estimated number.
>
> QKV Projection, FFN and MHA + Output Projection are **composite** entries — they are built from the rows above rather than from a kernel of their own, and what they add is the launch structure between them. Their `mean_rel_L1` is quoted next to the constituent kernel's for exactly that reason: it is how much the composition costs.

---

## Kernels

| Kernel | Detail | Best measured throughput (NPU2, units per entry) | Status |
|---|---|---|---|
| GEMM (BF16 in, FP32 out) | [`details/GEMM_bf16_in_fp32_out.md`](details/GEMM_bf16_in_fp32_out.md) | **9797 GFLOP/s** (external, 2048×8192×2048, full-chip 8×4) | ✅ |
| GEMM (BF16 in, BF16 out) | [`details/GEMM_bf16_in_bf16_out.md`](details/GEMM_bf16_in_bf16_out.md) | **8898 GFLOP/s** (fused-cast incl. cast, 2048×8192×2048, full-chip 8×4) | ✅ |
| GEMV (BF16) | [`details/GEMV_bf16.md`](details/GEMV_bf16.md) | **32.7 GFLOP/s** (memory-bound, 16384×3072, herd 8/8/8) | ✅ |
| RMSNorm (BF16) | [`details/RMSNorm_bf16.md`](details/RMSNorm_bf16.md) | **24.9 GB/s** (memory-bound, 2048×3072, herd 8) | ✅ |
| FlashAttention (BF16, GQA) | [`details/FlashAttention_bf16.md`](details/FlashAttention_bf16.md) | **1065–1131 GFLOP/s** (2048×2048, dk=64, 32q/8kv causal, full-chip 32 tiles) | ✅ |
| Element-wise Add (BF16) | [`details/EltwiseAdd_bf16.md`](details/EltwiseAdd_bf16.md) | **57.7 GB/s** (memory-bound, N=4194304, herd 8×1) | ✅ |
| SiLU-and-Mul (BF16) | [`details/SiLU_Mul_bf16.md`](details/SiLU_Mul_bf16.md) | **25.1 GB/s** (memory-bound, N=16777216, herd 8×1) | ✅ |
| RoPE (BF16, half-split) | [`details/RoPE_bf16.md`](details/RoPE_bf16.md) | **56.6 GB/s** (memory-bound, 49152×128, herd 8×1) | ✅ |
| LayerNorm (BF16, multi-row) | [`details/LayerNorm_bf16.md`](details/LayerNorm_bf16.md) | — (correctness only; see the scope note) | ✅ |
| AddNorm (BF16) | [`details/AddNorm_bf16.md`](details/AddNorm_bf16.md) | — (correctness only; see the scope note) | ✅ |
| QKV Projection (BF16, fused weight) | [`details/QKVProj_bf16.md`](details/QKVProj_bf16.md) | — (correctness only; see the scope note) | ✅ |
| FFN, GeLU (BF16, staged) | [`details/FFN_bf16.md`](details/FFN_bf16.md) | — (correctness only; see the scope note) | ✅ |
| MHA + Output Projection (BF16, fused) | [`details/MHAOutProj_bf16.md`](details/MHAOutProj_bf16.md) | — (correctness only; see the scope note) | ✅ |

---

## GEMM (f32 out) — tested shapes

`C[M,N] = A[M,K] @ B[K,N]`, shapes written `M×K×N`. **BF16 in, FP32 out** — always FP32-accumulate (no precision knob). GFLOPS is the fastest (external) path; `mean_rel_L1` = `mean|out−ref| / mean|ref|` vs an FP32 reference. Full per-path data, tolerances, and reproduce commands are in [`details/GEMM_bf16_in_fp32_out.md`](details/GEMM_bf16_in_fp32_out.md).

| (M×K×N) | best tile (m/kl2/kl1/n) | external GFLOPS | direct GFLOPS | mean_rel_L1 | Used by | Status |
|---|---|---|---|---|---|---|
| 2048×2048×2048 | 64/512/32/128 | 8508 | 5516 | 9.3e-3 | llama-3.2-1B Q/O proj | ✅ |
| 2048×2048×512 | 64/256/32/128 | 7342 | 4896 | 9.3e-3 | llama-3.2-1B K/V proj | ✅ |
| 2048×2048×8192 | 64/256/32/128 | 8278 | 5582 | 9.3e-3 | llama-3.2-1B Gate/Up proj | ✅ |
| 2048×8192×2048 | 64/256/32/128 | **9797** | 6010 | 9.3e-3 | llama-3.2-1B Down proj | ✅ |
| 512×512×512 | 32/256/32/128 | 1791 | 1536 | 9.3e-3 | K-sweep | ✅ |
| 1024×1024×1024 | 64/256/32/128 | 6256 | 4413 | 9.5e-3 | K-sweep | ✅ |
| 4096×4096×4096 | 64/512/32/128 | 9329 | 5791 | 9.4e-3 | K-sweep | ✅ |

> Measured on NPU2 (RyzenAI-npu4), June 2026. Two code-paths (external / direct-codegen); external is ~1.5–1.7× faster and bit-identical in accuracy to direct — see [`details/GEMM_bf16_in_fp32_out.md`](details/GEMM_bf16_in_fp32_out.md).

---

## GEMM (bf16 out) — tested shapes

`C[M,N] = A[M,K] @ B[K,N]`, **BF16 in, BF16 out** (half the DDR bytes of f32-out). `--high-precision true` (default) keeps FP32-accumulate + a single epilogue cast (`mean_rel_L1 ≈ 9.7e-3`, GPU standard); `false` is direct-codegen with per-L2-tile bf16 truncation (faster, 1.3e-2–1.9e-2). Within high-precision, `--method auto` picks **fused-cast** (`M*K*N ≥ 4e9`) or **drain** (else). GFLOPS for fused-cast includes the cast launch. Full data in [`details/GEMM_bf16_in_bf16_out.md`](details/GEMM_bf16_in_bf16_out.md).

| (M×K×N) | high-prec fused-cast | high-prec drain | low-prec direct | mean_rel_L1 (high / low) | Used by | Status |
|---|---|---|---|---|---|---|
| 2048×2048×2048 | **6215** | 6025 | 5230 | 9.7e-3 / 1.3e-2 | llama-3.2-1B Q/O proj + Qwen3-1.7B Q/O proj (square) + Qwen2.5-3B Q/O proj (square) | ✅ |
| 2048×2048×512 | 4083 | **5626** | 4765 | 9.7e-3 / 1.3e-2 | llama-3.2-1B K/V proj | ✅ |
| 2048×2048×8192 | **6893** | 5784 | 5287 | 9.7e-3 / 1.3e-2 | llama-3.2-1B Gate/Up proj | ✅ |
| 2048×8192×2048 | **8898** | 7234 | 5592 | 9.7e-3 / 1.9e-2 | llama-3.2-1B Down proj | ✅ |
| 512×512×512 | 482 | **1703** | 1750 | 9.7e-3 / 1.0e-2 | K-sweep | ✅ |
| 1024×1024×1024 | 2502 | **4637** | 4456 | 9.9e-3 / 1.1e-2 | K-sweep | ✅ |
| 4096×4096×4096 | **8423** | 7002 | 5509 | 9.9e-3 / 1.5e-2 | K-sweep | ✅ |
| 2048×1024×2048 | **4425** | — | — | 9.9e-3 / — | Qwen3-0.6B Q proj | ✅ |
| 2048×1024×1024 | — | **4980** | — | 9.4e-3 / — | Qwen3-0.6B K/V proj | ✅ |
| 2048×2048×1024 | **5392** | — | — | 9.7e-3 / — | Qwen3-0.6B O proj + Qwen3-1.7B K/V proj | ✅ |
| 2048×1024×3072 | ⚠️ | ⚠️ | **5006** | 9.4e-3 / 1.1e-2 | Qwen3-0.6B Gate/Up proj | ⚠️ |
| 2048×3072×1024 | **6461** | — | — | 9.9e-3 / — | Qwen3-0.6B Down proj | ✅ |
| 2048×896×896 | — (drain m32/n32) | **2516** | — | 9.4e-3 / — | Qwen2.5-0.5B Q/O proj | ✅ |
| 2048×896×128 | — (drain m32/n32) | **1890** | — | 9.4e-3 / — | Qwen2.5-0.5B K/V proj | ✅ |
| 2048×896×4864 | ⚠️ | — | **4320** | — / 1.11e-2 | Qwen2.5-0.5B Gate/Up proj | ⚠️ |
| 2048×4864×896 | **3640** (n32) | — | — | 9.8e-3 / — | Qwen2.5-0.5B Down proj | ✅ |
| 2048×1536×1536 | **4821** | — | — | 9.7e-3 / — | Qwen2.5-1.5B Q/O proj | ✅ |
| 2048×1536×256 | — (drain n64) | **3770** | — | 9.3e-3 / — | Qwen2.5-1.5B K/V proj | ✅ |
| 2048×1536×8960 | ⚠️ | — | **4165** (n64) | — / 1.2e-2 | Qwen2.5-1.5B Gate/Up proj | ⚠️ |
| 2048×8960×1536 | **8804** | — | — | 9.7e-3 / — | Qwen2.5-1.5B Down proj | ✅ |
| 2048×2048×6144 | **6729** | — | — | 9.7e-3 / — | Qwen3-1.7B Gate/Up proj | ✅ |
| 2048×6144×2048 | **8536** | — | — | 9.7e-3 / — | Qwen3-1.7B Down proj | ✅ |
| 2048×2048×256 | — | **4112** (drain m32/n64) | — | 9.3e-3 / — | Qwen2.5-3B K/V proj | ✅ |
| 2048×2048×11008 | ⚠️ | — | **4276** (n64, tile_k_l2=128) | — / 1.28e-2 | Qwen2.5-3B Gate/Up proj | ⚠️ |
| 2048×11008×2048 | **9447** | — | — | 9.8e-3 / — | Qwen2.5-3B Down proj | ✅ |
| 2048×2560×4096 | **fused-cast (m64/k256/n128)** | — | — | max_abs 1.22e-3 | Qwen3-4B Q proj | ✅ |
| 2048×2560×1024 | **fused-cast (m64/k256/n128)** | — | — | max_abs 9.77e-4 | Qwen3-4B K/V proj | ✅ |
| 2048×4096×2560 | **fused-cast (m64/k256/n128)** | — | — | max_abs 9.77e-4 | Qwen3-4B O proj (decoupled) | ✅ |
| 2048×2560×9728 | ⚠️ | — | **direct (m64/k128/n64)** | — / max_abs 2.93e-3 | Qwen3-4B Gate/Up proj | ⚠️ |
| 2048×9728×2560 | **fused-cast (m64/k256/n128)** | — | — | max_abs 4.88e-4 | Qwen3-4B Down proj | ✅ |
| 2048×3072×3072 | **7513** | — | — | 9.9e-3 / — | Llama-3.2-3B Q/O proj (square) | ✅ |
| 2048×3072×8192 | **7601** | — | — | 9.9e-3 / — | Llama-3.2-3B Gate/Up proj | ✅ |
| 2048×8192×3072 | **9092** | — | — | 9.7e-3 / — | Llama-3.2-3B Down proj | ✅ |

> **Qwen3-4B rows — emb=2560 (512-aligned, NOT 1024-aligned), q_dim=4096 decoupled (≠emb), kv_dim=1024, hidden=9728=512·19.** All proj N divisible by 4·TILE_N=512, so stock TILE_N=128 HERD_N=4 places. Q/K/V/O/Down PASS high-precision fused-cast directly (max_abs ≤ 1.22e-3, well within high-prec tolerance), K=2560/4096/9728 all use tile_k_l2=256. O proj is **decoupled** (K=q_dim=4096, N=emb=2560), the largest non-square O in the registry. **2048×2560×9728 (Gate/Up) ⚠️**: high-precision fused-cast FAILS at compile (`aie.dma_bd op Stride exceeds [1:1048576] range` on the f32-out B-tile DMA at N=9728 — same large-N class as Qwen2.5-3B's N=11008); the low-precision `direct` path (tile_k_l2=128, TILE_N=64) PASSES at max_abs 2.93e-3 — same Gate/Up low-prec tier-down as every Qwen sibling. Large-K Down (K=9728) does NOT trigger the bug (only large-N does). Qwen3-4B uses the qwen25_3b 5-ELF un-merge (o_res_norm / gate / up / HOST SwiGLU / down_add).

> **Qwen2.5-3B rows — emb=q_dim=2048 (1024-aligned, square O), hidden=11008=256·43 (NOT 512-aligned).** Q/O proj is square 2048×2048×2048 (reuses the llama Q/O row). K/V is **2048×2048×256** — thin N=256→TILE_N=64, drain TILE_M=32, K=2048 tile_k_l2=256 (differs from Qwen2.5-1.5B K/V only in K=2048 vs 1536). Down is **2048×11008×2048** — N=2048 stock TILE_N=128, K=11008 tile_k_l2=256, fused-cast PASSES high-precision at 9.8e-3. **2048×2048×11008 (Gate/Up) ⚠️**: both high-precision fused-cast AND low-prec direct at `tile_k_l2=256` fail aiecc with `aie.dma_bd op Stride 2818048 exceeds the [1:1048576] range` (stride = tile_k_l2·N); the low-precision `direct` path with **`tile_k_l2=128`** PASSES at 1.28e-2 (`atol=4e-3`) — same Gate/Up low-prec tier-down as every Qwen sibling, root cause here being the DMA stride range (not L1 over-allocation as in 1.5B).

> **Qwen3-1.7B rows — all dims 1024-aligned, square O.** emb=q_dim=2048 → O proj is square 2048×2048×2048 (reuses the llama Q/O row); K/V 2048×2048×1024 reuses the Qwen3-0.6B O-proj row. The two new shapes are Gate/Up **2048×2048×6144** (N=6144=512·12, stock TILE_N=128 HERD_N=4; `tile_k_l2=256` — `512` BD-exhausts at this N) and Down **2048×6144×2048** (K=6144, `tile_k_l2=256`). Both MKN=2.6e10 ≥ 4e9 → fused-cast, and both PASS high-precision directly at 9.7e-3 (no near-zero atol artifact, unlike the smaller-Qwen Gate/Up shapes) — no low-precision tier needed.

> **Qwen2.5-1.5B rows — 1536 is 512-aligned.** emb=q_dim=1536=512·3 is divisible by the default `4·TILE_N=512`, so **Q/O/Down (N=1536) place at the stock `TILE_N=128 HERD_N=4`** — no TILE_N shrink (contrast Qwen2.5-0.5B's 896). Only thin **K/V (N=256 → TILE_N=64, drain TILE_M=32)** and wide **Gate/Up (N=8960 → TILE_N=64)** drop below 128. K=1536 uses `tile_k_l2=256` (1536/256=6), K=8960 uses `tile_k_l2=256` (8960/256=35). **2048×1536×8960 (Gate/Up) ⚠️**: high-precision fused-cast (TILE_M=64 TILE_N=64) over-allocates L1 → compile fail; the low-precision `direct` path PASSES but needs `tile_k_l2=128` (tile_k_l2=256 also compile-fails at this N), at 1.2e-2 — same Gate/Up tier-down as the smaller Qwen siblings.

> **Qwen2.5-0.5B rows — non-512-aligned N.** Qwen2.5's projection widths (896, 128, 4864) are not divisible by the default `4·TILE_N=512`, and `HERD_N=1` (e.g. `TILE_N=128` for N=896) **fails at runtime** (`qds_device::wait() unexpected command state` — the fused-cast/drain paths assume the 8×4 array). The working recipe keeps `HERD_N=4` and shrinks `TILE_N` so `4·TILE_N | N`: **N=896/128 → TILE_N=32**, **N=4864 → TILE_N=64**. K=896 uses `tile_k_l2=128` (896/128=7), K=4864 uses `tile_k_l2=256`. The thin shapes need `METHOD=drain` (`tile_m=32`; `tile_m=64` over-allocates L1). No padding was required — every real shape placed and PASSED. **2048×896×4864 (Gate/Up) ⚠️**: high-precision fused-cast computes the in-tier result (9.4e-3) but the harness gate trips on 2 near-zero-reference elements (abs_err ≈ 1.6–1.9e-3 > high-prec `atol=1.5e-3`); PASSES on the low-precision `direct` path (`atol=4e-3`, 1.11e-2) — same artifact as Qwen3-0.6B Gate/Up.

> GFLOPS, all PASS. **Bold** = faster high-precision method (what `auto` picks); the `M*K*N ≥ 4e9` threshold matches the bold winner for all 7 shapes.
> Qwen3-0.6B rows: only the `auto`-selected high-precision method was swept (`—` = the other method not measured for that shape); all `auto` picks PASS at 9.4–9.9e-3. **2048×1024×3072 (Gate/Up) ⚠️**: both high-precision methods compute the in-tier result (mean_rel_L1 = 9.4e-3) but the harness element-wise gate trips on a single near-zero-reference output element (abs_err ≈ 1.7e-3 > the high-precision `atol = 1.5e-3`, `rtol·|ref|≈0`); the shape PASSES on the low-precision `direct` path (`atol = 4e-3`, 1.1e-2). Harness tolerance edge, not a datapath failure — see [`details/GEMM_bf16_in_bf16_out.md`](details/GEMM_bf16_in_bf16_out.md). fused-cast is tile_m=64, drain is tile_m=32. The high-precision tier preserves f32-out accuracy (9.3–9.9e-3) via a single cast; low-precision direct degrades with the L2-tile count (`K / tile_k_l2`). See [`details/GEMM_bf16_in_bf16_out.md`](details/GEMM_bf16_in_bf16_out.md).

<!-- BEGIN transformer-layer-sweep baseline_768 -->
### Transformer-layer execution study — `baseline_768` sweep

The projection GEMMs the transformer-layer execution study's case matrix needs, swept
across the full 9-point sequence ladder. Full per-candidate detail, and why the
high-precision `atol` is K-scaled here, in
[`details/GEMM_bf16_in_bf16_out.md`](details/GEMM_bf16_in_bf16_out.md).

**`qkv_proj`** — `K = 768` → `N = 2304`

| seq | (M×K×N) | fused-cast | drain | direct | best tile (m/kl2/kl1/n) (herd) | mean_rel_L1 (high / low) | Status |
|---|---|---|---|---|---|---|---|
| 64 | 64×768×2304 | 446 | **945** | 568 | 32/128/32/96 (2×4) | 9.4e-3 / 1.1e-2 | ✅ |
| 128 | 128×768×2304 | 952 | **1785** | 1150 | 32/256/32/96 (4×4) | 9.4e-3 / 1.1e-2 | ✅ |
| 256 | 256×768×2304 | 1511 | **3043** | 2242 | 32/256/32/96 (8×4) | 9.4e-3 / 1.1e-2 | ✅ |
| 512 | 512×768×2304 | 2146 | **3981** | 4003 | 32/256/32/96 (8×4) | 9.4e-3 / 1.1e-2 | ✅ |
| 1024 | 1024×768×2304 | 3124 | **4209** | 4896 | 32/256/32/96 (8×4) | 9.4e-3 / 1.1e-2 | ✅ |
| 2048 | 2048×768×2304 | 3875 | **4132** | 4580 | 32/256/32/96 (8×4) | 9.4e-3 / 1.1e-2 | ✅ |
| 4096 | 4096×768×2304 | **4226** | 4123 | 5027 | 64/256/32/96 (8×4) | 9.9e-3 / 1.1e-2 | ✅ |
| 8192 | 8192×768×2304 | **4694** | 4477 | 5122 | 64/256/32/96 (8×4) | 9.9e-3 / 1.1e-2 | ✅ |
| 16384 | 16384×768×2304 | **4867** | 4436 | 5180 | 64/256/32/96 (8×4) | 9.9e-3 / 1.1e-2 | ✅ |

**`ffn_up`** — `K = 768` → `N = 3072`

| seq | (M×K×N) | fused-cast | drain | direct | best tile (m/kl2/kl1/n) (herd) | mean_rel_L1 (high / low) | Status |
|---|---|---|---|---|---|---|---|
| 64 | 64×768×3072 | 478 | **1015** | 582 | 32/128/32/128 (2×4) | 9.4e-3 / 1.1e-2 | ✅ |
| 128 | 128×768×3072 | 1093 | **1953** | 1201 | 32/128/32/128 (4×4) | 9.5e-3 / 1.1e-2 | ✅ |
| 256 | 256×768×3072 | 1818 | **3463** | 2334 | 32/256/32/128 (8×4) | 9.4e-3 / 1.1e-2 | ✅ |
| 512 | 512×768×3072 | 2548 | **4280** | 4339 | 32/256/32/128 (8×4) | 9.4e-3 / 1.1e-2 | ✅ |
| 1024 | 1024×768×3072 | 3327 | **4516** | 4867 | 32/256/32/128 (8×4) | 9.4e-3 / 1.1e-2 | ✅ |
| 2048 | 2048×768×3072 | 4431 | **4513** | 5056 | 32/256/32/128 (8×4) | 9.4e-3 / 1.1e-2 | ✅ |
| 4096 | 4096×768×3072 | 4479 | **4689** | 5030 | 32/256/32/128 (8×4) | 9.4e-3 / 1.1e-2 | ✅ |
| 8192 | 8192×768×3072 | **5164** | 4743 | 5110 | 64/128/32/128 (8×4) | 9.9e-3 / 1.1e-2 | ✅ |
| 16384 | 16384×768×3072 | **5362** | 4702 | 5211 | 64/128/32/128 (8×4) | 9.9e-3 / 1.1e-2 | ✅ |

**`ffn_down`** — `K = 3072` → `N = 768`

| seq | (M×K×N) | fused-cast | drain | direct | best tile (m/kl2/kl1/n) (herd) | mean_rel_L1 (high / low) | Status |
|---|---|---|---|---|---|---|---|
| 64 | 64×3072×768 | 544 | **1462** | 648 | 32/256/32/96 (2×4) | 9.5e-3 / 1.4e-2 | ✅ |
| 128 | 128×3072×768 | 1263 | **2752** | 1320 | 32/256/32/96 (4×4) | 9.5e-3 / 1.4e-2 | ✅ |
| 256 | 256×3072×768 | 2099 | **4812** | 2587 | 32/512/32/96 (8×4) | 9.5e-3 / 1.4e-2 | ✅ |
| 512 | 512×3072×768 | 3272 | **5754** | 4957 | 32/512/32/96 (8×4) | 9.4e-3 / 1.4e-2 | ✅ |
| 1024 | 1024×3072×768 | 4730 | **6088** | 5234 | 32/512/32/96 (8×4) | 9.4e-3 / 1.4e-2 | ✅ |
| 2048 | 2048×3072×768 | 5948 | **6000** | 5485 | 32/512/32/96 (8×4) | 9.4e-3 / 1.4e-2 | ✅ |
| 4096 | 4096×3072×768 | **6927** | 6226 | 5604 | 64/512/32/96 (8×4) | 9.9e-3 / 1.4e-2 | ✅ |
| 8192 | 8192×3072×768 | **7510** | 6399 | 5532 | 64/512/32/96 (8×4) | 9.9e-3 / 1.4e-2 | ✅ |
| 16384 | 16384×3072×768 | **7842** | 6268 | 5707 | 64/512/32/96 (8×4) | 9.9e-3 / 1.4e-2 | ✅ |

**`o_proj`** — `K = 768` → `N = 768`

| seq | (M×K×N) | fused-cast | drain | direct | best tile (m/kl2/kl1/n) (herd) | mean_rel_L1 (high / low) | Status |
|---|---|---|---|---|---|---|---|
| 64 | 64×768×768 | 261 | **708** | 477 | 32/128/32/96 (2×4) | 9.4e-3 / 1.1e-2 | ✅ |
| 128 | 128×768×768 | 492 | **1403** | 969 | 32/128/32/96 (4×4) | 9.4e-3 / 1.1e-2 | ✅ |
| 256 | 256×768×768 | 780 | **2539** | 1900 | 32/128/32/96 (8×4) | 9.4e-3 / 1.1e-2 | ✅ |
| 512 | 512×768×768 | 1065 | **3505** | 3450 | 32/256/32/96 (8×4) | 9.4e-3 / 1.1e-2 | ✅ |
| 1024 | 1024×768×768 | 1789 | **4015** | 4384 | 32/256/32/96 (8×4) | 9.5e-3 / 1.1e-2 | ✅ |
| 2048 | 2048×768×768 | 2700 | **4277** | 4610 | 32/256/32/96 (8×4) | 9.5e-3 / 1.1e-2 | ✅ |
| 4096 | 4096×768×768 | 3622 | **4529** | 5047 | 32/256/32/96 (8×4) | 9.4e-3 / 1.1e-2 | ✅ |
| 8192 | 8192×768×768 | 4189 | **4499** | 4827 | 32/256/32/96 (8×4) | 9.4e-3 / 1.1e-2 | ✅ |
| 16384 | 16384×768×768 | **4799** | 4610 | 5254 | 64/256/32/96 (8×4) | 9.9e-3 / 1.1e-2 | ✅ |

<!-- END transformer-layer-sweep baseline_768 -->

<!-- BEGIN transformer-layer-sweep baseline_512 -->
### Transformer-layer execution study — `baseline_512` sweep

The projection GEMMs the transformer-layer execution study's case matrix needs, swept
across the full 9-point sequence ladder. Full per-candidate detail, and why the
high-precision `atol` is K-scaled here, in
[`details/GEMM_bf16_in_bf16_out.md`](details/GEMM_bf16_in_bf16_out.md).

**`qkv_proj`** — `K = 512` → `N = 1536`

| seq | (M×K×N) | fused-cast | drain | direct | best tile (m/kl2/kl1/n) (herd) | mean_rel_L1 (high / low) | Status |
|---|---|---|---|---|---|---|---|
| 64 | 64×512×1536 | 292 | **548** | 405 | 32/512/32/96 (2×4) | 9.2e-3 / 1.0e-2 | ✅ |
| 128 | 128×512×1536 | 545 | **1124** | 956 | 32/256/32/96 (4×4) | 9.3e-3 / 1.0e-2 | ✅ |
| 256 | 256×512×1536 | 876 | **1973** | 1847 | 32/512/32/96 (8×4) | 9.3e-3 / 1.0e-2 | ✅ |
| 512 | 512×512×1536 | 1218 | **2653** | 3287 | 32/256/32/96 (8×4) | 9.3e-3 / 1.0e-2 | ✅ |
| 1024 | 1024×512×1536 | 1929 | **2849** | 3778 | 32/256/32/96 (8×4) | 9.3e-3 / 1.0e-2 | ✅ |
| 2048 | 2048×512×1536 | 2565 | **2871** | 4460 | 32/512/32/96 (8×4) | 9.3e-3 / 1.0e-2 | ✅ |
| 4096 | 4096×512×1536 | **2751** | 2585 | 4620 | 64/256/32/128 (8×4) | 9.7e-3 / 1.0e-2 | ✅ |
| 8192 | 8192×512×1536 | **3377** | 3070 | 4684 | 64/256/32/96 (8×4) | 9.7e-3 / 1.0e-2 | ✅ |
| 16384 | 16384×512×1536 | **3571** | 3059 | 4730 | 64/256/32/96 (8×4) | 9.7e-3 / 1.0e-2 | ✅ |

**`ffn_up`** — `K = 512` → `N = 2048`

| seq | (M×K×N) | fused-cast | drain | direct | best tile (m/kl2/kl1/n) (herd) | mean_rel_L1 (high / low) | Status |
|---|---|---|---|---|---|---|---|
| 64 | 64×512×2048 | 334 | **606** | 423 | 32/256/32/128 (2×4) | 9.3e-3 / 1.0e-2 | ✅ |
| 128 | 128×512×2048 | 639 | **1156** | 955 | 32/256/32/128 (4×4) | 9.3e-3 / 1.0e-2 | ✅ |
| 256 | 256×512×2048 | 1032 | **2180** | 1955 | 32/256/32/128 (8×4) | 9.2e-3 / 1.0e-2 | ✅ |
| 512 | 512×512×2048 | 1472 | **2822** | 3555 | 32/256/32/128 (8×4) | 9.3e-3 / 1.0e-2 | ✅ |
| 1024 | 1024×512×2048 | 2177 | **2961** | 4338 | 32/256/32/128 (8×4) | 9.3e-3 / 1.0e-2 | ✅ |
| 2048 | 2048×512×2048 | 2603 | **2967** | 4579 | 32/512/32/128 (8×4) | 9.3e-3 / 1.0e-2 | ✅ |
| 4096 | 4096×512×2048 | **3305** | 3005 | 4675 | 64/256/32/128 (8×4) | 9.7e-3 / 1.0e-2 | ✅ |
| 8192 | 8192×512×2048 | **3596** | 3124 | 4388 | 64/256/32/128 (8×4) | 9.7e-3 / 1.0e-2 | ✅ |
| 16384 | 16384×512×2048 | **3770** | 3165 | 4599 | 64/256/32/128 (8×4) | 9.7e-3 / 1.0e-2 | ✅ |

**`ffn_down`** — `K = 2048` → `N = 512`

| seq | (M×K×N) | fused-cast | drain | direct | best tile (m/kl2/kl1/n) (herd) | mean_rel_L1 (high / low) | Status |
|---|---|---|---|---|---|---|---|
| 64 | 64×2048×512 | 386 | **857** | 488 | 32/256/32/64 (2×4) | 9.3e-3 / 1.3e-2 | ✅ |
| 128 | 128×2048×512 | 717 | **1657** | 1110 | 32/256/32/64 (4×4) | 9.3e-3 / 1.3e-2 | ✅ |
| 256 | 256×2048×512 | 1205 | **3211** | 2181 | 32/256/32/64 (8×4) | 9.3e-3 / 1.3e-2 | ✅ |
| 512 | 512×2048×512 | 1769 | **3945** | 4126 | 32/256/32/128 (8×4) | 9.3e-3 / 1.3e-2 | ✅ |
| 1024 | 1024×2048×512 | 2781 | **4325** | 4819 | 32/256/32/128 (8×4) | 9.3e-3 / 1.3e-2 | ✅ |
| 4096 | 4096×2048×512 | **4638** | 4392 | 5132 | 64/256/32/64 (8×4) | 9.7e-3 / 1.3e-2 | ✅ |
| 8192 | 8192×2048×512 | **4881** | 4318 | 5218 | 64/512/32/128 (8×4) | 9.7e-3 / 1.3e-2 | ✅ |
| 16384 | 16384×2048×512 | **5632** | 4670 | 5330 | 64/256/32/64 (8×4) | 9.7e-3 / 1.3e-2 | ✅ |

**`o_proj`** — `K = 512` → `N = 512`

| seq | (M×K×N) | fused-cast | drain | direct | best tile (m/kl2/kl1/n) (herd) | mean_rel_L1 (high / low) | Status |
|---|---|---|---|---|---|---|---|
| 64 | 64×512×512 | 141 | **349** | 273 | 32/256/32/128 (2×4) | 9.1e-3 / 1.0e-2 | ✅ |
| 128 | 128×512×512 | 249 | **685** | 610 | 32/256/32/128 (4×4) | 9.3e-3 / 1.0e-2 | ✅ |
| 256 | 256×512×512 | 389 | **1292** | 1215 | 32/256/32/128 (8×4) | 9.3e-3 / 1.0e-2 | ✅ |
| 1024 | 1024×512×512 | 953 | **2300** | 3164 | 32/256/32/64 (8×4) | 9.3e-3 / 1.0e-2 | ✅ |
| 2048 | 2048×512×512 | 1524 | **2590** | 3541 | 32/256/32/64 (8×4) | 9.3e-3 / 1.0e-2 | ✅ |
| 4096 | 4096×512×512 | 1918 | **2654** | 3967 | 32/256/32/64 (8×4) | 9.3e-3 / 1.0e-2 | ✅ |
| 8192 | 8192×512×512 | **2670** | 2201 | 3835 | 64/256/32/64 (8×4) | 9.7e-3 / 1.0e-2 | ✅ |
| 16384 | 16384×512×512 | **3093** | 2727 | 3765 | 64/256/32/64 (8×4) | 9.7e-3 / 1.0e-2 | ✅ |

<!-- END transformer-layer-sweep baseline_512 -->

<!-- BEGIN transformer-layer-sweep baseline_1024 -->
### Transformer-layer execution study — `baseline_1024` sweep

The projection GEMMs the transformer-layer execution study's case matrix needs, swept
across the full 9-point sequence ladder. Full per-candidate detail, and why the
high-precision `atol` is K-scaled here, in
[`details/GEMM_bf16_in_bf16_out.md`](details/GEMM_bf16_in_bf16_out.md).

**`qkv_proj`** — `K = 1024` → `N = 3072`

| seq | (M×K×N) | fused-cast | drain | direct | best tile (m/kl2/kl1/n) (herd) | mean_rel_L1 (high / low) | Status |
|---|---|---|---|---|---|---|---|
| 64 | 64×1024×3072 | 508 | **927** | 520 | 32/256/32/128 (2×4) | 9.5e-3 / 1.1e-2 | ✅ |
| 128 | 128×1024×3072 | 1073 | **1796** | 1234 | 32/256/32/128 (4×4) | 9.4e-3 / 1.1e-2 | ✅ |
| 256 | 256×1024×3072 | 1852 | **3200** | 2403 | 32/512/32/128 (8×4) | 9.4e-3 / 1.1e-2 | ✅ |
| 512 | 512×1024×3072 | 2754 | **3952** | 4506 | 32/256/32/128 (8×4) | 9.4e-3 / 1.1e-2 | ✅ |
| 1024 | 1024×1024×3072 | 3739 | **3977** | 4878 | 32/256/32/128 (8×4) | 9.4e-3 / 1.1e-2 | ✅ |
| 4096 | 4096×1024×3072 | **4709** | 4153 | 4967 | 64/256/32/128 (8×4) | 9.9e-3 / 1.1e-2 | ✅ |
| 8192 | 8192×1024×3072 | **4784** | 4142 | 5194 | 64/256/32/96 (8×4) | 9.9e-3 / 1.1e-2 | ✅ |
| 16384 | 16384×1024×3072 | **5042** | 4180 | 5286 | 64/256/32/128 (8×4) | 9.9e-3 / 1.1e-2 | ✅ |

**`ffn_up`** — `K = 1024` → `N = 4096`

| seq | (M×K×N) | fused-cast | drain | direct | best tile (m/kl2/kl1/n) (herd) | mean_rel_L1 (high / low) | Status |
|---|---|---|---|---|---|---|---|
| 64 | 64×1024×4096 | 538 | **943** | 527 | 32/256/32/128 (2×4) | 9.5e-3 / 1.1e-2 | ✅ |
| 128 | 128×1024×4096 | 1154 | **1847** | 1246 | 32/256/32/128 (4×4) | 9.5e-3 / 1.1e-2 | ✅ |
| 256 | 256×1024×4096 | 1995 | **3411** | 2359 | 32/256/32/128 (8×4) | 9.5e-3 / 1.1e-2 | ✅ |
| 512 | 512×1024×4096 | 3110 | **3992** | 4533 | 32/256/32/128 (8×4) | 9.4e-3 / 1.1e-2 | ✅ |
| 1024 | 1024×1024×4096 | **4003** | 3488 | 4459 | 64/256/32/128 (8×4) | 9.9e-3 / 1.1e-2 | ✅ |
| 2048 | 2048×1024×4096 | **4292** | 3932 | 5118 | 64/512/32/128 (8×4) | 9.9e-3 / 1.1e-2 | ✅ |
| 4096 | 4096×1024×4096 | **4800** | 4126 | 4910 | 64/256/32/128 (8×4) | 9.9e-3 / 1.1e-2 | ✅ |
| 8192 | 8192×1024×4096 | **4966** | 4065 | 5264 | 64/256/32/128 (8×4) | 9.9e-3 / 1.1e-2 | ✅ |
| 16384 | 16384×1024×4096 | **5053** | 4180 | 5222 | 64/256/32/128 (8×4) | 9.9e-3 / 1.1e-2 | ✅ |

**`ffn_down`** — `K = 4096` → `N = 1024`

| seq | (M×K×N) | fused-cast | drain | direct | best tile (m/kl2/kl1/n) (herd) | mean_rel_L1 (high / low) | Status |
|---|---|---|---|---|---|---|---|
| 64 | 64×4096×1024 | 648 | **1239** | 584 | 32/256/32/128 (2×4) | 9.4e-3 / 1.5e-2 | ✅ |
| 128 | 128×4096×1024 | 1318 | **2454** | 1364 | 32/256/32/128 (4×4) | 9.5e-3 / 1.5e-2 | ✅ |
| 256 | 256×4096×1024 | 2439 | **4656** | 2674 | 32/512/32/128 (8×4) | 9.4e-3 / 1.5e-2 | ✅ |
| 512 | 512×4096×1024 | 4081 | **5266** | 5122 | 32/256/32/128 (8×4) | 9.4e-3 / 1.5e-2 | ✅ |
| 1024 | 1024×4096×1024 | 5125 | **5243** | 5553 | 32/256/32/128 (8×4) | 9.4e-3 / 1.5e-2 | ✅ |
| 2048 | 2048×4096×1024 | **5835** | 5433 | 5551 | 64/512/32/128 (8×4) | 9.9e-3 / 1.5e-2 | ✅ |
| 4096 | 4096×4096×1024 | **6379** | 5433 | 5676 | 64/256/32/128 (8×4) | 9.9e-3 / 1.5e-2 | ✅ |
| 8192 | 8192×4096×1024 | **6655** | 5528 | 5735 | 64/256/32/128 (8×4) | 9.9e-3 / 1.5e-2 | ✅ |
| 16384 | 16384×4096×1024 | **6780** | 5563 | 5762 | 64/256/32/128 (8×4) | 9.9e-3 / 1.5e-2 | ✅ |

**`o_proj`** — `K = 1024` → `N = 1024`

| seq | (M×K×N) | fused-cast | drain | direct | best tile (m/kl2/kl1/n) (herd) | mean_rel_L1 (high / low) | Status |
|---|---|---|---|---|---|---|---|
| 64 | 64×1024×1024 | 365 | **745** | 468 | 32/256/32/128 (2×4) | 9.6e-3 / 1.1e-2 | ✅ |
| 128 | 128×1024×1024 | 679 | **1512** | 1101 | 32/256/32/128 (4×4) | 9.5e-3 / 1.1e-2 | ✅ |
| 256 | 256×1024×1024 | 1103 | **2886** | 2130 | 32/256/32/128 (8×4) | 9.5e-3 / 1.1e-2 | ✅ |
| 512 | 512×1024×1024 | 1543 | **3493** | 4058 | 32/512/32/128 (8×4) | 9.4e-3 / 1.1e-2 | ✅ |
| 4096 | 4096×1024×1024 | **4082** | 3766 | 5006 | 64/256/32/128 (8×4) | 9.9e-3 / 1.1e-2 | ✅ |
| 8192 | 8192×1024×1024 | **4570** | 3985 | 5153 | 64/256/32/128 (8×4) | 9.9e-3 / 1.1e-2 | ✅ |
| 16384 | 16384×1024×1024 | **4897** | 4183 | 4911 | 64/256/32/128 (8×4) | 9.9e-3 / 1.1e-2 | ✅ |

<!-- END transformer-layer-sweep baseline_1024 -->

---

## GEMV — tested shapes

`C[M] = A[M,K] @ B[K]`, shapes written `M×K`. The decode-time (batch = 1) projections of llama-3.2-1B. GEMV is **memory-bound** (reads the whole `M×K` matrix for one length-`M` output), so GFLOPS is far below GEMM; the fastest config is `herd_m=8` (all columns) with the largest L2-legal `tile_m`. Full data, tunables, and reproduce commands are in [`details/GEMV_bf16.md`](details/GEMV_bf16.md).

| (M×K) | best tile (herd_m/tile_m/m_input) | GFLOPS | mean_rel_L1 | Used by | Status |
|---|---|---|---|---|---|
| 2048×2048 | 8/8/8 | 25.5 | 1.6e-9 | llama-3.2-1B Q proj + Qwen3-1.7B decode Q/O proj + Qwen2.5-3B decode Q/O proj | ✅ |
| 512×2048 | 8/8/8 | 15.5 | 0.0 | llama-3.2-1B K/V proj | ✅ |
| 8192×2048 | 8/8/8 | 31.5 | 2.7e-8 | coverage | ✅ |
| 2048×8192 | 8/2/2 | 31.0 | 0.0 | coverage | ✅ |
| 16384×2048 | 8/8/8 | **30.6** | 0.0 | llama-3.2-1B LM-head + Qwen3-1.7B LM-head + Qwen2.5-3B LM-head (K=2048 partition datapath) | ✅ |
| 49152×2048 | 8/8/8 | 32.5 | 5.9e-8 | SmolLM2-1.7B LM-head | ✅ |
| 2048×1024 | 8/8/8 | 18.2 | 1.2e-6 | Qwen3-0.6B decode Q proj | ✅ |
| 1024×1024 | 8/8/8 | 14.3 | 0.0 | Qwen3-0.6B decode K/V proj | ✅ |
| 16384×1024 | 8/16/16 | 31.4 | 2.0e-8 | Qwen3-0.6B LM-head (per-partition) | ✅ |
| 896×896 | 8/16/16 | 9.5 | 0.0 | Qwen2.5-0.5B decode Q/O proj | ✅ |
| 128×896 | 8/16/16 | 2.7 | 0.0 | Qwen2.5-0.5B decode K/V proj | ✅ |
| 4864×896 | 8/16/16 | 20.3 | 0.0 | Qwen2.5-0.5B decode Gate/Up proj | ✅ |
| 896×4864 | 8/4/4 | 26.3 | 0.0 | Qwen2.5-0.5B decode Down proj | ✅ |
| 16384×896 | 8/16/16 | 28.5 | 7.2e-12 | Qwen2.5-0.5B LM-head (per-partition) | ✅ |
| 1536×1536 | 8/16/16 | 22.5 | 0.0 | Qwen2.5-1.5B decode Q/O proj | ✅ |
| 256×1536 | 8/16/16 | 7.5 | 0.0 | Qwen2.5-1.5B decode K/V proj | ✅ |
| 8960×1536 | 8/16/16 | 25.0 | 1.7e-9 | Qwen2.5-1.5B decode Gate/Up proj | ✅ |
| 1536×8960 | 8/2/2 | 30.6 | 2.2e-6 | Qwen2.5-1.5B decode Down proj | ✅ |
| 16384×1536 | 8/16/16 | 32.6 | 2.3e-8 | Qwen2.5-1.5B LM-head (per-partition) | ✅ |
| 1024×2048 | 8/8/8 | 21.0 | 0.0 | Qwen3-1.7B decode K/V proj | ✅ |
| 6144×2048 | 8/8/8 | 30.8 | 0.0 | Qwen3-1.7B decode Gate/Up proj | ✅ |
| 2048×6144 | 8/4/4 | 31.4 | 0.0 | Qwen3-1.7B decode Down proj | ✅ |
| 256×2048 | 8/8/8 | 10.1 | 0.0 | Qwen2.5-3B decode K/V proj | ✅ |
| 11008×2048 | 8/8/8 | 31.9 | 7.9e-8 | Qwen2.5-3B decode Gate/Up proj | ✅ |
| 2048×11008 | 8/2/1 | 27.6 | 0.0 | Qwen2.5-3B decode Down proj (K=11008 L1-bound → m_input=1) | ✅ |
| 4096×2560 | 8/8/8 | 30.1 | 7.3e-7 | Qwen3-4B decode Q proj | ✅ |
| 1024×2560 | 8/8/8 | 22.6 | 0.0 | Qwen3-4B decode K/V proj | ✅ |
| 2560×4096 | 8/4/4 | 29.4 | 0.0 | Qwen3-4B decode O proj (decoupled K=4096 → tile_m=4 m_input=4 to fit L2) | ✅ |
| 9728×2560 | 8/8/8 | 32.6 | 0.0 | Qwen3-4B decode Gate/Up proj | ✅ |
| 2560×9728 | 8/2/2 | 31.0 | 2.3e-10 | Qwen3-4B decode Down proj — standalone (model runs this on HOST: stitched-ELF L1 overflow) | ✅ |
| 16384×2560 | 8/8/8 | 30.2 | 4.2e-7 | Qwen3-4B LM-head (per-partition, K=2560) | ✅ |
| 1024×3072 | 8/8/8 | 24.5 | 0.0 | coverage (K=3072) | ✅ |
| 3072×1024 | 8/16/16 | 22.7 | 4.9e-10 | coverage (M=3072, K=1024) | ✅ |
| 3072×3072 | 8/8/8 | 30.4 | 1.8e-9 | coverage (K=3072) | ✅ |
| 3072×8192 | 8/2/2 | 29.4 | 0.0 | coverage (K=8192) | ✅ |
| 8192×3072 | 8/8/8 | 32.2 | 1.1e-7 | coverage (M=8192, K=3072) | ✅ |
| 16384×3072 | 8/8/8 | 32.6 | 3.4e-7 | LM-head coverage (K=3072) | ✅ |

> **Qwen3-4B GEMV.** Decode projections bit-identical (0.0) to the f32 ref. emb=2560 K, q_dim=4096 decoupled. O proj is **decoupled** (M=emb=2560, K=q_dim=4096) — at K=4096 the full `[m_input, K]` A tile constrains L2, so `tile_m=4, m_input=4` (vs the stock 8/8) keeps A=tile_m·herd_m·K·2 ≤ 512 KiB. Down proj is **2560×9728** (M=emb=2560, K=intermediate=9728); the standalone harness places at `8/2/2` (31.0 GFLOPS, 2.3e-10), but in the model it runs on **HOST** (stitched-ELF L1 overflow, same as Qwen2.5-3B's K=11008). LM-head reuses the shared 19-partition vocab=151936 datapath at K=2560 per partition (16384×2560 row, 30.2 GFLOPS).

> **Qwen2.5-3B GEMV.** Decode projections bit-identical (0.0) or ≤7.9e-8 to the f32 ref. Q/O proj is 2048×2048 (reuses the llama Q row); LM-head is K=2048 per-partition (reuses the 16384×2048 datapath row). K=11008 (Down proj) is the most L1-constrained GEMV in the registry — the harness loads the full `[m_input, K]` A tile + `[K]` B vector into L1 (no K-tiling), so at K=11008 even `tile_m=2, m_input=2` (44 KB A-tile) overflows the 64 KB L1; **`tile_m=2, m_input=1` (22 KB A-tile) PASSES**. (`tile_m=1` is rejected by the 4-byte transfer-length check.)

> **Qwen3-1.7B GEMV.** Decode projections all bit-identical (0.0) to the f32 ref. Q/O proj is 2048×2048 (reuses the llama Q row). K=6144 (Down proj) is the L2-constrained shape — `8·tile_m·6144·2 ≤ 256KB` forces `tile_m=2`. **LM-head is 151936×2048** — too tall single-shot (outer > 255 BD repeat limit, same as all siblings); run per-partition (n_part=8192, 19 partitions), and the K=2048 LM-head datapath is verified at partition scale by the 16384×2048 row above (8/8/8, mean_rel_L1=0.0).

> **Qwen2.5-1.5B GEMV.** Decode projections (Q/O/K/V/Gate-Up) bit-identical or ≤1.7e-9 to the f32 ref. K=8960 (Down proj) is the L2-constrained shape — `tile_m=2` places (`tile_m=1` fails the placement pass, not L2). **LM-head is 151936×1536** — too tall single-shot (outer > 255 BD repeat limit, same as all siblings); run per-partition, K=1536 datapath verified at partition scale by the 16384×1536 row (outer=128, mean_rel_L1=2.3e-8).

> **Qwen2.5-0.5B GEMV.** Decode projections (Q/O/Gate-Up/Down) all bit-identical to the f32 ref. K=4864 (Down proj) is the only L2-constrained shape — `8·tile_m·4864·2 ≤ 256KB` forces `tile_m=2`. **LM-head is 151936×896** — too tall single-shot (outer loop > 255 BD repeat limit, same as Qwen3/llama); run per-partition, the K=896 datapath verified at partition scale by the 16384×896 row (outer=128, mean_rel_L1=7.2e-12).

> **Qwen3-0.6B LM-head is 151936×1024** — too tall to run single-shot: the outer launch loop = `M/(tile_m·herd_m)` exceeds the 255 buffer-descriptor repeat-count limit at every legal tile (151936 = 8·16·1187 has no `tile_m` divisor between 16 and 1187), so it is run **per-partition** like llama-3.2-1B's LM-head. The 16384×1024 row above verifies the K=1024 LM-head datapath at partition scale (128 launches, PASS, mean_rel_L1 = 2.0e-8).

> This plain GEMV is the exact kernel for llama-3.2-1B decode's **Q / K / V projections and LM-head**. The **O / Gate / Up / Down** projections use *fused* cascade variants (GEMV+residual, GEMV+SwiGLU+RMSNorm) — separate kernels, separate registry entries; the 8192×2048 / 2048×8192 rows here are coverage shapes. See [`details/GEMV_bf16.md`](details/GEMV_bf16.md).
> GEMV uses an **FP32 vector accumulate** (not the BFP16-emulated MMA that GEMM uses), so accuracy is effectively exact — `mean_rel_L1 ≤ 2.7e-8`, several shapes bit-identical to the f32 reference, orders of magnitude tighter than BF16 GEMM's ~9e-3.

---

## RMSNorm — tested shapes

`y = x / sqrt(mean(x²) + eps) · weight`, per row; shapes written `M×N` (M = rows / seq, N = emb_dim = reduction axis). The per-layer norm of llama-3.2-1B. **Memory-bound** (streams the whole matrix for an elementwise op), so throughput is reported as bandwidth; the fastest config is `herd_x=8` (all columns, near-linear scaling). Full data, the precision caveat, and reproduce commands are in [`details/RMSNorm_bf16.md`](details/RMSNorm_bf16.md).

| (M×N) | herd_x | latency | bandwidth | mean_rel_L1 | Used by | Status |
|---|---|---|---|---|---|---|
| 2048×2048 | 8 | 911 µs | 18.4 GB/s | 4.2e-3 | llama-3.2-1B + Qwen3-1.7B + Qwen2.5-3B prefill RMSNorm | ✅ |
| 2048×1024 | 8 | 407 µs | 20.6 GB/s | 4.3e-3 | Qwen3-0.6B prefill RMSNorm | ✅ |
| 2048×128 | 8 | 155 µs | 6.8 GB/s | 4.6e-3 | Qwen3-0.6B + Qwen3-1.7B QK-norm (per-head, N=head_dim) | ✅ |
| 2048×896 | 8 | 398 µs | 18.4 GB/s | 4.2e-3 | Qwen2.5-0.5B prefill RMSNorm | ✅ |
| 2048×1536 | 8 | 570 µs | 22.1 GB/s | 4.3e-3 | Qwen2.5-1.5B prefill RMSNorm | ✅ |
| 2048×2560 | 8 | 867 µs | 24.2 GB/s | 4.2e-3 | Qwen3-4B prefill RMSNorm | ✅ |
| 2048×3072 | 8 | 1012 µs | **24.9 GB/s** | 4.2e-3 | Llama-3.2-3B prefill RMSNorm | ✅ |

> **Qwen3-0.6B QK-norm (2048×128)** is per-head RMSNorm over `head_dim=128` (Qwen3-specific q_norm/k_norm) — the same weighted-RMSNorm kernel with a small `N=128` reduction axis; verified PASS at 4.6e-3, confirming the kernel handles a 128-wide reduction. (Harness `eps = 1e-5`; Qwen3 `eps = 1e-6` — the difference is negligible vs the bf16 datapath error.)

> Follows the **GPU / HuggingFace standard**: the `sum(x²)` reduction is accumulated in **FP32** (matching PyTorch `rms_norm_composite` / HF `LlamaRMSNorm`), giving `mean_rel_L1 = 4.2e-3` — in line with the GEMM tier and passing the canonical bf16 `rtol = 1.6e-2`. (`atol = 5e-2` covers a few large-magnitude bf16 *output*-rounding ULPs, not a reduction relaxation.) The FP32 reduction costs essentially nothing on this memory-bound kernel. See [`details/RMSNorm_bf16.md`](details/RMSNorm_bf16.md).

---

## LayerNorm — tested shapes

`y = (x − mean(x)) · rsqrt(var(x) + eps)`, per row, **several rows per kernel call**; shapes written `M×N` (M = rows / seq, N = emb_dim = normalization axis). The encoder-block norm of the transformer-layer execution studies, over the ported `layer_norm_rows` kernel — **not** the direct-codegen `programming_examples/layer_norm/` example, which accumulates its statistics in bf16 and gates an order of magnitude looser. Statistics accumulate in **FP32** and the variance is **two-pass** `E[(x − mean)²]`, the same form as the FP32 oracle, so rows at a large common offset normalize correctly — the one-pass `E[x²] − E[x]²` form the kernel first shipped lost such rows' variance entirely, and J7a's `128x768_offset` opcheck row pins the regime. Full datapath, constraints, and reproduce commands in [`details/LayerNorm_bf16.md`](details/LayerNorm_bf16.md).

| (M×N) | herd (hx/hy) | rows_per_call | mean_rel_L1 | abs_err max | mismatches | Used by | Status |
|---|---|---|---|---|---|---|---|
| 512×512 | 8/1 | 8 | 8.1e-5 | 1.6e-2 | 0 / 262144 | transformer-layer studies, encoder block norm (hidden = 512) | ✅ |
| 4096×768 | 8/1 | 8 | 7.1e-5 | 1.6e-2 | 0 / 3145728 | transformer-layer studies, `baseline_768` block norm at the block's own sequence length | ✅ |

> The `4096×768` row is Phase D1's: the same kernel at the width and sequence length the encoder block runs. `mean_rel_L1` is unchanged across a 12× larger output and a 1.5× wider normalization axis, which is what a per-row reduction should do. It carries `atol = 5e-3` against the 512-row's `5e-2`, sized from the one-pass kernel's measured `atol_required` of 1.4e-3; the two-pass kernel's `atol_required` measures 0.0 and the 5e-3 stands rather than chasing an arbitrarily small number.

> `mean_rel_L1 = 8.1e-5` is the cleanest reduction in the registry — one bf16 rounding of an f32-exact value, the floor for a bf16-out kernel (the one-pass kernel measured 2.0e-3 on the same seeds). `rel_err max = 7.8e-3` sits under `rtol`, so every element is covered by `rtol` alone. Throughput is not recorded: Phase C1 gates numerics only.

---

## AddNorm — tested shapes

Weighted layer normalization and a residual, per row, fused into one kernel call; shapes written `M×N`. The sublayer boundary of the encoder block. **The weight is a runtime memref argument**, not baked into the MLIR as iron does — one compiled ELF serves every weight vector of that shape. **Two orderings, which are two different functions**, selected by `build_addnorm_module(pre_add=...)`:

| ordering | computes | statistics over | kernel | entry point |
|---|---|---|---|---|
| post-add (default) | `LayerNorm(x) · weight + residual` | `x` | `encoder.o` | `fused_add_layer_norm_2outs` |
| pre-add | `LayerNorm(x + residual) · weight` | `x + residual` | `addnorm_pre_add.o` | `fused_add_layer_norm_1outs` |

Pre-add is what a post-norm encoder (`encoder_bert`) computes at both of its normalization points. Full datapath and the one-call-per-tile constraint in [`details/AddNorm_bf16.md`](details/AddNorm_bf16.md).

Since 2026-08-11 both orderings keep **f32 two-pass statistics** (mean first, then `E[(x − mean)²]`), the same discipline as LayerNorm above and for the same measured reason — the one-pass `E[x²] − E[x]²` form they shipped with loses an offset row's variance entirely (collapse between `|mean|/σ` 2 and 4). The `64x512_offset` / `64x768_pre_add_offset` opcheck rows pin the regime. **The figures below are the two-pass kernels' first gated hardware run (2026-08-11)**; the tolerances were not widened for the move.

| (M×N) | ordering | herd (hx/hy) | rows_per_call | mean_rel_L1 | abs_err max | atol_required | mismatches | Used by | Status |
|---|---|---|---|---|---|---|---|---|---|
| 64×512 | post-add | 8/1 | 8 | 1.486e-3 | 3.1e-2 | 1.289e-2 | 0 / 32768 | transformer-layer studies, encoder sublayer boundary (hidden = 512) | ✅ |
| 64×768 | pre-add | 8/1 | 8 | 1.963e-3 | 3.1e-2 | 1.667e-4 | 0 / 49152 | transformer-layer studies, `baseline_768` encoder sublayer boundary | ✅ |
| 64×512 | post-add, offset regime (mean 8, σ 0.25) | 8/1 | 8 | 1.390e-3 | 3.1e-2 | **0.0** | 0 / 32768 | the variance-cliff pin (doc 23 item 2); one-pass measured `mean_rel_L1` 22.2 here | ✅ |
| 64×768 | pre-add, offset regime (mean 8, σ 0.25) | 8/1 | 8 | 1.409e-3 | 3.1e-2 | **0.0** | 0 / 49152 | the variance-cliff pin, pre-add object; one-pass measured `mean_rel_L1` 33.1 here | ✅ |

> **The pre-add row needs 26× less `atol` than the post-add one at a higher relative error, and the ordering is why.** Post-add finishes with `+ residual` in bf16, so an element where `norm · weight` nearly cancels the residual carries an absolute error set by the *residual's* magnitude while its own value sits near zero — `rtol` covers none of that, and `atol_required` jumps to 1.75e-2. Pre-add has no trailing add, so every error is proportional to the output carrying it. It is the same kernel with the cancellation removed, not a better datapath.
>
> The two rows differ in width as well as ordering, so they are not a controlled comparison of the ordering alone. Both were sized by the same rule: `atol` is `atol_required` rounded up ~3× (`5e-2` and `2e-3` respectively).

> `M = 64` is a **hard cap, not a sample**: the builder requires exactly one kernel call per tile, because two or more trips through the herd loop miscompile (0 of 512 elements outside tolerance at one trip, 491 of 512 at two, at `[8,64]`/`herd_x=1`). The distinguishing feature is three L3→L1 streams per tile against a column's two shim MM2S channels; the two-stream norms and adds beside it loop correctly. The builder raises rather than emitting the broken form. Lifting the cap needs the weight staged through L2, or the residual folded into `x`'s L3 buffer — neither is done yet.

---

## QKV Projection — tested shapes

`x[M,K] @ w_qkv[K,3K]` → `q`, `k`, `v`, each `[M,K]`; shapes written `M×K`. **One** GEMM over the fused weight, with C split three ways **on the device**: the registry's `fused-cast` method already owes a separate cast launch over every element of C, so the split rides it — three cast launches, each with a column offset on its read and its own bf16 destination. No host-side slice. Method and tiles come from `gemm_registry_config`, which raises on an unmeasured shape. Full datapath in [`details/QKVProj_bf16.md`](details/QKVProj_bf16.md).

| (M×K) | GEMM (M×K×3K) | method | tile (m/kl2/kl1/n) | mean_rel_L1 | abs_err max | mismatches | Used by | Status |
|---|---|---|---|---|---|---|---|---|
| 2048×1024 | 2048×1024×3072 | fused-cast | 64/256/32/128 | 9.9e-3 | 1.95e-3 | 0 / 6291456 | transformer-layer studies, attention projections | ✅ |
| 2048×2048 | 2048×2048×6144 | fused-cast | 64/256/32/128 | 9.7e-3 | 1.22e-3 | 0 / 12582912 | transformer-layer studies, attention projections | ✅ |
| 64×768 | 64×768×2304 | fused-cast | 64/128/32/96 | 9.9e-3 | 1.95e-3 | 0 / 147456 | transformer-layer studies, shortest point of the `baseline_768` ladder | ✅ |
| 4096×768 | 4096×768×2304 | fused-cast | 64/256/32/96 | 9.9e-3 | 1.95e-3 | 0 / 9437184 | transformer-layer studies, `baseline_768` block projections | ✅ |

> **`mean_rel_L1 = 9.9e-3` is the GEMM's own number** (the bf16-out page records 9.4e-3 for this shape), so the three split-cast launches add nothing measurable — which is the point of folding the split into a cast the datapath already owed.
>
> **This resolves the ⚠️ on `2048×1024×3072` fused-cast above.** That row's note diagnoses it as a harness tolerance edge: the datapath computes the in-tier result and the gate tripped on a single near-zero-reference element at `abs_err ≈ 1.7e-3` against `atol = 1.5e-3`. Measured here at `atol = 5e-3` over 3× as many elements: `abs_err max = 1.95e-3`, **zero** mismatches. That is the remedy the note itself proposed — relax the high-precision `atol` to match the other tiers, leave the GPU-standard `rtol` alone.
>
> **The two `baseline_768` rows are the ends of the sequence ladder**, and they are the two that exercise the per-row herd. `64×768` runs at herd `1×4` — `M = 64` cannot hold eight rows of `fused-cast`'s forced `tile_m = 64` — and `4096×768` at the file-level `8×4`; a builder that stopped reading the herd from the registry row would fail to build the first and still pass the second. `64×768` is also the shape whose original `fused-cast` row returned **zeros for two of nine cast sub-tiles**, which is why a resolution check is not a correctness check.
>
> **All four `M×K×3K` triples above are validated on hardware.** The registry holds **11** such triples: the Phase C4 sweep added the nine `baseline_768` `qkv_proj` shapes (`seq×768×2304`, the full sequence ladder), so the builder resolves all of those — but resolving a tiling and having run this operator's numerical check at that shape are different claims, and only the four rows above are the second. Of the 108 projection-GEMM shapes the execution-studies case matrix asks for, **41 are now registered** (the whole `baseline_768` family plus five incidental model shapes); the remaining 67 are `baseline_512` and `baseline_1024`, which are the same sweep tool over a different `--family` and are deliberately left as a later machine-time run. The builder still raises on those rather than guessing a tiling.

---

## FFN (GeLU) — tested shapes

`y = gelu(x[M,K] @ w_up[K,F]) @ w_down[F,K]`; shapes written `M×K×F`. Five launches in one ELF — up-projection, its cast, the activation, down-projection, its cast. The activation is the **tanh approximation** (`gelu_pytorch_tanh`), not erf; iron's oracle uses torch's erf default, whose difference hides at iron's 4e-2 tolerance and does not hide at this one. iron's `down_proj_depth` is the down-projection's registry `tile_k_l2` and is read from the registry, not passed in. Full datapath in [`details/FFN_bf16.md`](details/FFN_bf16.md).

| (M×K×F) | up GEMM | down GEMM | tile (m/kl2/kl1/n) | mean_rel_L1 | abs_err max | mismatches | Used by | Status |
|---|---|---|---|---|---|---|---|---|
| 2048×1024×3072 | fused-cast | fused-cast | 64/256/32/128 (both) | 1.6e-2 | 1.59e-3 | 0 / 2097152 | transformer-layer studies, encoder FFN sublayer | ✅ |
| 4096×768×3072 | drain | fused-cast | 32/256/32/128 up, 64/512/32/96 down | 1.6e-2 | 1.71e-3 | 0 / 3145728 | transformer-layer studies, `baseline_768` FFN sublayer | ✅ |

> **The `baseline_768` row is the only point on the sequence ladder where this operator builds at hidden 768**, and the reason is in the two method columns rather than in the numbers. At `K = 768` the up-projection takes `tile_n = 128` and the down-projection `tile_n = 96` at *every* sequence length, and two **same-method** GEMMs with different `tile_n` declare `f32_to_bf16_mn_<suffix>` twice with different memref types, which `stitch_elf` rejects. `seq = 4096` is the one point the registry puts them on different methods, and therefore on different objects. `64…2048` are `drain`/`drain` and `8192`/`16384` are `fused-cast`/`fused-cast`; both collide. That makes buildability here a property of the registry's winners rather than of the shape — a re-sweep that moved either projection onto the other's method would take the operator from *builds* to *does not build* with no source change. The fix is a symbol suffix minted per `(method, tile_n)` in `llms/shared/builders/gemm_builder.py`.
>
> Mixing the methods costs nothing measurable: `mean_rel_L1` is within 2% of the all-`fused-cast` row above.

> **`mean_rel_L1 = 1.6e-2` is ~1.6× a single GEMM's**, and that gap is what the composition costs: the device stages the up-projection output in bf16 and the activation kernel carries bf16 intermediates, while the reference is FP32 end to end. Reproducing either in the oracle would hide exactly the error it introduces.
>
> **One of seven resolvable shapes.** A shape needs a high-precision registry entry for *both* directions, `(M, K, F)` and `(M, F, K)`; seven expansions satisfy that today (`2048×1024×2048`, `2048×1024×3072`, `2048×2048×6144`, `2048×2048×8192`, `2048×2560×4096`, `2048×2560×9728` and `2048×3072×8192`) and only the row above has been run on hardware. The other six are a coverage gap, not a known failure. The case matrix's remaining FFN shapes lack a high-precision entry on one side and the builder raises on them rather than guessing.

---

## MHA + Output Projection — tested shapes

`y = softmax(Q Kᵀ / √d [+ causal mask]) V @ W_o`, the attention sublayer end to end in one ELF; shapes written `S×S, Hq/Hkv, d` with model width `E = Hq·d` and projection `S×E×E`. **Seq-first throughout**: the FlashAttention half writes `[S, E]`, which *is* the projection's `A` operand, so the fusion needs no transpose and no host step — only one dispatch instead of two. The attention half is `attn_npu2_seqfirst.py` and `attn_npu2.o` unmodified; what this entry adds is the launch structure. iron's `o_proj_acc_depth` is the projection's registry `tile_k_l2`. Full datapath in [`details/MHAOutProj_bf16.md`](details/MHAOutProj_bf16.md).

| S×S | Hq/Hkv | d | causal | O GEMM | tile (m/kl2/kl1/n) | mean_rel_L1 | abs_err max | atol_required | mismatches | Used by | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 512×512 | 8/8 | 64 | ✗ | drain | 32/256/32/128 | 4.64e-2 | 1.95e-2 | 1.85e-2 | 0 / 262144 | transformer-layer studies, attention sublayer | ✅ |
| 512×512 | 8/8 | 64 | ✓ | drain | 32/256/32/128 | 3.58e-2 | 5.86e-2 | 4.88e-2 | 0 / 262144 | transformer-layer studies, decoder attention sublayer | ✅ |
| 2048×2048 | 16/16 | 64 | ✓ | drain | 32/256/32/128 | 4.11e-2 | 5.08e-2 | 4.81e-2 | 0 / 2097152 | transformer-layer studies, prefill-sized attention sublayer | ✅ |
| 4096×4096 | 12/12 | 64 | ✗ | drain | 32/256/32/96 | 5.33e-2 | 9.03e-3 | 8.71e-3 | 0 / 3145728 | transformer-layer studies, `baseline_768` attention sublayer | ✅ |

> **`mean_rel_L1` sits in FlashAttention's band, not a GEMM's**, and that is the expected answer: the attention half *is* that kernel, and a projection whose own relative error is 4× smaller cannot pull the total down. The composition costs nothing measurable in relative terms.
>
> **`atol_required` — `max(|out−ref| − rtol·|ref|)`, the smallest `atol` the run would have passed at — is the column to read, not `abs_err max`.** Under causal masking the largest absolute error lands on a large-magnitude element `rtol` already covers: the first rows attend to a handful of keys, so `|y|` runs to 4.1 instead of 0.35 while the relative error is if anything lower. The causal rows carry `atol = 8e-2`, a 1.6× margin — below the registry's usual 2–3× and deliberately so, since `1e-1` is a hard ceiling and this datapath's honest error gets within a factor of two of it.
>
> **The `4096×4096` row has the loosest relative error and the tightest `atol`, which is the same effect running the other way.** Softmax over 4096 keys averages `V` eight times harder than over 512, so `|y|` shrinks by roughly `√8` and the same relative error lands closer to zero: `atol_required` falls to 8.7e-3 while `mean_rel_L1` rises to 5.33e-2. Its `atol` is 2.5e-2, a 2.9× margin and 4× below the ceiling — so the row the encoder block actually runs has the most headroom of the four, not the least. It is **non-causal** because `encoder_bert` is bidirectional; the causal rows above are a different device path and are not `baseline_768` evidence however large they are.
>
> **`head_dim = 64` throughout, on purpose**: `head_dim = 128` FlashAttention has been flaky (hang or NaN) on some NPU2 setups, and the builder rejects it rather than letting a mis-shaped call find that out on hardware. **No `fused-cast` projection has been run in this composition** — that method wants `runtime_loop_tiling_sizes=[2,2]` and this operator runs at `[1,1]` because the attention half needs it, so the combination is untested rather than known-good. Both are coverage gaps, not known failures.

---

## FlashAttention — tested shapes

Fused scaled-dot-product attention (online-softmax FlashAttention) with grouped-query attention and optional causal masking. **Compute-bound** (two matmuls Q@Kᵀ and P@V), so throughput is GFLOP/s. Kernel = `attn_npu2.o`, driven by the **heads-first** harness `attn_npu2.py`; verified on NPU2 across head dim 64/128, MHA & GQA, short & long sequences, causal & non-causal. (A **seq-first** variant `attn_npu2_seqfirst.py` drives the same `.o` for llama-3.2-1B prefill — bit-identical.) **All rows use the one near-unique full-chip config** `lqp=256, num_q_tiles=4, num_heads_per_unroll=2, num_cascade_stages=4` (FA's tile config is determined by the constraints, not tuned — see detail page). Full datapath, tunables, and reproduce commands in [`details/FlashAttention_bf16.md`](details/FlashAttention_bf16.md).

| lq×lk | dk/dv | heads q/kv | causal | window | dv_chunks | latency | GFLOP/s | mean_rel_L1 | Status |
|---|---|---|---|---|---|---|---|---|---|
| 2048×2048 | 64/64 | 32/8 | ✓ | 0 | 1 | 15.4–16.1 ms | **1065–1116** | 3.9e-2 | ✅ |
| 2048×2048 | 64/64 | 32/32 | ✓ | 0 | 1 | 16.9 ms | 2031 | 3.9e-2 | ✅ |
| 512×512 | 64/64 | 2/2 | ✗ | 0 | 1 | 0.73 ms | 184 | 4.4e-2 | ✅ |
| 512×512 | 64/64 | 12/6 | ✗ | 0 | 1 | 1.22 ms | 661 | 4.6e-2 | ✅ |
| 512×512 | 64/64 | 64/8 | ✗ | 0 | 1 | 3.79 ms | 1135 | 4.6e-2 | ✅ |
| 512×512 | 128/128 | 32/8 | ✗ | 0 | 2 | 4.38 ms | 980 | 4.4e-2 | ✅ |
| 512×512 | 128/128 | 28/4 | ✗ | 0 | 2 | 4.05 ms | 928 | 4.4e-2 | ✅ |
| 16384×16384 | 64/64 | 2/2 | ✓ | 0 | 1 | 39.6 ms | 1734 | 4.5e-2 | ✅ |
| 16384×16384 | 64/64 | 2/2 | ✗ | 0 | 1 | 40.1 ms | **3427** | 5.5e-2 | ✅ |
| 2048×2048 | 128/128 | 16/8 | ✓ | 0 | 2 | 17.6 ms | 979 | 3.8e-2 | ✅ |
| 2048×2048 | 64/64 | 14/2 | ✓ | 0 | 1 | 7.27 ms | 1035 | 3.8e-2 | ✅ |
| 2048×2048 | 128/128 | 12/2 | ✓ | 0 | 2 | 14.5 ms | 891 | 3.8e-2 | ✅ |
| 2048×2048 | 128/128 | 24/8 | ✓ | 0 | 2 | 25.9 ms | 995 | 3.8e-2 | ✅ |
| 2048×2048 | 128/128 | 32/8 | ✓ | 0 | 2 | 35.0 ms | 983 | 3.8e-2 | ✅ |
| 2048×2048 | 64/64 | 32/8 | ✓ | **512** | 1 | 14.5 ms | *n/a — see note* | 3.68e-2 | ✅ |

> **`window` is new, and `window = 0` means full causal — which is every row above and every shipped model.** It is a compile-time constant of the kernel object (`-DWINDOW_LEN`), not a runtime argument: the launch signature `attention_bf16(q, k, v, gp)` has no host scalar path, so a local/global model needs **two compiled attention modules**, not a toggle. `window = 0` is bit-identical to the pre-windowing kernel — verified at the object level, `sha256(attn_npu2.o)` unchanged and the preprocessed source byte-identical, so the `window = 0` rows above are unaffected by construction rather than by re-measurement.
>
> **The windowed row deliberately carries no GFLOP/s, and its latency is there to show the ABSENCE of a speedup.** Measured matched pair, same config, same session: `W = 512` **14.47 ms** vs `W = 0` **14.06 ms**, while the band discards 56% of the scores full causal keeps (21.9% surviving vs 50.0%). The windowed kernel is if anything ~3% *slower*, inside the ±5% run-to-run variation this table records elsewhere — so: no change, and nothing like the ~2.3× a work-proportional speedup would give. Masking here is **element-wise, not tile-skipping**: the KV chunk loop bound is a static compile-time constant (`attn_npu2.py`, the `chunks_per_stage` loop), so a windowed run streams and multiplies every `(Q tile, KV block)` pair exactly as an unwindowed one does and then masks *more* of the result — today's bare `return` for a below-diagonal block becomes a full `-inf` fill. A window is **correctness-only, zero speedup**, and quoting a throughput for it would compound the existing convention error (`perf_flops *= 0.5` under `causal` is a convention, not executed work; a band ratio would be a second one). The route that *would* pay is tile skipping, which is a documented `ERT_CMD_STATE_TIMEOUT` path plus two open compiler items — out of scope here and not attempted.
>
> **Its gate carries a negative control, and the control is the point.** Full causal is a strict superset of a `W = 512` window, so an implementation that silently degraded to full causal would still produce plausible output. `run_npu2_makefile_peano_causal_window512_negative.lit` rebuilds the same shape with the band switched off in the kernel and requires the banded reference to **reject** it, reading the completed comparison's own statistics rather than inverting an exit status. Measured: the unwindowed kernel is rejected on **197331 / 4194304 elements (4.70%)** at `mean_rel_L1 = 4.56e-1` and `atol_required = 5.34e-1`, i.e. **5.3× over** the `1e-1` ceiling it has to clear. The gate discriminates.
>
> **Banding did NOT cost accuracy headroom, which was not the expectation.** A matched pair measured in one session at this exact config: windowed `mean_rel_L1 = 3.676e-2`, unwindowed `3.856e-2`; `atol_required = 8.048e-2` for **both**, i.e. the same 1.24× margin under the `1e-1` ceiling, and `abs_err max = 8.398e-2` for both. The reason is structural: the worst-magnitude elements sit in the **first `W` rows**, where a `q − W < k ≤ q` band and full causal are *the same mask* (a row at `q_abs < W` attends to `[0, q_abs]` either way). A band cannot make that region worse, and it only removes contributions elsewhere. The prior expectation — that concentrating the softmax harder would push a banded row toward the ceiling — is **not what the hardware did**.

> **Qwen3-0.6B prefill attention** (`head_dim = 128`, 16q/8kv GQA, causal, lq=lk=2048): verified PASS at mean_rel_L1 = 3.8e-2 (full-output check, rtol 1.6e-2 / atol 1e-1) with the default full-chip config (`lqp=256, num_q_tiles=4, num_heads_per_unroll=2, num_cascade_stages=4`, `dv_chunks=2` for head_dim=128). Note: head_dim=128 FA has been flaky (hang/NaN) on some NPU2 setups; this run completed cleanly, and Qwen3-0.6B prefill can also fall back to CPU attention (`cpu_attn`) if a deployment hits the hang.

> **Qwen2.5-1.5B prefill attention** (`head_dim = 128`, 12q/2kv GQA, causal, lq=lk=2048): verified PASS at mean_rel_L1 = 3.83e-2 (full-output check, rtol 1.6e-2 / atol 1e-1) with the default full-chip config (`lqp=256, num_q_tiles=4, num_heads_per_unroll=2, num_cascade_stages=4`, `dv_chunks=2` for head_dim=128). head_dim=128 FA has been flaky (hang/NaN) on some NPU2 setups; this run completed cleanly, and prefill can fall back to CPU attention (`cpu_attn`) if a deployment hits the hang.

> **Qwen2.5-0.5B prefill attention** (`head_dim = 64`, 14q/2kv GQA, causal, lq=lk=2048): verified PASS at mean_rel_L1 = 3.83e-2 with the default full-chip config (`lqp=256, lkp=64, num_q_tiles=4, num_heads_per_unroll=2, num_cascade_stages=4`, `dv_chunks=1` for head_dim=64). head_dim=64 has no hang risk. Prefill can also fall back to CPU attention (`cpu_attn`).

> All rows measured on NPU2 with the heads-first harness at the default tiling (`lqp=256, num_q_tiles=4, num_heads_per_unroll=2, num_cascade_stages=4` = 32 tiles, full 8×4 array). Accuracy `mean_rel_L1 ≈ 3.9e-2` is ~4× the GEMM tier: FA chains **two BFP16-emulated MMAs** plus a **bf16 online-softmax**, so it is looser than a single matmul (looser than GPU FA's `5e-2` only by the `atol`, not the standard `rtol = 1.6e-2`); accuracy is set by the datapath, not the shape. The **2048, 32q/8kv causal** row is llama-3.2-1B prefill's config (seq-first harness, bit-identical to heads-first — verified `max abs diff = 0`); its GFLOP/s range is run-to-run timing variation. `head_dim=128` rows use `dv_chunks=2`. A separate tunable sweep found only 2 of 8 candidate 32-tile configs place (constraints: columns `num_heads_per_unroll × num_q_tiles ≤ 8`, rows `num_cascade_stages ≤ 4`, `num_heads_per_unroll ≤ 2`). See [`details/FlashAttention_bf16.md`](details/FlashAttention_bf16.md).

---

## Element-wise Add — tested shapes

`c = a + b`, per-element, BF16. The residual adds of llama-3.2-1B (the prefill residual is the fused `o_ffn` inline 2-D variant — same math; this entry measures the **standalone** `eltwise_add`). **Memory-bound** (O(N) streaming, zero arithmetic intensity), so throughput is bandwidth. The **cleanest** kernel in the registry — a single bf16 rounding, no accumulation. Full datapath, herd sweep, and reproduce commands in [`details/EltwiseAdd_bf16.md`](details/EltwiseAdd_bf16.md).

| N | best config (hx/hy/tile_n) | latency | bandwidth | mean_rel_L1 | Status |
|---|---|---|---|---|---|
| 1048576 | 8/1/2048 | 175 µs | 36.0 GB/s | 1.9e-3 | ✅ |
| 2097152 | 8/1/2048 | 277 µs | 45.4 GB/s | 1.9e-3 | ✅ |
| 4194304 (2048×2048) | 8/1/2048 | 437 µs | 57.7 GB/s | 1.9e-3 | ✅ (llama-3.2-1B + Qwen3-1.7B + Qwen2.5-3B residual, seq·emb) |
| 8388608 | 8/1/2048 | 798 µs | **63.0 GB/s** | 1.9e-3 | ✅ |
| 1835008 (2048×896) | 8/1/2048 | 243 µs | 45.3 GB/s | 1.9e-3 | ✅ (Qwen2.5-0.5B residual, seq·emb) |
| 3145728 (2048×1536) | 8/1/2048 | 364 µs | 51.9 GB/s | 1.9e-3 | ✅ (Qwen2.5-1.5B residual, seq·emb) |
| 5242880 (2048×2560) | 8/1/2048 | 516 µs | 61.0 GB/s | 1.9e-3 | ✅ (Qwen3-4B residual, seq·emb) |
| 6291456 (2048×3072) | 8/1/2048 | 614 µs | 61.4 GB/s | 1.9e-3 | ✅ (Llama-3.2-3B residual, seq·emb) |
| 262144 (512×512, 2-D) | 8/1/512 | — | — | 1.9e-3 | ✅ (transformer-layer studies, encoder residual) |
| 262144 (512×512, 2-D + causal mask) | 8/1/512 | — | — | 3.2e-3 | ✅ (transformer-layer studies, attention-score masking) |
| 3145728 (4096×768, 2-D) | 8/1/768 | — | — | 1.9e-3 | ✅ (transformer-layer studies, `baseline_768` block residual) |

> The last three rows are the **2-D-in / 2-D-out** variant (`_build_add_2d_to_2d`, the same builder llama's fused `o_ffn` prefill residual uses), reached through `transformer_layer/builders/elementwise_add.py`. Same arithmetic, same 1.9e-3; only the L3 layout differs, and keeping the output 2-D is what lets a downstream launch read it without an `expand_shape`. The **causal-mask** row is that builder with `causal_mask=True` and a torch-precomputed `-10000.0` triangular mask bound as the second operand — there is no device design of its own, which is exactly why it is a builder keyword rather than a sixth operator. Its higher `mean_rel_L1` (3.2e-3) and `abs_err max` (6.4e+1) come from the masked half of the tensor: bf16 spacing at |value| ≈ 10⁴ is 64, so a single ULP there is 6.4e+1, and `np.isclose` passes it on `rtol` (`1.6e-2 · 10⁴ = 160`). The unmasked half is a plain add of the same `randn` scores and carries a plain add's error. `-10000.0` rather than `-inf` because `-inf` in bf16 propagates NaN through the add. The `4096×768` row is Phase D1's `baseline_768` point, at the block's own activation shape; its `atol_required` is **0.0**, meaning `rtol` alone accounts for every element — a single bf16 rounding of an f32 sum is always within `rtol` of the f32 value, so `atol` is doing no work for this kernel at all. Throughput is not recorded for any of the three: these rows gate numerics only.

> `mean_rel_L1 = 1.9e-3` is the lowest in the registry — `c=a+b` rounds each output once (matching `torch.add` bf16: f32 sum, single round, no accumulation), bit-identical across all configs and `N`. Best config `herd_x=8, herd_y=1` for every shape: the 3-DMA-per-tile shim-channel limit caps the herd at one 8-column row (**cannot fill 32 tiles** — `herd_y>1` fails to place), but within that `herd_x` scales near-linearly (9→57.7 GB/s as herd_x 1→8). Highest bandwidth in the registry (pure streaming). See [`details/EltwiseAdd_bf16.md`](details/EltwiseAdd_bf16.md).

---

## SiLU-and-Mul — tested shapes

`out = SiLU(gate) · up`, `SiLU(x) = x·sigmoid(x)`, per-element, BF16. The SwiGLU activation of llama-3.2-1B prefill FFN (the standalone `silu_and_mul` is measured; llama runs the bit-identical 2-D `build_module_2d` variant). **Memory-bound** (O(N) streaming, ~1 op/byte), so throughput is bandwidth. sigmoid is computed via the hardware `aie::tanh` (`0.5·(1+tanh(g/2))`); the precision is the "bf16 + one transcendental" tier. Full datapath, sweep, and reproduce commands in [`details/SiLU_Mul_bf16.md`](details/SiLU_Mul_bf16.md).

| N | (as 2-D) | best config (hx/hy/tile_n) | latency | bandwidth | mean_rel_L1 | abs_err max | Status |
|---|---|---|---|---|---|---|---|
| 2097152 | — | 8/1/4096 | 569 µs | 22.1 GB/s | 1.0e-2 | 0.125 | ✅ |
| 4194304 | 2048×2048 | 8/1/4096 | 1052 µs | 23.9 GB/s | 1.0e-2 | 0.125 | ✅ |
| 8388608 | — | 8/1/4096 | 2247 µs | 22.4 GB/s | 1.0e-2 | 0.125 | ✅ |
| 16777216 | 2048×8192 | 8/1/4096 | 4016 µs | **25.1 GB/s** | 1.0e-2 | 0.125 | ✅ |
| 6291456 | 2048×3072 (seq·hidden) | 8/1/4096 | 1771 µs | 21.3 GB/s | 1.0e-2 | 0.125 | ✅ |
| 9961472 | 2048×4864 (seq·hidden) | 8/1/4096 | 2489 µs | 24.0 GB/s | 1.0e-2 | 0.125 | ✅ |
| 18350080 | 2048×8960 (seq·hidden) | 8/1/4096 | 4933 µs | 22.3 GB/s | 1.0e-2 | 0.188 | ✅ |
| 12582912 | 2048×6144 (seq·hidden) | 8/1/4096 | 3041 µs | 24.8 GB/s | 1.0e-2 | 0.125 | ✅ (Qwen3-1.7B SwiGLU) |
| 19922944 | 2048×9728 (seq·hidden) | 8/1/4096 | 5077 µs | 23.5 GB/s | 1.0e-2 | 0.125 | ✅ (Qwen3-4B SwiGLU) |
| 22544384 | 2048×11008 (seq·hidden) | 8/1/4096 | 5694 µs | 23.8 GB/s | 1.0e-2 | 0.188 | ✅ (Qwen2.5-3B SwiGLU) |

> **Qwen2.5-1.5B SwiGLU**: `N = 18350080 = seq·hidden = 2048·8960` (intermediate size 8960), verified PASS at 1.0e-2 with the default best config.

> **Qwen3-0.6B SwiGLU**: `N = 6291456 = seq·hidden = 2048·3072` (intermediate size 3072), verified PASS at 1.0e-2 with the default best config.

> **Qwen2.5-0.5B SwiGLU**: `N = 9961472 = seq·hidden = 2048·4864` (intermediate size 4864), verified PASS at 1.0e-2 with the default best config.

> `mean_rel_L1 = 1.0e-2` is an order of magnitude above Element-wise Add (1.9e-3): the hardware `aie::tanh<bf16>` LUT approximation plus a chain of bf16 roundings (vs a single rounding for a plain add). Verified element-wise over the full output (no cosine) at `rtol = 1.6e-2, atol = 8e-2` — `atol` covers the worst-case `tanh`-LUT element (`abs_err max = 0.125`); the mean error sits inside `rtol`. Best config `herd_x=8, herd_y=1, tile_n=4096` for every shape (= llama's default): `herd_y>1` fails the shim-channel limit and some `tile_n`/`herd_x` fail a non-monotonic buffer-descriptor limit, so the best config is the fastest one that places. `herd_x` scales 7.6× (1→8). See [`details/SiLU_Mul_bf16.md`](details/SiLU_Mul_bf16.md).

---

## RoPE — tested shapes

Rotary Position Embedding applied to Q/K, **half-split** convention (HuggingFace Llama `rotate_half`), per row; shapes written `rows × head_dim` (rows = n_heads·seq for prefill, n_heads for decode). BF16 in/out, per-element rotation (no reduction, no non-linearity — cos/sin come from a precomputed LUT). **Memory-bound** (streams input + LUT in, output out, ~1 flop/byte), so throughput is bandwidth; the fastest config is `herd_x=8` (all columns, near-linear). The kernel links the **same `rope_halfsplit.cc` (`rope.o`) llama uses** — not the interleaved `rope_lut/`/`rope_sincos/` decoys. Full data, the decoy/provenance note, and reproduce commands are in [`details/RoPE_bf16.md`](details/RoPE_bf16.md).

| (rows×head_dim) | herd (hx/hy) | latency | bandwidth | mean_rel_L1 | Used by | Status |
|---|---|---|---|---|---|---|
| 8×64 | 8/1 | 83 µs | 0.04 GB/s | 2.4e-3 | llama-3.2-1B decode RoPE-K | ✅ |
| 32×64 | 8/1 | 82 µs | 0.15 GB/s | 2.7e-3 | llama-3.2-1B decode RoPE-Q | ✅ |
| 2048×64 | 8/1 | 105 µs | 7.5 GB/s | 2.8e-3 | coverage | ✅ |
| 4096×64 | 8/1 | 118 µs | 13.3 GB/s | 2.8e-3 | coverage / Qwen2.5-0.5B prefill RoPE-K (rows=n_kv·seq=2·2048) | ✅ |
| 28672×64 | 8/1 | 303 µs | 36.4 GB/s | 2.8e-3 | Qwen2.5-0.5B prefill RoPE-Q (rows=n_heads·seq=14·2048) | ✅ |
| 16384×64 | 8/1 | 210 µs | 30.0 GB/s | 2.8e-3 | llama-3.2-1B prefill RoPE-K | ✅ |
| 65536×64 | 8/1 | 579 µs | 43.4 GB/s | 2.8e-3 | llama-3.2-1B prefill RoPE-Q | ✅ |
| 32768×128 | 8/1 | 477 µs | 52.8 GB/s | 2.8e-3 | Qwen3-0.6B + Qwen3-1.7B + Qwen2.5-3B prefill RoPE-Q (rows=n_heads·seq=16·2048) | ✅ |
| 16384×128 | 8/1 | 285 µs | 44.2 GB/s | 2.8e-3 | Qwen3-0.6B + Qwen3-1.7B prefill RoPE-K (rows=n_kv_heads·seq=8·2048) | ✅ |
| 49152×128 | 8/1 | 667 µs | **56.6 GB/s** | 2.8e-3 | Llama-3.2-3B prefill RoPE-Q (rows=n_heads·seq=24·2048) | ✅ |
| 24576×128 | 8/1 | 380 µs | 49.7 GB/s | 2.8e-3 | Qwen2.5-1.5B prefill RoPE-Q (rows=n_heads·seq=12·2048) | ✅ |
| 4096×128 | 8/1 | 149 µs | 21.1 GB/s | 2.8e-3 | Qwen2.5-1.5B + Qwen2.5-3B prefill RoPE-K (rows=n_kv_heads·seq=2·2048) | ✅ |

> **Qwen3-0.6B uses `head_dim = 128`** (vs llama's 64) — the two rows above are the first registry coverage of `head_dim = 128`; same half-split `rope_halfsplit.cc` kernel, verified PASS at 2.8e-3 (accuracy unchanged, set by the datapath not the head dim).

> `mean_rel_L1 = 2.8e-3` is the second-cleanest in the registry (above Element-wise Add 1.9e-3, below RMSNorm 4.2e-3): a rotation is a few bf16 multiplies and one add/sub per element with **no accumulation** — nothing to amplify error, and `|out| ≈ |x|` so no near-zero blowup. Verified element-wise over the full output (no cosine) at `rtol = 1.6e-2, atol = 5e-2`; bit-identical across all herd configs and shapes (decode rows 8/32 read slightly lower from smaller rotation angles). Best config `herd_x=8, herd_y=1` for every shape: each tile uses 3 shim DMAs (input/LUT in, output out), so `herd_x·herd_y>8` exhausts the shim channels (the herd **cannot fill 32 tiles**, same limit as Element-wise Add / SiLU); within 8 tiles `herd_x` scales 7.4× (1→8). Small shapes are latency-bound by a ~80 µs launch floor. See [`details/RoPE_bf16.md`](details/RoPE_bf16.md).
