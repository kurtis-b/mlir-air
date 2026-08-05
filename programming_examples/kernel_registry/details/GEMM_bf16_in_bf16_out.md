<!---//===- GEMM_bf16_in_bf16_out.md --------------------------*- Markdown -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//-->

# GEMM (BF16 in, BF16 out) — Kernel Detail

> Generic `C = A @ B` for the Q/K/V/O/Gate/Up/Down weight projections of a decoder-only LLM. **BF16 inputs, BF16 output** (half the DDR bytes of f32-out). The accumulation precision depends on the method — see below.
> Shapes are written **`M×K×N`**: `C[M,N] = A[M,K] @ B[K,N]` (M = output rows / seq, K = reduction, N = output cols).
>
> Companion: [`GEMM_bf16_in_fp32_out.md`](GEMM_bf16_in_fp32_out.md) (the f32-output sibling) · [`../supported_kernels.md`](../supported_kernels.md) · [`../README.md`](../README.md)
> **Scope: NPU2 (Strix / AIE2P) only.** All numbers, tile configs, and tolerances here are for the aie2p target, measured on real NPU2 (RyzenAI-npu4), full-chip herd 8×4, June 2026. Reproduce commands in "How to test" below.

---

## The precision knob

A bf16 output can be produced two ways with **different accumulation precision**, exposed as `--high-precision {true,false}`:

- **`--high-precision true` (default)** — FP32 accumulate across the whole K reduction, **single epilogue cast** to bf16. Precision `mean_rel_L1 ≈ 9.7e-3`, the GPU standard (same tier as the [f32-out page](GEMM_bf16_in_fp32_out.md)). A sub-knob `--method {auto,fused-cast,drain}` picks *how* the single cast is done (perf only, precision identical).
- **`--high-precision false`** — direct-codegen bf16, which truncates the partial sum to bf16 **once per L2 tile** (`K / tile_k_l2` times). Faster, but precision degrades with K (`mean_rel_L1` 1.3e-2 → 1.9e-2). This is the legacy llama-default behavior.

If you can consume an **f32** output directly, the [f32-out sibling](GEMM_bf16_in_fp32_out.md) is faster (no cast) at the same high precision.

---

## Builder

```
programming_examples/matrix_multiplication/bf16_in_bf16_out/run.py
  # high-precision true, method=fused-cast:
  build_module_gemm_cast(m, k, n, tile_m, tile_k_l2, tile_k_l1, tile_n, herd_m, herd_n,
                         arch="aie2p", cast_tile_n=...)        # GEMM f32-out + cast launch (one ELF)
  # high-precision true, method=drain:
  build_module(..., bfloat16, bfloat16, emit_external_call=True, drain_chunks=...)  # in-GEMM drain cast
  # high-precision false:
  build_module(..., bfloat16, bfloat16, direct_codegen=True)   # per-L2-tile bf16 truncation
```

The example is self-contained (`run.py` + `Makefile` + `mm_aie2p.cc` + lit + README). Contract knobs:

| Knob | Values | Meaning |
|---|---|---|
| `--high-precision` | `true` (default) / `false` | FP32-accumulate + single cast vs. per-L2-tile bf16 truncation |
| `--method` (high-precision only) | `auto` (default) / `fused-cast` / `drain` | how the single cast is realized; `auto` picks fused-cast for `M*K*N ≥ 4e9` else drain |

> The legacy `matrix_multiplication/bf16/` example carries the same builders behind low-level flags (`--direct-codegen`, `--emit-external --output-dtype bf16`, `--fused-bf16-cast`) and also targets NPU1 (aie2); this registry tracks the NPU2 contract-split `bf16_in_bf16_out/` example.

---

## Numerical datapath (what "BF16-in / BF16-out GEMM" means here)

```
A,B stored bf16 → 8×8×8 MMA (BFP16-emulated) → FP32 accumulator → [cast strategy] → BF16 output
```

- **MMA accumulator is always FP32** (`accfloat` / ACC2048). **aie2p uses BFP16 emulation** (`-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`): bf16 operands are cast to block-floating-point `v64bfp16ebs8` for the 8×8×8 MMA — a small block-quantization error a true bf16 GEMM does not have; `conv_even` rounding keeps it unbiased.
- **Where the FP32 partial sum lives between K-tiles is the precision distinction, and it differs by method** (verified from lowered IR). K is tiled twice: an outer **L2 loop** (`K / tile_k_l2`, wraps the herd) and an inner **L1 loop** (`tile_k_l2 / tile_k_l1`). The FP32 register accumulator only spans the **inner** loop; at each **L2-tile boundary** the partial sum is written to the L1 C buffer in *its* dtype and read back on the next L2 tile.
  - **`high-precision true`** (fused-cast / drain): the L1 C accumulator stays **f32** across all L2 tiles, and bf16 happens exactly **once** — fused-cast in a separate cast launch, drain in the GEMM's drain herd. So there is **no inter-tile truncation**; this is the GPU single-epilogue-cast standard. `mean_rel_L1 ≈ 9.7e-3`, flat across shape.
  - **`high-precision false`** (direct-codegen): the L1 C buffer is **bf16**, so the partial sum is truncated to bf16 and re-extended **once per L2 tile** → `K / tile_k_l2` truncations across K (e.g. Down K=8192, tile_k_l2=256 → **32 truncations**). This is *not* a single cast. Precision degrades with the truncation count: `mean_rel_L1` = 1.3e-2 at 4–8 truncations (O / Gate) but **1.9e-2 at 32 truncations** (Down). A larger `tile_k_l2` reduces truncations (and improves both accuracy and perf).

### The three high-precision methods

| Method | How the single cast is done | tile_m | Best for |
|---|---|---|---|
| **fused-cast** | external GEMM writes an **f32** C scratch at full `tile_m=64`, then a separate memory-bound `f32→bf16` cast launch in the **same module** (one fused ELF, two `air.launch`es) | 64 | **large** shapes (`M*K*N ≥ 4e9`) — the GEMM dominates and amortizes the fixed cast cost |
| **drain** | in-GEMM drain herd casts the f32 accumulator to bf16 once before DMA-out (single launch, self-contained) | 32 | **small / thin** shapes — no separate cast launch, but capped at `tile_m=32` |

`auto` picks fused-cast when `M*K*N ≥ 4e9` else drain — verified correct on all 7 tested shapes (below).

> **Why drain caps at tile_m=32.** At `tile_m=64` the bf16-out drain needs f32 acc (32 KB) + A/B ping-pong (24 KB) + a separate bf16 drain buffer (16 KB) = 72 KB > 64 KB L1. Removing the +16 KB via an **in-place cast** (one buffer, f32 view to accumulate + bf16 view to drain) is the natural fix but is **compiler-blocked** (verified 2026-06-16): the per-PE offset-0 variant needs `tile_k_l2=K` → L2 overflow at K≥2048; the herd-buffer variant's subview-of-view non-zero offset fails `airrt-to-npu` (`memref.cast offset:N vs offset:0 cast incompatible`). So drain's ceiling is tile_m=32; fused-cast sidesteps this by writing f32 (which fits at tile_m=64) and casting in a separate launch.

---

## Tunable parameters

Same GEMM knobs as the [f32-out page](GEMM_bf16_in_fp32_out.md#tunable-parameters). The **Recommended** column is the tile sweep's best for large GEMMs.

| Knob | Recommended | Hard constraint | Note |
|---|---|---|---|
| `tile_m` | 64 (fused) / 32 (drain) | `M % (tile_m × herd_m) == 0` | drain capped at 32 (L1 budget, see above) |
| `tile_k_l2` | 256–512 | `K % tile_k_l2 == 0` | also reduces direct-bf16 truncation count |
| `tile_k_l1` | 32 | `tile_k_l2 % tile_k_l1 == 0` | inner-accumulation chunk |
| `tile_n` | **128** | **`N % (tile_n × herd_n) == 0`** ⚠️ | **dominant perf knob** — 128 beats 64 by ~1.3–1.4× |
| `herd_m` | 8 | `M ≥ tile_m × herd_m` | row-parallel; 8 = full chip rows. Per-row override, see below |
| `herd_n` | 4 | `N ≥ tile_n × herd_n` | col-parallel; 4 = full chip cols → 8×4 = 32-tile herd |
| `--cast-tile-n` (fused only) | shape-dependent | `N % cast_tile_n == 0` | the cast launch's tile; ~10% spread, near-optimal at default |

⚠️ **Silent-corruption trap**: `N % (tile_n × herd_n) != 0` is **not asserted** — verify before running.

**The herd is per-row, not a global 8×4.** Most rows use the full chip, so the JSON carries `herd: [8, 4]` at the top level and says nothing per shape. A row whose `M` cannot hold 8 of its method's `tile_m` — every short-sequence row of the transformer-layer study sweep below, starting at `M = 64` — carries its own `herd` inside the method entry, which overrides the file-level one. `registry_lookup.gemm_config()` returns the effective pair alongside the tile; **build with it.** Assuming 8×4 at one of those shapes trips `M % (tile_m × herd_m) == 0` before anything compiles.

---

## Tolerances & reference (tier-aware)

Element-wise check over the **full output** against an f32 reference (cast to bf16 before compare): **every element must pass** `np.isclose(|a−e| ≤ atol + rtol·|e|)` (0% mismatch allowed).

| `--high-precision` | tier | rtol | atol | worst abs_err | margin |
|---|---|---|---|---|---|
| `true` (fused-cast / drain) | FP32-accumulate | 1.6e-2 | 1.5e-3 | ~6.1e-4 | ~2.5× |
| `false` (direct, per-L2-tile trunc) | bf16-accumulate | 1.6e-2 | 4e-3 | ~2.4e-3 | ~1.6× |

- **Reference** = CPU FP32 matmul cast to bf16. Inputs `randn / sqrt(K)` (variance-normalized).
- **`rtol = 1.6e-2` anchors to PyTorch's bf16 standard** (`torch.testing.assert_close`) for both tiers — the output is bf16, so per-element relative error is bounded by bf16 rounding (~2⁻⁸) regardless of accumulator precision.
- **`atol` encodes the precision tier** (measured worst-case `abs_err`, Down 2048×8192×2048). The high-precision `atol` (1.5e-3) is deliberately **below** the low-precision worst-case (2.4e-3): a high-precision path that silently regressed to bf16 truncation would **fail** the gate (the old shared 1.6e-2/4e-3 gate could not catch this). `mean_rel_L1` (printed) is diagnostic only.

---

## Tested shapes

> **Machine-readable source of truth: [`GEMM_bf16_in_bf16_out.json`](GEMM_bf16_in_bf16_out.json).**
> The tables below mirror that JSON. Model code (llama) reads the JSON via
> `kernel_registry/registry_lookup.py` `gemm_config(M,K,N,"bf16",precision)` to pick
> the method + tile sizes per shape — tiles are never hand-copied into the model.
> When you add/re-measure a shape, update the JSON first, then sync these tables.

Shapes cover LLM weight-projection shapes (the four 2048-row entries) and a square K-sweep (512³→4096³). Tiles are the fastest legal tile from an external-path sweep (full-chip herd 8×4). Measured on NPU2 (RyzenAI-npu4), June 2026, with this example's `run.py`, all PASS.

> **GFLOPS for fused-cast includes the cast launch** (`2·M·K·N / total`), so it is the true end-to-end bf16-out throughput. **Bold** = faster of the two high-precision methods, which `auto` picks.

### high-precision, `--method fused-cast` (tile_m=64) — fastest at large shapes

| (M, K, N) | tile (m/kl2/kl1/n) | GFLOPS (incl. cast) | mean_rel_L1 | Status |
|---|---|---|---|---|
| 2048×2048×2048 | 64/512/32/128 | **6215** | 9.7e-3 | ✅ |
| 2048×2048×512 | 64/256/32/128 | 4083 | 9.7e-3 | ✅ |
| 2048×2048×8192 | 64/256/32/128 | **6893** | 9.7e-3 | ✅ |
| 2048×8192×2048 | 64/256/32/128 | **8898** | 9.7e-3 | ✅ |
| 512×512×512 | 32/256/32/128 | 482 | 9.7e-3 | ✅ |
| 1024×1024×1024 | 64/256/32/128 | 2502 | 9.9e-3 | ✅ |
| 4096×4096×4096 | 64/512/32/128 | **8423** | 9.9e-3 | ✅ |
| 2048×1024×2048 | 64/256/32/128 | 4425 | 9.9e-3 | ✅ Qwen3-0.6B Q proj |
| 2048×2048×1024 | 64/256/32/128 | 5392 | 9.7e-3 | ✅ Qwen3-0.6B O proj |
| 2048×3072×1024 | 64/256/32/128 | 6461 | 9.9e-3 | ✅ Qwen3-0.6B Down proj |
| 2048×4864×896 | 64/256/32/32 | 3640 | 9.8e-3 | ✅ Qwen2.5-0.5B Down proj (N=896→TILE_N=32 HERD_N=4) |
| 2048×1536×1536 | 64/256/32/128 | 4821 | 9.7e-3 | ✅ Qwen2.5-1.5B Q/O proj (N=1536=512·3→default TILE_N=128) |
| 2048×8960×1536 | 64/256/32/128 | 8804 | 9.7e-3 | ✅ Qwen2.5-1.5B Down proj (N=1536→TILE_N=128; K=8960 tile_k_l2=256) |
| 2048×3072×3072 | 64/256/32/128 | 7513 | 9.9e-3 | ✅ Llama-3.2-3B Q/O proj (square) |
| 2048×3072×8192 | 64/256/32/128 | 7601 | 9.9e-3 | ✅ Llama-3.2-3B Gate/Up proj |
| 2048×8192×3072 | 64/256/32/128 | 9092 | 9.7e-3 | ✅ Llama-3.2-3B Down proj |
| 2048×2560×1024 | 64/256/32/128 | 6049 | 9.8e-3 | ✅ Qwen3-4B O proj (emb=2560) |
| 2048×2560×4096 | 64/256/32/128 | 7034 | 9.8e-3 | ✅ Qwen3-4B Q proj (emb=2560→4096) |
| 2048×2560×9728 | 64/**64**/32/128 | 5528 | 9.8e-3 | ✅ Qwen3-4B Gate/Up (N=9728: tile_k_l2≥128 DMA-stride fails; tile_k_l2=64 keeps high-prec, beats low) |
| 2048×4096×2560 | 64/256/32/128 | 7560 | 9.9e-3 | ✅ Qwen3-4B O proj alt (4096→2560) |
| 2048×9728×2560 | 64/256/32/128 | 8633 | 9.8e-3 | ✅ Qwen3-4B Down proj (9728→2560) |

### high-precision, `--method drain` (tile_m=32) — fastest at small / thin shapes

| (M, K, N) | tile (m/kl2/kl1/n) | GFLOPS | mean_rel_L1 | Status |
|---|---|---|---|---|
| 2048×2048×2048 | 32/512/32/128 | 6025 | 9.3e-3 | ✅ |
| 2048×2048×512 | 32/256/32/128 | **5626** | 9.3e-3 | ✅ |
| 2048×2048×8192 | 32/256/32/128 | 5784 | 9.3e-3 | ✅ |
| 2048×8192×2048 | 32/256/32/128 | 7234 | 9.3e-3 | ✅ |
| 512×512×512 | 32/256/32/128 | **1703** | 9.3e-3 | ✅ |
| 1024×1024×1024 | 32/256/32/128 | **4637** | 9.5e-3 | ✅ |
| 4096×4096×4096 | 32/512/32/128 | 7002 | 9.4e-3 | ✅ |
| 2048×1024×1024 | 32/256/32/128 | 4980 | 9.4e-3 | ✅ Qwen3-0.6B K/V proj (auto→drain; fused-cast over-allocs L1) |
| 2048×896×896 | 32/128/32/32 | 2516 | 9.4e-3 | ✅ Qwen2.5-0.5B Q/O proj (N=896→TILE_N=32 HERD_N=4; HERD_N=1 fails at runtime) |
| 2048×896×128 | 32/128/32/32 | 1890 | 9.4e-3 | ✅ Qwen2.5-0.5B K/V proj (thin N=128=4·32) |
| 2048×1536×256 | 32/256/32/64 | 3770 | 9.3e-3 | ✅ Qwen2.5-1.5B K/V proj (thin N=256=4·64→TILE_N=64) |

### low-precision (`--high-precision false`), direct-codegen bf16

| (M, K, N) | tile (m/kl2/kl1/n) | GFLOPS | mean_rel_L1 | abs_err max | Status |
|---|---|---|---|---|---|
| 2048×2048×2048 | 64/512/32/128 | 5230 | 1.3e-2 | 2.9e-3 | ✅ |
| 2048×2048×512 | 64/256/32/128 | 4765 | 1.3e-2 | 2.9e-3 | ✅ |
| 2048×2048×8192 | 64/256/32/128 | 5287 | 1.3e-2 | 3.2e-3 | ✅ |
| 2048×8192×2048 | 64/256/32/128 | 5592 | 1.9e-2 | 2.4e-3 | ✅ |
| 512×512×512 | 64/256/32/128 | 1750 | 1.0e-2 | 2.9e-3 | ✅ |
| 1024×1024×1024 | 64/256/32/128 | 4456 | 1.1e-2 | 2.4e-3 | ✅ |
| 4096×4096×4096 | 64/512/32/128 | 5509 | 1.5e-2 | 2.9e-3 | ✅ |
| 2048×1024×3072 | 64/256/32/128 | 5006 | 1.1e-2 | 2.9e-3 | ✅ Qwen3-0.6B Gate/Up proj |
| 2048×896×4864 | 64/128/32/64 | 4320 | 1.1e-2 | 2.9e-3 | ✅ Qwen2.5-0.5B Gate/Up proj (N=4864→TILE_N=64 HERD_N=4; high-prec atol artifact, see note) |
| 2048×1536×8960 | 64/128/32/64 | 4165 | 1.2e-2 | 3.4e-3 | ✅ Qwen2.5-1.5B Gate/Up proj (N=8960→TILE_N=64; high-prec fused-cast compile-fails L1, low-prec needs tile_k_l2=128) |
| 2048×2048×11008 | 64/128/32/64 | 4276 | 1.3e-2 | — | ✅ Qwen2.5-3B Gate/Up proj (N=11008→TILE_N=64; tile_k_l2=256 DMA-stride fails, needs tile_k_l2=128) |
| 2048×2560×9728 | 64/128/32/64 | 4397 | 1.4e-2 | — | ✅ Qwen3-4B Gate/Up alt (low-prec; high-prec fused-cast tile_k_l2=64 is faster+more accurate, see fused-cast table) |

> **Qwen2.5-1.5B note — 1536 is 512-aligned.** Unlike Qwen2.5-0.5B (896), Qwen2.5-1.5B's emb/q_dim = 1536 = 512·3 is divisible by the default `4·TILE_N = 512`, so **Q/O/Down (N=1536) place at the stock `TILE_N=128 HERD_N=4`** with no shrink. Only the thin **K/V (N=256 → TILE_N=64, drain)** and the wide **Gate/Up (N=8960 → TILE_N=64)** drop below 128. K=1536→`tile_k_l2=256` (1536/256=6); K=8960→`tile_k_l2=256` (8960/256=35). **Gate/Up (2048×1536×8960)**: the high-precision fused-cast at TILE_M=64/TILE_N=64 over-allocates L1 (NPU lowering pipeline fail); the low-precision direct path PASSES but only with `tile_k_l2=128` (tile_k_l2=256 also compile-fails at this N), at 1.2e-2 — the same Gate/Up tier-down as the smaller Qwen siblings. No padding was required for any Qwen2.5-1.5B shape.

> **Qwen2.5-0.5B note — non-512-aligned N.** Qwen2.5's projection widths (896, 128, 4864) are not divisible by the default `4·TILE_N = 512`, and `HERD_N=1` (e.g. `TILE_N=128` for N=896) **fails at runtime** (`qds_device::wait() unexpected command state` — the fused-cast/drain paths assume the 8×4 array). Working config: keep `HERD_N=4`, shrink `TILE_N` so `4·TILE_N | N` (TILE_N=32 for N∈{896,128}, TILE_N=64 for N=4864). K=896→`tile_k_l2=128`, K=4864→`tile_k_l2=256`. The thin Q/O/K/V shapes need `--method drain` (`tile_m=32`; `tile_m=64` over-allocates L1). **Gate/Up (2048×896×4864)** is the same near-zero-reference atol artifact as Qwen3's Gate/Up: high-precision computes the in-tier result (9.4e-3) but the harness gate trips on 2 near-zero elements (abs_err ≈ 1.6–1.9e-3 > high-prec `atol = 1.5e-3`); PASSES on the low-precision direct path (`atol = 4e-3`). No padding was required for any Qwen2.5 shape.

> **Qwen3-4B note — emb=2560 family.** All five Qwen3-4B projection shapes (2048×2560×{1024,4096,9728}, 2048×4096×2560, 2048×9728×2560) keep the **high-precision fused-cast** tier at the stock `TILE_N=128 HERD_N=4` (2560=512·5 and 9728=512·19 are both 512-aligned, no TILE_N shrink). The one wrinkle is the wide **Gate/Up (2048×2560×9728)**: at `tile_k_l2≥128` aiecc fails the `aie.dma_bd` stride check (stride = `tile_k_l2·N`; 256·9728=2490368 and 128·9728=1245184 both exceed the [1:1048576] range) — the same DMA-stride limit as Qwen2.5-3B Gate/Up. But unlike that sibling, here `tile_k_l2=64` (64·9728=622592 < limit) **places and keeps the high-precision tier** at 5528 GFLOPS / 9.8e-3, which is **strictly better than the low-precision direct path** (4397 GFLOPS / 1.4e-2). So best.high=fused-cast(tile_k_l2=64); the low-precision direct row is recorded for reference only. `2048×4096×2560` and `2048×9728×2560` (N=2560) place cleanly at tile_k_l2=256 (tile_k_l2=512 over-runs the DMA stride at N=2560).

> **Qwen3-0.6B note.** For the Qwen3 projection shapes only the `auto`-selected high-precision method was swept (the other method's cell is left blank in the index). **Gate/Up (2048×1024×3072)** is listed here under low-precision: both high-precision methods compute the in-tier result (mean_rel_L1 = 9.4e-3) but the harness element-wise gate trips on a single near-zero-reference output element (abs_err ≈ 1.7e-3 > the high-precision `atol = 1.5e-3`); the shape PASSES on the low-precision direct path (`atol = 4e-3`). This is a harness tolerance edge, not a datapath failure — the same fused-cast/drain datapath passes for Q, O, and Down. If the high-precision tier is required for Gate/Up, relax the harness high-precision `atol` to match the other tiers (does not touch the GPU-standard `rtol`).

**Reading the tables**:
- **The `auto` cross-over holds**: fused-cast wins large (Down 8898, Gate/Up 6893, O 6215, 4096³ 8423), drain wins small/thin (K/V 5626, 1024³ 4637, 512³ 1703). The `M*K*N ≥ 4e9` threshold picks the bold winner for all 7 shapes. At the tiniest shape (512³) low-precision direct (1750) edges drain (1703) — but it's a precision tier worse; `--high-precision true` stays the safe default.
- **Precision**: high-precision (fused/drain) holds the f32-accumulate tier (9.3–9.9e-3) at every shape — the single epilogue cast preserves it. Low-precision direct-bf16 degrades with the L2-tile count (`K / tile_k_l2`): Down (32 truncations) reaches 1.9e-2; O/Gate (4–8) stay ~1.3e-2.
- **vs the [f32-out page](GEMM_bf16_in_fp32_out.md)**: the bf16 epilogue cast costs ~7% on Down (fused 8898 vs external-f32 9797). If the consumer can take f32, skip the cast; if it needs bf16 at GPU precision, fused-cast delivers it.

---

<!-- BEGIN transformer-layer-sweep baseline_768 -->
### Transformer-layer execution study — `baseline_768` sweep

Written by `programming_examples/transformer_layer/sweep/registry_sweep.py`, which
measures every candidate tiling for a shape and keeps the fastest that passes. **Bold**
= the winner recorded in `best` for that tier. `—` = no legal candidate for that method
at this shape; `❌` = every candidate for it failed.

Two things differ from the model-deployment rows above, both forced by the sequence
ladder starting at 64:

- **`herd` is per-row.** `M % (tile_m × herd_m) == 0` cannot hold at `M = 64` with
  `herd_m = 8` for either method's forced `tile_m`, so short-sequence rows carry a
  per-method `herd` in the JSON that overrides the file-level `8×4`. The herd used is in
  the table's tile column.
- **The high-precision `atol` is carried forward at constant strictness, not held
  constant.** The harness scales its inputs by `1/sqrt(K)`, so the output magnitude —
  and the absolute error of a fixed-relative-precision datapath with it — goes as
  `K^-1/2`. The published `atol = 1.5e-3` is that rule evaluated at `K = 8192`; at this
  family's `K = 768` it is a 3.3× *tightening*, which is exactly the "harness tolerance
  edge, not a datapath failure" already recorded for Qwen3-0.6B Gate/Up and
  Qwen2.5-0.5B Gate/Up above. The sweep therefore checks the high tier at
  `1.5e-3 × sqrt(8192 / K)`, which reproduces the registry's own ~2.5× design margin at
  every K (measured: 2.5× at K=8192, 2.9× at K=3072, 2.8× at K=768). `rtol` is the
  canonical `1.6e-2` unchanged, the low tier's `4e-3` is unchanged, and every run must
  additionally land inside its tier's `mean_rel_L1` band — a scale-free check the
  example harness does not make at all.

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

## How to reproduce (correctness + performance, one command)

`make run` drives `run.py --compile-mode compile-and-run` (correctness + `--perf-iters` timing in one invocation). Example: the 2048×8192×2048 (Down) row.

```bash
cd programming_examples/matrix_multiplication/bf16_in_bf16_out

# ---- high-precision, auto method (Down → fused-cast) ----
make run HIGH_PRECISION=true METHOD=auto M=2048 K=8192 N=2048 \
  TILE_M=64 TILE_K_L2=256 TILE_K_L1=32 TILE_N=128 AIE_TARGET=aie2p PERF_ITERS=20

# ---- high-precision, drain (tile_m=32) ----
make run HIGH_PRECISION=true METHOD=drain M=2048 K=8192 N=2048 \
  TILE_M=32 TILE_K_L2=256 TILE_K_L1=32 TILE_N=128 AIE_TARGET=aie2p PERF_ITERS=20

# ---- low-precision direct-codegen bf16 ----
make run HIGH_PRECISION=false M=2048 K=8192 N=2048 \
  TILE_M=64 TILE_K_L2=256 TILE_K_L1=32 TILE_N=128 AIE_TARGET=aie2p PERF_ITERS=20
```

For another shape/tile from the tables, change `M/K/N` and the `TILE_*` vars to that row (drain rows are tile_m=32).

Notes:
- The lit tests cover every option: `run_npu2_high_precision_peano.lit` (auto), `..._fused_peano.lit`, `..._drain_peano.lit`, `run_npu2_low_precision_peano.lit` — each pins the Down shape and asserts `PASS!`.
- fused-cast forces ELF output (`instance_name=gemm_cast_bf16`); drain/direct use the default xclbin. (A wrong output_format mis-runs a multi-launch module → all-wrong output.)
- If the NPU is shared, serialize on-device runs (e.g. with `flock`) so timing isn't perturbed.
