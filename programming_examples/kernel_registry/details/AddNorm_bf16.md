<!---//===- AddNorm_bf16.md -----------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//-->

# AddNorm (BF16) — Kernel Detail

> Weighted layer normalization with a residual added after it, fused into one kernel call:
> `out = LayerNorm(x) · weight + residual`, per row. The sublayer boundary of an encoder-style transformer block. BF16 in/out, one-pass variance, **weight as a runtime memref argument**.
> Shapes are written **`M×N`**: `x[M, N]`, `residual[M, N]`, `weight[N]` → `out[M, N]` (M = rows / seq, N = embedding dim, the normalization axis).
>
> Companion: [`../supported_kernels.md`](../supported_kernels.md) · [`../README.md`](../README.md) · [`LayerNorm_bf16.md`](LayerNorm_bf16.md) (the unweighted, no-residual form)
> **Scope: NPU2 (Strix / AIE2P) only.** Measured on real NPU2, August 2026. Reproduce commands in "How to reproduce" below.

---

## Builder

```
programming_examples/transformer_layer/builders/addnorm.py
  build_addnorm_module(rows, cols, np_dtype=bfloat16, herd_x=8, rows_per_call=None)
  addnorm_reference(x, residual, weight, eps=1e-5)     # the FP32 two-pass oracle
```

Driven by `transformer_layer/opcheck.py --operator addnorm`; `make check-addnorm` is the same thing behind the lit test. The builder links `encoder.o` — built with the **addnorm half only** (`compile_encoder(build_ffn=False)`), because the FFN half also defines `ffn_gelu_bf16` and `ffn_eltwise_add_bf16_vector` and would collide with `addnorm_ffn.o` — and calls `fused_add_layer_norm_2outs(input, residual, weights, output1, output2, cols, rows_to_process)`.

**The weight is a runtime argument.** iron's `addnorm` bakes its weights into the MLIR via `np.load()` at generation time and hashes them into the artifact name, so every weight change forces a recompile. Here `weight` is a plain memref argument and one compiled ELF serves every weight vector of that shape.

**Two outputs, one drained.** The kernel writes its result twice because the fused encoder block feeds it to both the FFN and the next residual. Only `output1` reaches L3; `output2` stays in L1. A second L3 output would need a second shim S2MM channel per column for a byte-identical copy.

---

## Numerical datapath (what "BF16 AddNorm" means here)

```
x bf16 → Σx  (bf16 vector accumulate, 32 lanes) ─┐
      → Σx² (f32 vector accumulate)              ├→ mean, var = E[x²] − E[x]²  (f32, clamped ≥ 0)
                                                  └→ inv_std = aie::invsqrt(var + 1e-5)  (f32)
   → ((x − bf16(mean)) · bf16(inv_std)) · weight + residual   (bf16 vector) → bf16
```

- **Statistics come from `x` alone.** The residual never enters the mean or the variance; it is added after the weighted normalization. This is the **post-add** form. (`addnorm_ffn.cc` has a pre-add variant behind `-DADDNORM_PRE_ADD`, which folds the residual in *before* the statistics; that is a different operator and is not what this entry measures.)
- **Variance is one-pass**, `E[x²] − E[x]²`, clamped at zero before `invsqrt` — same formula, same cancellation caveat and same NaN-avoidance clamp as [`LayerNorm_bf16.md`](LayerNorm_bf16.md).
- **Four bf16 roundings in the epilogue**: `sub`, `mul` by `inv_std`, `mul` by `weight`, `add` of the residual. The residual add is where an output can land near zero through cancellation, which is what sets `rel_err max`.

---

## Numerical accuracy

Verified element-wise over the full output against the **two-pass FP32** reference:

| Metric (M×N = 64×512, `randn` x/residual, `uniform(0.5, 1.5)` weight, seed 3) | Measured |
|---|---|
| `mean_rel_L1 = mean｜out−ref｜ / mean｜ref｜` | **1.9e-3** |
| `rel_err max` | 6.8e+1 |
| `abs_err max` | 3.1e-2 |
| mismatches at `rtol=1.6e-2, atol=5e-2` | **0 / 32768** |

- **`mean_rel_L1 = 1.9e-3`**, level with Element-wise Add and below LayerNorm's 2.0e-3 — the residual add raises the typical output magnitude, so the same absolute rounding is a smaller fraction of it.
- **`rel_err max = 6.8e+1` is expected.** `LayerNorm(x)·weight` and `residual` are independent and comparable in magnitude, so their sum lands arbitrarily close to zero somewhere in a 32768-element output; the relative error there is unbounded while the absolute error stays at one bf16 ULP. `atol` is what covers it.
- **`abs_err max = 3.1e-2`**, one bf16 ULP at the largest outputs, inside `atol = 5e-2`.
- The weight is drawn `uniform(0.5, 1.5)` rather than all-ones: a trained LayerNorm gamma sits near 1, and an all-ones weight would let a dropped weight multiply pass unnoticed.

---

## Parameters & constraints

| Knob | Value | Constraint → source |
|---|---|---|
| `herd_x` | **8** | AIE columns (≤ 8). `rows % herd_x == 0` |
| `herd_y` | 1 (fixed) | each tile already drives three L3→L1 streams; see below |
| `rows_per_call` | **must be `rows // herd_x`** | one kernel call per tile — a correctness constraint, not a tuning knob |
| `cols` | multiple of **32** | the kernel's `N`. A non-multiple is silently truncated by `vector_chunks = cols / N` |
| L1 | `4·rows_per_call·cols·2 + cols·2 + 1024` ≤ 64 KiB | four activation tiles (x, residual, out1, out2), the weight, and the stack |
| `eps` | 1e-5 | `epsilon` in `fused_add_layer_norm_2`; must match the reference |

### One kernel call per tile

`rows` must equal `herd_x · rows_per_call`, so the herd's loop runs a single trip. **Two or more trips miscompile.** Measured on NPU2 at `[8, 64]`, `herd_x = 1`: the one-trip form is exact (0 of 512 elements outside tolerance) and the two-trip form is garbage (491 of 512). It reproduces with the weight fetched inside the loop or hoisted out of it, with `output2` drained to L3 or discarded, with ping-pong disabled, and under both of the runner's lock-race-condition fixes.

The distinguishing feature is **three distinct L3→L1 streams per tile** (`x`, `residual`, `weight`) against the two shim MM2S channels an AIE2P column has. The two-stream builders beside it — multi-row LayerNorm here, and `_build_add_2d_to_2d` in `llms/shared/builders/o_ffn_multi.py` — loop correctly for as many trips as you like. `build_addnorm_module` **raises** rather than emitting the broken form, because the symptom is partly-correct values and reads as a tolerance problem rather than a scheduling one.

The consequence is a row cap: at `cols = 512` over the full 8-column herd, **64 rows**. A larger activation needs the weight staged through L2, or the residual folded into the same L3 buffer as `x` so one strided DMA fetches both. Neither is done here; both are open for the later sub-phases.

---

## Tolerances & reference

Element-wise over the **full output**: every element must pass `|out−ref| ≤ atol + rtol·|ref|`, with zero permitted mismatches.

| Output dtype | rtol | atol |
|---|---|---|
| bf16 | 1.6e-2 | 5e-2 |

- **Reference** = CPU FP32 **two-pass** LayerNorm, multiplied by the f32 weight and added to the f32 residual, cast once to bf16. Two-pass on purpose: the device's one-pass variance is exactly the error the check should be able to see, so reproducing it in the oracle would hide it.
- `rtol = 1.6e-2` is held fixed across the registry; `atol = 5e-2` covers the worst-case single-element bf16 output rounding (`abs_err max ≈ 3.1e-2`).

---

## Tested shapes

| (M×N) | herd (hx/hy) | rows_per_call | mean_rel_L1 | abs_err max | mismatches | Used by | Status |
|---|---|---|---|---|---|---|---|
| 64×512 | 8/1 | 8 | 1.9e-3 | 3.1e-2 | 0 / 32768 | transformer-layer execution studies, encoder sublayer boundary (hidden = 512) | ✅ |

> `M = 64`, not 512, because of the one-call-per-tile cap above — 8 columns × 8 rows of L1-resident activation at `cols = 512`. It is a real 64-token chunk, not a toy: the operator is row-independent, so a longer sequence is the same arithmetic issued more times. What is *not* yet demonstrated is issuing it more times from one ELF, and that is stated rather than implied.

**Performance is not measured here.** C1 gates numerics only; latency and bandwidth are deliberately absent rather than estimated.

---

## How to reproduce

```bash
cd programming_examples/transformer_layer

# correctness on real NPU2, serialized on the repository lock (a DIFFERENT
# inode from the /tmp/npu.lock the runner takes internally).
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  make check-addnorm PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR

# the negative control: perturbs one element of the DEVICE input after the
# reference is computed, and MUST fail.
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  make check-addnorm OPCHECK_ARGS="--fault-inject input" \
       PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR
```

Each run writes `transformer_layer/results/addnorm__<shape>.json`; injected runs write into `results/fault/` instead.
