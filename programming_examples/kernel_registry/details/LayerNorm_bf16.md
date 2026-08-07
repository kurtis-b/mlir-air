<!---//===- LayerNorm_bf16.md ---------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//-->

# Multi-row LayerNorm (BF16) — Kernel Detail

> Unweighted layer normalization `y = (x − mean(x)) · rsqrt(var(x) + eps)`, per row, over **several rows per kernel call** — the form an encoder-style transformer block wants when a whole activation tile is resident in L1. BF16 in/out; the statistics accumulate in **FP32** and the variance is **two-pass**, so rows at a large common offset normalize correctly.
> Shapes are written **`M×N`**: input `x[M, N]`, output `y[M, N]` (M = rows / seq, N = embedding dim, the normalization axis).
>
> Companion: [`../supported_kernels.md`](../supported_kernels.md) · [`../README.md`](../README.md) · [`AddNorm_bf16.md`](AddNorm_bf16.md) (the weighted, residual-adding form)
> **Scope: NPU2 (Strix / AIE2P) only.** Measured on real NPU2, August 2026. Reproduce commands in "How to reproduce" below.

---

## Builder

```
programming_examples/transformer_layer/builders/layer_norm.py
  build_layer_norm_module(rows, cols, np_dtype=bfloat16, herd_x=8, rows_per_call=8)
  layer_norm_reference(x, eps=1e-5)          # the FP32 two-pass oracle
```

Driven by `transformer_layer/opcheck.py --operator layer_norm`; `make check-layer-norm` in `transformer_layer/` is the same thing behind the lit test. The builder links the external `layer_norm.o`, compiled from `programming_examples/layer_norm/layer_norm.cc` (`external_kernels.compile_layer_norm`), and calls its `layer_norm_rows(input, output, cols, rows_to_process)` entry point once per resident tile. Rows are split contiguously across an `herd_x × 1` AIE grid; each tile streams `rows_per_call` rows L3→L1, normalizes them in one call, and streams them back.

**This is not `programming_examples/layer_norm/layer_norm.py`.** That example is direct-codegen from the vector dialect, one row per launch, with its statistics accumulated in **bf16**, and it gates at `rtol = 5e-2, atol = 5e-1` — an order of magnitude looser than anything else in this registry. The two are not bit-equivalent and neither is the oracle for the other.

---

## Numerical datapath (what "BF16 multi-row LayerNorm" means here)

```
pass 1  x bf16 → widen to f32 → Σx (f32 vector accumulate) → mean  (f32)
pass 2  d = x − mean (exact in f32) → bf16(d)² (f32 accumulate)
                                    → var = E[(x−mean)²]  (f32, clamped ≥ 0)
                                    → inv_std = aie::invsqrt(var + 1e-5)  (f32)
pass 3  (x − mean) · inv_std  (f32, one rounding to bf16 at the store)
```

- **Variance is two-pass**, `E[(x − mean)²]`, and **every statistic accumulates in f32**. The kernel shipped one-pass (`E[x²] − E[x]²`, bf16 row sum) until J7a's round-3 review showed that form losing the variance entirely on a row whose mean is large next to its spread — it cancels below zero, clamps, and normalizes by `1/sqrt(eps)`, ~700 of every 768 elements outside tolerance at mean/σ = 32 — while the bf16 row sum put the mean itself off by whole ulps of the *sum*. Offset rows are valid inputs; do not reintroduce either. The zero-clamp before `invsqrt` stays (a negative operand returns NaN and would poison the row), though the two-pass form is non-negative by construction.
- **The deviations are exact.** Every bf16 value is exact in f32, so `x − mean` in f32 carries only f32 rounding; the deviation rounds to bf16 only for the squaring, ~2⁻⁹ relative on the variance at any mean.
- **The epilogue rounds once.** `(x − mean) · inv_std` is computed in f32 and rounds to bf16 at the store — one rounding where the one-pass kernel took three.

---

## Numerical accuracy

Verified element-wise over the full output against the **two-pass FP32** reference (the same form the kernel now computes, in the same order, at f32 precision — the measurement is the bf16 rounding, not a formula gap):

| Metric (M×N = 512×512, `randn` inputs, seed 2) | Measured |
|---|---|
| `mean_rel_L1 = mean｜y−ref｜ / mean｜ref｜` | **8.1e-5** |
| `rel_err max` | 7.8e-3 |
| `abs_err max` | 1.6e-2 |
| mismatches at `rtol=1.6e-2, atol=5e-2` | **0 / 262144** |

- **`mean_rel_L1 = 8.1e-5`** is the cleanest reduction in the registry — a single bf16 rounding of an f32-exact value, which is the floor for a bf16-out kernel. The one-pass kernel this replaced measured 2.0e-3 on the same inputs; the 25× gap was its bf16 row sum and three-rounding epilogue.
- **`rel_err max = 7.8e-3 < rtol`**: every element is covered by `rtol` alone (`atol_required` measures 0.0). One rounding of an f32 value is always within 2⁻⁹ relative, so near-zero outputs no longer carry the whole neighborhood's absolute error the old epilogue gave them.
- **`abs_err max = 1.6e-2`** is one bf16 ULP at the largest-magnitude outputs (|y| ≈ 4 σ), covered by `atol = 5e-2`.

---

## Tunable parameters

| Knob | Value | Constraint → source |
|---|---|---|
| `herd_x` | **8** | AIE columns (≤ 8). Two shim DMAs per tile (in, out), well inside a column's budget, so the full width places. `rows % (herd_x · rows_per_call) == 0` |
| `herd_y` | 1 (fixed) | not exposed; the rows are independent, so width is the only useful axis |
| `rows_per_call` | 8 | rows resident in L1 per `layer_norm_rows` call. `2 · rows_per_call · cols · 2` bytes must fit L1 with ping-pong |
| `cols` | multiple of **16** | `LN_VEC_LEN`. There is **no scalar tail** — a non-multiple silently drops the remainder |
| `eps` | 1e-5 | `kEpsilon` in `layer_norm.cc`; must match the reference |

Rows must be contiguous and exactly `cols` apart. A padded tile has to pass the padded stride as `cols`, which then normalizes over the padding too.

---

## Tolerances & reference

Element-wise over the **full output**: every element must pass `|y−ref| ≤ atol + rtol·|ref|`, with zero permitted mismatches.

| Output dtype | rtol | atol |
|---|---|---|
| bf16 | 1.6e-2 | 5e-2 |

- **Reference** = CPU FP32 **two-pass** LayerNorm (`mean`, then `mean((x−mean)²)`, then `(x−mean)/sqrt(var+eps)`), from bf16-rounded `randn` inputs, cast once to bf16. Not a bf16 reference: a bf16 oracle agrees with a bf16 device partly by being wrong in the same direction.
- `rtol = 1.6e-2` is PyTorch / vLLM's canonical bf16 tolerance and is held fixed across the registry. `atol = 5e-2` covers the worst-case single-element bf16 output rounding (`abs_err max ≈ 1.6e-2`).
- **Matches the GPU op in structure.** `torch.nn.LayerNorm` on bf16 upcasts to f32 for the statistics and rounds once; this kernel now does the same (f32 statistics, two-pass variance, one rounding at the store). The measured 8.1e-5 is the size of the remaining gap — the bf16 rounding of the squared deviations.

---

## Tested shapes

| (M×N) | herd (hx/hy) | rows_per_call | mean_rel_L1 | abs_err max | mismatches | Used by | Status |
|---|---|---|---|---|---|---|---|
| 512×512 | 8/1 | 8 | 8.1e-5 | 1.6e-2 | 0 / 262144 | transformer-layer execution studies, encoder block norm (hidden = 512) | ✅ |
| 4096×768 | 8/1 | 8 | 7.1e-5 | 1.6e-2 | 0 / 3145728 | transformer-layer execution studies, `baseline_768` block norm at the block's own sequence length | ✅ |

> The `baseline_768` row is Phase D1's, added so a block failure localizes to the integration rather than to an operator nobody had run at that width. `mean_rel_L1` is unchanged across a 12× larger output and a 1.5× wider normalization axis, which is what a per-row reduction should do. Only the `baseline_512` and `baseline_1024` families remain unrun here; a row appears only once it has been run on hardware. (Both rows re-measured August 2026 on the two-pass f32-statistics kernel; the one-pass kernel measured 2.0e-3 / 3.1e-2 on the same seeds.)
>
> **The two rows carry different `atol`** — 5e-2 and 5e-3. The 512-row's was the tier's default, chosen before `atol_required` was recorded; the 768-row's was sized from the one-pass kernel's measured `atol_required` of 1.4e-3 rounded up 3.5×, and stands although the two-pass kernel's `atol_required` measures 0.0 — a bound `rtol` alone now meets is not a reason to loosen or to chase an arbitrarily small number.
>
> The offset-row regime (mean large next to spread, where the one-pass form lost the variance entirely) is pinned by norm_tail's `128x768_offset` opcheck row, which exercises this same `layer_norm_rows` entry point through the J7a pipeline.

**Performance is not measured here.** C1 gates numerics only, so the latency / bandwidth columns the other kernels in this registry carry are deliberately absent rather than estimated. LayerNorm is memory-bound in the same way RMSNorm is (it streams the whole matrix for an O(N) op), so its bandwidth should land in the same band; that is an expectation, not a measurement, and it is not recorded as one.

---

## How to reproduce

```bash
cd programming_examples/transformer_layer

# correctness on real NPU2. Serialize on the repository lock, which is a
# DIFFERENT inode from the /tmp/npu.lock the runner takes internally.
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  make check-layer-norm PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR

# the negative control: perturbs one element of the DEVICE input after the
# reference is computed, and MUST fail.
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  make check-layer-norm OPCHECK_ARGS="--fault-inject input" \
       PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR

# every (operator, shape) the port claims, as JSON. No NPU.
make opcheck-list
```

Each run writes `transformer_layer/results/layer_norm__<shape>.json`, carrying the tolerances used, `ref_dtype`, the three error statistics, `n_elements` / `n_mismatch`, and the verdict. Injected runs write into `results/fault/` instead, so they can never be mistaken for — or overwrite — a clean one.
