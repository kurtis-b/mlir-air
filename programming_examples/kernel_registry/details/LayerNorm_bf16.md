<!---//===- LayerNorm_bf16.md ---------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//-->

# Multi-row LayerNorm (BF16) — Kernel Detail

> Unweighted layer normalization `y = (x − mean(x)) · rsqrt(var(x) + eps)`, per row, over **several rows per kernel call** — the form an encoder-style transformer block wants when a whole activation tile is resident in L1. BF16 in/out; the sum of squares accumulates in **FP32**, the row sum in bf16.
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
x bf16 → Σx  (bf16 vector accumulate, 16 lanes) ─┐
      → Σx²  (f32 vector accumulate)             ├→ mean, var = E[x²] − E[x]²  (f32, clamped ≥ 0)
                                                  └→ inv_std = aie::invsqrt(var + 1e-5)  (f32)
        → (x − bf16(mean)) · bf16(inv_std)  (bf16 vector) → bf16
```

- **Variance is one-pass**, `E[x²] − E[x]²`, so the row is read once for statistics rather than twice. It is algebraically equal to the two-pass form and numerically is not: it cancels catastrophically on a row whose mean is large next to its spread, and can round below zero — which is why the kernel clamps at zero before `invsqrt` (a negative operand there returns NaN and would poison the whole row).
- **The sum of squares accumulates in f32; the row sum does not.** `Σx` runs in a bf16 vector accumulator, so `mean` carries more error than an FP32 reduction would. On zero-mean activations `mean ≈ 0` and the resulting shift is negligible; on a large-mean row it is not, and neither is the cancellation above.
- **The epilogue is bf16.** `mean` and `inv_std` are computed in f32 and then broadcast as bf16, so the per-element normalization is three bf16 roundings deep (`sub`, `mul`, store).

---

## Numerical accuracy

Verified element-wise over the full output against the **two-pass FP32** reference — deliberately not the kernel's own one-pass formula, so the measurement includes the error the one-pass form introduces rather than cancelling it out:

| Metric (M×N = 512×512, `randn` inputs, seed 2) | Measured |
|---|---|
| `mean_rel_L1 = mean｜y−ref｜ / mean｜ref｜` | **2.0e-3** |
| `rel_err max` | 6.3e+1 |
| `abs_err max` | 3.1e-2 |
| mismatches at `rtol=1.6e-2, atol=5e-2` | **0 / 262144** |

- **`mean_rel_L1 = 2.0e-3`** sits in the cleanest tier of the registry, beside Element-wise Add (1.9e-3) and below RMSNorm (4.2e-3). A normalized output is O(1) by construction, so the bf16 epilogue roundings do not compound.
- **`rel_err max = 6.3e+1` is expected and is not a defect.** LayerNorm output is zero-mean, so some element of some row lands arbitrarily close to zero; its *relative* error is then unbounded while its *absolute* error stays at one bf16 ULP. This is exactly the case `atol` exists for, and it is why the registry's methodology fixes `rtol` and sizes `atol` rather than the reverse.
- **`abs_err max = 3.1e-2`** is one bf16 ULP at the largest-magnitude outputs (|y| ≈ 4 σ), covered by `atol = 5e-2`.

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
- `rtol = 1.6e-2` is PyTorch / vLLM's canonical bf16 tolerance and is held fixed across the registry. `atol = 5e-2` covers the worst-case single-element bf16 output rounding (`abs_err max ≈ 3.1e-2`).
- **Matches the GPU op in structure, not in bit pattern.** `torch.nn.LayerNorm` on bf16 upcasts to f32 for the statistics and rounds once; this kernel keeps `Σx²` in f32 but `Σx` in bf16 and rounds three times in the epilogue. The measured 2.0e-3 is the size of that gap.

---

## Tested shapes

| (M×N) | herd (hx/hy) | rows_per_call | mean_rel_L1 | abs_err max | mismatches | Used by | Status |
|---|---|---|---|---|---|---|---|
| 512×512 | 8/1 | 8 | 2.0e-3 | 3.1e-2 | 0 / 262144 | transformer-layer execution studies, encoder block norm (hidden = 512) | ✅ |
| 4096×768 | 8/1 | 8 | 2.0e-3 | 3.1e-2 | 0 / 3145728 | transformer-layer execution studies, `baseline_768` block norm at the block's own sequence length | ✅ |

> The `baseline_768` row is Phase D1's, added so a block failure localizes to the integration rather than to an operator nobody had run at that width. `mean_rel_L1` is unchanged across a 12× larger output and a 1.5× wider normalization axis, which is what a per-row reduction should do. Only the `baseline_512` and `baseline_1024` families remain unrun here; a row appears only once it has been run on hardware.
>
> **The two rows carry different `atol`** — 5e-2 and 5e-3. The 512-row's was the tier's default, chosen before `atol_required` was recorded; the 768-row's is its own measured `atol_required` of 1.4e-3 rounded up 3.5×. `abs_err max` is 22× that, all of it sitting on large-magnitude elements `rtol` already covers, which is the whole reason `abs_err max` is not the number to size an `atol` against.

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
