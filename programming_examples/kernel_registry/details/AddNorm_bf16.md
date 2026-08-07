<!---//===- AddNorm_bf16.md -----------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//-->

# AddNorm (BF16) — Kernel Detail

> Weighted layer normalization and a residual, fused into one kernel call, in either order:
> `out = LayerNorm(x) · weight + residual` (post-add, the default) or `out = LayerNorm(x + residual) · weight` (pre-add, `pre_add=True`), per row. The sublayer boundary of an encoder-style transformer block; a post-norm encoder such as `encoder_bert` wants the **pre-add** form. BF16 in/out, one-pass variance, **weight as a runtime memref argument**.
> Shapes are written **`M×N`**: `x[M, N]`, `residual[M, N]`, `weight[N]` → `out[M, N]` (M = rows / seq, N = embedding dim, the normalization axis).
>
> Companion: [`../supported_kernels.md`](../supported_kernels.md) · [`../README.md`](../README.md) · [`LayerNorm_bf16.md`](LayerNorm_bf16.md) (the unweighted, no-residual form)
> **Scope: NPU2 (Strix / AIE2P) only.** Measured on real NPU2, August 2026. Reproduce commands in "How to reproduce" below.

---

## Builder

```
programming_examples/transformer_layer/builders/addnorm.py
  build_addnorm_module(rows, cols, np_dtype=bfloat16, herd_x=8,
                       rows_per_call=None, pre_add=False)
  compile_addnorm_kernel(pre_add=False)                # puts the right object in the CWD
  addnorm_max_rows(cols, np_dtype, herd_x, pre_add)    # the L1 row cap at this width
  addnorm_reference(x, residual, weight, eps=1e-5)          # post-add FP32 oracle
  addnorm_pre_add_reference(x, residual, weight, eps=1e-5)  # pre-add FP32 oracle
```

Driven by `transformer_layer/opcheck.py --operator addnorm`; `make check-addnorm` is the same thing behind the lit test.

**`pre_add=` selects the object, the entry point and the L1 budget together** — it is not a flag on one kernel, because the two forms are in different translation units:

| `pre_add` | computes | object | source | entry point | L1 tile buffers |
|---|---|---|---|---|---|
| `False` | `LayerNorm(x) · weight + residual` | `encoder.o` | `encoder.cc` | `fused_add_layer_norm_2outs` | 4 |
| `True` | `LayerNorm(x + residual) · weight` | `addnorm_pre_add.o` | `addnorm_ffn.cc` + `-DADDNORM_PRE_ADD` | `fused_add_layer_norm_1outs` | 3 |

Both are built with the **addnorm half only** (`build_ffn=False`), because the FFN half also defines `ffn_gelu_bf16` and `ffn_eltwise_add_bf16_vector` and the two sources collide on them. `addnorm_pre_add.o` is deliberately **not** the compile gate's `addnorm_ffn_pre_add.o`, which is the full 11-symbol build — linking that would drag every FFN matmul microkernel into a core that never calls one.

**Pre-add is what a post-norm encoder needs.** `encoder_bert` normalizes after the residual add at both of its sublayer boundaries. The post-add form was the one iron's operator carried and the one validated first; they are different functions, and a block wired to the wrong one produces a plausible activation that is wrong by one residual add. Each has its own reference above — a single reference with an ordering branch is exactly as easy to read backwards as the kernel it checks.

**The two-output entry point is not needed for pre-add.** Under `-DADDNORM_PRE_ADD` its `output2` carries the raw `x + residual` sum forward as the *next* block's residual stream. `encoder_bert`'s second residual is `hidden`, the normalized output of the first norm, not that sum — so the one-output form carries everything the block needs and costs one fewer L1 tile buffer.

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

- The diagram is the **post-add** form: statistics come from `x` alone, the residual never enters the mean or the variance, and it is added after the weighted normalization. The **pre-add** form (`-DADDNORM_PRE_ADD`) folds `x + residual` in bf16 *before* Σx and Σx², normalizes that sum, and does **not** add the residual again — so its epilogue has three bf16 roundings rather than four, and none of them is a cancelling add. Both are measured in this entry.
- **Variance is one-pass**, `E[x²] − E[x]²`, clamped at zero before `invsqrt` — it cancels catastrophically on a row whose mean is large next to its spread, and the clamp avoids the NaN `aie::invsqrt` returns on a negative operand. This is the formula the multi-row `layer_norm_rows` kernel also carried until J7a's round-3 review moved that kernel to two-pass f32 statistics ([`LayerNorm_bf16.md`](LayerNorm_bf16.md) documents the offset-row regime that forced it); the fused kernels here still measure and gate the one-pass form, on zero-mean-ish activations where the cancellation does not bite.
- **Four bf16 roundings in the epilogue**: `sub`, `mul` by `inv_std`, `mul` by `weight`, `add` of the residual. The residual add is where an output can land near zero through cancellation, which is what sets `rel_err max`.

---

## Numerical accuracy

Verified element-wise over the full output against the **two-pass FP32** reference:

| Metric (`randn` x/residual, `uniform(0.5, 1.5)` weight, seed 3) | 64×512 post-add | 64×768 pre-add |
|---|---|---|
| `mean_rel_L1 = mean｜out−ref｜ / mean｜ref｜` | **1.9e-3** | **2.7e-3** |
| `rel_err max` | 6.8e+1 | 1.1e+2 |
| `abs_err max` | 3.1e-2 | 6.3e-2 |
| `atol_required = max(｜out−ref｜ − rtol·｜ref｜)` | 1.75e-2 | **6.65e-4** |
| mismatches at `rtol=1.6e-2` and the row's `atol` | **0 / 32768** (`atol=5e-2`) | **0 / 49152** (`atol=2e-3`) |

- **`mean_rel_L1 = 1.9e-3`**, level with Element-wise Add and below LayerNorm's 2.0e-3 — the residual add raises the typical output magnitude, so the same absolute rounding is a smaller fraction of it.
- **`rel_err max = 6.8e+1` is expected.** `LayerNorm(x)·weight` and `residual` are independent and comparable in magnitude, so their sum lands arbitrarily close to zero somewhere in a 32768-element output; the relative error there is unbounded while the absolute error stays at one bf16 ULP. `atol` is what covers it.
- **`abs_err max = 3.1e-2`**, one bf16 ULP at the largest outputs, inside `atol = 5e-2`.
- **The pre-add column's `atol_required` is 26× smaller** even though its relative error is higher and its `abs_err max` is twice as large. `atol_required` is the number that matters, and the ordering is the whole reason: post-add's trailing `+ residual` puts an absolute error the size of the *residual* onto elements whose own value has cancelled to near zero, which is precisely where `rtol` contributes nothing. Pre-add's errors are all proportional to the output carrying them, so `rtol` absorbs almost all of them and `abs_err max = 6.3e-2` — a bf16 ULP at the largest outputs, as above — needs no `atol` at all.
- The weight is drawn `uniform(0.5, 1.5)` rather than all-ones: a trained LayerNorm gamma sits near 1, and an all-ones weight would let a dropped weight multiply pass unnoticed.

---

## Parameters & constraints

| Knob | Value | Constraint → source |
|---|---|---|
| `herd_x` | **8** | AIE columns (≤ 8). `rows % herd_x == 0` |
| `herd_y` | 1 (fixed) | each tile already drives three L3→L1 streams; see below |
| `rows_per_call` | **must be `rows // herd_x`** | one kernel call per tile — a correctness constraint, not a tuning knob |
| `cols` | multiple of **32** | the kernel's `N`. A non-multiple is silently truncated by `vector_chunks = cols / N` |
| L1 | `T·rows_per_call·cols·2 + cols·2 + 1024` ≤ 64 KiB, `T` = 4 post-add / 3 pre-add | the activation tiles (x, residual, out1, and out2 only in the post-add form), the weight, and the stack. `addnorm_max_rows()` inverts this; it counts allocations, not aircc's ping-pong copies, so it is an upper bound |
| `eps` | 1e-5 | `epsilon` in `fused_add_layer_norm_2`; the same constant in both variants, and must match the reference |

### One kernel call per tile

`rows` must equal `herd_x · rows_per_call`, so the herd's loop runs a single trip. **Two or more trips miscompile.** Measured on NPU2 at `[8, 64]`, `herd_x = 1`: the one-trip form is exact (0 of 512 elements outside tolerance) and the two-trip form is garbage (491 of 512). It reproduces with the weight fetched inside the loop or hoisted out of it, with `output2` drained to L3 or discarded, with ping-pong disabled, and under both of the runner's lock-race-condition fixes.

The distinguishing feature is **three distinct L3→L1 streams per tile** (`x`, `residual`, `weight`) against the two shim MM2S channels an AIE2P column has. The two-stream builders beside it — multi-row LayerNorm here, and `_build_add_2d_to_2d` in `llms/shared/builders/o_ffn_multi.py` — loop correctly for as many trips as you like. `build_addnorm_module` **raises** rather than emitting the broken form, because the symptom is partly-correct values and reads as a tolerance problem rather than a scheduling one.

The consequence is a row cap: at `cols = 512` over the full 8-column herd, **64 rows**. A larger activation needs the weight staged through L2, or the residual folded into the same L3 buffer as `x` so one strided DMA fetches both. Neither is done here; both are open for the later sub-phases.

---

## Tolerances & reference

Element-wise over the **full output**: every element must pass `|out−ref| ≤ atol + rtol·|ref|`, with zero permitted mismatches.

| Output dtype | ordering | rtol | atol |
|---|---|---|---|
| bf16 | post-add | 1.6e-2 | 5e-2 |
| bf16 | pre-add | 1.6e-2 | 2e-3 |

- **Reference** — one per ordering, not one with a branch. Post-add: CPU FP32 **two-pass** LayerNorm of `x`, multiplied by the f32 weight, plus the f32 residual. Pre-add: the f32 sum `x + residual` normalized the same way and multiplied by the weight, with **no** second residual add. Both cast once to bf16 at the end. Two-pass on purpose: the device's one-pass variance is exactly the error the check should be able to see, so reproducing it in the oracle would hide it. The pre-add oracle likewise sums in f32 while the kernel sums in bf16, for the same reason.
- `rtol = 1.6e-2` is held fixed across the registry; each `atol` is that row's measured `atol_required` rounded up ~3×, per the registry methodology. Post-add's is set by its trailing residual add, not by the normalization.

---

## Tested shapes

| (M×N) | ordering | herd (hx/hy) | rows_per_call | mean_rel_L1 | abs_err max | atol_required | atol | mismatches | Used by | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| 64×512 | post-add | 8/1 | 8 | 1.9e-3 | 3.1e-2 | 1.75e-2 | 5e-2 | 0 / 32768 | transformer-layer execution studies, encoder sublayer boundary (hidden = 512) | ✅ |
| 64×768 | **pre-add** | 8/1 | 8 | 2.7e-3 | 6.3e-2 | 6.65e-4 | 2e-3 | 0 / 49152 | transformer-layer execution studies, `baseline_768` encoder sublayer boundary | ✅ |

> `M = 64`, not 512, because of the one-call-per-tile cap above — 8 columns × 8 rows of L1-resident activation. It is a real 64-token chunk, not a toy: the operator is row-independent, so a longer sequence is the same arithmetic issued more times. What is *not* yet demonstrated is issuing it more times from one ELF, and that is stated rather than implied.
>
> **The cap moves with `cols`, and it is derived rather than carried over.** `addnorm_max_rows(cols, ...)` returns `herd_x ×` the largest `rows_per_call` that fits L1: 120 at `cols = 512`, 80 post-add and 104 pre-add at `cols = 768` (pre-add allocates one fewer tile buffer, having only one output). Those count *allocations*; aircc ping-pongs the DMA-fed buffers on top, so the cap is an upper bound and not a target. Both rows run 64, well under, which also keeps the two measurements comparable.
>
> **The pre-add row needs 26× less `atol` at a higher relative error.** Post-add's trailing `+ residual` in bf16 puts an absolute error set by the residual's magnitude onto elements whose own value has cancelled to near zero, and `rtol` covers none of that. Pre-add has no trailing add, so every error is proportional to the output carrying it. Same kernel, cancellation removed — not a better datapath.

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
