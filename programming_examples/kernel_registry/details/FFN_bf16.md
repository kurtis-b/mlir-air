<!---//===- FFN_bf16.md ---------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//-->

# Feed-Forward Network, GeLU (BF16) — Kernel Detail

> The encoder-style transformer block's feed-forward sublayer, staged as up-projection → GeLU → down-projection in one ELF: `y = gelu(x[M, K] @ w_up[K, F]) @ w_down[F, K]`.
> Shapes are written **`M×K×F`** (M = rows / seq, K = embedding dim, F = the FFN inner dim).
>
> Companion: [`../supported_kernels.md`](../supported_kernels.md) · [`../README.md`](../README.md) · [`GEMM_bf16_in_bf16_out.md`](GEMM_bf16_in_bf16_out.md) (the two GEMMs) · [`SiLU_Mul_bf16.md`](SiLU_Mul_bf16.md) (the gated-FFN activation, a different operator)
> **Scope: NPU2 (Strix / AIE2P) only.** Measured on real NPU2, August 2026. Reproduce commands in "How to reproduce" below.

---

## Builder

```
programming_examples/transformer_layer/builders/ffn.py
  build_ffn_module(seq_len, emb_dim, ffn_dim, herd_m=8, herd_n=4,
                   gelu_herd_x=8, gelu_tile_n=None, gemm_spec_fn=None)
  ffn_arg_layout(...) / ffn_device_inputs(...)   # the signature is COMPUTED, see below
  ffn_reference(x, w_up, w_down)                 # the FP32 oracle

programming_examples/transformer_layer/builders/gelu.py
  build_gelu_module(rows, cols, np_dtype=bfloat16, herd_x=8, tile_n=None)
  gelu_tanh_reference(x)
```

Driven by `transformer_layer/opcheck.py --operator ffn`; `make check-ffn` is the same thing behind the lit test.

**Split along the staging seam.** iron's `ffn/design.py` is 1096 lines carrying both projections and the activation in one file. The activation is the one stage that is neither a GEMM nor registry-tiled, so it lifts out into `gelu.py` and what remains in `ffn.py` is composition. Porting convention 5.

**The signature is computed, not written out.** How many memref arguments there are depends on which method the registry picks for each GEMM — a `fused-cast` GEMM needs an f32 C-scratch argument and a `drain` GEMM does not. `ffn_arg_layout` builds the signature and `ffn_device_inputs` builds the matching host-side list, so a registry update that flips a method cannot leave a caller passing buffers in the wrong order.

---

## Datapath

Five `air.launch` operations in one ELF, at the shape below:

```
1. up GEMM     x[M,K] @ w_up[K,F]  → H_f32[M,F]     external mm.o, f32 accumulate
2. up cast     H_f32[M,F]          → h[M,F] bf16
3. GeLU        h[M,F]              → a[M,F] bf16    ffn_gelu_bf16, tanh approximation
4. down GEMM   a[M,F] @ w_down[F,K]→ Y_f32[M,K]     external mm.o, f32 accumulate
5. down cast   Y_f32[M,K]          → y[M,K] bf16
```

`h` and `a` are real DDR buffers in the signature, handed to the device zero-filled. They are staging, not outputs: the gate is on `y` alone.

### Where iron's `down_proj_depth` went

iron exposes a `down_proj_depth` knob for how deep the down-projection accumulates in the memory tile before draining. In AIR that quantity is the down-projection GEMM's **`tile_k_l2`**, which is measured per shape and stored in this registry. So it is not a builder parameter — it is read from `gemm_registry_config` with the rest of the tiling (porting convention 9), and printed at build time so it is visible in a log. At the shape below it is **256**.

### The activation is the tanh approximation

`ffn_gelu_bf16` calls `gelu_tanh_approx_bf16`:

```
0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
```

what HuggingFace calls `gelu_new` / `gelu_pytorch_tanh` — **not** the exact erf form. iron's oracle calls `torch.nn.functional.gelu`, whose default *is* the erf form; at iron's 4e-2 tolerance the difference hides, and at `rtol = 1.6e-2` it does not. `gelu_tanh_reference` is therefore the tanh form deliberately.

---

## Numerical accuracy

Verified element-wise over the **full output** against the FP32 reference:

| Metric (M×K×F = 2048×1024×3072, seed 5) | Measured |
|---|---|
| `mean_rel_L1 = mean｜out−ref｜ / mean｜ref｜` | **1.60e-2** |
| `rel_err max` | 2.9e+4 |
| `abs_err max` | 1.59e-3 |
| mismatches at `rtol=1.6e-2, atol=5e-3` | **0 / 2097152** |

- **`mean_rel_L1 = 1.6e-2` is ~1.6× a single GEMM's 9.9e-3**, and that gap is real rather than noise. Two things the reference does not reproduce show up in it: the device stages `h` in **bf16** between the two GEMMs, and the activation kernel carries its intermediates (`x²`, `x³`, the inner sum, `1 + tanh`) in bf16 too. The reference is FP32 end to end on purpose — reproducing the staging would hide exactly the error it introduces, the same rule [`LayerNorm_bf16.md`](LayerNorm_bf16.md) follows for one-pass variance.
- **The activation's worst case is where `1 + tanh(u)` cancels**, `x` around −2 to −3. bf16 near 1.0 has a spacing of 2⁻⁸, so a `1 + tanh` of 0.012 carries ~8% *relative* error while its *absolute* error stays around 1e-3 — and the down-projection then averages 3072 of those, which is why the end-to-end absolute error is smaller than the stage's.
- **`rel_err max = 2.9e+4` is expected.** Somewhere in a 2.1M-element output a reference value lands within a rounding of zero. `atol` is what covers it.

### Operand scale

| Tensor | Scale | Why |
|---|---|---|
| `x`, `w_up` | `N(0, K^-¼)` | puts `h` at **unit variance**, so GeLU is exercised over ±3 where it is nonlinear. At the GEMM sweep's `1/√K` the whole tensor would sit inside ±0.15, where the activation is indistinguishable from `0.5x` and this operator's own stage would go untested. |
| `w_down` | sized so `y` lands at `1/√F` | the magnitude the registry's GEMM sweep puts a depth-`F` reduction at, so `abs_err` here is on the same axis as the GEMM rows this is built from. |

This is stated because it has to be: the GEMM's error is dominated by its bfp16 MMUL emulation and is therefore proportional to the output's own magnitude, so an `atol` means nothing without the scale it was measured at.

---

## Parameters & constraints

| Knob | Value | Constraint → source |
|---|---|---|
| GEMM `herd_m` / `herd_n` | **8 / 4** | the array shape the registry tiles were measured at |
| GEMM tiles, both projections | from `gemm_registry_config` | never a constant; raises on an unmeasured shape |
| `down_proj_depth` | = down GEMM `tile_k_l2` = **256** | registry, not a parameter — see above |
| `gelu_herd_x` | **8** | AIE columns; `(M·F) % gelu_herd_x == 0` |
| GeLU herd `herd_y` | 1 (fixed) | two shim DMAs per tile is already the per-column budget at 8 columns |
| `gelu_tile_n` | defaults to `F` (one row) | `2·tile_n·2` bytes must fit L1 with ping-pong |
| `ffn_dim` | multiple of **16** | `gelu_tanh_approx_bf16` truncates its trailing partial vector and leaves those elements **untouched** — stale bytes, not zeros |
| kernel object | `encoder_ffn.o` | encoder.cc's **FFN half only**. The addnorm half would collide with `addnorm_ffn.o` on `ffn_gelu_bf16` and `ffn_eltwise_add_bf16_vector`, and this ELF needs none of it. `compile_kernels.py` checks both that the FFN symbols are present and that the addnorm ones are absent. |
| backend | `runtime_loop_tiling_sizes=[2,2]`, ELF output | BD-ID recycling; and multi-segment designs cannot use the xclbin path — see [`QKVProj_bf16.md`](QKVProj_bf16.md) |

---

## Tolerances & reference

Element-wise over the **full output**: every element must pass `|out−ref| ≤ atol + rtol·|ref|`, with zero permitted mismatches.

| Output dtype | rtol | atol |
|---|---|---|
| bf16 | 1.6e-2 | 5e-3 |

- **Reference** = CPU FP32 `gelu_tanh(x @ w_up) @ w_down`, cast once to bf16 at the end. FP32 throughout, tanh-form activation.
- `rtol = 1.6e-2` is held fixed across the registry. `atol = 5e-3` is the measured `abs_err max` of 1.59e-3 rounded up, a 3.1× margin.

---

## Tested shapes

| (M×K×F) | up GEMM | down GEMM | tile (m/kl2/kl1/n) | mean_rel_L1 | abs_err max | mismatches | Used by | Status |
|---|---|---|---|---|---|---|---|---|
| 2048×1024×3072 | fused-cast | fused-cast | 64/256/32/128 (both) | 1.6e-2 | 1.59e-3 | 0 / 2097152 | transformer-layer execution studies, encoder FFN sublayer | ✅ |
| 4096×768×3072 | drain | fused-cast | 32/256/32/128 up, 64/512/32/96 down | 1.6e-2 | 1.71e-3 | 0 / 3145728 | transformer-layer execution studies, `baseline_768` FFN sublayer | ✅ |

> **The `baseline_768` row is the only point on the sequence ladder where this operator builds at hidden 768, and the reason is in the two method columns.** At `K = 768` the up-projection takes `tile_n = 128` and the down-projection `tile_n = 96` at *every* sequence length — `N = 768` cannot use `tile_n = 128` at `herd_n = 4`, since `768 % 512 ≠ 0`. Two **same-method** GEMMs with different `tile_n` declare `f32_to_bf16_mn_<suffix>` twice with different memref types, and `stitch_elf` rejects the redefinition. `seq = 4096` is the one point the registry happens to put them on different methods, and therefore on different objects:
>
> | seq | up-proj | down-proj | |
> |---|---|---|---|
> | 64 … 2048 | `drain` t_n=128 | `drain` t_n=96 | collide |
> | **4096** | **`drain` t_n=128** | **`fused-cast` t_n=96** | builds |
> | 8192, 16384 | `fused-cast` t_n=128 | `fused-cast` t_n=96 | collide |
>
> So buildability here is a property of the registry's winners, not of the shape: a re-sweep that moved either projection onto the other's method would take the operator from *builds* to *does not build* with no source change. The fix is a symbol suffix minted per `(method, tile_n)` rather than per method, in `llms/shared/builders/gemm_builder.py`. Mixing the two methods costs nothing measurable — `mean_rel_L1` is within 2% of the all-`fused-cast` row above.

> **One of seven resolvable shapes, not the only one.** The constraint is that the registry must hold a high-precision entry for *both* directions, `(M, K, F)` and `(M, F, K)`. Seven expansions satisfy that today — `2048×1024×2048`, `2048×1024×3072`, `2048×2048×6144`, `2048×2048×8192`, `2048×2560×4096`, `2048×2560×9728` and `2048×3072×8192` — and only `2048×1024×3072` has been run on hardware. The other six are unmeasured here, which is a coverage gap rather than a known failure; nothing suggests they would not place. Beyond those, the case matrix's remaining FFN shapes have no high-precision entry on at least one side (e.g. `2048×896×4864` and `2048×1536×8960` are low-precision-only), and the builder raises on them rather than guessing a tiling. The builder raises on any other rather than guessing a tiling; filling that in is Phase C4's sweep.

**Performance is not measured here.** Phase C gates numerics only; latency and throughput are deliberately absent rather than estimated.

---

## How to reproduce

```bash
cd programming_examples/transformer_layer

# correctness on real NPU2, serialized on the repository lock (a DIFFERENT
# inode from the /tmp/npu.lock the runner takes internally).
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  make check-ffn PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR

# the negative control: perturbs one element of the DEVICE input after the
# reference is computed, and MUST fail.
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  make check-ffn OPCHECK_ARGS="--fault-inject input" \
       PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR
```

Each run writes `transformer_layer/results/ffn__<shape>.json`, carrying both resolved GEMM specs alongside the verdict; injected runs write into `results/fault/` instead.
