<!---//===- QKVProj_bf16.md -----------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//-->

# QKV Projection (BF16) — Kernel Detail

> The three attention projections as **one** GEMM over a fused weight, with the result split three ways on the device: `x[M, K] @ w_qkv[K, 3K]` → `q[M, K]`, `k[M, K]`, `v[M, K]`.
> Shapes are written **`M×K`** (M = rows / seq, K = embedding dim); the underlying GEMM is `M×K×3K`.
>
> Companion: [`../supported_kernels.md`](../supported_kernels.md) · [`../README.md`](../README.md) · [`GEMM_bf16_in_bf16_out.md`](GEMM_bf16_in_bf16_out.md) (the GEMM this is built from)
> **Scope: NPU2 (Strix / AIE2P) only.** Measured on real NPU2, August 2026. Reproduce commands in "How to reproduce" below.

---

## Builder

```
programming_examples/transformer_layer/builders/qkv_proj.py
  build_qkv_proj_module(seq_len, emb_dim, herd_m=8, herd_n=4,
                        split_herd_x=8, gemm_spec_fn=None)
  qkv_proj_reference(x, w_qkv)          # the FP32 oracle, returns (q, k, v)
```

Driven by `transformer_layer/opcheck.py --operator qkv_proj`; `make check-qkv-proj` is the same thing behind the lit test.

**Tiles and method are never written down here.** `gemm_registry_config(M, K, 3K, "bf16", "high")` returns both, and it *raises* on a shape nobody swept rather than guessing — the drift-bug class porting convention 9 exists to stop. The `gemm_spec_fn` hook can inject a spec for an unmeasured shape (the same hook `rms_qkv_qknorm_rope_multi.py` ships for `qwen3_4b`), and when it is used the injected method and tiles are written into the results artifact so the guess is visible rather than silent.

---

## Datapath — where the three-way split happens

Four `air.launch` operations in one ELF:

```
1. GEMM      x[M,K] @ w_qkv[K,3K] → C_f32[M,3K]      external mm.o, f32 accumulate
2. cast Q    C_f32[:, 0*K : 1*K]  → q[M,K] bf16
3. cast K    C_f32[:, 1*K : 2*K]  → k[M,K] bf16
4. cast V    C_f32[:, 2*K : 3*K]  → v[M,K] bf16
```

The registry's high-precision method for these shapes is **fused-cast**: the GEMM accumulates in f32 into a scratch buffer and a *separate* launch casts it down to bf16. That cast has to read every element of C exactly once regardless, so **the split is folded into it** — three cast launches, each with a column offset on its read and its own bf16 destination. Q, K and V therefore land in three separate DDR buffers with **no host-side slice of C anywhere**, and the split costs nothing beyond the cast that was already going to happen.

The consequence is a hard requirement, not a preference: a method with no f32 scratch (`drain`, which casts inside the GEMM) has nothing for the split to ride on. `build_qkv_proj_module` **raises** in that case rather than falling back to three full-tensor copies on a different numeric path.

Each split-cast launch is an `8×1` herd walking one row per iteration. A row's column band is `K` *contiguous* f32 at flat offset `r·3K + offset`, so every DMA is a plain 1-D descriptor and L1 holds one row in and one row out.

---

## Numerical accuracy

Verified element-wise over the **full output of all three projections** against the FP32 reference:

| Metric (M×K = 2048×1024, operands ~ `N(0, 1/√K)`, seed 4) | Measured |
|---|---|
| `mean_rel_L1 = mean｜out−ref｜ / mean｜ref｜` (Q / K / V) | **9.86e-3 / 9.86e-3 / 9.87e-3** |
| `rel_err max` | 1.7e+5 |
| `abs_err max` | 1.95e-3 |
| mismatches at `rtol=1.6e-2, atol=5e-3` | **0 / 6291456** |

- **`mean_rel_L1 = 9.9e-3` is the GEMM's own number**, and the GEMM page records 9.4e-3 for exactly this `2048×1024×3072` shape. The three split-cast launches add nothing measurable — which is the point of folding the split into a cast the datapath already owed.
- The error is dominated by the microkernel's **bfp16 MMUL emulation** (`-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`), not by the bf16 output rounding. It is therefore proportional to the output's own magnitude, which is why the operand scale below is quoted with it.
- **`rel_err max = 1.7e+5` is expected and meaningless on its own.** Somewhere in a 6.3M-element output a reference value lands within a rounding of zero; the relative error there is unbounded while the absolute error stays at one bf16 ULP. `atol` is what covers it, and the registry's methodology is explicit that a correlation gate would be blind to the systematic error this one catches.

### Operand scale

Both operands are drawn `N(0, 1/√K)`, which is the scale
[`matrix_multiplication/bf16_in_bf16_out/run.py`](../../matrix_multiplication/bf16_in_bf16_out/run.py) uses for the registry's own GEMM sweep, so it puts the product at `1/√K` too. This is stated because it has to be: with the error proportional to output magnitude, an `atol` means nothing without the scale it was measured at. Using the sweep's scale is what makes `mean_rel_L1` here directly comparable with the GEMM rows this operator is built from.

### This resolves the ⚠️ on `2048×1024×3072`

[`GEMM_bf16_in_bf16_out.md`](GEMM_bf16_in_bf16_out.md) marks fused-cast at this shape ⚠️, and its note diagnoses it precisely: both high-precision methods *compute the in-tier result* (9.4e-3), but the harness gate tripped on a single near-zero-reference element at `abs_err ≈ 1.7e-3` against that page's `atol = 1.5e-3`. This entry measures `abs_err max = 1.95e-3` over 3× as many elements and passes with **zero** mismatches at `atol = 5e-3` — corroborating the diagnosis rather than contradicting it. The GEMM page's own suggested remedy was to relax the high-precision `atol` to match the other tiers, leaving the GPU-standard `rtol` alone; that is exactly what this entry does.

---

## Parameters & constraints

| Knob | Value | Constraint → source |
|---|---|---|
| GEMM `herd_m` / `herd_n` | **8 / 4** | the array shape the registry tiles were measured at |
| GEMM tiles | from `gemm_registry_config` | never a constant; raises on an unmeasured shape |
| GEMM method | must expose an f32 scratch | the split rides the cast launch — see the datapath section |
| `split_herd_x` | **8** | AIE columns; `seq_len % split_herd_x == 0` |
| split herd `herd_y` | 1 (fixed) | two shim DMAs per tile is already the per-column budget at 8 columns |
| split L1 | `K·4 + K·2` bytes | one f32 row in, one bf16 row out, plus the ping-pong pair |
| `emb_dim` | multiple of **16** | the vectorized cast has no scalar tail |
| backend | `runtime_loop_tiling_sizes=[2,2]`, ELF output | BD-ID recycling; and multi-segment designs cannot use the xclbin path — see below |

### Multi-segment designs need ELF output

Each `air.launch` lowers to its own `aie.device`, driven by an `aiex.configure` / `aiex.run` runtime sequence. The xclbin path names a **single** instruction blob on the aircc command line (`-i air.insts.bin`), so a second segment collides on it and aiecc stops with `edge 'air.insts.bin' produced duplicate output path`. ELF output is what the shipped multi-launch llama builders use, for the same reason. It is a packaging concern only — the comparison downstream is identical.

---

## Tolerances & reference

Element-wise over the **full output**: every element of every projection must pass `|out−ref| ≤ atol + rtol·|ref|`, with zero permitted mismatches.

| Output dtype | rtol | atol |
|---|---|---|
| bf16 | 1.6e-2 | 5e-3 |

- **Reference** = CPU FP32 `x @ w_qkv` cast once to bf16, then sliced into thirds. Structurally identical to the device: the whole `K` reduction in f32, one epilogue rounding. It is *not* iron's bf16 oracle, which agrees with a bf16 device partly by being wrong in the same direction.
- `rtol = 1.6e-2` is held fixed across the registry. `atol = 5e-3` is the measured `abs_err max` of 1.95e-3 rounded up, a 2.6× margin — the same sizing the GEMM rows use.

---

## Tested shapes

| (M×K) | GEMM (M×K×3K) | method | tile (m/kl2/kl1/n) | mean_rel_L1 | abs_err max | mismatches | Used by | Status |
|---|---|---|---|---|---|---|---|---|
| 2048×1024 | 2048×1024×3072 | fused-cast | 64/256/32/128 | 9.9e-3 | 1.95e-3 | 0 / 6291456 | transformer-layer execution studies, attention projections | ✅ |
| 2048×2048 | 2048×2048×6144 | fused-cast | 64/256/32/128 | 9.7e-3 | 1.22e-3 | 0 / 12582912 | transformer-layer execution studies, attention projections | ✅ |
| 64×768 | 64×768×2304 | fused-cast | 64/128/32/96 | 9.9e-3 | 1.95e-3 | 0 / 147456 | transformer-layer execution studies, shortest point of the `baseline_768` ladder | ✅ |
| 4096×768 | 4096×768×2304 | fused-cast | 64/256/32/96 | 9.9e-3 | 1.95e-3 | 0 / 9437184 | transformer-layer execution studies, `baseline_768` block projections | ✅ |

> **The two `baseline_768` rows are the ends of the sequence ladder, and they are the pair that exercises the per-row herd.** `64×768` builds at herd `1×4` — `M = 64` cannot hold eight rows of `fused-cast`'s forced `tile_m = 64` — while `4096×768` builds at the file-level `8×4`. A builder that stopped reading the herd from the registry row would fail on the first and still pass the second, so both are here rather than one. `64×768` is also the shape whose original `fused-cast` row returned **zeros for two of nine cast sub-tiles**: it resolved and it was wrong, which is why a resolution check is not a correctness check.

> **Both `M×K×3K` triples that the GEMM registry held when this operator was validated are validated here.** The registry now holds **11**: Phase C4's sweep added the nine `baseline_768` `qkv_proj` shapes (`seq×768×2304` across the full sequence ladder). Those nine resolve, but they have not been through this operator's own numerical check — a registered tiling and a validated operator shape are different claims. Of the 108 projection-GEMM shapes the case matrix asks for, **41 are registered**; the remaining 67 (`baseline_512` and `baseline_1024`) are the same sweep tool over a different `--family`, and the builder raises on them rather than guessing a tiling.
>
> The two agree to within measurement noise (9.9e-3 / 9.7e-3), and the larger shape's `abs_err max` is *lower* — 1.22e-3 against 1.95e-3 — because its operand scale `1/√K` is smaller at K=2048. That is the reason the scale is quoted with the tolerance rather than left implicit.

**Performance is not measured here.** Phase C gates numerics only; latency and throughput are deliberately absent rather than estimated. The underlying GEMM's throughput is on [`GEMM_bf16_in_bf16_out.md`](GEMM_bf16_in_bf16_out.md).

---

## How to reproduce

```bash
cd programming_examples/transformer_layer

# correctness on real NPU2, serialized on the repository lock (a DIFFERENT
# inode from the /tmp/npu.lock the runner takes internally).
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  make check-qkv-proj PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR

# the negative control: perturbs one element of the DEVICE input after the
# reference is computed, and MUST fail.
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  make check-qkv-proj OPCHECK_ARGS="--fault-inject input" \
       PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR
```

Each run writes `transformer_layer/results/qkv_proj__<shape>.json`, carrying the resolved GEMM method and tiles alongside the verdict; injected runs write into `results/fault/` instead.
