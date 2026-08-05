<!---//===- MHAOutProj_bf16.md --------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//-->

# MHA + Output Projection (BF16) — Kernel Detail

> The transformer block's attention sublayer end to end in one ELF: `y = softmax(Q Kᵀ / √d [+ causal mask]) V @ W_o`.
> Shapes are written **`S×S, Hq/Hkv, d`** (S = sequence length, Hq/Hkv = query/key-value heads, d = head dim); the model width is `E = Hq·d` and the projection is `S×E×E`.
>
> Companion: [`../supported_kernels.md`](../supported_kernels.md) · [`../README.md`](../README.md) · [`FlashAttention_bf16.md`](FlashAttention_bf16.md) (the attention half) · [`GEMM_bf16_in_bf16_out.md`](GEMM_bf16_in_bf16_out.md) (the projection) · [`QKVProj_bf16.md`](QKVProj_bf16.md) (the sublayer's other end)
> **Scope: NPU2 (Strix / AIE2P) only.** Measured on real NPU2, August 2026. Reproduce commands in "How to reproduce" below.

---

## Builder

```
programming_examples/transformer_layer/builders/mha_out_proj.py
  build_mha_out_proj_module(seq_len, head_dim, num_heads, num_kv_heads=None,
                            causal=False, parallel_seq=256, parallel_heads=2,
                            kv_seq_tile=None, cascade_stages=4, num_q_tiles=None,
                            o_proj_acc_depth=None, o_herd_m=8, o_herd_n=4,
                            gemm_spec_fn=None)
  mha_out_proj_arg_layout(...) / mha_out_proj_device_inputs(...)  # signature is COMPUTED
  mha_out_proj_reference(q, k, v, w_o, num_heads, ...)            # the FP32 oracle

programming_examples/transformer_layer/builders/mha_attention.py
  attention_config(...) / build_attention_ir(cfg) / compile_attention_kernel(cfg)
  chunked_attention_reference(q, k, v, num_heads, ...)

programming_examples/transformer_layer/builders/o_proj.py
  o_proj_gemm_spec(...) / build_o_proj_ir(...) / o_proj_reference(attn_out, w_o)
```

Driven by `transformer_layer/opcheck.py --operator mha_out_proj`; `make check-mha-out-proj` is the same thing behind the lit test.

**Composed, not re-derived.** The attention half *is* `flash_attention/kernel_fusion_based/attn_npu2_seqfirst.py`, imported and parameterised — the same design and the same `attn_npu2.o` the FlashAttention rows were measured with, untouched. The projection half is the registry's GEMM builder. What this operator adds is the launch structure between them, which is why its `mean_rel_L1` is quoted next to FlashAttention's.

**Split by role, three modules.** iron's `mha_out_proj/design.py` is 1350 lines against `aie.iron` ObjectFifo / Worker / Runtime. Phase A split `encoder.cc` the same way: attention staging · O-projection staging · the entry layer. Porting convention 5.

---

## Datapath

Seq-first throughout. Two launch groups in one ELF:

```
1..N. FlashAttention   q[S, Hq·d], k[S, Hkv·d], v[S, Hkv·d] → attn_out[S, E] bf16
                       online softmax, cascade merge over `cascade_stages`
N+1.  O GEMM           attn_out[S, E] @ w_o[E, E]           → y[S, E] bf16
                       external mm.o, f32 accumulate, single epilogue cast
```

`attn_out` is a real DDR buffer in the signature, handed to the device zero-filled. It is staging, not an output: the gate is on `y` alone.

### Why seq-first and not heads-first

Both FlashAttention harnesses drive the same `attn_npu2.o` and are verified bit-identical (`max abs diff = 0`, see [`FlashAttention_bf16.md`](FlashAttention_bf16.md)). They differ only in the L3 layout: heads-first writes `[Hq·dv_chunks, S, dv_tile]`, seq-first writes `[S, Hq·d]`.

`[S, Hq·d]` **is** the projection's `A` operand — the same bytes, no repack. Composing the heads-first variant would need a transpose launch between the halves for no numerical gain. This is the `opt-layout-alignment` argument applied at build time rather than as a later optimisation, and it is why the fusion costs nothing beyond one dispatch.

### Where iron's `o_proj_acc_depth` went

It is the projection GEMM's **`tile_k_l2`**, measured per shape and stored in this registry. The builder exposes it as an argument because the phase document asks for it, but its default (`None`) reads the registry, and a value passed there is an explicit override — printed at build time and recorded in the results artifact next to the registry's own value. Porting convention 9. At every shape below it is **256**.

---

## Numerical accuracy

Verified element-wise over the **full output** against the FP32 reference:

| Metric (512×512, 8q/8kv, d=64, non-causal, seed 6) | Measured |
|---|---|
| `mean_rel_L1 = mean｜out−ref｜ / mean｜ref｜` | **4.64e-2** |
| `abs_err max` | 1.95e-2 |
| `atol_required` = `max(｜out−ref｜ − rtol·｜ref｜)` | 1.85e-2 |
| mismatches at `rtol=1.6e-2, atol=5e-2` | **0 / 262144** |

- **`mean_rel_L1` sits in FlashAttention's own band (3.9e-2), not a GEMM's (9.3e-3).** That is the expected answer, not a disappointment: the operator's attention half *is* that kernel, chaining two BFP16-emulated MMAs and a bf16 online softmax, and a projection whose own relative error is 4× smaller cannot pull the total down. The composition costs nothing measurable in *relative* terms.
- **`atol_required`, not `abs_err max`, is the number an `atol` should be quoted against.** For causal attention the largest absolute error lands on a large-magnitude element that `rtol` already covers — the first rows attend to a handful of keys, so `|y|` runs to 4.1 instead of 0.35 while the relative error is, if anything, lower. Quoting `abs_err max` alone would make the causal rows look twice as close to their tolerance as they are. `opcheck.py` records both.
- **Causal is a lower relative error and a larger absolute one.** 3.58e-2 vs 4.64e-2 `mean_rel_L1`, but 5.86e-2 vs 1.95e-2 `abs_err max`. Masking concentrates the softmax onto fewer keys, so less averaging happens — which both reduces accumulated rounding and widens the output's dynamic range.

### Operand scale

| Tensor | Scale | Why |
|---|---|---|
| `q`, `k`, `v` | `N(0, 1)` | the distribution the FlashAttention registry rows were measured at, which is PyTorch's own SDPA test standard |
| `w_o` | `N(0, 1/√E)` | the registry GEMM sweep's scale for a depth-`E` reduction, so the projection's contribution sits on the same axis as the GEMM rows |

This is stated because it has to be: the GEMM's error is dominated by its bfp16 MMUL emulation and is therefore proportional to the output's own magnitude, so an `atol` means nothing without the scale it was measured at.

---

## Parameters & constraints

| Knob | Value | Constraint → source |
|---|---|---|
| `parallel_seq` (iron) = `lqp` | **256** | the one near-unique full-chip FlashAttention config; `S % parallel_seq == 0` |
| `parallel_heads` (iron) = `num_heads_per_unroll` | **2** | physical columns = `num_q_tiles · parallel_heads ≤ 8` |
| `num_q_tiles` | **4** | defaults to `parallel_seq / kv_seq_tile`; herd column count |
| `cascade_stages` | **4** | herd row count, ≤ 4 on NPU2 |
| `kv_seq_tile` (iron `kv_seq_tile`) = `lkp` | **= `head_dim`** | shared-buffer mode. A larger `head_dim` puts the design on `dv_chunks > 1`, which the seq-first harness does not support and which the registry records as flaky (hang or NaN) on some NPU2 setups |
| `q_seq_tile` | `parallel_seq / num_q_tiles` | **must equal `kv_seq_tile` under causal masking** — the device compares BLOCK indices, so misaligned tiles mask the wrong triangle, silently |
| `o_proj_acc_depth` | = projection `tile_k_l2` = **256** | registry; the builder argument defaults to it |
| O GEMM `herd_m` / `herd_n` | **8 / 4** | the array shape the registry tiles were measured at |
| O GEMM tiles + method | from `gemm_registry_config` | never a constant; raises on an unmeasured shape |
| kernel objects | `attn_npu2.o` + the method's `mm_*.o` | both must be in the working directory when aiecc links |
| backend | `omit_pingpong="all"`, `runtime_loop_tiling_sizes=[1,1]`, ELF output, `omit_while_true_loop=False` | **FlashAttention's, not the GEMM's.** The `[2,2]` the standalone GEMM-backed operators use is BD-ID recycling for a fused-cast herd; the projection places without it, and raising it re-tiles the attention runtime loop too |

### `attn_npu2.o`'s `-D` flags are per *tile*, and a mismatch hangs

`-Dlqp` is the **Q tile size** (`parallel_seq / num_q_tiles`), not the Q chunk per launch; `-Ddk` / `-Ddv` are the `lkp`-sized **tile** while `-Ddk_full` / `-Ddv_full` are the full head dimension. They instantiate the matmul microkernels, so they must match the L1 buffer shapes the Python builder emits. `mha_attention.py` derives both from one config dict, and rebuilds with `force=True` because a shared working directory may hold an object built for another shape. A mismatch does not fail to link — it ends in `ERT_CMD_STATE_TIMEOUT`.

### `copy_O_tile_rows` is numerically a no-op and must not be deleted

The causal row helpers behind `-DCAUSAL_ROW_HELPERS` (`copy_O_tile_rows`, `store_row_value`, `copy_row_values`) are linked into every causal variant here. `copy_O_tile_rows` reads every element of an O tile and writes it straight back; that is the point, so a KV block entirely above the diagonal — which runs no matmul — still completes the consuming DMA's buffer descriptor. Removing it as dead code hangs the design.

This composition does **not call** them. The masking path it composes (`apply_causal_mask`) fills a wholly-masked score tile with `-inf` and lets the matmul run anyway, so no O tile is ever left untouched. They are the entry points a block-skipping variant needs, and keeping them linked is what lets one be added without re-deriving the flag set.

---

## Tolerances & reference

Element-wise over the **full output**: every element must pass `|out−ref| ≤ atol + rtol·|ref|`, with zero permitted mismatches.

| Output dtype | rtol | atol (non-causal) | atol (causal) |
|---|---|---|---|
| bf16 | 1.6e-2 | 5e-2 | 8e-2 |

- **Reference** = CPU **chunked FP32** attention at *every* sequence length, then the projection in f32, then a single rounding to bf16. iron's oracle computes bf16 SDPA below `seq_len 16384` and switches to FP32 chunked attention at and above it, so its own precision changes across the ladder and a tolerance calibrated at one end means something different at the other; that branch is dropped deliberately. Chunking is arithmetically exact, so applying it everywhere costs nothing and removes the discontinuity.
- **FP32 end to end.** The device stages `attn_out` in bf16 between the halves and the reference does not. That gap is measured rather than defined away, the same rule [`FFN_bf16.md`](FFN_bf16.md) follows for its intermediate activation.
- `rtol = 1.6e-2` is held fixed across the registry. `atol = 5e-2` is 2.7× the non-causal `atol_required` of 1.85e-2. The causal `atol = 8e-2` is 1.6× theirs (4.88e-2 and 4.81e-2), **below the registry's usual 2–3× margin and deliberately so**: `1e-1` is a hard ceiling — the loosest `atol` anywhere in the registry, and FlashAttention's own — and this datapath's honest error gets within a factor of two of it. Widening to `1e-1` would buy the usual margin and nothing else; `8e-2` keeps these rows strictly tighter than the tier they inherit.

---

## Tested shapes

| S×S | Hq/Hkv | d | causal | O GEMM | tile (m/kl2/kl1/n) | mean_rel_L1 | abs_err max | atol_required | mismatches | Used by | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 512×512 | 8/8 | 64 | ✗ | drain | 32/256/32/128 | 4.64e-2 | 1.95e-2 | 1.85e-2 | 0 / 262144 | transformer-layer execution studies, attention sublayer | ✅ |
| 512×512 | 8/8 | 64 | ✓ | drain | 32/256/32/128 | 3.58e-2 | 5.86e-2 | 4.88e-2 | 0 / 262144 | transformer-layer execution studies, decoder attention sublayer | ✅ |
| 2048×2048 | 16/16 | 64 | ✓ | drain | 32/256/32/128 | 4.11e-2 | 5.08e-2 | 4.81e-2 | 0 / 2097152 | transformer-layer execution studies, prefill-sized attention sublayer | ✅ |
| 4096×4096 | 12/12 | 64 | ✗ | drain | 32/256/32/96 | 5.33e-2 | 9.03e-3 | 8.71e-3 | 0 / 3145728 | transformer-layer execution studies, `baseline_768` attention sublayer | ✅ |

> **The `4096×4096` row has the loosest relative error and the tightest `atol_required` of the four, which is one effect seen twice.** Softmax over 4096 keys averages `V` eight times harder than over 512, so `|y|` shrinks by roughly `√8` and the same relative error lands closer to zero: `atol_required` falls to 8.7e-3 while `mean_rel_L1` rises to 5.33e-2. Its `atol` is 2.5e-2, a 2.9× margin and 4× below the hard `1e-1` ceiling — so the row the encoder block actually runs has the most headroom of the four, not the least.
>
> It is **non-causal** because `encoder_bert` is bidirectional: its golden reference builds an all-ones attention mask, and the `tril` one belongs to `decoder_gpt2`. The causal rows above are a different device path and are not `baseline_768` evidence however large they are.

> **The error is set by the datapath, not by the shape.** The 2048 row's output is 8× the 512 causal row's and its `mean_rel_L1` and `atol_required` land in the same band — the same property the FlashAttention rows record across their own ladder.
>
> **`head_dim = 64` throughout, on purpose.** `head_dim = 128` FlashAttention has been flaky (hang or NaN) on some NPU2 setups; `attention_config` rejects it here rather than letting a mis-shaped call find that out on hardware. Extending to 128 means extending the seq-first harness to `dv_chunks > 1` first, which is a FlashAttention change and not this operator's.
>
> **Coverage is limited by the projection's registry entries.** A shape needs `(S, E, E)` with a high-precision entry, and `S` a multiple of `parallel_seq = 256`. Eight satisfy both today — `512×512×512`, `1024×1024×1024`, `2048×896×896`, `2048×1024×1024` (all `drain`) and `2048×1536×1536`, `2048×2048×2048`, `2048×3072×3072`, `4096×4096×4096` (all `fused-cast`). Two of the eight are exercised above: `512×512×512` (twice, causal and not) and `2048×1024×1024`. **No `fused-cast` projection has been run in this composition.** The arg layout and the wiring for it are in place and exercised by `ffn`, but that method's herd is the one that wants `runtime_loop_tiling_sizes=[2,2]` for BD-ID recycling, and this operator runs at `[1,1]` because the attention half needs it — so the combination is untested rather than known-good. It is a coverage gap, not a known failure; resolving it is Phase C4's.

**Performance is not measured here.** Phase C gates numerics only; latency and throughput are deliberately absent rather than estimated.

---

## How to reproduce

```bash
cd programming_examples/transformer_layer

# correctness on real NPU2, serialized on the repository lock (a DIFFERENT
# inode from the /tmp/npu.lock the runner takes internally).
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  make check-mha-out-proj PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR

# the negative control: perturbs one element of the DEVICE input after the
# reference is computed, and MUST fail.
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  make check-mha-out-proj OPCHECK_ARGS="--fault-inject input" \
       PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR
```

The control injects into `w_o`, not into Q/K/V, and that is a measured choice rather than a convenience. Softmax normalisation damps a single-element perturbation of the attention inputs: on this reference, one element of K or V moves the output by at most 4.9e-3 and one of Q by 3.5e-2 — inside, or barely outside, an `atol` sized to this datapath's honest error. A control that close to the band measures where two numbers happened to land, not whether the check discriminates. One perturbed weight moves an entire output column by `attn_out[:, c] · Δ`, measured at 3.7e-1 (1.5 causal), one to two orders clear of it.

Each run writes `transformer_layer/results/mha_out_proj__<shape>.json`, carrying the resolved GEMM spec and the FlashAttention config alongside the verdict; injected runs write into `results/fault/` instead.
