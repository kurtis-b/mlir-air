# 06c — Phase C3: `mha_out_proj`

Third of Phase C's four sub-phases, and the largest single rewrite in the port. Read
[06](06-phase-c-operators.md) for the overview,
[06a](06a-phase-c1-gate-and-small-operators.md) for the check mechanism you must reuse, and
[02](02-porting-conventions.md) for binding house style.

One operator: fused attention plus output projection, with optional causal masking. iron's
`mha_out_proj/design.py` is 1350 lines against `aie.iron` ObjectFifo / Worker / Runtime, with
`parallel_seq`, `q_seq_tile`, `kv_seq_tile`, `emb_tile`, `parallel_heads` and `o_proj_acc_depth`
knobs.

## What to compose it from

Two existing pieces, neither of which you should modify:

- `programming_examples/flash_attention/kernel_fusion_based/` — `attn_npu2.py` (heads-first) and
  `attn_npu2_seqfirst.py` (seq-first, same object, verified bit-identical). Both already gate at
  the registry's FlashAttention tolerances (`rtol=1.6e-2, atol=1e-1`) and are the model to follow.
  The kernel takes shape-baked `-D` flags (`-Dlqp`, `-Dlkp`, `-Ddk`, `-Ddk_full`, `-Ddv`,
  `-Ddv_full`, `-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`, `-DROUND_CONV_EVEN`), where
  `lqp_tile = lqp / num_q_tiles`.
- The O-projection half of `llms/shared/builders/o_ffn_multi.py`.

Causal masking uses the row helpers Phase A added to `attn_npu2.cc` behind
`-DCAUSAL_ROW_HELPERS`: `copy_O_tile_rows`, `store_row_value`, `copy_row_values`.

**`copy_O_tile_rows` is numerically a no-op and must not be deleted as dead code.** Removing it
hangs the design with `ERT_CMD_STATE_TIMEOUT`, because the consuming DMA never sees its buffer
descriptor complete. The footgun is recorded in the kernel source; keep it recorded in yours.

Split per convention rule 5 — the seams are attention staging, O-projection staging, and the
`extern "C"` entry layer, mirroring how Phase A split `encoder.cc` into three files by role.

## The reference

iron's oracle (`iron/operators/mha_out_proj/reference.py`) has a trap beyond the bf16 one: it
computes bf16 SDPA below `seq_len 16384` and switches to FP32 chunked attention at and above it,
so the reference's own precision changes across the ladder. Tolerances calibrated at one end then
mean something different at the other.

Compute **chunked FP32 attention at every sequence length**. Port iron's `_chunked_attention`
shape — the block loop, the causal mask built from `q_positions`/`kv_positions`, `scale =
1/sqrt(d)` — and drop the branch.

Note the layout: FlashAttention's harness compares against an SDPA reference reshaped
`(heads, lq, dv_chunks, lkp) → (0, 2, 1, 3)` to match the device layout
(`attn_npu2.py:1350-1355`). Getting this wrong produces a correctly-shaped, wholly wrong result.

## Accuracy expectation

FlashAttention is looser than a single GEMM and legitimately so: it chains two BFP16-emulated MMAs
plus a bf16 online softmax, giving `mean_rel_L1 ≈ 3.9e-2`, about 4× the GEMM tier. The registry
records this and holds `rtol = 1.6e-2` with `atol = 1e-1`
(`kernel_registry/details/FlashAttention_bf16.md`).

`atol = 1e-1` is the loosest value anywhere in the registry, and the driver's objective check
rejects anything above it. If this operator needs more than that to pass, the answer is a defect
report, not a wider tolerance.

`head_dim = 128` FlashAttention has been flaky (hang or NaN) on some NPU2 setups; the registry
rows record it. The case matrix is `head_dim = 64` throughout, so you should not meet it — if you
do, that is a signal something is mis-shaped.

## Work items

1. `build_mha_out_proj_module(...)` in `transformer_layer/builders/`, split per rule 5, with
   `parallel_seq` / `parallel_heads` / `o_proj_acc_depth` as builder arguments.
2. Optional causal masking over the `-DCAUSAL_ROW_HELPERS` entry points.
3. FP32 chunked-attention reference as a module-level function beside the builder, at every
   sequence length.
4. Register the operator with `opcheck.py --list` so the driver enumerates it, including the
   causal and non-causal variants as separate shape keys.
5. `run_npu2_mha_out_proj_peano.lit` plus its `Makefile` targets.
6. Registry rows for every validated `(kernel, shape)`, carrying `mean_rel_L1`, `Used by`, status.
7. `black`; module docstrings stating the contract and its footguns.

## Gate

```
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  ninja -C build-xrt check-programming-examples-transformer-layer
```

Every test in the suite, including C1's and C2's. Plus the driver's objective check: freshness,
re-derived verdict, and a fault-injection run that must fail.

## Constraints

- **Do not modify `llms/shared/` or `flash_attention/`.** Compose from them. Goal 1
  (sliding-window attention) will later modify FlashAttention behaviour; leaving it untouched here
  is what keeps the two from colliding.
- Wrap every NPU command in `flock -x -w 1800 /tmp/mlir-air-npu.lock`. Never take
  `/tmp/npu.lock`.
- This is the biggest rewrite in the port. An honest partial report with a clear blocker is a
  better outcome than a rushed pass — the driver cross-checks your report against the gate anyway.
