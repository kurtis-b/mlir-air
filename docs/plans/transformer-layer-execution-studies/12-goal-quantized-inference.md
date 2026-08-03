# 12 — Goal 2: Quantized Inference

Extend quantized inference beyond the single int4-AWQ Llama example: close the int4 prefill
performance gap, reach a second model, and give the kernel registry a quantization axis.

## Correct the record first

`programming_examples/llms/llama32_1b_int4/README.md` says the int4 decode driver "lives in a
follow-up PR". **That is stale.** Commit `aa73c0d7` landed NPU int4 decode and the end-to-end
inference driver at ~17.8 tok/s — against 12.2 tok/s for bf16 Llama-3.2-1B — and both
`llama32_1b_int4_decode.py` and `llama32_1b_int4_inference.py` are in tree.

The README also still points at `../llama_kernel_builder/`, renamed to `../shared/` in
`2f20c2fa`.

Fix the README as step one, so the goal is scoped against reality rather than against a
document.

## The gate is weaker than it looks

`[Codex]` `Int4NpuRunner` runs **bf16 prefill on dequantized AWQ weights**, with int4 only in
decode. Its own docstring says so:

> Prefill is NPU bf16 (on dequantized AWQ weights) since the int4 prefill path is currently
> kernel-bound; decode is NPU int4.

So `make verify` for the int4 model **does not validate int4 prefill at all**. Any claim that
"the int4 model passes verify" is true but does not mean what it appears to mean.

## What already exists

More than the README suggests. Three prefill backends coexist behind `--prefill-dtype`:

| Backend | State |
|---|---|
| `bf16` | Default; 84 ms/layer, 1.38 s end-to-end at seq=2048 |
| `int4` | Works; 698 ms/layer, 11.2 s — same AWQ-quality output |
| `bfp16` | **Exists**, with its own lit test (`run_npu2_verify_prefill_bfp16.lit`, `Makefile:94`) |

Leaf-kernel support is broad: `matrix_multiplication/{int4_awq, bf16_x_bfp16, i8, i16}`,
`matrix_vector_multiplication/int4_awq`, `vector_matrix_multiplication/{i8, block_quantized_i8}`,
`dequant_awq/`, `decode_ffn_swiglu/matvec_int4_swiglu_rms.py`.

`[Codex]` So the gap is **not** "BFP16 does not exist" — it is that BFP16 has no
prefill-plus-decode end-to-end inference path.

## The real gaps

### 1. The correctness gates are conflated

Separate them. int4 prefill, int4 decode, and BFP16 end-to-end each need their own PASS/FAIL,
rather than one gate that silently exercises the bf16 path. Until that happens, performance work
on int4 prefill has no correctness signal behind it.

### 2. int4 prefill is 8x slower than bf16

698 ms/layer versus 84 ms. Two documented structural causes, both addressable:

- **Down GEMM L2 budget.** At K=8192 the Down projection hits the memtile L2 budget, capping
  `herd_m=2` — 8 processing elements instead of 32 — because
  `matrix_multiplication/int4_awq/matmul_int4_packed.py` cannot tile `K_L2 < K`. Fix: add L2
  K-tiling.
- **Peano immediate range.** The AIE2P `VLD_x_pstm_nrm_imm` 9-bit immediate range forces
  `tile_n=16`, versus bf16's `tile_n=128` — 16x more launch iterations. Fix: restructure the
  addressing or the loop to stay in range.

Decode does not have this problem: it is DMA-bandwidth-bound, which is exactly where int4's
halved weight footprint pays, hence the 17.8 versus 12.2 tok/s result.

### 3. Only 1 of 10 models is quantized

The int4 builders live in `llama32_1b_int4/multi_launch_builder/`
(`rms_gemms_rope_int4_multi.py`, `o_ffn_int4_multi.py`, `o_gemv_ffn_int4_multi.py`, and bfp16
siblings). Generalize them into `llms/shared/builders/`, mirroring how the bf16 builders were
hoisted in `2f20c2fa`.

`[Codex]` **Generalize the model-specific packing and compiler variants before moving them, not
after.** Hoisting model-specific code into a shared location and generalizing it there is how
shared infrastructure accumulates special cases.

Target Qwen3-1.7B or Llama-3.2-3B next.

### 4. The registry has no quantization axis

`kernel_registry` is bf16-only by declared scope — zero quantized rows — and
`registry_lookup.gemm_config(M, K, N, output_dtype, precision)` has no quantization parameter.
Its own docstring anticipates extension: "Currently provides GEMM lookups; other kernels add
their own as their JSON lands."

Add `details/GEMM_int4_awq.{md,json}` and extend the lookup signature. Without this, quantized
builders keep hardcoding tiles — exactly the drift problem the registry exists to prevent
(convention rule 9).

### 5. The study schema needs quantization fields

`[Codex]` **A `dtype` column is not enough to describe a quantized run.** The schema needs:

- packing scheme (e.g. `two_values_per_byte_low_nibble_first`)
- group size (AWQ g128 here)
- scale and zero-point layout
- accumulation type
- separate GEMM and GEMV contracts — they differ

Fold these into the Phase F schema v1 rather than bolting them on later. The columns can be
empty for bf16 rows. See [03-measurement-model.md](03-measurement-model.md).

### 6. Define the measurement before setting a target

`[Codex]` "Within 2x of bf16" is meaningless without stating what is timed: per-layer kernel
time, end-to-end prefill, or wall-clock TTFT including host work. Define the measurement first,
then set the target.

## Work items

1. Fix `llama32_1b_int4/README.md` — the stale decode claim and the stale
   `llama_kernel_builder` path.
2. Split the correctness gates: int4 prefill, int4 decode, BFP16 end-to-end.
3. Add L2 K-tiling to the int4 GEMM; work around the `tile_n=16` immediate-range constraint.
4. Generalize the int4 and bfp16 builders in place, then hoist to `llms/shared/builders/`.
5. Add `details/GEMM_int4_awq.{md,json}`; extend `registry_lookup` with a quantization axis.
6. Add the quantization fields to study schema v1.
7. Define the performance measurement precisely.
8. Deploy a second quantized model (Qwen3-1.7B or Llama-3.2-3B).
9. Build the BFP16 prefill-plus-decode end-to-end path.
10. Measure quantized modes through the Phase F harness.

## Gate

Two conditions:

1. A second quantized model passes `make verify` **under a gate that actually exercises the
   quantized path** — not bf16 prefill on dequantized weights.
2. int4 prefill per-layer latency is materially closer to bf16, under the measurement defined in
   work item 7.

## Risks

- The two int4 prefill bottlenecks are both toolchain-adjacent. The Peano immediate-range
  constraint in particular may not have a clean workaround at the source level.
- Hoisting builders to `llms/shared/` affects the shipped deployments; re-run `make verify`
  across all ten.
- AWQ checkpoints for a second model may not exist at the same quantization configuration
  (uint4 asymmetric, g128, bf16 lm_head), which would mean quantizing them locally.
