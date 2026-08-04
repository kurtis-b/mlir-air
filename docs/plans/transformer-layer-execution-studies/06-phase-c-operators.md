# 06 — Phase C: Operators

Re-express iron's six new operators as AIR builders, validate each on real NPU2 against an FP32
reference, and register every validated `(kernel, shape)` in `kernel_registry`.

The `design.py` files (~4.4k lines against `aie.iron` ObjectFifo / Worker / Runtime /
TensorAccessPattern / SequentialPlacer) are the part that must genuinely be rewritten — they have
no counterpart in AIR's launch-segment-herd model.

**This document is the overview. It is not any session's specification.** Phase C runs as four
sub-phases, each with its own spec, gate and objective check:

| Sub-phase | Spec | Covers |
|---|---|---|
| C1 | [06a](06a-phase-c1-gate-and-small-operators.md) | The gate mechanism, `causal_mask`, `addnorm`, `layer_norm`, `elementwise_add` |
| C2 | [06b](06b-phase-c2-qkv-proj-and-ffn.md) | `qkv_proj`, `ffn` |
| C3 | [06c](06c-phase-c3-mha-out-proj.md) | `mha_out_proj` — the largest rewrite |
| C4 | [06d](06d-phase-c4-coverage-sweep.md) | The registry sweep and shape coverage |

It is split because Phase B was 3,725 lines and took 362 minutes with blocking review findings
still open after round 3; Phase C's source material is 8,160 lines across five rewrites. One
session cannot hold it, and a late failure would re-run everything.

## Shape of the result

Per convention rules 1, 3 and 4, each iron `<op>/{op,design}.py` pair collapses into a single
`build_<name>_module(...)` function returning an `air.ir.Module`. There is no operator class and
no artifact-DAG file-loading seam. The FP32 reference lives in the same module as a module-level
function, following `programming_examples/weighted_rms_norm/weighted_rms_norm.py`.

New builders live in `programming_examples/transformer_layer/builders/` and **call** into
`llms/shared/builders/` without modifying it. Modifying `shared/` triggers the cross-deployment
regression rule in [13](13-verification-and-acceptance.md#the-cross-deployment-regression-rule) —
`make verify` on all ten shipped models — which is what made Phase B six hours long.

## The operators

| iron operator | design.py | Approach in MLIR-AIR | Sub-phase |
|---|---|---|---|
| `causal_mask` | — (86 L op.py) | **Not an operator.** Pure composition: elementwise-add with a torch-precomputed triangular mask. Becomes a builder keyword argument. | C1 |
| `addnorm` | 382 | Weighted LayerNorm + residual. **Change from iron:** it bakes weights into the MLIR via `np.load()` at generation time and hashes them into the artifact name. Pass weights as runtime memref arguments instead — otherwise every weight change forces a recompile. | C1 |
| `qkv_proj` | 561 | GEMM `A(M,K) @ B(K,3K)` with C split three ways at the runtime-sequence level. Closest existing analogue: `shared/builders/rms_gemms_rope_multi.py` minus RMSNorm and RoPE. | C2 |
| `ffn` | 1096 | Staged up-projection → fused GeLU → down-projection with `down_proj_depth` memory-tile accumulation staging. `programming_examples/ffn_swiglu/` is SwiGLU-shaped; this is GeLU-shaped. | C2 |
| `mha_out_proj` | 1350 | Largest. Fused attention + output projection, optional causal masking, `parallel_seq` / `parallel_heads` / `o_proj_acc_depth` knobs. Compose from `flash_attention/kernel_fusion_based/` plus the O-projection half of `o_ffn_multi.py`. | C3 |
| `dynamic_gemm` | 1009 | **Not ported.** Runtime M/N tail handling was one of three candidate answers to shape coverage; the sweep in C4 is the answer taken. See [06d](06d-phase-c4-coverage-sweep.md). | — |

Convention rule 5 applies: `mha_out_proj` (1350) and `ffn` (1096) both exceed the repository's
~800-line norm and must be split along their internal staging seams.

## The numerics standard — do not port iron's

`[Amended 2026-08-04]` An earlier version of this document said the `reference.py` torch oracles
"port verbatim". They must not. Reading them against
[13](13-verification-and-acceptance.md#correctness-standards-in-this-repository) shows three
divergences, each of which makes the gate laxer while looking compliant:

| | iron | This port |
|---|---|---|
| Reference dtype | **bf16** — `torch.rand(..., dtype=bfloat16)`, `torch.matmul` on bf16 | **FP32** from bf16-rounded inputs |
| Tolerance | `REL_TOL=4e-2`, `ABS_TOL=1.5e-1` (`block/run.py:66-72`) | the registry's `rtol` / `atol` |
| Mismatch budget | `ERROR_THRESHOLD=0.005` — 0.5% of elements may exceed tolerance | none; zero mismatches |

A bf16 reference "agrees" with a bf16 device result partly because both are wrong in the same
direction, and at `K=4096` the accumulated error is not small. Two further traps in the same
files:

- iron's FFN oracle uses `torch.nn.functional.gelu`, the **exact erf** form, while the ported
  kernel is `gelu_tanh_approx_bf16` (`transformer_layer/kernels/elementwise.cc`). At iron's
  tolerances the difference is invisible; at the registry's it is not. The reference must use the
  tanh approximation.
- iron's MHA oracle computes bf16 SDPA below `seq_len 16384` and FP32 chunked attention at and
  above it, so the reference's own precision changes across the ladder. Compute chunked FP32 at
  every length.

The registry's own methodology (`kernel_registry/README.md`) is the rule: hold `rtol = 1.6e-2`
fixed and size `atol` to the kernel's measured worst-case absolute error.

## Shape coverage

`registry_lookup.gemm_config()` **raises** on an unmeasured `(M, K, N)` rather than guessing —
deliberately, because hand-copied tile configs previously caused drift bugs. The two registry
JSONs hold 40 measured shapes (33 bf16-out + 7 f32-out).

`[Amended 2026-08-04]` This document previously estimated iron's matrix at "6 families × 9
sequence lengths × ~8 GEMM roles — several hundred distinct shapes … an order of magnitude more
than the registry has ever held". Measured against
`iron/applications/transformer_layer/study/block/cases.py`, it is smaller:

- The three decoder families are `dataclasses.replace()` clones of the three encoder families, so
  there are **3 distinct shape families**: hidden ∈ {512, 768, 1024}, ffn ∈ {2048, 3072, 4096},
  `head_dim = 64` throughout.
- `BLOCK_KINDS` holds **7** kinds, not 8. `causal_mask` is a `BlockCase` field with a benchmark
  function but is not a valid `--block` choice; iron's own full-suite run has **0** rows for it.
- iron's full-suite `block/results.csv` is 486 candidate rows, further pruned by
  `removed_cases.csv`.

Enumerating the distinct projection-GEMM triples gives **108** — `qkv_proj`, `ffn_up`, `ffn_down`
and `o_proj`, 27 each — of which **5 are already registered and 103 are missing**. The attention
GEMMs go through FlashAttention rather than `gemm_builder` and need no GEMM registry row.

So coverage is a ~3× registry expansion over an enumerable set. C4 builds the sweep that produces
it; see [06d](06d-phase-c4-coverage-sweep.md) for what is staged and what is deferred.

## Gate

Three conditions, all required, applied per sub-phase to the operators that sub-phase lands:

1. **Numerics** — each operator matches an **FP32** reference under full-output `np.isclose` at
   the registry's `rtol` / `atol`, with zero permitted mismatches. Not cosine similarity: the
   `kernel_registry` README is explicit that a correlation gate is blind to a systematic
   per-element scale error.
2. **Registration** — a row appended to both `kernel_registry/supported_kernels.md` and
   `details/<Kernel>_bf16.md`, carrying `mean_rel_L1`, `Used by` and status.
3. **Coverage** — every shape the sub-phase claims is either registered or provably covered by an
   explicitly injected, recorded spec.

Each sub-phase additionally faces a driver-side **negative control**: the driver re-runs the
operator with a fault injected into its input and requires the check to FAIL. See
[06a](06a-phase-c1-gate-and-small-operators.md) for the mechanism, which C2–C4 reuse.

## Three L3 input streams per tile miscompile in a multi-trip herd loop

`[Recorded by C1, 2026-08-04]` Every remaining operator in this phase takes at least three
inputs, so this is load-bearing for C2 and C3 rather than an `addnorm` curiosity.

A herd whose body streams **three distinct L3 buffers into L1** and loops more than once produces
wrong numbers. Measured on NPU2 with `fused_add_layer_norm_2outs` at `[8, 64]`, `herd_x = 1`: one
trip through the loop is exact (0 of 512 elements outside `rtol = 1.6e-2, atol = 5e-2`), two trips
give 491 of 512. It is unchanged by fetching the loop-invariant input inside the loop or hoisting
it out, by draining or discarding the unused second output, by `omit_pingpong="L1"` or `"all"`, and
by either `use_lock_race_condition_fix`. An AIE2P column has two shim MM2S channels; the
two-stream builders beside it — multi-row `layer_norm` here, and `_build_add_2d_to_2d` in
`llms/shared/builders/o_ffn_multi.py` — loop correctly for as many trips as you like.

`build_addnorm_module` raises rather than emitting the broken form, because the symptom is
partly-correct values and reads as a tolerance problem. The workaround C1 took is a row cap:
`rows == herd_x * rows_per_call`, one call per tile, which at `cols = 512` over the full herd is
64 rows.

That cap does not scale to C2/C3's shapes. The two candidate fixes, neither attempted:

- **Stage the loop-invariant operand through L2.** L3→L2 once at segment scope, then L2→L1 inside
  the herd, which costs a memtile channel rather than a shim one. This is how the GEMM builders
  already feed 32 tiles from three L3 operands.
- **Fold two operands into one L3 buffer** so a single strided DMA fetches both, and hand the
  kernel subviews. Cheaper to write, but it needs an extern call to accept an offset memref, which
  nothing in this repository does yet.

Budget for one of them before assuming a C2/C3 operator will simply loop.

## Risks

- `mha_out_proj` is the largest single rewrite in the port and depends on FlashAttention
  behaviour that Goal 1 will later modify. Coordinate the two.
- The sweep in C4 is bounded but long. It must be resumable, because `PL_STEP_TIMEOUT` is 3 hours.
- Touching `shared/builders/` affects the shipped LLM deployments. The sub-phases are scoped so
  that none needs to; if one does, re-run `make verify` across all ten.
