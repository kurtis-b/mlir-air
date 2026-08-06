# 08d — Phase E4: `runlist`

The fine-grained point of the taxonomy: the layer decomposed into small operators, all aggregated
into one XRT runlist. It is the mode that answers "what does the runtime seam buy you when there is
a lot to aggregate", and it is the only sub-phase in Phase E with genuinely new device work in it.

Read [08b](08b-phase-e2-coarse-and-instrumentation.md) first for the artifact contract, and
[08c](08c-phase-e3-offload.md) for the blocked-attention helpers this mode shares.

## Two operators do not exist, and one of them is further from existing than it looks

iron's `runlist` decomposes into GEMM, transpose, softmax, elementwise-mul, causal-mask, GeLU,
LayerNorm, add-and-norm and elementwise-add. Most of those are already here:

| Operator | Where |
|---|---|
| GEMM | `llms/shared/builders/gemm_builder.py` + `matrix_multiplication/` |
| softmax | `programming_examples/softmax/` |
| GeLU | `programming_examples/gelu/`, and `transformer_layer/builders/gelu.py` |
| LayerNorm | `programming_examples/layer_norm/`, and `builders/layer_norm.py` |
| add-and-norm | `builders/addnorm.py`, pre-add variant built by D1 |
| elementwise-add | `builders/elementwise_add.py` |
| causal-mask | `builders/elementwise_add.py`'s `causal_mask=` path |

Two are not:

- **`transpose`.** An *example* exists in three variants — `data_transfer_transpose/dma/`,
  `.../channel/`, `.../dma_bf16/` — and `dma_bf16/` ships a `transpose.cc`. What does not exist is a
  builder, a registry row, or anything shaped like the operators in `transformer_layer/builders/`.
  Treat those three as templates for the data movement, not as something to import.
  ([08](08-phase-e-execution-strategies.md) said nothing existed; that was wrong, and the
  `dma_bf16` variant is the closest starting point.)
- **`elementwise_mul`.** Nothing. There is no `eltwise_mul/` beside `eltwise_add/`, no builder, no
  registry entry, no `compile_*` in `external_kernels.py`. The closest things are
  `silu_and_mul/` (the multiply is *fused into* SiLU and not separable),
  `primitives/vector_examples/vector_mul/` (a tutorial, no builder wrapper) and
  `eltwise_add/` (the right shape, wrong operation).

  **One hardware constraint to check before designing it:**
  `programming_examples/weighted_rms_norm/weighted_rms_norm.py:58` records that the unit "does not
  legalize f32 vector elementwise mul". If your multiply is f32, find that out on day one rather
  than at link time.

This is the only new device work left in the plan. Budget for it, and if one of the two turns out
to be a multi-day problem, **say so in `work_not_completed` and land the mode without it rather than
faking a decomposition**. A `runlist` that is honestly missing one operator, with the gap recorded,
is worth more than one that quietly folds the multiply back into a fused kernel and then claims to be
the fine-grained mode.

## Do not carry iron's entry count across

iron reports **12 kernels and 16 runlist entries** (encoder; 13/17 decoder) -- not the 29/42 this plan
repeated for weeks and told you to expect. **Re-derive both at `baseline_768` regardless.** The reason is
the same one that makes `coarse` measure 131 rather than 12: `build_addnorm_module` requires
`rows == herd_x * rows_per_call`, which at `cols = 768` caps a call at 64 of the layer's 4096 rows,
so each normalization point is 64 dispatches. Any row-parallel operator you decompose to may have the
same property, and several of them will.

The number this mode produces is a measurement, not a target. Write it in `runlist/README.md` next to
what produced it.

**Three operators were validated at 768 by D1 specifically for you** — `layer_norm`,
`elementwise_add` and `causal_mask` — and `builders/block.py` says so in as many words: they are on
the shelf, unused by the block, because Phase E's finer-grained modes decompose down to them.
`encoder_bert` uses an all-ones attention mask, so `causal_mask` stays unused here too unless you
build the `decoder_gpt2` variant, which is not this sub-phase's job.

## One runlist, and the context ceiling

The premise of this mode is Phase B's answer: N separately-compiled ELFs, N `hw_context`s, **one
runlist** across them, bit-identical to sequential dispatch and 1.02–1.15× faster
([05a](05a-phase-b-runlist-spike-result.md)). Not one shared context — XRT rejects that three ways.

**The measured concurrent `hw_context` ceiling on this device is 32.** Context 33 fails with
`RuntimeError: DRM_IOCTL_AMDXDNA_CREATE_HWCTX IOCTL failed (err=-2)`, which is loud and happens at
load time, so an overrun surfaces as an exception rather than as wrong numbers. iron's `runlist`
wants 29, which fits with three to spare.

Two caveats on that margin, both recorded in [08 §Risks](08-phase-e-execution-strategies.md):

- The probe cycled **4 distinct ELFs** to reach 32 contexts. What is demonstrated is a limit on
  concurrent *contexts*, not on 29 *distinct* ELFs. If the ceiling depends on per-ELF resources
  rather than context count alone, 29 distinct designs could bind sooner.
- Three spare is thin. Anything else holding a context concurrently eats it.

**Re-probe with your real artifacts before relying on it**, and record what you measured.

## Instrumentation

Exactly the contract in [08b](08b-phase-e2-coarse-and-instrumentation.md): `DispatchVector` from
`llms/shared/infra/dispatch.py`, `as_row()` per `run_sequence` call, emitted on the fault-injected
path as well as the clean one. No mode gets its own counting.

`require_single_submission=True` on `run_sequence` is the argument that makes "one runlist" a checked
property rather than an intention — use it where the mode claims it, and where a dispatch argument
being a whole BO forces a second submission, record that as a measurement and explain it in the
README, the way `coarse`'s four submissions are explained.

## Work items

1. A `transpose` builder in `transformer_layer/builders/`, using `data_transfer_transpose/dma_bf16/`
   as the template for the data movement.
2. An `elementwise_mul` builder and its kernel, using `eltwise_add/` for the shape and
   `primitives/vector_examples/vector_mul/` for the operation. Check the f32 legalization constraint
   first.
3. `pattern/runlist/` — the fine-grained sequence, its `README.md`, its own `KernelCache` directory
   added to `transformer_layer/.gitignore` and the `clean` target in the same commit.
4. Share `offload`'s `_blocked_attention` / `_resolve_query_block_size` rather than reimplementing
   them, so both modes block attention identically.
5. A `runlist` operator spec in the `SPECS` catalogue, through the `dispatch` seam, recording
   `execution_mode: "runlist"`.
6. `run_npu2_runlist_peano.lit` — both recipes in one file, clean and
   `--fault-inject input --expect-failure`.
7. `opcheck` specs and lit `CHECK` lines for the two new operators, held to the same numerics
   contract as every other operator: full-output `np.isclose` against an FP32 reference, zero
   mismatches, `rtol` `1.6e-2`, `atol` at most `1e-1`.

## Gate

```
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  ninja -C build-xrt check-programming-examples-transformer-layer
```

Every test in the suite passes on real hardware, including the two new operator tests and the mode
test.

The driver then, independently: re-derives `runlist`'s verdict; requires exactly one fresh `runlist`
result at the forced configuration with full-layer `n_elements` and ≥8 distinctly-named clean stages;
validates the `dispatch_vectors` contract; requires `runlist`'s summed `runlist_entries` to **exceed
`coarse`'s** — the one ordinal claim this mode owns, and the reason `coarse` had to be measured
first; re-runs `runlist` under `--fault-inject input` and requires it to **fail**; and requires the
fault run's summed vector totals to equal the clean run's.

The `runlist > coarse` clause is not a formality. `coarse` already measures 131 entries, 128 of them
one operator's row blocking, so a decomposition that folds normalization back into a fused kernel can
easily come out *below* it. If that happens the honest response is to report the number, not to
inflate the decomposition.

## Risks

- **The two new operators are the schedule risk in Phase E.** Everything else in this phase is
  re-expression or wiring; these are device bring-up.
- **The context ceiling has three of margin and an untested assumption behind it.** Re-probe.
- **A fine-grained decomposition has more places for a layout mismatch to hide.** The ≥8-stage
  per-boundary comparison is what localizes it — the same mechanism that caught D2's zeroed
  up-projection columns, where the layer output alone said only that 54% of it was wrong.
