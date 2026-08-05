# 07a — Phase D1: The operators at `baseline_768`

Phase C validated each operator at whatever width was cheapest to bring it up. The block runs at
`baseline_768`, and almost none of those points are there. This sub-phase closes that gap, so that
when the block gate fails in [07b](07b-phase-d2-block-integration.md) the failure localizes to the
integration rather than to an operator nobody had ever run at that width.

There is no new device code here and no new mechanism. Every piece exists: the builders, the
`opcheck.py` contract, the per-operator lit tests. This sub-phase adds shapes to
`opcheck_specs.py`, and one builder keyword — see [The pre-add gap](#the-pre-add-gap-the-validated-addnorm-is-not-the-one-the-block-needs),
which is the only part of this sub-phase that is not bookkeeping.

## The family, and why the sequence length is forced

`baseline_768` is the only family whose projection GEMMs resolve — 36 of 36, against 2 and 3 for
the other two. Its parameters, from iron's `study/end_to_end/cases.py::FAMILY_SPECS`:

```
hidden = 768    ffn = 3072    num_heads = 12    head_dim = 64    variant = encoder_bert
```

**The block's sequence length is 4096, and it is not a free choice.** `build_ffn_module` stitches
the up- and down-projection GEMMs into one ELF. Two same-method GEMMs with different `tile_n`
cannot share an ELF — they declare `f32_to_bf16_mn_<suffix>` twice with different memref types and
`stitch_elf` rejects the redefinition. At `hidden = 768` the up-projection (`N = 3072`) takes
`tile_n = 128` and the down-projection (`N = 768`) takes `tile_n = 96`, at every point on the
ladder, so the pair only survives where the registry puts them on *different methods*:

| seq | up-proj | down-proj | |
|---|---|---|---|
| 64 … 2048 | `drain` t_n=128 | `drain` t_n=96 | collide |
| **4096** | **`drain` t_n=128** | **`fused-cast` t_n=96** | **builds** |
| 8192, 16384 | `fused-cast` t_n=128 | `fused-cast` t_n=96 | collide |

The example README already records this ("`build_ffn_module` therefore does not build at any
`baseline_768` point except `seq = 4096`"). The real fix mints the GEMM symbol suffix per
`(method, tile_n)` rather than per method, in `llms/shared/builders/gemm_builder.py`. That file is
off limits to this study: changing it triggers the cross-deployment regression rule and would put
`make verify` over ten shipped models inside every Phase D gate.

**Do not fix it here.** Run at `seq = 4096`, and leave the limitation recorded — Phase E must clear
it before it can walk the full ladder, and that is Phase E's cost to carry, not this phase's.

## The shape set

Add these to `SPECS` in `opcheck_specs.py`. Each gets its verdict in the existing
`run_npu2_<op>_peano.lit`, which needs a `CHECK` line per new shape.

| operator | `shape_key` | `shape` | notes |
|---|---|---|---|
| `qkv_proj` | `4096x768` | `seq_len 4096, emb_dim 768` | `64x768` already passes and **does not satisfy this** — the driver requires `seq_len == 4096`. Pins `fused-cast`. |
| `ffn` | `4096x768x3072` | `seq_len 4096, emb_dim 768, ffn_dim 3072` | the only buildable point, per the table above |
| `mha_out_proj` | `4096x768x12h` | `seq_len 4096, head_dim 64, num_heads 12, causal False` | **non-causal** — `encoder_bert` is. Its o_proj GEMM is `4096x768x768`. |
| `addnorm` | pre-add, `cols 768` | `rows` per the constraint below, plus `pre_add: True` | the new variant — see below |
| `layer_norm` | `cols 768` | rows free | |
| `elementwise_add` | `cols 768` | rows free | |
| `causal_mask` | **none** | | exempt — see below |

The three GEMM-backed operators are pinned to `seq_len == 4096` because that is where the block
runs and localizing a D2 failure is the whole point of this sub-phase. The three row-parallel ones
are not, because their builders derive the legal row count rather than accepting one.

Two things about how the driver reads this table, which decide how you record your results:

- **Only a shape `opcheck.py --list` declares counts.** The check intersects the listing with
  `results/`, then re-derives `n_mismatch`, `ref_dtype`, `rtol` and `atol` from each artifact
  rather than reading its `passed` flag. `results/` is gitignored, so a file there is invisible to
  the fingerprint, the tamper check and every review diff — a declared shape is the only kind that
  anything else in the harness can see.
- **The variant goes in the `shape` dict.** Record `pre_add` the way `mha_out_proj` records
  `causal`. Naming the operator distinctly (something containing `pre_add`) is also accepted.

Three constraints on the row counts, none of which is a free knob:

- `build_addnorm_module` requires `rows == herd_x * rows_per_call`, so the herd loop runs a single
  trip. At `cols = 768` less fits in L1 than at 512, so `rows_per_call` — and therefore the legal
  row count — may fall below the 64 the existing `64x512` row uses. Derive it; do not assume 64
  transfers.

  `[2026-08-05]` **It did not fall below 64.** D1 derived and exposed the arithmetic as
  `addnorm_max_rows()`: 120 at `cols = 512`, and at 768 **104 pre-add / 80 post-add**. The warning
  was right to say "derive it" and wrong about the direction. The number that matters downstream is
  104: it is what makes each of the layer's normalization points 64 dispatches, and therefore what
  dominates `coarse`'s dispatch vector — see [08](08-phase-e-execution-strategies.md).
- `mha_out_proj` at 4096 positions is four times the sequence of the largest Phase C point at
  three quarters of its heads. Budget compile and run time accordingly; it is the expensive row
  here.
- Do **not** pass explicit `herd_m` / `herd_n` to any builder. They override the registry row and
  fail to build at the short end of the ladder.

### `causal_mask` is exempt, and that is not an oversight

Its shape is `seq × seq`, not `seq × hidden` — there is no hidden dimension in it to widen, so
"the `baseline_768` widths" does not name a shape for it. And `encoder_bert` does not use it at
all: `generate_golden_reference` builds an all-ones attention mask for the encoder variant and a
`tril` one only for `decoder_gpt2`. Adding a 4096×4096 point would cost real hardware time to prove
something the block never exercises. The driver's objective check exempts it explicitly.

## The pre-add gap: the validated `addnorm` is not the one the block needs

This is the one real piece of work in this sub-phase, and it is the thing most likely to produce a
correctly-shaped wrong answer downstream if it is skipped.

`addnorm_reference` (`builders/addnorm.py:272`) computes

```
LayerNorm(x) * weight + residual          # normalize, THEN add
```

`encoder_bert` is post-norm. Both of its normalization points are

```
LayerNorm(attn_out + input) * weight      # add, THEN normalize
```

Those are different functions, and the one that has run on hardware is the wrong one.

The kernel already supports both. `compile_addnorm_ffn(pre_add=True)` in
`llms/shared/infra/external_kernels.py:376` selects it, its docstring states the two forms
directly, and Phase A already builds `addnorm_ffn_pre_add.o` with an assertion
(`compile_kernels.py::check_pre_add_variants_differ`) that the two objects are not byte-identical —
precisely so that `-DADDNORM_PRE_ADD` silently ceasing to reach the `fused_add_layer_norm`
templates gets caught at compile time rather than as a subtly wrong activation.

What is missing is everything above the object: no builder exposes the variant, no `opcheck` spec
covers it, and it has never been dispatched. Note the current `addnorm` builder targets
`encoder.o`, built from `encoder.cc`, which has **no** `pre_add` flag — the pre-add form lives in a
different translation unit (`addnorm_ffn.cc`), so this is a builder change and not a flag flip.

Work:

1. A `pre_add=` keyword on the addnorm builder path, in the spirit of `causal_mask=` on
   `build_elementwise_add_module` — one builder, the variant named at the call site.
2. A matching reference. Do not parameterize the existing one with a branch that is easy to read
   backwards; make the ordering visible at the call site and in the docstring.
3. An `opcheck` spec for it at `cols = 768`, and a `CHECK` line for it in the addnorm lit test.

The two-output form's `out2` carries the raw pre-add sum forward as the next block's residual
stream. `encoder_bert` does not need it — its second residual is `hidden`, the *normalized* output
of the first norm, not the raw sum — so the one-output form should suffice. Confirm that against
the reference rather than taking it on faith.

## The tolerance ceiling is hard

`rtol` is pinned at `1.6e-2` and `atol` must be at most `1e-1`. The driver re-derives both from the
results artifact and rejects anything outside them; it does not read your `passed` flag.

Widening a tolerance past those is not available to you. If a shape's honest measured error will
not fit — and `mha_out_proj`'s causal rows already sit within a factor of two of the ceiling — that
is a finding to report in `work_not_completed`, with the measured `atol_required`. It is a real
result about the datapath and it is far more useful than a number chosen to make a gate green.

Follow the existing rows' practice: state the measured `mean_rel_L1`, `abs_err_max` and
`atol_required` in a comment beside each new spec, and say what margin the chosen `atol` leaves.

## Gate

```
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  ninja -C build-xrt check-programming-examples-transformer-layer
```

Every test in the suite passes on real hardware, including every test earlier phases added.

The driver then checks, independently of anything you write:

- every results file is newer than the gate's start stamp;
- the verdict is re-derived from `n_mismatch`, `ref_dtype`, `rtol` and `atol` rather than trusted;
- each operator is re-run with `--fault-inject input` and that run **must fail**;
- each operator carries a `baseline_768` shape, read from the `shape` dict rather than the
  `shape_key` string, and only from a shape `--list` declares;
- and **every `baseline_768` point gets its own fault injection** unless the per-operator control
  above already covered it. That control takes each operator's *first* declared shape, which for
  `addnorm` today is the `64x512` post-add row — so without this clause the pre-add variant, the
  one function here that has never run on hardware, would be the only one never injected. It is
  also the one whose reference is most likely to agree with the device by construction, since you
  are writing both.

## Risks

- This is real hardware time on six operator points before the block runs at all. It is still
  cheaper than debugging a chained layer whose stages were never individually checked at that
  width — that is the whole argument for this sub-phase existing.
- `mha_out_proj` at 4096 is the one point here big enough to hit L1 or scratch limits that the
  smaller Phase C rows did not. If it does, say so; do not shrink the shape to make it pass, since
  the block needs exactly this point.
