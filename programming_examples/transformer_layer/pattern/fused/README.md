# `fused` — MLIR-level fusion via `stitch_elf`, one runlist submission

The most fused point of the Phase E execution-strategy taxonomy, and the one
that uses MLIR-AIR's own production mechanism: `stitch_elf`
(`llms/shared/infra/stitching.py`) splices independently built operator
modules into one multi-launch module before compilation, exactly as the
shipped LLM prefill/decode pipelines do. Measuring it is what makes this port
additive rather than a duplicate of iron — iron reaches one-artifact-many-
kernels through `aiecc --xclbin-input` incremental merge, which MLIR-AIR does
not need and this mode deliberately does not reproduce.

The CSV `execution_mode` value is `fused_elf`, mapped in one place
(`pattern/__init__.py::EXECUTION_MODE_CSV`).

> **`[2026-08-10]` The measured rows below are 4096-era and are withdrawn as comparisons; the
> mode now gates at 1024.** The SPECS row moved 4096 → 1024 on 2026-08-08 — the stitched tail's
> `plane_major` packing exceeds the shim `aie.dma_bd` cap above 1365 rows, so the 4096 build
> raises before aircc
> ([26 §6](../../../../docs/plans/transformer-layer-execution-studies/26-mode-rebuild-feasibility.md)).
> The repair run's vector at 1024 is `submissions 1 entries 3 air 11 herd 23 sync 19 bytes
> 56626176` (the down-projection resolves to `drain` there, so the tail takes 11 whole-tensor
> args instead of 16), and its numerics are `mean_rel_L1` 1.756e-2 at `atol_required` 5.813e-2 —
> a 1.72× margin, not §Numerics' 1.27×. The §What it measures row (1/3/16/24/19/184,025,088) and
> every comparison it makes against `coarse`'s 402-boundary figures are **suspended, not
> restated**: the two SPECS rows now sit at different lengths. Build cross-mode tables from a
> ladder run
> ([27](../../../../docs/plans/transformer-layer-execution-studies/27-common-ladder-result.md)),
> never from this file. The structure — three ELFs, three dispatches, one submission — is
> unchanged and current.
>
> **One unreconciled pair, recorded rather than averaged:** at the same 1024, the 2026-08-09
> ladder reads `sync 13 / bytes 42,467,328` against the repair run's `19 / 56,626,176`.
> Candidate mechanism, unmeasured: the gate's per-boundary verification readbacks sit inside its
> measured sequence and the ladder's production dispatch omits them. Do not quote either pair as
> the mode's vector without a fresh run.

## What this mode isolates

The removal of *intermediate host synchronization*, which is what MLIR-level
fusion **is**. The layer executes as ONE `KernelCache.run_sequence` call —
one runlist, one host submission, forced with
`require_single_submission=True` — over three ELFs:

```
entry 1   qkv_proj       the D2 module, unchanged
entry 2   mha_out_proj   the D2 module, unchanged (attention + o_proj)
entry 3   fused_tail     NEW: one stitched module — residual add, LayerNorm,
                         gamma multiply (ln1), the whole staged FFN, residual
                         add, LayerNorm, gamma multiply (ln2) — ten launches
                         over whole [4096, 768] tensors
```

Every intermediate is device-resident: q/k/v flow from entry 1 to entry 2 and
`attn_out` from entry 2 to entry 3 through the BO pool, and inside
`fused_tail` the normalization sums, the FFN staging and `hidden` (which
feeds both the FFN and the second residual add) are shared func args no host
sync ever touches. `coarse` pays 402 sync boundaries, 386 of them restaging
64-row `addnorm` bands through the host at the two normalization points,
because a dispatch argument is a whole BO; fusing the norms into a stitched
module over whole tensors removes all of that.

## Why three ELFs and not one — the mode's own finding

A whole-layer single-module stitch is **not** blocked by symbol collisions:
E1's `(method, tile_n)` naming fix removed those, and `fused_tail` co-links
the FFN's `drain` (tile_n 128) and `fused-cast` (tile_n 96) GEMMs beside six
more launches without a redefinition.

**That co-link is not, however, evidence for the E1 fix**, and an earlier
draft of this file called it "the proof", which it is not. Those two GEMMs
are on *different methods*, so their symbols differed before E1 as well
(`_m32` against `_m64`) — nothing here would have collided either way. The
collision E1 removed is same-method-different-`tile_n`, and at `seq = 4096`
the registry happens to place the FFN's two projections on different methods,
which is exactly why 4096 was the one ladder point that built at all. The
real evidence for the fix is the `seq = 64` `ffn` point E1 added, where both
projections resolve to `drain` at `tile_n` 128 and 96 and the pre-E1 names
would have been identical.

What blocks it is **backend settings**. One ELF is one aircc invocation, and
FlashAttention requires `omit_pingpong="all"` +
`runtime_loop_tiling_sizes=[1, 1]` — it does not place otherwise — while the
4096-row GEMMs require `[2, 2]` for BD-ID recycling.
`builders/mha_out_proj.py` documents the two settings as non-interchangeable
(a placement failure at best, wrong numbers at worst), so attention keeps its
own ELF, `qkv_proj` — which must execute *before* attention — keeps its own,
and everything after attention fuses into one module. This is also exactly
the shape of MLIR-AIR's production pipelines: no shipped deployment stitches
FlashAttention into a GEMM module either. The ELF ABI then aggregates all
three entries into one runlist (05a §5), so the layer is a single submission
regardless.

## Why the normalization is streamed, not row-blocked

`build_addnorm_module` caps a launch at 104 rows of 768 (L1, one kernel call
per tile), so reusing coarse's operator inside one module would need 64
launches per normalization point, each reading a 64-row *band* of a whole
tensor — and a band at a nonzero row offset cannot be routed into a slice's
args clause: `memref.cast` cannot cast an offset subview back to the identity
layout the launch signature declares. (The row-0 subview trick in
`o_gemv_ffn_multi.py` works only at offset 0.)

The decomposed `elementwise_add` / `layer_norm` / `elementwise_mul` builders
have no such cap — each walks all 4096 rows in one launch, and all three are
validated standalone at exactly 4096×768 in the SPECS catalogue. They are the
same decomposition `runlist` measured clean at every boundary, streamed
rather than banded. Note what this means for the two *recorded, not gating*
predictions: the faithful stitch did **not** row-block its normalization, so
`fused.runlist_entries < coarse.runlist_entries` holds trivially (3 < 131),
and `air_launches` lands at 16 ≥ coarse's 12 because the ten-launch tail
counts once as one ELF.

## What it measures

One recorded `DispatchVector` row — the whole layer is one sequence:

```
host_submissions  runlist_entries  air_launches  herd_launches  sync_boundaries      bytes
       1                3              16             24             19          184,025,088
```

The fault-injected twin totals identically — injection perturbs one input
element after the reference exists and never touches the dispatch path.

Reading the row against `coarse` (4 / 131 / 12 / 146 / 402 / 202,902,528):

- **19 sync boundaries against 402** is the gating clause this mode owns
  (`fused.sync_boundaries < coarse.sync_boundaries`): 9 host uploads (x, the
  six weights/broadcasts, and the two zero-filled f32 scratches — qkv's and
  the down-projection's) plus 10 boundary readbacks, none of them
  intermediate restaging.
- **16 air launches on 3 artifacts against 12 on 4** is the signature 08e
  predicts for this mode: `air_launches` counts launches *in the compiled
  module* once per distinct ELF, so fusing ten launches into `fused_tail`
  raises the count while the artifact count falls.
- **24 herd launches against 146** is the same asymmetry from the other side:
  `herd_launches` accumulates per dispatch step, and this mode has three
  steps where coarse re-dispatches its addnorm ELF 128 times.
- The bytes differ mostly by the two `[4096, 768]` gamma broadcasts this mode
  uploads (the streamed `elementwise_mul` takes two full tensors) against
  coarse's band restaging.

## Numerics

Entries 1 and 2 are the block's own modules, so q/k/v and both attention
boundaries reproduce the block's error band. The streamed add/ln/mul tail is
the same per-row arithmetic `runlist` measured clean at banded granularity —
LayerNorm's loop walks 512 rows per tile instead of 8, but every row's
arithmetic is independent. The whole-layer comparison measures `mean_rel_L1`
1.784e-2 at `atol_required` 7.896e-2 — a 1.27x margin under the 1e-1
ceiling, the thinnest of the four modes, and the composition says why:
device attention (whose error the host-attention modes avoid; block measures
1.688e-2) **plus** the decomposed norm tail (which stages bf16 between add,
LayerNorm and gamma multiply where the fused `addnorm` does not; `runlist`,
the same decomposition banded, measures 1.732e-2). The two effects stack —
block 1.688e-2 / runlist 1.732e-2 / fused 1.784e-2 — with every boundary
still at `n_mismatch` 0. That is a real, small numerical cost of this
fusion's norm decomposition, measured rather than defined away.

`[2026-08-07]` Refreshed after J7a moved `layer_norm_rows` to f32 two-pass
statistics: was 1.806e-2 at `atol_required` 7.572e-2, a 1.32x margin. Worth
noting that the mean improved while the margin **tightened** — `mean_rel_L1`
is an average and `atol_required` is a worst-element statistic, so they move
independently, and this mode has the least room of the four either way.

`[2026-08-10]` Superseded at the mode's current length: the 1024 repair run
measures `mean_rel_L1` 1.756e-2 at `atol_required` 5.813e-2 — a 1.72× margin
(banner above). The 1.784e-2 / 7.896e-2 / 1.27× here is the 4096-era figure
and survives as cell C2's
([30](../../../../docs/plans/transformer-layer-execution-studies/30-coarse-cells-built.md)).

## What it costs

A third full-layer compile in the lit suite: `fused_cache/` is this mode's
own ELF cache (`KernelCache` picks the directory by NAME, and two modes
pointed at one directory can trade ELFs whose fingerprints happen to agree —
numerically valid output attributed to the wrong execution boundary). The
cache is gitignored and in `make clean`, in the same commit that created it,
because the driver's negative control runs `opcheck.py` from the source
directory and the cache lands there — exactly the leak D2's `block_cache/`
had. The attention ELF is the dominant compile cost, and it is compiled again
here rather than shared with `block`'s or `coarse`'s cache, deliberately.
