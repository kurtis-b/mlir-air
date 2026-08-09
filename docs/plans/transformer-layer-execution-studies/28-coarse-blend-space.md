# 28 — What `coarse`'s blend is a blend OF, and what selects it

`[2026-08-09]` [03](03-measurement-model.md) defines `coarse` as "reconfiguration AND
sync overhead minimized together, by *mixing* `runlist` and `fused` per workload",
with the mechanism "per-operator choice between an individually dispatched kernel
and a fused region". The README has carried that as a blocker since the taxonomy
correction: *nothing in the port expresses such a choice*, so the mode cannot be
scoped.

This document scopes it. The space is derived from the artifact plans rather than
guessed, and it is much smaller than the definition's wording implies.

## The space is two axes and six cells

**A fused region is not free-form — it is an artifact somebody stitched.** Reading
the three modes' artifact plans against each other:

| | front: `qkv_proj` | front: `mha_out_proj` | tail |
|---|---|---|---|
| `block` / `coarse` | `build_qkv_proj_module` | `build_mha_out_proj_module` | `ffn` ELF + `addnorm` × N bands |
| `fused` | `build_qkv_proj_module` | `build_mha_out_proj_module` | one stitched `ln1+ffn+ln2` |
| `runlist` | three separate projections | per head `attn_scores`→`softmax`→`attn_output`, then `output_proj` | up / GeLU / down + per-band add / LayerNorm / multiply |

**`fused` and `coarse` build their front from the same two modules.**
`fused.py`'s `_ARTIFACT_BUILD` calls `build_qkv_proj_module(cfg["seq_len"],
cfg["emb_dim"])` and `build_mha_out_proj_module(...)` — exactly what
`block_config` resolves. The two modes differ in the **tail alone**.

So the choice is not per operator. It is:

| axis | levels |
|---|---|
| **front** | `block`-form (two ELFs, q/k/v device-resident) · `runlist`-form (decomposed, per-head attention) |
| **tail** | stitched · row-banded · fully decomposed |

**2 × 3 = 6 cells.** And the corners are not new modes:

|  | tail stitched | tail banded | tail decomposed |
|---|---|---|---|
| **front `block`** | **= `fused`** | **= `coarse` today** | new |
| **front `runlist`** | new | new | **= `runlist`** |

## The finding that makes this scoping hard, stated plainly

**The space `coarse` is defined to blend over CONTAINS the two things it blends.**
`(block, stitched)` is `fused` and `(runlist, decomposed)` is `runlist`. So
"`coarse` = the best cell" does not define a distinct mode — it re-derives one of
its own endpoints and the taxonomy collapses from four points to three.

And on the evidence already in hand it would collapse *to `fused`*.
[27](27-common-ladder-result.md) measured all four modes at 512 and 1024, twice:
`fused` is **fastest on latency at both lengths** and **lowest on DRAM bytes at
both lengths**. `coarse` is defined as minimizing reconfiguration and sync
*together*, and `fused` is 1 submission with 13 sync boundaries — the minimum
available. No interior cell can beat it on the axes the mode is defined by.

**Today's `coarse` is already an interior cell**, `(block, banded)`. What it lacks
is not blendedness — it is *provenance*: nothing shows the cell was chosen rather
than inherited from D2, which is the honest content of the README's "decision
procedure that does not exist".

## What actually selects the blend: what the workload ADMITS

The resolution is in the word the definition already uses — **per workload**. The
cells are not all available at every workload, and which are available is a
measured property of the shape:

- **`fused`'s stitched tail is bounded to 256..1024.** `plane_major` packing needs
  a plane stride of `rows*cols` against the shim `aie.dma_bd` cap of 1,048,576, so
  it caps at 1365 rows (`builders/norm_tail.py:262-273`). At **seq 2048 and above
  the entire top row of the table is unbuildable.**
- **The `runlist` front's attention tiles are legal at 512 and 1024 but not 256**
  — `n % (tile_n 128 * herd_n 4) != 0` at 256 ([27](27-common-ladder-result.md)).
  So the bottom row is unavailable at 256 with the tiles as measured.
- **One xclbin for the whole layer is blocked twice over** ([03](03-measurement-model.md)),
  so no cell can fuse the front into the tail regardless of shape.

**That is the decision procedure, and it is not circular:** the blend is selected
by which cells the workload admits, and among those, by measurement.

| workload | cells available | what `coarse` is |
|---|---|---|
| seq ≤ 1024 | all six | degenerate — `fused` is available and dominates, so `coarse` has no distinct claim |
| seq ≥ 2048 | bottom row only (no stitched tail) | **the mode's real territory**: the best buildable blend when full fusion is out of range |

**`coarse` is the mode you use when `fused` does not fit.** That is a per-workload
blend in the strict sense, it is decided by a measured constraint rather than a
preference, and it explains why the D2 block — built at 4096, where the stitched
tail cannot pack — landed on `(block, banded)` in the first place.

## What this means for the corrected mode

**The corrected `coarse` should be specified and measured at a sequence length
where `fused` is NOT available**, i.e. 2048 or 4096. Measuring the interior cells
at 1024 would be measuring a degenerate case and would report `fused` wearing
another mode's name.

That inverts the sequencing the README currently implies. `coarse` was placed
after `fused` and `runlist` on the grounds that "both must be right first"; that
is still true for the *builders*, but the mode's own measurement does not belong
at the length the other three now share.

**The three interior cells to measure, at 2048/4096:**

| cell | front | tail | status |
|---|---|---|---|
| C1 | `block` | banded | today's `coarse` — the incumbent, already gated at 4096 |
| C2 | `block` | decomposed | new |
| C3 | `runlist` | banded | new |

`(runlist, decomposed)` at those lengths is `runlist`, which is already gated at
4096 and is the fourth calibration point.

## What each new cell costs, measured against the code

Not free, and the reason is buffer plumbing rather than builders:

- **The region dispatch lives in closures.** `prepare_runlist`'s `_run_projections`
  / attention / tail steps and `prepare_fused`'s `dispatch` are nested functions
  inside their `prepare_*`, not module-level. Composing a new cell means
  extracting them to module scope **without behaviour change**, and proving that
  by re-running `run_npu2_runlist_peano.lit` and `run_npu2_fused_peano.lit`.
- **`fused`'s front and tail are coupled through packed buffers**, not a plain
  handoff: `packed1` plane 1 is `x` and plane 0 is written on device by entry 2,
  and the stitched tail reads the pair. Any cell pairing a `runlist` front with a
  stitched tail must write the front's output into that plane. (This is why C2 and
  C3 — which keep the banded or decomposed tail — are the cheap new cells, and any
  `(runlist, stitched)` cell is the expensive one. It is also moot at 2048+, where
  no stitched tail builds.)
- `builders/block.py`'s region functions (`_sequence_a`, `_sequence_norm`,
  `_sequence_ffn`) **are** module-level and reusable as-is, which is what makes C1
  and C2's front free.

## The one thing not to do

Do not measure the six cells at 1024 and declare a winner. Every cell there is
dominated by a cell that already has a mode name, and the result would be a fourth
mode that is `fused` with extra steps — the exact failure the taxonomy correction
was made to prevent.
