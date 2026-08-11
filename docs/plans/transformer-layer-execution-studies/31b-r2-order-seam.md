# 31b — R2: attaching the norm tails, and how the order seam actually resolves

`[2026-08-11]` Scoping output for increment **R2** of [31](31-fused-resident-tail.md)'s resident
fused tail. R2 attaches nt1 and nt2 to R1's one-segment FFN interior and deletes the `hidden`
family and `ffn_out` crossings — the remaining **9.0 MiB @1024** of [31a](31a-resident-byte-floor.md).
It gates strictly after R1, and R1's gate is parked on queue item 6b.

**No device was dispatched for anything in this document.** Every claim below is one of:

- **MEASURED** — with the probe, the arm and the number, hermetically at aircc pass-dump altitude;
- **SOURCE** — quoted from the compiler or target-model source, with the file;
- **DESIGN INTENT, UNVERIFIED** — stated as such, with what would verify it.

**Everything derived from a compiler dump here is PROVISIONAL pending queue item 6b.** The item-6b
fix changes shim BD emission; [31 §The gate ran](31-fused-resident-tail.md) already recorded that
structural literals taken from a pre-fix dump of a ≥3-clique module had to be *re-derived, not
compared against*. The same discipline applies to this document: re-run the probes after 6b lands
and re-derive. Both probes print the `air-opt` binary mtime for exactly this reason — the tree was
rebuilt underneath this session twice (`2026-08-11 11:05:06` → `13:03:54` → `13:06:01`), and a
number without that stamp cannot be placed.

---

## 1. What R2 must buy, to the byte

From [31a](31a-resident-byte-floor.md)'s itemized crossing table, the rows R2 removes:

| # | crossing | @512 | @1024 |
|---|---|---|---|
| 13 | `hidden` write | 786,432 | 1,572,864 |
| 14 | `hidden` mirror write (`packed2` plane 1) | 786,432 | 1,572,864 |
| 15 | `hidden` read (FFN A) | 786,432 | 1,572,864 |
| 22 | `ffn_out` write (`packed2` plane 0) | 786,432 | 1,572,864 |
| 23 | `packed2` read (`ffn_out` + `hidden` mirror) | 1,572,864 | 3,145,728 |
| | **R2 total** | **4,718,592 (4.5 MiB)** | **9,437,184 (9.0 MiB)** |

R1 removes 24.0 MiB @1024 (`ffn_up` ×2, `ffn_gelu` ×2); R1 + R2 together are the full 33.0 MiB
tail-internal figure, 48.9 % of everything residency can ever remove. `packed1` (row 11), both
gammas, both FFN weights and the `output` write stay — they are the tail's floor.

**The instrument warning is unchanged and load-bearing:** `bytes_transferred` counts host syncs and
will not move for any of this. Do not present its fall as the residency result.

---

## 2. The order seam, restated precisely — and why doc 31's prediction is too weak

[31 §The three seams](31-fused-resident-tail.md) states seam 2 as *"the norm-tail herds partition
rows BY COLUMN … while the ring consumes 64-row bands fed through one memtile"*, and predicts:
**"the row→band re-mapping is the only shape that obeys the rules."**

That prediction names the wrong side of the seam. The problem is finer than band ordering, and it
survives any row→band re-mapping:

- **A norm is a row-wise operation.** LayerNorm's statistics run over all `emb` columns of a row, so
  a norm tail can only ever emit **whole rows**, in whatever row order you like.
- **A GEMM's A operand is a blocked column strip of ALL its rows.** R1's up feed consumes
  `hidden[0:64, 32k' : 32k'+32]` in 8×8 microtile order, for `k' = 0…23`, four sweeps over.

Producing rows and consuming column strips is a *transpose of tiling*, not a reordering. No
assignment of rows to producer columns turns one into the other; the only thing that converts them
is **buffering all the rows of the band somewhere and re-reading it at a per-`k'` offset** — and
that read is precisely [23 §Never read a staged buffer at a per-iteration offset](23-rules-and-open-items.md).
That is the box doc 31 correctly identified and could not open.

### The resolution: re-map the FFN side, not the norm side

**Partition the GEMM herds by ROWS (M) instead of by output columns (N).** Then consumer core `c`
owns exactly the rows producer core `c` emits, nothing is reordered between them, and what is left
of the retile is *local to one core* — which makes it **compute, not a BD**, so the frozen-BD rule
does not reach it at all.

```
nt1 col c  ─L1→L1─►  up col c  ─L1→L1─►  gelu col c  ─L1→L1─►  down col c  ─L1→L1─►  nt2 col c
   rows R_c            rows R_c            rows R_c              rows R_c             rows R_c
                        └── in-core vector retile: row-major tile → blocked [M, tile_k]
```

with `R_c = [c·rows_per_core, (c+1)·rows_per_core)` and `rows_per_core = band / herd_x`.

Three consequences worth stating before the measurements:

1. **`hidden` never materializes anywhere** — not in DRAM, not in L2, not even whole in one L1
   buffer. It is `rows_per_call`-row tiles in flight, exactly as nt1 emits them.
2. **The down projection's C round trip disappears with it.** [31 seam 3](31-fused-resident-tail.md)
   predicted this (*"with the output on-chip, C is L1-resident trivially and what survives of J7b is
   the in-place kernel + first-iteration zero mechanics, not the hoist"*) and R2 is where it lands:
   a row-partitioned down core owns `y[rows_per_core, emb]`, which fits L1, so there is no C DMA to
   hoist and R1's four `shim→core` + four `core→shim` flows are gone.
3. **The GeLU→down memtile fan-out disappears too.** R1 needed it because *every* down core consumed
   *every* H chunk ([31 R1 status](31-fused-resident-tail.md)'s port-arithmetic correction). Row-
   partitioned, down core `c` consumes only its own column's chunks: one `core→core` edge.

**All herds must be the same width.** If nt1 ran at `herd_x = 2` and the FFN at 4, nt1 column 0's
rows would have to reach up cores 0 *and* 1 through one channel index with different data per
consumer — not expressible. This is a hard precondition of the whole design.

---

## 3. What was measured

Two probes, both hermetic (no NPU, no Peano, no kernel objects — aiecc writes every MLIR pass dump
before it compiles core ELFs, [23 §5](23-rules-and-open-items.md)):

- `agents/probes/probe_r2_order_seam.py` — one design arm, three controls, **all three controls
  verified failing**.
- `agents/probes/probe_r2_segment_budget.py` — the herd budget swept to its refusal, and the
  per-column shim census counted on both flow kinds.

Shape throughout: one 64-row band, `emb` 768, `ffn` 3072, `herd_x` 4, `tile_k` 32 → `rows_per_core`
16, four producer tiles of 4 rows, `group_n` 192.

### 3.1 The design arm routes — MEASURED

`probe_r2_order_seam.py --arm row_tiles`, `air-opt` mtime `2026-08-11 13:06:01`:

| clause | measured |
|---|---|
| routes to a final AIE design | **yes**, 59 dumps in 0.5 s, one tile-bearing `aie.device` |
| packet-typed channels, every dump | **0** |
| `core→core` flows | **4** = `herd_x` — one L1→L1 producer→consumer edge per column |
| band tile buffers surviving per consumer core | **4** of `rows_per_call·emb` = 3072 elements each |
| worst core L1 | **44,032 B of 65,536** |
| distinct `aie.dma_bd` offsets, core and memtile | **{0}** — no BD anywhere carries an offset |

The last row is the point. In this shape **nothing addresses a staged buffer at all**: every
channel transfer moves a whole buffer, and every offset that varies is either an L3-side launch-
argument offset (which the runtime sequence materializes per task) or a `memref.subview` feeding
`vector.transfer_read`/`transfer_write`, which is compute.

The worst core's 44,032 B decomposes as 4 band tiles (24,576 B) + C accumulator (6,144) + A (1,024)
+ B (12,288), with **nothing ping-ponged** — the same finding R1 recorded for its own composition.
DESIGN INTENT, UNVERIFIED: if a later change makes aircc ping-pong A and B, the core goes to
57,344 B, still under 64 KiB; if it also ping-pongs the band tiles it does not fit, and aiecc's
allocator refuses loudly.

### 3.2 CONTROL 1 — the obvious form is a SILENT MISCOMPILE. MEASURED

The natural way to assemble the band is one `rows_per_core·emb` L1 buffer filled by four
`ChannelGet`s at **literal** offsets. Literal offsets are exactly what the frozen-BD rule permits,
so this looks safe, and it compiles and routes.

It is wrong. Bisected across the dumps: the band survives through `pass_028`, and **`pass_029`
`air-shrink-memref-sizes-by-access`** rewrites it:

```
pass_028:  air.channel.get @r2_rows[...] (%results[6144] [3072] [1]) : (memref<12288xbf16, 2 : i32>)
pass_029:  air.channel.get @r2_rows[...] (%results[6144] [3072] [1]) : (memref<3072xbf16, 2 : i32>)
```

The type shrinks to **one get's size**; the gets keep offsets 3072 / 6144 / 9216 into it, and the
retile's reads keep addressing the full band — the final routed dump has a
`memref<3072xbf16, 2>` read at `affine_map<()[s0, s1] -> (s0 * 768 + s1 * 32 + 6144)>`, i.e. up to
element 12,256 of a 3,072-element buffer. **No error, no warning, no diagnostic on any decline
path.**

Mechanism (SOURCE, `mlir/lib/Transform/AIRDependencyScheduleOpt.cpp`,
`ShrinkMemrefSizesByAccessPattern`): the pass takes `overall_access_bounds` from
`air::getDataAccessShapeFromMemcpyOp`, compares it per dimension against the memref shape, and
shrinks when it is smaller. On a 1-D L1 buffer whose channel users each transfer 3072 elements, the
bound is 3072 regardless of the offsets, and the `memref.subview` / `vector.transfer_*` fix-ups it
then runs do not fail — they leave the dynamic-offset reads pointing past the new end.

This is the same *class* as the frozen-BD trap — a construction that passes every compile-time
check and returns wrong data — and it is why the design uses **one L1 buffer per producer tile**,
each filled whole. Then no buffer is bigger than the transfer that fills it and the pass is a no-op.
The probe asserts the breakage, so if the pass is fixed the control fails loudly and the design gets
re-derived rather than silently inheriting a better option.

### 3.3 CONTROL 2 — doc 31's own "stage bands in L2" candidate refuses at the memtile BD budget. MEASURED

`--arm l2_staged` builds exactly what [31 seam 2](31-fused-resident-tail.md) named: the band in a
memtile, the up feed reading it per `k'` step. It does not reach the frozen-BD question:

```
error: 'aie.memtile_dma' op has more than 48 blocks
```

48 is `getNumBDs(MemTile)` (SOURCE: mlir_aie `AIETargetModel.h`, `AIE2TargetModel::getNumBDs` —
MemTile 48, everything else 16). This is the **memtile analogue of wall 4** ([31 §The gate
ran](31-fused-resident-tail.md)'s shim exhaustion at 16), and it prices the escape hatch too: the
"just Python-unroll the `k'` feed to literal offsets" dodge needs `k_steps · herd_x` = 96 memtile
BDs against 48. Loud and deterministic, which is the failure shape H9/J1 prefer — but a refusal all
the same.

### 3.4 CONTROL 3 — a NEW compiler crash, with a builder-side dodge. MEASURED

`--wloop nested` writes the `w_up` refill as the natural `(group, k')` loop nest, which makes its
L3-side offset a **two-symbol** `affine.apply`:

```
#map1 = affine_map<()[s0, s1] -> (s0 * 589824 + s1 * 24576)>
```

When `air-split-l2-memref` decides to split that refill's L2 buffer across memtile columns it calls
`tileChannelOpByFactor`, which composes with the existing apply and then builds the replacement map
with **exactly one symbol**:

```cpp
// mlir/lib/Transform/AIRMiscPasses.cpp, tileChannelOpByFactor
AffineExpr add = originalExpr + original_map.getResult(0);
return AffineMap::get(0, 1, add);          // one symbol, whatever originalExpr uses
```

MLIR then asserts:

```
air-opt: mlir/lib/IR/MLIRContext.cpp:1237: static mlir::AffineMap mlir::AffineMap::get(...):
  Assertion `willBeValidAffineMap(dimCount, symbolCount, {result})' failed.
  #11 xilinx::tileChannelOpByFactor(...)
  #12 xilinx::AIRSplitL2MemrefForBufferConstraintPass::runOnOperation()
```

Deterministic on the round-tripped pre-split dump — `air-opt
--pass-pipeline='builtin.module(func.func(air-split-l2-memref))' pass_020_after_cse.mlir` aborts
**5/5** (SIGABRT, exit 134), which is the discipline
[31 §R1's gate is BLOCKED](31-fused-resident-tail.md) settled on after the fuse-channels crash
turned out to be an ASLR coin toss under aircc: replicate on the round-tripped dump, not through
the driver.

**This is not a corner.** [23](23-rules-and-open-items.md)'s standing rule is *advance on the L3
side*, and a two-level loop nest over an L3 operand produces a two-symbol map as a matter of course
— R1's own dump carries the identical `#map2 = ()[s0, s1] -> (s0 * 589824 + s1 * 24576)` on its
`w_up` refill and survives only because the pass declines to split that buffer there. R2 changes
which buffers the pass splits, and the defect surfaces.

**Reported, not fixed in-phase** ([31 §Design rules](31-fused-resident-tail.md): a compiler defect
exposed here is reported with its minimal shape). The apparent one-line shape of the fix is to size
the map from the operand count rather than hardcoding 1 — but that is the compiler phase's call,
not this document's.

**The builder-side dodge is exact, not a workaround.** The refill address is linear in
`g·k_steps + k`, so one flattened loop expresses the same transfers in the same order with a
one-symbol map:

| `--wloop` | result |
|---|---|
| `nested` | 21 dumps, **abort** in `air-split-l2-memref` |
| `flat` | **59 dumps, routed, 0.4 s** |

`flat` is the probe's default and is a precondition on the R2 builder. It costs nothing: same
transfers, same order, same L3-side advance.

### 3.5 The herd budget — MEASURED, and it rules out J7a's pipelines on both tails

NPU2 is 8 columns × 4 core rows = **32 core tiles** (SOURCE: `AIETargetModel.h`,
`BaseNPU2TargetModel::rows()` = 6 — one shim, one memtile, four core — and
`NPU2TargetModel::columns()` = 8). `probe_r2_segment_budget.py --arm herds` sweeps N herds of
`[4, 1]` in one segment:

| N | tiles | placed | rows used |
|---|---|---|---|
| 2 | 8 | 8 | {2: 0–7} |
| 4 | 16 | 16 | {2: 0–7, 3: 0–7} |
| 6 | 24 | 24 | {2, 3, 4} full |
| **8** | **32** | **32** | **{2, 3, 4, 5} full — the whole array** |
| **9** | 36 | — | `error: 'aie.tile' op row index (6) must be less than the number of rows in the device (6)` |

**Eight herds of width 4 place; nine refuses, loudly.** So the inventory R2 would want by default —
nt1 as J7a's three-herd pipeline, the FFN's three herds, nt2 as another three — is **nine, and does
not exist**. This is a measurement, not an inference, and it is the single biggest constraint on
R2's shape.

### 3.6 The column budget, counted where it actually lives — MEASURED

Doc 23's rule is *two or fewer L3-facing MM2S per COLUMN across the whole segment*. R1's shipped
structural clause ([31 gate arm (c)](31-fused-resident-tail.md),
`probe_ffn_resident_interior.py` clause F) counts `shim → core` flows **only**. An L2-staged refill
is a `shim → memtile` flow and consumes a shim MM2S port exactly the same way.

`probe_r2_segment_budget.py --arm shim`, on R1's own `build_ffn_resident_module`:

| shim column | MM2S total | of which → core | S2MM |
|---|---|---|---|
| 0 | 1 | 1 | 1 |
| **1** | **2** | **0** | 0 |
| 2 | 1 | 1 | 1 |
| 3 | 1 | 1 | 1 |
| 4 | 1 | 1 | 1 |
| 5 | 1 | 0 | 0 |
| | **7 of 16 ports**, worst column **2** | worst column **1** | 4 |

R1 is **not** over budget — but its own check reports a worst column of 1 where the true figure is
2. **The clause has a blind spot, and R2 is the increment that would walk into it**, because R2 adds
L2-staged streams (`w_up`, `w_down`, the gammas) and those are exactly the ones the clause cannot
see. Correcting arm (c) to count both flow kinds is a prerequisite of R2's gate, and is the reason
`probe_r2_segment_budget.py` exists as a separate arm rather than a note.

Memtile occupancy on the same routed design, for the same reason (a memtile has 6 MM2S / 6 S2MM):

| memtile column | MM2S out | S2MM in |
|---|---|---|
| 1 | 4 / 6 | 2 / 6 |
| 3 | 4 / 6 | **5 / 6** |

R1 already sits one port from full on one memtile — that is the GeLU→down fan-out plus the `w_down`
refill. R2 as designed **removes** that fan-out (§2, consequence 3), which is where R2's memtile
headroom comes from.

---

## 4. The R2 column-budget arithmetic, explicit

The feasibility question the mission names. Per-column L3-facing MM2S demand, for the design of §2.

**The five L3 operands** (`packed1` in, `gamma1`, `w_up`, `w_down`, `gamma2`; `output` is S2MM and
in a different budget). Two ways an operand reaches the array, and they cost differently:

- **herd-direct** (`dma_memcpy_nd` inside a herd on a launch argument) — one shim MM2S in
  **every column that herd occupies**. J7a's `packed1` and `gamma` fetches are this shape.
- **L2-staged** (a segment-scope refill into an L2 buffer, then channel puts) — the copy has no
  herd-side endpoint, so it is *"allocated globally across shim columns"* (`builders/ffn_accum.py`,
  and confirmed by §3.6: R1's three staged refills landed on shim columns 1, 1 and 5 while its
  herd-direct C fetches took columns 0, 2, 3, 4). One shim MM2S **total**, wherever the allocator
  puts it.

| operand | shape | cost |
|---|---|---|
| `packed1` | herd-direct into nt1's add stage | 1 MM2S × 4 columns |
| `w_up` | L2-staged | 1 MM2S, global |
| `w_down` | L2-staged | 1 MM2S, global |
| `gamma1` | **L2-staged** (see below) | 1 MM2S, global |
| `gamma2` | **L2-staged** | 1 MM2S, global |
| `output` | herd-direct out of nt2's scale stage | 1 **S2MM** × 4 columns |
| — | R1's down-herd C fetch/store | **deleted** (§2, consequence 2) |

**Total: 4 herd-direct MM2S + 4 global MM2S = 8 of 16 shim MM2S ports; worst column 2** if the
allocator ever puts a global stream in a `packed1` column, 1 otherwise. Plus 4 S2MM.

**Why the gammas must be staged.** J7a fetches `gamma` herd-direct in `stage_scale`, and with
`packed1` also herd-direct that is exactly two per column — J7a's whole budget, with nothing left.
R2 stacks a *second* scale stage (nt2's) into the segment; if placement lands it in the same
columns, that column takes three and re-enters the packet path. Staging both gammas through L2
removes the dependence on where `air-place-herds` puts things, which is the property the rule is
really about. Cost: 2 more L2 buffers of 1,536 B and 2 memtile MM2S groups.

**The fallback if it is still tight**: stage `packed1` through L2 as well, refilled per trip from an
L3-side offset and put whole (R1's own idiom). Then every column's herd-direct MM2S is **zero**, the
total is 5 global MM2S, and the budget stops being a risk at all — at the price of one memtile hop
on the tail's input.

> **STATUS: DESIGN INTENT, UNVERIFIED.** The arithmetic above is not measured. What *is* measured is
> (a) that the staged/herd-direct distinction behaves as stated on R1's routed design (§3.6), and
> (b) that the design arm's miniature — one herd-direct L3 stream in, one out, one staged weight
> feed — routes with 0 packet channels (§3.1). Verifying the full arithmetic needs the R2 module to
> exist; `probe_r2_segment_budget.py --arm shim` is written to be pointed at it.

---

## 5. The herd inventory

Bounded by §3.5's measurement: **at most 8 herds at `herd_x` = 4**, and 8 uses every tile on the
device with zero slack for the placer.

| inventory | herds | tiles | notes |
|---|---|---|---|
| nt1 J7a (3) + up + gelu + down + nt2 J7a (3) | **9** | 36 | **MEASURED REFUSED** |
| nt1 J7a (3) + up+gelu folded (1) + down + nt2 J7a (3) | 8 | 32 | fits at 100 % occupancy, no slack. The fold is **free at the object level** — `ffn_accum_mm.o` already exports `ffn_gelu_bf16` (§7.2) — so the only cost is the placer's zero headroom |
| nt1 J7a (3) + up + gelu + down + **nt2 fused (1)** | **7** | 28 | keeps J7a's pipeline on the first tail |
| **nt1 fused (1) + up + gelu + down + nt2 fused (1)** | **5** | **20** | recommended primary |

"fused" means the one-herd norm tail built on the `addnorm` pre-add kernel rather than J7a's
three-stage split. It is not a compromise on either axis that matters here:

- **Numerically it is better.** `builders/norm_tail.py`'s own FOOTGUNS record that the three-stage
  pipeline carries **one extra bf16 rounding** against the fused kernel, because the normalized
  tensor is materialized between `stage_norm` and `stage_scale` where the fused kernel folds the
  gamma multiply into the same pass. R2's payoff clause is `mean_rel_L1`; spending a rounding to
  keep a stage count is the wrong trade.
- **The residency claim is untouched.** Doc 03's space-multiplexed discriminator is *≥ stage-count
  core→core flows*, and the stages R2 is about are the tail's operator boundaries — nt1 → up → gelu
  → down → nt2 — not the internals of a normalization. At 5 herds the design still carries
  `4 × (herds − 1)` = 16 core→core edges plus nt1's second put to nt2's residual path.
- **The kernels are current.** Both `addnorm` fused variants moved to two-pass f32 on 2026-08-11
  ([23 §2](23-rules-and-open-items.md), item 7), so the variance cliff that would have disqualified
  them is gone and pinned by `64x768_pre_add_offset`.

> **STATUS: DESIGN INTENT, UNVERIFIED** that a fused norm-tail herd composes into this segment. The
> herd-count budget is measured; that *this particular* herd body fits beside the others is not.

---

## 6. What extends `builders/ffn_resident.py`, and what composes

### 6.1 A new builder, not an edit to `ffn_resident.py`

R1's `builders/ffn_resident.py` partitions all three herds by **output column** (`group_n =
emb/herd_x`, C group resident across the `k'` loop, GeLU fan-out through a memtile). R2 partitions by
**row**. That is not a parameter, it is a different module: the herd bodies, both feeds and both
packing helpers change.

`builders/ffn_resident.py` also has a **parked gate** that item 6b will unpark
(`run_npu2_ffn_resident_peano.lit`, `check-ffn-resident`, the SPECS row). Churning it now would put
a pending gate on shifting ground. **Recommendation: a new `builders/ffn_resident_rows.py`**, with
R1's file untouched until its own gate has run once.

### 6.2 What composes from the norm-tail builders, and what does not

`build_norm_tail_module` **cannot** be composed: it builds its own `launch` and `segment`, and
[31 §What exists to compose](31-fused-resident-tail.md) already measured that segment-level
composition gives one `aie.device` per launch — the opposite of residency.

What composes is the **stage bodies**. `builders/norm_tail.py` holds them as inline closures today;
R2 needs them lifted into helpers that take an insertion context and the channel names, so that
`build_norm_tail_module` and the R2 builder emit the same code:

- `stage_add`'s vector `addf` loop — in R2 it gets **simpler**: its two operands arrive as two
  separate L1 tiles from two channels, so the packed `[rows, 2, cols]` layout, its plane
  arithmetic, and the plane-major/row-interleaved question all disappear for the *internal*
  boundary. (`packed1`, the front→tail boundary, keeps its layout — it is out of scope,
  [31 §Non-goals](31-fused-resident-tail.md).)
- `stage_norm`'s `layer_norm_rows` call — unchanged.
- `stage_scale`'s vector `mulf` loop — unchanged except that its output goes to a channel rather
  than to L3, and in nt1's case to **two** channels (the up herd, and nt2's residual path).

These compose unchanged and must not be re-derived:

- `norm_tail_device_inputs` / `norm_tail_reference` (→ `addnorm_pre_add_reference`),
- `ffn_resident_reference` (→ `ffn_reference`),
- `_stage_l1_bytes`'s measured ping-pong factor, and `NORM_TAIL_VEC_LEN`, `EPS`.

### 6.3 What is genuinely new

1. **The in-core retile.** Blocked `[rows_per_core, tile_k]` built from `rows_per_call`-row tiles by
   `memref.subview` + `vector.transfer_read`/`transfer_write` — the exact idiom `norm_tail.py`
   already ships for its plane arithmetic, at `MICRO` granularity. Measured to compile and route
   (§3.1). Its unrolling structure is load-bearing: the microtile row `mi`, the microtile column
   `ki` and the sub-block `sb` are **Python literals**, which is what lets the *source tile buffer*
   be chosen at codegen time; only the row within a tile and the `k'` step are real loop induction
   variables.
2. **A precondition: `MICRO % rows_per_call == 0`.** A producer tile must not straddle a microtile
   row boundary, or the source buffer stops being a codegen-time choice. At `MICRO` 8 and
   `rows_per_call` 4 it holds. The builder should refuse otherwise, with the reason.
3. **A precondition: `(band / herd_x) % 16 == 0`.** `DIM_M` is `rows_per_core`, and the kernel
   static_asserts `DIM_M % 16` under `build_ffn`. **MEASURED** (§7.1): `DIM_M` 16 builds, `DIM_M` 8
   is refused at `encoder.cc:136`. So at `herd_x` 8 with a 64-row band the kernel refuses — another
   reason all herds sit at 4.
4. **`compile_ffn_accum_kernel` must take `tile_m`.** It hardcodes `tile_m=TILE_M` (64) today; R2
   needs `DIM_M = rows_per_core`. One parameter, defaulted from the module's shape the way
   `tile_n` already is — the same discipline that exists because two no-argument calls disagreeing
   links a microkernel whose blocked ABI does not match the transfers, with corrupt output and no
   link error.
5. **A flattened weight-refill loop** (§3.4), as a precondition rather than a style choice.
6. **One L1 buffer per producer tile** (§3.2), as a precondition rather than a style choice.
7. **`w_up`'s packing loses its column dimension.** Row-partitioned, B is broadcast to every core
   rather than sliced per column, so `ffn_resident_pack_w_up`'s `(sweep, k'-step, column)`-major
   layout becomes `(group, k'-step)`-major. `ffn_accum_pack_w` for `w_down` likewise.

---

## 7. The kernel-object `-D`-symbol constraints — MEASURED, and simpler than expected

R1 found that `-D`-baked symbols cannot coexist twice in one module: a second tile shape would be a
second object exporting the *same* `ffn_matmul_bf16_bf16_up_proj`, and two private `FuncOp`s cannot
share a symbol. R2 changes `DIM_M` and adds a norm-tail object, so both halves needed re-checking.
`probe_r2_segment_budget.py --arm objects` compiles them and reads the symbol tables rather than
trusting the source comments.

### 7.1 The row-partitioned microkernel builds, and its refusal is what sets the herd width

| `DIM_M` | result |
|---|---|
| **16** (`band`/`herd_x` = the design) | **BUILT** |
| 64 (R1's) | BUILT |
| **8** (`MICRO`) | **REFUSED** — `encoder.cc:136` static assertion |

So `DIM_M % 16` is not merely a comment in `external_kernels.py`; it is enforced, and it is what
makes `herd_x` = 8 at a 64-row band impossible (`rows_per_core` would be 8). **Every R2 herd sits at
width 4 because the microkernel says so**, and `MAX_PLACEABLE_HERD_X` = 4 agrees for an unrelated
reason. Both roads lead to 4, which is a comfortable place for a design to be.

### 7.2 Global symbols, counted — one collision, not three

Global (externally-linkable) defined symbols only. Counting *all* defined symbols makes every pair
look like it collides, because objects from one compiler share local assembler labels (`.LBB*`,
`.L_LEnd*`); `llvm-nm --extern-only` is the discriminating flag.

| pair | shared global symbols |
|---|---|
| `ffn_accum_mm.o` (M=16) ^ `encoder_ffn.o` | **8 — all of them** |
| `ffn_accum_mm.o` ^ `layer_norm.o` | 0 |
| `ffn_accum_mm.o` ^ `addnorm_ffn.o` (`build_ffn=False`, `pre_add=True`) | **0** |
| `encoder_ffn.o` ^ `layer_norm.o` | 0 |
| `encoder_ffn.o` ^ `addnorm_ffn.o` | **0** |
| `layer_norm.o` ^ `addnorm_ffn.o` | 0 |

And where each symbol R2 calls actually lives:

| symbol | defined in |
|---|---|
| `ffn_matmul_bf16_bf16_up_proj` | `ffn_accum_mm.o`, `encoder_ffn.o` |
| `ffn_zero_bf16_up_proj` | `ffn_accum_mm.o`, `encoder_ffn.o` |
| **`ffn_gelu_bf16`** | **`ffn_accum_mm.o`**, `encoder_ffn.o` |
| `layer_norm_rows` | `layer_norm.o` |

Two things follow, and both make R2 easier than §5 assumed:

1. **R2 does not need `encoder_ffn.o` at all.** `ffn_accum_mm.o` already exports `ffn_gelu_bf16` —
   the two are the same source (`encoder.cc` under `-DBUILD_FFN`) at different `-DDIM_*`, and GeLU is
   elementwise with its length as a runtime `i32`, so the tile defines do not reach it. Link the
   GeLU herd against `ffn_accum_mm.o` and the module drops from three objects to two, the 8-symbol
   collision stops existing, and **§5's 8-herd "fold GeLU into the up herd" option becomes free** —
   the fold was costed as needing a new GeLU entry point, and it does not.
   *This applies to R1 as well*: `builders/ffn_resident.py`'s FOOTGUN reasons about coexistence
   ("they may coexist here ONLY because no core links both") for a second object it need not link.
   **Not changed here** — R1's gate is parked and about to run; noted for whoever unparks it.
2. **`compile_addnorm_ffn`'s FOOTGUN is wider than the truth.** It says *"encoder.o and
   addnorm_ffn.o both define `ffn_gelu_bf16` and `ffn_eltwise_add_bf16_vector`, so they cannot be
   linked into one ELF as-is"*. That is a statement about the **default `build_ffn=True`**. Built
   with `build_ffn=False` — which is all a norm-tail herd needs — `addnorm_ffn.o` shares **zero**
   global symbols with either FFN object. The fused-norm-tail inventory of §5 therefore carries no
   symbol hazard whatsoever.

### 7.3 What still holds

- **One `FuncOp` per symbol, shared by both GEMM herds.** The up stage's tile shape **is** the down
  stage's, in all three dimensions — which is what forces `DIM_M = rows_per_core` on both and is
  fine only because the row partition makes them genuinely the same shape. (In R1 the same
  constraint forced `group_n = emb/herd_x`; in R2 it forces the M dimension instead.)
- **`layer_norm_rows` takes its shape at runtime**, so nt1 and nt2 can run at different
  `rows_per_call` off one declaration and one object.

> **STATUS.** The object builds and the symbol tables are **MEASURED**. What remains **UNVERIFIED**
> is the link itself: every structural probe here stops before it (deliberately — that is what makes
> them hermetic and seconds-fast), so "no core links two objects that collide" is checked by
> construction and by symbol table, not by a completed aiecc link. That arrives with R2's first
> `check-` recipe.

---

## 8. Foreseeable walls, and which are measurable hermetically today

| # | wall | status |
|---|---|---|
| 1 | Nine herds do not place | **MEASURED** — refuses at 9, loudly (§3.5) |
| 2 | `air-shrink-memref-sizes-by-access` silently shrinks a multi-get L1 band | **MEASURED**, with the dodge measured (§3.2) |
| 3 | `air-split-l2-memref` aborts on a two-symbol L3-side offset | **MEASURED**, deterministic on the round-tripped dump, with an exact dodge (§3.4) |
| 4 | An L2-staged band feed exhausts the memtile's 48 BDs | **MEASURED** (§3.3) |
| 4b | The microkernel refusing the row partition's `DIM_M` | **MEASURED** — 16 builds, 8 refused (§7.1) |
| 4c | Objects colliding on global symbols | **MEASURED** — one pair collides and R2 need not link it (§7.2); the **link** remains unverified |
| 5 | Per-column shim MM2S over 2 | **MEASURABLE** — `probe_r2_segment_budget.py --arm shim`, once the R2 module exists |
| 6 | Memtile port pressure (6/6) | **MEASURABLE**, same arm |
| 7 | `air-fuse-channels` channel census and compile time | **MEASURABLE** — R1 sits at 12 symbols / 1 s against the >1200 s wall at 90; R2 adds the tails' channels |
| 8 | Down-herd L1 fit (4 C accumulators + A + B ≈ 51,200 B of 65,536) | **MEASURABLE** once the down herd is row-partitioned; **not yet measured** |
| 9 | **Shim BD exhaustion — wall 4 / item 6b** | **NOT hermetically measurable.** It surfaces in `npu.air.mlir` *after* the MLIR dumps, during aiecc's lowering, so a dump-reading probe cannot see it. R2 changes the count in both directions — it **deletes** the per-K-step C fetch/store pair, and its weight feeds run more iterations. **Re-derive after 6b lands; do not predict.** |
| 10 | In-core retile cost | **Device question.** ~64 `vector<8xbf16>` moves per A operand. Against the same operand's `2 × 4 × 24` = 192 `aie::mmul` intrinsics it is roughly a 25 % instruction overhead — an estimate, not a measurement, and the ladder is what prices it |
| 11 | `mean_rel_L1` improving on `fused`'s 1.784e-2 at margin ≥ 1.27× | **Device question**, R2's payoff clause |

---

## 9. The verdict, and the bounded fallback

**R2 fits, and the order seam resolves — by re-mapping the FFN interior to a row partition, not by
re-mapping the norm tail.** The hand-off is L1→L1 per column with no BD offset anywhere; the design
arm routes with 0 packet-typed channels, 4 core→core flows and 44,032 B of the 65,536-byte core.
Three plausible alternatives were measured and all three fail: the literal-offset L1 band is a
silent miscompile, the L2-staged band exhausts the memtile's BDs, and the natural weight-refill loop
nest aborts the compiler.

Two things bound the design rather than block it: **eight herds at width 4** (so J7a's three-herd
pipeline cannot be used on both tails), and **the per-column shim budget**, which the arithmetic
clears at 8 of 16 ports and which R1's own gate clause currently cannot see.

**The bounded fallback, if R2-rows proves unbuildable**: keep R1's column-partitioned interior
verbatim and attach **nt2 only** — `hidden` continues to bounce through L3 as R1 expects, and the
join at the FFN's output goes on chip. That deletes rows 14, 22 and 23 = **6.0 MiB of the 9.0
@1024** for no new mechanism at all, leaving rows 13 and 15 (3.0 MiB) on the table. It is worth
naming as the floor so that "R2 is hard" never turns into "R2 is nothing".

## 10. Reproduce

```bash
export PYTHONPATH=/home/cj/mlir-air/build-xrt/python
export PATH=/home/cj/mlir-air/build-xrt/bin:\
/home/cj/mlir-air/sandbox/lib/python3.12/site-packages/mlir_aie/bin:$PATH

python3 agents/probes/probe_r2_order_seam.py           # design arm + 2 controls, ~15 s
python3 agents/probes/probe_r2_order_seam.py --arm row_tiles --wloop nested   # control 3
python3 agents/probes/probe_r2_segment_budget.py       # herd sweep + shim census

# the objects arm also needs Peano (kernel compiles only, still no device):
PEANO_INSTALL_DIR=/home/cj/mlir-air/sandbox/lib/python3.12/site-packages/llvm-aie \
  python3 agents/probes/probe_r2_segment_budget.py --arm objects
```

The **build** tree, not `install-xrt`: [15 §Which toolchain tree](15-environment-notes.md) — the
install predates the item-6a fuse-pass fix. Both probes print the `air-opt` mtime; record it beside
any number taken from them.
