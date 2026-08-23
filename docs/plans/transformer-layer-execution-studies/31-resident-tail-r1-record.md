# 31 — The resident FFN tail (R1): the consolidated record

Consolidated 2026-08-22 from docs 31, 31a, 31b, 37, 47, 49, 52, 53 and the prediction files
PREDICTION-28A-FIX, PREDICTION-28A-O3O6, PREDICTION-28B-LADDER, PREDICTION-MAXQ; their full text is
at git tag `pre-cleanup-20260821`. It is the one record of the `fused` resident-tail family: the
settled standing of increment R1 (§1), the operator's 2026-08-21 supertile reframe (§2), the byte
floor (§3), R1's design and the seven walls it hit (§4), the R2 order-seam design (§5), the
deterministic defect (§6, formerly doc 49), wall 7 and row 28 (§7, formerly doc 52), the balance
instrument (§8, formerly doc 47), workload-dependent mapping and the selector (§9, formerly doc 53),
and the tools that survive (§10). Narrative of how each conclusion was reached is dropped; every
measured number, devq id, fix, gate and standing rule is kept.

**Section map for code citations.** `builders/ffn_resident.py`, `ffn_resident_structure.py`,
`builders/test_ffn_resident.py` and `run_npu2_ffn_resident_peano.lit` cite the old docs by number:
"doc 49" → §6; "doc 52" → §7, with "doc 52 §7" → §7.3, "§10.6" → §7.6, "§13.7" → §7.8; "doc 53
section 2.3a / 2.4" → §9.2; "doc 31a" → §3; "doc 31b 3.6" → §5.3.
Plain-text mentions of docs 26 / 28 / 33 refer to material now in [25](25-mode-rebuilds-and-results.md);
docs 19 / 24 / 48 to [16](16-compiler-changes.md); docs 38 / 45 / 46 to [44](44-mapping-frameworks-synthesis.md).

Two sentences that every source doc ends with, and that still hold: **`fused`'s SPECS atol stays
PROVISIONAL, and no resident-tail latency or byte figure has ever been measured on hardware.**

---

## 1. The settled standing

### 1.1 Where R1 works: `herd_x = 1`, `down_K ≤ 6` `[2026-08-19]`

`down_K = ffn_dim / tile_k = chunks_per_group × herd_x × sweeps` — the number of times the
`w_down` feed's L2 buffer is filled and drained (§7.5).

| box | measurement |
|---|---|
| `herd_x = 1`, `down_K ≤ 4` | **21/21 PASS**, devq 300 (the hardware ladder after the item-23 fix, §6); the rungs that already passed are byte-identical to the pre-fix run, exactly the four that failed now differ and pass |
| `down_K = 5` | **FAIL 5/5 → PASS 5/5** on O2/T5/K5 by 28(a)'s rotation fix, devq 398, 25/25 prediction clauses (§7.8) |
| `down_K = 6` | **TIMEOUT 5/5 → PASS 5/5** on O3 by 28(b)'s pacing, devq 403, 0/2048 mismatches (§7.8). devq 330's TIMEOUT was measured against a binary in which the pacing never fired |
| `down_K ≥ 7` | still wedges (O6, `32×224 tk32`: TIMEOUT 5/5, sentinel 1.0000, cores 0/1, devq 403), **mechanism unresolved** — §7.8 |
| `herd_x ≥ 2` | wall 7 (§7.1): shared staging races; the builder fix (`shared_h_staging=False`) is correct 35/35 but does not compile above `128×128` at `herd_x 4` (48-block memtile cap); the compiler fix is **proved impossible** (§7.4) |

The gate shape (`768×3072` at `tile_k 32`, `herd_x 4`) is `down_K = 96` — it needs both walls
cleared, and `run_npu2_ffn_resident_peano.lit` stays `UNSUPPORTED: true`.

### 1.2 None of R1's walls were R1's

Through 2026-08-12 the project charged **six** walls to the one-segment residency composition. All
resolved ones were general compiler defects that R1 was merely the first design to reach:

| # | wall | what it actually was | status |
|---|---|---|---|
| 3 | `air-fuse-channels` SEGV (item 6a) | use-after-free on a ≥3-clique of sibling same-bounds nests, plus a hidden N-way miscompile (`ub = 2` hardcoded) and a third defect (dynamic offsets compared equal, sources' slices dropped) | **FIXED** 2026-08-11 (§4.4) |
| 4 | shim BD exhaustion (item 6b) | `hidden` refill: 96 tasks + `w_up`'s 1 = 97 live BDs on tile (1,0) vs 16; BD release emitted where the terminal `wait_all` was | **FIXED** 2026-08-11, `ea3b98ce` (§4.5) |
| 5 | channel-major shim issue order (item 6c) | two deadlocks; D1 already closed by 6b's sink, D2 (`w_down` c-major) closed by builder route E1 | **CLOSED** 2026-08-12 (§4.6) |
| 6 | memtile lock count (item 18) | `getLockValuePair` sized the semaphore from static reader *ops* (16) while emitting 4 BDs | **FIXED** 2026-08-12 (§4.7) |
| — | deterministic wrong answer (item 23) | `air-to-aie` lock-*placement*: acquire at each put, the core's own writes unguarded | **FIXED and gated** 2026-08-12 (§6) |
| 7 | `herd_x ≥ 2` race/hang (item 21) | a memtile staging buffer with `herd_x` writers on one counting semaphore, no participant identity; **timeout and wrong answer are one defect** (5/35 → 35/35 by a single-variable A/B). At `herd_x = 1` both hazards are vacuous — that, not the composition, is why the `herd_x=1` ladder is clean | **LOCATED**; compiler fix **proved impossible** (item 29); builder fix does not reach the gate (§7.1–7.4) |
| — | `down_K ≥ 5` (item 28) | **two** defects wearing one number (`maxq ≡ down_K` identically in this builder): (a) L2 slot-rotation **phase skew** (`== 5` → byte-deterministic permutation) **FIXED** 2026-08-19; (b) `down_K` outstanding shim task starts with no await (`≥ 6` → hang) **FIXED** 2026-08-19 (`c634f735`); `≥ 7` still wedges | (a)+(b) done; remainder unowned (§7.5–7.8) |

Item 30 (the `w_down` feed's readers unbound at `herd_x ≥ 2`) is latent and not shown reachable
(§7.9).

### 1.3 What this means

Neither remaining wall is a residency question. The mechanisms are proven; what remained was
compiler work (rows 21/28/29) with **no known route to the gate at `herd_x ≥ 2`** — wall 7's
compiler fix cannot exist and its builder fix does not compile at the gate shape — which is why
continuing R1 was an operator decision and not a default. §2 is that decision. Separately, the
band-serial weight term (§3.6) makes R1's tail alone move 1.77× the packaged layer's DRAM traffic
at 1024, so "is resident composition worth having" was already a live question
([44](44-mapping-frameworks-synthesis.md)'s FLAT and TileFlow entries raise it).

---

## 2. The operator's reframe `[2026-08-21, afternoon]`: supertiles

R1 is **reframed, not closed**. Model the array's workload as **supertiles** of per-core tiles:

- a supertile is the work the array processes in one execution; within it the per-core tiles are
  independent (mlir-air composes them as regions);
- one supertile then the next are **separate executions in the runtime sequence**;
- the gate shape becomes a sequence of supertiles, each inside the working box
  (`herd_x = 1`, `down_K ≤ 6`) — **wall 7's shared-L2 multi-writer buffer is designed out rather
  than fixed**;
- **each supertile should emit a FINISHED output block** (so `down_K = 96` per execution, which
  meets the unresolved `down_K ≥ 7` wedge of §7.8 head-on); accumulating down partials *across*
  executions is acceptable **only if it measures faster** than the finished-block form;
- **R1's first increment is therefore that two-form comparison on hardware.** Rows 6 / 28 / 30
  follow this.

Standing elsewhere from the same block: J1 closed (superseded by J7a); Goal 1 parked; the
big-three model leg will not be run (8/11 is the standing leg); docs 01–12 demoted to
[01-original-plan-superseded.md](01-original-plan-superseded.md).

### 2.1 `[2026-08-22]` The first increment, attempted: the box does not exist at the layer's width

Queue item 9's first step was to build the two forms at the gate width. Neither can be built, for
reasons that precede the `down_K ≥ 7` wedge (evidence `results/r1-supertile-20260822/`, devq 511;
builder checks host-only):

- **`down_K` at real width.** `down_K = chunks_per_group × herd_x × sweeps` with
  `chunks_per_group = (emb_dim / herd_x) / tile_k`, `tile_k ≤ 32` (`MAX_L1_TILE_K`, measured), and
  the builder requiring whole sweeps (`ffn_dim % (herd_x × group_n) == 0`, `group_n = emb_dim /
  herd_x`). The ≤ 6 box was measured on `emb × ffn` **toy widths** (`O3 = 32×192`: one chunk per
  sweep, six sweeps; re-run 3/3 PASS, devq 511, corr 0.99987, 0/2048). At `emb 768, herd_x 1` a
  single sweep is already **24** refills; the finished-block form is 96. Both forms sit above the
  wedge before anything else is considered.
- **L1 at real width, `herd_x 1`.** The builder refuses before `aircc`: *"the up core needs 205,824 B
  of L1 (pp A + pp B + resident C), over the 65,536-byte tile"* at `tile_k 32`; 152,576 at 16;
  125,952 at 8. The resident H group for one 64-row band is `64 × 768` bf16 = 96 KB on its own.
  **The `herd_x = 1` box cannot hold the layer's hidden group at any `tile_k`.** `herd_x 2` fits only
  at `tile_k 8`; `herd_x 4` (the gate shape) builds at 32/16/8 — and every `herd_x ≥ 2` form is the
  multi-column H staging that wall 7 (shared, §7.1) or the 48-block memtile cap (per-column, §1.1)
  blocks at this width.
- **What a real-width supertile would have to be.** To put one execution inside a `down_K ≤ 6`,
  `herd_x 1` box, the up herd's group width must be decoupled from `emb_dim` (today `group_n =
  emb_dim / herd_x`): an execution covering 64 rows × a 192-column hidden slice (6 chunks; resident
  C 24 KB + pp A 8 KB + pp B 24 KB ≈ 56 KB, fits) with its down partial accumulated across
  executions — 16 executions per band, 12 bands per 768-token layer = **192 executions**, each a
  launch boundary at the measured 106–108 µs ([57 §1.5](57-inference-path-optimizations-from-hexagon.md))
  ≈ **21 ms per layer in boundaries alone**, against the H round-trip the resident tail exists to
  remove (`64 × 3072` bf16 ≈ 393 KB per band, written and read: ~26 µs per band at 30 GB/s) and
  against `coarse`'s whole-layer **13.4 ms** at 512 under the new clock ([54 §1a](54-first-full-profile-and-decoder-families.md)).
  A finished-block execution (form A) at real width needs `herd_x 4` plus wall 7 cleared plus the
  wedge cleared — two walls whose routes are closed (§7.3: the lock fix proved impossible; the
  per-column builder fix does not compile at the gate shape) and one unresolved.

The two-form comparison therefore has no hardware instance to run. What R1 at real width would
cost in boundaries is larger by ~60× than what it saves in DRAM traffic, under today's launch
cost; the increment was returned to the operator with that arithmetic.

### 2.2 `[2026-08-22]` The operator's route: partial-sum staging, ported from iron's `addnorm_ffn`

The operator's answer: **use the partial-sum staging**, and port iron's fused
AN1 → FFN → AN2 pipeline as the reference — `dev-addnorm-ffn` (`5cebcd7`, 2026-02-08),
`operators/addnorm_ffn/design_old.py` (1,427 lines; its later siblings `ffn-an-new-1`
`operators/ffn_addnorm/design.py` and `dev-mha-an-combine-bufs` `operators/encoder_pipeline/design.py`
drop or extend the first AN stage). "Not the most optimized, but it fused those operations." It ran at
the layer's width in iron: test rows `(M 512, K 768, N 3072)` on 8 columns, 16-row tiles, `k 96/128`,
`n 128/96`, `down_proj_depth 6–8`, `nA 2–4` replicas × `nB 2–6` column slices.

**Its dataflow, and why it has none of R1's walls.** Per replica: two AN1 cores produce LN rows and
broadcast them to `nB` up-projection cores (no memtile); each up core computes an `(m, n)` tile of H
for its column slice and streams it **L1→L1** to its paired down core — H is never staged through a
shared buffer (wall 7 cannot arise) and never resident as a whole in any core (wall 1 cannot arise);
each down core applies GeLU, then accumulates its **private** `(m × K_out)` partial of C_Down as a
**ring through the memory tile** — `curr_acc_c` (L2→L1) / `new_acc_c` (L1→L2), depth
`down_proj_depth`, one `(m, k)` block in L1 at a time, `matmul_with_acc` per block — while `w_down`
streams per K chunk through the memtile as in J7b; after all H column tiles are consumed the `nB`
partials are reduced by an **L1→L1 chain** (`buffer_to_reduce` + `add`) into the AN2 cores. The
accumulator lives in L2 (`m × 768` bf16 per core), so `m` is small (16) and L1 holds a block.

**The port, in increments** (all gated; hermetic first, device second, shape last):

1. Builder `builders/tail_pipeline.py` (name provisional): one `air.segment`, four herds per replica
   (`an1 [2,1]`, `up [nB,1]`, `down [nB,1]`, `an2 [2,1]`), L3→L2→L1 feeds for rows (`x`, `residual`),
   `w_up` column slices and streamed `w_down` row chunks; up→down `L1→L1` channels; a private L2
   accumulator memref per down core with get/put per block; the reduction chain; AN2. Kernels: the
   lineage's own, already ported in `kernels/encoder.cc` (`fused_add_layer_norm_1outs`,
   `ffn_matmul_with_acc_bf16_bf16_down_proj`, `ffn_zero_bf16_down_proj`, `ffn_gelu_bf16`,
   `ffn_eltwise_add_bf16_vector`). Structure script + hermetic lit (compile through `aircc` with
   `debug_ir`, count flows/channels/BDs, no device), as `ffn_resident_structure.py` does.
2. Hardware at iron's baseline `(64, 48, 96)` on 2 columns, numerically exact against the
   AN→FFN→AN reference; then the 16-row-tile shapes. The re-execution gate shape runs before
   anything is cited.
3. The layer's width `(512, 768, 3072)` with iron's parameter rows; each AIR wall met is named
   and either derived as a skip or fixed at the builder (the BD stride cap, the 48-block memtile
   cap, the per-column shim budget, placement).
4. A lit in the suite; latency under the forward-only clock against `coarse`'s and `fused`'s
   AN1+FFN+AN2 stages at 512, same session; DRAM traffic from the dispatch vector. That number is
   the first resident-tail measurement the study has ever had.

What it does NOT promise: that AIR's lowering of a long `w_down` stream plus an L2 ring is free of
the `≥ 7` wedge — J7b's 96-step `w_down` stream passes on hardware, which is the evidence it is
R1's composition and not the stream; increment 2 is where that is learned.

---

---

## 3. The byte floor (formerly doc 31a) — host-only arithmetic, no device dispatched

Shapes `baseline_768` (emb 768, ffn 3072, 12 heads × 64), bf16, seq 512 and 1024. Reconciles
**exactly** against [25](25-mode-rebuilds-and-results.md)'s (doc 26 §C) 84.0 MiB crossing table and
[27](27-common-ladder-result.md)'s measured ladder bytes.

### 3.1 Two lenses

- **`bytes_transferred`** (the dispatch vector) counts host↔device `bo.sync()` traffic: it cannot
  see device-side DRAM traffic, excludes static-weight uploads in steady state, and **includes**
  verification readbacks of every intermediate boundary.
- **DRAM crossings** (doc 26 §C's lens) count each tensor once per device-side read and write. This
  is what residency is about; nothing measures it on hardware; it is a logical-traffic **floor**.

**The instrument warning, load-bearing everywhere below:** `bytes_transferred` will **not** show a
resident win — every removed crossing is device-side — and it will *drop anyway* because 9 of
`fused`'s 13 steady-state syncs are verification readbacks of boundaries a resident tail no
longer exposes. A gate must pin that fall as an ABI consequence, never present it as the residency
result.

### 3.2 Tensor sizes and the resident floor

| tensor | bytes @512 | @1024 |
|---|---|---|
| activation `[S,768]` bf16 | 786,432 | 1,572,864 |
| wide `[S,3072]` bf16 | 3,145,728 | 6,291,456 |
| packed `[2,S,768]` bf16 | 1,572,864 | 3,145,728 |
| `qkv_f32` `[S,2304]` f32 | 4,718,592 | 9,437,184 |
| `w_qkv` / `w_o` / `w_up` / `w_down` / `gamma1`,`gamma2` | 3,538,944 / 1,179,648 / 4,718,592 / 4,718,592 / 1,536 each | same |
| **weights total** | **14,158,848** (13.503 MiB) | same |

Resident floor = input + output + weights once: **15,731,712 (15.003 MiB) @512**,
**17,304,576 (16.503 MiB) @1024** — doc 26 §C's "irreducible 16.5 MiB", re-derived to the byte.

### 3.3 The measured packaged numbers reconstruct exactly

Doc 27 records `fused` at **21,233,664 B @512** and **42,467,328 @1024** over 13 syncs: 3 uploads
(`x` + `packed1` + `qkv_f32` = 7,077,888 / 14,155,776) + 10 readbacks (`q`,`k`,`v`, `attn_context`,
`packed1`, `hidden`, `ffn_up`, `ffn_gelu`, `packed2`, `output` = 14,155,776 / 28,311,552). Adding
the 6 static uploads (14,158,848) gives doc 26 §6's cold vector: **19 syncs, 56,626,176 B @1024** —
also exact. On this counter a resident layer's steady state is `x` + `output` = 1,572,864 @512 /
3,145,728 @1024, a **13.5×** reduction — mostly verification traffic disappearing.

### 3.4 DRAM crossings per boundary, packaged `fused`

| # | crossing | region | @512 | @1024 | resident? |
|---|---|---|---|---|---|
| 1 | `x` read (qkv A) | front | 786,432 | 1,572,864 | remains |
| 2 | `w_qkv` read | front | 3,538,944 | 3,538,944 | remains |
| 3 | `qkv_f32` write | front | 4,718,592 | 9,437,184 | removed |
| 4 | `qkv_f32` read (3 cast launches) | front | 4,718,592 | 9,437,184 | removed |
| 5 | `q`,`k`,`v` write | front | 2,359,296 | 4,718,592 | removed |
| 6 | `q`,`k`,`v` read | front | 2,359,296 | 4,718,592 | removed |
| 7 | `attn_context` write | front | 786,432 | 1,572,864 | removed |
| 8 | `attn_context` read | front | 786,432 | 1,572,864 | removed |
| 9 | `w_o` read | front | 1,179,648 | 1,179,648 | remains |
| 10 | `attn_out` write (`packed1` plane 0) | front | 786,432 | 1,572,864 | removed |
| 11 | `packed1` read | tail | 1,572,864 | 3,145,728 | removed (whole-layer); **remains** (tail-only) |
| 12 | `gamma1` read | tail | 1,536 | 1,536 | remains |
| 13 | `hidden` write | tail | 786,432 | 1,572,864 | removed |
| 14 | `hidden` mirror write (`packed2` plane 1) | tail | 786,432 | 1,572,864 | removed |
| 15 | `hidden` read (FFN A) | tail | 786,432 | 1,572,864 | removed |
| 16 | `w_up` read | tail | 4,718,592 | 4,718,592 | remains |
| 17 | `ffn_up` write | tail | 3,145,728 | 6,291,456 | removed |
| 18 | `ffn_up` read | tail | 3,145,728 | 6,291,456 | removed |
| 19 | `ffn_gelu` write | tail | 3,145,728 | 6,291,456 | removed |
| 20 | `ffn_gelu` read | tail | 3,145,728 | 6,291,456 | removed |
| 21 | `w_down` read | tail | 4,718,592 | 4,718,592 | remains |
| 22 | `ffn_out` write (`packed2` plane 0) | tail | 786,432 | 1,572,864 | removed |
| 23 | `packed2` read | tail | 1,572,864 | 3,145,728 | removed |
| 24 | `gamma2` read | tail | 1,536 | 1,536 | remains |
| 25 | `output` write | tail | 786,432 | 1,572,864 | remains |

Totals: front (1–10) 22,020,096 / 39,321,600; tail (11–25) 29,101,056 / 48,761,856;
**packaged 51,121,152 (48.753 MiB) @512 / 88,083,456 (84.003 MiB) @1024**; removed by whole-layer
residency 35,389,440 / 70,778,880 (doc 26 §C's "intermediate 67.5"). Doc 26 §C's own item list sums
to 64.5 MiB against its stated 67.5 — it omits the `ffn_out` write+read pair (3.0 MiB); this table
is the complete itemization.

### 3.5 The per-scope split and the headline

| scope | rows | @512 | @1024 | share of removable @1024 |
|---|---|---|---|---|
| tail-internal (R1: 17–20 = 24.0 MiB @1024; R2: 13–15, 22, 23 = 9.0 MiB) | | 17,301,504 | 34,603,008 | **48.9 %** |
| front-internal + `packed1` read | 3–8, 10, 11 | 18,087,936 | 36,175,872 | 51.1 % |

| | @512 | @1024 |
|---|---|---|
| packaged `bytes_transferred` (measured) | 21,233,664 | 42,467,328 |
| resident steady state on that counter | 1,572,864 | 3,145,728 |
| packaged DRAM crossings (floor) | 51,121,152 | 88,083,456 |
| **tail-resident crossings** (front + tail floor rows 11,12,16,21,24,25) | **33,819,648 (−33.8 %)** | **53,480,448 (−39.3 %)** |
| whole-layer resident floor | 15,731,712 (−69.2 %) | 17,304,576 (−80.4 %) |

Precision is the measured part of the payoff: every deleted crossing is a bf16 round-trip; `fused`
sits at `mean_rel_L1` 1.784e-2 with 1.27× headroom under the 1e-1 ceiling
([23 §3](23-rules-and-open-items.md)). That a resident tail improves it is a prediction, not a result.

### 3.6 Capacity, and the band-serial weight term (from doc 53 §6)

NPU2 on-chip storage is 32 × 64 KiB L1 + 8 × 512 KiB L2 = **6 MiB, not flat**. The `[S,3072]`
intermediate is 3.0 MiB @512, **6.0 MiB @1024** (the whole chip), 12.0 @2048, 24.0 @4096 — so
whole-tensor residency is out of reach at 1024+ (doc 03's bound, confirmed). Streaming residency
routes around it: per-stage L1 is seq-independent (norm-tail `stage_add` 37,888 B at
`rows_per_call` 4, 8 overflows once ping-ponged; ring cores 57,344 B at `tile_k 32`, just over at
64).

**But R1 is band-serial** (`seq_len == TILE_M = 64`, one dispatch per band advancing on launch
arguments), and each band dispatch fetches the whole `w_up` and `w_down` — 9,437,184 B, read off
devq 338/340's runtime sequence (`memref<2359296xbf16> offset 0 len 2359296` each). Against 31a's
lens, which counts weights once per layer:

| | @512 (8 bands) | @1024 (16 bands) |
|---|---|---|
| weights band-serial | **75,497,472** | **150,994,944** |
| tail total, floor → band-serial | 11,799,552 → 77,859,840 (6.6×) | 14,158,848 → 155,716,608 (11.0×) |
| R1's tail vs the packaged **layer** | **1.52×** | **1.77×** |
| residency removes / band-serial adds / **net** | 17,301,504 / 66,060,288 / **−48,758,784** | 34,603,008 / 141,557,760 / **−106,954,752** |

Crossover `33,792·S = 9,437,184·(S/64 − 1)` → **S = 83 rows, 1.30 bands**. This falsifies
**band-serial** residency, not resident composition; the term is orthogonal to walls 7 and 28; the
lever is weight retention across bands. `study/analytical_cost.py` reproduces every figure in both
tables to the byte (§9.4). Arithmetic on a logical-traffic floor — not a hardware byte figure.

---

## 4. R1 as built, and the seven walls (formerly doc 31 and doc 37)

### 4.1 What "resident" means, and what was measured before building

[03](03-measurement-model.md) splits **packaged** (`stitch_elf`: one configuration per segment,
hand-offs through DRAM — `fused`'s tail today) from **resident** (operators on the array
simultaneously, hand-offs L1→L1 — what `fused` is by definition and had never been). The
discriminator is countable: packaged = zero core→core flows between operators, resident = ≥
stage-count. Scope is the tail only (`fused` and `coarse` share their front, doc 28).

`probe_fused_resident_tail.py` (retired) stitched nt1 + `ffn_accum` + nt2 at 1024×768 and 4096×768
(~4 s, 59 dumps each): **each launch becomes its own `aie.device`** (three configurations — residency
holds only within a segment, so the resident tail is by construction a one-segment design no
builder emitted); pieces survive intact (32 core→core flows, 0 packet channels, ≤ 2 shim-inbound
per column); 15–16 channel symbols vs the >1200 s wall at 90 (doc 26 lane C); two composition
walls — `build_ffn_accum_module` builds only a 64-row band (`herd_x·herd_y ≤ 6` memtile feed
ports, `MAX_PLACEABLE_HERD_X = 4`), and at 4096 the mirror compose is ill-typed (plane-major vs
row-interleaved above the shim BD stride cap). A tail-only scope avoids the `[1,1]`/`[2,2]`
FlashAttention conflict entirely (no FA in the tail's aircc invocation) and stays an order of
magnitude under the fuse-channels wall.

The three seams: **(1) layout** (row-major norm output → 8×8 microtiles) — closed for an L3
producer by the shim's 4-D read pattern at zero cost; the on-chip producer case belongs to R2;
**(2) order** (column-striped norm herds vs band-serial ring) — doc 31 predicted "re-map the norm
tail's rows to band order"; 31b corrected it (§5.2); **(3) dataflow** (up → GeLU → down) — the up
projection iterating output columns outermost produces H's column blocks in exactly the down K
order. Column-budget arithmetic predicted the standalone widths cannot coexist (nt1 alone fills 2
per column at `herd_x 8`), which is why the phase was band-serial first. Increments: **R1** the
interior (deletes 24.0 MiB @1024), **R2** attach the norm tails (9.0 MiB), **R3** mode wiring with
both ladder lengths re-walked twice.

### 4.2 R1 status `[2026-08-11]`: built, structurally green

`builders/ffn_resident.py`, structure promoted to `ffn_resident_structure.py`. One 64×3072×768 band:
**one tile-bearing `aie.device`**, 12 channel symbols, `air-fuse-channels` 12 → 12, compile 1.0 s.
Seam 3 resolved with one correction: the GeLU→down hand-off **fans through a memtile by port
arithmetic** (every down core consumes every chunk; a down core's two S2MM ports are spoken for;
a channel has one physical source), so the core→core constant is `herd_x` (up→GeLU), not
`3×herd_x`. The kernel object is a composition constraint: `-D`-baked symbols cannot coexist
twice, so the up stage's group width IS the down stage's `tile_n` (`emb/herd_x`), both GEMM herds
link one 64×32×192 object, and `ffn_dim % (herd_x · group_n) == 0` is a precondition. Nothing
ping-pongs (up core C+A+B single = 40 KiB of 64). A compile-time wall routed around: 24 unrolled
`w_down` refill copies left `air-isolate-async-dma-loop-nests` non-terminating (>25 min); the
feed shape that compiles is real loops everywhere except the H5-literal sub-channel index.
Numeric arm registered at 64×3072×768 with `ffn` scaling; atol provisional.

### 4.3 The emulation arm `[2026-08-12]` (item 17) — it now interprets the module

`builders/test_ffn_resident.py` originally imported `ffn_resident_pack_w_up` and **never built the
module** (re-deriving every pattern by hand), so its 5.5e-12 figure certified a transcription of
the design; re-imposing E1's deleted c-major defect still printed 5/5. Rebuilt: it interprets the
built `air.ir.Module` (every `dma_memcpy_nd`/`channel.put`/`get` at its real offsets, `scf.for`
at real bounds, herds as concurrent actors, channels as FIFOs, f64) under two named models — (M1)
`air-dma-to-channel`'s per-textual-instance hoist; (M2) memtile lock pairing, k-th round reads
k-th landed value. Measured: shipped builder **5.457e-12** over 64×768, 8/8; the real pre-E1 builder
(`918c202f`) **4.716e+03**, 7/8 red. NC1 (the `c` loop re-unrolled) reproduces 4.715995e+03 to the
digit; NC2 (swapped `hidden` retile strides) 5.23e+03; a stale anchor reports `STALE`. Liveness
pinned: 768 in-place accumulates (4·4·24 up + 4·96 down), 20 zeros, 96 GeLU chunks, zero undrained
streams. Tamper-verified three ways. One arm, not two: the lit recipe
`run_ffn_resident_emulation_tests.lit` and the script are one command (8 → 10 clauses after wall 7,
§7.2). **It does not model** timing, BD folding, fuse-channels, wall 5's starvation or lock
counts — and (M2) was later **falsified** by wall 7 (values do not land in shim issue order at
`herd_x > 1`); clauses 9/10 now refuse a shared-staging module at `herd_x > 1`.

### 4.4 Wall 3 — `air-fuse-channels` crash (item 6a), FIXED 2026-08-11

The same `install-xrt` (2026-08-07) binary compiled R1's 284-line module clean twice then crashed
twice (SEGV in `air::isAsyncOp` ← `AIRFuseChannels::runOnFunction`); deterministic on the
round-tripped `pass_017` dump 10/10. Minimal shape (`probe_fuse_channels_sibling_nests.py`,
retired): N sibling same-bounds `scf.for` nests each with one textual refill — **N=2 fuses 5/5,
N=3 crashes 5/5**. R1 presents `herd_x = 4` such nests (H5 forces the `c` unroll). Three defects in
`mlir/lib/Transform/AIRDependencyScheduleOpt.cpp`: (1) no merge *roles* — on a 3-clique B's ops
entered the destination set after the erased set, and `wrapRegionsWithForLoops` clones-and-erases;
(2) the NFL wrap hardcoded `ub = 2` — a destination absorbing k sources needs `1 + k` slots; (3)
(from the same-day Codex review) dynamic offsets compared equal, so sibling nests reading different
L3 slices fused into one stream repeating the destination's slice — the first fix had preserved
this, and the revision keeps all sides with differing patterns on the merged channel
(`fuse_channels.mlir` func9 shape; func13's expectation had encoded the miscompile and was
corrected). Verified: N=3/N=4 dumps and R1's `pass_017` **10/10 clean** (old 10/10 SEGV);
`check-air-mlir` 491/0; regression lit `fuse_channels_sibling_nests.mlir` verified failing. **The
old pass's lucky green outputs were themselves wrong** (`@channel_4` left alive with its own 2-trip
wrap), so any structural literal from a pre-fix ≥3-clique dump must be re-derived; the structural
probe re-derived and passed 3/3. Compiler details: [16](16-compiler-changes.md).

### 4.5 Wall 4 — shim BD exhaustion (item 6b), FIXED 2026-08-11 (`ea3b98ce`)

The gate ran the day the fuse fix landed (through `build-xrt/python`): STRUCT arm passes; the
numeric arm refuses at `npu.air.mlir:1178`, `'aiex.dma_configure_task' op Too many simultaneously
active buffer descriptors on tile (1,0), which supports up to 16` — identical under both fusion
forms (the J1 wall of [23 §4](23-rules-and-open-items.md), measured on R1). Re-derived from the
runtime sequence (devq 231): the offending feed is **`hidden`** (`@air_channel_2`), **96 tasks** =
sweeps 4 × k_steps 24 (the deliberate sweep re-read, not the down feed's 24), **97 vs 16** live BDs
with `w_up`'s 1; the mechanism is that AIR emits a transfer's BD release where its token was joined
— R1 joins every token at one segment terminator. Loop-shaped BD programs are **arithmetically
unavailable** for this feed: the retile `sizes [8,4,8,8] strides [6144,8,768,1]` uses all four
hardware dimensions and no adjacent pair merges (6144 ≠ 32, 8 ≠ 6144, 768 ≠ 8); the 24-chunk loop
would be a fifth. The fix (`mlir/lib/Conversion/AIRRtToNpuPass.cpp`, `boundShimBdLiveness` /
`paceShimFeedForBdReuse`): set `issue_token`, `dma_await_task(t[i-depth])` **before task i's
configure** (an await one op later refused at task 16 — devq 233), depth = the tile's free budget
(15 = 16 − `w_up`'s held BD), every token consumed exactly once (tail drained), the paced run sunk
before the first pre-existing blocking op. No-op unless a tile is over budget. Verified:
`shim_bd_liveness_bound.mlir` fails pre-fix; `check-air-mlir` **492/0**; transformer-layer suite
**31/1/0** (devq 248; devq 241 same result one commit earlier); structural probe PASS, 59 dumps in
1.0 s, devq 239. The pacing was verified at pass and compile altitude **but not on hardware** —
R1 was the only module triggering it and it hung on wall 5.

### 4.6 Wall 5 — shim issue order (item 6c), CLOSED 2026-08-12 (formerly doc 37)

With BDs bounded the numeric arm **hangs**: `ERT_CMD_STATE_TIMEOUT`, `txn_op_idx 0xFFFFFFFF`
(devq 235). Doc 31 recorded the order as `[hidden ×96][w_up][w_down]` and that
`air.preserve_shim_dma_order` does not fix it (devq 236: `[ch2 ×96][ch3 ×96][ch4 ×96]`, still
times out). Doc 37's scoping (`b777517b`, no device, SOURCE + INFERENCE marked) found:

- the fragmentation is `DmaToChannelPass`'s **driver loop** (`AIRDmaToChannel.cpp:1543-1554`),
  which marks one external channel op at a time; the pattern is already N-ary;
- **`air.preserve_shim_dma_order` is a folding switch as well as an ordering one**
  (`AIRDependencyScheduleOpt.cpp:8499-8534` skips the whole launch region), so devq 236's
  `[96][96][96]` says nothing about the unmarked build; the README row's "no ordering of whole
  channel runs can satisfy R1" inference is **not established** by that artifact;
- `w_up` folds to **1** BD (doc 31's own wall-4 table), so a streaming co-operand is not a run;
  the correct statement is **"the unfoldable feed must be issued last"**;
- a second, independent deadlock: `w_down`'s c-major delivery (4 textual instances → 4 sibling
  hoisted loops concatenated) starves the up herd at sweep 1 with the inter-channel order perfect;
- six routes A–F weighed; route A (change the hoist) rejected for blast radius (17 lit tests,
  passes keyed in writing to the hoist's output shape); recommendation: one hermetic compile
  (the census below), then **route E** (builder-side), route C (generalize `air-fuse-packet-put-loops`)
  as the durable follow-up.

**The census ran** (same day; `air-opt` 2026-08-11 13:28:03, `aircc` sha256 `5cb08407…`, unmarked
build, 1.3 s): `hidden` **96** tasks, `w_up` **1** (`offset 0 len 2359296`), `w_down` **13**;
emitted order **`[w_up][w_down][hidden ×96]`** — **6b's sink had already fired**, route D not
needed; all 96 `hidden` configures carry `{air.bd_recycled, issue_token, repeat_count 7}` with
awaits at uniform depth 15; `w_down` offsets `0, 147456, 737280, 1327104, 1916928, 294912, …`
(column 0 across all four sweeps first — defect 2 survives folding in a stronger form); (1,0) 97 by
configure→free but `hidden` emits zero `dma_free_task` (ids recycled). **E1 on a copy**: `w_down`
collapses to **1** contiguous BD monotone in `(s,c,jj)`, channel symbols **12 → 9**, compile 1.4 s
vs 1.3 s; E2 inert by measurement. **The recorded `[hidden ×96][w_up][w_down]` order describes a
superseded binary**: devq 235/236 ran 13:06:15 / 13:08:52, `AIRRtToNpuPass.cpp` was relinked at
13:28:03, and which of "the sink was not yet in 235's build" or "the order was carried from an
earlier dump" is correct is not established (both scratchpads gone). Do not cite that order
against the current compiler. Why the baseline folds `w_down` to 13 is unexplained.

**E1 landed**: the re-taken census against the integrated build (items 8 + 9) is byte-identical;
`w_down` 13 → 1; `check-air-mlir` 494/0; STRUCT PASS on both binaries. The numeric arm then timed
out (devq 259, `fatal_error_type 0x0`) — predicted by item 18's lock-conservation bound (wall 6).
Two latent traps carried: the preserve marker's folding behaviour is documented only in source;
doc 19's step 1 should say the fragmentation is a driver loop ([16](16-compiler-changes.md)).

### 4.7 Wall 6 — memtile lock count (item 18), FIXED 2026-08-12

`air::getLockValuePair` (`AIRToAIESchedulingUtils.cpp:510-556`, `ceil(read_counter/write_counter)`
at `:550`) sized a memtile semaphore from static reader **ops**: `l2_b_down` 1 writer / 16 reader
ops (4 sub-channels × the 4-way `c` unroll) → 16, **while the same pass emitted 4 BDs** for them —
so at most 25 of the 96 `w_down` refills could ever complete (16·W ≤ 16 + 384). Compiler, not
builder: three siblings with the same shape get 4, and the pre-E1 builder yields 16 too. Fix:
reader ops sharing channel symbol + constant indices + access region collapse to one fill, scoped
to `write_counter == 1` (multi-writer buffers time-multiplex; `l2_h` would go 4 → 1 and starve).
Regression lit `memtile_lock_count_per_fill.mlir` verified failing; `check-air-mlir` **495/0**;
`l2_b_down` 4/4 with every lock on both memtiles and all eight core tiles conserving. The gate was
re-armed and run: STRUCT PASS, numeric arm **still times out** (`ERT_CMD_STATE_TIMEOUT`, `ctx_pc
0x28B060AD`) — necessary, not sufficient; the negative-control arm inconclusive (times out before
comparing). Wall 7 is below (§7).

### 4.8 Wall 7 characterized (devq 273–276) — the lit header's record

`probe_r1_rung.py` pre-fills the output BO with a sentinel and **reads it back on timeout**
(nobody had looked at the device's output under the hang before). With the module-interpreting
arm element-exact at all 11 shapes: `herd_x 1, 64×32×32` **5/5 PASS** (0/2048, corr 0.99986 —
R1's dataflow runs on hardware); `herd_x 2, 64×64×64` PASS/TIMEOUT/PASS/TIMEOUT/PASS from one ELF;
`herd_x 2, 64×128×128` TIMEOUT then 3031, 4426, 4502, 4502 mismatches (corr 0.73/0.44/0.42/0.42);
`herd_x 4` 5/5 TIMEOUT at every shape tried (4–96 hidden tasks, tiling `[2,2]`/`[1,1]`/none, paced
and unpaced), 0 of 4 down cores writing. Excluded: task count, 6b's pacing (herd_x 4 hangs at 4
tasks, unpaced regime: 4 awaits / 10 frees vs the paced rungs' 20 / 6), shim order,
`runtime_loop_tiling_sizes` (byte-identical `aie.air.mlir`, `npu.air.mlir`, `.ctrltext`, `.pdi`),
lock counts. **`ctx_pc 0x28B060AD` names nothing** — the firmware's clean-timeout report site in 11
recorded hangs across four unrelated designs (mha GEMM devq 58/61/62, offload 103/104/109, R1
235/236/259/268) and reproduced by a two-herd control with one unmatched `ChannelGet` (devq 273);
the only other value, `0x28B0EC98`, is the `fatal_error_type 0x10` fault. Backend knobs (devq 276,
5×): `use_lock_race_condition_fix` takes `herd_x 2` 3/5 → 0/5; `_v2` replaces the deadlock with a
deterministic wrong answer (5/5 FAIL, 2688/4096, corr 0.517); `omit_pingpong="all"` inert.
Re-measured after item 23's fix (devq 306, 128×128): `herd_x 2` TIMEOUT, TIMEOUT, FAIL, FAIL, FAIL
(3 distinct outputs); `herd_x 4` TIMEOUT ×5 — **21 and 23 are independent**.

### 4.9 The gate and the design rules

`run_npu2_ffn_resident_peano.lit` (`UNSUPPORTED: true`; delete that line and nothing else to
re-arm) runs `make check-ffn-resident-structure` (STRUCT), `make check-ffn-resident` (numbers,
`opcheck.py` full-output `np.isclose` vs the FP32 reference) and `make check-ffn-resident-fault`
(the `hidden(0,0)` negative control). The STRUCT verdict pins `PASS (1 device, 4 core->core,
K-loop 4 -> 2, 0 packet-typed channels, shim MM2S 7/16 worst column 2)` plus the census negative
control (`3 L3 streams per column -> widened census 3, pre-widening shim->core count 0, packet
streams 12`) and `shim col 1: MM2S 2 (circuit 2 = 0 ->core + 2 ->memtile, packet 0)` (§5.3). The
four-arm gate as specified: structural (one `aie.device`; core→core = derived stage-edge count;
≤ 2 per-column shim MM2S **demand** over both flow kinds and packet flows; zero packet channels;
liveness at both ends), numeric (`mean_rel_L1` ≤ `fused`'s 1.784e-2 at margin ≥ 1.27×), fault
injection (`ln1_weight`, index 3; fault-run dispatch totals must equal the clean run's), vector
pinning (old totals recorded beside new). Run the lit, not the scripts standalone.

Design rules (all measured elsewhere): ≤ 2 L3-facing MM2S per column across the whole segment;
L3-side offsets only (never read a staged buffer at a per-iteration offset — H10 refuses it);
one role per L1 buffer; no hand placement / depths / ring; no widened tolerance; do not touch
`mlir/` in-phase (report the minimal shape); the band loop advances on launch arguments.
Non-goals: whole-layer residency, front changes, seq > 1024, new kernel objects.

---

## 5. R2 — attaching the norm tails, and the order seam (formerly doc 31b) `[2026-08-11]`

No device dispatched; every claim MEASURED (hermetic pass-dump), SOURCE, or DESIGN INTENT,
UNVERIFIED. R2 deletes rows 13, 14, 15, 22, 23 of §3.4: **4,718,592 B (4.5 MiB) @512,
9,437,184 (9.0 MiB) @1024**.

### 5.1 Status: both design constraints dissolved — re-derive, do not inherit

31b avoided literal-offset L1 bands because of item 9 and flattened its refill nest because of
item 8; both were fixed 2026-08-12 ([16](16-compiler-changes.md)), so **R2's design must be
re-derived rather than inherited**, and everything dump-derived below is PROVISIONAL (the tree was
rebuilt under the session four times: 11:05:06 → 13:03:54 → 13:06:01 → 13:28:03; every measurement
reproduced across them). Under the supertile reframe (§2) R2 is not scheduled.

### 5.2 The resolution: partition the GEMM herds by ROWS, not the norm tail by bands

A norm emits whole rows; a GEMM's A operand is a blocked column strip of all rows. Producing rows
and consuming column strips is a transpose of tiling that no row→tile re-mapping bridges — only
buffering the band and re-reading it at a per-`k'` offset, which is the frozen-BD rule. Doc 31's
seam-2 prediction is therefore retracted: partition up/GeLU/down by **M** so consumer core `c` owns
exactly producer core `c`'s rows (`rows_per_core = band / herd_x`), and the retile becomes
**compute local to one core** (subview + `vector.transfer_*`), not a BD. Consequences: `hidden`
never materializes anywhere; the down C round trip disappears (R1's 4 `shim→core` + 4 `core→shim`
flows gone); the GeLU→down memtile fan-out disappears (one core→core edge). **All herds must be
the same width** (hard precondition).

### 5.3 Measured (`probe_r2_order_seam.py`, `probe_r2_segment_budget.py`, both retired)

Shape: one 64-row band, emb 768, ffn 3072, `herd_x 4`, `tile_k 32`, `rows_per_core 16`, `group_n 192`.

- **Design arm routes** (`air-opt` 2026-08-11 13:06:01): 59 dumps in 0.5 s, one tile-bearing
  device, 0 packet channels, **4** core→core flows, 4 band tile buffers per consumer core, worst
  core L1 **44,032 B** of 65,536 (4 band tiles 24,576 + C 6,144 + A 1,024 + B 12,288, nothing
  ping-ponged), and **every `aie.dma_bd` offset is 0**.
- **CONTROL 1** — one L1 band filled by four gets at literal offsets is a **silent miscompile**:
  `pass_029 air-shrink-memref-sizes-by-access` shrank `memref<12288xbf16,2>` to `<3072>` leaving the
  gets at 3072/6144/9216 (reads up to element 12,256), no diagnostic. Root: `overall_access_bounds`
  from sizes alone. → item 9, fixed.
- **CONTROL 2** — doc 31's "stage bands in L2" refuses: `'aie.memtile_dma' op has more than 48
  blocks` (`getNumBDs(MemTile)` = 48, everything else 16); the Python-unroll dodge needs
  `k_steps · herd_x` = 96.
- **CONTROL 3** — the natural nested `w_up` refill (two-symbol map `s0·589824 + s1·24576`) aborts
  `air-split-l2-memref` (`tileChannelOpByFactor` builds `AffineMap::get(0, 1, add)`), SIGABRT 5/5
  on the round-tripped `pass_020` dump; flat loop routes (59 dumps, 0.4 s). → item 8, fixed.
- **Herd budget**: N herds of `[4,1]` — 2/4/6/**8 place (32 tiles, rows {2,3,4,5} full)**; **9
  refuses** (`'aie.tile' op row index (6) must be less than the number of rows in the device (6)`).
  So nt1 J7a (3) + FFN (3) + nt2 J7a (3) = 9 does not exist.
- **Column census on R1's own module** (this is `doc 31b 3.6`, cited by `ffn_resident_structure.py`
  and the lit): R1's arm (c) counted `shim→core` only and read **4/16, worst column 1**; the truth
  over both flow kinds is **7 of 16 ports, worst column 2** (column 1: 2 `shim→memtile`; column 5:
  1 `shim→memtile`; columns 0/2/3/4: 1 `shim→core` each; S2MM 4). Memtile occupancy: col 1 MM2S
  4/6, S2MM 2/6; col 3 MM2S 4/6, **S2MM 5/6**. **`[2026-08-12]` FIXED, item 10**: the literal is
  re-derived from `build-xrt`'s aircc of 2026-08-11 13:28:03 (sha256 `5cb08407`) and reproduces
  column for column. Widening was **necessary but not sufficient**: over budget AIR emits **packet**
  flows, so a port census reads **0** on an over-budget column — the control (one herd of 4, three
  herd-direct L3 operands: 0 `aie.flow` inbound, **12 `aie.packet_flow`**) runs inside the gate on
  every invocation; reverting to shim→core-only moves the literal to `4/16 worst column 1` (red),
  dropping the packet half is caught by the control alone.

### 5.4 Column budget, herd inventory, kernel objects

Per-column demand for the row design (DESIGN INTENT, UNVERIFIED): `packed1` herd-direct
(1 MM2S × 4 columns), `w_up`/`w_down`/`gamma1`/`gamma2` L2-staged (1 global each), `output` 1 S2MM
× 4 — **8 of 16 MM2S, worst column 2**; fallback: stage `packed1` too (5 global, 0 herd-direct).
Gammas must be staged because two herd-direct scale stages in one column would be three.
Inventory: 9 herds MEASURED REFUSED; 8 (GeLU folded into up) fits at 100 % with no slack; 7 (nt2
fused); **5 (nt1 fused + up + gelu + down + nt2 fused), 20 tiles — recommended**; the fused
`addnorm` norm tail carries one fewer bf16 rounding than the three-stage pipeline and both fused
variants moved to two-pass f32 on 2026-08-11. Kernel objects (`--arm objects`, `llvm-nm
--extern-only`): `DIM_M` 16 and 64 build, **8 refused** (`encoder.cc:136`) — so every herd sits at
width 4 for a kernel reason and for `MAX_PLACEABLE_HERD_X` independently; `ffn_accum_mm.o` (M=16)
shares **all 8** globals with `encoder_ffn.o` and **0** with `layer_norm.o` / `addnorm_ffn.o`
(`build_ffn=False`); `ffn_accum_mm.o` already exports `ffn_gelu_bf16`, so R2 (and R1) need not link
`encoder_ffn.o` at all. The link itself is unverified. New pieces: the in-core retile
(`MICRO % rows_per_call == 0`, `(band/herd_x) % 16 == 0` preconditions), `compile_ffn_accum_kernel`
taking `tile_m`, a flattened refill loop, one L1 buffer per producer tile, `w_up` packing losing its
column dimension; a new `builders/ffn_resident_rows.py` rather than an edit. Foreseeable walls:
shim BD exhaustion is **not hermetically measurable** (surfaces in `npu.air.mlir`); in-core retile
≈ 25 % instruction overhead (estimate); `mean_rel_L1` is the device payoff clause. **Bounded
fallback**: keep R1's interior and attach nt2 only — rows 14, 22, 23 = 6.0 MiB of the 9.0 @1024.

---

## 6. The deterministic wrong answer (formerly doc 49, item 23) — FIXED and gated `[2026-08-12]`

**Verdict.** Not an eighth wall: a compiler defect in `air-to-aie`'s core-side lock placement
(`AIRToAIEPass.cpp`, `allocateCoreLocksPerMemcpyOp`, the `sharedStagingBuffer` path of #1515). For a
core that writes an L1 buffer once and sends it in **more than one** `air.channel.put` on one DMA
channel, the acquire sat immediately before each put — pacing put *i+1* against put *i* while no
acquire dominated the core's own **writes** — so the next round's `ffn_zero_bf16_up_proj` memset
overwrote the buffer while the last BD was still streaming. Both racers are fixed-rate hardware
from a fixed offset, hence a **byte-identical** wrong answer every run. Lock **counts** are conserved
either way — a placement defect, invisible to item 18's audit.

**Measured** (devq 278, 6 rungs, 20 dispatches, `--dump-npz` sha256 of the raw BO; all at
`herd_x 1, tile_k 32`): `32/32`, `32/64`, `32/128` PASS (1 sha each); `64/64` **FAIL 2000/4096,
corr 0.729**; `64/128` **FAIL 1788/4096, 0.663**; `128/128` **FAIL 1932/8192, 0.869** (the item-23
rung, 5 runs, 1 sha, reproducing 0.868935042). Minimal failing configuration
**`emb 64 ffn 64` (sweeps 1, k' 2, cpg 2)**; the trigger is `chunks_per_group > 1`. (Doc 49's
"`sweeps` is excluded by measurement" — from `32/128` at 4 sweeps passing — was **retracted** by
§7.5: it had only ever been tested to 4.) Arrival map (`probe_r1_arrival_map.py`, retired:
recover H via `y_hw @ inv(w_down)` at square shapes, relL1 0.085 on the control; `{H_i @ Wd_j}`
dictionary coefficient +0.9999 diagonal, residual 0.0162; per-(chunk, row-run) fit): **the last
chunk of every group arrives as its first 8-row run (256 elements = `tile_k·MICRO`), the other
seven runs zero** — `128/128` chunk 3 `0.88 0.00 0.00 …`, `64/64` chunk 1 `0.74 0.00`. IR at
`pass_045_after_air-to-aie`: BDs correct (`offset 0 / 256, len 2048, sizes [8,256], strides
[512,1]`); at `cpg = 1` the acquire is hoisted to the block head and dominates the memset, at
`cpg > 1` nothing guards it.

**Rate model, tested out of sample**: `f0 = min(1, (base + C)/(chunk_run·(ρ − 1)))`, fit on
`f0 = 0.74` (base 256) and `0.88` (base 768) → **ρ ≈ 15.3**, **C ≈ 2451** elements (~77 core
cycles). Predicted in writing for `96/96` (cpg 3): `f0 = 0.810`; **devq 294 measured 0.81**, runs
1–7 at 0.00, chunks 0–1 at 1.00. Excluded by measurement: sweeps/refills/step count (`32/128`);
the builder/AIR module (`probe_r1_emulate_shape.py` EXACT: 3.41e-13 at 128×128, 1.99e-13 at
96×96); the BDs; chunk mispairing (off-diagonal ≤ 0.006); lock counts; `use_lock_race_condition_fix_v2`
byte-identical 3/3 (devq 294); `omit_pingpong=L1` byte-identical 3/3; `fix` v1 3/3 TIMEOUT;
`runtime_loop_tiling_sizes` (inherited from item 21).

**The fix**: place the acquire before the earliest op touching the buffer since the previous DMA
on it (block start when none) — for a pure relay (`put, put, …`) the placement is **unchanged**,
which is why #1515's `air_channel_to_locks_shared_buffer.mlir` is unaffected. Traced on R1's up
herd: `acq; zero; for k'{mm}; put0; rel; acq; put1; rel`. **Verified at every altitude**: regression
lit `mlir/test/Conversion/AIRToAIE/air_channel_to_locks_shared_buffer_producer.mlir` verified
failing pre-fix (`build-xrt/bin/air-opt` 2026-08-12 10:58); `check-air-mlir` **497 → 498 / 0**;
transformer-layer suite **32 / 1 unsupported / 0**; ten models **10/10** (devq 305); hardware
ladder **21/21** (devq 300), passing rungs byte-identical. **Two null measurements caught** (devq
298, 299): full green 21-leg logs against an `aircc` nine hours stale, because
`air.tools.resolve_tool` prefers a bundled binary over PATH — `build-xrt/python` **plus**
`build-xrt/bin` are both required, and take 3 refused to run on a sha mismatch (trap 5). An ELF
hash is not a valid discriminator (not byte-reproducible).

---

## 7. Wall 7 and row 28 (formerly doc 52, items 21 / 28 / 29 / 30)

### 7.1 Wall 7 located `[2026-08-12]` (item 21; devq 308 60 legs, 309 120 legs)

R1's down feed stages every GeLU column's H chunk through **one** L2 buffer. At `herd_x > 1`
`air-to-aie` emits one S2MM channel per GeLU core onto one single-slot memtile buffer, all on the
same counting semaphore with identical counts — no participant identity, so neither writer order
nor reader binding is fixed; the device delivers an arbitrary interleaving of the columns' chunk
streams (every chunk whole, matched to the wrong `w_down` K step) or deadlocks when a reader takes a
peer's token. The wrong answers are **permutations, not truncations**: item 23's arrival model does
not fit (residual 0.66–0.84), the pairing dictionary fits at **0.0152–0.0158**; 7 wrong answers on
3 rungs (D2 `128×128` `[0,2,3,1]`, `[0,2,1,3]`; C2 `64×128` `[0,1,3,2]`; G2 `32×64` `[0,1,3,2]`,
`[0,2,1,3]`) are all interleavings of per-column streams, both down cores agreeing — the disorder is
at the memtile. The IR: `%buf21 %mem_tile_3_1 L2 slots 1 writers 2 readers 2`, both writers
`acq %lock_3_1_23>=2 rel %lock_3_1_24 x2` — identical. `isChainLockCandidate`
(`AIRToAIESchedulingUtils.cpp:650`) **excludes MIMO** and falls through to the legacy template its
own header calls one "that allows concurrent stage firing and races on the memtile DMA" — which
is also why `use_lock_race_condition_fix_v2` A/B'd byte-identical five times: **never reached**.

### 7.2 The A/B (devq 309, `aircc` sha `0651a0e5…`, 5 fresh processes per arm per rung)

| rung | emb×ffn | hx | cpg | swp | shared P/F/T | percol P/F/T | cross-arm y identical |
|---|---|---|---|---|---|---|---|
| B1 | 64×64 | 1 | 2 | 1 | 5/0/0 | 5/0/0 | yes |
| C1 | 64×128 | 1 | 2 | 2 | 5/0/0 | 5/0/0 | yes |
| D1 | 128×128 | 1 | 4 | 1 | 5/0/0 | 5/0/0 | yes |
| A2 | 32×32 | 2 | 1 | 1 | 2/0/3 | **5/0/0** | no |
| B2 | 64×64 | 2 | 1 | 1 | 3/0/2 | **5/0/0** | no |
| C2 | 64×128 | 2 | 1 | 2 | 0/0/5 | **5/0/0** | no |
| D2 | 128×128 | 2 | 2 | 1 | 0/3/2 | **5/0/0** | no |
| G2 | 32×64 | 2 | 1 | 2 | 0/2/3 | **5/0/0** | no |
| H4 | 64×64 | 4 | 1 | 1 | 0/0/5 | **5/0/0** | no |
| D4 | 128×128 | 4 | 1 | 1 | 0/0/5 | **5/0/0** | no |
| F1 | 96×96 tk16 | 1 | 6 | 1 | 0/0/5 | 0/0/5 | yes — row 28 |
| F2 | 96×96 tk16 | 2 | 3 | 1 | 0/0/5 | 0/0/5 | yes — confounded |

**5/35 → 35/35**; timeout and wrong answer vanish together; `herd_x=1` controls byte-identical
across arms; `B2 == B1`, `C2 == C1`, `D2 == D4 == D1` to the bit. The shared arm reproduces devq 306
byte-for-byte at D2 (`7f7d7f62…` sentinel, `5698c105…`, `4099fa90…`). Minimal failing
configuration **`emb 32, ffn 32, herd_x 2, tile_k 16`** (3/5 in devq 308 and 309). Excluded: item
23's flavour (four `cpg = 1` rungs fail), `cpg`, `sweeps`, `k_steps_up`, down-K count, lock counts,
lock placement, `runtime_loop_tiling_sizes`/shim order/task count/6b's pacing, `ctx_pc`.

**Not fixed**: (a) per-column staging **does not compile at the gate shape** — `herd_x 4, tile_k 32`:
128×128 OK, 256×256 / 384×384 / 768×768 / 768×3072 `'aie.memtile_dma' op has more than 48 blocks`
— so **`shared_h_staging` stays `True`** (flipping it turns `ffn_resident_structure.py` red,
measured; both arms are explicit flags `--shared-h-staging` / `--per-column-h-staging` on
`probe_r1_rung.py`); (b) **F1 hangs at `herd_x = 1`** — a different defect (row 28). The builder's
comment that the shared `l2_h` buffer serialized the per-column loops is **retracted**: that AIR
dependency does not survive lowering. Cost of the fix where it compiles: `(herd_x − 1)` L2 chunk
buffers. Regression: `test_ffn_resident.py` clauses 9/10 (per-column build reads one L2 allocation
per GeLU column; the shared build is **rejected** by the same predicate; neither asserts the
default is the fixed form).

### 7.3 The compiler fix §7.1 pointed at (this is `doc 52 §7` and `§8`) — proved impossible (item 29)

Built anyway (worktree `build-mimo`, `aircc` `9f5a52af…`; devq 313 build, 319 airhost, 321/322
`check-air-mlir`; `build-mimo` exists because `build-xrt`'s `CMAKE_HOME_DIRECTORY` is the shared
checkout): `isChainLockCandidate` admitting MIMO, `getOrCreateChainLockSet` two chains (`nW + nR − 1`
signal locks), shipped as `mimo-chain-lock` (pass option, `aircc` flag, `XRTBackend(mimo_chain_lock=)`),
default false, a falsifier arm. `--check-order` (Petri net over the tile's locks and BDs, streams
always-ready; calibrated: v2 fan-in `memtile_chain_lock_v2_fanin.mlir` ORDERED, R1's `l2_h` RACE)
on R1 D2: default **RACE**; v2 **refused**; `+ mimo-chain-lock` writers **ORDERED** (`S2MM 1, 0,
1, 0, …`) but read binding **OVERWRITE**. **An AIE2 BD carries one acquire field and one release
field** (`generateDmaBd`), so a writer's release orders the writers *or* binds the readers, never
both; a counting argument (2 writers / 2 readers / 1 slot forces `a₀ > 2ρ` and `a₀ < 2ρ`) closes
asymmetric counts, for any run length. The escape is more BDs: reader chains must distinguish
`P = herd_x × chunks_per_group` phases (D2: 1 `h`-BD shared vs 4 per-column, `P` = 4); at the gate
`P` = 24 → 24 `h` + 24 `b` BDs per channel × 4 against the 48-block cap — **both fixes hit the
same wall, the compiler one no later**; and the two-chain writer order is round-robin where R1
needs run length `cpg` (coincides only at `cpg = 1`). "The compiler fix scales" is **retracted**.
Shipped: v2 **refuses a MIMO memtile buffer by name** instead of silently racing; regression
`memtile_chain_lock_v2_mimo.mlir` verified failing; `check-air-mlir` **498 → 499 / 0**; default
path byte-identical (`aie.air.mlir` sha `5439c51d…` from both compilers; buffers 22/6/1 shared,
23/7/1 chain arm, 25/9/4 per-column). The advance prediction ("percol and shared become
byte-identical") was never reached and is reported **unresolved**. Reaching the gate needs the
period reduced (stage a whole column group per transfer, `P` → `herd_x`) — a builder change, not
attempted. A lead not chased: `--check-order` reports **OVERWRITE** on `%buf20`/`%buf24` — the
`w_down` feed buffer, 1 writer / 2 readers / 1 slot, legacy template — in **both** arms (v2's fan-out
chain reads ORDERED on it); it became row 30 (§7.4).

### 7.4 Rows 28 and 30 are different objects (`install-xrt` `aircc` `b6e3de13…`; devq 327–331, 140 legs)

§7.3's lead (doc 52 §8.8) — the `w_down` buffer OVERWRITE "in both arms and at `herd_x = 1`" — is **false**: that
buffer's reader count is `herd_x`, so at `herd_x = 1` it is 1:1 and reads ORDERED across nine
`herd_x=1` modules (0 hazards). The v2 A/B on F1 (devq 327) is a **null by construction and by
measurement**: `aie.air.mlir` sha `a1b66f22c8579595` both arms, 0/0/5 both, one `y` sha `17c9c1bb`
equal to devq 309's F1 (10/10); the positive control — the same design at `herd_x 2` refusing with
the MIMO diagnostic in the same job — proves the binary carries row 29's change. **Row 30 stays
latent**: present in D2p/H4p/D4p which pass 5/5.

### 7.5 Row 28 is `down_K ≥ 5` — the headline named the wrong axis

Over **32** rungs (devq 327/328/329/330/331 + 309's arms, geometry re-derived from `argv`):

| `down_K` | verdict (5 fresh processes each) | rungs |
|---|---|---|
| 2, 3, 4 | **PASS** | T2 T3 T4 K3 K4 O1 O5 B1s C1s D1s A2p B2p C2p D2p G2p H4p D4p |
| **5** | **FAIL** — byte-deterministic, one sha, ~50 % of elements | T5 K5 O2 |
| 6, 7, 8, 9, 12 | **TIMEOUT** — `y` sentinel 1.0000, 0 cores finished | A T7 T8 K6 O3 O4 O6 N1 N2 N3 N4 N5 |

Excluded by measurement: `tile_k` (K6 = `192×192` at tk32 hangs), `cpg` (K3 cpg 3 passes, N4 cpg 3
hangs, O3 cpg 1 hangs), `k_steps_up` (K3 k_up 3 passes, N2 hangs; N3 at T8's k_up 8 hangs),
`herd_x` (1 and 2), the H-staging arm, host task count (H4p/D4p 14 tasks pass, N4 12 hang),
`runtime_loop_tiling_sizes` **directly** (devq 331: `--tiling none` leaves A/K6/O3/T8 TIMEOUT and
T4 PASS byte-identical `69ad2530`; `4,4`'s `npu.air.mlir` byte-identical to `2,2`'s, `4d3cce96…`).
**Out-of-sample 6/6** (devq 330, prediction sha `d5be991a…`): O5 `32×128` (sweeps 4) PASS; O2
`32×160` (5) FAIL one sha; O3 `32×192` (6) TIMEOUT; O6 `32×224` (7) TIMEOUT; O1 `16×64` (4) PASS;
O4 `96×96 hx2 percol` (6) TIMEOUT — only `sweeps` moves, which retracts doc 49 §2's exclusion. Two
earlier models falsified: `PREDICTION-SEP.md` (cpg as the axis; N1–N4 all timed out) and
`--check-order`'s DEADLOCK as a predictor (measured FAIL at 5). Every rung is a valid probe
(`probe_r1_emulate_shape.py` EXACT: `96×96 tk16` 2.27e-13, `192×192 tk32` 5.12e-13, and `96×96
tk32`, `64×64 tk16`, `128×128 tk32`, `80×80 tk16`). `down_K` is the number of fills of `l2_b_down`
(`%buf20`/`%buf24`, row 30's buffer) — same buffer, different hazard; fixing 30 would not fix 28.
Two `--check-order` defects fixed (both false hazards): a regex dropped
`aie.dma_start(MM2S, 0, ^bb1, ^bb6, repeat_count = 1)` (prologue + steady-state at odd cpg ≥ 5), so a
2-BD chain was simulated where 6 exist; and `aie.core` lock ops (4 of 4 / 6 of 6 on compute tiles)
were unmodelled. Both now read `UNMODELLED`; self-test 3 → 6 cases.

### 7.6 Row 28 is two defects (this is `doc 52 §10`; `maxq` is §10.6)

**`maxq ≡ down_K` identically** in this builder (`k_steps_up = emb//tile_k`, `cpg = (emb//herd_x)//tile_k`
⇒ `sweeps·k_steps_up ≡ cpg·herd_x·sweeps`), so no shape separates them.

**(a) The `down_K = 5` wrong answer is an L2 slot-rotation phase skew.** O2 decomposed: arrival model
0.7780 (does not fit), pairing dictionary **0.0165**; permutation `σ = [0,1,4,2,3]`, identical in r1
and r3, **not** an interleaving. `air-to-aie` multi-buffers the feed's L2 buffer; every slot is 1:1
and sound; what is wrong is the **phase** of the consumer's BD chain against the producer's.
Delivered order read off `aie.air.mlir` (`repeat_count = 0` means once, so `= 1` executes twice):

| rung | `down_K` | slots | consumer program | delivered |
|---|---|---|---|---|
| O5 | 4 | 2 | one circular chain | `[0,1,2,3]` IN-STEP |
| O2 | 5 | 3 | `repeat_count = 1` prologue (2 slots) + 1-slot tail | `[0,1,3,4,2]` SKEWED |
| O3 | 6 | 2 | one circular chain | IN-STEP |
| O6 | 7 | 3 | `repeat_count = 2` prologue + tail | `[0,1,3,4,6,7,2]` STARVED |

The same `d` holds on both memtiles and composes by geometry: sweeps-driven (O2) `σ = d[d[p]] =
[0,1,4,2,3]`; cpg-driven (T5 `80×80 tk16`, K5 `160×160 tk32`) the up-feed skew is absorbed by the
k-reduction, `σ = d = [0,1,3,4,2]` — **measured exactly on all three**, zero free parameters.
`PREDICTION-ROT.md` (sha `0b18a1c0…`) had said `[0,1,4,2,3]` for all three: clause 1 falsified for
T5/K5, clauses 2–4 held; the repair was made after the data, hence the causal test below. Source:
`AIRToAIESchedulingUtils.cpp`, `air::getRepeatCounts` → `detectNBufferRotation`'s
`numBuffers >= 2 && ops.size() % numBuffers == 0` — the guard correctly detects a non-clean
rotation; the fallback (per-op trip bucketing, separately terminated tasks replaying a prefix)
is not order-preserving. `--check-rotation` over all 21 compiled rungs: `down_K` 2/3/4 IN-STEP
PASS; 5 SKEWED FAIL; 6 IN-STEP TIMEOUT; 7 STARVED TIMEOUT; 8/9/12 IN-STEP TIMEOUT — the skew is not
"odd", and the ≥ 6 hangs are in-step. **Causal test** (`--omit-pingpong L2`, devq 332,
`PREDICTION-PP.md` sha `e7bc2dd4…`, compile gate passed: no multi-slot rotation, `maxq` unchanged
4/5/6/7): O5 PASS; **O2 FAIL → PASS 5/5** (0/2048, corr 0.999871, abs_err_max 4.9e-3 at atol 5e-2;
shas `O5pp 75205755`, `O2pp ece5f178`); O3, O6 TIMEOUT unchanged — **4/4**. The skew is the whole
cause of the wrong answer and none of the hang.

**(b) The `≥ 6` hang is shim task-queue occupancy — `maxq`.** O5 and O3 differ only in trip counts
(572 lines each, `%c4 → %c6`, `memref<4096xbf16> → <6144>`). The runtime sequence pushes `down_K`
`dma_start_task`s for the `hidden` refill on **one** channel (`%shim_noc_tile_0_0 / MM2S 0`) with no
await and **before** the `w_up`/`w_down` pushes the consumers need; `aiex.npu.push_queue` is a bare
register write with no occupancy accounting. `maxq` (outstanding starts on the busiest channel
before the first await) on all 21 rungs: PASS **2, 3, 4**; FAIL **5**; TIMEOUT **6, 7, 8, 9, 12** —
`maxq == down_K` on every one. Excluded: BD length (K4 passes at a 16384-element `w_up` BD, O3
hangs at 6144), total task count, tiling, the rotation. **No queue-depth constant is claimed** —
only that 5 outstanding starts complete and 6 do not, in this push order. `--check-rotation`
(`--stream-len`, `--refuse-skew`; self-test 6 → 11) is a third hazard class the other two audits
are blind to. R1's ceiling at that point: `ffn_dim ≤ 4·tile_k` (5 with `--omit-pingpong L2`); the
gate's `down_K = 96` is IN-STEP by the rotation reading and the binding constraint was `maxq`.

### 7.7 28(b): the fold is arithmetically unavailable; pacing shipped, first inert, then landed

The fix §7.6 specified (doc 52 §10.9) — fold the refill's puts into one task with `repeat_count` "as
`@air_channel_0`/`@air_channel_1` already are" — is **refused at its premise**: O3's six
`@air_channel_2` tasks already carry `repeat_count = 7` with the same descriptor as those channels
(`sizes [8,4,8,8] strides [256,8,32,1]`); `repeat_count` **is** the descriptor's iteration
dimension and advances the address (settled against O5's passing `y`: `len 256, repeat_count 7`
returns a correct 2048-element output), so six identical copies need a stride-0 **fifth** dimension
over four irreducible ones (256 ≠ 8·4, 8 ≠ 32·8, 32 ≠ 1·8) — wall 4 from the other side. Shipped
instead: `boundIdenticalShimPutRuns` in `AIRRtToNpuPass.cpp` after `boundShimBdLiveness`, pacing a
run of **≥ 3 structurally identical** (`OperationEquivalence::exactValueMatch`) fire-and-forget
MM2S pairs on one shim channel to **depth 2** via `paceShimFeedForBdReuse` unchanged — 6b never
fired here because 6 BDs on a 16-BD tile is under budget: **the hang is channel task-queue
occupancy, not descriptor-pool exhaustion, and nothing was counting it** (doc 53 §3.1 shows the
same thing at the gate shape: 96 starts, peak outstanding **15** = 6b's `(16 − 1)/1`, four rungs
above the last passing value — pacing the wrong budget). Regression lit
`mlir/test/Conversion/AIRRtToNpu/identical_shim_put_run_bound.mlir` (six identical `@refill`, a
distinct-offset `@varying` control, `@weights` after the run) verified failing pre-fix;
`check-air-mlir` **499 → 500 / 0**. Initially **unbuilt** (the permission classifier refused the
copy into the shared checkout); then built (devq 334) and **did not fire on R1**: the step walked
`func::FuncOp` only while R1 presents `aie.runtime_sequence` (`AIEX::RuntimeSequenceOp`) — every
other trigger condition verified against the artifact — and the lit was green because its input
is a hand-written `func.func` the pass itself converts (pre- vs post-conversion shape). The device
arms of devq 334 reproduced the recorded ladder exactly as a harness control (O5 PASS `75205755f4d0865e`,
O2 FAIL, O3/O6 TIMEOUT). **`[2026-08-19]` LANDED (`c634f735`)**: both arrival shapes walked; the
pre-fix binary is a no-op on a sequence-form probe (1 `issue_token`) where the fixed one paces (7
`issue_token`s, depth-2 awaits, tail drained, `@varying` untouched); new lit
`identical_shim_put_run_bound_seq.mlir` (post-conversion) verified red pre-fix; `check-air-mlir`
**500 → 501 / 0**; suite **34/1/0**; ten models **10/10** (devq 357 + 358; 357 died at 7/10 to a
session interruption, the continuation re-proved provenance). The sink and the pacing move
together, so a PASS does not say which did it — the synthetic N-sweep instrument §7.6 asked for (push N
tasks on one shim channel with a gated consumer, N = 3..10) remains the only thing that may quote
a queue depth, and is unbuilt.

### 7.8 28(a) fixed `[2026-08-19]` (this is `doc 52 §13`), and the `≥ 6` ladder re-measured (`§13.7`)

`air::getRepeatCounts` gains a `NonCleanRotationPlan`: after the clean path refuses, sites grouped
by **`(channel declaration, memref type)`** — R1's bundled `@ffn_res_up_feed` carries the H stream
(`64x32xbf16`, 3-slot rotation) and the W stream (`32x32xbf16`, single buffer) under one symbol,
and this key is what the §7.6(a) fix needed — whose trips form a `{q, q+1}` **prefix** staircase get
the only order-preserving program: the whole cycle × q (`repeat_count = q−1`) plus the first `r`
BDs once; `generateDmaBdProgram` emits those two tasks, otherwise the bucketing fallback is
byte-for-byte what it was. Compile side: lit `air_channel_nonclean_rotation.mlir` (10 firings over
a 6-site cycle, q=1, r=4) red pre-fix at `--implicit-check-not=repeat_count`; `check-air-mlir`
**501 → 502 / 0** (devq 395); O5/O1/O3/O6 **byte-identical** by revert-cycle; O2/T5/K5 read IN-STEP
`[0,1,2,3,4]` on all three L2 feeds. Codex review (two rounds) **narrowed** acceptance: distinct-
buffer full cycle (site count == buffer count); singleton groups need offset-equivalent BDs
(`chansMappedToEquivalentBDs`); constant-index agreement; shim DMA excluded at compile time; the
cyclic-run generalization **dropped** (prefix only — every accepted shape has `q == 1`); per-group
shared-loop evidence (the run sites' only loop ancestor is one shared steady loop with `r ≥ 2`,
tails loop-free). Three refusal lits (`…_refuse_repeated_buffer`, `…_refuse_mixed_offsets`,
`…_refuse_disjoint_loops.mlir`); output byte-compared on all seven rungs after every round;
`check-air-mlir` lands at **505 / 0** (delta the four 28(a) lits). Suite and models: devq 401,
402 (eleven models incl. `smollm2_1_7b_int4`); counts in the README leg table.

**O6 correction to §7.6's ceiling**: O6's `w_down` consumer trips are `{3,3,1}` — not a staircase — so the
plan refuses by design, O6 stays SKEWED `[0,1,3,4,6,7,2]` and byte-identical at air-to-aie; "5 and 7
become correct" is half-corrected: **5 yes, 7 no**.

**The `≥ 6` ladder** (devq 403, 10 legs, `PREDICTION-28B-LADDER.md` sha `cb2c7472…`): air-to-aie
identity does not survive 28(b) (downstream, in `airrt-to-npu`) — today's O3 elf `0ed23416…` vs
devq 330's `b2c0f26f…`, O6 `e9a88a68…` vs `f8cb1cb7…`, O3 carrying 7 `issue_token`s — so devq
330's TIMEOUTs measured a dead binary.

| rung | `down_K` | devq 330 | devq 403 | prediction |
|---|---|---|---|---|
| O3 `32×192 tk32` | 6 | TIMEOUT 5/5 | **PASS 5/5**, 0/2048, corr 0.99987 | HELD |
| O6 `32×224 tk32` | 7 | TIMEOUT 5/5 | **TIMEOUT 5/5**, sentinel 1.0000, cores 0/1 | FALSIFIED (said FAIL) |

**R1's ceiling is `down_K ≤ 6`** (4 pre-fix, 5 by 28(a), 6 by 28(b)). **28-remainder is `down_K ≥ 7`,
mechanism unresolved**: either a wedge pacing-to-depth-2 does not close, or the still-skewed O6
rotation deadlocking under flow control instead of surfacing as wrong bytes — not separated; the
synthetic queue-occupancy instrument is the next tool, with "unskew O6's rotation builder-side,
re-dispatch" as a second arm. O6 also closes the O2 attribution: pacing does not silently green a
skewed module, so O2's flip belongs to 28(a).

### 7.9 The prediction files, as predicted / measured pairs

| file (sha) | predicted | measured |
|---|---|---|
| `PREDICTION-MAXQ.md` (`90b92618…`, 2026-08-12) | clause 0 compile gate `maxq` 6 → 2, run sunk past `w_up`/`w_down`; O3/O6 TIMEOUT → PASS; O2 stays FAIL `[0,1,4,2,3]`; O5/O1 same `y` sha with a changed control program; a **non-monotonic** ladder PASS 2/3/4, FAIL 5, PASS 6+; `check-air-mlir` 499 → 500 with the delta exactly the new lit | clause 5 HELD (499 → 500); **clause 0 FAILED** (devq 334, the step did not fire — traversal, §7.7), so clauses 1 and 3 were never validly tested; clause 2 held (O2 FAIL 5/5 in devq 334). After the traversal fix (devq 403): O3 PASS — clause 1 half-held; O6 TIMEOUT — falsified (§7.8); O2 by then fixed by 28(a) |
| `PREDICTION-28A-FIX.md` (`1070338e…`, 2026-08-19) | O5/O1 PASS 5/5; O2/T5/K5 FAIL → **PASS 5/5**; O5/O1 air-to-aie byte-identical revert vs apply (amended: the first draft compared against devq 330's Aug-12 shas `aefed272…`/`ca8ba8dd…`, confounded by 28(b)) | **25/25**, devq 398, every clause held |
| `PREDICTION-28A-O3O6.md` (2026-08-19) | O3 byte-identical, TIMEOUT; O6 DIFFERS, rotation IN-STEP, TIMEOUT | resolved at the compile gate, no dispatch: O3 identical; **O6 also identical**, still SKEWED (`{3,3,1}` trips); its "devq 330's TIMEOUTs ARE the post-fix measurements" conclusion was **wrong one level down** (superseded by 28B-LADDER) |
| `PREDICTION-28B-LADDER.md` (`cb2c7472…`, 2026-08-19) | O3 PASS 5/5; O6 **FAIL 5/5** with deterministic wrong bytes, one sha | O3 **PASS 5/5** HELD; O6 **TIMEOUT 5/5** FALSIFIED (devq 403) |

---

## 8. The balance instrument (formerly doc 47, item 25) `[2026-08-12]`

`study/balance.py` + `study/balance_ert.py` (tests 43 + 29 = 72; `run_study_host_tests.lit`
357/357 in 19 modules → **428/428 in 21**, FileCheck-verified in four directions), host-only, no
device, **no new constants**. Parts: a `[step × port]` demand matrix per column off a routed
`aie.air.mlir` (`step` = ASAP async-dependence level, **not** a cycle); `back_solve` of the
bandwidth a stall-free run would require; overflow as a **slope** (`slowdown = min(1,
budget/demand)`, worst column, never a legality predicate — doc 44's correction); `bottleneck` =
`max` over isolated times with the argmax named, unpriced resources reported; an ERT
`(component, action, arguments) → Cost` with a four-valued source per number (`measured` /
`counted` / `modelled` / `absent`), exact lookup, `REQUIRED_ARGUMENTS` enforced; `stage_gap`
refusing iron's two defects (prefix comparison; elided `l3_bytes` — 9,437,184 B of B_Up/B_Down).

**Found**: (1) the shipped `addnorm` artifact — column 0 MM2S demand **3** against budget 2, mux
depth 3, slowdown 0.667, runtime **×1.500**, 26,112 B; columns 1–7 MM2S 2, 24,576 B each; S2MM 1,
12,288 B each; total **296,448 B** = x + residual + weight + output exactly. (2) **The blind spot**:
on that artifact `aie.flow` shim→core reads **0** (all 8 flows are output drains) while 17
`aie.packet_flow` carry the demand — a `aie.flow`-only count reads zero exactly when over budget
(`norm_tail_structure.py` check 4's definition, not live there only because check 1 rejects packet
channels first). (3) A defect in itself: the `air.launch (… %c8, %c1)` iteration space was not
multiplying traffic — byte totals 8× low on the `matmul_bf16` artifact; fixed, A **8,388,608 B**
once, C **2,097,152 B** once, B **16,777,216 B** = 2,097,152 refetched 8×. (4) **Back-solve** on
one routed candidate (devq **293**, build class, no dispatch):
`gemm.direct(M=64, K=512, N=512, tile_m=64, tile_k_l2=256, tile_k_l1=32, tile_n=64, herd_m=1,
herd_n=4)`, measured **122.81 µs**
(`sweep/results/baseline_512/64x512x512__direct__a66c58c881fe6e18.json`, role o_proj, 2026-08-07,
Turbo-conditional): column 0 MM2S 655,360 B (B 524,288 once + A 65,536 twice) → **5.336 GB/s**,
S2MM 65,536 B → 0.534 GB/s. Read as a **lower bound** on the stall-free requirement; under 8 % of
iron's cross-toolchain 67.9–70.9 GB/s, so at 33.5 MFLOP (0.27 TFLOP/s) this candidate is not
shim-bound — dispatch overhead dominating is the inference. (5) **ERT seeded**: baseline_512 544
files (400 added, 132 merged, 12 failed), baseline_768 436 (422/0/14), baseline_1024 528
(386/127/15), plus 2 + 3 counted descriptors — **1,213 entries: 1,208 measured `ns`, 5 counted
bytes, 0 modelled**; every seeded entry carries the pre-2026-08-10 Turbo-conditional string. (6)
**Repeat spread**: 259 merged rows are repeat measurements of identical priced actions —
`(max−min)/min` **median 1.6 %, worst 42.2 %** (`gemm.fused-cast(M=512, K=4096, N=1024, …,
herd_m=8, herd_n=4)`: 1,086,524 vs 1,545,360 ns), **53 of 259 (20 %) exceed 5 %** — so a 5 % band
is inside noise for a fifth of the actions; `Cost` carries `ns_samples`/`ns_min`/`ns_max`, `ns` is
the minimum (doc 23's compare-minimums rule); a search must rank on `ns_min` with `ns_max` visible
or adopt iron's 1 %-band-plus-tie-break — not argmin on a scalar.

**Modelled: none, deliberately** — there is no measured AIR-native shim bandwidth in this tree
([25](25-mode-rebuilds-and-results.md)'s doc 33 deferred the memcpy operator; iron's figure is
order-of-magnitude), so every `shim_dma` entry has `bytes` counted and `ns` absent. The per-column
budget 2 is imported from `aircc_artifacts.SHIM_DMA_CHANNELS_PER_DIRECTION`. 16 injected defects
all rejected (baseline 71/71; two initially UNDETECTED were the harness not applying — `mutate.py`
now reports `PATCH DID NOT APPLY` as a failure). Fixtures are the real `addnorm` artifact and the
same with the weight stream removed (doc 23's `elementwise_add` row). Open: the sweep of 1,208
compiles; shim ports unpriced (doc 33's 11(a)); `stage_gap` has no measured input; the ERT is JSON
not a schema table; `run_study_host_tests.lit` has no `make` target; the step axis is coarse. It is
**not a search** and not a fused-composition model. Session-scratch artifacts (`item25-private/`)
were not shipped.

---

## 9. Workload-dependent mapping (formerly doc 53) `[2026-08-13]`, and the selector (row 31)

Host-only arithmetic over `builders/ffn_resident.py`'s constants (`group_n = emb//herd_x`,
`sweeps = ffn//(herd_x·group_n)`, `cpg = group_n//tile_k`, `k_steps_up = emb//tile_k`, up-core L1
`2·(2·TILE_M·tile_k + 2·tile_k·group_n + TILE_M·group_n) + 1024`, `MAX_PLACEABLE_HERD_X = 4`,
`MAX_L1_TILE_K = 32`, 48-block memtile cap). R1's mapping is **derived from the shape**, not chosen.

### 9.1 The static predicate cannot see `maxq`

`study/mapping_space.py`'s `r1_interior_demand()` models streams, not tasks (`shim_global=3`), so the
current builder and a staged-`hidden` variant present a byte-identical `Demand`; neither BD blocks
nor queue occupancy is a field. Not a defect: the predicate answers *can this be routed*, and
`maxq` lives in the gap ([16](16-compiler-changes.md)'s doc 48 material).

### 9.2 Staging `hidden` (this is `doc 53 §2.3a / §2.4`, cited by `ffn_resident.py`'s `stage_hidden`)

Bytes fit at the gate shape (`l2_a_up` 4,096 → 98,304 B, `l2_b_up` 49,152 unchanged; memtile total 53,248 → **147,456 B**,
28.1 % of 524,288). The Python-unrolled literal-offset proposal was refused at `k_steps_up 24 ×
herd_x 4 = 96` blocks vs 48 — then **compiled** (`probe_ffn_resident_interior.py --keep-dumps`,
devq **338**, 59 dumps in 2.1 s) and corrected twice: R1 spreads across **two** memtiles
(`mem_tile_1_1` up feed: 4 MM2S ch × 4 BD + 2 S2MM × 2, 20 `aie.dma_bd`, **26** top-level blocks;
`mem_tile_3_1` down feed: 4 × 2 + 5 × 1, 13 BDs, **22** blocks), and the corrected count is 4 × (24 +
2) = **104 BDs, ~114 blocks** — refused by a wider margin. The finding that matters: the memtile
chain is already a maximally folded 2-BD ping-pong cycle; **the 96 is purely shim-side**.
`probe_r1_staged_hidden.py` (retired; devq **339/340**; `stage_hidden=False` added, default path
byte-identical, module sha `2582c733e19f26ba`): **Q1** — the whole band in one 4-D shim read —
**builds**; **Q2** — the per-k' drain out of the staged buffer — **REFUSED by message**
(`'air.channel.put' op channel @channel_23: BD offset is not a compile-time constant … Stage the
operand per iteration from L3 instead`), H10's diagnostic doing its job. So a staged `hidden` needs
its own A-only channel (A and B share `CHANNEL_UP_FEED` FIFO today), which wants **8 memtile MM2S
ports against 6** — and a port census reads 0 when it packet-multiplexes. The probe's first run
printed `maxq = 25` for the control; it is **15** (a regex bucketed every task into one anonymous
channel) — the instrument now raises on an unattributed task.

### 9.3 `maxq ≡ down_K` ⇔ `group_n == emb/herd_x`, and at `emb ≥ 1024` there is no legal mapping

The identity is forced by the shared `-D` kernel object; any change decoupling `group_n` decouples
`maxq` **upward** (∝ `1/group_n`). At the gate shape the runtime sequence (devq 340 control arm)
shows `air_channel_2` (`hidden`, `%arg0 memref<64x768xbf16>`, `sizes [8,4,8,8] strides
[6144,8,768,1]`, `repeat_count 7`) **96 starts, peak outstanding 15**; `w_up`/`w_down`/C/`y` 1 each
— the closed form predicts starts exactly; the outstanding count separates at the gate because
6b's `depth = (16 − 1)/1 = 15` paces the BD pool, not the queue (PASS band 2/3/4, FAIL 5, TIMEOUT 6+).
Predicted effect of 28(b) at the gate: peak outstanding 15 → 2 (recorded before the fix; not yet
measured at the gate shape). A same-day caution conflating starts with outstanding was **retracted**.

| emb | ffn | legal `herd_x` | `down_K` |
|---|---|---|---|
| 512 | 2048 | [4] | 64 |
| 768 | 3072 | [4] | 96 |
| 1024 | 4096 | **NONE** | 128 |
| 1536 | 8960 | NONE | 240 |
| 2048 | 8192 | NONE | 256 |

Escapes at emb 1024: `herd_x 8, tile_k 32` (41,984 B fits; crosses `MAX_PLACEABLE_HERD_X`); `herd_x
4, tile_k 16` (54,272; `down_K` 256); `tile_k 8` (44,032; `down_K` 512). This is doc 44's finding 3
and doc 48's result at one axis.

**The two-workload split**: cut on **emb** (disjoint halves of `y`, no reduction), not on `ffn`
(partial sums — the N-writers-onto-one-buffer shape that *is* wall 7). It doubles `maxq` (emb 768:
`group_n` 192, L1 58,368 FIT, sweeps 4, `maxq` 96; emb 1024 no split: 256, 74,752 **OVER**, 128;
split 2: 128, 41,984 FIT, sweeps 8, **256**; split 4: 64, 25,600, **512**), so **fix the `hidden`
refill first, split second**. The split is forced by the **down** herd (both herds allocate
`l1_a/l1_b/l1_c` at `group_n`; breaking the `-D` identity frees only the up stage). Both halves
need all of `H` = `[TILE_M, ffn]`: recompute (+37.7 MB @512, +75.5 MB @1024 DRAM) should not be
built; materialize in L2 (393,216 B at ffn 3072 = 75 % of a memtile; **524,288 at 4096 = 100 %**,
must spread — whether the single-memtile fan-through survives is the last static question). The
band-serial weight term (§3.6) belongs before both.

### 9.4 The selector (row 31): bridge built `[2026-08-14]`, **closed as a negative finding `[2026-08-19]`**

Doc 48's predicate (declaration-side) and doc 47's instrument (routed-artifact-side) live on
opposite sides of the compile; pricing the legal space by compiling sizes at 2.1 s per point as
**3,721,772 points ≈ 90 days / 15,347 (structure, seam) ≈ 9 hours / 428 structures ≈ 15 minutes**,
every route modelling something — a decision, which the operator took: **an analytical model**,
`study/analytical_cost.py`, a fourth route evaluated on the declaration (`Demand.shim_mm2s_slots`,
placement-invariant, 7 of 16 on R1). Traffic half not modelled (31a's lens + §3.6's band term,
reproduced **to the byte** with a coincide-at-one-band discrimination control); rate half labelled
per term (single-port `measured` 5.336 GB/s devq 293; multi-port `modelled`, linear where doc 33's
ladder peaks at 4 tiles — **over-credits wide designs**); doc 47's 0-modelled property survives,
asserted by `ast` on imports. **Closed negative** ([55 §2](55-hexagon-llama-cpp-lessons-for-xdna2.md)):
the licence and the buildable set are disjoint — every axis distinguishing two buildable-and-
runnable designs prices as an exact tie at fixed seam scope (C1 vs C2's measured 6.6 % at 4096 is
dispatch structure, out of scope; byte order `runlist < fused < coarse < offload` against latency
order `fused < coarse < runlist < offload`), and where an axis moves the slot count the order is
purely the disclaimed port term. Both clauses pinned in a test; `RECORDED_MODE_POINTS` refreshed
from the re-walk (bytes identical, latencies down 9–24 %, orderings held); `rank()` prices illegal
declarations without filtering (noted); one agent-reported cost-vs-legality slot disagreement
unverified. For `fused`, a workload-dependent mapping is build-time selection plus a shape-keyed
cache, never runtime dispatch (`offload` takes loop bounds at runtime — [03](03-measurement-model.md)).
Do not copy iron's 1 %-band hardware search (§8's 42.2 % spread).

---

## 10. The tools that survive, and the code

**KEPT** (`agents/probes/`, the supertile seed):

- `probe_r1_rung.py` — build, compile and dispatch one R1 rung (`emb`, `ffn`, `herd_x`, `tile_k`,
  tiling, both H-staging arms, `--omit-pingpong`, `--reuse-elf`), pre-filling the output BO with a
  sentinel and reading it back **on timeout** (`herd_x` bits of progress out of a hang), `--dump-npz`
  making determinism a claim about bytes (sha256 of the raw BO). Every ladder in §6–§7 ran through it.
- `probe_r1_emulate_shape.py` — call the module-interpreting arm at one shape and report
  `max|y − ref|`; EXACT (1e-13 class) at every rung in §6–§7, which is what makes a hardware failure a
  statement about lowering rather than the builder.

**RETIRED at tag `pre-cleanup-20260821`** (commit `cbd2858e`; one line each):

- `probe_r1_arrival_map.py` — decomposed a wrong answer into per-(chunk, row-run) coefficients and
  fitted the `{H_i @ Wd_j}` pairing dictionary (`--self-test` refusing a within-column swap);
  produced §6's map and §7's permutations.
- `probe_aie_buffer_writer_race.py` — read an emitted `aie.air.mlir` and, per buffer, its writer/
  reader DMA channels (`--refuse-race`), simulated the lock protocol as a Petri net
  (`--check-order`: writer order and read binding separately) and read the multi-slot rotation
  phase (`--check-rotation`, `--stream-len`, `--refuse-skew`); self-test 11 cases calibrated both ways.
- `probe_r2_order_seam.py` — R2's row-partition design arm plus three controls verified failing (§5.3).
- `probe_r2_segment_budget.py` — the herd-count sweep to refusal, the per-column shim census over
  both flow kinds, and the kernel-object symbol tables (§5.3–5.4).
- `probe_fused_resident_tail.py` — the 2026-08-10 stitched nt1 + ffn_accum + nt2 composition census (§4.1).
- `probe_ffn_resident_interior.py` — the hermetic structural probe twin of `ffn_resident_structure.py` (§4.2, §9.2).
- `probe_r1_staged_hidden.py` — doc 53's Q1/Q2 staged-`hidden` arms (§9.2).
- `probe_fuse_channels_sibling_nests.py` — wall 3's minimal N-clique reproducer (§4.4).
- `probe_ffn_accum_bd_offset.py` — the frozen-BD miscompile witness the L3-side-offsets rule cites.

**Live code and gate**: `builders/ffn_resident.py` (the builder; flags `shared_h_staging` —
default `True`, deliberately — and `stage_hidden` — experimental, default off; both docstrings
cite this record), `ffn_resident_structure.py` (the STRUCT arm with the widened census and its
in-gate negative control), `builders/test_ffn_resident.py` (the module-interpreting emulation arm,
10 clauses, pinned by `run_ffn_resident_emulation_tests.lit`), and `run_npu2_ffn_resident_peano.lit`
(`UNSUPPORTED: true` — the device gate, parked; its header carries the wall-by-wall record of
§4.4–4.8). Compiler-side regression lits named above live under `mlir/test/` and are summarized in
[16](16-compiler-changes.md). `study/balance.py`, `study/balance_ert.py`, `study/analytical_cost.py`
and their tests are live (§8, §9.4).

`fused`'s SPECS atol stays **PROVISIONAL**, and **no resident-tail latency or byte figure has ever
been measured on hardware.**
