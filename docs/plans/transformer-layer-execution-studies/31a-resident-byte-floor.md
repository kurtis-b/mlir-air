# 31a — The resident byte floor, derived to the byte

`[2026-08-10]` Supporting analysis for [31](31-fused-resident-tail.md). Host-only arithmetic —
no device was dispatched for any number here. Shapes are `baseline_768` (emb 768, ffn 3072,
12 heads × 64), bf16 activations and weights, at seq 512 and 1024. Every derivation is shown so
each number is checkable to the byte; the two measured anchors it must reconcile against are
[27](27-common-ladder-result.md)'s ladder bytes and [26 §C](26-mode-rebuild-feasibility.md)'s
84.0 MiB crossing table, and it reconciles against both **exactly**.

## Two lenses, stated before any number

The port has two different byte counters and they measure different things:

- **`bytes_transferred`** (the dispatch vector, what doc 27's table records) counts **host↔device
  sync traffic** — `bo.sync()` in either direction. It cannot see device-side DRAM traffic at all
  (`fused.py`'s own footgun), it excludes static-weight uploads in steady state (content-keyed BOs
  upload once, and the ladder always warms up — doc 03's cold/steady note), and it **includes**
  verification readbacks of every intermediate boundary.
- **DRAM crossings** (doc 26 §C's lens) counts each tensor once per device-side read and once per
  device-side write of DRAM. This is the quantity residency is *about* — an L1→L1 hand-off removes
  a write+read pair from it — and nothing measures it on hardware; it is logical-traffic
  arithmetic, a **floor** (tiled GEMMs re-read operands, so real DMA traffic is higher).

The resident floor is defined on the second lens. The comparison against the measured 42.5 MB is
on the first. Conflating them is the error this section exists to prevent.

## Tensor sizes (the whole derivation base)

| tensor | formula | bytes @512 | bytes @1024 |
|---|---|---|---|
| activation `[S, 768]` bf16 | S·768·2 | 786,432 | 1,572,864 |
| wide `[S, 3072]` bf16 | S·3072·2 | 3,145,728 | 6,291,456 |
| packed `[2, S, 768]` bf16 | 2·S·768·2 | 1,572,864 | 3,145,728 |
| `qkv_f32` `[S, 2304]` f32 | S·2304·4 | 4,718,592 | 9,437,184 |
| `w_qkv` `[768, 2304]` bf16 | 768·2304·2 | 3,538,944 | same |
| `w_o` `[768, 768]` bf16 | 768·768·2 | 1,179,648 | same |
| `w_up` `[768, 3072]` bf16 | 768·3072·2 | 4,718,592 | same |
| `w_down` `[3072, 768]` bf16 | 3072·768·2 | 4,718,592 | same |
| `gamma1`, `gamma2` `[768]` bf16 | 768·2 each | 1,536 each | same |
| **weights total** | | **14,158,848** (13.503 MiB) | same |

## The resident floor

A fully resident fused layer moves across DRAM, per layer execution: the layer input once, the
layer output once, and each static weight once (weights live in DRAM and must be fetched to the
array every execution; only a host-upload counter can amortize them, not the DRAM interface).

| | @512 | @1024 |
|---|---|---|
| input `x` | 786,432 | 1,572,864 |
| output | 786,432 | 1,572,864 |
| weights | 14,158,848 | 14,158,848 |
| **resident floor** | **15,731,712 (15.003 MiB)** | **17,304,576 (16.503 MiB)** |

The @1024 figure is doc 26 §C's "irreducible 16.5 MiB" row, re-derived independently and agreeing
to the byte (its 13.5 + 1.5 + 1.5 is this table in MiB).

## Reconciling the measured packaged numbers — exact, both lengths, both instruments

Doc 27's ladder records `fused` at **21,233,664 bytes @512** and **42,467,328 @1024**, and doc 28
records **13 sync boundaries**. Both reconstruct exactly from `prepare_fused`'s buffer lists
(`pattern/fused/fused.py`): steady-state traffic is 3 uploads (`x`, `packed1`, `qkv_f32` — the
non-static `host_writes`; at 512/1024 the FFN methods resolve scratch-free and attention exposes
no `y_f32`, so no other f32 scratch exists — confirmed at 1024 by doc 26 §6's 11-arg tail count,
and at 512 by the byte-exact match itself) plus 10 readbacks (the `outputs` tuple):

| direction | tensors | @512 | @1024 |
|---|---|---|---|
| uploads (3) | `x` + `packed1` + `qkv_f32` | 786,432 + 1,572,864 + 4,718,592 = 7,077,888 | 1,572,864 + 3,145,728 + 9,437,184 = 14,155,776 |
| readbacks (10) | `q`+`k`+`v` + `attn_context` + `packed1` + `hidden` + `ffn_up` + `ffn_gelu` + `packed2` + `output` | 3·786,432 + 786,432 + 1,572,864 + 786,432 + 3,145,728 + 3,145,728 + 1,572,864 + 786,432 = 14,155,776 | 28,311,552 (each term doubled) |
| **total, 13 syncs** | | **21,233,664 = measured** | **42,467,328 = measured** |

Adding the 6 static uploads (`w_qkv, w_o, gamma1, w_up, w_down, gamma2` = 14,158,848) gives the
cold-dispatch vector doc 26 §6 records: **19 syncs, 56,626,176 bytes** @1024 — also exact. Two
independent instruments (sync count and bytes) at two lengths, all four reconstructed to the
byte, so the decomposition below rests on measured anchors rather than on reading the code alone.

**What this lens says about residency.** Of the 13 steady-state syncs, only `x` in and `output`
out serve execution of a resident layer; 9 readbacks are intermediate boundaries that exist
*because* the boundaries are L3 args (verification reads them), and the `packed1`/`qkv_f32`
uploads are packaging artifacts (the x-double-upload and the fused-cast scratch ABI). On this counter a
resident layer's steady state is `x` + `output` = **1,572,864 @512 / 3,145,728 @1024** — a
**13.5×** reduction at both lengths (every steady-synced tensor scales with S here, so the ratio
is length-invariant). But most of that 13.5× is *verification traffic disappearing*, not DRAM
traffic disappearing — which is exactly why the next section uses the other lens.

## DRAM crossings per boundary, packaged — and what residency removes

Each row is one device-side crossing of a launch-argument tensor in the packaged three-ELF
`fused` (entry 1 `qkv_proj`, entry 2 `mha_out_proj`, entry 3 `fused_tail`). Attention's interior
(scores/softmax) is on-chip inside FlashAttention and never a launch arg, so it does not appear.

| # | crossing | region | @512 | @1024 | resident? |
|---|---|---|---|---|---|
| 1 | `x` read (qkv A) | front | 786,432 | 1,572,864 | **remains** (layer input) |
| 2 | `w_qkv` read | front | 3,538,944 | 3,538,944 | **remains** (weight) |
| 3 | `qkv_f32` write (GEMM f32 C) | front | 4,718,592 | 9,437,184 | removed |
| 4 | `qkv_f32` read (3 cast launches) | front | 4,718,592 | 9,437,184 | removed |
| 5 | `q`,`k`,`v` write | front | 2,359,296 | 4,718,592 | removed |
| 6 | `q`,`k`,`v` read | front | 2,359,296 | 4,718,592 | removed |
| 7 | `attn_context` write | front | 786,432 | 1,572,864 | removed |
| 8 | `attn_context` read (o_proj A) | front | 786,432 | 1,572,864 | removed |
| 9 | `w_o` read | front | 1,179,648 | 1,179,648 | **remains** (weight) |
| 10 | `attn_out` write (`packed1` plane 0) | front | 786,432 | 1,572,864 | removed |
| 11 | `packed1` read (attn_out + x-as-residual) | tail | 1,572,864 | 3,145,728 | removed¹ |
| 12 | `gamma1` read | tail | 1,536 | 1,536 | **remains** (weight) |
| 13 | `hidden` write | tail | 786,432 | 1,572,864 | removed |
| 14 | `hidden` mirror write (`packed2` plane 1) | tail | 786,432 | 1,572,864 | removed |
| 15 | `hidden` read (FFN A) | tail | 786,432 | 1,572,864 | removed |
| 16 | `w_up` read | tail | 4,718,592 | 4,718,592 | **remains** (weight) |
| 17 | `ffn_up` write | tail | 3,145,728 | 6,291,456 | removed |
| 18 | `ffn_up` read (GeLU in) | tail | 3,145,728 | 6,291,456 | removed |
| 19 | `ffn_gelu` write | tail | 3,145,728 | 6,291,456 | removed |
| 20 | `ffn_gelu` read (down A) | tail | 3,145,728 | 6,291,456 | removed |
| 21 | `w_down` read | tail | 4,718,592 | 4,718,592 | **remains** (weight) |
| 22 | `ffn_out` write (`packed2` plane 0) | tail | 786,432 | 1,572,864 | removed |
| 23 | `packed2` read (ffn_out + hidden mirror) | tail | 1,572,864 | 3,145,728 | removed |
| 24 | `gamma2` read | tail | 1,536 | 1,536 | **remains** (weight) |
| 25 | `output` write | tail | 786,432 | 1,572,864 | **remains** (layer output) |

¹ In a *whole-layer* resident design row 11 disappears (attn_out arrives on-chip and x is still
resident from row 1). In a **tail-only** resident design it remains — it is the front→tail
boundary — which is the per-scope split below.

Totals, with doc 26 §C's @1024 column as the cross-check:

| | @512 | @1024 | doc 26 @1024 |
|---|---|---|---|
| front (rows 1–10) | 22,020,096 (21.000 MiB) | 39,321,600 (37.500 MiB) | — |
| tail (rows 11–25) | 29,101,056 (27.753 MiB) | 48,761,856 (46.503 MiB) | — |
| **packaged total** | **51,121,152 (48.753 MiB)** | **88,083,456 (84.003 MiB)** | **84.0 MiB** ✓ |
| resident floor (remains) | 15,731,712 (15.003 MiB) | 17,304,576 (16.503 MiB) | "irreducible 16.5" ✓ |
| removed by whole-layer residency | 35,389,440 (33.750 MiB) | 70,778,880 (67.500 MiB) | "intermediate 67.5" ✓ |

(A transcription note on doc 26 §C, flagged rather than silently absorbed: its itemized rows sum
to 64.5 MiB against its own stated 67.5 — the `ffn_out` write + read pair (rows 22 + plane 0 of
23 here, 3.0 MiB) is in its total but missing from its item list. This table carries the complete
itemization.)

### The per-scope split — what a resident TAIL buys

Doc 28: `fused` and `coarse` share the same front by construction, so the tail is where residency
work lives. Splitting the removable 67.5 MiB @1024 by scope:

| scope | crossings removed | @512 | @1024 | share of removable @1024 |
|---|---|---|---|---|
| **tail-internal** (rows 13–15, 17–20, 22, 23) | `hidden` ×3 + mirror-read, `ffn_up` ×2, `ffn_gelu` ×2, `ffn_out` ×2 | 17,301,504 (16.500 MiB) | 34,603,008 (33.000 MiB) | **48.9 %** |
| front-internal + front→tail boundary (rows 3–8, 10, 11) | `qkv_f32` ×2, `q/k/v` ×2, `attn_context` ×2, `attn_out`, `packed1` read | 18,087,936 (17.250 MiB) | 36,175,872 (34.500 MiB) | 51.1 % |

A tail-resident layer's crossing total is therefore **front unchanged + tail floor** =
39,321,600 + 14,158,848 = **53,480,448 (51.003 MiB) @1024** (against packaged 84.0 and the
whole-layer floor 16.5), and 22,020,096 + 11,799,552 = **33,819,648 (32.253 MiB) @512** — where
the tail floor is rows 11, 12, 16, 21, 24, 25 (the front→tail packed read, both gammas, both FFN
weights, the output write).

**Headline, both lenses:**

| | @512 | @1024 |
|---|---|---|
| packaged, measured `bytes_transferred` (doc 27) | 21,233,664 | 42,467,328 |
| resident steady state on that same counter | 1,572,864 | 3,145,728 |
| packaged DRAM crossings (derived floor) | 51,121,152 | 88,083,456 |
| tail-resident DRAM crossings | 33,819,648 (−33.8 %) | 53,480,448 (−39.3 %) |
| whole-layer resident floor | 15,731,712 (−69.2 %) | 17,304,576 (−80.4 %) |

`bytes_transferred` will **not** show the tail-resident win: every crossing the tail removes is
device-side. The counter's number changes only through what verification chooses to read back —
which is a gate-design fact [31](31-fused-resident-tail.md) has to handle, not a defect in the
mode.

## The capacity side

Verified numbers, not assumptions: NPU2 on-chip storage is 32 core tiles × 64 KiB L1
(`builders/norm_tail.py` `L1_BYTES = 64*1024`; doc 26 §C `getLocalMemorySize` = 0x10000) =
**2 MiB**, plus 8 memtiles × 512 KiB (`builders/ffn_accum.py` `L2_BYTES = 512*1024`; doc 26 §C
`getMemTileSize` = 0x80000; 8 columns per `python/air/backend/xrt.py`'s npu2 partition comment)
= **4 MiB**: **6 MiB total, and not a flat address space**.

The S×F intermediate against it:

| seq | `[S, 3072]` bf16 | vs 6 MiB chip |
|---|---|---|
| 512 | 3,145,728 = 3.0 MiB | half the chip — and two live at once in the packaged tail (`ffn_up`, `ffn_gelu`) |
| 1024 | 6,291,456 = 6.0 MiB | **the whole chip**, exactly as doc 03 records |
| 2048 | 12,582,912 = 12.0 MiB | 2× the chip |
| 4096 | 25,165,824 = 24.0 MiB | 4× the chip |

So *whole-tensor* residency of the FFN intermediate is out of reach at 1024 and above — this is
doc 03's capacity bound, confirmed. What the bound does **not** forbid is **streaming residency**:
a space-multiplexed tail never materializes the S×F tensor anywhere — it exists only as tiles in
flight on L1→L1 channels. The per-band capacity arithmetic is seq-independent: norm-tail
stage_add holds 2·3·rows_per_call·cols·2 + 1024 = 37,888 B per tile at rows_per_call 4
(`builders/norm_tail.py::_stage_l1_bytes`, with 8 measured to overflow once aircc ping-pongs both
tiles), and the ring's cores hold ping-ponged A (2·64·32·2 = 8,192 B) and B (2·32·192·2 =
24,576 B) beside the resident C (64·192·2 = 24,576 B) = 57,344 B of the 64 KiB tile at tile_k 32,
measured just over it at 64 (`builders/ffn_accum.py`). The capacity bound therefore bounds
*whole-tensor* residency, not the resident tail; the real bounds on the resident tail are the
column budget and the composition seams, which [31](31-fused-resident-tail.md) takes up.

## Reproduce

Every number above regenerates from the formulas in this file with a calculator; the derivation
script used while writing it lived in session scratch and is deliberately not shipped — the doc
IS the derivation. The two measured anchors are doc 27's table (`results/common-ladder-w1/`,
`-w2/`) and doc 26 §6's repair-run vector; if either changes, this document's reconciliation
section is the first thing to re-check.
