# 54 — The first `full` profile, the three walls it found, and the decoder families

`[2026-08-20]` Phase G's gate — "a full profile run with a complete `results_manifest.json`" —
had never been attempted; [34 §5.2](25-mode-rebuilds-and-results.md) estimated it at ~2 h cold and said
so. It was run (devq 427, 1902 s), it was not met (20 passed / 10 failed / 6 skipped), and
every one of the ten failures turned out to be a bound that can be read from the builder that
refuses. This document records the measurements, the three walls, the repairs, and the state
the gate was met in (devq 434, then a clean single-session walk 2, devq 435). It also records
the first walks of the two decoder families that had never run, and the iron adapter's first
run against a real iron tree, because both happened the same day and both are cited from the
README's status board.

Everything here is `baseline_768` (768 × 3072, 12 heads) unless a family is named. Turbo was
verified in-job on every run. Results roots are gitignored:
`results/g-full-baseline768-w{1,2}/`, `results/decoder-gpt2_512-smoke-w{1,2}/`,
`results/decoder-gpt2_medium_1024-smoke-w{1,2}/`, `results/iron-validation-20260820/`; each
holds its devq job log.

## 1. The matrix, as measured — walk 2 (devq 435, one session, 419 s warm, `complete: True`)

| mode | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|---|---|---|---|
| `coarse` | FA floor | FA floor | **22.5** | **43.5** | **82.9** | **153.3** | **325.7** | **716.0** | **1505.8** |
| `offload` | GEMM tile | GEMM tile | GEMM tile | **109.5** | **142.7** | **314.0** | **823.8** | **2195.1** | **6854.1** |
| `runlist` | GEMM tile | GEMM tile | GEMM tile | **72.8** | **131.2** | **261.5** | **633.9** | **1748.1** | softmax L1 |
| `fused` | packing | packing | **22.1** | **40.6** | **79.5** | packing | packing | packing | packing |

Average latency in ms, `--warmup 1 --samples 3` (the `full` profile's settings — NOT doc 32's
`2/5`, so do not splice these against doc 32's post-flip walk; see [32](32-cost-decomposed-ladder.md)
and the 2026-08-18 re-walk row for why warmup matters at 1024). A named cell is a `skipped` row
whose reason is derived from the builder (§3). 21 measured, 15 skipped, 0 failed.

DRAM traffic (`bytes_transferred`, walk-identical, pmode-independent):

| mode | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|---|---|
| `coarse` | 11.0 MB | 22.0 | 44.0 | 88.1 | 188.7 | 478.2 | **1,006.6** |
| `offload` | — | 44.0 | 99.1 | 284.7 | 957.9 | 3,512.2 | **13,452.7** |
| `runlist` | — | 20.4 | 40.9 | 81.8 | 176.2 | 453.0 | — |
| `fused` | 10.6 | 21.2 | 42.5 | — | — | — | — |

**The corrected axis at its extreme.** `offload` keeps the softmax on the host, so every
`[seq, seq]` score matrix crosses DRAM twice per head and its traffic grows as seq²: 2.3× `coarse`
at 1024, **13.4× at 16384**. The other three modes grow linearly. This is [03](03-measurement-model.md)'s
"reconfiguration cost against DRAM traffic" measured over a 256× range of sequence length, and
it is the cleanest single table the study has produced on that axis.

Dispatch vectors are length-independent where the mode is packaged (`offload` 30/90/90,
`fused` 1/23/13, `coarse` 4 submissions) and grow with length where it is not (`runlist` 17
submissions, `herd_launches` 151 → 873 over 512 → 8192).

**Walk-to-walk** (`compare_roots` w1 → w2, both `turbo (observed)`, 9/9 rows matched per
file including the skipped ones): `avg_latency_ms` drift medians `runlist` 0.61% < `fused`
0.72% < `coarse` 0.93% < `offload` 1.58%, p90 ≤ 4.29%, one `coarse` rung at 18.4% max
(`coarse` @256, the first rung after compile). `VERDICT: OK`. The same per-mode stability
ordering doc 32 recorded, now over nine lengths. **Cite absolute numbers from w2**: w1 was
measured partly from a dirty tree across two commits (its manifest flags this).

## 2. Seven first-ever measurements

Nothing had walked above 4096 or below 512 before this run.

- `coarse` at 8192 and 16384 **built and passed on the first attempt** — 716 / 1506 ms,
  `sync` 781 / 1550. The block's banded norm tail and the FlashAttention interior scale to
  16384 rows without a change.
- `offload` at 8192 / 16384: 2195 / 6854 ms, 30 dispatches either way.
- `runlist` at 8192: 1748 ms — only after the softmax repair (§3.2); 16384 is a wall.
- `coarse` and `fused` at 256: 22.5 / 22.1 ms, `fused`'s lower bound. At this length the
  two are within noise of each other, as they were at 1024 on 2026-08-18.

## 3. The three walls, and what each one is

All ten first-session failures were deterministic refusals **before aircc** (or, for one,
inside it with the message lost). They are the same class as `fused`'s packing bound, which
`profiles.skip_reason` has always recorded as a structural skip on exactly that criterion —
"the builder raises before aircc is reached". The profile's own rule is that such a skip must
be **derived from the builder's source, never typed**, and each one now is
(`study/profiles.py`: `FA_PARALLEL_SEQ`, `ATTN_GEMM_SEQ_MULTIPLE`, `softmax_fits_l1`;
`study/test_profiles.py` re-derives every constant from the refusing builder by `ast`;
`run_profile_bounds_tests.lit` constructs every module at every length with air and asserts
skip ⇔ refuses for all four modes).

### 3.1 The FlashAttention floor — `coarse` (and its cells, and `fused`) below 256

`builders/mha_attention.attention_config` requires `seq_len % parallel_seq == 0` with
`parallel_seq = 256` (iron's `lqp`, the Q rows one launch iteration owns) and raises
`seq_len (64) must be divisible by parallel_seq (256)`. This is also the origin of
`FUSED_SEQ_MIN`. A smaller `parallel_seq` is a builder change and would lift it.

### 3.2 The softmax row width — `runlist` at 8192 (repaired) and 16384 (wall)

`runlist` pinned `SOFTMAX_ROWS_PER_CALL = 2`, sized at 4096 where three `[2, 4096]` bf16
tiles are 48 KiB of a 64 KiB L1. At 8192 that is 96 KiB, and aircc failed with an **empty error
body** after 88 s (468 s at 16384) — the installed backend dropped the return code, so a killed
compiler and a compiler error were indistinguishable from the log; `python/air/backend/xrt.py`
now carries `returncode` in the message (install refresh pending). Reproduced with rc/stderr
kept (devq 428): a real compile failure of the softmax module, not memory.

Repair, the same one the layer-norm width wall took ([50 §7](25-mode-rebuilds-and-results.md),
`run_layer_norm_rows_tests.lit`): `builders/softmax.derive_rows_per_call` returns the largest
legal value **at or below the historical constant**, so 512–4096 emit byte-identical IR
(asserted as byte equality with the discrimination control that 1 vs 2 does change the
module), 8192 gets the one row that fits (49,160 B), and 16384 — where even one 16384-wide
row is 98,312 B — refuses **by name at config time** (`run_softmax_rows_tests.lit`, 9/9). An
explicit `rows_per_call=2` at 8192 still raises, which separates deriving the parameter from
deleting the constraint. Lifting the 16384 wall needs a column-chunked softmax: kernel work.

### 3.3 The attention GEMMs' tile multiple — `offload` and `runlist` below 512

Both modes dispatch the two per-head attention matmuls through fixed measured tiles
(`pattern/offload/offload.ATTENTION_GEMM_TILES`; `runlist` injects the same spec). The GEMM
builder asserts each dimension against its tile × herd before any IR is built. For
`attn_scores` ([seq, 64] @ [64, seq]) both `m` and `n` are the sequence length: `m` must divide
by `tile_m × herd_m = 32 × 8 = 256` (the `(64, 32, 8)` assertion at 64 and 128 is the **m**
check) and `n` by `tile_n × herd_n = 128 × 4 = 512` (the `(256, 128, 4)` at 256); for
`attn_output` `k` must divide by `tile_k_l2 = 256`. The binding multiple is 512. Smaller tiles,
measured the way these were ([offload.py](../../../programming_examples/transformer_layer/pattern/offload/offload.py)
§"THE TWO ATTENTION GEMMS"), would lift it.

### 3.4 What "complete" means after this

Every rung the current designs can build is measured; every rung they cannot is a `skipped`
row naming the builder clause that refuses it. That is exactly the standing `fused`'s six
skips have had since G0. The `ast` pins and the bounds lit are what make a builder change that
lifts a bound into a **test failure** rather than a stale skip — the stale-skip direction is
the dangerous one, because a skipped rung is not a failure and a walk would report complete
having never attempted a length the mode now supports.

## 4. The two decoder families that had never walked

`gpt2_small_768` walked on 2026-08-19. The other two declared decoder families had not.

| family | shape | walk 1 | walk 2 | `fused` | `coarse` | `runlist` | `offload` |
|---|---|---|---|---|---|---|---|
| `gpt2_512` | 512 × 2048, 8 heads | 4/4 (devq 429) | 4/4 (433) | 55.2 / 55.9 | 57.8 / 57.5 | 114.1 / 113.8 | 130.7 / 125.1 |
| `gpt2_medium_1024` | 1024 × 4096, 16 heads | **2/4** (430) | 4/4 (432) | — / 115.2 | — / 122.6 | 259.4 / 254.4 | 239.5 / 218.9 |

All at seq 1024 (the smoke profile's decoder length), ms avg, walk 1 / walk 2. Bytes
walk-identical. The `fused < coarse` ordering holds in all three decoder families.

**`gpt2_medium_1024`'s first walk failed on the two device-norm modes** — `coarse` `ffn_out`
`atol_required` 4.869e-1 (2 of 1,048,576 elements over 4.5e-1), `fused` 5.412e-1 (40 over) —
while `offload` and `runlist` read 12/12 clean and every mode's mean-relative error sat where
the 768 family puts it (coarse 4.95e-2, fused 5.57e-2; `offload`'s tail 2.40e-1, its
pre-norms being host f32). That is the element-wise tail of a 4096-deep bf16 reduction against
`DECODER_STAGE_ATOL`, a table **measured at one shape** (512 × 768 × 3072, devq 359/360) and
at no other width — the same "constant sized at one width" class as §3, in the tolerance
table rather than a builder. It is not the 25× relative-excess shape of the attention-scale
defect (README, 2026-08-19 row), and the discriminator is the mean-relative error, which did
not move.

Handled the way the base table was: `opcheck_layer.decoder_stage_atol(hidden)` is the one
authority the four modes read, with a **measured** per-width entry (`1024 → 6e-1` on exactly
`ffn_out` / `output`, 1.11× over fused's 5.412e-1); every other width returns the base table
byte-for-byte and a test pins that, plus that overrides may only name boundaries the base
table has. The confirmation walk read 4/4.

## 5. The iron adapter's first real run (success criterion 3)

`study/iron_adapter.py` had existed since schema v1 — row-level, refusing latency, power,
dispatch counts and `run_status` on documented grounds — and had never read a real iron tree.
Against iron's 162-row full suite (`results_unattended_full_suite_20260801_023954`, iron
`c885c1e4`, same HX 370) its declared key matched **0 rows**: this port stamps `study_case_id`
with the SPECS shape key (`512x768_encoder_bert`, the resume key) where iron stamps its family
id (`baseline_768`). The adapter now translates through `cases.FAMILY_SPECS`, refusing an
unknown family or a row contradicting its own width, and `validate_port` checks the seven
shape fields per shared point, reads no latency on either side, and names why a row has no
counterpart. Four roots validate `OK`, 0 disagreements. The pre-commit Codex review found the
first cut passing vacuously when nothing was compared; fixed and pinned.

**What the join puts side by side, and the adapter refuses to compare**: iron's `baseline_768`
reads `hybrid` 15.1 / 21.1 ms, `runlist` 24.5 / 29.3, `offload` 10.9 / 27.4 at 512 / 1024,
against this port's 43.5 / 82.9, 72.8 / 131.2, 109.5 / 142.7 (§1). Every shared point is
**2.7–9× lower on iron's side**. Three reasons that is not a result here: iron's
`timed_total_sec` is a power-sampler-chosen span on its power path (the file these came from);
iron's `offload`/`hybrid` keep attention and the non-linear operators on the host under the
superseded taxonomy where every mode here runs them on the array since 2026-08-09; iron's
pmode is not in its manifest. A gap this size is not explained by the first alone.
~~**Attributing it is a real open item**~~ **`[2026-08-22]` ATTRIBUTED** (queue item 8; evidence
`results/iron-gap-20260822/`, devq 503 / 504, both sides in one session under recorded Turbo,
iron at devel HEAD `cc7083f`):

**The two clocks have the same shape.** iron (`study/end_to_end/modes.py:2173`) wraps one
`_run_pattern_once` forward in `perf_counter`, warm-up excluded, `runs_per_sample` iterations,
avg/min/max — `timed_total_sec` is the plain sum of those latencies, so the "power-sampler span"
reading above was wrong. Ours (`study/run_mode.py`) wraps one `dispatch()` the same way.

**What differs is what a forward contains.** Every mode's dispatch seam runs the per-boundary
correctness comparison (`_stage_stats`: ten full-array float32 `abs`/`sum`/`max` passes per
forward) **inside** the clock; iron's region has no such check. Profiled (`prof_coarse_512.pstats`),
a `coarse` forward at 512 is ~42 ms = `_stage_stats` ~22 + `bo_pool.sync_to_device_if_needed`
~10 (H2D BO syncs, which the `sync_ms` column under-reports at ~1) + device execute+wait 7.4 +
~1. `run_mode.py --no-stage-stats` (opt-in; the warm-up still verifies) removes the first term
and is the like-for-like measurement:

| `baseline_768`, 512 | iron (devq 504) | ours, comparison in the clock (default) | ours, `--no-stage-stats` | ratio |
|---|---|---|---|---|
| hybrid ↔ `coarse` | **15.07** (min 14.77, 5 dispatches) | 44.90 (min 42.25) | **17.25** (min 16.17; device 7.39, sync 0.93) | 1.14× |
| hybrid ↔ `fused` | 15.07 | 38.52 | **14.83** (min 14.15; device 6.26) | 0.98× |
| runlist | **24.29** (min 24.22, 16) | 71.70 | **46.30** (min 45.34; device 21.74) | 1.91× |
| offload | **12.11** (min 10.89, 30) | 101.03 | **77.44** (min 75.08; device 60.49) | 6.4× |

So the device-resident modes are at parity with iron's `hybrid` once the comparison is out of the
clock. What remains: (a) **`runlist`**'s 1.9× and `coarse`'s residual ~10 ms are host-side BO
traffic — and `builders/block.py`'s `_sequence_*` rebuild every `BufferSpec` per forward, re-hashing
the static weights (`content_key`, rule S1: 36 SHA-256 calls over ~14 MB per six forwards ≈ 6 ms
each) before the pool decides nothing changed; iron keeps its weights resident once. (b)
**`offload`**'s 6.4× is `device_ms` itself — 60.5 of 77.4 ms for 30 submissions — because the
corrected `offload` runs attention and the small operators on the array where iron's keeps them
on the host; that is the taxonomy difference §1 names, measured. Two follow-ups, both operator
questions rather than increments: whether the study's measurement model should move the
comparison out of the clock by default (every standing number in [27](27-common-ladder-result.md),
[32](32-cost-decomposed-ladder.md) and §1 would shift down by the ~24–27 ms of comparison per
forward), and whether rule S1's content key should be computed once per plan rather than per
sequence (a `block.py` change that moves `coarse`/`fused` by ~6 ms).

## 6. Open after this document

- **`runlist` at 16384**: a column-chunked device softmax (kernel + builder).
- **Sub-256 / sub-512 lengths**: a smaller `parallel_seq` and smaller attention tiles, each
  measured as the current ones were; the `ast` pins flip the skips to rungs automatically.
- **The iron latency gap** (§5): unattributed.
- **The big-three model leg**: `qwen25_3b`, `llama32_3b`, `qwen3_4b` deferred — their
  `verify` is oomd-killed with the whole session ([15](15-environment-notes.md)); 8/11 is the
  standing leg.
- **`install-xrt` refresh** so the backend's `returncode` message reaches runs.
- **`attn_npu1.py`** carries the same Q-block counter fix as NPU2, unverified (no NPU1 here).
