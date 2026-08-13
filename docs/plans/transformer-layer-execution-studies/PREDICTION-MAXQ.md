# PREDICTION — row 28 defect (b), the `maxq` hang

Written 2026-08-12, BEFORE any compile of the fixed compiler and BEFORE any
device leg. Sha256 of this file is printed in leg 0 of every job below.

## What changed

`AIRRtToNpuPass.cpp`, new step `boundIdenticalShimPutRuns`, run once after
`boundShimBdLiveness`. It paces a run of >= 3 **structurally identical**
fire-and-forget MM2S configure/start pairs on ONE shim channel to depth 2,
reusing 6b's `paceShimFeedForBdReuse` unchanged (sink past the feeds it must
not out-order; issue_token + `dma_await_task(t[i-2])` before task i's
CONFIGURE; drain the tail).

The fold doc 52 §10.9 asked for is NOT what shipped, and §0 of the write-up
says why: `repeat_count` IS the descriptor's iteration dimension, so folding
`trip` identical copies needs a stride-0 fifth dimension the hardware does not
have. That is a refutation of the specified fix, recorded before this run.

## Clause 0 — compile-only gate, must pass before any device leg

- `O3` (`down_K` 6): `@air_channel_2`'s **6** configure/start pairs become
  **6 configures, each issue_token = true**, with 4 paced awaits (i=2..5)
  before their configures and a 2-await drain, and the whole run SUNK to after
  `dma_start_task` of `@air_channel_3` (`w_up`) and `@air_channel_4` (`w_down`).
  `maxq` on `%shim_noc_tile_0_0 / MM2S 0` drops **6 -> 2**.
- `O5` (`down_K` 4): 4 identical pairs, same rewrite, `maxq` **4 -> 2**.
- `O1` (`down_K` 4, `16x64`): same.
- Any rung whose `@air_channel_2` run is < 3 tasks is untouched.

## Clause 1 — the hangs

`O3` (`down_K` 6) and `O6` (`down_K` 7) go from **TIMEOUT 5/5 to PASS 5/5**,
one `y` sha256 across 5 fresh processes.

This is the clause that decides the row. If `maxq` is the mechanism, bounding
it at 2 removes the hang; if the hang survives at `maxq` 2, `maxq` is not the
mechanism and §10.6's separation was a correlation.

## Clause 2 — the wrong answer is NOT fixed

`O2` (`down_K` 5) stays **FAIL 5/5** with the SAME permutation
`sigma = [0,1,4,2,3]` and, at the shape level, the same `y` sha256 family as
devq 330's. Defect (a), the L2 slot-rotation phase skew, is untouched by this
change and must stay untouched. **If `O2` moves, say so loudly** — it would
mean this step perturbed the rotation, which it has no business doing.

Consequence, stated so it cannot be quietly dropped: with (b) fixed and (a)
open, the rung ladder becomes PASS at 2/3/4, **FAIL at 5**, PASS at 6+ — a
NON-MONOTONIC ladder. That is the signature of two independent defects and it
is a prediction, not a hedge.

## Clause 3 — the <=4 rungs do not move

`O5` (`down_K` 4) and `O1` (`down_K` 4) stay PASS 5/5.

They are **NOT** predicted byte-identical: their `@air_channel_2` runs are 4
identical tasks, which is >= 3, so this step paces them too and their
`npu.air.mlir` changes by construction. The check that distinguishes a fix from
a change is therefore stated differently here: the **`y` output sha256** of
`O5` and `O1` must be **byte-identical to the pre-fix `y` sha256**, across 5
fresh processes. Same answer, different control program.

## Clause 4 — where the shape stops being limited by this

`maxq` is pinned at 2 for every `down_K` after this change, so the hang
boundary should not reappear as `down_K` rises. The next binding constraint is
predicted to be **defect (a)**: the rotation skew fires whenever
`down_K % slots != 0`, so `down_K` 5, 7, 10, 11, ... give wrong answers and
6, 8, 9, 12 give right ones. The gate shape's `down_K = 96` is IN-STEP by
§10.4's reading, so if both clauses hold the gate is limited by neither — and
the next wall is whatever is above them, which is NOT predicted here because
nothing has measured it.

## Clause 5 — no shipped design moves

`check-air-mlir` goes **499 pass / 0 fail -> 500 pass / 0 fail**, the delta
being exactly the new regression lit
`mlir/test/Conversion/AIRRtToNpu/identical_shim_put_run_bound.mlir`. Any other
moved test is a falsification of this clause and is reported as one.

## Falsifiers, named

- `O3`/`O6` still TIMEOUT -> `maxq` is not the mechanism. Report as falsified.
- `O2` changes verdict or permutation -> the step touched defect (a). Report.
- `O5`/`O1` change their `y` sha -> the pacing changed an answer. Report.
- any `check-air-mlir` test other than the new one moves -> a shipped design
  moved. Report and stop.
