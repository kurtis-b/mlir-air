# PREDICTION-28A-O3O6 — written before this arm's dispatch, 2026-08-19

Same fixed toolchain as job 398 (libAirAggregateCAPI c01824de…). O3 and O6
are the down_K >= 6 rungs; §10.5 proved the rotation skew is NOT their cause
(O3 hung with an IN-STEP rotation; removing the rotation entirely left both
at TIMEOUT). The 28(a) fix changes O6's BD program (STARVED staircase ->
order-preserving cycle+remainder) and should leave O3's untouched
(down_K 6 = uniform trips over 3 slots, plan refuses on r == 0).

| rung | shape | down_K | baseline | predicted now |
|---|---|---|---|---|
| O3 | 32x192 tk32 hx1 | 6 | TIMEOUT 5/5, sentinel 1.0, cores 0/1 | TIMEOUT 5/5, same signature |
| O6 | 32x224 tk32 hx1 | 7 | TIMEOUT 5/5 | TIMEOUT 5/5 (rotation now IN-STEP, hang remains) |

Compile-gate: O3's air-to-aie output byte-identical pre/post fix; O6's
DIFFERS and its --check-rotation reads IN-STEP (baseline STARVED).

Falsifier that would be GOOD news but must be recorded as a surprise: any
O3/O6 leg PASS means the maxq starvation model of §10.6 is wrong and row
28-remainder needs re-opening at a different mechanism.

## RESOLVED AT THE COMPILE GATE — no dispatch run

The compile-gate prediction was HALF WRONG, and the wrong half made the
dispatch unnecessary:

- O3: byte-identical pre/post fix, as predicted.
- O6: ALSO byte-identical — the predicted "O6 DIFFERS, rotation IN-STEP"
  was falsified. O6's --check-rotation still reads SKEWED [0,1,3,4,6,7,2].
  Reason: O6's w_down consumer sites carry trips {3,3,1} (two in-loop sites
  at trip 3 plus a peel), which is NOT a {q,q+1} staircase, so
  NonCleanRotationPlan refuses by design. No two-task cycle+remainder
  program can spell A,B,A,B,A,B,C as an order-preserving rotation — the
  mis-order is decided by the AIR-level loop structure, not by BD task
  bucketing. Fixing it (if it ever matters) is builder-side work, and it
  cannot matter before the down_K >= 6 hang (28-remainder) is fixed, since
  O6 never completes a run to deliver bytes at all.

Since both modules are byte-identical to the pre-fix compiler's output,
devq 330's TIMEOUT 5/5 measurements ARE the post-fix measurements — the
device programs are the same bytes. Dispatching 10 timeout legs to re-read
a sha we already hold would be measurement theater.
