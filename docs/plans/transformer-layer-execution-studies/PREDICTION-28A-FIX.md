# PREDICTION-28A-FIX — written before any dispatch, 2026-08-19

The 28(a) fix (NonCleanRotationPlan in air::getRepeatCounts +
generateDmaBdProgram task emission, uncommitted working tree at this
timestamp) is on. Compiles happen off-queue into per-rung dirs; the measure
job dispatches 5 fresh-process legs per rung with --reuse-elf, atol 5e-2,
seed 13, tiling 2,2, shared H staging — identical conditions to devq 328/330.

## Compile-gate clauses (must hold before dispatch)

1. O2 / T5 / K5 `aie.air.mlir` --check-rotation reads **IN-STEP** on every
   multi-slot L2 rotation (baseline read SKEWED on the w_down feed).
2. O5 and O1's air-to-aie output (`aie.air.mlir` from air-opt on the rung's
   `placed.air.mlir`) is **byte-identical** with the fix reverted vs applied,
   measured on the CURRENT tree with the same air-opt build cycle. The plan
   only fires on a non-uniform {q,q+1} staircase; uniform-trip rungs must be
   untouched.
   [AMENDED before dispatch, after compile: the first draft of this clause
   compared today's O5/O1 elf sha against devq 330's Aug-12 provenance shas
   (aefed272…, ca8ba8dd…) and they do NOT match — but 28(b) and other
   commits landed between Aug-12 and today, so that comparison confounds
   this fix with everything else that landed. The clause as now stated is
   the one the fix is accountable to.]

## Dispatch predictions (5 fresh processes each)

| rung | shape | down_K | baseline (devq 328/330) | predicted now |
|---|---|---|---|---|
| O5 | 32x128 tk32 hx1 | 4 | PASS 5/5 | PASS 5/5, one y sha across legs |
| O1 | 16x64  tk16 hx1 | 4 | PASS 5/5 | PASS 5/5 |
| O2 | 32x160 tk32 hx1 | 5 | FAIL 5/5, one sha, σ=[0,1,4,2,3] | **PASS 5/5** |
| T5 | 80x80  tk16 hx1 | 5 | FAIL 5/5, σ=[0,1,3,4,2] | **PASS 5/5** |
| K5 | 160x160 tk32 hx1 | 5 | FAIL 5/5, σ=[0,1,3,4,2] | **PASS 5/5** |

Falsifier: any O2/T5/K5 leg FAIL or TIMEOUT, or any O5/O1 leg not PASS, or
clause 2 sha drift, kills the "order-preserving BD program" claim and the fix
does not land.

Not predicted here: O3/O6 (down_K >= 6) stay TIMEOUT per §10.5-§10.6 — that
is 28-remainder, a different mechanism, measured separately if at all.
