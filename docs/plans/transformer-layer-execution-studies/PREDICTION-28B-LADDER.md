# PREDICTION-28B-LADDER — written before this arm's dispatch, 2026-08-19

Supersedes PREDICTION-28A-O3O6.md's conclusion, which was WRONG one level
down and is retracted here rather than edited away: air-to-aie byte-identity
does NOT make the full artifact identical, because 28(b) (c634f735) lives in
airrt-to-npu, downstream. Measured: O3's elf is 0ed23416… vs devq 330's
b2c0f26f…, O6's e9a88a68… vs f8cb1cb7…, and O3's npu.air.mlir now carries
7 issue_tokens — the 28(b) pacing FIRES on these rungs' refill runs. The
down_K >= 6 TIMEOUT has never been measured against a compiler in which the
pacing fires. devq 330's verdicts are measurements of a binary that no
longer exists.

## The differential prediction, from the two-defect model with both fixes in

- O3 (32x192 tk32, down_K 6): rotation IN-STEP (measured today), refill now
  paced to depth 2 → the only known defect on it is gone → **PASS 5/5**.
- O6 (32x224 tk32, down_K 7): refill paced, but its rotation is STILL SKEWED
  [0,1,3,4,6,7,2] (the {3,3,1} trips refuse the 28(a) plan, §13.4) → the
  hang lifts and the skew becomes visible for the first time →
  **FAIL 5/5, a byte-deterministic wrong answer, one y sha across legs**.

The O6 clause is also the attribution control for today's O2 flip: O6 is
"pacing active + skew intact". If O6 FAILs deterministically, pacing does
not mask a skew-class wrong answer, so O2's FAIL→PASS belongs to the 28(a)
rotation fix, not to the pacing it rode in with.

Falsifiers, each named: O3 TIMEOUT → the >= 6 hang has a second mechanism
beyond queue occupancy that pacing does not close (28-remainder reopens at
§10.6's instrument). O6 TIMEOUT → same, and the skew stays latent. O6 PASS →
either the skew is not what §10 measured it to be, or the two memtile sides
agree at down_K 7 after all — the O2 attribution then needs its own arm
(pre-28(a) toolchain, full install, O2 dispatch).

## RESOLUTION [post-run — the sha stamped in devq 403 leg 0 (cb2c7472…) covers
## everything above this line]

Measured, devq 403, 10 legs:

- **O3: PASS 5/5** — mismatches 0/2048, corr 0.99987, one signature across
  legs. THE CLAUSE HELD: the down_K = 6 hang is gone under 28(b)'s pacing.
  devq 330's O3 TIMEOUT was a measurement of a binary that no longer exists.
- **O6: TIMEOUT 5/5** — sentinel 1.0000, cores finished 0/1, the baseline
  signature exactly. **THE CLAUSE WAS FALSIFIED**: predicted FAIL with
  deterministic wrong bytes, measured a hang. Per the named falsifier, the
  down_K >= 7 wedge has a mechanism pacing-to-depth-2 does not close — OR
  the still-skewed rotation now deadlocks under flow control instead of
  delivering wrong bytes (a skewed drain can hold a lock the paced fill
  needs; unpaced, the fill ran ahead and the skew surfaced as bytes).
  Not separated here; row 28-remainder is now "down_K >= 7, mechanism
  unresolved", bounded at 6 by O3's PASS.
- The O2-attribution corollary: pacing alone does not silently green a
  skewed module (O6 wedges rather than passes), consistent with O2's
  FAIL→PASS belonging to the 28(a) rotation fix.
