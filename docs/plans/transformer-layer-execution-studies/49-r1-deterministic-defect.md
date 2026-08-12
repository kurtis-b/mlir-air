# 49 — R1's deterministic wrong answer, located

`[2026-08-12]` Queue item 23. Artifacts: **devq 278** (reproduction + bisection),
**devq 294** (out-of-sample model test). Probe: `agents/probes/probe_r1_rung.py`
(+ `--dump-npz`). Regression test:
`mlir/test/Conversion/AIRToAIE/air_channel_to_locks_shared_buffer_producer.mlir`.

## The verdict in one paragraph

R1's deterministic wrong answer is **not** an eighth wall in the residency
composition. It is a **compiler defect in `air-to-aie`'s core-side lock
placement**, it has nothing to do with the resident design, and it fires for any
core that writes an L1 buffer once and then sends it out in **more than one**
`air.channel.put` on the same DMA channel. `AIRToAIEPass.cpp`'s
`sharedStagingBuffer` path (#1515) places the acquire *immediately before each
put*. That paces put *i+1* against put *i* — but no acquire dominates the core's
own **writes**, so after the last put's release the core falls straight into the
next round and overwrites the buffer **while the last BD is still streaming out
of it**. The core wins that race after roughly one run, so the final chunk is
delivered as *its first 8-row run, correct, followed by seven runs of zeros* —
zeros because the overwriter is `ffn_zero_bf16_up_proj`, the accumulator memset.
Both racers are fixed-rate hardware starting from a fixed offset, which is why a
race produces a **byte-identical** answer on every run.

## What was measured

### 1. It reproduces, and the determinism claim is a claim about bytes

`--dump-npz` was added to `probe_r1_rung.py` so "5/5 identical" could be checked
by `sha256` of the raw output BO rather than by a correlation that rounds the
same. **devq 278**, six rungs, 20 dispatches, one fresh process each:

| rung (herd_x=1, tile_k=32) | sweeps | k' | cpg | runs | distinct y sha256 | verdict |
|---|---|---|---|---|---|---|
| `emb 32  ffn 32`  | 1 | 1 | 1 | 3 | **1** | PASS |
| `emb 32  ffn 64`  | 2 | 1 | 1 | 3 | **1** | PASS |
| `emb 32  ffn 128` | 4 | 1 | 1 | 3 | **1** | PASS |
| `emb 64  ffn 64`  | 1 | 2 | 2 | 3 | **1** | **FAIL** 2000/4096, corr 0.729 |
| `emb 64  ffn 128` | 2 | 2 | 2 | 3 | **1** | **FAIL** 1788/4096, corr 0.663 |
| `emb 128 ffn 128` | 1 | 4 | 4 | **5** | **1** | **FAIL** 1932/8192, corr 0.869 |

Every rung is byte-deterministic. The item-23 rung reproduces exactly the
recorded numbers (1932/8192, corr 0.868935042).

### 2. The minimal failing configuration is `chunks_per_group = 2`

`emb_dim 64, ffn_dim 64, herd_x 1, tile_k 32` — **sweeps 1, k' 2, cpg 2**. It is
the smallest rung on the ladder that fails, and it is a 2-element output tile
away from the degenerate rung.

**`sweeps` is excluded by measurement.** `emb 32 / ffn 128` runs **4 sweeps** —
four w_down refill tasks, four H chunks, four down-K steps — and is **correct**.
`emb 64 / ffn 64` runs **one** sweep with two chunks and is wrong. So neither the
number of down-K steps, nor the number of w_down refills, nor the task count is
the trigger. The trigger is **more than one chunk out of one up-herd group**,
i.e. `chunks_per_group > 1`.

At `herd_x = 1` with `tile_k` pinned at `MAX_L1_TILE_K = 32`, `k_steps_up` and
`chunks_per_group` are the same number (`emb_dim/32`) and cannot be separated;
separating them needs `herd_x > 1`, which lands in item 21's race. The IR
attribution below removes the ambiguity without needing the separation.

### 3. Where the wrong elements are — a per-(chunk, row-run) arrival map

Three instruments, each validated on the passing control before being believed:

- **Recover the intermediate.** At `ffn_dim == emb_dim` the packed `w_down` is
  square, so `A_hat = y_hw @ inv(w_down)` recovers the H the down herd actually
  consumed — the resident intermediate that never touches DRAM and had never
  been observed. On the passing rung it recovers H at relL1 0.085.
- **Term decomposition.** `y = Σ_j H_j @ Wd_j` by design; least-squares `y_hw`
  over the full dictionary `{H_i @ Wd_j}`. Control: coefficient **+0.9999** on
  the single diagonal term, residual 0.0162 (the bf16 noise floor).
- **Per-(chunk, row-run) fit.** One basis matrix per (down-K step *j*, 8-row run
  *r*) of `H_j`; the runs occupy disjoint output rows so every coefficient is
  identifiable, and no matrix inversion is involved.

The arrival map is the result (1.00 = that run arrived):

```
emb 128 ffn 128, sweeps 1, cpg 4          emb 64 ffn 64, sweeps 1, cpg 2
 j (s,jj)  r0   r1   r2   r3 ...           j (s,jj)  r0   r1  ...
 0 (0,0)  0.99 1.00 1.00 1.00 ...          0 (0,0)  1.01 1.00 ...
 1 (0,1)  1.00 1.00 1.00 1.00 ...          1 (0,1)  0.74 0.00 ...  <== last of group
 2 (0,2)  1.00 1.00 1.00 1.00 ...
 3 (0,3)  0.88 0.00 0.00 0.00 ...  <== last of group
```

`emb 64 / ffn 128` (sweeps 2, cpg 2) fails on **both** groups' last chunk, so the
rule is *last chunk of every group*, not *last chunk overall* — and `emb 32 /
ffn 128`, whose fourth chunk *is* the last chunk overall, is entirely correct.

The structure is therefore: **one 8-row run of `tile_k*MICRO = 256` elements
survives; the other seven are zero.** 256-with-a-stride is a quantity that exists
in exactly one place in the design — the up herd's chunk put,
`sizes [1, 8, 256] strides [chunk_run, group_n*MICRO, 1]`. Every other transfer
on this path is a contiguous 2048.

### 4. The IR says why, and which pass

Dumped with `debug_ir` at both rungs. The `aie.dma_bd` descriptors are **correct**
at the failing rung — `offset = 0` and `offset = 256`, `len = 2048`,
`sizes = [8, 256]`, `strides = [512, 1]`. The defect is in the **core**, and it
is present already at `pass_045_after_air-to-aie`:

```
PASSING (cpg = 1)                     FAILING (cpg = 2)
^bb1:                                 ^bb2:
  use_lock(WLOCK, AcquireGreaterEqual)   func.call @ffn_zero_bf16_up_proj(%buf2)
^bb2:                                    scf.for k { acq A,B; mm; rel }
  func.call @ffn_zero_bf16_up_proj        use_lock(WLOCK, AcquireGreaterEqual)
  acq A,B; mm; rel                        use_lock(RLOCK, Release)
  use_lock(RLOCK, Release)                use_lock(WLOCK, AcquireGreaterEqual)
  cf.br ^bb1                              use_lock(RLOCK, Release)
                                          cf.br ^bb1   -> straight back to the zero
```

At `cpg = 1` the acquire is hoisted to the block head and dominates the memset.
At `cpg > 1` `sharedStagingBuffer` is true, both acquires sit at the puts, and
**nothing guards the memset at the top of the next round**. Lock counts are
conserved either way (N acquires, N releases) — this is a *placement* defect, not
a *count* defect, which is why item 18's conservation audit could not see it.

### 5. A rate model, fit on two rungs, and tested out of sample

If the corruption is the core's memset overtaking the in-flight BD, the surviving
prefix of run 0 is
`f0 = min(1, (base + C) / (chunk_run * (rho - 1)))` with
`base = (cpg-1)*chunk_run` and `rho = r_memset/r_dma`. Fitting the two clean
single-sweep rungs (`f0 = 0.74` at base 256, `f0 = 0.88` at base 768) gives
**rho ≈ 15.3** — a 32-lane bf16 vector store against a ~2 bf16/cycle core-to-core
stream — and **C ≈ 2451 elements**, about 77 core cycles of head start for the
DMA (a branch plus a call).

**The prediction was recorded before the measurement** (`PREDICTION.md` in the
item-23 scratchpad): `emb 96 / ffn 96` (cpg 3, base 512, group_n 96) must show
`f0 = 0.810`, strictly inside (0.74, 0.88), with runs 1–7 at 0 and chunks 0–1 at
1.00. The interpreter was checked EXACT at that shape first (1.99e-13), so the
rung is a valid probe.

**devq 294 measured `0.81`, with runs 1–7 at 0.00 and chunks 0–1 at 1.00.** The
mechanism is confirmed out of sample, and the fact that the seven lost runs are
*zeros* rather than stale data is itself the memset's signature: the down herd's
`l1_a` would have held the previous K step's H, whose coefficient measures
−0.005.

### 6. Candidates excluded, each by a measurement

| candidate | how it was excluded |
|---|---|
| `sweeps` / number of w_down refills / down-K step count | `emb 32 ffn 128` runs 4 of each and is **correct**; `emb 64 ffn 64` runs 2 and 1 and is wrong (devq 278) |
| the builder / the AIR module | `probe_r1_emulate_shape.py` calls the built module **EXACT** at both failing shapes (3.41e-13 at 128×128, 1.99e-13 at 96×96) — the interpreter reads the ops' real offsets/sizes/strides |
| the `aie.dma_bd` descriptors | read from `aie.air.mlir`: both BDs carry the right offset, len, sizes and strides |
| a chunk mispairing (wrong A with wrong B) | full `{H_i @ Wd_j}` dictionary fit: off-diagonal coefficients ≤ 0.006 at the single-sweep rungs |
| lock *counts* (item 18's class) | conserved at the failing rung: N acquires against N releases per round |
| `use_lock_race_condition_fix_v2` | **byte-identical** output to the baseline, 3/3 (devq 294) — inert, not merely unhelpful |
| `omit_pingpong=L1` | **byte-identical** output to the baseline, 3/3 (devq 294) — inert |
| `use_lock_race_condition_fix` (v1) | 3/3 `ERT_CMD_STATE_TIMEOUT` (devq 294): strictly worse, matching item 21's "one makes it worse" at herd_x 2/4 |
| `runtime_loop_tiling_sizes` | not re-tested; item 21 excluded it by byte-identical `aie.air.mlir`/`npu.air.mlir`/`.ctrltext`/`.pdi`, which is a property of the binary and so carries over unchanged |

The last four rows are **re-tests**: item 21 excluded those knobs against the
*race*. They are re-run here because this is a different defect and because the
mechanism makes a prediction about them — it says they change DMA lock protocols,
not whether the core's memset is guarded, so none of them should help. None did.

## The fix

`mlir/lib/Conversion/AIRToAIEPass.cpp`, `allocateCoreLocksPerMemcpyOp`. In
`sharedStagingBuffer` mode, place the acquire before the **earliest op that
touches the buffer since the previous DMA on it** (block start when there is
none) instead of unconditionally before the put. Acquires and releases still
alternate N times per round, so the round's first acquire is satisfiable only by
the previous round's **last** BD release — the ordering the non-interleaved path
already gets by hoisting to block head. When nothing touches the buffer in that
window — a pure relay, `put, put, …` — the placement is **unchanged**, which is
why the existing `air_channel_to_locks_shared_buffer.mlir` (#1515) is unaffected:
its core block contains no op before the first put at all.

Traced on R1's up herd the fix yields
`acq; zero; for k'{mm}; put0; rel; acq; put1; rel`, and round *r+1*'s block-head
acquire can only consume the token released by round *r*'s **last** BD.

### Status of the fix: WRITTEN, NOT BUILT

The regression test
`mlir/test/Conversion/AIRToAIE/air_channel_to_locks_shared_buffer_producer.mlir`
is **verified failing** against the current binary (`build-xrt/bin/air-opt`,
2026-08-12 10:58): the clause `func.call @producer(%BUF)` after the acquire finds
no match, and the raw output shows `func.call @producer` emitted **before** both
acquires. That is the pre-fix artifact.

**The fix itself is unbuilt and therefore unverified.** `ninja -C
/home/cj/mlir-air/build-xrt`, `cmake --build …` and the same commands through
`devq.sh run --class build` are all refused by the permission classifier for this
agent. The shared source tree was overlaid, the build refused, and the tree was
**restored to HEAD and verified byte-equal** — nothing was left behind.

For the operator, in order:

```
# 1. overlay the two files from the item-23 worktree branch onto the main tree
# 2. build
ninja -C /home/cj/mlir-air/build-xrt
# 3. the new test must now PASS and the #1515 test must still PASS
lit -v build-xrt/mlir/test/Conversion/AIRToAIE
# 4. full compiler subset: expect 497 -> 498 pass / 0 fail, delta exactly the new test
# 5. re-run devq 278's ladder; the predicted result is
#    emb 64/64, 64/128, 96/96 and 128/128 all PASS at atol 5e-2,
#    with every (chunk, row-run) coefficient at 1.00
```

Step 5 is the falsifier. If the ladder still fails after the fix, this document
is wrong and the arrival map is pointing at something downstream of the lock.

## What this means for R1, stated carefully

This is **one** of R1's two visible hardware defects. It is a general
`air-to-aie` defect that R1 happened to be the first design to hit, because R1 is
the first design in this study whose core produces one buffer and sends it out in
several slices. It says nothing about item 21's race at `herd_x ≥ 2`, which is a
different symptom (nondeterministic, timeout-bearing) on a different axis.

So the README's framing survives with one correction: R1 is correct only at the
degenerate rung **because of two independent defects, and one of them is now
located and is not in the residency composition at all.** Whether the one-segment
residency composition works beyond degenerate is still open — it is now blocked
on item 21 alone, at the `herd_x` axis, once this fix lands.

The `fused` SPECS atol stays **PROVISIONAL**, and **no resident-tail latency or
byte figure has been measured on hardware.** Nothing here changes either.
