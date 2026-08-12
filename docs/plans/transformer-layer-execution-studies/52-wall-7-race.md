# 52 — Wall 7 located: a memtile staging buffer with two writers and no order

`[2026-08-12]` Queue item 21. Artifacts: **devq 308** (the herd_x ladder, 60
legs), **devq 309** (the single-variable A/B, 120 legs), plus devq 306's dumps
re-analysed. Instruments:
`agents/probes/probe_aie_buffer_writer_race.py` (new),
`agents/probes/probe_r1_arrival_map.py` (extended to `herd_x >= 2`, with its own
`--self-test`). Regression: `builders/test_ffn_resident.py` clauses 9 and 10,
pinned in `run_ffn_resident_emulation_tests.lit` (8 → 10).

## The verdict in one paragraph

Wall 7 is **not** a wall in the residency composition and **not** a second
instance of item 23. R1's down feed stages every GeLU column's H chunk through
**one** L2 buffer. At `herd_x > 1` those `ChannelGet`s are `herd_x` different
sub-channels from `herd_x` different cores, so `air-to-aie` emits **one S2MM
channel per GeLU core onto one single-slot memtile buffer, every one of them
gated by the same counting semaphore with the same acquire/release counts**. A
counting semaphore has no participant identity, so the protocol fixes neither
which writer fills the slot next nor which reader consumes a given token. The
device therefore delivers an **arbitrary interleaving** of the columns' chunk
streams — every chunk whole, matched to the wrong `w_down` K step — and, when a
reader takes a peer's token instead, a **deadlock**. At `herd_x == 1` there is
exactly one writer and one reader and both hazards are vacuous, which is why the
entire `herd_x=1` ladder is clean. One staging buffer **per column** removes
both symptoms together: **35/35 dispatches PASS across five `herd_x=2` and two
`herd_x=4` rungs that were 5/35 before**, with the `herd_x=1` rungs
**byte-identical** across the two arms.

## 1. The wrong answers are permutations, not truncations

devq 306's dumps were never decomposed. Doing so is the whole finding.

The arrival map that cracked item 23 fits a per-`(chunk, row-run)` model — it can
only see a transfer arriving *truncated*. At `herd_x=2` that model **does not
fit** (residual relL1 0.66–0.84 against a 0.016 noise floor). The full
`{H_i @ Wd_j}` **pairing dictionary** fits at **0.0152–0.0158**: every chunk
arrived whole and was multiplied by another K step's `w_down`.

Read as a permutation (stream position → H chunk), across **7 wrong answers on
3 rungs** (devq 308, and rungs devq 306 never ran):

| rung | herd_x | cpg | sweeps | permutation | per-column arrival order |
|---|---|---|---|---|---|
| D2 `128×128` | 2 | 2 | 1 | `[0,2,3,1]` | c0 `[0,1]`  c1 `[2,3]` |
| D2 `128×128` | 2 | 2 | 1 | `[0,2,1,3]` | c0 `[0,1]`  c1 `[2,3]` |
| C2 `64×128`  | 2 | 1 | 2 | `[0,1,3,2]` | c0 `[0,2]`  c1 `[1,3]` |
| G2 `32×64`   | 2 | 1 | 2 | `[0,1,3,2]` | c0 `[0,2]`  c1 `[1,3]` |
| G2 `32×64`   | 2 | 1 | 2 | `[0,2,1,3]` | c0 `[0,2]`  c1 `[1,3]` |

**Every one is an interleaving of the per-column streams**: each column's own
chunks always arrive in their own order, only the merge between columns varies.
That is exactly what `herd_x` writers racing for one slot can do and exactly what
they cannot do otherwise — a column cannot overtake itself, because its chunks go
down one channel. **Both down cores report the same permutation in every case**,
which places the disorder upstream of the fan-out, at the memtile.

The interleaving clause is not free to be true: `probe_r1_arrival_map.py
--self-test` synthesises a within-column swap and requires the clause to
**refuse** it, alongside a cross-column case it must accept and an identity it
must not flag. All three pass, and the `herd_x=1` calibration is unchanged — the
probe still reproduces doc 49's published map for the item-23 rung to the
digit.

## 2. The IR says why, and the difference set is one buffer

`agents/probes/probe_aie_buffer_writer_race.py` reads an emitted `aie.air.mlir`
and reports, per buffer, the DMA channels that write and read it. On the same
design at three widths (compile only, no device):

```
herd_x=1  (64×64)   0 multi-writer buffers
herd_x=2  (64×64 and 128×128)
  %buf21 %mem_tile_3_1 L2  slots 1  writers 2  readers 2   <== RACE
     writer S2MM 1 <- %tile_2_2  acq %lock_3_1_23>=2  rel %lock_3_1_24 x2
     writer S2MM 2 <- %tile_3_2  acq %lock_3_1_23>=2  rel %lock_3_1_24 x2
     reader MM2S 0               acq %lock_3_1_24>=1  rel %lock_3_1_23 x1
     reader MM2S 1               acq %lock_3_1_24>=1  rel %lock_3_1_23 x1
herd_x=4  (64×64)   %buf37, ONE slot, FOUR writers, four readers, same shape
```

`%tile_2_2` and `%tile_3_2` are the two GeLU cores. The two writer entries are
**identical** — same lock symbols, same counts, same direction. Nothing in the
DMA program distinguishes them, so nothing orders them.

**The compiler already knows this is wrong, and already has the mechanism.**
`AIRToAIESchedulingUtils.h` documents the v2 chain-lock template as replacing
"the legacy `1 cap (init=N) + 1 done counter` template **that allows concurrent
stage firing and races on the memtile DMA**", and `isChainLockCandidate`
(`AIRToAIESchedulingUtils.cpp:650`) says of the fan-in case that "the chain-lock
is **required** here to prevent write-side corruption, so the opt-out below is
NOT honored". But its last clause is

```cpp
  // Single-writer/single-reader (legacy 1:1) or MIMO (M writers + N
  // readers) are NOT chain-lock candidates; legacy lock template applies.
  return false;
```

R1's `l2_h` at `herd_x >= 2` is **MIMO: `herd_x` writers and `herd_x` readers** —
the excluded case. It falls through to the legacy counted lock, and the very
corruption the fan-in chain lock exists to prevent is reintroduced. **This also
explains, mechanically, why `use_lock_race_condition_fix_v2` A/B'd
byte-identical to baseline five times**: the predicate returned false, so v2 was
never applied to this buffer. That exclusion was recorded as "inert"; it is
better described as "never reached".

## 3. The A/B: one variable, and both symptoms move together

`build_ffn_resident_module(..., shared_h_staging=)` selects one L2 H staging
buffer (the shipped form) or one per GeLU column. Nothing else differs — same
shapes, same compiler (`build-xrt` `aircc` sha `0651a0e5…`, provenance-gated in
leg 0), same operands, same seed. Five fresh processes per (arm, rung); **devq
309, 120 legs**.

| rung | emb×ffn | hx | cpg | swp | shared P/F/T | percol P/F/T | cross-arm y identical |
|---|---|---|---|---|---|---|---|
| B1 | 64×64   | 1 | 2 | 1 | 5/0/0 | 5/0/0 | **yes** |
| C1 | 64×128  | 1 | 2 | 2 | 5/0/0 | 5/0/0 | **yes** |
| D1 | 128×128 | 1 | 4 | 1 | 5/0/0 | 5/0/0 | **yes** |
| A2 | 32×32   | 2 | 1 | 1 | 2/0/3 | **5/0/0** | no |
| B2 | 64×64   | 2 | 1 | 1 | 3/0/2 | **5/0/0** | no |
| C2 | 64×128  | 2 | 1 | 2 | 0/0/5 | **5/0/0** | no |
| D2 | 128×128 | 2 | 2 | 1 | 0/3/2 | **5/0/0** | no |
| G2 | 32×64   | 2 | 1 | 2 | 0/2/3 | **5/0/0** | no |
| H4 | 64×64   | 4 | 1 | 1 | 0/0/5 | **5/0/0** | no |
| D4 | 128×128 | 4 | 1 | 1 | 0/0/5 | **5/0/0** | no |
| F1 | 96×96 tk16 | 1 | 6 | 1 | 0/0/5 | 0/0/5 | yes — **see §5** |
| F2 | 96×96 tk16 | 2 | 3 | 1 | 0/0/5 | 0/0/5 | yes — **see §5** |

Three things this settles.

- **Timeout and wrong answer are ONE defect.** They vanish together, at every
  `herd_x >= 2` rung, under a change that touches nothing else. 35/35 where the
  shared arm was 5/35. No separate hang mechanism is needed or supported.
- **The change moves nothing at `herd_x = 1`.** All three controls come back
  **byte-identical** across the arms — at one column, one buffer per column *is*
  one buffer. So the fix cannot be what carries the `herd_x=1` result.
- **`herd_x >= 2` now reproduces `herd_x = 1` to the bit.** `B2`'s output sha
  equals `B1`'s, `C2`'s equals `C1`'s, and `D2`'s and `D4`'s both equal `D1`'s.
  The herd width has become invisible in the answer, which is what it should
  always have been.

The shared arm also **reproduces devq 306 byte-for-byte** at `D2` — the same
three output sha prefixes (`7f7d7f62…` the sentinel, `5698c105…`, `4099fa90…`),
from an independently compiled ELF hours later.

## 4. Candidates excluded, each by a measurement

| candidate | how it was excluded |
|---|---|
| item 23's flavour (unguarded core write behind a shared *staging* buffer, needs `cpg>1`) | **four `herd_x=2` rungs with `cpg = 1` fail** — A2, B2, C2, G2 (devq 308). `cpg>1` is not necessary, so this is not item 23 |
| `chunks_per_group` as the axis | A2/B2/C2/G2 have `cpg=1` and fail; D1/C1/B1 have `cpg` 4/2/2 and pass 5/5. `cpg` neither necessary nor sufficient |
| `sweeps`, `k_steps_up`, down-K step count | vary independently across the failing set (sweeps 1 and 2, k_up 2 and 4, down_K 2 and 4) with no boundary |
| lock **counts** (item 18's class) | conserved: each writer releases N, each of N readers takes 1. Read off the emitted memtile program |
| lock **placement** (item 23's class) | every acquire dominates its own BD; the defect is *between* participants, not within one |
| `use_lock_race_condition_fix_v2` | byte-identical 5× (item 21's earlier A/B) — and now **explained**: `isChainLockCandidate` returns false for MIMO, so v2 never applied here |
| `runtime_loop_tiling_sizes`, shim issue order, task count, 6b's pacing | excluded earlier by measurement; nothing here disturbs those |
| `ctx_pc 0x28B060AD` | closed; it is the firmware's clean-timeout report site |

**The minimal failing configuration is `emb 32, ffn 32, herd_x 2, tile_k 16`** —
sweeps 1, `k_steps_up` 2, `chunks_per_group` 1, down_K 2. Two chunks, one per
GeLU column: the smallest object that has two writers at all. It fails 3/5
(devq 308) and 3/5 (devq 309's shared arm).

## 5. Two things this does NOT fix, stated plainly

**(a) The shipped gate shape does not compile with per-column staging.** At
`768×3072, herd_x 4` the extra buffers push the down-feed memtile's DMA program
over a hardware cap:

```
error: 'aie.memtile_dma' op has more than 48 blocks
```

Compile-only sweep (no device), `herd_x = 4`, `tile_k 32`:

| emb×ffn | shared | percol |
|---|---|---|
| 128×128 | OK | **OK** |
| 256×256 | OK | over 48 blocks |
| 384×384 | OK | over 48 blocks |
| 768×768 | OK | over 48 blocks |
| 768×3072 (the gate) | OK | over 48 blocks |

So the builder-side fix **proves the mechanism and unblocks the measured ladder,
but does not reach the gate**. The version that would scale is the compiler-side
one: order the `herd_x` writers on the **single** buffer with the chain lock that
already exists, which costs signal locks rather than BDs. That is a change to
`isChainLockCandidate` plus MIMO support in the chain-lock emitter, and it is the
next person's first move (§7).

**Therefore `shared_h_staging` stays `True` by default and nothing shipped
moves.** Flipping it would trade a hang for a compile refusal at the gate shape
and would turn `ffn_resident_structure.py` red — which was measured, not
assumed: with per-column as the default that probe fails with the 48-block
diagnostic, and with the default restored it is back to `PASS (1 device, 4
core->core, K-loop 4 -> 2, shim MM2S 7/16 worst column 2)`. The two arms are
spelled explicitly on the probe (`--shared-h-staging`, `--per-column-h-staging`,
mutually exclusive) so no measurement depends on which one is the default.

**(b) `96×96` at `tile_k 16` hangs at `herd_x = 1` too, and that is a different
defect.** `F1` (`herd_x=1`, cpg 6, k_up 6, down_K 6) is **5/5 TIMEOUT in both
arms with byte-identical output**, so it is untouched by wall 7 and present
without a second writer. `F2` is the same shape at `herd_x=2` and is therefore
**confounded** — it must not be read as a wall-7 datum. This rung is new: the
`21/21 herd_x=1` ladder (devq 300) and doc 49's out-of-sample `96×96` were all
`tile_k 32`. **Filed as a new open item**, not chased here.

## 6. What this costs the design, and what it corrects

The builder's own comment said the opposite of the truth and said it emphatically:

> "ORDERING across the four per-c loops … is carried by the SHARED `l2_h`
> staging buffer: c+1's first get writes `l2_h` (WAR on c's last A put), so
> air-dependency serializes the chain c0 < c1 < c2 < c3 without a hand token.
> Giving each c its own staging buffer would delete exactly that serialization —
> do not 'parallelize' it."

That AIR-level dependency **does not survive lowering**: the `herd_x` gets become
`herd_x` independent DMA channels, and the serialization the comment relies on is
expressed nowhere in the DMA program. The comment is retracted in place, with the
measurement attached. Cost of the fix at the shapes where it compiles:
`(herd_x - 1)` extra L2 chunk buffers, budgeted in `l2_need`.

**The interpreter's model (M2) is falsified in the same stroke.** It assumes an
L2 staging buffer's values "land in the order the shim issued them", which is why
it has certified R1 element-exact at all 11 ladder shapes while hardware
disagreed. Clauses 9 and 10 of `test_ffn_resident.py` pin the **discriminator**:
the per-column build must read one L2 allocation per GeLU column, and the shared
build must be **REJECTED** by the same predicate. Without clause 10 the new
clause could pass by looking at nothing. What they deliberately do **not** assert
is that the shipped default is the fixed form — it is not (§5), and a check that
claimed otherwise would be false. This keeps the discriminator from rotting while
the compiler-side fix is outstanding.

## 7. What the next person should do first

1. **Take the compiler fix, not the builder one, to the gate.** Extend
   `air::isChainLockCandidate` to the MIMO case and teach
   `getOrCreateChainLockSet` two chains (a writer chain of `nW` stages and a
   reader chain of `nR`) instead of one. The fan-in chain lock is already
   documented as *required* to prevent write-side corruption; MIMO is the same
   corruption with readers attached. Gate it on the probe: the module must go
   from 1 race to 0 with **no** change in buffer count, and `check-air-mlir` owes
   a regression test with a MIMO L2 buffer verified failing first.
2. **Then re-run devq 309's ladder plus the gate shape.** The prediction to
   record beforehand is that `percol` and `shared` become byte-identical at every
   rung once the compiler orders the writers, because the shared form would then
   be correct.
3. **Chase `F1` separately** (`96×96, tile_k 16, herd_x 1`, 5/5 TIMEOUT). It is a
   `herd_x = 1` hang, so it is neither item 21 nor item 23, and it is the only
   thing now known to fail on the `herd_x = 1` axis.
4. Do **not** re-litigate `ctx_pc`, the three race-named backend knobs, task
   count, shim order, lock counts or `runtime_loop_tiling_sizes` — §4.

`fused`'s SPECS atol stays **PROVISIONAL**, and **no resident-tail latency or
byte figure has been measured on hardware.** Nothing here changes either.
