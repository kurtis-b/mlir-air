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

---

## 8 — `[2026-08-12]` The compiler fix §7.1 asked for does not exist. Here is the proof, and what shipped instead

Queue item 22. Compiler: worktree build `build-mimo`, `aircc` sha
`9f5a52af…`, provenance-gated in every leg (the pre-fix reference is
`build-xrt`'s `0651a0e5…`, the same binary §3 used). **`build-mimo` exists
because `build-xrt`'s `CMAKE_HOME_DIRECTORY` is the shared checkout**: a
worktree agent running `ninja -C build-xrt` would have compiled the parent's
sources and produced a green-looking binary containing none of this change —
the toolchain-provenance trap one level up. Artifacts: devq **313** (compiler
build), **319** (airhost runtime), **321** and **322** (`check-air-mlir`), all
build class; the R1 arm dumps in §8.3/§8.5 are compile-only, no device, as
§5's sweep was. Instrument:
`agents/probes/probe_aie_buffer_writer_race.py --check-order` (new this
evening). Regression: `mlir/test/Conversion/AIRToAIE/
memtile_chain_lock_v2_mimo.mlir`, **verified failing pre-fix**.

### 8.1 The prediction, recorded first, and what happened to it

Recorded before building anything, per §7.2:

> Once the compiler orders the writers, `percol` and `shared` become
> byte-identical at every rung.

**That prediction was never reached, and it is reported unresolved rather than
confirmed.** It presupposes a compiler fix that orders the writers *and* binds
the readers. §8.3 measures that no such fix exists on a single slot. There was
therefore no fixed compiler to run the ladder against, and running it anyway
would have produced a number with nothing behind it. The falsifier fired one
level up — on the premise, not on the outcome.

The two side predictions were reached and both held: **buffer count does not
change** (§8.5) and **`check-air-mlir` moves by exactly the new test** (§8.6).

### 8.2 What §7.1 asked for, built and measured

`isChainLockCandidate` was extended to admit MIMO, and
`getOrCreateChainLockSet` given two chains — `nW` writer stages then `nR`
reader stages, sharing the writer→reader handoff lock, so `nW + nR - 1` signal
locks plus the cap. Ping-pong is suppressed for MIMO on purpose: the whole
claim being tested is that the writers can be ordered on the **one** buffer, and
splicing a twin would quietly turn it into the per-column fix.

It ships as `mimo-chain-lock` (pass option, `aircc` flag, and
`XRTBackend(mimo_chain_lock=)`), **default false**, and it is labelled a
falsifier arm in all three places — the same way `shared_h_staging` is kept.

### 8.3 It orders the writers. It is still wrong, and it cannot not be

The instrument is new because the old one could not answer this. `--refuse-race`
is a **shape** test: >1 writer channel on a 1-slot buffer. A shape test cannot
tell an unordered counting semaphore from a correctly chained one — both have N
writer channels on one slot — so it can neither certify a fix nor refute one.

`--check-order` simulates the emitted lock protocol instead: places are the
tile's `aie.lock`s with their declared `init`, transitions are the
`aie.dma_bd`s with their single acquire and single release advancing a
per-channel program counter along `aie.next_bd`, streams are modelled
always-ready (the conservative reading of *does the DMA program alone force
this*). It explores the reachable state space and decides two things
separately: is the k-th writer the same channel on every interleaving, and is
every fill consumed by every reader before the next lands.

Calibration, both directions, before use: the existing v2 **fan-in** chain
(`memtile_chain_lock_v2_fanin.mlir`, 4 writers) reads **ORDERED**; R1's shared
`l2_h` reads **RACE**. Neither is assumed.

On R1 D2 (`128×128, herd_x 2`), one line per arm:

| arm | writer order | read binding |
|---|---|---|
| shipped default (v2 off) | **RACE** — write #0 by either channel | — |
| `use-lock-race-condition-fix-v2` | **refused** (§8.4) | — |
| `+ mimo-chain-lock` | **ORDERED** — `S2MM 1, 0, 1, 0, …`, one channel per position | **OVERWRITE** — fill #0 destroyed before *either* reader read it |

The two-chain form delivers exactly half the constraint and loses the other
half. That is not an implementation slip, it is the budget:

> **An AIE2 BD descriptor carries one acquire-lock field and one release-lock
> field.** `generateDmaBd` emits exactly one of each. So a writer's single
> release goes **either** to the next writer (ordering the writers, leaving the
> readers unsignalled) **or** to the readers (binding the readers, leaving the
> writers on a shared counting semaphore — which is the legacy template, and is
> wall 7). There is no third place to put it.

The remaining freedom is asymmetric acquire/release *counts*, and a counting
argument closes it. Take the minimal case — 2 writers, 2 readers, one slot,
required cycle `w0, {r0,r1}, w1, {r0,r1}`. Both writers must be gated on
something the readers release (else a writer can fire before the reads), so
both readers must feed each writer's input place; two readers releasing into
two *different* places lets one writer fire on one reader's release alone, so
the writers share one input place `G`, fed `2ρ` per fill. With `a₀ + a₁ = 4ρ`
per period, "w1 enabled after the first read pair" gives `X ≥ 2ρ` and "w0 not
re-enabled at that same moment" gives `X < 2a₀ − 2ρ`, forcing `a₀ > 2ρ`; while
"w1 disabled at period start" with `X ≥ 2ρ` forces `a₀ < 2ρ`. Contradiction,
independent of the constants. The same argument with a writer run length of
`k ≥ 2` (R1's `chunks_per_group`) contradicts identically.

**The escape is not a cleverer lock assignment — it is more BDs.** The reader
chains must distinguish all `P = herd_x × chunks_per_group` phases, one BD per
fill. Which is measurable and measured: R1 D2's reader chain is **1** `h`-BD in
the shared arm and **4** in the per-column arm, and `P` = 2 × 2 = 4.

### 8.4 So v2 refuses MIMO by name instead of silently racing

`isChainLockCandidate` still excludes MIMO, and that exclusion is now
documented as load-bearing rather than as an oversight. What changed is the
fall-through: under `use-lock-race-condition-fix-v2` a MIMO memtile buffer now
**errors**, naming the buffer, the counts, and the one-acquire/one-release
reason. v2 is the "do not race on the memtile DMA" mode; silently emitting the
racing legacy template was the worst available behaviour, and it is precisely
why v2 read as **inert** in five A/Bs when it had simply never been reached
(§2). A diagnostic on that path is what would have found wall 7 on day 1.

### 8.5 Nothing shipped moves, and that is checked rather than argued

The default path (v2 off) is untouched. R1 D2's emitted `aie.air.mlir` is
**byte-identical** across the pre-fix and post-fix compilers — sha256
`5439c51d…` from both `0651a0e5…` and `9f5a52af…`. Buffer counts:

| arm | `aie.buffer` total | L2 | H staging on the down-feed memtile |
|---|---|---|---|
| shared, pre-fix | 22 | 6 | **1** |
| shared, post-fix default | 22 | 6 | **1** |
| `mimo-chain-lock` | 23 | 7 | **1** |
| per-column (the builder fix) | 25 | 9 | **4** |

The chain arm's `+1` is v2's ping-pong twin on the *`w_down`* fan-out buffer,
a different buffer. So the compiler arm does keep the H staging at one buffer —
the property that would have let it reach the gate. It simply is not correct.

### 8.6 Gate

`check-air-mlir` **512 discovered / 498 pass / 0 fail** before,
**513 / 499 / 0** after; the delta is exactly
`memtile_chain_lock_v2_mimo.mlir` and no other test's expectations were
touched. That test pins three things and was **verified failing first**: with
the pre-fix binary its refusal line exits 0 (the module compiles silently,
which is the defect) and its `mimo-chain-lock` line fails on an unknown option.

### 8.7 What this means for §5 and §7, stated plainly

§5 and §7.1 both assert that the compiler-side fix is "the version that would
scale", costing signal locks rather than BDs. **That is retracted.** Any
correct scheme must distinguish `P = herd_x × chunks_per_group` phases in the
reader BD chains, which is the same BD cost as per-column staging — measured
equal at D2 (4 `h`-BDs either way). At the gate shape (`768×3072, herd_x 4`:
`group_n` 192, `cpg` 6, `sweeps` 4) `P` = 24, so a correct reader chain is 24
`h`-BDs + 24 `b`-BDs per MM2S channel across 4 channels, against a 48-block cap
for the whole `memtile_dma`. **Both fixes hit the same wall, and the compiler
one hits it no later.** That is why §5's per-column sweep starts failing at
`256×256`, and it is not a property of per-column staging.

There is a second, independent reason the two-chain form is wrong at the gate
even ignoring the read side: its writer order is strict **round-robin**
(measured: `S2MM 1, 0, 1, 0, …`), while R1 needs run length `cpg` — at D2 the
correct order is `c0, c0, c1, c1`, which the per-column arm's BD chain shows
directly (reader chain `buf19, buf18` from one column then `buf17, buf16` from
the other). Round-robin happens to coincide with the requirement only at
`cpg = 1`, which is 6 of the 7 failing rungs and **not** the gate.

**`shared_h_staging` therefore stays `True`, and cannot yet become the
default in either direction.** Per-column remains correct and compiles to
`128×128` at `herd_x 4` (§5); above that neither arm both compiles and is
correct. Reaching the gate needs the *period* reduced, not the locks changed —
e.g. staging a whole column group per transfer so `P` falls from
`herd_x × cpg` to `herd_x` — which is a builder change and is not attempted
here.

### 8.8 A lead this instrument turned up, not chased

`--check-order` reports **OVERWRITE** on `%buf20`/`%buf24` — the `w_down`
down-feed buffer, 1 writer, 2 readers, one slot, legacy counted template — in
**both** arms and at `herd_x = 1`, where the read side is unbound for the same
one-release reason. It is present in the arm that is 35/35 on hardware, so it
is a latent hazard the DMA program does not exclude rather than a fired one,
and the probe's epistemic status is the same as §2's ("no ordering mechanism",
not "reachable"). It is worth pointing at **F1** (§5b: `96×96, tile_k 16,
herd_x 1`, 5/5 TIMEOUT in both arms, the only known `herd_x = 1` failure) —
a `herd_x = 1` hang whose only unbound buffer is this one. Not investigated
here; filed as a lead with its instrument attached. Note that v2's fan-out
chain **does** fix this buffer (arm C reads ORDERED on `%buf20`), so there is a
cheap experiment available.

---

## 9 — `[2026-08-12]` Rows 28 and 30 are NOT the same object. Row 28 is `down_K >= 5`, and §8.8's reason for pairing them was false

Queue rows 28 and 30. Compiler: `install-xrt`, `aircc` sha256 `b6e3de13…`
(mtime 16:59:35), printed from python in every job's leg 0 and **refused** on
mismatch — `air.tools.resolve_tool` prefers the bundled binary, so PATH does not
decide this. Artifacts: devq **327** (the A/B + localizing ladder, 40 legs),
**328** (bracket, 20), **329** (separating controls, 25), **330**
(out-of-sample, 30), **331** (the tiling arm, 25) — 140 device legs, one fresh
process each, all measure class. Compile-only sweeps are off-queue, as §5's and
§8.3's were. Instruments: `probe_aie_buffer_writer_race.py --check-order` (two
defects found in it, both fixed), `probe_r1_rung.py --dump-npz`,
`probe_r1_emulate_shape.py`.

### 9.1 The premise §8.8 rested on is false

§8.8 pointed row 30 at row 28 on one clause: the `w_down` feed's OVERWRITE holds
"in both wall-7 arms **and at `herd_x = 1`**". That clause is the whole reason
the two rows were candidates for one object, and it is **wrong**.

Row 30's buffer is 1 writer / **2 readers** / one slot. The reader count is the
number of **down cores**, and the down herd is `herd_x` wide. At `herd_x = 1`
there is one down core, so the same buffer is 1 writer / 1 reader — the legacy
1:1 shape, with no read-binding hazard to have.

Measured compile-only across **nine** `herd_x = 1` modules (row 28's own rung,
doc 49's out-of-sample rung, four ladder rungs, `sweeps` 1/2/4, both `tile_k`,
**both** H-staging arms): every L2 buffer is **1 writer, 1 reader**, and
`--check-order` reads **0 not provably sound** on all nine. Row 30's shape does
not occur once.

The instrument is not silently broken there: on the same afternoon's `D2` module
it still reads `RACE` on `%buf21` and `OVERWRITE` on `%buf20`, and that module's
`aie.air.mlir` sha256 is `5439c51d…` — **byte-equal to the one §8.5 recorded**.
Same object, re-measured.

**Row 28's module does not contain row 30's buffer shape.** The hypothesis is
refused at the premise, one level above the experiment, exactly as §8.1's was.

### 9.2 The arm, and why it needed no isolation

The brief's complication was row 29: v2 now **refuses** a MIMO memtile buffer by
name, so enabling it on R1 might refuse rather than run. It does not, and the
reason is structural rather than lucky. `isUnorderableMimoMemtileBuffer` is
`nW > 1 && nR > 1`. At `herd_x = 1` **no** L2 buffer has a second writer or a
second reader (§9.1), so nothing is MIMO, nothing is even a chain-lock
candidate, and the refusal cannot fire. No isolation, no `air.no_chain_lock`
pin, no synthetic probe module: **row 28's rung already sits on the axis where
v2's refusal is out of scope.**

That is asserted with a positive control rather than by reading the predicate.
The same design at `herd_x = 2`, in the same job, compiles to

```
error: 'aie.buffer' op v2 chain-lock: memtile buffer has 2 writers and 2 readers
(MIMO) on a single slot, which no per-BD lock protocol can order …
```

so the refusal **is** live in this binary. A refusal at `herd_x 2` beside a clean
compile at `herd_x 1` is the discriminator, and it doubles as a stronger
provenance check than the sha alone: it proves the compiler under test carries
this afternoon's change.

### 9.3 The A/B is a null, by construction and by measurement

At `herd_x = 1` v2 has nothing to apply to, so it should emit the same module.
It does: `aie.air.mlir` sha256 **`a1b66f22c8579595` from both arms**, and
`--check-order` reports an identical per-buffer table. Five fresh processes per
arm, devq 327:

| arm | rung | P/F/T | distinct `y` sha256 | `y` sentinel | cores finished |
|---|---|---|---|---|---|
| A — shipped (v2 off) | `96×96 tk16 hx1` | **0/0/5** | 1 — `17c9c1bb` | 1.0000 | 0/1 |
| C — `use_lock_race_condition_fix_v2` | same | **0/0/5** | 1 — `17c9c1bb` | 1.0000 | 0/1 |

Zero spread: 10 processes, one output sha, and that sha is **`17c9c1bb`, equal
to devq 309's `F1` across both of its arms (10/10)** — reproduced today on an
independently compiled ELF, the same move §3 made against devq 306.

**Verdict on the briefed experiment: it cannot settle the hypothesis, and it is
reported as a null rather than as evidence.** v2's fan-out chain does fix
`%buf20` — at `herd_x >= 2`, where `%buf20` has two readers. Row 28 is at
`herd_x = 1`.

### 9.4 What row 28 actually is: `down_K >= 5`, and the headline named the wrong axis

Row 28 is filed as "`96×96` at `tile_k 16`". It is neither.

At `herd_x = 1` with `emb == ffn`, `chunks_per_group`, `k_steps_up` and the down
herd's K-step count are all `emb/tile_k` — one number wearing three names, which
is why the rung looked like a `tile_k` story. Two moves separate them: `ffn >
emb` raises `down_K` and `sweeps` while holding `cpg` and `k_up`; and `herd_x 2`
with **per-column** staging (item 21's measured fix, 35/35 — the shared arm would
confound every leg) makes `k_up = 2·cpg`.

Over **32** rungs — devq 327/328/329/330/331 plus devq 309's recorded arms, whose
geometry was re-derived from 309's own `argv` and **not** from §3's table (§3's
`C2` is `tile_k 32`; reading it as 16 cost this investigation an hour and one
wrong model) — exactly one quantity separates the set:

```
down_K  =  ffn_dim / tile_k  =  chunks_per_group  x  herd_x  x  sweeps
```

| `down_K` | verdict, 5 fresh processes each | rungs |
|---|---|---|
| 2, 3, 4 | **PASS** | T2 T3 T4 K3 K4 O1 O5 B1s C1s D1s A2p B2p C2p D2p G2p H4p D4p |
| **5** | **FAIL** — byte-deterministic wrong answer, one sha, ~50% of elements | T5 K5 O2 |
| 6, 7, 8, 9, 12 | **TIMEOUT** — `y` sentinel **1.0000**, 0 cores finished | A T7 T8 K6 O3 O4 O6 N1 N2 N3 N4 N5 |

`down_K` is the **product** of the three counts that were each tested separately
and each excluded, which is why every single-count model dies:

- **`tile_k` is excluded.** Every `down_K` measured at both `tile_k` gives the
  same verdict — 3 PASS/PASS, 4 PASS/PASS, 5 FAIL/FAIL, 6 TIMEOUT/TIMEOUT. `K6`
  is `192×192` at **`tile_k 32`** and hangs 5/5.
- **`chunks_per_group` is excluded.** `K3` has `cpg 3` and passes; `N4` has
  `cpg 3` and hangs; `O3` has `cpg 1` and hangs.
- **`k_steps_up` is excluded.** `K3` has `k_up 3` and passes; `N2` has `k_up 3`
  and hangs. `N3` carries `T8`'s exact `k_up 8` at `cpg 4` and hangs anyway.
- **`herd_x` is excluded.** It fires at 1 and at 2 (`N3 N4 N5 O4`).
- **The H-staging arm is excluded.** It fires in both (`N3 N4 N5 O4` are
  per-column).
- **The host DMA task count is excluded.** `H4p`/`D4p` issue 14 tasks and pass;
  `N4` issues 12 and hangs.
- **`runtime_loop_tiling_sizes` is excluded BY MEASUREMENT** (devq 331), not by
  the inheritance §4 recorded. `--tiling none` leaves `A`, `K6`, `O3` and `T8`
  at 5/5 TIMEOUT with their sha unchanged, and leaves `T4` at 5/5 PASS
  **byte-identical** (`69ad2530`) to its `2,2` twin. `--tiling 4,4` is inert one
  level earlier: its `npu.air.mlir` is byte-identical to `2,2`'s (`4d3cce96…`).

**Recorded before the measurement and confirmed 6 for 6** (devq 330, prediction
file sha `d5be991a…` printed in its leg 0). `O5/O2/O3/O6` are `emb 32,
tile_k 32, herd_x 1`: `cpg 1`, `k_steps_up 1`, `group_n 32` — the most degenerate
per-step geometry the builder admits. **Only `sweeps` moves.**

| rung | `sweeps` | `down_K` | predicted | measured |
|---|---|---|---|---|
| O5 `32×128` | 4 | 4 | PASS 5/5 | **PASS 5/5** |
| O2 `32×160` | 5 | 5 | FAIL 5/5, one sha | **FAIL 5/5, one sha** |
| O3 `32×192` | 6 | 6 | TIMEOUT 5/5, sentinel 1.0 | **TIMEOUT 5/5, sentinel 1.0** |
| O6 `32×224` | 7 | 7 | TIMEOUT 5/5 | **TIMEOUT 5/5** |
| O1 `16×64`  | 4 | 4 | PASS 5/5 | **PASS 5/5** |
| O4 `96×96` hx2 percol | 1 | 6 | TIMEOUT 5/5 | **TIMEOUT 5/5** |

`O5` **is doc 49 §2's own `emb 32 / ffn 128` rung**, from which it concluded
"**`sweeps` is excluded by measurement**". **That exclusion is retracted**:
`sweeps` was never taken above 4. At 5 the same family gives a deterministic
wrong answer and at 6 it hangs, with every other count pinned at 1.

The two earlier models are recorded as **falsified**, not quietly dropped.
`PREDICTION-SEP.md` predicted `cpg` as the axis and put N1–N4 at PASS; all four
timed out. Before that, `--check-order`'s DEADLOCK verdict was taken as a
predictor and predicted TIMEOUT at `down_K 5`; the measurement is FAIL, and §9.7
is why.

Every rung is a valid probe: `probe_r1_emulate_shape.py` calls the built module
**EXACT** at `96×96 tk16` (2.27e-13), `192×192 tk32` (5.12e-13), `96×96 tk32`,
`64×64 tk16`, `128×128 tk32` and `80×80 tk16`. The builder and the AIR module
are right at all of them; the defect is below AIR.

### 9.5 So: same buffer, different hazard

`down_K` is not an arbitrary count. It is **the number of times the `w_down`
feed's L2 buffer is filled and drained** — `l2_b_down`, which is `%buf20` /
`%buf24`: **row 30's buffer.** The two rows land on the same feed.

They are still **different objects**, and this is the part to keep:

- Row 30's hazard is **reader binding**, and it needs `>= 2` readers, i.e.
  `herd_x >= 2`. Row 28 fires at `herd_x = 1`, where that buffer has one reader
  and `--check-order` reads ORDERED.
- Row 30's hazard is **present in rungs that pass 5/5** — `D2p`, `H4p`, `D4p`
  are `herd_x` 2 and 4 with `down_K 4`. So it stays *latent, not shown
  reachable*, exactly as §8.8 filed it.
- The one mechanism that would have connected them (v2's fan-out chain) is
  **inert** on row 28's rung, measured (§9.3).

**Fixing row 30 would not fix row 28.** But row 28 does raise the value of that
buffer: it is now the site of a second, independent, *demonstrated* defect, where
before it carried only a latent one.

### 9.6 How far row 28 is localized, and what is still open

Localized to: **the `w_down` feed path, at a refill count above 4**, below AIR
(the interpreter is exact), independent of `tile_k`, `herd_x`, `group_n`, `cpg`,
`k_steps_up`, the H-staging arm, the host task count, `runtime_loop_tiling_sizes`
and both lock-race knobs.

The sentinel read-back localizes the hang further: **`y` sentinel fraction is
exactly 1.0000 and `cores_finished` is 0 on every timing-out rung**, so no down
core ever reached its hoisted C store — the design stalls upstream of the last
write rather than partway through it. `ctx_health` is the firmware constant
(`ctx_pc=0x28b060ad`, `txn_op_idx=0xffffffff`) on all of them, as §4 closed.

**Not established, and the honest gap**: *why* 4. A threshold at exactly 4 wants
a capacity of 4, and the obvious candidate — `runtime_loop_tiling_sizes = [2,2]`,
`2 x 2 = 4` — is now excluded by measurement. The next instrument is not another
shape sweep: it is a per-BD reading of the `w_down` feed's lock protocol as a
function of `down_K` on the down-feed memtile, comparing `down_K` 4 against 5 and
6 on modules that are otherwise identical. `O5`/`O2`/`O3` are exactly that set —
same `emb`, same `tile_k`, same `herd_x`, differing only in `ffn_dim`.

**This is a compiler-side diagnosis and NO COMPILER FILE WAS TOUCHED here.** It
owes `check-air-mlir`, the transformer-layer suite and the ten-model leg when
someone takes it, which is why it stops at the diagnosis.

### 9.7 Two defects in `--check-order`, both fixed, both in the same direction

The instrument was reporting hazards that are not there, twice.

1. **A dropped op.** `_RE_DMA_START_FULL` required `)` immediately after the
   second block label, so it **silently dropped**
   `aie.dma_start(MM2S, 0, ^bb1, ^bb6, repeat_count = 1)`. `air-to-aie` emits
   that prologue + steady-state pair at odd `chunks_per_group >= 5`. The
   simulator therefore ran a **2-BD** steady-state chain where the emitted
   channel has **6**, wedged, and reported `DEADLOCK` on `T5`, `T7` and `K5` —
   whose measured hardware failure is a byte-deterministic **wrong answer**, not
   a hang. A dropped op is the worst shape for a checker: a confident verdict
   about a program that was never emitted.
2. **A missing participant.** The net models the DMA program and nothing else.
   On a memtile that is the whole protocol (measured on R1: a memtile's locks are
   touched by **0** core ops). On an L1 tile it is not — `aie.core` holds **4 of
   4 and 6 of 6** of every compute tile's DMA locks. The missing releases
   produced a spurious `DEADLOCK` on R1's `%buf7`, on a flag
   `--refuse-unordered` would have gated a build on.

Both now read **`UNMODELLED`**, which is not a hazard verdict and does not gate.
The decision lives in `order_verdict()`, which `--self-test` calls, so each guard
is **calibrated in both directions** — a must-fire case and a must-not-fire case:
self-test **3 cases -> 6**.

**Nothing recorded moves.** §8.3's calibration re-verified after the change: the
v2 fan-in chain reads ORDERED, the v2 fan-out chain reads ORDERED, and R1's `D2`
still reads `RACE` on `%buf21` and `OVERWRITE` on `%buf20`. After the fix,
`--check-order` reports **0 hazards on all 17 `herd_x = 1` rungs measured today,
6 of which fail on hardware** — so the instrument is *silent* about row 28's
defect. That is the honest position, and strictly better than the previous
confident-and-wrong `DEADLOCK`.

### 9.8 What the next person should do first

1. **Read the `w_down` feed's lock protocol against `down_K`**, per BD, on the
   down-feed memtile, over `O5` (`down_K 4`, passes), `O2` (5, wrong answer) and
   `O3` (6, hangs) — three modules differing only in `ffn_dim`. The wrong answer
   at exactly 5 is the most informative rung in the set and it is
   *byte-deterministic*, so it decomposes the way item 23's did
   (`probe_r1_arrival_map.py` with `--dump-npz` beside it).
2. **Do not re-litigate** `tile_k`, `chunks_per_group`, `k_steps_up`, `sweeps` in
   isolation, `herd_x`, the H-staging arm, the host task count,
   `runtime_loop_tiling_sizes`, or either lock-race knob — §9.4, each excluded by
   a measurement here.
3. **Row 30 stays open and stays latent.** It is on the same buffer, which makes
   it more interesting, but nothing here shows it reachable and `D2p`/`H4p`/`D4p`
   carry it while passing 5/5.
4. **`sweeps` is no longer excluded** anywhere it appears as an exclusion (doc 49
   §2, and §4's table by inheritance). It was only ever tested to 4.

`fused`'s SPECS atol stays **PROVISIONAL**, and **no resident-tail latency or
byte figure has been measured on hardware.** Nothing here changes either.

---

## 10 — `[2026-08-12]` Row 28 is **two** defects. The `down_K = 5` wrong answer is an L2 slot-rotation **phase skew** in `air-to-aie`, proved causal on hardware; the `down_K >= 6` hang is a **different** mechanism and survives removing it

Queue row 28, "why 4". Compiler: `install-xrt`, `aircc` sha256 `b6e3de13…`
(mtime 16:59:35), printed from python in leg 0 of every job and **refused** on
mismatch. Artifacts: devq **332** (the `--omit-pingpong L2` arm, 20 device legs,
5 fresh processes per rung, expected leg count printed and counted), plus devq
327–331's existing dumps re-analysed and their 21 compiled modules re-read.
Compile-only sweeps are off-queue, as §5's, §8.3's and §9.1's were. Instruments:
`probe_r1_arrival_map.py` (unchanged), `probe_r1_rung.py --dump-npz`
(unchanged), and `probe_aie_buffer_writer_race.py` **`--check-rotation`** (new
here, self-test **6 cases -> 11**).

§9.8 asked for a per-BD read of the feed's lock protocol over `O5`/`O2`/`O3`.
That read is below. It found something the lock protocol cannot express: the
locks are **sound** and the **order** is wrong.

### 10.1 The `down_K = 5` wrong answer is a pure permutation

`O2` (`32x160`, `tile_k 32`, `herd_x 1`), devq 330's dumps, decomposed for the
first time:

| model | residual relL1 |
|---|---|
| per-`(chunk, row-run)` **arrival** (item 23's model) | **0.7780** — does not fit |
| full `{H_i @ Wd_j}` **pairing** dictionary | **0.0165** — the bf16 noise floor |

Every chunk arrives **whole** and is multiplied by another K step's `w_down`.
Read as a permutation, `position -> H chunk`:

```
O2  down_K 5   sigma = [0, 1, 4, 2, 3]      identical in r1 and r3
```

It is **not** an interleaving — at `herd_x = 1` there is one producing column,
so wall 7's writer race cannot produce it, and the probe says so by name. The
`down_K = 4` controls `O5`, `O1` and `K4` all read the identity at residual
0.0158, and `T3` (`down_K 3`) reads the identity too.

### 10.2 The permutation is predicted, exactly, by the emitted BD chains

The feed's L2 buffer is **multi-buffered**: `air-to-aie` emits `S` distinct
`aie.buffer`s and rotates them. Every slot is **1 writer / 1 reader**, every
acquire dominates its own BD, lock counts are conserved — `--check-order` reads
the whole module clean, correctly. What is expressed nowhere is the **phase**:
the order in which the consumer's BD chain visits the slots against the order
the producer's chain fills them.

Deciding it needs no timing model. With one writer and one reader on a slot the
lock pair forces the n-th read of slot `b` to see the n-th write of `b` on
**every** interleaving, so the delivered item of the p-th consumer firing is a
function of the two chains alone. Reading them off `aie.air.mlir`:

| rung | `down_K` | slots | consumer's BD program | delivered |
|---|---|---|---|---|
| `O5` | 4 | 2 | one circular chain | `[0,1,2,3]` **IN-STEP** |
| `O2` | 5 | 3 | `repeat_count = 1` prologue (2 slots) **+** a 1-slot tail | `[0,1,3,4,2]` **SKEWED** |
| `O3` | 6 | 2 | one circular chain | `[0,1,2,3,4,5]` **IN-STEP** |
| `O6` | 7 | 3 | `repeat_count = 2` prologue **+** tail | `[0,1,3,4,6,7,2]` **STARVED** |

`repeat_count = 0` means "do it once and do not repeat" (`AIEOps.td`), so a
`repeat_count = 1` chain executes **twice** — read from the dialect definition,
not assumed.

The same `d = [0,1,3,4,2]` holds on **both** memtiles: `%mem_tile_0_1` (the up
feed, `hidden` + `w_up`) and `%mem_tile_1_1` (the down feed, `H` + `w_down`).
The two compose, and **which** composition applies is fixed by the geometry:

- **`sweeps`-driven** (`O2`: cpg 1, k_up 1, sweeps 5). The up feed's skew selects
  which `w_up` chunk produces the t-th H chunk, so H itself comes out permuted;
  the down feed then permutes again. `sigma = d[d[p]] = [0,1,4,2,3]`.
- **`cpg`-driven** (`T5`/`K5`: cpg 5, k_up 5, sweeps 1). The up herd's five
  k-steps **accumulate into one L1 accumulator**, and `A_k` and `Wup_k` are
  skewed by the *same* `d`, so the pairs stay matched and the sum is
  order-independent. The up-feed skew is **invisible**. Only the down feed's
  applies: `sigma = d = [0,1,3,4,2]`.

**Both arms measured, on shapes with mirror-image geometry:**

| rung | `emb x ffn` | `tile_k` | cpg | k_up | sweeps | predicted | measured |
|---|---|---|---|---|---|---|---|
| `O2` | 32x160 | 32 | 1 | 1 | 5 | `[0,1,4,2,3]` | **`[0,1,4,2,3]`** |
| `T5` | 80x80 | 16 | 5 | 5 | 1 | `[0,1,3,4,2]` | **`[0,1,3,4,2]`** |
| `K5` | 160x160 | 32 | 5 | 5 | 1 | `[0,1,3,4,2]` | **`[0,1,3,4,2]`** |

Zero free parameters: `d` is read off the emitted chains, and the composition
rule is forced by commutativity of the k-reduction.

**The recorded prediction was half wrong and is reported as such.**
`PREDICTION-ROT.md` (sha `0b18a1c0…`, written before T5/K5 were touched) said
all three would read `[0,1,4,2,3]`. Clauses 2–4 held (the pairing model fits at
0.0161/0.0164 against arrival residuals of 0.75/0.77; not an interleaving; one
map per rung across replicates). **Clause 1 was falsified for T5 and K5.** The
repair — that a k-reduction absorbs the up-feed skew — is forced rather than
fitted, but it was made *after* seeing the data, and that is exactly why §10.5
exists: the model was re-tested causally instead of declared confirmed.

### 10.3 Where the compiler does it

`mlir/lib/Conversion/AIRToAIESchedulingUtils.cpp`, `air::getRepeatCounts`. Its
`detectNBufferRotation` admits a rotation only when

```cpp
  // Valid rotation: multiple unique buffers, total ops divisible by buffer
  // count
  unsigned numBuffers = uniqueBuffers.size();
  return numBuffers >= 2 && ops.size() % numBuffers == 0;
```

When that holds, every op is given repeat count 0, `generateDmaBdProgram`
(`AIRToAIEPass.cpp:6440`) sees `repeat_counts.size() == 1`, sets
`infiniteBDLoopMode`, and emits **one circular chain** — correct at any length.
When it does not hold, control falls through to per-op trip-count grouping,
`infiniteBDLoopMode` goes false, and each group becomes a **separately
terminated task**. Those tasks replay a *prefix* of the rotation `k` times and
then the remainder, while the producer keeps rotating over all `S` slots. **The
divisibility guard correctly detects that this is not a clean rotation; the
fallback it guards is not order-preserving, and nothing downstream checks it.**

A compiler defect, general to any `air.channel` whose multi-buffered L2 staging
has a fill count that is not a multiple of the slot count. R1 is only the first
design here to sit on one.

### 10.4 The rotation reading separates all 21 compiled rungs

`--check-rotation`, compile-only, against devq 327–331's recorded verdicts:

| `down_K` | rungs | rotation phase | verdict |
|---|---|---|---|
| 2, 3, 4 | T2 K3 T3 K4 O1 O5 T4 | **IN-STEP** | PASS |
| **5** | K5 O2 T5 | **SKEWED** `[0,1,3,4,2]` | **FAIL** |
| 6 | A K6 N4 O3 O4 | **IN-STEP** | TIMEOUT |
| **7** | O6 T7 | **STARVED** (wants item 7 of a 7-item stream) | TIMEOUT |
| 8, 9, 12 | N3 T8 N2 N1 N5 | **IN-STEP** | TIMEOUT |

Two things fall straight out. The skew is **not** simply "odd": `down_K` 3 and 9
are odd, divide their slot count, and are IN-STEP. And **the `>= 6` hangs at even
`down_K` are IN-STEP**, so the rotation cannot be what hangs them.

### 10.5 The causal test: `--omit-pingpong L2`, devq 332

Every rung measured so far confounds the two candidates, because in this builder
`k_steps_up = emb//tile_k` and `chunks_per_group = (emb//herd_x)//tile_k`, so

```
maxq  ==  sweeps * k_steps_up  ==  chunks_per_group * herd_x * sweeps  ==  down_K
```

**identically** — no shape can move one without the other. That is why §9 could
not get past "why 4" by sweeping shapes, and it is a property of the builder's
geometry, not of the measurement.

`--omit-pingpong L2` breaks the tie: it removes the L2 multi-buffering, so a
one-slot buffer has **no phase to get wrong** and the skew becomes vacuous,
while the runtime sequence is untouched. **PREDICTION-PP.md** (sha
`e7bc2dd4…`) was written before any compile in this arm, with a compile-only
gate that had to pass first. Both gate clauses passed:

- `--check-rotation` reports **no multi-slot rotation** on all four L2-arm
  modules, where the baseline reports SKEWED / STARVED / IN-STEP;
- `maxq` is **unchanged** at 4 / 5 / 6 / 7.

| rung | `down_K` | baseline | rotation | **predicted** | **measured** |
|---|---|---|---|---|---|
| `O5` | 4 | PASS 5/5 | IN-STEP | PASS 5/5 | **PASS 5/5** |
| `O2` | 5 | **FAIL 5/5**, `sigma [0,1,4,2,3]` | SKEWED | **PASS 5/5** | **PASS 5/5** |
| `O3` | 6 | TIMEOUT 5/5 | IN-STEP | TIMEOUT 5/5 | **TIMEOUT 5/5** |
| `O6` | 7 | TIMEOUT 5/5 | STARVED | TIMEOUT 5/5 | **TIMEOUT 5/5** |

**4 for 4.** One distinct `y` sha256 per rung across 5 fresh processes
(`O5pp 75205755`, `O2pp ece5f178`); `O2` goes from ~50% of elements wrong to
**0/2048 mismatches, corr 0.999871, abs_err_max 4.9e-3** against an atol of
5e-2. The two timing-out rungs are unmoved down to their signature: `y` sentinel
**1.0000**, `cores finished 0/1`, exactly as the baseline.

So: **the slot-rotation phase skew is the whole cause of the `down_K = 5` wrong
answer, and it is not the cause of the `>= 6` hang.** Row 28 is two defects.

### 10.6 The `>= 6` hang: bounded, named, not concluded

`O5` and `O3` differ **only in trip counts** — both `aie.air.mlir` are 572 lines
and the diff is `%c4 -> %c6`, `memref<4096xbf16> -> memref<6144xbf16>` and the
matching loop bounds. Same tiles, buffers, locks, BD chains and flows. So the
hang cannot be a structural property of the emitted device program. What does
scale is the **runtime sequence**:

```
push @air_channel_1 (y out, issue_token)      shim_noc_2_0 S2MM 0
push @air_channel_0                           shim_noc_2_0 MM2S 0
push @air_channel_2  x down_K                 shim_noc_0_0 MM2S 0   <== the hidden refill
push @air_channel_3  (w_up,   ONE task)       shim_noc_0_0 MM2S 1
push @air_channel_4  (w_down, ONE task)       shim_noc_1_0 MM2S 0
await @air_channel_1
```

`down_K` separate `aiex.dma_start_task`s land on **one** hardware channel with
no `dma_await_task` between them, and **before** the `w_up` and `w_down` pushes
that the consumers need in order to drain them. `aiex.npu.push_queue` is a bare
register write to that channel's Start_Queue; the compiler does no
queue-occupancy accounting anywhere. AIE2 has a per-channel hardware task queue
with a sticky "attempt to write to full task queue" error bit, so an
over-subscribed channel is a real failure mode rather than a hypothetical one.

Call it `maxq` — outstanding starts on the busiest channel before the first
await, read off `npu.air.mlir`. Measured on all 21 compiled rungs:

| | `maxq` values |
|---|---|
| PASS | **2, 3, 4** |
| FAIL | **5** |
| TIMEOUT | **6, 7, 8, 9, 12** |

with `maxq == down_K` on every one, on the same channel
(`%shim_noc_tile_0_0 / MM2S 0`) in every one.

Excluded, each against the artifact:

- **BD length.** `K4` passes with a **16384**-element single `w_up` BD; `O3` hangs
  with **6144**. Not the length.
- **Total task count.** §9.4's exclusion stands and is now explained: `H4p`/`D4p`
  issue 14 tasks and pass, `N4` issues 12 and hangs — their `maxq` is 4 and 6.
  It is per-channel occupancy, not the total.
- **`runtime_loop_tiling_sizes`.** `--tiling none` and `--tiling 4,4` leave
  `maxq` **unchanged** at every rung (T4/A/K6/O3/T8), as they leave the verdicts
  unchanged (devq 331).
- **The L2 rotation.** devq 332, §10.5.

**This is bounded, not concluded.** What is measured is that `maxq` separates
every rung and that everything else scaling with `down_K` in the artifact is
excluded. What is **not** established is the failure mode (a blocked control
stream versus dropped pushes) or the queue's depth. **No constant is claimed**:
the observation is only that 5 outstanding starts complete and 6 do not, on this
channel, in this push order, and that number is *not* asserted to be a
documented queue depth. Naming one without measuring it is exactly the move this
project has been burned by twice.

### 10.7 The instrument

`probe_aie_buffer_writer_race.py --check-rotation [--stream-len N]
[--refuse-skew]`. For every multi-slot producer/consumer channel pair it reports
the **delivered order** and a verdict: `IN-STEP`, `SKEWED` (a permutation of the
stream — a plausible-looking wrong answer), or `STARVED` (the consumer waits for
a fill the producer's chain never reaches — a hang). `--stream-len` is what turns
a skew into a starvation, and it is `down_K` for R1.

This is a third hazard class, and the first two instruments are blind to it by
construction: lock counts are conserved (item 18's audit passes), every acquire
dominates its own BD (item 23's audit passes), and there is only one writer per
slot (item 21's audit passes). Nothing is raced. The stream is simply delivered
in the wrong order.

Calibrated **in both directions**, on synthetic modules differing only in how the
consumer's chain is cut into tasks: a circular chain over 3 slots must read
IN-STEP; the same BDs split as a 2-slot prologue plus a tail must read SKEWED
with delivered `[0,1,3,4,2]`; that split run three times against a 7-item stream
must read STARVED; a circular chain whose period divides the stream must **not**
fire; and a single-slot channel must get **no verdict at all**. Self-test
**6 cases -> 11**.

**Nothing recorded moves.** `--check-order` on the `D2` module §8.3 and §9.1
pinned still reads **RACE** on `%buf21` and **OVERWRITE** on `%buf20`, and
`--check-rotation` reads **IN-STEP** on that same module — which is right, since
`D2` at `herd_x 2` passes 5/5 in the per-column arm.

### 10.8 What R1's reachable shapes are

`down_K = ffn_dim / tile_k` still predicts everything, but it is now two
constraints wearing one number:

- **A compiler-defect boundary** (the rotation skew), and it is removable — devq
  332 removed it with a shipped knob, and §10.3 names the fix site. With the
  rotation correct, `down_K` 5 and 7 become correct.
- **A shim-occupancy boundary**, and that is the binding one. It is why
  `down_K 6` hangs with a perfectly in-step rotation, and fixing the rotation
  does not move it (measured, not argued).

R1's reachable ceiling today is therefore `ffn_dim <= 4 * tile_k` for a correct
answer — 5 with `--omit-pingpong L2`, which costs the L2 double buffering. The
gate shape `768x3072` at `tile_k 32` is `down_K = 96`. **Neither defect is
anywhere near it, and the binding one is `maxq`, not the rotation.** Any plan
that reaches the gate has to make the `hidden` refill stop issuing one shim task
per `(sweep, k-step)` — one task with `repeat_count = down_K` is what this same
compiler already emits for `@air_channel_0` and `@air_channel_1`.

**This is a compiler-side diagnosis and NO COMPILER FILE WAS TOUCHED here.** It
owes `check-air-mlir`, the transformer-layer suite and the ten-model leg when
someone takes the fix, which is why it stops at the diagnosis and the
instrument.

### 10.9 What the next person should do first

1. **Take `maxq`, not the rotation.** The rotation defect is understood, located
   and worth fixing, but it moves R1's ceiling from 4 to 5. `maxq` is what stands
   between here and the gate. First move: fold a launch-level loop of identical
   `air.channel.put`s into ONE task with `repeat_count = trip`, the way
   `@air_channel_0` already is, and re-run `O3`/`K6`/`A` (`down_K 6`). **Record
   the prediction first**: if `maxq` is the mechanism, `down_K` 6 and 8 become
   PASS with the rotation untouched.
2. **Nail the failure mode before designing around it.** The next instrument is a
   synthetic module that pushes N tasks on one shim channel with a gated
   consumer, N swept 3..10 — it measures the limit directly instead of inferring
   it from R1, and it decides blocked-versus-dropped. Do not quote a queue depth
   until it has.
3. **Fix the rotation at `detectNBufferRotation`** (§10.3). Two shapes: emit a
   phase-correct chain when `ops.size() % numBuffers != 0`, or **refuse by name**
   the way v2 now refuses MIMO (§8.4). Silently mis-delivering is the worst of
   the three. Either needs a lit test verified failing first and the full suites.
4. **Do not re-litigate** anything in §9.4's exclusion list, and add to it: the
   L2 rotation is excluded **as the cause of the `>= 6` hang** by devq 332, and
   BD length, total task count and `runtime_loop_tiling_sizes` are excluded as
   causes of it by §10.6.
5. **Row 30 stays open and stays latent**, unchanged by any of this.

`fused`'s SPECS atol stays **PROVISIONAL**, and **no resident-tail latency or
byte figure has been measured on hardware.** Nothing here changes either.

---

## 11 — `[2026-08-12]` Row 28 defect (b): the fold §10.9 specified is **arithmetically unavailable**, and the pacing that replaces it is written but **UNBUILT**

Queue row 28, defect (b). **NO MEASUREMENT IN THIS SECTION.** No compiler was
built, no rung was compiled, no device leg was run. What is here is one
refutation proved off a recorded artifact, one implementation, one regression
lit verified failing pre-fix, and a prediction recorded before any run. The
obstruction that stopped it is named in §11.5 and it is not a technical one.

Instruments: none new. Artifact re-read: devq 332's `O3` compile
(`row28why-private/elf/O3/air_project/npu.air.mlir`, the `down_K = 6` rung).
Prediction file `PREDICTION-MAXQ.md`, sha256 `90b92618…`, written before the
first compile of the changed compiler.

### 11.1 Why the existing folding path does not reach R1's `hidden` refill

It **does** reach it. That is the finding, and it inverts §10.9's premise.

Read straight off `O3`'s emitted runtime sequence, all six `@air_channel_2`
tasks are **byte-identical to each other and to the two channels §10.9 held up
as the ones that already fold**:

```
%2..%7 = aiex.dma_configure_task_for @air_channel_2 {
  aie.dma_bd(%arg0 : memref<64x32xbf16> offset = 0 len = 256
             sizes = [8, 4, 8, 8] strides = [256, 8, 32, 1])
} {repeat_count = 7 : i32}
```

`@air_channel_0` and `@air_channel_1` carry the **same** descriptor and the
**same `repeat_count = 7`**. The refill is not missing the fold; each of its six
transfers is *already folded*, and what is unfolded is the **launch-level loop
around them** — `down_K` copies of a descriptor that is itself complete.

So the question is whether that outer loop can be folded too, and it cannot:

- `repeat_count` **is** the descriptor's iteration dimension. `airrt-to-npu`
  sets it to `sizes[0] - 1` and, when `strides[0] != 0`, *also* emits dim 0 in
  the BD layout to carry the iteration stride. `repeat_count + 1` executions
  therefore **advance** the address by `strides[0]`; they cannot restart it.
- That is settled against a **passing** artifact rather than from the dialect
  docs: `@air_channel_1` is the output `y`, `len = 256`, `repeat_count = 7`,
  and `O5` (`down_K 4`) returns a **correct 2048-element** `y`. 256 x 8 = 2048.
  If `repeat_count` were a queue-level re-issue that restarted the BD, `y` would
  be eight overwrites of its first 256 elements and `O5` would not pass.
- Concatenating six identical copies of that descriptor therefore needs an
  outer dimension of size 6 **at stride 0**, on top of an iteration dimension
  already in use — a **fifth** hardware dimension.
- And the four in use are irreducible. `sizes = [8,4,8,8]`,
  `strides = [256,8,32,1]` is the retile (row-block, microtile column, row,
  element) and **no adjacent pair is mergeable**: 256 != 8x4, 8 != 32x8,
  32 != 1x8.

**§10.9's first move is refused at its premise**, one level above the
experiment, exactly as §9.1's hypothesis was. It is the same wall item 6b
already recorded from the other side ("the retile already uses all four
hardware dimensions … the chunk loop would be a fifth") — 6b hit it on the
96-task feed and this is the same feed at `down_K` tasks.

### 11.2 What shipped instead, and why pacing rather than folding

6b's machinery is the right shape and is reused **unchanged**:
`paceShimFeedForBdReuse` — sink the run past the feeds it must not out-order,
`issue_token` + `dma_await_task(t[i-depth])` before task `i`'s **configure**
(the allocator hands the ID out there), drain the tail so every token is
consumed exactly once.

What 6b does **not** do is fire here. Its trigger is `peak live BDs > tile BD
pool`, and six live BDs on a 16-BD tile is comfortably under budget. **That is
precisely why this shape survived 6b**: the hang is an *occupancy* of the
channel's task queue, not an exhaustion of the tile's descriptor pool, and
nothing was counting it.

New step `boundIdenticalShimPutRuns` in `AIRRtToNpuPass.cpp`, run once after
`boundShimBdLiveness`. It paces a run of **>= 3 structurally identical**
fire-and-forget MM2S configure/start pairs on one shim channel to **depth 2**.

**The trigger is structural and no queue depth is claimed.** §10.6 was careful
that "5 outstanding starts complete and 6 do not" is *not* a documented queue
depth, and nothing here converts that observation into a constant. What fires
the step is that the run is a loop of **identical** puts — a trip-shaped
occupancy the design can raise without bound and that the compiler never
folded. Identity is tested with `OperationEquivalence::exactValueMatch`, so two
puts of *different slices* of the same operand never match; a feed whose
transfers carry distinct offsets is a real multi-part transfer and is left
alone. Depth 2 is not a capacity claim either: it is the smallest in-flight set
that still overlaps a transfer with the next configure, and it is the same
per-feed figure `boundShimBdLiveness` already reserves out of the BD pool.

Pacing at `i-2` cannot deadlock a ring consumer: the consumer takes fills in
order, so fill `i-2` is retired before fill `i` could be accepted, whatever the
staging buffer's slot count.

**Note the two effects are not separable by this change.** The sink moves the
refill run *after* the `w_up`/`w_down` starts, which §10.6 names as part of the
shape, and the pacing bounds the outstanding count. A PASS would not say which
of the two did it. Separating them needs the synthetic instrument §10.9.2 asks
for, and that instrument is still unbuilt.

### 11.3 The regression lit, verified failing pre-fix

`mlir/test/Conversion/AIRRtToNpu/identical_shim_put_run_bound.mlir`. Six
identical `@refill` transfers plus a **negative control** on its own channel —
six transfers with **distinct** offsets, which must come out untouched — plus a
one-shot `@weights` feed issued after the run, which is what the sink must move
the run behind.

Lowered by the **pre-fix** `build-xrt/bin/air-opt` it reproduces R1's shape
exactly: `@out`, six `@varying`, six `@refill` with no token and no await,
`@weights` last, then fourteen clustered `dma_free_task`s. FileCheck **exits 1**
on it, on the intended clause — after `@weights` there is no `@refill`
configure, i.e. the run was never sunk and never paced.

Calibrated in both directions inside the one file: the `@varying` control's
`CHECK-NOT: issue_token` is what a trigger keyed on *count* rather than on
*identity* would trip over.

### 11.4 The prediction, recorded before any run

`PREDICTION-MAXQ.md`, sha256 `90b92618…`, five clauses and four named
falsifiers. In summary: `O3`/`O6` (`down_K` 6/7) TIMEOUT 5/5 -> **PASS 5/5**;
`O2` (`down_K` 5) **stays FAIL 5/5** with the same `sigma = [0,1,4,2,3]`,
because defect (a) is untouched; `O5`/`O1` (`down_K` 4) stay PASS with a
**byte-identical `y` sha** but a **changed control program** (their runs are
also >= 3 identical tasks, so they are paced too — they are *not* predicted
byte-identical at the IR level, and that is stated up front rather than
discovered); `check-air-mlir` 499 -> 500 with the delta being exactly the new
lit.

The consequence is recorded so it cannot be quietly dropped: with (b) fixed and
(a) open the ladder becomes PASS 2/3/4, **FAIL 5**, PASS 6+ — **non-monotonic**,
which is the signature of two independent defects.

### 11.5 The obstruction, named: this is UNBUILT and UNMEASURED

`build-xrt` is configured with `CMAKE_HOME_DIRECTORY=/home/cj/mlir-air`, so
building a compiler change requires the edit to be in the **shared checkout**.
This work was done in a git worktree, and **the permission classifier refuses
the copy into the shared checkout**. No build, therefore:

- **`check-air-mlir` NOT RUN.** Baseline 499/0 stands unverified against this
  change; the predicted 500/0 is a prediction, not a result.
- **No rung compiled**, so §11.4's clause 0 (the compile-only gate: `maxq`
  6 -> 2 on `O3`, the run sunk past `w_up`/`w_down`) is **unverified**. Clause 0
  was written as a gate that must pass *before* any device leg precisely so
  that this cannot be skipped.
- **No device leg.** Every rung figure in §11.4 is a prediction.
- The transformer-layer suite and the ten-model leg are **owed**, as §10.8 said
  they would be for whoever took the fix.

**Nothing in §§8–10 moves.** No measurement is added, corrected or retracted
here, and the only claim §11 makes about hardware is a prediction with its
falsifiers written down first.

### 11.6 What the next person should do first

1. **Build it and run clause 0**, which is compile-only and needs no device:
   `maxq` on `%shim_noc_tile_0_0 / MM2S 0` must read **2** at every `down_K`,
   and the refill run must sit after `@air_channel_3`/`@air_channel_4`'s starts.
   If clause 0 fails, stop — the device legs would be uninterpretable.
2. **`check-air-mlir`, and read the delta by name.** The step is a no-op unless
   a run of >= 3 identical puts exists on one shim channel; any test other than
   the new lit that moves falsifies that and is the red flag to stop on.
3. **Then the rungs**, >= 5 fresh processes per arm, `O5`/`O2`/`O3`/`O6`.
   `O2` staying FAIL is as much the result as `O3` turning PASS: it is what
   shows the step did not reach into defect (a).
4. **Do not read a PASS as "maxq was the mechanism."** The sink and the pacing
   move together here (§11.2). §10.9.2's synthetic N-sweep is still the only
   thing that decides blocked-versus-dropped and still the only thing that may
   quote a queue depth.
5. **Defect (a) is untouched and stays open** at
   `detectNBufferRotation`'s divisibility fallback (§10.3).

`fused`'s SPECS atol stays **PROVISIONAL**, and **no resident-tail latency or
byte figure has been measured on hardware.** Nothing here changes either.

## §12 `[2026-08-12, 21:20]` Why `boundIdenticalShimPutRuns` does not fire — traversal, not trigger

§11's step is in the pipeline (`AIRRtToNpuPass.cpp:2004`, immediately after `boundShimBdLiveness`)
and `check-air-mlir` went 499 → 500 with its regression lit green. It still does not fire on R1.
Measured on the rebuilt compiler, devq 334, `work_maxq/O3/air_project/npu.air.mlir`.

**The cause is the traversal, not the trigger.** The step opens with

```cpp
module.walk([&](func::FuncOp f) { funcOps.push_back(f); });
```

so it handles `func::FuncOp` only. R1's module presents its runtime sequence as
`aie.runtime_sequence` (`AIEX::RuntimeSequenceOp`):

```
aie.device(npu2) @ffn_resident_seg {
  func.func private @ffn_zero_bf16_up_proj(...)        <- declaration, empty body
  ...
  aie.runtime_sequence @ffn_resident_seg_sequence(%arg0: ...) {
    %2 = aiex.dma_configure_task_for @air_channel_2 { ... } {repeat_count = 7 : i32}
    aiex.dma_start_task(%2)                            <- x6, the run to be paced
```

The walk therefore collects only the private *declarations*, every one of which hits
`f.getBody().empty() → continue`. The sequence is never visited, `collectShimBdTasks` is never
called on it, and the step is a silent no-op.

**Why the lit is green anyway, which is the part worth keeping.** The lit's input is a
hand-written `func.func` (`identical_shim_put_run_bound.mlir:101`) and its own CHECK expects
`aie.runtime_sequence` (`:50`) — the pass converts one to the other. So the lit exercises the
**pre-conversion** shape while aircc presents the **post-conversion** one. The test and the
target are on opposite sides of a conversion the pass performs, and nothing in either says so.
This is the project's dominant defect class once more: a check that passes and cannot fail for
the case it exists to cover.

**Everything else was verified against the artifact and passes**, so the trigger itself is sound:

| gate | R1's module |
|---|---|
| `idxs.size() >= 3` | 6 tasks on `@air_channel_2` |
| `t.mm2s && t.start && !issue_token` | MM2S feed, started, no token |
| no `PreserveShimDmaOrder` / `CoalescedShimFeed` / `ShimFeedNoPace` / `kBdRecycled` | none present |
| no packet BD | none |
| exactly one release, a `dma_free_task` | one each, 9 frees at `:354-362` |
| every release after the run's last start | all frees follow every start |
| `exactValueMatch` across the run | textually identical, same `%arg0`, same `repeat_count = 7` |

**The fix is one line** — walk `AIEX::RuntimeSequenceOp` as well as `func::FuncOp` (or walk the
op interface both satisfy) — **plus a second lit case in the post-conversion shape**, without
which the lit will keep passing over a step that cannot fire.

**Deliberately not applied here.** A compiler change owes the transformer-layer suite and the
ten-model leg, ~75 minutes, and the operator is due back; an ungated compiler edit is worth less
than a named one-line change with its test gap written down. `PREDICTION-MAXQ.md`'s clauses 1
and 3 remain **untested**, not falsified — nothing has yet run a compiler in which the step fires.
