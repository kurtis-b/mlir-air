# Transformer-Layer Execution Studies — Port Plan

Porting the transformer-layer execution-strategy studies from the AMD IRON repository (`iron`,
commit `1e014c1`) into MLIR-AIR, then building on the resulting measurement harness: the four
execution modes on the corrected taxonomy, the compiler work they forced, and the `llms/` inference
path measured and optimized against what `ggml-hexagon` does. These documents are the working plan;
they live in the repository rather than the published docs site (`mkdocs.yml` excludes `plans/`)
because they describe work in progress rather than how to use MLIR-AIR. Consolidated 2026-08-22
from the 82-doc tree (its full text is at git tag `pre-cleanup-20260821`, commit `60e287d3`).

## Read in this order

| Doc | What it covers |
|---|---|
| [README.md](README.md) | This file: where things stand, the traps and rules, the status board, the work queue |
| [01-original-plan-superseded.md](01-original-plan-superseded.md) | The original plan (context, inventory, Phases A–G, the two goals, acceptance, the retired port-loop harness), demoted: its "who sequences the work" framing was superseded on 2026-08-08 and its mechanisms by the mode rebuilds |
| [02-porting-conventions.md](02-porting-conventions.md) | **How iron code is refactored into MLIR-AIR house style.** A hard requirement, not advice; reviewable checklist |
| [03-measurement-model.md](03-measurement-model.md) | **The definition of the four modes, and it is current**: reconfiguration cost against DRAM traffic; what is implemented against it; the dispatch vector; CSV schema; `[2026-08-10]` §The vocabulary (submission vs dispatch, packaged vs resident, the role-style names, the knobs-and-costs axis map) |
| [05b-phase-b-buffer-rules.md](05b-phase-b-buffer-rules.md) | **The buffer rules `programming_examples/llms/shared/infra/bo_pool.py` implements** (under `llms/`, not `transformer_layer/`): ownership, synchronization, bank and aliasing; "a rule that is not in this list is not enforced"; its O3 logical-size rule is the easiest way in the seam to produce plausible garbage |
| [15-environment-notes.md](15-environment-notes.md) | Toolchain state and the setup traps that silently hollow out hardware gates: the two CMake flags, build vs install tree, the bare-shell env, oomd, the device queue, the compiler-provenance rule |
| [16-compiler-changes.md](16-compiler-changes.md) | Every compiler change the study forced — H/H1s, H8, H9, H10, J1 (closed), J7a, J7b, the silent-miscompile classes, the fused decoder re-execution fix — with its gate and what measurement falsified |
| [23-rules-and-open-items.md](23-rules-and-open-items.md) | **Read before building anything.** The per-column shim budget, the L3-side offset rule, probe altitude, one process per measurement, the `addnorm` cliff, the open items nobody has claimed |
| [25-mode-rebuilds-and-results.md](25-mode-rebuilds-and-results.md) | The four modes rebuilt on the corrected taxonomy and every layer-level result: the retracted first ladder, the feasibility corrections, `coarse`'s six-cell space and C1, `offload`'s shared xclbin and the 4096 hardware verdict, the memcpy/Phase G scopings, iron's encoder pipeline, the coverage sweep and the width wall, the blind-check census |
| [27-common-ladder-result.md](27-common-ladder-result.md) | **The first four-mode comparison at one sequence length** (512 and 1024, walked twice); the crossover that did not survive walk 2 — read §The crossover that did not survive before running one walk of anything |
| [31-resident-tail-r1-record.md](31-resident-tail-r1-record.md) | `fused`'s resident tail: the R1 spec and byte floor, R2's order seam, wall 5's design, the balance instrument, item 23, wall 7 and the `down_K` defects, the shape-derived mapping, the sealed predictions and their outcomes. R1's box is `herd_x = 1`, `down_K ≤ 6`; `[2026-08-21]` reframed as supertiles |
| [32-cost-decomposed-ladder.md](32-cost-decomposed-ladder.md) | **The cost-decomposed four-mode ladder and the pmode anomaly it caught** (RESOLVED: Turbo reset to `Default`); §The post-flip walk is the standing cross-mode ordering — read its first section before citing any latency |
| [35-goals-1-and-2.md](35-goals-1-and-2.md) | Goal 1 (sliding window: W1 built and gated, the vacuous-gate hazard at 25 %, the model path parked) and Goal 2 (quantized: q4_0 on the shipped kernel, SmolLM2-1.7B int4 done) |
| [44-mapping-frameworks-synthesis.md](44-mapping-frameworks-synthesis.md) | **Five mapping frameworks side by side: none expresses our central axis**; the instrument they point at; FLAT and TileFlow's corrections (four composition states, three words); bibliography section for the seven per-framework readings |
| [54-first-full-profile-and-decoder-families.md](54-first-full-profile-and-decoder-families.md) | **`[2026-08-20]` The first `full` profile — Phase G's gate met**: the 9-length × 4-mode matrix (walk 2, the one to cite), DRAM traffic over a 256× range, the three derived walls, the decoder families, the iron adapter's first real run and the 2.7–9× gap it refuses to compare |
| [55-hexagon-llama-cpp-lessons-for-xdna2.md](55-hexagon-llama-cpp-lessons-for-xdna2.md) | `[2026-08-20]` How `ggml-hexagon` runs a token (source `6503355df0eb`), what transfers: same ~30–38 GB/s weight-stream rate on 4× different bytes, so the decode lever is int4 GEMV then one dispatch per token. Codex's report folded in |
| [56-full-model-mixed-precision-study-plan.md](56-full-model-mixed-precision-study-plan.md) | `[2026-08-20]` PLAN: the analytical planner (`llms/shared/plan/`) and the study run on full models with mixed precision (schema v3, ubatch buckets); H0–H5; **`[2026-08-21]` H0 LANDED** (§4). Codex's review folded in |
| [57-inference-path-optimizations-from-hexagon.md](57-inference-path-optimizations-from-hexagon.md) | `[2026-08-20/21]` The `llms/` decode token decomposed: launch boundaries 106–108 µs each, the re-execution family (§1.5's seven-row table), the twelve-row translation table, the 41–58 ms/token band, experiments O1–O7 with their measured outcomes (§5). Codex's review folded in |

## Where things stand `[2026-08-21]`

**Environment first.** The NPU power mode is a measurement condition and it does not persist across
reboot or `amdxdna` reload: `sudo xrt-smi configure --device 0000:64:00.1 --pmode turbo` (operator),
verified with `xrt-smi examine -r platform`, before any latency. It read Turbo through every job of
2026-08-21 (devq 450–484); no reboot since 08-13. **The tree is the whole state**: tip `244fefe9`
(the cleanup closed 2026-08-22), tree clean; evidence roots are gitignored local copies under
`programming_examples/transformer_layer/results/` (`hexagon-opt-20260821/` has every probe script,
its devq job script, JSON results and `LOOP-QUEUE.md`; `cleanup-20260821/` has the cleanup
scoreboard, job scripts and Codex verdicts). **Two things learned about running probes**: a compile
inside a `measure`-class devq job holds the device lock for its whole duration (the int4 compile
blocked the queue 20 min before it was moved to a detached build), and `XRTBackend.compile` writes
`air_project/` into the *cwd*, so two compiles from one directory clobber each other — give each
its own subdirectory. Everything else environmental is in [15](15-environment-notes.md).

**What 2026-08-21 did** (status-board rows below; numbers in [57 §1.5 and §5](57-inference-path-optimizations-from-hexagon.md), [56 §4](56-full-model-mixed-precision-study-plan.md)):

1. **The launch-boundary cost pinned at 106–108 µs per configuration**, geometry held fixed, three
   estimates agreeing (devq 450–451). Codex's "19 devices × 2 segments" form is unbuildable —
   `air-to-aie` makes one `aie.device` per `air.segment`.
2. **The Qwen3-0.6B decode token 89 → ~75 ms from three launch-count changes, all through `make
   verify`**: `m_input = 8` on the LM head (−1.0 ms); the QKV stage at **4 launches** instead of 8 —
   one GEMV over `[wq; wk; wv]`, one per-row-weighted QK-norm, one RoPE, no new kernel,
   bit-identical (−11.5 ms); the LM head as **9 × 16384 + 4480 mixed partitions** (10 launches, 64
   pad rows; −0.85 ms kernel). **Clean idle-host profile at `fd7e17b8` (devq 486): 12.96 / 13.12
   tok/s = 77 ms per token**, from 89 ms (11.2 tok/s) on 08-20. Qwen3-1.7B got all three:
   **128–130 ms/token** (devq 487).
3. **O2 (runlist pairs) measured small** — −2 to −4 ms/token once the prototype's own overhead was
   removed (`dispatch._plan_memo`); the prototype was kept behind `QWEN3_DECODE_RUNLIST=1` until the 2026-08-22 cleanup removed it (tag `pre-cleanup-20260821`); doc 57 §5 holds the measurement.
4. **Doc 56 H0 — the analytical planner — landed** (`llms/shared/plan/`, `PLAN: 10/10` in the seam
   lit): reproduces both golden drivers' ELF sequences and launch counts from structure,
   `study_skip ≡ profiles.skip_reason`, solver vs registry with every mismatch named. Its first
   finding was item 2's LM-head partitioning.
5. **The LM-head re-execution defect: gated, then avoided.** `make check-lm-head-reexec` read 5 / 7
   wrong on the 19 × 8192 head and 7 / 7 clean on the re-partitioned one; the XFAIL is gone and the
   lit is a regression guard. **The family is not understood**: [57 §1.5](57-inference-path-optimizations-from-hexagon.md)'s
   seven-row table has two other forms that *hang* (a 3-launch host-RMSNorm QKV stage, measured at
   only −0.7 ms/token and not shipped; an RMSNorm-prologue + 16384-row GEMVs ELF).
6. **O4 (int4 LM head) measured and closed for now**: the one-launch form cannot exist (BD repeat
   cap at 4 iterations; its 19-iteration compile never finished, 75 min ×2); the ten-launch form
   saves **0.46 ms/token** because the int4 packed GEMV streams at 11 GB/s (dequant-bound). A faster
   int4 kernel is the prerequisite ([57 §5 item 6](57-inference-path-optimizations-from-hexagon.md)).

**The regression legs**, re-run 2026-08-21 after the `dispatch.py` memo and the shared builder change,
then after cleanup cluster B:

| Leg | Now | History |
|---|---|---|
| `check-air-mlir` | **505 / 0** (devq 412; not re-run on 08-21, no compiler source moved) | 489 (H10) → 491 (6a) → 492 (6b) → 497 (items 8, 9, wall 6, H8) → 498 (item 23) → 499 (wall 7 MIMO lit) → 500 → 501 (28(b)'s `identical_shim_put_run_bound_seq.mlir`, red pre-fix) → 505 (28(a)'s `air_channel_nonclean_rotation.mlir` at its `--implicit-check-not=repeat_count` clause plus three refusal cases) |
| transformer-layer suite | **38 pass / 1 unsupported / 0 fail** (devq 493, +1 the re-homed H9 lit, 96 s) | 31/1/0 → 32/1/0 → 34/1/0 (devq 401) → 35/1/0 (devq 422, +fused re-exec lit) → 37/1/0 (devq 471; the seam lit also pins `PLAN: 10/10`); the 1 is R1's parked gate |
| PR-safe host allowlist | **14 / 14** through lit with the regex ninja passes; workflow guard at `Passed=14` | 1 → 10 (devq 261, `Excluded=352 Passed=10`) → 12 → 14 (`run_softmax_rows_tests`, `run_profile_bounds_tests`); `build.ninja`'s filter refreshes at the next CMake configure |
| study host suite | **611 / 611 in 27 modules**, pinned by `run_study_host_tests.lit` | 103 → 231 in 17 → 357 in 19 → 429 in 21 → 517 in 23 → 539 in 24 → 563 in 25 → 567 (devq 401) → 587 in 25 (FileCheck-verified red at 567, 581, 584, 585) → 600 → 603 → 607 → 610 → 611 (cluster B's review rounds) |
| seam lit | **35 + 10** (`dispatch` 35/35, `plan` 10/10) | — |
| shipped models | **8 / 11 is the STANDING LEG** (devq 472 six-model leg 6/6 after the O1 port and the plan memo; devq 416's two llama passes stand) — `qwen25_3b`, `llama32_3b`, `qwen3_4b` **not run, by operator decision** (their `verify` oomd-kills the session, [15](15-environment-notes.md)) | 10/10 ×4 (devq 305, 326, 337, 357–358) → 11/11 (devq 402, `smollm2_1_7b_int4` joined) → 8/11 |

`install-xrt` was refreshed after each compiler change; check with `ls -l`, never `cmp`.

**Standing numbers to cite.** Layer study: [54 §1](54-first-full-profile-and-decoder-families.md)
walk 2 for the nine-length matrix and DRAM table (`full`'s `--warmup 1 --samples 3` — do not splice
against doc 32's `2/5`), cross-mode orderings from [32 §The post-flip walk](32-cost-decomposed-ladder.md)
and its 2026-08-18 re-walk (absolute numbers from `results/rewalk-doc32-w{1,2}`), decoder families
from [54 §4](54-first-full-profile-and-decoder-families.md). Model path: [57 §1.5](57-inference-path-optimizations-from-hexagon.md)
(boundary 106–108 µs, 146 µs per `xrt.run`), §5 items 3c / 5 / 5b (kernel-line deltas, each a
same-session before/after under recorded Turbo), and **§1.1's correction line: Qwen3-0.6B bf16
decode 77 ms/token (12.96–13.12 tok/s), devq 486, idle host, Turbo**. **Do not cite** the June
`llms/` table (pmode-unrecorded), Hexagon's figures as like-for-like ([55 §5](55-hexagon-llama-cpp-lessons-for-xdna2.md)),
or iron's latency figures against the DEFAULT study numbers — the only like-for-like iron
comparison is [54 §5](54-first-full-profile-and-decoder-families.md)'s `[2026-08-22]` table (devq
504, `--no-stage-stats`, same session): coarse 17.25 / fused 14.83 vs iron hybrid 15.07 ms at 512.

**Operator decisions `[2026-08-21, afternoon]`** — every "operator decision" row is settled:

- **R1 is reframed, not closed.** Model the array's workload as **supertiles** of per-core tiles
  (mlir-air composes them as regions); one supertile then the next are **separate executions in
  the runtime sequence**. The gate shape becomes a sequence of supertiles each inside the working
  box (`herd_x = 1`, `down_K ≤ 6`) — wall 7's shared-L2 multi-writer buffer is designed out rather
  than fixed. **Each supertile should produce a finished output block** (`down_K = 96` per
  execution, which meets the unresolved `down_K ≥ 7` wedge, [31 §wall 7](31-resident-tail-r1-record.md));
  accumulating down partials *across* executions is acceptable only if it measures faster than the
  finished-block form — so R1's first increment is that two-form comparison on hardware. Rows 6 /
  28 / 30 follow this.
- **J1 closed**, superseded by J7a. **Goal 1** (sliding window / Gemma 3) **parked**.
- **The big-three model leg will not be run**; 8 / 11 is the standing leg, not a gap.
- **The iron latency gap**: approved — build and run `~/iron` at **devel HEAD** (`cc7083f`, one
  commit past the ported `1e014c1`) on this NPU, one devq job per side.
- **Docs 01–12 are demoted** to one "original plan, superseded" doc.
- Goal 2 step 5 was already done (`smollm2_1_7b_int4`, 08-19) and is off the open list.

**The queue.** Before any of the post-cleanup items: the branch cleanup — the branch was +131,003
lines over `main` across 473 files (A artifacts 80 + 12 binaries / B agents 15,140 / C plan docs
29,742 / D transformer_layer 57,886 / E llms + kernels 18,746 / F compiler 8,854 / other 555), and the
standing priority is to cluster code with its docs, consolidate, and minimize what the branch adds.
Compiler code (`mlir/lib`, `mlir/include`, `python/air/backend`) stays as-is; only its docs and
tests consolidate. The branch is squashed per cluster at the end; at most ONE Codex review round
per cluster commit (operator, 2026-08-22). **Clusters A and B have landed** (`3d08333b`;
`cbd2858e..244fefe9`) — the scoreboard, gate logs and Codex verdicts are in
`results/cleanup-20260821/`. Then C (plan docs, this consolidation), D (`transformer_layer`:
builders vs `llms/shared/builders`, `*_structure.py` + `opcheck*` into `study/` or gone, dead modes,
lit dedupe; R1 code kept as the supertile seed), E (`llms/`: O2 prototype out, qwen3 driver dedupe,
golden JSON regenerability), F (compiler docs and `mlir/test` dedupe), close (re-run gates,
before/after table, squash, README points at the consolidated docs). The post-cleanup items are
§The work queue.

## Traps and rules

Environment traps (CMake flags, build vs install tree, compiler provenance, the bare shell, oomd,
the device queue) are in [15](15-environment-notes.md). These are the study's own.

**Measurement.**

- **Trap 0 — pmode.** At `Default` the verdict rung (`study/run_mode.py --mode offload --seq 1024
  --warmup 2 --samples 5`) reads ~2.5–2.7 s (82 ms per `hw_context` load) against **156 ms, 3.7
  ms/load** at Turbo — a ~15–20× error that presents as a compiler regression and cost a day.
  Latencies recorded 2026-08-10 are `Default`-conditional, pre-08-10 Turbo-conditional; bytes and
  counts are pmode-independent. Re-measure a whole comparison after any pmode change, never splice.
  `[2026-08-12]` The trap lives in the data: `study/manifest.py` records the mode in a `conditions`
  block (`schema.CONDITION_FIELDS`, a block not a column so `SCHEMA_VERSION` stays 2), `compare_roots.py`
  **refuses** two roots recording different modes (`[SPLICED]`). Roots recorded before 2026-08-12 read `unknown` and are *flagged, not refused*
  — **do not stamp a mode you did not observe**; re-walk to condition a comparison. The shipped
  throughput floor's own pmode is `unknown` and is flagged, not refused, by design (re-seeding it
  to clear a flag is what the driver-owned floor file exists to prevent). Full chain: [32](32-cost-decomposed-ladder.md).
- **Walk anything twice.** One walk published a crossover a second refuted (J3, then [27](27-common-ladder-result.md)).
  `offload` drifted up to 120 % within one walk and 23–32 % walk to walk on the ELF path against
  2–10 % for the other three; the shared default removed that. And **a mode-vs-mode comparison at
  warmup 1 compares warmup tails** — `fused`@1024 needs warmup ≥ 2 to repeat to 0.4 % (devq 353).
- **Build a cross-mode table from a ladder run, never from the catalogue.** The SPECS rows still
  span two lengths (`fused` 1024, the other three 4096); `fused`'s `sync 19` against `coarse`'s 402
  stays withdrawn (at 1024 they are 13 and 107).
- **Compare distributions, not a run against a number.** A four-mode table was published from runs
  taken beside builds — `coarse` at 4096 read 731 ms there and 467/477 ms on a quiet host (1.55×);
  a "5.9 % improvement" was three fresh runs against one stale high baseline. Conditions are part of
  the measurement: compilation sits outside the clock, host dispatch inside; nothing CPU-heavy runs
  beside a timed region, which `devq`'s build/measure classes enforce.
- **A recorded claim with no artifact may simply be wrong.** Doc 16's "`attn_output` timed out on
  the one configuration tried, of 828" shaped two days of planning; the first configuration passes
  at every rung by all three methods and the 828 is unsourced. Check for a log, checkpoint or test
  before planning around a number.
- **A count reconciled by arithmetic is a count nobody ran.** Three merges each moved
  `run_study_host_tests.lit`'s pin against the same 357 baseline; the resolution was to run the
  merged suite, pin what it printed, and verify the pin red at the stale value — never add deltas.
- **`attention_path` is not a covariate** (since 2026-08-09 all four modes run attention on the
  device); `study/test_attention_path.py` asserts the end state. The first ladder's host-vs-device
  slope split (1.23–1.27 vs 1.03–1.17) cannot be reproduced and any rerun showing it needs a new
  explanation.
- **Distinguishability is ordinal, never threshold** (`coarse` measures 131 entries, 128 of them
  `addnorm`'s row blocking, so an absolute would measure L1 capacity). The dispatch vectors have a
  negative control: the fault-injected run's summed vector totals must **equal** the clean run's —
  `study/distinguish.py` since cluster B, live in `run_profile.gate()`, failing closed on NaN,
  duplicate lengths, failed rows as skips and swapped skip identity.
- **A stage gate's absolute ceiling can be vacuous at a boundary** — `runlist`'s unscaled attention
  passed every walk at 25× the other modes' relative error; the top-k token gate accepted a
  full-causal degradation on 1 of 4 window-crossing prompts. Read the relative column too.

**Design.**

- **The per-column shim MM2S budget is two, across the whole segment** — stacked herds put one tile
  of each into every column, so their L3 demands add; over two, AIR packet-multiplexes onto one
  queue ([23 §1](23-rules-and-open-items.md)). Counting it: an L2-staged refill is a `shim→memtile`
  flow on the same port as a herd-direct `shim→core` fetch (a census counting only the latter read
  worst column 1 where the truth is 2), and **over budget the routed design shows FEWER shim flows**
  (zero inbound `aie.flow`, twelve `aie.packet_flow` on the control) — count **demand**, circuit
  flows by source port plus one per packet-multiplexed stream, as `ffn_resident_structure.py` does.
- **Advance a staged buffer on the L3 side, never on the L2 read** — an IV offset is inexpressible
  on an `aie.dma_bd` and the compiler used to emit a stale-offset chain that hangs ([23 §2](23-rules-and-open-items.md);
  H10 made it a refusal).
- **`runtime_loop_tiling_sizes` is not inert and the lowered IR will say it is**: `[2,2]` hangs
  `mha_out_proj` @4096 (8/8 vs 0/8, Fisher p = 1.6e-4) while `aie.air.mlir` is op-identical; it also
  leaves context-corrupting residue under the ELF ABI (3.8141e-01 divergence in exactly one cell of a
  2×2 factorial, non-accumulating). Byte-identical for R1. Never settle inertness on an IR dump —
  `aie.air.mlir` is not byte-reproducible (~95 lines between compiles of one preset), nor is the ELF.
- **Give every L1 buffer one role** — a buffer that is both DMA destination and kernel output does
  not read back what the kernel wrote (`softmax`'s first version returned its input at all three
  shapes).
- **A normalization's fault-injection target is chosen by measurement** — `(rows-1, 0)` left
  softmax's negative control passing at two of three shapes; move the target, not the tolerance.
- **No tolerance is widened for any mode.** The layer `atol` sits at the hard `1e-1` ceiling;
  margins at the layer output: `offload` 1.73×, `runlist` 1.43×, `block`/`coarse` 1.43×
  (`atol_required` 1.663e-2 after item 7), `fused` 1.72× at 1024 (`atol_required` 5.813e-2); the
  retired 4096-era 1.27× (7.896e-2) survives as cell C2's margin. If a mode needs more, the answer
  is a recorded finding.
- **Reference oracles are re-expressed, not ported.** iron's per-operator gate is bf16 at `rtol=4e-2`
  with a 0.5 % mismatch budget; its end-to-end gate is `FINAL_REL_TOL=0.1`, `FINAL_ABS_TOL=0.5`, 5 %,
  only at `seq_len <= 512` (`REFERENCE_VALIDATION_MAX_SEQ_LEN`), degrading to a finite-output check
  above. This port: FP32 reference (`pattern/reference.py`, `generate_golden_reference()` with its
  load-bearing `WEIGHT_DRAW_ORDER` — use it, do not re-port iron's bf16 original), the registry's
  `rtol`/`atol`, zero mismatches, at `seq 4096`; erf GeLU; the MHA oracle's precision switch at
  `seq_len 16384` ([01](01-original-plan-superseded.md)).
- **Shape coverage is a sweep, not a redesign**: 108 distinct projection-GEMM shapes across the
  case matrix, the 36 `baseline_768` registered by C4, hidden 512/768/1024 36/36 since 2026-08-07.
- **One `KernelCache` directory per mode** (`BLOCK_CACHE_DIR`): the cache is keyed by fingerprint but
  the directory by name, so two modes sharing one trade artifacts and attribute valid numbers to the
  wrong execution boundary. `coarse` wraps `builders/block.py` (four `run_sequence` calls; the
  normalization points are 64 dispatches each because `build_addnorm_module` caps rows per call —
  `coarse`'s vector is dominated by `addnorm`, not the GEMMs).
- **Match a probe's altitude to its claim** ([23 §3](23-rules-and-open-items.md)): `air-opt` answers
  "does this pass fire", not "does this compile". **A fixture proves only the shape it runs** —
  Phase H's four variants were green at `herd_x=1` while the miscompile lived one column wider;
  H9's `multicolumn` clause (now `addnorm_multitrip.py`) was verified failing before the fix.
- **Structural literals from a pre-fix dump are invalid, not unlucky**: the old `air-fuse-channels`'s
  green R1 compiles were wrong too (an extra channel with pairwise 2-slot wraps where one 4-slot
  stream belongs). Re-derive, never compare against.
- **From the survey** ([44](44-mapping-frameworks-synthesis.md)): we have four composition states
  and three words (packaged, resident, interleaved, and the unnamed `Para` iron's autotune converged
  on); fusing by interleaving is not fusing by residency, and FLAT rejected co-residency; which
  sub-chain to fuse is a function of the rung (attention is 12 % of the layer at N=512, 79 % at 16K);
  and every four-mode comparison here is at a fixed tiling, under which TileFlow's Table 7 shows
  granularity looking 18× important while three of four granularities tie under searched tiling —
  no packaged-vs-resident-vs-interleaved claim is publishable until it survives a tile-size search
  on both sides. R1's wall 5 and FLAT §5 are the same object (a shared outer loop makes
  channel-major unreachable by construction).

**Load-bearing questions already answered.**

| Question | Answer | Where |
|---|---|---|
| Can separately-compiled ELFs share one runlist? | Yes — N ELFs, N `hw_context`s, one runlist; bit-identical to sequential, 1.02–1.15× faster. **Not** by sharing one context (XRT rejects that three ways) | [01](01-original-plan-superseded.md) (05a) |
| Concurrent `hw_context`s on NPU2? | 32; 33 fails with `DRM_IOCTL_AMDXDNA_CREATE_HWCTX err=-2`; `runlist` wants 29 | [15](15-environment-notes.md) |
| Can the two attention matmuls run on device? | Yes, at every rung, `attn_output` by all three GEMM methods; the registry holds no `K = 64`/`N = 64` row and the sweep cannot stage one (`FAMILY_HIDDEN × ROLE_KN_MULTIPLES`, minimum hidden 512), so the tiles are injected through `gemm_spec_fn`. Two bounding clusters: `herd_n = 1` at N=64 hangs, `tile_n = 8` fails the microkernel's assert | [25](25-mode-rebuilds-and-results.md) (26) |
| Can one xclbin hold the whole layer? | No, twice: `[2,2]` hangs `mha_out_proj` @4096, and `air-fuse-channels` is O(N²) in channels and did not finish in 1200 s on a 90-channel stitch | [25](25-mode-rebuilds-and-results.md) (26) |
| Can N instruction streams share one xclbin? | Yes (`93e15a64`): five shapes at `context_loads 1` vs 30. Needs THREE identifiers per stream — `kernel_name` (duplicate ⇒ xclbinutil refuses, the only loud one), `instance_name` (substring match), `kernel_id` (routes to a PDI via `dpu_kernel_ids`; every AIR compile defaults to `0x901`; a collision times out at one shape and returns garbage at `mean_rel_L1` 1.41 with no error at the other) | [25](25-mode-rebuilds-and-results.md) (29) |
| Can the shared xclbin hold every shape? | No — single-launch modules only: at 4096 the `fused-cast` down-projection is two `air.launch` ops and `XRTBackend.compile`'s fixed `insts="air.insts.bin"` makes aiecc refuse (*duplicate output path*); the `drain` pin keeps the chain single-launch. In-stream `load_pdi` faults the firmware (`fatal_error_type 0x10`); `--expand-load-pdis` is a no-op on aiecc 1.4.0 | [25](25-mode-rebuilds-and-results.md) (29) |
| Is `_evict_context` what makes `offload` noisy? | Not isolated — but removable: ELF path 316.9 % / 134.1 % intra-walk at 512 vs 17.6 % / 14.0 % shared, `runlist` in band as control; the switch changes ABI *and* reconfiguration; the 1024 rung did not reproduce its own 61.6 %/59.8 % (read 9.0 %/10.5 %) | [25](25-mode-rebuilds-and-results.md) (29) |
| Widest common sequence ladder? | 512 and 1024 only: `fused`'s `plane_major` stride cap (1365 rows) above; `256 % (tile_n 128 × herd_n 4) ≠ 0` for `offload`/`runlist` below — a tile constraint, now a derived skip; `full` walks 64–16384 with 15 derived skips | [27](27-common-ladder-result.md) · [54 §3](54-first-full-profile-and-decoder-families.md) |
| Does DRAM traffic order as the taxonomy predicts? | Cold at 1024: `fused` 42.5 < `coarse` 44.0 < `runlist` 55.2 < `offload` 99.1 MB; warm since 08-10: `runlist` (40.9) < `fused` < `coarse` < `offload`, every gap decomposing to the byte; `offload`/`runlist` 1.79× at 1024, 5.1× at 4096, 13.45 GB vs 1.01 GB at 16384 | [32](32-cost-decomposed-ladder.md) · [54](54-first-full-profile-and-decoder-families.md) |
| Which mode is fastest? | `fused` < `coarse` < `runlist` < `offload` on averages and minimums, 16/16, 512 and 1024 (post-flip walk; re-walk 2026-08-18 holds it 8/8 with `fused` ≈ `coarse` at 1024 within noise); holds in all three decoder families | [32 §The post-flip walk](32-cost-decomposed-ladder.md) · [54 §4](54-first-full-profile-and-decoder-families.md) |

## Status board

A phase is `done` only when its gate passes. Rows are in landing order; every row keeps the numbers
that appear nowhere else.

| Item | Status | Numbers and ids | Doc |
|---|---|---|---|
| A — AIE2P kernels | done 2026-08-04 (18 min) | every kernel to `.o` with Peano; compile-only lit | [01](01-original-plan-superseded.md) |
| B — runtime seam | done 2026-08-04 (362 min) | multi-ELF runlist on hardware, identical to sequential, lower latency. The driver-run gate never touched the NPU (2 tests, 16 s); the hardware claim stood on a self-report until `run_npu2_runlist_gate.lit` re-ran all four legs | [01](01-original-plan-superseded.md) · [05b](05b-phase-b-buffer-rules.md) |
| C1 — gate mechanism + small operators | done 2026-08-04 (61 min) | `opcheck.py` + fault-injection negative control; `causal_mask`, `addnorm`, `layer_norm`, `elementwise_add` | [01](01-original-plan-superseded.md) |
| C2 — `qkv_proj`, `ffn` | done 2026-08-04 (45 min) | full-output `np.isclose` at registry tolerance vs FP32 | [01](01-original-plan-superseded.md) |
| C3 — `mha_out_proj` | done 2026-08-04 (68 min) | FlashAttention tolerance, causal and non-causal | [01](01-original-plan-superseded.md) |
| C4 — coverage sweep | done 2026-08-04 (504 + 66 min) | 36 `baseline_768` shapes through `gemm_config()`; registry 33 → 69 bf16-out GEMM shapes, pre-existing rows byte-identical; ten models verify. C1–C4 = 21 of 40 driver invocations, ~12 h; C4 halted once on a driver bug (registry mtime). Round-3 loose end: `64x768x2304` QKV lacked its `fused-cast` row and the resolution gate checked winners not required methods — both fixed, the sweep's fused-cast config replaced one returning zeros for two of nine cast sub-tiles | [01](01-original-plan-superseded.md) |
| D1 — operators at `baseline_768` | done 2026-08-05 (11 min) | every operator, including the pre-add `addnorm` (`-DADDNORM_PRE_ADD`, built by A, never dispatched before) | [01](01-original-plan-superseded.md) |
| D2 — block integration | done 2026-08-05 (156 min) | one `encoder_bert` layer at `baseline_768` seq 4096 matches FP32 torch over 4096×768, zero mismatches, ten per-boundary intermediates; D1+D2 21 of 40 invocations, ~4.5 h. Found `compile_gemm_mm`'s `tile_n` object collision (`mm_m32.o`/`mm_m64.o` baking `-DDIM_N`), worked around by interleaving. `atol` at the `1e-1` ceiling, 1.35× over `atol_required` 7.4e-2, `mean_rel_L1` 1.7e-2 | [01](01-original-plan-superseded.md) |
| E1 — unblock the ladder | done 2026-08-05 (79 min) | `(method, tile_n)` naming in `llms/shared/builders/gemm_builder.py` closes both collisions; `ffn` at a second ladder point; ten models | [01](01-original-plan-superseded.md) |
| E2 — `coarse` + instrumentation | done 2026-08-05 (38 min) | full scope behind a measured dispatch vector | [01](01-original-plan-superseded.md) |
| E3 — `offload` | done 2026-08-05 (55 min) | matches, aggregates nothing; attention then in host torch (superseded 08-08) | [01](01-original-plan-superseded.md) |
| E4 — `runlist` | done 2026-08-05 (91 min) | more entries than `coarse` | [01](01-original-plan-superseded.md) |
| E5 — `fused` + distinguishability | done 2026-08-05 (62 min) | all four vectors separate as the taxonomy predicts | [01](01-original-plan-superseded.md) |
| H — compiler hardening | halted 2026-08-06 at `confirm/3`, superseded by H1s | `gate-h.sh` four legs; spec corrected after the halt | [16](16-compiler-changes.md) |
| H1s — skip, do not refuse | done 2026-08-06 (109 min) | five legs: build + install, `check-air-mlir`, suite, decode throughput vs recorded floor, `make verify` × 10 | [16](16-compiler-changes.md) |
| J3 — sequence ladder | done 2026-08-08, 16/16 twice; **retracted as a mode ranking** | crossover `fused` at 512/2048, `coarse` at 4096; slopes device 1.03–1.17 vs host 1.23–1.27; 1024 indistinguishable across walks. Ranks four implementations predating the taxonomy | [25](25-mode-rebuilds-and-results.md) |
| Taxonomy correction | recorded 2026-08-08 | `runlist` = every operator on device; `offload` = one xclbin, N instruction streams, all linear on NPU, all non-linear on host; `coarse` = per-workload blend; `fused` = whole layer on the array. No mode met it that day | [03](03-measurement-model.md) |
| J1 — collapse the norm dispatches | blocked 2026-08-06 (`fix/1`); **closed 2026-08-21** by operator, superseded by J7a | multi-column multi-trip `addnorm` miscompiled 4070/4096 at `herd_x=8`, 2 trips; guards `52b57c8f`, `ef5e1cf1`; `coarse` stays at 131 entries; after H9: shim BD exhaustion at 6 trips vs 64, loud | [16](16-compiler-changes.md) |
| H9 — fuse packet put loops through `scf.parallel` | done 2026-08-07 (184 min) | `multicolumn` 3747+/4096 wrong → exact; 10/10 models; three review rounds each found a real defect. `[2026-08-21]` fixture re-homed: `addnorm_multitrip.py`, `run_npu2_addnorm_multitrip_peano.lit`, `make check-addnorm-multitrip` | [16](16-compiler-changes.md) |
| J7a — norm-tail pipeline | done 2026-08-07 (87 min) | `mean_rel_L1` 3.620e-3 at 4096×768, 4.7× under 1.688e-2; zero packet-typed channels; `layer_norm` ~25–26× more accurate for ~13 % throughput | [16](16-compiler-changes.md) |
| J7b — accumulator ring | done 2026-08-07 (58 min) | in-place accumulator dispatched; C DMAs hoisted out of the K loop | [16](16-compiler-changes.md) |
| H10 — non-constant BD offsets | substance verified 2026-08-08; tamper check halted on five documented gate files | `H GATE: PASS` five legs, `check-air-mlir` 489/489, 11.44 tok/s vs 9.43 floor, 10/10; the unchecked `std::optional` deref located | [16](16-compiler-changes.md) |
| F — study harness | gate passes 2026-08-08 (`smoke_gate` PASS, `manifest complete: True`, four clauses); plot tier done 2026-08-14; `memcpy_bandwidth` alone remains (a device design) | no unmerged worktree (`exper/phase-f-study-harness` tip `4775722e` is an ancestor). Item 4 four of five, item 3 portable: `resource_usage.py` (`core_to_core_flows` 16/40 space-multiplexed on a norm-tail compile, 0/116 on `transformer_layer`; devq 238), `component_groups.py` (`offload` @1024 attributes 79.8 of 159.8 ms, 50.1 % unattributed, job 246 — the item-4 row reads 80.0; `sync 90` / `bytes 99090432`), `run_lock.py`, `cases.py`, `power.py` (RAPL/hwmon), `compare_roots.py`, `select_rows.py`; three modules unported (`npu_runtime_checks.py` superseded by `require_turbo`). Host suite 103 → 231 in 17 | [25](25-mode-rebuilds-and-results.md) · [01](01-original-plan-superseded.md) |
| Corrected `offload` — attention on device | done 2026-08-08 | `run_npu2_offload_peano.lit` both recipes; 10/10 stages, `submissions 30 entries 30 air 31 herd 91 sync 91 bytes 970457088`; `attn_context` 11.4× margin, `output` 1.73×; 6.9× the DRAM traffic | [25](25-mode-rebuilds-and-results.md) |
| `fused` build repair — SPECS 4096 → 1024 | done 2026-08-08 | gate was red and unrun (mode bounded 256..1024); 10/10 at 1024, `mean_rel_L1` 1.756e-2, `atol_required` 5.813e-2; cross-mode `sync` vs `coarse` suspended | [25](25-mode-rebuilds-and-results.md) |
| Backend-preset conflict | recorded 2026-08-08, re-measured 2026-08-12 at 5 repeats/arm | `runtime_loop_tiling_sizes` `[1,1]` 8/8 PASS, `[2,2]` 0/8 `ERT_CMD_STATE_TIMEOUT`, Fisher p = 1.6e-4; `.ctrltext` 11 % / 10 % / 2.96× smaller under `[2,2]`; byte-identical for R1; retracts the feasibility doc's §4. Probe `probe_backend_preset_hardware.py` (`--compile-only`; retired to the tag in cluster B) | [25](25-mode-rebuilds-and-results.md) |
| Device `softmax` | done 2026-08-09 | `builders/softmax.py` over `softmax_streaming.o`; 512×512, 4096×768, 64×4096 (`rows_per_call` 8 → 2); `mean_rel_L1` 1.60–1.63e-2, `atol` 2.7–2.9× `atol_required`, plus a `mean_rel_L1_max` ceiling | [25](25-mode-rebuilds-and-results.md) |
| Corrected `runlist` — every operator on device | done 2026-08-09 | 427 entries over 17 runlists; `submissions 17 entries 427 air 50 herd 488 sync 451 bytes 190513152`; one submission per head is a memory bound (~800 MiB batched, ~70 MiB per head) | [25](25-mode-rebuilds-and-results.md) |
| First result on the corrected axis | recorded 2026-08-09 | at 4096: `runlist` 190,513,152 B vs `offload` 970,457,088 (5.1×); attention 25,165,824 vs 830,472,192 (**33.0×**), everything else 165,347,328 vs 139,984,896 (0.85× — the norm-chain confound opposes the effect); `runlist`'s non-attention total byte-identical to its pre-rebuild pin | [03](03-measurement-model.md) |
| `attention_path` retired as a covariate | recorded 2026-08-09 | `study/test_attention_path.py` asserts all four modes on device | [27](27-common-ladder-result.md) |
| The four modes at one length | recorded 2026-08-09, 8/8 twice | 512 and 1024; DRAM `fused` 42.5 < `coarse` 44.0 < `runlist` 55.2 < `offload` 99.1 MB at 1024 (warm superseded 08-10); `fused` fastest, `coarse` second; `runlist`/`offload` indistinguishable then; `offload` drifts to 120 % | [27](27-common-ladder-result.md) |
| `offload` N instruction streams | done 2026-08-09 (`93e15a64`) | `context_loads 1 kernel_attaches 4 over 30 dispatches` vs 30 loads; vector unchanged 30/30/30/90/90/99,090,432; `E1 GATE: PASS` lit 28/28, 10/10; no latency claim (four A/B runs overlap) | [25](25-mode-rebuilds-and-results.md) |
| `offload` shared path gated; variance explained; default flipped | done 2026-08-09; 4096 bound lifted and default flipped 2026-08-11 | suite 28/28 (494.5 s), host 84/84, seam 31/31, `phase_e_checks` 30/30; pins `context_loads 1 kernel_attaches 4` shared vs `30 / 0` ELF, verified failing crosswise. Shared path ~20 % slower at 512 (97.5–99.5 vs 78.9–82.0 ms). Route 3 (`drain` pin) gates at 4096: 10/10, cold delta 293,200 B = five insts streams, −12,582,912 vs ELF = fused-cast f32 scratch; suite 30/30 (519.7 s, 24 workers), 10/10 (~66 min). ELF path is `AIR_OFFLOAD_LEGACY_ELF=1`; `AIR_OFFLOAD_SHARED_XCLBIN` raises; every pre-flip `offload` latency predates the flip | [25](25-mode-rebuilds-and-results.md) |
| Corrected `coarse` | done 2026-08-09 | C2 4/389 and C3 17/169 predicted host-side before running (model reproduces `coarse` 4/131, `runlist` 17/427); C1 < C2 < C3 < C6 both walks at 2048/4096, avgs and minimums; front axis ~1.5–1.6× the tail's effect; `coarse` = C1 (block front, banded tail); `make check-coarse-c2`, `check-coarse-c3`, `coarse-cell-structure` | [25](25-mode-rebuilds-and-results.md) |
| Cost instruments — schema v2 | done 2026-08-10 (`eeb37a19`, `4ced893b`) | `device_ms`/`sync_ms`/`host_cpu_ms` + `context_loads`/`kernel_attaches`, v1 prefix pinned; offload-ELF 30, `runlist` 24, `coarse`/`fused` 0; `cache.py` had always counted ELF loads | [03](03-measurement-model.md) |
| Small confounds — items 4 + 5 | done 2026-08-10 (`2f66fc86`, `e2996fbd`) | targeted pool eviction (warm `runlist` −14,352,384 B, cold unchanged); the seam gate's latency clause on interleaved minimums, not widened | [25](25-mode-rebuilds-and-results.md) |
| Multi-launch xclbin packaging | compile half 2026-08-10 (`623768f2`), installed 08-11; **dispatch NO** | fixture `test/xrt/56` verified failing unpatched; 29 single-launch dispatches clean at 4096, the multi-launch module faults the firmware (`fatal_error_type 0x10`) | [25](25-mode-rebuilds-and-results.md) |
| `fused` resident-tail scoping | done 2026-08-10 (`601c54ae`) | byte floor 15.0 / 16.5 MiB at 512 / 1024 vs 48.75 / 84.0 packaged; J7a×2 + J7b within the column budget at 1024 and 4096; hermetic probe with failing control | [31](31-resident-tail-r1-record.md) |
| Resident tail R1 | built 2026-08-11 (`0507a1e5`); device gate parked `UNSUPPORTED`; **reframed 2026-08-21** | emulation 8/8 at 5.457e-12, rejects the pre-E1 builder at 4.716e+03 (the first arm never built the module). Walls: 6a `air-fuse-channels` crash N=3 5/5 (N=2 fuses 5/5) fixed, `fuse_channels_sibling_nests.mlir`, `check-air-mlir` 491; 6b shim BD exhaustion (`hidden` 96 tasks + 1 = 97 live BDs on tile (1,0) vs 16) fixed `ea3b98ce`, `shim_bd_liveness_bound.mlir`, 492/0, suite 31/1/0 (devq 248), paced on hardware never; 6c wall 5 `ERT_CMD_STATE_TIMEOUT` (devq 235/236; `air.preserve_shim_dma_order` does not fix it and also disables folding); E1 census on the unmarked build: `[w_up][w_down][hidden ×96]`, `w_down` 13 → 1 BD, channel symbols 12 → 9. Box: `herd_x = 1`, `down_K ≤ 6` (≤4 at devq 300 21/21; 5 by 28(a) devq 398; 6 by 28(b) devq 403; ≥7 wedges). SPECS atol PROVISIONAL; no resident-tail latency or byte figure has ever been measured | [31](31-resident-tail-r1-record.md) |
| First cost-decomposed ladder | recorded 2026-08-10 | warm `runlist` < `fused` < `coarse` < `offload`; reconfiguration offload 30 / runlist 24 / coarse 0 / fused 0; `hw_context` ~78–80 ms/load vs ≤2.6 — RESOLVED 08-11 as pmode (Turbo: 156 ms, 3.7 ms/load) | [32](32-cost-decomposed-ladder.md) |
| `layer_norm` offset-regime row | done 2026-08-10 (`b4fe19a3`) | `run_npu2_layer_norm_peano.lit` three shapes; `mean_rel_L1` 9.819e-5, `atol_required` 0.0 pinned | [23](23-rules-and-open-items.md) |
| `addnorm` two-pass f32 + offset rows — item 7 | done 2026-08-11 (`9278be34`) | offset rows `mean_rel_L1` 1.390e-3 / 1.409e-3, `atol_required` 0.0 vs the one-pass kernel's 22.2 / 33.1 collapse (cliff at `|mean|/sigma` ~4; workload worst row 0.115); `block`/`coarse` 1.688e-2 → 1.663e-2 (margin 1.35× → 1.43×), `runlist` 1.746e-2, `fused` unchanged | [23](23-rules-and-open-items.md) |
| `air-fuse-packet-put-loops` decline diagnostic | done 2026-08-10 (`1b15a1b0`) | four `-verify-diagnostics` cases, lit 2/2 | [16](16-compiler-changes.md) |
| R2 order-seam scoping | done 2026-08-11; **re-derive, do not inherit** (its two constraints dissolved with items 8 and 9) | partition GEMM herds by rows (M), not re-map the norm tail; design arm 4 core→core with every `aie.dma_bd` offset 0; `l2_staged` refuses at 48 memtile blocks; budget arm 8 herds of width 4, refuses 9; opened items 8, 9, 10 | [31](31-resident-tail-r1-record.md) |
| Phase F — items 4 and 3-portable | advanced 2026-08-11 | see the F row; host suite 103/103 → 231/231 in 17 modules, pin re-verified both directions | [25](25-mode-rebuilds-and-results.md) |
| R1 column census — item 10 | done 2026-08-12 | `shim→core`-only read 4/16 worst 1 where the truth is 7/16 worst 2; clause counts MM2S demand; literal re-derived from the 2026-08-11 13:28:03 aircc (sha256 `5cb08407`); control: 3 herd-direct streams refused, 12 `aie.packet_flow` | [31](31-resident-tail-r1-record.md) |
| Three silent-miscompile classes — items 8, 9, wall 6 | done 2026-08-12 | `air-split-l2-memref` one-symbol map (+2 more hardcodes, one giving wrong addresses); `air-shrink-memref-sizes-by-access` `<12288xbf16>` → `<3072>` under gets at 3072/6144/9216, `EXIT=0`; `getLockValuePair` `ceil(16/1) = 16` vs 4 BDs; `check-air-mlir` 492 → 497; suite 31/1/0; 10/10 | [16](16-compiler-changes.md) |
| H8 — automatic pipeline fusion | first cut done 2026-08-12 | declarative `air.pipeline_group`/`air.pipeline_stage`/`air.staging`, `air-fuse-pipeline-launches` opt-in via `air-opt`; byte-identical reproduction at four shape/variant combinations; 9 negative + 6 positive controls; 497/0; host 357/357; cannot express R1 (segment-scope `scf.for` + cross-stage alloc vs `IsolatedFromAbove`) | [16](16-compiler-changes.md) |
| G — unattended runner + CI | **DONE 2026-08-20** | G0 (08-12): `profiles.py` + `run_profile.py`, `run_status="skipped"` emitted for the first time, row counts in the manifest; devq 256 cold 347 s smoke 4/4, `tree_dirt_after_run: []`, RAPL 3390/3465 samples; CI 1 → 10 PR-safe tests by allowlist (devq 261 `Excluded=352 Passed=10`; guard fixed from `Total Discovered Tests: 10` to 362); four doc-10 behaviours dropped, sudo collapsed to `xrt-smi configure`. G1 (08-12, `869b8684`, `5d598bd2`): `resume.py` with per-rung ledger and row-digest audit, only `passed` reused, `SCHEMA_VERSION` stays 2; M5 corrected (sweep already run 36/36); `tinybert_512` devq 304 301 s vs 570-min estimate; host 357 → 409 in 20 (517/517 in 23 after merges). Decoder graph `coarse` (`4cbaedaf`, devq 359–361): `run_decoder_block`, `512x768x12h_causal` row, first walk 5/12 dirty audited as chain accumulation, `DECODER_STAGE_ATOL`, confirmation 12/12; fused-cast qkv ~1 % mean-rel under pre-norm input vs ≤5e-3 raw. All four modes on the decoder (devq 414). `KNOWN_REGISTRY_GAPS` empty | [25](25-mode-rebuilds-and-results.md) · [54](54-first-full-profile-and-decoder-families.md) |
| Goal 1 — sliding window | W1 done 2026-08-12; **parked 2026-08-21** | `-DWINDOW_LEN` bands `apply_causal_mask`; `W=0` sha-identical to the old object. Gate devq 262 (2048×2048 causal): `mean_rel_L1` 3.676e-2, `atol_required` 8.048e-2, 1.24× margin, 0/4194304; control 4.70 % mismatches, 5.3× over. Top-k gate accepted full-causal on 1 of 4 prompts (6 of 32 tokens agreeing); zero speedup (14.47 vs 14.06 ms, 56 % more scores discarded); transformers 5.10.2 ignores `config.sliding_window` on Llama; Gemma-3 weights are 12 KB license-gated stubs | [35](35-goals-1-and-2.md) |
| Wall 7's compiler fix — closed NEGATIVE | 2026-08-12 | an AIE2 BD has one acquire and one release field (2 writers / 2 readers / 1 slot forces `a₀ > 2ρ` and `a₀ < 2ρ`); v2 refuses MIMO by name; `--check-order` Petri-net check; `check-air-mlir` 499/0 (+1 MIMO lit); both fixes hit the 48-block cap | [31](31-resident-tail-r1-record.md) |
| Wall 7 / item 21 — located | 2026-08-12 (devq 308/309) | memtile staging buffer with `herd_x` writers on one counting semaphore; per-column staging A/B 5/35 → 35/35; pairing-dictionary residual 0.015 vs 0.66–0.84 on devq 306's dumps; fix does not compile at the gate shape (48-block limit); emulation lit 8 → 10 clauses | [31](31-resident-tail-r1-record.md) |
| Compiler — `air-to-aie` lock placement (item 23) | done 2026-08-12 (devq 300/305) | acquire at each put left the core's own writes unguarded; predicted 0.810 vs measured 0.81 out of sample; regression lit red pre-fix; 497 → 498; suite 32/1/0; 10/10; ladder 21/21 with passing rungs byte-identical. Devq 298/299 reported the pre-fix answer from a stale compiler | [31](31-resident-tail-r1-record.md) |
| Static legality + space size — item 26 | done 2026-08-12 | 115,343,360 → 3,721,772 legal (31×); 21 vs iron's 7 on the replication slice; 15,347 structures; 59 % priced not refused (a hard shim filter would delete 2,181,680); `run_mapping_space_tests.lit` | [16](16-compiler-changes.md) |
| Balance instrument — item 25 | done 2026-08-12 (`7d17ea15`) | 1,213 ERT entries: 1,208 measured / 5 counted / 0 modelled; iron's 67.9–70.9 GB/s not imported; `addnorm` column 0 demand 3 / budget 2 priced ×1.500; 259 repeats spread median 1.6 % / worst 42.2 %; back-solve devq 293: 655,360 B / 122.81 µs = 5.336 GB/s; 72 tests, 16 injected defects; host 357 → 429 in 21 | [31](31-resident-tail-r1-record.md) |
| Phase G — two walks into two roots | done 2026-08-14 (devq 352, 1110 s) | `compare_roots` read `manifest.json` vs the runner's `results_manifest.json` (items 15/16 silently inert) — fixed; drift medians `runlist` 1.79 % < `coarse` 2.27 % < `offload` 4.86 % < `fused` 11.36 % (p90 21.19 %, `fused`@1024 101.5 then 80.0, @512 39.8 → 40.4; n = 2) | [25](25-mode-rebuilds-and-results.md) |
| Cross-mode ordering at 1024 — re-walk | RESOLVED 2026-08-18 (devq 353; doc 32's conditions) | `fused` < `coarse` 8/8 (@1024 77.30/77.65 vs 82.78/78.41; margin 0.76 ms avg / 0.04 min); devq 352's inversion was warmup 1; all 8 (mode, length) absolutes down 13–26 % vs the post-flip trees (coarse@1024 105 → 78–83, runlist 151 → 130–137, offload 180 → 153–157, fused 101 → 77); bytes and vectors unchanged; roots `results/rewalk-doc32-w{1,2}` | [32](32-cost-decomposed-ladder.md) |
| Mapping selector — item 31 | bridge built 2026-08-14; **closed negative 2026-08-19** | `study/analytical_cost.py` prices on the declaration; `shim_mm2s_slots` 7 of 16 on R1; traffic reproduces doc 53 §6 to the byte (devq 338/340), rate labelled per term (5.336 GB/s measured, multi-port modelled, over-credits wide designs); licence and buildable set disjoint (C1 vs C2's 6.6 % at 4096 invisible); `RECORDED_MODE_POINTS` latencies down 9–24 %; slot authority unified `801b068c` (`peak_shim_mm2s_slots`) | [31](31-resident-tail-r1-record.md) |
| Phase G — emb-1024 width wall | done 2026-08-14 | `rows_per_call` never derived (default 8 ⇒ `64 | rows`); `norm_rows` 32 at emb 1024, 4 per core legal; `derive_rows_per_call` bounded above by the default, IR byte-identical at five shapes; `run_layer_norm_rows_tests.lit` 9/9. PR-safe CI target repaired (filter reached `/bin/sh` bare): 369 discovered, 357 excluded, 12 passed | [25](25-mode-rebuilds-and-results.md) |
| Goal 2 — step 5's blockers cleared | 2026-08-14 | `gguf_q4_0.py` self-test 10 → 13 legs; `bartowski/SmolLM2-1.7B-Instruct-Q4_0.gguf` ungated, histogram `{Q6_K 1, F32 49, Q4_1 3, Q4_0 165}`; `8192 % 2048 == 0`, `32 × 64 == 2048`, full MHA (`head_count_kv 32`); embedding from the bf16 checkpoint (`awq_pack.py:270-277`); rms/rms bands q4_0 0.0828–0.0853, transcoded q4_1→q4_0 0.1109–0.1124 (out of family), requantized 0.0869–0.0884 | [35](35-goals-1-and-2.md) |
| Figure tier — item 11(b) | done 2026-08-14 | `study/plots.py` over schema v2; packages under a full-freeze constraints file, `pip freeze` diff empty, `make verify` PASS after; v2 had silently broken `ladder_report` on 15 of 19 trees (`read_rows_compatible`); decomposition remainder 84–89 % for `fused` is uninstrumented time; suite 519 in 23 → 539 in 24 | [25](25-mode-rebuilds-and-results.md) |
| `runlist`/`coarse_c3` attention UNSCALED | found and fixed 2026-08-19 (`99db5808`); ceiling tightened | no `1/sqrt(head_dim)` since the runlist front landed; relative error 3.7–9.8e-2 (25×) inside the 1e-3 absolute ceiling; device: pre-fix corr 0.9993 unscaled / 0.45 scaled, post-fix 0.9996 scaled / 0.44; `attn_context` 3.72e-2 → 1.393e-2 @512, 4.48e-2 → 1.466e-2 @1024 (`offload` 1.36e-2); ceiling 1e-3 → 4e-4 (honest max 3.165e-4, defect 4.631–5.706e-4; 1.26× / ≥1.16×); devq 363–364, 380; `results/decoder-gpt2-first-walk/` | [25](25-mode-rebuilds-and-results.md) |
| Goal 2 — quantization | steps 1–4, 6 done 2026-08-12; **step 5 DONE 2026-08-19** | q4_0 = shipped kernel with `z ≡ 8`; devq 257 corr 0.999996, both controls failing; symmetric variant bit-identical (0/2048), 4.750 → 4.500 bits/weight; 46.50 MiB per weight pass (48.00 idealized; 165 of 168 linears Q4_0, three `ffn_down` Q4_1); compute 1.6× at `gs=32`. Step 5 (`16ae3b22`, `4cddf4fe`; devq 372–379): `smollm2_1_7b_int4/` `make verify` PASS vs plain HF bf16, 11.1 decode tok/s; two shared defects fixed (EOS-count prompt length; gs=128 micro-kernel restaged before every link); 3/3 shared-driver regression | [35](35-goals-1-and-2.md) |
| Phase G — two decoder families | done 2026-08-20 (devq 429/433, 430/432) | `gpt2_512` 4/4 first walk (@1024 `fused` 55.2, `coarse` 57.8, `runlist` 114.1, `offload` 130.7 ms); `gpt2_medium_1024` 2/4 then 4/4: `coarse` `atol_required` 4.869e-1 (2 of 1,048,576), `fused` 5.412e-1 (40) vs the 4.5e-1 ceiling measured at one width; mean-rel coarse 4.95e-2, fused 5.57e-2, offload tail 2.40e-1; `decoder_stage_atol(hidden)` with `1024 → 6e-1`; confirmation devq 432 (115.2 / 122.6 / 218.9 / 254.4 ms) | [54 §4](54-first-full-profile-and-decoder-families.md) |
| Phase G — the `full` profile, gate MET | 2026-08-20 (w1 devq 427/431/434, w2 devq 435) | w1 session 1 (1902 s, `MemoryHigh=20G`) 20/10/6 `complete: False`; w1 final 21 measured + 15 derived skips + 0 failed; w2 one session 419 s warm, `results/g-full-baseline768-w2/` — **cite w2** (w1 partly from a dirty tree at `f3f13f31`, w2 clean at `964dfbc1`); `compare_roots` OK, drift `runlist` 0.61 % < `fused` 0.72 % < `coarse` 0.93 % < `offload` 1.58 %, p90 ≤ 4.29 %, one `coarse` rung 18.4 %. Firsts: `coarse` 717.8 / 1565.4 ms at 8192 / 16384 (`sync` 781 / 1550), `offload` 2226 / 6938, `fused` 22.0 at 256, `coarse` 27.6; `offload` 13.45 GB vs `coarse` 1.01 GB per layer at 16384. Skips derived by `ast` (`FA_PARALLEL_SEQ` 256, `ATTN_GEMM_SEQ_MULTIPLE` 512, `softmax_fits_l1`), `run_profile_bounds_tests.lit` skip ⇔ refuses; reproduces devq 427/431's 7/6/5/3. `runlist` @8192 empty aircc error (rc dropped) → softmax `SOFTMAX_ROWS_PER_CALL = 2` = 96 KiB at 8192 (48 KiB at 4096) → `derive_rows_per_call`, `run_softmax_rows_tests.lit` 9/9 (allowlist → 13); `runlist` @8192 1762.0 ms (`sync` 828, devq 431), 16384 refused by name in 1 s | [54](54-first-full-profile-and-decoder-families.md) |
| Fused decoder re-execution wall | done 2026-08-20 (`03402cc1`; `14209f71`'s frozen-BD framing retracted) | causal FA kernel's uninitialized L1 Q-block counter never wrapped; fix wraps modulo `num_lq_iters × NQ` in both NPU2 builders (`attn_npu1.py` patched, unverified). Prediction four clauses met (devq 413: three dispatches 12/12, corr 0.9995 / 0.438, d2 == d3); falsifier devq 426: dispatch 2 FAIL, 67,687 mismatches, corr 0.4382, exit 2; decoder walk 4/4 (devq 414); suite 35/1/0 (devq 422); 505/0 (devq 412); models 8/11 (devq 416 + 425). `run_npu2_fused_decoder_reexec_peano.lit` | [16](16-compiler-changes.md) |
| Success criterion 3 — iron adapter | done 2026-08-20 | `study/iron_adapter.py` key matched 0 of 162 rows (shape key vs family id) → translated via `cases.FAMILY_SPECS`; `validate_port` checks seven shape fields, reads no latency (pinned against a 10⁷× gap); `results/iron-validation-20260820/` 6/6, 3/3, 3/3, 0 disagreements; host 584. iron `baseline_768` (`c885c1e4`, 2026-08-01, pmode unrecorded): `hybrid` 15.1 / 21.1, `runlist` 24.5 / 29.3, `offload` 10.9 / 27.4 ms at 512 / 1024 vs this port (devq 353) `coarse` 40.8–43.1 / 78.4–82.8, `runlist` 70.7–73.3 / 130.0–137.3, `offload` 98.8–101.7 / 153.2–157.3 — 2.7–9×, NOT a result (timing region, different modes, no pmode) | [54 §5](54-first-full-profile-and-decoder-families.md) |
| The `ggml-hexagon` study | done 2026-08-20 (`63a8b95c`) | source `6503355df0eb`; PR #25085 Llama-1B Q4_0 4,028 tok/s prefill / 54 decode on S26+ (~2.4× / ~4.4× this repo's padded bf16) | [55](55-hexagon-llama-cpp-lessons-for-xdna2.md) · [56](56-full-model-mixed-precision-study-plan.md) |
| The `llms/` decode token decomposed | done 2026-08-20 (devq 436, 444, 440, 445–449) | Qwen3-0.6B bf16 89 ms/token = 28 × (`o_gemv_ffn` 1.53 + `rms_qkv` 1.03) + `lm_head` 9.7 + CPU attention 3.4; 145 ms at ~1,900 context (CPU attention 54 ms); GEMVs 8–15 GB/s vs the LM head's 33; boundary probe 9.96 / 12.23 / 9.12 ms (19 × 8192 vs 38 × 4096, 319 MB; `m_input = 8` control 120–165 µs); 327 boundaries ≈ 38 %; Llama-1B int4 65 ms with a 14.9 ms bf16 LM head, int4 GEMVs 4–14 GB/s; int4 driver existed at 17.8 tok/s (READMEs fixed); band 41–58 ms/token; `results/hexagon-opt-20260820/` | [57](57-inference-path-optimizations-from-hexagon.md) |
| LM-head re-execution defect — found | OPEN 2026-08-20 (devq 446–448) | 19 × 8192 ELF back-to-back: partition 0, rows ≥ 64, max diff 3.4 of output scale, exact after any other ELF; 38 × 4096 exact; persists at repeat ~127 (`m_input 8`, 14–107 bad rows) — follows launch size, not repeat count; production never adjacent, top-5 gate blind | [57 §1.4](57-inference-path-optimizations-from-hexagon.md) |
| Pending install item | done 2026-08-20 | `install-xrt` backend prints `aircc compilation failed (returncode 1):` | [15](15-environment-notes.md) |
| Launch-boundary cost isolated | done 2026-08-21 (devq 450–451) | 106–108 µs per configuration: additive 108.1; tiny-only slopes 106.4 / 106.1; fit 106.3 + 146 µs fixed per `xrt.run`; near-empty launches ~4 µs own work; `t1 = 251 µs` floor for a lone ELF; production head streams 40.8 GB/s between boundaries | [57 §1.5](57-inference-path-optimizations-from-hexagon.md) |
| LM-head defect gated, then avoided | gated 2026-08-21 (devq 452); clean 7/7 on the new head (devq 482/484) | production ELF 5 of 7 wrong (back-to-back ×2, after 0.5 s idle, with a new input; bad rows from row 24, partition 0); `lm_head_reexec_gate.py`, `make check-lm-head-reexec`, `run_npu2_lm_head_reexec.lit`, XFAIL dropped; family avoided, not understood | [57 §1.5](57-inference-path-optimizations-from-hexagon.md) |
| `m_input = 8` on the LM head | landed 2026-08-21 (devq 453–455) | verify 2/2; `lm_head_gemv` 9.77 / 9.79 → 8.79 / 8.79 ms (−1.0, predicted −0.85) | [57 §5](57-inference-path-optimizations-from-hexagon.md) |
| O2 — runlist pairs | measured small 2026-08-21 (devq 456/457/460) | 57 → 30 submissions; layer loop 2560 / 2564 vs 2614 / 2637 ms → −2 ms/token after `plan_pool` memo + preload; prototype removed 2026-08-22 (tag `pre-cleanup-20260821`) | [57](57-inference-path-optimizations-from-hexagon.md) |
| O1 first cut — QKV at 4 launches | landed 2026-08-21 (devq 461–463; `5ad7af60`) | probe 1.125 → 0.680 ms/layer (111 µs per removed boundary); `rms_qkv` 1.03 → 0.62, per-layer 2.92 → 2.52, −11.5 ms/token; verify 2/2; remaining: RMSNorm prologue + QK-norm/RoPE epilogue (column-owns-heads GEMV) for 4 → 1–2 | [57 §5 item 5](57-inference-path-optimizations-from-hexagon.md) |
| O1 + `m_input = 8` → Qwen3-1.7B | landed 2026-08-21 (devq 465–470; `61a40a00`) | probe 1.326 → 0.873 ms/layer; `rms_qkv` 0.80; LM head 16.4 → 15.9–16.2; ≈127–130 ms/token; `shared/infra/decode_qkv4.py`; Qwen3-4B not ported (verify oomd-deferred) | [57](57-inference-path-optimizations-from-hexagon.md) |
| Doc 56 H0 — the planner | done 2026-08-21 (`859dee7f`, `decba364`) | `llms/shared/plan/`, `PLAN: 10/10` in `run_seam_tests.lit`; golden JSON for Qwen3-0.6B / Llama-1B; solver vs registry 61 / 136 identical, mismatches classed; finding: 10 × 16384 instead of 19 × 8192 | [56 §4](56-full-model-mixed-precision-study-plan.md) |
| LM head 19 × 8192 → 9 × 16384 + 4480 | landed 2026-08-21 (devq 476/478/479; `93ef7040`) | probe −1.10 ms exact; `lm_head_gemv` 8.79 → 7.90 ms; ELF 8.3 → 2.6 MB; `build_lm_head_gemv_module(parts=…)`; token 89 → ~75 ms | [57 §5 item 5b](57-inference-path-optimizations-from-hexagon.md) |
| O4 — int4 LM head | closed for now 2026-08-21 (devq 488) | one-launch form hits the `push_queue` repeat cap at 4 iterations; ten-launch 7.37 vs 7.82 ms = −0.46; int4 GEMV 11 GB/s; stitched probe had a static correctness bug, not chased | [57 §5 item 6](57-inference-path-optimizations-from-hexagon.md) |
| Mixed-partition head → Qwen3-1.7B | landed 2026-08-21 (devq 485, 487; `fd7e17b8`) | 19 → 10 launches at K = 2048; `lm_head_gemv` 15.9–16.2 → 14.6–14.95 ms; 7.70 / 7.83 tok/s ≈ 128–130 ms/token | [57](57-inference-path-optimizations-from-hexagon.md) |
| Cleanup cluster A — root artifacts | done 2026-08-21 (`3d08333b`; decisions block `c4dd1f2a`) | 16 files → 0 (12 `*.o`, `air.mlir`, `coarse_cache/`, one `.o.tmp`); root ignore rules anchored; +lines 131,003 → 130,953; Codex PASS, 0 blocking (`results/cleanup-20260821/review/verdict-A.json`). Baseline gates devq 489 (host legs), 491 (six models); devq 490's suite run was contaminated by the `.o.tmp` delete | [15](15-environment-notes.md) |
| Cleanup cluster B — `agents/` | done 2026-08-21/22 (`cbd2858e` .. `244fefe9`, five review rounds) | port-loop harness and 19 one-shot probes retired (at the tag); `agents/` 15,140 → 1,714 lines; kept `devq.sh`, `devq-selftest.sh`, `audit-gemm-object-links.py`, `doctor.sh`, `bootstrap-*.sh`, `port-loop/lib-env.sh` (out-of-repo consumer), `schema/review.json`, `probe_r1_rung.py` + `probe_r1_emulate_shape.py`. Re-homed gates: H9's `herd_x=8` fixture → `addnorm_multitrip.py` + lit + make target; `phase_e_checks` distinguish → `study/distinguish.py`, live in `run_profile.gate()`, fails closed (NaN, failed rows, duplicate lengths, skip identity); `require_turbo` refusing-branch test. Accepted retirements: `gate-h` throughput floor, opcheck-tree clauses. Gates: host 587 → 611 in 27 modules, seam 35 + 10, suite 38/1/0 (devq 493), six-model leg unchanged | [15](15-environment-notes.md) |
| Cleanup cluster C — plan docs | done 2026-08-22 (`1fa9a78f` + review fixes `f12f198a`) | 82 → 18 docs, 29,766 → 8,871 lines; six parallel consolidations under one preserve-every-number brief; 0 broken links; 20 outside citations repointed; one Codex round: six blocking, all fixed (a retracted two-trip rule, four retired-probe commands, a dead harness item, a stale two-gates claim, two code strings building the old 05a path) | [this README](README.md) |
| Cleanup cluster D — `transformer_layer/` | done 2026-08-22 (`db0ba4f6`) | no code deleted: builders IMPORT `llms/shared/builders`; every `study/` module is consumed by `run_profile`/`run_ladder`/`manifest`, a lit, or the H0 planner; `coarse_c2/c3` are live modes; lit dedupe would weaken shape gates. tl README 1,688 → 976; [16 §11](16-compiler-changes.md) corrected (plane-major was the spec; builder ships `[rows, 2, cols]`); the doc-number table below. Codex review stopped externally, no verdict | — |
| Cleanup cluster E — `llms/` | done 2026-08-22 (`d2d652d0`) | the O2 decode-runlist prototype removed (275 lines + 4 hooks; [57 §5](57-inference-path-optimizations-from-hexagon.md) keeps the measurement); qwen3 0.6B/1.7B duplication is `main`'s convention; golden and registry JSON stay pretty-printed drift guards; Goal 1's W1 kernel + lits stay. Gate devq 494: verify PASS, reexec 7/7. Codex review stopped externally, no verdict | — |
| Cleanup cluster F — compiler | closed 2026-08-22, no change | docs landed in C as [16](16-compiler-changes.md); the 33 `mlir/test` files are paired positive/negative regression lits, one pair per defect class — nothing to merge without weakening a gate; compiler code untouched (operator) | — |
| Item 8 — the iron latency gap, attributed | done 2026-08-22 (devq 503 / 504; `results/iron-gap-20260822/`) | Both clocks have the same shape; ours runs the per-boundary comparison INSIDE it (~22–27 ms of a forward) and iron's does not. Like-for-like at `baseline_768`/512, same session, Turbo, `--no-stage-stats`: iron hybrid **15.07** vs ours coarse **17.25** (1.14×) / fused **14.83** (0.98×); runlist 24.29 vs **46.30** (1.91×, host BO traffic + per-forward weight re-hashing); offload 12.11 vs **77.44** (6.4×: `device_ms` 60.5 over the SAME 30 GEMM dispatches and the same linear-device / non-linear-host partition — ~2.0 ms per submission vs ~0.4; which of the 30 carry it is not instrumented, so this residual is NOT attributed). Default numbers unchanged; two operator questions in the queue | [54 §5](54-first-full-profile-and-decoder-families.md) |
| Cleanup close — final gates | done 2026-08-22 (devq 495/496/497) | `check-air-mlir` 505 / 7 xfail / 7 unsupported (= baseline); study host **611/611** in 27 modules (was 587); seam 35/35 + 10/10; tl suite **38 / 1 / 0** (39 lits, was 38); six-model verify **6/6**. Branch squashed per cluster; not pushed | results/cleanup-20260821/ (local) |
| Operator decisions 2026-08-21 | recorded (`c4dd1f2a`) | R1 → supertiles (finished block per execution, `down_K = 96`; two-form comparison first); J1 closed; Goal 1 parked; big three not run (8/11 permanent); iron gap at devel HEAD `cc7083f` approved; docs 01–12 demoted; Goal 2 step 5 already done | this file |

## The work queue

Ordered as the operator decided. The cleanup is done (2026-08-22); what it removed, per cluster,
against `main` (lines the branch adds; `results/cleanup-20260821/scoreboard-step{0,7}.txt`):

| cluster | before | after | what went |
|---|---|---|---|
| A root artifacts | 80 + 12 binaries | 0 | 12 `*.o`, `air.mlir`, `coarse_cache/`, one `.o.tmp` |
| B `agents/` | 15,140 | 1,714 | the port-loop harness and 19 probes; two gates re-homed stronger (H9 lit, `study/distinguish.py`) |
| C plan docs | 29,742 | 8,871 | 82 → 18 docs |
| D `transformer_layer/` | 57,886 | 58,254 | README −712; +1,080 for the re-homed gates and their tests; no code deleted |
| E `llms/` + kernels | 18,746 | 18,454 | the O2 prototype |
| F compiler | 8,854 | 8,854 | untouched |
| other | 555 | 560 | `.gitignore` anchors |
| **total** | **131,003** | **96,707** | **−34,296 (−26 %)**; host suite 587 → 611 tests, tl suite 38 → 39 lits |

Then:

| # | Item | Gate |
|---|---|---|
| 8 | ~~**IRON** — latency-gap attribution~~ **done 2026-08-22** — the gap is the per-stage comparison inside our clock (parity for the device-resident modes without it); `offload`'s per-submission residual (~2.0 vs ~0.4 ms over the same 30 dispatches) stays unattributed; [54 §5](54-first-full-profile-and-decoder-families.md). Two operator questions follow (measurement model; rule S1 per-plan hashing) | done |
| 9 | **R1** — the supertile first increment: finished block per execution (`down_K = 96`, meets the ≥7 wedge) vs accumulate-across-executions (in-box, `ffn_accum`), builder/sequence work, one devq measure per form | the first-ever measured resident-tail latency and byte figure; the faster form becomes R1's route |
| 10 | **REEX** — the re-execution family matrix: launch-0 kind × GEMV geometry × dispatch index, a BD dump per configuration, over [57 §1.5](57-inference-path-optimizations-from-hexagon.md)'s seven rows. Prerequisite for shipping any new multi-launch form; until then every new form runs the gate shape (`fused_reexec_gate.py` / `lm_head_reexec_gate.py`) before shipping | every row classified hang / wrong / clean with a named mechanism |
| 11 | **5b** — column-owns-heads GEMV kernel: QK-norm + RoPE epilogue, QKV stage 3 → 1–2 launches, ~6 ms/token on 0.6B (each column streams whole 128-row heads; the L2-staged tile distribution cannot). New core kernel | `make verify` + the reexec gate shape from 10 |
| 12 | **LLAMA** — Llama-1B mixed-partition head port (2,816 pad rows, ~0.36 ms) | rides 11's verify leg |
| 13 | **6b** — doc 56 H1a: model adapter + runner, schema v3, the kernel-scaling curve keyed by `Plan.sha` — last, so the curve is measured on the settled kernels from 11/12 | [56 §4](56-full-model-mixed-precision-study-plan.md) |

Latent, not scheduled: queue row 30 (the `w_down` feed's readers unbound at `herd_x ≥ 2`; `--check-order`
OVERWRITE, present in `D2p`/`H4p`/`D4p` which pass 5/5; null at `herd_x = 1`, devq 327) — designed out
by the supertile model; the `down_K ≥ 7` wedge (O6) — met by the finished-block form; lifting a derived
skip into a rung (smaller attention tiles below 512, a column-chunked softmax for `runlist` at 16384,
[54 §3](54-first-full-profile-and-decoder-families.md)); `memcpy_bandwidth` (a multi-core AIR memcpy
operator that does not exist, which is why `roofline/run.py` stays unported); `attn_npu1.py`'s
unverified parity patch; a real `initial_value` through `air-to-aie` for the FA boot counter.

Closed and not worth re-opening (each has its row above or its record in the linked doc): queue items
1 (offload shared path at 4096), 1b (the pmode anomaly), 2, 3, 4, 5, 6a, 6b, 6c, 7, 8, 9, 10, 11 (DEFER:
11(a) is 11(b)'s prerequisite), 12 (Phase G met, Goal 2 done, Goal 1 parked), 13, 14 (OOM did not
reproduce: 10.53 / 12.57 GiB peak, devq 255), 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28
(a: rotation skew, devq 398; b: unawaited shim tasks, `c634f735`, devq 403; `runtime_loop_tiling_sizes`
excluded directly at devq 331; `--omit-pingpong L2` 4/4 at devq 332; the `PREDICTION-ROT` sha
`0b18a1c0` falsified, `PREDICTION-PP` `e7bc2dd4` held), 29 (closed negative — read it before proposing
anything in that area), 31 (closed negative), J1, the big-three leg, and the 2026-08-20 evening list
items 1–6. The `prepare_fused` half of `coarse`'s extraction was deliberately not done (no cell at
2048+ uses a stitched tail).

## Provenance

The source is `iron` commit `1e014c1` "Add transformer-layer execution-strategy studies" (145
files, ~58.6k insertions), validated there by an 888/888-job suite run; the latency-gap work targets
iron's devel HEAD `cc7083f`, one commit past it. This plan was reviewed by Codex before approval;
findings that materially changed it are marked `[Codex]` in the phase documents, and the 2026-08-20
docs carry their review corrections marked `[per Codex review]`. Sealed predictions
(`PREDICTION-MAXQ.md`, sha256 `90b92618…`, clauses 1 and 3 still untested; `PREDICTION-FUSED-REEXEC`,
all four clauses met; `PREDICTION-ROT`, falsified; `PREDICTION-PP`, held) are recorded with their
outcomes in [31](31-resident-tail-r1-record.md) and [16](16-compiler-changes.md). The 82-doc tree
before consolidation — every retired phase spec, research reading, prediction file, the port-loop
harness and the 19 retired probes — is at git tag `pre-cleanup-20260821` (`60e287d3`).

### Where the old doc numbers went

Code comments and lit headers still cite the pre-consolidation numbers ("doc 31b §3.6", "doc 09").
Resolve them here; the retired text itself is at tag `pre-cleanup-20260821`.

| old | now in | old | now in |
|---|---|---|---|
| 00, 01, 04, 05, 05a, 06, 06a–d, 07, 07a, 07b, 08, 08a–e, 09, 10, 11, 12, 13, 14 | [01](01-original-plan-superseded.md) | 16, 17, 18, 19, 20, 21, 22, 24, 48, PREDICTION-FUSED-REEXEC | [16](16-compiler-changes.md) |
| 25, 26, 28, 29, 30, 33, 34, 38, 50, 51 | [25](25-mode-rebuilds-and-results.md) | 31, 31a, 31b, 37, 47, 49, 52, 53, PREDICTION-28A-*/28B/MAXQ | [31](31-resident-tail-r1-record.md) |
| 35, 36 | [35](35-goals-1-and-2.md) | 39, 40, 41, 42, 43, 45, 46 | [44 §Bibliography](44-mapping-frameworks-synthesis.md) |
| 55a, 56a, 57a (verbatim Codex reports) | their parents 55, 56, 57 | unchanged | 02, 03, 05b, 15, 23, 27, 32, 44, 54, 55, 56, 57 |
