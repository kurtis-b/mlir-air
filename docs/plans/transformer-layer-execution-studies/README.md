# Transformer-Layer Execution Studies — Port Plan

Porting the transformer-layer execution-strategy studies from the AMD IRON repository
(`iron`, commit `1e014c1`) into MLIR-AIR, then building two follow-on capabilities on the
resulting measurement harness: SOTA model coverage via sliding-window attention, and
quantized inference.

These documents are the working plan. They live in the repository rather than the published
docs site (`mkdocs.yml` excludes `plans/`), because they describe work in progress rather than
how to use MLIR-AIR.

## Read in this order

| Doc | What it covers |
|---|---|
| [00-context-and-goals.md](00-context-and-goals.md) | Why this port, what is being ported, success criteria |
| [01-port-inventory.md](01-port-inventory.md) | Per-artifact triage: port / adapt / rewrite / drop |
| [02-porting-conventions.md](02-porting-conventions.md) | **How iron code is refactored into MLIR-AIR house style.** Reviewable checklist |
| [03-measurement-model.md](03-measurement-model.md) | The execution-boundary taxonomy, the dispatch vector, CSV schema v1 |
| [04-phase-a-kernels.md](04-phase-a-kernels.md) | AIE2P device kernels |
| [05-phase-b-runtime-seam.md](05-phase-b-runtime-seam.md) | Runlist aggregation + BO liveness pooling |
| [06-phase-c-operators.md](06-phase-c-operators.md) | The six new operators as AIR builders — **overview**; the four sub-phase specs are [06a](06a-phase-c1-gate-and-small-operators.md) · [06b](06b-phase-c2-qkv-proj-and-ffn.md) · [06c](06c-phase-c3-mha-out-proj.md) · [06d](06d-phase-c4-coverage-sweep.md) |
| [07-phase-d-block-integration.md](07-phase-d-block-integration.md) | Single-block integration gate — **overview**; the two sub-phase specs are [07a](07a-phase-d1-operators-at-baseline-768.md) · [07b](07b-phase-d2-block-integration.md) |
| [08-phase-e-execution-strategies.md](08-phase-e-execution-strategies.md) | The four execution strategies |
| [09-phase-f-study-harness.md](09-phase-f-study-harness.md) | The seven measurement studies |
| [10-phase-g-unattended-runner-and-ci.md](10-phase-g-unattended-runner-and-ci.md) | Unattended suite runner, CI wiring |
| [11-goal-sota-sliding-window.md](11-goal-sota-sliding-window.md) | Goal 1 — sliding-window / local-global attention |
| [12-goal-quantized-inference.md](12-goal-quantized-inference.md) | Goal 2 — quantized inference |
| [13-verification-and-acceptance.md](13-verification-and-acceptance.md) | Every gate, in one place |
| [14-the-port-loop-harness.md](14-the-port-loop-harness.md) | The automated driver: how it works, how to run a phase, what it learned the hard way |
| [15-environment-notes.md](15-environment-notes.md) | Toolchain state and the setup traps that silently hollow out hardware gates |
| [16-compiler-work-and-remaining-essence.md](16-compiler-work-and-remaining-essence.md) | **Start here for what remains.** Tranche H (compiler) and tranche J (the study), the corrected root cause, and what AIR automates versus what iron writes by hand |
| [17-phase-h-compiler-hardening.md](17-phase-h-compiler-hardening.md) | Phase H spec plus its attempt-by-attempt record — including two of its own claims that measurement falsified |
| [18-phase-h1s-skip-not-refuse.md](18-phase-h1s-skip-not-refuse.md) | H's correction, run fresh rather than resumed: the safety proof declines to *transform*, never to compile |
| [19-phase-j1-collapse-norm-dispatches.md](19-phase-j1-collapse-norm-dispatches.md) | J1 — blocked, with both walls it hit measured and recorded |
| [20-phase-h9-fuse-through-parallel.md](20-phase-h9-fuse-through-parallel.md) | H9 — the packet fusion that only ever worked on one column, and what it took to fix |
| [21-phase-j7a-norm-tail-pipeline.md](21-phase-j7a-norm-tail-pipeline.md) | J7a — the norm-tail pipeline. **The first working piece of the dataflow goal** |
| [22-phase-j7b-accumulator-ring.md](22-phase-j7b-accumulator-ring.md) | **The next phase, staged and not started.** Partial sums that never leave the chip, with the compiler forming the ring |
| [23-rules-and-open-items.md](23-rules-and-open-items.md) | **Start here.** The rules that govern later work, and the open items nobody has claimed |
| [24-phase-h10-non-constant-bd-offsets.md](24-phase-h10-non-constant-bd-offsets.md) | **Substance verified, tamper baseline not clean.** The silent miscompile J7b lost a session to, located: an unchecked `std::optional` deref in `air-to-aie` |
| [25-first-study-result-sequence-ladder.md](25-first-study-result-sequence-ladder.md) | **`[retracted 2026-08-08]` It measures the current implementations, not the study's four modes.** This entry used to promote it as "the first study RESULT rather than a capability" and as a `fused`/`coarse` crossover. That promotion is withdrawn. The measurement itself stands — 16 rungs, walked twice on hardware, every rung validated — but the four things it ranks are the four *implementations* named `runlist` / `offload` / `coarse` / `fused`, and none of them yet matches the corrected taxonomy in [03 §The taxonomy](03-measurement-model.md). Two of the four still run attention on the host, which is what its slopes split on. Read it as a measurement of what is built today |

## Status board

Update the status column as phases land. A phase is `done` only when its gate passes — see
[13-verification-and-acceptance.md](13-verification-and-acceptance.md).

| Phase | Gate | Status |
|---|---|---|
| A — AIE2P kernels | Every kernel compiles to `.o` with Peano; compile-only lit passes | **done** 2026-08-04 (18 min) |
| B — runtime seam | Multi-ELF runlist on hardware: numerically identical to sequential, lower latency | **done** 2026-08-04 (362 min) |
| C1 — gate mechanism + small operators | `opcheck.py` and its fault-injection negative control; `causal_mask`, `addnorm`, `layer_norm`, `elementwise_add` pass on hardware | **done** 2026-08-04 (61 min) |
| C2 — `qkv_proj`, `ffn` | Both pass full-output `np.isclose` at registry tolerance vs an FP32 reference | **done** 2026-08-04 (45 min) |
| C3 — `mha_out_proj` | Passes at the registry's FlashAttention tolerance, causal and non-causal | **done** 2026-08-04 (68 min) |
| C4 — coverage sweep | The 36 `baseline_768` shapes resolve through `gemm_config()`; registry rows written; ten shipped models still pass `make verify` | **done** 2026-08-04 (504 min + 66 min re-run) |
| D1 — operators at `baseline_768` | Every operator passes `opcheck` at the `baseline_768` widths, including the pre-add `addnorm` | **done** 2026-08-05 (11 min) |
| D2 — block integration | One full transformer layer matches the torch reference on hardware | **done** 2026-08-05 (156 min) |
| E1 — unblock the ladder | `(method, tile_n)` names separate; `ffn` passes at a second ladder point; ten shipped models still verify | **done** 2026-08-05 (79 min) |
| E2 — `coarse` + instrumentation | `coarse` matches at full scope behind a measured dispatch vector | **done** 2026-08-05 (38 min) |
| E3 — `offload` | `offload` matches, and aggregates nothing | **done** 2026-08-05 (55 min) |
| E4 — `runlist` | `runlist` matches, with more runlist entries than `coarse` | **done** 2026-08-05 (91 min) |
| E5 — `fused` + distinguishability | `fused` matches, and all four modes' dispatch vectors separate as the taxonomy predicts | **done** 2026-08-05 (62 min) |
| H — compiler hardening | `gate-h.sh` four legs: build + install, `check-air-mlir`, transformer-layer suite, `make verify` × 10 | **halted** 2026-08-06 at `confirm/3`, and **superseded by H1s** rather than resumed — its spec was corrected after the halt, so its fingerprint baseline no longer describes what is being gated |
| H1s — skip, do not refuse | `gate-h.sh` **five** legs: build + install, `check-air-mlir`, transformer-layer suite, **decode throughput vs a recorded floor**, `make verify` × 10 | **done** 2026-08-06 (109 min) |
| J3 — sequence ladder | Four modes walked across 512/1024/2048/4096 with a comparison that survives a second walk | **done** 2026-08-08 — 16/16 rungs twice. **A crossover:** `fused` leads at 512 and 2048, `coarse` at 4096; slopes split on attention placement (device 1.03–1.17, host 1.23–1.27), not on dispatch structure. The 1024 ordering did NOT survive the second walk and is recorded as indistinguishable. See [25](25-first-study-result-sequence-ladder.md). **`[2026-08-08]` The four things it ranks are the four current implementations, not the four modes the study means** — see the taxonomy-correction row below |
| Taxonomy correction — what the four modes isolate | none; a specification correction, not a phase gate | **recorded** 2026-08-08. The study's author corrected the axis to **reconfiguration cost against DRAM traffic**: `runlist` = every operator individually **on the device**, nothing on the host; `offload` = reconfiguration minimized by dynamic partitioning — one xclbin, one instruction stream, matmul loop bounds from a **runtime parameter**, with **all linear** operators (six projections + both attention matmuls) on the NPU and **all non-linear** (softmax, both LayerNorms, GeLU) on the host; `coarse` = a per-workload **blend** of `runlist` and `fused`; `fused` = whole layer on the array, one xclbin, only the layer input and output crossing DRAM. **No mode meets its corrected definition today** — [03 §What is implemented instead](03-measurement-model.md) sizes each of the four gaps, and every measurement recorded so far, [25](25-first-study-result-sequence-ladder.md) included, ranks implementations rather than the taxonomy. The other documents in this directory still use the superseded "who sequences the work" framing; rewriting them is **deliberately deferred** until the corrected mechanisms are real |
| J1 — collapse the norm dispatches | transformer-layer suite, then `coarse` `runlist_entries` ≤ 10 | **blocked** 2026-08-06, stopped by operator at `fix/1`. The collapse does not happen and cannot yet: multi-column multi-trip `addnorm` **silently miscompiles** (measured 4070/4096 at `herd_x=8`, 2 trips). Phase H's packet fix works only at `herd_x=1`, which is the only width its fixture ever ran. The guard is refined to the measured boundary instead of lifted (`52b57c8f`, `ef5e1cf1`); `coarse` stays at 131 entries. **[2026-08-07]** H9 fixed the miscompile; J1 is still blocked, now on shim **BD exhaustion at 6 trips** against a 64-trip target — it refuses loudly instead of corrupting silently. The route to the same collapse is J7a, which never enters the packet path |
| H9 — fuse packet put loops through `scf.parallel` | `gate-h.sh` five legs, plus a driver fixture variant at `herd_x=8` that must go from corrupt to exact | **done** 2026-08-07 (184 min) — `multicolumn` 3747+/4096 wrong → exact; 10/10 models; three review rounds each found a real defect in the combiner/token handling that the gate could not reach |
| J7a — norm-tail pipeline | transformer-layer suite; `mean_rel_L1` ≤ block's 1.688e-2; zero packet-typed channels | **done** 2026-08-07 (87 min) — 3.620e-3 at 4096×768, 4.7× under the bound; compiler-derived placement and depth; `layer_norm` itself improved ~25× as a side effect |
| J7b — accumulator ring | transformer-layer suite; the in-place accumulator dispatched; C DMAs hoisted out of the K loop | **done** 2026-08-07 (58 min) |
| H10 — non-constant BD offsets | `gate-h.sh` five legs, plus four objective clauses: an IV-dependent L2 offset refused by message, the SAME builder at 2 trips still compiling, a constant offset compiling, an L3-side moving offset compiling | **substance verified 2026-08-08; tamper check halted on documented changes** — `H GATE: PASS` all five legs (`check-air-mlir` 489/489, hardware suite, 11.44 tok/s vs a 9.43 floor, 10/10 models) and the objective check passed. The tamper check then halted on five gate files whose provenance is recorded below. The compiler fix is sound; the phase's *baseline* is not clean. See [24](24-phase-h10-non-constant-bd-offsets.md) |
| F — study harness | `execution-smoke-test` yields ≥1 `run_status=passed` row per measurement CSV | **in progress** on `exper/phase-f-study-harness` (a worktree, unmerged). **The gate itself passes on hardware over all four modes** as of 2026-08-08 — `smoke_gate` PASS, `manifest complete: True`, and all four distinguishability clauses hold on the measured vectors ([09](09-phase-f-study-harness.md)). Work items 1, 2, 6, 8 done, plus the runner, results I/O, gate and manifest. Items 3 (the ~19k-line plot/analysis tier), 4, 5, 7 remain; **item 3 is blocked** — matplotlib/pandas/seaborn are absent and must not be installed while gates run |
| Corrected `offload` — attention on device | `run_npu2_offload_peano.lit`, both recipes, at the corrected 30-dispatch boundary | **done** 2026-08-08. 10/10 stages clean, `submissions 30 entries 30 air 31 herd 91 sync 91 bytes 970457088`, negative control exact through the attention half. No registry write, no compiler work, no tolerance widened — `attn_context` 11.4× margin, `output` 1.73×. Costs 6.9× the DRAM traffic, which is the mode's result |
| `fused` build repair — SPECS row 4096 → 1024 | `run_npu2_fused_peano.lit`, both recipes | **done** 2026-08-08. The gate was **red and unrun**: the row was left at 4096 while the mode has always been bounded to 256..1024, so it raised before aircc. Now green at 1024 — 10/10 stages, `mean_rel_L1` 1.756e-2 at `atol_required` 5.813e-2. Its cross-mode `sync` comparison against `coarse` is **suspended**, not restated: the two rows are now at different sequence lengths |
| Backend-preset conflict — settled on hardware | none; a measurement that retracts [26 §4](26-mode-rebuild-feasibility.md) | **recorded** 2026-08-08. `runtime_loop_tiling_sizes` is **not inert**: `[2,2]` hangs `mha_out_proj` @4096 3/3, `[1,1]` passes 3/3, `omit_pingpong` irrelevant either way. Restores the conflict `fused.py` / `mha_out_proj.py` / `block.py` document, with a corrected reason. `agents/probes/probe_backend_preset_hardware.py` |
| Device `softmax` operator | `run_npu2_softmax_peano.lit`, clean + negative control, three shapes | **done** 2026-08-09. `builders/softmax.py` over the existing `softmax_streaming.o`; no kernel written. 512×512, 4096×768 and **64×4096** (attention width, where `rows_per_call` drops 8 → 2 on L1). `mean_rel_L1` 1.60–1.63e-2, `atol` 2.7–2.9× `atol_required`, plus a `mean_rel_L1_max` ceiling because a softmax row spans three orders of magnitude and an element-wise `atol` alone is loose at the bottom. Two corrections to [26 §5](26-mode-rebuild-feasibility.md) recorded there |
| Corrected `runlist` — every operator on device | `run_npu2_runlist_peano.lit`, clean + negative control | **done** 2026-08-09. **427 entries over 17 runlists, nothing on the host.** Per head `attn_scores` → `softmax` → `attn_output`, device-resident inside one submission; one submission per head is a memory bound (~800 MiB if batched, ~70 MiB per head), not a schedule choice. 10/10 stages clean, `submissions 17 entries 427 air 50 herd 488 sync 451 bytes 190513152`. In the end it never touched `builders/mha_attention.py`, so the `fused` serialization this table warned about was not needed |
| **The first result on the corrected axis** | none; a measurement | **recorded** 2026-08-09. `runlist` moves **190,513,152** bytes against `offload`'s **970,457,088** for the same layer — **5.1×**, produced entirely by where the softmax runs. `offload` puts it on the host, so every `[4096, 4096]` score matrix crosses DRAM twice per head; `runlist` keeps it on the array. Two modes differing in exactly the corrected taxonomy's variable — reconfiguration against DRAM traffic — rather than in attention placement, which is the confound every earlier comparison carried |
| `attention_path` retired as a covariate | none; a consequence | **recorded** 2026-08-09. With `runlist` on the device, **all four modes are**. The first sequence ladder's headline — slopes splitting on attention placement, host 1.23–1.27 against device 1.03–1.17 — **cannot be reproduced**, because no mode sits on the host side any more. `study/test_attention_path.py` now asserts that end state rather than the two-value invariant it was written with |
| Corrected `coarse` | its own phase | not started; a blend of `runlist` and `fused`, so both must be right first |
| G — unattended runner + CI | Full profile run completes with a complete `results_manifest.json` | not started |
| Goal 1 — sliding window | `make verify` passes with window-crossing prompts | not started |
| Goal 2 — quantization | Second quantized model passes a gate that exercises the quantized path | not started |

## `[2026-08-08]` Where things stand, for a session picking this up cold

**Read [26](26-mode-rebuild-feasibility.md) first.** The four execution modes were re-specified by
the author on 2026-08-08, all four implementations diverge from the new definitions, and doc 26
opens with **six things the plan had wrong** — three of which were blocking work that turned out not
to be blocked. Then [03 §The taxonomy](03-measurement-model.md) for what the modes now mean, and
[23](23-rules-and-open-items.md) for the rules that govern everything downstream.

> **`[2026-08-08, later the same day]` Two of doc 26's six are themselves now settled by hardware,
> in opposite directions.** Its **§4 is retracted**: `runtime_loop_tiling_sizes` is *not* inert, and
> the backend-settings conflict it declared false is real — `mha_out_proj` at `[2,2]` compiles and
> then hangs, 3/3, against 3/3 clean passes at `[1,1]`, with `omit_pingpong` irrelevant either way.
> Its **§6 is confirmed and fixed**: `make check-fused` really was red, and the row is moved to 1024
> and green. Doc 26 carries both corrections inline. The methodological lesson is §4's, and it is
> the same one this plan keeps paying for — a compile-only observation ("the lowered IR is
> identical") was turned into a hardware conclusion ("the knob is inert"), and the caveat against
> exactly that was written down in the same section and then not applied.

**The one-paragraph version.** The modes are no longer defined by *who sequences the work* but by
**reconfiguration cost against DRAM traffic**: `runlist` pays per-operator reconfiguration with
everything on device, `offload` minimizes it (one xclbin, N instruction streams — matching iron; all
LINEAR operators on the NPU, all NON-LINEAR on the host), `fused` eliminates DRAM traffic between
operators, and `coarse` blends `runlist` and `fused`. Every measurement recorded before that
correction — including [25](25-first-study-result-sequence-ladder.md)'s ladder and crossover — ranks
four implementations that are not those four modes.

**Two rules to know before designing anything.**

*A column has **two shim MM2S channels**, and the budget is per column **across the whole
segment*** — three stacked 8-wide herds put one tile of each into every column, so their L3 demands
add. Exceed two and AIR packet-multiplexes onto one queue. Keep every column at two or fewer
L3-facing streams; put the rest on L1→L1 channels, and pack co-indexed L3 operands into one strided
fetch. This explains why `fused`'s decomposed tail always ran 64 trips on 8 columns correctly, why
`addnorm` needed its one-trip guard, why J1's L2-staged weight failed, and why J7a works.

*Advance a staged buffer on the **L3** side, never on the L2 read.* `[2026-08-07]` An
induction-variable offset is materializable on an L3 operand (the runtime sequence programs it per
task) and **inexpressible** on an L2/L1 one (an `aie.dma_bd` offset is static). The compiler does
not say so — it dereferences an unchecked `std::optional` and emits a chain that repeats a stale
offset forever, which presents as a hardware hang with no compile-time signal. J7b lost a session
to it. See [23](23-rules-and-open-items.md) and [24](24-phase-h10-non-constant-bd-offsets.md).

**Where the live threads stand:**

1. **~~Corrected `offload` is unblocked and is the next thing.~~ DONE `[2026-08-08]`, and gated.**
   Both attention matmuls are on the device, tiles injected through `gemm_spec_fn`
   (`gemm_spec_source: registry+injected`); only the softmax between them, both LayerNorms and the
   GeLU are on the host. `run_npu2_offload_peano.lit` **passes on hardware**, both recipes: 10/10
   stages clean, 30 dispatch vectors, `submissions 30 entries 30 air 31 herd 91 sync 91 bytes
   970457088`, and the negative control still exact through the attention half. No registry write,
   no compiler work, no new operator, and **no tolerance was widened** — `attn_context` needs
   `atol` 8.800e-05 against the 1.0e-03 the boundary allows (11.4×) and the layer output 5.788e-02
   against the 1e-1 ceiling (1.73×, the widest of the four modes).

   **What it costs is the mode's result, not a regression:** a host softmax between two device
   matmuls sends the full `[seq, seq]` score matrix through DRAM twice per head, so bytes go
   139,984,896 → 970,457,088, a **6.9×** increase, and the mode is much slower at 4096 than the
   six-GEMM form it replaces. Pricing that is the point. What remains for this mode is the
   N-instruction-streams-under-one-xclbin half, which is untouched and is where its
   reconfiguration-minimizing claim actually gets tested.
2. **~~Corrected `runlist` needs one new operator: a device softmax.~~ THE OPERATOR IS DONE
   `[2026-08-09]`; the mode is not.** `builders/softmax.py` wraps the existing streaming family in
   `programming_examples/softmax/softmax.cc` — no kernel was written, and iron's
   `aie_kernels/aie2p/softmax.cc` was never needed. Three gated shapes including 64×4096, the
   attention width `runlist` actually wants. Two things it cost, both worth knowing before the next
   builder:

   - **Give every L1 buffer one role.** Normalizing back into the DMA-destination buffer — dead by
     then, and legal as far as the kernel's `__restrict` is concerned — made the design return
     **the input unchanged** from hardware at all three shapes. `builders/layer_norm.py` keeps one
     role per buffer and that is not style.
   - **A normalization needs its injection target chosen by measurement.** The standard
     `(rows-1, 0)` left the negative control **passing** at two of three shapes, and at 512×512 no
     `atol` admitting the clean run could have rejected the injection. The target is now the last
     row's argmax.

   What remains for the mode itself: `pattern/runlist/` with nothing on the host.
3. **Corrected `fused` — ~~fix its build before anything else~~ build FIXED `[2026-08-08]`.** It
   could not build its own SPECS shape: `fused.py:37` has always said the mode is bounded to
   256..1024 and the row was left at 4096, so `prepare_fused` raised in `builders/norm_tail.py`
   (plane_major stride over the shim `aie.dma_bd` cap) before aircc. The gate was run, confirmed
   red, and the row is **moved to 1024**; `run_npu2_fused_peano.lit` now passes on hardware, 10/10
   stages, `mean_rel_L1` 1.756e-2 at `atol_required` 5.813e-2. Two registry facts moved with the
   shape: the FFN down-projection resolves to `drain` at 1024 (`fused-cast` at 4096), so the
   stitched tail takes 11 whole-tensor args, not 16.

   **The settings conflict is REAL** — doc 26 §4's refutation of it is retracted, see the note
   above. `[2,2]` hangs `mha_out_proj` on hardware. So one xclbin stays blocked, now for a measured
   reason plus `air-fuse-channels`, and the no-DRAM-between-operators half remains
   capacity-bounded (6 MiB on chip against one 6 MiB S×F intermediate at 1024).

   **The cross-mode comparison this gate used to make is suspended**, not restated: it read its
   `sync 19` against `coarse`'s 402, and `coarse` is still a 4096 row while `fused` is now 1024.
   Re-establishing that ranking means measuring both at one length, which is Phase F's job.
4. **Corrected `coarse` is last, by definition** — it is a mix of `runlist` and `fused`, so both
   must be right first. `[2026-08-09]` `runlist` is done and `fused` builds again, so this is now
   the front of the queue. Note what "a blend" has to mean concretely before it is scoped: `coarse`
   must choose, per workload, between an individually dispatched kernel and a fused region, and
   nothing in the port yet expresses that choice — the D2 block it currently wraps is a fixed
   five-kernel sequence, not a decision procedure.
5. **`devq` is the device scheduler, and the migration is now DONE `[2026-08-08]`.**
   `agents/scripts/devq.sh` — builds run concurrently, a measure runs alone with no build in
   flight, stale jobs reconcile by process liveness. **Use `devq.sh run`, not `submit`**: `run` is
   the drop-in for `flock -x LOCK CMD` because it relays the job's output to stdout and exits with
   the job's status, where `submit` diverts output to the job log and returns an id — substituting
   *that* at a gate blanks the FileCheck while still exiting 0. `run` was added for this migration,
   together with a guard that refuses to nest (an inner measure would otherwise queue behind the
   device lock its own parent runner holds and report a lock timeout 30 minutes later).

   All 23 `flock` sites in `phases.sh` are migrated — the seven live `phase_gate_cmd` arms, three
   objective checks that dispatch, and the heredoc gate descriptions a session copies — plus
   `llms/llama32_1b_int4/Makefile`'s `run-inference`. **`make chat` deliberately keeps the bare
   lock:** the broker's runner is `setsid` with stdin from `/dev/null`, so a REPL under it reads EOF
   on the first prompt. `devq-selftest.sh` is **20/20** (was 14), the six new clauses covering
   `run`'s output relay, its status propagation and the nesting refusal.
6. **H10 ran and its substance passed** — five gate legs green, `check-air-mlir` 489/489, 11.44
   tok/s against a 9.43 floor. Its **tamper check halted** on five gate-defining files with
   documented provenance, and was deliberately not re-fingerprinted; see
   [24](24-phase-h10-non-constant-bd-offsets.md).
7. **J7b landed** ([22](22-phase-j7b-accumulator-ring.md)) — the accumulator ring, formed by the
   compiler. `mean_rel_L1` 1.417e-2 at `atol_required` 1.383e-3, K-loop data movement 4 → 2, zero
   packet-typed channels. Its implement session halted on a budget cap with a hardware hang; the
   hang was the compiler defect above, not the design.
8. **J7a landed** ([21](21-phase-j7a-norm-tail-pipeline.md)) — the first piece of iron's dataflow
   form on this port. Three herds, L1→L1 channels, **placement and buffer depth derived by the
   compiler**, `mean_rel_L1` 3.620e-3 against a 1.688e-2 target. Its round-3 fix also made
   `layer_norm` ~26× more accurate, for a measured ~13% throughput cost ([23 §1](23-rules-and-open-items.md)).
9. **J1 is blocked, and precisely.** Not on correctness any more — H9 fixed the miscompile — but on
   shim **BD exhaustion at six trips** against a 64-trip target. It now refuses loudly instead of
   corrupting silently. **Not on the goal path**: J7a reaches the same dispatch collapse without
   the packet queue.
10. **H8 is untouched** and is the largest remaining item: the pass that *derives* on-chip staging
   rather than having the builder declare it. It wanted J7 as a hand-written reference to validate
   against, and J7a and J7b are now two.

**One latent cliff worth knowing about, measured and not reached.** The fused `addnorm` keeps
one-pass bf16 variance and collapses completely once a row's `|mean|/sigma` exceeds ~4 — most
elements wrong, not slightly wrong. This workload's worst row is 0.115, a ~35× margin, so the
recorded figures stand; but nothing pins it. [23 §2](23-rules-and-open-items.md) has the sweep and
what it would cost to fix.

**`[2026-08-08]` Three things that cost THIS run time, so they do not cost the next one:**

- **A recorded claim with no artifact behind it may simply be wrong.** Doc 16 said `attn_output`
  "timed out on the one configuration tried" out of 828 legal ones, and that sentence shaped the
  plan for two days. The first canonical configuration tried **passes**, at every ladder rung, by
  all three methods. The 828 figure is unsourced and unreproducible. `attn_scores`' passing claim
  had no artifact either — it happened to be true. **When a doc asserts a measurement, check that a
  checkpoint, log or test exists behind it before planning around it.**
- **Compare distributions, not a run against a number.** This cost the run twice in one day. A
  four-mode latency table was published from runs taken while builds ran alongside them — `coarse`
  at 4096 read 731 ms there and 467/477 ms on a quiet host, a **1.55×** inflation. Then a "5.9%
  improvement" from pipelining was three fresh runs measured against a single stale high baseline;
  repeat runs put the ranges on top of each other. Both errors are the same shape, and both were
  caught only by re-measuring.
- **Measurement conditions are part of the measurement.** Compilation sits outside the clock;
  host-side dispatch does not. Nothing CPU-heavy may run beside a timed region — which is what
  `devq`'s build/measure classes now enforce, so use it rather than re-learning this.

**Two things that cost the previous run time:**

- **Match a probe's altitude to its claim.** `air-opt` with a hand-built pass list answers "does
  this pass fire", not "does this compile". A construction measured as lowering cleanly through
  `air-opt` never compiled under `aircc`, because `air-to-aie` rewrites callee signatures
  afterwards. Use `aircc` / `XRTBackend.compile(debug_ir=True)` for anything downstream of it.
- **A fixture proves only the shape it runs.** Phase H's four fixture variants were green for a
  whole phase while a silent miscompile lived one column wider — every one of them ran at
  `herd_x=1`. H9's `multicolumn` clause exists for that reason, and it was verified FAILING before
  the fix landed.

Phases A and B were executed by the automated driver — see
[14-the-port-loop-harness.md](14-the-port-loop-harness.md). Both passed their gate, objective
check and tamper check. All ten shipped LLM deployments still pass `make verify` after Phase B's
changes to `llms/shared/infra/cache.py`.

`[2026-08-04]` With one correction: **Phase B's driver-run gate never touched the NPU.** Its
`phase_gate_cmd` was `ninja check-programming-examples-transformer-layer`, and that suite held only
a compile-only test and a host-only test — 2 tests, 16 seconds, per
`agents/.state/port-loop/phase-B/gate.log`. The hardware runlist result recorded in
[05a](05a-phase-b-runlist-spike-result.md) was produced by `make runlist-gate`, which the session
ran and self-reported. `run_npu2_runlist_gate.lit` now puts that gate in the suite; it has been
re-run and all four legs pass, so the claim stands — but it stood on a self-report until then.

Phase C ran as C1–C4 on 2026-08-04, 21 of 40 invocations, ~12 hours wall clock. All four passed
gate, objective and tamper checks. C4 halted once on a driver bug rather than on its own work —
the objective check demanded a registry mtime no honest run could produce — recorded in
[14](14-the-port-loop-harness.md). The registry grew from 33 to 69 bf16-out GEMM shapes with every
pre-existing row byte-identical, and all ten shipped LLM deployments still pass `make verify`.

Phase D ran as D1 and D2 on 2026-08-05, 21 of 40 invocations, ~4.5 hours wall clock (of which
about an hour was a provider outage). Both passed gate, objective and tamper checks. One full
`encoder_bert` layer at `baseline_768`, `seq = 4096`, now matches an FP32 torch golden model on
real hardware over its whole 4096x768 output with zero mismatches, and localizes to any of ten
per-boundary intermediates.

Three things Phase D established that were not known when it was specified:

- **The pre-add `addnorm` was missing.** The operator Phase C validated computes
  `LayerNorm(x) * weight + residual`; `encoder_bert` needs `LayerNorm(x + residual) * weight`. The
  kernel supported both behind `-DADDNORM_PRE_ADD` and Phase A already built the object, but no
  builder exposed it and nothing had ever dispatched it. It is now built, validated, and its
  negative control demonstrated.
- **`compile_gemm_mm`'s object name is a second instance of the `tile_n` collision.** It names its
  object from the GEMM method alone while baking `tile_n` in as `-DDIM_N`, so the FFN's
  up-projection and the o-projection write the same file and one silently gets the other's
  micro-kernel. D2 works around it by interleaving; the real fix is the same `(method, tile_n)`
  naming in `llms/shared/builders/gemm_builder.py` that the ladder needs. **Phase E now has two
  reasons to make that change.**
- **The layer's tolerance has no headroom.** `atol` sits at the hard `1e-1` ceiling with a 1.35x
  margin over the measured `atol_required` of 7.4e-2. The cause is output scale, not error --
  `mean_rel_L1` is 1.7e-2, in line with the per-operator rows -- but Phase E chains this same
  arithmetic four ways, and there is nowhere for a mode to drift.

**A loose end that C4 exposed, closed on 2026-08-05.** The three review rounds were the whole
review budget, so a finding raised in round 3 was fixed by round 3's fix session and then *nothing
re-reviewed it*. C4's round-3 review raised two blocking findings — the `64x768x2304` QKV shape
lacked the `fused-cast` row its builder pins, and the resolution gate checked only each row's
winner rather than the method a builder actually requires. Both were fixed (that shape now carries
all three methods, and the sweep's fused-cast configuration for it replaced one returning zeros for
two of nine cast sub-tiles), and both fixes were verified by hand afterwards — but by the loop's
structure, not by a fourth Codex round. The driver now runs a narrow **confirm review** over the
final round's fix diff before the gate; see [14](14-the-port-loop-harness.md).

## Picking this up in a new session

Read [00-context-and-goals.md](00-context-and-goals.md) and
[02-porting-conventions.md](02-porting-conventions.md) first — the conventions document is a hard
requirement, not advice, and ported code is rewritten to MLIR-AIR style rather than transplanted.

Then, before touching anything:

- [15-environment-notes.md](15-environment-notes.md) — the toolchain was four layers stale on
  2026-08-03 and had to be upgraded end to end. Two CMake flags are lost on any clean rebuild and
  silently hollow out every hardware gate if missing. Read this before running a gate.
- [05a-phase-b-runlist-spike-result.md](05a-phase-b-runlist-spike-result.md) — the plan's
  load-bearing assumption, answered. **Do not act on §"The resolution" in
  [05-phase-b-runtime-seam.md](05-phase-b-runtime-seam.md); the mechanism it proposes is wrong.**
- [14-the-port-loop-harness.md](14-the-port-loop-harness.md) — how the automated driver works and
  how to run the next phase through it.

**The next phase is E (the four execution strategies).** Its specification is
[08-phase-e-execution-strategies.md](08-phase-e-execution-strategies.md), rewritten on 2026-08-05
against what Phase D actually produced.

### What Phase D left you

Do not rebuild any of it. The example's own
`programming_examples/transformer_layer/README.md` is the authoritative file-by-file inventory.

| Piece | Where |
|---|---|
| The FP32 golden model | `pattern/reference.py` — `generate_golden_reference()` for both `encoder_bert` and `decoder_gpt2`, `fuse_qkv_weight()`, per-boundary helpers, and the load-bearing `WEIGHT_DRAW_ORDER`. **Use it; do not re-port iron's bf16 original.** |
| Its independence check | `pattern/test_reference.py` — seven host-only tests pinning the composition against a straight-line transcription, including the three substitutions a numerical comparison would survive (erf vs tanh GeLU, post-add vs pre-add residual, QKV column order) |
| One assembled layer | `builders/block.py` — `block_config()`, `run_block()`, `describe_block()`, `BLOCK_BOUNDARIES`, over four `KernelCache.run_sequence` calls |
| Its gate | `run_npu2_block_peano.lit`, `run_reference_tests.lit`, `run_block_cache_tests.lit`, plus `opcheck.py --operator block` and its fault-injected twin |
| Operators at `baseline_768` | every one, including the pre-add `addnorm` variant D1 had to build because nothing had ever dispatched it |

**`coarse` is most of the way built already.** `builders/block.py` is a fused-operator sequence
over **four** runlists — one per `run_sequence` call, because a dispatch argument is a whole BO —
which is what [08](08-phase-e-execution-strategies.md) calls `coarse`. Phase E's job there is to
give it a strategy directory and route it through the shared instrumentation, not to write it
again. The instrumentation exists too: `DispatchVector` in `llms/shared/infra/dispatch.py`, built
in Phase B.

**The dispatch vector already exists and is already recorded.** The block writes one per sequence
into its results artifact. The four it measured, in order (qkv+mha, norm 1, ffn, norm 2):

```
host_submissions  runlist_entries  air_launches  herd_launches  sync_boundaries      bytes
       1                2               6             10              9          80,216,064
       1               64               1             64            193          18,875,904
       1                1               4              8              7          84,934,656
       1               64               1             64            193          18,875,904
```

Read the two 64-entry rows before designing anything: the normalization points are **64 dispatches
each**, not one launch, because `build_addnorm_module` caps rows per call. `coarse`'s dispatch
numbers are therefore dominated by `addnorm`, not by the GEMMs — which is a real result about
where the cost sits, and one the taxonomy should be able to explain.

### Four decisions Phase E had to take before writing code, and did

`[2026-08-05]` All four are recorded in [08](08-phase-e-execution-strategies.md) and enforced by
the harness rather than left to a session:

- **`coarse` wraps `builders/block.py`; it does not re-home it.** The block is enrolled in
  `run_npu2_block_peano.lit`, in `opcheck --operator block` and in the D1/D2 coverage clauses E1
  re-runs. Moving it churns gate files for nothing.
- **The layout is `pattern/<mode>/`**, per 08's own tree, with **a separate `KernelCache` directory
  per mode**. That last part is not style: a cached ELF is keyed by fingerprint but the cache
  *directory* is chosen by name (`BLOCK_CACHE_DIR`), so two modes sharing one can trade artifacts
  and produce valid numbers attributed to the wrong execution boundary.
- **Distinguishability is ordinal, never threshold.** `coarse` already measures 131 entries, 128 of
  them `addnorm`'s row blocking, so any absolute number would be measuring L1 capacity rather than
  the taxonomy. Four gating clauses; two further predictions recorded but not halting.
- **`offload`'s attention stays in host torch**, so it dispatches six projection GEMMs rather than
  eight. Its two attention GEMMs (`4096x64x4096`, `4096x4096x64`) resolve in no registry, and
  **the sweep cannot be made to produce them**: `sweep_families.py` derives K and N from
  `FAMILY_HIDDEN × ROLE_KN_MULTIPLES` with a minimum hidden of 512, so no `--family` stages a 64 in
  the K or N position. 08 offered "sweep them in" as one of two options; it is not available. This
  makes `offload` a hybrid boundary, which its README must say.

### Two things Phase E had to decide first

- **`[2026-08-05]` The ladder is still blocked at one point, and there are now two reasons.**
  Everything runs at `seq = 4096` only. `build_ffn_module`'s up- and down-projections collide on
  `f32_to_bf16_mn_<suffix>` at every other point on the ladder, and D2 found a second instance one
  layer down: `compile_gemm_mm` names its object from the GEMM method alone (`mm_m32.o` /
  `mm_m64.o`) while baking `tile_n` in as `-DDIM_N`, so the FFN's up-projection and the
  o-projection write the same file and one silently gets the other's micro-kernel. D2 works around
  the second by interleaving inside `builders/block.py`; **any caller that builds several of these
  operators together without interleaving hits it again, silently.** One fix closes both: a
  `(method, tile_n)`-aware symbol and object name in `llms/shared/builders/gemm_builder.py`. That
  file was off limits to Phases C and D. Phase E needs the ladder, so it is Phase E's to make —
  and doing so puts `make verify` over the ten shipped models inside its gate.
- **The layer's tolerance has no headroom.** `atol` sits at the hard `1e-1` ceiling with a 1.35x
  margin over a measured `atol_required` of 7.4e-2. The cause is output scale, not error
  (`mean_rel_L1` is 1.7e-2, in line with the per-operator rows), but Phase E chains the same
  arithmetic four different ways against the same oracle. If a mode needs more than that, the
  answer is a recorded finding, not a wider tolerance — the driver rejects anything above `1e-1`.

### The harness has an E entry

`[2026-08-05]` Built. `PL_PHASES_IN_SCOPE` reads `'["E1","E2","E3","E4","E5"]'` and all seven
dispatchers carry arms for each. What it consists of:

| Piece | Where |
|---|---|
| Five sub-phase specs, one per session | [08a](08a-phase-e1-unblock-the-ladder.md) · [08b](08b-phase-e2-coarse-and-instrumentation.md) · [08c](08c-phase-e3-offload.md) · [08d](08d-phase-e4-runlist.md) · [08e](08e-phase-e5-fused-and-distinguishability.md) |
| E1's two-leg gate | `agents/scripts/port-loop/gate-e1.sh` — lit suite, then `make verify` over the ten shipped models |
| The objective checks | `agents/scripts/port-loop/phase_e_checks.py`, with its fixtures in `phase_e_selftest.py` |
| Their both-directions test | `python3 agents/scripts/port-loop/phase_e_checks.py selftest` — 27 clauses, no hardware |

Three things about it worth knowing before touching it:

- **The checks are a module, not a heredoc.** Every other phase embeds its objective check in
  `phases.sh`; Phase E's are far larger and, more to the point, a module can be run in both
  directions. `selftest` builds conforming and violating artifact sets in a temp directory and
  asserts the verdict flips for each clause. The pass direction is also demonstrated against real
  data: D2's `block` artifact pair satisfies the full-layer scope, the vector contract and the
  provenance clause unmodified.
- **The dispatch vectors have a negative control now.** `results/` is gitignored, so a fabricated
  `dispatch_vectors` block is invisible to `guard_fingerprint`, `guard_check_tamper` and every
  Codex diff — freshness alone never stopped it, and no phase before E noticed. The driver already
  re-runs each operator under `--fault-inject input`; Phase E additionally requires that run's
  summed vector totals to **equal** the clean run's. A session cannot know those six numbers
  without dispatching.
- **The driver's own scripts are fingerprinted**, as of this phase, and are in no allowlist. Every
  anti-reward-hacking layer policed what a diff did to a *gate*; none watched the thing that runs
  the gates, while sessions run under `--permission-mode bypassPermissions`. Any edit under
  `agents/scripts/port-loop/` now halts the run.

**The allowlist did not need to widen**, contrary to what [14](14-the-port-loop-harness.md)
predicted. `guard_gate_files()` covers `.lit` files, example `Makefile`s,
`programming_examples/CMakeLists.txt`, `kernel_registry/details/*.json` and `llms/verify/*.py`;
`gemm_builder.py` is in none of them, and E1's second gate leg *runs* the ten shipped models rather
than editing them. Keeping `^programming_examples/transformer_layer/` is what stops E1 quietly
touching a shipped model's `Makefile` to make its own regression leg pass.

Two decisions taken on 2026-08-04, now reflected throughout these documents:

- **The reference oracles are re-expressed, not ported verbatim.** `[2026-08-05]` The figure this
  plan long quoted -- bf16 at `rtol=4e-2` with a 0.5% mismatch budget -- is iron's **per-operator**
  gate (`BLOCK_*` in `study/end_to_end/modes.py:110-125`). Its **end-to-end mode** gate is looser
  still: `FINAL_REL_TOL=0.1`, `FINAL_ABS_TOL=0.5`, a **5%** mismatch budget, and it only runs at
  `seq_len <= 512` (`REFERENCE_VALIDATION_MAX_SEQ_LEN`) -- above that it degrades to a
  finite-output check, with separate spot checks at 512/2048/8192. This port uses an FP32
  reference, the registry's `rtol`/`atol`, and zero mismatches, at the full `seq 4096`. Details and the two further traps (erf vs tanh GeLU, the
  MHA oracle's precision switch at `seq_len 16384`) are in
  [06 §The numerics standard](06-phase-c-operators.md#the-numerics-standard--do-not-port-irons).
- **Shape coverage is a sweep, not a redesign.** The case matrix needs 108 distinct
  projection-GEMM shapes, not the "several hundred" previously estimated. C4 built the sweep tool
  and registered the 36 `baseline_768` shapes, which is what Phases D and E run on. The other two
  families are a later machine-time run of the same tool against a different `--family`: no code
  change, just hardware hours. **Phase F's case matrix needs them**, so budget that run before F
  rather than inside it.

## Load-bearing questions already answered

| Question | Answer | Where |
|---|---|---|
| Can separately-compiled ELFs share one runlist? | Yes — N ELFs, N `hw_context`s, one runlist. Bit-identical to sequential, 1.02–1.15× faster. **Not** by sharing one context; XRT rejects that three ways. | [05a](05a-phase-b-runlist-spike-result.md) |
| How many concurrent `hw_context`s does NPU2 grant? | 32 (33 fails with `DRM_IOCTL_AMDXDNA_CREATE_HWCTX err=-2`). Phase E's `runlist` mode wants 29 — fits, with three to spare. Caveats on the margin recorded. | [08 §Risks](08-phase-e-execution-strategies.md) |
| Does a full layer survive the real runtime path? | Yes. One `encoder_bert` layer at `baseline_768`, `seq 4096`, matches an FP32 torch oracle over its whole 4096×768 output with zero mismatches, and localizes to any of ten per-boundary intermediates. | [07b](07b-phase-d2-block-integration.md) |
| Can the whole sequence ladder be built? | **Not yet.** `seq = 4096` is the only point where the FFN's two projections do not collide, at the symbol level *and* the object level. One `(method, tile_n)` naming fix in `llms/shared/builders/gemm_builder.py` closes both; nothing before Phase E was permitted to make it. It also blocks the `fused` mode outright at any sequence length. | [08](08-phase-e-execution-strategies.md) |
| Is there tolerance headroom for four modes? | Thin. The layer needs `atol` `1e-1` — the hard ceiling — at 1.35× its measured requirement. Driven by output scale, not by error. | [07b](07b-phase-d2-block-integration.md) |

## Provenance

The source is `iron` commit `1e014c1` "Add transformer-layer execution-strategy studies"
(145 files, ~58.6k insertions), validated there by an 888/888-job suite run. This plan was
reviewed by Codex before approval; findings that materially changed it are marked `[Codex]`
in the phase documents.
