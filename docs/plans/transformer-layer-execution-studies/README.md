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
| [03-measurement-model.md](03-measurement-model.md) | **The definition of the four modes, and it is current.** The corrected taxonomy (reconfiguration cost against DRAM traffic), what is implemented against it today, the dispatch vector, CSV schema v1 — and `[2026-08-10]` **§The vocabulary**: the standard terms (submission against dispatch, packaged against resident composition, the role-style names) and the knobs-and-costs axis map |
| [04-phase-a-kernels.md](04-phase-a-kernels.md) | AIE2P device kernels |
| [05-phase-b-runtime-seam.md](05-phase-b-runtime-seam.md) | Runlist aggregation + BO liveness pooling — **overview**; the two sub-docs are [05a](05a-phase-b-runlist-spike-result.md) (the spike result: N ELFs, N `hw_context`s, one runlist) · [05b](05b-phase-b-buffer-rules.md) (**the buffer rules `programming_examples/llms/shared/infra/bo_pool.py` implements** — note the path is under `llms/`, not `transformer_layer/`; ownership, synchronization, bank and aliasing; "a rule that is not in this list is not enforced", and its O3 logical-size rule is the easiest way in the whole seam to produce plausible garbage) |
| [06-phase-c-operators.md](06-phase-c-operators.md) | The six new operators as AIR builders — **overview**; the four sub-phase specs are [06a](06a-phase-c1-gate-and-small-operators.md) · [06b](06b-phase-c2-qkv-proj-and-ffn.md) · [06c](06c-phase-c3-mha-out-proj.md) · [06d](06d-phase-c4-coverage-sweep.md) |
| [07-phase-d-block-integration.md](07-phase-d-block-integration.md) | Single-block integration gate — **overview**; the two sub-phase specs are [07a](07a-phase-d1-operators-at-baseline-768.md) · [07b](07b-phase-d2-block-integration.md) |
| [08-phase-e-execution-strategies.md](08-phase-e-execution-strategies.md) | The four execution strategies |
| [09-phase-f-study-harness.md](09-phase-f-study-harness.md) | The seven measurement studies |
| [10-phase-g-unattended-runner-and-ci.md](10-phase-g-unattended-runner-and-ci.md) | Unattended suite runner, CI wiring. **`[2026-08-12]` half obsolete and annotated in place — read its header first.** G0 is built (`study/profiles.py` + `study/run_profile.py`, `run_status="skipped"` emitted for the first time, row counts in `manifest.py`) and so is the CI leg (a second lit target for the **10** PR-safe tests, wired into `buildAndTestRyzenAI.yml` with the count asserted). **Four of its behaviours are recorded as dropped** on measured grounds — the `@reboot` crontab hook, TTM transitions, thermal gating and `turbostat` — which collapses its passwordless-sudo block to `xrt-smi configure` alone |
| [11-goal-sota-sliding-window.md](11-goal-sota-sliding-window.md) | Goal 1 — sliding-window / local-global attention |
| [12-goal-quantized-inference.md](12-goal-quantized-inference.md) | Goal 2 — quantized inference |
| [13-verification-and-acceptance.md](13-verification-and-acceptance.md) | Every gate, in one place |
| [14-the-port-loop-harness.md](14-the-port-loop-harness.md) | The automated driver: how it works, how to run a phase, what it learned the hard way |
| [15-environment-notes.md](15-environment-notes.md) | Toolchain state and the setup traps that silently hollow out hardware gates |
| [16-compiler-work-and-remaining-essence.md](16-compiler-work-and-remaining-essence.md) | Tranche H (compiler) and tranche J (the study), and what AIR automates versus what iron writes by hand. **Its J2 row is false** — the `attn_output` timeout and the "828 legal configurations" it cites have no artifact and the first configuration tried passes |
| [17-phase-h-compiler-hardening.md](17-phase-h-compiler-hardening.md) | Phase H spec plus its attempt-by-attempt record — including two of its own claims that measurement falsified |
| [18-phase-h1s-skip-not-refuse.md](18-phase-h1s-skip-not-refuse.md) | H's correction, run fresh rather than resumed: the safety proof declines to *transform*, never to compile |
| [19-phase-j1-collapse-norm-dispatches.md](19-phase-j1-collapse-norm-dispatches.md) | J1 — blocked, with both walls it hit measured and recorded |
| [20-phase-h9-fuse-through-parallel.md](20-phase-h9-fuse-through-parallel.md) | H9 — the packet fusion that only ever worked on one column, and what it took to fix |
| [21-phase-j7a-norm-tail-pipeline.md](21-phase-j7a-norm-tail-pipeline.md) | J7a — the norm-tail pipeline. **The first working piece of the dataflow goal** |
| [22-phase-j7b-accumulator-ring.md](22-phase-j7b-accumulator-ring.md) | J7b — **landed** 2026-08-07. Partial sums that never leave the chip, with the compiler forming the ring |
| [23-rules-and-open-items.md](23-rules-and-open-items.md) | **Read before building anything.** The design rules that govern later work — the per-column shim budget, the L3-side offset rule, one process per measurement — and the open items nobody has claimed |
| [24-phase-h10-non-constant-bd-offsets.md](24-phase-h10-non-constant-bd-offsets.md) | **Substance verified, tamper baseline not clean.** The silent miscompile J7b lost a session to, located: an unchecked `std::optional` deref in `air-to-aie` |
| [25-first-study-result-sequence-ladder.md](25-first-study-result-sequence-ladder.md) | **`[retracted]` Do not cite the crossover or the slopes.** It ranks four *implementations* that predate the corrected taxonomy, and its explanation — that the slopes split on attention placement — is now **unreproducible**, because as of 2026-08-09 all four modes run attention on the device. The measurement stands as a record of what was built on 2026-08-08 |
| [26-mode-rebuild-feasibility.md](26-mode-rebuild-feasibility.md) | **The feasibility record for the mode rebuilds — read it for its CORRECTIONS, not as current state.** It opens with six things the plan had wrong, three of which have since been settled by hardware, two of them against what it concluded. **§4 is retracted** (`runtime_loop_tiling_sizes` is not inert) and **§5 is corrected twice** (the missing row-max is the single-shot kernel's, not the streaming family's; `SM_LOG2E` is a base conversion, not a blocking scale). Its §6 was right and is fixed |
| [27-common-ladder-result.md](27-common-ladder-result.md) | **The first four-mode comparison at one sequence length**, 512 and 1024, walked twice. DRAM traffic orders exactly as the taxonomy predicts; `fused` fastest and `coarse` second at both lengths; `runlist` and `offload` indistinguishable. **A crossover that walk 1 reported did not survive walk 2** — read its §The crossover that did not survive before running one walk of anything |
| [28-coarse-blend-space.md](28-coarse-blend-space.md) | **What `coarse`'s blend is a blend OF, and what selects it.** The space is two axes and six cells, derived from the artifact plans — and **two of the six ARE `fused` and `runlist`**, so "pick the best cell" collapses the taxonomy. Resolves it: the blend is selected by **what the workload admits**, and `coarse` is the mode you use where `fused` does not fit. **Measure it at 2048/4096, not at 1024**, or it reports `fused` under another name |
| [29-offload-n-streams.md](29-offload-n-streams.md) | **`offload`'s reconfiguration half, landed and now GATED.** Five shapes in one xclbin, the array configured once instead of thirty times, dispatch vector unchanged by design. Records the **three** identifiers a stream needs and the two that fail silently; the stale-install trap that made a no-op run look like a success; **the `[2026-08-09]` gate and the 4096 wall that bounds it to single-launch modules** — so everything the document originally claimed is a **1024** result; **the variance measurement that confirms `_evict_context` is why this mode is noisy**, along with the ~20% best-case latency it costs to remove; and **`[2026-08-11]` §The hardware verdict**: in-stream `load_pdi` FAULTS the firmware, the `--expand-load-pdis` fallback is falsified (no-op on this aiecc, claim had no artifact), and route 3 — the `drain` pin — has the shared path **running and gating at 4096**, the mode's own spec shape; **`[2026-08-11]` the default FLIPPED to the shared path** per its recorded decision — the ELF path is the legacy/control opt-in (`AIR_OFFLOAD_LEGACY_ELF=1`; the retired `AIR_OFFLOAD_SHARED_XCLBIN` raises), no pinned gate literal moved, every recorded `offload` latency/variance number predates the flip, and **the owed re-walk ran the same day** — [32 §The post-flip walk](32-cost-decomposed-ladder.md) |
| [30-coarse-cells-built.md](30-coarse-cells-built.md) | **`coarse`'s two interior cells, built, gated and measured — and the mode's blend now has provenance.** C2 and C3 passed their first hardware run; their entry counts were predicted host-side beforehand by a model that also reproduces `coarse`'s and `runlist`'s shipped gate literals. **The four cells' dispatch vectors are ADDITIVE** across the two axes on all six columns (bytes exactly), which is a provenance check nobody wrote. The ladder puts C1 first at 2048 and 4096 over two walks, so **`coarse` = C1** — an interior cell, so the taxonomy does not collapse. Also records two things that are not about `coarse`: the `runlist` catalogue row was **stale** (it described the pre-rebuild host-attention mode, tolerance figures included), and `run_npu2_runlist_gate.lit`'s latency clause has **no margin** and is now intermittent under suite contention (red once, green twice, same code). **`[2026-08-12]` read that last finding with its own resolution, which is inside doc 30 and was missing here**: the intermittency was answered on 2026-08-10 by comparing interleaved **minimums** instead of medians (queue item 5), which fixed the flakiness *without* widening the strict inequality — so the clause's remaining exposure was the power mode, not the margin, and queue item 13 guarded it rather than adding a tolerance |
| [31-fused-resident-tail.md](31-fused-resident-tail.md) | **Phase spec for `fused`'s resident tail** — the mode's definitional gap (packaged today, resident by definition) scoped without churning any mode file: what exists to compose (J7a ×2 + J7b route within the column budget at both lengths, probed hermetically, negative control verified failing), the walls the tail scope avoids, gates, increments, non-goals — and **`[2026-08-11]` its two new status sections: R1 is BUILT and structurally green** (one-segment FFN interior, hermetic probe green, ~~dataflow emulated element-exact, 5/5 host tests~~ — **`[2026-08-12]` that citation was EMPTY and is corrected in §"The emulation arm was blind"**: the arm never built the module; the rebuilt one measures **5.457e-12 on the module the builder emits** and **rejects the real pre-E1 builder at 4.716e+03**, 8/8) **and its device gate is PARKED on a pinned compiler crash**: `air-fuse-channels` segfaults on ≥3 mutually-mergeable sibling channel nests (N=2 fuses 5/5, N=3 crashes 5/5; reproducer probe shipped) — **`[2026-08-11]` later: the crash is FIXED in source, the gate RAN, and it is RE-PARKED one wall further** (§The gate ran): STRUCT arm green on the corrected fusion; the numeric arm's full compile stops at shim BD exhaustion — 24 sequential refill tasks vs tile (1,0)'s 16 BDs, the J1 wall, deterministic and loud (queue item 6b). [31a](31a-resident-byte-floor.md) is the byte-floor derivation: whole-layer resident is **15.0/16.5 MiB** at 512/1024 against 48.75/84.0 MiB packaged crossings, checkable to the byte |
| [31b-r2-order-seam.md](31b-r2-order-seam.md) | **R2's scoping — and it CORRECTS doc 31's own prediction about which side of the order seam moves.** Doc 31 expected the seam to close by re-mapping the norm tail's row→tile assignment to band order; a norm emits whole rows while a GEMM's A operand is a blocked column strip of all rows, so no row→tile re-mapping converts one into the other. The re-mapping that works is **partitioning the GEMM herds by rows (M) instead of by output columns (N)** — producer core `c` and consumer core `c` then own the same rows, the retile becomes local to one core (compute, not a BD) and the frozen-BD rule never applies. Carries **two new compiler defects, both reproduced** (queue items 8 and 9) and a **blind spot in R1's shipped gate arm** (it counts `shim→core` only and reads 1 for a column actually at 2 of the 2-per-column budget). Everything dump-derived is PROVISIONAL — read its header before using a constant |
| [32-cost-decomposed-ladder.md](32-cost-decomposed-ladder.md) | **The first cost-decomposed four-mode ladder (2026-08-10), and the machine anomaly it caught.** Byte totals walk-identical; warm DRAM ordering is now **`runlist` < `fused` < `coarse` < `offload`** (the targeted eviction moved `runlist` down by exactly its static set); the reconfiguration columns on their first walk (offload 30 / runlist 24 / coarse 0 / fused 0). **Read its first section before citing any latency**: the machine's `hw_context` load cost rose ~30× overnight, bisected to the environment — **`[2026-08-11]` RESOLVED: it was the NPU power mode** (Turbo reset to `Default` by an overnight self-reboot; §RESOLVED has the full diagnostic chain and the standing Turbo rule). Its 2026-08-10 latencies stand as `Default`-conditional; bytes and counts are pmode-independent. **`[2026-08-11]` §The post-flip walk**: the first unconditional four-mode comparison — Turbo verified, `offload` on the shared default, 16/16 orderings `fused` < `coarse` < `runlist` < `offload`, all four modes separate for the first time |
| [33-memcpy-bandwidth-scoping.md](33-memcpy-bandwidth-scoping.md) | **Queue item 11(a), and the answer is DEFER — because it is not 11(b)'s sibling, it is 11(b)'s PREREQUISITE.** `roofline/run.py` refuses to run without the memcpy CSV and makes `peak_bandwidth_gbps` literally the memory roof's slope, so it is the roofline's only empirical input. Corrects the queue row's premise twice, both artifact-backed: iron's "four case axes" are really **two** (`SIZE_LADDER` and `NUM_CHANNELS` are single-valued, so dropping `num_channels` removes no information — iron never varied it), and iron's kernel arm is `failed_validation` at 8 and 16 cores, 2 of 8 rows. **Read its `[2026-08-12]` merge-verification section before quoting the imported ceiling**: the 4-shim-tile rung is 64.3 in one measurement and 70.2–70.9 in two others, the peak is at 4 tiles not 8 (**non-monotonic**), and three of the five artifacts are one run copied — identical to the digit on latency |
| [34-phase-g-scoping.md](34-phase-g-scoping.md) | **Phase G investigated, and doc 10's spec is substantially obsolete** — roughly half of iron's 2,494-line `unattended_reboot.py` already exists here in better form (`devq.sh`, and a pmode enforcement chain that *refuses* where iron warns). Doc 10's literal gate sentence is satisfiable by two existing scripts today. **The real blocker is reachability, not machinery**: `run_mode.py` hardcodes `encoder_bert` and every mode SPECS row is `emb_dim 768`, so **one of six declared families is reachable** and a "full profile" needs a Phase-C-sized coverage sweep first. Recommends dropping four of doc 10's behaviours on measured grounds, and surfaces the **latency gates with no pmode guard** (queue item 13) |
| [35-goal-1-sliding-window-scoping.md](35-goal-1-sliding-window-scoping.md) | **Goal 1 investigated: the kernel change is small, the goal around it is not.** Causality for all ten shipped models is ONE function (`apply_causal_mask`), absolute positions are already derivable in-kernel, so a band is two extra branches — but masking is **element-wise, not tile-skipping**, so a window is *correctness-only, zero speedup*, and the skipping route that would pay is a documented hang path plus queue items 8 and 9 verbatim. **Its most important finding is that the gate as specified may be VACUOUS** — top-5 set inclusion at first divergence only, and full causal is a superset of the window, so an implementation that silently degrades to full causal would frequently still pass. Corrects the spec three times (the prior-art branch has never run Gemma 3 on the NPU; the weights are not on this machine and are license-gated) |
| [36-goal-2-quantized-scoping.md](36-goal-2-quantized-scoping.md) | **Goal 2 investigated, and the "first" quantized model is more complete than doc 12 implies** — int4 dequantization is **on device, in-core, in-tile**, and `make verify` DOES exercise the quantized path: 31 of 32 gated tokens per prompt come out of int4 decode kernels (doc 12's Gate line and doc 13:50 overstate the gap). **Doc 12 names the wrong second model**: undocumented hard asserts (`hidden_dim % emb_dim == 0`, `n_heads·head_dim == emb_dim`) leave only three of nine candidates fitting, Llama-3.2-3B FAILS them, and **SmolLM2-1.7B — which doc 12 never mentions — is shape-exact**. Top risk is external and unresolved: a compatible AutoAWQ checkpoint may not exist. Also finds the int4 verify lit **disabled** (queue item 14) |

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
| F — study harness | `execution-smoke-test` yields ≥1 `run_status=passed` row per measurement CSV | **in progress**, ~~on `exper/phase-f-study-harness` (a worktree, unmerged)~~ **`[2026-08-11]` correction: there is no unmerged worktree.** `exper/phase-f-study-harness` (tip `4775722e`) is a full ancestor of this branch, 0 unmerged commits — all Phase F work is here. **The gate itself passes on hardware over all four modes** as of 2026-08-08 — `smoke_gate` PASS, `manifest complete: True`, all four distinguishability clauses hold on the measured vectors ([09](09-phase-f-study-harness.md)). Work items 1, 2, ~~5~~, 6, ~~7~~, 8 done (5 without pytest, 7 scoped to result trees). **`[2026-08-11]` item 4 is four of five and item 3's PORTABLE half is done**: `study/resource_usage.py` (`core_to_core_flows` verified on real artifacts in both directions — 16/40 space-multiplexed on a fresh norm-tail compile, 0/116 time-multiplexed on `transformer_layer`; devq job 238), `component_groups.py` (first real table: `offload` @1024 attributes 79.8 ms of 159.8, **50.1% unattributed**, job 246), `run_lock.py`, `cases.py`, `power.py` (root-free RAPL/hwmon backends, verified live), `compare_roots.py`, `select_rows.py`. Host suite 103/103 → **231/231 in 17 modules**. **What remains: `memcpy_bandwidth` (needs a multi-core AIR memcpy operator that does not exist) and the plot/analysis tier** — `regenerate_plots.py`, `roofline/{run,test}.py`, every `plot_*.py` — **still blocked**: matplotlib/pandas/seaborn are absent and must not be installed while gates run |
| Corrected `offload` — attention on device | `run_npu2_offload_peano.lit`, both recipes, at the corrected 30-dispatch boundary | **done** 2026-08-08. 10/10 stages clean, `submissions 30 entries 30 air 31 herd 91 sync 91 bytes 970457088`, negative control exact through the attention half. No registry write, no compiler work, no tolerance widened — `attn_context` 11.4× margin, `output` 1.73×. Costs 6.9× the DRAM traffic, which is the mode's result |
| `fused` build repair — SPECS row 4096 → 1024 | `run_npu2_fused_peano.lit`, both recipes | **done** 2026-08-08. The gate was **red and unrun**: the row was left at 4096 while the mode has always been bounded to 256..1024, so it raised before aircc. Now green at 1024 — 10/10 stages, `mean_rel_L1` 1.756e-2 at `atol_required` 5.813e-2. Its cross-mode `sync` comparison against `coarse` is **suspended**, not restated: the two rows are now at different sequence lengths |
| Backend-preset conflict — settled on hardware | none; a measurement that retracts [26 §4](26-mode-rebuild-feasibility.md) | **recorded** 2026-08-08. `runtime_loop_tiling_sizes` is **not inert**: `[2,2]` hangs `mha_out_proj` @4096 3/3, `[1,1]` passes 3/3, `omit_pingpong` irrelevant either way. Restores the conflict `fused.py` / `mha_out_proj.py` / `block.py` document, with a corrected reason. `agents/probes/probe_backend_preset_hardware.py` |
| Device `softmax` operator | `run_npu2_softmax_peano.lit`, clean + negative control, three shapes | **done** 2026-08-09. `builders/softmax.py` over the existing `softmax_streaming.o`; no kernel written. 512×512, 4096×768 and **64×4096** (attention width, where `rows_per_call` drops 8 → 2 on L1). `mean_rel_L1` 1.60–1.63e-2, `atol` 2.7–2.9× `atol_required`, plus a `mean_rel_L1_max` ceiling because a softmax row spans three orders of magnitude and an element-wise `atol` alone is loose at the bottom. Two corrections to [26 §5](26-mode-rebuild-feasibility.md) recorded there |
| Corrected `runlist` — every operator on device | `run_npu2_runlist_peano.lit`, clean + negative control | **done** 2026-08-09. **427 entries over 17 runlists, nothing on the host.** Per head `attn_scores` → `softmax` → `attn_output`, device-resident inside one submission; one submission per head is a memory bound (~800 MiB if batched, ~70 MiB per head), not a schedule choice. 10/10 stages clean, `submissions 17 entries 427 air 50 herd 488 sync 451 bytes 190513152`. In the end it never touched `builders/mha_attention.py`, so the `fused` serialization this table warned about was not needed |
| **The first result on the corrected axis** | none; a measurement | **recorded** 2026-08-09. `runlist` moves **190,513,152** bytes against `offload`'s **970,457,088** for the same layer — **5.1×**, produced entirely by where the softmax runs. `offload` puts it on the host, so every `[4096, 4096]` score matrix crosses DRAM twice per head; `runlist` keeps it on the array. Two modes differing in exactly the corrected taxonomy's variable — reconfiguration against DRAM traffic — rather than in attention placement, which is the confound every earlier comparison carried |
| `attention_path` retired as a covariate | none; a consequence | **recorded** 2026-08-09. With `runlist` on the device, **all four modes are**. The first sequence ladder's headline — slopes splitting on attention placement, host 1.23–1.27 against device 1.03–1.17 — **cannot be reproduced**, because no mode sits on the host side any more. `study/test_attention_path.py` now asserts that end state rather than the two-value invariant it was written with |
| **The four modes at one sequence length** | none; a measurement, walked twice | **recorded** 2026-08-09. 512 and 1024, all four modes, 8/8 rungs twice. **DRAM traffic orders as the taxonomy predicts at both lengths** — `fused` 42.5 MB < `coarse` 44.0 < `runlist` 55.2 < `offload` 99.1 at 1024, byte-identical across walks. **`[2026-08-10]` The warm ordering is superseded**: the targeted pool eviction moved `runlist` to 40.9 MB at 1024 — below `fused` and `coarse` — see [32](32-cost-decomposed-ladder.md). On latency `fused` is fastest and `coarse` second at both lengths; **`runlist` and `offload` are indistinguishable** — averages and minimums disagree, and each flips between walks. **A crossover walk 1 reported did NOT survive walk 2.** `offload` alone drifts up to 120% intra-walk, corroborating [03](03-measurement-model.md)'s wider band for it from a fresh measurement. Trap 1 below is closed by this; the SPECS rows still span two lengths, so build cross-mode tables from a ladder run, never from the catalogue. See [27](27-common-ladder-result.md) |
| `offload` N instruction streams under one xclbin | `E1 GATE` five legs, plus the mode's own lit suite | **done** 2026-08-09 (`93e15a64`). Five shapes in ONE xclbin: `context_loads 1 kernel_attaches 4 over 30 dispatches`, against 30 loads on the ELF path. **Dispatch vector unchanged** — 30/30/30/90/90/99,090,432 — which is the design, since the mode makes one `run_sequence` call per GEMM either way, so the existing gate is a correctness check on the change. Needed THREE distinct identifiers per stream (`kernel_name`, `instance_name`, `kernel_id`), two of which fail silently. **No latency claim**: four interleaved A/B runs overlap on avg and min. `E1 GATE: PASS` — lit 28/28 on NPU2, 10/10 models verify. ~~**Opt-in and UNGATED**: no lit recipe runs the shared path.~~ **`[2026-08-09]` GATED** — see the row below, which also records that these figures are **1024** figures: the chain does not build at the mode's own 4096. See [29](29-offload-n-streams.md) |
| `offload` shared path gated, and its variance explained | `run_npu2_offload_peano.lit` third recipe, plus four measurement walks | **done** 2026-08-09. The gap [29](29-offload-n-streams.md) recorded against itself is closed: the suite now pins `context_loads 1 kernel_attaches 4` on the shared path and `context_loads 30 kernel_attaches 0` on the ELF one, **verified failing** against the same mode at the same length over ELF packaging. **Suite 28/28 on NPU2** (494.5 s), host tests 84/84, seam 31/31, `phase_e_checks` 30/30. **It gates at 1024, not 4096, for a measured reason**: the shared path is bounded to SINGLE-LAUNCH modules — at 4096 the down-projection is a two-launch `fused-cast` and `XRTBackend.compile`'s fixed `air.insts.bin` collides with itself (`aiecc: edge ... produced duplicate output path`), where only the ELF branch omits `-i` and lets aircc derive a name per launch. **So doc 29's landing was always a 1024 result and did not say so.** `[2026-08-11]` **The 1024 bound is lifted**: the shared recipe now gates at 4096 through the drain pin (route 3 — the platform faults in-stream `load_pdi`, so the chain stays single-launch; doc 29 §The hardware verdict). Separately, **[27](27-common-ladder-result.md)'s variance hypothesis is SUPPORTED, not confirmed**: four interleaved walks with `runlist` as a same-conditions control put `offload`'s intra-walk spread at **316.9% / 134.1% on the ELF path against 17.6% / 14.0% on the shared one** at 512, both walks. Switching packaging removes the variance — but the switch changes the ABI *as well as* the reconfiguration, so it does not isolate `_evict_context`; the control rules out environmental drift, not the ABI. Isolating it needs a third arm (xclbin ABI, eviction forced on) and a knob that does not exist. **The default did not move** — the shared path costs ~20% on best-case latency at 512 (97.5–99.5 ms against 78.9–82.0), so it is a trade, and flipping invalidates every recorded `offload` number |
| Corrected `coarse` | `run_npu2_coarse_c2_peano.lit`, `run_npu2_coarse_c3_peano.lit`, plus a two-walk ladder at 2048/4096 | **done** 2026-08-09. The two interior cells built, gated and measured, and **`coarse`'s blend now has provenance**: it is cell **C1 = (block front, banded tail)**, chosen by measurement rather than inherited from D2. Both cells passed their FIRST hardware run — 10/10 stages, negative controls failing as required — and their entry counts were **predicted host-side before they ran** (C2 4/389, C3 17/169) from a model that reproduces `coarse`'s own 4/131 and `runlist`'s 17/427. Ladder walked twice: **C1 < C2 < C3 < C6 survives both walks on averages and minimums at both lengths**. **The taxonomy did NOT collapse** — the winner is an interior cell, so `coarse` stays distinct from `runlist` (slowest) and from `fused` (unbuildable at these lengths), which is exactly the risk [28](28-coarse-blend-space.md) flagged. **The front axis dominates the tail axis** by roughly an order of magnitude in effect size (~1.5–1.6× against a tail effect that is clean only at 4096 and unresolved at 2048). See [30](30-coarse-cells-built.md) |
| Cost instruments — schema v2 | study host tests + dispatch/seam tests + `phase_e_checks` selftest; both pinned offload gate lines re-verified on NPU | **done** 2026-08-10 (`eeb37a19`, delta fix `4ced893b`). Both taxonomy costs are columns for all four modes: `device_ms`/`sync_ms`/`host_cpu_ms` + `context_loads`/`kernel_attaches`, v1 prefix pinned. First rows confirmed every known truth (offload-ELF 30, `runlist` 24, `coarse`/`fused` 0); found `cache.py` had always counted ELF loads — the modes never surfaced it |
| Small confounds — queue items 4 + 5 | `check-runlist` + fault twin + `check-coarse-c3`; isolated seam-gate re-run | **done** 2026-08-10 (`2f66fc86`, `e2996fbd`). Targeted pool eviction (warm `runlist` −14,352,384 bytes, cold totals unchanged so no literal moved) and the seam gate's verdict on interleaved minimums (not widened) |
| Multi-launch xclbin packaging — the 4096 compile wall | fixture `test/xrt/56` (verified failing unpatched), the real fused-cast module, the five-shape chain; suite 30/30 + ten models 10/10 on the pre-activation install | **compile half done** 2026-08-10 (`623768f2`), **activated in install-xrt** 2026-08-11 by the operator, fixture re-verified against the real install. **`[2026-08-11]` DISPATCHED, and the answer is NO**: the packaging works (29 single-launch dispatches clean off the shared xclbin at 4096) but the multi-launch module **faults the firmware** (`fatal_error_type 0x10`); the scoped fallback was falsified and route 3 (the `drain` pin) landed instead — [29 §The hardware verdict](29-offload-n-streams.md) |
| `fused` resident-tail scoping | hermetic structural probe (negative control verified failing); byte-floor derivation checkable to the byte | **done** 2026-08-10 (`601c54ae`). [31](31-fused-resident-tail.md) is the phase spec, [31a](31a-resident-byte-floor.md) the floor (84.0 → 16.5 MiB at 1024); J7a×2 + J7b route within the column budget at 1024 AND 4096; stitching is time-multiplexed at segment granularity, so residency needs a one-segment builder. Also found doc 26 §C's itemization missing its `ffn_out` row |
| Resident tail R1 — the one-segment FFN interior | hermetic structural probe (8 clauses); f64 dataflow emulation, element-exact; ~~5/5~~ **8/8** host tests with their own lit (**`[2026-08-12]` the emulation arm did not build the module until queue item 17 rebuilt it**; it now interprets the built `air.ir.Module`, measures **5.457e-12**, and is tamper-verified red against the real pre-E1 builder); device gate written but PARKED | **built, structurally green, device gate blocked** 2026-08-11 (merge `0507a1e5`). Up-proj + GeLU + J7b's down ring as three herds of ONE segment per 64-row band; retile seam at zero cost via the shim's 4-D pattern; GeLU→down fans through a memtile by port arithmetic; both GEMM herds share one kernel object (a hard −D-symbol constraint fixing `group_n`). **Blocked by a pinned compiler crash**: `air-fuse-channels` segfaults on ≥3 mutually-mergeable sibling `scf.for` channel nests (N=2 fuses 5/5, N=3 crashes 5/5; nondeterministic under aircc — ASLR — deterministic under `air-opt`; use-after-free shape in the NFL merge path). R1's down feed presents 4 such nests, forced by H5. Reproducer: `agents/probes/probe_fuse_channels_sibling_nests.py`. Gate parked `UNSUPPORTED` with the blocker named, not flaking the suite. Queue item 6a — **`[2026-08-11]` FIXED in source** (roles kept disjoint + per-destination 1 + k trip counts, regression lit `fuse_channels_sibling_nests.mlir`); the gate then ran through `build-xrt/python` and hit the next wall — shim BD exhaustion, item 6b |
| First cost-decomposed ladder | none; a measurement, walked twice — and a **machine anomaly** | **recorded** 2026-08-10 ([32](32-cost-decomposed-ladder.md)). Bytes walk-identical; warm DRAM ordering now `runlist` < `fused` < `coarse` < `offload`; reconfiguration columns on first walk. **Latency quarantined** at recording time (`hw_context` loads ~78–80 ms against ≤2.6 on 2026-08-09); **`[2026-08-11]` anomaly RESOLVED — it was the NPU pmode**, reset from Turbo to `Default` by an overnight self-reboot at the exact onset; Turbo re-set → verdict rung **156 ms, 3.7 ms/load**, healthy. The walk's latencies stand as `Default`-conditional; trap 0 has the standing rule |
| `layer_norm` offset-regime row | `run_npu2_layer_norm_peano.lit`, three shapes | **done** 2026-08-10 (`b4fe19a3`). The large-mean boundary is a pinned catalogue row: `mean_rel_L1` 9.819e-5, `atol_required` 0.0 — a one-pass revert now fails the suite, not a probe nobody runs |
| `addnorm` two-pass f32 + offset rows — item 7 | `run_npu2_addnorm_peano.lit` (both variants + both offset rows); suite green; provenance re-measure via devq | **done** 2026-08-11 (merge `9278be34`). Both fused variants moved to two-pass f32 (J7a's discipline; staged one-pass forms stay, undispatched and documented). Offset rows' first hardware run: `mean_rel_L1` 1.390e-3 / 1.409e-3, **`atol_required` 0.0** on both — against the one-pass kernel's measured 22.2 / 33.1 collapse in the same regime. Provenance refresh: `block`/`coarse` 1.688e-2 → **1.663e-2** (margin 1.35× → 1.43×), `runlist` 1.746e-2 (worst element improved), `fused` **unchanged to the digit** (its tail is the layer_norm path). No tolerance widened |
| `air-fuse-packet-put-loops` decline diagnostic | four `-verify-diagnostics` cases; pass lit tests 2/2 | **done** 2026-08-10 (`1b15a1b0`). Doc 23's proposal as specified: ≥2 same-bounds packet put loops left unfused at trip count > 1 warn with channels and trip count; one-trip and different-bounds declines verified silent |
| R2 order-seam scoping | hermetic probes with negative controls verified failing; no device run | **done** 2026-08-11 ([31b](31b-r2-order-seam.md)). **Corrects doc 31's seam-2 prediction**: the re-mapping that closes the order seam is partitioning the GEMM herds by **rows (M)**, not re-mapping the norm tail to band order — a norm emits whole rows, a GEMM's A operand is a blocked column strip of all rows, and no row→tile re-mapping bridges that. With row-partitioned herds the retile is local to one core (compute, not a BD), so the frozen-BD rule never applies and `hidden` never materializes. Probes re-run independently at merge: design arm routes 4 core→core with **every `aie.dma_bd` offset 0**, `l2_staged` control refuses at `'aie.memtile_dma' op has more than 48 blocks`, budget arm places 8 herds of width 4 and refuses 9. Opened queue items 8, 9, 10. Everything dump-derived is PROVISIONAL |
| Phase F — items 4 and 3-portable | study host suite green at its pinned counts; device legs via devq | **advanced** 2026-08-11. Item 4 is **four of five** — two of the "pending" modules were already done under other names, and `memcpy_bandwidth` is re-scoped (iron's operator has no AIR equivalent, and one of its four case axes, `num_channels`, is not an input here — routing produces it). Item 3's **named portable tier closes**; three modules deliberately unported (including `npu_runtime_checks.py`, superseded by the *stricter* `require_turbo`). `resource_usage.py` finally consumes the artifact item 2 pinned and nothing had read, adding `core_to_core_flows` — **doc 03's AIE-role-style axis is now measurable per design** rather than per hand-written gate. Host suite **103/103 → 231/231 in 17 modules**, pinned literal moved with it and re-verified through FileCheck in both directions. First hardware run of the component table: **half of `offload`'s layer at 1024 (80.0 ms of 159.8) is unattributed host overhead**, with `sync 90` / `bytes 99090432` matching doc 03's recorded steady state exactly. No package installed |
| R1 column census — queue item 10 | `check-ffn-resident-structure` green with the census's own negative control refusing an over-budget design; the re-pinned literal FileCheck-verified in both directions | **done** 2026-08-12. Arm (c) counted `shim→core` circuit flows only and read **4/16, worst column 1** where the truth is **7/16, worst column 2** (an L2-staged refill is a `shim→memtile` flow on the same port). Widening to both flow kinds was **necessary but not sufficient** — over budget AIR emits *packet* flows sharing one queue, so a port count reads **0** on an over-budget column; the clause counts per-column MM2S **demand** instead. Literal **re-derived** from `build-xrt` aircc 2026-08-11 13:28:03 (sha256 `5cb08407`), not carried from the pre-6b dump — and it reproduces 31b §3.6 column for column. Negative control (3 herd-direct L3 streams → census 3, refused; 12 `aie.packet_flow` measured) runs inside the gate on every invocation |
| G — unattended runner + CI | Full profile run completes with a complete `results_manifest.json` | **G0 done** 2026-08-12. `study/profiles.py` + `run_profile.py`: three profiles over the one reachable family, every count **computed not typed**, so retargeting a profile retargets its gate; the five unreachable families are named with reasons and `test_profiles.py` **re-derives that claim from the sources** (ast over `opcheck_specs.py`), so a future row fails a test rather than making the table a lie. **Both blind checks closed**: `run_status="skipped"` had been in the schema since v1 with nothing emitting it, and the manifest validated FILES — a CSV holding 1 of 9 rungs reported `complete: True`, demonstrated on the real `postflip-ladder-w1` and correctly refused under the new row counts. **Ran**: devq 256, cold, 347 s, smoke 4/4, `smoke_gate` PASS, `tree_dirt_after_run` empty; no latency quoted, because one walk is not a result. **CI leg 1 → 10** PR-safe tests via an allowlist (not `--filter-out`, so an NPU test cannot join silently); doc 10's block was wrong three ways, all measured. Four of doc 10's behaviours **dropped** with evidence, collapsing passwordless sudo to `xrt-smi configure` alone. **Left**: resume, and the coverage sweep that would make `full` mean the declared 6×9 matrix — `run_mode.py` hardcodes `encoder_bert`, so one of six families is reachable |
| Goal 1 — sliding window | `make verify` passes with window-crossing prompts | not started |
| Goal 2 — quantization | Second quantized model passes a gate that exercises the quantized path | not started |

## `[2026-08-12]` Where things stand, for a session picking this up cold

**Verify NPU power mode is Turbo before measuring any latency.** The 2026-08-10 "machine anomaly"
(hw_context creation ~78–80 ms/load against ≤2.6 the day before) was **resolved 2026-08-11: it
was the `xrt-smi` power mode**, reset from Turbo to `Default` by an overnight self-reboot at the
exact onset — non-persistent state, which is why an `amdxdna` reload and a full reboot both
failed to clear it and nothing on disk had changed. At Turbo the verdict rung
(`study/run_mode.py --mode offload --seq 1024 --warmup 2 --samples 5`) reads **156 ms avg,
3.7 ms/load**, inside doc 29's 164–183 ms band; at `Default` the same rung reads ~2.5–2.7 s.
So the first hardware action of any session is
`sudo xrt-smi configure --device 0000:64:00.1 --pmode turbo` (needs the operator), verified with
`xrt-smi examine -r platform`, and **re-set after every reboot or driver reload**. Latencies
recorded on 2026-08-10 are `Default`-conditional; pre-08-10 records are Turbo-conditional
([32](32-cost-decomposed-ladder.md), trap 0 below). Byte and count instrumentation is
pmode-independent.

**`[2026-08-12]` The two owed verifications are GREEN, and both were measurements rather than
inferences.** The install refresh moved the ten shipped models onto the 2026-08-11 compiler, and the
claim that 6a/6b are no-ops for every shipped design had only ever been checked in the BUILD tree —
`POST-INSTALL TEN-MODEL: PASS`, all 10 verify under the refreshed install (devq 252, 65 min, Turbo
verified in-job). **Provenance caveat, self-reported and recorded rather than dropped**: a
concurrently-running agent dispatched one dry run to the device **off-queue** during job 252, which
held the device lock. **The verdict stands, and the reason is directional**: contention pushes a
correctness gate toward *false failure* — timeouts, `hw_context` exhaustion — and cannot make wrong
output match an HF bf16 token set, so a PASS obtained under contention is at least as strong as one
without. The run's wall-clock is not a latency figure and no latency was claimed from it. The
lesson is queued as item 19. And items 8 and 9, each verified only in its own isolated tree, were built together
for the first time: **`check-air-mlir` 494 pass / 0 fail** (devq 254), the predicted 492 baseline plus
one regression lit each. A third result falls out of the same build: R1's `npu.air.mlir` is
**byte-identical** before and after that integration, so neither compiler fix moves R1's runtime
sequence and every structural literal derived from the earlier binary carries over unchanged.

**`[2026-08-12]` The branch is the whole state — there is nothing unmerged anywhere.** The three
parallel streams of 2026-08-11 (item 6b's compiler fix, R2's scoping, Phase F's items 4 and
3-portable) are merged into `exper/transformer-layer-execution-studies`, tip `d87c3701`, tree
clean. Four git worktrees stay registered under `.claude/worktrees/` — three agent worktrees and
`phase-f` — and all four are **fully merged (0 unmerged commits) and clean**, so they hold nothing
and are safe to remove; do not read them as pending work. ~~**What is NOT in the branch is the
install**~~ **`[2026-08-12]` The install is REFRESHED and the two trees now agree.**
`ninja -C build-xrt install` ran (no compile or link steps — `build-xrt` was already current, so it
was a copy); `install-xrt/bin/air-opt`, `install-xrt/bin/aircc` and
`install-xrt/python/air/_mlir_libs/_air*.so` are all **2026-08-11 13:28**, matching `build-xrt`
exactly. So both compiler fixes (6a's fusion correction, 6b's BD pacing) are now in **every probe
and model path** as well as in the lit suites, and the two resolution paths of
[15 §Which toolchain tree](15-environment-notes.md) no longer diverge. **Verified by artifact, not
by timestamp alone**: `agents/probes/probe_fuse_channels_sibling_nests.py --nests 4 --tries 5`
resolves `install-xrt/bin/air-opt` directly and now reports `{'ok': 5}` / "does not reproduce",
against the pre-refresh binary's deterministic 5/5 SEGV — and its aircc leg succeeded, which is the
same tree. 6b's pacing rides the same install but was **not** independently re-verified here; its
evidence stays the build-tree regression lit and `check-air-mlir` 492/0.

**All four modes are corrected, gated, and — as of late 2026-08-11 — fully separated by one
measurement.** The modes landed 2026-08-08/09 (`coarse` = cell C1 of [28](28-coarse-blend-space.md)'s
space, chosen by measurement, [30](30-coarse-cells-built.md)); 2026-08-10 added schema v2 — the
`device_ms`/`sync_ms`/`host_cpu_ms` decomposition plus `context_loads`/`kernel_attaches` as
columns for every mode — and the first cost-decomposed ladder ([32](32-cost-decomposed-ladder.md)).
**The standing cross-mode comparison is now [32 §The post-flip walk](32-cost-decomposed-ladder.md)**:
`offload` on its shared-xclbin default, Turbo verified, two walks — `fused` < `coarse` <
`runlist` < `offload` on averages and minimums at 512 and 1024, 16/16 orderings. Suite green on
NPU2 with the two-pass `addnorm` kernels (31 pass / 1 unsupported — the re-parked R1 gate — / 0
fail); the ten shipped models re-verified **10/10 under the new install on 2026-08-11** (item
1(c), closed). The same 31/1/0 re-ran unchanged under item 6b's compiler fix (devq 248), which
is the evidence that the BD-liveness step is a no-op for every design already under budget.

**Read the retractions, not just the claims.** The documents keep falsified claims with dated
retractions attached rather than deleting them — a sentence you find here may be marked false
three paragraphs later. Anything dated before 2026-08-08 predates the taxonomy correction and
ranks *implementations*, not the four modes; latency is pmode-conditional per trap 0 (2026-08-10's
records are `Default`, pre-08-10's are Turbo); [27](27-common-ladder-result.md)'s warm DRAM
ordering is superseded by [32](32-cost-decomposed-ladder.md)'s (the targeted pool eviction moved
`runlist` below `fused` and `coarse`); doc 29's original fallback sentence
(`--expand-load-pdis`) is falsified in place by its §The hardware verdict; and **every `offload`
latency/variance figure recorded before 2026-08-11 describes the now-retired ELF default** —
the current default is the shared xclbin, and the post-flip walk is the record for it.

**What 2026-08-11 settled, in one list.** Four queue items closed that day and the earlier
versions of this paragraph tracked each one twice as it moved; the retraction chains live in the
phase docs, and this is the settled state. Read the linked section, not this summary, before
citing a number.

- **Item 3 — `offload`'s shared xclbin is the DEFAULT, and the 4096 wall is gone.** The
  multi-launch dispatch experiment answered **NO**: in-stream `load_pdi` faults the firmware
  (`fatal_error_type 0x10`), and the scoped `--expand-load-pdis` fallback was **falsified** (a
  no-op on this aiecc; the claim had no artifact). Route 3 landed instead — the shared chain pins
  a `fused-cast` winner to the shape's `drain` row and runs 10/10 clean at 4096 with
  `context_loads 1` over 30 dispatches. `AIR_OFFLOAD_LEGACY_ELF=1` is now the ELF opt-in; no
  pinned gate literal moved. The owed re-walk ran the same day — **all four modes separate for
  the first time** ([29 §The hardware verdict](29-offload-n-streams.md),
  [32 §The post-flip walk](32-cost-decomposed-ladder.md)).
- **Item 6a — the `air-fuse-channels` crash was three defects, not one.** The diagnosed
  use-after-free, a silent N-way miscompile behind it (pairwise 2-slot wraps where 1 + k belong),
  and — found by the same-day review, which the first fix had faithfully *preserved* — a loose
  dynamic-offset comparison that let sibling nests reading different L3 slices fuse into clones of
  one transfer. Fixed in `AIRDependencyScheduleOpt.cpp` with a regression lit. **The sharp
  consequence: the old pass's lucky-green outputs were themselves wrong, so any structural literal
  derived from a pre-fix dump must be re-derived, never compared against.**
- **Item 6b — shim BD exhaustion is fixed, and the wall had been mis-attributed twice.** The
  offending feed is `hidden` at **96** tasks (not the down feed at 24), 97 live BDs against tile
  (1,0)'s 16. `ea3b98ce` paces the offending MM2S feed with completion-token awaits in
  `airrt-to-npu`. **Awaits, not frees** — a compiler-inserted `dma_free_task` before completion is
  a race. The other J1 candidate, loop-shaped BD programs, is **closed as arithmetically
  unavailable** for a retiling feed (all four hardware BD dimensions in use, no mergeable pair).
  No-op unless a tile is over budget, which is why nothing shipped moves: `check-air-mlir` 492/0,
  suite 31/1/0 against the final binary. **Stated evidence gap: R1 is the only module that
  triggers the recycling and it hangs on 6c, so the pacing is verified at pass and compile
  altitude and NOT on hardware.**
- **Item 7 — `addnorm` is two-pass f32 and the variance cliff is pinned.** The offset rows' first
  hardware run measured `atol_required` **0.0** on both variants, against the one-pass kernel's
  22.2 / 33.1 collapse in the same regime; the provenance refresh moved `block`/`coarse` to
  1.663e-2 (margin 1.43×) and left `fused` unchanged to the digit, as its tail predicts
  ([23 §2](23-rules-and-open-items.md)).

**The one thing blocking the study's definitional gap is item 6c.** `fused` is *packaged* today
and *resident* by definition ([03 §The vocabulary](03-measurement-model.md)), and R1 — the
one-segment FFN interior — is the increment that closes it. R1 is built and structurally green,
but **its device gate has never passed**: with BDs bounded the numeric arm compiles for the first
time and then times out on hardware (`ERT_CMD_STATE_TIMEOUT`) because the shim issue order is
channel-major while R1's consumers need round-major interleave. `air.preserve_shim_dma_order` was
measured and does **not** reach it — the grouping is made upstream by `air-dma-to-channel`'s
per-channel hoisting. So **`fused`'s SPECS atol stays PROVISIONAL**, the emulation tests and the
structure arm are the standing evidence, and no resident-tail latency or byte figure has been
measured on hardware. Item 6c was scoped rather than attempted because the fix is structurally
larger than 6b's brief ([31 §Wall 5](31-fused-resident-tail.md)).

**Read in this order.** [03 §The taxonomy](03-measurement-model.md) for what the four modes mean —
that is the definition and it is current — and its §The vocabulary (submission against dispatch,
packaged against resident, knobs against costs). Then this section, then
[23](23-rules-and-open-items.md) for the design rules that have cost sessions (the per-column shim
budget, the L3-side offset rule), then [32](32-cost-decomposed-ladder.md) before ANY measurement.
For the resident-`fused` phase, [31](31-fused-resident-tail.md) is the spec and
[31a](31a-resident-byte-floor.md) the byte floor. [26](26-mode-rebuild-feasibility.md) stays worth
reading *for its corrections*, not as current state.

**If you are picking up `coarse`, do not re-derive it.** [28](28-coarse-blend-space.md) is the
scoping and [30](30-coarse-cells-built.md) is the result; the two sibling cells stay runnable
(`make check-coarse-c2`, `check-coarse-c3`, `make coarse-cell-structure`) precisely so that
re-deciding the blend costs a measurement rather than a rebuild.

**The one-paragraph version.** The modes are not defined by *who sequences the work* but by
**reconfiguration cost against DRAM traffic**: `runlist` pays per-operator reconfiguration with
everything on device, `offload` minimizes reconfiguration (one xclbin, N instruction streams —
matching iron; all LINEAR operators on the NPU, all NON-LINEAR on the host), `fused` eliminates DRAM
traffic between operators, and `coarse` blends `runlist` and `fused`.

### The work queue

`[2026-08-12]` **The open rows are 6c, 12, 16, 17, 18 and 19, and they are not sequenced** — 6c
blocks only the resident tail, and everything else is independent of it and of each other. The
struck rows are kept because their evidence chains are cited elsewhere; read the bold verdict, not
the whole cell.

**`[2026-08-12]` Four rows closed that day and two are new — and the compiler half of the queue is
now empty.** **8** (the two-symbol offset map, sized from the map it composes — plus two further
hardcodes of the same class the fix uncovered, one of them silently producing wrong addresses at two
symbols), **9** (the silent multi-get memref shrink, fixed with a real extent computation) and
**10** (the column census, which now counts per-column MM2S *demand* over both flow kinds and over
packet flows, with its negative control running inside the gate on every invocation) all landed.
**A consequence bigger than any of the three: BOTH of R2's design constraints have moved.** 31b
avoided literal-offset L1 bands because of 9, and flattened its refill nest to one loop to dodge 8.
Neither constraint exists now, so **R2's design should be re-derived rather than inherited** — and
31b marks everything dump-derived PROVISIONAL for exactly this reason. **11** closed as DEFER —
11(a) turned out to be 11(b)'s prerequisite rather than its sibling. **12**'s three options were
each scoped
([34](34-phase-g-scoping.md), [35](35-goal-1-sliding-window-scoping.md),
[36](36-goal-2-quantized-scoping.md)) and remain a decision for the operator rather than a task.
**13 and 14 were spun out of those investigations** — a missing pmode guard on the two
latency-asserting gates, and the int4 model's disabled verify lit. Neither was visible from the
queue before, and 13 is the one that would ambush an unattended runner.

**`[2026-08-12]` later that day: 11 closed as DEFER, 12 investigated, and 13 + 14 are new.** Item
11(a) turned out to be 11(b)'s prerequisite rather than its sibling, so it defers to it and the row
is closed either way. Item 12's three options each have a scoping doc now ([34](34-phase-g-scoping.md),
[35](35-goal-1-sliding-window-scoping.md), [36](36-goal-2-quantized-scoping.md)) and remain a
decision for the operator. **Items 13 and 14 were spun out of those investigations** — a missing
pmode guard on the two latency-asserting gates, and the int4 model's disabled verify lit. Neither
was visible from the queue before, and 13 is the one that would ambush an unattended runner.
**13 is now CLOSED, same day** — both gates refuse off Turbo, the refusal is matched by the lit so
it cannot be quietly deleted, and the floor file records the mode it was measured at. ~~**14 is still
open.**~~ **14 is now CLOSED too, same day** — the CI-runner OOM it was disabled for does not
reproduce on this 31 GiB host with or without the subprocess split (10.53 / 12.57 GiB peak, devq
255, both arms PASS), and the lit is re-enabled and green. One thing 13 leaves behind, deliberately: the shipped floor's own pmode is `unknown` and is
*flagged, not refused*, because re-seeding it to clear a flag is the move the driver-owned floor
file exists to prevent — an operator re-seed between phases is what resolves it. **And closing 13
turned up a third gate of the same class, now item 15**: `compare_roots.py` gates on
`avg_latency_ms` with no pmode guard, and being a *comparison of two recorded roots* it is the
literal case trap 0's closing sentence forbids. It cannot be fixed the same way — a live
`require_turbo()` says nothing about a run recorded last week — so it is blocked on doc 34's M4.

| # | Work | Size | Spec |
|---|---|---|---|
| **1** | **The shared `offload` path now runs AND GATES at 4096 — the mode's own spec shape** (2026-08-11, route 3 of the hardware verdict). The chain of findings, in order: the five-shape chain builds at 4096 (`623768f2`, compile half); the multi-launch dispatch experiment answered NO — 29 single-launch dispatches clean off the shared xclbin, the one multi-launch module (`fused-cast` down-proj) **faults the firmware** (`fatal_error_type 0x10`) — and the scoped fallback was **falsified** (`--expand-load-pdis` is a no-op on aiecc 1.4.0's raw-insts edge; the ELF ABI's multi-launch works via aiebu-embedded PDIs; the "19→174 KB" claim had no artifact). **The landing: `offload_config` pins a `fused-cast` winner to the shape's `drain` row under the shared path** (~10% priced on that GEMM, ELF path keeps the winner), measured 10/10 stages clean with `context_loads 1 kernel_attaches 4` over 30 dispatches, byte provenance exact (cold delta 293,200 = the five insts streams; −12,582,912 vs ELF = the fused-cast f32 C scratch), and `run_npu2_offload_peano.lit` re-ran green with all three recipes at 4096. ~~**Remaining of (c): the ten-model regression**~~ — the suite half re-ran the same day: **30/30** (519.7 s, 24 workers, the drain-pinned 4096 recipe among them; the runlist gate's no-margin latency clause held under this contention), and **`[2026-08-11]` later the ten-model half re-ran under the new install: 10/10 PASS** (devq job, gate-e1 leg-2 procedure, ~66 min). **Item 1 is CLOSED** | done | [29 §The hardware verdict](29-offload-n-streams.md) |
| ~~1b~~ | **RESOLVED 2026-08-11 — the anomaly was the NPU power mode.** The reboot did NOT clear it (82.5 ms/load) and a held-context pin refuted runtime PM (82.2 ms/load with the device active); the discriminator was the boot history — the machine **rebooted itself at 01:09 on 2026-08-10, the exact onset**, resetting the non-persistent `xrt-smi` pmode from the **Turbo** the healthy window's uninterrupted boot had carried (set for C4's `require_turbo()` sweep) back to `Default`. Confirmed by re-measure: Turbo → **156.2 ms avg, 3.7 ms/load**, inside doc 29's band. **New standing rule: Turbo is a measurement condition** — re-set and verify it after every reboot/driver reload, before any latency run (trap 0 below); 08-10's recorded latencies are `Default`-conditional, pre-08-10's are Turbo-conditional, bytes/counts unaffected | done | [32](32-cost-decomposed-ladder.md) |
| ~~2~~ | **DONE 2026-08-10.** Schema v2 (`eeb37a19`): `device_ms` / `sync_ms` / `host_cpu_ms` plus `context_loads` / `kernel_attaches`, appended after the pinned v1 prefix, semantics written per field. First hardware rows at 1024 confirm every known truth — `offload`-ELF 30 loads with 18.8 ms of host compute, `runlist` 24 (its per-head eviction, a measured middle regime), `coarse`/`fused` 0 — and both pinned offload gate lines re-verified on NPU | done | [03](03-measurement-model.md) |
| 3 | **CODE HALF DONE `[2026-08-11]` — the default IS the shared path.** `AIR_OFFLOAD_LEGACY_ELF=1` is the legacy/control opt-in (the retired `AIR_OFFLOAD_SHARED_XCLBIN` raises on any setting), the two cache directories stay unmerged, and all three lit recipes re-pointed with **no pinned literal moved**: the default recipe runs with no env var and still pins `context_loads 1 kernel_attaches 4`; the legacy pair sets the var and still pins `context_loads 30 kernel_attaches 0`. ~~**Remaining — the owed re-walk**~~ **RAN the same day — item 3 is CLOSED.** Both walks, 8/8 rungs each, Turbo verified in-job, `offload` on the shared default (`npu_unique_xclbin_count 1` in every row): **`fused` < `coarse` < `runlist` < `offload` on averages AND minimums, both walks, both lengths — 16/16 orderings, all four modes fully separate for the first time** ([32 §The post-flip walk](32-cost-decomposed-ladder.md)). `offload`'s timed-region `context_loads` is 0 (standing context) against the ELF era's 30/dispatch; its ELF-era 120–316% spread is gone. One declared confound: the same day's two-pass `addnorm` rides in the three modes that dispatch it, so cross-walk comparisons against pre-flip records are not single-variable | done | [32 §The post-flip walk](32-cost-decomposed-ladder.md) |
| ~~4~~ | **DONE 2026-08-10.** `evict_attention_contexts` evicts by artifact now (`KernelCache.evict_pools_for`), so the content-keyed static-weight pools survive the per-head evictions. Measured: warm `runlist` bytes 190,513,152 → **176,160,768** (−14,352,384 — the static set, to the byte) and sync 451 → 443. **Cold totals unchanged**, so no gate literal moved; `check-runlist`, fault twin and `check-coarse-c3` re-ran green | done | [30](30-coarse-cells-built.md) |
| ~~5~~ | **DONE 2026-08-10.** Decided: the seam gate's verdict compares interleaved **minimums** (`agg_min < seq_min`, both legs) — doc 23 §1's own convention, since contention only inflates samples. The inequality was NOT widened; medians and the win count stay reported. Validated isolated: leg A 1.0054× on minimums, leg B 1.0148×, `PHASE B GATE: PASS` | done | [30](30-coarse-cells-built.md) |
| **6** | **The `fused` resident tail — R1 BUILT 2026-08-11; the gate RAN the same day and is RE-PARKED on wall 4 (item 6b).** The one-segment FFN interior (up-proj + GeLU + J7b's down ring, three herds per 64-row band) is structurally green: hermetic probe 8/8 clauses — **re-verified 3/3 against the FIXED fuse pass**, so the derived constants hold on the corrected fusion — ~~dataflow emulated element-exact in f64, 5/5 host tests~~ **`[2026-08-12]` read that clause with item 17's correction — the emulation arm never built the module; the rebuilt arm interprets it, measures 5.457e-12 and is 8/8**, both kernel objects compiling. Wall 3 (the `air-fuse-channels` crash, item 6a) is FIXED and the gate was armed and run via devq through `build-xrt/python` (no install refresh needed on this path): **STRUCT arm passes; the numeric arm hits wall 4** — shim BD exhaustion, deterministic and loud (item 6b). ~~SPECS atol stays provisional~~ **`[2026-08-11]` wall 4 is FIXED** (`ea3b98ce`, item 6b) and the numeric arm now compiles through the BD allocator for the first time — **and stops on wall 5, the channel-major shim issue order** (`ERT_CMD_STATE_TIMEOUT`, item 6c). The STRUCT arm re-ran PASS against the fixed compiler. SPECS atol stays PROVISIONAL; the emulation tests and the structure arm remain the standing evidence — **and `[2026-08-12]` the emulation half of that sentence is only worth citing from item 17 onward: before it, the arm did not build the module** | Unblock 6c, then re-arm the recipe + one gate run | [31](31-fused-resident-tail.md) §status · [31a](31a-resident-byte-floor.md) |
| **6a** | **FIXED IN SOURCE 2026-08-11, and the fix is CLOSED — the gate reached the next allocator the same day.** The crash was two defects wearing one stack trace: the diagnosed use-after-free (the pairwise loop had no merge *roles* — on a 3-clique the third pair put one channel's ops in both the fuse-destination and erased sets, and `wrapRegionsWithForLoops` clones-and-erases what the erase loop then reads) **plus a silent N-way miscompile behind it** (the NFL wrap hardcoded 2 time-multiplex slots; k absorbed sources need 1 + k — the LB/UB path already composed N-way via its attr increments). Fix in `AIRDependencyScheduleOpt.cpp`: roles kept disjoint across pairs; per-destination trip counts; loud decline if two destinations with different counts ever share a wrap region. **Verified**: N=3 old SEGV/hang → fixed 10/10; N=4 clean; **R1's own `pass_017` dump old 10/10 SEGV → fixed 10/10 clean**; `check-air-mlir` 491 pass / 0 fail; regression lit `fuse_channels_sibling_nests.mlir` verified failing on the old binary. **The same-day Codex review then found a THIRD defect the first fix had faithfully preserved** (its "N=2 bit-identical" check proved preservation of a miscompile): the loose dynamic-offset comparison let sibling nests reading DIFFERENT L3 slices fuse into clones of the destination's transfer, silently dropping the sources' data. The revision moves the test to strict structural equivalence per SIDE — identical-pattern sides multiplex at 1 + k, differing sides keep ALL their ops on the merged channel (func9's documented split shape; func13's expectation had encoded the miscompile and was corrected) — and closes two residual hazards (nested-op UAF via recursive region validation; multiplicative LB/UB×NFL mixing via per-destination strategy). **Sharper finding**: the old pass's *lucky-green* R1 outputs were themselves wrong, so structural literals derived from pre-fix dumps must be re-derived, not compared against. The gate then RAN through `build-xrt/python` (the install refresh turned out not to be on the gate's path; it stays owed for `install-xrt` consumers) and hit wall 4 — item 6b | done; the gate's next blocker is 6b | [31](31-fused-resident-tail.md) §status |
| ~~**6b**~~ | **FIXED `[2026-08-11]` — the BD wall is gone, and the gate is now parked one layer further down (item 6c).** The wall as recorded was mis-attributed twice, corrected from the emitted runtime sequence: the offending feed is **`hidden`**, not the down feed, at **96** tasks (sweeps 4 × k_steps 24 — the 4 is the sweep re-read, not `herd_x`), 96 + `w_up`'s 1 = **97 live BDs on tile (1,0) against 16**. Mechanism: AIR emits a transfer's BD release where the `airrt.wait_all` that joined its token was, and R1 joins every token at one segment terminator. **Fix `ea3b98ce`** (`airrt-to-npu`): a tile over budget has its offending MM2S feed paced with completion-token awaits — `issue_token` + `dma_await_task(t[i-depth])` before task `i`'s **configure** (the allocator hands the ID out there; an await one op later is one ID too late — measured, the first form refused at task 16 instead of task 0), plus a drain so every token is consumed exactly once, and the paced run sunk behind the feeds it must not out-order. **Awaits, not frees**: mlir-aie's own guidance is that a `dma_free_task` before completion is a race, so a compiler-inserted free has no argument to offer. **[23 §4](23-rules-and-open-items.md)'s other candidate — loop-shaped BD programs — is CLOSED as arithmetically unavailable for a retiling feed**: `hidden`'s descriptor already uses all four hardware BD dimensions with no mergeable adjacent pair, so the chunk loop would be a fifth. No-op unless a tile is over budget, so no shipped design moves: regression lit `shim_bd_liveness_bound.mlir` **verified failing** pre-fix, `check-air-mlir` **492/0**, transformer-layer suite **31/1/0** against the final binary (devq 248), structural probe PASS against the fixed pass (devq 239). **Evidence gap, stated**: R1 is the only module that triggers the recycling and it hangs on 6c, so the pacing is verified at pass and compile altitude but **not on hardware** | done; the gate's next blocker is 6c | [31 §Wall 4 is fixed](31-fused-resident-tail.md) · [23 §4](23-rules-and-open-items.md) |
| **6c** | **Wall 5 — R1's shim issue order is CHANNEL-MAJOR and its consumers are not** `[2026-08-11]`. With BDs bounded the numeric arm compiles for the first time and then **times out on hardware** (`ERT_CMD_STATE_TIMEOUT`, devq 235). Measured from the runtime sequence: the three coupled L3 feeds are issued `@air_channel_2` (`hidden`) ×96, then `w_up`, then `w_down` — but an up core cannot consume `hidden` chunk 0 without its `w_up` block (the memtile BD chain interleaves them A,B,A,B) and `hidden`'s L2 landing pad holds one chunk, so a feed cannot drain before its co-operand is issued. **`air.preserve_shim_dma_order` does NOT fix it** — measured, devq 236: with the marker the grouping is still `[ch2 ×96][ch3 ×96][ch4 ×96]` and the module still times out, because the grouping is produced **upstream** by `air-dma-to-channel`'s per-channel loop hoisting ([19](19-phase-j1-collapse-norm-dispatches.md) §"Why it is safe now", step 1) and the marker only *prevents further* regrouping. Second, independent order defect in the same dump: `w_down`'s deliveries are non-monotonic in its own K index (c-major, sweep inner) because H5 forces the sub-channel index literal → 4 textual instances → 4 sibling per-channel loops concatenated. ~~**Inference, marked**: no ordering of whole channel runs can satisfy R1; it needs round-major interleave.~~ **`[2026-08-12]` DESIGNED, and that inference is NOT established by the artifact it rests on** ([37](37-wall-5-order-seam-design.md)). The `[96][96][96]` grouping is from devq **236, marker-ON** — and `air.preserve_shim_dma_order` is a **folding** switch as well as an ordering one, verified in source at `AIRDependencyScheduleOpt.cpp:8499-8534` ("no per-channel BD regrouping/**folding**"; the early return "skips per-channel BD folding for the whole launch region"). So it says nothing about the unmarked build's fold state, where doc 31 §Wall 4's own table records `w_up` at **1 BD** — folded to one wide streaming transfer, which is not a run and cannot starve. The design also finds **two independent deadlocks, not one**: D1 inter-channel, and D2 — `w_down` c-major against a sweep-major producer — which deadlocks on its own with inter-channel order perfect. It **rejects both routes this row named** (A's blast radius is four passes keyed in writing to the hoist's output shape) and recommends **route E: builder-side, zero compiler change**, since `w_down`'s refill carries no channel index (H5 constrains the `CHANNEL_G` get beside it, not the refill) — with route C held as the durable follow-up. Weighs that 31b's R2 deletes the `hidden` crossings entirely, so heavy machinery would be bought for a configuration R2 removes. **Next step is one hermetic compile, ~13 s, no device** — §5's census settles all three unpinned facts, and row 2 (has 6b's sink already fired?) decides between route E alone and route D first  **`[2026-08-12]` THE CENSUS RAN — route E confirmed, E1 alone is the whole remaining structural fix, and the order this row records is REFUTED.** Three arms, no device. Measured on the **unmarked** build (`preserve_shim_dma_order` count 0 — the build the design argued had never been read): `hidden` **96** tasks, `w_up` **1** (one BD spanning the whole array), `w_down` **13** — so §1.3's premise holds and **route C's trigger is not met**. Emitted order is **`[w_up][w_down][hidden ×96]`**, i.e. **6b's sink already fired**, so **route D is not needed** and defect 2 is the sole surviving order defect. **This row's own sentence — `@air_channel_2 ×96, then w_up, then w_down` — describes a SUPERSEDED BINARY**: devq 235/236 ran 13:06/13:08, `AIRRtToNpuPass.cpp` relinked 13:28, and 6b's fix (`ea3b98ce`) is the only order-producing change in that window. Which of *the sink was not yet in 235's build* or *the order was carried from an earlier dump* is correct is **not established** — both scratchpads are gone. **E1 on a copy**: `w_down` collapses to **1** contiguous BD monotone in `(s,c,jj)` — defect 2 gone — with channel symbols **12 → 9** and compile 1.4 s vs 1.3 s, no blow-up; **E2 is inert by measurement**. **Two things gate committing to it**: E1 deletes the shared-buffer WAR chain the builder's own comment relies on, so its token graph is cross-nest and **its correctness is unverified** (emulation tests + a device run settle it); and the census binary **predates items 8 and 9**, with item 8's `air-split-l2-memref` squarely on R1's L3-offset path, so **re-take the census against the integrated build first**  **`[2026-08-12]` E1 LANDED AND THE GATE RAN — wall 5 is closed and wall 6 is confirmed on hardware.** The re-taken census against the integrated build (items 8 + 9) is **byte-identical** to the pre-integration one, so neither compiler fix moves R1's runtime sequence and doc 37's stated worry retires. E1 folds `w_down` **13 → 1** contiguous monotone BD; `check-air-mlir` 494/0; STRUCT arm PASS on both binaries against all four pinned literals. **The numeric arm then timed out** (`ERT_CMD_STATE_TIMEOUT`, `fatal_error_type 0x0`, devq 259) — **predicted in advance by item 18's lock-conservation bound**, which is arm-independent, so E1 is correct and insufficient. **The recipe is re-parked** on wall 6, its re-arm deliberately left uncommitted so the tree does not claim an armed gate. SPECS atol stays **PROVISIONAL**; still no resident-tail latency or byte figure on hardware. Note for whoever takes wall 6: 31a's 84.0 → 16.5 MiB is a whole-layer @1024 figure while R1 prices one 64-row band, so this gate cannot by itself produce that number | Wall 6 is item 18; E1 is landed and needs no rework | [37](37-wall-5-order-seam-design.md) · [31 §Wall 5](31-fused-resident-tail.md) · [23 §4](23-rules-and-open-items.md) |
| ~~**8**~~ | **DONE `[2026-08-12]` — the map is sized from the map it composes, and the fix uncovered two more hardcodes of the same class.** As filed: `air-split-l2-memref` lifted the expression out of the offset's existing `affine.apply` — whose map may reference N symbols — then rebuilt the replacement as `AffineMap::get(0, 1, add)`, naming symbol positions the map did not declare, so a two-level nest over an L3 operand tripped `willBeValidAffineMap` and SIGABRTed (**exit 134, 5/5**). All three sites now delegate to one `composeAffineMap` helper that shapes the map from the original and widens it to what the composed expression actually references, keeping `getNumInputs()` equal to the operand count. **Which site fires was MEASURED**: instrumenting all three shows the reproducer hits the *third* (`composeAffineExprFromSizes`), not the first as the composition path suggests — and the existing corpus exercises all three (18 / 6 / 128), so one-symbol behaviour stays covered. **Two further defects of the same class, found while fixing**: the `replace(...)` calls passed literal `(0,1)`/`(1,0)` result counts, which would drop a trailing symbol and abort on their own; and the `air.execute`-wrapped path bound operands via `getUsedValuesDefinedAbove`, which has no ordering guarantee against the wrapped apply's symbols — harmless at one symbol and **wrong addresses at two**. That one was latent and silent, the same class as item 9. **Verified**: regression lit covering both composition paths and both operand paths, negative control checked **per case** against the preserved unpatched binary (134 each, 0 patched); `check-air-mlir` **493/0** against a 492 baseline, and the same suite with only `air-opt` swapped for the unfixed binary gives 480/13 with the set-diff showing exactly one status change and **zero regressions**. Output checked semantically rather than for absence of a crash — both launch IVs survive and the overlap stride matches the pre-existing `test9` golden with only the base widened | done | [31b §3.4](31b-r2-order-seam.md) |
| ~~**9**~~ | **DONE `[2026-08-12]` — the silent shrink is fixed with a real extent computation, and it moves shipped output.** As filed: `pass_029` rewrote `memref<12288xbf16, 2>` → `<3072>` while leaving the gets at 3072/6144/9216, reading past the end with **no error, no warning** — the frozen-BD trap's class ([31b §3.2](31b-r2-order-seam.md)). **Root cause**: `getDataAccessShapeFromMemcpyOp` bounded the buffer from sizes and strides alone — offsets were used *only* to test emptiness — so N gets of equal volume at different offsets all measured as one get. **Fix is outcome (a), not a blanket refusal**: a new extent function computes `offset + size` per dimension, so a band whose gets reach only 6144 still shrinks to **6144**; declining with a diagnostic is reserved for offsets that cannot be classified. Offsets are classified against what the shrinkage fix-up actually does, so the two cannot drift. A latent out-of-bounds read on the empty offsets vector was fixed alongside. **Verified**: regression lit `shrink_memref_multi_get_band.mlir` (4 cases) **verified failing** on the unpatched binary at three distinct CHECK lines *and* on `-verify-diagnostics`; **481 pass / 12 fail against an unpatched 480 / 12** in the same isolated tree, identical failure list (the 12 are an absent `aircc` in an uninstalled tree; 480 + 12 = the main tree's 492). **Two consequences, recorded not absorbed**: five allocs in `loop_fusion.mlir`'s *output* stop shrinking — the same defect on channel ops, pinned by no CHECK, so it moved silently before and would have again (no shipped design depends on it: zero offset-bearing `ChannelGet` in `programming_examples/`) — and **`probe_r2_order_seam.py --arm row_band` now fails BY DESIGN**, since its clause E asserts that control must stay broken. **So 31b's reason for excluding literal-offset L1 bands is gone and R2's design should be re-derived, not left to inherit the better option silently** | done | [31b §3.2](31b-r2-order-seam.md) |
| ~~**10**~~ | **DONE `[2026-08-12]` — the census counts the rule now, and it can fail.** As filed: `ffn_resident_structure.py`'s column census counted `shim→core` flows only and read **1** for a column carrying **2** of the 2-per-column budget ([31b §3.6](31b-r2-order-seam.md); the row used to cite §7.1, which is the `-D`-symbol section — the measurement is §3.6). Widening it to both flow kinds was **necessary but not sufficient**, and that is the finding: over budget AIR does **not** emit a third circuit flow, it converts the design to **packet** flows sharing one shim queue, so a census counting surviving *ports* reads **0** on an over-budget column — blind exactly when it is needed. The clause now counts per-column MM2S **demand**, circuit flows by distinct source port (a broadcast is one stream) **plus one per packet-multiplexed stream. Re-derived, not carried**: `build-xrt` aircc **2026-08-11 13:28:03** (sha256 `5cb08407`, carrying 6a + its review round + 6b) gives **7 of 16 ports, worst column 2** — column for column identical to 31b's pre-6b table, and `4 shim→core + 3 shim→memtile` reproduces. Pinned in the verdict line (`shim MM2S 7/16 worst column 2`) and FileCheck-verified green on real output. **The negative control is inside the gate**: three herd-direct L3 operands per column → census reads 3, refuses; measured 0 inbound `aie.flow` and 12 `aie.packet_flow`. Tamper-verified both ways — the `shim→core`-only revert moves the literal to `4/16 worst column 1` (FileCheck red) *and* fails the control; the circuit-only revert leaves the literal at 7/16 and is caught by the control alone. Probe twin and `--arm shim` recount independently to the same figures | done | [31b §3.6](31b-r2-order-seam.md) |
| **11** | ~~**Phase F's remainder, and it is two unlike halves**~~ **`[2026-08-12]` INVESTIGATED — they are not two unlike halves. (a) is (b)'s PREREQUISITE, and (a) is DEFERRED to it** ([33](33-memcpy-bandwidth-scoping.md)). `roofline/run.py` raises `FileNotFoundError("Missing memcpy-bandwidth CSV … Run study.memcpy_bandwidth.run first.")` and makes `peak_bandwidth_gbps` literally the memory roof's slope — so memcpy is the roofline's **only empirical input**, and "nothing is blocked on it" is true exactly until 11(b) is taken. The row's own premise was wrong twice, both re-verified at merge: iron's "four case axes" are **two** (`SIZE_LADDER = (8388608,)` and `NUM_CHANNELS = (2,)` are single-valued — iron never varied channels, so dropping that axis loses nothing), and iron's kernel arm is `failed_validation` at 8 and 16 cores. A bandwidth ceiling would bound ~2.8% of the measured layer, and every decision available today survives a 2× error in it. **Caution if you quote the imported ceiling**: it is a band, not a point, and it is non-monotonic — see doc 33's merge-verification section. Original text follows. (a) **`memcpy_bandwidth` is re-scoped from a port to a device design**: iron's operator has no AIR equivalent, and one of its four case axes — `num_channels` — is not an input here at all, since routing produces it. Nothing is blocked on it; the component table already measures what the study needs. (b) **The plot/analysis tier stays blocked on an install**: `regenerate_plots.py`, `roofline/{run,test}.py` and every `plot_*.py` need matplotlib/pandas/seaborn, which **must not be installed while gates run**. That is a scheduling constraint, not a technical one — it needs an exclusive window. ~~the same window the owed `ninja -C build-xrt install` needs~~ **`[2026-08-12]` the install half is DONE** (refreshed and verified by artifact), so 11(b) is now the only claimant on that window | (a) a device design; (b) an exclusive window | [09](09-phase-f-study-harness.md) |
| **12** | ~~**Phase G and the two goals — never started, and never chosen.**~~ ~~**`[2026-08-12]` ALL THREE INVESTIGATED — the choice is now informed, and it is still yours.**~~ **`[2026-08-12]` CHOSEN: Phase G. G0 is BUILT and the CI leg with it** ([10](10-phase-g-unattended-runner-and-ci.md) §What G0 shipped). **One profile, one command, one manifest**: `study/profiles.py` (three profiles — `smoke` 4 rungs, `ladder` 16, `full` 36 — with every count *derived* from the tables, doc 10's own instruction) and `study/run_profile.py` (refuse off-Turbo → `run_lock` → power → walk → smoke gate → `results_manifest.json` + `profile_run.json`). **Two defects closed, in the order that makes them work**: `run_status="skipped"` had been in the schema since v1 with **nothing emitting it**, so a structurally inapplicable rung (`fused` outside its 256..1024 packing bound) was recorded identically to a broken one — `run_ladder.walk` now writes it and starts **no child process**; and the manifest validated FILES, so **a CSV that should hold nine rungs and held one reported `complete: True`** — it now checks three counts per file (rows, rows that must pass, rows that must be skipped) against the profile. `run_lock.py` and `power.py` had no callers at all; the runner is the caller both were written for, with the power block **run-level and labelled as such** (SoC watts over the whole walk, compilation included, no sensor here measures the NPU). **The CI leg's before/after is 1 → 10 PR-safe tests**, and doc 10's CMake item could not be applied as written: its target name already exists and **its filter matches 0 of 32 tests** (measured, `lit --show-tests` + lit's own filter semantics). The existing target keeps its name and loses its false "safe as a PR gate" comment — 22 of its 32 tests `REQUIRE ryzen_ai_npu2`; a second `-host` target carries an explicit **allowlist** of the 10, and the workflow **asserts the count** because a lit suite that selects nothing exits 0. **Reachability is NOT fixed and is not pretended away**: one of six families is reachable, the other five are named with reasons in every run report. **Four of doc 10's behaviours are recorded as dropped** with the measurement behind each. **G0 RAN END TO END** — devq job **256**, `measure`, cold, **347 s**: `smoke` 4/4 passed, `smoke_gate` PASS, **`complete: True` with `row_counts_checked: true`** and per-file `rows 1/1 passed 1/1 skipped 0/0`, the first manifest here whose completeness is a statement about rows. Its `profile_run.json` records `devq_job_id 256`, the five unwalked families, **`tree_dirt_after_run: []`** (the runner asserts doc 15's leak rule itself) and `power_backend rapl_package` at 3390/3465 samples — `power.py`'s first caller. **No latency is quoted from it**: one walk is not a result. The CI leg's 10 tests ran green too (devq **261**, `Excluded=352 Passed=10`, guard PASS). **And the guard caught its own author**: its first version asserted `Total Discovered Tests: 10`, which is **362** — `add_lit_testsuite` filters a whole-tree lit run and records the rest as `Excluded` — so it refused a fully green run (devq 258) until it was rewritten to `lib-guard.sh`'s invariant, then re-verified against the real log plus four synthetic refusals. **Still open**: resume, doc 10's README item 5, and the coverage sweep that would make `full` mean the declared matrix. Original investigation text follows. **ALL THREE INVESTIGATED — the choice is now informed.** Each has a scoping doc, and each investigation found that its own spec overstates or misdirects the work. **Phase G** ([34](34-phase-g-scoping.md)): doc 10's spec is substantially obsolete — `devq.sh` already replaced its whole serialization tier and pmode enforcement here *refuses* where iron warns; the gate sentence is satisfiable by existing scripts today. Its real blocker is **reachability** — one of six declared families is reachable (`run_mode.py` hardcodes `encoder_bert`, every SPECS row is `emb_dim 768`), so a "full profile" needs a Phase-C-sized sweep first. Recommends dropping four of doc 10's behaviours on measured grounds. **Goal 1** ([35](35-goal-1-sliding-window-scoping.md)): the kernel change is two branches in ONE function shared by all ten models, but a window is **correctness-only with zero speedup** (masking is element-wise, not tile-skipping), the speedup route is a hang path plus queue items 8 and 9 verbatim, and **the specified gate may be vacuous** — full causal is a superset of the window, so a degraded implementation would often still pass. Weights are absent and license-gated. **Goal 2** ([36](36-goal-2-quantized-scoping.md)): `make verify` DOES exercise the quantized path (31 of 32 gated tokens are int4 decode), and doc 12 **names the wrong second model** — hard shape asserts fail Llama-3.2-3B and leave SmolLM2-1.7B, which doc 12 never mentions, as the shape-exact candidate; the top risk is that a compatible checkpoint may not exist, checkable in an hour. **Still a decision, not a blocker** — but now a decision with three costed options and two spun-out items (13, 14) | A phase each; pick one | [34](34-phase-g-scoping.md) · [35](35-goal-1-sliding-window-scoping.md) · [36](36-goal-2-quantized-scoping.md) |
| ~~**13**~~ | **DONE `[2026-08-12]` — both gates now refuse off Turbo, and the refusal is itself gated.** Original finding ([34](34-phase-g-scoping.md) M1): `require_turbo` was wired into exactly three files — `run_mode.py`, `component_groups.py`, `sweep/registry_sweep.py` (which *refuses*, stricter than iron's warn-and-continue) — and **`gate-h.sh` leg 4 was not one of them** though it asserts decode throughput against a recorded 9.435 tok/s floor, with `run_npu2_runlist_gate.lit`'s latency clause the same class. At `Default` those run ~15–20× slow, so they fail **reading as a compiler regression** — trap 0's failure mode reaching the one place that halts an unattended run. **The fix reuses `require_turbo` rather than adding a second parser**, via `agents/scripts/port-loop/pmode_guard.py`. (a) **`gate-h.sh` gains a leg 0** that refuses before leg 1's build — up front because the check is milliseconds against legs 1–3's hour (the file's own cheapest-first rule) and because **leg 3 is exposed too**, since the transformer-layer suite contains the runlist gate — and **re-checks at leg 4**, since a driver reload during that hour resets the mode under the running gate. (b) **`runlist_gate.py` refuses and returns 2** before compiling, printing a banner `run_npu2_runlist_gate.lit` now matches, so **deleting the guard turns the lit red** — verified three ways through real FileCheck (banner+turbo passes; banner absent fails naming the missing line; banner reading `default` fails). (c) **The floor file now carries `npu_power_mode`**, `seed-throughput-baseline.sh` refuses to seed off Turbo and stamps the observed mode, and leg 4 refuses a floor/run pmode *mismatch*. **The floor was NOT re-seeded and no number moved** (11.1 tok/s, `floor_fraction` 0.85 untouched). **Both branches demonstrated** by putting a stub `xrt-smi` reporting `Default` ahead of the real one on PATH — which exercises the real parse path and leaves no bypass in the production one: `gate-h.sh` refuses at leg 0 and never prints a leg 1 banner, proceeds to leg 1 at real Turbo; `runlist_gate.py` exits 2 with no compile, prints the banner and compiles at Turbo; `pmode_guard.py selftest` **11/11 clauses in both directions**. Suites unmoved: study host **231/231 in 17 modules**, sweep 6/6 + 8/8, seam 32/32 + 35/35. **Two things stated rather than claimed.** The floor's original pmode is **not recoverable as an observation** — the 2026-08-06 seed falls inside trap 0's uninterrupted-Turbo window, but that is an inference from boot history, so it is recorded `unknown` and **flagged loudly rather than refused** (refusing would make the gate unpassable until someone re-seeds, which is the pressure the driver-owned floor exists to remove). And the runlist clause was **never observed failing at `Default`** — setting the pmode needs root; the guard rests on trap 0's standing rule and on the mechanism, not on a measured red **[inference]**. **Operational note:** five of the seven files touched are gate-defining (`guard_gate_files()` covers all of `agents/scripts/port-loop/` and every `programming_examples/**/*.lit`), so a phase whose base commit predates this will report them as tampering — take the next fingerprint baseline from a commit that includes it | done | [34](34-phase-g-scoping.md) · trap 0 |
| ~~**14**~~ | **DONE `[2026-08-12]` — the OOM does not reproduce, the gate is re-armed, and it PASSES in lit.** As filed: `llms/llama32_1b_int4/run_npu2_verify.lit` was `// REQUIRES: false`, alone among the ten, from `18d1dac2` (2026-06-18, "[CI] Disable llama32_1b_int4 verify: OOM on amdhx370 runner") — a CI host-memory problem, never an int4 defect. **The disable was already stale when it was found**: `verify_runner.py`'s `auto` gate phase, which splits NPU-capture and HF-compare into separate subprocesses, landed `7f2e03d8` **2026-07-14 — a month AFTER the disable**, and is the same change that let `nightlyPerfBenchmark.yml` drop its 3B/4B exclusion ("No LIT_FILTER_OUT needed", peaks ~24-27 GB). The 1B int4 model was never revisited. **Measured before deciding** (devq **255**, 31 GiB host — the size class the disable named, Turbo; peak RSS summed over the whole process tree at 5 Hz, cross-checked against `/usr/bin/time -v`): the lit end to end peaks at **10.53 GiB** with 17.3 GiB still free, and the **legacy single-process path** (`--gate-phase both`, the pre-fix shape, recompiling every kernel inside the measured process) peaks at **12.57 GiB** with 15.4 GiB free. **Both PASS — it does not OOM with OR without the split**, so the split is headroom, not what holds the test up. Re-enabled with `ryzen_ai_npu2, peano, hfweights_amd_llama_3_2_1b_...` and **no `hf_token`** (verify_adapter resolves tokenizer *and* HF reference config to one ungated AWQ repo shipping its own `tokenizer.json`; matches the bfp16-prefill sibling). **The lit then ran green**: `PASS ... run_npu2_verify.lit`, 1/1, 462.6 s. **Two corrections to this row as filed**: lit CI was not testing "zero quantized code" — `run_npu2_compile.lit` is live, so quantized *compilation* was gated and only quantized *numerics* were not; and the nightly's status collapse (fail > pass > skip) meant the model's reported verify status came from the bfp16-prefill sibling, so the dashboard read green for a gate that never ran. **Census of the same pattern**: zero other `REQUIRES: false` in `programming_examples/` (`transformer_layer/run_npu2_ffn_resident_peano.lit` is `UNSUPPORTED: true` but is the honest opposite — ~60 lines of reason, wall 5, queue item 6c, explicit re-arm instruction); **four bare undocumented `REQUIRES: false` in `test/airhost/`** (06, 33, 45, 51) from `7abdcbe9`, low-stakes (`ENABLE_RUN_AIRHOST_TESTS` is OFF by default, no workflow runs it, needs a VCK5000); and `run_npu2_profile.lit` carries an `hf_token` it does not need | done | [36 §1.9](36-goal-2-quantized-scoping.md) |
| ~~**15**~~ | **DONE `[2026-08-12]` — the condition is in the data now, and M4 closes with it.** As filed: `compare_roots.py` gates on `avg_latency_ms` at 15% (35% for `offload`) with `main()` returning 1 and **zero** `pmode`/`turbo`/`power_mode` matches, so a Turbo-vs-`Default` pair — ~1500–2000% drift — printed `VERDICT: PROBLEM` in the vocabulary of a code regression. It could not be fixed the way 13 was, because `require_turbo()` reads the mode NOW and this compares runs recorded EARLIER. **Half 1 (M4): a `conditions` block on the manifest, declared in `schema.CONDITION_FIELDS`** — `npu_power_mode` plus how it was obtained (`observed` / `probed_at_manifest_build` / `unknown`), the provenance verbatim, and when. **The versioning is the design decision**: a `results` COLUMN would bump `SCHEMA_VERSION` to 3, and `results_io.read_rows` rejects both a header and a version mismatch, so the bump that took 56 v1 CSVs out of every reader on 08-10 would have taken the 16 that survive — including the exact roots this tool is pointed at. So it follows the file's own `RESOURCE_FIELDS` precedent ("adding a table is not a version bump"): a new declared block, **`SCHEMA_VERSION` stays 2**, pinned by a test, and it is deliberately NOT in `_FIELDS_BY_TABLE` so nothing can write it as a CSV. **Half 2: `compare_conditions()` runs before any CSV** and REFUSES a known mismatch (a failure naming the pmode, and the gating fields drop to `[SPLICED]` so the tool cannot return a red *latency* verdict for a condition change) while FLAGGING an unrecorded one and continuing to gate. **The refuse/flag split is reasoned, not copied**: 13 flagged its floor because refusing left "re-seed until the gate passes" as the only exit from a driver-owned file; a results root is neither shipped nor driver-owned, so the exit from a mismatch is to re-walk — which is what trap 0 prescribes anyway — while an *unrecorded* root cannot be stamped after the fact and refusing there would make the tool useless against the whole recorded corpus. **Both branches demonstrated on the real recorded artifacts**: `postflip-ladder-w{1,2}` copied byte-identical, stamped three ways, verdict `OK` / `OK` / **REFUSED** on the same numbers. **Nothing recorded moved** — the pre-change binary and this one produce identical output over all 6 readable v2 root pairs (every number, every WARN/FAIL, every exit code; the only textual delta is the drift tag column widening to fit `[SPLICED]`), and 16/16 v2 CSVs still parse. `results/phasef_smoke` stays unreadable — **confirmed pre-existing**, identical 4 failures before and after. Two things found and left: the recorded ladder walks carry **no manifest at all**, which `compare_manifests` used to `SKIP` in silence and which is now a loud flag; and `compare_manifests` has been diffing a `toolchain` block **`manifest.py` has never written**. Host suite **231/231 → 265/265 in 17 modules** (+12 schema, +10 manifest, +12 compare_roots), and `run_study_host_tests.lit`'s pinned literal moved with it | done | [34 §M4](34-phase-g-scoping.md) · trap 0 |
| **16** | **`compare_manifests` has always diffed a `toolchain` block `manifest.py` never writes** `[2026-08-12]`, found while closing item 15. The loop iterates an empty key, so the toolchain half of a root comparison has **never** compared anything — it has been silently vacuous for as long as it has existed, which is the same shape of defect as item 10's census reading a column that goes blind exactly when it matters. Item 15 deliberately did not fix it: it added `xrt_version` and the toolchain pin to `CONDITION_FIELDS` as a **declaration** rather than a design, on the grounds that the pmode is the one that had actually cost a day and scope discipline beat opportunism. Two adjacent facts recorded there: the recorded ladder walks carry **no manifest at all** (previously a silent `SKIP`, now a loud flag), and a toolchain mismatch is exactly the condition that would make a cross-root latency comparison meaningless in the same way a pmode mismatch does — so this is item 15's own argument applied to the other half of the condition | Small: write the block, then let the existing diff do its job | [34](34-phase-g-scoping.md) · trap 0 |
| ~~**17**~~ | **DONE `[2026-08-12]` — the arm interprets the built module now, and it REJECTS the defect it was blind to.** **The deliverable first**: the *real* pre-E1 builder source (`918c202f`, unmodified, loaded and built — not a modelled injection) now comes out at **max |y − ref| = 4.716e+03** and takes the arm to **7/8** with clause 5 red and the lit's FIRST `CHECK` unmatched, while the shipped builder is **element-exact at 5.457e-12** and 8/8. The two modules differ exactly where doc 37 said: **3 textual segment-scope refills vs 6, four of the six landing in the same L2 buffer**. **What the arm was testing before**: nothing about the module. It imported `ffn_resident_pack_w_up` and re-derived every DMA pattern, sub-channel index and loop order by hand — and that packer is **byte-identical across the E1 revert** (AST-compared), so its 5/5 was *provably invariant* under the defect, not merely unlucky. **What it tests now**: it builds `build_ffn_resident_module` and INTERPRETS the `air.ir.Module` — every `air.dma_memcpy_nd`/`air.channel.put`/`get` at the offsets, sizes and strides the op carries, every `scf.for` at its real bounds, every `func.call` by the symbol the builder named, three herds as four concurrent actors each, four channels as FIFOs, f64 throughout — under **two named models**: (M1) `air-dma-to-channel`'s hoist (each TEXTUAL segment-scope dma is its own auto channel, its L3 side hoisted into a nest cloned from its enclosing `scf.for`s, siblings concatenated in textual order) and (M2) the memtile lock pairing (one staging buffer, one lock pair, the k-th round reads the k-th landing; **sibling auto channels landing in the same buffer share ONE stream** — that is what makes the c-major delivery visible). **The negative controls live inside the arm** and are FileCheck-matched: NC1 (the refill's `c` loop Python-unrolled back into sibling nests) reproduces the real pre-E1 builder **to the digit, 4.715995e+03 both ways**; NC2 (the `hidden` retile's two inner strides swapped — seam 1's off-by-one) lands at 5.23e+03. A control whose ANCHOR goes stale reports `STALE` and fails its clause — scoring 'not applicable' as 'rejected' would have been this same item again. **Tamper-verified three ways**: neuter NC1 → 7/8 and the `e+0x` match fails; break NC1's anchor → `STALE`, 7/8; install the pre-E1 builder → red on the first CHECK. Liveness pinned so clause 5 cannot pass vacuously (dispatch census 768/20/96 all DERIVED, channel put/get counts, **0 undrained staged streams**). **Steps 3.1 and 3.2 were indeed one arm** — there is no `make` target, there never was, and none was added: the recipe now says so in its own header, and what it adds over the bare script is the assertions. **Scope stated, not overclaimed**: the arm reads the AIR module, so it models addressing and delivery ORDER and models NEITHER wall 5's D1 nor wall 6's lock conservation (item 18) — a mismatch is element-visible here and a timeout on hardware. Suite unmoved: study host tests **299/299 in 19 modules**, the 10 PR-safe lits 10/10. **Census of the same shape nearby**: no other transformer_layer host arm is blind in this sense — all 23 modules call their namesake — but `study/test_ladder_report.py` never exercises `ladder_report.load`/`main` (it hand-builds the post-`load` row shape), four modules hand-transcribe a value production also computes (`test_block_cache.py:65` copies the gate SPECS row; `test_run_ladder.py:31`; `test_profiles.py:143`; `test_component_groups.py:38`), `pattern/test_blocked_attention.py` has **no negative control at all**, and — the direct sibling of this item — **every other host arm has BOTH a `make` target and a lit**, where the make target runs the script with no FileCheck, so the pinned counts those lits carry are not enforced under `make`. Filed, not fixed | done | [31](31-fused-resident-tail.md) §status |
| ~~17 (as filed)~~ | **R1's emulation arm is BLIND to the defect it exists to catch** `[2026-08-12]`, found while landing E1. `builders/test_ffn_resident.py` imports only `ffn_resident_pack_w_up` and **never builds the module**, so it cannot see a change to the refill nest at all. **Proven, not suspected**: re-imposing the exact c-major order defect E1 removes still printed **5/5**. The lit recipe just runs the same script, so steps 3.1 and 3.2 of the gate are one arm wearing two names. This is item 10's failure mode in the numeric half — a check that reads as present and is blind exactly where it is needed — and it is why E1's correctness rests on a *structural* substitute (memtile DMA programs byte-identical with buffer names erased) rather than on the emulation. **Any future claim of 'dataflow emulated element-exact' for R1 must state which arm actually ran** | Small to diagnose, real to fix: the arm must build the module | [31](31-fused-resident-tail.md) §status |
| **18** | **A lock-conservation bound predicted R1's timeout BEFORE the run, and the run CORROBORATED it** `[2026-08-12]`. On memtile (3,1), `l2_b_down`'s producer does `AcquireGreaterEqual 16` / `Release 16` while its four consumers do `1`/`1` each, init 16/0. Per mlir-aie's `AIEOps.td:1473-1476` acquire-GE **decrements by value** and release **increments by value**, so conservation gives 16·W ≤ 16 + 384 — **at most 25 of the 96 `w_down` refills can ever complete**. Three controls in the same module (`l2_a_up`, `l2_b_up`, `l2_h`) all use producer count = consumer count and balance exactly; `l2_b_down` alone is 16, and `air.refeed_count` — the documented feature that legitimately sets acquire = release = N — is **absent from the entire pipeline**. **The gate then ran (devq 259) and the numeric arm timed out**: `ERT_CMD_STATE_TIMEOUT` with `fatal_error_type 0x0`, a clean hang rather than the firmware fault (`0x10`) the offload multi-launch produced. **Corroborated, not proven** — the prediction was made in advance, is specific and closes arithmetically, which is far stronger than a post-hoc explanation, but a timeout admits other causes and where the 16 is computed is still not located. **Arm-independent**: identical in baseline and E1, so E1 neither causes nor fixes it | Find where the 16 comes from; that is wall 6 | [37](37-wall-5-order-seam-design.md) §4 · devq 259 |
| **19** | **A compile-only flag that was parsed and never branched on dispatched to the device off-queue** `[2026-08-12]`, self-reported by the Goal 2 agent. The run went to hardware **while devq job 252 held the device lock** for the 65-minute ten-model regression. 252 passed and the direction of risk means its verdict survives (contention causes false *failures*, not false passes), **but that was luck rather than design**. Two things worth fixing rather than remembering: a `--compile-mode` that is parsed and unused fails **open**, straight onto shared hardware, so the default should be the safe branch and the dispatching path should be the one that must be asked for; and nothing outside devq stops a dispatch, so the lock is advisory against any script that does not consult it. Related: the shared-scratchpad clobbering the R1 agent defended against with a fail-loud wrapper — same class, both are *unenforced* conventions in a now-parallel workflow | Small: fail closed, and consider making the device lock non-advisory | devq · [23 §3](23-rules-and-open-items.md) |
| ~~7~~ | **DONE `[2026-08-11]` — `addnorm` is two-pass f32 and the cliff is pinned.** Both fused variants moved (mirroring J7a's layer_norm fix; staged one-pass forms stay, undispatched and documented); the offset rows' first hardware run measured `mean_rel_L1` **1.390e-3 / 1.409e-3** with `atol_required` **0.0** against the one-pass kernel's 22.2 / 33.1 collapse; the provenance refresh moved `block`/`coarse` to **1.663e-2** (margin 1.35× → 1.43×), `runlist` to 1.746e-2 (worst element improved), and left `fused` **unchanged to the digit** — correctly, its tail is the layer_norm path. Suite green with the new kernels; no tolerance widened | done | [23 §2](23-rules-and-open-items.md) |

**Order for a fresh session** `[2026-08-12]`.

1. **Trap 0 first, always.** Set Turbo, verify it, and only then measure — it resets on every
   reboot and driver reload, and a `Default`-pmode latency is ~15–20× off any recorded number.
   This costs one command and has already invalidated a day of measurement once.
2. **Read [03 §The taxonomy and §The vocabulary](03-measurement-model.md)** if you have not. The
   four modes isolate reconfiguration cost against DRAM traffic, and the knobs-and-costs map is
   the axis every measurement here is reported on.
3. **Then pick from the queue above.** The open set is small and none of it is sequenced:
   **6c** is the only item blocking the resident tail (and therefore `fused`'s definition);
   ~~**8**, **9** and **10**~~ **8, 9 and 10 all closed 2026-08-12** — the two-symbol offset map is
   sized from the map it composes (plus two more hardcodes of the same class the fix uncovered, one
   of them silently producing wrong addresses); the silent shrink now computes a real extent; the
   column census counts demand and carries its own negative control. **Read 8's and 9's rows before
   touching R2**: 9's fix invalidates 31b's reason for avoiding literal-offset L1 bands, and 8
   removes the abort that forced R2's flatten-the-refill-nest dodge — so **both of R2's design
   constraints have moved** and it should be re-derived rather than inherited;
   ~~**11** is Phase F's remainder~~
   **11 is closed** (11(a) defers to 11(b), whose only claimant is now an exclusive window);
   **12** is Phase G and the two goals — **all three now scoped, none chosen**, which makes it the
   one row that is a decision rather than a task; ~~**13** is a missing pmode guard~~ **13 closed
   the same day** — both gates refuse off Turbo now, the refusal is itself gated by a lit banner,
   and the throughput floor carries the mode it was measured at;
   ~~**14** is the int4 model's disabled verify lit~~ **14 closed the same day** — the OOM it was
   disabled for does not reproduce on a host of the size the disable named (10.53 GiB peak split /
   12.57 GiB single-process, devq 255), the lit is re-enabled and green, and lit CI now carries the
   tree's only quantized end-to-end gate — which had been reporting **green off a sibling test**;
   ~~**15** is the third unguarded latency gate~~ **15 closed `[2026-08-12]`, and with it**
   [34](34-phase-g-scoping.md)'s **M4** — the measurement condition is recorded in the manifest now
   (a new declared block, **no schema version bump**, so all 16 v2 CSVs still read) and
   `compare_roots` **refuses** a pmode mismatch while **flagging** an unrecorded one.

   **One operational consequence of 13, for anyone running a port-loop phase**: five of the seven
   files it touched are gate-defining (`guard_gate_files()` covers all of
   `agents/scripts/port-loop/` and every `programming_examples/**/*.lit`), so a phase whose
   fingerprint baseline predates this work will report them as tampering. Take the next baseline
   from a commit that includes it.
4. **If you take 6c, R2 is already designed.** [31b](31b-r2-order-seam.md) scopes the next
   increment so it starts from a design rather than a blank page — and it **corrects doc 31's own
   prediction** about which side of the order seam moves. After 6c: re-arm
   `run_npu2_ffn_resident_peano.lit`, one gate run, measured atol into the SPECS row.
5. ~~**Housekeeping the operator owes, whenever an exclusive window opens**: `ninja -C build-xrt
   install`~~ **`[2026-08-12]` DONE — the install is refreshed and verified by artifact** (cold-start
   section above), so probes and models resolve the 2026-08-11 compiler. Item 11(b)'s plotting
   packages still want an exclusive window; that half is unchanged.

Closed and not worth re-opening: items **1** (offload shared path gates at 4096; ten models 10/10
under the new install), **1b** (the machine anomaly — it was the pmode), **2**, **4**, **5**
(2026-08-10), **3**, **6a**, **6b** and **7** (2026-08-11, itemized above).

`[2026-08-09]` **`coarse` came off this list**, which is what makes item 1 the front of the queue —
see [30](30-coarse-cells-built.md). Two notes from that work that are not captured anywhere else:
the `prepare_fused` half of the extraction its spec asked for was **deliberately not done** (no cell
at 2048+ uses a stitched tail because none *builds* there, so it would have churned a gated mode for
a composition nothing calls), and items 4 and 5 above are both things the phase found rather than
things it set out to do.

### The four modes, as they actually are today

| Mode | State | What is left |
|---|---|---|
| `runlist` | **Corrected and gated** 2026-08-09. 427 entries over 17 runlists, nothing on the host. Per head `attn_scores` → `softmax` → `attn_output`, device-resident inside one submission. `[2026-08-10]` Its reconfiguration is now a **measured middle regime** — `context_loads` 24/dispatch, its per-head attention eviction — and the targeted pool eviction dropped its warm traffic by exactly the static set, putting it **lowest of the four modes in warm DRAM bytes** ([32](32-cost-decomposed-ladder.md)) | Nothing for the definition. One submission per head is a memory bound (~800 MiB if batched), not a schedule choice |
| `offload` | **Corrected and gated on BOTH halves** as of 2026-08-09. The LINEARITY half (2026-08-08): both attention matmuls on device, only softmax / both LayerNorms / GeLU on the host. The RECONFIGURATION half (`93e15a64`, [29](29-offload-n-streams.md)): five GEMM shapes in **one xclbin**, `context_loads 1` against the ELF path's 30, dispatch vector unchanged by design. `_evict_context`'s 30 reloads were the maximum reconfiguration cost in the mode defined to minimize it | **`[2026-08-11]` The 4096 wall is FULLY DOWN — the shared path runs and gates at the mode's own spec shape** via the `drain` pin (route 3: the dispatch experiment answered NO — in-stream `load_pdi` faults the firmware — and the `--expand-load-pdis` fallback was falsified, [29 §The hardware verdict](29-offload-n-streams.md)) — **and the default FLIPPED to it the same day**, per the recorded decision: the ELF path is the legacy/control opt-in (`AIR_OFFLOAD_LEGACY_ELF=1`; the retired `AIR_OFFLOAD_SHARED_XCLBIN` raises), no pinned gate literal moved. ~~One thing left: the four-mode re-walk~~ **The re-walk ran the same day** ([32 §The post-flip walk](32-cost-decomposed-ladder.md)): on the shared default the mode's timed-region `context_loads` is 0, its ELF-era variance is gone, and it now separates cleanly from `runlist` (slowest of the four on every statistic). Nothing left for the definition |
| `fused` | **Builds and gates again** 2026-08-08, at **1024** — it was red and unrunnable, pinned at a 4096 its own builder rejects. `[2026-08-10]` Its definitional gap (packaged, not resident) is now a **specced phase with a measured prize**: 84.0 → 16.5 MiB of DRAM crossings at 1024, the J7a+J7b composition already routing within the column budget hermetically ([31](31-fused-resident-tail.md), queue item 6) | One xclbin stays blocked for the *measured* reasons (the `[1,1]`/`[2,2]` conflict plus `air-fuse-channels`) — and `[2026-08-11]` the tail scope, which avoids those two, hit a THIRD: `air-fuse-channels` **segfaults** on ≥3 mutually-mergeable sibling channel nests (R1's down feed has 4; queue item 6a, reproducer shipped). **Later the same day: 6a's fix landed in source** — R1's dump fuses 10/10 into the correct single 4-slot multiplexed stream; the gate waits on the install refresh. R1 is otherwise built and structurally green. Whole-tensor residency stays capacity-bounded; streaming residency is not (per-stage L1 fits are seq-independent, [31a](31a-resident-byte-floor.md)) |
| `coarse` | **Corrected and gated** 2026-08-09. It is cell **C1 = (block front, banded tail)** of [28](28-coarse-blend-space.md)'s six-cell space, and the cell was **chosen by measurement**: the four cells that build at seq ≥ 2048 were walked twice at 2048 and 4096 and C1 is fastest on averages and minimums at both. The dispatch never changed — `builders/block.py`, as always — so what landed is provenance, plus the two sibling cells as runnable gates so re-deciding costs a measurement rather than a rebuild | Nothing for the definition. The `blend_cell` selection is recorded in the artifact; a workload that admits a stitched tail (seq ≤ 1024) would need a fresh walk, and there `fused` is available and dominates anyway |

### The first result on the corrected axis

Host↔device bytes for the same layer at 4096, decomposed — because the headline ratio is the
*weaker* of the two numbers here:

| | attention | everything else | total |
|---|---|---|---|
| `offload` (host softmax) | 830,472,192 | 139,984,896 | **970,457,088** |
| `runlist` (device softmax) | 25,165,824 | 165,347,328 | **190,513,152** |
| ratio | **33.0×** | 0.85× | 5.1× |

**On the attention component it is 33×**, and that is the number the taxonomy is about. `offload`
puts the softmax on the host, so each head's `[4096, 4096]` score matrix crosses DRAM in both
directions; `runlist` keeps it on the array and only `q_h`/`k_h_t`/`v_h` in and `ctx_h` out cross.

The 5.1× total **understates** it, because the two modes also differ in norm-chain granularity and
that difference runs the *other* way: `runlist` bands its two normalization points into 64 dispatches
each and pays 25 MB more than `offload`, which does its norms on the host. So this is not a
single-variable comparison in the strict sense — but the confound opposes the effect rather than
producing it, which is the direction that makes a result safe to read.

**A free provenance check that the decomposition is real:** `runlist`'s non-attention total,
165,347,328, is *byte-identical* to the total its gate pinned before the rebuild. The rebuild
touched attention and nothing else, and the arithmetic says so independently of anyone's account
of what changed.

Every earlier cross-mode comparison, [25](25-first-study-result-sequence-ladder.md) included,
differed in attention *placement* as well as in the variable under study, and was confounded in the
direction of its own conclusion.

### Five traps in the current state, before you measure anything

**0. `[2026-08-11]` The NPU power mode is a measurement condition, and it silently resets.**
The 2026-08-10 "machine anomaly" (`hw_context` creation ~78–80 ms/load against ≤2.6 the day
before, inflating the per-dispatch-load modes ~15–20×) was the `xrt-smi` **pmode**: the healthy
records were measured at **Turbo** (set for C4's `require_turbo()` sweep and carried by one
uninterrupted boot, 08-03 → 08-10), and an overnight self-reboot at 01:09 on 2026-08-10 — the
exact onset — reset it to `Default`. Confirmed both ways on 2026-08-11: at `Default` the verdict
rung reads ~2.5–2.7 s (82 ms/load, surviving a driver reload, a full reboot, and a held-context
PM pin); at Turbo the same rung reads **156 ms (3.7 ms/load)**, inside doc 29's band. **The
setting does not persist** across reboot or `amdxdna` reload, so the first hardware action of
any session is `sudo xrt-smi configure --device 0000:64:00.1 --pmode turbo` (operator), verified
via `xrt-smi examine -r platform`, then the verdict rung. Latencies recorded 2026-08-10 are
`Default`-conditional; pre-08-10 records are Turbo-conditional; bytes and counts are
pmode-independent. See [32](32-cost-decomposed-ladder.md) and queue item 1b — and re-measure a
whole comparison after any pmode change, never splice.

**`[2026-08-12]` This trap now lives in the data as well as in this prose (queue item 15, doc 34
M4).** `study/manifest.py` records the mode a run was measured at in a `conditions` block —
declared in `schema.CONDITION_FIELDS`, and a new BLOCK rather than a new `results` column precisely
so `SCHEMA_VERSION` stays 2 and no recorded CSV becomes unreadable — and `compare_roots.py`
**refuses** a comparison whose two roots record different modes, dropping its latency fields to
`[SPLICED]` so it cannot report a condition change as a code regression. **The prose rule above is
still the operative one for everything already on disk**: every root recorded before 2026-08-12
carries no such block, reads back as `unknown`, and is *flagged rather than refused* — a recorded
run's mode is not recoverable from its files, so **do not stamp one you did not observe**. Re-walk
to condition a comparison.

**1. ~~The modes are no longer at the same sequence length.~~ MEASURED `[2026-08-09]`, and the
comparison exists now — see [27](27-common-ladder-result.md).** All four modes were walked at **512
and 1024**, twice, 8/8 rungs each time. DRAM traffic orders exactly as the taxonomy predicts at both
lengths; `fused` is fastest and `coarse` second; `runlist` and `offload` are indistinguishable.

**The trap itself is unchanged for anyone reading the catalogue**, because the SPECS rows were not
moved: `fused` is still a 1024 row and the other three are still 4096. So a table assembled from the
SPECS rows still spans two lengths. **Build a cross-mode table from a ladder run, never from the
catalogue.** `fused`'s old `sync 19` against `coarse`'s 402 stays withdrawn; at 1024 the two are 13
and 107, and `coarse`'s 402 was a 64-band figure that does not survive the length change.

**Two things that comparison establishes about how to run the next one.** A single walk would have
published a crossover that a second walk refuted, which is the J3 failure repeating — so walk
anything twice. And `offload` drifts up to 120% within one walk against 2-10% for the other three,
enough to invert a ranking on its own, which is a fresh corroboration of the wider band
[03](03-measurement-model.md) already gives it.

**2. `attention_path` is no longer a covariate.** All four modes are on the device side now. The
first sequence ladder's headline — slopes splitting on attention placement, host 1.23–1.27 against
device 1.03–1.17 — **cannot be reproduced**, because no mode sits on the host side any more. A rerun
showing separated slopes is measuring something else and needs a new explanation, not the old one
restated. `study/test_attention_path.py` asserts that end state and will fail if a mode moves back.

**3. ~~`coarse` needs a decision procedure that does not exist~~ — RESOLVED 2026-08-09, and the
answer was measured rather than argued.** The space is two axes over six cells (derivation below,
still current); the four that build at seq ≥ 2048 were built, gated and walked twice; and `coarse`
is **cell C1 = (block front, banded tail)**, which is what it already dispatched — the phase bought
*provenance*, not a new dispatch. **The taxonomy did not collapse**: the winner is an interior cell,
so `coarse` stays distinct from both endpoints. The one thing to carry forward from the derivation
is *where* to measure, because at 1024 the answer would have been `fused` wearing another name. See
[30](30-coarse-cells-built.md). The rest of this section is the derivation that got there.

`[2026-08-09]` **A fused region is not free-form — it is an artifact somebody stitched**, and
reading the artifact plans makes the space small and specific. `fused_config`
(`pattern/fused/fused.py:381-420`) builds its front from `build_qkv_proj_module` and
`build_mha_out_proj_module` — **the same two modules `block_config` uses**. `fused` and `coarse`
therefore differ in the **tail alone**: `fused` has one stitched `ln1+ffn+ln2`, `coarse` has an
`ffn` ELF plus a row-banded `addnorm`. (A consequence for [27](27-common-ladder-result.md): the
7-10% `fused`-over-`coarse` latency gap it measures is *entirely* a tail effect, since the front is
identical by construction.)

So the blend has **two axes, not three**:

| axis | levels |
|---|---|
| **front** (qkv → attention → o_proj) | the `block`/`fused` form (two ELFs, q/k/v device-resident) · the `runlist` form (three projections, per-head `attn_scores`→`softmax`→`attn_output`, `output_proj`) |
| **tail** (ln1 → ffn → ln2) | stitched (`fused_tail`) · row-banded (`ffn` ELF + `addnorm` ×N) · fully decomposed (`runlist`'s up/GeLU/down + per-band add/LayerNorm/multiply) |

**2 × 3 = 6 cells, and two of them are already modes:** `(block-front, stitched)` **IS** `fused`, and
`(runlist-front, decomposed)` **IS** `runlist`. That is the sharp form of the scoping problem — the
space `coarse` is defined to blend over *contains the two things it blends*, so "pick the best cell"
would just re-derive one of them and collapse the taxonomy. On [27](27-common-ladder-result.md)'s
evidence it would collapse to `fused`, which is fastest AND lowest-byte at both measured lengths.

**The resolution is the word the definition already uses — *per workload*.** The cells are not all
available at every shape, and which are is a measured constraint: `fused`'s stitched tail caps at
1365 rows, so **at seq 2048+ the entire stitched row is unbuildable**. `coarse` is the mode you use
where `fused` does not fit, and today's `coarse` — `(block-front, banded)` — is already that cell,
chosen implicitly by D2 having been built at 4096. What it lacks is not blendedness but *provenance*.

**Consequence for sequencing, and it inverts what this README used to imply:** the corrected `coarse`
must be measured at **2048 or 4096**, not at the 1024 the other three now share. At 1024 every cell
is dominated by one that already has a mode name. Full derivation, the three interior cells worth
measuring, and what each costs: [28](28-coarse-blend-space.md).

**Any configuration selecting a fused level inherits `fused`'s 256..1024 bound**, so a corrected
`coarse` is a 1024 row — the same length [27](27-common-ladder-result.md) puts the cross-mode
comparison on.

**4. `[2026-08-11]` "Compiles today" is not "compiles tomorrow" for a module with ≥3
mutually-mergeable sibling channel nests.** `air-fuse-channels` has a pinned use-after-free
(queue item 6a): on such a module the SAME aircc binary over the SAME input is an **ASLR coin
toss** — measured 2 clean compiles then 2 segfaults, no change in between — while
`air-opt --air-fuse-channels` on the pre-fuse dump crashes deterministically 10/10. Two
consequences. A green compile of such a module proves nothing about the next run, so do not
gate anything on one; and a nondeterministic aircc segfault stopping after `pass_017` is THIS
bug, not your builder — check the shape against
`agents/probes/probe_fuse_channels_sibling_nests.py` before debugging IR. The boundary is
measured from both sides: N=2 fuses cleanly 5/5, N=3 crashes 5/5.

**`[2026-08-11]` later: FIXED in source (item 6a), with one sharpening and one residue.** The
sharpening: the coin toss's *green* outcomes were also wrong — the old pass's clean R1 compiles
left an extra channel alive with pairwise 2-slot wraps where one 4-slot multiplexed stream
belongs — so a structural literal derived from ANY pre-fix dump of such a module is invalid, not
merely unlucky. ~~The residue: aircc reads the pass from `install-xrt`, so this trap stands for
aircc until the operator refreshes the install from `build-xrt`~~ — **`[2026-08-12]` the residue is
GONE: the install was refreshed and the reproducer, which resolves `install-xrt/bin/air-opt`, is
now clean 5/5 with its aircc leg succeeding.** The trap is closed on both resolution paths. What
survives it is the *sharpening*, which is not about staleness at all: any structural literal
derived from a pre-fix dump of such a module is invalid and must be re-derived. (A same-day review round then also fixed the
pairwise semantics the crash had been hiding — heterogeneous-offset sides now keep all their
ops instead of erasing sources against a cloned destination; doc 31 §status.) The regression is
pinned by `mlir/test/.../fuse_channels_sibling_nests.mlir`.

### Two rules to know before designing anything

*A column has **two shim MM2S channels**, and the budget is per column **across the whole
segment*** — three stacked 8-wide herds put one tile of each into every column, so their L3 demands
add. Exceed two and AIR packet-multiplexes onto one queue. Keep every column at two or fewer
L3-facing streams; put the rest on L1→L1 channels, and pack co-indexed L3 operands into one strided
fetch. This explains why `fused`'s decomposed tail always ran 64 trips on 8 columns correctly, why
`addnorm` needed its one-trip guard, why J1's L2-staged weight failed, and why J7a works.

`[2026-08-12]` **Two things about *counting* it, both learned the hard way by queue item 10.**
First, an **L2-staged refill is a `shim→memtile` flow and burns the same MM2S port** as a
herd-direct `shim→core` fetch — a census that counts only the latter under-reports (on R1: worst
column 1 where the truth is 2). Second, and less obvious: **over budget, the routed design shows
FEWER shim flows, not more.** AIR converts the streams to `aie.packet_flow` sharing one queue, so
counting surviving *ports* on an over-budget column reads 0. Measured on a purpose-built control
(one herd of 4, three herd-direct L3 operands): zero inbound `aie.flow`, twelve `aie.packet_flow`.
A budget check must count **demand** — circuit flows by distinct source port, plus one per
packet-multiplexed stream — or it goes blind exactly when it matters.
`ffn_resident_structure.py` counts it that way and carries the control that proves it can fail.

*Advance a staged buffer on the **L3** side, never on the L2 read.* `[2026-08-07]` An
induction-variable offset is materializable on an L3 operand (the runtime sequence programs it per
task) and **inexpressible** on an L2/L1 one (an `aie.dma_bd` offset is static). The compiler does
not say so — it dereferences an unchecked `std::optional` and emits a chain that repeats a stale
offset forever, which presents as a hardware hang with no compile-time signal. J7b lost a session
to it. See [23](23-rules-and-open-items.md) and [24](24-phase-h10-non-constant-bd-offsets.md).

### Three things the mode rebuilds established that are not in any phase spec

- **`runtime_loop_tiling_sizes` is not inert, and the lowered IR will tell you it is.** `[2,2]`
  makes `mha_out_proj` @4096 compile and then **hang** — `ERT_CMD_STATE_TIMEOUT`, 3/3, against 3/3
  clean passes at `[1,1]` — while aircc's `aie.air.mlir` is identical op-for-op between the two
  settings. `omit_pingpong` is irrelevant at that shape and had been cited as half the reason. So
  the backend-settings conflict `fused.py` / `mha_out_proj.py` / `block.py` document is **real**, and
  doc 26 §4's compile-only refutation of it is retracted. Reproduce with
  `agents/probes/probe_backend_preset_hardware.py`.
- **Give every L1 buffer one role.** A buffer that is both a DMA destination and a kernel output
  does not read back what the kernel wrote. `builders/softmax.py`'s first version normalized into
  its own DMA-destination buffer — dead by then, and legal as far as the kernel's `__restrict` is
  concerned — and the design returned **the input unchanged** at all three shapes.
- **A normalization needs its fault-injection target chosen by measurement.** The standard
  `(rows-1, 0)` left softmax's negative control **passing** at two of three shapes: `+2.0` on a
  low-probability element moves the tensor less than `atol`, and at 512×512 the injected run's
  `abs_err_max` equalled the clean run's, so no `atol` admitting the clean run could reject it. The
  target had to move, not the tolerance.

### The device queue

**`devq` is the device scheduler and the migration is done** `[2026-08-08]`.
`agents/scripts/devq.sh` — builds run concurrently, a measure runs alone with no build in flight,
stale jobs reconcile by process liveness. **Use `devq.sh run`, not `submit`**: `run` is the drop-in
for `flock -x LOCK CMD` because it relays the job's output to stdout and exits with the job's
status, where `submit` diverts output to the job log and returns an id — substituting *that* at a
gate blanks the FileCheck while still exiting 0. It also refuses to nest. All 23 `flock` sites in
`phases.sh` and `llama32_1b_int4`'s `run-inference` are migrated; `make chat` keeps the bare lock
because the runner is `setsid` with stdin from `/dev/null` and a REPL under it reads EOF.
`devq-selftest.sh` is **20/20**.

### Compiler-side threads, unchanged by the mode work

- **H8 is untouched** and is the largest remaining compiler item: the pass that *derives* on-chip
  staging rather than having the builder declare it. It wanted J7 as a hand-written reference to
  validate against, and J7a and J7b are now two.
- **H10 ran and its substance passed** — five gate legs green, `check-air-mlir` 489/489, 11.44
  tok/s against a 9.43 floor. Its **tamper check halted** on five gate-defining files with
  documented provenance and was deliberately not re-fingerprinted; see
  [24](24-phase-h10-non-constant-bd-offsets.md).
- **J7a** ([21](21-phase-j7a-norm-tail-pipeline.md)) and **J7b**
  ([22](22-phase-j7b-accumulator-ring.md)) landed — the norm-tail pipeline and the compiler-formed
  accumulator ring. J7a's round-3 fix also made `layer_norm` ~26× more accurate for a measured ~13%
  throughput cost ([23 §1](23-rules-and-open-items.md)).
- **J1 is blocked, and precisely.** Not on correctness — H9 fixed the miscompile — but on shim **BD
  exhaustion at six trips** against a 64-trip target. It refuses loudly instead of corrupting
  silently, and it is **not on the goal path**: J7a reaches the same dispatch collapse without the
  packet queue.

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

## Environment and conventions, before touching anything

> **`[2026-08-09]` This section and everything below it is HISTORY plus setup.** It records how the
> port was built and what each phase left behind. For current state read §Where things stand above;
> the phase narratives below describe a Phase E that has since been superseded by the mode rebuilds,
> and their cross-mode numbers predate both the taxonomy correction and those rebuilds. Setup, the
> conventions and the environment traps are still live and still mandatory.

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

~~**The next phase is E (the four execution strategies).**~~ **`[2026-08-09]` Phase E ran, and
was then superseded.** Its five sub-phases all landed
([08](08-phase-e-execution-strategies.md) is the spec), and the taxonomy they were built against was
corrected on 2026-08-08. **All four** modes have since been rebuilt against the corrected
definitions; `08` and its sub-specs describe the superseded ones and carry reversal notes where a
claim was measured false. ~~**The next work is `coarse`**, and it needs a decision procedure that
does not exist yet.~~ ~~**`[2026-08-09]` `coarse` is still the next work, but the decision procedure
now EXISTS.**~~ **`[2026-08-09]` `coarse` is DONE** — [28](28-coarse-blend-space.md) derived the
decision procedure from the artifact plans (two axes over six cells, selected by what the workload
admits) and [30](30-coarse-cells-built.md) executed it: the two interior cells were built and gated,
all four buildable cells were walked twice at 2048 and 4096, and the mode is cell C1 with the
measurement recorded in its artifact. **The four modes are no longer what this plan is waiting on.**
See §The work queue (formerly §What to do next). ~~It is now the `offload` chain plus two small
items the `coarse` work found.~~ **`[2026-08-12]` The `offload` chain closed on 2026-08-11**; the
queue is now the resident tail's blocker (6c), three unclaimed compiler/census items, Phase F's
remainder, and the unstarted Phase G plus two goals.

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
| Can the whole sequence ladder be built? | **Yes**, since E1 made the `(method, tile_n)` naming fix in `llms/shared/builders/gemm_builder.py`. The symbol- and object-level collisions that pinned everything to `seq = 4096` are closed. | [08](08-phase-e-execution-strategies.md) |
| Is there tolerance headroom for four modes? | Thin but sufficient, and it did not shrink when attention moved on-device. Measured at the layer output against the hard `1e-1` ceiling: `offload` 1.73×, `runlist` 1.43×, `block` 1.35×, `fused` ~~1.27× (at 1024)~~ **`[2026-08-10]` 1.72× at 1024** — the repair run's `atol_required` 5.813e-2 ([26 §6](26-mode-rebuild-feasibility.md)). The 1.27× was the retired 4096-era row's figure (`atol_required` 7.896e-2, [23 §3](23-rules-and-open-items.md)), mislabeled 1024 here; it survives as cell **C2**'s margin ([30](30-coarse-cells-built.md)). **No tolerance has been widened for any mode.** | [07b](07b-phase-d2-block-integration.md) |
| Can the two attention matmuls run on this device? | **Yes**, both, at every ladder rung, `attn_output` by all three GEMM methods. The registry genuinely holds no `K = 64` / `N = 64` row and the sweep cannot stage one, but that is a *catalogue* constraint; the tiles are injected through the `gemm_spec_fn` hatch. Two failure clusters bound the space and are fully characterised: `herd_n = 1` at N=64 hangs, `tile_n = 8` fails the microkernel's own assert. | [26](26-mode-rebuild-feasibility.md) · `pattern/offload/offload.py` |
| Can one xclbin hold the whole layer? | **No**, blocked twice. `runtime_loop_tiling_sizes` `[2,2]` makes `mha_out_proj` @4096 hang on hardware (3/3) while `[1,1]` passes (3/3), and one ELF is one aircc invocation — so FlashAttention and the wide GEMMs cannot co-compile. Separately `air-fuse-channels` is O(N²) in channels and did not finish in 1200 s on a 90-channel stitch. | `agents/probes/probe_backend_preset_hardware.py` · [26](26-mode-rebuild-feasibility.md) |
| Is `attention_path` still a covariate? | **No, not since 2026-08-09.** All four modes run attention on the device. Any comparison whose explanation rests on that split cannot be re-tested. | `study/test_attention_path.py` |
| Does DRAM traffic order as the taxonomy predicts, across all four modes at one length? | ~~**Yes**: `fused` < `coarse` < `runlist` < `offload`.~~ **`[2026-08-10]` Warm, it is now `runlist` < `fused` < `coarse` < `offload`** — the targeted pool eviction removed `runlist`'s per-layer static re-uploads (its 55.2 MB at 1024 was exactly the static set above today's 40.9), and every pairwise gap decomposes to the byte. The taxonomy's *intermediate-traffic* story is unchanged; what moved was re-upload traffic that was never intermediate. The `offload`/`runlist` ratio remains O(seq²)-driven: 1.79× at 1024 against 5.1× at 4096 cold. | [32](32-cost-decomposed-ladder.md) · [27](27-common-ladder-result.md) |
| Which mode is fastest, at a common length? | **`fused`, at both 512 and 1024**, with `coarse` second — both survive two walks on averages and on minimums. ~~**`runlist` and `offload` are indistinguishable**: the two statistics disagree and each flips between walks.~~ **`[2026-08-11]` The post-flip walk separates all four**: `fused` < `coarse` < `runlist` < `offload` on both statistics, both walks, both lengths — the indistinguishability was the ELF path's variance, and the shared default removed it | [32 §The post-flip walk](32-cost-decomposed-ladder.md) · [27](27-common-ladder-result.md) |
| What is the widest common sequence ladder? | **512 and 1024 only.** Above 1024 is `fused`'s `plane_major` stride cap (1365 rows). 256 fails for `offload` and `runlist` on the injected attention tile — `256 % (tile_n 128 × herd_n 4) ≠ 0` — a **tile** constraint, not a hardware one. A third rung, and therefore any scaling exponent, costs a retuned and revalidated 256 tile. | [27](27-common-ladder-result.md) |
| How much does `offload` really drift? | **Up to 120% within a single walk**, and 23-32% walk to walk, against 2-10% for the other three — enough to invert a ranking by itself. Independently corroborates the wider band [03](03-measurement-model.md) inherited from iron. ~~Candidate mechanism, unmeasured~~ — **`[2026-08-09]` a REMOVABLE cause found, see the next row**, though not yet narrowed to eviction. And the drift is worse than 120%: a fresh ELF walk read **316.9%** at 512 | [27](27-common-ladder-result.md) · [29](29-offload-n-streams.md) |
| Is `_evict_context` what makes `offload` noisy? | **Not established — but the variance is removable.** Four interleaved walks, `runlist` as a same-conditions control in each: at 512 the intra-walk spread is **316.9% / 134.1%** on the ELF path and **17.6% / 14.0%** on the shared xclbin, in both walks, while the control stayed in band. **The intervention is not single-variable**: it stops the per-dispatch reconfiguration *and* changes the ABI, and the control rules out environmental drift rather than the ABI. Eviction is the leading candidate; the isolating arm is the xclbin ABI with eviction forced on, and that knob does not exist. Two further caveats — the 1024 rung did **not** reproduce its own 61.6%/59.8% baseline (it read 9.0%/10.5% on the same path), so do not quote the effect *size* from one measurement; and the shared path costs **~20% of best-case latency at 512**. The ELF arm's minimums reproduce [27](27-common-ladder-result.md)'s, which is what says the two measurements are of the same thing | [29](29-offload-n-streams.md) |
| Can the shared xclbin hold the whole mode at EVERY shape? | **No — it is bounded to SINGLE-LAUNCH modules.** It does hold all five at 512 and 1024. At 4096 the down-projection resolves to `fused-cast`, which is two `air.launch` ops, and `XRTBackend.compile`'s fixed `insts="air.insts.bin"` — passed as `-i` on the xclbin branch and omitted entirely on the ELF branch — makes aiecc refuse: *"edge 'air.insts.bin' produced duplicate output path"*. 1024 is where all five shapes are single-launch, which is why doc 29's landing figures are 1024 figures and why the gate sits there | [29](29-offload-n-streams.md) |
| Can N instruction streams share one xclbin? | **Yes — demonstrated, then LANDED** (`93e15a64`, [29](29-offload-n-streams.md)). `offload` runs five shapes from one xclbin at `context_loads 1` against 30. **It needs THREE distinct identifiers per stream and no caller set any:** `kernel_name` (duplicate ⇒ xclbinutil refuses the merge, the only loud one), `instance_name` (the loader matches by substring), `kernel_id` (routes to a PDI via `dpu_kernel_ids`; every AIR compile defaults to `0x901`). Collide the id and the second kernel times out at one shape and returns garbage at `mean_rel_L1` 1.41 **with no error** at the other. The instruction streams are a red herring — byte-identical either way. | [29](29-offload-n-streams.md) |
| Can `offload` share one `hw_context` across its dispatches? | **Yes — under the xclbin ABI, at the mode's existing tiling.** A 2×2 factorial on `q_proj` 1024×768×768 puts the corruption in **exactly one cell of four**: `elf`+`[2,2]` diverges from its own first run by 3.8141e-01 (replicated 2/2), while `elf`+`[1,1]`, `xclbin`+`[2,2]` and `xclbin`+`[1,1]` are bit-identical over 4 runs. `_evict_context` attributes it to "these runtime-tiled GEMM ELFs" as a class; it is the ELF ABI **and** `[2,2]` together, and either knob alone removes it. It also does **not accumulate** — runs 2-4 are identical to each other, so it is a one-time state change, not decay. **This UNBLOCKS the N-streams-one-xclbin work**, which needs the xclbin ABI anyway and therefore needs no retune. | `agents/probes/probe_context_reuse.py` |
| What does `runtime_loop_tiling_sizes = [2,2]` actually break? | **Two separate things, both measured on hardware.** It hangs `mha_out_proj` @4096 (`ERT_CMD_STATE_TIMEOUT` 3/3, `probe_backend_preset_hardware.py`) **and** it leaves context-corrupting residue in a plain projection GEMM under the ELF ABI (`probe_context_reuse.py`). Two independent refutations of [26 §4](26-mode-rebuild-feasibility.md)'s compile-only "inert" finding. A fix addressing one would leave the other live. | `agents/probes/` |

## Provenance

The source is `iron` commit `1e014c1` "Add transformer-layer execution-strategy studies"
(145 files, ~58.6k insertions), validated there by an 888/888-job suite run. This plan was
reviewed by Codex before approval; findings that materially changed it are marked `[Codex]`
in the phase documents.
