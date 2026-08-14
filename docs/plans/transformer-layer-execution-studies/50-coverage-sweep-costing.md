# 50 — The coverage sweep, costed: the estimate was wrong by two orders of magnitude

`[2026-08-12]` Phase G item (c). Repo `/home/cj/mlir-air`, worktree branch
`worktree-agent-a0bab073cb6414184` off `exper/transformer-layer-execution-studies` at `39a08a8b`.

> **`[2026-08-12]` §1.1 AND §5's `baseline_1024` ROW ARE CORRECTED IN §7.** The "one real hole" this
> document records — `2048x1024x3072` has no `drain` row, so `offload`/`runlist` fail at seq 2048 —
> is **half right and its consequence is false**. The registry fact holds; nothing asks for it.
> Neither mode resolves a `3h` shape at all. `baseline_1024` was **36/36, not 35/36**, and it is now
> walked. Read §7 before acting on §1.1, §3's `baseline_1024` row, or §5.

Doc 34 §M5 and §4.1 size "widen the matrix past `baseline_768`" as **unbounded — C4's precedent is
504 + 66 min of gate time alone**, and doc 10's header repeats it as "a Phase-C-sized coverage sweep
first". `study/profiles.py` carried the same claim in its `UNREACHABLE_FAMILIES` reasons, and
`test_profiles.py` re-derived it from `opcheck_specs.py` by `ast` so that it could not silently rot.

**It rotted anyway, because the test re-derived the wrong file.** The claim was measured today and
it is false for two of the five families. This document is the measurement.

---

## 0. The one-paragraph answer

The coverage the two encoder-width families were said to need **has been in the tree since
2026-08-07**. `kernel_registry/details/GEMM_bf16_in_bf16_out.json` holds **36 of 36** projection
triples for each of hidden 512, 768 and 1024. The blocker was never coverage: it was that
`study/run_mode.py::_shape_for` overrode `seq_len` and not the width, so no caller could ask for a
family it already had the kernels for. Making width a parameter is **~40 lines**, and
`tinybert_512` walked end to end on hardware in **301 s** — against an estimate of **570 minutes**.
The three decoder families are a different and unchanged story: `decoder_gpt2` is a distinct layer
graph, not a flag, and its cost is a D2-class integration per mode. **Nothing here was a sweep.**

---

## 1. What was measured, and how

Three probes, all read-only, no device, no writes into the repo.

| probe | script | result |
|---|---|---|
| registry triples per family | `check_registry.py` over the JSON | **36/36** for each of hidden 512, 768, 1024 |
| resolution **through the owning builder** | `check_resolution.py`: `qkv_gemm_spec`, `ffn_gemm_specs`, `resolve_gemm_spec(o_proj)`, and the `drain` re-resolution `offload`/`runlist` perform | 36/36 at 512, 36/36 at 768, ~~**35/36 at 1024**~~ **36/36 at 1024 — §7**; the probe applied the `drain` re-resolution to the **qkv** shape, which `offload` never resolves |
| whole-layer config assembly | `check_block_config.py`: `builders/block.py::block_config` at 256…4096 | assembles at all three widths |

The second probe is the load-bearing one and is why the first alone would not have been enough:
`qkv_proj` **pins** `method="fused-cast"` (`builders/qkv_proj.py`), `offload`/`runlist` re-resolve the
projection through `drain`, and `resolve_gemm_spec` asserts `M % (tile_m*herd_m) == 0` and
`N % (tile_n*herd_n) == 0` *after* the lookup. A triple can be present and still not resolve for the
method a mode needs — which is exactly the one failure found.

`norm_rows` derives per width and is legal at all three (64 at 512, 64 at 768, 32 at 1024), which
`builders/block.py:228-230` warns is not automatic: "a row count that happened to fit at one width is
a placement failure at the next."

### 1.1 The one real hole ~~— **`[2026-08-12]` IT IS NOT A HOLE. See §7.**~~

> **CORRECTED IN §7.** The first sentence below is true and every sentence after it is false.
> `offload` and `runlist` never resolve a `3h` shape, so they never ask for this row.

`2048x1024x3072` has **no `drain` row**. ~~`offload` and `runlist` re-resolve qkv through `drain`, so
those two modes fail at seq 2048 for `baseline_1024`.~~ `coarse` and `fused` are unaffected — they use
the pinned `fused-cast`, which is present.

~~This is recorded as `profiles.KNOWN_REGISTRY_GAPS`, **not** as a skip. A skip is a claim about what a
*mode supports*; this is a missing measurement, so the rung is run and its refusal is the result —
`cases.py`'s own rule that "pre-declaring a failure is how a matrix stops being a measurement".
`test_profiles.py` asserts each declared gap is **still a gap**, so sweeping that one method makes
the list fail rather than leave a stale warning behind.~~

> **`[2026-08-12]` The instinct was right and the entry was wrong.** Recording it as a *failure to be
> run* rather than a skip is what let the walk overturn it — had it been a skip, `run_ladder` would
> have written `run_status="skipped"`, started no child process, and the manifest would have reported
> a complete walk that never attempted the rung. **The rung ran, and it passed.** But "assert each
> declared gap is still a gap" only ever checked that the method was still *missing*, which is true
> of every method nobody asks for, so nothing could have retired it. `KNOWN_REGISTRY_GAPS` is now
> empty and the test additionally requires a gap to name a `(triple, method)` some consumer **pins**.
> See §7.6.

---

## 2. Why the estimate was wrong

Not arithmetic. The estimate was a correct reading of the wrong file, protected by a test that
re-derived it from the same wrong file.

`test_profiles.py::test_reachable_family_is_the_one_the_runner_can_build` parsed `opcheck_specs.py`
and asserted every whole-layer SPECS row was `emb_dim 768`. **That assertion was true and is still
true.** The inference drawn from it — "so hidden 512 and 1024 are out of reach" — was false, because
a SPECS row is a (builder, tolerance) pair with one shape written beside it, and the ladder has
overridden that shape's `seq_len` at eight lengths the row does not name since J3. The widths that
decide reachability are the *registry's*.

Two commits on 2026-08-07 took the registry 69 → 103 → 136 rows across the `baseline_512` and
`baseline_1024` families. Doc 34 was written on 2026-08-12 and sized the work against C4, which
predates them. Nobody read the file in between.

**The generalizable lesson, and it is about this project's own strongest habit.** Re-deriving a claim
from source is the discipline that has repeatedly saved this study, and it failed here in the one way
it can: *a re-derivation is only as good as its choice of source.* The check was mechanically
perfect. What it never asked was whether `opcheck_specs.py` was the file that decided the answer.
`test_profiles.py` now reads the registry JSON, asserts the converse (no unreachable family may cite
a coverage reason), and asserts each declared method gap is still open — three directions, so it
cannot pass by declaring nothing reachable.

---

## 3. The per-family cost table

Costs are the concrete work items. "Device time" excludes the walk itself, which is the measurement
rather than the unblocking.

| family | width | new registry rows | sweep minutes | new kernels | builder change | code | **total to unblock** |
|---|---|---|---|---|---|---|---|
| **`tinybert_512`** | 512 | **0** (36/36, all methods) | **0** | 0 | 0 | width parameter, shared | **~40 lines, shared** ✅ **done and walked** |
| `baseline_1024` | 1024 | **0** (36/36 triples) | 0 | 0 | **`runlist`'s norm band — §7.5** | same shared change | ~~**0 further**; 2 of 36 rungs fail on one missing `drain` method — a **~2 min** single-shape sweep to close, or leave as a recorded gap~~ **WITHDRAWN (§7): no rung fails on the registry and no sweep is needed.** Walked at devq 307: `coarse`, `offload`, `fused` green; **`runlist` fails at all 9 ladder points** on `norm_rows`=32 against a layer-norm block of 64 — a builder change, not a row |
| `gpt2_small_768` | 768 | 0 | 0 | 0 | **decoder layer graph** | pre-norm order, plain residual add, masked add in `runlist`/`offload` | **D2-class, ~156 min/mode** of integration; no sweep |
| `gpt2_512` | 512 | 0 | 0 | 0 | same graph | same | as above; width already free |
| `gpt2_medium_1024` | 1024 | 0 | 0 | 0 | same graph | same | as above, plus `baseline_1024`'s `drain` gap |

**Against doc 34's estimate of "unbounded — 504 + 66 min" for the width families: measured at 301 s
of device time and one shared parameter.**

The three decoders are unchanged from doc 34's reading and are *not* cheap. `decoder_gpt2` needs the
norm **before** attention, a plain `elementwise_add` residual the encoder block never dispatches
(`builders/addnorm.py`'s two-output entry point exists for it and is unbuilt here), and a causal
masked add between the score GEMM and softmax that `runlist` and `offload` have **no masking step for
at all**. The causal *kernel* exists and is hardware-validated — `opcheck_specs.py` carries causal
`mha_out_proj` rows at 8 heads/emb 512 and 16 heads/emb 1024 — so this is layer wiring, not a kernel
gap. `gpt2_small_768` is the cheapest of the three because its width is already the default: the
graph is the whole cost.

### 3.1 Why `tinybert_512` was chosen as the existence proof

Cheapest by measurement, not by guess:

1. **36/36 resolve through the owning builders**, including the pinned `fused-cast` and the `drain`
   re-resolution — ~~where `baseline_1024` is 35/36.~~ **`baseline_1024` is also 36/36 (§7);** this
   reason for preferring 512 did not hold, though reasons 2 and 3 did.
2. **`norm_rows` is 64**, the same band D1 and D2 validated at 768. `baseline_1024`'s 32 is a band no
   whole-layer run has exercised, and `builders/block.py:228-230` warns specifically about that.
3. **`fused`'s packing bound is widest at 512** — (256, 2048) against 768's and 1024's (256, 1024) —
   so more of the ladder is attemptable.

---

## 4. The existence proof — devq job **304**, `measure`, Turbo, cold

`agents/.state/devq/jobs/job-000304.log`, `job-000304.meta` `exit=0`, **301 s** for leg 1.

```
[run-mode] coarse  @ 1024x512_encoder_bert: passed   avg  67.495 ms  subs  4  herd  49  sync 107
[run-mode] offload @ 1024x512_encoder_bert: passed   avg 116.283 ms  subs 22  herd  66  sync  66
[run-mode] runlist @ 1024x512_encoder_bert: passed   avg  99.845 ms  subs 13  herd 171  sync 138
[run-mode] fused   @ 1024x512_encoder_bert: passed   avg  67.376 ms  subs  1  herd  23  sync  13
[manifest] complete: True   (4 CSVs, rows 1/1  passed 1/1  skipped 0/0 each)
[run-profile] smoke: passed 4  (301s wall)
```

All ten per-boundary stages clean for every mode; attention ran as 8 heads, correctly following the
family rather than the 768 row's 12.

**No latency is quoted as a result.** README trap 1: a single walk once published a crossover a second
walk refuted. Those four numbers are evidence that four modes *built and validated at a width the
runner could not reach this morning*, and nothing more. A ranking needs two walks into two roots and
`compare_roots`.

### 4.1 The inherited-tolerance question, answered with the artifact

`_shape_for` overriding the width means a run inherits an `atol` measured at another shape — the same
liberty the ladder takes on `seq_len`, but worth checking the first time. Measured at emb 512, final
boundary, four modes:

| | `mean_rel_L1` | `atol_required` | ceiling | margin |
|---|---|---|---|---|
| best | 1.253e-2 | 5.218e-2 | 1.0e-1 | 1.92× |
| worst | 1.665e-2 | 6.038e-2 | 1.0e-1 | 1.66× |

Comfortably inside, and comparable to the 768 row's own recorded 1.35–1.72×. The `1e-1` value is a
*defect ceiling*, not a fitted tolerance — exceeding it is a defect report, never a widened number.

---

## 5. What is now reachable, and what is refused

| family | verdict | why |
|---|---|---|
| `tinybert_512` | **reachable, walked** | devq 304 |
| `baseline_768` | reachable, walked | the standing default |
| `baseline_1024` | **reachable, WALKED** (§7) | ~~35/36; `offload`/`runlist` fail at seq 2048 on one missing `drain` method~~ **36/36** — the missing `drain` row is demanded by nothing; devq **307** |
| `gpt2_512` | refused by name | `decoder_gpt2` layer graph |
| `gpt2_small_768` | refused by name | same; cheapest decoder — width already free |
| `gpt2_medium_1024` | refused by name | same |

**Refused, not omitted, and the distinction is the whole point.** Overriding the width alone and
stamping the row `decoder_gpt2` would produce a perfectly valid-looking *bidirectional* measurement
under a *causal* name. Nothing downstream could ever detect it — not the stage comparisons, not the
manifest, not `compare_roots`. So `run_mode.UNBUILDABLE_VARIANTS` refuses before anything is prepared,
`Profile.__post_init__` refuses at construction rather than after a cold walk fails 36 times, and
`test_profiles.py` asserts every unreachable family's variant is refused *and* that every reachable
one is not — so the guard cannot pass by refusing everything.

---

## 6. What `full` means now

`full` is still the nine-point ladder over **one** family and still does not claim otherwise. What
changed is that the family is now a **parameter**: `--family tinybert_512` retargets the plan and
**every expected count is re-derived**, including `fused`'s applicability bound, which moves with the
width because the cap is on `rows*cols`. At `ladder`, `fused.csv` expects 2 passed / 2 skipped at 768
and 3 passed / 1 skipped at 512 — with **no number edited anywhere**, which is what the profile
table's discipline was for.

Walking the declared 6×9 matrix is now **three `full` walks plus a decoder integration**, not a
coverage sweep. Sizing the three walks from job 304's 301 s for 4 rungs and doc 25's ~2.8 min/rung
cold averaged over the ladder: ~1.7 h each cold, ~5 min warm, **[inference — extrapolated; no `full`
walk has ever run]**. That is a scheduling decision, not a phase.
---

## 7. `[2026-08-12]` The `baseline_1024` gap was not a gap — and the wall that is really there

> **`[2026-08-14]` THE WIDTH WALL IS CLOSED, and it was narrower than this section concluded.**
> §7.0 ends with "a *width* wall, it is nothing to do with the registry", which is right, and §7's
> decision to leave it unfixed rested on *"every plausible repair changes what a `runlist` row
> means"*. That is true of the repairs this section had in view — moving `norm_rows`, or the L1
> margin, or `herd_x` — and false of the one that was available.
>
> `build_layer_norm_module`'s **`rows_per_call` was never derived**. It defaulted to 8, and with
> `herd_x = 8` that silently requires `64 | rows`. `block.norm_rows` is derived per width and says so
> in its own docstring — "a row count that happened to fit at one width is a placement failure at the
> next" — and `rows_per_call` was the other half of the same sum, carried over as a constant. At
> emb 1024 `norm_rows` derives **32**, each of the 8 cores owns **4** rows, and **4 was legal all
> along**.
>
> `derive_rows_per_call` is bounded **above** by the historical default, which is what makes it safe
> to land beside gated designs: wherever 8 was legal it still returns 8, and the **emitted IR is
> byte-identical** — asserted at five shapes, with a control that `rows_per_call` 4 vs 8 really does
> change the module, or the equality would hold whatever the derivation returned. The divisibility
> rule is untouched: an **explicit** `rows_per_call = 8` at 32 rows still raises, which is what
> separates deriving the parameter from deleting the constraint.
>
> `run_layer_norm_rows_tests.lit`, 9 host-only clauses. **Not yet walked on device** — a `runlist`
> row at emb 1024 needs `run_mode.py`, which refuses off Turbo, and the machine is at `Default`.
> The builder-side claim is proven; the ladder rung is owed.

`[2026-08-12]` Repo `/home/cj/mlir-air`, worktree branch `worktree-agent-a8cbef1a620b6f8f9` off
`exper/transformer-layer-execution-studies` at `35e9c382`. NPU power mode **Turbo**, verified.

This section closes the item §1.1 and §5 left open, and it closes it in neither of the two ways the
costing anticipated. The **~2 minute sweep was never needed**; the hole it would have filled is
demanded by nothing. And walking the family found a **different and real** wall that no registry row
can fix.

### 7.0 The one-paragraph answer

`2048x1024x3072` really has no `drain` row — that half of §1.1 is a fact and it still holds. The
consequence drawn from it was false. **`offload` and `runlist` never resolve a `3h` shape at all**:
both chains are `proj` `(seq, h, h)`, `up` `(seq, h, 4h)`, `down` `(seq, 4h, h)`. The only consumer of
`(seq, h, 3h)` is `qkv_proj`, which **pins `fused-cast`** — present. So nothing asks for `drain` at
that triple and `baseline_1024` was **36/36, not 35/36**. The walk confirms it: devq **307**, `ladder`,
Turbo — **`offload` passed at all four points including the seq 2048 the record named**. What the walk
also found is that **`runlist` cannot build at emb 1024 at any sequence length** — `norm_rows` derives
**32** there and `build_layer_norm_module` requires a multiple of **64**. That is a *width* wall, it
is nothing to do with the registry, and it is the risk §3.1 point 2 named and nobody had measured.

### 7.1 How the wrong consequence got recorded

§1's second probe is described as "the `drain` re-resolution `offload`/`runlist` perform". That
re-resolution is real: `offload._chain_spec` re-resolves a `fused-cast` winner to `drain`, because a
two-launch module faults the firmware's in-stream `load_pdi` (doc 29). The probe applied it to **all
four projection roles**. `offload` applies it to **three**, and the qkv `3h` shape is not one of them
— so the probe modelled a pin against a shape the pinning module never resolves, and the one shape
where that mattered is the one the registry happens to lack.

**This is §2's lesson one level in.** G1 corrected M5 by re-deriving reachability from the **registry**
instead of from `opcheck_specs.py` — the right fix, and §2 ends with *"a re-derivation is only as good
as its choice of source"*. Then G1's own resolution probe chose the wrong source for the second half
of the same question: it took `offload`'s method pin from the mode's prose rather than from the mode's
**chain**. The correction and the defect have the same shape. Being careful which *file* you read is
not the same as being careful which *call site* you model.

Note also that the **shipped** check said so all along: `registry_sweep.py --family baseline_1024
--verify-resolution` reports `PASS: all 36 baseline_1024 shapes resolve`. It was not consulted; a
bespoke probe was written beside it, and the extra re-resolution went into the bespoke one.

### 7.2 What was measured — host-only probes

No writes into the registry: `git diff` over `programming_examples/kernel_registry/` is empty after
every probe below, including the two that write to copies.

| probe | what it did | result |
|---|---|---|
| `registry_sweep.py --family baseline_1024 --verify-resolution` | the **shipped** resolution check, through the owning builders | **PASS: all 36 resolve** |
| the same at `baseline_512` / `baseline_768` (controls) | | PASS 36/36 each |
| `offload._chain_spec` replayed over the whole 9-point ladder | its real re-resolution, `proj`/`up`/`down` | **0 failures**; all 27 resolve to `drain` |
| `runlist`'s three `resolve_gemm_spec` calls, whole ladder | | **0 failures** |
| `qkv_gemm_spec(seq, 1024)`, whole ladder | the only `3h` consumer | **9/9, all `fused-cast`, all from the registry** |
| `resolve_gemm_spec(2048, 1024, 3072, method="drain")` | the missing row itself | raises — **and nothing calls this** |

### 7.3 The row could not have been written even if it were needed

This changes the *cost* answer §3 gives for any future missing method on this family, so it is worth
recording. `2048x1024x3072`'s entry is `used_by: "Qwen3-0.6B Gate/Up proj"` — **a shipped deployment
owns it**, not this study's sweep. Three arms, all against a copy:

| arm | setup | outcome |
|---|---|---|
| **A** | the entry as it stands, plus a passing synthetic `drain` row | `add_missing_methods` reports `not_owned=[(2048, 1024, 3072)]`, **file bytes unchanged** |
| **B** | identical, `used_by` rewritten to the sweep's own string | still refused — `ShapeAlreadyRegistered`: the entry's `direct` row carries `mean_rel_L1: 0.0113` (3 s.f.) and `registry_writer._round_rel` emits **2 s.f.**, so the writer cannot re-render the entry it is required to preserve byte-for-byte |
| **C** | control: `64x1024x3072`, which the sweep **does** own — one method removed from the copy, then added back | **accepted**: `added=[((64, 1024, 3072), ['drain'])]`, file bytes move |

C is what makes A and B mean anything: the mechanism works, and it refuses this entry for two
independent reasons. **The append-only rule is behaving exactly as designed** — `best` is derived per
tier, so admitting a faster `drain` would silently re-point `best.high` away from the `fused-cast` row
Qwen3-0.6B resolves against. A sweep must not be the thing that changes a shipped deployment's kernel.

So the honest cost line for a *genuinely* needed method on a deployment-owned row is not "~2 minutes
of sweep". It is: **the sanctioned writer will not take it at all**, and the options are a builder
change or a recorded gap — never a hand edit, which is the drift the registry exists to prevent.

**No sweep was run**, and that is the finding rather than an omission: device time spent measuring a
row nothing resolves and the writer would refuse twice over buys nothing. §3's `baseline_1024` cell —
"a ~2 min single-shape sweep to close" — is withdrawn.

### 7.4 The walk — devq **307**, `measure`, Turbo, cold

`ladder` at `baseline_1024`, chosen over `smoke` deliberately: `smoke` is seq 1024 only and the record
named **seq 2048**, so only a profile that walks 2048 could settle it.
`agents/.state/devq/jobs/job-000307.log`.

```
[ladder] coarse    seq   512  avg    71.266 ms  subs   4  herd   49  sync  107  (103s)
[ladder] coarse    seq  1024  avg   151.113 ms  subs   4  herd   82  sync  204  (102s)
[ladder] coarse    seq  2048  avg   326.058 ms  subs   4  herd  147  sync  397  (110s)
[ladder] coarse    seq  4096  avg   825.318 ms  subs   4  herd  276  sync  782  (137s)
[ladder] offload   seq   512  avg   140.153 ms  subs  38  herd  114  sync  114  ( 29s)
[ladder] offload   seq  1024  avg   215.012 ms  subs  38  herd  114  sync  114  ( 31s)
[ladder] offload   seq  2048  avg   773.544 ms  subs  38  herd  114  sync  114  ( 35s)   <-- the rung the record said would fail
[ladder] offload   seq  4096  avg  1598.426 ms  subs  38  herd  114  sync  114  ( 47s)
[ladder] runlist   seq   512  FAILED  ValueError: rows (32) must be divisible by herd_x*rows_per_call (64)
[ladder] runlist   seq  1024  FAILED  (same)
[ladder] runlist   seq  2048  FAILED  (same)
[ladder] runlist   seq  4096  FAILED  (same)
[ladder] fused     seq   512  avg    66.769 ms  subs   1  herd   23  sync   13  (125s)
[ladder] fused     seq  1024  avg   146.538 ms  subs   1  herd   24  sync   14  (144s)
[ladder] fused     seq  2048  SKIPPED  bounded to 256..1024 at emb 1024
[ladder] fused     seq  4096  SKIPPED  bounded to 256..1024 at emb 1024
```

`coarse` **4/4**, `offload` **4/4**, `fused` **2 passed + 2 skipped**, `runlist` **0/4**. Whole walk:
**871.6 s**, `rungs_by_status {passed 10, failed 4, skipped 2}`, `rungs_by_source {measured 14,
reused 0, skipped 2}`, `devq_job_id 307`, `tree_dirt_after_run` 5 entries and **0 untracked** (the
author's own edits — the untracked half is the one that would show a leak). `aircc` resolved to
`/home/cj/mlir-air/install-xrt/bin/aircc`, mtime **2026-08-12 14:03:46**, against a walk that ran
15:44–15:58 — the current install, not a stale one.

**The manifest reports `complete: False` with `row_counts_checked: true`, and that is the correct
answer**, with both reasons naming the same mode:

```
runlist.csv: 4 row(s), none with run_status=passed. First failure: ValueError: rows (32) ...
runlist.csv: expected 4 passed row(s), found 0 (passed 0, failed 4, skipped 0 of 4)
```

Per file: `coarse` 4/4 passed, `offload` 4/4 passed, `fused` 2 passed / 2 skipped against an expected
2 and 2, `runlist` 0 passed / 4 failed against an expected 4. Three of four CSVs meet their derived
expectation exactly. The completeness clause counts **rows, not files** — the defect G0 closed — and
that is what makes this walk report the truth: `smoke` would have been four green rungs at seq 1024,
`complete: True`, touching neither the seq 2048 the record named nor the wall below.

**No latency is quoted as a result.** README trap 1 — these numbers are evidence that three modes
built and validated at a width never walked before, and nothing more. A ranking needs two walks into
two roots and `compare_roots`.

### 7.5 The real `baseline_1024` wall: `norm_rows` 32 against a layer-norm block of 64

`runlist` builds a standalone layer-norm module at `pattern/runlist/runlist.py:445`,
`build_layer_norm_module(rows, emb_dim, bfloat16)`, with `rows = builders.block.norm_rows(seq, emb)`.
That builder's defaults are `herd_x=8, rows_per_call=8` and it requires `rows % 64 == 0`
(`builders/layer_norm.py:89`). `norm_rows` maximises a *different* constraint — the largest multiple
of `NORM_HERD_X = 8` that divides `seq_len` and fits `addnorm`'s L1 cap at `NORM_ROW_MARGIN = 0.75`:

| emb | addnorm L1 cap | at 0.75 margin | `norm_rows` (every ladder point) | `rows % 64 == 0` |
|---|---|---|---|---|
| 512 | 160 | 120 | **64** | yes — **9/9** |
| 768 | 104 | 78 | **64** | yes — **9/9** |
| **1024** | **80** | **60** | **32** | **no — 0/9** |

At emb 1024 the cap falls below 64, so `norm_rows` takes the next legal band down, 32 — and every
`runlist` rung at that width fails at build, at **every** sequence length. It is a **width** wall, not
a ladder point, and it fails in ~1–3 s without reaching aircc.

`coarse` is unaffected at the same `norm_rows = 32` — its own log line is `addnorm pre-add 32x1024
x128 dispatches (L1 cap 80)` — because the `addnorm` path it uses needs only a multiple of
`NORM_HERD_X = 8`. `runlist` announces the same band as `norm chains banded at 32 rows x128
(builders.block.norm_rows — coarse's schedule)` and then fails, which is the discriminator stated in
the log itself: **two consumers of one derived row count, with constraints that differ by 8x**, and
the one that inherits "coarse's schedule" cannot actually build it.

This is exactly what `builders/block.py:227-229` warns about — *"a row count that happened to fit at
one width is a placement failure at the next"* — and what §3.1 point 2 flagged as the reason
`tinybert_512` was preferred: "`baseline_1024`'s 32 is a band no whole-layer run has exercised". It has
now been exercised. **Not fixed here**, and deliberately not: the plausible repairs (lower
`rows_per_call` for this call site, raise the margin, band the norm differently) each change what a
`runlist` row *means*, and doing that inside the walk that discovered it would make the fix and its
evidence the same run.

### 7.6 The test that could not have caught this, and the one that now can

`KNOWN_REGISTRY_GAPS` is **empty** and the clause guarding it is rewritten.

The old clause asserted, per declared gap, that the method was **still missing**. That was true — and
it is true of *every method nobody asks for*, so it would have passed for as long as the entry
existed. Fed the entry verbatim it still passes today (measured, not assumed). The suite could not
have woken up.

`test_a_declared_registry_gap_must_be_one_some_mode_actually_demands` keeps that clause and adds the
missing half: a declared gap must also name a `(triple, method)` some consumer **pins**. The pins live
in a new `profiles.PINNED_PROJECTION_METHODS`, and
`test_the_pinned_methods_are_re_derived_from_the_modules_that_pin_them` keeps that table honest by
`ast`-reading the two files that carry them — `qkv_proj.SCRATCH_METHOD` and `offload`'s `_chain_spec`
re-resolution — because a table copied out of a source file agrees with the day it was copied. The
pre-existing reachability test also stopped checking only that the **triple** is present and now
checks that the **pinned method** is, which is the check that would have caught this class at source.

**Negative controls**, each demonstrated failing on the input that drives it:

| control | input | outcome |
|---|---|---|
| **1** | the pre-fix `KNOWN_REGISTRY_GAPS` entry, verbatim | old clause **passes** (`still a gap -> True`); new test **REFUSES** — *"the consumer of that role pins `'fused-cast'` … Nothing asks for `'drain'` there"* |
| 2 | registry with `drain` deleted from `1024x1024x1024` | reachability test **REFUSES**, naming `offload`'s `_chain_spec` as the pin |
| 3 | `PINNED_PROJECTION_METHODS[(1,3)]` tampered to `drain` | pin test **REFUSES** — *"qkv_proj pins 'fused-cast' and the table says 'drain'"* |
| 4 | the real tree | all three pass |

Control 1 is the one the item turns on: **the same input passes the old test and fails the new one.**

Host suite **517 → 519 in 23 modules**, measured on this tree, and the literal verified with
FileCheck in both directions: 519/519 accepted, the stale **517 refused**, a shrunk 518/518 refused,
a 22-module run refused.

### 7.7 What this changes elsewhere

- §1.1, §1's probe table, §3.1 point 1 and §5's `baseline_1024` row are annotated in place above;
  doc 34 §M5's "35/36 at 1024" bullet and its `KNOWN_REGISTRY_GAPS` paragraph likewise.
- **`baseline_1024` is walked but `runlist` is down at that width.** The family is reachable for
  `coarse`, `offload` and `fused`; a four-mode comparison at emb 1024 is blocked until §7.5 is fixed.
- **`full` at `baseline_1024` is not unblocked by this** — it attempts 64, 128, 8192 and 16384, which
  no mode has been measured at for any family, exactly as §6 says.
- **Still open, unchanged**: the decoder layer graph, and **two walks into two roots**, which nothing
  here has done. This walk is one.
