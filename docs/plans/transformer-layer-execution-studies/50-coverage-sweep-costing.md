# 50 — The coverage sweep, costed: the estimate was wrong by two orders of magnitude

`[2026-08-12]` Phase G item (c). Repo `/home/cj/mlir-air`, worktree branch
`worktree-agent-a0bab073cb6414184` off `exper/transformer-layer-execution-studies` at `39a08a8b`.

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
| resolution **through the owning builder** | `check_resolution.py`: `qkv_gemm_spec`, `ffn_gemm_specs`, `resolve_gemm_spec(o_proj)`, and the `drain` re-resolution `offload`/`runlist` perform | 36/36 at 512, 36/36 at 768, **35/36 at 1024** |
| whole-layer config assembly | `check_block_config.py`: `builders/block.py::block_config` at 256…4096 | assembles at all three widths |

The second probe is the load-bearing one and is why the first alone would not have been enough:
`qkv_proj` **pins** `method="fused-cast"` (`builders/qkv_proj.py`), `offload`/`runlist` re-resolve the
projection through `drain`, and `resolve_gemm_spec` asserts `M % (tile_m*herd_m) == 0` and
`N % (tile_n*herd_n) == 0` *after* the lookup. A triple can be present and still not resolve for the
method a mode needs — which is exactly the one failure found.

`norm_rows` derives per width and is legal at all three (64 at 512, 64 at 768, 32 at 1024), which
`builders/block.py:228-230` warns is not automatic: "a row count that happened to fit at one width is
a placement failure at the next."

### 1.1 The one real hole

`2048x1024x3072` has **no `drain` row**. `offload` and `runlist` re-resolve qkv through `drain`, so
those two modes fail at seq 2048 for `baseline_1024`. `coarse` and `fused` are unaffected — they use
the pinned `fused-cast`, which is present.

This is recorded as `profiles.KNOWN_REGISTRY_GAPS`, **not** as a skip. A skip is a claim about what a
*mode supports*; this is a missing measurement, so the rung is run and its refusal is the result —
`cases.py`'s own rule that "pre-declaring a failure is how a matrix stops being a measurement".
`test_profiles.py` asserts each declared gap is **still a gap**, so sweeping that one method makes
the list fail rather than leave a stale warning behind.

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
| `baseline_1024` | 1024 | **0** (36/36 triples) | 0 | 0 | 0 | same shared change | **0 further**; 2 of 36 rungs fail on one missing `drain` method — a **~2 min** single-shape sweep to close, or leave as a recorded gap |
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
   re-resolution — where `baseline_1024` is 35/36.
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
| `baseline_1024` | **reachable, not yet walked** | 35/36; `offload`/`runlist` fail at seq 2048 on one missing `drain` method |
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
