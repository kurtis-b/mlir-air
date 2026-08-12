# 47 — The balance instrument: what was built, what is measured, what is modelled

`[2026-08-12]` Queue item **25**, specified by [44 §The instrument](44-mapping-frameworks-synthesis.md)
and by the two defects [38 §3.3](38-iron-encoder-pipeline-reference.md) found in iron's hand-rolled
version. Two new modules, two new host-test modules, 72 new tests, **no new constants**.

> **Read the "measured vs modelled" table in §3 before quoting any number from here.** The whole
> reason this document exists is that a modelled constant presented as measured is the failure the
> instrument is built to prevent, and there is one number in this tree that is very tempting to
> import and must not be — see §3.3.

## TL;DR

**Built, host-only, no simulator and no hardware run**: a `[step × port]` demand matrix per column
read off a routed AIE artifact; a static back-solve of the bandwidth a stall-free execution would
have required; overflow priced as a **slope** with demand printed beside budget; latency as a `max`
over per-resource isolated times whose argmax names the resource; and a
`(component, action, arguments) → cost` table holding **measured nanoseconds and counted bytes**.

**The first thing it found**, on the shipped `addnorm` artifact and reproducing doc 23's measured
row exactly: **column 0 MM2S demand 3 against budget 2**, three logical channels on one physical
channel, priced at `slowdown 0.667` / runtime `×1.500` rather than rejected.

**The second thing it found is a blind spot in how this tree counts that demand.** Counting
per-column ingress as shim→core `aie.flow` ops reads **0** on that design — because AIR's reaction
to exceeding the budget is to emit `aie.packet_flow` instead. A demand counter that reads zero
exactly when the design is over budget is a check that cannot fail. The instrument counts DMA
*allocations*, which exist in both forms.

**Seeded from artifacts that already existed**: 1,213 ERT entries — **1,208 measured device
latencies** out of the 1,508 GEMM sweep results, **5 counted byte costs** from two routed designs,
and **zero modelled numbers**.

**A finding that bears directly on doc 44's framing.** 259 of those sweep results are repeat
measurements of an *identical* priced action. Their `(max−min)/min` spread has **median 1.6% and
worst 42.2%**, and **53 of the 259 exceed 5%**. Doc 44's headline — "480,000 mappings within 5% of
peak performance vary 19× in energy" — assumes a 5% band means something. On this tree's own
measurements, a 5% band is inside the repeat-measurement noise for **20% of the actions that were
measured twice**. That does not weaken doc 44's argument (its point is that ties on one metric hide
variation on another); it sharpens the practical consequence: **a mapping search here must compare
distributions, and the ERT was extended to carry one rather than a scalar.**

## 1. What was built

`programming_examples/transformer_layer/study/balance_ert.py` and `.../study/balance.py`, with
`test_balance_ert.py` (29 tests) and `test_balance.py` (43 tests).

| # | part | from | where |
|---|---|---|---|
| 1 | `[step × port]` demand matrix per column | SCALE-Sim ([42](42-research-scalesim.md)) | `balance.demand_matrix` |
| 2 | static back-solve of required bandwidth | SCALE-Sim `InterfaceBandwidth: CALC` | `balance.back_solve` |
| 3 | overflow as a **slope**, demand beside budget | Timeloop ([41](41-research-timeloop.md)) + MAESTRO ([43](43-research-maestro.md)) | `balance.balance_ports`, `PortBalance.warning` |
| 4 | latency = `max` over isolated times, argmax **names** it | Timeloop's bottleneck model | `balance.bottleneck` |
| 5 | `(component, action, **arguments**) → cost`, measured | Accelergy's ERT ([40](40-research-accelergy.md)) | `balance_ert.Ert` |
| + | iron's full-vs-stage metric, **both defects fixed** | [38 §3.3](38-iron-encoder-pipeline-reference.md) | `balance.stage_gap` |

Run it:

```
python3 study/balance.py <air_project-dir> [--duration-ns N] [--json out.json]
python3 study/balance_ert.py --seed-gemm-sweep sweep/results/baseline_768 \
    --seed-routed-design <air_project-dir> --out results/ert.json
```

Neither touches the NPU, and neither needs the study's toolchain env — both read files.

### 1.1 The demand matrix, and the two demand numbers

Input is `air_project/aie.air.mlir`, the same artifact `resource_usage.py` reads. Every
`air.channel.put`/`get` carrying a `metadataArray` is shim-facing; its `metadataArray` entry
resolves through `aie.shim_dma_allocation` to a `(column, direction, physical channel)`, and its
operand carries the BD's own `[offsets][sizes][strides]`.

Two numbers come out, and they answer different questions:

- **`static_demand`** — distinct logical channels allocated to a column and direction over the whole
  design. This is what doc 23's rule is about ("the budget is per COLUMN across the whole segment")
  and it is what AIR actually allocated. **The budget check uses this one.**
- **`peak_concurrent_demand`** — the max over the step axis. Always ≤ the static one; it says
  *where* the pressure is.

Reporting only the concurrent number would let a design look compliant because its streams never
overlap, when AIR has already committed both to one physical channel regardless.

**`step` is not a hardware cycle.** SCALE-Sim's rows are simulated cycles; ours are ASAP levels over
the launch-level async dependence graph. That is strictly less information and it is all a static
read can support — and doc 44's whole argument for this design is that *not* needing a simulator is
what makes a search affordable.

### 1.2 Why the budget is a slope and not a predicate

Doc 44 corrects an earlier proposal of this project's — make the per-column MM2S budget a legality
predicate — and the correction is why `balance.py` has no `is_legal`. Exceeding the budget does not
break correctness; AIR packet-multiplexes and the design runs slower. Filtering would have hidden
the exact failure mode we are trying to see.

So `balance_ports` returns a record for **every** port including the compliant ones, and the
over-budget ones get `slowdown = min(1, budget/demand)` plus a warning that prints both numbers.
`charged_ns` then charges the shortfall into runtime — MAESTRO charges *both* channels, not either.
It takes the **worst** column, not the product: the columns are parallel resources.

`test_over_budget_is_priced_not_filtered` pins this, and the injected defect "the per-column budget
becomes a legality predicate" is rejected by 4 tests (§4).

### 1.3 iron's two defects, made structurally impossible

`stage_gap(full, stages)` is doc 38 §3.4's minimum viable port of iron's balance metric. It refuses
the inputs that carry the defects rather than warning about them:

- **Defect 1, the prefix.** iron's `debug=7` "isolated stage" `addnorm1` kept `mha_debug=0`, so the
  whole MHA front-end computed inside it — the reported "max isolated stage" already contained
  another entry in the same comparison. Every entry must declare `contains`, and one strictly
  containing another raises `PrefixComparison`.
- **Defect 2, the elided weights.** `addnorm1` set `need_bup_weights` and `need_bdown_weights` both
  False, so **9,437,184 bytes** of B_Up/B_Down (768×3072 + 3072×768 at bf16) that the `full` build
  reads were never fetched. Every entry must declare `l3_bytes` and every stage's must equal
  `full`'s, which turns doc 38 §3.4 step 3's recommendation into a precondition.
- **A third, structural.** The result is a dataclass with `to_json`, so the gap cannot be computed
  without something to write. iron's `--output-json` defaulted to `None`, and that is why none of
  its numbers has a file behind it.

The defect-1 guard is **vacuous on an empty `contains`**, and that is stated rather than hidden:
`bottleneck` reports `containment_checked`, and `stage_gap` — where the guard is the point —
*requires* `contains` on every entry.

### 1.4 The ERT, and why the arguments are the module

Doc 44 specifies this in one sentence with the defect attached: *"a `dma_transfer` is a function of
`(n_words, n_dims, stride)`, not a scalar — given our BD-stride walls, a counter reporting 'number
of DMA transfers' has already destroyed the information we need."*

`REQUIRED_ARGUMENTS` names, per `(component, action)`, the arguments an entry **must** carry, and
`insert` raises without them. `lookup` is **exact**: two transfers agreeing on `n_words` and
differing in `strides` are different objects to the BD allocator, and a nearest-match fallback would
erase precisely the difference doc 23's retile finding is about. A miss names the near neighbours so
the difference is visible.

`Cost` carries a **four-valued source per number**, not a boolean per entry:

| source | meaning |
|---|---|
| `measured` | a stopwatch on hardware produced it; `provenance` is the artifact and `condition` the NPU power mode |
| `counted` | read exactly off a compiled artifact (bytes from `sizes` × element width); reproducible without the device |
| `modelled` | computed from an assumption, which `provenance` must state |
| `absent` | nothing known; the value **must** be `None` |

A value is present exactly when its source is not `absent` — enforced in `__post_init__`, so an
unpriced action can never read as a free one.

## 2. What it found

### 2.1 The `addnorm` artifact reproduces doc 23's measured row, and prices it

`python3 study/balance.py <repo>/air_project` against the shipped `addnorm_seg` design
(`/home/cj/mlir-air/air_project/aie.air.mlir`):

| column | dir | demand | budget | mux depth | slowdown | charged | bytes |
|---|---|---|---|---|---|---|---|
| 0 | MM2S | **3** | 2 | **3** | 0.667 | **×1.500** | 26,112 |
| 0 | S2MM | 1 | 2 | 1 | 1.000 | ×1.000 | 12,288 |
| 1–7 | MM2S | 2 | 2 | 2 | 1.000 | ×1.000 | 24,576 each |
| 1–7 | S2MM | 1 | 2 | 1 | 1.000 | ×1.000 | 12,288 each |

That is doc 23's table — *"`addnorm` (x, residual, weight) — 3 L3→L1 streams, 3 packet-typed"* —
read off the artifact rather than off `air-dma-to-channel`, with the overflow **priced** instead of
flagged. Column 0 carries the weight broadcast plus x plus residual; the other seven carry x plus
residual.

The byte totals are exact and check against the operator's own tensors: 26,112 + 7×24,576 +
8×12,288 = **296,448 B** = x (98,304) + residual (98,304) + weight (1,536) + output (98,304).

Artifact: `.../scratchpad/item25-private/balance.txt` and `balance.json`;
reproduce with `python3 study/balance.py /home/cj/mlir-air/air_project`.

### 2.2 The blind spot: counting `aie.flow` reads zero exactly when the design is over budget

Measured on the same artifact:

| counted object | count | shim→core |
|---|---|---|
| `aie.flow` | 8 | **0** (all eight are core→shim output drains) |
| `aie.packet_flow` | 17 | 3 sourced at column 0, 2 at each of columns 1–7 |

So a per-column ingress count defined over `aie.flow` reads **0** on the design that is 1.5× over
budget. `norm_tail_structure.py`'s check 4 is defined that way (`kind(s) == "shim" and kind(d) ==
"core"` over `_FLOW_RE`); it is *not* a live defect there, because that script's check 1 rejects any
packet-typed channel first and would fire — but the demand count itself is blind, and anything that
reuses that definition without check 1 beside it inherits a check that cannot fail.

`test_flow_only_counting_reads_zero_where_demand_is_three` pins both halves on one fixture.

### 2.3 A missing traffic multiplier the demand table could not have shown

Building this found one defect in itself, and it is worth recording because of its shape. On the
matmul artifact (`programming_examples/transformer_layer/air_project`, `matmul_bf16` over
2048×2048 · 2048×512), the demand table was correct — 8 columns, max demand 2, no overflow — while
every byte total was **8× too small**, because `air.launch (…) in (%arg5=%c8, %arg6=%c1)` repeats
its whole body and the trip counter only walked `scf.for`/`affine.for`.

Fixed, the totals close exactly against the operands: A **8,388,608 B** fetched once (= 2048×2048×2),
C **2,097,152 B** written once, and B **16,777,216 B** — the 2,097,152-byte operand refetched
**8×**, once per launch instance. That 8× amplification on B is exactly the kind of thing this
instrument exists to make visible, and it was invisible in the same run that reported the demand
correctly.

### 2.4 The back-solve, end to end, on a routed design with its own measured latency

One build-class devq job (job **293**, exit 0, no dispatch) routed a single ERT-priced candidate so
the back-solve could run against a latency measured for *the same* shape, method and tiling:

```
gemm.direct(M=64, K=512, N=512, tile_m=64, tile_k_l2=256, tile_k_l1=32,
            tile_n=64, herd_m=1, herd_n=4)
measured 122.81 us   sweep/results/baseline_512/64x512x512__direct__a66c58c881fe6e18.json
                     (role o_proj, status passed, recorded 2026-08-07 -> Turbo-conditional)
```

| column | dir | demand | budget | bytes | rate at 122,810 ns |
|---|---|---|---|---|---|
| 0 | MM2S | 2 | 2 | 655,360 | **5.336 GB/s** |
| 0 | S2MM | 1 | 2 | 65,536 | **0.534 GB/s** |

The byte total closes exactly against the operands and is itself informative: 655,360 =
B (512×512×2 = 524,288, fetched **once**) + A (64×512×2 = 65,536, fetched **twice** — one A
refetch per round of the 8 n-tiles over `herd_n=4`). C is 65,536 out.

**Read the unit.** The duration handed in is a *measured achieved* latency, so 5.336 GB/s is the
rate the run **sustained**, which is a **lower bound** on what a stall-free run would require (a
stall-free run is no longer, and a shorter duration needs a higher rate). SCALE-Sim's `CALC` proper
is the other reading — a hypothetical stall-free duration — and the function does not know which it
was handed, so `render` prints the distinction and `test_a_measured_duration_gives_a_lower_bound_on_the_requirement`
pins the ordering.

**What it says, as an inference.** Against iron's cross-toolchain 67.9–70.9 GB/s (doc 33's
order-of-magnitude import, *not* a measurement of this toolchain), 5.336 GB/s is under 8%. So this
candidate's 122.81 µs is **not shim-bandwidth-bound** — which at 33.5 MFLOP in 122.81 µs (0.27
TFLOP/s) is consistent with dispatch overhead dominating at this shape. That is a falsifiable claim
produced from a static artifact read plus one already-recorded number, with no new device time, and
it is the shape of answer the instrument exists to give.

Artifacts: `.../item25-private/backsolve.txt`, `backsolve.json`, and the routed design at
`.../item25-private/gemm64x512x512/air_project/aie.air.mlir`.

### 2.5 The ERT, seeded entirely from artifacts that already existed

`bash .../scratchpad/item25-private/seed_ert.sh`:

| source | files | added | merged | failed |
|---|---|---|---|---|
| `sweep/results/baseline_512` | 544 | 400 | 132 | 12 |
| `sweep/results/baseline_768` | 436 | 422 | 0 | 14 |
| `sweep/results/baseline_1024` | 528 | 386 | 127 | 15 |
| routed `addnorm` design | — | 2 counted descriptors | — | — |
| routed `matmul_bf16` design | — | 3 counted descriptors | — | — |

**1,213 entries: 1,208 with a measured `ns`, 5 with a counted `bytes`, 0 modelled.** The 41 `failed`
are candidates whose `status` is not `passed` — a latency from a candidate that failed its numeric
check measures the wrong thing.

**Condition.** The sweep files are dated 2026-08-04..07, and the README records that latencies
recorded before 2026-08-10 are Turbo-conditional. Every seeded entry carries that string in
`Cost.condition`; the CLI default is `npu_power_mode=unknown`, never `turbo`. NPU power mode was
verified **Turbo** at the time of this work (`xrt-smi examine -r platform | grep -i "Power Mode"` →
`Turbo`, 2026-08-12) — which matters only for provenance here, because **nothing in this item
measured a latency**.

### 2.6 The repeat-measurement spread, and what it does to a 5% band

The 259 `merged` rows are the finding. They are not duplicates in the file-hash sense: they are
repeat measurements of an action with identical shape, identical method, identical tiling **and
identical role**, differing only in when they ran.

```
gemm.fused-cast(M=512, K=4096, N=1024, tile_m=64, tile_k_l2=512, tile_k_l1=32,
                tile_n=64, herd_m=8, herd_n=4)
    1,086,524 ns  and  1,545,360 ns    — 42.2% apart
```

Across all 259: **median spread 1.6%, worst 42.2%, and 53 of 259 (20%) exceed 5%.**

The ERT was extended rather than made to pick a winner. `Cost` carries `ns_samples`, `ns_min` and
`ns_max`, and `ns` is the **minimum** — doc 23 §open item 1 settled that for this study ("compare
minimums, not medians"; the min-to-min figures there agreed to 0.3 points across a 12× shape range
while the medians did not). Dropping the repeats would have left the table asserting one number for
an action this tree has measured to vary by up to 42%.

**Consequence for the mapping search, stated as an inference:** doc 44 recommends the instrument
partly on Timeloop's "480,000 mappings within 5% of peak vary 19× in energy". On this tree's own
measurements, a 5% latency band does not separate 20% of repeat-measured actions from themselves.
Any search built on this ERT must therefore rank on `ns_min` with `ns_max` visible, or adopt iron's
1%-band-plus-tie-break shape (doc 38 §6.1) — **not** argmin on a scalar.

## 3. Measured, counted, modelled — the table to read before quoting anything

### 3.1 Measured (a stopwatch on hardware, artifact cited)

- **1,208 GEMM latencies**, `sweep/results/baseline_{512,768,1024}/*.json`, Turbo-conditional
  (pre-2026-08-10). Each carries its own file path in `Cost.provenance`.

### 3.2 Counted (exact arithmetic on a compiled artifact, no device)

- **Per-column channel demand, multiplex depth and byte totals** for any routed design. These are
  properties of `aie.air.mlir` — the same file `resource_usage.py` parses — and are reproducible
  from the file without the device being free.
- **`shim_dma.dma_transfer` byte costs**: `n_words × element_bytes × trip_count`.
- **The per-column budget, 2**, is `aircc_artifacts.SHIM_DMA_CHANNELS_PER_DIRECTION` — imported, not
  restated, so there is one definition.

### 3.3 Modelled — and there is currently **none**, deliberately

`ns_source == "modelled"` appears **zero** times in the seeded table, and the one place it was
tempting is worth naming:

> **There is no measured AIR-native shim bandwidth in this tree.** Doc [33](33-memcpy-bandwidth-scoping.md)
> deferred the memcpy operator, and iron's **67.9–70.9 GB/s** is measured on iron's toolchain — doc
> 33 marks it as *"an order-of-magnitude statement, not a measurement"*. Writing it in as the shim
> byte rate would have made every port cost a cross-toolchain constant wearing a measurement's
> label, and would have let `bottleneck` rank ports against compute as though both were measured.

So every `shim_dma` entry has `bytes` counted and **`ns` absent**, and `bottleneck` reports any
unpriced resource in `unpriced` with `is_complete == False`. That is the honest state, and it is
exactly what makes the **back-solve** the useful question: rather than dividing traffic by a
bandwidth nobody measured, it asks what bandwidth a stall-free run *would have required*.

### 3.4 Not a cost at all

`step` is an ASAP async-dependence level, not a cycle. It orders transfers and bounds concurrency;
it does not time anything.

## 4. How every check here was shown able to fail

The dominant defect class in this study is checks that could not fail — six were found in one day.
So each of the 71 tests names the defect it catches, and **16 injected defects** were run against
the pair of modules. All 16 are rejected; baseline 71/71.

| injected defect | rejected by |
|---|---|
| the per-column budget becomes a legality predicate (violations filtered) | 4 tests |
| demand counts transfers instead of distinct logical channels | 1 |
| an unknown trip count is assumed to be 1 | 2 |
| the `air.launch` iteration space stops multiplying traffic | 8 |
| an unattributed transfer is dropped instead of reported | 2 |
| the prefix guard is removed (iron defect 1) | 2 |
| the elided-traffic guard is removed (iron defect 2) | 1 |
| an unpriced resource is scored as zero instead of named | 1 |
| the back-solve accepts a zero duration | 1 |
| `charged_ns` multiplies the columns instead of taking the worst | 1 |
| `REQUIRED_ARGUMENTS` stops being enforced (a scalar `dma_transfer`) | 4 |
| `lookup` falls back to the nearest entry instead of missing | 1 |
| a repeat measurement overwrites instead of widening the spread | 5 |
| an absent cost may carry a value (unpriced reads as free) | 1 |
| a timed cost no longer needs a provenance path | 1 |
| the seed keeps candidates that failed their numeric check | 1 |

Artifact: `.../scratchpad/item25-private/mutations.txt`; harness `mutate.py`.

Two of the mutations initially came back **UNDETECTED** and both were the harness aiming at the
wrong line, not the tests — recorded because the honest reading of a mutation report requires
checking that the patch applied at all, and `mutate.py` now reports `PATCH DID NOT APPLY` as a
failure rather than as a pass.

**Every fixture is real.** `_MULTIPLEXED` is the shipped `addnorm` routed artifact, trimmed to two
columns with everything the instrument reads left verbatim; `_COMPLIANT` is the same design with
the weight stream removed, which is doc 23's `elementwise_add` row. Every check that can pass is
asserted to read *differently* on the two.

**The suite pin moved and was verified in four directions.** `run_study_host_tests.lit` goes
`357/357 in 19 modules` → **`428/428 in 21 modules`**, checked with the real FileCheck
(`/home/cj/mlir-air/my_install/mlir/bin/FileCheck`): the real output matches; `427/427` is rejected;
`20 modules` is rejected; and the **old** pin no longer matches the new output. Artifact:
`.../scratchpad/item25-private/verify_pin.sh`.

## 5. What is still open

1. ~~**The back-solve has not been exercised on a routed design with a matching measured
   latency.**~~ **CLOSED in this item** — §2.4. It ran on exactly one candidate, which is enough to
   show the loop closes and not enough to be a result. What is still open is the *sweep*: routing
   every ERT-priced candidate would make the byte side of the table as complete as the latency
   side, and it is 1,208 compiles rather than one.
2. **The shim ports are unpriced, and that is doc 33's item 11(a).** Until an AIR-native byte rate
   exists, `bottleneck` can rank compute against compute but names any port as `unpriced`. Doc 33's
   recommendation stands — build it when 11(b) (roofline) is taken, in the AIR-native herd-width
   form, not as a port of iron's degenerate `num_channels=(2,)` sweep.
3. **`stage_gap` has no measured input.** The arithmetic and both guards are pinned, but nothing in
   this tree compiles stage-truncated variants. Doc 38 §3.4 is the recipe; it is a build+measure
   item, and step 3 of it ("do not elide") is now a precondition the function enforces rather than
   advice.
4. **The ERT is JSON, not a schema table.** `schema.py` says "adding a field is a schema version
   bump", and a new table would touch a module three other threads share. A `balance` table with
   `results_io`/`schema` validation is the natural next step and was deliberately not taken here.
5. **`run_study_host_tests.lit` still has no `make` target**, unlike every other host-only arm,
   which all go through `lit_pin`. That predates this item and was left alone; it means the 428 pin
   is asserted by `lit` and not by `make`.
6. **The step axis is coarse.** ASAP levels over the launch-level async graph put a whole
   dependence-free ingress at one level. That is correct and conservative for the budget check
   (which uses `static_demand` anyway) but it means `peak_concurrent_demand` is currently a weak
   refinement. A finer axis needs the runtime sequence, not the routed design.

## 6. What this does not do

It is **not a search**, and doc 44's argument is that this was the right order: 480,000 mappings
within 5% of peak vary 19× in energy, so the objective matters more than the search, and building a
search against an unsound objective converges confidently on the wrong point — which is exactly what
iron's instrument defect did. It is also **not a fused-composition model**: doc 44's finding that
none of the five surveyed frameworks can express our central axis is unchanged by this, and
[46](46-research-tileflow.md)'s tree of scopes remains the thing to read before designing one.
