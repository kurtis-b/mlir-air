# Item 11(a) — `memcpy_bandwidth`: build it, drop it, or defer it?

Scoping investigation, 2026-08-12. Read-only; nothing in the repo changed, no build, no device job.
Repo `/home/cj/mlir-air`, branch `exper/transformer-layer-execution-studies`.

> **Branch note.** The task named tip `b777517b`. During this investigation the operator committed
> `2d6756ca` ("docs: the install is refreshed, and the divergence it caused is closed"); `b777517b`
> is an ancestor of it. That commit **closes the install-staleness caveat** in this task's
> constraints (`install-xrt/bin/air-opt` now matches `build-xrt` at 2026-08-11 13:28, verified by
> artifact rather than mtime) and — directly relevant here — rewrites item 11's own row so that
> **11(b) is now the sole claimant on the exclusive window**. That changes the recommendation's
> timing, not its direction. See §4.

> **Pmode.** NPU power mode is `Default`, not Turbo. **No latency claim in this document is
> originated here.** Every millisecond figure is quoted from a recorded log with its artifact
> named, and every ratio derived from one is marked as an inference.

---

## TL;DR

**DEFER — and bind it explicitly to 11(b)'s roofline decision, which is the only thing that
actually consumes it.**

Three facts drive that:

1. **A bandwidth operator is not the instrument that attributes the missing half.** The component
   table's own log names the missing instrument, and it is per-stage `record_kernel`/`record_cpu`
   calls on `pattern/`, not a memcpy. The 80.0 ms unattributed remainder is labelled *"host
   overhead outside every instrumented region"* — device-side by exclusion, it is not.
2. **But it is a hard input to `roofline`, which is the other half of the same queue item.**
   `roofline/run.py` refuses to run without the memcpy CSV, and `peak_bandwidth_gbps` is literally
   the slope of the memory roof. So "nothing is blocked on it" is true *today* and stops being true
   the moment 11(b) is taken — and 11(b) just became the only claimant on the window it needs.
3. **Its real value is a ceiling for the memory-bound half of the layer, and that value is robust
   to a 2× error** — which means iron's already-measured number serves every decision that can be
   made now, as a clearly-labelled imported constant. A precise AIR-native ceiling only earns its
   cost when you need to tell "at the ceiling" from "3× under it" for a *named* operator, and that
   is exactly the roofline tier's question.

**What would change it to "build":** 11(b) being taken (roofline ported) — then 11(a) is required,
not optional, and should be built in the AIR-native herd-width form of §3, not as a port.
**What would change it to "drop":** 11(b) being dropped or the roofline being re-scoped to a
compute-only ceiling.

---

## 1. What iron's `memcpy_bandwidth` measured, and which axis it lands on

### The measurement

iron's study drives `iron.operators.mem_copy.op.AIEMemCopy` over a **fixed** transfer and sweeps
the array configuration, reporting achieved DRAM↔NPU bandwidth.

Fixed shape (`/home/cj/iron/iron/applications/transformer_layer/study/memcpy_bandwidth/cases.py:10-21`):

| constant | value |
|---|---|
| `SIZE_LADDER` | `(8388608,)` elements — a one-point "ladder" |
| element type | bf16, `ELEMENT_BYTES = 2` → 16,777,216 B in |
| `total_moved_bytes` | 33,554,432 B (in + out) |
| `FIXED_TILE_SIZE` | 4096 |
| iteration schedule | 10 warmup / 500 timed (README, "Measurement") |

Swept axes, as **shipped**:

| axis | shipped values | note |
|---|---|---|
| `size` | 1 value | not swept |
| `num_cores` | `(2, 4, 8, 16)` | the real axis |
| `num_channels` | **`(2,)`** | **degenerate — never varied** |
| `bypass` | `(False, True)`; default run is `True` only | kernel vs no-kernel path |

The study's own README states the mapping the sweep is really for
(`.../memcpy_bandwidth/README.md:41-46`): *"With `2` channels fixed, the sweep maps directly to
shim-tile usage"* — 2 cores → 1 shim tile, 4 → 2, 8 → 4, 16 → 8. And the operator's design file
says the same thing in a comment (`/home/cj/iron/iron/operators/mem_copy/design.py:163`):
*"Memcpy is designed to use every column's shimDMA in-out pairs"*.

The canonical plot is a **single bar chart over shim-tile count**
(`results/memcpy_bandwidth/bandwidth_by_shim_tiles.svg`, README "Output Contract").

**So iron's study is a shim-tile scaling curve for peak achievable DRAM bandwidth**, not a
four-dimensional sweep. That reframes the queue row's premise (see §1.3).

### Measured result

`/home/cj/iron/iron/applications/transformer_layer/results_unattended_full_suite_20260801_023954/memcpy_bandwidth/results.csv`
(dated 2026-08-02, the `bypass=all` 8-row form):

| cores | shim tiles | bypass GB/s | kernel GB/s | kernel status |
|---|---|---|---|---|
| 2 | 1 | 44.68 | 45.28 | passed |
| 4 | 2 | 64.95 | 63.64 | passed |
| 8 | 4 | **70.86** | 70.79 | **failed_validation** (3,876 bad elements) |
| 16 | 8 | 67.88 (`is_overall_peak`) | 67.80 | **failed_validation** (68,297 bad elements) |

Two things to carry forward:

- **The curve saturates at ~4 shim tiles.** 1 → 2 tiles buys 45 %; 2 → 4 buys 9 %; 4 → 8 is
  *negative* (70.86 → 67.88). The useful information in the whole study is essentially "≈68–71
  GB/s, reached by 4 shim tiles."
- **iron's own kernel arm is broken at 8 and 16 cores** — two of eight rows are
  `failed_validation`. The default run hides this by measuring `bypass=True` only. A port that
  reproduced the sweep faithfully would be porting a partially-red study.

The shorter default run (`results/memcpy_bandwidth/results.csv`, 2026-08-03, bypass-only) agrees:
44.69 / 64.67 / 64.32 / 67.58, peak 67.58.

### Which axis of doc 03's knobs-and-costs map this lands on

Doc 03 §The vocabulary, "The axes — three knobs, three costs"
(`docs/plans/transformer-layer-execution-studies/03-measurement-model.md:137+`):

| kind | axis | instrument |
|---|---|---|
| knob | graph coverage | `test_attention_path.py` |
| knob | execution boundary | `host_submissions` / `runlist_entries` / `air_launches` / `herd_launches` |
| knob | AIE role style | core→core flow count |
| cost | reconfiguration | `context_loads` / `kernel_attaches` |
| cost | **off-chip traffic** | **`bytes_transferred`** |
| cost | synchronization | `sync_boundaries`, `device_ms`/`sync_ms`/`host_cpu_ms` |

`memcpy_bandwidth` lands on the **off-chip traffic** cost — but *not* as a second instrument for
it. `bytes_transferred` measures **how many** bytes a mode moves. A memcpy study measures **how
fast bytes can move at all**. It is the **denominator that converts the traffic cost from a count
into a time**, and it is the only thing in the seven studies that supplies one.

That is a real gap in the axis map: today the traffic cost is measured in bytes and the
synchronization cost in milliseconds, and **nothing in the tree relates the two**. Whether that
gap is worth closing is §4.

Secondarily it touches the **AIE role style** knob from below — a memcpy herd is the minimal
`single-function` design, and its shim-channel allocation is exactly what `resource_usage.py`
already reports. See §3.

### Correcting the queue row's premise

The queue row (README item 11) says iron's operator has *"four case axes"* and that `num_channels`
*"is not an input here at all"*. Both halves need a footnote:

- **The four-axis framing overstates iron's study.** As shipped, `size` has one value and
  `num_channels` has one value. iron's own case matrix is **two** axes (`num_cores` × `bypass`),
  and `num_cores` is a proxy for shim-tile count.
- **So dropping `num_channels` costs nothing, because iron never varied it either.** The row is
  right that AIR cannot accept it as an input; it is worth recording that this removes no
  information from the shipped study. The axis that *does* carry the study — shim-tile count — is
  expressible in AIR (§3).

This matters for the decision: the re-scope is smaller than the row implies. The obstacle is not
"one of four axes is missing"; it is "the operator does not exist" (§3), and separately "what is
it for" (§2, §4).

---

## 2. What already covers this ground here — and is a bandwidth operator the instrument for the
   unattributed 50.1 %?

**No. It is answering a different question, and the component table's own log says which
instrument is missing.**

### What the component table actually reported

`agents/.state/devq/jobs/job-000246.log` (job 246, class `measure`, exit 0), `offload` @ 1024:

```
| group                        | kind       | ms     | components | complete |
| GEMMs (NPU)                  | `device`   | 64.388 | 0/8        | no       |
| Non-linear operations (host) | `host_cpu` | 10.914 | 5/5        | yes      |
| Data sync                    | `sync`     |  4.494 | 0/0        | yes      |

attributed: 79.795 ms
whole layer: 159.795 ms
UNATTRIBUTED: 80.000 ms (50.1%) -- host overhead outside every instrumented region.
```

Same run: `dispatch totals: submissions 30 entries 30 air 30 herd 90 sync 90 bytes 99090432`,
`reconfiguration: context_loads 1 kernel_attaches 4 over 30 dispatches`, 10/10 stages clean.

### The log names the missing instrument, and it is not a memcpy

Verbatim from the same log:

```
not attributed to components:
- GEMMs (NPU): q_proj, k_proj, v_proj, attn_scores, attn_output, output_proj, up_proj, down_proj
  These need the per-stage record_kernel/record_cpu calls doc 09 scopes (2026-08-08);
  the taxonomy above is what they will fill in.
```

`study/component_groups.py:21-43` scopes it identically — the module is *"an honest partial"*
because *"`pattern/` contains no timing at all today"* and per-stage `record_kernel`/`record_cpu`
calls are *"new instrumentation on an untimed seam"*.

And the 80.0 ms is explicitly **host** overhead — *"outside every instrumented region"*. A DRAM
bandwidth ceiling has no purchase on it whatsoever.

### The arithmetic bound: how much of the layer could a bandwidth result even speak to?

*(Inference — arithmetic on the job-246 figures above. Marked as such.)*

Decompose the 159.795 ms by whether a bandwidth ceiling could constrain it at all:

| component | ms | share | bandwidth-sensitive? |
|---|---|---|---|
| unattributed host overhead | 80.000 | 50.1 % | **no** — host-side by the log's own label |
| GEMMs (NPU), mode total | 64.388 | 40.3 % | partly — includes both compute and shim streaming, **not separable today** |
| non-linear ops (host) | 10.914 | 6.8 % | no |
| data sync | 4.494 | 2.8 % | **yes** — this is `bo.sync()` time by definition |

So the **only** component a bandwidth ceiling directly bounds is **4.494 ms of 159.795 — 2.8 % of
the layer.** And splitting the 64.388 ms device total into compute versus movement requires the
*same* per-stage instrumentation the log already names. A memcpy operator does not unlock it.

That is the decisive answer to the question as posed: **the bandwidth operator is not the
instrument that would attribute the missing half.** The instrument for that is per-stage timing on
`pattern/`, which is a different, already-scoped piece of work.

### What the sync figure does say — and why it is interesting but small

*(Inference. Cross-toolchain comparison; see the caveat below.)*

- offload @1024 sync rate: 99,090,432 B ÷ 4.494 ms = **≈ 22.0 GB/s**
- against iron's measured peak of **67.9–70.9 GB/s**

So `offload`'s host↔device syncs run at roughly **a third of achievable bandwidth** — there is
genuinely ~3 ms of headroom in that path. But ~3 ms of a 159.8 ms layer is ~1.9 %. The finding is
real and small.

> **Caveat, stated because the project enforces it.** iron's 67.9 GB/s is measured on iron's
> toolchain (results dated 2026-08-02, same machine by location but not proven so), and AIR's
> `bytes_transferred` is defined by `schema.py:218` as *"Bytes moved across those boundaries per
> layer"* while iron's `bandwidth_gbps` is `total_moved_bytes / latency` (verified: 33,554,432 ÷
> 496.5 µs = 67.58, matching the CSV column). The two are like-for-like **in definition** — both
> count both directions — but the comparison imports a constant across toolchains. It is an
> order-of-magnitude statement, not a measurement.

### What else in the tree already covers this ground

| already here | what it covers | file |
|---|---|---|
| `bytes_transferred` (schema v1) | the traffic **count** for every mode, per layer | `study/schema.py:218` |
| `device_ms` / `sync_ms` / `host_cpu_ms` (schema v2) | the time decomposition the traffic would be priced against | `study/schema.py:317-350` |
| `bandwidth_gbps` | **already a schema column** — "Achieved bandwidth for movement-bound candidates", in the *tuning* table | `study/schema.py:429-433` |
| `shim_{s2mm,mm2s,dma}_channels_used` + utilizations | what iron passes in as `num_channels`, **as an observed column** | `study/resource_usage.py:171-178` |
| `core_to_core_flows` | the AIE-role-style knob, per design | `study/resource_usage.py` |
| component groups | the attribution table, honestly partial | `study/component_groups.py` |

Note `bandwidth_gbps` **already exists in the schema** — so the column a memcpy study would write
is defined; only the producer is missing. That lowers the port cost slightly and is worth knowing.

### The dependency the queue row does not mention: `roofline`

This is the fact that flips "drop" to "defer".

`roofline/run.py` **hard-requires** the memcpy CSV:

- `/home/cj/iron/.../study/roofline/run.py:1675-1678` — refuses to run:
  `"Missing memcpy-bandwidth CSV: {path}. Run study.memcpy_bandwidth.run first."`
- `:1710-1711` — `peak_bandwidth = peak_memcpy_bandwidth_gbps(memcpy_rows)` and
  `peak_bandwidth_by_shim_tiles = peak_memcpy_bandwidth_by_shim_tiles_gbps(memcpy_rows)`
- `:1462` — **the memory roof is that number**:
  `roof_y = np.minimum(peak_bandwidth_gbps * x_values, compute_ceiling)`
- `:1599-1601` — per-shim-tile roofs, falling back to `peak_bandwidth_gbps * shim_tiles / 8`

And the compute half of the roof is **not** measured — it is derived from constants
(`run.py:1359-1366`, constants at `:49-53`): 32 cores × 64 bf16 MACs/cycle × 2 FLOPs/MAC × 1.8 GHz
= 7372.8 GFLOPS. **iron measures the memory roof and computes the FLOP roof.** So the memcpy study
is the *only* empirical input to iron's roofline, and a roofline drawn without it has an unsourced
slope — which on this project is precisely a "claim without an artifact".

`memcpy_bandwidth` is also wired into iron's suite plumbing
(`study/unattended_smoke_job.py:23,46,73`; `study/unattended_reboot.py:30,411,962-971`), i.e. it is
a first-class member of the unattended suite that Phase G (queue item 12) would mirror.

### Verifying "nothing is blocked on it" independently

**Confirmed at the gate level.** `study/smoke_gate.py` takes its expected CSVs from repeatable
`--expect` CLI arguments (`:117-118`) with **no hardcoded study list**; `study/manifest.py`
likewise (`:165`). Grep finds **no** `memcpy` reference anywhere under
`programming_examples/transformer_layer/study/`, and `study/cases.py` defines no study-id registry
that reserves a slot for it. Nothing in the AIR tree names it.

So the doc claim holds **today**, and its scope is exactly: *no gate and no shipped module
references it*. It says nothing about 11(b).

---

## 3. What an AIR-native equivalent would actually be

The operator genuinely does not exist. Doc 09:179-203 is right that the two nearest examples are
`herd sizes=[1, 1]`; I verified all three passthrough variants:

- `programming_examples/passthrough/passthrough_dma/passthrough_dma.py:55` — `sizes=[1, 1]`
- `programming_examples/passthrough/passthrough_channel/passthrough_channel.py:49` — `sizes=[1, 1]`
- `programming_examples/passthrough/passthrough_kernel/passthrough_kernel.py:54` — `sizes=[1, 1]`,
  `link_with="passThrough.cc.o"`

### The pieces that exist

| piece | where | note |
|---|---|---|
| multi-worker tiled copy shape | `programming_examples/channel_examples/channel_size/channel_size.py` | `Channel("ChanIn", size=[H/th, W/tw])` + `herd(sizes=[H/th, W/tw])` + per-worker `ChannelGet`/`ChannelPut` (`:37-38, 67-71, 88, 102`). **This is the multi-core form `passthrough_*` lacks** — exactly as doc 09 says |
| the kernel | `programming_examples/passthrough/passthrough_kernel/passThrough.cc` | **the same kernel iron links** — iron's `op.py:63-73` pulls `aie_kernels/generic/passThrough.cc`. The non-bypass arm is free |
| the `bypass` distinction | — | iron: `of_ins[i].cons().forward()` with no `Worker` (`design.py:191-192`). AIR: herd body does `ChannelGet` → `ChannelPut` with no loop between, vs `link_with` + kernel call |
| the observed channel column | `study/resource_usage.py:171-178` | `shim_s2mm_channels_used`, `shim_mm2s_channels_used`, `shim_dma_channels_used` + utilizations |
| the output column | `study/schema.py:429-433` | `bandwidth_gbps` already defined |
| the routed artifact to read | `study/aircc_artifacts.py:63` | `aie.air.mlir`, and it is emitted even when the core-ELF link fails (doc 09:217-224) |

### What `num_channels` becomes

Not an axis — an **observed column**, and the observation has structure worth measuring.

Doc 23's governing rule (`23-rules-and-open-items.md:11`):

> **A column has two shim MM2S channels, and the budget is per COLUMN across the whole segment.**
> Exceed it and AIR packet-multiplexes onto one queue.

`SHIM_DMA_CHANNELS_PER_DIRECTION = 2` (`study/aircc_artifacts.py:69`), 8 columns on the part.
Evidence that routing produces the count without being asked — job 238
(`agents/.state/devq/jobs/job-000238.log:14,17`):

```
[verify]   shim ch / tiles         17 / 8
[resource-usage] norm_tail: 24 cores, ... shim 17ch over 8 tiles, flows 16/40 core->core
```

**The mapping to iron's axis is exact and pleasing.** A memcpy worker needs one L3→L1 stream and
one L1→L3 stream. So for a herd of `[rows, cols]`:

| AIR herd | workers | cols | MM2S per column | iron equivalent | vs doc 23 budget |
|---|---|---|---|---|---|
| `[1, 2]` | 2 | 2 | 1 | 2 cores / 1 shim tile | under |
| `[1, 4]` | 4 | 4 | 1 | 4 cores / 2 shim tiles | under |
| `[1, 8]` | 8 | 8 | 1 | 8 cores / 4 shim tiles | under |
| `[2, 8]` | 16 | 8 | 2 | **16 cores / 8 shim tiles** | **exactly at budget** |
| `[4, 8]` | 32 | 8 | 4 | *not expressible in iron* | **over — packet-multiplexed** |

*(Inference — the column arithmetic is mine, derived from doc 23:11 and the one-in/one-out
structure of a memcpy worker. It is a prediction the design would test, not a measured fact.)*

**iron's top rung is precisely AIR's budget boundary**, and AIR can express one rung beyond it that
iron cannot. That last row is the AIR-native question:

> The question stops being "how does bandwidth vary as I ask for more channels" and becomes "what
> does the compiler allocate, and what bandwidth does that reach" — which is a compiler result, not
> a configuration sweep. *(doc 09:201-203)*

Concretely: **what does crossing the per-column shim budget into packet multiplexing cost in
bandwidth?** Nothing in the tree answers that, and it bears directly on the design rule every
downstream phase (J7b, J7c, R1, R2) is built around. That is the one genuinely new thing this
study would produce.

### Sketch of the build

| piece | shape | est. size |
|---|---|---|
| `builders/mem_copy.py` (or `programming_examples/passthrough/passthrough_multicore/`) | `channel_size.py` generalized: parameterized herd `[rows, cols]`, bf16, tile 4096, bypass flag switching `link_with` | ~150–250 lines |
| `study/memcpy_bandwidth.py` | cases + dispatch + timing + validation + CSV; drops iron's `num_channels` axis and all plotting | ~250–350 lines (iron's runner is 586 with plotting) |
| schema | reuse `bandwidth_gbps`; likely a new small table or reuse of `TUNING_FIELDS` | small |
| `cases.py` | a shim-width ladder, one fixed size | ~40 lines |
| gate | one lit recipe + host tests, plus a negative control | ~100 lines |
| device runs | via `agents/scripts/devq.sh`, `--class measure`, Turbo required | 5 rungs × 2 arms |

*(All sizes are estimates, marked as such.)*

**Risks worth pricing before committing:**

1. **The over-budget rung lands in the packet-multiplexing path** — the regime that until H9
   *silently misdelivered every trip after the first on more than one column* (doc 23:11-13). H9
   fixed it, but this design would be deliberately driving into the tree's most historically
   fragile routing regime. It needs a real numeric check, not a structural one.
2. **iron's own kernel arm is red at 8 and 16 cores** (§1). Whatever the non-bypass arm is
   measuring there, it is not a validated copy. An AIR port should not inherit the expectation
   that it passes.
3. **Turbo is a measurement condition** (trap 0). A bandwidth number taken at `Default` pmode is
   worthless, and pmode resets on every reboot/driver reload.

---

## 4. Recommendation: **DEFER**, bound to 11(b)

### The recommendation

**Do not build it now. Bind it to the roofline decision, and record the binding in the queue row so
the coupling is not rediscovered.**

Rewrite item 11 so that (a) and (b) are **not** "two unlike halves" but one sequenced pair:

> **11(b) roofline requires 11(a).** `roofline/run.py` refuses to run without the memcpy CSV and
> uses `peak_bandwidth_gbps` as the memory roof's slope. If 11(b) is taken, 11(a) is its
> prerequisite and must be built first, in the AIR-native herd-width form. If 11(b) is dropped or
> re-scoped to a compute-only ceiling, 11(a) can be dropped with it.

### Why not "build now"

- The problem it was reached for — the 50.1 % unattributed — **is not a bandwidth problem**, and
  the log names the real instrument (§2). Building it now would answer a question nobody asked.
- Its directly-bounded share of the measured layer is **2.8 %** (§2).
- Nothing gates on it; verified independently, not taken from the docs (§2).
- It would drive into the packet-multiplexing regime (§3, risk 1) for a result no current phase
  consumes.

### Why not "drop"

- **The roofline dependency is real, verified in source, and in the same queue item.** Dropping
  (a) silently guts (b).
- **The memory-bound half of the layer is exactly where the study's live work is.** iron's own
  roofline points show which operators sit left of the ridge — from
  `.../roofline/kernel_points.csv` (n=76, OI range 0.167 → 8286.9, median 42.7), the operators with
  operational intensity < 30 FLOPs/byte are: `add`, `causal_mask`, `attn_scale`, `attn_softmax`,
  `add_norm`, `add_norm1`, `add_norm2`, `ln1`, `ln2`, `gelu`. **That is the norm tail and the FFN
  elementwise stages — precisely what R1/R2 and the `fused` resident tail (queue item 6) are
  rebuilding.**
- And the ceiling is what makes those points *readable*. Same CSV: `add` @ seq 16384 measures
  **68.12 / 68.18 GB/s** against a measured peak of 67.88 — i.e. **at the ceiling, nothing left to
  win**; `add` @ seq 256 measures **19.55 / 24.90 GB/s** — **3.5× of headroom**. Without a measured
  peak those two numbers are indistinguishable in kind. That is the study's actual product, and it
  is a good one.

### What the recommendation is worth — the honest limit

*(Inference, arithmetic on cited figures. Marked.)*

The reason this is "defer" and not "build" is that **every decision available today is robust to a
2× error in the ceiling**, so iron's existing number already serves them.

Worked example, the most decision-relevant one available. Doc 31a's resident-tail byte floor is
**84.0 MiB packaged → 16.5 MiB resident at 1024**. Priced at iron's ~67.9 GB/s:

- 84.0 MiB = 88,080,384 B → ≈ 1.30 ms
- 16.5 MiB = 17,301,504 B → ≈ 0.25 ms
- **prize ≈ 1.0 ms**, against job 246's 159.795 ms layer → **≈ 0.65 %**

That is a genuinely useful result — it says the resident tail's DRAM-traffic prize at 1024 is worth
about a millisecond, not tens of them, so its justification has to be something other than
bandwidth (or a longer sequence). **And the conclusion survives the ceiling being wrong by 2× in
either direction** (0.5 ms or 2 ms — same verdict). So it can be stated *now*, from the imported
constant, without building anything.

The build only earns its cost when the question becomes per-operator and quantitative — "is this
`layer_norm` at the ceiling or 3× under it?" — which is the roofline tier's question and nobody
else's.

### Interim action, at roughly zero cost

Record iron's measured ceiling in the AIR docs **as an explicitly-labelled imported constant**, so
that order-of-magnitude bandwidth reasoning is available without being mistaken for an AIR result:

> **Imported constant, not an AIR measurement.** NPU DRAM↔array bandwidth measured by iron's
> `memcpy_bandwidth` study at 44.7 GB/s (1 shim tile) → 64.9 (2) → 70.9 (4) → 67.9 (8), fixed
> 32 MiB moved, bypass path, Turbo.
> Artifact: `/home/cj/iron/iron/applications/transformer_layer/results_unattended_full_suite_20260801_023954/memcpy_bandwidth/results.csv`
> (2026-08-02). Different toolchain; use for order-of-magnitude bounds only. Superseded the moment
> item 11(a) produces an AIR-native number.

This is worth doing because the alternative — someone needing a bandwidth bound and inventing one —
is exactly the failure mode the project's "no claim without an artifact" rule exists to prevent.

### Triggers, stated so the deferral is not indefinite

| trigger | action |
|---|---|
| 11(b) taken / roofline ported | **Build 11(a) first**, AIR-native form (§3). It is a prerequisite, not a companion |
| 11(b) dropped, or roofline re-scoped to compute-only | **Drop 11(a)** |
| Phase G (item 12) taken | Re-examine — iron wires memcpy into the unattended suite (`unattended_smoke_job.py`, `unattended_reboot.py`), so the suite shape may pull it in |
| Someone needs a per-operator "fraction of peak" for the norm tail / FFN elementwise stages | **Build it** — that is the one question only this instrument answers, and it is R1/R2's neighbourhood |
| A design deliberately crosses the per-column shim budget and its bandwidth cost matters | **Build it**, at least the `[4, 8]` rung — nothing else measures that |

### One caution on scope, if it is ever built

Build it as **a ceiling plus a compiler-behaviour probe**, not as a port of iron's study. The port
framing carries a degenerate `num_channels` axis, a one-point size "ladder", and a kernel arm that
is red at half its rungs. The AIR-native framing — herd width against observed shim allocation,
including the over-budget rung iron cannot express — is both smaller and the only version that
produces something iron's does not have.

---

## Appendix — artifacts cited

| claim | artifact |
|---|---|
| iron sweep axes, fixed size, degenerate `num_channels` | `/home/cj/iron/iron/applications/transformer_layer/study/memcpy_bandwidth/cases.py:10-21` |
| shim-tile mapping, output contract, 10/500 iterations | `/home/cj/iron/.../study/memcpy_bandwidth/README.md:36-49, 80-130` |
| "every column's shimDMA in-out pairs"; bypass = `forward()` no Worker | `/home/cj/iron/iron/operators/mem_copy/design.py:163, 191-192` |
| operator links `aie_kernels/generic/passThrough.cc` | `/home/cj/iron/iron/operators/mem_copy/op.py:63-73` |
| measured bandwidths; 2 rows `failed_validation` | `/home/cj/iron/.../results_unattended_full_suite_20260801_023954/memcpy_bandwidth/results.csv` |
| bypass-only default run | `/home/cj/iron/.../results/memcpy_bandwidth/results.csv` |
| roofline refuses without memcpy CSV | `/home/cj/iron/.../study/roofline/run.py:1675-1678` |
| memory roof **is** `peak_bandwidth_gbps` | `/home/cj/iron/.../study/roofline/run.py:1462`, peaks at `:1710-1711`, per-tile at `:1599-1601` |
| compute roof is theoretical, not measured | `/home/cj/iron/.../study/roofline/run.py:1359-1366`, constants `:49-53` |
| operational-intensity distribution; memory-bound operator list; `add` at/under ceiling | `/home/cj/iron/.../results_unattended_full_suite_20260801_023954/roofline/kernel_points.csv` |
| memcpy in iron's unattended suite | `/home/cj/iron/.../study/unattended_smoke_job.py:23,46,73`; `.../study/unattended_reboot.py:30,411,962-971` |
| component table: 64.388 / 10.914 / 4.494, 79.795 of 159.795, 80.0 unattributed (50.1 %) | `agents/.state/devq/jobs/job-000246.log` |
| bytes 99,090,432; sync 90; `context_loads 1 kernel_attaches 4` over 30 dispatches | same log |
| missing instrument = per-stage `record_kernel`/`record_cpu` | same log; `programming_examples/transformer_layer/study/component_groups.py:21-43` |
| `bytes_transferred` definition | `programming_examples/transformer_layer/study/schema.py:218` |
| `device_ms`/`sync_ms`/`host_cpu_ms` definitions | `programming_examples/transformer_layer/study/schema.py:317-350` |
| `bandwidth_gbps` already a schema column | `programming_examples/transformer_layer/study/schema.py:429-433` |
| shim channel columns (the observed `num_channels`) | `programming_examples/transformer_layer/study/resource_usage.py:171-178` |
| routed artifact pinned; AIE2P constants; `SHIM_DMA_CHANNELS_PER_DIRECTION = 2` | `programming_examples/transformer_layer/study/aircc_artifacts.py:63, 67-69` |
| 17 shim channels over 8 tiles, unrequested | `agents/.state/devq/jobs/job-000238.log:14,17` |
| per-column shim budget rule | `docs/plans/.../23-rules-and-open-items.md:11` |
| knobs-and-costs axis map | `docs/plans/.../03-measurement-model.md:137+` |
| doc 09's re-scope of this item | `docs/plans/.../09-phase-f-study-harness.md:179-203` |
| resident-tail byte floor 84.0 → 16.5 MiB @1024 | `docs/plans/.../31a-resident-byte-floor.md` (via README status board) |
| all three passthrough variants are `herd sizes=[1,1]` | `programming_examples/passthrough/passthrough_dma/passthrough_dma.py:55`; `.../passthrough_channel/passthrough_channel.py:49`; `.../passthrough_kernel/passthrough_kernel.py:54` |
| multi-worker copy shape exists | `programming_examples/channel_examples/channel_size/channel_size.py:37-38, 67-71, 88, 102` |
| `passThrough.cc` present in AIR tree | `programming_examples/passthrough/passthrough_kernel/passThrough.cc` |
| no gate references memcpy (`--expect` is CLI-supplied) | `programming_examples/transformer_layer/study/smoke_gate.py:117-118`; `.../manifest.py:165` |
| install refreshed; 11(b) sole claimant on the window | commit `2d6756ca` |

**Inferences, collected** (each marked in place): the 2.8 %/40.3 %/50.1 % bandwidth-sensitivity
split; offload's ≈22.0 GB/s sync rate and ≈1.54 GB/s device-total rate; the ridge-point arithmetic;
the herd-width → MM2S-per-column mapping table; the ≈1.0 ms / ≈0.65 % resident-tail prize; all
build-size estimates.

---

## `[2026-08-12]` Verified at merge, not taken on report

Five load-bearing claims were re-checked independently before this document was accepted. All
five hold:

| claim | re-check |
|---|---|
| `roofline` hard-requires the memcpy CSV | `run.py` raises `FileNotFoundError("Missing memcpy-bandwidth CSV: … Run study.memcpy_bandwidth.run first.")` — confirmed |
| the memory roof **is** `peak_bandwidth_gbps` | `roof_y = np.minimum(peak_bandwidth_gbps * x_values, compute_ceiling)` — confirmed verbatim |
| `num_channels` degenerate, `SIZE_LADDER` one point | `SIZE_LADDER = (8388608,)`, `NUM_CHANNELS = (2,)` — confirmed; the real axes are `TWO_CHANNEL_CORES = (2,4,8,16)` × `BYPASS_VALUES = (False, True)`, i.e. **two** |
| kernel arm `failed_validation` at 8 and 16 cores, 2 of 8 rows | confirmed from the `run_status` / `failure_message` columns: `validation failed: output=3876` (8 cores), `output=68297` (16 cores), both `run_status` False. The passing arm was quoted correctly |
| the ceiling figures | confirmed from `results_unattended_full_suite_20260801_023954`: 44.68 / 64.95 / 70.86 / 67.88 GB/s |

### One correction, and it changes how the constant must be quoted

The document quotes the ceiling as four point values. **The 4-shim-tile (8-core) rung is not stable
across runs**, and the way the artifacts are laid out actively invites over-counting it:

| artifact | 8-core, bypass |
|---|---|
| `results/` | 64.32 GB/s (latency 521.7 µs) |
| `results_unattended_execution_smoke_20260803_024305/` | 64.32 GB/s (latency 521.7 µs) |
| `results_unattended_execution_smoke_20260803_095245/` | 64.32 GB/s (latency 521.7 µs) |
| `results_unattended_full_suite_20260427_131305/` | 70.24 GB/s (latency 477.7 µs) |
| `results_unattended_full_suite_20260801_023954/` | 70.86 GB/s (latency 473.6 µs) |

The first three agree **to the digit on latency**, which means they are one measurement copied into
three trees, not three corroborating runs. So this is **1 smoke measurement against 2 full-suite
runs, differing ~10%** — not "three artifacts say 64.3". Anyone reading the tree by counting files
would reach the opposite conclusion from anyone reading it by counting measurements. The other three
rungs are tight across all runs (2-core 44.68–45.63, 4-core 64.34–64.95, 16-core 67.58–68.38).

Two consequences. **Quote the imported constant as a band, not a point** — and note that the peak
is at 4 shim tiles, not 8, so the curve is **non-monotonic** and "more shim tiles is more bandwidth"
is false on iron's own data. And the interim block in §"Interim action" should carry that caveat
wherever it is copied. This is the project's standing
[[compare-distributions-not-numbers]] rule reaching an imported number: comparing one recorded
figure against a fresh run has produced two wrong published claims here before.
