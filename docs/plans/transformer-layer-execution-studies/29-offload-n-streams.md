# 29 — `offload`: N instruction streams under one xclbin

`[2026-08-09]` **Landed** (`93e15a64`). The second half of the corrected `offload`
mode: the array is configured **once** per layer instead of thirty times.

[03](03-measurement-model.md) defines `offload` as *"reconfiguration MINIMIZED by
dynamic partitioning"*. Until this landed the mode did the opposite — it
implemented iron's linear/non-linear *partition* while paying a full `hw_context`
teardown and setup before **every** dispatch, which is the maximum reconfiguration
cost available rather than the minimum. 03 said so ("not yet measuring what it is
for"); this closes it.

## What it does

```
[offload] stages: 10/10 clean
[offload] dispatch totals: submissions 30 entries 30 air 30 herd 90 sync 90 bytes 99090432
[offload] reconfiguration: context_loads 1 kernel_attaches 4 over 30 dispatches
```

Five GEMM shapes compiled into one xclbin by chaining `xclbin_input`, loaded once,
with each shape binding its own kernel and its own instruction stream onto the
standing context.

**The dispatch vector is unchanged, and that is the design, not a disappointment.**
This mode makes one `run_sequence` call per GEMM whether the array is configured
once or thirty times, so all six fields are identical to the shipped ELF path.
That is what makes the existing gate a *correctness check on the change* rather
than a measurement of it: if any field moved, the change broke something.

It also means reconfiguration **cannot** be a seventh vector field — the vector
cannot see it. `KernelCache.reconfiguration_counts()` counts it separately, and
`describe_offload` prints it.

## The three identifiers, and why each matters

A stream needs all three distinct. No caller in this tree set any of them before
this phase, and only one fails loudly:

| identifier | what it keys | duplicate ⇒ |
|---|---|---|
| `kernel_name` | the `EMBEDDED_METADATA` entry | **xclbinutil REFUSES the merge** — *"Kernel name already exists in the EMBEDDED_METADATA section: 'MLIR_AIE'"*. The only loud one |
| `instance_name` | the kernel's name in the xclbin | the loader's **substring** match (`xrt.py:634`) returns whichever came first — the wrong program with the right buffers |
| `kernel_id` | the PDI the kernel routes to in the merged `AIE_PARTITION` | the second kernel executes against the **first's array configuration**: `ERT_CMD_STATE_TIMEOUT` at one shape, garbage at `mean_rel_L1` 1.41 **with no error raised** at another |

`pattern/offload` previously named every `drain` GEMM `matmul_bf16` and set no
kernel id at all — harmless when each artifact carries its own xclbin, fatal when
they share one. At 1024 all five shapes resolve to `drain`, so they would have
collided on every axis at once.

**`probe_one_xclbin_n_streams.py` found only the first two.** It sets
`kernel_name` and `instance_name` to the same string, so it never exercised
`kernel_name` alone; the third surfaced only when a real five-shape mode was built
on the mechanism. Worth remembering before reading a two-kernel probe as proof
that an N-kernel path works.

## What landed, by file

| piece | where |
|---|---|
| `attach_kernel` — bind another kernel out of an already-loaded xclbin, reusing device and context | `python/air/backend/xrt.py` |
| `compile_shared_xclbin` — the chained build; validates all three identifiers up front | `llms/shared/infra/cache.py` |
| `ensure_loaded` — artifacts sharing an `output_binary` share one context | `llms/shared/infra/cache.py` |
| `reconfiguration_counts()` — the observable | `llms/shared/infra/cache.py` |
| `plan_submissions(config_of=...)` — the xclbin split rule keyed on **configuration** identity, `artifact_of` as default proxy | `llms/shared/infra/dispatch.py` |
| per-shape identifiers, own cache dir, no eviction on the shared path | `pattern/offload/offload.py` |

Two mechanical traps, both found by hitting them:

- **Each link in the chain needs its own output name.** aircc writes relative to
  cwd, so reusing one base name feeds a compile the file it is about to write and
  xclbinutil refuses: *"The following output file is also used for input"*.
- **Only the last link holds every kernel**, so a partial rebuild is meaningless.
  Any stale member rebuilds the whole chain.

## The failure that is worth more than the feature

**The first run that looked like it worked was doing nothing.** It reported
`context_loads 5 kernel_attaches 0` — one context per artifact — while every
printed line still claimed one xclbin.

The runtime imports `air` from `install-xrt/`, not the `python/air/` source that
was edited. `attach_kernel` and `loaded_binary` simply did not exist at run time,
and the lookup was written as `getattr(backend, "loaded_binary", None)`, which
swallowed the missing attribute and **degraded silently to "no sharing"**.

That is the [15](15-environment-notes.md) toolchain-staleness trap wearing a new
hat, and the damage would not have been a crash — it would have been a mode
reporting a 30× reconfiguration reduction it never made. `ensure_loaded` now
**raises** when artifacts share a binary and no loaded backend reports it, naming
the stale install as the likely cause.

**The general rule this earns:** a capability probed by `getattr(..., default)`
degrades quietly by construction. When the degraded path is *indistinguishable in
the logs* from the working one, the default has to be an error.

## No latency claim is made

Four interleaved A/B runs at seq 1024, five samples each:

| | median avg | median min | min spread |
|---|---|---|---|
| shared xclbin | 164.3 ms | 158.6 ms | **8.0 ms** |
| ELF (shipped) | 182.5 ms | 163.9 ms | 20.5 ms |

**The distributions overlap on both statistics, so no latency difference is
established.** The mode improved 30× on its own defining axis with no measurable
time effect at this shape, which is itself a result: the `hw_context` reload is
not a dominant cost at 1024 for this workload.

One lead, explicitly not a claim: the shared path's min-spread is 2.5× tighter.
That would fit eviction driving the 120% intra-walk variance
[27](27-common-ladder-result.md) recorded for this mode, and it is the reason the
next measurement below is worth taking. n = 4.

## `[2026-08-09]` The lead, taken: packaging collapses the variance

The measurement [27](27-common-ladder-result.md) said would settle it, run as
**four walks** — `{ELF, shared} × {walk 1, walk 2}` at 512 and 1024, interleaved
A/B/A/B so time-of-day drift cannot align with an arm — with **`runlist` walked
inside every one as a same-conditions control**, because 27's 2–10% band was
measured on a different day. `--warmup 2 --samples 5`, one process per rung,
every cache warmed as a `--class build` job first. 16/16 rungs passed.

**Intra-walk spread, `(max-min)/min` over five samples — 27's own statistic:**

| mode | seq | ELF w1 | ELF w2 | shared w1 | shared w2 |
|---|---|---|---|---|---|
| `offload` | 512 | **316.9%** | **134.1%** | 17.6% | 14.0% |
| `offload` | 1024 | 9.0% | 10.5% | 5.8% | 5.5% |
| `runlist` (control) | 512 | 15.3% | 7.6% | 5.5% | 8.1% |
| `runlist` (control) | 1024 | 6.5% | 2.4% | 4.0% | 4.1% |

**At 512 the effect is unambiguous and it is an order of magnitude**: 316.9% and
134.1% on the ELF path against 17.6% and 14.0% on the shared one, in both walks,
while the control stayed inside its band in the same walks. **Switching this
mode to the shared xclbin removes its variance.** That is the result, and it is
solid.

### What this does NOT establish, and the experiment that would

**It does not isolate `_evict_context`.** The env var changes **two** things at
once: the array stops being reconfigured per dispatch, *and* the ABI changes
from ELF to xclbin — different kernel objects (`xrt.ext.kernel` against
`xrt.kernel`), an explicit instruction BO, different BO types, extra launch
arguments. This document itself reaches for the ABI difference a few paragraphs
down to explain the best-case latency regression, so it cannot also treat the
ABI as inert here. The `runlist` control rules out *environmental* drift; it
says nothing about which half of a two-variable intervention did the work.

So: eviction remains the leading candidate — it is the only per-dispatch
host/driver work in the mode, and host-side work is exactly what host conditions
perturb — but "`_evict_context` is the mechanism" is **not** what these four
walks show. [27](27-common-ladder-result.md)'s hypothesis is **supported, not
confirmed.**

**The isolating experiment, unclaimed:** hold the ABI fixed and vary only the
eviction. `probe_context_reuse.py` already establishes that `xclbin`+`[2,2]` and
`xclbin`+`[1,1]` are both bit-identical over four runs, so the xclbin ABI can
safely run *with* eviction forced on — that arm plus the two already walked
separates the variables. It needs a knob to force eviction on the shared path,
which does not exist today.

**Two further qualifications, both of which cut against over-reading the size:**

- **The shared path does not quite reach the control's band at 512** — 14–18%
  against the control's 5.5–8.1%. A large collapse, not a total one.
- **The 1024 rung did not reproduce its own baseline.** 27 recorded 61.6% /
  59.8% there on the ELF path; today the same path read 9.0% / 10.5%, so at 1024
  both arms are already in band and the rung is uninformative. The ELF baseline
  is itself unstable day to day, which is a caution about the effect *size* —
  not about its existence, which 512 establishes twice.

**A provenance check that the ELF arm is the same thing 27 measured:** its
minimums reproduce 27's almost exactly — 82.0 / 78.9 ms at 512 here against
27's 78.2 / 79.9 — while its averages are inflated by the outliers, which is
what 27 saw too (its walk-2 average was 111.4 against a 79.9 minimum).

### The cost the collapse comes with, which no one predicted

| seq | ELF avg / min (w1, w2) | shared avg / min (w1, w2) |
|---|---|---|
| 512 | 163.5/**82.0**, 111.5/**78.9** | 103.7/**97.5**, 105.5/**99.5** |
| 1024 | 176.6/168.7, 165.6/159.6 | 164.1/160.1, 164.4/161.5 |

**The shared path's best case at 512 is ~20% WORSE** — 97.5–99.5 ms against
78.9–82.0 — while its average is far better because it has no tail.
[23 §1](23-rules-and-open-items.md) settled that minimums, not medians, are the
truer measure here and that host jitter flatters the median; on that convention
the shared path is *slower* at 512 and level at 1024. So this is a trade —
variance for best-case latency — and not the free win the reconfiguration
counter alone suggests. **Not** grounds to leave the mode on a path that
contradicts its own definition, but it is the reason the default was not flipped
in the same change as the gate.

**It is reproduced but not yet EXPLAINED, and the tell is that it does not scale.**
A fixed per-submission cost of the xclbin ABI would fit: 30 submissions × ~0.7 ms
covers the whole 512 gap, and the same absolute cost disappears into 1024's much
larger total — which is exactly the shape observed. The competing reading is that
it is specific to the small shape. Nothing in hand separates them.

**What would separate them, and why it cannot be read off today's artifacts:**
`prepare_offload` already measures `device_ms`, `sync_ms` and `host_cpu_ms` per
run and returns all three in its `extra` dict — and `run_mode.py` reads none of
them, while schema v1 has **no column for any of them**. So the decomposition is
computed and thrown away on every rung the study has ever walked. Unlike
`npu_unique_xclbin_count`, which was already a v1 column waiting to be filled,
adding these is a **schema version bump** (`schema.py:53`), which is why it is
recorded here rather than done in passing. With them persisted, the ~17–21 ms is
attributable to submission, sync or host arithmetic in one walk. A cheaper
independent probe: one GEMM, warmed, over ELF / standalone xclbin / merged
xclbin — which separates the ABI from the context sharing, since `load` takes
`xrt.ext.kernel` with embedded instructions for ELF and `xrt.kernel` plus an
explicit instruction BO and three extra launch arguments for xclbin.

Artifacts: `results/offload-ctx-{elf,shared}-w{1,2}/` (gitignored, schema v1).
The arms are distinguishable in the CSVs themselves —
`npu_unique_xclbin_count` reads **1** on the shared arm and **0** on the ELF one
— zero because that path loads `.elf` artifacts through `xrt.elf()` and loads no
xclbin at all; it is counted off the artifacts the run actually loaded rather
than inferred from the setting. **It is not the reconfiguration count.** That is
1 against 30, it is the mode's real axis, and schema v1 has nowhere to put it —
adding a column for it is the same version bump §The cost the collapse comes
with asks for. `bytes_transferred` is byte-identical between the two arms at
both lengths.

## ~~Known gap — the shared path is NOT gated~~ CLOSED `[2026-08-09]`

> The gap as it stood: nothing in the lit suite exercised
> `AIR_OFFLOAD_SHARED_XCLBIN=1`, and `run_npu2_offload_peano.lit` pinned the
> `reconfiguration:` line on **neither** path, so a regression to 30 context
> loads passed green. The mode's central claim was printed, not enforced.

`run_npu2_offload_peano.lit` now has a **third recipe** and pins the counters on
both paths:

| path | pinned |
|---|---|
| ELF (clean half, 4096) | `context_loads 30 kernel_attaches 0 over 30 dispatches` |
| shared (new recipe, 1024) | `context_loads 1 kernel_attaches 4 over 30 dispatches` |

plus, on the shared recipe, the mode's own "ONE xclbin over 5 shapes" line, three
stage comparisons at zero mismatches, `stages: 10/10 clean`, the full dispatch
vector, and the run's `passed` verdict. `make check-offload-shared` is the
target. **Transformer-layer suite 28/28 on NPU2** (494.5 s), study host tests
84/84, dispatch/seam unit tests 31/31, `phase_e_checks` selftest 30/30.

**One thing the gate had to learn from the run rather than from the design:**
under the xclbin ABI the dispatch vector is only at its steady-state value
**after a warmup**. The first call uploads each artifact's **instruction
stream** once — `sync_instruction_bos`, which the ELF ABI skips entirely
because an ELF embeds its instructions — so the cold call reads
`sync 95 bytes 99141520` and every later one reads `sync 90 bytes 99090432`,
which is what [27](27-common-ladder-result.md)'s ladder records. Five artifacts
is five extra sync boundaries and exactly **51,088 bytes**, the total size of
the five cached `.insts.bin` files; the ELF path reads `sync 90` on its first
dispatch and on every one after. So the target dispatches twice, both totals
lines are pinned, and **anything reading a dispatch vector from a single cold
dispatch under this ABI is reading an inflated one**. The reconfiguration
counters are unaffected either way, which is what a per-layer array
configuration should look like.

**Verified in the failing direction**, which is the only thing that makes it a
gate. The shared prefix run against the *same mode at the same sequence length
over the ELF packaging* — i.e. exactly the regression it exists to catch —
fails, and fails on the reconfiguration line rather than incidentally:

```
// SHARED: reconfiguration: ONE xclbin over 5 shapes, ...
                 X error: no match found
    5:  reconfiguration: 5 xclbins, 30 hw_context loads for 30 dispatches
```

### The 4096 wall the gate had to route around, and its exact cause

**The chain does not build at 4096, so the recipe gates at 1024.** This is worth
more than the workaround, because it bounds the mechanism this document landed:

- At 4096 the down-projection resolves to **`fused-cast`, which is two
  `air.launch` ops**. Every other shape at either length is single-launch
  `drain`. Measured, not inferred: `air_launches=2 herd=4` for
  `off_gemm_4096x3072x768`, `1`/`3` for all nine others.
- `XRTBackend.compile` defaults `insts="air.insts.bin"` — a **fixed** name — and
  the xclbin branch passes it through as `-i`. The ELF branch passes
  `--elf-name` and **no `-i` at all** (`python/air/backend/xrt.py:307-316`), so
  only the ELF path lets aircc derive a name per launch.
- A two-launch module under the xclbin ABI therefore asks aiecc to write every
  instruction stream to one path, and it refuses:
  `aiecc: edge 'air.insts.bin' produced duplicate output path './air.insts.bin'`.

**So the shared path is bounded to SINGLE-LAUNCH modules today**, and 1024 is
where all five of this mode's shapes are single-launch. The memtile
"Failed to allocate buffer" lines in that build log are warnings and a red
herring; the failure is the duplicate output path.

**This also dates the original landing claim.** Everything in this document was
demonstrated at 1024 — the `bytes 99090432` above is a 1024 figure — and nothing
here said so, while the mode's own gate runs at 4096. That is the README's
"a fixture proves only the shape it runs", one more time.

**Unclaimed. The FIRST blocker is located; the whole fix is not yet scoped.**
Parameterizing the instruction-file name per launch on the xclbin path clears
the compile error, and that much is precise. It is **not** established that it
is sufficient: `XRTCompileArtifact` carries a **single** `insts` string
(`xrt.py:35-54`) and the xclbin load path checks and reads exactly one file
(`xrt.py:629-636`), so a module that emits several instruction streams may also
need artifact-shape and runtime changes — and `sync_instruction_bos` uploads one
stream per artifact identity. Treat "rename the file per launch" as the next
experiment, not as the plan.

Either way it is a change to `python/air/backend/xrt.py`, which every shipped
model loads, so it needs an `install-xrt` rebuild and the ten-model regression —
its own phase, not a patch.

**Do NOT implement it by simply dropping `-i`.** The obvious symmetry with the
ELF branch is a trap: `output_format` defaults to `xclbin`, and the same
`else` branch also serves `txn`, while `pdi` passes `-i` on its own line
(`xrt.py:307-316`). Removing the flag would change the artifact contract for
every xclbin and txn caller in the tree. Scope the change to per-launch naming
within the xclbin case, and give it backend-level tests — a two-launch module
compiled to xclbin is the fixture, and it does not exist today, which is why
this shipped broken.

#### `[2026-08-10]` The wall is down at COMPILE time — and the rename was a third of the fix

The five-shape chain now **builds at 4096**: one shared xclbin holding all
five kernels (`matmul_bf16_{proj,up,attn_scores,attn_output}` +
`gemm_cast_bf16_down`), verified by dumping the final `AIE_PARTITION` — seven
PDIs, five kernel-owning. The change is confined to
`XRTBackend.compile`/`_finalize_multi_launch_xclbin` in
`python/air/backend/xrt.py`, takes effect only when the module has more than
one `air.launch` AND `output_format == "xclbin"`, and needed **three** pieces,
of which the doc's predicted rename was only the first:

1. **The rename is a `{0}` template, not a name.** aiecc's `--npu-insts-name`
   and `--xclbin-name` accept a per-device `{0}` substitution; the fixed name
   collides because a two-launch module reaches aiecc as **three**
   `aie.device` ops — one per launch plus a `main` orchestration device whose
   runtime sequence inlines every launch's DMA program with an
   `aiex.npu.load_pdi` between them. That main device IS the ELF path's
   combined-stream mechanism, already generated in xclbin mode; the xclbin
   path just discarded it.
2. **The artifact is the main device's pair, repackaged.** One kernel, one
   instruction stream, one xclbin — `XRTCompileArtifact`, `load()`,
   `attach_kernel`, `sync_instruction_bos` and the dispatch loop are all
   untouched, because the multi-launch execution lives INSIDE the one stream
   exactly as it does inside an ELF.
3. **aiecc leaves the main xclbin unexecutable, so the backend finishes the
   packaging.** The main partition as emitted holds only the (empty) main PDI
   while the stream's `load_pdi` words reference the per-launch PDIs by ids no
   partition entry carries — and those ids restart at 1 per compile, so two
   chained multi-launch kernels would collide, and single-launch links already
   sit at `pdi_id 0x1`. The backend walks the stream (header + per-launch
   bodies, each byte-identical to that device's own stream), renumbers the ids
   off the link's `kernel_id` (`0x903 → 0x9031, 0x9032`), patches the words,
   and merges the per-launch PDIs into the partition as **kernel-less** entries
   via `xclbinutil --add-replace-section`. Every layout assumption raises with
   evidence when violated. aiecc's own `--xclbin-input` merge on the two
   subsequent chain links accepted the kernel-less entries.

The fixture the last paragraph asked for exists:
`test/xrt/56_multi_launch_xclbin_compile` (compile-only, two-launch, checks
the partition/stream/id agreement). Verified in the failing direction: on the
unpatched backend it dies on the duplicate-output-path error this section
documents.

**What this does NOT establish: hardware.** The artifact executes in-stream
`load_pdi` (opcode `0x8`) against partition-resident PDIs — aiecc's default
lowering for multi-device xclbin, but never yet dispatched by this tree. The
alternative if hardware balks: `--expand-load-pdis` also completes in xclbin
mode (measured: the main stream grows 19 KB → 174 KB, embedding the config as
writes — the exact content the ELF gate validates at 4096), but aircc only
passes that flag on its ELF branch, so it needs an aircc change. **`[2026-08-11]`
Both halves of this fallback sentence are now falsified — see the hardware
verdict below.** Both paths still need everything in item 3. The
install-rebuild + ten-model regression phase this section already scoped is
unchanged, and the shared recipe's 1024 pin can only move to 4096 after the
hardware gate passes.

### `[2026-08-11]` The hardware verdict: load_pdi faults, and the scoped fallback does not exist

**The dispatch experiment ran.** The five-shape chain compiled at 4096 into one
shared xclbin (the two-launch `fused-cast` down-projection included, packaged
by the multi-launch backend), and the first-ever multi-launch xclbin dispatch
in this tree was attempted on NPU2 (XRT 2.21.0, firmware 1.1.2.64, Turbo
pmode). **Twenty-nine single-launch dispatches executed clean off the shared
xclbin at 4096** — the packaging, chaining and kernel routing all work — and
the ONE multi-launch module faulted the firmware on its first submission:
`off_gemm_4096x3072x768`, `ERT_CMD_STATE_TIMEOUT` with `fatal_error_type
0x10`, `fatal_error_exception_pc 0x161AD`, `txn_op_idx 0xFFFFFFFF` (a context
firmware exception, not a slow kernel). **In-stream `load_pdi` against
partition-resident PDIs does not execute on this platform.**

**The fallback as scoped is not implementable, and its measurement claim has
no artifact.** `--expand-load-pdis` is a no-op on mlir-aie 1.4.0's `aiecc` for
BOTH output edges, established by three probes on the two-launch fixture
module: (a) an aircc passthrough, verified present on the echoed aiecc command
line, leaves the main stream byte-identical (4,800 B, leading word
`0x00020008` — a load_pdi); (b) requesting `--get-full-elf` alongside the
xclbin/insts edges changes nothing; (c) pure ELF mode emits the same 4,800 B
`.ctrltext.0`. The ELF ABI's multi-launch support never came from expansion:
`aiebu` embeds per-sequence `.pdi.N` sections next to each `.ctrltext.N` and
the ELF loader resolves reconfiguration from those — packaging the raw-insts
xclbin path cannot use. The "19 KB → 174 KB" figure above matches no expanded
main stream: the gated 4096 down-projection ELF's 176 KB section
(`.ctrltext.2`) is a *launch* DMA program, and its main sequence
(`.ctrltext.0`) is 2,288 B with no load_pdi-shaped words on the 16-byte grid.

**Routes that remain, none of them the one-line aircc change predicted above:**

1. **Upstream mlir-aie**: wire expansion (`aie-materialize-runtime-sequences` +
   `aie-expand-load-pdi`, both present in `aie-opt`) into aiecc's npu-insts
   edge. A wheel rebuild — a toolchain change with doc-15 blast radius.
2. **The ctrl-packet flow** (`--load-pdi-to-ctrl-pkt` + control overlay):
   exists in this aiecc, but changes routing and is mutually exclusive with
   expansion; its overlay cost on an 8-column chain is unscoped.
3. **Mode-level, no compiler change**: force a single-launch method for the
   4096 down-projection in the shared chain (`drain` instead of `fused-cast`),
   sidestepping multi-launch entirely. Numerics must re-gate at that method;
   the registry precedent for not-this-shape's-fastest methods exists.
4. **Split-chain, no compiler change**: shared xclbin over the four
   single-launch shapes plus the gated ELF artifact for the down-projection on
   a second standing context — `context_loads 2` instead of 30, preserving
   "reconfiguration minimized" to within one extra load.

**`[2026-08-11]` Route 3 taken, and the shared path is now GATED AT 4096.**
`offload_config` re-resolves a `fused-cast` tier winner to the shape's
measured `drain` row under the shared path only (`_chain_spec`; the ELF path
keeps the winner, and a shape with no drain row raises through the registry —
the correct refusal). The pin is priced in the run log — `down 4096x3072x768
drain (registry, pinned over fused-cast)`, 6,226 against 6,927 GFLOP/s on
that GEMM (~10%) — and ENFORCED by the recipe, which fails if the pin is lost
or a retune changes the chain.

Measured at 4096, then gated: **10/10 stages clean on both dispatches, and
`context_loads 1 kernel_attaches 4` over 30 dispatches** — the array
configured once at the mode's own spec shape. The byte provenance holds to
the byte twice over: the cold−steady delta is 293,200 = the five cached
instruction streams exactly (the same identity read 51,088 at 1024), and
against the ELF clean half the steady totals differ by exactly 12,582,912 =
4096 × 768 × 4 — the `fused-cast` module's f32 C scratch, which the pinned
`drain` module does not stage — plus one launch's worth of air/herd/sync
(31/91/91 → 30/90/90). `run_npu2_offload_peano.lit` re-ran green with all
three recipes at 4096 (lit 1/1, 95.9 s, 2026-08-11). The single-sample
latency (989.9 ms avg, Turbo) is a smoke figure, not a record — walk it
before citing it.

Item 3's flip is now decidable at the canonical shape. Routes 1/2/4 stay
scoped above in case a future shape has no single-launch row to pin to.

## What this does not do

- ~~**It does not make the shared path the default.**~~ **`[2026-08-11]` It
  does now — flipped; see the dated close-out at the end of this bullet.** As
  it stood: `AIR_OFFLOAD_SHARED_XCLBIN=1`
  opts in, and the gated ELF path is untouched, with its own cache directory —
  the two builds produce artifacts with identical NAMES over different ABIs, so a
  shared directory could trade them and credit a 30-reconfiguration run to the
  one-reconfiguration claim. `[2026-08-09]` **Still true after the variance
  measurement, and now a decision rather than an omission**: the shared path
  wins decisively on variance and loses ~20% on best-case latency at 512, and
  flipping the default would additionally invalidate every recorded `offload`
  number — [27](27-common-ladder-result.md)'s four-mode table included — since
  they all describe the ELF path. That re-walk is the cost of the flip and it
  belongs to whoever takes it.

  **The decision, and the order it fixes.** Reviewed independently and settled
  the same way: **fix the 4096 wall first, then flip.** Defaulting to a path
  that cannot build at the mode's own gated shape would be worse than keeping a
  default that is explicitly temporary — and once 4096 builds, the flip should
  be decided on the canonical shape rather than extrapolated from 512, with the
  ELF path renamed to what it then is, a legacy/control packaging. The argument
  that survives the latency objection: **latency should not decide which
  implementation gets to be called `offload`.** The taxonomy defines the mode by
  minimized reconfiguration; a default that maximizes it contradicts the
  definition, and a small-shape best-case win does not settle that.

  **`[2026-08-11]` FLIPPED, per this decision, the day its precondition was
  met.** The shared path gates at 4096 (§The hardware verdict above), so the
  default moved in the same tree state: an unconfigured run now takes the
  shared xclbin, and the ELF path is renamed to what it now is — the
  **legacy/control** packaging, opt-in via `AIR_OFFLOAD_LEGACY_ELF=1`.
  `AIR_OFFLOAD_SHARED_XCLBIN` is retired and **RAISES** if set in any form:
  `=1` would be redundant today, `=0` would silently run the OPPOSITE
  packaging to the one a stale script was written to measure, and a silently
  wrong packaging is exactly the failure class this document records — so per
  its own rule, the default has to be an error. The cache directories did
  NOT move: `offload_shared_cache` stays the shared path's own and
  `offload_cache` the legacy path's, so the identical-names-over-different-ABIs
  trade this bullet describes remains impossible. The lit recipes flipped
  which side sets the env var and nothing else: the shared recipe (now the
  default, **no** env var) still pins `context_loads 1 kernel_attaches 4`,
  the legacy recipe (`AIR_OFFLOAD_LEGACY_ELF=1`, set inside its Makefile
  targets, the fault twin included — at 4096 the two packagings' totals
  differ, so the fault twin must ride the same arm as its clean half) still
  pins `context_loads 30 kernel_attaches 0`, and every dispatch-total
  literal is unchanged. What the flip re-dates: **EVERY recorded `offload`
  latency/variance number — [27](27-common-ladder-result.md)'s four-mode
  table included — predates it and describes the ELF path.** The four-mode
  re-walk is the cost this section priced; it is now owed, and it is a
  separate hardware task, not part of the flip change.
- **It does not deliver runtime-parameterized loop bounds.** That is the increment
  *beyond* iron parity, deferred by [03](03-measurement-model.md) and still
  blocked in the stack ([26 §A](26-mode-rebuild-feasibility.md)).

## Verification

`E1 GATE: PASS` in 4254 s — transformer-layer lit suite **28/28 on NPU2**
(including `run_npu2_offload_peano.lit`, 150.9 s of real dispatch) and **all ten
shipped models still verify**. That is the standing regression clause for
anything touching `llms/shared/`, which this does, plus the installed `air`
backend. Dispatch unit tests 31/31, study host tests 84/84, `phase_e_checks`
selftest 30/30.
