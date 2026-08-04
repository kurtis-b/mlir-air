# 05a — Phase B spike result: the multi-ELF runlist works, but not the way 05-phase-b proposed

[05-phase-b-runtime-seam.md](05-phase-b-runtime-seam.md) calls this the plan's load-bearing
assumption and asks for it to be spiked before anything else starts. It was, twice.

**The mechanism 05-phase-b proposed does not work. The capability it wanted does.** Several
separately-compiled AIR ELFs cannot be bound into one `hw_context` — XRT rejects that three
different ways, recorded in §§1–3 below. They do not need to be. `xrt::runlist` dispatches each
entry against the context its *kernel* came from, so N ELFs means N `hw_context`s and still one
runlist, and that is bit-identical to sequential dispatch and measurably faster (§5).

The first pass of this spike tested only the shape 05-phase-b wrote down — everything into one
context — found that every route to it fails, and concluded the multi-ELF runlist was impossible.
That inference was wrong: it assumed runlist entries must share a context, which XRT does not
require. §5 is the correction and `make runlist-gate` leg A is the standing measurement.

| | |
|---|---|
| Measured on | NPU2 (`amdxdna` 2.21.0_20260514, firmware 1.1.2.64) |
| XRT | 2.21.0, hash `4eb1f4392a012b4e6eca759762389c612537f7c7` |
| Artifacts | Separately-compiled AIR GEMM ELFs from `shared/builders/gemm_builder.py`, registry tiles |
| Reproduce | `make runlist-gate` in `programming_examples/transformer_layer/` |

## The claim that was tested

05-phase-b §"The resolution" proposed:

```python
ctx  = pyxrt.hw_context(device, pyxrt.elf(first_elf))
mod  = pyxrt.module(pyxrt.elf(path))
pyxrt.module(mod, ctx)                       # bind each further ELF into it
kern = pyxrt.ext.kernel(ctx, mod, kernel_name)
rl   = pyxrt.runlist(ctx); rl.add(pyxrt.run(kern)); ...
```

The phase document already flagged this as "API-shape evidence, not hardware evidence". The
overload signatures are real; the combination is not legal. §§1–3 are why.

## What does not work: one context, several ELFs

### 1. A full ELF cannot be rebound into another context

`xrt::module(parent, hwctx)` throws `Invalid instruction buffer size` — for an ELF-created
context *and* for an xclbin-created one, and for the very ELF the context was created from.
AIR ELFs report `xrt::elf::is_full_elf() == true`: they carry `.pdi.*` array configuration and
per-kernel `.ctrltext.N` in COMDAT groups, so they have no separable instruction buffer for that
overload to move.

### 2. The two `ext::kernel` overloads are mutually exclusive

| Call | XRT's requirement |
|---|---|
| `ext::kernel(ctx, module, name)` | ctx from **XCLBIN**, module **not** from a full ELF |
| `ext::kernel(ctx, name)` | ctx from a **full ELF** |

Passing a full-ELF context to the 3-arg form fails with *"xrt::hw_context passed is not created
using XCLBIN"*; passing an xclbin context to the 2-arg form fails with *"xrt::hw_context passed is
not created using full ELF"*. Asking a full-ELF context for another ELF's kernel by name fails
with *"Unable to find group idx for given kernel"*. There is no combination that puts a second
full ELF's kernel into an existing context.

### 3. `hw_context::add_config()` accepts exactly one ELF

XRT 2.21 has `xrt::hw_context::add_config(const xrt::elf&)`, documented as adding a config ELF
when "configuration matches with existing one". It is **not bound in pyxrt**, so it was tested
from C++ directly. Result:

- Empty-QoS context, `add_config(elf_a)` → OK. `add_config(elf_b)` → `kernel already exists,
  cannot use this ELF with this hw ctx`.
- `add_config(elf_a)` **twice** → the second call fails with the *same* message.

The rejection is not a name collision. It was retested with every user-visible symbol made
distinct — entry function, AIR segment, and the generated `_Z4main...` arity (3-arg drain GEMM
vs 4-arg fused-cast GEMM) — and with the identical ELF. One config ELF per context is the limit.

### 4. Sharing one *xclbin* context across artifacts runs, and is silently wrong

The xclbin ABI *does* let one `hw_context` submit a runlist of runs carrying different
instruction BOs. It executes without error. But the array configuration comes from the xclbin,
not the instruction stream, so only the artifact that supplied the context is correct:

```
xgemm_512x512x512  : runlist == sequential -> True
xgemm_1024x1024x1024: runlist == sequential -> False
```

This is the dangerous failure mode: no exception, no timeout, wrong numbers. It is why
`plan_submissions` still splits at every artifact change **under the xclbin ABI** and raises
`RunlistSplitError` rather than emitting such a runlist. §5 does not apply to the xclbin path and
does not weaken this rule: there, the configuration lives in the xclbin behind the context, so
the entry cannot bring its own.

## 5. What does work: N contexts, one runlist

A `runlist` is constructed *against* a context, but it is not restricted *to* it. Each entry is
an `xrt::run` over a kernel that already carries its own context, and XRT dispatches it there:

```python
ctx_a = pyxrt.hw_context(dev, pyxrt.elf(elf_a))   # one context per ELF
ctx_b = pyxrt.hw_context(dev, pyxrt.elf(elf_b))
kern_a, kern_b = pyxrt.ext.kernel(ctx_a, name_a), pyxrt.ext.kernel(ctx_b, name_b)

rl = pyxrt.runlist(ctx_a)                          # any one of the contexts
rl.add(run_of(kern_a)); rl.add(run_of(kern_b))     # entries from both
rl.execute(); rl.wait()
```

Measured on the study's own artifacts — three separately-compiled fused-cast GEMM ELFs at the
Llama-3.2-1B seq_len-2048 projection shapes (2048×2048×2048, 2048×2048×8192, 2048×8192×2048):

- **Bit-identical to sequential dispatch**, for every entry, with the output BOs filled with
  `0xA5` first so a skipped entry cannot pass by leaving the baseline's bytes in place.
- **Independent of ordering**: `ABC`, `CBA` and `BAC` all match.
- **Independent of which context hosts the runlist**: building it on A's, B's or C's context all
  match, including a context whose ELF is not among the entries' first.
- **Repeatable**: one runlist object executed four times, correct each time.
- Three full-ELF `hw_context`s live simultaneously on one device is fine. The study's `runlist`
  mode wants 29; that number has not been probed and is the remaining scaling question.

Latency, interleaved medians (alternating sequential and aggregated, because the effect is
smaller than thermal drift across two back-to-back blocks):

| Entries | sequential | runlist | | runlist wins |
|---|---|---|---|---|
| 3 distinct ELFs (q, gate, down), 25 pairs | 20.826 ms | 19.949 ms | 1.044× | 25/25 |
| 3 distinct ELFs, gate run, 15 pairs | 20.236 ms | 19.770 ms | 1.024× | 15/15 |
| 2 entries on one ELF (gate + up), 15 pairs | 19.591 ms | 19.056 ms | 1.028× | 14/15 |
| 512×512×512 GEMM ×2, one ELF | 0.888 ms | 0.774 ms | 1.15× | |

The saving is one host submission — a fixed cost of order 100 µs — so it is large relative to
small kernels and small relative to large ones. That ratio *is* the axis the study wants to
measure: submission cost matters exactly to the degree that the work per submission is small.
The win count is reported alongside the median because at these sizes a median that moved by less
than the sample spread would not be a measurement.

### 6. Unrelated defect found en route: the standalone drain GEMM ELF is single-shot

Not caused by this port and not a Phase B change, but it was found by anchoring the gate's
baseline against an FP32 oracle, and it will mislead anyone who compiles a GEMM the way the
study's `offload` mode does.

A `drain`-method GEMM compiled to its own ELF (registry shape 2048×2048×512) is correct on the
**first** invocation after load and wrong on every one after it:

```
same bo_key, call 0: 100.00% of elements within rtol=1.6e-2 atol=1.5e-3
same bo_key, call 1:  10.63%
same bo_key, call 2:  10.52%
new bo_key  k0:       10.68%
```

It is not a buffer-reuse problem: a fresh BO set is equally wrong. The `fused-cast` method at
the same seq_len is correct across arbitrarily many calls and BO sets, which is why the shipped
deployments do not hit this — they reach drain GEMMs only inside fused multi-launch ELFs, never
as a standalone artifact.

Two things follow. The study's `offload` mode cannot use standalone drain ELFs until this is
fixed, and the registry's `best.high = drain` shapes are the ones affected. `runlist_gate.py`
leg D re-measures it on every run so the finding cannot go stale.

Separately: omitting `runtime_loop_tiling_sizes=[2, 2]` and `stack_size=2048` from a standalone
GEMM ELF's backend kwargs produces an artifact that compiles, loads, runs, and returns different
numbers on every call. Both the example's own runner and the shipped `O_FFN_BACKEND` preset set
them; they are not optional.

## Consequences for the plan

Per 05-phase-b: *"If this fails, `runlist` and `coarse` collapse into `offload` and the study
loses its central axis."* It does not fail. Concretely:

- **`runlist` (29 kernels, 42 entries) and `coarse` (5–6 kernels, 12 entries) as specified in
  [01-port-inventory.md](01-port-inventory.md) can be built from separately-compiled ELFs**, one
  `hw_context` per artifact. The taxonomy in [03-measurement-model.md](03-measurement-model.md)
  keeps four distinct points, reached as originally intended.
- The open question is **how many concurrent `hw_context`s NPU2 grants**. Three is measured; 29
  is not. If the device runs out it says so — `xrt.hw_context` raises at load time — so the
  failure would be an exception during `ensure_loaded`, not a quietly wrong number. Reaching 29
  would then need the sequence broken into groups, and the dispatch vector would report the
  resulting submission count honestly.
- One xclbin merge (iron's `aiecc --xclbin-input`) is still a non-goal, and is now also
  unnecessary. Route 2 below is still the only way to reconfigure *within* one context.

### Routes that remain relevant

1. **One xclbin containing every design** — iron's `aiecc --xclbin-input` incremental merge.
   Explicitly a non-goal in [00-context-and-goals.md](00-context-and-goals.md), dropped in the
   port inventory, and no longer needed for aggregation.
2. **Control-packet reconfiguration between entries.** `test/xrt/24_ctrlpkt_config_2gemms_4x4`
   proves the mechanism on this repository: one base xclbin, one kernel, and alternating
   *(reconfigure, run)* pairs in a single runlist. Still the only route that puts several designs
   through *one* context, which matters if the concurrent-context limit turns out to bind. aiecc
   exposes the parts (`--generate-ctrl-pkt-overlay`, `--ctrlpkt-elf-name`,
   `--load-pdi-to-ctrl-pkt`), but `XRTBackend` exposes none of them and `aircc` does not drive
   them. A new compilation path, not a runtime seam.
3. **One ELF holding several `aie.device`s switched by `load_pdi`.** A single `aircc` invocation,
   so the artifacts are not separately compiled — the `fused_elf` point of the taxonomy reached
   from a different direction.

## What Phase B shipped

- `shared/infra/dispatch.py` aggregates a dispatch sequence into runlists. Under the ELF ABI the
  whole sequence is one submission whatever artifacts it spans; under the xclbin ABI it splits at
  every artifact change and **raises** `RunlistSplitError` rather than emitting the runlist §4
  measured to be silently wrong.
- Each entry's run is built from *its own* artifact's kernel. Taking the submission's first
  kernel for every entry would execute the wrong program with the right buffers.
- The dispatch vector records the *true* `host_submissions_per_layer`, so a sequence that had to
  be split reports the split honestly instead of claiming one submission.
- BO pooling, dirty-bit sync and the dispatch vector are independent of all of this and are
  complete. The buffer rules are in [05b-phase-b-buffer-rules.md](05b-phase-b-buffer-rules.md).
  Measured on the same five-GEMM layer, with the B operands declared static as a real layer's
  weights are: 18 declared buffers land on 15 pool slots, the whole layer runs as **one**
  submission, and the second dispatch of the same sequence is bit-identical to the first while
  moving **117 MB less** — 10 sync boundaries instead of 15, because the weights are already
  resident. That reuse depends on pools being keyed by sequence value rather than by plan object
  identity (O5); keyed the other way the second pass re-uploads all 117 MB and nothing fails
  loudly enough to notice.

`make runlist-gate` measures every claim above on every run: leg A the cross-artifact runlist,
leg A2 the xclbin refusal, leg B within-artifact aggregation, leg C the whole layer through
`KernelCache.run_sequence` in one submission, leg D the drain defect.
