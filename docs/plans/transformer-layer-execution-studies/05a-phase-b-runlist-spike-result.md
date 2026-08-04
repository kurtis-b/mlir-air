# 05a — Phase B spike result: the multi-ELF runlist does not work on NPU2

[05-phase-b-runtime-seam.md](05-phase-b-runtime-seam.md) calls this the plan's load-bearing
assumption and asks for it to be spiked before anything else starts. It was. **It fails.**

This document records what was measured, so the finding is citable and nobody re-derives it.

| | |
|---|---|
| Measured on | NPU2 (`amdxdna` 2.21.0_20260514, firmware 1.1.2.64) |
| XRT | 2.21.0, hash `4eb1f4392a012b4e6eca759762389c612537f7c7` |
| Artifacts | Two separately-compiled AIR GEMM ELFs from `shared/builders/gemm_builder.py`, registry tiles |
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
overload signatures are real; the combination is not legal.

## What actually happens

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

Passing a full-ELF module to the 3-arg form fails with *"xrt::module passed is created using
full ELF"*; passing an xclbin context to the 2-arg form fails with *"xrt::hw_context passed is
not created using full ELF"*. There is no combination that puts a second full ELF's kernel into
an existing context.

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

### 4. Sharing one xclbin context across artifacts runs, and is silently wrong

The xclbin ABI *does* let one `hw_context` submit a runlist of runs carrying different
instruction BOs. It executes without error. But the array configuration comes from the xclbin,
not the instruction stream, so only the artifact that supplied the context is correct:

```
xgemm_512x512x512  : runlist == sequential -> True
xgemm_1024x1024x1024: runlist == sequential -> False
```

This is the dangerous failure mode: no exception, no timeout, wrong numbers. Any aggregation
implementation must refuse to build such a runlist rather than emit it.

### 5. What does work: aggregation inside one artifact's context

A runlist over several runs of one artifact's kernel is bit-identical to sequential dispatch and
measurably faster. The saving is one host submission — a fixed cost of order 100 µs — so it is
large relative to small kernels and small relative to large ones:

| Pair | sequential (2 submissions) | runlist (1 submission) | |
|---|---|---|---|
| 512×512×512 GEMM ×2 | 0.888 ms | 0.774 ms | 1.15× |
| 2048×2048×8192 GEMM ×2 (gate + up) | 19.37 ms | 19.04 ms | 1.02× |

That ratio *is* the axis the study wants to measure: submission cost matters exactly to the
degree that the work per submission is small. It also means a latency comparison on large
kernels needs interleaved medians rather than two back-to-back blocks — thermal drift over a
20 ms×15 benchmark is larger than a 300 µs effect. `runlist_gate.py` does that.

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
loses its central axis."*

Concretely:

- **`runlist` (29 kernels, 42 entries) and `coarse` (5–6 kernels, 12 entries) as specified in
  [01-port-inventory.md](01-port-inventory.md) cannot be built from separately-compiled ELFs.**
  Their entries span artifacts, and artifacts cannot share a context.
- `offload` and `fused_elf` are unaffected — both are already single-artifact-per-submission.
- The taxonomy in [03-measurement-model.md](03-measurement-model.md) still has four distinct
  points, but the middle two must be reached differently.

### The three remaining routes, and why none is a Phase B change

1. **One xclbin containing every design** — iron's `aiecc --xclbin-input` incremental merge.
   Explicitly a non-goal in [00-context-and-goals.md](00-context-and-goals.md) and dropped in
   the port inventory.
2. **Control-packet reconfiguration between entries.** `test/xrt/24_ctrlpkt_config_2gemms_4x4`
   proves the mechanism on this repository: one base xclbin, one kernel, and alternating
   *(reconfigure, run)* pairs in a single runlist. aiecc exposes the parts
   (`--generate-ctrl-pkt-overlay`, `--ctrlpkt-elf-name`, `--load-pdi-to-ctrl-pkt`), but
   `XRTBackend` exposes none of them and `aircc` does not drive them. This is a new compilation
   path, not a runtime seam.
3. **One ELF holding several `aie.device`s switched by `load_pdi`.** aiecc's `--elf-name` takes a
   `{0}` multi-device template and `--expand-load-pdis` exists to avoid full PDI reloads. This is
   a single `aircc` invocation, so the artifacts are not separately compiled — it is the
   `fused_elf` point of the taxonomy reached from a different direction.

Route 2 is the one that preserves the study's axis. It needs a decision and a phase of its own.

## What Phase B shipped instead

The runtime seam was built to the constraint rather than around it:

- `shared/infra/dispatch.py` aggregates a dispatch sequence into runlists, submits one runlist
  per artifact context, and **raises** `RunlistSplitError` rather than emitting a runlist whose
  entries span configurations (failure mode 4 above).
- The dispatch vector records the *true* `host_submissions_per_layer`, so a sequence that had to
  be split reports the split honestly instead of claiming one submission.
- BO pooling, dirty-bit sync and the dispatch vector are independent of this blocker and are
  complete.

The gate's numerical-identity and lower-latency requirements are met for aggregation within an
artifact and are **not** met across artifacts. `make runlist-gate` reports both, and fails on
the cross-artifact leg rather than reporting a pass.
