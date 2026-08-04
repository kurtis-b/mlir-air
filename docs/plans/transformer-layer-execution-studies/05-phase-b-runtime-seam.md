# 05 — Phase B: Runtime Seam

Two host-side additions: runlist aggregation, and buffer-object liveness pooling.

**This phase contains the plan's load-bearing assumption. Spike it before committing to
anything else.**

> **The spike was run. The assumption holds; the mechanism §"The resolution" proposes does
> not.** An AIR ELF is a *full* ELF carrying its own array configuration, and a `hw_context`
> accepts exactly one of those — so the `pyxrt.module(mod, ctx)` rebinding below is rejected by
> XRT 2.21.0 / NPU2, three different ways. It is also unnecessary: a runlist is constructed
> *against* a context but is not restricted *to* it, so N ELFs become N `hw_context`s and still
> one runlist, bit-identical to sequential dispatch and measurably faster. Read
> [05a-phase-b-runlist-spike-result.md](05a-phase-b-runlist-spike-result.md) before acting on
> anything in §1 — it records both halves and the measurements. The buffer rules asked for in
> work item 2 are in [05b-phase-b-buffer-rules.md](05b-phase-b-buffer-rules.md).

## 1. Runlist aggregation

### The problem

A `pyxrt.runlist` is constructed against a **single** `hw_context`. MLIR-AIR's
`XRTBackend.load()` (`python/air/backend/xrt.py:504`) creates one `hw_context` per artifact:

```python
self.elf = xrt.elf(artifact.output_binary)
self.context = xrt.hw_context(self.device, self.elf)
self.kernel = xrt.ext.kernel(self.context, artifact.kernel)
```

So N separately-loaded ELFs cannot naively share a runlist. At first look this appears to
resurrect iron's `aiecc --xclbin-input` incremental-merge dependency, which we do not want to
reproduce.

### The resolution

`pyxrt` in this environment exposes the module API. Verified interactively:

```python
pyxrt.module.__init__     # (elf) | (capsule, int, uuid) | (module, hw_context)
pyxrt.ext.kernel.__init__ # (hw_context, module, str) | (hw_context, str)
pyxrt.runlist.__init__    # () | (hw_context)
```

which gives:

```python
ctx  = pyxrt.hw_context(device, pyxrt.elf(first_elf))     # one context for all kernels
mod  = pyxrt.module(pyxrt.elf(path))
pyxrt.module(mod, ctx)                                    # bind each further ELF into it
kern = pyxrt.ext.kernel(ctx, mod, kernel_name)            # 3-arg overload takes a module
rl   = pyxrt.runlist(ctx)
rl.add(pyxrt.run(kern)); ...
rl.execute(); rl.wait()
```

Multiple independently-compiled ELFs bind into one `hw_context` and submit as one runlist.
**Do not reproduce iron's xclbin merge.**

### Why this is still an assumption

`[Codex]` The above is API-shape evidence, not hardware evidence. The repository's only runlist
usage (`test/xrt/24_ctrlpkt_config_2gemms_4x4/test.cpp:216`) uses runs from the *same* context
and kernel. It does not demonstrate that separately-compiled ELF artifacts can share one
runlist on NPU2.

**If this fails, `runlist` and `coarse` collapse into `offload` and the study loses its central
axis.** Spike it first, with the real artifacts, before any other phase starts.

### Where the code goes

`[Codex]` Implement the aggregation in **`KernelCache`**, not only in `XRTBackend.load()`.
`KernelCache.load_and_run()` bypasses the `load()` invoker and submits `xrt.run` directly
(`shared/infra/cache.py:511`), so a runlist path added only to the backend would be dead code
on the path the LLM stack actually uses.

Specify explicitly:

- **Common context requirements** — which artifacts may share a context, and what happens when
  they cannot.
- **Per-run ABI** — the ELF path sets buffer args from index 0 (instructions are embedded in the
  ELF); the xclbin path uses opcode at 0, instruction BO at 1, instruction length at 2, and
  buffers from 3. The aggregation must not conflate them.
- **Buffer-argument mapping** — which BO belongs to which run.
- **Error handling** — a failed run inside a runlist must be attributable to that run.
- **One timing scope** covering the whole runlist, matching the dispatch-vector definitions in
  [03-measurement-model.md](03-measurement-model.md).

## 2. BO liveness pooling

### What it does

Ported in concept from `iron/iron/common/aie_context.py:44-224`: compute per-buffer live ranges
over the dispatch sequence (producer index → last consumer index), bin by 4 KiB-rounded size,
mark overlapping same-size buffers as conflicting, and assign pool slots so non-overlapping
buffers share one BO. Static weight buffers go into a content-keyed pool so identical weights
across operators map to one BO.

### Why it is not a drop-in

`[Codex]` This is **not** a ~180-line generalization of what `KernelCache` already does.

Current reuse is scoped by kernel name and size and carries explicit lifetime caveats
(`cache.py:321`, `cache.py:464`) — notably that outputs drawn from the shared pool are zero-copy
views overwritten by the next call. Cross-kernel reuse must additionally respect:

- XRT context
- memory group / bank
- alignment
- argument type
- in-place writes
- host-view lifetime

A naive size-bin allocator will produce invalid aliasing or stale reads.

**Define buffer ownership, host/device synchronization rules, bank compatibility and aliasing
rules before implementing.** Then test: overlapping live ranges, output reuse, cross-artifact
buffers, and context changes.

`KernelCache` already has hand-rolled special cases — `static_input_indices`,
`intermediate_indices`, `shared_nonstatic` — which the allocator should subsume rather than sit
beside.

## 3. Dirty-bit synchronization

Port iron's sync discipline: only written buffers sync to device, only declared outputs sync
back. iron previously synced every BO in both directions on every run; fixing that is what made
its latency numbers meaningful.

Without it, measured latency is not comparable to iron's and the port cannot be validated.

## What is explicitly dropped

Convention rule 2 — do not port:

| iron artifact | Lines | Covered instead by |
|---|---|---|
| `compilation.py` artifact DAG | 712 | `KernelCache` + native `aircc` (compile caching, rebuild-on-flag-change) |
| `AIEContext` | 246 | `KernelCache` + `air.tools` path resolution |
| `AIEDeviceManager` | 88 | `XRTBackend` device lifecycle |

BO pooling is the only idea worth extracting from that tier.

## Work items

1. **Spike first**: two separately-compiled ELFs, bound as modules into one `hw_context`,
   submitted as one runlist on NPU2. Confirm numerical correctness against sequential dispatch.
2. Write the buffer ownership / sync / bank / aliasing rules document before touching the
   allocator.
3. Implement runlist aggregation in `KernelCache`, with the ABI distinction handled explicitly.
4. Implement the liveness allocator, subsuming the existing special-case indices.
5. Implement dirty-bit sync tracking.
6. Add the dispatch-vector instrumentation from
   [03-measurement-model.md](03-measurement-model.md) — one implementation, called by all modes.
7. Tests: overlapping live ranges, output reuse, cross-artifact buffers, context changes.

## Gate

A hardware test using the **exact separately-compiled artifacts the study will use** shows a
multi-ELF runlist that is:

1. numerically identical to sequential dispatch, and
2. measurably lower latency.

Wrap it in `flock -x -w 1800 /tmp/mlir-air-npu.lock`.

## Risks

- **The multi-ELF runlist may not work on NPU2.** This is the plan's single biggest unknown.
- Changes to `shared/infra/cache.py` affect all ten shipped LLM deployments. Re-run `make verify`
  across them before merging — see
  [13-verification-and-acceptance.md](13-verification-and-acceptance.md).
- `KernelCache` serializes on `filelock.FileLock("/tmp/npu.lock")`, deliberately a *different*
  inode from the outer `flock /tmp/mlir-air-npu.lock` convention, to avoid flock self-deadlock.
  Do not unify them.
