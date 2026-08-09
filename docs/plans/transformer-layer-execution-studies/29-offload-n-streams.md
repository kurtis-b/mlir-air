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

## Known gap — the shared path is NOT gated

**Nothing in the lit suite exercises `AIR_OFFLOAD_SHARED_XCLBIN=1`.** The whole
new path — `compile_shared_xclbin`, `attach_kernel`, the sharing branch in
`ensure_loaded` — is verified by hand-runs and by the E1 gate only in the sense
that the E1 gate proves it did not *break* anything. And
`run_npu2_offload_peano.lit` does not pin the `reconfiguration:` line, so a
regression to 30 context loads would pass green.

This is recorded rather than quietly carried: **the mode's central claim is
printed, not enforced.** Closing it means a second recipe in
`run_npu2_offload_peano.lit` that runs the shared path and FileChecks both the
reconfiguration counts and the unchanged dispatch vector.

## What this does not do

- **It does not make the shared path the default.** `AIR_OFFLOAD_SHARED_XCLBIN=1`
  opts in, and the gated ELF path is untouched, with its own cache directory —
  the two builds produce artifacts with identical NAMES over different ABIs, so a
  shared directory could trade them and credit a 30-reconfiguration run to the
  one-reconfiguration claim.
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
