# 08e — Phase E5: `fused`, and the distinguishability gate

The most fused point of the taxonomy: MLIR-level fusion before compilation, via `stitch_elf`. This is
MLIR-AIR's own production mechanism — measuring it is what makes this port additive rather than a
duplicate of iron.

It is also the sub-phase that closes the set, so its gate is the one that asks whether the four modes
**separate**. That question is the whole point of the taxonomy, and a failure to separate is a
result rather than a defect.

Read [08b](08b-phase-e2-coarse-and-instrumentation.md) for the artifact contract. This mode satisfies
the same contract with a fourth execution boundary behind it.

## Why this mode could not be built before E1

`stitch_elf` (`llms/shared/infra/stitching.py:318`) splices MLIR text fragments into one module. A
whole-layer stitch has to co-link the layer's four projection GEMMs, and the block's own resolved
specs are:

```
qkv       fused-cast  tile_n = 96        ffn_down  fused-cast  tile_n = 96
ffn_up    drain       tile_n = 128       o_proj    drain       tile_n = 96
```

Two `drain` GEMMs at different `tile_n` in one module declare `f32_to_bf16_mn_<suffix>` twice with
different memref types, and `stitch_elf` collects both texts into one set and fails to parse
(`stitching.py:408-412`). That is not a sequence-length problem — it holds at `seq = 4096` and at
every other point. [08a](08a-phase-e1-unblock-the-ladder.md) removed it by minting the suffix per
`(method, tile_n)`. If E1 has not landed, this mode cannot be built at all; check that it has before
you start.

## What is not ported

iron reaches one-xclbin-many-kernels through `aiecc --xclbin-input` incremental merge, chaining
`XclbinArtifact.xclbin_input` with `--xclbin-instance-name` and `--xclbin-kernel-id 0x801, 0x802, …`
so each new kernel merges into the incoming xclbin.

**Do not reproduce that.** MLIR-AIR does not need it: multiple independently-compiled ELFs bind as
modules into one `hw_context`, which is Phase B's result. This mode is a thin wrapper over existing
`stitch_elf` usage, not a new packaging mechanism.

## Instrumentation, and the one field that behaves differently here

The contract is [08b](08b-phase-e2-coarse-and-instrumentation.md)'s, unchanged. One field deserves
attention before you read your own numbers:

`air_launches` is counted **once per distinct ELF**, while `herd_launches` accumulates **per step**
(`dispatch.py:122-153`, and the implementation at `:425-456`). The asymmetry is deliberate and it is
[03](03-measurement-model.md)'s, not an accident of the implementation. A mode that fuses many
launches into one ELF therefore shows a large `air_launches` on a small number of artifacts, which is
exactly the signature this mode is supposed to have — and it is also why comparing `air_launches`
across modes needs care rather than a naive inequality. See the gate below for which comparisons
gate and which are only recorded.

## Work items

1. `pattern/fused/` — a `stitch_elf` wrapper over the whole layer, its `README.md`, its own
   `KernelCache` directory added to `transformer_layer/.gitignore` and the `clean` target in the same
   commit.
2. A `fused` operator spec in the `SPECS` catalogue, through the `dispatch` seam, recording
   `execution_mode: "fused_elf"`.
3. `run_npu2_fused_peano.lit` — both recipes in one file, clean and
   `--fault-inject input --expect-failure`.
4. **A `README.md` for each of the four strategy directories**, if any is still missing: what
   boundary it isolates, what it costs, and its measured dispatch vector. This is work item 7 of
   [08](08-phase-e-execution-strategies.md) and E5 is where it is finally checkable, because all four
   numbers exist.
5. **Record the four-mode dispatch-vector table** in the example's `README.md`. It is the phase's
   headline result and it should not live only in `results/`, which is gitignored.

## The gate

```
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  ninja -C build-xrt check-programming-examples-transformer-layer
```

Every test in the suite passes on real hardware.

Then the driver checks `fused` exactly as it checked the other three — re-derived verdict, exactly
one fresh result at the forced configuration, full-layer `n_elements`, ≥8 distinctly-named clean
stages, the `dispatch_vectors` contract, a fault-injected run that must fail, and fault-run vector
totals equal to the clean run's.

And then it checks the thing this whole phase exists to establish.

### Distinguishability

The driver sums each mode's recorded vectors itself — `Σ round(entries_per_submission ×
host_submissions)` for entries, plain sums for the rest — and prints the full four-by-six table
whether it passes or fails.

**Gating clauses.** Each follows from the definition of the execution boundary, not from a guess at
how a mode is implemented:

1. **Distinctness.** No two modes' six-field totals are equal. This is the floor: if two modes
   produce the same vector, the vector is not measuring the boundary.
2. **`offload` is the host-mediated extreme.** Its `host_submissions` strictly exceeds every other
   mode's, and its `runlist_entries == host_submissions` — it aggregates nothing.

   > **`[2026-08-09]` The clause holds; the name is the superseded taxonomy.** At 1024 `offload`
   > is 30 submissions against `runlist` 17, `coarse` 4 and `fused` 1, `entries == submissions`.
   > But [03](03-measurement-model.md) corrected `offload` to the **reconfiguration-minimizing**
   > mode, and since 2026-08-09 it configures the array **once** per layer rather than 30 times
   > ([29](29-offload-n-streams.md)) — so "host-mediated extreme" is now doubly a misnomer, while
   > what the clause tests (it aggregates nothing) is unaffected and still worth gating.
3. **`runlist` is finer than `coarse`.** `runlist.runlist_entries > coarse.runlist_entries`.
4. **`fused` removes intermediate host sync**, which is what MLIR-level fusion *is*:
   `fused.sync_boundaries < coarse.sync_boundaries`.

**Recorded, not gating:** `fused.runlist_entries < coarse.runlist_entries`, and
`fused.air_launches >= coarse.air_launches`. Both assume a particular decomposition — a faithful
whole-layer stitch may still row-block its normalization, and the `air_launches` /`herd_launches`
asymmetry above makes the second a weaker claim than it looks. The driver prints them with a verdict
and does not halt on them. If either is false, that belongs in the README as a finding.

### Why the criterion is ordinal

`coarse` measures **4 submissions, 131 runlist entries, 12 AIR launches, 146 herd launches, 402 sync
boundaries** — and 128 of those 131 entries are `addnorm`'s row blocking, before any fine-grained
mode exists. [08](08-phase-e-execution-strategies.md) and [03](03-measurement-model.md) both warn
that "`runlist` many, `coarse` few" may therefore be a comparison between a number and itself. Any
absolute threshold here would be measuring `build_addnorm_module`'s L1 capacity rather than the
taxonomy. Ordering survives that; thresholds do not.

### If it does not separate

Then the taxonomy is not measuring what it claims, and [08 §Gate](08-phase-e-execution-strategies.md)
is explicit that the measurement model needs revisiting **before Phase F consumes it**. The driver
halts and says so.

That halt is not a failure of your implementation, and the response is not to adjust a mode until the
inequality holds. Report the measured table in `work_not_completed` with what you believe each mode's
boundary actually is, and let the model be revised. A mode tuned to satisfy a predicted inequality
would make every downstream measurement meaningless in a way nothing later could detect.

## Risks

- **This gate can fail for reasons outside this sub-phase.** Clause 3 is about `runlist` and clause 2
  about `offload`; a defect in either surfaces here, at the end, after their own gates passed. That
  is unavoidable — cross-mode separation cannot be checked before the modes exist — and it is why the
  halt message must name which clause failed and with which numbers.
- **`stitch_elf` failures are loud but unhelpful.** A redefinition error names the symbol and not the
  two GEMMs that produced it. If one appears after E1, the naming fix has a case it did not cover;
  report which two specs collided.
- **The tolerance has no headroom.** `atol` sits at the hard `1e-1` ceiling at 1.35× the block's
  measured requirement, and this is the fourth mode chained against the same oracle. If `fused` needs
  more, that is a finding about what MLIR-level fusion does to the arithmetic — a real result — and
  not a tolerance to widen. The driver rejects anything above `1e-1` regardless.
