# 06d — Phase C4: The coverage sweep

Last of Phase C's four sub-phases. Read [06](06-phase-c-operators.md) for the overview and
[02](02-porting-conventions.md) for binding house style.

C1–C3 built the operators. C4 builds the tool that measures tile configurations for the shapes the
case matrix needs and writes them into `kernel_registry`, so Phase D and Phase E can resolve a
shape without a `KeyError`.

## The problem

`registry_lookup.gemm_config()` **raises** on an unmeasured `(M, K, N)` rather than guessing —
deliberately, because hand-copied tile configs previously caused drift and stale-config bugs
(`kernel_registry/registry_lookup.py` docstring). Its error message tells the caller to run a
sweep and add the shape.

The case matrix needs **108** distinct projection-GEMM triples (`qkv_proj`, `ffn_up`, `ffn_down`,
`o_proj`, 27 each, from hidden ∈ {512, 768, 1024} × the 9-point sequence ladder). **5** are
registered today. The other 103 raise.

Three resolutions were considered ([06](06-phase-c-operators.md)); this is the one taken. iron's
`block/run.py` already sweeps per-operator candidates and records a winner per `(operator, shape)`,
so registry coverage accrues as a byproduct of the measurement the study wants anyway. This is
also convention rule 9 — registry as the single source of truth.

## What to build

`programming_examples/transformer_layer/sweep/registry_sweep.py`.

Per `(operator, shape)`: build each candidate configuration, check it numerically through the
C1 `opcheck.py` path, time it, and record the fastest passing candidate as that shape's row.

Read `iron/applications/transformer_layer/study/block/run.py` (1313 lines) for the mechanics worth
carrying — not the code, the design:

- **Resume.** `load_existing_rows_from_paths` reuses passing rows keyed on the shape *plus a
  config signature*, so editing a candidate invalidates its reuse rather than silently keeping a
  stale winner. You need this: `PL_STEP_TIMEOUT` is 3 hours and this sweep is longer.
- **Subprocess isolation per candidate.** Long-sequence runs leak and fragment enough that iron
  runs each candidate in its own process with an aggressive cleanup between cases.
- **Turbo enforcement.** iron fails a measurement outright if `xrt-smi` does not report turbo. Do
  the same — a non-turbo number is not comparable to a turbo one, and silently mixing them
  corrupts the registry.
- **Per-case checkpointing**, so a kill mid-sweep loses one case, not the run.

## Writing to the registry

Rows go into `kernel_registry/details/GEMM_bf16_in_bf16_out.json`, mirrored into
`details/GEMM_bf16_in_bf16_out.md` and `supported_kernels.md`. The JSON entry schema is fixed —
match it exactly:

```json
{
  "M": 2048, "K": 2048, "N": 2048,
  "used_by": "…",
  "methods": {
    "<method>": { "tile": {"tile_m":…, "tile_k_l2":…, "tile_k_l1":…, "tile_n":…},
                  "gflops": …, "mean_rel_L1": …, "tier": "high"|"low" }
  },
  "best": { "high": "<method>", "low": "<method>" }
}
```

`tile_m` is dictated by the method, not measured: drain is 32, fused-cast is 64
(`gemm_builder.py:21-26`), and `_spec_with_tiles` asserts the registry agrees.

**Append only.** Never rewrite an existing shape's entry. The 40 shapes already in the two JSONs
are what the ten shipped LLM deployments resolve against, and re-measuring one into a different
winner would change their behaviour without anyone asking. The driver's objective check verifies
that every shape present at the sub-phase base commit is byte-identical afterwards.

Two structural notes:

- The tamper check fingerprints `kernel_registry/details/*.json` but **not**
  `supported_kernels.md` or `details/*.md`. The JSON path is in this sub-phase's gate allowlist
  deliberately; the markdown rows are unprotected by that layer, so the objective check covers
  them instead. Keep the two in sync — the markdown is what a human reads.
- `registry_lookup` scans `data["shapes"]` linearly. Fine at 33 entries; if this sweep grows it by
  an order of magnitude and lookup becomes hot, say so in your report rather than rewriting the
  lookup as a side effect of this sub-phase.

## Scope: what is staged, what is deferred

Registering all 103 missing shapes is a very long hardware run, and Phase D needs one family.

**Acceptance is the `baseline_768` family across the full 9-point ladder — 36 shapes:**

| Role | Shape | Count |
|---|---|---|
| `qkv_proj` | `(seq, 768, 2304)` | 9 |
| `ffn_up` | `(seq, 768, 3072)` | 9 |
| `ffn_down` | `(seq, 3072, 768)` | 9 |
| `o_proj` | `(seq, 768, 768)` | 9 |

with `seq ∈ {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384}`.

The `512`/`2048` and `1024`/`4096` families are the same tool run over a different family id, and
are deliberately left as a later machine-time run rather than a blocker on Phase D. Make the
family a command-line argument so that run needs no code change.

If a shape will not place, record it as a `❌` row with the error rather than omitting it. A
missing row and a known-broken row look identical to a later reader otherwise, and
`supported_kernels.md` already has a status legend for exactly this (✅ / ⚠️ / ❌).

## Work items

1. `sweep/registry_sweep.py` — resumable, checkpointed, subprocess-isolated, turbo-enforcing,
   family selectable on the command line.
2. Sweep the 36 `baseline_768` shapes and write their rows into the registry JSON and both
   markdown pages.
3. Verify every one resolves: `gemm_config(m, k, n)` returns rather than raising, for all 36.
4. A `run_npu2_*.lit` covering the resolution check (cheap, no sweep) so regressions surface.
5. `black`; a module docstring stating the contract and its footguns — especially append-only.

## Gate

```
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  ninja -C build-xrt check-programming-examples-transformer-layer
```

plus, as this sub-phase closes Phase C and adds rows the shipped models read, the
cross-deployment regression check from
[13](13-verification-and-acceptance.md#the-cross-deployment-regression-rule): `make verify` on all
ten shipped models, serialized under `flock`. Adding rows is additive — lookup is an exact
`(M, K, N)` scan — but this is the check that catches what matters, and it is the one most likely
to be skipped.

The driver's objective check additionally requires: the 36 shapes resolve through `gemm_config`,
the pre-existing 40 shapes are byte-identical to the base commit, and the registry JSON is newer
than the gate stamp.

## Constraints

- **Append only.** Never modify an existing registry shape.
- **Do not modify `llms/shared/`.**
- Wrap every NPU command in `flock -x -w 1800 /tmp/mlir-air-npu.lock`. Never take
  `/tmp/npu.lock`.
- The sweep is long. Checkpoint, and report honestly how far it got — a partial sweep with 36
  shapes registered is the acceptance condition, not "all 108 or nothing".
