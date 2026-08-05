# 07 — Phase D: Single-Block Integration Gate

A short phase with one purpose: prove that one complete transformer layer works through the
real runtime path before four execution strategies are built on top of it.

## Why this phase exists

`[Codex]` Phase C's per-operator `np.isclose` checks are necessary but not sufficient. They do
not exercise:

- AIR launch argument maps
- layout transitions between operators
- external-kernel linking across a multi-operator sequence
- BO reuse and synchronization under the Phase B allocator
- complete multi-launch layer assembly

Every one of those is a documented source of silent corruption in this repository. Without a
block-level gate, the first place they would surface is inside a four-way comparison, where
attributing a discrepancy to a mode versus to the integration is far harder.

This mirrors the repository's own deployment discipline: `phase-2-single-block-validation`
exists as a distinct gate between per-kernel validation and full-model assembly for exactly this
reason.

## What Phase C left you

`[2026-08-04]` All of this exists and is gated on real hardware. Do not rebuild any of it; the
example's own `programming_examples/transformer_layer/README.md` is the authoritative file-by-file
inventory.

| Piece | Where |
|---|---|
| Operator builders | `transformer_layer/builders/` — `elementwise_add` (with the `causal_mask=` keyword), `layer_norm`, `addnorm`, `qkv_proj`, `gelu`, `ffn`, `mha_attention`, `o_proj`, `mha_out_proj`, plus `gemm_spec.py` |
| Numerical check | `transformer_layer/opcheck.py` — the CLI, the results artifact, and the fault-injection negative control |
| Registry sweep | `transformer_layer/sweep/` — `registry_sweep.py`, `registry_writer.py`, and their host-only tests |
| Per-operator gates | one `run_npu2_<op>_peano.lit` each, plus `run_npu2_fault_control_peano.lit` |

**Reuse `opcheck.py`. Do not write a second checker.** It already owns the contract the driver's
objective check depends on — FP32 reference, registry `rtol`/`atol`, zero mismatches, a
machine-readable verdict per `(operator, shape)`, and `--fault-inject input` which must fail.
Phase D adds a `block` entry to it rather than standing up a parallel mechanism.

Two footguns in it worth knowing before you start:

- `opcheck.py --list` imports `XRTRunner` at module scope, so it needs the AIR environment sourced
  even though it touches no hardware and needs no NPU.
- Injected runs are written to `results/fault/`, deliberately in a subdirectory, so a
  fault-injected result can never be mistaken for or overwrite a clean one.

## The family is forced, and the operators are not all validated at it

This is the thing most likely to cost a day if it is discovered from a failing run rather than
read here.

`registry_lookup.gemm_config()` raises on an unmeasured shape, so a family is usable only if its
projection GEMMs are registered. After C4:

| Family | hidden / ffn | GEMM shapes registered |
|---|---|---|
| `tinybert_512` | 512 / 2048 | 2 of 36 |
| **`baseline_768`** | **768 / 3072** | **36 of 36** |
| `baseline_1024` | 1024 / 4096 | 3 of 36 |

So Phase D runs at **`baseline_768`** — it is the only family whose GEMMs resolve.

But Phase C validated its operators at whatever widths each was cheapest to bring up, and only one
of those points is at 768:

```
elementwise_add  512x512          layer_norm   512x512        addnorm  64x512
qkv_proj         2048x1024, 2048x2048, 64x768   ffn  2048x1024x3072
mha_out_proj     512x512x8h, 512x512x8h_causal, 2048x1024x16h_causal
```

Every operator except `qkv_proj` is unvalidated at `hidden = 768`. The GEMM *tiles* for those
shapes are registered, which is a different claim from the operator computing the right answer at
that width. **Extending each operator's `opcheck` shape set to the `baseline_768` widths is
Phase D's first work item, not an optional extra** — otherwise the block gate is the first thing
that ever runs those shapes, and a failure will not localize.

## The golden model needs the same correction Phase C made

`pattern/reference.py` (172 lines, pure torch) is the correctness anchor for this phase *and* all
of Phase E. An earlier version of this document said to port it **verbatim**. Do not.

It defaults to `dtype: str = "bf16"` and builds every tensor — inputs, all four attention weight
matrices, both LayerNorm weights, both FFN weights — at that dtype. That is exactly the defect
corrected in [06 §The numerics standard](06-phase-c-operators.md#the-numerics-standard--do-not-port-irons),
and it is *worse* here than it was per-operator: this reference chains eight GEMMs plus two
LayerNorms plus a softmax, so a bf16 oracle accumulates error in the same direction as the device
and the comparison flatters itself the whole way down.

Port the structure; compute in FP32 from bf16-rounded inputs, as every Phase C reference does.

## Known failure modes to check for

These are drawn from the repository's own debugging skills and are the specific things this gate
is designed to catch:

| Symptom | Typical cause |
|---|---|
| All-zero output from a herd | Bare herd not wrapped in a launch/segment |
| Silent corruption in GEMM output | `N % (tile_n × herd_n) != 0` |
| Correct standalone, wrong when chained | BO reuse without correct synchronization |
| `ERT_CMD_STATE_TIMEOUT` | `instance_name` not matching the emitted `func.func @name` |
| Correct first call, wrong on subsequent calls | Stale buffer contents under pooling |
| NaN in attention output | L1 overflow at large head dimension |

The last three are direct interactions with Phase B's allocator, which is why this gate follows
it rather than preceding it.

C4 added one more worth carrying: a GEMM configuration that returned **zeros for two of the nine
sub-tiles of each cast worker** while still resolving from the registry and still producing a
plausibly-shaped output. A registry lookup proves a row exists, never that it computes. That is
why the per-boundary intermediate comparison below is a work item and not a nicety.

## Work items

1. Extend each operator's `opcheck` shape set to the `baseline_768` widths and re-validate on
   hardware, per the section above.
2. Port `pattern/reference.py` with an FP32 reference computed from bf16-rounded inputs.
3. Assemble one `encoder_bert` layer at `baseline_768`, one sequence length, through
   `KernelCache` + runlist — the same seam Phase B built, the same builders Phase C landed.
4. Compare against the torch reference element-wise, through `opcheck.py`.
5. Add an intermediate-value comparison per operator boundary, so a failure localizes to a stage
   rather than to "the layer".
6. Wrap the hardware run in `flock -x -w 1800 /tmp/mlir-air-npu.lock`.

## Gate

One full transformer layer matches the torch golden model end-to-end on real hardware, via a
`run_npu2_block_peano.lit` in the existing `check-programming-examples-transformer-layer` suite.

If the element-wise comparison fails, the per-boundary intermediates identify which stage
diverged; do not proceed to Phase E on a layer that only approximately matches.

## Risks

- This phase has no new device code, so a failure here means something in Phase B or C is wrong
  in a way its own gate did not catch. Budget time for iterating back into those phases rather
  than treating this as a formality.
- Work item 1 is real hardware time on nine operator/shape points before the block runs at all.
  It is still cheaper than debugging a chained layer whose stages were never individually checked
  at that width.
