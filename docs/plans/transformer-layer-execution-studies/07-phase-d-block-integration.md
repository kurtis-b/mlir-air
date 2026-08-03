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

## What to build

Assemble **one** complete transformer layer — the `encoder_bert` variant at a single family and
sequence length — through the real path:

```
KernelCache → air.launch → runlist → host readback
```

Not a mock, not a subset. The same `KernelCache` the study will use, the same runlist
aggregation from Phase B, the same builders from Phase C.

Validate against `pattern/reference.py`, the pure-torch golden model ported verbatim from iron.

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

## Work items

1. Port `pattern/reference.py` (172 lines, pure torch) verbatim — this is the correctness anchor
   for this phase *and* all of Phase E.
2. Assemble one `encoder_bert` layer at one family / sequence length through
   `KernelCache` + runlist.
3. Compare against the torch reference element-wise.
4. Add an intermediate-value comparison per operator boundary, so a failure localizes to a
   stage rather than to "the layer".
5. Wrap the hardware run in `flock -x -w 1800 /tmp/mlir-air-npu.lock`.

## Gate

One full transformer layer matches the torch golden model end-to-end on real hardware.

If the element-wise comparison fails, the per-boundary intermediates identify which stage
diverged; do not proceed to Phase E on a layer that only approximately matches.

## Risks

- This phase has no new device code, so a failure here means something in Phase B or C is wrong
  in a way its own gate did not catch. Budget time for iterating back into those phases rather
  than treating this as a formality.
