# 19 — Phase J1: collapse the norm dispatches

The largest structural gap between this port and iron, and the highest-value item in tranche J.
`builders/addnorm.py` forbids more than one trip of its row loop, so the layer's two normalization
points are row-blocked into 64 dispatches each — **128 of `coarse`'s 131 runlist entries** against
iron's `hybrid` total of 5.

Lift the guard, re-measure `coarse`, and replace the one distinguishability clause that lifting it
makes vacuous.

## Why it is safe now, and why that is measured rather than argued

The guard's docstring blames three L3→L1 streams per tile against a column's two shim MM2S
channels. **That was the symptom, not the cause.** The cause was the shim feed order under packet
multiplexing:

1. `air-dma-to-channel` hoists each L3-side DMA into its own launch-scope loop, so a herd filling
   N buffers per iteration produces N sibling per-channel put loops.
2. Packet-multiplexed onto one shim MM2S queue, that queue serialises in task order — whole channel
   after whole channel.
3. The consuming tile's BD ring expects them interleaved per iteration. At one trip the two orders
   coincide; at two or more, every packet after the first lands in the wrong buffer.

Fixed by **`air-fuse-packet-put-loops`** (`bfb647d9`). The driver's own fixture runs the
`--variant inside` two-trip loop on hardware with **zero mismatches**, at exactly the
`cols=64, rows=8, rows_per_call=4` shape the guard was written against. Ping-pong was never
involved: `--omit-ping-pong-transform=all` reproduced the identical 481/512 corruption, and
`air.disable_ping_pong` on that loop changes nothing because the loop is never labeled at all
(its callee is unannotated).

## The arithmetic, which is not what the plan's shorthand suggests

Computed from the builder's own helper at the block's width, `cols=768`, `herd_x=8`, pre-add:

| | value |
|---|---|
| `addnorm_max_rows(768, herd_x=8, pre_add=True)` | 104 rows per launch |
| L1 per tile at the cap | 62,464 of 65,536 bytes |
| today | `block.py` bands at **64 rows** → **64 dispatches** per normalization point |
| lifted | **1** launch, `rows_per_tile = 512` |

**The trip count is 64, not 2.** `rows_per_call` must divide `rows_per_tile = 512` — a
non-divisor makes the final trip read and write past the band — and the L1 cap is 13, so the
largest legal value is **8**, giving 512/8 = 64 trips per tile.

That is a **32× extrapolation from the only multi-trip result anyone has measured**. Do not treat
the fixture's two trips as evidence for 64. Walk it: 2 trips, then 8, then 64, at `cols=768`, and
report the first count that fails if one does. A partial result here is a finding worth having, not
a failure — the whole point of the guard was that the failure mode looks numerical rather than
structural.

## Work items

**J1a — lift the guard.** `builders/addnorm.py:264-271` raises unless
`rows_per_call == rows // herd_x`. Remove the raise; keep the L1 budget check, which is a real
constraint. Rewrite the module docstring's §"ONE KERNEL CALL PER TILE" around the packet feed
order and `air-fuse-packet-put-loops` — it currently states the row cap as a correctness constraint
and names the wrong mechanism. Keep `addnorm_max_rows`: it still bounds `rows_per_call`.

**J1b — stop banding in the block.** `builders/block.py` row-blocks both normalization points into
64 dispatches (sequences B and D). One launch over all 4096 rows replaces each. The docstring's
§"Because `addnorm` cannot be dispatched over 4096 rows" is the passage to correct.

**J1c — re-measure `coarse`** and record the new dispatch vector in its README beside the old one.
Expect 131 entries → about 5. Report what you measure; do not tune a mode until a number is
reached.

**J1d (this is J4) — replace distinguishability clause 3.** `runlist entries > coarse entries`
becomes true by construction once coarse drops to ~5, so it stops measuring anything. Use
`herd_launches` instead — 404 against 146 — which counts executed work rather than dispatch
packaging and which neither mode fixes by construction. The clause lives in `08e`'s gate and is
enforced through `phase_e_checks.py compare`; switch its `--field`.

## Gate

```
flock -x -w 1800 /tmp/mlir-air-npu.lock  ninja -C build-xrt check-programming-examples-transformer-layer
```

The whole transformer-layer suite on hardware, including every mode's lit test and the D1/D2
coverage clauses. Allowlist: `^programming_examples/transformer_layer/`.

Then the driver's objective check, which a session cannot satisfy by writing a laxer test:

```
phase_e_checks.py mode --operator coarse --max-field runlist_entries --max-value 10
```

Four things at once, and the last is the one that matters:

- every per-boundary `n_mismatch` is 0 over the full 4096×768 layer;
- eight distinct clean stage boundaries, so the artifact describes the whole layer;
- the **fault-injected twin's summed vector totals equal the clean run's** — a session cannot know
  those six numbers without dispatching;
- `runlist_entries ≤ 10`, down from 131.

That last clause exists because **the mode agreeing with the oracle proves nothing here.** It
agreed at 131 too. The collapse is the entire result, so it is asserted separately or the phase can
pass having changed nothing. Both directions are covered by
`python3 agents/scripts/port-loop/phase_e_checks.py selftest` (30 clauses, no hardware).

## What this phase must not do

- **Do not widen a tolerance.** The layer sits at `atol` `1e-1`, the hard ceiling, at 1.35× its
  measured requirement. If the collapsed form needs more, that is a finding, not a knob — and the
  driver rejects anything above `1e-1`.
- **Do not disable ping-pong** anywhere to make a trip count work. It is measurably expensive
  (12.4 → 7.8 tok/s on a shipped model) and it is not the mechanism here.
- **Do not touch `mlir/`.** If the collapse exposes a compiler defect, report it in
  `work_not_completed` with the shape that shows it; the fix is a later phase.
- **Do not reach for the `air.channel` rewrite** as a workaround. It is measured correct at 2 trips
  and it **cannot reach this band count at all**: 64 bands × 4 channels is far past the 16 locks a
  tile has, and `air-to-aie` fails with `'aie.lock' op lock assigned invalid id (maximum is 15)`.
  That is H5, and it is not a prerequisite for this phase — the direct route needs no channels.
