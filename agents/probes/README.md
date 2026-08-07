# Measurement probes

One-off tools that produced the measurements load-bearing specs in
`docs/plans/transformer-layer-execution-studies/` rest on. **Not tests** — nothing runs them in CI,
and they are kept so a later session can re-derive a claim instead of trusting prose.

Run from the repo root with the toolchain on PATH:

```bash
. agents/scripts/port-loop/lib-env.sh && pl_env_ensure   # needs a log_error shim if run bare
python3 agents/probes/<probe>.py
```

| probe | what it establishes | cited by |
|---|---|---|
| `probe_packet_streams.py` | The **per-column shim stream budget**. `addnorm` (3 L3→L1 streams) gets 3 packet-typed channels; `elementwise_add` (2) and `layer_norm` (1) get none. This is why `fused`'s decomposed tail always ran 64 trips on 8 columns correctly and the fused `addnorm` could not. | [23](../../docs/plans/transformer-layer-execution-studies/23-rules-and-open-items.md) |
| `probe_j7a_pipeline.py` | The three-herd norm tail places at `herd_x=8`, and **packing x\|residual takes it to zero packet-typed channels** (`--packed`). Without packing it enters the packet path. | [21](../../docs/plans/transformer-layer-execution-studies/21-phase-j7a-norm-tail-pipeline.md) |
| `probe_accum_hoist.py` | The accumulator-ring 2×2: the ring forms **only** with the in-place kernel **and** the accumulator allocated inside the K loop (`--shape inloop`). Pass-level, via `air-opt`. | [22](../../docs/plans/transformer-layer-execution-studies/22-phase-j7b-accumulator-ring.md) |
| `probe_accum_aircc.py` | The same claim at **`aircc` altitude** rather than a pass list — `pass_006` shows 4 → 2 data-movement ops in the K loop. Written after an `air-opt`-only claim was falsified by a real compile. | [22](../../docs/plans/transformer-layer-execution-studies/22-phase-j7b-accumulator-ring.md) |
| `probe_ffn_accum_bd_offset.py` | **A per-iteration L2 read offset is dropped past the unroll limit.** `aie.dma_bd` offsets are static; at 2 trips the loop unrolls and each carries its own literal offset, at 4+ the chain cycles with the offset frozen at 0. Silently wrong data, then a hang. Doc 16's H5 with a worse failure mode than H5 records. | [22](../../docs/plans/transformer-layer-execution-studies/22-phase-j7b-accumulator-ring.md) · [23](../../docs/plans/transformer-layer-execution-studies/23-rules-and-open-items.md) |
| `probe_layer_norm_twopass_cost.py` | **What J7a's f32 two-pass variance cost: ~13%** (min-to-min, +13.2% at 4096×768 and +13.5% at 512×512) for ~26× accuracy. Carries its own provenance check — the reconstructed one-pass kernel must reproduce the 1.969e-3 the catalogue records, or the "before" build is not the old kernel. | [23 §1](../../docs/plans/transformer-layer-execution-studies/23-rules-and-open-items.md) |
| `probe_addnorm_variance_cliff.py` | **The fused `addnorm` collapses on large-mean rows; `layer_norm` does not.** Three modes: `compare` (does it break), `sweep` (where — measured at \|mean\|/σ between 2 and 4, against a hand-derived 16 that is wrong by ~4×), `reachability` (host-only: this workload's worst row is 0.115, a ~35× margin, so the defect is latent and the recorded mode figures stand). | [23 §2](../../docs/plans/transformer-layer-execution-studies/23-rules-and-open-items.md) |

**Why the last two both exist.** `air-opt` with a hand-built pass list answers *"does this pass
fire"*; it does not answer *"does this compile"*. The two diverge wherever a later pass rewrites
what was measured — `air-to-aie` normalizing external callee signatures is the case that cost this
plan a spec claim. Match the probe's altitude to the claim.
