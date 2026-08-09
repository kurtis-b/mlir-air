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
| `probe_backend_preset_hardware.py` | **`runtime_loop_tiling_sizes` is NOT inert, and `omit_pingpong` is irrelevant at this shape.** A replicated 2×2 on `mha_out_proj` @4096: `[1,1]` passes 3/3 with byte-identical statistics whether ping-pong is on or off; `[2,2]` gives `ERT_CMD_STATE_TIMEOUT` 3/3, likewise either way. The shipped preset re-run through the same harness is the control and passes, so the timeouts belong to the preset. **Refutes [26 §4](../../docs/plans/transformer-layer-execution-studies/26-mode-rebuild-feasibility.md)**, which read "identical lowered IR" as "inert" from a compile-only spike. | [26 §4](../../docs/plans/transformer-layer-execution-studies/26-mode-rebuild-feasibility.md) |

| `probe_context_reuse.py` | **`_evict_context`'s corruption is one CELL, not a class — and the ABI `offload`'s next phase needs is a clean one.** A 2×2 on `q_proj` 1024×768×768, four executions of one artifact on one input pair: `elf`+`[2,2]` diverges from its own run 1 by **3.8141e-01** (replicated 2/2), while `elf`+`[1,1]`, `xclbin`+`[2,2]` and `xclbin`+`[1,1]` are all bit-identical 4/4. The evicting control is clean and reproduces the `9.6e-3` reference error the mode's docstring cites. Also: the corruption **does not accumulate** — runs 2-4 are identical to each other. Second independent hardware refutation of "`runtime_loop_tiling_sizes` is inert". | [27](../../docs/plans/transformer-layer-execution-studies/27-common-ladder-result.md) |

| `probe_one_xclbin_n_streams.py` | **N instruction streams under one xclbin WORKS, and needs two distinct identifiers per stream.** Two GEMMs of different shape chained with `--xclbin-input` both execute correctly from one `hw_context`, either compile order. Requires a distinct `instance_name` (the loader matches by substring, `xrt.py:634`) **and** a distinct `kernel_id` — the merged `AIE_PARTITION` routes a kernel to its PDI by `dpu_kernel_ids` and every AIR compile defaults to `0x901`. Both-directions control: collide the ids and the second kernel times out at one shape, returns **garbage at `mean_rel_L1` 1.41 with no error raised** at the other. The `insts.bin` is byte-identical either way, so the instruction stream is a red herring and the discriminating field needs `xclbinutil --dump-section AIE_PARTITION:JSON`. | [26 §Sizing](../../docs/plans/transformer-layer-execution-studies/26-mode-rebuild-feasibility.md) |

**Why `probe_accum_hoist.py` and `probe_accum_aircc.py` both exist.** `air-opt` with a hand-built
pass list answers *"does this pass fire"*; it does not answer *"does this compile"*. The two diverge
wherever a later pass rewrites what was measured — `air-to-aie` normalizing external callee
signatures is the case that cost this plan a spec claim. Match the probe's altitude to the claim.

**Why `probe_backend_preset_hardware.py` and `probe_context_reuse.py` both exist.** They are the
same knob caught doing two different things. The first says `[2,2]` HANGS FlashAttention at 4096;
the second says `[2,2]` leaves context-corrupting residue in a plain projection GEMM under the ELF
ABI. Neither subsumes the other, and a fix that addressed only one would leave the other live.
