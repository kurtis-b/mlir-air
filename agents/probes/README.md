# Measurement probes

One-off tools that produced measurements the specs in
`docs/plans/transformer-layer-execution-studies/` rest on. **Not tests** — nothing runs them in CI.

Run from a scratch directory with the toolchain on PATH (they write `air.mlir`, `*.o` and
`air_project/` into the cwd):

```bash
. agents/scripts/port-loop/lib-env.sh && pl_env_ensure   # needs log_info/log_error shims if run bare
python3 /home/cj/mlir-air/agents/probes/<probe>.py
```

| probe | what it is for | cited by |
|---|---|---|
| `probe_r1_rung.py` | The R1 resident-FFN rung runner: builds one `(herd_x, down_K)` point of `builders/ffn_resident.py`, runs it through devq, reads the output BO back **on timeout** (`--dump-npz` makes determinism a claim about bytes). The seed of the supertile increment (queue item 9). | [49](../../docs/plans/transformer-layer-execution-studies/31-resident-tail-r1-record.md) · [52](../../docs/plans/transformer-layer-execution-studies/31-resident-tail-r1-record.md) |
| `probe_r1_emulate_shape.py` | Its host-side shape emulation (imported by the rung runner). | [49](../../docs/plans/transformer-layer-execution-studies/31-resident-tail-r1-record.md) |

**Retired probes** (2026-08-21 cleanup; every one is at git tag `pre-cleanup-20260821` under
`agents/probes/`, and the claim each established is recorded in the doc named): `probe_packet_streams`,
`probe_layer_norm_twopass_cost`, `probe_addnorm_variance_cliff` → doc 23; `probe_j7a_pipeline` → 21;
`probe_accum_hoist`, `probe_accum_aircc`, `probe_ffn_accum_bd_offset` → 22; `probe_backend_preset_hardware`
→ 26 §4; `probe_context_reuse` → 27; `probe_one_xclbin_n_streams` → 29; `probe_fused_resident_tail`,
`probe_ffn_resident_interior`, `probe_fuse_channels_sibling_nests` → 31/31a; `probe_r2_order_seam`,
`probe_r2_segment_budget` → 31b; `probe_r1_arrival_map`, `probe_aie_buffer_writer_race` → 49/52;
`probe_r1_staged_hidden` → 53; `probe_norm_tail_plane_major` (uncited).
