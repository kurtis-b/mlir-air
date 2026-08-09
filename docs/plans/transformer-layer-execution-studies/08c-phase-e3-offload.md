# 08c — Phase E3: `offload`

The host-mediated extreme of the taxonomy. The host owns the layer, holds every intermediate, and
sends one GEMM at a time to the device. It exists to be the expensive end of the comparison: the most
submissions, the most sync boundaries, no aggregation at all.

Read [08b](08b-phase-e2-coarse-and-instrumentation.md) first. It settles the artifact contract —
operator name, forced shape, stages, `dispatch_vectors` — and this sub-phase satisfies exactly that
contract with a different execution boundary behind it.

## `offload` dispatches six GEMMs, not eight, and this is a decision rather than a shortcut

[08](08-phase-e-execution-strategies.md) originally listed eight offloaded GEMMs:

```
q_proj  k_proj  v_proj  attn_scores  attn_output  output_proj  up_proj  down_proj
```

**Two of those cannot be dispatched as registry GEMMs on this device, and the registry cannot be
made to hold them.** At `baseline_768`, `seq = 4096`:

```
attn_scores   4096 x   64 x 4096      attn_output   4096 x 4096 x   64
```

> **`[superseded 2026-08-08]` The registry facts below are right; the conclusion drawn from them
> is not.** No such row exists and the sweep genuinely cannot stage one — but that is a
> **catalogue** constraint, not a hardware one. Both shapes are measured passing on real NPU2 at
> every rung of the ladder, at 0% allowed mismatch over the full output, and `attn_output` passes
> by all three GEMM methods. `offload` now dispatches both, with tiles injected through the
> `gemm_spec_fn` escape hatch; the corrected mode is gated in
> `run_npu2_offload_peano.lit` at 30 dispatches. One further claim in the paragraph below is
> simply false: `attn_scores` does **not** need `tile_k_l2` 256. `tile_k_l2 = 64` is what passes,
> and at K = 64 it is forced, because K admits no other L2 tile.

`gemm_config()` raises `KeyError` on both — there is no `K = 64` or `N = 64` bf16-out row anywhere
(`registry_lookup.py:115`). Nor can the C4 sweep produce one: `sweep_families.py` derives K and N
from `FAMILY_HIDDEN × ROLE_KN_MULTIPLES` with a minimum hidden of 512 and M from `SEQ_LADDER`, so no
`--family` stages a 64 in the K or N position; and `attn_scores` would need `K = 64` against a
minimum `tile_k_l2` of 256, which does not tile.

**So attention stays in host torch**, alongside the softmax, scaling, masking, reshapes,
normalization and residuals that were always going to. The device dispatches:

```
q_proj  k_proj  v_proj  output_proj  up_proj  down_proj
```

Record `attention_path` in the artifact so the mode's own record says which boundary it actually
drew, and say plainly in `offload/README.md` that this makes `offload` a *hybrid* boundary rather
than a pure per-GEMM device implementation. That is a real qualification on what the mode measures
and it must not be discovered later from the code.

Three consequences worth stating up front:

- It does not weaken the mode's role in the comparison. Six host submissions and six sync-heavy
  round trips is still strictly the most host-mediated of the four, and the distinguishability gate
  asks for an ordering rather than the number eight.
- It is the numerically conservative choice. The layer's `atol` sits at the hard `1e-1` ceiling at
  1.35× its measured requirement, and host FP32 attention lands closer to the FP32 oracle than the
  device path does. This mode has the most headroom of the four, not the least.
- **`03` and `13` said eight too**, and both are being corrected alongside this document. If you
  find a third place that still says eight, report it in `work_not_completed`.

## The rule that decides whether this mode measures anything

**The mode computes; the oracle checks. They may not share arithmetic.**

`pattern/reference.py` exports per-boundary helpers — `chunked_attention_reference`,
`gelu_tanh_reference`, `addnorm_pre_add_reference` and the rest — and this mode does more host math
than any other, so the temptation to call them is real and it would be fatal. An `attn_context` stage
where the mode's "actual" and the oracle's "expected" are the same function call compares a value
against itself and passes no matter what is wrong with it.

What you may import from `pattern/reference.py`: `generate_golden_reference`, `fuse_qkv_weight`,
`WEIGHT_DRAW_ORDER`, `ENCODER_BOUNDARIES` — the inputs, the weight layout, the draw contract, the
names. What you may not: anything that computes a boundary.

This is the same hazard `pattern/test_reference.py` exists to police from the other side. Expect a
reviewer to look for it.

## Blocked attention

Port `_blocked_attention` and `_resolve_query_block_size` from iron. Above a scratch threshold the
computation switches to blocking over query blocks, because a long sequence cannot materialize the
full score matrix. iron's constants:

```
MAX_ATTENTION_SCRATCH_BUFFER_BYTES = 3 GiB
MIN_BLOCKED_QUERY_BLOCK_SIZE       = 256
```

At the gate configuration this does not trigger — 4096 × 4096 × 12 heads in fp32 is about 805 MB —
but the ladder runs to 16384, where the same tensor is about 12.9 GB, and this machine has 31 GB of
RAM shared with everything else. The threshold is the point.

`offload` shares this logic with `runlist` in iron. Keep the sharing so both mode directories block
attention identically; put it somewhere both can import rather than in `offload/` alone.

## Instrumentation: use `run_sequence`, not `load_and_run`

[03](03-measurement-model.md) describes this mode as `KernelCache.load_and_run` per GEMM. That
predates Phase B. `load_and_run` produces no `DispatchVector`, and a mode with no vector cannot be
compared with the other three.

Dispatch each GEMM as a **one-step `run_sequence` call**. Six calls, six recorded vectors, each with
`host_submissions_per_layer == 1` and `runlist_entries_per_submission == 1.0`. Summed by the driver
that is six submissions and six entries — which is what "aggregates nothing" means, and the
distinguishability gate checks exactly that equality.

Do not batch the six into one runlist to make a number look better. If they were batched this would
be `coarse`.

## Work items

1. `pattern/offload/` — the mode module, `README.md`, its own `KernelCache` directory added to
   `transformer_layer/.gitignore` and the `clean` target in the same commit.
2. The six GEMM dispatches, each a one-step `run_sequence`.
3. Host torch for everything else, written independently of `pattern/reference.py`'s boundary
   helpers.
4. `_blocked_attention` / `_resolve_query_block_size`, placed so `runlist` can share them.
5. An `offload` operator spec in the `SPECS` catalogue, through the `dispatch` seam, recording
   `execution_mode: "offload"` and `attention_path`.
6. `run_npu2_offload_peano.lit` — both recipes in one file, clean and
   `--fault-inject input --expect-failure`, following `run_npu2_block_peano.lit`.

## Gate

```
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  ninja -C build-xrt check-programming-examples-transformer-layer
```

Every test in the suite passes on real hardware.

The driver then, independently: re-derives `offload`'s verdict from `n_mismatch` / `ref_dtype` /
`rtol` / `atol`; requires exactly one fresh `offload` result at the forced configuration
(`seq_len 4096, emb_dim 768, ffn_dim 3072, 12 heads, head_dim 64`) with `n_elements == 3145728` and
≥8 distinctly-named clean stages; validates the `dispatch_vectors` contract; requires
`runlist_entries == host_submissions` summed over the recorded vectors, and at least six of each;
re-runs `offload` under `--fault-inject input` and requires it to **fail**; and requires the fault
run's summed vector totals to equal the clean run's.

## Risks

- **This mode is intrinsically noisy** — roughly ten times the run-to-run latency drift of the
  others, and an XRT version change alone has moved it 19–39% at `seq_len >= 4096` while leaving the
  others within 0.6%. That is a property of host-mediated dispatch, not a bug, and
  [03](03-measurement-model.md)'s wider comparator tolerances for it must be preserved. It does not
  affect this sub-phase's gate, which is numerical rather than temporal.
- **Host memory, not device memory, is the ceiling here.** Every intermediate lives in host RAM at
  once. If the gate configuration is tight, say so; do not shrink the shape.
- **Six BOs of weights re-uploaded per layer.** That is the mode being itself. Do not optimize it.
