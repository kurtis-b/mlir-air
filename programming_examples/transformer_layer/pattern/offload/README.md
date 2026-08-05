# `offload` — the host owns the layer, one GEMM per dispatch

The host-mediated extreme of the Phase E execution-strategy taxonomy. The host
holds every intermediate in RAM and sends one GEMM at a time to the device:
six one-step `KernelCache.run_sequence` calls —

```
q_proj  k_proj  v_proj  output_proj  up_proj  down_proj
```

— each recording one `DispatchVector` row with one host submission holding one
runlist entry. Summed by the driver that is six submissions and six entries:
the mode aggregates **nothing**, which is its clause in the distinguishability
gate. Everything between the GEMMs — attention, softmax, both normalization
points, the GeLU, reshapes, residuals — is host torch. The mode exists to be
the expensive end of the comparison: the most submissions, the most sync
boundaries, no aggregation at all.

## This is a hybrid boundary, and the artifact says so

iron's offload dispatches **eight** GEMMs, including the two attention GEMMs.
On this device those two cannot be dispatched as registry GEMMs, and the
registry cannot be made to hold them: at the gate configuration they are
`4096 x 64 x 4096` (attn_scores) and `4096 x 4096 x 64` (attn_output), and
`gemm_config()` raises `KeyError` on both — there is no `K = 64` or `N = 64`
bf16-out row anywhere, and the C4 sweep derives K and N from
`FAMILY_HIDDEN x ROLE_KN_MULTIPLES` with a minimum hidden of 512, so no
`--family` can stage a 64 in either position. `attn_scores` would additionally
need `K = 64` against a minimum `tile_k_l2` of 256, which does not tile.

**So attention stays in host torch**, alongside the softmax, scaling,
masking, reshapes, normalization and residuals that were always going to.
That makes `offload` a *hybrid* boundary rather than a pure per-GEMM device
implementation, and the artifact records `attention_path:
"host_torch_fp32_blocked"` so the mode's own record says which boundary it
actually drew. Three consequences, from 08c:

- It does not weaken the mode's role: six host submissions and six sync-heavy
  round trips is still strictly the most host-mediated of the four, and the
  distinguishability gate asks for an ordering, not the number eight.
- It is the numerically conservative choice: host FP32 attention lands closer
  to the FP32 oracle than the device path does, so this mode has the most
  headroom of the four.
- The plan documents that said eight are corrected; the artifact contract
  never counted to eight.

## Blocked attention, shared with `runlist`

`pattern/blocked_attention.py` ports iron's `_blocked_attention` /
`_resolve_query_block_size` pair, which `offload` and `runlist` share in iron
so both modes block identically — the port keeps that sharing by putting the
module in `pattern/`, not here. Above a scratch cap
(`MAX_ATTENTION_SCRATCH_BUFFER_BYTES = 3 GiB`) the computation blocks over
query rows so a long sequence never materializes the full
`[heads, seq, seq]` f32 score tensor. At the gate configuration it does not
trigger — 4096 x 4096 x 12 heads in f32 is ~805 MB — but the ladder runs to
16384, where the same tensor is ~12.9 GB against 31 GB of host RAM; there the
block resolves to 4096 rows. The recorded `query_block_size` in the artifact
is the value the run actually used.

## The oracle-independence rule

The mode computes; the oracle checks; they may not share arithmetic. This
mode does more host math than any other, so every host stage is torch —
`F.layer_norm`, `F.gelu(approximate="tanh")`, `blocked_attention`'s torch
softmax — while the oracle's boundaries are the numpy operator references
behind `pattern/reference.py`. `offload.py` imports the golden draws and the
boundary names from the reference and none of its arithmetic. A stage whose
"actual" and "expected" are the same function call compares a value against
itself and passes no matter what is wrong with it.

## What the numbers look like, and why

Six recorded `DispatchVector` rows, one per GEMM, in dispatch order:

- Every row: 1 submission, 1 runlist entry. Batching them into one runlist
  would record one submission over six entries — that is `coarse`, not this
  mode, and `run_sequence` under the ELF ABI merges everything it is given
  into one submission, which is why the six dispatches are six SEPARATE
  calls.
- q/k/v/output_proj ride one shared `4096x768x768` drain ELF (1 `air.launch`
  each); up_proj is its own `4096x768x3072` drain ELF; down_proj is a
  `4096x3072x768` fused-cast ELF (2 launches: GEMM into an f32 scratch, then
  the cast). Sharing a *binary* is not aggregation — each dispatch is its own
  submission — and three ELF compiles instead of six is real minutes on every
  gate.
- Each drain dispatch syncs 3 buffers (A and B up, C back); the fused-cast
  dispatch syncs 4 (A, B and the zero-filled f32 scratch up, C back). x is
  re-uploaded for each of q/k/v and every weight is uploaded per dispatch:
  **six BOs of weights re-uploaded per layer is the mode being itself — do
  not optimize it.** Nothing is declared static for the same reason.

The fault-injected twin totals identically — injection perturbs one input
element after the reference exists and never touches the dispatch path — and
the driver requires that equality as proof the vectors were measured.

## What it costs

A third full-layer run in the lit suite (`run_npu2_offload_peano.lit` beside
`block` and `coarse`), compiling three ELFs into its own `offload_cache/` —
the cache directory is chosen by NAME, so modes must never share one (two
modes pointed at one directory can trade ELFs whose fingerprints happen to
agree, attributing numerically valid output to the wrong execution boundary).
`offload_cache/` is gitignored and in `make clean`, in the same commit that
created it, because the driver's negative control runs `opcheck.py` from the
source directory and the cache lands there — the leak D2's `block_cache/`
had. This mode is also intrinsically noisy in LATENCY (~10x the run-to-run
drift of the others; an XRT version change alone has moved it 19–39% at
`seq_len >= 4096`) — that is a property of host-mediated dispatch the
study's comparator tolerances already budget for, and it does not affect this
gate, which is numerical rather than temporal.
