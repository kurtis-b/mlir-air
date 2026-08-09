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

iron's offload dispatches **eight** GEMMs, including the two attention GEMMs,
and `[2026-08-08]` **so does this one.** At the gate configuration they are
`4096 x 64 x 4096` (attn_scores) and `4096 x 4096 x 64` (attn_output), and
`gemm_config()` does raise `KeyError` on both: there is no `K = 64` or
`N = 64` bf16-out row anywhere, and the C4 sweep derives K and N from
`FAMILY_HIDDEN x ROLE_KN_MULTIPLES` with a minimum hidden of 512, so no
`--family` can stage a 64 in either position.

**That is a catalogue constraint, not a hardware one**, and this README used
to draw the wrong conclusion from it. Both shapes are measured passing on real
NPU2 at every rung of the ladder, at 0% allowed mismatch over the full output;
`attn_output` passes by all three GEMM methods. The tiles are injected through
the `gemm_spec_fn` escape hatch every builder ships and recorded as
`gemm_spec_source: registry+injected`. A further claim that travelled with the
old one — that `attn_scores` "would need `K = 64` against a minimum
`tile_k_l2` of 256, which does not tile" — is false too: `tile_k_l2 = 64` is
what passes, and at K = 64 it is forced.

**So attention is on the device**, and only the softmax between the two
matmuls stays on the host, with both LayerNorms and the GeLU. That is the
corrected taxonomy's rule — every LINEAR operator on the NPU, every NON-LINEAR
one on the host — rather than the *hybrid* boundary this mode used to draw.
The artifact records `attention_path:
"device_gemm_host_softmax"` so the mode's own record says which boundary it
actually drew — a results tree mixing that value with the old
`host_torch_fp32_blocked` is mixing two different modes, and the sequence
ladder's slopes split on exactly this covariate.

**What it costs, which is the mode's own result.** A host softmax between two
device matmuls means the full `[seq, seq]` score matrix crosses DRAM twice per
head: 970,457,088 bytes over 30 dispatches against 139,984,896 over six, a
**6.9x** increase. This mode is therefore much slower at 4096 than the
six-GEMM form it replaces. That is what the partition costs when the
non-linear operator sits between two matmuls; it is priced, not broken.

**What it did not cost: accuracy.** Measured at the gate configuration,
`attn_context` needs `atol` 8.800e-05 against the 1.0e-03 the boundary
allows — an 11.4x margin — and the layer output needs 5.788e-02 against the
1e-1 hard ceiling, a **1.73x** margin, wider than `block` (1.35x), `runlist`
(1.41x) or `fused` (1.27x). No tolerance was widened for this change.

Three consequences of the OLD host-attention boundary, from 08c, kept because
they explain the shape of the mode's history:

- It did not weaken the mode's role: six host submissions and six sync-heavy
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

## A reused `hw_context` corrupts an ELF's second execution — measured

The first bring-up run of this mode failed with a signature worth recording:
`q` (the first execution of the shared proj ELF) clean at the GEMM's own
9.6e-3 relative error, `k` and `v` (its second and third executions) wrong at
3.56e-1 — uniformly across rows and columns, roughly one third of the
reduction lost. A controlled experiment on the same cached ELFs pinned it
down:

- second execution, **same inputs**: wrong (3.56e-1) — so not a data issue;
- the wrong output matches neither the previous weights' product (1.42) nor
  the previous result (1.42) — so not a stale-B or stale-C readback;
- second execution in a **fresh `hw_context`**: clean (9.6e-3), every time.

So re-executing one of these runtime-tiled GEMM ELFs in a reused context
returns wrong numbers from the second execution onward; the corruption is
device-side state the ELF leaves behind. Nothing else in the example ever
re-executes a GEMM ELF across submissions — the block's GEMM ELFs each run
once per process, and its re-executed addnorm ELF (no runtime loop tiling)
re-runs clean — which is why this mode is where the failure surfaced.
`_evict_context` in `offload.py` therefore reloads the context (and drops
the BO pools, whose buffers were allocated against the evicted backend's
device wrapper) before every dispatch. That cost is charged to latency, not
to the dispatch vector: nothing is static, so the sync and byte counts are
identical either way. **If E4's `runlist` re-executes a GEMM ELF inside a
single runlist, measure before assuming either behaviour.**

## What the numbers look like, and why

The measured stage table at the gate configuration (clean run): every
boundary at `n_mismatch 0`, end-to-end `mean_rel_L1 1.396e-2` with
`atol_required 5.489e-2` against the 1e-1 ceiling — a 1.82x margin, against
the all-device block's 1.35x (1.688e-2 / 7.398e-2). Host FP32 attention and
host norms land closer to the FP32 oracle than the device path does, as 08c
predicted: `attn_context` needs only 3.5e-5 of absolute tolerance here where
the fused device attention needs 2.3e-4, and `hidden` 1.3e-3 where the
device addnorm needs 1.2e-2.

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
