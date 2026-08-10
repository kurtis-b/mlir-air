# `offload` — every linear operator on the device, reconfiguration minimized

> **`[2026-08-09]` The title and this opening were rewritten.** They described
> "the host-mediated extreme … six one-step `run_sequence` calls … the mode
> aggregates nothing", which is the **superseded** taxonomy plus a dispatch
> count that two corrections have since moved. Sections below carry their own
> dates; where one still argues from the old framing it says so.

The mode that **MINIMIZES reconfiguration**, which is one axis of the corrected
taxonomy (`docs/plans/transformer-layer-execution-studies/03-measurement-model.md`
§The taxonomy). Its host/device split is decided by **linearity**:

- **every LINEAR operator on the NPU** — the six projections *and* both
  attention matmuls, dispatched per head;
- **every NON-LINEAR operator on the host** — the softmax between the two
  attention matmuls, both LayerNorms, the GeLU.

At the gate configuration that is **30 dispatches**: 6 projections + 2 attention
matmuls × 12 heads. Each is one one-step `KernelCache.run_sequence` call
recording one `DispatchVector` row, so the summed vector is 30 submissions over
30 entries — the mode still aggregates **nothing**, and that part of the old
framing survives the correction.

## Two packaging paths, and the difference between them IS the mode

`[2026-08-09]` The same 30 dispatches are served two ways. They compute the same
thing and produce an identical dispatch vector; what differs is how often the
array is configured, which is the axis the mode is defined by.

| | ELF path (**default**) | shared xclbin (`AIR_OFFLOAD_SHARED_XCLBIN=1`) |
|---|---|---|
| binaries | 5 xclbins | **1**, five shapes chained |
| `context_loads` per layer | **30** — `_evict_context` tears down and reloads before every dispatch | **1**, plus 4 kernel attaches |
| dispatch vector | identical | identical |
| builds at | 512 · 1024 · 4096 | **512 · 1024 only** |

Both are gated: `run_npu2_offload_peano.lit` pins the reconfiguration counters
on each, and the shared recipe was verified failing against an ELF-packaged run
of the same mode at the same length. `make check-offload-shared` runs it.

**Three things to know before touching either.**

- **The shared path needs THREE distinct identifiers per stream** —
  `kernel_name`, `instance_name` and `kernel_id` — and only the first fails
  loudly. A duplicate `instance_name` returns the wrong program via the loader's
  substring match; a duplicate `kernel_id` runs the second kernel against the
  first's array configuration and returns garbage at `mean_rel_L1` 1.41 **with
  no error raised**. `compile_shared_xclbin` validates all three up front.
- **It builds only where every module is SINGLE-LAUNCH.** At 4096 the
  down-projection is a two-launch `fused-cast`, and `XRTBackend.compile`'s fixed
  `insts="air.insts.bin"` — passed as `-i` on the xclbin branch, omitted
  entirely on the ELF branch — makes aiecc refuse with *"edge 'air.insts.bin'
  produced duplicate output path"*. That is why the shared gate is at 1024 and
  why the default has not moved.
- **The two paths get SEPARATE cache directories** (`offload_cache` /
  `offload_shared_cache`), chosen by `OFFLOAD_CACHE_DIR` from the same env var.
  They emit artifacts with identical names over different ABIs, so one directory
  could hand a 30-reconfiguration run's artifact to the one-reconfiguration
  claim.

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

## Blocked attention — `[2026-08-09]` NO LONGER THIS MODE'S PATH

> **This section describes what `offload` used to do.** Since attention moved to
> the device (2026-08-08) this mode imports exactly one name from
> `pattern/blocked_attention.py` — `round_bf16` — and computes attention as
> per-head device matmuls with `_host_softmax_bf16` between them
> (`ATTENTION_PATH = "device_gemm_host_softmax"`). The blocking machinery below
> now serves **`runlist`**, which is why the module still lives in `pattern/`.
> Kept because the scratch-cap reasoning is what sized the host side of this
> mode and still explains `runlist`'s.

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

> **`[2026-08-09]` The accuracy figures in the next paragraph are the OLD
> host-attention form's** — `1.396e-2` / `5.489e-2` / 1.82x. With attention on
> the device the mode measures `atol_required` **5.788e-02** against the same
> 1e-1 ceiling, a **1.73x** margin (§This is a hybrid boundary, above, has the
> current numbers). The reasoning below — that host FP32 attention lands closer
> to the oracle — explains why the margin was *wider* before and is kept for
> that; it no longer describes what the mode does.

The measured stage table at the gate configuration (clean run): every
boundary at `n_mismatch 0`, end-to-end `mean_rel_L1 1.396e-2` with
`atol_required 5.489e-2` against the 1e-1 ceiling — a 1.82x margin, against
the all-device block's 1.35x (1.688e-2 / 7.398e-2). Host FP32 attention and
host norms land closer to the FP32 oracle than the device path does, as 08c
predicted: `attn_context` needs only 3.5e-5 of absolute tolerance here where
the fused device attention needs 2.3e-4, and `hidden` 1.3e-3 where the
device addnorm needs 1.2e-2.

**`[2026-08-09]` 30** recorded `DispatchVector` rows, in dispatch order — six
projections and two attention matmuls on each of twelve heads:

- Every row: 1 submission, 1 runlist entry. Batching them into one runlist
  would record one submission over many entries — that is `coarse`, not this
  mode, and `run_sequence` under the ELF ABI merges everything it is given
  into one submission, which is why the dispatches are SEPARATE calls.
- **Five compiled shapes, not five binaries per se.** q/k/v/output_proj ride one
  `4096x768x768` drain artifact (1 `air.launch` each); up_proj is
  `4096x768x3072` drain; down_proj is `4096x3072x768` **fused-cast (2
  launches**: GEMM into an f32 scratch, then the cast — this is the shape that
  blocks the shared chain at 4096); `attn_scores` is `4096x64x4096` and
  `attn_output` `4096x4096x64`, both drain, both on injected tiles. Sharing an
  artifact is not aggregation — each dispatch is its own submission — and five
  compiles instead of thirty is real minutes on every gate.
- **Under the xclbin ABI a cold dispatch does not produce the steady-state
  vector.** The first call uploads each artifact's instruction stream once
  (`sync_instruction_bos`); at 1024 that is `sync 95 bytes 99141520` against a
  steady `sync 90 bytes 99090432` — five artifacts, five sync boundaries, and
  exactly 51,088 bytes, the total size of the five cached `.insts.bin` files.
  The ELF path skips it (an ELF embeds its instructions) and reads `sync 90`
  from its first dispatch. Every recorded number in the study is steady-state.
- Each drain dispatch syncs 3 buffers (A and B up, C back); the fused-cast
  dispatch syncs 4 (A, B and the zero-filled f32 scratch up, C back). x is
  re-uploaded for each of q/k/v and every weight is uploaded per dispatch:
  **six BOs of weights re-uploaded per layer is the mode being itself — do
  not optimize it.** Nothing is declared static for the same reason.

The fault-injected twin totals identically — injection perturbs one input
element after the reference exists and never touches the dispatch path — and
the driver requires that equality as proof the vectors were measured.

## What it costs

A full-layer run in the lit suite (`run_npu2_offload_peano.lit` beside
`block` and `coarse`), compiling **five shapes** into its own `offload_cache/`
— the cache directory is chosen by NAME, so modes must never share one (two
modes pointed at one directory can trade artifacts whose fingerprints happen to
agree, attributing numerically valid output to the wrong execution boundary).
`offload_cache/` is gitignored and in `make clean`, in the same commit that
created it, because the driver's negative control runs `opcheck.py` from the
source directory and the cache lands there — the leak D2's `block_cache/`
had. `[2026-08-09]` The shared path adds a third recipe at 1024 and its own
`offload_shared_cache/`, both likewise gitignored and cleaned.

**This mode is the noisiest of the four, and `[2026-08-09]` the drift is now
known to be REMOVABLE rather than intrinsic.** Four interleaved walks with
`runlist` as a same-conditions control put the intra-walk spread at **316.9% /
134.1%** on the ELF path against **17.6% / 14.0%** on the shared xclbin, at seq
512, in both walks. So it is **not** "a property of host-mediated dispatch" as
this section used to say — switching packaging removes it.

**Which half of the switch does it is not established.** The env var changes the
reconfiguration *and* the ABI at once, and the control rules out environmental
drift rather than the ABI, so `_evict_context` is the leading candidate and not
a demonstrated cause. Two further caveats: the effect size is unstable day to
day (the 1024 rung did not reproduce its own recorded baseline), and the shared
path costs ~20% of best-case latency at 512. None of it affects this gate, which
is numerical rather than temporal. Full table:
`docs/plans/transformer-layer-execution-studies/29-offload-n-streams.md`.
