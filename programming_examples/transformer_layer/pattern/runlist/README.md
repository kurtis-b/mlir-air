# `runlist` — the fine-grained point of the execution-strategy taxonomy

The D2 `encoder_bert` layer with `coarse`'s dispatch schedule held fixed and
every one of its dispatch units refined into single-operator kernels,
aggregated into runlists: iron's `pattern/runlist` re-expressed over
`KernelCache` and the Phase B dispatch model. `prepare_runlist` is the mode's
entry in the `SPECS` catalogue; `make check-runlist` / `check-runlist-fault`
run it.

## The decomposition: a strict refinement of `coarse`'s schedule

391 entries over five runlists, each entry one operator kernel. The left
column is `coarse`'s measured dispatch schedule (4 submissions, 131 entries);
the right is this mode, every unit split into its constituent operators **at
the same granularity**:

| `coarse`'s unit | entries | this mode's refinement | entries |
|---|---|---|---|
| `qkv_proj` (fused GEMM) | 1 | runlist 1: `q_proj`, `k_proj`, `v_proj` (registry GEMM 4096x768x768, one module compiled to three OWN ELFs — see footguns) | 3 |
| `mha_out_proj` (fused) | 1 | **host torch** blocked attention (`pattern/blocked_attention.py`, shared with `offload`), then runlist 2: `output_proj` (OWN ELF of the same proj module) | 1 |
| 64 × `addnorm` ln1 (pre-add, row-banded) | 64 | runlist 3: 64 × (residual add → LayerNorm → gamma multiply), `builders/elementwise_add.py` + `layer_norm.py` + `elementwise_mul.py` at `[64, 768]` | 192 |
| `ffn` (fused up+gelu+down) | 1 | runlist 4: `up_proj` (4096x768x3072), GeLU (`builders/gelu.py`), `down_proj` (4096x3072x768) | 3 |
| 64 × `addnorm` ln2 (pre-add, row-banded) | 64 | runlist 5: 64 × (residual add → LayerNorm → gamma multiply) | 192 |

Every coarse dispatch unit maps onto one or more finer units, so
`runlist_entries > coarse` — the one ordinal clause this mode owns — holds
**by construction**: 391 against 131. The band size is IMPORTED from
`builders.block.norm_rows` (the L1 cap `coarse`'s fused `addnorm` measured:
64 rows at width 768), never re-derived or tuned here, so the two modes share
one schedule as a matter of code rather than coincidence.

Intermediates inside a runlist stay device-resident: q/k/v never touch the
host before the attention readback, each band's residual sum feeds its
LayerNorm and gamma multiply on device, and `ffn_up`/`ffn_gelu` chain into
the down projection. That cross-artifact chaining under the Phase B allocator
is what the mode measures; the five submission seams are the same whole-BO
restage seams `coarse`'s four are, plus one more for host attention (below).

## Why the norm chains are banded when the decomposed kernels could stream

The decomposed `add`/`ln`/`mul` kernels have no L1 cap forcing 64-row
dispatches — each can walk all 4096 rows in one launch, and the first
structure this mode tried did exactly that: 13 entries over two runlists.
That landed BELOW `coarse`'s 131 and failed the ordinal clause, and the
failure was diagnostic, not incidental: the streaming structure changed TWO
variables at once — operator granularity AND the dispatch schedule — and at
the normalization points it was 64× *coarser*-grained than `coarse` itself
(one streaming launch where `coarse` dispatches 64 bands), so its entry count
measured the schedule change, not the decomposition. A "fine-grained" mode
whose dispatch units are the largest the kernels allow is not a point on the
granularity axis the taxonomy orders.

Holding the schedule at `coarse`'s own and splitting each fused `addnorm`
band call into its three constituent operators makes the comparison
controlled: the two modes differ in exactly one variable, operator
granularity, and the entry count measures it. The measured cost of that
control is carried in the vector (next section) rather than hidden: banding
restages the norm-chain operands through the host, so sync boundaries rise
from the streaming structure's 21 to 403 and bytes from ~153 MB to ~165 MB —
still below `coarse`'s 402-sync/203-MB shape on bytes, one above it on sync.

iron's encoder runlist is 16 entries at this sequence length; the three
missing here are its on-device attention interior — `k_transpose`,
`attn_scores`, `attn_scale`/`attn_softmax`/`attn_output` — see the next
section. Its count is not carried across for the same reason `coarse`'s 131
is not iron's 12: entry counts are re-derived at `baseline_768` under this
hardware's dispatch caps (08d §Do not carry iron's entry count across).

## Why attention is host torch, and what that removes

> **`[2026-08-08]` This section describes what this mode does TODAY, and it is
> not what the corrected taxonomy requires.** `runlist` is defined as every
> operator individually **on the device, nothing on the host**, so host
> attention is a gap to close, not a property. The reasoning below is also
> wrong about why: the missing registry rows are a CATALOGUE constraint, not a
> hardware one — both attention shapes are measured passing on real NPU2, and
> `offload` now dispatches them (see `pattern/offload/README.md`). What
> `runlist` additionally needs, and `offload` does not, is a **device
> softmax**. `[2026-08-09]` That operator now exists and is gated —
> `builders/softmax.py`, `run_npu2_softmax_peano.lit`, three shapes including
> 64x4096 — so this mode is unblocked. Until it is rebuilt it prices host torch
> rather than reconfiguration: 24.15 ms of host attention at 1024, 47.8% of its
> total.
>
> **Three sizing facts for whoever does the rebuild**, established while
> building the operator:
>
> - At attention width the softmax module builds with `rows_per_call = 2` and
>   no higher: three `[rows_per_call, cols]` bf16 buffers live in L1, so 2 needs
>   48 KiB and 4 would need 96 KiB against a 64 KiB L1.
> - **The row loop is inside the herd body**, so one softmax is ONE `air.launch`
>   regardless of row count — 256 loop trips per tile at `[4096, 4096]`, not 256
>   dispatches. The entry-count consequence of moving attention on-device is
>   therefore 12 softmax dispatches plus 24 GEMMs, not thousands, which is what
>   this mode's identity (391 entries over 5 runlists) has to be re-derived
>   against.
> - `[4096, 4096]` is **compile-verified only**. The hardware-verified
>   attention-width row is `64x4096`; the full per-head score matrix is 32 MiB
>   in and 32 MiB out and has not been run. Do not treat "it builds" as "it
>   passes" — that inference is exactly what doc 26 §4 got wrong.
>
> The GEMM tiles are already measured and are in `pattern/offload/offload.py`
> as `ATTENTION_GEMM_TILES`; inject them the same way rather than re-searching.

The two attention GEMMs are `4096 x 64 x 4096` and `4096 x 4096 x 64`; no
`K = 64` or `N = 64` bf16-out row exists in the registry and the C4 sweep
cannot stage one (08c has the derivation). So attention runs on the host
through `blocked_attention` — the SAME implementation and query blocking
`offload` used before its rebuild (08d work item 4). This is the one submission seam a whole-BO
argument does not explain, and it is why "one runlist" — the mode's premise
when attention is on device, as it is in iron's 29-kernel decomposition — is
not reachable on this hardware: a host stage between the projections and the
output projection forces at least two submissions before banding adds its
restage seams.

That decision removes iron's attention-interior entries, including
`k_transpose`. A device transpose whose only consumer is a host `torch`
matmul that re-layouts its operands anyway would add an entry while measuring
nothing, so it is not dispatched; the `transpose` operator (new in this
phase, `builders/transpose.py`) is validated standalone through `opcheck.py`
instead, ready for any future mode whose scores GEMM is on device.

## The measured dispatch vector

Driver-summed totals for one layer (clean and fault-injected runs identical;
the six literals are pinned in `run_npu2_runlist_peano.lit`, both halves):

```
host submissions   5      runlist entries  391     air launches     14
herd launches    404      sync boundaries  403     bytes    165,347,328
```

Against `coarse`'s 4 / 131 / 12 / 146 / 402 / 202,902,528: more entries (the
ordinal clause, by construction), more submissions (the attention seam plus
the banding restages), 2.8× the herd launches (384 of the 404 are the two
chains' band launches), one more sync boundary, and fewer bytes — the
streaming q/k/v/attention path is leaner than `coarse`'s fused mha even with
the norm chains restaged, and the gamma broadcasts shrink from `[4096, 768]`
to one shared `[64, 768]` band per norm point.

All ten stage boundaries clean; layer output mean_rel_L1 1.732e-2 at
atol_required 7.077e-2 — a 1.41x margin under the 1e-1 ceiling, between
`offload`'s 1.82x (host norms) and the block's 1.35x (device fused norms),
which is where device norms + host f32 attention should land. The banded
chains produce bit-identical boundary tensors to the streaming structure
(row-wise and element-wise operators do not see the banding), so these
figures match the first structure's exactly — the restructuring moved
dispatch structure, not arithmetic.

## Footguns (each cost time; read before editing)

- **A runtime-tiled GEMM ELF corrupts on re-execution INSIDE a single
  runlist too — measured here, the new result this mode adds to `offload`'s.**
  The first bring-up shared one proj ELF for q/k/v in runlist 1: execution 1
  (q) came back clean at 9.3e-3 mean_rel_L1, executions 2 and 3 (k, v) at
  3.539e-1 and 3.561e-1 — the exact signature `offload` measured across
  submissions (3.56e-1). `offload`'s fix, evicting the context between
  dispatches, is structurally unavailable inside one runlist: entries of one
  artifact share its `hw_context` by construction. So the four projections
  are ONE module compiled to FOUR artifacts (`rl_gemm_{q,k,v,o}_proj_…`),
  each with its own context, and no GEMM ELF ever executes twice. Four
  compiles of one module per clean cache is the cost; do not "deduplicate"
  them back.
- **The band `add`/`ln`/`mul` ELFs execute 64 times per chain and 128 times
  per layer in one context each, and that is measured clean.** No runtime
  loop tiling, same class as the block's 64-fold re-executed `addnorm` ELF —
  every stage boundary downstream of them is exact at the same tolerances the
  streaming structure met. The GEMM-class failure does not extend to them.
- **The gamma multiply's second operand is a materialized broadcast, shared
  across bands.** `elementwise_mul` takes two full `[64, 768]` tensors (its
  docstring records why there is no broadcast form), so each LayerNorm weight
  is tiled to ONE `[64, 768]` band on the host (`broadcast_row_weight`) and
  declared static + content-keyed — every band multiplies by the same rows,
  so one 96 KB buffer serves all 64 multiplies, where the streaming structure
  materialized 6 MB per norm point. Under fault injection the content key
  changes and the perturbed weight is re-uploaded; nothing special-cases the
  injected path.
- **Band inputs are contiguous copies, never views.** A `BufferSpec` is sized
  from the array and the host write reads it flat, so a non-contiguous slice
  would upload the wrong bytes without complaining — the same rule
  `builders/block.py::_sequence_norm` records.
- **`hw_context` demand is 10 concurrent, measured against a ceiling of 32.**
  By the time runlist 5 executes, the three projection contexts from runlist 1
  are still loaded and seven more have joined them (o_proj, add, ln, mul, up,
  gelu, down): ten DISTINCT designs resident simultaneously, demonstrated by
  the mode running rather than by a synthetic probe. The 32 ceiling was probed
  with 4 cycled ELFs (08 §Risks), so this is also the first data point with
  ten distinct designs — comfortably inside, and the margin no longer rests
  on the cycled-ELF assumption alone. iron's 29-context appetite does not
  arise here because attention is host and the band operators share their
  ELFs across all 128 band entries.
- **`RUNLIST_CACHE_DIR` is this mode's own**, in `transformer_layer/.gitignore`
  and the Makefile `clean` target. Modes never share a cache directory:
  `KernelCache` picks it by name, and two modes pointed at one can trade
  fingerprint-matching ELFs, attributing valid output to the wrong execution
  boundary.
- **The decomposed norm has one more bf16 rounding than the oracle's path**
  (the residual sum is rounded before the statistics are taken — the same gap
  the fused `addnorm` kernel has, plus the rounding between `ln` and `mul`).
  It fits the same `hidden`/`output` stage tolerances; if a future width does
  not, that is a property of the decomposition to report, not a tolerance to
  widen.
