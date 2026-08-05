# `runlist` — the fine-grained point of the execution-strategy taxonomy

The D2 `encoder_bert` layer decomposed into single-operator kernels, aggregated
into runlists: iron's `pattern/runlist` re-expressed over `KernelCache` and the
Phase B dispatch model. `prepare_runlist` is the mode's entry in the `SPECS`
catalogue; `make check-runlist` / `check-runlist-fault` run it.

## The decomposition, and where each piece comes from

Thirteen entries over two runlists, each entry one operator kernel:

| # | entry | kernel | ELF |
|---|---|---|---|
| 1 | q_proj | registry GEMM 4096x768x768 | `rl_gemm_q_proj_4096x768x768` |
| 2 | k_proj | same module, OWN ELF (see below) | `rl_gemm_k_proj_4096x768x768` |
| 3 | v_proj | same module, OWN ELF | `rl_gemm_v_proj_4096x768x768` |
| — | attention | **host torch** (`pattern/blocked_attention.py`, shared with `offload`) | — |
| 4 | output_proj | same module, OWN ELF | `rl_gemm_o_proj_4096x768x768` |
| 5 | residual add | `builders/elementwise_add.py` | `rl_add_4096x768` |
| 6 | LayerNorm | `builders/layer_norm.py` (unweighted) | `rl_ln_4096x768` |
| 7 | gamma multiply | `builders/elementwise_mul.py` | `rl_mul_4096x768` |
| 8 | up_proj | registry GEMM 4096x768x3072 | `rl_gemm_up_4096x768x3072` |
| 9 | GeLU | `builders/gelu.py` | `rl_gelu_4096x3072` |
| 10 | down_proj | registry GEMM 4096x3072x768 | `rl_gemm_down_4096x3072x768` |
| 11 | residual add | same ELF as 5 | `rl_add_4096x768` |
| 12 | LayerNorm | same ELF as 6 | `rl_ln_4096x768` |
| 13 | gamma multiply | same ELF as 7 | `rl_mul_4096x768` |

Intermediates between entries in one runlist stay device-resident: `attn_out`
feeds the residual add without touching the host, `hidden` feeds both the
up-projection and the second residual add. That cross-artifact chaining under
the Phase B allocator is what the mode measures.

iron's encoder runlist is 16 entries at this sequence length. The three missing
here are its on-device attention interior — `k_transpose`, `attn_scores`,
`attn_scale`/`attn_softmax`/`attn_output` — see the next section.

## Why attention is host torch, and what that removes

The two attention GEMMs are `4096 x 64 x 4096` and `4096 x 4096 x 64`; no
`K = 64` or `N = 64` bf16-out row exists in the registry and the C4 sweep
cannot stage one (08c has the derivation). So attention runs on the host
through `blocked_attention` — the SAME implementation and query blocking
`offload` uses (08d work item 4), making the two modes' attention boundaries
identical by construction.

That decision removes iron's attention-interior entries, including
`k_transpose`. A device transpose whose only consumer is a host `torch`
matmul that re-layouts its operands anyway would add an entry while measuring
nothing, so it is not dispatched; the `transpose` operator (new in this phase,
`builders/transpose.py`) is validated standalone through `opcheck.py` instead,
ready for any future mode whose scores GEMM is on device.

## The measured dispatch vector, and the finding it carries

Driver-summed totals for one layer (clean and fault-injected runs identical;
the six literals are pinned in `run_npu2_runlist_peano.lit`, both halves):

```
host submissions   2      runlist entries   13     air launches     11
herd launches     26      sync boundaries   21     bytes    152,567,808
```

All ten stage boundaries clean; layer output mean_rel_L1 1.755e-2 at
atol_required 7.011e-2 — a 1.43x margin under the 1e-1 ceiling, between
`offload`'s 1.82x (host norms) and the block's 1.35x (device fused norms),
which is where device norms + host f32 attention should land.

**13 entries lands BELOW `coarse`'s 131**, and that is the honest number, not
an under-decomposition: this mode dispatches ten distinct operator kernels
where `coarse` dispatches five. `coarse`'s count is dominated by one
operator's hardware cap — 128 of its 131 entries are `addnorm`'s row blocking
(one kernel call per tile, ≤64 rows per dispatch at width 768) — while every
operator this mode decomposes to (`elementwise_add`, `layer_norm`,
`elementwise_mul`, `gelu`) streams all 4096 rows inside a single launch, so
each normalization point is 3 entries here against 64 there.

08d anticipates exactly this outcome and prescribes reporting the number
rather than inflating the decomposition (row-blocking operators that stream,
or splitting GEMMs). The consequence is that the driver's
`runlist_entries > coarse` ordinal clause — the one E4 owns — FAILS on this
hardware: `runlist_entries` does not order the taxonomy's fine-vs-coarse axis
at `baseline_768`, because it measures dispatch-cap artifacts, not
granularity. Per 08 §Gate that is a finding about the measurement model, to
be resolved before Phase F consumes these numbers (a field that does order
the axis here: distinct operator kernels per layer — 10 vs 5 — or entries
net of row-blocking).

The number stands on more than the first decomposition tried. Every
restructuring that would raise it was checked, and each is excluded by a
measured constraint rather than by preference:

- **Row-banding the streaming operators to `coarse`'s 64-row granularity.**
  A dispatch argument is a whole BO — `run.set_arg` takes a buffer, never a
  buffer and an offset (`builders/block.py` §WHY THE LAYER IS FOUR DISPATCH
  SEQUENCES) — so bands must be cut and re-concatenated on the HOST at every
  GEMM boundary, and each cut is a new submission. Banding `add`/`ln`/`mul`/
  `gelu` uniformly makes 7 submissions and ~450 entries; 7 submissions
  breaks the E5 ordering clause that `offload`'s 6 exceed every other
  mode's. Variants that stay under 6 exist (band the norm chains, stream
  `gelu`: 5 submissions, ~390 entries) but their per-operator granularity is
  chosen FROM the inequalities, which is the "mode tuned until an inequality
  holds" the Phase E gate text forbids — and every banded variant trades the
  device-resident chaining this mode exists to measure for restage traffic,
  raising its sync and byte counts above `coarse`'s. Fine-grained by entry
  count, more host-mediated than the coarse mode by every other field, is
  not a point on the taxonomy's axis.
- **Banding the GEMMs.** A runtime-tiled GEMM ELF corrupts from its second
  execution in one `hw_context` (measured, first footgun below), so 64 bands
  across the six GEMM positions would need ~384 distinct ELFs against the
  32-context ceiling.
- **Dispatching iron's attention interior on device** (`k_transpose`,
  scale, softmax). The two attention GEMMs resolve in no registry and the
  sweep cannot stage them (previous section), so their neighbours' operands
  stay host-side either way, and each device entry between them ships the
  ~400 MB bf16 score tensor across the host boundary in both directions.
  The reachable count tops out near iron's 16 — still nowhere near 131, at
  a multiple of the transfer cost.

So no faithful decomposition of this layer on this hardware exceeds
`coarse`'s count: 128 of those 131 entries come from `addnorm`'s
three-input-stream L1 cap (one kernel call per tile), a constraint none of
the two-stream operators this mode decomposes to shares. The 08d premise
that "several" of the decomposed operators would row-block the same way is
what the measurement refutes.

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
  compiles of one module per clean cache is the cost of thirteen entries
  over two submissions; do not "deduplicate" them back.
- **The `add`/`ln`/`mul` ELFs execute twice inside runlist 2, and that is
  measured clean.** No runtime loop tiling, same class as the block's 64-fold
  re-executed `addnorm` ELF — and in the corrupted bring-up run their second
  executions still produced stage errors at the expected levels, so the
  GEMM-class failure does not extend to them.
- **The gamma multiply's second operand is a materialized broadcast.**
  `elementwise_mul` takes two full `[4096, 768]` tensors (its docstring
  records why there is no broadcast form), so each LayerNorm weight is tiled
  to 6 MB on the host (`broadcast_row_weight`) and declared static +
  content-keyed — uploaded once, like iron's materialized causal mask. Under
  fault injection the content key changes and the perturbed weight is
  re-uploaded; nothing special-cases the injected path.
- **`hw_context` demand is 10 concurrent, measured against a ceiling of 32.**
  By the time runlist 2 executes, the three projection contexts from runlist 1
  are still loaded and seven more join them (o_proj, up, down, add, ln, mul,
  gelu): ten DISTINCT designs resident simultaneously, demonstrated by the
  mode running rather than by a synthetic probe. The 32 ceiling was probed
  with 4 cycled ELFs (08 §Risks), so this is also the first data point with
  ten distinct designs — comfortably inside, and the margin no longer rests
  on the cycled-ELF assumption alone. iron's 29-context appetite does not
  arise here because attention is host and the streaming operators share
  ELFs across their two entries.
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
