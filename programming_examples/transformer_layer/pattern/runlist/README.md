# `runlist` — the fine-grained point of the execution-strategy taxonomy

The D2 `encoder_bert` layer decomposed into single-operator kernels, aggregated
into runlists: iron's `pattern/runlist` re-expressed over `KernelCache` and the
Phase B dispatch model. `prepare_runlist` is the mode's entry in the `SPECS`
catalogue; `make check-runlist` / `check-runlist-fault` run it.

## The decomposition, and where each piece comes from

Thirteen entries over two runlists, each entry one operator kernel:

| # | entry | kernel | ELF |
|---|---|---|---|
| 1 | q_proj | registry GEMM 4096x768x768 | `rl_gemm_4096x768x768` |
| 2 | k_proj | same module, same ELF | " |
| 3 | v_proj | same module, same ELF | " |
| — | attention | **host torch** (`pattern/blocked_attention.py`, shared with `offload`) | — |
| 4 | output_proj | registry GEMM 4096x768x768 | `rl_gemm_4096x768x768` (fresh `hw_context`) |
| 5 | residual add | `builders/elementwise_add.py` | `rl_add_4096x768` |
| 6 | LayerNorm | `builders/layer_norm.py` (unweighted) | `rl_ln_4096x768` |
| 7 | gamma multiply | `builders/elementwise_mul.py` | `rl_mul_4096x768` |
| 8 | up_proj | registry GEMM 4096x768x3072 | `rl_gemm_4096x768x3072` |
| 9 | GeLU | `builders/gelu.py` | `rl_gelu_4096x3072` |
| 10 | down_proj | registry GEMM 4096x3072x768 | `rl_gemm_4096x3072x768` |
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
host submissions 2, runlist entries 13 — see the lit recipe for the full
measured vector (air/herd/sync/bytes), which is hardware truth, not a target.
```

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

## Footguns (each cost time; read before editing)

- **The proj ELF re-executes, and the two failure regimes differ.** Across
  submissions in a reused `hw_context`, a runtime-tiled GEMM ELF returns
  wrong numbers from its second execution onward (`offload` measured it), so
  `output_proj` gets a fresh context — `_evict_context` between the runlists.
  Inside ONE runlist (q/k/v: three executions, one context, by construction)
  the same ELF re-executes CLEANLY — measured here, by the q/k/v stage
  comparisons at `atol 5e-3`, which the corruption mode (3.6e-1 mean_rel_L1)
  cannot pass. Do not fold the two cases into one rule; they are different
  mechanisms.
- **The `add`/`ln`/`mul` ELFs execute twice inside runlist 2.** No runtime
  loop tiling, same class as the block's 64-fold re-executed `addnorm` ELF;
  clean, verified by the `output` stage.
- **The gamma multiply's second operand is a materialized broadcast.**
  `elementwise_mul` takes two full `[4096, 768]` tensors (its docstring
  records why there is no broadcast form), so each LayerNorm weight is tiled
  to 6 MB on the host (`broadcast_row_weight`) and declared static +
  content-keyed — uploaded once, like iron's materialized causal mask. Under
  fault injection the content key changes and the perturbed weight is
  re-uploaded; nothing special-cases the injected path.
- **`hw_context` demand is 7 concurrent, measured against a ceiling of 32.**
  Runlist 2 holds contexts for all seven distinct ELFs at once (proj, up,
  down, add, ln, mul, gelu). The 32 ceiling was probed with 4 cycled ELFs
  (08 §Risks); this mode's own run demonstrates 7 DISTINCT designs resident
  simultaneously, comfortably inside it. iron's 29-context appetite does not
  arise here because attention is host and repeated operators share ELFs.
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
