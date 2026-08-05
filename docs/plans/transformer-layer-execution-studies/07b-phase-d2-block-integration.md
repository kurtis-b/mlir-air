# 07b — Phase D2: The block integration gate

One complete `encoder_bert` transformer layer, assembled from the Phase C builders, dispatched
through the Phase B runtime seam, matching a torch golden model element-wise on real hardware.

[07a](07a-phase-d1-operators-at-baseline-768.md) has already validated every operator this uses at
`baseline_768`. If something here fails, the per-boundary comparison below says which stage
diverged — and because each stage passed on its own in D1, a stage-level failure points at the
integration: launch argument maps, layout transitions between operators, external-kernel linking
across a multi-operator sequence, or BO reuse under the Phase B allocator.

## The configuration, and why none of it is a choice

```
family = baseline_768    hidden = 768    ffn = 3072    num_heads = 12    head_dim = 64
seq_len = 4096           variant = encoder_bert        non-causal
```

`baseline_768` because it is the only family whose GEMMs resolve. `seq_len = 4096` because it is
the only point where `build_ffn_module` builds at all — the `tile_n` collision in
[07a](07a-phase-d1-operators-at-baseline-768.md#the-family-and-why-the-sequence-length-is-forced),
which you are not fixing here. Non-causal because `encoder_bert` is.

## The golden model: port the structure, not the numerics

`/home/cj/iron/iron/applications/transformer_layer/pattern/reference.py` (172 lines, pure torch) is
the correctness anchor for this phase *and* all of Phase E. An earlier version of the Phase D
document said to port it verbatim. **Do not.**

It defaults to `dtype: str = "bf16"` and builds every tensor at that dtype — inputs, all four
attention weight matrices, both LayerNorm weights, both FFN weights. That is the defect corrected in
[06 §The numerics standard](06-phase-c-operators.md#the-numerics-standard--do-not-port-irons), and
it is worse here than it was per-operator: this reference chains eight GEMMs plus two LayerNorms
plus a softmax, so a bf16 oracle accumulates error in the same direction as the device and the
comparison flatters itself the whole way down.

Port the structure; compute in FP32 from bf16-rounded inputs, as every Phase C reference does.

Three details of the original that must survive the re-expression:

- **The RNG draw order is load-bearing.** `torch.manual_seed(seed)` then, in order: `input`,
  `q_weight`, `k_weight`, `v_weight`, `attn_output_weight`, `ln1_weight`, `ffn_up_weight`,
  `ffn_down_weight`, `ln2_weight`. Reordering the draws changes every tensor. Note `ln*_weight` is
  `torch.rand` (uniform), not `randn`, and `val_range = 0.05` scales the `randn` draws.
- **The biases are `torch.zeros`.** So the device operator having no bias term is not a gap — and
  because a zeros tensor consumes no RNG, the bias draws do not appear in the order above. Keep
  them out of it.
- **`include_output=False` is an escape hatch**, not dead code: it skips materializing the whole
  attention pipeline so the 16384 rung stays tractable for Phase F's runner. Keep it.

The structure itself, for `encoder_bert` (post-norm; the `decoder_gpt2` variant is pre-norm and
causal, and should be ported alongside since Phase E needs it):

```
q, k, v      = x @ Wq, x @ Wk, x @ Wv
attn         = softmax(q kᵀ / sqrt(head_dim)) v          # per head, no mask
attn_out     = concat_heads(attn) @ Wo
hidden       = LayerNorm(attn_out + x,      ln1_weight)   # add, THEN normalize
ffn_out      = gelu(hidden @ W_up) @ W_down
output       = LayerNorm(ffn_out + hidden,  ln2_weight)
```

iron's GeLU here is `torch.nn.functional.gelu`, i.e. **erf**, while the device kernel is the tanh
approximation. [06 §The numerics standard](06-phase-c-operators.md#the-numerics-standard--do-not-port-irons)
records this trap; `gelu_tanh_reference` in `builders/gelu.py` is what the FFN leg is actually
approximating.

Put the reference at `programming_examples/transformer_layer/pattern/reference.py` — Phase E builds
its four strategy directories beside it and imports this one shared copy.

## Assembling the layer

Five operator launches, in order:

| # | operator | in → out | dispatches |
|---|---|---|---|
| 1 | `qkv_proj` | `x` → `q, k, v` | 1 |
| 2 | `mha_out_proj` | `q, k, v, Wo` → `attn_out` (fused attention + output projection, non-causal) | 1 |
| 3 | `addnorm` **pre-add** | `attn_out, x, ln1_weight` → `hidden` | **64** |
| 4 | `ffn` | `hidden, W_up, W_down` → `ffn_out` | 1 |
| 5 | `addnorm` **pre-add** | `ffn_out, hidden, ln2_weight` → `output` | **64** |

> **`[2026-08-05]` The dispatch column was added after the fact.** Written without it, this table
> reads as one launch per row, and the normalization points are not: `build_addnorm_module`
> requires `rows == herd_x * rows_per_call`, which at `cols = 768` caps a call well below the
> layer's 4096 rows, so each normalization point is 64 dispatches. The operators are right; the
> count was not. It matters for Phase E, because it means `coarse`'s dispatch vector is dominated
> by `addnorm` rather than by the GEMMs.

Note what is *not* in that list: `layer_norm` and `elementwise_add` are not on the `encoder_bert`
path — the residual add lives inside the pre-add `addnorm` — and `causal_mask` is decoder-only.
They are validated at 768 by D1 because Phase E's finer-grained modes will decompose down to them,
not because this block calls them.

Dispatch through `KernelCache` + runlist, the seam Phase B built.
`runlist_gate.py::leg_c_run_sequence` is the working template: build a `DispatchStep` per launch, a
`BufferSpec` per argument (weights `static=True` with a `content_key` so they land in the
content-keyed pool; the layer output `host_output=True`), then one
`cache.run_sequence(steps, buf_specs, kwargs, arrays)`.

Two mechanical traps that will cost you a day each if you meet them from a failing run:

- **A multi-`air.launch` design must set `output_format="elf"`.** The xclbin path fails with
  `air.insts.bin produced duplicate output path`. `_GEMM_RUNNER_KWARGS` in `opcheck_specs.py`
  already carries this, along with `runtime_loop_tiling_sizes=[2, 2]` for BD-ID recycling.
- **The backend kwargs a sequenced artifact needs are not optional.** `runlist_gate.py`'s
  `backend_kwargs_for` records that dropping them "compiles and loads fine and then produces
  results that are wrong on the first call and *different* on every call after it."

`run_sequence` returns results that are **zero-copy views into pool memory**. Copy them before a
second pass, or the per-boundary comparison below will read whatever the next launch wrote.

## The per-boundary comparison is a work item, not a nicety

Capture and compare the intermediate after every operator boundary — at minimum `q`, `k`, `v`,
`attn_out`, `hidden`, the FFN's up-projection, its GeLU output, `ffn_out`, and `output`. Record each
as a stage in the results artifact, as a `stages` list of objects carrying `name`, `n_elements` and
`n_mismatch`.

The driver requires at least eight stages, **distinct names**, `n_mismatch == 0` on each, and each
`n_elements` no smaller than one `4096 × 768` boundary tensor — every boundary in this layer is at
least that (the FFN interior is `4096 × 3072`, larger). Those three constraints exist because the
weaker form of this clause was satisfiable by repeating one trivial entry eight times, which
localizes nothing.

C4 is why this is mandatory. It found a GEMM configuration that returned **zeros for two of the
nine sub-tiles of each cast worker** while still resolving from the registry and still producing a
plausibly-shaped output — 30% of the output wrong, and nothing above it noticed. A registry lookup
proves a row exists; it never proves it computes. A single end-to-end comparison over a layer that
ends in a LayerNorm can absorb a surprising amount of upstream damage before it trips.

The driver's objective check requires at least eight stages, each with `n_elements > 0` and
`n_mismatch == 0`, so this cannot be quietly dropped.

## The check goes through `opcheck.py`

**Reuse it. Do not write a second checker.** It already owns the contract the driver depends on:
FP32 reference, registry `rtol`/`atol`, zero mismatches, a machine-readable verdict per
`(operator, shape)`, and `--fault-inject input` which must fail. Add a `block` entry to
`opcheck_specs.py` — a `_prepare_block(shape, seed=...)` and a `SPECS` row — exactly as every
operator before it did.

> **`[2026-08-05]` This document said `opcheck.py` itself should not need to change. That was
> wrong, and the work falsified it.** The claim assumed the block is one `air.ir.Module` that
> `XRTRunner.run_test` can run. It cannot be: `build_addnorm_module` caps rows per call, so each
> normalization point is many dispatches and the layer is several ELFs rather than one module.
> `opcheck.py` gained an additive `dispatch` seam for operators of that shape, plus `stage_stats`
> so the per-boundary evidence is produced by the same `_RecordingRunner._check_outputs` that
> decides the verdict — such an operator's verdict is the conjunction of its end-to-end and
> per-boundary comparisons.

Two footguns in it worth knowing before you start:

- `opcheck.py --list` imports `XRTRunner` at module scope, so it needs the AIR environment sourced
  even though it touches no hardware.
- Injected runs are written to `results/fault/`, deliberately in a subdirectory, so a fault-injected
  result can never be mistaken for or overwrite a clean one.

Choose the injection target with the same care `mha_out_proj` did: perturb an input with no
averaging operator in front of it. A perturbation to `x` passes through a softmax, which damps it;
`mha_out_proj` therefore injects into `w_o`. A block-level injection that the layer's two
LayerNorms wash out would make the negative control pass under injection, and the driver fails the
sub-phase for exactly that.

**The tolerance ceiling is hard.** `rtol` is pinned to `1.6e-2` and `atol` must be at most `1e-1`;
the driver re-derives both. Eight chained GEMMs, two LayerNorms and a softmax against an FP32
oracle is the most demanding thing this contract has been asked to cover, and the final LayerNorm
is the only reason to expect it to fit at all — it renormalizes the output to roughly unit scale.
If the honest `atol_required` does not fit under `1e-1`, report it in `work_not_completed` with the
measured value and the per-stage errors showing where it accumulated. That is a real result. A
widened tolerance is not, and the driver will reject it anyway.

## Gate

```
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  ninja -C build-xrt check-programming-examples-transformer-layer
```

with a new `run_npu2_block_peano.lit` in it. Enrolment is path-based (`--filter
"transformer_layer/"` in `programming_examples/CMakeLists.txt`), so a new `.lit` anywhere under the
example directory joins the suite with no CMake change.

Model it on the existing `run_npu2_<op>_peano.lit` files, and copy their two boilerplate warnings
verbatim: do **not** wrap the recipe in `/tmp/mlir-air-npu.lock` (the caller holds it and BSD
`flock(2)` self-deadlocks; device locking happens one layer down on `/tmp/npu.lock` inside
`XRTRunner`), and never spell a lit directive name in prose, because lit scans the whole file and
a second directive leaves the test UNRESOLVED.

The driver then checks, independently of anything you write: the `block` results are fresh, the
verdict re-derives, the fault-injected run fails, D1's per-operator `baseline_768` coverage is
still intact, and **at least one** block result has the recorded `shape` at the top of this
document, an `n_elements` equal to the full `4096 × 768` layer output — so a comparison over a
slice cannot pass — and a clean stage list meeting the constraints above. A smaller bring-up block
shape beside the gate point is fine; it is a declared shape, so it is held to the same numerics
contract as everything else.

The block's `shape` dict must use these key names, matching the conventions the other operators
already use: `seq_len`, `emb_dim`, `ffn_dim`, `num_heads`, `head_dim`.

## Risks

- This phase has no new device code, so a failure here means something in Phase B or C is wrong in
  a way its own gate did not catch. Budget time for iterating back into those phases rather than
  treating this as a formality.
- If the element-wise comparison fails, the per-boundary intermediates identify which stage
  diverged. Do not proceed to Phase E on a layer that only approximately matches.

## Known failure modes to check for

Drawn from this repository's own debugging skills; these are the specific things the gate exists to
catch.

| Symptom | Typical cause |
|---|---|
| All-zero output from a herd | Bare herd not wrapped in a launch/segment |
| Silent corruption in GEMM output | `N % (tile_n × herd_n) != 0` |
| Correct standalone, wrong when chained | BO reuse without correct synchronization |
| `ERT_CMD_STATE_TIMEOUT` | `instance_name` not matching the emitted `func.func @name` |
| Correct first call, wrong on subsequent calls | Stale buffer contents under pooling |
| NaN in attention output | L1 overflow at large head dimension |

The last three are direct interactions with Phase B's allocator, which is why this gate follows it
rather than preceding it.
