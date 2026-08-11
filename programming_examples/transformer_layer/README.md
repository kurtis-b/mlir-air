# Transformer Layer — AIE2P Device Kernels

Phase A of the [transformer-layer execution
studies](../../docs/plans/transformer-layer-execution-studies/04-phase-a-kernels.md):
the C++ device kernels a full encoder/decoder block needs, compiled with Peano
for AIE2P.

It also holds Phase B's runtime-seam gate, the Phase C/D1 operator builders and
their numerical checks, the Phase C4 registry sweep, Phase D2's block
integration gate — one whole `encoder_bert` layer assembled from those operators
and compared against a shared golden model at every boundary it passes — and,
under `pattern/`, Phase E's execution strategies, of which `coarse` is the
first. The kernel half needs no
NPU, which is what keeps it safe as a PR gate; the seam half is split the same
way — host-only unit tests, plus one hardware gate. The operator checks all need
one.

```bash
make compile                 # build every object and check its symbols (no NPU)
make seam-tests              # BO pooling + runlist aggregation rules (no NPU)

flock -x -w 1800 /tmp/mlir-air-npu.lock make runlist-gate   # NEEDS AN NPU
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  ninja -C build-xrt check-programming-examples-transformer-layer   # all three, NEEDS AN NPU
```

The lit suite needs an NPU because `run_npu2_runlist_gate.lit` dispatches. It did not always: the
suite held only the compile-only and host-only tests, so Phase B's hardware claim was gated by
nothing the driver ran. `make compile` and `make seam-tests` remain individually NPU-free, which
is what keeps them usable as a PR check.

**The two NPU locks are different inodes, and they must not nest.**
`/tmp/mlir-air-npu.lock` above is the *invocation* lock: a human, or the port-loop driver, takes
it once around the whole suite. Device access itself is locked one layer down on `/tmp/npu.lock`,
which `XRTRunner.run_test` and `KernelCache` hold across xclbin load and dispatch, and which
`programming_examples/lit.cfg.py`'s NPU substitution uses for every other hardware test in the
tree. That is why no `.lit` recipe here wraps its own commands in the outer lock: the caller
already holds that inode, and BSD `flock(2)` treats a second `open()` of the same file as a
foreign lock, so a nested acquire would block against its own parent until the timeout expired.
Keeping the layers separate is also what lets the Peano compiles overlap while the dispatches
serialize against every other NPU job in the repository.

## The Phase B runtime seam

`runlist_gate.py` is
[Phase B](../../docs/plans/transformer-layer-execution-studies/05-phase-b-runtime-seam.md)'s
gate: three separately-compiled ELFs in one runlist, bit-identical to sequential dispatch and
measurably faster, plus the whole layer through the seam in one submission.

The plan's proposed *mechanism* — binding several ELFs into one XRT `hw_context` — does not work
on XRT 2.21.0 / NPU2, and does not need to. An AIR ELF is a *full* ELF carrying its own array
configuration and a `hw_context` accepts exactly one of those, but a runlist is constructed
*against* a context rather than restricted *to* it: each entry is dispatched on the context its
kernel came from, so N ELFs means N contexts and still one runlist.
[05a](../../docs/plans/transformer-layer-execution-studies/05a-phase-b-runlist-spike-result.md)
records both halves and every measurement.

The one aggregation that is silently wrong is the *xclbin* cross-artifact runlist — there the
configuration is in the xclbin, not the run, so it executes and returns wrong numbers with no
error. The seam refuses to build it and leg A2 of the gate keeps that refusal honest.

What the seam delivers, and what the gate measures:

| | |
|---|---|
| `llms/shared/infra/dispatch.py` | Groups a dispatch sequence into the submissions the hardware allows — one, spanning artifacts, under the ELF ABI — and owns the six-field dispatch vector. Under the xclbin ABI it splits at every artifact change and refuses `require_single`, because that runlist *executes* and returns wrong numbers with no error. |
| `llms/shared/infra/bo_pool.py` | Live ranges over the sequence, 4 KiB-binned slot sharing, a content-keyed static-weight pool, and a dirty bit per BO so only written buffers sync to device and only declared outputs come back. |

The rules both modules implement are written down in
[05b](../../docs/plans/transformer-layer-execution-studies/05b-phase-b-buffer-rules.md) *before*
the code, and the module docstrings name its sections. Read it before changing either module —
the failure modes here (a pooled BO is larger than its buffer, returned arrays are zero-copy
views into pool memory, an xclbin-ABI slot is keyed by argument index because that picks the
memory bank) all produce plausible wrong numbers rather than errors.

## What lives here

| File | Contents |
|---|---|
| `kernels/encoder.cc` | Encoder-block kernels: staged FFN (`-DBUILD_FFN`) and weighted add-norm (`-DBUILD_ADDNORM`). Holds the contract docs and the `extern "C"` entry points |
| | Built to **two** objects: `encoder.o` (addnorm half, for `addnorm`) and `encoder_ffn.o` (FFN half, for `ffn`) |
| `kernels/encoder_matmul.cc` | The encoder's 2x2-expanded `aie::mmul` microkernels, included by `encoder.cc` |
| `kernels/encoder_layer_norm.cc` | The encoder's LayerNorm reductions, fused and staged, included by `encoder.cc` |
| `kernels/addnorm_ffn.cc` | Fused add-norm + FFN staging, both residual orderings behind `-DADDNORM_PRE_ADD`. Holds the contract docs and the `extern "C"` entry points |
| `kernels/addnorm_ffn_matmul.cc` | The FFN's 1x4-expanded `aie::mmul` microkernels, included by `addnorm_ffn.cc` |
| `kernels/addnorm_ffn_norm.cc` | The fused add-norm templates and tile passthroughs, included by `addnorm_ffn.cc`. The only file `-DADDNORM_PRE_ADD` reaches |
| `kernels/elementwise.cc` | `eltwise_vadd` and `gelu_tanh_approx_bf16`, textually included by both kernels |
| `compile_kernels.py` | The compile-and-check driver the lit test runs |
| `run_npu2_compile_peano.lit` | Phase A's compile-only gate. Peano, no NPU |
| `run_seam_tests.lit` | Phase B's host-only rules tests. No NPU, no XRT |
| `run_npu2_runlist_gate.lit` | Phase B's hardware gate: the four runlist legs on a real NPU |
| `opcheck.py` | Phase C's numerical check: the CLI, the results artifact, and the fault-injection negative control. What counts as EVIDENCE |
| `opcheck_prepare.py` | HOW each operator is built and fed: one `prepare_<operator>` each, plus every per-operator footgun |
| `opcheck_specs.py` | WHICH `(operator, shape)` the port claims and at what `atol`, with the measurement behind every tolerance |
| `builders/elementwise_add.py` | 2-D element-wise add and the `causal_mask=` keyword over it, plus their FP32 reference |
| `builders/layer_norm.py` | Multi-row LayerNorm over `layer_norm_rows`, plus its two-pass FP32 reference |
| `builders/softmax.py` | Row-wise softmax over the STREAMING family in `programming_examples/softmax/softmax.cc` (init / partial / normalize), plus its FP32 reference. Not the single-shot `softmax_bf16` in the same file, which subtracts no row max and fixes the row width |
| `builders/addnorm.py` | Weighted LayerNorm and a residual in **either order** (`pre_add=`), weight as a runtime argument, plus one FP32 reference per ordering |
| `builders/qkv_proj.py` | One GEMM over the fused `[K, 3K]` weight with C split three ways on the device, plus its FP32 reference |
| `builders/gelu.py` | The FFN activation stage over `ffn_gelu_bf16`, plus its FP32 tanh-approximation reference |
| `builders/ffn.py` | Staged up-projection / GeLU / down-projection composition, plus its FP32 reference |
| `builders/ffn_resident.py` | Doc 31 R1: the same FFN as ONE segment — up ring, GeLU herd and the J7b down ring on channels, the `[seq, ffn]` interior never crossing DRAM. Reference imported from `ffn.py` |
| `builders/test_ffn_resident.py` | Host-only: the interior's addressing arithmetic emulated to EXACTNESS in f64 — packing, shim retile, chunk order. The arm that stays live while the device gate is parked. No NPU, no aircc |
| `ffn_resident_structure.py` | Host-only: the interior IS resident — one `aie.device`, core→core stage edges, the column budget, the down ring's hoist, the dispatched kernels. No NPU. PARKED with the device gate on the `air-fuse-channels` crash (doc 31) |
| `builders/mha_attention.py` | Attention staging: the seq-first FlashAttention design point, its kernel `-D` flags, and the chunked FP32 oracle |
| `builders/o_proj.py` | O-projection staging: the registry lookup, the GEMM sub-kernel, and its FP32 oracle |
| `builders/mha_out_proj.py` | The entry layer that composes the two into one ELF, plus the composed FP32 reference |
| `builders/block.py` | Phase D2: the five operator launches assembled into four `KernelCache.run_sequence` calls, and every boundary read back |
| `builders/block_cache.py` | When a cached block ELF is the ELF the current source would build: the artifact fingerprint and its persistence |
| `builders/test_block_cache.py` | Host-only: which cached block ELFs `check-block` reuses, and which it rebuilds |
| `pattern/reference.py` | The shared golden model — iron's draw order and structure, this repository's FP32-from-bf16 numerics, both workload variants, every boundary. Phase E's strategy directories import this one copy |
| `pattern/test_reference.py` | Host-only: that the golden model is the layer it claims to be |
| `pattern/coarse/cells.py` | `coarse`'s blend space: the six cells, and the composition that builds the two interior ones out of the block and runlist halves rather than reimplementing either |
| `pattern/coarse_c2/`, `pattern/coarse_c3/` | The two interior cells as catalogue operators — a preparer and an ELF cache each, and nothing else |
| `coarse_cells_structure.py` | Host-only: what each cell WILL dispatch, derived from the configs, with the model checked against `coarse`'s and `runlist`'s shipped gate literals |
| `run_npu2_<op>_peano.lit` | One per operator: that operator's numerical gate on a real NPU |
| `run_npu2_fault_control_peano.lit` | Phase C's negative control: the injected run must FAIL |
| `run_npu2_block_peano.lit` | Phase D2's gate: the whole layer on a real NPU, every boundary checked, then the same layer under injection which must FAIL |
| `run_block_cache_tests.lit` | Phase D2's host-only cache-reuse tests. No NPU, no XRT |
| `run_reference_tests.lit` | Phase D2's host-only golden-model checks: the composition, and the three substitutions no tolerance would catch. No NPU, no XRT |
| `sweep/sweep_families.py` | Which shapes the case matrix needs, and which tilings are worth trying for each |
| `sweep/sweep_measure.py` | One candidate end to end — build, numerical check, timing. The process the sweep forks |
| `sweep/registry_writer.py` | The append-only write into the registry JSON and both markdown pages |
| `sweep/registry_sweep.py` | Orchestration: turbo gate, resume, checkpointing, winner selection, the CLI |
| `sweep/sweep_report.py` | Downstream of a finished sweep: the resolution assertion and the family markdown |
| `sweep/test_registry_writer.py` | What the writer will and will not do to an existing shape, against a temporary registry |
| `sweep/test_sweep_families.py` | That the sweep's duplicated method table still agrees with `gemm_builder`'s, and that no two `tile_n` it may plan share an object |
| `sweep/run_npu2_registry_resolution.lit` | Every `baseline_768` shape resolves through the builder that owns it. No NPU |
| `sweep/run_sweep_writer_tests.lit` | The writer's append-only guards. No NPU |
| `sweep/run_sweep_families_tests.lit` | The duplicated method table and the per-`tile_n` object naming. No NPU |

There are two compiled objects, not six: `encoder.cc` and `addnorm_ffn.cc` are
the only translation units, and each `#include`s its two siblings the way
`matrix_multiplication/bf16_in_fp32_out/mm_aie2p.cc` includes `zero.cc`.

Kernels that already had an MLIR-AIR home were extended in place rather than
copied here:

| Kernel | Where | Opt-in flag |
|---|---|---|
| `matmul_init_*`, `matmul_with_acc_*` | `matrix_multiplication/bf16_in_fp32_out/mm_aie2p.cc` | `-DGENERATE_MATMUL_INIT_KERNELS`, `-DGENERATE_MATMUL_WITH_ACC_KERNELS` |
| Two-pass streaming softmax | `softmax/softmax.cc` | `-DSOFTMAX_STREAMING` |
| Multi-row LayerNorm | `layer_norm/layer_norm.cc` | — (new file) |
| Causal-mask row helpers | `flash_attention/kernel_fusion_based/attn_npu2.cc` | `-DCAUSAL_ROW_HELPERS` |

Every flag defaults off, and the default-build objects for the three
pre-existing sources were verified byte-identical to their pre-port versions.
The ten shipped LLM deployments that link them are unaffected.

## Why the two kernels are each three files

As first landed, `encoder.cc` (973 lines) and `addnorm_ffn.cc` (1116) both ran
past the ~800-line module guideline in
[02-porting-conventions.md](../../docs/plans/transformer-layer-execution-studies/02-porting-conventions.md).
Each is now split along the seam it already had internally — **matmul
microkernels · normalization templates · `extern "C"` entry points** — leaving
every source between 245 and 493 lines.

The seam is by *role*, not by the `-DBUILD_FFN` / `-DBUILD_ADDNORM` build flags.
Those flags gate only the entry-point layer, so cutting there would have left the
matmul and LayerNorm template bodies in one oversized file regardless. The role
seam also puts each footgun next to the code it applies to: `-DADDNORM_PRE_ADD`
is read entirely inside `addnorm_ffn_norm.cc`, and the variance clamp that keeps
`aie::invsqrt` off a negative operand is documented in the two normalization
files rather than in a header 700 lines away.

Splitting changed no code and no object: the sources are included textually, so
`encoder.o` and `addnorm_ffn.o` are still built from one translation unit each,
with the same flags and the same symbols the compile gate checks.

## The operator checks

`opcheck.py` is the single numerical entry point for every operator this port
lands, and `builders/` holds the operators themselves — one
`build_<name>_module()` function per operator with its FP32 reference beside it,
no operator class and no `op.py`/`design.py` pair.

The check is three modules, each knowing one thing, because together they passed
porting convention 5's ~800-line cap twice:

| module | knows |
|---|---|
| `opcheck.py` | what counts as EVIDENCE — the recording runner, the injection, the results artifact, the negative-control verdict, the CLI |
| `opcheck_prepare.py` | HOW each operator is built and fed — one `prepare_<operator>` each |
| `opcheck_specs.py` | WHICH `(operator, shape)` is claimed, and at what `atol` |

Adding a shape touches only the catalogue. Adding an operator touches the
catalogue and the preparers. Changing what counts as evidence touches only
`opcheck.py`.

```bash
make opcheck-list                 # every (operator, shape) claimed, as JSON. No NPU.
flock -x -w 1800 /tmp/mlir-air-npu.lock make check-layer-norm PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR
flock -x -w 1800 /tmp/mlir-air-npu.lock make check-fault-control PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR
```

The verdict is `XRTRunner.run_test`'s, not a comparison written here:
`np.isclose` over the full output at `rtol = 1.6e-2` with zero permitted
mismatches, against a reference computed in float32 from bf16-rounded inputs.
`opcheck.py` subclasses the runner only to copy out the error statistics on the
way past. Each run writes `results/<operator>__<shape>.json`; the registry rows
in `programming_examples/kernel_registry/` are the durable record.

`--fault-inject input` is the negative control. It perturbs one element of the
array handed to the **device**, after the reference has been computed from the
clean one, and the run must then FAIL. This is the layer a laxer test cannot
satisfy: a reference compared against itself, a tolerance wide enough to swallow
anything, and an ignored flag all still report PASS under injection.
`check-fault-control` runs it with `--expect-failure` so the suite gates on it.
That flag reports the control's own verdict — exit 0 only if the comparison ran
and rejected the perturbed run — rather than inverting the exit status, which
would read a missing `PEANO_INSTALL_DIR`, a kernel link error or an absent NPU
as a caught fault. Injected runs write into `results/fault/` so they can never
overwrite a clean verdict.

**A shape that is not in `opcheck.py --list` is validated by nothing.**
`results/` is gitignored, so a results file for an undeclared shape is invisible
to the fingerprint, to the tamper check and to every review diff. Adding a shape
means adding it to `SPECS` in `opcheck_specs.py` *and* adding its `CHECK` line to
that operator's `.lit`; a results file on its own is not evidence.

**And a newly added shape usually needs an injection of its own.** The shared
`check-fault-control` injects one operator, because the machinery is shared; the
driver injects each operator's *first* declared shape. Neither reaches a row
appended later, so the newest row in the file — the one whose PASS is least
earned, since nothing yet shows the comparison can reject it — is the one nobody
perturbs. `check-ffn-ladder-fault` is the seq-64 point's, and the pattern to copy.

### The `baseline_768` set (Phase D1)

Phase C brought each operator up at whatever width was cheapest. The block runs
at `baseline_768` — hidden 768, ffn 3072, 12 heads × head_dim 64,
`encoder_bert` — and almost none of those points were there. Each operator now
carries one, so that a block failure localizes to the integration rather than to
an operator nobody had run at that width:

| operator | `shape_key` | measured `mean_rel_L1` | `atol_required` | `atol` | margin |
|---|---|---|---|---|---|
| `elementwise_add` | `4096x768` | 1.879e-3 | 0.0 | 5e-2 | `rtol` alone covers it |
| `layer_norm` | `4096x768` | 1.969e-3 | 1.419e-3 | 5e-3 | 3.5× |
| `addnorm` | `64x768_pre_add` | 2.687e-3 | 6.646e-4 | 2e-3 | 3.0× |
| `qkv_proj` | `4096x768` | 9.863e-3 | 1.773e-3 | 5e-3 | 2.8× |
| `ffn` | `4096x768x3072` | 1.569e-2 | 1.472e-3 | 5e-3 | 3.4× |
| `mha_out_proj` | `4096x768x12h` | 5.335e-2 | 8.706e-3 | 2.5e-2 | 2.9× |

`rtol` is `1.6e-2` for all of them, as everywhere in the registry, and every
`atol` is inside the hard `1e-1` ceiling. The three GEMM-backed rows are pinned
to `seq_len = 4096` because that is where the block runs. For `ffn` there was
also no choice until Phase E1: 4096 was
[the only point that built at all](#things-that-will-bite-you) at hidden 768.
`ffn` now carries a second ladder point at `seq = 64` — `mean_rel_L1` 1.561e-2,
`atol_required` 1.150e-3, the same 5e-3 `atol` at a 4.3× margin — where both
projections resolve to `drain` at `tile_n` 128 and 96, the pairing that used to
fail to parse. It runs both GEMMs at herd 2×4 rather than the file-level 8×4,
because at `M = 64` drain's forced `tile_m = 32` admits at most `herd_m = 2` and
`resolve_gemm_spec` reads the herd off the registry row.
The three row-parallel ones are not pinned, because their builders derive
the legal row count; they run at 4096 rows anyway, except `addnorm`, whose L1
budget caps it (see below).

`causal_mask` has no `baseline_768` row and that is not an oversight. Its shape
is `seq × seq`, so the family's *width* does not name a shape for it, and
`encoder_bert` never builds one — the golden reference uses an all-ones attention
mask for the encoder variant and a `tril` one only for `decoder_gpt2`.

## The block integration gate (Phase D2)

One whole `encoder_bert` layer, at the configuration
[07b](../../docs/plans/transformer-layer-execution-studies/07b-phase-d2-block-integration.md)
forces: `seq_len 4096`, hidden 768, ffn 3072, 12 heads × head_dim 64, non-causal.

```bash
make reference-tests                                        # the golden model, no NPU
make block-cache-tests                                      # what the ELF cache reuses, no NPU
flock -x -w 1800 /tmp/mlir-air-npu.lock make check-block       PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR
flock -x -w 1800 /tmp/mlir-air-npu.lock make check-block-fault PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR
```

Five operator launches over four separately compiled ELFs:

| # | operator | in → out |
|---|---|---|
| 1 | `qkv_proj` | `x` → `q, k, v` |
| 2 | `mha_out_proj` | `q, k, v, w_o` → `attn_context, attn_out` |
| 3 | `addnorm` **pre-add** ×64 | `attn_out, x, ln1_weight` → `hidden` |
| 4 | `ffn` | `hidden, w_up, w_down` → `ffn_up, ffn_gelu, ffn_out` |
| 5 | `addnorm` **pre-add** ×64 | `ffn_out, hidden, ln2_weight` → `output` |

`layer_norm`, `elementwise_add` and `causal_mask` are not on this path — the
residual add lives inside the pre-add `addnorm` and `encoder_bert` is
bidirectional.

### The golden model

`pattern/reference.py` ports the **structure** of iron's
`generate_golden_reference` and not its numerics. The draw order is load-bearing
and is preserved exactly (`input`, then `q/k/v/attn_output` weights, then
`ln1_weight`, `ffn_up`, `ffn_down`, `ln2_weight`; `ln*` is `rand`, not `randn`;
the biases are `zeros` and consume no RNG, so they are not in the order). What
is dropped is `dtype="bf16"` throughout: this chain is eight GEMMs, two
LayerNorms and a softmax, and a bf16 oracle accumulates error in the same
direction as the device the whole way down. Draws are f32 rounded once to bf16 —
what the device is actually given — and the arithmetic is f32.

Each boundary is computed by calling the operator oracle `opcheck.py` already
validated that operator against in D1, so there is one implementation of each
piece of arithmetic rather than two that can drift.
`pattern/test_reference.py` pins that composition against a straight-line torch
transcription of iron's structure, because a composition can be well-typed and
still be the wrong layer.

### The ten boundaries, and why they are not a nicety

| boundary | elements | `mean_rel_L1` | `atol_required` | `atol` | margin |
|---|---|---|---|---|---|
| `q` / `k` / `v` | 3145728 each | 9.7e-3 | 3.1e-3 | 5e-3 | 1.6× |
| `attn_context` | 3145728 | 1.774e-1 | 2.288e-4 | 1e-3 | 4.4× |
| `attn_out` | 3145728 | 1.406e-1 | 7.371e-4 | 2.5e-3 | 3.4× |
| `hidden` | 3145728 | 5.181e-3 | 1.176e-2 | 3.5e-2 | 3.0× |
| `ffn_up` | 12582912 | 1.154e-2 | 4.977e-2 | 1.5e-1 | 3.0× |
| `ffn_gelu` | 12582912 | 1.660e-2 | 4.519e-2 | 1.5e-1 | 3.3× |
| `ffn_out` | 3145728 | 1.783e-2 | 1.144e-1 | 3.0e-1 | 2.6× |
| `output` | 3145728 | 1.688e-2 | 7.398e-2 | 1e-1 | 1.35× |

The layer's own relative error, 1.688e-2, is within 8% of the `ffn` operator's
1.569e-2 at the same shape: the FFN dominates and nothing downstream amplifies
it. What makes the *absolute* number large is scale. The golden model's
activations run around 1 where the registry's GEMM sweep puts a depth-3072
reduction at `1/sqrt(3072)`, roughly 60× smaller; the `ffn` row's 1.472e-3
scaled by that is 9e-2, which is what reaches the second LayerNorm and, divided
by its ~1.2 row standard deviation and multiplied by a gamma in [0, 1), is the
measured 7.4e-2. **`atol` is 1e-1, the hard ceiling, at a 1.35× margin** — the
thinnest in this example, stated rather than padded because there is nowhere to
pad to. The final LayerNorm renormalizing to roughly unit scale is the only
reason the layer fits under the ceiling at all.

`attn_context` and `attn_out` have a *relative* error above 14% and an
`atol_required` three orders of magnitude below everything else, which is one
fact and not two. iron's `val_range = 0.05` puts attention scores around 5e-3, so
the softmax is nearly uniform, the attention output is an average of 4096 V rows,
and `attn_out` lands around 1e-3 against a residual `x` around 5e-2. A near-uniform
average is a small difference of similar numbers, so its relative error is large
and its absolute error is tiny — and the attention half contributes a few percent
of what the first LayerNorm sees. That last part is why the per-boundary
comparison is a work item:

- Perturbing one element of `w_o`, or of the fused QKV weight, by the shared
  `FAULT_DELTA` of 2.0 puts **zero** elements of the layer output outside the
  tolerance band. A negative control injected there would pass under injection
  and prove nothing. The block's control goes into `ln1_weight`, which reaches
  8% of the output through two paths (the FFN's input *and* the second addnorm's
  residual); `opcheck_specs.py` records the measurement for all seven candidates.
- Swapping the tanh GeLU for the erf form moves `ffn_gelu` by at most 4.7e-4
  absolute and the layer output by 6.1e-4, so *no* tolerance in this example
  would see it — the stage list does not help here either. That oracle is pinned
  by identity in `pattern/test_reference.py` instead.
- The 25% of `ffn_up` that came back zero during bring-up (below) reached the
  layer output as "54% of elements wrong", which says only that something is
  broken. The stage list said `ffn_up`, and that everything before it was exact.

### Four dispatch sequences, not one

`addnorm` cannot be dispatched over 4096 rows. Its kernel drives three L3→L1
streams per tile against a column's two shim MM2S channels, so it takes exactly
one kernel call per tile and L1 caps it at `addnorm_max_rows(768, pre_add=True)`
= 104 rows. The layer's two normalization points are therefore **row-blocked
into 64 dispatches each of the 64×768 pre-add shape D1 measured** — the strongest
form the constraint allows, since the block then runs the operator that was
validated rather than a wider one that was not.

The consequence is a host restage between operators, because a dispatch argument
is a whole BO: `run.set_arg` takes a buffer, never a buffer and an offset. So the
layer is four `run_sequence` calls — `qkv_proj + mha_out_proj`, ln1, `ffn`, ln2 —
of which only the first is fully device-resident (`q`, `k` and `v` are produced
by one artifact and consumed by the next without touching the host). A single
sequence would be preferable and is not available; raising `rows` past the cap is
not the way to get one, because the builder raises precisely because the two-trip
form miscompiles rather than failing.

### The ELF cache is keyed by fingerprint, not by name

`check-block` and `check-block-fault` share four cached ELFs under
`block_cache/`, because compilation depends on the shape and not on the data and
rebuilding the layer to perturb one weight would double the gate's hardware time
for nothing. What makes that sharing safe is *what counts as the same artifact*.

An artifact name carries the shape and nothing else. The tiles and the method
come from `kernel_registry` and the IR comes from the builders in this
directory, so "four binaries whose names match the shape" is also satisfied by
four binaries built from a registry row that has since been re-swept, or from a
builder that has since been fixed. Running those while the results artifact
records the freshly resolved specs would be a **passing gate for an
implementation that never reached the device** — the one failure this whole
example exists to prevent.

So `compile_block_artifacts` builds all four MLIR modules on every call (about
0.1s against the minutes of compilation they gate) and reuses a cached ELF only
where a recorded fingerprint over its built MLIR, its resolved configuration,
every device kernel source and its backend kwargs still matches
(`builders/block_cache.py`). A miss recompiles that artifact and only that one.
The configuration is fingerprinted alongside the MLIR rather than being treated
as redundant with it because a registry row's `tile_k_l1` reaches the ELF
through `compile_gemm_mm`'s `-D` flags, which the IR does not carry.

What it cannot see is the toolchain — peano, mlir-aie, aircc and the AIE API
headers are version state shared with the rest of the repository. After a
toolchain bump the cache is stale and nothing here will say so; `make clean` is
what invalidates it. `make block-cache-tests` pins both directions of the rule.

It also cannot see a change in the ORDER things are compiled in, which is not a
gap in the digest so much as a consequence of covering the right things: E1 moved
the external-object compiles from interleaved-per-artifact to all-then-all, and
every input to the fingerprint was unchanged, so all four ELFs would have been
reused and the run would have proved nothing. Deleting `block_cache/` is what
makes an ordering change testable, and the evidence is that every stage figure
comes back byte-identical — `ffn_up` above all, since that is the boundary the
interleaving existed to protect.

## The `coarse` execution strategy (Phase E2)

`pattern/coarse/` is the first of Phase E's four execution-strategy modes: few
fused kernels over one runlist per sequence — renamed from iron's mode per
porting convention 7, with the old name surviving only as the CSV
`execution_mode` value. **The mode is not a second block.** `builders/block.py`
stays exactly where it is, with its lit, opcheck and coverage enrolments
untouched; `pattern/coarse/coarse.py` is the mode layer only — the shared
full-layer preparer (`opcheck_layer.prepare_layer_dispatch`) pointed at the
mode's own ELF cache, its own operator name in the `SPECS` catalogue, and its
`execution_mode` value from the one mapping in
`pattern/__init__.py::EXECUTION_MODE_CSV`. `pattern/coarse/README.md` has the
mode's measured dispatch vectors and what they calibrate.

Three parts of the E2 contract are easy to get wrong, and each is enforced
somewhere that will catch it:

**Each mode gets its own `KernelCache` directory, and it is not a style
choice.** The fingerprint that gates ELF reuse is sound, but the cache
*directory* is chosen by name (`KernelCache(cache_dir=...)`), so two modes
pointed at one directory can trade ELFs whose fingerprints happen to agree —
numerically valid output attributed to the wrong execution boundary, a failure
no equivalence check would surface. `coarse` uses `coarse_cache/`, gitignored
and in `make clean` in the same commit that created it, because the driver's
negative control runs `opcheck.py` from the source tree and the cache lands
there — the leak `block_cache/` already had once.

**The dispatch vectors are recorded, never counted by hand.** Every row in a
mode's `dispatch_vectors` is `DispatchVector.as_row()` straight out of
`KernelCache.run_sequence` (`llms/shared/infra/dispatch.py`), one per
sequence. Two things about the six keys bite if assumed otherwise:
`runlist_entries_per_submission` is a derived **mean**, so total entries are
`Σ round(mean × submissions)` — never a naive sum of the means, which only
agrees while every submission count is 1 — and the driver rejects any row
whose product is not a whole number of entries, because that is the shape a
fabricated number takes.

**The fault-injected run carries the vectors too.** The driver re-runs the
mode with `--fault-inject input`, requires that run to *fail*, and requires
the fault artifact's six summed totals to *equal* the clean run's — injection
perturbs one input element after the reference exists and never touches the
dispatch path. The instrumentation is therefore unconditional in the shared
preparer's dispatch closure, which also validates every recorded row against
the `as_row()` schema and prints the six driver-style summed totals
(`opcheck_layer.dispatch_vector_totals`); the lit recipes pin that totals
line to one set of literals in *both* halves, so a conditional shortcut, a
malformed row, or a fault run whose totals drift all fail in the suite before
the driver's independent totals comparison sees them.

The cost worth stating: the suite now runs **two** full-layer tests — `block`
and `coarse` — and each lit test starts with `make clean` in its own working
directory, so the second compiles its four ELFs rather than inheriting the
first's cache. Real minutes on every gate, and the price of keeping D2's gate
provable against its own artifact set while the modes stay unable to trade
ELFs.

## The `offload` execution strategy (Phase E3)

> **`[2026-08-09]` Read the two dated blocks below before this paragraph.** It
> opens on the **superseded** taxonomy — `offload` is not "the host-mediated
> extreme", it is the mode that **minimizes reconfiguration** — and on a
> six-dispatch shape that two corrections have moved to **30**. The paragraph is
> kept because the *non-aggregation* argument in it is still exactly right and
> is still the mode's clause in the distinguishability gate. Current state:
> `pattern/offload/README.md`.

`pattern/offload/` is the host-mediated extreme of the taxonomy: the host owns
the layer, holds every intermediate, and dispatches **six** registry GEMMs one
at a time — `q_proj k_proj v_proj output_proj up_proj down_proj` — each as a
**one-step `run_sequence` call**, so the driver-summed vector is six
submissions over six runlist entries and the mode aggregates *nothing*. That
non-aggregation is the mode's own clause in the Phase E distinguishability
gate, and it is why the six dispatches are six separate calls: under the ELF
ABI `run_sequence` merges every step it is given into one submission, so
handing it all six steps at once would record `coarse`'s shape, not this
mode's. Everything between the GEMMs — attention, softmax, both norms, the
GeLU — is host torch. `pattern/offload/README.md` has the full story; the
parts that cost time to learn:

**`[2026-08-08]` It is eight linear operators over five shapes — 30 dispatches
— and the artifact says so.** The two attention GEMMs (`4096x64x4096`,
`4096x4096x64`) resolve in no registry, and the sweep genuinely cannot stage
one, but that is a *catalogue* constraint rather than a hardware one: both are
measured passing on real NPU2, and their tiles are injected through the
`gemm_spec_fn` escape hatch (`gemm_spec_source: registry+injected`). So
attention is on the device and only the softmax between the two matmuls is
host torch, with both LayerNorms and the GeLU. The artifact records
`attention_path: "device_gemm_host_softmax"`.

**`[2026-08-09]` It has TWO packaging paths, and the difference between them is
the mode's defining axis.** The same 30 dispatches run either over five
xclbins with the `hw_context` torn down and reloaded before each one — 30
reconfigurations, the ELF path, still the **default** — or over **one** xclbin
holding all five shapes, loaded once with four kernel attaches. Set
`AIR_OFFLOAD_SHARED_XCLBIN=1` for the second; `make check-offload-shared` gates
it. The dispatch vector is identical either way, by design, because the mode
makes one `run_sequence` call per GEMM regardless — so reconfiguration is
counted separately, by `KernelCache.reconfiguration_counts()`, and both recipes
in `run_npu2_offload_peano.lit` pin it.

Two facts worth carrying: the shared chain **only builds where every module is
single-launch**, which rules out 4096, where the down-projection is a two-launch
`fused-cast`; and this mode's status as the noisiest of the four is
**measured to be removable** — 316.9% intra-walk spread on the ELF path against
17.6% on the shared one at 512 — though the switch changes the ABI as well as
the reconfiguration, so `_evict_context` is the leading candidate rather than a
demonstrated cause. Both are written up in
`docs/plans/transformer-layer-execution-studies/29-offload-n-streams.md`.

This mode used to dispatch six GEMMs and call itself a *hybrid* boundary,
keeping attention in host torch through `pattern/blocked_attention.py` — that
was the superseded taxonomy plus the catalogue constraint read as a hardware
one. `blocked_attention` still serves `runlist`, which is why it lives in
`pattern/` and not in a mode directory.

**The mode computes; the oracle checks; they may not share arithmetic.** This
mode does more host math than any other, so every host stage is torch
(`F.layer_norm`, `F.gelu(approximate="tanh")`, torch softmax) while the
oracle's boundaries come from the numpy operator references. A stage whose
"actual" and "expected" are the same function call compares a value against
itself. `run_blocked_attention_tests.lit` pins the two attention
implementations against each other on identical inputs — the comparison the
hardware gate cannot perform, because its attention inputs already carry
device GEMM error.

**One ELF serves four dispatches, and that is not aggregation.**
q/k/v/output_proj are the same `4096x768x768` module, so they share one
compiled drain ELF (three ELF compiles per gate instead of six); each dispatch
is still its own submission with its own recorded vector. The weights are
deliberately *not* static and `x` is re-uploaded for each of q/k/v — six
weight BOs per layer is the mode being itself; do not optimize it.

**A plain GEMM ELF's `instance_name` is the method's func name.** The drain
module emits `matmul_bf16` and the fused-cast module `gemm_cast_bf16`, and
`instance_name` must equal the emitted `func.func` name — a mismatch does not
fail to load, it times out with `ERT_CMD_STATE_TIMEOUT` a long way from the
cause. `pattern/offload/offload.py::_METHOD_FUNC` is the one place that
mapping lives.

## The `runlist` execution strategy (Phase E4)

`pattern/runlist/` is the fine-grained point of the taxonomy: `coarse`'s
dispatch schedule held fixed and every one of its dispatch units refined into
single-operator kernels — **391 single-operator entries over five runlists**.
The fused qkv becomes q/k/v (3 entries); the fused mha+out becomes host
blocked attention (the same implementation `offload` uses, unchanged) plus an
output_proj entry; each of the two normalization points stays at `coarse`'s
own 64-row banding (`builders.block.norm_rows`, imported — never re-derived
here) with each fused `addnorm` band call split into residual add, LayerNorm
and gamma multiply (2 × 64 × 3 = 384 entries); the fused ffn becomes up_proj,
GeLU, down_proj (3 entries). Each runlist is forced single-submission with
`require_single_submission=True`, and intermediates inside a runlist stay
device-resident (q/k/v chain to nothing but the readback; each band's
residual sum feeds its LayerNorm and multiply on device; `ffn_up`/`ffn_gelu`
chain into the down projection). `pattern/runlist/README.md` has the full
story; the parts that cost time to learn:

**The two operators that did not exist.** `builders/transpose.py` (iron's
`k_transpose`, re-expressed over `data_transfer_transpose/dma_bf16/`'s
movement — contiguous tiles plus a scalar tile kernel, because a bf16
DMA-stride transpose is illegal; the tile shape is baked into the OBJECT NAME,
`transpose_m64n96.o`, so shapes cannot overwrite each other) and
`builders/elementwise_mul.py` (built from nothing: `eltwise_add`'s streaming
2-D shape with `vector_mul`'s `arith.mulf`, **bf16 end to end because the AIE
vector unit does not legalize f32 vector element-wise multiply** —
`weighted_rms_norm.py` records the constraint and its bf16 epilogue is the
precedent that this form legalizes). Both hold the full opcheck contract;
transpose's check is BIT-exactness, and its lit recipe pins the zeros.

**Why the norm chains are banded when the decomposed kernels could stream.**
The `add`/`ln`/`mul` kernels have no L1 cap forcing 64-row dispatches, and
the first structure tried let them stream all 4096 rows per launch: 13
entries over two runlists, which landed BELOW `coarse`'s 131 and failed the
one ordinal clause the mode owns. That structure changed two variables at
once — operator granularity AND the dispatch schedule — and at the
normalization points it was 64× *coarser*-grained than `coarse` itself, so
its entry count measured the schedule change, not the decomposition. The
banded structure holds the schedule at `coarse`'s own (the band size is
imported from `builders.block.norm_rows`), so the two modes differ in exactly
one variable and `runlist_entries > coarse` holds by construction — every
coarse dispatch unit maps onto one or more finer units.
`pattern/runlist/README.md` records both structures and the measured tables.

**No GEMM ELF executes twice — measured, the hard way.** The first bring-up
shared one `4096x768x768` ELF for q/k/v inside one runlist; executions two and
three returned the exact corruption signature `offload` measured across
submissions (k 3.539e-1, v 3.561e-1 mean_rel_L1 against q's clean 9.3e-3;
offload's measured mode is 3.56e-1). So the reused-context failure holds
INSIDE a single runlist too — and there `offload`'s fix (context eviction) is
structurally unavailable, since entries of one artifact share its context by
construction. The four projections are therefore one module compiled to FOUR
artifacts, each with its own `hw_context`; the band `add`/`ln`/`mul` ELFs (no
runtime loop tiling) execute 64 times per chain and 128 times per layer in
one context each, measured clean at every stage boundary — the same class as
the block's 64-fold re-executed `addnorm`.

**The gamma multiply's second operand is a materialized broadcast.**
`elementwise_mul` takes two full tensors, so each LayerNorm weight is tiled
to one `[64, 768]` band on the host (`broadcast_row_weight`) and declared
static + content-keyed — every band multiplies by the same rows, so one
buffer serves all 64 multiplies. Under fault injection the content key
changes and the perturbed weight re-uploads — nothing special-cases the
injected path, which is what the driver's clean-equals-fault totals clause
checks.

**Transpose is validated standalone, not dispatched by the mode.** Its
consumer in iron's decomposition is the on-device attention scores GEMM, which
`[2026-08-08]` this hardware turns out to dispatch fine — `offload` now does.
The reason transpose is still not on a mode's dataflow is narrower than "the
hardware cannot": `offload` does the K transpose on the host as a contiguous
copy beside the head slice, and a device transpose feeding it would add an
entry while measuring nothing. It becomes a live question for `runlist`, whose
corrected form puts the whole attention interior on the device.

## The `fused` execution strategy (Phase E5)

`pattern/fused/` is the most fused point of the taxonomy: MLIR-level fusion
before compilation, via `stitch_elf` — MLIR-AIR's own production mechanism,
which is what makes this port additive rather than a duplicate of iron. The
layer executes as **one runlist submission of three entries over three
ELFs**: the D2 `qkv_proj` and `mha_out_proj` modules unchanged, then
`fused_tail` — a new ten-launch stitched module holding residual add,
LayerNorm and gamma multiply (ln1), the whole staged FFN, and add, LayerNorm,
gamma multiply (ln2), all over whole `[4096, 768]` tensors. Whole tensors are
the point: `coarse` pays 386 of its 402 sync boundaries restaging 64-row
`addnorm` bands through the host, and fusing the norms into one module
removes all of it. `pattern/fused/README.md` has the full story; the parts
that cost time to learn:

**One ELF for the whole layer is not available, and the reason is backend
settings, not symbols.** E1's `(method, tile_n)` naming fix removed the
symbol collisions — `fused_tail` co-links the FFN's drain (tile_n 128) and
fused-cast (tile_n 96) GEMMs beside six more launches without a
redefinition. What cannot be crossed is that one ELF is one aircc
invocation: FlashAttention requires `omit_pingpong="all"` +
`runtime_loop_tiling_sizes=[1, 1]` (it does not place otherwise) while the
4096-row GEMMs require `[2, 2]` for BD-ID recycling, and
`builders/mha_out_proj.py` records the settings as non-interchangeable. So
attention keeps its own ELF — exactly as every shipped LLM pipeline keeps
FlashAttention out of its stitched GEMM modules — and the ELF ABI aggregates
the three entries into one submission anyway.

**The normalization is streamed, not row-blocked, because a band cannot be
aliased.** Reusing coarse's `addnorm` inside one module would need 64
launches per normalization point, each reading a 64-row band of a whole
tensor — and a band at a nonzero row offset cannot be routed into a slice's
args clause (`memref.cast` cannot cast an offset subview back to the
identity layout; the row-0 trick in `o_gemv_ffn_multi.py` works only at
offset 0). The decomposed `add`/`ln`/`mul` builders walk all 4096 rows in
one launch each and are validated standalone at exactly 4096×768, so the
tail streams them — the same decomposition `runlist` measured clean, at
streaming rather than banded granularity.

**The fusion has a measured numerical cost.** Device attention (block
1.688e-2 `mean_rel_L1`) plus the decomposed norm tail (runlist 1.732e-2)
stack to 1.784e-2 at `atol_required` 7.896e-2 — a 1.27x margin under the
1e-1 ceiling, the thinnest of the four modes, every boundary still
`n_mismatch` 0. (`[2026-08-07]` refreshed after J7a moved `layer_norm_rows`
to f32 two-pass statistics; was 1.806e-2 at 7.572e-2, a 1.32x margin. The
mean improved and the margin tightened — they are different statistics.)

## The four-mode dispatch-vector table (Phase E's headline result)

> **`[2026-08-09]` EVERY ROW BELOW IS SUPERSEDED. Do not cite this table.** It
> records the four implementations as they stood before the taxonomy was
> corrected and the modes rebuilt against it. Two rows have since been
> re-measured at this same configuration:
>
> | mode | subs | entries | air | herd | sync | bytes |
> |---|---|---|---|---|---|---|
> | `offload` | 30 | 30 | 31 | 91 | 91 | 970,457,088 |
> | `runlist` | 17 | 427 | 50 | 488 | 451 | 190,513,152 |
>
> — and the other two cannot be restated here at all, because **`fused` no
> longer builds at 4096**: its stitched tail caps at 1365 rows, so it is bounded
> to 256..1024. A four-mode table at this configuration is therefore not
> something that can exist any more.
>
> **The current cross-mode comparison is at 512 and 1024**, walked twice, in
> `docs/plans/transformer-layer-execution-studies/27-common-ladder-result.md`.
> Build cross-mode tables from a ladder run, never from per-mode catalogue rows
> — those still sit at two different sequence lengths.
>
> The distinguishability *reasoning* below is kept because it is still how the
> gate works: the criterion is ordinal over driver-summed totals, never an
> absolute threshold.

Driver-summed totals over each mode's recorded `DispatchVector` rows, at the
forced configuration (seq 4096, emb 768, ffn 3072, 12 heads × 64), clean and
fault-injected runs totaling identically:

| mode | host submissions | runlist entries | air launches | herd launches | sync boundaries | bytes |
|---|---|---|---|---|---|---|
| `offload` | 6 | 6 | 7 | 19 | 19 | 139,984,896 |
| `runlist` | 5 | 391 | 14 | 404 | 403 | 165,347,328 |
| `coarse` | 4 | 131 | 12 | 146 | 402 | 202,902,528 |
| `fused` | 1 | 3 | 16 | 24 | 19 | 184,025,088 |

All four gating clauses of the distinguishability check hold on these
numbers: no two vectors are equal; `offload`'s 6 submissions exceed every
other mode's and it aggregates nothing (entries = submissions); `runlist`'s
391 entries exceed `coarse`'s 131; `fused`'s 19 sync boundaries are below
`coarse`'s 402. Both recorded-but-not-gating predictions also hold: `fused`
has fewer entries than `coarse` (3 < 131 — the faithful stitch did *not*
row-block its normalization into entries) and at least as many air launches
(16 ≥ 12 — ten launches fused into one ELF, counted once per artifact, which
is the signature 08e predicts for this mode).

Reading the columns: submissions order the modes by host mediation
(offload 6 → fused 1); entries order them by dispatch granularity (runlist
391 → fused 3); `coarse` and `runlist` sit within one sync boundary of each
other (402 vs 403) because both restage the same norm bands through the
host, while `offload` and `fused` both land at 19 for opposite reasons —
offload holds every intermediate on the host *between* single-GEMM
dispatches (its syncs are all argument traffic), fused holds every
intermediate on the device and syncs only the layer's true inputs and the
ten read-back boundaries. The `air`/`herd` asymmetry (16/24 for fused
against 12/146 for coarse) is the deliberate one from
03-measurement-model.md: `air_launches` counts launches in the compiled
module once per distinct ELF, `herd_launches` accumulates per dispatch step.

## `coarse`'s blend cells (C2 and C3)

`coarse` is defined as a per-workload **blend** of `runlist` and `fused`, and
until
`docs/plans/transformer-layer-execution-studies/28-coarse-blend-space.md`
nothing here expressed such a choice: the mode wrapped the D2 block, which is
one blend, chosen implicitly by D2 having been built at 4096.

Doc 28 derives the space from the artifact plans rather than from the
definition's wording, and it is **two axes, not a choice per operator** —
`fused` and `coarse` build their front from the same two modules and differ in
the tail alone:

| | tail stitched | tail banded | tail decomposed |
|---|---|---|---|
| **front `block`** | = `fused` | = `coarse` today (C1) | **C2** |
| **front `runlist`** | — | **C3** | = `runlist` |

Two of the six cells already have mode names, which is the sharp form of the
scoping problem: the space `coarse` blends over *contains the two things it
blends*, so "the best cell" would re-derive an endpoint. The resolution is the
word the definition already uses — *per workload*. `fused`'s stitched tail
needs a `plane_major` plane stride of `rows × cols` against the shim
`aie.dma_bd` cap of 1,048,576, so it caps at 1365 rows (`builders/norm_tail.py`)
and **at seq ≥ 2048 the entire stitched row is unbuildable**. `coarse` is the
mode you use where `fused` does not fit.

`pattern/coarse/cells.py` builds the two interior cells. It **composes**, it
does not reimplement: the block half's `_sequence_a` / `_sequence_norm` /
`_sequence_ffn` and the runlist half's `run_projections` /
`run_attention_interior` / `run_o_proj` / `run_norm_chain` / `run_ffn` are
called as they are, so a cell measures the code the D2 and E4 gates validated.
C1 is deliberately **not** built there — it is `pattern/coarse/coarse.py` over
`builders/block.py`, and a second implementation would be a fork.

Three things a composed cell needs that a single-half mode does not, each with
a guard:

- **A cross-half GEMM object collision check.** `ek.compile_gemm_mm` names its
  object from `(tile_m, tile_n)` alone while `tile_k_l1` is a compile flag, so
  two GEMMs agreeing on the first two and differing on the third write one file
  with two micro-kernels — the silent failure D2 hit between the FFN's
  up-projection and the o-projection. Each config already checked *itself*;
  `cells._check_cell_objects` is what checks across a block half and a runlist
  half.
- **A subset compile.** Both `compile_block_artifacts` and
  `compile_runlist_artifacts` take a `keys` argument, defaulting to everything,
  so a cell builds only the artifacts it dispatches instead of large ELFs
  nothing runs.
- **A gamma adapter.** The banded tail's `addnorm` takes the `[emb]` weight
  directly; the decomposed tail's `elementwise_mul` takes a host-materialized
  `[norm_rows, emb]` broadcast. The composer adapts at the seam rather than
  changing either callee.

The cells' shape is **predicted before they run**, host-side, by
`coarse_cells_structure.py` (`make coarse-cell-structure`, no NPU and no
Peano). Each half contributes independently — front `block` 1 submission / 2
entries, front `runlist` `2 + heads` / `4 + 3·heads`, tail banded 3 /
`1 + 2·blocks`, tail decomposed 3 / `3 + 6·blocks` — and **two of the four
combinations are already pinned by shipped gates**, which is what makes the
model checkable rather than plausible: it reproduces `coarse`'s recorded 4/131
and `runlist`'s 17/427 from the same arithmetic. At seq 4096 that gives:

| cell | front | tail | submissions | entries |
|---|---|---|---|---|
| C1 (`coarse`) | block | banded | 4 | 131 |
| **C3** | runlist | banded | 17 | **169** |
| **C2** | block | decomposed | 4 | **389** |
| C6 (`runlist`) | runlist | decomposed | 17 | 427 |

Each cell refines exactly one half of C1 and neither refines both, so the
ordinal claim the pair owns is `coarse 131 < C3 169 < C2 389 < runlist 427` —
ordinal, never a threshold, as everywhere else in Phase E. A cell landing
outside that bracket is not the cell it claims to be.

A cell is **not a fifth taxonomy point.** Its `execution_mode` is `coarse`'s
CSV value; the cell travels in the artifact as `blend_cell`, and cells separate
in a results tree by `study_case_label` and by the per-mode CSV filename.

**The measured answer, walked twice at 2048 and 4096:** `C1 < C2 < C3 < C6` on
averages and on minimums at both lengths, so **`coarse` is C1** — the cell it
already dispatched, now chosen rather than inherited. The front axis dominates:
a block front is ~1.5–1.6× faster than a runlist front, an effect roughly an
order of magnitude larger than the tail axis, which separates cleanly only at
4096. Because the winner is an *interior* cell, `coarse` stays distinct from
`runlist` and from `fused` and the taxonomy does not collapse to three points.
`pattern/coarse/coarse.py` records the selection and its reason in every results
artifact. Full derivation, the byte accounting and the two findings that are not
about `coarse`:
[`30-coarse-cells-built.md`](../../docs/plans/transformer-layer-execution-studies/30-coarse-cells-built.md).

## The registry sweep

The three GEMM-backed operators above take their tile sizes and method from
`kernel_registry`, which **raises** on a shape nobody measured rather than
guessing — hand-copied tile configs previously caused drift and stale-config
bugs. `sweep/` is the tool that makes a shape resolvable.

```bash
make registry-plan                # every (shape, candidate) it would measure. No NPU.
make registry-resolution          # every shape resolves. No NPU; this is the lit test.
make registry-writer-tests        # the writer's append-only guards. No NPU.
make sweep-families-tests         # the duplicated method table has not drifted. No NPU.
flock -x -w 1800 /tmp/mlir-air-npu.lock make registry-sweep
make registry-write               # fold the results into the registry. No NPU.
```

`sweep/` is five modules for the same ~800-line reason `opcheck` is three:
`sweep_families.py` says which shapes and candidates exist, `sweep_measure.py` is
one candidate end to end (the process the orchestrator forks), `registry_sweep.py`
is the orchestrator and the CLI, `sweep_report.py` is everything downstream of a
finished sweep — the resolution assertion and the family markdown — and
`registry_writer.py` owns the append-only JSON write. `sweep_report` imports two
functions from `registry_sweep`, so `registry_sweep` imports it inside `main()`;
the dependency has a direction and the CLI is the only place that needs both.

Per `(shape, candidate)` it builds the configuration, checks it through the same
`opcheck.py` comparison every operator here is gated on, times it, and keeps the
fastest that passes. `FAMILY=` selects which width of the case matrix to sweep;
`baseline_768` is the one Phase D needs and the one currently registered, and the
other two families are the same tool over a different id.

**The sweep never re-measures a registered shape.** The rows already in the
registry are what the ten shipped LLM deployments resolve against, and
re-measuring one into a different winner would change their behaviour without
anyone asking. The writer refuses, the orchestrator skips, and the JSON is edited
as text rather than re-serialized so every pre-existing byte is identical by
construction.

The one edit an existing entry accepts is **adding a method it has no row for**,
and only to an entry whose `used_by` says this sweep wrote it — a row a shipped
model owns is unreachable through that path, and a re-render that would change
any method already present raises instead of writing. It exists because a shape
can be registered and still unbuildable by the operator that needs it, which is
what `64×768×2304` was; see the `best.high` note below. Measuring such a shape
again needs `SWEEP_ARGS="--remeasure-registered"`, since measuring writes only to
the results directory and it is the write that is append-only.

Two things about these rows differ from the model-deployment rows next to them,
both forced by the sequence ladder starting at 64 and both written up on the
registry's detail page:

- **`herd` is per-row.** `M % (tile_m × herd_m) == 0` cannot hold at `M = 64`
  with the full 8 rows for either high-precision method's forced `tile_m`, so the
  short-sequence rows carry a per-method `herd` overriding the file-level `8×4`.
  `gemm_config()` hands that herd back next to the tile and `builders/gemm_spec.py`
  merges it into the build recipe, so the operator builders here take their herd
  from the row rather than defaulting to `8×4`. **Passing an explicit `herd_m` /
  `herd_n` to a builder overrides the row and will fail to build at the short end
  of the ladder** — leave them `None` unless you are deliberately experimenting.
- **The high-precision `atol` is carried forward at constant strictness rather
  than held constant.** The GEMM harness scales its inputs by `1/sqrt(K)`, so the
  output magnitude — and the absolute error of a fixed-relative-precision
  datapath with it — goes as `K^-1/2`. The published `1.5e-3` is the registry's
  "≈2.5× the measured worst case" rule evaluated at `K = 8192`; at this family's
  `K = 768` the same constant is a 3.3× *tightening*. `sweep_measure.py`'s module
  docstring has the three-point hardware calibration and why the low tier, whose
  error comes from a different mechanism, is left unscaled.

## Things that will bite you

**An object with no symbols still links.** `encoder.cc` and `addnorm_ffn.cc`
emit nothing unless `-DBUILD_FFN` and/or `-DBUILD_ADDNORM` is passed. Peano
produces a valid, small, empty `.o` and the link succeeds; the failure surfaces
much later at dispatch. `compile_kernels.py` enforces a size floor and an
explicit per-object symbol list for exactly this reason, and
`compile_encoder` / `compile_addnorm_ffn` refuse to build with both flags off.

**`encoder.o` and `addnorm_ffn.o` cannot share an ELF.** Both define
`ffn_gelu_bf16` and `ffn_eltwise_add_bf16_vector`. Linking them together is a
duplicate-symbol error. Rename one set, or pick one kernel per ELF. That is
why `encoder.cc` is built twice, to two objects with one half each: `addnorm`
links the addnorm half and `ffn` links `encoder_ffn.o`, so neither drags in the
symbols the other's ELF would collide on. `compile_kernels.py` checks both that
`encoder_ffn.o`'s FFN symbols are present and that its addnorm symbols are
absent — a presence check alone would not notice `build_addnorm=False` silently
ceasing to work, and the result would break only at link time in whichever
design happened to combine them.

**Two GEMMs of the same method but different `tile_n` used to be unable to share
an ELF, and the same mismatch across *separate* ELFs returned zeros instead of
failing.** Both are fixed as of Phase E1, and the history is here because the
silent half cost a gate cycle to localize and the fix is easy to undo by accident.

The root cause was one line of naming. `llms/shared/builders/gemm_builder.py`
minted both the MLIR symbol suffix and the `mm_*.o` filename from the GEMM
**method alone** — `_m32` / `mm_m32.o` for `drain`, `_m64` / `mm_m64.o` for
`fused-cast` — while `tile_n` arrived separately as a tile parameter and was baked
into the object as `-DDIM_N`. The two never met. But `mm_aie2p.cc` is compiled per
`(tile_m, tile_n)` and the private FuncOps the GEMM builder declares
(`f32_to_bf16_mn<suffix>`, `zero_f32_mn<suffix>`,
`op_has_no_registered_library_name<suffix>`) carry operand memref types that are
functions of `tile_n`, so **two GEMMs of one method at two `tile_n` are two
different micro-kernels with two different signatures.**

*Loudly, in one ELF:* `stitch_elf` collects each slice's private declarations into
one `set()` and re-parses, so the same symbol at two memref types is
`redefinition of symbol named ...`. Every shipped model shape lands on
`tile_n = 128`, which is why nothing hit this before; the study's FFN does not,
because `N = 768` cannot use `tile_n = 128` at `herd_n = 4` (`768 % 512 != 0`) and
settles on 96 against the up-projection's 128. So `build_ffn_module` **did not
build at any `baseline_768` point except `seq = 4096`**:

| seq | up-proj | down-proj | before E1 | after E1 |
|---|---|---|---|---|
| 64 … 2048 | `drain` t_n=128 | `drain` t_n=96 | collide | builds |
| **4096** | **`drain` t_n=128** | **`fused-cast` t_n=96** | builds | builds |
| 8192, 16384 | `fused-cast` t_n=128 | `fused-cast` t_n=96 | collide | builds |

That single row is why Phase D1's FFN point is at `seq = 4096` and why the block
runs there — a coincidence of the registry, not a property of the shape. It is
also why E1 comes before every execution strategy: E2–E5 each need more than one
sequence length, and one of them could not have been built at all.

*Silently, across separate ELFs:* `compile_gemm_mm` wrote its object named from
the method, so two operators of one method at different `tile_n` wrote the **same
file** with different contents and each ELF linked whichever landed last. Phase D2
has exactly that pair — the FFN's up-projection is `drain` at `tile_n = 128`, the
o-projection is `drain` at `tile_n = 96` — and building every object up front and
then every ELF gave the FFN a 96-wide micro-kernel for its 128-wide tile.

Nothing failed. The ELF built, loaded, dispatched, and returned **exactly zero
for 32 of every 128 up-projection columns** — 25% of `ffn_up`, uniformly across
every row block and every column tile, with the other 75% correct to
`mean_rel_L1 = 1.2e-2`. The GeLU passed the zeros through and the
down-projection's 3072-deep reduction smeared them over every element of the FFN
output, which reached the layer output as "54% of elements wrong". It is the same
shape of defect C4 found — a plausibly-shaped output that resolves cleanly from
the registry and does not compute — and the per-boundary stage list is what
localized it to `ffn_up` in one run.

D1 never met it because each operator ran in its own `opcheck.py` invocation and
no single operator holds two same-method GEMMs at different `tile_n`. Any caller
that builds several of these operators together did.

### What E1 changed, and the four ways to reintroduce it

`gemm_builder.gemm_variant_names(tile_m, tile_n)` is now the **single authority**
for both names: `(32, 128) -> ("_m32n128", "mm_m32n128.o")`. `tile_n` is a
required argument of `gemm_method_spec`, `with_tile_n` re-mints a resolved spec's
names, and `external_kernels.compile_gemm_mm_variant` derives the compile side
from the same function. `compile_block_artifacts` no longer interleaves — it
builds every object, then every ELF — and its docstring keeps the history above
and marks it historical.

Four things that will bite whoever touches this next:

1. **Never spell `sym_suffix=` and `out_name=` by hand.** Call
   `compile_gemm_mm_variant(tile_m, tile_n, tile_k_l1)`. A hand-written pair that
   disagrees with what the module asks for is either an unresolved symbol at link
   time or — the worse one — an object at the wrong `-DDIM_N` that links cleanly.
2. **Never write `spec["tile_n"] = N`.** Use
   `gemm_builder.with_tile_n(spec, N)`. Several callers retile after resolving,
   because the registry's `tile_n` for a narrow `N` can be numerically broken and
   the builder pads `N` out to admit a wider one. Before E1 the bare assignment
   was harmless; now it leaves the module asking for an object nobody compiles.
   `qwen25_0_5b` was doing exactly that and was **correct only by accident** — it
   asked for `mm_m32.o`, which happened to be compiled at `DIM_N=128`, the value
   it had overridden to. E1 turned that into a visible link failure.
3. **`cache.prepare_air_project` globs `mm_m*.o`; do not turn it back into a
   list.** `tile_n` takes four values across the registry, so there are up to
   eight of these objects and which ones exist depends on which shapes the caller
   resolved. An object that never reached `air_project/` fails inside aiecc,
   several frames from the list that omitted it.
4. **A module fingerprint does not notice a change in compile ORDER.** Verifying
   the interleaving removal meant deleting `build_peano/block_cache` first —
   `block_artifact_fingerprint` covers the resolved specs, the emitted MLIR, the
   kernel sources and the backend kwargs, all of which were unchanged, so all four
   ELFs would have been reused and the run would have proved nothing. The check
   that mattered was that every stage figure came back byte-identical, `ffn_up`
   included.

**Audit a shared-naming change before you spend hardware on it.** The E1 gate's
second leg is `make verify` in all ten shipped model directories and takes hours,
which is the worst possible place to discover a link mismatch. What found
`qwen25_0_5b` in about a minute instead: run each model's `compile_all_kernels`
with `external_kernels._compile_kernel` replaced by a recorder and
`KernelCache.compile_and_cache` replaced by one that only scrapes `link_with = "…"`
out of the module text, then compare the objects each module *references* against
the `-DDIM_N` each object was *built at*. Every module still builds (about a
second each) and no aiecc runs. Nine models agreed and one did not.

It is a script, so the next shared-naming change is one command:

```bash
python3 agents/scripts/audit-gemm-object-links.py    # all ten. No NPU, no lock.
```


**`addnorm` caps at 104 rows at width 768, so the layer is row-blocked.** Three
L3→L1 streams per tile (x, residual, weight) against a column's two shim MM2S
channels means exactly one kernel call per tile, and L1 then caps `rows` at
`herd_x × (what fits)`. The block's 4096 rows go through as 64 dispatches of the
64×768 shape D1 measured. That is not a tuning choice and raising it is not
available: `build_addnorm_module` raises above the cap, and it raises because the
two-trip form *miscompiles* rather than failing.

**A dispatch argument is a whole BO.** `run.set_arg` takes a buffer, never a
buffer and an offset, and `bo_pool` allocates per named buffer. An operator that
consumes one 4096-row tensor in 64 row bands therefore needs 64 buffers, and the
tensor has to be cut into them on the host — which is why the block is four
`run_sequence` calls rather than one, and why only the first of them
(`qkv_proj` → `mha_out_proj`) is fully device-resident.

**`builders/gelu.py`'s docstring overstates the erf/tanh gap.** It says the two
forms "differ by up to ~1e-3 absolute around |x| = 2 and would not survive
`rtol = 1.6e-2` on its own". Measured over `x ∈ [-6, 6]`: the worst absolute
difference is **4.7e-4**, at `x = 2.70`, and the `atol_required` for the
substitution over the whole range is **3.6e-4** — comfortably inside any `atol`
in this example. Using the tanh form is still correct, because it is what the
kernel computes; what is wrong is the claim that a tolerance check would catch
the substitution. In the whole D2 layer the swap moves `ffn_gelu` by at most
4.7e-4 absolute and the layer output by 6.1e-4, against a loosest `atol` of
1e-3 — invisible at the boundary *and* at the output.
`pattern/test_reference.py` pins the activation by identity against
`gelu_tanh_reference` instead.

**`best.high` is the fastest method, not the one every builder can use.**
`build_qkv_proj_module` folds its three-way C split into `fused-cast`'s separate
cast launch, so it can only build that method — `drain` casts inside the GEMM and
exposes no f32 scratch to slice. `drain`'s `tile_m = 32` beats `fused-cast`'s 64
on short sequences, so `best.high` is `drain` for most of the ladder even though
a measured `fused-cast` row sits beside it in the same entry. The QKV lookup
therefore asks for `fused-cast` by name (`resolve_gemm_spec(..., method=...)`),
which is still fully registry-driven — the tiles and the herd are that method's
own measured row at that exact shape — and the run log says when the pinned
method is not the shape's fastest. The other two GEMM builders wire either
method and read `best.high` unchanged.

So **"the shape is in the registry" and "the operator can build it" are
different claims**, and `make registry-resolution` checks the second one: each
role goes through its own builder's spec resolver rather than through
`gemm_config()`. `64×768×2304` is why. It was first registered with `drain` and
`direct` only — every `fused-cast` candidate in the grid of the day failed
(`mean_rel_L1 ≈ 0.46`, ~30% of elements wrong) — so it resolved for the generic
lookup and `qkv_proj` at `seq = 64` raised `KeyError`. A generic-lookup check
called that a pass.

**A legal `cast_tile_n` is not necessarily a correct one**, which is what that
row's failure turned out to be. `fused-cast`'s separate cast launch collapses
`M×N` to 1-D, hands each of 8 workers a contiguous chunk, and walks it in
`cast_tile_n` steps. At `64×768×2304` the chunk is 18432 and the harness default
of 2048 divides it exactly — and two of the nine sub-tiles come back **zero**.
The same candidate at `cast_tile_n = 1024` passes at 446 GFLOP/s. Nothing about
the shape predicts which values are safe, so `cast_tile_n` is a swept knob
(`CAST_TILE_N_PREFERENCE`, fused-cast only) rather than a derived one, and the
numerical check decides. `drain` still wins the tier here at 945; the
`fused-cast` row exists so `qkv_proj` has something to build.

**Multi-segment designs cannot use the xclbin output path.** Every `air.launch`
lowers to its own `aie.device` under an `aiex.configure` / `aiex.run` runtime
sequence, and the xclbin path names a single instruction blob on the aircc
command line, so a second segment collides on it: `edge 'air.insts.bin'
produced duplicate output path`. Build multi-launch designs with
`output_format="elf"`, as the shipped multi-launch llama builders do. Packaging
only — nothing downstream of it changes.

**The GEMM's error is ~1% of the output's own magnitude, not of one bf16 ULP.**
`-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16` puts the multiply in block
floating point, and `mean_rel_L1` lands at 9.3e-3 (registry) / 9.9e-3
(measured here) rather than the ~2e-3 a single epilogue rounding would give.
So a GEMM-backed operator's absolute error scales with how large you make its
output, and an `atol` is meaningless without the operand scale it was measured
at. `opcheck.py` uses the scale the registry's own GEMM sweep uses, and says
so.

**`-DADDNORM_PRE_ADD` changes numerics, not shapes.** Without it, statistics run
over `input` and the residual is added after normalization. With it, statistics
and normalization both run over `input + residual`, and the two-output form
exports the raw pre-add sum through `output2` as the next block's residual
stream. Getting this backwards produces correctly-shaped, subtly wrong
activations. The compile driver asserts the two objects differ.

**And `pre_add=` is a builder keyword, not a flag flip, because the two forms
are in different translation units.** `build_addnorm_module(pre_add=True)`
changes three things together: it links `addnorm_pre_add.o` built from
`addnorm_ffn.cc` rather than `encoder.o` built from `encoder.cc` (which has no
pre-add path at all), it calls `fused_add_layer_norm_1outs` rather than
`fused_add_layer_norm_2outs`, and it allocates three L1 tile buffers rather than
four. Call `compile_addnorm_kernel(pre_add=...)` to put the right object in the
working directory — the objects have different names, so getting it wrong is a
link error rather than a wrong answer, which is the one part of this that is
safe. What is *not* safe is the reference: `addnorm_reference` and
`addnorm_pre_add_reference` are separate functions on purpose, and checking one
form against the other's oracle is the mistake this whole variant exists to make
impossible to overlook. `addnorm_pre_add.o` is deliberately **not** the compile
gate's `addnorm_ffn_pre_add.o`: that one is the full 11-symbol build, and linking
it would drag every FFN matmul microkernel into a core that never calls one.

**`addnorm`'s row cap moves with `cols`, and the derived cap is not a target.**
`addnorm_max_rows(cols, ...)` returns `herd_x ×` the largest `rows_per_call` that
fits L1 — at `cols = 768` that is 80 post-add and 104 pre-add, against 120 at
`cols = 512`. It counts *allocations*; aircc ping-pongs the DMA-fed buffers on
top, so a shape sitting at the cap can still fail to place. Both `opcheck` rows
run 64 rows, comfortably under, which also makes the 512-wide and 768-wide
measurements directly comparable.

**Pre-add measures ~26× tighter than post-add at a *higher* relative error, and
that is the ordering, not the kernel.** `atol_required` is 6.6e-4 for
`64x768_pre_add` against 1.7e-2 for `64x512`, while `mean_rel_L1` goes the other
way (2.7e-3 against 1.9e-3). Post-add finishes with `+ residual` in bf16, so an
element where `norm × weight` nearly cancels the residual carries an absolute
error set by the *residual's* magnitude while its own value sits near zero — and
`rtol` covers none of that. Pre-add has no trailing add, so every error is
proportional to the output carrying it. Do not read the tighter number as the
better datapath.

**`-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16` must be a `-D`.** It has to be
visible before `<aie_api/aie.hpp>` to change `aie::mmul` behaviour, so a source
`#define` is too late. Both kernels `static_assert` on it under `BUILD_FFN`.

**`addnorm` breaks if a tile calls the kernel twice.** `build_addnorm_module`
requires `rows == herd_x * rows_per_call`, so the herd loop runs a single trip,
and it raises rather than emitting anything else. Two trips miscompile: at
`[8, 64]` with `herd_x=1`, one trip is exact (0 of 512 elements outside
tolerance) and two trips give 491 of 512, unchanged by fetching the weight
inside the loop or hoisting it out, by draining or discarding `output2`, by
disabling ping-pong, or by either lock-race-condition fix. Three L3→L1 streams
per tile against a column's two shim MM2S channels is what distinguishes it from
the two-stream builders next door, which loop fine. The symptom is
partly-correct values, so it reads as a tolerance problem.

**`copy_O_tile_rows` is numerically a no-op, and deleting it hangs the design.**
It reads every element of a FlashAttention O tile and writes it straight back.
That is the point: a KV block entirely above the causal diagonal runs no matmul,
so without it the consuming DMA never sees its buffer descriptor complete and
the run ends in `ERT_CMD_STATE_TIMEOUT`. It lives behind `-DCAUSAL_ROW_HELPERS`
in `flash_attention/kernel_fusion_based/attn_npu2.cc` alongside
`store_row_value` and `copy_row_values`; `mha_out_proj` links all three into
every causal variant. It does not *call* them, because the masking path it
composes (`apply_causal_mask`) fills a wholly-masked score tile with `-inf` and
lets the matmul run anyway — they are the entry points a block-skipping variant
needs, and keeping them linked is what lets one be added without re-deriving the
flag set.

**FlashAttention's `-D` flags are per *tile*, and a mismatch hangs rather than
fails.** `-Dlqp` is the Q tile size (`parallel_seq / num_q_tiles`), not the Q
chunk per launch; `-Ddk` / `-Ddv` are the `lkp`-sized tile while `-Ddk_full` /
`-Ddv_full` are the full head dimension. They instantiate the matmul
microkernels, so they must match the L1 buffer shapes the Python builder emits.
`builders/mha_attention.py` derives both from one config dict so they cannot
drift, and rebuilds with `force=True` because a shared working directory may
hold an `attn_npu2.o` built for another shape.

**`abs_err_max` is not the margin on an `atol`.** For an operator whose outputs
span a wide dynamic range — causal attention, where the first rows attend to a
handful of keys — the largest absolute error sits on a large-magnitude element
that `rtol` already covers. `opcheck.py` records `atol_required`, the smallest
`atol` the run would have passed at (`max(|a-e| - rtol*|e|)`), and that is the
number an `atol` should be quoted against.

**`-DDEBUG_AIE_KERNELS` needs a value.** The sources test
`#if DEBUG_AIE_KERNELS == 0` / `== 1`; a bare `-DDEBUG_AIE_KERNELS` expands to
nothing and fails to compile. Pass `=0` (pass input through) or `=1` (pass
residual through).

## Licensing

Mixed, deliberately. `compile_kernels.py`, the `Makefile` and the `.lit` are new
work under MIT, matching the rest of `programming_examples/`. The three files
under `kernels/` carry Apache-2.0 with AMD copyright: their numeric bodies are
substantially carried over from the source they were ported from, and
attribution is preserved even though this project is MIT. Both projects are
AMD-copyright, so Apache-2.0 files live here without conflict.

## Phase H findings: the multi-trip miscompile was the shim feed order

Phase H hardened the compiler against the miscompile behind
`builders/addnorm.py`'s one-trip guard. What the diff cannot show:

- **The root cause was NOT the ping-pong transform.** Compiling the two-trip
  addnorm shape with `--omit-ping-pong-transform=all` produced the identical
  481/512-wrong result. The real defect: `air-dma-to-channel` hoists each L3
  DMA into its own launch-scope loop, and when the input channels are
  packet-multiplexed onto one shim MM2S queue ("npu_dma_packet"), the host
  pushes whole channel after whole channel while the tile's BD ring expects
  the streams interleaved per iteration. One trip coincides; two or more
  trips misdeliver every packet after the first iteration. The new
  `air-fuse-packet-put-loops` pass restores the interleave, and
  `CanonicalizeAsyncOpDeps` models packet channels as one shared stream
  resource so the ordering chain survives pruning.
- **`func.call` is now visible to dependency analysis.** A callee with
  `llvm.emit_c_interface` classifies memref operands from its argument
  attributes (`llvm.readonly`/`llvm.writeonly`). An unannotated memref
  operand stays unknown: read-versus-write is not established, and the
  compiler never guesses a direction. Builders that want ping-pong across an
  external call must annotate the callee's arguments.
- **Unprovable ping-pong candidates are never transformed.** A candidate
  whose own per-iteration buffers have an unclassifiable use (e.g. an
  unannotated external call) is left untransformed with a warning: correct,
  just single-buffered. A candidate the transform would provably corrupt
  REFUSES to compile: a duplicated buffer that is READ with no producer
  provably refilling it every iteration. Opt-outs: `air.disable_ping_pong`
  on the loop, or `--omit-ping-pong-transform`. Both attributes are now
  validated by the AIR dialect verifier after every pass.
- **A never-read buffer is vacuously safe to rotate** (attempt 3). The reuse
  edge exists to hold the next fill behind the buffer's readers; once every
  use is classified, an empty consumer set means provably NO reader, so
  alloc-only scratch and fill-only staging buffers label and rotate
  harmlessly. Attempt 2 refused them, and then edited three pre-existing lit
  tests (`label_ping_pong_loops`, `label_ping_pong_multifill_alloc`,
  `label_ping_pong_disable_opt_out`) to add drain consumers that dodged the
  refusal — the harness halted on exactly that. The tests were the evidence;
  the predicate was the defect. All three are restored to their phase-base
  content and the `no_consumer` case in
  `label_ping_pong_external_call_proof.mlir` now guards against the refusal
  returning.
- **The proof's subject is exactly the buffers the rotation duplicates**
  (attempt 5, after two mis-drawn lines). Attempt 1 refused ANY
  unclassifiable use of an unprivatized memref, and broke 8 of 24 hardware
  tests — every GEMM-bearing operator — because a GEMM's accumulator
  legitimately carries data across K-loop iterations. Attempt 4 narrowed
  that to unprivatized memrefs whose data provably carries INTO the loop
  through a classified pre-loop fill (the hoisted-weight shape), and that
  still broke 3 of 10 shipped LLM deployments (`llama32_1b_int4`,
  `qwen3_0_6b`, `qwen3_1_7b`): each reads a loop-invariant weight or scale
  buffer, filled once before the loop, through an unannotated external
  call — and produces correct output, because one physical buffer read by
  every iteration has no rotation hazard. The rule that survives: a memref
  the rotation does not privatize (not in the loop's `hoist_alloc` set) is
  none of the transform's business, whatever its fill pattern; the Skip and
  Refuse verdicts apply only to the loop's own duplicated buffers.
- **When H1/H2 change a lit test's outcome, the input stays and the CHECKs
  move** (attempt 4). The three `ping_pong_shared_resident_ring*.mlir` tests
  had their unannotated `@acc` callee annotated with `llvm.emit_c_interface`
  + `llvm.readonly` to preserve their old transformed outcome — which
  deleted the coverage of exactly the path H2 changes. All three inputs are
  restored to their phase-base IR and now assert the new outcome: both
  get-loops SKIPPED with a warning (`-verify-diagnostics` catches the
  diagnostic; `--implicit-check-not=hoist_alloc/unroll` proves nothing was
  labeled). The transformed-path coverage each one used to carry lives in a
  new `*_annotated.mlir` companion, identical but for the callee argument
  attributes.
- **The hoisted-weight shape needs no refusal, measured** (attempt 5). At
  the miscompile's own shape (`cols=64, rows=8, rows_per_call=4`, two
  trips), hoisting the weight DMA out of the loop compiles under the scoped
  rule — the loop's own buffers are unprovable so it stays single-buffered,
  and `l1_w` is one physical buffer the rotation never touches — and the
  hardware output is exact, zero mismatches. That matches what
  `builders/addnorm.py` recorded all along: the two-trip corruption was
  identical whether the weight was hoisted or not *and with ping-pong
  disabled*, i.e. it was the shim feed order (fixed by
  `air-fuse-packet-put-loops`), never a rotation hazard. A refusal keyed on
  "data carries across iterations" was aimed at a hazard that does not
  exist when the rotation leaves the buffer alone — which is why it could
  only be satisfied by also refusing the three shipped models above.

Footgun: the two-trip fixture shape fully UNROLLS under ping-pong labeling
(trip count == unroll factor), so `air-ping-pong-transform`'s dependency
rebuild never runs on it; a fix living only there is untestable at 2 trips.

Footgun: skipping unprovable candidates means every unannotated
external-call loop that used to be ping-ponged silently drops to single
buffering — including the decode GEMV inner loop the shipped models use
(`backend_presets.py` records dropping ping-pong there cost 12.4 → 7.8
tok/s end-to-end, though that number measured a global disable, not this
per-loop skip). `make verify` gates correctness, not tok/s, so this shows
up in no gate. The recovery is per-design and deliberate: annotate the
callee's memref arguments (`llvm.readonly`/`llvm.writeonly`) so the proof
passes — input tiles are safe to mark readonly; an accumulator that is both
read and written by the kernel must stay unannotated, which is fine
whenever it is not one of the loop's own per-iteration buffers.

## Phase J1 findings: the norm-dispatch collapse is compiler-blocked, measured

J1's plan was to lift `builders/addnorm.py`'s one-trip guard — Phase H having
fixed the shim feed-order miscompile behind it — and collapse the layer's two
row-blocked normalization points (64 dispatches each, 128 of `coarse`'s 131
runlist entries) into one launch each: 4096×768 over the 8-column herd, 64
trips of 8 rows per tile. **That does not work, and the reason is measured,
not argued.** `air-fuse-packet-put-loops` fixed exactly the shape its fixture
pins — sibling `scf.for` put loops in one block, i.e. a ONE-column herd — and
no shape the block needs reaches it. The walk (pre-add, unannotated callee,
rtol 1.6e-2 / atol 2e-3, zero permitted mismatches, all on NPU2):

| shape (trips × rows_per_call, cols, herd_x)      | result |
|---|---|
| 2×4, 64, herd 1 (the fixture's shape)            | exact |
| 2×8, 768, herd 1                                 | exact |
| 8×8, 768, herd 1                                 | refuses to compile: shim BD exhaustion (16-BD cap, `aiex.dma_configure_task`) |
| 2×4, 64, herd 8                                  | 4070/4096 mismatched |
| 2×4, 64, herd 8, weight DMA hoisted              | 4039/4096 mismatched |
| 2×8, 768, herd 8                                 | 97,726/98,304 mismatched |
| 64×8, 768, herd 8 — **the J1 target shape**      | compiles; 3,130,958/3,145,728 mismatched |
| weight staged through L2, herd 8                 | placement failure: `no ShimNOCTile has sufficient DMA capacity` for the weight put |
| weight via L2 (broadcast OR per-column replica), herd 4 | routing failure: `'aie.connect' op … targets same dst as another connect op` on the first core tile |

Three distinct walls, none reachable from the builder:

1. **The fusion pass does not fire on multi-column herds.** With `herd_x >= 2`,
   `air-dma-to-channel` wraps each per-tile put loop in `scf.parallel` and
   leaves the broadcast weight's put loop beside them;
   `air-fuse-packet-put-loops` matches only sibling `scf.for` loops sharing a
   block, so its output IR is byte-identical to its input (checked in the
   `--debug-ir` dumps, pass 026 vs 027) and the packet feed-order corruption
   returns from the second trip on. Hoisting the weight does not help: x and
   residual still share a packet queue in the wrong order.
2. **Where fusion does fire (`herd_x == 1`), packet puts do not scale.** Each
   put in the fused loop lowers to its own simultaneously-active
   `aiex.dma_configure_task`; a shim tile has 16 BDs, so the trip count caps
   near 4–5 and 8 trips already refuse to compile. The refusal is loud, which
   is why the builder still permits multi-trip at `herd_x=1` only.
3. **Freeing the shim by staging the weight through L2 trips two further
   defects**: at herd 8 there is no MM2S channel left anywhere for the
   L3→L2 weight put (x/residual fill all sixteen), and at herd 4 the
   L2→L1 weight path — broadcast or per-column-replicated via a zero-stride
   put — makes `air-to-aie` emit conflicting stream-switch routes.

Consequences, encoded in the code rather than left in prose:
`build_addnorm_module` now permits multi-trip only at `herd_x == 1` and its
raise names the real mechanism (the old guard's "three streams against two
MM2S channels" was the trigger for packet multiplexing, not the fault
itself); `builders/block.py` keeps its row-blocked normalization sequences,
and `coarse` keeps its measured 131-entry vector. The collapse needs compiler
work first: packet put-loop fusion through `scf.parallel`, or loop-shaped
packet BD programs on the shim — either lands in `mlir/`, which a porting
phase does not touch. Every failing shape above reproduces from a plain
`XRTRunner.run_test` of `build_addnorm_module` (or the driver fixture's
`build` for the hoisted row) at the listed configuration.

## Phase H9 findings: packet put-loop fusion now reaches multi-column herds

H9 removed J1's wall #1: `air-fuse-packet-put-loops` fires at `herd_x >= 2`.
The multicolumn shape that was 4070/4096 wrong (2 trips × 4 rows, cols 64,
herd 8) is numerically exact, measured 4/4 repeat runs on NPU2, and 4 trips ×
herd 8 is exact 3/3. What the diff cannot show:

- **"Walk the scf.parallel body" — the obvious fix — is vacuous.** At
  `herd_x >= 2`, `air-dma-to-channel` wraps each hoisted per-tile put loop in
  its **own** launch-scope `scf.parallel` (one loop per wrapper), with the
  broadcast weight's loop beside them, unwrapped. No parallel body ever holds
  two loops, so per-body fusion finds nothing: the groups that matter span
  the wrappers. The pass instead *sequentializes* each eligible wrapper —
  cloning its body per iteration, ascending, at the wrapper's position — and
  then runs the ordinary sibling fusion over the flattened block. That is not
  a new lowering decision: `airrt-to-npu` unrolls launch-scope parallels into
  exactly that ascending sequential task order anyway (compare the
  `dma_configure_task_for` order in any `--debug-ir` pass_058 dump); the fix
  merely materializes the order early enough for the fusion to see it.
- **Why fusing across former iterations is safe here.** Each non-broadcast
  packet channel's per-column endpoint allocates to that column's own shim
  MM2S queue (`aie.shim_dma_allocation` in the placed IR: `channel_1_c` and
  `channel_2_c` both sit on column c's queue 0). Different iterations of the
  wrapper therefore never share a queue, and reordering between them is
  unobservable on any shared stream; the order each queue *does* depend on —
  that column's own per-trip interleave — is exactly what fusion restores.
  On column 0 the weight broadcast shares the queue with `x0`/`res0`, and the
  fused program order gives that queue the ring's exact expectation
  (`w, x0, r0` per trip) where today it got `w, w, x0(all), r0(all)`.
- **The residual ordering the argument does NOT close.** Tiles 1..7 receive
  the weight from column 0's queue but x/res from their own column's queue;
  their relative arrival is enforced by nothing stronger than task-issue
  order and transfer timing. This is not new: every shipped single-trip
  multi-column design (including this example's `block.py` normalization
  dispatches) already stands on the same discipline at trip 0. H9 extends it
  to trip boundaries and measured it stable (7/7 exact across 2- and 4-trip
  probes), but it is a timing discipline, not a proven order. If a
  multi-column multi-trip shape ever fails intermittently with corruption
  confined to the weight-adjacent buffers, start here.
- **Running the fusion upstream of the wrapper was rejected.** The hoisted
  put loops only reach their final shape after the last
  `air-isolate-async-dma-loop-nests`, which re-splits fused loops — and that
  pass necessarily runs after `air-dma-to-channel` creates the wrappers. There
  is no pipeline point where the loops exist fused-fusable and unwrapped.
- **Wrappers with a live result token are expanded too** (fixed in review:
  the first cut declined them). Declining looked harmless because the
  wrapper results are dead in practice (`air-dependency-canonicalize` prunes
  the terminal joins) — but that made the correctness fix conditional on a
  cleanup pass having run, and a token that survived pruning would have
  silently brought the whole-channel feed order back. The pass now replaces
  a live result with an `air.wait_all` over the init values and every
  iteration's reduce operand — semantically what the `scf.reduce` combiner
  (itself a token join) computed — inserted immediately before the result's
  earliest user. That placement is the load-bearing part: at the wrapper's
  own position the join would be a user of every member's result sitting
  before the fusion point, and the fusion's dominance check would
  (correctly) refuse the group; before the earliest user it lands after
  every later wrapper's clones, the fusion fires, and downstream consumers
  end up waiting on the fused loop itself. Shipped single-trip designs are
  untouched either way: their one-trip put loops canonicalize into bare
  puts before the pass runs, so their wrappers hold no candidates.
- **The shim 16-BD wall now binds multicolumn, loudly.** Each fused put still
  lowers to its own simultaneously-active `aiex.dma_configure_task`, so
  column 0's shim carries `3 × trips` tasks (weight + x + res) and refuses at
  6 trips (18 > 16) with the BD-exhaustion diagnostic; 5 trips is the deepest
  compilable width-8 depth today (measured at cols 64, `rows_per_call` 4:
  rows 160 compiles, rows 192 refuses). J1's 64-trip target therefore moves
  from *compiles silently wrong* to *refuses loudly* — the correct failure
  mode, and the next compiler phase's subject (loop-shaped packet BD programs
  on the shim rather than one task per iteration).

## Phase J7a findings: the norm-tail pipeline, and the two spec premises full compilation falsified

`builders/norm_tail.py` is the norm tail — add, LayerNorm, gamma scale — as three herds in one
segment joined by L1→L1 channels (`norm_tail_a2b`, `norm_tail_b2c`, both `[herd_x, 1]` bundles,
column i to column i). x and residual travel in ONE packed L3 buffer fetched by one DMA, so the
segment holds exactly two shim MM2S streams per column (packed in, gamma in) and the lowered IR
carries **zero packet-typed channels** — the sum and the normalized tensor never leave the array.
Measured on NPU2: `128x768` and `4096x768` both PASS at rtol 1.6e-2 / atol 5e-2 with zero
mismatches, `mean_rel_L1` 3.590e-3 / 3.620e-3, negative control failing as required. No herd
carries a placement attribute and no buffer a depth; `air-place-herds` seats all three herds and
the ping-pong labelling picks depth.

The operator's lit gate runs three arms, because each sees a failure the others cannot:
`check-norm-tail-structure` (host-only, but through the REAL aircc binary — every claimed shape
must route as three herds of 8 on 16 core-tile-to-core-tile flows, take at most 2 shim-facing
inbound streams per column, and count zero `npu_dma_packet` in every dump; so a third L3-facing
stream fails at compile time rather than past trip one on hardware, **and** a stage edge that
silently round-tripped through L3 fails too — which the earlier `air-dma-to-channel`-only version
could not see, `[2026-08-07]`), `check-norm-tail` (the
numerical check, whose spec rows also enforce `mean_rel_L1_max` 1.688e-2 — the whole-layer figure
the resident pipeline must beat, so a pipeline that is element-wise correct but round-trips its
intermediates through L3 fails even though every element sits inside rtol/atol), and
`check-norm-tail-fault` (the negative control, which must fail under injection).

Two things the phase spec proposed were falsified by FULL compilation — both of its probes had
stopped at `air-opt`, and both walls live further down, in aiecc:

- **The strided-callee route does not reach hardware.** The spec's plan for stage_add was the
  existing C add on two plane subviews of the packed L1 tile, with the callee DECLARING the
  strided types (`strided<[cols, 1], offset: rpc*cols>`). That module lowers cleanly through
  `air-dma-to-channel` — and dies in `air-to-aie`, which normalizes every external callee
  signature to the identity layout (deliberately: bare-pointer C ABI;
  `mlir/lib/Conversion/AIRToAIEPass.cpp`, the `normalizedInputs` block) and then finds
  `memref.cast` refuses strided-with-offset → identity. So the spec's open question — whether the
  lowered call passes a base pointer including the subview offset — is MOOT: no strided operand
  reaches an external kernel today, at any offset. The pipeline's add is instead
  `elementwise_add`'s own direct-codegen stage body (bf16 `vector<16>` addf on 1-D subviews of
  the packed tile held flat), which changes no C and no compiler.
- **Plane-major packing cannot be programmed at the block's shape.** `[2, rows, cols]` with its
  one 3-D fetch (`strides [rows*cols, cols, 1]`) hits the shim `aie.dma_bd` stride cap of 2^20:
  at 4096×768 the plane stride is 3,145,728 and aiecc refuses ("Stride 2 exceeds the
  [1:1048576] range"). At 128 rows it compiles and is numerically exact, so the failure is purely
  BD addressing. The shipped packing is per-row pairs — `[rows, 2, cols]`,
  `np.stack(axis=1)` — whose band fetch is contiguous (max stride `2*cols`) at any row count.
  Only stage_add's codegen ever reads the packed layout; the one C kernel in the design
  (`layer_norm_rows`) sees the contiguous sum tile.

Footguns that cost time, in the order they fired:

- **aircc ping-pongs BOTH of stage_add's tiles** — the DMA-fed packed tile and the channel-put
  sum tile. At `rows_per_call=8`, cols 768, that is 24+24+12+12 KiB on one tile and aiecc's
  allocator refuses. The builder's default is `rows_per_call=4` and its L1 check carries the
  measured ×2.
- **The asymmetric-input discipline is load-bearing, not decorative.** LayerNorm is
  scale-invariant, so if stage_add ever read the same operand twice (`LN(x+x) == LN(x)`), inputs
  drawn from one distribution would soften the failure. `prepare_norm_tail` draws x standard
  normal and residual `normal(0.75, 1.5)`; keep it that way.
- **The pipeline's honest numerical cost vs the fused addnorm kernel is one extra bf16
  rounding** (the normalized tensor materialized between stage_norm and stage_scale):
  mean_rel_L1 3.6e-3 against the fused kernel's 2.7e-3 at the same width. Both are far under the
  whole-layer 1.688e-2 figure the driver's clause bounds this by.
- **The norm stage's statistics were the round-3 review finding, and the fix is in the C
  kernel, not the pipeline.** `layer_norm_rows` shipped with a bf16 row sum and one-pass
  variance (`E[x²] − E[x]²`); on a row whose mean is large next to its spread — mean 8, σ 0.25,
  a valid input under the builder's contract — the cancellation drives the variance below zero,
  the NaN-clamp floors it at exactly zero, and the row normalizes by `1/sqrt(eps)`: ~700 of
  every 768 elements land outside tolerance while zero-mean activations pass untouched.
  `layer_norm.cc` now keeps f32 statistics, computes the variance two-pass (deviations exact in
  f32 at any common offset), and rounds once at the store. The `128x768_offset` opcheck row
  pins the regime — residual identically zero so the bf16 sum is exact and the row isolates the
  statistics — and every design linking `layer_norm.o` improved for free: the `layer_norm`
  operator's own rows went from mean_rel_L1 2.0e-3 to 8.1e-5 (rtol now covers every element),
  and the pipeline's from 4.4e-3 to 3.6e-3. The fused addnorm kernels (`encoder.cc`,
  `addnorm_ffn_norm.cc`) still carry the one-pass form; their file-header footgun notes say so
  and their gates measure it.

## Phase J7b: `ffn_accum` — the compiler-formed accumulator ring

`builders/ffn_accum.py` writes the FFN down-projection as the NAIVE K loop —
fetch C, call the in-place `ffn_matmul_bf16_bf16_up_proj`, store C, every K
step — and `air-hoist-dma-in-accum-pattern` lifts the C pair so the partial
sums stay L1-resident. Measured at aircc altitude on the real module: K-loop
data movement 4 → 2, zero packet-typed channels, full aiecc compile.
`ffn_accum_structure.py` gates both counts host-only.

Four walls were measured getting there; each is documented in the builder's
docstring, in short:

1. The naive form's three per-core input streams (A, B, C fetch) exceed the
   2-per-column shim MM2S budget and `air-dma-to-channel` packet-multiplexes
   every input, at any herd width — A auto-broadcasts and still counts 1.
2. Staging one operand through L2 fixes the shim and exposes the CORE: an
   AIE2P core tile has two S2MM DMA channels, and feed + direct fetch +
   hoisted C fetch is three. `aiecc` refuses the circuit route. (The phase's
   original 1×1 aircc probe never saw this: its streams had been silently
   packet-upgraded, and packets share ports.) Resolution: A and B share ONE
   memtile feed channel — two gets per K step, host-pre-tiled so every
   transfer is contiguous — leaving one core port for the hoisted C fetch.
3. The spec's pre-loop zero-and-STORE (`ffn_zero_bf16_up_proj` on a scratch
   tile, stored to `y`) is a second shim S2MM stream per column and
   `aie-place-tiles` refuses placement. The zero that fits the budget works
   on the L1 tile instead: the same kernel called inside the K loop under a
   `k == 0` guard, between the accumulator fetch and the accumulate — no
   DMA, so the column budget never sees it, and `y`'s initial DDR contents
   never reach the result whether or not the ring formed (review round 2;
   before that, `y` had to enter zeroed, which XRTRunner's zero-filled
   output placeholders both satisfied and concealed). Same pass also refuses
   the C pair's slots at herd_x=6 with free capacity — herd_x=4 places.

4. **A per-iteration L2 read offset is silently dropped past the unroll
   limit** — the wall that cost this phase its first session. Staging A whole
   in the memtile and putting each K step's slice from an advancing offset
   compiles, places, keeps zero packet-typed channels and hoists 4 → 2, and
   is wrong past two K steps: `aie.dma_bd` offsets are static, so at 2 trips
   the loop unrolls and each BD carries its own literal offset, while at 4+
   the chain cycles with every offset frozen at 0. The core then stalls, the
   accumulator store never fires, and `y` comes back byte-identical to what
   the host wrote (seed it with 1.0 and 4096/4096 elements return 1.0); at
   the spec shape it does not return at all — `ERT_CMD_STATE_TIMEOUT`, which
   is how this was first met. Not ping-pong (`omit_pingpong="all"` is
   identical). Resolution: **both operands advance on the L3 side**, each
   staged in a small per-K-step buffer put from a static offset, so no BD
   ever needs a moving offset. This is why B always worked and A did not.
   Reproduce with `agents/probes/probe_ffn_accum_bd_offset.py`.

Measured at `64x3072x768` (herd 4×1, `tile_n` 192, `tile_k` 32, 96 K steps):
`mean_rel_L1` 1.417e-2, `abs_err_max` 1.831e-3, `atol_required` 1.383e-3, zero
mismatches over 49152 elements at the GEMM tier's 5e-3 — a 3.6× margin. The
relative error is an order above the other GEMM rows and that is the ring's
honest cost, not a defect: the in-place kernel's C is bf16, so the running sum
rounds to bf16 once per K step (96 times here) where a drain-to-f32 GEMM rounds
once.
