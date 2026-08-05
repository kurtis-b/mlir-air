# Transformer Layer — AIE2P Device Kernels

Phase A of the [transformer-layer execution
studies](../../docs/plans/transformer-layer-execution-studies/04-phase-a-kernels.md):
the C++ device kernels a full encoder/decoder block needs, compiled with Peano
for AIE2P.

It also holds Phase B's runtime-seam gate, the Phase C/D1 operator builders and
their numerical checks, the Phase C4 registry sweep, and Phase D2's block
integration gate — one whole `encoder_bert` layer assembled from those operators
and compared against a shared golden model at every boundary it passes. The kernel half needs no
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
| `builders/addnorm.py` | Weighted LayerNorm and a residual in **either order** (`pre_add=`), weight as a runtime argument, plus one FP32 reference per ordering |
| `builders/qkv_proj.py` | One GEMM over the fused `[K, 3K]` weight with C split three ways on the device, plus its FP32 reference |
| `builders/gelu.py` | The FFN activation stage over `ffn_gelu_bf16`, plus its FP32 tanh-approximation reference |
| `builders/ffn.py` | Staged up-projection / GeLU / down-projection composition, plus its FP32 reference |
| `builders/mha_attention.py` | Attention staging: the seq-first FlashAttention design point, its kernel `-D` flags, and the chunked FP32 oracle |
| `builders/o_proj.py` | O-projection staging: the registry lookup, the GEMM sub-kernel, and its FP32 oracle |
| `builders/mha_out_proj.py` | The entry layer that composes the two into one ELF, plus the composed FP32 reference |
| `builders/block.py` | Phase D2: the five operator launches assembled into four `KernelCache.run_sequence` calls, and every boundary read back |
| `builders/block_cache.py` | When a cached block ELF is the ELF the current source would build: the artifact fingerprint and its persistence |
| `builders/test_block_cache.py` | Host-only: which cached block ELFs `check-block` reuses, and which it rebuilds |
| `pattern/reference.py` | The shared golden model — iron's draw order and structure, this repository's FP32-from-bf16 numerics, both workload variants, every boundary. Phase E's strategy directories import this one copy |
| `pattern/test_reference.py` | Host-only: that the golden model is the layer it claims to be |
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
