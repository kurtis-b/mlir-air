# Transformer Layer — AIE2P Device Kernels

Phase A of the [transformer-layer execution
studies](../../docs/plans/transformer-layer-execution-studies/04-phase-a-kernels.md):
the C++ device kernels a full encoder/decoder block needs, compiled with Peano
for AIE2P.

It also holds Phase B's runtime-seam gate. The kernel half needs no NPU, which is
what keeps it safe as a PR gate; the seam half is split the same way — host-only
unit tests, plus one hardware gate.

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
| `opcheck.py` | Phase C's numerical check: the CLI, the results artifact, and the fault-injection negative control |
| `builders/elementwise_add.py` | 2-D element-wise add and the `causal_mask=` keyword over it, plus their FP32 reference |
| `builders/layer_norm.py` | Multi-row LayerNorm over `layer_norm_rows`, plus its two-pass FP32 reference |
| `builders/addnorm.py` | Weighted LayerNorm + residual over `fused_add_layer_norm_2outs`, weight as a runtime argument, plus its FP32 reference |
| `run_npu2_<op>_peano.lit` | One per operator: that operator's numerical gate on a real NPU |
| `run_npu2_fault_control_peano.lit` | Phase C's negative control: the injected run must FAIL |

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

## The Phase C operator checks

`opcheck.py` is the single numerical entry point for every operator this port
lands, and `builders/` holds the operators themselves — one
`build_<name>_module()` function per operator with its FP32 reference beside it,
no operator class and no `op.py`/`design.py` pair.

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

## Things that will bite you

**An object with no symbols still links.** `encoder.cc` and `addnorm_ffn.cc`
emit nothing unless `-DBUILD_FFN` and/or `-DBUILD_ADDNORM` is passed. Peano
produces a valid, small, empty `.o` and the link succeeds; the failure surfaces
much later at dispatch. `compile_kernels.py` enforces a size floor and an
explicit per-object symbol list for exactly this reason, and
`compile_encoder` / `compile_addnorm_ffn` refuse to build with both flags off.

**`encoder.o` and `addnorm_ffn.o` cannot share an ELF.** Both define
`ffn_gelu_bf16` and `ffn_eltwise_add_bf16_vector`. Linking them together is a
duplicate-symbol error. Rename one set, or pick one kernel per ELF.

**`-DADDNORM_PRE_ADD` changes numerics, not shapes.** Without it, statistics run
over `input` and the residual is added after normalization. With it, statistics
and normalization both run over `input + residual`, and the two-output form
exports the raw pre-add sum through `output2` as the next block's residual
stream. Getting this backwards produces correctly-shaped, subtly wrong
activations. The compile driver asserts the two objects differ.

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
