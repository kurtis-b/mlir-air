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
ninja -C build-xrt check-programming-examples-transformer-layer

flock -x -w 1800 /tmp/mlir-air-npu.lock make runlist-gate   # NEEDS AN NPU
```

## The Phase B runtime seam

`runlist_gate.py` is
[Phase B](../../docs/plans/transformer-layer-execution-studies/05-phase-b-runtime-seam.md)'s
gate. **It currently fails, on purpose.** The plan's load-bearing assumption — that several
separately-compiled ELFs can be bound into one XRT `hw_context` and submitted as one runlist —
does not hold on XRT 2.21.0 / NPU2. An AIR ELF is a *full* ELF: it carries its own array
configuration, and a `hw_context` accepts exactly one of those.
[05a](../../docs/plans/transformer-layer-execution-studies/05a-phase-b-runlist-spike-result.md)
records every route that was tried and the three that remain, none of which is a Phase B change.

What the seam does deliver, and what legs B–D of the gate measure:

| | |
|---|---|
| `llms/shared/infra/dispatch.py` | Groups a dispatch sequence into the submissions the hardware allows, and owns the six-field dispatch vector. Refuses to build a runlist spanning configurations — one of those *executes* and returns wrong numbers with no error, so refusing is the only safe behaviour. |
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
