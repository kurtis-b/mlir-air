# Transformer Layer — AIE2P Device Kernels

Phase A of the [transformer-layer execution
studies](../../docs/plans/transformer-layer-execution-studies/04-phase-a-kernels.md):
the C++ device kernels a full encoder/decoder block needs, compiled with Peano
for AIE2P.

This example builds device kernels; it does not dispatch them. There is no
`run` target and no NPU is required, which is what makes the suite safe as a PR
gate.

```bash
make compile                 # build every object and check its symbols
ninja -C build-xrt check-programming-examples-transformer-layer
```

## What lives here

| File | Contents |
|---|---|
| `kernels/encoder.cc` | Encoder-block kernels: staged FFN (`-DBUILD_FFN`) and weighted add-norm (`-DBUILD_ADDNORM`) |
| `kernels/addnorm_ffn.cc` | Fused add-norm + FFN staging, both residual orderings behind `-DADDNORM_PRE_ADD` |
| `kernels/elementwise.cc` | `eltwise_vadd` and `gelu_tanh_approx_bf16`, textually included by the two above |
| `compile_kernels.py` | The compile-and-check driver the lit test runs |

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

## On the size of the two kernel sources

`encoder.cc` and `addnorm_ffn.cc` both run past the ~800-line module guideline
in
[02-porting-conventions.md](../../docs/plans/transformer-layer-execution-studies/02-porting-conventions.md).
They are kept whole rather than split, for two reasons. Each already carries the
seam the guideline asks for — `-DBUILD_FFN` and `-DBUILD_ADDNORM` select two
disjoint halves that share no code, so a build takes only the half it needs. And
the phase document names these two artifacts explicitly, so splitting them would
make the port harder to check against the plan, not easier. Splitting along the
existing `#ifdef` boundary remains a clean follow-up if the guideline is meant
to bind C++ sources as strictly as it binds Python modules.

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
