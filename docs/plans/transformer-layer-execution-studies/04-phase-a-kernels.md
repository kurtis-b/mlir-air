# 04 — Phase A: AIE2P Device Kernels

Port the C++ device kernels. This phase comes first because it proves the Peano / `aie_api`
path end to end with the least dependency on anything else.

## What is being ported

From `iron/aie_kernels/aie2p/`:

| Source | Lines | What it adds |
|---|---|---|
| `encoder.cc` | 1061 | New. Encoder-block kernels backing the `ffn` operator. |
| `addnorm_ffn.cc` | 931 | New. Fused add-norm + FFN staging. |
| `addnorm_ffn_addnorm.cc` | 936 | New. Near-duplicate of the above with an extra trailing add-norm stage. |
| `mm.cc` delta | +1463 | `matmul_init_*` (zero-then-multiply) and `matmul_with_acc_*` (accumulate into an explicit `pAcc`) variants across the existing r/s/t template family, plus their instantiation macros. |
| `softmax.cc` delta | +68 | Two-pass streaming softmax: `init_softmax_scale_buffer`, `partial_softmax_rows_bf16`, `normalize_softmax_rows_bf16`, `copy_softmax_scale_bf16`. |
| `layer_norm.cc` delta | +104 | Multi-row `layer_norm_rows(input, output, cols, rows)` and `add_layer_norm_rows(in1, in2, out, cols, rows)`. |
| `mha.cc` delta | +170 | Causal-mask helpers: `copy_O_tile_rows`, `store_row_value`, `copy_row_values`. |
| `aie_kernel_utils.h` | — | Pragma abstraction shim. Port only if MLIR-AIR has no equivalent. |

These are plain AIE2P core kernels: `#include <aie_api/aie.hpp>`, `aie::mmul` / `aie::vector` /
`aie::load_v`, `extern "C"` entry points taking plain pointers and `int32_t`, with tiling fixed
at compile time through `-DDIM_M` / `-DDIM_K` / `-DDIM_N` and feature macros
(`-Dbf16_bf16_ONLY`, `-DROUND_CONV_EVEN`, `-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`,
`-DBUILD_FFN`, `-DGENERATE_MATMUL_WITH_ACC_KERNELS`, `-DGENERATE_MATMUL_INIT_KERNELS`,
`-DOPT_PERF_ENABLED`).

Because the build uses `--no-xchesscc --no-xbridge --peano`, the `__chess__` branches of
`aie_kernel_utils.h` are never exercised — no chess intrinsics are in play.

## Destination

`programming_examples/transformer_layer/kernels/`

MLIR-AIR's convention is that `.cc` lives next to the example that owns it, not in a central
kernel library. `runtime_lib/` holds host runtimes and contains no AIE core kernels.

## Build path

Compile through the existing `llms/shared/infra/external_kernels.py` mechanism rather than a
new one. It already resolves the `aie_api/aie.hpp` include directory
(`_get_aie_include_dir()`: `which aie-opt` → `MLIR_AIE_INSTALL_DIR` → `my_install/mlir-aie`)
and carries the aie2p Peano flag set:

```
-O2 -std=c++20 --target=aie2p-none-unknown-elf -DNDEBUG
-I <aie-opt>/../include -D__AIE_API_AIE_ADF_HPP__
-Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
```

Add a `compile_*` entry point per new kernel, following the shape of the existing
`compile_gemm_mm` / `compile_attn_npu2` / `compile_silu_and_mul`.

## Constraints and corrections

**Target the right GEMM source.** `[Codex]` The shared LLM path compiles
`matrix_multiplication/bf16_in_fp32_out/mm_aie2p.cc`
(`shared/infra/external_kernels.py:133`), **not** the bf16-output variant. The repository holds
several `mm_aie2p.cc` files with different ABIs:

```
matrix_multiplication/bf16/mm_aie2p.cc
matrix_multiplication/bf16_in_bf16_out/mm_aie2p.cc
matrix_multiplication/bf16_in_fp32_out/mm_aie2p.cc     <- the LLM path
matrix_multiplication/i8/mm_aie2p.cc
matrix_multiplication/i16/mm_aie2p.cc
```

Map each iron kernel variant to the exact AIR builder ABI and source file it must satisfy, then
validate symbol uniqueness, linking and runtime argument compatibility. Extending the wrong
file produces code that compiles and is never used.

**Do not port iron's `llvm-objcopy --redefine-sym` step.** `[Codex]` iron renames symbols
post-hoc so two differently-parameterized copies of `mm.cc` can coexist in one archive.
MLIR-AIR already solves this at the source: `compile_gemm_mm(...)` takes `sym_suffix=` and
`out_name=` precisely so two differently-tiled `mm.o` variants link into one ELF. Use the
existing mechanism.

**Merge the near-duplicate.** Convention rule 8: `addnorm_ffn.cc` and
`addnorm_ffn_addnorm.cc` differ by a trailing stage. Make it one source behind a `-D` flag,
matching how the repository already parameterizes kernels.

**`.cc` including `.cc` is fine.** iron's kernels `#include "../generic/add.cc"`, `"gelu.cc"`,
`"zero.cc"` — single translation units rather than separate compilation. This is **not** a
deviation: `matrix_multiplication/bf16_in_fp32_out/mm_aie2p.cc` includes `zero.cc` the same way.
Keep the pattern; just place the included sources where the includes can resolve.

**Verify, do not assume, the flags.** iron's compile flags and `-D` macro sets were tuned
against its own toolchain pinning (`mlir_aie==v1.2.1`). MLIR-AIR is on mlir-aie v1.4.0. Check
each flag survives rather than transplanting the flag string.

## Work items

1. Confirm whether `aie_kernel_utils.h` has an MLIR-AIR equivalent; port it only if not.
2. Land `encoder.cc` and the merged `addnorm_ffn.cc` under
   `programming_examples/transformer_layer/kernels/`, with MIT headers if rewritten or
   Apache-2.0 if verbatim (convention rule 6).
3. Diff iron's `mm.cc` additions against `bf16_in_fp32_out/mm_aie2p.cc` and extend that file
   with the `matmul_init_*` / `matmul_with_acc_*` families.
4. Apply the `softmax.cc`, `layer_norm.cc` and `mha.cc` deltas to their MLIR-AIR counterparts
   (`programming_examples/softmax/softmax.cc`, `layer_norm/`,
   `flash_attention/kernel_fusion_based/`).
5. Add `compile_*` entry points to `external_kernels.py`.
6. Add a compile-only `.lit` test and register a
   `check-programming-examples-transformer-layer` lit suite in
   `programming_examples/CMakeLists.txt`.
7. Run clang-format / clang-tidy.

## Gate

Every kernel compiles to a `.o` with Peano, and the compile-only `.lit` test passes:

```bash
ninja check-programming-examples-transformer-layer
```

Compile-only means no NPU is required, which keeps this suite safe as a PR gate.

## Risks

- Flag drift between mlir-aie v1.2.1 (iron) and v1.4.0 (MLIR-AIR) may change codegen or break
  compilation outright.
- Extending `mm_aie2p.cc` touches a file the ten shipped LLM deployments depend on. Any change
  there requires re-running `make verify` across all of them — see
  [13-verification-and-acceptance.md](13-verification-and-acceptance.md).
