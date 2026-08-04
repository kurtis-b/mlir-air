# 06b — Phase C2: `qkv_proj` and `ffn`

Second of Phase C's four sub-phases. Read [06](06-phase-c-operators.md) for the overview,
[06a](06a-phase-c1-gate-and-small-operators.md) for the check mechanism you must reuse, and
[02](02-porting-conventions.md) for binding house style.

C1 built `opcheck.py`, its CLI contract, its results artifact and its fault-injection negative
control. **Extend it; do not build a second checker.** Every constraint in
[06a](06a-phase-c1-gate-and-small-operators.md#what-the-driver-checks-and-why-it-is-shaped-this-way)
applies here unchanged — the driver runs the same freshness, re-derived-verdict and
negative-control checks against your operators.

## `qkv_proj`

GEMM `A(M, K) @ B(K, 3K)` with C split three ways at the runtime-sequence level, so Q, K and V
land in separate output buffers without a host-side slice.

Closest existing analogue: `llms/shared/builders/rms_gemms_rope_multi.py:191`
(`build_rms_gemms_rope_module`) minus the RMSNorm and RoPE launches. Read it before starting — it
is 6 launches and 13 memref args, and the argument-wiring pattern is what you are reusing.

**Tiles come from the registry, never from a constant.** Resolve through
`gemm_registry_config(m, k, n)` (`llms/shared/builders/gemm_builder.py:29`), which returns the
method spec merged with the registry tiles. It **raises** on an unmeasured shape — deliberately,
because hand-copied tile configs previously caused drift bugs (`registry_lookup.py` docstring).

Only 5 of the case matrix's 108 projection-GEMM shapes are registered today. That is C4's problem,
not yours: validate `qkv_proj` at shapes that already resolve, and record in your structured
report which shapes you could not reach. Do **not** invent a fallback that guesses tiles. If you
need a shape outside the registry, use the existing precedent — the `gemm_spec_fn` injection hook
(`rms_qkv_qknorm_rope_multi.py:441`, already shipping for `qwen3_4b`) — and record the injected
spec in the results artifact so the guess is visible rather than silent.

iron's oracle (`iron/operators/qkv_proj/reference.py`) computes in bf16. Yours computes in FP32
from bf16-rounded inputs, per [06](06-phase-c-operators.md#the-numerics-standard--do-not-port-irons).

## `ffn`

Staged up-projection → GeLU → down-projection, with `down_proj_depth` memory-tile accumulation
staging. iron's `ffn/design.py` is 1096 lines; convention rule 5 requires splitting it along the
staging seam it already has internally.

`programming_examples/ffn_swiglu/prefill/` is the structural model — in particular its `Makefile`,
whose `run` target has a `compile-kernel` prerequisite that builds the `.cc` with Peano and copies
the object into `build_peano/air_project/`. Copy that shape. Do **not** copy its tolerances
(`rtol=1e0, atol=0.5`), which are not a gate.

**The activation already exists.** `ffn_gelu_bf16` in `transformer_layer/kernels/elementwise.cc`
was landed and symbol-checked in Phase A. Your new work is the AIR staging, not the kernel.

It is the **tanh approximation**, not exact erf GeLU. iron's oracle calls
`torch.nn.functional.gelu`, whose default is the erf form; at iron's `4e-2` tolerance the
difference hides, at `rtol = 1.6e-2` it does not. Compute the reference with the tanh
approximation and state that in the docstring, so the next reader does not "fix" it back.

Kernel-build traps carried over from Phase A, all recorded in
`programming_examples/transformer_layer/README.md`:

- `-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16` must be a `-D` on the command line. It has to be
  visible before `<aie_api/aie.hpp>` to change `aie::mmul` behaviour, so a source `#define` is too
  late.
- `-DDEBUG_AIE_KERNELS` needs a **value** (`=0` or `=1`); a bare `-D` expands to nothing and fails
  to compile.
- `encoder.cc` and `addnorm_ffn.cc` emit an empty-but-valid object if built without `-DBUILD_FFN`
  / `-DBUILD_ADDNORM`. The link succeeds and the failure surfaces at dispatch.
  `compile_kernels.py` enforces a size floor and a per-object symbol list for exactly this reason;
  extend that plan rather than side-stepping it.
- `encoder.o` and `addnorm_ffn.o` cannot co-link — both define `ffn_gelu_bf16`.

## Work items

1. `build_qkv_proj_module(...)` in `transformer_layer/builders/`, tiles from
   `gemm_registry_config`, C split three ways at the runtime-sequence level.
2. `build_ffn_module(...)`, GeLU-shaped, split along its staging seam per rule 5.
3. FP32 references beside each builder as module-level functions, tanh-approximation GeLU.
4. Register both operators with `opcheck.py --list`, so the driver enumerates them.
5. `run_npu2_qkv_proj_peano.lit` and `run_npu2_ffn_peano.lit`, plus their `Makefile` targets.
6. Registry rows for every validated `(kernel, shape)`, carrying `mean_rel_L1`, `Used by`, status.
7. `black`; module docstrings stating the contract and its footguns.

## Gate

```
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  ninja -C build-xrt check-programming-examples-transformer-layer
```

Every test in the suite, including C1's and the pre-existing compile, seam and runlist-gate tests.
Plus the driver's objective check: freshness, re-derived verdict, and a fault-injection run per
operator that must fail.

## Constraints

- **Do not modify `llms/shared/`.** Call `gemm_builder`; do not edit it. Modifying it triggers the
  ten-model `make verify` regression rule ([13](13-verification-and-acceptance.md)).
- Wrap every NPU command in `flock -x -w 1800 /tmp/mlir-air-npu.lock`. Never take
  `/tmp/npu.lock`.
- If a shape will not place, report it as a blocker with the shape and the error. Do not relax a
  tolerance, skip a shape silently, or guess a tile config to get past it.
