# 06a — Phase C1: The gate mechanism and the small operators

First of Phase C's four sub-phases. Read [06](06-phase-c-operators.md) for the overview and
[02](02-porting-conventions.md) for binding house style before starting.

C1 does two things: it builds the numerical-check mechanism that C2, C3 and C4 all reuse, and it
lands the four operators small enough to shake that mechanism out.

## Step 0 — close the Phase B gate gap

Already done outside the harness, recorded here so the state is not mistaken for yours:
`run_npu2_runlist_gate.lit` now runs `make runlist-gate` on real hardware inside
`check-programming-examples-transformer-layer`, and the nine `.o` files committed by `bf69ed69`
are untracked. Do not undo either.

The reason matters for how you work: Phase B's driver-run gate held only a compile-only test and a
host-only test, so its central hardware claim reached the status board on the session's own
say-so. Your gate is real hardware. Treat a passing self-assessment as worth nothing until the
driver's gate agrees.

## The check mechanism

`programming_examples/transformer_layer/opcheck.py` — one module, the single entry point for every
operator's numerical check in C1 through C4.

Do not write a new comparison. `XRTRunner.run_test` (`python/air/backend/xrt_runner.py:165`,
`_check_outputs` at `:394`) already is the required gate: `np.isclose` over the full output,
`max_mismatch_percentage` defaulting to 0, bf16 upcast to float64 before comparison. With
`report_precision=True` it prints

```
[precision] Output 0 (N elements): mean_rel_L1=… | rel_err max=… | abs_err max=… | rtol=… atol=…
```

which is exactly the field set a `kernel_registry` row carries. `weighted_rms_norm`,
`eltwise_add` and `flash_attention` already call it at registry tolerances and are the examples to
follow. `layer_norm` (`rtol=5e-2, atol=5e-1`), `ffn_swiglu` (`rtol=1e0`) and the fused builders
(`0.2 / 0.5` plus a correlation threshold) are **not** — `rtol=1e0` is no relative gate at all.

### CLI contract

The driver calls this directly, not through your Makefile. Keep the interface exactly as
specified; the objective check depends on it.

```
python3 opcheck.py --list
    JSON array to stdout: [{"operator": "...", "shape_key": "..."}, ...]
    Every (operator, shape) this sub-phase claims. No NPU required.

python3 opcheck.py --operator <op> [--shape-key <k>]
    Run the check on hardware. Exit 0 if and only if it passed.

python3 opcheck.py --operator <op> [--shape-key <k>] --fault-inject input
    The negative control. Perturbs one element of one input AFTER the reference is
    computed, so the device result must now disagree with it. MUST exit non-zero.
```

### The results artifact

Each run writes `programming_examples/transformer_layer/results/<operator>__<shape_key>.json`:

| Field | Meaning |
|---|---|
| `operator`, `shape_key`, `shape` | identity; `shape` is a dict of the named dimensions |
| `rtol`, `atol` | tolerances actually used |
| `ref_dtype` | dtype the reference was computed in — must be `"float32"` |
| `mean_rel_L1`, `rel_err_max`, `abs_err_max` | as `report_precision` prints them |
| `n_elements`, `n_mismatch` | full output size and the `np.isclose` failure count |
| `passed` | the verdict |
| `fault_injected` | `null` on a normal run, the injection mode otherwise |

### What the driver checks, and why it is shaped this way

You author both the builder and its reference, so the driver reading back your own verdict would
prove nothing. It instead:

1. Requires every results file to be **newer than the gate's start stamp**. A stale file from an
   earlier run does not count.
2. Re-derives the verdict rather than trusting `passed`: `n_mismatch == 0`, `ref_dtype ==
   "float32"`, `rtol == 1.6e-2` exactly, and `atol <= 1e-1` (the loosest value anywhere in the
   registry, FlashAttention's).
3. Runs `--fault-inject input` for every listed operator and **requires it to fail**.

Point 3 is the one that cannot be satisfied by making the test laxer. A reference compared against
itself, a tolerance wide enough to swallow anything, or an ignored `--fault-inject` flag all still
*pass* under injection — and the driver fails the sub-phase for it. This is the harness's own
lesson from Phase A, written down in [14](14-the-port-loop-harness.md): every fix there was only
trusted after confirming it failed when it should.

Implement the injection so it genuinely perturbs the device's input. Adding the perturbation to
the reference instead would satisfy the letter of the check and defeat its purpose.

### Per-operator lit

One `run_npu2_<op>_peano.lit` per operator, modelled on
`weighted_rms_norm/run_makefile_peano_multi_tile.lit`: scratch dir, `make clean`,
`make -f %S/Makefile <target> PEANO_INSTALL_DIR=%PEANO_INSTALL_DIR | FileCheck %s`, gating on
`CHECK: PASS!`. The suite matches by path
(`programming_examples/CMakeLists.txt:170`, `--filter "transformer_layer/"`), so no CMake change
is needed.

lit scans the **whole file** for its directives, so naming one in prose — even in a comment —
registers as a second directive and leaves the test UNRESOLVED. This has already bitten once.

## The operators

### `causal_mask` — a keyword argument, not an operator

iron's `AIECausalMask` subclasses `AIEElementwiseAdd` and adds a precomputed triangular tensor as
a static second input (`iron/operators/causal_mask/op.py`). There is no device design at all.

Make it a `causal_mask=` keyword on the elementwise-add builder path, taking a torch-precomputed
mask registered as a static buffer. Convention rule 8. Reuse `_build_add_2d_to_2d`
(`llms/shared/builders/o_ffn_multi.py:66`) — it keeps the output 2-D so downstream launches read
it without an `expand_shape`.

iron's mask fills with `-10000.0` rather than `-inf`; keep that, and say why in the docstring
(`-inf` in bf16 propagates NaN through the subsequent add).

### `addnorm` — weighted LayerNorm plus residual

**Weights are runtime memref arguments.** iron bakes them into the MLIR through `np.load()` at
generation time and hashes them into the artifact name, so every weight change forces a recompile.
Do not reproduce that.

Three traps, all documented in the kernel sources Phase A landed:

- `add_layer_norm_rows` (`programming_examples/layer_norm/layer_norm.cc:182`) is **unweighted**.
  The weighted fused forms you need are `fused_add_layer_norm_1outs`,
  `fused_add_layer_norm_2outs` and `ln_mul_weights_1outs` in `transformer_layer/kernels/`
  (symbol lists at `compile_kernels.py:199-278`).
- That kernel computes **one-pass** variance `E[x²] − E[x]²`, while `layer_norm/layer_norm.py`
  computes two-pass `mean((x−mean)²)`. Its header says outright that one must not be used as the
  numerical oracle for the other. Compute the reference in FP32 with the numerically stable
  two-pass form and size `atol` to the measured worst case, per the registry's methodology.
- `encoder.o` and `addnorm_ffn.o` **cannot share an ELF** — both define `ffn_gelu_bf16` and
  `ffn_eltwise_add_bf16_vector`. One kernel object per ELF, or rename a symbol set.

Also: `cols` must be a multiple of 16 or the remainder is silently dropped, and a padded tile must
pass the padded stride, which then normalizes the padding too.

### `layer_norm` and `elementwise_add`

Multi-row LayerNorm over `layer_norm_rows`, and the 2-D elementwise add the residual and
`causal_mask` both need. Both have existing single-purpose examples
(`programming_examples/layer_norm/`, `programming_examples/eltwise_add/`); the new work is the
multi-row and 2-D forms the block needs, in `transformer_layer/builders/`.

`eltwise_add` already gates at registry tolerances (`rtol=1.6e-2, atol=5e-2`) — match it.

## Work items

1. `opcheck.py` implementing the CLI contract, the results artifact, and fault injection.
2. FP32 references as module-level functions beside their builders, per convention rule 4.
3. `causal_mask` as a builder keyword argument over the 2-D elementwise-add path.
4. `build_addnorm_module(...)` with runtime weight arguments.
5. Multi-row `layer_norm` and 2-D `elementwise_add` builders under `transformer_layer/builders/`.
6. One `run_npu2_<op>_peano.lit` per operator, plus the `Makefile` targets they call.
7. Registry rows for every validated `(kernel, shape)` in `kernel_registry/supported_kernels.md`
   and `details/<Kernel>_bf16.md`, carrying `mean_rel_L1`, `Used by` and status. Create the detail
   page if the kernel has none, following the section order of `RMSNorm_bf16.md`.
8. `black` over all new Python; module docstrings stating the contract and its footguns.

## Gate

```
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  ninja -C build-xrt check-programming-examples-transformer-layer
```

Every test in the suite passes, including the pre-existing compile, seam and runlist-gate tests.

The driver additionally runs the objective check described above — freshness, re-derived verdict,
and the fault-injection negative control — which you cannot influence by changing a test.

## Constraints

- **Do not modify `llms/shared/`.** New builders live in `transformer_layer/builders/` and call
  into it. Modifying it triggers the ten-model `make verify` regression rule
  ([13](13-verification-and-acceptance.md)), which is what made Phase B six hours long.
- Wrap every NPU command in `flock -x -w 1800 /tmp/mlir-air-npu.lock`. Never take
  `/tmp/npu.lock`.
- Modules under ~800 lines; plain `build_*_module()` functions; no `AIE`-prefixed classes.
