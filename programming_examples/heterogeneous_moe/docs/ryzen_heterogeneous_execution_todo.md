# Ryzen Heterogeneous Execution To-Do Roadmap

This is the close-out roadmap for `programming_examples/heterogeneous_moe`.
The current MoE harness is a useful MLIR-AIR placement and plumbing prototype,
but it does not prove efficient heterogeneous execution on Ryzen. The next
exploration should move to an MLIR-AIR-first LLM-linear benchmark centered on
GEMM, GEMV, direct GPU/NPU handoff, and low-bit inference patterns.

## Current Verdict

The MoE harness should be archived as a reference harness after this roadmap is
written and any final results are preserved. It has served its purpose:

| Area | Current state | Verdict |
| --- | --- | --- |
| AIR placement | Router, experts, and aggregation can be independently assigned to CPU, iGPU, or NPU paths. | Useful reference. |
| Correctness | `top1` and `top2` routing have deterministic CPU validation and structured reports. | Useful reference. |
| Runtime plumbing | CPU, GPU, NPU, and mixed placements share one harness, manifest, trace, and result flow. | Useful reference. |
| Transfer semantics | Transfers are NumPy host-array copies or aliases. Direct iGPU-to-NPU peer transfer is not implemented. | Not proof of efficient heterogeneity. |
| Routing and packing | Top-k, route packing, and expert-output packing are CPU-side. | Not representative of fused accelerator execution. |
| Quantization | Quantized model presets record metadata, but execution is bf16 after dequantized weight loading. | No int4 execution proof. |
| Workload shape | Defaults and sweeps are tiny MoE-shaped kernels. Model presets still chunk into small fixed kernels. | Not enough to show an LLM crossover. |
| Baselines | CPU/GPU/NPU and mixed cases can be measured, but there is no direct-handoff mixed path or final crossover study. | Incomplete. |

Bluntly: this harness proves that the repository can compile, place, validate,
and time small AIR kernels across the Ryzen CPU/iGPU/NPU stack. It does not prove
that splitting a real inference workload across Ryzen devices is faster than a
single device, or even faster than a host-staged mixed path.

## Direction

Do not expand the MoE harness into a larger research project. The primary future
workload should be LLM-style linear layers:

- Batched bf16 GEMM for prefill-like work.
- bf16 GEMV for decode-like work.
- Fused int4-weight dequantization plus bf16 accumulation for decode and
  bandwidth-limited linear layers.
- Explicit GPU/NPU handoff in both directions.

This is intentionally not a GNN roadmap and not another MoE roadmap. GNNs add
irregular memory and graph scheduling questions before the basic Ryzen handoff
and linear-layer crossover questions are settled. More MoE work keeps exercising
routing and packing machinery instead of the linear kernels that dominate common
LLM inference paths.

External stacks are context, not integration targets. AMD's Ryzen AI LLM docs
describe NPU-only and hybrid NPU+iGPU LLM execution modes, with hybrid execution
targeting prefill and decode performance through OGA. ONNX Runtime documents
weight-only int4/uint4 MatMul quantization patterns such as `MatMulNBits`.
`llama.cpp` is a useful reference point for local LLM inference and low-bit
quantized model practice. This roadmap does not plan a llama.cpp or ONNX Runtime
integration.

References:

- [AMD Ryzen AI LLM deployment overview](https://ryzenai.docs.amd.com/en/latest/llm/overview.html)
- [AMD Ryzen AI OGA flow](https://ryzenai.docs.amd.com/en/latest/hybrid_oga.html)
- [ONNX Runtime quantization docs](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)

## Target Hypothesis

The hypothesis to test is:

> iGPU prefill with batched bf16 GEMM plus NPU decode with fused int4-weight
> dequantization and bf16 GEMV can beat CPU-only, iGPU-only, NPU-only, and
> host-staged mixed execution for at least one practical Ryzen LLM-linear shape
> range.

This split is plausible, not proven. A required falsification control is the
opposite split:

> NPU prefill plus iGPU decode must also be measured.

The final answer must come from timing, correctness, and transfer evidence, not
from intuition about which device "should" own each phase.

## Shape Ladder

The new benchmark should use a shape ladder that separates CI, bring-up, and
LLM-like evidence.

| Tier | Purpose | Example shapes |
| --- | --- | --- |
| Tiny CI | Deterministic correctness, serialization, result schema, and CPU-safe tests. | `M=1..4`, `K=64..256`, `N=64..256`; GEMV and small GEMM. |
| Medium bring-up | GPU/NPU compile and runtime debugging without full model-scale cost. | `M=1,8,32`, `K=512..2048`, `N=512..4096`. |
| LLM-like | Crossover evidence for hidden and intermediate dimensions. | Hidden sizes around `2048`, `3072`, `4096`, `5120`, `8192`; MLP expansion around `2.5x..4x`; decode `M=1`; prefill `M=32..2048`. |

The exact initial values can be tuned to local compiler and memory limits, but
the ladder must preserve the distinction between "small enough for CI" and
"large enough to say something about LLM linear layers."

## Required Baselines

Every reported crossover claim must include these baselines:

| Baseline | Why it is required |
| --- | --- |
| CPU-only | Correctness reference and fallback performance floor. |
| iGPU-only | The mixed path must beat a simple GPU deployment for relevant shapes. |
| NPU-only | The mixed path must beat NPU-only for relevant shapes. |
| Host-staged mixed | Keeps the current NumPy-style transfer cost visible as a baseline. |
| Direct-handoff mixed | The only path that can count as efficient heterogeneous Ryzen execution. |

Host-staged mixed execution is a baseline, not the finish line.

## Handoff Gap

The direct handoff gap is the central blocker.

The current Python/XRT path is host-oriented. In `python/air/backend/xrt.py`,
runtime invocation allocates `pyxrt` BOs, writes NumPy arrays, calls
`bo.sync(XCL_BO_SYNC_BO_TO_DEVICE)`, runs the kernel, calls
`bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE)`, and reads NumPy arrays back. Local
`pyxrt` introspection exposes `bo.map` and `bo.sync`, but not Python-visible BO
import/export methods. That is enough for host-staged execution; it is not enough
for a direct iGPU/NPU tensor handoff.

The lower-level runtime has pieces that may be usable:

- Local XRT C/C++ headers expose imported BO constructors, `export_buffer()`,
  `xrtBOExport()`, and `xrtBOImport()`.
- The GPU runtime under `runtime_lib/airgpu` already has HIP VMem
  POSIX-file-descriptor export/import machinery for GPU peer mappings.

Those facts are opportunities, not proof that iGPU/NPU direct sharing already
works. Milestone 2 must build and validate a C++/runtime bridge or equivalent
device-resident tensor abstraction before any "efficient heterogeneous" claim.

That future abstraction must define:

- Ownership: which runtime owns the allocation and exported handle.
- Lifetime: how long an imported handle remains valid and who closes it.
- Coherency: when producers make writes visible to consumers.
- Fence/sync: explicit ordering between GPU kernels, NPU kernels, and host code.
- Addressing and shape metadata: dtype, shape, stride/layout, byte size, and
  offset.
- Trace labels: device events proving a direct path did not materialize a NumPy
  host array.

Direct-handoff result files must make the proof machine-checkable. At minimum,
they should report `transfer_summary.model = device_resident_direct_handoff`,
`device_resident_buffers = true`, per-edge handoff mechanisms, synchronization
events, and a count of NumPy host arrays used on the direct path. That count must
be zero for the GPU/NPU edge being claimed as direct.

## Ordered Milestones

### Milestone 1: bf16 Host-Staged LLM-Linear Benchmark

Status as of May 2, 2026: implemented as
`programming_examples/heterogeneous_moe/llm_linear` with top-level
`compile_llm_linear.py` and `run_llm_linear_suite.py` CLIs.

Implemented scope:

- Portable `llm_linear/default_linear_manifest.json` and
  `llm_linear/default_linear_matrix.json`.
- Shape suites for `tiny_ci`, `medium`, and `llm_like` tiers.
- Two-stage benchmark: prefill GEMM `X[M,K] @ Wp[K,H] -> P[M,H]`, then decode
  GEMV `P[M-1,:] @ Wd[H,N] -> Y[N]`.
- CPU NumPy reference validation for prefill, decode input, and final output.
- AIR text generation for prefill and decode kernels plus GPU/NPU compile
  helper wiring through the existing MLIR-AIR toolchain helpers.
- Result JSON/CSV/report fields for placement, shape tier, dtype, bytes moved,
  compile/load exclusion, cold/warm timing blocks, validation status, transfer
  semantics, and device-residency truth flags.
- Matrix cases for `cpu_only`, `gpu_only`, `npu_only`,
  `gpu_prefill_npu_decode_host`, and `npu_prefill_gpu_decode_host`.
- `transfer_mode=direct` is fail-closed and explicitly unsupported. Milestone 1
  does not claim direct handoff.

Verification run:

```bash
cd /home/cj/mlir-air/programming_examples/heterogeneous_moe
../../sandbox/bin/python -m pytest tests/test_llm_linear_*
../../sandbox/bin/python -m pytest tests
../../sandbox/bin/python run_llm_linear_suite.py --suite tiny_ci --case-filter cpu_only --iterations 1 --warmup 0 --require-correctness
```

Observed result: focused LLM-linear tests passed, the full harness test suite
passed with 62 tests, and the CPU-only tiny CI smoke wrote outputs under the
ignored `llm_linear/artifacts/benchmarks/latest` tree.

Hardware limitations not resolved in this milestone:

- GPU and NPU compile/run paths are wired but were not executed in the CPU-safe
  verification above.
- Mixed GPU/NPU cases remain NumPy host-staged.
- Direct GPU/NPU device-resident handoff remains the Milestone 2 blocker.
- The generated AIR kernels are explicit bring-up sources, not tuned LLM-scale
  tilings.

Goal: replace the MoE-shaped question with the right linear-layer question while
staying on the existing safe host-staged runtime model.

Checklist:

- Add an MLIR-AIR-first benchmark for bf16 GEMM and GEMV shapes.
- Keep CPU reference correctness for every case.
- Generate or compile GPU and NPU paths through the MLIR-AIR stack.
- Report prefill-like GEMM and decode-like GEMV separately.
- Include CPU-only, iGPU-only, NPU-only, and host-staged mixed placements.
- Report shape, dtype, bytes moved, compile/load exclusion policy, warm/cold
  timing, and validation status.
- Include both proposed split directions: iGPU-prefill/NPU-decode and
  NPU-prefill/iGPU-decode.

Acceptance:

- Tiny CI shapes pass without hardware-specific requirements.
- Hardware runs produce correctness and timing reports for medium shapes.
- The benchmark clearly labels host-staged transfer semantics and makes no
  direct-handoff claim.

### Milestone 2: Direct Bidirectional GPU/NPU Handoff

Status as of May 3, 2026: implemented through the AIR generator, runtime, and
native bridge boundary, but not accepted as a working direct handoff path on the
current machine because the low-level XRT/HIP import probe fails closed.

Implemented scope:

- `DeviceResidentTensor` records owner, backend, dtype, shape, stride, byte
  size, exported handle metadata, synchronization state, and trace identity.
- `transfer_mode=direct` now means "prove a GPU/NPU direct edge or fail"; it no
  longer silently falls back to host staging for direct mixed cases.
- The benchmark matrix includes both direct split directions:
  `gpu_prefill_npu_decode_direct` and `npu_prefill_gpu_decode_direct`.
- Result artifacts can report `device_resident_direct_handoff`,
  per-edge mechanisms, sync events, and NumPy host materialization counts when a
  native bridge records such an edge.
- GPU artifact compilation has a device-resident option that omits
  `air-gpu-host-staging` for future direct executor work.
- LLM-linear AIR generation now stages L3 operands through DMA-visible L2/L1
  buffers before herd access, removing the `air-to-rocdl`/GPU outlining blocker
  caused by direct L3 `memref.load` operations inside herds.
- `llm_linear/native/direct_bridge.cpp` provides a C ABI for XRT-owned BO
  allocation/export, HIP VMem fd import, no-host-staging GPU shared-library
  invocation, XRT kernel launch, synchronization reporting, and both
  `gpu_prefill_npu_decode` and `npu_prefill_gpu_decode` directions.
- `llm_linear/runtime.py` now calls the native bridge when
  `transfer_mode=direct` is requested and the bridge probe succeeds. It records
  the direct edge with zero NumPy host materializations and keeps host-staged
  mixed execution as the baseline.

Current hardware blocker:

- On the local Ryzen AI/XRT/HIP stack, the native probe isolates each candidate
  XRT BO flag in a child process and reports that `p2p`, `device_only`,
  `carveout`, and `normal` BO export handles all abort during HIP VMem fd
  import. Direct mode therefore remains fail-closed on this machine until the
  XRT/HIP interop layer can import one of those BO types without crashing.

Goal: make the mixed path device-resident. This milestone is required before the
roadmap can be considered finished.

Checklist:

- Keep the checked-in `DeviceResidentTensor` interface as the AIR runtime
  contract for direct mixed LLM-linear cases.
- Use the checked-in C++ bridge as the implementation point for XRT BO export,
  HIP VMem import, GPU direct shared-library calls, and XRT kernel launches.
- Keep both GPU-to-NPU and NPU-to-GPU directions wired through the bridge.
- Preserve explicit synchronization reporting for each handoff.
- Preserve trace and result fields that prove no NumPy host array is in the
  direct handoff path.
- Keep the host-staged mixed path as a baseline and regression control.

Acceptance:

- A GPU-produced tensor can be consumed by an NPU kernel without materializing a
  NumPy host array on the claimed edge.
- An NPU-produced tensor can be consumed by a GPU kernel without materializing a
  NumPy host array on the claimed edge.
- Direct handoff and host-staged handoff both run the same correctness tests.
- Result artifacts distinguish direct handoff from host staging in a way that
  can be audited after the run.

### Milestone 3: Fused int4 Weight Dequantization

Status as of May 3, 2026: implemented for decode GEMV in the `llm_linear`
CPU-safe path.

Implemented scope:

- Decode weights can be generated as signed int4 or uint4 packed storage with
  block size, quant axis, scale layout, optional zero points, and
  low-nibble-first packing metadata.
- CPU decode supports fused unpack/dequantize plus linear GEMV and validates
  against a dequantize-then-linear baseline.
- Result JSON/CSV/report fields include decode weight storage, dequant time,
  linear time, packed bytes read, scale bytes read, and zero-point bytes read.

Goal: test the low-bit decode pattern that makes NPU GEMV potentially
interesting for LLM inference.

Checklist:

- Add int4 or uint4 packed-weight metadata as future benchmark input schema:
  block size, quant axis, signedness, scales, zero points when applicable, and
  packing layout.
- Implement fused weight dequantization inside GEMV first, with bf16 compute and
  CPU reference validation.
- Extend to GEMM only after GEMV correctness and layout are stable.
- Compare fused dequant+linear against dequantize-then-linear baselines.
- Keep representative patterns aligned with weight-only MatMul quantization
  practice, without importing ONNX Runtime or llama.cpp as dependencies.

Acceptance:

- int4 dequant+GEMV matches the CPU reference within dtype-aware tolerances.
- Reports separate dequant overhead, linear compute time, and bytes read when
  that detail is available.
- No current MoE result is reinterpreted as int4 evidence.

### Milestone 4: Final Crossover and Speedup Study

Status as of May 3, 2026: report plumbing is implemented, but the final hardware
study has not been run because Milestone 2 is not accepted.

Implemented scope:

- Suite reports now include a crossover section when audited direct mixed cases
  are present.
- Speedup rows compare each direct mixed split against CPU-only, GPU-only,
  NPU-only, and the matching host-staged mixed split.
- Each row is classified as `wins`, `loses`, or `inconclusive`.

Goal: answer whether heterogeneous Ryzen execution wins, and where.

Checklist:

- Run the full shape ladder across CPU-only, iGPU-only, NPU-only, host-staged
  mixed, and direct-handoff mixed baselines.
- Include both iGPU-prefill/NPU-decode and NPU-prefill/iGPU-decode.
- Report correctness, latency, throughput, transfer/sync cost, and speedup
  tables.
- Mark shape ranges where mixed execution wins, loses, or is inconclusive.
- Preserve raw JSON/CSV/trace artifacts and a concise markdown summary.

Acceptance:

- Every speedup claim names its baseline and shape range.
- The final report shows when direct-handoff mixed execution beats GPU-only,
  NPU-only, and host-staged mixed execution.
- If no crossover exists, the conclusion says so and records the bottleneck.

## Future Interfaces

These are current or proposed interfaces:

- `DeviceResidentTensor`: current checked-in Python contract for allocation
  owner, backend, dtype, shape, strides, byte size, exported handle metadata,
  synchronization state, and trace identity.
- `transfer_mode = direct`: request device-resident handoff and fail if an edge
  would fall back to a host array.
- Quantized weight metadata: current decode-weight schema for block size,
  signedness, quant axis, scale layout, zero-point layout, and packed-weight
  byte order.
- Result fields for direct handoff: current audit schema for edge mechanism,
  exported/imported handle type, sync primitive, NumPy host materialization
  count, and device-residency truth flags.

Do not retrofit these names into the MoE harness unless doing so directly helps
the LLM-linear benchmark. The MoE harness should remain a compact reference.

## Verification Requirements

For the current checked-in implementation:

- The README link to this file must resolve.
- This file must not claim that current direct GPU/NPU handoff is accepted.
- Focused CPU-safe checks must pass:
  `../../sandbox/bin/python -m pytest tests/test_llm_linear_*`.
- Full harness checks should pass when time allows:
  `../../sandbox/bin/python -m pytest tests`.
- A CPU-only tiny CI smoke should still write ignored outputs:
  `../../sandbox/bin/python run_llm_linear_suite.py --suite tiny_ci --case-filter cpu_only --iterations 1 --warmup 0 --require-correctness`.

For direct-handoff acceptance:

- The native bridge must build:
  `llm_linear/native/build_direct_bridge.sh /tmp/libllm_linear_direct_bridge.so`.
- `LLM_LINEAR_DIRECT_BRIDGE_SO=/tmp/libllm_linear_direct_bridge.so` must report a
  successful `probe_direct_bridge()` on the target machine before direct cases
  are run.
- Correctness must pass against a CPU reference.
- Host-staged and direct-handoff paths must be timed separately.
- GPU-to-NPU and NPU-to-GPU handoff proof must be present in result artifacts.
- int4 dequant+linear correctness must pass before performance claims.
- Crossover and speedup tables must include CPU-only, iGPU-only, NPU-only,
  host-staged mixed, and direct-handoff mixed baselines.

## Close-Out Rule

After this roadmap is linked from the README, the MoE harness should be treated
as archived unless a change is needed to preserve it as a reference, fix a bug in
existing validation, or compare old host-staged behavior against the new
LLM-linear benchmark. Future efficient Ryzen heterogeneous work should move to
the LLM-linear roadmap.
