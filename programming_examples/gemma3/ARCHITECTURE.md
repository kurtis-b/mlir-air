# Gemma3 Implementation Architecture

This document is the implementation guide for organizing Gemma3 work in
MLIR-AIR. It replaces the earlier chronological model-loop roadmap with a
backend-oriented architecture:

- shared model, artifact, reference, result, and measurement contracts,
- CPU implementation,
- GPU/iGPU implementation,
- NPU implementation modeled after the Llama 3.2 1B example.

It is scoped to source-level work in this repository. It is not a claim that
the repository reproduces FastFlowLM, the Gemma3 paper, or an end-to-end
production Gemma3 runtime.

Primary local sources:

- Canonical Gemma3 example: `programming_examples/gemma3`.
- Kernel wrappers: `programming_examples/gemma3/gemma3/kernels`; Peano sources: `programming_examples/gemma3/aie_kernels`.
- NPU model-runtime reference: `programming_examples/llama32_1b`.
- Paper target ledger: `programming_examples/gemma3/data/paper_targets.json`.

Primary paper source:

- "Mapping Gemma3 onto an Edge Dataflow Architecture", arXiv:2602.06063v2,
  revised 2026-02-24.

## Status And Non-Claims

The current tree has useful pieces, but they are not yet one clean backend
architecture.

Implemented today:

- CPU references for synthetic Gemma3 model-loop bring-up.
- HF/Torch real-artifact CPU and ROCm/iGPU benchmark paths.
- Paper target comparison and result JSON machinery.
- Kernel-level NPU mappings for Q4NX, BF16 tiled MM, FusedDQP, FlowQKV,
  FlowKV, RoPE, residual add, RMSNorm wrappers, Q/K normalization bridges, and
  GeGLU/down diagnostic slices.
- A stitched-ELF decode track for Gemma3 1B that follows the Llama-style
  multi-launch pattern for several decode subgraphs.
- Real-shape planning for NPU preflight, weights, BOs, buffer bindings, kernel
  argument bindings, and model-runner launch order.

Do not report the current Gemma3 code as:

- an end-to-end Gemma3 deployment,
- a FastFlowLM reproduction,
- a paper-performance reproduction,
- a complete text-plus-vision VLM,
- or a validated model-accuracy result.

The near-term goal is to make each backend explicit and make the NPU path look
like the Llama 3.2 1B example: cached kernels, preloaded static weights,
per-layer BO ownership, explicit prefill/decode entrypoints, and clear host
fallback accounting.

## Architecture Overview

The Gemma3 code should be organized around a shared core plus backend-specific
execution layers.

```text
                    shared Gemma3 core
     config | artifacts | weights | references | KV/cache contracts
                |              |              |
                v              v              v
          CPU backend     GPU/iGPU backend    NPU backend
          HF/Torch        HF/Torch ROCm       MLIR-AIR/XRT
          references      paper baseline      stitched ELFs
```

The shared core owns model semantics and evidence contracts. Backends own device
execution.

Backend names in code and result records:

| Name | Meaning |
| --- | --- |
| `cpu` | CPU references and HF/Torch CPU paper baselines. |
| `igpu` | ROCm/HF/Torch integrated-GPU paper baseline. |
| `npu` | MLIR-AIR/XRT/NPU2 execution. |

Use "GPU" only as prose for the backend category. Use `igpu` in flags, result
JSON, and comparisons where the current implementation targets ROCm on the
integrated GPU.

## Shared Core

Shared code must be importable without AIR, XRT, pyxrt, ROCm, or hardware
availability. Device-specific packages should be imported lazily inside backend
entrypoints.

Shared responsibilities:

| Responsibility | Current files | Rule |
| --- | --- | --- |
| Model/layer metadata | `gemma3.core.config`, `gemma3.core.artifacts` | Defines model variants, local/global attention metadata, dimensions, and artifact discovery. |
| Synthetic weights | `gemma3.core.weights` | Provides deterministic smoke inputs and Q4NX-compatible metadata. |
| CPU math references | `gemma3.core.reference` | Defines correctness semantics for projections, attention, nonlinear ops, residuals, logits, and KV updates. |
| Runtime manifests | `gemma3.core.runtime` | Records artifact names, static inputs, intermediate outputs, and fallback metadata without importing AIR/XRT. |
| Result contracts | `gemma3.evidence.results`, `gemma3.evidence.paper_compare`, `paper_targets.json` | Produces machine-readable local results and paper deltas. |
| Environment/power metadata | `gemma3.evidence.environment`, `gemma3.evidence.power` | Captures run conditions and timed-window power metadata. |
| Blocker ledger | `gemma3.evidence.reproduction_blockers` | Reports why a backend/model cell is not paper-comparable. |

Shared core invariants:

- A result must name model variant, backend, metric, prompt length, decode token
  count, timed-window policy, and host fallbacks.
- Shape contracts are named before tensors cross a backend boundary.
- CPU reference functions are the correctness source until model accuracy is
  validated against a real-tokenizer path.
- Synthetic checksums are not model accuracy.
- Compile-only success is not runtime support.

### Model Contracts

Gemma3 text execution has two phases:

- Prefill consumes prompt chunks and appends K/V rows to the per-layer cache.
- Decode consumes one token at a time and attends over the current K/V cache.

Layer metadata must represent attention behavior explicitly:

```text
layer_index
attention_kind = local_swa | global_full | vision_nca
window_len = 0 for global_full, positive for local_swa
causal = true for text, false for vision_nca
```

Do not encode local/global behavior as scattered command-line conditionals.
Backends should read a layer metadata table.

### KV-Cache Contract

Track K and V separately:

```text
kv_cache[layer][kv_group][token_range][head_dim]
```

Minimum metadata:

- `layer_index`
- `kv_group`
- `head_dim`
- `cache_len`
- `window_len`
- `token_base`
- `tokens_valid`
- `layout_version`

For prefill, append K/V for the whole prompt chunk. For decode, append one K/V
entry per layer for the current token. Local SWA layers should read only the
legal sliding window when the backend contract permits it. Global layers should
read the full current context.

### Weight Contract

Gemma3 NPU projection work uses Q4NX-style packed weights where the paper
kernel contract requires it:

```text
Q4NX_ROWS = 32
Q4NX_COLS = 256
w_bf16[row, col] = scale[col] * q4[row, col] + min[col]
```

Shared weight planning should define:

- logical tensor name,
- dtype and quantization format,
- padded shape,
- packed byte range,
- layer index,
- projection family,
- backend-ready layout.

Backends may materialize different device layouts, but the mapping from real
safetensor name to logical Gemma3 tensor must remain shared.

## CPU Backend

The CPU backend is the correctness and paper-baseline anchor.

Current implementation:

- `gemma3.core.reference` implements synthetic CPU references.
- `gemma3.model.model_loop`, `gemma3.model.prefill`, and `gemma3.model.decode` run the
  synthetic model loop and record per-stage metadata.
- `gemma3.evidence.real_execution` runs real Gemma3 artifacts through HF/Torch on CPU.
- `gemma3.evidence.results` and `gemma3.evidence.paper_compare` turn those runs into local
  paper-cell records.

Target CPU behavior:

```text
load real artifacts
tokenize prompt deterministically
run HF/Torch prefill or decode benchmark on CPU
capture timed window
capture CPU/package power when requested
write result JSON
compare against paper target
```

CPU backend rules:

- It may use HF/Torch for real-model baselines.
- It must use the same prompt lengths, decode-token counts, and timing-window
  policy as GPU/iGPU and NPU paper cells.
- It should remain runnable without AIR imports.
- It should expose any deviation from paper hardware, precision, tokenizer, or
  measurement policy as an explained deviation, not a hidden normalization.

## GPU/iGPU Backend

The GPU/iGPU backend is the ROCm paper-baseline path.

Current implementation:

- `gemma3.evidence.real_execution` selects ROCm through Torch CUDA APIs when
  `backend=igpu`.
- `gemma3.evidence.power` records ROCm SMI graphics-package power when available.
- Initial 1B 1k iGPU cells exist in the result bundle and comparison scripts.

Target iGPU behavior:

```text
load real artifacts
move model and inputs to ROCm device
synchronize before and after timed windows
run prefill or decode benchmark
capture GPU rail power when requested
write result JSON
compare against paper target
```

iGPU backend rules:

- The public backend identifier is `igpu`.
- It should reuse the CPU backend's artifact loading and prompt construction.
- It must synchronize the device around timed regions.
- It must not be used to infer NPU speedups unless a matching local NPU result
  and power record exist.

## NPU Backend

The NPU backend is the MLIR-AIR/XRT implementation track. It should converge on
the Llama 3.2 1B organization rather than remain a set of independent probes.

Target device:

- AMD Ryzen AI NPU2 / XDNA2 / AIE2P.
- Physical array: 8 columns by 4 rows, 32 compute tiles.
- Memory hierarchy: L3 host/XRT BOs, L2 memory tiles, L1 compute-tile memory,
  shim DMA endpoints.

Production routes should prefer explicit L3-to-L2-to-L1 staging for
full-physical paths when direct shim DMA would over-allocate resources.

### Llama 3.2 1B Pattern To Copy

Llama 3.2 1B is the control-plane and runtime reference, not the source for
Gemma-specific math.

| Llama component | Gemma3 target |
| --- | --- |
| `llama32_1b_inference.py` | Unified Gemma session with compile-only, run-only, verify, profile, and paper-benchmark modes. |
| `llama32_1b_prefill.py` | Per-layer Gemma prefill orchestration using Q4NX/BF16 MM/FlowQKV and validated nonlinear kernels. |
| `llama32_1b_decode.py` | Per-token Gemma decode using FusedDQP/FlowKV, KV update, and explicit host fallback handling. |
| `llama32_1b_weights.py` | Gemma real/synthetic weight containers and backend layout preparation. |
| `llama32_1b_reference.py` | CPU reference semantics, intermediate names, tolerances, and shape validation. |
| `kernel_builder/cache.py` | Gemma kernel cache, artifact manifest, lazy XRT load, per-layer BOs, static input skipping, intermediate output skipping. |
| `multi_launch_builder/` | Gemma stitched prefill/decode subgraphs with explicit public BO arguments. |

Llama patterns to preserve:

- Compile or load artifacts once.
- Preload static weights before timed inference.
- Use per-layer BO keys so weights and mutable intermediates do not alias
  accidentally across layers.
- Mark static input indices and intermediate output indices explicitly.
- Fuse multiple `air.launch` regions into one ELF only after each substep has
  isolated CPU-reference and hardware evidence.
- Keep host fallbacks visible in stage metadata and result JSON.

Gemma-specific differences:

- Projection weights are Q4NX/int4 where the paper kernels require them.
- Prefill projections use Q4NX dequantization plus BF16 tiled MM.
- Decode projections use FusedDQP where supported.
- Attention uses FlowQKV for prefill and FlowKV or tiled-stat attention for
  decode.
- QK-Norm is required and must not be assumed equivalent to Llama.
- Text layers mix local sliding-window and global attention.
- Vision is a later non-causal prefill-like path.

### Current NPU Assets

Kernel-level assets:

| Role | Current source | Status |
| --- | --- | --- |
| Q4NX dequantization | `programming_examples/gemma3/gemma3/kernels/q4nx.py`, `programming_examples/gemma3/aie_kernels/q4nx_opt.cc` | Standalone validated/sweepable route. |
| BF16 tiled MM | `programming_examples/gemma3/gemma3/kernels/bf16_tiled_mm.py` | Uses the optimized AIE2P BF16 GEMM path. |
| FusedDQP | `programming_examples/gemma3/gemma3/kernels/fused_dqp.py`, `programming_examples/gemma3/aie_kernels/fused_dqp_opt.cc` | Decode projection candidate; paper route passes standalone, pipeline route is diagnostic-only. |
| FlowQKV | `programming_examples/gemma3/gemma3/kernels/flowqkv.py`, `programming_examples/gemma3/aie_kernels/flow_attention_opt.cc` | Prefill attention candidate. |
| FlowKV | `programming_examples/gemma3/gemma3/kernels/flowkv.py`, `programming_examples/gemma3/aie_kernels/flow_attention_opt.cc` | Decode attention candidate and `Q_CHUNK=1` specialization. |
| RoPE | `programming_examples/gemma3/gemma3/kernels/rope_halfsplit.py`, `programming_examples/llama32_1b/kernel_builder/rope_halfsplit.cc` | Half-split RoPE wrapper. |
| Residual add | `programming_examples/gemma3/gemma3/kernels/residual_add.py` | BF16 vector-add wrapper. |
| GeGLU/down | `programming_examples/gemma3/gemma3/kernels/geglu.py`, `gemma3.probes.stitched_decode` | Decode FFN diagnostic path. |

Model-level NPU planning assets:

| Responsibility | Current source |
| --- | --- |
| Real-shape preflight | `gemma3.npu.preflight` |
| Per-layer wiring manifest | `gemma3.npu.wiring` |
| Static projection weight plan | `gemma3.npu.weight_plan` |
| Norm weight plan | `gemma3.npu.norm_weight_plan` |
| BO allocation plan | `gemma3.npu.bo_plan` |
| Static preload evidence | `gemma3.npu.static_preload`, `gemma3.npu.norm_preload` |
| Buffer binding plan | `gemma3.npu.buffer_binding` |
| Kernel argument binding plan | `gemma3.npu.argument_binding` |
| Launch-order manifest | `gemma3.npu.model_runner` |
| Runtime shell | `gemma3.npu.inference_runtime` |
| Stitched decode MLIR | `gemma3.probes.stitched_decode`, `gemma3.npu.stitching` |

Diagnostic/probe files should remain support artifacts. New paper-parity work
should promote behavior into a Llama-style NPU runtime path before using it for
headline timing.

### Target NPU Runtime

`gemma3.npu.inference_runtime` now owns the production-shaped runtime shell. It prepares real 1B/1k setup state, validates kernel argument bindings, records static-input/readback policy, and gates `generate()` on production NPU prefill K/V before any decode launch. The current runtime decode evidence is `results/gemma3_1b_npu_runtime_decode_loop.json`: it is blocked before decode with `generate-prefill-kv-cache-blocked` and `production-prefill-runtime-artifacts-not-cached`, records zero decode launches, and preserves the NPU-prefill K/V blockers. The stitched 26-layer decode loop remains diagnostic evidence until `run_npu_prefill()` produces a Gemma-owned K/V cache.

Use `docs/npu_runtime_loop.md` as the operational runbook for choosing the next
1B text NPU runtime blocker, running the exact commands, and deciding whether
production evidence is sufficient. The Makefile `model-loop` target mirrors the
Llama-style organization (`compile/cache -> prepare_runtime -> run_npu_prefill
-> generate -> profile/verify`) for Gemma3, and
`gemma3.evidence.npu_runtime_contracts` is the pass/fail boundary for accepted
Gemma3 evidence. Llama remains a control-plane reference only; it is not a
source of accepted Gemma3 measurements.

The target runtime shape is:

```text
prepare_runtime()
  - load model metadata and weights
  - compile or load cached Gemma ELFs
  - allocate per-layer BO sets
  - preload static projection and norm weights
  - validate kernel argument bindings

run_npu_prefill()
  - embed/token input from host or shared frontend
  - run every layer over prompt chunks
  - produce NPU-owned K/V cache
  - run final norm/logits path or explicitly account host fallback

generate()
  - for each decode token:
      - run every layer through cached stitched ELFs
      - append one K/V entry per layer
      - run attention over NPU-produced cache
      - run final norm/logits/sampling or explicitly account host fallback
```

Compile, ELF load, BO creation, static preload, and argument validation stay
outside timed inference windows. Timed prefill TTFT covers prefill execution.
Timed decode TPS covers post-prefill decode-token execution. Any host bridge,
reduction, logits, or sampling that remains in a result must be named and
timed/accounted or excluded from paper-parity claims.

### Prefill Flow

Each Gemma text prefill layer should eventually execute:

```text
input_state
  -> RMSNorm
  -> Q/K/V projection
     -> Q4NX dequantization where needed
     -> BF16 tiled MM
  -> QK-Norm
  -> RoPE
  -> append K/V to cache
  -> FlowQKV attention
     -> causal full attention for global layers
     -> causal SWA for local layers
  -> output projection
  -> residual update
  -> FFN gate/up/down path
  -> output_state
```

The first production NPU prefill milestone may keep some operations on host, but
the metadata must classify every stage as `npu`, `host-fallback`, or `missing`.
The current blocker is that no official 1k NPU prefill paper cell is wired
through this full path.

### Decode Flow

Each Gemma text decode layer should eventually execute:

```text
single_token_state
  -> RMSNorm
  -> FusedDQP Q/K/V projection
  -> QK-Norm
  -> RoPE
  -> append one K/V entry to cache
  -> FlowKV or tiled-stat attention over current cache
  -> FusedDQP output projection
  -> residual update
  -> FusedDQP gate/up projections
  -> GeGLU
  -> FusedDQP down projection
  -> residual update
  -> output_state
```

Current stitched decode slices already cover much of this route for 1B:

- decode ingress: `RMSNorm -> Q/K/V -> QK-Norm -> RoPE`,
- attention plus O projection,
- post-attention RMSNorm plus residual,
- pre-FF RMSNorm plus gate/up projection,
- GeGLU plus down projection,
- post-feedforward residual diagnostic integration.

The clean 26-layer diagnostic is correctness-clean but still not a paper cell:
prefill K/V is not NPU-produced, tiled-stat reduction can be host-side, logits
and sampling are not NPU-promoted, and production contiguous static-weight BOs
are not yet used by the FusedDQP route.

### NPU Output Modes

Public output modes:

- `auto`
- `direct`
- `l2-gather`

`packet-direct` is diagnostic-only and must not be exposed as a public model
loop mode.

Supported public mode matrix:

| Kernel | 2x4 modes | 4x4 modes | 8x4 modes |
| --- | --- | --- | --- |
| Q4NX | direct, l2-gather | direct, l2-gather | l2-gather |
| FusedDQP | direct, l2-gather | direct, l2-gather | l2-gather |
| FlowQKV | direct, l2-gather | direct, l2-gather | l2-gather |
| FlowKV | direct | direct | l2-gather |

Unsupported modes must fail early with a classified reason:

- hardware resource limit,
- packet S2MM backend limitation,
- AIE routing bug,
- channel/runtime scheduling bug.

Never silently rewrite unsupported output modes inside the model loop.

## Paper Target Ledger

Paper targets are tracked so local CPU/iGPU/NPU results can be compared with
the same schema. PDF/HTML v2 tables are the primary numeric target. Abstract
and secondary FastFlowLM pages are secondary headline sources when they
disagree.

Treat a local result as `PAPER_MATCH` only when it is within 20 percent of the
paper value for the same model, sequence length, backend, metric, and power
mode. Treat a local result outside 20 percent as `EXPLAINED_DEVIATION` only
when the root cause is documented with a concrete artifact.

Prefill TTFT targets are seconds. Decode targets are tokens per second.

| Metric | Model | Backend | 1k | 2k | 4k | 8k | 16k | 32k | 64k | 128k |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prefill TTFT | 1B | NPU | 0.95 | 1.47 | 2.46 | 4.42 | 8.45 | 17.19 | n/a | n/a |
| Prefill TTFT | 1B | iGPU | 0.51 | 1.14 | 2.61 | 6.64 | 21.3 | 95.1 | n/a | n/a |
| Prefill TTFT | 1B | CPU | 4.06 | 8.06 | 17.2 | 36.2 | 75.5 | 165 | n/a | n/a |
| Prefill TTFT | 4B | NPU | 1.81 | 2.81 | 4.79 | 8.37 | 16.17 | 33.5 | n/a | n/a |
| Prefill TTFT | 4B | iGPU | 2.05 | 4.26 | 9.54 | 23.8 | 71.5 | 265 | n/a | n/a |
| Prefill TTFT | 4B | CPU | 20.3 | 42.4 | 85.4 | 176 | 766 | 832 | n/a | n/a |
| Decode TPS | 1B | NPU | 41.1 | 40.5 | 39.5 | 37.3 | 33.6 | 27.9 | OOC | OOC |
| Decode TPS | 1B | iGPU | 38.0 | 53.9 | 42.3 | 33.5 | 25.0 | 13.6 | n/a | n/a |
| Decode TPS | 1B | CPU | 41.9 | 40.8 | 41.7 | 40.2 | 38.1 | 33.8 | n/a | n/a |
| Decode TPS | 4B | NPU | 18.2 | 18.0 | 17.8 | 17.3 | 16.3 | 14.8 | 13.2 | 11.2 |
| Decode TPS | 4B | iGPU | 18.6 | 17.4 | 15.3 | 12.6 | 9.2 | 5.9 | 3.3 | 1.9 |
| Decode TPS | 4B | CPU | 14.6 | 13.5 | 13.9 | 13.0 | 11.4 | 10.8 | 7.5 | 4.1 |

Vision targets from the paper body:

| Metric | Model | Backend | Value |
| --- | --- | --- | ---: |
| Vision TTFT | 4B vision tower | NPU | 2.6 sec |
| Vision TTFT | 4B vision tower | iGPU | 7.45 sec |
| Vision TTFT | 4B vision tower | CPU | 38.55 sec |
| Vision speedup | 4B vision tower | NPU vs iGPU | 2.9x |
| Vision speedup | 4B vision tower | NPU vs CPU | 14.8x |

Average power targets from the paper's cross-hardware power table:

| Category | Model | CPU W | GPU W | NPU W | Total W |
| --- | --- | ---: | ---: | ---: | ---: |
| NPU decoding | 1B | 2.8 | 0 | 1.8 | 4.6 |
| NPU decoding | 4B | 2.9 | 0 | 1.6 | 4.5 |
| NPU prefill | 1B | 0.0 | 3.1 | 1.2 | 4.3 |
| NPU prefill | 4B | 0.0 | 3.4 | 1.1 | 4.5 |
| iGPU decoding | 1B | 31 | 22 | 0 | 53 |
| iGPU decoding | 4B | 31 | 23 | 0 | 54 |
| iGPU prefill | 1B | 33 | 24 | 0 | 57 |
| iGPU prefill | 4B | 33 | 25 | 0 | 58 |
| CPU decoding | 1B | 29 | 0 | 0 | 29 |
| CPU decoding | 4B | 26 | 0 | 0 | 26 |
| CPU prefill | 1B | 24 | 0 | 0 | 24 |
| CPU prefill | 4B | 30 | 0 | 0 | 30 |

## Backend Comparison Policy

Every paper comparison must identify:

- backend: `cpu`, `igpu`, or `npu`,
- metric: prefill TTFT, decode TPS, vision TTFT, power, or TPS/W,
- model variant,
- prompt length,
- decode token count,
- timed-window boundaries,
- warmup and timed iterations,
- host fallbacks,
- power telemetry source,
- result JSON path.

Similarity formulas:

```text
latency_delta_pct = abs(local_seconds - paper_seconds) / paper_seconds * 100
throughput_delta_pct = abs(local_tps - paper_tps) / paper_tps * 100
speedup_delta_pct = abs(local_speedup - paper_speedup) / paper_speedup * 100
power_delta_pct = abs(local_watts - paper_watts) / paper_watts * 100
```

Do not use lit wall time, compile time, manifest generation time, model load
time, BO allocation time, static preload time, or synthetic checksum time as
model latency.

Do not claim speedup from NPU-only data. Speedup requires matching local CPU or
iGPU baseline data from the same benchmark harness.

Projection-weight policy for paper comparisons:

- `--quantized-weights required` is the default for paper-result generation. It
  requires a valid shared cache at `<weights_dir>/q4nx/q4nx_manifest.json` and
  records `quantized_weights_status`, `q4nx_manifest`,
  `q4nx_manifest_sha256`, and `projection_weight_source=q4nx`.
- CPU and iGPU HF/BF16 benchmark paths are diagnostic baselines only. They are
  blocked under `--quantized-weights required` until native packed-Q4NX
  projection operators back the benchmark path. Use `--quantized-weights off`
  only when intentionally collecting non-comparable HF baseline data.
- The runtime decode path now writes the shared Q4NX manifest payloads into one
  contiguous static projection BO before timed decode and binds FusedDQP stitched
  projection arguments as manifest-offset sub-buffers. The direct per-column
  diagnostic route remains labeled as preloaded runner BO plumbing.

## Current Paper-Parity Blockers

Use `gemma3.evidence.reproduction_blockers` as the machine-readable blocker ledger.

Use `docs/npu_runtime_loop.md` for the required blocker resolution order and
the evidence gates for clearing production NPU blockers.

Current 1B text blockers:

| Blocker | Meaning |
| --- | --- |
| `production-prefill-runtime-artifacts-not-cached` | The runtime cache has no per-layer `gemma3_prefill_kv_L*` artifacts for launching production prefill K/V. |
| `prefill-1k-npu-not-wired` | No official 1024-token NPU prefill paper cell runs through the full NPU path. |
| `npu-prefill-kv-cache-not-wired` | Decode diagnostics use synthetic or HF/CPU-produced prefill cache, not an NPU-produced cache. |
| `npu-attention-reduction-not-wired` | Tiled-stat attention can use NPU tile work, but cross-tile softmax/stat reduction is still host-side in the diagnostic path. |

Current 4B text blockers:

- model-kernel launch not wired at the same level as 1B,
- nonlinear model-stage promotion incomplete,
- paper-shape hardware rerun required.

Current 4B vision blockers:

- text 4B blockers,
- vision NPU path not implemented,
- vision NPU path not validated.

Blocker resolution order:

1. Keep focus on 1B 1k text NPU.
2. Promote NPU prefill to produce the decode KV cache.
3. Move tiled attention-stat reduction onto the NPU or explicitly classify it
   outside paper parity.
4. Route FusedDQP through production contiguous static-weight BOs.
5. Wire final norm/logits/sampling policy into timed result records.
6. Only then expand to longer 1B contexts, 4B text, and vision.

## Implementation Roadmap

Each iteration should resolve one backend boundary or one paper blocker.

### Phase 1: Documented Backend Boundaries

Done when:

- This document names shared, CPU, iGPU, and NPU responsibilities.
- Existing files can be grouped under those responsibilities without moving
  code.
- CPU/iGPU/NPU result records use consistent backend names.

### Phase 2: Shared Core Cleanup

Done when:

- Shared modules import without AIR/XRT/pyxrt/ROCm hardware dependencies.
- Model config, artifact discovery, weight metadata, KV-cache metadata, and
  result schema are shared by all backends.
- Backend-specific package imports are lazy.

### Phase 3: CPU And iGPU Baseline Harness

Done when:

- CPU and iGPU run through the same real-artifact benchmark interface.
- Prompt construction, timed windows, power metadata, and result JSON fields are
  identical except for backend-specific device details.
- Initial 1B cells are either `PAPER_MATCH` or `EXPLAINED_DEVIATION`.

### Phase 4: Llama-Style NPU Runtime Shell

Done when Gemma has the NPU equivalent of:

```text
prepare_runtime()
run_npu_prefill()
generate()
```

with:

- cached ELF artifacts,
- per-layer BO keys,
- static input and intermediate output policies,
- static weight preload outside timed windows,
- argument binding validation before launches,
- explicit fallback metadata.

### Phase 5: NPU 1B Prefill And KV Handoff

Done when:

- 1B 1k prefill runs through the NPU path or produces a narrower
  artifact-backed failure.
- K/V cache rows produced by prefill are consumed by decode without host
  replacement.
- Result JSON distinguishes prefill execution, KV construction, and decode
  execution timing.

### Phase 6: NPU Decode Paper Cell

Done when:

- The 26-layer 1B decode loop consumes NPU-produced prefill cache.
- Attention reduction is NPU-owned or explicitly classified as a host fallback
  outside paper parity.
- Static projection weights use production contiguous BO routing.
- Final norm/logits/sampling are NPU-promoted or timed/accounted as host work.
- The 1B 1k decode paper target has a local result or a precise blocker.

### Phase 7: Expansion

Only after 1B 1k NPU correctness, timing, and power evidence is clean:

- run longer 1B contexts,
- expand to 4B text,
- implement and validate the 4B vision path,
- collect TPS/W and final paper-parity reports.

## Evidence Policy

Every evidence update should record:

- exact command,
- git branch and dirty status,
- environment summary,
- model variant,
- backend,
- metric,
- prompt length and decode count,
- artifact format,
- warmup/timed iteration counts,
- timing windows,
- power source if sampled,
- result JSON path,
- blocker classification if not paper-comparable.

Useful focused checks:

```bash
source /home/cj/iron/ironenv/bin/activate
sandbox/bin/lit -v --filter=model_loop build-xrt/programming_examples
sandbox/bin/lit -v --filter=geglu_compile_only build-xrt/programming_examples
PYTHONPATH=programming_examples/gemma3 python -m gemma3.core.quantized_weights --self-test
PYTHONPATH=programming_examples/gemma3 python -m gemma3.evidence.reproduction_blockers
PYTHONPATH=programming_examples/gemma3 python -m gemma3.evidence.paper_compare --validate
```

For doc-only edits, run:

```bash
git diff --check
rg -n "^#{1,4} " programming_examples/gemma3/ARCHITECTURE.md
```

## Guardrails

- Do not claim FastFlowLM parity until real weights, tokenizer, CPU/iGPU/NPU
  baselines, correctness, timing, and power are all present.
- Do not claim paper speedup from one backend alone.
- Do not claim TPS/W without local power logs.
- Do not hide host fallbacks in timed windows.
- Do not expose diagnostic modes as public support just to fill result cells.
- Do not treat compile-only success as runtime support.
- Do not treat synthetic checksums as model accuracy.
- Do not edit `/home/cj/mlir-aie`, reboot, change NPU power mode, delete build
  directories, or clean rebuild without explicit approval.
- Do not commit xclbins, ELFs, large safetensors, tokenizer caches, trace dumps,
  or generated debug IR unless intentionally added as compact fixtures.

## First Implementable Checklist

The next engineering change should not broaden scope. It should take one item
from this list:

- Route decode projections through the manifest-backed contiguous static Q4NX BO
  instead of runner-owned FusedDQP BO sets.
- Wire 1B NPU prefill far enough to produce K/V cache rows for decode.
- Replace host tiled-stat reduction with an NPU-owned reduction path.
- Normalize CPU and iGPU result generation through one shared backend harness.

For 1B text NPU work, use `docs/npu_runtime_loop.md` to select exactly one
production blocker, record the evidence contract, and stop after the accepted
hardware result.

Stop after one item, update result evidence, and rerun the blocker ledger before
starting the next increment.
