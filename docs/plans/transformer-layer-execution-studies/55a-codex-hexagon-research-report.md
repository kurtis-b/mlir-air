# 55a — Codex research report: Qualcomm Hexagon NPU and llama.cpp's `ggml-hexagon` backend

`[2026-08-20]` Produced by Codex (session `01a02062-2c43-7fa3-96b3-956cdcd9fd0e`) on the prompt recorded in [55](55-hexagon-llama-cpp-lessons-for-xdna2.md) §0, and reproduced here **verbatim** as the raw input to doc 55. Doc 55 §0 lists the two places where it was corrected against this repository. Claims it marks **(inferred)** / **(uncertain)** keep those marks.

---

Live web access was available and used. Sources were checked through 20 August 2026. Because upstream `ggml-hexagon` has changed substantially since its 2025 merge, this report distinguishes the original HVX-oriented backend from the current HMX-capable implementation. Claims not established by a primary source are marked **(inferred)** or **(uncertain)**.

## 1. Qualcomm Hexagon NPU architecture for LLM inference

### Scalar, HVX, and HMX units

“Hexagon NPU” is not a single monolithic accelerator. It is a heterogeneous DSP subsystem containing:

- A scalar VLIW core for control flow, address generation, reductions, and scalar arithmetic.
- HVX, the 1,024-bit/128-byte SIMD vector unit used for elementwise operations, quantization, GEMV, and tiled GEMM.
- HMX, Qualcomm’s matrix/tensor engine, used for dense two-dimensional INT8/FP16-style matrix operations.
- DMA engines connecting shared DDR and VTCM.

Qualcomm describes Hexagon as an SMT/VLIW processor integrating scalar, vector, and tensor accelerators around shared memory. Its compiler material depicts tensor and vector engines operating alongside DMA-managed scratch memory rather than as a cache-coherent GPU-like hierarchy. [Qualcomm Hexagon overview](https://www.qualcomm.com/processors/hexagon), [Qualcomm AI compiler presentation](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/AI-compiler-with-polyhedral-mapper.pdf).

HVX has 128-byte architectural vectors. Vector contexts are dynamically associated with DSP hardware threads; the number of contexts is implementation-specific, not fixed by the v68/v69/v73 ISA version. Scalar and HVX accesses are coherent within the DSP’s own memory hierarchy and use the same DSP virtual addresses. [Hexagon v69 HVX Programmer’s Reference](https://docs.qualcomm.com/doc/80-N2040-49/80-N2040-49_REV_AA_Qualcomm_Hexagon_V69_HVX_ProgrammerS_Reference_Manual.pdf), [v73 HVX Programmer’s Reference](https://docs.qualcomm.com/bundle/publicresource/80-N2040-54.pdf).

HMX is a matrix-MAC engine, but Qualcomm does not publish a clean table of HMX-only MACs/cycle, clock, or TOPS for Snapdragon 8 Gen 1/2/3 and 8 Elite. Platform “AI TOPS” should therefore not be treated as HMX throughput. For comparison, Qualcomm publishes 48 dense TOPS for the entire AI subsystem of QCS8550 and 45 TOPS for the Snapdragon X Elite NPU, not isolated HMX numbers. [QCS8550 specifications](https://www.qualcomm.com/internet-of-things/products/q8-series/qcs8550), [Snapdragon X Elite product brief](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/images/company/news-media/media-center/press-kits/snapdragon-summit-2023/documents/SnapdragonXEliteProductBrief.pdf).

### VTCM: size and management

VTCM—Vector Tightly Coupled Memory—is explicitly managed on-chip SRAM:

- It occupies normal DSP virtual-address space but is not an automatically filled cache.
- Software or the compiler decides which weight and activation tiles occupy it.
- DMA normally moves tiles between shared DDR and VTCM.
- It avoids cache eviction and is intended to provide predictable high bandwidth and lower energy.

The ISA leaves VTCM capacity implementation-specific. Widely used compiler target defaults are 4 MiB for v68 and 8 MiB for v69/v73/v75. These are deployment defaults, not an architectural guarantee. [Apache TVM Hexagon target defaults](https://xinetzone.github.io/tvm/_modules/tvm/target/target.html). Current `ggml-hexagon` logs report 8 MiB on tested v75 and v81 systems; 8 MiB for v79 is therefore plausible but still **(inferred)** rather than guaranteed for every product. [v75 runtime log](https://github.com/ggml-org/llama.cpp/issues/26759), [v81 runtime log](https://github.com/ggml-org/llama.cpp/issues/25876).

Applications do not simply assume that all VTCM is permanently theirs. They request it through the HAP compute-resource API. Current llama.cpp queries `HAP_compute_res_query_VTCM`, acquires a scratch allocation, and can release and reacquire it so that another stack such as QNN can run; see [`ggml/src/ggml-hexagon/htp/main.c`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/htp/main.c).

### DSP threads

Hexagon supports simultaneous multithreading, but the number of hardware threads and HVX contexts is product-specific. Software queries both at runtime. Current llama.cpp normally uses one worker per available HVX context and maintains separate HVX work/DMA queues plus an HMX queue; see [`htp/main.c`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/htp/main.c) and [`htp/work-queue.c`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/htp/work-queue.c).

Observed configurations include four threads/HVX contexts on one v75 system and eight on v81. I did not find a primary Qualcomm table giving exact mobile v68/v69/v73/v79 thread counts. In particular, claims that v79 universally has six HVX contexts should be treated as **uncertain**.

### FastRPC and CPU-to-DSP communication

FastRPC is the control plane between the application processor and Hexagon:

1. A host stub marshals an IDL-defined call.
2. The FastRPC user library passes it to the kernel driver.
3. The message crosses RPMsg/transport into a DSP process domain.
4. A generated skeleton invokes the DSP implementation.

Qualcomm’s implementation and architecture are public in the [FastRPC repository](https://github.com/qualcomm/fastrpc). Linux’s driver also shows the import and mapping of DMA-BUF-backed memory into a FastRPC process domain. [Linux `drivers/misc/fastrpc.c`](https://github.com/torvalds/linux/blob/master/drivers/misc/fastrpc.c).

A synchronous FastRPC per GGML operation would be expensive. Modern Hexagon backends therefore use FastRPC primarily to establish the session and shared queues, then submit work through shared queue structures and notifications.

### Shared DDR, rpcmem, ION/DMA-BUF, and cache handling

The CPU and DSP ultimately access the same system DDR, but generally through distinct virtual mappings and cache domains:

- Older Android platforms commonly allocate shareable memory through ION.
- Newer systems use DMA-BUF heaps.
- Qualcomm’s `rpcmem` obtains a shareable allocation and file descriptor.
- FastRPC imports that descriptor into the DSP process domain.
- Software still performs explicit cache clean/invalidate operations at ownership transitions.

Thus “zero-copy” means that CPU and DSP can map the same physical allocation without a staging copy. It does not imply universal hardware cache coherence or freedom from cache-maintenance overhead. Current llama.cpp’s explicit dirty tracking and `qurt_mem_cache_clean`/invalidate operations demonstrate this distinction; see [`ggml/src/ggml-hexagon/htp/htp-tensor.c`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/htp/htp-tensor.c) and host allocation/mapping in [`ggml-hexagon.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/ggml-hexagon.cpp).

### Power levels, HAP power APIs, and DCVS

Hexagon applications vote for compute resources and performance through HAP APIs. DCVS—Dynamic Clock and Voltage Scaling—can vary core and bus levels based on load and thermal conditions.

Current llama.cpp requests compute application class, powers HVX and HMX, disables adaptive DCVS for the session, and votes core and bus minimum/target/maximum levels to their maximum performance settings. On newer architectures it also uses the HMX v2 power API and maximum voltage/performance-corner votes. This is a benchmark-oriented “maximum performance” policy rather than energy-optimal adaptive operation; see [`htp/main.c`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/htp/main.c).

### Generational differences and quantitative limits

| Hexagon generation | Representative product | VTCM evidence | Memory capability | Relevant change |
|---|---|---:|---:|---|
| v68 | Snapdragon 888-era | 4 MiB compiler default | LPDDR5-class | Earlier HVX/tensor generation; not a primary target of the 2025 llama.cpp merge. |
| v69 | Snapdragon 8 Gen 1 | 8 MiB compiler default | LPDDR5 up to 3200 MHz | Faster fused scalar/vector/tensor subsystem; still below the v73 minimum advertised by the original upstream backend. |
| v73 | Snapdragon 8 Gen 2; commonly associated with X Elite **(uncertain for every X Elite SKU)** | 8 MiB compiler default | LPDDR5X up to 4200 MHz; X Elite 136 GB/s | First generation explicitly targeted by the merged llama.cpp backend. |
| v75 | Snapdragon 8 Gen 3 | 8 MiB default and observed | LPDDR5X up to 4800 MHz | Four HVX contexts observed; HMX supported by current backend. |
| v79 | Snapdragon 8 Elite | 8 MiB **(inferred)** | LPDDR5X up to 5300 MHz | Current flagship target of the original upstream benchmark. |
| v81 | 8 Elite Gen 5-era | 8 MiB observed | Product-specific | Current upstream source supports it; outside the user-requested generation list but relevant to newer benchmark results. |

The mobile memory-rate figures come from Qualcomm’s [8 Gen 1 brief](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/snapdragon_8_gen_1_mobile_platform_product_brief_1.pdf), [8 Gen 2 brief](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/Snapdragon-8-Gen-2-Product-Brief.pdf), [8 Gen 3 brief](https://docs.qualcomm.com/doc/87-71408-1/87-71408-1_REV_C_Snapdragon_8_gen_3_Mobile_Platform_Product_Brief.pdf), and [8 Elite brief](https://docs.qualcomm.com/bundle/publicresource/87-83196-1_REV_A_Snapdragon_8_Elite_Mobile_Platform_Product_Brief.pdf). Assuming a 64-bit mobile memory interface gives theoretical peaks of approximately 51.2, 67.2, 76.8, and 84.8 GB/s respectively **(inferred)**. Sustained bandwidth available to HTP is lower and shared with CPU, GPU, ISP, and display. X Elite’s published 136 GB/s corresponds to its wider 128-bit LPDDR5X interface.

No reliable primary source was found for HMX-only throughput or HMX clock on these products. Those values should not be reverse-engineered from system TOPS.

## 2. llama.cpp’s `ggml-hexagon` backend

### Lineage and current scope

Qualcomm engineers Rajdeep Ganguly and Todor Boinovski contributed the backend through [PR #16547](https://github.com/ggml-org/llama.cpp/pull/16547), merged on 22 October 2025 as commit `63d2fc4`. That initial implementation targeted v73/v75/v79/v81 and was principally an HVX backend supporting Q4_0, Q8_0, MXFP4, and F32.

HMX was not part of that original merge. It arrived through [PR #20693](https://github.com/ggml-org/llama.cpp/pull/20693), merged as `74c42ee1f4f0fa3609c8aef543edb6f307826063` on 19 March 2026. Later changes added general fusion, rewritten HMX/HVX matmul, flash attention, 32×32 repacking, DMA pipelines, and graph caching—particularly [PR #23835](https://github.com/ggml-org/llama.cpp/pull/23835), [PR #24954](https://github.com/ggml-org/llama.cpp/pull/24954), and [PR #25085](https://github.com/ggml-org/llama.cpp/pull/25085).

### Host/DSP split and supported operations

The host remains responsible for:

- GGML graph construction and backend scheduling.
- Tokenization, sampling, and the outer autoregressive loop.
- Selecting HTP-compatible graph partitions.
- Repacking weights during buffer initialization.
- CPU execution of unsupported operations, data types, layouts, or shapes.

Current DSP code supports substantially more than matmul. The dispatch switch in [`ggml-hexagon.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/ggml-hexagon.cpp) includes:

- `MUL_MAT`, `MUL_MAT_ID`, and `ADD_ID`.
- ADD, SUB, MUL, DIV.
- NORM, L2_NORM, RMS_NORM, SCALE, CLAMP, SQR, SQRT, and row reductions.
- RoPE and softmax.
- SILU, GELU variants, sigmoid, tanh, softplus, and GLU/SwiGLU/GEGLU forms.
- `FLASH_ATTN_EXT`.
- Row gather/scatter, copies, permutations, concatenation, padding, and several newer model-specific operators.

The original 2025 merge advertised matmul, MoE matmul-id, RMSNorm, RoPE, softmax, GLU/SwiGLU, and elementary operations; flash attention was added experimentally later and is now implemented by HVX/HMX kernels in [`htp/flash-attn-ops.c`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/htp/flash-attn-ops.c).

This is therefore no longer a “matmul-only” backend. Nevertheless, any unsupported node forces a scheduler boundary and may move intermediate data through a host-visible buffer.

### Session and `dspqueue` dispatch model

The backend does not make one synchronous FastRPC call per GGML operation, and it does not compile the entire model into one fixed HTP graph.

At session startup, host code:

- Opens a FastRPC process domain.
- Maps shared buffers.
- Creates and exports a `dspqueue`.
- Makes an initial FastRPC call that starts the DSP service and imports the queue.

Thereafter, GGML operations are serialized into operation descriptors and written to the shared queue. The default batch can hold 1,024 operations, with up to 16 pending batches. The DSP’s `process_opbatch()` walks those descriptors, invokes the associated kernels, and posts one response for the batch. See [`ggml-hexagon.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/ggml-hexagon.cpp), [`htp/main.c`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/htp/main.c), and [`htp/work-queue.c`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/htp/work-queue.c).

Consequently, round trips per token are:

> Number of HTP graph partitions and queue batches, plus synchronization at CPU/HTP boundaries.

They are not intrinsically one per operation or one per token. An early Llama 3.2 1B log contained 503 nodes divided into 41 backend splits. Current fusion and flash-attention work reduces that number, but the exact count remains model-, quantization-, and option-dependent.

Current code also caches precomputed operation nodes and kernel parameters for recurring graph shapes. This avoids repeatedly solving layouts and rebuilding descriptors, but remains dynamic GGML execution rather than QAIRT-style whole-model compilation.

### Buffer management and weight residency

Host-visible HTP buffers are allocated using `rpcmem_alloc2`, exported as file descriptors, and mapped into the FastRPC process domain. Compatible GGML tensors can therefore reside in a physical DDR allocation visible to both CPU and DSP. See the allocator and buffer classes in [`ggml-hexagon.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/ggml-hexagon.cpp).

Weights are normally:

1. Allocated in mapped DDR.
2. Repacked once into an HTP-friendly layout.
3. Kept resident for subsequent prefill/decode calls.
4. DMA-tiled into VTCM when a kernel executes.

There is no per-token retransmission of the full model. “Zero-copy” is qualified: the same allocation can be shared, but repacking creates a transformed resident copy and host/DSP ownership transitions require explicit cache maintenance. Each FastRPC process domain also has a practical virtual-mapping limit of roughly 3.5 GiB in the reported Android configurations, which motivates multiple sessions.

### VTCM, `spad`, and DMA pipelining

On startup the DSP queries available VTCM and requests the usable region through HAP compute-resource APIs. The software scratch allocator divides that region among operation-specific `spad` objects. A `spad` is a scratchpad descriptor, usually backed by VTCM but able to fall back to a DDR scratch allocation if necessary; see [`htp/htp-vtcm.h`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/htp/htp-vtcm.h) and [`htp/main.c`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/htp/main.c).

Weights are too large to live permanently in VTCM. Matmul and attention therefore stream tiles:

- DMA loads the next weight/activation tile.
- HVX or HMX computes on the current tile.
- Double-buffered scratch areas alternate roles.
- Results are written back or retained for the next fused operation when possible.

Flash attention visibly allocates paired K, V, and score scratch buffers and overlaps their DMA/compute phases in [`htp/flash-attn-ops.c`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/htp/flash-attn-ops.c). [PR #24954](https://github.com/ggml-org/llama.cpp/pull/24954) similarly restored activation prefetch and configurable DMA depth for matmul.

The initial graph optimizer also grouped Q/K/V and FFN projections sharing an activation so that dynamic activation quantization could be reused instead of repeated.

### Quantization and repacked layouts

The initial backend supported Q4_0, Q8_0, MXFP4, and F32. Current repacking code additionally recognizes Q4_1 and IQ4_NL, while HMX paths support selected F16/F32 cases; exact combinations remain operation- and architecture-dependent. The authoritative implementation is in [`ggml-hexagon.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/ggml-hexagon.cpp) and [`htp/matmul-ops.c`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/htp/matmul-ops.c).

Repacking is necessary because native GGUF quantized rows are designed for portable CPU decoding, not for:

- 128-byte HVX vector loads.
- HMX tile shapes.
- Contiguous scale/zero-point access.
- DMA alignment and burst size.
- Simultaneous access by multiple DSP workers.

The original implementation used a 32×4×2-oriented arrangement that grouped 256 elements into two HVX vectors and separated quantized payloads from scales. The newer matmul work uses 32×32 tiled packing with HMX-aligned DMA. Repacking is paid once at model load and amortized over every generated token.

### DSP multithreading

The backend queries the hardware-thread and HVX-unit counts and defaults to using all available HVX contexts. Work is divided into rows, matrix tiles, or MoE expert ranges and submitted through a common worker queue. Each HVX worker has a DMA queue; HMX has its own execution queue. See [`htp/main.c`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/htp/main.c), [`htp/work-queue.c`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/htp/work-queue.c), and [`htp/dma-queue.c`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/htp/dma-queue.c).

`GGML_HEXAGON_NHVX` can reduce the worker count. It cannot create additional physical vector units, and additional ggml devices do not create additional HMX engines.

### Environment variables and options

Current upstream source recognizes the following principal options:

| Variable | Function |
|---|---|
| `GGML_HEXAGON_NDEV` | Number of logical ggml Hexagon devices/sessions. |
| `GGML_HEXAGON_NHVX` | Number of HVX workers; zero means all available. |
| `GGML_HEXAGON_NHMX` | HMX worker/session control. |
| `GGML_HEXAGON_USE_HMX` | Enable HMX. |
| `GGML_HEXAGON_HOSTBUF` | Use host-visible shared buffer allocation. |
| `GGML_HEXAGON_VERBOSE` | Diagnostic verbosity. |
| `GGML_HEXAGON_PROFILE` | Profiling controls. |
| `GGML_HEXAGON_OPSTAGE` | Enable queue and/or compute stages. |
| `GGML_HEXAGON_OPBATCH` | Maximum operations in a queue batch. |
| `GGML_HEXAGON_OPQUEUE` | Number of pending batches. |
| `GGML_HEXAGON_OPPOLL` | Polling behavior. |
| `GGML_HEXAGON_OPFUSION` | Generic operation fusion. |
| `GGML_HEXAGON_OPFILTER` | Select/filter offloaded operations. |
| `GGML_HEXAGON_MM_SELECT` | Matmul fallback/selection order. |
| `GGML_HEXAGON_FA_SELECT` | Flash-attention selection order. |
| `GGML_HEXAGON_ARCH` | Architecture override. |
| `GGML_HEXAGON_ETM`, `VMEM`, `MBUF`, `OPTRACE` | Tracing and memory/debug controls. |

The defaults and parsing are in [`ggml-hexagon.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/ggml-hexagon.cpp).

Two frequently cited names are historical:

- `GGML_HEXAGON_OPMASK` existed around the 2025 merge. Current source splits that function between `OPSTAGE` and `OPFILTER`.
- `GGML_HEXAGON_EXPERIMENTAL` temporarily gated experimental flash attention. It was removed in favor of `OPFILTER`; see the [b8754 release notes](https://newreleases.io/project/github/ggml-org/llama.cpp/release/b8754).

### Multiple logical NPU devices and sessions

`GGML_HEXAGON_NDEV=N` presents one physical Hexagon NPU as multiple ggml devices. Each device has its own FastRPC session/process-domain address space; the model can be divided by layer among them.

The principal benefit is address-space capacity. A model larger than one DSP process domain’s mapping limit can be spread over several sessions. The current Snapdragon documentation uses four HTP sessions for a large GPT-OSS-20B configuration. [Snapdragon backend developer guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/developer.md).

This does not multiply physical HMX/HVX throughput or DDR bandwidth. It may also introduce extra scheduler boundaries when execution moves from one session’s layers to another. On older Android/HTP combinations the source forces a single device below v75, reflecting platform limitations.

### Prefill versus decode

Prefill presents large token dimensions and is predominantly GEMM. HMX can use large two-dimensional tiles, DMA pipelines, and high arithmetic intensity.

Decode usually has a single current token and reduces many projections to GEMV-like, bandwidth-dominated operations. The backend deterministically chooses HMX, tiled-HVX, flat-HVX, or CPU paths from tensor dimensions, type, VTCM capacity, and `MM_SELECT`; it does not run a general online autotuner. Parameters are cached once selected.

This explains why HMX and flash attention can increase prompt processing by multiples while token generation improves much less. Decode still streams a large fraction of every layer’s weights from DDR for every token.

### Power/clock setup

The HTP service votes maximum core and bus performance, disables DCVS adaptation, prevents sleep during work, and powers both HVX and HMX. This is implemented in [`htp/main.c`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/htp/main.c). It improves reproducibility and short-burst latency but can produce thermal throttling during long runs.

### Reported performance and remaining bottlenecks

The following are the upstream figures I could verify:

| Platform/model | Quantization | Prefill | Decode | Qualification |
|---|---:|---:|---:|---|
| Snapdragon 8 Elite, Galaxy S25+, Llama 3.2 1B | Q4_0 | `pp128` 169.42 ± 1.75 tok/s | `tg64` 51.54 ± 1.13 tok/s | Published with the original backend. A short interactive run reported 136.21 prompt and 51.57 generation tok/s. [PR #16547](https://github.com/ggml-org/llama.cpp/pull/16547) |
| 8 Elite Gen 5/v81, Galaxy S26+, Llama 3.2 1B | repository test configuration | 4,027.72 tok/s | 54.22 tok/s | Newer silicon and rewritten HMX/FA backend; not an 8 Elite v79 result. [PR #25085](https://github.com/ggml-org/llama.cpp/pull/25085) |
| 8 Elite Gen 5/v81, Galaxy S26+, Llama 3.2 3B | repository test configuration | 1,819.01 tok/s | 25.01 tok/s | Context only; not the requested v79/X Elite platform. [PR #25085](https://github.com/ggml-org/llama.cpp/pull/25085) |
| Unidentified newer 16-GB Snapdragon device, GPT-OSS-20B | Mixed GGUF containing MXFP4 tensors, four sessions | 51.25 tok/s over a 197-token prompt | 18.39 tok/s | Device/SoC is not identified, so it cannot safely be attributed to 8 Elite or X Elite. [Developer guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/developer.md) |

I found no verified upstream 8 Elite or X Elite token-rate table for Llama 3.2 3B or Llama 3.1 8B. PR #16547 says those models were tested but does not publish comparable measurements. I also found no defensible X Elite HTP result for GPT-OSS-20B. CPU-only X Elite llama.cpp numbers are not substitutes for HTP results.

The remaining bottlenecks are:

- **DDR bandwidth:** decode repeatedly streams resident weights from DDR. The original author explicitly identified native GGUF layout and DMA efficiency as crucial to usable bandwidth. [PR #16547](https://github.com/ggml-org/llama.cpp/pull/16547)
- **Backend splits:** unsupported quant formats or operations leave many small HTP regions. A v73 report with unsupported Q4_K tensors showed hundreds of graph splits and little useful NPU work. [Discussion #22791](https://github.com/ggml-org/llama.cpp/discussions/22791)
- **Queue/transport synchronization:** `dspqueue` amortizes FastRPC, but every HTP/CPU boundary still needs completion and cache ownership handling.
- **Host-side work:** tokenization and sampling are small, but host attention, fallback normalization, copies, or quant conversion can dominate if they fragment the graph.
- **Thermals and shared bandwidth:** maximum power voting is not a guarantee of sustained clock during long generation.

## 3. Earlier and alternative Qualcomm paths

### QNN-based `ggml-qnn`

Two prominent upstream efforts explored Qualcomm’s higher-level QNN API.

[PR #12326](https://github.com/ggml-org/llama.cpp/pull/12326) was closed without merging after a large prototype. Its discussion distinguished three designs:

1. Turn each GGML operation into a small QNN graph.
2. Invoke custom cDSP kernels directly for each operation.
3. Convert an entire GGML graph into one large QNN graph.

The first design is unattractive for autoregressive inference because graph construction/finalization, FastRPC transport, tensor binding, and CPU/QNN transitions can exceed the compute time of small elementwise operations. It also prevents QNN’s compiler from seeing a whole layer and jointly choosing fusion, VTCM allocation, or DMA schedules. That is the systems problem with “graph per op”; it does not mean that QNN itself is unsuitable.

The third design exposes more optimization opportunity but requires translating hundreds of GGML nodes and many operation variants into a stable, target-specific QNN graph. Contributors described the SDK as large and comparatively opaque and found the integration difficult to maintain. Those are contributor observations in PR #12326, not an official Qualcomm or llama.cpp maintainer judgment.

[PR #12063](https://github.com/ggml-org/llama.cpp/pull/12063) is a more ambitious graph-mapping and caching effort. As of the research date it remained a draft, with F16/F32 Llama testing documented and quantized matmul/fallback work still incomplete. It should not be characterized as a rejected finished backend: the public evidence supports “unfinished and unmerged,” not “proven technically impossible.”

The direct `ggml-hexagon` backend occupies a middle ground. It preserves llama.cpp’s dynamic GGML scheduler but implements low-level kernels, repacking, VTCM planning, fusion, and shared queues without constructing a QNN graph for every operation.

### Genie and QAIRT

Qualcomm’s Genie/QAIRT deployment model is substantially more ahead-of-time:

- The model is converted to ONNX/QNN-compatible form.
- It is quantized with Qualcomm tooling such as AIMET.
- Large models are split into graph parts.
- Separate AR-1 decode and AR-128-style prefill variants are compiled.
- Corresponding parts are linked into shared-weight QNN context binaries.
- Genie orchestrates those binaries, KV-cache state, token generation, and sampling.

This flow is documented in Qualcomm AI Hub’s [LLM onboarding guide](https://github.com/qualcomm/ai-hub-models/blob/main/tutorials/llm/onboarding.md) and the [Gen AI Inference Extensions overview](https://www.qualcomm.com/developer/software/gen-ai-inference-extensions). Sample deployments contain several context binaries rather than one dynamically interpreted GGML graph. [Genie Python sample](https://github.com/qualcomm/qai-appbuilder/blob/main/samples/genie/python/README.md).

The trade-off is clear:

- QAIRT/Genie can compile globally across graph parts, specialize prefill and decode shapes, plan VTCM, share weights, and minimize runtime decisions.
- `ggml-hexagon` is easier to integrate with GGUF models and llama.cpp’s dynamic scheduler, but relies on operation coverage, local fusion, cached parameter solving, and queue batching to approach the same efficiency.
- QAIRT artifacts are more model-, shape-, SDK-, and target-specific.
- GGML retains portability and rapid model support at the cost of more dynamic boundaries.

Qualcomm’s newer GenieX even exposes distinct QAIRT-binary and llama.cpp/GGUF routes, reinforcing that these are complementary deployment models rather than the same backend. [GenieX SDK](https://github.com/qualcomm/GenieX/blob/main/sdk/README.md).

## 4. SYSTEM-LEVEL lessons for AMD Ryzen AI NPU / MLIR-AIR

The local execution study provides an unusually useful decomposition. For its measured configuration, `offload`, `runlist`, `coarse`, and `fused` respectively produced 6/5/4/1 host submissions, 6/391/131/3 runlist entries, and 19/403/402/19 synchronization boundaries. These counts are configuration-specific, but they show that “fewer host calls,” “fewer device commands,” and “fewer synchronization boundaries” are different objectives. [MLIR-AIR execution-strategy study](/home/cj/mlir-air/docs/plans/transformer-layer-execution-studies/08-phase-e-execution-strategies.md:15).

### Dispatch overhead

**Hexagon lesson.** A synchronous FastRPC per operation is too expensive. `dspqueue` batches operation descriptors; fusion and flash attention reduce graph splits; cached descriptors avoid repeated setup.

**AMD analogue.** One `xrt.run()` per leaf kernel pays tens to hundreds of microseconds repeatedly. Runlists amortize host submission; multi-launch ELFs move several `air.launch` regions below one XRT dispatch. The local optimization guide reports roughly 50–200 µs per XRT call and a Llama example improving from 16 to 3 calls per layer. [Multi-launch guidance](/home/cj/mlir-air/.codex/skills/opt-merge-multi-launch-kernels/SKILL.md:3), [prefill optimization](/home/cj/mlir-air/.codex/skills/phase-4-prefill-optimization/SKILL.md:104).

**Verdict: transfers strongly.** `offload` is appropriate as a correctness baseline, not an assumed performance endpoint. `runlist` corresponds most closely to Hexagon’s queued descriptors; `coarse` and `fused` correspond to moving work below the dispatch boundary. The twist is that a runlist may still contain hundreds of synchronization-heavy entries, so host-call count alone is insufficient.

### Memory residency

**Hexagon lesson.** Keep repacked weights in shared DDR for the lifetime of the model. Transfer only tiles into VTCM and avoid recreating or rewriting buffers per token.

**AMD analogue.** Allocate XRT buffer objects once, preload per-layer weights, reuse overwritten intermediate BOs, and avoid host round trips between kernels. DDR is still “device-visible,” but it is not free: shim DMA traffic and cache/ownership transitions remain costs.

**Verdict: transfers directly.** All four modes should use persistent BOs. The advantage grows in decode because every avoidable weight upload or BO allocation is repeated once per generated token. `coarse`/`fused` additionally permit intermediate activations to remain in device-managed storage rather than returning to host DRAM.

### On-chip memory management

**Hexagon lesson.** Treat VTCM as explicit scratch, not a cache. Select tile sizes from available capacity and overlap DMA with compute using double buffers. Fusion is valuable only while the fused live set still fits.

**AMD analogue.** XDNA2 does not offer one flat 8-MiB scratchpad. It has distributed 64-KiB tile L1 memories and 512-KiB memtile L2 memories, connected through explicit routes and shim/memtile DMA.

**Verdict: transfers with a twist.** The principle transfers; the allocator does not. MLIR-AIR must plan placement, routes, locks, DMA buffer descriptors, and double buffering across multiple tiles. The kernel registry should record L1/L2 footprint, memtile bank use, DMA depth, route/channel demand, and whether producer/consumer layouts permit on-device handoff. A “whole layer fused” artifact may be worse if it exhausts BDs, channels, tile memory, or routing.

### Host/device partitioning per operation

**Hexagon lesson.** Offloading only matmul is fragile. One unsupported norm, RoPE, quant format, or attention node can fragment a transformer block into many DSP regions. Broad operation coverage and small, deliberate fusion patterns often matter more than the isolated peak of one matmul kernel.

**AMD analogue.** Host attention in `offload`, individually dispatched GEMMs, and host-side transposes create the same boundary problem. The relevant cost is:

`kernel time + dispatch + synchronization + boundary bytes + layout conversion`.

**Verdict: transfers strongly.** Partition at stable layer subgraphs—such as RMSNorm/QKV/RoPE and attention-output/FFN groups—not simply at every operator with an NPU implementation. Keep `offload` as a diagnostic baseline. Let measured boundary cost determine whether attention remains on the host. `fused` should be preferred only when it preserves numerical correctness and does not create worse DRAM traffic or resource pressure.

### Quantized-weight layout

**Hexagon lesson.** Portable GGUF layout is not accelerator layout. Repack once into tiles matching vector width, matrix-engine shape, scale access, and DMA alignment, then keep the result resident.

**AMD analogue.** NPU kernels should consume registry-defined packed weights aligned with AIE vector/MAC shapes and shim/memtile DMA bursts. Activations should use a consistent sequence-first or other producer-consumer-compatible layout to eliminate host transposes.

**Verdict: transfers directly.** Treat packed layout as part of the kernel ABI, not an incidental preprocessing detail. The registry should version it and record:

- Supported source quant format.
- Packed block/tile shape.
- Scale/zero-point organization.
- Alignment and padding.
- Required activation layout.
- Load-time repack cost and amortization point.
- Whether prefill and decode share the same packed weights.

A nominal Q4 kernel that rereads awkward scales or repacks every call can lose to a wider datatype with a clean DMA layout.

### Per-workload tuning

**Hexagon lesson.** Current `ggml-hexagon` does not perform general empirical autotuning per model. It computes parameters deterministically from shape, datatype, available VTCM, thread count, and static cost rules, with environment-variable overrides. It caches the result. Prefill and decode naturally select different GEMM/GEMV and attention paths.

**AMD analogue.** MLIR-AIR already has the stronger foundation: a registry of measured tile configurations and explicit execution modes.

**Verdict: transfers with a twist.** Do not choose one global “best mode.” Extend registry keys and measurements to include:

- Prefill versus decode.
- Sequence length and query/key length.
- Batch size.
- Hidden/FFN/head dimensions.
- Datatype and packed-weight layout.
- Execution mode.
- XRT submissions, runlist entries, sync boundaries, and bytes.
- L1/L2 footprint and DMA depth.
- Power mode and thermal state.

Use deterministic lookup among measured configurations, with a safe fallback. Static model-name tuning is too coarse: Llama 3.2 1B prefill and one-token decode are materially different workloads even when their weights are identical.

### Power mode

**Hexagon lesson.** The backend explicitly votes maximum core/bus/HMX performance and disables DCVS. That produces good short-run numbers but does not describe energy efficiency or long-duration thermal behavior.

**AMD analogue.** XRT/NPU power mode, clock state, hardware-context configuration, temperature, and competing system traffic can materially change dispatch latency and sustained bandwidth.

**Verdict: transfers directly.** Use maximum-performance mode for kernel rooflines and controlled comparison, but report it. For production conclusions, also measure a sustained regime after warm-up and thermal stabilization. The registry should either key results by power mode or reject comparisons made under different modes.

Overall, the Hexagon evidence does not imply that `fused` must always win. It implies that dispatch, boundary bytes, packed layout, scratch capacity, and power state must be optimized together. For MLIR-AIR, the most defensible strategy is:

1. Preserve `offload` as the transparent correctness/control baseline.
2. Use `runlist` to amortize XRT submission while retaining kernel modularity.
3. Use `coarse` for repeatable layer subgraphs whose intermediates can remain on-device.
4. Use `fused` where registry evidence shows that reduced dispatch and synchronization outweigh increased ELF specialization, resource pressure, or DRAM traffic.

That is the direct analogue of `ggml-hexagon`’s evolution from isolated HVX operations toward persistent buffers, queued dispatch, HMX specialization, tiled repacking, flash attention, and selective fusion—without requiring every model to become a fully compiled QAIRT graph.

