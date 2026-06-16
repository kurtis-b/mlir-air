# Gemma3

This programming example collects source-level MLIR-AIR/NPU2 kernels for a
Gemma3-style transformer dataflow and makes their physical herd shape selectable
from the command line. The supported shapes are `2x4`, `4x4`, and `8x4`.
For the roadmap from these standalone kernels to a host-driven Gemma3 model
loop, see [Gemma3 Architecture](ARCHITECTURE.md).

The example uses Peano/AIE2P microkernels throughout:

- `gemma3.kernels.q4nx` with `aie_kernels/q4nx_opt.cc`: Q4NX int4 block dequantization for 32x256 blocks.
- `gemma3.kernels.bf16_tiled_mm`: a Gemma-style wrapper around the optimized BF16 tiled GEMM
  implementation in `../matrix_multiplication/bf16/mm_aie2p.cc`.
- `gemma3.kernels.fused_dqp` with `aie_kernels/fused_dqp_opt.cc`: fused Q4NX dequantization and
  projection over 32x256 row blocks.
- `gemma3.kernels.flowqkv` with `aie_kernels/flow_attention_opt.cc`: grouped prefill attention over a
  query chunk and per-group KV cache rows.
- `gemma3.kernels.flowkv` with `aie_kernels/flow_attention_opt.cc`: decode attention as the q-chunk-1
  specialization of the grouped attention path.

## Quick Start

From this directory, with the usual MLIR-AIR NPU environment active:

```bash
make run-q4nx HERD_SHAPE=2x4 COMPILE_MODE=compile-only OUTPUT_FORMAT=elf
make run-mm HERD_SHAPE=4x4 COMPILE_MODE=compile-only OUTPUT_FORMAT=elf
make run-fused-dqp HERD_SHAPE=8x4 COMPILE_MODE=compile-only OUTPUT_FORMAT=elf
make run-flowqkv HERD_SHAPE=4x4 COMPILE_MODE=compile-only OUTPUT_FORMAT=elf
make run-flowkv HERD_SHAPE=2x4 COMPILE_MODE=compile-only OUTPUT_FORMAT=elf
make run-q4nx-8x4-rowband-fallback COMPILE_MODE=compile-only OUTPUT_FORMAT=elf
make run-fused-dqp-paper COMPILE_MODE=compile-only OUTPUT_FORMAT=elf
make run-flowqkv-paper COMPILE_MODE=compile-only OUTPUT_FORMAT=elf
make run-flowkv-paper COMPILE_MODE=compile-only OUTPUT_FORMAT=elf
```

Run the full compile sweep:

```bash
make sweep COMPILE_MODE=compile-only OUTPUT_FORMAT=elf
```

Run a hardware validation target by changing `COMPILE_MODE`:

```bash
make run-q4nx HERD_SHAPE=2x4 COMPILE_MODE=compile-and-run OUTPUT_FORMAT=elf
```

For hardware runs, source the MLIR-AIR environment first and XRT setup after it so
`pyxrt` remains on `PYTHONPATH`. Do not wrap these targets in an outer
`flock /tmp/npu.lock`; `XRTRunner` already takes that lock internally. Set
`DEBUG_IR=1` on the Make target to retain XRTBackend lowering artifacts for
routing/debug inspection.


## Model Runtime Loop

The model-level Gemma3 1B/1k NPU loop is scaffolded separately from the
standalone kernel targets. Use the Llama 3.2 1B example as the reference pattern
for runtime organization only: cached artifacts, per-layer BO reuse, static
input skipping, K/V handoff, and profile/verify separation. Llama output is not
accepted Gemma3 evidence.

```bash
make model-blockers
make model-prepare
make model-prefill
make model-generate
make model-validate
make model-loop
```

`model-validate` runs `gemma3.evidence.npu_runtime_contracts --allow-blocked`
against the current result files so the present blocked state is inspectable.
To clear a blocker, run the same validator without `--allow-blocked`; the
runtime contract is documented in `docs/npu_runtime_loop.md`.

## Herd Shapes

`HERD_SHAPE` controls both the physical herd dimensions and the default logical
workload size. The Python drivers expose the same setting as `--herd-shape`.

| Shape | CTs | Default workload |
| --- | ---: | --- |
| `2x4` | 8 | 8 Q4NX/projection/attention groups, MM M=64 N=128 |
| `4x4` | 16 | 16 Q4NX/projection/attention groups, MM M=128 N=128 |
| `8x4` | 32 | 32 Q4NX/projection/attention groups, MM M=256 N=128 |

The problem dimensions remain intentionally compact so the example is usable as
a compile and runtime smoke test. Override `Q_CHUNK`, `KV_LEN`, `KV_CHUNK`,
`HEAD_DIM`, `MM_M`, `MM_N`, and related Make variables for larger experiments.

`Q4NX_OUTPUT_MODE`, `FUSED_DQP_OUTPUT_MODE`, `FLOWQKV_OUTPUT_MODE`, and
`FLOWKV_OUTPUT_MODE` select `auto`, `direct`, or `l2-gather` where supported.
`auto` keeps Q4NX on L2-gather for every shape and uses
L2-gather for 8x4 FusedDQP/FlowQKV/FlowKV routes. Unsupported combinations
fail during argument parsing with the resource, packet-backend, routing,
or runtime-scheduling reason instead of reaching AIR lowering. Individual
drivers also accept `packet-direct` only to report the diagnostic unsupported
reason; sweeps do not expose it.

| Kernel | 2x4 modes | 4x4 modes | 8x4 modes |
| --- | --- | --- | --- |
| Q4NX | direct, l2-gather | direct, l2-gather | l2-gather |
| FusedDQP | direct, l2-gather | direct, l2-gather | l2-gather |
| FlowQKV | direct, l2-gather | direct, l2-gather | l2-gather |
| FlowKV | direct | direct | l2-gather |

Q4NX and FusedDQP 8x4 direct output exhaust shim DMA channels. A
packet-direct diagnostic route was also tested for both kernels, but it is not
exposed as a supported mode: the AIR is compile-legal and now lowers to
distinct packet-tagged shim allocations, but hardware validation corrupts output
when multiple independent CT rows share one shim S2MM packet channel. Treat the
remaining packet-direct issue as a packet-route/shim DMA schedule limitation,
not as a full-physical Gemma success path. FlowQKV and FlowKV 8x4 direct output
are likewise not exposed. FlowKV small-shape L2 gather is not exposed because
the source-level diagnostic route now compiles only when staged through an
explicit CT-to-L2 output channel, but 2x4 hardware execution still times out
with `ERT_CMD_STATE_TIMEOUT` and `ctx_pc=0x28B060AD`; `xrt-smi examine -r all`
then reports no active contexts. Treat this remaining mode as a channel/runtime
scheduling bug, not as a supported gather route.

`run-q4nx-8x4-rowband-fallback` is a logical 8x4 Q4NX workload using two
sequential host-side physical 4x4 row-band executions. It is a
fallback/workaround target, not a full physical 8x4 utilization result.

`FUSED_DQP_SCHEDULE_MODE`, `FLOWQKV_SCHEDULE_MODE`, and
`FLOWKV_SCHEDULE_MODE` select `smoke` or `paper`. The default `smoke` mode keeps
the validated compact herd sweep. The `run-*-paper` targets enable the
paper-style data layouts and online accumulator kernels with compact defaults;
use `PAPER_FUSED_DQP_COL_BLOCKS`, `FUSED_DQP_COL_CHUNK`,
`FUSED_DQP_PIPE_ROW_CHUNK`, `PAPER_KV_LEN`, `PAPER_KV_CHUNK`,
`PAPER_HEAD_DIM`, `PAPER_KV_GROUPS`, and `PAPER_HEADS_PER_KV` to scale
those targets.
`FLOWQKV_KV_STAGING` and `FLOWKV_KV_STAGING` select `replicated`,
`shared`, or `pipeline` paper K/V staging. FlowQKV and FlowKV default to
`pipeline`, mapping the attention score/softmax work and value-apply work onto
separate herds connected by an AIR channel.

## Mapping Notes

Q4NX maps one 32x256 dequantization block per CT. Packed weights and the BF16
scale/min parameter pack are packed into one L3 byte buffer, staged to L2, then
split into contiguous L1 buffers before calling the Peano microkernel. All
shapes use the L2-gather output route by default; direct output is limited to
2x4 and 4x4 because 8x4 direct output exhausts shim DMA channels.

BF16 tiled MM reuses the optimized AIE2P GEMM path. Matrix rows scale with herd
rows and matrix columns scale with herd columns. The wrapper keeps the GEMM
L3/L2/L1 tiling, Peano microkernel, and runtime tiling behavior from the
reference programming example.

FusedDQP smoke mode maps one 32x256 projection row block per CT. The wrapper
packs weights, scale, min, and the activation vector into one L3 byte buffer,
then splits the payload in L1 before the Peano call. Paper mode keeps the Q4NX
weight block plus scale/min pack per row/column block, stages the activation
vector once per column block, and accumulates across `COL_BLOCKS` using the
`fused_dqp_accum_block_opt` entry point. The optimized kernel processes the
paper's 16x8 row/column sub-blocks internally. FusedDQP `pipeline` mode is a
diagnostic two-herd dequant/project mapping that streams dequantized `16x8`
row/column tiles over an AIR channel. It is compile-legal with debug IR, but
current hardware execution still times out with no active context left after
XRT cleanup. This remains true after reducing to one column block and after
using a larger column chunk to reduce inner channel iterations, so it should not
be reported as a passing physical result.

FlowQKV and FlowKV smoke mode map one attention group per CT. Q, K, and V are
packed as one BF16 input per group, staged through L2, and split into contiguous
Q/K/V L1 buffers before the Peano attention microkernel runs. FlowQKV paper mode
uses tile-shaped Q data and maps CTs to shared GQA K/V groups. The default
`pipeline` staging splits the physical rows into score/softmax and value-apply
stages, with Q/K/V staged through L2 and normalized attention weights crossing
the inter-herd AIR channel in `PAPER_KV_CHUNK` pieces. The `replicated` K/V
staging materializes each CT's selected K/V group in the input buffer before
L2 staging and remains available as a compact diagnostic. The `shared` staging mode keeps source-level shared
per-KV-group K/V inputs as a diagnostic, but it expands to a larger fanout
schedule during lowering. FlowKV paper mode uses the paper's 2x4 /
four-KV-group placement as two 1x4 herds: a score/softmax herd produces BF16
attention weights in `PAPER_KV_CHUNK` pieces, sends those chunks over an AIR
worker-to-worker channel, and an apply herd accumulates the chunks with V to
produce the decode output. The fused Flow attention microkernel still uses
chunked online-softmax state (`m`, `l`, and `Y`) instead of a full score buffer.
The split FlowKV and FlowQKV pipelines now chunk the CT boundary; FlowQKV still
keeps full K in the score stage for correct softmax normalization and copies V
through L1 chunk buffers in the apply stage to avoid duplicate AIE DMA routes.

These examples are meant to be readable, sweepable AIR mappings. They do not use
binary disassembly, generated instruction traces, or model-runtime integration.
The paper modes are experimental kernel-level mappings and should not be reported
as end-to-end FastFlowLM or Gemma3 runtime parity.

## Hardware Acceptance Snapshot

Status as of 2026-05-29 on NPU Strix, XRT 2.21.0, compact defaults
(`KV_LEN=32`, `HEAD_DIM=64`, `OUTPUT_FORMAT=elf`):

| Target set | Result |
| --- | --- |
| 2x4 `run-q4nx`, `run-mm`, `run-fused-dqp`, `run-flowqkv`, `run-flowkv` | PASS |
| 4x4 `run-q4nx`, `run-mm`, `run-fused-dqp`, `run-flowqkv`, `run-flowkv` | PASS |
| 8x4 `run-mm`, `run-flowqkv`, `run-flowkv`, `run-q4nx`, `run-fused-dqp` | PASS |
| `run-q4nx-8x4-rowband-fallback` | PASS as logical row-band fallback |
| `run-fused-dqp-paper`, `run-flowqkv-paper`, `run-flowkv-paper` | PASS |
| `run-fused-dqp-pipeline` | TIMEOUT, diagnostic-only |

The FlowKV 2x4 L2-gather diagnostic and FusedDQP pipeline timeout both
reproduce as `ERT_CMD_STATE_TIMEOUT` with `ctx_pc=0x28B060AD`; a follow-up
`xrt-smi examine -r all` shows no active hardware contexts. The timeout
remains when runtime-loop tiling is removed, when ping-pong buffering is
enabled, when the lock-race fix is
disabled, and when both Peano dequant/project calls are removed from a temporary
diagnostic module. The first bad boundary is therefore the lowered AIE/runtime
execution of the two-herd inter-worker channel schedule, not the FusedDQP
microkernels or the supported L2-gather output route. Keep this mode
diagnostic-only until the channel/lock/DMA schedule is fixed. The focused
reproducer is `test/xrt/56_gemma_fused_dqp_channel_repro`; its lit coverage is
compile-only, while `run.py --compile-mode compile-and-run --output-format elf`
reproduces the timeout on current hardware.
