# 00 — Context and Goals

## What is being ported

`iron` (AMD's IRON Python API over MLIR-AIE) commit `1e014c1`, "Add transformer-layer
execution-strategy studies": 145 files, ~58.6k insertions. It builds a full transformer layer
three different ways on an AMD NPU and measures the cost of each execution boundary.

| iron mode | Paper label | What it does |
|---|---|---|
| `offload` | offload | Host owns the layer; 8 GEMMs are offloaded one dispatch at a time |
| `runlist` | runlist | Fine-grained NPU operator sequence, intermediates moved explicitly |
| `hybrid` | coarse runlist | Runlist orchestration over a few coarse *fused* kernels |

> **`[2026-08-09]` This table describes IRON, and it is not this port's taxonomy.** It is kept
> because it is what the source repository implements and what the paper labels say. The study's
> author corrected the axis on 2026-08-08 to **reconfiguration cost against DRAM traffic**, which
> reverses two things above: `offload` is the mode with the *least* reconfiguration rather than the
> most host-mediated one, and `coarse` is a per-workload **blend** of `runlist` and `fused` rather
> than a point of its own. The CSV keys here are still current; the descriptions are not. For what
> the four modes mean today read [03 §The taxonomy](03-measurement-model.md), and for what is built
> against it read the README's status board.

Around those sit seven measurement studies — `block`, `end_to_end`, `memory_tile_staging`,
`resource_usage`, `host_comparison`, `memcpy_bandwidth`, `roofline` — over a shared case matrix
of two workloads (`encoder_bert`, `decoder_gpt2`), six model families, and a 64..16384 sequence
ladder. An unattended runner drives the whole suite across reboots, and an iGPU ROCm baseline
provides the host comparison.

Size of the source commit, by area:

| Area | Lines added | Files |
|---|---|---|
| `iron/applications/transformer_layer/study` | 36,848 | 62 |
| `iron/operators` | 11,780 | 47 |
| `aie_kernels` (AIE2P C++) | 4,771 | 8 |
| `iron/applications/transformer_layer/pattern` | 3,888 | 13 |
| `iron/common` | 740 | 7 |

## Why it is not a file copy

iron expresses device programs through the **IRON Python API** — `aie.iron` with ObjectFifo,
Worker, Runtime, TensorAccessPattern, SequentialPlacer. MLIR-AIR expresses the same class of
designs through the **AIR dialect**: `air.launch` / `air.segment` / `air.herd`, compiled by the
native `aircc` driver and dispatched through `python/air/backend/xrt.py`.

The two stacks also differ in house style — iron models operators as an `AIE*`-prefixed class
hierarchy, MLIR-AIR as plain `build_*_module()` functions. Porting file-faithfully would import
conventions this repository does not use, leaving the result permanently foreign. See
[02-porting-conventions.md](02-porting-conventions.md).

## Why do it

MLIR-AIR already implements one point on the execution-boundary spectrum — fused multi-launch
ELFs assembled by `stitch_elf` — and ships ten decoder-only LLMs built that way. What it has
never had is anything to compare that choice against, or the instrumentation to make the
comparison quantitative.

Concretely, the repository today has:

- No execution-strategy taxonomy. The fused form is the only form; there is no measured
  `offload` or fine-grained `runlist` baseline.
- No power or energy measurement, no roofline analysis, no memory-tile staging depth, no tile
  resource-usage accounting, no memcpy bandwidth measurement. The existing perf pipeline
  (`Profiler` → `bench/extract_perf.py` → `perf-history`) records TTFT and decode tokens/sec
  and nothing else.
- No unattended, checkpointed, reboot-surviving suite runner.
- No encoder (BERT-style LayerNorm + GeLU) workload — everything is decoder-only
  RMSNorm + SwiGLU.
- No iGPU host baseline wired into the LLM measurement flow, despite `AIRToROCDLPass` and
  `runtime_lib/airgpu` existing on `main`.

## Intended outcome

MLIR-AIR gains a measurable execution-strategy taxonomy for transformer layers, plus the
power / roofline / resource measurement it currently lacks. Two follow-on goals then build on
that harness rather than standing alone:

- **Goal 1 — SOTA models**: sliding-window / local-global attention, unlocking Gemma 3 and
  Mistral-style architectures that every current deployment gate rejects.
  See [11-goal-sota-sliding-window.md](11-goal-sota-sliding-window.md).
- **Goal 2 — quantized inference**: extend beyond the single int4-AWQ Llama example to a second
  model, close the int4 prefill performance gap, and give the kernel registry a quantization
  axis. See [12-goal-quantized-inference.md](12-goal-quantized-inference.md).

## Decisions taken

| Decision | Choice | Rationale |
|---|---|---|
| Branch | `exper/transformer-layer-execution-studies`, cut from `main` | Matches the repository's `exper/*` convention |
| Code location | New `programming_examples/transformer_layer/`, importing `llms/shared/` | Clean base; no 28k-line merge against a diverged branch |
| Plan docs | `docs/plans/transformer-layer-execution-studies/`, excluded from the docs site | AGENTS.md puts human-facing docs in `docs/`; this is in-progress work, not user documentation |
| Case matrix | Staged: iron's matrix first, then the shipped Llama/Qwen/SmolLM2 shapes | The iron matrix is the only way to validate the port against existing results; the shipped shapes are what matters long-term |
| SOTA scope | Sliding-window / local-global attention only | Builds directly on in-flight `exper/gemma3-dataflow` work |

## Success criteria

1. All four execution modes produce numerically identical transformer-layer output against a
   shared torch reference, and their dispatch vectors differ as the taxonomy predicts.
2. A measurement suite runs unattended to completion and emits a manifest with no missing files
   or rows.
3. Results are comparable — through an explicit adapter — to the iron result trees, so the port
   can be validated rather than merely run. `[2026-08-20]` **Met, narrowly and on purpose**:
   `study/iron_adapter.py` joins this port's roots to iron's on identity and validates SHAPE
   agreement per shared point (0 disagreements over four roots against iron's 162-row full
   suite); it refuses to compare latency, power, dispatch counts or `run_status`, each for a
   documented reason — see the README status board row and [03](03-measurement-model.md).
4. The ten shipped LLM deployments still pass `make verify` after every shared-infrastructure
   change.
5. No iron-shaped code lands: no `AIE*` operator classes, no `op.py`/`design.py` pairs, no
   `REUSE.toml`, no module materially over the repository's ~800-line norm.

## Non-goals

- Reproducing iron's `aiecc --xclbin-input` incremental xclbin merge. MLIR-AIR reaches
  multi-kernel dispatch differently.
- Porting iron's `compilation.py` artifact DAG, `AIEContext`, or `AIEDeviceManager`.
  `KernelCache` + `aircc` + `XRTBackend` already cover that ground.
- Integrating llama.cpp, ONNX Runtime, or the Ryzen AI OGA flow. External stacks are context,
  not integration targets.
- MoE, MLA, or encoder-decoder architectures. Goal 1 is scoped to sliding-window only.
