# 01 — The original phase plan (superseded)

Consolidated 2026-08-22 from docs 00, 01, 04, 05, 05a, 06, 06a–06d, 07, 07a, 07b, 08, 08a–08e, 09,
10, 11, 12, 13 and 14; their full text is at git tag `pre-cleanup-20260821`. This is the record of
the phase plan that ported iron commit `1e014c1` into MLIR-AIR: the goals and success criteria as
stated, and per phase A–G what was specified, its gate, what landed, and the rules and measurements
each produced. `[2026-08-21]` **Demoted to "the original plan, superseded."** These docs carried the
pre-08-09 "who sequences the work" framing of the four modes, which the corrected taxonomy in
[03](03-measurement-model.md) (reconfiguration cost against DRAM traffic) replaced. Conventions
stay in [02](02-porting-conventions.md), the buffer rules in [05b](05b-phase-b-buffer-rules.md).

## 1. Context and goals (formerly 00)

**What was ported.** iron (AMD's IRON Python API over MLIR-AIE) commit `1e014c1`, "Add
transformer-layer execution-strategy studies": 145 files, ~58.6k insertions. It builds a full
transformer layer three ways on an AMD NPU and measures the cost of each execution boundary.

| iron mode | Paper label | iron's description |
|---|---|---|
| `offload` | offload | Host owns the layer; 8 GEMMs offloaded one dispatch at a time |
| `runlist` | runlist | Fine-grained NPU operator sequence, intermediates moved explicitly |
| `hybrid` | coarse runlist | Runlist orchestration over a few coarse *fused* kernels |

`[2026-08-09]` That table is iron's taxonomy, not this port's. The axis was corrected on 2026-08-08
to reconfiguration cost against DRAM traffic: `offload` is the mode with the *least*
reconfiguration, and `coarse` is a per-workload **blend** of `runlist` and `fused`, not a point of
its own. The CSV keys stay; the descriptions do not — read [03 §The
taxonomy](03-measurement-model.md).

Around the modes sit seven studies — `block`, `end_to_end`, `memory_tile_staging`,
`resource_usage`, `host_comparison`, `memcpy_bandwidth`, `roofline` — over two workloads
(`encoder_bert`, `decoder_gpt2`), six model families and a 64..16384 sequence ladder, driven by an
unattended reboot-surviving runner with an iGPU ROCm baseline.

| Source area | Lines added | Files |
|---|---|---|
| `iron/applications/transformer_layer/study` | 36,848 | 62 |
| `iron/operators` | 11,780 | 47 |
| `aie_kernels` (AIE2P C++) | 4,771 | 8 |
| `iron/applications/transformer_layer/pattern` | 3,888 | 13 |
| `iron/common` | 740 | 7 |

**Why not a file copy.** iron expresses designs through `aie.iron` (ObjectFifo, Worker, Runtime,
TensorAccessPattern, SequentialPlacer) and an `AIE*`-prefixed operator class hierarchy; MLIR-AIR
uses `air.launch`/`air.segment`/`air.herd`, `aircc`, `python/air/backend/xrt.py`, and plain
`build_*_module()` functions. See [02](02-porting-conventions.md).

**Why do it.** MLIR-AIR implemented one point on the spectrum — fused multi-launch ELFs from
`stitch_elf`, shipping ten decoder-only LLMs — with nothing to compare it against: no taxonomy, no
power/energy, roofline, memory-tile staging, tile resource or memcpy-bandwidth measurement (the
perf pipeline `Profiler` → `bench/extract_perf.py` → `perf-history` records TTFT and decode tok/s
only), no unattended checkpointed runner, no encoder (LayerNorm + GeLU) workload, no iGPU baseline
despite `AIRToROCDLPass` and `runtime_lib/airgpu` existing on `main`.

**Decisions taken.**

| Decision | Choice | Rationale |
|---|---|---|
| Branch | `exper/transformer-layer-execution-studies`, cut from `main` | the `exper/*` convention |
| Code location | new `programming_examples/transformer_layer/`, importing `llms/shared/` | no 28k-line merge against a diverged branch |
| Plan docs | `docs/plans/transformer-layer-execution-studies/`, excluded from the docs site | in-progress work, not user docs |
| Case matrix | staged: iron's matrix first, then the shipped Llama/Qwen/SmolLM2 shapes | iron's matrix validates the port; shipped shapes matter long-term |
| SOTA scope | sliding-window / local-global attention only | builds on `exper/gemma3-dataflow` |

**Success criteria, with status.**

1. All four modes produce numerically identical layer output against one torch reference and their
   dispatch vectors differ as the taxonomy predicts — **met** (§7.1).
2. A measurement suite runs unattended to completion with a complete manifest — **met 2026-08-20**
   (§9).
3. Results comparable to iron's trees through an explicit adapter. `[2026-08-20]` **Met, narrowly
   and on purpose**: `study/iron_adapter.py` joins on identity and validates SHAPE agreement per
   shared point (0 disagreements over four roots against iron's 162-row full suite); it refuses to
   compare latency, power, dispatch counts or `run_status`, each for a documented reason (README
   status board; [03](03-measurement-model.md)).
4. The ten shipped LLM deployments still pass `make verify` after every shared-infrastructure
   change — the standing rule in §11.3.
5. No iron-shaped code: no `AIE*` classes, no `op.py`/`design.py` pairs, no `REUSE.toml`, no module
   materially over the ~800-line norm.

**Non-goals.** iron's `aiecc --xclbin-input` incremental xclbin merge; `compilation.py`,
`AIEContext`, `AIEDeviceManager` (covered by `KernelCache` + `aircc` + `XRTBackend`); llama.cpp /
ONNX Runtime / Ryzen AI OGA integration; MoE, MLA, encoder-decoder.

## 2. Port inventory (formerly 01)

Four dispositions: **PORT** (import-path and convention changes only), **ADAPT** (same structure,
different plumbing), **REWRITE** (re-expressed against the AIR device API), **DROP**. Every PORT and
ADAPT item is still subject to [02](02-porting-conventions.md). Totals: PORT ~19,000 lines (Phase
F), ADAPT ~6,500 (B, F), REWRITE ~9,000 (C, E), DROP ~1,500.

| iron artifact | Lines | Disposition | Became |
|---|---|---|---|
| `aie_context.py` BO liveness allocator | ~180 of 246 | ADAPT | BO pooling in `KernelCache` (§4.3) |
| `aie_base.py` dirty-bit sync | ~60 of 338 | ADAPT | only written buffers sync in, only declared outputs sync back |
| `AIEOperatorBase` hierarchy | ~280 | DROP | `build_*_module()` + `KernelCache` |
| `aie_context.py` remainder, `aie_device_manager.py`, `compilation.py` DAG | ~66, 88, 712 | DROP | `air.tools`, `XRTBackend`, `KernelCache` + `aircc` |
| `utils.py` bf16 helpers, `test_utils.py` `run_test()` | 52, 149 | PORT, ADAPT | check `xrt_runner.type_mapper` first |
| `causal_mask` (`op.py` 86) | — | DROP as operator | `causal_mask=` keyword on elementwise-add |
| `qkv_proj`, `addnorm`, `ffn`, `mha_out_proj` (`design.py`) | 561, 382, 1096, 1350 | REWRITE | `transformer_layer/builders/` (§5) |
| `dynamic_gemm` | 1009 / 430 | **DROP** `[2026-08-04]` | C4's registry sweep is the shape-coverage answer taken |
| `gemm/design_batched.py`, `layer_norm/design_weighted.py`, `softmax` +240, `transpose` +109, `elementwise_mul` +118 | 988, 298 | REWRITE where used | `transpose`/`elementwise_mul` were new device work, assigned to E4 |
| `pattern/reference.py` | 172 | DONE (D2) | `transformer_layer/pattern/reference.py`, FP32 from bf16-rounded inputs, `WEIGHT_DRAW_ORDER` preserved |
| `{offload,runlist,hybrid}/reference.py` shims | 8 each | DROP | import the shared reference |
| `offload/op.py` | 689 | REWRITE | host torch + single-GEMM dispatches |
| `runlist/op.py` | 1566 | REWRITE | `[2026-08-05]` iron: **12 kernels, 16 entries** encoder (13/17 decoder), `op.py:851-940`; the 29/42 earlier drafts quoted was wrong and misled E4 |
| `hybrid/op.py` | 709 | REWRITE → `coarse` | `[2026-08-05]` iron: **5 kernels, 5 entries** encoder (6/7 decoder), `op.py:519-584`; "12" summed both variants. Largely already built as `builders/block.py`: 4 sequences, two norm points at 64 dispatches each |
| xclbin incremental merge | — | DROP | multiple ELF modules in one `hw_context` (later corrected, §4.2) |
| study PORT tier (zero `iron` imports) | ~19,000 | PORT | `run_lock.py` 39, `plot_families.py` 132, `npu_runtime_checks.py` 243, `results_manifest.py` 379, `compare_results_roots.py` 511 + test 283, `regenerate_plots.py` 270, `unattended_smoke_job.py` 216, `end_to_end/power.py` 408, `plot_*.py` ~1,700, `cases.py` 752+752+135, `select.py` 798, `roofline/run.py` 1772 + test 720 |
| `end_to_end/run.py`, `host_comparison/run.py` | 1048, 1784 | ADAPT | generic arg parsing / resume; swap reference and join columns |
| `resource_usage/analysis.py`, `run.py` | 299, 1623 | ADAPT | `[2026-08-07]` artifact resolved: `air_project/aie.air.mlir`, all three regexes match unmodified (24 cores / 88 buffers / 17 allocations on a real compile); `study/aircc_artifacts.py` |
| `unattended_reboot.py`, its test, `conftest.py` | 2494, 1790, 172 | ADAPT | later **not ported** (§9) |
| `end_to_end/modes.py`, `block/run.py`, `run_selected_component_aggregates.py`, `memcpy_bandwidth/run.py` | 2336, 1313, 1578, 586 | REWRITE | §8.3 |
| `pytest.ini` (`python_files = test.py` excludes `study/test_*.py`), `REUSE.toml`, `requirements.txt`, result-tree `.gitignore` | — | ADAPT, DROP, ADAPT, PORT | a completed results root is ~2.4 GB |

Of ~37 study modules only **seven** import `iron`. `matplotlib`/`seaborn`/`pandas` are used by ~40
modules and declared in no iron requirements file; `fcntl`/`pwd` make the tier POSIX-only.

## 3. Phase A — AIE2P device kernels (formerly 04)

**Specified.** Port `iron/aie_kernels/aie2p/`: `encoder.cc` 1061 (new; backs `ffn`),
`addnorm_ffn.cc` 931, `addnorm_ffn_addnorm.cc` 936 (near-duplicate, merged behind a `-D` flag per
convention rule 8), `mm.cc` +1463 (`matmul_init_*` zero-then-multiply and `matmul_with_acc_*`
into explicit `pAcc`), `softmax.cc` +68 (two-pass streaming: `init_softmax_scale_buffer`,
`partial_softmax_rows_bf16`, `normalize_softmax_rows_bf16`, `copy_softmax_scale_bf16`),
`layer_norm.cc` +104 (`layer_norm_rows`, `add_layer_norm_rows`), `mha.cc` +170 (causal-mask
helpers `copy_O_tile_rows`, `store_row_value`, `copy_row_values`), `aie_kernel_utils.h` (port only
if no equivalent). Plain `aie_api` kernels, tiling via `-DDIM_M/-DDIM_K/-DDIM_N`, feature macros
`-Dbf16_bf16_ONLY`, `-DROUND_CONV_EVEN`, `-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`,
`-DBUILD_FFN`, `-DGENERATE_MATMUL_WITH_ACC_KERNELS`, `-DGENERATE_MATMUL_INIT_KERNELS`,
`-DOPT_PERF_ENABLED`. Built `--no-xchesscc --no-xbridge --peano`, so `__chess__` branches never run.

**Destination and build path.** `programming_examples/transformer_layer/kernels/` (`.cc` lives
next to its example; `runtime_lib/` holds no AIE core kernels). Compile through
`llms/shared/infra/external_kernels.py` (`_get_aie_include_dir()`: `which aie-opt` →
`MLIR_AIE_INSTALL_DIR` → `my_install/mlir-aie`) with the aie2p Peano flags `-O2 -std=c++20
--target=aie2p-none-unknown-elf -DNDEBUG -I <aie-opt>/../include -D__AIE_API_AIE_ADF_HPP__
-Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body`; one `compile_*` entry per
kernel after `compile_gemm_mm` / `compile_attn_npu2` / `compile_silu_and_mul`.

**Rules.** `[Codex]` Target the GEMM source the LLM path compiles:
`matrix_multiplication/bf16_in_fp32_out/mm_aie2p.cc` (`external_kernels.py:133`), not `bf16/`,
`bf16_in_bf16_out/`, `i8/` or `i16/` — extending the wrong file compiles and is never used. Do not
port iron's `llvm-objcopy --redefine-sym` step: `compile_gemm_mm(sym_suffix=, out_name=)` already
lets two differently-tiled `mm.o` link into one ELF. `.cc` including `.cc` is the house pattern
(`mm_aie2p.cc` includes `zero.cc`). Verify every flag against mlir-aie v1.4.0 rather than
transplanting iron's v1.2.1 string. Extending `mm_aie2p.cc` touches a file the ten shipped models
depend on (§11.3).

**Gate.** Every kernel compiles to a `.o` with Peano and a compile-only `.lit` passes: `ninja
check-programming-examples-transformer-layer`, registered in `programming_examples/CMakeLists.txt`.
**Landed** 2026-08-04 in 18 min (§12.3); the example README is the file-by-file inventory. Three
build traps recorded there: `-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16` must be a command-line
`-D` (a source `#define` is too late for `<aie_api/aie.hpp>`); `-DDEBUG_AIE_KERNELS` needs a value
(`=0`/`=1`); `encoder.cc` and `addnorm_ffn.cc` emit an empty-but-valid object without
`-DBUILD_FFN`/`-DBUILD_ADDNORM`, so `compile_kernels.py` enforces a size floor and a per-object
symbol list; `encoder.o` and `addnorm_ffn.o` cannot co-link (both define `ffn_gelu_bf16` and
`ffn_eltwise_add_bf16_vector`).

## 4. Phase B — runtime seam (formerly 05) and its spike result (formerly 05a)

### 4.1 What 05 specified

Two host-side additions: runlist aggregation and BO liveness pooling, plus dirty-bit sync and the
dispatch-vector instrumentation of [03](03-measurement-model.md). `XRTBackend.load()`
(`python/air/backend/xrt.py:504`) creates one `hw_context` per artifact, and a `pyxrt.runlist` is
constructed against one context. 05 proposed binding further ELFs into the first context via
`pyxrt.module(mod, ctx)` and `pyxrt.ext.kernel(ctx, mod, name)` — flagged `[Codex]` as API-shape
evidence only (the repository's sole runlist use,
`test/xrt/24_ctrlpkt_config_2gemms_4x4/test.cpp:216`, uses one context and kernel). *"If this
fails, `runlist` and `coarse` collapse into `offload` and the study loses its central axis."* Rules
that stand: implement aggregation in `KernelCache`, not only `XRTBackend.load()` (`load_and_run()`
bypasses `load()`, `shared/infra/cache.py:511`); keep the ELF ABI (buffers from index 0) distinct
from the xclbin ABI (opcode 0, instruction BO 1, length 2, buffers from 3); one timing scope per
runlist; a failed run attributable to its entry. Dropped (rule 2): `compilation.py` 712,
`AIEContext` 246, `AIEDeviceManager` 88. **Gate**: a hardware test on the exact separately-compiled
study artifacts showing a multi-ELF runlist numerically identical to sequential dispatch and
measurably faster. `KernelCache` serializes on `filelock.FileLock("/tmp/npu.lock")`, deliberately a
different inode from `/tmp/mlir-air-npu.lock` — never unify them.

### 4.2 Spike result: the multi-ELF runlist works, N contexts and one runlist

Measured on NPU2 (`amdxdna` 2.21.0_20260514, firmware 1.1.2.64), XRT 2.21.0 hash
`4eb1f4392a012b4e6eca759762389c612537f7c7`, separately-compiled AIR GEMM ELFs from
`shared/builders/gemm_builder.py` at registry tiles. Reproduce: `make runlist-gate` in
`programming_examples/transformer_layer/`. The first pass tested only the one-context shape, found
every route fails, and wrongly concluded the multi-ELF runlist was impossible; §4.2.2 is the
correction and leg A of `make runlist-gate` is the standing measurement.

**4.2.1 One context, several ELFs: rejected three ways.** (1) `xrt::module(parent, hwctx)` throws
`Invalid instruction buffer size` for ELF- and xclbin-created contexts alike, even for the ELF the
context came from — AIR ELFs are `is_full_elf() == true`, carrying `.pdi.*` configuration and
per-kernel `.ctrltext.N` in COMDAT groups. (2) `ext::kernel(ctx, module, name)` requires an XCLBIN
context and a non-full-ELF module; `ext::kernel(ctx, name)` requires a full-ELF context —
*"xrt::hw_context passed is not created using XCLBIN"* / *"... not created using full ELF"* /
*"Unable to find group idx for given kernel"*. (3) `hw_context::add_config(const xrt::elf&)` (not
bound in pyxrt, tested from C++): `add_config(elf_a)` OK, `add_config(elf_b)` → `kernel already
exists, cannot use this ELF with this hw ctx`, and `add_config(elf_a)` twice fails identically —
retested with every user-visible symbol distinct (entry, AIR segment, `_Z4main...` arity, 3-arg
drain vs 4-arg fused-cast). One config ELF per context. (4) Sharing one **xclbin** context across
artifacts runs and is **silently wrong** — configuration comes from the xclbin, not the instruction
stream: `xgemm_512x512x512: runlist == sequential -> True`, `xgemm_1024x1024x1024 -> False`. No
exception, wrong numbers. Hence `plan_submissions` splits at every artifact change under the xclbin
ABI and raises `RunlistSplitError`.

**4.2.2 N contexts, one runlist.** A runlist is constructed against a context but each entry is
dispatched on the context its kernel came from. Measured on three separately-compiled fused-cast
GEMM ELFs at the Llama-3.2-1B seq 2048 projection shapes (2048×2048×2048, 2048×2048×8192,
2048×8192×2048): bit-identical to sequential dispatch for every entry with outputs pre-filled
`0xA5`; independent of order (`ABC`, `CBA`, `BAC`); independent of hosting context (A's, B's or
C's, including a context whose ELF is not among the entries' first); repeatable (one runlist object
executed four times). Three full-ELF contexts live at once is fine. Latency, interleaved medians:

| Entries | sequential | runlist | ratio | runlist wins |
|---|---|---|---|---|
| 3 distinct ELFs (q, gate, down), 25 pairs | 20.826 ms | 19.949 ms | 1.044× | 25/25 |
| 3 distinct ELFs, gate run, 15 pairs | 20.236 ms | 19.770 ms | 1.024× | 15/15 |
| 2 entries on one ELF (gate + up), 15 pairs | 19.591 ms | 19.056 ms | 1.028× | 14/15 |
| 512×512×512 GEMM ×2, one ELF | 0.888 ms | 0.774 ms | 1.15× | |

The saving is one host submission, of order 100 µs — large relative to small kernels, small
relative to large ones, which is the axis the study measures. **Concurrent `hw_context` ceiling:
32** `[2026-08-05]`, probed by holding every context open until XRT refused; 33 fails with
`RuntimeError: DRM_IOCTL_AMDXDNA_CREATE_HWCTX IOCTL failed (err=-2)` — loud, at load time, in
`ensure_loaded`. The probe cycled 4 distinct ELFs, so it bounds concurrent *contexts*, not 29
distinct ELFs; three spare is thin; re-probe with real artifacts before relying on it. Routes that
remain relevant: control-packet reconfiguration between entries
(`test/xrt/24_ctrlpkt_config_2gemms_4x4`; aiecc's `--generate-ctrl-pkt-overlay`,
`--ctrlpkt-elf-name`, `--load-pdi-to-ctrl-pkt`, none driven by `XRTBackend`/`aircc`) — the only way
to put several designs through one context; and one ELF holding several `aie.device`s switched by
`load_pdi` (the `fused_elf` point from another direction). The xclbin merge stays a non-goal.

**4.2.3 Unrelated defect: the standalone drain GEMM ELF is single-shot.** A `drain`-method GEMM
compiled to its own ELF (registry shape 2048×2048×512) is correct on the first invocation after
load and wrong after: `call 0: 100.00%` of elements within `rtol=1.6e-2 atol=1.5e-3`, `call 1:
10.63%`, `call 2: 10.52%`, `new bo_key k0: 10.68%`. Not buffer reuse — a fresh BO set is equally
wrong. `fused-cast` at the same seq_len is correct across arbitrarily many calls; the shipped
deployments reach drain GEMMs only inside fused multi-launch ELFs. `offload` cannot use standalone
drain ELFs until fixed; the registry's `best.high = drain` shapes are affected; leg D re-measures
it on every run. Separately: omitting `runtime_loop_tiling_sizes=[2, 2]` and `stack_size=2048`
from a standalone GEMM ELF's backend kwargs compiles, loads, runs and returns different numbers on
every call; the example runner and `O_FFN_BACKEND` both set them.

**4.2.4 What Phase B shipped.** `shared/infra/dispatch.py` aggregates a dispatch sequence into
runlists: one submission under the ELF ABI whatever artifacts it spans; split and raise
`RunlistSplitError` under the xclbin ABI. Each entry's run is built from its own artifact's kernel.
The dispatch vector records the true `host_submissions_per_layer`. BO pooling, dirty-bit sync and
the vector are complete; rules in [05b](05b-phase-b-buffer-rules.md). Measured on the five-GEMM
layer with B operands static: 18 declared buffers on 15 pool slots, the layer as **one**
submission, second dispatch bit-identical while moving **117 MB less** — 10 sync boundaries
instead of 15 — which depends on pools keyed by sequence value, not plan object identity (O5).
`make runlist-gate` legs: **A** cross-artifact runlist, **A2** xclbin refusal, **B**
within-artifact aggregation, **C** the whole layer through `KernelCache.run_sequence` in one
submission, **D** the drain defect. `run_npu2_runlist_gate.lit` runs it inside the suite (§12.2).
Phase B cost 362 min, 3,725 lines (§12.3).

### 4.3 Dirty-bit sync and BO pooling (specified)

Only written buffers sync to device, only declared outputs sync back — without it latency is not
comparable to iron's. Pooling (from `aie_context.py:44-224`): live ranges over the dispatch
sequence, 4 KiB-rounded size bins, conflict marking, content-keyed static pool. `[Codex]` Not a
drop-in over `KernelCache`'s name-and-size reuse (`cache.py:321`, `:464`: pool outputs are
zero-copy views overwritten by the next call); cross-kernel reuse must respect XRT context, memory
group/bank, alignment, argument type, in-place writes and host-view lifetime, and subsume
`static_input_indices`, `intermediate_indices`, `shared_nonstatic`.

## 5. Phase C — operators (formerly 06, 06a–06d)

Re-express iron's operators as AIR builders, validate on NPU2 against an FP32 reference, register
every `(kernel, shape)` in `kernel_registry`. Split four ways because Phase B was 3,725 lines in
362 min with blocking findings open after round 3, and C's source was 8,160 lines across five
rewrites. Each `<op>/{op,design}.py` pair collapses into one `build_<name>_module(...)` (rules 1,
3, 4); builders in `programming_examples/transformer_layer/builders/` **call**
`llms/shared/builders/` without modifying it (modifying it triggers §11.3, which made Phase B six
hours). `mha_out_proj` (1350) and `ffn` (1096) split along staging seams (rule 5).

### 5.1 The numerics standard — iron's is not ported

`[Amended 2026-08-04]` iron's `reference.py` oracles do not "port verbatim":

| | iron | this port |
|---|---|---|
| Reference dtype | bf16 (`torch.rand(..., dtype=bfloat16)`, bf16 matmul) | FP32 from bf16-rounded inputs |
| Tolerance | `REL_TOL=4e-2`, `ABS_TOL=1.5e-1` (`block/run.py:66-72`) | registry `rtol`/`atol` |
| Mismatch budget | `ERROR_THRESHOLD=0.005` (0.5%) | zero |

`[2026-08-05]` iron's end-to-end gate is looser still: `modes.py:511-566` uses `FINAL_REL_TOL=0.1`,
`FINAL_ABS_TOL=0.5`, a 5% budget against bf16, only at `seq_len <= 512`
(`REFERENCE_VALIDATION_MAX_SEQ_LEN=512`); above that only finiteness. Two further traps: iron's FFN
oracle is exact-erf `torch.nn.functional.gelu` while the kernel is `gelu_tanh_approx_bf16`
(`kernels/elementwise.cc`) — `[2026-08-05]` D2 measured the worst difference over `[-6, 6]` at
**4.7e-4** at `x = 2.70`, inside `rtol = 1.6e-2`, so no gate would notice the wrong form and
`pattern/test_reference.py` is what pins it; iron's MHA oracle is bf16 SDPA below `seq_len 16384`
and FP32 chunked at and above — compute chunked FP32 at every length. Registry methodology: hold
`rtol = 1.6e-2`, size `atol` to the measured worst-case absolute error.

### 5.2 Shape coverage

`registry_lookup.gemm_config()` **raises** on an unmeasured `(M, K, N)`. Registry: 40 shapes (33
bf16-out + 7 f32-out) when written; `[2026-08-05]` **76** (69 + 7) after C4. `[Amended 2026-08-04]`
iron's matrix is 3 distinct families (hidden ∈ {512, 768, 1024}, ffn ∈ {2048, 3072, 4096},
`head_dim = 64`; decoders are `dataclasses.replace()` clones), `BLOCK_KINDS` 7 not 8 (`causal_mask`
has 0 rows in iron's full suite), `block/results.csv` 486 rows pruned by `removed_cases.csv`.
Distinct projection-GEMM triples: **108** (`qkv_proj`, `ffn_up`, `ffn_down`, `o_proj`, 27 each);
5 registered, 103 missing; C4 registered the 36 `baseline_768`. Attention GEMMs go through
FlashAttention and need no row — **except in `offload`** (§7.6).

### 5.3 Gate (all sub-phases)

1. **Numerics** — full-output `np.isclose` vs FP32 at registry `rtol`/`atol`, zero mismatches (not
   cosine: blind to a systematic per-element scale error). 2. **Registration** — rows in
   `kernel_registry/supported_kernels.md` and `details/<Kernel>_bf16.md` with `mean_rel_L1`, `Used
   by`, status. 3. **Coverage** — every claimed shape registered or covered by an explicitly
   injected, recorded spec. Plus the driver's **negative control**: `--fault-inject input` must FAIL.
   Gate command: `flock -x -w 1800 /tmp/mlir-air-npu.lock ninja -C build-xrt
   check-programming-examples-transformer-layer`; never take `/tmp/npu.lock`; modules under ~800
   lines.

### 5.4 Three L3 input streams per tile miscompile in a multi-trip herd loop

`[C1, 2026-08-04]` A herd streaming three distinct L3 buffers into L1 and looping more than once is
wrong. `fused_add_layer_norm_2outs` at `[8, 64]`, `herd_x = 1`: one trip exact (0 of 512 outside
`rtol = 1.6e-2, atol = 5e-2`), two trips 491 of 512. Unchanged by hoisting the invariant input,
draining the second output, `omit_pingpong="L1"`/`"all"`, or either `use_lock_race_condition_fix`.
An AIE2P column has two shim MM2S channels; two-stream builders (multi-row `layer_norm`,
`_build_add_2d_to_2d` in `o_ffn_multi.py`) loop correctly. `build_addnorm_module` raises rather
than emitting the form; C1's workaround is `rows == herd_x * rows_per_call` (64 rows at `cols = 512`
over the full herd). Candidate fixes, neither attempted: stage the invariant operand through L2;
fold two operands into one L3 buffer with an offset memref. `[2026-08-05]` D1 derived the real caps
(`addnorm_max_rows()`: 120 at 512; 104 pre-add / 80 post-add at 768) and D2 showed the cost: of the
block's 131 runlist entries, **128 are the two normalization points** at 64 dispatches each; of 402
sync boundaries, 386 — `coarse`'s vector measures `addnorm` row blocking, not GEMM cost. Its
eventual resolution is the norm-tail work in [16](16-compiler-changes.md).

### 5.5 C1 — the check mechanism and the small operators (formerly 06a)

Step 0: `run_npu2_runlist_gate.lit` now runs `make runlist-gate` on hardware inside the suite and
the nine `.o` files committed by `bf69ed69` are untracked. **`opcheck.py`** is the single entry for
every operator check; it wraps `XRTRunner.run_test` (`xrt_runner.py:165`, `_check_outputs` `:394`:
`np.isclose` over the full output, `max_mismatch_percentage` 0, bf16 upcast to float64,
`report_precision=True` printing `mean_rel_L1 | rel_err max | abs_err max | rtol atol`).
`weighted_rms_norm`, `eltwise_add`, `flash_attention` gate at registry tolerances; `layer_norm`
(`rtol=5e-2, atol=5e-1`), `ffn_swiglu` (`rtol=1e0`) and the fused builders (`0.2 / 0.5` + a
correlation threshold) do not. CLI: `opcheck.py --list` (JSON `[{operator, shape_key}]`, no NPU);
`--operator <op> [--shape-key <k>]` exit 0 iff passed; `--fault-inject input` perturbs one element
of one DEVICE input after the reference is computed and MUST exit non-zero. Results artifact
`results/<operator>__<shape_key>.json`: `operator`, `shape_key`, `shape`, `rtol`, `atol`,
`ref_dtype` (must be `"float32"`), `mean_rel_L1`, `rel_err_max`, `abs_err_max`, `n_elements`,
`n_mismatch`, `passed`, `fault_injected`. The driver requires files newer than the gate stamp,
re-derives the verdict (`n_mismatch == 0`, `ref_dtype == "float32"`, `rtol == 1.6e-2` exactly,
`atol <= 1e-1` — FlashAttention's, the loosest in the registry), and runs injection per operator.
One `run_npu2_<op>_peano.lit` per operator, modelled on
`weighted_rms_norm/run_makefile_peano_multi_tile.lit`, gating on `CHECK: PASS!`; enrolment is
path-based (`CMakeLists.txt:170`, `--filter "transformer_layer/"`); lit scans the whole file, so a
directive name in prose leaves the test UNRESOLVED. Operators: `causal_mask` as a keyword over
`_build_add_2d_to_2d` (`o_ffn_multi.py:66`), mask fill `-10000.0` not `-inf` (bf16 NaN);
`addnorm` with weights as runtime memref args (iron bakes them via `np.load()` and hashes them into
the artifact name) — `add_layer_norm_rows` (`layer_norm.cc:182`) is unweighted, the weighted forms
are `fused_add_layer_norm_1outs/2outs`, `ln_mul_weights_1outs` (`compile_kernels.py:199-278`); the
kernel's one-pass `E[x²] − E[x]²` must not be the oracle for the two-pass form; `cols` must be a
multiple of 16; multi-row `layer_norm` and 2-D `elementwise_add` (`rtol=1.6e-2, atol=5e-2`). Cost
61 min.

### 5.6 C2 — `qkv_proj` and `ffn` (formerly 06b)

`qkv_proj`: `A(M,K) @ B(K,3K)` with C split three ways at the runtime-sequence level; analogue
`rms_gemms_rope_multi.py:191` (6 launches, 13 memref args). Tiles from `gemm_registry_config`
(`gemm_builder.py:29`), never a constant; 5 of 108 shapes registered — record unreachable ones,
never guess; the `gemm_spec_fn` injection hook (`rms_qkv_qknorm_rope_multi.py:441`, `qwen3_4b`)
is the precedent, recorded in the artifact. `ffn`: GeLU-shaped staged up → GeLU → down with
`down_proj_depth`; `ffn_swiglu/prefill/` is the Makefile model (not its `rtol=1e0, atol=0.5`);
`ffn_gelu_bf16` already exists; tanh approximation in the reference. Cost 45 min.

### 5.7 C3 — `mha_out_proj` (formerly 06c)

Composed from `flash_attention/kernel_fusion_based/` (`attn_npu2.py` heads-first,
`attn_npu2_seqfirst.py`, bit-identical; `rtol=1.6e-2, atol=1e-1`; `-Dlqp -Dlkp -Ddk -Ddk_full -Ddv
-Ddv_full`, `lqp_tile = lqp / num_q_tiles`) and the O-projection half of `o_ffn_multi.py`, neither
modified. Causal masking via `-DCAUSAL_ROW_HELPERS`. **`copy_O_tile_rows` is numerically a no-op
and must not be deleted** — removing it hangs with `ERT_CMD_STATE_TIMEOUT`. Reference: chunked FP32
at every length; layout `(heads, lq, dv_chunks, lkp) → (0, 2, 1, 3)` (`attn_npu2.py:1350-1355`).
FlashAttention's `mean_rel_L1 ≈ 3.9e-2` is ~4× the GEMM tier, `atol = 1e-1` is the registry
ceiling; `head_dim = 128` has been flaky (hang or NaN) but the matrix is 64 throughout. Cost 68
min.

### 5.8 C4 — the coverage sweep (formerly 06d)

`sweep/registry_sweep.py`: per `(operator, shape)` build each candidate, check through
`opcheck.py`, time it, record the fastest passing candidate. Carried from iron's `block/run.py`:
resume keyed on shape plus a config signature, subprocess isolation per candidate, turbo
enforcement (fail, not warn), per-case checkpointing (`PL_STEP_TIMEOUT` is 3 hours). Rows into
`kernel_registry/details/GEMM_bf16_in_bf16_out.json` (schema: `M, K, N, used_by, methods{<method>:
{tile{tile_m, tile_k_l2, tile_k_l1, tile_n}, gflops, mean_rel_L1, tier}}, best{high, low}`),
mirrored into the `.md` and `supported_kernels.md`; `tile_m` is dictated by method (drain 32,
fused-cast 64, `gemm_builder.py:21-26`); `registry_lookup` scans `data["shapes"]` linearly (fine at
33). **Append only** — the pre-existing 40 shapes must be byte-identical after. The tamper check
fingerprints `details/*.json` but not the markdown. Acceptance: `baseline_768` over the 9-point
ladder `{64 … 16384}` — `qkv_proj (seq,768,2304)`, `ffn_up (seq,768,3072)`, `ffn_down
(seq,3072,768)`, `o_proj (seq,768,768)`, 36 shapes, bf16-out registry 33 → 69; unplaceable shapes
recorded as ❌ rows. `512`/`1024` families deferred (same tool, `--family`; Phase F's matrix needs
them). Gate: the suite plus `make verify` on all ten shipped models (`gate-c4.sh`); driver: 36
resolve, 40 unchanged, JSON newer than the stamp (the last was unsatisfiable — §12.2). Cost 504 min
+ 66 min gate re-run; found a GEMM config returning zeros for 2 of 9 sub-tiles of each cast worker
while resolving from the registry (§6.2).

### 5.9 Risks recorded

`mha_out_proj` depends on FlashAttention behaviour Goal 1 would modify; the sweep must be resumable;
touching `shared/builders/` requires the ten-model verify.

### 5.10 Phase C outcome

All four sub-phases passed first time, 10 of 40 invocations, ~12 h; C1–C3 averaged 58 min. Landed:
`builders/` (`elementwise_add` with `causal_mask=`, `layer_norm`, `addnorm`, `qkv_proj`, `gelu`,
`ffn`, `mha_attention`, `o_proj`, `mha_out_proj`, `gemm_spec.py`), `opcheck.py`, `sweep/`
(`registry_sweep.py`, `registry_writer.py`), one `run_npu2_<op>_peano.lit` each plus
`run_npu2_fault_control_peano.lit`.

## 6. Phase D — single-block integration (formerly 07, 07a, 07b)

One complete layer through the real runtime path before four strategies are built on it;
per-operator checks do not exercise launch argument maps, layout transitions, external-kernel
linking across a sequence, BO reuse under the allocator, or multi-launch assembly (mirrors
`phase-2-single-block-validation`). `[2026-08-05]` **Done.** D1 in 11 min, D2 in 156, 21 of 40
invocations. One `encoder_bert` layer at `baseline_768`, `seq = 4096`, matches an FP32 oracle over
the whole 4096×768 output with zero mismatches and localizes to ten per-boundary intermediates.

### 6.1 D1 — the operators at `baseline_768` (formerly 07a)

`baseline_768` (`FAMILY_SPECS`: hidden 768, ffn 3072, 12 heads, head_dim 64, `encoder_bert`) is
the only family whose GEMMs resolve (36 of 36, against 2 and 3 for `tinybert_512` and
`baseline_1024`). **`seq = 4096` is forced**: `build_ffn_module` stitches up- and down-projection
into one ELF, two same-method GEMMs with different `tile_n` redefine `f32_to_bf16_mn_<suffix>`:

| seq | up-proj (N=3072) | down-proj (N=768) | |
|---|---|---|---|
| 64 … 2048 | `drain` t_n=128 | `drain` t_n=96 | collide |
| **4096** | `drain` t_n=128 | `fused-cast` t_n=96 | builds |
| 8192, 16384 | `fused-cast` t_n=128 | `fused-cast` t_n=96 | collide |

The fix (suffix per `(method, tile_n)` in `gemm_builder.py`) was off limits to C and D. Shapes
added to `opcheck_specs.py`: `qkv_proj 4096x768` (`64x768` does not satisfy; pins `fused-cast`),
`ffn 4096x768x3072`, `mha_out_proj 4096x768x12h` non-causal (o_proj `4096x768x768`), `addnorm`
pre-add at `cols 768`, `layer_norm` and `elementwise_add` at `cols 768` (rows derived by the
builder); `causal_mask` exempt (`seq × seq`, unused by `encoder_bert`). No explicit
`herd_m`/`herd_n`. **The pre-add gap**: `addnorm_reference` (`builders/addnorm.py:272`) computes
`LayerNorm(x) * weight + residual`; `encoder_bert` is post-norm, `LayerNorm(x + residual) *
weight`. `compile_addnorm_ffn(pre_add=True)` (`external_kernels.py:376`) and
`check_pre_add_variants_differ` existed; no builder exposed it — `encoder.cc` has no `pre_add`
flag, the form lives in `addnorm_ffn.cc`. D1 added `pre_add=` and a distinct reference. Tolerance
ceiling: `rtol 1.6e-2`, `atol <= 1e-1`, re-derived; report `atol_required` rather than widen
(`mha_out_proj`'s causal rows already sit within a factor of two of the ceiling). Driver
clauses: freshness, re-derived verdict, injection per operator, a `baseline_768` shape per operator
read from the `shape` dict, and **every `baseline_768` point gets its own fault injection** (the
generic control takes the first declared shape, `addnorm`'s `64x512` post-add row).

### 6.2 D2 — the block integration gate (formerly 07b)

Configuration: `baseline_768`, hidden 768, ffn 3072, 12 heads, head_dim 64, `seq_len 4096`,
`encoder_bert`, non-causal. **Golden model** at `pattern/reference.py`: iron's structure, FP32 from
bf16-rounded inputs (iron's `dtype="bf16"` everywhere chains eight GEMMs, two LayerNorms and a
softmax in the device's error direction). RNG draw order is load-bearing: `torch.manual_seed(seed)`
then `input`, `q_weight`, `k_weight`, `v_weight`, `attn_output_weight`, `ln1_weight`,
`ffn_up_weight`, `ffn_down_weight`, `ln2_weight`; `ln*_weight` is `torch.rand`, `val_range = 0.05`
scales the `randn` draws; biases are `torch.zeros` (no RNG); `include_output=False` keeps the 16384
rung tractable. Structure (`encoder_bert`, post-norm; `decoder_gpt2` pre-norm causal ported
alongside): `q,k,v = x@W`; `attn = softmax(qkᵀ/√d)v`; `attn_out = concat@Wo`; `hidden =
LayerNorm(attn_out + x, ln1)`; `ffn_out = gelu(hidden@W_up)@W_down`; `output = LayerNorm(ffn_out +
hidden, ln2)`. The layer:

| # | operator | in → out | dispatches |
|---|---|---|---|
| 1 | `qkv_proj` | `x` → `q, k, v` | 1 |
| 2 | `mha_out_proj` | `q, k, v, Wo` → `attn_out` | 1 |
| 3 | `addnorm` pre-add | `attn_out, x, ln1` → `hidden` | **64** |
| 4 | `ffn` | `hidden, W_up, W_down` → `ffn_out` | 1 |
| 5 | `addnorm` pre-add | `ffn_out, hidden, ln2` → `output` | **64** |

`[2026-08-05]` The dispatch column was added after the fact; two claims were falsified by the work:
each normalization point is 64 dispatches, not one launch, and `opcheck.py` did need to change — it
gained an additive `dispatch` seam and `stage_stats`, so such an operator's verdict is the
conjunction of end-to-end and per-boundary comparisons. `layer_norm`/`elementwise_add` are not on
the `encoder_bert` path (validated at 768 for Phase E's finer modes). Dispatch through
`KernelCache.run_sequence` (`runlist_gate.py::leg_c_run_sequence` is the template: `DispatchStep`,
`BufferSpec` with `static=True` + `content_key` for weights, `host_output=True`). Traps: a
multi-`air.launch` design must set `output_format="elf"` (xclbin fails `air.insts.bin produced
duplicate output path`); backend kwargs are not optional and there is no single set — GEMM operators
want `runtime_loop_tiling_sizes=[2, 2]`, `mha_out_proj` wants `omit_pingpong="all"` with `[1, 1]`
(`_GEMM_BACKEND`, `_ADDNORM_BACKEND` in `builders/block.py`); `run_sequence` returns zero-copy pool
views — copy before a second pass. **Per-boundary comparison** is a work item: ≥8 stages with
distinct names, `n_mismatch == 0`, each `n_elements ≥ 4096 × 768` (C4's 2-of-9-sub-tile zeros were
30% of an output and nothing above noticed; an end LayerNorm absorbs upstream damage). Injection
into `w_o` (an input with no averaging operator in front; `x` passes through a softmax). Injected
runs write to `results/fault/`. `opcheck.py --list` imports `XRTRunner`, so it needs the AIR env.
Gate: `run_npu2_block_peano.lit` in the suite (no `/tmp/mlir-air-npu.lock` inside a recipe — BSD
`flock(2)` self-deadlocks; device locking is `/tmp/npu.lock` inside `XRTRunner`); driver: fresh
`block` result at the forced shape, `n_elements = 4096 × 768`, clean stage list, shape keys
`seq_len, emb_dim, ffn_dim, num_heads, head_dim`. Known failure modes: all-zero herd output (bare
herd outside launch/segment); silent GEMM corruption (`N % (tile_n × herd_n) != 0`); correct
standalone wrong chained (BO reuse sync); `ERT_CMD_STATE_TIMEOUT` (`instance_name` ≠ emitted
`func.func @name`); correct first call wrong after (stale pooled buffers); NaN in attention (L1
overflow at large head dim). **Outcome**: the layer's `atol` sits at the `1e-1` ceiling at 1.35×
its measured `atol_required` 7.4e-2 (`mean_rel_L1` 1.7e-2 — output scale, not error); D2 leaked a
6.3 MB `block_cache/` into the tree, as [15](15-environment-notes.md) predicted.

## 7. Phase E — the four execution strategies (formerly 08, 08a–08e)

### 7.1 Outcome: the four modes, measured `[2026-08-05]`

All five sub-phases passed gate, objective and tamper checks — 24 of 60 invocations, ~8.5 hours.
The layer computes identically in all four modes (full 4096×768, zero mismatches, ten clean stages
each):

| mode | submissions | entries | air launches | herd launches | sync boundaries | bytes |
|---|---|---|---|---|---|---|
| `offload` | 6 | 6 | 7 | 19 | 19 | 139,984,896 |
| `runlist` | 5 | 391 | 14 | 404 | 403 | 165,347,328 |
| `coarse` | 4 | 131 | 12 | 146 | 402 | 202,902,528 |
| `fused` | 1 | 3 | 16 | 24 | 19 | 184,025,088 |

`fused` collapses `coarse`'s 402 sync boundaries to 19 and 131 entries to 3 while carrying more AIR
launches (16 vs 12) — the work moves below the runlist into the ELF. Both non-gating predictions
held. **Three of the plan's four predicted vectors were wrong**: [03](03-measurement-model.md)
predicted `offload` 8 submissions / 8 sync boundaries, `runlist` 1 / ~29 entries, `coarse` 1 / ~6;
measured 6/19, 5/391, 4/131 — attention leaving the device for `offload`, a whole-BO dispatch
argument forcing multiple submissions, and `build_addnorm_module`'s 64-row cap. Phase F takes these
from the table. (The `offload` row is the six-GEMM form; it was reversed on 2026-08-08 to 30
dispatches, §7.6 — job 246's totals in §8.4 are the eight-operator form at 1024.)

**One gating clause stopped being a test.** `runlist.runlist_entries > coarse.runlist_entries` is
true by construction: E4's first structure measured 13 entries over 2 runlists (below `coarse`),
review round 2 raised *"The implementation knowingly fails the mandatory E4 ordinal gate"*, and
the fix restructured the mode to 391 by importing `coarse`'s band size (`builders.block.norm_rows`)
and subdividing each unit — defensible (isolates granularity) but a definition. Lessons: a numeric
criterion in a gate description will be optimized for by a reviewer; clause 3 should use a field
neither mode controls by construction — `herd_launches` (404 vs 146). That replacement now lives in
`study/distinguish.py` (§7.9). **`runlist` cannot be one runlist**: 5 submissions, because a host
stage between the projections and the output projection forces at least two, and re-executing one
GEMM ELF inside a single runlist corrupts, forcing per-projection ELFs against the 32-context
ceiling (`pattern/runlist/README.md`).

### 7.2 Sub-phases and decisions

| Sub-phase | Specified to land |
|---|---|
| E1 | the `(method, tile_n)` naming fix, a second ladder point, two over-cap module splits |
| E2 | `coarse` as a strategy directory, and the artifact contract |
| E3 | `offload` |
| E4 | `runlist`, plus `transpose` and `elementwise_mul` |
| E5 | `fused`, and the four-mode distinguishability gate |

Decisions before code: `coarse` wraps `builders/block.py`; `pattern/<mode>/` with a separate
`KernelCache` directory per mode; distinguishability is ordinal over driver-summed totals;
`offload` attention in host torch (reversed 2026-08-08). Corrections to 08's own claims:
`stitch_elf` is `llms/shared/infra/stitching.py:318` and `compile_gemm_mm` is
`external_kernels.py:133`, not `gemm_builder.py`; `gemm_method_spec` has no external callers
(nineteen files import from `gemm_builder`, all via `gemm_registry_config` or `_build_gemm_module`;
`qwen25_0_5b_prefill.py:61` imports it by name and never calls it); a `transpose` example exists in
`data_transfer_transpose/{dma,channel,dma_bf16}/` (no builder or registry row); `elementwise_mul`
exists nowhere. D2's `pattern/` (oracle) vs `builders/` (block) split departs from convention rule
4 because one oracle serves four modes. The block's four vectors (qkv+mha, norm 1, ffn, norm 2):

```
host_submissions  runlist_entries  air_launches  herd_launches  sync_boundaries      bytes
       1                2               6             10              9          80,216,064
       1               64               1             64            193          18,875,904
       1                1               4              8              7          84,934,656
       1               64               1             64            193          18,875,904
```

128 of `coarse`'s 131 entries (98%) are one operator's row blocking; four submissions because a
dispatch argument is a whole BO. Tolerance: `atol` at the `1e-1` ceiling at 1.35× over 7.4e-2.

### 7.3 E1 — unblock the sequence ladder (formerly 08a)

`gemm_builder.py:21-26` minted suffix and object from method alone (`fused-cast` `_m64`/`mm_m64.o`
tile_m 64; `drain` `_m32`/`mm_m32.o` tile_m 32); `gemm_method_spec` (`:61`), `_spec_with_tiles`
(`:45`). Two symptoms: `stitch_elf` redefinition (operand types are functions of `tile_n`,
`bf16_in_bf16_out/run.py:238-244`; `_extract_private_funcs` collects both into one `set()`,
`stitching.py:408-412`); `compile_gemm_mm` writes the same file for two `-DDIM_N` values — D2
measured the FFN up-projection getting the o-projection's 96-wide micro-kernel and returning
**exactly zero for 32 of every 128 output columns** (54% of the layer output wrong), worked around
by interleaving in `block.py:373-429`. Fix: `fused-cast, tile_n 96 → _m64n96 / mm_m64n96.o`,
`drain, tile_n 128 → _m32n128 / mm_m32n128.o`; `tile_n ∈ {32, 64, 96, 128}` so up to eight
objects; `sweep_families.py:107-111` duplicates the table (comment at `:45`); `direct` raises in
`gemm_method_spec` (`gemm_builder.py:101`) deliberately. Behaviour-preserving: every shape resolves
to the same tiles, method and micro-kernel. Also: remove the interleaving workaround; `ffn` at a
second ladder point (`seq = 64`: `64x768x3072` and `64x3072x768` both `drain`, tile_n 128 and 96 —
precisely the removed collision); split `opcheck_specs.py` (1043 lines) and `registry_sweep.py`
(866) along the mechanism/catalogue seam (`opcheck.py:124` imports `SPECS`). No registry rows. Gate:
`flock -x -w 1800 /tmp/mlir-air-npu.lock agents/scripts/port-loop/gate-e1.sh` — lit suite and
`make verify` in all ten `llms/<model>/`; driver: `4096x768x3072` (`drain`, 128) and
`4096x768x768` (`drain`, 96) get distinct `sym_suffix` and `obj`; a fresh non-4096 `ffn` result with
its own injection; D1/D2 re-derived. E1's implement session exhausted `PL_STEP_BUDGET` after six
commits and exposed the implement-halt base-poisoning defect (§12.2).

### 7.4 E2 — `coarse` and the artifact contract (formerly 08b)

`pattern/coarse/coarse.py` wraps `builders/block.py` (`block_config`, `run_block`, `describe_block`,
`BLOCK_BOUNDARIES`) — enrolled in `run_npu2_block_peano.lit`, `opcheck.py --operator block` and the
D1/D2 clauses. **Each mode gets its own `KernelCache` directory**: `block_cache.py` fingerprints
config + MLIR + kernel sources, but the directory is chosen by name (`BLOCK_CACHE_DIR`,
`opcheck_specs.py:550`), so two modes on one directory can trade ELFs and attribute valid output to
the wrong boundary; add it to `.gitignore` and `clean` in the same commit. Contract per mode, via
the `dispatch` seam (`opcheck.py:330`; `_prepare_block` `opcheck_specs.py:553` is the model):
`operator` = mode name; `shape` = `seq_len 4096, emb_dim 768, ffn_dim 3072, num_heads 12, head_dim
64`; `n_elements = 3145728`; ≥8 distinctly-named stages each `n_mismatch == 0` and ≥ one 4096×768
tensor; `rtol 1.6e-2`, `atol ≤ 1e-1`; `ref_dtype "float32"`; `execution_mode` (`"hybrid"` for
`coarse`, mapped in one place); `dispatch_vectors` as `DispatchVector.as_row()` dicts
(`dispatch.py:120`, `:172`). `runlist_entries_per_submission` is a derived mean (`:166`): the
driver reconstructs `Σ round(entries_per_submission × host_submissions)` — `(1, 2.0), (1, 64.0),
(1, 1.0), (1, 64.0)` → 131 — and rejects a non-integral product. The fault-injected run carries the
vectors too and its totals must equal the clean run's (`block` clean and fault both total
4 / 131 / 12 / 146 / 402 / 202,902,528). `run_npu2_coarse_peano.lit` holds both recipes (clean and
`--fault-inject input --expect-failure`) sharing one ELF cache. iron's count is 5 entries over 5
kernels (not "12"). Two full-layer runs (`block`, `coarse`) now live in the suite and each lit test
starts with `make clean` — real minutes per gate, kept so D2 stays provable.

### 7.5 E3 — `offload` (formerly 08c)

Rule: **the mode computes and the oracle checks; they may not share arithmetic.** Importable from
`pattern/reference.py`: `generate_golden_reference`, `fuse_qkv_weight`, `WEIGHT_DRAW_ORDER`,
`ENCODER_BOUNDARIES`; not `chunked_attention_reference`, `gelu_tanh_reference`,
`addnorm_pre_add_reference`. Blocked attention ported (`_blocked_attention`,
`_resolve_query_block_size`; `MAX_ATTENTION_SCRATCH_BUFFER_BYTES = 3 GiB`,
`MIN_BLOCKED_QUERY_BLOCK_SIZE = 256`; at 4096 the score tensor is ~805 MB, at 16384 ~12.9 GB on a
31 GB host) in `pattern/blocked_attention.py`, shared with `runlist`. Each GEMM a one-step
`run_sequence` (`load_and_run` produces no `DispatchVector`), so summed `runlist_entries ==
host_submissions`, at least six of each. Record `attention_path`. Six BOs of weights re-uploaded
per layer is the mode being itself. Noise: ~10× the run-to-run drift of the others; an XRT version
change alone moved it 19–39% at `seq_len >= 4096` with the others within 0.6% —
[03](03-measurement-model.md)'s wider comparator tolerances stand.

### 7.6 `offload`'s attention: six GEMMs, reversed to eight

`[2026-08-05]` At `baseline_768`, 4096, `attn_scores 4096×64×4096` and `attn_output 4096×4096×64`
raise in `gemm_config()` (`registry_lookup.py:115`): no `K = 64`/`N = 64` bf16-out row (minimum K
512, minimum N 128 across 69 shapes), and `sweep_families.py` derives K, N from `FAMILY_HIDDEN ×
ROLE_KN_MULTIPLES` (min hidden 512) so no `--family` can stage one. Decided then: attention in host
torch, six GEMMs (`q_proj k_proj v_proj output_proj up_proj down_proj`), `offload` a *hybrid*
boundary. **`[reversed 2026-08-08]` `offload` dispatches EIGHT linear operators — 30 dispatches —
with both attention matmuls on the device.** The catalogue constraint was misread as a hardware
one: both shapes pass on real NPU2 at every rung, 0% allowed mismatch, `attn_output` by all three
methods, tiles injected via `gemm_spec_fn`; `attn_scores` does not need `tile_k_l2` 256 —
`tile_k_l2 = 64` passes and is forced at K = 64. The numerical argument also fell: with device
attention the layer needs `atol` 5.788e-02, a **1.73×** margin — wider than `block`, `runlist` or
`fused`. Gated in `run_npu2_offload_peano.lit` at 30 dispatches.

### 7.7 E4 — `runlist` (formerly 08d)

Existing operators: GEMM (`gemm_builder` + `matrix_multiplication/`), softmax, GeLU, LayerNorm,
add-and-norm (pre-add, D1), elementwise-add, causal-mask. New device work: `transpose` (from
`data_transfer_transpose/dma_bf16/transpose.cc`) and `elementwise_mul` (`eltwise_add/` shape,
`primitives/vector_examples/vector_mul/` operation; `weighted_rms_norm.py:58` records the unit "does
not legalize f32 vector elementwise mul"). Re-derive the entry count at `baseline_768` (iron's
12/16 is not the target). One runlist over N contexts (1.02–1.15× per §4.2.2) against the 32
ceiling (re-probe). `require_single_submission=True` on `run_sequence` makes "one runlist" a
checked property. Gate adds the `runlist > coarse` entries clause (§7.1 for how it went).

### 7.8 E5 — `fused` (formerly 08e)

A `stitch_elf` wrapper over the whole layer; could not be built before E1 (block specs: `qkv`
fused-cast 96, `ffn_down` fused-cast 96, `ffn_up` drain 128, `o_proj` drain 96 — two `drain` at
different `tile_n`). `execution_mode: "fused_elf"`. `air_launches` is counted once per distinct ELF,
`herd_launches` per step (`dispatch.py:122-153`, `:425-456`), so a fusing mode shows large
`air_launches` on few artifacts. Work items 4–5: a README per strategy directory with its measured
vector, and the four-mode table in the example README. `fused`'s norm tail (§8.2) is the thinnest
margin: `atol_required` 7.896e-2, 1.27×.

### 7.9 The distinguishability criterion (formerly 08e §Distinguishability; now `study/distinguish.py`)

Ordinal over driver-summed totals, never absolute thresholds — a threshold would measure
`build_addnorm_module`'s L1 capacity. The driver prints the four-by-six table whether it passes or
fails. Gating clauses: (1) **distinctness** — no two modes' six-field totals equal; (2) **`offload`
aggregates nothing** — `host_submissions` strictly exceeds every other mode's and `runlist_entries
== host_submissions` (`[2026-08-09]` the clause holds — at 1024 `offload` is 30 submissions vs
`runlist` 17, `coarse` 4, `fused` 1 — but its "host-mediated extreme" name is the superseded
taxonomy: [03](03-measurement-model.md) corrected `offload` to the reconfiguration-minimizing mode,
and since 2026-08-09 it configures the array once per layer,
[25](25-mode-rebuilds-and-results.md)); (3) **`runlist` finer than `coarse`** — originally
`runlist_entries`, replaced by `herd_launches` (J4); (4) **`fused` removes intermediate host sync**
— `fused.sync_boundaries < coarse.sync_boundaries`. Recorded, not gating: `fused.runlist_entries <
coarse.runlist_entries`, `fused.air_launches >= coarse.air_launches`. If it does not separate, the
measurement model is revisited before Phase F consumes it; a mode tuned to satisfy a predicted
inequality would make every downstream measurement meaningless. Implemented first in
`agents/scripts/port-loop/phase_e_checks.py` (27 selftest clauses, no hardware); ported to
`programming_examples/transformer_layer/study/distinguish.py` when the harness retired (§12).

## 8. Phase F — the study harness (formerly 09)

`[2026-08-11]` There is no unmerged Phase F worktree: `exper/phase-f-study-harness` (tip `4775722e`)
is a full ancestor of the experiment branch, 0 unmerged commits.

### 8.1 The seven studies

| Study | iron `run.py` | Depends on |
|---|---|---|
| `block` | 1313 | NPU; per-operator sweep — became the registry sweep (C4) |
| `end_to_end` | 1048 (+ `modes.py` 2336) | NPU; all four modes |
| `memory_tile_staging` | 558 | `block/results.csv` via `--reference-input` |
| `resource_usage` | 1623 (+ `analysis.py` 299) | build tree only; no NPU |
| `host_comparison` | 1784 | ROCm iGPU + the end-to-end CSV |
| `memcpy_bandwidth` | 586 | NPU |
| `roofline` | 1772 | CSVs only; no NPU |

Schema prerequisites ([03](03-measurement-model.md)): a versioned schema with `schema_version` and
field semantics; an explicit iron adapter; `fused_elf` as an `execution_mode` value. Conventions 5,
7, 8, 10, 11 land here (split `modes.py`; one `hybrid`/coarse mapping; route measurement through
`Profiler` and `llms/bench/extract_perf.py`; standard pytest discovery). Environment: pin
`matplotlib` (26 imports), `seaborn` (10), `pandas` (6); `host_comparison` needs
`torch-2.9.1+rocm7.2.1` from `repo.radeon.com`, conflicting with the CPU-only index; POSIX-only.

### 8.2 Carried from Phase E: `fused`'s norm tail and the attention-placement confound

GEMMs are staged correctly in every mode (`bf16_in_bf16_out/run.py:69` keeps an f32 accumulator
across K; `fused.py` derives scratch from `needs_f32_scratch`; `q`/`k`/`v` `mean_rel_L1` 9.7e-3 vs
the registry's 9.3e-3). `fused_tail` decomposes `addnorm` into `elementwise_add` → `layer_norm` →
`elementwise_mul`, rounding to bf16 between launches. Whole-layer `mean_rel_L1`: `block` 1.688e-2,
`runlist` 1.732e-2, `fused` 1.784e-2; `fused` `atol_required` 7.896e-2 (1.27×). `[2026-08-07]`
Refreshed from the J7b gate after J7a moved `layer_norm_rows` to f32 two-pass statistics;
previously `runlist` 1.755e-2, `fused` 1.806e-2 at 7.572e-2 (1.32×) — the mean improved while the
worst-element margin tightened. Cause: `build_addnorm_module` caps a launch at 104 rows of 768 and
`memref.cast` will not cast an offset subview back to the identity layout, so a stitched module
cannot reuse the banded fused operator (compiler work, [16](16-compiler-changes.md)).

| mode | attention | normalization |
|---|---|---|
| `offload` | host torch, blocked (`pattern/blocked_attention.py`) — later device, §7.6 | host torch |
| `runlist` | host torch, blocked, shared | decomposed, banded at 64 rows |
| `coarse` | device FlashAttention (`mha_out_proj`) | fused `addnorm`, banded |
| `fused` | device FlashAttention | decomposed, streamed |

So a mode-versus-mode latency varies more than the boundary it isolates. Rules: never compare modes
on latency alone (surface `attention_path` and per-boundary data); consider a fifth measured point.
`[2026-08-08]` The cheaper measurement fix is convention 10's `Profiler`: `record_kernel` vs
`record_cpu` ("CPU attention fallback"); `pattern/` contained no timing (`grep perf_counter` empty),
the latency was the single `perf_counter` in `study/run_mode.py`. Until per-stage instrumentation
lands, cross-mode latency stays confounded, and `study/ladder_report.py` says so in its docstring.

### 8.3 Retargeting, resource usage, memcpy, power

| Module | `[2026-08-11]` state |
|---|---|
| `end_to_end/modes.py` 2336 | done as `study/run_mode.py` + `opcheck_specs.SPECS` + `pattern/{coarse,offload,runlist,fused}/` (`run_mode._spec_for()`; no module over 1,100) |
| `block/run.py` 1313 | done as `sweep/registry_sweep.py` + `sweep_families.py` / `sweep_measure.py` / `sweep_report.py` / `registry_writer.py`, with a per-candidate timeout iron lacks (iron's FFN sweep never converged) |
| `run_selected_component_aggregates.py` 1578 | done — `study/component_groups.py`, an honest partial |
| `resource_usage/{run,analysis}.py` | done — `study/resource_usage.py` |
| `memcpy_bandwidth/run.py` 586 | **open** — the operator does not exist |

**`study/resource_usage.py`** keeps iron's three regexes (`aie.core`, `aie.buffer`,
`aie.*dma_allocation`; constants `AIE_TILE_LOCAL_MEMORY_BYTES=65536`,
`MEM_TILE_LOCAL_MEMORY_BYTES=524288`, `SHIM_DMA_CHANNELS_PER_DIRECTION=2`) over
`air_project/aie.air.mlir` and adds `core_to_core_flows`. Verified (devq job 238, build class,
`agents/.state/devq/jobs/job-000238.log`): a norm-tail compile reads **24 cores, 40 flows, 16
core→core → space-multiplexed**, matching [23 §5](23-rules-and-open-items.md)'s 2 × `herd_x` at
`herd_x = 8` and the 40/24 counts `aircc_artifacts.py` recorded when item 2 closed; the
`transformer_layer` project reads **0/116 → time-multiplexed**, as does every other artifact. Fixes
an iron defect: `shim_dma_channels_used` keyed on channel number read one channel of four when S2MM
0 and MM2S 0 were both busy; keyed on `(direction, channel)` here. Job 238's compile stopped at the
per-core edge (`aiecc: edge 'chesslinked_{0}.ll' (key 'norm_tail_seg_core_7_4') failed`, the
bare-shell gap) and the counts come from the routed design emitted before it — a routed-design
analysis survives a compile that dies downstream. **`memcpy_bandwidth`**: iron calls `AIEMemCopy`;
MLIR-AIR has none (`passthrough_{dma,channel}.py` are `herd sizes=[1, 1]`); iron's `(size,
num_cores, num_channels, bypass)` axis does not exist — shim channel count is what routing produces
(the norm-tail compile allocates 17 shim channels over 8 tiles unasked). Re-shape: `num_cores` →
herd size over `channel_examples/channel_size`; `bypass` → load/store loop over L1; `num_channels`
→ an observed column. `roofline/run.py` is deliberately unported: `memcpy_bandwidth` is its only
empirical input, so porting it would import iron's 64.3–70.9 GB/s band as a constant.

**Power `[2026-08-08]`** — iron's backend cannot run here and no NPU counter exists:

| path | result |
|---|---|
| `sudo -n turbostat --show PkgWatt` (iron's) | unavailable — password required |
| `turbostat --no-msr` unprivileged | installed (2026.02.14), exits 0, **no samples** |
| `/sys/class/powercap/intel-rapl:0/energy_uj` (`package-0`) | readable; 19.96 W over 2.00 s (`intel-rapl:0:0` core not readable) |
| `/sys/class/hwmon/hwmon10/power1_average`, `PPT` (`amdgpu`) | readable, 22.05 W |
| `amdxdna` sysfs | `power_state` only — no energy or power counter |

A power comparison between modes partly measures host CPU work; "watts per token on the NPU" is not
measurable here, "SoC watts while executing this mode" is.

### 8.4 Work items and what landed

1. Versioned schema — **done**, `study/schema.py`. 2. Resource artifact — **done**,
`study/aircc_artifacts.py`. 3. Port the ~19k tier — `[2026-08-14]` **done except
`memcpy_bandwidth`**; the plot tier landed 2026-08-14 as `study/plots.py` (latency, DRAM traffic, the
v2 decomposition, reconfiguration counters, over `ladder_report.load`); `matplotlib`, `pandas`,
`seaborn`, `pytest` installed in an exclusive window under a full-freeze constraints file,
`pip freeze` diff empty, `make verify` PASS after. 4. Retarget five modules — four of five. 5. pytest
into lit — **done `[2026-08-08]` without pytest**: `run_study_host_tests.lit` +
`study/run_host_tests.py`, plain `test_*` + `assert`; the `--iterations`/`--csv-output`/`metrics`
machinery still unported (port its unbracketed-node-ID fix with it — an `INTERNALERROR` on every
non-parametrized test under `--iterations 1`). 6. Dependencies — `study/requirements.txt`. 7.
`.gitignore` — `results/`, `results_unattended_*/`, **not** `*.csv` (iron's own
`!**/removed_cases.csv`, `!**/*_candidates.json` negations show why); zero tracked files newly
ignored. 8. iron adapter — `study/iron_adapter.py`.

`[2026-08-08]` Portability census over iron's 50 `study/` modules (transitive imports): 13 modules
/ 9,518 lines need plotting libraries directly, 18 / 13,409 blocked transitively, **32 / 22,729
portable**; of the named port tier only `results_manifest.py` (380, superseded),
`regenerate_plots.py` (271) and `roofline/run.py` (1,773) are blocked. Still blocked on the
install: `regenerate_plots.py` (270), `roofline/run.py` (1772) + `test.py` (720),
`plot_selected_component_groups_vs_pattern.py` (621), `plot_dataflow_blocks_vs_pattern.py` (405),
`plot_tps_by_pattern.py` (327), `plot_staging_depth.py` (325), `plot_best_latency.py` (323).
`[2026-08-11]` Ports: `run_lock.py` → `study/run_lock.py` (per OUTPUT file; pointing it at a
device-lock inode deadlocks against the launching wrapper); `plot_families.py` +
`end_to_end/cases.py` (772) → `study/cases.py` (one table; brings `effective_gflops_per_sec`);
`end_to_end/select.py` (217) → `study/select_rows.py` (`select` is stdlib and the directory is on
`sys.path`); `end_to_end/power.py` → `study/power.py`; `compare_results_roots.py` →
`study/compare_roots.py`; `cases.py` group tables → `study/component_groups.py`. Deliberately not
ported: `results_manifest.py` (superseded by `study/manifest.py`), `npu_runtime_checks.py` + test
(superseded by `registry_sweep.py`'s `require_turbo()`/`TurboNotEnforced`, which refuses where iron
warns — a `Default`-pmode latency is ~15–20× off, [32](32-cost-decomposed-ladder.md)),
`unattended_smoke_job.py` (gate half → `study/smoke_gate.py`, runner half → Phase G). `power.py`
keeps the modified-Z filter, interquartile fallback, ≥10/≥6 policy and the rule that a filter which
widens the spread is skipped; two root-free backends `rapl_package` (wrap-safe) and `amdgpu_ppt`;
known limit: when >¾ of samples share a value the MAD and both quartiles are 0 and a lone spike
survives — compare `max_power_w` against `avg_power_w`. `compare_roots.py` closes three holes: the
CSV list is the caller's, rows are read through the schema, no rename exception table; the dispatch
vector is compared as an identifier. Host-suite pin history: 61 → 103 → 133 → 196 → 229 → 231 →
**`[2026-08-12]` `265/265 passed in 19 modules`** (6 → 10 → 17 → 19 modules; the last move was G0's
`test_profiles.py` 15 + `test_run_profile.py` 9 + manifest row-count 7 + ladder-skip 3),
re-verified shrinking (a doctored `264/264` fails the `CHECK`; earlier `228/228` and `60/60`
likewise); ~0.4 s. `iter_cases` carried iron's short-circuit (naming a family made the variant
filter a no-op) — fixed.

**Component aggregate** (`study/component_groups.py`): reports per group components accounted for
against the taxonomy's count, by name, with `is_complete`; named host components exist only for
`offload` (`attention_layout`, `softmax`, `ln1`, `gelu`, `ln2`). First real table — devq job
**246**, `measure` class, `--mode offload --seq 1024 --warmup 1`, `job-000246.log`, Turbo verified:

| group | kind | ms | components | complete |
|---|---|---|---|---|
| GEMMs (NPU) | `device` | 64.388 | 0/8 | no |
| Non-linear operations (host) | `host_cpu` | 10.914 | 5/5 | yes |
| Data sync | `sync` | 4.494 | 0/0 | yes |

Attributed 79.795 ms of a 159.795 ms layer — remainder 80.000 ms, **50.1%** (a denominator, not a
latency). Dispatch totals `submissions 30 entries 30 air 30 herd 90 sync 90 bytes 99090432`,
`context_loads 1 kernel_attaches 4` — `sync 90` and `bytes 99090432` match
[03](03-measurement-model.md)'s steady-state `offload` figures at 1024.
[03](03-measurement-model.md) records the remainder dominating `runlist` at 1024 (device 44 / sync
6.4 / host 0 against ~1959 total). Jobs 242, 243, 245 died before dispatch on `PEANO_INSTALL_DIR`,
`MLIR_AIE_INSTALL_DIR`, and `/opt/xilinx/xrt/python` on `PYTHONPATH` — the last surfaces as
`ModuleNotFoundError` at the first dispatch, minutes in.

### 8.5 Gate: `execution-smoke-test`

≥1 row per measurement CSV with `run_status=passed` — not file existence (iron's smoke test
reported 21/21 on an environment where every measurement failed) — and the first `failure_message`
verbatim. `[2026-08-08]` **Passed on hardware over all four modes** via `run_mode.py` at
`baseline_768` 4096, `--warmup 1 --samples 2`: `smoke_gate` PASS (4 CSVs), `manifest complete:
True`, artifacts in `results/phasef_smoke/`:

| mode | subs | entries | air | herd | sync | bytes | avg ms |
|---|---|---|---|---|---|---|---|
| `coarse` | 4 | 131 | 12 | 146 | 396 | 188,743,680 | 731.6 |
| `offload` | 6 | 6 | 7 | 19 | 19 | 139,984,896 | 606.5 |
| `runlist` | 5 | 391 | 14 | 404 | 395 | 150,994,944 | 787.7 |
| `fused` | 1 | 3 | 16 | 24 | 13 | 157,286,400 | 537.1 |

All four clauses hold including J4's replacement; both non-gating predictions hold. **The latency
column is contaminated and retracted**: taken with builds, a formatter and the test suite running
alongside; `coarse` re-measured at 466.9 ms and 476.9 ms on a quiet host against 731.6 — a 1.55×
inflation. The structural columns stand. Measurement conditions became a rule in
[23](23-rules-and-open-items.md). Risk: `host_comparison` needs an iGPU on the NPU host.

## 9. Phase G — unattended runner and CI (formerly 10)

`[2026-08-12]` The spec was written against iron's design and half was obsolete:
[25](25-mode-rebuilds-and-results.md) (doc 34) found half of iron's 2,494-line
`unattended_reboot.py` exists here in better form, a quarter should be dropped, and the remainder
is small. **Do not port it.** iron's profiles: `full` 888 jobs (11 h to 2 days), `paper` 834 (~20
h; helper studies at seq_len 512/2048/8192 only), `execution-smoke-test` 21, `smoke-test` 3
(measures nothing); wall clock swings 4× on `build/` warmth. `[Codex]` Never hard-code 888/834/21/3
— derive counts from the profile.

**What G0 shipped.** `study/profiles.py` (`smoke`/`ladder`/`full` over the one reachable family,
the five unreachable with reasons, the `fused` 256..1024 applicability rule, `expected_files`/
`expected_rows` derived); `study/run_profile.py` (refuse off-Turbo → `run_lock` → sample power →
walk → `smoke_gate` → `results_manifest.json` + `profile_run.json`; `--dry-run`, `--gate-only`);
`study/manifest.py` (three count clauses per file: total, passed, skipped); `study/run_ladder.py`
(`walk(..., skip_reason=fn)` writes `run_status="skipped"`, which existed since schema v1 and
nothing emitted, reason in `failure_message` prefixed `skipped:`); the `-host` CMake target and the
PR step. Profiles: `smoke` 4 rungs; `ladder` 16 (14 measured, 2 skipped: `fused` at 2048, 4096);
`full` 36 (30 / 6: `fused` at 64, 128, 2048, 4096, 8192, 16384) — `full` is the nine-point ladder
over one family and attempts untried rungs deliberately; `smoke` and `ladder` are the profiles to
gate on. devq job 224: 8 rungs (4 modes × {512, 1024}) cold **631 s**, warm **32 s** (~20×;
per-rung 98/102/29/30/55/57/128/132 s vs 5/5/2/3/6/7/2/2). **Ran end to end — devq job 256**,
`measure` class, Turbo, cold caches, **347 s**: smoke-gate PASS (4 CSVs), `complete: True`, each
CSV `rows 1/1 passed 1/1 skipped 0/0`; `profile_run.json` carries `devq_job_id: 256`,
`tree_dirt_after_run: []`, `power_backend: rapl_package`, 3390 of 3465 samples retained, avg 25.1 W
(min 17.4, max 33.0) — a run condition (mostly aircc), never a results row. Single walk: no latency
quoted (README trap 1: a single walk once published a crossover a second walk refuted).

**Resume** (still open at G0): copy `REUSABLE_STATUSES` (`registry_sweep.py:177`) separating
candidate verdicts from machine verdicts. **Power mode**: `require_turbo()` is the single
implementation (`run_mode.py`, `component_groups.py`, `pmode_guard.py`, `run_profile.py`);
`run_profile` takes it up front, `run_mode` per rung. Two iron fixes re-filed: TTM 1% band (602-page
shift halted iron at job 885 of 888; the 26 GB override was for six 16384-token iGPU jobs —
`host_comparison`, unported); the empty-mask column drop in
`plot_selected_component_groups_vs_pattern` (`KeyError: '_pair_key'`) — plot tier.

**Host prerequisites, measured.** `xrt-smi` needed (reading pmode needs no root; `configure` is the
operator's action); `amd-ttm` unused; `turbostat` cannot run; `sensors` dropped; `rocm-smi` only
`host_comparison`; `crontab` dropped. No script runs `sudo`. **NPU serialization**: superseded by
`agents/scripts/devq.sh` — bare `flock` has no FIFO order (a writer blocked 3197 ms while a later
reader acquired in 4 µs); devq has monotonic sequence numbers, a measure/build barrier, liveness
reconciliation; 23 `flock` sites migrated, `devq-selftest.sh` 20/20. `run_profile.py` takes no
device lock; `run_lock.py` is a third inode. **`run`, never `submit`, from a gate** (`submit`
blanks the gate's FileCheck and exits 0).

**CI wiring** — the proposed block was wrong three ways: the target name already existed since
Phase A; the filter `transformer_layer/.*/run_npu2_compile` matches 0 of 32 tests; "compile-only"
is false — 32 tests, 22 `REQUIRES: ryzen_ai_npu2`, 1 Peano-only, 9 host-only, and on an NPU-less
runner all 22 report UNSUPPORTED with exit 0. Fourth: `check-programming-examples-peano` already
pulls 22 of them (21 NPU-gated) into PR CI on `amd8845hs` (NPU1) and `amdhx370` (NPU2) —
pre-existing, unchanged. Shipped: `check-programming-examples-transformer-layer-host`, an explicit
allowlist of the 10 PR-safe tests (`run_npu2_compile_peano`, `run_block_cache_tests`,
`run_blocked_attention_tests`, `run_ffn_resident_emulation_tests`, `run_reference_tests`,
`run_seam_tests`, `run_study_host_tests`, `run_npu2_registry_resolution`,
`run_sweep_families_tests`, `run_sweep_writer_tests`), `[.]lit$` not `\.lit$` (CMake eats the
backslash). `buildAndTestRyzenAI.yml` asserts **`Passed = 10`** and only `Passed`/`Excluded`
reported — never `Total Discovered Tests: 10` (it is 362; the first guard failed a fully green run,
devq job 258, filtered 10/10 passed). Verified devq job **261**, `job-000261.log`, 19.9 s: `lit
exit=0`, `Excluded=352 Passed=10`, PASS. `-j1` not applied to either target (the hardware target
passed 30/30 in 519.7 s at 24 workers; `run_npu2_runlist_gate.lit`'s latency clause goes
intermittent under contention, [25](25-mode-rebuilds-and-results.md)). Three known-red lit failures
outside both targets: `llms/llama32_1b_int4/.../run_o_gemv_ffn_int4_fused_npu2_peano.lit`,
`conv2d_14x14/run_npu2_makefile_peano.lit`,
`matrix_vector_multiplication/bf16/run_npu2_makefile_peano.lit` (2/2 reproducible, predating the
study). Work item 7: **no measurement workflow** — this laptop is no CI runner; the nightly
(`nightlyPerfBenchmark.yml`, cron `17 4 * * *`) is on `amdryzenai5pro340`; the measurement half is
`systemd-inhibit --what=handle-lid-switch:sleep:idle agents/scripts/devq.sh run --class measure --
python3 study/run_profile.py --profile ... --out-dir ...`.

**Deliberately dropped `[2026-08-12]`**: the `@reboot` crontab hook (`systemd-inhibit ... setsid
nohup` covers lid/idle/logout; removes the reboot-loop class that halted iron at 885 of 888); TTM
transitions; thermal gating (no artifact shows throttling affecting any number); `turbostat` power.

**Gate**: a full profile completes with no missing files or rows, counts from the profile.
`[2026-08-12]` The "rows" half was the missing piece (a CSV holding one of nine rungs reported
`complete: True`). **`[2026-08-20]` MET**: `run_profile --profile full` (36 rungs) reads `complete:
True`, `row_counts_checked: True`, smoke gate PASS — 21 measured, 15 structurally skipped, 0 failed
— walked twice (devq 434 resumed across three sessions; devq 435 from scratch, 419 s warm;
`compare_roots` OK). Ten rungs initially failed in builders before aircc; one wall repaired, the
rest became skips whose reasons are read from the refusing builders (`ast`-pinned;
`run_profile_bounds_tests.lit` asserts skip ⇔ refuses). All six families have walked
([54](54-first-full-profile-and-decoder-families.md)). Open: the README prerequisites/recovery
sections (item 5). Risks that stand: a `full` profile monopolizes the NPU; ~2.4 GB per root;
unattended measurement is where this project published wrong claims (1.55× inflation; a "5.9%
improvement" from three fresh runs against one stale number) — conditions belong in the data, and
the manifest still records git and platform provenance but no measurement condition; `install-xrt`
vs `build-xrt` diverge whenever `mlir/` changes — check with `ls -l`, never `cmp` (RUNPATH
rewrite).

## 10. Goals 1 and 2 (formerly 11, 12)

### 10.1 Goal 1 — SOTA models via sliding-window attention

Outcome: **parked** — see [35](35-goals-1-and-2.md). As specified: unlock sliding-window /
local-global attention (Gemma 3, Mistral). `[Codex]` A new model deployment plus attention and
KV-cache work, not a mask tweak. The ten shipped LLMs (Llama-3.2-1B/3B, SmolLM2-1.7B,
Qwen2.5-0.5B/1.5B/3B, Qwen3-0.6B/1.7B/4B) are all full-causal, ≤4B. Prior art:
`exper/gemma3-dataflow`, ~15 commits, a 182-file parallel tree at `programming_examples/gemma3/`
(own `core/` package, own kernels `q4nx.cc`, `q4nx_opt.cc`, `flow_attention*.cc`, `fused_dqp*.cc`,
own docs, `data/paper_targets.json`) not following the `llms/<model>/` contract — reconcile vs keep
parallel is the first decision. Gates to update: `deploy-new-llm/SKILL.md` Step 2 (rejects
`sliding_window` + `use_sliding_window=true`) and `phase-0-build-cpu-reference/SKILL.md` Step 2's
allowlist (`Gemma3ForCausalLM`), in both `.claude/skills/` and `.codex/skills/`; fix the 15
`SKILL.md` files still naming `llms/llama_kernel_builder/` (renamed `llms/shared/` in `2f20c2fa`;
`deploy-new-llm` Step 3 `test -d`'s it and always reports MISSING). Work: banded masking with
absolute positions and window offsets in `attn_npu2.py`; windowed KV cache (eviction or ring
buffer) with RoPE position correctness in `attn_decode_npu2.py`; alternating local/global layers as
a first-class axis; windowed-FA registry rows; Gemma-3-1B/4B in `llms/hf_models.txt` (feeds
`downloadLLMWeights.yml`, `HF_HUB_OFFLINE=1`); measure across all four modes. **Gate**: `make
verify` exits 0 **with prompts that cross the window boundary** — the standard top-5 token-set
check over 32 tokens from short prompts never reaches the window edge; long-prompt fixtures in
`verify/prompts/` are part of the gate.

### 10.2 Goal 2 — quantized inference

Outcome: **done** — `make verify` passed 2026-08-19 on `smollm2_1_7b_int4` under a gate exercising
the quantized path; see [35](35-goals-1-and-2.md). As specified: `llama32_1b_int4/README.md` was
stale — commit `aa73c0d7` landed NPU int4 decode at ~17.8 tok/s against 12.2 tok/s bf16
Llama-3.2-1B, and it still pointed at `../llama_kernel_builder/`. `[Codex]` `Int4NpuRunner` runs
bf16 prefill on dequantized AWQ weights, int4 only in decode, so the int4 `make verify` did not
validate int4 prefill. Prefill backends behind `--prefill-dtype`: `bf16` (84 ms/layer, 1.38 s
end-to-end at seq 2048), `int4` (698 ms/layer, 11.2 s, same AWQ-quality output), `bfp16` (exists,
`run_npu2_verify_prefill_bfp16.lit`, `Makefile:94`). Leaf support: `matrix_multiplication/{int4_awq,
bf16_x_bfp16, i8, i16}`, `matrix_vector_multiplication/int4_awq`, `vector_matrix_multiplication/{i8,
block_quantized_i8}`, `dequant_awq/`, `decode_ffn_swiglu/matvec_int4_swiglu_rms.py`. Gaps: (1)
conflated gates — split int4 prefill, int4 decode, BFP16 end-to-end; (2) int4 prefill 8× slower —
Down GEMM at K=8192 hits the memtile L2 budget, `herd_m=2` (8 PEs not 32) because
`matmul_int4_packed.py` cannot tile `K_L2 < K`; the AIE2P `VLD_x_pstm_nrm_imm` 9-bit immediate
forces `tile_n=16` vs bf16's 128 (16× more iterations); decode is DMA-bound, where int4's halved
footprint pays; (3) 1 of 10 models quantized — generalize `llama32_1b_int4/multi_launch_builder/`
(`rms_gemms_rope_int4_multi.py`, `o_ffn_int4_multi.py`, `o_gemv_ffn_int4_multi.py`, bfp16 siblings)
before hoisting to `llms/shared/builders/`; target Qwen3-1.7B or Llama-3.2-3B; (4) no registry
quantization axis — add `details/GEMM_int4_awq.{md,json}`, extend `gemm_config(M, K, N,
output_dtype, precision)`; (5) schema needs packing scheme, group size (AWQ g128), scale/zero-point
layout, accumulation type, separate GEMM/GEMV contracts, in schema v1; (6) define the measurement
before a target ("within 2× of bf16" is meaningless unstated). **Gate**: a second quantized model
passes `make verify` under a gate that exercises the quantized path, and int4 prefill per-layer
latency materially closer to bf16 under the defined measurement. Risks: AWQ checkpoints at the same
configuration (uint4 asymmetric, g128, bf16 lm_head) may need local quantization.

## 11. Verification and acceptance (formerly 13)

### 11.1 Environment

`source utils/env_setup.sh <install> <mlir-aie> <llvm-aie> <llvm>`; `source <xrt>/setup.sh`;
`sudo xrt-smi configure --pmode turbo`; `xrt-smi examine -r all` confirms `Power Mode : Turbo`.
Every NPU command under `flock -x -w 1800 /tmp/mlir-air-npu.lock` (since superseded by
`agents/scripts/devq.sh`, §9); `KernelCache` serializes on `/tmp/npu.lock`, a different inode —
do not unify. Prefer incremental `ninja` then the narrowest test.

### 11.2 Gate table

| Level | Command | Gates on |
|---|---|---|
| Conventions | `black --check .`, clang-format/clang-tidy, the [02](02-porting-conventions.md) checklist | no iron-shaped code; no module materially over ~800 lines |
| Kernels compile | `make compile` in `transformer_layer/` | Phase A. `[2026-08-05]` The PR-safe subset is the individual `make` targets: `check-programming-examples-transformer-layer` has needed an NPU since C1 (16 tests then, ten `REQUIRES: ryzen_ai_npu2`; [15](15-environment-notes.md)) |
| Runlist spike | hardware test on the real artifacts | Phase B (§4.2) |
| Operator numerics | `opcheck.py --operator <op>`, one `run_npu2_<op>_peano.lit` each | Phase C — full-output `np.isclose` at registry tolerances vs FP32, zero mismatches |
| Check discriminates | `opcheck.py --operator <op> --fault-inject input` must **fail** | Phase C |
| Registry coverage | rows in `supported_kernels.md` + `details/<Kernel>_bf16.md`, `gemm_config()` resolves | Phase C |
| Operators at the block's width | `opcheck.py` at `baseline_768` | D1, incl. pre-add `addnorm` |
| Single block | `opcheck.py --operator block`, `run_npu2_block_peano.lit` | D2 — full `seq × hidden`, zero mismatches, ten clean stages |
| Golden-model composition | `make reference-tests`, `run_reference_tests.lit` | D2 — pins erf vs tanh, post- vs pre-add, QKV column order |
| The sequence ladder | `gate-e1.sh` — lit suite **and** `make verify` over ten models | E1 |
| Strategy equivalence | `opcheck.py` `dispatch` seam, one `run_npu2_<mode>_peano.lit` per mode | E2–E5, full 4096×768, ≥8 stages; not bare `pytest pattern/` |
| Dispatch-vector provenance | fault run's summed totals equal the clean run's | E2–E5 (`results/` is gitignored) |
| Strategy distinguishability | vectors summed by the driver; now `study/distinguish.py` | E5 — **passed 2026-08-05**; `runlist > coarse` by construction, replaced by `herd_launches` |
| Harness plumbing | `unattended_reboot smoke-test` | Phase F — never ported (§9) |
| End-to-end setup | `execution-smoke-test` → `study/smoke_gate.py` | Phase F — ≥1 `run_status=passed` row per CSV |
| Full suite | `study/run_profile.py --profile full --out-dir <root>` | Phase G — **`[2026-08-20]` passed**, 36 rungs, 21 + 15 skips, walked twice |
| Cross-run sanity | `study/compare_roots.py --baseline <old> --candidate <new> --csv …` | median/p90 drift within per-mode tolerance; same-toolchain only — iron trees via `study/iron_adapter.py --iron-root … --root … --csv …`, SHAPE agreement never drift |
| Sliding window | `make verify` with window-crossing prompts | Goal 1 |
| Quantized path | `make verify` under a quantized gate | Goal 2 — **passed 2026-08-19**, `smollm2_1_7b_int4` |
| LLM regression | `make verify` in each `llms/<model>/` | eleven models since 2026-08-19; `[2026-08-20]` **8/11 is the standing leg** — the three ≥3B are deferred, oomd-killed with the session ([15](15-environment-notes.md)) |

### 11.3 The cross-deployment regression rule

Phase B modified `shared/infra/cache.py`; Phase C may touch `shared/builders/`; Phase A extended
`bf16_in_fp32_out/mm_aie2p.cc`; Goal 2 hoists builders; `[2026-08-05]` E1 joined them
(`gemm_builder.py`), and is the only E sub-phase that did (`gate-e1.sh`). **After any
shared-infrastructure change, re-run `make verify` on every sibling model**, serialized:

```bash
for m in llama32_1b llama32_1b_int4 llama32_3b smollm2_1_7b \
         qwen25_0_5b qwen25_1_5b qwen25_3b qwen3_0_6b qwen3_1_7b qwen3_4b; do
  (cd programming_examples/llms/$m && flock -x -w 1800 /tmp/mlir-air-npu.lock make verify) \
    || echo "REGRESSION: $m"
done
```

The most expensive check and the one most likely to be skipped.

### 11.4 Correctness standards and pre-existing issues

Kernel numerics gate on element-wise `np.isclose` vs FP32 (not cosine); per-layer cosine (`make
diagnosis`) is informational; model correctness gates on top-k token-set inclusion (k=5, first
divergence over 32 tokens) vs HF bf16, mirroring vLLM's `check_logprobs_close`; reductions
accumulate in FP32. Pre-existing issues: 15 `SKILL.md` files naming `llama_kernel_builder/`
(`2f20c2fa`); `verify/README.md` titled for Llama-3.2-1B documenting a removed
`runners/npu_runner.py`; `llama32_1b_int4/README.md` stale; `docs/ai_skills.md` a placeholder.

## 12. The port-loop harness (formerly 14) — retired

`agents/scripts/port-loop.sh` ran phases A through E5 unattended (and the later compiler phases,
[16](16-compiler-changes.md)): a fresh `claude -p` session per phase, three sequential Codex
review→fix rounds, gates run by the driver. **Retired in the 2026-08-21 cleanup** (commit
`cbd2858e`: agents/ 15,816 → 2,783 lines); its full source is at tag `pre-cleanup-20260821`.
`agents/scripts/devq.sh` and `agents/scripts/port-loop/lib-env.sh` survive; the Phase E
distinguishability check was re-homed to `study/distinguish.py`. What follows is the record.

### 12.1 Operation

Commands: `help`, `dry-run`, `status` (`agents/.state/port-loop/STATUS.md`), `start`, `resume`,
`stop`, `run-one <phase> <step>`, `resume-at <phase> <step> [round] [base-sha]`. Detached:
`PL_CODEX_EFFORT=medium systemd-inhibit --what=handle-lid-switch:sleep:idle setsid nohup
./agents/scripts/port-loop.sh resume > agents/.state/port-loop/driver.log 2>&1 &`. Scope in
`PL_PHASES_IN_SCOPE` (`phases.sh`, a hard assignment); a phase is a `case` arm in seven
dispatchers; `phase_objective_check` defaulted to `return 0` (fails **open**), the other three fail
closed. `cmd_loop` reads `.phases` from `state.json`; `resume-at` with an unknown phase resolves to
phase 0 — after changing scope, `start`. `--output-format json` buffers until a session ends. Step
machine: `preflight → implement → commit → [review → fix → commit] × 3 → confirm → gate →
hardware-check → objective-check → tamper-check → advance | halt`. Three rounds because Phase A
round 2 passed and round 3 failed on byte-identical code — rounds are repeated samples of a
non-deterministic detector (Phase B converged 3 → 2 → 1 → clear). `confirm` `[2026-08-05]` reviews
only the last fix's diff (`prompts/confirm.md`), because the last fix was otherwise never reviewed
(C4; D2 round 3's two blocking findings — the block lit gate never invoked `check-block-fault`, and
the golden-model identity tests were enrolled in no lit test — were the first it confirmed; one
codex invocation, about two minutes, and skipped when no fix ran).
`run-one <phase> objective-check` must source the venv (`pl_env_ensure`) or E's naming clause fails
on `No module named 'ml_dtypes'` and reports a live symbol collision. Watch with `tail -f
driver.log`; a bounded `until` wait on `state.json` status delivers a notification, a persistent
`tail -f | grep` did not. `PL_CODEX_EFFORT=medium`: a `medium` review found a bug an `xhigh` missed
in 2.5 min vs 19. Guardrails: sessions under `--permission-mode bypassPermissions`; a run-scoped
`pre-push` fence; snapshot commits; deleted-file detection; `timeout`, `--max-budget-usd`,
`--max-turns`; invocation cap and wall-clock deadline; hardware under `/tmp/mlir-air-npu.lock`,
never `/tmp/npu.lock`. Self-reports (`work_completed`, `work_not_completed`, `blockers`,
`gate_files_touched`) are claims to be checked, never evidence.

### 12.2 Anti-reward-hacking, and how each layer failed first

| Layer | What it does | How it broke |
|---|---|---|
| Gate-file fingerprint | hashes every `.lit`, example `Makefile`, `CMakeLists.txt`, registry JSON and verify module against a per-phase allowlist | baseline taken from the working tree ten hours after the phase's commits — vacuous; now hashed at the phase base via `git show`, absent files `ABSENT_AT_BASE` |
| Objective check | driver-side assertion on build products | accepted any fresh object over 4 KiB; then collected symbols globally so stale objects satisfied it; now every `extern "C"` symbol in every source must be defined by an object this gate rebuilt |
| Codex `weakened_gates` | review reports checks the diff weakened | conflated with inherent limits; split into `weakened_gates` (halts) and `gate_limitations` (recorded) |

Further lessons, each with its incident: **the gate that ran no hardware** `[2026-08-04]` — Phase
B's gate ran 2 tests, 329 excluded, 16 s, and the multi-ELF claim stood on a self-report for a day;
fixed by `run_npu2_runlist_gate.lit` and `pl_assert_gate_ran_hardware` in `lib-guard.sh`
(`needs_hardware=yes` requires a `.lit` whose `REQUIRES` names `ryzen_ai_npu2`, and from the lit
summary that `Passed` and `Excluded` are the only nonzero categories with `Passed` equal to the
tracked file count — no slack: `Passed >= npu_tests` (9 of 13) let one `XFAIL` through; tested
against `phase-B/gate.log`, `Unsupported: 9`, `Expectedly Failed: 1`, no summary, no NPU test,
`Passed: 13`). **A check no honest run could pass** — C4's "registry JSON newer than the gate
stamp" was unsatisfiable (the sweep runs hours before the gate); replaced by a git proof (shapes
absent at base, present now). **A coverage clause is not a correctness clause** — D's first draft
trusted `passed` from gitignored `results/`, ignored `seq_len`, and injected only the first declared
shape; a coverage clause needs its own freshness, verdict re-derivation and negative control. **The
recovery paths are the least-tested code** `[2026-08-05]` — a `529 Overloaded` incident found
`state_halt` not recording where it halted (resume re-ran the phase, emptied the review diff, made
the tamper check vacuous) and retries charged against `PL_MAX_INVOCATIONS` (nine attempts, no token
billed); an implement-step halt (E1, six commits) poisoned the base the same way until `run_phase`
keyed on resuming rather than ordinal. **The driver watches itself** — `guard_gate_files()`
fingerprints `port-loop.sh` and `port-loop/` with no allowlist covering them; the driver's own
commits sit before the review base, which hid the vacuous fingerprint and also produced a wrong D1
round-3 finding about fault injection. **Review the harness alongside the phase; test the negative
path.** Convention violation recorded, not fixed: `phases.sh` 1306 lines (1028 before E) — ~1050
lines are objective checks, so a two-way split leaves the second file over the cap. The dispatch
vectors' negative control (fault totals equal clean totals, §7.4) closed the gap that `results/`
being gitignored opened. `phase_e_checks.py selftest` (27 clauses) — its first version shared
fixture vectors by reference so mutations leaked between cases.

### 12.3 Measured cost

| Phase | Invocations | Wall clock |
|---|---|---|
| A | — | 18 min |
| B | — | 362 min |
| A + B | 11 of 40 | 380 min |
| C1 / C2 / C3 | — | 61 / 45 / 68 min |
| C4 | — | 504 min, then 66 min to re-run the gate |
| C1–C4 | 10 of 40 | ~12 h |
| D1 | — | 11 min of work in a ~2 h window (provider outage) |
| D2 | — | 156 min |
| D1 + D2 | 21 of 40 | ~4.5 h, ~1 h outage |
| E1–E5 | 24 of 60 | ~8.5 h |

Phase B's hours were the runlist spike and ten `make verify` runs; C4's were the sweep plus the
same ten-model check. Splitting paid for itself: C's 8,160 source lines cost the same order of
invocations as B's 3,725 with every gate passing first time; D2's two rounds of fixes never re-ran
D1's hardware. Codex spend was dominated by four aborted Phase A restarts on harness bugs. The
review base is phase-local (`HEAD` at phase entry).
