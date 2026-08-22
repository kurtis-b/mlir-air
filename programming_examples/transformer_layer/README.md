# Transformer Layer — AIE2P Device Kernels

The transformer-layer execution studies' code: the C++ device kernels one encoder/decoder block
needs (Phase A, Peano for AIE2P), the runtime-seam gate (Phase B), the operator builders and their
numerical checks (Phases C/D1), the registry sweep (C4), the block integration gate (D2), the four
execution strategies under `pattern/` (Phase E), the study harness under `study/` (Phases F/G),
and the hardware gates of the compiler phases (H, H9, J7a, J7b). Rewritten 2026-08-22 as the
directory's user-facing README; the previous 1,688-line version is at git tag
`pre-cleanup-20260821`. The plan documents are in
[docs/plans/transformer-layer-execution-studies/](../../docs/plans/transformer-layer-execution-studies/README.md)
— read that index first for where things stand; this file says how to run what is here and which
facts about it will cost time to relearn.

```bash
make compile                 # build every object and check its symbols (no NPU)
make seam-tests              # BO pooling + runlist aggregation rules (no NPU)

ninja -C build-xrt check-programming-examples-transformer-layer-host   # the PR-safe subset

flock -x -w 1800 /tmp/mlir-air-npu.lock make runlist-gate   # NEEDS AN NPU
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  ninja -C build-xrt check-programming-examples-transformer-layer   # the whole suite, NEEDS AN NPU
```

**Two lit targets, and the split is the point.** `check-programming-examples-transformer-layer` is
the whole suite, enrolled by path (`--filter "transformer_layer/"`, so a new `.lit` here joins it
with no CMake change). Counted on the tracked `.lit` files on 2026-08-22: 39 tests, **24 carrying
`REQUIRES: ryzen_ai_npu2`**, one Peano-only (`run_npu2_compile_peano.lit`), one feature-gated
(`run_pipeline_fusion_tests.lit`, `REQUIRES: air_fuse_pipeline_launches`), 13 host-only. It is the
local regression gate and **not** a PR gate: with no NPU2 the 24 report UNSUPPORTED and lit still
exits 0. `check-programming-examples-transformer-layer-host` is the PR gate: an explicit allowlist in
`programming_examples/CMakeLists.txt` (the Peano compile plus the 13 host-only tests) — not a
`--filter-out`, which would enrol every future NPU-gated `.lit` silently. Add a new host-only test
to the list in the commit that adds it. Run lit with the toolchain environment sourced (it invokes a
bare `python3`, which has no `ml_dtypes`), and after any clean rebuild re-check the two CMake flags
without which every NPU test is UNSUPPORTED
([15](../../docs/plans/transformer-layer-execution-studies/15-environment-notes.md)).

**Running a study profile** — one profile, one command, one manifest:

```bash
systemd-inhibit --what=handle-lid-switch:sleep:idle \
  ../../agents/scripts/devq.sh run --class measure -- \
    python3 study/run_profile.py --profile smoke --out-dir results/smoke-w1
```

`run`, never `submit` (`submit` diverts output to the job log and still exits 0), and the `measure`
class is what keeps builds off the box while the clock runs. Profiles: `smoke`, `ladder`, `full`
(`study/profiles.py`). `--dry-run` prints the plan and the expected row counts without touching the
device; `--gate-only` re-verifies a recorded results root and rewrites its manifest. Walk it
**twice** into two roots and compare with `study/compare_roots.py --baseline A --candidate B`: a
single walk once published a crossover that a second walk refuted
([27](../../docs/plans/transformer-layer-execution-studies/27-common-ladder-result.md)).

**The two NPU locks are different inodes, and they must not nest.** `/tmp/mlir-air-npu.lock` is the
*invocation* lock, taken once around a whole suite (by a human or `devq.sh`). Device access is
locked one layer down on `/tmp/npu.lock`, held by `XRTRunner.run_test`, `KernelCache` and
`programming_examples/lit.cfg.py`'s NPU substitution across load and dispatch. No `.lit` recipe here
wraps its commands in the outer lock: the caller holds it, and BSD `flock(2)` treats a second
`open()` of the same file as a foreign lock, so a nested acquire blocks against its own parent until
the timeout ([15 §Locks](../../docs/plans/transformer-layer-execution-studies/15-environment-notes.md)).

## The Phase B runtime seam

`runlist_gate.py` (`make runlist-gate`, `run_npu2_runlist_gate.lit`) is Phase B's hardware gate:
three separately compiled ELFs in one runlist, bit-identical to sequential dispatch and faster, plus
the whole five-GEMM layer through the seam in one submission. The plan's proposed mechanism —
several ELFs in one `hw_context` — does not work on XRT 2.21.0 / NPU2 and is not needed: an AIR ELF
is a full ELF carrying its own array configuration, a context accepts exactly one, but a runlist is
constructed *against* a context and each entry dispatches on the context its kernel came from. So N
ELFs means N contexts and still one runlist. Measured facts, all in
[01 §4.2](../../docs/plans/transformer-layer-execution-studies/01-original-plan-superseded.md):
bit-identical across orders and hosting contexts at the Llama-3.2-1B seq-2048 projection shapes;
runlist over sequential 1.024–1.15× (three distinct ELFs 20.826 → 19.949 ms, 25/25 pairs); the
concurrent `hw_context` ceiling is **32** (33 fails loudly at load); the standalone `drain` GEMM ELF
is single-shot (100 % → 10.63 % on call 1) and leg **D** re-measures it every run; the five-GEMM
layer runs as **one** submission moving **117 MB less** on its second dispatch (10 sync boundaries,
not 15). Legs: **A** cross-artifact runlist, **A2** xclbin refusal, **B** within-artifact
aggregation, **C** the layer through `KernelCache.run_sequence`, **D** the drain defect.
`[2026-08-10]` The latency verdict compares interleaved **minimums** (`agg_min < seq_min`; leg A
24.786 vs 24.921 ms, leg B 23.263 vs 23.607 ms when validated), because the strict
`agg_ms < seq_ms` clause went intermittent under suite contention
([25 §4](../../docs/plans/transformer-layer-execution-studies/25-mode-rebuilds-and-results.md)).

The one aggregation that is silently wrong is the *xclbin* cross-artifact runlist — configuration
lives in the xclbin, not the run, so it executes and returns wrong numbers with no error. The seam
refuses to build it and leg A2 keeps the refusal honest.

| | |
|---|---|
| `llms/shared/infra/dispatch.py` | Groups a dispatch sequence into the submissions the hardware allows — one, spanning artifacts, under the ELF ABI — and owns the six-field dispatch vector. Under the xclbin ABI it splits at every configuration change and refuses `require_single` |
| `llms/shared/infra/bo_pool.py` | Live ranges over the sequence, 4 KiB-binned slot sharing, a content-keyed static-weight pool, and a dirty bit per BO so only written buffers sync to device and only declared outputs come back |

The rules both modules implement are in
[05b](../../docs/plans/transformer-layer-execution-studies/05b-phase-b-buffer-rules.md), written
before the code; the module docstrings name its sections. Read it before changing either: a pooled
BO larger than its buffer, a zero-copy view into pool memory, an xclbin-ABI slot keyed by argument
index — all produce plausible wrong numbers rather than errors.

## What lives here

| File | Contents |
|---|---|
| `kernels/encoder.cc` | Encoder-block kernels: staged FFN (`-DBUILD_FFN`) and weighted add-norm (`-DBUILD_ADDNORM`); contract docs and the `extern "C"` entry points. Built to **two** objects: `encoder.o` (addnorm half) and `encoder_ffn.o` (FFN half) |
| `kernels/encoder_matmul.cc`, `kernels/encoder_layer_norm.cc` | The encoder's 2x2-expanded `aie::mmul` microkernels and its LayerNorm reductions, included by `encoder.cc` |
| `kernels/addnorm_ffn.cc` | Fused add-norm + FFN staging, both residual orderings behind `-DADDNORM_PRE_ADD`; contract docs and entry points |
| `kernels/addnorm_ffn_matmul.cc`, `kernels/addnorm_ffn_norm.cc` | The FFN's 1x4-expanded microkernels; the fused add-norm templates and tile passthroughs — the only file `-DADDNORM_PRE_ADD` reaches |
| `kernels/elementwise.cc` | `eltwise_vadd` and `gelu_tanh_approx_bf16`, included by both kernels |
| `kernels/transpose.cc` | bf16 tile transpose, copied from `data_transfer_transpose/dma_bf16/` so the block fingerprint covers the source the transpose ELF links |
| `compile_kernels.py`, `run_npu2_compile_peano.lit` | The compile-and-check driver and Phase A's compile-only gate. Peano, no NPU |
| `runlist_gate.py`, `run_seam_tests.lit`, `run_npu2_runlist_gate.lit` | Phase B: the host-only rules tests and the four-leg hardware gate |
| `opcheck.py` / `opcheck_prepare.py` / `opcheck_specs.py` | Phase C's numerical check: what counts as EVIDENCE (runner, injection, results artifact, CLI) / HOW each operator is built and fed / WHICH `(operator, shape)` is claimed and at what `atol` |
| `opcheck_layer.py` | How the whole-layer checks are prepared: the D2 block and every Phase E mode share `prepare_layer_dispatch` and `dispatch_vector_totals` |
| `lit_pin.py` | Runs a host test and asserts the pins its `.lit` carries |
| `builders/elementwise_add.py`, `layer_norm.py`, `softmax.py`, `addnorm.py`, `qkv_proj.py`, `gelu.py`, `ffn.py` | One `build_<name>_module()` per operator with its FP32 reference beside it. `softmax` is the STREAMING family in `softmax/softmax.cc` (init / partial / normalize), not the single-shot `softmax_bf16`; `addnorm` takes `pre_add=` and has one reference per ordering; `qkv_proj` is one GEMM over the fused `[K, 3K]` weight, C split three ways on the device |
| `builders/mha_attention.py`, `o_proj.py`, `mha_out_proj.py` | Seq-first FlashAttention staging with its `-D` flags and chunked FP32 oracle; the O-projection; the entry layer composing both into one ELF |
| `builders/transpose.py`, `builders/elementwise_mul.py` | The two operators `runlist` needed that did not exist (see that mode) |
| `builders/gemm_spec.py` | `resolve_gemm_spec(m, k, n)`: the registry-resolved GEMM recipe, herd merged from the row |
| `builders/block.py`, `block_cache.py`, `test_block_cache.py` | Phase D2: five operator launches in four `KernelCache.run_sequence` calls, every boundary read back; the ELF-cache fingerprint; host-only reuse tests |
| `builders/norm_tail.py`, `norm_tail_structure.py` | J7a: add → LayerNorm → gamma as three herds in one segment on L1→L1 channels; the host-only structural arm (through the real aircc) |
| `builders/ffn_accum.py`, `ffn_accum_structure.py` | J7b: the FFN down-projection as a naive K loop that `air-hoist-dma-in-accum-pattern` turns into an accumulator ring; the structural arm |
| `builders/ffn_resident.py`, `test_ffn_resident.py`, `ffn_resident_structure.py` | R1 ([31](../../docs/plans/transformer-layer-execution-studies/31-resident-tail-r1-record.md)): the FFN as ONE segment, `[seq, ffn]` never crossing DRAM. `test_ffn_resident.py` builds the module and INTERPRETS it in f64 (max \|y − ref\| 5.457e-12, two FileCheck-matched negative controls); `ffn_resident_structure.py` counts the per-column shim MM2S budget as demand (7/16 ports, worst column 2). Neither needs an NPU |
| `builders/pipeline_spec.py`, `pipeline_fusion_structure.py` | H8: the declarative co-residency surface and the gate that `air-fuse-pipeline-launches` reproduces the hand-written J7a module |
| `builders/test_layer_norm_rows.py`, `test_softmax_rows.py`, `test_profile_bounds.py` | Host-only: the row-width walls and the profile's derived bounds, at every ladder point |
| `addnorm_multitrip.py`, `run_npu2_addnorm_multitrip_peano.lit` | `[2026-08-21]` H9's hardware gate: `addnorm` at two trips of the row loop in five variants (see the operator checks) |
| `pattern/reference.py`, `test_reference.py` | The shared golden model — iron's draw order, this repository's FP32-from-bf16 numerics, both workload variants, every boundary — and its host-only checks |
| `pattern/blocked_attention.py`, `test_blocked_attention.py` | Query-blocked host attention; `run_blocked_attention_tests.lit` pins it against the device decomposition |
| `pattern/coarse/`, `offload/`, `runlist/`, `fused/` | The four execution strategies, each with its own README, ELF cache and catalogue row |
| `pattern/coarse/cells.py`, `pattern/coarse_c2/`, `pattern/coarse_c3/`, `coarse_cells_structure.py` | `coarse`'s blend space, the two interior cells as catalogue operators, and the host-only prediction of what each dispatches |
| `run_npu2_<op>_peano.lit` | One per operator and per mode: its numerical gate on a real NPU. `run_npu2_fault_control_peano.lit` is the negative control; `run_npu2_block_peano.lit` the D2 gate; `run_npu2_fused_decoder_reexec_peano.lit` the fused decoder's two-dispatch re-execution gate ([16 §13](../../docs/plans/transformer-layer-execution-studies/16-compiler-changes.md)) |
| `run_reference_tests.lit`, `run_block_cache_tests.lit`, `run_ffn_resident_emulation_tests.lit`, `run_layer_norm_rows_tests.lit`, `run_softmax_rows_tests.lit`, `run_profile_bounds_tests.lit`, `run_mapping_space_tests.lit`, `run_study_host_tests.lit`, `run_pipeline_fusion_tests.lit` | Host-only lits. `run_study_host_tests` runs `study/run_host_tests.py` (every `study/test_*.py`); `run_mapping_space_tests` walks the resident-tail mapping space (~1 min, [16 §14](../../docs/plans/transformer-layer-execution-studies/16-compiler-changes.md)) |
| `study/` | The harness: `run_profile.py` (the one entry), `profiles.py`, `run_mode.py`, `run_ladder.py`, `resume.py`, `compare_roots.py`, `distinguish.py`, `manifest.py`, `schema.py`, `mapping_space.py`, `fused_reexec_gate.py`, `iron_adapter.py`, and a `test_*.py` beside each |
| `sweep/` | The C4 registry sweep: `sweep_families.py` (which shapes and tilings), `sweep_measure.py` (one candidate end to end), `registry_sweep.py` (orchestration, CLI), `sweep_report.py` (resolution assertion, family markdown), `registry_writer.py` (append-only write); three host-only lits |

There are two compiled objects from `kernels/`, not six: `encoder.cc` and `addnorm_ffn.cc` are the
only translation units; each `#include`s its two siblings the way
`matrix_multiplication/bf16_in_fp32_out/mm_aie2p.cc` includes `zero.cc`. Kernels that already had an
MLIR-AIR home were extended in place behind a flag that defaults off, the default-build objects
verified byte-identical to their pre-port versions:

| Kernel | Where | Opt-in flag |
|---|---|---|
| `matmul_init_*`, `matmul_with_acc_*` | `matrix_multiplication/bf16_in_fp32_out/mm_aie2p.cc` | `-DGENERATE_MATMUL_INIT_KERNELS`, `-DGENERATE_MATMUL_WITH_ACC_KERNELS` |
| Two-pass streaming softmax | `softmax/softmax.cc` | `-DSOFTMAX_STREAMING` |
| Multi-row LayerNorm | `layer_norm/layer_norm.cc` | — (new file) |
| Causal-mask row helpers | `flash_attention/kernel_fusion_based/attn_npu2.cc` | `-DCAUSAL_ROW_HELPERS` |

## Why the two kernels are each three files

As first landed, `encoder.cc` (973 lines) and `addnorm_ffn.cc` (1116) both ran past the ~800-line
module guideline in
[02-porting-conventions.md](../../docs/plans/transformer-layer-execution-studies/02-porting-conventions.md).
Each is split along the seam it already had — **matmul microkernels · normalization templates ·
`extern "C"` entry points** — leaving every source between 245 and 493 lines. The seam is by
*role*, not by the `-DBUILD_FFN` / `-DBUILD_ADDNORM` flags, which gate only the entry-point layer;
it also puts each footgun next to its code: `-DADDNORM_PRE_ADD` is read entirely inside
`addnorm_ffn_norm.cc`, and the variance clamp keeping `aie::invsqrt` off a negative operand is
documented in the two normalization files. Splitting changed no code and no object: the sources are
included textually, so `encoder.o` and `addnorm_ffn.o` are still one translation unit each with the
same flags and the same symbols the compile gate checks.

## The operator checks

`opcheck.py` is the single numerical entry point for every operator here; `builders/` holds the
operators, one `build_<name>_module()` with its FP32 reference beside it, no operator class. Adding
a shape touches only `opcheck_specs.py`; adding an operator touches the catalogue and
`opcheck_prepare.py`; changing what counts as evidence touches only `opcheck.py`.

```bash
make opcheck-list                 # every (operator, shape) claimed, as JSON. No NPU.
flock -x -w 1800 /tmp/mlir-air-npu.lock make check-layer-norm PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR
flock -x -w 1800 /tmp/mlir-air-npu.lock make check-fault-control PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR
```

Per-operator targets, each with a `run_npu2_<op>_peano.lit`: `check-elementwise-add`,
`check-causal-mask`, `check-layer-norm`, `check-softmax` (+ `check-softmax-fault`, its own control
because the injection target is chosen per shape by measurement), `check-addnorm`, `check-qkv-proj`,
`check-ffn` (+ `check-ffn-ladder-fault`), `check-mha-out-proj`, `check-transpose`,
`check-elementwise-mul`, `check-norm-tail` (+ `-fault`, `-structure`), `check-ffn-accum` (+ `-fault`,
`-structure`), `check-ffn-resident` (+ `-fault`, `-structure`), `check-pipeline-fusion`,
`check-fused-decoder-reexec`, `check-addnorm-multitrip`. `qkv_proj`, `ffn` and `mha_out_proj` take
their tiles and method from `kernel_registry`, which **raises** on a shape nobody swept.

The verdict is `XRTRunner.run_test`'s: `np.isclose` over the full output at `rtol = 1.6e-2` with
zero permitted mismatches, against a reference computed in float32 from bf16-rounded inputs.
`opcheck.py` subclasses the runner only to copy out the error statistics. Each run writes
`results/<operator>__<shape>.json`; the registry rows in `programming_examples/kernel_registry/` are
the durable record. `opcheck.py` records `atol_required` — the smallest `atol` the run would have
passed at, `max(|a−e| − rtol·|e|)` — and that, not `abs_err_max`, is what an `atol` is quoted
against.

`--fault-inject input` is the negative control: it perturbs one element of the array handed to the
**device**, after the reference was computed from the clean one, and the run must FAIL. A reference
compared against itself, a tolerance wide enough to swallow anything, and an ignored flag all still
PASS under injection. `check-fault-control` runs it with `--expect-failure`, which reports the
control's own verdict — exit 0 only if the comparison ran and rejected the perturbed run — rather
than inverting the exit status, which would read a missing `PEANO_INSTALL_DIR`, a link error or an
absent NPU as a caught fault. Injected runs write into `results/fault/`.

**A shape not in `opcheck.py --list` is validated by nothing.** `results/` is gitignored, so a
results file for an undeclared shape is invisible to every review. Adding a shape means adding it to
`SPECS` in `opcheck_specs.py` *and* adding its `CHECK` line to the operator's `.lit`. **And a new
shape usually needs an injection of its own**: `check-fault-control` injects one operator and the
driver injects each operator's *first* declared shape, so the newest row is the one nobody perturbs.
`check-ffn-ladder-fault` (`64x768x3072`) is the pattern to copy.

### The `baseline_768` set (Phase D1)

The block runs at `baseline_768` — hidden 768, ffn 3072, 12 heads × head_dim 64, `encoder_bert` —
so each operator carries one row there, and a block failure localizes to the integration:

| operator | `shape_key` | measured `mean_rel_L1` | `atol_required` | `atol` | margin |
|---|---|---|---|---|---|
| `elementwise_add` | `4096x768` | 1.879e-3 | 0.0 | 5e-2 | `rtol` alone covers it |
| `layer_norm` | `4096x768` | 1.969e-3 | 1.419e-3 | 5e-3 | 3.5× |
| `addnorm` | `64x768_pre_add` | 2.687e-3 | 6.646e-4 | 2e-3 | 3.0× |
| `qkv_proj` | `4096x768` | 9.863e-3 | 1.773e-3 | 5e-3 | 2.8× |
| `ffn` | `4096x768x3072` | 1.569e-2 | 1.472e-3 | 5e-3 | 3.4× |
| `mha_out_proj` | `4096x768x12h` | 5.335e-2 | 8.706e-3 | 2.5e-2 | 2.9× |

`rtol` is `1.6e-2` throughout and every `atol` is inside the hard `1e-1` ceiling. The three
GEMM-backed rows are pinned to `seq_len = 4096` because the block runs there; for `ffn` there was no
choice until Phase E1 — 4096 was the only point that built at hidden 768 (see the `tile_n` trap
below). `ffn` now carries a second ladder point at `seq = 64`: `mean_rel_L1` 1.561e-2,
`atol_required` 1.150e-3, the same 5e-3 at a 4.3× margin, both projections `drain` at `tile_n` 128
and 96, herd 2×4 rather than the file-level 8×4 because at `M = 64` drain's forced `tile_m = 32`
admits at most `herd_m = 2`. The row-parallel operators are not pinned; they run 4096 rows except
`addnorm`, whose L1 budget caps it. `causal_mask` has no `baseline_768` row: its shape is
`seq × seq` and `encoder_bert` never builds one. `[2026-08-11]` The LayerNorm statistics moved to
two-pass f32 in every dispatched kernel, and the catalogue gained the offset-regime rows
`128x768_offset` (`layer_norm`, `norm_tail`), `64x512_offset` and `64x768_pre_add_offset`
(`addnorm`) — numbers under the J7a findings below.

### The H9 multi-trip gate `[2026-08-21]`

`make check-addnorm-multitrip` (`run_npu2_addnorm_multitrip_peano.lit`) runs
`addnorm_multitrip.py` in five variants — `inside`, `hoisted`, `annotated`, `annotated_hoisted`
(one column, `cols=64, rows=8, rows_per_call=4`, the exact shape of the original miscompile) and
`multicolumn` (`herd_x=8`, two trips per column) — all at TWO trips of the row loop, the count
`builders/addnorm.py` forbids, with exact clauses rather than a tolerance. It ran from the retired
port-loop harness until the 2026-08-21 cleanup; this lit is now its only runner. The claim is about
the COMPILER: do not edit the fixture to make a compiler change pass.

## The block integration gate (Phase D2)

One whole `encoder_bert` layer at the forced configuration: `seq_len 4096`, hidden 768, ffn 3072,
12 heads × head_dim 64, non-causal.

```bash
make reference-tests                                        # the golden model, no NPU
make block-cache-tests                                      # what the ELF cache reuses, no NPU
flock -x -w 1800 /tmp/mlir-air-npu.lock make check-block       PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR
flock -x -w 1800 /tmp/mlir-air-npu.lock make check-block-fault PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR
```

Five operator launches over four separately compiled ELFs — `layer_norm`, `elementwise_add` and
`causal_mask` are not on this path (the residual add lives inside the pre-add `addnorm`;
`encoder_bert` is bidirectional):

| # | operator | in → out |
|---|---|---|
| 1 | `qkv_proj` | `x` → `q, k, v` |
| 2 | `mha_out_proj` | `q, k, v, w_o` → `attn_context, attn_out` |
| 3 | `addnorm` **pre-add** ×64 | `attn_out, x, ln1_weight` → `hidden` |
| 4 | `ffn` | `hidden, w_up, w_down` → `ffn_up, ffn_gelu, ffn_out` |
| 5 | `addnorm` **pre-add** ×64 | `ffn_out, hidden, ln2_weight` → `output` |

**The golden model.** `pattern/reference.py` ports the **structure** of iron's
`generate_golden_reference` (draw order preserved exactly: `input`, then `q/k/v/attn_output`
weights, `ln1_weight`, `ffn_up`, `ffn_down`, `ln2_weight`; `ln*` is `rand`; biases are `zeros` and
consume no RNG) and not its bf16 numerics: draws are f32 rounded once to bf16 and the arithmetic is
f32. Each boundary is computed by the operator oracle D1 already validated, so there is one
implementation of each piece of arithmetic. `pattern/test_reference.py` pins the composition against
a straight-line torch transcription, because a composition can be well-typed and still be the wrong
layer.

**The ten boundaries:**

| boundary | elements | `mean_rel_L1` | `atol_required` | `atol` | margin |
|---|---|---|---|---|---|
| `q` / `k` / `v` | 3145728 each | 9.7e-3 | 3.1e-3 | 5e-3 | 1.6× |
| `attn_context` | 3145728 | 1.774e-1 | 2.288e-4 | 1e-3 | 4.4× |
| `attn_out` | 3145728 | 1.406e-1 | 7.371e-4 | 2.5e-3 | 3.4× |
| `hidden` | 3145728 | 5.181e-3 | 1.176e-2 | 3.5e-2 | 3.0× |
| `ffn_up` | 12582912 | 1.154e-2 | 4.977e-2 | 1.5e-1 | 3.0× |
| `ffn_gelu` | 12582912 | 1.660e-2 | 4.519e-2 | 1.5e-1 | 3.3× |
| `ffn_out` | 3145728 | 1.783e-2 | 1.144e-1 | 3.0e-1 | 2.6× |
| `output` | 3145728 | 1.688e-2 | 7.398e-2 | 1e-1 | 1.35× |

The layer's relative error, 1.688e-2, is within 8 % of the `ffn` row's 1.569e-2: the FFN dominates
and nothing downstream amplifies it. The *absolute* number is large because of scale — activations
run around 1 where the registry's GEMM sweep puts a depth-3072 reduction at `1/sqrt(3072)`, ~60×
smaller; the `ffn` row's 1.472e-3 scaled by that is 9e-2 at the second LayerNorm, which divided by
its ~1.2 row standard deviation and multiplied by a gamma in [0, 1) is the measured 7.4e-2.
**`output`'s `atol` is 1e-1, the hard ceiling, at a 1.35× margin** — stated rather than padded
because there is nowhere to pad to. `attn_context`/`attn_out` carry relative error above 14 % and
`atol_required` three orders below everything else, one fact: iron's `val_range = 0.05` makes the
softmax nearly uniform, so attention output is an average of 4096 V rows landing around 1e-3 against
a residual around 5e-2. Hence the per-boundary comparison is a work item, not a nicety:

- Perturbing one element of `w_o` or of the fused QKV weight by the shared `FAULT_DELTA` of 2.0 puts
  **zero** layer-output elements outside tolerance. The block's control goes into `ln1_weight`,
  which reaches 8 % of the output through two paths; `opcheck_specs.py` records all seven
  candidates.
- Swapping the tanh GeLU for the erf form moves `ffn_gelu` by at most 4.7e-4 and the output by
  6.1e-4 — no tolerance here would see it; `pattern/test_reference.py` pins that oracle by identity.
- The 25 % of `ffn_up` that came back zero during bring-up reached the output as "54 % of elements
  wrong"; the stage list said `ffn_up`, and that everything before it was exact.

**Four dispatch sequences, not one.** `addnorm` drives three L3→L1 streams per tile against a
column's two shim MM2S channels, so it takes one kernel call per tile and L1 caps it at
`addnorm_max_rows(768, pre_add=True)` = 104 rows; the two normalization points are row-blocked
into **64 dispatches each of the 64×768 pre-add shape D1 measured**. A dispatch argument is a whole
BO (`run.set_arg` takes a buffer, never a buffer plus an offset), so the layer is four
`run_sequence` calls — `qkv_proj + mha_out_proj`, ln1, `ffn`, ln2 — of which only the first is fully
device-resident.

**The ELF cache is keyed by fingerprint, not by name.** `check-block` and `check-block-fault` share
four cached ELFs under `block_cache/`. An artifact name carries only the shape; the tiles come from
the registry and the IR from the builders, so "four binaries whose names match" is also satisfied by
binaries from a re-swept row or a since-fixed builder — a passing gate for an implementation that
never reached the device. `compile_block_artifacts` builds all four MLIR modules on every call
(~0.1 s) and reuses a cached ELF only where the recorded fingerprint over its built MLIR, resolved
configuration, every device kernel source and its backend kwargs still matches
(`builders/block_cache.py`; the configuration is fingerprinted because a row's `tile_k_l1` reaches
the ELF through `-D` flags the IR does not carry). What it cannot see: the toolchain (after a bump,
`make clean`), and compile ORDER (E1's interleaving change was verified by deleting `block_cache/`
and checking every stage figure came back byte-identical). `make block-cache-tests` pins both
directions.

## The `coarse` execution strategy (Phase E2)

**What it is.** `pattern/coarse/coarse.py` is `builders/block.py` — the D2 layer, five launches over
four ELFs, four `run_sequence` calls — wrapped with its own ELF cache (`coarse_cache/`), its own
`SPECS` row and its `execution_mode` CSV value from `pattern/__init__.py::EXECUTION_MODE_CSV`
(iron's old name, per porting convention 7). Not a second block: `block.py` and its enrolments are
untouched. `make check-coarse` / `check-coarse-fault`; `run_npu2_coarse_peano.lit`.

**Dispatch vector**, four recorded rows (qkv+mha, norm 1, ffn, norm 2) at 4096:

```
host_submissions  runlist_entries  air_launches  herd_launches  sync_boundaries      bytes
       1                2               6             10              9          80,216,064
       1               64               1             64            193          18,875,904
       1                1               4              8              7          84,934,656
       1               64               1             64            193          18,875,904
```

Totals **4 / 131 / 12 / 146 / 402 / 202,902,528**. 128 of the 131 entries — 98 % — are `addnorm`'s
row blocking, which is why any threshold on entries would measure `build_addnorm_module`'s L1
capacity rather than the taxonomy. Three parts of the contract are enforced somewhere that catches
them:

- **Each mode gets its own `KernelCache` directory.** The fingerprint is sound but the directory is
  chosen by name, so two modes on one directory can trade ELFs whose fingerprints agree — valid
  output attributed to the wrong boundary. Every cache is gitignored and in `make clean`, because
  the negative control runs `opcheck.py` from the source tree.
- **The vectors are recorded, never counted.** Every row is `DispatchVector.as_row()` out of
  `KernelCache.run_sequence`. `runlist_entries_per_submission` is a **mean**, so totals are
  `Σ round(mean × submissions)`, and the driver rejects a row whose product is not whole.
- **The fault-injected run carries the vectors too**, and its six summed totals must equal the
  clean run's: injection perturbs one input element after the reference exists and never touches
  the dispatch path. The lit recipes pin the totals line to one set of literals in both halves.

The cost: each lit test starts with `make clean` in its own working directory, so `coarse` compiles
its four ELFs rather than inheriting `block`'s — real minutes per gate, and the price of the two
modes never trading ELFs. `pattern/coarse/README.md` has the rest.

## The `offload` execution strategy (Phase E3)

**What it is.** The mode that **minimizes reconfiguration**
([03 §The taxonomy](../../docs/plans/transformer-layer-execution-studies/03-measurement-model.md)),
its host/device split decided by linearity: every LINEAR operator on the NPU — the six projections
and both attention matmuls, per head — and every non-linear one (the softmax between the attention
matmuls, both LayerNorms, the GeLU) in host torch. At the gate configuration that is **30
dispatches** (6 + 2 × 12 heads), each a one-step `KernelCache.run_sequence` call, so the mode
aggregates *nothing* — its own clause in the distinguishability gate, and why the dispatches are
separate calls (under the ELF ABI `run_sequence` would merge them into `coarse`'s shape). The two
attention GEMMs (`4096x64x4096`, `4096x4096x64`) resolve in no registry; their tiles are injected
through `gemm_spec_fn` (`gemm_spec_source: registry+injected`, `attention_path:
"device_gemm_host_softmax"`). `[2026-08-09]` The earlier "host-mediated extreme, six dispatches"
framing is superseded; the non-aggregation argument survives. `make check-offload` /
`check-offload-fault` / `check-offload-shared`; `run_npu2_offload_peano.lit`.

**Two packaging paths, and the difference IS the mode.** The same 30 dispatches run either over
five xclbins with the `hw_context` torn down before each (**30** `context_loads`, the ELF path) or
over **one** xclbin holding all five shapes, loaded once with 4 kernel attaches. `[2026-08-11]` The
shared path is the **default**; the ELF path is the legacy/control arm, opt-in
`AIR_OFFLOAD_LEGACY_ELF=1`; the retired `AIR_OFFLOAD_SHARED_XCLBIN` **raises** if set in any form.
Reconfiguration is counted by `KernelCache.reconfiguration_counts()` — it cannot be a seventh
vector field, since the vector is identical on both paths — and the lit pins `context_loads 1
kernel_attaches 4` on the default recipe and `context_loads 30 kernel_attaches 0` on the legacy one.
The shared path needs THREE distinct identifiers per stream (`kernel_name`, `instance_name`,
`kernel_id`) and only the first fails loudly: a duplicate `kernel_id` runs the second kernel against
the first's configuration and returns garbage at `mean_rel_L1` 1.41 with no error
([25 §5.1](../../docs/plans/transformer-layer-execution-studies/25-mode-rebuilds-and-results.md)).
It runs only where every module is **single-launch** — a PLATFORM bound since the 2026-08-11
hardware verdict (in-stream `load_pdi` faults the firmware: `ERT_CMD_STATE_TIMEOUT`,
`fatal_error_type 0x10`), so at 4096 the two-launch `fused-cast` down-projection is re-resolved to
the shape's measured `drain` row (6,226 vs 6,927 GFLOP/s, ~10 %) under the shared path only
([25 §5.5](../../docs/plans/transformer-layer-execution-studies/25-mode-rebuilds-and-results.md)).

**Dispatch vector.** At 4096, ELF path (clean half): **30 / 30 / 31 / 91 / 91 / 970,457,088**; shared
path at 4096 differs by exactly 12,582,912 bytes (the `fused-cast` f32 C scratch) and one launch
(30 / 90 / 90). At 1024, shared, steady: `submissions 30 entries 30 air 30 herd 90 sync 90 bytes
99090432`; the cold call reads `sync 95 bytes 99141520` because the xclbin ABI uploads each
artifact's instruction stream once (51,088 bytes = the five `.insts.bin`) — **a vector read from a
single cold dispatch under this ABI is inflated**, and the target dispatches twice. Variance: at 512
the ELF path's intra-walk spread was **316.9 % / 134.1 %** against the shared path's 17.6 % / 14.0 %,
the switch removes it — though it changes the ABI as well as eviction, so `_evict_context` is the
leading candidate, not a demonstrated cause. **Every `offload` latency/variance number before
2026-08-11 describes the ELF path**; the post-flip four-mode walk is
[32 §The post-flip walk](../../docs/plans/transformer-layer-execution-studies/32-cost-decomposed-ladder.md).

Three things that cost time: **the mode computes, the oracle checks, they may not share
arithmetic** — every host stage is torch (`F.layer_norm`, `F.gelu(approximate="tanh")`, torch
softmax) while the oracle's boundaries come from the numpy references, and
`run_blocked_attention_tests.lit` pins the two attention implementations against each other on
identical inputs. **One ELF serves four dispatches, and that is not aggregation** —
q/k/v/output_proj are the same `4096x768x768` module; the weights are deliberately not static and
`x` is re-uploaded for each of q/k/v; do not optimize it. **A plain GEMM ELF's `instance_name` is
the method's func name** — `drain` emits `matmul_bf16`, `fused-cast` `gemm_cast_bf16`; a mismatch
does not fail to load, it times out with `ERT_CMD_STATE_TIMEOUT`. `pattern/offload/offload.py::_METHOD_FUNC`
is the one place that mapping lives. `pattern/offload/README.md` has the rest.

## The `runlist` execution strategy (Phase E4)

**What it is.** Every operator on the device, nothing on the host, dispatched individually and
aggregated into runlists: **427 entries over 17 runlists** (`[2026-08-09]`):

| runlist | contents | entries |
|---|---|---|
| 1 | `q_proj`, `k_proj`, `v_proj` | 3 |
| 2..13 | **per head:** `attn_scores` → `softmax` → `attn_output`, device-resident | 3 × 12 = 36 |
| 14 | `output_proj` | 1 |
| 15 | 64 × (residual add → LayerNorm → gamma multiply), ln1 | 192 |
| 16 | `up_proj`, GeLU, `down_proj` | 3 |
| 17 | 64 × (residual add → LayerNorm → gamma multiply), ln2 | 192 |

Each runlist is forced single-submission (`require_single_submission=True`); intermediates inside
one stay device-resident. The attention tiles are imported from `offload`'s `ATTENTION_GEMM_TILES`;
the softmax is `builders/softmax.py` at `[4096, 4096]`, `rows_per_call = 2`. One submission per
head is a memory bound (all twelve at once would hold ~800 MiB of scores and probabilities against
~70 MiB per head), not a schedule choice. The earlier form — host blocked attention, **5 submissions
over 391 entries** — is superseded; its catalogue row was corrected in place
([25 §4](../../docs/plans/transformer-layer-execution-studies/25-mode-rebuilds-and-results.md)).
`make check-runlist` / `check-runlist-fault`; `run_npu2_runlist_peano.lit`.

**Dispatch vector** at 4096: **17 / 427 / 50 / 488 / 451 / 190,513,152**. Host↔device bytes against
`offload` at the same layer: attention 25,165,824 vs 830,472,192 (**33.0×**), everything else
165,347,328 vs 139,984,896 (0.85×), total 5.1× — the headline understates it, since this mode pays
~25 MB more on its banded norm chains. Accuracy cost of the bf16 device softmax: `attn_context`
`mean_rel_L1` 6.665e-2 against `offload`'s 1.554e-2; the output needs `atol_required` 6.981e-2
against the 1e-1 ceiling (1.43×). With `runlist` on the device all four modes are, so
`attention_path` is no longer a covariate (`study/test_attention_path.py` asserts it).

Things that cost time: **the two operators that did not exist** — `builders/transpose.py` (iron's
`k_transpose` over `data_transfer_transpose/dma_bf16/`'s movement, a scalar tile kernel because a
bf16 DMA-stride transpose is illegal; the tile shape is in the OBJECT NAME, `transpose_m64n96.o`;
validated standalone, BIT-exact, not dispatched by the mode) and `builders/elementwise_mul.py`
(`eltwise_add`'s streaming shape with `vector_mul`'s `arith.mulf`, **bf16 end to end because the
AIE vector unit does not legalize f32 vector multiply**, per `weighted_rms_norm.py`). **Why the norm
chains are banded** when the decomposed kernels could stream all 4096 rows: a streaming structure
(13 entries over two runlists) landed BELOW `coarse`'s 131 and changed two variables at once; the
band size is imported from `builders.block.norm_rows`, so the two modes differ in exactly one
variable. **No GEMM ELF executes twice.** Sharing one `4096x768x768` ELF for q/k/v inside one runlist
returned the corruption `offload` had measured across submissions (k 3.539e-1, v 3.561e-1
`mean_rel_L1` against q's clean 9.3e-3; offload's mode 3.56e-1) and context eviction is
structurally unavailable inside a runlist, so the four projections are one module compiled to FOUR
artifacts; the band `add`/`ln`/`mul` ELFs execute 64 times per chain in one context each, measured
clean. **The gamma multiply's second operand is a materialized broadcast**, one `[64, 768]` band
declared static and content-keyed; under injection the key changes and it re-uploads — nothing
special-cases the injected path. `pattern/runlist/README.md` has the rest.

## The `fused` execution strategy (Phase E5)

**What it is.** MLIR-level fusion before compilation via `stitch_elf`
(`llms/shared/infra/stitching.py`), MLIR-AIR's own production mechanism — what makes this port
additive rather than a duplicate of iron. The layer is **one runlist submission of three entries
over three ELFs**: the D2 `qkv_proj` and `mha_out_proj` modules unchanged, then `fused_tail`, a
ten-launch stitched module holding add, LayerNorm, gamma (ln1), the staged FFN, and add, LayerNorm,
gamma (ln2) over whole tensors. CSV value `fused_elf`. `make check-fused` / `check-fused-fault`;
`run_npu2_fused_peano.lit`. `pattern/fused/README.md` has the rest.

**One ELF for the whole layer is blocked by backend settings, not symbols.** E1's `(method,
tile_n)` naming removed the collisions — `fused_tail` co-links `drain` (tile_n 128) and
`fused-cast` (tile_n 96) GEMMs, though on different methods that pair never collided; the evidence
for E1 is the `seq = 64` `ffn` point. One ELF is one aircc invocation: FlashAttention needs
`omit_pingpong="all"` + `runtime_loop_tiling_sizes=[1, 1]` while the 4096-row GEMMs need `[2, 2]`
for BD-ID recycling (`builders/mha_out_proj.py` records them as non-interchangeable), so attention
keeps its own ELF, as every shipped LLM pipeline does. **The normalization is streamed, not
row-blocked**, because a band at a nonzero row offset cannot be routed into a slice's args clause
(`memref.cast` cannot cast an offset subview to the identity layout; the row-0 trick in
`llms/shared/builders/o_gemv_ffn_multi.py` works only at offset 0).

**Bounded to 256..1024** `[2026-08-08]`: the stitched tail's `plane_major` packing needs a plane
stride of `rows × cols` against the shim `aie.dma_bd` cap of 1,048,576, so it caps at 1365 rows at
width 768 and the SPECS row moved 4096 → 1024. **Dispatch vector** at 1024 (repair run):
`submissions 1 entries 3 air 11 herd 23 sync 19 bytes 56626176` (the down-projection is `drain`
there, so the tail takes 11 whole-tensor args instead of 16), numerics `mean_rel_L1` 1.756e-2 at
`atol_required` 5.813e-2 (1.72×). One unreconciled pair at the same length: the 2026-08-09 ladder
reads `sync 13 / bytes 42,467,328` — candidate mechanism, unmeasured: the gate's per-boundary
readbacks sit inside its measured sequence. Do not quote either as the mode's vector without a fresh
run. The 4096-era row — **1 / 3 / 16 / 24 / 19 / 184,025,088**, `mean_rel_L1` 1.784e-2 at
`atol_required` 7.896e-2, a 1.27× margin (`[2026-08-07]` refreshed after J7a moved
`layer_norm_rows` to two-pass f32; was 1.806e-2 at 7.572e-2, 1.32×) — is suspended, not restated.

## The four-mode dispatch-vector table

> **`[2026-08-09]` EVERY ROW BELOW IS SUPERSEDED. Do not cite this table.** It records the four
> implementations before the taxonomy was corrected. Two rows have been re-measured at 4096:
>
> | mode | subs | entries | air | herd | sync | bytes |
> |---|---|---|---|---|---|---|
> | `offload` | 30 | 30 | 31 | 91 | 91 | 970,457,088 |
> | `runlist` | 17 | 427 | 50 | 488 | 451 | 190,513,152 |
>
> The other two cannot be restated here: **`fused` no longer builds at 4096**, so a four-mode table
> at this configuration cannot exist. The cross-mode comparison is at 512 and 1024, walked twice, in
> [27](../../docs/plans/transformer-layer-execution-studies/27-common-ladder-result.md); the
> standing ordering is [32 §The post-flip walk](../../docs/plans/transformer-layer-execution-studies/32-cost-decomposed-ladder.md);
> the 9-length × 4-mode matrix is
> [54](../../docs/plans/transformer-layer-execution-studies/54-first-full-profile-and-decoder-families.md).
> Build cross-mode tables from a ladder run, never from per-mode catalogue rows.

Driver-summed totals at seq 4096, clean and fault-injected runs totaling identically (Phase E's
original result, `[2026-08-05]`):

| mode | host submissions | runlist entries | air launches | herd launches | sync boundaries | bytes |
|---|---|---|---|---|---|---|
| `offload` | 6 | 6 | 7 | 19 | 19 | 139,984,896 |
| `runlist` | 5 | 391 | 14 | 404 | 403 | 165,347,328 |
| `coarse` | 4 | 131 | 12 | 146 | 402 | 202,902,528 |
| `fused` | 1 | 3 | 16 | 24 | 19 | 184,025,088 |

The distinguishability *reasoning* is still how the gate works: the criterion is **ordinal** over the
recorded vectors, never an absolute threshold. The gate is `study/distinguish.py`, run by
`run_profile.gate()` over a root's `<mode>.csv` rows at every length where all four modes passed
(`study/test_distinguish.py` runs each clause both ways): (1) no two modes record the same vector;
(2) `offload` submits more than every other mode and aggregates nothing
(`runlist_entries_per_submission == 1`); (3) `runlist` EXECUTES more than `coarse`
(`herd_launches` — entries would hold by construction, since `runlist` is `coarse`'s schedule
decomposed); (4) `fused` crosses fewer sync boundaries than `coarse`. Reading the columns:
submissions order the modes by host mediation; entries by dispatch granularity; `coarse` and
`runlist` sit within one sync boundary of each other because both restage the norm bands through the
host; `air_launches` counts launches once per distinct ELF, `herd_launches` accumulates per dispatch
step ([03](../../docs/plans/transformer-layer-execution-studies/03-measurement-model.md)).

## `coarse`'s blend cells (C2 and C3)

`coarse` is defined as a per-workload **blend** of `runlist` and `fused`. The space is derived from
the artifact plans and is **two axes, not a choice per operator** — `fused` and `coarse` build their
front from the same two modules and differ in the tail alone:

| | tail stitched | tail banded | tail decomposed |
|---|---|---|---|
| **front `block`** | = `fused` | = `coarse` (C1) | **C2** |
| **front `runlist`** | — | **C3** | = `runlist` (C6) |

The space contains the two things it blends, so "the best cell" would re-derive an endpoint. The
resolution is *per workload*: `fused`'s stitched tail caps at 1365 rows (`builders/norm_tail.py`),
so **at seq ≥ 2048 the entire stitched row is unbuildable** and `coarse` is the mode you use where
`fused` does not fit. `pattern/coarse/cells.py` **composes** the block half's `_sequence_a` /
`_sequence_norm` / `_sequence_ffn` and the runlist half's `run_projections` /
`run_attention_interior` / `run_o_proj` / `run_norm_chain` / `run_ffn` as they are, so a cell
measures the code the D2 and E4 gates validated; it refuses to build C1, `fused` or `runlist`.
Three guards a composed cell needs: **a cross-half GEMM object collision check**
(`cells._check_cell_objects` — `compile_gemm_mm` names its object from `(tile_m, tile_n)` while
`tile_k_l1` is a compile flag, the silent failure D2 hit); **a subset compile** (`keys=` on both
`compile_*_artifacts`); **a gamma adapter** at the seam (banded `addnorm` takes `[emb]`, decomposed
`elementwise_mul` a `[norm_rows, emb]` broadcast). `make check-coarse-c2` / `-c3` (+ `-fault`);
`make coarse-cell-structure` (host-only, no Peano) predicts each cell's shape from the configs —
front `block` 1 / 2, front `runlist` `2 + heads` / `4 + 3·heads`, tail banded 3 / `1 + 2·blocks`,
tail decomposed 3 / `3 + 6·blocks` — and reproduces the two pinned endpoints (4/131, 17/427).

**Measured at `4096x768_encoder_bert`** (`[2026-08-09]`, 10/10 boundaries clean, controls failing):

| cell | front | tail | subs | entries | air | herd | sync | bytes | `mean_rel_L1` / `atol_required` |
|---|---|---|---|---|---|---|---|---|---|
| C1 `coarse` | block | banded | 4 | 131 | 12 | 146 | 402 | 202,902,528 | 1.688e-2 / 7.398e-2 (1.35×) |
| C3 | runlist | banded | 17 | 169 | 46 | 232 | 451 | 190,319,616 | 1.654e-2 / 7.266e-2 (1.38×) |
| C2 | block | decomposed | 4 | 389 | 16 | 402 | 402 | 203,096,064 | 1.784e-2 / 7.896e-2 (**1.27×**) |
| C6 `runlist` | runlist | decomposed | 17 | 427 | 50 | 488 | 451 | 190,513,152 | 1.746e-2 / 6.981e-2 (1.43×) |

The ordinal claim `coarse 131 < C3 169 < C2 389 < runlist 427` holds; the vectors are additive
(`C1 + C6 = C2 + C3` on every column), so the `runlist` front moves 12,582,912 bytes fewer than the
`block` front and the decomposed tail costs 193,536 bytes more. A cell is **not a fifth taxonomy
point**: its `execution_mode` is `coarse`'s CSV value and it travels as `blend_cell`. **The answer,
walked twice at 2048 and 4096: `C1 < C2 < C3 < C6`** on averages and minimums at both lengths, so
**`coarse` is C1** — chosen rather than inherited, recorded in every results artifact. The front
axis dominates (block ~1.5–1.6× faster than runlist); the tail axis separates cleanly only at 4096
— do not quote a 2048 tail effect. The ms table, the byte accounting and the two side findings:
[25 §4](../../docs/plans/transformer-layer-execution-studies/25-mode-rebuilds-and-results.md).

## The registry sweep

The GEMM-backed operators take their tiles and method from `kernel_registry`, which **raises** on a
shape nobody measured; `sweep/` makes a shape resolvable.

```bash
make registry-plan                # every (shape, candidate) it would measure. No NPU.
make registry-resolution          # every shape resolves. No NPU; this is the lit test.
make registry-writer-tests        # the writer's append-only guards. No NPU.
make sweep-families-tests         # the duplicated method table has not drifted. No NPU.
flock -x -w 1800 /tmp/mlir-air-npu.lock make registry-sweep
make registry-write               # fold the results into the registry. No NPU.
```

Per `(shape, candidate)` it builds the configuration, checks it through the same `opcheck.py`
comparison, times it, and keeps the fastest that passes. `FAMILY=` selects the width of the case
matrix (`baseline_768` is the one Phase D needs); `SWEEP_ARGS=` passes flags through.
`sweep_report` imports two functions from `registry_sweep`, so `registry_sweep` imports it inside
`main()` — the dependency has a direction.

**The sweep never re-measures a registered shape.** Those rows are what the ten shipped LLM
deployments resolve against; the writer refuses, the orchestrator skips, and the JSON is edited as
text so every pre-existing byte is identical by construction. The one edit an existing entry
accepts is **adding a method it has no row for**, and only to an entry whose `used_by` says this
sweep wrote it; a re-render that would change any method already present raises. It exists because
a shape can be registered and still unbuildable by the operator that needs it (`64×768×2304`, below).
Measuring such a shape again needs `SWEEP_ARGS="--remeasure-registered"`.

Two things about these rows differ from the model-deployment rows, both forced by the ladder
starting at 64: **`herd` is per-row** — `M % (tile_m × herd_m) == 0` cannot hold at `M = 64` with
8 rows, so short-sequence rows carry a per-method `herd` that `gemm_config()` hands back and
`builders/gemm_spec.py` merges into the recipe; **passing an explicit `herd_m` / `herd_n` to a
builder overrides the row and fails to build at the short end** — leave them `None`. **The
high-precision `atol` is carried forward at constant strictness**: the harness scales inputs by
`1/sqrt(K)`, so the published `1.5e-3` (the "≈2.5× the measured worst case" rule at `K = 8192`) is
a 3.3× *tightening* at `K = 768`; `sweep_measure.py`'s docstring has the three-point calibration.

## Running a profile: invocation and recovery

`[2026-08-12]` The unattended runner is one command; there is no daemon, crontab hook or reboot
orchestration (dropped on measured grounds,
[25 §7](../../docs/plans/transformer-layer-execution-studies/25-mode-rebuilds-and-results.md)).

```bash
systemd-inhibit --what=handle-lid-switch:sleep:idle \
  agents/scripts/devq.sh run --class measure -- \
    python3 study/run_profile.py --profile ladder --out-dir results/ladder-w1
```

`run`, never `submit`. The `systemd-inhibit` wrapper is not optional on this host — a lid close
suspends it mid-walk. Flags: `--family` retargets the profile to another case-matrix family (every
expected count re-derived, including `fused`'s bound); `--warmup` (1), `--samples` (3),
`--runs-per-sample` (1), `--study-id`, `--power-backend`, `--dry-run`, `--gate-only`, `--resume`.

**Prerequisites are checked, not documented.** `run_profile.environment_problems()` refuses at
start — a shell with no `pyxrt` or no `ml_dtypes` (both fail *late* otherwise, after every kernel
has compiled), and a working directory that is not this one (aircc and `KernelCache` write relative
to cwd and only this directory's `.gitignore` covers the debris). The power mode and the device lock
are refused by their own guards. The one thing no script can do: **the NPU power mode needs root
and is not persistent** — every reboot and `amdxdna` reload resets it, and at `Default` this host
measures ~15–20× slow end to end (82 ms per `hw_context` load against 3.7 ms at Turbo;
[15 §The NPU power mode](../../docs/plans/transformer-layer-execution-studies/15-environment-notes.md)):

```bash
sudo xrt-smi configure --device 0000:64:00.1 --pmode turbo
xrt-smi examine -r platform | grep -i "Power Mode"     # verify
```

**The gate.** `run_profile.gate()` verifies every row, then runs `study/distinguish.py`'s four
clauses (above) at every length where all four modes passed; a length where a mode is skipped is
reported, not failed — skips are derived from the refusing builders. `python3 study/distinguish.py
<root> [--seq-len N]` runs it standalone.

**Recovery.** A cold `ladder` walk is ~45 minutes, compilation dominating ~20×. Resume the same root:

```bash
python3 study/run_profile.py --profile ladder --out-dir results/ladder-w1 --resume
```

Every rung with a `passed` row is carried forward; **failed rungs are re-run on purpose** (a
retained failure is a claim about code that may no longer be there); structural skips are
re-derived every session. Without `--resume`, a root that already holds CSVs is **refused** — there
is no overwrite flag; wanting one is wanting a different `--out-dir`. What the run leaves behind:

| file | question |
|---|---|
| `<mode>.csv` | what was measured — rewritten after every rung, so a killed walk keeps what it got |
| `results_manifest.json` | is the walk complete — row counts derived from the profile, the power mode, the toolchain, the session attribution |
| `profile_run.json` | what this session did — plan, per-rung outcomes, reused-vs-measured counts, the devq job id |
| `walk_sessions.json` | who measured which rung — appended after every rung |

```bash
python3 study/resume.py results/ladder-w1 --profile ladder          # what a resume would do
python3 study/resume.py results/ladder-w1 --profile ladder --audit  # does the ledger match the files
python3 study/run_profile.py --profile ladder --out-dir results/ladder-w1 --gate-only
```

**Resuming does not make a walk complete that is not one.** Every carried-forward row is re-hashed
afterwards; a reused rung whose row moved is a defect. Two sessions are two populations and the
`walk` block says so; a splice across power modes is refused outright, across a toolchain or git sha
flagged.

## Things that will bite you

Environment traps are in [15](../../docs/plans/transformer-layer-execution-studies/15-environment-notes.md),
one line each here: the two CMake flags without which every NPU test is UNSUPPORTED and the suite
exits 0; `aircc` resolves to the **bundled** binary under `install-xrt/` regardless of PATH, so a
rebuilt compiler needs both `PYTHONPATH=build-xrt/python` and `PATH=build-xrt/bin`; the
compiler lit subset runs from `build-xrt/mlir/test`; generated artifacts (`*.o`, `air.mlir`,
`air_project/`, every `*_cache/`) land in THIS directory because everything runs with it as cwd —
check `git status` before committing, and add any new cache to `.gitignore` and `make clean` in the
same commit; `XRTBackend.compile` writes `air_project/` into the cwd, so two compiles from one
directory clobber each other; a compile inside a `measure`-class devq job holds the device lock for
its whole duration; never touch a build or cache directory while devq shows a `measure` running;
commits need `PRE_COMMIT_ALLOW_NO_CONFIG=1`. Every hardware make target takes
`PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR` explicitly; an unset one inside `--expect-failure` would read
as a caught fault if the flag inverted the exit status, which is why it does not.

**An object with no symbols still links.** `encoder.cc` and `addnorm_ffn.cc` emit nothing unless
`-DBUILD_FFN` and/or `-DBUILD_ADDNORM` is passed; Peano produces a valid empty `.o`, the link
succeeds, and the failure surfaces at dispatch. `compile_kernels.py` enforces a size floor and an
explicit per-object symbol list, and `compile_encoder` / `compile_addnorm_ffn` refuse both flags off.

**`encoder.o` and `addnorm_ffn.o` cannot share an ELF.** Both define `ffn_gelu_bf16` and
`ffn_eltwise_add_bf16_vector`. That is why `encoder.cc` is built twice: `addnorm` links the addnorm
half and `ffn` links `encoder_ffn.o`. `compile_kernels.py` checks that `encoder_ffn.o`'s FFN symbols
are present *and* its addnorm symbols absent.

**Two GEMMs of one method at different `tile_n` used to collide in one ELF, and across separate
ELFs returned zeros instead of failing.** Fixed in Phase E1; the history is here because the fix is
easy to undo. `gemm_builder.py` minted the MLIR symbol suffix and the `mm_*.o` filename from the
method alone (`_m32` / `mm_m32.o`) while `tile_n` arrived as `-DDIM_N`; but the private FuncOps'
memref types are functions of `tile_n`, so two GEMMs of one method at two `tile_n` are two
micro-kernels. *Loudly, in one ELF:* `stitch_elf` re-parses the declarations into one set —
`redefinition of symbol named ...`. Every shipped model lands on `tile_n = 128`; the study's FFN does
not (`768 % 512 != 0` at `herd_n = 4` settles on 96 against the up-projection's 128), so
`build_ffn_module` built at **no `baseline_768` point except `seq = 4096`**:

| seq | up-proj | down-proj | before E1 | after E1 |
|---|---|---|---|---|
| 64 … 2048 | `drain` t_n=128 | `drain` t_n=96 | collide | builds |
| **4096** | **`drain` t_n=128** | **`fused-cast` t_n=96** | builds | builds |
| 8192, 16384 | `fused-cast` t_n=128 | `fused-cast` t_n=96 | collide | builds |

*Silently, across ELFs:* `compile_gemm_mm` wrote the object named from the method, so the FFN's
`drain` at `tile_n = 128` and the o-projection's `drain` at 96 wrote the **same file** and each ELF
linked whichever landed last. Nothing failed; the ELF returned **exactly zero for 32 of every 128
up-projection columns** — 25 % of `ffn_up`, the other 75 % correct to `mean_rel_L1 = 1.2e-2` — which
the down-projection smeared into "54 % of elements wrong" at the layer output. D1 never met it
because no single operator holds two same-method GEMMs at different `tile_n`.
`gemm_builder.gemm_variant_names(tile_m, tile_n)` is now the single authority
(`(32, 128) -> ("_m32n128", "mm_m32n128.o")`); four ways to reintroduce it:

1. **Never spell `sym_suffix=` / `out_name=` by hand.** Call `compile_gemm_mm_variant(tile_m, tile_n, tile_k_l1)`.
2. **Never write `spec["tile_n"] = N`.** Use `gemm_builder.with_tile_n(spec, N)`; the bare
   assignment now leaves the module asking for an object nobody compiles (`qwen25_0_5b` was correct
   only by accident; E1 turned that into a visible link failure).
3. **`cache.prepare_air_project` globs `mm_m*.o`; do not turn it back into a list** — up to eight
   objects exist depending on which shapes the caller resolved.
4. **A module fingerprint does not notice compile ORDER** (above).

Audit a shared-naming change before spending hardware on it — the E1 gate's second leg is `make
verify` in all ten model directories and takes hours:

```bash
python3 agents/scripts/audit-gemm-object-links.py    # all ten. No NPU, no lock, ~a minute.
```

**`addnorm` caps at 104 rows at width 768, so the layer is row-blocked.** Three L3→L1 streams per
tile (x, residual, weight) against a column's two shim MM2S channels means one kernel call per tile
and L1 caps `rows` at `herd_x × (what fits)`: `addnorm_max_rows(cols, ...)` gives 80 post-add and
104 pre-add at `cols = 768`, 120 at 512. It counts *allocations*; aircc ping-pongs the DMA-fed
buffers on top, so a shape at the cap can still fail to place. Both `opcheck` rows run 64 rows.
**A dispatch argument is a whole BO**, so an operator consuming a 4096-row tensor in 64 bands needs
64 buffers cut on the host — why the block is four `run_sequence` calls.

**Multi-trip `addnorm` is permitted only at `herd_x == 1`, and the raise names the mechanism.**
Two trips used to corrupt the output (481–497 of 512 at `[8, 64]`, `herd_x=1`, unchanged by
hoisting the weight, by draining `output2`, or by disabling ping-pong): the shim feed order under
packet multiplexing, fixed by `air-fuse-packet-put-loops` (H) and extended to multi-column herds
(H9) — see the findings below. What remains is the shim's 16-BD wall: each fused put is its own
`aiex.dma_configure_task`, so 8 trips refuse at `herd_x=1` and 6 at `herd_x=8`. `build_addnorm_module`
raises above the measured boundary; `builders/block.py` keeps its 64-dispatch sequences.

**`-DADDNORM_PRE_ADD` changes numerics, not shapes.** Without it statistics run over `input` and the
residual is added after normalization; with it both run over `input + residual` and the two-output
form exports the raw sum through `output2`. Getting it backwards produces correctly-shaped, subtly
wrong activations; the compile driver asserts the two objects differ. **And `pre_add=` is a builder
keyword, not a flag flip**: `build_addnorm_module(pre_add=True)` links `addnorm_pre_add.o` from
`addnorm_ffn.cc` rather than `encoder.o` (no pre-add path), calls `fused_add_layer_norm_1outs`
rather than `_2outs`, and allocates three L1 buffers rather than four. Call
`compile_addnorm_kernel(pre_add=...)`; the objects have different names, so a mismatch is a link
error. What is *not* safe is the reference: `addnorm_reference` and `addnorm_pre_add_reference` are
separate on purpose. `addnorm_pre_add.o` is deliberately not the compile gate's
`addnorm_ffn_pre_add.o` (the full 11-symbol build). **Pre-add measures ~26× tighter than post-add at
a *higher* relative error** — `atol_required` 6.6e-4 for `64x768_pre_add` against 1.7e-2 for
`64x512`, `mean_rel_L1` 2.7e-3 against 1.9e-3 — because post-add finishes with `+ residual` in bf16
and a near-cancelling element carries an error set by the residual's magnitude. Do not read the
tighter number as the better datapath.

**`builders/gelu.py`'s docstring overstates the erf/tanh gap.** Measured over `x ∈ [-6, 6]` the worst
absolute difference is **4.7e-4** at `x = 2.70`, `atol_required` **3.6e-4** — inside any `atol`
here; in the D2 layer the swap moves `ffn_gelu` by 4.7e-4 and the output by 6.1e-4 against a loosest
`atol` of 1e-3. `pattern/test_reference.py` pins the activation by identity.

**`best.high` is the fastest method, not the one every builder can use.** `build_qkv_proj_module`
folds its three-way C split into `fused-cast`'s separate cast launch and can build only that method;
`drain`'s `tile_m = 32` beats `fused-cast`'s 64 on short sequences, so the QKV lookup asks for
`fused-cast` by name (`resolve_gemm_spec(..., method=...)`). So **"the shape is in the registry"
and "the operator can build it" are different claims**, and `make registry-resolution` checks the
second through each role's own builder. `64×768×2304` was first registered with `drain` and `direct`
only — every `fused-cast` candidate failed (`mean_rel_L1 ≈ 0.46`, ~30 % wrong) — so `qkv_proj` at
`seq = 64` raised `KeyError` while a generic lookup called it a pass. **A legal `cast_tile_n` is not
necessarily a correct one**: the cast launch walks each of 8 workers' contiguous chunk in
`cast_tile_n` steps; at that shape the chunk is 18432, the default 2048 divides it exactly, and two
of nine sub-tiles come back **zero**; at 1024 it passes at 446 GFLOP/s. `cast_tile_n` is a swept knob
(`CAST_TILE_N_PREFERENCE`, fused-cast only); `drain` still wins the tier at 945.

**Multi-segment designs cannot use the xclbin output path.** Every `air.launch` lowers to its own
`aie.device`, and the xclbin path names one instruction blob, so a second segment collides: `edge
'air.insts.bin' produced duplicate output path`. Use `output_format="elf"`, as the shipped
multi-launch builders do. (The 2026-08-10 multi-launch xclbin packaging exists but in-stream
`load_pdi` faults the firmware — `offload` above.)

**The GEMM's error is ~1 % of the output's own magnitude, not one bf16 ULP.**
`-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16` puts the multiply in block floating point, and
`mean_rel_L1` lands at 9.3e-3 (registry) / 9.9e-3 (here) rather than ~2e-3, so an `atol` is
meaningless without the operand scale it was measured at; `opcheck.py` uses the registry sweep's
scale. **That flag must be a `-D`** — visible before `<aie_api/aie.hpp>`; both kernels
`static_assert` on it under `BUILD_FFN`.

**`copy_O_tile_rows` is numerically a no-op, and deleting it hangs the design.** A KV block
entirely above the causal diagonal runs no matmul, so without the passthrough the consuming DMA
never sees its buffer descriptor complete: `ERT_CMD_STATE_TIMEOUT`. It lives behind
`-DCAUSAL_ROW_HELPERS` with `store_row_value` and `copy_row_values`; `mha_out_proj` links all three
into every causal variant without calling them, so a block-skipping variant can be added without
re-deriving the flag set.

**FlashAttention's `-D` flags are per *tile*, and a mismatch hangs rather than fails.** `-Dlqp` is
the Q tile (`parallel_seq / num_q_tiles`); `-Ddk` / `-Ddv` are the `lkp`-sized tile, `-Ddk_full` /
`-Ddv_full` the full head dimension. `builders/mha_attention.py` derives both from one config dict
and rebuilds with `force=True`, because a shared working directory may hold an `attn_npu2.o` built
for another shape.

**`-DDEBUG_AIE_KERNELS` needs a value.** The sources test `== 0` / `== 1`; a bare define expands to
nothing and fails to compile. Pass `=0` (pass input through) or `=1` (pass residual through).

## Compiler-phase findings with their hardware gates here

Each compiler phase's full record — defect, mechanism, commits, lits, measured numbers — is in
[16](../../docs/plans/transformer-layer-execution-studies/16-compiler-changes.md). What follows is
what a user of this directory needs, plus the measurements that live only here.

### Phase H — the multi-trip miscompile was the shim feed order ([16 §1, §2, §4](../../docs/plans/transformer-layer-execution-studies/16-compiler-changes.md))

The two-trip `addnorm` corruption (481/512 at `cols=64, rows=8, rows_per_call=4`) was **not**
ping-pong — `--omit-ping-pong-transform=all` reproduced it identically — but `air-dma-to-channel`
hoisting each L3 DMA into its own put loop while packet multiplexing serialized whole channel after
whole channel against a BD ring expecting per-iteration interleave; `air-fuse-packet-put-loops`
restores the interleave (gated here by `check-addnorm-multitrip`'s `inside` variant). H2 made
`func.call` visible to dependency analysis through `llvm.readonly` / `llvm.writeonly` on
`llvm.emit_c_interface` callees; an unannotated operand stays unknown. The ping-pong safety proof
**skips, never refuses** (H1s), and its subject is exactly the buffers the rotation privatizes.
Two footguns recorded only here: **the two-trip fixture shape fully UNROLLS under ping-pong
labeling** (trip count == unroll factor), so `air-ping-pong-transform`'s dependency rebuild never
runs on it — a fix living only there is untestable at 2 trips; and **skipping unprovable candidates
silently drops every unannotated external-call loop to single buffering**, including the decode
GEMV loop (12.4 → 7.8 tok/s measured for a global disable) — `make verify` gates correctness, not
tok/s; annotate input tiles `readonly`, leave a read-and-written accumulator unannotated. The
`no_consumer` case in `label_ping_pong_external_call_proof.mlir` guards against the refusal of
never-read buffers (attempt 2) returning.

### Phase J1 — the norm-dispatch collapse, measured blocked, then closed ([16 §5, §6](../../docs/plans/transformer-layer-execution-studies/16-compiler-changes.md))

J1 tried to lift the one-trip guard and collapse the two normalization points (128 of `coarse`'s
131 entries) into one launch each: 4096×768 over the 8-column herd, 64 trips of 8 rows. The walk
(pre-add, unannotated callee, rtol 1.6e-2 / atol 2e-3, zero permitted mismatches, NPU2) — 16 §5
points here for the L2-staged arms:

| shape (trips × rows_per_call, cols, herd_x) | result |
|---|---|
| 2×4, 64, herd 1 (the fixture's shape) | exact |
| 2×8, 768, herd 1 | exact |
| 8×8, 768, herd 1 | refuses to compile: shim BD exhaustion (16-BD cap, `aiex.dma_configure_task`) |
| 2×4, 64, herd 8 | 4070/4096 mismatched |
| 2×4, 64, herd 8, weight DMA hoisted | 4039/4096 mismatched |
| 2×8, 768, herd 8 | 97,726/98,304 mismatched |
| 64×8, 768, herd 8 — **the J1 target** | compiles; 3,130,958/3,145,728 mismatched |
| weight staged through L2, herd 8 | placement failure: `no ShimNOCTile has sufficient DMA capacity` for the weight put |
| weight via L2 (broadcast OR per-column replica), herd 4 | routing failure: `'aie.connect' op … targets same dst as another connect op` on the first core tile |

Three walls: the fusion pass did not fire on multi-column herds (output IR byte-identical, pass 026
vs 027 — fixed by H9); where it fires, each put is its own `aiex.dma_configure_task` against 16
shim BDs; and staging the weight through L2 trips the two placement/routing failures above. Every
failing shape reproduces from a plain `XRTRunner.run_test` of `build_addnorm_module`.
`[2026-08-21]` Closed: J7a reaches the same collapse by packing x and residual into one fetch and
never entering the packet path.

### Phase H9 — packet put-loop fusion reaches multi-column herds ([16 §5](../../docs/plans/transformer-layer-execution-studies/16-compiler-changes.md))

`air-dma-to-channel` wraps each per-tile put loop in its **own** `scf.parallel`, so "walk the
parallel body" is vacuous; the pass *sequentializes* each eligible wrapper into ascending
per-iteration clones (the order `airrt-to-npu` unrolls to anyway — compare `dma_configure_task_for`
in any pass_058 dump) and fuses the flattened block; wrappers with a live result token are expanded
too, the token replaced by an `air.wait_all` before its earliest user. Gated by the `multicolumn`
variant. Numbers only here: the `herd_x=8`, 2-trip shape went from 4070/4096 wrong to exact
**4/4** repeat runs; 4 trips × herd 8 exact **3/3**; **7/7** exact across the 2- and 4-trip probes;
the BD wall measured at cols 64, `rows_per_call` 4: **rows 160 compiles, rows 192 refuses** (5 trips
is the deepest width-8 depth). Why cross-iteration fusion is safe: each non-broadcast channel's
per-column endpoint sits on its own column's shim queue (`channel_1_c`, `channel_2_c` on column c's
queue 0), and the fused order gives column 0's queue the ring's exact expectation (`w, x0, r0` per
trip) where it got `w, w, x0(all), r0(all)`. **The residual ordering the argument does NOT close**:
tiles 1..7 receive the weight from column 0's queue and x/res from their own; their relative arrival
is a timing discipline every shipped single-trip multi-column design already stands on, not a
proven order. If a multi-column multi-trip shape ever fails intermittently with corruption confined
to the weight-adjacent buffers, start here.

### Phase J7a — the norm-tail pipeline, and the two spec premises full compilation falsified ([16 §11](../../docs/plans/transformer-layer-execution-studies/16-compiler-changes.md))

`builders/norm_tail.py`: add → LayerNorm → gamma as three herds in one segment on L1→L1 channels
(`norm_tail_a2b`, `norm_tail_b2c`), x and residual in ONE packed L3 buffer, two shim MM2S streams
per column, **zero packet-typed channels**, no placement and no depth declared. Measured on NPU2:
`128x768` and `4096x768` PASS at rtol 1.6e-2 / atol **5e-2**, `mean_rel_L1` 3.590e-3 / 3.620e-3;
`check-norm-tail` also enforces `mean_rel_L1_max` 1.688e-2; `check-norm-tail-structure` goes through
the REAL aircc (three herds of 8 on 16 core-to-core flows, ≤ 2 shim inbound per column, zero
`npu_dma_packet`); `check-norm-tail-fault`. Two spec premises fell to FULL compilation — both probes
had stopped at `air-opt` (16 §11 still states the strided-callee plan; this is the measured
outcome): **the strided-callee route does not reach hardware** — `air-to-aie` normalizes every
external callee signature to the identity layout (`AIRToAIEPass.cpp`, the `normalizedInputs`
block) and `memref.cast` refuses strided-with-offset → identity, so no strided operand reaches an
external kernel at any offset; the add is `elementwise_add`'s direct-codegen stage body instead.
**Plane-major packing cannot be programmed at the block's shape** — `[2, rows, cols]` needs a plane
stride of `rows*cols` against the shim `aie.dma_bd` cap of 2^20: at 4096×768 that is 3,145,728 and
aiecc refuses (`Stride 2 exceeds the [1:1048576] range`); at 128 rows it compiles and is exact. The
default packing is per-row pairs `[rows, 2, cols]` (contiguous bands, max stride `2*cols`);
`plane_major=True` is the opt-in `fused` needs (a contiguous producer writes plane 0 at offset 0),
bounded to 1365 rows at width 768. Footguns: aircc ping-pongs BOTH of `stage_add`'s tiles — at
`rows_per_call=8` that is 24+24+12+12 KiB and the allocator refuses; the default is 4 with the
measured ×2 in the L1 check. The asymmetric inputs are load-bearing (`LN(x+x) == LN(x)`):
`prepare_norm_tail` draws x standard normal and residual `normal(0.75, 1.5)`. The pipeline's cost
over the fused kernel is one extra bf16 rounding: 3.6e-3 against 2.7e-3. **The norm statistics were
the round-3 finding**, fixed in the C kernel: `layer_norm_rows` shipped a bf16 row sum and one-pass
variance, and on a row with mean 8, σ 0.25 the cancellation floored the variance at zero and ~700 of
768 elements failed; `layer_norm.cc` now keeps f32 two-pass statistics — the `layer_norm` rows went
from `mean_rel_L1` 2.0e-3 to 8.1e-5 and the pipeline's from 4.4e-3 to 3.6e-3 (`128x768_offset` pins
the regime). `[2026-08-11]` Both dispatched `addnorm` variants followed: the offset rows measured
`mean_rel_L1` 1.390e-3 / 1.409e-3 with `atol_required` 0.0, against the one-pass kernel's 22.2 /
33.1 collapse, the zero-mean rows refreshed to 1.486e-3 / 1.963e-3
([23](../../docs/plans/transformer-layer-execution-studies/23-rules-and-open-items.md)). Only
`encoder.cc`'s staged forms, dispatched by no builder, keep one-pass statistics.

### Phase J7b — `ffn_accum`, the compiler-formed accumulator ring ([16 §7, §12](../../docs/plans/transformer-layer-execution-studies/16-compiler-changes.md))

`builders/ffn_accum.py` writes the FFN down-projection as the NAIVE K loop — fetch C, call the
in-place `ffn_matmul_bf16_bf16_up_proj`, store C — and `air-hoist-dma-in-accum-pattern` lifts the C
pair: K-loop data movement **4 → 2**, zero packet-typed channels, full aiecc compile
(`check-ffn-accum-structure`; the numbers are identical whether the ring formed or not, so the
structural clause IS the gate). Four walls, each in the builder's docstring: (1) three per-core
input streams exceed the 2-per-column shim MM2S budget; (2) staging one operand through L2 exposes
the core's two S2MM channels (the 1×1 `air-opt` probe never saw it: its streams had been silently
packet-upgraded) — A and B share ONE memtile feed channel, two gets per K step, host-pre-tiled; (3)
the spec's pre-loop zero-and-STORE is a second shim S2MM stream per column and `aie-place-tiles`
refuses — the zero runs on the L1 tile under a `k == 0` guard, no DMA, so `y`'s initial DDR contents
never reach the result; the same pass refuses the C pair's slots at **`herd_x=6` with free
capacity; `herd_x=4` places**; (4) a per-iteration L2 read offset is silently frozen at 0 past two
trips (H10 now refuses it; reproduce from the retired `agents/probes/probe_ffn_accum_bd_offset.py`
at tag `pre-cleanup-20260821`) — **both operands advance on the L3 side**. Measured at `64x3072x768`
(**herd 4×1, `tile_n` 192, `tile_k` 32, 96 K steps**): `mean_rel_L1` 1.417e-2, **`abs_err_max`
1.831e-3**, `atol_required` 1.383e-3, zero mismatches over 49152 at the GEMM tier's 5e-3 — a
**3.6×** margin. The relative error is an order above the other GEMM rows and that is the ring's
honest cost: the in-place kernel's C is bf16, so the running sum rounds 96 times where a
drain-to-f32 GEMM rounds once. `check-ffn-accum` / `-fault` / `-structure`; `run_npu2_ffn_accum_peano.lit`.

## Licensing

Mixed, deliberately. `compile_kernels.py`, the `Makefile`, the `.lit` files, the builders, the
modes, the harness and `kernels/transpose.cc` are new work under MIT, matching the rest of
`programming_examples/`. The seven ported sources under `kernels/` (`encoder*.cc`,
`addnorm_ffn*.cc`, `elementwise.cc`) carry Apache-2.0 with AMD copyright: their numeric bodies are
substantially carried over from the source they were ported from, and attribution is preserved even
though this project is MIT. Both projects are AMD-copyright, so Apache-2.0 files live here without
conflict.
