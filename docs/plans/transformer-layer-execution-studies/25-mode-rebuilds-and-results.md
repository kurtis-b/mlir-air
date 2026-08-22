# 25 — Mode rebuilds and results

Consolidated 2026-08-22 from docs 25 (first ladder result), 26 (mode-rebuild feasibility), 28
(coarse blend space), 29 (offload N streams), 30 (coarse cells built), 33 (memcpy bandwidth
scoping), 34 (Phase G scoping + G1), 38 (iron encoder-pipeline reference), 50 (coverage-sweep
costing) and 51 (blind-check census); their full text is at git tag `pre-cleanup-20260821`. The
standing four-mode numbers live in [27](27-common-ladder-result.md) (512/1024, two walks),
[32](32-cost-decomposed-ladder.md) (cost-decomposed and post-flip walks) and
[54](54-first-full-profile-and-decoder-families.md) (first full profile, decoder families); this
doc links to them and does not restate their tables. Code docstrings cite the old filenames
(`26-…` §4/§5, `28-…`, `29-…` §What this does not do / §The hardware verdict, `30-…`, `50-…` §7);
the section headings below carry those names.

---

## 1. The first study result: the `baseline_768` sequence ladder (was doc 25) — RETRACTED

**`[retracted 2026-08-08, unreproducible as of 2026-08-09]` Do not cite the crossover or the
slopes.** (1) It ranked four *implementations*, not the four modes — the taxonomy was corrected on
2026-08-08 ([03 §The taxonomy](03-measurement-model.md)) and none of the four matched it. (2) Its
explanation cannot be re-tested: the slopes split on attention placement (host-attention modes
1.23–1.27 vs device-attention 1.03–1.17), and as of 2026-08-09 all four modes run attention on the
device. **What survives:** the measurement itself (16 rungs, walked twice, every rung validated) and
the reason `attention_path` became a per-row covariate. The one clean cross-mode number is DRAM
traffic at 4096 — `runlist` 190,513,152 bytes vs `offload` 970,457,088 — which differs in the
taxonomy's own variable only ([03](03-measurement-model.md)).

**How measured `[2026-08-08]`.** `study/run_ladder.py`, one child process per rung (rule in
[23](23-rules-and-open-items.md); the first attempt without it produced five false failures),
`--samples 3 --warmup 1`, `baseline_768` (`emb 768`, `ffn 3072`, `12 heads × 64`), bf16, under
`/tmp/mlir-air-npu.lock`, quiet host, compilation outside the clock, zero mismatches against the
FP32 golden at every rung. Two independent 16-rung walks: `results/j3_ladder_iso2/` authoritative
(rows carry `attention_path`), `results/j3_ladder_iso/` the replicate; each one CSV per mode, schema
v1, plus `report.md` from `study/ladder_report.py`. (Gitignored.)

| mode | 512 | 1024 | 2048 | 4096 | slope | attention |
|---|---|---|---|---|---|---|
| `fused` | 46.7 / 45.0 | 97.7 / 99.0 | 197.6 / 195.6 | 524.9 / 536.2 | 1.15 / 1.17 | device |
| `coarse` | 53.0 / 48.6 | 106.7 / 98.6 | 204.5 / 214.6 | 465.1 / 455.2 | 1.03 / 1.08 | device |
| `offload` | 57.2 / 56.0 | 117.3 / 124.3 | 274.4 / 273.9 | 782.1 / 813.6 | 1.26 / 1.27 | host |
| `runlist` | 59.8 / 61.7 | 130.3 / 136.9 | 303.8 / 292.8 | 811.2 / 819.3 | 1.25 / 1.23 | host |

ms, walk 1 / walk 2. Run-to-run spread 0.2 % to 9.0 % (worst on `coarse`). Structural columns
bit-identical across both walks for all four modes. Slope = least-squares `log(latency)` vs
`log(seq_len)` over four rungs — "closer to linear/quadratic", not a model.

Margins: walk 1 `fused` by 13.4 / 9.2 / 3.5 % at 512/1024/2048, `coarse` by 12.9 % at 4096; walk 2
`fused` by 7.9 % at 512, **`coarse` by 0.4 % at 1024**, `fused` by 9.7 % at 2048, `coarse` by 17.8 %
at 4096. So at 1024 the two were indistinguishable at this sample count (the first draft claimed
`fused` won 1024; the second walk refuted it); the crossover between 2048 and 4096 was the largest
effect (12.9 % and 17.8 %, same direction). `fused` = 1 submission, 23 herd launches, 12 sync
boundaries at every length; `coarse` = 4 submissions, sync 59 → 396.

Structure (what `phase_e_checks.py`'s four distinguishability clauses read): `coarse` 4 subs, herd
33 → 146, sync 59 → 396; `offload` 6, 18 → 19, 18 → 19; `runlist` 5, 67 → 404, 58 → 395; `fused` 1,
23 → 24, 12 → 13. Not supported by it: any NPU-efficiency claim (host-attention modes' clock
covers host torch), any distribution (three samples, two walks), "`fused` is the fastest mode",
any power/energy claim (no sensor on this platform).

Reproduce: `flock -x -w 7200 /tmp/mlir-air-npu.lock python3 study/run_ladder.py --modes
coarse,offload,runlist,fused --seqs 512,1024,2048,4096 --out-dir results/j3_ladder_iso2 --study-id
j3-ladder --samples 3`, then `study/ladder_report.py … --md`. Warm ELF caches: ~90 s of device
time; cold ~45 min (compilation). **Nothing CPU-heavy may run alongside** — that inflated an earlier
table by 1.55×.

---

## 2. Rebuilding the four modes to the corrected taxonomy: three feasibility spikes (was doc 26)

`[2026-08-08]` [03](03-measurement-model.md) was corrected: the modes span **reconfiguration cost
against DRAM traffic**, not "who sequences the work", and every mode as implemented was wrong
against it. No spike edited a repository file. Spike B: 33 device jobs, 976 s (16.3 min) of lock
inside a 45-min box, turbo verified via `xrt-smi`. Spike A: 0 device jobs. Spike C: compile-only,
every `aircc` via `agents/scripts/devq.sh submit --class build`. Scratch:
`/home/cj/.claude/jobs/e75c34c9/tmp/{spikeA,spikeB,spikeC}/`.

### 2.1 (doc 26 §1) J2's blocker does not exist — `attn_output` passes on the first configuration

[16](16-compiler-changes.md)'s J2 row said `attn_output` (4096×4096×64) "timed out on the one
configuration tried, out of 828". Measured: `drain`, `tk2=256 tk1=32 tn=16`, herd 8×4 → **0 /
262,144 mismatches**, `mean_rel_L1` 9.417e-3, `abs_err_max` 7.324e-4 vs `atol` 2.121e-3 (3.46×),
1179.3 µs, 1820.9 GFLOP/s. Confirmed via a second entry point (`bf16_in_bf16_out/run.py:1088`
`__main__`: `Latency (us): 1199.4 … mean_rel_L1=9.417e-03 | abs_err max=7.324e-04 … PASS!`,
byte-identical statistics). All three methods work: `direct` 2344.8 GFLOP/s at `mean_rel_L1`
1.542e-2; `fused-cast` `tn=16 ctn=2048` 1586.3 GFLOP/s.

### 2.2 (doc 26 §2) `attn_scores` had no artifact — now it does

The claim "already passes on hardware" entered with `7f27599e` carrying prose only. Re-established:
`attn_scores` 4096×64×4096 `drain tk2=64 tk1=32 tn=128` herd 8×4 → 0 / 16,777,216, `mean_rel_L1`
9.386e-3, 2901.4 µs, 740.1 GFLOP/s. `tk2=64` is forced (K=64 admits no other L2 tile).

### 2.3 (doc 26 §3) "828 legal configurations" is unsourced

Enumerating from `bf16_in_bf16_out/run.py:63-66` plus the cast-launch divisibility at `:734` over
`sweep_families.py`'s knob lists gives **1584** for `attn_output` (fused-cast 1296 / drain 144 /
direct 144) and **660** for `attn_scores`. No sub-product equals 828. Do not cite it.

### 2.4 (doc 26 §4) `runtime_loop_tiling_sizes` is DECISIVE, not inert — the retraction, re-measured

The spike's compile-only entry claimed the backend knob was inert and the documented
`fused`/`mha_out_proj`/`block.py` settings conflict false. **Retracted 2026-08-08** the same day by
hardware, and **re-measured 2026-08-12**; the retraction stands at 8/8 vs 0/8. The probe that
refuted it, `agents/probes/probe_backend_preset_hardware.py` (hardware arms 08-08 and 08-12,
`--compile-only` artifact arms 08-12), was retired in the 2026-08-21 cleanup; it is at tag
`pre-cleanup-20260821` and `agents/probes/README.md` records it against this section.

**2026-08-08 factorial**, `mha_out_proj` @4096×768, twelve heads, non-causal, one process per arm
on an exclusive device (devq **58–63**):

| tiling | ping-pong | n | result |
|---|---|---|---|
| `[1,1]` | OFF (shipped) | 2 | PASS — 0 / 3,145,728, `mean_rel_L1` 5.3348e-02, `atol_required` 8.7061e-03 vs `atol` 2.5e-02 (2.87×) |
| `[1,1]` | ON | 1 | PASS — byte-identical statistics |
| `[2,2]` | ON | 2 | `ERT_CMD_STATE_TIMEOUT` |
| `[2,2]` | OFF | 1 | `ERT_CMD_STATE_TIMEOUT` |

So the tiling is decisive and `omit_pingpong` irrelevant at this shape. The conflict the three
files document is REAL: FlashAttention @4096 requires `[1,1]`, the wide GEMMs are built at `[2,2]`,
one ELF is one aircc invocation; only the stated *reason* needed correcting (tiling, not
`omit_pingpong`). The outcome was a third branch neither "placement failure at best" nor "wrong
numbers at worst" covered: **it hangs**.

**The refuted compile-only reasoning (kept as a record).** Diffing aircc's `air_project/aie.air.mlir`
`[1,1]` vs `[2,2]` (channels renumbered, lines sorted): `mha_out_proj` @4096 PP-off 280 `aie.dma_bd`
/ 44 `shim_dma_allocation` / 628 `aie.buffer` / 424 `aie.lock`, "identical"; `fused_tail` @1024
PP-on 600/98/524/808; `qkv_proj` @4096 PP-on 304/68/236/376; the raw `mha` diff 98 lines of
`@channel_17`/`@channel_19` renumbering. `omit_pingpong` compiled both halves both ways: `mha/gemm`
PASS 105.2 s at 344 bd / 660 buf / 488 lock; `tail/attn` PASS 76.0 s at 456 / 424 / 736; zero
packet-typed channels. The ELF waited at
`/home/cj/.claude/jobs/e75c34c9/tmp/spikeC/run_mha_gemm_4096/probe_mha_gemm.elf`.

**`[2026-08-12]` re-measurement (queue item 24).** Two things had weakened the 3/3: for R1 the same
two settings are byte-identical through `aie.air.mlir`, `npu.air.mlir`, `.ctrltext` and `.pdi`
(queue item 21 — inert *there*), and wall 7 ([31](31-resident-tail-r1-record.md)) showed this
composition class hanging nondeterministically (`PASS/TIMEOUT/PASS/TIMEOUT/PASS`; a clean 3-vs-3
arises ~3 % of the time against a fair coin).

Compile-only first: four `--class build` compiles, two per arm, each in an empty cwd (`attn` =
`[1,1]` + `omit_pingpong`; `t22pp` = `[2,2]` + `omit_pingpong`), devq **279–282**:

| artifact | `[1,1]` | `[2,2]` | stable within arm? |
|---|---|---|---|
| `aie.air.mlir` | 640,697 B | 640,697 B | **NO** — differs compile-to-compile |
| `placed.air.mlir` | 107,593 B | 107,593 B | **NO** |
| `npu.air.mlir` | 1,507,775 B | 1,306,762 B | length yes, bytes no |
| `at_attn_seg.pdi` | 352,480 B | 352,480 B, different content | yes |
| `op_matmul_seg.pdi` | 138,160 B | identical | yes |
| `…at_attn_seg_sequence.bin` = `.ctrltext.0` | 402,448 B | 356,368 B (−11.4 %) | yes |
| `…op_matmul_seg_sequence.bin` = `.ctrltext.1` | 88,080 B | 29,712 B (2.96×) | yes |
| `…main_mha_out_proj.bin` = `.ctrltext.2` | 1,026,496 B | 922,048 B (−10.2 %) | yes |

The knob is not inert at this shape. The `attn` ELF reproduces the 2026-08-08 spike-C ELF
byte-for-byte. **And `aie.air.mlir` is not byte-reproducible**: two compiles of one preset differ
by 94 lines (`attn`) and 98 lines (`t22pp`) — the same size as the 98-line diff the spike dismissed
as renumbering. That diff was compiler noise. **Rule: settle inertness on `.bin` / `.pdi` /
`.ctrltext`, never on an IR dump, always with a same-arm repeat as control.**

Hardware: ten runs, five per arm, interleaved `attn, t22pp, …`, one process each, empty cwd,
`devq.sh submit --class measure`, Turbo verified — devq **283–292**: `attn` 283/285/287/289/291 all
PASS with 0 / 3,145,728, `mean_rel_L1` 5.3348e-02, `atol_required` 8.7061e-03 (identical to the
bit); `t22pp` 284/286/288/290/292 all `ERT_CMD_STATE_TIMEOUT`, `ctx_pc 0x28B060AD`. **5/5 and 0/5,
no mixed arm**; pooled with 58–63: **`[1,1]` 8/8 PASS, `[2,2]` 0/8**. Fisher exact p = 0.0079 on
the ten, 1.6e-4 pooled. H3 (inert) refuted by artifacts; H2 (both arms one flaky distribution)
refuted — wall 7's signature is a mixed arm, and a race does not reproduce `atol_required` to five
figures eight times; H1 survives. Repeat arithmetic: at a 50 % flake rate five runs show a mixed
record with probability 93.75 % (three: 75 %); zero failures in five leaves a 95 % upper bound of
45 % on a hidden failure rate (31 % pooled over eight). Five is right for "is this wall 7?", not for
"is this ever flaky?".

**Mechanism, corrected.** The `air-opt-shim-dma-bds` early-exit (`AIRDependencyScheduleOpt.cpp:8287-8291`)
is real and does **not** fire for this design: the pass lowers `air.launch` to `scf.for` first
(`AIRLaunchToScfForPattern`, `:8275`) and collects the shim band after, so `mha_out_proj`'s 2-D
attention launch grid and projection launch **are** the band it tiles by `findLargestFactor(trip_count, 2)`.
R1 is a design where the early-exit does fire — same knob, inert there, decisive here. The honest
general statement is **design-dependent**; the cheap test is `probe_backend_preset_hardware.py
<preset> --compile-only` twice per arm, diff the `.bin`/`.pdi` rows.

Standing: the `fused.py`, `block.py`, `mha_out_proj.py` conflict comments stand (`fused.py` and
`block.py` quote the 3/3 as history); `mha_out_proj.py`'s comment was corrected (it asserted the
`aie.air.mlir` sameness and the early-exit). The `fused.py:63-74` / `mha_out_proj.py:111-130` /
`block.py:90` prose is NOT to be softened — substitute the measured basis and drop `omit_pingpong`
from the stated reason.

### 2.5 (doc 26 §5) A device softmax kernel already exists, and this port builds it

`programming_examples/softmax/softmax.cc:316` defines `softmax_bf16` (single-shot, LUT exp), driven
by `softmax.py`, gated by `run_npu2_makefile_peano.lit` (`REQUIRES ryzen_ai_npu2, peano`, `--n 2048
--herd-n 4`). Behind `-DSOFTMAX_STREAMING`, `softmax.cc:241-311` exports `init_softmax_scale_buffer`,
`partial_softmax_rows_bf16` (runtime `row_width`, `num_rows` — shape-generic),
`normalize_softmax_rows_bf16`, `copy_softmax_scale_bf16`; init → partial → normalize is a plain
row-wise softmax. `compile_kernels.py:172-183` lists `softmax_streaming.o` via
`ek.compile_softmax_streaming()` (`llms/shared/infra/external_kernels.py:471-484`); `README.md:135`
documents the opt-in. Gaps as first read: `softmax_bf16` subtracts no row max and hardcodes
`zero_vectorized<bfloat16, 1, 256, 16>` (wide rows must use the streaming family); no
`builders/softmax.py`, no `opcheck_specs.py` row, no fault control.

**`[2026-08-09]` BUILT**: `builders/softmax.py`, three measured `opcheck_specs.py` rows,
`run_npu2_softmax_peano.lit` passing with its negative control. Corrections: the row-max gap belongs
to the single-shot kernel — `partial_softmax_alias_bf16` keeps a running row max, so streaming is
the safe path; `SM_LOG2E` is the base conversion of an `exp2`-based `exp` (`exp2(x·log2(e) − m)`),
exactly right for plain softmax; for attention `partial_softmax_alias_bf16` takes `scale`, so
`1/sqrt(head_dim)` is `scale = SM_LOG2E / sqrt(head_dim)` plus a wrapper parameter. Two rules the
build cost: **one role per L1 buffer** (normalizing back into the DMA-destination buffer returned
the input unchanged at all three shapes); **the standard injection target `(rows-1, 0)` does not
discriminate for a normalization** — `+2.0` on a low-probability element moved the tensor 1.06e-3 /
7.43e-3 against `atol` 7.5e-3, and at 512×512 the injected `abs_err_max` equalled the clean run's;
the target is now the last row's argmax per shape, clearing `atol` by 12–18×. The corrected-`runlist`
softmax phase is builder + gate over an existing kernel, not kernel development.

### 2.6 (doc 26 §6) `fused` could not build its own SPECS shape — confirmed red, fixed

`opcheck_specs.py:782-790` pinned the fused row at seq 4096; `build_norm_tail_module(seq_len, emb,
plane_major=True)` raises `ValueError: plane_major packing needs a plane stride of rows*cols
(4096*768 = 3145728), over the shim aie.dma_bd cap of 1048576` (`norm_tail.py:262-273`, 1.5 s);
`fused.py:37` already said "BOUNDED TO 256..1024"; `compile_fused_artifacts` (`fused.py:492-495`)
rebuilds every module even with `run_only=True`, so the Aug-5 cached `fused_tail_4096x768x3072.elf`
cannot rescue it. **`[2026-08-08]` CONFIRMED RED, FIXED**: row moved to **1024**;
`run_npu2_fused_peano.lit` passes both recipes, 10/10 stages, `mean_rel_L1` 1.756e-2,
`atol_required` 5.813e-2 vs 1e-1 (1.72×). At 1024 the FFN down-projection's fastest high-precision
row is `drain` (at 4096 `fused-cast`), exposing no f32 C scratch, so the stitched tail takes 11
whole-tensor args instead of 16. Totals at 1024: `submissions 1 entries 3 air 11 herd 23 sync 19
bytes 56626176` (air 16 → 11, bytes 184,025,088 → 56,626,176, sync unchanged at 19). The old
`sync 19` vs `coarse` 402 comparison spans two lengths and is withdrawn.

### 2.7 Lane verdicts

| Lane | Question | Verdict | Cost |
|---|---|---|---|
| B — `runlist` | attention interior on device? | feasible-with-changes, scope shrinks | 33 device jobs, 976 s |
| A — `offload` | matmul loop bounds from a runtime parameter? | blocked as posed; absorbed into [03](03-measurement-model.md) by `e58a2170` | 0 device jobs |
| C — `fused` | one xclbin, no DRAM between operators? | split: documented blocker REAL (§2.4), `air-fuse-channels` a second real one, half 2 capacity-bounded | compile-only + 6 device jobs |

**B — attention on device.** 32 configurations: 18 passed, 10 `failed_precision`, 4 `failed_build`
(`attn_output` 24 tried / 14 passed; `attn_scores` 8 / 4). Checkpoints: signature-keyed JSON under
`/home/cj/.claude/jobs/e75c34c9/tmp/spikeB/results/`. Verification reused from
`sweep/sweep_measure.py`: `XRTRunner._check_outputs` `np.isclose` over the full output at 0 %
allowed mismatch. Ladder (µs), `attn_output` drain / direct, `attn_scores` drain: 512 72.7 / 84.8 /
103.2; 1024 153.6 / 117.4 / 225.5; 2048 351.9 / 275.3 / 738.1; 4096 1179.3 / 915.8 / 2901.4 — 8/8
for `attn_output`. Failure clusters: **`herd_n=1`** (`tn=32` or `64`) at N=64 returns essentially the
host-written buffer (`mean_rel_L1` ≈ 1.00, 231,517 / 262,144 wrong) at a flat 6,144,000 µs per
iteration — the hang signature, matching `sweep_families.py`'s recorded `herd_n=1` "placed but FAILED
AT RUNTIME" at N=896 (almost certainly J2's timeout); **`tile_n=8`**: `drain`/`fused-cast` fail to
build at `mm_aie2p.cc:161` `static_assert(n % (2 * t) == 0)`, `direct` builds and returns garbage
(262,144 / 262,144, 298–597 ms) — the microkernel narrows the legal space below `run.py`'s asserts.
`attn_scores direct` fails the gate but not hardware: `mean_rel_L1` 9.46e-3, 3,528 of 16.7 M outside
the low tier's unscaled `atol=4e-3` (needs 6.72e-3) while passing the high tier's K-scaled
`1.5e-3·sqrt(8192/64)` = 1.70e-2; at K=64 `direct`'s per-L2-tile truncation is a single epilogue cast.

**A — runtime-parameterized loop bounds.** Three mechanisms, none reaches a loop bound. (1) RTP:
core side built (`AIRToAIEPass.cpp:329-346` allocates `__air_herd_rtp_<x>_<y>`, `:451-508` rewrites
scalar operands to `memref.load`); host side compile-time only — `AIRRtToNpuPass.cpp:892-901` emits
`NpuWriteRTPOp` only when `dyn_cast_if_present<arith::ConstantOp>` succeeds (non-constants silently
skipped, `rtp_slot++` still advances), `AIEX.td:768-782` declares `I32Attr:$value`. (2)
`npu.address_patch` (`AIEX.td:917-931`) patches DDR addresses only. (3) The control scratchpad
(`AIEX.td:953-1071`): zero occurrences of "scratchpad" in `mlir/`, `python/`, `runtime_lib/`;
additive only, always 8 contiguous bytes, forces `*addr = result & 0xFFFFFFFC`, caps at 32 StateTable
entries; `get_ctrl_scratchpad_bo` exists in `libxrt_coreutil.so` (XRT 2.21.0) and `xrt_kernel.h:660`
but in neither `pyxrt*.so` (`pyxrt.cpp:196-217` binds `add_callback/set_arg/start/state/wait/wait2`);
on shim `d0_size` shares DMA_BDX_3 with `d0_stride` (`AIEDmaToNpu.cpp:562-564`), so only
`buffer_length` (DMA_BDX_0) is cleanly reachable, additively, in multiples of 4. **Structural killer:**
`AIETranslateNpuToBinary` (`mlir-aie/lib/Targets/AIETargetNPU.cpp:242-318`) `TypeSwitch`es over nine
ops (sync, write32, blockwrite, maskwrite32, load_pdi, address_patch, preempt, create_scratchpad,
update_from_scratchpad) into a flat `std::vector<uint32_t>` — no branch/jump/call/loop opcode;
`aiex.run` (`AIEX.td:567-586`) inlines; loops are `loopUnrollFull`ed (`AIRRtToNpuPass.cpp:1850`,
`:1977`, hard `signalPassFailure()`); `air_project/npu.air.mlir:588-748` holds 25
`dma_configure_task_for` and zero `scf.for`. Stream length is a function of shape. Corroboration
only: `offload_cache/` `.ctrltext` totals 227,440 / 346,736 / 196,112 bytes for the three GEMM
shapes — confounded by three recipes (`drain/tile_n96`, `drain/tile_n128`,
`fused-cast/tile_m64/tile_n96`). The three `.insts.bin` there are byte-identical 2,288-byte stale
leftovers (md5 `10855cd4…`), unread on the ELF path (`xrt.py:567` sets `self.bo_instr = None`) — do
not cite them. [03](03-measurement-model.md) absorbed this (`e58a2170`): `offload` = one xclbin, N
instruction streams; runtime bounds deferred.

**C — one xclbin, no DRAM between operators.** The whole-layer stitch parses (`qkv_proj` +
`mha_out_proj` + `fused_tail` as three `KernelSlice`s, `_TAIL_BUFFER_ALIAS`; 20 func args / 1015
lines / 11 `air.launch` @1024) and aircc never leaves the AIR pipeline: `air-opt --air-fuse-channels`
on `mha_out_proj` @256 (`pass_017`, 45 channels) 64 s rc=0; `fused_tail` @256 (45) 53 s rc=0;
stitched `mha_tail` @256 (**90**) **rc=124 at 600 s and again at 1200 s**. Controls: `mha_out_proj`
@256 end-to-end in 97 s (60 debug-IR passes), `fused_tail` @256 in 81 s; 2× channels → ≥18× pass
time; preset-independent; full-layer @1024 killed at 1355 s with `aie.air.mlir` never written.
Source: `AIRDependencyScheduleOpt.cpp:5064` `AIRFuseChannels`, pair loop `:5139` with
`checkIfTemporalMergeable` + IR mutation, `renameSymbols` `:5083` a second O(N²)
(`SymbolTable::replaceAllSymbolUses` per iteration); no fixed-point loop, so **slow, not hung**;
`mergeChannels` skipped by default (`:5301`). Half 2 is capacity-bounded: "device-resident" in
`fused.py` means no host sync; all 17 func args at 1024 are L3 memrefs. Logical tensor traffic at
1024×768×3072 bf16 (real DMA traffic is higher): `ffn_up` + `ffn_gelu` 24.0 MiB (29 %); `qkv_f32`
18.0 (21 %); `q`,`k`,`v` 9.0 (11 %); `hidden` + mirror 6.0 (7 %); `attn_context` 3.0 (4 %);
`attn_out` 3.0 (4 %); `x` re-read 1.5 (2 %); `ffn_out` 3.0 (4 %) (`[2026-08-10]` row was missing —
items summed to 64.5 against 67.5; found by 31a's re-derivation, now
[31](31-resident-tail-r1-record.md)); **total 84.0 MiB** (49.5 read + 34.5 write); irreducible
(weights 13.5 + x 1.5 + out 1.5) 16.5 MiB (20 %); **intermediate 67.5 MiB (80 %)**. J7a removed
~15 MiB of a ~99 MiB baseline (four S×E norm args at 3 MiB + two gammas `[S,E]` → `[E]`), invisible
to `bytes_transferred`. J7b buys `fused` nothing: `fused.py:149,292` imports `build_ffn_module`, and
the fused down-projection accumulates in L2 (`tile_k_l2=512`). **Hard ceiling:** 32 cores × 64 KiB
L1 (`getLocalMemorySize` = 0x10000) = 2 MiB + 8 memtiles × 512 KiB (`getMemTileSize` = 0x80000) =
4 MiB → 6 MiB, not a flat space; one S×F intermediate is 6 MiB at 1024, 24 MiB at 4096; reachable
only under ~128–256-row sequence blocking, and a Q band still needs all of K/V (3 MiB at 1024,
12 MiB at 4096).

### 2.8 Sizing, rules and order

Worktree-sized: rewrite the three lit gates first (`run_npu2_offload_peano.lit:42,44` and
`run_npu2_runlist_peano.lit:47,49` required `attention host torch fp32`; `run_npu2_fused_peano.lit:46,85`
pinned `3 entries over 3 ELFs` and `sync 19`); corrected `offload` (two attention GEMMs on device via
the `gemm_spec_fn` escape hatch, `drain/tk2=256/tk1=32/tn=16/herd 8×4` for `attn_output`,
`drain/tk2=64/tk1=32/tn=128/herd 8×4` for `attn_scores`; six dispatches → eight ×12 heads); fix the
`fused` SPECS row (~1 h); correct the backend-settings claims (~half a day); correct the "no K=64/N=64
row exists" language in 11 places (16:160, 03:149-155, 08c:23-31 and 09:66 — the last two now
in [01](01-original-plan-superseded.md) — plan `README.md:267`,
`programming_examples/transformer_layer/README.md:484`, `pattern/blocked_attention.py:22-28`,
`pattern/runlist/README.md:70-72`, `pattern/runlist/runlist.py:154`, `pattern/offload/README.md:24`,
`pattern/offload/offload.py:25` — a catalogue constraint, not hardware; re-grep, line numbers drift).
Own phase: device softmax operator; corrected `runlist`; N streams under one xclbin (**LANDED
`93e15a64`**, §5); scope `air-fuse-channels` to same-launch pairs (only if one whole-layer xclbin
matters); GeLU as a GEMM epilogue (−12 MiB of 84, −14 %, ~1–2 days, the only cheap DRAM win);
corrected `coarse` (§3–§4). Order: 0 corrections/gates → 1 hardware run (DONE, overturned §2.4) → 2
corrected `offload` → 3 softmax → 4 corrected `runlist` → 5 N-streams (touches `dispatch.py:685`,
shared with `fused`; not concurrent with C) → 6 GeLU epilogue → 7 corrected `coarse`.

**Do not**: open a worktree for runtime-parameterized loop bounds (needs mlir-air unroll removal,
`DmaToNpuPattern`, RTP host side, 27 `getStaticScfForTripCountAsInt` call sites across 6 files;
mlir-aie `NpuWriteBdOp`, `AIEDmaToNpu.cpp`, `AIETargetNPU.cpp`; a rebuilt XRT; a txn format that
does not exist). Delete `qkv_f32` (−18 MiB): `qkv_proj.py:297-308` raises for scratch-free specs;
the alternative is the strided-producer wall at `norm_tail.py:88-104` (H7, `mlir/`). Search more
attention tiles: the space is 1584, the answer was the first one.

Open after the spikes: `air-fuse-channels` wall time on a stitch (>1200 s at 90 channels, seq 256;
one uncapped `air-opt …/spikeC/mt256_pass017.mlir --air-fuse-channels`, CPU-only, not overlapping a
timed measurement); whether a whole-layer ELF survives `air-split-l2-memref`'s per-column shim cap
(8 cols × 2), 23 herds vs 32 tiles, 16 BDs per shim column; the 12-head dispatch; numerical margin
on real attention operands (spike inputs were seeded gaussians scaled by `1/sqrt(K)`); one tiling
recipe for all three offload shapes (3 builds + 3 measures, only if the one-stream increment is
taken); `npu.update_from_scratchpad` on a shim BD (moot without a branch opcode); absolute DRAM
numbers (logical floor; no device-side counter). Operational: Spike B's candidates waited 800–1050 s
behind lane C's 1100–1400 s builds — do not co-schedule build and measure lanes;
`registry_sweep.py` cannot stage either attention shape (`FAMILY_HIDDEN × ROLE_KN_MULTIPLES`, min
hidden 512) — use `gemm_spec_fn`, nothing requires editing the registry.

---

## 3. What `coarse`'s blend is a blend OF, and what selects it (was doc 28)

`[2026-08-09]` [03](03-measurement-model.md) defines `coarse` as "reconfiguration AND sync overhead
minimized together, by mixing `runlist` and `fused` per workload". A fused region is an artifact
somebody stitched, so the space is derived from the artifact plans: `fused` and `coarse` build their
front from the same two modules (`build_qkv_proj_module`, `build_mha_out_proj_module` — what
`block_config` resolves) and differ in the tail alone; `runlist` uses three separate projections,
per-head `attn_scores`→`softmax`→`attn_output` then `output_proj`, and up / GeLU / down + per-band
add / LayerNorm / multiply.

**Two axes, six cells.** Front: `block`-form (two ELFs, q/k/v device-resident) · `runlist`-form
(decomposed, per-head attention). Tail: stitched · row-banded · fully decomposed.

|  | tail stitched | tail banded | tail decomposed |
|---|---|---|---|
| front `block` | **= `fused`** | **= `coarse` today** (C1) | new (C2) |
| front `runlist` | new | new (C3) | **= `runlist`** (C6) |

The space contains its own endpoints, so "`coarse` = best cell" would collapse the taxonomy to three
points — and on [27](27-common-ladder-result.md)'s evidence at 512/1024 it would collapse to `fused`
(fastest and lowest DRAM bytes at both lengths; 1 submission, 13 sync boundaries — the minimum).
Today's `coarse` is already an interior cell, `(block, banded)`; what it lacked was provenance.

**What selects the blend is what the workload ADMITS:** `fused`'s stitched tail is bounded to
256..1024 (`plane_major` needs a plane stride of `rows*cols` against the shim `aie.dma_bd` cap
1,048,576 → caps at 1365 rows, `norm_tail.py:262-273`), so at seq ≥ 2048 the whole top row is
unbuildable; the `runlist` front's attention tiles are legal at 512/1024 but not 256 (`n % (tile_n
128 * herd_n 4) != 0`, [27](27-common-ladder-result.md)); one xclbin for the whole layer is blocked
twice over ([03](03-measurement-model.md)), so no cell fuses front into tail. Hence: seq ≤ 1024 —
all six cells, `coarse` degenerate (`fused` dominates); seq ≥ 2048 — bottom row only, **the mode's
real territory**. `coarse` is the mode you use when `fused` does not fit, which is why the D2 block
(built at 4096) landed on `(block, banded)`.

**Rules.** The corrected `coarse` is specified and measured at 2048/4096 where `fused` is NOT
available; cells to measure there: C1 `(block, banded)` (incumbent, gated at 4096), C2
`(block, decomposed)`, C3 `(runlist, banded)`; `(runlist, decomposed)` at those lengths is `runlist`,
the fourth calibration point. **Do not measure the six cells at 1024 and declare a winner** — every
cell there is dominated by one with a mode name. Costs: `prepare_runlist`'s and `prepare_fused`'s
dispatch steps were nested closures and had to be extracted without behaviour change, proved by
re-running `run_npu2_runlist_peano.lit` / `run_npu2_fused_peano.lit`; `fused`'s front and tail couple
through packed buffers (`packed1` plane 1 is `x`, plane 0 written on device by entry 2), so any
`(runlist, stitched)` cell is the expensive one (moot at 2048+); `builders/block.py`'s `_sequence_a`
/ `_sequence_norm` / `_sequence_ffn` are module-level, so C1/C2's front is free.

---

## 4. `coarse`'s two interior cells, built and gated (was doc 30)

`[2026-08-09]` Landed: `pattern/coarse/cells.py` (the space as data, two-half config, composed
dispatch, shared preparer); `pattern/coarse_c2/`, `pattern/coarse_c3/` (one thin preparer + ELF cache
each); `coarse_cells_structure.py` (host-only structural arm); `run_npu2_coarse_c2_peano.lit`,
`run_npu2_coarse_c3_peano.lit` (clean + negative control + structure arm); two `opcheck_specs.py`
rows at `4096x768_encoder_bert`, `atol` at the `1e-1` ceiling. **`coarse` itself did not move**: C1 is
still `pattern/coarse/coarse.py` over `builders/block.py`, and `cells.py` *refuses* to build C1,
`fused` or `runlist` (a second implementation would measure something D2 never validated). The
composition calls, it does not copy: `runlist.py`'s six dispatch regions moved from closures to
module level on the `(cache, cfg, ...)` convention — the only extraction taken; the `fused.py`
extraction was deliberately NOT done (no stitched tail builds at ≥ 2048; `fused` stayed green as an
unmodified control). The extraction is inert, measured: `run_npu2_runlist_peano.lit` reproduces
`submissions 17 entries 427 air 50 herd 488 sync 451 bytes 190513152` byte for byte on both halves.

**First hardware run**, `4096x768_encoder_bert`, 10/10 boundaries clean, negative controls failing,
clean and injected totals equal:

| cell | front | tail | subs | entries | air | herd | sync | bytes |
|---|---|---|---|---|---|---|---|---|
| C1 `coarse` | block | banded | 4 | 131 | 12 | 146 | 402 | 202,902,528 |
| C3 | runlist | banded | 17 | 169 | 46 | 232 | 451 | 190,319,616 |
| C2 | block | decomposed | 4 | 389 | 16 | 402 | 402 | 203,096,064 |
| C6 `runlist` | runlist | decomposed | 17 | 427 | 50 | 488 | 451 | 190,513,152 |

Entry counts predicted host-side before the runs: front `block` 1/2, front `runlist` `2+heads` /
`4+3·heads`, tail banded 3 / `1+2·blocks`, tail decomposed 3 / `3+6·blocks`; the model recovers both
pinned endpoints (4/131, 17/427). Ordinal claim holds: **131 < 169 < 389 < 427**. **The vectors are
additive**: `C1 + C6 = C2 + C3` on every column — submissions 21, entries 558, air 62, herd 634, sync
853, bytes 393,415,680 — no interaction term, so the composition is what it claims independently
of anyone's account. Solving: the `runlist` front moves **12,582,912 bytes fewer** than the `block`
front (exactly `2 × [4096, 768]` bf16); the decomposed tail costs **193,536 bytes more** than banded.

**Error tracks the tail**, all at `atol = 1e-1` over 3,145,728 elements, zero mismatches: C3 banded
`mean_rel_L1` 1.654e-2 / `atol_required` 7.266e-2 (1.38×); C1 banded 1.688e-2 / 7.398e-2 (1.35×); C6
decomposed 1.746e-2 / 6.981e-2 (1.43×); C2 decomposed 1.784e-2 / 7.896e-2 (**1.27×**, the least
headroom any whole-layer row carries — a recorded fact; past `1e-1` is a defect report, never a
moved ceiling). No tolerance widened.

**The ladder** (`--warmup 2 --samples 5`, one process per rung, ELFs pre-built by a build-class
pass; 8/8 per walk), ms avg w1 / w2, min w1 / w2:

| cell | 2048 avg | 2048 min | 4096 avg | 4096 min |
|---|---|---|---|---|
| C1 `coarse` | 215.6 / 216.2 | 204.0 / 200.6 | 479.0 / 459.8 | 455.2 / 444.7 |
| C2 | 238.4 / 220.5 | 225.6 / 206.6 | 490.6 / 509.8 | 485.3 / 488.1 |
| C3 | 309.0 / 309.9 | 304.5 / 296.8 | 748.2 / 764.8 | 724.1 / 745.4 |
| C6 `runlist` | 333.3 / 322.0 | 327.5 / 311.8 | 805.6 / 858.6 | 779.0 / 805.9 |

**C1 < C2 < C3 < C6 survives both walks on averages and minimums at both lengths.** The FRONT axis
is unambiguous: block ~1.5–1.6× faster than runlist, no overlap. The TAIL axis is clean only at
4096 (C1's slowest min 455.2 below C2's fastest 485.3); at 2048 the C1/C2 average gap is 10.6 %
(w1) / 2.0 % (w2) against intra-walk spreads of 15–18 %, C2's average moved −7.5 % between walks —
**C1 ≤ C2, not separated; do not quote a 2048 tail effect.** Intra-walk spread here 1.8–20.3 %,
wider than [27](27-common-ladder-result.md)'s 2–10 %; the bands are not interchangeable.

**DRAM bytes, cold (gate) vs warm (ladder, last of five samples after two warmups)**, identical
across walks: C1 202,902,528 → 188,743,680 (drop 14,158,848); C2 203,096,064 → 188,743,680
(14,352,384); C3 190,319,616 → same (0); C6 190,513,152 → same (0). Derivable to the byte: cold
C2−C1 = 193,536 = the gamma broadcast `2 × (64 × 768 × 2) − 2 × (768 × 2)` (banded `addnorm` takes
the `[emb]` weight, decomposed `elementwise_mul` a materialized `[norm_rows, emb]` band); C1's drop =
its static-weight set `w_qkv + w_o + w_up + w_down` at 4,718,592 each + two `[emb]` gammas at 3,072.
The runlist-front cells dropped zero because `evict_attention_contexts` cleared all `cache._pools`
once per head (~14 MB of ~190 MB, ~7 %; not the reason for the ~60 % latency gap). **`[2026-08-10]`
REMOVED**: eviction is targeted (`KernelCache.evict_pools_for` via `signature_kernels`); pooled
ELF-ABI BOs are `xrt.ext.bo`, device-level, surviving a context unload; warm `runlist` now reads
`sync 443 bytes 176160768` vs cold 451 / 190,513,152 — a drop of exactly 14,352,384 across 8 skipped
uploads; cold totals unchanged, every gate literal stands; `check-runlist`, its fault twin and
`check-coarse-c3` green, dispatch host tests 32/32.

**`coarse` = C1 = (block front, banded tail)**, now chosen rather than inherited, recorded in the
mode's artifact (`blend_cell`, `blend_front`, `blend_tail`, `blend_selected_by`). The taxonomy does
not collapse: at 2048/4096 the winner is an interior cell, distinct from `runlist` (slowest) and from
`fused` (unbuildable). It does not rank `coarse` against `fused` (that is
[27](27-common-ladder-result.md)'s at ≤ 1024).

Two side findings. (1) The `runlist` catalogue row was stale (pre-`52b93e1a`: "host torch attention",
"5 submissions over 391 entries"; recorded `mean_rel_L1` 1.732e-2 / `atol_required` 7.077e-2 vs the
fresh 1.746e-2 / 6.981e-2 — a different computation, not a regression); corrected in place, old
figures kept as non-comparable. Nothing fails when a catalogue comment goes stale
([16](16-compiler-changes.md)'s lesson). (2) `run_npu2_runlist_gate.lit`'s latency clause
(`agg_ms < seq_ms`, strict, no margin) went intermittent under suite contention: red once at
`sequential 25.191 ms / runlist 25.277 ms` (−87 µs, 0.9966×) with its three bit-identical checks
passing; green on the isolated re-run (31 s) and in a second full 30-test suite (30/30, 541 s).
The criterion was NOT widened — 05a (now [01](01-original-plan-superseded.md)) measured the real
effect at 1.02–1.15×. **`[2026-08-10]` Decided: the verdict compares interleaved MINIMUMS**
(`agg_min < seq_min`, both legs of `runlist_gate.py`, medians and win count reported beside), per
[23 §1](23-rules-and-open-items.md); validated isolated: leg A 24.786 vs 24.921 ms (135 µs,
1.0054×), leg B 23.263 vs 23.607 ms (344 µs, 1.0148×), `PHASE B GATE: PASS` all four legs. Suite
standing state 30/30 (`b795deb1`, 541 s).

---

## 5. `offload`: N instruction streams under one xclbin (was doc 29)

`[2026-08-09]` **Landed** (`93e15a64`): the array is configured once per layer instead of thirty
times. Before, the mode paid a full `hw_context` teardown before every dispatch — the maximum
reconfiguration cost, against [03](03-measurement-model.md)'s "reconfiguration MINIMIZED by dynamic
partitioning". Five GEMM shapes chained via `xclbin_input` into one xclbin, loaded once, each shape
binding its own kernel and instruction stream:

```
[offload] stages: 10/10 clean
[offload] dispatch totals: submissions 30 entries 30 air 30 herd 90 sync 90 bytes 99090432
[offload] reconfiguration: context_loads 1 kernel_attaches 4 over 30 dispatches
```
(a **1024** figure). The dispatch vector is unchanged by design — one `run_sequence` per GEMM either
way — so the existing gate is a correctness check on the change; reconfiguration cannot be a
seventh vector field. `KernelCache.reconfiguration_counts()` counts it; `describe_offload` prints it.

### 5.1 The three identifiers — the rule

A stream needs all three distinct; no caller in the tree set any, and only one fails loudly:

| identifier | keys | duplicate ⇒ |
|---|---|---|
| `kernel_name` | the `EMBEDDED_METADATA` entry | xclbinutil REFUSES the merge (*"Kernel name already exists in the EMBEDDED_METADATA section: 'MLIR_AIE'"*) — the only loud one |
| `instance_name` | the kernel's name in the xclbin | the loader's substring match (`xrt.py:634`) returns whichever came first |
| `kernel_id` | the PDI in the merged `AIE_PARTITION` (`dpu_kernel_ids`; every AIR compile defaults to `0x901`) | the second kernel runs against the first's array configuration: `ERT_CMD_STATE_TIMEOUT` at one shape, garbage at `mean_rel_L1` 1.41 with no error at another |

`pattern/offload` had named every `drain` GEMM `matmul_bf16` with no kernel id; at 1024 all five
shapes resolve to `drain`. The retired `probe_one_xclbin_n_streams.py` (at tag
`pre-cleanup-20260821`, per `agents/probes/README.md`) found only the first two — it set
`kernel_name` = `instance_name`; the third surfaced only on the real five-shape mode. A two-kernel
probe is not proof that an N-kernel path works.

### 5.2 What landed, and two traps

`attach_kernel` (`python/air/backend/xrt.py`); `compile_shared_xclbin` (chained build, validates all
three identifiers), `ensure_loaded` (artifacts sharing an `output_binary` share one context),
`reconfiguration_counts()` — all `llms/shared/infra/cache.py`; `plan_submissions(config_of=...)`
(xclbin split rule keyed on configuration identity, `artifact_of` the default proxy) in
`llms/shared/infra/dispatch.py`; per-shape identifiers, own cache dir, no eviction on the shared path
in `pattern/offload/offload.py`. Traps: each chain link needs its own output name (aircc writes
relative to cwd; *"The following output file is also used for input"*); only the last link holds
every kernel, so any stale member rebuilds the whole chain.

**The silent-degradation failure.** The first "working" run reported `context_loads 5
kernel_attaches 0`: the runtime imported `air` from `install-xrt/`, not the edited `python/air/`;
`getattr(backend, "loaded_binary", None)` swallowed the missing attribute and degraded to "no
sharing" — [15](15-environment-notes.md)'s staleness trap. `ensure_loaded` now **raises** when
artifacts share a binary and no loaded backend reports it. **Rule: a capability probed by
`getattr(..., default)` degrades quietly by construction; when the degraded path is
indistinguishable in the logs, the default has to be an error.**

### 5.3 Latency: no claim; variance: the collapse

Four interleaved A/B runs at 1024, five samples: shared median avg 164.3 ms / median min 158.6 /
min spread 8.0 ms; ELF 182.5 / 163.9 / 20.5. Distributions overlap — no latency difference; the
`hw_context` reload is not dominant at 1024. **`[2026-08-09]` The lead taken**: four walks `{ELF,
shared} × {w1, w2}` at 512 and 1024, A/B/A/B, `runlist` walked inside each as control, `--warmup 2
--samples 5`, one process per rung, caches warmed by `--class build` first, 16/16. Intra-walk
`(max-min)/min`: `offload` 512 ELF **316.9 % / 134.1 %** vs shared 17.6 % / 14.0 %; `offload` 1024
ELF 9.0 / 10.5 % vs shared 5.8 / 5.5 %; `runlist` control 512 15.3 / 7.6 vs 5.5 / 8.1 %; 1024 6.5 /
2.4 vs 4.0 / 4.1 %. **Switching to the shared xclbin removes the mode's variance** (at 512, an order
of magnitude, both walks, control in band). Not established: that `_evict_context` is the mechanism —
the env var changed eviction AND the ABI (`xrt.ext.kernel` vs `xrt.kernel`, explicit instruction BO,
extra launch args); [27](27-common-ladder-result.md)'s hypothesis is supported, not confirmed. The
isolating experiment (xclbin ABI with eviction forced on — `probe_context_reuse.py`, retired →
[27](27-common-ladder-result.md), showed `xclbin`+`[2,2]` and `xclbin`+`[1,1]` bit-identical over
four runs) needs a knob that does not exist. Qualifications: shared at 512 is 14–18 % against the
control's 5.5–8.1 % (large collapse, not total); the 1024 ELF baseline did not reproduce 27's 61.6 /
59.8 % (read 9.0 / 10.5 %). ELF minimums reproduce 27's (82.0 / 78.9 ms at 512 vs 78.2 / 79.9).

**The cost**: 512 ELF avg/min 163.5/82.0, 111.5/78.9 vs shared 103.7/97.5, 105.5/99.5; 1024 ELF
176.6/168.7, 165.6/159.6 vs shared 164.1/160.1, 164.4/161.5. The shared best case at 512 is ~20 %
WORSE (97.5–99.5 vs 78.9–82.0); on [23 §1](23-rules-and-open-items.md)'s minimum convention the
shared path is slower at 512 and level at 1024 — a trade of variance for best case. A fixed
per-submission cost fits (30 × ~0.7 ms covers the 512 gap); unexplained. `prepare_offload` computes
`device_ms`, `sync_ms`, `host_cpu_ms` and schema v1 had no column (a version bump, `schema.py:53`;
v2 since 2026-08-10). Artifacts `results/offload-ctx-{elf,shared}-w{1,2}/` (schema v1);
`npu_unique_xclbin_count` reads 1 on the shared arm and 0 on ELF (`xrt.elf()` loads no xclbin);
`bytes_transferred` byte-identical between arms.

### 5.4 The gate — `make check-offload-shared`, CLOSED `[2026-08-09]`

`run_npu2_offload_peano.lit` gained a third recipe and pins the counters on both paths: ELF (clean
half, 4096) `context_loads 30 kernel_attaches 0 over 30 dispatches`; shared (1024) `context_loads 1
kernel_attaches 4 over 30 dispatches`, plus the "ONE xclbin over 5 shapes" line, three stage
comparisons at zero mismatches, `stages: 10/10 clean`, the full vector, `passed`. Suite 28/28 on NPU2
(494.5 s), study host tests 84/84, dispatch/seam unit tests 31/31, `phase_e_checks` selftest 30/30.
**Rule learned from the run:** under the xclbin ABI the vector is steady only after a warmup — the
cold call uploads each artifact's instruction stream (`sync_instruction_bos`, skipped by the ELF ABI)
so it reads `sync 95 bytes 99141520` and every later one `sync 90 bytes 99090432`; five artifacts =
five sync boundaries and exactly **51,088 bytes** (the five cached `.insts.bin`). The target
dispatches twice and pins both; **a vector read from a single cold dispatch under this ABI is
inflated**. Verified failing: the shared prefix against the ELF packaging fails on
`reconfiguration: 5 xclbins, 30 hw_context loads for 30 dispatches`.

### 5.5 The 4096 wall, the multi-launch xclbin, and the hardware verdict (doc 29 §The hardware verdict)

At 4096 the down-projection resolves to `fused-cast` — two `air.launch` ops (`air_launches=2 herd=4`
for `off_gemm_4096x3072x768`; `1`/`3` for all nine others). `XRTBackend.compile` defaults
`insts="air.insts.bin"` and the xclbin branch passes it as `-i` (the ELF branch passes `--elf-name`
and no `-i`, `xrt.py:307-316`), so aiecc refuses: `edge 'air.insts.bin' produced duplicate output
path './air.insts.bin'` (the memtile "Failed to allocate buffer" lines are a red herring). Shared path
bounded to single-launch modules; recipe gated at 1024. **Do NOT fix by dropping `-i`** — the same
`else` serves `txn`, `pdi` passes `-i` on its own line. Any `xrt.py` change needs an `install-xrt`
rebuild and the ten-model regression.

**`[2026-08-10]` Compile wall down — three pieces, the rename was a third.** The chain builds at
4096 (seven PDIs, five kernel-owning: `matmul_bf16_{proj,up,attn_scores,attn_output}` +
`gemm_cast_bf16_down`), confined to `XRTBackend.compile`/`_finalize_multi_launch_xclbin`, active
only for >1 `air.launch` AND `output_format == "xclbin"`: (1) `--npu-insts-name`/`--xclbin-name`
take a `{0}` template — a two-launch module reaches aiecc as three `aie.device` ops (one per launch
plus a `main` device whose sequence inlines each launch's DMA program with `aiex.npu.load_pdi`
between); (2) the artifact is the main device's pair, repackaged — `XRTCompileArtifact`, `load()`,
`attach_kernel`, `sync_instruction_bos`, dispatch loop untouched; (3) aiecc leaves the main xclbin
unexecutable (main partition holds only the empty main PDI; `load_pdi` ids restart at 1 per compile,
single-launch links sit at `pdi_id 0x1`), so the backend walks the stream, renumbers ids off the
link's `kernel_id` (`0x903 → 0x9031, 0x9032`), patches the words, merges per-launch PDIs as
kernel-less entries via `xclbinutil --add-replace-section`. Fixture
`test/xrt/56_multi_launch_xclbin_compile` (compile-only; dies on the duplicate-path error unpatched).

**`[2026-08-11]` Hardware verdict: in-stream `load_pdi` faults.** NPU2, XRT 2.21.0, firmware
1.1.2.64, Turbo: **29 single-launch dispatches executed clean off the shared xclbin at 4096**; the
one multi-launch module faulted on first submission — `off_gemm_4096x3072x768`,
`ERT_CMD_STATE_TIMEOUT`, `fatal_error_type 0x10`, `fatal_error_exception_pc 0x161AD`, `txn_op_idx
0xFFFFFFFF` (a firmware exception). **The scoped fallback does not exist**: `--expand-load-pdis` is a
no-op on mlir-aie 1.4.0's `aiecc` for both edges (aircc passthrough leaves the main stream
byte-identical at 4,800 B, leading word `0x00020008`; `--get-full-elf` changes nothing; pure ELF mode
emits the same 4,800 B `.ctrltext.0`); the ELF ABI's multi-launch support comes from `aiebu`'s
per-sequence `.pdi.N` sections. The "19 KB → 174 KB" claim matched no artifact (the 4096 ELF's 176 KB
`.ctrltext.2` is a launch program; its `.ctrltext.0` is 2,288 B). Routes: (1) upstream mlir-aie
expansion (`aie-materialize-runtime-sequences` + `aie-expand-load-pdi` into aiecc's npu-insts edge,
wheel rebuild); (2) ctrl-packet flow (`--load-pdi-to-ctrl-pkt`, mutually exclusive with expansion,
unscoped); (3) mode-level: single-launch `drain` for the 4096 down-projection; (4) split-chain
(`context_loads 2`). **Route 3 taken, shared path GATED AT 4096**: `offload_config._chain_spec`
re-resolves a `fused-cast` winner to the shape's measured `drain` row under the shared path only
(ELF path keeps the winner; no drain row raises through the registry); priced in the log — `down
4096x3072x768 drain (registry, pinned over fused-cast)`, 6,226 vs 6,927 GFLOP/s (~10 %) — and
enforced by the recipe. Measured at 4096: 10/10 stages on both dispatches, `context_loads 1
kernel_attaches 4` over 30; cold−steady delta 293,200 = the five instruction streams (51,088 at
1024); vs the ELF clean half the steady totals differ by exactly 12,582,912 = 4096 × 768 × 4 (the
`fused-cast` f32 C scratch) plus one launch's air/herd/sync (31/91/91 → 30/90/90). Lit 1/1, 95.9 s,
2026-08-11. Single-sample 989.9 ms avg (Turbo) is a smoke figure.

### 5.6 What this does not do — the default flip (doc 29 §What this does not do)

The shared path was opt-in (`AIR_OFFLOAD_SHARED_XCLBIN=1`) with its own cache directory (identical
artifact NAMES over different ABIs must not share a directory). Decision (reviewed independently):
**fix the 4096 wall first, then flip** — latency should not decide which implementation gets to be
called `offload`; the taxonomy defines it by minimized reconfiguration. **`[2026-08-11]` FLIPPED
the day the precondition was met**: the shared xclbin is the default; the ELF path is the
**legacy/control** packaging, opt-in `AIR_OFFLOAD_LEGACY_ELF=1`; `AIR_OFFLOAD_SHARED_XCLBIN` is
retired and **RAISES** if set in any form (`=0` would silently run the opposite packaging). Cache
dirs unchanged (`offload_shared_cache` / `offload_cache`). Lit recipes flipped which side sets the
env var only: default recipe pins `context_loads 1 kernel_attaches 4`; legacy recipe
(`AIR_OFFLOAD_LEGACY_ELF=1` inside its Makefile targets, fault twin on the same arm) pins
`context_loads 30 kernel_attaches 0`; every literal unchanged. **Every `offload` latency/variance
number recorded before 2026-08-11 — [27](27-common-ladder-result.md)'s table included — describes
the ELF path**; the re-walk ran the same day — [32 §The post-flip walk](32-cost-decomposed-ladder.md):
16/16 orderings `fused` < `coarse` < `runlist` < `offload`, ELF-era variance gone, timed-region
`context_loads` 0. Still not delivered: runtime-parameterized loop bounds (§2.7 A).

**Verification:** `E1 GATE: PASS` in 4254 s — suite 28/28 on NPU2 (`run_npu2_offload_peano.lit`
150.9 s of dispatch), all ten shipped models verify (the standing regression clause for anything
touching `llms/shared/` or the installed `air` backend), dispatch unit tests 31/31, study host tests
84/84, `phase_e_checks` selftest 30/30.

---

## 6. Item 11(a) — `memcpy_bandwidth`: DEFER, bound to 11(b) (was doc 33)

Scoping 2026-08-12, read-only, no device job; pmode `Default`, **no latency originated here**.
Commit `2d6756ca` closed the install-staleness caveat (`install-xrt/bin/air-opt` matches `build-xrt`
at 2026-08-11 13:28, verified by artifact) and made 11(b) the sole claimant on the exclusive window.

**Verdict: DEFER.** (1) A bandwidth operator is not the instrument for the unattributed half —
job 246's log names per-stage `record_kernel`/`record_cpu` on `pattern/`. (2) It is a hard input to
`roofline` (`roofline/run.py:1675-1678` refuses without the memcpy CSV; `:1710-1711`
`peak_bandwidth = peak_memcpy_bandwidth_gbps(...)`; `:1462` `roof_y = np.minimum(peak_bandwidth_gbps
* x_values, compute_ceiling)`; `:1599-1601` per-shim-tile roofs; the compute roof is theoretical —
32 cores × 64 bf16 MACs/cycle × 2 × 1.8 GHz = 7372.8 GFLOPS, `:1359-1366`, `:49-53`). (3) Every
decision available today is robust to a 2× ceiling error. Triggers: 11(b) taken → build 11(a) first,
AIR-native herd-width form; 11(b) dropped → drop; Phase G taken → re-examine (iron wires memcpy into
`unattended_smoke_job.py:23,46,73`, `unattended_reboot.py:30,411,962-971`); a per-operator
"fraction of peak" for the norm tail / FFN elementwise → build; a design crossing the per-column
shim budget → build at least the `[4, 8]` rung.

**iron's study.** `AIEMemCopy`, fixed `SIZE_LADDER = (8388608,)` bf16 (16,777,216 B in;
`total_moved_bytes` 33,554,432), `FIXED_TILE_SIZE` 4096, 10 warmup / 500 timed
(`iron/applications/transformer_layer/study/memcpy_bandwidth/cases.py:10-21`, README:36-49);
`num_cores` `(2, 4, 8, 16)` (the real axis, = shim tiles 1/2/4/8 with 2 channels fixed),
`num_channels` `(2,)` degenerate, `bypass` `(False, True)` default `True` only. So it is a
shim-tile scaling curve, two axes, not four; dropping `num_channels` costs nothing.
`results_unattended_full_suite_20260801_023954/memcpy_bandwidth/results.csv` (2026-08-02): 2 cores /
1 tile 44.68 bypass, 45.28 kernel; 4 / 2 64.95, 63.64; 8 / 4 **70.86**, 70.79 (**`failed_validation`**,
3,876 bad elements); 16 / 8 67.88 (`is_overall_peak`), 67.80 (`failed_validation`, 68,297). Saturates
at ~4 tiles (1→2 +45 %, 2→4 +9 %, 4→8 negative); the kernel arm is red at 8 and 16 cores. Default
run (`results/memcpy_bandwidth/results.csv`, 2026-08-03, bypass-only): 44.69 / 64.67 / 64.32 / 67.58.

**`[2026-08-12]` verified at merge, with one correction: the 4-shim-tile rung is not stable** —
8-core bypass reads 64.32 GB/s (latency 521.7 µs) in `results/` and both
`results_unattended_execution_smoke_20260803_{024305,095245}` (one measurement copied into three
trees), 70.24 (477.7 µs) in `…full_suite_20260427_131305`, 70.86 (473.6 µs) in
`…full_suite_20260801_023954` — 1 smoke vs 2 full-suite runs, ~10 % apart; other rungs tight (2-core
44.68–45.63, 4-core 64.34–64.95, 16-core 67.58–68.38). **Quote the imported constant as a band, not
a point; the peak is at 4 tiles, the curve is non-monotonic.** Interim constant, labelled: *NPU
DRAM↔array bandwidth measured by iron's `memcpy_bandwidth` at 44.7 GB/s (1 shim tile) → 64.9 (2) →
70.9 (4) → 67.9 (8), fixed 32 MiB, bypass, Turbo; different toolchain; order-of-magnitude only;
superseded when 11(a) produces an AIR number.*

**Where it lands on [03](03-measurement-model.md)'s axes:** off-chip traffic cost — not a second
instrument but the denominator converting `bytes_transferred` (count) into time; nothing in the tree
relates the two. Already here: `bytes_transferred` (`schema.py:218`), `device_ms`/`sync_ms`/
`host_cpu_ms` (v2, `schema.py:317-350`), `bandwidth_gbps` **already a schema column**
(`schema.py:429-433`, tuning table), `shim_{s2mm,mm2s,dma}_channels_used` + utilizations
(`resource_usage.py:171-178`), `core_to_core_flows`, `component_groups.py`.

**Job 246** (`agents/.state/devq/jobs/job-000246.log`, `measure`, exit 0), `offload` @1024: GEMMs
(NPU) `device` 64.388 ms, 0/8 components; non-linear (host) `host_cpu` 10.914, 5/5; data sync
`sync` 4.494; attributed 79.795 of 159.795 ms; **UNATTRIBUTED 80.000 ms (50.1 %)** — "host overhead
outside every instrumented region"; totals `submissions 30 entries 30 air 30 herd 90 sync 90 bytes
99090432`, `context_loads 1 kernel_attaches 4`, 10/10 clean. Inference: only the 4.494 ms (2.8 %) is
directly bandwidth-bounded; the 64.388 (40.3 %) is not separable today; sync rate 99,090,432 B ÷
4.494 ms ≈ 22.0 GB/s vs iron's 67.9–70.9 → ~3 ms headroom ≈ 1.9 % of the layer. Cross-toolchain
caveat: iron's `bandwidth_gbps` = `total_moved_bytes / latency` (33,554,432 ÷ 496.5 µs = 67.58,
matches the CSV); like-for-like in definition, an order-of-magnitude statement.

**AIR-native form.** All three passthrough examples are `herd sizes=[1, 1]`
(`passthrough_dma.py:55`, `passthrough_channel.py:49`, `passthrough_kernel.py:54`, the latter
`link_with="passThrough.cc.o"` — the same kernel iron links, `op.py:63-73`); the multi-worker shape is
`channel_examples/channel_size/channel_size.py` (`:37-38, 67-71, 88, 102`); bypass = herd body
`ChannelGet` → `ChannelPut` with no kernel (iron: `forward()` no Worker, `design.py:191-192`); routed
artifact `aie.air.mlir` (`aircc_artifacts.py:63`). `num_channels` becomes an observed column with
structure: [23](23-rules-and-open-items.md) rule — two shim MM2S channels per column, budget per
COLUMN across the segment, exceed it and AIR packet-multiplexes (`SHIM_DMA_CHANNELS_PER_DIRECTION =
2`, `aircc_artifacts.py:69`; job 238 shows `shim ch / tiles 17 / 8`, `norm_tail: 24 cores … shim
17ch over 8 tiles, flows 16/40`). Inference mapping (one in + one out stream per worker): herd
`[1,2]`/`[1,4]`/`[1,8]` → 1 MM2S per column, under budget = iron's 2/4/8 cores; `[2,8]` 16 workers, 2
per column, **exactly at budget** = iron's 16 cores / 8 tiles; `[4,8]` 32 workers, 4 per column,
**over — packet-multiplexed, not expressible in iron**. The genuinely new question: what crossing
the per-column budget costs in bandwidth. Sketch (estimates): `builders/mem_copy.py` ~150–250 lines;
`study/memcpy_bandwidth.py` ~250–350 (iron's runner 586 with plotting); `cases.py` ~40; gate ~100;
5 rungs × 2 arms via `devq.sh --class measure`, Turbo required. Risks: the over-budget rung drives
into the regime that before H9 silently misdelivered every trip after the first on >1 column
([23](23-rules-and-open-items.md)); iron's kernel arm is red at 8/16; a `Default`-pmode bandwidth
number is worthless. Build it as a ceiling plus compiler-behaviour probe, not a port.

**Worked bound (inference):** 31a's resident-tail floor 84.0 MiB packaged → 16.5 MiB resident at 1024
(now [31](31-resident-tail-r1-record.md)), at ~67.9 GB/s: 88,080,384 B ≈ 1.30 ms, 17,301,504 B ≈
0.25 ms, **prize ≈ 1.0 ms ≈ 0.65 %** of job 246's 159.795 ms — same verdict at 0.5 or 2 ms. iron's
`roofline/kernel_points.csv` (n=76, OI 0.167 → 8286.9, median 42.7): operators under 30 FLOPs/byte
are `add`, `causal_mask`, `attn_scale`, `attn_softmax`, `add_norm`, `add_norm1`, `add_norm2`, `ln1`,
`ln2`, `gelu` — the norm tail and FFN elementwise; `add` @16384 68.12 / 68.18 GB/s vs peak 67.88
(at ceiling), @256 19.55 / 24.90 (3.5× headroom). Gate-level: `smoke_gate.py:117-118` and
`manifest.py:165` take `--expect` from the caller; no `memcpy` reference under `study/`.

---

## 7. Phase G — unattended runner and CI (was doc 34)

Investigation at tip `b777517b`, nothing built, no device job; pmode `Default`, no latency claim.
Three host-only commands: `smoke_gate.py results/phasef_smoke --expect {coarse,offload,runlist,fused}.csv`
→ `FAIL (4 problems)` exit 1 (v1 CSVs against `SCHEMA_VERSION = 2`, `schema.py:71`, since 2026-08-10;
`results_io.read_rows` `:71-113` rejects header and version mismatch — `missing=['context_loads',
'device_ms','host_cpu_ms','kernel_attaches','sync_ms']`); the same on `results/postflip-ladder-w1` →
`PASS (4 CSVs, each with a passed row)`; `manifest.py` on it → `complete: True`. v2 trees:
`results/ladder-v2-w{1,2}`, `results/postflip-ladder-w{1,2}` (8 CSVs); 56 CSVs are v1. **Phase G's
literal gate sentence is satisfiable today; what is missing is the word *profile*.**

**Doc 10's framing (now [01](01-original-plan-superseded.md)) — a port of iron's 2,494-line
`unattended_reboot.py` + 1,790-line test — is obsolete.** Already here: `agents/scripts/devq.sh`
(321 lines; FIFO broker, monotonic sequence numbers, build/measure barrier; readers-writer design
refuted by measurement — writer blocked 3197 ms while a later reader acquired in 4 µs, `devq.sh:4-10`;
all 23 `flock` sites migrated; `devq-selftest.sh` 20/20 in 6 groups; liveness via
`/proc/<pid>/environ` `DEVQ_JOB_ID`, `devq.sh:80-92`; 248 jobs on disk; `run` vs `submit` footgun —
`submit` blanks the FileCheck and exits 0, `devq.sh:26-30`, `phases.sh:717-727`); the port-loop
driver (`port-loop.sh` 645, `phases.sh` 2,260, six `lib-*.sh`, `phase_e_checks.py` 770,
`phase_e_selftest.py` 365) with `pl_assert_gate_ran_hardware` in `lib-guard.sh` (the most reusable
thing for CI — `Passed` must reach the tracked `.lit` count, `Passed`+`Excluded` the only nonzero
categories); the study tier (17 modules, host tests 231/231 → **265/265 in 17 modules** after M4,
~0.4 s, pinned by `run_study_host_tests.lit`): manifest (`complete` = measured, `manifest.py:14-25,148`),
smoke gate (`smoke_gate.py:56-110`), `require_npu_power_mode_turbo` (refuses, iron warns;
`registry_sweep.py:209-225`), `run_lock.py` and `power.py` (ported, tested, **no callers**; every
`avg_power_w` column empty), `compare_roots.py`, `cases.py:76-166` (6 families × 9-point ladder,
consumed only by `select_rows.py:70`), `run_ladder.py` (walks mode × seq, rewrites each CSV per rung,
`:189-201`). Absent: resume, `state.json`/job plan/profile, crontab/TTM/thermal/reboot.

**CI side.** `programming_examples/CMakeLists.txt:169-176` already declares
`check-programming-examples-transformer-layer` (doc 10's item 6 would duplicate the target; its
filter `transformer_layer/.*/run_npu2_compile` matches nothing — the test is top-level
`run_npu2_compile_peano.lit`); its comment ("no NPU dispatch … safe as a PR gate") is wrong: **32
`.lit` files, 22 `REQUIRES … ryzen_ai_npu2`, 1 Peano-only, 9 host-only** (matches the recorded
31 pass / 1 unsupported); on an NPU-less runner all 22 report UNSUPPORTED and lit exits 0. No
workflow references it. PR-safe subset: 10 tests — `run_npu2_compile_peano.lit` plus
`run_block_cache_tests`, `run_blocked_attention_tests`, `run_ffn_resident_emulation_tests`,
`run_reference_tests`, `run_seam_tests`, `run_study_host_tests`, `sweep/run_sweep_families_tests`,
`sweep/run_sweep_writer_tests` (`sweep/run_npu2_registry_resolution.lit` has no `REQUIRES` — check).
Precedent for the opt-in half: `nightlyPerfBenchmark.yml` (runner `amdryzenai5pro340`, cron
`17 4 * * *`, `cancel-in-progress: false`, `timeout-minutes: 300`) — a Krackan Point NPU2, not this
Strix laptop; study results cannot ride it.

**The missing list, by value per hour.** M1 pmode guard on the two latency gates — `gate-h.sh` leg 4
(floor 11.1 tok/s × 0.85 = **9.435 tok/s**, recorded `2026-08-06T18:24:05Z` at `d72a2ccf` inside the
Turbo window 08-03 → 08-10; `throughput-baseline.json` records `recorded_utc`, `recorded_at_sha`,
`n_tokens`, `prompt`, `context_len` and not the pmode) and `run_npu2_runlist_gate.lit`'s clause; at
`Default` both fail spuriously (the verdict rung reads ~2.5–2.7 s against 156 ms at Turbo).
**`[2026-08-12]` DONE (queue item 13)**: `port-loop/pmode_guard.py` imports `require_turbo`;
`gate-h.sh` leg 0 refuses before the build, re-checks at leg 4, leg 3 exposed too; `runlist_gate.py`
refuses with exit 2 and a banner the lit matches; the floor file carries `npu_power_mode`, the seed
script refuses off Turbo, leg 4 refuses a mismatch; no number moved; `pmode_guard.py selftest` 11/11
both directions; the shipped floor's pmode recorded `unknown`. M2 no suite profile. M3 manifest
validates files not rows (`manifest.py:130-150`; keys `complete, created_at_utc, expected_files, git,
incomplete_reasons, missing_files, repo_root, results_root, schema_version, study_id, system`). **M4
CLOSED `[2026-08-12]` (queue item 15)**: `schema.CONDITION_FIELDS` `conditions` block
(`npu_power_mode`, source `observed`/`probed_at_manifest_build`/`unknown`, provenance, when) written
by `build_manifest` — a block, not a column, because a column bumps `SCHEMA_VERSION` to 3 and takes
the 16 surviving v2 CSVs out of every reader; `SCHEMA_VERSION` stays 2, pinned; old manifests read
`unknown`/`absent`; `compare_roots` refuses a recorded mismatch, flags unknown; 6/6 v2 root pairs
byte-identical vs the pre-change binary, 16/16 v2 CSVs parse; `xrt_version`/toolchain pin are
declarations (note `compare_manifests` diffed a `toolchain` key never written). M5 runner reaches one
family (`run_mode.py:170` hardcodes `encoder_bert`, `_shape_for` varies only `seq_len`) — **corrected
by G1/§9**. M6 `run_status="skipped"` declared (`schema.py:640`) and never emitted — must land before
M3. M7 no resume (copy `REUSABLE_STATUSES`, `registry_sweep.py:177`). M8 nothing orchestrates (the
Phase F manifest was hand-assembled: `results_root /tmp/phasef_results`, `repo_root
…/.claude/worktrees/phase-f`). M9 `run_lock`/`power` callers (~20 lines each). M10 TTM/empty-mask
items belong to the plot tier (11(b); matplotlib/pandas/seaborn must not be installed while gates
run). M11 CI target rename/re-filter/re-comment (~1 h + one round trip). M12 preflight collapses to
one sudo binary, `xrt-smi configure` (`turbostat` unusable, `sudo -n` fails; `doctor.sh:292` has the
shape, checks no pmode).

**Hazards CI must encode.** 3.1 pmode resets on reboot/`amdxdna` reload (`sudo xrt-smi configure
--device 0000:64:00.1 --pmode turbo`); fail-closed in the study path (`run_mode.py:392-400` returns
2 writing no row; per-rung re-check; synthesized failed row `run_ladder.py:158-176`); rules: refuse
not warn, record the observed pmode, treat "could not determine" as refusal, never splice across a
pmode change. 3.2 the two CMake flags `-DXRT_COREUTIL=/opt/xilinx/xrt/lib/libxrt_coreutil.so
-DENABLE_RUN_XRT_TESTS=ON` ([15](15-environment-notes.md)) — lost on a wipe, not a reconfigure;
`utils/build-mlir-air-using-wheels.sh` sets `XRT_LIB_DIR`, `XRT_BIN_DIR`, `XRT_INCLUDE_DIR`,
`ENABLE_RUN_XRT_TESTS=ON` (117-120) but no `XRT_COREUTIL`; current config `xrt_bin_dir =
"/opt/xilinx/xrt/bin"`, `enable_run_xrt_tests = ON`; `buildAndTestRyzenAI.yml:126-127` and
`nightlyPerfBenchmark.yml:141-142` pass `ENABLE_RUN_XRT_TESTS` but not `XRT_COREUTIL`; assert the
config lines before and the lit category invariant after. 3.3 a laptop: lid suspends, battery idle 15
min, `KillUserProcesses=no`; `systemd-inhibit --what=handle-lid-switch:sleep:idle setsid nohup … &`
is documented and implemented in no script; the measurement half is a local operator-invoked
script. 3.4 three known-red lit failures outside every gate
(`llms/llama32_1b_int4/multi_launch_builder/run_o_gemv_ffn_int4_fused_npu2_peano.lit`,
`conv2d_14x14/run_npu2_makefile_peano.lit`, `matrix_vector_multiplication/bf16/run_npu2_makefile_peano.lit`)
plus six NPU1-only in `check-programming-examples-peano` (`matrix_multiplication/{bf16,i16,i8}`,
`-mllvm -aie-disable-fold-imm`); an explicit allowlist keyed by path, shrinking fine, growing needs a
reason (`buildAndTestRyzenAI.yml:152`'s `|| ninja` retry is a different mitigation). 3.5 two lock
inodes never unified: `/tmp/mlir-air-npu.lock` (`flock -x -w 1800`) vs `/tmp/npu.lock`
(`KernelCache` and lit suites; taking it from a wrapper deadlocks) — 15 call sites; `devq.sh:40`
takes only the first and refuses to nest (exit 2, selftest 6); take the device through `devq run
--class measure`, never name a lock path. 3.6 `hw_context` ceiling 32 (33 → `DRM_IOCTL_AMDXDNA_CREATE_HWCTX
err=-2`); in-process looping made `runlist` fail 2048 and 4096 with "Failed to load ELF kernel …"
(`run_ladder.py:32-52`); one process per rung; `-j1` on hardware suites (the transformer-layer
target does not pass it, `CMakeLists.txt:174`; the four `llms` targets do; suite passed at 24 workers
30/30 in 519.7 s but the runlist gate went intermittent under that contention — two targets, one
measurement-adjacent at `-j1`). 3.7 gates leak artifacts (`.o`, `air.mlir`, `air.elf`,
`air.insts.bin`, `air_project/`, `*_cache/`; eleven committed in `bf69ed69`, D2's 6.3 MB
`block_cache/`; "a new artifact directory is the default outcome of adding a `KernelCache`-backed
gate"); runner asserts `git status --porcelain` clean, every cache dir joins `.gitignore` and
`clean` in the same commit; a results root is ~2.4 GB — retention is part of the design.

**Effort:** M1 1–2 h; M11 1 h + CI round trip; G0 (profile + runner + manifest counts: M2, M3, M6,
M8, M9) ~400–600 lines + one devq measure window; M7 ~150 lines; M5 was sized "unbounded — C4's
504 + 66 min" (**wrong, §9**). Drop from doc 10, decided against: `@reboot` crontab (sudo puts it in
root's crontab); TTM page limits (`amd-ttm` nowhere; iron's 26 GB override was for six 16384-token
iGPU `host_comparison` jobs, unported, ROCm wheel conflicts with the CPU-only index); thermal gating
(no artifact shows throttling); `turbostat` (replaced by `power.py`'s two root-free sysfs backends).
Dropping them removes the passwordless-sudo block except `xrt-smi configure` and the reboot-loop
class that halted iron at job 885 of 888. First increment: M1 alone, then G0 — `study/profiles.py`
(named profiles of `(mode, family, seq)` rungs with expected CSVs derived; `smoke` = 4 × 1,
`ladder` = 4 × {512,1024,2048,4096}, `full` = what the matrix reaches; iron's 888/834/21/3 must not
become acceptance criteria), `study/run_profile.py` (`require_turbo()`, `run_lock` on the root,
`run_ladder.walk`, `smoke_gate` + `manifest`, invoked as `devq.sh run --class measure -- python3
study/run_profile.py --profile …` under `systemd-inhibit`), `manifest.py` gains `expected_rows`, row
counts, `conditions`; host tests pinned and verified shrinking.

**Wall clock, artifact-backed:** 8 cold rungs (4 modes × {512,1024}) **631 s** (job 224, per-rung
98/102/29/30/55/57/128/132 s); same 8 warm **32 s** (5/5/2/3/6/7/2/2); both walks 11.0 min
(`job-000224.meta`); 16-rung walk warm ~90 s, cold ~45 min (§1); full transformer-layer suite (32
tests) 8.1 / 8.3 / 8.5 / 8.7 / 8.9 / 8.6 min (jobs 177/132/135/212/241/248); ten-model `make verify`
63.2 min (job 222); suite + ten models 73.6 / 73.6 / 73.8 min (jobs 081/085/075); C4 sweep 504 + 66
min; Phase B 362 min; host suite ~0.4 s. Compilation dominates: ~20× cold/warm swing. Estimate
(inference, ~2.8 min/rung cold): `smoke` 4 rungs ~5 min / ~15 s; `ladder` ×2 32 rungs ~45 + ~2 min /
~3 min; one family × 9 × 4 ×2 72 rungs ~1.7–2 h / ~5 min; six families 216 rungs ~10 h; + registry
sweep +8.4 h; "full" as doc 10 imagined ~18–20 h cold, ~2–4 h warm (iron's 11 h – 2 days for 888
jobs is the sanity check). **`[2026-08-20]` one full profile has run: 1902 s cold for `baseline_768`'s
36 rungs (devq 427), 419 s warm (devq 435) — [54](54-first-full-profile-and-decoder-families.md).**
A full profile monopolizes the NPU (`measure` is an absolute barrier); cadence: `smoke` per change,
suite nightly (~9 min), `ladder` weekly (~45 min cold), `full` on explicit dispatch. **Two walks is
the standing rule** (README trap 1). The honest blocker was M5, not effort.

### 7.1 `[2026-08-12]` G1 — resume, doc 10 item 5, and the coverage sweep measured

Worktree `worktree-agent-a0bab073cb6414184` off `39a08a8b`; commit `869b8684`; Turbo verified.
**M7 CLOSED**: `study/resume.py`, `run_ladder.walk(..., reuse=, on_rung=)`,
`manifest.build_manifest(walk=)`, `schema`'s `WALK`/`SESSION`/`SESSION_RUNG` blocks, `run_profile
--resume`. Guarantees: a rung with a `passed` row is not re-run; every rung appears once; the
completeness verdict is unchanged by resuming (row-count clauses know nothing about sessions); a
reused rung whose final row does not hash to the prior digest is a **`RESUME DEFECT`**; an unclaimed
row is `rungs_unattributed`; a splice across power modes is refused, across toolchain/git sha
flagged. Cannot: make a spliced walk one measurement; resume mid-rung; detect a distorted
measurement. Attribution keyed by `(execution_mode, seq_len)` outside the CSV (a `session_id` column
would bump to v3 — item 15's decision not revisited). **Only `passed` is reused** (unlike the sweep's
`REUSABLE_STATUSES`: a registry row is keyed by a `MEASUREMENT_CONTRACT` hash; a results CSV is not,
and a retained failure is a claim about code that may be gone); a skip is re-derived every session.
The ledger is written per rung by the walker and re-hashed afterwards (bookkeeping agrees with
itself whatever the walk did). Also closed: `run_profile`'s gate never passed `conditions=`, so every
profile manifest read `npu_power_mode: unknown` on a run that had refused to start off Turbo —
"never stamp a condition you did not observe", inverted. **Doc 10 item 5 SPLIT**: the prerequisites
table dropped (five of six tools already dropped; `xrt-smi` only read, `require_turbo` refuses when
missing); `run_profile.environment_problems()` refuses at start on the two Python modules a bare
devq shell lacks (`pyxrt` — not added by `env_setup.sh`, dies at first dispatch with a
`ModuleNotFoundError`; `ml_dtypes` at first builder import) and on a cwd that is not the example's;
README §"Running a profile: invocation and recovery" written. **M5 CORRECTED** — §9. Negative
controls named in `study/test_resume.py`'s docstring (the load-bearing one: a walker that ignores
`reuse` and re-measures — only re-hashing catches it). Host suite **357 → 409 in 20 modules**
(renamed test → 395/395 refused; hidden module → 372/372 in 19 refused); `SCHEMA_VERSION` stays 2.
Still open after G1: `baseline_1024` (then walked, §9.7), the three decoder families (D2-class
integration per mode; `gpt2_small_768` cheapest), two walks into two roots, and
`tree_dirt_after_run` cannot distinguish a leak from an author (job 304 listed eleven modified, zero
untracked — narrow to untracked paths).

---

## 8. iron `encoder_pipeline` — extraction reference (was doc 38)

Extracted 2026-08-12 from `/home/cj/iron`, branch `extend_enc_pipeline` (tip `64a1f29`, 27 commits
ahead of `devel`, unmerged); read-only. Citation forms: `iron:<path>:<line>` (on the branch);
`iron@<rev>:<path>` (deleted, recovered via `git show`); **[MEASURED]** / **[INFERENCE]** /
**[UNSOURCED]** as marked.

### 8.1 Provenance warning — read before using any number

**Three operator generations, not one.** G1 `operators/encoder_pipeline/` (pre-split; deleted
`b5538a7`, `a1327c9`, 2026-03-13; knobs `parallel_heads`, `parallel_ffn`=`nB_tiles_distributed`,
`proj_acc_depth`, `o_proj_acc_group_size`; runtime switch `ln1_staging_design ∈ {"memtile","ddr"}`).
G2 `encoder_pipeline_ddr/` (deleted) + `encoder_pipeline_memtile/` (survives; `op.py:69-70` `del
ln1_staging_design`, `design.py:153-154` `stage_ln1_to_ddr = False`). G3 `iron/operators/encoder_pipeline/`
(new from `70d2e29`, 2026-03-17; `design.py` 6764 lines vs G1's 4962; adds `parallel_seq`,
`seq_tile`, `kv_seq_tile`, `emb_tile`, `ffn_tile`; **DDR staging only, no switch**; test-ID scheme
`test.py:887`). `encoder_pipeline_archive/README.md:6-8` still names `_ddr` as live — stale.
**`current_status.md` documents G1**; its case IDs, `lnstage_*` selections, constraints and
`full`-vs-stage numbers are G1. The brief's "on no other branch" is false (`design.py` also on
`dev-mha-an-combine-bufs` 4512 lines, `thesis_design_patterns` 4459, `update_bert_clean` 4459,
`origin/dev-encoder-pipeilne` 954, `origin/update_bert_rebased` 3651); the load-bearing half holds —
on neither `devel` nor `final_exec_strats`, the tree the mlir-air study was ported from.

**No machine-readable measurement artifact exists anywhere in the repo** (`git log --all
--diff-filter=A -- '*encoder_pipeline*results*.csv'` empty; `--output-json` defaulted to `None`,
`profile_debug_modes.py:176-181`; stage-profile tests carry no `@pytest.mark.metrics`). **Every
latency in `current_status.md`, `design_optimization_plan.md`, `debug_findings_2026-03-04.md`,
`ffn_latency_optimization_log_2026-03-08.md` is hand-transcribed console stdout** — single-run
readings (forensics: exact differences `16488.12 − 11739.34 = 4748.78`, `12226.31 − 8984.99 =
3241.32`, `4349 − 3514 = 835`; two-decimal precision matches the profiler's `:.2f` at
`profile_debug_modes.py:224`, not pytest's `:.1f` at `:332`; nothing computes the subtraction).
Recoverable only from git: `design_ln1_ddr.py` (482 lines), `profile_debug_modes.py` (265),
G1 `test.py` (370), `design_ln1_memtile.py` (102), `op.py` (595), `reference.py` (202) at
`b5538a7^`; `0fe8e8c:operators/encoder_pipeline/README.md` lines 111, 126-127. `b5538a7`/`a1327c9`
reachable only from `origin/update_bert_rebased`. Working copies were in that session's
`scratchpad/iron-ref-private/` (prefix `DELETED__`).

### 8.2 What the fused design is

`MHA + AddNorm1 + FFN + AddNorm2`, one BERT layer, bf16, `d=64` only (G3 `README.md:16`); LN1/LN2
two-pass (every LN input FIFO fed twice). Eight compute stages, one core each (G3 non-seq path):
0 QKV projection (optional, `staged_hidden_states`, `design.py:5263,5297,5330`); 1 QKᵀ
`batched_matmul_qk` `:1545`; 2 online softmax `:1562`; 3 PV + rescale `:1601`; 4 O-proj +
cross-head accumulation `:1679` (grouped `:1778-1869`); 5 AddNorm1 `:1914`/`:1956`
(`ln_calc_sum_sumsq`, `fused_add_layer_norm_1outs_fp32weights`); 6 FFN up **+ GELU in the same core**
`:1995`, `:2012` (no separate activation core — against R1's own GeLU herd); 7 FFN down + reduction
chain `:2014` (`:2118-2160`); 8 AddNorm2 `:2217`. Hand-offs: core→core direct (QK→softmax `memA`
`:1324`, softmax→PV `memP` `:1334` + `scaleOF` `:1344`, PV→O-proj `:1348`, O-proj head cascade
`outOPart` `:1358`, O-proj→LN1 `:1405`, FFN-up→down `ffnUpOut` `:1527`, FFN-down branch chain
`:1527-1533`, FFN-down→LN2 `:1534`; tiles adjacent — `:533-536` rows 2/3/4/5 of one column per
head); memtile at `row=1` (Q/K/V fan-out `:1183-1208`, W_O `:1313-1321`, O-proj accumulation loop
`:1352, 1390-1404`, FFN-down loop `:1497-1526`, weight staging `:1453-1526`, LN2→shim `:1536-1543`);
**exactly one interior DRAM crossing: LN1's output** — per Q-block (`:6449`) `rt.drain(ln1StageOut…)`
`:6683`, `finish_task_group` `:6691`, `rt.fill(inLNFromDDR…)` `:6694` → FFN-up, `ffnRFromDDR` `:6702`
and `:6710` → AddNorm2 residual passes 1 and 2; four shim transactions per Q-block; no on-chip edge
LN1→FFN-up in G3 (seq-par path `:5067-5080`, `:5200-5213`, `:5215-5241`); scratch in the `OR` tail
(`:896`, `ln1_dram_stage_rows` `:474-476`). **One configuration**: `Program(NPU2(), rt).resolve_program(SequentialPlacer())`
`:621-623`; one kernel, one runlist entry, five BOs (`op.py:1313-1318`, `:1364-1376`; `O` aliases
`OR` `:1363`); workers `rt.start`ed once (`:6353-6367`) with infinite loops, host `rt.fill`/`rt.drain`
in task groups. **iron does not have the multi-segment residency constraint because it never
creates a second segment** — it sidestepped the problem R1 faces. iron's benchmark names three NPU
paths — `encoder_pipeline` (fused), `gemm_only` (`4 + 2*num_heads` dispatches), `operator_runlist`
(`BENCHMARKING_METHODOLOGY.md:12-16, 137-181`); the fused one wins [MEASURED]. iron's `fused` is
resident with one exception; mlir-air's shipped `fused` is packaged [INFERENCE].

### 8.3 Balancing machinery

Knobs (`topology.py:53-74`): `seq_tile`, `kv_seq_tile`, `emb_tile`, `ffn_tile` (no core effect);
`parallel_seq` (× whole 8-stage lane), `parallel_heads` (+4 cores/head), `parallel_ffn` (+2
cores/branch), `proj_acc_depth`, `o_proj_acc_group_size`, `ffn_down_acc_group_size`; six layout
overrides (`topology.py:16-25`). Constraints: `emb_tile * proj_acc_depth == embed_sz`
(`design.py:361-365`); `ffn_intermediate_size % (emb_tile * parallel_ffn) == 0`
(`memtile/cases.py:62-64,115-116`); `opg <= parallel_heads` and `parallel_heads % opg == 0`
(`:117-118`); `nB_tiles_distributed <= ffn_col_groups` (`memtile/design.py:234-238`); `seq_len %
seq_tile == 0`, `(seq_len // seq_tile) % parallel_seq == 0` (README:76-78). Budgets
(`memtile/design.py:530-560`, `resource_utilization_2026-03-04.md:3-15`): `parallel_heads*4 + 3 +
2*effective_ffn_branches <= 32`; `5 + estimated_b_weight_streams + ln1_ddr_streams <= 16` (5 fixed =
Q/K/V/W_O/R; `ln1_ddr_streams` 0 memtile / 1 ddr); memtile DMA 6 in / 6 out, compute-tile 2 / 2,
memtile BD budget 48. Routing rule with no AIR analogue: the FFN-down reduction chain must be N/E/S
neighbours ending adjacent to LN2, else "chain minus one bypass core" with two LN2 inputs
(`memtile/design.py:296-357`); westward edges rejected (`:256-264`). **Selection = three
mechanisms, none derivation**: (a) G1/G2 greedy auto-prune (`while True:` + `prune_or_fail`,
`:489-565`, drops one FFN branch per retry); (b) G3 hand-authored `TOPOLOGY_PLACEMENTS`
(`placements.py:155`, 13-tuple key → explicit `(col,row)`), `LOW_HEAD_TAILS` (`:86-152`) with only 7
`(ph, pffn)` combos — (1,1) (1,2) (1,4) (2,1) (2,2) (2,4) (4,1) — and
`SUPPORTED_ENCODER_PIPELINE_TOPOLOGIES` (`:1538`); (c) exhaustive measured autotune with a per-shape
cache (`npu_inference.py:169-204`: `--topology-policy {fixed, cache, autotune}` default `cache`,
`--topology-cache npu_topology_cache_latest.json`; `cooldown_before_benchmark` between candidates;
`select_autotune_topology` = **1 % latency band, then max compute tiles, then current family, then
id**; cache key `topology.py:508-522`, `find_cached_topology` `:525-542`). **The winner is not a
balanced pipeline** [MEASURED `PERFORMANCE_IMPROVEMENT_PLAN.md:24-27`]: `seq32_kv64__ps2_ph2_pffn2`
at seq64, `seq32_kv64__ps4_ph1_pffn1` at seq128+ — four lanes × 8 cores = 32 cores
(`placements.py:1214-1257`), each a private copy of the whole chain over a sequence slice, shared
K/V/W_O ingress (`:1193`), `transport_groups` of two (`:1258-1310`); "remaining wins are more likely
to come from lower data movement and synchronization" (`:104-106`). [INFERENCE] **The question for
a resident AIR interior is not "how to balance three herds" but "how many copies fit".**

### 8.4 The balance metric (full vs isolated-stage gap) and its two defects

[MEASURED, hand-transcribed] `current_status.md:38-53` (also `design_optimization_plan.md:20-39`,
`debug_findings_2026-03-04.md:16-28`): memtile control `512seq/96e/4ph/4pffn/8pacc/4opg` full
16488.12 µs, max stage `addnorm1` 11739.34, gap 4748.78, ratio 1.405; "fast DDR"
`512seq/128e/4ph/6pffn/6pacc/2opg` 12226.31 / 8984.99 / 3241.32 / 1.361; high-`pacc`
`64seq/64q/48e/6ph/1pffn/16pacc/2opg` ~4349 / 3514 / ~835 / 1.238 (full table
`design_optimization_plan.md:103-113`: `self_attn` 1637, `mha_input` 1568, `residual` 1572, `ffn_up`
1590, `ffn_down` 1387, `addnorm2` 1063, `mha` 1068, `addnorm1` 3514, `full` 4349); memtile control
after LN micro-opts (`ffn_latency_optimization_log_2026-03-08.md:48-59`): full ≈16065, `mha` 7442,
`ffn_up` 7340, `ffn_down` 6647, `addnorm2` 7170, `addnorm1` 12207, `addnorm1_stats` 12797,
`addnorm1_post` 7740. Timing: `perf_counter` around `xrt_kernel(...).wait()` only
(`aie_base.py:263-272`, runlist `:222-225`), BO syncs outside; arithmetic mean over `timed_iters`
after `warmup_iters`, inputs written once (`test_utils.py run_test`); `warmup=3, timed=20` (G1,
profiler), `warmup=10, timed=100` (G3 `test.py:1186-1187`); pytest adds `--iterations` default 5
(`conftest.py:29-34`) with a `CSVReporter` (`:78-95, 100-126`) — mean-of-5-means-of-20. Isolated
stages are **separately compiled binaries** keyed by `debug` in the artifact stem (`op.py:225-227`)
and `-DDEBUG={mha_debug}` (`:365,369`), clean rebuild per stage. Mode table
(`memtile/debug_modes.py:6-53`, `(mha_debug, ffn_stage_only, an1_mode, an2_mode)`): −1 `full`
`(0,None,−1,−1)`; 0 `self_attn` `(1,None,0,0)`; 1 `mha_input` `(−1,None,0,0)`; 2 `residual`
`(−1,None,1,1)`; 3 `ffn_up` `(−1,0,0,0)`; 4 `ffn_down` `(−1,1,0,0)`; 5 `addnorm2` `(−1,2,0,0)`; 6
`mha` `(0,3,0,1)`; 7 `addnorm1` `(0,4,−1,1)`; 8 `addnorm1_stats` `(0,5,−1,1)`; 9 `addnorm1_post`
`(0,6,−1,1)`. The dataflow graph is preserved, only arithmetic neutralized (`memtile/design.py:2635-2636`).
The max is computed (`contributes_to_bottleneck` True only for `ffn_up, ffn_down, addnorm2, mha,
addnorm1, addnorm1_stats, addnorm1_post`; `_find_bottleneck` `:119-132`); the subtraction is not.

**Defect 1 — `addnorm1` is a prefix, not a stage** (`mha_debug=0`, the whole MHA computes): LN1's
marginal cost in the high-`pacc` table is ~2.4 ms, not 3.5; stage latencies are neither additive nor
disjoint; `full − max` under-estimates exposed serialization when the max is a prefix (all three
cases). **Defect 2 — weight DDR traffic elided in the winning mode** (`need_bup_weights =
ffn_stage_only in (None, 0)`, `need_bdown_weights = … (None, 1)`, `memtile/design.py:1910-1911`,
`:4288-4289`): `addnorm1` is `ffn_stage_only = 4`, so `B_Up` (768×3072) and `B_Down` (3072×768) —
~9.4 MB bf16 — are never fetched; a fraction of the 3.2–4.7 ms "gap" is weight traffic;
`current_status.md:56-58` over-reads its instrument. Oddities: `addnorm1_stats` 12797 > `addnorm1`
12207 (impossible if stats ⊂ addnorm1); "FFN-up remains the dominant stage" (`…03-08.md:20`)
contradicted by its own table [UNSOURCED]. **Minimum viable port:** N+1 compiled variants keyed on a
stage enum in the artifact identity; full staging/FIFO graph, compute only for stage k; **do not
elide inactive-stage weight DMAs**; make each max a true single stage; same stopwatch, mean over
≥20 after ≥3 warmups, record the full distribution; **always emit `full`, `max_stage`, `gap`, ratio
to a JSON/CSV**.

### 8.5 `memtile` vs `ddr` staging — "DDR beats on-chip" does NOT hold as stated

`memtile`: LN1 broadcast on-chip (`memtile/hooks.py:6-8`, `:31-53`). `ddr`: drain once from one
source branch, refill and broadcast (`design_ln1_ddr.py:8-9`, `:197-243`, `:97-119`). Four
confounds: different topology (96e/4pffn/8pacc/4opg vs 128e/6pffn/6pacc/2opg,
`debug_findings_2026-03-04.md:19-24`); the like-for-like run was stated (`:13-15`) and the DDR number
at the control topology and the memtile number at the fast-DDR topology (`memtile/cases.py:42-45`,
`(512,64,12,3072,32,64,128,4,6,6)`) are **missing from every artifact**; different runtime schedule
(DDR ships a software-pipelined `schedule_runtime_tap`, `design_ln1_ddr.py:373-461`, deferring
`tg_ln1_refill_and_weights` to the next Q-block `:401-406`; memtile finishes inline
`memtile/design.py:200-211`; `should_prefill_ffn_weights` DDR False `:365-366` / memtile True
`:190-194`; `adjust_wait_ffn_weight_fill` `:357-362` vs `:181-189`); different resource envelope
(DDR +1 shim stream, frees memtile output channels; the 16-head/4096-ffn/128-embtile family is `ddr`
only, `resource_utilization_2026-03-04.md:45`, `_DDR_ONLY_REGULAR_CASES` `cases.py:215-228`).
Supported: one uncontrolled pairing, 12226.31 vs 16488.12 µs. Not supported: DRAM staging faster at
equal topology and schedule. [INFERENCE] DDR staging is a **resource-relief move** that unlocked 6
FFN branches at `emb_tile=128`. Strongest evidence is revealed preference: G3 stages LN1 through DDR
unconditionally (`:6683-6717`, `:5067-5080`, `:5200-5241`); `mode_split_plan.md:315-328`,
`memtile/op.py:170` default `"ddr"`. **The generalizable rule: eliminating a DRAM crossing is paid
in on-chip channel and memtile-BD budget — the crossings worth eliminating are point-to-point; a
high-fanout broadcast may be cheaper through DRAM** (eight of nine interior hand-offs on-chip, the
broadcast one through DRAM).

### 8.6 Failure modes mapped onto R1's walls

Pressure points (`current_status.md:69-76`): O-proj accumulation L1 (`memOW*_cons`), FFN-down
root-core L1 (`memBDown*_cons`), tail memtile output channels on `64q/48e/16pacc`, final `memLN2`
shim drain; `ERT_CMD_STATE_TIMEOUT` on `4pheads_6pffn_8pacc` (`debug_findings:54`); a split-weight
TAP order bug (mismatches 32702 → 7, `:38-41`); a grouped FFN-down accumulation ordering bug
(`:43-46`). **Shim-drain wall = R1 wall 4's family** ([31](31-resident-tail-r1-record.md)),
strictly harder: `'aie.dma_bd' op Allocator exhausted available BD IDs (maximum 24 available for
channel 0)` (1ph/1pffn/12pacc) and `channel 3` (4ph/4pffn/12pacc/4opg), `shim_drain_bd_repro/README.md:41,86`;
the exhausted object is the LN2 output drain `@memLN2`; frozen MLIR at three stages checked in. The
limit is BD **count per channel**, not descriptor shape (`:136-143`: a direct row-major drain did not
move the boundary; the driver is the runtime output tap count); the output drain is the last thing
to break (`:54-56`); iron never solved it — it narrowed the envelope (`12pacc` excluded,
`resource_utilization:40-45`). **Wait relaxation** (`current_status.md:64-67`; `optimization_plan.md:36-38`;
`PERFORMANCE_IMPROVEMENT_PLAN.md:165-166`): removing `wait=True` on per-head K/V/W_O fills
(`design.py:6540-6618`; W_O already conditional `wait=not use_staged_hidden_state_kv_cache`
`:6617`) broke correctness — structurally a read-before-write race, the same class as R1's wall 7
[INFERENCE], but NOT the same observation: iron never calls it intermittent; iron's gate is a
mismatch budget (`REL_TOL=4e-2`, `ABS_TOL=1.5e-1`, `ERROR_THRESHOLD=0.005` → 1966 of 393216 allowed
at 512seq/64d/12heads, `memtile/cases.py:18-20,155-161`; `nearly_equal` `norm = |a|+|b|`,
`test_utils.py:11-29`; G3 loosened to `0.05`, `test.py:38-40`), accepted on "without changing
mismatch counts" (`current_status.md:60-62`); its only recorded nondeterminism is a timeout
(`PERFORMANCE_IMPROVEMENT_PLAN.md:209,255,182`; "Timeouts must be treated as real failures",
`:120-122, 507-510, 539`). **Claim only:** "iron independently found that removing per-fill
completion waits on head-block ingress breaks correctness and concluded overlap must come from
structural change". iron's retained overlap = **task-group deferral** (`design_ln1_ddr.py:401-406,
448-461`; G3 `:6743-6744, 6746-6758` via `pending_ln1_refill_tg`) plus the targeted
`wait=effective_ffn_branches > 1` on B_Up/B_Down (`:6728, 6739`).

### 8.7 Transferability

As design: one program / one dispatch / N BOs; stage-truncated binaries as a balance instrument
(fix §8.4's defects first); task-group deferral; the 1 % band + prefer-more-tiles tie-break; thermal
cooldown between candidates; per-shape topology cache; route the highest-fanout intermediate through
DRAM; replicate the whole pipeline per lane; neighbour-only reduction chains with a legality check.
As existence proof only: a fully resident layer runs on NPU2 in one configuration (`100 passed, 40
skipped` / `115 passed, 40 skipped` at `ERROR_THRESHOLD=0.005` — 23 base cases, last 3 DDR-only →
20/23 params × 5 iterations = 100/115; 8 stage modes × 5 = 40 skipped via `ENABLE_STAGE_PROFILE_TESTS
= False`; counts predate the expansion to 42 cases / 36 memtile-eligible, so `current_status.md:23-25`'s
"current validated state" is [UNSOURCED]); both staging modes work; `full/max_stage = 1.238` is
reachable; `aiecc`'s BD allocator is a ceiling a mature design hit and did not defeat. Does NOT
transfer: `placements.py` (1540 lines of `(col,row)`), the `choose_*_mem_tile_col`/`adjust_*_cols`
hooks, the exact latencies (pinned to `mlir_aie==0.0.1.2026031811+71fb44f147`, README:36-44), the
mismatch-budget gate (do not port; cite as a caveat), `nearly_equal`. Known-broken on the branch
(`memtile/README.md:20-31`): `…96embtile_1pheads_1pffn_8pacc_1opg`,
`…96embtile_4pheads_4pffn_8pacc_4opg` (**the memtile control that produced 16488.12 µs** — treat
that number as historical), `…48embtile_6pheads_2pffn_16pacc_2opg` (compile-time memtile DMA
pressure). File map: G3 `design.py` 6764 (seq-par `:2400-5261`, non-seq `:5263-6760`), `op.py`
1829, `placements.py` 1540, `topology.py` 833 (no search logic), `test.py` 1330; memtile `design.py`
4674, `hooks.py` 97, `debug_modes.py` 65, `cases.py` 241.

---

## 9. The coverage sweep, costed: the estimate was wrong by two orders of magnitude (was doc 50)

`[2026-08-12]` Phase G item (c); worktree `worktree-agent-a0bab073cb6414184` at `39a08a8b`. §7's M5
sized "widen the matrix past `baseline_768`" as unbounded (C4's 504 + 66 min); `profiles.py`
carried it in `UNREACHABLE_FAMILIES` and `test_profiles.py` re-derived it by `ast` from
`opcheck_specs.py`. **It rotted anyway: the test re-derived the wrong file.**

**Answer.** `kernel_registry/details/GEMM_bf16_in_bf16_out.json` holds **36 of 36** projection
triples at each of hidden 512, 768, 1024 — two commits on 2026-08-07, 69 → 103 → 136 rows. The
blocker was `study/run_mode.py::_shape_for` overriding `seq_len` and not the width (~40 lines to
parameterize). `tinybert_512` walked end to end in **301 s** against an estimate of **570 minutes**.
The three decoders are unchanged: `decoder_gpt2` is a distinct layer graph (norm before attention,
plain `elementwise_add` residual — `builders/addnorm.py`'s two-output entry point exists, unbuilt;
causal masked add between score GEMM and softmax that `runlist`/`offload` have no step for; the
causal kernel exists — `opcheck_specs.py` carries causal `mha_out_proj` rows at 8 heads/emb 512 and
16 heads/emb 1024), D2-class ~156 min/mode of integration, no sweep; `gpt2_small_768` cheapest.
Three host-only probes: `check_registry.py` 36/36 per width; `check_resolution.py` through the
owning builder (`qkv_gemm_spec`, `ffn_gemm_specs`, `resolve_gemm_spec(o_proj)`, plus the `drain`
re-resolution — `qkv_proj` pins `fused-cast`; `resolve_gemm_spec` asserts `M % (tile_m*herd_m) == 0`
and `N % (tile_n*herd_n) == 0` after lookup) 36/36 at 512 and 768, 36/36 at 1024 (**§9.7**; first
read 35/36); `check_block_config.py` assembles at all three widths 256…4096. `norm_rows` derives 64
at 512, 64 at 768, 32 at 1024 (`builders/block.py:228-230`: "a row count that happened to fit at one
width is a placement failure at the next"). **Lesson:** a re-derivation is only as good as its choice
of source; `test_profiles.py` now reads the registry, asserts the converse (no unreachable family may
cite a coverage reason), and asserts each declared method gap is still open.

**Existence proof — devq 304**, `measure`, Turbo, cold, 301 s (`job-000304.log`, `exit=0`):
`coarse` @`1024x512_encoder_bert` passed avg 67.495 ms subs 4 herd 49 sync 107; `offload` 116.283
ms subs 22 herd 66 sync 66; `runlist` 99.845 ms subs 13 herd 171 sync 138; `fused` 67.376 ms subs 1
herd 23 sync 13; `manifest complete: True (4 CSVs, rows 1/1 passed 1/1 skipped 0/0 each)`;
attention ran as 8 heads. **No latency quoted as a result** (one walk). Inherited tolerance at emb
512, final boundary: best `mean_rel_L1` 1.253e-2 / `atol_required` 5.218e-2 (1.92×), worst 1.665e-2
/ 6.038e-2 (1.66×) vs the `1e-1` defect ceiling; 768's own is 1.35–1.72×. `tinybert_512` chosen
because `norm_rows` is 64 (the band D1/D2 validated) and `fused`'s packing bound is widest at 512
((256, 2048) vs (256, 1024) at 768/1024). Reachable: `tinybert_512`, `baseline_768`, `baseline_1024`;
refused by name (`run_mode.UNBUILDABLE_VARIANTS`, `Profile.__post_init__`, `test_profiles.py` both
directions): `gpt2_512`, `gpt2_small_768`, `gpt2_medium_1024` — overriding width alone would produce
a bidirectional measurement under a causal name that nothing downstream could detect. `full` is the
nine-point ladder over one family, family a parameter (`--family`), every expected count re-derived
(`fused.csv` at `ladder` expects 2 passed / 2 skipped at 768, 3 / 1 at 512); walking the 6×9 matrix
is three `full` walks plus a decoder integration (~1.7 h each cold, ~5 min warm, inference).

### 9.7 (doc 50 §7) `[2026-08-12]` The `baseline_1024` gap was not a gap — and the width wall

Worktree `worktree-agent-a8cbef1a620b6f8f9` at `35e9c382`, Turbo. `2048x1024x3072` really has no
`drain` row, and the consequence drawn ("`offload`/`runlist` fail at seq 2048") was false: **both
chains are `(seq, h, h)`, `(seq, h, 4h)`, `(seq, 4h, h)` and never resolve a `3h` shape**; the only
`(seq, h, 3h)` consumer is `qkv_proj`, pinning `fused-cast`, present. The probe applied `offload`'s
`_chain_spec` re-resolution (real — it re-resolves a `fused-cast` winner to `drain`, §5.5) to all
four roles instead of three. The shipped check said so all along: `registry_sweep.py --family
baseline_1024 --verify-resolution` → `PASS: all 36 baseline_1024 shapes resolve` (and 36/36 at 512,
768); `_chain_spec` over the 9-point ladder 0 failures (27 → `drain`); `runlist`'s three
`resolve_gemm_spec` calls 0 failures; `qkv_gemm_spec(seq, 1024)` 9/9 `fused-cast`;
`resolve_gemm_spec(2048, 1024, 3072, method="drain")` raises and nothing calls it. **The row could
not have been written anyway**: the entry is `used_by: "Qwen3-0.6B Gate/Up proj"` — arm A
(`add_missing_methods` with a synthetic passing `drain` row) reports `not_owned=[(2048, 1024,
3072)]`, bytes unchanged; arm B (`used_by` rewritten) refused `ShapeAlreadyRegistered` because the
`direct` row carries `mean_rel_L1: 0.0113` (3 s.f.) and `registry_writer._round_rel` emits 2 s.f.;
arm C control (`64x1024x3072`, sweep-owned, method removed and re-added) accepted. `best` is
derived per tier, so admitting a faster `drain` would re-point `best.high` away from the row Qwen3
resolves against. **Rule: a sweep must not change a shipped deployment's kernel; a needed method on
a deployment-owned row is a builder change or a recorded gap, never a hand edit.** No sweep was
run; §3's "~2 min sweep" withdrawn.

**The walk — devq 307**, `ladder` at `baseline_1024`, Turbo, cold (`job-000307.log`): `coarse`
512 71.266 ms (subs 4 herd 49 sync 107, 103 s), 1024 151.113 (4/82/204, 102 s), 2048 326.058
(4/147/397, 110 s), 4096 825.318 (4/276/782, 137 s); `offload` 512 140.153 (38/114/114, 29 s), 1024
215.012 (31 s), 2048 **773.544** (35 s — the rung the record said would fail), 4096 1598.426 (47 s);
`runlist` FAILED ×4 `ValueError: rows (32) must be divisible by herd_x*rows_per_call (64)`; `fused`
512 66.769 (1/23/13, 125 s), 1024 146.538 (1/24/14, 144 s), 2048/4096 SKIPPED (bounded 256..1024 at
emb 1024). Whole walk 871.6 s, `rungs_by_status {passed 10, failed 4, skipped 2}`, `rungs_by_source
{measured 14, reused 0, skipped 2}`, `tree_dirt_after_run` 5 entries, 0 untracked; `aircc` =
`install-xrt/bin/aircc` mtime 2026-08-12 14:03:46 against a 15:44–15:58 walk. Manifest `complete:
False` with `row_counts_checked: true` — correct: `runlist.csv: expected 4 passed row(s), found 0`;
the completeness clause counts rows, not files (`smoke` would have reported `complete: True`). No
latency quoted as a result.

**The real wall: `norm_rows` 32 against a layer-norm block of 64.** `runlist.py:445` builds
`build_layer_norm_module(rows, emb_dim, bfloat16)` with `rows = block.norm_rows(seq, emb)`; the
builder's defaults `herd_x=8, rows_per_call=8` require `rows % 64 == 0` (`layer_norm.py:89`);
`norm_rows` maximises the largest multiple of `NORM_HERD_X = 8` dividing `seq_len` under `addnorm`'s
L1 cap at `NORM_ROW_MARGIN = 0.75`: emb 512 cap 160 → 120 → 64 (9/9 ok); 768 cap 104 → 78 → 64
(9/9); **1024 cap 80 → 60 → 32 (0/9)**. A width wall, failing in ~1–3 s before aircc at every
length; `coarse` is unaffected at the same 32 (`addnorm pre-add 32x1024 x128 dispatches (L1 cap
80)`) — two consumers of one derived row count with constraints differing by 8×. Deliberately not
fixed in the walk that found it. **`[2026-08-14]` CLOSED, narrower than concluded**:
`build_layer_norm_module`'s `rows_per_call` was never derived (defaulted to 8; with `herd_x = 8`
that silently requires `64 | rows`); at emb 1024 each of 8 cores owns 4 rows and **4 was legal all
along**. `derive_rows_per_call` is bounded above by the historical default — wherever 8 was legal it
returns 8 and the emitted IR is byte-identical (asserted at five shapes, with a control that 4 vs 8
really changes the module); an explicit `rows_per_call = 8` at 32 rows still raises.
`run_layer_norm_rows_tests.lit`, 9 host-only clauses. Not yet walked on device (machine at
`Default`); the ladder rung is owed.

**The test that could not have caught this.** The old clause asserted a declared gap's method was
still missing — true of every method nobody asks for. `KNOWN_REGISTRY_GAPS` is now empty;
`test_a_declared_registry_gap_must_be_one_some_mode_actually_demands` requires a gap to name a
`(triple, method)` some consumer pins (`profiles.PINNED_PROJECTION_METHODS`, kept honest by
`test_the_pinned_methods_are_re_derived_from_the_modules_that_pin_them` `ast`-reading
`qkv_proj.SCRATCH_METHOD` and `offload._chain_spec`); the reachability test checks the pinned method,
not just the triple. Negative controls: (1) the pre-fix entry verbatim — old clause passes, new test
REFUSES; (2) `drain` deleted from `1024x1024x1024` — reachability REFUSES naming `_chain_spec`; (3)
`PINNED_PROJECTION_METHODS[(1,3)]` tampered to `drain` — REFUSES; (4) real tree passes. Host suite
**517 → 519 in 23 modules** (517 refused, 518/518 refused, 22-module run refused). Consequences:
`baseline_1024` walked but a four-mode comparison there is blocked until the `runlist` rung is
walked; `full` at `baseline_1024` still attempts 64, 128, 8192, 16384 which no mode has been
measured at; decoder graph and two-walks-into-two-roots still open.

---

## 10. The filed-but-not-fixed census of blind checks (was doc 51)

`[2026-08-12]` Items 17 and 19's "filed, not fixed" lists, worked. By the time the worktree reached
`39a08a8b`, commit **`0d2ae8d5`** ("test: five checks that could not fail, made able to fail") had
landed all five; the work here is the independent re-demonstration, which found one claim in that
commit message that does not hold and corrected a docstring. A fix for a check that could not fail
is itself a check and can have the same disease.

| Check | Could not detect | Proving input | State |
|---|---|---|---|
| `study/test_ladder_report.py` | `ladder_report.load`/`main` — hand-built the post-`load` row shape | `_ok` relaxed `== "passed"` → `!= "failed"` (a skipped rung counted) | FIXED (`0d2ae8d5`); pre-fix 11/11 green, post-fix red at `test_load_does_not_count_a_skipped_rung_as_passed` |
| `study/test_run_ladder.py:31` | `_spec()` stand-in agreeing with the catalogue only until it moved | `coarse` row `ffn_dim 3072 → 4096` | FIXED; pre 8/8 green, post red at `test_the_stand_in_row_still_matches_the_catalogue` |
| `study/test_profiles.py:143` | `expected_files()` vs a typed list, not what `walk` writes | `<mode>.csv → <mode>_results.csv` | FIXED; pre 15/15, post red at `test_every_profile_expects_exactly_what_a_walk_of_it_writes` |
| `study/test_component_groups.py:38` | typed `{mode: path}` naming `cells.py` while `coarse.py` sat unread (both open zero host buckets) | `time_cpu("smuggled_bucket")` in `coarse.py` | FIXED; pre 20/20, post red at `test_the_host_bucket_derivation_can_tell_the_modes_apart` |
| `builders/test_block_cache.py:65` | the gate's own `SPECS` row, transcribed | block gate row `4096x768 → 2048x768` | FIXED WITH A CORRECTION (below) |
| `pattern/test_blocked_attention.py` | anything — no negative control; the two "independent" implementations folding together | `chunked_attention_reference` delegating to `blocked_attention` | FIXED; pre 5/5 green on the collapse, post red at `test_the_two_implementations_are_not_the_same_arithmetic` |
| every host arm's `make` target | the lit's pinned count (`make` ran the script bare) | one test function deleted | FIXED via `lit_pin.py`; bare `9/9 passed` exit 0; `make blocked-attention-tests` exit 2, `lit_pin: FAILED` |
| `--seed`, `llms/verify/verify_runner.py` | a seeded verification not seeded (parsed, never read) | re-introduced `--seed` | FIXED (removed); guard names `['--seed (args.seed)']` |
| `--arch`, `vector_tanh.py` | `aie2` silently built `aie2p` | `--arch aie2` vs `aie2p`, `--print-module-only` | FIXED here |
| `-v/--verbose`, `attn.py` | `-v` did nothing; call site pinned `verbose=False` | AST audit | FIXED here |

Suite 357/357 in 19 modules before and after; no lit literal moved.

**Method and the trap.** Each row: check at `0d2ae8d5^`, defect injected into the production file,
old and new check run on the same tree. `inject.sh` purges `__pycache__` before every invocation —
the first `ladder_report` run read **356/357 with a pristine tree** because `== "passed"` and `!=
"failed"` are both 11 characters and the edit-and-restore happened within one second; CPython
validates `.pyc` on `(source_mtime_seconds, source_size)`, so stale injected bytecode was reused.
**Rule: any before/after that edits a same-length string and re-runs within a second is suspect
unless it purges.**

**`test_block_cache` — the claim that does not hold.** `0d2ae8d5`'s message says each derived check
goes red when the production value moves; for three of four it does. For `test_block_cache.py`,
moving the gate row `4096x768 → 2048x768`: PRE-FIX exit 0, 9/9; POST-FIX exit 0, 10/10; `HEAD SHAPE
= {'seq_len': 2048, 'emb_dim': 768, 'ffn_dim': 3072, 'num_heads': 12, 'head_dim': 64}` — `SHAPE`
silently followed. The pin `test_the_shape_under_test_is_the_gate_s_own_specs_row` asserts `SHAPE ==
rows[0]["shape"]` where `SHAPE` comes from that same row — `dict(x) == x`. Not a defect: derivation
is correct here, drift is **absorbed**, not detected; what the pin catches is structural (row
missing/duplicated, fields `block_config` needs). The docstring ("the assertion that turns that
drift red") was false and is corrected to point at `test_run_ladder.py`'s catalogue pin as the
drift-detecting check (its fixture is deliberately not derived). Trades: `test_run_ladder.py`
transcription → AST-derived relations (not tautological); `test_profiles.py` → a stubbed `walk`;
`test_component_groups.py` → a text scan of the whole package; `test_block_cache.py` → tautology,
accepted and stated.

**Dead flags — the count did not reproduce.** Item 19 recorded "0 across 115 study and agent
scripts", 9 in the wider tree, two fixed, "7 remaining". Measured: **2**
(`flash_attention/dataflow_based/attn.py:709 -v`,
`primitives/vector_examples/vector_tanh/vector_tanh.py:136 --arch`), reproducing in the worktree and
the shared checkout; the gap is scope — the vendored `llvm/` subtree carries 201 and the `sandbox/`
venv 302. **A count published without the root it was taken at is not reproducible**; "7 remaining"
is retracted, superseded by 2, both fixed. `vector_tanh.py --arch` declared `choices=["aie2",
"aie2p"]` and nothing read it (siblings `vector_exp/mul/reciprocal/rsqrt` thread `arch` into
`build_module`); BEFORE both choices exit 0 with a 34-line module, md5
`44bd7e0628c4de646d3839f1bf3f51e7`; AFTER `--arch aie2` exit 2 `invalid choice: 'aie2' (choose from
'aie2p')`, `aie2p` unchanged — fixed by **refusing**, not wiring (`math.tanh → aievec.tanh →
xllvm.intr.aie2p.tanh`, no `aie2` path); the flag kept because Makefiles drive it via `AIE_TARGET ?=
aie2p`, and `make AIE_TARGET=aie2` now stops; a post-parse guard reads the value so it cannot go dead
again. `attn.py -v` wired to `args.verbose` (default unchanged). Audit after: 0 dead flags in the
repository's own sources.

**`lit_pin.py`** reads the pins out of the sibling `.lit` and asserts them — one mechanism, pin in
one place, no `FileCheck` needed (it is in neither `build-xrt/bin` nor `install-xrt/bin`; lit reaches
it by absolute path in `lit.site.cfg.py`). Verified on all six rewired targets (seven invocations):
`block-cache-tests`, `blocked-attention-tests`, `reference-tests`, `sweep-families-tests`,
`registry-writer-tests`, `seam-tests` (`POOL` and `DISPATCH`). **Left out deliberately: `make
registry-resolution`** — blind only to the shape set shrinking (at 12 shapes it prints `PASS: all 12
… resolve` exit 0 while the lit's `CHECK-COUNT-36` goes red); `lit_pin` refuses `CHECK-COUNT`,
`CHECK-NOT` and `{{regex}}` rather than approximating them; the target is parameterised by `FAMILY`
while the lit pins `baseline_768`; device-gated. If closed, teach `lit_pin` `COUNT`/`NOT`
deliberately. Nothing on items 17/19's lists is outstanding.
