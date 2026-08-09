# 26 — Rebuilding the four modes to the corrected taxonomy: three feasibility spikes

`[2026-08-08]` [03](03-measurement-model.md) was corrected by the study's author on this date: the
four modes span **reconfiguration cost against DRAM traffic**, not "who sequences the work". Every
mode as implemented is wrong against that definition. Three feasibility spikes ran to price the
rebuild before any of it is scoped into phases. This document is the scope decision they produce.

**No spike edited a repository file.** Spike B ran 33 device jobs — 976 s (16.3 min) of held device
lock inside a 45-min box, turbo verified via `xrt-smi` before starting. Spike A ran zero device jobs
(reading plus symbol inspection). Spike C ran compile-only, every `aircc` through
`agents/scripts/devq.sh submit --class build`. Scratch lives under
`/home/cj/.claude/jobs/e75c34c9/tmp/{spikeA,spikeB,spikeC}/`.

---

## Read this part first: six things the plan has wrong

The plan has already been wrong twice today. These are cheap to state and expensive to discover
during implementation.

### 1. J2's blocker does not exist. `attn_output` passes on the **first** configuration tried.

[16](16-compiler-work-and-remaining-essence.md):160 says `attn_output` (4096×4096×64) "timed out on
the one configuration tried, out of 828 legal ones; search the rest." There was no search to run:

| shape | method | tiles | herd | result |
|---|---|---|---|---|
| `attn_output` 4096×4096×64 | `drain` | `tk2=256 tk1=32 tn=16` | 8×4 | **0 / 262,144 mismatches**, `mean_rel_L1` 9.417e-3, `abs_err_max` 7.324e-4 against `atol` 2.121e-3 (3.46× margin), 1179.3 µs, 1820.9 GFLOP/s |

Confirmed through a **second independent entry point** — the example harness's own `__main__`
(`programming_examples/matrix_multiplication/bf16_in_bf16_out/run.py:1088`), which printed
`Latency (us): 1199.4 ... mean_rel_L1=9.417e-03 | abs_err max=7.324e-04 ... PASS!`. Byte-identical
error statistics to the sweep record, so this is not a sweep-harness artifact.

**All three GEMM methods work at this shape.** `direct` is faster and lower-precision (2344.8
GFLOP/s, `mean_rel_L1` 1.542e-2); `fused-cast` at `tn=16 ctn=2048` also passes (1586.3 GFLOP/s).

### 2. `attn_scores` was never actually demonstrated either — until now.

The same J2 row asserts `attn_scores` "already passes on hardware". Grepping `agents/`,
`agents/probes/`, `programming_examples/` and `sweep/` for `attn_scores` or `828` finds **prose
only** — no probe, no sweep script, no artifact. The claim entered with `7f27599e` carrying nothing.
It has now been re-established from scratch: `attn_scores` 4096×64×4096 `drain tk2=64 tk1=32 tn=128`
herd 8×4 → 0 / 16,777,216 mismatches, `mean_rel_L1` 9.386e-3, 2901.4 µs, 740.1 GFLOP/s. `tk2=64` is
forced; K=64 admits no other L2 tile. The claim is **true and now has a checkpoint behind it**.

### 3. "828 legal configurations" is unsourced and not reproducible. Treat it as a placeholder.

Enumerating from the harness's own asserts (`bf16_in_bf16_out/run.py:63-66` plus the cast-launch
divisibility at `:734`) over `sweep_families.py`'s knob lists, without its preference truncation,
gives **1584** for `attn_output` (fused-cast 1296 / drain 144 / direct 144) and **660** for
`attn_scores`. No sub-product of the grid equals 828. Do not cite it again.

### 4. ~~`fused`'s "backend settings conflict" is false. `runtime_loop_tiling_sizes` is **inert**.~~

> **`[retracted 2026-08-08]` This entry is wrong, and the claim it retracted was right.** The
> hardware run this section itself asked for has now happened, and `runtime_loop_tiling_sizes`
> is **decisive**, not inert. A replicated 2×2 factorial on `mha_out_proj` @4096×768, twelve
> heads, non-causal — each arm in its own process on an exclusive device:
>
> | tiling | ping-pong | n | result |
> |---|---|---|---|
> | `[1,1]` | OFF (shipped) | 2 | **PASS** — 0 / 3,145,728 mismatches, `mean_rel_L1` 5.3348e-02, `atol_required` 8.7061e-03 against `atol` 2.5e-02, a 2.87× margin |
> | `[1,1]` | ON | 1 | **PASS** — byte-identical statistics to the above |
> | `[2,2]` | ON | 2 | **`ERT_CMD_STATE_TIMEOUT`** |
> | `[2,2]` | OFF | 1 | **`ERT_CMD_STATE_TIMEOUT`** |
>
> So the discriminating variable is the tiling, and **`omit_pingpong` is irrelevant at this
> shape** — ping-pong ON passes, and `[2,2]` hangs whether it is set or not. The shipped preset
> re-run through the same probe harness is the control and passes, which is what makes the
> timeouts a property of the preset rather than of the probe.
>
> **The conflict `fused.py`, `mha_out_proj.py` and `block.py` document is therefore REAL:**
> FlashAttention at 4096 requires `[1,1]`, the wide GEMMs are built at `[2,2]`, and one ELF is
> one aircc invocation. Only the stated *reason* needed correcting — it is the tiling sizes and
> not `omit_pingpong` — and it is now measured instead of asserted. Do **not** make the prose
> edits this document's §Recommended order step 0 asks for; make the opposite ones.
>
> **The methodological error is worth more than the finding.** Everything measured below is
> accurate: the lowered `aie.air.mlir` really is identical between the two settings, and
> `air-opt-shim-dma-bds` really does early-exit. What does not follow is "therefore the value is
> never consumed". Identical IR *at the altitude that was diffed* is not inertness; something
> downstream of that dump consumes it. This section even wrote the caveat down — "compile-only
> refutes *a placement failure at best*, it does not refute *wrong numbers at worst*" — and then
> stated the conclusion without the caveat attached. The real outcome was a third branch neither
> disjunct covered: it neither places badly nor computes wrongly, **it hangs**.
>
> Reproduce with `python3 agents/probes/probe_backend_preset_hardware.py [attn|gemm|t11nopp|t22pp]`,
> one process per arm via `agents/scripts/devq.sh submit --class measure`.

The refuted reasoning is kept below as a record.

`pattern/fused/fused.py:63-74`, `builders/mha_out_proj.py:111-130`, `builders/block.py:90`,
`opcheck_specs.py:775-781`, `run_npu2_fused_peano.lit:12-16` and [03](03-measurement-model.md) all
state that FlashAttention needs `omit_pingpong="all"` + `runtime_loop_tiling_sizes=[1,1]` while the
4096-row GEMMs need `[2,2]`, so one ELF is impossible. Compiled both ways and diffed aircc's lowered
`air_project/aie.air.mlir` (channels renumbered, lines sorted):

| design | `[1,1]` vs `[2,2]` | lowered IR |
|---|---|---|
| `mha_out_proj` @4096 (PP off) | 280 `aie.dma_bd` / 44 `shim_dma_allocation` / 628 `aie.buffer` / 424 `aie.lock`, both | **identical** |
| `fused_tail` @1024 (PP on) | 600 / 98 / 524 / 808, both | **identical** |
| `qkv_proj` @4096 (PP on) | 304 / 68 / 236 / 376, both | **identical** |

The raw `mha` diff is 98 lines of `@channel_17`/`@channel_19` renumbering and nothing else.
Mechanism: `air-opt-shim-dma-bds` (`mlir/lib/Transform/AIRDependencyScheduleOpt.cpp:7985-8000`)
early-exits when there is no shim-level `scf.for` to tile, so the value is never consumed.

And `omit_pingpong` — the only knob that changes anything — **compiles both halves both ways** at
their canonical shapes: `mha/gemm` (the GEMM preset over FlashAttention) PASS in 105.2 s at 344 bd /
660 buf / 488 lock; `tail/attn` (the attention preset over the tail) PASS in 76.0 s at 456 / 424 /
736. Zero packet-typed channels in every build. Nothing exhausted BDs, nothing failed to place.

**The caveat that bounds this:** compile-only refutes *"a placement failure at best"*. It does not
refute *"wrong numbers at worst"*. The structurally new configuration is ping-pong ON over the
FlashAttention half (+64 BDs, +32 L1 buffers versus its shipped form) and it needs one hardware run
against the `mha_out_proj` oracle before the correction lands. The ELF is built and waiting at
`/home/cj/.claude/jobs/e75c34c9/tmp/spikeC/run_mha_gemm_4096/probe_mha_gemm.elf`.

### 5. A device softmax **kernel already exists in this tree**, and this port already builds it.

`[2026-08-08, established while writing this document — not by a spike]` Spike B reported that no
standalone device softmax exists and that a corrected `runlist` therefore needs a softmax operator
built from nothing. That is right about the *builder* and wrong about the *kernel*:

- `programming_examples/softmax/softmax.cc:316` defines `softmax_bf16` (single-shot, LUT-based exp),
  driven by an AIR design in `softmax.py` that is **gated on this hardware** —
  `run_npu2_makefile_peano.lit` REQUIRES `ryzen_ai_npu2, peano` and runs `--n 2048 --herd-n 4`.
- Behind `-DSOFTMAX_STREAMING`, `softmax.cc:241-311` exports the whole online-softmax recurrence:
  `init_softmax_scale_buffer`, `partial_softmax_rows_bf16`, `normalize_softmax_rows_bf16`,
  `copy_softmax_scale_bf16`. `partial_softmax_rows_bf16` takes `row_width` and `num_rows` as runtime
  `int32` arguments, so it is shape-generic; the degenerate one-block call sequence
  (init → partial → normalize) **is** a plain row-wise softmax.
- **This port already knows how to build it.**
  `programming_examples/transformer_layer/compile_kernels.py:172-183` lists `softmax_streaming.o`
  with all five symbols, via `ek.compile_softmax_streaming()`
  (`llms/shared/infra/external_kernels.py:471-484`). `README.md:135` documents the opt-in flag.

What is genuinely missing is a `builders/softmax.py`, an `opcheck_specs.py` row with a measured
`atol`, and a fault-injection control — plus two real gaps: `softmax_bf16` **does not subtract a row
max** (safe for the `randn` gate input, a numerical risk on real scores) and hardcodes
`zero_vectorized<bfloat16, 1, 256, 16>`, so wide rows must go through the streaming family; and the
streaming entry point hardcodes `SM_LOG2E` as its scale, so attention's `1/sqrt(head_dim)` has to be
applied upstream (`elementwise_mul`, or folded into Q) or the entry point extended. No builder in
`programming_examples/transformer_layer/builders/` references `softmax_streaming.o` today, and the
object is not currently built in the tree.

> **`[2026-08-09]` BUILT, and two of the gaps above are misstated.** `builders/softmax.py` exists,
> three `opcheck_specs.py` rows are measured and gated, and `run_npu2_softmax_peano.lit` passes on
> hardware with its negative control. Corrections to the paragraph above:
>
> - **The row-max gap belongs to the single-shot kernel, not the streaming family.**
>   `partial_softmax_alias_bf16` computes a running row max and rebases the exponentials on it, so
>   the streaming path is the numerically *safe* one. The paragraph reads as though the whole file
>   lacks it.
> - **`SM_LOG2E` is not "the scale" in the sense implied.** It is the base conversion for an
>   `exp2`-based `exp` — the kernel computes `exp2(x·log2(e) − m)` — so for a plain softmax it is
>   exactly the right value and there is nothing to extend. It matters only for attention, and the
>   plumbing is already one level down: `partial_softmax_alias_bf16` takes `scale` as an argument,
>   so folding in `1/sqrt(head_dim)` is `scale = SM_LOG2E / sqrt(head_dim)` plus a parameter on the
>   wrapper, not a kernel rewrite.
>
> Two things the build cost that were not on anyone's list, both caught by measurement rather than
> by reading:
>
> - **One role per L1 buffer.** The first version normalized back into the DMA-destination buffer,
>   which was dead by then and legal as far as the kernel's `__restrict` is concerned. The design
>   returned **the input unchanged** at all three shapes. A buffer that is both a DMA destination
>   and a kernel output does not read back what the kernel wrote.
> - **The standard injection target does not discriminate for a normalization.** `(rows-1, 0)`, what
>   every other row-wise operator here uses, left the injected run **passing** at two of three
>   shapes — `+2.0` on a low-probability element moves the tensor 1.06e-3 / 7.43e-3 against an
>   `atol` of 7.5e-3. At 512×512 the injected run's `abs_err_max` *equalled the clean run's*, so no
>   `atol` admitting the clean run could reject it. The target is now the last row's argmax, chosen
>   per shape by measurement, which clears `atol` by 12–18×.

**Consequence: the corrected-`runlist` softmax phase is smaller than Spike B priced it.** It is
builder + gate work over an existing, hardware-gated kernel, not new kernel development.

### 6. `fused` **cannot build its own SPECS shape today**. `make check-fused` is presumed broken.

`opcheck_specs.py:782-790` pins the fused row at `seq_len` 4096. `fused.py:334-337` calls
`build_norm_tail_module(seq_len, emb, plane_major=True)`, and `builders/norm_tail.py:262-273` raises
before aircc is ever reached:

```
ValueError: plane_major packing needs a plane stride of rows*cols (4096*768 = 3145728),
over the shim aie.dma_bd cap of 1048576
```

Reproduced twice, 1.5 s each. `fused.py:37` already says the mode is "BOUNDED TO 256..1024"; the
SPECS row was never moved with it. `compile_fused_artifacts` (`fused.py:492-495`) rebuilds every
module on every call even with `run_only=True`, so the cached
`fused_tail_4096x768x3072.elf` (dated Aug 5, pre-J7a) cannot rescue it. ~~**Unverified:** nobody ran
`make check-fused` to confirm the gate is red.~~

> **`[2026-08-08]` CONFIRMED RED, AND NOW FIXED.** The gate was run: it raised the `ValueError`
> above and never reached the device. The SPECS row is moved to **1024**, the top of the mode's own
> supported range, and `run_npu2_fused_peano.lit` **passes** on hardware — both recipes, 10/10
> stages clean, `mean_rel_L1` 1.756e-2, `atol_required` 5.813e-2 against the 1e-1 ceiling (1.72×).
>
> Two things moved with the shape that were not predicted here, both registry facts rather than
> gate edits: at 1024 the FFN down-projection's fastest high-precision row is **`drain`**, where at
> 4096 it was `fused-cast`, and `drain` exposes no f32 C scratch — so the stitched tail takes **11**
> whole-tensor args instead of 16. Dispatch totals at 1024 are `submissions 1 entries 3 air 11
> herd 23 sync 19 bytes 56626176`.
>
> **A comparison was suspended rather than restated.** The gate used to read its `sync 19` against
> `coarse`'s 402 and call that the mode's gating clause. `coarse` is still a 4096 row and `fused` is
> now 1024, so that ranking spans two sequence lengths and is withdrawn until both are measured at
> one. (`sync` happens to be unchanged at 19; `air` fell 16 → 11 and bytes 184,025,088 → 56,626,176.)

---

## Lane verdicts

| Lane | Question | Verdict | Cost paid |
|---|---|---|---|
| **B — `runlist`** | Can the attention interior run on the device? | **feasible-with-changes**, and the scope *shrinks* | 33 device jobs, 976 s of lock |
| **A — `offload`** | Can matmul loop bounds come from a runtime parameter, so one xclbin + one instruction stream serves several GEMM shapes? | **blocked as posed**; already absorbed into [03](03-measurement-model.md) by `e58a2170` | 0 device jobs |
| **C — `fused`** | One xclbin for the whole layer, and no DRAM between operators? | **split**: the documented blocker is **real** after all (`[corrected 2026-08-08]` — see §4's retraction; the spike's compile-only refutation of it did not survive hardware), a second one is real too (`air-fuse-channels`), and half 2 is capacity-bounded | compile-only + 6 device jobs |

### B — attention on device

32 configurations measured on hardware: **18 passed, 10 `failed_precision`, 4 `failed_build`**
(`attn_output` 24 tried / 14 passed; `attn_scores` 8 tried / 4 passed). Checkpoints are
signature-keyed JSON under `/home/cj/.claude/jobs/e75c34c9/tmp/spikeB/results/`, resumable — a
killed run loses one candidate. The verification mechanism is reused verbatim from
`sweep/sweep_measure.py`: build, then `XRTRunner._check_outputs` `np.isclose` over the **full**
output at **0 % allowed mismatch**, plus timing.

**The ladder — `attn_output` passes at every rung, both single-launch methods, 8/8:**

| seq | `attn_output` drain | `attn_output` direct | `attn_scores` drain |
|---|---|---|---|
| 512 | 72.7 µs | 84.8 µs | 103.2 µs |
| 1024 | 153.6 | 117.4 | 225.5 |
| 2048 | 351.9 | 275.3 | 738.1 |
| 4096 | 1179.3 | 915.8 | 2901.4 |

**The two failure clusters, fully characterised — one of them almost certainly *is* J2's timeout:**

- **`herd_n=1`** (`tn=32` or `64`) at N=64: the design runs, returns essentially the host-written
  buffer (`mean_rel_L1` ≈ 1.00, 231,517 / 262,144 wrong) at a flat 6,144,000 µs per iteration. That
  is the hang signature, and it matches `sweep_families.py`'s own recorded footgun that `herd_n=1`
  "placed but FAILED AT RUNTIME" at N=896. **If the one configuration doc 16 tried used `herd_n=1`,
  that is the whole of J2's blocker.**
- **`tile_n=8`**: `drain` and `fused-cast` fail to BUILD at `mm_aie2p.cc:161`,
  `static_assert(n % (2 * t) == 0)`; `direct` builds and returns garbage (262,144 / 262,144 wrong,
  298–597 ms). The microkernel narrows the legal space below what `run.py`'s asserts admit — a real
  correction to any "legal configuration" count.

**`attn_scores direct` fails the gate but is not a hardware failure.** `mean_rel_L1` 9.46e-3 with
3,528 of 16.7 M elements outside tolerance: it misses the low tier's deliberately-unscaled
`atol=4e-3` while needing 6.72e-3, and passes comfortably under the high tier's K-scaled rule
(`1.5e-3·sqrt(8192/64)` = 1.70e-2). At K=64 there is exactly one L2 tile, so `direct`'s per-L2-tile
truncation *is* a single epilogue cast — high-precision by construction at this shape, and the tier
model cannot see that.

### A — runtime-parameterized loop bounds

Three runtime-parameter mechanisms exist in the stack and **none reaches a loop bound**:

1. **RTP is half-built, and the built half is the wrong half.** AIR already makes the *core* side
   runtime-capable — `AIRToAIEPass.cpp:329-346` allocates `__air_herd_rtp_<x>_<y>` and `:451-508`
   rewrites herd scalar operands into `memref.load` inside the core body. The *host* side is
   compile-time only: `AIRRtToNpuPass.cpp:892-901` emits `NpuWriteRTPOp` **only** when
   `dyn_cast_if_present<arith::ConstantOp>` succeeds (non-constant operands are silently skipped
   while `rtp_slot++` still advances), and `AIEX.td:768-782` declares `I32Attr:$value`.
2. **`npu.address_patch`** (`AIEX.td:917-931`) patches DDR **addresses** only. This is exactly why
   buffers can be rebound per run but sizes cannot.
3. **The control scratchpad** (`AIEX.td:953-1071`) is the only true host→firmware runtime parameter
   in the stack, and mlir-air uses **none** of it — zero occurrences of "scratchpad" across
   `mlir/`, `python/`, `runtime_lib/`. It is additive only ("It cannot set an absolute value"),
   always writes 8 contiguous bytes, forces `*addr = result & 0xFFFFFFFC`, and caps at 32 StateTable
   entries. It is also unreachable from Python: `get_ctrl_scratchpad_bo` exists in
   `libxrt_coreutil.so` (XRT 2.21.0) and `xrt_kernel.h:660`, but the string appears in **neither**
   `pyxrt*.so`, and `pyxrt.cpp:196-217` binds only `add_callback/set_arg/start/state/wait/wait2`.
   Even if bound, it cannot cleanly patch a BD size: on shim, `d0_size` shares DMA_BDX_3 with
   `d0_stride` (`AIEDmaToNpu.cpp:562-564`), so an additive 64-bit delta carries into the neighbouring
   stride field. Only `buffer_length` (DMA_BDX_0, full word) is cleanly reachable, additively, in
   multiples of 4.

**The structural killer sits below all of that.** `AIETranslateNpuToBinary`
(`mlir-aie/lib/Targets/AIETargetNPU.cpp:242-318`) walks the runtime sequence with a `TypeSwitch` over
exactly nine ops — sync, write32, blockwrite, maskwrite32, load_pdi, address_patch, preempt,
create_scratchpad, update_from_scratchpad — appending to a flat `std::vector<uint32_t>`. **There is
no branch, jump, call or loop opcode**, and `aiex.run` (`AIEX.td:567-586`) is explicitly "by
inlining its instructions at the call site". Loops are fully unrolled upstream
(`AIRRtToNpuPass.cpp:1850`, `:1977` → `loopUnrollFull`, hard `signalPassFailure()` on failure), and
real output confirms it: `air_project/npu.air.mlir:588-748` holds 25 `dma_configure_task_for` and
**zero** `scf.for`. So the instruction stream's *length* is a function of the shape. Patching field
values cannot fix that; adding a branch opcode is firmware/format work in a repo nobody here owns.

Corroboration from `offload_cache/`, offered as corroboration only: `.ctrltext` totals of 227,440 /
346,736 / 196,112 bytes for the three offload GEMM shapes. **This is confounded** —
`resolve_gemm_spec` gives the three shapes three different recipes (`drain/tile_n96`,
`drain/tile_n128`, `fused-cast/tile_m64/tile_n96`), so size differs for recipe reasons too. The
unroll + no-branch argument is what is load-bearing. (Also: the three `.insts.bin` in that directory
are byte-identical 2,288-byte **stale leftovers**, md5 `10855cd4…`, and are not read on the ELF path
— `xrt.py:567` sets `self.bo_instr = None`. Do not cite them.)

**[03](03-measurement-model.md) has already absorbed this** (`e58a2170`): `offload` now matches
iron — one xclbin, N instruction streams — and runtime bounds are recorded as a deferred increment.
This lane's finding is therefore *already in the taxonomy*; what remains is the code.

### C — one xclbin, and no DRAM between operators

**The real one-xclbin blocker is `air-fuse-channels`, and it is a compiler pass.** The whole-layer
stitch itself is fine — `qkv_proj` + `mha_out_proj` + `fused_tail` as three `KernelSlice`s over one
signature (with `mha`'s `y` and the tail's `packed1` as two typed args over one BO, the
`_TAIL_BUFFER_ALIAS` device `fused.py` already uses) parses at 20 func args / 1015 lines / 11
`air.launch` @1024. aircc then never leaves the AIR pass pipeline:

| module | channels | `air-opt --air-fuse-channels` |
|---|---|---|
| `mha_out_proj` @256 (`pass_017`) | 45 | 64 s, rc=0 |
| `fused_tail` @256 (`pass_017`) | 45 | 53 s, rc=0 |
| **`mha_tail` @256 (stitched)** | **90** | **rc=124 at 600 s, and again at 1200 s** |

Controls: each module compiles **end to end** through the same aircc at the same shape —
`mha_out_proj` @256 in 97 s (60 debug-IR passes, real ELF), `fused_tail` @256 in 81 s. Note that
channel fusion is already ~2/3 of a single-module compile. **2× the channels → at least 18× the pass
time and still unfinished.** Preset-independent: the stall reproduces with no tiling flags at all,
and full-layer @1024 was killed at 1355 s with `aie.air.mlir` never written.

Source: `AIRDependencyScheduleOpt.cpp:5064` `AIRFuseChannels`. The pair loop at `:5139` is
for-i/for-j>i over every channel with `checkIfTemporalMergeable` plus IR mutation inside it, and
`renameSymbols` at `:5083` is a **second** O(N²) doing a whole-module
`SymbolTable::replaceAllSymbolUses` per iteration. There is no fixed-point loop, so **it terminates
— it is slow, not hung.** The `mergeChannels` O(N²) is skipped by default (aircc passes no
aggressive mode, `:5301`).

**Half 2 — "no DRAM between operators" — is capacity-bounded, not engineering-bounded.**
"Device-resident" in `fused.py` means *no host sync*; all 17 func args of the layer at 1024 are
L3/DDR memrefs and every inter-operator boundary is a DDR buffer. Logical tensor traffic at
1024×768×3072 bf16 (each tensor counted once per read and once per write; real DMA traffic is
**higher**, since tiled GEMMs re-read A per N-tile and B per M-tile):

| item | MiB | share |
|---|---|---|
| `ffn_up` + `ffn_gelu` (two S×F tensors, each written then read; GeLU is its own launch through DDR) | 24.0 | 29 % |
| `qkv_f32` (the fused-cast S×3E×4B scratch, written by the GEMM, read back by 3 split-cast launches) | 18.0 | 21 % |
| `q`, `k`, `v` | 9.0 | 11 % |
| `hidden` + its mirror (written twice, read twice) | 6.0 | 7 % |
| `attn_context` | 3.0 | 4 % |
| `attn_out` (`packed1` plane 0) | 3.0 | 4 % |
| `x` re-read as residual | 1.5 | 2 % |
| **total** | **84.0** (49.5 read + 34.5 write) | |
| irreducible (weights 13.5 + x in 1.5 + output out 1.5) | 16.5 | 20 % |
| **intermediate activation traffic** | **67.5** | **80 %** |

J7a already removed ~15 MiB of a ~99 MiB baseline (four S×E norm args at 3 MiB each, plus both
gammas going from host-materialised `[S,E]` broadcasts to `[E]` weights) — real, and invisible to
`bytes_transferred` exactly as documented. **J7b buys `fused` nothing today**: `fused.py:149,292`
imports `build_ffn_module`, never `build_ffn_accum_module`, and the fused down-projection already
accumulates in L2 (`tile_k_l2=512`), so the per-K-step C round trip the ring removes is not on this
path at all.

**The hard ceiling.** NPU2 on-chip capacity is 32 cores × 64 KiB L1 (`getLocalMemorySize` = 0x10000)
= 2 MiB plus 8 memtiles × 512 KiB (`getMemTileSize` = 0x80000) = 4 MiB, **6 MiB total, and not a flat
address space**. At seq 1024 one S×F intermediate is 6 MiB — the whole chip. At 4096 it is 24 MiB.
So "DRAM traffic only for the layer input and output" is **arithmetically out of reach** for the
whole layer at the sequence lengths this study runs. It becomes reachable only under whole-layer
sequence blocking (~128–256 rows resident), and attention blocks that: a Q band still needs all of K
and V (3 MiB at 1024, 12 MiB at 4096), so K/V stream from DRAM per band — trading intermediate
traffic for repeated KV traffic, probably a net loss at these lengths.

---

## Sizing: worktree-sized versus its own port-loop phase

| Work item | Home | Size | Why |
|---|---|---|---|
| Rewrite the three lit gates to demand the corrected behaviour | `programming_examples/transformer_layer/*.lit` | **worktree** | Must land *before* implementation so the work has something honest to fail against |
| **Corrected `offload`**: two attention GEMMs on device, only softmax / both LayerNorms / GeLU on host | `pattern/offload/offload.py` | **worktree** | Both matmuls proven at every rung with measured tiles. No registry write, no compiler work, no new operator — inject via the `gemm_spec_fn` escape hatch every builder already ships |
| **Fix the `fused` SPECS row** (4096 → 1024, or row-interleaved packing above 1365 rows) | `opcheck_specs.py:782-790` | **worktree**, ~1 h | The row cannot build today |
| **Correct the backend-settings claims** in `fused.py:63-74`, `mha_out_proj.py:111-130`, `block.py:90` | those three files | **worktree**, ~half a day | Gated on one hardware run of `probe_mha_gemm.elf` |
| **Correct the "no K=64/N=64 row exists" language** in 11 places | docs + `pattern/` | **worktree** | It is a *catalogue* constraint, not a hardware one |
| **Device softmax operator** (builder + opcheck row + measured atol + fault control) over the existing `softmax_streaming.o` | `builders/softmax.py` + `opcheck_specs.py` | **own phase** | New gated operator; needs a scale decision and a row-max decision (see §5) |
| **Corrected `runlist`**: every operator on device | `pattern/runlist/` | **own phase** | Depends on the softmax operator; changes the mode's entry count and every pinned total |
| **N instruction streams under one xclbin** for `offload` | ~~`python/air/backend/xrt.py` + `programming_examples/`~~ **`aircc --xclbin-input` and below** | **own phase, compiler** | ~~Already plumbed (`xclbin_input`, `xrt.py:80`, `:372-374`) but never dispatched; touches the shared `dispatch.py` path~~ **`[2026-08-09]` The plumbing is there and it DOES NOT WORK.** Two GEMMs chained into one xclbin both appear in `get_kernels()`; only the one compiled FIRST executes correctly, and reversing the order reverses which breaks. The second times out or returns garbage at `mean_rel_L1` 1.41 with no error, by shape. Both are correct standalone. So this is not the dispatch-layer change this row sized — `plan_submissions` and `ensure_loaded` are the easy part and have nothing to run on. `agents/probes/probe_one_xclbin_n_streams.py` |
| **Scope `air-fuse-channels`** to same-launch channel pairs | `mlir/lib/Transform/AIRDependencyScheduleOpt.cpp` | **own phase, compiler** | Only needed if one whole-layer xclbin actually matters |
| **GeLU as a GEMM epilogue** (−12 MiB of 84, −14 %) | `llms/shared/builders/gemm_builder.py` + an `mm_*.o` variant | **own phase**, ~1–2 days | Same shape of change as the existing fused-cast epilogue; the only cheap DRAM win left |
| **Corrected `coarse`** | `pattern/coarse/` | **own phase** | Defined as a mix of `runlist` and `fused`; both must be correct first |

**Runtime-parameterized loop bounds: do not open a worktree.** It is not "a substantial new `mlir/`
feature" — it needs `mlir-air` (unroll removal, `DmaToNpuPattern`, RTP host side, plus the 27
`getStaticScfForTripCountAsInt` call sites across 6 files), `mlir-aie` (`NpuWriteBdOp`,
`AIEDmaToNpu.cpp`, `AIETargetNPU.cpp`), a rebuilt XRT for the pyxrt binding, **and** a txn format
that does not exist.

**Deleting `qkv_f32` (−18 MiB, −21 %): do not attempt.** `builders/qkv_proj.py:297-308` raises for
any scratch-free spec precisely because the three-way Q/K/V split rides the scratch-bearing method's
separate cast launch. The alternative — one GEMM writing three column bands at offsets — is the
strided-producer wall at `builders/norm_tail.py:88-104`. That is an `mlir/` change (H7).

**Any further attention tile search: not worth doing.** The space is 1584 configurations, the answer
was in the first one, all three methods work, and both failure clusters are fully characterised.

---

## Recommended order of execution

The sequencing constraints are: B unblocks A (offload's attention matmuls are the same device
attention problem, and the author's linearity rule puts them on the NPU); B and C are coupled
(`builders/mha_attention.py` is edited by B, and `builders/mha_out_proj.py:94,153` imports its
config, IR builder and compile functions); A crosses the shared dispatch path
(`llms/shared/infra/dispatch.py:685`) that `fused` also uses; D is last.

**0. Corrections and gate rewrites, before any implementation.** All worktree-sized, all
independent, all cheap, and one of them (the `fused` SPECS row) may already be a red gate.
- The three lit rewrites: `run_npu2_offload_peano.lit:42,44` and `run_npu2_runlist_peano.lit:47,49`
  currently *require* `attention host torch fp32`, and `run_npu2_fused_peano.lit:46,85` pins
  `3 entries over 3 ELFs` and `sync 19`. Every one of those CHECK lines certifies the obsolete
  implementation and will pass the wrong build.
- Doc 16:160 (the J2 row), doc 03:149-155 (the `[2026-08-05]` blockquote), doc 08c:23-31, doc 09:66,
  this directory's `README.md:267`, `programming_examples/transformer_layer/README.md:484`,
  `pattern/blocked_attention.py:22-28`, `pattern/runlist/README.md:70-72`,
  `pattern/runlist/runlist.py:154`, `pattern/offload/README.md:24`, `pattern/offload/offload.py:25`.
  All eleven repeat the K=64/N=64 claim in a way that reads as a hardware constraint. (Spike B's
  own list gave `docs/README.md:267` and `03:43/113`; both pointers are wrong — the plan
  directory's README carries :267, and doc 03 was renumbered by today's `df7153b2`/`e58a2170`.
  Re-grep before editing rather than trusting any line number in this document.)
- `opcheck_specs.py:782-790` (the fused seq_len). **`[2026-08-08]` The `fused.py` /
  `mha_out_proj.py` / `block.py` backend-settings prose is NOT to be softened** — step 1 measured
  it and it is right; see §4's retraction. What those three files need is the measured basis
  substituted for the asserted one, and `omit_pingpong` dropped from the stated reason.

**1. ~~One hardware run each, to close the two cheapest unknowns.~~ DONE `[2026-08-08]`, and it
overturned §4.** Six device jobs on the `mha_out_proj` side rather than one: the GEMM preset
`ERT_CMD_STATE_TIMEOUT`s at 4096, the shipped preset passes through the same harness as a control,
and a 2×2 over (tiling × ping-pong) puts the cause on `runtime_loop_tiling_sizes` with
`omit_pingpong` irrelevant. Replicated 3/3 per tiling level. See §4.

The `fused_tail` half was **not** run and is not blocking: `fused_tail` has no standalone opcheck
row — its oracle is the `fused` mode's own per-boundary comparison — so the question is answered by
making `make check-fused` runnable (the `opcheck_specs.py` seq_len fix above) rather than by a
bespoke probe with a bespoke oracle. Note the direction that matters is already settled the other
way: the attention preset over the tail is ping-pong OFF, and ping-pong turns out not to be the
variable that decides anything here.

**2. Corrected `offload`.** It is the *cheap proof* of device attention and it is worktree-sized:
replace the `blocked_attention` call in `pattern/offload/offload.py` with two dispatched GEMMs plus
host softmax/scale/mask, injecting `drain / tk2=256 / tk1=32 / tn=16 / herd 8×4` for `attn_output`
and `drain / tk2=64 / tk1=32 / tn=128 / herd 8×4` for `attn_scores`. Six dispatches become eight
(×12 heads), and `attention_path` flips off `host_torch`. **This is B's work landing in A's mode**,
which is why it comes before the `runlist` rebuild rather than after it.

**3. The device softmax operator**, its own phase, over the existing `softmax_streaming.o`.

**4. Corrected `runlist`**, its own phase, once softmax is gated.

**5. `offload`'s N-streams-one-xclbin**, its own phase — and note it touches `dispatch.py:685`,
shared with `fused`, so it must not run concurrently with C work.

**6. `fused`'s GeLU epilogue** (the only remaining cheap DRAM win) and, only if one whole-layer
xclbin is judged to matter, the `air-fuse-channels` scoping phase.

**7. Corrected `coarse`.**

---

## Not yet answerable, and the measurement that would answer it

| Question | Why it is open | The measurement | Cost |
|---|---|---|---|
| ~~Do the off-preset ELFs produce **correct numbers**?~~ **ANSWERED `[2026-08-08]`: the GEMM preset does not RUN.** `[2,2]` `ERT_CMD_STATE_TIMEOUT`s 3/3 at 4096; `[1,1]` passes 3/3; ping-pong is irrelevant either way. Neither branch of "placement failure at best / wrong numbers at worst" was right — it hangs. §4 is retracted | — | `agents/probes/probe_backend_preset_hardware.py` | 6 device jobs, spent |
| How long does `air-fuse-channels` actually take on a stitch? | Bounded below only: >1200 s at 90 channels, seq 256. It terminates (no fixed-point loop) but the bound could be 30 min or 3 h | One unattended `air-opt /home/cj/.claude/jobs/e75c34c9/tmp/spikeC/mt256_pass017.mlir --air-fuse-channels` with no cap | CPU-only, overnight, **must not overlap a timed measurement** |
| Would a whole-layer ELF survive the rest of the pipeline? | Nothing past `air-fuse-channels` has ever run on an 11-launch module: `air-split-l2-memref`'s per-column shim cap (8 cols × 2), herd placement for 23 herds against 32 tiles, the 16-BD-per-shim-column budget | Only answerable after the pass finishes once | — |
| Is `make check-fused` red today? | Only established that `prepare_fused(seq_len=4096)` raises in `build_norm_tail_module` | Run it | 1 device job |
| Does the **12-head** dispatch work? | Only the per-head GEMM shape was measured. The layer dispatches 12 of each; a batched or 12×-looped form and its dispatch-vector consequences are unmeasured | Build and measure the looped form as part of step 2 | folded into the offload phase |
| Numerical margin on **real** attention operands? | Inputs were the sweep harness's seeded gaussians scaled by `1/sqrt(K)` (`sweep_measure.py:_inputs_and_reference`) — the right buildability gate, and the same one every registry row used, but not a real softmax probability matrix | The `offload` gate's own end-to-end comparison, once attention is on device | folded into step 2 |
| Could one tiling recipe cover all three offload GEMM shapes without a perf regression? | They resolve to three today; one stream would force one | Compile and measure all three under one pinned recipe | 3 builds + 3 measures — **only worth paying if the one-stream increment is ever taken** |
| Can `npu.update_from_scratchpad` patch a shim BD register on NPU2? | The op doc implies yes for the *address* field; the only existing test (`scratchpad_regwrite`) patches an L1 `aie.buffer` | Device experiment — but it is moot while the instruction stream has no branch, so **deprioritize** | — |
| Are the DRAM numbers right in absolute terms? | They are **logical tensor traffic**, a floor. Tiled GEMMs re-read A per N-tile and B per M-tile, and nothing in the port records device-side DDR traffic (`fused.py`'s `bytes_transferred` footgun) | The ranking is sound; the absolutes need a hardware traffic counter this platform does not expose | unresolvable here |

---

## Two operational notes for whoever schedules the follow-up

- **Wall time was dominated by queue contention, not device time.** Spike B's 976 s of device lock
  spread across a much longer wall clock because single candidates waited 800–1050 s behind lane C's
  1100–1400 s builds. Budget wall clock accordingly, and do not co-schedule a build lane with a
  measurement lane.
- **`registry_sweep.py` cannot stage either attention shape**, and that is fine.
  `sweep_families.py` derives K and N from `FAMILY_HIDDEN × ROLE_KN_MULTIPLES` with a minimum hidden
  of 512, so no `--family` puts a 64 in the K or N slot. The fix is the `gemm_spec_fn` escape hatch
  (inject the measured tiles) or a new attention role. **Nothing requires editing the registry.**
