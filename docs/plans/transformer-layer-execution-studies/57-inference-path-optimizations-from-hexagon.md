# 57 — Optimizing the `llms/` inference path: the `ggml-hexagon` mechanisms, translated, with the measurements that rank them

`[2026-08-20]` The operator asked, while away, for the inference-path optimizations that
follow from [55](55-hexagon-llama-cpp-lessons-for-xdna2.md)'s reading of llama.cpp's Hexagon
backend — mechanism to mechanism, methodology to methodology — worked through with Codex. This
document does that in the order the evidence demands: §1 measures where a decode token and a
prefill layer actually spend their time today (new runs, Turbo verified, devq 436–446); §2
names the one structural fact those numbers expose; §3 is the translation table, each row a
Hexagon mechanism → the mechanism here → the optimization → a prediction → the experiment that
gates it; §4 is the predicted budget after each step; §5 the experiments in the order to run
them; §6 what does not transfer; §7 Codex's review. Numbers marked *arithmetic* are derived
from measured ones and say so. Every job log, the probe scripts and the probe's JSON are in
`programming_examples/transformer_layer/results/hexagon-opt-20260820/` (gitignored, local).

## 1. Where the time goes today

### 1.1 Qwen3-0.6B bf16, decode — devq 436 (`make profile N_TOKENS=32`, Turbo observed before and after, commit `debf9be2`)

32 generated tokens, short prompt (context ≤ ~60). Per token, from the profiler's own buckets:

| Component | Per call | Calls / token | Per token | Bytes streamed | Effective rate |
|---|---|---|---|---|---|
| `o_gemv_ffn` (O + residual + RMSNorm + gate/up + SwiGLU + down) | 1.53 ms NPU run (BO write 0.01) | 28 | **42.8 ms** | 23.1 MB | **15 GB/s** |
| `rms_qkv_qknorm_rope_gemv` (RMSNorm + Q/K/V + QK-norm + RoPE) | 1.03 ms | 28 | **28.8 ms** | 8.4 MB | **8 GB/s** |
| `lm_head_gemv` (19 partitions × 8192 rows) | 9.68 ms | 1 | **9.7 ms** | 311 MB logical / **319 MB padded** | **33 GB/s** |
| `decode_attention_cpu` | 0.12 ms | 28 | 3.4 ms | — | — |
| embed / final norm / BO writes / reads | — | — | ~1 ms | — | — |
| **Total** (profiler: 2738.6 ms NPU + 106.9 ms CPU over 32 tokens) | | | **~89 ms ⇒ 11.2 tok/s** | **1.19 GB** | **13 GB/s overall** |

Three things the table says on its own:

- **The machine streams weights at 32 GB/s when the launch is large** (`lm_head`: 19
  launches of 16 MB, 0.51 ms each). **The per-layer decode ELFs reach 8–15 GB/s.** If the
  per-layer work ran at the `lm_head` rate it would take 28 × (31.5 MB / 32 GB/s) = **27.6 ms**
  instead of 71.6 — *arithmetic*.
- **Decode for this model is not bandwidth-bound.** 1.19 GB/token at 89 ms is 13 GB/s overall;
  the study's single-shim-port figure is 5.3 GB/s (`analytical_cost.py:49`) and eight columns
  are in use by the GEMV builders (`n_cores=8`, `_STAGE2_HERD_COLS=8`).
- **CPU attention is negligible at short context** (3.4 ms/token) — and will not stay so:
  `decode_attention_cpu` (`qwen3_0_6b_decode.py:285-316`) converts the *whole* cached slice
  `k_cache[:, :seq_len, :].astype(float32)` (and `v`) per layer per token — at context 2048
  that is 2 × 8 × 2048 × 128 × 4 B = **16.8 MB of conversion per layer, ~470 MB per token** —
  and loops over 16 heads in Python. **Measured (devq 444, Turbo, 16 tokens each):** 0.90 ms
  per layer at ~1,000 tokens of context and **1.93 ms at ~1,900** — **25 and 54 ms per token**,
  linear in context — while the NPU part stayed at ~88 ms; the token grows from 89 ms to
  ~145 ms (**6.9 tok/s**) at ~1,900 context. The wall is real and it is host-side.

The Llama-3.2-1B bf16 path (June table: 12.2 tok/s, 2.47 GB/token) implies ~30 GB/s overall
because its weight matrices are bigger per launch (emb 2048, hidden 8192) and amortize the
same fixed costs better — which is the first hint of §2.

**`[2026-08-21]` The table above is the 2026-08-20 baseline; three landed changes move it** (§5
items 3c, 5 and 5b, each a same-session before/after under recorded Turbo): `lm_head` 9.77 →
8.79 (`m_input = 8`) → **7.63 ms** (9 × 16384 + 4480), `rms_qkv` 1.03 → **0.62 ms** (8 → 4
launches), `o_gemv_ffn` 1.53 unchanged. **Clean idle-host profile of the tree at `fd7e17b8`
(devq 486, `make profile N_TOKENS=32` ×2, load 1.2): 12.96 / 13.12 tok/s — 77 ms per token**,
per layer 2.45–2.47 ms, layer loop 2187–2217 ms / 896 invocations, against 89 ms (11.2 tok/s)
on 2026-08-20. Boundaries per token: 28 × 7 + 10 = **206** (from 327), ≈ 22 ms ≈ 29 % of the new
token; the three changes moved no new bytes and added no kernel.

### 1.2 Qwen3-0.6B bf16, prefill at `seq_len = 2048` (same run)

| Kernel | NPU run | BO write | Per layer | Share | Rate |
|---|---|---|---|---|---|
| `flash_attn` (head-first, hd 128, causal) | 21.23 ms | 1.41 ms (**24 MB** host-transposed Q/K/V) | 22.7 ms | **51 %** | 34 GFLOP dense / 17 causal-effective ⇒ **0.8 TFLOPS effective** |
| `o_ffn_qwen` | 12.71 ms | 0.69 ms | 13.5 ms | 30 % | 47 GFLOP ⇒ **3.7 TFLOPS** |
| `rms_qkv_qknorm_rope` | 7.72 ms | 0.30 ms | 8.2 ms | 18 % | 17 GFLOP ⇒ 2.2 TFLOPS |
| Σ kernels | | | **44.8 ms / layer**, 1,256 ms / 28 layers | | 2048 tokens ⇒ 1,630 tok/s kernel-only |
| layer-time incl. host transposes and KV extract | | | 51.8 ms / layer (1,451 ms) | | |

And the whole prefill is spent on **2,048 padded tokens regardless of the prompt**
(`qwen3_0_6b_inference.py:645-649`): a 60-token prompt pays the full 1.45 s.

### 1.3 Llama-3.2-1B int4, decode — devq 440 (caches compiled in devq 437; Turbo observed)

The int4 driver builds a `Profiler` but never prints it, so this ran through a wrapper that
calls `report()` (the wrapper forced CPU *prefill* attention by mistake — its TTFT is not a
number; its decode path is the production one). 32 tokens after a 57-token prompt,
**66 ms/token = 15.3 tok/s** (devq 438, the plain driver: 64 ms, 15.7).

| Component | Per call | Calls / token | Per token | Bytes | Rate |
|---|---|---|---|---|---|
| `lm_head_gemv` — **still bf16** (tied embeddings, 128256 × 2048 × 2 B) | 14.85 ms | 1 | **14.9 ms (23 %)** | 525 MB logical / **537 MB padded** | **36 GB/s** |
| `o_gemv_ffn_int4` (3 launches: O+add, gate/up cascade, down+add) | 2.04 ms | 16 | **32.6 ms** | ~28 MB int4 + scales | **14 GB/s** |
| `rms_qkv_int4_rope` (6 launches) | 0.77 ms | 16 | **12.3 ms** | ~3.2 MB | **4 GB/s** |
| `decode_attention_cpu` | 0.24 ms | 16 | 3.8 ms | — | — |
| **Total** | | | **~65 ms** | **~1.0 GB** (0.5 of it the bf16 head) | 15 GB/s overall |

Read against the bf16 path's 12.2 tok/s (82 ms): int4 removed ~1.45 GB of the 2.47 GB per
token and bought 16 ms, because (i) the bf16 LM head is now the single largest item, (ii) the
int4 GEMVs stream at 4–14 GB/s — *slower per byte* than the bf16 ones, dequant and the
per-launch fixed cost dominating at these sizes — and (iii) the token still carries
16 × (6 + 3) + 8 = **152 launch boundaries**. Every one of those is a §3 row (O4, O3, O1).

### 1.4 The per-launch cost, isolated — devq 445 (`probe_pdi_cost.py`, Turbo observed)

Identical LM-head work — the same 155,648 × 1024 bf16 weights, the same input, 319 MB
streamed — built two ways with `build_lm_head_gemv_module`: **19 launches × 8192 rows** and
**38 launches × 4096 rows**. 3 warm-ups, then 20 timed calls each, interleaved:

| Variant | p50 | min | avg | max |
|---|---|---|---|---|
| 19 × 8192 | **9.834 ms** | 9.639 | 9.852 | 10.164 |
| 38 × 4096 | **11.901 ms** | 11.853 | 11.978 | 12.468 |

**Δp50 = 2.067 ms for 19 extra boundaries ⇒ 109 µs per in-ELF launch boundary**, at
32.4 GB/s for the 19-launch form (which matches the production `lm_head` line in §1.1 to 2 %).
Compile time is its own cost of launch count: 54 s for 19 launches, **592 s for 38**.

Reproduced (devq 446): p50 9.96 vs 12.05 ms, **110 µs per boundary**, 32.0 GB/s.

**What the probe does and does not isolate** `[per Codex review]`: bytes, arithmetic, kernel
source and weights are identical, but halving `n_part` also halves each launch's internal
iteration count (`launch_size = m / tile_m / herd_m`, `matvec.py:108`: 128 → 64) and moves the
broadcast-DMA repeat geometry from the 255 limit to ~127. The 2.07 ms therefore contains the
reconfiguration **plus** whatever the changed per-launch DMA schedule costs; BD-count equality
is not established. "109 µs per boundary" is the right order and the right direction — the
per-layer gaps in §1.1 say the same thing independently — but as an *isolated* boundary cost
it is ~~**unverified**~~ **`[2026-08-21]` verified at 106–108 µs with the geometry held fixed — §1.5**. The isolating experiments: 38 correct 4096-row segments run either as 38
devices or as **19 devices each performing two segments** (same descriptors and repeat
geometry, only the configuration count differs — not yet run); and, for the repeat edge,
19 × 8192 at `m_input = 4` (repeat 255) against `m_input = 8` (~127), same bytes and launch
count — **run (devq 449)**:

| Variant | p50 | Rate |
|---|---|---|
| 19 × 8192, `m_input 4` (production, repeat 255) | 9.96 ms | 32.0 GB/s |
| 19 × 8192, `m_input 8` (repeat ~127) | **9.12 ms** | 35.0 GB/s |
| 38 × 4096, `m_input 4` (repeat ~127) | 12.23 ms | 26.1 GB/s |

The shorter repeat geometry is **faster** per byte, not slower — so the 38-launch form enjoys
it too, and the confound makes the isolated boundary cost **larger** than the naive delta, not
smaller: between (12.23 − 9.96) / 19 = **120 µs** and (12.23 − 9.12) / 19 = **164 µs**
depending on which single-launch geometry the 4096-row form matches. Every conclusion drawn
from "~110 µs" below holds with that band; ~~the 19-devices-×-2-segments probe would pin it~~
**`[2026-08-21]` pinned at 106–108 µs by §1.5's geometry-fixed probe (devq 451) — the band
collapses to the naive delta, and the repeat-geometry speed-up is a separate, additive effect.**
Two by-products: **`m_input = 8` is a free ~0.85 ms/token on the production LM head** if it
verifies (§5 item 4); and the per-launch geometry is itself an O3 knob.

**A defect found on the way, not chased** (devq 446/447). The probe's correctness check
passed the 38 × 4096 form (max 2.5e-3 of output scale, bit-identical across runs) and failed
the 19 × 8192 form: **2,455–2,800 of its 155,648 outputs wrong, all in partition 0, rows 64
onward (the second 64-row tile of the first launch), and non-deterministic run to run (max
diff 3.4)** — but **only when the ELF re-executes immediately after itself**; run once after a
different ELF it is exact ("p19 after p38: bad 0"). The production Qwen LM head is this exact
ELF (`_LM_N_PART = 8192`, pinned at the BD repeat-count limit `n_part/32 − 1 = 255`) and its
top-5 gate passes because decode never runs `lm_head` back-to-back — an `o_gemv_ffn` always
precedes it. Any design that does run it back-to-back (batched logits, a per-token runlist
that places it adjacent to itself, re-scoring) will hit this. It is the same family as the
fused decoder's re-execution wall ([PREDICTION-FUSED-REEXEC](16-compiler-changes.md):
state left in the partition by one execution corrupting the next), with the repeat-count edge
as one suspect — though the pattern (non-deterministic, partition 0 only, from row 64, healed
by any intervening configuration) fits **stale repeat / buffer / configuration state** better
than a last-tile defect `[per Codex review]` — and devq 448 **rules the repeat limit out**:
19 × 8192 at `m_input = 8` (repeat ~127) is still non-deterministic back-to-back (partition 0,
14–107 bad rows at 48, 96, 288, … instead of ~2,800 from row 64), while 38 × 4096 at the same
repeat is exact. The defect follows the **launch size (128 iterations per launch)**, not the
repeat count. Production's gate cannot see it: only the token
set gates, the full-logit comparison is informational (`verify/report.py:54`,
`comparators.py:184`). The settling experiment: the same 19-launch ELF at `n_part = 4096`-sized partitions but 8192
rows via two BDs, re-executed back-to-back; and the existing two-dispatch re-execution gate
shape applied to `lm_head_gemv` — **`[2026-08-21]` done, §1.5: the gate is checked in
(`qwen3_0_6b/lm_head_reexec_gate.py`, lit `run_npu2_lm_head_reexec.lit` under `XFAIL`), the
production artifact reads 5 of 7 dispatches wrong, a 0.5 s idle does not heal it, any other
ELF does.** The timing comparison is unaffected — both forms move the
same bytes through the same kernel, the timed loop alternated the two ELFs (the correct
case), and the per-boundary figure is corroborated independently by §1.1's per-layer gaps
(`rms_qkv`: 1.03 ms measured − 0.26 ms at 32 GB/s = 0.77 ms over 8 boundaries ≈ 0.1 ms each).

### 1.5 `[2026-08-21]` The boundary cost isolated, and the defect gated — devq 450–452

**Codex's isolating design cannot be built as stated.** "19 devices each performing two
4096-row segments" assumes two `air.segment`s in one `air.launch` share a device; they do not
— `air-to-aie` creates one `aie.device` per segment (`AIRToAIEPass.cpp`,
`createAIEModulesAndOutlineCores`: `module.walk([&](air::SegmentOp s) …)` then one
`AIE::DeviceOp::create` per segment), so that form is 38 configurations again.

**The isolation run instead holds the per-launch geometry fixed and varies only the
configuration count.** `probe_boundary_cost.py` (evidence root
`results/hexagon-opt-20260821/`) keeps the 19 production launches byte-for-byte
(`build_lm_head_gemv_module`, `n_part = 8192`, `m_input = 4`, repeat 255 — the seeded
`p19x8192.elf` is the 2026-08-20 build of the same module) and adds *tiny* launches of the
same GEMV: the same 8-column herd, the same kernel object, the same BD shape, over 64 rows —
one launch iteration, 128 KB. Five ELFs, 3 warm-ups, 20 timed calls each, variants
interleaved so no ELF re-executes back-to-back (§1.4's defect pattern), every output checked
against f32 (all five exact, max 2.66e-3):

| ELF | launches | p50 | min | avg | max |
|---|---|---|---|---|---|
| `p19x8192` (production) | 19 | **9.829 ms** | 9.620 | 9.892 | 10.301 |
| `p19x8192_t19` (19 production + 19 tiny, interleaved; +0.76 % bytes) | 38 | **11.882 ms** | 11.730 | 11.946 | 12.301 |
| `t1` | 1 | 0.251 ms | 0.193 | 0.284 | 0.475 |
| `t19` | 19 | 2.167 ms | 2.127 | 2.202 | 2.373 |
| `t38` | 38 | 4.184 ms | 4.144 | 4.213 | 4.411 |

Three estimates of the per-boundary cost, from two independent forms:

- **additive**: (11.882 − 9.829) / 19 = **108.1 µs** — 19 extra configurations *between*
  the production launches, whose own descriptors did not change;
- **tiny-only slopes**: (t19 − t1) / 18 = **106.4 µs**, (t38 − t19) / 19 = **106.1 µs**;
  least-squares over N = 1, 19, 38: **106.3 µs per launch, 146 µs intercept** (the fixed
  per-`xrt.run` cost: submission, completion wait, the 2 KB input and 128 B output syncs).

They agree to 2 %, and the additive form agreeing with the tiny-only form says the cost does
**not depend on the neighbouring device** — it is a per-configuration constant. Each figure
still contains the tiny launch's own work (128 KB at the streaming rate is ~4 µs, plus one
kernel call and one L1→L3 return), so **~100 µs is reconfiguration proper** and 106–108 µs
is the number to charge per `air.launch` boundary in a multi-launch ELF. Doc 57's "~110 µs"
stands; the 120–164 µs band of §1.4 was the repeat-geometry effect stacked on top of it, and
that effect is a *separate*, additive knob (O3), not an error bar on the boundary. Two
by-products: the production LM head spends 19 × 107 = **2.0 ms of its 9.83 ms in
boundaries**, so its 319 MB actually stream at **40.8 GB/s** between them (the 32.4 GB/s
figure is the boundary-diluted rate); and `t1 = 251 µs` is the floor for *any* decode ELF
dispatched on its own — 57 of them per token is 14 ms before a byte moves, which is O2's
ceiling (§5).

**The re-execution defect, on the production artifact.** `lm_head_reexec_probe.py` loaded
`build_peano/decode_kernel_cache/lm_head_gemv.elf` itself (8,605,040 bytes, built
2026-08-20 03:32) and dispatched it seven times (devq 452):

| dispatch | pattern | verdict | bad rows (partition 0 only) |
|---|---|---|---|
| d1 | first after load | **clean** (max_rel 2.50e-3) | — |
| d2 | immediately after d1 | WRONG, max_rel 0.93 | 2,455 from row 64 |
| d3 | immediately after d2 | WRONG, max_rel 0.91, **≠ d2** (max diff 2.88) | 741 from rows 24, 32, 40, 64… |
| d4 | after a **0.5 s host idle**, nothing in between | WRONG, max_rel 0.92 | 2,194 from rows 24, 32, 64… |
| d5 | a **different input**, back-to-back | WRONG, max_rel 1.15 | 2,063 |
| d6 | after a **different ELF** (a 1-launch GEMV) | **clean**, bit-identical to d1 | — |
| d7 | immediately after d6 | WRONG, max_rel 0.93 | 2,084 from row 64 |

What this adds to §1.4: the idle does not heal it, so it is **stale state left in the
partition or context, not a race with in-flight work**; the input does not matter; the
rows are tile boundaries (multiples of 8) from row 24 on, not "64 onward" only; and the
healing is exactly one intervening configuration of anything else. The gate is checked in as
`qwen3_0_6b/lm_head_reexec_gate.py` + `make check-lm-head-reexec` +
`run_npu2_lm_head_reexec.lit` — it compiles the production builder's module fresh (so a fix
anywhere from builder to device is what flips it), needs no HF weights, and carries
`XFAIL: *` while the defect is open: a fix turns it into an XPASS, which is the instruction
to drop that line. Through `make check-lm-head-reexec` from a scratch cwd (the lit's
invocation) it reads the same 2 / 7 (devq 458).

**`[2026-08-21, later]` More data points, and no single rule yet.** The same-ELF re-execution
family, every observation under recorded Turbo, each ELF built from the same `matvec.py`
GEMV and the same stitching:

| ELF (launch 0 first) | dispatch 1 | back-to-back dispatches | devq |
|---|---|---|---|
| `19 × GEMV(8192, m_input 4)` — the 08-20 production head, 128 iterations/launch | clean | **wrong values**, non-deterministic, partition 0 | 446–448, 452, 458 |
| `19 × GEMV(8192, m_input 8)` | clean | wrong (fewer rows) | 448 |
| `38 × GEMV(4096, m_input 4)`, 64 iterations | clean | clean | 446–447 |
| `9 × GEMV(16384, m_input 8) + GEMV(4480)` — **the production head since 03:22 today**, 256 iterations | clean | **clean ×3** | 482 |
| `RMSNorm(1024) → 9 × GEMV(16384, m_input 8) + GEMV(4480)` | **HANG** (`ERT_CMD_STATE_TIMEOUT`) | — | 482 |
| `GEMV(4096, m_input 4) → QK-norm → RoPE` — the 3-launch QKV form | clean | **HANG** at dispatch 2, every time; clean when alternated with another ELF | 475, 477, 480, 481, 483 |
| `RMSNorm → GEMV(4096, m_input 4) → QK-norm → RoPE` — the 4-launch production stage | clean | clean ×3 (and 28 × per token in production, never adjacent) | 461–463, 480 |

What survives: the defect is **not** "a GEMV at launch 0" (row 4 refutes it), **not** the repeat
count alone (rows 1 and 4 are both at repeat 255), and **not** a property of the 8192-row
launch only (row 6 hangs at 4096). What the rows do share is that a `matvec.py` GEMV launch's
partition state after one execution is sometimes not what the next configuration of the
same partition expects — and which configurations are fragile is a function of the launch
geometry that this table has five points on and no model of. Two things are settled:
**production today (row 4 head, row 7 stage) re-executes clean**, and the 3-launch QKV form
(row 6) is **not shipped** — it bought only 0.027 ms/layer (0.7 ms/token; the first launch's
boundary is evidently cheaper than a mid-ELF one, and 29 µs of it moved to the host) while
being the one form that hangs. A side finding from the same probe: the host RMSNorm is
*more accurate* than the device kernel (vs f32: `normed` 3.6e-3 against 4.1e-2 — the device
truncates `rstd` to bf16 before the multiply), which is a kernel improvement to make
independently of launch counts. ~~The systematic study … is the next step for whoever takes
the defect.~~

**`[2026-08-22]` THE MECHANISM, NAMED: LOAD_PDI parity** (queue item 10; evidence
`results/reexec-matrix-20260822/`, devq 527–538). On the full-ELF path every `aiex.configure`
(and every `npu.load_pdi` reset) becomes one `LOAD_PDI`, which aiecc's `aie-expand-load-pdi`
turns into *load an EMPTY PDI — the firmware's partition reset — then the device's configuration
as inline writes*, alternating **two** empty PDIs by the load's *position* in the stream
(`AIEExpandLoadPdi.cpp:66`, `"empty_" + (index % 2)`, "to avoid PDI address caching"). The NPU2
firmware **skips a `LOAD_PDI` whose id equals the last one loaded** (aie2ps ISA, `LOAD_PDI`:
"consecutive loading of same pdi results in following loading skipped by the uC") and remembers
that id across dispatches. The alternation restarts at position 0 every dispatch, so a dispatch
issuing an **odd** number of loads ends on the empty PDI it starts with, and the next dispatch of
the same ELF in the same context begins with a load the firmware skips: launch 0 runs with no
partition reset on the previous dispatch's final DMA-channel / lock state. (Which state:
*inferred*, not probed — the inline configuration re-inits locks and BDs but never resets a
channel, so a core or memtile channel left mid-BD with a pre-acquired credit is the candidate;
the measured facts are the parity rule and the symptoms below.) A different ELF in between loads a different id (heals); idle
and input cannot matter; partition 0 because launch 0 is the unreset one. The symptom then
depends on launch 0's device: a multi-iteration GEMV has its weight tile overwritten under the
core (wrong rows, non-deterministic), the QKV / RMS forms hang with every output written, a
one-iteration launch is benign — so an odd count is a *vulnerability*, not a guaranteed
failure. The seven rows, re-read: loads 19, 19, 38, 10, 11, 3, 4 — **odd ⇔ wrong/hang, 7/7**; row 5's "hang at dispatch 1" was its *second* dispatch (the probe's
heal call was its first). Discriminators on the device (devq 528/529): the same 8192-row launch
×20 is clean ×5 and ×1 is wrong at dispatch 2; ×37 of the "benign" 4096 geometry is wrong.
The reset rule this tree already had (`deviceHasRepeatCountDMAs` / cascade, [16 §13](16-compiler-changes.md))
never fired here: the GEMV's repeat is the *shim* task's, which that rule does not inspect, and
rows 3/4/7 were clean without any reset — by parity.

**The fix** ([16 §18](16-compiler-changes.md)): `AIRRtToNpuPass` counts the `LOAD_PDI`s the
emitted stream will carry (merged head loads once, loop trips multiplied) and, when odd, appends
one load of a tile-less `@air_dispatch_end_reset` device at the dispatch's end — one trailing
empty-PDI load, zero writes, **26–32 µs per dispatch on odd ELFs only** (devq 538); lit
`mlir/test/Conversion/AIRRtToNpu/load_pdi_parity_pad.mlir` (seven cases), verified failing on
the pre-fix compiler rebuilt from 33d8967a (devq 550). With the first version every row and discriminator re-executes **clean ×5** (devq 533/534, 14 ELFs),
`check-lm-head-reexec` 7/7 (devq 535), `qwen3_0_6b` verify PASS with every kernel recompiled
(devq 536), the transformer-layer suite 40/1/0 (devq 537), `check-air-mlir` 506/7/7; the
complete fix (review round, 16 §18) re-gated under Turbo (devq 551): compile + verify PASS, reexec 7/7, tl suite 40/1/0. The 19 ×
8192 head is no longer defective; the 10-launch head's "clean" of 08-21 was parity. Left open:
whether the firmware keys the skip on id alone or (id, address) per context — only "not id alone
across contexts" is shown; upstream could make `expand-load-pdi`'s alternation dispatch-aware
instead of this mlir-air-side pad.

## 2. The structural fact: every `air.launch` boundary is a partition reconfiguration

[29](25-mode-rebuilds-and-results.md) records the multi-launch mechanism: a module with N
`air.launch` ops lowers to **N `aie.device` ops plus a `main` device** whose runtime sequence
issues a configure/run pair per launch (`mlir/lib/Conversion/AIRRtToNpuPass.cpp:1443`). On the
ELF path the per-launch images travel as `.pdi.N` / `.ctrltext.N` sections and the **ELF
loader resolves the reconfiguration** — a raw in-stream `load_pdi` faults NPU2 firmware
([29 §The hardware verdict](25-mode-rebuilds-and-results.md)) — so the precise statement is that **each
launch selects and configures a distinct device image** `[per Codex review]`. Either way the
array has no resident program: a "launch" costs a reconfiguration, not a descriptor.

Count them per decode token for Qwen3-0.6B: `rms_qkv_qknorm_rope_gemv` is 8 launches
(`rms_qkv_qknorm_rope_multi.py:670`: RMSNorm, Q, K, V, QK-norm ×2, RoPE ×2), `o_gemv_ffn` is 3
(`o_gemv_ffn_multi.py`: `matvec_2tile_add`, `matvec_swiglu_rms` cascade, `matvec_2tile_add`),
`lm_head` 19 — **28 × 11 + 19 = 327 launch boundaries per token**, against 57 `xrt.run`s and
one logical token. **Measured: 109 µs per boundary, ~~with §1.4's caveat that the probe did not hold the per-launch DMA geometry constant~~ `[2026-08-21]` and 106–108 µs with the geometry held constant (§1.5)** — corroborated independently by the per-layer gaps. Applied to the 308 per-layer
boundaries that is **33.6 ms of the 89 ms token — 38 %** — before a single weight byte moves.
It also explains the June int4 result: Llama-1B int4 decode went 12.2 → 17.8 tok/s (1.46×,
15.3 re-measured today) on a ~4× narrower weight stream because its token still carries
152 boundaries (16.6 ms) and a 525 MB bf16 LM head (14.9 ms).

This is the exact inverse of the Hexagon design. `ggml-hexagon`'s DSP runs **one persistent
program**; an "op" in a batch is a descriptor the program interprets, and 1024 of them cost one
`dspqueue_write` and zero reconfiguration. The lesson [55](55-hexagon-llama-cpp-lessons-for-xdna2.md)
called "op batching" lands here one level lower than `xrt.run`: **it is the launch count, not
the submission count, that has to fall.**

A second, smaller structural fact from the same place: `_LM_N_PART = 8192` is pinned by a BD
**repeat-count limit** (`qwen3_0_6b_decode.py:85`: `n_part/32 − 1 = 255`) — one launch cannot
stream more than 8192 rows of this GEMV, which is why the LM head is 19 launches at all.
Hexagon's DMA descriptors have no such cap; the XDNA analog is a BD chain or an outer loop in
the runtime sequence rather than one BD's repeat count.

## 3. The translation table

Each row: the Hexagon mechanism (55 §4, corrected) → what this path does today → the
optimization → predicted effect on the Qwen3-0.6B token of §1.1 (*arithmetic* unless marked)
→ the gate.

| # | Hexagon mechanism | Here, today | Optimization | Predicted effect | Gate / experiment |
|---|---|---|---|---|---|
| **O1** | **No reconfiguration between ops**: one persistent program; an op is a descriptor | 327 PDI loads / token (§2) | **Cut launch boundaries per layer 11 → ≤ 3**: (a) Q, K, V as **one** GEMV launch over the concatenated `[wq; wk; wv]` (N = 4096) — Hexagon's `MUL_MAT_QKV`; (b) the M = 1 vector ops (RMSNorm, QK-norm, RoPE, SwiGLU glue) as **prologue/epilogue inside the GEMV core program** rather than their own launches — Hexagon's `RMS_NORM+MUL` / `MUL_MAT+ADD` fusions (at M = 1 these touch 1–4 K elements; a launch for each is all overhead); (c) `lm_head` partitions under **one** `aie.device` as a single-device BD chain (an "outer loop in the runtime sequence" does not work: the lowering deliberately resets at launch end, `AIRRtToNpuPass.cpp:1037` `[per Codex review]`). (a) is builder work; **(b) and (c) need new core kernels** — stitching only concatenates launch bodies (`stitching.py:383`); `matvec_swiglu_rms.py` shows fused epilogues are possible, not that they are mechanical | **with O3, jointly** −25 … −35 ms / token (§4) | the isolating probe of §1.4 first; then `rms_qkv` 8 → 1–2 launches as the first kernel change, gated by the model's `make verify` and the profiler's `rms_qkv` line |
| **O2** | **Op batching**: ≤ 1024 ops per `dspqueue_write`, ≤ 16 in flight | 57 `xrt.run` + wait per token, one per ELF | `dispatch.run_sequence(require_single_submission=True)` over adjacent runs: layer L's `o_gemv_ffn` → layer L+1's `rms_qkv` have no host op between them ⇒ **57 → 30 submissions** today (RMS₀; 27 `(O_L, RMS_{L+1})` pairs; O₂₇; LM head — the final RMSNorm is a CPU barrier before the head, `inference.py:431` `[per Codex review]`), lower once attention and the final norm are on device (O6) | 0 … −6 ms / token **until measured** (host submit/wait per run, ~50–200 µs) | exists; a driver flag; dispatch vector shows `host_submissions` |
| **O3** | **All HVX threads stream, each with its own DMA queue** (`n_threads` rows × per-thread `dma_queue`) | per-layer ELFs at 8–15 GB/s vs 32 GB/s for `lm_head`; `matvec_2tile_add` uses **one core per column** (`n_cores=8`, `herd [8,1]`) | After O1, re-measure each stage's streaming rate; where below the large-GEMV reference: both shim input channels per column, 2–4 rows per column splitting the output rows — **new mappings, not the existing `lm_head` geometry** `[per Codex review]` | **not additive with O1**: the joint O1+O3 saving is `71.6 ms − (measured post-fusion time)`, bounded below by the 27.6 ms weight-stream floor plus vector/dequant work. 32 GB/s is an *achieved reference* for large eight-column GEMVs, not a ceiling (8 × 5.336 = 42.7 GB/s of shim ports before contention) | per-stage rate from the profiler's `NPU Run` column vs bytes |
| **O4** | **The output matrix is quantized too** (Q4_0/Q8_0 `output.weight`; repacked once) | `lm_head` bf16, 311 MB / token = 9.7 ms at 32 GB/s | int4 (q4_0 gs 32 symmetric or AWQ) LM-head GEMV — `matvec_int4_packed.py` exists, `symmetric=True`; one resident packed copy beside the bf16 embedding rows | **−6 … −7 ms / token** (311 → ~90 MB at the same rate) | verify top-5 token set (the head is where int4 error shows first); `bytes_transferred` |
| **O5** | **All weights narrow, resident, repacked once** | bf16 per-layer weights, 881 MB / token | `w4_decode` for Qwen3-0.6B — the Llama int4 QKV builder lacks Qwen's per-head QK-norm, so this is a builder change, not reuse `[per Codex review]`; **after O1**, because the Llama int4 data says weight width alone buys 1.46× while boundaries dominate, and the int4 GEMVs measured at 4–14 GB/s cannot be assumed to reach the bf16 reference rate | **no budget credit today**; 881 → ~250 MB is the byte ceiling, the rate is unmeasured | prediction doc first ([56 §4 H2b](56-full-model-mixed-precision-study-plan.md)); verify; `quant_*` columns populated |
| **O6** | **Attention rides in the batch** (`FLASH_ATTN_EXT` at M = 1, KV in mapped DDR, when F16 K/V) | CPU attention; whole KV slice converted bf16→f32 per layer per token; Python loop over heads; KV written on host | (i) now: keep the host KV cache in f32 and vectorize over GQA groups — removes the per-token conversion; (ii) then: the `attn_decode_npu2` kernel (device KV; its `pos` is **compile-time** today — `pos_host`, `attn_decode_npu2.py:470` — so first make it a run-time argument) generalized to hd 128 / GQA 2, inside the per-token runlist | (i) removes the copies and the Python loop, but **cannot hold attention at ~3 ms**: at 2,048 context the host must still read ~470 MB of K/V per token (16.8 MB × 28) — a constant-factor win over today's 54 ms, to measure `[per Codex review]`; (ii) is the real fix and needs a kernel/interface change (`pos` at run time) — **no budget credit today** | long-prompt profile (ctx 1024, 2048) before/after |
| **O7** | **Per-token state stays resident**; position is an op parameter | RoPE LUT for `pos` rebuilt with `np.tile` and uploaded **per layer** (2 BO writes × 28 = 56 / token, identical across layers); a dead 6 MB `np.zeros((hidden, emb))` allocated per layer per token (`_run_o_gemv_ffn`) | share one LUT BO across layers (`shared_nonstatic` pool) or resident full table + `pos` as a run-time arg (no kernel here takes one yet); preallocate the dead args once | small at bf16 (BO write 0.01 ms/call); ~1 ms/token of host allocation; matters once O1–O5 shrink the token | profiler `BO Write` and CPU buckets |
| **O8** | **`ubatch`: the physical chunk is the compute shape**; a 60-token prompt is one 64/128-row ubatch | every prompt padded to 2048 (`inference.py:645`) | compile prefill at `M ∈ {128, 256, 512, 1024}` and run ⌈L / M⌉ chunks ([56 §3.4](56-full-model-mixed-precision-study-plan.md)); single-chunk prompts need no new kernel | **TTFT for a 60-token prompt 1.45 s → ~0.1 s**; at 1024 tokens ~0.7 s | [56](56-full-model-mixed-precision-study-plan.md) H1a/H1b gates |
| **O9** | **HMX flash attention** (2026 rework: +40 % prefill) | FA is 51 % of the prefill layer at 0.8 TFLOPS causal-effective vs 3.7 for the FFN GEMM; 24 MB of host-transposed Q/K/V uploaded per layer | (i) causal block skipping in the head-first kernel (half the tiles are masked); (ii) **hypothesis, undemonstrated**: move the seq↔head transpose into the DMA by addressing bf16 pairs as 32-bit elements — the repository establishes only that the sub-32-bit innermost stride must be 1 (`data_transfer_transpose/dma_bf16/transpose_bf16.py:9`); legality of the reinterpretation, alignment and the on-device handoff are unproven `[per Codex review]` | (i) up to −10 ms / layer (−280 ms / prefill) — causal masking is applied after a dense K matmul today (`attn_npu2.py:734`), so the skip is real; (ii) −1.4 ms upload + ~5 ms/layer of host transposes **if** it works | FA lit at 2048 hd 128; verify |
| **O10** | **Plans cached by graph uid; kernel params precomputed on the host** | artifact cache keyed by name (`cache.py:358`) | plan-hash keying ([56 §3.3](56-full-model-mixed-precision-study-plan.md)) | correctness of every experiment above (no silent shape collisions) | [56](56-full-model-mixed-precision-study-plan.md) H1a |
| **O11** | **Power voted at session start** | Turbo required by the study runner; `make profile` prints nothing about pmode | print observed pmode in the profile header; refuse off-Turbo in `llms/` drivers as the study does | measurement validity | one-line change |
| **O12** | **Per-op cycle counters and PMU events returned with the batch** | host-side three-segment timing per `xrt.run` | optional: per-launch timestamps from the runtime sequence (trace) — after O1 there are few enough launches to read by eye | — | — |

## 4. The decode budget — a defensible band, not a waterfall `[rewritten per Codex review]`

The first draft added O1, O3, O5 and O6 as independent savings down to ~18 ms/token. They are
not independent (removing boundaries also removes part of the "rate gap" charged to O3) and
two of them have no measured rate behind them. The defensible statement, on §1's measurements:

| Item | Credit | Condition |
|---|---|---|
| baseline (devq 436) | **89 ms / token** (11.2 tok/s) | short context; 145 ms at ~1,900 context (devq 444) |
| O1 + O3 **jointly** | **−25 … −35 ms** | the joint saving is `71.6 − (post-fusion measured)`, floored by the 27.6 ms weight-stream time plus vector/dequant work; credited only after the isolating probe (§1.4) |
| O4 (int4 LM head) | **−6 … −7 ms** | after the accuracy gate |
| O2 + O7 | **0 … −6 ms** | until measured |
| O5, O6 | **0** today | O5's int4 rate and O6's device attention are unmeasured |
| **Conditional band** | **41 – 58 ms / token (17 – 24 tok/s)** | |

The Hexagon-style end state — one submission, a few launches, narrow resident weights,
attention in the batch, at a number near llama.cpp's 54 tok/s for Llama-1B on v81 — remains the
*direction*; as a figure it is a stretch hypothesis, not arithmetic, until O5 and O6 have
rates. Moving from the band to that figure is what [56](56-full-model-mixed-precision-study-plan.md)'s
H2/H3 phases measure.

## 5. Experiments, in order

1. ~~**Per-boundary cost**~~ **DONE** (devq 445, §1.4): **109 µs per boundary**.
2. ~~**int4 decode decomposition**~~ **DONE** (devq 440, §1.3): int4 GEMVs at 4–14 GB/s, the
   bf16 LM head 23 % of the token, 152 boundaries.
3. ~~**Long-context decode profile**~~ **DONE** (devq 444, §1.1): 25 / 54 ms per token of CPU
   attention at ~1,000 / ~1,900 context.
3b. ~~**Repeat-geometry isolation**~~ **DONE** (devq 448/449, §1.4): `m_input 8` is 8.5 %
   faster at the same launch count; the defect is not the repeat limit. ~~**Still to run**: the
   19-devices-×-2-segments probe and the `lm_head` back-to-back re-execution gate.~~
   **`[2026-08-21]` both DONE** (devq 450–452, 458, §1.5): boundary cost **106–108 µs**
   with the geometry held fixed (the two-segments form is unbuildable — one device per
   segment — so the isolation adds near-empty configurations instead); the gate is checked
   in under `XFAIL`, 5 / 7 dispatches wrong on the production artifact, idle does not heal.
3c. ~~**`m_input = 8` on the production LM head**: `make verify` for Qwen3-0.6B; predicted
   −0.85 ms/token.~~ **`[2026-08-21]` DONE and LANDED** (devq 453 → 454 → 455, same session,
   Turbo observed): `make verify` **PASS** (2 / 2 prompts) with
   `build_lm_head_gemv_qwen_module` at `m_input = 8`; `make profile N_TOKENS=32` twice
   before and twice after — `lm_head_gemv` kernel **9.77 / 9.79 → 8.79 / 8.79 ms**
   (**−1.0 ms/token**, the prediction was −0.85), `o_gemv_ffn` 1.54 → 1.53, `rms_qkv`
   1.04 → 1.03 (unchanged within noise). The ELF shrank 8,605,040 → 8,307,120 bytes (half
   the kernel calls per tile). The LM-head re-execution defect is unchanged by this (devq
   448 already showed it at `m_input = 8`); the gate of §1.5 stays red.
4. ~~**O2 prototype** behind a driver flag: `run_sequence` over the (L `o_gemv_ffn`, L+1
   `rms_qkv`) pairs; dispatch vector `host_submissions 57 → 29`; `make verify`.~~
   **`[2026-08-21]` DONE, MEASURED, SMALL** (devq 456, 457, 460). `qwen3_0_6b_decode_runlist.py` (removed in the 2026-08-22 cleanup, tag `pre-cleanup-20260821`; was)
   behind `QWEN3_DECODE_RUNLIST=1` / `--decode-runlist`: the 27 (L `o_gemv_ffn`, L+1
   `rms_qkv`) pairs as one `run_sequence` each, `x` device-resident between them; dispatch
   vector per token **57 → 30 submissions** (27 × 2 entries + 3 singles), `air` 327
   unchanged by construction. `make verify` **PASS** (2 / 2). Cost, `make profile
   N_TOKENS=32` ×2 against the same-session `m_input = 8` baseline (devq 455): first
   prototype **slower** — layer-loop wall 3264 / 3323 vs 2614 / 2637 ms, pair avg 3.01 /
   3.08 ms against 1.53 + 1.04 for the two separate runs — because `run_sequence`
   re-derived the pool plan every call (0.18 ms) and the 27 pair pools uploaded their
   weights inside the first measured token. With the plan memoized
   (`dispatch._plan_memo`) and the pools preloaded: **2560 / 2564 ms** layer-loop wall,
   pair **2.70 / 2.71 ms avg, 2.59 min** — **−2 ms/token, about 2 %**. The mechanism
   works and is correct; its ceiling is the per-`xrt.run` fixed cost (§1.5's 146 µs
   intercept, minus a runlist entry's own cost) × 27, and that is what it delivers. It
   stayed behind the flag (the pools doubled the resident decode weights) and was never the
   production path; the 2026-08-22 cleanup removed the module and the flag (tag
   `pre-cleanup-20260821` holds them). The launch count is the cost that matters;
   O1 is where the token goes. **On top of O1's 4-launch stage** (devq 464, ported to the
   9-arg ABI): layer-loop wall **2186 / 2120 vs 2247 / 2249 ms**, pair 2.27 / 2.23 ms avg
   (2.17 min) against 1.53 + 0.62 — **−2 to −4 ms/token**; the saving per pair is the same
   ~0.1 ms, a larger share of a shorter token.
5. ~~**O1 first cut**: `rms_qkv_qknorm_rope_gemv` as one GEMV launch over `[wq; wk; wv]` with
   QK-norm + RoPE as an epilogue; predicted `rms_qkv` line 1.03 → ~0.4 ms (8.4 MB at 32 GB/s
   + 1 boundary); `make verify`.~~ **`[2026-08-21]` FIRST CUT DONE and LANDED: 8 → 4
   launches, no new kernel** (devq 461 probe, 462 verify, 463 profile).
   `build_rms_qkv_qknorm_rope_gemv4_module`: RMSNorm; **one** GEMV over the row-packed
   `[wq; wk; wv]` (4096 × 1024); **one** QK-norm over the Q|K rows with a per-row weight
   (`_build_qknorm_1d(per_row_weight=True, in_total=qkv_dim)` — the weight is the host-tiled
   `[q_norm × 16; k_norm × 8]`, static, and the launch takes the whole `qkv` arg and reads
   its first 24 rows, so V rides untouched in the tail); **one** RoPE over Q|K with the
   position LUT tiled 24×. Same kernels, same bytes. Probe: outputs **bit-identical** to the
   8-launch ELF on all four boundaries, deterministic; **1.125 → 0.680 ms per layer, −0.445
   ms = 111 µs per removed boundary** — §1.5's constant, recovered a third way. Production
   (`_RMS_QKV_KERNEL = "rms_qkv_qknorm_rope_gemv4"`, single-owner helpers `rms_qkv4_args` /
   `run_rms_qkv4` in `qwen3_0_6b_decode.py`, preload and the O2 module ported): `make verify`
   **PASS** (2 / 2); `make profile N_TOKENS=32` ×2: `rms_qkv` kernel **1.03 → 0.62 ms**,
   per-layer 2.92 → **2.52 ms**, layer-loop wall 2614 / 2637 → **2247 / 2249 ms** over 896
   invocations = **−11.5 ms/token**. With item 3c the token is **89 → ~76.5 ms** (−14 %)
   from two changes that moved no new bytes. A wall met on the way: the `memref.subview` +
   `memref.cast` prelude that `o_gemv_ffn_multi` uses fails when the alias has ONE use
   (`'memref.cast' op using value defined outside the region` — the cast is sunk into the
   launch region, its subview is not; devq 459), hence `in_total`.
   **Ported to Qwen3-1.7B the same day** (devq 465–470; emb 2048, same head geometry):
   stage probe 1.326 → 0.873 ms/layer (113 µs per boundary), `make verify` PASS with the
   4-launch stage (devq 466) and again with `m_input = 8` on its LM head (469); `make profile
   N_TOKENS=32` ×2: `rms_qkv` **0.80 ms**, `o_gemv_ffn` 2.67, LM head 16.4 → **15.9–16.2 ms**
   (K = 2048 gains less from `m_input = 8` than K = 1024 did, and the two reps straddle the
   noise), per layer ~4.0 ms, layer loop 3554–3647 ms / 32 tokens — **≈127–130 ms per 1.7B
   token** under recorded Turbo, the first such number (the June table is pmode-unrecorded;
   the stage A/B says the 8-launch form was ≈12.7 ms/token slower). Qwen3-4B keeps the
   8-launch builder: its verify is oomd-deferred, so the port cannot be gated. The host side
   of the 4-launch stage now lives in `shared/infra/decode_qkv4.py` (one owner of the 9-arg
   layout for every Qwen3 driver).
   **What is left of O1** (the predicted ~0.4 ms needs 4 → 1–2): ~~the RMSNorm launch
   (1 boundary; fold as a prologue on the broadcast `x` or precompute `rstd` host-side)~~
   **`[2026-08-21, later]` tried as host RMSNorm (3 launches): −0.027 ms/layer only, and
   the form hangs when re-executed back-to-back (§1.5's table, row 6) — not shipped; the
   builder keeps `host_rmsnorm=True` for the day the defect is understood** — and
   the QK-norm + RoPE pair (2 boundaries; an epilogue needs each column to own whole heads,
   i.e. a column-contiguous row distribution that the current L2-staged GEMV does not have —
   a new core kernel). Each remaining boundary is worth ~107 µs × 28 = 3 ms/token.
5c. **`[2026-08-23]` O1 second half — the head-aligned GEMV with QK-norm + RoPE as an in-core
   epilogue: 4 → 2 launches, LANDED** (devq 553 compile, 555 probe, 556 verify, 557 reexec,
   558 profile; evidence `results/o1-epilogue-20260822/`). New kernel
   `matrix_vector_multiplication/bf16/mv_heads.cc` (`qkv_heads_chunk_bf16`: `mv.cc`'s matvec
   with a row stride; on a head's last chunk the epilogue runs QK-norm — f32 sum of squares,
   rstd truncated to bf16 as `_build_qknorm_1d` does — then `rope_halfsplit`'s loop verbatim;
   `HEAD_DIM` baked in, object `mv_heads_hd128.o`), builder
   `build_rms_qkv_qknorm_rope_gemv2_module` (5-arg ABI) and `_build_qkv_heads_gemv` in
   `rms_qkv_qknorm_rope_multi.py`, host side `decode_qkv4.py` (`prep_weights_2`, `run_2`,
   `qkv2_gather`), `QWEN3_RMS_QKV_LAUNCHES` (default **2**) in the 0.6B driver. **Mapping**:
   column `tx` owns logical rows `[512 tx, +512)` = heads `4tx..4tx+3` (cols 0–3 Q, 4–5 K,
   6–7 V), but the *whole-head-per-iteration* tile (256 KB per column, 4 iterations) measured
   **0.13 ms/layer slower** than matvec.py's GEMV — the single-buffered memtile tile refills
   only after the core drains all 16 chunks, so fill and drain serialize (tile sweep 8 rows
   0.463 → 128 rows 0.577–0.588 vs 0.444 ms; a chunked L3→L2 fill loop changed nothing; devq
   544/545/552). Production therefore keeps matvec.py's 8-row tile and 64 iterations: iteration
   `i` gives column `tx` chunk `i mod 16` of head `i div 16`; the core accumulates the head in
   a persistent L1 buffer over 16 consecutive iterations and runs the epilogue on the last
   chunk. Three walls shaped the rest: a core tile has **two inbound DMA channels** (A from the
   memtile, B from the shim; a fourth stream fails the router, devq 540), so the chunk TAG and
   Q/K/V KIND are baked into a 64-element row padding of the static weight (+6.25 % bytes,
   rows stored iteration-major) and the epilogue operands ride in B as `[normed | lut | q_norm
   | k_norm]` (1408 elements per chunk); the locality verifier rejects a per-iteration write to
   the head's logical slot (`iteration variable does not appear in any offset of this access`),
   so each iteration writes a disjoint 1024-element slot (128 KB) that the host gathers.
   Layout + slots cost the single GEMV 0.444 → 0.492 ms (devq 552), ≈48 µs/layer against the
   2 × ~107 µs of boundaries removed. **Probe** (devq 555, random weights, the driver's LUT at
   six positions, two x): `normed` and `v` bit-identical to the 4-launch form; `q`/`k` vs an f32
   reference **3.4–7.1e-3 of scale (rq2) vs 9.6–15.5e-3 (rq4)** — the epilogue is the more
   accurate path; rq2 vs rq4 differ by 1–2 bf16 ulp on ~80 % of elements (not bit-identical,
   stated); deterministic; stage **0.672 → 0.494 ms/layer** interleaved ×20 (−0.178 = 89 µs
   per removed boundary). `make verify` **PASS** 2 / 0 (devq 556). **Re-execution gate**
   (devq 557, the `lm_head_reexec_gate` shape on the production module): d2–d5 back-to-back
   bit-identical to d1, d6 after another ELF, d7 new input — **7/7 clean**. **Profile** (devq
   558, same session, Turbo before and after, `make profile N_TOKENS=32` ×2 each):
   4 launches **13.04 / 12.91 tok/s** (76.7 / 77.5 ms; layer loop 2201.7 / 2227.8 ms per 896
   invocations; `rms_qkv` 0.65 / 0.66, kernel 0.61 / 0.62) → 2 launches **13.40 / 13.52
   tok/s** (74.6 / 74.0 ms; layer loop 2135.6 / 2114.4; `rms_qkv` **0.49**, kernel 0.44);
   `lm_head_gemv` 7.68–7.72 and `o_gemv_ffn` 1.57–1.61 unchanged. **−2.1 … −3.5 ms/token**
   (layer-loop pairs (2201.7 − 2135.6) / 32 = 2.06 and (2227.8 − 2114.4) / 32 = 3.54;
   76.7 / 77.5 → 74.6 / 74.0 ms by tok/s) **against the −6 predicted**: the `rms_qkv` line
   moved −0.16 … −0.17 ms × 28 = −4.6 ms, and that line already contains the ~48 µs/layer
   layout/slot cost (it is why the line did not fall by the full 2 × ~107 µs); the gap between
   −4.6 and the token delta is the other lines' rep-to-rep spread (the 4-launch reps themselves
   differ by 0.8 ms/token). Boundaries per token 206 → **150** (28 × 5 + 10). The 1.7B driver keeps its 4-launch names (re-verified on
   the same compiler: `make verify` PASS 2 / 0, devq 559). Open: the 128 KB slot D2H and the +6 % weight
   bytes are the ~48 µs still paid (a partial BO sync or a host index trick could recover
   some); the form assumes `rows_per_col % head_dim == 0` (true for 0.6B / 1.7B / 4B); the
   1.7B port (K = 2048) is not done.
5d. **`[2026-08-23]` Llama-3.2-1B head (queue item 12): the mixed partition does NOT pay; `m_input
   8` does — LANDED** (devq 563 probe, 564 landing; evidence `results/llama1b-head-20260823/`).
   The queue carried "2,816 pad rows, ~0.36 ms" — byte arithmetic (2816 × 4 KB at 32 GB/s),
   never measured. Measured, four cells interleaved ×20 under Turbo on the head alone (random
   weights, every cell correct vs f32, max_rel 2.75e-3, argmax agrees):
   8 × 16384 at `m_input 4` (production) **15.10** ms p50 (min 14.84); 7 × 16384 + 13568 at
   `m_input 4` **15.15** (14.91); 7 × 16384 + 13568 at `m_input 8` **14.16** (13.84);
   8 × 16384 at `m_input 8` **13.91** (13.83). Dropping the pad rows is never faster — the
   13568-row tail (212 iterations, not a multiple of the preset's 16-iteration loop tiling) costs
   what its 11.5 MB saves — while `m_input 8` alone is **−1.19 ms** (as on the Qwen3 heads,
   §1.4). A preset trap on the way: the Qwen probe's backend fails aiecc on a 16384-row partition
   at `m_input 4` (`push_queue ... Repeat count exceeds the [0:255] range`, devq 562); Llama's
   `LM_GEMV_BACKEND` carries `runtime_loop_tiling_sizes [16, 16]`, which is what keeps it under
   the cap. **Port**: `build_lm_head_gemv_llama_module` (`m_input=8`, partitions unchanged, host
   slicing untouched); the re-execution gate factored into `shared/infra/lm_head_reexec.py`
   (the Qwen wrapper keeps its name and lit; Llama gets `make check-lm-head-reexec`). **Landing**
   (devq 564, same session, Turbo before and after): `make verify` PASS 2 / 0;
   `check-lm-head-reexec` 7/7 clean, bit-identical; `make profile N_TOKENS=32` ×2 before / after
   on the instruct prompt (9 tokens to EOT, 10 head calls each): `lm_head_gemv` **14.94 / 14.94
   → 13.86 / 13.98 ms** avg (min 14.71 / 14.74 → 13.73 / 13.74), per-layer wall 4.82 / 4.73 →
   4.66 / 4.72 ms, `o_gemv_ffn` 3.13 / 3.10 → 3.12 / 3.13 and `rms_gemv_rope` 0.96 → 0.91–0.93
   unchanged within spread: **−1.0 ms per token** (≈ 89 → 88 ms by 16 × per-layer wall + head;
   the README's 92 ms / 10.8 tok/s is the June figure, pmode unrecorded). The Qwen3-0.6B
   wrapper on the shared gate body: 7/7 (devq 565). Llama-3.2-3B (emb 3072) and the int4
   sibling keep `m_input 4` — not measured here.
5b. **`[2026-08-21]` LM-head partitioning, the planner's finding — DONE and LANDED** (devq
   476 probe, 478 verify, 479 profile). Doc 56 H0's planner derived that at `m_input = 8`
   the BD repeat cap admits 16384-row partitions, and that stitching takes launches of
   *different* shapes, so the Qwen3-0.6B head can be **9 × 16384 + 1 × 4480 = 10 launches
   with 64 pad rows** instead of 19 × 8192 with 3,712: 9 boundaries and 7.6 MB fewer.
   Probe (same weights, exact outputs): **9.348 → 8.253 ms** (−1.10; predicted −1.2).
   Production (`_LM_PARTS` in `qwen3_0_6b_decode.py`, `build_lm_head_gemv_module(parts=…)`,
   preload / `_run_lm_head` / the re-exec gate over the partition list): `make verify`
   **PASS**, `lm_head_gemv` kernel **8.79 → 7.90 / 7.95 ms**, ELF 8.3 → 2.56 MB, compile
   42 → 13 s. The layer-loop wall of that profile (2395–2433 ms) is **not** a token number:
   a detached int4 compile was pinning a core and the per-layer host glue slowed 0.2 ms
   while the device kernel lines did not move — re-profiled clean below when the compile
   ended. **Ported to Qwen3-1.7B the same night** (devq 485 verify PASS, 487 profile):
   `lm_head_gemv` 15.9–16.2 → **14.6–14.95 ms**, the token **7.70 / 7.83 tok/s ≈ 128–130 ms**.
   Llama-1B's head has the same 2,816-pad-row waste (7 × 16384 + 13568, same launch count;
   ~0.36 ms) and is not yet ported.
6. **O4**: int4 LM head; predicted 9.7 → ~3 ms. **`[2026-08-21]` BLOCKED on compile time,
   unmeasured.** `probe_o4_lm_head_int4.py` (evidence root) builds the head with the int4 packed
   GEMV (`matvec_int4_packed.build_module`, RTN uint4 gs=128 via `awq_pack.fake_quantize_awq_int4`,
   `pack_inputs`), whose `M_PER_LAUNCH` chunks are *iterations of one `air.launch`* — so the whole
   head would be **one configuration** instead of ten. The module builds and parses (one
   `air.launch`, 231 lines) but `aircc` did not finish in **75 minutes, twice** (devq 468, then a
   detached build in its own cwd), single-threaded in the MLIR pipeline before any core compile.
   The llama int4 per-layer GEMVs (one iteration, M = 2048) compile in seconds, so the cost is in
   the 19-iteration × 8-core × 128-tile runtime sequence some pass unrolls. **Bisected the same
   night** (`job_o4_bisect.sh`): 1 and 2 iterations compile in 16–17 s; **4 iterations fail at
   `aiecc` — `'aiex.npu.push_queue' op Repeat count exceeds the [0:255] range`** — the single
   launch's iteration loop folds into one BD repeat, so the "one configuration for the whole head"
   form is impossible past 2 iterations at this geometry (the same cap that pins the bf16 head's
   rows per launch), and the 19-iteration compile was lost in a pass before that check. The int4
   head therefore takes the bf16 head's shape — ten stitched `air.launch`es, 10 boundaries —
   and what int4 buys is bytes alone (319 → ~88 MB): `probe_o4b_int4_stitched.py`, devq 488 —
   **measured, and closed for now**: the ten-launch int4 head (81 MB packed, RTN gs=128) runs in
   **7.37 ms against the bf16 head's 7.82 ms, −0.46 ms/token**; the int4 packed GEMV streams at
   **11 GB/s** (the bf16 head streams at ~47 GB/s between its boundaries), so the kernel is
   dequant-bound exactly as §1.3's per-layer int4 GEMVs were (4–14 GB/s), and a 3.8× byte cut buys
   6 %. The stitched form also returned wrong values (max_rel 0.69 against the kernel's own CPU
   dequant reference, deterministic — a static mapping error in the ten-slice stitch, not chased:
   the timing is representative, the ceiling is 0.46 ms). **The prerequisite for O4 is a faster
   int4 GEMV** (dequant throughput, HMX-class on Hexagon), a kernel project; the accuracy question
   (RTN int4 on the LM head under the top-5 gate) stays unasked until then.
7. Then O3 / O5 / O6(ii) / O8 / O9 per [56](56-full-model-mixed-precision-study-plan.md)'s phases.

## 6. What does not transfer, restated for the inference path

`NDEV` sessions (no VA cliff); the `M > 4` HMX threshold (an HMX/HVX fact); Q8 activation
quantization as the first precision step (Hexagon's HMX path does not use it either, and the
decode GEMVs here are overhead-bound, not MAC-bound); a latency-model mode selector. And one
thing that transfers only with a warning: Hexagon's decode numbers are the *ceiling* this
table walks toward, measured on a different memory system; the comparison that matters is
each row's own before/after under recorded Turbo.

## 7. Codex review

Report: 57a (the verbatim Codex review, retired 2026-08-22 to git tag `pre-cleanup-20260821`), verbatim. Verdict as delivered:
"major revision — the launch-count diagnosis is directionally persuasive, but 109 µs is not an
isolated PDI-boundary measurement, and the 18–22 ms endpoint rests on overlapping or
unsupported assumptions." Applied, each marked above: the ELF reconfiguration mechanism
restated (device images via `.pdi.N` sections, not a raw `load_pdi`); the probe's confound
(per-launch iteration count and repeat geometry change with `n_part`) and the two isolating
experiments; the defect re-read as stale state rather than a last-tile bug, with the note that
the token gate cannot see it; launch counts confirmed; padded LM-head bytes (318.8 / 536.9 MB)
for rates; 32 GB/s demoted from ceiling to achieved reference; O1(b) and O1(c) re-costed as
new kernels and O1(c)'s mechanism corrected; O2 = 30 not 29; O3 made non-additive with O1; O5
re-costed (QK-norm, unmeasured int4 rate); O6(i)'s 3 ms claim withdrawn (the host must read
~470 MB/token at 2,048 context regardless); O9(ii) marked undemonstrated; O10–O12 moved out of
the performance budget; §4 replaced by the 41–58 ms conditional band.

Codex's list of Hexagon mechanisms this document under-uses — the capacity/live-set fusion
planner, one-time shared activation preparation reused across stacked matmuls, and explicit
host-fallback split accounting — is [56](56-full-model-mixed-precision-study-plan.md)'s H0
by another name, and is why O1's fusion decisions should be made by that planner rather than
by hand once it exists.
