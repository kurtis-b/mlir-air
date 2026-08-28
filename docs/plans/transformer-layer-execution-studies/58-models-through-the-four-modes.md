# 58 — Real models through the four execution modes (option 4a)

`[2026-08-26]` **PLAN.** Operator decision, this date: of the two ways to put the mode axis and
the model axis in one experiment — (a) port real models INTO the four modes, or (b) express the
mode axis in the model runner — take **(a)**. This doc is the decomposition, the gates, and the
honest cost. It supersedes nothing; doc 56 §3.1's "the two axes are orthogonal and stay so"
remains the description of the *sibling* design, and this program is the deliberate exception
to it.

## 1. What is actually missing, from source

The four modes (`coarse`, `offload`, `runlist`, `fused`; `study/profiles.py:307`) each execute
**one synthetic transformer layer** — an encoder-BERT or decoder-GPT2 graph of LayerNorm + GeLU
+ MHA (`builders/block.py:183,202`) — over **weights the seam generates itself**
(`opcheck_layer.py:476`, `generate_golden_reference`). They have no GEMV/M=1 path, write
`dtype="bf16"` unconditionally (`run_mode.py:272`), and never populate a `quant_*` column.

| # | Missing | Where | Model-specific | First estimate |
|---|---|---|---|---|
| 1 | **Weights-injection seam.** `prepare_layer_dispatch(shape, seed, cache_dir, label, extra)` takes six shape scalars and generates everything else; fault injection indexes into the generated `inputs` list, so the contract is coupled | `opcheck_layer.py`, 4 × `pattern/*/`, `opcheck_specs.py` | no | days |
| 2 | **A decoder-LLM layer graph** — RMSNorm, SwiGLU (3 GEMMs), RoPE, GQA widths, optional QK-norm | new `builders/*`, 4 × `pattern/*/`, `pattern/reference.py`, 4 × lit | per family | **see §2 — smaller than it looks** |
| 3 | **N-layer loop, embedding, LM head, KV cache, sampling** — `BLOCK_INPUT_NAMES` is one layer's seven tensors (`builders/block.py:169`) | new | mechanism no, shapes yes | weeks |
| 4 | **Registry rows at the model's shapes** — `resolve_gemm_spec` RAISES on an unmeasured shape, deliberately, no fallback (`builders/qkv_proj.py:46-53`) | `sweep/`, `kernel_registry/` | yes | **partly done for Qwen3-0.6B** (item 13 landed six rows at M 512/1024; M 2048 already present) |
| 5 | **Mode-specific structural bounds** — `fused` caps at `rows*cols ≤ 2^20` (512 tokens at emb 2048); `offload`'s shared-xclbin is bounded to single-launch modules; `fused` cannot be one ELF (FA wants `runtime_loop_tiling_sizes=[1,1]`, wide GEMMs `[2,2]`, and `[2,2]` hangs `mha_out_proj`, `ERT_CMD_STATE_TIMEOUT` 3/3) | builders | no | open — some modes may simply refuse some models |
| 6 | **A `dtype`/`quant_*` producer for layer rows** | `run_mode.py`, `pattern/*/` | no | small, after 1–2 |
| 7 | **A workload→mode dispatcher** (doc 03:87, doc 55:69-88 — "nothing in the port expresses such a choice yet") | new | no | open design, and NOT required for this program |

## 1a. `[2026-08-26]` M0 ran — three findings, and the plan below is amended by them

Phase M0 executed read-only ([58a](58a-4a-m0-scope.md), 605 lines, verbatim). It did what M0
existed to do: it replaced a guess with a number, and it corrected two things in this doc's own
first draft. Read 58a before starting any phase.

**(i) §2's cost-halving conclusion holds, but its stated mechanism was wrong.** Every file in
`llms/shared/builders/` is a whole-ELF **stage** builder — its own `__init__.py:5-6` says so —
and `gemm_builder` is *not* the leaf exception §2 originally named. The real importable leaves
are top-level example dirs (`weighted_rms_norm`, the best-shaped leaf in the tree;
`silu_and_mul.build_module_2d`) and the **private single-launch `@module_builder` functions
inside** the shipped stage builders: `_build_rope_2d` (`rms_gemms_rope_multi.py:61`),
`_build_qknorm_2d` (`rms_qkv_qknorm_rope_multi.py:90`), `_build_add_2d_to_2d` (which the study
*already* imports, `builders/elementwise_add.py:57`). With `stitching.stitch_elf`, the
**device-IR half** of gap 2 really is re-assembly. The **oracle** and **shape-plumbing** halves
are not, and this doc's first draft priced neither.

**(ii) Qwen3-0.6B is the wrong first family, and one of its cells is a structural refusal.**
`block_config` (`builders/block.py:296`) and `runlist_config` (`pattern/runlist/runlist.py:287`)
**require `num_heads * head_dim == emb_dim`**, which Qwen3-0.6B violates (2048 ≠ 1024). And the
study composes the **seq-first** FA, which asserts `dv_chunks == 1`
(`attn_npu2_seqfirst.py:121`), while `fa_headfirst.py:7-13` states that head_dim 128 **must**
use the heads-first kernel with two host transposes. Consequences: **`(fused, Qwen3-0.6B)` is a
structural refusal** — host transposes contradict its one-submission definition — and
`(coarse, Qwen3-0.6B)` is measurable only with a caveat. This is **falsifiable in one compile,
no device time**, and is the first hour of M1.

> `[2026-08-27]` **M1 ran that compile and this paragraph stands, with one correction and one
> extension.** Correction: the equality is required by **four** configs, not the two named here —
> `offload_config` (`offload.py:403`) and `fused_config` raise it too, for a different stated
> reason ("the head reshape around host attention"). Extension: the seq-first refusal at
> `head_dim = 128` is now measured (devq 731/733) and so is the head-first route's placement at
> both rungs (devq 786), so "measurable only with a caveat" for `coarse` is confirmed rather than
> predicted. §1b(iv) has the transcript and the re-derived family decision, which agrees with this
> paragraph's.

**(iii) The registry is legal at M = 512/1024 and holed at M = 2048.** All six Qwen3-0.6B
prefill GEMM shapes carry all three methods at M = 512/1024; at **M = 2048 five of six carry
exactly one high-precision method**, so `offload`'s shared-xclbin `drain` pin
(`pattern/offload/offload.py:420-423`) **raises at M = 2048**, and filling those holes is
blocked by the append-only writer's ownership rule (`registry_writer.py:196,246`). So
**M = 512/1024 are the only rungs where all four modes are shape-legal** (`fused` caps at 1024
rows at emb 1024). Llama-3.2-1B is the exact inverse: head_dim 64 satisfies the MHA equality and
has no FA problem, but its rows exist only at M = 2048, where `fused`'s 512-row cap makes that
cell a refusal.

**Revised estimate** (replacing "weeks-to-months per mode per family"): M1 **1 session**,
M2 **3–5**, M3 **3–4**, M4 **2–3**, M5 **1–2** — **10–15 agent-sessions ≈ 2–3 weeks per family
for all four modes**. M4 is far cheaper than §3 assumed because `shared/model_adapter.py` and
`study/run_model.py` already own the N-layer loop, embedding, LM head, KV cache and verify: M4
is a **swap, not a build**.

**Amendments to §3 adopted from M0's recommendation:**
- **M2 runs `offload` first, not `coarse`.** `offload` needs *zero* new device builders for
  RMSNorm / QK-norm / RoPE / SwiGLU — by the mode's own definition those are host numpy, and the
  reference implementations are importable from `qwen3_0_6b_cpu_helpers.py`. It therefore
  produces the first real (model, mode) cell **without touching the attention problem**. Merge
  the first-cell measurement into M2 so that phase delivers a number even if the attention spike
  goes badly.
- **Llama-3.2-1B is not deferred to "after M4".** It is the *cheaper* family for `coarse` and a
  useful control; run it alongside once M2 has a shape.
- **No decode / M = 1 column.** There is no GEMV registry or resolver at all
  (`registry_lookup.py` has no GEMV entry point), so decode-in-a-mode is a separate program, not
  a column of this one.

**A trap M1 must not walk into** (58a §3): the fault-injection negative control's
`FAULT_DELTA = 2.0` was calibrated against `val_range 0.05` synthetic draws, and the
static-weight `content_key` must stay **content-derived**. A name-keyed or in-place-mutated real
weight would make the injected run **PASS** — silently voiding the gate every later phase rests
on.

**Three named uncertainties**: the head_dim-128 attention question; whether a real-weight layer
fits under the `1e-1` hard ceiling that today sits at a 1.35× margin (`opcheck_specs.py:834-851`
— "a defect report, never a wider tolerance"); and whether M4 is genuinely a swap.

## 1b. `[2026-08-27]` M1 ran — the first hour settled F2, and the seam found a second trap

Phase M1 executed (queue item 31, branch HEAD `f0262b18`, evidence root
`programming_examples/transformer_layer/results/item31-4a-m1-20260827/`). Four things changed.

**(iv) `[2026-08-27, corrected]` The seq-first ROUTE is refused at head_dim 128. The CELL is not.**
The first draft of this section read devq 731/733 as proving `(coarse, Qwen3-0.6B)` a structural
refusal. That was wrong and the pre-commit review caught it: those compiles falsify one route, not
the cell. What is measured, and what each measurement licenses:

- `attention_config(1024, 128, 16, num_kv_heads=8, causal=True)` **PASSES** — `kv_seq_tile`
  defaults to `head_dim`, so `dk_chunks = dv_chunks = 1` and the seq-first kernel's own
  `dv_chunks == 1` assert is satisfied *formally*, exactly as 58a §4.1's first escape route said.
  The IR builds (444 lines).
- **The seq-first design at `lkp = 128` does not place.** **devq 731** (seq 1024) and **devq 733**
  (seq 512), aircc/aiecc compile-only: `'aie.tile' op allocated buffers exceeded available memory`,
  six `memref<128x128xbf16, 2 : i32>` buffers of 32768 bytes each against an L1 of four 16 KiB
  banks (`0x0-0xFFFF`); the sequential fallback needs `0x30E0B` ≈ 200 KiB. 58a inferred "three
  cannot coexist"; the measurement says six, 3.1× over. **This licenses**: the study's CURRENT
  attention composition (`builders/mha_attention.py:13-16`, `:291` — seq-first) cannot serve
  head_dim 128. Nothing more.
- The other side of that route is closed too: `kv_seq_tile = 64` raises in `attention_config`, and
  calling the seq-first `build_module` directly at `lkp=64, dk=dv=128` raises its own
  `dv_chunks == 1` assert.
- **The HEAD-FIRST route places, at this study's own rungs.** **devq 786**, compile-only:
  `attn_npu2.py` at `lkp = 64`, `lqp = 256`, `num_q_tiles = 4`, `dv_chunks = 2`, causal, 16q/8kv —
  the knob set `fa_headfirst.py:80-99` uses — **compiles clean at seq 512 AND seq 1024** (609-line
  IR both). And the registry already records that route PASSING on device at Qwen3-0.6B's own
  configuration: `kernel_registry/details/FlashAttention_bf16.md:158`, 2048×2048, dk/dv 128/128,
  16q/8kv, causal, `dv_chunks = 2`, 17.6 ms, mean_rel_L1 3.8e-2. **This licenses**: the route is
  available; a placement pass is not a numerical result, and the numerical evidence is the
  registry's, at 2048.
- **Control**: the seq-first harness at Llama-3.2-1B's point (head_dim 64, seq 2048, 32q/8kv,
  causal) compiles clean — **devq 732**. So devq 731/733 are about head_dim 128, not a broken probe.
- **The MHA equality binds in EVERY mode, not two of them.** `probe/mode_config_headdim_probe.py`
  calls all five mode configs at both families' real shapes:
  `block_config` (`block.py:296`), `runlist_config` (`runlist.py:289`), **`offload_config`
  (`offload.py:403`)** and `fused_config` all raise
  `num_heads * head_dim (16 * 128) must equal emb_dim (1024)` on Qwen3-0.6B; **all five pass
  unchanged on Llama-3.2-1B** (32 × 64 = 2048) and on the study's synthetic layer. Note the two
  distinct reasons the raises give: `block`/`fused` blame "the seq-first attention layout and the
  fused QKV weight", while `offload`/`runlist` blame "the head reshape around host attention" —
  the same constraint reached by a different route, which is why no mode escapes it.

**The corrected cell classification.**

| cell | status | why |
|---|---|---|
| `(fused, Qwen3-0.6B)` | **structural refusal** | The only route that places at head_dim 128 needs **two host transposes** (`fa_headfirst.py:140`, `:172`). `fused` is *defined* as one host submission with every intermediate device-resident (`pattern/fused/fused.py:8-16`, `:54-61`). Host transposes there are not a cost, they are a different mode. |
| `(coarse, Qwen3-0.6B)` | **NOT refused — reachable, not attempted in M1** | `coarse` is few fused kernels over one runlist per sequence; it does not forbid host mediation, and 58a §4.1 already asked for it to be recorded "measured-with-a-caveat, the transposes reported in `host_cpu_ms`". Reaching it needs `mha_out_proj`/`attention_config` taught the head-first path — real builder work M0 did not price, but bounded work, not a refusal. |
| `(offload, Qwen3-0.6B)`, `(runlist, Qwen3-0.6B)` | **reachable, and NOT free** — an earlier draft of this row said "not affected at all, head_dim-agnostic", which was false and contradicted the bullet above it | It is true that neither mode composes FlashAttention. It does not follow that either is head_dim-agnostic, and all three dependences are in their own source. **(1)** Both raise the MHA equality (`offload.py:403`, `runlist.py:289`). **(2)** Their attention GEMMs are PER HEAD and carry `head_dim` in the shape — `attn_scores` is `(seq, head_dim, seq)` and `attn_output` is `(seq, seq, head_dim)` (`offload.py:445`, `:449`) — and their tiles are INJECTED rather than registry-resolved, every one measured at **K = 64 only**, with `offload.py:256` recording "tk2=64 is FORCED: K=64 admits no other L2 tile". Qwen3-0.6B's `head_dim = 128` is therefore two new shapes needing two new tile MEASUREMENTS, exactly as 58a §2.2.3 said. **(3)** The per-head host loop slices `q`/`k`/`v` at `h * head_dim` across `[seq, num_heads * head_dim]` (`offload.py:842-844`), so `k`/`v` are assumed `num_heads` wide: **GQA is not expressible today** and Qwen3-0.6B is 16q/8kv. |

**The family decision, re-derived on the corrected facts — and it goes back to Llama-3.2-1B.**
The draft above reversed M0's choice on the strength of "`offload` is head_dim-agnostic, so M2's
first cell never meets the question". That premise is false, so the reversal is **withdrawn**.
Re-derived against §1a's amendment that **M2 runs `offload` first**:

| what M2 has to write | Qwen3-0.6B | Llama-3.2-1B |
|---|---|---|
| MHA equality, 4 config call sites | **violated** — must be decoupled in all four before ANY mode resolves | **satisfied** — no change |
| attention GEMM tiles for `offload`/`runlist` | `(seq, 128, seq)` and `(seq, seq, 128)` — **two new shapes, two new device measurements** | `(seq, 64, seq)` and `(seq, seq, 64)` — **the measured K=64 tiles apply as they are** |
| GQA in the per-head host loop | needed (16q/8kv) | needed (32q/8kv) — same work either way |
| QKV split | 2048+1024+1024 over emb 1024, unequal | 2048+512+512 over emb 2048, unequal — same work either way |
| `coarse` attention | head-first port + two host transposes | **seq-first, as the study composes it today** (devq 732 compiles it clean) |
| `fused` | refusal (host transposes contradict the mode) | refusal by construction (512-row cap at emb 2048 vs no registry rows below M = 2048; 58a §4.5) |
| registry rows | M = 512 and 1024, all methods — **a two-rung ladder** | **M = 2048 only** — a single point |

**Recommendation: Llama-3.2-1B is the first family**, which is where M0 landed in §1a(ii) and
where this document now returns. Every line above except the last favours it, and the two that
matter most are the ones this item measured: it needs no equality decoupling to resolve at all,
and its attention GEMM tiles are the ones already measured. Qwen3-0.6B's advantage is one axis —
registry coverage at two rungs — and 58a §4.3 already records that the M = 2048 holes cannot be
filled without changing the sweep writer's ownership rule, so that advantage does not compound.

**What this costs the program, stated plainly.** Llama-3.2-1B's single rung is M = 2048, so M2
delivers a POINT rather than a slope, and doc 58 §3's M5 matrix loses the sequence axis for the
first family. `(fused, Llama-3.2-1B)` is a refusal, so M3 records three cells and one refusal.
Qwen3-0.6B is not abandoned: it is the second family, it is where the two-rung ladder lives, and
its `coarse` cell is the one that needs the head-first port devq 786 just showed places.

**M2's corrected starting point**: `(offload, Llama-3.2-1B, seq 2048, prefill, one layer)`. It
needs no MHA-equality change, no new attention tile measurement, and no FlashAttention work at
all; what it does need is the unequal QKV split (`offload_config` currently resolves q/k/v/o as
one shared `[seq, emb] @ [emb, emb]` module, `offload.py:396-398`) and the GQA broadcast in the
per-head loop.

**(v) The seam landed, and its defining gate holds at host level.**
`generate_golden_reference(..., weights=...)` (`pattern/reference.py:276`, validated by
`check_weights` at `:237` against `weight_shapes` at `:217`) replaces the drawn tensor set without
moving the input activation, the draw order or the oracle; `layer_inputs(golden, names)` (`:422`)
builds any mode's ordered device-input list from that mode's own NAMES tuple, so the four
hand-written positional lists are gone and 58a §3.2(a)'s coupling cannot drift. `weights=` is
threaded through all five preparers, and each records `weight_source` in its artifact.
Two host controls, and `[2026-08-27]` the second is new because the first did not cover what it
was read as covering.

**`control/seam_sha.py` — the golden model and the input lists.** **SELF IDENTICAL (134 claims)**:
injecting the generated weights reproduces input, weights, every boundary, the output and all five
modes' ordered input lists byte for byte at four configurations, with the arrays passed through
**by identity** so `content_key_once`'s id-keyed cache still amortises. **BASELINE IDENTICAL (102
shared keys)** against a worktree of the pre-M1 commit. **What this licenses**: that
`generate_golden_reference` and `layer_inputs` behave identically. **What it does NOT**: it calls
those two DIRECTLY, so it never executes a preparer — a preparer that changed its seed or
reordered its inputs would leave it green, and would leave the current-tree A/B green too, since
both of that A/B's legs go through the same changed preparer.

**`control/preparer_contract_sha.py` — the five preparers, driven for real.**
**IDENTICAL (189 claims, 0 differing, 0 dropped, 0 new)**, devq **822** (baseline) and **827**
(comparison). It calls each mode's real `SPECS` preparer on the generated path at
512x768x3072x12, both variants, and digests the ordered `inputs` list element by element, the
`expected` output, the `inject` pair, `atol`, `record_extra` (minus the one key M1 adds), and
every entry of the mode's `block_fingerprint.json` — which `builders/block_cache.py` computes over
the built MLIR, the resolved registry config, the device kernel sources and the backend kwargs, so
a change that would produce a different ELF is a different digest here. The baseline is not the
bare commit: it is **the working tree with item 31's nine files reverted and its new file
removed**, so the concurrent items' edits are on both sides and the only difference is this
item's. `[2026-08-27]` It was re-cut and re-run once for exactly that reason — the concurrent
items moved under the first cut, so its earlier 189-claim result no longer isolated this diff.
The figure above is from the re-cut.
**What this licenses**: at these shapes and variants, all five preparers build the same tensors in
the same order, hand `opcheck` the same contract, and would compile the same ELFs. **What it does
NOT**: anything about the injected path (that is `seam_sha`'s SELF claim and the device A/B), or
about the `dispatch` closure's runtime behaviour (that is the five mode lits and the device A/B,
both of which execute it).

**And on the DEVICE**, which is the gate as §3 states it. Each mode's real `SPECS` row run twice
through `opcheck.run_spec` — once with the stock preparer, once with the same weight set injected
from outside (`control/injected_device_ab.py --leg ab`, Turbo before and after every leg):

| mode | shape | devq | boundaries compared (device / host) | returned output sha | mean_rel_L1 |
|---|---|---|---|---|---|
| `fused` | 1024x768 | **790** | 10 / 0 | `8e1ada62fe397245` | 1.756e-2 |
| `block` | 4096x768 | **791** | 10 / 0 | `9d23f9265ae46b61` | 1.663e-2 |
| `coarse` | 4096x768 | **792** | 10 / 0 | `9d23f9265ae46b61` | 1.663e-2 |
| `offload` | 4096x768 | **793** | **7 / 3** | `f17d7e85cd8c5553` | 1.381e-2 |
| `runlist` | 4096x768 | **794** | 10 / 0 | `f25540ab5556c160` | 1.733e-2 |

**`[2026-08-27, corrected]` What is compared, and on which side of the boundary.** The first
version of this A/B hashed only the value the dispatch RETURNED. For `offload` that value is
computed by host `_host_addnorm` (`offload.py:760`), so a device intermediate that differed and
was then flattened by a host normalisation would have kept the same digest — the leg could not
have established what it claimed. It now captures **every per-boundary array off the `stage_stats`
callback the preparer feeds**, which is where the DEVICE's own arrays arrive, and compares all of
them. For `offload` that is seven device-written GEMM outputs (`q`, `k`, `v`, `attn_context`,
`attn_out`, `ffn_up`, `ffn_out`) alongside the three host-computed ones; for the other four modes
every boundary is device-written by the mode's own definition. Every leg: all host inputs
identical, all dispatch buffers identical, **all boundaries bit-identical**, both runs PASS with
`stages_passed=True`, `weight_source` recorded per leg, and the delta hook present on the injected
path and absent on the generated one.

**A PASS here licenses**: "with weights injected from outside, this mode produced the same bytes
at every boundary it reports, including the ones the NPU wrote." It licenses nothing about
boundaries a mode does not report. `block` and `coarse` share a digest because `prepare_coarse`
dispatches `builders/block.py` — the seam did not make them agree, they already did.

**(vi) The named trap fired, and it is bigger than the delta.** 58a §3.2(c) predicted that
`FAULT_DELTA = 2.0` might not trip at real-weight magnitudes. Measured at 512x768x3072x12 with
real layer-0 tensors sliced to the study's shapes
(`evidence/fault-delta-512-{encoder,decoder}.log`; the generated row reproduces
`opcheck_layer.py`'s recorded table element for element, which is what licenses the method):

| variant | weights | layer output rms | ln1+2.0 max&#124;d&#124; | elements outside band |
|---|---|---|---|---|
| encoder | generated | 0.5716 | 1.526e+0 | 36855 |
| encoder | real Qwen3-0.6B | 0.5428 | 8.523e+0 | 19621 |
| encoder | real Llama-3.2-1B | 0.2163 | 3.488e+0 | 2295 |
| decoder | generated | 1.4153 | 1.842e+0 | 43948 |
| decoder | real Qwen3-0.6B | 0.3130 | 6.914e-1 | **25** |
| decoder | real Llama-3.2-1B | 0.0789 | 7.202e-2 | **0** |

The **encoder** is safe — the constant survives at both families, with margin. The **decoder** is
not, and the mechanism is not the gamma (O(1) in every source) but the output SCALE: the encoder
is post-norm so its final boundary is renormalized, while the pre-norm decoder's is the raw
residual sum, whose scale is set by weights that are 2-4× smaller than the generated `randn*0.05`
over a four-multiply chain — against an `atol` that is 4.5× wider. **At Llama's real-weight scale
the decoder's band (4.5e-1) exceeds the whole layer output's absmax (3.75e-1): every candidate
reads zero elements outside, and a device returning ZEROS would pass.** That is a defect report
about the absolute tolerance table, not about the delta, and it is the sharpest thing M1 found.
M0's uncertainty #2 asked whether a real-weight layer would EXCEED the `1e-1` ceiling; the answer
at the decoder is the opposite and worse — it falls so far below that the ceiling stops
discriminating. **M2 must settle the tolerance question before it trusts a real-weight cell**, and
a per-boundary relative bound is the obvious candidate.

**What M1 landed for it:** the delta now DERIVES rather than being hard-coded.
`derive_fault_delta` (`opcheck_layer.py:447`) doubles from `opcheck.py`'s constant until the worst
element sits `FAULT_EXCESS_MIN = 2.0` outside the band it will actually be compared at, measured
with the same oracle the preparer already owns, and **RAISES at the cap** rather than running a
vacuous control. `fault_delta_hook` (`:493`) returns `None` for the generated path — which
therefore pays nothing and is byte-identical — and a deferred callable otherwise, resolved by
`opcheck.py:331` only on a run that is actually injecting. Measured: the derivation is a no-op at
every encoder case (returns 2.0), lifts real Qwen3-0.6B decoder to a healthy margin, and needs
**32.0** for real Llama-3.2-1B decoder (0 → 1248 elements outside).

**Confirmed on device** (`control/injected_device_ab.py --leg fault`, `coarse` at
512x768x3072x12 — the shape `DECODER_STAGE_ATOL` was measured at, devq 359/360 — Turbo before and
after every leg):

| variant | weights | atol | derived delta | devq | clean | injected | stages firing | output mismatches |
|---|---|---|---|---|---|---|---|---|
| encoder | generated | 1e-1 | 2.0 | **795** | PASS | FAIL | 5/10 | 37566 |
| encoder | real Qwen3-0.6B | 1e-1 | 2.0 | **796** | PASS | FAIL | 5/10 | 19704 |
| encoder | real Llama-3.2-1B | 1e-1 | 2.0 | **797** | PASS | FAIL | 5/10 | 2288 |
| decoder | generated | 4.5e-1 | 2.0 | **798** | PASS | FAIL | 12/12 | 45539 |
| decoder | real Qwen3-0.6B | 4.5e-1 | **4.0** | **799** | PASS | FAIL | 12/12 | 212 |
| decoder | real Llama-3.2-1B | 4.5e-1 | **32.0** | **800** | PASS | FAIL | 11/12 | 1200 |

**`[2026-08-27, corrected]` What the counterfactual actually shows — the claim is smaller than the
first draft's, and it is the true one.** The first draft reported the forced-`FAULT_DELTA` run as
an injected PASS and concluded the shipped constant made the gate vacuous. It does not.
`opcheck.py`'s verdict is a CONJUNCTION of the final-output tolerance check AND every per-boundary
stage comparison (`opcheck.py:360-370`), and the first draft read only the final-output
`n_mismatch`. Re-run with the classification taken from `passed` and `stages_passed` together:

| leg | devq | final-output check | stages firing | conjunction | shipped `_negative_control_verdict` | classification |
|---|---|---|---|---|---|---|
| real Llama-3.2-1B decoder, δ forced to **2.0** | **801** | **VACUOUS** — 0 mismatches, excess 0.16× | **5 of 12** (`ln_in` 186×, `q` 18×, `k` 20×, `v` 3.0×, `attn_context` 1.6×) | **rejects** (`passed=False`) | **RED** (rc=1) | `output_vacuous_conjunction_rejects` |
| real Qwen3-0.6B decoder, δ forced to **2.0** | **802** | fires, but by **1.49×** — 25 elements | 12 of 12 | rejects | GREEN | `discriminates` |
| real Llama-3.2-1B decoder, δ forced to **0.005** | **803** | vacuous, excess 0.01× | **0 of 12** | **PASSES** (`passed=True`, `stages_passed=True`) | RED | `fully_vacuous` |

So the honest statement, in three parts:

1. **The conjunctive gate does NOT go vacuous at the shipped constant.** At δ = 2.0 with real
   Llama-3.2-1B decoder weights the stage comparisons catch the fault — `ln_in`, the boundary
   directly downstream of the perturbed `ln1_weight` and carrying the table's tightest `atol`
   (3.5e-2), sits **186× outside its band**. The stage conjunction is the real protection here,
   not the final-output check.
2. **What DOES go vacuous is the final-output comparison — the one component
   `_negative_control_verdict` requires to be the rejecter.** It treats a FAIL with
   `n_mismatch == 0` as a problem ("the tolerance check is not what rejected it"), so the shipped
   control turns **RED**, not silently green. That is exactly 58a §3.2(c)'s prediction — "an
   injected real-weight run that happens to round back into the band turns a working gate into a
   red one" — arriving as written.
3. **`derive_fault_delta` is therefore a TIGHTENING of one component, not a rescue from a vacuous
   control.** It restores the property the negative control asserts: that the compared tensor's own
   tolerance check is what rejects the run. Its value is a gate that stays green for the right
   reason rather than red for a confusing one — worth having, and a smaller claim than the first
   draft made.

The gate *can* be made fully vacuous, and the bound is measured rather than argued: at δ = 0.005
(devq 803) the injected run passes every stage and the output check, `passed=True`. `ln_in`'s
excess is linear in δ away from the device's own error floor, so the threshold lies between those
two measured points, near δ ≈ 0.011 by extrapolation from the δ = 2.0 point — **400× below the
shipped constant**, i.e. at a perturbation nobody would call a fault injection. **Scope**: this
protection exists because all four modes RECORD per-boundary stages. A preparer that recorded none
would have the final-output check as its whole gate, and there δ = 2.0 at this weight scale would
be genuinely vacuous — which is the condition to check before adding one.

Two further data points fall out of the clean legs, and both matter for M2. Real weights are
**easier** than the generated draws on this comparison, not harder: `atol_required` runs 2.770e-2
(Qwen3) and 3.562e-3 (Llama) on the encoder against the generated 5.820e-2, and 4.814e-2 /
3.594e-3 on the decoder against 3.038e-1. So M0's uncertainty #2 — "does a real-weight layer fit
under the `1e-1` hard ceiling" — is answered YES with room at these shapes. The problem is the
opposite one, stated above: the margin is so large that an absolute band on the final tensor stops
discriminating, and the per-boundary table is what carries the gate.

**(vii) A WALL, pre-existing and outside M1 — RAISED, THEN CLEARED THE SAME DAY.**
Item 28's fail-closed herd-rows guard (`llms/shared/infra/dispatch.py:737`, called from
`cache.py:513`, landed in `f0262b18`) refuses every whole-layer GEMM artifact the study builds:

```
ValueError: blk_qkv_proj_4096x768: this module has a multi-row air.herd
(g_herd_0=4, g_herd_0=4, g_herd_0=4) and is being compiled WITHOUT a lock-race fix.
```

All five of `run_npu2_{block,coarse,offload,runlist,fused}_peano.lit` fail this way in 86 s,
before any device work (**devq 743**). It is **not M1's**: the same `python3 opcheck.py --operator
fused` run inside a detached worktree of `f0262b18`, with nothing from this item on the path,
fails identically (**devq 745**, exit 1). The guard's own note says "a guard that fails closed
refuses every call site it does not know about", and item 28's green sweep covered `llms/`
shipped models only — while the study's artifacts are the *same* 8×4 GEMM herds that note names
as having "never hung", and have been the branch's recorded mode measurements for months. The fix
is one entry family in `HERD_ROWS_MEASURED_GREEN` (`blk_*`, `rl_gemm_*`, `off_gemm_*`,
`fused_*`), which is in `llms/shared/` and therefore an operator decision, not M1's.

**CLEARED, THEN SUPERSEDED — read §1b(ix) with this paragraph, not instead of it.** Item 28's
**round 3** re-swept the study tier by measurement (22 artifacts at rows = 4, 7 at rows = 1,
nothing undecidable) and green-listed the study's artifact prefixes. With **both** that fix and
M1's seam in the tree, all five of `run_npu2_{block,coarse,offload,runlist,fused}_peano.lit`
PASS — **devq 762, 5/5, Turbo before and after**, and M1's device half above was measured on that
tree. That green list **no longer exists**: item 28's round 4 deleted it and replaced refusal with
enforcement, and three of those five gates now fault on the device. §1b(ix) is that story; the
fix described in the paragraph above is therefore the fix that WAS applied, not the one now in the
tree. Two things are worth carrying
forward from the episode: the guard's blast radius is exactly the
`KernelCache.compile_and_cache` path, so per-operator gates (which go through
`XRTRunner.run_test`) were never affected and could not have detected it; and a fail-closed guard
swept over one tier silently red-lines every other tier that uses the same call, which is a
cross-tier gate-coverage question this branch does not currently answer.

**(viii) `[2026-08-27]` What each of M1's gates compares, and what a pass licenses.** The
pre-commit review's five blocking findings were of one family: three of them were gates that
reported success while measuring something other than what they claimed. That family — not any
single bug — is the thing to carry forward, so every M1 gate is written out here with its scope,
and each control now states the same thing in its own docstring.

| gate | compares | side of the host/device boundary | a pass licenses | it does NOT license |
|---|---|---|---|---|
| `control/seam_sha.py` SELF | golden-model tensors, every boundary, all five modes' ordered input lists, at 4 configurations | host only, no compile | `generate_golden_reference` + `layer_inputs` are identical with weights injected | anything about the preparers — it calls neither of them |
| `control/seam_sha.py` BASELINE | the same digests vs a pre-M1 worktree | host only | the same two functions are unchanged | same exclusion |
| `control/preparer_contract_sha.py` | `inputs` / `expected` / `inject` / `atol` / `record_extra` / every ELF fingerprint, for 5 preparers × 2 variants | host; compiles for real, does not dispatch | the generated path of all five preparers builds the same tensors, hands the same contract, and would compile the same ELFs | the injected path; the `dispatch` closure at run time; shapes not run |
| `injected_device_ab.py --leg ab` | every host input, every dispatch buffer, **every per-boundary array** the dispatch produced, and the returned output | **mixed, and the split is per mode.** The boundary arrays are read off the `stage_stats` callback, which is where the NPU's arrays arrive for a device-computed boundary — but not every reported boundary is device-computed. `block`/`coarse`/`runlist`/`fused`: **10 of 10 device**. `offload`: **7 device** (`q`, `k`, `v`, `attn_context`, `attn_out`, `ffn_up`, `ffn_out`) / **3 host** (`hidden`, `ffn_gelu`, `output` — `_host_addnorm` at `offload.py:760`, `_host_gelu` at `:858`). The driver prints the split per leg and marks each line `[dev]` or `[host]`. | that mode produced the same bytes at every boundary it reports; for the four device-resident modes that is every boundary, for `offload` it is the seven the NPU wrote plus three host ones | that `offload`'s three host boundaries say anything about the NPU; boundaries a mode does not report; modes not run |
| `injected_device_ab.py --leg fault` | the CONJUNCTION `opcheck` actually applies — final-output tolerance **and** every stage comparison — classified from `passed` and `stages_passed` together | device | the stated classification of what rejected the injected run | that the final-output number alone decides anything; it does not |
| `run_npu2_*_peano.lit` ×5 | the shipped gates, clean + fault, unchanged | device | the modes still pass their own gates on this tree | nothing about the injected path — the lits do not use it |

Two gate defects were found and fixed inside this item rather than by the review, and they belong
in the same list because they are the same shape. `control/run_preparer_sha.sh` ended with
`echo "... $?"`, so the script's own exit status was the `echo`'s and **devq recorded a failed run
as `done 0`** — a gate that cannot go red. And the forced-delta counterfactual first shared its
`(operator, shape_key)` with the derived-delta leg it counterfactuals, so `_write_result`
overwrote one experiment with the other; renaming the files was not enough, because the key is
recorded INSIDE the artifact, and the fix is the key.

**(ix) `[2026-08-27]` The herd-rows guard's FOURTH round hangs three of the five mode gates, and
it is not M1's.** §1b(vii) records the guard's first round refusing to compile the study's
artifacts, and item 28 clearing it: **devq 762, all five `run_npu2_*_peano.lit` PASS**, 12:05, on a
tree carrying item 28's round-3 prefix green-list AND M1's seam. Re-running the same five gates at
14:11 on the same working tree gave **3 of 5 FAILED with a device fault** — `ERT_CMD_STATE_TIMEOUT`
at runlist submission 0, entry 0, on `blk_qkv_proj_4096x768` (`block`, `coarse`) and
`fused_qkv_proj_1024x768` (`fused`); `offload` and `runlist` still pass. **devq 809.**

**Three-arm attribution, all at the same operator and shape:**

| arm | tree | devq | result |
|---|---|---|---|
| (a) | working tree, item 31 present | **812** | device hang, `blk_qkv_proj_4096x768`, `ERT_CMD_STATE_TIMEOUT` |
| (b) | working tree, **item 31's nine files reverted and its new file removed** | **813** | **identical device hang**, same kernel, same error |
| (c) | pristine `f0262b18` | **815** | **never reaches the device** — the round-1 guard refuses the compile (`...is being compiled WITHOUT a lock-race fix`), which is §1b(vii)'s original wall. Not a device arm. |

Arm (b) settles it: with item 31 entirely absent the hang is unchanged. The ELF evidence agrees
independently — `control/preparer_contract_sha.py` (devq 822/827) compared the artifact
fingerprints of the two hanging kernels across exactly that revert and found them **identical**,
and that digest covers the built MLIR, the resolved config, the device kernel sources and the
backend kwargs, i.e. everything that determines the binary.

**The mechanism, from the diff rather than from a bisect.** Between devq 762 and devq 809 item 28
landed a **fourth** round: `check_herd_rows_lock_fix` (which REFUSED an unexempted multi-row herd)
was replaced by a function that **injected `use_lock_race_condition_fix=True` into the backend
kwargs of every multi-row herd** inside `KernelCache.compile_and_cache` (`cache.py:506`, `:516`),
and the exemption list was deleted entirely. That **changes the ELF** for all 22 study artifacts.
The study's block/fused QKV split-cast had been running green without the fix for months; with it
forced, it hangs. `off_gemm_*` and `rl_gemm_*` took the same fix and did not hang, so the flag was
never universally fatal — it was fatal for this artifact form.

> `[2026-08-27]` **ADDENDUM — the rule above was WITHDRAWN in item 28's round 6, and the fault is
> closed at its source.** The paragraph stands as history: round 4's inject-everywhere rule really
> is what faulted the block/fused QKV split-cast form, and the three-arm bisect is what
> established it. Three of its citations no longer resolve, and are marked rather than silently
> repaired:
>
> - `ensure_lock_fix_for_multi_row` **no longer exists**. The function is now
>   `ensure_lock_fix_for_marked_herds` (`dispatch.py:695`); the `:675` in the original text is
>   stale.
> - The `backend_presets.py` note this passage originally quoted — *"There is no exemption list …
>   the fix is a cost paid by every multi-row herd rather than a property tested for"* — has been
>   **deleted from that file**; `grep 'exemption list' backend_presets.py` returns nothing today.
>   It is reproduced here **as a quote from history**, not as text a reader can check against its
>   source.
> - "the operator routes it", below, **happened and resolved**.
>
> The rule today: the flag is supplied **iff** `matvec.py` stamped `air.lock_race_fix_required`
> (`LOCK_RACE_FIX_REQUIRED_ATTR`, `dispatch.py:615`; stamped at `matvec.py:358`) on a herd above
> one row. A module carrying no mark is returned untouched, and the three shipped `8 × 4` kernels
> — `o_ffn_qwen`, `rms_qkv_qknorm_rope`, `o_gemv_ffn` — come back `fix=None`, verified on the real
> modules. **The study's block/fused QKV split-cast is a GEMM herd and carries no mark, so it no
> longer takes the flag**, and the ELF it builds is again the one devq 762 measured 5/5 green on.
>
> **Does this change M1's own evidence or its timeline argument? No — and the reason is the
> timeline itself.** Every device leg M1 rests on ran 13:19-13:33, before round 4 landed at 13:40,
> so all of them were already taken on ELFs built WITHOUT the flag — the same ELF round 6
> restores. The timeline paragraph below is unchanged in substance: it explains why devq 762 read
> 5/5 and devq 809 read 3/5 with M1 identical between them, and round 6 simply ends that bracket.
> What changes is availability, not validity: **devq 809's three faulting gates are expected to
> pass again and are now re-runnable**, where before they were blocked. No device work was re-run
> for this addendum.
>
> **`[2026-08-27, MEASURED]` The expectation above has since been tested rather than left as an
> expectation: devq 828 ran all five mode lits on the final combined tree and they are 5/5 PASS**
> — `block`, `coarse` and `fused`, the three that faulted at devq 809, among them. Turbo before
> and after, five legs counted rather than an exit code trusted, and the guard's shape recorded in
> the log at run time (`mark=air.lock_race_fix_required`, no exemption registry, no
> `with_herd_rows`). So the fault is closed **by measurement on the tree that ships**, not by
> argument from the rule.

**Not debugged further and not worked around**: it is a change in `llms/shared/`, outside this
item, and the operator routes it. **`[2026-08-27]` That routing happened: item 28's round 6
withdrew the rule and the fault is closed at its source — see the addendum above.**

**Where M1's own device evidence sits on that timeline, stated so a reader can check it.**
`dispatch.py` changed at **13:40**. Every device leg M1 rests on ran **before** it: the A/B
(devq 790-794, 13:19-13:28) and all nine negative-control and counterfactual legs
(devq 795-803, 13:29-13:33). So those numbers were taken on the pre-round-4 ELF. That does not
weaken them — every one is an A-vs-B comparison in which BOTH arms use whatever ELF the tree
builds, so the seam's claim is ELF-independent by construction — but it is why the five mode gates
read 5/5 at devq 762 and 3/5 at devq 809 with M1 unchanged between them. The preparer-contract
control is the one piece deliberately re-run on the CURRENT tree, round 4 included, because its
whole subject is the ELF the tree would build.

**Revised estimate.** M1 came in at 1 session as 58a predicted. **M2 stays at 3-5 sessions**, on
**Llama-3.2-1B** (§1b(iv); the intermediate draft that reversed to Qwen3-0.6B is withdrawn, and
the reversal's premise with it). Llama-3.2-1B needs no MHA-equality decoupling and reuses the
already-measured K = 64 attention tiles, which pays for the M = 2048-only registry constraint it
adds and closes 58a §5's only unbounded item — `coarse` attention — as an ordinary seq-first port
rather than an open cost. The herd-rows wall cost hours rather than a session. **Two things are
NOT priced into that number**: §1b(vi)'s tolerance question, worth one session if M2 has to
replace the absolute per-boundary table with a relative bound before it can trust a real-weight
cell; and §1b(ix)'s device hang, which is outside this item and must be resolved before any phase
can measure `block`, `coarse` or `fused` on hardware at all.

## 2. The cost-halving fact (as first drafted — see §1a(i) for the correction)

`llms/shared/` is off limits to **edit** — "editing it puts all ten shipped deployments' `make
verify` on the line" (`builders/gemm_spec.py:49-53`) — but the study **already imports it and
does not own it** (`builders/block_cache.py:29`, `builders/elementwise_add.py:54-55`,
`study/test_ubatch_prefill.py:189`). So gap 2 is **not** a reimplementation of RMSNorm / SwiGLU
/ RoPE / QK-norm; it is a re-*assembly* of shipped leaves at four different execution
boundaries. The open question phase 1 must answer precisely: which shipped pieces are exposed
at leaf granularity (`gemm_builder` is; the fused stage builders like
`rms_qkv_qknorm_rope_multi` are whole-ELF and are the wrong granularity), and what must be
written because only a fused form exists.

## 3. Phases and gates

Each phase lands before the next starts, prediction-first, with its own Codex round — the
discipline items 13–20 used.

| Phase | Deliverable | Gate |
|---|---|---|
| **M0 — scope** | Read-only: the leaf-granularity inventory of §2, per mode the exact list of what must be written, and a revised estimate per phase. No code. | A written scope with file:line citations and a per-phase estimate the operator can accept or stop on |
| **M1 — the seam** | Weights injection into `prepare_layer_dispatch` and all four patterns, without disturbing the generated-weights path or the fault-injection contract | Every mode runs its EXISTING synthetic layer with weights injected from outside, **bit-identical** to the generated-weight run; existing lits green |
| **M2 — one layer, one mode** | A real Qwen3-0.6B decoder layer through `coarse` (simplest, fastest, and its registry rows exist) with real weights | Layer output matches the shipped driver's layer within the bf16 family bound, on device, with the re-execution gate shape; `quant_*`/`dtype` producer landed (gap 6) |
| **M3 — one layer, four modes** | The same layer through `offload`, `runlist`, `fused` | Same correctness gate per mode; the structural bounds of gap 5 recorded as refusals with their reasons, not worked around |
| **M4 — the model** | N-layer loop, embedding, LM head, KV cache, sampling in the mode harness | A whole Qwen3-0.6B forward through ≥1 mode, gated by the production top-5 verify against HF |
| **M5 — the matrix** | The measurement the program exists for: model × mode × dtype, on real weights | Two walks, `compare_roots` OK, verify per artifact set, every cell either measured or a recorded refusal |

~~**Second family (Llama-1B, GQA + no QK-norm) only after M4**, and only if M0's estimate
holds.~~ **Superseded by §1a**: Llama-1B is the cheaper family for `coarse` and runs alongside
once M2 has a shape. The M2 row's "coarse (simplest, fastest…)" is likewise superseded — M2
runs `offload` first.

`[2026-08-27]` **§1b amends this row, and §1a(ii)'s family choice STANDS.** An intermediate M1
draft reversed the family to Qwen3-0.6B on the premise that `offload` and `runlist` are
head_dim-agnostic; §1b(iv) measured that premise false — all five mode configs raise the MHA
equality on Qwen3-0.6B, and `offload`/`runlist` additionally need two new attention-GEMM tile
measurements at `head_dim = 128` and a GQA broadcast they cannot express today. The reversal is
withdrawn. **M2 and M3 run Llama-3.2-1B first**, starting at
`(offload, Llama-3.2-1B, seq 2048, prefill, one layer)`; Qwen3-0.6B is the second family and the
one with the two-rung registry ladder. `(fused, ·)` is a refusal for BOTH families, for different
reasons (§1b(iv) and 58a §4.5). `(coarse, Qwen3-0.6B)` is reachable through the head-first
attention route, which devq 786 shows places at both of its rungs — M3's work, not M2's.

**On hardware readiness**: §1b(ix) records three of the five mode gates faulting with
`ERT_CMD_STATE_TIMEOUT` (devq 809) for a reason outside this item — item 28's round-4
inject-everywhere rule, which changed the ELF. `[2026-08-27]` **Round 6 withdrew that rule and the
fault is closed at its source** (§1b(ix)'s addendum): the study's GEMM herds carry no
`air.lock_race_fix_required` mark and so no longer take the flag, restoring the ELF devq 762
measured 5/5 green on. `[2026-08-27]` **The five gates have since been re-run and are 5/5
PASS on the tree that ships** (devq 828, all five legs green, Turbo before and after) — so this
is now a measurement rather than an inference from the rule, and **M2 no longer needs to spend
its first act on it.** What M2 does still owe here is the hardware-readiness question this
paragraph exists for, which the mode gates do not answer.

## 4. What this program does NOT buy

The four modes are boundary shapes for **one layer**. M4/M5 measure a model *whose layers are
executed in a mode*, which is not the same as a model-level execution boundary (the LM head,
embedding and sampling sit outside the mode in every phase). Doc 56's H3 stages remain the
model-level boundary work. Anyone reading a mode × model number must read it as "the layer loop
ran this way", and the rows must say so.

## 5. Risks, named now

- **Divergence.** M2+ creates a second assembly of every model's layer. If the shipped builders
  change, the study's assembly rots. Mitigation: import leaves, never copy them; pin the
  imported symbols in a host test.
- **Registry raises.** Any shape without a measured row stops a build (gap 4). Qwen3-0.6B is
  covered at M 512/1024/2048; every other family needs a sweep first.
- **Modes that cannot.** `fused` at emb 2048 caps at 512 tokens; some (model, mode) cells will
  be structural refusals. Those are results, not failures, and M3/M5 must record them as such.
- **Cost.** The inventory's first estimate was weeks-to-months per mode per family. §2 cuts
  gap 2 substantially; M0 exists to replace that guess with a number before the expensive
  phases start.
