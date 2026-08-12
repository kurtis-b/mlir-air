# Goal 1 — SOTA models via sliding-window / local-global attention: scoping investigation

`[2026-08-12]` Read-only survey against `exper/transformer-layer-execution-studies` tip `b777517b`.
Nothing was built, changed, or run on the device. No latency claim appears in this document.
Every figure carries the file it came from; inferences are marked **INFERENCE**.

Spec under review: [`docs/plans/transformer-layer-execution-studies/11-goal-sota-sliding-window.md`](../../../../home/cj/mlir-air/docs/plans/transformer-layer-execution-studies/11-goal-sota-sliding-window.md).
Queue row **12** in the README says the choice has been offered and not taken. This document is
the material for taking it.

Absolute paths throughout. `install-xrt/` staleness (2026-08-07 vs `build-xrt` 2026-08-11) does
not bear on anything here: no figure below is derived from a compiled artifact.

---

## 0. The three corrections to the spec, first

The plan directory's convention is that a falsified claim is retracted in place rather than
deleted. Three of doc 11's claims do not survive contact with the code.

**(a) "there is in-flight work to build on" — the in-flight work has never executed Gemma 3 on the
NPU, by its own record.** `exper/gemma3-dataflow`'s own results table
(`programming_examples/gemma3/docs/results.md`, on that branch) classifies both NPU paper cells as
`REAL_MODEL_EXECUTION_NOT_IMPLEMENTED`:

| cell | backend | local value | classification |
|---|---|---|---|
| `gemma3_1b_npu_prefill_1k_blocked_initial.json` | NPU | **blocked** | `REAL_MODEL_EXECUTION_NOT_IMPLEMENTED` |
| `gemma3_1b_npu_decode_1k_blocked_initial.json` | NPU | **blocked** | `REAL_MODEL_EXECUTION_NOT_IMPLEMENTED` |

The CPU and iGPU cells on the same table did produce numbers. The branch tip commit `53348bfe`
(2026-06-23) states its own decode variants are "not promoted as paper-ready evidence". And the
windowed kernel it carries, `programming_examples/gemma3/aie_kernels/flow_attention.cc`, opens with
"Correctness-first ... intentionally simple and is meant to be replaced by a multi-CT scheduled
version after validation" — it is a scalar triple-nested loop that recomputes every score **three
times** (once for the row max, once for the denominator, once per output element) with a bit-hack
`fast_exp_approx`. It is a reference, not a deployable kernel.

So the prior art is real and useful — see §2.4, it is the best available *design input* — but doc 11's
rationale that sliding-window is "the cheapest next capability **because** there is in-flight work
to build on" overstates what that work delivers.

**(b) The two skill trees are not "byte-near identical copies", and the `llama_kernel_builder`
staleness is 11 of 15 files in ONE tree, not 15 in both.** Measured:

- `.claude/skills/` — 11 of 15 `SKILL.md` files reference `programming_examples/llms/llama_kernel_builder/`,
  29 occurrences. The four clean ones are `opt-layout-alignment`, `phase-3-full-model-validation`,
  `phase-6-finalize-and-learn`, `phase-7-independent-evaluator`.
- `.codex/skills/` — **zero** references. That tree is already migrated to
  `programming_examples/llms/shared/infra/`.
- All 15 file pairs **differ** (`cmp` DIFFERs on every one), in two ways only: the YAML frontmatter
  `description` is rewritten for Codex, and the path migration above. The *gate text itself is
  identical* across trees.

Consequence for work item 2: the **gate edit** is a both-trees edit; the **staleness fix** is a
`.claude`-only edit. Doc 11 and [13 §Pre-existing issues](../../../../home/cj/mlir-air/docs/plans/transformer-layer-execution-studies/13-verification-and-acceptance.md)
both say "all 15 ... in both", and that is wrong in both halves.

The `test -d` claim **is** correct and is worse than doc 11 says — it is broken *today*:
`.claude/skills/deploy-new-llm/SKILL.md:97-101` runs
`test -d programming_examples/llms/llama_kernel_builder && ... && echo OK || echo MISSING`,
lines 103-105 mark it required and instruct a halt, and the directory does not exist
(`programming_examples/llms/shared/` replaced it in `2f20c2fa`). The `.codex` copy already tests
`programming_examples/llms/shared/infra`.

**(c) The four execution modes cannot measure a windowed model, because they have never run a
CAUSAL one.** Work item 8 ("Measure the new model across all four execution modes via the Phase F
harness") is written as if causality were a parameter of the harness. It is not — see §3.1. Every
mode hardcodes the non-causal encoder variant at a literal.

---

## 1. What the goal actually requires, mechanically

### 1.1 The architecture class

Doc 11 scopes to Gemma 3 and Mistral-style local-global. Concretely a windowed model changes three
things relative to the ten shipped models (all full-causal — Llama-3.2 ×2, SmolLM2, Qwen2.5 ×3,
Qwen3 ×3, plus the int4 Llama):

1. **The mask is two-sided.** Full causal keeps `k ≤ q`; a window keeps `q − W < k ≤ q`. It is a
   *band*, not a triangle.
2. **The pattern is per layer.** Gemma 3 interleaves local and global. The gemma3 branch records
   the pattern it detected as `"5-local-1-global"`
   (`programming_examples/gemma3/gemma3/npu/preflight.py:116`, on that branch) and carries
   `sliding_window=512` as its default at five sites
   (`npu/argument_binding.py:352`, `npu/bo_plan.py:326`, `npu/buffer_binding.py:320`,
   `npu/model_runner.py:498`, `npu/wiring.py:483`).
   *The upstream HF config was not verified locally — the Gemma-3 weights are not present (§4.6) —
   so treat 512 and 5:1 as **that branch's recorded values**, not as a checked config read.*
3. **The KV cache can be bounded** at `W` for local layers. The gemma3 branch already sizes it that
   way host-side: `_local_cache_tokens()` returns `min(decode_context, sliding_window)`
   (`npu/bo_plan.py:126-129`), selected per layer by `attention_kind == "global_full"`
   (`:153`). This is an optimization, not a correctness requirement — see §1.4.

### 1.2 At the kernel level

**One C++ function is the entire causality implementation for every shipped model.**
`programming_examples/flash_attention/kernel_fusion_based/attn_npu2.cc:634-701`:

```c
void apply_causal_mask(bfloat16 *g, int32_t q_block_idx, int32_t kv_block_idx)
```

Three branches on **block indices**:

| branch | line | behaviour |
|---|---|---|
| `kv_block_idx > q_block_idx` | `:640-650` | broadcast-fill the whole `lqp*lkp` tile with bf16 `-inf` (`0xff80`), return |
| `kv_block_idx < q_block_idx` | `:652-655` | **return immediately, touching nothing** |
| equal (diagonal) | `:657-700` | per row, `int mask_start = row + 1;` (`:668`), then 8-wide `aie::select` over the column-major 8×8 tiled `G` layout |

Both the head-first and seq-first builders call it with identical arguments
(`attn_npu2.py:731-751` and `attn_npu2_seqfirst.py:729-749` are line-for-line the same) and link the
same `attn_npu2.o`. So a banded variant of this one function propagates to all ten models with no
per-model kernel work. That is the single highest-leverage edit in the tree for this goal.

**Absolute positions are already derivable inside the kernel.** Doc 11 lists "absolute-position
handling and window offsets" as work. The information is present:

- `kv_block = ty * chunks_per_stage + chunk_iter` (`attn_npu2.py:733-741`) — a **global** KV block
  index in units of `lkp`.
- `q_block = counter_buf[0] + tx` (`:742-746`) — a **global** Q block index in units of `lqp`,
  where `counter_buf` is a persistent 3-or-4-element i32 L1 scratch acting as a software program
  counter (`attn_npu2.py:519-524`, boot-initialized `:643-662`, advanced `+NQ` per launch iteration
  `:1013-1046`).
- `lqp` and `lkp` are compile-time `#define`s (`attn_npu2.cc:23-29`, set by
  `-Dlqp=... -Dlkp=...` at `llms/shared/infra/external_kernels.py:276-277`).

Therefore `q_abs = q_block_idx * lqp + row` and `k_abs = kv_block_idx * lkp + col` are computable
in-kernel today. **What is missing is only `W`.** The banded function is a four-or-five-way branch
on the block distance `d = q_block_idx − kv_block_idx`:

| `d` | today | banded |
|---|---|---|
| `< 0` | all `-inf` | unchanged |
| `0` | diagonal partial | unchanged (window includes the diagonal) |
| `0 < d < floor(W/lkp)` | untouched early-return | unchanged |
| `d ≈ W/lkp` | untouched early-return | **new**: a second partial edge, needing `q_abs`/`k_abs`, not `row + 1` |
| `d > ceil(W/lkp)` | untouched early-return | **new**: full `-inf` fill |

Note the direction of the cost: today's `d > 0` case is a bare `return`. A band converts many of
those into full-tile `-inf` writes — **more** stores, not fewer.

### 1.3 At the builder level

`causal` is a plain Python build kwarg, default `False` (`attn_npu2.py:67`, documented `:84`;
identically in `attn_npu2_seqfirst.py:67`). It threads through as (i) shape asserts, (ii) a
conditional `external_func` declaration of `apply_causal_mask` (`attn_npu2.py:219-220`), (iii) the
counter buffer, (iv) the call site.

**There is no host scalar argument path.** The launch signature is
`attention_bf16(q_in, k_in, v_in, gp_out)` (`attn_npu2.py:257`). A window size must therefore be
baked by the builder as a `ConstantOp` (or a `-D`), i.e. **per ELF**. Consequence for local-global:
alternating layers need **two compiled attention modules per model**, not a runtime toggle. The
gemma3 branch reached the same conclusion independently — its `flow_attention.cc` takes
`WINDOW_LEN`, `CAUSAL` and `QUERY_BASE` as `#ifndef`/`#define` compile-time macros (`:11-32`) and
selects the band with `if constexpr (WINDOW_LEN > 0) start = (end > WINDOW_LEN) ? (end - WINDOW_LEN) : 0;`
(`:62-63`).

Two builder-level pins a banded variant inherits:

- `causal ⟹ lq == lk` and `lqp // num_q_tiles == lkp` (`attn_npu2.py:106-111`; the seq-first twin
  at `:115-120`). Square Q-tile/K-chunk alignment is what makes block-index comparison legal.
- `lkp = 64` is effectively fixed by the L1 budget plus that pin
  (`kernel_registry/details/FlashAttention_bf16.md:119, :124`). **So a window is quantized to
  multiples of 64** unless the partial-select path at `attn_npu2.cc:657-700` is generalized.
  512 is a multiple of 64, so Gemma 3's recorded window is fine as-is.

### 1.4 At the decode level — much cheaper than doc 11 assumes

Doc 11's work item 4 asks for "eviction or ring-buffer indexing so the cache stops growing" plus
"RoPE position correctness under that indexing — positions are absolute, cache slots are not".

**All ten shipped models run decode attention on the host, in NumPy.** Every `<model>_decode.py`
calls `decode_attention_cpu` — `llama32_1b_decode.py:98` (reused by `llama32_3b_decode.py:68`),
`qwen3_0_6b_decode.py:285`, `qwen3_1_7b:284`, `qwen3_4b:315`, `qwen25_0_5b:270`, `qwen25_1_5b:270`,
`qwen25_3b:318`, `llama32_1b_int4:135`. `qwen3_0_6b_decode.py:32` states it outright: "Decode
attention is CPU (decode_attention_cpu), matching llama."

The host cache is uniformly `(n_kv_heads, max_seq, head_dim)` with an integer `current_pos`, sliced
`k_cache[:, :seq_len, :]` where `seq_len = current_pos + 1` (`qwen3_0_6b_decode.py:292-304`) and
written at `k_cache_layer[:, current_pos, :]` (`:389`). Absolute indexing, fixed max-seq, **no ring
and no eviction anywhere**.

Consequences:

- **Windowed decode correctness is a slice change** — `k_cache[:, max(0, pos-W+1) : pos+1, :]` per
  model. No kernel work, and **the RoPE-position problem doc 11 raises does not arise**, because
  slots stay absolute; RoPE is applied at the absolute position before the write.
- The ring buffer is only needed to bound *memory*. It is an optimization, and taking it is what
  *creates* the RoPE/slot-aliasing problem doc 11 describes. **Recommend not taking it in a first
  increment.**
- (The standalone NPU decode kernel `programming_examples/attention_decode/attn_decode_npu2.py`
  does exist, with `kv_cache_size = [NKV, seq_len, n]` at `:105` and a **compile-time** `pos_host`
  baked into the ELF (`:76`, loop bounds `:360`, `:370`, writeback offset `:1456`, `:1464`). It
  masks by pre-filling the attention buffer with `-99.0` (`:851-856`) and only writing `[0, pos)`.
  No shipped model uses it.)

### 1.5 At the mask level in the study tree — a second, unconnected mechanism

`programming_examples/transformer_layer/` has its own masking primitive, and it is not the
FlashAttention one. There is no `builders/causal_mask.py`; `causal_mask` is a keyword on the
elementwise-add builder (`builders/elementwise_add.py:18-24` records why — iron's `AIECausalMask`
had no device design, so it collapsed into a keyword).

```python
def causal_mask_bias(seq_len, np_dtype=bfloat16, fill=CAUSAL_MASK_FILL):   # elementwise_add.py:111
    mask = torch.full((seq_len, seq_len), float(fill), dtype=torch.float32)
    mask = torch.triu(mask, diagonal=1)                                     # :124
    return mask.numpy().astype(np_dtype)
```

- Host-computed, `[seq, seq]` bf16, staged in L3 as the `b` operand of an elementwise add.
- `CAUSAL_MASK_FILL = -10000.0` (`:61`), **not** `-inf` — because it is *added*, and bf16 `-inf`
  in an add goes NaN (`:27-31`). The device FlashAttention path uses true `-inf` (`0xff80`,
  `attn_npu2.cc:636`). Two different mask conventions in one repository.
- `build_elementwise_add_module(..., causal_mask=True)` emits **byte-identical MLIR** either way;
  the flag only asserts `rows == cols` (`:88-92`, stated at `:13-16`).
- It is gated standalone (`opcheck_specs.py:218` at `512x512`, `run_npu2_causal_mask_peano.lit`,
  `Makefile` target `check-causal-mask`) and **consumed by no attention path anywhere.**
  `opcheck_specs.py:114-120` explains why there is no `baseline_768` row: the encoder variant uses
  an all-ones mask.

Banding it is a one-line change (`triu(k=1)` → `triu(k=1) + tril(k=-W)`). But it is a disconnected
primitive, and wiring it *into* an attention path has a cost the study already measured — §3.2.

---

## 2. What supports it, what blocks it

### 2.1 Generalizes cheaply (host / reference side)

| Location | Change |
|---|---|
| `flash_attention/kernel_fusion_based/attn_npu2.py:1341-1343` | harness oracle: `np.triu(..., k=1) → -1e9` becomes a band |
| `transformer_layer/builders/mha_attention.py:364-366` | `chunked_attention_reference`: `np.where(kv_positions > q_positions, -inf, ...)` gains a lower bound. **Caveat at `:368-369`** — it assumes no row is ever wholly `-inf`; true for causal and for a window that includes the diagonal, false otherwise |
| `transformer_layer/pattern/blocked_attention.py:171-176` | `masked_fill(kv_positions > q_positions, -inf)` — same |
| each `llms/<model>/<model>_cpu_helpers.py` | the FP32 reference mask, e.g. `llama32_1b_cpu_helpers.py:77`, `qwen3_0_6b_cpu_helpers.py:69`, `qwen25_1_5b_cpu_helpers.py:42` (+6 more) |
| `transformer_layer/builders/elementwise_add.py:124` | `triu(k=1)` → banded bias |
| `mha_attention.py:137` `attention_config(...)` | dict-passing style — a `window` key costs nothing structurally |

### 2.2 The load-bearing blocker: masking is element-wise, **not** tile skipping

The KV chunk loop bound is a static compile-time constant independent of the Q tile —
`attn_npu2.py:693-694` builds `c_chunks_h = ConstantOp(index_type, chunks_per_stage)` and iterates
the whole range; the L3→L2 streaming at `:535` and `:552` pushes `chunks_per_stage * dk_chunks`
blocks unconditionally. Every (Q tile, KV block) pair runs both matmuls and is then masked
(`:731-751`, comment "4b. Apply causal mask (after matmul, before softmax)").
`mha_attention.py:62-64` states it in prose: the composed path "fills a wholly-masked score tile
with `-inf` and lets the matmul run anyway, so no O tile is ever left untouched."

Three consequences, and they set the whole shape of the goal:

1. **A banded mask is cheap and low-risk to make CORRECT** — one C++ function plus one scalar.
2. **It buys exactly zero speedup.** The skipped region is still streamed and multiplied. A
   windowed model would run at full-causal cost.
3. **The route that would buy the speedup is documented as hang-prone and has never been
   exercised.** `attn_npu2.cc:703-775` (`#ifdef CAUSAL_ROW_HELPERS`, always linked via
   `external_kernels.py:238, :285-286`) holds `copy_O_tile_rows`, `store_row_value`,
   `copy_row_values`, with an explicit FOOTGUN at `:715-718`:

   > "copy_O_tile_rows is numerically a no-op -- it reads every element and writes it straight
   > back. That is the point, not an oversight. Deleting it as dead code hangs the design
   > (ERT_CMD_STATE_TIMEOUT) because the consuming DMA never sees its BD complete."

   `mha_attention.py:55-66` records that these symbols are kept linked *specifically* so a
   block-skipping variant can be added "without re-deriving the flag set". They are scaffolding for
   a variant nobody has built.

A latent measurement trap that travels with this: `attn_npu2.py:1361-1363` does
`perf_flops *= 0.5` when `causal`. The reported FLOP count is a **convention**, not executed work.
Anyone quoting GFLOP/s for a windowed variant would compound that error by the band ratio.

### 2.3 Numerics headroom is thin, and banding pushes the wrong way

`kernel_registry/details/FlashAttention_bf16.md:130-139` — element-wise over the full output vs an
FP32 reference, `rtol = 1.6e-2`, **`atol = 1e-1`** (looser than GEMM's `4e-3`, sized to FA's
measured worst case: two BFP16-emulated MMAs plus a bf16 online softmax). Measured
`mean_rel_L1 ≈ 3.8–5.5e-2`. `:78` already warns that `rel_err max` is meaningless here because
causal masking produces near-zero references.

In the study tree the same pressure is visible with numbers: the causal `mha_out_proj` rows carry
`atol = 8e-2` at only **1.64× `atol_required`** against a hard `1e-1` ceiling
(`transformer_layer/opcheck_specs.py:711-723`).

A band concentrates the softmax harder than causal does and produces *more* fully-masked blocks and
*more* near-zero outputs. **INFERENCE:** a banded row is likely to land closer to that ceiling than
the causal row does; whether it clears it is exactly what a first increment should measure rather
than assume. The relevant safety net already exists in the kernel —
`attn_npu2.cc:286` initializes the softmax max to `0xff7f` (bf16 lowest), not `-inf`, explicitly
"For fully-masked rows (all -inf)" — but that path is currently rare and would get exercised far
harder.

### 2.4 The registry

`programming_examples/kernel_registry/` (**not** `llms/kernel_registry/` as doc 11's task framing
has it). `supported_kernels.md:495-522` is the FlashAttention section.

- **`causal` is already a registered column** (`:499`), with 14 rows at `:501-514` covering both
  settings: causal ✓ at 2048×2048 (dk/dv 64/64 and 128/128, several head configs) and 16384×16384;
  causal ✗ at 512×512 and 16384×16384. All share `lqp=256, num_q_tiles=4, num_heads_per_unroll=2,
  num_cascade_stages=4`.
- Registered `dk`/`dv` are **64 and 128 only**. Any model whose head_dim is not one of those needs a
  newly validated row — and the repository ships a dedicated skill, `debug-fa-runtime-failure`,
  for FA hangs and NaN at head_dim ≥ 128, so this is a known-sharp axis.
- **Zero mentions of windowing or sliding** anywhere in `kernel_registry/`, `flash_attention/` or
  `attention_decode/`.

### 2.5 Nothing in `llms/` knows what a window is

Greps over `programming_examples/llms/` (excluding build artifacts): `sliding` 0 hits,
`sliding_window` 0, `local_global` 0, `layer_types` 0, `is_causal` 0; `window` 1 hit, unrelated
(`llama32_1b_int4/Makefile:89`, "Top-K window for HF comparison"). The only `causal=True` call
sites are `shared/infra/fa_headfirst.py:92`, `llama32_1b_prefill.py:190`,
`llama32_1b_int4_prefill.py:1220`, `qwen25_0_5b_prefill.py:966`. **There is no `causal=False` call
site in `llms/` at all** — non-causal is exercised only by the standalone registry harness.

In `transformer_layer/`, `sliding` has zero hits and every `window` hit is `study/power.py`
sampling windows or `run_lock.py`'s "Windows" the OS. **Careful:** "band"/"banded" throughout that
tree means 64-row *activation* bands (addnorm/FFN row tiling — `coarse_c3`'s "banded tail",
`ffn_resident.py:88`, `addnorm.py:130`), never a banded attention mask.

### 2.6 The two architecture gates

Both reject, and they reject differently.

`.claude/skills/deploy-new-llm/SKILL.md:83-84` — verbatim:

```
- Has sliding-window attention (`sliding_window` set in config AND
  `use_sliding_window=true`)
```

A **conjunction**. `.claude/skills/phase-0-build-cpu-reference/SKILL.md:106-109` — verbatim:

```
- Architecture must be in `["LlamaForCausalLM", "MistralForCausalLM"
  (only if no sliding window), "Qwen2ForCausalLM", "Qwen3ForCausalLM"]`
  ...
- Reject if: MoE layers, sliding-window attention, MLA, encoder-decoder
```

A **closed four-entry allowlist**, plus an *unconditional* sliding-window rejection with no
`use_sliding_window` conjunct. So the two gates disagree for any model that sets `sliding_window`
without `use_sliding_window`: it passes `deploy-new-llm` Step 2 and then fails `phase-0` Step 2.
Doc 11 spotted the allowlist; it did not spot that the two conditions are inconsistent with each
other. Both trees carry identical gate text (§0b).

### 2.7 The `exper/gemma3-dataflow` landing decision

Doc 11 calls this "the first decision of this goal". The measurements say **do not land it — mine
it.**

| fact | value | how measured |
|---|---|---|
| tip / date | `53348bfe`, 2026-06-23 (dormant ~7 weeks) | `git log -1` |
| merge-base with this branch | `90dc5e92`, **2026-05-12** | `git merge-base` |
| divergence | **221 ahead / 422 behind** this branch | `git rev-list --left-right --count` |
| whole-branch diff vs merge-base | **231 files, 112,129 insertions** | `git diff --stat` |
| `programming_examples/gemma3/` | 182 files, 17 commits touch it | `git ls-tree`, `git rev-list --count` |
| compiler changes it carries | **~2,348 lines across 22 `mlir/` files** | `git diff --stat -- mlir/` |
| `programming_examples/llms/` on that branch | **does not exist** | `git ls-tree -d` |
| verify adapter / prompts / `make verify` on that branch | **none** | `git ls-tree -r` |

Doc 11's "182-file parallel tree" and "roughly 15 commits" both check out for the `gemma3/`
subtree. What it omits is everything around it:

- **The `llms/` reorganization is not on that branch at all.** So "reconcile into `llms/`" is not a
  move — it is a port across a reorg into a directory layout the branch has never seen, plus a
  `verify_adapter.py` written from scratch, plus a `Makefile` with `verify`/`verify-full`/
  `diagnosis` targets it has none of (its targets are all kernel-level: `run-q4nx`, `run-mm`,
  `run-fused-dqp`, `run-flowqkv`, `run-flowkv`, `run-geglu`, `run-rope-halfsplit`,
  `model-blockers`, `model-prepare`, `model-prefill`).
- **The compiler surfaces overlap head-on.** Both branches have heavily rewritten the same three
  files since the 2026-05-12 merge-base:

  | file | gemma3 branch | this branch |
  |---|---|---|
  | `mlir/lib/Conversion/AIRToAIEPass.cpp` | +545 | +2337 |
  | `mlir/lib/Conversion/AIRRtToNpuPass.cpp` | +14 | +1910 |
  | `mlir/lib/Transform/AIRMiscPasses.cpp` | +109 | +449 |

  (Some of this branch's volume is inherited from `main`; the conflict surface is real either way.
  The gemma3 side's touched test files include `direct_shim_s2mm_channel_exhaustion.mlir`,
  `shim_packet_flow_npu.mlir` and `shim_pkt_channel_sharing.mlir` — precisely the shim/packet
  machinery H9, item 6a and item 6b rewrote here.)

**What to mine instead:** the `WINDOW_LEN` band formulation in `aie_kernels/flow_attention.cc:27-28,
:62-63`, the per-layer local-cache sizing in `npu/bo_plan.py:126-153`, and the `5-local-1-global`
detection in `npu/preflight.py:116, :127`. Those are design inputs worth an hour of reading and
zero merge risk.

---

## 3. The four execution modes, and the standing design rules

### 3.1 Today the modes are non-causal encoders, at a literal

Every mode passes `workload_variant="encoder_bert"` as a hardcoded argument:

- `transformer_layer/pattern/coarse/cells.py:590`
- `transformer_layer/pattern/runlist/runlist.py:867`
- `transformer_layer/pattern/offload/offload.py:874`
- `transformer_layer/pattern/fused/fused.py:571`
- and `transformer_layer/study/run_mode.py:174` writes `row["workload_variant"] = "encoder_bert"`
  into the CSV unconditionally (the shape key at `:160` is likewise `f"{seq_len}x{emb}_encoder_bert"`).

Each also records `"causal": False` in its artifact (`builders/block.py:271, :368`;
`pattern/fused/fused.py:465`; `pattern/coarse/cells.py:673`; `pattern/runlist/runlist.py:994`), and
`pattern/reference.py:93` defines `encoder_bert` as "POST-norm and **non-causal**".

The `gpt2_*` (causal) families are declared in the case matrix and gated by nothing —
`study/cases.py:55-58` says so in as many words: "`gpt2_*` families are declared and no mode is
gated on them today ... a decoder run needs the causal-mask path, which is a builder keyword
argument here rather than an operator".

Worse for work item 8: **the decomposed mode interiors have no mask step at all.**
`runlist.py:643` `run_attention_head` materializes full `[seq_len, seq_len]` bf16 score and prob
buffers (`:669-670`) and dispatches `attn_scores` GEMM → `softmax` → `attn_output` — no mask
between them. `offload.py:800` `_device_attention` is documented "Non-causal multi-head attention
with BOTH matmuls on the device", with `_host_softmax_bf16` (`:779`) an unmasked
`torch.softmax(scores * scale, dim=-1)`.

So work item 8 sits behind an unscoped prerequisite: make the four modes run a **causal** layer.
That is a mode rebuild — a new mask dispatch (or a masked softmax) inside `runlist`/`coarse`, a host
mask in `offload`, `causal=True` through `fused`'s FlashAttention path — with all four gates
(`run_npu2_{runlist,coarse,offload,fused}_peano.lit`) and their negative controls re-derived. It is
not a parameter flip and doc 11 does not price it.

### 3.2 The per-column shim budget (doc 23 §1) — the additive-mask route costs a stream

The rule: **a column has two shim MM2S channels, and the budget is per column across the whole
segment**; exceed it and AIR packet-multiplexes onto one queue. Doc 23's measured table:

| builder | L3→L1 streams | packet-typed channels |
|---|---|---|
| `addnorm` (x, residual, weight) | 3 | **3 — multiplexed** |
| `elementwise_add` (a, b) | 2 | 0 |
| `layer_norm` (in) | 1 | 0 |

Two candidate ways to window the *decomposed* modes, with different budget consequences:

- **Additive `[seq,seq]` mask tensor** (the `causal_mask` primitive of §1.5, fed as elementwise-add's
  `b` operand). This makes the mask a **third L3 operand** into the scores stage — the exact
  `addnorm` shape that measured 3 packet-typed channels. **INFERENCE, marked:** this route puts the
  masked scores stage back on the packet path, and no measurement of a masked attention interior
  exists to check it against. It also costs DRAM traffic in a study whose axis *is* DRAM traffic:
  `[4096, 4096]` bf16 is 33.5 MiB read once per band.
- **Band in-kernel** (the FlashAttention route of §1.2). No new operand, no new stream, no budget
  impact. This is another argument for making the kernel route the first increment.

Also relevant: **queue item 10** — R1's shipped column census (`ffn_resident_structure.py`) counts
`shim→core` flows only and reads **1** for a column actually carrying **2** of the 2-per-column
budget ([31b §7.1](../../../../home/cj/mlir-air/docs/plans/transformer-layer-execution-studies/31b-r2-order-seam.md)).
Any structural arm written for a windowed design would inherit that blind spot until item 10 is
fixed.

### 3.3 The L3-side offset rule (doc 23 §2) — this is where windowing actually bites

The rule: **advance a staged buffer on the L3 side, never on the L2 read.** An `aie.dma_bd` offset
is static; an IV-dependent offset on an L2/L1 operand does not fail loudly — past the unroll limit
the chain freezes every offset at 0, the core stalls, and the output buffer comes back byte-identical
to what the host wrote (or `ERT_CMD_STATE_TIMEOUT`). J7b lost a session to it.

The windowed pattern changes exactly the property that made the causal KV feed safe:

> Under full causal, query band `b` consumes KV tiles `[0 .. b]` — a **prefix**. A KV feed can
> stream contiguously from a fixed base; only the trip count varies with `b`.
> Under a window, band `b` consumes `[b − W/T .. b]` — a **moving range**. The feed's base offset
> now advances with the band index.

**INFERENCE, marked, and it is the sharpest thing in this document:** a KV feed indexed by *both*
the band and the k-step is a **two-symbol** affine map on an L3 operand — which is **open queue item
8** verbatim: `air-split-l2-memref`'s `tileChannelOpByFactor` builds its replacement with
`AffineMap::get(0, 1, add)`, exactly one symbol, and SIGABRTs on `()[s0, s1] -> (s0*C0 + s1*C1)`;
verified at **three** hardcoded sites, `mlir/lib/Transform/AIRMiscPasses.cpp:1671`, `:1674`, `:1681`.
Item 8's own entry notes that doc 23's L3-side rule "produces two-symbol maps as a matter of
course".

Two qualifications that keep this honest:

- **It does not bite the mask-only route.** Today's FA path streams the full KV span unconditionally
  (`attn_npu2.py:535, :552, :693-694`), so no offset moves and item 8 is not reached. The collision
  arrives with the *tile-skipping* variant — i.e. with the performance half of the goal, not the
  correctness half.
- **Queue item 9** is the same class of hazard on the other side:
  `air-shrink-memref-sizes-by-access` silently shrinks a multi-get L1 band
  (`memref<12288xbf16,2>` → `<3072>` while the gets stay at 3072/6144/9216), no error, no warning.
  A banded L1 read at literal offsets is precisely the shape that trips it.

**Net:** correctness-only windowing is design-rule-neutral. Performance windowing collides with two
already-open, unclaimed compiler items (8 and 9) plus the documented `copy_O_tile_rows` hang path.
That is a coherent argument for sequencing: correctness first, and only then decide whether the
performance half is worth unblocking two compiler defects for.

### 3.4 One more mode-level pin

`softmax.py` requires `cols % 64 == 0` (`builders/softmax.py:130`) and has **no scalar tail**
(`:59-62`) — a windowed row width that is not a multiple of 64 would be silently truncated. Relevant
only if a band ever shrinks the effective row width in the decomposed path.

---

## 4. "window-crossing prompts" — what the gate would need, and whether it discriminates

### 4.1 What the gate is, exactly

- `GATE_N_TOKENS = 32`, `GATE_K = 5` — `programming_examples/llms/verify/verify_runner.py:70-71`.
- `compute_topk_set_check()` — `verify/comparators.py:175-249`. It walks NPU-chosen vs HF-chosen
  tokens in lockstep, skips identical steps, and **at the first divergence only** requires each
  side's chosen token to appear in the other's top-5; then it returns (`:222`).
- Pass/fail: `Report.has_failure()` — `verify/report.py:54-61`, "Only the verify-mode top-k gate
  signals failure"; `verify_runner.py:490-493` exits 1 on FAIL.
- `make verify` passes `--max-prompts 2`; `make verify-full` runs all 8
  (e.g. `llms/llama32_1b/Makefile:101-104`).
- Reference: `verify/runners/hf_runner.py:34`, `AutoModelForCausalLM.from_pretrained(...,
  torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)` at `:65-71`, **CPU** (docstring `:4`), cache
  reset per prompt (`:84`).

### 4.2 The prompts today

`verify/prompts/instruct.txt` (13 lines / 159 words) and `verify/prompts/base.txt`
(15 lines / 136 words) — 8 prompts each, roughly 3–30 BPE tokens. **Nothing in the shipped set comes
within two orders of magnitude of a 512-token window.** A new fixture file is unavoidable.

A wrinkle: `--prompt-style` (`verify_runner.py:63-69, 204-215`) is never passed by any Makefile, so a
new prompt file needs a Makefile flag or a new default. (Same wrinkle already makes
`make verify MODEL=base` read `prompts/instruct.txt`, contradicting `verify/README.md:46-49` — a
pre-existing inconsistency, not caused by this goal.)

### 4.3 The length ceiling — and the good news

```python
max_seq = 2048  # Production prefill kernels are tiled for seq_len=2048.
```
`verify_runner.py:275` — hardcoded, not a CLI flag. Enforcement is **silent truncation**, never an
assertion: `_prompt_tokens()` at `:376-378` returns `ptoks[:max_seq]`; the diagnosis path truncates
at `:346-347`. Every adapter then EOS-pads back *up* to `max_seq` so verify hits the same kernel
shape `make run` does (`llama32_1b/verify_adapter.py:159-166`, and the identical idiom in all nine
others).

**The good news, and it is the single most decision-relevant fact in this section:** the window the
gate needs to cross is **512** (the gemma3 branch's recorded value, §1.1), and `512 < 2048`. A
600–1500 token prompt crosses the local window *and* fits the existing prefill tile. **No `max_seq`
change, no new prefill shape, no re-tiling is required for a Gemma-3-style 512 window.** (A
Mistral-style 4096 window would require all of it — worth stating explicitly, because doc 11 names
both families in one breath and they are not the same job.)

Two things still to check rather than assume:
- The KV cache is allocated `(n_layers, n_kv_heads, max_seq, head_dim)`
  (`llama32_1b_inference.py:450-451`) and `decode_step` writes at `current_pos = prompt_len + i`
  with no bound check — a prompt longer than `2048 − 32` silently overruns. A window-crossing
  fixture must stay under that, and it comfortably can.
- EOS padding after the prompt interacts with a window in a way it does not with full causal. At
  the read position (`prompt_len − 1`) a 512-window attends to `[prompt_len − 512, prompt_len − 1]`,
  all real tokens, so it should be fine — but this is a **check to run**, not a conclusion.

### 4.4 The reference side needs no work

`HfRunner` applies whatever windowing the HF config declares. It accepts `max_seq`
(`hf_runner.py:41, :56`) and **never enforces it** — no truncation, no assertion. So the reference
is windowed for free, given `transformers` support and weights on disk. Adapters may inject a
pre-built model via `model=` (`verify_runner.py:317-324`), the hook `llama32_1b_int4` already uses.

### 4.5 **The gate as specified may not discriminate** — the most important finding in this section

The gate reads the **first divergence only**, over 32 greedy tokens, on a prompt whose length it
does not otherwise constrain (`comparators.py:175-249`).

**INFERENCE, marked, and it should be tested before the gate is trusted:** a windowing
implementation that silently degrades to full causal — the most likely bug class here, since full
causal is the *superset* and the existing default — will frequently still pass. Full causal on a
600-token prompt attends to a superset of the correct band; the logits shift, but greedy top-1 over
32 continuation tokens is a coarse instrument, and the check stops at the first disagreement.

This is exactly the hazard the plan directory already guards against everywhere else. Doc 13's gate
table has a dedicated row — "**Check discriminates** — `opcheck.py --operator <op> --fault-inject
input`, which must **fail** ... a vacuous check passes under injection; the driver fails the phase
for it" — and doc 13 §Two gates that deserve emphasis makes the same argument about
`execution-smoke-test` checking rows rather than files.

**Recommendation: the gate must be `make verify` with window-crossing prompts PLUS a negative
control** — the same prompts with the window disabled (or set to full length) must **FAIL** the same
check. Without that, "`make verify` passes with window-crossing prompts" is not evidence that
anything was windowed. `make diagnosis` (per-layer cosine) cannot substitute: it is informational
and never gates (`report.py:54-61`; doc 13 §Correctness standards).

### 4.6 Weights and CI

- `programming_examples/llms/hf_models.txt` holds **16** repo IDs (4 Llama-3.2, 6 Qwen2.5, 3 Qwen3,
  2 SmolLM2, 1 AWQ Llama). **No Gemma, no Mistral.**
- `.github/workflows/downloadLLMWeights.yml:122` reads that file directly to `snapshot_download`
  into the persistent HF cache the nightly reads with `HF_HUB_OFFLINE=1`. It is
  `workflow_dispatch`-only, on the self-hosted `amdryzenai5pro340` runner.
- **The Gemma-3 weights are not on this machine.**
  `~/.cache/huggingface/hub/models--google--gemma-3-1b-pt` and `...-3-4b-pt` are **12 KB stubs
  containing only `refs/main`** — a resolved commit sha (`fcf18a2a...` for 1b-pt) and no
  `snapshots/`, no `blobs/`. Someone resolved the ref and never pulled the weights. Gemma is a
  license-gated HF repo, so this needs an accepted licence on the account the runner uses, then a
  workflow run.

---

## 5. Effort, risk, and the smallest first increment

### 5.1 Work decomposition (my estimate — **INFERENCE**, sized against the phase durations recorded in the README status board)

| # | Work | What it needs | Size |
|---|---|---|---|
| W1 | **Banded `apply_causal_mask` + `window` builder kwarg + banded oracle + one registry row + one lit gate with a negative control** | `attn_npu2.cc:634-701`, `attn_npu2.py` (+ seq-first twin), `FlashAttention_bf16.md` | small, self-contained; **the only piece that produces evidence about windowing on this stack** |
| W2 | Both architecture gates + the `llama_kernel_builder` staleness | 2 files × 2 trees for the gates; 11 files in `.claude` for the paths | small, but do it **after** W1 — opening the gate before the kernel exists invites a deployment that cannot pass |
| W3 | **Gemma 3 deployment** — the 7 `deploy-new-llm` phases | weights loader, cpu_helpers, per-kernel validation at Gemma's head_dim, QK-norm, the 5:1 layer pattern, `verify_adapter.py`, Makefile, 3 lit files, 10-model regression | **the bulk.** A full model bring-up |
| W4 | Land / reconcile `exper/gemma3-dataflow` | see §2.7 | **recommend: don't.** Mine it instead |
| W5 | Four-mode measurement of the new model (doc 11 item 8) | first make the four modes run a *causal* layer at all (§3.1) | a mode rebuild that doc 11 does not price |
| W6 | Windowed decode | a NumPy slice per model (§1.4); the ring buffer is optional and is what *creates* doc 11's RoPE problem | small if the ring is declined |
| W7 | Tile-skipping (the performance half) | `copy_O_tile_rows` hang path + queue items 8 and 9 | **large and blocked**; explicitly out of scope for a first increment |

### 5.2 Risks

1. **The stated gate may be vacuous** (§4.5). Highest-priority risk, because it would let the goal
   report success without evidence — the exact failure mode this directory has retracted claims for
   twice.
2. **Numerics headroom is thin** (§2.3): registry `atol = 1e-1` already, `mean_rel_L1 ≈ 3.8–5.5e-2`,
   and the study's causal rows sit at 1.64× `atol_required`. Banding pushes toward the ceiling.
3. **Zero speedup from the correctness route**, and the speedup route is a documented hang
   (§2.2) plus two open compiler defects (§3.3).
4. **Local/global means two ELFs per model** (§1.3) — no runtime scalar path. That multiplies
   compile time and symbol pressure, and this study has already been bitten by symbol collisions
   when one method minted two shapes (doc 13 §cross-deployment rule: `shared/builders/gemm_builder.py`
   mints names from the GEMM method alone — `shared/infra/stitching.py:318`,
   `shared/infra/external_kernels.py:133`).
5. **`attn_npu2.cc` is shared by all ten models.** Any edit owes the 10-model `make verify`
   regression, which doc 13 calls "the most expensive check in the plan and the one most likely to
   be skipped".
6. **External dependency**: license-gated weights that are not on the machine (§4.6). This is the
   only queue item that depends on something outside the repository.
7. `deploy-new-llm` Step 3 halts today (§0b), so the deployment skill path is broken before the
   first step is taken.

### 5.3 The smallest first increment that produces evidence rather than scaffolding

**Band the existing FlashAttention mask and gate it as a kernel row, with a negative control.**

Concretely:

1. Add a `window` build kwarg to `attn_npu2.py` / `attn_npu2_seqfirst.py` (default 0 = unwindowed),
   baked as a `ConstantOp` and passed as a third `i32` to `apply_causal_mask` — or, matching the
   gemma3 branch's own choice, as a `-DWINDOW_LEN` through `external_kernels.py:274-286`. Widen the
   `external_func` declaration at `attn_npu2.py:220` / `attn_npu2_seqfirst.py:232` accordingly.
2. Band `apply_causal_mask` (`attn_npu2.cc:634-701`) using `q_abs = q_block_idx*lqp + row` and
   `k_abs = kv_block_idx*lkp + col`, which the compile-time `lqp`/`lkp` (`:23-29`) already make
   available. Keep the three existing branches; add the two new ones from §1.2.
3. Band the harness oracle at `attn_npu2.py:1341-1343`.
4. Gate it: one new lit recipe beside `flash_attention/kernel_fusion_based/run_npu2_makefile_peano_causal.lit`,
   at a shape **already in the registry** (e.g. 2048×2048, dk/dv 64/64, causal ✓ —
   `supported_kernels.md:501-514`) with `W = 512` (a multiple of `lkp = 64`, so no partial-select
   generalization is needed). Element-wise `np.isclose` over the full output at the registry's
   `rtol = 1.6e-2` / `atol = 1e-1`, per `FlashAttention_bf16.md:130-139`.
5. **Negative control, non-optional**: the same fixture with the band disabled must **FAIL**. Doc
   13's "Check discriminates" row is the precedent.
6. Write the measured row into `kernel_registry/details/FlashAttention_bf16.md` and
   `supported_kernels.md` with a new `window` column — `causal` is already a column (`:499`), so
   the precedent exists.
7. Re-run the ten-model `make verify` (doc 13's rule), since `attn_npu2.cc` is shared. `window = 0`
   should be bit-identical to today, which makes that regression a cheap, strong check.

**Why this is the right first move.** It answers the two questions that actually gate the goal —
*does a band pass at the registry's tolerance on real hardware*, and *how much headroom is left* —
before anyone commits to a model deployment, a branch reconciliation, or the skill-gate edits. It
touches one C++ function that is the entire causality implementation for all ten shipped models. It
needs no weights, no HF licence, no branch merge, no new prefill shape. It is falsifiable and it
carries its own negative control, which the goal's stated gate does not.

**Deliberate non-goals for increment 1:** no tile skipping (§2.2, the hang path), no decode ring
buffer (§1.4, decode is host NumPy), no architecture-gate edits (§2.6, W2 comes after), no branch
landing (§2.7), no four-mode work (§3.1).

**The decision point after increment 1** is then a real one rather than a guess: if the banded row
passes with headroom, W2+W3 (the Gemma 3 deployment) is a known-shape bring-up with its risk
concentrated in head_dim and weight access. If it lands against the `1e-1` ceiling, the goal needs a
numerics answer before anything else, and that is much better to learn from one kernel gate than
from a stalled model deployment.

---

## 6. How Goal 1 compares as a next move, on its own terms

*(Phase G and Goal 2 are being scoped in parallel; nothing here anticipates those findings.)*

**What Goal 1 uniquely unlocks.**

- **The one architectural axis the study explicitly names as missing.** Doc 11's own framing:
  ten shipped models exercise "attention norm/bias, head dimension, and hidden size — not attention
  *span*". Everything else in the queue deepens what exists; this widens it.
- **A model class, not a metric.** Sliding-window / local-global is the gating architecture for a
  large share of current open-weight releases. Nothing else in the queue changes which models the
  repository can accept.
- **A second capability axis for the kernel registry** — the first non-full-causal, non-trivial mask
  variant, in a registry where `causal` is already a column.
- It is the only queue item that produces a **user-visible capability** rather than infrastructure,
  precision, or compiler correctness.

**What Goal 1 risks, specifically.**

- **Its stated gate may not discriminate** (§4.5). Alone among the open items, its acceptance
  criterion needs to be strengthened before it is worth running.
- **It is three weakly-coupled sub-projects behind one queue row** — a kernel change, a full model
  bring-up (7 skill phases), and a branch reconciliation — and only the first is small. The row
  reads as one decision and is not.
- **The correctness half delivers no performance**, and the performance half is blocked on a
  documented `ERT_CMD_STATE_TIMEOUT` path plus two unclaimed compiler defects (items 8 and 9). So
  "SOTA model coverage" here means *runs correctly*, not *runs well*, unless W7 is also taken.
- **Its four-mode measurement half is blocked on an unscoped prerequisite** — the modes have never
  run a causal layer (§3.1). Any promise that Goal 1 feeds the study's measurement axis is
  premature.
- **It is the only item with an external, license-gated dependency** (§4.6) and the only one that
  edits a file shared by all ten shipped models on every iteration (§5.2.5).
- Its prior art is dormant since 2026-06-23, 221/422 diverged, structured against a directory
  layout that no longer exists, and self-classified as `REAL_MODEL_EXECUTION_NOT_IMPLEMENTED`
  (§0a, §2.7).

**The asymmetry that matters for the decision.** Goal 1's *tail* is the largest and least bounded of
anything in the queue, and its *head* — increment W1 — is among the cheapest genuinely-informative
pieces of work available: one C++ function, one builder kwarg, one gated registry row with a
negative control, no weights, no merge, no new shape. Those two are separable. **Taking W1 is
defensible even if the full goal is not chosen**, because it converts the goal's biggest unknown
(does a band clear the FlashAttention tolerance on this hardware?) into a measured row, and because
the banded mask is a capability the registry can carry on its own.

Taking W3/W4 without W1 first is the move to avoid: it front-loads the branch decision and the
weight access behind a kernel question nobody has answered.

---

## Provenance

Every figure above traces to one of:

- Repository files at tip `b777517b`, cited by absolute path and line.
- `exper/gemma3-dataflow` at tip `53348bfe`, read via `git show` / `git ls-tree` / `git grep` — never
  checked out.
- `git rev-list` / `git diff --stat` / `git merge-base` counts, run read-only.
- `~/.cache/huggingface/hub/` directory listing (Gemma stubs).

Claims marked **INFERENCE** are: the packet-multiplexing consequence of an additive-mask route
(§3.2), the two-symbol L3 offset / queue-item-8 collision for a tile-skipping windowed feed (§3.3),
the numerics-headroom expectation for a banded row (§2.3), the gate's discrimination weakness
(§4.5), and the W1–W7 sizing (§5.1). No measurement was taken for any of them and none should be
cited as one.

No latency figure appears in this document, by instruction. NPU power mode was reported as having
just been set to Turbo; nothing here depends on it.
