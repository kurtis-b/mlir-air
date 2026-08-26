# Queue item 25 / doc 58 phase **M0** — the scope, read-only

`[2026-08-26]` Branch `exper/transformer-layer-execution-studies`, HEAD `ea3902f0` ("docs: doc 58
— the plan for running real models through the four execution modes (operator chose 4a)").

**Read-only.** No file outside this directory was written, nothing was compiled, no device job was
submitted, no devq id was consumed. Every claim below carries a `file:line`. Where a claim is
derived rather than read, it is labelled **INFERENCE**.

M0's job, from doc 58's phase table (`docs/plans/transformer-layer-execution-studies/58-models-through-the-four-modes.md:47`):
the leaf-granularity inventory of §2, per mode the exact list of what must be written, and a
revised estimate per phase — so the operator can accept the program or stop on it before M1.

---

## 0. The three findings that change the plan

Stated first because they are what M0 exists to produce.

**F1 — §2's cost-halving claim is TRUE but its stated mechanism is WRONG, and the correction
moves work rather than removing it.** Doc 58:36-38 says the shipped leaves live in
`llms/shared/builders/` and names `gemm_builder` as the leaf and `rms_qkv_qknorm_rope_multi` as
the wrong-granularity fused form. Read: **every one of the eight files in
`llms/shared/builders/` is a whole-ELF stage builder** — the module's own `__init__.py:5-6` calls
them "multi-launch ELF builders that stitch GEMM / RMSNorm / RoPE / attention sub-kernels into a
single ELF for one XRT invocation". `gemm_builder.py` is not an exception: its public entry
`_build_gemm_module` (`llms/shared/builders/gemm_builder.py:253`) is a thin adapter over
`matrix_multiplication/bf16_in_bf16_out/run.py:43`, which emits a full `air.launch`
(`run.py:268`). The real leaves are (a) the per-kernel example directories at the top of
`programming_examples/`, and (b) **private single-launch `@module_builder` functions living
inside the shipped stage builders**. That second set is the one the study already uses
(`transformer_layer/builders/elementwise_add.py:57` imports `_build_add_2d_to_2d` from
`shared.builders.o_ffn_multi`), and it is where the RoPE and QK-norm leaves are. §1 below is the
inventory. Net effect on cost: gap 2's *device-IR* half really is re-assembly; gap 2's
*correctness-oracle* and *shape-plumbing* halves are not, and doc 58 prices neither.

**F2 — Qwen3-0.6B's attention is not portable into `coarse` or `fused` as they stand, and this is
the one unbounded item in the program.** The study's attention is the **seq-first**
FlashAttention (`transformer_layer/builders/mha_attention.py:13-16`, imported at
`mha_attention.py:291`). `flash_attention/kernel_fusion_based/attn_npu2_seqfirst.py:121-125`
asserts `dv_chunks == 1`. `llms/shared/infra/fa_headfirst.py:7-13` states the consequence
directly: *"At head_dim=128 the kernel needs dv_chunks=2 (lkp=64, dv_tile=64), which seq-first
cannot express, and its dk_chunks>1 path hangs. So head_dim=128 MUST use the HEAD-FIRST kernel
`attn_npu2.py` with host-side transposes"* — and the registry says the same
(`kernel_registry/details/FlashAttention_bf16.md:294`). Qwen3-0.6B is head_dim 128
(`llms/shared/plan/graph.py:224`). The head-first path costs **two host transposes per layer**
(`fa_headfirst.py:140`, `:172`), which is exactly the property `coarse`'s sequence A
(`builders/block.py:68-71`, "fully device-resident … without touching the host") and `fused`'s
one-submission claim (`pattern/fused/fused.py:8-16`) exist to have. §4 records this as the
sharpest structural refusal.

**F3 — the two families are complementary, and doc 58's family order is the expensive one.**
Doc 58:54 says "Second family (Llama-1B, GQA + no QK-norm) only after M4". But Llama-3.2-1B is
head_dim **64** (`llms/shared/plan/graph.py:229`), which is the study's entire attention case
matrix (`builders/mha_attention.py:77-83`), and satisfies `num_heads * head_dim == emb_dim`
(32 × 64 = 2048), which `builders/block.py:296` and `pattern/runlist/runlist.py:287` *require*
and Qwen3-0.6B *violates* (16 × 128 = 2048 ≠ 1024; `llms/qwen3_0_6b/qwen3_0_6b_weights.py:11-12`
names the decoupling). Conversely Qwen3-0.6B has measured GEMM rows at M = 512/1024 where
Llama-1B has none, and `fused`'s packing cap admits Qwen3-0.6B to 1024 rows but Llama-1B to only
512 — a sequence at which no Llama-1B GEMM row exists. Neither family is "the easy one"; they are
easy in different places. §4 has the arithmetic; §6 has the recommendation.

---

## 1. The leaf-granularity inventory (doc 58 §2's claim, tested)

### 1.1 What the study needs, and what exists

The pieces a Qwen3-0.6B decoder layer needs are exactly the node list of
`llms/shared/plan/graph.py::decoder_graph` (`graph.py:145-217`; nodes at `:192-216`) — that DAG is itself a finding and
is treated in §1.3.

| Layer piece | What exists, and where | Granularity | Verdict for a mode study |
|---|---|---|---|
| **RMSNorm** (row-wise, weighted) | `weighted_rms_norm/weighted_rms_norm.py:44` `build_module(M, N, np_dtype, vector_size=16, herd_x=1)`; f32 oracle `rms_norm_reference(x, weight, eps=1e-5)` at `:325` | **Bare herd, no `air.launch`** (herd emitted at `:96-99` / `:220-223`) | **True leaf.** The best-shaped piece in the tree. Callers wrap it themselves — `o_ffn_multi.py:303` wraps it with `_wrap_ir_in_launch`. 4 shipped import sites (`o_ffn_multi.py:248`, `rms_gemms_rope_multi.py:223`, `rms_qkv_bias_rope_multi.py:377`, `rms_qkv_qknorm_rope_multi.py:519`). |
| **QKV projection** | `llms/shared/builders/gemm_builder.py:253` `_build_gemm_module(m, k, n, tile_m, tile_k_l2, tile_k_l1, tile_n, herd_m=8, herd_n=4, external_fused_cast=False, external_bf16_out=False, sym_suffix="", link_with_name="mm.o")` | Emits `air.launch` (via `matrix_multiplication/bf16_in_bf16_out/run.py:268`); a fused-cast build is **2 launches** | **Leaf at launch granularity, usable.** Already imported by the study: `transformer_layer/builders/ffn.py:73`. The *study's own* fused-QKV builder is NOT usable — see §2.1. |
| **QK-norm** (per-head RMSNorm) | `llms/shared/builders/rms_qkv_qknorm_rope_multi.py:90` `_build_qknorm_2d(outer_rows, outer_cols, head_dim, np_dtype, eps, herd_x, vector_size=16)`; 1-D sibling at `:266` | `@module_builder`, emits **one** `air.launch` (`:153`) | **Private leaf, usable.** Underscore-prefixed and undocumented as public API; the study already depends on one such symbol (`builders/elementwise_add.py:57`). Needs a pin test (doc 58:77). |
| **RoPE** | `llms/shared/builders/rms_gemms_rope_multi.py:61` `_build_rope_2d(outer_rows, outer_cols, embed_dim, np_dtype, herd_x)`; 1-D sibling `rms_gemv_rope_multi.py:267`; LUT oracle `rope_lut/rope_lut.py:147 generate_lut(...)` | `@module_builder`, emits **one** `air.launch` (`:131`) | **Private leaf, usable.** `rope_lut/rope_lut.py:39 build_module(seq_len, embed_dim, np_dtype_in, herd_x=1)` is a bare-herd public alternative but **nothing imports it** — the two shipped import sites take `generate_lut` only (`rms_gemms_rope_multi.py:529`, `rms_gemv_rope_multi.py:627`). `rope_halfsplit/rope_halfsplit.py:57` bakes its own launch and has zero importers. Qwen3's convention is `halfsplit` (`graph.py:41`, `:224`), which is what `_build_rope_2d` + the driver's LUT already implement. |
| **Attention** | seq-first `flash_attention/kernel_fusion_based/attn_npu2_seqfirst.py:56 build_module(lk, lkp, lq, lqp, dk, dv, num_q_tiles, num_cascade_stages, num_heads, num_kv_heads=None, causal=False, num_heads_per_unroll=2, window=0)`; heads-first `attn_npu2.py:56` (same signature); host plumbing `llms/shared/infra/fa_headfirst.py:114 npu_fa_headfirst(...)`, `:181 npu_fa_headfirst_kv(...)` | Emits a full `air.launch` (`attn_npu2_seqfirst.py:304`, `attn_npu2.py:326`) | **Whole-ELF boundary, and the WRONG one for Qwen3.** seq-first refuses head_dim 128 (`:121`). Heads-first works but its L3 layout is `[heads*dv_chunks, seq, dv_tile]` and needs the two host transposes at `fa_headfirst.py:140,172`. See F2 / §4.1. |
| **O projection** | Same `_build_gemm_module`. The *study's* wrapper `builders/o_proj.py:68 o_proj_gemm_spec(seq_len, emb_dim, ...)` and `builders/mha_out_proj.py:233 add("w_o", memref<{emb_dim}x{emb_dim}>)` are **square-only** | launch | Leaf usable; **the study's wrapper is not** — Qwen3's O is (2048 → 1024). New signature required. |
| **SwiGLU** (gate/up/down) | GEMMs: `_build_gemm_module` ×3. Activation: `silu_and_mul/silu_and_mul.py:143 build_module_2d(rows, cols, tile_n, np_dtype_in, herd_x=8, herd_y=1)`; oracle `silu_reference(x)` at `:249` | `build_module_2d` emits `air.launch` (`:173`) | **Leaf, usable and public.** 7 import sites; the shipped one is `o_ffn_multi.py:343`. `swiglu/swiglu.py:43` is a bare-herd sibling with zero importers and its oracle trapped inside `__main__` (`:223`) — do not use it. |
| **Residual adds** | `llms/shared/builders/o_ffn_multi.py:68 _build_add_2d_to_2d(rows, cols, np_dtype, vector_size=16, herd_x=8, herd_y=1)`, and the study's own wrapper `transformer_layer/builders/elementwise_add.py:64 build_elementwise_add_module(rows, cols, np_dtype=bfloat16, causal_mask=False, vector_size=16, herd_x=8, herd_y=1)` + f32 oracle `:128` | `@module_builder`, one launch (`o_ffn_multi.py:379` has the 2D→1D sibling) | **Already done.** This is the one piece the study has at the right granularity today, and it is the precedent for importing a private symbol. |

### 1.2 The three whole-ELF forms that are the WRONG granularity, and what they prove

- `rms_qkv_qknorm_rope_multi.py:462 build_rms_qkv_qknorm_rope_module(...)` — **8 launches** in one
  ELF (RMSNorm, Q/K/V GEMMs, QK-norm ×2, RoPE ×2; the file's own header, `:17-25`).
- `o_ffn_multi.py:182 build_o_ffn_module(seq_len=2048, emb_dim=2048, hidden_dim=8192, print_kernels=False)`
  — **8 stages / 8–12 launches** (O, residual add, FFN RMSNorm, gate, up, SwiGLU, down, FFN add;
  header at `:7-16`). Its private form `_build_o_ffn(..., q_dim=None, ...)` (`:206`) already
  decouples Qwen's `q_dim`.
- `lm_head_gemv_multi.py:39 build_lm_head_gemv_module(...)` — 8 launches, decode-only.

Doc 58:36-38's characterisation of these is correct: each *is* a baked boundary decision. What the
inventory adds is that **their internals are the leaves**, reachable as private
`@module_builder` functions, and that `llms/shared/infra/stitching.py` is a documented, already-used
re-assembly toolkit: `FuncArg` (`:248`), `KernelSlice` (`:261`), `alloc_gemm_scratch` (`:286`),
`stitch_elf` (`:318`), `_wrap_ir_in_launch` (`:99`, which self-skips when the IR already carries a
launch, `:137-139`). `transformer_layer/builders/qkv_proj.py:98` already imports three of these.

**So the correct statement of §2 is:** the study can re-assemble a decoder layer at any boundary it
likes from *single-launch leaves* + `stitch_elf`, and the leaves for RMSNorm / RoPE / QK-norm /
SwiGLU / GEMM / add all exist. That is a real halving of gap 2's device-IR work. It says nothing
about attention (F2), nothing about the oracle, and nothing about shape plumbing.

### 1.3 The piece doc 58 does not mention at all, and it is worth a phase

`llms/shared/plan/graph.py:145 decoder_graph(spec)` already builds the **exact** typed DAG doc 58
gap 2 describes: `attn_norm_L` (rms_norm), `q/k/v_proj_L`, `q_norm_L`/`k_norm_L`
(rms_norm_per_head, conditional on `spec.qk_norm`), `rope_q_L`/`rope_k_L`, `kv_append_L`,
`attention_L` (with `n_heads`/`n_kv_heads`/`head_dim`/`scale` attrs), `o_proj_L`, `residual_1_L`,
`ffn_norm_L`, `gate_proj_L`, `up_proj_L`, `swiglu_L`, `down_proj_L`, `residual_2_L`
(`graph.py:192-216`), over `QWEN3_0_6B` (`:221-225`) and `LLAMA32_1B` (`:227-230`) specs with every
shape symbolic in `(M, kv_len)`. `plan.py:240 fuse(...)` then groups those nodes into the shipped
stage names (`rms_qkv_qknorm_rope`, `flash_attn`, `o_ffn_qwen`; `plan.py:305`, `:328`, `:335`) with
derived launch counts.

**INFERENCE:** a mode's assembly for a family is a *different fusion policy over the same graph*.
`fuse` is already parameterised by phase, precision plan and forced methods; a fifth grouping
("one stage per node" = `runlist`, "one stage per linear op" = `offload`, "one stage per shipped
group" ≈ `coarse`, "one stage" = `fused`) is the natural expression of the mode axis. This is not
required for M1–M3, but it means the mode study and the model study can share one description of
the layer instead of two, which is directly the divergence risk doc 58:66-68 names.

### 1.4 Leaves that do NOT exist in reusable form

- **No RoPE builder with a public, imported entry point.** Every production RoPE is emitted by the
  private `_build_rope_2d` / `_build_rope_1d` inside a fused stage builder.
- **No importable device-side seq↔head repack.** `transformer_layer/builders/transpose.py:95
  build_transpose_module(rows, cols, np_dtype=bfloat16, herd_x=8, tile_rows=64)` is a **2-D**
  transpose; the head-first FA repack is a 4-D permute (`fa_headfirst.py:150-153`, `:174-176`).
- **No GEMV in the registry at all.** `kernel_registry/registry_lookup.py` has `gemm_config`
  (`:67`) and `gemm_config_method` (`:124`) and no GEMV entry point; `details/GEMV_bf16.md` is
  markdown with no JSON. The M=1 path is a different kernel family
  (`matrix_vector_multiplication/bf16/matvec.py:32 build_module(m, k, tile_m, m_input, herd_m, np_dtype_in, np_dtype_out, link_with="mv.o")`)
  with a different config schema (`herd_m/tile_m/m_input`, not GEMM tiles). Doc 58:15 calls this
  "no GEMV/M=1 path"; it is more than that — `resolve_gemm_spec` cannot express it even in
  principle, so a decode-mode cell needs a second resolver, not a second row.
- **Dead demo builders that look like leaves and are not:** `rms_norm/rms_norm.py:40`,
  `layer_norm/layer_norm.py:41`, `softmax/softmax.py:25`, `eltwise_add/eltwise_add.py:38`,
  `swiglu/swiglu.py:43`, `rope_sincos/rope_sincos.py:43` — all bare-herd, all with **zero
  importers**, none with an importable f32 reference. The study reimplemented `layer_norm`,
  `softmax` and `eltwise_add` rather than importing them
  (`transformer_layer/builders/layer_norm.py:124`, `softmax.py:169`, `elementwise_add.py:64`).
  A future reader should not mistake these for the shipped path.

### 1.5 What the study has today, restated

`transformer_layer/builders/` has **no RMSNorm, no RoPE, no QK-norm and no SwiGLU** — its layer is
LayerNorm + GeLU throughout (`builders/addnorm.py`, `builders/layer_norm.py`, `builders/gelu.py`,
`builders/ffn.py`). Confirmed by grep: the only occurrences of "rms"/"rope"/"swiglu" in
`builders/*.py` and `pattern/*/*.py` are prose references to the shipped files. So gap 2's
*inventory* is: six new study-side builder modules, each a thin wrapper over an existing leaf.

---

## 2. Per mode: the concrete list of what must be written

Common to all four (write once):

| # | Item | Where | Gap |
|---|---|---|---|
| C1 | Weights injection through `prepare_layer_dispatch` and the four pattern preparers | `opcheck_layer.py:429`, `pattern/{coarse,offload,runlist,fused}/*.py` | **1** |
| C2 | A Qwen3 family shape record: `head_dim` and `n_kv_heads` carried, not derived. `cases.FamilySpec` (`study/cases.py:88-97`) has no `head_dim` and no `n_kv_heads`; `Workload.attention_head_size` (`cases.py:180-181`) and `run_mode._shape_for` (`study/run_mode.py:187`) both compute `hidden // heads`, and `run_mode._shape_for`'s docstring says so explicitly (`:173-175`: "every family in the matrix is `hidden // heads == 64`") | `study/cases.py`, `study/run_mode.py` | 2 |
| C3 | A decoder-LLM golden model: RMSNorm, QK-norm, RoPE, GQA attention, SwiGLU, two residual adds — per-boundary f32, in the draw order convention `pattern/reference.py:23-27` pins. `generate_golden_reference` (`reference.py:210`) knows only `encoder_bert` / `decoder_gpt2` (`reference.py:178`) | `pattern/reference.py` (new variant) + a new boundary tuple | 2 |
| C4 | A `decoder_llm` stage-atol table and the `1e-1` ceiling re-measured at real-weight scale (`opcheck_layer.py:164`, `:219`, `:248`) | `opcheck_layer.py` | 2 |
| C5 | `dtype` / `quant_*` producer: `run_mode.py:272` writes `row["dtype"] = "bf16"` unconditionally and no `quant_*` column is populated (schema fields at `study/schema.py:131`, `:272-283`) | `study/run_mode.py`, `pattern/*/` | **6** |
| C6 | SPECS rows + lit recipes per (mode, family) — `opcheck_specs.py:134` list, and one `run_npu2_<mode>_peano.lit` each | `opcheck_specs.py`, `*.lit` | 2 |
| C7 | Pin tests for every private shipped symbol imported (`_build_rope_2d`, `_build_qknorm_2d`, `_build_add_2d_to_2d`, `_build_gemm_module`), per doc 58:66-68's divergence mitigation | `study/` host test | risk 5 |

### 2.1 `coarse` — gaps 1, 2, 5, 6

`coarse` dispatches `builders/block.py`'s artifacts through `prepare_layer_dispatch`
(`pattern/coarse/coarse.py:108-123`), so a Qwen3 layer here means a Qwen3 `block_config` /
`run_block` sibling. New/changed:

1. **`block_config`'s MHA equality must go.** `builders/block.py:296-301` raises when
   `num_heads * head_dim != emb_dim`. Qwen3-0.6B is 16 × 128 = 2048 ≠ 1024. Not a one-line
   deletion: the docstring at `:284-286` says the fused QKV weight, the attention layout and the
   output projection all assume it, and all three are true.
2. **Three separate QKV GEMMs, not the fused split-cast.**
   `builders/qkv_proj.py:267 build_qkv_proj_module(seq_len, emb_dim, ...)` hardcodes
   `n_total = 3 * emb_dim` (`:295`) and an equal three-way C split. Qwen3's QKV is
   2048 + 1024 + 1024 = 4096 over emb 1024 — 4× emb, unequal. (The registry *does* hold
   `M×1024×4096`, so the shape is measured; the **split** is what the builder cannot express.)
3. **A rectangular O projection.** `builders/o_proj.py:68 o_proj_gemm_spec(seq_len, emb_dim, ...)`
   and `builders/mha_out_proj.py:233` type `w_o` as `memref<emb_dim x emb_dim>`. Qwen3's is
   (2048, 1024).
4. **Attention: F2.** `mha_out_proj_config` (`builders/mha_out_proj.py:161`) →
   `attention_config` (`builders/mha_attention.py:137`) builds the seq-first design only
   (`mha_attention.py:13-16`, `:291`). See §4.1 for the three options and their costs.
5. **RMSNorm replaces `addnorm`.** `builders/addnorm.py:280 build_addnorm_module(...)` is a fused
   LayerNorm + residual with `addnorm_max_rows` row-blocking (`:233`) — 80 rows at emb 1024 per
   `builders/block.py:255`. A Qwen3 layer's norms are RMSNorm over whole tensors
   (`weighted_rms_norm.build_module`, no L1 row cap of that kind), so `norm_rows` /
   `NORM_ROW_MARGIN` / the four-sequence structure (`block.py:41-77`) is **replaced, not adapted** —
   which is a simplification, and removes coarse's 402 sync boundaries
   (`pattern/fused/fused.py:58-61` records that number).
6. **SwiGLU FFN.** `builders/ffn.py:176 build_ffn_module(seq_len, emb_dim, ffn_dim)` is up → GeLU →
   down. Qwen3 is gate + up → SiLU·mul → down: three GEMMs plus `silu_and_mul.build_module_2d`.
7. **A `run_decoder_llm_block`** boundary list and dispatch sequence beside
   `run_block` / `run_decoder_block` (`builders/block.py`), plus a `BLOCK_INPUT_NAMES` sibling of
   ~13 entries (x, attn_norm_w, wq, wk, wv, q_norm, k_norm, wo, ffn_norm_w, w_gate, w_up, w_down,
   rope_lut).
8. Plus C1–C7.

### 2.2 `offload` — gaps 1, 2, 5, 6 (**the cheapest of the four for a real model**)

`offload` is "linear ops on device, everything else host" (`pattern/offload/offload.py:30-34`,
`:888-891`: for the decoder its pre-norms and residual adds are already host arithmetic and **no
device artifact changes**). For Qwen3 that means:

1. **Seven device GEMM dispatches instead of six.** `OFFLOAD_GEMMS` (`offload.py:233-240`) is
   q/k/v/output/up/down; SwiGLU adds `gate_proj`. All seven shapes resolve — see §4.3.
2. **Host RMSNorm, QK-norm, RoPE, SwiGLU** in numpy/torch beside the existing `_host_addnorm`
   (`offload.py:760`), `_host_softmax_bf16` (`:782`), `_host_gelu` (`:858`). The reference
   implementations exist and are importable: `llms/qwen3_0_6b/qwen3_0_6b_cpu_helpers.py:30
   qk_norm_per_head(x, weight, n_heads, head_dim, eps=1e-6)` and its siblings.
   **This is the whole point:** offload needs **zero new device builders** for the non-linear parts.
3. **New attention GEMM tiles at head_dim 128.** `ATTENTION_GEMM_TILES` (`offload.py:272`) holds
   two measured, *injected* specs at K = 64 — the comment at `offload.py:254` says
   "tk2=64 is FORCED: K=64 admits no other L2 tile". Qwen3 needs `attn_scores` (seq, 128, seq) and
   `attn_output` (seq, seq, 128). These are injected through `gemm_spec_fn` and recorded as
   `gemm_spec_source: injected` (`offload.py:76-80`), so **no registry write is required** — but a
   measurement is, to keep the mode honest.
4. **GQA broadcast on the host**: 16 Q heads over 8 KV heads, a slice-index map. Trivial.
5. **The `SHARED_XCLBIN` drain pin is a refusal at M = 2048** — §4.3.
6. Plus C1–C7 (minus C7's device-leaf pins, which offload does not use).

### 2.3 `runlist` — gaps 1, 2, 5, 6 (**the most work, but no new risk**)

`runlist` is "every operator its own device kernel, nothing on the host"
(`pattern/runlist/runlist.py:106-112`). For Qwen3 that is one device artifact per node of
`decoder_graph`:

1. **RMSNorm ELF** (whole-tensor, from `weighted_rms_norm.build_module` wrapped by
   `_wrap_ir_in_launch`) — replaces the banded `layer_norm` + `elementwise_mul` chain
   (`runlist.py:839 run_norm_chain`).
2. **QK-norm ELF** ×2 (`_build_qknorm_2d`).
3. **RoPE ELF** ×2 (`_build_rope_2d`).
4. **SwiGLU ELF** (`silu_and_mul.build_module_2d`).
5. **Seven GEMM artifacts** (q/k/v/o/gate/up/down), each its own compile — the mode's
   no-re-execution rule (`runlist.py:78-89`) means one artifact per role, not one shared module.
6. **Attention: the same head_dim-128 GEMM tiles as offload** (runlist imports offload's specs,
   `runlist.py:38-40`), plus the per-head causal mask add and device softmax
   (`builders/softmax.py:169`) — 16 heads → 16 submissions (`runlist_submission_count`,
   `runlist.py:414`).
7. **Its own MHA equality raise** at `runlist.py:287-291` must go, same as coarse's.
8. Plus C1–C7.

`runlist` is the mode whose *definition* survives the port most cleanly: its finest-granularity
claim is unaffected by RMSNorm/SwiGLU/RoPE, and it never wanted a device-resident attention path.

### 2.4 `fused` — gaps 1, 2, **5**, 6 (**partially refused before any code**)

`fused` is one runlist over three ELFs, forced single-submission
(`pattern/fused/fused.py:8-16`). For Qwen3:

1. **The `fused_tail` stitch is LayerNorm-shaped.** `build_fused_tail_module` (`fused.py:297`) and
   the `norm_tail` pipeline it composes (`builders/norm_tail.py`, via `fused.py:361-363`) implement
   `gamma * LN(x) + residual`. A Qwen3 tail is RMSNorm + SwiGLU + a second RMSNorm — a **new
   stitch**, not a parameterisation.
2. **The packed-plane cap**: §4.2.
3. **The three-ELF split is set by the FA-vs-GEMM `runtime_loop_tiling_sizes` conflict**
   (`fused.py:63-80`): FlashAttention needs `[1, 1]`, the GEMMs `[2, 2]`, and `[2, 2]` on
   `mha_out_proj` **hangs**, `ERT_CMD_STATE_TIMEOUT` 3/3 against 3/3 clean at `[1, 1]`
   (also recorded at `builders/block.py:100-107`). That conflict is unchanged by the model, so
   `fused` stays ≥ 3 ELFs for any family. §4.4.
4. **F2 lands hardest here.** Two host transposes inside a mode defined as *no intermediate host
   sync* (`fused.py:8-16`) is not a cost, it is a change of what the mode is.
5. Plus C1–C7.

---

## 3. The weights-injection seam (gap 1), in detail

### 3.1 What `prepare_layer_dispatch` requires

`opcheck_layer.py:429-431`:

```python
def prepare_layer_dispatch(shape, seed=42, cache_dir=BLOCK_CACHE_DIR, label="block", extra=None):
```

It takes six shape scalars out of `shape` (`:462-465`), reads `workload_variant` from the same dict
(`:469`), calls `block_config(...)` (`:475`) and then `generate_golden_reference(...)` (`:478-480`)
— which **is** the weights source. `weights = golden["weights"]` (`:481`) and the ordered
`inputs` list is built from it at `:484-492`. The returned contract is
`{"inputs", "expected", "inject", "dispatch", "record_extra"}` (`:559-568`), plus `"atol"` for the
decoder (`:576`).

The four pattern preparers repeat exactly this shape: `prepare_coarse` delegates
(`pattern/coarse/coarse.py:108`); `prepare_offload` (`pattern/offload/offload.py:872`) re-derives
its own nine-entry list at `:905-915`; `prepare_runlist` (`pattern/runlist/runlist.py:985`) at
`:1017`; `prepare_fused` (`pattern/fused/fused.py:788`) at `:818-826`.

So the minimal seam is: **an optional `weights` argument threaded through five preparers**, each of
which stops calling `generate_golden_reference` for the tensors and starts calling it for the
input activation and the per-boundary reference only. That is genuinely small. What makes it more
than a parameter is everything below.

### 3.2 The four coupling points, in order of sharpness

**(a) `inputs` is a positional LIST, and each mode has its own names tuple.**
`builders/block.py:169-177` `BLOCK_INPUT_NAMES` (7 entries, fused `w_qkv`);
`pattern/offload/offload.py:307-319` `OFFLOAD_INPUT_NAMES` (9, q/k/v **separate**);
`pattern/runlist/runlist.py:190-...` `RUNLIST_INPUT_NAMES` ("Identical to offload's");
`pattern/fused/fused.py:192-...` `FUSED_INPUT_NAMES` ("Identical to the block's"). The block
docstring states the reason the list is a list rather than a dict
(`builders/block.py:24-27`): *"`opcheck.py`'s fault injection perturbs `prepared["inputs"][i]`;
the order below is therefore part of the contract between this module and `opcheck_specs.py`"*.
Each `dispatch` closure then **unpacks positionally** — `offload.py:958`
(`x, w_q, w_k, w_v, w_o, ln1_weight, w_up, w_down, ln2_weight = device_inputs`),
`fused.py:819`. A Qwen3 layer's list is ~13 entries, so all four tuples, all four unpackings and
all four `inject` indices move together.

**(b) The injection index is derived, which is the good news.**
`opcheck_layer.py:565` is `"inject": (BLOCK_INPUT_NAMES.index("ln1_weight"), (0,))` — an index
*computed from the names tuple*, and the other three modes do the same
(`fused.py:1230`). So adding weights does not silently mis-index **provided the names tuple is
extended in the same commit**. That is the one thing a host test should assert:
`len(inputs) == len(NAMES)` for every mode, and `NAMES.index(target)` resolving.

**(c) The injection must still make the run FAIL, and that is a property of the weight
distribution, not of the code.** `opcheck.py:255-270 _inject(inputs, where, delta=FAULT_DELTA)`
adds `FAULT_DELTA = 2.0` (`opcheck.py:148`) to one element of one device input. That constant was
sized against the **generated** draws: `pattern/reference.py:131 VAL_RANGE = 0.05` scales the
`randn` weights and the norm weights are `torch.rand` on [0, 1)
(`reference.py:30-33`). `opcheck.py:145-147` says the choice is "two orders of magnitude above the
tolerance band **at these input scales**". Real Qwen3 RMSNorm weights are O(1) and real activations
are not `randn * 0.05`. The negative control (`opcheck.py:414-451`) requires the injected run to
FAIL and treats a PASS as proof the check is not reading the device's inputs — so **an injected
real-weight run that happens to round back into the band turns a working gate into a red one, and
the fix is a re-measured `FAULT_DELTA` / target, not a widened tolerance.** This is the
measurement M1 must actually take; it is not implied by "bit-identical to the generated-weight
run", because a bit-identical run is by construction one that injected the *same* arrays.
The measurement that chose `ln1_weight` in the first place is recorded at
`opcheck_layer.py:164`+ and `pattern/offload/offload.py:875-880` ("every attention-side candidate
puts ZERO elements outside the band"); the equivalent for a real-weight Qwen3 layer is unmeasured.

**(d) The content key must be over CONTENT.** Weights are declared `static=True` with a
`content_key` (`builders/block.py:114-117`; `pattern/runlist/runlist.py:99-103`;
`pattern/fused/fused.py:129-131`, each saying "under fault injection the key changes and the
perturbed weight is re-uploaded"). `content_key(buf)` is `sha256:<hex>` of the bytes
(`llms/shared/infra/bo_pool.py:87`, `:517`) and `content_key_once(buf)`
(`bo_pool.py:557-564`) caches **by `id(buf)` with an identity re-check**. `_inject` builds a fresh
array (`opcheck.py:262-266`), so today the key necessarily changes.
**The trap M1 can walk into:** a weights-injection design that keys a real weight by a *name*
("layer0.wq") or that mutates a loaded array in place would leave the cached key intact, the pool
would reuse the clean BO, and the injected run would **PASS** — the exact outcome
`_negative_control_verdict` (`opcheck.py:414`, `:436`) is written to catch, arriving as a red gate with
a confusing message. `bo_pool.py:567 forget_content_key(buf)` exists for the in-place case and its
docstring (`:553`) says so. Also note the operator's standing rule "S1 content key once per plan":
hashing 13 real weight tensors per dispatch is setup-time cost, outside the clock
(`study/run_mode.py:300-311`), and `content_key_once` already amortises it — but only if the
weight arrays are long-lived objects, which a per-dispatch `np.asarray(...)` would defeat.

### 3.3 What M1's gate should therefore be

Doc 58:48 says "**bit-identical** to the generated-weight run; existing lits green". Correct and
necessary. **Add one clause:** the negative control must be re-run under injected weights at the
real weight scale for at least one mode, and `FAULT_DELTA` / the injection target re-recorded if it
no longer trips. Without that clause M1 can pass while leaving the gate that makes every later
phase meaningful silently vacuous.

---

## 4. Structural refusals (gap 5), before any code is written

Shapes used throughout: Qwen3-0.6B `emb_dim = 1024`, `n_heads = 16`, `n_kv_heads = 8`,
`head_dim = 128`, `hidden_dim = 3072` (`llms/shared/plan/graph.py:221-225`), hence
`q_dim = 2048`, `kv_dim = 1024` (`graph.py:45-51`).

### 4.1 Attention at head_dim 128 — the hard one (`coarse`, `fused`)

- `builders/mha_attention.py:137 attention_config(seq_len, head_dim, num_heads, num_kv_heads=None, causal=False, parallel_seq=256, parallel_heads=2, kv_seq_tile=None, cascade_stages=4, num_q_tiles=None)`
  defaults `kv_seq_tile = head_dim` (`:176`) and **raises if they differ** (`:182-189`), with the
  reason given at `:77-83`: *"`head_dim = 128` FlashAttention has been flaky (hang or NaN) on some
  NPU2 setups … This operator's case matrix is `head_dim = 64` throughout, so `attention_config`
  rejects anything that would put it on the `dv_chunks > 1` path"*.
- The design it composes is seq-first (`:15-16`, imported at `:291`), and
  `attn_npu2_seqfirst.py:116-125` computes `dv_chunks = dv // lkp` and asserts it is 1, telling the
  caller to *"Use attn_npu2.py for the dv_chunks > 1 / heads-first layout"*.
- The two escape routes and their arithmetic:
  - **`kv_seq_tile = 128`** satisfies both asserts formally (`dk_chunks = dv_chunks = 1`), and the
    other constraints hold at seq 512/1024: `seq % parallel_seq(256) == 0`;
    `seq % (kv_seq_tile × cascade_stages) = seq % 512 == 0`;
    `num_q_tiles = 256 // 128 = 2`; columns `= num_q_tiles × parallel_heads = 4 ≤ 8`
    (`mha_attention.py:217-221`); causal's `q_seq_tile == kv_seq_tile` holds (128 == 128,
    `:228-234`); `num_heads % num_kv_heads = 16 % 8 == 0` (`:210-214`). **INFERENCE:** it will not
    place — at lkp 128 the K, V and score tiles are 128×128 bf16 = 32 KiB each against one 64 KiB
    L1 tile (the figure `study/profiles.py:277-282` uses for the softmax bound), so three of them
    cannot coexist. **This is an inference, not a measurement**, and it is the cheapest thing M2
    can falsify: one compile attempt, no device time.
  - **Heads-first + two host transposes** is the shipped answer
    (`llms/shared/infra/fa_headfirst.py:7-13`, `:114 npu_fa_headfirst(...)`, transposes at `:140`
    and `:172`). It works — the registry records Qwen3-0.6B's own 2048×2048, 128/128, 16q/8kv row
    passing at 3.8e-2 (`kernel_registry/details/FlashAttention_bf16.md:158`) — but the same note
    (`:174`) warns *"long-sequence `head_dim=128` FA has been flaky (`ERT_CMD_STATE_TIMEOUT` /
    NaN) on some NPU2 setups"* with `cpu_attn` as the documented fallback.
- **Verdict.** `(coarse, Qwen3-0.6B)` and `(fused, Qwen3-0.6B)` are **bounded, not impossible**:
  they are reachable only through the head-first kernel and only by admitting two host transposes
  into the mode's dataflow. For `fused` that contradicts the mode's own definition
  (`pattern/fused/fused.py:8-16`, "one host submission"; `:54-61`, every intermediate
  device-resident). **Recommendation: record `(fused, Qwen3-0.6B)` as a structural refusal with
  this reason, and record `(coarse, Qwen3-0.6B)` as measured-with-a-caveat, the transposes reported
  in `host_cpu_ms`.** Neither is a workaround; both are results, which is what doc 58:50 asks M3
  to produce.

### 4.2 `fused`'s packing cap — arithmetic at emb 1024

`study/profiles.py:226 FUSED_PLANE_STRIDE_CAP = 2**20` = 1,048,576, from the shim `aie.dma_bd`
plane-stride field; `study/profiles.py:302` picks the largest ladder point with `s * emb ≤ cap`;
`pattern/fused/fused.py:40-44` is the mechanism ("`packed1`/`packed2` are PLANE-MAJOR … That caps
the mode at rows*cols <= 2^20 (1024 rows at emb 768)").

| Family | emb | cap / emb | ladder points inside | `fused_seq_range` |
|---|---|---|---|---|
| Qwen3-0.6B | 1024 | 1024 rows | 256, 512, 1024 | (256, 1024) |
| Llama-3.2-1B | 2048 | 512 rows | 256, 512 | (256, 512) |

`FUSED_SEQ_MIN = 256` (`profiles.py:232`) is a tiling floor and does not move with width.
**Consequence for Qwen3-0.6B: `fused` refuses seq ≥ 2048.** Combined with §4.3, `fused`'s only
legal Qwen3-0.6B rungs are **512 and 1024** — and 512/1024 is also where the registry is complete,
so the cap and the registry agree for once.

### 4.3 The registry method asymmetry — `offload`'s single-launch/shared-xclbin bound

`offload`'s shared xclbin is *"bounded to SINGLE-LAUNCH modules by the platform"* — a multi-launch
xclbin compiles and chains but its in-stream `load_pdi` faults the NPU firmware
(`pattern/offload/offload.py:407-419`, fatal_error_type 0x10). `fused-cast` is two launches by
construction, so `_chain_spec` **re-resolves every fused-cast winner to the shape's measured
`drain` row, and the registry raises if none exists** (`offload.py:420-423`).

Measured coverage of Qwen3-0.6B's seven prefill GEMM shapes
(`kernel_registry/details/GEMM_bf16_in_bf16_out.json`; rows for M=512/1024 landed by commit
`0f1cedd7`, "the six Qwen3-0.6B prefill GEMM rows at M = 512, 1024"):

| Role | (K, N) | M=256 | M=512 | M=1024 | M=2048 |
|---|---|---|---|---|---|
| Q proj | 1024 → 2048 | **absent** | fc / drain / direct | fc / drain / direct | **fused-cast only** |
| K/V proj | 1024 → 1024 | fc / drain / direct | fc / drain / direct | fc / drain / direct | **drain only** |
| O proj | 2048 → 1024 | **absent** | fc / drain / direct | fc / drain / direct | **fused-cast only** |
| gate / up | 1024 → 3072 | fc / drain / direct | fc / drain / direct | fc / drain / direct | **fused-cast, direct (no drain)** |
| down | 3072 → 1024 | **absent** | fc / drain / direct | fc / drain / direct | **fused-cast only** |

Raise texts, from `kernel_registry/registry_lookup.py:115-121` (shape absent) and `:137-141`
(method absent: *"gemm_config_method: shape MxKxN (out=bf16) has no method '<m>' … (available:
[…])"*), surfaced by `transformer_layer/builders/gemm_spec.py:112 resolve_gemm_spec(m, k, n, output_dtype="bf16", precision="high", method=None)`.

**Verdict.**
- `offload` at Qwen3-0.6B **M = 2048 is a refusal today**: four of five shapes have no `drain` row
  and the shared-xclbin path pins `drain`. (`AIR_OFFLOAD_LEGACY_ELF=1`, `offload.py:206`, escapes
  the pin at the cost of 30 reconfigurations per layer — i.e. it measures a different mode.)
- **M = 512 and M = 1024 are the only rungs where all four modes are shape-legal** for
  Qwen3-0.6B: every method present, `fused`'s cap satisfied, `seq % FA_PARALLEL_SEQ(256) == 0`
  (`profiles.py:247`), `seq % ATTN_GEMM_SEQ_MULTIPLE(512) == 0` (`profiles.py:264`), and
  `softmax_fits_l1(seq)` comfortably (`profiles.py:284-286`: the ceiling is 10,922 columns).
- **Filling the M=2048 holes is itself blocked.** The sweep writer is append-only:
  `sweep/registry_writer.py:196` raises `ShapeAlreadyRegistered` for a `(M,K,N)` already
  present, and `add_missing_methods` skips rows whose `used_by` belongs to another owner
  (`registry_writer.py:246`). The M=2048 keys are owned by the shipped-model rows. So "sweep
  the missing drain rows" is a change to the writer's ownership rules, not a sweep.

### 4.4 The FA-vs-GEMM `runtime_loop_tiling_sizes` conflict — model-independent

`pattern/fused/fused.py:64-80` and `builders/block.py:100-107`: FlashAttention requires
`runtime_loop_tiling_sizes=[1, 1]`; the GEMM-backed artifacts are built at `[2, 2]` for BD-ID
recycling; `[2, 2]` on `mha_out_proj` **compiles and then hangs** — `ERT_CMD_STATE_TIMEOUT` 3/3 at
4096 against 3/3 clean at `[1, 1]`. Two corrections travel with it and are worth repeating so
nobody re-derives them: `omit_pingpong="all"` is **not** part of the conflict, and the lowered IR
is **identical op-for-op** between the two settings, so a compile-only comparison "refutes" this
and is wrong to (`fused.py:81-88`). This is a property of the backend, not of the model, so it
survives the port unchanged: **`fused` cannot be one ELF for any family**, and the mode's floor is
three ELFs (four for the decoder, `fused.py:803`).

### 4.5 Llama-3.2-1B, for contrast (doc 58:54's second family)

`emb 2048, n_heads 32, head_dim 64, n_kv 8, hidden 8192` (`graph.py:227-230`).
- `num_heads × head_dim = 2048 == emb_dim` → `block_config`'s raise (`block.py:296`) and
  `runlist_config`'s (`runlist.py:287`) **pass unchanged**.
- head_dim 64 → the seq-first FA the study already composes is the right kernel; **F2 does not
  apply**.
- Registry: at M = 2048 all of (2048,2048), (2048,512), (2048,8192), (8192,2048) carry all three
  methods. At M = 256/512/1024 **only (M, 2048, 512) exists** — every other Llama-1B shape is
  absent below 2048.
- `fused`'s cap at emb 2048 is 512 rows (§4.2). **So `(fused, Llama-3.2-1B)` is a refusal by
  construction**: the only sequences it can pack are ones the registry has not measured, and the
  only sequence the registry has measured is one it cannot pack.
- Still needs everything in §2's C-list and the RMSNorm / SwiGLU / RoPE / unequal-QKV-split work
  (Llama's QKV is 2048 + 512 + 512 = 3072 ≠ 3 × emb).

---

## 5. Revised estimate per phase

**Calibration.** One "agent-session" here = one queue item as this branch has actually run them:
read + prediction + implement + device measure through devq + one Codex round + one commit. From
`git log`: items 16–20 landed 2026-08-26 (five in a day, on scaffolding that already existed);
item 13 (H1a: model adapter + `run_model.py` + schema v3 + a two-model curve) was one session on
2026-08-23; items 9's four increments took most of 2026-08-22. So a session is roughly a day, and a
phase that needs new device bring-up is 2–4 sessions, not one.

| Phase | Doc 58's implied size | **M0's estimate** | Why |
|---|---|---|---|
| **M1 — the seam** | "days" (gap 1 row, doc 58:20) | **1 session** (**cheaper**) | Five preparers gain an optional `weights=`; the injection index is already derived from each mode's names tuple (§3.2b) and the content key is already content-derived (§3.2d). The one real task is the re-measured negative control (§3.3). No device build changes, so the lits are re-runs. |
| **M2 — one layer, one mode** | one phase | **3–5 sessions** (**more expensive**) — and see §6, the mode should not be `coarse` | Six new study builder modules (§1.5), a new golden model + boundary tuple + atol table (C3/C4), the family-shape plumbing that today derives `head_dim` (C2), the MHA-equality raise, the unequal QKV split, the rectangular O proj — **and, if the mode is `coarse`, the head_dim-128 attention question (§4.1), which is the only item in the program with no upper bound.** |
| **M3 — one layer, four modes** | one phase | **3–4 sessions** (**as assumed, if M2 lands the leaves**) | Once M2 has the six leaves and the oracle, `runlist` is assembly (2.3) and `offload` is host code (2.2). `fused` is a new tail stitch plus a refusal to write up. The per-mode lits and SPECS rows are mechanical. |
| **M4 — the model** | "weeks" (gap 3 row, doc 58:22) | **2–3 sessions** (**substantially cheaper**), *if scoped as a swap* | Doc 58:22 prices an N-layer loop, embedding, LM head, KV cache and sampling as new. **They are not new.** `llms/shared/model_adapter.py:1-8` is a narrow interface over the shipped drivers' `prepare_runtime` / `run_npu_prefill` / `run_npu_decode_step`, and `study/run_model.py` already walks it under the study's discipline with the production `verify_against_hf` gate (`model_adapter.py:835`). M4 is "make a mode the layer executor inside that loop", not "write the loop". **Risk that flips this back to weeks:** the drivers' BO pool and static weight residency (`opt-buffer-object-reuse`'s `static_input_indices`) and chunked prefill's rectangular FA (`plan.py:328-334`) are properties of the driver's assembly, and a mode that re-assembles the layer may not inherit them. |
| **M5 — the matrix** | one phase | **1–2 sessions**, device-time bound | Two walks + `compare_roots` + per-artifact-set verify is existing machinery (`study/run_profile.py`, `compare_roots.py`, `manifest.py`). Cost is device queue time, not code. |

**Total, one family, prefill only: 10–15 agent-sessions ≈ 2–3 weeks at this branch's observed
cadence.** Doc 58:73-75's "weeks-to-months per mode per family" is replaced by:
**weeks per family for all four modes**, with the caveat that two of the sixteen (model, mode)
cells named here are refusals rather than measurements.

### The three biggest uncertainties, named

1. **Attention at head_dim 128 (§4.1).** The only unbounded item. If the `kv_seq_tile = 128`
   seq-first configuration places (my inference says it will not), `coarse` and `fused` are cheap.
   If it does not, `coarse` costs two host transposes, `fused` is a refusal, and someone has to
   decide whether a mode that transposes on the host is still that mode. **Falsifiable in one
   compile, no device time — do this first.**
2. **The correctness gate at real-weight scale.** `opcheck_specs.py:834-851` records that the
   whole-layer `atol` is `1e-1`, the **hard ceiling**, at a **1.35× margin**, and that exceeding it
   "is a defect report or a smaller `val_range`, not a wider tolerance". A real-weight Qwen3 layer
   has a different dynamic range from `randn × 0.05`, and there is nowhere to pad to. If the
   ceiling is exceeded, M2 stalls on a numerics question, not an engineering one. (The shipped
   drivers' own bound is different — the FA registry row is 3.8e-2 mean_rel_L1,
   `FlashAttention_bf16.md:158` — so the two gates disagree about what "correct" means and M2 must
   pick one and say which.)
3. **Whether M4 is a swap or a rewrite.** The optimistic estimate above rests on
   `model_adapter`/`run_model` carrying the model-level pieces. Its footguns
   (`model_adapter.py:47-62`: one `ModelSession` per process; `prepare` refuses rather than
   compiles; the drivers compile for one prefill M) are compatible with a mode-swapped layer in
   principle and untested in practice.

### One correction to doc 58's own table

Doc 58:23's gap-4 row says the registry is "**partly done for Qwen3-0.6B** … M 2048 already
present". True and misleading: the M=2048 rows are present but **single-method** (§4.3), which is
precisely the case `offload`'s `drain` pin and any fused-cast-pinned builder cannot use. The rows
that make the program work are the **M = 512 / 1024** rows item 13 added, not the M = 2048 ones.

---

## 6. Recommendation

### 6.1 The cheapest order that produces a real measurement earliest

**Step 0 (hours, no device): falsify the head_dim-128 seq-first configuration.** One
`attention_config(seq_len=1024, head_dim=128, num_heads=16, num_kv_heads=8, causal=True)` +
`build_attention_ir` + an aircc compile. Two outcomes, both cheap, and the answer reorders
everything after it. This is not a phase; it is the first hour of M1.

**Step 1 — M1 as written**, plus the re-measured negative control (§3.3). 1 session.

**Step 2 — M2 on `offload`, not `coarse`.** Doc 58:49 chose `coarse` because it is "simplest,
fastest, and its registry rows exist". For a *synthetic* layer that is right. For a *real Qwen3
layer* it is backwards: `coarse` needs the entire device-side re-assembly **and** owns the one
unbounded item, while `offload` needs **zero new device builders** for RMSNorm/QK-norm/RoPE/SwiGLU
(they become host numpy, which is the mode's definition — `offload.py:888-891`), seven registry
GEMMs that all resolve at M = 512/1024, and two injected attention-GEMM specs. **`offload` at
Qwen3-0.6B, seq 1024, prefill, one layer is the first real (model, mode) cell this program can
produce, and it is reachable without touching attention's layout problem at all.** Its host
reference implementations are already written and importable
(`llms/qwen3_0_6b/qwen3_0_6b_cpu_helpers.py:30`+).

**Step 3 — M2b: `coarse` on Qwen3-0.6B**, carrying step 0's answer, with the six leaf wrappers.
This is where the device-side re-assembly lands and where every later mode's builders come from.

**Step 4 — M3: `runlist`, then `fused`.** `runlist` reuses M2b's leaves at finer granularity and
carries no new risk. `fused` is attempted and, on present reading, **recorded as a refusal** for
Qwen3-0.6B (§4.1) — which doc 58:50 already asks for.

**Step 5 — M4 scoped as a swap** (§5), and **M5 as written**.

### 6.2 What to cut or merge

- **Merge `offload` out of M3 and into M2.** It is the cheapest cell and it de-risks the phase
  gate: M2 then delivers a real measurement even if step 0 goes badly.
- **Cut "sampling" from M4's scope, and say so.** Doc 58:58-62 already argues the LM head,
  embedding and sampling sit *outside* the mode in every phase. Then M4 should not list them as
  things to build — it should list "the mode's layer, inside the shipped driver's loop, gated by
  the driver's own verify". That is both cheaper and more honest about what the number means.
- **Do not defer Llama-3.2-1B to "after M4" (doc 58:54).** It is head_dim 64 and satisfies the MHA
  equality every study builder asserts (§4.5), so it is the **cheaper family for `coarse`**, exactly
  where Qwen3-0.6B is dearest — while being a refusal for `fused` and having no registry rows below
  M = 2048. Consider running `(coarse, Llama-3.2-1B, seq 2048)` **as part of M2b**, as the control
  that separates "the mode cannot execute a real decoder layer" from "the mode cannot execute
  head_dim 128".
- **Do not attempt a decode / M = 1 cell in this program.** §1.4: there is no GEMV registry and no
  GEMV resolver; `resolve_gemm_spec` cannot express M = 1, and the M = 1 shapes are a different
  kernel family reached only through the shipped drivers. A decode mode cell is a separate program,
  not a column of this matrix.
- **Add one item doc 58 does not have: the symbol pins (C7).** Four private shipped symbols become
  load-bearing for the study the moment M2 lands. Doc 58:66-68 names the mitigation ("pin the
  imported symbols in a host test") but no phase owns it. Give it to M2, where the imports appear.

### 6.3 The one thing that would change this recommendation

If step 0 shows the seq-first kernel places at `kv_seq_tile = 128`, then `coarse` and `fused`
become ordinary ports, `(fused, Qwen3-0.6B)` stops being a refusal, and doc 58's original
M2-on-`coarse` ordering is right after all. That is a one-compile question and it should be
answered before the operator commits to M1's successor.
