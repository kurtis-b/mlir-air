# 56 — Plan: an analytical planner, and the study refactored to run full models with mixed precision

`[2026-08-20]` The operator set two focus items after [55](55-hexagon-llama-cpp-lessons-for-xdna2.md):

1. **Automate inference optimization** in this framework with llama.cpp's `ggml-hexagon` backend
   as the reference — specifically the way it reduces overheads (batched dispatch, resident
   buffers, cached plans) and decides resource use and fusion **analytically** (capacity-driven
   legality cascades and a reload-count model, never a measured table per model).
2. **Refactor the "IRON study"** (`programming_examples/transformer_layer/study`, today: one
   transformer layer, four modes, bf16, prefill only) so it runs **full models with mixed
   precision** modeled on `ggml-hexagon`, measuring prefill tok/s against `ubatch` and decode
   tok/s, and keeping the study's discipline (schema'd CSVs, dispatch vector, derived skips,
   manifests, resume, walk-to-walk comparison, verify gate against HF).

This is the plan, after one round with Codex: the first draft was written from a subagent's
inventory of the repository, then sent to Codex for an adversarial read together with doc 55
(56a (the verbatim Codex review, retired 2026-08-22 to git tag `pre-cleanup-20260821`), verbatim). §7 lists what it changed. §1 restates the
reference precisely enough to copy; §2 the inventory; §3 the architecture; §4 the phases with
gates; §5 the first measurable milestone; §6 what was cut. Citations are `file:line` at
`debf9be2`. Targets are stated as targets, never as results.

## 1. The reference, reduced to what we copy

From [55 §4](55-hexagon-llama-cpp-lessons-for-xdna2.md) as corrected, and the source at
`ggml-org/llama.cpp@6503355df0eb`:

| Hexagon mechanism | What it actually is | Copy as |
|---|---|---|
| **Op batching** (`fit_op`: ≤ 1024 ops, ≤ `HTP_OP_MAX_BUFS` buffers, bounded tensor descriptors, session vmem; one `dspqueue_write` per batch; ≤ 16 in flight) | Whole-token work in a few round trips; `fit_op` does not test VTCM — the per-op planners do | one `xrt.runlist` per token via `dispatch.run_sequence(require_single_submission=True)` (exists); **one submit ≠ one context ≠ one image** — three targets, staged (§4 H3) |
| **Plan cache by graph uid** (one cached slot per session) | Kernel params and fusion computed once per shape | `Plan` is a content-hashed JSON; compiled artifacts keyed by plan hash, not by name |
| **Resident, repacked weights** (`rpcmem` + `fastrpc_mmap` once; repack in `buffer_set_tensor` to padded 32×32 tiles) | "Resident" = mapped DDR; VTCM holds only tiles in flight | `static_input_indices` + `bo_key` (exists); repack = the AWQ/q4_0 packers (exist) |
| **Kernel-family cascade** (`MM_SELECT`: HMX iff `M > 4` ∧ K%32 ∧ padded N%32 → HVX tiled → HVX flat → CPU) | Legality first, capacity second, fixed order | candidate providers per op family; **no hard `M` threshold** — `M > 4` is an HMX/HVX fact, not an XDNA2 one; choose by legality then by the traffic model |
| **Capacity solvers** (HMX: `compute_chunks` minimizes `mblocks·N·3 + nblocks·M·2` reload counts under VTCM, largest chunk on ties, `pipeline` iff `M > 32`; HVX: deepest prefetch 16 → 2 that fits) | **Traffic** models, not latency models | the same objective over L1/L2 reloads with the builders' bounds lifted into pure functions, **spatially** (per-column L2, channels, BDs); the measured registry is the override table |
| **Fusion by budget** (`try_fuse_node`: quantized QKV / gate-up and matmul+add guarded by "VTCM needed ≤ budget"; `RMS_NORM+MUL` unguarded) | Fusion is mostly a capacity question | the seven multi-launch builders *are* the fusion patterns; the planner picks which applies by the budget rules the README states by hand |
| **`supports_op` + `OPFILTER`** | `(op, type, shape) → device \| other backend`; split count is the diagnostic and the split is not free | a `placement` predicate `device \| host \| refuse` with `host_ops` and boundary bytes recorded per row; study default `refuse_on_unplanned_split` |
| **Power voted at `start()`** | DCVS off, corners MAX | Turbo refusal (exists) + pmode recorded on every row (exists) |
| **Mixed precision** (weights Q4_0/Q4_1/Q8_0/IQ4_NL/MXFP4 repacked; matmul in f32/f16, out f32; HVX quant path: f32 → Q8_0/Q8_1 activations in VTCM; HMX path: weights → fp16 tiles, f32 activations → fp16 tiles; norms/softmax f32; KV dtype a run-time choice, FA requires F16 K/V) | **Separate storage type, scratch compute type, accumulator type and model-visible type** — not "int4 × int8 everywhere" | a `precision_plan` with those four fields per tensor class; phase 1 = audit the existing int4 decode path, bf16 or `bfp16ebs8` for prefill; on-device activation quantization last |
| **`ubatch`** (`n_ubatch` default 512; the reference scripts run `--ubatch-size 1024`; `M` of every prefill matmul; each ubatch = one `graph_compute`) | Throughput rises with `M` while `U ≤ m_chunk`, flattens once VTCM caps the reusable row tile; no universal saturation point | compile prefill at `M ∈ {128, 256, 512, 1024}`; a true ubatch curve holds the **logical prompt fixed** and varies only the physical chunk (§3.4) |

Anchors. Published: Llama-3.2-1B Q4_0 **4,028 tok/s prefill / 54 tok/s decode** (S26+, PR
#25085, 766-token prompt, `-ub` not reported); Qwen3.5-2B 985 (S25+) / 1,301 (S26+). The
operator's figure for Qwen3-0.6B, **≥ 3,300 tok/s prefill at ubatch 512–1024 with HMX**, is
plausible on the S26+ scaling and unproven on S25+ — it is the target, not a bound. Here today
(June 2026, pmode unrecorded): Qwen3-0.6B TTFT 1.52 s at a 2048-token padded prefill (~1,350
padded tok/s incl. two host transposes per layer), **11.7 tok/s decode at 57 dispatches/token**
with attention on the CPU; Llama-3.2-1B bf16 12.2 tok/s; **Llama-3.2-1B int4 decode 17.8 tok/s
(exists; `llama32_1b_int4/Makefile:19`)**.

## 2. What exists, and the gaps — the inventory that shapes the plan

Reusable as-is (all `programming_examples/`):

- **One-submit dispatch**: `llms/shared/infra/dispatch.py:219 plan_submissions` — under the ELF
  ABI every step aggregates into one `xrt.runlist` built on any one artifact's context (the
  artifacts keep their own contexts); `run_sequence(..., require_single_submission=True)`
  (`:513`) returns results and a `DispatchVector` whose `as_row()` (`:174`) already emits the
  study's CSV keys. The `llms/` drivers do not use it — they call `KernelCache.load_and_run`
  once per ELF (`cache.py:731`).
- **Residency**: `static_input_indices` / `intermediate_indices` / `bo_key` (`cache.py:731`),
  `bo_pool.plan_pool` with content-keyed statics (`bo_pool.py:297`), one `hw_context` per
  distinct binary with `context_loads` / `kernel_attaches` counters (`cache.py:567, :639`),
  shared-xclbin attachment (`cache.py:462`).
- **Fusion patterns**: seven builders in `llms/shared/builders/` (`rms_gemms_rope`,
  `rms_qkv_bias_rope`, `rms_qkv_qknorm_rope`, `o_ffn`, `rms_gemv_rope`, `o_gemv_ffn`,
  `lm_head_gemv`), each with a GEMV twin where decode needs one; textual stitching
  `stitching.stitch_elf` (`stitching.py:318`).
- **`seq_len` is `M`**: every prefill builder passes `seq_len` straight to
  `gemm_registry_config(seq_len, K, N)` and sizes every activation by it
  (`o_ffn_multi.py:239`, `qwen3_0_6b_prefill.py:142`); `compile_all_kernels(cache, config,
  seq_len, cpu_attn)` (`qwen3_0_6b_prefill.py:454`) takes it as a parameter. A prefill ELF at
  `M = 512` is `seq_len = 512` for the projection and FFN ELFs; the FA ELF is **square**
  (`lk = lq = seq_len`, `fa_headfirst.py:82`), which is the obstacle to chunking (§3.4).
- **Device capacities, already written down**: `study/mapping_space.py:140-189` — 8 columns × 4
  core rows, two shim channels per direction per column, six memtile channels per direction,
  two core DMA channels per direction, 64 KiB L1 per core, 512 KiB memtile per column; and its
  docstring's warning (`:9-99`) that per-column placement cannot be reduced to a scalar budget.
- **Quantization assets**: AWQ packers (`llama32_1b_int4/awq_pack.py`, `awq_repacker.py` — the
  GEMV layout `A_q[M, K/2] u8`, `A_s`, `A_z`), GGUF q4_0 (`smollm2_1_7b_int4`, `symmetric=True`
  drops the Z plane), int4 GEMV kernels (`matrix_vector_multiplication/int4_awq/`, lit-gated at
  2048×2048 and 2048×8192), int4 multi-launch decode builders
  (`llama32_1b_int4/multi_launch_builder/`), **and a working int4 decode driver**:
  `llama32_1b_int4_inference.py` runs bf16 prefill then int4 decode through `rms_qkv_int4_rope`
  / `o_gemv_ffn_int4`, `make verify` gates it, `make run-inference` runs it, and the Makefile
  header records ~56 ms/token — the two READMEs that call decode "a follow-up" are stale. A
  `bfp16ebs8` GEMM (`bfp16_gemm_builder.py` → `matrix_multiplication/bf16_x_bfp16`): block
  floating point with a shared 8-bit exponent per 8, the AIE2P-native narrow format and the
  nearest analog of HMX's fp16 tile path.
- **On-device decode attention**: `attention_decode/attn_decode_npu2.py` — fused RMS + Q/K/V
  GEMV + RoPE + KV-cache write + single-query attention, KV cache resident as `[NKV, seq_len, n]`,
  **with the position `pos_host` baked in at compile time** (`attn_decode_npu2.py:80, :470`:
  `ConstantOp.create_index(pos_host)`, loop bounds and KV-slot DMA offsets use it — one
  compile per position), head_dim 64, GQA group 4, lit-gated including a post-prefill case.
  Unused by `llms/`. Making `pos` a run-time argument is the first thing it needs
  `[corrected per Codex re-read]`.
- **Verify**: `verify/runners/base.py:27` — `build_runner(...)` with `prefill(prompt_tokens) →
  PrefillRecord` and `decode_step(token, pos) → DecodeStepRecord`; gate
  `compute_topk_set_check` at `GATE_N_TOKENS 32`, `GATE_K 5`.
- **Study discipline**: schema v2 with the seven `quant_*` columns "present and empty for bf16
  rows" and separate GEMM/GEMV contracts (`schema.py:261-274`); `run_mode.run` takes counters
  from the `extra` dict the mode's `dispatch` returns (`run_mode.py:379-462`) and keeps setup
  outside the clock; `profiles.Profile` with computed `rungs()` / `expected_rows()` /
  `skip_reason`; `run_ladder` one subprocess per rung with incremental rows; `resume` with
  re-hashing; `compare_roots` with per-mode tolerances; Turbo refusal
  (`sweep/registry_sweep.py:209`); the pinned host-suite count.
- **Model description**: `<model>_weights.py::LlamaConfig` (`n_layers, emb_dim, n_heads,
  head_dim, n_kv_heads, hidden_dim, vocab_size, rope_base, qk_norm, tie_word_embeddings, dtype`)
  and a prose `ARCHITECTURE.md` per model with the ELF sequence and the NPU/CPU split.

Gaps (each becomes a deliverable):

| Gap | Evidence | Consequence |
|---|---|---|
| **Artifact cache keyed by name only** | `cache.py:358`; recompiling at another `seq_len` silently overwrites | a plan hash must be the key |
| **No builder separates `M` from `seq_len`; no chunked prefill; FA square** | grep `chunk\|ubatch` in `llms/` → nothing; `fa_headfirst.py:53-94` | chunking needs rectangular `(Lq, Lk)` attention, per-layer KV append, position-correct RoPE/mask |
| **Registry is bf16/f32 only and lacks Qwen3-0.6B's short-`M` rows** | `registry_lookup.py:35`; the JSON has gate/up at short `M` but Q/O/down for these dims only at `M = 2048` | six sweep rows (Q, O, down at 512 and 1024) before any Qwen ubatch point; int4/bfp16 tiles must be derived and marked unmeasured |
| **No machine-readable model graph** | `LlamaConfig` lacks norm type, activation, eps, bias; the ELF sequence is markdown | a small typed graph |
| **Decode = one `xrt.run` per ELF + CPU attention** | 57/token for Qwen3-0.6B (`inference.py:421`, `decode.py:285`) | submission aggregation first, on-device attention later |
| **`seq_len = 2048` literal; prompt EOS-padded; TTFT starts before tokenizing** | `qwen3_0_6b_inference.py:570, :645-649` | every published TTFT includes padded work; a model adapter seam is needed before any runner |
| **Study rows have no scope / phase / M / tokens-per-second** | schema v2 | schema v3 (§3.6) |
| **int4 decode at 1.46× bf16, unexplained** | 17.8 vs 12.2 tok/s; ~0.7 GB/token at ~30 GB/s is ~23 ms of a 56 ms token | decompose before building |

## 3. Architecture

### 3.1 Where it lives

Two packages, one contract between them:

- `programming_examples/llms/shared/plan/` — the **planner**: pure Python, importable without
  `air`, reusing `profiles.skip_reason`'s bounds, `derive_rows_per_call` and `gemm_config` as
  **leaf predicates** (not generalizing `profiles.py` into a compiler). It turns a `ModelGraph`
  and a `Workload` into a `Plan`.
- `programming_examples/transformer_layer/study/run_model.py` (+ `model_profiles.py`) — the
  **runner**: executes a `Plan` through a narrow **model adapter** over the `llms/` infra,
  measures under the study's discipline, writes schema v3 rows. The layer study is untouched;
  the model runner is a sibling sharing `schema`, `manifest`, `resume`, `compare_roots`,
  `power`, `results_io`. Execution boundary (the four modes) and numerical precision are
  orthogonal axes and stay so.

The adapter seam the runner needs, and the drivers lack today: `prepare(model, precision_plan,
compiled_shapes)`, `prefill(token_ids, ubatch_policy, state)`, `decode(state, n_tokens)`,
`dispatch_vector(scope)`, `verify_against_hf(...)`.

### 3.2 `ModelGraph` — the analog of the ggml graph

A small typed DAG, not a ggml clone and not direct builder calls (builders are lowering
endpoints; driving them directly loses lifetime, layout, precision, KV state and the host/device
boundary). Tensors carry: id, shape as a function of `(M, kv_len)`, logical dtype, **storage
dtype, compute dtype, accumulator dtype**, layout, lifetime, storage class (`weight |
activation | kv_state | scratch`). Nodes carry: op, inputs/outputs, attributes, phase
predicate, repeated-layer index. The model is a repeated block template plus embedding, final
norm, LM head and the recurrent KV state. Golden JSON for **Qwen3-0.6B and Llama-3.2-1B**
(not all ten deployments) is written from their `ARCHITECTURE.md` and pinned by a test.

### 3.3 `Plan` and the planner pipeline

`plan(graph, workload, caps) → Plan`, with `Workload = (phase, M, kv_len, ctx,
precision_plan)` and `DeviceCaps` lifted from `mapping_space.py`. Pure stages, each testable
alone:

1. **Candidates per op** — providers implementing `supports(op, shape, dtype_plan, caps)`,
   `resources(...) → spatial demand`, `estimated_cost(...) → bytes, launches`,
   `artifact_key(...)`, `lower(...) → builder call`. Families: GEMM (bf16 registry methods;
   bfp16), GEMV (bf16; int4 packed), attention (square FA; rectangular FA; decode attention),
   norm/glue, host fallback.
2. **Placement** (`supports_op` analog) — `device | host | refuse` from the lifted bounds
   (head_dim 128 ⇒ head-first FA + host transposes today; hidden % 512 and emb < 2560 ⇒ lean
   forms; `attention_config`'s `parallel_seq` floor; softmax/layer-norm L1 width; BD stride
   caps). `refuse` is a derived skip; `host` is counted in `host_ops` with boundary bytes.
3. **Capacity solver** (`compute_chunks` analog) — per matmul choose `(tile_m, tile_n,
   tile_k_l1, tile_k_l2, herd_m, herd_n)` minimizing reload counts — weight-panel reloads
   `⌈M / (tile_m·herd_m)⌉ × weight_bytes` plus activation reloads `⌈N / (tile_n·herd_n)⌉ ×
   act_bytes` — under **spatial** legality: per-column 512 KiB L2 (eight resources, not one
   pool), 64 KiB L1 (A + B ping-pong + C), shim/memtile/core channel counts, BDs, stride caps;
   sequentially reused resources combine with `max`, simultaneously live ones with sum.
   **Registry policy**: the measured registry overrides wherever it has the exact shape; a
   derived candidate is marked `analytical_unmeasured`, needs an explicit policy to compile,
   and is never written back as "best" until swept and verified. The solver's pick is checked
   against the registry's best on the 36 swept shapes — a ranking validation that is possible
   here because it ranks tiles of one kernel by traffic, not modes by latency.
4. **Fusion, residency, dispatch** — group consecutive device ops into the builder patterns
   whose combined *spatial* demand fits; plan BO residency (statics, intermediates, KV
   ownership and layout); group launches into submissions (one runlist per decode token, one
   per prefill chunk, split only at a `host` op, every split recorded).

The `Plan` carries all stage outputs plus every rejected alternative with its reason and the
prediction source (`measured | analytical | forced`). Its SHA is the **artifact cache key** and
is written to every results row. Plans are cached separately from artifacts, keyed by model /
checkpoint revision / phase / `M` bucket / context bucket / precision plan / caps / planner
version / registry revision / toolchain.

### 3.4 `ubatch` on XDNA2

Per-shape compilation means `M` is baked. Compile a **geometric set of query-`M` buckets**,
`M ∈ {1, 128, 256, 512, 1024}` (64 only if padding to 128 proves costly; `M = 4` not at all —
it has no XDNA2 meaning), and an **attention context grid** keyed by `(Lq, Lk)`: for a fixed
1024-token prompt, ubatch 1024 needs `(1024, 1024)`; ubatch 512 needs `(512, 512)` and
`(512, 1024)`; ubatch 256 adds `(256, 256 | 512 | 768 | 1024)`. That triangular grid is the
principal artifact-growth problem, and the alternative to compiling it is to make the kv trip
count a run-time argument (a runtime-sequence / RTP value; `attn_decode_npu2.py` does **not**
do this today — its `pos` is compile-time `[corrected per Codex re-read]`) — decided by
measurement in H1b. Correct incremental prefill is chunk-outer / layer-inner with per-layer KV
append, each chunk attending to all earlier chunks, position-correct RoPE and mask, and a check
that padded tail rows cannot affect valid rows; padding is masked and excluded from the
throughput numerator.

Two curves, kept distinct because they answer different questions:

- **Kernel-scaling curve** — prompt length = `M`, no chunking, no new kernel: `tok/s` at
  `M ∈ {128 … 2048}`. Validates the runner, schema, verify gate and artifact keying (H1a). This is
  *not* an ubatch curve.
- **ubatch curve** — logical prompt fixed (1024 tokens, the same token ids), physical chunk
  varied (512 vs 1024): the operator's hypothesis, and the first real milestone (H1b, §5).

### 3.5 Mixed precision, staged

`precision_plan` names, per tensor class, storage / compute / accumulator / visible dtype:

- **`bf16`** — today's path, the baseline.
- **`w4_decode`** — int4 storage (AWQ gs=128 or q4_0 gs=32 symmetric) for decode GEMV, bf16
  compute and accumulate, bf16 prefill weights resident separately. **This exists for
  Llama-3.2-1B and measures 17.8 tok/s.** The work is not to build it but to put it under the
  study's decomposition and find the ~30 ms/token that is not weight stream (H2a), then to
  bring the same plan to Qwen3-0.6B (H2b).
- **`w_bfp16_prefill`** — `bfp16ebs8` storage for prefill GEMM through the existing builder;
  the better analog of HMX's fp16 tiles than a dequant-on-tile int4 GEMM (whose two walls —
  `K_L2 < K` tiling, `tile_n = 16` from the Peano immediate range — stay a separate repair).
- **`a8`** — on-device activation quantization to int8 per op, shared across Q/K/V and gate/up;
  AIE2P integer MAC route; named so the schema has a row, scheduled last.

The `quant_*` columns are **populated**, not duplicated: `quant_gemm_contract` and
`quant_gemv_contract` differ under `w4_decode`, which is what they were for.

### 3.6 Schema v3 — additive model scope

Layer rows are unchanged. Appended last (`schema.py:377`'s rule): `measurement_scope`
(`layer | model`), `model_id`, `phase` (`prefill | decode`), `logical_token_count`,
`ubatch_tokens`, `context_start_tokens`, `context_end_tokens`, `measured_token_count`,
`tokens_per_second`, `precision_plan_id`, `plan_hash`, `host_ops`, and
`model_dispatch_vector_json` — a strictly validated `{scope, host_submissions, runlist_entries,
air_launches, herd_launches, sync_boundaries, bytes_transferred}` for the whole phase or per
token. Reused: `seq_len` = physical `M` (`ubatch_tokens` for prefill, 1 for decode),
`weights_source` = checkpoint + immutable revision, and every timing / power / quant / outcome /
selected-config / provenance / failure field. The per-layer dispatch columns stay **null** in
model rows rather than being silently redefined. `execution_mode` keeps doc 03's meaning (the
per-ELF `load_and_run` path is `hybrid`, the one-runlist path is `runlist`). `resume.row_key`
and `compare_roots.KEY_FIELDS` gain `measurement_scope, model_id, phase, ubatch_tokens,
context_end_tokens, precision_plan_id`; `compare_roots` gates `tokens_per_second` with the
per-mode tolerances.

### 3.7 Correctness gate

The model's verify adapter (top-5 token-set vs HF, 32 tokens) is the gate for every plan, run
at the plan's ubatch and precision; per-layer cosine and final-logit top-k are the lenses. A row
is `passed` only if the gate passed under the same plan hash. Study runs default to
`refuse_on_unplanned_split`.

## 4. Phases and gates

| Phase | Deliverable | Gate |
|---|---|---|
| **H0 — planner, host-only, two models** — **DONE 2026-08-21**, see below | `llms/shared/plan/`: `ModelGraph` + golden JSON for Qwen3-0.6B and Llama-3.2-1B, `DeviceCaps`, candidate providers over the lifted bounds, placement, solver with registry override and `analytical_unmeasured` marking, fusion/residency/dispatch grouping, `Plan` hash. | Host suite: the plan for each of the two models **reproduces its hand-built ELF sequence and NPU/CPU split**; every study skip in `profiles.skip_reason` is reproduced by `placement`; solver vs registry on the 36 swept shapes, mismatches explained by a named bound. Pinned count moves and is verified red. Host tests cannot validate routing or ranking — that is H1/H4's job. |
| **H1a — model adapter + runner, fixed shapes (S0)** | The adapter seam over the Qwen3-0.6B and Llama-1B drivers; `study/run_model.py`; schema v3; `model-smoke` profile; manifest / resume / `compare_roots` extended; artifacts keyed by plan hash; the kernel-scaling curve at `M ∈ {512, 1024, 2048}` and decode at `ctx ∈ {512, 1024, 2048}`. Six registry sweep rows for Qwen3-0.6B at `M = 512, 1024`. | Two walks, `complete: True`, `compare_roots OK`, verify PASS per plan hash, Turbo recorded, failures as complete rows. |
| **H1b — valid ubatch prefill (S1)** | Incremental causal prefill for Qwen3-0.6B: chunk-outer / layer-inner, per-layer KV append, rectangular head-first FA at `(512, 512)`, `(512, 1024)`, `(1024, 1024)` (or the run-time trip count), masked tail; EOS padding gone. | **The two-point ubatch curve** (§5): same 1024-token prompt at ubatch 512 and 1024; verify PASS at both; TTFT and prefill-only tok/s reported separately. |
| **H2a — decompose the existing int4 decode (S3.1)** — **DONE 2026-08-26**, see below | `llama32_1b_int4` decode under `run_model.py`: `device / sync / host` decomposition, dispatch vector, `quant_*` populated, a prediction written first. | The 56 ms token attributed: weight stream vs dequant vs dispatch vs host attention/glue. |
| **H2b — `w4_decode` for Qwen3-0.6B** — **DONE 2026-08-26**, see below | The planner's int4 GEMV candidates (head_dim 128, QK-norm) over the existing int4 builders. | verify PASS; decode tok/s against the prediction from H2a's attribution. |
| **H3 — fewer submissions per token, staged** | (1) re-execute one decode projection artifact across layers; (2) aggregate the two projections per layer into one runlist; (3) aggregate all layers; (4) move KV update + attention on device (the `attn_decode_npu2` kernel generalized to head_dim 128 / GQA 2, device-owned KV layout, context-length parameterization); (5) glue + LM head. | Each step: dispatch vector shows the reduction; verify PASS; the re-execution gate shape (`fused_reexec_gate.py`) extended to N tokens. |
| **H4 — prefill precision** — **DONE 2026-08-26, a priced NEGATIVE**, see below | `w_bfp16_prefill` through the existing bfp16 GEMM; planner GEMM family gains the entry. | verify PASS; prefill tok/s vs `bf16` at the same ubatch. |
| **H5 — planner-selected cells (S4)** | Not a Cartesian matrix: `(model, phase, prompt length, ubatch, context start/end, precision plan, power mode, cold/warm, fallback policy)` cells the planner selects plus negative controls; two walks; the standing numbers. | `complete: True`, `compare_roots OK`, every row `passed` or a derived skip, a prediction before each kernel/design experiment. |

H0 and H1a are independent and can run in parallel; H2a needs only H1a's runner; H2b and H3
need H1a; H4 needs H0; H5 last.

**`[2026-08-21]` H0 landed** — `programming_examples/llms/shared/plan/` (`graph.py`, `caps.py`,
`placement.py`, `plan.py`, `golden/`, `test_plan.py`; pure Python, no `air`), its gate pinned in
`transformer_layer/run_seam_tests.lit` (`PLAN: 10/10 passed`). What the gate shows:

- `decoder_graph(spec)` for Qwen3-0.6B and Llama-3.2-1B, golden JSON pinned;
- `plan()` **reproduces both drivers' shipped sequences from structure**: per-layer ELFs, launch
  counts and the host split equal the cached manifests — qwen prefill `rms_qkv_qknorm_rope` 9 /
  `flash_attn` 1 + two host transposes / `o_ffn_qwen` 12, decode `rms_qkv_qknorm_rope_gemv4` 4 /
  host attention / `o_gemv_ffn` 3, LM head 19; llama 7 / 1 / 12, 6 / 3, LM head 8. The launch
  counts are *derived*: a GEMM costs one launch plus a cast launch when its registry method is
  `fused-cast` (that is why 8 ops make 9 and 12 launches), the lean-form predicate (emb < 2560,
  hidden % 512) picks the fused O+FFN cascade over the split forms, the LM head's partitions
  follow the driver pin — and where the pin is not the BD-repeat cap's best the plan says so:
  for Qwen3-0.6B **10 × 16384 at `m_input = 8` would cost 9 boundaries (~1.0 ms/token) fewer than
  the shipped 19 × 8192** (recorded as a rejected alternative; untested — an O3 knob);
- the doc 57 token counts fall out: 215 launches and 57 submissions per Qwen3-0.6B decode token;
  the analytical cost (boundaries × 107 µs + submissions × 146 µs + weight stream at 32 GB/s)
  predicts 68.8 ms against the measured ~76.5 (host attention and glue are not modelled);
- `study_skip` equals `profiles.skip_reason` over every reachable family × the four profile
  modes × the sequence ladder, and `DeviceCaps` is cross-checked against the study constants;
- the capacity solver (traffic-minimizing tiles under L2/L1 legality) against the registry's
  136 swept shapes: **61 identical; every mismatch in one of three named classes** — 22 differ
  only in K-panel depth (same traffic), 40 are the method tier's forced `tile_m` (drain /
  fused-cast, a precision choice the traffic model cannot see), 13 are a narrower N tile (a
  channel/BD budget it does not model). The registry overrides wherever it has the shape; the
  solver's picks are marked `analytical_unmeasured` and never written back.

Not in H0 (by design): routing, ranking by latency, and anything that needs the device — H1's
runner takes those. The `Plan` SHA is the artifact key H1a will write into every row.

**`[2026-08-23]` H1a landed** (queue item 13; evidence `results/h1a-20260823/`, local: job scripts,
`devq.sh log` copies, the walk roots, the verify reports, `compare_walk3_walk4.txt`; the one Codex
review round is `review/verdict-13.json`, five blocking findings, all fixed below). What landed:
the adapter seam `llms/shared/model_adapter.py` over the two drivers (`prepare` / `prefill` /
`decode` / `dispatch_vector` / `verify_against_hf`, built on their own `Session` / `prepare_runtime` /
`run_npu_prefill` / `run_npu_decode_step`; the per-model difference is one `ModelBinding` row, no
fork; the dispatch vector is read off the drivers' `Profiler` per phase through ONE arithmetic,
`dispatch_vector_from_trace`, and the same record is derived statically from a cache manifest +
`Plan`); the runner `transformer_layer/study/run_model.py` + `model_profiles.py` (`model-smoke`:
both models, decode at ctx 512 / 1024 / 2048 on the shipped M = 2048 set, the Qwen3-0.6B
kernel-scaling prefill curve at M 512 / 1024 / 2048; one worker process per artifact set, the
production `make verify` command line after each set over the set's own prompts); **schema v3**
(`study/schema.py`: the thirteen §3.6 columns appended last, per-layer dispatch columns refused in
a model row, the seven-key vector validated strictly; `resume.row_key` / `compare_roots.KEY_FIELDS`
gain the six model columns, the ledger gains `model_key`; `compare_roots` gates `tokens_per_second`
at the mode's band and treats `plan_hash` as an identifier; the recorded v2 roots keep reading
through `read_rows_compatible`, which `compare_roots` / `smoke_gate` / `manifest` now use — a v2
root against a v3 root fails on `schema_version` rather than comparing silently); the host suite
615 → **656 in 30 modules** (`run_study_host_tests.lit`); and the planner brought to the
items-11/12 kernels (decode QKV `rms_qkv_qknorm_rope_gemv2` at 2 launches, the Qwen head derived
as 9 × 16384 + 4480; `PLAN: 10/10`, golden regenerated) so that **for both models and both phases
every plan stage's launch count equals the cached manifest's** — qwen prefill 85 submissions /
626 executed launches, decode 57 / 150 per token; llama 49 / 328 and 33 / 152.

**What binds a `passed` row to what it measured** (the review's five findings, each now a clause
with a host test and a live check): (1) the gate runs on EXACTLY the timed artifact set — the
verify adapters LOAD the caches named by `LLMS_VERIFY_PREFILL_CACHE` / `_DECODE_CACHE` (never
compile on that path; the production `make verify`, unset, is what it was), and the worker
sha256s every ELF's bytes before timing, the runner again before and after the gate: all three
must agree or every row of the set is `failed` with the mismatch; (2) a row is `passed` only if
the gate process exited 0, wrote its own report into a fresh `mkdtemp` directory this call
created, and that report says OK for the row's prompt; (3) the gate's prompt IS the timed prompt
— a prefill rung's full M tokens (the 32 generation slots come from `LLMS_VERIFY_MAX_SEQ = M + 32`
and `LLMS_VERIFY_PREFILL_M = M`, the pad target decoupled from the KV capacity) — and the row
records the verified prompt length; (4) a forced GEMM method is part of the `Plan` and its hash
(`plan(..., forced={"o_ffn_qwen": "fused-cast"})`, source `forced`, planner `h0.3`), so the
M = 1024 rows hash as the plan that built their artifacts and the worker refuses a rung whose
plan's launch counts differ from the manifest's; (5) the provenance test runs over RECORDED driver
traces (`study/fixtures/h1a_driver_traces/`, five Profiler traces the production drivers produced
on the device, devq 574) against a plan recomputed in the test, with a negative control, and the
worker checks live on every rung that the driver's dispatch equals the plan's (per forward / per
token) — `{'host_submissions': (measured, predicted)}` fails the row.

What the gate shows (devq 575 walk 3, devq 576 walk 4; HEAD `3a1fd6c9`; Turbo observed before
and after both; `complete: True` both; `compare_roots OK`, 0 warnings, 0 failures, identifier
mismatches 0, tok/s drift median 1.3 % / p90 2.7 % qwen and 1.1 % / 1.6 % llama inside the 5 %
warn band; the production gate PASS on every artifact set, on the timed bytes, at the timed
prompt lengths — 2048 / 480 / 992 / 2016 tokens at M = 2048, 1024 at M = 1024, 32 tokens, top-5
set vs HF bf16; a walk takes ~4 min now that the gate loads instead of compiling). The first
pass (devq 571 / 572, the same numbers within the band) is **superseded**: its gates recompiled
the caches and verified M − 32-token prompts.

| row (walk 3 / walk 4) | tok/s | ms per forward or token | device / sync / host-cpu ms | per-phase or per-token vector |
|---|---|---|---|---|
| Qwen3-0.6B prefill M = 2048, prompt 2048 (kernel-scaling) | **1367 / 1371** | 1498 / 1494 | 1185 / 85 / 26 | 85 submissions, 626 launches, 1018 herds, 347 syncs, 1997 MB |
| Qwen3-0.6B prefill M = 1024, prompt 1024 (kernel-scaling; O+FFN cascade **forced fused-cast**, plan source `forced`, see below) | **1359 / 1338** | 754 / 765 | 590 / 43 / 12 | 85, 598, 990, 347, 999 MB |
| Qwen3-0.6B prefill M = 512 | skipped (wall 2, below) | | | |
| Qwen3-0.6B decode, ctx 480 → 512 | **12.01 / 12.17** | 83.3 / 82.2 | 62.9 / 0.9 / 11.6 | 57, 150, 206, 179, 4.3 MB per token |
| Qwen3-0.6B decode, ctx 992 → 1024 | **10.62 / 10.33** | 94.2 / 96.8 | 63.4 / 0.9 / 21.7 | same |
| Qwen3-0.6B decode, ctx 2016 → 2048 | **7.31 / 7.35** | 136.7 / 136.1 | 65.6 / 1.7 / 55.6 | same |
| Llama-3.2-1B prefill M = 2048, prompt 2048 | **1736 / 1739** | 1180 / 1178 | 1106 / 52 / 9 | 49, 328, 552, 201, 1208 MB |
| Llama-3.2-1B decode, ctx 480 → 512 | **10.75 / 10.83** | 93.1 / 92.4 | 76.9 / 0.7 / 7.7 | 33, 152, 184, 153, 0.7 MB per token |
| Llama-3.2-1B decode, ctx 992 → 1024 | **10.03 / 10.18** | 99.7 / 98.3 | 76.1 / 0.9 / 14.2 | same |
| Llama-3.2-1B decode, ctx 2016 → 2048 | **8.85 / 8.99** | 113.0 / 111.2 | 77.2 / 0.9 / 25.6 | same |

Reading it: the decode context cost is the host attention and nothing else — device ms is flat
(63 → 66 qwen, 76 → 77 llama) while `host_cpu_ms` (the `decode_attention_cpu` + glue buckets)
grows 11.6 → 21.7 → 55.6 ms/token on qwen, 7.7 → 14.2 → 25.6 on llama; the short-prompt 77 ms
token of §1.1 / doc 57 reads 83 ms at ctx 512 for that reason. The per-token vector is the
plan's to the launch (150 and 152 executed launches; 57 and 33 submissions), checked live on
every rung. The prefill clock is the forward only (tokenization and EOS padding outside;
`measured_token_count` = valid prompt tokens; the first-forward BO allocation lands in the
warmup). Qwen prefill at M = 2048 leaves ~200 ms per forward outside device + sync + host-cpu —
untimed Python glue that `host_cpu_ms` does not bucket. ~~`host_ops` counts the instrumented
buckets only (n_layers + 1 per decode token, + 2 per prefill), fewer than the plan's host ops
(kv_append and the embed lookup are not bucketed).~~ **`[2026-08-25]` closed (queue item 15)**:
every planned host stage is now a named `time_cpu` bucket — `kv_append` in both decode blocks
and both prefill loops (renamed from `kv_cache_extract` to the plan's stage name), the
head-first FA transposes (`transpose_seq_to_head` / `transpose_head_to_seq` in
`fa_headfirst.py`, which also moves that part of the ~200 ms into `host_cpu_ms`), and the
adapter's decode `embed_lookup` — so measured `host_ops` equals `plan_total_host_ops`
(qwen 58 / 86 per decode token / prefill forward, llama 34 / 18) and the runner's live check
fails the row on any inequality. Verified on the device: walk 5 ran the check green end to end
(devq 579; its qwen numbers are a transient — attention 2.2× and device +7 % across the board,
gone on the immediate re-run) and walk 6 (devq 580) is the citable one: qwen decode 11.71 /
10.03 / 7.09 tok/s at ctx 512 / 1024 / 2048 with `kv_append` 0.10–0.14 ms and `embed_lookup`
0.01 ms per token (the buckets cost nothing; they only name what was already inside the token),
`host_ops` 58 / 86 (qwen decode / prefill) and 34 / 18 (llama) — equal to the plan on every
rung. The five fixture traces are walk 6's.

**Two walls, met on the kernel-scaling curve.** (1) `o_ffn_qwen` (and the shared `o_ffn`) is
fused-cast-only: its slices bind a 4-arg GEMM with an f32 scratch, and the registry's best at
M = 512 / 1024 for O, gate/up and down is **`drain`** (3 args, no scratch) — devq 568:
`use of value '%arg15' expects different type than prior uses: 'memref<512x1024xbf16>' vs
'memref<512x1024xf32>'`. The builder now refuses that case by name; a `gemm_method=` knob
(`gemm_registry_config(..., method=)`, `compile_all_kernels(o_ffn_gemm_method=)`) forces the
cascade's only form, the deviation is written to the artifact set's `compile.json`, folded into
the `Plan` that hashes the rows (`forced`), and copied onto every row (`artifact_deviation`, the
label). The M = 1024 point is that forced form (fused-cast rows measured 3910 / 3739 / 4677
GFLOP/s for O / gate-up / down against drain's 4751 / 3977 / 5148, registry); its QKV stage is
the registry-driven all-drain form (8 launches, no cast — hence 598 executed launches, not 626,
and the plan says exactly that). (2) At M = 512 the forced form does not build either: the
registry's fused-cast row for 512 × 1024 × 3072 (gate/up) was measured at `tile_n 96` while O
and down are `tile_n 128`, and the cascade links one `mm.o` variant per ELF — devq 570: `Qwen
o_ffn assumes all 4 GEMMs share ... got O=fused-cast_m64n128 G=fused-cast_m64n96
D=fused-cast_m64n128`. Building it needs an explicit policy for an unmeasured `n128` tile
(§3.3's `analytical_unmeasured` rule) or the cascade taught to co-link two variants; neither was
taken here. The M = 512 row is a complete skipped row carrying that text in both walks. A
drain-capable cascade (per-GEMM arg maps and extern sets, as `rms_qkv_qknorm_rope_multi` already
does through `alloc_gemm_scratch`) is the repair that makes the curve the registry's own at
every M.

**`[2026-08-25]` Both walls closed** (queue item 14; evidence `results/item14-20260825/`, local:
job scripts, compiled roots, the two walk roots, `compare_walk7_walk8.txt`, the re-execution
gate). The cascade is PER-GEMM now: the shared `o_ffn_multi._build_o_ffn` takes each GEMM's
registry method + tile_n independently (drain = 3-arg / 1 launch / tile_m 32, fused-cast =
4-arg + f32 scratch + cast launch / tile_m 64) with per-GEMM `mm.o` objects and sym suffixes
co-linked in one ELF — exactly the `rms_qkv_qknorm_rope_multi` mechanics, via a new air-free
layout owner `gemm_builder.o_ffn_gemm_layout` (specs + `alloc_gemm_scratch` tail; base arg 15)
that the builder, `compile_all_kernels`'s mm-variant compiles and the driver's
`restore_scratch_layout` / `_o_ffn_call` all read (one owner, so a loaded artifact set is
called with exactly its own arg count). `build_o_ffn_qwen_module` is a thin delegate
(`q_dim=`, `func_name="o_ffn_qwen"`) — the ~250-line Qwen copy is gone — and `gemm_method=`
survives as the explicit override, test-only, still recorded as a deviation when used. The
stitched M = 2048 module is BYTE-IDENTICAL before/after for both models (captured/regenerated,
string-equal); the recompiled M = 2048 ELFs match the shipped ones to their sizes and launch
counts with only aircc's per-run `aie_image` metadata differing (913/576/228 bytes of 9.4/6.6/
2.4 MB — the untouched flash_attn builder shows the same drift, the control; devq 582,
`m2048-policy-identity.md`). Host suite 656 → **660/660 in 30 modules** (+4 test_model_adapter:
the layout vs the PLANNER's launch count at all three M — two independent derivations — the
forced M = 1024 layout reproducing the recorded 12-launch artifact, the M = 512 two-variant mix
`_m64n96`/`_m64n128` that wall 2 refused, and the driver bound to the layout owner by ast);
seam suite unchanged.

On the device (all Turbo before/after): the M = 512 / 1024 sets compiled with NO forced method
(devq 582, ~5 min each, `artifact_deviation: null`, o_ffn_qwen **8 air launches / 16 herd** =
the plan's own derivation, QKV 8/14); production `make verify` PASS on the recompiled M = 2048
path (devq 584, topk 2/2, rc 0). Walks 7 and 8 (devq 585 / 586, `results/item14-20260825/`,
compiled root the item-14 sets + the shipped M 2048): **`complete: True` both, 10/10 passed,
the curve 3/3** — Qwen3-0.6B prefill M 512 **1270 / 1273 tok/s** (the first measured M = 512
point), M 1024 **1372 / 1353** on the drain plan (486 launches, `forced = {}`, plan source
back to derived, no `artifact_deviation` label; the forced fused-cast walks 3/4 read 1359 /
1338 at 598 launches), M 2048 **1376 / 1390** (626 launches, unchanged plan); decode ctx
512/1024/2048 11.94/10.08/7.30 and 12.04/10.43/7.19; llama untouched (prefill 1759.9/1759.9,
decode 10.87/10.14/8.89 and 10.69/10.09/8.70). `compare_roots` walk7 vs walk8 **VERDICT: OK**
(0 warnings, 0 failures, identifier mismatches 0, tok/s drift med 1.19 % qwen / 1.11 % llama);
verify PASS per artifact set on the timed bytes (M 512 and 1024 1/1, M 2048 4/4 both models),
the item-15 `host_ops` live check green on every rung. The two-dispatch re-execution gate
(doc 57 §1.5 shape: one loaded set, two back-to-back dispatches through the production call
sites, `reexec_o_ffn_gate.py`) **PASS on both new all-drain sets** (devq 589): o_ffn corr
0.9997 vs the CPU cascade reference on BOTH dispatches, QKV's V path 0.99994, dispatch 2
byte-equal to dispatch 1 for every output, at M = 512 and 1024.

**The deviation plumbing had to change, and it explains devq 583.** A LOADED artifact set must
be restored with ITS OWN layout, not the registry's: the old forced fused-cast M = 1024 set run
through the per-GEMM driver got the drain restore — 15 args set on a 19-arg ELF, and
`load_and_run`'s ELF path `set_arg`s only the bos it is given, so the four f32 C-scratch args
the fused-cast GEMMs write stayed UNBOUND. That is a nondeterministic wrong answer the token
gate catches only sometimes — the disjoint-top-5 M = 1024 verify FAIL devq 583 hit on bytes
walks 3–6 had passed (under the old 19-arg driver). `restore_scratch_layout` now takes
`o_ffn_gemm_method=`; `model_adapter.prepare` and the qwen verify adapter's loaded-cache path
read the set's `compile.json` `artifact_deviation` and pass it (661st host test pins all three
hops by source). Re-run of the exact 583 scenario — the production gate ×3 on the old forced
M = 1024 set — PASS 2/2 every leg (devq 590, Turbo before/after).

**`[2026-08-25]` Item 14's one review round** (devq 591, one blocking + one non-blocking,
both fixed; no second round): metadata alone was still trusted — absent or garbled
compile.json restored the registry layout unchecked. Now the **ELF is the ABI ground
truth**: `dispatch.elf_arg_count` reads the kernel's buffer-argument count from the
binary's own `.dynsym` (the numeric symbol names ARE the argument indices; verified
against every shipped cache) and `load_and_run` refuses — before `ensure_loaded`, at the
one chokepoint every load path dispatches through — any call that does not match it, and
any manifest `n_args` contradicting its ELF (recorded at compile time now).
`artifact_content_sha` folds each cache's `artifact_deviation` into the timed-vs-verified
identity (absence hashes as absence; `wall_s`/`cwd` stay out). Seam suite 35 → **40/40**
(synthetic-ELF parser, both refusal directions, the stale-manifest clause, the chokepoint's
position by source), study 661 → **662/662**. On the device: a deliberately mismatched
cache (the forced M = 1024 ELFs with compile.json deleted, then garbled) is REFUSED by the
production gate with `ArgCountMismatchError: o_ffn_qwen: called with 15 arguments but the
loaded artifact ... declares 19 ... refusing before any device work` (devq 592 / clean
re-run 594, control leg with metadata restored PASS 2/2); a fresh model-smoke walk with the
validation live on every dispatch is `complete: True` 10/10, curve 3/3 (M 512/1024/2048 =
1281 / 1346 / 1393 tok/s), verify PASS per set, walk8→walk9 drift med 1.0 % (devq 593).
The `--run-only` semantics chosen for a forced cache: refuse at the first o_ffn dispatch
(the run-only driver never reads compile.json); a set with valid metadata restores
correctly through the adapters. Non-blocking: the recorded deviation's `why` text updated
to the per-GEMM reality.

**Also found, and fixed.** The drivers' `--run-only` path (`make run` / `make profile`) left
`qwen3_0_6b_prefill._FUSED_SCRATCH_FOR` at `None`, so the block runner passed 17 args to the
M = 2048 QKV ELF that declares 18 (Q is fused-cast there). `qwen3_0_6b_prefill.restore_scratch_layout`
now derives the layout from the registry exactly as `compile_all_kernels` leaves it
(`alloc_gemm_scratch`, base arg 17), and `build_session --run-only`, the verify adapter on a
loaded artifact set and the model adapter all call it. Whether the 17-arg call was benign on the
device (devq 486's 12.96 tok/s came from it) was not separately tested; the gate now runs the
18-arg path and passes.

**`[2026-08-25]` H1b landed** (queue item 16; evidence `results/item16-h1b-20260825/`, local:
the probe + decision record, the compiled sets, the re-execution gate, the two walk roots,
`compare_walk1_walk2.txt`, devq logs). **The §3.4 decision, by probe**: rectangular head-first
FA (the compile-time `(Lq, Lk)` grid), NOT a run-time kv trip count — the FA launch signature
`attention_bf16(q,k,v,gp)` has no host-scalar path and every kv bound is a compile-time
constant threaded through ~1,600 builder lines, while the rectangular form is ~20 lines
confined to the causal counter: the builder always carried `lq`/`lk` independently and
`apply_causal_mask` takes GLOBAL block indices, so "the lq queries are the LAST lq rows of the
lk context" is exactly "boot and wrap the q_block counter at `(lk−lq)/64` instead of 0"
(`attn_npu2.py`; `lq == lk` emits byte-identical IR, sha-pinned before/after). Probe (walls
doctrine): tiny `(256, 512)` through FULL aircc in 7.5 s, no walls (devq 595); on the NPU
cos 0.9991 vs the offset-mask f32 reference, dispatch 2 byte-equal, square-mask control 0.35
(devq 596). The run-time trip count stays the recorded alternative for a finer ubatch grid
(the triangular artifact growth starts at ubatch 256).

What landed: `run_npu_prefill_chunked` (qwen3_0_6b, the ONE owner of the incremental path —
chunk-outer / layer-inner, per-layer `kv_append` BEFORE attention so a chunk attends to every
earlier chunk's KV plus its own straight off the cache, RoPE LUT sliced at ABSOLUTE positions
and passed NON-static (`_fused_qknorm_rope_call(lut_static=False)` — the static-skip would
silently RoPE chunk 2 at chunk 1's positions), NO padding path: a partial chunk is a refusal,
EOS padding gone); `fa_headfirst.npu_fa_headfirst_kv` + `fa_cache_name` (the FA fed from the
host KV cache; square "flash_attn" at chunk 1, `flash_attn_ctx<Lk>` after);
`compile_all_kernels(attn_ctx_lens=)` / `run_model.py compile --fa-ctx`; the adapter's
`prefill(ubatch_policy="chunked")` with per-chunk profiler deltas; **the plan models the
chunked phase** (`plan_ubatch_prefill`: per-chunk stages composed in order, the FA artifact
named per context, one chunk degenerates to the whole plan with the same sha; 169 submissions /
171 host_ops / 962 launches at 2 × 512 over 28 layers — the live dispatch and host_ops checks
enforce exactly these, never loosened); the `ubatch-curve` profile (two UBATCH-labelled rungs,
`ubatch_tokens` 512/1024, `context 0→1024`, per-chunk records + TTFT in every row); and the
gate runs the SAME chunked path (`LLMS_VERIFY_UBATCH` → the verify adapter's chunked branch,
its 32-token decode continuing from the incrementally built KV cache — decode correctness
after chunked prefill IS the point). Host suite 662 → **679/679 in 31 modules**
(`test_ubatch_prefill.py`: the chunk math against an f32 reference — composition equality,
the kernel's block mask vs the elementwise causal mask, KV layout / V-pack byte-equality, a
garbage tail provably unable to touch valid rows — the refusals, the composed plan's totals,
and the hops by source); seam PLAN 10 → **11/11**.

**Gates on silicon** (all Turbo before/after): the M=512 set compiled WITH
`flash_attn_ctx1024` and the M=1024 set, both `artifact_deviation: null`, launch counts = the
plan's (devq 597). The two-dispatch re-execution gate (doc 57 §1.5 shape, production call
site): first run FAILED an arbitrary 0.999 f32-cos threshold — `rect(512,1024) dispatch 1 cos
0.998987 < 0.999`, identical on dispatch 2 (devq 598) — settled by measurement, not by
lowering the number (devq 599, then STRICTLY re-asserted after the review round, devq 604):
no cosine criterion at all — accuracy is the kernel family's own per-element standard
verbatim (the standalone harness's `np.isclose(rtol=1.6e-2, atol=1e-1)`), **0 mismatches on
every shape and dispatch** (rect atol_required 2.41e-2 — per-element TIGHTER than the squares'
9.25e-2); dispatch 2 byte-equal per shape; and **the composition identity ASSERTED BIT-EXACT**
— the rectangular kernel's 512 rows equal rows [512:1024) of the square (1024,1024) device
kernel byte-for-byte (`bit-equal=1.0, max_abs=0`), so the rectangular change introduced no
numeric deviation at all; the square-mask negative control misclosed on 303k/1.05M elements.

**The two-point curve** (devq 600 walk 1 / 601 walk 2; the SAME 1024-token prompt, both points
chunked-path, 3 timed samples each after 1 warmup; `complete: True` both, verify PASS per set
on the timed bytes, `compare_roots` **VERDICT: OK** — 0 warnings, 0 failures, tok/s drift med
1.48 %):

| point | tok/s (w1 / w2) | TTFT s | device / sync / host-cpu ms | per-forward vector |
|---|---|---|---|---|
| ubatch 512 (2 chunks) | **1179.9 / 1173.6** | 0.854–0.889 | 697 / 55 / 46–49 | 169 subs, 962 launches, 1746 herds, 795 syncs, 1233 MB |
| ubatch 1024 (1 chunk) | **1334.5 / 1302.1** | 0.753–0.792 | 605–614 / 54–58 / 50 | 85 subs, 486 launches, 878 herds, 403 syncs, 1175 MB |

Per chunk at ubatch 512: chunk 0 (ctx 0→512) 0.39–0.41 s, 84 subs / 476 launches / 587 MB;
chunk 1 (ctx 512→1024, rectangular FA, LM head after) 0.45–0.48 s, 85 / 486 / 646 MB. Reading
it against the §1 hypothesis (throughput rises with ubatch until a capacity cap): the
direction holds on XDNA2 — halving the ubatch costs **~11 %** (and +0.09–0.10 s TTFT), and the
cost is attributable: +84 submissions and +476 launch boundaries (the composed plan's own
derivation), +59 MB boundary traffic (chunk 2's FA re-reads the full 1024-row KV), and chunk
2's rectangular FA runs 2× the keys per query. The 1-chunk ubatch-1024 point reads 1302–1334
vs the kernel-scaling whole-path M=1024 standing numbers 1346–1393 (walks 7–9): the
incremental form's price at one chunk (~1–3 %) is the per-layer KV append plus the FA fed
from a contiguous KV-cache slice copy. Kernel-scaling standing numbers unmoved.

**`[2026-08-26]` Item 16's one review round** (devq 602; 3 blocking + 1 non-blocking, all
fixed, no second round). (1) The FA standalone CLI's causal oracle applied the SQUARE mask at
rectangular shapes — fixed (mask offset by lk−lq) and the rectangular case joined the CLI's
own verify path (`run_npu2_makefile_peano_causal_rect.lit`; on device: PASS, 0/65536
mismatches at the family standard, 3.66× atol margin, devq 603). (2) The re-execution gate
now ASSERTS everything it claims (above; devq 604). (3) The chunked scheduler is tested as
CODE, both sides of the NPU boundary: host — a fake-NPU KernelCache (float64→bf16 contracts,
load_and_run's static-skip REPRODUCED) runs the real `run_npu_prefill_chunked` end-to-end
against the real `run_npu_prefill`: token, logits and every layer's KV byte-equal, plus a
teeth check that forcing `lut_static=True` breaks equality exactly at chunk 2's positions
(host suite 679 → **681/681 in 31**); device — the same 1024-token prompt three ways
(whole-1024, chunked-512, and a no-chunking CONTROL: single-shot on the shipped M=2048 set,
which MEASURES the cross-M tiling divergence the tolerance derives from; gate factor 2×
declared before the control ran): **chunked diverges LESS than the single-shot control at
every layer** (worst chunked/control mismatch-fraction ratio 0.76 k / 0.80 v; logits 0.21 vs
0.25), layer-0 K/V BYTE-equal (the chunk mechanics — absolute RoPE, per-layer append, both
chunk boundaries — carry no tolerance), top-1 equal across all three, top-5 equal for the
pair under test; the control's own 5th slot differs from whole-1024 — the cross-tiling
instability that is exactly why §3.7 gates against HF top-k set-inclusion, not cross-plan
equality (legs devq 607). (4) The square-IR pin now covers the production 1024 square too
(three shapes, byte-identical against the pre-change builder). Kernels unchanged in the
round, so one fresh walk revalidates: walk 3 (devq 606) `complete: True`, verify PASS both
sets, ubatch 512 / 1024 = **1125.2 / 1345.3 tok/s**, `compare_roots` walk1→walk3 OK (drift
med 2.72 %, max 4.63 %, inside the band).

### `[2026-08-26]` Item 27 — the GEMV used 8 of 32 cores, and the read ceiling is 50–54 GB/s, not 40.8

**`results/item27-herd-rows-20260826/`**; prediction 02:04:25Z predates the first device job
(devq 673, 02:32) by file time; devq 670–681.

**The fact.** `matvec.py:163` built `@herd(sizes=[herd_m, 1])` and every shipped caller passed
`herd_m = 8`, so every GEMV in the tree — the LM head, QKV, the int4 O+FFN cascade — occupied
**8 of the device's 32 compute tiles: all 8 columns, 1 of 4 core rows.** The compiled artifact
says so directly (`results/o1-epilogue-20260822/air_project/npu.air.mlir` names only
`aie.tile(0..7, 0|1|2)`, and shows the L2 A-panel as eight `memref<1x8x1088xbf16, 1>` buffers,
one per memtile — which also settles that the `herd_m` axis IS the column axis).

**What was built.** `matvec.py` gains `herd_rows` (default 1; the eight shipped call-site
shapes are **byte-identical**, SHA-256 checked both directions by `control/ir_sha.py --check`,
and an explicit `herd_rows=1` equals the default). The split is **output-stationary** — core
`(tx,ty)` owns `tile_m` rows and the full K, so no partial sum crosses a core and no reduction
is added. At decode M=1 there is no weight reuse, so weight-stationary is not an available
choice; a **K split** is the alternative and is unbuilt, and the shape that forces it is named:
`M < tile_m·herd_m·herd_rows = 256` rows, i.e. `kv_dim < 256`. **Rows cost no DDR path** —
one shim MM2S per column at every row count, verified in the artifacts; the memtile fans out.

**The curve** (K=1024, `tile_m=8`, M ∈ {1536, 3072, 6144}, each geometry its OWN fitted slope
and intercept per item 23 §7.3, residuals ≤ 2.5 µs, Turbo before and after every job):

| cores (8 cols × rows) | int4 GB/s | bf16 GB/s | fixed µs (int4 / bf16) |
|---|---|---|---|
| 8 (rows 1) | **22.79** | **35.72** | 180.5 / 158.8 |
| 16 (rows 2) | **38.99** (1.71×) | **44.43** (1.24×) | 222.3 / 208.5 |
| 32 (rows 4) | **53.85** (2.36×) | **50.03** (1.40×) | 336.8 / 325.5 |

**Item 23 §7.2's "the shipped int4 GEMV is compute-bound at 8 cores" is CONFIRMED** — the
falsifier needed < 1.2× at 16 cores and got 1.71×. The instrument reproduces item 23: its
unmodified builder reads 20.84 GB/s here against 21.10 there, 1.2 % apart (devq 679).

**The wall, and the three-way contest settled.** Two different kernels on two different byte
streams converge at **50.03 and 53.85 GB/s**. **All three contested ceilings are too low**:
DAM-RS's read-half **36.5 is exceeded by 47 %**, **§4's own 40.8 by 32 %**, PANG26's **45.3 by
19 %**. DAM-RS's 72.9 aggregate is not reached (53.85 is 74 % of it) and is not contradicted.
A geometry sweep holding cores fixed and varying columns (devq 680) resolves it into **three
nested ceilings — ~4.5 GB/s per core, ~10.1 GB/s per column, 50–54 GB/s device-wide** — since
8 columns × 4 rows demands 81 GB/s at the per-column figure and receives 50–54.

**CORRECTION APPLIED `[2026-08-26]`** (it was proposed here and left unapplied by item 27; the
operator session applied it at §4:1188 in the same form). §4:1108's "the machine's *maximum* measured rate
(40.8 GB/s)" is contradicted by measurement: 40.8 is one production kernel's rate at 8 of 32
compute tiles, not a device property. The replacement text is drafted in the item's
`RESULTS.md` §12; **it is proposed for the operator rather than made silently**, and every
figure in this document derived from 40.8 *as a maximum* should be re-derived at 50–54.
Doc 57 §7.7's two OPEN flags can both close on this evidence.

**What it means for the decode token — and what may NOT be claimed.** Rows raise the slope AND
the fixed cost (~+6.5 µs per added core, and `tile_m=16` does not shrink it, devq 681), so the
answer is a byte threshold, not a yes/no: **2 rows pay above 2.29 MB of packed int4 / 9.06 MB
of bf16; 4 rows above 6.18 / 20.82 MB.** Priced at shipped shapes, the Qwen3-0.6B LM head
partition (33.6 MB) wants 2 rows (−12 %), the 1.7B partition (67.1 MB) wants 4 (−18 %), the
`o_gemv_ffn_int4` cascade (6.04 MB/call) wants 2 (−15 %), and the QKV GEMVs (2.1–8.4 MB) want
1. **No ms/token figure is claimed**: item 23's F2 forbids carrying this standalone harness's
fixed cost into the cascade, so only the slope transfers — for the int4 cascade that is
**110 µs of streaming time per call removed at 2 rows**, against an intercept that must be
re-measured in the driver. **Nothing is flipped; `herd_rows` lands at 1 and no caller passes
it.**

**Two walls, both with their exact text in the item's §6.** (1) At `herd_rows > 1` each
column's L2 A tile becomes single-writer / multi-reader and the shipped decode preset **hangs
the device** (`ERT_CMD_STATE_TIMEOUT`, devq 673); bisected over three knobs (devq 674), **only
`--use-lock-race-condition-fix` unblocks it** — ping-pong and `runtime_loop_tiling_sizes` are
irrelevant — and it costs +0.8 % at 8 cores. **Any future caller of `herd_rows > 1` must set a
lock fix or the device hangs.** (2) `air-split-l2-memref` **aborts aircc** (returncode −6, in
`tileChannelOpByFactor`) on a 2-D L2 staging memref; flattening the buffers works around it in
the builder, `mlir/` untouched, but the pass still refuses the int4 row builder at `herd_m` 4
and 2. It deserves its own queue item; the reproducer is `run_matrix.sh`.
**`[2026-08-27, item 29]` The pass's launch-endpoint cap
(`max-launch-channels-mm2s = num_cols * 2 = 16`) means it never RAN on the 8-column arms this
item shipped**, so "11 / 11 combinations compile" is not evidence about that pass at those
geometries. Item 29 found four defects in it, one a silent miscompile. §4's curve is unaffected —
the arms that produced it do not go through the pass.

**Correctness.** A derived per-element bound, no cosine: for bf16 the three rounding sites of
`mv.cc` charged separately (exact bf16 products, `2⁻²⁴` per lane accumulation, a binary-tree
`reduce_add`, and a **2⁻⁸ half-ulp** bf16 store — the guaranteed figure, where item 23's family
used the optimistic 2⁻⁹); for int4, item 23's own `fold_bounds` imported unmodified.
**0 violations of 150,528 elements over 42 measured points**, worst point at 0.994 of its bound.
`air.launch` count and PDI count are **invariant with rows** (1 and 5 at rows 1/2/4), so no new
multi-launch form exists and LOAD_PDI parity is unchanged. Study host suite 720/720 in 33,
unchanged.

### `[2026-08-26]` Item 28 — landing the rows: they are worth **nothing** on the shipped LM head, and **−14.5 %** once you use what they unlock

`results/item28-land-herd-rows-20260826/`, base commit `2e14f533`, devq **684, 688, 691, 692,
699, 703, 705**. Prediction before the first device job (`PREDICTION.md`, its addendum written
between devq 688 and devq 691 and saying so). Turbo before and after every measure job.

**§4's own prediction, applied to the shipped kernel, was wrong, and the falsifier it named
fired.** Item 27's byte-threshold model priced the Qwen3-0.6B LM-head partition (33.55 MB) at
**−12 %** for two core rows. Measured on the production head, three alternating rounds per arm
in one session (devq 688, `make profile`'s clock):

| arm | cores | `lm_head_gemv` device ms | vs 1 row |
|---|---|---|---|
| 1 row | 8 | 7.58, 7.59, 7.57 | — |
| 2 rows | 16 | 7.53, 7.57, 7.55 | **−0.4 %** |
| 4 rows | 32 | 8.26, 8.30, 8.27 | **+9.1 %** |

The rows are real — the artifacts place 8 / 16 / 32 compute tiles on rows {2} / {2,3} /
{2,3,4,5} — so this is rows measured and found not to pay. **The reason is that the production
head was ALREADY at the wall this document's own §4 located.** 311.16 MB in 7.58 ms is
**41.1 GB/s** end to end, and ~48 GB/s once its ten launch boundaries are charged, against the
50–54 GB/s device ceiling. Item 27's harness read 35.72 GB/s at the same 8-core geometry — it
was 13 % *below* the shipped kernel, which is why doubling its cores bought it 24 % and doubling
the shipped kernel's bought 0.4 %. **This is item 23's F2 discipline biting one level up: not
only does the harness's intercept not transfer, its HEADROOM does not either.**

**What rows are actually worth is a bigger partition.** The activation broadcast's BD repeat is
`M / (herd_m · m_input · herd_rows) − 1` against a `[0:255]` hardware range, so a partition may
carry **16384 / 32768 / 65536 rows at 1 / 2 / 4 core rows**, and the vocab needs **10 / 5 / 3**
launches. The 5-launch form at one row does not build — `error: 'aiex.npu.push_queue' op Repeat
count exceeds the [0:255] range` (devq 691 leg 5, compile-only, never dispatched). Measured on a
standalone head harness (devq 691, a third clock, 0 violations of a derived per-element bound at
every point, worst 0.994 of the bound):

| partitions | rows | launches | ms (3 walks) | GB/s | vs shipped |
|---|---|---|---|---|---|
| 9 × 16384 + 4480 | 1 | 10 | 7.678, 7.663, 7.661 | 40.6 | — |
| 9 × 16384 + 4480 | 2 | 10 | 7.579, 7.578, 7.548 | 41.1 | −1.1 % |
| 4 × 32768 + 20864 | 2 | 5 | 6.852, 6.807, 6.844 | 45.5 | **−10.7 %** |
| **2 × 65536 + 20864** | **4 / 4 / 2** | **3** | 6.464, 6.492, 6.470 | **48.1** | **−15.6 %** |

**Two of the three steps are clean controls and the third is not, so only two are attributed**
`[2026-08-27]`: **A → B** holds the launch count and moves rows 1 → 2 (**−1.1 %**, the rows'
whole worth); **B → C** holds rows at 2 and moves launches 10 → 5 (**−0.734 ms = 147 µs per
launch boundary**, beside doc 57 §1.5's independently measured 106–108 for a different kernel).
**C → D moves both axes at once, and there is no legal 3-launch one-row control to separate them
— the BD cap forbids it — so D's gain is NOT decomposed into "boundaries" and "rate" here.** An
earlier draft of this block published per-row boundary constants and streaming rates for 1 and
4 rows; those were a model rather than a measurement and are withdrawn. The four end-to-end
rates in the table are bytes over time and stand.

**LANDED** on Qwen3-0.6B, whose driver now DERIVES its partitioning from the row count
(`lm_head_parts`), so `QWEN3_LM_HERD_ROWS=1` reproduces the pre-item-28 head byte for byte.
A-B-A on `make profile`'s clock, one session, one prompt (devq 699): `lm_head_gemv`
**7.58 → 6.48 ms (−14.5 %)**, decode **17.68 → 18.10 tok/s (+2.4 %)**,
**56.58 → 55.25 ms/token (−1.33)**; the 1-row arm read 17.68 / 17.69 / 17.67 / 17.60 across the
walk, so the A-B-A shows no drift. The re-execution gate is **7/7 clean at 3 launches** — an ODD
LOAD_PDI count, which doc 57 §1.5's defect family is about, and the compiler pad holds. The other
three decode ELFs are byte-identical across the whole job. **`make verify` PASS on both
precisions** (devq 703). **Two `w4-default-qwen` walks, 6/6 rungs each, verify PASS on both
artifact sets, `compare_roots` VERDICT OK** with 0 identifier mismatches and 0.13 % median
`device_ms` drift (devq 707) — and the runner's own live check passed on every rung, which is
what says the driver and the planner still agree after the head moved (143 `air.launch` per
decode token against 150). The planner learned the row count as a `plan.py` constant rather than
a `ModelSpec` field, deliberately: `asdict(spec)` is inside `plan()`'s hashed body, so a field
would have moved **every** model's plan sha, including llama32_1b's, which has no part in this.
Qwen3-0.6B's three recorded driver traces are re-recorded (devq 707/712); llama32_1b's two are
untouched.

**§4's "any future caller of `herd_rows > 1` must set a lock fix" is TOO BROAD, and this item
found out by refusing the shipped prefill.** A first draft of the compile-time guard read it
literally; `o_ffn_qwen` runs **four `8 × 4` GEMM herds — all 32 cores, `link_with =
mm_m64n128.o`, no lock fix — and has never hung.** **Nor is it only the GEMMs: the shipped bf16
`o_gemv_ffn`'s stage 2 (`matvec_swiglu_rms`, a decode GEMV) is `@herd(sizes=[8, 4])` at
`_STAGE2_N_CASCADE = 4` — 32 compute tiles, no lock fix, today.** So §4's premise "every shipped
GEMV used 8 of 32 cores" is not general either: the FFN GEMV has used 32 all along, by a **K
cascade** rather than an output-stationary row split. **And item 27's explanation does not
discriminate**: the prefill GEMM herds take an L2 A panel of `memref<8x1x64x256xbf16, 1>` — one
slab per column read by all four rows, the very single-writer/multi-reader shape it blamed — and
do not hang.

**`[2026-08-27, review round 6]` The name-registry guard described in earlier drafts of this
section did NOT ship, and neither did its call-site half `backend_presets.with_herd_rows()`.
Both are deleted** — see `results/item28-land-herd-rows-20260826/RESULTS.md` §9g. Two rounds of
evidence killed them, and the reason is worth keeping: **a registry has to answer for a kernel it
has never heard of, and both answers are wrong.** Treat the unknown as needing the fix and you
inject into forms the flag FAULTS — devq 812/813 measured exactly that on the QKV split-cast
form. Treat it as safe and the rule fails open on the one case it exists for. `with_herd_rows()`
was worse than redundant: it derived the flag from a **row count**, a second injection trigger
that reached every `8 × 4` module in the tree.

What ships is **narrow and positive**. `matvec.py` stamps `air.lock_race_fix_required` on the
herd it builds above one row; `ensure_lock_fix_for_marked_herds` supplies the flag **iff that
mark is present**, before a backend is constructed, and refuses a caller that contradicts it with
an explicit `False`. **A module with no mark is returned untouched** — `o_ffn_qwen`,
`rms_qkv_qknorm_rope` and `o_gemv_ffn`, all `8 × 4`, come back `fix=None`, verified on the real
modules rather than argued. And geometry that cannot be decoded does **not** "count as
multi-row": it **raises**, because neither applying the flag nor withholding it is defensible for
a module that was never read. An earlier version reported such a herd as **one row** — the same
assert-what-was-never-established defect this whole item is about, sitting inside the decode that
had been reported as fail-closed.

**§4's `o_gemv_ffn_int4` −15 % is re-priced and is really −3.1 %.** §4 priced the cascade as ONE
6.04 MB call; it is THREE `air.launch` ops of 1.098 / 3.293 / 1.647 MB, each paying its own fixed
cost, against a 2.29 MB crossover. Measured at the three real stage shapes on item 27's own
harness (devq 692): stage 1 **+11.8 %**, stage 2 **−7.5 %**, stage 3 **+6.1 %** at two rows —
the per-stage best is **−3.1 %**, i.e. **−0.69 ms/token** over 28 layers, and taking rows on all
three is **+2.1 %**, a loss. Item 27's model predicted every one of those six points within
1.5 %. **Nothing is wired there**: the int4 builders hard-code `sizes=[N_CORES, 1]` and the packed
tile order encodes the row mapping, so it is a builder port plus a weight-ABI change — its own
item, now with its ceiling attached.

## 5. The first measurable milestone

**A two-point Qwen3-0.6B bf16 prefill curve at ubatch 512 and 1024 over the same 1024-token
prompt, followed by the normal decode verify gate** (Codex's answer, adopted over the first
draft's "prompt = ubatch" milestone, which is kernel-scaling data and not an ubatch curve).

Required, in order:

1. Six measured registry rows: Q `(M, 1024, 2048)`, O `(M, 2048, 1024)`, down `(M, 3072, 1024)`
   at `M = 512` and `1024` (gate/up and K/V dims are already covered).
2. Head-first FA extended from square to `(512, 512)`, `(512, 1024)`, `(1024, 1024)` — or the
   kv trip count as a run-time argument; whichever the first compile-and-measure favours.
3. Chunk-outer / layer-inner scheduling with per-layer KV append and positional mask.
4. Per point: `1024 / prefill_elapsed` tok/s, TTFT, per-chunk timing, dispatch vector and
   host/device splits, artifact and compile counts, observed Turbo, final logits / top-k against
   HF, and the 32-token production verify after prefill.

H1a's kernel-scaling curve comes first as the cheaper validation of the runner and schema
(no new kernel), and is labelled as such in every row.

## 6. Considered and cut

- A general op-graph IR or a ggml port — a small semantic graph feeding the existing builders
  is enough for six architectures with four deltas.
- A latency cost model as a mode selector — closed negative as row 31
  ([53](31-resident-tail-r1-record.md)); the reference has none either. The planner ranks
  tiles by traffic and fixes the dispatch structure; it never ranks modes.
- `M = 4` buckets until a multi-token decode workload exists.
- All-model H0 parity — two models.
- Automatic execution of unmeasured derived tiles — derive, mark, require a policy.
- int4 prefill repair inside the first mixed-precision milestone — audit the existing int4
  decode first.
- On-device activation quantization before packed-weight decode is stable.
- "One image / one context / one submit per token" as one deliverable — staged (H3).
- A Cartesian profile matrix — planner-selected cells with negative controls.
- Rewriting the layer study — it stays; its rows and gates are cited by four docs and the
  iron adapter.

## 7. Codex review, and what it changed

Prompt: verify the operator's ≥ 3,300 tok/s Qwen3-0.6B figure and the ubatch/opbatch
dependence against published numbers and source; restate the prefill comparison; audit doc
55 §4 items 1–9 against `ggml-org/llama.cpp` master; state the mixed-precision contract
precisely; then critique the first draft of this plan (H0–H4) and answer five design questions.
Report: 56a (the verbatim Codex review, retired 2026-08-22 to git tag `pre-cleanup-20260821`), verbatim, SOURCE / INFERENCE / UNVERIFIABLE labelled.

**Verdicts on the understanding** (all applied to doc 55, marked `[per Codex review]`):
≥ 3,300 is a plausible S26+ target, unproven on S25+, and no controlled ubatch sweep is
published; PR #25085 does not report `-ub`; Qwen3.5-2B 1,300 was S26+ not S25+; `fit_op`
bounds ops / distinct buffers / descriptors / vmem and not VTCM; `RMS_NORM+MUL` fusion is
unguarded; repack is in `buffer_set_tensor`; "resident" is DDR; KV dtype is a run-time choice
(official example Q8_0; FA needs F16); **the HMX path uses fp16 tiles, not Q8 activations**;
decode attention on device is conditional; "decode is weight-stream bound" is a hypothesis;
the QNN causal story was overstated; and — the one that re-ranked the plan — **the int4 decode
path already exists at 17.8 tok/s (1.46×)**, so "~3–4× from weight bytes" is refuted as
already-evidenced.

**Changes to the plan**: H0 scoped to two models; "GEMM vs GEMV by `M`" made candidate-driven
(no `M > 4` port); fusion legality made spatial (per-column L2, channels, `max` vs sum);
registry "derive-on-miss" replaced by derive + `analytical_unmeasured` + explicit policy; the
runlist ≠ context ≠ image distinction; the model adapter seam named as a prerequisite; H1 split
into H1a (fixed shapes, kernel-scaling curve) and H1b (valid ubatching, the real first
milestone); H2 split into audit-existing (H2a) / Qwen (H2b) / prefill precision (H4) / `a8`
(last); H3 staged in five steps; schema v3 replaced by Codex's additive field set with per-layer
dispatch columns left null in model rows; H5 changed from a Cartesian matrix to planner-selected
cells with negative controls; the registry's missing Q/O/down rows at `M = 512/1024` for
Qwen3-0.6B surfaced as a prerequisite.

**Second pass** (Codex re-read of this revised document, session `01a0207e-304b…`, 2026-08-20
evening): it confirmed the §5 registry prerequisite (only Q, O and down are missing at
`M = 512 / 1024` for Qwen3-0.6B; gate/up and K/V have exact rows) and caught one error this
revision had introduced — `attn_decode_npu2.py`'s `pos` is a **compile-time** value, not a
run-time argument (corrected in §2 and §3.4, and in 57). The task then stalled in a web search
for two hours and was cancelled; no further findings were produced.

**Not adopted**: nothing substantive — one wording point: Codex's "16 buffers is not
established by `fit_op`" is true of the function but `HTP_OP_MAX_BUFS` is 16 at this sha
(`htp-ops.h:112`), so the bound is stated with its constant.

**`[2026-08-26]` H2a landed** (queue item 17; evidence `results/item17-h2a-20260826/`, local:
`PREDICTION.md`, the re-execution check + log, the two walk roots, `compare_walk1_walk2.txt`,
devq logs). The EXISTING int4 driver bound like the bf16 models — one `ModelBinding` row
(`precision_plans=("w4_decode",)`, Session takes `model_path`, `quant_contract_module`), no fork:
the seam-facing driver deltas are an alias (`load_weights = load_weights_awq`), a re-export
(`generate_rope_lut`), the accepted `profile=` kwarg, and every planned host stage a named
`time_cpu` bucket (`kv_append` in the decode block and the prefill loop — the item-15 shape).
The verify adapter gained the bf16 adapters' loaded-cache hop (`LLMS_VERIFY_*`: the gate LOADS
the timed bytes, never compiles on that path). **The plan side, minimal as briefed**: the
`w4_decode` decode plan names the SHIPPED sequence — `rms_qkv_int4_rope` 6 launches /
`o_gemv_ffn_int4` 3 / `lm_head_gemv` 8 (bf16 head), host `embed_lookup` + 16 × (`kv_append`,
`decode_attention_cpu`) + `final_rms_norm` — **33 submissions / 152 launches / 34 host ops per
token**, each int4 GEMV's why naming the quant contract (`W4_GEMV_CONTRACT`, mirroring
`awq_repacker.quant_contract`'s name — the `fa_cache_name` pattern, host-test-pinned); prefill
under `w4_decode` is the bf16 prefill plan (§3.5's dequantized-weights copy); a qk-norm model's
`w4_decode` is refused until H2b. The `quant_*` columns are POPULATED from the packing code
(`awq_repacker.quant_contract`: gs 128 read from the loader's own default by ast and the
checkpoint id's `-g128-`; packing/scale/zero layouts beside the functions that implement them;
the accumulator pinned against `mv_int4_bf16.cc`'s `accfloat`), and `quant_gemm_contract` ≠
`quant_gemv_contract` on every row — prefill is host-dequant bf16 GEMMs, decode is in-kernel
`(q−z)·s` — which is what the two columns were for (§3.5/§3.6). Host suite 685/685 in 31
(+3 adapter, +1 profiles), seam PLAN 11 → **12/12**.

**The prediction predates the measurement by file time**: `PREDICTION.md` written 01:04:01,
first measure job (devq 608) submitted 01:16:47 (`job-000608.meta`). Its provenance finding
first: the brief's "~56 ms/token (17.8 tok/s)" is the JUNE Makefile-header number (commit
aa73c0d7), pmode-unrecorded, **no artifact in tree**; the Turbo-recorded baseline is doc 57
§1.3 (devq 438/440: 64–66 ms at ~57-token context), and the prediction was stated against
devq 440's per-kernel lines, not the June header.

**Walks** (devq 609 walk 1 / 610 walk 2, the `w4-decode` profile on the SHIPPED build_peano
caches — ELFs of 2026-08-19; Turbo before/after; `complete: True` both, 3/3 passed; verify
**PASS** both walks, 3/3 prompts (480/992/2016 tokens, top-5 vs the AWQ-dequant-patched HF
reference) on the LOADED timed bytes; `compare_roots` **VERDICT: OK**, 0 failures — 3 warnings,
all from walk 1's host-attention transient below; `device_ms` drift med **0.9 %**). The
dispatch vector measured = the plan's on every rung, live: 33 / 33 / 152 / 184, `host_ops` 34.

| ctx (per token) | predicted (PREDICTION.md §2C) | walk 1 | **walk 2 (citable)** | tok/s w1 / w2 |
|---|---|---|---|---|
| 480 → 512 | 72–88 ms | 73.7 | **68.2** | 13.56 / **14.67** |
| 992 → 1024 | 78–95 ms | 77.0 | **73.6** | 12.99 / **13.60** |
| 2016 → 2048 | 90–107 ms | 102.0 | **89.2** | 9.80 / **11.21** |

Walk 1's host attention (12.3 / 14.9 / 39.0 ms) is transient-inflated — its ctx-2048 max token
read 505.7 ms while the min tokens agree across walks (85.3 vs 83.6) — the same class as item
15's walk-5 qwen transient; walk 2's attention (7.5 / 13.0 / 25.5 ms) equals the bf16
sibling's (7.7 / 14.2 / 25.6, walks 3/4), which is what the prediction asserted (same head
geometry, same numpy code). Per-component, predicted → measured (walk 2 / walk 1 per call):

| component | predicted | measured |
|---|---|---|
| `lm_head_gemv` (bf16, 536.9 MB padded) | 14.4–15.3 ms | 14.61–15.18 (36.6 GB/s incl. its 8 boundaries) |
| `o_gemv_ffn_int4` (28.541 MB packed) | 2.0–2.6 ms | **1.96–1.99** — 2–3 % below the band's devq-440 edge: **14.4–14.6 GB/s**, not the head probe's 11 |
| `rms_qkv_int4_rope` (3.293 MB packed) | 0.72–0.82 ms | **0.69–0.71** — 0.64 of it is its 6 boundaries |
| device_ms / token | 59–70 ms | **56.9–58.1**, ctx-flat |
| sync / kv_append+embed+final_norm | ~0.8 / ~0.3 | 0.7–0.9 / 0.11–0.12 |
| untimed Python glue | 4–10 ms | 2.9–4.7 |

**The attribution the gate asks for.** The token's 1046.2 MB of weights (52.7 qkv + 456.7
o_ffn int4-packed + 536.9 bf16 head — exact packed-BO bytes, computed from the packers) at the
machine's boundary-free 40.8 GB/s stream class would take **25.7 ms**; the device spends *(40.8 is one kernel's rate at 8 of 32 tiles, not a device maximum — item 27; this attribution is at THAT geometry and is not re-derived here)*
**~57.0**. The ~31 ms that is not weight stream is charged, not independently measured, so its
split is a set of BOUNDS `[2026-08-26, per this commit's review]`: **16.3 ms of launch
boundaries** (152 × 107 µs — 10.3 of it in the 16 QKV calls: that line is ~90 % boundary, 0.64
of its 0.70 ms/call, its "4.7 GB/s" being boundary dilution, not dequant), **~4.8 ms of fixed
submission cost** (33 `xrt.run`s × the §1.5 fit's 146 µs — inside `kernel_ms`, which times
`run.start()+wait()`), and a **residual ≤ ~10.7 ms attributable to dequant, nearly all in the
`o_gemv_ffn_int4` line** (per call 1.97 = 0.32 boundaries + 0.15 submission + 0.70
stream-at-40.8 + **≤ 0.80 residual**; × 16 ≈ 12.8, with the bf16 head ~0.6; "dequant" is the
residual after the charges, an upper bound — isolating it would need a dequant-free control
kernel at the same shape, not run here). The closure to ~1 % is the charges summing to the
measured device_ms, i.e. consistency of the charge set, not an independent confirmation of any
single term. Host attention (7.5 → 25.5 ms, ctx) and ~3–5 ms of glue ride on top. Against the bf16 sibling (92.0 / 98.6 / 112.5 ms, walks 7/8):
**1.35× / 1.34× / 1.26×** — not the 4× byte ratio, for exactly these two reasons; the June
"56 ms / 17.8 tok/s" header is not reproducible under the study clock at any measured context
(nearest point 68.2 ms at ctx 512; the short-context arithmetic lands at ~65 ms = devq 438's
64). What H2b/H3 buy is now priced: a faster int4 GEMV (O4's prerequisite) is worth up to
~10.7 ms/token here (the dequant residual bound), boundary removal up to 16.3 and submission
reduction up to ~4.8, and all are device-side — the bf16 head
(14.6 ms, 26 % of device_ms and 21.4 % of the walk-2 ctx-512 token) is the single largest line and O4's accuracy
question is still unasked.

**The re-execution check** (gate 5; devq 608): the shipped decode ELFs predate the 08-22
parity fix and none is in doc 57 §1.5's family table, so the two-dispatch device check ran on
the shipped bytes through the production `load_and_run` call sites (synthetic AWQ weights, f32
dequant references). **The parity rule called all three, 3/3, predicted in PREDICTION.md §3
before the run**: `lm_head_gemv` (8 loads, even) and `rms_qkv_int4_rope` (6, even) clean —
d1 correct, d2/d3 back-to-back and d4 byte-equal; `o_gemv_ffn_int4` (**3 loads, ODD**) d1
correct (cos 0.99701 vs the dequant cascade reference), **d2/d3 back-to-back WRONG**
(2047/2048 output values differ, max|d| = 263), **healed by one intervening other-ELF
dispatch** (d4 byte-equal to d1). Production is unaffected — the token's sequence never
dispatches the same ELF back-to-back — and the walks above ran on exactly that alternation;
recompiling the caches under the current (post-fix) compiler is the repair whenever these
artifacts are next rebuilt, and doc 57 §1.5's rule now has ten rows for ten.

**`[2026-08-26]` H2b landed** (queue item 18; evidence `results/item18-h2b-20260826/`, local:
`PREDICTION.md`, the compiled w4 decode set, the re-execution gate + log, the two walk roots,
`compare_walk1_walk2.txt`, the llama byte-identity control, the compile-toolchain sha record, devq logs 612–621; review round devq 617, two blocking + two non-blocking findings, all fixed, no second round). **The
prediction predates the device work by file time**: `PREDICTION.md` 02:10:29, first device
job (devq 612, build) submitted 02:30:37, first measure job 02:32 (`job-000612.meta`). Priced
per candidate from H2a's charge model, THEN built only what pays:

- **`o_gemv_ffn_int4` for qwen — built.** The llama 3-launch int4 cascade admits qwen's
  shapes with two parameter changes and NO new kernel: `q_dim=` on
  `build_o_gemv_ffn_int4_module` (the O GEMV decoupled, M=emb 1024 / K=q_dim 2048 — the same
  delta the bf16 `build_o_gemv_ffn_qwen_module` applies) and **k_chunk 1024** (= emb; stage-2
  swiglu_rms needs K == K_CHUNK; K_div 2/1/3 for O/gate-up/down against one `mv_int4_bf16.o`
  at DIM_K=1024, the object tag now carrying k_chunk beside gs — the same stale-.o class).
  The llama builder's IR is byte-identical under the defaults (sha-pinned control). Launch
  structure UNCHANGED: 3 launches / 1 submission — the w4 plan's dispatch vector equals the
  bf16 plan's (57 / 150 / 206 / 179, host_ops 58), checked live on every rung.
- **int4 QKV — NOT built, priced negative** (the plan records both forms as rejected
  candidates): any launch-structure fallback loses more to boundaries (+0.214 ms/layer at
  the measured 107 µs constant, devq 450) than the byte ceiling saves (6.19 MB/layer at
  40.8 GB/s = 0.152 ms/layer) at ANY dequant speed; the 2-launch in-core-epilogue form
  (mv_heads + in-kernel dequant + the slab-vs-tag-padding layout merge) is a NEW kernel
  family for a ≤2.3–4.5 ms/token ceiling with Q/K quantization error injected into
  attention — deferred, not H2b's "over the existing builders". The walks measure the bf16
  QKV line it would have to beat: 0.441–0.469 ms/call.
- **int4 LM head — NOT built**, cited from O4's measurement (doc 57 §5 item 6, devq 488:
  −0.46 ms ceiling at 11 GB/s dequant-bound; the one-launch form cannot compile past 2
  iterations, devq 468); the plan records it rejected.

What else landed: `w4_decode_pack.py` (ONE owner: the `QWEN3_W4_DECODE` flag — default OFF,
bf16 stays the operator default; RTN asym uint4 gs=128 fake-quantize + pack of wo /
gate-up-interleaved / down; the bf16 fields REPLACED by the dequantized copy so prefill,
decode and the verify oracle compute ONE model — §3.5's dequantized-weights copy; the
`quant_*` columns derived from the llama contract owner with the provenance fields rewritten
to the RTN reality, same `quant_gemv_contract_name`); the flag wired binding → `prepare`
(env BEFORE the driver's import-time read; the bf16 row PINS the flag to 0) → driver dispatch
→ loader → the verify adapter's `build_hf_model` (patches o/gate/up/down ONLY, returns None
on bf16) → the gate subprocess env; `run_model.py compile-decode` + plan-aware decode-set
discovery (`<root>/<model>/<plan>/decode_kernel_cache`; a missing set is a skip NAMING the
command, never a fall-through to the shipped bf16 bytes); the `w4-decode-qwen` profile. Host
suite 685 → **691/691 in 32** (packed dims ≡ the builder's `_packed_dims`, bit-exact dequant
substitution, flag hops by source; 690 at the commit, +1 in the review round — the llama
default-IR byte-identity golden moved from the ignored evidence control into
`test_w4_decode_pack.py`); seam PLAN 12 → **13/13**; bf16 plan shas and fixtures
untouched.

**Gates on silicon** (Turbo before/after every job): compile devq 612 (39 s, no walls);
**re-execution SHIP gate, parity predicted first — devq 621 strict** (`[2026-08-26]` review
round: the first pass, devq 614, used a cosine criterion and is superseded; the strict
rewrite reads the stage intermediates back and checks EACH kernel family per-element at ITS
OWN device-side input, zero mismatches, no cosine — the instructive failed run devq 618
showed a composed reference cannot carry a single-stage bound: the launch-1 RMSNorm's known
bf16-rstd accuracy, doc 57 §1.5, measured 4.74e-2 of scale here). `o_gemv_ffn_int4`
3 LOAD_PDIs ODD → the 16 §18 pad engages → **stage 1 (K_div=2) 0/1024 mismatches at the
packed_add harness bound (rtol 0.1, atol 0.05; atol_required 3.2e-4), stage 2 (interleaved
gate/up, the swiglu_rms harness's own `cpu_reference`) 0/3072 at (0.15, 0.5), stage 3
(K_div=3) 0/1024 at (0.1, 0.05)**, d5 new-input 0 mismatches, d2/d3/d4 byte-equal; gemv2
(2, even) v 0/1024 at the bf16-GEMV bound, q/k 0 mismatches at item 16's fused standard vs
a host model of mv_heads.cc's own number path; head (10, even) 0/16384. One characterized
fact: the gemv2's raw 64-slot output buffer varies on back-to-back dispatches in scratch
slots 0–14, healing after another ELF, with ZERO overlap with the production gather (the
gate asserts the disjointness — an overlap is a hard failure). **Walks devq 615/616**: `complete: True` both, 3/3 passed, verify
**PASS 3/3 prompts both walks** (top-5 vs the dequant-patched HF reference, on the LOADED
timed bytes), `compare_roots` **VERDICT: OK** (0 warnings, 0 failures, tok/s drift med
3.33 %, device_ms 0.60 %).

| ctx (per token) | predicted (§2A) | walk 1 | walk 2 | tok/s w1 / w2 | bf16 (walks 7/8) |
|---|---|---|---|---|---|
| 480 → 512 | 59.0–70.6 ms | 67.3 | **67.1** | 14.87 / **14.91** | 83.8 / 83.1 |
| 992 → 1024 | 71.8–86.0 ms | 83.3 | **80.6** | 12.00 / **12.40** | 99.2 / 95.9 |
| 2016 → 2048 | 112.9–125.9 ms | 123.6 | **119.5** | 8.09 / **8.37** | 137.0 / 139.1 |

**The verdict against the prediction: w4_decode pays on Qwen3-0.6B — −13 to −20 ms/token
(+19–24 % tok/s at ctx 512/1024, +11–16 % at 2048), all of it the O+FFN cascade's weight
bytes at an unchanged launch structure; all six measured tokens inside the predicted bands.**
Per line: `o_gemv_ffn_int4` **1.069–1.087 ms/call** at ctx 512/1024 (band 0.75–1.10; the
ctx-2048 legs read 1.137–1.147 — an UNEXPLAINED 3–4 % excess over the band top: the walks
co-vary context, host-attention time and device timing, so no cause is isolated; host-side
contention is one unmeasured hypothesis, and note the CPU attention completes before the
O+FFN dispatch within a token), the bf16 head 7.62–7.69, device_ms **49.9–52.9 ctx-flat**
vs bf16's 62.9–65.6.
**Where the prediction missed: the dequant residual.** The cascade's non-boundary rate is
**10.0 GB/s on packed bytes** (1.071 − 0.321 boundaries − 0.146 submission = 0.604 ms for
6.038 MB) — the int4 head probe's 11 GB/s class, not the llama cascade's 19 GB/s the band's
optimistic edge carried; the residual after the charges is 0.46 ms/call = **12.8 ms/token of
dequant excess** vs the predicted ≤11.2 (13 % over; plausibly k_chunk 1024's doubled
per-chunk fixed cost vs llama's 2048 — not separately measured). O4's conclusion sharpens:
a faster int4 GEMV is now worth up to ~12.8 ms/token on this model's own decode. **`[2026-08-27, corrected 2026-08-28]` That
ceiling is now measured against, and it under-states what a GEMV change is worth.** The
r=64 experiment ([doc 57 §5B](57-inference-path-optimizations-from-hexagon.md), evidence
`results/r64-shipped-20260827/`, devq 831–852) took **0.61–0.67 ms/token** here —
`o_gemv_ffn_int4` **1.0402 → 1.0164 ms/call** at ctx 512, **−2.1 to −2.3 %** across three
contexts, on eight counterbalanced walks with n=4 per arm and arms proven distinct by
`artifact_sha_timed` in the run record. **Two cautions for anyone pricing a GEMV change
from the 10.0 GB/s figure above.** (i) It **under-predicts by ~2×**: the division built on
it (marginal compute = 47 % of the kernel region) put the ceiling at −1.05 % and the
measurement came in at −2.20 %. The same division applied to llama32_1b_int4, whose
cascade is 91 % accounted for, predicted −2.64 % and measured **−3.54 %** — so the
fraction-accounted-for **ranks** two consumers correctly and **understates** both. Treat
it as closer to a lower bound than an upper one. (ii) It is invisible at token scale here:
the same walks' **bf16 rungs, identical bytes in both arms, move by more than the effect
does**, so no tok/s number is claimed. The mechanism is unknown; ELF size was tested on
both models and rejected. ~~bf16 stays
the default and the standing-numbers table is unchanged; the operator flips
`QWEN3_W4_DECODE` (or runs the `w4-decode-qwen` profile) to take the 14.9 tok/s path.~~
**`[2026-08-26]` superseded by the flip below.**

**`[2026-08-26]` H2b's default flipped — `w4_decode` is Qwen3-0.6B's production decode**
(queue item 24; evidence `results/item24-w4-default-20260826/`, local: `PREDICTION.md`, the
job scripts, the lit logs, the walk root, `compare_*.txt`, devq logs 649, 652–655, 659–660, 662–665;
prediction written 14:18:44, first device job submitted 14:20:37). The
operator's decision was "flip it on by default, **responsibly**", and the substance is the
second word: item 18 left the w4 path with **no standing gate at all** (its device gate was a
one-off, devq 621) and with the correctness bar for a quantized default never stated.

**The bar, decided.** Every shipped default in this tree is held to top-5 token-set inclusion
vs HF bf16 over 32 greedy tokens. For this path that phrase hides a fork, because the verify
adapter patches the HF reference with the SAME dequantized O+FFN weights the kernel consumes
(item 18's oracle, and the right design for what it measures): **that comparison cannot see
quantization error, because both sides carry it.** So the bar is now two clauses, both in the
suite, plus the bf16 arm:

| arm of `run_npu2_verify.lit` | proves | oracle | measured |
|---|---|---|---|
| `QWEN3_W4_DECODE=0 make verify` | the bf16 A/B path has not rotted | plain HF bf16 | **PASS 2/2**, first divergence steps 23 / 25, mutual rank #2 |
| `QWEN3_W4_DECODE=1 make verify` | **NPU drift** — the int4 cascade computes the quantized model correctly | dequant-patched HF | **PASS 2/2**, steps 19 / 14, mutual rank #2 |
| `make verify-quant-bar` (NEW) | **the quantization bar** — the int4 default's tokens are still top-5-included in the REAL checkpoint's | plain HF bf16 | **PASS 2/2**, steps 16 / 2, mutual ranks #2/#2 and #3/#2 |

Arm 3 is the clause that makes the flip responsible, and it needed **no new flag**: it reuses
one w4 NPU capture and runs the compare phase at `QWEN3_W4_DECODE=0`, which is exactly what
makes `build_hf_model` return `None` and the framework load the plain checkpoint. Both
precisions are PINNED in the lit rather than inherited, so a future default move cannot
silently change what the test means, and the arms are distinguished in the OUTPUT (arm 2 must
print the adapter's patch line; arm 3 carries the matching `CHECK-NOT`).

**What the standing gate still does not prove, said out loud**: per-element numerics inside the
int4 cascade — the token-set criterion stops at the first divergence. Item 18's per-stage SHIP
gate remains the instrument (each stage read back, checked at its own kernel family's published
`rtol`/`atol` against its own device-side input) and stays a RELEASE-time gate for a change to
the int4 cascade builder or `mv_int4_bf16.cc`: ~8 minutes and a bespoke harness against the
cascade's private arg indices, for a subject that moves about monthly. It was re-run here: devq 654, and again as devq 664 on the final shipped bytes — `o_gemv_ffn_int4` carries **3 LOAD_PDIs, ODD**, so the [16 §18](16-compiler-changes.md) pad is what makes it clean, and it did: 0/1024, 0/3072, 0/1024 mismatches at each stage's own family bound, d2/d3 back-to-back and d4 after another ELF byte-equal to d1, d5 new-input clean; `rms_qkv_qknorm_rope_gemv2` and `lm_head_gemv` (even) clean too.

**Suite cost, measured rather than asserted**: the naive form — three plain `make verify` invocations, each recompiling the whole set because `compile_and_cache` never skips — took **1332 s** (devq 652); the shared form takes **592 s** (devq 662) for the same three arms and the same three PASSes. The difference is entirely compiles: a gate arm itself is ~17–25 s of NPU capture plus HF reference decode, and a full compile is ~7.5 min. **The quantization bar costs ONE NPU capture and one HF decode and no compile at all** — about 25 s on this host.

**The flip itself** is one constant (`w4_decode_pack.W4_DEFAULT`); the flag keeps its name and
its sense, so there are not two ways to mean the same thing. Two things were tightened because
a default flip makes them reachable: an unparseable `QWEN3_W4_DECODE` is now a REFUSAL rather
than a silent bf16, and a decode cache compiled before the flip is refused at `prepare_runtime`
— the one place the driver, the verify adapter and `model_adapter.prepare` all pass through —
with both repairs named, instead of a bare `KeyError` from `load_and_run` after a prefill
already ran. `make clean-build` joins `make clean`: the three qwen lits use it, because
`clean`'s `rm -rf ../verify/reports` escapes the caller's cwd (item 20's recorded hazard).

**The standing numbers, re-taken after the flip.** Study runner, profile `w4-default-qwen`
(new: `decode_points`, the decode mirror of item 20's `prefill_points`, so BOTH precisions run
in ONE walk, one session, one prompt per context — an A/B across two sessions measures session
drift as much as it measures the precision):

| ctx (per token) | plan | tok/s | ms/token | device | sync | host-cpu |
|---|---|---|---|---|---|---|
| 480 → 512 | **w4_decode (default)** | **15.05** | **66.44** | 48.99 | 0.88 | 11.49 |
| 480 → 512 | bf16 | 12.46 | 80.25 | 61.29 | 0.78 | 11.98 |
| 992 → 1024 | **w4_decode** | **12.60** | **79.38** | 48.71 | 0.96 | 23.83 |
| 992 → 1024 | bf16 | 10.57 | 94.58 | 61.65 | 1.02 | 23.78 |
| 2016 → 2048 | **w4_decode** | **8.41** | **118.85** | 49.46 | 1.56 | 57.77 |
| 2016 → 2048 | bf16 | 7.25 | 137.95 | 64.05 | 1.75 | 58.51 |

**In-session, same prompt, same prefill bytes: 1.208× / 1.192× / 1.161× tok/s — −13.8 / −15.2
/ −19.1 ms per token at ctx 512 / 1024 / 2048.** `complete: True`, 6/6 passed, verify PASS on
both artifact sets on the timed bytes, and the per-token dispatch vector is **identical across
the two precisions on every rung** (57 submissions / 150 launches / 206 herds / 179 syncs /
4.34 MB, `host_ops` 58, live-checked) — item 18's central structural claim, now measured
against its own bf16 arm in one session rather than across two. All of the difference is
`device_ms`, which is ctx-flat at 48.7–49.5 ms under w4 against 61.3–64.1 under bf16.

**Against the prediction** (`PREDICTION.md` §2, written before any device job): five of the six
walk rungs landed inside their bands; the bf16 ctx-512 point read 80.25 ms against a
81.0–86.0 band, 0.9 % under. All three predicted in-session ratios held (1.17–1.27 / 1.13–1.25 /
1.08–1.20 against 1.208 / 1.192 / 1.161). On the `make profile` clock BOTH arms came in ~2–3 %
faster than their bands (w4 17.27–17.68 against 15.9–17.1, bf16 13.72–13.93 against 13.2–13.6),
which is the session and not the model — the reason for measuring the two arms alternating in
one of them. `compare_roots` against item 18's walk 2: **identifier mismatches 0** and tok/s
drift median 0.93 % / p90 1.57 % on the three comparable `w4_decode` rows; the PROBLEM verdict
is the three bf16 rows item 18 never had, plus the anticipated WARN that item 18's manifest
predates the `timing` contract block (queue item 19) so its `device_ms` definition is not
recoverable — the token-level figures, which is what this table cites, are contract-independent.

`make profile N_TOKENS=32` — **a different clock, never in the same table** (operator rule
2026-08-22): **w4 (the new default) 17.27 / 17.46 / 17.68 tok/s** (56.6–57.9 ms/token) against the **bf16 arm's 13.72 / 13.73 / 13.93** (71.8–72.9 ms) — three runs each, alternating, one session, devq 663; **+27 % on the median**. TTFT is 1.48–1.50 s on both arms, as predicted: prefill is bf16 GEMMs under either plan. Per call: `o_gemv_ffn_int4` 1.08–1.09 ms, `rms_qkv_qknorm_rope_gemv2` 0.46–0.47, `lm_head_gemv` 7.67–7.69.

Host suite 715 → **720/720 in 33 modules**; seam `PLAN 14/14`, `DISPATCH 42/42`, `POOL 33/33`,
`QKV4 4/4` unchanged; verify-runner 5/5.

**`[2026-08-26]` H3 stages 1–2 landed, 3–5 scoped** (queue item 19; evidence
`results/item19-h3-20260826/`, local: `STAGES-3-5-SCOPE.md`, per-stage predictions, devq logs
622–630; **review round devq 626, six blocking findings + one non-blocking, all fixed — the
corrections are folded into this block and named where they land**). The phase's headline is
a **corrected constant**, not a saving.

**The ladder was priced against a number nobody had decomposed — and it still is not
decomposed.** Doc 57 §1.5's **146 µs** per-`xrt.run` intercept was fitted over whole
`load_and_run` calls on **3-argument** tiny ELFs. Stage 1 measured a run-build term of 57.5 µs
on a **15-argument** ELF and this block first called the pair a "split"; **that claim is
withdrawn** (review finding 1). Measured at a stated ABI instead — `bind_ms` timed as its own
phase on the three shipped decode ELFs at 5 / 15 / 21 arguments (devq 628, reproduced 630) —
building and binding a run costs **10.2 / 20.2 / 36.5 µs**, i.e. **1.58 µs/argument + 0.6 µs**
(R² 0.92), so at the intercept's own 3-argument ABI it is **~5 µs, about 4 % of the 146**. The
other ~141 µs is **not decomposed by this item** and doc 57 §1.4 now says so.

What the numbers do support, each at its own ABI: not rebuilding a 15-argument run saves
**57.5 µs per call** (devq 622; the *timed* construction is 20.2 µs of that and the rest is
unattributed), the submission itself is **16.8 µs**, and a submission removed at the token
level is worth **46.5–61.5 µs** (devq 623 / 624 — two p50s of the same interleaved experiment
twelve minutes apart). Taking the **largest supported** estimate, as an upper bound must
(review finding 2), *every* submission the whole ladder could remove is worth
**56 × 61.5 µs = 3.44 ms/token** (2.60 ms at the smaller estimate) — against the same token's
**150 `air.launch` boundaries × 107 µs = 16.1 ms**. H3's entire budget is **under a quarter**
of O1/O3's, and that is the number the remaining stages are priced against.

**Stage 1 — the mechanism exists and is already production** (devq 622, PASS). "One ELF loaded
once, BOs re-pointed per layer" is what the drivers already do (`context_loads 1` over 112
dispatches of one artifact). 28 entries of the shipped `o_gemv_ffn.elf` in **one runlist** are
byte-identical to 28 separate submissions, stable over 8 back-to-back repeats and after ~80×28
timed dispatches, and a new input set moves every layer — so **stages 3/4's mechanism is
correct**, and the LOAD_PDI parity risk does not fire (all three shipped decode ELFs carry an
even device-PDI count). p50 over the dispatch region: production 42.369, cached runs 40.760,
one submission 40.306, one submission with fresh runs 41.620 ms. Both halves of the stage's
prediction were wrong in opposite directions; the correction above is the finding.

**Stage 2 — O2 priced and NOT landed; the `xrt.run` cache landed instead** (devq 623/624/625).
Four forms of a whole decode token, twice, all four byte-identical, each with 8 back-to-back
**replays of one identical token** clean: P0 67.10 / 67.38, P0c 64.90 / 65.52, P1 (30
submissions, vector live at **30 subs / 57 entries**) 65.68 / 65.89, P1c 63.64 / 63.86 ms. **O2 pairs are worth −1.427 /
−1.484 ms/token** — under the *pre-registered* 1.5 ms threshold, against ~864 MB of doubled
resident weights (or a preload rewrite) plus a plan and live-check extension. Priced, stopped:
the "do not double resident weights; if unavoidable, price it and stop" clause of the row —
and **recorded where a planner will find it**, as a `Plan.rejected` candidate on every decode
plan carrying its measurement (review finding 6). It moves **no plan hash**: `rejected` is
outside `plan()`'s hashed body, so a measured negative can never invalidate an artifact set
keyed by a plan sha, and a host test pins that property.
What landed is the **`xrt.run` cache** in `load_and_run` (default ON): profiles **13.44 /
13.30 / 13.47 / 13.26 → 13.99 / 13.94 / 13.97 / 14.13 tok/s** over two sessions (devq 624, 627)
— **−2.5…−4.6 ms/token, +3.7…+6.6 %** (pooled means 74.8 → 71.4 ms, +4.8 %), inside the
predicted band — `make verify` **PASS 2/0** twice, and the dispatch vector **unchanged**
(57/57/150/206), the honest gate here because this makes a submission cheaper, not rarer.
**The N-token gate was rebuilt** (review finding 3: the first one replayed ONE identical token
with no advancing state and judged it against itself, so a cached run that stopped executing
would have passed): `stage2_ntoken_gate.py` runs a genuinely autoregressive chain — token id,
position, KV cache and hidden state all advance — and judges **every step against the
cache-OFF path**, plus a staleness probe that mutates an input between two dispatches of the
same cached run and requires the output to move. **devq 627: 12/12 steps bit-identical, 10
distinct token ids, 12/12 distinct logits vectors, KV advanced, staleness probe clean.** The
gate driver itself was rewritten to accumulate and fail fast (review finding 4: it printed leg
return codes and ignored them, so devq 624's green status proved only that the script reached
its last line).
Blast radius, since it is shared infra and on by default (devq 625): `check-runlist`,
`check-offload` (context_loads 30 unmoved), `check-fused-decoder-reexec` 12/12 corr 0.9995 and
`llama32_1b make verify` all PASS. One hazard found and closed: the same reuse inside
`run_sequence` must key on the pool object, whose entry would then pin the pool and defeat
`pattern/offload`'s per-dispatch eviction (~25 MB/shape) — so that half is **default OFF**
(`LLMS_CACHE_XRT_RUNS_SEQ`), the shipped sequence path is byte-for-byte HEAD's, and a host test
pins the reason.

**Two consequences for anyone reading older numbers.** (1) **`device_ms` was silently
changing meaning, and that is now FIXED rather than annotated** (review finding 5).
`load_and_run` timed the host-side run construction *inside* `t_kernel_ms`, which
`dispatch_vector_from_trace` sums into `device_ms` — so an environment variable moved a
"device" number by 30–50 µs/call with the device doing the same work. Both halves of the
repair landed: the construction is now its own `bind_ms` phase and `kernel_ms` is **start+wait
only on both paths**, so `device_ms` no longer depends on the cache state (measured: decode
`NPU Run` totals 1978.5 / 1980.4 ms cache-off vs 1973.8 / 1964.0 cache-on, while the `Bind`
column moves **43.3 / 44.3 → 1.1 / 0.9 ms**, devq 627); and a new manifest **`timing` block**
records `kernel_ms_contract` and `xrt_run_cache` per root, with `compare_roots.compare_timing`
**REFUSING** a comparison across two contracts (the pmode rule, for the pmode's reason) and
flagging an absent stamp. Every root recorded before this commit is `bind_and_start_wait` and
reads as `absent`; token-level figures are contract-independent. (2) **An open question worth more
than stages 3 and 5 combined**: the shipped bf16 `o_gemv_ffn.elf` carries **6 device PDIs for
its 3 `air.launch`es** (`readelf -S`; corroborated by the compiled LM-head module's 10
configures / 11 devices and by stage 1's clean back-to-back dispatches, which require an even
count). If the ~107 µs is per *PDI* rather than per `air.launch`, the token's real
configuration count is 28 × (2 + 6) + 10 = **234 ≈ 25 ms/token** and doc 57 §2 undercounts this
cascade ~2×. Unsettled — it needs the §1.4 ladder re-run against PDI counts, not launch counts.

**Left, with falsifiable predictions** (`STAGES-3-5-SCOPE.md`): **stage 3 is structurally
unreachable before stage 4** — host attention splits every layer, so the pair form's 30
submissions is already the maximum; its ceiling once unblocked is **−3.4 ms/token** (56 × the
largest supported per-removal estimate, 61.5 µs; −2.6 ms at the smaller 46.5 µs one). **Stage 4
is the only large prize left**: predicted **−2.0 / −7.0 / −15.6 ms/token** at ctx 512 / 1024 /
2048 with a **crossover near ctx 350**, and four walls to probe in order (head_dim 128 L1
capacity; the runtime trip count against the `push_queue` [0:255] cap — 256 iterations at ctx
2048 sits exactly on it; device-owned KV at 235 MB for ctx 2048; GQA 2). **Stage 5 is predicted
negligible and negative** in its extra-launch form (−0.1…+0.06 ms/token); only the
in-core-prologue version is worth building, and that is a kernel change.


**`[2026-08-26]` H4 landed, and it is a priced NEGATIVE** (queue item 20; evidence
`results/item20-h4-20260826/`, local: `PREDICTION.md`, `RESULTS.md`, the launch-count
derivation and artifact-set assembly tools, the assembled bfp16 set with its provenance, the
re-execution gate + records at two declared input scales, seven walk roots, five
`compare_walk*.txt`, devq logs 634–648. **One Codex review round, devq 641, four blocking
findings, all four fixed below and named where they land; no second round.**)
**The prediction predates the device by file time**: `PREDICTION.md` mtime 19:25:02Z, this
item's first devq job (631) submitted 19:44:28Z (`job-000631.meta`). Turbo observed before and
after every job.

**The brief's byte arithmetic was wrong and correcting it is half the answer.** `bfp16ebs8` is
**9 bytes per 8 elements = 9 bits/elt**, not 4.5: **1.778× under bf16, not 3.5×**
(`matmul_bf16_x_bfp16.pack_b_bfp16ebs8`, `BFP16_BYTES_PER_BLOCK = 9`). Llama-3.2-1B's prefill
weight set is 1946.2 MB bf16 against **1094.7 MB** bfp16 — and the packer produced exactly
1094.7 MB on the device path, so the arithmetic is confirmed, not assumed.

**The A/B, and what it does and does not hold fixed.** One walk, one model, one compiled
`M = 2048`, one 2048-token prompt, one session; the same AWQ checkpoint and the same
dequantized bf16 weights, of which the bfp16 BOs are a transcode; the same `flash_attn.elf`
byte for byte (sha `603f1f41cacc`); the same driver, the same host prefill loop, and — since
the review round — **the same BO residency policy** (`review finding 2`: the bf16 branch runs
`shared_nonstatic=True` on both fused stages and gives `flash_attn` no per-layer `bo_key`,
while the bfp16 branch did neither, so the arms differed in BO residency and address reuse as
well as in their GEMM ELFs; `_run_layer_bfp16` now takes the policy as a parameter, the
inference driver passes the bf16 branch's, and walks 5–7 are the re-measure). `rms_gemms_rope`
7 launches / `o_ffn` 12 become `rms_gemms_rope_bfp16` **6** / `o_ffn_bfp16` **8**: 328 → **248**
launches per forward at 49 submissions and 18 host ops either way, checked LIVE against the
plan on every rung of every walk.

**What still co-varies, and therefore what this A/B can and cannot attribute.** Between the two
arms the following change *together*: `tile_m` (64/32 → 32), `tile_n` (**128 → 32**),
`tile_k_l1` (32 → 128), `tile_k_l2`, the GEMM microkernel itself (`mm_aie2p.cc` fused-cast/drain
against `mm_bf16_x_bfp16.cc`), the presence of the fused-cast cast launches, the weight operand
type and its packed BO layout, and the per-launch DMA/BD schedule those tiles generate.
**No measurement in this item isolates any one of them**, and none is claimed to.

| | tok/s (walk 6 / walk 7) | ms per forward | device / sync / host-cpu ms | GEMM-stage TFLOP/s |
|---|---|---|---|---|
| bf16 (`w4_decode`) | **1754.6 / 1750.9** | 1167.2 / 1169.7 | 1097.8 / 1097.4 · 46–48 · 13.2 / 12.6 | **5.08** |
| `bfp16` (`w_bfp16_prefill`) | **1201.5 / 1210.3** | 1704.6 / 1692.1 | 1629.9 / 1628.2 · 46–50 · 13.7 / 11.2 | **3.02** |

`compare_roots` walk6 → walk7 **VERDICT: OK** (0 warnings, 0 failures, identifier mismatches 0,
tok/s drift median 0.48 % / p90 0.74 %, `device_ms` 0.07 %); the production top-5 gate **PASS on
both arms in all seven walks**, on the LOADED timed bytes at the timed prompt length. Every
predicted quantity landed inside its band: bf16 1750.9/1754.6 in 1625–1780, bfp16 1201.5/1210.3
in 860–1225, bfp16 device 1628–1630 in 1500–2400, Δlaunches −80 exactly, the packed bytes exact.
**Matching the residency policy moved the device delta from +549 ms to +531 ms — ~18 ms, 3.3 %
of it — so the confound was real, is now removed, and the headline stands.**

**Where the time moved, per call, over walks 5–7 (residency matched; ≤ 1 % spread).** QKV
**7.228–7.250 → 10.584–10.672** ms (+47 %); `flash_attn` **18.616–18.663 → 18.577–18.653** (the
control, 0.3 %); O+FFN **41.806–41.843 → 71.633–72.015** (+72 %); LM head unchanged. Per forward
the two GEMM stages go **785 → 1319 ms** over the same 3.986 TFLOP of work — **5.08 → 3.02
TFLOP/s, 0.59×**, the first measured `bfp16ebs8` GEMM rate in this tree.

**The accounting, with the residual labelled as a residual** (`review finding 1`). Δdevice per
forward is **+531 ms**. This item can price exactly one term of it: the **80 removed launch
boundaries, −8.6 ms** at doc 57 §1.5's constant. The weight-byte term it cannot price: 851.4 MB
at **40.8 GB/s** takes 20.9 ms. `[2026-08-26, item 27 — APPLIED]` **40.8 is NOT the machine's
maximum**; it is the production bf16 LM head's rate between its launch boundaries (doc 57 §1.5),
measured at **8 of the device's 32 compute tiles**. Filling all four core rows measures
**50.03 GB/s** for the same bf16 GEMV and **53.85 GB/s** for the int4 GEMV (devq 677/678, own
fitted slopes per geometry, 0 bound violations of 150,528 elements across all 42 successful
points, devq 677–681). **What that does and does not establish** (per this correction's own
pre-commit review): it is **an observed rate for the tested access pattern — two related
K = 1024, `tile_m = 8`, output-stationary decode-weight-stream GEMVs at one 32-tile geometry —
NOT a device-wide ceiling.** Their convergence, and the fact that 8 columns × 4 rows would
demand ~81 GB/s at the per-column rate observed at 16 cores, suggest a *shared bottleneck* for
that pattern; they do not distinguish a DDR/device limit from a common kernel, DMA or dataflow
limit, and **no read-only device sweep has been run by anyone**. The per-core ~4.5 and
per-column ~10.1 GB/s figures are likewise observed rates at their own geometries, not
independently established capacities — item 27's falsifier F4 explicitly invalidated the
instruction-count model that could have supported a capacity claim. At 50.03 the same 851.4 MB
takes **17.0 ms**, not 20.9. The original sentence's own reasoning — "because 40.8 is a maximum,
20.9 ms is a LOWER bound on the time those bytes occupy, not an upper bound on what removing them
saves" — **rested on the false premise and is withdrawn**, and so is its companion claim that
"correcting it can only make the weight-byte term larger": with 40.8 no longer a demonstrated
maximum, 20.9 ms is **neither** a universal lower nor upper bound, and the 17.0 ms figure is a
counterexample in the smaller direction. Note also that neither measured rate is a *prefill-GEMM*
rate, which is what this passage's term actually needs. How much of that time is exposed rather
than overlapped with compute is not measured here. `PREDICTION.md` used it as an upper bound on
the saving; that was a reasoning error whose direction is now **undetermined**, and the bfp16 arm
is still 531 ms slower regardless. So with
`W ≥ 0` the unmeasured weight-byte saving, the **residual after the modelled terms is
`R = 531 + 8.6 + W ≥ 540 ms`**, and R is charged to *the whole co-varying set listed above*, not
to any member of it.

**A hypothesis, not a conclusion** (`review finding 1`). The narrowed N tile is the most
plausible single member of that set — the traffic argument is that `tile_n·herd_n` falls
128 → 32 per pass, i.e. 4× the A-panel re-reads, and A is the activation matrix, which weight
storage does not shrink. **It is a hypothesis and this item does not test it**, and the item's
own traffic model is evidence against taking it at face value: it predicted a 2.52× GEMM-stage
ratio and measured **1.69×**, so the model does not describe these arms. **The test is a bfp16
control at `tile_n = 128`** — per-GEMM tiles and co-linked `mm_bf16_x_bfp16.o` variants, exactly
item 14's repair applied to the bfp16 cascade — which is item 22. Until that runs, the correct
statement is: *`w_bfp16_prefill` as built costs 531 ms of device time per forward more than bf16
prefill, and which of the co-varying differences is responsible is unknown.*

**The format's own numbers are good, which is why the question is worth re-asking.** At each
GEMM family's **own harness input distribution** (`A, B ~ N(0, 1/√K)` —
`bf16_in_bf16_out/run.py:1222`, `matmul_bf16_x_bfp16.py:640`), every stage of every prefill ELF
in both arms is **inside its published per-element bound: 6/6 and 8/8 stages, 0 elements
outside** (devq 647), with the bfp16 GEMMs' mean relative L1 **0.0068–0.0080** against the bf16
arm's 0.0098–0.0107. There is no dequant pass anywhere: the block's shared 8-bit exponent IS the
scale and the MMUL applies it — doc 57 §5b's claim, confirmed structurally and worth nothing
here at this shape.

**The registry decision was declared before the measurement** (`PREDICTION.md` §6) and is
unchanged by it: use the builder's own tiles and mark every row `analytical_unmeasured` (§3.3),
because `gemm_config` has no quant axis and the tiles are not free — `o_ffn_bfp16_multi` needs
ONE `(tile_m, tile_n, tile_k_l1)` for all four GEMMs (shared private kernel decls in one ELF)
and the single `mm_bf16_x_bfp16.o` bakes `DIM_M/DIM_N/DIM_K`. The plan records it as a rejected
alternative carrying that cost.

**The re-execution two-dispatch gate, parity predicted first — 5 unique ELFs, 5/5, so doc 57
§1.5's family goes 10/10 → 15/15** (`review finding 3`: the first version counted six legs, but
the two `flash_attn` legs load the SAME sha256 `603f1f41cacc` and exercise the same
one-LOAD_PDI configuration; a different cache path or intervening arm is not a new parity
configuration, so the rule counts **unique ELFs**, not legs). Run at BOTH declared input scales,
identical results (devq 647 family, 648 production): `rms_gemms_rope_bfp16` (6 loads, even),
`o_ffn_bfp16` (8, even) and `o_ffn` bf16 (12, even) **clean** — d2/d3/d4 byte-equal to d1;
`flash_attn` (**1, odd**) d1 correct and **d2/d3 wrong at rel L1 0.62**, healed byte-exactly by
one intervening dispatch; `rms_gemms_rope` bf16 (**7, odd**) d1 correct and **d2 HANGS**.
Production alternates `rms → flash_attn → o_ffn` and never re-executes a prefill ELF
back-to-back. Two by-products. **(i) A new row for §1.5**: the documented "one intervening
dispatch heals it" holds for the wrong-VALUE case; when the odd ELF *hangs*, the timeout is
followed by ABORT, then `unexpected command state`, then `bad command state` — the XRT context
is poisoned for the rest of the process and the heal cannot be evaluated, which is why the gate
runs one process per ELF. **(ii) A gate whose inputs do not exercise the kernel cannot see the
defect**: at a small input scale the FA logits are ~0.16, the softmax is nearly uniform and
`attn_out` barely depends on `q`, so the same corruption read as 0.009 % of elements; at a
realistic logit spread it is rel L1 0.62.

**What "correct per family" means here, exactly** (`review finding 4`). The gate's first version
set `passed` from `rel_L1 ≤ 5e-2` **even when the published per-element bound was violated**, and
the first version of this block then claimed d1 was correct per family on records where 19–29 %
of the bf16 arm's Q/K/V, gate/up and down elements were outside theirs. The criterion is now the
family bound itself, and the gate declares its input scale, because a published `atol` belongs to
the distribution its harness measured at — `kernel_registry/details/FFN_bf16.md:92` says so
outright. At the **family** scale the claim above is true and is a real correctness-per-family
result. At **production activation scale** (norm weights O(1), so GEMM outputs ~45× the
harness's) the bf16 arm's GEMMs sit **19.6–29.0 % outside** their published `atol` while their
`rel_L1` stays at 0.0098–0.0112, and the bfp16 arm's sit 0 % outside because its family's bound
is looser (rtol 0.1 / atol 0.05 against 1.6e-2 / 1.5e-3); that gap is a fixed `atol` failing to
scale, not a datapath fault, and neither arm's numbers can be read as the other's. One of the
"failures" was the gate's own: SwiGLU was being judged at the generic elementwise bound instead
of the SiLU-and-Mul registry page's `atol = 8e-2`, and passes at both scales once judged by its
own family.

**The host transient, characterized — and why the verdict is not read off `tokens_per_second`
alone.** Over seven walks `device_ms` is stable to ≤ 1 % and every per-kernel line to ≤ 1.9 %,
while the `kv_append` host bucket (identical numpy in both arms, reading `k_roped`/`v` out of
device-mapped BOs into a transposed copy) read **7.4 … 1080.5 ms per forward** across fourteen
arm-walks; the worst is one 4.3 s sample inside walk 2, and walk 5 caught another after the
residency fix, so BO count is not its cause. Walks 1, 2 and 5 are recorded and `compare_roots`
calls their pairings PROBLEM, with **every** gate failure driven by `host_cpu_ms` and none by
`device_ms` (0.07–0.37 %) or `min_latency_ms` (≤ 1.5 %). The transient-free check agrees on all
seven walks: min-sample forward 1161–1203 ms bf16 and 1690–1723 ms bfp16, ratio **0.68–0.69
every time**. Same class as item 15's walk 5 and item 17's walk 1.

**What landed in code.** `LLAMA32_1B_INT4_PREFILL_DTYPE` on the existing int4 driver (default
`bf16` — the operator default is unchanged; read once at import, the H2b flag shape), the
load-time `bfp16ebs8` transcode of the SAME dequantized array both arms compute over,
`_run_layer_bfp16` gaining `with_kv=`, the bf16 sibling's per-layer argument cache and its
**`shared_nonstatic` residency policy** (default off, so the standalone verify/diagnosis paths,
which persist per-layer intermediates, keep the behaviour they had),
`awq_bfp_pack.quant_contract()` as the ONE owner of the plan's `quant_*` columns (the plan
mirrors only its NAME and imports nothing from a model directory), the verify adapter refusing to
compile bf16 ELFs under a bfp16 plan, the planner's bfp16 GEMM family with its refusals and its
recorded registry-policy rejection (bf16 plan shas untouched), per-plan contract owners and
`PRECISION_PLAN_PHASE` in the adapter, and — the one structural change — **the study's ARTIFACT
SET is now keyed by `(model, M, precision plan)`**, since a plan selects ELFs (H2b the decode
set, H4 the prefill set): two rungs at one `M` under two plans are two sets, two worker processes
and two gates inside one walk. Host suite 705 → **715/715 in 33 modules** from this item
(`study/test_bfp16_prefill.py`, whose flag-hop test now also pins the residency parity); the
tree's pin reads **719** because queue item 24 is adding four tests to the same suite
concurrently. Seam `PLAN 14/14` unchanged.

**Left.** The bfp16 arm is measured AS BUILT and the one experiment its result argues for is the
one this item did not run: **item 22, the bfp16 cascade with per-GEMM tiles at `tile_n = 128`**,
which is the only thing that would turn the tile hypothesis into an attribution. Until then
`w_bfp16_prefill` is a working, gate-passing, accuracy-neutral plan that costs 48 % more device
time than bf16 prefill, and **bf16 remains the default for every model**.

**`[2026-08-27]` H4 FOLLOW-UP — the tile WAS the confound, and at the registry's width bfp16
prefill is a marginal WIN** (queue item 22; evidence `results/item22-bfp16-tile-20260826/`:
`PREDICTION.md` 10:56:58 PDT and `PREDICTION_ADDENDUM.md` 11:12:38, each predating its own first
device job by file time; `RESULTS.md`; `evidence/` — the split-pass finding, the ELF-equivalence
control, the GEMM tile sweep, the re-measured bf16 control, the walk table and the reexec
records; devq 734–757. Turbo observed before and after every measure job.)

**First, the contamination question item 29 raised, settled — and settled the opposite way round
from the guess.** `air-split-l2-memref` **does** run on the bfp16 stitchers' L2 staging at the
shipped 8-column geometry, and it **does** split. Rebuilt with `aircc --debug-ir`, both ELFs cut
the A panel `8x1x32x2048xbf16` into 8, the packed-weight buffer `1x4x16x4608xi8` into 4 and the C
buffer `8x4x32x32xbf16` into 8 (`o_ffn_bfp16` also cuts the Down GEMM's K-l2 staging), with **zero
decline remarks** from a pass that emits one on every decline path. Item 29's cap
(`max-launch-channels-mm2s = 16`) does not bite here because THIS builder stages the whole herd's
A panel in ONE launch-level `air.channel.put` — 2 launch-level MM2S endpoints, so an ×8 split
lands at 9 ≤ 16 — where item 27's builder already had `herd_m + 1 = 9` and its ×2 landed at 17.
**The cap is a property of the builder's launch-endpoint count, not of the column count**, and
item 29's "the production 8-column geometry has always been on the declining side of the cap" is
true of its own builder, not of this one. **But item 20 is NOT contaminated**: recompiling both
stitchers today at the shipped geometry reproduces the 2026-08-11 ELFs at IDENTICAL SIZE, differing
only inside ≤ 56-byte clusters at a regular stride (205 bytes / 36 clusters for
`rms_gemms_rope_bfp16`, 61 / 10 for `o_ffn_bfp16`; those clusters are not shown to be PDI
headers — see below) — and the CONTROL settles what that means: **two builds made TODAY, minutes
apart, on one compiler, differ by 149 bytes in 29 windows of the same kind**, i.e. more than the
shipped ELF differs from either. **What that comparison does and does not show** (both review rounds). It shows equal SIZE, 205
of 3.86 M bytes differing in 36 clusters of ≤56 bytes at a regular stride, and a CONTROL in which
two builds made today differ the same way. It does NOT show that those regions are PDI headers or
that the code and configuration payload is unchanged — raw offsets were grouped, the container was
not parsed, and the claim that it did is withdrawn. So for **defect D, which announces nothing**
(it compiles green and returns wrong values, unlike A's compile error, B's SIGABRT and C's
timeout, none of which item 20 saw), the load-bearing evidence is **item 20's own full-output
element-wise checks**: every stage of both prefill ELFs inside its published per-element bound at
the family scale, 0 elements outside, top-5 gate PASSING on seven walks — which a miscompile
putting 6031 of 6144 elements out of bound does not survive. **Its
0.685–0.691× stands as measured.**

**`tile_n > 32` did not compile before this item, for a one-expression reason.**
`matmul_bf16_x_bfp16.py`'s `drain_dst_layout` gave the drain buffer's n_b stride as `tile_n * r`
where the contiguous buffer it types has `tile_m * t` — the value its two siblings in the same
file (`c_subview_layout`, and the drain DMA's own `src_strides`) both use, each with the comment
"skip full M-block column". The two are equal ONLY while `tile_n == tile_m`, and every shipped
bfp16 GEMM is 32×32, so it was invisible; at 64 and 128 aircc rejects the module with a
`'func.call' op operand type mismatch` naming the two strided types. Corrected: the module is
**byte-identical at 32×32** (so nothing shipped moves) and 64/128 then compile green through full
aircc at every model GEMM shape (12/12). That first claim is **re-runnable** — the item's review
refused a version of it that retained only a post-fix module and a note saying `diff -q` had been
clean. `results/item22-bfp16-tile-20260826/tools/drainfix_evidence.py` rebuilds BOTH the pre- and
post-fix modules from source in one run and records paired sha256: at 32×32
`pre = post = f8df6496…` with an EMPTY diff, and at 32×64 they differ (162 lines), which is where
a stride correction is supposed to change something.

**The tile, isolated.** Standalone `bf16 × bfp16ebs8` GEMM at the four shapes the stitchers run,
herd 8×4, `tile_m = 32`, `tile_k_l1 = 128`, one microkernel source, **`tile_n` the only parameter
that moves**; two interleaved rounds, Turbo, ≤ 4.6 % spread (devq 742 / 746):

| GEMM | `tn=32` | `tn=128` | `t(128)/t(32)` |
|---|---|---|---|
| O / Q 2048×2048×2048 | 5.814 ms | **3.273** | 0.563 |
| gate / up 2048×2048×8192 | 22.446 | **11.956** | 0.533 |
| down 2048×8192×2048 | 16.221 | **9.117** | 0.562 |
| K / V 2048×2048×512 | 1.748 | **1.140** | 0.652 |

The O GEMM goes **2.955 → 5.249 TFLOP/s**; down **4.236 → 7.538**. Accuracy is invariant in the
width — mean relative L1 **0.00656 at every width and every shape**, 0 elements outside the
family's own bound — which is what the format's structure predicts, since its group is 8 elements
along K and `tile_n` does not touch it. So **H4's hypothesis is confirmed at its own named test**:
the N tile was worth ~1.8× on this GEMM, and the residual `R ≥ 540 ms` charged to the whole
co-varying set is now mostly attributable to ONE member of it. What still co-varies **between the
two arms** is unchanged and is NOT separated here: `tile_m` (32 vs 64), `tile_k_l1` (128 vs 32),
`tile_k_l2`, the microkernel, and the bf16 arm's fused-cast launches.

**And the model number — the sentence H4 could not write.** Five walks, two rungs each, one
2048-token prompt, one session per walk, the same `flash_attn.elf` byte for byte, the production
top-5 gate PASSING on all TEN rungs on the LOADED timed bytes (sha equal before timing, before
the gate and after it), 49 / **248** / 472 against 49 / 328 / 552 checked live:

| | tok/s (walks 1 / 2 / 3) | device ms per forward | QKV /layer | `flash_attn` /layer | O+FFN /layer |
|---|---|---|---|---|---|
| bf16 (`w4_decode`) | 1297.2 † / 1753.1 / 1758.4 | 1097.845 / 1099.259 / 1095.503 | 7.226–7.266 | 18.565–18.659 | 41.763–41.857 |
| **bfp16 `tile_n = 128`** | **1777.3 / 1756.7 / 1780.9** | **1087.367 / 1088.395 / 1085.743** | **6.918–6.959** | 18.524–18.626 | **41.480–41.521** |

**Device ratio 0.9905 / 0.9901 / 0.9911 / 0.9923 / 0.9857 — bfp16 is 0.8–1.4 % FASTER than bf16**, five times (walks 4 and 5, devq 784 and 808, are the re-gates after each review round, on identical ELF bytes), where at
`tile_n = 32` it was 0.685–0.691×; against item 20's own bfp16 arm that is **1.497–1.500×** (the
cross-session comparison is licensed by the bf16 arm, which reads within 0.21 % of item 20's).
† walk1's bf16 `tok/s` is dragged to 1297 by one 424 ms `kv_append` sample — §8b's characterised
host transient, identical numpy in both arms — while its `device_ms` sits within 0.05 % of item
20's. `compare_roots` on the two transient-free roots (walk2 → walk3) is **VERDICT OK**: 0
warnings, 0 failures, identifier mismatches 0, tok/s drift median **0.84 %**, `device_ms`
**0.29 %**; walk1 → walk2 says PROBLEM with all three failures on `host_cpu_ms`-driven columns and
`device_ms` drift 0.11 %, which is §8b's pattern exactly. Every predicted quantity landed inside
its band except one, stated as a miss: the speed-up over `tn=32` was predicted at 1.50–1.55× and
measured **1.497 / 1.498 / 1.500×**. The two central predictions — device ms (1082 predicted, 1087.4 measured) and
the ratio (0.99 predicted, 0.9905 measured) — landed within 0.5 %, and the projection was made
from standalone GEMM rates plus each arm's own measured non-GEMM overhead. **That method
transferred here**, which is worth recording against item 23's F2 and item 28's finding that a
harness's HEADROOM does not: it transferred because the standalone arm is the same kernel at the
same shape, not a proxy for it.

**The re-execution gate on both new ELFs, parity predicted first**: `rms_gemms_rope_bfp16` (6
loads, even) and `o_ffn_bfp16` (8, even) both **clean** at BOTH declared input scales — d1 within
the family bound 6/6 and 8/8, d2/d3/d4 byte-equal to d1, d5 moved, `parity_rule_holds` — so doc 57
§1.5's family goes **15/15 → 17/17** (devq 755).

**A same-session bf16 GEMM control, and a discrepancy reported rather than reconciled.** At the
registry's own tiles the two formats are a **dead heat at GEMM level**: per layer all seven GEMMs
take 41.94 ms bf16 against 41.86 ms bfp16 `tn=128` (bfp16 ahead on down by 8 % and on O by 2 %,
behind on K/V by 20 %). The registry's published June rows for those same shapes are **15–22 %
ABOVE** what the same harness, tiles and shapes measure here on Turbo today (6215 vs 5140 GFLOP/s
at 2048³); this item does not reconcile that and uses only its own re-measured numbers.

**What landed in code** (as revised by the item's own three review rounds — devq 760, six
blocking; devq 787, four more plus a weakened gate; devq 811, four more plus three
non-blocking; all fixed). The `tile_n` correction in
`matmul_bf16_x_bfp16.py`; the N tile as a selectable parameter
(`LLAMA32_1B_INT4_BFP16_TILE_N`, unsupported widths refuse at import); and a guard built around
**one invariant** — *the layout the weights were packed in equals the layout the ELF about to
consume them was built for* — after the first attempt enforced it at four call sites with four
different answers to "what does a missing fact mean?", which produced a guard that was
simultaneously too permissive and **too strict: it locked bfp16 bootstrapping**, since the
compile path (the act that establishes what a set was built for) was made to demand that fact
already existed. The two facts are established differently and that is the whole design. **Fact
A, what the buffers ARE, is DERIVED from them**: `pack_b_bfp16ebs8` emits
`[N/tile_n, K/tile_k_l1, tile_n·tile_k_l1//8·9]`, so against the dense array the tiles are
solvable and the record axis cross-checks them — every (layer, field) pair is solved and they
must agree, so a repacked or swapped layer REFUSES instead of leaving a stale record, and there
is no stamp to forge. **Fact B, what the ELF expects, is not derivable at all** (its weight
argument is the same byte count at every width, which is the hazard) so it is DECLARED by
whatever built the set — and a declaration is **evidence, not authority**: one naming a layout
this build cannot produce is not read as a declaration, which closes the self-certification a
hand-written `tile_n = 256` sidecar otherwise gets by matching a packer told the same number.
CREATE writes the declaration and proceeds; CONSUME requires it. The check runs once per driver,
where the buffers meet an ELF, and `load_awq_weights_bfp`'s cache parameter was deleted rather
than tightened. **CREATE is itself constrained**, because the third round found the write was
unconditional: a set of 32-wide ELFs plus a request for 128 skipped both compiles as "already
built" and then RELABELLED itself 128 — the guard rubber-stamping the very mismatch it exists
to catch. An existing declaration is now READ-ONLY, a set may be built into only when it is
empty or already declares that layout (so a partially populated set cannot become a
mixed-width set under one label), and the declaration is written only by an invocation that
actually compiled. The same round closed two more consume paths that the newly configurable
width had turned live: item 20's re-execution gate now packs for **the set's own declared
layout** rather than the environment, and both stitcher `__main__` self-tests rebuild
`mm_bf16_x_bfp16.o` from their own constants so a module built at 32 cannot link a 128-wide
object left behind by a production compile. The shipped `verify_kernel_cache` carries a declaration whose `why` cites this
item's rebuild-and-compare as the evidence for 32. The plan package mirrors the same env read (it
stays dependency-free) so a plan priced at 128 hashes differently from one priced at 32. Also repaired on the way: both of
the int4/bfp16 drivers' `prepare_air_project` replacements took only `quant` while
`KernelCache.compile_and_cache` had grown to pass `int4_gs` and `int4_k_chunk`, so **every compile
through that driver died with a `TypeError` before building a kernel** — a host test now reads the
caller's keywords out of `cache.py` and pins that both hooks accept them (the first version of
that test took the LAST call site of two and was vacuous; it now unions them, and it was verified
failing against each pre-fix signature). Host suite **728 → 741/741 in 34 modules** from this
item, green with the width override unset and at 32 / 64 / 128, and
`run_study_host_tests.lit`'s pinned counter moves with it; **every bypass either review found has
a test verified failing before its fix** — including the bootstrap lock, whose test is read from
the driver's AST with import aliases resolved, after tamper-checking showed an aliased early
check walking past the first version. Env hygiene is enforced by the suite runner rather than
remembered per test: a test that leaves `LLAMA32_1B_INT4_BFP16_TILE_N` changed now FAILS, because
in a single-process suite a leaked width silently weakens every test after it. `bf16` remains every model's default and the bfp16 default width stays 32, because the
shipped `verify_kernel_cache` ELFs are built at 32 and the geometry check would (correctly) refuse
a 128 packer against them; flipping the default means rebuilding that cache first.
