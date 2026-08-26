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
| **H2a — decompose the existing int4 decode (S3.1)** | `llama32_1b_int4` decode under `run_model.py`: `device / sync / host` decomposition, dispatch vector, `quant_*` populated, a prediction written first. | The 56 ms token attributed: weight stream vs dequant vs dispatch vs host attention/glue. |
| **H2b — `w4_decode` for Qwen3-0.6B** | The planner's int4 GEMV candidates (head_dim 128, QK-norm) over the existing int4 builders. | verify PASS; decode tok/s against the prediction from H2a's attribution. |
| **H3 — fewer submissions per token, staged** | (1) re-execute one decode projection artifact across layers; (2) aggregate the two projections per layer into one runlist; (3) aggregate all layers; (4) move KV update + attention on device (the `attn_decode_npu2` kernel generalized to head_dim 128 / GQA 2, device-owned KV layout, context-length parameterization); (5) glue + LM head. | Each step: dispatch vector shows the reduction; verify PASS; the re-execution gate shape (`fused_reexec_gate.py`) extended to N tokens. |
| **H4 — prefill precision** | `w_bfp16_prefill` through the existing bfp16 GEMM; planner GEMM family gains the entry. | verify PASS; prefill tok/s vs `bf16` at the same ubatch. |
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

**Also found, and fixed.** The drivers' `--run-only` path (`make run` / `make profile`) left
`qwen3_0_6b_prefill._FUSED_SCRATCH_FOR` at `None`, so the block runner passed 17 args to the
M = 2048 QKV ELF that declares 18 (Q is fused-cast there). `qwen3_0_6b_prefill.restore_scratch_layout`
now derives the layout from the registry exactly as `compile_all_kernels` leaves it
(`alloc_gemm_scratch`, base arg 17), and `build_session --run-only`, the verify adapter on a
loaded artifact set and the model adapter all call it. Whether the 17-arg call was benign on the
device (devq 486's 12.96 tok/s came from it) was not separately tested; the gate now runs the
18-arg path and passes.

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
