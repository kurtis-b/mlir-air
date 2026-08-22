# 35 — Goals 1 and 2: sliding-window attention (parked) and quantized inference (done)

Consolidated 2026-08-22 from docs 35 (Goal 1 scoping + its W1 increment) and 36 (Goal 2 scoping);
their full text is at git tag `pre-cleanup-20260821`. The two goal specs they investigated (docs 11 and
12) are demoted into [01-original-plan-superseded.md](01-original-plan-superseded.md). **Goal 1 was
PARKED by the operator on `[2026-08-21]`** (README §Operator decisions): this doc keeps its three spec
corrections, the mechanical requirement summary, the risks, and the one increment that was built and
gated (W1). **Goal 2 is DONE**: q4_0 steps 1–4 and 6 `[2026-08-12]`, step 5's blockers `[2026-08-14]`,
step 5 itself `[2026-08-19]` — `programming_examples/llms/smollm2_1_7b_int4/`. Every number below
carries the provenance its source gave it.

---

## Part A — Goal 1: sliding-window / local-global attention. PARKED `[2026-08-21]`

Survey `[2026-08-12]`, read-only, against tip `b777517b`. No latency claim in the scoping; the only
measured figures are W1's (§A.3).

### A.0 Three corrections to the spec (doc 11, now [01](01-original-plan-superseded.md))

- **(a) "in-flight work to build on" — `exper/gemma3-dataflow` never executed Gemma 3 on the NPU.** Its
  own `programming_examples/gemma3/docs/results.md` classifies both NPU cells
  (`gemma3_1b_npu_{prefill,decode}_1k_blocked_initial.json`) `REAL_MODEL_EXECUTION_NOT_IMPLEMENTED`;
  tip `53348bfe` (2026-06-23) calls its decode variants "not promoted as paper-ready evidence"; its
  `aie_kernels/flow_attention.cc` is a scalar triple loop recomputing every score three times. Design
  input, not a deployable kernel.
- **(b) The skill trees are not "byte-near identical" and the `llama_kernel_builder` staleness is 11 of
  15 files in ONE tree.** `.claude/skills/`: 11 of 15 `SKILL.md` reference the dead path, 29 occurrences
  (clean: `opt-layout-alignment`, `phase-3`, `phase-6`, `phase-7`); `.codex/skills/`: zero — already on
  `llms/shared/infra/`. All 15 pairs `cmp` DIFFER (frontmatter + path migration only; gate text
  identical). `deploy-new-llm/SKILL.md:97-105` does `test -d programming_examples/llms/llama_kernel_builder`
  and halts on MISSING — **broken today**, the dir went in `2f20c2fa`. Doc 13's "all 15 in both" (now
  [01](01-original-plan-superseded.md)) was wrong in both halves.
- **(c) The four execution modes cannot measure a windowed model: they have never run a CAUSAL one.**
  Every mode hardcodes `workload_variant="encoder_bert"` (`coarse/cells.py:590`, `runlist.py:867`,
  `offload.py:874`, `fused.py:571`; `run_mode.py:174` writes it into the CSV unconditionally) and
  records `"causal": False`. The decomposed interiors have **no mask step at all** (`runlist.py:643`
  dispatches scores GEMM → softmax → output; `offload.py:779` `_host_softmax_bf16` is unmasked). Doc
  11's work item 8 therefore sits behind an unpriced mode rebuild with all four
  `run_npu2_{runlist,coarse,offload,fused}_peano.lit` gates re-derived.

### A.1 What the goal requires, mechanically

- **The mask is a band**, `q − W < k ≤ q`, per layer. The gemma3 branch records `"5-local-1-global"`
  (`npu/preflight.py:116`) and `sliding_window=512` at five sites (`npu/argument_binding.py:352`,
  `bo_plan.py:326`, `buffer_binding.py:320`, `model_runner.py:498`, `wiring.py:483`) — that branch's
  values, not a checked HF config (weights absent, §A.4).
- **One C++ function is the entire causality implementation for every shipped model**:
  `flash_attention/kernel_fusion_based/attn_npu2.cc:634-701` `apply_causal_mask(g, q_block_idx,
  kv_block_idx)` — `kv > q` fills bf16 `-inf` (`0xff80`); `kv < q` returns untouched; diagonal selects
  per row from `mask_start = row + 1`. Head-first and seq-first builders call it identically
  (`attn_npu2.py:731-751`, `attn_npu2_seqfirst.py:729-749`). Absolute positions are derivable in-kernel:
  `q_abs = q_block_idx*lqp + row`, `k_abs = kv_block_idx*lkp + col` (`lqp`/`lkp` are `-D` defines,
  `external_kernels.py:276-277`). A band is two new branches on `d = q_block − kv_block` (a second partial
  edge at `d ≈ W/lkp`, a full fill at `d > ceil(W/lkp)`) — **more** stores than today's bare return.
- **No host scalar path** (`attention_bf16(q_in, k_in, v_in, gp_out)`, `attn_npu2.py:257`): `W` is baked
  per ELF, so local/global needs **two compiled attention modules per model**. Pins inherited:
  `causal ⟹ lq == lk` and `lqp // num_q_tiles == lkp` (`attn_npu2.py:106-111`); `lkp = 64` fixed by L1
  (`FlashAttention_bf16.md:119,:124`) so **W is quantized to multiples of 64** (512 is fine).
- **Decode is host NumPy in all ten models** (`decode_attention_cpu`; cache `(n_kv_heads, max_seq,
  head_dim)`, absolute `current_pos`, no ring, no eviction). Windowed decode correctness is a slice
  change `k_cache[:, max(0, pos-W+1):pos+1, :]`; doc 11's RoPE/slot problem only arises if a ring buffer
  is taken — **do not take it in a first increment**. (`attention_decode/attn_decode_npu2.py` bakes
  `pos_host` at compile time and masks with `-99.0`; no shipped model uses it.)
- **A second, unconnected mask primitive** in the study tree: `builders/elementwise_add.py:111`
  `causal_mask_bias` — host `[seq,seq]` bf16 `triu(k=1)`, `CAUSAL_MASK_FILL = -10000.0` (not `-inf`: bf16
  `-inf` in an add goes NaN), gated standalone (`opcheck_specs.py:218` at 512×512,
  `run_npu2_causal_mask_peano.lit`, `make check-causal-mask`) and consumed by no attention path. Two
  mask conventions in one repo. Banding it is `triu(k=1) + tril(k=-W)`.
- **Host/reference side generalizes cheaply**: harness oracle `attn_npu2.py:1341-1343`;
  `mha_attention.py:364-366` `chunked_attention_reference` (caveat `:368-369` — assumes no row wholly
  `-inf`, true for a window including the diagonal); `pattern/blocked_attention.py:171-176`; each
  `<model>_cpu_helpers.py` mask; `attention_config(...)` is dict-passing so a `window` key is free.
- **Registry** (`programming_examples/kernel_registry/`, not `llms/kernel_registry/`): `causal` was
  already a column (14 rows, `lqp=256, num_q_tiles=4, num_heads_per_unroll=2, num_cascade_stages=4`);
  `dk/dv` registered at **64 and 128 only** (head_dim ≥ 128 has its own `debug-fa-runtime-failure` skill).
  `llms/`: `sliding`/`sliding_window`/`local_global`/`layer_types`/`is_causal` 0 hits; `causal=True` at
  four sites only; no `causal=False` site at all. In `transformer_layer/` "band" means 64-row
  *activation* bands, never a mask.
- **Two architecture gates disagree**: `deploy-new-llm/SKILL.md:83-84` rejects on the *conjunction*
  `sliding_window` set AND `use_sliding_window=true`; `phase-0-build-cpu-reference/SKILL.md:106-109` is a
  closed four-entry allowlist plus an *unconditional* sliding-window rejection. A model setting
  `sliding_window` alone passes the first and fails the second.
- **`softmax.py` requires `cols % 64 == 0`** with no scalar tail (`builders/softmax.py:59-62,:130`).

### A.2 The load-bearing blocker: masking is element-wise, not tile-skipping

The KV chunk loop bound is a compile-time constant (`attn_npu2.py:693-694` `chunks_per_stage`; L3→L2
streaming at `:535`, `:552` pushes every block). So a band is **cheap to make correct and buys zero
speedup**. The route that would pay — block skipping — is `attn_npu2.cc:703-775`'s `CAUSAL_ROW_HELPERS`
scaffolding with the FOOTGUN at `:715-718` (deleting the no-op `copy_O_tile_rows` hangs the design,
`ERT_CMD_STATE_TIMEOUT`), never exercised. Trap: `attn_npu2.py:1361-1363` does `perf_flops *= 0.5` under
`causal` — a convention, not executed work; a windowed GFLOP/s would compound it by the band ratio.

**Design rules that bite the performance half, not the correctness half** (doc [23](23-rules-and-open-items.md)):
- *Per-column shim budget* (23 §1, measured: `addnorm` 3 streams → 3 packet-typed channels;
  `elementwise_add` 2 → 0; `layer_norm` 1 → 0). An additive `[seq,seq]` mask into the decomposed modes
  is a third L3 operand — the `addnorm` shape — and costs `[4096,4096]` bf16 = 33.5 MiB per band
  (INFERENCE, unmeasured). The in-kernel band costs no stream. R1's column census counts `shim→core` flows
  only and reads 1 for a column carrying 2 ([31 §column census](31-resident-tail-r1-record.md), formerly 31b §7.1).
- *L3-side offset rule* (23 §2): a windowed KV feed is a **moving range**, a two-symbol affine map on an
  L3 operand — open queue item 8 verbatim (`air-split-l2-memref` `tileChannelOpByFactor` builds
  `AffineMap::get(0, 1, add)` at `AIRMiscPasses.cpp:1671,:1674,:1681` and SIGABRTs on two symbols). Item 9
  (`air-shrink-memref-sizes-by-access` shrinking `memref<12288xbf16,2>` → `<3072>` silently) is the same
  hazard on the L1 side. **Correctness-only windowing is design-rule-neutral; performance windowing
  collides with items 8 and 9 plus the hang path.**

### A.3 W1 — the first increment, BUILT and GATED `[2026-08-12]`; the entry point if Goal 1 is ever resumed

`-DWINDOW_LEN` bands the one `apply_causal_mask` all ten models link, through
`external_kernels.py` (matching the gemma3 branch's own `WINDOW_LEN`/`CAUSAL`/`QUERY_BASE` macro choice,
`flow_attention.cc:11-32,:62-63`). Gates: `flash_attention/kernel_fusion_based/run_npu2_makefile_peano_causal_window512.lit`
and `..._window512_negative.lit`; registry row `supported_kernels.md` 2048×2048, dk/dv 64/64, heads
32/8, causal ✓, **window 512**, dv_chunks 1, 14.5 ms, GFLOP/s *n/a by design*, mean_rel_L1 3.68e-2.

| measured (devq **262**, registry's 2048×2048 causal row) | W = 512 | W = 0 |
|---|---|---|
| `mean_rel_L1` | **3.676e-2** | 3.856e-2 |
| `atol_required` / `abs_err max` (ceiling 1e-1) | 8.048e-2 / 8.398e-2 (**1.24× margin**) | identical |
| mismatches (`np.isclose`, rtol 1.6e-2, atol 1e-1) | **0 / 4,194,304** | — |
| latency (matched pair, same session) | **14.47 ms** | **14.06 ms** |
| scores surviving the mask | 21.9% | 50.0% (band discards 56% more) |
| negative control (band switched off vs banded reference) | **197,331 / 4,194,304 (4.70%)** rejected, `mean_rel_L1` 4.56e-1, `atol_required` 5.34e-1 = **5.3× over** | — |

- `W = 0` is the old code **verbatim**: `sha256(attn_npu2.o)` identical, preprocessed source
  byte-identical, `W = 512` differing as the discrimination control — so the ten-model regression holds
  by construction.
- **The scoping's thin-headroom worry is FALSIFIED** (registry `rtol = 1.6e-2`, `atol = 1e-1`, measured
  `mean_rel_L1 ≈ 3.8–5.5e-2`, `FlashAttention_bf16.md:130-139`; the study's causal `mha_out_proj` rows at
  `atol = 8e-2` only 1.64× `atol_required`, `opcheck_specs.py:711-723`): banding does not push toward the
  ceiling, because the worst-magnitude outputs sit in the first `W` rows where band and full causal are
  the same mask. The `0xff7f` softmax-max init for fully-masked rows (`attn_npu2.cc:286`) is the safety net.
- **Zero speedup confirmed**: 14.47 vs 14.06 ms, inside the ±5% run-to-run band; ~2.3× would be the
  work-proportional figure.
- **The specified gate (`make verify` with window-crossing prompts) is NOT what W1 uses, on evidence.**
  Host-only test, Llama-3.2-1B-Instruct bf16, `ref` = real 512 window, `test` = degraded to full causal,
  32 greedy tokens through the real `compute_topk_set_check` (`verify/comparators.py:175-249`, first
  divergence only; `GATE_N_TOKENS = 32`, `GATE_K = 5`): the gate **accepted the degradation on 1 of 4**
  window-crossing prompts — `generic_a` (847 tokens), only 6 of 32 tokens agreeing; the other three
  failed at 2/32, 2/32, 0/32. Hazard quantified at 25% on that fixture. Rule: **a Goal 1 gate is
  element-wise `np.isclose` with a negative control verified failing on device**; `make diagnosis` never
  gates. Doc 13's "Check discriminates" row ([01](01-original-plan-superseded.md)) is the precedent.
- **transformers 5.10.2 silently ignores `config.sliding_window` and `config.layer_types` on Llama** —
  setting them changed the logits by exactly 0. An HF reference windowed that way is silently unwindowed.

### A.4 Risks and the remaining work, if ever resumed

| # | Work | Verdict |
|---|---|---|
| W1 | banded `apply_causal_mask` + registry row + lit with negative control | **done** (§A.3) |
| W2 | both architecture gates (2 files × 2 trees) + the 11-file `.claude` path staleness | small; only after a kernel exists |
| W3 | Gemma 3 deployment (7 `deploy-new-llm` phases, head_dim, QK-norm, 5:1 pattern, adapter, Makefile, 3 lits, 10-model regression) | the bulk; a full bring-up |
| W4 | land `exper/gemma3-dataflow` | **don't — mine it.** Tip `53348bfe`; merge-base `90dc5e92` (2026-05-12); **221 ahead / 422 behind**; 231 files, 112,129 insertions; `gemma3/` 182 files, 17 commits; **~2,348 lines across 22 `mlir/` files**, overlapping head-on (`AIRToAIEPass.cpp` +545 vs +2337 here, `AIRRtToNpuPass.cpp` +14 vs +1910, `AIRMiscPasses.cpp` +109 vs +449); `llms/` does not exist there; no verify adapter. Mine `flow_attention.cc:27-28,:62-63`, `bo_plan.py:126-153`, `preflight.py:116,:127` |
| W5 | four-mode measurement | blocked on the unscoped causal mode rebuild (§A.0c) |
| W6 | windowed decode | a NumPy slice per model; decline the ring |
| W7 | tile skipping | **large and blocked** (§A.2) |

Standing facts for a resumption: `verify_runner.py:275` `max_seq = 2048` is hardcoded and enforced by
**silent truncation** (`:376-378`), adapters EOS-pad back up; a 512 window crosses inside 2048, so a
600–1500-token fixture needs no new prefill shape (a Mistral-style 4096 window needs all of it); a
prompt longer than `2048 − 32` silently overruns the KV cache (`llama32_1b_inference.py:450-451`).
Shipped prompts are 8 each at 3–30 tokens (`instruct.txt` 13 lines/159 words, `base.txt` 15/136) — a
new fixture file is unavoidable, and `--prompt-style` is passed by no Makefile. `HfRunner` never
enforces `max_seq`, so the reference is windowed for free *given* `transformers` support (see the 5.10.2
finding). `hf_models.txt` holds 16 repo IDs (4 Llama-3.2, 6 Qwen2.5, 3 Qwen3, 2 SmolLM2, 1 AWQ) — **no
Gemma, no Mistral**; `~/.cache/huggingface/hub/models--google--gemma-3-{1b,4b}-pt` are **12 KB stubs**
(`refs/main` only, sha `fcf18a2a...` for 1b-pt), license-gated. `attn_npu2.cc` is shared by all ten
models, so every edit owes the ten-model `make verify`. Risks: vacuous gate (settled: use §A.3's rule);
zero speedup from the correctness route; two ELFs per local/global model (symbol-collision precedent:
`gemm_builder.py` mints names from the GEMM method alone, `stitching.py:318`, `external_kernels.py:133`);
the external license-gated weight dependency; `deploy-new-llm` Step 3 halts today.

---

## Part B — Goal 2: quantized inference. DONE `[2026-08-19]`

Scoping `[2026-08-12]`, read-only, tip `b777517b` → `2d6756ca` mid-investigation (docs only).

### B.1 What the first quantized model (`llms/llama32_1b_int4`) actually is

- Checkpoint `amd/Llama-3.2-1B-Instruct-awq-uint4-asym-g128-bf16-lmhead` (AutoAWQ gemm-v1, ungated; the
  only AWQ repo of 16 in `hf_models.txt`). Nibble order `AWQ_PACK_ORDER = [0,2,4,6,1,3,5,7]`
  (`awq_repacker.py:37`). Three packers: `awq_pack.py` (prefill GEMM), `awq_repacker.py` (decode GEMV:
  `A_q[M, K/2] uint8`, `A_s[n_groups, M] bf16`, `A_z[n_groups, M] uint8`), `awq_bfp_pack.py` (bfp16).
  Round trip lossless vs AutoAWQ's CUDA dequant (`awq_pack.py:10-13`).
- **Dequant is on device, in-core, in-tile**: `mv_int4_bf16.cc:13-14` `dequant(A)[r,k] = (q − z[r,g(k)]) ·
  s[r,g(k)]`; packed nibbles cross DRAM → L2 → L1 packed. One micro-kernel, two configs
  (`external_kernels.py:301-323`): GEMV `-DDIM_M=8 -DDIM_K=2048 -DDIM_GS=128`, GEMM `-DDIM_M=16`.
- Builders are model-private by policy (`llms/README.md:105-106`): `rms_qkv_int4_rope_multi.py` (decode,
  6 launches), `o_gemv_ffn_int4_multi.py` (decode, 3 stages, 15-arg ABI), `rms_gemms_rope_{int4,bfp16}_multi.py`
  / `o_ffn_{int4,bfp16}_multi.py` (prefill); `o_gemv_ffn_int4_fused.py` (three herds on two cascade
  chains) is dispatched by no model. Three prefill backends: `choices=["int4","bf16","bfp16"]`
  (`llama32_1b_int4_prefill.py:970`).
- **`make verify` does exercise the quantized path: 31 of 32 gated tokens per prompt come from the int4
  decode ELFs** (`verify_runner.py:126-138`; `verify_adapter.py:303-318`). Not gated: int4 prefill, bfp16
  end-to-end (prefill is bf16 on dequantized weights, `verify_adapter.py:196-202`). Reusable gate design:
  `verify_adapter.py:90-171` builds the HF reference from the AWQ config and overwrites every Linear
  with the AWQ-dequantized bf16, tightening the gate from (quant_error + NPU_drift) to (NPU_drift).
- Evidence: `agents/.state/devq/jobs/job-000222.log` (`ten-model-verify`, `exit=0`, ~63 min):
  `llama32_1b_int4: pass`, `TEN-MODEL: PASS`; `Q int4 GEMV (M=2048, K=2048)`, `K/V int4 GEMV (M=512, K=2048)`.

### B.2 The int4 verify lit — disabled for CI OOM, re-enabled `[2026-08-12]` (queue item 14 CLOSED)

`run_npu2_verify.lit:11` was `// REQUIRES: false`, alone among the ten (`18d1dac2`, 2026-06-17, "OOM on
amdhx370 runner"; wired two commits earlier in `f0a031bc`). The `auto` two-subprocess gate phase
(`verify_runner.py:380-405`) is `7f2e03d8` (2026-07-14), a month **after** the disable, and let
`nightlyPerfBenchmark.yml` drop its exclusions ("llama32_3b ~24GB, qwen25_3b ~25GB, qwen3_4b ~27GB").
Measured on this 31 GiB host (devq **255**, Turbo; peak RSS summed over the process tree at 5 Hz,
cross-checked against `ru_maxrss`):

| arm | peak tree RSS | min free | result |
|---|---|---|---|
| the lit end to end (`auto` split) | **10.53 GiB** (capture-npu 10.53 / compare-hf 9.59) | 17.3 GiB | **PASS**, 462.6 s |
| legacy single process (`--gate-phase both`) | **12.57 GiB** | 15.4 GiB | **PASS** |

The naive `NPU + HF` sum is 20.12 GiB; the single process measured 12.57 because host numpy weight
duplicates drop once resident in BOs. **Re-enabled** with `ryzen_ai_npu2, peano,
hfweights_amd_llama_3_2_1b_instruct_awq_uint4_asym_g128_bf16_lmhead`, no `hf_token` (the AWQ repo ships
its own tokenizer; `run_npu2_compile.lit`'s "gated upstream tokenizer" comment corrected). Two
corrections: `run_npu2_compile.lit` was live so quantized *compilation* was gated, only numerics were
not; and the nightly collapses a model's verify lits fail > pass > skip, so the dashboard read green
from the passing bfp16-prefill sibling for a gate that never ran. Reported, not changed:
`run_npu2_profile.lit:12` requires `hf_token` for the ungated checkpoint.

### B.3 The red lit (`run_o_gemv_ffn_int4_fused_npu2_peano.lit`) is beside the path

Cause: `Makefile_o_gemv_ffn_int4_fused:22` omits `-D__AIE_API_AIE_ADF_HPP__` (the guard
`external_kernels.py:62-77` documents: `aie.hpp → aie_adf.hpp → adf/stream.hpp → adf.h`, Vitis-only).
Every quantized sibling Makefile carries it; repo-wide exactly two lack it (this one and
`data_transfer_transpose/dma_bf16/Makefile`). The shipped decode compiles the identical `.cc` with the
flag set. **Not a Goal 2 prerequisite.** Whether the fixed lit then passes on device is untested.

### B.4 The shape contract the int4 decode builders impose (undocumented hard asserts)

`o_gemv_ffn_int4_multi.py:119-124` ⟹ `hidden_dim % emb_dim == 0`; `rms_qkv_int4_rope_multi.py:380-382`
⟹ `n_heads·head_dim == emb_dim`, `k_chunk == emb_dim`.

| Model | emb | hidden | `hidden % emb` | `n_heads·head_dim == emb` | fits |
|---|---:|---:|---:|:---:|---|
| `llama32_1b` | 2048 | 8192 | 0 | ✓ | shipped |
| `smollm2_1_7b` | 2048 | 8192 | 0 | ✓ | **yes, no new stage** (MHA: `kv_dim` 2048 vs 512; 24 vs 16 layers) |
| `qwen3_1_7b` | 2048 | 6144 | 0 | ✓ | yes on shapes; needs an int4 QK-norm stage (bf16 has `rms_qkv_qknorm_rope_multi.py`, no int4 twin; 6 → 8 launches); O-GEMV degenerates (`q_dim = 16×128 = emb`) |
| `qwen3_0_6b` | 1024 | 3072 | 0 | ✗ (2048) | no |
| `llama32_3b` | 3072 | 8192 | **2048** | ✓ | **no — doc 12's named candidate fails the FFN assert** |
| `qwen25_0_5b` / `1_5b` / `3b` | 896 / 1536 / 2048 | 4864 / 8960 / 11008 | 384 / 896 / 768 | ✓ | no |
| `qwen3_4b` | 2560 | 9728 | 1408 | ✗ (4096) | no — both |

### B.5 q4_0 on the shipped kernel — steps 1–4 and 6 `[2026-08-12]`

`q4_0`'s `d·(q − 8)` is the kernel's `(q − z)·s` with **`z ≡ 8`**, group size 32 (`DIM_GS` is already a
kernel parameter; `gs=32` legal, `NSUB = gs/r = 1`), so **no kernel change** — only the dependency-free
GGUF reader/packer `matrix_vector_multiplication/int4_awq/gguf_q4_0.py` (all-8s `Z` plane, fp16 `d` →
bf16 scale, nibble de-interleave). Device gate devq **257**: corr **0.999996** on a real SmolLM2 tensor,
**both negative controls verified failing on device**, their correlations predicted on host to four
decimals beforehand. Step 6's symmetric variant drops the `Z` plane: **bit-identical to step 4 on
device** (0/2048 differ, corr 1.000000000) at **4.750 → 4.500 bits/weight**; the retired `wrong-zero`
control is **refused** rather than left to pass vacuously. Traffic **46.50 MiB** per weight pass on this
checkpoint (48.00 on the 24×7 idealization — 165 of 168 linears are Q4_0; three `ffn_down` are Q4_1,
which the kernel cannot consume). Compute 1.6× at `gs=32`, unchanged by the variant, confirmed in AIE2P
assembly. Step 5 was then sized 3–5 weeks plus the llama.cpp RoPE un-permute (cosine **0.03 → 0.996**)
and a route for the 3 Q4_1 tensors and the Q6_K tied embedding.

### B.6 Step 5's blockers cleared `[2026-08-14]`; step 5 DONE `[2026-08-19]`

- **Checkpoint risk settled**: `bartowski/SmolLM2-1.7B-Instruct-Q4_0.gguf` exists, ungated; histogram
  `{Q6_K 1, F32 49, Q4_1 3, Q4_0 165}` reproduced exactly. Shape-exactness verified against the real
  config (`8192 % 2048 == 0`, `32 × 64 == 2048`); SmolLM2 is **full MHA** (`head_count_kv 32`), so k/v are
  2048×2048 not 2048×512. Q6_K blocker dissolves: no `output.weight`, the template takes the embedding
  from the bf16 checkpoint (`awq_pack.py:270-277`). Q4_1 route chosen by measurement (rms/rms vs the bf16
  source): accepted q4_0 band **0.0828–0.0853**; transcoded q4_1→q4_0 **0.1109–0.1124** (out of family,
  refused); re-quantized from the bf16 source **0.0869–0.0884** (in family, route (d) of `gguf_q4_0.py`'s
  decision record). `gguf_q4_0.py --self-test` 10 → 13 legs, two negative controls.
- **Step 5** (`16ae3b22`, `4cddf4fe`; devq **372–379**): `programming_examples/llms/smollm2_1_7b_int4/` —
  bf16 NPU prefill on the q4_0-dequantized weights (`rms_gemms_rope` → `flash_attn` 32q/32kv → `o_ffn`),
  int4 NPU decode at `gs=32` (`rms_qkv_int4_rope` 6 launches → CPU MHA attention → `o_gemv_ffn_int4` 3
  stages), bf16 `lm_head_gemv` (tied 49152×2048 over 8 partitions); 24 layers, `rope_theta` 130000;
  `blk.{0,1,10}.ffn_down` Q4_1 promoted. `make verify` **PASS against the PLAIN HF bf16 reference**
  (delta deliberately includes quantization error; devq 378); host chain from the same payloads devq 377;
  3-model shared-driver regression (`llama32_1b`, `llama32_1b_int4`, `smollm2_1_7b`) 3/3 PASS devq 379;
  e2e coherent at **11.1 decode tok/s**. The 3–5-week sizing collapsed because every blocker had a recorded
  route and the bf16 SmolLM2 deployment existed. Two latent SHARED-code defects fixed on the way, both
  ChatML-class: the prompt-length-by-EOS-count heuristic read logits at the wrong row for any model whose
  EOS appears inside its chat template; the per-compile kernel sweep restaged the gs=128 micro-kernel
  canonical before every aiecc link, so a gs=32 build silently linked the wrong group size (hence
  `int4_gs` threaded through the backend kwargs). Codex review of the diff clean.

### B.7 Corrections to the record (doc 36 §8, one line each)

1. Doc 12's gate condition 1 and doc 13:50 (both now [01](01-original-plan-superseded.md)) overstated the gap — `make verify` runs int4 decode for 31/32
   tokens; the true gap was int4 prefill and bfp16 end-to-end.
2. Doc 12 work item 6 (schema quant fields) was already DONE: `study/schema.py:257-274`, `SCHEMA_VERSION = 2`,
   seven `quant_*` columns; `run_mode.py:177` still hardcodes `row["dtype"] = "bf16"` and nothing produces them.
3. "Qwen3-1.7B or Llama-3.2-3B next" treated non-equivalent options as interchangeable; 3B fails the FFN
   assert; SmolLM2-1.7B (never mentioned) was the shape-exact one — and is what shipped.
4. The red fused lit was never a prerequisite (§B.3).
5. `llama32_1b_int4/README.md` was staler than doc 12 said: "two prefill backends" (three); "`make chat`
   not yet present" (it is, `Makefile:231`); the perf table at `:21-27` predates the Turbo-pmode rule and
   is unconditioned — do not lift its numbers.
6. `run_npu2_verify.lit` `REQUIRES: false` was recorded in no doc; closed §B.2.

### B.8 Effort-table verdicts (doc 12's items, now [01](01-original-plan-superseded.md)) and standing rules

| item | verdict |
|---|---|
| 1 README truth-up | minutes; staler than doc 12 said (§B.7 item 5) |
| 2 split the gates per backend | recommended first (~2 days); `make verify-prefill PREFILL_DTYPE=int4` and `run_npu2_verify_prefill_bfp16.lit` already gate two of three |
| 3 int4 prefill perf (L2 K-tiling; `tile_n=16` immediate range) | toolchain-adjacent, off any critical path |
| 4 hoist int4 builders to `shared/` | **not before a second model exists** — §B.4 shows nothing coherent to generalize toward; triggers the ~63-min ten-model regression (doc 13's "most likely to be skipped" check, [01](01-original-plan-superseded.md)) |
| 5 registry quantization axis | ~2–3 days; `matrix_vector_multiplication/int4_awq` already has on-device `PASS!` lits at 2048×2048 and 2048×8192; registry has zero quantized rows (15 `*_bf16*` detail files, `gemm_config` has no quant parameter, reference contract is CPU FP32 — a `(q−z)·s` reference must be written down); `i8`/`i16` matmul lits are compile-only |
| 6 schema fields | already done (v2) |
| 7 perf measurement definition | hours; schema v2's `device_ms`/`sync_ms`/`host_cpu_ms` |
| 8 second quantized model | **DONE** `[2026-08-19]` |
| 9 bfp16 end-to-end; 10 quantized modes through Phase F | downstream of 2 and of a `quant_*` producer |

Standing facts that bite: the 7-phase skill chain is bf16 end-to-end (`--dtype bf16|fp16`, parsed then
never read; scope gate rejects on topology only; an AWQ `config.json` still says `LlamaForCausalLM` and is
admitted then fails without a diagnostic; `_bf16` recurs 12× in phase-1; phase-7 hardcodes
`torch.bfloat16`; the int4 dir `77fdd1dc` 2026-06-03 predates the skills `dfe6573d` 2026-06-22) — a
second quantized model is infrastructure with a reusable gate, not a workflow run. Trap 0 (Turbo pmode)
applies to any latency; byte/count evidence is pmode-independent. Install/lit split ([15](15-environment-notes.md) §Which toolchain tree): `make verify`
resolves `install-xrt`, lits resolve `build-xrt`; a quantized gate should say which tree it resolves.

### B.9 Interaction with the four modes — one axis moved, one untouched (INFERENCE, testable)

The taxonomy is reconfiguration cost against DRAM traffic ([03](03-measurement-model.md)). Quantization moves DRAM traffic (packed cost ≈ **4.19 bits/weight** at g128, `4 + (16+8)/128`;
`_packed_dims` `tile_bytes = M_TILE·(K_CHUNK//2) + n_gpc·M_TILE·2 + n_gpc·M_TILE`) and leaves
reconfiguration (`context_loads`/`kernel_attaches`) untouched. Since the modes separate partly on warm
DRAM bytes (`runlist` lowest by the static weight set, 14,352,384 bytes,
[25](25-mode-rebuilds-and-results.md); `fused`'s specced prize 84.0 → 16.5 MiB at 1024,
[31](31-resident-tail-r1-record.md)), the prediction is a **narrowed spread between modes**. Only
`o_gemv_ffn_int4_fused.py` is a cascade-chained fully-fused quantized block, and it is the red lit of
§B.3. No producer of `quant_*` rows exists; the harness hardcodes `dtype = "bf16"`.
