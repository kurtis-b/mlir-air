# Goal 2 — Quantized Inference: scoping investigation

`[2026-08-12]` Read-only investigation against `exper/transformer-layer-execution-studies`, tip
`b777517b` at start. Nothing built, nothing run on device, `build-xrt` untouched; working tree
clean throughout. **A concurrent session advanced the branch to `2d6756ca` mid-investigation**
(`docs: the install is refreshed, and the divergence it caused is closed`). Only two doc files
changed — `15-environment-notes.md` and `README.md` — and no code citation below is affected;
the one consequence is folded into §5.

Spec under investigation: [`12-goal-quantized-inference.md`](../../../../home/cj/mlir-air/docs/plans/transformer-layer-execution-studies/12-goal-quantized-inference.md).
Queue row: README row **12** — "never started, never chosen; the choice has been offered and not
taken".

Every figure below carries the file it came from. Inferences are marked **[inference]**. Four of
doc 12's own statements are contradicted by the tree; those are collected in §8.

---

## 0. The one-paragraph answer

The "first" quantized model is `llms/llama32_1b_int4` and it is **more complete than doc 12
implies**: dequantization happens **on device, inside the AIE core**, one micro-kernel serves both
the prefill GEMM and the decode GEMV, and **`make verify` does exercise the quantized path — 31 of
its 32 gated tokens per prompt come out of the int4 decode ELFs.** What `make verify` does not
exercise is int4 *prefill* and bfp16 end-to-end. The red lit test doc 15 records is **beside** the
path, not on it: its cause is a missing `-D__AIE_API_AIE_ADF_HPP__` in one sub-example Makefile,
and the shipped decode path compiles the identical `.cc` with that flag already set. A second
quantized model is **not** the 7-phase deployment workflow with a different dtype — that chain is
bf16 end-to-end and does not even reject a quantized checkpoint. The int4 decode builders carry
undocumented shape asserts that **exclude 6 of the 9 other shipped models, including
Llama-3.2-3B**, which doc 12 names as a candidate. The top risk is external: an AutoAWQ safetensors
checkpoint at the right config for a second model **may not exist**, and that is checkable in an
hour with no device.

---

## 1. What the existing quantized path actually is

### 1.1 The checkpoint and its format

`amd/Llama-3.2-1B-Instruct-awq-uint4-asym-g128-bf16-lmhead` — AutoAWQ "gemm v1", uint4 asymmetric,
group size 128, bf16 lm_head. Un-gated (no `HF_TOKEN`).

- `programming_examples/llms/hf_models.txt` — the only AWQ repo in the whole list of 16.
- `programming_examples/llms/llama32_1b_int4/Makefile:37` (`MODEL ?=`).
- `programming_examples/llms/llama32_1b_int4/verify_adapter.py:55` (`_DEFAULT_AWQ_MODEL`).

HF-side tensor layout, from `llms/llama32_1b_int4/awq_repacker.py:9-16`:

```
qweight:  [in_features=K, out_features // 8] int32   (8 uint4 nibbles packed along N)
qzeros:   [K // group_size, out_features // 8] int32 (same packing)
scales:   [K // group_size, out_features] fp16
```

Nibble interleave is AWQ's `[0, 2, 4, 6, 1, 3, 5, 7]` — `awq_repacker.py:37` (`AWQ_PACK_ORDER`),
inverse at `:39`.

### 1.2 Two packers, one per compute shape

| File | Bridges to | Target layout |
|---|---|---|
| `llama32_1b_int4/awq_pack.py` | `matrix_multiplication/int4_awq/matmul_int4_packed.pack_inputs` | **prefill GEMM** packed BO |
| `llama32_1b_int4/awq_repacker.py` | `matrix_vector_multiplication/int4_awq/matvec_int4_packed.pack_inputs` | **decode GEMV** packed BO |
| `llama32_1b_int4/awq_bfp_pack.py` | the bfp16 prefill stitchers | bfp16 |

Decode target layout (`awq_repacker.py:17-22`): `A_q[M=out, K/2] uint8` (col 2i = low nibble,
2i+1 = high), `A_s[n_groups, M] bf16`, `A_z[n_groups, M] uint8`.

`awq_pack.py:10-13` states the round trip is **numerically lossless** against AutoAWQ's CUDA
dequant, because the kernel's in-tile dequant uses AWQ's own `(q-z)*s`.

`awq_pack.py` also keeps `fake_quantize_awq_int4` + `pack_weight_for_int4_gemm` for the standalone
GEMM example, which has no AWQ checkpoint of its own (`awq_pack.py:24-26`).

### 1.3 Dequantization point: **on device, in the AIE core, in-tile**

This is the single most load-bearing fact and doc 12 never states it.

`programming_examples/matrix_vector_multiplication/int4_awq/mv_int4_bf16.cc:13-14`:

```
c[0..m] += dequant(A)[m, k] @ b[k]
where dequant(A)[r, k] = (q[r, k] - z[r, g(k)]) * s_a[r, g(k)]
```

and `:175-177` — "Dequant produces UNSCALED bf16 W (just nibble unpack + zero subtract). Per
(m_b, n_b): preload the f32 c tile, then for each group do an unscaled MMUL, convert to f32 vec,
multiply by f32 scale". The per-group scale fold is the cold path, `:311`.

Consequence: **the packed nibbles cross DRAM → L2 → L1 in packed form.** The byte reduction is real
at every memory level, not only at DRAM. This is what makes quantization a DRAM-traffic statement
rather than a compute statement (§5).

### 1.4 One micro-kernel, two configurations

`mv_int4_bf16.cc` is compiled twice from the same source:

- decode GEMV: `-DDIM_M=8 -DDIM_K=2048 -DDIM_GS=128` → `mv_int4_bf16_gemv.o`
- prefill GEMM: `-DDIM_M=16` → `mv_int4_bf16_matmul.o`

both staged to the canonical `mv_int4_bf16.o` that `link_with` expects —
`llms/shared/infra/external_kernels.py:301-323`. Its includes are just `aie_api/aie.hpp` and
`stdint.h` (`mv_int4_bf16.cc:31-32`); everything else is `#ifdef`-parameterized shape.

### 1.5 Builders — all model-private, by design

`llms/llama32_1b_int4/multi_launch_builder/`:

| File | Role |
|---|---|
| `rms_qkv_int4_rope_multi.py` | **decode** RMSNorm + int4 Q/K/V GEMV + RoPE Q/K — 6 launches |
| `o_gemv_ffn_int4_multi.py` | **decode** O-GEMV + ResAdd + RMS + gate/up + SwiGLU + Down — 3 stages, 15-arg ABI |
| `rms_gemms_rope_int4_multi.py`, `o_ffn_int4_multi.py` | **prefill** int4 |
| `rms_gemms_rope_bfp16_multi.py`, `o_ffn_bfp16_multi.py` | **prefill** bfp16 |
| `o_gemv_ffn_int4_fused.py` (59 KB) + `test_o_gemv_ffn_int4_fused.cpp` | fully-**fused** decode block, three herds LA/LGU/LD on two cascade chains — **not dispatched by any model** |

The only *shared* builder the int4 model uses is `shared/builders/lm_head_gemv_multi.py`, which is
bf16 and architecture-orthogonal — `llama32_1b_int4_decode.py:118`.

`programming_examples/llms/README.md:105-106` states the split as policy:

> The `int4` example keeps its own quantized builders since the int4/bfp16 GEMM ABIs differ from
> bf16.

### 1.6 Three prefill backends, not two

`llama32_1b_int4_prefill.py:970` — `choices=["int4", "bf16", "bfp16"]`. `make compile`
(`Makefile:104-121`) compiles all three; a `compile-bfp16` fast-path exists at `:123`.

### 1.7 Does `make verify` exercise the quantized path? **Yes — the decode half.**

The chain, end to end:

1. `Makefile:161` — `verify:` runs `verify/verify_runner.py --runner=llama32_1b_int4.verify_adapter
   --prompts topk_token --max-prompts 2`.
2. `verify/verify_runner.py:70-71` — `GATE_N_TOKENS = 32`, `GATE_K = 5`.
3. `verify_runner.py:126-138` (`_generate_with_topk`) — token **0** comes from `runner.prefill()`;
   tokens **1..31** each come from `runner.decode_step()`.
4. `llama32_1b_int4/verify_adapter.py:303-318` — `Int4NpuRunner.decode_step` calls
   `run_npu_decode_step(...)`, which dispatches the int4 decode ELFs.

**So 31 of the 32 gated tokens per prompt, per gate run, are produced by int4 decode kernels
reading packed nibble weights and dequantizing in-core.**

What is *not* gated by `make verify`: int4 prefill, bfp16 prefill, bfp16 end-to-end.
`verify_adapter.py:196-202` says so in its own docstring:

> Prefill is NPU bf16 (on dequantized AWQ weights) since the int4 prefill path is currently
> kernel-bound; decode is NPU int4.

**Gate design note worth keeping.** `verify_adapter.py:90-171` (`build_hf_model`) constructs the HF
reference from the AWQ checkpoint's *config only*, then overwrites every Linear with the
AWQ-dequantized bf16. Its stated purpose (`:99-101`): "Tightens the verify gate from
(quant_error + NPU_drift) down to (NPU_drift) since both sides see exactly the same bf16 tensor
values." That is the correct construction and it is the single most reusable artifact for a second
quantized model.

### 1.8 Evidence that it passes

`agents/.state/devq/jobs/job-000222.log` (devq job 221→222, `class=measure`,
`name=ten-model-verify`, `exit=0`, ~63 min):

- `:198` — `llama32_1b_int4: pass`
- final line — `TEN-MODEL: PASS -- all 10 shipped models still verify`
- `:139-152` — the int4 decode kernels compiled in that same run
  (`Compiled rms_qkv_int4_rope`, `Compiled o_gemv_ffn_int4`), `:166` — "Pre-loading int4 decode
  weights into per-layer BOs", `:143-145` — `Q int4 GEMV (M=2048, K=2048)`,
  `K/V int4 GEMV (M=512, K=2048)`.

This is the artifact behind README queue item 1's "10/10 PASS under the new install".

### 1.9 **New finding: the int4 verify lit is disabled, alone among the ten models**

`llms/llama32_1b_int4/run_npu2_verify.lit:11` — `// REQUIRES: false`.

Every sibling is live. Checked:

| Model | `REQUIRES` |
|---|---|
| `llama32_1b` | `ryzen_ai_npu2, peano, hf_token, hfweights_meta_llama_llama_3_2_1b_instruct` |
| `qwen3_1_7b` | `ryzen_ai_npu2, peano, hf_token, hfweights_qwen_qwen3_1_7b` |
| `llama32_3b` | `ryzen_ai_npu2, peano, hf_token, hfweights_meta_llama_llama_3_2_3b_instruct` |
| **`llama32_1b_int4`** | **`false`** |

Provenance: `git log -S"REQUIRES: false"` → `18d1dac2` (2026-06-17),
**"[CI] Disable llama32_1b_int4 verify: OOM on amdhx370 runner (#1686)"** — a CI host-memory
problem, *not* an int4 correctness problem. It was wired into CI two commits earlier
(`f0a031bc`, "[llama32_1b_int4] Wire int4 verify lit into CI").

Consequence for Goal 2: **the one quantized gate in the tree runs only through the manual
ten-model `make verify` loop (doc 13:84-90), never in a lit suite.** Note that
`verify_runner.py:380-405` since added a two-subprocess `auto` gate phase whose stated purpose is
"Peak host RAM = max(NPU, HF) instead of the sum" — **[inference]** that may already have resolved
the OOM that caused the disable; testable by flipping `REQUIRES` and running the lit, no new code.

Note also: `run_npu2_verify_prefill_bfp16.lit:11` **is** live
(`REQUIRES: ryzen_ai_npu2, peano, hfweights_amd_llama_...`), so the bfp16 *prefill* backend has a
lit gate that the int4 model's own end-to-end gate does not.

### 1.9.1 `[2026-08-12]` RESOLVED — queue item 14 closed; the inference above was right, for a reason it did not name

The **[inference]** in §1.9 is **CONFIRMED, and dated**: the `auto` subprocess split is
`7f2e03d8` (2026-07-14), which is **a month AFTER** the 2026-06-18 disable. The same change is
what let `nightlyPerfBenchmark.yml` drop its 3B/4B exclusion — its `env` block now records
"Measured verify peaks: llama32_3b ~24GB, qwen25_3b ~25GB, qwen3_4b ~27GB. No LIT_FILTER_OUT
needed." The 1B int4 model was simply never revisited.

**Measured before deciding** (devq **255**; this 31 GiB NPU2 host is the same size class as the
"~32GB runner" the disable names; Turbo. Method: peak RSS summed over the whole process tree,
sampled at 5 Hz, cross-checked against `/usr/bin/time -v`'s `ru_maxrss` — the tree sum is the
OOM-relevant quantity, `ru_maxrss` only reports the largest single process):

| arm | peak tree RSS | min free | result |
|---|---|---|---|
| the lit end to end (`auto` split, what CI runs) | **10.53 GiB** (capture-npu 10.53 / compare-hf 9.59) | 17.3 GiB | **PASS**, 462.6 s |
| legacy single process (`--gate-phase both` = the pre-fix shape) | **12.57 GiB** | 15.4 GiB | **PASS** |

**The OOM does not reproduce with OR without the split**, so on this host the split is headroom
rather than the thing holding the test up. Note the naive `NPU + HF` sum of the split's two phases
is **20.12 GiB**, but the single-process arm actually measured **12.57 GiB** — the sum badly
over-estimates, because the host numpy weight duplicates are dropped once resident in BOs. That is
why this was measured rather than derived. (The 12.57 GiB arm also recompiled every kernel inside
the measured process, so it is itself an over-estimate.)

**Decision: re-enabled** with `ryzen_ai_npu2, peano,
hfweights_amd_llama_3_2_1b_instruct_awq_uint4_asym_g128_bf16_lmhead` and **no `hf_token`** —
`verify_adapter.py` resolves the tokenizer *and* the HF reference config to that one ungated AWQ
repo, which ships its own `tokenizer.json`. (`run_npu2_compile.lit`'s comment claimed the verify
path "fetches the gated upstream tokenizer"; that was stale and is corrected in the same change.)

**Two corrections to §1.9's consequence sentence.** (a) lit CI was not testing *zero* quantized
code — `run_npu2_compile.lit` is live (`REQUIRES: ryzen_ai_npu2, peano`), so quantized
**compilation** was gated; only quantized **numerics** were not. (b) The hole was worse than
"untested": `nightlyPerfBenchmark.yml` collapses a model's several `run_npu2_verify*` lits with
precedence fail > pass > skip, so this model's reported verify status came from the passing
bfp16-prefill sibling — the dashboard read green for a gate that never ran.

Also found while sweeping for the same pattern: `run_npu2_profile.lit:12` requires `hf_token`
for the same ungated AWQ checkpoint, so it skips on any runner without a token for no reason.
Reported, not changed — arming it would mean arming a gate this work did not run.

---

## 2. The red lit test — beside the path, and a one-line cause

Doc 15:166-168 records:

> `llms/llama32_1b_int4/multi_launch_builder/run_o_gemv_ffn_int4_fused_npu2_peano.lit` — kernel
> compile fails: `aie_api/adf/stream.hpp` includes `adf.h`, a Chess-only header, under Peano […]
> The int4 *model* itself verifies (10/10 run the same day includes it); only this sub-example's
> lit is red.

### 2.1 The cause is a missing compiler define, and it is documented elsewhere in the tree

`llms/shared/infra/external_kernels.py:62-77` — `_PEANO_FLAGS` names this exact include chain and
guards it:

```
# Short-circuit aie_api's ADF graph headers: aie.hpp -> aie_adf.hpp (guarded
# by __AIE_API_AIE_ADF_HPP__) -> adf/stream.hpp -> #include <adf.h>. adf.h is
# a Vitis-only header absent from the Peano include path, so without this
# guard the compile fails with "'adf.h' file not found". These compute
# kernels don't use the ADF stream API; the XRT kernel tests pass the same
# define.
"-D__AIE_API_AIE_ADF_HPP__",
```

The failing sub-example's Makefile does not pass it:

- `llms/llama32_1b_int4/multi_launch_builder/Makefile_o_gemv_ffn_int4_fused:22` —
  `PEANOWRAP2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf $(WARNING_FLAGS) -DNDEBUG -I $(AIEOPT_DIR)/include`

Every quantized sibling does:

- `matrix_vector_multiplication/int4_awq/Makefile:25` ✓
- `matrix_multiplication/int4_awq/Makefile:31` ✓
- `matrix_multiplication/bf16_x_bfp16/Makefile:26` ✓

Repo-wide census (all Makefiles defining `PEANOWRAP2P_FLAGS`): exactly **two** lack the define —
this one and `data_transfer_transpose/dma_bf16/Makefile`.

And the `.cc` it compiles is `matrix_vector_multiplication/int4_awq/mv_int4_bf16.cc` — **the same
file the shipped decode path compiles**, through `_PEANO_FLAGS`, which carries the define
(`external_kernels.py:312`).

### 2.2 Is it on Goal 2's path? **No.**

The shipped decode dispatches `multi_launch_builder.o_gemv_ffn_int4_multi.build_o_gemv_ffn_int4_module`
(`llama32_1b_int4_decode.py:102-108`). The red lit gates `o_gemv_ffn_int4_fused.py`, a *different*
design — "Three herds (LA / LGU / LD) wired by two cascade chains"
(`run_o_gemv_ffn_int4_fused_npu2_peano.lit:10-13`). No model imports it.

**Verdict: beside the path. Not a Goal 2 prerequisite, and it must not be budgeted as one.**

**[inference]** Adding `-D__AIE_API_AIE_ADF_HPP__` to that Makefile clears the reported compile
error. Whether the test then *passes on device* is untested and could unmask a second failure —
the lit's `CHECK: PASS!` requires a clean numerical run. Cost to find out: one edit and one lit
run. Worth doing opportunistically; not worth putting on a critical path.

### 2.3 Second-order relevance, worth recording

`o_gemv_ffn_int4_fused.py` is the only **cascade-chained, fully-fused, quantized** block in the
tree. It is the natural artifact if Goal 2 ever wants to say something about quantization ×
execution mode (§5). Today it does not compile. That makes §2's freebie more interesting than its
size suggests — but still not a prerequisite.

---

## 3. What a second quantized model would take

### 3.1 Reusable as-is

- Both AWQ bridges (`awq_pack.py`, `awq_repacker.py`, `awq_bfp_pack.py`) — these are *format*-level,
  not model-level. `awq_repacker.py` ships a self-test (`python3 awq_repacker.py`) that verifies the
  repack dequantizes to the same bf16 as a direct dense dequant (`awq_repacker.py:21-24`).
- The device micro-kernel `mv_int4_bf16.cc` — shape-parameterized by `-DDIM_M/-DDIM_K/-DDIM_GS`.
- The verify-adapter construction (`verify_adapter.py:90-171`) — HF reference built from the AWQ
  config + dequantized weights, isolating NPU drift. **The most valuable reusable artifact.**
- `shared/builders/lm_head_gemv_multi.py`.

### 3.2 Model-specific and must be written

- The HF key remap in `llama32_1b_int4_weights.py:123-138`
  (`model.layers.{i}.input_layernorm.weight`, `{base}.qweight/.qzeros/.scales`) — same class of
  work phase-0 already describes for bf16 loaders, plus the three quantized tensors.
- The decode builders, per §3.3.

### 3.3 **The shape contract nobody has written down — and it excludes Llama-3.2-3B**

Extracted from the asserts. These are hard `assert`s, not warnings.

`o_gemv_ffn_int4_multi.py:119-124`:
```python
assert emb_dim % k_chunk == 0 and hidden_dim % k_chunk == 0
assert emb_dim == k_chunk, "Stage 2 int4 swiglu_rms requires emb_dim == k_chunk"
```
→ with `k_chunk` free, this reduces to **`hidden_dim % emb_dim == 0`**.

`rms_qkv_int4_rope_multi.py:380-382`:
```python
assert q_total == emb_dim          # n_heads * head_dim == emb_dim
assert k_total == kv_dim           # n_kv_heads * head_dim == kv_dim
assert k_chunk == emb_dim, "K_CHUNK must equal emb_dim for single-chunk GEMV"
```

Applied to every shipped model (configs read from each `<model>/<model>_weights.py`):

| Model | emb | hidden | `hidden % emb` | `n_heads·head_dim` | `== emb`? | Fits int4 decode builders? |
|---|---:|---:|---:|---:|:---:|---|
| `llama32_1b` | 2048 | 8192 | 0 ✓ | 2048 | ✓ | **shipped** |
| `smollm2_1_7b` | 2048 | 8192 | 0 ✓ | 2048 | ✓ | **YES — no new stage** |
| `qwen3_1_7b` | 2048 | 6144 | 0 ✓ | 2048 | ✓ | **YES on shapes; needs a QK-norm stage** |
| `qwen3_0_6b` | 1024 | 3072 | 0 ✓ | 2048 | ✗ | no — QKV assert fails |
| `llama32_3b` | 3072 | 8192 | **2048 ✗** | 3072 | ✓ | **NO — FFN assert fails** |
| `qwen25_0_5b` | 896 | 4864 | 384 ✗ | 896 | ✓ | no |
| `qwen25_1_5b` | 1536 | 8960 | 896 ✗ | 1536 | ✓ | no |
| `qwen25_3b` | 2048 | 11008 | 768 ✗ | 2048 | ✓ | no |
| `qwen3_4b` | 2560 | 9728 | 1408 ✗ | 4096 | ✗ | no — fails both |

**This contradicts doc 12.** Doc 12 §3 and work item 8 say "Target Qwen3-1.7B or Llama-3.2-3B
next", as if interchangeable. Llama-3.2-3B is the one that **cannot** use the shipped int4 decode
builders: `8192 % 3072 = 2048`, tripping `o_gemv_ffn_int4_multi.py:119` before anything else runs.

### 3.4 The two real candidates

**SmolLM2-1.7B — cheapest structurally.** Identical `emb_dim`/`hidden_dim`/`head_dim` to
Llama-3.2-1B. Pure Llama-architecture decoder: RMSNorm + SwiGLU + RoPE, no QK-norm, no QKV bias.
Differs only in `n_kv_heads` (32 MHA vs 8 GQA → `kv_dim` 2048 vs 512, which
`build_rms_qkv_int4_rope_module` takes as a parameter, `rms_qkv_int4_rope_multi.py:363`) and
`n_layers` (24 vs 16). **Zero new builder stages.** Blocker is the checkpoint, not the code.

**Qwen3-1.7B — doc 12's candidate; fits the shapes, needs one new stage.** Passes every assert at
the shipped `k_chunk=2048`. Needs:
- a QK-norm stage in the int4 QKV builder. bf16 has `shared/builders/rms_qkv_qknorm_rope_multi.py`
  (used at `qwen3_1_7b/qwen3_1_7b_decode.py:53`); there is **no int4 counterpart**. A 6-launch ELF
  becomes 8-launch.
- the decoupled O-GEMV (`qwen3_1_7b_decode.py:95`, `build_o_gemv_ffn_qwen_module(emb_dim, q_dim,
  hidden_dim)`) — for 1.7B this **degenerates**: `q_dim = 16 × 128 = 2048 = emb_dim`. So no work.

### 3.5 The checkpoint question — the actual gating risk

`hf_models.txt` lists exactly one AWQ repo. Doc 12's Risk 3 names this and is right to.

A web check of AMD's HF org finds many `awq-uint4-asym-g128` checkpoints, but the small-model
siblings are predominantly **ONNX-packaged** (`-onnx-hybrid`, `-onnx-ryzen-strix`), not the
AutoAWQ safetensors gemm-v1 `qweight/qzeros/scales` layout `awq_repacker.py` consumes. Examples
found: `amd/Llama-3.2-3B-Instruct-awq-g128-int4-asym-bf16-onnx-ryzen-strix`,
`amd/Qwen2.5-7B-Instruct-awq-uint4-asym-g128-lmhead-g32-fp16-onnx-hybrid`. Third-party AutoAWQ
safetensors checkpoints exist at larger sizes (`hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`).

**I could not confirm a non-ONNX AutoAWQ safetensors checkpoint at the right config for
Qwen3-1.7B, SmolLM2-1.7B, or Llama-3.2-3B.** Marked **unconfirmed**, not "absent" — a search-result
list is not an org listing. Resolvable in under an hour with the HF hub API, no device, no build.
**This should be step 0 of any Goal 2 decision**, because the answer can invert the model choice or
force locally quantizing a model with `autoawq` (CPU/GPU work with its own correctness question,
unrelated to MLIR-AIR).

### 3.6 Registry state — zero quantized rows

| Fact | Cite |
|---|---|
| All 15 files in `kernel_registry/details/` are `*_bf16*`; no int4/AWQ/i8/i16/bfp16 detail file | `programming_examples/kernel_registry/details/` |
| `grep -rniE "int4\|awq"` across the whole registry tree returns **zero** hits | — |
| `gemm_config(M, K, N, output_dtype="bf16", precision="high")` — **no quantization parameter**; `output_dtype` hard-gated to `bf16`/`f32` | `kernel_registry/registry_lookup.py:67`, `:35-38` |
| The only extension guidance is per-*kernel*, not per-*dtype*: "Currently provides GEMM lookups; other kernels add their own as their JSON lands." | `registry_lookup.py:12` |
| Reference standard is CPU FP32 with bf16 upcast — a quantized kernel's reference is a different contract that has to be written down | `kernel_registry/README.md:34` |

Nine quantized example dirs exist and all have working run targets + lit tests, none registered:

| Directory | Run target | lit | In registry |
|---|---|---|---|
| `matrix_multiplication/int4_awq` | `run_packed`, `run_llama_qproj`, … | 2 (`CHECK: PASS!`) | ✗ |
| `matrix_multiplication/bf16_x_bfp16` | `run`, `profile` | 2 (`PASS!`) | ✗ |
| `matrix_multiplication/i8` | `run8x4` … | 7 (**compile-only CHECK**) | ✗ |
| `matrix_multiplication/i16` | `run8x4` … | 7 (**compile-only CHECK**) | ✗ |
| `matrix_vector_multiplication/int4_awq` | `run_packed`, `run_packed_add` | 3 (`PASS!`) | ✗ |
| `vector_matrix_multiplication/i8/single_core` | `run` | 2 (`PASS!`) | ✗ |
| `vector_matrix_multiplication/block_quantized_i8/single_core` | `run` | 2 (`PASS!`) | ✗ |
| `dequant_awq` | `run`, `profile` | 5 (`PASS!`) | ✗ |
| `decode_ffn_swiglu` (`matvec_int4_swiglu_rms.py`) | `run_int4` | 1 (`PASS!`) | ✗ |

Two path corrections vs. doc 12's list: `vector_matrix_multiplication/{i8,block_quantized_i8}` are
parent dirs; the buildable unit is `single_core/` in each. And note the `i8`/`i16` matmuls are a
**weaker tier** — their lits assert only `Compilation completed successfully!`, so they carry no
on-device numerical verification at all.

**The useful consequence:** `matrix_vector_multiplication/int4_awq` already has *on-device
`PASS!` lits at 2048×2048 and 2048×8192* — the registry's own harness standard
(`kernel_registry/README.md:32`). Registering it is packaging existing green runs, not new
measurement.

---

## 4. Does the deployment skill chain apply? **Mostly no.**

This materially *raises* the effort estimate versus the framing in the task brief.

| Question | Finding | Cite |
|---|---|---|
| `--dtype` values | `bf16\|fp16` only. No quantized value. And the flag is parsed at Step 1 then **never referenced again** in any phase — even `fp16` is nominal | `.claude/skills/deploy-new-llm/SKILL.md:76` |
| Scope gate | Rejects on **topology only** (MoE, sliding-window, MLA, encoder-decoder). Never on weight dtype | `deploy-new-llm/SKILL.md:78-93` |
| Phase-0 allowlist | `["LlamaForCausalLM", "MistralForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM"]` — an AWQ checkpoint's `config.json` still says `LlamaForCausalLM`, so it is **admitted and then fails downstream with no diagnostic** | `phase-0-build-cpu-reference/SKILL.md:106-109` |
| Quantization mentioned? | **Zero substantive hits** for `int4\|awq\|quant\|bfp16\|i8\|dequant` across all 9 phase skills. Two incidental literature citations only, both about why a cosine threshold is what it is | `phase-2/SKILL.md:71`, `phase-3/SKILL.md:65` |
| Phase-0 loader | Copy-and-edit off the dense bf16 loader; enumerated variations are config defaults, weight-name remap, `rope_base`, `tie_word_embeddings`, QKV bias. No `qweight`/`qzeros`/`scales`, no group size, no dequant | `phase-0/SKILL.md:113-134` |
| Phase-1 registry row | `details/<Kernel>_bf16.md` — the literal `_bf16` recurs **12×** in that one file. A quantized deployment has nowhere schema-conformant to record a row | `phase-1-kernel-validation/SKILL.md:88-92` |
| Phase-7 audit | Hardcodes `torch_dtype=torch.bfloat16` | `phase-7/SKILL.md:38,102,150` |
| Was the int4 model built via the chain? | **No.** int4 dir created `77fdd1dc` 2026-06-03; skills first committed `dfe6573d` 2026-06-22. No `llama32_1b_int4/docs/`, no `TODO.md`/`ARCHITECTURE.md`/`evaluation_report.md`. Zero mentions of `int4` in any skill | — |

Plus a live breakage that doc 13:113-115 already flags as a chore:
`.claude/skills/deploy-new-llm/SKILL.md:97-107` does
`test -d programming_examples/llms/llama_kernel_builder` and treats MISSING as fatal. That path was
renamed to `llms/shared/` in `2f20c2fa`, so **the `.claude` entry point hard-halts at Step 3 on
today's tree.** 11 of 15 `.claude` SKILL.md files carry the dead path; `.codex/skills/` is the
migrated copy and is correct.

**What does transfer:** the *gate methodology*. Phases 2, 3, 6, 7 gate on per-layer cosine and
top-k token-set inclusion against an HF reference — dtype-agnostic contracts, and
`verify_adapter.py:90-171` already shows how to build that reference from dequantized AWQ. **What
does not:** Phase 0 (loader), Phase 1 (registry), Phases 4–5 (optimization recipes assume the bf16
builders).

Honest framing: **a second quantized model is not a workflow instantiation, it is new
infrastructure with a reusable gate.**

---

## 5. Interaction with the four modes — not orthogonal, and on exactly one of the two axes

The taxonomy is **reconfiguration cost against DRAM traffic** (README §The one-paragraph version;
`03-measurement-model.md` §The taxonomy). Quantization moves **one** of those two axes and leaves
the other alone:

- **DRAM traffic: moved.** Because dequant is in-core (§1.3), packed nibbles cross L3, L2 and L1.
  Per weight element the packed cost is 4 bits plus `(16-bit scale + 8-bit zero)/128` ≈ **4.19
  bits**, against bf16's 16. The builder already computes the per-tile figure —
  `o_gemv_ffn_int4_multi.py:_packed_dims` → `tile_bytes = M_TILE*(K_CHUNK//2) + n_gpc*M_TILE*2 +
  n_gpc*M_TILE`. **No latency claim is made here**; this is a byte claim, structural, and checkable
  without hardware.
- **Reconfiguration: untouched.** `context_loads` / `kernel_attaches` are functions of how many
  xclbins and instruction streams the schedule needs — independent of weight dtype.

Why this matters rather than being a footnote: the modes are separated in part by **warm DRAM
bytes**. README queue item 4 moved `runlist` to lowest warm bytes by exactly the static weight set
(14,352,384 bytes, `30-coarse-cells-built.md`), and `fused`'s specced prize is 84.0 → 16.5 MiB of
DRAM crossings at 1024 (README item 6, `31-fused-resident-tail.md`).

**[inference, testable]** Quantization shrinks the *weight* component of that traffic ~4× while
leaving the *activation/intermediate* component untouched. It therefore does **not** scale the four
modes uniformly: it compresses the axis on which `fused` wins and leaves `offload`'s reconfiguration
advantage intact — so the expected effect is a **narrowed spread between modes**. That is a genuine
study result, distinct from any deployment result, and it is the strongest intellectual argument
for Goal 2 within this study.

The harness is not ready to produce it:

- `study/run_mode.py:177` — `row["dtype"] = "bf16"`, hardcoded.
- `grep -rn "quant_"` across `programming_examples/transformer_layer/` outside `schema.py` returns
  **zero producers**.

**But the schema already carries the fields.** `study/schema.py:257-274` defines all seven columns
Codex specified in doc 03 §Quantization fields — `quant_packing_scheme`, `quant_group_size`,
`quant_scale_layout`, `quant_zero_point_layout`, `quant_accum_type`, and separate
`quant_gemm_contract` / `quant_gemv_contract` — with the reasoning recorded at `schema.py:50-52`:

> The quantization fields are here NOW and empty for bf16 rows (doc 03). Bolting them on later
> renumbers the schema and invalidates every row already written; a `dtype` column alone cannot
> describe a quantized run.

`SCHEMA_VERSION = 2` (`schema.py:71`). **Doc 12 work item 6 is done.** What remains is a producer.

### Standing rules that bite Goal 2

- **Cross-deployment regression rule** (doc 13:65-93): doc 12 work item 4 (hoist int4 builders into
  `shared/`) triggers a ten-model re-verify. job-000222's metadata puts that at ~63 min of
  *exclusive* device time. Doc 12 §3 carries `[Codex]`'s warning to generalize *before* hoisting.
  Given §3.3 — the shipped int4 builders have hard shape asserts only two other models satisfy —
  hoisting them today would move special cases into shared infrastructure, which is exactly the
  failure that warning names. **Item 4 should not be attempted before there is a second model to
  generalize against.**
- **Trap 0 (Turbo pmode)** applies to any measurement. Goal 2's first increments should produce
  byte/count evidence, which the README cold-start notes is pmode-independent.
- **Install/lit split** (doc 15 §Which toolchain tree): `make verify` for the models resolves
  `install-xrt`; the lit suites resolve `build-xrt`. `[2026-08-12, updated mid-investigation]`
  The two trees **were** four days apart when this investigation started (tip `b777517b`); a
  concurrent session refreshed the install in `2d6756ca` and doc 15:195-214 now records them as
  agreeing (both 2026-08-11 13:28, verified by probe artifact rather than timestamp). So a Goal 2
  gate expressed as a lit and one expressed as `make verify` currently test the *same* compiler —
  but the split is structural and reopens the moment `mlir/` changes, so a quantized gate should
  say which tree it resolves.

---

## 6. Effort, risk, and the smallest first increment

### Three candidate increments, ranked

**(A) Split the int4 gate — ~2 days, no new model, no new kernel.** Doc 12 work item 2.
Most pieces exist: `make verify-prefill PREFILL_DTYPE=int4` (`Makefile:138`) and
`run_npu2_verify_prefill_bfp16.lit` already gate two of the three backends. Missing: an `int4`
sibling of the bfp16 prefill lit; ~~a decision on `run_npu2_verify.lit`'s `REQUIRES: false`~~
(**settled `[2026-08-12]`, §1.9.1: measured, re-enabled, and green — it was the `auto` gate phase,
which post-dates the disable by a month**); and three named PASS/FAIL lines instead of
one. **Evidence produced:** for the first time the repo can state, per backend, which quantized
paths are gated. This is also the honest reading of Goal 2's own gate text.

**(B) Register the int4 GEMV as the registry's first quantized row — ~2-3 days.** Doc 12 work
item 5. `matrix_vector_multiplication/int4_awq` already has on-device `PASS!` lits at 2048×2048 and
2048×8192 — the registry's own harness standard. Work is `details/GEMV_int4_awq.{md,json}` + a
quantization axis on `gemm_config`. **Real design risk:** the registry's reference precision is
CPU FP32 with bf16 upcast (`kernel_registry/README.md:34`); a quantized kernel's reference is
`(q-z)*s` in FP32 — a different contract that must be written down, not assumed. **Evidence
produced:** the schema for recording quantized kernels, a prerequisite for both a second model and
the study's quant columns.

**(C) A second quantized model — 3-5 weeks, and externally gated.** The actual Goal 2 gate.
Sequence: confirm a compatible AutoAWQ safetensors checkpoint (§3.5, hours, no device) → pick per
§3.3's table → weight remap + repack → decode builder (SmolLM2: re-parameterize; Qwen3-1.7B: new
QK-norm stage) → verify adapter → gate.

### Recommendation

**Do (A), then (B) — and make the first hour of (A) the checkpoint check from §3.5, because its
answer decides whether (C) is possible at all.**

Rationale: (A) and (B) are self-contained, produce durable artifacts, need no new checkpoint and no
new kernel, and are both prerequisites for (C) being *verifiable* rather than merely built. The
checkpoint check costs an hour and can invert the entire plan.

### Effort table

| Doc 12 item | Assessment |
|---|---|
| 1 — fix `llama32_1b_int4/README.md` | Minutes. **Staler than doc 12 says** — see §8. |
| 2 — split correctness gates | ~2 days. **Recommended first.** |
| 3 — int4 prefill perf (L2 K-tiling; `tile_n=16` immediate range) | High variance. Both causes toolchain-adjacent; doc 12's own risk section says the Peano immediate-range one may have no clean source-level workaround. **Keep off any critical path.** |
| 4 — generalize + hoist builders to `shared/` | **Do not start before item 8.** Triggers the ~63-min ten-model regression; §3.3 shows there is nothing coherent to generalize toward yet. |
| 5 — registry quantization axis | ~2-3 days. **Recommended second.** |
| 6 — schema quantization fields | **ALREADY DONE** (`schema.py:257-274`, v2). Remaining work is a producer, not fields. |
| 7 — define the perf measurement | Hours; doc 03's schema v2 decomposition (`device_ms`/`sync_ms`/`host_cpu_ms`) already supplies the vocabulary. |
| 8 — second quantized model | 3-5 weeks **with** a checkpoint; indefinite without. |
| 9 — BFP16 prefill+decode end-to-end | Downstream of 2. |
| 10 — measure quantized modes through Phase F | Downstream of 6-producer + 8. |

### Risks, Goal-2-specific

1. **Checkpoint availability — top risk, unresolved, cheap to resolve.** Unlike the other queue
   items' risks it lives outside the repo. §3.5.
2. **The int4 builders' shape asserts are undocumented and exclude 6 of 9 candidates**, including
   the one doc 12 names. Anyone scoping from the spec picks wrong. §3.3.
3. **The one quantized gate is disabled in lit and nobody's docs say so.** A second quantized model
   inherits the same host-memory shape. §1.9.
4. **Hoisting to `shared/` triggers the most expensive check in the plan** (doc 13:90 — "the one
   most likely to be skipped").
5. **The skill chain admits a quantized checkpoint and then fails without a diagnostic** (§4) —
   a trap for whoever tries the obvious thing first.

---

## 7. Attractiveness on its own terms

*(Phase G and Goal 1 are under parallel investigation; nothing here speculates about their
findings.)*

**For:**

- **Least invention required per unit of gate.** A working end-to-end quantized model, a device-side
  dequant micro-kernel serving both GEMM and GEMV, both AWQ packers, an ungated checkpoint, and a
  verify adapter whose reference construction is exactly what a second model needs. Nothing has to
  be invented at the kernel level.
- **Three cheap, independently valuable increments** (§6 A/B, plus the README truth-up) that leave
  artifacts even if the second model never lands.
- **Its schema work is already paid** (v2, §5) — no versioning cost, no invalidated rows.
- **It is the only unstarted item that would put a second data point on the study's DRAM-traffic
  axis**, which is half the taxonomy — and §5's narrowed-spread prediction is a real, testable
  study result rather than a deployment result.
- **Zero coupling.** Independent of 6c, of the owed `ninja -C build-xrt install`, and of exclusive
  device time except for the ten-model regression.

**Against:**

- **Its headline gate depends on an artifact outside the repository.** No other queue item has an
  external dependency of this kind, and I could not confirm the artifact exists (§3.5). If it does
  not, the gate requires locally quantizing a model — real work, its own correctness question, and
  nothing to do with MLIR-AIR.
- **The repo's own answer to "how do we add a model" does not cover quantization at any phase**
  (§4), so the second model is new infrastructure, not a workflow run.
- **The gate has two conditions and one may be structurally unreachable.** Condition 2 (int4 prefill
  materially closer to bf16) rests on two causes doc 12 itself flags as possibly having no clean
  source-level fix. A two-condition gate with one unreachable condition is a goal that can sit at
  90% indefinitely. **If Goal 2 is taken, condition 2 should be renegotiated up front** — either
  split into its own goal or restated as "measured and attributed" rather than "materially closer".
- **It closes no definitional gap.** Unlike item 6c, nothing in the four modes is blocked on it.
- **Doc 12 is the least-maintained spec in the directory** — last modified 2026-08-03, no dated
  retractions, and four statements contradicted by the tree (§8). Anyone starting from it starts
  from a document rather than the code, which is exactly what its own §"Correct the record first"
  warns against.

---

## 8. Corrections to the record

Per the directory's convention of dated retractions rather than deletions. Five items — four in
doc 12, one in the model README that doc 12 partially catches.

1. **Doc 12's Gate condition 1 and doc 13:50 overstate a true narrower point.** "A gate that
   actually exercises the quantized path — not bf16 prefill on dequantized weights" reads as though
   `make verify` touches no quantized machinery. It does: **31 of 32 gated tokens per prompt come
   out of the int4 decode ELFs** (§1.7). The accurate statement of the gap is *int4 prefill* and
   *bfp16 end-to-end*, which doc 12's own §"The gate is weaker than it looks" states correctly.
   The gate text should be tightened to match its own body.

2. **Doc 12 work item 6 — "Add the quantization fields to study schema v1" — is DONE.**
   `study/schema.py:257-274` carries all seven fields at `SCHEMA_VERSION = 2`, with the reasoning
   recorded at `:50-52`. What remains is a *producer*; `study/run_mode.py:177` hardcodes
   `row["dtype"] = "bf16"` and nothing writes a `quant_*` column.

3. **Doc 12 §3 / work item 8 — "Target Qwen3-1.7B or Llama-3.2-3B next" — treats two
   non-equivalent options as interchangeable, and one of them does not fit.** Llama-3.2-3B trips
   `o_gemv_ffn_int4_multi.py:119` (`8192 % 3072 ≠ 0`). Qwen3-1.7B fits every assert but needs a
   QK-norm stage that has no int4 counterpart. **SmolLM2-1.7B, which doc 12 never mentions, is the
   shape-exact candidate requiring no new builder stage.** §3.3-3.4.

4. **Doc 12's implied framing of the red lit as potentially prerequisite is not supported.** The
   failing sub-example is `o_gemv_ffn_int4_fused.py`, which no model dispatches; the shipped decode
   uses `o_gemv_ffn_int4_multi.py`. The cause is a missing `-D__AIE_API_AIE_ADF_HPP__` that the
   shipped path already sets and that every quantized sibling Makefile carries. §2.

5. **`llama32_1b_int4/README.md` is staler than doc 12's §"Correct the record first" says.** Doc 12
   catches two things (the decode "follow-up PR" claim at `README.md:10,33`; the
   `../llama_kernel_builder/` path at `:103`). Three more:
   - `:8` — "ships **two** prefill backends". There are **three**;
     `llama32_1b_int4_prefill.py:970` has `choices=["int4","bf16","bfp16"]` and `Makefile:104-121`
     compiles all three.
   - `:88` — "`make chat` is not yet present". It is: `Makefile:231`.
   - The performance table at `:21-27` predates the Turbo-pmode rule (README trap 0) and carries no
     pmode condition. Any figure taken from it is unconditioned. *(No latency claim is made in this
     document; this note exists so the next reader does not lift those numbers.)*

**Also newly recorded, in no doc:** `llms/llama32_1b_int4/run_npu2_verify.lit:11` was
`REQUIRES: false` — disabled upstream in `18d1dac2` (2026-06-17) for **CI-runner OOM**, not for any
int4 defect. It was the only one of the ten models whose verify lit was disabled. §1.9.
**`[2026-08-12]` CLOSED (queue item 14): measured on this 31 GiB host, the OOM does not reproduce
with or without the subprocess split (10.53 / 12.57 GiB peak, devq 255), and the lit is re-enabled
and passing. §1.9.1.**

---

## 9. Artifact index

| Claim | File |
|---|---|
| In-core dequant formula | `programming_examples/matrix_vector_multiplication/int4_awq/mv_int4_bf16.cc:13-14,175-177` |
| One micro-kernel, two `-DDIM_M` configs | `programming_examples/llms/shared/infra/external_kernels.py:301-323` |
| adf.h guard + its documented include chain | `llms/shared/infra/external_kernels.py:62-77` |
| Missing guard in the red test's Makefile | `llms/llama32_1b_int4/multi_launch_builder/Makefile_o_gemv_ffn_int4_fused:22` |
| Guard present in every quantized sibling | `matrix_vector_multiplication/int4_awq/Makefile:25`; `matrix_multiplication/int4_awq/Makefile:31`; `matrix_multiplication/bf16_x_bfp16/Makefile:26` |
| Decode dispatches the *multi*, not the *fused*, builder | `llms/llama32_1b_int4/llama32_1b_int4_decode.py:102-108` |
| Gate = 32 tokens × k=5; 31 tokens from decode | `llms/verify/verify_runner.py:70-71,126-138` |
| Decode step runs int4 ELFs | `llms/llama32_1b_int4/verify_adapter.py:303-318` |
| Prefill is bf16-on-dequant, by design | `llms/llama32_1b_int4/verify_adapter.py:196-202` |
| HF reference patched with AWQ-dequant to isolate NPU drift | `llms/llama32_1b_int4/verify_adapter.py:90-171` |
| int4 model passes; ten models pass | `agents/.state/devq/jobs/job-000222.log:198` + final line; `job-000222.meta` (`exit=0`) |
| int4 verify lit disabled (as filed) | `llms/llama32_1b_int4/run_npu2_verify.lit:11`; commit `18d1dac2` |
| ~~disabled~~ **re-enabled and green `[2026-08-12]`**; OOM does not reproduce (10.53 GiB split / 12.57 GiB single-process peak, 31 GiB host) | `agents/.state/devq/jobs/job-000255.log` (`PASS ... run_npu2_verify.lit`, both arms + both sampler summaries); `job-000255.meta` (`exit=0`) |
| the `auto` split post-dates the disable by a month | `18d1dac2` 2026-06-18 vs `7f2e03d8` 2026-07-14; `.github/workflows/nightlyPerfBenchmark.yml:30-36` |
| bfp16 prefill lit live | `llms/llama32_1b_int4/run_npu2_verify_prefill_bfp16.lit:11` |
| FFN-side shape assert | `llms/llama32_1b_int4/multi_launch_builder/o_gemv_ffn_int4_multi.py:119-124` |
| QKV-side shape asserts | `llms/llama32_1b_int4/multi_launch_builder/rms_qkv_int4_rope_multi.py:380-382` |
| Model configs used in §3.3's table | each `programming_examples/llms/<model>/<model>_weights.py` |
| Qwen3 uses a QK-norm builder with no int4 counterpart | `llms/qwen3_1_7b/qwen3_1_7b_decode.py:53`; `llms/shared/builders/rms_qkv_qknorm_rope_multi.py` |
| Registry has no quantization parameter | `kernel_registry/registry_lookup.py:67,35-38,12` |
| Registry reference-precision contract | `kernel_registry/README.md:34` |
| Quant schema fields exist at v2 | `programming_examples/transformer_layer/study/schema.py:50-52,71,257-274` |
| Study hardcodes bf16 | `programming_examples/transformer_layer/study/run_mode.py:177` |
| Skill chain dtype values | `.claude/skills/deploy-new-llm/SKILL.md:76` |
| Phase-0 architecture allowlist | `.claude/skills/phase-0-build-cpu-reference/SKILL.md:106-109` |
| Phase-1 hardcodes `_bf16` registry pages | `.claude/skills/phase-1-kernel-validation/SKILL.md:88-92` |
| `.claude` entry point halts on the renamed path | `.claude/skills/deploy-new-llm/SKILL.md:97-107` |
| int4 model predates the skill chain | `77fdd1dc` (2026-06-03) vs `dfe6573d` (2026-06-22) |
| int4 builders are private by policy | `programming_examples/llms/README.md:105-106` |
| Red-test baseline of three | `docs/plans/.../15-environment-notes.md:166-168` |
| Ten-model regression rule | `docs/plans/.../13-verification-and-acceptance.md:65-93` |

External sources consulted for §3.5 (checkpoint availability):
[amd/Llama-3.2-3B-Instruct-awq-g128-int4-asym-bf16-onnx-ryzen-strix](https://huggingface.co/amd/Llama-3.2-3B-Instruct-awq-g128-int4-asym-bf16-onnx-ryzen-strix),
[amd/Qwen2.5-7B-Instruct-awq-uint4-asym-g128-lmhead-g32-fp16-onnx-hybrid](https://huggingface.co/amd/Qwen2.5-7B-Instruct-awq-uint4-asym-g128-lmhead-g32-fp16-onnx-hybrid),
[hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4](https://huggingface.co/hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4).
</content>
</invoke>
