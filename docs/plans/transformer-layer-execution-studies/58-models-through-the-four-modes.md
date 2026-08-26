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

## 2. The cost-halving fact

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

**Second family (Llama-1B, GQA + no QK-norm) only after M4**, and only if M0's estimate holds.

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
