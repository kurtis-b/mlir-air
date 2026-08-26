# LLAMA-3.2-1B int4-AWQ Prefill on AMD NPU2

End-to-end prefill of an AWQ-uint4 quantized LLAMA-3.2-1B (e.g.
`amd/Llama-3.2-1B-Instruct-awq-uint4-asym-g128-bf16-lmhead`) on AMD
NPU2 (AIE2P) hardware via MLIR-AIR, with a top-K correctness gate
against a CPU bf16 reference built from the same dequantized weights.

The example ships two prefill backends behind a `--prefill-dtype` flag.
The bf16 path is the recommended one for prefill today; the int4 prefill
path is preserved for its kernel work. End-to-end inference
(`llama32_1b_int4_inference.py`, `make run-inference` / `make chat`) runs
bf16 NPU prefill on dequantized AWQ weights followed by **int4 NPU decode**
(`rms_qkv_int4_rope` + `o_gemv_ffn_int4` ELFs), 68–89 ms/token by context
(14.7 / 13.6 / 11.2 tok/s at ctx 512/1024/2048; Turbo recorded, devq 610,
doc 56 §4 H2a).

## Performance

NPU2 (AMD Strix), seq=2048, 16 layers, NPU flash attention, prompt
"The capital of France is":

| Backend (`--prefill-dtype`) | per-layer | end-to-end | top-10 vs HF | argmax |
|---|---|---|---|---|
| `bf16` (default) | 84 ms | **1.38 s** | 8/10 | "Paris" ✓ |
| `int4`           | 698 ms | 11.2 s | 8/10 | "Paris" ✓ |

Both backends consume the same AWQ checkpoint and produce identical
AWQ-quality output. The 8× gap is structural to the current int4 GEMM
kernel:

- Down GEMM hits the memtile L2 budget at K=8192, capping `herd_m=2`
  (8 PEs vs 32) — `matmul_int4_packed.py` can't tile `K_L2 < K`.
- Peano AIE2P `VLD_x_pstm_nrm_imm` 9-bit immediate range forces
  `tile_n=16` (16× more launch iterations than bf16's `tile_n=128`).

Decode streams int4 weights (a 4× narrower payload than bf16 before
scales/zero-points); the int4 decode driver is `llama32_1b_int4_decode.py`,
wired by `llama32_1b_int4_inference.py`. Measured under the study runner
(Turbo recorded, the forward-pass clock; doc 56 §4 H2a, 2026-08-26, devq
609/610): **68.2 / 73.6 / 89.2 ms/token at context 512/1024/2048** (14.7 /
13.6 / 11.2 tok/s) against the bf16 sibling's 92.0 / 98.6 / 112.5 —
1.35×/1.34×/1.26×, not the ~4× the byte ratio alone would predict. The
attribution (doc 56 §4 H2a): the device's ~57 ms is a 25.7 ms weight-stream
floor + 16.3 ms of launch boundaries (152 × 107 µs) + ~15.5 ms of
dequant-bound GEMV time, nearly all in the `o_gemv_ffn_int4` line (14.5 GB/s
effective vs the machine's 40.8); host attention (7.5 → 25.5 ms by context)
rides on top. The earlier "~56 ms/token (17.8 tok/s)" here was the June
header — pmode-unrecorded, no artifact in tree.

## Prerequisites

1. **MLIR-AIR base environment** — AMD NPU2 hardware, Peano compiler,
   the project's standard env: `source utils/env_setup.sh ...`

2. **Extra Python packages**:
   ```bash
   pip install -r requirements.txt
   ```
   Installs `safetensors`, `huggingface_hub`, `transformers`, `torch`.

3. **AWQ checkpoint** — the default
   (`amd/Llama-3.2-1B-Instruct-awq-uint4-asym-g128-bf16-lmhead`) is
   **not gated** and downloads without a token. The HF reference path
   does pull the upstream tokenizer behind the AWQ checkpoint, which
   may be gated — in that case `huggingface-cli login` first.

## Quick Start

```bash
# Compile both prefill backends (int4 + bf16). ~1-2 min the first time
# (BF16 stitchers compile fast; int4 stitchers take longer on Down GEMM).
make compile

# Run NPU prefill end-to-end with the bf16 backend (default, ~1.4 s).
# Prints HF top-K, NPU top-K, overlap and argmax match.
make run

# Same but with the int4 backend (~11 s; same AWQ-quality output).
make run PREFILL_DTYPE=int4

# Run with per-kernel + per-layer profiling breakdown.
make profile

# Top-K correctness gate (used by run_npu2_verify.lit). PASS iff overlap
# >= MIN_OVERLAP (default 6) AND argmax matches HF.
make verify
make verify-int4    # same gate against the int4 backend

# Multi-prompt sweep: runs each prompt in PROMPTS_FILE (defaults to the
# bf16 sibling's verify/prompts/instruct.txt). PASS iff EVERY prompt
# meets the per-prompt overlap + argmax criteria.
make verify-full

# Diagnosis lens: per-layer ffn_out cosine vs HF bf16 reference. Single
# prompt, informational only (no PASS/FAIL gate). Last-layer cosine is
# computed post-final-RMSNorm to match HF's hidden_states[-1] convention.
make diagnosis
```

`make {verify-prefill,verify-prefill-full}` are decode-independent and run on
either backend (`PREFILL_DTYPE=bf16` or `int4`); `make verify` runs the full
bf16-prefill + int4-decode path through the shared `verify/` gate, and
`make chat` is the interactive REPL over the same path.

## Key Files

| Path | Purpose |
|---|---|
| `llama32_1b_int4_prefill.py` | Driver: loads AWQ, runs either backend, prints top-K vs HF |
| `awq_pack.py` | AWQ qweight/qzeros/scales → int4 GEMM packed BO + bf16 dense |
| `gemm_builder.py` | int4 GEMM wrapper (per-model; bf16 sibling under `../llama32_1b/`) |
| `multi_launch_builder/rms_gemms_rope_int4_multi.py` | int4 RMSNorm + Q/K/V + RoPE Q/K (6-launch ELF) |
| `multi_launch_builder/o_ffn_int4_multi.py` | int4 O + ResAdd + FFN-RMS + Gate/Up + SwiGLU + Down + FFN-Add (8-launch ELF) |
| `Makefile` | Canonical `compile / run / verify / profile / clean` targets |
| `run_npu2_compile.lit` | Compile-only smoke test (no HF_TOKEN needed) |
| `run_npu2_verify.lit` | End-to-end top-K gate (HF_TOKEN required) |

Shared scaffolding lives one level up at `../llama_kernel_builder/`
(MLIR stitching, kernel cache, external `.o` compilation, SwiGLU + RoPE
C sources) — used by both this example and `../llama32_1b/`. Llama-side
helpers that aren't strictly kernel infra (the `LlamaWeights` /
`LlamaConfig` dataclasses, the bf16 CPU helpers, the bf16 prefill
stitchers) live in `../llama32_1b/` and are imported via `sys.path`
until a shared verify / loader package consolidates them. The int4 GEMM
module is imported from
`../matrix_multiplication/int4_awq/matmul_int4_packed.py`.

The shared verify subsystem under `../verify/` (top-K gate,
per-layer diagnosis, prompt fixtures, HF + NPU runners) is plugged
into via `verify_adapter.py` (see `make verify`).
