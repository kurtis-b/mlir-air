# SmolLM2-1.7B int4 (GGUF q4_0) on NPU2

SmolLM2-1.7B deployed from the `bartowski/SmolLM2-1.7B-Instruct-GGUF` **Q4_0**
checkpoint — the second quantized example after `llama32_1b_int4`, whose
config-driven machinery it reuses.

**Shipped shape** (mirrors `llama32_1b_int4`): bf16 NPU prefill on the
q4_0-**dequantized** weights + int4 NPU decode on the packed BOs at **gs = 32**
(q4_0's block size; the AWQ example runs gs = 128). The prefill dequant is taken
from the q4_0 payloads — never the original bf16 — so the whole model IS the
quantized model and `make verify` exercises it.

```
make compile     # one-time kernel compile (bf16 prefill @ SmolLM2 shapes + gs=32 int4 decode)
make run         # NPU prefill + int4 NPU decode
make verify      # top-k token-set inclusion gate vs the PLAIN HF bf16 model
make chat        # interactive REPL
GGUF=/path/to/other.gguf make run   # else $SMOLLM2_GGUF, else the hub file via the HF cache
```

## What is model-specific here (everything else is inherited)

- `smollm2_1_7b_int4_weights.py` — the GGUF q4_0 loader. Dual layout (dense
  bf16 for prefill, packed decode BOs), q/k RoPE row un-permute (llama.cpp
  convention, applied only to checkpoint-provenance payloads;
  `test_int4_weights.py` pins it against the HF weights), the three promoted
  Q4_1 `ffn_down` tensors re-quantized from the bf16 HF reference,
  embeddings/norms/tied lm_head from the bf16 HF checkpoint (the GGUF's Q6_K
  embedding is never consumed).
- `smollm2_1_7b_int4_decode.py` — kernel compilation at SmolLM2 shapes
  (emb 2048, full MHA kv_dim 2048, hidden 8192, 24 layers) and **gs = 32**,
  threaded through `int4_gs` in the backend kwargs so the per-compile kernel
  sweep stages the right `mv_int4_bf16.o` variant before every aiecc link.
- `smollm2_1_7b_int4_inference.py` / `verify_adapter.py` — thin wrappers over
  `llama32_1b_int4`'s config-driven machinery. The verify reference is the
  PLAIN `HuggingFaceTB/SmolLM2-1.7B-Instruct` bf16 model, so the gate's delta
  deliberately includes q4_0 quantization error — the quantization is what is
  shipped.

## Gates

- `make verify` (the `run_npu2_verify.lit` recipe): top-k token-set inclusion
  vs the plain HF bf16 model, 2 prompts x 32 tokens, k=5, `--model instruct`.
  Binary PASS/FAIL; the artifact is the device log cited by the PR that landed
  this gate. `make diagnosis` is the single-prompt lens beside it.
- Host, no device: `python3 test_int4_weights.py` (`run_int4_weights_host.lit`).
- The two shared-driver seams this deployment needed (`int4_gs` staging,
  `prompt_len`) are regression-checked by `make verify` in `llama32_1b_int4`
  and `smollm2_1_7b`.

The lit recipes (`run_npu2_compile.lit`, `run_npu2_verify.lit`) are collected by
path into the nightly llms compile/verify suites, not the PR gate; see
[ARCHITECTURE.md](ARCHITECTURE.md) for the execution shape.
