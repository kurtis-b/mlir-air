# SmolLM2-1.7B int4 (GGUF q4_0) on NPU2

The second quantized model (transformer-layer study goal 2, step 5): SmolLM2-1.7B
deployed from the `bartowski/SmolLM2-1.7B-Instruct-GGUF` **Q4_0** checkpoint.

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
GGUF=/path/to/other.gguf make run   # another Q4_0 checkpoint
```

## What is model-specific here (everything else is inherited)

- `smollm2_1_7b_int4_weights.py` — the GGUF q4_0 loader. Dual layout (dense
  bf16 for prefill, packed decode BOs), q/k RoPE row un-permute (llama.cpp
  convention; cosine 0.03 → 0.996, applied only to checkpoint-provenance
  payloads), the three promoted Q4_1 `ffn_down` tensors re-quantized from the
  bf16 HF reference, embeddings/norms/tied lm_head from the bf16 HF checkpoint
  (the GGUF's Q6_K embedding is never consumed).
- `smollm2_1_7b_int4_decode.py` — kernel compilation at SmolLM2 shapes
  (emb 2048, full MHA kv_dim 2048, hidden 8192, 24 layers) and **gs = 32**,
  threaded through `int4_gs` in the backend kwargs so the per-compile kernel
  sweep stages the right `mv_int4_bf16.o` variant before every aiecc link.
- `smollm2_1_7b_int4_inference.py` / `verify_adapter.py` — thin wrappers over
  `llama32_1b_int4`'s config-driven machinery. The verify reference is the
  PLAIN `HuggingFaceTB/SmolLM2-1.7B-Instruct` bf16 model, so the gate's delta
  deliberately includes q4_0 quantization error — the quantization is what is
  shipped.

## Verified

- `make verify` PASS (top-k token-set inclusion vs HF bf16; devq 378).
- Step-level decode-block probe: k/v/block-output at corr ≥ 0.9999 against the
  host chain computed from the same q4_0 payloads (devq 377).
- E2e: first token equals HF's top-1; ~11 decode tok/s at 24 layers.
- Shared-driver regression after the two shared fixes this deployment
  surfaced (ChatML prompt-length, gs canonical staging): `llama32_1b`,
  `llama32_1b_int4`, `smollm2_1_7b` all PASS (devq 379).

Not yet ported from the siblings: the `run_npu2_*.lit` CI recipes and an
`ARCHITECTURE.md`.
