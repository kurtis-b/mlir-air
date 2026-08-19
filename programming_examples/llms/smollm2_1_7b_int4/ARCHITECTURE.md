# SmolLM2-1.7B int4 (GGUF q4_0) — Architecture

Companion to [README.md](README.md). Mirrors the two exemplars it inherits
from — `../smollm2_1_7b/` (the bf16 SmolLM2, for shapes and MHA deltas) and
`../llama32_1b_int4/` (the AWQ int4 Llama, for the quantized execution shape)
— deltas called out below.

## Model Config

24 layers, emb_dim=2048, n_heads=32, head_dim=64, **n_kv_heads=32 (pure MHA)**,
hidden_dim=8192, vocab_size=49152, rope_theta=130000, tied embeddings.
Checkpoint: `bartowski/SmolLM2-1.7B-Instruct-GGUF` **Q4_0** (165 of 168 linears
Q4_0; `blk.{0,1,10}.ffn_down` Q4_1; embedding Q6_K, never consumed).

## Quantization: q4_0 on the shipped asymmetric kernel

q4_0 dequantizes as `w = d*(q - 8)` — blocks of 32, fp16 scale `d` (bf16 in
the packed BOs), nibbles `q in [0,15]`, fixed zero-point 8. That is the
shipped int4 kernel's `(q - z)*s` with `z = 8`, so the device kernel is
unchanged; only `DIM_GS` moves (32 here, 128 in the AWQ example), which is a
compile-time define threaded through `int4_gs` in the backend kwargs — see
`smollm2_1_7b_int4_decode.py` for why the threading is load-bearing (the
per-compile kernel sweep restages the canonical `mv_int4_bf16.o` before every
aiecc link, and the first build without it linked the gs=128 default).

## Deltas vs llama32_1b_int4 (the int4 exemplar)

| Axis | here | llama32_1b_int4 | Impact |
|---|---|---|---|
| checkpoint format | GGUF q4_0 | AutoAWQ safetensors | loader only (`gguf_q4_0.py` primitives) |
| group size | **32** | 128 | `DIM_GS` recompile + `pack_inputs` gs |
| zero-point plane | all-8s (implicit) | per-group asym | same kernel, constant Z |
| q/k row order | llama.cpp RoPE-permuted | HF order | `llama_unpermute_rows` on checkpoint payloads only |
| n_kv_heads | 32 (MHA) | 8 (GQA) | `kv_dim = emb_dim`; `group_size = 1` in CPU decode attention |
| n_layers | 24 | 16 | config-only; 1.5x decode BO count |
| verify reference | **plain HF bf16** | AWQ-dequant-patched HF | the gate INCLUDES quantization error here |

## Execution shape (per layer)

```
Prefill (bf16 NPU, on q4_0-dequantized weights — dequant FROM the payloads,
         bf16 scales, so prefill computes the packed BOs' numbers):
  rms_gemms_rope.elf → flash_attn.elf (32q/32kv) → o_ffn.elf
  (the bf16 smollm2 kernel set, registry-driven, MHA-safe)

Decode (int4 NPU, gs=32, + CPU KV-cache attention):
  rms_qkv_int4_rope.elf (6 launches: RMS + int4 Q/K/V GEMV + RoPE Q/K)
    → CPU attention (MHA, group_size=1)
    → o_gemv_ffn_int4.elf (3 stages: int4 O+add, int4 gate/up swiglu+RMS,
                           int4 down+add; gate/up nibble-interleaved in one BO)
Final: lm_head_gemv.elf (bf16, tied embedding, 49152x2048 over 8 partitions)
```

## Weight flow

`smollm2_1_7b_int4_weights.load_weights_gguf_q4_0` opens BOTH checkpoints:
the bf16 HF model supplies embeddings/norms/tied lm_head and the quantization
references for the three promoted Q4_1 tensors (re-quantized to q4_0, route
(d) of `gguf_q4_0.py`'s decision record; transcoding was measured worse and
refused); the GGUF supplies the seven linears per layer, each landing twice —
dense bf16 `[in, out]` for prefill and `(A_q, A_s, A_z)` packed decode BOs.
The RoPE un-permute applies only to checkpoint-provenance q/k payloads (a
promoted payload is already in HF row order).
