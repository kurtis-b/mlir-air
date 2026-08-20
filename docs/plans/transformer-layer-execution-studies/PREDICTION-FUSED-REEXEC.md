# PREDICTION-FUSED-REEXEC — written before any dispatch, 2026-08-19

## The located mechanism

The fused decoder's re-execution wall (dispatch 1 = 12/12 clean, dispatch
2..n = UNMASKED attention with clean q/k/v, devq 382-384) is the causal
mha's Q-block counter. `attn_npu2.py` keeps per-core causal state in an
UNINITIALIZED L1 buffer (`causal_ctr`: [0]=q_block base, [1]=boot flag,
[2]=head_local, [3]=dv_iter). The boot flag fires only on zeroed memory;
head_local and dv_iter wrap; **q_block only ever advances** (+NQ per
head-group wrap, never modulo). A complete execution ends with
q_base = num_lq_iters * NQ, past every kv block. L1 persists across host
dispatches, so dispatch 2 loads boot=1 (no re-init) and q_base too large:
`kv_block_idx > q_block_idx` is never true, apply_causal_mask never fills,
attention is exactly unmasked while everything else is right.

This also explains the two controls that framed the wall:
- coarse re-executes causally because its flow re-initializes the
  partition per dispatch (L1 zeroed, boot flag fires every time);
- evicting/reloading the fused hw_context did NOT heal it (devq 384)
  because the reload rewrites only CDO-initialized state, and causal_ctr
  is an uninitialized AllocOp — "device-side, outside the context".

## The fix under test

`_emit_counter_increment` now wraps the q advance:
`q_wrapped = remsi(q_cur + NQ, num_lq_iters * NQ)`. Within one execution
no in-flight value changes (the sum only reaches the bound at the very
end), so dispatch-1 behavior is unchanged by construction; the counter
returns to its boot state at the end of every complete execution.

## Predictions (fused decoder gpt2_small 512x768, 3 dispatches of one
## prepared stitch, capturing all 12 boundaries per dispatch)

1. Compile gate: the causal mha module's MLIR gains exactly the remsi
   wrap (constant 8 at this shape: num_lq_iters 2 x NQ 4); the ENCODER
   mha module's MLIR is byte-identical (fingerprint cache hit).
2. Dispatch 1: 12/12 boundaries clean under DECODER_STAGE_ATOL — same as
   devq 382's baseline dispatch 1.
3. Dispatches 2 and 3: attn_context corr vs the CAUSAL reference ~1
   (baseline: 0.9994 vs the UNMASKED reference), and all 12 boundaries
   clean — the wall is gone.
4. Dispatch 2 == dispatch 3 bytes (the design is re-execution-stable).

Falsifiers: d2/d3 still unmasked → the counter is not the (only) state,
back to BD forensics. d1 regressing → the wrap changed an in-flight value
and the "unchanged by construction" claim is wrong — revert immediately.
d2 clean but d3 not → the wrap bound is off by one execution's worth.
