# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""O2 prototype: the decode token as `run_sequence` pairs, behind a flag.

Doc 57 section 3 (O2) and section 5 item 4: today's decode token makes 57
`xrt.run` submissions -- per layer `rms_qkv_qknorm_rope_gemv` then, after the
CPU attention, `o_gemv_ffn`; then `lm_head_gemv`. The CPU attention sits
between the two kernels OF a layer, but nothing sits between layer L's
`o_gemv_ffn` and layer L+1's `rms_qkv_qknorm_rope_gemv`: the first writes the
hidden state `x`, the second reads it. Under the ELF ABI `KernelCache.run_sequence`
puts runs from two artifacts into ONE runlist (shared/infra/dispatch.py), so
each such pair becomes one submission and `x` never crosses to the host between
them -- it is read back once, after the pair, because the NEXT layer's residual
add and the host need it anyway.

Submissions per token: 1 (`rms_qkv` L0) + 27 pairs + 1 (`o_gemv_ffn` L27)
+ 1 (`lm_head`) = 30, against 57.  The launch-boundary count (327) is
UNCHANGED -- this prototype moves only the submission count, which is the
smaller of the two structural costs doc 57 measured, so it is measured, not
predicted (the dispatch vector of every pair is available for that).

Enable with the environment variable ``QWEN3_DECODE_RUNLIST=1`` (or
``--decode-runlist`` on the CLI, which sets it) -- every path that reaches
``run_npu_decode_step`` (``make run/profile/verify``) then takes this one.

Prototype costs, deliberately not hidden:
  * The pairs' pools hold a SECOND resident copy of layers 1..27's QKV weights
    and layers 0..26's FFN weights (the `load_and_run` per-layer BOs the
    standard preload creates stay allocated): ~2x decode weight memory. The
    fix is a preload that knows the flag; not done for the prototype.
  * Each pair has its own pool (the plan signature carries the layer's static
    content keys), so the 6 MB dead `z_hidden_emb` slot of `o_gemv_ffn`'s ABI
    is allocated 27 times. It is declared as written so it is never uploaded.
"""

import os
import time

import numpy as np
from ml_dtypes import bfloat16

from shared.infra.bo_pool import BufferSpec, DispatchStep, content_key
from qwen3_0_6b_decode import (
    _RMS_QKV_KERNEL,
    _o_gemv_ffn_backend,
    _rms_qkv_qknorm_rope_gemv_backend,
    _run_o_gemv_ffn,
    decode_attention_cpu,
    prep_rms_qkv4_weights,
    rms_qkv4_lut,
    run_rms_qkv4,
)

_FLAG = "QWEN3_DECODE_RUNLIST"


def enabled():
    return os.environ.get(_FLAG, "") not in ("", "0")


def _static_keys(lw, li, config):
    """Content keys for one layer's static buffers, computed once and cached."""
    keys = getattr(lw, "_runlist_ckeys", None)
    if keys is None:
        prep_rms_qkv4_weights(lw, config)
        keys = {
            "wo": content_key(lw._wo_t),
            "packed": content_key(lw._packed_rms_buf),
            "wgateup": content_key(lw._wgateup_t),
            "wdown": content_key(lw._wdown_t),
            "norm": content_key(np.ascontiguousarray(lw.attn_norm).astype(bfloat16)),
            "wqkv": content_key(lw._wqkv_t),
            "qknw": content_key(lw._qk_norm_w),
        }
        lw._runlist_ckeys = keys
    return keys


_ZEROS = {}


def _zeros(emb, hid):
    """Dead/intermediate placeholder arrays, allocated once: they supply only
    dtype and size to the pool (declared written, never uploaded), and the
    6 MB `z_hid_emb` zero-fill per pair per token would otherwise cost more
    than the submission it saves."""
    key = (emb, hid)
    if key not in _ZEROS:
        _ZEROS[key] = (np.zeros(emb, dtype=bfloat16), np.zeros(hid, dtype=bfloat16),
                       np.zeros((hid, emb), dtype=bfloat16))
    return _ZEROS[key]


def _pair_sequence(lw_o, li_o, lw_r, li_r, config, attn_out, x_res, lut):
    """Steps/specs/arrays for (o_gemv_ffn of layer li_o, rms_qkv4 of layer li_r)."""
    emb, hid = config.emb_dim, config.hidden_dim
    q_dim = config.n_heads * config.head_dim
    kv_dim = config.n_kv_heads * config.head_dim
    qk_dim = q_dim + kv_dim
    ko, kr = _static_keys(lw_o, li_o, config), _static_keys(lw_r, li_r, config)
    P = f"L{li_o}_"          # prefix: one pool per pair
    specs, arrays = {}, {}

    def buf(name, arr, *, static=False, host_output=False, ckey=None):
        specs[name] = BufferSpec(name, arr.size * arr.itemsize, static=static,
                                 host_output=host_output, content_key=ckey)
        arrays[name] = arr
        return name

    z_emb, z_hid, z_hid_emb = _zeros(emb, hid)

    # --- o_gemv_ffn (15-arg ABI, dead args kept; see qwen3_0_6b_decode) ---
    o_args = (
        buf(P + "wo", lw_o._wo_t, static=True, ckey=ko["wo"]),            # 0
        buf(P + "attn_out", attn_out),                                      # 1 host
        buf(P + "dead2", z_emb),                                            # 2 dead
        buf(P + "x_res", x_res),                                            # 3 host
        buf(P + "dead4", z_emb),                                            # 4 dead
        buf(P + "dead5", z_emb),                                            # 5 dead
        buf(P + "packed", lw_o._packed_rms_buf, static=True, ckey=ko["packed"]),  # 6
        buf(P + "wgateup", lw_o._wgateup_t, static=True, ckey=ko["wgateup"]),     # 7
        buf(P + "dead8", z_hid),                                            # 8 dead
        buf(P + "dead9", z_hid_emb),                                        # 9 dead (6 MB)
        buf(P + "dead10", z_hid),                                           # 10 dead
        buf(P + "swiglu", z_hid),                                           # 11 intermediate
        buf(P + "wdown", lw_o._wdown_t, static=True, ckey=ko["wdown"]),     # 12
        buf(P + "dead13", z_emb),                                           # 13 dead
        buf(P + "x_out", z_emb, host_output=True),                          # 14 -> next layer's x
    )
    # Dead args are declared written so they count as produced (never uploaded).
    o_step = DispatchStep("o_gemv_ffn", o_args, writes=(2, 4, 5, 8, 9, 10, 11, 13, 14))

    # --- rms_qkv_qknorm_rope_gemv4 (9-arg ABI) reading x_out in place of x_in ---
    r_args = (
        P + "x_out",                                                        # 0 x_in (device-resident)
        buf(P + "norm_w", np.ascontiguousarray(lw_r.attn_norm).reshape(emb).astype(bfloat16),
            static=True, ckey=kr["norm"]),                                  # 1
        buf(P + "normed", z_emb),                                           # 2
        buf(P + "wqkv", lw_r._wqkv_t, static=True, ckey=kr["wqkv"]),        # 3
        buf(P + "qkv", np.zeros(qk_dim + kv_dim, dtype=bfloat16), host_output=True),  # 4 q|k|v
        buf(P + "qk_norm_w", lw_r._qk_norm_w, static=True, ckey=kr["qknw"]),  # 5
        buf(P + "qk_n", np.zeros(qk_dim, dtype=bfloat16)),                  # 6
        buf(P + "lut", lut),                                                # 7 host (position)
        buf(P + "qk_roped", np.zeros(qk_dim, dtype=bfloat16), host_output=True),  # 8
    )
    r_step = DispatchStep(_RMS_QKV_KERNEL, r_args, writes=(2, 4, 6, 8))
    backend_kwargs = {
        "o_gemv_ffn": _o_gemv_ffn_backend(),
        _RMS_QKV_KERNEL: _rms_qkv_qknorm_rope_gemv_backend(),
    }
    return [o_step, r_step], specs, arrays, backend_kwargs, P


_SEQ_CACHE = {}


def _cached_pair(cache, config, lw_o, li_o, lw_r, li_r, attn_out, x_res, lut):
    """Steps/specs are layer-constant; only three host arrays change per token."""
    key = (id(cache), li_o)
    ent = _SEQ_CACHE.get(key)
    if ent is None:
        ent = _pair_sequence(lw_o, li_o, lw_r, li_r, config, attn_out, x_res, lut)
        _SEQ_CACHE[key] = ent
    steps, specs, arrays, bk, P = ent
    arrays[P + "attn_out"] = attn_out
    arrays[P + "x_res"] = x_res
    arrays[P + "lut"] = lut
    return ent


def preload_pairs(cache, config, weights):
    """Dispatch every pair once with zero inputs so the pools exist and the
    static weights are resident before the first measured token (the standard
    preload does the same for its per-layer BOs)."""
    emb = config.emb_dim
    q_dim = config.n_heads * config.head_dim
    kv_dim = config.n_kv_heads * config.head_dim
    z_emb, z_q = np.zeros(emb, dtype=bfloat16), np.zeros(q_dim, dtype=bfloat16)
    z_lut = np.zeros(q_dim + kv_dim, dtype=bfloat16)
    _was = cache.profiler.enabled
    cache.profiler.enabled = False
    for li in range(config.n_layers - 1):
        _run_pair(cache, config, weights.layers[li], li, weights.layers[li + 1], li + 1,
                  z_q, z_emb, z_lut)
    cache.profiler.enabled = _was
    print(f"  [runlist] pre-loaded {config.n_layers - 1} pair pools (O2 prototype).")


def _run_pair(cache, config, lw_o, li_o, lw_r, li_r, attn_out, x_res, lut):
    steps, specs, arrays, bk, P = _cached_pair(cache, config, lw_o, li_o, lw_r, li_r,
                                               attn_out, x_res, lut)
    t0 = time.perf_counter()
    results, vector = cache.run_sequence(steps, specs, bk, arrays,
                                         require_single_submission=True)
    cache.profiler.record_kernel("pair_o_gemv_ffn+rms_qkv", time.perf_counter() - t0)
    # Copy: the views are pool memory, overwritten by the next sequence.
    q_dim = config.n_heads * config.head_dim
    kv_dim = config.n_kv_heads * config.head_dim
    qkv = np.array(results[P + "qkv"], copy=True)
    qk_roped = np.array(results[P + "qk_roped"], copy=True)
    out = {
        "x_out": np.array(results[P + "x_out"], copy=True),
        "v": np.ascontiguousarray(qkv[q_dim + kv_dim:]),
        "q_roped": np.ascontiguousarray(qk_roped[:q_dim]),
        "k_roped": np.ascontiguousarray(qk_roped[q_dim:]),
    }
    return out, vector


def run_npu_decode_step_runlist(x_decode_bf16, weights, config, decode_cache,
                                rope_lut_bf16, k_cache, v_cache, current_pos,
                                run_lm_head, rms_norm, final_norm, eps):
    """One decode step with (o_gemv_ffn_L, rms_qkv_{L+1}) pairs as single submissions.

    Mirrors `run_npu_decode_step` + `run_decode_block` exactly in arithmetic;
    only the submission structure differs. `run_lm_head`, `rms_norm`,
    `final_norm`, `eps` are passed in so this module owns no second copy of
    the LM-head path.
    """
    n_layers = config.n_layers
    n_heads, n_kv_heads, head_dim = config.n_heads, config.n_kv_heads, config.head_dim
    vocab_size = weights.lm_head.shape[0]
    if not getattr(run_npu_decode_step_runlist, "_announced", False):
        # One line per process so a verify/profile log proves which path ran.
        print(f"[runlist] O2 decode path ACTIVE ({_FLAG}=1): "
              f"{n_layers - 1} (o_gemv_ffn_L, rms_qkv_L+1) pairs per token", flush=True)
        run_npu_decode_step_runlist._announced = True
    x = x_decode_bf16.copy().flatten().astype(bfloat16)
    vectors = []

    # Layer 0's QKV stage: the standard single-kernel path (nothing precedes it).
    lut = rms_qkv4_lut(rope_lut_bf16, current_pos, config)
    t0 = decode_cache.profiler.start_layer()
    v, q_roped, k_roped = run_rms_qkv4(decode_cache, weights.layers[0], x, lut, config, 0)

    for li in range(n_layers):
        lw = weights.layers[li]
        k_cache[li][:, current_pos, :] = k_roped.reshape(n_kv_heads, head_dim)
        v_cache[li][:, current_pos, :] = v.reshape(n_kv_heads, head_dim)
        with decode_cache.profiler.time_cpu("decode_attention_cpu"):
            attn_out = decode_attention_cpu(q_roped, k_cache[li], v_cache[li],
                                            current_pos, n_heads, n_kv_heads, head_dim)
        if li + 1 < n_layers:
            out, vec = _run_pair(decode_cache, config, lw, li, weights.layers[li + 1], li + 1,
                                 np.ascontiguousarray(attn_out).astype(bfloat16), x, lut)
            vectors.append(vec)
            x = out["x_out"]
            v, q_roped, k_roped = out["v"], out["q_roped"], out["k_roped"]
        else:
            x = _run_o_gemv_ffn(np.ascontiguousarray(attn_out).astype(bfloat16), x, lw,
                                config, decode_cache, li)
            x = np.asarray(x).flatten().astype(bfloat16)
        decode_cache.profiler.end_layer(li, t0)
        t0 = decode_cache.profiler.start_layer()

    with decode_cache.profiler.time_cpu("final_rms_norm"):
        x_normed = rms_norm(x.astype(np.float32).reshape(1, config.emb_dim), final_norm, eps=eps)
    logits = run_lm_head(decode_cache, weights, x_normed.flatten().astype(bfloat16), vocab_size)
    run_npu_decode_step_runlist.last_vectors = vectors
    return int(np.argmax(logits)), logits


def describe_last_step():
    """Dispatch-vector summary of the last step's pairs, for `make profile`."""
    vecs = getattr(run_npu_decode_step_runlist, "last_vectors", None)
    if not vecs:
        return "[runlist] no pair vectors recorded"
    subs = sum(v.host_submissions for v in vecs)
    ents = sum(v.runlist_entries for v in vecs)
    air = sum(v.air_launches for v in vecs)
    sync = sum(v.sync_boundaries for v in vecs)
    by = sum(v.bytes_transferred for v in vecs)
    return (f"[runlist] pairs {len(vecs)}: submissions {subs} entries {ents} air {air} "
            f"sync {sync} bytes {by}  (+3 single submissions: rms_qkv L0, o_gemv_ffn L27, lm_head)")
