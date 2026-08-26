# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host tests for the H1b chunked (incremental) prefill (doc 56 sections 3.4 / 5).

What is pinned, device-free:

  1. THE MATH the device path implements, against an f32 reference: chunked
     causal attention (each chunk over the KV built so far, queries offset to
     the END of the context) composes to EXACTLY whole-prompt causal
     attention; the kernel's BLOCK mask (apply_causal_mask semantics on global
     block indices, the q_block counter based at (Lk-Lq)/64) equals the
     elementwise causal mask; RoPE position slices; the KV cache layout and
     the dv-chunk V pack from the cache vs from seq-first activations; a
     padded/garbage tail beyond the prompt CANNOT affect valid rows and is
     overwritten by the first decode append.
  2. THE REFUSALS: no padding path -- a prompt that is not a whole number of
     chunks is refused by the driver, the composed plan and the profile.
  3. THE PLAN models the chunked phase: per-chunk stages with the rectangular
     FA artifact named per context, totals the runner's live checks enforce
     (169 submissions / 171 host ops for 2 x 512 over 28 layers), the sha
     distinct from every square plan's, `plan_for` routing kv_len > M, and
     the planner's FA stage name agreeing with `fa_headfirst.fa_cache_name`.
  4. THE PROFILE: `ubatch-curve` is two UBATCH-labelled prefill rungs over the
     SAME 1024-token prompt with distinct resume keys, gate prompts of the
     full logical length, gate max_seq = prompt + 32, and the compiled-set
     skip rule.
  5. THE HOPS, by source: the worker picks the chunked policy off the curve
     label; the gate subprocess gets LLMS_VERIFY_UBATCH; the verify adapter
     branches on it into `run_npu_prefill_chunked`; the driver's chunk block
     passes lut_static=False (a static-skipped LUT would silently RoPE chunk
     2 at chunk 1's positions).
"""

from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.dirname(_HERE)
_PE = os.path.dirname(_EXAMPLE)
for _p in (_PE, os.path.join(_PE, "llms"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import model_profiles  # noqa: E402
from shared import model_adapter as ma  # noqa: E402
from shared.plan import NPU2_CAPS, QWEN3_0_6B, Workload, decoder_graph, plan, plan_ubatch_prefill  # noqa: E402


# ---------------------------------------------------------------------------
# 1. the math
# ---------------------------------------------------------------------------

def _sdpa(q, k, v, mask):
    """f32 SDPA over one head: mask[i, j] True = attend."""
    s = (q @ k.T) / np.sqrt(q.shape[1])
    s = np.where(mask, s, -np.inf)
    p = np.exp(s - s.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    return p @ v


def _causal(n):
    return np.tril(np.ones((n, n), dtype=bool))


def _rect_causal(lq, lk):
    """The rectangular mask: queries are the LAST lq rows of the lk context."""
    off = lk - lq
    return np.array([[j <= off + i for j in range(lk)] for i in range(lq)])


def test_chunked_attention_composes_to_whole():
    """Chunk c's attention over KV[0:end] with the rectangular mask == rows
    [start:end) of whole-prompt causal attention -- per head, GQA."""
    rng = np.random.default_rng(7)
    n, ub, hd, n_heads, n_kv = 8 * 16, 16, 8, 4, 2
    for h in range(n_heads):
        kv = h * n_kv // n_heads
        q = rng.standard_normal((n, hd))
        k = rng.standard_normal((n, hd))
        v = rng.standard_normal((n, hd))
        whole = _sdpa(q, k, v, _causal(n))
        for c in range(n // ub):
            s, e = c * ub, (c + 1) * ub
            got = _sdpa(q[s:e], k[:e], v[:e], _rect_causal(ub, e))
            assert np.allclose(got, whole[s:e], atol=1e-12), (h, c)


def test_block_mask_equals_elementwise_causal():
    """apply_causal_mask's BLOCK rule with the offset q_block base ==
    the elementwise mask, at the kernel's 64-row/col granularity."""
    B = 64
    for lq, lk in ((512, 1024), (512, 512), (256, 512), (1024, 1024)):
        off_blocks = (lk - lq) // B
        elementwise = _rect_causal(lq, lk)
        block = np.zeros((lq, lk), dtype=bool)
        for qb in range(lq // B):
            g_qb = qb + off_blocks  # the counter's global q block
            for kb in range(lk // B):
                tile = np.ones((B, B), dtype=bool)
                if kb > g_qb:
                    tile[:] = False          # above diagonal: full -inf fill
                elif kb == g_qb:
                    tile = _causal(B)        # diagonal block: col <= row
                # kb < g_qb: untouched (all attend)
                block[qb * B:(qb + 1) * B, kb * B:(kb + 1) * B] = tile
        assert np.array_equal(block, elementwise), (lq, lk)


def test_kv_cache_layout_and_v_pack_agree():
    """The cache append (seq-first -> [n_kv, seq, hd]) and the dv-chunk V pack
    FROM THE CACHE are byte-equal to fa_headfirst's seq-first pack."""
    rng = np.random.default_rng(11)
    from ml_dtypes import bfloat16

    n_kv, hd, lkp, seq = 2, 128, 64, 32
    dvc = hd // lkp
    v_seq = rng.standard_normal((seq, n_kv * hd)).astype(bfloat16)
    # the whole-path pack (npu_fa_headfirst)
    pack_seq = np.ascontiguousarray(
        v_seq.reshape(seq, n_kv, hd).transpose(1, 0, 2)
        .reshape(n_kv, seq, dvc, lkp).transpose(0, 2, 1, 3)
        .reshape(n_kv * dvc, seq, lkp)
    )
    # the chunked path: append to the cache, then pack from the cache
    cache = np.zeros((n_kv, seq + 8, hd), dtype=bfloat16)
    for s, e in ((0, seq // 2), (seq // 2, seq)):
        cache[:, s:e, :] = v_seq[s:e].reshape(e - s, n_kv, hd).transpose(1, 0, 2)
    pack_cache = np.ascontiguousarray(
        cache[:, :seq, :].reshape(n_kv, seq, dvc, lkp).transpose(0, 2, 1, 3)
        .reshape(n_kv * dvc, seq, lkp)
    )
    assert np.array_equal(pack_seq.view(np.uint16), pack_cache.view(np.uint16))


def test_padded_tail_cannot_affect_valid_rows():
    """IF a padded/garbage tail sat beyond the prompt in the last chunk's KV,
    causality alone keeps every VALID row's output byte-identical -- tail keys
    are LATER positions than every valid query. And the first decode append
    at pos = prompt_len overwrites the tail row, so decode state is clean."""
    rng = np.random.default_rng(13)
    n, hd, tail = 32, 8, 4
    q = rng.standard_normal((n, hd))
    k = rng.standard_normal((n + tail, hd))
    v = rng.standard_normal((n + tail, hd))
    # same shapes both sides (one reduction order); only the tail CONTENT
    # varies -- finite garbage, exactly what stale/padded KV rows would hold
    k_dirty, v_dirty = k.copy(), v.copy()
    k_dirty[n:] = 1e6 * rng.standard_normal((tail, hd))
    v_dirty[n:] = -1e6 * rng.standard_normal((tail, hd))
    # valid queries over the WIDER context, mask limited to k_abs <= q_abs:
    # every tail key is a LATER position than every valid query
    mask = np.array([[j <= i for j in range(n + tail)] for i in range(n)])
    clean = _sdpa(q, k, v, mask)
    dirty = _sdpa(q, k_dirty, v_dirty, mask)
    assert np.array_equal(clean, dirty)
    # and the masked-context result IS causal attention over the prompt alone
    assert np.allclose(clean, _sdpa(q, k[:n], v[:n], _causal(n)), atol=1e-12)
    # decode append overwrites the first tail row
    cache = np.zeros((n + tail, hd))
    cache[n:] = 7.0  # stale tail
    cache[n] = rng.standard_normal(hd)  # kv_append at current_pos = n
    assert not np.any(cache[n] == 7.0)


def test_rope_slice_is_absolute_positions():
    """The chunk LUT slice: rows [start:end) of the position table, repeated
    per head -- the DRIVER's arithmetic, mirrored. Chunk 2's first row is
    position 512's, never position 0's."""
    lut = np.arange(2048 * 4, dtype=np.float32).reshape(2048, 4)
    n_heads, start, ub = 3, 512, 512
    got = np.repeat(lut[start:start + ub], n_heads, axis=0)
    assert np.array_equal(got[0], lut[512]) and np.array_equal(got[n_heads], lut[513])
    assert got.shape == (ub * n_heads, 4)


# ---------------------------------------------------------------------------
# 2. the refusals
# ---------------------------------------------------------------------------

def test_driver_refuses_partial_and_padded():
    from types import SimpleNamespace

    sys.path.insert(0, os.path.join(_PE, "llms", "qwen3_0_6b"))
    from qwen3_0_6b_inference import run_npu_prefill_chunked

    tok = SimpleNamespace(eos_token_id=0)
    for bad_ids, why in (
        ([1] * 1000, "not a whole number of chunks"),
        ([1] * 511 + [0], "contains the EOS id"),
    ):
        try:
            run_npu_prefill_chunked(bad_ids, None, None, None, None, np.zeros((2048, 4)), 2048, tokenizer=tok, ubatch=512)
            raise AssertionError(f"no refusal for: {why}")
        except ValueError as e:
            assert why.split()[-2] in str(e) or why in str(e), (why, e)
    # rope LUT shorter than the prompt is a refusal, not an index error
    try:
        run_npu_prefill_chunked([1] * 1024, None, None, None, None, np.zeros((10, 4)), 2048, tokenizer=tok, ubatch=512)
        raise AssertionError("no refusal for a short rope LUT")
    except ValueError as e:
        assert "rope LUT" in str(e)


def test_plan_and_profile_refuse_partial():
    g = decoder_graph(QWEN3_0_6B)
    try:
        plan_ubatch_prefill(g, 1000, 512)
        raise AssertionError("plan_ubatch_prefill accepted a partial chunk")
    except ValueError as e:
        assert "whole number" in str(e)
    from dataclasses import replace

    prof = replace(model_profiles.profile("ubatch-curve"), ubatch_points=(("qwen3_0_6b", 1000, 512),))
    try:
        prof.rungs()
        raise AssertionError("the profile accepted a partial chunk")
    except ValueError as e:
        assert "whole number" in str(e)


def test_adapter_rejects_unknown_policy():
    a = ma.ModelAdapter("qwen3_0_6b")
    try:
        a.prefill([1, 2, 3], "sliding")
        raise AssertionError("unknown policy accepted")
    except ValueError as e:
        assert "whole" in str(e) and "chunked" in str(e)


# ---------------------------------------------------------------------------
# 3. the plan models the chunked phase
# ---------------------------------------------------------------------------

def test_composed_plan_totals_and_stages():
    g = decoder_graph(QWEN3_0_6B)
    p = plan_ubatch_prefill(g, 1024, 512, ctx=512)
    # 2 chunks x 28 layers x (QKV + FA + o_ffn) + LM head
    assert p.total_submissions == 2 * 28 * 3 + 1, p.total_submissions
    # 2 x (embed + 28 x (2 transposes + kv_append)) + final norm
    assert p.total_host_ops == 2 * (1 + 28 * 3) + 1, p.total_host_ops
    dev = [s.name for s in p.stages if s.where == "device"]
    assert dev == ["rms_qkv_qknorm_rope", "flash_attn", "o_ffn_qwen",
                   "rms_qkv_qknorm_rope", "flash_attn_ctx1024", "o_ffn_qwen", "lm_head_gemv"], dev
    hst = [s.name for s in p.stages if s.where == "host"]
    assert hst == ["embed_lookup", "transpose_seq_to_head", "kv_append", "transpose_head_to_seq"] * 2 + ["final_rms_norm"] and len(hst) == 9, hst
    assert p.workload["n_chunks"] == 2 and p.workload["ubatch_tokens"] == 512 and p.workload["logical_tokens"] == 1024
    # weights resident once, not per chunk
    assert p.resident_weight_bytes == plan(g, Workload("prefill", 512, 512, 512), NPU2_CAPS).resident_weight_bytes
    # sha: its own identity, 64-hex, stable, distinct from every square plan
    assert len(p.sha) == 64 and p.sha == plan_ubatch_prefill(g, 1024, 512, ctx=512).sha
    assert p.sha != plan(g, Workload("prefill", 512, 512, 512), NPU2_CAPS).sha
    assert p.sha != plan(g, Workload("prefill", 1024, 1024, 512), NPU2_CAPS).sha
    # one chunk IS the whole plan (same execution, same sha)
    assert plan_ubatch_prefill(g, 1024, 1024, ctx=1024).sha == plan(g, Workload("prefill", 1024, 1024, 1024), NPU2_CAPS).sha


def test_plan_for_routes_chunked():
    p = ma.plan_for("qwen3_0_6b", "prefill", 512, 1024, 512)
    assert p.workload.get("n_chunks") == 2 and p.total_host_ops == 171
    # kv_len == M stays the ordinary plan
    q = ma.plan_for("qwen3_0_6b", "prefill", 1024, 1024, 1024)
    assert "n_chunks" not in q.workload
    # decode is untouched by the routing
    d = ma.plan_for("qwen3_0_6b", "decode", 1, 1024, 2048)
    assert d.workload["phase"] == "decode"


def test_planner_fa_name_agrees_with_fa_headfirst():
    from shared.infra.fa_headfirst import fa_cache_name

    g = decoder_graph(QWEN3_0_6B)
    p = plan_ubatch_prefill(g, 1024, 512, ctx=512)
    fa = [s.name for s in p.stages if s.name.startswith("flash_attn")]
    assert fa == [fa_cache_name(512, 512), fa_cache_name(512, 1024)], fa
    assert fa_cache_name(512, 512) == "flash_attn" and fa_cache_name(512, 1024) == "flash_attn_ctx1024"


def test_predicted_vector_over_composed_plan():
    """model_dispatch_vector_from_manifest over the composed plan and a fake
    manifest: the live check's own arithmetic at the chunked shape."""
    p = ma.plan_for("qwen3_0_6b", "prefill", 512, 1024, 512)
    counts = {"rms_qkv_qknorm_rope": {"air_launches": 8, "herd_launches": 14},
              "o_ffn_qwen": {"air_launches": 8, "herd_launches": 16},
              "flash_attn": {"air_launches": 1, "herd_launches": 1},
              "flash_attn_ctx1024": {"air_launches": 1, "herd_launches": 1},
              "lm_head_gemv": {"air_launches": 10, "herd_launches": 10}}
    v = ma.model_dispatch_vector_from_manifest(p, counts, "prefill")
    assert v["host_submissions"] == 169
    assert v["air_launches"] == 2 * 28 * (8 + 1 + 8) + 10, v
    # a manifest WITHOUT the rectangular ELF is the exact mismatch to surface
    del counts["flash_attn_ctx1024"]
    try:
        ma.model_dispatch_vector_from_manifest(p, counts, "prefill")
        raise AssertionError("missing rectangular FA artifact went unnoticed")
    except KeyError as e:
        assert "flash_attn_ctx1024" in str(e)
    assert ma.plan_launches_match_manifest(p, counts)  # non-empty problem list


# ---------------------------------------------------------------------------
# 4. the profile
# ---------------------------------------------------------------------------

def test_ubatch_curve_profile():
    prof = model_profiles.profile("ubatch-curve")
    rungs = prof.rungs()
    assert [r.case_id for r in rungs] == ["qwen3_0_6b/prefill/M512/ctx1024/bf16", "qwen3_0_6b/prefill/M1024/ctx1024/bf16"]
    assert all(r.curve == model_profiles.UBATCH and r.phase == "prefill" for r in rungs)
    a, b = rungs
    assert (a.ubatch_tokens, a.context_end, a.prompt_tokens) == (512, 1024, 1024)
    assert (b.ubatch_tokens, b.context_end, b.prompt_tokens) == (1024, 1024, 1024)
    # the SAME prompt at both points; the gate prefills it whole + 32 slots
    assert a.gate_prompt_tokens == b.gate_prompt_tokens == 1024
    assert a.gate_max_seq == b.gate_max_seq == 1024 + model_profiles.GATE_N_TOKENS
    # distinct resume identities
    assert a.extra != b.extra and a.seq == 512 and b.seq == 1024
    # unbound: both skip with the compile hint; bound: measurable
    assert all(r.skip_reason and "compile" in r.skip_reason for r in rungs)
    bound = prof.bind({("qwen3_0_6b", 512): {"prefill_cache": "p", "decode_cache": "d"},
                       ("qwen3_0_6b", 1024): {"prefill_cache": "p2", "decode_cache": "d"}})
    assert all(r.skip_reason is None for r in bound.rungs())
    assert bound.artifact_sets() == [("qwen3_0_6b", 512), ("qwen3_0_6b", 1024)]
    counts = bound.expected_rows()["model_qwen3_0_6b.csv"]
    assert counts == {"rows": 2, "measured": 2, "skipped": 0}
    assert bound.summary()["ubatch_points"] == [["qwen3_0_6b", 1024, 512], ["qwen3_0_6b", 1024, 1024]]


def test_decode_rung_gate_max_seq_unchanged():
    """The gate_max_seq refactor must not move the decode rungs' gate capacity
    (M + 32, NOT ctx + 32) nor kernel-scaling prefill's (context_end == M)."""
    prof = model_profiles.profile("model-smoke")
    for r in prof.rungs():
        if r.phase == "decode":
            assert r.gate_max_seq == r.M + model_profiles.GATE_N_TOKENS, r.case_id
        else:
            assert r.gate_max_seq == r.M + model_profiles.GATE_N_TOKENS == r.context_end + model_profiles.GATE_N_TOKENS, r.case_id


# ---------------------------------------------------------------------------
# 5. the hops, by source
# ---------------------------------------------------------------------------

def _src(path):
    return open(path, encoding="utf-8").read()


def test_worker_policy_and_gate_env_by_source():
    s = _src(os.path.join(_HERE, "run_model.py"))
    assert '"chunked" if rung.curve == model_profiles.UBATCH else "whole"' in s
    assert '"LLMS_VERIFY_UBATCH": str(gate["ubatch"])' in s
    assert '"ubatch": M if ubatch_rungs else None' in s


def test_verify_adapter_branch_by_source():
    s = _src(os.path.join(_PE, "llms", "qwen3_0_6b", "verify_adapter.py"))
    assert 'os.environ.get("LLMS_VERIFY_UBATCH")' in s
    assert "run_npu_prefill_chunked(" in s
    # the chunked gate takes the EXACT prompt: no pad, no truncation
    assert "list(prompt_tokens)," in s


def test_chunk_block_lut_not_static_by_source():
    s = _src(os.path.join(_PE, "llms", "qwen3_0_6b", "qwen3_0_6b_prefill.py"))
    body = s.split("def run_transformer_block_qwen3_chunk", 1)[1]
    assert "lut_static=False" in body, "a static-skipped LUT ropes chunk 2 at chunk 1's positions"
    assert "rope_lut_bf16[pos_start:kv_end]" in body
    # kv_append precedes the FA call: the chunk attends to its own rows off the cache
    assert body.index('time_cpu("kv_append")') < body.index("npu_fa_headfirst_kv(")


# ---------------------------------------------------------------------------
# 6. `[2026-08-26]` item-16 review, blocking finding 3(a): the REAL driver
#    chunk scheduler, end-to-end, against the single-shot path on a fake NPU.
#
# The NumPy tests above pin the MATH; this one exercises the CODE --
# `run_npu_prefill_chunked` itself (per-layer cache append, LUT slice+refresh,
# fused-block arg wiring, chunk-boundary state, final norm + LM head) against
# `run_npu_prefill` on the SAME fake kernels, asserting identical prefill
# token, final logits and KV cache contents. The fake NPU boundary follows
# the fake-runtime pattern of shared/infra/test_dispatch.py / test_bo_pool.py:
# a KernelCache stand-in whose load_and_run computes each ELF's contract in
# float64->bf16 -- and which REPRODUCES the real cache's static-skip semantics
# (a static arg is stored on the bo_key's first call and later calls use the
# STORED bytes), so the chunked path's lut_static=False is load-bearing here
# exactly as on silicon. A teeth check forces lut_static=True and asserts the
# equality BREAKS (chunk 2 roped at chunk 1's positions).
# ---------------------------------------------------------------------------


class _FakeNpuCache:
    """KernelCache stand-in: real Profiler, fake kernels, REAL static-skip."""

    def __init__(self, profiler):
        self.profiler = profiler
        self._static_store: dict = {}

    # -- the kernels ---------------------------------------------------------

    @staticmethod
    def _rms(x, w, eps=1e-6):
        x = x.astype(np.float64)
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w.astype(np.float64)

    @staticmethod
    def _rope(x, lut_flat, n_heads, head_dim):
        """Halfsplit RoPE from the flattened per-head LUT (S*n_heads*head_dim)."""
        S = x.shape[0]
        half = head_dim // 2
        xh = x.astype(np.float64).reshape(S, n_heads, head_dim)
        lut = lut_flat.astype(np.float64).reshape(S, n_heads, head_dim)
        cos, sin = lut[..., :half], lut[..., half:]
        x1, x2 = xh[..., :half], xh[..., half:]
        out = np.empty_like(xh)
        out[..., :half] = x1 * cos - x2 * sin
        out[..., half:] = x2 * cos + x1 * sin
        return out.reshape(S, n_heads * head_dim)

    def _run_qkv(self, a, cfg):
        from ml_dtypes import bfloat16

        nh, nkv, hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        normed = np.asarray(self._rms(np.asarray(a[0], np.float64), a[1]), dtype=bfloat16)
        q = np.asarray(normed.astype(np.float64) @ a[3].astype(np.float64), dtype=bfloat16)
        k = np.asarray(normed.astype(np.float64) @ a[5].astype(np.float64), dtype=bfloat16)
        v = np.asarray(normed.astype(np.float64) @ a[7].astype(np.float64), dtype=bfloat16)
        S = q.shape[0]
        qn = np.asarray(self._rms(q.astype(np.float64).reshape(S, nh, hd), a[9]).reshape(S, nh * hd), dtype=bfloat16)
        kn = np.asarray(self._rms(k.astype(np.float64).reshape(S, nkv, hd), a[10]).reshape(S, nkv * hd), dtype=bfloat16)
        q_roped = np.asarray(self._rope(qn, a[13], nh, hd), dtype=bfloat16)
        k_roped = np.asarray(self._rope(kn, a[14], nkv, hd), dtype=bfloat16)
        return {8: v, 15: q_roped, 16: k_roped}

    def _run_o_ffn(self, a):
        from ml_dtypes import bfloat16

        attn, wo, x_resid = (np.asarray(a[0], np.float64), np.asarray(a[1], np.float64), np.asarray(a[3], np.float64))
        proj = np.asarray(attn @ wo, dtype=bfloat16)
        res1 = np.asarray(proj.astype(np.float64) + x_resid, dtype=bfloat16)
        normed2 = np.asarray(self._rms(res1.astype(np.float64), a[5]), dtype=bfloat16)
        gate = np.asarray(normed2.astype(np.float64) @ a[7].astype(np.float64), dtype=bfloat16)
        up = np.asarray(normed2.astype(np.float64) @ a[9].astype(np.float64), dtype=bfloat16)
        g = gate.astype(np.float64)
        sw = np.asarray(g / (1 + np.exp(-g)) * up.astype(np.float64), dtype=bfloat16)
        down = np.asarray(sw.astype(np.float64) @ a[12].astype(np.float64), dtype=bfloat16)
        out = np.asarray(down.astype(np.float64) + res1.astype(np.float64), dtype=bfloat16)
        return {14: out.ravel()}

    def _run_fa(self, a):
        from ml_dtypes import bfloat16

        q_hf, k_hf, v_hf = a[0], a[1], a[2]
        nh, lq, hd = q_hf.shape
        nkv, lk, _ = k_hf.shape
        lkp = 64
        dvc = hd // lkp
        off = lk - lq
        v_full = (v_hf.reshape(nkv, dvc, lk, lkp).transpose(0, 2, 1, 3).reshape(nkv, lk, hd)).astype(np.float64)
        out = np.zeros((nh, lq, hd))
        scale = 1.0 / np.sqrt(hd)
        for h in range(nh):
            kv = h * nkv // nh
            s = (q_hf[h].astype(np.float64) @ k_hf[kv].astype(np.float64).T) * scale
            for i in range(lq):
                s[i, off + i + 1:] = -np.inf
            p = np.exp(s - s.max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
            out[h] = p @ v_full[kv]
        packed = np.asarray(
            out.reshape(nh, lq, dvc, lkp).transpose(0, 2, 1, 3).reshape(nh * dvc, lq, lkp), dtype=bfloat16
        )
        return {3: packed}

    def _run_lm(self, a):
        from ml_dtypes import bfloat16

        x = np.asarray(a[0], np.float64)
        outs = {}
        for p in range((len(a) - 1) // 2):
            w = np.asarray(a[1 + 2 * p], np.float64)
            outs[2 + 2 * p] = np.asarray(w @ x, dtype=bfloat16)
        return outs

    # -- the boundary --------------------------------------------------------

    def load_and_run(self, name, backend_kwargs, *inputs, output_indices=None,
                     static_input_indices=None, intermediate_indices=None,
                     bo_key=None, shared_nonstatic=False):
        key = bo_key if bo_key is not None else name
        eff = list(inputs)
        for i in sorted(static_input_indices or []):
            sk = (key, i)
            if sk in self._static_store:
                eff[i] = self._static_store[sk]     # the real cache SKIPS the write
            else:
                self._static_store[sk] = np.array(inputs[i], copy=True)
        self._config_for_kernels = getattr(self, "_config_for_kernels", None)
        if name == "rms_qkv_qknorm_rope":
            outs = self._run_qkv(eff, self._config_for_kernels)
        elif name == "o_ffn_qwen":
            outs = self._run_o_ffn(eff)
        elif name.startswith("flash_attn"):
            outs = self._run_fa(eff)
        elif name == "lm_head_gemv":
            outs = self._run_lm(eff)
        else:
            raise KeyError(f"fake NPU has no kernel {name!r}")
        self.profiler.record_kernel(name, 0.0)
        self.profiler.record_breakdown(name, 0.0, 0.0, 0.0, len(inputs), 0, 1)
        return tuple(outs.get(i, np.asarray(x)) for i, x in enumerate(inputs))


def _fake_model(seed=99):
    """(config, weights, tokenizer, rope_lut, token_ids) at toy width, real head_dim."""
    from types import SimpleNamespace

    from ml_dtypes import bfloat16

    sys.path.insert(0, os.path.join(_PE, "llms", "qwen3_0_6b"))
    from qwen3_0_6b_decode import _LM_PARTS
    from qwen3_0_6b_weights import generate_rope_lut

    rng = np.random.default_rng(seed)
    cfg = SimpleNamespace(emb_dim=16, n_heads=2, n_kv_heads=1, head_dim=128, hidden_dim=8, n_layers=2)
    q_dim, kv_dim = cfg.n_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim
    layers = []
    for _ in range(cfg.n_layers):
        layers.append(SimpleNamespace(
            attn_norm=rng.uniform(0.5, 1.5, cfg.emb_dim).astype(bfloat16),
            wq=rng.uniform(-0.3, 0.3, (cfg.emb_dim, q_dim)).astype(bfloat16),
            wk=rng.uniform(-0.3, 0.3, (cfg.emb_dim, kv_dim)).astype(bfloat16),
            wv=rng.uniform(-0.3, 0.3, (cfg.emb_dim, kv_dim)).astype(bfloat16),
            q_norm=rng.uniform(0.5, 1.5, cfg.head_dim).astype(bfloat16),
            k_norm=rng.uniform(0.5, 1.5, cfg.head_dim).astype(bfloat16),
            wo=rng.uniform(-0.3, 0.3, (q_dim, cfg.emb_dim)).astype(bfloat16),
            ffn_norm=rng.uniform(0.5, 1.5, cfg.emb_dim).astype(bfloat16),
            w_gate=rng.uniform(-0.3, 0.3, (cfg.emb_dim, cfg.hidden_dim)).astype(bfloat16),
            w_up=rng.uniform(-0.3, 0.3, (cfg.emb_dim, cfg.hidden_dim)).astype(bfloat16),
            w_down=rng.uniform(-0.3, 0.3, (cfg.hidden_dim, cfg.emb_dim)).astype(bfloat16),
        ))
    vocab = sum(_LM_PARTS)
    weights = SimpleNamespace(
        layers=layers,
        embed_table=rng.uniform(-1, 1, (32, cfg.emb_dim)).astype(bfloat16),
        final_norm=rng.uniform(0.5, 1.5, cfg.emb_dim).astype(np.float32),
        lm_head=np.zeros((vocab, 1), dtype=bfloat16),  # shape[0] is the vocab
        _lm_weight_parts_gemv=[rng.uniform(-0.3, 0.3, (rows, cfg.emb_dim)).astype(bfloat16) for rows in _LM_PARTS],
    )
    tok = SimpleNamespace(eos_token_id=0)
    lut_cfg = SimpleNamespace(head_dim=cfg.head_dim, rope_base=1_000_000.0)
    rope_lut = generate_rope_lut(config=lut_cfg, seq_len=32).astype(bfloat16)
    token_ids = [1 + (i % 30) for i in range(16)]  # no EOS id
    return cfg, weights, tok, rope_lut, token_ids


def _run_paths(force_static_lut=False):
    """(whole, chunked) results on fresh fake caches: (token, logits, k, v)."""
    from shared.infra.cache import Profiler

    sys.path.insert(0, os.path.join(_PE, "llms", "qwen3_0_6b"))
    import qwen3_0_6b_prefill as qp
    from qwen3_0_6b_inference import run_npu_prefill, run_npu_prefill_chunked

    cfg, weights, tok, rope_lut, ids = _fake_model()
    saved = (qp._FUSED_SCRATCH_FOR, qp._OFFN_SCRATCH_FOR)
    orig_fused = qp._fused_qknorm_rope_call
    qp._FUSED_SCRATCH_FOR = (None, None, None)
    qp._OFFN_SCRATCH_FOR = (None, None, None, None)
    if force_static_lut:  # the teeth check: reintroduce the LUT static-skip bug
        qp._fused_qknorm_rope_call = lambda *a, **kw: orig_fused(*a, **{**kw, "lut_static": True})
    try:
        def caches():
            pc = _FakeNpuCache(Profiler(enabled=True))
            pc._config_for_kernels = cfg
            dc = _FakeNpuCache(Profiler(enabled=True))
            return pc, dc

        pc, dc = caches()
        whole = run_npu_prefill(list(ids), weights, cfg, pc, dc, rope_lut, 24,
                                tokenizer=tok, cpu_attn=False, quiet=True)
        pc2, dc2 = caches()
        chunk = run_npu_prefill_chunked(list(ids), weights, cfg, pc2, dc2, rope_lut, 24,
                                        tokenizer=tok, ubatch=8, quiet=True)
    finally:
        qp._FUSED_SCRATCH_FOR, qp._OFFN_SCRATCH_FOR = saved
        qp._fused_qknorm_rope_call = orig_fused
    return whole, chunk, len(ids)


def test_chunked_driver_equals_single_shot_on_fake_npu():
    (tok_w, logits_w, k_w, v_w, len_w), (tok_c, logits_c, k_c, v_c, len_c), n = _run_paths()
    assert len_w == len_c == n
    assert tok_w == tok_c, (tok_w, tok_c)
    assert np.array_equal(np.asarray(logits_w), np.asarray(logits_c))
    # identical KV cache contents over the prompt rows, every layer, byte-equal
    assert np.array_equal(k_w[:, :, :n, :].view(np.uint16), k_c[:, :, :n, :].view(np.uint16))
    assert np.array_equal(v_w[:, :, :n, :].view(np.uint16), v_c[:, :, :n, :].view(np.uint16))
    # and beyond the prompt both caches are untouched zeros
    assert not np.any(k_c[:, :, n:, :].view(np.uint16)) and not np.any(v_c[:, :, n:, :].view(np.uint16))


def test_fake_npu_static_skip_has_teeth():
    """Force lut_static=True through the chunked path: the fake's static-skip
    must reproduce the silent-wrong-RoPE failure (chunk 2 roped at chunk 1's
    positions), i.e. the equality above must BREAK -- proving the test can
    catch the bug class it exists for."""
    (tok_w, logits_w, k_w, v_w, _), (tok_c, logits_c, k_c, v_c, _), n = _run_paths(force_static_lut=True)
    same_k = np.array_equal(k_w[:, :, :n, :].view(np.uint16), k_c[:, :, :n, :].view(np.uint16))
    same_logits = np.array_equal(np.asarray(logits_w), np.asarray(logits_c))
    assert not (same_k and same_logits), (
        "forcing the LUT static-skip did NOT change the chunked result: the fake "
        "NPU does not model the static-store and the equality test has no teeth")
    # chunk 1 (positions 0..7) is unaffected -- the divergence is exactly the
    # later chunk's positions
    assert np.array_equal(k_w[:, :, :8, :].view(np.uint16), k_c[:, :, :8, :].view(np.uint16))
    assert not np.array_equal(k_w[:, :, 8:n, :].view(np.uint16), k_c[:, :, 8:n, :].view(np.uint16))


def _main():
    import traceback

    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_") and callable(o)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS {name}")
        except Exception:
            failed.append(name)
            print(f"  FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
