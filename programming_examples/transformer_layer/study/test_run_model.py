# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host tests for `run_model.py` (doc 56 H1a): the runner's whole flow with an
injected worker and verifier, no device.

EVERY CLAUSE IS PAIRED WITH THE INPUT THAT MAKES IT FIRE, resume's rule: a
verdict that fails on one prompt fails ONLY that rung; a worker that raises
leaves complete failed rows; `--no-verify` cannot produce a passed row; a
dry run produces complete skipped rows; a second session reuses the passed
rows and the ledger says so; two identical walks compare OK; and the prompt
builder hits an exact token count under a tokenizer that is not one-word-
one-token.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import compare_roots  # noqa: E402
import model_profiles as mp  # noqa: E402
import resume  # noqa: E402
import results_io  # noqa: E402
import run_model  # noqa: E402
import schema  # noqa: E402

_PLAN_HASH = "7" * 64


def _profile(compiled_Ms=((2048,), (2048,))):
    prof = mp.profile("model-smoke")
    compiled = {}
    for m, Ms in zip(prof.models, compiled_Ms):
        for M in Ms:
            compiled[(m, M)] = {"prefill_cache": f"/c/{m}/M{M}/prefill_kernel_cache", "decode_cache": f"/c/{m}/decode_kernel_cache"}
    return prof.bind(compiled, {("qwen3_0_6b", 512): "six registry rows pending (devq 567)"})


def _result(rung, tps, *, samples=3):
    n = samples if rung.phase == "prefill" else rung.n_tokens
    per = (rung.prompt_tokens if rung.phase == "prefill" else 1) / tps
    return {
        "weights_source": "org/model@deadbeef", "prepare_s": 12.0, "warmup": 1, "samples_s": [per] * n,
        "logical_tokens": rung.prompt_tokens if rung.phase == "prefill" else rung.n_tokens,
        "measured_tokens": (rung.prompt_tokens * n) if rung.phase == "prefill" else rung.n_tokens,
        "dispatch": {"scope": "prefill" if rung.phase == "prefill" else "decode_token", "host_submissions": 85, "runlist_entries": 85,
                     "air_launches": 626, "herd_launches": 1018, "sync_boundaries": 340, "bytes_transferred": 1234},
        "decomposition": {"device_ms": 10.0, "sync_ms": 1.0, "host_cpu_ms": 2.0, "host_ops": 59},
        "distinct_elfs": 4, "context_loads": 0, "kernel_attaches": 0,
    }


def _fake_worker(tps=100.0, *, raise_on=None, fail_on=(), nonce=0.0):
    def worker(model_id, M, rungs, compiled):
        if raise_on == (model_id, M):
            raise RuntimeError("worker exploded")
        rows, gate_map, prompts = [], {}, []
        for r in rungs:
            g = r.gate_prompt_tokens
            if g not in prompts:
                prompts.append(g)
            gate_map[r.case_id] = prompts.index(g)
            if r.case_id in fail_on:
                rows.append(run_model.failed_row(r, "s", "ERT_CMD_STATE_TIMEOUT", "org/model@deadbeef"))
            else:
                rows.append(run_model.measured_row(r, "s", _result(r, tps + nonce), _PLAN_HASH, {"curve": r.curve, "prompt_tokens": r.prompt_tokens}))
        return {"model_id": model_id, "M": M, "rows": rows, "gate": {"prompts_file": f"/p/{model_id}_M{M}.txt", "prompt_tokens": prompts, "map": gate_map}, "errors": []}
    return worker


def _fake_verifier(fail_prompt_idx=None, *, raise_=False):
    def verifier(model_id, M, gate, compiled):
        if raise_:
            raise RuntimeError("verify subprocess died")
        per = [{"prompt_idx": i, "status": "FAIL" if i == fail_prompt_idx else "OK", "fail_reason": "top-5 miss" if i == fail_prompt_idx else None, "divergence_step": None}
               for i in range(len(gate["prompt_tokens"]))]
        return {"passed": all(p["status"] == "OK" for p in per), "returncode": 0, "per_prompt": per, "report_json": "/r.json", "cache_root": "/c", "max_seq": M}
    return verifier


def _run(d, prof, **kw):
    kw.setdefault("worker_fn", _fake_worker())
    kw.setdefault("verifier", _fake_verifier())
    kw.setdefault("npu_power_mode", "turbo")
    return run_model.run(prof, d, study_id="s", **kw)


def _rows(d, name):
    return {r["study_case_id"]: r for r in results_io.read_rows(Path(d) / name)}


class _Tok:
    """Two tokens for words of 6+ characters, one otherwise, plus a BOS."""

    def encode(self, text):
        ids = [1]
        for w in text.split():
            ids += [hash(w) % 1000 + 2] * (2 if len(w) >= 6 else 1)
        return ids


def test_prompt_of_length_is_exact_under_an_uneven_tokenizer():
    tok = _Tok()
    src = run_model.prompt_source_text()
    for n in (17, 480, 992, 2016, 2048):
        text = run_model.prompt_of_length(tok.encode, n, src, forbidden_ids={0})
        assert len(tok.encode(text)) == n, n
    try:
        run_model.prompt_of_length(tok.encode, 5, src, forbidden_ids={1})  # BOS is forbidden -> refused
    except ValueError as e:
        assert "forbidden" in str(e)
    else:
        raise AssertionError("expected a refusal")


def test_a_full_walk_writes_complete_v3_rows_and_a_complete_manifest():
    prof = _profile()
    with tempfile.TemporaryDirectory() as d:
        report = _run(d, prof)
        q = _rows(d, "model_qwen3_0_6b.csv")
        l = _rows(d, "model_llama32_1b.csv")
        assert len(q) == 6 and len(l) == 4
        assert report["complete"] is True, report["incomplete_reasons"]
        assert report["rungs_by_status"] == {"passed": 8, "skipped": 2}
        for row in list(q.values()) + list(l.values()):
            schema.validate_row(row)
            assert row["measurement_scope"] == "model" and row["host_submissions_per_layer"] is None
        pre = q["qwen3_0_6b/prefill/M2048/ctx2048/bf16"]
        assert pre["run_status"] == "passed" and pre["plan_hash"] == _PLAN_HASH
        assert abs(float(pre["tokens_per_second"]) - 100.0) < 1e-6
        assert "kernel-scaling" in pre["study_case_label"]
        assert json.loads(pre["selected_config_json"])["verify"]["passed"] is True
        dec = q["qwen3_0_6b/decode/M2048/ctx512/bf16"]
        assert dec["context_start_tokens"] == "480" and dec["context_end_tokens"] == "512" and dec["seq_len"] == "1"
        assert json.loads(dec["model_dispatch_vector_json"])["scope"] == "decode_token"
        skipped = q["qwen3_0_6b/prefill/M512/ctx512/bf16"]
        assert skipped["run_status"] == "skipped" and "devq 567" in skipped["failure_message"]
        manifest = json.loads((Path(d) / run_model.MANIFEST_NAME).read_text())
        assert manifest["complete"] is True
        assert manifest[schema.CONDITIONS_KEY]["npu_power_mode"] == "turbo"
        assert manifest[schema.WALK_KEY]["rungs_measured"] == 8
        assert report["verify"]["qwen3_0_6b_M2048"]["passed"] is True


def test_a_gate_failure_on_one_prompt_fails_only_that_rung():
    """The M=2048 set's gate prompts are [2016, 480, 992] tokens (the prefill
    rung and the ctx-2048 decode rung share the 2016 one). Failing prompt 1
    (480 tokens) fails the decode rung at ctx 512 ONLY; the prefill rung and
    the other contexts stay passed, and the whole-set verdict is recorded."""
    prof = _profile()
    with tempfile.TemporaryDirectory() as d:
        report = _run(d, prof, verifier=_fake_verifier(fail_prompt_idx=1))
        q = _rows(d, "model_qwen3_0_6b.csv")
        assert q["qwen3_0_6b/decode/M2048/ctx512/bf16"]["run_status"] == "failed"
        assert "verify gate FAIL" in q["qwen3_0_6b/decode/M2048/ctx512/bf16"]["failure_message"]
        assert q["qwen3_0_6b/decode/M2048/ctx1024/bf16"]["run_status"] == "passed"
        assert q["qwen3_0_6b/decode/M2048/ctx2048/bf16"]["run_status"] == "passed"
        assert q["qwen3_0_6b/prefill/M2048/ctx2048/bf16"]["run_status"] == "passed"
        assert json.loads(q["qwen3_0_6b/prefill/M2048/ctx2048/bf16"]["selected_config_json"])["verify"]["prompt_idx"] == 0
        cfg = json.loads(q["qwen3_0_6b/prefill/M2048/ctx2048/bf16"]["selected_config_json"])
        assert cfg["verify"]["gate_passed_all_prompts"] is False and cfg["verify"]["passed"] is True
        assert report["complete"] is False


def test_no_verify_cannot_produce_a_passed_row():
    with tempfile.TemporaryDirectory() as d:
        report = _run(d, _profile(), verify=False)
        for row in _rows(d, "model_llama32_1b.csv").values():
            assert row["run_status"] == "failed" and "verify gate not run" in row["failure_message"]
        assert report["complete"] is False and report["verify"] == {}


def test_a_raising_worker_propagates_and_leaves_the_session_interrupted_honestly():
    """The production worker is a subprocess that ALWAYS writes rows
    (`run_worker` writes failed rows when it dies); an injected worker that
    raises is the crash path, and the ledger must show the session closed
    with what it had rather than claim rungs it never attributed."""
    prof = _profile()
    with tempfile.TemporaryDirectory() as d:
        try:
            _run(d, prof, worker_fn=_fake_worker(raise_on=("llama32_1b", 2048)))
        except RuntimeError as e:
            assert "worker exploded" in str(e)
        else:
            raise AssertionError("expected the worker's exception to propagate")
        ledger = resume.Ledger.load(d)
        assert ledger.sessions[-1]["status"] == "complete"  # closed in `finally`, with its rungs so far
        recorded = {tuple(r["model_key"]) for r in ledger.sessions[-1]["rungs"]}
        assert all(k[1] == "qwen3_0_6b" for k in recorded) and len(recorded) == 6
    with tempfile.TemporaryDirectory() as d:
        report = _run(d, prof, worker_fn=_fake_worker(fail_on=("llama32_1b/decode/M2048/ctx2048/bf16",)), verifier=_fake_verifier(raise_=True))
        l = _rows(d, "model_llama32_1b.csv")
        assert l["llama32_1b/decode/M2048/ctx2048/bf16"]["failure_message"] == "ERT_CMD_STATE_TIMEOUT"
        assert l["llama32_1b/decode/M2048/ctx512/bf16"]["run_status"] == "failed"
        assert "gate returncode None" in l["llama32_1b/decode/M2048/ctx512/bf16"]["failure_message"]
        assert report["verify"]["llama32_1b_M2048"]["error"].startswith("RuntimeError")
        assert len(l) == 4


def test_a_dry_run_touches_no_worker_and_writes_complete_skipped_rows():
    def boom(*a, **k):
        raise AssertionError("dry run reached a worker")

    with tempfile.TemporaryDirectory() as d:
        report = _run(d, _profile(), worker_fn=boom, verifier=boom, dry_run=True)
        rows = list(_rows(d, "model_qwen3_0_6b.csv").values()) + list(_rows(d, "model_llama32_1b.csv").values())
        assert len(rows) == 10 and all(r["run_status"] == "skipped" for r in rows)
        assert sum("dry run" in r["failure_message"] for r in rows) == 8
        assert report["complete"] is False and report["dry_run"] is True
        for row in rows:
            schema.validate_row(row)


def test_a_resumed_walk_reuses_passed_rows_and_walks_a_newly_compiled_set():
    prof = _profile()
    with tempfile.TemporaryDirectory() as d:
        _run(d, prof)
        before = _rows(d, "model_qwen3_0_6b.csv")

        def no_worker(*a, **k):
            raise AssertionError("a resumed walk with every rung passed must reach no worker")

        report = _run(d, prof, worker_fn=no_worker, verifier=no_worker, resume=True)
        assert report["rungs_by_source"] == {"measured": 0, "reused": 8, "skipped": 2}
        assert report["resume_defects"] == []
        after = _rows(d, "model_qwen3_0_6b.csv")
        for k in before:
            assert resume.row_digest(before[k]) == resume.row_digest(after[k]), k
        # the M=512 set appears: its skip is re-derived into a walk, the rest reused
        prof2 = _profile(((512, 2048), (2048,)))
        seen = []

        def worker(model_id, M, rungs, compiled):
            seen.append((model_id, M, [r.case_id for r in rungs]))
            return _fake_worker()(model_id, M, rungs, compiled)

        report = _run(d, prof2, worker_fn=worker, resume=True)
        assert seen == [("qwen3_0_6b", 512, ["qwen3_0_6b/prefill/M512/ctx512/bf16"])]
        assert report["rungs_by_source"] == {"measured": 1, "reused": 8, "skipped": 1}
        rows = _rows(d, "model_qwen3_0_6b.csv")
        assert rows["qwen3_0_6b/prefill/M512/ctx512/bf16"]["run_status"] == "passed"
        assert report["complete"] is True, report["incomplete_reasons"]
        manifest = json.loads((Path(d) / run_model.MANIFEST_NAME).read_text())
        assert manifest[schema.WALK_KEY]["walk_source"] == "resumed" and manifest[schema.WALK_KEY]["session_count"] == 3


def test_two_walks_compare_ok_and_a_throughput_drift_is_gated():
    prof = _profile()
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        _run(a, prof)
        _run(b, prof, worker_fn=_fake_worker(nonce=2.0))  # +2%: inside hybrid's 5% warn band
        rep = compare_roots.compare_roots(Path(a), Path(b), prof.expected_files())
        assert rep.failures == 0, rep.render()
        assert any("tokens_per_second" in line and "[GATE]" in line for line in rep.lines)
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        _run(a, prof)
        _run(b, prof, worker_fn=_fake_worker(nonce=-25.0))
        rep = compare_roots.compare_roots(Path(a), Path(b), prof.expected_files())
        assert rep.failures >= 1 and any("tokens_per_second" in line and "exceeds the fail" in line for line in rep.lines)


def test_gate_subcommand_regates_a_tree_without_a_device():
    prof = _profile()
    with tempfile.TemporaryDirectory() as d:
        _run(d, prof)
        built = run_model.gate(prof, d)
        assert built["complete"] is True
        assert built[schema.CONDITIONS_KEY]["npu_power_mode"] == "unknown"  # a re-gate never stamps a mode


def test_worker_rows_are_built_from_the_measurement_not_reconciled():
    """`measured_row` derives every rate from the samples it is handed."""
    r = [x for x in _profile().rungs() if x.case_id == "llama32_1b/decode/M2048/ctx1024/bf16"][0]
    row = run_model.measured_row(r, "s", _result(r, 16.0), _PLAN_HASH, {})
    assert row["latency_sample_count"] == 32 and abs(row["tokens_per_second"] - 16.0) < 1e-9
    assert abs(row["avg_latency_ms"] - 62.5) < 1e-9 and row["measured_token_count"] == 32
    assert row["effective_gflops_per_sec"] > 0 and row["attention_path"] == "host_numpy"
    assert json.loads(row["model_dispatch_vector_json"])["air_launches"] == 626
    schema.validate_row(row)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"run_model tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
