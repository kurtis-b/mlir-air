# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The MODEL runner (doc 56 section 3.1, H1a): walk a `model_profiles.ModelProfile`
through `llms/shared/model_adapter.ModelAdapter` under the study's discipline and
write schema v3 model rows.

    python3 study/run_model.py run --profile model-smoke --out-dir R --study-id ID
    python3 study/run_model.py run ... --resume          # carry passed rungs forward
    python3 study/run_model.py run ... --dry-run         # no device: complete skipped rows
    python3 study/run_model.py gate --profile model-smoke --out-dir R   # re-gate a tree
    python3 study/run_model.py compile --model qwen3_0_6b --M 512 --compiled-root C
    python3 study/run_model.py plan --profile model-smoke [--compiled-root C]

CONTRACT
    The layer study (`run_profile.py` / `run_ladder.py` / `run_mode.py`) is
    untouched; this is its sibling and SHARES `schema`, `manifest`, `resume`,
    `compare_roots`, `power`, `results_io`, `smoke_gate` and `run_profile`'s
    Turbo refusal and device preflight. Nothing here re-implements a gate.

    THE DISCIPLINE, in order: refuse off Turbo (`run_profile._require_turbo`,
    exit 2 before a root exists); refuse a held device (`device_preflight`);
    scan the root and open the ledger (resume); per ARTIFACT SET (one model at
    one compiled prefill M) run ONE worker subprocess that prepares the adapter
    once and measures every rung on that set -- the drivers' `prepare_runtime`
    is one-shot per process, which is why the worker is a process and not a
    call; then run the production verify gate over that set's own prompts
    (outside the clock, a subprocess) and stamp every row of the set with its
    verdict: a row is `passed` ONLY if the gate passed on its prompt under the
    same artifact set (doc 56 section 3.7); write the CSVs; audit the ledger;
    gate the root; write the manifest and the run report.

    THE CLOCK (operator rule 2026-08-22) is the adapter's: forward pass only.
    The worker times nothing else and the parent times nothing at all.

    EVERY PLAN HASH IS WRITTEN: each row carries `Plan.sha` for its own
    workload (`model_adapter.plan_for`), the artifact cache key doc 56 section
    3.3 names, and the verify verdict is recorded beside it with the prompt it
    ran. A failure of any kind is a COMPLETE row with the message.

FOOTGUNS
    - A prefill rung whose artifact set is not compiled is a derived SKIP
      (complete row, `run_status=skipped`, the reason in `failure_message`);
      the profile counts it as skipped and the manifest is complete without
      it. Compile it (`compile`, a build-class devq job, its own cwd), then
      `run --resume` -- the skip is re-derived and becomes a walk.
    - `--no-verify` leaves every measured row `failed` with the message
      "verify gate not run": a row that nobody gated is not a passed row. Use
      it for a smoke of the mechanism, never for a number.
    - The worker's prompts are built from the verify prompt set's text to an
      EXACT token count under the gate's own tokenization (`encode`, no chat
      template), and the gate runs those same prompts, so the measurement and
      the verdict are one workload. Prompts are written under the root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.dirname(_HERE)
_PE = os.path.dirname(_EXAMPLE)
for _p in (_PE, os.path.join(_PE, "llms"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import manifest  # noqa: E402
import model_profiles  # noqa: E402
import power  # noqa: E402
import resume as resume_mod  # noqa: E402
import results_io  # noqa: E402
import run_profile  # noqa: E402
import schema  # noqa: E402
import smoke_gate  # noqa: E402
from shared import model_adapter  # noqa: E402

MANIFEST_NAME = run_profile.MANIFEST_NAME
RUN_REPORT_NAME = "model_run.json"
WORKLOAD_VARIANT = "decoder_llm"


# ---------------------------------------------------------------------------
# Prompts: an exact token count under the gate's tokenization.
# ---------------------------------------------------------------------------


def prompt_source_text(verify_prompts_dir: str | Path | None = None) -> str:
    """The verify prompt set's lines, joined, as the repeated text."""
    d = Path(verify_prompts_dir) if verify_prompts_dir else Path(_PE) / "llms" / "verify" / "prompts"
    lines = []
    for name in ("instruct.txt", "base.txt"):
        p = d / name
        if p.is_file():
            lines += [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    if not lines:
        lines = ["The quick brown fox jumps over the lazy dog while the river runs to the sea."]
    return " ".join(lines)


_FILLERS = ("and", "the", "of", "to", "a", "in", "it", "is")


def prompt_of_length(encode, n_tokens: int, source: str, *, forbidden_ids=()) -> str:
    """Text whose `encode` is EXACTLY `n_tokens` ids, built from `source`
    repeated; raises if no exact fit is found. `forbidden_ids` (the EOS the
    drivers count prompt length by) must not appear."""
    words = source.split()
    if not words:
        raise ValueError("empty prompt source")
    # enough words: grow geometrically, then binary-search the word count
    count = max(8, n_tokens // 2)
    while len(encode(" ".join((words * (count // len(words) + 1))[:count]))) < n_tokens:
        count *= 2
    lo, hi = 1, count
    pool = words * (hi // len(words) + 1)
    while lo < hi:
        mid = (lo + hi) // 2
        if len(encode(" ".join(pool[:mid]))) >= n_tokens:
            hi = mid
        else:
            lo = mid + 1
    base = pool[: max(lo - 1, 0)]
    # base encodes to < n_tokens; pad with short fillers to the exact count
    for _ in range(64):
        text = " ".join(base)
        ids = encode(text) if base else []
        k = len(ids)
        if k == n_tokens:
            if any(i in forbidden_ids for i in ids):
                raise ValueError("prompt contains a forbidden (EOS) id")
            return text
        if k > n_tokens:
            base = base[:-1]
            continue
        base = base + [_FILLERS[len(base) % len(_FILLERS)]]
    raise ValueError(f"could not build a prompt of exactly {n_tokens} tokens")


# ---------------------------------------------------------------------------
# Rows.
# ---------------------------------------------------------------------------


def _base_row(rung: model_profiles.ModelRung, study_id: str, weights_source: str | None) -> dict:
    spec = model_adapter.MODELS[rung.model_id].spec
    row = schema.empty_row("results")
    row.update(
        {
            "study_id": study_id,
            "study_case_id": rung.case_id,
            "study_case_label": rung.label,
            "workload_variant": WORKLOAD_VARIANT,
            "backend": "xrt",
            "execution_mode": schema.EXECUTION_MODE_CSV[rung.mode],
            "attention_path": "device" if rung.phase == "prefill" else "host_numpy",
            "seq_len": rung.seq,
            "hidden_size": spec.emb_dim,
            "intermediate_size": spec.hidden_dim,
            "num_attention_heads": spec.n_heads,
            "attention_head_size": spec.head_dim,
            "batch_size": 1,
            "dtype": "bf16",
            "use_bias": False,
            "weights_source": weights_source,
            "process_model": "subprocess",
            "measurement_scope": "model",
            "model_id": rung.model_id,
            "phase": rung.phase,
            "ubatch_tokens": rung.ubatch_tokens,
            "context_start_tokens": rung.context_start,
            "context_end_tokens": rung.context_end,
            "precision_plan_id": rung.precision_plan,
        }
    )
    return row


def skipped_row(rung: model_profiles.ModelRung, study_id: str, reason: str) -> dict:
    row = _base_row(rung, study_id, None)
    row["run_status"] = "skipped"
    row["failure_message"] = reason
    row["logical_token_count"] = rung.prompt_tokens if rung.phase == "prefill" else rung.n_tokens
    schema.validate_row(row)
    return row


def failed_row(rung: model_profiles.ModelRung, study_id: str, message: str, weights_source=None) -> dict:
    row = _base_row(rung, study_id, weights_source)
    row["run_status"] = "failed"
    row["failure_message"] = message.splitlines()[0][:300] if message else "failed"
    row["logical_token_count"] = rung.prompt_tokens if rung.phase == "prefill" else rung.n_tokens
    schema.validate_row(row)
    return row


def weights_only_gflops(model_id: str, tokens: int, seconds: float) -> float | None:
    """2 x (matmul weight elements) x tokens / s. Attention FLOPs excluded; the
    row's selected_config_json says so."""
    spec = model_adapter.MODELS[model_id].spec
    per_layer = spec.emb_dim * (spec.q_dim + 2 * spec.kv_dim) + spec.q_dim * spec.emb_dim + 3 * spec.emb_dim * spec.hidden_dim
    total = per_layer * spec.n_layers + spec.emb_dim * spec.vocab_size
    return (2.0 * total * tokens / seconds) / 1e9 if seconds > 0 else None


def measured_row(rung, study_id: str, result: dict, plan_hash: str, extras: dict) -> dict:
    """A complete row from a worker's measurement dict (see `worker`)."""
    row = _base_row(rung, study_id, result["weights_source"])
    samples = result["samples_s"]
    n = len(samples)
    total = float(sum(samples))
    row.update(
        {
            "warmup_runs": result["warmup"],
            "runs_per_sample": 1,
            "measured_inference_count": n,
            "latency_sample_count": n,
            "timed_total_sec": total,
            "avg_latency_ms": 1000.0 * total / n if n else None,
            "min_latency_ms": 1000.0 * min(samples) if n else None,
            "max_latency_ms": 1000.0 * max(samples) if n else None,
            "compile_setup_time_ms": 1000.0 * float(result["prepare_s"]),
            "effective_gflops_per_sec": weights_only_gflops(rung.model_id, result["measured_tokens"], total),
            "validation_error_count": None,
            "run_status": "passed",
            "failure_message": "",
            "npu_dispatch_count": result["dispatch"]["host_submissions"] * (n if rung.phase == "prefill" else 1),
            "npu_unique_instruction_binary_count": result["distinct_elfs"],
            "npu_unique_xclbin_count": 0,
            "selected_config_json": json.dumps(extras, sort_keys=True),
            "device_ms": result["decomposition"]["device_ms"],
            "sync_ms": result["decomposition"]["sync_ms"],
            "host_cpu_ms": result["decomposition"]["host_cpu_ms"],
            "context_loads": result["context_loads"],
            "kernel_attaches": result["kernel_attaches"],
            "logical_token_count": result["logical_tokens"],
            "measured_token_count": result["measured_tokens"],
            "tokens_per_second": result["measured_tokens"] / total if total > 0 else None,
            "plan_hash": plan_hash,
            "host_ops": result["decomposition"]["host_ops"],
            "model_dispatch_vector_json": json.dumps({k: result["dispatch"][k] for k in schema.MODEL_DISPATCH_VECTOR_KEYS}),
        }
    )
    schema.validate_row(row)
    return row


def stamp_verdict(row: dict, verdict: dict | None, prompt_idx: int | None) -> dict:
    """Apply the gate's verdict to a measured row IN PLACE: `passed` only if the
    gate passed on this row's prompt; otherwise `failed` with the reason. The
    verdict is recorded inside selected_config_json either way."""
    if row["run_status"] != "passed":
        return row
    cfg = json.loads(row["selected_config_json"] or "{}")
    if verdict is None:
        row["run_status"] = "failed"
        row["failure_message"] = "verify gate not run (--no-verify): an ungated row is not a passed row"
        cfg["verify"] = {"ran": False}
    else:
        mine = [p for p in verdict["per_prompt"] if p["prompt_idx"] == prompt_idx]
        # the row's OWN prompt decides it (doc 56 section 3.7: the gate at the
        # row's plan); the whole-set verdict is recorded beside it.
        ok = verdict.get("returncode") is not None and len(mine) == 1 and mine[0]["status"] == "OK"
        cfg["verify"] = {
            "ran": True, "passed": ok, "gate_passed_all_prompts": verdict["passed"], "prompt_idx": prompt_idx,
            "per_prompt": mine, "report_json": verdict.get("report_json"), "returncode": verdict.get("returncode"),
            "cache_root": verdict.get("cache_root"), "max_seq": verdict.get("max_seq"),
        }
        if not ok:
            row["run_status"] = "failed"
            why = (mine[0].get("fail_reason") if mine else None) or f"gate returncode {verdict.get('returncode')}"
            row["failure_message"] = f"verify gate FAIL on this row's prompt: {why}"[:300]
    row["selected_config_json"] = json.dumps(cfg, sort_keys=True)
    schema.validate_row(row)
    return row


# ---------------------------------------------------------------------------
# The worker: one artifact set, one process.
# ---------------------------------------------------------------------------


def worker(args) -> int:
    """Prepare the adapter on ONE artifact set and measure its rungs; write
    `--out` JSON: {"rows": [...], "gate": {"prompts_file", "map"}, ...}."""
    spec = json.loads(Path(args.rungs).read_text(encoding="utf-8"))
    rungs = [model_profiles.ModelRung(**r) for r in spec["rungs"]]
    model_id, M = spec["model_id"], int(spec["M"])
    prompts_dir = Path(args.prompts_dir)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    out = {"model_id": model_id, "M": M, "rows": [], "gate": None, "errors": []}
    reserve = max([r.n_tokens for r in rungs] + [model_profiles.GATE_N_TOKENS]) + spec["decode_warmup"]

    adapter = model_adapter.ModelAdapter(model_id)
    try:
        ms = adapter.prepare(model_id, spec["precision_plan"], spec["compiled"], n_tokens=reserve)
    except Exception as exc:  # every rung of the set fails as a complete row
        msg = f"{type(exc).__name__}: {exc}"
        out["rows"] = [failed_row(r, spec["study_id"], msg) for r in rungs]
        out["errors"].append(msg)
        Path(args.out).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
        return 0
    tok = ms.session.tokenizer
    encode = lambda text: list(tok.encode(text))  # noqa: E731
    source = prompt_source_text()
    forbidden = {tok.eos_token_id}
    caches = (ms.session.prefill_cache, ms.session.decode_cache)
    artifact = model_adapter.artifact_key(model_id, M, spec["precision_plan"], [ms.prefill_cache_dir, ms.decode_cache_dir])
    note_path = ms.prefill_cache_dir / model_adapter.COMPILE_NOTE
    deviation = json.loads(note_path.read_text(encoding="utf-8")).get("artifact_deviation") if note_path.is_file() else None

    prompt_cache: dict[int, str] = {}

    def prompt(n):
        if n not in prompt_cache:
            text = prompt_of_length(encode, n, source, forbidden_ids=forbidden)
            assert len(encode(text)) == n
            (prompts_dir / f"prompt_{n}.txt").write_text(text + "\n", encoding="utf-8")
            prompt_cache[n] = text
        return prompt_cache[n]

    def reconf():
        return tuple(sum(c.reconfiguration_counts()[i] for c in caches) for i in (0, 1))

    gate_prompts: list[int] = []
    gate_map: dict[str, int] = {}
    for rung in rungs:
        g = rung.gate_prompt_tokens
        if g not in gate_prompts:
            gate_prompts.append(g)
        gate_map[rung.case_id] = gate_prompts.index(g)

    for rung in rungs:
        try:
            text = prompt(rung.prompt_tokens)
            ids = encode(text)
            plan_ = model_adapter.plan_for(model_id, rung.phase, rung.seq, rung.context_end, M, rung.precision_plan)
            c0 = reconf()
            if rung.phase == "prefill":
                res = adapter.prefill(ids, "whole", None, samples=spec["prefill_samples"], warmup=spec["prefill_warmup"])
                warm = spec["prefill_warmup"]
            else:
                pre = adapter.prefill(ids, "whole", None, samples=1, warmup=0)
                res = adapter.decode(pre.state, rung.n_tokens, warmup=spec["decode_warmup"])
                warm = spec["decode_warmup"]
            c1 = reconf()
            result = {
                "weights_source": ms.weights_source, "prepare_s": ms.prepare_s, "warmup": warm,
                "samples_s": res.samples_s, "logical_tokens": res.logical_tokens, "measured_tokens": res.measured_tokens,
                "dispatch": res.dispatch, "decomposition": res.decomposition,
                "distinct_elfs": res.decomposition.get("distinct_elfs"), "context_loads": c1[0] - c0[0], "kernel_attaches": c1[1] - c0[1],
            }
            extras = {
                "curve": rung.curve, "prompt_tokens": rung.prompt_tokens, "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "gate_prompt_tokens": rung.gate_prompt_tokens, "artifact_key": artifact,
                "prefill_cache": str(ms.prefill_cache_dir), "decode_cache": str(ms.decode_cache_dir),
                "plan_workload": plan_.workload, "plan_est_us": plan_.est_us, "plan_total_launches": plan_.total_launches,
                "plan_total_submissions": plan_.total_submissions, "plan_total_host_ops": plan_.total_host_ops,
                "samples_s": res.samples_s, "tokens": res.tokens[:64], "context": [res.context_start, res.context_end],
                "effective_gflops_note": "weights-only FLOPs (2 x matmul weight elements x tokens); attention excluded",
                "host_ops_note": "named Profiler.time_cpu buckets only; untimed host glue (FA transposes, Python) is in avg_latency_ms and not here",
                "phase_dispatch_vector": adapter.dispatch_vector("decode") if rung.phase == "decode" else res.dispatch,
                "qkv_scratch_layout": adapter._scratch_layout,
                "artifact_deviation": deviation,
            }
            row = measured_row(rung, spec["study_id"], result, plan_.sha, extras)
            if deviation:
                row["study_case_label"] += f" [ARTIFACT DEVIATES FROM PLAN: o_ffn GEMMs forced {deviation['o_ffn_gemm_method']}]"
            out["rows"].append(row)
            print(f"[worker] {rung.case_id}: {res.tokens_per_second:.2f} tok/s over {len(res.samples_s)} sample(s); dispatch {res.dispatch}", flush=True)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            msg = f"{type(exc).__name__}: {exc}"
            out["rows"].append(failed_row(rung, spec["study_id"], msg, ms.weights_source))
            out["errors"].append(f"{rung.case_id}: {msg}")
    gate_file = prompts_dir / "gate_prompts.txt"
    gate_file.write_text("".join(prompt(n) + "\n" for n in gate_prompts), encoding="utf-8")
    out["gate"] = {"prompts_file": str(gate_file), "prompt_tokens": gate_prompts, "map": gate_map}
    Path(args.out).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return 0


def run_worker(model_id: str, M: int, rungs: list, *, compiled: dict, study_id: str, profile, out_dir: Path) -> dict:
    """Spawn `worker` for one artifact set and return its JSON."""
    work = out_dir / "work" / f"{model_id}_M{M}"
    work.mkdir(parents=True, exist_ok=True)
    spec = {
        "model_id": model_id, "M": M, "study_id": study_id, "compiled": {"prefill_M": M, **compiled},
        "precision_plan": profile.precision_plan, "prefill_samples": profile.prefill_samples,
        "prefill_warmup": profile.prefill_warmup, "decode_warmup": profile.decode_warmup,
        "rungs": [
            {"model_id": r.model_id, "phase": r.phase, "M": r.M, "context_end": r.context_end, "n_tokens": r.n_tokens,
             "precision_plan": r.precision_plan, "curve": r.curve, "skip_reason": None}
            for r in rungs
        ],
    }
    (work / "rungs.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    out_json = work / "worker_out.json"
    if out_json.exists():
        out_json.unlink()
    cmd = [sys.executable, os.path.abspath(__file__), "worker", "--rungs", str(work / "rungs.json"), "--out", str(out_json), "--prompts-dir", str(out_dir / "prompts" / f"{model_id}_M{M}")]
    log = work / "worker.log"
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=str(work), stdout=fh, stderr=subprocess.STDOUT, text=True)
    if not out_json.exists():
        tail = log.read_text(encoding="utf-8", errors="replace")[-2000:]
        msg = f"worker exited {proc.returncode} without writing its result; log tail: {tail}"
        return {"model_id": model_id, "M": M, "rows": [failed_row(r, study_id, msg) for r in rungs], "gate": None, "errors": [msg]}
    return json.loads(out_json.read_text(encoding="utf-8"))


def run_verify(model_id: str, M: int, gate: dict, *, compiled: dict, out_dir: Path) -> dict:
    """The production gate over the set's prompts, in the set's cache root."""
    adapter = model_adapter.ModelAdapter(model_id)
    cache_root = Path(compiled["prefill_cache"]).parent
    report_dir = out_dir / "verify" / f"{model_id}_M{M}"
    cwd = cache_root if M == model_profiles.SHIPPED_PREFILL_M else Path(compiled["prefill_cache"]).parent / "verify_cwd"
    extra_env = {}
    note = Path(compiled["prefill_cache"]) / model_adapter.COMPILE_NOTE
    if note.is_file():
        deviation = json.loads(note.read_text(encoding="utf-8")).get("artifact_deviation") or {}
        if deviation.get("o_ffn_gemm_method"):
            extra_env["LLMS_VERIFY_O_FFN_GEMM_METHOD"] = deviation["o_ffn_gemm_method"]
    verdict = adapter.verify_against_hf(gate["prompts_file"], report_dir, cache_root=cache_root, max_seq=M, cwd=cwd, extra_env=extra_env)
    (report_dir / "verify.log").write_text(verdict.pop("log") or "", encoding="utf-8")
    (report_dir / "verdict.json").write_text(json.dumps(verdict, indent=1, default=str), encoding="utf-8")
    return verdict


# ---------------------------------------------------------------------------
# The walk, the gate, the report.
# ---------------------------------------------------------------------------


def gate(profile, out_dir: str | Path, repo=None, conditions=None, toolchain=None, walk=None) -> dict:
    """Smoke gate + manifest over an existing tree; no device. `run_profile.gate`'s
    shape without the four-mode distinguishability clause (a model walk has
    one execution mode and nothing to distinguish)."""
    out_dir = Path(out_dir)
    expected = profile.expected_files()
    expected_rows = profile.expected_rows()
    problems = smoke_gate.check_results_root(out_dir, expected, expected_rows=expected_rows)
    for line in problems:
        print(f"[smoke-gate] {line}")
    print(f"[smoke-gate] {'FAIL' if problems else 'PASS'} ({len(expected)} CSV(s))")
    built = manifest.build_manifest(out_dir, expected, repo=repo, expected_rows=expected_rows, conditions=conditions, toolchain=toolchain, walk=walk)
    built["distinguish"] = {"gated_lengths": 0, "lines": ["not applicable: a model walk has one execution mode"]}
    manifest.write_manifest(out_dir / MANIFEST_NAME, built)
    print(f"[manifest] wrote {out_dir / MANIFEST_NAME}")
    print(f"[manifest] complete: {built['complete']}")
    for reason in built["incomplete_reasons"]:
        print(f"[manifest]   {reason}")
    return built


def run(profile, out_dir, *, study_id: str, worker_fn=None, verifier=None, repo=None, resume: bool = False,
        npu_power_mode: str | None = None, verify: bool = True, dry_run: bool = False, power_backend: str = "auto") -> dict:
    """Walk, gate, write. `worker_fn(model_id, M, rungs, compiled=...)` and
    `verifier(model_id, M, gate, compiled=...)` are injectable for the host
    tests; production uses `run_worker` / `run_verify`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    worker_fn = worker_fn or (lambda model_id, M, rungs, compiled: run_worker(model_id, M, rungs, compiled=compiled, study_id=study_id, profile=profile, out_dir=out_dir))
    verifier = verifier or (lambda model_id, M, g, compiled: run_verify(model_id, M, g, compiled=compiled, out_dir=out_dir))
    repo_root = Path(repo) if repo else Path(_PE).parent
    expected_files = profile.expected_files()

    prior = resume_mod.scan(out_dir, expected_files)
    ledger = resume_mod.Ledger.load(out_dir)
    conditions = manifest.observe_conditions(npu_power_mode)
    toolchain = manifest.observe_toolchain()
    plan = resume_mod.plan(profile, prior, enabled=resume)
    for line in resume_mod.describe(plan):
        print(f"[run-model] {line}")

    started = datetime.now(timezone.utc)
    ledger.open_session(profile=profile.name, started_utc=started.isoformat(), devq_job_id=os.environ.get("DEVQ_JOB_ID"),
                        git_sha=resume_mod.git_sha(repo_root), npu_power_mode=conditions["npu_power_mode"],
                        toolchain_fingerprint=resume_mod.toolchain_fingerprint(toolchain))
    rungs = profile.rungs()
    by_key = {resume_mod.rung_key(r.mode, r.seq, r.extra): r for r in rungs}
    rows_by_key: dict[tuple, dict] = {}
    verdicts: dict[str, dict] = {}
    t0 = time.perf_counter()
    try:
        with power.open_monitor(power_backend) as monitor:
            # skips and reuses first: neither reaches a worker
            for key, rung in by_key.items():
                code_key = (rung.mode, rung.seq) + rung.extra
                if rung.skip_reason:
                    row = skipped_row(rung, study_id, rung.skip_reason)
                    rows_by_key[key] = row
                    ledger.record_rung(rung.mode, rung.seq, row, "skipped")
                elif code_key in plan.reuse:
                    row, _digest = plan.reuse[code_key]
                    rows_by_key[key] = row
                    ledger.record_rung(rung.mode, rung.seq, row, "reused")
            # then one worker per artifact set, then its gate
            for (model_id, M) in profile.artifact_sets():
                todo = [r for r in rungs if (r.model_id, r.M) == (model_id, M) and r.skip_reason is None
                        and (r.mode, r.seq) + r.extra in set(plan.remeasure)]
                if not todo:
                    continue
                compiled = profile.compiled[(model_id, M)]
                if dry_run:
                    for r in todo:
                        row = skipped_row(r, study_id, "dry run: no device was dispatched to")
                        rows_by_key[resume_mod.rung_key(r.mode, r.seq, r.extra)] = row
                        ledger.record_rung(r.mode, r.seq, row, "skipped")
                    continue
                result = worker_fn(model_id, M, todo, compiled=compiled)
                verdict = None
                if verify and result.get("gate"):
                    try:
                        verdict = verifier(model_id, M, result["gate"], compiled=compiled)
                    except Exception as exc:
                        verdict = {"passed": False, "returncode": None, "per_prompt": [], "report_json": None, "error": f"{type(exc).__name__}: {exc}"}
                    verdicts[f"{model_id}_M{M}"] = {k: v for k, v in verdict.items() if k != "log"}
                gate_map = (result.get("gate") or {}).get("map") or {}
                for row in result["rows"]:
                    rung = next(r for r in todo if r.case_id == row["study_case_id"])
                    stamp_verdict(row, verdict, gate_map.get(rung.case_id))
                    rows_by_key[resume_mod.rung_key(rung.mode, rung.seq, rung.extra)] = row
                    ledger.record_rung(rung.mode, rung.seq, row, "measured")
            power_columns = monitor.stats()
    finally:
        ledger.close_session()
    wall = time.perf_counter() - t0

    # write one CSV per model, rung order
    rows_out = [rows_by_key[k] for k in by_key if k in rows_by_key]
    for rel in expected_files:
        mine = [r for r in rows_out if f"model_{r['model_id']}.csv" == rel]
        if mine:
            results_io.write_rows(out_dir / rel, mine)
    fidelity = resume_mod.fidelity_problems(plan.reuse, rows_out)
    for line in fidelity:
        print(f"[run-model] RESUME DEFECT {line}")
    after = resume_mod.scan(out_dir, expected_files)
    walk_block = resume_mod.walk_block(ledger, after, profile=profile, fidelity=fidelity)
    built = gate(profile, out_dir, repo=repo, conditions=conditions, toolchain=toolchain, walk=walk_block)
    conditions_after = manifest.observe_conditions() if not dry_run else conditions

    by_status: dict[str, int] = {}
    for row in rows_out:
        by_status[str(row["run_status"])] = by_status.get(str(row["run_status"]), 0) + 1
    report = {
        "study_id": study_id,
        "profile": profile.summary(),
        "started_utc": started.isoformat(),
        "wall_clock_sec": round(wall, 3),
        "rungs": [{"case_id": r["study_case_id"], "run_status": r["run_status"], "tokens_per_second": r.get("tokens_per_second"),
                   "plan_hash": r.get("plan_hash"), "failure_message": r.get("failure_message") or None} for r in rows_out],
        "rungs_by_status": by_status,
        "rungs_by_source": run_profile._rung_sources(ledger),
        "resume_requested": resume,
        "resume_defects": fidelity,
        "session_id": ledger.sessions[-1]["session_id"],
        "condition_splices": walk_block["condition_splices"],
        "npu_power_mode_before": conditions["npu_power_mode"],
        "npu_power_mode_after": conditions_after["npu_power_mode"],
        "verify": verdicts,
        "complete": built["complete"],
        "incomplete_reasons": built["incomplete_reasons"],
        "power_over_whole_walk": power_columns,
        "devq_job_id": os.environ.get("DEVQ_JOB_ID"),
        "dry_run": dry_run,
    }
    (out_dir / RUN_REPORT_NAME).write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"[run-model] wrote {out_dir / RUN_REPORT_NAME}")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _bound_profile(args):
    from dataclasses import replace

    prof = model_profiles.profile(args.profile)
    if getattr(args, "models", None):
        wanted = tuple(m.strip() for m in args.models.split(",") if m.strip())
        prof = replace(prof, models=wanted, prefill_Ms={m: prof.prefill_Ms[m] for m in wanted})
    compiled, notes = model_profiles.discover_compiled(prof.models, args.compiled_root)
    return prof.bind(compiled, notes)


def _print_plan(prof):
    print(f"[run-model] profile {prof.name}: {prof.description}")
    for r in prof.rungs():
        print(f"[run-model]   {'SKIP' if r.skip_reason else 'run '} {r.case_id}" + (f"  {r.skip_reason}" if r.skip_reason else ""))
    for rel, counts in prof.expected_rows().items():
        print(f"[run-model] expect {rel:<24} rows {counts['rows']:>3}  passed {counts['measured']:>3}  skipped {counts['skipped']:>3}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, out_required=True):
        p.add_argument("--profile", default="model-smoke")
        p.add_argument("--out-dir", required=out_required)
        p.add_argument("--models", default=None, help="comma-separated subset of the profile's models")
        p.add_argument("--compiled-root", default=None)
        return p

    r = common(sub.add_parser("run"))
    r.add_argument("--study-id", required=True)
    r.add_argument("--resume", action="store_true")
    r.add_argument("--dry-run", action="store_true", help="no device: every measurable rung is a complete skipped row")
    r.add_argument("--no-verify", action="store_true")
    r.add_argument("--power-backend", default="auto")
    common(sub.add_parser("gate"))
    common(sub.add_parser("plan"), out_required=False)
    c = sub.add_parser("compile")
    c.add_argument("--model", required=True)
    c.add_argument("--M", type=int, required=True)
    c.add_argument("--compiled-root", required=True)
    c.add_argument("--o-ffn-gemm-method", default=None, help="force the Qwen O+FFN cascade's GEMM method (recorded as a deviation)")
    w = sub.add_parser("worker")
    w.add_argument("--rungs", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--prompts-dir", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "worker":
        return worker(args)
    if args.cmd == "compile":
        target = Path(args.compiled_root) / args.model / f"M{args.M}"
        cache = target / model_adapter.PREFILL_CACHE
        print(f"[run-model] compiling {args.model} prefill at M={args.M} into {cache} (cwd {target})")
        try:
            man = model_adapter.compile_prefill(args.model, args.M, cache, cwd=target, o_ffn_gemm_method=args.o_ffn_gemm_method)
        except Exception as exc:
            # the wall, verbatim, where `discover_compiled` reads it into the
            # rung's skip reason -- a point that cannot be compiled says why
            cache.mkdir(parents=True, exist_ok=True)
            for stale in (cache / "manifest.json",):
                if stale.exists():
                    stale.unlink()
            (cache / model_adapter.COMPILE_NOTE).write_text(json.dumps(
                {"model_id": args.model, "M": args.M, "o_ffn_gemm_method": args.o_ffn_gemm_method,
                 "failed": f"{type(exc).__name__}: {str(exc).splitlines()[-1][:400]}"}, indent=1), encoding="utf-8")
            raise
        print(f"[run-model] compiled {sorted(k for k in man if not k.startswith('_'))} in {man['_compile']['wall_s']:.1f}s")
        return 0
    prof = _bound_profile(args)
    if args.cmd == "plan":
        _print_plan(prof)
        return 0
    if args.cmd == "gate":
        built = gate(prof, args.out_dir)
        return 0 if built["complete"] else 1
    _print_plan(prof)
    mode = None
    if not args.dry_run:
        mode = run_profile._require_turbo()
        print(f"[run-model] NPU power mode before: {mode}")
        if run_profile.device_preflight() is False:
            return 2
    report = run(prof, args.out_dir, study_id=args.study_id, resume=args.resume, npu_power_mode=mode,
                 verify=not args.no_verify, dry_run=args.dry_run, power_backend=args.power_backend)
    print(f"[run-model] NPU power mode after: {report['npu_power_mode_after']}")
    print(f"[run-model] complete: {report['complete']}  rungs {report['rungs_by_status']}")
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
