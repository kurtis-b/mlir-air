#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gemma3 paper-benchmark result JSON helpers."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

from gemma3_artifacts import MODEL_SPECS, discover_model_artifacts, model_spec
from gemma3_environment import capture_environment
from gemma3_nonlinears import paper_match_blockers
from gemma3_paper_compare import load_targets, target_by_id

DEFAULT_POWER_WATTS = {"cpu": None, "gpu": None, "npu": None, "total": None}
TABLE_BY_METRIC = {
    "prefill_ttft_seconds": "prefill/decode text tables",
    "decode_tps": "prefill/decode text tables",
    "vision_ttft_seconds": "vision TTFT table",
}
UNIT_BY_METRIC = {
    "prefill_ttft_seconds": "seconds",
    "decode_tps": "tokens_per_second",
    "vision_ttft_seconds": "seconds",
}


def infer_metric(model_variant: str, decode_tokens: int, explicit_metric: str | None = None) -> str:
    if explicit_metric:
        return explicit_metric
    if model_variant.endswith("vision") and decode_tokens == 0:
        return "vision_ttft_seconds"
    if decode_tokens > 0:
        return "decode_tps"
    return "prefill_ttft_seconds"


def find_paper_target(
    *,
    model_variant: str,
    backend: str,
    metric: str,
    sequence_length: int,
) -> dict[str, Any]:
    data = load_targets()
    matches = [
        target
        for target in data["targets"]
        if target.get("model_variant") == model_variant
        and target.get("backend") == backend
        and target.get("metric") == metric
        and int(target.get("sequence_length", -1)) == sequence_length
    ]
    if len(matches) != 1:
        raise ValueError(
            "expected one paper target for "
            f"model={model_variant} backend={backend} metric={metric} "
            f"sequence_length={sequence_length}, got {len(matches)}"
        )
    return matches[0]


def missing_artifact_notes(inventory: Any) -> list[str]:
    notes: list[str] = []
    if not inventory.has_weight_files:
        notes.append("missing *.safetensors")
    if not inventory.tokenizer_exists:
        notes.append("missing tokenizer file")
    if not inventory.optional_packages.get("safetensors", False):
        notes.append("missing python:safetensors")
    if not any(inventory.optional_packages.get(pkg, False) for pkg in ("tokenizers", "sentencepiece", "transformers")):
        notes.append("missing python tokenizer package")
    return notes


def fallback_records() -> list[dict[str, Any]]:
    return [
        {
            "name": blocker.operation,
            "status": blocker.timed_window_status,
            "contributes_to_timing": True,
            "measured": False,
        }
        for blocker in paper_match_blockers()
    ]


def build_paper_result(
    *,
    model_variant: str,
    backend: str,
    weights_dir: Path | None,
    tokenizer: Path | None,
    prompt_len: int,
    decode_tokens: int,
    metric: str | None,
    warmup_iters: int,
    timed_iters: int,
    artifact_format: str,
    compile_time_included: bool,
    command: list[str] | None = None,
    power_sample: bool = False,
    trace_size: int | None = None,
    debug_ir: bool = False,
) -> dict[str, Any]:
    metric = infer_metric(model_variant, decode_tokens, metric)
    phase = "decode" if metric == "decode_tps" else "prefill"
    spec = model_spec(model_variant)
    spec.validate_sequence_length(prompt_len, phase=phase)
    target = find_paper_target(
        model_variant=model_variant,
        backend=backend,
        metric=metric,
        sequence_length=prompt_len,
    )
    inventory = discover_model_artifacts(model_variant, weights_dir=weights_dir, tokenizer=tokenizer)
    env = capture_environment(
        artifact_format=artifact_format,
        compile_time_included=compile_time_included,
        timing_window="runtime_only",
        require_hardware=False,
    )

    notes = missing_artifact_notes(inventory)
    if not inventory.can_load_real_artifacts:
        classification = "MISSING_REAL_ARTIFACTS"
        correctness = "BLOCKED_REAL_ARTIFACTS"
    else:
        classification = "REAL_MODEL_EXECUTION_NOT_IMPLEMENTED"
        correctness = "BLOCKED_EXECUTION_NOT_IMPLEMENTED"
        notes.append("real artifact inventory is present but execution is not implemented")
    if env.get("missing_paper_fields"):
        notes.append("environment is not paper-comparable: " + ",".join(env["missing_paper_fields"]))
    if power_sample:
        notes.append("power sampling requested but no telemetry backend is implemented")
    if trace_size is not None:
        notes.append(f"trace_size={trace_size} requested for future NPU run")
    if debug_ir:
        notes.append("debug_ir requested for future compile run")

    command_text = shlex.join(command if command is not None else sys.argv)
    paper_value = target.get("paper_value")
    if paper_value is None and "paper_min" in target and "paper_max" in target:
        paper_value = None

    return {
        "schema_version": 1,
        "target_id": target["id"],
        "paper_source": target.get("paper_source", "arxiv_pdf_v2"),
        "paper_table": TABLE_BY_METRIC.get(metric, "paper_targets.json"),
        "model_variant": model_variant,
        "backend": backend,
        "metric": metric,
        "sequence_length": prompt_len,
        "decode_tokens": decode_tokens,
        "local_value": None,
        "paper_value": paper_value,
        "unit": target.get("unit", UNIT_BY_METRIC.get(metric, "unknown")),
        "delta_pct": None,
        "classification": classification,
        "correctness": correctness,
        "host_fallbacks": fallback_records(),
        "command": command_text,
        "git_commit": env["git"].get("commit"),
        "dirty_worktree": env["git"].get("dirty_worktree"),
        "xrt_version": env["xrt"].get("version"),
        "npu_power_mode": env["npu"].get("power_mode"),
        "artifact_format": artifact_format,
        "warmup_iters": warmup_iters,
        "timed_iters": timed_iters,
        "compile_time_included": compile_time_included,
        "power_watts": dict(DEFAULT_POWER_WATTS),
        "environment_comparable": env.get("paper_comparable"),
        "missing_environment_fields": env.get("missing_paper_fields", []),
        "artifact_inventory": inventory.to_json_dict(),
        "notes": notes,
    }


def write_result_json(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


def format_result(result: dict[str, Any]) -> str:
    notes = ";".join(result.get("notes", [])) or "none"
    return (
        f"GEMMA3_PAPER_RESULT target={result['target_id']} "
        f"class={result['classification']} correctness={result['correctness']} "
        f"local={result['local_value']} paper={result['paper_value']} notes={notes}"
    )


def _self_test() -> None:
    result = build_paper_result(
        model_variant="gemma3-1b",
        backend="npu",
        weights_dir=Path("/tmp/gemma3_missing_weights"),
        tokenizer=None,
        prompt_len=1024,
        decode_tokens=128,
        metric=None,
        warmup_iters=3,
        timed_iters=10,
        artifact_format="elf",
        compile_time_included=False,
        command=["gemma3_results.py", "--self-test"],
    )
    if result["target_id"] != "decode_tps_gemma3_1b_npu_1024":
        raise AssertionError(result["target_id"])
    if result["classification"] != "MISSING_REAL_ARTIFACTS":
        raise AssertionError(result["classification"])
    if not result["host_fallbacks"]:
        raise AssertionError("expected nonlinear fallback records")
    print(format_result(result))
    print("GEMMA3_RESULT_JSON_SELF_TEST: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma3 paper-benchmark result JSON helper")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model-variant", choices=sorted(MODEL_SPECS), default="gemma3-1b")
    parser.add_argument("--backend", choices=["cpu", "igpu", "npu"], default="npu")
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--prompt-len", type=int, default=1024)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--metric", choices=sorted(TABLE_BY_METRIC))
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--timed-iters", type=int, default=10)
    parser.add_argument("--artifact-format", choices=["elf", "xclbin"], default="elf")
    parser.add_argument("--compile-time-included", action="store_true")
    parser.add_argument("--power-sample", action="store_true")
    parser.add_argument("--trace-size", type=int)
    parser.add_argument("--debug-ir", action="store_true")
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return 0

    result = build_paper_result(
        model_variant=args.model_variant,
        backend=args.backend,
        weights_dir=args.weights_dir,
        tokenizer=args.tokenizer,
        prompt_len=args.prompt_len,
        decode_tokens=args.decode_tokens,
        metric=args.metric,
        warmup_iters=args.warmup_iters,
        timed_iters=args.timed_iters,
        artifact_format=args.artifact_format,
        compile_time_included=args.compile_time_included,
        power_sample=args.power_sample,
        trace_size=args.trace_size,
        debug_ir=args.debug_ir,
        command=sys.argv,
    )
    print(format_result(result))
    if result["classification"] == "MISSING_REAL_ARTIFACTS":
        print("GEMMA3_PAPER_BENCHMARK_BLOCKED: missing_real_artifacts")
    if args.result_json:
        write_result_json(result, args.result_json)
        print(f"GEMMA3_RESULT_JSON: {args.result_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
