#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gemma3 real paper-reproduction blocker ledger."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

from gemma3.core.artifacts import discover_model_artifacts
from gemma3.core.config import synthetic_text_config
from gemma3.evidence.environment import capture_environment
from gemma3.evidence.paper_compare import unmeasured_host_fallbacks
from gemma3.evidence.results import fallback_records
from gemma3.model.vision import Gemma3VisionConfig, run_vision_prefill_or_disabled
from gemma3.paths import RESULTS_DIR


PRODUCTION_STATIC_BO_BLOCKER = "production-contiguous-static-weight-bo-not-used-by-fused-dqp-route"
NPU_RUNTIME_DECODE_LOOP_EVIDENCE = RESULTS_DIR / "gemma3_1b_npu_runtime_decode_loop.json"


@dataclass(frozen=True)
class PhaseBlockerStatus:
    phase: str
    model_variant: str
    status: str
    blockers: tuple[str, ...]

    def format(self) -> str:
        blockers = ",".join(self.blockers) if self.blockers else "none"
        return (
            f"phase {self.phase} model={self.model_variant} "
            f"status={self.status} blockers={blockers}"
        )


def _artifact_blockers(model_variant: str, weights_dir: Path | None = None) -> list[str]:
    inventory = discover_model_artifacts(model_variant, weights_dir=weights_dir)
    blockers: list[str] = []
    if not inventory.has_weight_files:
        blockers.append("missing-safetensors")
    if not inventory.config_exists:
        blockers.append("missing-config-json")
    if not inventory.tokenizer_exists:
        blockers.append("missing-tokenizer")
    if inventory.has_vision and not inventory.processor_exists:
        blockers.append("missing-processor")
    if not inventory.optional_packages.get("safetensors", False):
        blockers.append("missing-python-safetensors")
    if not any(inventory.optional_packages.get(pkg, False) for pkg in ("tokenizers", "sentencepiece", "transformers")):
        blockers.append("missing-python-tokenizer-package")
    return blockers




def _has_runtime_static_projection_bo_evidence(model_variant: str) -> bool:
    if model_variant != "gemma3-1b":
        return False
    try:
        data = json.loads(NPU_RUNTIME_DECODE_LOOP_EVIDENCE.read_text(encoding="utf-8"))
    except Exception:
        return False
    loop = data.get("npu_decode_loop")
    if not isinstance(loop, dict):
        return False
    return (
        data.get("status") == "DECODE_RUNTIME_PASS_WITH_BLOCKERS"
        and loop.get("status") == "DECODE_LOOP_DIAGNOSTIC_PASS"
        and loop.get("static_projection_argument_mode") == "manifest-contiguous-static-bo"
        and int(loop.get("static_projection_bo_set_count", 0) or 0) > 0
        and PRODUCTION_STATIC_BO_BLOCKER not in loop.get("remaining_paper_gaps", [])
    )


def _execution_blockers(model_variant: str, weights_dir: Path | None = None) -> list[str]:
    try:
        from gemma3.npu.wiring import build_wiring_plan

        blockers = list(build_wiring_plan(model_variant, weights_dir=weights_dir).blockers)
    except Exception:
        return ["npu-model-execution-not-implemented"]
    if _has_runtime_static_projection_bo_evidence(model_variant):
        blockers = [blocker for blocker in blockers if blocker != PRODUCTION_STATIC_BO_BLOCKER]
    return blockers


def _vision_blockers() -> list[str]:
    try:
        result = run_vision_prefill_or_disabled(
            synthetic_text_config(n_layers=2, local_window_len=4),
            Gemma3VisionConfig(enabled=True, image_token_count=4),
        )
    except Exception:
        return ["vision-path-contract-missing"]
    if result.status != "synthetic-cpu-reference":
        return ["vision-path-contract-missing"]
    return ["vision-npu-path-not-validated"]


def phase_blocker_statuses(weights_dir: Path | None = None) -> tuple[PhaseBlockerStatus, ...]:
    env = capture_environment(require_hardware=False)
    env_blockers = ["environment-not-paper-comparable"] if env.get("missing_paper_fields") else []
    nonlinear_blockers = (
        ["unmeasured-nonlinear-host-fallbacks"]
        if unmeasured_host_fallbacks({"host_fallbacks": fallback_records()})
        else []
    )
    rows = []
    for phase, model_variant, extra in (
        ("F", "gemma3-1b", ()),
        ("G", "gemma3-4b", ()),
        ("H", "gemma3-4b-vision", tuple(_vision_blockers())),
    ):
        artifact_blockers = _artifact_blockers(model_variant, weights_dir)
        execution_blockers = (
            _execution_blockers(model_variant, weights_dir) if not artifact_blockers else []
        )
        blockers = tuple(
            dict.fromkeys(
                artifact_blockers
                + env_blockers
                + nonlinear_blockers
                + execution_blockers
                + list(extra)
            )
        )
        rows.append(
            PhaseBlockerStatus(
                phase=phase,
                model_variant=model_variant,
                status="BLOCKED" if blockers else "READY",
                blockers=blockers,
            )
        )
    return tuple(rows)


def _self_test() -> None:
    statuses = phase_blocker_statuses()
    if len(statuses) != 3:
        raise AssertionError(statuses)
    for status in statuses:
        if status.status != "BLOCKED":
            raise AssertionError(status)
        print(status.format())
    print("GEMMA3_REPRODUCTION_BLOCKERS_SELF_TEST: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma3 paper-reproduction blocker ledger")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--weights-dir", type=Path)
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return 0
    for status in phase_blocker_statuses(weights_dir=args.weights_dir):
        print(status.format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
