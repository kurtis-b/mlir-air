# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any

VALID_BACKENDS = {"cpu", "gpu", "npu"}
VALID_DTYPES = {"bf16", "f16"}
VALID_TRANSFER_MODES = {"host", "direct"}


def _require_positive_int(container: dict[str, Any], key: str) -> None:
    value = container.get(key)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    model = manifest.get("model")
    if not isinstance(model, dict):
        raise ValueError("manifest.model must be an object")
    for key in ("M", "K", "H", "N"):
        _require_positive_int(model, key)
    if model.get("dtype") not in VALID_DTYPES:
        raise ValueError(f"model.dtype must be one of {sorted(VALID_DTYPES)}")

    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("manifest.runtime must be an object")
    stage_backends = runtime.get("stage_backends")
    if not isinstance(stage_backends, dict):
        raise ValueError("runtime.stage_backends must be an object")
    for stage in ("prefill", "decode"):
        backend = stage_backends.get(stage)
        if backend not in VALID_BACKENDS:
            raise ValueError(f"runtime.stage_backends.{stage} is invalid: {backend}")
    transfer_mode = runtime.get("transfer_mode", "host")
    if transfer_mode not in VALID_TRANSFER_MODES:
        raise ValueError(f"runtime.transfer_mode is invalid: {transfer_mode}")

    for section in ("inputs", "weights", "paths", "compiler", "artifacts"):
        if section not in manifest:
            raise ValueError(f"manifest.{section} is required")
    return manifest


def validate_case(case: dict[str, Any]) -> dict[str, Any]:
    if not case.get("name"):
        raise ValueError("case.name is required")
    for key in ("prefill_backend", "decode_backend"):
        backend = case.get(key)
        if backend not in VALID_BACKENDS:
            raise ValueError(f"{key} is invalid: {backend}")
    transfer_mode = case.get("transfer_mode", "host")
    if transfer_mode not in VALID_TRANSFER_MODES:
        raise ValueError(f"case.transfer_mode is invalid: {transfer_mode}")
    return case


def case_stage_backends(case: dict[str, Any]) -> dict[str, str]:
    validate_case(case)
    return {
        "prefill": str(case["prefill_backend"]),
        "decode": str(case["decode_backend"]),
    }


def contains_npu(stage_backends: dict[str, str]) -> bool:
    return any(backend == "npu" for backend in stage_backends.values())


def case_map(matrix: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {case["name"]: case for case in matrix.get("cases", [])}


def select_cases(matrix: dict[str, Any], names: list[str]) -> list[dict[str, Any]]:
    cases = case_map(matrix)
    return [cases[name] for name in names if name in cases]


def required_backends(cases: list[dict[str, Any]], allow_npu: bool) -> set[str]:
    needed: set[str] = set()
    for case in cases:
        for backend in case_stage_backends(case).values():
            if backend == "gpu":
                needed.add("gpu")
            elif backend == "npu" and allow_npu:
                needed.add("npu")
    return needed
