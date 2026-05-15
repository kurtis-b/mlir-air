# SPDX-License-Identifier: MIT

from __future__ import annotations

import copy
from typing import Any

from .schema import required_backends, select_cases

SHAPE_LADDER: dict[str, list[dict[str, Any]]] = {
    "tiny_ci": [
        {"name": "tiny_m1_k64_h64_n32", "M": 1, "K": 64, "H": 64, "N": 32},
        {"name": "tiny_m4_k128_h128_n64", "M": 4, "K": 128, "H": 128, "N": 64},
    ],
    "medium": [
        {"name": "medium_m8_k512_h512_n256", "M": 8, "K": 512, "H": 512, "N": 256},
        {
            "name": "medium_m32_k1024_h2048_n512",
            "M": 32,
            "K": 1024,
            "H": 2048,
            "N": 512,
        },
    ],
    "llm_like": [
        {
            "name": "llm_m32_k2048_h8192_n2048",
            "M": 32,
            "K": 2048,
            "H": 8192,
            "N": 2048,
        },
        {
            "name": "llm_m128_k4096_h11008_n4096",
            "M": 128,
            "K": 4096,
            "H": 11008,
            "N": 4096,
        },
    ],
}


def shape_workloads(
    suite: str, base_manifest: dict[str, Any], matrix: dict[str, Any]
) -> list[dict[str, Any]]:
    workloads = []
    for shape in SHAPE_LADDER[suite]:
        manifest = copy.deepcopy(base_manifest)
        manifest["model"].update(
            {
                "M": shape["M"],
                "K": shape["K"],
                "H": shape["H"],
                "N": shape["N"],
                "shape_tier": suite,
            }
        )
        manifest["paths"]["artifacts"] = f"artifacts/{suite}/{shape['name']}"
        manifest["paths"][
            "generated_air_sources"
        ] = f"artifacts/{suite}/{shape['name']}/generated_air"
        manifest["workload"] = {
            "suite": suite,
            "name": shape["name"],
            "description": "bf16 prefill GEMM followed by bf16 decode GEMV",
        }
        workloads.append(
            {
                "suite": suite,
                "name": shape["name"],
                "shape": copy.deepcopy(shape),
                "manifest": manifest,
                "cases": copy.deepcopy(matrix["cases"]),
            }
        )
    return workloads


def suite_workloads(
    suites: list[str], base_manifest: dict[str, Any], matrix: dict[str, Any]
) -> list[dict[str, Any]]:
    workloads: list[dict[str, Any]] = []
    for suite in suites:
        workloads.extend(shape_workloads(suite, base_manifest, matrix))
    return workloads


__all__ = [
    "SHAPE_LADDER",
    "required_backends",
    "select_cases",
    "shape_workloads",
    "suite_workloads",
]
