# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any

BASELINES = [
    "cpu_only",
    "gpu_only",
    "npu_only",
    "gpu_prefill_npu_decode_host",
    "npu_prefill_gpu_decode_host",
    "gpu_prefill_npu_decode_direct",
    "npu_prefill_gpu_decode_direct",
]


def crossover_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for workload in summary.get("workloads", []):
        cases = {
            result["case_name"]: result
            for result in workload.get("cases", [])
            if result.get("correctness", {}).get("validation_status") == "pass"
        }
        for mixed in (
            "gpu_prefill_npu_decode_direct",
            "npu_prefill_gpu_decode_direct",
        ):
            if mixed not in cases:
                continue
            mixed_ms = float(cases[mixed]["latency_ms"]["mean"])
            for baseline in (
                "cpu_only",
                "gpu_only",
                "npu_only",
                mixed.replace("_direct", "_host"),
            ):
                if baseline not in cases:
                    continue
                baseline_ms = float(cases[baseline]["latency_ms"]["mean"])
                speedup = baseline_ms / mixed_ms if mixed_ms > 0.0 else 0.0
                rows.append(
                    {
                        "suite": workload.get("suite"),
                        "workload": workload.get("name"),
                        "mixed_case": mixed,
                        "baseline": baseline,
                        "mixed_mean_ms": mixed_ms,
                        "baseline_mean_ms": baseline_ms,
                        "speedup": speedup,
                        "classification": classify_speedup(speedup),
                    }
                )
    return rows


def classify_speedup(speedup: float) -> str:
    if speedup >= 1.05:
        return "wins"
    if speedup <= 0.95:
        return "loses"
    return "inconclusive"


def crossover_markdown(summary: dict[str, Any]) -> list[str]:
    rows = crossover_rows(summary)
    if not rows:
        return [
            "## Crossover",
            "",
            "No audited direct-handoff mixed cases were present in this run.",
        ]
    lines = [
        "## Crossover",
        "",
        "| Suite | Workload | Direct case | Baseline | Speedup | Classification |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['suite']} | "
            f"{row['workload']} | "
            f"{row['mixed_case']} | "
            f"{row['baseline']} | "
            f"{row['speedup']:.3f} | "
            f"{row['classification']} |"
        )
    return lines
