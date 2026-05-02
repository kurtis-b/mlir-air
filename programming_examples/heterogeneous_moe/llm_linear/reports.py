# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any


def suite_summary_markdown(summary: dict[str, Any], title: str) -> str:
    lines = [
        f"# {title}",
        "",
        f"- Schema: `{summary.get('schema_version')}`",
        f"- Measurement mode: `{summary.get('measurement_mode')}`",
        f"- Transfer mode: `{summary.get('transfer_mode')}`",
        "",
        "| Suite | Workload | Case | Placement | Mean e2e ms | Prefill ms | Decode ms | Validation | Transfer bytes |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for workload in summary.get("workloads", []):
        for result in workload.get("cases", []):
            placement = (
                f"{result['stage_backends']['prefill']}->"
                f"{result['stage_backends']['decode']}"
            )
            lines.append(
                "| "
                f"{workload['suite']} | "
                f"{workload['name']} | "
                f"{result['case_name']} | "
                f"{placement} | "
                f"{result['latency_ms']['mean']:.6f} | "
                f"{result['stage_latency_ms']['prefill']['mean']:.6f} | "
                f"{result['stage_latency_ms']['decode']['mean']:.6f} | "
                f"{result['correctness']['validation_status']} | "
                f"{result['transfer_summary']['total_bytes']} |"
            )
    if any(workload.get("skipped") for workload in summary.get("workloads", [])):
        lines.extend(["", "## Skipped"])
        for workload in summary.get("workloads", []):
            for skipped in workload.get("skipped", []):
                lines.append(
                    f"- `{workload['name']}` / `{skipped['case_name']}`: {skipped['reason']}"
                )
    lines.extend(
        [
            "",
            "Direct GPU/NPU handoff is unsupported in Milestone 1. Mixed cases are host-staged.",
        ]
    )
    return "\n".join(lines)
