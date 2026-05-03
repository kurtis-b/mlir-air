# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any

from .crossover import crossover_markdown


def suite_summary_markdown(summary: dict[str, Any], title: str) -> str:
    lines = [
        f"# {title}",
        "",
        f"- Schema: `{summary.get('schema_version')}`",
        f"- Measurement mode: `{summary.get('measurement_mode')}`",
        f"- Transfer mode: `{summary.get('transfer_mode')}`",
        f"- Decode weight storage: `{summary.get('decode_weight_storage', 'bf16')}`",
        "",
        "| Suite | Workload | Case | Placement | Mean e2e ms | Prefill ms | Decode ms | Validation | Transfer | Quant |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for workload in summary.get("workloads", []):
        for result in workload.get("cases", []):
            placement = (
                f"{result['stage_backends']['prefill']}->"
                f"{result['stage_backends']['decode']}"
            )
            transfer = result.get("transfer_summary", {})
            quantized = result.get("quantized_decode", {})
            quant = "bf16"
            if quantized.get("enabled"):
                metadata = quantized.get("metadata") or {}
                detail = quantized.get("detail") or {}
                fused = "hw" if quantized.get("hardware_fused") else "cpu"
                quant = (
                    f"{quantized.get('quant_kind') or metadata.get('quant_kind')}:{fused} "
                    f"deq={detail.get('dequant_ms', 0.0):.6f}ms"
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
                f"{transfer.get('transfer_semantics')} / {transfer.get('total_bytes')} B | "
                f"{quant} |"
            )
    lines.extend(["", *crossover_markdown(summary)])
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
            "Direct GPU/NPU handoff is only claimed when result artifacts report audited device-resident buffers and zero NumPy host materializations on the GPU/NPU edge.",
        ]
    )
    return "\n".join(lines)
