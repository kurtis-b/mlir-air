# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path
from typing import Any

from manifest import EDGE_STUDY_SCHEMA_VERSION


def display(value: Any) -> str:
    return "-" if value is None else str(value)


def format_case(case: dict[str, Any]) -> str:
    stages = case["stage_backends"]
    return f"{stages['router']} / {stages['expert0']} / {stages['expert1']} / {stages['aggregation']}"


def matrix_report_markdown(summary: dict[str, Any], title: str) -> str:
    cases = summary["cases"]
    skipped = summary.get("skipped", [])
    if not cases:
        raise ValueError("No benchmark cases found in summary JSON")

    includes_npu = any("npu" in case["stage_backends"].values() for case in cases)
    best_case = min(cases, key=lambda case: case["latency_ms"]["mean"])
    lines = [
        f"# {title}",
        "",
        "## Scope",
        "",
        "- This report summarizes the current heterogeneous MoE benchmark matrix.",
        "- The runtime supports independent router, expert, and aggregation placement across CPU, GPU, and NPU backends.",
        "- Transfer accounting uses the NumPy host-array model unless a result explicitly states otherwise.",
        "",
        "## Summary",
        "",
        f"- Fastest case: `{best_case['case_name']}` at `{best_case['latency_ms']['mean']:.3f} ms` mean latency.",
        f"- Fastest case stage placement: `{format_case(best_case)}`.",
        "",
        "## Results",
        "",
        "| Case | Router Mode | Stage Placement | Mean ms | Max Abs Error | Torch | Expert Overlap us |",
        "| --- | --- | --- | ---: | ---: | --- | ---: |",
    ]

    for case in cases:
        lines.append(
            "| "
            + " | ".join(
                [
                    case["case_name"],
                    case["router_mode"],
                    format_case(case),
                    f"{case['latency_ms']['mean']:.3f}",
                    f"{case['stage_metrics']['output']['max_abs_error']:.6f}",
                    "pass" if case["torch_validation"]["ok"] else "fail",
                    f"{case['trace_summary']['overlap']['expert0_expert1_us']:.3f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The trace overlap metric is derived from host-visible spans and is suitable for development triage.",
            (
                "- This report includes NPU hardware-backed runs."
                if includes_npu
                else "- NPU-tagged cases were skipped for this report."
            ),
        ]
    )

    if skipped:
        lines.extend(["", "## Skipped Cases", ""])
        for item in skipped:
            lines.append(f"- `{item['case_name']}`: {item['reason']}.")

    lines.append("")
    return "\n".join(lines)


def suite_summary_markdown(summary: dict[str, Any], title: str) -> str:
    lines = [
        f"# {title}",
        "",
        f"- Workloads: `{len(summary['workloads'])}`",
        f"- Cases run: `{sum(len(workload['cases']) for workload in summary['workloads'])}`",
        "",
        "## Fastest Per Workload",
        "",
        "| Workload | Model | Class | Shape | Context | Routed Tokens | Chunk | Scale | Routing Profile | Fastest Case | Mean ms | Transfer MB | Torch |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | --- |",
    ]
    for workload in summary["workloads"]:
        if not workload["cases"]:
            continue
        fastest = min(workload["cases"], key=lambda case: case["latency_ms"]["mean"])
        model = workload["model"]
        preset = workload.get("model_preset", {})
        shape = f"{model['hidden_size']}x{model['ffn_size']}"
        lines.append(
            "| "
            + " | ".join(
                [
                    workload["name"],
                    display(preset.get("model_id")),
                    display(preset.get("model_class")),
                    shape,
                    str(workload.get("context_length", "-")),
                    str(workload.get("routed_tokens", model["batch_tokens"])),
                    str(workload.get("kernel_chunk_tokens", model["batch_tokens"])),
                    f"{workload['input_scale']:.4f}",
                    workload["routing_profile"],
                    fastest["case_name"],
                    f"{fastest['latency_ms']['mean']:.3f}",
                    f"{fastest['transfer_summary']['total_bytes'] / 1_000_000.0:.3f}",
                    "pass" if fastest["torch_validation"]["ok"] else "fail",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## All Results",
            "",
            "| Workload | Model | Context | Routed Tokens | Chunk | Case | Mean ms | P95 ms | Transfer MB | Max Abs Error | Expert Overlap us |",
            "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for workload in summary["workloads"]:
        preset = workload.get("model_preset", {})
        for case in workload["cases"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        workload["name"],
                        display(preset.get("model_id")),
                        str(workload.get("context_length", "-")),
                        str(workload.get("routed_tokens", workload["model"]["batch_tokens"])),
                        str(workload.get("kernel_chunk_tokens", workload["model"]["batch_tokens"])),
                        case["case_name"],
                        f"{case['latency_ms']['mean']:.3f}",
                        f"{case['latency_ms']['p95']:.3f}",
                        f"{case['transfer_summary']['total_bytes'] / 1_000_000.0:.3f}",
                        f"{case['stage_metrics']['output']['max_abs_error']:.6f}",
                        f"{case['trace_summary']['overlap']['expert0_expert1_us']:.3f}",
                    ]
                )
                + " |"
            )
    lines.append("")
    return "\n".join(lines)


def is_single_backend(case: dict[str, Any]) -> bool:
    backends = case.get("stage_backends", {})
    values = {backends.get("router"), backends.get("expert0"), backends.get("expert1"), backends.get("aggregation")}
    return len(values) == 1


def is_mixed_backend(case: dict[str, Any]) -> bool:
    return not is_single_backend(case)


def case_backend_name(case: dict[str, Any]) -> str:
    if is_single_backend(case):
        return next(iter(set(case["stage_backends"].values())))
    return "mixed"


def bottleneck(case: dict[str, Any]) -> dict[str, Any]:
    latency_us = float(case.get("latency_ms", {}).get("mean", 0.0)) * 1000.0
    transfer_us = float(case.get("transfer_summary", {}).get("total_elapsed_us", 0.0))
    category = case.get("device_events", {}).get("by_category", {})
    stage_us = float(category.get("stage", {}).get("total_us", 0.0))
    control_us = float(category.get("control", {}).get("total_us", 0.0))
    candidates = {
        "stage_compute_or_launch": stage_us,
        "transfer_or_staging": transfer_us,
        "cpu_control": control_us,
    }
    name, value = max(candidates.items(), key=lambda item: item[1])
    return {
        "primary": name,
        "primary_us": value,
        "share_of_mean_latency": value / latency_us if latency_us > 0.0 else None,
        "components_us": candidates,
    }


def flatten_suite_cases(suite_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for workload in suite_summary.get("workloads", []):
        for case in workload.get("cases", []):
            rows.append(
                {
                    "suite": workload["suite"],
                    "workload": workload["name"],
                    "case": case,
                    "backend_kind": case_backend_name(case),
                    "is_single_backend": is_single_backend(case),
                    "is_mixed_backend": is_mixed_backend(case),
                }
            )
    return rows


def compact_case(case: dict[str, Any] | None) -> dict[str, Any] | None:
    if case is None:
        return None
    return {
        "case_name": case["case_name"],
        "latency_ms": case["latency_ms"],
        "stage_backends": case["stage_backends"],
        "transfer_model": case["transfer_summary"].get("model"),
        "transfer_bytes": case["transfer_summary"]["total_bytes"],
        "transfer_elapsed_us": case["transfer_summary"]["total_elapsed_us"],
        "torch_ok": case["torch_validation"]["ok"],
        "npu_executed": case["npu_development"]["executed"],
        "bottleneck": bottleneck(case),
    }


def summarize_workload(workload: dict[str, Any]) -> dict[str, Any]:
    cases = workload.get("cases", [])
    single = [case for case in cases if is_single_backend(case)]
    mixed = [case for case in cases if is_mixed_backend(case)]
    fastest_single = min(single, key=lambda case: case["latency_ms"]["mean"], default=None)
    fastest_mixed = min(mixed, key=lambda case: case["latency_ms"]["mean"], default=None)
    fastest_overall = min(cases, key=lambda case: case["latency_ms"]["mean"], default=None)
    mixed_speedup_vs_best_single = None
    mixed_wins = False
    if fastest_single and fastest_mixed:
        single_ms = float(fastest_single["latency_ms"]["mean"])
        mixed_ms = float(fastest_mixed["latency_ms"]["mean"])
        mixed_speedup_vs_best_single = single_ms / mixed_ms if mixed_ms > 0.0 else None
        mixed_wins = bool(mixed_speedup_vs_best_single and mixed_speedup_vs_best_single > 1.0)
    return {
        "suite": workload["suite"],
        "name": workload["name"],
        "model": workload["model"],
        "context_length": workload.get("context_length"),
        "routed_tokens": workload.get("routed_tokens"),
        "kernel_chunk_tokens": workload.get("kernel_chunk_tokens"),
        "case_count": len(cases),
        "skipped": workload.get("skipped", []),
        "fastest_overall": compact_case(fastest_overall),
        "fastest_single_backend": compact_case(fastest_single),
        "fastest_mixed_backend": compact_case(fastest_mixed),
        "mixed_speedup_vs_best_single": mixed_speedup_vs_best_single,
        "mixed_wins": mixed_wins,
    }


def build_edge_study_summary(suite_summary: dict[str, Any], command: list[str]) -> dict[str, Any]:
    workload_summaries = [summarize_workload(workload) for workload in suite_summary.get("workloads", [])]
    flat_cases = flatten_suite_cases(suite_summary)
    return {
        "schema_version": EDGE_STUDY_SCHEMA_VERSION,
        "command": command,
        "study_questions": [
            "Can CPU/iGPU/NPU execute the MoE harness end to end correctly?",
            "Which fixed placement wins for each workload after transfer and launch overhead?",
            "Does any mixed CPU/iGPU/NPU placement beat the best single-backend baseline?",
            "Which bottleneck blocks efficient heterogeneous execution?",
        ],
        "acceptance_criteria": [
            "Every latency must have correctness status, placement, transfer accounting, and cold/warm measurement mode.",
            "Heterogeneous speedups are claimed only when mixed placement beats the best single-backend measured case.",
            "Direct iGPU/NPU peer movement is not assumed unless transfer events report a real implementation.",
        ],
        "workloads": workload_summaries,
        "case_count": len(flat_cases),
        "mixed_case_count": sum(1 for row in flat_cases if row["is_mixed_backend"]),
        "single_backend_case_count": sum(1 for row in flat_cases if row["is_single_backend"]),
    }


def edge_study_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Heterogeneous MoE Edge-Efficiency Study",
        "",
        "## Study Questions",
        "",
    ]
    lines.extend(f"- {question}" for question in summary["study_questions"])
    lines.extend(
        [
            "",
            "## Workload Results",
            "",
            "| Workload | Fastest Single | Single Mean ms | Fastest Mixed | Mixed Mean ms | Mixed Speedup | Mixed Wins | Primary Bottleneck |",
            "| --- | --- | ---: | --- | ---: | ---: | --- | --- |",
        ]
    )
    for workload in summary["workloads"]:
        single = workload["fastest_single_backend"]
        mixed = workload["fastest_mixed_backend"]
        speedup = workload["mixed_speedup_vs_best_single"]
        lines.append(
            "| "
            + " | ".join(
                [
                    workload["name"],
                    single["case_name"] if single else "-",
                    f"{single['latency_ms']['mean']:.3f}" if single else "-",
                    mixed["case_name"] if mixed else "-",
                    f"{mixed['latency_ms']['mean']:.3f}" if mixed else "-",
                    f"{speedup:.3f}" if speedup is not None else "-",
                    "yes" if workload["mixed_wins"] else "no",
                    mixed["bottleneck"]["primary"] if mixed else "-",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Interpretation Guardrails", ""])
    lines.extend(f"- {criterion}" for criterion in summary["acceptance_criteria"])
    lines.append("")
    return "\n".join(lines)


def write_markdown(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
