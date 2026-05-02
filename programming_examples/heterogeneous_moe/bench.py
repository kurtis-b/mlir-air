#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from case_runner import RunCaseOptions, apply_case_to_manifest, run_case_with_trace
from manifest import load_json, project_dir, save_json
from orchestrator import MoERuntime
from results import CSV_FIELDNAMES, result_csv_row


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the fixed-placement heterogeneous MoE benchmark.")
    parser.add_argument("--manifest", default="default_manifest.json", help="Manifest path relative to this directory.")
    parser.add_argument("--prepare", action="store_true", help="Compile and load any selected non-CPU executors, then exit.")
    parser.add_argument("--iterations", type=int, default=None, help="Override iteration count from the manifest.")
    parser.add_argument("--warmup", type=int, default=None, help="Override warmup count from the manifest.")
    parser.add_argument(
        "--measurement-mode",
        choices=["cold", "warm", "both"],
        default="warm",
        help="Measure cold start, warm steady-state, or both. Validation is always untimed.",
    )
    parser.add_argument("--router-mode", choices=["top1", "top2"], default=None)
    parser.add_argument("--router-backend", choices=["cpu", "npu", "gpu"], default=None)
    parser.add_argument("--expert0-backend", choices=["cpu", "npu", "gpu"], default=None)
    parser.add_argument("--expert1-backend", choices=["cpu", "npu", "gpu"], default=None)
    parser.add_argument("--aggregation-backend", choices=["cpu", "npu", "gpu"], default=None)
    parser.add_argument("--transfer-mode", choices=["host", "peer", "auto"], default=None)
    parser.add_argument("--case-name", default=None, help="Optional label used in structured outputs.")
    parser.add_argument("--results-out", default=None, help="Optional JSON file for benchmark results.")
    parser.add_argument("--csv-out", default=None, help="Optional CSV file for a one-row benchmark summary.")
    parser.add_argument("--trace-out", default=None, help="Optional Chrome-trace JSON output.")
    parser.add_argument("--trace-summary-out", default=None, help="Optional JSON file for trace summary output.")
    parser.add_argument("--stage-metrics-out", default=None, help="Optional JSON file for per-stage correctness metrics.")
    parser.add_argument("--transfer-summary-out", default=None, help="Optional JSON file for transfer accounting summary.")
    parser.add_argument("--device-events-out", default=None, help="Optional JSON file for host/device event summary.")
    parser.add_argument("--npu-dev-report-out", default=None, help="Optional JSON file for the host-side NPU development report.")
    parser.add_argument("--require-torch", action="store_true", help="Fail if torch-backed validation is unavailable or fails.")
    args = parser.parse_args()

    manifest_path = (project_dir() / args.manifest).resolve()
    manifest = load_json(manifest_path)
    case = {
        "name": args.case_name or "adhoc",
        "router_mode": args.router_mode,
        "router_backend": args.router_backend,
        "expert0_backend": args.expert0_backend,
        "expert1_backend": args.expert1_backend,
        "aggregation_backend": args.aggregation_backend,
        "transfer_mode": args.transfer_mode,
    }

    if args.prepare:
        with MoERuntime(apply_case_to_manifest(manifest, case)) as runtime:
            runtime.prepare()
        print("Prepared selected executors.")
        return 0

    results, trace = run_case_with_trace(
        manifest,
        case,
        RunCaseOptions(
            manifest_path=manifest_path,
            case_name=args.case_name or "adhoc",
            iterations=args.iterations,
            warmup=args.warmup,
            measurement_mode=args.measurement_mode,
            command_line=[sys.executable, *sys.argv],
        ),
    )

    if args.require_torch and not results["torch_validation"]["ok"]:
        raise SystemExit(f"torch validation failed: {results['torch_validation']['message']}")

    print(results)

    if args.results_out:
        save_json(Path(args.results_out), results)
    if args.csv_out:
        csv_path = Path(args.csv_out)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=CSV_FIELDNAMES,
            )
            writer.writeheader()
            writer.writerow(result_csv_row(results))
    if args.trace_out:
        if trace is not None:
            trace.dump(Path(args.trace_out))
    if args.trace_summary_out:
        save_json(Path(args.trace_summary_out), results["trace_summary"])
    if args.stage_metrics_out:
        save_json(Path(args.stage_metrics_out), results["stage_metrics"])
    if args.transfer_summary_out:
        save_json(Path(args.transfer_summary_out), results["transfer_summary"])
    if args.device_events_out:
        save_json(Path(args.device_events_out), results["device_events"])
    if args.npu_dev_report_out:
        save_json(Path(args.npu_dev_report_out), results["npu_development"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
