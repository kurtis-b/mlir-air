#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import subprocess
import sys

from manifest import load_json, project_dir, save_json
from reports import build_edge_study_summary, edge_study_markdown

PROFILE_SUITES = {
    "smoke": ["shape_sweep"],
    "routing": ["shape_sweep", "routing_sweep"],
    "model": ["model_presets"],
    "full": ["shape_sweep", "routing_sweep", "model_presets"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the canonical heterogeneous MoE edge-efficiency study.")
    parser.add_argument("--manifest", default="default_manifest.json", help="Base manifest relative to this directory.")
    parser.add_argument("--matrix", default="default_benchmark_matrix.json", help="Benchmark matrix relative to this directory.")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_SUITES),
        default="routing",
        help="Study breadth. Use full for the complete shape/routing/model suite.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/benchmarks/edge_study/latest",
        help="Output directory relative to this directory.",
    )
    parser.add_argument("--iterations", type=int, default=None, help="Override timed iterations for every case.")
    parser.add_argument("--warmup", type=int, default=None, help="Override warmup iterations for every case.")
    parser.add_argument(
        "--measurement-mode",
        choices=["cold", "warm", "both"],
        default="both",
        help="Measure cold start, warm steady-state, or both.",
    )
    parser.add_argument("--allow-npu", action="store_true", help="Run NPU-tagged cases.")
    parser.add_argument("--require-torch", action="store_true", help="Fail if torch validation is unavailable or fails.")
    parser.add_argument("--workload-filter", nargs="*", default=[], help="Forwarded workload name filters.")
    parser.add_argument("--case-filter", nargs="*", default=[], help="Forwarded exact case-name filters.")
    args = parser.parse_args()

    root = project_dir()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    suite_output = output_dir / "suite"
    cmd = [
        sys.executable,
        "run_workload_suite.py",
        "--manifest",
        args.manifest,
        "--matrix",
        args.matrix,
        "--suite",
        *PROFILE_SUITES[args.profile],
        "--output-dir",
        str(suite_output),
        "--measurement-mode",
        args.measurement_mode,
    ]
    if args.iterations is not None:
        cmd.extend(["--iterations", str(args.iterations)])
    if args.warmup is not None:
        cmd.extend(["--warmup", str(args.warmup)])
    if args.allow_npu:
        cmd.append("--allow-npu")
    if args.require_torch:
        cmd.append("--require-torch")
    if args.workload_filter:
        cmd.append("--workload-filter")
        cmd.extend(args.workload_filter)
    if args.case_filter:
        cmd.append("--case-filter")
        cmd.extend(args.case_filter)

    completed = subprocess.run(cmd, cwd=root, check=False)
    if completed.returncode != 0:
        return completed.returncode

    suite_summary = load_json(suite_output / "summary.json")
    study_summary = build_edge_study_summary(suite_summary, [sys.executable, *sys.argv])
    save_json(output_dir / "edge_efficiency_summary.json", study_summary)
    (output_dir / "edge_efficiency_report.md").write_text(edge_study_markdown(study_summary), encoding="utf-8")
    print(f"Wrote edge-efficiency study outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
