#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import time
from pathlib import Path

from reference import random_inputs
from runtime import _load_json, _project_dir, _save_json, load_runtime, update_manifest_backends


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the fixed-placement heterogeneous MoE benchmark.")
    parser.add_argument("--manifest", default="default_manifest.json", help="Manifest path relative to this directory.")
    parser.add_argument("--prepare", action="store_true", help="Compile and load any selected non-CPU executors, then exit.")
    parser.add_argument("--iterations", type=int, default=None, help="Override iteration count from the manifest.")
    parser.add_argument("--warmup", type=int, default=None, help="Override warmup count from the manifest.")
    parser.add_argument("--router-mode", choices=["top1", "top2"], default=None)
    parser.add_argument("--router-backend", choices=["cpu", "npu", "gpu"], default=None)
    parser.add_argument("--expert0-backend", choices=["cpu", "npu", "gpu"], default=None)
    parser.add_argument("--expert1-backend", choices=["cpu", "npu", "gpu"], default=None)
    parser.add_argument("--aggregation-backend", choices=["cpu", "npu", "gpu"], default=None)
    parser.add_argument("--transfer-mode", choices=["host", "peer", "auto"], default=None)
    parser.add_argument("--results-out", default=None, help="Optional JSON file for benchmark results.")
    parser.add_argument("--trace-out", default=None, help="Optional Chrome-trace JSON output.")
    args = parser.parse_args()

    manifest_path = (_project_dir() / args.manifest).resolve()
    manifest = _load_json(manifest_path)
    manifest = update_manifest_backends(
        manifest,
        router_backend=args.router_backend,
        expert0_backend=args.expert0_backend,
        expert1_backend=args.expert1_backend,
        aggregation_backend=args.aggregation_backend,
        transfer_mode=args.transfer_mode,
        router_mode=args.router_mode,
    )

    runtime = load_runtime(manifest_path)
    runtime.manifest = manifest
    runtime.transfer.mode = manifest["runtime"]["transfer_mode"]
    runtime.executors = runtime._make_executors()

    if args.prepare:
        runtime.prepare()
        print("Prepared selected executors.")
        return 0

    iterations = args.iterations or manifest["benchmark"]["iterations"]
    warmup = args.warmup if args.warmup is not None else manifest["benchmark"]["warmup"]
    inputs = random_inputs(runtime.cfg, manifest["inputs"]["seed"])

    for _ in range(warmup):
        runtime.run(inputs, router_mode=manifest["runtime"]["router_mode"])

    latencies_ms = []
    last_run = None
    for _ in range(iterations):
        start = time.perf_counter_ns()
        last_run = runtime.run(inputs, router_mode=manifest["runtime"]["router_mode"])
        end = time.perf_counter_ns()
        latencies_ms.append((end - start) / 1_000_000.0)

    assert last_run is not None
    results = {
        "router_mode": manifest["runtime"]["router_mode"],
        "stage_backends": manifest["runtime"]["stage_backends"],
        "transfer_mode": manifest["runtime"]["transfer_mode"],
        "iterations": iterations,
        "warmup": warmup,
        "latency_ms": {
            "mean": sum(latencies_ms) / len(latencies_ms),
            "min": min(latencies_ms),
            "max": max(latencies_ms),
        },
        "max_abs_error": last_run["max_abs_error"],
        "torch_validation": last_run["torch_validation"],
    }

    print(results)

    if args.results_out:
        _save_json(Path(args.results_out), results)
    if args.trace_out:
        last_run["trace"].dump(Path(args.trace_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

