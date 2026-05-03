#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse

from llm_linear.compile import populate_artifacts, populate_direct_gpu_artifacts
from llm_linear.manifest import load_json, resolve_package_path, save_json
from llm_linear.schema import validate_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compile AIR GPU/NPU artifacts for the llm_linear benchmark."
    )
    parser.add_argument(
        "--manifest",
        default="default_linear_manifest.json",
        help="Manifest path relative to llm_linear/.",
    )
    parser.add_argument(
        "--manifest-out",
        default="artifacts/compiled_linear_manifest.json",
        help="Output manifest path relative to llm_linear/.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["gpu", "npu"],
        default=["gpu"],
        help="Backends to compile.",
    )
    parser.add_argument(
        "--direct-gpu",
        action="store_true",
        help="Also compile device-resident GPU artifacts without host staging.",
    )
    args = parser.parse_args(argv)

    manifest = load_json(resolve_package_path(args.manifest))
    validate_manifest(manifest)
    compiled = populate_artifacts(manifest, set(args.backends))
    if args.direct_gpu:
        compiled = populate_direct_gpu_artifacts(compiled)
    out_path = resolve_package_path(args.manifest_out)
    save_json(out_path, compiled)
    print(f"Wrote compiled llm_linear manifest to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
