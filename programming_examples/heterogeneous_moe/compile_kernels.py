#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse

from compile import populate_artifacts
from manifest import load_json, project_dir, save_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compile heterogeneous MoE AIR kernels for NPU and/or GPU."
    )
    parser.add_argument(
        "--manifest",
        default="default_manifest.json",
        help="Manifest path relative to the heterogeneous_moe directory.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["npu", "gpu"],
        default=["npu", "gpu"],
        help="Backends to compile.",
    )
    parser.add_argument(
        "--manifest-out",
        default="artifacts/compiled_manifest.json",
        help="Output manifest path relative to the heterogeneous_moe directory.",
    )
    args = parser.parse_args(argv)

    manifest_path = (project_dir() / args.manifest).resolve()
    manifest_out_path = (project_dir() / args.manifest_out).resolve()
    manifest = load_json(manifest_path)
    manifest = populate_artifacts(manifest, set(args.backends))
    save_json(manifest_out_path, manifest)
    print(f"Wrote compiled manifest with artifacts: {manifest_out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
