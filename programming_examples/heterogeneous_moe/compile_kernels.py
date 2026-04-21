#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
from pathlib import Path

from runtime import _load_json, _project_dir, _save_json, populate_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile heterogeneous MoE AIR kernels for NPU and/or GPU.")
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
    args = parser.parse_args()

    manifest_path = (_project_dir() / args.manifest).resolve()
    manifest = _load_json(manifest_path)
    manifest = populate_artifacts(manifest, set(args.backends))
    _save_json(manifest_path, manifest)
    print(f"Updated manifest with compiled artifacts: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

