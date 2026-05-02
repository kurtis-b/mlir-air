#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

from __future__ import annotations

import tempfile
from pathlib import Path

from kernels import KernelConfig, default_air_filenames, write_default_air_sources
from manifest import project_dir


def _check(cfg: KernelConfig) -> list[str]:
    mismatches: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        generated = write_default_air_sources(cfg, Path(tmp))
        names = default_air_filenames(cfg)
        for key, name in names.items():
            golden = project_dir() / "air" / name
            if not golden.exists():
                mismatches.append(f"missing golden: {golden}")
                continue
            if generated[key].read_text(encoding="utf-8") != golden.read_text(encoding="utf-8"):
                mismatches.append(f"stale golden: {golden}")
    return mismatches


def main(argv: list[str] | None = None) -> int:
    del argv
    configs = [
        KernelConfig(batch_tokens=4, hidden_size=16, ffn_size=32, dtype="bf16"),
    ]
    mismatches: list[str] = []
    for cfg in configs:
        mismatches.extend(_check(cfg))
    if mismatches:
        for mismatch in mismatches:
            print(mismatch)
        return 1
    print("Golden AIR files match regenerated default sources.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
