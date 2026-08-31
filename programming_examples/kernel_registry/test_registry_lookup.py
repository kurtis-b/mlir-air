# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Host regression for the registry's per-row herd (review of #38): both
lookup functions return the effective herd (per-method override first,
file-level 8x4 fallback), and the canonical builder recipe
(gemm_registry_config) threads it into the spec instead of dropping it.

    python3 test_registry_lookup.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PE = os.path.dirname(_HERE)
for _p in (_PE, os.path.join(_PE, "llms")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from kernel_registry.registry_lookup import (
    gemm_config,
    gemm_config_method,
)  # noqa: E402


def test_override_row_returns_its_herd():
    cfg = gemm_config(64, 768, 2304)  # short-M sweep row, per-row herd
    assert tuple(cfg["herd"]) == (2, 4), cfg["herd"]
    cfg = gemm_config_method(64, 768, 2304, "bf16", "direct")
    assert tuple(cfg["herd"]) == (1, 4), cfg["herd"]
    # review of #40, P2: the ffn_down short-M key, both tiers
    cfg = gemm_config(64, 3072, 768)
    assert tuple(cfg["herd"]) == (2, 4), cfg["herd"]  # high tier -> drain
    cfg = gemm_config(64, 3072, 768, "bf16", "low")
    assert tuple(cfg["herd"]) == (1, 4), cfg["herd"]  # low tier -> direct
    # review of #41, P2: the o_proj short-M key, both tiers
    cfg = gemm_config(64, 768, 768)
    assert tuple(cfg["herd"]) == (2, 4), cfg["herd"]  # high tier -> drain
    cfg = gemm_config(64, 768, 768, "bf16", "low")
    assert tuple(cfg["herd"]) == (1, 4), cfg["herd"]  # low tier -> direct
    # review of #44, P2: the baseline_512 ffn_up short-M key, both tiers
    cfg = gemm_config(64, 512, 2048)
    assert tuple(cfg["herd"]) == (2, 4), cfg["herd"]  # high tier -> drain
    cfg = gemm_config(64, 512, 2048, "bf16", "low")
    assert tuple(cfg["herd"]) == (1, 4), cfg["herd"]  # low tier -> direct
    # review of #45, P2: the baseline_512 o_proj short-M key, both tiers
    cfg = gemm_config(64, 512, 512)
    assert tuple(cfg["herd"]) == (2, 4), cfg["herd"]  # high tier -> drain
    cfg = gemm_config(64, 512, 512, "bf16", "low")
    assert tuple(cfg["herd"]) == (1, 4), cfg["herd"]  # low tier -> direct
    # review of #46, P2: the baseline_1024 qkv short-M key, both tiers
    cfg = gemm_config(64, 1024, 3072)
    assert tuple(cfg["herd"]) == (2, 4), cfg["herd"]  # high tier -> drain
    cfg = gemm_config(64, 1024, 3072, "bf16", "low")
    assert tuple(cfg["herd"]) == (1, 4), cfg["herd"]  # low tier -> direct
    # review of #47, P2: the baseline_1024 ffn_up short-M key, both tiers
    cfg = gemm_config(64, 1024, 4096)
    assert tuple(cfg["herd"]) == (2, 4), cfg["herd"]  # high tier -> drain
    cfg = gemm_config(64, 1024, 4096, "bf16", "low")
    assert tuple(cfg["herd"]) == (1, 4), cfg["herd"]  # low tier -> direct


def test_default_row_falls_back_to_file_level_herd():
    cfg = gemm_config(2048, 2048, 2048)  # a pre-series main row, no override
    assert tuple(cfg["herd"]) == (8, 4), cfg["herd"]
    cfg = gemm_config_method(2048, 2048, 2048, "bf16", cfg["method"])
    assert tuple(cfg["herd"]) == (8, 4), cfg["herd"]


def test_registry_config_threads_the_herd_into_the_spec():
    from shared.builders.gemm_builder import gemm_registry_config

    spec = gemm_registry_config(64, 768, 2304, "bf16", "high")
    assert (spec.get("herd_m"), spec.get("herd_n")) == (2, 4), (
        "the canonical recipe dropped the per-row herd -- a short-M build "
        "through it would trip M % (tile_m * herd_m)"
    )
    spec = gemm_registry_config(2048, 2048, 2048, "bf16", "high")
    assert (spec.get("herd_m"), spec.get("herd_n")) == (8, 4), (
        "a default row must carry the file-level 8x4 -- same value the "
        "builders' own params default to"
    )


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"registry lookup tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
