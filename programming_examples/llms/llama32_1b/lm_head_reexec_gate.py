# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Llama-3.2-1B's LM-head re-execution gate (shared/infra/lm_head_reexec.py):
the production 8 x 16384 head at m_input 8 (`[2026-08-23]`, queue item 12),
dispatched seven times, 7/7 clean required. `make check-lm-head-reexec`.

Usage: python3 lm_head_reexec_gate.py [CACHE_DIR]   (default ./lm_head_reexec_cache)
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent, _HERE.parent.parent / "matrix_vector_multiplication" / "bf16"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from shared.infra.backend_presets import LM_GEMV_BACKEND  # noqa: E402
from shared.infra.lm_head_reexec import run_gate  # noqa: E402
from llama32_1b_decode import build_lm_head_gemv_llama_module  # noqa: E402
from llama32_1b_inference import _LM_N_PART, _LM_N_PARTITIONS  # noqa: E402

EMB = 2048


def main():
    cache_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd() / "lm_head_reexec_cache"
    return run_gate(cache_dir, lambda: build_lm_head_gemv_llama_module(EMB),
                    [_LM_N_PART] * _LM_N_PARTITIONS, LM_GEMV_BACKEND, EMB)


if __name__ == "__main__":
    sys.exit(main())
