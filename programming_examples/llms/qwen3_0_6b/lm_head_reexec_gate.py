# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The LM-head GEMV's RE-EXECUTION gate: the production ELF, dispatched back-to-back.

`[2026-08-21]` Doc 57 section 1.4 found (devq 446-448) and this gate confirms
on the production artifact (devq 452) that the 19 x 8192 `lm_head_gemv`
multi-launch ELF returns WRONG, NON-DETERMINISTIC values when it executes
immediately after itself: partition 0 (the first launch), ~700-2,800 of its
8192 rows (rows 24, 32, 40, 64.. -- tile boundaries), max error ~0.9 of the
output scale, different on every repeat. It is exact on the first dispatch
after load, exact after ANY other ELF has run in between, and wrong again on
the very next back-to-back dispatch. A 0.5 s host idle between dispatches
does NOT heal it, so it is stale partition/context state, not a race with
in-flight work. Production decode never runs the head twice in a row (an
`o_gemv_ffn` always precedes it) and the verify gate judges token sets, not
logits, so `make verify` cannot see it; any batched-logits, re-scoring or
per-token runlist design that makes the head adjacent to itself will.

Shape of `study/fused_reexec_gate.py`: ONE kernel, compiled fresh from the
production builder (`build_lm_head_gemv_qwen_module`, so a fix anywhere
between builder and device is what flips this), dispatched seven times in the
patterns above, every dispatch judged fail-closed against an f32 reference
(clean = finite, no element beyond 1e-2 of the output scale -- the honest
bf16 error is 2.5e-3 -- and bit-identical to the first dispatch for the same
input). PASS requires 7/7.

`[2026-08-21, 03:40]` With the production head re-partitioned to
9 x 16384 + 4480 (doc 57 section 5 item 5b) this gate reads 7/7 clean (devq
482, 484): the shipped artifact no longer shows the defect, the lit's XFAIL is
gone, and the gate guards against the family's return. The defect is avoided,
not understood -- doc 57 section 1.5's table has the seven configurations.

Usage: python3 lm_head_reexec_gate.py [CACHE_DIR]   (default ./lm_head_reexec_cache)
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent, _HERE.parent.parent / "matrix_vector_multiplication" / "bf16"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from shared.infra.lm_head_reexec import run_gate  # noqa: E402
from qwen3_0_6b_decode import (  # noqa: E402
    _LM_PARTS,
    _lm_gemv_backend,
    build_lm_head_gemv_qwen_module,
)

EMB = 1024


def main():
    cache_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd() / "lm_head_reexec_cache"
    return run_gate(cache_dir, lambda: build_lm_head_gemv_qwen_module(EMB), _LM_PARTS,
                    _lm_gemv_backend(), EMB)


if __name__ == "__main__":
    sys.exit(main())
