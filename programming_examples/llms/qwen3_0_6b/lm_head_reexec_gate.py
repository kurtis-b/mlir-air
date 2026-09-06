# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The LM-head GEMV's RE-EXECUTION gate: the production ELF, dispatched back-to-back.

The research branch (doc 57 section 1.4, devq 446-448, confirmed on the
production artifact at devq 452) found that the 19 x 8192 `lm_head_gemv`
multi-launch ELF returned WRONG, NON-DETERMINISTIC values when it executed
immediately after itself: partition 0 (the first launch), ~700-2,800 of its
8192 rows at tile boundaries, max error ~0.9 of the output scale, different on
every repeat. It was exact on the first dispatch after load, exact after ANY
other ELF ran in between, and wrong again on the very next back-to-back
dispatch. A 0.5 s host idle did not heal it, so it was stale partition/context
state rather than a race with in-flight work.

It is invisible to the shipped gates: production decode never runs the head
twice in a row (an `o_gemv_ffn` always precedes it), and `make verify` judges
token sets rather than logits. Any batched-logits, re-scoring or per-token
runlist design that makes the head adjacent to itself would meet it.

The partition list comes from the decode module (`_LM_PARTS`), so the gate
always measures the geometry the model actually ships.

Device gate, not collected by any CI target: `llms/` is filtered out of
`check-programming-examples-peano`. Run it through the device scheduler:

    agents/scripts/devq.sh run --class measure -- \\
      bash -c "source agents/.state/tlenv.sh && \\
               python3 programming_examples/llms/qwen3_0_6b/lm_head_reexec_gate.py"

Usage: python3 lm_head_reexec_gate.py [CACHE_DIR]   (default ./lm_head_reexec_cache)
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (
    _HERE,
    _HERE.parent,
    _HERE.parent.parent / "matrix_vector_multiplication" / "bf16",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from shared.infra.lm_head_reexec import run_gate  # noqa: E402
from qwen3_0_6b_decode import (  # noqa: E402
    _LM_PARTS,
    _lm_gemv_backend,
    build_lm_head_gemv_qwen_module,
)

EMB = 1024

# Imported, never reconstructed. The gate slices the weight matrix and shapes
# the output buffers per partition, so a list that disagrees with the module's
# own partitioning produces garbage rather than an error: reconstructing it as
# `[_LM_N_PART] * _LM_N_PARTITIONS` read 10 x 16384 against a module built as
# 9 x 16384 + 4480 and reported 0/7 with every failure in partition 9. One
# source of truth, the decode module's.
LM_PARTS = _LM_PARTS


def main():
    cache_dir = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd() / "lm_head_reexec_cache"
    )
    return run_gate(
        cache_dir,
        lambda: build_lm_head_gemv_qwen_module(EMB),
        LM_PARTS,
        _lm_gemv_backend(),
        EMB,
    )


if __name__ == "__main__":
    sys.exit(main())
