# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Llama-3.2-1B's LM-head re-execution gate.

Same gate as Qwen3-0.6B's, on this model's head: one kernel compiled fresh
from the production builder and dispatched seven times in the patterns that
expose the back-to-back defect. The history, the LOAD_PDI parity mechanism
and the control are documented in `shared/infra/lm_head_reexec.py` and
`qwen3_0_6b/run_npu2_lm_head_reexec.lit`.

Two things differ from Qwen3-0.6B and both are deliberate:

* The partitioning is 8 x 16384, equal, so there is no `_LM_PARTS` to import
  -- the list is the shorthand expanded. It is still built from the decode
  module's own constants rather than written out here, because a list that
  disagrees with the module's partitioning produces garbage rather than an
  error (it did, on Qwen: devq 939).
* The backend is `LM_GEMV_BACKEND`, which is what this model's decode
  compiles the head with. That matters more than it looks: it carries
  `runtime_loop_tiling_sizes`, and the DMA repeat count depends on it. The
  same 16384-row partitions compile under it at `m_input` 4 but FAIL under
  Qwen's untiled `_lm_gemv_backend()` with "Repeat count exceeds the [0:255]
  range" (devq 944 against 945/946). Passing the wrong backend here would not
  be a slower gate, it would be a different kernel -- which is also why the
  module under test comes from the decode module's own production builder
  rather than being rebuilt here.

Device gate, not collected by any CI target: `llms/` is filtered out of
`check-programming-examples-peano`. Run it through the device scheduler:

    agents/scripts/devq.sh run --class measure -- \\
      bash -c "source agents/.state/tlenv.sh && \\
               python3 programming_examples/llms/llama32_1b/lm_head_reexec_gate.py"

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

from shared.infra.backend_presets import LM_GEMV_BACKEND  # noqa: E402
from shared.infra.lm_head_reexec import run_gate  # noqa: E402
from llama32_1b_decode import build_lm_head_gemv_llama_module  # noqa: E402
from llama32_1b_inference import _LM_N_PART, _LM_N_PARTITIONS  # noqa: E402

EMB = 2048

# The decode module's own production builder, so a change to the head's
# geometry or `m_input` reaches this gate without anyone remembering to update
# it -- the same reason the partition list is read from the model's constants.
LM_PARTS = [_LM_N_PART] * _LM_N_PARTITIONS


def main():
    cache_dir = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd() / "lm_head_reexec_cache"
    )
    return run_gate(
        cache_dir,
        lambda: build_lm_head_gemv_llama_module(EMB),
        LM_PARTS,
        dict(LM_GEMV_BACKEND),
        EMB,
        tag="llama32_1b lm_head_gemv",
    )


if __name__ == "__main__":
    sys.exit(main())
