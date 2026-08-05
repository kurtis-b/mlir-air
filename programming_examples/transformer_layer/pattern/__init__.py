# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The transformer-layer patterns: one shared golden model, then the strategies.

``reference.py`` is the correctness anchor for the whole block -- the tensor
draws (in iron's order, which is load-bearing), the layer's structure for both
workload variants, and every operator boundary it passes through. Phase E's
execution-strategy directories are built BESIDE it and import this one copy;
iron carries three 8-line ``reference.py`` re-exports of its shared model and
porting convention 8 deletes them rather than reproducing them.

Nothing here touches hardware or builds MLIR. The device side of a pattern is
``builders/block.py``.
"""
