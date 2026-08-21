# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The analytical planner (doc 56, H0): ModelGraph + Workload + DeviceCaps -> Plan.

Pure Python, importable without `air`. Reuses the study's leaf predicates
(profiles.skip_reason's bounds, the kernel registry) rather than generalizing
them into a compiler. See docs/plans/transformer-layer-execution-studies/56.
"""
from .graph import ModelSpec, Tensor, Node, ModelGraph, decoder_graph, QWEN3_0_6B, LLAMA32_1B  # noqa: F401
from .caps import DeviceCaps, NPU2_CAPS  # noqa: F401
from .placement import Workload, Placement, place, study_skip  # noqa: F401
from .plan import Plan, Stage, plan  # noqa: F401
