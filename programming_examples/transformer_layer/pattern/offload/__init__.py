# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``offload`` execution strategy: host-owned layer, one GEMM per dispatch."""

from pattern.offload.offload import (  # noqa: F401
    ATTENTION_PATH,
    OFFLOAD_CACHE_DIR,
    OFFLOAD_GEMMS,
    OFFLOAD_INPUT_NAMES,
    offload_config,
    prepare_offload,
)
