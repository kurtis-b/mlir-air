# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Qwen2.5-1.5B BF16 Inference on NPU2 (MLIR-AIR). `make help` lists the targets;
# the body is shared with the other bf16 qwen examples in ../common.mk.

MODEL_TITLE    := Qwen2.5-1.5B
INFERENCE_PY   := qwen25_1_5b_inference.py
RUNNER_ADAPTER := qwen25_1_5b.verify_adapter

include $(dir $(realpath $(firstword $(MAKEFILE_LIST))))../common.mk
