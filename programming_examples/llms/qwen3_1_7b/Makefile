# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Qwen3-1.7B BF16 Inference on NPU2 (MLIR-AIR). `make help` lists the targets;
# the body is shared with the other bf16 qwen examples in ../common.mk.

MODEL_TITLE    := Qwen3-1.7B
INFERENCE_PY   := qwen3_1_7b_inference.py
RUNNER_ADAPTER := qwen3_1_7b.verify_adapter

include $(dir $(realpath $(firstword $(MAKEFILE_LIST))))../common.mk
