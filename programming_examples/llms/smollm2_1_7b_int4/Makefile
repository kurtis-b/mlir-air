# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# SmolLM2-1.7B int4 (GGUF q4_0) Inference on NPU2 (MLIR-AIR)
#
# Shipped shape: bf16 NPU prefill on the q4_0-dequantized weights + int4 NPU
# decode on the packed BOs (gs=32). The whole model is the quantized model.
#
# Quick start:
#   make compile        — Compile all kernels (cached to disk)
#   make run            — Run inference: NPU prefill + int4 NPU decode
#   make verify         — Top-k token-set gate vs the PLAIN HF bf16 model

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

# Configurable parameters
N_TOKENS ?= 1000
PROMPT   ?= What is the capital of France?
GGUF     ?= /home/cj/models/gguf/SmolLM2-1.7B-Instruct-Q4_0.gguf
export SMOLLM2_GGUF := $(GGUF)

.PHONY: help compile run profile chat verify verify-full diagnosis all clean

help:
	@echo "============================================================"
	@echo " SmolLM2-1.7B int4 (GGUF q4_0) Inference on NPU2 (MLIR-AIR)"
	@echo "============================================================"
	@echo ""
	@echo "Quick start:"
	@echo "  make compile          Compile all kernels (one-time, cached)"
	@echo "  make run              Run inference ($(N_TOKENS) tokens)"
	@echo "  make chat             Interactive chat REPL (streaming output)"
	@echo "  make profile          Run with profiling breakdown"
	@echo ""
	@echo "More targets:"
	@echo "  make verify           Top-k token-set inclusion gate vs the PLAIN HF bf16"
	@echo "                        model (2 prompts x 32 tokens, k=5). The delta"
	@echo "                        deliberately includes q4_0 quantization error."
	@echo "  make verify-full      Same, full prompt set"
	@echo "  make diagnosis        Single-prompt lens (informational)"
	@echo ""
	@echo "Options (override with make VAR=value):"
	@echo "  N_TOKENS=1000         Max decode tokens for run/profile/chat"
	@echo "  PROMPT=\"...\"          Input prompt text"
	@echo "  GGUF=path             The Q4_0 GGUF checkpoint (default: $(GGUF))"

compile:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(srcdir)/smollm2_1_7b_int4_inference.py --compile-only

run:
	cd $(BUILD_DIR) && python3 $(srcdir)/smollm2_1_7b_int4_inference.py \
		--run-only --n-tokens $(N_TOKENS) --prompt "$(PROMPT)" --gguf "$(GGUF)"

profile:
	cd $(BUILD_DIR) && python3 $(srcdir)/smollm2_1_7b_int4_inference.py \
		--run-only --n-tokens $(N_TOKENS) --profile --prompt "$(PROMPT)" --gguf "$(GGUF)"

chat:
	cd $(BUILD_DIR) && python3 $(srcdir)/smollm2_1_7b_int4_inference.py \
		--run-only --interactive --n-tokens $(N_TOKENS) --gguf "$(GGUF)"

all: compile profile

# The shared verify framework lives under programming_examples/llms/verify/.
VERIFY_RUNNER = $(srcdir)/../verify/verify_runner.py
RUNNER_ADAPTER = smollm2_1_7b_int4.verify_adapter

## Top-k token-set inclusion gate (NPU int4 vs PLAIN HF bf16, 2 prompts x 32 tokens, k=5).
verify:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts topk_token --model instruct --max-prompts 2

verify-full:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts topk_token --model instruct

diagnosis:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts single --prompt "$(PROMPT)" --model instruct

clean:
	rm -rf build_peano build_chess verify_reports
