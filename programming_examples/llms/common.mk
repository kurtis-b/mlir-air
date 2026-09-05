# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Shared Makefile body for the bf16 LLM examples.
#
# The five qwen bf16 model Makefiles were byte-identical once the directory name
# and the display title were substituted, so the body lives here once. A model
# Makefile sets three variables and includes this file:
#
#     MODEL_TITLE    := Qwen2.5-0.5B          # shown by `make help`
#     INFERENCE_PY   := qwen25_0_5b_inference.py
#     RUNNER_ADAPTER := qwen25_0_5b.verify_adapter
#     include $(dir $(realpath $(firstword $(MAKEFILE_LIST))))../common.mk
#
# `srcdir` below still resolves to the MODEL's directory, not this one:
# `$(firstword $(MAKEFILE_LIST))` is the including Makefile, and an include does
# not change it. Models needing extra targets (qwen3_0_6b's int4 decode axis)
# keep their own Makefile rather than overriding a recipe from here.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

PYTHON ?= $(if $(filter Windows_NT,$(OS)),python,python3)

# Build directory
ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

# Configurable parameters
N_TOKENS ?= 1000
PROMPT   ?= What is the capital of France?
MODEL    ?= instruct

.PHONY: help compile run profile chat verify verify-full diagnosis all clean

help:
	@echo "============================================================"
	@echo " $(MODEL_TITLE) Inference on NPU2 (MLIR-AIR)"
	@echo "============================================================"
	@echo ""
	@echo "Quick start:"
	@echo "  make compile          Compile all kernels (one-time, cached)"
	@echo "  make run              Run inference ($(N_TOKENS) tokens)"
	@echo "  make chat             Interactive chat REPL (streaming output)"
	@echo "  make profile          Run with profiling breakdown"
	@echo ""
	@echo "More targets:"
	@echo "  make verify           Top-k token-level inclusion gate vs HF bf16 (2 prompts × 32 tokens, k=5) — fast CI gate"
	@echo "  make verify-full      Same as above but runs the full prompt set (longer, exhaustive)"
	@echo "  make diagnosis        Per-layer ffn_out cosine + max_abs vs HF bf16 (single prompt, informational)"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean            Remove all build artifacts and verify reports"

## Compile all kernels
compile:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(srcdir)/$(INFERENCE_PY) --compile-only

## Run unified inference
run:
	cd $(BUILD_DIR) && $(PYTHON) $(srcdir)/$(INFERENCE_PY) \
		--run-only --n-tokens $(N_TOKENS) --prompt "$(PROMPT)" --model $(MODEL)

## Run with detailed profiling breakdown
profile:
	cd $(BUILD_DIR) && $(PYTHON) $(srcdir)/$(INFERENCE_PY) \
		--run-only --n-tokens $(N_TOKENS) --profile --prompt "$(PROMPT)" --model $(MODEL)

## Interactive chat
chat:
	cd $(BUILD_DIR) && $(PYTHON) $(srcdir)/$(INFERENCE_PY) \
		--run-only --interactive --n-tokens $(N_TOKENS) --model $(MODEL)

all: compile profile

VERIFY_RUNNER = $(srcdir)/../verify/verify_runner.py

## Top-k token-level inclusion gate (NPU vs HF bf16, 2 prompts × 32 tokens, k=5).
verify:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts topk_token --model $(MODEL) --max-prompts 2

## Full-sweep variant of `make verify`.
verify-full:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts topk_token --model $(MODEL)

## Diagnosis lens (per-layer ffn_out cosine vs HF bf16, single prompt, informational)
diagnosis:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts single --prompt "$(PROMPT)" --model $(MODEL)

## Remove all build artifacts and verify reports
clean:
	rm -r $(BUILD_DIR) 2>/dev/null || true
	rm -rf $(srcdir)/../verify/reports
	@echo "Build directory and verify/reports/ removed. Run 'make compile' to rebuild."
