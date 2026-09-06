# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Qwen3-0.6B Inference on NPU2 (MLIR-AIR)
#
# `[2026-08-26]` DECODE PRECISION (doc 56 H2b, queue item 24): the decode O+FFN
# cascade runs **int4 (`w4_decode`) by default**. `QWEN3_W4_DECODE=0` selects
# the bf16 decode for A/B; that env is the one and only way to name the axis,
# for every target below (`compile`, `run`, `profile`, `verify`). Prefill, the
# QKV stage and the LM head are bf16 under both.
#
# Quick start:
#   make compile        — Compile all kernels (cached to disk)
#   make run            — Run inference: NPU prefill + NPU decode
#   make profile        — Run with per-kernel profiling breakdown

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

.PHONY: help compile run profile chat verify verify-full verify-quant-bar diagnosis check-lm-head-reexec all clean clean-build

help:
	@echo "============================================================"
	@echo " Qwen3-0.6B Inference on NPU2 (MLIR-AIR)"
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
	@echo "  make verify-quant-bar The int4 default's QUANTIZATION bar: the same gate, judged against the UNPATCHED HF bf16 checkpoint"
	@echo "  make diagnosis        Per-layer ffn_out cosine + max_abs vs HF bf16 (single prompt, informational)"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean            Remove all build artifacts and verify reports"
	@echo "  make clean-build      Remove THIS directory's build tree only (leaves ../verify/reports/ alone)"
	@echo ""
	@echo "Decode precision: int4 w4_decode by default; QWEN3_W4_DECODE=0 for bf16."

## Compile all kernels
compile:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(srcdir)/qwen3_0_6b_inference.py --compile-only

## Run unified inference
run:
	cd $(BUILD_DIR) && $(PYTHON) $(srcdir)/qwen3_0_6b_inference.py \
		--run-only --n-tokens $(N_TOKENS) --prompt "$(PROMPT)" --model $(MODEL)

## Run with detailed profiling breakdown
profile:
	cd $(BUILD_DIR) && $(PYTHON) $(srcdir)/qwen3_0_6b_inference.py \
		--run-only --n-tokens $(N_TOKENS) --profile --prompt "$(PROMPT)" --model $(MODEL)

## Interactive chat
chat:
	cd $(BUILD_DIR) && $(PYTHON) $(srcdir)/qwen3_0_6b_inference.py \
		--run-only --interactive --n-tokens $(N_TOKENS) --model $(MODEL)

all: compile profile

VERIFY_RUNNER = $(srcdir)/../verify/verify_runner.py
RUNNER_ADAPTER = qwen3_0_6b.verify_adapter

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

## `[2026-08-26]` queue item 24 — THE QUANTIZATION BAR for the w4_decode default.
##
## `make verify` judges the NPU against an oracle `verify_adapter.build_hf_model`
## patched with the SAME dequantized O+FFN weights the kernel consumes. That is
## the right gate for "did the NPU compute the model it was given", and it
## CANNOT see quantization error -- by construction, since both sides carry it.
## A default that ships quantization error needs the other half stated too, so
## this target reuses ONE w4 NPU capture and judges it a second time against the
## UNPATCHED HF bf16 checkpoint: the same top-5 token-set inclusion bar, k=5,
## 32 greedy tokens, that every bf16 default in this tree is held to.
##
## Mechanism, deliberately with no new flag: the capture phase runs with
## QWEN3_W4_DECODE=1 (int4 decode on the NPU) and the compare phase with
## QWEN3_W4_DECODE=0, which is exactly what makes `build_hf_model` return None
## and the framework load the plain checkpoint. Cost: one NPU capture + one HF
## reference decode; no extra device time beyond the capture.
verify-quant-bar:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && QWEN3_W4_DECODE=1 $(PYTHON) $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts topk_token --model $(MODEL) --max-prompts 2 \
		--gate-phase capture-npu --gate-file quant_bar_gate.json
	cd $(BUILD_DIR) && QWEN3_W4_DECODE=0 $(PYTHON) $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts topk_token --model $(MODEL) --max-prompts 2 \
		--gate-phase compare-hf --gate-file quant_bar_gate.json

## Diagnosis lens (per-layer ffn_out cosine vs HF bf16, single prompt, informational)
diagnosis:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts single --prompt "$(PROMPT)" --model $(MODEL)

## LM-head ELF back-to-back re-execution gate (no HF weights)
check-lm-head-reexec:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(srcdir)/lm_head_reexec_gate.py lm_head_reexec_cache

## Remove all build artifacts and verify reports
clean:
	rm -r $(BUILD_DIR) 2>/dev/null || true
	rm -rf $(srcdir)/../verify/reports
	@echo "Build directory and verify/reports/ removed. Run 'make compile' to rebuild."

## `[2026-08-26]` queue item 24: the cwd-scoped half of `clean`, for the lits.
## `clean` also deletes `$(srcdir)/../verify/reports/` -- a SOURCE-tree path
## that is not scoped to the caller's cwd and that other examples' evaluation
## reports cite by filename. A test that only needs to force a fresh compile of
## its own build tree must not take that with it.
clean-build:
	rm -r $(BUILD_DIR) 2>/dev/null || true
	@echo "$(BUILD_DIR)/ removed (verify/reports/ untouched). Run 'make compile' to rebuild."
