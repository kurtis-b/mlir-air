# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

BUILD_DIR := build_peano

# FFN matmul tile shape baked into encoder.cc / addnorm_ffn.cc at compile time.
# Constrained by the static_asserts in those sources: addnorm_ffn's up_proj
# needs TILE_N % 32 and its down_proj needs TILE_K % 32, which is stricter than
# encoder.cc's TILE_M % 16 / TILE_N % 16 / TILE_K % 8.
TILE_M ?= 64
TILE_K ?= 64
TILE_N ?= 64

.PHONY: help compile runlist-gate seam-tests clean

help:
	@echo "Transformer-layer AIE2P device kernels + the Phase B runtime seam."
	@echo ""
	@echo "  make compile        Build every kernel object and check its symbols."
	@echo "                      No NPU required; this is the PR-safe target."
	@echo "  make seam-tests     Host-only tests for BO pooling and runlist"
	@echo "                      aggregation. No NPU required."
	@echo "  make runlist-gate   Phase B hardware gate. NEEDS AN NPU. Wrap it:"
	@echo "                        flock -x -w 1800 /tmp/mlir-air-npu.lock make runlist-gate"
	@echo "  make clean          Remove $(BUILD_DIR)."
	@echo ""
	@echo "Variables: TILE_M=$(TILE_M) TILE_K=$(TILE_K) TILE_N=$(TILE_N)"
	@echo "           GATE_ARGS (e.g. --run-only to reuse cached ELFs)"

compile:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) \
	    python3 $(srcdir)/compile_kernels.py \
	        --tile-m $(TILE_M) --tile-k $(TILE_K) --tile-n $(TILE_N)

# Host-only. The seam's rules are unit-testable without hardware, deliberately.
seam-tests:
	cd $(srcdir)/../llms && python3 shared/infra/test_bo_pool.py
	cd $(srcdir)/../llms && python3 shared/infra/test_dispatch.py

# NPU required. Serialize on the repository lock, which is a DIFFERENT inode from
# the /tmp/npu.lock KernelCache takes internally -- both are BSD flock(2), so one
# inode would self-deadlock.
runlist-gate:
	cd $(srcdir) && python3 runlist_gate.py $(GATE_ARGS)

clean:
	rm -rf $(BUILD_DIR) __pycache__ runlist_gate_cache
