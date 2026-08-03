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

.PHONY: help compile clean

help:
	@echo "Transformer-layer AIE2P device kernels (compile-only)."
	@echo ""
	@echo "  make compile   Build every kernel object and check its symbols."
	@echo "  make clean     Remove $(BUILD_DIR)."
	@echo ""
	@echo "Variables: TILE_M=$(TILE_M) TILE_K=$(TILE_K) TILE_N=$(TILE_N)"
	@echo ""
	@echo "There is no 'run' target: this example builds device kernels, it"
	@echo "does not dispatch them. No NPU is required."

compile:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) \
	    python3 $(srcdir)/compile_kernels.py \
	        --tile-m $(TILE_M) --tile-k $(TILE_K) --tile-n $(TILE_N)

clean:
	rm -rf $(BUILD_DIR) __pycache__
