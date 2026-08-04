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

# Extra flags for opcheck.py, e.g. OPCHECK_ARGS="--fault-inject input".
OPCHECK_ARGS ?=

.PHONY: help compile runlist-gate seam-tests clean opcheck-list \
        check-elementwise-add check-causal-mask check-layer-norm check-addnorm

help:
	@echo "Transformer-layer AIE2P device kernels, the Phase B runtime seam, and"
	@echo "the Phase C operator numerical checks."
	@echo ""
	@echo "  make compile        Build every kernel object and check its symbols."
	@echo "                      No NPU required; this is the PR-safe target."
	@echo "  make seam-tests     Host-only tests for BO pooling and runlist"
	@echo "                      aggregation. No NPU required."
	@echo "  make opcheck-list   Every (operator, shape) Phase C claims, as JSON."
	@echo "                      No NPU required."
	@echo "  make runlist-gate   Phase B hardware gate. NEEDS AN NPU. Wrap it:"
	@echo "                        flock -x -w 1800 /tmp/mlir-air-npu.lock make runlist-gate"
	@echo "  make check-elementwise-add | check-causal-mask | check-layer-norm |"
	@echo "       check-addnorm  One operator's numerical check against its FP32"
	@echo "                      reference. NEEDS AN NPU; wrap it the same way."
	@echo "  make clean          Remove $(BUILD_DIR)."
	@echo ""
	@echo "Variables: TILE_M=$(TILE_M) TILE_K=$(TILE_K) TILE_N=$(TILE_N)"
	@echo "           GATE_ARGS (e.g. --run-only to reuse cached ELFs)"
	@echo "           OPCHECK_ARGS (e.g. --fault-inject input, which must FAIL)"

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

# Host-only: just enumerates what the operator checks below claim.
opcheck-list:
	python3 $(srcdir)/opcheck.py --list

# NPU required, same lock as runlist-gate. Run from $(BUILD_DIR) rather than
# from $(srcdir): opcheck.py compiles its external kernel objects into the
# CURRENT directory, which is where aiecc's link_with search looks, and the
# source tree is not the place for them. Results always land next to
# opcheck.py regardless of where it is run from.
define opcheck
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) \
	    python3 $(srcdir)/opcheck.py --operator $(1) $(OPCHECK_ARGS)
endef

check-elementwise-add:
	$(call opcheck,elementwise_add)

check-causal-mask:
	$(call opcheck,causal_mask)

check-layer-norm:
	$(call opcheck,layer_norm)

check-addnorm:
	$(call opcheck,addnorm)

# The negative control, with its exit status DELIBERATELY INVERTED: it succeeds
# only when the injected run fails. A check that ignored --fault-inject, or one
# whose tolerance were wide enough to swallow a perturbed input, would still
# report PASS here -- and this target would then fail, which is the point.
# One operator, because the injection machinery is shared by all four and the
# driver runs it against each of them directly.
check-fault-control:
	mkdir -p $(BUILD_DIR)
	if cd $(BUILD_DIR) && PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) \
	       python3 $(srcdir)/opcheck.py --operator elementwise_add \
	           --fault-inject input; then \
	    echo "NEGATIVE CONTROL: FAIL (injected run passed, so the check is not"; \
	    echo "  reading the buffer the device was actually given)"; \
	    exit 1; \
	else \
	    echo "NEGATIVE CONTROL: PASS (injected run failed, as required)"; \
	fi

clean:
	rm -rf $(BUILD_DIR) __pycache__ runlist_gate_cache
