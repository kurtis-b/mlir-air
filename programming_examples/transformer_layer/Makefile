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

# Which width of the case matrix `make registry-sweep` measures. baseline_768 is
# the family Phase D needs; the other two are the same tool over a different id.
FAMILY ?= baseline_768

# Extra flags for registry_sweep.py, e.g. SWEEP_ARGS="--role o_proj --seq 64".
SWEEP_ARGS ?=

.PHONY: help compile runlist-gate seam-tests clean opcheck-list \
        check-elementwise-add check-causal-mask check-layer-norm check-addnorm \
        check-qkv-proj check-ffn check-mha-out-proj check-fault-control \
        check-block check-block-fault reference-tests block-cache-tests \
        registry-resolution registry-sweep registry-write registry-plan \
        registry-writer-tests

help:
	@echo "Transformer-layer AIE2P device kernels, the Phase B runtime seam, and"
	@echo "the Phase C operator numerical checks."
	@echo ""
	@echo "  make compile        Build every kernel object and check its symbols."
	@echo "                      No NPU required; this is the PR-safe target."
	@echo "  make seam-tests     Host-only tests for BO pooling and runlist"
	@echo "                      aggregation. No NPU required."
	@echo "  make reference-tests"
	@echo "                      Host-only checks that the transformer-layer"
	@echo "                      golden model is the layer it claims to be."
	@echo "                      No NPU required."
	@echo "  make block-cache-tests"
	@echo "                      Host-only checks on which cached block ELFs"
	@echo "                      check-block will reuse and which it rebuilds."
	@echo "                      No NPU required."
	@echo "  make opcheck-list   Every (operator, shape) Phase C claims, as JSON."
	@echo "                      No NPU required."
	@echo "  make runlist-gate   Phase B hardware gate. NEEDS AN NPU. Wrap it:"
	@echo "                        flock -x -w 1800 /tmp/mlir-air-npu.lock make runlist-gate"
	@echo "  make check-elementwise-add | check-causal-mask | check-layer-norm |"
	@echo "       check-addnorm | check-qkv-proj | check-ffn | check-mha-out-proj"
	@echo "                      One operator's numerical check against its FP32"
	@echo "                      reference. NEEDS AN NPU; wrap it the same way."
	@echo "                      qkv-proj, ffn and mha-out-proj take their GEMM"
	@echo "                      tiles and method from kernel_registry, which"
	@echo "                      RAISES on a shape nobody swept rather than"
	@echo "                      guessing."
	@echo "  make check-block    The whole encoder_bert layer: five operator"
	@echo "                      launches over four ELFs, against the golden"
	@echo "                      model, with every boundary compared. NEEDS AN"
	@echo "                      NPU; wrap it the same way."
	@echo "  make check-block-fault"
	@echo "                      That layer's negative control. It must FAIL"
	@echo "                      under injection, and this target succeeds"
	@echo "                      exactly when it does."
	@echo "  make check-ffn-ladder-fault"
	@echo "                      The seq-64 ladder point's own negative control."
	@echo "                      The shared one injects an operator's FIRST"
	@echo "                      shape, which for ffn is a Phase C row, so"
	@echo "                      without this the newest point is the only one"
	@echo "                      never injected. NEEDS AN NPU."
	@echo "  make registry-plan  Every (shape, candidate) the sweep would measure."
	@echo "                      No NPU required."
	@echo "  make registry-resolution"
	@echo "                      Assert every $(FAMILY) shape resolves through the"
	@echo "                      operator builder that owns it, not just through"
	@echo "                      the registry -- a builder that pins a method needs"
	@echo "                      that method's row, not the row's winner. No NPU"
	@echo "                      required; this is what the lit test runs."
	@echo "  make registry-sweep Measure the tile configs for $(FAMILY) and"
	@echo "                      checkpoint them. NEEDS AN NPU, takes about an"
	@echo "                      hour, resumable. Wrap it:"
	@echo "                        flock -x -w 1800 /tmp/mlir-air-npu.lock make registry-sweep"
	@echo "  make registry-write Fold the swept results into the registry JSON and"
	@echo "                      both markdown pages. Append-only. No NPU required."
	@echo "  make registry-writer-tests"
	@echo "                      Host-only tests for what the writer will and will"
	@echo "                      not do to an existing shape. No NPU required."
	@echo "  make clean          Remove $(BUILD_DIR)."
	@echo ""
	@echo "Variables: TILE_M=$(TILE_M) TILE_K=$(TILE_K) TILE_N=$(TILE_N)"
	@echo "           GATE_ARGS (e.g. --run-only to reuse cached ELFs)"
	@echo "           OPCHECK_ARGS (e.g. --fault-inject input, which must FAIL)"
	@echo "           FAMILY=$(FAMILY), SWEEP_ARGS (e.g. --role o_proj --seq 64)"

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

# The two GEMM-backed operators. Their tile sizes and method come from
# kernel_registry per shape, so there is nothing to set here; the registry
# raises rather than guessing if the shape was never swept.
check-qkv-proj:
	$(call opcheck,qkv_proj)

check-ffn:
	$(call opcheck,ffn)

# The Phase E1 ladder point's OWN negative control, and it needs its own target
# rather than riding on check-fault-control.
#
# check-fault-control injects into one operator, because the injection machinery
# is shared; the driver injects into each operator's FIRST declared shape. For
# `ffn` that first shape is a Phase C row (2048x1024x3072), so neither of them
# ever perturbs 64x768x3072 -- the one point on the sequence ladder this
# sub-phase added, and the only one whose two GEMMs are both `drain`. It would be
# the single shape in the file with no injection of its own. D1 recorded that
# exact gap for the baseline_768 rows; this is it not being repeated.
#
# Cheap enough to be worth doing here rather than only in the driver: seq 64 is
# the shortest point on the ladder, so the extra ELF compile and dispatch are
# seconds, and the coverage then lives in the suite where it survives a driver
# change.
check-ffn-ladder-fault:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) \
	    python3 $(srcdir)/opcheck.py --operator ffn --shape-key 64x768x3072 \
	        --fault-inject input --expect-failure

# Fused attention + output projection. Both external objects are built into the
# working directory by opcheck.py: attn_npu2.o with the causal row helpers and
# the tile-shape -D flags this design implies, and the mm_*.o the registry's
# resolved method names. The attention half's -D flags are derived from the same
# config the builder uses -- a mismatch there does not fail to link, it HANGS.
check-mha-out-proj:
	$(call opcheck,mha_out_proj)

# Phase D2: the whole encoder_bert layer, five operator launches over four
# separately compiled ELFs dispatched through KernelCache.run_sequence, against
# pattern/reference.py. Same lock, same working directory, and for one extra
# reason here: the four ELFs are cached in ./block_cache under $(BUILD_DIR), so
# a clean run and the fault-injected run below share them. Compilation depends
# on the shape and not on the data, and rebuilding four ELFs to perturb one
# weight would double this target's hardware time for nothing.
check-block:
	$(call opcheck,block)

# The block's negative control, kept beside it rather than folded into
# check-fault-control: that target is the shared injection machinery's control
# and runs a cheap 512x512 operator, while this one is a whole layer and costs
# real hardware time. It injects into ln1_weight, which was chosen by
# MEASUREMENT -- perturbing the attention weights instead puts zero elements
# outside the tolerance band, because the layer's two LayerNorms absorb them
# entirely. See the block section of opcheck_specs.py for the numbers.
check-block-fault:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) \
	    python3 $(srcdir)/opcheck.py --operator block \
	        --fault-inject input --expect-failure

# Host-only: the golden model is the layer it claims to be. No NPU, no MLIR.
reference-tests:
	python3 $(srcdir)/pattern/test_reference.py

# Host-only: what check-block's ELF cache will and will not reuse. Builds the
# four MLIR modules and stubs everything past them, so it needs no NPU and
# writes nothing outside a temporary directory -- which is why it can run from
# $(srcdir) rather than from $(BUILD_DIR) like the checks above.
block-cache-tests:
	python3 $(srcdir)/builders/test_block_cache.py

# The negative control: it succeeds only when the injected run is REJECTED BY
# THE COMPARISON. A check that ignored --fault-inject, or one whose tolerance
# were wide enough to swallow a perturbed input, would report PASS under
# injection -- and this target would then fail, which is the point.
#
# --expect-failure does the inversion inside opcheck.py rather than around it
# here. Inverting the exit status in this recipe would take EVERY non-zero exit
# as a caught fault, including a missing PEANO_INSTALL_DIR, a kernel that fails
# to compile or link, and an NPU that never came up -- none of which compare
# anything, all of which would print a passing negative control, and any of
# which would let the operator gate advance on a check that never ran.
#
# One operator, because the injection machinery is shared by all four and the
# driver runs it against each of them directly.
check-fault-control:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) \
	    python3 $(srcdir)/opcheck.py --operator elementwise_add \
	        --fault-inject input --expect-failure

# The Phase C4 registry sweep. Three of the four targets touch no hardware; only
# registry-sweep does, and it is the one that takes an hour.
#
# None of them takes a lock. registry_sweep.py runs each candidate in its own
# subprocess and XRTRunner holds /tmp/npu.lock across every dispatch, so the
# repository lock belongs OUTSIDE this recipe -- taking it here as well would
# deadlock against the wrapper that is supposed to be there.
registry-plan:
	python3 $(srcdir)/sweep/registry_sweep.py --family $(FAMILY) --plan $(SWEEP_ARGS)

registry-resolution:
	python3 $(srcdir)/sweep/registry_sweep.py --family $(FAMILY) --verify-resolution $(SWEEP_ARGS)

registry-sweep:
	python3 $(srcdir)/sweep/registry_sweep.py --family $(FAMILY) $(SWEEP_ARGS)

registry-write:
	python3 $(srcdir)/sweep/registry_sweep.py --family $(FAMILY) --write-registry $(SWEEP_ARGS)

registry-writer-tests:
	python3 $(srcdir)/sweep/test_registry_writer.py

clean:
	rm -rf $(BUILD_DIR) __pycache__ runlist_gate_cache block_cache
