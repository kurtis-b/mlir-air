# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
PYTHON ?= python3

# Architecture selection: aie2 or aie2p (default)
AIE_TARGET ?= aie2p
TARGET_DEVICE ?= npu2

# Transform script path: can be overridden via command line.
TRANSFORM_SCRIPT_PATH ?=

ifeq ($(TRANSFORM_SCRIPT_PATH),)
  ifeq ($(AIE_TARGET),aie2p)
    TRANSFORM_SCRIPT := $(srcdir)/transform_aie2p.mlir
  else
    TRANSFORM_SCRIPT := $(srcdir)/transform_aie2.mlir
  endif
else
  TRANSFORM_SCRIPT := $(TRANSFORM_SCRIPT_PATH)
endif

# Optional custom input IR. Empty means run.py generates shape/layout-specific IR.
INPUT_IR_PATH ?=
ifneq ($(INPUT_IR_PATH),)
  INPUT_IR_ARG := --input-ir $(INPUT_IR_PATH)
endif

# Matrix dimensions.
M ?= 1024
K ?= 1024
N ?= 1024
TILE_M ?= 512
TILE_N ?= 256
SOTA_M ?= 4032
SOTA_K ?= 4320
SOTA_N ?= 4608
SOTA_TILE_M ?= 576
SOTA_TILE_N ?= 1152

# Benchmark contract.
OUTPUT_TYPE ?= int32
B_LAYOUT ?= row
VALIDATION ?= samples
VALIDATION_SAMPLES ?= 64
RNG_SEED ?= 42
ACCEPTANCE_TOPS ?= 36.15
REQUIRE_ACCEPTANCE ?= 0
ifeq ($(REQUIRE_ACCEPTANCE),1)
  REQUIRE_ACCEPTANCE_ARG := --require-acceptance
endif

# Verbosity level for test output (0 = minimal, 1+ = verbose)
VERBOSITY ?= 0

# Profiling loop counts
WARMUPS ?= 10
ITERATIONS ?= 20
RUNTIME_LOOP_TILING ?= 2,4
TRANSFORM_VARIANT ?= default
KERNEL_IMPL ?= vectorized
AIRCC_DEBUG_IR ?= 0
EXTERNAL_SCHEDULE ?= software-pipeline
EXTERNAL_K_PACKS ?= 9
EXTERNAL_BLOCK_M ?= 3
EXTERNAL_BLOCK_N ?= 2
EXTERNAL_CORE_M_PACKS ?= 18
EXTERNAL_ACTIVE_M_PACKS ?= 18
EXTERNAL_CORE_N_PACKS ?= 18
EXTERNAL_C_STRIDE_M_PACKS ?= 18
EXTERNAL_KERNEL_STYLE ?= hand-scheduled
EXTERNAL_KERNEL_OBJECT ?= $(abspath $(BUILD_DIR)/mm_$(EXTERNAL_KERNEL_STYLE)_$(EXTERNAL_SCHEDULE)_k$(EXTERNAL_K_PACKS)_bm$(EXTERNAL_BLOCK_M)_bn$(EXTERNAL_BLOCK_N)_cm$(EXTERNAL_CORE_M_PACKS)_am$(EXTERNAL_ACTIVE_M_PACKS)_cn$(EXTERNAL_CORE_N_PACKS)_cs$(EXTERNAL_C_STRIDE_M_PACKS).o)
PEANO_DIAGNOSTICS ?= 0
EXTERNAL_SCHEDULE_CHOICES := software-pipeline
ifeq ($(filter $(EXTERNAL_SCHEDULE),$(EXTERNAL_SCHEDULE_CHOICES)),)
  $(error EXTERNAL_SCHEDULE must be one of $(EXTERNAL_SCHEDULE_CHOICES))
endif
EXTERNAL_KERNEL_STYLE_CHOICES := hand-scheduled
ifeq ($(filter $(EXTERNAL_KERNEL_STYLE),$(EXTERNAL_KERNEL_STYLE_CHOICES)),)
  $(error EXTERNAL_KERNEL_STYLE must be one of $(EXTERNAL_KERNEL_STYLE_CHOICES))
endif
ifeq ($(EXTERNAL_SCHEDULE),software-pipeline)
  EXTERNAL_SCHEDULE_VALUE := 3
endif
ifeq ($(EXTERNAL_KERNEL_STYLE),hand-scheduled)
  EXTERNAL_KERNEL_STYLE_VALUE := 1
endif

ifeq ($(KERNEL_IMPL),external-mmul)
  ifneq ($(AIE_TARGET),aie2p)
    $(error KERNEL_IMPL=external-mmul requires AIE_TARGET=aie2p)
  endif
  ifneq ($(TARGET_DEVICE),npu2)
    $(error KERNEL_IMPL=external-mmul requires TARGET_DEVICE=npu2)
  endif
  EXTERNAL_KERNEL_ARG := --external-kernel-object $(EXTERNAL_KERNEL_OBJECT)
  EXTERNAL_KERNEL_DEP := $(EXTERNAL_KERNEL_OBJECT)
endif
ifeq ($(AIRCC_DEBUG_IR),1)
  AIRCC_DEBUG_IR_ARG := --aircc-debug-ir
endif

# Trace controls. Trace timing is diagnostic and should not be used for acceptance.
TRACE_SIZE ?= 0
TRACE_OFFSET ?= 0
TRACE_FILE ?= trace_data.txt

# Output controls.
OUTPUT_FORMAT ?= xclbin
DEBUG_IR_PATH ?= ir_debug/final.mlir
BUILD_DIR ?= build
ARTIFACT_DIR ?= $(abspath $(BUILD_DIR)/artifacts)
# Environment setup.
AIEOPT_DIR ?= $(shell realpath $(dir $(shell which aie-opt))/..)
ifeq ($(AIE_TARGET),aie2p)
  PEANO_TARGET := aie2p-none-unknown-elf
else
  PEANO_TARGET := aie2-none-unknown-elf
endif
WARNING_FLAGS ?= -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
PEANO_FLAGS ?= -O2 -std=c++20 --target=$(PEANO_TARGET) $(WARNING_FLAGS) -DNDEBUG -I$(AIEOPT_DIR)/include
ifeq ($(PEANO_DIAGNOSTICS),1)
  PEANO_FLAGS += -Rpass=.* -Rpass-missed=.* -Rpass-analysis=.*
endif

RUN_PY_COMMON_ARGS := \
	$(INPUT_IR_ARG) \
	--transform-script $(TRANSFORM_SCRIPT) \
	-M $(M) -K $(K) -N $(N) \
	--tile-m $(TILE_M) \
	--tile-n $(TILE_N) \
	--output-type $(OUTPUT_TYPE) \
	--b-layout $(B_LAYOUT) \
	--runtime-loop-tiling-sizes $(RUNTIME_LOOP_TILING) \
	--output-format $(OUTPUT_FORMAT) \
	--artifact-dir $(ARTIFACT_DIR) \
	--transform-variant $(TRANSFORM_VARIANT) \
	--kernel-impl $(KERNEL_IMPL) \
	$(EXTERNAL_KERNEL_ARG) \
	--external-schedule $(EXTERNAL_SCHEDULE) \
	--external-kernel-style $(EXTERNAL_KERNEL_STYLE) \
	--external-k-packs $(EXTERNAL_K_PACKS) \
	--external-block-m $(EXTERNAL_BLOCK_M) \
	--external-block-n $(EXTERNAL_BLOCK_N) \
	--external-core-m-packs $(EXTERNAL_CORE_M_PACKS) \
	--external-active-m-packs $(EXTERNAL_ACTIVE_M_PACKS) \
	--external-core-n-packs $(EXTERNAL_CORE_N_PACKS) \
	--external-c-stride-m-packs $(EXTERNAL_C_STRIDE_M_PACKS) \
	$(AIRCC_DEBUG_IR_ARG) \
	--trace-size $(TRACE_SIZE) \
	--trace-offset $(TRACE_OFFSET) \
	--trace-file $(TRACE_FILE) \
	--target-device $(TARGET_DEVICE) \
	--seed $(RNG_SEED)

PROFILE_ARGS := \
	-x air.xclbin -k MLIR_AIE -i air.insts.bin \
	-M $(M) -K $(K) -N $(N) \
	--tile_m $(TILE_M) \
	--tile_n $(TILE_N) \
	-v $(VERBOSITY) \
	--warmups $(WARMUPS) \
	--iterations $(ITERATIONS) \
	--b-layout $(B_LAYOUT) \
	--output-type $(OUTPUT_TYPE) \
	--validation $(VALIDATION) \
	--validation-samples $(VALIDATION_SAMPLES) \
	--seed $(RNG_SEED) \
	--acceptance-tops $(ACCEPTANCE_TOPS) \
	$(REQUIRE_ACCEPTANCE_ARG)

.PHONY: all run run-aie2 run-aie2p compile-xclbin profile build-test-exe build-external-mmul debug_ir acceptance-int8 acceptance-int8-external profile-int8-sota-best clean

all: run

run: run-$(AIE_TARGET)

run-aie2: $(EXTERNAL_KERNEL_DEP)
	mkdir -p $(BUILD_DIR) $(ARTIFACT_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(srcdir)/run.py $(RUN_PY_COMMON_ARGS)

run-aie2p: $(EXTERNAL_KERNEL_DEP)
	mkdir -p $(BUILD_DIR) $(ARTIFACT_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(srcdir)/run.py $(RUN_PY_COMMON_ARGS)

compile-xclbin: $(EXTERNAL_KERNEL_DEP)
	mkdir -p $(BUILD_DIR) $(ARTIFACT_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(srcdir)/run.py $(RUN_PY_COMMON_ARGS) --compile-only

build-external-mmul: $(EXTERNAL_KERNEL_OBJECT)

$(EXTERNAL_KERNEL_OBJECT): $(srcdir)/mm.cc
	@mkdir -p $(dir $@)
	@if [ -z "$(PEANO_INSTALL_DIR)" ]; then \
		echo "Error: PEANO_INSTALL_DIR is not set. Please source the Peano environment or export PEANO_INSTALL_DIR."; \
		exit 1; \
	fi
	$(PEANO_INSTALL_DIR)/bin/clang++ $(PEANO_FLAGS) -DEXTERNAL_SCHEDULE=$(EXTERNAL_SCHEDULE_VALUE) -DEXTERNAL_KERNEL_STYLE=$(EXTERNAL_KERNEL_STYLE_VALUE) -DEXTERNAL_K_PACKS=$(EXTERNAL_K_PACKS) -DEXTERNAL_BLOCK_M=$(EXTERNAL_BLOCK_M) -DEXTERNAL_BLOCK_N=$(EXTERNAL_BLOCK_N) -DEXTERNAL_CORE_M_PACKS=$(EXTERNAL_CORE_M_PACKS) -DEXTERNAL_ACTIVE_M_PACKS=$(EXTERNAL_ACTIVE_M_PACKS) -DEXTERNAL_CORE_N_PACKS=$(EXTERNAL_CORE_N_PACKS) -DEXTERNAL_C_STRIDE_M_PACKS=$(EXTERNAL_C_STRIDE_M_PACKS) -c $< -o $@
	@echo "Kernel compiled successfully: $@"

profile: compile-xclbin build-test-exe
	cd $(BUILD_DIR) && ./test.exe $(PROFILE_ARGS)

acceptance-int8:
	$(MAKE) -f $(srcdir)/Makefile profile \
		M=$(SOTA_M) K=$(SOTA_K) N=$(SOTA_N) \
		TILE_M=$(SOTA_TILE_M) TILE_N=$(SOTA_TILE_N) \
		OUTPUT_TYPE=int8 B_LAYOUT=column TRANSFORM_VARIANT=sota-int8 \
		WARMUPS=20 ITERATIONS=100 VALIDATION=samples REQUIRE_ACCEPTANCE=1 \
		ACCEPTANCE_TOPS=$(ACCEPTANCE_TOPS) BUILD_DIR=$(BUILD_DIR) \
		ARTIFACT_DIR=$(ARTIFACT_DIR) TRANSFORM_SCRIPT_PATH=$(TRANSFORM_SCRIPT)

acceptance-int8-external:
	$(MAKE) -f $(srcdir)/Makefile profile-int8-sota-best \
		ACCEPTANCE_TOPS=$(ACCEPTANCE_TOPS) BUILD_DIR=$(BUILD_DIR) \
		ARTIFACT_DIR=$(ARTIFACT_DIR) TRANSFORM_SCRIPT_PATH=$(TRANSFORM_SCRIPT)

profile-int8-sota-best:
	$(MAKE) -f $(srcdir)/Makefile profile \
		M=$(SOTA_M) K=$(SOTA_K) N=$(SOTA_N) \
		TILE_M=$(SOTA_TILE_M) TILE_N=$(SOTA_TILE_N) \
		OUTPUT_TYPE=int8 B_LAYOUT=column TRANSFORM_VARIANT=sota-int8 \
		KERNEL_IMPL=external-mmul EXTERNAL_KERNEL_STYLE=hand-scheduled \
		EXTERNAL_BLOCK_M=3 EXTERNAL_BLOCK_N=2 EXTERNAL_K_PACKS=9 \
		EXTERNAL_CORE_M_PACKS=18 EXTERNAL_ACTIVE_M_PACKS=18 \
		EXTERNAL_CORE_N_PACKS=18 EXTERNAL_C_STRIDE_M_PACKS=18 \
		RUNTIME_LOOP_TILING=2,4 TRACE_SIZE=0 AIRCC_DEBUG_IR=0 \
		WARMUPS=5 ITERATIONS=20 VALIDATION=samples REQUIRE_ACCEPTANCE=0 \
		ACCEPTANCE_TOPS=$(ACCEPTANCE_TOPS) BUILD_DIR=$(BUILD_DIR) \
		ARTIFACT_DIR=$(ARTIFACT_DIR) TRANSFORM_SCRIPT_PATH=$(TRANSFORM_SCRIPT)

build-test-exe:
	@GPP=$$( \
		for bin in /usr/bin/g++-*; do \
			ver=$$(echo $$bin | grep -oE '[0-9]+$$'); \
			if [ "$$ver" -ge 13 ] 2>/dev/null; then \
				echo "$$ver $$bin"; \
			fi; \
		done | sort -nr | head -n1 | awk '{print $$2}' \
	); \
	if [ -z "$$GPP" ]; then \
		echo "Error: No g++ version >= 13 found in /usr/bin."; \
		exit 1; \
	fi; \
	if [ -z "$$XILINX_XRT" ]; then \
		echo "Error: XILINX_XRT environment variable not set. Please source xrt/setup.sh."; \
		exit 1; \
	fi; \
	if [ -z "$(AIEOPT_DIR)" ]; then \
		echo "Error: AIEOPT_DIR is not set. Please source utils/env_setup.sh."; \
		exit 1; \
	fi; \
	echo "Using compiler: $$GPP"; \
	mkdir -p $(BUILD_DIR); \
	cd $(BUILD_DIR) && $$GPP $(srcdir)/test.cpp -o test.exe -std=c++23 -Wall \
		-I$$XILINX_XRT/include -L$$XILINX_XRT/lib \
		-I$(AIEOPT_DIR)/runtime_lib/x86_64/test_lib/include \
		-L$(AIEOPT_DIR)/runtime_lib/x86_64/test_lib/lib \
		-luuid -lxrt_coreutil -lrt -lstdc++ -ltest_utils

debug_ir:
	mkdir -p $(BUILD_DIR) $(ARTIFACT_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(srcdir)/run.py $(RUN_PY_COMMON_ARGS) --debug-ir $(DEBUG_IR_PATH)


clean:
	rm -rf $(BUILD_DIR) __pycache__
