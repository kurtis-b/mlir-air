# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
repodir := $(realpath $(srcdir)/../../..)

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

# Trace controls. Trace timing is diagnostic and should not be used for acceptance.
TRACE_SIZE ?= 0
TRACE_OFFSET ?= 0
TRACE_FILE ?= trace_data.txt

# Output controls.
OUTPUT_FORMAT ?= xclbin
DEBUG_IR_PATH ?= ir_debug/final.mlir
BUILD_DIR ?= build
ARTIFACT_DIR ?= $(abspath $(BUILD_DIR)/artifacts)
STATIC_REPORT_PREFIX ?= npu_gemm

# Environment setup.
AIEOPT_DIR ?= $(shell realpath $(dir $(shell which aie-opt))/..)
DISASSEMBLE ?= $(repodir)/utils/isa_inspect/disassemble.sh

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
	--trace-size $(TRACE_SIZE) \
	--trace-offset $(TRACE_OFFSET) \
	--trace-file $(TRACE_FILE) \
	--target-device $(TARGET_DEVICE) \
	--seed $(RNG_SEED)

PROFILE_ARGS := \
	-x air.xclbin -k MLIR_AIE -i air.insts.bin \
	-M $(M) -K $(K) -N $(N) \
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

.PHONY: all run run-aie2 run-aie2p compile-xclbin profile build-test-exe debug_ir static-report disassemble acceptance-int8 clean

all: run

run: run-$(AIE_TARGET)

run-aie2:
	mkdir -p $(BUILD_DIR) $(ARTIFACT_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(srcdir)/run.py $(RUN_PY_COMMON_ARGS)

run-aie2p:
	mkdir -p $(BUILD_DIR) $(ARTIFACT_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(srcdir)/run.py $(RUN_PY_COMMON_ARGS)

compile-xclbin: static-report
	mkdir -p $(BUILD_DIR) $(ARTIFACT_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(srcdir)/run.py $(RUN_PY_COMMON_ARGS) --compile-only

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

static-report:
	mkdir -p $(ARTIFACT_DIR)
	$(PYTHON) $(srcdir)/analyze_npu_gemm.py \
		-M $(M) -K $(K) -N $(N) \
		--tile-m $(TILE_M) \
		--tile-n $(TILE_N) \
		--output-type $(OUTPUT_TYPE) \
		--b-layout $(B_LAYOUT) \
		--runtime-loop-tiling $(RUNTIME_LOOP_TILING) \
		--trace-size $(TRACE_SIZE) \
		--transform-variant $(TRANSFORM_VARIANT) \
		--acceptance-tops $(ACCEPTANCE_TOPS) \
		--transform-script $(TRANSFORM_SCRIPT) \
		--transformed-ir $(BUILD_DIR)/$(DEBUG_IR_PATH) \
		--artifact-dir $(ARTIFACT_DIR) \
		--prefix $(STATIC_REPORT_PREFIX)

disassemble:
	mkdir -p $(ARTIFACT_DIR)
	@found=0; \
	for elf in $$(find $(BUILD_DIR) -type f \( -name 'bare_matmul*_core_*.elf' -o -name '*.elf' \) | sort); do \
		found=1; \
		prefix=$$(echo $$elf | sed 's#[^A-Za-z0-9_.-]#_#g; s#^_*##; s#_*$$##'); \
		$(DISASSEMBLE) npu --kind elf --mcpu aie2p --triple aie2p-none-unknown-elf --output-dir $(ARTIFACT_DIR) --prefix $$prefix $$elf || exit $$?; \
	done; \
	if [ "$$found" -eq 0 ]; then echo "No per-core ELF files found under $(BUILD_DIR)"; fi; \
	if [ -f $(BUILD_DIR)/air.insts.bin ]; then \
		$(DISASSEMBLE) npu --kind txn --output-dir $(ARTIFACT_DIR) --prefix npu_air_insts $(BUILD_DIR)/air.insts.bin; \
	else \
		echo "No air.insts.bin found under $(BUILD_DIR)"; \
	fi

clean:
	rm -rf $(BUILD_DIR) __pycache__
