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
KERNEL_IMPL ?= vectorized
OMIT_PING_PONG ?=
AIRCC_DEBUG_IR ?= 0
EXTERNAL_SCHEDULE ?= software-pipeline
EXTERNAL_K_PACKS ?= 9
EXTERNAL_BLOCK_M ?= 2
EXTERNAL_BLOCK_N ?= 2
EXTERNAL_CORE_M_PACKS ?= 18
EXTERNAL_ACTIVE_M_PACKS ?= $(EXTERNAL_CORE_M_PACKS)
EXTERNAL_CORE_N_PACKS ?= 18
EXTERNAL_C_STRIDE_M_PACKS ?= $(EXTERNAL_CORE_M_PACKS)
EXTERNAL_ATB_C_OFFSET ?= 0
ATB_K_CHUNK_ELEMENTS ?= 864
PERFORMANCE_MODE ?= 0
EXTERNAL_KERNEL_STYLE ?= peano-mmul
EXTERNAL_KERNEL_OBJECT ?= $(abspath $(BUILD_DIR)/mm_$(EXTERNAL_KERNEL_STYLE)_$(EXTERNAL_SCHEDULE)_k$(EXTERNAL_K_PACKS)_bm$(EXTERNAL_BLOCK_M)_bn$(EXTERNAL_BLOCK_N)_cm$(EXTERNAL_CORE_M_PACKS)_am$(EXTERNAL_ACTIVE_M_PACKS)_cn$(EXTERNAL_CORE_N_PACKS)_cs$(EXTERNAL_C_STRIDE_M_PACKS)_aco$(EXTERNAL_ATB_C_OFFSET).o)
XLLVM_PROBE_MLIR ?= $(srcdir)/xllvm_mmul_probe.mlir
XLLVM_PROBE_LL ?= $(abspath $(BUILD_DIR)/xllvm_mmul_probe.ll)
XLLVM_PROBE_OBJECT ?= $(abspath $(BUILD_DIR)/xllvm_mmul_probe.o)
PEANO_DIAGNOSTICS ?= 0
EXTERNAL_BASELINE_INNER_MAC_DENSITY ?= 0.176471
EXTERNAL_MAX_INNER_NOP_PER_MAC ?= 1.25
ATB_REFERENCE_DIR ?= /home/cj/mlir-aie/programming_examples/ml/block_datatypes/gemm_asymmetric_tile_buffering/config2
ATB_REFERENCE_SOURCE ?= $(ATB_REFERENCE_DIR)/mm_bfp_mixed.cc
ATB_REFERENCE_OBJECT ?= $(ATB_REFERENCE_DIR)/build/mm_192x128x96.o

ifeq ($(TRANSFORM_VARIANT),sota-int8-atb)
  ifeq ($(origin M),file)
    M := $(SOTA_M)
  endif
  ifeq ($(origin K),file)
    K := $(SOTA_K)
  endif
  ifeq ($(origin N),file)
    N := $(SOTA_N)
  endif
  ifeq ($(origin TILE_M),file)
    TILE_M := $(SOTA_TILE_M)
  endif
  ifeq ($(origin TILE_N),file)
    TILE_N := $(SOTA_TILE_N)
  endif
  ifeq ($(origin OUTPUT_TYPE),file)
    OUTPUT_TYPE := int8
  endif
  ifeq ($(origin B_LAYOUT),file)
    B_LAYOUT := column
  endif
  ifeq ($(origin KERNEL_IMPL),file)
    KERNEL_IMPL := external-mmul
  endif
  ifeq ($(origin OMIT_PING_PONG),file)
    OMIT_PING_PONG := L1-partial-a
  endif
  ifeq ($(origin RUNTIME_LOOP_TILING),file)
    RUNTIME_LOOP_TILING := 1,1
  endif
  ifeq ($(origin EXTERNAL_KERNEL_STYLE),file)
    EXTERNAL_KERNEL_STYLE := native-mmul
  endif
  ifeq ($(origin EXTERNAL_K_PACKS),file)
    EXTERNAL_K_PACKS := 18
  endif
  ifeq ($(origin EXTERNAL_BLOCK_M),file)
    EXTERNAL_BLOCK_M := 2
  endif
  ifeq ($(origin EXTERNAL_BLOCK_N),file)
    EXTERNAL_BLOCK_N := 2
  endif
  ifeq ($(origin EXTERNAL_CORE_M_PACKS),file)
    EXTERNAL_CORE_M_PACKS := 18
  endif
  ifeq ($(origin EXTERNAL_ACTIVE_M_PACKS),file)
    EXTERNAL_ACTIVE_M_PACKS := 6
  endif
  ifeq ($(origin EXTERNAL_CORE_N_PACKS),file)
    EXTERNAL_CORE_N_PACKS := 18
  endif
  ifeq ($(origin EXTERNAL_C_STRIDE_M_PACKS),file)
    EXTERNAL_C_STRIDE_M_PACKS := 6
  endif
endif

ifeq ($(TRANSFORM_VARIANT),sota-int8-atb-v2)
  ifeq ($(origin M),file)
    M := $(SOTA_M)
  endif
  ifeq ($(origin K),file)
    K := $(SOTA_K)
  endif
  ifeq ($(origin N),file)
    N := $(SOTA_N)
  endif
  ifeq ($(origin TILE_M),file)
    TILE_M := $(SOTA_TILE_M)
  endif
  ifeq ($(origin TILE_N),file)
    TILE_N := $(SOTA_TILE_N)
  endif
  ifeq ($(origin OUTPUT_TYPE),file)
    OUTPUT_TYPE := int8
  endif
  ifeq ($(origin B_LAYOUT),file)
    B_LAYOUT := column
  endif
  ifeq ($(origin KERNEL_IMPL),file)
    KERNEL_IMPL := external-mmul
  endif
  ifeq ($(origin OMIT_PING_PONG),file)
    OMIT_PING_PONG := L1-partial-a
  endif
  ifeq ($(origin RUNTIME_LOOP_TILING),file)
    RUNTIME_LOOP_TILING := 2,4
  endif
  ifeq ($(origin EXTERNAL_KERNEL_STYLE),file)
    EXTERNAL_KERNEL_STYLE := native-mmul
  endif
  ifeq ($(origin EXTERNAL_K_PACKS),file)
    EXTERNAL_K_PACKS := 18
  endif
  ifeq ($(origin EXTERNAL_BLOCK_M),file)
    EXTERNAL_BLOCK_M := 2
  endif
  ifeq ($(origin EXTERNAL_BLOCK_N),file)
    EXTERNAL_BLOCK_N := 2
  endif
  ifeq ($(origin EXTERNAL_CORE_M_PACKS),file)
    EXTERNAL_CORE_M_PACKS := 18
  endif
  ifeq ($(origin EXTERNAL_ACTIVE_M_PACKS),file)
    EXTERNAL_ACTIVE_M_PACKS := 6
  endif
  ifeq ($(origin EXTERNAL_CORE_N_PACKS),file)
    EXTERNAL_CORE_N_PACKS := 18
  endif
  ifeq ($(origin EXTERNAL_C_STRIDE_M_PACKS),file)
    EXTERNAL_C_STRIDE_M_PACKS := 18
  endif
  ifeq ($(origin EXTERNAL_ATB_C_OFFSET),file)
    EXTERNAL_ATB_C_OFFSET := 1
  endif
endif

EXTERNAL_SCHEDULE_CHOICES := baseline flat manual-unroll software-pipeline
ifeq ($(filter $(EXTERNAL_SCHEDULE),$(EXTERNAL_SCHEDULE_CHOICES)),)
  $(error EXTERNAL_SCHEDULE must be one of $(EXTERNAL_SCHEDULE_CHOICES))
endif
EXTERNAL_KERNEL_STYLE_CHOICES := peano-mmul hand-scheduled native-mmul native-mmul-atb-ref asm-microkernel
ifeq ($(filter $(EXTERNAL_KERNEL_STYLE),$(EXTERNAL_KERNEL_STYLE_CHOICES)),)
  $(error EXTERNAL_KERNEL_STYLE must be one of $(EXTERNAL_KERNEL_STYLE_CHOICES))
endif
ifeq ($(EXTERNAL_KERNEL_STYLE),asm-microkernel)
  $(error EXTERNAL_KERNEL_STYLE=asm-microkernel is reserved; dense 3x2 requires six v64acc32 accumulators but this AIE2P target exposes five 2048-bit DM accumulator registers)
endif
ifeq ($(EXTERNAL_KERNEL_STYLE),native-mmul-atb-ref)
  ifneq ($(TRANSFORM_VARIANT),sota-int8-atb-v2)
    $(error EXTERNAL_KERNEL_STYLE=native-mmul-atb-ref is only valid with TRANSFORM_VARIANT=sota-int8-atb-v2)
  endif
endif
ifeq ($(EXTERNAL_SCHEDULE),baseline)
  EXTERNAL_SCHEDULE_VALUE := 0
else ifeq ($(EXTERNAL_SCHEDULE),flat)
  EXTERNAL_SCHEDULE_VALUE := 1
else ifeq ($(EXTERNAL_SCHEDULE),manual-unroll)
  EXTERNAL_SCHEDULE_VALUE := 2
else ifeq ($(EXTERNAL_SCHEDULE),software-pipeline)
  EXTERNAL_SCHEDULE_VALUE := 3
endif
ifeq ($(EXTERNAL_KERNEL_STYLE),peano-mmul)
  EXTERNAL_KERNEL_STYLE_VALUE := 0
else ifeq ($(EXTERNAL_KERNEL_STYLE),hand-scheduled)
  EXTERNAL_KERNEL_STYLE_VALUE := 1
else ifeq ($(EXTERNAL_KERNEL_STYLE),native-mmul)
  EXTERNAL_KERNEL_STYLE_VALUE := 2
else ifeq ($(EXTERNAL_KERNEL_STYLE),asm-microkernel)
  EXTERNAL_KERNEL_STYLE_VALUE := 3
else ifeq ($(EXTERNAL_KERNEL_STYLE),native-mmul-atb-ref)
  EXTERNAL_KERNEL_STYLE_VALUE := 4
endif

EXTERNAL_ASM_GATE_ARGS := --require-clean
ifneq ($(filter $(EXTERNAL_KERNEL_STYLE),native-mmul native-mmul-atb-ref asm-microkernel),)
  EXTERNAL_ASM_GATE_ARGS += --baseline-inner-mac-density $(EXTERNAL_BASELINE_INNER_MAC_DENSITY)
  EXTERNAL_ASM_GATE_ARGS += --max-inner-nop-per-mac $(EXTERNAL_MAX_INNER_NOP_PER_MAC)
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
  EXTERNAL_ASM_REPORT_DEP := inspect-external-mmul
endif
ifneq ($(OMIT_PING_PONG),)
  OMIT_PING_PONG_ARG := --omit-ping-pong $(OMIT_PING_PONG)
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
LOWERED_IR_PATH ?=
STATIC_REPORT_PREFIX ?= npu_gemm
ifneq ($(LOWERED_IR_PATH),)
  LOWERED_IR_ARG := --lowered-ir $(LOWERED_IR_PATH)
endif
ifeq ($(PERFORMANCE_MODE),1)
  PERFORMANCE_MODE_ARG := --performance-mode
endif

# Environment setup.
AIEOPT_DIR ?= $(shell realpath $(dir $(shell which aie-opt))/..)
AIE_TRANSLATE ?= $(shell which aie-translate)
DISASSEMBLE ?= $(repodir)/utils/isa_inspect/disassemble.sh
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
	--atb-k-chunk-elements $(ATB_K_CHUNK_ELEMENTS) \
	$(OMIT_PING_PONG_ARG) \
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

.PHONY: all run run-aie2 run-aie2p compile-xclbin profile build-test-exe build-external-mmul inspect-external-mmul inspect-atb-reference probe-xllvm-mmul debug_ir static-report disassemble acceptance-int8 acceptance-int8-external acceptance-int8-atb acceptance-int8-atb-v2 acceptance-int8-atb-v2-candidate clean

all: run

run: run-$(AIE_TARGET)

run-aie2: $(EXTERNAL_KERNEL_DEP)
	mkdir -p $(BUILD_DIR) $(ARTIFACT_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(srcdir)/run.py $(RUN_PY_COMMON_ARGS)

run-aie2p: $(EXTERNAL_KERNEL_DEP)
	mkdir -p $(BUILD_DIR) $(ARTIFACT_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(srcdir)/run.py $(RUN_PY_COMMON_ARGS)

compile-xclbin: static-report $(EXTERNAL_KERNEL_DEP) $(EXTERNAL_ASM_REPORT_DEP)
	mkdir -p $(BUILD_DIR) $(ARTIFACT_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(srcdir)/run.py $(RUN_PY_COMMON_ARGS) --compile-only
	@if [ "$(TRANSFORM_VARIANT)" = "sota-int8-atb" ] || [ "$(TRANSFORM_VARIANT)" = "sota-int8-atb-v2" ]; then \
		$(MAKE) -f $(srcdir)/Makefile static-report BUILD_DIR=$(BUILD_DIR) ARTIFACT_DIR=$(ARTIFACT_DIR) LOWERED_IR_PATH=$(ARTIFACT_DIR)/air_module.lowered.mlir; \
	fi
	@if [ "$(KERNEL_IMPL)" = "external-mmul" ]; then \
		$(MAKE) -f $(srcdir)/Makefile disassemble BUILD_DIR=$(BUILD_DIR) ARTIFACT_DIR=$(ARTIFACT_DIR); \
	fi

build-external-mmul: $(EXTERNAL_KERNEL_OBJECT) inspect-external-mmul

probe-xllvm-mmul: $(XLLVM_PROBE_OBJECT) inspect-external-mmul
	mkdir -p $(ARTIFACT_DIR)
	$(PYTHON) $(srcdir)/analyze_external_mmul_asm.py \
		--object $(XLLVM_PROBE_OBJECT) \
		--llvm-objdump $(PEANO_INSTALL_DIR)/bin/llvm-objdump \
		--artifact-dir $(ARTIFACT_DIR) \
		--prefix $(STATIC_REPORT_PREFIX).xllvm_probe \
		--schedule $(EXTERNAL_SCHEDULE) \
		--kernel-style native-mmul \
		--external-k-packs 1 \
		--external-block-m 2 \
		--external-block-n 2 \
		--external-core-m-packs 2 \
		--external-active-m-packs 2 \
		--external-core-n-packs 2 \
		--require-clean --min-mac-density 0.01 --min-inner-mac-density 0.01 \
		--max-inner-nop-per-mac 4.0
	$(PYTHON) -c 'import json; from pathlib import Path; prod=json.loads(Path("$(ARTIFACT_DIR)/$(STATIC_REPORT_PREFIX).external_mmul.asm.json").read_text()); probe=json.loads(Path("$(ARTIFACT_DIR)/$(STATIC_REPORT_PREFIX).xllvm_probe.asm.json").read_text()); p=prod["inner_counts"]["mac_density_per_op_slot"]; x=probe["inner_counts"]["mac_density_per_op_slot"]; integ="YES" if x >= p * 1.05 and probe["inner_counts"]["vector_stack_ref_lines"] == 0 else "NO"; print(f"xllvm_probe_inner_mac_density={x:.6f}"); print(f"peano_native_inner_mac_density={p:.6f}"); print(f"xllvm_probe_speedup_vs_peano={x / p if p else 0.0:.6f}"); print(f"xllvm_probe_runtime_integration={integ}")'

$(XLLVM_PROBE_LL): $(XLLVM_PROBE_MLIR)
	@mkdir -p $(dir $@)
	@if [ -z "$(AIE_TRANSLATE)" ]; then \
		echo "Error: aie-translate is not on PATH. Please source utils/env_setup.sh."; \
		exit 1; \
	fi
	$(AIE_TRANSLATE) $< --mlir-to-llvmir > $@

$(XLLVM_PROBE_OBJECT): $(XLLVM_PROBE_LL)
	@mkdir -p $(dir $@)
	@if [ -z "$(PEANO_INSTALL_DIR)" ]; then \
		echo "Error: PEANO_INSTALL_DIR is not set. Please source the Peano environment or export PEANO_INSTALL_DIR."; \
		exit 1; \
	fi
	$(PEANO_INSTALL_DIR)/bin/clang -O2 --target=$(PEANO_TARGET) -c $< -o $@
	@echo "XLLVM probe compiled successfully: $@"

inspect-external-mmul: $(EXTERNAL_KERNEL_OBJECT)
	mkdir -p $(ARTIFACT_DIR)
	$(PYTHON) $(srcdir)/analyze_external_mmul_asm.py \
		--object $(EXTERNAL_KERNEL_OBJECT) \
		--llvm-objdump $(PEANO_INSTALL_DIR)/bin/llvm-objdump \
		--artifact-dir $(ARTIFACT_DIR) \
		--prefix $(STATIC_REPORT_PREFIX).external_mmul \
		--schedule $(EXTERNAL_SCHEDULE) \
		--kernel-style $(EXTERNAL_KERNEL_STYLE) \
		--external-k-packs $(EXTERNAL_K_PACKS) \
		--external-block-m $(EXTERNAL_BLOCK_M) \
		--external-block-n $(EXTERNAL_BLOCK_N) \
		--external-core-m-packs $(EXTERNAL_CORE_M_PACKS) \
		--external-active-m-packs $(EXTERNAL_ACTIVE_M_PACKS) \
		--external-core-n-packs $(EXTERNAL_CORE_N_PACKS) \
		--atb-reference-source $(ATB_REFERENCE_SOURCE) \
		$(EXTERNAL_ASM_GATE_ARGS)

inspect-atb-reference:
	mkdir -p $(ARTIFACT_DIR)
	$(PYTHON) $(srcdir)/analyze_external_mmul_asm.py \
		--reference-only \
		--artifact-dir $(ARTIFACT_DIR) \
		--prefix $(STATIC_REPORT_PREFIX).atb_reference \
		--atb-reference-source $(ATB_REFERENCE_SOURCE)
	@if command -v xchesscc >/dev/null 2>&1 && command -v xchesscc_wrapper >/dev/null 2>&1; then \
		echo "Building optional Chess ATB reference object"; \
		$(MAKE) -C $(ATB_REFERENCE_DIR) use_chess=1 use_placed=1 devicename=npu2 build/mm_192x128x96.o || true; \
		if [ -f $(ATB_REFERENCE_OBJECT) ]; then \
			$(PYTHON) $(srcdir)/analyze_external_mmul_asm.py \
				--object $(ATB_REFERENCE_OBJECT) \
				--llvm-objdump $(PEANO_INSTALL_DIR)/bin/llvm-objdump \
				--artifact-dir $(ARTIFACT_DIR) \
				--prefix $(STATIC_REPORT_PREFIX).atb_reference.chess \
				--schedule software-pipeline \
				--kernel-style native-mmul-atb-ref \
				--external-k-packs 16 \
				--external-block-m 2 \
				--external-block-n 2 \
				--external-core-m-packs 24 \
				--external-active-m-packs 4 \
				--external-core-n-packs 12 \
				--atb-reference-source $(ATB_REFERENCE_SOURCE) || true; \
		fi; \
	else \
		echo "Skipping optional Chess ATB reference build: xchesscc/xchesscc_wrapper not available"; \
	fi

$(EXTERNAL_KERNEL_OBJECT): $(srcdir)/mm.cc
	@mkdir -p $(dir $@)
	@if [ -z "$(PEANO_INSTALL_DIR)" ]; then \
		echo "Error: PEANO_INSTALL_DIR is not set. Please source the Peano environment or export PEANO_INSTALL_DIR."; \
		exit 1; \
	fi
	$(PEANO_INSTALL_DIR)/bin/clang++ $(PEANO_FLAGS) -DEXTERNAL_SCHEDULE=$(EXTERNAL_SCHEDULE_VALUE) -DEXTERNAL_KERNEL_STYLE=$(EXTERNAL_KERNEL_STYLE_VALUE) -DEXTERNAL_K_PACKS=$(EXTERNAL_K_PACKS) -DEXTERNAL_BLOCK_M=$(EXTERNAL_BLOCK_M) -DEXTERNAL_BLOCK_N=$(EXTERNAL_BLOCK_N) -DEXTERNAL_CORE_M_PACKS=$(EXTERNAL_CORE_M_PACKS) -DEXTERNAL_ACTIVE_M_PACKS=$(EXTERNAL_ACTIVE_M_PACKS) -DEXTERNAL_CORE_N_PACKS=$(EXTERNAL_CORE_N_PACKS) -DEXTERNAL_C_STRIDE_M_PACKS=$(EXTERNAL_C_STRIDE_M_PACKS) -DEXTERNAL_ATB_C_OFFSET=$(EXTERNAL_ATB_C_OFFSET) -c $< -o $@
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
	$(MAKE) -f $(srcdir)/Makefile profile \
		M=$(SOTA_M) K=$(SOTA_K) N=$(SOTA_N) \
		TILE_M=$(SOTA_TILE_M) TILE_N=$(SOTA_TILE_N) \
		OUTPUT_TYPE=int8 B_LAYOUT=column TRANSFORM_VARIANT=sota-int8 \
		KERNEL_IMPL=external-mmul WARMUPS=20 ITERATIONS=100 \
		VALIDATION=samples REQUIRE_ACCEPTANCE=1 \
		ACCEPTANCE_TOPS=$(ACCEPTANCE_TOPS) BUILD_DIR=$(BUILD_DIR) \
		ARTIFACT_DIR=$(ARTIFACT_DIR) TRANSFORM_SCRIPT_PATH=$(TRANSFORM_SCRIPT)

acceptance-int8-atb:
	$(MAKE) -f $(srcdir)/Makefile profile \
		M=$(SOTA_M) K=$(SOTA_K) N=$(SOTA_N) \
		TILE_M=$(SOTA_TILE_M) TILE_N=$(SOTA_TILE_N) \
		OUTPUT_TYPE=int8 B_LAYOUT=column TRANSFORM_VARIANT=sota-int8-atb \
		KERNEL_IMPL=external-mmul EXTERNAL_KERNEL_STYLE=native-mmul \
		EXTERNAL_BLOCK_M=2 EXTERNAL_BLOCK_N=2 EXTERNAL_K_PACKS=18 \
		EXTERNAL_CORE_M_PACKS=18 EXTERNAL_ACTIVE_M_PACKS=6 EXTERNAL_CORE_N_PACKS=18 EXTERNAL_C_STRIDE_M_PACKS=6 \
		OMIT_PING_PONG=L1-partial-a WARMUPS=20 ITERATIONS=100 \
		VALIDATION=samples REQUIRE_ACCEPTANCE=1 \
		ACCEPTANCE_TOPS=$(ACCEPTANCE_TOPS) BUILD_DIR=$(BUILD_DIR) \
		ARTIFACT_DIR=$(ARTIFACT_DIR) TRANSFORM_SCRIPT_PATH=$(TRANSFORM_SCRIPT)

acceptance-int8-atb-v2:
	$(MAKE) -f $(srcdir)/Makefile profile \
		M=$(SOTA_M) K=$(SOTA_K) N=$(SOTA_N) \
		TILE_M=$(SOTA_TILE_M) TILE_N=$(SOTA_TILE_N) \
		OUTPUT_TYPE=int8 B_LAYOUT=column TRANSFORM_VARIANT=sota-int8-atb-v2 \
		KERNEL_IMPL=external-mmul EXTERNAL_KERNEL_STYLE=native-mmul \
		EXTERNAL_BLOCK_M=2 EXTERNAL_BLOCK_N=2 EXTERNAL_K_PACKS=18 \
		EXTERNAL_CORE_M_PACKS=18 EXTERNAL_ACTIVE_M_PACKS=6 EXTERNAL_CORE_N_PACKS=18 EXTERNAL_C_STRIDE_M_PACKS=18 \
		EXTERNAL_ATB_C_OFFSET=1 ATB_K_CHUNK_ELEMENTS=$(ATB_K_CHUNK_ELEMENTS) \
		OMIT_PING_PONG=L1-partial-a RUNTIME_LOOP_TILING=2,4 \
		WARMUPS=5 ITERATIONS=20 VALIDATION=samples REQUIRE_ACCEPTANCE=0 \
		PERFORMANCE_MODE=1 ACCEPTANCE_TOPS=$(ACCEPTANCE_TOPS) BUILD_DIR=$(BUILD_DIR) \
		ARTIFACT_DIR=$(ARTIFACT_DIR) TRANSFORM_SCRIPT_PATH=$(TRANSFORM_SCRIPT)

acceptance-int8-atb-v2-candidate:
	$(MAKE) -f $(srcdir)/Makefile profile \
		M=$(SOTA_M) K=$(SOTA_K) N=$(SOTA_N) \
		TILE_M=$(SOTA_TILE_M) TILE_N=$(SOTA_TILE_N) \
		OUTPUT_TYPE=int8 B_LAYOUT=column TRANSFORM_VARIANT=sota-int8-atb-v2 \
		KERNEL_IMPL=external-mmul EXTERNAL_KERNEL_STYLE=native-mmul-atb-ref \
		EXTERNAL_BLOCK_M=2 EXTERNAL_BLOCK_N=2 EXTERNAL_K_PACKS=18 \
		EXTERNAL_CORE_M_PACKS=18 EXTERNAL_ACTIVE_M_PACKS=6 EXTERNAL_CORE_N_PACKS=18 EXTERNAL_C_STRIDE_M_PACKS=18 \
		EXTERNAL_ATB_C_OFFSET=1 ATB_K_CHUNK_ELEMENTS=$(ATB_K_CHUNK_ELEMENTS) \
		OMIT_PING_PONG=L1-partial-a RUNTIME_LOOP_TILING=2,4 \
		WARMUPS=5 ITERATIONS=20 VALIDATION=samples REQUIRE_ACCEPTANCE=0 \
		PERFORMANCE_MODE=1 ACCEPTANCE_TOPS=$(ACCEPTANCE_TOPS) BUILD_DIR=$(BUILD_DIR) \
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
		--kernel-impl $(KERNEL_IMPL) \
		--external-schedule $(EXTERNAL_SCHEDULE) \
		--external-kernel-style $(EXTERNAL_KERNEL_STYLE) \
		--external-k-packs $(EXTERNAL_K_PACKS) \
		--external-block-m $(EXTERNAL_BLOCK_M) \
		--external-block-n $(EXTERNAL_BLOCK_N) \
		--external-core-m-packs $(EXTERNAL_CORE_M_PACKS) \
		--external-active-m-packs $(EXTERNAL_ACTIVE_M_PACKS) \
		--external-core-n-packs $(EXTERNAL_CORE_N_PACKS) \
		--external-c-stride-m-packs $(EXTERNAL_C_STRIDE_M_PACKS) \
		--atb-k-chunk-elements $(ATB_K_CHUNK_ELEMENTS) \
		$(OMIT_PING_PONG_ARG) \
		$(PERFORMANCE_MODE_ARG) \
		--acceptance-tops $(ACCEPTANCE_TOPS) \
		--transform-script $(TRANSFORM_SCRIPT) \
		--transformed-ir $(ARTIFACT_DIR)/air_module.final.mlir \
		$(LOWERED_IR_ARG) \
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
