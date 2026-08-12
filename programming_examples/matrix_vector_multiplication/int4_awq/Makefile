# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# int4 AWQ GEMV / GEMV+R examples (packed Q+S+Z BO).
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

OUTPUT_FORMAT ?= elf
OUTPUT_FORMAT_FLAG = --output-format $(OUTPUT_FORMAT)

# Shapes / tiling (overridable). Defaults match the Llama Wo decode GEMV.
M       ?= 2048
K       ?= 2048
GS      ?= 128
M_TILE  ?= 8
K_CHUNK ?= 2048

AIEOPT_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)
WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
PEANOWRAP2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include -D__AIE_API_AIE_ADF_HPP__

PY_ARGS = --m $(M) --k $(K) --gs $(GS) --m-tile $(M_TILE) --k-chunk $(K_CHUNK)

all: run_packed

print_packed:
	${powershell} python3 ${srcdir}/matvec_int4_packed.py $(OUTPUT_FORMAT_FLAG) -p $(PY_ARGS)

print_packed_add:
	${powershell} python3 ${srcdir}/matvec_int4_packed_add.py $(OUTPUT_FORMAT_FLAG) -p $(PY_ARGS)

compile-kernel:
	mkdir -p $(BUILD_DIR)
	@if [ -z "$(PEANO_INSTALL_DIR)" ]; then \
		echo "Error: PEANO_INSTALL_DIR not set (source utils/env_setup.sh)."; \
		exit 1; \
	fi
	$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} \
		-DDIM_M=$(M_TILE) -DDIM_K=$(K_CHUNK) -DDIM_GS=$(GS) \
		-c ${srcdir}/mv_int4_bf16.cc -o $(BUILD_DIR)/mv_int4_bf16.o

# GEMV: D = dequant(A) @ B
run_packed: compile-kernel
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
	  ${powershell} python3 ${srcdir}/matvec_int4_packed.py $(OUTPUT_FORMAT_FLAG) $(PY_ARGS)

# --- llama.cpp q4_0 -------------------------------------------------------
# q4_0 is the same kernel with the zero-point pinned to 8 and gs pinned to 32,
# so GS is forced here rather than inherited. Point GGUF at a checkpoint to run
# on real weights; without it the harness synthesizes a q4_0 payload.
#   make run_q4_0 GGUF=/path/model.gguf TENSOR=blk.0.attn_q.weight
#   make run_q4_0_negative NEG=no-uninterleave GGUF=...
GGUF   ?=
TENSOR ?= blk.0.attn_q.weight
NEG    ?= no-uninterleave
Q4_0_SRC = $(if $(GGUF),--gguf $(GGUF) --tensor $(TENSOR),--m $(M) --k $(K))
Q4_0_ARGS = --m-tile $(M_TILE) --k-chunk $(K_CHUNK) $(Q4_0_SRC)

selftest_q4_0:
	${powershell} python3 ${srcdir}/gguf_q4_0.py --self-test

run_q4_0:
	$(MAKE) -f $(firstword $(MAKEFILE_LIST)) compile-kernel GS=32
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
	  ${powershell} python3 ${srcdir}/matvec_q4_0.py $(OUTPUT_FORMAT_FLAG) \
	    --check-reference $(Q4_0_ARGS)

# The gate is only worth its PASS if a broken pack fails. Inverted exit code:
# this target succeeds when the device run does NOT match.
run_q4_0_negative:
	$(MAKE) -f $(firstword $(MAKEFILE_LIST)) compile-kernel GS=32
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
	  ${powershell} python3 ${srcdir}/matvec_q4_0.py $(OUTPUT_FORMAT_FLAG) \
	    --negative-control $(NEG) $(Q4_0_ARGS)

# GEMV + fused residual add (LA-style 2-tile cascade)
run_packed_add: compile-kernel
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
	  ${powershell} python3 ${srcdir}/matvec_int4_packed_add.py $(OUTPUT_FORMAT_FLAG) $(PY_ARGS)

# ELF profile harnesses (compile module, then run XRT benchmark loop)
build-test-packed-exe:
	@$(MAKE) build-test-exe-impl SRC=test_packed.cpp OUT=test_packed.exe

build-test-packed-add-exe:
	@$(MAKE) build-test-exe-impl SRC=test_packed_add.cpp OUT=test_packed_add.exe

profile_packed: build-test-packed-exe compile-kernel
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
	  ${powershell} python3 ${srcdir}/matvec_int4_packed.py --compile-mode compile-only --output-format elf $(PY_ARGS)
	cd $(BUILD_DIR) && ./test_packed.exe -e air.elf -k "main:matvec_int4_packed" \
	  -M $(M) -K $(K) -G $(GS) -T $(M_TILE) -C $(K_CHUNK)

profile_packed_add: build-test-packed-add-exe compile-kernel
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
	  ${powershell} python3 ${srcdir}/matvec_int4_packed_add.py --compile-mode compile-only --output-format elf $(PY_ARGS)
	cd $(BUILD_DIR) && ./test_packed_add.exe -e air.elf -k "main:matvec_int4_packed_add" \
	  -M $(M) -K $(K) -G $(GS) -T $(M_TILE) -C $(K_CHUNK)

build-test-exe-impl:
	@GPP=$$( \
		for bin in /usr/bin/g++-*; do \
			ver=$$(echo $$bin | grep -oE '[0-9]+$$'); \
			if [ "$$ver" -ge 13 ] 2>/dev/null; then \
				echo "$$ver $$bin"; \
			fi; \
		done | sort -nr | head -n1 | awk '{print $$2}' \
	); \
	if [ -z "$$GPP" ]; then \
		echo "Error: No g++ version >= 13 found in /usr/bin."; exit 1; \
	fi; \
	if [ -z "$$XILINX_XRT" ]; then \
		echo "Error: XILINX_XRT not set (source xrt/setup.sh)."; exit 1; \
	fi; \
	if [ -z "$(AIEOPT_DIR)" ]; then \
		echo "Error: AIEOPT_DIR unset (source utils/env_setup.sh)."; exit 1; \
	fi; \
	mkdir -p $(BUILD_DIR); \
	cd $(BUILD_DIR) && $$GPP ${srcdir}/$(SRC) -o $(OUT) -std=c++23 -Wall \
		-I$$XILINX_XRT/include -L$$XILINX_XRT/lib \
		-I$(AIEOPT_DIR)/runtime_lib/x86_64/test_lib/include \
		-L$(AIEOPT_DIR)/runtime_lib/x86_64/test_lib/lib \
		-luuid -lxrt_coreutil -lrt -lstdc++ -ltest_utils

clean:
	rm -rf $(BUILD_DIR) __pycache__
