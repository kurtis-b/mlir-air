# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

OUTPUT_FORMAT ?= xclbin
OUTPUT_FORMAT_FLAG := --output-format $(OUTPUT_FORMAT)
COMPILE_MODE ?= compile-and-run

AIEOPT_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)
AIE_API_INCLUDE ?= $(shell \
	if [ -n "$(MLIR_AIE_INSTALL_DIR)" ] && [ -d "$(MLIR_AIE_INSTALL_DIR)/include/aie_api" ]; then \
		echo "$(MLIR_AIE_INSTALL_DIR)/include"; \
	elif [ -n "$(MLIR_AIE_INSTALL_DIR)" ] && [ -d "$(MLIR_AIE_INSTALL_DIR)/lib/python3.12/site-packages/mlir_aie/include/aie_api" ]; then \
		echo "$(MLIR_AIE_INSTALL_DIR)/lib/python3.12/site-packages/mlir_aie/include"; \
	else \
		echo "$(AIEOPT_DIR)/include"; \
	fi)
WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
PEANOWRAP2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIE_API_INCLUDE}
PEANOWRAP2P_OPT_FLAGS = -O3 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -DOPT_PERF_ENABLED -I ${AIE_API_INCLUDE}

Q4NX_ROWS ?= 32
Q4NX_COLS ?= 256

MM_M ?= 128
MM_K ?= 256
MM_N ?= 256
MM_TILE_M ?= 32
MM_TILE_K_L2 ?= 64
MM_TILE_K_L1 ?= 32
MM_TILE_N ?= 32
MM_HERD_M ?= 4
MM_HERD_N ?= 4
MM_TILE_M_DIV_4 := $(shell echo $$(( $(MM_TILE_M) / 4 )))
MM_TILE_N_DIV_4 := $(shell echo $$(( $(MM_TILE_N) / 4 )))
MM_TILE_M_DIV_8 := $(shell echo $$(( $(MM_TILE_M) / 8 )))
MM_TILE_N_DIV_8 := $(shell echo $$(( $(MM_TILE_N) / 8 )))

OPT_MM_M ?= 256
OPT_MM_K ?= 256
OPT_MM_N ?= 256
OPT_MM_TILE_M ?= 32
OPT_MM_TILE_K_L2 ?= 64
OPT_MM_TILE_K_L1 ?= 32
OPT_MM_TILE_N ?= 32
OPT_MM_HERD_M ?= 8
OPT_MM_HERD_N ?= 4
OPT_MM_TILE_M_DIV_4 := $(shell echo $$(( $(OPT_MM_TILE_M) / 4 )))
OPT_MM_TILE_N_DIV_4 := $(shell echo $$(( $(OPT_MM_TILE_N) / 4 )))
OPT_MM_TILE_M_DIV_8 := $(shell echo $$(( $(OPT_MM_TILE_M) / 8 )))
OPT_MM_TILE_N_DIV_8 := $(shell echo $$(( $(OPT_MM_TILE_N) / 8 )))

Q_CHUNK ?= 4
KV_LEN ?= 32
HEAD_DIM ?= 64
KV_CHUNK ?= 16
FLOWQKV_QUERY_BASE ?= 0
FLOWKV_QUERY_BASE ?= $(shell echo $$(( $(KV_LEN) - 1 )))
WINDOW_LEN ?= 0
FLOWQKV_CAUSAL ?= 1
FLOWKV_CAUSAL ?= 1
FLOWQKV_CAUSAL_FLAG := $(if $(filter 1 true yes,$(FLOWQKV_CAUSAL)),--causal,)
FLOWKV_CAUSAL_FLAG := $(if $(filter 1 true yes,$(FLOWKV_CAUSAL)),--causal,)

all: run-q4nx run-mm run-fused-dqp run-flowqkv run-flowkv
all-opt: run-q4nx-opt run-mm-opt run-fused-dqp-opt run-flowqkv-opt run-flowkv-opt
compile-opt-kernels: compile-q4nx-opt-kernel compile-mm-opt-kernel compile-fused-dqp-opt-kernel compile-flowqkv-opt-kernel compile-flowkv-opt-kernel

dump-asm-opt: dump-asm-q4nx-opt dump-asm-mm-opt dump-asm-fused-dqp-opt dump-asm-flowqkv-opt dump-asm-flowkv-opt

check-peano:
	@if [ -z "$(PEANO_INSTALL_DIR)" ] || [ ! -x "$(PEANO_INSTALL_DIR)/bin/clang++" ]; then \
		echo "Error: NPU2 build requires PEANO_INSTALL_DIR with clang++."; \
		exit 1; \
	fi

prepare-air-project:
	mkdir -p $(BUILD_DIR)/air_project

print-q4nx:
	python3 $(srcdir)/q4nx.py -p --rows $(Q4NX_ROWS) --cols $(Q4NX_COLS) $(OUTPUT_FORMAT_FLAG)

compile-q4nx-kernel: check-peano
	mkdir -p $(BUILD_DIR)
	$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} \
		-DQ4NX_ROWS=$(Q4NX_ROWS) -DQ4NX_COLS=$(Q4NX_COLS) \
		-c $(srcdir)/q4nx.cc -o $(BUILD_DIR)/q4nx.o

run-q4nx: compile-q4nx-kernel prepare-air-project
	cp $(BUILD_DIR)/q4nx.o $(BUILD_DIR)/air_project/q4nx.o
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		python3 $(srcdir)/q4nx.py --rows $(Q4NX_ROWS) --cols $(Q4NX_COLS) \
		--compile-mode $(COMPILE_MODE) $(OUTPUT_FORMAT_FLAG)

compile-q4nx-opt-kernel: check-peano
	mkdir -p $(BUILD_DIR)
	$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_OPT_FLAGS} \
		-DQ4NX_ROWS=$(Q4NX_ROWS) -DQ4NX_COLS=$(Q4NX_COLS) \
		-c $(srcdir)/q4nx_opt.cc -o $(BUILD_DIR)/q4nx_opt.o

run-q4nx-opt: compile-q4nx-opt-kernel prepare-air-project
	cp $(BUILD_DIR)/q4nx_opt.o $(BUILD_DIR)/air_project/q4nx_opt.o
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		python3 $(srcdir)/q4nx.py --rows $(Q4NX_ROWS) --cols $(Q4NX_COLS) \
		--kernel-name q4nx_dequant_block_opt --object-file q4nx_opt.o \
		--compile-mode $(COMPILE_MODE) $(OUTPUT_FORMAT_FLAG)

print-fused-dqp:
	python3 $(srcdir)/fused_dqp.py -p --rows $(Q4NX_ROWS) --cols $(Q4NX_COLS) $(OUTPUT_FORMAT_FLAG)

compile-fused-dqp-kernel: check-peano
	mkdir -p $(BUILD_DIR)
	$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} \
		-DQ4NX_ROWS=$(Q4NX_ROWS) -DQ4NX_COLS=$(Q4NX_COLS) \
		-c $(srcdir)/fused_dqp.cc -o $(BUILD_DIR)/fused_dqp.o

run-fused-dqp: compile-fused-dqp-kernel prepare-air-project
	cp $(BUILD_DIR)/fused_dqp.o $(BUILD_DIR)/air_project/fused_dqp.o
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		python3 $(srcdir)/fused_dqp.py --rows $(Q4NX_ROWS) --cols $(Q4NX_COLS) \
		--compile-mode $(COMPILE_MODE) $(OUTPUT_FORMAT_FLAG)

compile-fused-dqp-opt-kernel: check-peano
	mkdir -p $(BUILD_DIR)
	$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_OPT_FLAGS} \
		-DQ4NX_ROWS=$(Q4NX_ROWS) -DQ4NX_COLS=$(Q4NX_COLS) \
		-c $(srcdir)/fused_dqp_opt.cc -o $(BUILD_DIR)/fused_dqp_opt.o

run-fused-dqp-opt: compile-fused-dqp-opt-kernel prepare-air-project
	cp $(BUILD_DIR)/fused_dqp_opt.o $(BUILD_DIR)/air_project/fused_dqp_opt.o
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		python3 $(srcdir)/fused_dqp.py --rows $(Q4NX_ROWS) --cols $(Q4NX_COLS) \
		--kernel-name fused_dqp_block_opt --object-file fused_dqp_opt.o \
		--compile-mode $(COMPILE_MODE) $(OUTPUT_FORMAT_FLAG)

print-mm:
	python3 $(srcdir)/bf16_tiled_mm.py -p --m $(MM_M) --k $(MM_K) --n $(MM_N) \
		--tile-m $(MM_TILE_M) --tile-k-l2 $(MM_TILE_K_L2) --tile-k-l1 $(MM_TILE_K_L1) \
		--tile-n $(MM_TILE_N) --herd-m $(MM_HERD_M) --herd-n $(MM_HERD_N) $(OUTPUT_FORMAT_FLAG)

compile-mm-kernel: check-peano
	mkdir -p $(BUILD_DIR)
	$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 \
		'-Dcombos(X)=X(bfloat16,bf16,bfloat16,bf16,8,8,8)' \
		-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 \
		-DDIM_M=$(MM_TILE_M) -DDIM_N=$(MM_TILE_N) -DDIM_K=$(MM_TILE_K_L1) \
		-DDIM_N_DIV_4=$(MM_TILE_N_DIV_4) -DDIM_M_DIV_4=$(MM_TILE_M_DIV_4) \
		-DDIM_N_DIV_8=$(MM_TILE_N_DIV_8) -DDIM_M_DIV_8=$(MM_TILE_M_DIV_8) \
		-c $(srcdir)/../matrix_multiplication/bf16/mm_aie2p.cc -o $(BUILD_DIR)/mm.o

run-mm: compile-mm-kernel prepare-air-project
	cp $(BUILD_DIR)/mm.o $(BUILD_DIR)/air_project/mm.o
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		python3 $(srcdir)/bf16_tiled_mm.py --m $(MM_M) --k $(MM_K) --n $(MM_N) \
		--tile-m $(MM_TILE_M) --tile-k-l2 $(MM_TILE_K_L2) --tile-k-l1 $(MM_TILE_K_L1) \
		--tile-n $(MM_TILE_N) --herd-m $(MM_HERD_M) --herd-n $(MM_HERD_N) \
		--compile-mode $(COMPILE_MODE) $(OUTPUT_FORMAT_FLAG)

compile-mm-opt-kernel: check-peano
	mkdir -p $(BUILD_DIR)
	$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_OPT_FLAGS} -DBIT_WIDTH=8 \
		'-Dcombos(X)=X(bfloat16,bf16,bfloat16,bf16,8,8,8)' \
		-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 \
		-DDIM_M=$(OPT_MM_TILE_M) -DDIM_N=$(OPT_MM_TILE_N) -DDIM_K=$(OPT_MM_TILE_K_L1) \
		-DDIM_N_DIV_4=$(OPT_MM_TILE_N_DIV_4) -DDIM_M_DIV_4=$(OPT_MM_TILE_M_DIV_4) \
		-DDIM_N_DIV_8=$(OPT_MM_TILE_N_DIV_8) -DDIM_M_DIV_8=$(OPT_MM_TILE_M_DIV_8) \
		-c $(srcdir)/../matrix_multiplication/bf16/mm_aie2p.cc -o $(BUILD_DIR)/mm_opt.o

run-mm-opt: compile-mm-opt-kernel prepare-air-project
	cp $(BUILD_DIR)/mm_opt.o $(BUILD_DIR)/air_project/mm.o
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		python3 $(srcdir)/bf16_tiled_mm.py --m $(OPT_MM_M) --k $(OPT_MM_K) --n $(OPT_MM_N) \
		--tile-m $(OPT_MM_TILE_M) --tile-k-l2 $(OPT_MM_TILE_K_L2) --tile-k-l1 $(OPT_MM_TILE_K_L1) \
		--tile-n $(OPT_MM_TILE_N) --herd-m $(OPT_MM_HERD_M) --herd-n $(OPT_MM_HERD_N) \
		--compile-mode $(COMPILE_MODE) $(OUTPUT_FORMAT_FLAG)

print-flowqkv:
	python3 $(srcdir)/flowqkv.py -p --q-chunk $(Q_CHUNK) --kv-len $(KV_LEN) \
		--head-dim $(HEAD_DIM) --query-base $(FLOWQKV_QUERY_BASE) \
		--window-len $(WINDOW_LEN) $(FLOWQKV_CAUSAL_FLAG) $(OUTPUT_FORMAT_FLAG)

compile-flowqkv-kernel: check-peano
	mkdir -p $(BUILD_DIR)
	$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} \
		-DQ_CHUNK=$(Q_CHUNK) -DKV_LEN=$(KV_LEN) -DHEAD_DIM=$(HEAD_DIM) \
		-DQUERY_BASE=$(FLOWQKV_QUERY_BASE) -DWINDOW_LEN=$(WINDOW_LEN) \
		-DCAUSAL=$(if $(filter 1 true yes,$(FLOWQKV_CAUSAL)),1,0) \
		-c $(srcdir)/flow_attention.cc -o $(BUILD_DIR)/flowqkv.o

run-flowqkv: compile-flowqkv-kernel prepare-air-project
	cp $(BUILD_DIR)/flowqkv.o $(BUILD_DIR)/air_project/flowqkv.o
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		python3 $(srcdir)/flowqkv.py --q-chunk $(Q_CHUNK) --kv-len $(KV_LEN) \
		--head-dim $(HEAD_DIM) --query-base $(FLOWQKV_QUERY_BASE) \
		--window-len $(WINDOW_LEN) $(FLOWQKV_CAUSAL_FLAG) \
		--compile-mode $(COMPILE_MODE) $(OUTPUT_FORMAT_FLAG)

compile-flowqkv-opt-kernel: check-peano
	mkdir -p $(BUILD_DIR)
	$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_OPT_FLAGS} \
		-DQ_CHUNK=$(Q_CHUNK) -DKV_LEN=$(KV_LEN) -DHEAD_DIM=$(HEAD_DIM) \
		-DQUERY_BASE=$(FLOWQKV_QUERY_BASE) -DWINDOW_LEN=$(WINDOW_LEN) \
		-DCAUSAL=$(if $(filter 1 true yes,$(FLOWQKV_CAUSAL)),1,0) \
		-c $(srcdir)/flow_attention_opt.cc -o $(BUILD_DIR)/flowqkv_opt.o

run-flowqkv-opt: compile-flowqkv-opt-kernel prepare-air-project
	cp $(BUILD_DIR)/flowqkv_opt.o $(BUILD_DIR)/air_project/flowqkv_opt.o
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		python3 $(srcdir)/flowqkv.py --q-chunk $(Q_CHUNK) --kv-len $(KV_LEN) \
		--head-dim $(HEAD_DIM) --query-base $(FLOWQKV_QUERY_BASE) \
		--window-len $(WINDOW_LEN) $(FLOWQKV_CAUSAL_FLAG) \
		--kernel-name flowqkv_chunk_bf16_opt --object-file flowqkv_opt.o \
		--compile-mode $(COMPILE_MODE) $(OUTPUT_FORMAT_FLAG)

print-flowkv:
	python3 $(srcdir)/flowkv.py -p --kv-len $(KV_LEN) --head-dim $(HEAD_DIM) \
		--query-base $(FLOWKV_QUERY_BASE) --window-len $(WINDOW_LEN) \
		$(FLOWKV_CAUSAL_FLAG) $(OUTPUT_FORMAT_FLAG)

compile-flowkv-kernel: check-peano
	mkdir -p $(BUILD_DIR)
	$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} \
		-DQ_CHUNK=1 -DKV_LEN=$(KV_LEN) -DHEAD_DIM=$(HEAD_DIM) \
		-DQUERY_BASE=$(FLOWKV_QUERY_BASE) -DWINDOW_LEN=$(WINDOW_LEN) \
		-DCAUSAL=$(if $(filter 1 true yes,$(FLOWKV_CAUSAL)),1,0) \
		-c $(srcdir)/flow_attention.cc -o $(BUILD_DIR)/flowkv.o

run-flowkv: compile-flowkv-kernel prepare-air-project
	cp $(BUILD_DIR)/flowkv.o $(BUILD_DIR)/air_project/flowkv.o
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		python3 $(srcdir)/flowkv.py --kv-len $(KV_LEN) --head-dim $(HEAD_DIM) \
		--query-base $(FLOWKV_QUERY_BASE) --window-len $(WINDOW_LEN) \
		$(FLOWKV_CAUSAL_FLAG) --compile-mode $(COMPILE_MODE) $(OUTPUT_FORMAT_FLAG)

compile-flowkv-opt-kernel: check-peano
	mkdir -p $(BUILD_DIR)
	$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_OPT_FLAGS} \
		-DQ_CHUNK=1 -DKV_LEN=$(KV_LEN) -DHEAD_DIM=$(HEAD_DIM) \
		-DQUERY_BASE=$(FLOWKV_QUERY_BASE) -DWINDOW_LEN=$(WINDOW_LEN) \
		-DCAUSAL=$(if $(filter 1 true yes,$(FLOWKV_CAUSAL)),1,0) \
		-c $(srcdir)/flow_attention_opt.cc -o $(BUILD_DIR)/flowkv_opt.o

run-flowkv-opt: compile-flowkv-opt-kernel prepare-air-project
	cp $(BUILD_DIR)/flowkv_opt.o $(BUILD_DIR)/air_project/flowkv_opt.o
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		python3 $(srcdir)/flowkv.py --kv-len $(KV_LEN) --head-dim $(HEAD_DIM) \
		--query-base $(FLOWKV_QUERY_BASE) --window-len $(WINDOW_LEN) \
		$(FLOWKV_CAUSAL_FLAG) --kernel-name flowkv_decode_bf16_opt --object-file flowkv_opt.o \
		--compile-mode $(COMPILE_MODE) $(OUTPUT_FORMAT_FLAG)

dump-asm-q4nx-opt: compile-q4nx-opt-kernel
	$(PEANO_INSTALL_DIR)/bin/llvm-objdump -d $(BUILD_DIR)/q4nx_opt.o > $(BUILD_DIR)/q4nx_opt.S

dump-asm-fused-dqp-opt: compile-fused-dqp-opt-kernel
	$(PEANO_INSTALL_DIR)/bin/llvm-objdump -d $(BUILD_DIR)/fused_dqp_opt.o > $(BUILD_DIR)/fused_dqp_opt.S

dump-asm-mm-opt: compile-mm-opt-kernel
	$(PEANO_INSTALL_DIR)/bin/llvm-objdump -d $(BUILD_DIR)/mm_opt.o > $(BUILD_DIR)/mm_opt.S

dump-asm-flowqkv-opt: compile-flowqkv-opt-kernel
	$(PEANO_INSTALL_DIR)/bin/llvm-objdump -d $(BUILD_DIR)/flowqkv_opt.o > $(BUILD_DIR)/flowqkv_opt.S

dump-asm-flowkv-opt: compile-flowkv-opt-kernel
	$(PEANO_INSTALL_DIR)/bin/llvm-objdump -d $(BUILD_DIR)/flowkv_opt.o > $(BUILD_DIR)/flowkv_opt.S

clean:
	rm -rf build_peano build_chess __pycache__
