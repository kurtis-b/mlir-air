# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
this_makefile := $(realpath $(firstword $(MAKEFILE_LIST)))
repo_root := $(realpath $(srcdir)/../..)
PYTHON ?= python3
PYTHON_VERSION := $(shell $(PYTHON) -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_SITE_PACKAGES ?= $(repo_root)/sandbox/lib/python$(PYTHON_VERSION)/site-packages
MLIR_AIR_PYTHON ?= $(repo_root)/build-xrt/python
MLIR_AIE_PYTHON_ROOT ?= $(PYTHON_SITE_PACKAGES)/mlir_aie
MLIR_AIE_PYTHON ?= $(MLIR_AIE_PYTHON_ROOT)/python
GEMMA3_PYTHONPATH = $(srcdir):$(MLIR_AIR_PYTHON):$(PYTHON_SITE_PACKAGES):$(MLIR_AIE_PYTHON):$${PYTHONPATH}
RUN_PYTHON = PYTHONPATH=$(GEMMA3_PYTHONPATH) $(PYTHON)

ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

OUTPUT_FORMAT ?= elf
OUTPUT_FORMAT_FLAG := --output-format $(OUTPUT_FORMAT)
DEBUG_IR ?= 0
DEBUG_IR_FLAG := $(if $(filter 1 true yes,$(DEBUG_IR)),--debug-ir,)
COMPILE_MODE ?= compile-only
HERD_SHAPE ?= 2x4
HERD_DIMS := $(subst x, ,$(HERD_SHAPE))
HERD_ROWS := $(word 1,$(HERD_DIMS))
HERD_COLS := $(word 2,$(HERD_DIMS))

AIEOPT_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)
AIE_API_INCLUDE ?= $(shell \
	if [ -n "$(MLIR_AIE_INSTALL_DIR)" ] && [ -d "$(MLIR_AIE_INSTALL_DIR)/include/aie_api" ]; then \
		echo "$(MLIR_AIE_INSTALL_DIR)/include"; \
	elif [ -n "$(MLIR_AIE_INSTALL_DIR)" ] && [ -d "$(MLIR_AIE_INSTALL_DIR)/lib/python$(PYTHON_VERSION)/site-packages/mlir_aie/include/aie_api" ]; then \
		echo "$(MLIR_AIE_INSTALL_DIR)/lib/python$(PYTHON_VERSION)/site-packages/mlir_aie/include"; \
	elif [ -d "$(MLIR_AIE_PYTHON_ROOT)/include/aie_api" ]; then \
		echo "$(MLIR_AIE_PYTHON_ROOT)/include"; \
	else \
		echo "$(AIEOPT_DIR)/include"; \
	fi)
WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
PEANOWRAP2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIE_API_INCLUDE}
PEANOWRAP2P_OPT_FLAGS = -O3 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -DOPT_PERF_ENABLED -I ${AIE_API_INCLUDE}

Q4NX_ROWS ?= 32
Q4NX_COLS ?= 256
MM_TILE_M ?= 32
MM_TILE_K_L2 ?= 64
MM_TILE_K_L1 ?= 32
MM_TILE_N ?= 32
MM_M ?= $(shell echo $$(( $(HERD_ROWS) * $(MM_TILE_M) )))
MM_K ?= 256
MM_N ?= $(shell echo $$(( $(HERD_COLS) * $(MM_TILE_N) )))
MM_TILE_M_DIV_4 := $(shell echo $$(( $(MM_TILE_M) / 4 )))
MM_TILE_N_DIV_4 := $(shell echo $$(( $(MM_TILE_N) / 4 )))
MM_TILE_M_DIV_8 := $(shell echo $$(( $(MM_TILE_M) / 8 )))
MM_TILE_N_DIV_8 := $(shell echo $$(( $(MM_TILE_N) / 8 )))

Q_CHUNK ?= 4
KV_LEN ?= 32
KV_CHUNK ?= 32
HEAD_DIM ?= 64
FLOWQKV_QUERY_BASE ?= 0
FLOWKV_QUERY_BASE ?= $(shell echo $$(( $(KV_LEN) - 1 )))
WINDOW_LEN ?= 0
FLOWQKV_CAUSAL ?= 1
FLOWKV_CAUSAL ?= 1
FLOWQKV_VARIANT ?= causal
FLOWKV_VARIANT ?= causal
FLOW_HERD_ROWS ?= 1
FLOW_HERD_COLS ?= 1
FLOWQKV_CAUSAL_FLAG := $(if $(filter 1 true yes,$(FLOWQKV_CAUSAL)),--causal,)
FLOWKV_CAUSAL_FLAG := $(if $(filter 1 true yes,$(FLOWKV_CAUSAL)),--causal,)

Q4NX_OUTPUT_MODE ?= auto
FUSED_DQP_OUTPUT_MODE ?= auto
FLOWQKV_OUTPUT_MODE ?= auto
FLOWKV_OUTPUT_MODE ?= auto
FUSED_DQP_SCHEDULE_MODE ?= smoke
FLOWQKV_SCHEDULE_MODE ?= smoke
FLOWKV_SCHEDULE_MODE ?= smoke
FLOWQKV_KV_STAGING ?= pipeline
FLOWKV_KV_STAGING ?= pipeline
FUSED_DQP_COL_BLOCKS ?= 1
FUSED_DQP_COL_CHUNK ?= 32
FUSED_DQP_PIPE_ROW_CHUNK ?= 16
PAPER_FUSED_DQP_COL_BLOCKS ?= 2
PAPER_KV_GROUPS ?= 4
PAPER_HEADS_PER_KV ?= 2
PAPER_Q_CHUNK ?= $(Q_CHUNK)
PAPER_KV_LEN ?= $(KV_LEN)
PAPER_KV_CHUNK ?= 16
PAPER_HEAD_DIM ?= $(HEAD_DIM)
SWEEP_OUTPUT_MODES ?= auto
SWEEP_OUTPUT_MODE_FLAGS := $(foreach mode,$(SWEEP_OUTPUT_MODES),--output-mode $(mode))
OPT_MM_M ?= 256
OPT_MM_K ?= 256
OPT_MM_N ?= 256
OPT_HERD_SHAPE ?= 8x4
GEGLU_N ?= 6912
GEGLU_TILE_N ?= 288
RESIDUAL_N ?= 1152
RESIDUAL_TILE_N ?= 288
ROPE_ROWS ?= 4
ROPE_HEAD_DIM ?= 256
ROPE_HERD_X ?= 4
FLOWQKV_TILED_STATS_OUTPUT_MODE ?= direct
FLOWQKV_TILED_STATS_HOST_BATCH_TILES ?= 0

all: run-q4nx run-mm run-fused-dqp run-flowqkv run-flowkv
all-opt: run-q4nx-opt run-mm-opt run-fused-dqp-opt run-flowqkv-opt run-flowkv-opt
sweep:
	$(RUN_PYTHON) -m gemma3.kernels.sweep_herds --compile-mode $(COMPILE_MODE) --output-format $(OUTPUT_FORMAT) $(SWEEP_OUTPUT_MODE_FLAGS)

check-peano:
	@if [ -z "$(PEANO_INSTALL_DIR)" ] || [ ! -x "$(PEANO_INSTALL_DIR)/bin/clang++" ]; then \
		echo "Error: NPU2 build requires PEANO_INSTALL_DIR with clang++."; \
		exit 1; \
	fi

prepare-air-project:
	mkdir -p $(BUILD_DIR)/air_project

compile-q4nx-kernel: check-peano
	mkdir -p $(BUILD_DIR)
	$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_OPT_FLAGS} \
		-DQ4NX_ROWS=$(Q4NX_ROWS) -DQ4NX_COLS=$(Q4NX_COLS) \
		-c $(srcdir)/aie_kernels/q4nx_opt.cc -o $(BUILD_DIR)/q4nx.o

run-q4nx: compile-q4nx-kernel prepare-air-project
	cp $(BUILD_DIR)/q4nx.o $(BUILD_DIR)/air_project/q4nx.o
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		$(RUN_PYTHON) -m gemma3.kernels.q4nx --rows $(Q4NX_ROWS) --cols $(Q4NX_COLS) \
		--herd-shape $(HERD_SHAPE) --kernel-name q4nx_dequant_block_opt \
		--object-file q4nx.o --compile-mode $(COMPILE_MODE) $(OUTPUT_FORMAT_FLAG) \
		$(DEBUG_IR_FLAG) --output-mode $(Q4NX_OUTPUT_MODE)

# Logical 8x4 Q4NX workload using two host-side physical 4x4 row bands.
run-q4nx-8x4-rowband-fallback: compile-q4nx-kernel prepare-air-project
	cp $(BUILD_DIR)/q4nx.o $(BUILD_DIR)/air_project/q4nx.o
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		$(RUN_PYTHON) -m gemma3.kernels.q4nx --rows $(Q4NX_ROWS) --cols $(Q4NX_COLS) \
		--herd-shape 4x4 --row-blocks 4 --col-blocks 4 --repeat-row-bands 2 \
		--kernel-name q4nx_dequant_block_opt --object-file q4nx.o \
		--compile-mode $(COMPILE_MODE) $(OUTPUT_FORMAT_FLAG) $(DEBUG_IR_FLAG) --output-mode l2-gather

compile-fused-dqp-kernel: check-peano
	mkdir -p $(BUILD_DIR)
	$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_OPT_FLAGS} \
		-DQ4NX_ROWS=$(Q4NX_ROWS) -DQ4NX_COLS=$(Q4NX_COLS) \
		-DFUSED_DQP_COL_BLOCKS=$(FUSED_DQP_COL_BLOCKS) \
		-DFUSED_DQP_COL_CHUNK=$(FUSED_DQP_COL_CHUNK) \
		-DFUSED_DQP_PIPE_ROW_CHUNK=$(FUSED_DQP_PIPE_ROW_CHUNK) \
		-c $(srcdir)/aie_kernels/fused_dqp_opt.cc -o $(BUILD_DIR)/fused_dqp.o

run-fused-dqp: compile-fused-dqp-kernel prepare-air-project
	cp $(BUILD_DIR)/fused_dqp.o $(BUILD_DIR)/air_project/fused_dqp.o
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		$(RUN_PYTHON) -m gemma3.kernels.fused_dqp --rows $(Q4NX_ROWS) --cols $(Q4NX_COLS) \
		--herd-shape $(HERD_SHAPE) --kernel-name fused_dqp_block_opt \
		--object-file fused_dqp.o --compile-mode $(COMPILE_MODE) $(OUTPUT_FORMAT_FLAG) \
		$(DEBUG_IR_FLAG) --output-mode $(FUSED_DQP_OUTPUT_MODE) --schedule-mode $(FUSED_DQP_SCHEDULE_MODE) \
		--col-blocks $(FUSED_DQP_COL_BLOCKS) --col-chunk $(FUSED_DQP_COL_CHUNK) \
		--pipe-row-chunk $(FUSED_DQP_PIPE_ROW_CHUNK)

run-fused-dqp-paper:
	$(MAKE) -f $(this_makefile) run-fused-dqp HERD_SHAPE=4x4 FUSED_DQP_SCHEDULE_MODE=paper \
		FUSED_DQP_OUTPUT_MODE=l2-gather FUSED_DQP_COL_BLOCKS=$(PAPER_FUSED_DQP_COL_BLOCKS)

run-fused-dqp-pipeline:
	$(MAKE) -f $(this_makefile) run-fused-dqp HERD_SHAPE=8x4 FUSED_DQP_SCHEDULE_MODE=pipeline \
		FUSED_DQP_OUTPUT_MODE=l2-gather FUSED_DQP_COL_BLOCKS=$(PAPER_FUSED_DQP_COL_BLOCKS) \
		FUSED_DQP_COL_CHUNK=8 FUSED_DQP_PIPE_ROW_CHUNK=16

compile-mm-kernel: check-peano
	mkdir -p $(BUILD_DIR)
	$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_OPT_FLAGS} -DBIT_WIDTH=8 \
		'-Dcombos(X)=X(bfloat16,bf16,bfloat16,bf16,8,8,8)' \
		-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 \
		-DDIM_M=$(MM_TILE_M) -DDIM_N=$(MM_TILE_N) -DDIM_K=$(MM_TILE_K_L1) \
		-DDIM_N_DIV_4=$(MM_TILE_N_DIV_4) -DDIM_M_DIV_4=$(MM_TILE_M_DIV_4) \
		-DDIM_N_DIV_8=$(MM_TILE_N_DIV_8) -DDIM_M_DIV_8=$(MM_TILE_M_DIV_8) \
		-c $(srcdir)/../matrix_multiplication/bf16/mm_aie2p.cc -o $(BUILD_DIR)/mm.o

run-mm: compile-mm-kernel prepare-air-project
	cp $(BUILD_DIR)/mm.o $(BUILD_DIR)/air_project/mm.o
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		$(RUN_PYTHON) -m gemma3.kernels.bf16_tiled_mm --m $(MM_M) --k $(MM_K) --n $(MM_N) \
		--tile-m $(MM_TILE_M) --tile-k-l2 $(MM_TILE_K_L2) --tile-k-l1 $(MM_TILE_K_L1) \
		--tile-n $(MM_TILE_N) --herd-shape $(HERD_SHAPE) \
		--compile-mode $(COMPILE_MODE) $(OUTPUT_FORMAT_FLAG)

compile-flowqkv-kernel: check-peano
	mkdir -p $(BUILD_DIR)
	$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_OPT_FLAGS} \
		-DQ_CHUNK=$(Q_CHUNK) -DKV_LEN=$(KV_LEN) -DKV_CHUNK=$(KV_CHUNK) -DHEAD_DIM=$(HEAD_DIM) \
		-DQUERY_BASE=$(FLOWQKV_QUERY_BASE) -DWINDOW_LEN=$(WINDOW_LEN) \
		-DCAUSAL=$(if $(filter 1 true yes,$(FLOWQKV_CAUSAL)),1,0) \
		-c $(srcdir)/aie_kernels/flow_attention_opt.cc -o $(BUILD_DIR)/flowqkv.o

run-flowqkv: compile-flowqkv-kernel prepare-air-project
	cp $(BUILD_DIR)/flowqkv.o $(BUILD_DIR)/air_project/flowqkv.o
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		$(RUN_PYTHON) -m gemma3.kernels.flowqkv --q-chunk $(Q_CHUNK) --kv-len $(KV_LEN) \
		--kv-chunk $(KV_CHUNK) --head-dim $(HEAD_DIM) --herd-shape $(HERD_SHAPE) \
		--query-base $(FLOWQKV_QUERY_BASE) --window-len $(WINDOW_LEN) \
		$(FLOWQKV_CAUSAL_FLAG) --kernel-name flowqkv_chunk_bf16_opt \
		--object-file flowqkv.o --compile-mode $(COMPILE_MODE) $(OUTPUT_FORMAT_FLAG) \
		$(DEBUG_IR_FLAG) --output-mode $(FLOWQKV_OUTPUT_MODE) --schedule-mode $(FLOWQKV_SCHEDULE_MODE) \
		--kv-groups $(PAPER_KV_GROUPS) --heads-per-kv $(PAPER_HEADS_PER_KV) \
		--kv-staging $(FLOWQKV_KV_STAGING) --variant $(FLOWQKV_VARIANT)

compile-flowkv-kernel: check-peano
	mkdir -p $(BUILD_DIR)
	$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_OPT_FLAGS} \
		-DQ_CHUNK=1 -DKV_LEN=$(KV_LEN) -DKV_CHUNK=$(KV_CHUNK) -DHEAD_DIM=$(HEAD_DIM) \
		-DQUERY_BASE=$(FLOWKV_QUERY_BASE) -DWINDOW_LEN=$(WINDOW_LEN) \
		-DCAUSAL=$(if $(filter 1 true yes,$(FLOWKV_CAUSAL)),1,0) \
		-c $(srcdir)/aie_kernels/flow_attention_opt.cc -o $(BUILD_DIR)/flowkv.o

run-flowkv: compile-flowkv-kernel prepare-air-project
	cp $(BUILD_DIR)/flowkv.o $(BUILD_DIR)/air_project/flowkv.o
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		$(RUN_PYTHON) -m gemma3.kernels.flowkv --kv-len $(KV_LEN) --kv-chunk $(KV_CHUNK) --head-dim $(HEAD_DIM) \
		--herd-shape $(HERD_SHAPE) --query-base $(FLOWKV_QUERY_BASE) \
		--window-len $(WINDOW_LEN) $(FLOWKV_CAUSAL_FLAG) \
		--kernel-name flowkv_decode_bf16_opt --object-file flowkv.o \
		--compile-mode $(COMPILE_MODE) $(OUTPUT_FORMAT_FLAG) \
		$(DEBUG_IR_FLAG) --output-mode $(FLOWKV_OUTPUT_MODE) --schedule-mode $(FLOWKV_SCHEDULE_MODE) \
		--kv-groups $(PAPER_KV_GROUPS) --heads-per-kv $(PAPER_HEADS_PER_KV) \
		--kv-staging $(FLOWKV_KV_STAGING) --variant $(FLOWKV_VARIANT)

run-flowqkv-paper:
	$(MAKE) -f $(this_makefile) run-flowqkv HERD_SHAPE=8x4 FLOWQKV_SCHEDULE_MODE=paper \
		FLOWQKV_OUTPUT_MODE=l2-gather Q_CHUNK=$(PAPER_Q_CHUNK) KV_LEN=$(PAPER_KV_LEN) \
		KV_CHUNK=$(PAPER_KV_CHUNK) HEAD_DIM=$(PAPER_HEAD_DIM)

run-flowkv-paper:
	$(MAKE) -f $(this_makefile) run-flowkv HERD_SHAPE=2x4 FLOWKV_SCHEDULE_MODE=paper \
		FLOWKV_OUTPUT_MODE=direct KV_LEN=$(PAPER_KV_LEN) KV_CHUNK=$(PAPER_KV_CHUNK) \
		HEAD_DIM=$(PAPER_HEAD_DIM)

run-q4nx-opt: run-q4nx

run-fused-dqp-opt: run-fused-dqp

run-flowqkv-opt: run-flowqkv

run-flowkv-opt: run-flowkv

run-mm-opt:
	$(MAKE) -f $(this_makefile) run-mm HERD_SHAPE=$(OPT_HERD_SHAPE) \
		MM_M=$(OPT_MM_M) MM_K=$(OPT_MM_K) MM_N=$(OPT_MM_N)

run-geglu: prepare-air-project
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		$(RUN_PYTHON) -m gemma3.kernels.geglu \
		--compile-mode $(COMPILE_MODE) $(OUTPUT_FORMAT_FLAG) --n $(GEGLU_N) --tile-n $(GEGLU_TILE_N)

run-residual-add: prepare-air-project
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		$(RUN_PYTHON) -m gemma3.kernels.residual_add \
		--compile-mode $(COMPILE_MODE) $(OUTPUT_FORMAT_FLAG) --n $(RESIDUAL_N) --tile-n $(RESIDUAL_TILE_N)

run-rope-halfsplit: prepare-air-project
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		$(RUN_PYTHON) -m gemma3.kernels.rope_halfsplit \
		--compile-mode $(COMPILE_MODE) $(OUTPUT_FORMAT_FLAG) --rows $(ROPE_ROWS) \
		--head-dim $(ROPE_HEAD_DIM) --herd-x $(ROPE_HERD_X)

compile-flowqkv-tiled-stats-kernel: check-peano
	mkdir -p $(BUILD_DIR)
	$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} \
		-DQ_CHUNK=$(Q_CHUNK) -DKV_TILE=$(KV_CHUNK) -DHEAD_DIM=$(HEAD_DIM) \
		-c $(srcdir)/aie_kernels/flow_attention_stats.cc -o $(BUILD_DIR)/flowqkv_tiled_stats.o

run-flowqkv-tiled-stats: compile-flowqkv-tiled-stats-kernel prepare-air-project
	cp $(BUILD_DIR)/flowqkv_tiled_stats.o $(BUILD_DIR)/air_project/flowqkv_tiled_stats.o
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		$(RUN_PYTHON) -m gemma3.kernels.flowqkv_tiled_stats \
		--q-chunk $(Q_CHUNK) --kv-len $(KV_LEN) --kv-tile $(KV_CHUNK) \
		--head-dim $(HEAD_DIM) --herd-rows $(FLOW_HERD_ROWS) --herd-cols $(FLOW_HERD_COLS) \
		--host-batch-tiles $(FLOWQKV_TILED_STATS_HOST_BATCH_TILES) --kernel-name flowqkv_tile_stats_bf16 \
		--object-file flowqkv_tiled_stats.o --compile-mode $(COMPILE_MODE) $(OUTPUT_FORMAT_FLAG) \
		--output-mode $(FLOWQKV_TILED_STATS_OUTPUT_MODE)

profile-paper: run-fused-dqp-paper run-flowqkv-paper run-flowkv-paper

clean:
	rm -rf build_peano build_chess __pycache__
