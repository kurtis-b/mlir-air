# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

CXX ?= g++
BUILD_DIR ?= build
TARGET := $(BUILD_DIR)/int8_gemm_cpu

CXXFLAGS ?= -O3 -g -std=c++17 -Wall -Wextra -march=native -fno-omit-frame-pointer -pthread
SIZE ?= 1024
WARMUPS ?= 0
ITERATIONS ?= 1
THREADS ?= 12
EXTRA_ARGS ?=

.PHONY: all clean run

all: $(TARGET)

$(TARGET): int8_gemm.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD_DIR):
	mkdir -p $@

run: $(TARGET)
	$(TARGET) --size $(SIZE) --warmups $(WARMUPS) --iterations $(ITERATIONS) --threads $(THREADS) $(EXTRA_ARGS)

clean:
	rm -rf $(BUILD_DIR)
