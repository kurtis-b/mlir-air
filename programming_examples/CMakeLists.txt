# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.10)

if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

project(air LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(MLIR REQUIRED CONFIG)
find_package(Python3 COMPONENTS Interpreter)

if(AIR_ENABLE_AIE)
  find_package(LibXAIE)
  if(NOT AIE_BINARY_DIR)
    find_package(AIE REQUIRED CONFIG)
  endif()
  find_package(Vitis 2023.2 COMPONENTS AIE AIE2)
endif()

if(NOT AIR_BINARY_DIR)
  find_package(AIR REQUIRED CONFIG)
endif()

find_package(XRT)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
if(AIR_ENABLE_AIE)
  message(STATUS "Using AIEConfig.cmake in: ${AIE_DIR}")
endif()
message(STATUS "Using AIRConfig.cmake in: ${AIR_DIR}")

option(LLVM_INCLUDE_TOOLS "Generate build targets for the LLVM tools." ON)
option(LLVM_BUILD_TOOLS "Build the LLVM tools. If OFF, just generate build targets." ON)

# ON: build and run end-to-end tests
# OFF: build only
option(ENABLE_RUN_XRT_TESTS "Run end-to-end tests on Ryzen AI with XRT" OFF)

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
add_definitions(${LLVM_DEFINITIONS})

set(ENABLE_CHESS_TESTS ON CACHE BOOL "Enable backend tests using xchesscc")

set(AIR_TEST_LIT_ARGS "-sv --timeout 1000 --time-tests --order=random")

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

set(TEST_DEPENDS
  FileCheck count not
  )

# Peano-only test suite (non-LLM peano tests; filtered by filename suffix).
# LLM tests are handled by the dedicated targets below so they can use -j1
# (sequential) to avoid concurrent NPU inference OOM crashes.
add_lit_testsuite(check-programming-examples-peano "Running AIR programming examples (Peano backend only)"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TEST_DEPENDS}
  ARGS ${AIR_TEST_LIT_ARGS} --filter "peano" --filter-out "llms/"
)
set_target_properties(check-programming-examples-peano PROPERTIES FOLDER "Tests")

# Chess-only test suite
add_lit_testsuite(check-programming-examples-chess "Running AIR programming examples (Chess backend only)"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TEST_DEPENDS}
  ARGS ${AIR_TEST_LIT_ARGS} --filter "chess"
)
set_target_properties(check-programming-examples-chess PROPERTIES FOLDER "Tests")

# Unfiltered full test suite
add_lit_testsuite(check-programming-examples "Running AIR programming examples (all backends)"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TEST_DEPENDS}
  ARGS ${AIR_TEST_LIT_ARGS}
)
set_target_properties(check-programming-examples PROPERTIES FOLDER "Tests")

# LLM compile-only suite: all models, no HF download, no NPU dispatch.
# Matched by path (llms/.*/run_npu2_compile) so lit filename need NOT carry
# any backend suffix. Sequential (-j1) because concurrent LLM compilations
# can exhaust host memory on CI runners.
add_lit_testsuite(check-programming-examples-llms-compile
  "Running LLM compile-only tests (all models, sequential)"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TEST_DEPENDS}
  ARGS ${AIR_TEST_LIT_ARGS} -j1 --filter "llms/.*/run_npu2_compile"
)
set_target_properties(check-programming-examples-llms-compile PROPERTIES FOLDER "Tests")

# LLM compile-only suite: llama32_1b only (fast PR gate, always runs in CI).
add_lit_testsuite(check-programming-examples-llms-compile-llama32_1b
  "Running LLM compile test: llama32_1b only (PR gate)"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TEST_DEPENDS}
  ARGS ${AIR_TEST_LIT_ARGS} -j1 --filter "llms/llama32_1b/run_npu2_compile"
)
set_target_properties(check-programming-examples-llms-compile-llama32_1b PROPERTIES FOLDER "Tests")

# LLM verify suite: all models, requires hf_token + NPU.
# Sequential (-j1) mandatory — concurrent LLM inference causes NPU OOM.
# Not invoked in PR CI (hf_token unavailable in fork PRs); run by repo
# owners via workflow_dispatch or post-merge CI with secrets available.
add_lit_testsuite(check-programming-examples-llms-verify
  "Running LLM end-to-end verify tests (all models, hf_token required, sequential)"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TEST_DEPENDS}
  ARGS ${AIR_TEST_LIT_ARGS} -j1 --filter "llms/.*/run_npu2_verify"
)
set_target_properties(check-programming-examples-llms-verify PROPERTIES FOLDER "Tests")

# LLM profiling suite: all models, requires hf_token + NPU. Compiles then
# profiles each model, emitting <model>.perf.json (TTFT + decode tok/s) next to
# the run. Sequential (-j1) — concurrent LLM inference causes NPU OOM. Not a PR
# gate (needs secrets + the dedicated perf runner); driven by the nightly perf
# workflow. Correctness is gated separately by the verify suite above.
add_lit_testsuite(check-programming-examples-llms-profile
  "Running LLM profiling (all models, hf_token required, sequential)"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TEST_DEPENDS}
  ARGS ${AIR_TEST_LIT_ARGS} -j1 --filter "llms/.*/run_npu2_profile"
)
set_target_properties(check-programming-examples-llms-profile PROPERTIES FOLDER "Tests")

# Transformer-layer execution-study suite: THE WHOLE SUITE, AND IT NEEDS AN NPU.
#
# `[2026-08-12]` This comment used to read "Peano object builds plus a symbol
# check, no NPU dispatch and no HF download. Safe as a PR gate on any runner
# that has Peano". That was true through Phases A and B and has been false since
# Phase C1: the suite now carries the runlist gate and every operator gate.
# Measured on the discovered test list, not inferred: 32 tests, of which
# 22 carry `REQUIRES: ryzen_ai_npu2`, 1 is Peano-only and 9 are host-only.
#
# Wiring THIS target into PR CI on the strength of the old comment would produce
# the worst available outcome: on a runner with no NPU2 all 22 report
# UNSUPPORTED and lit still exits 0 -- a hardware gate that runs zero hardware
# tests and reports success. Use it as the local regression gate (which is what
# agents/scripts/port-loop drives it as, with lib-guard.sh asserting that Passed
# reaches the tracked .lit count and that Passed+Excluded are the only nonzero
# categories); use the -host target below for CI.
#
# Enrolment is path-based, so a new .lit anywhere under the example joins this
# suite with no CMake change.
#
# Not `-j1`: the suite has been run at 24 workers and passed (30/30 in 519.7 s).
# But doc 30 records run_npu2_runlist_gate.lit's latency clause going
# intermittent under exactly that contention, so if this target is ever used as
# a measurement-adjacent gate rather than a correctness one, run it -j1.
add_lit_testsuite(check-programming-examples-transformer-layer
  "Running the transformer-layer execution-study suite (NEEDS AN NPU: 22 of 32)"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TEST_DEPENDS}
  ARGS ${AIR_TEST_LIT_ARGS} --filter "transformer_layer/"
)
set_target_properties(check-programming-examples-transformer-layer PROPERTIES FOLDER "Tests")

# Transformer-layer PR gate: the subset that dispatches nothing.
#
# 11 tests -- run_npu2_compile_peano (Peano objects plus a symbol check) and the
# ten host-only ones: the study host suite, the seam/reference/block-cache/
# blocked-attention/FFN-resident emulation checks, the mapping-space census, and
# the sweep's family, writer and registry-resolution checks. No NPU, no XRT
# dispatch, no HF token, no secrets.
#
# `run_mapping_space_tests` is the slowest of them at about a minute: it walks
# the whole declared mapping space and pins the two space sizes exactly. It runs
# no compiler and touches no device -- the cost is Python enumerating ~10^5
# (structure, seam) pairs, which is the measurement itself.
#
# AN EXPLICIT ALLOWLIST, NOT A FILTER-OUT, AND THE DIRECTION IS THE POINT.
# A negative filter would enrol every FUTURE .lit automatically, so a new
# NPU-gated test would silently join the PR gate and report UNSUPPORTED while
# the suite exits 0 -- the failure this file already carries a warning about.
# An allowlist fails the other way: a new host-only test is simply not gated
# until someone adds it here, which is visible and harmless. Add it here in the
# same commit that adds the test.
#
# `run_npu2_registry_resolution` carries `npu2` in its name because that is the
# TARGET ITS ROWS DESCRIBE -- kernel_registry is an NPU2-only registry. The
# check itself resolves rows through the production builders and dispatches
# nothing; its own header says adding a hardware requirement "would only stop
# the regression from surfacing on a compile-only runner".
#
# `[.]lit$` rather than `\.lit$`: CMake eats a backslash before a
# non-escapable character, so the anchor would silently become `.lit$`.
#
# `[2026-08-14]` THE FILTER PASSES THROUGH THREE PARSERS AND EACH ONE WANTED
# SOMETHING DIFFERENT. This target could not run at all before this commit --
# `/bin/sh: 1: Syntax error: "(" unexpected` -- and the breakage predates the
# change that found it (verified by stashing: the 11-test form fails
# identically). It was green once, at devq 261, so a toolchain regenerated
# build.ninja differently at some point and nothing re-ran the target. A gate
# nobody runs stops being a gate silently, which is this suite's own subject.
#
#   * add_lit_testsuite does NOT shell-quote ARGS, so `(` and `|` reach
#     /bin/sh bare. They are escaped with `\\(` / `\\|` -- doubled, because
#     CMake collapses one level -- and sh strips the backslash and hands lit
#     the plain regex character.
#   * `$` must be `$$` or ninja rejects its own generated file
#     (`bad $-escape`).
#   * Single-quoting instead is NOT a fix and was tried: CMake stops seeing a
#     quoted argument, parses the parens as its own syntax, and emits
#     `transformer_layer/ ( run_...` with spaces injected.
#
# Verified end to end rather than by reading: 369 discovered, 357 excluded,
# **12 passed, 0 failed**, and the workflow's `Passed=12` guard checked against
# that real log in both directions. Run it with the toolchain environment
# sourced -- lit invokes a bare `python3`, and the system one has no
# `ml_dtypes`, which fails all eight builder-touching tests with an empty
# FileCheck stdin.
add_lit_testsuite(check-programming-examples-transformer-layer-host
  "Running transformer-layer PR-safe tests (12, no NPU dispatch)"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TEST_DEPENDS}
  ARGS ${AIR_TEST_LIT_ARGS} --filter "transformer_layer/\\(run_npu2_compile_peano\\|run_block_cache_tests\\|run_blocked_attention_tests\\|run_ffn_resident_emulation_tests\\|run_layer_norm_rows_tests\\|run_softmax_rows_tests\\|run_mapping_space_tests\\|run_reference_tests\\|run_seam_tests\\|run_study_host_tests\\|sweep/run_npu2_registry_resolution\\|sweep/run_sweep_families_tests\\|sweep/run_sweep_writer_tests\\)[.]lit$$"
)
set_target_properties(check-programming-examples-transformer-layer-host PROPERTIES FOLDER "Tests")

add_dependencies(check-all check-programming-examples)
