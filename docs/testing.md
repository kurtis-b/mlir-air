
# Testing

Testing is implementing using the [https://llvm.org/docs/CommandGuide/lit.html](lit framework).  The goal of this testing is to test both individual passes as part of unit testing, and end-to-end functionality of different parts of the toolchain as part of integration testing, and eventually to measure and track performance of each component.

Tests are generally run from a build directory using ninja:
```
$ cd build
$ ninja check-air
```

## Lint And Format

Install the root pre-commit hooks if you want local staged-file hygiene:

```
$ sandbox/bin/python3 -m pip install -r utils/requirements_dev.txt
$ sandbox/bin/pre-commit install
```

Before sending a branch for review, run the repository hygiene checks against files changed from the branch base:

```
$ sandbox/bin/pre-commit run --from-ref upstream/main --to-ref HEAD
$ git clang-format --diff upstream/main
$ git diff --name-only upstream/main -- '*.py'
$ sandbox/bin/python3 -m black --check --diff <changed-python-files>
```

To apply formatting, run `git clang-format upstream/main` and rerun Black on the same changed Python file list without `--check --diff`. The existing GitHub static-analysis workflow runs `clang-tidy`; there is no separate local `clang-tidy` gate required for the validation lanes below.

These hooks are convenience checks only. CI and the documented validation lanes below remain authoritative because Git hooks are local and bypassable. Use `SKIP=<hook-id>` only for narrow local skips, not as branch validation.

## Testing an Install Area

It is almost always much faster to cross-compile these tools for embedded processors (e.g. ARM/AArch64) rather than compiling locally.  To test a cross-compiled build, the tests can be configured using cmake independently from the rest of the source code.  This leverages standard cmake mechanisms to export information about an install area.

```
$ cd aie/test
$ mkdir build
$ cd build
$ cmake -GNinja .. -DCMAKE_MODULE_PATH=/home/xilinx/acdc/cmakeModules/cmakeModulesXilinx/
```
Note that CMAKE_MODULE_PATH needs to be an absolute path at the moment

## Unit Testing

Most unit tests check the behavior of individual compilation passes.  In general, we follow [https://llvm.org/docs/TestingGuide.html] best practices from LLVM, such as `FileCheck`.

```
// RUN: aie-opt --aie-create-pathfinder-flows --aie-find-flows %s | FileCheck %s
// CHECK: %[[T23:.*]] = AIE.tile(2, 3)
```

## On-board Integration Testing (vck5000)

If no board is available, then designs will still be compiled (enabling some minimal testing).  However, on a board, the tests will automatically be run as well.  This is controlled by the cmake `ENABLE_RUN_AIRHOST_TESTS` option, the lit configuration and the `%run_on_board` substitution:
```
$ cmake -GNinja .. -DCMAKE_MODULE_PATH=/home/xilinx/acdc/cmakeModules/cmakeModulesXilinx/ -DENABLE_RUN_AIRHOST_TESTS=ON
```
```
// RUN: clang ... -o test.elf
// RUN: %run_on_board test.elf
```

When a board is available, `%run_on_board test.elf` becomes `sudo test.elf`, executing the elf file.  If the execution fails (i.e., returns a negative return value), then the test will fail.  If no board is available then `%run_on_board test.elf` becomes `echo test.elf`, to disable running the test.  Note that this mechanism means that the executable must be self-checking and cannot use the common `FileCheck` mechanism to check the output of running `test.elf`.

Board tests must also be serialized because they assume exclusive access to the hardware.  Currently tests are serialized by adding `flock /tmp/board.lock` to the `%run_on_board` command line. The complete `%run_on_board <command line>` substitution is `sudo flock /tmp/board.lock <command line>`.

## Branch Validation Lanes

Use these lanes when validating changes across compiler, Python, GPU, MoE, and hardware-backed runtime paths. These are branch-validation lanes, not source line coverage metrics, except for the explicit opt-in MoE coverage target.

CI-safe compiler and CPU checks:

```
$ ninja -C build-gpu-lit check-air-mlir
$ ninja -C build-gpu-lit check-air-cpp
$ ninja -C build-gpu-lit check-airmlir-conversion-airtorocdl
$ ninja -C build check-heterogeneous-moe-coverage
$ cd programming_examples/heterogeneous_moe
$ python3 smoke_tests.py --lane ci
```

The MoE coverage target writes reports under `programming_examples/heterogeneous_moe/artifacts/coverage/latest` and remains outside `check-all`. For MoE setup details, backend prerequisites, hardware gates, result interpretation, and troubleshooting, see the [heterogeneous MoE exploration guide](../programming_examples/heterogeneous_moe/docs/exploration.md).

GPU-local checks, when a ROCm/AMDGPU LLVM toolchain is configured:

```
$ cd programming_examples/heterogeneous_moe
$ export LLVM_INSTALL_DIR="$(git rev-parse --show-toplevel)/llvm/install-amdgpu"
$ export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
$ python3 smoke_tests.py --lane gpu-all
```

AIE/Python checks, from an AIE-enabled build with Python bindings and Peano available:

```
$ ninja -C build check-air-mlir
$ ninja -C build check-air-python
$ ninja -C build check-air-e2e-peano
$ ninja -C build check-programming-examples-peano
```

NPU/XRT hardware checks must be opt-in. Source XRT setup, confirm `xrt-smi examine` reports the expected Ryzen AI NPU, configure with `ENABLE_RUN_XRT_TESTS=ON`, then run the hardware lane:

```
$ source /opt/xilinx/xrt/setup.sh
$ xrt-smi examine
$ ninja -C build-xrt check-air-e2e-peano
$ cd programming_examples/heterogeneous_moe
$ python3 smoke_tests.py --lane npu --allow-npu
```

-----

<p align="center">Copyright&copy; 2019-2024 Advanced Micro Devices, Inc.</p>
