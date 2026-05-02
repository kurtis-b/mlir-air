# ./test/lit.cfg.py -*- Python -*-
#
# Copyright (C) 2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# -*- Python -*-

import os
import platform
import re
import subprocess
import tempfile
import shutil
from pathlib import Path
import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "AIR_TEST"

config.recursiveExpansionLimit = 10
config.test_format = lit.formats.ShTest(
    not llvm_config.use_lit_shell,
    extra_substitutions=[("%T", "%t.dir")],
    preamble_commands=["rm -rf %T && mkdir -p %T"],
)
config.environment["PYTHONPATH"] = "{}:{}:{}".format(
    os.path.join(config.air_obj_root, "python"),
    os.path.join(config.aie_obj_root, "python"),
    os.path.join(config.xrt_dir, "python"),
)

# os.environ['PYTHONPATH']
print("Running with PYTHONPATH", config.environment["PYTHONPATH"])

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".lit"]

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = []

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.air_obj_root, "test")
air_runtime_lib = os.path.join(
    config.air_obj_root, "runtime_lib", config.runtime_test_target
)

config.substitutions.append(("%PYTHON", config.python_executable))
config.substitutions.append(("%CLANG", "clang++ -fuse-ld=lld -DLIBXAIENGINEV2"))
config.substitutions.append(("%LIBXAIE_DIR%", config.libxaie_dir))
config.substitutions.append(
    (
        "%AIE_RUNTIME_DIR%",
        os.path.join(config.aie_obj_root, "runtime_lib", config.runtime_test_target),
    )
)
config.substitutions.append(("%aietools", config.vitis_aietools_dir))

test_lib_path = os.path.join(
    config.aie_obj_root, "runtime_lib", config.runtime_test_target, "test_lib"
)
config.substitutions.append(
    (
        "%test_utils_flags",
        "-lboost_program_options -lboost_filesystem "
        + f"-I{test_lib_path}/include -L{test_lib_path}/lib -ltest_utils",
    )
)

# for xchesscc_wrapper
llvm_config.with_environment("AIETOOLS", config.vitis_aietools_dir)


def _discover_xrt_root():
    candidates = []
    for value in (
        os.environ.get("XILINX_XRT"),
        os.environ.get("XRT_DIR"),
        config.xrt_dir,
    ):
        if value:
            candidates.append(value)
    xrtsmi = shutil.which("xrt-smi")
    if xrtsmi:
        candidates.append(str(Path(xrtsmi).resolve().parents[1]))
    for root in candidates:
        include_dir = os.path.join(root, "include")
        lib_dir = os.path.join(root, "lib")
        bin_dir = os.path.join(root, "bin")
        if os.path.exists(os.path.join(bin_dir, "xrt-smi")) and os.path.isdir(
            include_dir
        ):
            return root, bin_dir, lib_dir, include_dir
    return config.xrt_dir, config.xrt_bin_dir, config.xrt_lib_dir, config.xrt_include_dir


def _discover_libxaie():
    candidates = []
    for value in (
        config.libxaie_dir,
        os.environ.get("LIBXAIE_DIR"),
        os.path.join(
            config.aie_obj_root, "runtime_lib", config.runtime_test_target, "xaiengine"
        ),
        os.path.join(config.aie_obj_root, "runtime_lib", "x86_64", "xaiengine"),
    ):
        if value:
            candidates.append(value)

    for candidate in candidates:
        include_dir = os.path.join(candidate, "include")
        lib_dir = os.path.join(candidate, "lib")
        if not os.path.exists(os.path.join(include_dir, "xaiengine", "xaiegbl.h")):
            continue
        for library_name in ("libxaiengine.so", "libxaienginecdo.so"):
            library_path = os.path.join(lib_dir, library_name)
            if not os.path.exists(library_path):
                continue
            compat_root = os.path.join(config.test_exec_root, "lit_support", "libxaie")
            compat_include = os.path.join(compat_root, "include")
            compat_lib = os.path.join(compat_root, "lib")
            os.makedirs(compat_include, exist_ok=True)
            os.makedirs(compat_lib, exist_ok=True)
            root_header = os.path.join(compat_include, "xaiengine.h")
            if os.path.lexists(root_header):
                os.unlink(root_header)
            os.symlink(os.path.join(include_dir, "xaiengine.h"), root_header)
            include_link = os.path.join(compat_include, "xaiengine")
            if os.path.lexists(include_link):
                os.unlink(include_link)
            os.symlink(os.path.join(include_dir, "xaiengine"), include_link)
            library_link = os.path.join(compat_lib, "libxaiengine.so")
            if os.path.lexists(library_link):
                os.unlink(library_link)
            os.symlink(library_path, library_link)
            cdo_driver = None
            for ancestor in [Path(candidate).resolve(), *Path(candidate).resolve().parents]:
                maybe_driver = ancestor / "lib" / "libcdo_driver_mlir_aie.a"
                if maybe_driver.exists():
                    cdo_driver = maybe_driver
                    break
            if cdo_driver:
                cdo_link = os.path.join(compat_lib, "libcdo_driver_mlir_aie.a")
                if os.path.lexists(cdo_link):
                    os.unlink(cdo_link)
                os.symlink(cdo_driver, cdo_link)
            return compat_root
    return config.libxaie_dir


config.xrt_dir, config.xrt_bin_dir, config.xrt_lib_dir, config.xrt_include_dir = (
    _discover_xrt_root()
)
config.libxaie_dir = _discover_libxaie()
config.libxaie_support_flags = ""
if os.path.exists(os.path.join(config.libxaie_dir, "lib", "libcdo_driver_mlir_aie.a")):
    config.libxaie_support_flags = (
        " -L"
        + os.path.join(config.libxaie_dir, "lib")
        + " -lcdo_driver_mlir_aie"
    )
if (
    "chess" in config.available_features
    or os.path.exists(
        os.path.join(os.path.dirname(config.peano_tools_dir), "lib", "aie-none-unknown-elf", "crt0.o")
    )
):
    config.available_features.add("aie1-toolchain")
config.environment["PYTHONPATH"] = "{}:{}:{}".format(
    os.path.join(config.air_obj_root, "python"),
    os.path.join(config.aie_obj_root, "python"),
    os.path.join(config.xrt_dir, "python"),
)
config.substitutions.insert(0, ("%LIBXAIE_DIR%", config.libxaie_dir))

if config.hsa_found:
    # Getting the path to the ROCm directory. hsa-runtime64 points to the cmake
    # directory so need to go up three directories
    rocm_root = os.path.join(config.hsa_dir, "..", "..", "..")
    print("Found ROCm:", rocm_root)
    config.substitutions.append(("%HSA_DIR%", "{}".format(rocm_root)))
    config.substitutions.append(
        (
            "%airhost_libs%",
            " -I"
            + air_runtime_lib
            + "/airhost/include"
            + " -L"
            + air_runtime_lib
            + "/airhost -Wl,--whole-archive -lairhost -Wl,--no-whole-archive"
            + " -Wl,-rpath,"
            + air_runtime_lib
            + "/airhost -Wl,-rpath,"
            + os.path.join(config.libxaie_dir, "lib")
            + " -Wl,-rpath,{}/lib".format(rocm_root)
            + config.libxaie_support_flags
            + " -lpthread -lstdc++ -ldl -lrt -lelf",
        )
    )
    if config.enable_run_airhost_tests:
        config.substitutions.append(("%run_on_board", "flock /tmp/vck5000.lock"))
    else:
        print("Skipping execution of airhost tests (ENABLE_RUN_AIRHOST_TESTS=OFF)")
        config.substitutions.append(("%run_on_board", "echo"))
else:
    print("ROCm not found")
    config.excludes.append("airhost")


run_on_npu1 = "echo"
run_on_npu2 = "echo"
xrt_flags = ""

# XRT
if config.xrt_lib_dir and config.enable_run_xrt_tests:
    print("xrt found at", config.xrt_dir)
    xrt_flags = "-I{} -L{} -luuid -lxrt_coreutil".format(
        config.xrt_include_dir, config.xrt_lib_dir
    )
    config.available_features.add("xrt")

    try:
        xrtsmi = shutil.which("xrt-smi") or os.path.join(config.xrt_bin_dir, "xrt-smi")
        result = subprocess.run(
            [xrtsmi, "examine"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        result = result.stdout.decode("utf-8").split("\n")
        # Older format is "|[0000:41:00.1]  ||RyzenAI-npu1  |"
        # Newer format is "|[0000:41:00.1]  |NPU Phoenix  |"
        p = re.compile(r"[\|]?(\[.+:.+:.+\]).+\|(RyzenAI-(npu\d)|NPU (\w+))\W*\|")
        for l in result:
            m = p.match(l)
            if not m:
                continue
            print("Found Ryzen AI device:", m.group(1))
            model = "unknown"
            if m.group(3):
                model = str(m.group(3))
            if m.group(4):
                model = str(m.group(4))
            print(f"\tmodel: '{model}'")
            config.available_features.add("ryzen_ai")
            run_on_npu = (
                f"flock /tmp/npu.lock {config.air_src_root}/utils/run_on_npu.sh"
            )
            if model in ["npu1", "Phoenix"]:
                run_on_npu1 = run_on_npu
                config.available_features.add("ryzen_ai_npu1")
                print("Running tests on NPU1 with command line: ", run_on_npu1)
            elif model in ["npu4", "Strix"]:
                run_on_npu2 = run_on_npu
                config.available_features.add("ryzen_ai_npu2")
                print("Running tests on NPU4 with command line: ", run_on_npu2)
            else:
                print(f"WARNING: xrt-smi reported unknown NPU model '{model}'.")
            break
    except Exception as e:
        print(f"Failed to run xrt-smi: {e}")
else:
    print("xrt not found or xrt tests disabled")
    config.excludes.append("xrt")

config.substitutions.append(("%run_on_npu1%", run_on_npu1))
config.substitutions.append(("%run_on_npu2%", run_on_npu2))
config.substitutions.append(("%xrt_flags", xrt_flags))
config.substitutions.append(("%XRT_DIR", config.xrt_dir))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

llvm_config.use_default_substitutions()

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.air_obj_root, "test")
config.aie_tools_dir = os.path.join(config.aie_obj_root, "bin")
config.aie_python_tools_dir = os.path.join(config.aie_obj_root, "python/aie/utils")
config.air_tools_dir = os.path.join(config.air_obj_root, "bin")

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.peano_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.aie_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.aie_python_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.air_tools_dir, append_path=True)

config.substitutions.append(("%LLVM_TOOLS_DIR", config.llvm_tools_dir))

tool_dirs = [config.aie_tools_dir, config.aie_python_tools_dir, config.llvm_tools_dir]

# Test if Peano is available
try:
    result = subprocess.run(
        [os.path.join(config.peano_tools_dir, "llc"), "-mtriple=aie", "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if re.search("Xilinx AI Engine", result.stdout.decode("utf-8")) is not None:
        config.available_features.add("peano")
        config.substitutions.append(
            ("%PEANO_INSTALL_DIR", os.path.dirname(config.peano_tools_dir))
        )
        print("Peano found: " + os.path.join(config.peano_tools_dir, "llc"))
        peano_flags = "-O2 -std=c++20 -DNDEBUG -I{}".format(
            os.path.join(config.aie_obj_root, "include")
        )
        config.substitutions.append(("%peano_flags", peano_flags))
    else:
        print("Peano not detected at expected path:", config.peano_tools_dir)
except Exception:
    print("Peano check failed.")

# Test if Chess is available
if not config.enable_chess_tests:
    print("Chess tests disabled.")
else:
    print("Looking for Chess...")

    chess_path = shutil.which("xchesscc")
    if chess_path:
        print("Chess found: " + chess_path)
        config.available_features.add("chess")
        lm_license_file = os.getenv("LM_LICENSE_FILE")
        xilinxd_license_file = os.getenv("XILINXD_LICENSE_FILE")

        if lm_license_file:
            llvm_config.with_environment("LM_LICENSE_FILE", lm_license_file)
        if xilinxd_license_file:
            llvm_config.with_environment("XILINXD_LICENSE_FILE", xilinxd_license_file)

        # Optionally validate license
        validate_chess = False
        if validate_chess:
            result = subprocess.run(
                ["xchesscc", "+v"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if len(result.stderr.decode("utf-8")) == 0:
                config.available_features.add("valid_xchess_license")
        else:
            if lm_license_file or xilinxd_license_file:
                config.available_features.add("valid_xchess_license")
            else:
                print("WARNING: Chess license environment variables not found.")

    elif os.getenv("XILINXD_LICENSE_FILE") is not None:
        print("Chess license found")
        llvm_config.with_environment(
            "XILINXD_LICENSE_FILE", os.getenv("XILINXD_LICENSE_FILE")
        )
    else:
        print("Chess not found")

tool_dirs = [
    config.aie_tools_dir,
    config.aie_python_tools_dir,
    config.air_tools_dir,
    config.llvm_tools_dir,
    config.peano_tools_dir,
]
aiecc_args = []
if "chess" not in config.available_features and os.path.exists(
    os.path.join(config.peano_tools_dir, "llc")
):
    aiecc_args = [
        "--no-xchesscc",
        "--no-xbridge",
        "--peano=" + os.path.dirname(config.peano_tools_dir),
    ]
aircc_args = []
if "chess" not in config.available_features and os.path.exists(
    os.path.join(config.peano_tools_dir, "llc")
):
    aircc_args = [
        "--no-xchesscc",
        "--no-xbridge",
        "--peano=" + os.path.dirname(config.peano_tools_dir),
    ]
    if "ryzen_ai_npu2" in config.available_features:
        aircc_args.append("--device=npu2")
    elif "ryzen_ai_npu1" in config.available_features:
        aircc_args.append("--device=npu1")
tools = [
    "aie-opt",
    "aie-translate",
    ToolSubst("aiecc.py", command=FindTool("aiecc.py"), extra_args=aiecc_args),
    ToolSubst("aircc", command=FindTool("aircc"), extra_args=aircc_args),
    "air-opt",
    "ld.lld",
    ToolSubst("llc", command=os.path.join(config.llvm_tools_dir, "llc")),
    "llvm-objdump",
    "mlir-translate",
    ToolSubst("opt", command=os.path.join(config.llvm_tools_dir, "opt")),
]

llvm_config.add_tool_substitutions(tools, tool_dirs)
