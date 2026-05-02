# ./test/lit.cfg.py -*- Python -*-
#
# Copyright (C) 2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# -*- Python -*-

import os
import re
import subprocess
import shutil
import importlib.util

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool


def _load_lit_helpers():
    helper_path = os.path.join(config.air_src_root, "utils", "lit_config_helpers.py")
    spec = importlib.util.spec_from_file_location("air_lit_config_helpers", helper_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lit_helpers = _load_lit_helpers()


# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "AIR_PROGRAMMING_EXAMPLES"

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
config.test_exec_root = os.path.join(config.air_obj_root, "programming_examples")
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

# for xchesscc_wrapper
llvm_config.with_environment("AIETOOLS", config.vitis_aietools_dir)


config.xrt_dir, config.xrt_bin_dir, config.xrt_lib_dir, config.xrt_include_dir = (
    lit_helpers.discover_xrt_root(config)
)
config.libxaie_dir = lit_helpers.discover_libxaie(
    config,
    os.path.join(config.air_obj_root, "programming_examples", "lit_support", "libxaie"),
)
config.environment["PYTHONPATH"] = "{}:{}:{}".format(
    os.path.join(config.air_obj_root, "python"),
    os.path.join(config.aie_obj_root, "python"),
    os.path.join(config.xrt_dir, "python"),
)
config.substitutions.insert(0, ("%LIBXAIE_DIR%", config.libxaie_dir))

run_on_npu1 = "echo"
run_on_npu2 = "echo"
xrt_flags = ""

run_on_npu1, run_on_npu2, xrt_flags = lit_helpers.configure_xrt_features(
    config, f"flock /tmp/npu.lock {config.air_src_root}/utils/run_on_npu.sh"
)

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
config.air_tools_dir = os.path.join(config.air_obj_root, "bin")

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.peano_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.aie_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.air_tools_dir, append_path=True)

config.substitutions.append(("%LLVM_TOOLS_DIR", config.llvm_tools_dir))

tool_dirs = [config.aie_tools_dir, config.llvm_tools_dir]

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
