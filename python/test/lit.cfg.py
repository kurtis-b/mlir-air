# ./python/test/lit.cfg.py -*- Python -*-

# Copyright (C) 2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# -*- Python -*-

import os
import sys
import importlib.util

import lit.formats

from lit.llvm import llvm_config


def _load_lit_helpers():
    helper_path = os.path.join(config.air_src_root, "utils", "lit_config_helpers.py")
    spec = importlib.util.spec_from_file_location("air_lit_config_helpers", helper_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lit_helpers = _load_lit_helpers()


# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "AIRPYTHON"

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

try:
    import torch_mlir

    config.available_features.add("torch_mlir")
except:
    print("torch_mlir not found")
    pass


spensor_name = "spensor"
# If spensor is already imported or can be imported
if spensor_name in sys.modules or importlib.util.find_spec(spensor_name):
    print(spensor_name + " found")
    config.available_features.add("spensor")

print("Running with PYTHONPATH", config.environment["PYTHONPATH"])

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".py"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.air_obj_root, "python", "test")
air_runtime_lib = os.path.join(config.air_obj_root, "runtime_lib")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))
config.substitutions.append(("%PYTHON", config.python_executable))

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = []

run_on_npu1 = "echo"
run_on_npu2 = "echo"
xrt_flags = ""


config.xrt_dir, config.xrt_bin_dir, config.xrt_lib_dir, config.xrt_include_dir = (
    lit_helpers.discover_xrt_root(config)
)
config.environment["PYTHONPATH"] = "{}:{}:{}".format(
    os.path.join(config.air_obj_root, "python"),
    os.path.join(config.aie_obj_root, "python"),
    os.path.join(config.xrt_dir, "python"),
)

run_on_npu1, run_on_npu2, xrt_flags = lit_helpers.configure_xrt_features(
    config, f"{config.air_src_root}/utils/run_on_npu.sh"
)

config.substitutions.append(("%run_on_npu1%", run_on_npu1))
config.substitutions.append(("%run_on_npu2%", run_on_npu2))
config.substitutions.append(("%xrt_flags", xrt_flags))
config.substitutions.append(("%XRT_DIR", config.xrt_dir))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

llvm_config.use_default_substitutions()

config.excludes.append("lit.cfg.py")

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

config.aie_tools_dir = os.path.join(config.aie_obj_root, "bin")
config.air_tools_dir = os.path.join(config.air_obj_root, "bin")

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.aie_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.air_tools_dir, append_path=True)

tool_dirs = [config.aie_tools_dir, config.air_tools_dir, config.llvm_tools_dir]
tools = [
    "aie-opt",
    "aie-translate",
    "aiecc.py",
    "aircc",
    "air-opt",
    "clang",
    "clang++",
    "ld.lld",
    "llc",
    "llvm-objdump",
    "mlir-translate",
    "opt",
]

llvm_config.add_tool_substitutions(tools, tool_dirs)
