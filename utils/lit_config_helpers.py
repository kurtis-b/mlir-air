# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path


def discover_xrt_root(config):
    candidates = []
    for value in (
        os.environ.get("XILINX_XRT"),
        os.environ.get("XRT_DIR"),
        getattr(config, "xrt_dir", None),
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

    return (
        getattr(config, "xrt_dir", ""),
        getattr(config, "xrt_bin_dir", ""),
        getattr(config, "xrt_lib_dir", ""),
        getattr(config, "xrt_include_dir", ""),
    )


def discover_libxaie(config, compat_root):
    aie_obj_root = getattr(config, "aie_obj_root", "")
    libxaie_dir = getattr(config, "libxaie_dir", "")
    runtime_target = getattr(config, "runtime_test_target", "")
    candidates = []
    for value in (
        libxaie_dir,
        os.environ.get("LIBXAIE_DIR"),
        os.path.join(aie_obj_root, "runtime_lib", runtime_target, "xaiengine")
        if aie_obj_root
        else None,
        os.path.join(aie_obj_root, "runtime_lib", "x86_64", "xaiengine")
        if aie_obj_root
        else None,
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
            compat_include = os.path.join(compat_root, "include")
            compat_lib = os.path.join(compat_root, "lib")
            os.makedirs(compat_include, exist_ok=True)
            os.makedirs(compat_lib, exist_ok=True)

            _replace_symlink(
                os.path.join(include_dir, "xaiengine.h"),
                os.path.join(compat_include, "xaiengine.h"),
            )
            _replace_symlink(
                os.path.join(include_dir, "xaiengine"),
                os.path.join(compat_include, "xaiengine"),
            )
            _replace_symlink(library_path, os.path.join(compat_lib, "libxaiengine.so"))

            cdo_driver = _find_cdo_driver(candidate)
            if cdo_driver:
                _replace_symlink(
                    cdo_driver, os.path.join(compat_lib, "libcdo_driver_mlir_aie.a")
                )
            return compat_root

    return libxaie_dir


def configure_xrt_features(config, run_on_npu_command):
    run_on_npu1 = "echo"
    run_on_npu2 = "echo"
    xrt_flags = ""

    if getattr(config, "xrt_lib_dir", "") and getattr(
        config, "enable_run_xrt_tests", False
    ):
        print("xrt found at", config.xrt_dir)
        xrt_flags = "-I{} -L{} -luuid -lxrt_coreutil".format(
            config.xrt_include_dir, config.xrt_lib_dir
        )
        config.available_features.add("xrt")

        try:
            model = detect_npu_model(config.xrt_bin_dir)
            if model:
                config.available_features.add("ryzen_ai")
                if model in ["npu1", "Phoenix"]:
                    run_on_npu1 = run_on_npu_command
                    config.available_features.add("ryzen_ai_npu1")
                    print("Running tests on NPU1 with command line: ", run_on_npu1)
                elif model in ["npu4", "Strix"]:
                    run_on_npu2 = run_on_npu_command
                    config.available_features.add("ryzen_ai_npu2")
                    print("Running tests on NPU4 with command line: ", run_on_npu2)
                else:
                    print(f"WARNING: xrt-smi reported unknown NPU model '{model}'.")
        except Exception as exc:
            print(f"Failed to run xrt-smi: {exc}")
    else:
        print("xrt not found or xrt tests disabled")
        config.excludes.append("xrt")

    return run_on_npu1, run_on_npu2, xrt_flags


def detect_npu_model(xrt_bin_dir):
    xrtsmi = shutil.which("xrt-smi") or os.path.join(xrt_bin_dir, "xrt-smi")
    result = subprocess.run(
        [xrtsmi, "examine"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    lines = result.stdout.decode("utf-8").split("\n")
    pattern = re.compile(
        r"[\|]?(\[.+:.+:.+\]).+\|(RyzenAI-(npu\d)|NPU (\w+))\W*\|"
    )
    for line in lines:
        match = pattern.match(line)
        if not match:
            continue
        print("Found Ryzen AI device:", match.group(1))
        model = "unknown"
        if match.group(3):
            model = str(match.group(3))
        if match.group(4):
            model = str(match.group(4))
        print(f"\tmodel: '{model}'")
        return model
    return None


def has_peano_aie1_support(config):
    peano_tools_dir = getattr(config, "peano_tools_dir", "")
    if not peano_tools_dir:
        return False
    return os.path.exists(
        os.path.join(
            os.path.dirname(peano_tools_dir),
            "lib",
            "aie-none-unknown-elf",
            "crt0.o",
        )
    )


def _replace_symlink(target, link_name):
    if os.path.lexists(link_name):
        os.unlink(link_name)
    os.symlink(target, link_name)


def _find_cdo_driver(candidate):
    for ancestor in [Path(candidate).resolve(), *Path(candidate).resolve().parents]:
        maybe_driver = ancestor / "lib" / "libcdo_driver_mlir_aie.a"
        if maybe_driver.exists():
            return maybe_driver
    return None
