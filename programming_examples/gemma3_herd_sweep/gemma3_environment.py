#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gemma3 paper-reproduction environment capture."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_PAPER_ENV_FIELDS = (
    "git.commit",
    "git.dirty_worktree",
    "python.executable",
    "tools.aircc.path",
    "tools.air-opt.path",
    "tools.aiecc.path",
    "tools.llc.path",
    "hardware.cpu.model",
    "hardware.memory.total_kib",
    "runtime.artifact_format",
    "runtime.compile_time_included",
    "runtime.timing_window",
    "xrt.version",
    "xrt.examine_summary",
    "npu.power_mode",
)

REPO_TOOL_CANDIDATES = {
    "aircc": (
        REPO_ROOT / "install-xrt" / "bin" / "aircc",
        REPO_ROOT / "build-xrt" / "bin" / "aircc",
    ),
    "air-opt": (
        REPO_ROOT / "install-xrt" / "bin" / "air-opt",
        REPO_ROOT / "build-xrt" / "bin" / "air-opt",
    ),
    "aiecc": (
        REPO_ROOT / "sandbox" / "bin" / "aiecc",
    ),
}


def _run(args: list[str], *, cwd: Path = REPO_ROOT, timeout: int = 10) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            args,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return {"available": False, "argv": args, "returncode": None, "stdout": "", "stderr": "not found"}
    except subprocess.TimeoutExpired as exc:
        return {"available": True, "argv": args, "returncode": None, "stdout": exc.stdout or "", "stderr": "timeout"}
    return {
        "available": True,
        "argv": args,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _first_line(text: str) -> str | None:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return None


def _find_tool(name: str) -> tuple[str | None, str | None]:
    path = shutil.which(name)
    if path:
        return path, "PATH"
    for candidate in REPO_TOOL_CANDIDATES.get(name, ()):
        if candidate.exists():
            return str(candidate), "repo-local"
    return None, None


def _tool(name: str) -> dict[str, Any]:
    path, source = _find_tool(name)
    info: dict[str, Any] = {"path": path, "available": bool(path), "source": source}
    if path:
        version = _run([path, "--version"], timeout=5)
        info["version_line"] = _first_line(version.get("stdout", "") or version.get("stderr", ""))
    return info


def _git_info() -> dict[str, Any]:
    branch = _run(["git", "branch", "--show-current"])
    commit = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    return {
        "branch": _first_line(branch["stdout"]),
        "commit": _first_line(commit["stdout"]),
        "dirty_worktree": bool(status["stdout"]),
        "status_porcelain": status["stdout"].splitlines(),
    }


def _cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return platform.processor() or None
    for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or None


def _memory_total_kib() -> int | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1])
    return None


def _lspci_devices() -> list[str]:
    lspci = shutil.which("lspci")
    if not lspci:
        return []
    result = _run([lspci], timeout=10)
    if result.get("returncode") != 0:
        return []
    keywords = ("vga", "3d controller", "display", "npu", "accelerator", "amd")
    return [line for line in result["stdout"].splitlines() if any(key in line.lower() for key in keywords)]


def _xrt_info(require_hardware: bool) -> dict[str, Any]:
    xrt_smi = shutil.which("xrt-smi") or "/usr/lib/bin/xrt-smi"
    if not Path(xrt_smi).exists() and shutil.which("xrt-smi") is None:
        info = {
            "xrt_smi_path": None,
            "version": None,
            "examine_summary": None,
            "available": False,
            "missing_reason": "xrt-smi not found on PATH or /usr/lib/bin/xrt-smi",
        }
        if require_hardware:
            info["required_failure"] = True
        return info
    version = _run([xrt_smi, "--version"], timeout=10)
    examine = _run([xrt_smi, "examine", "-r", "all"], timeout=30)
    return {
        "xrt_smi_path": xrt_smi,
        "available": True,
        "version": _first_line(version.get("stdout", "") or version.get("stderr", "")),
        "version_command": version,
        "examine_summary": examine.get("stdout") or examine.get("stderr") or None,
        "examine_returncode": examine.get("returncode"),
    }


def _npu_info(xrt: dict[str, Any]) -> dict[str, Any]:
    text = xrt.get("examine_summary") or ""
    power_mode = None
    for line in text.splitlines():
        if "power" in line.lower() and "mode" in line.lower():
            power_mode = line.strip()
            break
    return {
        "power_mode": power_mode,
        "power_mode_source": "xrt-smi examine -r all" if power_mode else None,
    }


def capture_environment(
    *,
    artifact_format: str = "elf",
    compile_time_included: bool = False,
    timing_window: str = "runtime_only",
    require_hardware: bool = False,
) -> dict[str, Any]:
    xrt = _xrt_info(require_hardware)
    env = {
        "schema_version": 1,
        "repo_root": str(REPO_ROOT),
        "git": _git_info(),
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
        },
        "paths": {
            "MLIR_AIR_INSTALL_DIR": os.environ.get("MLIR_AIR_INSTALL_DIR"),
            "MLIR_AIE_INSTALL_DIR": os.environ.get("MLIR_AIE_INSTALL_DIR"),
            "PEANO_INSTALL_DIR": os.environ.get("PEANO_INSTALL_DIR"),
            "LLVM_INSTALL_DIR": os.environ.get("LLVM_INSTALL_DIR"),
            "XILINX_XRT": os.environ.get("XILINX_XRT"),
        },
        "tools": {
            "aircc": _tool("aircc"),
            "air-opt": _tool("air-opt"),
            "aiecc": _tool("aiecc"),
            "llc": _tool("llc"),
        },
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu": {"model": _cpu_model()},
            "memory": {"total_kib": _memory_total_kib()},
            "pci_devices": _lspci_devices(),
        },
        "xrt": xrt,
        "npu": _npu_info(xrt),
        "runtime": {
            "artifact_format": artifact_format,
            "compile_time_included": compile_time_included,
            "timing_window": timing_window,
            "trace_enabled": False,
        },
    }
    env["paper_comparable"] = paper_comparable(env)
    env["missing_paper_fields"] = missing_paper_fields(env)
    if require_hardware and env["missing_paper_fields"]:
        env["required_hardware_failure"] = True
    return env


def _get_path(data: dict[str, Any], dotted: str) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def missing_paper_fields(env: dict[str, Any]) -> list[str]:
    missing = []
    for field in REQUIRED_PAPER_ENV_FIELDS:
        value = _get_path(env, field)
        if value is None or value == "":
            missing.append(field)
    return missing


def paper_comparable(env: dict[str, Any]) -> bool:
    return not missing_paper_fields(env)


def validate_environment_for_paper(env: dict[str, Any]) -> None:
    missing = missing_paper_fields(env)
    if missing:
        raise ValueError(f"environment is not paper-comparable; missing fields: {missing}")


def _self_test() -> None:
    env = capture_environment(require_hardware=False)
    for key in ("schema_version", "git", "python", "tools", "hardware", "xrt", "runtime"):
        if key not in env:
            raise AssertionError(f"missing environment key: {key}")
    if env["runtime"]["artifact_format"] != "elf":
        raise AssertionError("default artifact format should be elf")
    for name in ("aircc", "air-opt", "aiecc"):
        tool = env["tools"][name]
        if tool["available"] and tool["source"] not in ("PATH", "repo-local"):
            raise AssertionError(f"unexpected tool source for {name}: {tool['source']}")
    print(f"GEMMA3_ENV_CAPTURE_SELF_TEST: comparable={env['paper_comparable']} missing={len(env['missing_paper_fields'])}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture Gemma3 paper-reproduction environment metadata")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--require-hardware", action="store_true")
    parser.add_argument("--artifact-format", default="elf")
    parser.add_argument("--compile-time-included", action="store_true")
    parser.add_argument("--timing-window", default="runtime_only")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return 0
    env = capture_environment(
        artifact_format=args.artifact_format,
        compile_time_included=args.compile_time_included,
        timing_window=args.timing_window,
        require_hardware=args.require_hardware,
    )
    if args.require_hardware:
        validate_environment_for_paper(env)
    if args.json:
        print(json.dumps(env, indent=2, sort_keys=True))
    else:
        print(
            "GEMMA3_ENV_CAPTURE: "
            f"comparable={env['paper_comparable']} "
            f"missing={len(env['missing_paper_fields'])} "
            f"commit={env['git']['commit']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
