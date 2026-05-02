# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from manifest import EDGE_STUDY_SCHEMA_VERSION, repo_dir, stable_json_hash


def _run_metadata_command(
    cmd: list[str], cwd: Path | None = None, timeout_s: float = 2.0
) -> str | None:
    try:
        completed = subprocess.run(
            cmd,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _tool_path(name: str, env_var: str | None = None) -> str | None:
    if env_var and os.environ.get(env_var):
        return str(Path(os.environ[env_var]).expanduser())
    return shutil.which(name)


def collect_run_metadata(
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    command_line: list[str] | None = None,
) -> dict[str, Any]:
    repo = repo_dir()
    git_sha = _run_metadata_command(["git", "rev-parse", "HEAD"], cwd=repo)
    tracked_status = _run_metadata_command(
        ["git", "status", "--short", "--untracked-files=no"], cwd=repo
    )
    return {
        "schema_version": EDGE_STUDY_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command_line": command_line or [],
        "manifest_path": str(manifest_path),
        "manifest_sha256": stable_json_hash(manifest),
        "git": {
            "repo": str(repo),
            "sha": git_sha,
            "tracked_dirty": bool(tracked_status),
            "tracked_status_line_count": (
                0 if not tracked_status else len(tracked_status.splitlines())
            ),
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "environment": {
            name: os.environ.get(name)
            for name in (
                "AIRCC_PATH",
                "AIR_OPT_PATH",
                "LLVM_INSTALL_DIR",
                "MLIR_AIE_INSTALL_DIR",
                "ROCM_PATH",
                "XILINX_XRT",
                "LD_LIBRARY_PATH",
            )
            if os.environ.get(name)
        },
        "tools": {
            "aircc": _tool_path("aircc", "AIRCC_PATH"),
            "air-opt": _tool_path("air-opt", "AIR_OPT_PATH"),
            "mlir-opt": _tool_path("mlir-opt"),
            "mlir-translate": _tool_path("mlir-translate"),
            "clang": _tool_path("clang"),
            "xrt-smi": _tool_path("xrt-smi"),
            "rocminfo": _tool_path("rocminfo"),
        },
        "devices": {
            "cpu": platform.processor() or platform.machine(),
            "gpu_arch": manifest.get("compiler", {}).get("gpu_arch"),
            "npu_device": manifest.get("compiler", {}).get("npu_device"),
        },
    }
