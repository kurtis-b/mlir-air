# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "llm-linear-v1"


def package_dir() -> Path:
    return Path(__file__).resolve().parent


def project_dir() -> Path:
    return package_dir().parent


def repo_dir() -> Path:
    return project_dir().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def stable_json_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_package_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return package_dir() / candidate


def artifact_root(manifest: dict[str, Any]) -> Path:
    return package_dir() / manifest["paths"]["artifacts"]


def generated_air_source_root(manifest: dict[str, Any]) -> Path:
    paths = manifest.setdefault("paths", {})
    root = paths.get("generated_air_sources")
    if not root:
        root = str(Path(paths.get("artifacts", "artifacts")) / "generated_air")
        paths["generated_air_sources"] = root
    return package_dir() / root


def apply_case_to_manifest(
    manifest: dict[str, Any],
    case: dict[str, Any],
    *,
    transfer_mode: str | None = None,
) -> dict[str, Any]:
    stage_backends = manifest.setdefault("runtime", {}).setdefault("stage_backends", {})
    stage_backends["prefill"] = case["prefill_backend"]
    stage_backends["decode"] = case["decode_backend"]
    manifest["runtime"]["transfer_mode"] = (
        transfer_mode
        or case.get("transfer_mode")
        or manifest["runtime"]["transfer_mode"]
    )
    return manifest


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
        "schema_version": SCHEMA_VERSION,
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
