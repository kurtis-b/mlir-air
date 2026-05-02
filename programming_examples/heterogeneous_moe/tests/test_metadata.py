# SPDX-License-Identifier: MIT

from __future__ import annotations

import subprocess
from pathlib import Path

import metadata
from metadata import _run_metadata_command, _tool_path, collect_run_metadata


def test_metadata_command_success_and_failures(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        metadata.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, stdout="ok\n"),
    )
    assert _run_metadata_command(["tool"], cwd=tmp_path) == "ok"

    monkeypatch.setattr(
        metadata.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, stdout="bad\n"),
    )
    assert _run_metadata_command(["tool"], cwd=tmp_path) is None

    def raise_error(*args, **kwargs):
        raise OSError("missing")

    monkeypatch.setattr(metadata.subprocess, "run", raise_error)
    assert _run_metadata_command(["tool"], cwd=tmp_path) is None


def test_tool_path_and_collect_run_metadata(monkeypatch, tmp_path: Path, default_manifest: dict) -> None:
    monkeypatch.setenv("AIRCC_PATH", "/opt/aircc")
    assert _tool_path("aircc", "AIRCC_PATH") == "/opt/aircc"
    monkeypatch.delenv("AIRCC_PATH")
    monkeypatch.setattr(metadata.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert _tool_path("aircc") == "/usr/bin/aircc"

    monkeypatch.setattr(metadata, "_run_metadata_command", lambda cmd, cwd=None, timeout_s=2.0: "abc" if "rev-parse" in cmd else "")
    info = collect_run_metadata(tmp_path / "manifest.json", default_manifest, command_line=["cmd"])
    assert info["schema_version"] == "edge-study-v1"
    assert info["command_line"] == ["cmd"]
    assert info["git"]["sha"] == "abc"
    assert info["git"]["tracked_dirty"] is False
    assert info["devices"]["npu_device"] == default_manifest["compiler"]["npu_device"]
