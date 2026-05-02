# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

EDGE_STUDY_SCHEMA_VERSION = "edge-study-v1"


def project_dir() -> Path:
    return Path(__file__).resolve().parent


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


def artifact_root(manifest: dict[str, Any]) -> Path:
    return project_dir() / manifest["paths"]["artifacts"]


def generated_air_source_root(manifest: dict[str, Any]) -> Path:
    paths = manifest.setdefault("paths", {})
    root = paths.get("generated_air_sources")
    if not root:
        root = str(Path(paths.get("artifacts", "artifacts")) / "air_sources")
        paths["generated_air_sources"] = root
    return project_dir() / root


def update_manifest_backends(
    manifest: dict[str, Any],
    router_backend: str | None = None,
    expert0_backend: str | None = None,
    expert1_backend: str | None = None,
    aggregation_backend: str | None = None,
    transfer_mode: str | None = None,
    router_mode: str | None = None,
) -> dict[str, Any]:
    stage_backends = manifest["runtime"]["stage_backends"]
    if router_backend:
        stage_backends["router"] = router_backend
    if expert0_backend:
        stage_backends["expert0"] = expert0_backend
    if expert1_backend:
        stage_backends["expert1"] = expert1_backend
    if aggregation_backend:
        stage_backends["aggregation"] = aggregation_backend
    if transfer_mode:
        manifest["runtime"]["transfer_mode"] = transfer_mode
    if router_mode:
        manifest["runtime"]["router_mode"] = router_mode
    return manifest
