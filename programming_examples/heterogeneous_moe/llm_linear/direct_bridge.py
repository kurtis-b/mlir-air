# SPDX-License-Identifier: MIT

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DirectBridgeStatus:
    available: bool
    library_path: str | None
    diagnostic: str


def probe_direct_bridge() -> DirectBridgeStatus:
    path = os.environ.get("LLM_LINEAR_DIRECT_BRIDGE_SO")
    if not path:
        return DirectBridgeStatus(
            available=False,
            library_path=None,
            diagnostic=(
                "LLM_LINEAR_DIRECT_BRIDGE_SO is unset; direct GPU/NPU handoff "
                "requires a native bridge that can export/import XRT BO and HIP "
                "VMem handles"
            ),
        )
    candidate = Path(path).expanduser()
    if not candidate.exists():
        return DirectBridgeStatus(
            available=False,
            library_path=str(candidate),
            diagnostic=f"LLM_LINEAR_DIRECT_BRIDGE_SO does not exist: {candidate}",
        )
    try:
        library = ctypes.CDLL(str(candidate))
    except OSError as exc:
        return DirectBridgeStatus(
            available=False,
            library_path=str(candidate),
            diagnostic=f"failed to load direct bridge: {exc}",
        )
    probe = getattr(library, "llm_linear_direct_bridge_probe", None)
    if probe is None:
        return DirectBridgeStatus(
            available=False,
            library_path=str(candidate),
            diagnostic=(
                "direct bridge library is missing llm_linear_direct_bridge_probe"
            ),
        )
    probe.restype = ctypes.c_int
    try:
        ok = int(probe()) == 0
    except Exception as exc:  # pragma: no cover - depends on native bridge
        return DirectBridgeStatus(
            available=False,
            library_path=str(candidate),
            diagnostic=f"direct bridge probe raised: {exc}",
        )
    return DirectBridgeStatus(
        available=ok,
        library_path=str(candidate),
        diagnostic=(
            "direct bridge probe succeeded"
            if ok
            else "direct bridge probe reported unavailable"
        ),
    )
