# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gemma3 synthetic scaling policy checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from common import OUTPUT_MODE_KERNELS, parse_herd_shape, resolve_output_mode, unsupported_output_mode_reason
from gemma3_config import Gemma3TextConfig, synthetic_text_config
from gemma3_runtime import Gemma3RuntimeManifest, prepare_runtime


SCALING_HERD_SHAPES = ("2x4", "4x4", "8x4")


@dataclass(frozen=True)
class Gemma3ScalingReport:
    herd_shape: str
    modes: dict[str, str]
    artifact_count: int
    cache_dir: Path

    def format(self) -> str:
        modes = " ".join(f"{kernel}={self.modes[kernel]}" for kernel in OUTPUT_MODE_KERNELS)
        return (
            f"scale:{self.herd_shape} {modes} "
            f"artifacts={self.artifact_count} cache_dir={self.cache_dir}"
        )


@dataclass(frozen=True)
class Gemma3UnsupportedDiagnostic:
    herd_shape: str
    mode: str
    kernel: str
    failure_class: str
    reason: str

    def format(self) -> str:
        return (
            f"unsupported:{self.herd_shape}:{self.mode}:{self.kernel} "
            f"class={self.failure_class} reason={self.reason}"
        )


def _classify_reason(reason: str) -> str:
    if "shim S2MM DMA channel budget" in reason:
        return "hardware-resource-limit"
    if "packet-direct" in reason:
        return "packet-s2mm-backend-limitation"
    if "FlowKV small-shape L2 gather" in reason:
        return "channel-runtime-scheduling-bug"
    return "shape-contract-error"


def scaling_config(herd_shape: str) -> Gemma3TextConfig:
    return synthetic_text_config(n_layers=2, local_window_len=4, herd_shape=herd_shape)


def prepare_scaling_manifests(cache_root: Path | str = "/tmp/gemma3_scaling_manifests") -> tuple[Gemma3ScalingReport, ...]:
    reports = []
    root = Path(cache_root)
    for herd_shape in SCALING_HERD_SHAPES:
        config = scaling_config(herd_shape)
        cache_dir = root / herd_shape
        manifest: Gemma3RuntimeManifest = prepare_runtime(
            config,
            cache_dir=cache_dir,
            compile_only=True,
        )
        reports.append(
            Gemma3ScalingReport(
                herd_shape=herd_shape,
                modes=config.resolved_output_modes(),
                artifact_count=len(manifest.artifacts),
                cache_dir=cache_dir,
            )
        )
    return tuple(reports)


def unsupported_diagnostics() -> tuple[Gemma3UnsupportedDiagnostic, ...]:
    cases = [
        ("8x4", "direct", "q4nx"),
        ("8x4", "direct", "fused_dqp"),
        ("8x4", "direct", "flowqkv"),
        ("8x4", "direct", "flowkv"),
        ("8x4", "packet-direct", "q4nx"),
        ("8x4", "packet-direct", "fused_dqp"),
        ("2x4", "l2-gather", "flowkv"),
        ("4x4", "l2-gather", "flowkv"),
    ]
    diagnostics = []
    for herd_shape, mode, kernel in cases:
        rows, cols = parse_herd_shape(herd_shape)
        try:
            resolve_output_mode(mode, rows, cols, kernel)
        except ValueError as exc:
            reason = unsupported_output_mode_reason(mode, rows, cols, kernel) or str(exc)
            diagnostics.append(
                Gemma3UnsupportedDiagnostic(
                    herd_shape=herd_shape,
                    mode=mode,
                    kernel=kernel,
                    failure_class=_classify_reason(reason),
                    reason=reason,
                )
            )
        else:
            raise AssertionError(f"expected unsupported mode: {herd_shape} {kernel} {mode}")
    return tuple(diagnostics)


def validate_scaling_policy(reports: tuple[Gemma3ScalingReport, ...]) -> None:
    report_by_shape = {report.herd_shape: report for report in reports}
    if set(report_by_shape) != set(SCALING_HERD_SHAPES):
        raise AssertionError(f"missing scaling reports: {report_by_shape}")
    for shape in ("2x4", "4x4"):
        modes = report_by_shape[shape].modes
        if modes["flowkv"] != "direct":
            raise AssertionError(f"{shape} FlowKV must remain direct by default: {modes}")
    full = report_by_shape["8x4"].modes
    for kernel in OUTPUT_MODE_KERNELS:
        if full[kernel] != "l2-gather":
            raise AssertionError(f"8x4 {kernel} must use l2-gather, got {full[kernel]}")


def _self_test() -> None:
    reports = prepare_scaling_manifests()
    diagnostics = unsupported_diagnostics()
    validate_scaling_policy(reports)
    for report in reports:
        print(report.format())
    for diagnostic in diagnostics:
        print(diagnostic.format())
    print("GEMMA3_SCALING_POLICY_SELF_TEST: PASS")


if __name__ == "__main__":
    _self_test()
