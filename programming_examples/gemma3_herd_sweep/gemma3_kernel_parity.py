#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gemma3 standalone kernel parity matrix for paper-reproduction bring-up."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json

from common import parse_herd_shape, resolve_output_mode, unsupported_output_mode_reason


@dataclass(frozen=True)
class Gemma3KernelParityTarget:
    name: str
    paper_role: str
    make_target: str
    herd_shape: str
    output_mode: str
    schedule_mode: str
    compile_only_command: str
    hardware_command: str
    production_candidate: bool
    current_status: str
    notes: str

    def format(self) -> str:
        prod = "production" if self.production_candidate else "diagnostic"
        return (
            f"kernel {self.name} role={self.paper_role} target={self.make_target} "
            f"herd={self.herd_shape} mode={self.output_mode} schedule={self.schedule_mode} "
            f"status={self.current_status} class={prod}"
        )


@dataclass(frozen=True)
class Gemma3KernelDiagnosticExclusion:
    kernel: str
    herd_shape: str
    output_mode: str
    failure_class: str
    reason: str

    def format(self) -> str:
        return (
            f"diagnostic_exclusion kernel={self.kernel} herd={self.herd_shape} "
            f"mode={self.output_mode} class={self.failure_class} reason={self.reason}"
        )


def _make_command(target: str, *, compile_mode: str = "compile-only", extra: str = "") -> str:
    pieces = [
        "make -C programming_examples/gemma3_herd_sweep",
        target,
        f"COMPILE_MODE={compile_mode}",
        "OUTPUT_FORMAT=elf",
    ]
    if extra:
        pieces.append(extra)
    return " ".join(pieces)


def kernel_parity_targets() -> tuple[Gemma3KernelParityTarget, ...]:
    rows = [
        ("q4nx_smoke_2x4", "Q4NX dequant", "run-q4nx", "2x4", "auto", "smoke", True, "snapshot-pass"),
        ("q4nx_smoke_4x4", "Q4NX dequant", "run-q4nx", "4x4", "auto", "smoke", True, "snapshot-pass"),
        ("q4nx_smoke_8x4", "Q4NX dequant", "run-q4nx", "8x4", "l2-gather", "smoke", True, "snapshot-pass"),
        ("bf16_mm_2x4", "BF16 tiled MM", "run-mm", "2x4", "n/a", "smoke", True, "snapshot-pass"),
        ("bf16_mm_4x4", "BF16 tiled MM", "run-mm", "4x4", "n/a", "smoke", True, "snapshot-pass"),
        ("bf16_mm_8x4", "BF16 tiled MM", "run-mm", "8x4", "n/a", "smoke", True, "snapshot-pass"),
        ("fused_dqp_smoke_8x4", "decode fused dequant/projection", "run-fused-dqp", "8x4", "l2-gather", "smoke", True, "snapshot-pass"),
        ("fused_dqp_paper", "decode fused dequant/projection paper layout", "run-fused-dqp-paper", "4x4", "l2-gather", "paper", True, "snapshot-pass"),
        ("flowqkv_smoke_8x4", "prefill chunked attention", "run-flowqkv", "8x4", "l2-gather", "smoke", True, "snapshot-pass"),
        ("flowqkv_paper", "prefill FlowQKV paper layout", "run-flowqkv-paper", "8x4", "l2-gather", "paper", True, "snapshot-pass"),
        ("flowkv_smoke_8x4", "decode FlowKV attention", "run-flowkv", "8x4", "l2-gather", "smoke", True, "snapshot-pass"),
        ("flowkv_paper", "decode FlowKV paper layout", "run-flowkv-paper", "2x4", "direct", "paper", True, "snapshot-pass"),
        ("q4nx_rowband_fallback", "logical 8x4 Q4NX row-band fallback", "run-q4nx-8x4-rowband-fallback", "4x4", "l2-gather", "smoke", False, "snapshot-pass-fallback"),
        ("fused_dqp_pipeline", "diagnostic channel pipeline", "run-fused-dqp-pipeline", "8x4", "l2-gather", "pipeline", False, "snapshot-timeout"),
    ]
    targets = []
    for name, role, target, herd, mode, schedule, production, status in rows:
        extra = f"HERD_SHAPE={herd}"
        if mode != "n/a" and target not in ("run-fused-dqp-paper", "run-flowqkv-paper", "run-flowkv-paper", "run-q4nx-8x4-rowband-fallback", "run-fused-dqp-pipeline"):
            if target == "run-q4nx":
                extra += f" Q4NX_OUTPUT_MODE={mode}"
            elif target == "run-fused-dqp":
                extra += f" FUSED_DQP_OUTPUT_MODE={mode} FUSED_DQP_SCHEDULE_MODE={schedule}"
            elif target == "run-flowqkv":
                extra += f" FLOWQKV_OUTPUT_MODE={mode} FLOWQKV_SCHEDULE_MODE={schedule}"
            elif target == "run-flowkv":
                extra += f" FLOWKV_OUTPUT_MODE={mode} FLOWKV_SCHEDULE_MODE={schedule}"
        targets.append(
            Gemma3KernelParityTarget(
                name=name,
                paper_role=role,
                make_target=target,
                herd_shape=herd,
                output_mode=mode,
                schedule_mode=schedule,
                compile_only_command=_make_command(target, compile_mode="compile-only", extra=extra),
                hardware_command=_make_command(target, compile_mode="compile-and-run", extra=extra),
                production_candidate=production,
                current_status=status,
                notes="Snapshot status is from README; rerun hardware before paper claims.",
            )
        )
    return tuple(targets)


def _classify(reason: str) -> str:
    if "shim S2MM DMA channel budget" in reason:
        return "hardware-resource-limit"
    if "packet-direct" in reason:
        return "packet-s2mm-backend-limitation"
    if "FlowKV small-shape L2 gather" in reason:
        return "channel-runtime-scheduling-bug"
    return "shape-contract-error"


def diagnostic_exclusions() -> tuple[Gemma3KernelDiagnosticExclusion, ...]:
    cases = [
        ("q4nx", "8x4", "direct"),
        ("fused_dqp", "8x4", "direct"),
        ("flowqkv", "8x4", "direct"),
        ("flowkv", "8x4", "direct"),
        ("q4nx", "8x4", "packet-direct"),
        ("fused_dqp", "8x4", "packet-direct"),
        ("flowkv", "2x4", "l2-gather"),
        ("flowkv", "4x4", "l2-gather"),
    ]
    exclusions = []
    for kernel, herd, mode in cases:
        rows, cols = parse_herd_shape(herd)
        try:
            resolve_output_mode(mode, rows, cols, kernel)
        except ValueError as exc:
            reason = unsupported_output_mode_reason(mode, rows, cols, kernel) or str(exc)
            exclusions.append(Gemma3KernelDiagnosticExclusion(kernel, herd, mode, _classify(reason), reason))
        else:
            raise AssertionError(f"expected unsupported diagnostic: {kernel} {herd} {mode}")
    return tuple(exclusions)


def validate_kernel_matrix() -> None:
    targets = kernel_parity_targets()
    if not any(target.schedule_mode == "paper" and target.production_candidate for target in targets):
        raise AssertionError("paper-mode production targets are missing")
    for target in targets:
        if target.production_candidate and target.herd_shape == "8x4" and target.output_mode == "direct":
            raise AssertionError(f"8x4 direct output must not be a production target: {target.name}")
    exclusions = diagnostic_exclusions()
    if len(exclusions) != 8:
        raise AssertionError(f"expected 8 diagnostic exclusions, got {len(exclusions)}")


def _self_test() -> None:
    validate_kernel_matrix()
    for target in kernel_parity_targets():
        print(target.format())
    for exclusion in diagnostic_exclusions():
        print(exclusion.format())
    print("GEMMA3_KERNEL_PARITY_MATRIX_SELF_TEST: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma3 standalone kernel parity matrix")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return 0
    data = {
        "schema_version": 1,
        "targets": [asdict(target) for target in kernel_parity_targets()],
        "diagnostic_exclusions": [asdict(exclusion) for exclusion in diagnostic_exclusions()],
    }
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        for target in kernel_parity_targets():
            print(target.format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
