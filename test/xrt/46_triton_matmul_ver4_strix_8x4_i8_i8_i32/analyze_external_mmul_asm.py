#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc.

"""Disassemble and gate the Strix external INT8 mmul kernel."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEDULES = ("baseline", "flat", "manual-unroll", "software-pipeline")
KERNEL_STYLES = (
    "peano-mmul",
    "hand-scheduled",
    "native-mmul",
    "native-mmul-atb-ref",
    "asm-microkernel",
)
ATB_REFERENCE_FUNCTION = "matmul_vectorized_bfp16"
DEFAULT_MIN_MAC_DENSITY = 0.05
DEFAULT_MIN_INNER_MAC_DENSITY = 0.05
DEFAULT_DENSITY_SPEEDUP_TARGET = 1.30
DEFAULT_MAX_INNER_NOP_PER_MAC = 999.0


@dataclass(frozen=True)
class Instruction:
    address: int
    payload: str


def positive_int(value: str) -> int:
    parsed = int(value, 10)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def find_objdump(explicit: str | None) -> str:
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    peano = os.environ.get("PEANO_INSTALL_DIR")
    if peano:
        candidates.append(str(Path(peano) / "bin" / "llvm-objdump"))
    candidates.append("llvm-objdump")
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise FileNotFoundError(
        "could not find llvm-objdump; pass --llvm-objdump or set PEANO_INSTALL_DIR"
    )


def disassemble(objdump: str, obj: Path) -> str:
    cmd = [
        objdump,
        "-d",
        "--triple=aie2p-none-unknown-elf",
        "--mcpu=aie2p",
        str(obj),
    ]
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


def instruction_payloads(disasm: str) -> list[Instruction]:
    payloads: list[Instruction] = []
    for line in disasm.splitlines():
        match = re.match(r"^\s*([0-9a-fA-F]+):", line)
        if not match:
            continue
        payload = line.split(":", 1)[1]
        payload = re.sub(r"^(?:\s*[0-9a-fA-F]{2})+\s+", "", payload)
        payloads.append(Instruction(int(match.group(1), 16), payload.strip()))
    return payloads


def has_mac(payload: str) -> bool:
    return bool(re.search(r"\bvmac\b|\bmac\b|\bmmac\b", payload))


def has_vector_store(payload: str) -> bool:
    return bool(re.search(r"\bvst(?:\.[A-Za-z0-9_.]+)?\b", payload))


def detect_inner_window(instructions: list[Instruction]) -> dict[str, Any]:
    mac_indices = [
        idx for idx, inst in enumerate(instructions) if has_mac(inst.payload)
    ]
    if not mac_indices:
        return {
            "detected": False,
            "start_index": None,
            "end_index_exclusive": None,
            "start_address": None,
            "end_address": None,
            "reason": "no MAC instructions found",
        }

    first_mac = mac_indices[0]
    last_mac = mac_indices[-1]
    start = max(0, first_mac - 8)
    end = None
    for idx in range(last_mac + 1, len(instructions)):
        if has_vector_store(instructions[idx].payload):
            end = idx
            break
    if end is None:
        end = min(len(instructions), last_mac + 1)
    return {
        "detected": end > start,
        "start_index": start,
        "end_index_exclusive": end,
        "start_address": instructions[start].address,
        "end_address": instructions[end - 1].address if end > start else None,
        "reason": "first-MAC to first-post-MAC vector-store window with load context",
    }


def count_ops(instructions: list[Instruction]) -> dict[str, Any]:
    payloads = [inst.payload for inst in instructions]
    joined = "\n".join(payloads)
    op_slots = 0
    for payload in payloads:
        op_slots += sum(1 for part in payload.split(";") if part.strip())

    vector_stack_lines = [
        payload
        for payload in payloads
        if "[sp" in payload and re.search(r"\bv(?:ld|st)", payload)
    ]
    stack_lines = [payload for payload in payloads if "[sp" in payload]
    call_lines = [payload for payload in payloads if re.search(r"\bcall\w*\b", payload)]

    mac_count = len(re.findall(r"\bvmac\b|\bmac\b|\bmmac\b", joined))
    vector_load_count = len(re.findall(r"\bvld(?:a|b)?(?:\.[A-Za-z0-9_.]+)?\b", joined))
    vector_store_count = len(re.findall(r"\bvst(?:\.[A-Za-z0-9_.]+)?\b", joined))
    scalar_load_count = len(re.findall(r"\bld[ab]?\b", joined))
    scalar_store_count = len(re.findall(r"\bst\b", joined))
    branch_count = len(re.findall(r"\b(?:jz|jnz|jump|bra|ret)\b", joined))
    nop_count = len(re.findall(r"\bnop[a-z]*\b", joined))

    return {
        "instruction_lines": len(payloads),
        "operation_slots": op_slots,
        "mac": mac_count,
        "vector_load": vector_load_count,
        "vector_store": vector_store_count,
        "scalar_load": scalar_load_count,
        "scalar_store": scalar_store_count,
        "branch": branch_count,
        "nop": nop_count,
        "stack_ref_lines": len(stack_lines),
        "vector_stack_ref_lines": len(vector_stack_lines),
        "call_lines": len(call_lines),
        "mac_density_per_op_slot": mac_count / op_slots if op_slots else 0.0,
        "nop_like_per_mac": nop_count / mac_count if mac_count else None,
        "stack_ref_examples": stack_lines[:8],
        "vector_stack_ref_examples": vector_stack_lines[:8],
        "call_examples": call_lines[:8],
    }


def _extract_c_function_body(source: str, name: str) -> str:
    start = source.find(f"void {name}")
    if start < 0:
        return ""
    brace = source.find("{", start)
    if brace < 0:
        return ""
    depth = 0
    for pos in range(brace, len(source)):
        if source[pos] == "{":
            depth += 1
        elif source[pos] == "}":
            depth -= 1
            if depth == 0:
                return source[brace + 1 : pos]
    return ""


def atb_reference_metrics(source_path: Path | None) -> dict[str, Any]:
    status: dict[str, Any] = {
        "source": str(source_path) if source_path else None,
        "source_available": bool(source_path and source_path.is_file()),
        "xchesscc_available": bool(shutil.which("xchesscc")),
        "xchesscc_wrapper_available": bool(shutil.which("xchesscc_wrapper")),
        "chess_disassembly_available": False,
        "source_metrics": None,
        "notes": [],
    }
    if not status["xchesscc_available"] or not status["xchesscc_wrapper_available"]:
        status["notes"].append(
            "Chess compiler tools are unavailable; using source-derived reference metrics only"
        )
    if not source_path or not source_path.is_file():
        status["notes"].append("ATB reference source was not found")
        return status

    source = source_path.read_text(encoding="utf-8")
    body = _extract_c_function_body(source, ATB_REFERENCE_FUNCTION)
    if not body:
        status["notes"].append(f"function {ATB_REFERENCE_FUNCTION} was not found")
        return status

    mac_ops = len(re.findall(r"\bmac_8x8_8x8T\b", body))
    pop_ops = len(re.findall(r"\.pop\(", body))
    push_ops = len(re.findall(r"\.push\(", body))
    ping_pong_mentions = len(re.findall(r"_pong\b", body))
    c_accumulators = len(re.findall(r"acc\d+_data", body.split("acc0_data =", 1)[0]))
    k_groups = mac_ops // 4 if mac_ops % 4 == 0 else None
    status["source_metrics"] = {
        "function": ATB_REFERENCE_FUNCTION,
        "mac_calls": mac_ops,
        "macs_per_2x2_step": 4,
        "source_k_groups": k_groups,
        "vector_pop_calls": pop_ops,
        "vector_push_calls": push_ops,
        "ping_pong_vector_mentions": ping_pong_mentions,
        "c_accumulator_variables_before_mac_window": c_accumulators,
        "uses_four_c_accumulators": c_accumulators == 4,
        "uses_ping_pong_temporaries": ping_pong_mentions > 0,
        "bfp_specific_layout": True,
    }
    status["notes"].append(
        "Reference metrics are source-derived from config2 and are not production gates"
    )
    return status


def baseline_comparison(
    args: argparse.Namespace, inner_counts: dict[str, Any]
) -> dict[str, Any]:
    candidate_density = inner_counts["mac_density_per_op_slot"]
    baseline_density = args.baseline_inner_mac_density
    candidate_nop = inner_counts["nop_like_per_mac"]
    baseline_nop = args.max_inner_nop_per_mac
    return {
        "baseline_label": "native-mmul K=9 static gate",
        "baseline_inner_mac_density": baseline_density,
        "candidate_inner_mac_density": candidate_density,
        "inner_mac_density_ratio": (
            candidate_density / baseline_density if baseline_density else None
        ),
        "baseline_inner_nop_per_mac": baseline_nop,
        "candidate_inner_nop_per_mac": candidate_nop,
        "candidate_density_meets_baseline": (
            candidate_density >= baseline_density if baseline_density else None
        ),
        "candidate_nop_meets_baseline": (
            candidate_nop <= baseline_nop if candidate_nop is not None else None
        ),
    }


def variant_role(args: argparse.Namespace) -> str:
    block = (args.external_block_m, args.external_block_n)
    if args.kernel_style == "hand-scheduled":
        return "diagnostic-hand-scheduled"
    if args.kernel_style == "native-mmul":
        if block == (2, 2):
            return "production-spill-free-native-unrolled"
        if block == (3, 2):
            return "blocked-native-over-dm-register-budget"
        return "diagnostic-native-microkernel"
    if args.kernel_style == "native-mmul-atb-ref":
        if block == (2, 2):
            return "candidate-atb-ref-cadence-native-2x2"
        return "invalid-atb-ref-cadence-shape"
    if args.kernel_style == "asm-microkernel":
        if block == (2, 2):
            return "diagnostic-asm-spill-free-baseline"
        if block == (3, 2):
            return "blocked-asm-over-dm-register-budget"
        return "diagnostic-asm-microkernel"
    if block == (2, 2):
        return "fallback-baseline"
    if block == (3, 2):
        return "production-candidate"
    if block == (2, 3):
        return "diagnostic-control"
    if block == (4, 2):
        return "diagnostic-main-plus-tail"
    return "diagnostic"


def make_report(args: argparse.Namespace, objdump: str, disasm: str) -> dict[str, Any]:
    instructions = instruction_payloads(disasm)
    counts = count_ops(instructions)
    inner_window = detect_inner_window(instructions)
    if inner_window["detected"]:
        inner_insts = instructions[
            inner_window["start_index"] : inner_window["end_index_exclusive"]
        ]
    else:
        inner_insts = []
    inner_counts = count_ops(inner_insts)
    expected_vmacs_per_k_pack = args.external_block_m * args.external_block_n
    aie2p_dm_accumulator_registers = 5
    expected_runtime_mmul_ops = (
        args.external_active_m_packs
        * args.external_core_n_packs
        * args.external_k_packs
    )
    issues: list[str] = []

    if (
        args.kernel_style in ("native-mmul", "asm-microkernel")
        and expected_vmacs_per_k_pack > aie2p_dm_accumulator_registers
    ):
        issues.append(
            "dense raw block requires "
            f"{expected_vmacs_per_k_pack} v64acc32 accumulators, but AIE2P exposes "
            f"{aie2p_dm_accumulator_registers} 2048-bit DM accumulator registers"
        )

    if counts["mac"] == 0:
        issues.append("no AIE MAC instructions were found in the external kernel")
    if not inner_window["detected"]:
        issues.append(f"inner MAC loop was not detected: {inner_window['reason']}")
    if counts["vector_stack_ref_lines"]:
        issues.append(
            "vector load/store stack references were found; this indicates accumulator or vector spills"
        )
    if inner_counts["vector_stack_ref_lines"]:
        issues.append(
            "vector stack references were found inside the detected MAC window"
        )
    if counts["call_lines"]:
        issues.append("call instructions were found in the external kernel")
    if inner_counts["call_lines"]:
        issues.append("call instructions were found inside the detected MAC window")
    if (
        args.kernel_style == "native-mmul-atb-ref"
        and args.baseline_inner_mac_density > 0.0
        and inner_counts["mac_density_per_op_slot"] < args.baseline_inner_mac_density
    ):
        issues.append(
            "candidate inner MAC density "
            f"{inner_counts['mac_density_per_op_slot']:.4f} is below the "
            f"K=9 native baseline ({args.baseline_inner_mac_density:.4f})"
        )
    if counts["mac_density_per_op_slot"] < args.min_mac_density:
        issues.append(
            "static MAC density "
            f"{counts['mac_density_per_op_slot']:.4f} is below "
            f"{args.min_mac_density:.4f}"
        )
    if inner_counts["mac_density_per_op_slot"] < args.min_inner_mac_density:
        issues.append(
            "inner MAC density "
            f"{inner_counts['mac_density_per_op_slot']:.4f} is below "
            f"{args.min_inner_mac_density:.4f}"
        )
    if (
        args.baseline_inner_mac_density > 0.0
        and args.external_block_m * args.external_block_n > 4
    ):
        required = args.baseline_inner_mac_density * args.density_speedup_target
        if inner_counts["mac_density_per_op_slot"] < required:
            issues.append(
                "inner MAC density "
                f"{inner_counts['mac_density_per_op_slot']:.4f} is below "
                f"{args.density_speedup_target:.2f}x baseline ({required:.4f})"
            )
    inner_nop_per_mac = inner_counts["nop_like_per_mac"]
    if inner_nop_per_mac is not None and inner_nop_per_mac > args.max_inner_nop_per_mac:
        issues.append(
            f"inner NOP-like slots per MAC {inner_nop_per_mac:.3f} exceeds "
            f"{args.max_inner_nop_per_mac:.3f}"
        )

    return {
        "object": str(args.object),
        "llvm_objdump": objdump,
        "schedule": args.schedule,
        "kernel_style": args.kernel_style,
        "variant_role": variant_role(args),
        "external_k_packs": args.external_k_packs,
        "external_block": [args.external_block_m, args.external_block_n],
        "external_core_m_packs": args.external_core_m_packs,
        "external_active_m_packs": args.external_active_m_packs,
        "external_core_n_packs": args.external_core_n_packs,
        "atb_ratio": (
            args.external_core_m_packs // args.external_active_m_packs
            if args.external_core_m_packs % args.external_active_m_packs == 0
            else None
        ),
        "external_block_accumulators": args.external_block_m * args.external_block_n,
        "expected_vmacs_per_k_pack": expected_vmacs_per_k_pack,
        "aie2p_dm_accumulator_registers": aie2p_dm_accumulator_registers,
        "expected_runtime_mmul_ops_per_core_tile": expected_runtime_mmul_ops,
        "inner_window": inner_window,
        "counts": counts,
        "inner_counts": inner_counts,
        "baseline_comparison": baseline_comparison(args, inner_counts),
        "atb_reference": atb_reference_metrics(args.atb_reference_source),
        "gate": {
            "require_clean": args.require_clean,
            "min_mac_density": args.min_mac_density,
            "min_inner_mac_density": args.min_inner_mac_density,
            "baseline_inner_mac_density": args.baseline_inner_mac_density,
            "density_speedup_target": args.density_speedup_target,
            "max_inner_nop_per_mac": args.max_inner_nop_per_mac,
            "passed": not issues,
            "issues": issues,
        },
    }


def write_counts_table(f, title: str, counts: dict[str, Any]) -> None:
    f.write(f"\n## {title}\n\n")
    f.write("| Counter | Value |\n| --- | --- |\n")
    for key in (
        "instruction_lines",
        "operation_slots",
        "mac",
        "vector_load",
        "vector_store",
        "scalar_load",
        "scalar_store",
        "branch",
        "nop",
        "stack_ref_lines",
        "vector_stack_ref_lines",
        "call_lines",
    ):
        f.write(f"| `{key}` | `{counts[key]}` |\n")
    f.write(
        "| `mac_density_per_op_slot` | "
        f"`{counts['mac_density_per_op_slot']:.6f}` |\n"
    )
    nop_per_mac = counts["nop_like_per_mac"]
    f.write(
        "| `nop_like_per_mac` | " f"`{nop_per_mac:.6f}` |\n"
        if nop_per_mac is not None
        else "| `nop_like_per_mac` | `n/a` |\n"
    )


def write_markdown(
    path: Path, report: dict[str, Any], json_path: Path, disasm_path: Path
) -> None:
    counts = report["counts"]
    inner_counts = report["inner_counts"]
    gate = report["gate"]
    inner = report["inner_window"]
    with path.open("w", encoding="utf-8") as f:
        f.write("# External INT8 MMUL Assembly Report\n\n")
        f.write("## Candidate\n\n")
        f.write("| Field | Value |\n| --- | --- |\n")
        f.write(f"| Object | `{report['object']}` |\n")
        f.write(f"| Schedule | `{report['schedule']}` |\n")
        f.write(f"| Kernel style | `{report['kernel_style']}` |\n")
        f.write(f"| Variant role | `{report['variant_role']}` |\n")
        f.write(f"| Block | `{report['external_block']}` |\n")
        f.write(f"| K packs | `{report['external_k_packs']}` |\n")
        f.write(f"| Core M packs | `{report['external_core_m_packs']}` |\n")
        f.write(f"| Active M packs | `{report['external_active_m_packs']}` |\n")
        f.write(f"| Core N packs | `{report['external_core_n_packs']}` |\n")
        f.write(f"| ATB ratio | `{report['atb_ratio']}` |\n")
        f.write(f"| Accumulators | `{report['external_block_accumulators']}` |\n")
        f.write(
            f"| Expected VMACs/K pack | `{report['expected_vmacs_per_k_pack']}` |\n"
        )
        f.write(
            "| AIE2P DM accumulator registers | "
            f"`{report['aie2p_dm_accumulator_registers']}` |\n"
        )
        f.write(
            "| Expected runtime mmul ops/core tile | "
            f"`{report['expected_runtime_mmul_ops_per_core_tile']}` |\n"
        )
        f.write(f"| Inner loop detected | `{'yes' if inner['detected'] else 'no'}` |\n")
        f.write(
            f"| Inner address window | `{inner['start_address']}..{inner['end_address']}` |\n"
        )
        f.write(f"| Gate | `{'PASS' if gate['passed'] else 'FAIL'}` |\n")
        write_counts_table(f, "Whole Function Counts", counts)
        write_counts_table(f, "Detected Inner MAC Window Counts", inner_counts)

        comparison = report.get("baseline_comparison") or {}
        if comparison:
            f.write("\n## Candidate vs Baseline\n\n")
            f.write("| Metric | Candidate | Baseline |\n| --- | --- | --- |\n")
            ratio = comparison["inner_mac_density_ratio"]
            ratio_text = f"{ratio:.6f}x" if ratio is not None else "n/a"
            f.write(
                "| Inner MAC density | "
                f"`{comparison['candidate_inner_mac_density']:.6f}` | "
                f"`{comparison['baseline_inner_mac_density']:.6f}` |\n"
            )
            candidate_nop = comparison["candidate_inner_nop_per_mac"]
            candidate_nop_text = (
                f"{candidate_nop:.6f}" if candidate_nop is not None else "n/a"
            )
            f.write(
                "| Inner NOP/MAC | "
                f"`{candidate_nop_text}` | "
                f"`{comparison['baseline_inner_nop_per_mac']:.6f}` |\n"
            )
            f.write(
                f"| Density ratio | `{ratio_text}` | `{comparison['baseline_label']}` |\n"
            )

        reference = report.get("atb_reference") or {}
        if reference:
            f.write("\n## ATB Reference Availability\n\n")
            f.write("| Field | Value |\n| --- | --- |\n")
            f.write(f"| Source | `{reference.get('source')}` |\n")
            f.write(
                f"| Source available | `{'yes' if reference.get('source_available') else 'no'}` |\n"
            )
            f.write(
                f"| xchesscc available | `{'yes' if reference.get('xchesscc_available') else 'no'}` |\n"
            )
            f.write(
                f"| xchesscc_wrapper available | `{'yes' if reference.get('xchesscc_wrapper_available') else 'no'}` |\n"
            )
            metrics = reference.get("source_metrics")
            if metrics:
                f.write("\n## Source-Derived ATB Reference Metrics\n\n")
                f.write("| Metric | Value |\n| --- | --- |\n")
                for key, value in metrics.items():
                    f.write(f"| `{key}` | `{value}` |\n")
            if reference.get("notes"):
                f.write("\n")
                for note in reference["notes"]:
                    f.write(f"- {note}\n")

        f.write("\n## Gate Issues\n\n")
        if gate["issues"]:
            for issue in gate["issues"]:
                f.write(f"- {issue}\n")
        else:
            f.write("- none\n")
        if counts["vector_stack_ref_examples"]:
            f.write("\n## Vector Stack References\n\n")
            for line in counts["vector_stack_ref_examples"]:
                f.write(f"- `{line}`\n")
        if counts["call_examples"]:
            f.write("\n## Calls\n\n")
            for line in counts["call_examples"]:
                f.write(f"- `{line}`\n")
        f.write(f"\nJSON: `{json_path}`\n")
        f.write(f"\nDisassembly: `{disasm_path}`\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", type=Path, default=None)
    parser.add_argument("--llvm-objdump", default=None)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="external_mmul")
    parser.add_argument("--schedule", choices=SCHEDULES, default="software-pipeline")
    parser.add_argument("--kernel-style", choices=KERNEL_STYLES, default="peano-mmul")
    parser.add_argument("--external-k-packs", type=positive_int, default=9)
    parser.add_argument("--external-block-m", type=positive_int, default=2)
    parser.add_argument("--external-block-n", type=positive_int, default=2)
    parser.add_argument("--external-core-m-packs", type=positive_int, default=18)
    parser.add_argument("--external-active-m-packs", type=positive_int, default=18)
    parser.add_argument("--external-core-n-packs", type=positive_int, default=18)
    parser.add_argument(
        "--min-mac-density",
        type=positive_float,
        default=DEFAULT_MIN_MAC_DENSITY,
    )
    parser.add_argument(
        "--min-inner-mac-density",
        type=positive_float,
        default=DEFAULT_MIN_INNER_MAC_DENSITY,
    )
    parser.add_argument(
        "--baseline-inner-mac-density",
        type=nonnegative_float,
        default=0.0,
    )
    parser.add_argument(
        "--density-speedup-target",
        type=positive_float,
        default=DEFAULT_DENSITY_SPEEDUP_TARGET,
    )
    parser.add_argument(
        "--max-inner-nop-per-mac",
        type=positive_float,
        default=DEFAULT_MAX_INNER_NOP_PER_MAC,
    )
    parser.add_argument(
        "--atb-reference-source",
        type=Path,
        default=Path(
            "/home/cj/mlir-aie/programming_examples/ml/block_datatypes/"
            "gemm_asymmetric_tile_buffering/config2/mm_bfp_mixed.cc"
        ),
    )
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()
    if args.external_block_m not in (2, 3, 4):
        parser.error("--external-block-m must be 2, 3, or 4")
    if args.external_block_n not in (2, 3):
        parser.error("--external-block-n must be 2 or 3")
    if (args.external_block_m, args.external_block_n) == (4, 3):
        parser.error("4x3 is not supported because the 18-pack M tile needs a tail")
    if args.external_active_m_packs > args.external_core_m_packs:
        parser.error("--external-active-m-packs must be <= --external-core-m-packs")
    if args.external_core_m_packs % args.external_active_m_packs:
        parser.error("--external-active-m-packs must divide --external-core-m-packs")
    if args.external_active_m_packs % args.external_block_m:
        parser.error(
            "--external-active-m-packs must be divisible by --external-block-m"
        )
    if args.external_core_n_packs % args.external_block_n:
        parser.error("--external-core-n-packs must be divisible by --external-block-n")
    if args.reference_only:
        return args
    if args.object is None:
        parser.error("--object is required unless --reference-only is set")
    if not args.object.is_file():
        parser.error(f"object file not found: {args.object}")
    return args


def write_reference_markdown(
    path: Path, report: dict[str, Any], json_path: Path
) -> None:
    reference = report["atb_reference"]
    with path.open("w", encoding="utf-8") as f:
        f.write("# ATB Reference Report\n\n")
        f.write("| Field | Value |\n| --- | --- |\n")
        f.write(f"| Source | `{reference.get('source')}` |\n")
        f.write(
            f"| Source available | `{'yes' if reference.get('source_available') else 'no'}` |\n"
        )
        f.write(
            f"| xchesscc available | `{'yes' if reference.get('xchesscc_available') else 'no'}` |\n"
        )
        f.write(
            f"| xchesscc_wrapper available | `{'yes' if reference.get('xchesscc_wrapper_available') else 'no'}` |\n"
        )
        metrics = reference.get("source_metrics")
        if metrics:
            f.write("\n## Source-Derived Metrics\n\n")
            f.write("| Metric | Value |\n| --- | --- |\n")
            for key, value in metrics.items():
                f.write(f"| `{key}` | `{value}` |\n")
        if reference.get("notes"):
            f.write("\n## Notes\n\n")
            for note in reference["notes"]:
                f.write(f"- {note}\n")
        f.write(f"\nJSON: `{json_path}`\n")


def main() -> int:
    args = parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    if args.reference_only:
        report = {
            "mode": "atb-reference",
            "atb_reference": atb_reference_metrics(args.atb_reference_source),
        }
        json_path = args.artifact_dir / f"{args.prefix}.reference.json"
        md_path = args.artifact_dir / f"{args.prefix}.reference.md"
        json_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        write_reference_markdown(md_path, report, json_path)
        print(f"atb_reference_report={md_path}")
        print(f"atb_reference_report_json={json_path}")
        ref = report["atb_reference"]
        print(
            f"atb_reference_source_available={'yes' if ref['source_available'] else 'no'}"
        )
        print(
            f"atb_reference_xchesscc_available={'yes' if ref['xchesscc_available'] else 'no'}"
        )
        return 0

    objdump = find_objdump(args.llvm_objdump)
    disasm = disassemble(objdump, args.object)
    report = make_report(args, objdump, disasm)

    disasm_path = args.artifact_dir / f"{args.prefix}.disasm.s"
    json_path = args.artifact_dir / f"{args.prefix}.asm.json"
    md_path = args.artifact_dir / f"{args.prefix}.asm.md"
    disasm_path.write_text(disasm, encoding="utf-8")
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_markdown(md_path, report, json_path, disasm_path)

    print(f"external_mmul_asm_report={md_path}")
    print(f"external_mmul_asm_report_json={json_path}")
    print(f"external_mmul_mac_count={report['counts']['mac']}")
    print(
        "external_mmul_mac_density="
        f"{report['counts']['mac_density_per_op_slot']:.6f}"
    )
    print(f"external_mmul_inner_mac_count={report['inner_counts']['mac']}")
    print(
        "external_mmul_inner_mac_density="
        f"{report['inner_counts']['mac_density_per_op_slot']:.6f}"
    )
    inner_nop_per_mac = report["inner_counts"]["nop_like_per_mac"]
    if inner_nop_per_mac is not None:
        print(f"external_mmul_inner_nop_per_mac={inner_nop_per_mac:.6f}")
    print(
        "external_mmul_vector_stack_refs="
        f"{report['counts']['vector_stack_ref_lines']}"
    )
    print(
        "external_mmul_inner_vector_stack_refs="
        f"{report['inner_counts']['vector_stack_ref_lines']}"
    )
    print(f"external_mmul_gate={'PASS' if report['gate']['passed'] else 'FAIL'}")

    if args.require_clean and not report["gate"]["passed"]:
        for issue in report["gate"]["issues"]:
            print(f"external_mmul_gate_issue={issue}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
