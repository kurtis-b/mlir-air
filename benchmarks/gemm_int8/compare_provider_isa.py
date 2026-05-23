#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.

"""Compare MLIR-AIR and rocBLAS/Tensile gfx1150 INT8 GEMM ISA evidence."""

from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

M = N = K = 1024
OPS = 2.0 * M * N * K
IDEAL_BYTES = M * K + K * N + M * N * 4
DEFAULT_PROVIDER = "rocblas_tensile"
DEFAULT_AIR_TUNED_PROVIDER = "air_tuned"
ROCMLIR_REFERENCE_PROVIDER = "rocmlir_reference"
ROCMLIR_REFERENCE_SOURCE = "compiler-reference"
AIR_TUNED_DIRECT_VARIANT = "global_128x128_bpack_w4_direct"
DIRECT_CANONICAL_VARIANT = "global_128x128_bpack_w4_direct_canonical"
DIRECT_PREFETCH_VARIANT = "global_128x128_bpack_w4_prefetch"
DIRECT_RAWPTR_VARIANT = "global_128x128_bpack_w4_direct_rawptr"
DIRECT_RAWPTR_U2_VARIANT = "global_128x128_bpack_w4_direct_rawptr_u2"
DIRECT_VARIANTS = {
    AIR_TUNED_DIRECT_VARIANT,
    DIRECT_CANONICAL_VARIANT,
    DIRECT_PREFETCH_VARIANT,
    DIRECT_RAWPTR_VARIANT,
    DIRECT_RAWPTR_U2_VARIANT,
}
ACCEPTED_MLIR_AIR_BASELINE_VARIANT = "lds_128x128_rocmlir_k32_pipe3"
AIR_TUNED_ACCEPTANCE_PCT = 95.0
CANDIDATE_IMPROVEMENT_PCT = 5.0
DEFAULT_TENSILE_LIBRARY = (
    "TensileLibrary_Type_I8I_HPA_Contraction_l_Ailk_Bljk_Cijk_Dijk_gfx1150.co"
)
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
ELF_MAGIC = b"\x7fELF"

COUNTER_PATTERNS = {
    "instructions": r"^\s+(?:s|v|global|buffer|flat|ds|image|scratch)_[A-Za-z0-9_]+",
    "wmma": r"\bv_wmma_i32_16x16x16_iu8\b",
    "global_load": r"\b(?:global|buffer)_load",
    "global_load_b128": r"\b(?:global|buffer)_load_b128\b",
    "global_load_b64": r"\b(?:global|buffer)_load_b64\b",
    "global_load_b32": r"\b(?:global|buffer)_load_b32\b",
    "global_load_u8": r"\b(?:global|buffer)_load(?:_d16(?:_hi)?|_u8)?_u8\b",
    "global_store": r"\b(?:global|buffer)_store",
    "global_store_b32": r"\b(?:global|buffer)_store_b32\b",
    "ds_read": r"\bds_(?:read|load)_",
    "ds_read_b128": r"\bds_(?:read|load)_b128\b",
    "ds_write": r"\bds_(?:write|store)_",
    "ds_write_b128": r"\bds_(?:write|store)_b128\b",
    "ds_write_b8": r"\bds_(?:write|store)_b8\b",
    "ds_swizzle": r"\bds_swizzle_b32\b",
    "barriers": r"\bs_barrier\b",
    "waitcnt": r"\bs_waitcnt\b",
    "scratch_markers": r"\bscratch\b",
}

FIELDNAMES = (
    "kernel",
    "source",
    "status",
    "validation",
    "median_tops",
    "cv_tops_pct",
    "mlir_air_pct_of_air_tuned",
    "passes_air_tuned_95pct",
    "candidate_improvement_pct",
    "keep_candidate",
    "static_delta_wmma",
    "static_delta_global_load_b128",
    "static_delta_lds_ops",
    "static_delta_barriers",
    "static_delta_waitcnt",
    "static_delta_vgprs",
    "static_delta_sgprs",
    "static_delta_lds_bytes",
    "static_delta_scratch_markers",
    "static_delta_vgpr_spills",
    "static_delta_sgpr_spills",
    "profile_avg_ns",
    "profile_calls",
    "macro_tile_m",
    "macro_tile_n",
    "k_tile",
    "matrix_instruction",
    "workgroup_size",
    "wavefront_size",
    "group_segment_bytes",
    "private_segment_bytes",
    "vgprs",
    "sgprs",
    "vgpr_spills",
    "sgpr_spills",
    "instructions",
    "wmma",
    "global_load",
    "global_load_b128",
    "global_load_b64",
    "global_load_b32",
    "global_load_u8",
    "global_store",
    "global_store_b32",
    "ds_read",
    "ds_read_b128",
    "ds_write",
    "ds_write_b128",
    "ds_write_b8",
    "ds_swizzle",
    "barriers",
    "waitcnt",
    "scratch_markers",
    "wmma_per_barrier",
    "wmma_per_waitcnt",
    "global_load_b128_per_wmma",
    "lds_ops_per_wmma",
    "isa_path",
    "metadata_path",
    "profile_path",
    "notes",
)


@dataclass
class CommandResult:
    ok: bool
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class TextSymbol:
    address: int
    name: str
    binding: str
    symbol_type: str


@dataclass
class KernelEvidence:
    kernel: str
    source: str
    status: str = "PASS"
    validation: str = ""
    median_tops: str = ""
    cv_tops_pct: str = ""
    profile_avg_ns: str = ""
    profile_calls: str = ""
    macro_tile_m: str = ""
    macro_tile_n: str = ""
    k_tile: str = ""
    matrix_instruction: str = ""
    workgroup_size: str = ""
    wavefront_size: str = ""
    group_segment_bytes: str = ""
    private_segment_bytes: str = ""
    vgprs: str = ""
    sgprs: str = ""
    vgpr_spills: str = ""
    sgpr_spills: str = ""
    counters: dict[str, int] = field(default_factory=dict)
    isa_path: str = ""
    metadata_path: str = ""
    profile_path: str = ""
    notes: list[str] = field(default_factory=list)
    extra_fields: dict[str, str] = field(default_factory=dict)

    def to_row(self) -> dict[str, str]:
        row = {key: "" for key in FIELDNAMES}
        for key in (
            "kernel",
            "source",
            "status",
            "validation",
            "median_tops",
            "cv_tops_pct",
            "profile_avg_ns",
            "profile_calls",
            "macro_tile_m",
            "macro_tile_n",
            "k_tile",
            "matrix_instruction",
            "workgroup_size",
            "wavefront_size",
            "group_segment_bytes",
            "private_segment_bytes",
            "vgprs",
            "sgprs",
            "vgpr_spills",
            "sgpr_spills",
            "isa_path",
            "metadata_path",
            "profile_path",
        ):
            row[key] = str(getattr(self, key))
        for key in COUNTER_PATTERNS:
            row[key] = str(self.counters.get(key, ""))
        row["wmma_per_barrier"] = ratio(
            self.counters.get("wmma", 0), self.counters.get("barriers", 0)
        )
        row["wmma_per_waitcnt"] = ratio(
            self.counters.get("wmma", 0), self.counters.get("waitcnt", 0)
        )
        row["global_load_b128_per_wmma"] = ratio(
            self.counters.get("global_load_b128", 0), self.counters.get("wmma", 0)
        )
        lds_ops = self.counters.get("ds_read", 0) + self.counters.get("ds_write", 0)
        row["lds_ops_per_wmma"] = ratio(lds_ops, self.counters.get("wmma", 0))
        row.update(self.extra_fields)
        row["notes"] = "; ".join(note for note in self.notes if note)
        return row


def ratio(numerator: int | float, denominator: int | float) -> str:
    return f"{numerator / denominator:.6f}" if denominator else ""


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_capture(
    argv: Sequence[object],
    *,
    env: dict[str, str] | None = None,
    input_bytes: bytes | None = None,
) -> CommandResult:
    try:
        completed = subprocess.run(
            [str(arg) for arg in argv],
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            check=False,
        )
    except OSError as exc:
        return CommandResult(False, "", str(exc))
    return CommandResult(
        completed.returncode == 0,
        completed.stdout.decode("utf-8", errors="replace"),
        completed.stderr.decode("utf-8", errors="replace"),
    )


def rocm_root() -> Path:
    return Path(os.environ.get("ROCM_PATH", "/opt/rocm"))


def find_tool(name: str, rocm: Path, repo: Path) -> str | None:
    candidates = (
        repo / "llvm" / "install-amdgpu" / "bin" / name,
        rocm / "llvm" / "bin" / name,
        rocm / "lib" / "llvm" / "bin" / name,
        Path(f"/opt/rocm-7.2.0/lib/llvm/bin/{name}"),
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return shutil.which(name)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def benchmark_csv_rows(out_dir: Path) -> list[dict[str, str]]:
    return read_csv_rows(out_dir / "gpu_provider_baselines.csv")


def sweep_csv_rows(out_dir: Path) -> list[dict[str, str]]:
    return read_csv_rows(out_dir / "gpu_variant_sweep.csv")


def rocmlir_csv_rows(out_dir: Path) -> list[dict[str, str]]:
    return read_csv_rows(out_dir / "gpu_rocmlir_reference.csv")


def row_for_artifacts(rows: Sequence[dict[str, str]], artifacts: str) -> dict[str, str]:
    for row in rows:
        if row.get("artifacts") == artifacts:
            return row
    return {}


def row_for_variant(rows: Sequence[dict[str, str]], variant: str) -> dict[str, str]:
    for row in rows:
        if row.get("variant") == variant:
            return row
    return {}


def row_for_source(
    rows: Sequence[dict[str, str]], source: str, provider_prefix: str = ""
) -> dict[str, str]:
    for row in rows:
        if row.get("source") == source and (
            not provider_prefix or row.get("provider", "").startswith(provider_prefix)
        ):
            return row
    return {}


def row_for_provider(rows: Sequence[dict[str, str]], provider: str) -> dict[str, str]:
    for row in rows:
        if row.get("provider") == provider:
            return row
    return {}


def parse_csv_float(value: str) -> float | None:
    try:
        return float(value) if value not in {"", "n/a"} else None
    except ValueError:
        return None


def best_mlir_sweep_row(rows: Sequence[dict[str, str]]) -> dict[str, str]:
    passing = [row for row in rows if row.get("status") == "PASS"]
    timed = [
        row
        for row in passing
        if parse_csv_float(row.get("median_tops", "")) is not None
    ]
    if timed:
        return max(
            timed, key=lambda row: parse_csv_float(row.get("median_tops", "")) or 0.0
        )
    return passing[0] if passing else {}


def first_profile_row(path: Path, kernel_substring: str = "") -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if kernel_substring:
        for row in rows:
            if kernel_substring in row.get("Name", ""):
                return row
    return rows[0] if rows else {}


def provider_kernel_from_profile(
    out_dir: Path, provider: str
) -> tuple[str, Path, dict[str, str]]:
    profile = (
        out_dir
        / "build"
        / "gpu_provider_baselines"
        / "artifacts"
        / "profiles"
        / provider
        / f"{provider}_kernel_stats.csv"
    )
    row = first_profile_row(profile)
    return row.get("Name", ""), profile, row


def count_isa(isa: str) -> dict[str, int]:
    return {
        name: len(re.findall(pattern, isa, flags=re.MULTILINE))
        for name, pattern in COUNTER_PATTERNS.items()
    }


def parse_metadata_block(readobj: str, kernel_name: str) -> str:
    name_match = re.search(
        rf"^\s*(?:-\s*)?\.name:\s+{re.escape(kernel_name)}\s*$",
        readobj,
        flags=re.MULTILINE,
    )
    idx = (
        name_match.start()
        if name_match
        else readobj.find(f".name:           {kernel_name}")
    )
    if idx < 0:
        return ""
    start = readobj.rfind("\n  - ", 0, idx)
    if start < 0:
        start = readobj.rfind("\n- ", 0, idx)
    if start < 0:
        start = readobj.rfind("\namdhsa.kernels:", 0, idx)
    if start < 0:
        start = 0
    end = readobj.find("\n  - ", idx + len(kernel_name))
    if end < 0:
        end = readobj.find("\namdhsa.target:", idx)
    return readobj[start:end] if end >= 0 else readobj[start:]


def metadata_name_containing(readobj: str, substring: str) -> str:
    for match in re.finditer(
        r"^\s*(?:-\s*)?\.name:\s+([^\n]+)$", readobj, flags=re.MULTILINE
    ):
        name = match.group(1).strip()
        if substring in name:
            return name
    return ""


def first_metadata_name(readobj: str) -> str:
    match = re.search(r"^\s*(?:-\s*)?\.name:\s+([^\n]+)$", readobj, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def metadata_value(block: str, key: str) -> str:
    match = re.search(rf"\.{re.escape(key)}:\s+([^\n]+)", block)
    return match.group(1).strip() if match else ""


def apply_asm_metadata(evidence: KernelEvidence, isa_text: str) -> bool:
    def value(name: str) -> str:
        match = re.search(rf"\.amdhsa_{re.escape(name)}\s+([0-9]+)", isa_text)
        return match.group(1) if match else ""

    found = False
    for attr, name in (
        ("group_segment_bytes", "group_segment_fixed_size"),
        ("private_segment_bytes", "private_segment_fixed_size"),
        ("vgprs", "next_free_vgpr"),
        ("sgprs", "next_free_sgpr"),
    ):
        parsed = value(name)
        if parsed and not getattr(evidence, attr):
            setattr(evidence, attr, parsed)
            found = True
    if not evidence.wavefront_size and re.search(
        r"\.amdhsa_wavefront_size32\s+1", isa_text
    ):
        evidence.wavefront_size = "32"
        found = True
    if found:
        evidence.notes.append("asm_metadata=amdhsa_directives")
    return found


def apply_metadata(
    evidence: KernelEvidence, readobj_text: str, kernel_name: str
) -> None:
    block = parse_metadata_block(readobj_text, kernel_name)
    if not block:
        containing_name = metadata_name_containing(readobj_text, kernel_name)
        if containing_name:
            block = parse_metadata_block(readobj_text, containing_name)
            evidence.notes.append(f"metadata_symbol={containing_name}")
    if not block:
        evidence.notes.append("metadata block not found")
        return
    evidence.group_segment_bytes = metadata_value(block, "group_segment_fixed_size")
    evidence.private_segment_bytes = metadata_value(block, "private_segment_fixed_size")
    evidence.workgroup_size = metadata_value(block, "max_flat_workgroup_size")
    evidence.wavefront_size = metadata_value(block, "wavefront_size")
    evidence.vgprs = metadata_value(block, "vgpr_count")
    evidence.sgprs = metadata_value(block, "sgpr_count")
    evidence.vgpr_spills = metadata_value(block, "vgpr_spill_count")
    evidence.sgpr_spills = metadata_value(block, "sgpr_spill_count")


def parse_mlir_schedule(evidence: KernelEvidence, summary_text: str) -> None:
    variant = ""
    match = re.search(
        r"^int8_gemm_variant:\s+([^\n]+)", summary_text, flags=re.MULTILINE
    )
    if match:
        variant = match.group(1).strip()
    tile = re.search(r"(?:lds|global)_(\d+)x(\d+)", variant)
    if tile:
        evidence.macro_tile_m, evidence.macro_tile_n = tile.group(1), tile.group(2)
    ktile = re.search(r"_k(\d+)", variant)
    evidence.k_tile = (
        "16"
        if variant in {DIRECT_CANONICAL_VARIANT, DIRECT_RAWPTR_U2_VARIANT}
        else (
            "32" if variant in DIRECT_VARIANTS else (ktile.group(1) if ktile else "64")
        )
    )
    evidence.matrix_instruction = "16x16x16_iu8"
    if variant:
        evidence.notes.append(f"variant={variant}")
    if variant in DIRECT_VARIANTS:
        evidence.notes.append("wave_tile=64x64")
        evidence.notes.append("groupM=8")
        evidence.notes.append("B_layout=packed_NxK")
        evidence.notes.append("pipeline=direct_global_no_lds")
        if variant == DIRECT_CANONICAL_VARIANT:
            evidence.notes.append("schedule=canonical_k16_no_lds")
        elif variant == DIRECT_PREFETCH_VARIANT:
            evidence.notes.append("schedule=register_prefetch_k32")
        elif variant == DIRECT_RAWPTR_VARIANT:
            evidence.notes.append("schedule=rawptr_linear_grid_k32")
            evidence.notes.append("kernel_abi=bare_ptr")
        elif variant == DIRECT_RAWPTR_U2_VARIANT:
            evidence.notes.append("schedule=rawptr_linear_grid_k16_unroll2")
            evidence.notes.append("kernel_abi=bare_ptr")


def parse_tensile_schedule(evidence: KernelEvidence, kernel: str) -> None:
    mt = re.search(r"_MT(\d+)x(\d+)x(\d+)_", kernel)
    if mt:
        evidence.macro_tile_m, evidence.macro_tile_n, evidence.k_tile = (
            mt.group(1),
            mt.group(2),
            mt.group(3),
        )
    mi = re.search(r"_MI(\d+)x(\d+)x(\d+)x\d+_", kernel)
    if mi:
        evidence.matrix_instruction = "x".join(mi.groups()) + "_iu8"
    wg = re.search(r"_WG(\d+)_(\d+)_(\d+)", kernel)
    if wg:
        evidence.notes.append(
            f"workgroup_shape={wg.group(1)}x{wg.group(2)}x{wg.group(3)}"
        )
    for token in ("GLVWA", "GLVWB", "LPB", "PGR", "PLR", "WS"):
        match = re.search(rf"_{token}([A-Za-z0-9]+)", kernel)
        if match:
            evidence.notes.append(f"{token}={match.group(1)}")


def extract_tensile_hsaco_binary_safe(
    co_path: Path, artifacts_dir: Path
) -> tuple[Path | None, str]:
    if not co_path.exists():
        return None, f"Tensile object not found: {co_path}"
    data = co_path.read_bytes()
    if data.startswith(ELF_MAGIC):
        hsaco = artifacts_dir / "tensile_extracted.hsaco"
        hsaco.write_bytes(data)
        return hsaco, "input was already an ELF code object"
    zstd_offset = data.find(ZSTD_MAGIC)
    if zstd_offset < 0:
        return None, "CCOB zstd payload not found"
    zstd = shutil.which("zstd") or shutil.which("zstdcat") or shutil.which("unzstd")
    if not zstd:
        return None, "zstd command not found"
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(data[zstd_offset:])
        temp_name = temp.name
    try:
        cmd = (
            [zstd, "-dc", temp_name]
            if Path(zstd).name != "zstdcat"
            else [zstd, temp_name]
        )
        completed = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
        )
    finally:
        try:
            Path(temp_name).unlink()
        except OSError:
            pass
    if completed.returncode != 0:
        return (
            None,
            f"zstd decompression failed: {completed.stderr.decode('utf-8', errors='replace').strip()}",
        )
    elf_offset = completed.stdout.find(ELF_MAGIC)
    if elf_offset < 0:
        return None, "decompressed CCOB payload did not contain an ELF code object"
    hsaco = artifacts_dir / "tensile_extracted.hsaco"
    hsaco.write_bytes(completed.stdout[elf_offset:])
    return (
        hsaco,
        f"extracted zstd payload at CCOB offset {zstd_offset}, ELF offset {elf_offset}",
    )


def parse_objdump_symbols(symbol_table: str) -> list[TextSymbol]:
    symbols: list[TextSymbol] = []
    for line in symbol_table.splitlines():
        parts = line.split()
        if len(parts) < 5 or not re.fullmatch(r"[0-9A-Fa-f]+", parts[0]):
            continue
        address = int(parts[0], 16)
        binding = parts[1]
        symbol_type = ""
        name = ""
        if len(parts) >= 6 and parts[2] == "F" and parts[3] == ".text":
            symbol_type = "F"
            name = " ".join(parts[5:])
        elif parts[2] == ".text":
            name = " ".join(parts[4:])
        else:
            continue
        if name.startswith(".protected "):
            name = name[len(".protected ") :]
        if name:
            symbols.append(TextSymbol(address, name, binding, symbol_type))
    return symbols


def select_text_symbol(
    symbols: Sequence[TextSymbol], kernel: str
) -> tuple[TextSymbol | None, str]:
    for symbol in symbols:
        if (
            symbol.name == kernel
            and symbol.binding == "g"
            and symbol.symbol_type == "F"
        ):
            return symbol, "matched public rocprof kernel symbol"
    for symbol in symbols:
        if symbol.name == kernel:
            return (
                symbol,
                "matched rocprof kernel symbol without public-function binding",
            )
    prefix = kernel.replace("_BH_MT", "_BH_GB_MT")
    matches = [
        symbol
        for symbol in symbols
        if symbol.name.startswith(prefix) and symbol.name.endswith("_preloaded")
    ]
    if matches:
        return (
            matches[0],
            "rocprof symbol was a wrapper; matched first GB preloaded compute symbol",
        )
    return None, "matching Tensile text symbol not found"


def next_public_function_address(
    symbols: Sequence[TextSymbol], start: int
) -> int | None:
    candidates = sorted(
        symbol.address
        for symbol in symbols
        if symbol.address > start
        and symbol.binding == "g"
        and symbol.symbol_type == "F"
    )
    return candidates[0] if candidates else None


def provider_symbol_body(isa: str, symbol_substring: str) -> str:
    lines = isa.splitlines()
    start = None
    for index, line in enumerate(lines):
        if symbol_substring in line and re.search(
            r"^[0-9a-fA-F]+\s+<.*>:", line.strip()
        ):
            start = index
            break
    if start is None:
        return ""
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if re.search(r"^[0-9a-fA-F]+\s+<.*>:", lines[index].strip()):
            end = index
            break
    return "\n".join(lines[start:end])


def read_semicolon_paths(value: str) -> list[Path]:
    return [Path(item) for item in value.split(";") if item]


def disassemble_tensile_region(
    llvm_objdump: str,
    hsaco: Path,
    symbol: str,
    output: Path,
    gpu_arch: str,
) -> tuple[bool, str, str]:
    symbols_path = output.with_suffix(".symbols.txt")
    symbols_result = run_capture([llvm_objdump, "-t", hsaco])
    if not symbols_result.ok:
        fallback = run_capture(
            [
                llvm_objdump,
                "-d",
                f"--mcpu={gpu_arch}",
                f"--disassemble-symbols={symbol}",
                hsaco,
            ]
        )
        if fallback.ok:
            write_text(output, fallback.stdout)
            return (
                True,
                symbol,
                "symbol table unavailable; used --disassemble-symbols fallback",
            )
        return False, symbol, symbols_result.stderr.strip() or fallback.stderr.strip()
    write_text(symbols_path, symbols_result.stdout)
    symbols = parse_objdump_symbols(symbols_result.stdout)
    selected, note = select_text_symbol(symbols, symbol)
    if not selected:
        return False, symbol, note
    stop = next_public_function_address(symbols, selected.address)
    argv: list[object] = [
        llvm_objdump,
        "-d",
        f"--mcpu={gpu_arch}",
        f"--start-address=0x{selected.address:x}",
    ]
    if stop:
        argv.append(f"--stop-address=0x{stop:x}")
    argv.append(hsaco)
    result = run_capture(argv)
    if not result.ok:
        return False, selected.name, result.stderr.strip()
    write_text(output, result.stdout)
    if stop:
        note = f"{note}; disassembled 0x{selected.address:x}..0x{stop:x}"
    else:
        note = f"{note}; disassembled from 0x{selected.address:x} to EOF"
    return True, selected.name, note


def readobj_notes(llvm_readobj: str, hsaco: Path, output: Path) -> tuple[str, str]:
    result = run_capture(
        [llvm_readobj, "--file-headers", "--notes", "--sections", "--symbols", hsaco]
    )
    if not result.ok:
        return "", result.stderr.strip()
    write_text(output, result.stdout)
    return result.stdout, ""


def profile_mlir_air(
    args: argparse.Namespace, repo: Path, artifacts_dir: Path
) -> tuple[Path | None, str]:
    final_mlir = args.mlir_final_mlir
    if final_mlir is None:
        sweep_rows = sweep_csv_rows(args.out_dir)
        direct_row = row_for_variant(sweep_rows, AIR_TUNED_DIRECT_VARIANT)
        mlir_row = (
            row_for_source(benchmark_csv_rows(args.out_dir), "mlir-air")
            or best_mlir_sweep_row(sweep_rows)
            or direct_row
        )
        row_artifacts = mlir_row.get("artifacts", "")
        if row_artifacts:
            final_mlir = Path(row_artifacts) / "gpu_int8_gemm.final.mlir"
        else:
            final_mlir = (
                args.out_dir
                / "build"
                / "gpu"
                / "artifacts"
                / "gpu_int8_gemm.final.mlir"
            )
    if not final_mlir.exists():
        return None, f"MLIR final artifact not found: {final_mlir}"
    mlir_runner = (
        args.mlir_runner
        or os.environ.get("MLIR_RUNNER")
        or repo / "llvm" / "install-amdgpu" / "bin" / "mlir-runner"
    )
    airgpu = (
        args.airgpu_lib
        or os.environ.get("AIRGPU_LIB")
        or repo / "build-gpu" / "lib" / "libairgpu.so"
    )
    rocprof = args.rocprof or shutil.which("rocprofv3") or "/opt/rocm/bin/rocprofv3"
    if not Path(mlir_runner).exists():
        return None, f"mlir-runner not found: {mlir_runner}"
    if not Path(airgpu).exists():
        return None, f"libairgpu.so not found: {airgpu}"
    if not Path(rocprof).exists():
        return None, f"rocprofv3 not found: {rocprof}"
    profile_dir = artifacts_dir / "profiles" / "mlir_air"
    profile_dir.mkdir(parents=True, exist_ok=True)
    log = artifacts_dir / "mlir_air_rocprof.log"
    env = os.environ.copy()
    env.setdefault("AIRGPU_USE_HIP_MALLOC", "1")
    argv = [
        rocprof,
        "--kernel-trace",
        "--stats",
        "--summary",
        "--output-format",
        "csv",
        "--output-directory",
        profile_dir,
        "--output-file",
        "mlir_air",
        "--",
        mlir_runner,
        "--entry-point-result=void",
        f"--shared-libs={airgpu}",
        final_mlir,
    ]
    with log.open("w", encoding="utf-8") as output:
        output.write("+ " + " ".join(shlex.quote(str(arg)) for arg in argv) + "\n")
        output.flush()
        completed = subprocess.run(
            [str(arg) for arg in argv],
            stdout=output,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    if completed.returncode != 0:
        return log, f"MLIR rocprof run failed; see {log}"
    return (
        profile_dir / "mlir_air_kernel_stats.csv",
        f"MLIR rocprof profile captured at {profile_dir}",
    )


def build_mlir_evidence(
    args: argparse.Namespace,
    rows: Sequence[dict[str, str]],
    llvm_readobj: str,
    artifacts_dir: Path,
) -> KernelEvidence:
    provider_row = row_for_source(rows, "mlir-air")
    sweep_rows = sweep_csv_rows(args.out_dir)
    direct_row = row_for_variant(sweep_rows, AIR_TUNED_DIRECT_VARIANT)
    best_sweep_row = best_mlir_sweep_row(sweep_rows)
    gpu_artifacts = args.out_dir / "build" / "gpu" / "artifacts"
    selected_row = provider_row or best_sweep_row or direct_row
    row_artifacts = selected_row.get("artifacts", "")
    sweep_row = row_for_artifacts(sweep_rows, row_artifacts) if row_artifacts else {}
    if row_artifacts and Path(row_artifacts).exists():
        gpu_artifacts = Path(row_artifacts)
    isa_path = args.mlir_isa or gpu_artifacts / "gpu_int8_gemm.isa.s"
    hsaco = args.mlir_hsaco or gpu_artifacts / "gpu_int8_gemm.hsaco"
    summary = read_text(gpu_artifacts / "gpu_int8_gemm.summary.txt")
    evidence = KernelEvidence("forward_module", "mlir-air")
    evidence.isa_path = str(isa_path)
    if provider_row:
        evidence.validation = provider_row.get("validation", "")
        evidence.notes.append("selected_from=gpu_provider_baselines")
    elif best_sweep_row:
        evidence.validation = "PASS" if best_sweep_row.get("status") == "PASS" else ""
        evidence.notes.append("selected_from=gpu_variant_sweep")
    elif direct_row:
        evidence.validation = "PASS" if direct_row.get("status") == "PASS" else ""
        evidence.notes.append("selected_from=gpu_variant_sweep")
    else:
        evidence.validation = provider_row.get("validation", "")
    evidence.median_tops = selected_row.get("median_tops", "") or sweep_row.get(
        "median_tops", ""
    )
    evidence.cv_tops_pct = selected_row.get("cv_tops_pct", "") or sweep_row.get(
        "cv_tops_pct", ""
    )
    if selected_row.get("status"):
        evidence.status = selected_row["status"]
    if not isa_path.exists():
        evidence.status = "FAIL"
        evidence.notes.append(f"MLIR ISA not found: {isa_path}")
        return evidence
    evidence.counters = count_isa(read_text(isa_path))
    parse_mlir_schedule(evidence, summary)
    if any(
        note.startswith("variant=") and note.split("=", 1)[1] in DIRECT_VARIANTS
        for note in evidence.notes
    ):
        evidence.source = "compiler-generated direct"
    readobj_path = gpu_artifacts / "gpu_int8_gemm.code_object.readobj.txt"
    readobj_text = read_text(readobj_path)
    if not readobj_text and hsaco.exists():
        readobj_path = artifacts_dir / "mlir_air.readobj.txt"
        readobj_text, err = readobj_notes(llvm_readobj, hsaco, readobj_path)
        if err:
            evidence.notes.append(err)
    evidence.metadata_path = str(readobj_path) if readobj_path.exists() else ""
    apply_metadata(evidence, readobj_text, "forward_module")
    profile_candidates = (
        args.out_dir
        / "build"
        / "gpu"
        / "artifacts"
        / "profiles"
        / "mlir_air"
        / "mlir_air_kernel_stats.csv",
        artifacts_dir / "profiles" / "mlir_air" / "mlir_air_kernel_stats.csv",
    )
    for profile_path in profile_candidates:
        profile = first_profile_row(profile_path, "forward_module")
        if profile:
            evidence.profile_path = str(profile_path)
            evidence.profile_avg_ns = profile.get("AverageNs", "")
            evidence.profile_calls = profile.get("Calls", "")
            break
    else:
        evidence.notes.append(
            "MLIR rocprof stats not found; rerun analyzer with --profile-mlir for dynamic kernel stats"
        )
    annotate_mlir_candidate_retention(evidence, sweep_rows, selected_row, sweep_row)
    return evidence


def build_air_tuned_evidence(
    args: argparse.Namespace, rows: Sequence[dict[str, str]], artifacts_dir: Path
) -> KernelEvidence:
    provider_row = row_for_provider(rows, DEFAULT_AIR_TUNED_PROVIDER)
    evidence = KernelEvidence("airTuned128x128Kernel", "air-owned")
    evidence.macro_tile_m = "128"
    evidence.macro_tile_n = "128"
    evidence.k_tile = "32"
    evidence.matrix_instruction = "16x16x16_iu8"
    evidence.workgroup_size = "128"
    evidence.wavefront_size = "32"
    evidence.notes.append("fixed_contract=1024x1024x1024_i8_i8_i32")
    evidence.notes.append("wave_tile=64x64")
    evidence.notes.append("groupM=8")
    evidence.notes.append("B_layout=packed_NxK")
    if not provider_row:
        evidence.status = "SKIP"
        evidence.notes.append(
            "air_tuned provider row not found; rerun benchmark with --gpu-provider-baselines"
        )
        return evidence
    evidence.status = provider_row.get("status", "") or evidence.status
    evidence.validation = provider_row.get("validation", "")
    evidence.median_tops = provider_row.get("median_tops", "")
    evidence.cv_tops_pct = provider_row.get("cv_tops_pct", "")
    disasm_paths = read_semicolon_paths(provider_row.get("disassemble_log", ""))
    evidence.isa_path = provider_row.get("disassemble_log", "")
    isa_text = "\n".join(read_text(path) for path in disasm_paths)
    body = provider_symbol_body(isa_text, "airTuned128x128Kernel")
    if body:
        evidence.counters = count_isa(body)
    elif isa_text:
        evidence.status = "WARN" if evidence.status == "PASS" else evidence.status
        evidence.counters = count_isa(isa_text)
        evidence.notes.append(
            "air_tuned symbol body not isolated; counters cover full provider code object"
        )
    else:
        evidence.status = "WARN"
        evidence.notes.append("air_tuned provider ISA not found")
    profile_kernel, profile_path, profile = provider_kernel_from_profile(
        args.out_dir, DEFAULT_AIR_TUNED_PROVIDER
    )
    if profile_kernel:
        evidence.kernel = profile_kernel
    if profile_path.exists():
        evidence.profile_path = str(profile_path)
    if profile:
        evidence.profile_avg_ns = profile.get("AverageNs", "")
        evidence.profile_calls = profile.get("Calls", "")
    readobj_paths = (
        sorted(Path(provider_row.get("artifacts", "")).glob("provider_*.readobj.txt"))
        if provider_row.get("artifacts")
        else []
    )
    for readobj_path in readobj_paths:
        readobj_text = read_text(readobj_path)
        if "airTuned128x128Kernel" in readobj_text:
            evidence.metadata_path = str(readobj_path)
            apply_metadata(evidence, readobj_text, "airTuned128x128Kernel")
            break
    else:
        evidence.notes.append("air_tuned metadata block not found")
    return evidence


def first_existing(paths: Sequence[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def first_glob(root: Path, patterns: Sequence[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    return None


def first_semicolon_path(value: str, predicate=lambda path: True) -> Path | None:
    for path in read_semicolon_paths(value):
        if path.exists() and predicate(path):
            return path
    return None


def rocmlir_reference_row(
    out_dir: Path, provider_rows: Sequence[dict[str, str]]
) -> dict[str, str]:
    reference_rows = rocmlir_csv_rows(out_dir)
    row = row_for_provider(reference_rows, ROCMLIR_REFERENCE_PROVIDER)
    return row or row_for_provider(provider_rows, ROCMLIR_REFERENCE_PROVIDER)


def find_reference_artifact(
    root: Path | None, explicit: Path | None, patterns: Sequence[str]
) -> Path | None:
    if explicit:
        return explicit if explicit.exists() else explicit
    if not root or not root.exists():
        return None
    return first_glob(root, patterns)


def reference_profile_path(row: dict[str, str], root: Path | None) -> Path | None:
    profile = first_semicolon_path(
        row.get("profile_log", ""), lambda path: path.suffix == ".csv"
    )
    if profile:
        return profile
    if root and root.exists():
        return first_glob(root, ("*kernel_stats.csv", "*profile*.csv", "*rocprof*.csv"))
    return None


def parse_rocmlir_schedule(
    evidence: KernelEvidence, kernel: str, mlir_text: str
) -> None:
    parse_tensile_schedule(evidence, kernel)
    if (
        not evidence.macro_tile_m
        and evidence.group_segment_bytes == "24576"
        and evidence.counters.get("wmma", 0) >= 128
    ):
        evidence.macro_tile_m = "128"
        evidence.macro_tile_n = "128"
        evidence.k_tile = "32"
        evidence.notes.append("inferred_schedule=lds_pipe3_128x128x32")
    if not evidence.matrix_instruction and (
        "wmma" in mlir_text.lower() or evidence.counters.get("wmma", 0)
    ):
        evidence.matrix_instruction = "16x16x16_iu8"
    tile = re.search(
        r"(?:MT|macro[_-]?tile[_-]?)(\d+)x(\d+)(?:x(\d+))?",
        kernel + "\n" + mlir_text,
        flags=re.IGNORECASE,
    )
    if tile and not evidence.macro_tile_m:
        evidence.macro_tile_m, evidence.macro_tile_n = tile.group(1), tile.group(2)
        if tile.group(3):
            evidence.k_tile = tile.group(3)
    if re.search(r"\b(?:amdgpu|rocdl)\.wmma\b", mlir_text):
        evidence.notes.append("mlir_contains_wmma_op")
    if re.search(r"\bamdgpu\.mfma\b", mlir_text):
        evidence.notes.append("mlir_contains_mfma_op")


def build_rocmlir_reference_evidence(
    args: argparse.Namespace,
    rows: Sequence[dict[str, str]],
    llvm_objdump: str,
    llvm_readobj: str,
    artifacts_dir: Path,
) -> KernelEvidence:
    row = rocmlir_reference_row(args.out_dir, rows)
    root_text = str(args.rocmlir_artifacts_dir or row.get("artifacts", ""))
    root = Path(root_text) if root_text else None
    isa_path = args.rocmlir_isa or first_semicolon_path(
        row.get("disassemble_log", ""),
        lambda path: path.suffix in {".s", ".isa"} or path.name.endswith(".isa.s"),
    )
    hsaco = args.rocmlir_hsaco or find_reference_artifact(
        root,
        None,
        ("rocmlir_reference.hsaco", "rocmlir.hsaco", "kernel.hsaco", "*.hsaco", "*.co"),
    )
    readobj_path = args.rocmlir_readobj or first_semicolon_path(
        row.get("profile_log", ""),
        lambda path: "readobj" in path.name or "metadata" in path.name,
    )
    readobj_path = readobj_path or find_reference_artifact(
        root,
        None,
        (
            "rocmlir_reference.readobj.txt",
            "rocmlir.readobj.txt",
            "kernel.readobj.txt",
            "*.readobj.txt",
            "*readobj*.txt",
            "*metadata*.txt",
        ),
    )
    mlir_path = find_reference_artifact(
        root,
        args.rocmlir_mlir,
        ("rocmlir_reference.mlir", "rocmlir.mlir", "kernel.mlir", "*.mlir"),
    )
    if not isa_path:
        isa_path = find_reference_artifact(
            root,
            None,
            (
                "rocmlir_reference.isa.s",
                "rocmlir.isa.s",
                "kernel.isa.s",
                "*.isa.s",
                "*.s",
            ),
        )
    evidence = KernelEvidence(
        args.rocmlir_symbol or "rocmlir_reference", ROCMLIR_REFERENCE_SOURCE
    )
    evidence.validation = row.get("validation", "")
    evidence.median_tops = row.get("median_tops", "")
    evidence.cv_tops_pct = row.get("cv_tops_pct", "")
    if row.get("status"):
        evidence.status = row["status"]
    evidence.notes.append("fixed_contract=1024x1024x1024_i8_i8_i32")
    if row.get("notes"):
        evidence.notes.append(row["notes"])
    if not root and not isa_path and not hsaco and not readobj_path and not mlir_path:
        evidence.status = "SKIP"
        evidence.notes.append(
            "rocMLIR reference artifacts not found; pass --rocmlir-artifacts-dir or run.py --gpu-rocmlir-reference"
        )
        return evidence
    if not isa_path and hsaco and hsaco.exists():
        isa_path = artifacts_dir / "rocmlir_reference.isa.s"
        result = run_capture([llvm_objdump, "-d", f"--mcpu={args.gpu_arch}", hsaco])
        if result.ok:
            write_text(isa_path, result.stdout)
            evidence.notes.append(f"disassembled_hsaco={hsaco}")
        else:
            evidence.notes.append(
                result.stderr.strip() or f"rocMLIR HSACO disassembly failed: {hsaco}"
            )
    readobj_text = read_text(readobj_path) if readobj_path else ""
    if not readobj_text and hsaco and hsaco.exists():
        readobj_path = artifacts_dir / "rocmlir_reference.readobj.txt"
        readobj_text, err = readobj_notes(llvm_readobj, hsaco, readobj_path)
        if err:
            evidence.notes.append(err)
    if readobj_text and evidence.kernel == "rocmlir_reference":
        evidence.kernel = (
            args.rocmlir_symbol or first_metadata_name(readobj_text) or evidence.kernel
        )
    asm_metadata = False
    if isa_path and isa_path.exists():
        evidence.isa_path = str(isa_path)
        isa_text = read_text(isa_path)
        evidence.counters = count_isa(isa_text)
        asm_metadata = apply_asm_metadata(evidence, isa_text)
    else:
        evidence.status = "WARN" if evidence.status != "SKIP" else evidence.status
        evidence.notes.append(
            "rocMLIR ISA not found; static instruction counters unavailable"
        )
    if readobj_path and readobj_path.exists():
        evidence.metadata_path = str(readobj_path)
        apply_metadata(evidence, readobj_text, evidence.kernel)
    elif evidence.status != "SKIP" and not asm_metadata:
        evidence.notes.append("rocMLIR metadata/readobj not found")
    elif evidence.status != "SKIP" and asm_metadata:
        evidence.notes.append("rocMLIR metadata inferred from ASM directives")
    mlir_text = read_text(mlir_path) if mlir_path else ""
    if mlir_path and mlir_path.exists():
        evidence.notes.append(f"mlir={mlir_path}")
    if hsaco and hsaco.exists():
        evidence.notes.append(f"hsaco={hsaco}")
    parse_rocmlir_schedule(evidence, evidence.kernel, mlir_text)
    if evidence.counters.get("wmma", 0) and not evidence.matrix_instruction:
        evidence.matrix_instruction = "16x16x16_iu8"
    profile_path = args.rocmlir_profile or reference_profile_path(row, root)
    if profile_path:
        profile = first_profile_row(
            profile_path,
            evidence.kernel if evidence.kernel != "rocmlir_reference" else "",
        )
        evidence.profile_path = str(profile_path)
        if profile:
            evidence.profile_avg_ns = profile.get("AverageNs", "")
            evidence.profile_calls = profile.get("Calls", "")
    return evidence


def build_tensile_evidence(
    args: argparse.Namespace,
    rows: Sequence[dict[str, str]],
    llvm_objdump: str,
    llvm_readobj: str,
    artifacts_dir: Path,
) -> KernelEvidence:
    kernel, profile_path, profile = provider_kernel_from_profile(
        args.out_dir, DEFAULT_PROVIDER
    )
    if not kernel:
        kernel = args.tensile_symbol or ""
    if not kernel:
        evidence = KernelEvidence("unknown", "rocblas-tensile", status="WARN")
        evidence.notes.append("rocBLAS/Tensile profile kernel name not found")
        return evidence
    evidence = KernelEvidence(kernel, "rocblas-tensile")
    evidence.profile_path = str(profile_path) if profile_path.exists() else ""
    if profile:
        evidence.profile_avg_ns = profile.get("AverageNs", "")
        evidence.profile_calls = profile.get("Calls", "")
    provider_row = row_for_source(rows, "external", DEFAULT_PROVIDER)
    evidence.validation = provider_row.get("validation", "")
    evidence.median_tops = provider_row.get("median_tops", "")
    evidence.cv_tops_pct = provider_row.get("cv_tops_pct", "")
    if provider_row.get("status"):
        evidence.status = provider_row["status"]
    parse_tensile_schedule(evidence, kernel)
    tensile_co = (
        args.tensile_co
        or rocm_root() / "lib" / "rocblas" / "library" / DEFAULT_TENSILE_LIBRARY
    )
    hsaco, note = extract_tensile_hsaco_binary_safe(tensile_co, artifacts_dir)
    evidence.notes.append(note)
    if not hsaco:
        evidence.status = "WARN"
        return evidence
    disasm = artifacts_dir / "tensile_selected.isa.s"
    ok, disassembled_symbol, disasm_note = disassemble_tensile_region(
        llvm_objdump, hsaco, kernel, disasm, args.gpu_arch
    )
    if ok:
        evidence.isa_path = str(disasm)
        evidence.counters = count_isa(read_text(disasm))
        if disassembled_symbol != kernel:
            evidence.notes.append(f"disassembled_symbol={disassembled_symbol}")
        evidence.notes.append(disasm_note)
    else:
        evidence.status = "WARN"
        evidence.notes.append(f"Tensile symbol disassembly failed: {disasm_note}")
    readobj_path = artifacts_dir / "tensile_extracted.readobj.txt"
    readobj_text, err = readobj_notes(llvm_readobj, hsaco, readobj_path)
    if err:
        evidence.notes.append(err)
    evidence.metadata_path = str(readobj_path) if readobj_path.exists() else ""
    apply_metadata(evidence, readobj_text, kernel)
    return evidence


def numeric(value: str) -> float | None:
    try:
        return float(value) if value not in {"", "n/a"} else None
    except ValueError:
        return None


def annotate_mlir_candidate_retention(
    evidence: KernelEvidence,
    sweep_rows: Sequence[dict[str, str]],
    selected_row: dict[str, str],
    sweep_row: dict[str, str],
) -> None:
    selected_variant = sweep_row.get("variant") or selected_row.get("variant", "")
    baseline_row = row_for_variant(sweep_rows, ACCEPTED_MLIR_AIR_BASELINE_VARIANT)
    baseline_tops = parse_csv_float(baseline_row.get("median_tops", ""))
    selected_tops = numeric(evidence.median_tops)
    if selected_variant == ACCEPTED_MLIR_AIR_BASELINE_VARIANT:
        evidence.extra_fields["candidate_improvement_pct"] = "0.000"
        evidence.extra_fields["keep_candidate"] = "accepted_best"
        return
    if baseline_tops is None or baseline_tops <= 0.0 or selected_tops is None:
        return
    improvement = ((selected_tops - baseline_tops) / baseline_tops) * 100.0
    evidence.extra_fields["candidate_improvement_pct"] = f"{improvement:.3f}"
    validated = evidence.status == "PASS" and evidence.validation in {"", "PASS"}
    no_scratch = evidence.counters.get("scratch_markers", 0) == 0
    no_spills = evidence.vgpr_spills in {"", "0"} and evidence.sgpr_spills in {"", "0"}
    keep = (
        validated
        and no_scratch
        and no_spills
        and improvement >= CANDIDATE_IMPROVEMENT_PCT
    )
    evidence.extra_fields["keep_candidate"] = "yes" if keep else "no"


def static_delta(lhs: str, rhs: str) -> str:
    lhs_value = numeric(lhs)
    rhs_value = numeric(rhs)
    if lhs_value is None or rhs_value is None:
        return ""
    delta = lhs_value - rhs_value
    return f"{delta:+.0f}" if delta.is_integer() else f"{delta:+.3f}"


def annotate_mlir_air_tuned_gate(
    mlir: KernelEvidence, air_tuned: KernelEvidence
) -> None:
    mlir_tops = numeric(mlir.median_tops)
    tuned_tops = numeric(air_tuned.median_tops)
    if mlir_tops is not None and tuned_tops is not None and tuned_tops > 0.0:
        pct = (mlir_tops / tuned_tops) * 100.0
        mlir.extra_fields["mlir_air_pct_of_air_tuned"] = f"{pct:.3f}"
        mlir.extra_fields["passes_air_tuned_95pct"] = (
            "yes" if pct >= AIR_TUNED_ACCEPTANCE_PCT else "no"
        )
    m_row = mlir.to_row()
    a_row = air_tuned.to_row()
    lds_ops_m = str(mlir.counters.get("ds_read", 0) + mlir.counters.get("ds_write", 0))
    lds_ops_a = str(
        air_tuned.counters.get("ds_read", 0) + air_tuned.counters.get("ds_write", 0)
    )
    delta_sources = {
        "static_delta_wmma": (m_row.get("wmma", ""), a_row.get("wmma", "")),
        "static_delta_global_load_b128": (
            m_row.get("global_load_b128", ""),
            a_row.get("global_load_b128", ""),
        ),
        "static_delta_lds_ops": (lds_ops_m, lds_ops_a),
        "static_delta_barriers": (m_row.get("barriers", ""), a_row.get("barriers", "")),
        "static_delta_waitcnt": (m_row.get("waitcnt", ""), a_row.get("waitcnt", "")),
        "static_delta_vgprs": (m_row.get("vgprs", ""), a_row.get("vgprs", "")),
        "static_delta_sgprs": (m_row.get("sgprs", ""), a_row.get("sgprs", "")),
        "static_delta_lds_bytes": (
            m_row.get("group_segment_bytes", ""),
            a_row.get("group_segment_bytes", ""),
        ),
        "static_delta_scratch_markers": (
            m_row.get("scratch_markers", ""),
            a_row.get("scratch_markers", ""),
        ),
        "static_delta_vgpr_spills": (
            m_row.get("vgpr_spills", ""),
            a_row.get("vgpr_spills", ""),
        ),
        "static_delta_sgpr_spills": (
            m_row.get("sgpr_spills", ""),
            a_row.get("sgpr_spills", ""),
        ),
    }
    for key, (lhs, rhs) in delta_sources.items():
        mlir.extra_fields[key] = static_delta(lhs, rhs)


def air_tuned_comparison_notes(
    mlir: KernelEvidence, air_tuned: KernelEvidence
) -> list[str]:
    if air_tuned.status == "SKIP":
        return [
            "air_tuned provider evidence is unavailable; rerun with --gpu-provider-baselines."
        ]
    notes: list[str] = []
    mlir_tops = numeric(mlir.median_tops)
    tuned_tops = numeric(air_tuned.median_tops)
    if mlir.source == "compiler-generated direct":
        notes.append(
            "MLIR-AIR row is the compiler-generated direct global-load schedule, not the HIP air_tuned provider."
        )
    if mlir_tops is not None and tuned_tops is not None and tuned_tops > 0.0:
        pct = (mlir_tops / tuned_tops) * 100.0
        gate = "passes" if pct >= 95.0 else "does not pass"
        notes.append(
            f"MLIR-AIR reaches {pct:.3f}% of air_tuned median TOPS, so it {gate} the 95% acceptance gate."
        )
    else:
        notes.append(
            "MLIR-AIR versus air_tuned TOPS percentage is unavailable from the parsed benchmark rows."
        )
    m_row = mlir.to_row()
    a_row = air_tuned.to_row()
    for key, label in (
        ("group_segment_bytes", "LDS bytes"),
        ("vgprs", "VGPRs"),
        ("sgprs", "SGPRs"),
        ("barriers", "barriers"),
        ("waitcnt", "waitcnt"),
        ("global_load_b128", "b128 global loads"),
        ("scratch_markers", "scratch markers"),
    ):
        if m_row.get(key, "") or a_row.get(key, ""):
            notes.append(
                f"{label}: MLIR-AIR={m_row.get(key) or 'n/a'}, air_tuned={a_row.get(key) or 'n/a'}."
            )
    return notes


def rocmlir_comparison_notes(
    mlir: KernelEvidence, air_tuned: KernelEvidence, rocmlir: KernelEvidence
) -> list[str]:
    if rocmlir.status == "SKIP":
        return [
            "rocMLIR reference evidence is unavailable; pass --rocmlir-artifacts-dir or run the benchmark with --gpu-rocmlir-reference."
        ]
    notes: list[str] = [
        "rocMLIR is treated as compiler-reference evidence only; air_tuned remains the performance acceptance oracle."
    ]
    m_row = mlir.to_row()
    a_row = air_tuned.to_row()
    r_row = rocmlir.to_row()
    if rocmlir.macro_tile_m or rocmlir.k_tile:
        notes.append(
            f"Schedule shape: MLIR-AIR={mlir.macro_tile_m or 'n/a'}x{mlir.macro_tile_n or 'n/a'}x{mlir.k_tile or 'n/a'}, "
            f"air_tuned={air_tuned.macro_tile_m or 'n/a'}x{air_tuned.macro_tile_n or 'n/a'}x{air_tuned.k_tile or 'n/a'}, "
            f"rocMLIR={rocmlir.macro_tile_m or 'n/a'}x{rocmlir.macro_tile_n or 'n/a'}x{rocmlir.k_tile or 'n/a'}."
        )
    for key, label in (
        ("wmma", "WMMA"),
        ("global_load_b128", "b128 global loads"),
        ("group_segment_bytes", "LDS bytes"),
        ("barriers", "barriers"),
        ("waitcnt", "waitcnt"),
        ("vgprs", "VGPRs"),
        ("scratch_markers", "scratch markers"),
    ):
        if m_row.get(key, "") or a_row.get(key, "") or r_row.get(key, ""):
            notes.append(
                f"{label}: MLIR-AIR={m_row.get(key) or 'n/a'}, "
                f"air_tuned={a_row.get(key) or 'n/a'}, rocMLIR={r_row.get(key) or 'n/a'}."
            )
    mlir_tops = numeric(mlir.median_tops)
    rocmlir_tops = numeric(rocmlir.median_tops)
    if mlir_tops is not None and rocmlir_tops is not None and rocmlir_tops > 0.0:
        notes.append(
            f"MLIR-AIR reaches {(mlir_tops / rocmlir_tops) * 100.0:.3f}% of the rocMLIR reference median TOPS row."
        )
    return notes


def rocmlir_signature_gap_notes(
    mlir: KernelEvidence, air_tuned: KernelEvidence, rocmlir: KernelEvidence
) -> list[str]:
    if rocmlir.status == "SKIP":
        return ["rocMLIR reference evidence is unavailable."]
    notes = [
        "rocMLIR signature targets are advisory; measured air_tuned parity remains the acceptance gate."
    ]
    m_row = mlir.to_row()
    a_row = air_tuned.to_row()
    r_row = rocmlir.to_row()
    for key, label in (
        ("group_segment_bytes", "LDS bytes"),
        ("vgprs", "VGPRs"),
        ("sgprs", "SGPRs"),
        ("wmma", "static WMMA"),
        ("barriers", "static barriers"),
        ("waitcnt", "static waitcnt"),
        ("global_load_b128_per_wmma", "b128 global loads/WMMA"),
        ("lds_ops_per_wmma", "LDS ops/WMMA"),
        ("scratch_markers", "scratch markers"),
    ):
        values = [f"MLIR-AIR={m_row.get(key) or 'n/a'}"]
        if air_tuned.status != "SKIP":
            values.append(f"air_tuned={a_row.get(key) or 'n/a'}")
        values.append(f"rocMLIR={r_row.get(key) or 'n/a'}")
        notes.append(f"{label}: " + ", ".join(values) + ".")
    return notes


def missing_pieces(mlir: KernelEvidence, tensile: KernelEvidence) -> list[str]:
    items: list[str] = []
    m_row = mlir.to_row()
    t_row = tensile.to_row()
    if mlir.macro_tile_m and tensile.macro_tile_m:
        items.append(
            f"Tensile uses macro tile {tensile.macro_tile_m}x{tensile.macro_tile_n}x{tensile.k_tile}; "
            f"MLIR-AIR uses {mlir.macro_tile_m}x{mlir.macro_tile_n}x{mlir.k_tile}. "
            "The current MLIR-AIR path leaves output-column reuse on the table."
        )
    if mlir.group_segment_bytes and tensile.group_segment_bytes:
        items.append(
            f"Tensile allocates {tensile.group_segment_bytes} B LDS per workgroup versus "
            f"{mlir.group_segment_bytes} B in MLIR-AIR, consistent with a larger staged tile and deeper copy/MMA schedule."
        )
    if mlir.vgprs and tensile.vgprs:
        items.append(
            f"Tensile spends {tensile.vgprs} VGPRs versus {mlir.vgprs} in MLIR-AIR, trading occupancy for a larger accumulator/register tile."
        )
    if (
        numeric(t_row["wmma_per_barrier"]) is not None
        and numeric(m_row["wmma_per_barrier"]) is not None
    ):
        items.append(
            f"Tensile has {t_row['wmma_per_barrier']} static WMMA/barrier versus "
            f"{m_row['wmma_per_barrier']} for MLIR-AIR, indicating better barrier amortization."
        )
    t_wait = numeric(t_row["wmma_per_waitcnt"])
    m_wait = numeric(m_row["wmma_per_waitcnt"])
    if t_wait is not None and m_wait is not None:
        if t_wait > m_wait:
            items.append(
                f"Tensile has {t_row['wmma_per_waitcnt']} static WMMA/waitcnt versus "
                f"{m_row['wmma_per_waitcnt']} for MLIR-AIR, indicating better wait amortization."
            )
        else:
            items.append(
                f"Tensile has {t_row['wmma_per_waitcnt']} static WMMA/waitcnt versus "
                f"{m_row['wmma_per_waitcnt']} for MLIR-AIR; static wait counts are not the winning signal here, so dynamic counters should decide whether waits are hidden."
            )
    t_load = numeric(t_row["global_load_b128_per_wmma"])
    m_load = numeric(m_row["global_load_b128_per_wmma"])
    if t_load is not None and m_load is not None and t_load < m_load:
        items.append(
            f"Tensile issues {t_row['global_load_b128_per_wmma']} static b128 global loads per WMMA versus "
            f"{m_row['global_load_b128_per_wmma']} for MLIR-AIR, consistent with better global-load amortization."
        )
    if tensile.counters.get("ds_swizzle", 0) > mlir.counters.get("ds_swizzle", 0):
        items.append(
            "Tensile emits LDS swizzle instructions where the current MLIR-AIR row does not, so LDS bank layout is a concrete gap to inspect."
        )
    if (
        numeric(tensile.median_tops) is not None
        and numeric(mlir.median_tops) is not None
    ):
        speedup = numeric(tensile.median_tops) / numeric(mlir.median_tops)
        items.append(
            f"Measured median speedup is {speedup:.3f}x at the fixed benchmark contract."
        )
    return items


def write_reports(
    out_dir: Path,
    rows: Sequence[dict[str, str]],
    mlir: KernelEvidence,
    air_tuned: KernelEvidence,
    rocmlir: KernelEvidence,
    tensile: KernelEvidence,
) -> tuple[Path, Path]:
    csv_path = out_dir / "gpu_provider_isa_compare.csv"
    md_path = out_dir / "gpu_provider_isa_compare.md"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(FIELDNAMES))
        writer.writeheader()
        writer.writerows(rows)
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# GPU INT8 GEMM ISA Comparison\n\n")
        f.write(
            f"Shape: `M=N=K={M}`, ideal bytes `{IDEAL_BYTES}`, operational intensity `{OPS / IDEAL_BYTES:.6f}` ops/byte.\n\n"
        )
        f.write("## Summary\n\n")
        f.write(
            "| Kernel | Source | Status | Validation | Median TOPS | Avg ns | Tile | WG | VGPR | SGPR | LDS B | WMMA | Barrier | Wait | ISA |\n"
        )
        f.write(
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
        )
        for row in rows:
            tile = "x".join(
                part
                for part in (row["macro_tile_m"], row["macro_tile_n"], row["k_tile"])
                if part
            )
            f.write(
                f"| `{row['kernel']}` | {row['source']} | {row['status']} | {row['validation'] or 'n/a'} | "
                f"{row['median_tops'] or 'n/a'} | {row['profile_avg_ns'] or 'n/a'} | {tile or 'n/a'} | "
                f"{row['workgroup_size'] or 'n/a'} | {row['vgprs'] or 'n/a'} | {row['sgprs'] or 'n/a'} | "
                f"{row['group_segment_bytes'] or 'n/a'} | {row['wmma'] or '0'} | {row['barriers'] or '0'} | "
                f"{row['waitcnt'] or '0'} | `{row['isa_path'] or 'n/a'}` |\n"
            )
        mlir_row = mlir.to_row()
        f.write("\n## Optimization Gates\n\n")
        f.write("| Field | Value |\n| --- | --- |\n")
        f.write(
            "| MLIR-AIR / air_tuned median TOPS | `{}` |\n".format(
                (mlir_row.get("mlir_air_pct_of_air_tuned", "") + "%")
                if mlir_row.get("mlir_air_pct_of_air_tuned", "")
                else "n/a"
            )
        )
        f.write(
            "| Passes 95% air_tuned gate | `{}` |\n".format(
                mlir_row.get("passes_air_tuned_95pct", "") or "n/a"
            )
        )
        f.write(
            "| Candidate improvement vs accepted best | `{}` |\n".format(
                (mlir_row.get("candidate_improvement_pct", "") + "%")
                if mlir_row.get("candidate_improvement_pct", "")
                else "n/a"
            )
        )
        f.write(
            "| Keep candidate | `{}` |\n".format(
                mlir_row.get("keep_candidate", "") or "n/a"
            )
        )
        f.write(
            "| Static delta WMMA | `{}` |\n".format(
                mlir_row.get("static_delta_wmma", "") or "n/a"
            )
        )
        f.write(
            "| Static delta b128 global loads | `{}` |\n".format(
                mlir_row.get("static_delta_global_load_b128", "") or "n/a"
            )
        )
        f.write(
            "| Static delta LDS ops | `{}` |\n".format(
                mlir_row.get("static_delta_lds_ops", "") or "n/a"
            )
        )
        f.write(
            "| Static delta waitcnt | `{}` |\n".format(
                mlir_row.get("static_delta_waitcnt", "") or "n/a"
            )
        )
        f.write(
            "| Static delta VGPR / SGPR | `{}` / `{}` |\n".format(
                mlir_row.get("static_delta_vgprs", "") or "n/a",
                mlir_row.get("static_delta_sgprs", "") or "n/a",
            )
        )
        f.write("\n## Static Ratios\n\n")
        f.write(
            "| Kernel | WMMA/barrier | WMMA/waitcnt | global_load_b128/WMMA | LDS ops/WMMA | ds_swizzle | Scratch |\n"
        )
        f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        for row in rows:
            f.write(
                f"| `{row['kernel']}` | {row['wmma_per_barrier'] or 'n/a'} | {row['wmma_per_waitcnt'] or 'n/a'} | "
                f"{row['global_load_b128_per_wmma'] or 'n/a'} | {row['lds_ops_per_wmma'] or 'n/a'} | "
                f"{row['ds_swizzle'] or '0'} | {row['scratch_markers'] or '0'} |\n"
            )
        f.write("\n## MLIR-AIR vs air_tuned\n\n")
        for item in air_tuned_comparison_notes(mlir, air_tuned):
            f.write(f"- {item}\n")
        f.write("\n## MLIR-AIR vs air_tuned vs rocMLIR\n\n")
        for item in rocmlir_comparison_notes(mlir, air_tuned, rocmlir):
            f.write(f"- {item}\n")
        f.write("\n## rocMLIR Signature Gap\n\n")
        for item in rocmlir_signature_gap_notes(mlir, air_tuned, rocmlir):
            f.write(f"- {item}\n")
        f.write("\n## Missing Pieces vs Tensile\n\n")
        for item in missing_pieces(mlir, tensile):
            f.write(f"- {item}\n")
        f.write("\n## Notes\n\n")
        for row in rows:
            f.write(f"- `{row['kernel']}`: {row['notes'] or 'n/a'}\n")
        f.write(f"\nCSV: `{csv_path}`\n")
    return csv_path, md_path


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="benchmark output directory created by benchmarks/gemm_int8/run.py",
    )
    parser.add_argument(
        "--gpu-arch",
        default="gfx1150",
        help="AMDGPU chip used for disassembly (default: gfx1150)",
    )
    parser.add_argument(
        "--rocm-path",
        type=Path,
        default=rocm_root(),
        help="ROCm install path (default: ROCM_PATH or /opt/rocm)",
    )
    parser.add_argument(
        "--tensile-co",
        type=Path,
        default=None,
        help="override rocBLAS/Tensile packed .co path",
    )
    parser.add_argument(
        "--tensile-symbol",
        default="",
        help="override selected rocBLAS/Tensile kernel symbol",
    )
    parser.add_argument(
        "--rocmlir-artifacts-dir",
        type=Path,
        default=None,
        help="directory containing rocMLIR reference ISA, HSACO, readobj, MLIR, or profile artifacts",
    )
    parser.add_argument(
        "--rocmlir-isa",
        type=Path,
        default=None,
        help="override rocMLIR reference ISA path",
    )
    parser.add_argument(
        "--rocmlir-hsaco",
        type=Path,
        default=None,
        help="override rocMLIR reference HSACO/code object path",
    )
    parser.add_argument(
        "--rocmlir-readobj",
        type=Path,
        default=None,
        help="override rocMLIR reference readobj/metadata path",
    )
    parser.add_argument(
        "--rocmlir-mlir",
        type=Path,
        default=None,
        help="override rocMLIR generated MLIR path",
    )
    parser.add_argument(
        "--rocmlir-profile",
        type=Path,
        default=None,
        help="override rocMLIR rocprof kernel stats CSV path",
    )
    parser.add_argument(
        "--rocmlir-symbol",
        default="",
        help="override rocMLIR kernel symbol/metadata name",
    )
    parser.add_argument(
        "--mlir-isa", type=Path, default=None, help="override MLIR-AIR ISA path"
    )
    parser.add_argument(
        "--mlir-hsaco", type=Path, default=None, help="override MLIR-AIR hsaco path"
    )
    parser.add_argument(
        "--mlir-final-mlir",
        type=Path,
        default=None,
        help="override MLIR-AIR final MLIR path for --profile-mlir",
    )
    parser.add_argument(
        "--profile-mlir",
        action="store_true",
        help="capture rocprofv3 kernel stats for the MLIR-AIR final MLIR before comparing",
    )
    parser.add_argument(
        "--mlir-runner",
        type=Path,
        default=None,
        help="override mlir-runner path for --profile-mlir",
    )
    parser.add_argument(
        "--airgpu-lib",
        type=Path,
        default=None,
        help="override libairgpu.so path for --profile-mlir",
    )
    parser.add_argument(
        "--rocprof",
        type=Path,
        default=None,
        help="override rocprofv3 path for --profile-mlir",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    args.out_dir = args.out_dir.resolve()
    repo = Path(__file__).resolve().parents[2]
    artifacts_dir = args.out_dir / "build" / "gpu_provider_isa_compare" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    llvm_objdump = find_tool("llvm-objdump", args.rocm_path, repo)
    llvm_readobj = find_tool("llvm-readobj", args.rocm_path, repo)
    if not llvm_objdump or not llvm_readobj:
        print("ERROR: llvm-objdump and llvm-readobj are required", file=sys.stderr)
        return 2
    if args.profile_mlir:
        profile_path, note = profile_mlir_air(args, repo, artifacts_dir)
        write_text(artifacts_dir / "mlir_profile_note.txt", note + "\n")
    rows = benchmark_csv_rows(args.out_dir)
    mlir = build_mlir_evidence(args, rows, llvm_readobj, artifacts_dir)
    air_tuned = build_air_tuned_evidence(args, rows, artifacts_dir)
    rocmlir = build_rocmlir_reference_evidence(
        args, rows, llvm_objdump, llvm_readobj, artifacts_dir
    )
    tensile = build_tensile_evidence(
        args, rows, llvm_objdump, llvm_readobj, artifacts_dir
    )
    annotate_mlir_air_tuned_gate(mlir, air_tuned)
    report_rows = [mlir.to_row()]
    if air_tuned.status != "SKIP":
        report_rows.append(air_tuned.to_row())
    report_rows.append(rocmlir.to_row())
    report_rows.append(tensile.to_row())
    csv_path, md_path = write_reports(
        args.out_dir, report_rows, mlir, air_tuned, rocmlir, tensile
    )
    print(f"ISA comparison report: {md_path}")
    print(f"ISA comparison CSV: {csv_path}")
    return (
        1
        if any(row["status"] == "FAIL" for row in (mlir.to_row(), tensile.to_row()))
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
