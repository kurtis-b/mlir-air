#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# fmt: off

"""Build, inspect, and optionally run the shared int8 GEMM benchmark."""

from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

M = N = K = 1024
TARGET_TOPS = {"cpu": 4.0, "gpu": 15.0, "npu": 36.0}
DEFAULT_GPU_INT8_GEMM_VARIANT = "lds_128x64_wmma4"
GPU_INT8_GEMM_VARIANTS = (
    DEFAULT_GPU_INT8_GEMM_VARIANT,
    "lds_128x64_bpack",
    "lds_128x64_bpack_swizzle",
    "lds_128x64_bpack_pipe2",
    "lds_128x64_bpack_pipe2_grouped",
    "lds_128x64_bpack_swizzle_grouped",
    "lds_128x64_bpack_frag",
)
GPU_INT8_GEMM_SWEEP_VARIANTS = GPU_INT8_GEMM_VARIANTS
DEFAULT_GPU_INT8_GEMM_GROUP_SIZE = 4
GPU_INT8_GEMM_GROUP_SIZES = (2, 4, 8)
DEFAULT_GPU_INT8_GEMM_SWEEP_GROUP_SIZES = GPU_INT8_GEMM_GROUP_SIZES
DEFAULT_GPU_INT8_GEMM_SWEEP_REPETITIONS = 3
GPU_INT8_GEMM_GROUPED_SWIZZLE_VARIANT = "lds_128x64_bpack_swizzle_grouped"


def positive_int(value: str) -> int:
    parsed = int(value, 10)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value, 10)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def gpu_group_size(value: str) -> int:
    parsed = positive_int(value)
    if parsed not in GPU_INT8_GEMM_GROUP_SIZES:
        choices = ", ".join(str(size) for size in GPU_INT8_GEMM_GROUP_SIZES)
        raise argparse.ArgumentTypeError(f"value must be one of: {choices}")
    return parsed


def gpu_group_sizes(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(gpu_group_size(item.strip()) for item in value.split(",") if item.strip())
    except argparse.ArgumentTypeError:
        raise
    if not parsed:
        raise argparse.ArgumentTypeError("at least one group size is required")
    return tuple(dict.fromkeys(parsed))


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def count_regex(text_or_path: str | Path, pattern: str) -> int:
    text = read_text(text_or_path) if isinstance(text_or_path, Path) else text_or_path
    return len(re.findall(pattern, text, flags=re.MULTILINE))


def last_kv_value(path: Path, key: str) -> str:
    matches = re.findall(rf"\b{re.escape(key)}=([^\s]+)", read_text(path))
    return matches[-1] if matches else ""


def timing_field(path: Path, domain: str, field_name: str) -> str:
    for line in read_text(path).splitlines():
        if f"timing_domain={domain}" in line:
            return dict(re.findall(r"([A-Za-z0-9_.-]+)=([^\s]+)", line)).get(field_name, "")
    return ""


def mlir_string_attr(path: Path, attr_name: str) -> str:
    matches = re.findall(rf'{re.escape(attr_name)} = "([^"]+)"', read_text(path))
    return matches[-1] if matches else ""


def mlir_int_attr(path: Path, attr_name: str) -> str:
    matches = re.findall(rf'{re.escape(attr_name)} = ([0-9]+)(?: : [a-z0-9]+)?', read_text(path))
    return matches[-1] if matches else ""


def summary_metric(path: Path, metric_name: str) -> str:
    matches = re.findall(rf'{re.escape(metric_name)}:\s*([^\s]+)', read_text(path))
    return matches[-1] if matches else ""


def gpu_variant_uses_packed_b(variant: str) -> bool:
    return variant != DEFAULT_GPU_INT8_GEMM_VARIANT


def gpu_variant_uses_fragment_b(variant: str) -> bool:
    return variant == "lds_128x64_bpack_frag"


def gpu_b_pack_function(variant: str) -> str:
    return "mgpuPackBFragI8I32" if gpu_variant_uses_fragment_b(variant) else "mgpuPackBI8I32"


def to_tops(gops: str) -> str:
    try:
        return f"{float(gops) / 1000.0:.6f}" if gops else "n/a"
    except ValueError:
        return "n/a"


def parse_gops_tops(gops: str) -> float | None:
    try:
        return float(gops) / 1000.0 if gops else None
    except ValueError:
        return None


def parse_tops(value: str) -> float | None:
    try:
        return float(value) if value else None
    except ValueError:
        return None


def parse_float(value: str) -> float | None:
    try:
        return float(value) if value else None
    except ValueError:
        return None


def fmt_float(value: float | None) -> str:
    return f"{value:.6f}" if value is not None else ""


def set_perf_tops(result: "BackendResult", tops: float | None) -> None:
    result.perf_tops = tops
    if tops is None or result.target_tops is None:
        result.target_pct = "n/a"
        return
    result.target_pct = f"{(tops / result.target_tops) * 100.0:.1f}%"


def us_to_ms(us: str) -> str:
    try:
        return f"{float(us) / 1000.0:.6f}" if us else "n/a"
    except ValueError:
        return "n/a"


def sanitize_prefix(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "artifact"


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"expected exactly one occurrence of {old!r}")
    return text.replace(old, new, 1)


@dataclass
class RunContext:
    repo: Path
    out_dir: Path
    build_root: Path
    logs_dir: Path
    warmups: int
    iterations: int
    gpu_arch: str
    gpu_int8_gemm_variant: str
    gpu_int8_gemm_group_size: int
    run_enabled: bool
    cpu_threads: int
    npu_runtime_loop_tiling: str

    @property
    def disassemble(self) -> Path:
        return self.repo / "utils" / "isa_inspect" / "disassemble.sh"


@dataclass
class BackendResult:
    backend: str
    build_dir: Path
    artifacts_dir: Path
    status: str = "SKIP"
    evidence: str = "not selected"
    runtime: str = "not run"
    perf_domain: str = "not run"
    perf_count: str = "n/a"
    perf_latency: str = "n/a"
    perf_throughput: str = "n/a"
    perf_tops: float | None = None
    target_tops: float | None = None
    target_pct: str = "n/a"
    perf_notes: str = "not run"
    logs: dict[str, Path] = field(default_factory=dict)


def backend_result(ctx: RunContext, name: str, create: bool = False) -> BackendResult:
    build_dir = ctx.build_root / name
    result = BackendResult(name, build_dir, build_dir / "artifacts")
    result.target_tops = TARGET_TOPS.get(name)
    if create:
        result.build_dir.mkdir(parents=True, exist_ok=True)
        result.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return result


def log_path(ctx: RunContext, result: BackendResult, stem: str) -> Path:
    path = ctx.logs_dir / f"{result.backend}_{stem}.log"
    result.logs[stem] = path
    return path


def run_capture(log: Path, argv: Sequence[object], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> tuple[bool, Path]:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as output:
        cd = f"cd {cwd} && " if cwd else ""
        output.write(f"+ {cd}{' '.join(shlex.quote(str(arg)) for arg in argv)}\n")
        output.flush()
        try:
            completed = subprocess.run([str(arg) for arg in argv], cwd=str(cwd) if cwd else None, env=env, stdout=output, stderr=subprocess.STDOUT, text=True, check=False)
            return completed.returncode == 0, log
        except OSError as exc:
            output.write(f"ERROR: {exc}\n")
            return False, log


def run_logged(ctx: RunContext, result: BackendResult, stem: str, argv: Sequence[object], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> tuple[bool, Path]:
    return run_capture(log_path(ctx, result, stem), argv, cwd=cwd, env=env)


def note_run_failure(result: BackendResult, log: Path, label: str = "run") -> None:
    result.runtime = f"{label} failed; see {log}"
    result.perf_notes = f"{label} failed; see {log}"


def parse_host_perf(ctx: RunContext, result: BackendResult, log: Path, domain: str) -> None:
    avg_us, min_us, max_us, gops = (last_kv_value(log, key) for key in ("avg_us", "min_us", "max_us", "gops"))
    result.perf_domain = domain
    result.perf_count = last_kv_value(log, "iterations") or str(ctx.iterations)
    result.perf_latency = f"mean {avg_us or 'n/a'} us ({us_to_ms(avg_us)} ms), min {min_us or 'n/a'} us, max {max_us or 'n/a'} us"
    result.perf_throughput = f"{gops or 'n/a'} GOPS ({to_tops(gops)} TOPS)"
    set_perf_tops(result, parse_gops_tops(gops))


def cpu_backend(ctx: RunContext) -> BackendResult:
    result = backend_result(ctx, "cpu", True)
    source_dir = ctx.repo / "test" / "cpu" / "int8_gemm"
    binary = result.build_dir / "int8_gemm_cpu"
    disasm = result.artifacts_dir / "cpu_int8_gemm.disasm.s"
    ok, log = run_logged(ctx, result, "build", ["make", "-C", source_dir, f"BUILD_DIR={result.build_dir}"])
    if not ok:
        result.status, result.evidence = "WARN", f"CPU benchmark build failed; see {log}"
        return result
    ok, log = run_logged(ctx, result, "disassemble", [ctx.disassemble, "cpu", "--output-dir", result.artifacts_dir, "--prefix", "cpu_int8_gemm", "--symbol", "cpu_i8_gemm_vnni", "--expect", "vpdpbusd", binary])
    if not ok:
        result.status, result.evidence = "FAIL", f"CPU disassembly did not show required VNNI marker; see {log}"
        return result
    if ctx.run_enabled:
        ok, log = run_logged(ctx, result, "run", [binary, "--warmups", ctx.warmups, "--iterations", ctx.iterations, "--threads", ctx.cpu_threads])
        result.runtime = f"ran; see {log}" if ok else result.runtime
        if not ok:
            note_run_failure(result, log)
    vnni = count_regex(disasm, r"\bvpdpbusd\b")
    zmm = count_regex(disasm, r"\bzmm[0-9]+")
    result.status = "PASS" if vnni else "FAIL"
    result.evidence = f"vpdpbusd={vnni}, zmm_refs={zmm}"
    if ctx.run_enabled and "failed" in result.runtime and result.status == "PASS":
        result.status = "WARN"
    if ctx.run_enabled and (run_log := result.logs.get("run")) and run_log.exists():
        parse_host_perf(ctx, result, run_log, last_kv_value(run_log, "timing_domain") or "host_steady_clock")
        result.perf_notes = f"threads={last_kv_value(run_log, 'threads') or ctx.cpu_threads}; warmups={ctx.warmups}; validation={last_kv_value(run_log, 'validation') or 'unknown'}"
    return result


def gpu_tools(repo: Path) -> tuple[str | None, str | None]:
    runner = os.environ.get("MLIR_RUNNER") or shutil.which("mlir-runner")
    if not runner and (candidate := repo / "llvm" / "install-amdgpu" / "bin" / "mlir-runner").exists():
        runner = str(candidate)
    airgpu = os.environ.get("AIRGPU_LIB")
    if not airgpu and os.environ.get("MLIR_AIR_INSTALL_DIR"):
        airgpu = str(Path(os.environ["MLIR_AIR_INSTALL_DIR"]) / "lib" / "libairgpu.so")
    if not airgpu and (candidate := repo / "install-gpu" / "lib" / "libairgpu.so").exists():
        airgpu = str(candidate)
    return runner, airgpu


def gpu_compile_env(repo: Path) -> dict[str, str]:
    env = os.environ.copy()
    if not env.get("AIR_OPT") and (candidate := repo / "build-gpu" / "bin" / "air-opt").exists():
        env["AIR_OPT"] = str(candidate)
    if not env.get("MLIR_OPT") and (candidate := repo / "llvm" / "install-amdgpu" / "bin" / "mlir-opt").exists():
        env["MLIR_OPT"] = str(candidate)
    return env


def run_gpu_final_mlir(ctx: RunContext, result: BackendResult, final_mlir: Path, stem: str) -> tuple[bool, Path]:
    runner, airgpu = gpu_tools(ctx.repo)
    log = log_path(ctx, result, stem)
    if not runner or not Path(runner).exists():
        write_text(log, "ERROR: mlir-runner not found\n")
        return False, log
    if not airgpu or not Path(airgpu).exists():
        write_text(log, "ERROR: libairgpu.so not found\n")
        return False, log
    env = os.environ.copy()
    env.setdefault("AIRGPU_USE_HIP_MALLOC", "1")
    return run_capture(log, [runner, "--entry-point-result=void", f"--shared-libs={airgpu}", final_mlir], env=env)


def gpu_run_metrics(log: Path) -> dict[str, float]:
    fields = {
        "mean_ms": parse_float(timing_field(log, "kernel_event", "mean_ms")),
        "min_ms": parse_float(timing_field(log, "kernel_event", "min_ms")),
        "max_ms": parse_float(timing_field(log, "kernel_event", "max_ms")),
        "tops": parse_tops(timing_field(log, "kernel_event", "tops")),
    }
    return {key: value for key, value in fields.items() if value is not None}


def summarize_gpu_repetition_metrics(logs: Sequence[Path]) -> dict[str, float]:
    metrics = [gpu_run_metrics(log) for log in logs]
    metrics = [entry for entry in metrics if {"mean_ms", "min_ms", "tops"} <= set(entry)]
    if not metrics:
        return {}
    mean_ms = [entry["mean_ms"] for entry in metrics]
    min_ms = [entry["min_ms"] for entry in metrics]
    tops = [entry["tops"] for entry in metrics]
    return {
        "repetitions": float(len(metrics)),
        "median_mean_ms": statistics.median(mean_ms),
        "min_mean_ms": min(mean_ms),
        "best_kernel_min_ms": min(min_ms),
        "median_tops": statistics.median(tops),
        "max_tops": max(tops),
    }


def apply_gpu_repetition_summary(ctx: RunContext, result: BackendResult, summary: dict[str, float], variant: str, group_size: int, run_logs: Sequence[Path]) -> None:
    if not summary:
        return
    repetitions = int(summary["repetitions"])
    result.perf_domain = "kernel_event"
    result.perf_count = f"{repetitions}x{ctx.iterations}"
    result.perf_latency = (
        f"median mean {summary['median_mean_ms']:.6f} ms, "
        f"min mean {summary['min_mean_ms']:.6f} ms, "
        f"best kernel min {summary['best_kernel_min_ms']:.6f} ms"
    )
    result.perf_throughput = f"median {summary['median_tops']:.6f} TOPS, max {summary['max_tops']:.6f} TOPS"
    set_perf_tops(result, summary["median_tops"])
    result.perf_notes = f"variant={variant}; group_m={group_size}; repetitions={repetitions}; warmups={ctx.warmups}"
    result.runtime = f"ran {repetitions} repetition(s); see {', '.join(str(log) for log in run_logs)}"


def render_gpu_mlir(source: Path, dest: Path, ctx: RunContext) -> None:
    text = read_text(source)
    for old, new in (
        ("%c10 = arith.constant 10 : index", f"%c10 = arith.constant {ctx.warmups} : index"),
        ("%c5 = arith.constant 5 : index", f"%c5 = arith.constant {ctx.iterations} : index"),
        ("%c10_i64 = arith.constant 10 : i64", f"%c10_i64 = arith.constant {ctx.warmups} : i64"),
        ("%c5_i64 = arith.constant 5 : i64", f"%c5_i64 = arith.constant {ctx.iterations} : i64"),
    ):
        text = replace_once(text, old, new)
    if gpu_variant_uses_packed_b(ctx.gpu_int8_gemm_variant):
        text = render_gpu_packed_b_mlir(text, ctx.gpu_int8_gemm_variant)
    write_text(dest, text)


def render_gpu_packed_b_mlir(text: str, variant: str) -> str:
    pack_function = gpu_b_pack_function(variant)
    text = replace_once(
        text,
        "  llvm.func @mgpuCheckOutputI8I32(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64) -> i32\n",
        "  llvm.func @mgpuCheckOutputI8I32(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64) -> i32\n"
        f"  llvm.func @{pack_function}(!llvm.ptr, !llvm.ptr, i64, i64)\n",
    )
    text = replace_once(
        text,
        "    %alloc_1 = memref.alloc() : memref<1024x1024xi32>\n",
        "    %alloc_1 = memref.alloc() : memref<1024x1024xi32>\n"
        "    %alloc_2 = memref.alloc() : memref<1024x1024xi8>\n",
    )
    text = replace_once(
        text,
        "    %b_ptr = llvm.inttoptr %b_ptr_i64 : i64 to !llvm.ptr\n",
        "    %b_ptr = llvm.inttoptr %b_ptr_i64 : i64 to !llvm.ptr\n"
        "    %bpack_intptr = memref.extract_aligned_pointer_as_index %alloc_2 : memref<1024x1024xi8> -> index\n"
        "    %bpack_ptr_i64 = arith.index_cast %bpack_intptr : index to i64\n"
        "    %bpack_ptr = llvm.inttoptr %bpack_ptr_i64 : i64 to !llvm.ptr\n",
    )
    text = replace_once(
        text,
        "    llvm.call @mgpuInitI8I32(%a_ptr, %b_ptr, %m64, %n64, %k64) : (!llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()\n",
        "    llvm.call @mgpuInitI8I32(%a_ptr, %b_ptr, %m64, %n64, %k64) : (!llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()\n"
        f"    llvm.call @{pack_function}(%b_ptr, %bpack_ptr, %n64, %k64) : (!llvm.ptr, !llvm.ptr, i64, i64) -> ()\n",
    )
    text = replace_once(
        text,
        "    gpu.memcpy %memref_2, %alloc_0 : memref<1024x1024xi8>, memref<1024x1024xi8>\n",
        "    gpu.memcpy %memref_2, %alloc_2 : memref<1024x1024xi8>, memref<1024x1024xi8>\n",
    )
    text = replace_once(
        text,
        "    memref.dealloc %alloc_0 : memref<1024x1024xi8>\n"
        "    memref.dealloc %alloc_1 : memref<1024x1024xi32>\n",
        "    memref.dealloc %alloc_0 : memref<1024x1024xi8>\n"
        "    memref.dealloc %alloc_1 : memref<1024x1024xi32>\n"
        "    memref.dealloc %alloc_2 : memref<1024x1024xi8>\n",
    )
    return text


def gpu_backend(ctx: RunContext) -> BackendResult:
    result = backend_result(ctx, "gpu", True)
    source = ctx.repo / "test" / "gpu" / "int8_gemm" / "air_sync.mlir"
    generated = result.build_dir / "int8_gemm.air_sync.mlir"
    isa = result.artifacts_dir / "gpu_int8_gemm.isa.s"
    summary = result.artifacts_dir / "gpu_int8_gemm.summary.txt"
    outline_mlir = result.artifacts_dir / "gpu_int8_gemm.outline.mlir"
    final_mlir = result.artifacts_dir / "gpu_int8_gemm.final.mlir"
    try:
        render_gpu_mlir(source, generated, ctx)
        write_text(log_path(ctx, result, "render"), f"generated {generated}\nwarmups={ctx.warmups}\niterations={ctx.iterations}\n")
    except Exception as exc:  # noqa: BLE001 - report rendering failures in the same log flow.
        log = log_path(ctx, result, "render")
        write_text(log, f"failed to render GPU MLIR: {exc}\n")
        result.status, result.evidence = "WARN", f"GPU MLIR render failed; see {log}"
        return result
    ok, log = run_logged(ctx, result, "disassemble", [ctx.disassemble, "gpu", "--gpu-arch", ctx.gpu_arch, "--int8-gemm-variant", ctx.gpu_int8_gemm_variant, "--int8-gemm-group-size", ctx.gpu_int8_gemm_group_size, "--output-dir", result.artifacts_dir, "--prefix", "gpu_int8_gemm", "--expect", "v_wmma_i32_16x16x16_iu8", "--forbid", r"v_wmma_.*16x16x64|v_swmmac|swmmac", generated], env=gpu_compile_env(ctx.repo))
    if not ok:
        result.status, result.evidence = "WARN", f"GPU lowering/disassembly failed or required marker was absent; see {log}"
        return result
    runtime_logs: list[Path] = []
    if ctx.run_enabled:
        ok, log = run_gpu_final_mlir(ctx, result, final_mlir, "run")
        if ok:
            runtime_logs.append(log)
        else:
            note_run_failure(result, log)
    wmma = count_regex(isa, r"\bv_wmma_i32_16x16x16_iu8\b")
    barriers = count_regex(isa, r"\bs_barrier\b")
    scratch = count_regex(isa, r"uses_flat_scratch\s+1")
    spills = count_regex(summary, r"spill_count = [1-9]|_spill_count: [1-9]")
    variant = mlir_string_attr(outline_mlir, "air.gpu.int8_gemm_variant") or "unknown"
    group_m = mlir_int_attr(outline_mlir, "air.gpu.int8_gemm_group_m") or str(ctx.gpu_int8_gemm_group_size)
    vgprs = summary_metric(summary, ".vgpr_count") or "n/a"
    sgprs = summary_metric(summary, ".sgpr_count") or "n/a"
    global_load_b128 = count_regex(isa, r"\bglobal_load_b128\b")
    global_load_lds = count_regex(isa, r"\bglobal_load(?:_async)?(?:_to)?_lds")
    ds_store_b128 = count_regex(isa, r"\bds_store_b128\b")
    ds_store_b8 = count_regex(isa, r"\bds_store_b8(?:_d16_hi)?\b")
    ds_load_b128 = count_regex(isa, r"\bds_(?:read|load)_b128\b")
    global_store_b32 = count_regex(isa, r"\bglobal_store_b32\b")
    waitcnt = count_regex(isa, r"\bs_waitcnt\b")
    result.status = "PASS" if wmma and scratch == 0 and spills == 0 else "FAIL"
    if ctx.run_enabled and "failed" in result.runtime and result.status == "PASS":
        result.status = "WARN"
    result.evidence = (
        f"variant={variant}, group_m={group_m}, wmma={wmma}, barriers={barriers}, vgprs={vgprs}, "
        f"sgprs={sgprs}, scratch_markers={scratch}, spills={spills}, "
        f"global_load_b128={global_load_b128}, global_load_lds={global_load_lds}, "
        f"ds_store_b128={ds_store_b128}, ds_store_b8={ds_store_b8}, "
        f"ds_load_b128={ds_load_b128}, global_store_b32={global_store_b32}, waitcnt={waitcnt}"
    )
    if ctx.run_enabled and runtime_logs:
        apply_gpu_repetition_summary(
            ctx,
            result,
            summarize_gpu_repetition_metrics(runtime_logs),
            variant,
            int(group_m),
            runtime_logs,
        )
    return result



def evidence_map(result: BackendResult) -> dict[str, str]:
    return dict(re.findall(r"([A-Za-z0-9_.-]+)=([^,\s]+)", result.evidence))


def run_gpu_variant_sweep(
    ctx: RunContext,
    variants: Sequence[str],
    group_sizes: Sequence[int],
    repetitions: int,
) -> BackendResult:
    rows: list[dict[str, str]] = []
    results: list[BackendResult] = []
    candidates = [
        (variant, group_size)
        for variant in variants
        for group_size in (
            group_sizes if variant == GPU_INT8_GEMM_GROUPED_SWIZZLE_VARIANT else (ctx.gpu_int8_gemm_group_size,)
        )
    ]
    for variant, group_size in candidates:
        prefix = sanitize_prefix(f"{variant}_g{group_size}" if variant == GPU_INT8_GEMM_GROUPED_SWIZZLE_VARIANT else variant)
        variant_ctx = RunContext(
            ctx.repo,
            ctx.out_dir / "gpu_sweep" / prefix,
            ctx.build_root / "gpu_sweep" / prefix,
            ctx.logs_dir / "gpu_sweep" / prefix,
            ctx.warmups,
            ctx.iterations,
            ctx.gpu_arch,
            variant,
            group_size,
            False,
            ctx.cpu_threads,
            ctx.npu_runtime_loop_tiling,
        )
        for path in (variant_ctx.out_dir, variant_ctx.build_root, variant_ctx.logs_dir):
            path.mkdir(parents=True, exist_ok=True)
        result = gpu_backend(variant_ctx)
        runtime_failures = 0
        run_logs: list[Path] = []
        if ctx.run_enabled and result.status == "PASS":
            final_mlir = result.artifacts_dir / "gpu_int8_gemm.final.mlir"
            for rep in range(1, repetitions + 1):
                ok, log = run_gpu_final_mlir(variant_ctx, result, final_mlir, f"run_rep{rep}")
                if ok:
                    run_logs.append(log)
                else:
                    runtime_failures += 1
            if run_logs:
                apply_gpu_repetition_summary(
                    variant_ctx,
                    result,
                    summarize_gpu_repetition_metrics(run_logs),
                    variant,
                    group_size,
                    run_logs,
                )
            if runtime_failures:
                result.status = "WARN"
                result.runtime = f"{result.runtime}; {runtime_failures} repetition failure(s)"
                result.perf_notes = f"{result.perf_notes}; runtime_failures={runtime_failures}"
        results.append(result)
        evidence = evidence_map(result)
        metrics = summarize_gpu_repetition_metrics(run_logs) if run_logs else {}
        row = {
            "variant": variant,
            "group_m": str(group_size),
            "status": result.status,
            "repetitions": str(int(metrics.get("repetitions", 0))) if metrics else "0",
            "median_mean_ms": fmt_float(metrics.get("median_mean_ms")),
            "min_mean_ms": fmt_float(metrics.get("min_mean_ms")),
            "best_kernel_min_ms": fmt_float(metrics.get("best_kernel_min_ms")),
            "median_tops": fmt_float(metrics.get("median_tops")),
            "max_tops": fmt_float(metrics.get("max_tops")),
            "timing_domain": result.perf_domain,
            "count": result.perf_count,
            "latency": result.perf_latency,
            "throughput": result.perf_throughput,
            "runtime": result.runtime,
            "artifacts": str(result.artifacts_dir),
            "run_logs": ";".join(str(log) for log in run_logs),
            "disassemble_log": str(result.logs.get("disassemble", "")),
        }
        for key in ("wmma", "barriers", "vgprs", "sgprs", "scratch_markers", "spills", "global_load_b128", "global_load_lds", "ds_store_b128", "ds_store_b8", "ds_load_b128", "global_store_b32", "waitcnt"):
            row[key] = evidence.get(key, "")
        rows.append(row)

    csv_path = ctx.out_dir / "gpu_variant_sweep.csv"
    md_path = ctx.out_dir / "gpu_variant_sweep.md"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["variant", "status"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ranked = [result for result in results if result.status == "PASS" and result.perf_tops is not None]
    if ranked:
        best = max(ranked, key=lambda result: result.perf_tops or 0.0)
        best_evidence = evidence_map(best)
        best_variant = best_evidence.get("variant", ctx.gpu_int8_gemm_variant)
        best_group = best_evidence.get("group_m", str(ctx.gpu_int8_gemm_group_size))
        best_label = f"{best_variant} group_m={best_group}"
    else:
        passing = [result for result in results if result.status == "PASS"]
        best = passing[0] if passing else (results[0] if results else backend_result(ctx, "gpu"))
        best_label = "n/a (runtime disabled)" if not ctx.run_enabled else evidence_map(best).get("variant", ctx.gpu_int8_gemm_variant)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# GPU INT8 GEMM Variant Sweep\n\n")
        f.write(f"Best variant: `{best_label}`\n\n")
        f.write("| Variant | Group M | Status | Reps | Median TOPS | Max TOPS | Median Mean ms | Min Mean ms | VGPRs | Spills | ds_store_b8 | waitcnt | Artifacts |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for row in rows:
            f.write(f"| `{row['variant']}` | {row['group_m']} | {row['status']} | {row['repetitions']} | {row['median_tops'] or 'n/a'} | {row['max_tops'] or 'n/a'} | {row['median_mean_ms'] or 'n/a'} | {row['min_mean_ms'] or 'n/a'} | {row['vgprs'] or 'n/a'} | {row['spills'] or 'n/a'} | {row['ds_store_b8'] or 'n/a'} | {row['waitcnt'] or 'n/a'} | `{row['artifacts']}` |\n")
        f.write(f"\nCSV: `{csv_path}`\n")

    best.runtime = f"{best.runtime}; sweep csv {csv_path}; sweep report {md_path}"
    best.perf_notes = f"best_variant={best_label}; sweep_candidates={len(rows)}; {best.perf_notes}"
    return best

def find_npu_elves(build_dir: Path) -> list[Path]:
    return sorted(build_dir.rglob("bare_matmul*_core_*.elf")) or sorted(build_dir.rglob("*.elf"))


def npu_env(ctx: RunContext) -> dict[str, str]:
    env = os.environ.copy()
    if not env.get("PYTHON") and (candidate := ctx.repo / "sandbox" / "bin" / "python3").exists():
        env["PYTHON"] = str(candidate)
    for candidate in sorted((ctx.repo / "sandbox" / "lib").glob("python*/site-packages/mlir_aie")):
        if (candidate / "runtime_lib" / "x86_64" / "test_lib" / "include" / "cxxopts.hpp").exists():
            env.setdefault("AIEOPT_DIR", str(candidate))
            break
    bin_paths = []
    for candidate in (ctx.repo / "install-xrt" / "bin", ctx.repo / "install" / "bin", ctx.repo / "build-xrt" / "bin", ctx.repo / "build" / "bin", ctx.repo / "sandbox" / "bin"):
        if (candidate / "aircc").exists() or (candidate / "aiecc").exists() or (candidate / "aiecc.py").exists():
            bin_paths.append(str(candidate))
    if bin_paths:
        env["PATH"] = os.pathsep.join([*bin_paths, env.get("PATH", "")]).rstrip(os.pathsep)
    python_paths = []
    for candidate in (
        ctx.repo / "install-xrt" / "python",
        ctx.repo / "install" / "python",
        ctx.repo / "build-xrt" / "python",
        ctx.repo / "build" / "python",
        ctx.repo / "python",
    ):
        if (candidate / "air" / "backend" / "xrt.py").exists():
            python_paths.append(str(candidate))
    if python_paths:
        env["PYTHONPATH"] = os.pathsep.join([*python_paths, env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    for path in [env.get("PEANO_INSTALL_DIR", ""), *(str(path) for path in sorted((ctx.repo / "sandbox/lib").glob("python*/site-packages/llvm-aie")))]:
        if path and (Path(path) / "bin" / "llc").exists():
            env["PEANO_INSTALL_DIR"] = path
            break
    return env


def npu_backend(ctx: RunContext) -> BackendResult:
    result = backend_result(ctx, "npu", True)
    env = npu_env(ctx)
    source_dir = ctx.repo / "test" / "xrt" / "46_triton_matmul_ver4_strix_8x4_i8_i8_i32"
    build_stamp = result.build_dir / "gemm_int8_build_key.txt"
    build_key = f"runtime_loop_tiling={ctx.npu_runtime_loop_tiling}\n"
    reused = (result.build_dir / "air.xclbin").exists() and (result.build_dir / "air.insts.bin").exists() and find_npu_elves(result.build_dir) and read_text(build_stamp) == build_key
    if reused:
        compile_note = f"reused build_dir={result.build_dir}"
        write_text(log_path(ctx, result, "build"), f"Reusing existing NPU artifacts in {result.build_dir}\n")
    else:
        compile_note = "fresh compile"
        ok, log = run_logged(ctx, result, "build", ["make", "-C", source_dir, f"BUILD_DIR={result.build_dir}", "AIE_TARGET=aie2p", f"M={M}", f"K={K}", f"N={N}", f"RUNTIME_LOOP_TILING={ctx.npu_runtime_loop_tiling}", "compile-xclbin"], env=env)
        if not ok:
            result.status, result.evidence = "WARN", f"NPU compile-xclbin failed; see {log}"
            return result
        write_text(build_stamp, build_key)
    elves = find_npu_elves(result.build_dir)
    if not elves:
        result.status, result.evidence = "WARN", f"NPU build produced no per-core ELF files under {result.build_dir}"
        return result
    disasm_failures = 0
    for elf in elves:
        prefix = sanitize_prefix(str(elf.relative_to(result.build_dir).with_suffix("")))
        ok, _ = run_logged(ctx, result, f"disassemble_{prefix}", [ctx.disassemble, "npu", "--kind", "elf", "--mcpu", "aie2p", "--triple", "aie2p-none-unknown-elf", "--output-dir", result.artifacts_dir, "--prefix", prefix, elf], env=env)
        disasm_failures += 0 if ok else 1
    txn_note = "transaction stream not generated"
    insts = result.build_dir / "air.insts.bin"
    if insts.exists():
        ok, log = run_logged(ctx, result, "disassemble_air_insts", [ctx.disassemble, "npu", "--kind", "txn", "--output-dir", result.artifacts_dir, "--prefix", "npu_air_insts", insts], env=env)
        txn_note = "transaction stream disassembled" if ok else f"transaction stream disassembly failed; see {log}"
    if ctx.run_enabled:
        exe = result.build_dir / "test.exe"
        if not exe.exists():
            ok, log = run_logged(ctx, result, "build_test_exe", ["make", "-C", source_dir, f"BUILD_DIR={result.build_dir}", "AIE_TARGET=aie2p", "build-test-exe"], env=env)
            if not ok or not exe.exists():
                note_run_failure(result, log, "build-test-exe")
        if exe.exists():
            ok, log = run_logged(ctx, result, "profile", ["./test.exe", "-x", "air.xclbin", "-k", "MLIR_AIE", "-i", "air.insts.bin", "-M", M, "-K", K, "-N", N, "-v", "0", "--warmups", ctx.warmups, "--iterations", ctx.iterations, "--b-layout", "row"], cwd=result.build_dir, env=env)
            result.runtime = f"ran; see {log}" if ok else result.runtime
            if not ok:
                note_run_failure(result, log, "profile")
    combined = "\n".join(read_text(path) for path in sorted(result.artifacts_dir.glob("*.disasm.s")))
    vmac = count_regex(combined, r"\bvmac\b")
    vloads = count_regex(combined, r"\bvld[ab]?\b|\bvlda\b|\bvldb\b")
    vstores = count_regex(combined, r"\bvst\b")
    result.evidence = f"{compile_note}, core_elves={len(elves)}, disasm_failures={disasm_failures}, vmac={vmac}, vloads={vloads}, vstores={vstores}, {txn_note}"
    result.status = "PASS" if disasm_failures == 0 and vmac > 0 else "WARN"
    if ctx.run_enabled and "failed" in result.runtime and result.status == "PASS":
        result.status = "WARN"
    if ctx.run_enabled and (run_log := result.logs.get("profile")) and run_log.exists():
        parse_host_perf(ctx, result, run_log, "host run.wait")
        result.perf_notes = f"runtime_loop_tiling={ctx.npu_runtime_loop_tiling}; b_layout={last_kv_value(run_log, 'b_layout') or 'row'}; warmups={last_kv_value(run_log, 'warmups') or ctx.warmups}; validation={last_kv_value(run_log, 'validation') or 'unknown'}; excludes output BO sync; timing wraps run.wait"
    return result


def selected_backends(name: str) -> list[str]:
    return ["cpu", "gpu", "npu"] if name == "all" else [name]


def strict_failed(results: dict[str, BackendResult], selected: set[str], run_enabled: bool) -> bool:
    for name in selected:
        result = results[name]
        if result.status not in {"PASS", "SKIP"}:
            return True
        if run_enabled and result.target_tops is not None:
            if result.perf_tops is None or result.perf_tops < result.target_tops:
                return True
    return False


def write_report(report: Path, ctx: RunContext, args: argparse.Namespace, results: dict[str, BackendResult]) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", encoding="utf-8") as f:
        f.write("# GEMM int8 Benchmark Report\n\n")
        f.write(f"Artifacts: `{ctx.out_dir}`\n\n")
        f.write("## Run Controls\n\n| Field | Value |\n| --- | --- |\n")
        for key, value in (("Selected backend", args.backend), ("Execute kernels", args.run), ("Strict mode", args.strict), ("GPU sweep variants", args.gpu_sweep_variants), ("GPU sweep repetitions", args.gpu_sweep_repetitions), ("GPU sweep group sizes", ",".join(str(size) for size in args.gpu_sweep_group_sizes)), ("Warmups", args.warmups), ("Iterations", args.iterations), ("CPU threads", args.cpu_threads), ("NPU runtime loop tiling", args.npu_runtime_loop_tiling), ("Build root", ctx.build_root), ("GPU arch", args.gpu_arch), ("GPU int8 GEMM variant", args.gpu_int8_gemm_variant), ("GPU int8 GEMM group size", args.gpu_int8_gemm_group_size), ("Shape", f"M=N=K={M}, int8 x int8 -> int32")):
            f.write(f"| {key} | `{value}` |\n")
        f.write("\n## ISA Verdicts\n\n| Backend | Status | Evidence | Runtime |\n| --- | --- | --- | --- |\n")
        for name in ("cpu", "gpu", "npu"):
            result = results[name]
            f.write(f"| {name.upper()} | {result.status} | {result.evidence} | {result.runtime} |\n")
        f.write("\n## Performance\n\n| Backend | Timing domain | Count | Latency | Throughput | Target TOPS | % of Target | Notes |\n| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for name in ("cpu", "gpu", "npu"):
            result = results[name]
            target = f"{result.target_tops:.3f}" if result.target_tops is not None else "n/a"
            f.write(f"| {name.upper()} | {result.perf_domain} | {result.perf_count} | {result.perf_latency} | {result.perf_throughput} | {target} | {result.target_pct} | {result.perf_notes} |\n")
        f.write("\n## Logs\n\n")
        for name in ("cpu", "gpu", "npu"):
            result = results[name]
            entries = ", ".join(f"{stem} `{path}`" for stem, path in sorted(result.logs.items()))
            f.write(f"- {name.upper()}: {entries or 'n/a'}\n")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["all", "cpu", "gpu", "npu"], default="all", help="backend to process (default: all)")
    parser.add_argument("--out-dir", type=Path, required=True, help="report/artifact root")
    parser.add_argument("--build-dir", type=Path, default=None, help="shared build root; backend subdirectories are created below it")
    parser.add_argument("--run", action="store_true", help="execute selected kernels")
    parser.add_argument("--strict", action="store_true", help="exit non-zero when any selected backend is not PASS")
    parser.add_argument("--gpu-arch", default=os.environ.get("AIR_GPU_CHIP", "gfx1150"), help="AMDGPU chip for GPU lowering (default: AIR_GPU_CHIP or gfx1150)")
    parser.add_argument("--gpu-int8-gemm-variant", choices=GPU_INT8_GEMM_VARIANTS, default=os.environ.get("AIR_INT8_GEMM_VARIANT", DEFAULT_GPU_INT8_GEMM_VARIANT), help="GPU INT8 GEMM lowering variant (default: %(default)s)")
    parser.add_argument("--gpu-int8-gemm-group-size", type=gpu_group_size, default=gpu_group_size(os.environ.get("AIR_INT8_GEMM_GROUP_SIZE", str(DEFAULT_GPU_INT8_GEMM_GROUP_SIZE))), choices=GPU_INT8_GEMM_GROUP_SIZES, help="GPU INT8 GEMM grouped M size for grouped variants (default: %(default)s)")
    parser.add_argument("--gpu-sweep-variants", action="store_true", help="run the fixed GPU INT8 GEMM variant sweep and write CSV/Markdown evidence")
    parser.add_argument("--gpu-sweep-group-sizes", type=gpu_group_sizes, default=DEFAULT_GPU_INT8_GEMM_SWEEP_GROUP_SIZES, help="comma-separated grouped M sizes for grouped swizzle sweep rows (default: 2,4,8)")
    parser.add_argument("--gpu-sweep-repetitions", type=positive_int, default=DEFAULT_GPU_INT8_GEMM_SWEEP_REPETITIONS, help="runtime repetitions per GPU sweep candidate (default: 3)")
    parser.add_argument("--cpu-threads", type=positive_int, default=12, help="CPU worker threads passed to the CPU benchmark (default: 12)")
    parser.add_argument("--npu-runtime-loop-tiling", default="2,4", metavar="M,N", help="AIR runtime loop tiling sizes for NPU compile (default: 2,4)")
    parser.add_argument("--warmups", type=nonnegative_int, default=10, help="warmup iterations for every backend (default: 10)")
    parser.add_argument("--iterations", type=positive_int, default=20, help="timed iterations for every backend (default: 20)")
    args = parser.parse_args(argv)
    tiling_values = args.npu_runtime_loop_tiling.split(",")
    if len(tiling_values) != 2:
        parser.error("--npu-runtime-loop-tiling must contain two positive integers")
    try:
        parsed_tiling = [positive_int(value) for value in tiling_values]
    except argparse.ArgumentTypeError as exc:
        parser.error(f"--npu-runtime-loop-tiling: {exc}")
    args.npu_runtime_loop_tiling = f"{parsed_tiling[0]},{parsed_tiling[1]}"
    if args.gpu_sweep_variants and args.backend not in {"all", "gpu"}:
        parser.error("--gpu-sweep-variants requires --backend all or --backend gpu")
    return args


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    out_dir = args.out_dir.resolve()
    ctx = RunContext(Path(__file__).resolve().parents[2], out_dir, args.build_dir.resolve() if args.build_dir else out_dir / "build", out_dir / "logs", args.warmups, args.iterations, args.gpu_arch, args.gpu_int8_gemm_variant, args.gpu_int8_gemm_group_size, args.run, args.cpu_threads, args.npu_runtime_loop_tiling)
    for path in (ctx.out_dir, ctx.build_root, ctx.logs_dir):
        path.mkdir(parents=True, exist_ok=True)
    selected = {"gpu"} if args.gpu_sweep_variants else set(selected_backends(args.backend))
    runners = {"cpu": cpu_backend, "gpu": gpu_backend, "npu": npu_backend}
    results = {name: backend_result(ctx, name) for name in ("cpu", "gpu", "npu")}
    if args.gpu_sweep_variants:
        results["gpu"] = run_gpu_variant_sweep(ctx, GPU_INT8_GEMM_SWEEP_VARIANTS, args.gpu_sweep_group_sizes, args.gpu_sweep_repetitions)
    else:
        for name in ("cpu", "gpu", "npu"):
            if name in selected:
                results[name] = runners[name](ctx)
    report = out_dir / "gemm_int8_report.md"
    write_report(report, ctx, args, results)
    print(f"Report: {report}")
    return 1 if args.strict and strict_failed(results, selected, args.run) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
