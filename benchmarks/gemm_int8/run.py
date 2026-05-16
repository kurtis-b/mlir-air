#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# fmt: off

"""Build, inspect, and optionally run the shared int8 GEMM benchmark."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

M = N = K = 1024


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


def to_tops(gops: str) -> str:
    try:
        return f"{float(gops) / 1000.0:.6f}" if gops else "n/a"
    except ValueError:
        return "n/a"


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
    run_enabled: bool

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
    perf_notes: str = "not run"
    logs: dict[str, Path] = field(default_factory=dict)


def backend_result(ctx: RunContext, name: str, create: bool = False) -> BackendResult:
    build_dir = ctx.build_root / name
    result = BackendResult(name, build_dir, build_dir / "artifacts")
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
        ok, log = run_logged(ctx, result, "run", [binary, "--warmups", ctx.warmups, "--iterations", ctx.iterations])
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
        avg_us, min_us, max_us, gops = (last_kv_value(run_log, key) for key in ("avg_us", "min_us", "max_us", "gops"))
        result.perf_domain = last_kv_value(run_log, "timing_domain") or "host_steady_clock"
        result.perf_count = str(ctx.iterations)
        result.perf_latency = f"mean {avg_us or 'n/a'} us, min {min_us or 'n/a'} us, max {max_us or 'n/a'} us"
        result.perf_throughput = f"{gops or 'n/a'} GOPS ({to_tops(gops)} TOPS)"
        result.perf_notes = f"warmups={ctx.warmups}; validation={last_kv_value(run_log, 'validation') or 'unknown'}"
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


def render_gpu_mlir(source: Path, dest: Path, ctx: RunContext) -> None:
    text = read_text(source)
    for old, new in (
        ("%c10 = arith.constant 10 : index", f"%c10 = arith.constant {ctx.warmups} : index"),
        ("%c5 = arith.constant 5 : index", f"%c5 = arith.constant {ctx.iterations} : index"),
        ("%c10_i64 = arith.constant 10 : i64", f"%c10_i64 = arith.constant {ctx.warmups} : i64"),
        ("%c5_i64 = arith.constant 5 : i64", f"%c5_i64 = arith.constant {ctx.iterations} : i64"),
    ):
        text = replace_once(text, old, new)
    write_text(dest, text)


def gpu_backend(ctx: RunContext) -> BackendResult:
    result = backend_result(ctx, "gpu", True)
    source = ctx.repo / "test" / "gpu" / "int8_gemm" / "air_sync.mlir"
    generated = result.build_dir / "int8_gemm.air_sync.mlir"
    isa = result.artifacts_dir / "gpu_int8_gemm.isa.s"
    summary = result.artifacts_dir / "gpu_int8_gemm.summary.txt"
    final_mlir = result.artifacts_dir / "gpu_int8_gemm.final.mlir"
    try:
        render_gpu_mlir(source, generated, ctx)
        write_text(log_path(ctx, result, "render"), f"generated {generated}\nwarmups={ctx.warmups}\niterations={ctx.iterations}\n")
    except Exception as exc:  # noqa: BLE001 - report rendering failures in the same log flow.
        log = log_path(ctx, result, "render")
        write_text(log, f"failed to render GPU MLIR: {exc}\n")
        result.status, result.evidence = "WARN", f"GPU MLIR render failed; see {log}"
        return result
    ok, log = run_logged(ctx, result, "disassemble", [ctx.disassemble, "gpu", "--gpu-arch", ctx.gpu_arch, "--output-dir", result.artifacts_dir, "--prefix", "gpu_int8_gemm", "--expect", "v_wmma_i32_16x16x16_iu8", "--forbid", r"v_wmma_.*16x16x64|v_swmmac|swmmac", generated])
    if not ok:
        result.status, result.evidence = "WARN", f"GPU lowering/disassembly failed or required marker was absent; see {log}"
        return result
    if ctx.run_enabled:
        runner, airgpu = gpu_tools(ctx.repo)
        log = log_path(ctx, result, "run")
        if not runner or not Path(runner).exists():
            write_text(log, "ERROR: mlir-runner not found\n")
            note_run_failure(result, log)
        elif not airgpu or not Path(airgpu).exists():
            write_text(log, "ERROR: libairgpu.so not found\n")
            note_run_failure(result, log)
        else:
            env = os.environ.copy()
            env.setdefault("AIRGPU_USE_HIP_MALLOC", "1")
            ok, log = run_capture(log, [runner, "--entry-point-result=void", f"--shared-libs={airgpu}", final_mlir], env=env)
            result.runtime = f"ran; see {log}" if ok else result.runtime
            if not ok:
                note_run_failure(result, log)
    wmma = count_regex(isa, r"\bv_wmma_i32_16x16x16_iu8\b")
    barriers = count_regex(isa, r"\bs_barrier\b")
    scratch = count_regex(isa, r"uses_flat_scratch\s+1")
    spills = count_regex(summary, r"spill_count = [1-9]|_spill_count: [1-9]")
    result.status = "PASS" if wmma and scratch == 0 and spills == 0 else "FAIL"
    if ctx.run_enabled and "failed" in result.runtime and result.status == "PASS":
        result.status = "WARN"
    result.evidence = f"wmma={wmma}, barriers={barriers}, scratch_markers={scratch}, spills={spills}"
    if ctx.run_enabled and (run_log := result.logs.get("run")) and run_log.exists():
        result.perf_domain = "kernel_event"
        result.perf_count = timing_field(run_log, "kernel_event", "count") or "n/a"
        result.perf_latency = f"mean {timing_field(run_log, 'kernel_event', 'mean_ms') or 'n/a'} ms, min {timing_field(run_log, 'kernel_event', 'min_ms') or 'n/a'} ms, max {timing_field(run_log, 'kernel_event', 'max_ms') or 'n/a'} ms"
        result.perf_throughput = f"{timing_field(run_log, 'kernel_event', 'tops') or 'n/a'} TOPS"
        result.perf_notes = f"host_dispatch_wait_mean_ms={timing_field(run_log, 'host_dispatch_wait', 'mean_ms') or 'n/a'}; peak_pct={last_kv_value(run_log, 'kernel_event_peak_pct') or 'n/a'}; warmups={ctx.warmups}"
    return result


def find_npu_elves(build_dir: Path) -> list[Path]:
    return sorted(build_dir.rglob("bare_matmul*_core_*.elf")) or sorted(build_dir.rglob("*.elf"))


def npu_env(ctx: RunContext) -> dict[str, str]:
    env = os.environ.copy()
    for path in [env.get("PEANO_INSTALL_DIR", ""), *(str(path) for path in sorted((ctx.repo / "sandbox/lib").glob("python*/site-packages/llvm-aie")))]:
        if path and (Path(path) / "bin" / "llc").exists():
            env["PEANO_INSTALL_DIR"] = path
            break
    return env


def npu_backend(ctx: RunContext) -> BackendResult:
    result = backend_result(ctx, "npu", True)
    env = npu_env(ctx)
    source_dir = ctx.repo / "test" / "xrt" / "46_triton_matmul_ver4_strix_8x4_i8_i8_i32"
    reused = (result.build_dir / "air.xclbin").exists() and (result.build_dir / "air.insts.bin").exists() and find_npu_elves(result.build_dir)
    if reused:
        compile_note = f"reused build_dir={result.build_dir}"
        write_text(log_path(ctx, result, "build"), f"Reusing existing NPU artifacts in {result.build_dir}\n")
    else:
        compile_note = "fresh compile"
        ok, log = run_logged(ctx, result, "build", ["make", "-C", source_dir, f"BUILD_DIR={result.build_dir}", "AIE_TARGET=aie2p", f"M={M}", f"K={K}", f"N={N}", "compile-xclbin"], env=env)
        if not ok:
            result.status, result.evidence = "WARN", f"NPU compile-xclbin failed; see {log}"
            return result
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
            run_logged(ctx, result, "build_test_exe", ["make", "-C", source_dir, f"BUILD_DIR={result.build_dir}", "AIE_TARGET=aie2p", "build-test-exe"], env=env)
        ok, log = run_logged(ctx, result, "profile", ["./test.exe", "-x", "air.xclbin", "-k", "MLIR_AIE", "-i", "air.insts.bin", "-M", M, "-K", K, "-N", N, "-v", "0", "--warmups", ctx.warmups, "--iterations", ctx.iterations], cwd=result.build_dir, env=env)
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
        result.perf_notes = f"warmups={last_kv_value(run_log, 'warmups') or ctx.warmups}; excludes output BO sync; timing wraps run.wait"
    return result


def selected_backends(name: str) -> list[str]:
    return ["cpu", "gpu", "npu"] if name == "all" else [name]


def write_report(report: Path, ctx: RunContext, args: argparse.Namespace, results: dict[str, BackendResult]) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", encoding="utf-8") as f:
        f.write("# GEMM int8 Benchmark Report\n\n")
        f.write(f"Artifacts: `{ctx.out_dir}`\n\n")
        f.write("## Run Controls\n\n| Field | Value |\n| --- | --- |\n")
        for key, value in (("Selected backend", args.backend), ("Execute kernels", args.run), ("Strict mode", args.strict), ("Warmups", args.warmups), ("Iterations", args.iterations), ("Build root", ctx.build_root), ("GPU arch", args.gpu_arch), ("Shape", f"M=N=K={M}, int8 x int8 -> int32")):
            f.write(f"| {key} | `{value}` |\n")
        f.write("\n## ISA Verdicts\n\n| Backend | Status | Evidence | Runtime |\n| --- | --- | --- | --- |\n")
        for name in ("cpu", "gpu", "npu"):
            result = results[name]
            f.write(f"| {name.upper()} | {result.status} | {result.evidence} | {result.runtime} |\n")
        f.write("\n## Performance\n\n| Backend | Timing domain | Count | Latency | Throughput | Notes |\n| --- | --- | --- | --- | --- | --- |\n")
        for name in ("cpu", "gpu", "npu"):
            result = results[name]
            f.write(f"| {name.upper()} | {result.perf_domain} | {result.perf_count} | {result.perf_latency} | {result.perf_throughput} | {result.perf_notes} |\n")
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
    parser.add_argument("--warmups", type=nonnegative_int, default=10, help="warmup iterations for every backend (default: 10)")
    parser.add_argument("--iterations", type=positive_int, default=20, help="timed iterations for every backend (default: 20)")
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    out_dir = args.out_dir.resolve()
    ctx = RunContext(Path(__file__).resolve().parents[2], out_dir, args.build_dir.resolve() if args.build_dir else out_dir / "build", out_dir / "logs", args.warmups, args.iterations, args.gpu_arch, args.run)
    for path in (ctx.out_dir, ctx.build_root, ctx.logs_dir):
        path.mkdir(parents=True, exist_ok=True)
    selected = set(selected_backends(args.backend))
    runners = {"cpu": cpu_backend, "gpu": gpu_backend, "npu": npu_backend}
    results = {name: backend_result(ctx, name) for name in ("cpu", "gpu", "npu")}
    for name in ("cpu", "gpu", "npu"):
        if name in selected:
            results[name] = runners[name](ctx)
    report = out_dir / "gemm_int8_report.md"
    write_report(report, ctx, args, results)
    print(f"Report: {report}")
    return 1 if args.strict and any(results[name].status not in {"PASS", "SKIP"} for name in selected) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
