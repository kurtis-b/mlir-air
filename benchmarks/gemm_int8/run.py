#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.

"""Build, inspect, and optionally run the shared int8 GEMM benchmark."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence


M = 1024
N = 1024
K = 1024


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


def shell_join(argv: Sequence[object]) -> str:
    import shlex

    return " ".join(shlex.quote(str(arg)) for arg in argv)


def relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def count_regex(path: Path, pattern: str) -> int:
    text = read_text(path)
    if not text:
        return 0
    return len(re.findall(pattern, text, flags=re.MULTILINE))


def first_match(path: Path, pattern: str) -> str:
    text = read_text(path)
    if not text:
        return ""
    match = re.search(pattern, text, flags=re.MULTILINE)
    return match.group(0).strip() if match else ""


def last_kv_value(path: Path, key: str) -> str:
    text = read_text(path)
    matches = re.findall(rf"\b{re.escape(key)}=([^\s]+)", text)
    return matches[-1] if matches else ""


def timing_field(path: Path, domain: str, field_name: str) -> str:
    for line in read_text(path).splitlines():
        if f"timing_domain={domain}" not in line:
            continue
        fields = dict(re.findall(r"([A-Za-z0-9_.-]+)=([^\s]+)", line))
        if field_name in fields:
            return fields[field_name]
    return ""


def to_tops(gops: str) -> str:
    if not gops:
        return "n/a"
    try:
        return f"{float(gops) / 1000.0:.6f}"
    except ValueError:
        return "n/a"


def us_to_ms(us: str) -> str:
    if not us:
        return "n/a"
    try:
        return f"{float(us) / 1000.0:.6f}"
    except ValueError:
        return "n/a"


def sanitize_prefix(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "artifact"


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one occurrence of {old!r}, found {count}")
    return text.replace(old, new, 1)


@dataclass
class CommandResult:
    returncode: int
    log: Path

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def run_capture(
    log: Path,
    argv: Sequence[object],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> CommandResult:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as output:
        if cwd is not None:
            output.write(f"+ cd {cwd} && {shell_join(argv)}\n")
        else:
            output.write(f"+ {shell_join(argv)}\n")
        output.flush()
        try:
            completed = subprocess.run(
                [str(arg) for arg in argv],
                cwd=str(cwd) if cwd else None,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        except OSError as exc:
            output.write(f"ERROR: {exc}\n")
            return CommandResult(127, log)
    return CommandResult(completed.returncode, log)


def write_log(log: Path, message: str) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(message, encoding="utf-8")


@dataclass
class BackendResult:
    backend: str
    status: str = "SKIP"
    evidence: str = "not selected"
    runtime: str = "not run"
    perf_domain: str = "not run"
    perf_count: str = "n/a"
    perf_latency: str = "n/a"
    perf_throughput: str = "n/a"
    perf_notes: str = "not run"
    build_dir: Path | None = None
    artifacts_dir: Path | None = None
    sources: list[Path] = field(default_factory=list)
    logs: dict[str, Path] = field(default_factory=dict)


class BackendAdapter:
    name = "backend"

    def __init__(
        self,
        repo: Path,
        out_dir: Path,
        build_root: Path,
        logs_dir: Path,
        warmups: int,
        iterations: int,
        gpu_arch: str,
        run_enabled: bool,
    ) -> None:
        self.repo = repo
        self.out_dir = out_dir
        self.build_dir = build_root / self.name
        self.artifacts_dir = self.build_dir / "artifacts"
        self.logs_dir = logs_dir
        self.warmups = warmups
        self.iterations = iterations
        self.gpu_arch = gpu_arch
        self.run_enabled = run_enabled
        self.disassemble = repo / "utils" / "isa_inspect" / "disassemble.sh"
        self.result = BackendResult(
            backend=self.name,
            build_dir=self.build_dir,
            artifacts_dir=self.artifacts_dir,
        )

    def log_path(self, stem: str) -> Path:
        path = self.logs_dir / f"{self.name}_{stem}.log"
        self.result.logs[stem] = path
        return path

    def run_command(
        self,
        stem: str,
        argv: Sequence[object],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        return run_capture(self.log_path(stem), argv, cwd=cwd, env=env)

    def build(self) -> bool:
        raise NotImplementedError

    def inspect_isa(self) -> bool:
        raise NotImplementedError

    def run(self) -> bool:
        return True

    def parse_result(self) -> None:
        pass

    def execute(self) -> BackendResult:
        self.build_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        if not self.build():
            return self.result
        if not self.inspect_isa():
            return self.result
        if self.run_enabled:
            self.run()
        self.parse_result()
        return self.result


class CpuAdapter(BackendAdapter):
    name = "cpu"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.source_dir = self.repo / "test" / "cpu" / "int8_gemm"
        self.binary = self.build_dir / "int8_gemm_cpu"
        self.disasm = self.artifacts_dir / "cpu_int8_gemm.disasm.s"
        self.result.sources = [self.source_dir / "int8_gemm.cpp"]

    def build(self) -> bool:
        cmd = ["make", "-C", self.source_dir, f"BUILD_DIR={self.build_dir}"]
        completed = self.run_command("build", cmd)
        if not completed.ok:
            self.result.status = "WARN"
            self.result.evidence = f"CPU benchmark build failed; see {completed.log}"
            return False
        return True

    def inspect_isa(self) -> bool:
        cmd = [
            self.disassemble,
            "cpu",
            "--output-dir",
            self.artifacts_dir,
            "--prefix",
            "cpu_int8_gemm",
            "--symbol",
            "cpu_i8_gemm_vnni",
            "--expect",
            "vpdpbusd",
            self.binary,
        ]
        completed = self.run_command("disassemble", cmd)
        if not completed.ok:
            self.result.status = "FAIL"
            self.result.evidence = f"CPU disassembly did not show required VNNI marker; see {completed.log}"
            return False
        return True

    def run(self) -> bool:
        cmd = [
            self.binary,
            "--warmups",
            self.warmups,
            "--iterations",
            self.iterations,
        ]
        completed = self.run_command("run", cmd)
        if not completed.ok:
            self.result.runtime = f"run failed; see {completed.log}"
            self.result.perf_notes = f"run failed; see {completed.log}"
            if self.result.status == "PASS":
                self.result.status = "WARN"
            return False
        self.result.runtime = f"ran; see {completed.log}"
        return True

    def parse_result(self) -> None:
        vnni_count = count_regex(self.disasm, r"\bvpdpbusd\b")
        zmm_count = count_regex(self.disasm, r"\bzmm[0-9]+")
        self.result.status = "PASS" if vnni_count > 0 else "FAIL"
        if (
            self.run_enabled
            and "failed" in self.result.runtime
            and self.result.status == "PASS"
        ):
            self.result.status = "WARN"
        self.result.evidence = f"vpdpbusd={vnni_count}, zmm_refs={zmm_count}"

        if not self.run_enabled:
            return
        run_log = self.result.logs.get("run")
        if not run_log or not run_log.exists():
            return
        avg_us = last_kv_value(run_log, "avg_us")
        min_us = last_kv_value(run_log, "min_us")
        max_us = last_kv_value(run_log, "max_us")
        gops = last_kv_value(run_log, "gops")
        validation = last_kv_value(run_log, "validation")
        domain = last_kv_value(run_log, "timing_domain") or "host_steady_clock"
        self.result.perf_domain = domain
        self.result.perf_count = str(self.iterations)
        self.result.perf_latency = (
            f"mean {avg_us or 'n/a'} us, min {min_us or 'n/a'} us, "
            f"max {max_us or 'n/a'} us"
        )
        self.result.perf_throughput = f"{gops or 'n/a'} GOPS ({to_tops(gops)} TOPS)"
        self.result.perf_notes = (
            f"warmups={self.warmups}; validation={validation or 'unknown'}"
        )


class GpuAdapter(BackendAdapter):
    name = "gpu"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.source = self.repo / "test" / "gpu" / "int8_gemm" / "air_sync.mlir"
        self.generated_source = self.build_dir / "int8_gemm.air_sync.mlir"
        self.final_mlir = self.artifacts_dir / "gpu_int8_gemm.final.mlir"
        self.isa = self.artifacts_dir / "gpu_int8_gemm.isa.s"
        self.summary = self.artifacts_dir / "gpu_int8_gemm.summary.txt"
        self.result.sources = [self.source, self.generated_source]

    def render_mlir(self) -> None:
        text = read_text(self.source)
        text = replace_once(
            text,
            "%c10 = arith.constant 10 : index",
            f"%c10 = arith.constant {self.warmups} : index",
        )
        text = replace_once(
            text,
            "%c5 = arith.constant 5 : index",
            f"%c5 = arith.constant {self.iterations} : index",
        )
        text = replace_once(
            text,
            "%c10_i64 = arith.constant 10 : i64",
            f"%c10_i64 = arith.constant {self.warmups} : i64",
        )
        text = replace_once(
            text,
            "%c5_i64 = arith.constant 5 : i64",
            f"%c5_i64 = arith.constant {self.iterations} : i64",
        )
        self.generated_source.write_text(text, encoding="utf-8")

    def build(self) -> bool:
        try:
            self.render_mlir()
        except Exception as exc:  # noqa: BLE001 - report the rendering failure.
            log = self.log_path("render")
            write_log(log, f"failed to render GPU MLIR: {exc}\n")
            self.result.status = "WARN"
            self.result.evidence = f"GPU MLIR render failed; see {log}"
            return False
        write_log(
            self.log_path("render"),
            (
                f"generated {self.generated_source}\n"
                f"warmups={self.warmups}\niterations={self.iterations}\n"
            ),
        )
        return True

    def inspect_isa(self) -> bool:
        cmd = [
            self.disassemble,
            "gpu",
            "--gpu-arch",
            self.gpu_arch,
            "--output-dir",
            self.artifacts_dir,
            "--prefix",
            "gpu_int8_gemm",
            "--expect",
            "v_wmma_i32_16x16x16_iu8",
            "--forbid",
            r"v_wmma_.*16x16x64|v_swmmac|swmmac",
            self.generated_source,
        ]
        completed = self.run_command("disassemble", cmd)
        if not completed.ok:
            self.result.status = "WARN"
            self.result.evidence = (
                "GPU lowering/disassembly failed or required marker was absent; "
                f"see {completed.log}"
            )
            return False
        return True

    def run(self) -> bool:
        mlir_runner = os.environ.get("MLIR_RUNNER")
        if not mlir_runner:
            mlir_runner = shutil.which("mlir-runner")
        if not mlir_runner:
            candidate = self.repo / "llvm" / "install-amdgpu" / "bin" / "mlir-runner"
            if candidate.exists():
                mlir_runner = str(candidate)

        airgpu_lib = os.environ.get("AIRGPU_LIB")
        if not airgpu_lib and os.environ.get("MLIR_AIR_INSTALL_DIR"):
            airgpu_lib = str(
                Path(os.environ["MLIR_AIR_INSTALL_DIR"]) / "lib" / "libairgpu.so"
            )
        if not airgpu_lib:
            candidate = self.repo / "install-gpu" / "lib" / "libairgpu.so"
            if candidate.exists():
                airgpu_lib = str(candidate)

        log = self.log_path("run")
        if not mlir_runner or not Path(mlir_runner).exists():
            write_log(log, "ERROR: mlir-runner not found\n")
            self.result.runtime = f"run failed; see {log}"
            self.result.perf_notes = f"run failed; see {log}"
            if self.result.status == "PASS":
                self.result.status = "WARN"
            return False
        if not airgpu_lib or not Path(airgpu_lib).exists():
            write_log(log, "ERROR: libairgpu.so not found\n")
            self.result.runtime = f"run failed; see {log}"
            self.result.perf_notes = f"run failed; see {log}"
            if self.result.status == "PASS":
                self.result.status = "WARN"
            return False

        env = os.environ.copy()
        env.setdefault("AIRGPU_USE_HIP_MALLOC", "1")
        cmd = [
            mlir_runner,
            "--entry-point-result=void",
            f"--shared-libs={airgpu_lib}",
            self.final_mlir,
        ]
        completed = run_capture(log, cmd, env=env)
        if not completed.ok:
            self.result.runtime = f"run failed; see {completed.log}"
            self.result.perf_notes = f"run failed; see {completed.log}"
            if self.result.status == "PASS":
                self.result.status = "WARN"
            return False
        self.result.runtime = f"ran; see {completed.log}"
        return True

    def parse_result(self) -> None:
        wmma_count = count_regex(self.isa, r"\bv_wmma_i32_16x16x16_iu8\b")
        barrier_count = count_regex(self.isa, r"\bs_barrier\b")
        scratch = count_regex(self.isa, r"uses_flat_scratch\s+1")
        spills = count_regex(self.summary, r"spill_count = [1-9]|_spill_count: [1-9]")
        wavefront = first_match(self.summary, r"wavefront_size|amdhsa_wavefront_size32")
        vgprs = first_match(self.summary, r"vgpr_count|amdhsa_next_free_vgpr")
        sgprs = first_match(self.summary, r"sgpr_count|amdhsa_next_free_sgpr")
        lds = first_match(self.isa, r"amdhsa_group_segment_fixed_size[^\n]*")

        self.result.status = (
            "PASS" if wmma_count > 0 and scratch == 0 and spills == 0 else "FAIL"
        )
        if (
            self.run_enabled
            and "failed" in self.result.runtime
            and self.result.status == "PASS"
        ):
            self.result.status = "WARN"
        evidence = [
            f"wmma={wmma_count}",
            f"barriers={barrier_count}",
            f"scratch_markers={scratch}",
            f"spills={spills}",
        ]
        evidence.extend(item for item in (wavefront, vgprs, sgprs, lds) if item)
        self.result.evidence = ", ".join(evidence)

        if not self.run_enabled:
            return
        run_log = self.result.logs.get("run")
        if not run_log or not run_log.exists():
            return
        count = timing_field(run_log, "kernel_event", "count")
        min_ms = timing_field(run_log, "kernel_event", "min_ms")
        mean_ms = timing_field(run_log, "kernel_event", "mean_ms")
        max_ms = timing_field(run_log, "kernel_event", "max_ms")
        tops = timing_field(run_log, "kernel_event", "tops")
        host_mean = timing_field(run_log, "host_dispatch_wait", "mean_ms")
        peak_pct = last_kv_value(run_log, "kernel_event_peak_pct")
        self.result.perf_domain = "kernel_event"
        self.result.perf_count = count or "n/a"
        self.result.perf_latency = (
            f"mean {mean_ms or 'n/a'} ms, min {min_ms or 'n/a'} ms, "
            f"max {max_ms or 'n/a'} ms"
        )
        self.result.perf_throughput = f"{tops or 'n/a'} TOPS"
        self.result.perf_notes = (
            f"host_dispatch_wait_mean_ms={host_mean or 'n/a'}; "
            f"peak_pct={peak_pct or 'n/a'}; warmups={self.warmups}"
        )


class NpuAdapter(BackendAdapter):
    name = "npu"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.source_dir = (
            self.repo / "test" / "xrt" / "46_triton_matmul_ver4_strix_8x4_i8_i8_i32"
        )
        self.result.sources = [
            self.source_dir / "asm_src.mlir",
            self.source_dir / "transform_aie2p.mlir",
            self.source_dir / "test.cpp",
            self.source_dir / "Makefile",
        ]
        self.npu_elves: list[Path] = []
        self.disasm_failures = 0
        self.txn_note = "transaction stream not generated"
        self.compile_note = "fresh compile"

    def existing_build_is_complete(self) -> bool:
        return (
            (self.build_dir / "air.xclbin").exists()
            and (self.build_dir / "air.insts.bin").exists()
            and bool(self.find_elves())
        )

    def find_elves(self) -> list[Path]:
        elves = sorted(self.build_dir.rglob("bare_matmul*_core_*.elf"))
        if not elves:
            elves = sorted(self.build_dir.rglob("*.elf"))
        return elves

    def build(self) -> bool:
        build_log = self.log_path("build")
        if self.existing_build_is_complete():
            self.compile_note = f"reused build_dir={self.build_dir}"
            write_log(
                build_log,
                (
                    f"Reusing existing NPU artifacts in {self.build_dir}\n"
                    f"air.xclbin={self.build_dir / 'air.xclbin'}\n"
                    f"air.insts.bin={self.build_dir / 'air.insts.bin'}\n"
                ),
            )
            return True

        cmd = [
            "make",
            "-C",
            self.source_dir,
            f"BUILD_DIR={self.build_dir}",
            "AIE_TARGET=aie2p",
            f"M={M}",
            f"K={K}",
            f"N={N}",
            "compile-xclbin",
        ]
        completed = run_capture(build_log, cmd)
        if not completed.ok:
            self.result.status = "WARN"
            self.result.evidence = f"NPU compile-xclbin failed; see {completed.log}"
            return False
        return True

    def inspect_isa(self) -> bool:
        self.npu_elves = self.find_elves()
        if not self.npu_elves:
            self.result.status = "WARN"
            self.result.evidence = (
                f"NPU build produced no per-core ELF files under {self.build_dir}"
            )
            return False

        for elf in self.npu_elves:
            relative = elf.relative_to(self.build_dir).with_suffix("")
            prefix = sanitize_prefix(str(relative))
            log = self.log_path(f"disassemble_{prefix}")
            cmd = [
                self.disassemble,
                "npu",
                "--kind",
                "elf",
                "--mcpu",
                "aie2p",
                "--triple",
                "aie2p-none-unknown-elf",
                "--output-dir",
                self.artifacts_dir,
                "--prefix",
                prefix,
                elf,
            ]
            if not run_capture(log, cmd).ok:
                self.disasm_failures += 1

        insts = self.build_dir / "air.insts.bin"
        if insts.exists():
            log = self.log_path("disassemble_air_insts")
            cmd = [
                self.disassemble,
                "npu",
                "--kind",
                "txn",
                "--output-dir",
                self.artifacts_dir,
                "--prefix",
                "npu_air_insts",
                insts,
            ]
            if run_capture(log, cmd).ok:
                self.txn_note = "transaction stream disassembled"
            else:
                self.txn_note = f"transaction stream disassembly failed; see {log}"
        return True

    def run(self) -> bool:
        exe = self.build_dir / "test.exe"
        if not exe.exists():
            cmd = [
                "make",
                "-C",
                self.source_dir,
                f"BUILD_DIR={self.build_dir}",
                "AIE_TARGET=aie2p",
                "build-test-exe",
            ]
            self.run_command("build_test_exe", cmd)

        cmd = [
            "./test.exe",
            "-x",
            "air.xclbin",
            "-k",
            "MLIR_AIE",
            "-i",
            "air.insts.bin",
            "-M",
            M,
            "-K",
            K,
            "-N",
            N,
            "-v",
            "0",
            "--warmups",
            self.warmups,
            "--iterations",
            self.iterations,
        ]
        completed = self.run_command("profile", cmd, cwd=self.build_dir)
        if not completed.ok:
            self.result.runtime = f"profile failed; see {completed.log}"
            self.result.perf_notes = f"profile failed; see {completed.log}"
            if self.result.status == "PASS":
                self.result.status = "WARN"
            return False
        self.result.runtime = f"ran; see {completed.log}"
        return True

    def parse_result(self) -> None:
        disasm_files = sorted(self.artifacts_dir.glob("*.disasm.s"))
        combined = "\n".join(read_text(path) for path in disasm_files)
        vmac_count = len(re.findall(r"\bvmac\b", combined))
        vload_count = len(re.findall(r"\bvld[ab]?\b|\bvlda\b|\bvldb\b", combined))
        vstore_count = len(re.findall(r"\bvst\b", combined))
        acq_count = len(re.findall(r"\bacq\b", combined))
        rel_count = len(re.findall(r"\brel\b", combined))
        self.result.evidence = (
            f"{self.compile_note}, core_elves={len(self.npu_elves)}, "
            f"disasm_failures={self.disasm_failures}, vmac={vmac_count}, "
            f"vloads={vload_count}, vstores={vstore_count}, acq={acq_count}, "
            f"rel={rel_count}, {self.txn_note}"
        )
        self.result.status = (
            "PASS" if self.disasm_failures == 0 and vmac_count > 0 else "WARN"
        )
        if (
            self.run_enabled
            and "failed" in self.result.runtime
            and self.result.status == "PASS"
        ):
            self.result.status = "WARN"

        if not self.run_enabled:
            return
        run_log = self.result.logs.get("profile")
        if not run_log or not run_log.exists():
            return
        avg_us = last_kv_value(run_log, "avg_us")
        min_us = last_kv_value(run_log, "min_us")
        max_us = last_kv_value(run_log, "max_us")
        gops = last_kv_value(run_log, "gops")
        warmups = last_kv_value(run_log, "warmups") or str(self.warmups)
        iterations = last_kv_value(run_log, "iterations") or str(self.iterations)
        if not avg_us:
            text = read_text(run_log)
            match = re.search(r"Avg NPU matmul time: ([0-9.]+)us\.", text)
            avg_us = match.group(1) if match else ""
            match = re.search(r"Avg NPU gflops: ([0-9.]+)", text)
            gops = match.group(1) if match else gops
            match = re.search(r"Min NPU matmul time: ([0-9.]+)us\.", text)
            min_us = match.group(1) if match else ""
            match = re.search(r"Max NPU matmul time: ([0-9.]+)us\.", text)
            max_us = match.group(1) if match else ""
        self.result.perf_domain = "host run.wait"
        self.result.perf_count = iterations
        self.result.perf_latency = (
            f"mean {avg_us or 'n/a'} us ({us_to_ms(avg_us)} ms), "
            f"min {min_us or 'n/a'} us, max {max_us or 'n/a'} us"
        )
        self.result.perf_throughput = f"{gops or 'n/a'} GOPS ({to_tops(gops)} TOPS)"
        self.result.perf_notes = (
            f"warmups={warmups}; excludes output BO sync; timing wraps run.wait"
        )


def make_skipped(name: str, build_root: Path, out_dir: Path) -> BackendResult:
    return BackendResult(
        backend=name,
        status="SKIP",
        evidence="not selected",
        build_dir=build_root / name,
        artifacts_dir=build_root / name / "artifacts",
    )


def selected_backends(name: str) -> list[str]:
    return ["cpu", "gpu", "npu"] if name == "all" else [name]


def report_path_list(paths: Iterable[Path], repo: Path) -> str:
    values = [f"`{relpath(path, repo)}`" for path in paths]
    return ", ".join(values) if values else "n/a"


def preview(path: Path | None, lines: int) -> str:
    if path is None:
        return "Backend not selected."
    text = read_text(path)
    if not text:
        return f"unavailable: {path}"
    return "\n".join(text.splitlines()[:lines])


def write_report(
    report: Path,
    repo: Path,
    out_dir: Path,
    build_root: Path,
    args: argparse.Namespace,
    results: dict[str, BackendResult],
) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    cpu = results["cpu"]
    gpu = results["gpu"]
    npu = results["npu"]
    cpu_disasm = (
        (cpu.artifacts_dir or Path()) / "cpu_int8_gemm.disasm.s"
        if cpu.status != "SKIP"
        else None
    )
    gpu_summary = (
        (gpu.artifacts_dir or Path()) / "gpu_int8_gemm.summary.txt"
        if gpu.status != "SKIP"
        else None
    )
    first_npu_disasm = None
    if npu.status != "SKIP" and npu.artifacts_dir and npu.artifacts_dir.exists():
        disasm_files = sorted(npu.artifacts_dir.glob("*.disasm.s"))
        first_npu_disasm = disasm_files[0] if disasm_files else None

    with report.open("w", encoding="utf-8") as f:
        f.write("# GEMM int8 Benchmark Report\n\n")
        f.write(f"Artifacts: `{out_dir}`\n\n")
        f.write("## Run Controls\n\n")
        f.write("| Field | Value |\n| --- | --- |\n")
        f.write(f"| Selected backend | `{args.backend}` |\n")
        f.write(f"| Execute kernels | `{args.run}` |\n")
        f.write(f"| Strict mode | `{args.strict}` |\n")
        f.write(f"| Warmups | `{args.warmups}` |\n")
        f.write(f"| Iterations | `{args.iterations}` |\n")
        f.write(f"| Build root | `{build_root}` |\n")
        f.write(f"| GPU arch | `{args.gpu_arch}` |\n\n")

        f.write("## Comparison Contract\n\n")
        f.write("| Field | Value |\n| --- | --- |\n")
        f.write("| Shape | M=N=K=1024 |\n")
        f.write("| Inputs | int8 A, int8 B, values 0..7 |\n")
        f.write("| Output | int32 C |\n")
        f.write("| Operation count | 2 * M * N * K integer ops |\n")
        f.write(
            "| Timing domains | CPU `host_steady_clock`; GPU `kernel_event`; "
            "NPU `host run.wait` |\n\n"
        )

        f.write("## ISA Verdicts\n\n")
        f.write(
            "| Backend | Status | Evidence | Runtime |\n| --- | --- | --- | --- |\n"
        )
        for name in ("cpu", "gpu", "npu"):
            result = results[name]
            f.write(
                f"| {name.upper()} | {result.status} | {result.evidence} | "
                f"{result.runtime} |\n"
            )
        f.write("\n")

        f.write("## Performance\n\n")
        f.write("| Backend | Timing domain | Count | Latency | Throughput | Notes |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for name in ("cpu", "gpu", "npu"):
            result = results[name]
            f.write(
                f"| {name.upper()} | {result.perf_domain} | {result.perf_count} | "
                f"{result.perf_latency} | {result.perf_throughput} | "
                f"{result.perf_notes} |\n"
            )
        f.write("\n")

        f.write("## Source Paths\n\n")
        for name in ("cpu", "gpu", "npu"):
            result = results[name]
            f.write(f"- {name.upper()}: {report_path_list(result.sources, repo)}\n")
        f.write("\n")

        f.write("## Artifact Paths\n\n")
        for name in ("cpu", "gpu", "npu"):
            result = results[name]
            f.write(
                f"- {name.upper()}: build `{result.build_dir}`, "
                f"artifacts `{result.artifacts_dir}`\n"
            )
        f.write("\n")

        f.write("## Logs\n\n")
        for name in ("cpu", "gpu", "npu"):
            result = results[name]
            if not result.logs:
                f.write(f"- {name.upper()}: n/a\n")
                continue
            log_entries = ", ".join(
                f"{stem} `{path}`" for stem, path in sorted(result.logs.items())
            )
            f.write(f"- {name.upper()}: {log_entries}\n")
        f.write("\n")

        f.write("## CPU Disassembly Preview\n\n```asm\n")
        f.write(preview(cpu_disasm, 60))
        f.write("\n```\n\n")
        f.write("## GPU Summary Preview\n\n```text\n")
        f.write(preview(gpu_summary, 80))
        f.write("\n```\n\n")
        f.write("## NPU Disassembly Preview\n\n```asm\n")
        if first_npu_disasm:
            f.write(preview(first_npu_disasm, 80))
        else:
            f.write("No NPU core disassembly available.")
        f.write("\n```\n")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build, inspect, and optionally run the shared int8 GEMM benchmark."
    )
    parser.add_argument(
        "--backend",
        choices=["all", "cpu", "gpu", "npu"],
        default="all",
        help="backend to process (default: all)",
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True, help="report/artifact root"
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=None,
        help="shared build root; backend subdirectories are created below it",
    )
    parser.add_argument("--run", action="store_true", help="execute selected kernels")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when any selected backend is not PASS",
    )
    parser.add_argument(
        "--gpu-arch",
        default=os.environ.get("AIR_GPU_CHIP", "gfx1150"),
        help="AMDGPU chip for GPU lowering (default: AIR_GPU_CHIP or gfx1150)",
    )
    parser.add_argument(
        "--warmups",
        type=nonnegative_int,
        default=10,
        help="warmup iterations for every backend (default: 10)",
    )
    parser.add_argument(
        "--iterations",
        type=positive_int,
        default=20,
        help="timed iterations for every backend (default: 20)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    out_dir = args.out_dir.resolve()
    build_root = args.build_dir.resolve() if args.build_dir else out_dir / "build"
    logs_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    build_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    factories: dict[str, Callable[..., BackendAdapter]] = {
        "cpu": CpuAdapter,
        "gpu": GpuAdapter,
        "npu": NpuAdapter,
    }
    selected = set(selected_backends(args.backend))
    results: dict[str, BackendResult] = {
        name: make_skipped(name, build_root, out_dir) for name in ("cpu", "gpu", "npu")
    }

    for name in ("cpu", "gpu", "npu"):
        if name not in selected:
            continue
        adapter = factories[name](
            repo,
            out_dir,
            build_root,
            logs_dir,
            args.warmups,
            args.iterations,
            args.gpu_arch,
            args.run,
        )
        results[name] = adapter.execute()

    report = out_dir / "gemm_int8_report.md"
    write_report(report, repo, out_dir, build_root, args, results)
    print(f"Report: {report}")

    if args.strict:
        for name in selected:
            if results[name].status not in {"PASS", "SKIP"}:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
