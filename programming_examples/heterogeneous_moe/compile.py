# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from kernels import KernelConfig, default_air_filenames, write_default_air_sources, write_split_expert_air_sources
from manifest import artifact_root, generated_air_source_root


def _require_tool(name: str) -> str:
    exe = shutil.which(name)
    if not exe:
        raise RuntimeError(f"Required tool '{name}' is not on PATH")
    return exe


def _tool_from_env_or_path(env_var: str, tool_name: str) -> str:
    env_path = os.environ.get(env_var)
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if not path.exists():
            raise RuntimeError(f"{env_var} is set but does not exist: {path}")
        return str(path)
    return _require_tool(tool_name)


def _aircc_tool() -> str:
    return _tool_from_env_or_path("AIRCC_PATH", "aircc")


def _air_opt_tool() -> str:
    return _tool_from_env_or_path("AIR_OPT_PATH", "air-opt")


def _kernel_config(manifest: dict[str, Any]) -> KernelConfig:
    return KernelConfig(
        batch_tokens=manifest["model"]["batch_tokens"],
        hidden_size=manifest["model"]["hidden_size"],
        ffn_size=manifest["model"]["ffn_size"],
        dtype=manifest["model"]["dtype"],
    )


def resolve_air_sources(manifest: dict[str, Any], backend: str) -> dict[str, Path]:
    if backend not in {"gpu", "npu"}:
        raise ValueError(f"Unsupported source backend: {backend}")
    cfg = _kernel_config(manifest)
    source_dir = generated_air_source_root(manifest)
    names = default_air_filenames(cfg)
    paths = {key: source_dir / name for key, name in names.items()}
    if not all(path.exists() for path in paths.values()):
        write_default_air_sources(cfg, source_dir)
    return paths


def _llvm_lib_dir() -> Path:
    env = os.environ.get("LLVM_INSTALL_DIR")
    if env:
        return Path(env) / "lib"
    mlir_opt = shutil.which("mlir-opt")
    if not mlir_opt:
        raise RuntimeError("LLVM_INSTALL_DIR is unset and mlir-opt is not on PATH")
    return Path(mlir_opt).resolve().parent.parent / "lib"


def _llvm_bin_dir() -> Path:
    env = os.environ.get("LLVM_INSTALL_DIR")
    if env:
        return Path(env) / "bin"
    mlir_translate = shutil.which("mlir-translate")
    if not mlir_translate:
        raise RuntimeError("LLVM_INSTALL_DIR is unset and mlir-translate is not on PATH")
    return Path(mlir_translate).resolve().parent


def _llvm_clang() -> str:
    clang = _llvm_bin_dir() / "clang"
    if clang.exists():
        return str(clang)
    raise RuntimeError(
        "Could not find LLVM clang next to mlir-translate. "
        "Set LLVM_INSTALL_DIR to the local LLVM build used for GPU compilation."
    )


def _llvm_mlir_opt() -> str:
    mlir_opt = _llvm_bin_dir() / "mlir-opt"
    if mlir_opt.exists():
        return str(mlir_opt)
    raise RuntimeError("Could not find mlir-opt in LLVM_INSTALL_DIR/bin")


def _llvm_mlir_translate() -> str:
    mlir_translate = _llvm_bin_dir() / "mlir-translate"
    if mlir_translate.exists():
        return str(mlir_translate)
    raise RuntimeError("Could not find mlir-translate in LLVM_INSTALL_DIR/bin")


def default_gpu_shared_libs() -> list[str]:
    llvm_lib_dir = _llvm_lib_dir()
    libs = [
        str(llvm_lib_dir / "libmlir_rocm_runtime.so"),
        str(llvm_lib_dir / "libmlir_runner_utils.so"),
        str(llvm_lib_dir / "libmlir_c_runner_utils.so"),
    ]
    rocm_root = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    hip = rocm_root / "lib" / "libamdhip64.so"
    if hip.exists():
        libs.append(str(hip))
    return libs


def _sanitize_gpu_llvm_ir(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    filtered = [line for line in lines if not line.startswith("@llvm.global_dtors =")]
    if filtered != lines:
        path.write_text("\n".join(filtered) + "\n", encoding="utf-8")


def compile_npu(source: Path, output_dir: Path, device: str) -> dict[str, str]:
    return compile_npu_with_args(source, output_dir, device, [])


def compile_npu_with_args(source: Path, output_dir: Path, device: str, extra_args: list[str]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    xclbin = output_dir / f"{source.stem}.{device}.xclbin"
    insts = output_dir / f"{source.stem}.{device}.insts.bin"
    cmd = [
        _aircc_tool(),
        *extra_args,
        "--device",
        device,
        str(source),
        "-o",
        str(xclbin),
        "-i",
        str(insts),
    ]
    subprocess.run(cmd, check=True)
    return {"xclbin": str(xclbin), "insts": str(insts)}


def compile_npu_if_needed(source: Path, output_dir: Path, device: str) -> dict[str, str]:
    return compile_npu_if_needed_with_args(source, output_dir, device, [])


def compile_npu_if_needed_with_args(source: Path, output_dir: Path, device: str, extra_args: list[str]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    xclbin = output_dir / f"{source.stem}.{device}.xclbin"
    insts = output_dir / f"{source.stem}.{device}.insts.bin"
    if (
        xclbin.exists()
        and insts.exists()
        and xclbin.stat().st_mtime >= source.stat().st_mtime
        and insts.stat().st_mtime >= source.stat().st_mtime
    ):
        return {"xclbin": str(xclbin), "insts": str(insts)}
    return compile_npu_with_args(source, output_dir, device, extra_args)


def _divisors_at_most(value: int, limit: int) -> list[int]:
    return [candidate for candidate in range(min(value, limit), 0, -1) if value % candidate == 0]


def compile_npu_tiled_expert(cfg: KernelConfig, source_dir: Path, output_dir: Path, device: str) -> dict[str, Any]:
    compile_args = ["--omit-ping-pong-transform"]
    best_error: list[str] = []
    for ffn_tile in _divisors_at_most(cfg.ffn_size, 128):
        hidden_cfg = KernelConfig(
            batch_tokens=cfg.batch_tokens,
            hidden_size=cfg.hidden_size,
            ffn_size=ffn_tile,
            dtype=cfg.dtype,
        )
        hidden_sources = write_split_expert_air_sources(hidden_cfg, source_dir)
        try:
            hidden_artifact = compile_npu_if_needed_with_args(
                hidden_sources["expert_hidden"],
                output_dir,
                device,
                compile_args,
            )
        except subprocess.CalledProcessError as exc:
            best_error = (exc.stdout or "").splitlines()[-10:]
            continue

        for output_tile in _divisors_at_most(cfg.hidden_size, 128):
            output_cfg = KernelConfig(
                batch_tokens=cfg.batch_tokens,
                hidden_size=output_tile,
                ffn_size=ffn_tile,
                dtype=cfg.dtype,
            )
            output_sources = write_split_expert_air_sources(output_cfg, source_dir)
            try:
                output_artifact = compile_npu_if_needed_with_args(
                    output_sources["expert_output"],
                    output_dir,
                    device,
                    compile_args,
                )
            except subprocess.CalledProcessError as exc:
                best_error = (exc.stdout or "").splitlines()[-10:]
                continue

            return {
                "mode": "tiled_split",
                "compile_args": compile_args,
                "tiling": {
                    "ffn_tile": ffn_tile,
                    "output_tile": output_tile,
                    "ffn_tiles": cfg.ffn_size // ffn_tile,
                    "output_tiles": cfg.hidden_size // output_tile,
                },
                "hidden": {
                    "source": str(hidden_sources["expert_hidden"]),
                    "artifact": hidden_artifact,
                },
                "output": {
                    "source": str(output_sources["expert_output"]),
                    "artifact": output_artifact,
                },
            }

    error_text = "\n".join(best_error) if best_error else "No candidate tile shapes compiled successfully."
    raise RuntimeError(
        f"Failed to compile tiled NPU expert for {cfg.batch_tokens}x{cfg.hidden_size}x{cfg.ffn_size} {cfg.dtype}.\n"
        f"{error_text}"
    )


def compile_npu_parallel_expert(cfg: KernelConfig, source_dir: Path, output_dir: Path, device: str) -> dict[str, Any]:
    compile_args = ["--omit-ping-pong-transform"]
    split_sources = write_split_expert_air_sources(cfg, source_dir)
    try:
        hidden_artifact = compile_npu_if_needed_with_args(
            split_sources["expert_hidden"],
            output_dir,
            device,
            compile_args,
        )
        output_artifact = compile_npu_if_needed_with_args(
            split_sources["expert_output"],
            output_dir,
            device,
            compile_args,
        )
        return {
            "mode": "parallel_split",
            "compile_args": compile_args,
            "hidden": {
                "source": str(split_sources["expert_hidden"]),
                "artifact": hidden_artifact,
            },
            "output": {
                "source": str(split_sources["expert_output"]),
                "artifact": output_artifact,
            },
        }
    except subprocess.CalledProcessError:
        return compile_npu_tiled_expert(cfg, source_dir, output_dir, device)


def compile_gpu(source: Path, output_dir: Path, arch: str, entrypoint: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outlined_mlir = output_dir / f"{source.stem}.{arch}.outlined.mlir"
    output_mlir = output_dir / f"{source.stem}.{arch}.gpu.mlir"
    output_llvm = output_dir / f"{source.stem}.{arch}.ll"
    output_so = output_dir / f"{source.stem}.{arch}.so"
    outline_cmd = [
        _air_opt_tool(),
        str(source),
        "-air-to-rocdl",
        "-air-gpu-outlining",
        "-air-gpu-host-staging",
        "-o",
        str(outlined_mlir),
    ]
    subprocess.run(outline_cmd, check=True)
    pipeline = (
        f"builtin.module("
        f"rocdl-attach-target{{chip={arch} O=3}},"
        f"gpu.module(convert-gpu-to-rocdl{{chipset={arch} runtime=HIP}},reconcile-unrealized-casts),"
        f"gpu-module-to-binary,"
        f"func.func(gpu-async-region),"
        f"gpu-to-llvm,"
        f"convert-to-llvm,"
        f"reconcile-unrealized-casts)"
    )
    lower_cmd = [
        _llvm_mlir_opt(),
        str(outlined_mlir),
        f"--pass-pipeline={pipeline}",
        "-o",
        str(output_mlir),
    ]
    subprocess.run(lower_cmd, check=True)
    translate_cmd = [
        _llvm_mlir_translate(),
        "--mlir-to-llvmir",
        str(output_mlir),
        "-o",
        str(output_llvm),
    ]
    subprocess.run(translate_cmd, check=True)
    _sanitize_gpu_llvm_ir(output_llvm)

    shared_libs = default_gpu_shared_libs()
    rpaths = [f"-Wl,-rpath,{Path(lib).parent}" for lib in shared_libs]
    clang_cmd = [
        _llvm_clang(),
        "-shared",
        "-fPIC",
        "-Wno-override-module",
        str(output_llvm),
        "-o",
        str(output_so),
        *shared_libs,
        *rpaths,
    ]
    subprocess.run(clang_cmd, check=True)
    return {"mlir": str(output_mlir), "llvm": str(output_llvm), "so": str(output_so), "entry": entrypoint}


def compile_gpu_parallel_expert(cfg: KernelConfig, source_dir: Path, output_dir: Path, arch: str) -> dict[str, Any]:
    split_sources = write_split_expert_air_sources(cfg, source_dir)
    hidden_artifact = compile_gpu(
        split_sources["expert_hidden"],
        output_dir,
        arch,
        "expert_hidden",
    )
    output_artifact = compile_gpu(
        split_sources["expert_output"],
        output_dir,
        arch,
        "expert_output",
    )
    return {
        "mode": "parallel_split",
        "hidden": {
            "source": str(split_sources["expert_hidden"]),
            "artifact": hidden_artifact,
        },
        "output": {
            "source": str(split_sources["expert_output"]),
            "artifact": output_artifact,
        },
    }


def populate_artifacts(manifest: dict[str, Any], backends: set[str]) -> dict[str, Any]:
    cfg = _kernel_config(manifest)
    air_sources = write_default_air_sources(cfg, generated_air_source_root(manifest))
    artifact_dir = artifact_root(manifest)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    compiler_cfg = manifest["compiler"]
    gpu_entries = {
        "router": "router_math",
        "expert": "expert_mlp",
        "aggregation": "aggregate_outputs",
    }
    for key, source in air_sources.items():
        artifact_entry = manifest["artifacts"].setdefault(key, {})
        if "npu" in backends:
            if key == "expert":
                artifact_entry["npu"] = compile_npu_parallel_expert(
                    cfg,
                    source.parent,
                    artifact_dir / "npu",
                    compiler_cfg["npu_device"],
                )
            else:
                artifact_entry["npu"] = compile_npu(source, artifact_dir / "npu", compiler_cfg["npu_device"])
        if "gpu" in backends:
            if key == "expert":
                artifact_entry["gpu"] = compile_gpu_parallel_expert(
                    cfg,
                    source.parent,
                    artifact_dir / "gpu",
                    compiler_cfg["gpu_arch"],
                )
            else:
                artifact_entry["gpu"] = compile_gpu(
                    source,
                    artifact_dir / "gpu",
                    compiler_cfg["gpu_arch"],
                    gpu_entries[key],
                )
    return manifest
