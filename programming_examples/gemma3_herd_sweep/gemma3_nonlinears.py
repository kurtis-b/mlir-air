# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gemma3 nonlinear implementation registry and CPU contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from ml_dtypes import bfloat16

from gemma3_config import Gemma3TextConfig, synthetic_text_config
from gemma3_reference import apply_rope_halfsplit, geglu, qk_norm, rms_norm


_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Gemma3NonlinearSpec:
    operation: str
    model_status: str
    implementation_path: str
    compile_lit: str | None
    cpu_reference: str
    tensor_contract: str
    compile_status: str
    hardware_status: str
    tolerance: str
    timed_window_status: str
    fallback: str
    notes: str

    @property
    def implementation_exists(self) -> bool:
        return (_REPO_ROOT / self.implementation_path).exists()

    @property
    def compile_lit_exists(self) -> bool:
        if self.compile_lit is None:
            return True
        return (_REPO_ROOT / self.compile_lit).exists()

    @property
    def blocks_paper_match(self) -> bool:
        return self.timed_window_status == "unmeasured-host-fallback"

    def format(self) -> str:
        lit = self.compile_lit if self.compile_lit else "none"
        return (
            f"nonlinear {self.operation} status={self.model_status} "
            f"source={self.implementation_path} compile_lit={lit} "
            f"cpu_ref={self.cpu_reference} compile={self.compile_status} "
            f"hardware={self.hardware_status} tolerance={self.tolerance} "
            f"timed_window={self.timed_window_status} fallback={self.fallback} "
            f"contract={self.tensor_contract}"
        )


def nonlinear_registry(config: Gemma3TextConfig | None = None) -> tuple[Gemma3NonlinearSpec, ...]:
    config = config or synthetic_text_config()
    return (
        Gemma3NonlinearSpec(
            operation="rms_norm",
            model_status="standalone-npu-reuse-candidate",
            implementation_path="programming_examples/weighted_rms_norm/weighted_rms_norm.py",
            compile_lit="programming_examples/weighted_rms_norm/run_makefile_peano.lit",
            cpu_reference="gemma3_reference.rms_norm",
            tensor_contract=f"flatten [*, {config.emb_dim}] rows to weighted_rms_norm[M,N]",
            compile_status="compile-lit-available",
            hardware_status="hardware-smoke-pass-M8-N1152-N2560-elf",
            tolerance="rtol=5e-2 atol=5e-1 corr>=0.99 hardware-smoke",
            timed_window_status="unmeasured-host-fallback",
            fallback="host CPU reference until Gemma model wiring uses the validated standalone kernel",
            notes="Standalone weighted_rms_norm ELF compile-and-run passed for M=8 N=1152 and M=8 N=2560.",
        ),
        Gemma3NonlinearSpec(
            operation="qk_norm",
            model_status="standalone-npu-reuse-candidate",
            implementation_path="programming_examples/weighted_rms_norm/weighted_rms_norm.py",
            compile_lit="programming_examples/weighted_rms_norm/run_makefile_peano.lit",
            cpu_reference="gemma3_reference.qk_norm",
            tensor_contract=(
                f"flatten [tokens, heads, {config.head_dim}] to "
                f"[tokens * heads, {config.head_dim}]"
            ),
            compile_status="compile-lit-available",
            hardware_status="hardware-smoke-pass-M32-N256-elf",
            tolerance="rtol=5e-2 atol=5e-1 corr>=0.99 hardware-smoke",
            timed_window_status="unmeasured-host-fallback",
            fallback="host CPU reference until Gemma model wiring uses the validated per-head kernel",
            notes="Standalone weighted_rms_norm ELF compile-and-run passed for flattened per-head M=32 N=256.",
        ),
        Gemma3NonlinearSpec(
            operation="rope",
            model_status="kernel-source-reuse-candidate",
            implementation_path="programming_examples/llama32_1b/kernel_builder/rope_halfsplit.cc",
            compile_lit=None,
            cpu_reference="gemma3_reference.apply_rope_halfsplit",
            tensor_contract=f"half-split LUT [cos..., sin...] over head_dim={config.head_dim}",
            compile_status="source-only-no-gemma-air-wrapper",
            hardware_status="pending-gemma-air-wrapper-hardware",
            tolerance="identity-lut-cpu-checksum-only",
            timed_window_status="unmeasured-host-fallback",
            fallback="host CPU reference until a Gemma AIR wrapper is added",
            notes="The standalone rope_sincos example is head_size=48/even-odd and is not used.",
        ),
        Gemma3NonlinearSpec(
            operation="mlp_activation",
            model_status="gemma-specific-compile-only-candidate",
            implementation_path="programming_examples/gemma3_dataflow_kernels/geglu.py",
            compile_lit="programming_examples/gemma3_dataflow_kernels/run_geglu_compile_only.lit",
            cpu_reference="gemma3_reference.geglu",
            tensor_contract=f"GeGLU gate/up vectors of hidden_dim={config.hidden_dim}",
            compile_status="compile-lit-available",
            hardware_status="hardware-smoke-pass-n1024-tile256-elf",
            tolerance="rtol=1e-1 atol=5e-2 hardware-smoke",
            timed_window_status="unmeasured-host-fallback",
            fallback="host CPU reference until Gemma model wiring uses the validated standalone kernel",
            notes="Standalone ELF compile-and-run passed for n=1024 tile_n=256 with Peano/XRT.",
        ),
        Gemma3NonlinearSpec(
            operation="residual_add",
            model_status="host-fallback",
            implementation_path="programming_examples/llama32_1b/multi_launch_builder/o_ffn_multi.py",
            compile_lit=None,
            cpu_reference="gemma3_reference residual add sites",
            tensor_contract=f"elementwise add over emb_dim={config.emb_dim}",
            compile_status="not-promoted",
            hardware_status="not-promoted",
            tolerance="exact-bf16-cpu-contract",
            timed_window_status="unmeasured-host-fallback",
            fallback="host CPU reference until fused layer builders are introduced",
            notes="Reuse the Llama multi-launch placement pattern, not its model math.",
        ),
    )


def paper_match_blockers(config: Gemma3TextConfig | None = None) -> tuple[Gemma3NonlinearSpec, ...]:
    return tuple(spec for spec in nonlinear_registry(config) if spec.blocks_paper_match)


def validate_registry_paths(config: Gemma3TextConfig | None = None) -> None:
    missing = []
    for spec in nonlinear_registry(config):
        if not spec.implementation_exists:
            missing.append(spec.implementation_path)
        if not spec.compile_lit_exists:
            missing.append(spec.compile_lit)
    if missing:
        raise FileNotFoundError(f"missing Gemma3 nonlinear implementation paths: {missing}")


def validate_cpu_contracts(config: Gemma3TextConfig | None = None) -> dict[str, float]:
    config = config or synthetic_text_config()
    rng = np.random.default_rng(123)
    x = rng.normal(size=(2, config.emb_dim)).astype(np.float32).astype(bfloat16)
    row_weight = np.linspace(0.5, 1.5, config.emb_dim, dtype=np.float32).astype(bfloat16)
    rms = rms_norm(x, row_weight)

    q = rng.normal(size=(2, config.n_heads, config.head_dim)).astype(np.float32).astype(bfloat16)
    head_weight = np.linspace(0.75, 1.25, config.head_dim, dtype=np.float32).astype(bfloat16)
    q_normed = qk_norm(q, head_weight)
    rope_lut = np.tile(
        np.concatenate(
            [
                np.ones(config.head_dim // 2, dtype=np.float32),
                np.zeros(config.head_dim // 2, dtype=np.float32),
            ]
        ),
        (2, 1),
    ).astype(bfloat16)
    roped = apply_rope_halfsplit(q_normed, rope_lut[:, None, :])

    gate = rng.normal(size=(2, config.hidden_dim)).astype(np.float32).astype(bfloat16)
    up = rng.normal(size=(2, config.hidden_dim)).astype(np.float32).astype(bfloat16)
    activated = geglu(gate, up)
    residual = (x.astype(np.float32) + rms.astype(np.float32)).astype(bfloat16)

    if rms.shape != x.shape:
        raise AssertionError(f"rms_norm shape mismatch: {rms.shape}")
    if q_normed.shape != q.shape:
        raise AssertionError(f"qk_norm shape mismatch: {q_normed.shape}")
    if roped.shape != q.shape:
        raise AssertionError(f"rope shape mismatch: {roped.shape}")
    if activated.shape != gate.shape:
        raise AssertionError(f"mlp activation shape mismatch: {activated.shape}")
    if residual.shape != x.shape:
        raise AssertionError(f"residual shape mismatch: {residual.shape}")

    return {
        "rms_norm": float(np.sum(rms.astype(np.float32))),
        "qk_norm": float(np.sum(q_normed.astype(np.float32))),
        "rope": float(np.sum(roped.astype(np.float32))),
        "mlp_activation": float(np.sum(activated.astype(np.float32))),
        "residual_add": float(np.sum(residual.astype(np.float32))),
    }


def measure_cpu_contracts(
    config: Gemma3TextConfig | None = None,
    *,
    timed_iters: int = 3,
) -> dict[str, dict[str, Any]]:
    if timed_iters <= 0:
        raise ValueError("timed_iters must be positive")
    config = config or synthetic_text_config()
    rng = np.random.default_rng(123)
    x = rng.normal(size=(2, config.emb_dim)).astype(np.float32).astype(bfloat16)
    row_weight = np.linspace(0.5, 1.5, config.emb_dim, dtype=np.float32).astype(bfloat16)
    q = rng.normal(size=(2, config.n_heads, config.head_dim)).astype(np.float32).astype(bfloat16)
    head_weight = np.linspace(0.75, 1.25, config.head_dim, dtype=np.float32).astype(bfloat16)
    rope_lut = np.tile(
        np.concatenate(
            [
                np.ones(config.head_dim // 2, dtype=np.float32),
                np.zeros(config.head_dim // 2, dtype=np.float32),
            ]
        ),
        (2, 1),
    ).astype(bfloat16)
    gate = rng.normal(size=(2, config.hidden_dim)).astype(np.float32).astype(bfloat16)
    up = rng.normal(size=(2, config.hidden_dim)).astype(np.float32).astype(bfloat16)
    residual_rhs = rms_norm(x, row_weight)

    def residual_add() -> np.ndarray:
        return (x.astype(np.float32) + residual_rhs.astype(np.float32)).astype(bfloat16)

    operations = {
        "rms_norm": lambda: rms_norm(x, row_weight),
        "qk_norm": lambda: qk_norm(q, head_weight),
        "rope": lambda: apply_rope_halfsplit(q, rope_lut[:, None, :]),
        "mlp_activation": lambda: geglu(gate, up),
        "residual_add": residual_add,
    }
    measurements: dict[str, dict[str, Any]] = {}
    for name, operation in operations.items():
        output = operation()
        start = perf_counter()
        for _ in range(timed_iters):
            output = operation()
        elapsed_ms = (perf_counter() - start) * 1000.0 / float(timed_iters)
        measurements[name] = {
            "elapsed_ms": elapsed_ms,
            "timed_iters": timed_iters,
            "shape": tuple(int(dim) for dim in output.shape),
            "checksum": float(np.sum(output.astype(np.float32))),
            "measurement_source": "synthetic-cpu-contract-microbenchmark",
        }
    return measurements


def format_registry(config: Gemma3TextConfig | None = None) -> str:
    return "\n".join(spec.format() for spec in nonlinear_registry(config))


def _self_test() -> None:
    config = synthetic_text_config(n_layers=2)
    validate_registry_paths(config)
    checksums = validate_cpu_contracts(config)
    measurements = measure_cpu_contracts(config, timed_iters=1)
    print(format_registry(config))
    for name, checksum in checksums.items():
        print(f"{name}_checksum={checksum:.6f}")
    for name, measurement in measurements.items():
        print(
            f"{name}_fallback_measurement=measured-host-fallback "
            f"elapsed_ms={measurement['elapsed_ms']:.6f} "
            f"source={measurement['measurement_source']}"
        )
    blockers = paper_match_blockers(config)
    print(f"nonlinear_paper_gate=PAPER_MATCH_BLOCKED unresolved={len(blockers)}")
    for blocker in blockers:
        print(f"nonlinear_blocker operation={blocker.operation} timed_window={blocker.timed_window_status}")
    print("GEMMA3_NONLINEAR_REGISTRY_SELF_TEST: PASS")


if __name__ == "__main__":
    _self_test()
