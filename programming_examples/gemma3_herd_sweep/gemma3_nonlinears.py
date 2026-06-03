# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gemma3 nonlinear implementation registry and CPU contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    tensor_contract: str
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

    def format(self) -> str:
        lit = self.compile_lit if self.compile_lit else "none"
        return (
            f"nonlinear {self.operation} status={self.model_status} "
            f"source={self.implementation_path} compile_lit={lit} "
            f"fallback={self.fallback} contract={self.tensor_contract}"
        )


def nonlinear_registry(config: Gemma3TextConfig | None = None) -> tuple[Gemma3NonlinearSpec, ...]:
    config = config or synthetic_text_config()
    return (
        Gemma3NonlinearSpec(
            operation="rms_norm",
            model_status="standalone-npu-reuse-candidate",
            implementation_path="programming_examples/weighted_rms_norm/weighted_rms_norm.py",
            compile_lit="programming_examples/weighted_rms_norm/run_makefile_peano.lit",
            tensor_contract=f"flatten [*, {config.emb_dim}] rows to weighted_rms_norm[M,N]",
            fallback="host CPU reference until Gemma launch wiring is validated",
            notes="Weighted RMSNorm math matches Gemma row RMSNorm.",
        ),
        Gemma3NonlinearSpec(
            operation="qk_norm",
            model_status="standalone-npu-reuse-candidate",
            implementation_path="programming_examples/weighted_rms_norm/weighted_rms_norm.py",
            compile_lit="programming_examples/weighted_rms_norm/run_makefile_peano.lit",
            tensor_contract=(
                f"flatten [tokens, heads, {config.head_dim}] to "
                f"[tokens * heads, {config.head_dim}]"
            ),
            fallback="host CPU reference until per-head BO layout is validated",
            notes="Same RMS math, with per-head weights and no cross-head reduction.",
        ),
        Gemma3NonlinearSpec(
            operation="rope",
            model_status="kernel-source-reuse-candidate",
            implementation_path="programming_examples/llama32_1b/kernel_builder/rope_halfsplit.cc",
            compile_lit=None,
            tensor_contract=f"half-split LUT [cos..., sin...] over head_dim={config.head_dim}",
            fallback="host CPU reference until a Gemma AIR wrapper is added",
            notes="The standalone rope_sincos example is head_size=48/even-odd and is not used.",
        ),
        Gemma3NonlinearSpec(
            operation="mlp_activation",
            model_status="gemma-specific-compile-only-candidate",
            implementation_path="programming_examples/gemma3_dataflow_kernels/geglu.py",
            compile_lit="programming_examples/gemma3_dataflow_kernels/run_geglu_compile_only.lit",
            tensor_contract=f"GeGLU gate/up vectors of hidden_dim={config.hidden_dim}",
            fallback="host CPU reference until hardware validation is recorded",
            notes="Existing ffn_swiglu uses SiLU, so Gemma GeGLU needs its own kernel.",
        ),
        Gemma3NonlinearSpec(
            operation="residual_add",
            model_status="host-fallback",
            implementation_path="programming_examples/llama32_1b/multi_launch_builder/o_ffn_multi.py",
            compile_lit=None,
            tensor_contract=f"elementwise add over emb_dim={config.emb_dim}",
            fallback="host CPU reference until fused layer builders are introduced",
            notes="Reuse the Llama multi-launch placement pattern, not its model math.",
        ),
    )


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


def format_registry(config: Gemma3TextConfig | None = None) -> str:
    return "\n".join(spec.format() for spec in nonlinear_registry(config))


def _self_test() -> None:
    config = synthetic_text_config(n_layers=2)
    validate_registry_paths(config)
    checksums = validate_cpu_contracts(config)
    print(format_registry(config))
    for name, checksum in checksums.items():
        print(f"{name}_checksum={checksum:.6f}")
    print("GEMMA3_NONLINEAR_REGISTRY_SELF_TEST: PASS")


if __name__ == "__main__":
    _self_test()
