# SPDX-License-Identifier: MIT

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from llm_linear.manifest import load_json
from llm_linear.quantization import (
    decode_gemv_fused_dequant,
    dequantize_packed_weights,
    metadata_for_packed_weights,
    pack_4bit,
    quantize_weight_matrix,
    unpack_4bit,
)
from llm_linear.reference import (
    LinearConfig,
    decode_gemv,
    random_inputs,
    random_weights,
    run_reference,
    stage_metrics,
    workload_bytes,
)
from llm_linear.schema import (
    case_stage_backends,
    contains_npu,
    required_backends,
    validate_case,
    validate_manifest,
)
from llm_linear.suites import SHAPE_LADDER, suite_workloads


def _linear_manifest(moe_dir: Path) -> dict:
    return load_json(moe_dir / "llm_linear" / "default_linear_manifest.json")


def _linear_matrix(moe_dir: Path) -> dict:
    return load_json(moe_dir / "llm_linear" / "default_linear_matrix.json")


def test_linear_manifest_matrix_and_required_backends(moe_dir: Path) -> None:
    manifest = _linear_manifest(moe_dir)
    matrix = _linear_matrix(moe_dir)
    assert validate_manifest(manifest) is manifest
    names = {case["name"] for case in matrix["cases"]}
    assert {
        "cpu_only",
        "gpu_only",
        "npu_only",
        "gpu_prefill_npu_decode_host",
        "npu_prefill_gpu_decode_host",
        "gpu_prefill_npu_decode_direct",
        "npu_prefill_gpu_decode_direct",
    } <= names

    cpu_case = matrix["cases"][0]
    mixed_case = matrix["cases"][3]
    assert validate_case(cpu_case) is cpu_case
    assert case_stage_backends(mixed_case) == {"prefill": "gpu", "decode": "npu"}
    assert contains_npu(case_stage_backends(mixed_case))
    assert required_backends([cpu_case, mixed_case], allow_npu=False) == {"gpu"}
    assert required_backends([cpu_case, mixed_case], allow_npu=True) == {
        "gpu",
        "npu",
    }

    bad = copy.deepcopy(manifest)
    bad["model"]["K"] = 0
    with pytest.raises(ValueError, match="K must be a positive integer"):
        validate_manifest(bad)


def test_linear_reference_math_and_bytes() -> None:
    cfg = LinearConfig(M=2, K=4, H=3, N=5, dtype="f16", shape_tier="tiny_ci")
    inputs = random_inputs(cfg, seed=1, scale=0.25)
    weights = random_weights(cfg, seed=2, scale=0.125)
    reference = run_reference(cfg, inputs, weights)
    assert reference["prefill"].shape == (2, 3)
    assert reference["decode_input"].shape == (3,)
    assert reference["output"].shape == (5,)

    metrics = stage_metrics(reference, reference, cfg.dtype)
    assert all(metric["allclose"] for metric in metrics.values())
    assert workload_bytes(cfg)["prefill_weights"] == 4 * 3 * 2


def test_linear_suite_ladder(moe_dir: Path) -> None:
    manifest = _linear_manifest(moe_dir)
    matrix = _linear_matrix(moe_dir)
    workloads = suite_workloads(["tiny_ci", "medium"], manifest, matrix)
    assert len(workloads) == len(SHAPE_LADDER["tiny_ci"]) + len(SHAPE_LADDER["medium"])
    assert workloads[0]["suite"] == "tiny_ci"
    assert workloads[0]["manifest"]["model"]["shape_tier"] == "tiny_ci"
    assert "artifacts/tiny_ci" in workloads[0]["manifest"]["paths"]["artifacts"]


def test_linear_quantization_pack_metadata() -> None:
    values = np.asarray([0, 1, 15, 7, 3], dtype=np.uint8)
    packed = pack_4bit(values, signed=False)
    assert packed.tolist() == [0x10, 0x7F, 0x03]
    assert (
        unpack_4bit(packed, count=values.size, signed=False).tolist() == values.tolist()
    )

    signed = np.asarray([-8, -1, 0, 7], dtype=np.int8)
    signed_packed = pack_4bit(signed, signed=True)
    assert (
        unpack_4bit(signed_packed, count=signed.size, signed=True).tolist()
        == signed.tolist()
    )

    metadata = metadata_for_packed_weights(
        quant_kind="int4", shape=(8, 16), block_size=32, quant_axis=1
    )
    assert metadata.signed is True
    assert metadata.packing == "two_values_per_byte_low_nibble_first"


def test_linear_quantized_decode_matches_dequantized_baseline() -> None:
    rng = np.random.default_rng(3)
    vector = rng.standard_normal(8, dtype=np.float32) * np.float32(0.25)
    weights = rng.standard_normal((8, 6), dtype=np.float32) * np.float32(0.125)
    packed = quantize_weight_matrix(
        weights, quant_kind="int4", block_size=4, quant_axis=0
    )
    dequantized = dequantize_packed_weights(packed, dtype_name="f16")
    fused, detail = decode_gemv_fused_dequant(vector, packed, "f16")
    baseline = decode_gemv(vector, dequantized, "f16")
    np.testing.assert_allclose(fused, baseline)
    assert detail["packed_weight_bytes_read"] == packed.packed.nbytes
    assert packed.descriptor()["quant_kind"] == "int4"
