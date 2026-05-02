# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
import pytest

import reference
from kernels import KernelConfig
from numerics import (
    array_error_metrics,
    bf16_bits_to_float32,
    decode_npu_array,
    encode_npu_array,
    encoded_array_summary,
    float32_to_bf16_bits,
    host_array_dtype,
    is_bf16_dtype,
    normalize_dtype_name,
    npu_buffer_dtype,
    quantize_array,
    quantize_scalar,
)
from reference import (
    aggregate_outputs,
    aggregate_packed_outputs,
    expert_mlp,
    optional_torch_validation,
    pack_expert_outputs,
    random_inputs,
    random_weights,
    routed_inputs,
    router_logits,
    run_reference,
    softmax_rows,
    topk_weights,
    validation_tolerances,
)


def test_dtype_aliases_and_bf16_roundtrip() -> None:
    values = np.asarray([1.0, -2.25, 3.125], dtype=np.float32)

    assert normalize_dtype_name("float16") == "f16"
    assert normalize_dtype_name("half") == "f16"
    assert normalize_dtype_name("bfloat16") == "bf16"
    assert is_bf16_dtype("BF16")
    assert host_array_dtype("bf16") == np.dtype(np.float32)
    assert host_array_dtype("f16") == np.dtype(np.float16)
    assert npu_buffer_dtype("bf16") == np.dtype(np.uint16)
    assert npu_buffer_dtype("f16") == np.dtype(np.float16)

    bits = float32_to_bf16_bits(values)
    decoded = bf16_bits_to_float32(bits)
    np.testing.assert_allclose(decoded, quantize_array(values, "bf16"))

    encoded = encode_npu_array(values, "bf16")
    assert encoded.dtype == np.uint16
    np.testing.assert_allclose(decode_npu_array(encoded, "bf16"), decoded)
    assert encode_npu_array(values, "f16").dtype == np.float16
    assert decode_npu_array(values.astype(np.float16), "f16").dtype == np.float16
    assert isinstance(quantize_scalar(1.2, "f16"), float)


@pytest.mark.parametrize("func", [quantize_array, encode_npu_array, decode_npu_array])
def test_unsupported_dtype_errors(func) -> None:
    with pytest.raises(ValueError, match="Unsupported dtype"):
        func(np.asarray([1.0], dtype=np.float32), "int8")


def test_array_error_metrics_and_encoded_summary() -> None:
    actual = np.asarray([[1.0, 2.0], [3.0, 4.1]], dtype=np.float32)
    expected = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    metrics = array_error_metrics(actual, expected, atol=0.11, rtol=0.0)

    assert metrics["allclose"] is True
    assert metrics["max_abs_error"] == pytest.approx(0.1, abs=1e-6)
    assert metrics["atol"] == 0.11
    summary = encoded_array_summary(actual[:, ::-1], "bf16")
    assert summary["shape"] == [2, 2]
    assert summary["encoded_dtype"] == "uint16"
    assert summary["elements"] == 4
    assert summary["nbytes"] == 8


def test_random_inputs_profiles_and_errors(small_cfg: KernelConfig) -> None:
    balanced = random_inputs(small_cfg, seed=1, routing_profile="balanced")
    expert0 = random_inputs(small_cfg, seed=1, routing_profile="expert0_hot")
    expert1 = random_inputs(small_cfg, seed=1, routing_profile="expert1_hot")
    alternating = random_inputs(small_cfg, seed=1, routing_profile="alternating")

    assert balanced.shape == (2, 4)
    assert np.all(expert0 >= 0)
    assert np.all(expert1 >= 0)
    assert np.all(alternating[0] >= 0)
    assert np.all(alternating[1] <= 0)
    with pytest.raises(ValueError, match="Unsupported routing profile"):
        random_inputs(small_cfg, seed=1, routing_profile="bad")


def test_random_weights_profiles_and_reference_math(small_cfg: KernelConfig) -> None:
    inputs = random_inputs(small_cfg, seed=2)
    weights = random_weights(small_cfg, seed=3, routing_profile="alternating")

    logits = router_logits(inputs, weights.router, small_cfg.dtype)
    probs = softmax_rows(logits, small_cfg.dtype)
    top2 = topk_weights(logits, "top2", small_cfg.dtype)
    top1 = topk_weights(logits, "top1", small_cfg.dtype)
    expert0_in, expert1_in = routed_inputs(inputs, top2, small_cfg.dtype)
    expert0_out = expert_mlp(
        expert0_in, weights.expert0_w1, weights.expert0_w2, small_cfg.dtype
    )
    expert1_out = expert_mlp(
        expert1_in, weights.expert1_w1, weights.expert1_w2, small_cfg.dtype
    )
    packed = pack_expert_outputs(expert0_out, expert1_out, small_cfg.dtype)

    assert logits.shape == (2, 2)
    np.testing.assert_allclose(
        probs.astype(np.float32).sum(axis=1), np.ones(2), atol=1e-3
    )
    np.testing.assert_allclose(top2, probs)
    np.testing.assert_allclose(
        top1.astype(np.float32).sum(axis=1), np.ones(2), atol=1e-3
    )
    np.testing.assert_allclose(
        aggregate_packed_outputs(packed, top2, small_cfg.dtype),
        aggregate_outputs(expert0_out, expert1_out, top2, small_cfg.dtype),
    )

    bundle = run_reference(small_cfg, inputs, weights, "top2")
    assert set(bundle) == {
        "logits",
        "weights",
        "expert0_input",
        "expert1_input",
        "expert0_output",
        "expert1_output",
        "packed_expert_outputs",
        "output",
    }
    with pytest.raises(ValueError, match="Unsupported router mode"):
        topk_weights(logits, "bad", small_cfg.dtype)
    with pytest.raises(ValueError, match="Unsupported routing profile"):
        random_weights(small_cfg, seed=3, routing_profile="bad")


def test_validation_tolerances_and_optional_torch_paths(
    monkeypatch, small_cfg: KernelConfig
) -> None:
    inputs = random_inputs(small_cfg, seed=4)
    weights = random_weights(small_cfg, seed=5)
    quantized = run_reference(small_cfg, inputs, weights, "top2")

    assert validation_tolerances("bf16")["atol"] > validation_tolerances("f16")["atol"]

    def missing_torch(*args, **kwargs):
        raise ImportError("no torch")

    monkeypatch.setattr(reference, "torch_reference", missing_torch)
    result = optional_torch_validation(inputs, weights, "top2", small_cfg.dtype)
    assert result == {"ran": False, "ok": False, "message": "torch not installed"}

    def fake_torch(*args, **kwargs):
        return {
            name: np.asarray(value, dtype=np.float32)
            for name, value in quantized.items()
        }

    monkeypatch.setattr(reference, "torch_reference", fake_torch)
    result = optional_torch_validation(
        inputs,
        weights,
        "top2",
        small_cfg.dtype,
        actual=quantized,
        quantized_reference=quantized,
    )
    assert result["ran"] is True
    assert result["ok"] is True
    assert result["actual_vs_torch"]["output"]["allclose"] is True

    bad_actual = {name: np.array(value, copy=True) for name, value in quantized.items()}
    bad_actual["output"] = bad_actual["output"] + np.float32(100.0)
    result = optional_torch_validation(
        inputs, weights, "top2", small_cfg.dtype, actual=bad_actual
    )
    assert result["ok"] is False
    assert result["message"] == "actual outputs differ from torch reference"
