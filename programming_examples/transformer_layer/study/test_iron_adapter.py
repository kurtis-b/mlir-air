# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only checks that the iron adapter refuses what it should.

    python3 study/test_iron_adapter.py

No NPU, no MLIR, no iron checkout: the fixture below is a synthetic row over
iron's real column names. Plain test_* functions with a main() runner, matching
pattern/test_reference.py (porting convention 11).

The tests that matter are the refusals. An adapter that carries everything over
is worse than no adapter, because it produces a table that looks joinable and
compares numbers measured over different clocks -- which is the failure doc 03
wrote the adapter to prevent.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import iron_adapter  # noqa: E402
import schema  # noqa: E402

# A synthetic row over iron's RESULTS_CSV_FIELDNAMES. Values are arbitrary; the
# column names are the real ones.
IRON_ROW = {
    "study_id": "s1",
    "study_case_id": "c1",
    "study_case_label": "bert 512",
    "workload_variant": "encoder_bert",
    "backend": "xrt",
    "execution_mode": "hybrid",
    "pattern_label": "coarse runlist",
    "seq_len": 512,
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_attention_heads": 12,
    "attention_head_size": 64,
    "batch_size": 1,
    "dtype": "bf16",
    "use_bias": True,
    "weights_source": "drawn",
    "warmup_runs": 5,
    "runs_per_sample": 4,
    "measured_inference_count": 137,
    "latency_sample_count": 34,
    "timed_total_sec": 1.25,
    "avg_latency_ms": 9.1,
    "min_latency_ms": 8.7,
    "max_latency_ms": 11.0,
    "compile_setup_time_ms": 812.0,
    "host_qkv_precompute_ms": 3.2,
    "effective_gflops_per_sec": 410.0,
    "power_backend": "amdsmi",
    "avg_power_w": 12.5,
    "effective_gflops_per_sec_per_watt": 32.8,
    "npu_dispatch_count": 29,
    "npu_unique_instruction_binary_count": 3,
    "npu_unique_xclbin_count": 1,
    "process_model": "in_process",
    "validation_error_count": 0,
    "run_status": "passed",
    "failure_message": "",
    "selected_candidate_ids_json": "[1,2]",
    "selected_config_json": "{}",
    "is_best": True,
}


def _raises(exc, match, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc as e:
        assert match in str(e), f"expected {match!r} in {str(e)!r}"
        return
    raise AssertionError(f"expected {exc.__name__} containing {match!r}")


def test_adapted_row_is_valid_against_the_schema():
    out = iron_adapter.adapt_iron_row(IRON_ROW)
    schema.validate_row(out, "results")  # must not raise


def test_hybrid_crosses_unchanged():
    """It is this study's CSV value too (convention 7), so nothing to translate."""
    out = iron_adapter.adapt_iron_row(IRON_ROW)
    assert out["execution_mode"] == "hybrid"


def test_shape_and_identity_cross():
    out = iron_adapter.adapt_iron_row(IRON_ROW)
    assert out["seq_len"] == 512
    assert out["hidden_size"] == 768
    assert out["workload_variant"] == "encoder_bert"
    assert out["warmup_runs"] == 5


def test_latency_does_not_cross():
    """The headline refusal: iron times a different span, two different ways."""
    out = iron_adapter.adapt_iron_row(IRON_ROW)
    for field in (
        "avg_latency_ms",
        "min_latency_ms",
        "max_latency_ms",
        "timed_total_sec",
    ):
        assert out[field] is None, f"{field} was carried over and must not be"


def test_power_does_not_cross():
    out = iron_adapter.adapt_iron_row(IRON_ROW)
    for field in schema.POWER_FIELDNAMES:
        assert out[field] is None, f"{field} was carried over and must not be"


def test_run_status_does_not_cross():
    """iron's `passed` is a materially weaker claim; it must not be adopted."""
    out = iron_adapter.adapt_iron_row(IRON_ROW)
    assert out["run_status"] is None
    assert "iron_run_status='passed'" in out["failure_message"]


def test_dispatch_vector_does_not_cross_from_one_scalar():
    out = iron_adapter.adapt_iron_row(IRON_ROW)
    for field in schema.DISPATCH_VECTOR_FIELDNAMES:
        assert out[field] is None, f"{field} cannot come from npu_dispatch_count"


def test_attention_path_is_left_unknown_rather_than_guessed():
    """Inferring it from execution_mode would fabricate the confound it exposes."""
    out = iron_adapter.adapt_iron_row(IRON_ROW)
    assert out["attention_path"] is None


def test_what_was_dropped_is_recorded():
    out = iron_adapter.adapt_iron_row(IRON_ROW)
    msg = out["failure_message"]
    assert "incomparable field" in msg
    for field in ("avg_latency_ms", "npu_dispatch_count", "run_status"):
        assert field in msg, f"{field} was dropped silently"


def test_requiring_an_incomparable_field_raises_with_the_reason():
    """The root reason names the mechanism; derived fields point at their root."""
    _raises(
        iron_adapter.IncomparableField,
        "power-sampler-chosen",
        iron_adapter.adapt_iron_row,
        IRON_ROW,
        require=("timed_total_sec",),
    )
    _raises(
        iron_adapter.IncomparableField,
        "derived from timed_total_sec",
        iron_adapter.adapt_iron_row,
        IRON_ROW,
        require=("avg_latency_ms",),
    )


def test_requiring_a_comparable_field_is_fine():
    out = iron_adapter.adapt_iron_row(IRON_ROW, require=("seq_len", "dtype"))
    assert out["seq_len"] == 512


def test_requiring_an_unknown_field_raises_rather_than_passing_silently():
    _raises(
        iron_adapter.IncomparableField,
        "nothing can be claimed about it",
        iron_adapter.adapt_iron_row,
        IRON_ROW,
        require=("some_new_iron_column",),
    )


def test_incomparable_reason_is_available_without_adapting():
    assert iron_adapter.incomparable_reason("avg_latency_ms")
    assert iron_adapter.incomparable_reason("seq_len") is None


def test_a_partial_iron_row_still_adapts():
    """Real trees have older rows with fewer columns."""
    out = iron_adapter.adapt_iron_row({"study_id": "s2", "seq_len": 128})
    schema.validate_row(out, "results")
    assert out["study_id"] == "s2" and out["seq_len"] == 128


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"iron-adapter tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
