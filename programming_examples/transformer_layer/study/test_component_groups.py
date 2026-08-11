# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only checks on the component-group aggregation.

    python3 study/test_component_groups.py

No NPU and no toolchain: ``aggregate`` takes the ``extra`` dict a dispatch
returns, so it is tested against fixtures shaped like real ones.

The host-bucket taxonomy is a CLAIM ABOUT OTHER FILES -- which
``Profiler.time_cpu`` names the pattern modules open -- so it is re-derived from
those sources here rather than trusted, the same way ``test_attention_path.py``
guards the attention map. That check reads text; it imports nothing.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import component_groups as cg  # noqa: E402
import results_io  # noqa: E402
import schema  # noqa: E402

_TIME_CPU_RE = re.compile(r'time_cpu\(\s*"([a-zA-Z_0-9]+)"\s*\)')

#: mode code name -> the pattern module that implements its dispatch.
_PATTERN_SOURCES = {
    "offload": "pattern/offload/offload.py",
    "runlist": "pattern/runlist/runlist.py",
    "coarse": "pattern/coarse/cells.py",
    "fused": "pattern/fused/fused.py",
}


def _raises(exc, match, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc as e:
        assert match in str(e), f"expected {match!r} in {str(e)!r}"
        return
    raise AssertionError(f"expected {exc.__name__} containing {match!r}")


def _extra(**overrides):
    """An extra dict shaped like the one a mode's dispatch returns."""
    extra = {
        "device_ms": 44.0,
        "sync_ms": 6.4,
        "host_cpu_ms": {},
    }
    extra.update(overrides)
    return extra


def _host_buckets_in(relative_path):
    text = (Path(_EXAMPLE) / relative_path).read_text(encoding="utf-8")
    return set(_TIME_CPU_RE.findall(text))


def test_every_csv_execution_mode_has_a_taxonomy():
    for mode in schema.EXECUTION_MODES:
        assert mode in cg.COMPONENT_GROUPS, mode


def test_the_taxonomy_keys_on_the_csv_value_not_the_code_name():
    """Convention 7's direction, and the error says so."""
    assert "hybrid" in cg.COMPONENT_GROUPS
    assert "coarse" not in cg.COMPONENT_GROUPS
    _raises(ValueError, "convention 7", cg.groups_for, "coarse")


def test_every_group_kind_is_a_declared_one():
    for specs in cg.COMPONENT_GROUPS.values():
        for spec in specs:
            assert spec.kind in cg.GROUP_KINDS, spec


def test_every_mode_has_exactly_one_device_group_and_one_sync_group():
    """More than one device group would imply a split that does not exist yet."""
    for mode, specs in cg.COMPONENT_GROUPS.items():
        kinds = [s.kind for s in specs]
        assert kinds.count("device") == 1, mode
        assert kinds.count("sync") == 1, mode


def test_group_labels_are_unique_within_a_mode():
    for mode, specs in cg.COMPONENT_GROUPS.items():
        labels = [s.label for s in specs]
        assert len(set(labels)) == len(labels), mode


def test_the_host_taxonomy_matches_what_the_pattern_modules_actually_time():
    """A claim about other files, re-derived rather than trusted.

    `offload` is the only mode that opens `Profiler.time_cpu` buckets; the other
    three run no host compute, which is why their host group is empty. If a mode
    grows or loses a bucket, this fails here rather than silently dropping that
    component out of every aggregate.
    """
    for code_name, relative_path in _PATTERN_SOURCES.items():
        csv_mode = schema.EXECUTION_MODE_CSV[code_name]
        declared = {
            component
            for spec in cg.groups_for(csv_mode)
            if spec.kind == "host_cpu"
            for component in spec.components
        }
        assert declared == _host_buckets_in(relative_path), (
            code_name,
            declared,
            _host_buckets_in(relative_path),
        )


def test_offload_is_the_only_mode_with_host_components():
    for mode, specs in cg.COMPONENT_GROUPS.items():
        host = [c for s in specs if s.kind == "host_cpu" for c in s.components]
        assert bool(host) == (mode == "offload"), mode


def test_a_named_host_bucket_lands_in_its_group():
    totals = cg.aggregate(_extra(host_cpu_ms={"softmax": 2.5, "ln1": 1.5}), "offload")
    host = next(t for t in totals if t.kind == "host_cpu")
    assert host.ms == 4.0
    assert host.component_count == 2
    assert host.expected_component_count == 5
    assert not host.is_complete
    assert set(host.missing_components) == {"attention_layout", "gelu", "ln2"}


def test_an_empty_bucket_dict_is_a_measured_zero_for_a_mode_with_no_host_work():
    """`fused` runs nothing on the host; its host group is COMPLETE at 0.0."""
    totals = cg.aggregate(_extra(host_cpu_ms={}), "fused_elf")
    host = next(t for t in totals if t.kind == "host_cpu")
    assert host.ms == 0.0
    assert host.is_complete
    assert host.missing_components == ()


def test_a_missing_host_key_is_unmeasured_not_zero():
    extra = _extra()
    del extra["host_cpu_ms"]
    host = next(t for t in cg.aggregate(extra, "offload") if t.kind == "host_cpu")
    assert host.ms is None
    assert host.component_count == 0


def test_a_bucket_the_taxonomy_does_not_know_is_not_silently_summed():
    """A new bucket must show up as taxonomy drift, not as a bigger number."""
    totals = cg.aggregate(
        _extra(host_cpu_ms={"softmax": 2.0, "mystery": 99.0}), "offload"
    )
    host = next(t for t in totals if t.kind == "host_cpu")
    assert host.ms == 2.0


def test_the_device_group_is_a_mode_total_with_nothing_attributed():
    """The honest partial: the milliseconds are real, the attribution is not."""
    device = next(t for t in cg.aggregate(_extra(), "runlist") if t.kind == "device")
    assert device.ms == 44.0
    assert device.component_count == 0
    assert device.expected_component_count == 12
    assert not device.is_complete
    assert len(device.missing_components) == 12


def test_a_missing_device_key_is_none_not_zero():
    extra = _extra()
    del extra["device_ms"]
    device = next(t for t in cg.aggregate(extra, "runlist") if t.kind == "device")
    assert device.ms is None


def test_the_sync_group_carries_the_mode_sync_total():
    sync = next(t for t in cg.aggregate(_extra(), "hybrid") if t.kind == "sync")
    assert sync.ms == 6.4
    assert sync.is_complete  # it expects no components; it IS a total


def test_attributed_ms_ignores_unmeasured_groups():
    extra = _extra()
    del extra["device_ms"]
    assert cg.attributed_ms(cg.aggregate(extra, "runlist")) == 6.4


def test_the_groups_do_not_claim_to_sum_to_the_layer_latency():
    """Doc 03's remainder, printed rather than implied."""
    totals = cg.aggregate(_extra(), "runlist")
    report = cg.render(totals, execution_mode="runlist", avg_latency_ms=1959.0)
    assert "UNATTRIBUTED" in report
    # 1959.0 - (44.0 device + 6.4 sync + 0.0 host), worked out here rather than
    # read back off the module.
    assert "attributed: 50.400 ms" in report
    assert "1908.600 ms" in report
    assert "97.4%" in report


def test_the_report_names_what_is_not_attributed():
    report = cg.render(cg.aggregate(_extra(), "runlist"), execution_mode="runlist")
    assert "not attributed to components" in report
    assert "qkvo_proj" in report
    assert "record_kernel" in report


def test_rows_validate_and_round_trip():
    totals = cg.aggregate(_extra(host_cpu_ms={"softmax": 1.0}), "offload")
    built = cg.rows(
        totals,
        execution_mode="offload",
        study_case_id="4096x768_encoder_bert",
        workload_variant="encoder_bert",
        seq_len=4096,
    )
    for row in built:
        schema.validate_row(row, "component")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "components.csv"
        results_io.write_rows(path, built, table="component")
        back = results_io.read_rows(path, table="component")
    assert len(back) == len(built)
    assert {r["group_kind"] for r in back} == set(cg.GROUP_KINDS)


def test_missing_components_are_recorded_as_json_not_as_a_count():
    """A count says how much is missing; the names say what to instrument."""
    totals = cg.aggregate(_extra(), "hybrid")
    device = next(
        r
        for r in cg.rows(totals, execution_mode="hybrid")
        if r["group_kind"] == "device"
    )
    assert json.loads(device["missing_components_json"]) == [
        "qkv_proj",
        "mha_out_proj",
        "add_norm1",
        "ffn",
        "add_norm2",
    ]


def test_a_group_with_no_milliseconds_writes_a_failed_row():
    extra = _extra()
    del extra["device_ms"]
    device = next(
        r
        for r in cg.rows(cg.aggregate(extra, "runlist"), execution_mode="runlist")
        if r["group_kind"] == "device"
    )
    assert device["run_status"] == "failed"
    assert device["avg_latency_ms"] is None
    assert "no milliseconds" in device["failure_message"]


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"component-group tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
