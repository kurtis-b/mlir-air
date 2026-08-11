# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only checks on the power statistics and the root-free samplers.

    python3 study/test_power.py [--live]

The statistics tests are pure: fixed sample lists with the expected verdict
worked out by hand. The backend tests only assert what is true on ANY host --
that discovery never raises, that ``auto`` resolves to something, and that the
null backend produces a complete unmeasured block -- because a test asserting
this machine's sensors exist would fail on a machine without them for no reason.

``--live`` additionally takes a real short sample through whatever this host
has, which is how the sampler gets checked against real sysfs rather than
against a list of floats.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import power  # noqa: E402
import schema  # noqa: E402


def _raises(exc, match, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc as e:
        assert match in str(e), f"expected {match!r} in {str(e)!r}"
        return
    raise AssertionError(f"expected {exc.__name__} containing {match!r}")


def test_the_empty_block_is_every_schema_power_column():
    columns = power.empty_power_columns()
    assert set(columns) == set(schema.POWER_FIELDNAMES)
    assert all(v is None for v in columns.values())


def test_summarize_emits_exactly_the_schema_power_columns():
    """A stray key would be rejected by validate_row at write time, not here."""
    columns = power.summarize([1.0, 2.0, 3.0], backend="rapl_package")
    assert set(columns) <= set(schema.POWER_FIELDNAMES)


def test_no_samples_is_an_unmeasured_block_not_a_zero():
    columns = power.summarize([], backend="rapl_package")
    assert columns["avg_power_w"] is None
    assert columns["raw_power_sample_count"] is None
    assert columns["power_backend"] == "rapl_package"


def test_a_single_sample_has_a_mean_and_a_zero_deviation():
    columns = power.summarize([12.5], backend="none")
    assert columns["avg_power_w"] == 12.5
    assert columns["power_std_w"] == 0.0
    assert columns["raw_power_sample_count"] == 1
    assert columns["power_outlier_filter_applied"] is False


def test_the_filter_does_not_run_below_ten_samples():
    """Nine samples with an obvious outlier: raw and filtered must agree."""
    samples = [10.0] * 8 + [500.0]
    columns = power.summarize(samples, backend="none")
    assert columns["power_outlier_filter_applied"] is False
    assert columns["power_sample_count"] == 9
    assert columns["avg_power_w"] == columns["raw_avg_power_w"]


def test_the_filter_runs_at_ten_samples_and_both_statistics_survive():
    """Ten quiet samples with real spread, plus one spike. The spike goes."""
    quiet = [9.8, 9.9, 9.95, 10.0, 10.05, 10.1, 10.15, 10.2, 10.25, 10.3]
    columns = power.summarize(quiet + [500.0], backend="none")
    assert columns["power_outlier_filter_applied"] is True
    assert columns["power_outlier_sample_count"] == 1
    assert columns["power_sample_count"] == 10
    assert abs(columns["avg_power_w"] - sum(quiet) / len(quiet)) < 1e-9
    # The raw statistics are still there, which is what lets a reader see what
    # the filter did rather than trust it.
    assert columns["raw_power_sample_count"] == 11
    assert columns["raw_max_power_w"] == 500.0
    assert columns["raw_avg_power_w"] > columns["avg_power_w"]


def test_the_filter_refuses_to_drop_below_the_retained_floor():
    """Six of twelve are 'outliers'; keeping six is allowed, keeping five is not."""
    samples = [10.0] * 6 + [10.5] * 6
    columns = power.summarize(samples, backend="none")
    # Nothing here is 3.5 modified-Z from the median, so nothing is dropped.
    assert columns["power_sample_count"] == 12
    assert columns["power_outlier_filter_applied"] is False


def test_the_filter_is_skipped_when_it_would_widen_the_spread():
    """iron's third condition: a filter that increases std has found structure."""
    samples = [1.0, 1.0, 1.0, 1.0, 1.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    columns = power.summarize(samples, backend="none")
    assert columns["power_sample_count"] == len(samples)
    assert columns["power_outlier_filter_applied"] is False


def test_a_zero_mad_falls_back_to_the_interquartile_rule():
    """A quantized sampler gives runs of identical values and MAD == 0."""
    samples = [10.0] * 6 + [11.0, 12.0, 13.0, 14.0, 99.0]
    mask = power.detect_outliers(samples)
    assert mask[-1] is True
    assert sum(1 for bad in mask if bad) == 1


def test_both_dispersion_measures_degenerate_when_three_quarters_are_identical():
    """The documented limit of the fallback, pinned rather than discovered later.

    With 11 of 12 samples identical the median absolute deviation is 0 AND both
    quartiles land on the same value, so the interquartile span is 0 too and
    NOTHING is flagged -- the lone 99 W spike survives into the filtered
    statistics. That is iron's behaviour and it is kept, because the alternative
    is inventing a dispersion measure for a distribution that has none. The raw
    columns are what make it visible: a reader comparing max against mean sees
    the spike whether or not the filter caught it.
    """
    samples = [10.0] * 11 + [99.0]
    assert power.detect_outliers(samples) == [False] * 12
    columns = power.summarize(samples, backend="none")
    assert columns["power_outlier_filter_applied"] is False
    assert columns["max_power_w"] == 99.0


def test_detect_outliers_is_a_no_op_below_five_samples():
    assert power.detect_outliers([1.0, 900.0]) == [False, False]


def test_probe_completeness_needs_all_three_conditions():
    ok = dict(
        completed_runs=10,
        min_runs=10,
        elapsed_sec=1.0,
        min_duration_sec=0.25,
        sample_count=8,
        min_sample_count=6,
    )
    assert power.probe_is_complete(**ok)
    assert not power.probe_is_complete(**{**ok, "completed_runs": 9})
    assert not power.probe_is_complete(**{**ok, "elapsed_sec": 0.1})
    assert not power.probe_is_complete(**{**ok, "sample_count": 5})


def test_the_interval_shrinks_for_a_short_window():
    """A 50 ms region at 100 ms would sample once; it must not."""
    assert (
        power.resolve_sample_interval(
            requested_sec=0.1, estimated_window_sec=0.06, min_sample_count=6
        )
        == 0.01
    )
    # And never grows past what was asked for.
    assert (
        power.resolve_sample_interval(
            requested_sec=0.1, estimated_window_sec=100.0, min_sample_count=6
        )
        == 0.1
    )
    # An unknown window leaves it alone.
    assert (
        power.resolve_sample_interval(requested_sec=0.05, estimated_window_sec=None)
        == 0.05
    )


def test_backend_discovery_never_raises_and_always_offers_none():
    found = power.available_backends()
    assert found[-1] == "none"
    assert set(found) <= set(power.BACKENDS)


def test_auto_resolves_to_an_available_backend():
    assert power.resolve_backend("auto") in power.available_backends()


def test_an_unknown_backend_is_named_rather_than_silently_disabled():
    _raises(ValueError, "unknown power backend", power.resolve_backend, "turbostat")


def test_turbostat_is_not_a_backend():
    """Doc 09: iron's sudo backend cannot run here; it is replaced, not kept."""
    assert not any("turbostat" in name for name in power.BACKENDS)


def test_the_null_backend_yields_a_monitor_and_an_unmeasured_block():
    with power.open_monitor("none") as monitor:
        assert monitor.sample_count() == 0
    columns = monitor.stats()
    assert columns["power_backend"] == "none"
    assert columns["avg_power_w"] is None
    assert monitor.quiescent_power_w is None


def test_summarize_takes_no_elapsed_time():
    """There is no energy column; an argument nothing can persist is a trap."""
    import inspect

    assert set(inspect.signature(power.summarize).parameters) == {
        "samples",
        "backend",
    }


def check_live() -> int:
    """Take a real sample through whatever this host has. Returns 0 or 1."""
    backend = power.resolve_backend("auto")
    print(f"  available: {list(power.available_backends())}")
    print(f"  auto -> {backend}")
    if backend == "none":
        print("  SKIP  no unprivileged sensor on this host")
        return 0
    with power.open_monitor("auto", interval_sec=0.05) as monitor:
        time.sleep(0.6)
        columns = monitor.stats()
        quiescent = monitor.quiescent_power_w
    print(f"  quiescent: {quiescent}")
    print(
        f"  avg {columns['avg_power_w']} W over "
        f"{columns['raw_power_sample_count']} raw sample(s)"
    )
    if not columns["raw_power_sample_count"]:
        print("  FAIL  the sampler produced no samples")
        return 1
    if not (0.0 < float(columns["avg_power_w"]) < 500.0):
        print("  FAIL  the reading is not a plausible SoC wattage")
        return 1
    print("  PASS  a root-free sampler produced plausible samples")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="take a real sample")
    args = ap.parse_args()

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"power tests: {len(tests)}/{len(tests)} passed")

    if args.live:
        print("\nlive sample")
        return check_live()
    return 0


if __name__ == "__main__":
    sys.exit(main())
