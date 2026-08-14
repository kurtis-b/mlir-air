# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only checks on the figure tier.

    python3 study/test_plots.py

WHY THIS MODULE EXISTS AT ALL
    A plot is the one artifact in this study that is read by looking at it, so a
    wrong one does not raise -- it renders. Every check here is aimed at a
    failure that would otherwise produce a *plausible picture*: a stacked bar of
    columns that were never measured, a line drawn straight across a rung that
    failed, a latency chart carrying no power mode, or a component absent from
    the row drawn at the same height as one measured to be zero.

    Each is pinned in BOTH directions where a direction exists. Asserting that a
    v1 tree refuses the decomposition proves nothing on its own -- a function
    that always refused would pass it -- so the v2 case asserts the same figure
    renders, and the absent-component check is paired with a measured-zero
    control that must NOT be reported as absent.
"""

from __future__ import annotations

import csv
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import plots  # noqa: E402
import results_io  # noqa: E402
import schema  # noqa: E402

#: The v1 schema is the first 65 of the current 70 fields; the appended five are
#: the decomposition and reconfiguration set. ``test_results_io`` pins that this
#: prefix relation actually holds against the shipped schema, so this constant
#: cannot drift into describing a layout the schema no longer has.
_V1_FIELD_COUNT = 65


def _row(mode, seq, ms, *, status="passed", v2=True, host_cpu=0.0, device=None):
    row = schema.empty_row("results")
    row["execution_mode"] = schema.EXECUTION_MODE_CSV.get(mode, mode)
    row["study_id"] = "test-plots"
    row["seq_len"] = seq
    row["study_case_id"] = f"{seq}x768_encoder_bert"
    row["study_case_label"] = f"{mode} seq {seq}"
    row["run_status"] = status
    row["avg_latency_ms"] = ms
    row["min_latency_ms"] = None if ms is None else ms * 0.9
    row["max_latency_ms"] = None if ms is None else ms * 1.1
    row["failure_message"] = None if status == "passed" else "synthetic failure"
    row["host_submissions_per_layer"] = 1
    row["herd_launches"] = 2
    row["sync_boundaries"] = 3
    row["bytes_transferred"] = 4096 * seq
    if v2 and ms is not None:
        row["device_ms"] = ms * 0.25 if device is None else device
        row["sync_ms"] = ms * 0.05
        row["host_cpu_ms"] = host_cpu
        row["context_loads"] = 30 if mode == "offload" else 0
        row["kernel_attaches"] = 4 if mode == "offload" else 0
    return row


def _v2_tree(spec):
    """``{mode: [row, ...]}`` written through the real validating writer."""
    tmp = tempfile.TemporaryDirectory()
    for mode, rows in spec.items():
        results_io.write_rows(os.path.join(tmp.name, f"{mode}.csv"), rows)
    return tmp


def _v1_tree(spec):
    """The same, truncated to the v1 column set and stamped schema_version 1.

    Written with the csv module rather than ``results_io``: the point of the
    fixture is a file the CURRENT writer can no longer produce, and generating
    it through that writer would test the wrong thing.
    """
    tmp = tempfile.TemporaryDirectory()
    names = [f.name for f in schema.fields_for("results")][:_V1_FIELD_COUNT]
    for mode, rows in spec.items():
        path = os.path.join(tmp.name, f"{mode}.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=names, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                out = {k: ("" if row.get(k) is None else row[k]) for k in names}
                out["schema_version"] = 1
                writer.writerow(out)
    return tmp


def _footer(fig):
    return "\n".join(t.get_text() for t in fig.texts)


# --- the v1/v2 discrimination, both directions -----------------------------


def test_v1_tree_refuses_the_decomposition_figure():
    """Empty columns must not be stacked as zero -- the chart would look measured."""
    tmp = _v1_tree({"fused": [_row("fused", 512, 10.0, v2=False)]})
    data = plots.load(tmp.name)
    try:
        plots.decomposition(data, plots.conditions(tmp.name))
    except plots.MissingDecomposition as exc:
        assert "device_ms" in str(exc)
        return
    finally:
        tmp.cleanup()
    raise AssertionError("a v1 tree produced a decomposition figure")


def test_v1_tree_refuses_the_reconfiguration_figure():
    tmp = _v1_tree({"offload": [_row("offload", 512, 10.0, v2=False)]})
    data = plots.load(tmp.name)
    try:
        plots.reconfiguration(data, plots.conditions(tmp.name))
    except plots.MissingDecomposition as exc:
        assert "context_loads" in str(exc)
        return
    finally:
        tmp.cleanup()
    raise AssertionError("a v1 tree produced a reconfiguration figure")


def test_v2_tree_renders_both_v2_figures():
    """The control for the two refusals above: a function that always raised
    would pass them, and this is what says it does not."""
    with _v2_tree({"offload": [_row("offload", 512, 10.0)]}) as root:
        data = plots.load(root)
        cond = plots.conditions(root)
        assert plots.decomposition(data, cond) is not None
        assert plots.reconfiguration(data, cond) is not None


def test_regenerate_reports_skips_rather_than_swallowing_them():
    tmp = _v1_tree({"fused": [_row("fused", 512, 10.0, v2=False)]})
    try:
        written, skipped = plots.regenerate(tmp.name, os.path.join(tmp.name, "fig"))
    finally:
        tmp.cleanup()
    assert len(skipped) == 2, skipped
    assert any("cost_decomposition" in s for s in skipped)
    assert any("reconfiguration" in s for s in skipped)
    assert len(written) == 6  # 3 figures x png+svg


# --- failed rungs are gaps, not interpolations -----------------------------


def test_a_failed_rung_is_not_plotted():
    """Drawing a line across a failed rung invents a measurement."""
    with _v2_tree(
        {
            "fused": [
                _row("fused", 512, 10.0),
                _row("fused", 1024, None, status="failed"),
                _row("fused", 2048, 40.0),
            ]
        }
    ) as root:
        fig = plots.latency(plots.load(root), plots.conditions(root))
        xs = list(fig.axes[0].lines[0].get_xdata())
    assert xs == [512, 2048], xs


def test_a_failed_rung_is_named_in_the_footer():
    with _v2_tree(
        {
            "fused": [
                _row("fused", 512, 10.0),
                _row("fused", 1024, None, status="failed"),
                _row("fused", 2048, 40.0),
            ]
        }
    ) as root:
        fig = plots.latency(plots.load(root), plots.conditions(root))
    assert "fused @ 1024" in _footer(fig)


# --- absent component vs measured zero, both directions --------------------


def test_an_absent_component_is_named_rather_than_drawn_as_zero():
    rows = [_row("offload", 512, 10.0)]
    rows[0]["host_cpu_ms"] = None
    with _v2_tree({"offload": rows}) as root:
        fig = plots.decomposition(plots.load(root), plots.conditions(root))
    assert "Component absent" in _footer(fig)
    assert "host CPU" in _footer(fig)


def test_a_measured_zero_is_NOT_reported_as_absent():
    """The discrimination control. schema.py: a recorded 0.0 for host_cpu_ms is
    a measurement -- the mode instruments host compute and ran none. Reporting
    it as absent would erase a real result."""
    with _v2_tree({"fused": [_row("fused", 512, 10.0, host_cpu=0.0)]}) as root:
        fig = plots.decomposition(plots.load(root), plots.conditions(root))
    assert "Component absent" not in _footer(fig)


def test_a_negative_remainder_is_flagged_as_a_defect():
    """Disjoint parts cannot exceed the whole; a large POSITIVE remainder is
    expected and must not be flagged the same way."""
    with _v2_tree({"fused": [_row("fused", 512, 10.0, device=100.0)]}) as root:
        fig = plots.decomposition(plots.load(root), plots.conditions(root))
    assert "NEGATIVE remainder" in _footer(fig)


def test_a_large_positive_remainder_is_not_called_a_defect():
    with _v2_tree({"fused": [_row("fused", 512, 100.0, device=1.0)]}) as root:
        fig = plots.decomposition(plots.load(root), plots.conditions(root))
    footer = _footer(fig)
    assert "NEGATIVE remainder" not in footer
    assert "NOT a defect" in footer


# --- the power-mode stamp, which trap 0 makes load-bearing -----------------


def test_conditions_are_unrecoverable_without_a_manifest():
    with _v2_tree({"fused": [_row("fused", 512, 10.0)]}) as root:
        cond = plots.conditions(root)
    assert cond["npu_power_mode"] == "unrecoverable"


def test_the_stamp_appears_in_every_figure():
    with _v2_tree({"offload": [_row("offload", 512, 10.0)]}) as root:
        data, cond = plots.load(root), plots.conditions(root)
        for fig in (
            plots.latency(data, cond),
            plots.dram(data, cond),
            plots.decomposition(data, cond),
            plots.reconfiguration(data, cond),
        ):
            assert "NPU power mode" in _footer(fig)


def test_conditions_come_from_the_manifest_when_present():
    with _v2_tree({"fused": [_row("fused", 512, 10.0)]}) as root:
        block = schema.empty_conditions()
        block["npu_power_mode"] = "turbo"
        block["npu_power_mode_source"] = "xrt-smi"
        with open(os.path.join(root, plots.MANIFEST_NAME), "w", encoding="utf-8") as fh:
            json.dump({schema.CONDITIONS_KEY: block}, fh)
        cond = plots.conditions(root)
        fig = plots.latency(plots.load(root), cond)
    assert cond["npu_power_mode"] == "turbo"
    assert "turbo" in _footer(fig)


# --- the CLI's gate clause -------------------------------------------------


def test_require_all_fails_on_a_v1_tree():
    tmp = _v1_tree({"fused": [_row("fused", 512, 10.0, v2=False)]})
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = plots.main(
                [tmp.name, "--out", os.path.join(tmp.name, "f"), "--require-all"]
            )
    finally:
        tmp.cleanup()
    assert rc == 1
    assert "skipped" in buf.getvalue()


def test_require_all_passes_on_a_v2_tree():
    with _v2_tree({"offload": [_row("offload", 512, 10.0)]}) as root:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = plots.main([root, "--out", os.path.join(root, "f"), "--require-all"])
    assert rc == 0


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"plots tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
