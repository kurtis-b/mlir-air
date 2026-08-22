"""Both directions of the cross-mode distinguishability gate (``distinguish.py``)."""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import distinguish  # noqa: E402

_WALK2_512 = distinguish.WALK2_512_VECTORS


def _vectors(**overrides):
    v = {m: dict(_WALK2_512[m]) for m in _WALK2_512}
    for mode, fields in overrides.items():
        v[mode].update(fields)
    return v


def test_the_measured_walk_is_separated():
    assert distinguish.distinguish(_vectors()) == []


def test_identical_vectors_fail_the_floor():
    v = _vectors()
    v["fused"] = dict(v["coarse"])
    fails = distinguish.distinguish(v)
    assert any("coarse and fused have identical" in f for f in fails), fails


def test_offload_must_submit_more_than_every_other_mode():
    fails = distinguish.distinguish(_vectors(offload=dict(host_submissions_per_layer=4)))
    assert any("offload host_submissions_per_layer 4 does not exceed coarse" in f for f in fails), fails


def test_offload_must_not_aggregate():
    fails = distinguish.distinguish(_vectors(offload=dict(runlist_entries_per_submission=2.0)))
    assert any("offload aggregates" in f for f in fails), fails


def test_runlist_must_execute_more_than_coarse():
    fails = distinguish.distinguish(_vectors(runlist=dict(herd_launches=33)))
    assert fails == ["runlist herd_launches 33 does not exceed coarse's 33"], fails


def test_fused_must_cross_fewer_sync_boundaries_than_coarse():
    fails = distinguish.distinguish(_vectors(fused=dict(sync_boundaries=59)))
    assert fails == ["fused sync_boundaries 59 is not below coarse's 59"], fails


def test_a_missing_mode_is_named_not_passed():
    v = _vectors()
    del v["runlist"]
    assert distinguish.distinguish(v) == ["need all four modes, have ['coarse', 'fused', 'offload']"]


def test_a_missing_field_is_named_not_passed():
    fails = distinguish.distinguish(_vectors(coarse=dict(herd_launches=None)))
    assert fails == ["coarse.herd_launches=None is not a number"], fails


def test_a_nan_cannot_buy_distinctness():
    # runlist and fused byte-identical except a NaN in one field: NaN != NaN would pass the
    # distinctness floor if validation did not fail closed first.
    v = _vectors()
    v["fused"] = dict(v["runlist"])
    v["fused"]["bytes_transferred"] = float("nan")
    fails = distinguish.distinguish(v)
    assert fails == ["fused.bytes_transferred=nan is not finite"], fails


def test_negative_fractional_and_boolean_values_are_refused():
    fails = distinguish.distinguish(_vectors(coarse=dict(sync_boundaries=-1)))
    assert fails == ["coarse.sync_boundaries=-1 is negative"], fails
    fails = distinguish.distinguish(_vectors(coarse=dict(herd_launches=33.5)))
    assert fails == ["coarse.herd_launches=33.5 is a count and must be a whole number"], fails
    fails = distinguish.distinguish(_vectors(coarse=dict(herd_launches=True)))
    assert fails == ["coarse.herd_launches=True is not a number"], fails


def test_zero_submissions_a_fractional_entry_total_and_zero_bytes_are_refused():
    fails = distinguish.distinguish(_vectors(fused=dict(host_submissions_per_layer=0)))
    assert "fused records 0 host submissions" in fails[0], fails
    fails = distinguish.distinguish(_vectors(coarse=dict(runlist_entries_per_submission=4.8)))
    assert "19.2 runlist entries over 4 submission(s)" in fails[0], fails
    fails = distinguish.distinguish(_vectors(offload=dict(bytes_transferred=0)))
    assert fails == ["offload moved zero bytes; nothing was transferred to a device"], fails


def test_strings_from_a_csv_are_numbers_here():
    v = {m: {k: str(x) for k, x in _WALK2_512[m].items()} for m in _WALK2_512}
    assert distinguish.distinguish(v) == []


def test_a_root_gates_only_lengths_where_all_four_passed(tmp_path=None):
    import tempfile
    import results_io
    import schema
    root = tempfile.mkdtemp(prefix="distinguish-")
    base = {f.name: None for f in schema.fields_for("results")}
    base["schema_version"] = schema.SCHEMA_VERSION
    csv_name = {"offload": "offload", "runlist": "runlist", "coarse": "hybrid", "fused": "fused_elf"}
    for mode in distinguish.MODES:
        rows = []
        for n in (512, 1024):
            row = dict(base, execution_mode=csv_name[mode], seq_len=n, run_status="passed",
                       **_WALK2_512[mode])
            if mode == "fused" and n == 1024:
                row["run_status"] = "skipped"
            rows.append(row)
        results_io.write_rows(os.path.join(root, f"{mode}.csv"), rows)
    gated, lines = distinguish.gate_root(root)
    assert gated == 1, (gated, lines)
    assert lines == ["seq 1024: not gated (fused skipped)"], lines


def test_a_failed_row_is_a_failure_not_a_skip():
    import tempfile
    import results_io
    import schema
    root = tempfile.mkdtemp(prefix="distinguish-")
    base = {f.name: None for f in schema.fields_for("results")}
    base["schema_version"] = schema.SCHEMA_VERSION
    csv_name = {"offload": "offload", "runlist": "runlist", "coarse": "hybrid", "fused": "fused_elf"}
    for mode in distinguish.MODES:
        rows = []
        for n in (512, 1024):
            row = dict(base, execution_mode=csv_name[mode], seq_len=n, run_status="passed",
                       **_WALK2_512[mode])
            if mode == "fused" and n == 1024:
                row["run_status"] = "failed"
            rows.append(row)
        if mode == "runlist":
            rows = rows[:1]  # no row at all at 1024
        results_io.write_rows(os.path.join(root, f"{mode}.csv"), rows)
    gated, lines = distinguish.gate_root(root)
    assert gated == 1, (gated, lines)
    assert lines == ["seq 1024: FAIL runlist has no row; fused run_status=failed"], lines
    gated, lines = distinguish.gate_root(root, seq_len=2048)
    assert (gated, lines) == (0, ["seq 2048: FAIL no mode has a row"]), (gated, lines)


def test_an_expected_length_missing_from_every_mode_is_a_failure():
    import tempfile
    import results_io
    import schema
    root = tempfile.mkdtemp(prefix="distinguish-")
    base = {f.name: None for f in schema.fields_for("results")}
    base["schema_version"] = schema.SCHEMA_VERSION
    csv_name = {"offload": "offload", "runlist": "runlist", "coarse": "hybrid", "fused": "fused_elf"}
    for mode in distinguish.MODES:
        # Two rows per mode, BOTH at 512: the row count a 512+1024 walk would have, with
        # 1024 vanished from every CSV.
        rows = [dict(base, execution_mode=csv_name[mode], seq_len=512, run_status="passed",
                     **_WALK2_512[mode]) for _ in range(2)]
        results_io.write_rows(os.path.join(root, f"{mode}.csv"), rows)
    gated, lines = distinguish.gate_root(root, expected_seqs=(512, 1024))
    assert gated == 1, (gated, lines)
    assert "seq 1024: FAIL no mode has a row" in lines, lines
    dups = [line for line in lines if "has more than one row" in line]
    assert len(dups) == 4 and dups[0] == "seq 512: FAIL offload has more than one row", lines
    # Without the declared lengths the absence is invisible -- which is why gate() passes them.
    gated, lines = distinguish.gate_root(root)
    assert not any("seq 1024" in line for line in lines), lines


def test_float_noise_in_the_mean_does_not_buy_distinctness():
    v = _vectors()
    v["fused"] = dict(v["runlist"])
    v["fused"]["runlist_entries_per_submission"] = 91 / 17 + 1e-11
    fails = distinguish.distinguish(v)
    assert any("runlist and fused have identical" in f for f in fails), fails


def test_a_near_integral_entry_total_is_refused_at_the_retired_tolerance():
    fails = distinguish.distinguish(_vectors(coarse=dict(runlist_entries_per_submission=4.750000125)))
    assert "19.0000005 runlist entries over 4 submission(s)" in fails[0], fails


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"distinguish tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
