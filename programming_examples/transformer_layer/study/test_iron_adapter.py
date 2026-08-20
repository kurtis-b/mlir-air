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
    "study_case_id": "baseline_768",
    "study_case_label": "BERT-Base",
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


# --- `[2026-08-20]` identity translation, the tree reader, and validate_port ---


def test_iron_family_id_becomes_this_ports_shape_key():
    """The join key. Pinned to BOTH derivations on this side: what run_mode
    stamps on a measured row and what run_ladder stamps on a synthesized one,
    so the adapter cannot drift from either."""
    import run_ladder

    out = iron_adapter.adapt_iron_row(IRON_ROW)
    assert out["study_case_id"] == "512x768_encoder_bert"
    assert out["study_case_id"] == run_ladder._case_identity(512, "baseline_768")[0]
    # iron's id survives where the schema says a label lives: never parsed.
    assert out["study_case_label"] == "iron:baseline_768"


def test_decoder_family_translates_with_its_variant():
    row = dict(IRON_ROW, study_case_id="gpt2_small_768", workload_variant="decoder_gpt2")
    out = iron_adapter.adapt_iron_row(row)
    assert out["study_case_id"] == "512x768_decoder_gpt2"


def test_unknown_iron_family_is_refused_not_transliterated():
    _raises(
        iron_adapter.UnknownIronCase,
        "not a family in cases.FAMILY_SPECS",
        iron_adapter.adapt_iron_row,
        dict(IRON_ROW, study_case_id="bert_xxl_4096"),
    )


def test_a_row_contradicting_its_own_family_is_refused():
    """baseline_768 with hidden_size 1024 is a corrupt tree, not a 1024 case."""
    _raises(
        iron_adapter.UnknownIronCase,
        "disagrees with its own case",
        iron_adapter.adapt_iron_row,
        dict(IRON_ROW, hidden_size=1024),
    )
    _raises(
        iron_adapter.UnknownIronCase,
        "disagrees with its own case",
        iron_adapter.adapt_iron_row,
        dict(IRON_ROW, workload_variant="decoder_gpt2"),
    )


def test_a_row_without_a_case_id_is_left_untranslated():
    out = iron_adapter.adapt_iron_row({"study_id": "s2", "seq_len": 128})
    assert out["study_case_id"] is None


def _iron_tree(tmp, rows):
    import csv
    import os

    os.makedirs(os.path.join(tmp, "end_to_end"), exist_ok=True)
    path = os.path.join(tmp, iron_adapter.IRON_RESULTS_REL)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(IRON_ROW))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return tmp


def _port_root(tmp, rows_by_csv):
    import os

    import results_io

    os.makedirs(tmp, exist_ok=True)
    for name, rows in rows_by_csv.items():
        results_io.write_rows(os.path.join(tmp, name), rows)
    return tmp


def _port_row(mode_csv, seq, hidden=768, variant="encoder_bert", **over):
    row = schema.empty_row("results")
    row.update(
        study_id="t",
        study_case_id=f"{seq}x{hidden}_{variant}",
        study_case_label="t",
        workload_variant=variant,
        backend="xrt",
        execution_mode=mode_csv,
        seq_len=seq,
        hidden_size=hidden,
        intermediate_size=hidden * 4,
        num_attention_heads=12,
        attention_head_size=64,
        batch_size=1,
        dtype="bf16",
        run_status="passed",
    )
    row.update(over)
    return row


def test_read_iron_results_refuses_a_root_without_the_file():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        assert not iron_adapter.is_iron_root(tmp)
        _raises(FileNotFoundError, "not an iron results root", iron_adapter.read_iron_results, tmp)


def test_read_iron_results_adapts_every_row_and_splits_by_mode():
    import tempfile

    rows = [
        dict(IRON_ROW, execution_mode="hybrid"),
        dict(IRON_ROW, execution_mode="runlist"),
        dict(IRON_ROW, execution_mode="offload", seq_len=1024),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        _iron_tree(tmp, rows)
        assert iron_adapter.is_iron_root(tmp)
        adapted = iron_adapter.read_iron_results(tmp)
        assert len(adapted) == 3
        for row in adapted:
            schema.validate_row(row, "results")
            assert row["avg_latency_ms"] is None  # the refusal holds at tree level
        by_csv = iron_adapter.split_by_mode_csv(adapted)
        assert set(by_csv) == {"coarse.csv", "runlist.csv", "offload.csv"}
        assert by_csv["offload.csv"][0]["study_case_id"] == "1024x768_encoder_bert"


def test_split_refuses_a_mode_with_no_per_mode_csv():
    row = schema.empty_row("results")
    row["execution_mode"] = "fused_elf"
    _raises(iron_adapter.IncomparableField, "no per-mode CSV", iron_adapter.split_by_mode_csv, [row])


def test_validate_port_agrees_when_the_shapes_agree():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        iron = _iron_tree(tmp + "/iron", [dict(IRON_ROW, execution_mode="hybrid")])
        port = _port_root(tmp + "/port", {"coarse.csv": [_port_row("hybrid", 512)]})
        report = iron_adapter.validate_port(iron, port, ["coarse.csv"])
        text = report.render()
        assert report.failures == 0, text
        assert "shape agrees on 7/7 fields" in text
        assert "iron_run_status='passed'" in text
        assert "VERDICT: OK" in text


def test_validate_port_fails_on_a_shape_disagreement():
    """The negative control: the same key, a different layer."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        iron = _iron_tree(tmp + "/iron", [dict(IRON_ROW, execution_mode="hybrid")])
        port = _port_root(
            tmp + "/port",
            {"coarse.csv": [_port_row("hybrid", 512, num_attention_heads=16)]},
        )
        report = iron_adapter.validate_port(iron, port, ["coarse.csv"])
        text = report.render()
        assert report.failures == 1, text
        assert "num_attention_heads: port='16' iron='12'" in text
        assert "VERDICT: PROBLEM" in text


def test_validate_port_never_reads_latency():
    """A wildly different latency on the iron side changes nothing: the
    refusal is the contract, and a tree-level path must not reopen it."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        iron = _iron_tree(
            tmp + "/iron", [dict(IRON_ROW, execution_mode="hybrid", avg_latency_ms=0.001)]
        )
        port = _port_root(
            tmp + "/port", {"coarse.csv": [_port_row("hybrid", 512, avg_latency_ms=9000.0)]}
        )
        report = iron_adapter.validate_port(iron, port, ["coarse.csv"])
        assert report.failures == 0 and report.warnings == 0, report.render()
        assert "latency" not in report.render().lower()


def test_validate_port_fails_when_nothing_was_compared():
    """The Codex finding: information-only branches must not add up to OK.
    Three vacuous shapes -- a port root wholly outside iron's grid, an empty
    named port CSV, and a header-only iron file -- each end PROBLEM."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        iron = _iron_tree(tmp + "/iron", [dict(IRON_ROW, execution_mode="hybrid")])
        # (a) every port row is a fused rung: no counterpart can exist.
        port = _port_root(tmp + "/port_a", {"fused.csv": [_port_row("fused_elf", 512)]})
        report = iron_adapter.validate_port(iron, port, ["fused.csv"])
        text = report.render()
        assert "iron has no 'fused_elf' mode" in text
        assert report.failures == 1 and "proved nothing about the port" in text
        assert "VERDICT: PROBLEM" in text
        # (b) a named port CSV with a header and no rows -- forged by hand,
        # since results_io refuses to write one for the same reason.
        import csv
        import os

        os.makedirs(tmp + "/port_b")
        with open(tmp + "/port_b/coarse.csv", "w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=list(schema.RESULTS_FIELDNAMES)).writeheader()
        report = iron_adapter.validate_port(iron, tmp + "/port_b", ["coarse.csv"])
        assert "holds no rows" in report.render() and report.failures >= 1
        # (c) a header-only iron file against a real port row.
        empty_iron = _iron_tree(tmp + "/iron_empty", [])
        port = _port_root(tmp + "/port_c", {"coarse.csv": [_port_row("hybrid", 512)]})
        report = iron_adapter.validate_port(empty_iron, port, ["coarse.csv"])
        assert report.failures >= 1 and "VERDICT: PROBLEM" in report.render()


def test_validate_port_reports_a_partial_shape_agreement_as_partial():
    """Unset fields are counted out of the agreement and warned, never folded
    into 'agrees on 7'."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        iron = _iron_tree(tmp + "/iron", [dict(IRON_ROW, execution_mode="hybrid")])
        port = _port_root(
            tmp + "/port",
            {"coarse.csv": [_port_row("hybrid", 512, num_attention_heads=None, dtype=None)]},
        )
        report = iron_adapter.validate_port(iron, port, ["coarse.csv"])
        text = report.render()
        assert "shape agrees on 5/7 fields" in text
        assert report.warnings == 1 and "num_attention_heads (port unset)" in text
        assert report.failures == 0
        # And a row with NO shape field set on both sides is a failure, not OK.
        bare = _port_row("hybrid", 512)
        for f in iron_adapter.SHAPE_FIELDS:
            bare[f] = None
        port = _port_root(tmp + "/port_bare", {"coarse.csv": [bare]})
        report = iron_adapter.validate_port(iron, port, ["coarse.csv"])
        assert report.failures == 1 and "nothing was compared" in report.render()


def test_validate_port_reports_a_malformed_iron_tree_rather_than_raising():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        iron = _iron_tree(tmp + "/iron", [dict(IRON_ROW, seq_len="five-twelve")])
        port = _port_root(tmp + "/port", {"coarse.csv": [_port_row("hybrid", 512)]})
        report = iron_adapter.validate_port(iron, port, ["coarse.csv"])
        assert report.failures == 1 and "iron tree unreadable" in report.render()


def test_validate_port_names_why_a_row_has_no_counterpart():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        iron = _iron_tree(
            tmp + "/iron",
            [
                dict(IRON_ROW, execution_mode="hybrid"),
                # runlist at 512 exists in iron -- under ANOTHER family.
                dict(
                    IRON_ROW,
                    execution_mode="runlist",
                    study_case_id="gpt2_small_768",
                    workload_variant="decoder_gpt2",
                ),
            ],
        )
        port = _port_root(
            tmp + "/port",
            {
                "fused.csv": [_port_row("fused_elf", 512)],
                "coarse.csv": [_port_row("hybrid", 3000)],
                "runlist.csv": [_port_row("runlist", 512)],
            },
        )
        report = iron_adapter.validate_port(
            iron, port, ["fused.csv", "coarse.csv", "runlist.csv"]
        )
        text = report.render()
        assert "iron has no 'fused_elf' mode" in text
        assert "seq_len outside iron's ladder" in text
        # iron has the mode and the length but not the row: that is a WARN,
        # because it is the shape of a broken family translation.
        assert report.warnings == 1 and "check the family translation" in text
        # Nothing was compared, so the verdict is PROBLEM for THAT reason and
        # no other.
        assert report.failures == 1 and "proved nothing about the port" in text


def test_validate_port_fails_with_no_csvs_and_with_a_missing_csv():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        iron = _iron_tree(tmp + "/iron", [IRON_ROW])
        port = _port_root(tmp + "/port", {})
        report = iron_adapter.validate_port(iron, port, [])
        assert report.failures == 1 and "proved nothing" in report.render()
        report = iron_adapter.validate_port(iron, port, ["coarse.csv"])
        # Two failures, both load-bearing: the named file is missing, AND
        # nothing was compared as a consequence.
        text = report.render()
        assert report.failures == 2 and "missing in the port root" in text
        assert "proved nothing about the port" in text


def test_validate_port_fails_on_an_untranslatable_iron_tree():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        iron = _iron_tree(tmp + "/iron", [dict(IRON_ROW, study_case_id="mystery")])
        port = _port_root(tmp + "/port", {"coarse.csv": [_port_row("hybrid", 512)]})
        report = iron_adapter.validate_port(iron, port, ["coarse.csv"])
        assert report.failures == 1 and "iron tree unreadable" in report.render()


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"iron-adapter tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
