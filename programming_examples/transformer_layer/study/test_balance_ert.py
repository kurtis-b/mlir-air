# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host tests for ``balance_ert.py``. No NPU, no toolchain, no results tree.

WHAT THESE ARE REALLY GUARDING
    Doc 44 specifies the ERT in one sentence with a defect attached: "a
    ``dma_transfer`` is a function of ``(n_words, n_dims, stride)``, not a
    scalar -- given our BD-stride walls, a counter reporting 'number of DMA
    transfers' has already destroyed the information we need."

    So the two load-bearing tests here are
    ``test_dma_transfer_priced_as_a_scalar_is_refused`` and
    ``test_lookup_is_exact_on_strides``. Everything else guards the honesty
    invariants: a value is present exactly when its source is not ``absent``,
    a timed cost carries a provenance path, and a repeat measurement widens a
    spread instead of overwriting a number.

    The sweep-seeding tests build their JSON in a temp directory rather than
    reading ``sweep/results``, so this suite stays hermetic and never asserts
    anything about this host's recorded measurements.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import balance_ert as E  # noqa: E402


def _dma_arguments(**overrides):
    arguments = {
        "n_words": 8192,
        "n_dims": 4,
        "strides": (65536, 256, 2048, 1),
        "element_bytes": 2,
    }
    arguments.update(overrides)
    return arguments


def _counted(size: int = 16384) -> E.Cost:
    return E.Cost(bytes=size, bytes_source="counted", provenance="fixture")


def _measured(ns: float, path: str = "fixture.json") -> E.Cost:
    return E.Cost(ns=ns, ns_source="measured", provenance=path)


# --------------------------------------------------------------------------
# The arguments are the point
# --------------------------------------------------------------------------


def test_dma_transfer_priced_as_a_scalar_is_refused():
    # THE defect doc 44 names. A dma_transfer keyed on a count has already lost
    # what the BD-stride walls need.
    ert = E.Ert()
    try:
        ert.add("shim_dma", "dma_transfer", _counted(), transfer_count=17)
    except ValueError as e:
        assert "n_words" in str(e) and "strides" in str(e)
    else:
        raise AssertionError(
            "a dma_transfer entered the table with no descriptor shape"
        )
    assert len(ert) == 0


def test_dma_transfer_missing_only_the_strides_is_still_refused():
    # The near miss, which is the realistic one: someone records words and
    # dimensions and drops the stride list.
    ert = E.Ert()
    arguments = _dma_arguments()
    del arguments["strides"]
    try:
        ert.add("shim_dma", "dma_transfer", _counted(), **arguments)
    except ValueError as e:
        assert "strides" in str(e)
    else:
        raise AssertionError("strides are not optional; doc 23's retile is a stride")


def test_dma_transfer_with_the_full_descriptor_is_accepted():
    # Failing direction for the two above.
    ert = E.Ert()
    ert.add("shim_dma", "dma_transfer", _counted(), **_dma_arguments())
    assert len(ert) == 1


def test_gemm_requires_every_tiling_factor():
    ert = E.Ert()
    try:
        ert.add("gemm", "direct", _measured(1000.0), M=512, K=768, N=3072)
    except ValueError as e:
        assert "tile_m" in str(e)
    else:
        raise AssertionError("a gemm was priced without the tiling that produced it")


def test_a_new_gemm_action_inherits_the_argument_requirement():
    # `gemm`'s row is keyed on the "*" action so a method nobody has written
    # yet cannot enter with a smaller argument set than its siblings.
    ert = E.Ert()
    try:
        ert.add("gemm", "some-new-method", _measured(1.0), M=1, K=1, N=1)
    except ValueError:
        pass
    else:
        raise AssertionError("a new gemm method bypassed REQUIRED_ARGUMENTS")


# --------------------------------------------------------------------------
# Lookup is exact
# --------------------------------------------------------------------------


def test_lookup_is_exact_on_strides():
    # Two transfers agreeing on n_words and differing in strides are different
    # objects to the BD allocator. A nearest-match fallback would erase exactly
    # the difference the table exists to preserve.
    ert = E.Ert()
    ert.add("shim_dma", "dma_transfer", _counted(), **_dma_arguments())
    same = ert.lookup("shim_dma", "dma_transfer", **_dma_arguments())
    assert same.bytes == 16384
    try:
        ert.lookup(
            "shim_dma",
            "dma_transfer",
            **_dma_arguments(strides=(6144, 8, 768, 1)),
        )
    except E.ErtMiss as e:
        assert "strides" in str(e)
    else:
        raise AssertionError(
            "a lookup differing only in stride order returned the other "
            "descriptor's cost"
        )


def test_lookup_miss_names_the_difference():
    ert = E.Ert()
    ert.add("shim_dma", "dma_transfer", _counted(), **_dma_arguments())
    try:
        ert.lookup("shim_dma", "dma_transfer", **_dma_arguments(n_dims=3))
    except E.ErtMiss as e:
        assert "n_dims" in str(e)


def test_lookup_miss_on_an_unknown_component_is_still_a_miss():
    ert = E.Ert()
    try:
        ert.lookup("nothing", "at-all", x=1)
    except E.ErtMiss:
        pass
    else:
        raise AssertionError("lookup returned a cost for a component with no entries")


def test_argument_order_is_not_part_of_the_key():
    ert = E.Ert()
    ert.add("shim_dma", "dma_transfer", _counted(), **_dma_arguments())
    reordered = dict(reversed(list(_dma_arguments().items())))
    assert ert.lookup("shim_dma", "dma_transfer", **reordered).bytes == 16384


def test_strides_stay_a_sequence_and_not_a_string():
    # A stringified stride list compares equal for two different dimension
    # orders under some renderings, and the dimension ORDER is what makes a
    # retile a retile (doc 23).
    key = E.ActionKey.of("shim_dma", "dma_transfer", **_dma_arguments())
    assert key.argument_dict["strides"] == (65536, 256, 2048, 1)
    other = E.ActionKey.of(
        "shim_dma", "dma_transfer", **_dma_arguments(strides=(1, 2048, 256, 65536))
    )
    assert key != other


# --------------------------------------------------------------------------
# Honesty invariants on Cost
# --------------------------------------------------------------------------


def test_a_value_is_present_exactly_when_its_source_is_not_absent():
    for kwargs in (
        {"ns": 1.0, "ns_source": "absent"},
        {"ns": None, "ns_source": "measured", "provenance": "p"},
        {"bytes": 1, "bytes_source": "absent"},
        {"bytes": None, "bytes_source": "counted"},
    ):
        try:
            E.Cost(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"Cost({kwargs}) built a number whose source disagrees with it; "
                "an unpriced action reading as a free one is the failure this "
                "pairing prevents"
            )


def test_a_timed_cost_needs_a_provenance_path():
    try:
        E.Cost(ns=1.0, ns_source="measured")
    except ValueError as e:
        assert "provenance" in str(e)
    else:
        raise AssertionError("a latency entered the table with no artifact behind it")


def test_an_unknown_source_is_refused():
    try:
        E.Cost(ns=1.0, ns_source="probably", provenance="p")
    except ValueError:
        pass
    else:
        raise AssertionError("a source outside COST_SOURCES was accepted")


def test_condition_defaults_to_unknown_and_not_to_turbo():
    # Doc 32 measures a ~15-20x Turbo/Default error. A latency defaulted to
    # Turbo would be a claim nobody made.
    cost = _measured(100.0)
    assert cost.condition == E.CONDITION_UNKNOWN
    assert "turbo" not in cost.condition.lower()


def test_by_source_reports_the_split():
    ert = E.Ert()
    ert.add("shim_dma", "dma_transfer", _counted(), **_dma_arguments())
    ert.add(
        "gemm",
        "direct",
        _measured(1000.0),
        M=512,
        K=768,
        N=3072,
        tile_m=64,
        tile_k_l2=256,
        tile_k_l1=32,
        tile_n=96,
        herd_m=8,
        herd_n=4,
    )
    assert ert.by_source() == {
        "measured": 1,
        "counted": 0,
        "modelled": 0,
        "absent": 1,
    }
    assert ert.bytes_by_source()["counted"] == 1


# --------------------------------------------------------------------------
# Repeat measurements are a distribution, not an overwrite
# --------------------------------------------------------------------------


def _gemm_key():
    return E.ActionKey.of(
        "gemm",
        "fused-cast",
        M=512,
        K=4096,
        N=1024,
        tile_m=64,
        tile_k_l2=512,
        tile_k_l1=32,
        tile_n=64,
        herd_m=8,
        herd_n=4,
    )


def test_observe_widens_the_spread_and_keeps_the_minimum():
    # The real pair from the sweep tree: 1,086,524 ns and 1,545,360 ns for one
    # identical priced action -- a 42.2% spread. A table holding whichever
    # arrived last would assert one of them as the cost.
    ert = E.Ert()
    key = _gemm_key()
    ert.insert(key, _measured(1_545_360.0, "second.json"))
    assert ert.observe(key, 1_086_524.0, provenance="first.json")
    cost = ert.lookup("gemm", "fused-cast", **key.argument_dict)
    assert cost.ns == 1_086_524.0
    assert cost.ns_min == 1_086_524.0
    assert cost.ns_max == 1_545_360.0
    assert cost.ns_samples == 2
    assert abs(cost.ns_spread - (1_545_360.0 - 1_086_524.0) / 1_086_524.0) < 1e-12
    # The provenance follows the minimum, which is the file a reader checks.
    assert cost.provenance == "first.json"


def test_observe_keeps_the_minimum_when_the_new_sample_is_slower():
    ert = E.Ert()
    key = _gemm_key()
    ert.insert(key, _measured(1000.0, "fast.json"))
    ert.observe(key, 2000.0, provenance="slow.json")
    cost = ert.lookup("gemm", "fused-cast", **key.argument_dict)
    assert cost.ns == 1000.0 and cost.ns_max == 2000.0
    assert cost.provenance == "fast.json"


def test_observe_on_an_unpriced_key_reports_the_miss():
    ert = E.Ert()
    assert ert.observe(_gemm_key(), 1.0, provenance="p") is False
    assert len(ert) == 0


def test_spread_is_none_below_two_samples():
    cost = _measured(1000.0)
    assert cost.ns_samples == 1
    assert cost.ns_min == cost.ns_max == 1000.0
    assert cost.ns_spread is None


def test_insert_refuses_a_conflicting_duplicate():
    ert = E.Ert()
    key = _gemm_key()
    ert.insert(key, _measured(1000.0))
    try:
        ert.insert(key, _measured(2000.0))
    except ValueError as e:
        assert "already priced differently" in str(e)
    else:
        raise AssertionError("one action carried two costs and the last one won")
    ert.insert(key, _measured(2000.0), replace=True)
    assert ert.lookup("gemm", "fused-cast", **key.argument_dict).ns == 2000.0


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def test_json_round_trip_preserves_the_argument_structure():
    ert = E.Ert()
    ert.add("shim_dma", "dma_transfer", _counted(), **_dma_arguments())
    key = _gemm_key()
    ert.insert(key, _measured(1000.0, "a.json"))
    ert.observe(key, 900.0, provenance="b.json")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ert.json"
        ert.save(path)
        back = E.Ert.load(path)

    assert len(back) == 2
    dma = back.lookup("shim_dma", "dma_transfer", **_dma_arguments())
    assert dma.bytes == 16384 and dma.bytes_source == "counted"
    gemm = back.lookup("gemm", "fused-cast", **key.argument_dict)
    assert (gemm.ns, gemm.ns_min, gemm.ns_max, gemm.ns_samples) == (
        900.0,
        900.0,
        1000.0,
        2,
    )


def test_load_refuses_a_foreign_format_version():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ert.json"
        path.write_text(
            json.dumps({"ert_format_version": 99, "records": []}), encoding="utf-8"
        )
        try:
            E.Ert.load(path)
        except ValueError as e:
            assert "99" in str(e)
        else:
            raise AssertionError("a table of an unknown shape was read as this one")


# --------------------------------------------------------------------------
# Seeding
# --------------------------------------------------------------------------


def _sweep_file(directory: Path, name: str, *, status="passed", latency=100.0, **over):
    candidate = {
        "method": "direct",
        "tile_m": 64,
        "tile_k_l2": 256,
        "tile_k_l1": 32,
        "tile_n": 96,
        "herd_m": 8,
        "herd_n": 4,
    }
    candidate.update(over.pop("candidate", {}))
    blob = {
        "shape": {"M": 512, "K": 768, "N": 3072, "role": over.pop("role", "qkv_proj")},
        "candidate": candidate,
        "status": status,
        "latency_us": latency,
    }
    blob.update(over)
    path = directory / name
    path.write_text(json.dumps(blob), encoding="utf-8")
    return str(path)


def test_seed_counts_added_merged_and_failed_separately():
    # A bare `skipped` total would report a repeat measurement and a failed
    # candidate as the same event; they are not.
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        paths = [
            _sweep_file(directory, "a.json", latency=100.0),
            _sweep_file(directory, "b.json", latency=120.0),
            _sweep_file(directory, "c.json", status="failed_build", latency=None),
            _sweep_file(
                directory, "d.json", candidate={"tile_m": None}, latency=50.0
            ),
        ]
        ert = E.Ert()
        counts = E.seed_from_gemm_sweep(ert, paths)
    assert counts == {
        "added": 1,
        "merged": 1,
        "failed": 1,
        "incomplete": 1,
        "unreadable": 0,
    }
    assert len(ert) == 1


def test_seed_folds_the_repeat_into_a_spread():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        paths = [
            _sweep_file(directory, "a.json", latency=100.0),
            _sweep_file(directory, "b.json", latency=120.0),
        ]
        ert = E.Ert()
        E.seed_from_gemm_sweep(ert, paths)
        cost = ert.entries()[0].cost
    assert (cost.ns, cost.ns_min, cost.ns_max, cost.ns_samples) == (
        100_000.0,
        100_000.0,
        120_000.0,
        2,
    )


def test_seed_excludes_candidates_that_failed_their_numeric_check():
    # A latency from a candidate that failed measures the wrong thing.
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        path = _sweep_file(directory, "a.json", status="failed_precision", latency=1.0)
        ert = E.Ert()
        assert E.seed_from_gemm_sweep(ert, [path])["failed"] == 1
        assert len(ert) == 0
        # Failing direction: the same file with status passed is taken.
        ert2 = E.Ert()
        good = _sweep_file(directory, "b.json", status="passed", latency=1.0)
        assert E.seed_from_gemm_sweep(ert2, [good])["added"] == 1


def test_seed_stamps_the_condition_it_was_told():
    with tempfile.TemporaryDirectory() as tmp:
        path = _sweep_file(Path(tmp), "a.json")
        ert = E.Ert()
        E.seed_from_gemm_sweep(ert, [path], condition="npu_power_mode=turbo (test)")
    assert ert.entries()[0].cost.condition == "npu_power_mode=turbo (test)"


def test_seed_does_not_key_on_the_role():
    # Two roles at one shape and tiling are the same priced action; keying on
    # role would make the table un-reusable across the layer.
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        paths = [
            _sweep_file(directory, "a.json", role="qkv_proj", latency=100.0),
            _sweep_file(directory, "b.json", role="out_proj", latency=110.0),
        ]
        ert = E.Ert()
        counts = E.seed_from_gemm_sweep(ert, paths)
    assert counts["added"] == 1 and counts["merged"] == 1


class _FakeTransfer:
    def __init__(self, n_words, n_dims, strides, element_bytes):
        self.n_words = n_words
        self.n_dims = n_dims
        self.strides = strides
        self.element_bytes = element_bytes


def test_seed_from_transfers_counts_bytes_and_leaves_time_absent():
    # Doc 33 deferred the AIR-native bandwidth operator, so this tree holds no
    # measured shim byte rate. A seeded `ns` here would be iron's
    # cross-toolchain constant wearing a measurement's label.
    ert = E.Ert()
    added = E.seed_from_transfers(
        ert,
        [
            _FakeTransfer(8192, 4, (65536, 256, 2048, 1), 2),
            _FakeTransfer(8192, 4, (65536, 256, 2048, 1), 2),  # same descriptor
            _FakeTransfer(768, 1, (1,), 2),
            _FakeTransfer(None, 0, (), 2),  # unsizeable: excluded, not zero
        ],
        provenance="aie.air.mlir",
    )
    assert added == 2
    cost = ert.lookup(
        "shim_dma",
        "dma_transfer",
        n_words=8192,
        n_dims=4,
        strides=(65536, 256, 2048, 1),
        element_bytes=2,
    )
    assert cost.bytes == 16384 and cost.bytes_source == "counted"
    assert cost.ns is None and cost.ns_source == "absent"


def test_report_states_the_measured_counted_split():
    ert = E.Ert()
    ert.add("shim_dma", "dma_transfer", _counted(), **_dma_arguments())
    key = _gemm_key()
    ert.insert(key, _measured(1000.0, "a.json"))
    ert.observe(key, 900.0, provenance="b.json")
    text = E.report(ert)
    assert "measured" in text and "counted" in text and "modelled" in text
    assert "1 action(s) were measured more than once" in text
    assert "MINIMUM" in text
