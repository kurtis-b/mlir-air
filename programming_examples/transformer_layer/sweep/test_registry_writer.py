# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only tests for what the registry writer will and will not do to a shape.

The rule these exist to hold: the sweep may APPEND a shape, and may add a METHOD
to a shape it wrote itself, and may do nothing else. The rows already in
`kernel_registry` are what the ten shipped LLM deployments resolve against, so a
writer that quietly re-measured one would change their behaviour without anyone
asking. `registry_writer`'s docstring states the rule; this is what checks it,
because the only other thing that would notice is the driver's byte-identity
check on a registry that has already been overwritten.

Every test runs against a TEMPORARY COPY of a synthetic registry -- `GEMM_JSON`
is repointed for the duration and restored afterwards. Nothing here touches the
real file, which is also why the guards can be provoked into failing at all.

No NPU, no XRT, no framework: pytest is not installed in this repository's
sandbox venv, so these are plain functions plus a `__main__` runner, the same
shape as `shared/infra/test_bo_pool.py`. Convention rule 11 asks for `test_*.py`
discovery, which pytest honours unchanged where it does exist.

    python3 sweep/test_registry_writer.py
"""

import json
import sys
import tempfile
import traceback
from contextlib import contextmanager
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import registry_writer  # noqa: E402
from sweep_families import ShapeSpec  # noqa: E402

# One family shape and the string the sweep writes for it. The `used_by` match
# is the whole ownership test, so it is derived from ShapeSpec rather than
# spelled out -- a change to that format must not silently make every entry
# look unowned.
OWNED = ShapeSpec(family="baseline_768", role="qkv_proj", seq=64, M=64, K=768, N=2304)
FOREIGN = ShapeSpec(
    family="baseline_768", role="o_proj", seq=2048, M=2048, K=768, N=768
)


@contextmanager
def raises(exc_type, match=None):
    """Minimal stand-in for `pytest.raises`, so the file needs no framework."""
    try:
        yield
    except exc_type as exc:
        if match is not None and match not in str(exc):
            raise AssertionError(
                f"{exc_type.__name__} raised but {match!r} not in {str(exc)!r}"
            ) from exc
        return
    raise AssertionError(f"expected {exc_type.__name__}, nothing was raised")


def method_record(method, tier, gflops, tile_n=96, herd=(1, 4)):
    """One `select_winners` method record, as `build_entry` consumes it."""
    return {
        "candidate": {
            "method": method,
            "tile_m": 64 if method != "drain" else 32,
            "tile_k_l2": 128,
            "tile_k_l1": 32,
            "tile_n": tile_n,
            "herd_m": herd[0],
            "herd_n": herd[1],
        },
        "gflops": gflops,
        "mean_rel_L1": 0.0099,
        "tier": tier,
    }


def row(shape, methods, best, note=None):
    return {"shape": shape, "methods": methods, "best": best, "note": note}


DRAIN_ONLY = {"drain": method_record("drain", "high", 945, herd=(2, 4))}
DRAIN_AND_CAST = {
    "drain": method_record("drain", "high", 945, herd=(2, 4)),
    "fused-cast": method_record("fused-cast", "high", 446),
}


_REGISTRY_HEAD = '{\n  "kernel": "GEMM",\n  "target": "npu2",\n  "herd": [\n    8,\n    4\n  ],\n  "shapes": [\n'


@contextmanager
def temp_registry(*rows):
    """A synthetic registry holding `rows`, with `GEMM_JSON` pointed at it.

    The first entry is laid down directly because `append_shapes` splices a
    leading comma and so cannot seed an empty array; every entry after it goes
    through the real append, so the layout under test is the one the writer
    itself produces.
    """
    real = registry_writer.GEMM_JSON
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "GEMM_bf16_in_bf16_out.json"
        head = registry_writer._entry_block(registry_writer.build_entry(rows[0]))
        path.write_text(_REGISTRY_HEAD + head + "\n  ]\n}")
        registry_writer.GEMM_JSON = path
        try:
            for r in rows[1:]:
                registry_writer.append_shapes([r])
            yield path
        finally:
            registry_writer.GEMM_JSON = real


def test_append_then_add_method_leaves_other_entries_byte_identical():
    seed_foreign = row(FOREIGN, DRAIN_ONLY, {"high": "drain"})
    with temp_registry(row(OWNED, DRAIN_ONLY, {"high": "drain"}), seed_foreign) as path:
        before = path.read_text()
        report = registry_writer.add_missing_methods(
            [row(OWNED, DRAIN_AND_CAST, {"high": "drain"})]
        )
        assert report["added"] == [(OWNED.key, ["fused-cast"])], report
        after = json.loads(path.read_text())
        by_key = {(s["M"], s["K"], s["N"]): s for s in after["shapes"]}
        assert sorted(by_key[OWNED.key]["methods"]) == ["drain", "fused-cast"]
        # The untouched neighbour must survive as TEXT, not merely re-parse
        # equal: the file is spliced, and a whole-file re-serialization is the
        # failure mode the splice exists to avoid.
        foreign_block = registry_writer._entry_block(
            registry_writer.build_entry(seed_foreign)
        )
        assert foreign_block in before and foreign_block in path.read_text()


def test_add_method_keeps_the_existing_methods_untouched():
    with temp_registry(row(OWNED, DRAIN_ONLY, {"high": "drain"})) as path:
        registry_writer.add_missing_methods(
            [row(OWNED, DRAIN_AND_CAST, {"high": "drain"})]
        )
        entry = json.loads(path.read_text())["shapes"][0]
        assert (
            entry["methods"]["drain"]
            == registry_writer.build_entry(row(OWNED, DRAIN_ONLY, {"high": "drain"}))[
                "methods"
            ]["drain"]
        )


def test_add_method_refuses_to_re_measure_an_existing_method():
    """The core of the append-only rule: a changed number is not an addition."""
    faster = dict(DRAIN_AND_CAST)
    faster["drain"] = method_record("drain", "high", 1200, herd=(2, 4))
    with temp_registry(row(OWNED, DRAIN_ONLY, {"high": "drain"})) as path:
        before = path.read_text()
        with raises(registry_writer.ShapeAlreadyRegistered, match="may only ADD"):
            registry_writer.add_missing_methods([row(OWNED, faster, {"high": "drain"})])
        assert path.read_text() == before


def test_add_method_will_not_touch_a_shape_another_owner_registered():
    """A shape a shipped model registered is unreachable through this path."""
    seed = row(OWNED, DRAIN_ONLY, {"high": "drain"})
    with temp_registry(seed) as path:
        before = path.read_text()
        # Same (M, K, N), different owner: what a family sharing a shape with a
        # deployment looks like.
        other = ShapeSpec(
            family="baseline_1024", role="ffn_up", seq=64, M=64, K=768, N=2304
        )
        assert other.used_by != OWNED.used_by
        report = registry_writer.add_missing_methods(
            [row(other, DRAIN_AND_CAST, {"high": "drain"})]
        )
        assert report["not_owned"] == [OWNED.key] and not report["added"]
        assert path.read_text() == before


def test_add_method_is_a_no_op_when_nothing_is_missing():
    with temp_registry(row(OWNED, DRAIN_AND_CAST, {"high": "drain"})) as path:
        before = path.read_text()
        report = registry_writer.add_missing_methods(
            [row(OWNED, DRAIN_AND_CAST, {"high": "drain"})]
        )
        assert report["unchanged"] == [OWNED.key] and not report["added"]
        assert path.read_text() == before


def test_add_method_refuses_a_shape_that_is_not_registered_at_all():
    with temp_registry(row(FOREIGN, DRAIN_ONLY, {"high": "drain"})) as path:
        before = path.read_text()
        with raises(KeyError, match="append it instead"):
            registry_writer.add_missing_methods(
                [row(OWNED, DRAIN_AND_CAST, {"high": "drain"})]
            )
        assert path.read_text() == before


def test_append_still_refuses_a_registered_shape():
    """`add_missing_methods` must not have opened a second way in through here."""
    with temp_registry(row(OWNED, DRAIN_ONLY, {"high": "drain"})) as path:
        before = path.read_text()
        with raises(registry_writer.ShapeAlreadyRegistered, match="append-only"):
            registry_writer.append_shapes(
                [row(OWNED, DRAIN_AND_CAST, {"high": "drain"})]
            )
        assert path.read_text() == before


def test_note_and_best_follow_the_added_method():
    """Both are derived from the method set, so both may move; nothing else may."""
    stale = row(
        OWNED,
        DRAIN_ONLY,
        {"high": "drain"},
        note="fused-cast: no candidate passed (failed_precision).",
    )
    fixed = row(OWNED, DRAIN_AND_CAST, {"high": "drain", "low": "direct"})
    with temp_registry(stale) as path:
        registry_writer.add_missing_methods([fixed])
        entry = json.loads(path.read_text())["shapes"][0]
        assert "_note" not in entry
        assert entry["best"] == {"high": "drain", "low": "direct"}
        assert entry["used_by"] == OWNED.used_by


def _main():
    """Run every test_* in definition order and report. Exit 1 on any failure."""
    tests = [
        (n, o) for n, o in globals().items() if n.startswith("test_") and callable(o)
    ]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS {name}")
        except Exception:
            failed.append(name)
            print(f"  FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
