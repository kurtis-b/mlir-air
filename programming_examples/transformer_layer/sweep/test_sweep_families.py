# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only tests that the sweep plans configurations the builders can build.

`sweep_families.METHODS` duplicates two things it does not own: the `tile_m` each
GEMM method forces, and -- through `gemm_variant_names` -- how a configuration's
micro-kernel object is named. It duplicates them because the sweep plans without
importing the builders, and because it also knows `direct`, which
`gemm_method_spec` deliberately does not.

A duplicate is only safe while something checks it. If the two tables drift, the
sweep plans candidates the builders reject: `_spec_with_tiles` asserts the
registry's `tile_m` equals the method's, so a drifted row does not mis-tune a
model, it aborts one at build time -- hours into a gate, several frames inside
aircc. These tests are the cheap place for that to fail instead.

`sweep_families.py`'s own docstring names this file as the check, so the two
statements are each other's evidence.

No NPU, no XRT, no framework: pytest is not installed in this repository's
sandbox venv, so these are plain functions plus a `__main__` runner, the same
shape as `test_registry_writer.py`.

    python3 sweep/test_sweep_families.py
"""

import sys
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_EXAMPLE_DIR = _HERE.parent
_PROJ_ROOT = _EXAMPLE_DIR.parent
for _p in (str(_PROJ_ROOT), str(_PROJ_ROOT / "llms"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared.builders.gemm_builder import (  # noqa: E402
    _METHOD_TILE_M,
    gemm_method_spec,
    gemm_variant_names,
    with_tile_n,
)

from sweep_families import (  # noqa: E402
    METHODS,
    TILE_N_PREFERENCE,
    candidates_for_shape,
    shapes_for_family,
)


def test_the_duplicated_tile_m_table_agrees_with_the_authoritative_one():
    """Every method `gemm_builder` knows has the same `tile_m` here.

    One direction only. `METHODS` additionally knows `direct`, which
    `gemm_method_spec` raises on because there is no external object for it to
    name -- that asymmetry is deliberate and pre-existing, so it is not checked
    as a mismatch. What is checked is that no method appears in both with two
    different `tile_m`, which is the drift that breaks a build.
    """
    for method, tile_m in _METHOD_TILE_M.items():
        assert method in METHODS, (
            f"{method!r} is a method gemm_builder can build and sweep_families "
            f"does not know; the sweep would never plan a candidate for it"
        )
        assert METHODS[method]["tile_m"] == tile_m, (
            f"tile_m for {method!r} is {METHODS[method]['tile_m']} in "
            f"sweep_families.METHODS and {tile_m} in the authoritative "
            f"gemm_builder._METHOD_TILE_M"
        )


def test_the_two_high_precision_methods_are_in_the_high_tier():
    """A method `gemm_method_spec` builds is a high-precision method.

    The tier is the sweep's own column, but it is not free: `gemm_builder` only
    mints specs for the two methods that f32-accumulate across the whole K
    reduction and cast once. A method landing in the `low` tier here while
    `gemm_method_spec` accepts it would mean the sweep electing it as a `high`
    winner for a shape whose caller asked for high precision.
    """
    for method in _METHOD_TILE_M:
        assert METHODS[method]["tier"] == "high", method
        assert METHODS[method]["high_precision"] is True, method


def test_every_planned_candidate_mints_a_spec():
    """Each candidate the sweep would measure resolves to a buildable spec.

    The end-to-end version of the table check: for every candidate of every
    baseline_768 shape, ask `gemm_method_spec` for the method at that candidate's
    `tile_n` and require the `tile_m` it forces to be the one the candidate
    carries. `direct` candidates are skipped -- they have no external object.

    This is what would have caught a drifted `METHODS` row, and it is also what
    would catch a candidate generator that stopped putting `tile_m` in the
    candidate at all.
    """
    checked = 0
    for shape in shapes_for_family("baseline_768"):
        for candidate in candidates_for_shape(shape):
            if candidate.method not in _METHOD_TILE_M:
                continue
            spec = gemm_method_spec(candidate.method, candidate.tile_n)
            assert spec["tile_m"] == candidate.tile_m, (
                f"{shape.label} {candidate.label}: candidate tile_m "
                f"{candidate.tile_m} != method tile_m {spec['tile_m']}"
            )
            assert spec["tile_n"] == candidate.tile_n
            checked += 1
    assert checked > 0, "no external-method candidate was planned at all"


def test_each_tile_n_the_sweep_may_plan_gets_its_own_object():
    """No two `tile_n` the sweep can elect share a micro-kernel object name.

    This is the Phase E1 property, checked at the planning end rather than the
    building end. `mm_aie2p.cc` bakes `tile_n` in as `-DDIM_N`, so two
    configurations of one method at different `tile_n` are two different
    micro-kernels; naming them the same means one overwrites the other and aiecc
    links whichever landed last, silently. `sweep_families` may plan seven
    `tile_n` values for one method, so this is not a hypothetical.
    """
    for method, tile_m in _METHOD_TILE_M.items():
        names = {}
        for tile_n in TILE_N_PREFERENCE:
            sym_suffix, obj = gemm_variant_names(tile_m, tile_n)
            assert obj not in names, (
                f"{method} tile_n={tile_n} and tile_n={names.get(obj)} both name "
                f"{obj}; one would overwrite the other"
            )
            names[obj] = tile_n
            assert sym_suffix in obj, (sym_suffix, obj)


def test_retiling_a_spec_moves_its_object_its_suffix_and_its_build_kwargs():
    """`with_tile_n` moves all three views of a variant's identity together.

    Several callers override `tile_n` after resolving a spec, because the
    registry's `tile_n` for a narrow N can be numerically wrong and the builder
    pads N out to admit a wider one. A bare `spec["tile_n"] = ...` used to be
    harmless and is not any more: it leaves the module asking for an object
    nobody compiles. This is what those callers use instead.
    """
    spec = gemm_method_spec("drain", 32)
    retiled = with_tile_n(spec, 128)
    assert retiled["tile_n"] == 128
    assert retiled["sym_suffix"] == "_m32n128"
    assert retiled["obj"] == "mm_m32n128.o"
    assert retiled["build_kwargs"]["sym_suffix"] == "_m32n128"
    assert retiled["build_kwargs"]["link_with_name"] == "mm_m32n128.o"
    # The method's own structure is untouched, and the original is not mutated.
    assert retiled["tile_m"] == spec["tile_m"] == 32
    assert retiled["needs_f32_scratch"] is False
    assert spec["tile_n"] == 32
    assert spec["obj"] == "mm_m32n32.o"


def test_a_spec_cannot_be_minted_without_a_tile_n():
    """`gemm_method_spec` requires `tile_n`, and raises on an unknown method.

    Required rather than defaulted on purpose: a caller that does not know its
    `tile_n` cannot know which object to link, and a default would put every such
    caller back on one shared name -- which is the collision Phase E1 removed.
    """
    try:
        gemm_method_spec("drain")
    except TypeError:
        pass
    else:
        raise AssertionError("gemm_method_spec accepted a call with no tile_n")

    try:
        gemm_method_spec("direct", 128)
    except ValueError:
        pass
    else:
        raise AssertionError("gemm_method_spec accepted 'direct'")


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
    print(f"\nsweep families tests: {len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
