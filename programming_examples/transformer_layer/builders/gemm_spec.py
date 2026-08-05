# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Registry-resolved GEMM build recipes for the transformer-layer builders.

CONTRACT
    ``resolve_gemm_spec(m, k, n)`` returns ``llms/shared``'s
    ``gemm_registry_config`` spec -- method, ``tile_m`` / ``tile_k_l2`` /
    ``tile_k_l1`` / ``tile_n``, symbol suffix, object name, build kwargs --
    with the HERD those tiles were measured at merged in as ``herd_m`` /
    ``herd_n``, and checks the pair actually tiles this shape.

    ``resolve_gemm_spec(..., method="fused-cast")`` returns that method's
    measured row instead of the shape's fastest one, for a builder whose
    STRUCTURE fixes the method (see below).

    ``spec_herd(spec, herd_m, herd_n)`` is what a builder calls to decide the
    herd for one GEMM: an explicit caller override first, the spec's own herd
    second. Pass it straight to ``_build_gemm_module``.

WHY A BUILDER MAY PIN THE METHOD
    A registry row's ``best.high`` is the fastest high-precision method for that
    shape and nothing else. The two high-precision methods are not
    interchangeable at the module level: ``fused-cast`` accumulates into an f32
    C scratch and casts in a SECOND launch, ``drain`` casts inside the one GEMM
    launch and exposes no scratch at all. ``qkv_proj.py`` folds its three-way C
    split into that second launch, so it can only build ``fused-cast`` -- and
    the sequence ladder is full of shapes where ``drain`` is the faster method
    and therefore the row's winner.

    Pinning the method here keeps that builder registry-driven: the tiles and
    the herd still come from what was MEASURED for ``fused-cast`` at this exact
    shape. It is not a fallback and it guesses nothing --
    a shape with no ``fused-cast`` row raises, exactly as an unmeasured shape
    does. The alternative, taking ``best.high`` and rejecting it afterwards,
    fails on a shape the registry can in fact serve.

WHY THIS WRAPPER EXISTS
    ``gemm_registry_config`` copies out the method spec and the four tile sizes
    and stops there, so the per-row ``herd`` the Phase C4 sweep records never
    reaches a builder. Every builder here therefore defaulted to the full 8x4
    array -- right for every shape the shipped deployments use (all ``M =
    2048``) and wrong at the short end of the study's sequence ladder. At
    ``M = 64`` the drain method's forced ``tile_m = 32`` admits at most
    ``herd_m = 2``, and ``matrix_multiplication/bf16_in_bf16_out/run.py:62``
    asserts ``M % (tile_m * herd_m) == 0`` before it compiles anything.

    The merge lives here rather than inside ``gemm_registry_config`` because
    ``llms/shared/`` is off limits to this study: editing it puts all ten
    shipped deployments' ``make verify`` on the line (``builders/__init__.py``).
    The consequence to know is that a model resolving through
    ``gemm_registry_config`` still receives no herd. That is harmless today --
    every model-registered shape is ``M = 2048`` at the file-level 8x4 -- and
    the day it stops being true it fails at build time with the divisibility
    assert, not with wrong numbers.

FOOTGUNS
    - The herd is NOT a free knob. It is part of what was measured, so
      overriding it invalidates the tiling without invalidating the lookup.
      The override parameters exist for experiments, not for tuning.
    - An INJECTED spec (the ``gemm_spec_fn`` escape hatch every builder here
      ships) carries no herd unless its author put one in, so ``spec_herd``
      falls back to 8x4 for it. An injected spec for a short-``M`` shape has to
      supply ``herd_m`` / ``herd_n`` itself -- nothing here can derive them,
      because the point of the hatch is that nothing measured the shape.
    - ``resolve_gemm_spec`` raises on a registry row whose herd does not tile
      its own shape. That is a corrupt row, not a caller error; the alternative
      is ``run.py``'s bare ``assert (64, 32, 8)`` several frames deeper, which
      names neither the shape nor the file to fix.
    - A pinned ``method`` skips the row's tier check, because it names the
      method directly rather than asking for a tier's winner. Pin only a
      high-precision method (``fused-cast`` / ``drain``); pinning ``direct``
      would silently drop a caller that asked for ``precision="high"`` onto the
      low tier.
"""

import os
import sys

_PROJ_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
for _p in (_PROJ_ROOT, os.path.join(_PROJ_ROOT, "llms")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: The array every shape measured before per-row herds existed used, and the
#: fallback for a spec that carries no herd of its own (an injected one).
DEFAULT_HERD = (8, 4)


def _check_gemm_herd(m, n, spec, herd_m, herd_n):
    """Raise unless ``(herd_m, herd_n)`` tiles ``m`` x ``n`` with these tiles.

    The same two conditions ``bf16_in_bf16_out/run.py`` asserts, checked at the
    seam that chose the numbers so the message can say which row is wrong.
    """
    if m % (spec["tile_m"] * herd_m):
        raise ValueError(
            f"registry config for M={m} is not buildable: "
            f"M % (tile_m={spec['tile_m']} * herd_m={herd_m}) != 0. "
            f"The herd a row was measured at is part of the row; a short-M "
            f"shape needs the smaller herd recorded with it."
        )
    if n % (spec["tile_n"] * herd_n):
        raise ValueError(
            f"registry config for N={n} is not buildable: "
            f"N % (tile_n={spec['tile_n']} * herd_n={herd_n}) != 0."
        )


def resolve_gemm_spec(m, k, n, output_dtype="bf16", precision="high", method=None):
    """The registry's full build recipe for one GEMM, herd included.

    ``method=None`` takes the tier's winner, which is what a builder that can
    wire either method wants. Naming a method takes that method's measured row
    instead -- for a builder whose module structure fixes it, see the docstring
    above.

    Raises (via the lookup) on an unmeasured shape, or on a shape with no row
    for the named method, rather than guessing -- that is the whole point of the
    registry, see ``registry_lookup.py``.
    """
    from kernel_registry.registry_lookup import gemm_config, gemm_config_method

    # `_spec_with_tiles` is the same merge `gemm_registry_config` performs one
    # line further in, including its tile_m-agrees-with-the-method assert. It is
    # reached directly only because `gemm_registry_config` hardcodes the tier
    # winner as the method, and `llms/shared/` is off limits to this study.
    from shared.builders.gemm_builder import _spec_with_tiles, gemm_registry_config

    if method is None:
        spec = dict(gemm_registry_config(m, k, n, output_dtype, precision))
        cfg = gemm_config(m, k, n, output_dtype, precision)
    else:
        cfg = gemm_config_method(m, k, n, output_dtype, method, precision)
        spec = dict(_spec_with_tiles(cfg["method"], cfg["tile"]))
        # What the row would have chosen on its own. Recorded so a pinning
        # builder's run log can say it is not using the fastest method measured
        # for this shape, rather than leaving that cost invisible. Reporting
        # only, so a row with no winner for the tier leaves it unset instead of
        # turning a successful resolution into a failed one.
        try:
            winner = gemm_config(m, k, n, output_dtype, precision)["method"]
        except KeyError:
            winner = None
        spec["tier_winner"] = winner
    spec["herd_m"], spec["herd_n"] = cfg["herd"]
    _check_gemm_herd(m, n, spec, spec["herd_m"], spec["herd_n"])
    return spec


def spec_herd(spec, herd_m=None, herd_n=None):
    """``(herd_m, herd_n)`` to build one GEMM with: override, then spec, then 8x4."""
    return (
        herd_m if herd_m is not None else spec.get("herd_m", DEFAULT_HERD[0]),
        herd_n if herd_n is not None else spec.get("herd_n", DEFAULT_HERD[1]),
    )


def describe_herd(spec, herd_m=None, herd_n=None):
    """``8x4``-style label for a run log, after overrides are applied."""
    return "{}x{}".format(*spec_herd(spec, herd_m, herd_n))
