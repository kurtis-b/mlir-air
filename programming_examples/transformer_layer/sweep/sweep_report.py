# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What is done with a finished sweep: resolution assertion, and family markdown.

CONTRACT
    ``verify_resolution(shapes)`` -- for each shape, ask THE OPERATOR BUILDER
    THAT OWNS IT to resolve its GEMM configuration, and return the failures.
    Touches no hardware and reads no results tree; it reads the registry.

    ``write_family_markdown(family, results_dir, perf_iters, ...)`` -- re-render
    the family's section of both markdown pages from EVERY shape the family has,
    fail-closed on a shape whose row can no longer be re-derived or would render
    differently from the JSON entry it mirrors.

WHY THESE TWO ARE ONE MODULE, AND NOT PART OF registry_sweep.py
    Porting convention 5 caps a module at ~800 lines and ``registry_sweep.py``
    was at 866. The seam is the same mechanism-versus-consumer one the package
    already draws elsewhere (``sweep_measure.py`` measures one candidate,
    ``sweep_families.py`` says which candidates exist, ``registry_writer.py``
    writes the registry): what stays in ``registry_sweep.py`` is the ORCHESTRATOR
    -- fan candidates out to subprocesses, checkpoint each verdict, resume,
    elect winners -- and what moves here is everything DOWNSTREAM of a finished
    sweep.

    The two belong together because they answer the same question from two
    directions: after the sweep, is the registry usable, and does what it
    publishes still match what it holds. Neither needs an NPU. Neither is
    reachable from the measurement path, so a bug here cannot corrupt a
    measurement -- which is the property worth having, given how expensive a
    measurement is to repeat.

FOOTGUNS
    - ``verify_resolution`` asks the BUILDER, not ``registry_lookup``. Those are
      different questions and the weaker one passes more often; see the function's
      own docstring.
    - ``write_family_markdown`` ignores a ``--role`` / ``--seq`` filter on
      purpose. ``registry_writer.write_markdown`` replaces its whole delimited
      section, so a filtered render deletes the other roles' rows from two pages
      -- and the registry's tamper check fingerprints only the JSON, so nothing
      else would notice.
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_EXAMPLE_DIR = _HERE.parent
_PROJ_ROOT = _EXAMPLE_DIR.parent
# _EXAMPLE_DIR ahead of _PROJ_ROOT so `builders` is this example's package. See
# registry_sweep.py's copy of this preamble on why the order matters.
for _p in (str(_PROJ_ROOT), str(_EXAMPLE_DIR), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import registry_writer  # noqa: E402
from sweep_families import shapes_for_family  # noqa: E402

from registry_sweep import collect_rows, load_all_results  # noqa: E402

#: Which operator builder owns each role. Named rather than derived, because
#: this is the mapping the check is ABOUT -- a role whose builder moves must
#: move here too, and an unknown role raises rather than resolving generically.
ROLE_BUILDERS = {
    "qkv_proj": "build_qkv_proj_module",
    "ffn_up": "build_ffn_module",
    "ffn_down": "build_ffn_module",
    "o_proj": "build_mha_out_proj_module",
    # `[2026-08-23]` the qwen3_0_6b model family (doc 56 H1a): the llms builders
    # resolve through `gemm_registry_config(M, K, N)` directly -- the Q GEMM in
    # `rms_qkv_qknorm_rope_multi.build_rms_qkv_qknorm_rope_module` (K = emb,
    # N = q_dim) and the O GEMM in `qwen3_0_6b_prefill.build_o_ffn_qwen_module`
    # (K = q_dim, N = emb).
    "q_proj": "build_rms_qkv_qknorm_rope_module",
    "o_proj_q": "build_o_ffn_qwen_module",
}


def builder_gemm_spec(shape):
    """``(builder, spec)`` the operator that owns this shape resolves for it.

    Calls the PRODUCTION resolver -- ``qkv_proj.qkv_gemm_spec``,
    ``ffn.ffn_gemm_specs``, ``o_proj.o_proj_gemm_spec`` -- rather than
    re-deriving what each builder wants. Those three functions were split out of
    their builders precisely so a caller can learn what a build would resolve
    without building it, and reusing them is what keeps this check from drifting
    away from the code it is checking.

    The distinction that matters: ``gemm_config`` answers "what is this row's
    fastest high-precision method", and for the QKV shapes that is usually
    ``drain``. ``qkv_proj`` cannot build ``drain`` -- its three-way C split rides
    ``fused-cast``'s separate cast launch -- so it asks for ``fused-cast`` BY
    NAME, and a row with no ``fused-cast`` entry raises for it while resolving
    perfectly happily for the generic lookup. Checking the generic lookup
    therefore proves nothing about whether the operator can be built.
    """
    from builders.ffn import ffn_gemm_specs
    from builders.o_proj import o_proj_gemm_spec
    from builders.qkv_proj import qkv_gemm_spec

    if shape.role not in ROLE_BUILDERS:
        raise KeyError(
            f"no operator builder is recorded for role {shape.role!r}; add it "
            f"to ROLE_BUILDERS with the resolver it calls rather than letting "
            f"this shape be checked against the generic lookup"
        )
    builder = ROLE_BUILDERS[shape.role]
    if shape.role in ("q_proj", "o_proj_q"):
        return builder, _llms_gemm_registry_config(shape.M, shape.K, shape.N)
    if shape.role == "qkv_proj":
        # emb_dim is K; the builder derives N = 3 * emb_dim itself.
        return builder, qkv_gemm_spec(shape.M, shape.K)[0]
    if shape.role == "ffn_up":
        return builder, ffn_gemm_specs(shape.M, shape.K, shape.N)[0]
    if shape.role == "ffn_down":
        # The down projection is (seq, ffn_dim, emb_dim), so K and N swap
        # relative to the builder's (emb_dim, ffn_dim) argument order.
        return builder, ffn_gemm_specs(shape.M, shape.N, shape.K)[1]
    return builder, o_proj_gemm_spec(shape.M, shape.K)[0]


def _llms_gemm_registry_config(m, k, n):
    """The llms builders' resolver (`shared/builders/gemm_builder.py`), imported
    the way those builders import it: with `programming_examples/llms` on the
    path, ahead of this example's `builders` package (see the module note in
    `registry_sweep.py` on the trailing-colon PYTHONPATH)."""
    import sys
    from pathlib import Path

    llms = str(Path(__file__).resolve().parents[2] / "llms")
    if llms not in sys.path:
        sys.path.insert(0, llms)
    from shared.builders.gemm_builder import gemm_registry_config

    spec = dict(gemm_registry_config(m, k, n, "bf16", "high"))
    # The llms builders take the herd as a builder argument (8 x 4 throughout
    # qwen3_0_6b_prefill.py), not from the spec; the check needs it.
    spec.setdefault("herd_m", 8)
    spec.setdefault("herd_n", 4)
    return spec


def verify_resolution(shapes):
    """Assert every shape resolves THROUGH ITS PRODUCTION BUILDER, buildably.

    Returns the failures. Three claims, and the first is the one that a check
    against ``gemm_config`` alone silently misses:

    - The operator that owns the shape can resolve it. Each role goes through
      its own builder's spec resolver, so a builder that pins a method gets its
      pinned method looked up. ``64x768x2304`` passed a generic-lookup check
      while ``qkv_proj`` raised ``KeyError`` on it, because the row's winner was
      ``drain`` and the row carried no ``fused-cast`` entry at all.
    - The resolved row carries a herd that tiles its own shape. The example
      builders assert ``M % (tile_m * herd_m) == 0`` and
      ``N % (tile_n * herd_n) == 0`` (``bf16_in_bf16_out/run.py:62,65``) before
      they compile anything, and the ladder starts at ``M = 64`` where neither
      method's forced ``tile_m`` fits the file-level 8x4. ``resolve_gemm_spec``
      already refuses such a row; this re-checks it against the shape the sweep
      asked about rather than the one the spec was resolved for.
    - The method is one a builder can emit. That falls out of going through the
      builders: ``_spec_with_tiles`` raises on a method ``gemm_builder`` does not
      know, so a row whose high-precision winner was recorded as ``direct``
      fails here instead of inside every model that later read it.

    What it does NOT do is compile anything. Building all 36 modules costs an
    aircc run apiece; resolution is the failure this sub-phase introduces and the
    one a registry write can regress.
    """
    failures = []
    for shape in shapes:
        try:
            builder, spec = builder_gemm_spec(shape)
        except (KeyError, ValueError) as exc:
            # The lookup's message inlines every measured shape, which is 2 kB
            # per failure and drowns the list of what actually failed.
            failures.append((shape, str(exc).split(". Measured shapes")[0]))
            continue
        herd_m, herd_n = spec["herd_m"], spec["herd_n"]
        if shape.M % (spec["tile_m"] * herd_m) or shape.N % (spec["tile_n"] * herd_n):
            failures.append(
                (
                    shape,
                    f"{builder} resolves it to {spec['method']} "
                    f"tile_m={spec['tile_m']} tile_n={spec['tile_n']} at herd "
                    f"{herd_m}x{herd_n}, which does not tile "
                    f"{shape.M}x{shape.N}",
                )
            )
            continue
        print(
            f"[resolve] {shape.label} ({shape.role}, seq={shape.seq}) -> "
            f"{builder} {spec['method']} {spec['tile_m']}/{spec['tile_k_l2']}"
            f"/{spec['tile_k_l1']}/{spec['tile_n']} herd {herd_m}x{herd_n}"
        )
    return failures


def write_family_markdown(family, results_dir, perf_iters, candidate_kwargs=None):
    """Re-render a family's markdown section from EVERY shape the family has.

    ``registry_writer.write_markdown`` replaces its whole delimited section, so
    handing it a ``--role`` / ``--seq`` subset would delete the other roles' and
    sequences' rows from two pages -- and the registry's tamper check
    fingerprints only the JSON, so nothing else would notice. The JSON append
    honours the filter; the markdown is always rebuilt from the full family.

    Fail-closed twice, because the markdown MIRRORS the JSON and a page that
    quietly stops matching it is the one failure nothing else here would catch:

    - A shape this family's sweep registered in the JSON but that has no
      checkpoint left to re-derive a row from. Rendering without it deletes the
      row while the JSON goes on claiming the shape.
    - A shape whose row would render differently from the entry the JSON already
      holds -- what a re-run with a narrower ``--tile-n-options`` /
      ``--tile-k-l2-options`` grid produces, since a smaller candidate set can
      elect a different winner from the same checkpoints.

    Restoring the results directory, or re-running with the grid the rows were
    measured at, is the fix in both cases. Neither is repaired by writing.
    """
    shapes = shapes_for_family(family)
    all_records = load_all_results(shapes, results_dir, perf_iters, candidate_kwargs)
    rows = collect_rows(shapes, all_records, candidate_kwargs)
    rendered = {row["shape"].key: row for row in rows}

    # Only the entries this family's own sweep wrote: a shape some model
    # registered first is not this section's to mirror, and this sweep skips it
    # rather than measuring it.
    registered = {
        (s["M"], s["K"], s["N"]): s for s in registry_writer.load_registry()["shapes"]
    }
    owned = {
        s.key: registered[s.key]
        for s in shapes
        if s.key in registered and registered[s.key].get("used_by") == s.used_by
    }

    orphaned = [s.label for s in shapes if s.key in owned and s.key not in rendered]
    if orphaned:
        raise RuntimeError(
            f"{len(orphaned)} {family} shapes are recorded in "
            f"{registry_writer.GEMM_JSON.name} but have no measurement under "
            f"{results_dir}: {orphaned}. Writing the markdown section now would "
            f"drop their rows from the two pages while the JSON keeps claiming "
            f"them. Restore the checkpoints, or re-sweep those shapes, first."
        )

    drifted = [
        rendered[key]["shape"].label
        for key in owned
        if key in rendered and registry_writer.build_entry(rendered[key]) != owned[key]
    ]
    if drifted:
        raise RuntimeError(
            f"{len(drifted)} {family} rows would render differently from the "
            f"entry {registry_writer.GEMM_JSON.name} already holds: {drifted}. "
            f"The two pages mirror that JSON, and the JSON is append-only, so "
            f"the disagreement cannot be resolved by writing. Re-run with the "
            f"candidate grid these rows were measured at, or re-sweep them."
        )

    registry_writer.write_markdown(family, rows)
    return rows
