# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Append swept shapes into ``kernel_registry``: the JSON, and both markdown pages.

CONTRACT
    ``append_shapes(rows)`` adds shapes that are NOT already in
    ``details/GEMM_bf16_in_bf16_out.json`` and raises on any that are.
    ``write_markdown(family, rows)`` mirrors the same rows into
    ``details/GEMM_bf16_in_bf16_out.md`` and ``supported_kernels.md``.

    A ``row`` is ``{"shape": ShapeSpec, "methods": {name: record}, "best": {...},
    "note": str|None}`` as assembled by ``registry_sweep.select_winners``.

APPEND ONLY -- THE FOOTGUN THIS MODULE EXISTS TO PREVENT
    The shapes already in the registry are what the ten shipped LLM deployments
    resolve against. Re-measuring one into a different winner would change their
    behaviour without anyone asking, so this module will not do it: a shape whose
    ``(M, K, N)`` is already present raises ``ShapeAlreadyRegistered`` and nothing
    is written.

    The JSON is edited as TEXT, not re-serialized. Splicing new entries in before
    the closing bracket makes every pre-existing byte identical by construction,
    which is what the sub-phase's objective check verifies. Re-serializing would
    round-trip 33 entries through ``json.dumps`` and depend on that round trip
    being exact -- true today, and a silent whole-file diff the day it is not.
    ``_verify_append`` re-reads and re-parses afterwards rather than trusting it.

    The tamper check that fingerprints ``details/*.json`` does NOT cover the two
    markdown pages, so nothing but this module's own care keeps them honest.
    Every markdown write is delimited by a named HTML comment and replaces what
    is between the delimiters, so re-running the sweep updates its own section
    and cannot touch a hand-written one.

WHAT ``herd`` IS DOING IN THE METHOD ENTRIES
    The JSON's top-level ``herd`` records 8x4 because every shape registered
    before this sweep used the full array. The sequence ladder starts at 64, and
    ``M % (tile_m * herd_m) == 0`` cannot hold at 64 with ``herd_m = 8`` for
    EITHER method's forced ``tile_m``. So the short-sequence rows carry a
    per-method ``herd`` that overrides the file-level default, and a ``_note``
    saying so.

    ``registry_lookup.gemm_config`` copies out ``method``/``tile``/``gflops``/
    ``mean_rel_L1`` and ignores everything else, so the extra key is invisible to
    every existing consumer. It is recorded anyway because without it the row
    cannot be reproduced -- and a consumer that assumed 8x4 at M=64 would hit the
    example's own divisibility assert, which fails loudly rather than quietly.
"""

import json
import re
from pathlib import Path

_REGISTRY = Path(__file__).resolve().parent.parent.parent / "kernel_registry"
GEMM_JSON = _REGISTRY / "details" / "GEMM_bf16_in_bf16_out.json"
GEMM_MD = _REGISTRY / "details" / "GEMM_bf16_in_bf16_out.md"
INDEX_MD = _REGISTRY / "supported_kernels.md"

# The three-space-indented tail every entry is spliced in front of. Asserted, not
# assumed: if the file's shape ever changes this fails before it writes.
_JSON_TAIL = "\n  ]\n}"

# Entries live at indent 4 inside "shapes": [ ... ].
_ENTRY_INDENT = "    "

DEFAULT_HERD = [8, 4]


class ShapeAlreadyRegistered(Exception):
    """Raised instead of overwriting a shape the registry already holds."""


def load_registry():
    return json.loads(GEMM_JSON.read_text())


def registered_shape_keys():
    return {(s["M"], s["K"], s["N"]) for s in load_registry()["shapes"]}


def _round_rel(x):
    """mean_rel_L1 to two significant figures, matching every existing entry."""
    return float(f"{x:.2g}")


def build_entry(row):
    """One registry JSON entry from one swept shape."""
    shape = row["shape"]
    methods = {}
    for name, record in row["methods"].items():
        entry = {
            "tile": (
                record["candidate"]["tile"]
                if "tile" in record["candidate"]
                else {
                    "tile_m": record["candidate"]["tile_m"],
                    "tile_k_l2": record["candidate"]["tile_k_l2"],
                    "tile_k_l1": record["candidate"]["tile_k_l1"],
                    "tile_n": record["candidate"]["tile_n"],
                }
            ),
            "gflops": int(round(record["gflops"])),
            "mean_rel_L1": _round_rel(record["mean_rel_L1"]),
            "tier": record["tier"],
        }
        herd = [record["candidate"]["herd_m"], record["candidate"]["herd_n"]]
        if herd != DEFAULT_HERD:
            entry["herd"] = herd
        methods[name] = entry

    entry = {
        "M": shape.M,
        "K": shape.K,
        "N": shape.N,
        "used_by": shape.used_by,
    }
    if row.get("note"):
        entry["_note"] = row["note"]
    entry["methods"] = methods
    entry["best"] = dict(row["best"])
    return entry


def _verify_append(before_shapes, new_keys):
    """Re-read the file and prove the append did only what it claimed."""
    after = load_registry()
    after_by_key = {(s["M"], s["K"], s["N"]): s for s in after["shapes"]}
    for shape in before_shapes:
        key = (shape["M"], shape["K"], shape["N"])
        if after_by_key.get(key) != shape:
            raise RuntimeError(
                f"append corrupted the pre-existing shape {key}; the registry "
                f"file has been left in a state that must be reverted by hand"
            )
    missing = [k for k in new_keys if k not in after_by_key]
    if missing:
        raise RuntimeError(f"append did not land these shapes: {missing}")


def append_shapes(rows):
    """Splice new shapes into the registry JSON. Returns the keys written."""
    text = GEMM_JSON.read_text()
    if not text.endswith(_JSON_TAIL):
        raise RuntimeError(
            f"{GEMM_JSON} does not end with the expected shapes-array tail; "
            f"refusing to splice into a file whose layout changed"
        )

    before = load_registry()
    before_shapes = [dict(s) for s in before["shapes"]]
    existing = {(s["M"], s["K"], s["N"]) for s in before_shapes}

    entries, keys = [], []
    for row in rows:
        key = row["shape"].key
        if key in existing:
            raise ShapeAlreadyRegistered(
                f"{key} is already in the registry. This sweep is append-only: "
                f"re-measuring a registered shape would change what the shipped "
                f"deployments resolve, which is not this sub-phase's to do."
            )
        if key in keys:
            raise ValueError(f"{key} appears twice in one append")
        entries.append(build_entry(row))
        keys.append(key)

    if not entries:
        return []

    blocks = []
    for entry in entries:
        body = json.dumps(entry, indent=2)
        blocks.append(
            ",\n" + "\n".join(_ENTRY_INDENT + line for line in body.splitlines())
        )

    GEMM_JSON.write_text(text[: -len(_JSON_TAIL)] + "".join(blocks) + _JSON_TAIL)
    _verify_append(before_shapes, keys)
    return keys


def _delimited(name, body):
    return f"<!-- BEGIN {name} -->\n{body}\n<!-- END {name} -->\n"


def _splice_markdown(path, name, body, anchor):
    """Replace this sweep's delimited section, or insert it before ``anchor``."""
    text = path.read_text()
    block = _delimited(name, body)
    pattern = re.compile(
        rf"<!-- BEGIN {re.escape(name)} -->.*?<!-- END {re.escape(name)} -->\n",
        re.DOTALL,
    )
    if pattern.search(text):
        path.write_text(pattern.sub(block, text))
        return
    if anchor not in text:
        raise RuntimeError(f"anchor {anchor!r} not found in {path}")
    path.write_text(text.replace(anchor, block + "\n" + anchor, 1))


def _fmt_gflops(row, method):
    record = row["methods"].get(method)
    if record is None:
        return "❌" if row["failed"].get(method) else "—"
    value = int(round(record["gflops"]))
    # Bold marks the faster of the two HIGH-precision methods, which is what
    # `best.high` records and what the existing index table bolds. `direct` is
    # the only low-tier method, so bolding it would mark nothing.
    is_best = record["tier"] == "high" and row["best"].get("high") == method
    return f"**{value}**" if is_best else str(value)


def _fmt_exp(value):
    """`9.4e-3`, not python's `9.4e-03` -- every existing registry row's style."""
    return f"{value:.1e}".replace("e-0", "e-").replace("e+0", "e+")


def _fmt_tier_rel(row, tier):
    for method, record in row["methods"].items():
        if record["tier"] == tier and row["best"].get(tier) == method:
            return _fmt_exp(record["mean_rel_L1"])
    return "—"


def _fmt_best_config(row):
    method = row["best"].get("high") or row["best"].get("low")
    if method is None:
        return "—"
    candidate = row["methods"][method]["candidate"]
    tile = (
        f"{candidate['tile_m']}/{candidate['tile_k_l2']}"
        f"/{candidate['tile_k_l1']}/{candidate['tile_n']}"
    )
    herd = f"{candidate['herd_m']}×{candidate['herd_n']}"
    return f"{tile} ({herd})"


def _status(row):
    if row["best"].get("high"):
        return "✅"
    if row["best"].get("low"):
        return "⚠️"
    return "❌"


def _role_table(rows):
    lines = [
        "| seq | (M×K×N) | fused-cast | drain | direct | best tile (m/kl2/kl1/n) (herd) "
        "| mean_rel_L1 (high / low) | Status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        shape = row["shape"]
        lines.append(
            f"| {shape.seq} | {shape.M}×{shape.K}×{shape.N} "
            f"| {_fmt_gflops(row, 'fused-cast')} | {_fmt_gflops(row, 'drain')} "
            f"| {_fmt_gflops(row, 'direct')} | {_fmt_best_config(row)} "
            f"| {_fmt_tier_rel(row, 'high')} / {_fmt_tier_rel(row, 'low')} "
            f"| {_status(row)} |"
        )
    return "\n".join(lines)


def _family_body(family, rows, heading_level, preamble):
    by_role = {}
    for row in rows:
        by_role.setdefault(row["shape"].role, []).append(row)
    hashes = "#" * heading_level
    parts = [f"{hashes} Transformer-layer execution study — `{family}` sweep", ""]
    parts.append(preamble)
    parts.append("")
    for role, role_rows in by_role.items():
        head = role_rows[0]["shape"]
        parts.append(f"**`{role}`** — `K = {head.K}` → `N = {head.N}`")
        parts.append("")
        parts.append(_role_table(role_rows))
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


_DETAIL_PREAMBLE = """\
Written by `programming_examples/transformer_layer/sweep/registry_sweep.py`, which
measures every candidate tiling for a shape and keeps the fastest that passes. **Bold**
= the winner recorded in `best` for that tier. `—` = no legal candidate for that method
at this shape; `❌` = every candidate for it failed.

Two things differ from the model-deployment rows above, both forced by the sequence
ladder starting at 64:

- **`herd` is per-row.** `M % (tile_m × herd_m) == 0` cannot hold at `M = 64` with
  `herd_m = 8` for either method's forced `tile_m`, so short-sequence rows carry a
  per-method `herd` in the JSON that overrides the file-level `8×4`. The herd used is in
  the table's tile column.
- **The high-precision `atol` is carried forward at constant strictness, not held
  constant.** The harness scales its inputs by `1/sqrt(K)`, so the output magnitude —
  and the absolute error of a fixed-relative-precision datapath with it — goes as
  `K^-1/2`. The published `atol = 1.5e-3` is that rule evaluated at `K = 8192`; at this
  family's `K = 768` it is a 3.3× *tightening*, which is exactly the "harness tolerance
  edge, not a datapath failure" already recorded for Qwen3-0.6B Gate/Up and
  Qwen2.5-0.5B Gate/Up above. The sweep therefore checks the high tier at
  `1.5e-3 × sqrt(8192 / K)`, which reproduces the registry's own ~2.5× design margin at
  every K (measured: 2.5× at K=8192, 2.9× at K=3072, 2.8× at K=768). `rtol` is the
  canonical `1.6e-2` unchanged, the low tier's `4e-3` is unchanged, and every run must
  additionally land inside its tier's `mean_rel_L1` band — a scale-free check the
  example harness does not make at all.\
"""

_INDEX_PREAMBLE = """\
The projection GEMMs the transformer-layer execution study's case matrix needs, swept
across the full 9-point sequence ladder. Full per-candidate detail, and why the
high-precision `atol` is K-scaled here, in
[`details/GEMM_bf16_in_bf16_out.md`](details/GEMM_bf16_in_bf16_out.md).\
"""


def write_markdown(family, rows):
    """Mirror the family's rows into both markdown pages."""
    name = f"transformer-layer-sweep {family}"
    _splice_markdown(
        GEMM_MD,
        name,
        _family_body(family, rows, 3, _DETAIL_PREAMBLE),
        "## How to reproduce",
    )
    _splice_markdown(
        INDEX_MD,
        name,
        _family_body(family, rows, 3, _INDEX_PREAMBLE),
        "---\n\n## GEMV — tested shapes",
    )
