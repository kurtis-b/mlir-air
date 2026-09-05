# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The markdown shape tables must agree with the JSON they mirror.

`details/GEMM_*.json` calls itself "the single source of truth for tile configs +
measured performance; the .md tables mirror this". Nothing enforced that, and the
same hand-sync drift has already been found once at a smaller scale (PR #64,
where three copies of the kernel-scope list disagreed).

This checks the mirror rather than generating it. Generating would rewrite the
Status column, which carries the per-row tiling rationale ("N=896 -> TILE_N=32
HERD_...") that the JSON's `used_by` does not have -- 101 of 207 rows differ
there, in both directions. So the numbers are checked and the prose is left
alone: a row may say more than the JSON, but it may not say a different number.

WHAT IS COVERED: the per-method sections, whose rows start with the shape
(`| 2048x2048x2048 | ...`). Each row is bound to ITS SECTION's method, so a
fused-cast row cannot be satisfied by the drain measurement of the same shape.

WHAT IS NOT: the "Transformer-layer execution study" sweep tables. They use a
different layout (`| seq | (MxKxN) | fused-cast | drain | direct | ...`, one row
per seq with all three methods side by side) and carry their own byte-compare
gate against the `pre-port-20260829` tag, described in that section. Extending
to them is a separate change; this file states the limit rather than implying
the whole document is covered.

    python kernel_registry/test_registry_tables.py
"""

import json
import math
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DETAILS = _HERE / "details"

# `| 2048×2048×2048 | 64/512/32/128 | **6215** | 9.7e-3 | ... |`
_ROW = re.compile(r"^\|\s*(\d+)×(\d+)×(\d+)\s*\|([^|]*)\|([^|]*)\|([^|]*)\|")
_HERD = re.compile(r"\((?:herd\s*)?\*{0,2}(\d+)\s*[×x]\s*(\d+)\*{0,2}\)")
_SECTION = re.compile(r"^###\s+(.*)$")

# Which JSON `methods` key each markdown section's rows belong to. A section
# whose heading matches nothing here contributes no rows -- see WHAT IS NOT.
_SECTION_METHOD = (
    ("--method fused-cast", "fused-cast"),
    ("--method drain", "drain"),
    ("low-precision", "direct"),
    ("external", "external"),
)


def _clean(cell):
    """Strip the presentation markers: bold for a tier winner, † for a footnote."""
    return cell.replace("**", "").replace("†", "").strip()


def _section_method(heading):
    low = heading.lower()
    for needle, method in _SECTION_METHOD:
        if needle in low:
            return method
    return None


def _split_tile(cell):
    """`16/144/48/80 (herd 4×4)` -> ("16/144/48/80", [4, 4]).

    A row carries the herd only when it overrides the file-level default, and
    the JSON records that same override per method -- so it is checked, not
    stripped, and its ABSENCE is checked too (see `_herd_matches`).
    """
    m = _HERD.search(cell)
    herd = [int(m.group(1)), int(m.group(2))] if m else None
    return _HERD.sub("", cell).strip(), herd


def _md_rows(path):
    """(method, M, K, N, tile, herd, gflops, mean_rel_L1, raw_line) per row."""
    rows = []
    method = None
    for line in path.read_text().splitlines():
        head = _SECTION.match(line)
        if head:
            method = _section_method(head.group(1))
            continue
        if method is None:
            continue
        m = _ROW.match(line)
        if not m:
            continue
        tile, herd = _split_tile(_clean(m.group(4)))
        rows.append(
            (
                method,
                int(m.group(1)),
                int(m.group(2)),
                int(m.group(3)),
                tile,
                herd,
                _clean(m.group(5)),
                _clean(m.group(6)),
                line,
            )
        )
    return rows


def _json_methods(path):
    """{(method, M, K, N): (tile, gflops, mean_rel_L1, herd)} -- keyed BY METHOD.

    Pooling every method under the shape would let a row match a sibling
    method's numbers, which is how a fused-cast row could quietly carry drain's
    throughput.
    """
    doc = json.loads(path.read_text())
    out = {}
    for shape in doc["shapes"]:
        for name, method in shape.get("methods", {}).items():
            t = method.get("tile") or {}
            tile = "/".join(
                str(t[k])
                for k in ("tile_m", "tile_k_l2", "tile_k_l1", "tile_n")
                if k in t
            )
            out[(name, shape["M"], shape["K"], shape["N"])] = (
                tile,
                method.get("gflops"),
                method.get("mean_rel_L1"),
                method.get("herd"),
            )
    return out


def _gflops_matches(cell, value):
    # "—" is a row with no recorded throughput; the JSON carries null for it.
    if cell in ("—", "-", ""):
        return value is None
    return value is not None and cell == str(value)


def _rel_l1_matches(cell, value):
    """The tables render this column rounded: JSON 0.00998 is shown as `1.0e-2`.

    Compare within half of the cell's last displayed digit rather than against
    one rendering: 0.0135 is stored just BELOW 1.35e-2, so Python renders
    `1.3e-02` while the table, rounded by hand, says `1.4e-2`. Both are correct
    roundings; a changed measurement (0.0135 -> 0.0123) still falls outside.
    """
    if not cell or value is None:
        return True  # the column is not always populated; nothing to contradict
    try:
        c = float(cell)
    except ValueError:
        return True  # prose in that column is not a number to disagree with
    v = float(value)
    if v == 0:
        return c == 0
    half = 0.5 * 10 ** (math.floor(math.log10(abs(v))) - 1)
    return abs(c - v) <= half * (1 + 1e-9)


def _herd_matches(md_herd, json_herd, row_text):
    """Absence is meaningful, with one documented exception.

    An annotation in the tile cell must match the JSON exactly, and its absence
    normally means "file-level herd" -- so deleting one fails. The exception is
    real and in the data: a few rows state the override in the Status prose
    instead (`64×12288×960` says "tile_m=16 herd_m=4" while the JSON records
    [4, 4]). Those are accepted only when the row actually mentions a herd, so
    a row with no mention at all still cannot hide an override.
    """
    if md_herd is not None or not json_herd:
        return list(md_herd or []) == list(json_herd or [])
    return "herd" in row_text.lower()


def _pairs():
    pairs = [
        (_DETAILS / f"{stem}.md", _DETAILS / f"{stem}.json")
        for stem in ("GEMM_bf16_in_bf16_out", "GEMM_bf16_in_fp32_out")
    ]
    for md, js in pairs:
        assert md.exists() and js.exists(), f"missing {md.name} / {js.name}"
    return pairs


# Rows the per-method sections carry today, per file. Exact, not a floor: a
# deleted row must fail, and `>=` would have let it through.
_EXPECTED_ROWS = {"GEMM_bf16_in_bf16_out.md": 207, "GEMM_bf16_in_fp32_out.md": 7}


def test_every_markdown_row_matches_its_own_methods_json_entry():
    """A row may say MORE than the JSON, but never a different number."""
    for md, js in _pairs():
        entries = _json_methods(js)
        for method, M, K, N, tile, herd, gflops, rel, raw in _md_rows(md):
            key = (method, M, K, N)
            got = entries.get(key)
            assert got, f"{md.name}: {M}×{K}×{N} has no `{method}` method in {js.name}"
            assert got[0] == tile, f"{md.name}: {key} tile {tile} != {got[0]}"
            assert _gflops_matches(
                gflops, got[1]
            ), f"{md.name}: {key} gflops {gflops} != {got[1]}"
            assert _rel_l1_matches(
                rel, got[2]
            ), f"{md.name}: {key} mean_rel_L1 {rel} != {got[2]}"
            assert _herd_matches(
                herd, got[3], raw
            ), f"{md.name}: {key} herd {herd} != {got[3]}"


def test_no_table_row_has_gone_missing():
    """The count is exact per file, so a deleted row fails here.

    It also fails if the row regex or the section binding stops matching, which
    is what keeps the check above from passing while inspecting nothing.
    """
    for md, _ in _pairs():
        rows = _md_rows(md)
        assert len(rows) == _EXPECTED_ROWS[md.name], (
            f"{md.name}: parsed {len(rows)} bound rows, expected "
            f"{_EXPECTED_ROWS[md.name]} — a row was added, removed, or is no "
            f"longer being parsed"
        )
        # and no row may be counted twice: each (method, shape) appears once
        keys = [(r[0], r[1], r[2], r[3]) for r in rows]
        dupes = {k for k in keys if keys.count(k) > 1}
        assert not dupes, f"{md.name}: duplicate rows for {sorted(dupes)[:3]}"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"registry table tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
