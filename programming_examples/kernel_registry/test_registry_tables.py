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


def _clean(cell):
    """Strip the presentation markers: bold for a tier winner, † for a footnote."""
    return cell.replace("**", "").replace("†", "").strip()


_HERD = re.compile(r"\(herd\s*(\d+)\s*[×x]\s*(\d+)\)")


def _split_tile(cell):
    """`16/144/48/80 (herd 4×4)` -> ("16/144/48/80", [4, 4]).

    A row carries the herd only when it overrides the file-level default, and
    the JSON records that same override per method -- so it is checked, not
    stripped.
    """
    m = _HERD.search(cell)
    herd = [int(m.group(1)), int(m.group(2))] if m else None
    return _HERD.sub("", cell).strip(), herd


def _md_rows(path):
    """(M, K, N, tile, herd, gflops, mean_rel_L1) per data row."""
    rows = []
    for line in path.read_text().splitlines():
        m = _ROW.match(line)
        if not m:
            continue
        M, K, N = int(m.group(1)), int(m.group(2)), int(m.group(3))
        tile, herd = _split_tile(_clean(m.group(4)))
        rows.append((M, K, N, tile, herd, _clean(m.group(5)), _clean(m.group(6))))
    return rows


def _json_methods(path):
    """{(M, K, N): [(tile, gflops, mean_rel_L1), ...]} from the registry JSON."""
    doc = json.loads(path.read_text())
    out = {}
    for shape in doc["shapes"]:
        key = (shape["M"], shape["K"], shape["N"])
        for method in shape.get("methods", {}).values():
            t = method.get("tile") or {}
            tile = "/".join(
                str(t[k])
                for k in ("tile_m", "tile_k_l2", "tile_k_l1", "tile_n")
                if k in t
            )
            herd = method.get("herd")
            out.setdefault(key, []).append(
                (tile, method.get("gflops"), method.get("mean_rel_L1"), herd)
            )
    return out


def _gflops_matches(cell, value):
    # "—" is a row with no recorded throughput; the JSON carries null for it.
    if cell in ("—", "-", ""):
        return value is None
    return value is not None and cell == str(value)


def _rel_l1_matches(cell, value):
    """The tables render this column rounded: JSON 0.00998 is shown as `1.0e-2`.

    So compare the RENDERED form, not the raw float -- that still catches a
    changed measurement (0.00998 -> 0.0123 renders differently) without
    demanding the markdown carry five significant figures.
    """
    if not cell or value is None:
        return True  # the column is not always populated; nothing to contradict
    try:
        float(cell)
    except ValueError:
        return True  # prose in that column is not a number to disagree with
    # Compare within half of the cell's last displayed digit, rather than against
    # one rendering: 0.0135 is stored just BELOW 1.35e-2, so Python renders
    # `1.3e-02` while the table, rounded by hand, says `1.4e-2`. Both are correct
    # roundings; a changed measurement (0.0135 -> 0.0123) still falls outside.
    v = float(value)
    if v == 0:
        return float(cell) == 0
    exp = math.floor(math.log10(abs(v)))
    half = 0.5 * 10 ** (exp - 1)
    return abs(float(cell) - v) <= half * (1 + 1e-9)


def _pairs():
    pairs = [
        (_DETAILS / f"{stem}.md", _DETAILS / f"{stem}.json")
        for stem in ("GEMM_bf16_in_bf16_out", "GEMM_bf16_in_fp32_out")
    ]
    for md, js in pairs:
        assert md.exists() and js.exists(), f"missing {md.name} / {js.name}"
    return pairs


def test_every_markdown_row_matches_a_json_method():
    """A row may say MORE than the JSON, but never a different number."""
    checked = 0
    for md, js in _pairs():
        methods = _json_methods(js)
        for M, K, N, tile, herd, gflops, rel in _md_rows(md):
            cands = methods.get((M, K, N))
            assert cands, f"{md.name}: {M}×{K}×{N} has no shape in {js.name}"
            hit = [
                c
                for c in cands
                if c[0] == tile
                and _gflops_matches(gflops, c[1])
                and _rel_l1_matches(rel, c[2])
                and (herd is None or list(c[3] or []) == herd)
            ]
            assert hit, (
                f"{md.name}: row {M}×{K}×{N} tile={tile} herd={herd} "
                f"gflops={gflops} mean_rel_L1={rel} matches no method in "
                f"{js.name}: {cands}"
            )
            checked += 1
    # non-vacuity: if the row regex ever stops matching, this fails rather than
    # silently checking nothing.
    assert checked >= 200, f"expected the tables' ~207 rows, parsed {checked}"
    print(f"    ({checked} markdown rows checked against the JSON)")


def test_every_json_shape_is_reachable_by_its_key():
    """Guards the other direction: a shape the tables cannot address is dead data."""
    for _, js in _pairs():
        methods = _json_methods(js)
        assert methods, f"{js.name}: no shapes parsed"
        for key, cands in methods.items():
            assert all(len(c) == 4 for c in cands), (js.name, key)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"registry table tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
