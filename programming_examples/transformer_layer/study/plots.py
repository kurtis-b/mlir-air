# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Render a ladder results tree as figures. The graphical half of ``ladder_report.py``.

    python3 study/plots.py <ladder-results-dir> [--out DIR] [--modes a,b]

CONTRACT
    Four figure families over a tree of ``<mode>.csv``, all on the CURRENT
    schema rather than iron's column names:

      ``latency``          avg latency against ``seq_len``, min-max band, fitted
                           log-log slope in the legend.
      ``dram``             ``bytes_transferred`` against ``seq_len`` -- the
                           taxonomy's DRAM-traffic axis (doc 03).
      ``decomposition``    ``device_ms`` / ``sync_ms`` / ``host_cpu_ms`` stacked
                           per rung, with the UNATTRIBUTED remainder drawn
                           explicitly. Schema v2 only.
      ``reconfiguration``  ``context_loads`` / ``kernel_attaches`` per rung --
                           the taxonomy's other axis. Schema v2 only.

WHY THIS IS A REWRITE AND NOT A PORT
    iron's plot tier reads iron's results layout: ``results_all_power.csv``,
    ``tuning_all_power.csv``, an ``execution_mode`` vocabulary that predates the
    taxonomy correction, and no cost decomposition at all because iron's schema
    has nowhere to put one. Doc 03 is explicit that copying iron's column names
    does not define what they mean under AIR's timing model. So this reads
    ``schema.py``'s fields through ``results_io``, and porting convention 8
    (delete redundancy on the way in) is why it calls ``ladder_report.load``
    instead of parsing the tree a second time.

    That reuse is load-bearing rather than tidy: the text report and the figures
    then decide "did this rung pass" in exactly ONE place, so a figure cannot
    show a point the report calls failed.

FOOTGUNS
    - **A v1 tree has EMPTY decomposition columns, and zero is a number.**
      Every pre-`[2026-08-10]` tree predates schema v2, so ``device_ms`` and the
      reconfiguration counters read as ``None``. Drawing those as 0 produces a
      chart that looks measured and is not -- a stacked bar of nothing, and a
      reconfiguration panel showing every mode at zero, which is precisely the
      claim ``offload`` exists to refute. Both v2 figures raise
      ``MissingDecomposition`` naming the tree instead. ``regenerate()`` reports
      the skip; it does not swallow it.
    - **A latency figure with no power mode is a trap, not a minor omission.**
      Trap 0: at ``Default`` the verdict rung reads ~15-20x slow, which presents
      as a compiler regression. Every figure is stamped with the pmode from the
      tree's ``results_manifest.json``; where there is no manifest the stamp
      reads ``unrecoverable`` IN THE FIGURE rather than being left off.
      ``manifest.py`` records that pre-`[2026-08-12]` runs cannot have it
      recovered, so this is a permanent property of the older trees.
    - **Failed rungs are gaps, never interpolated.** A line drawn across a
      missing rung invents a measurement. Failed and absent rungs are excluded
      from the path and from the fit, and named in the figure's footer.
    - **The fitted slope is descriptive.** Same bound as ``ladder_report``: no
      slope below three passing rungs, and three or four rungs over one decade
      cannot separate ``n^2`` from ``n log n``.
    - **The decomposition is NOT a partition, and a big remainder is not a bug.**
      ``schema.py`` states it outright (``device_ms.timing``): the three
      components are disjoint but "do NOT sum to ``avg_latency_ms``" -- untimed
      layout, Python overhead and the stage comparisons are in the total and in
      none of them. On the post-flip trees the remainder runs 37-47% for
      ``offload`` and 84-89% for ``fused``; that is uninstrumented time, and
      reading it as lost device time would be a wrong conclusion drawn off a
      correct chart. The figure says so in its own footer rather than relying on
      a reader knowing.
    - **An ABSENT component is not a measured zero.** ``host_cpu_ms`` is 0.0 for
      ``fused``/``coarse``/``runlist`` because those modes instrument host
      compute and ran none -- a measurement. An empty field instead means the
      mode reported no such component, and the schema forbids fabricating a zero
      for it. Absent components contribute no height and are named in the footer.
    - **The remainder can go NEGATIVE**, and unlike a large remainder that IS a
      defect: disjoint parts cannot exceed the whole. Drawn below the axis and
      called out.
    - Import order matters: this module selects the ``Agg`` backend at import.
      Gates run with no display, and matplotlib otherwise picks an interactive
      backend and fails at figure creation rather than at import.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")  # before pyplot; see the last footgun

import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import ladder_report  # noqa: E402
import schema  # noqa: E402

#: Stable per-mode colours so two walks' figures can be read side by side.
#: ``coarse`` is the mode name; ``hybrid`` is its CSV value (schema footgun).
MODE_COLOURS = {
    "fused": "#1b6ca8",
    "coarse": "#2e8b57",
    "runlist": "#c8791a",
    "offload": "#a03050",
}
_FALLBACK_COLOUR = "#666666"

#: The v2 decomposition, in stack order.
_DECOMPOSITION = (
    ("device_ms", "device", "#1b6ca8"),
    ("sync_ms", "sync", "#6fa8c8"),
    ("host_cpu_ms", "host CPU", "#c8791a"),
)

_RECONFIGURATION = (
    ("context_loads", "hw_context loads"),
    ("kernel_attaches", "kernel attaches"),
)

MANIFEST_NAME = "results_manifest.json"


class MissingDecomposition(RuntimeError):
    """A v2-only figure was asked for over a tree that has no v2 columns.

    Typed rather than a bare ``ValueError`` so ``regenerate`` can report the
    skip precisely -- doc 41's note that Timeloop rejects resources by typed
    failure, applied to the one case here that has a wrong-looking alternative
    (drawing ``None`` as zero).
    """


def _num(value: object) -> float | None:
    """CSV numeric or None. Empty string is unset -- see ``results_io``."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def conditions(root: str) -> dict:
    """The tree's measurement conditions, or an explicit unrecoverable stamp.

    Reads ``results_manifest.json`` when the tree has one. A tree without a
    manifest gets ``npu_power_mode='unrecoverable'`` rather than ``'unknown'``:
    the schema's ``unknown`` means "nobody looked", while these trees were
    written before the conditions block existed and nothing can look now.
    """
    path = os.path.join(root, MANIFEST_NAME)
    if not os.path.isfile(path):
        block = schema.empty_conditions()
        block["npu_power_mode"] = "unrecoverable"
        block["npu_power_mode_source"] = "no manifest in tree"
        block["npu_power_mode_detail"] = (
            f"no {MANIFEST_NAME} under {root}; manifest.py records that runs "
            "before [2026-08-12] carry no conditions block and cannot have one "
            "recovered. Read any latency here as pmode-conditional."
        )
        return block
    with open(path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    block = manifest.get(schema.CONDITIONS_KEY)
    if not isinstance(block, dict):
        block = schema.empty_conditions()
        block["npu_power_mode_source"] = f"{MANIFEST_NAME} carries no block"
    return block


def _stamp(cond: dict) -> str:
    mode = cond.get("npu_power_mode", "unknown")
    observed = cond.get("observed_at_utc") or "time not recorded"
    return f"NPU power mode: {mode}  ·  {observed}"


def load(root: str, modes: list[str] | None = None) -> dict[str, list[dict]]:
    """``{mode: rows}`` via ``ladder_report.load`` -- one reader for both tiers."""
    if modes is None:
        modes = sorted(
            f[:-4] for f in os.listdir(root) if f.endswith(".csv") and f != "report.csv"
        )
    return ladder_report.load(root, modes)


def has_decomposition(data: dict[str, list[dict]]) -> bool:
    """True when any row carries a v2 ``device_ms``. See the v1 footgun."""
    return any(
        _num(row.get("device_ms")) is not None
        for rows in data.values()
        for row in rows
        if row["_ok"]
    )


def _colour(mode: str) -> str:
    return MODE_COLOURS.get(mode, _FALLBACK_COLOUR)


def _failed(data: dict[str, list[dict]]) -> list[str]:
    return [
        f"{mode} @ {row['_seq']}"
        for mode, rows in sorted(data.items())
        for row in rows
        if not row["_ok"]
    ]


def _footer(fig, cond: dict, extra: list[str] | None = None) -> None:
    """Conditions stamp plus any caveats, under every figure. Never optional."""
    lines = [_stamp(cond)]
    lines.extend(extra or [])
    fig.text(
        0.01,
        0.005,
        "\n".join(lines),
        fontsize=7,
        va="bottom",
        ha="left",
        color="#444444",
    )


def latency(
    data: dict[str, list[dict]], cond: dict, *, y_scale: str = "log"
) -> "plt.Figure":
    """Avg latency against sequence length, min-max band, slope in the legend."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for mode in sorted(data):
        rows = [r for r in data[mode] if r["_ok"] and r["_ms"]]
        if not rows:
            continue
        xs = [r["_seq"] for r in rows]
        ys = [r["_ms"] for r in rows]
        colour = _colour(mode)
        exp = ladder_report.exponent(data[mode])
        label = f"{mode}" + (f"  (slope {exp:.2f})" if exp is not None else "")
        ax.plot(xs, ys, marker="o", color=colour, label=label, linewidth=1.8)
        lo = [_num(r.get("min_latency_ms")) for r in rows]
        hi = [_num(r.get("max_latency_ms")) for r in rows]
        if all(v is not None for v in lo) and all(v is not None for v in hi):
            ax.fill_between(xs, lo, hi, color=colour, alpha=0.15, linewidth=0)

    ax.set_xscale("log", base=2)
    ax.set_yscale(y_scale)
    ax.set_xlabel("sequence length (tokens)")
    ax.set_ylabel("latency (ms, avg; band = min..max)")
    ax.set_title("Layer latency by sequence length")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, fontsize=9)

    caveats = ["Shaded band is min..max over the run's samples."]
    failed = _failed(data)
    if failed:
        caveats.append("Excluded (did not pass): " + ", ".join(failed))
    if not any(ladder_report.exponent(rows) is not None for rows in data.values()):
        caveats.append("No slope fitted: fewer than three passing rungs per mode.")
    _footer(fig, cond, caveats)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def dram(data: dict[str, list[dict]], cond: dict) -> "plt.Figure":
    """Bytes crossing DRAM against sequence length -- the taxonomy's own axis."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    drew = False
    for mode in sorted(data):
        pts = [
            (r["_seq"], _num(r.get("bytes_transferred")) / 1e6)
            for r in data[mode]
            if r["_ok"] and _num(r.get("bytes_transferred")) is not None
        ]
        if not pts:
            continue
        drew = True
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            marker="s",
            color=_colour(mode),
            label=mode,
            linewidth=1.8,
        )
    if not drew:
        raise MissingDecomposition("no passing row carries bytes_transferred")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("sequence length (tokens)")
    ax.set_ylabel("bytes transferred (MB)")
    ax.set_title("DRAM traffic by sequence length")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, fontsize=9)

    caveats = ["Byte totals are pmode-independent (doc 32)."]
    failed = _failed(data)
    if failed:
        caveats.append("Excluded (did not pass): " + ", ".join(failed))
    _footer(fig, cond, caveats)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def decomposition(data: dict[str, list[dict]], cond: dict) -> "plt.Figure":
    """Stacked device/sync/host-CPU per rung, with the unattributed remainder.

    Raises ``MissingDecomposition`` on a v1 tree rather than stacking zeros.
    """
    if not has_decomposition(data):
        raise MissingDecomposition(
            "tree has no device_ms on any passing row: it predates schema v2 "
            f"(current SCHEMA_VERSION={schema.SCHEMA_VERSION}). Refusing to "
            "stack empty columns as zero."
        )

    bars = [
        (mode, row)
        for mode in sorted(data)
        for row in data[mode]
        if row["_ok"] and _num(row.get("device_ms")) is not None
    ]
    fig, ax = plt.subplots(figsize=(max(7.5, 1.05 * len(bars) + 3), 4.8))
    xs = range(len(bars))
    bottoms = [0.0] * len(bars)
    unreported = []
    for key, label, colour in _DECOMPOSITION:
        vals = []
        for mode, row in bars:
            value = _num(row.get(key))
            if value is None:
                # schema.py, host_cpu_ms: "A recorded 0.0 is a MEASUREMENT ...
                # an empty field means the mode reported no such component at
                # all; never write a fabricated zero for the latter." Nothing
                # can be drawn for an absent component, so it contributes no
                # height -- but it is named in the footer so the bar is not
                # read as a measured zero.
                unreported.append(f"{mode} @ {row['_seq']} {label}")
                value = 0.0
            vals.append(value)
        ax.bar(xs, vals, bottom=bottoms, color=colour, label=label, width=0.68)
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    negatives = []
    remainders = []
    for i, (mode, row) in enumerate(bars):
        total = row["_ms"] or 0.0
        rem = total - bottoms[i]
        remainders.append(rem)
        if rem < 0:
            negatives.append(f"{mode} @ {row['_seq']}")
    ax.bar(
        xs,
        remainders,
        bottom=bottoms,
        color="#bbbbbb",
        hatch="//",
        edgecolor="#888888",
        label="unattributed",
        width=0.68,
    )

    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{m}\n{r['_seq']}" for m, r in bars], fontsize=8)
    ax.set_ylabel("latency (ms)")
    ax.set_title("Where the layer's time goes")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, fontsize=9)

    caveats = [
        "Unattributed = avg_latency_ms - (device + sync + host CPU). The three "
        "components are DISJOINT and are NOT expected to sum to the total "
        "(schema.py, device_ms.timing): untimed layout, Python overhead and the "
        "stage comparisons sit in the total and in none of the three. A large "
        "remainder is uninstrumented time, NOT a defect.",
    ]
    if unreported:
        caveats.append(
            "Component absent from the row (not a measured zero): "
            + ", ".join(unreported)
        )
    if negatives:
        caveats.append(
            "NEGATIVE remainder -- the disjoint parts exceed the total, which "
            "IS a defect: " + ", ".join(negatives)
        )
    _footer(fig, cond, caveats)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def reconfiguration(data: dict[str, list[dict]], cond: dict) -> "plt.Figure":
    """``context_loads`` and ``kernel_attaches`` per rung. Schema v2 only."""
    bars = [
        (mode, row)
        for mode in sorted(data)
        for row in data[mode]
        if row["_ok"] and _num(row.get("context_loads")) is not None
    ]
    if not bars:
        raise MissingDecomposition(
            "tree has no context_loads on any passing row: it predates schema "
            "v2. Refusing to draw every mode at zero, which is the opposite of "
            "what the offload mode's own axis measures."
        )

    fig, axes = plt.subplots(
        1, 2, figsize=(max(9.0, 1.05 * len(bars) + 4), 4.4), sharex=True
    )
    for ax, (key, title) in zip(axes, _RECONFIGURATION):
        vals = [_num(row.get(key)) or 0.0 for _, row in bars]
        ax.bar(
            range(len(bars)),
            vals,
            color=[_colour(m) for m, _ in bars],
            width=0.68,
        )
        ax.set_xticks(range(len(bars)))
        ax.set_xticklabels([f"{m}\n{r['_seq']}" for m, r in bars], fontsize=8)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel("count per measured layer")
    fig.suptitle("Reconfiguration cost")

    _footer(fig, cond, ["Counts are pmode-independent (doc 32)."])
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    return fig


def write_figure(fig, out_dir: str, stem: str) -> list[str]:
    """Write ``stem.png`` and ``stem.svg``; returns the paths and closes the figure."""
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for ext, kwargs in (("png", {"dpi": 200}), ("svg", {})):
        path = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(path, bbox_inches="tight", **kwargs)
        written.append(path)
    plt.close(fig)
    return written


def regenerate(
    root: str, out_dir: str | None = None, modes: list[str] | None = None
) -> tuple[list[str], list[str]]:
    """Render every figure the tree supports.

    Returns ``(written, skipped)`` -- skips are REPORTED, never swallowed, so a
    v1 tree produces two figures and two named refusals rather than four
    figures, two of which would be fictional.
    """
    out_dir = out_dir or os.path.join(root, "figures")
    data = load(root, modes)
    if not data:
        raise FileNotFoundError(f"no <mode>.csv under {root}")
    cond = conditions(root)

    written, skipped = [], []
    for stem, builder in (
        ("latency_log", lambda: latency(data, cond, y_scale="log")),
        ("latency_linear", lambda: latency(data, cond, y_scale="linear")),
        ("dram_traffic", lambda: dram(data, cond)),
        ("cost_decomposition", lambda: decomposition(data, cond)),
        ("reconfiguration", lambda: reconfiguration(data, cond)),
    ):
        try:
            written.extend(write_figure(builder(), out_dir, stem))
        except MissingDecomposition as exc:
            skipped.append(f"{stem}: {exc}")
    return written, skipped


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root")
    ap.add_argument(
        "--out", default=None, help="figure directory (default <root>/figures)"
    )
    ap.add_argument("--modes", default=None, help="comma-separated; default every CSV")
    ap.add_argument(
        "--require-all",
        action="store_true",
        help="exit non-zero if any figure was skipped (use on a v2 tree)",
    )
    args = ap.parse_args(argv)

    modes = [m for m in args.modes.split(",") if m] if args.modes else None
    written, skipped = regenerate(args.root, args.out, modes)
    for path in written:
        print(f"[plots] wrote {path}")
    for note in skipped:
        print(f"[plots] SKIPPED {note}")
    if skipped and args.require_all:
        print(f"[plots] FAIL: {len(skipped)} figure(s) skipped under --require-all")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
