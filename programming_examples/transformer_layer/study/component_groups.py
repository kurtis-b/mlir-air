# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Where a mode's layer latency went, by named component group.

    python3 study/component_groups.py --mode offload --out results/components.csv

CONTRACT
    ``aggregate(extra, mode)`` turns one dispatch's ``extra`` dict -- what
    ``run_mode.py`` already collects -- into one ``GroupTotal`` per component
    group of that mode: the group's milliseconds, how many components it
    accounted for, how many the taxonomy says it has, and which are missing by
    name. ``render`` prints them; ``rows`` puts them into schema ``component``
    rows; the CLI dispatches a mode once and writes the CSV.

    ``COMPONENT_GROUPS`` is the taxonomy: per CSV ``execution_mode``, an ordered
    tuple of group labels and the components each expects. It is derived from
    the modes AS BUILT, not copied from iron's, whose group tables describe
    iron's decomposition of iron's three modes.

THIS IS AN HONEST PARTIAL, AND THE COLUMNS SAY WHICH PART
    iron's ``run_selected_component_aggregates.py`` re-benchmarks each component
    individually, so every group is fully attributed. MLIR-AIR does not have
    that instrumentation yet, and doc 09 scopes it precisely: ``pattern/``
    "contains no timing at all today", per-stage ``record_kernel``/``record_cpu``
    calls are "new instrumentation on an untimed seam", and they land in Lane 1
    files shared with ``main``. What exists today is:

    - **Named host components.** ``Profiler.time_cpu`` buckets, which
      ``offload`` populates (``attention_layout``, ``softmax``, ``ln1``,
      ``gelu``, ``ln2``) and the other three modes leave empty BY CONSTRUCTION,
      because they run no host compute.
    - **Unnamed device and sync totals.** Every mode sums
      ``device_submission_ms`` and ``host_sync_ms`` over its dispatch vectors.
      The vectors carry no component name, so these are MODE TOTALS.

    So a device group's row reports ``component_count`` 0 against its
    ``expected_component_count`` and names every missing component. That is the
    point: the table shows what is attributed and what is not, instead of
    presenting a mode total under a group label as though the group had been
    measured. When the per-stage instrumentation lands, the same taxonomy fills
    in and ``is_complete`` starts reading True.

FOOTGUNS
    - **The groups do not sum to ``avg_latency_ms``.** Doc 03: the three ms
      components "are disjoint and do NOT sum to ``avg_latency_ms``; the
      remainder is unattributed host overhead", and at 1024 that remainder
      dominates the per-operator modes -- ``runlist`` reads device 44 / sync 6.4
      / host 0 against ~1959 total. ``render`` prints the remainder explicitly
      rather than leaving a reader to notice the columns are short.
    - **An empty host bucket dict is a measured zero, not a missing
      measurement.** ``coarse`` and ``fused`` run nothing on the host, so their
      host groups are complete at 0.0 ms. A MISSING ``host_cpu_ms`` key is
      unmeasured and stays ``None``. That is ``run_mode._component_ms``'s
      distinction, and it is preserved here rather than re-decided.
    - **``group_kind`` is load-bearing.** A ``host_cpu`` row is a named
      component; a ``device`` row is a mode total wearing a group label. Do not
      rank them against each other.
    - **The CLI dispatches.** It needs the NPU, and it goes through the queue:
      ``agents/scripts/devq.sh run --class measure -- python3 study/component_groups.py ...``
      ``run``, never ``submit`` -- ``submit`` diverts output to the job log and
      exits 0, which blanks a gate's FileCheck while still passing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.dirname(_HERE)
_PE = os.path.dirname(_EXAMPLE)
for _p in (_PE, os.path.join(_PE, "llms"), _EXAMPLE, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import results_io  # noqa: E402
import schema  # noqa: E402

#: Which instrument produced a group's number. See the schema field.
GROUP_KINDS = ("host_cpu", "device", "sync")


@dataclass(frozen=True)
class GroupSpec:
    """One component group: where its number comes from and what it expects."""

    label: str
    kind: str
    components: tuple[str, ...]


#: Keyed by CSV ``execution_mode`` (convention 7: `coarse` records as `hybrid`).
#:
#: The DEVICE component lists are the operators each mode dispatches, taken from
#: the mode's own printed dispatch sequence, and they are what
#: `expected_component_count` counts. They are deliberately NOT used to split a
#: device total -- there is nothing to split it by yet -- only to say what a
#: complete attribution would contain.
#:
#: The HOST component lists are the `Profiler.time_cpu` bucket names the mode
#: actually opens. `test_component_groups.py` re-derives those from the pattern
#: sources and fails if this table drifts, the same way `test_attention_path.py`
#: guards the attention map -- both are claims about other files.
COMPONENT_GROUPS: dict[str, tuple[GroupSpec, ...]] = {
    "offload": (
        GroupSpec(
            "GEMMs (NPU)",
            "device",
            (
                "q_proj",
                "k_proj",
                "v_proj",
                "attn_scores",
                "attn_output",
                "output_proj",
                "up_proj",
                "down_proj",
            ),
        ),
        GroupSpec(
            "Non-linear operations (host)",
            "host_cpu",
            ("attention_layout", "softmax", "ln1", "gelu", "ln2"),
        ),
        GroupSpec("Data sync", "sync", ()),
    ),
    "runlist": (
        GroupSpec(
            "Every operator (NPU)",
            "device",
            (
                "ln1",
                "qkvo_proj",
                "k_transpose",
                "attn_scores",
                "attn_scale",
                "attn_softmax",
                "attn_output",
                "add",
                "ln2",
                "up_proj",
                "gelu",
                "down_proj",
            ),
        ),
        GroupSpec("Non-linear operations (host)", "host_cpu", ()),
        GroupSpec("Data sync", "sync", ()),
    ),
    "hybrid": (
        GroupSpec(
            "Packaged front + banded tail (NPU)",
            "device",
            ("qkv_proj", "mha_out_proj", "add_norm1", "ffn", "add_norm2"),
        ),
        GroupSpec("Non-linear operations (host)", "host_cpu", ()),
        GroupSpec("Data sync", "sync", ()),
    ),
    "fused_elf": (
        GroupSpec(
            "Packaged layer (NPU)",
            "device",
            ("rms_gemms_front", "flash_attention", "norm_tail"),
        ),
        GroupSpec("Non-linear operations (host)", "host_cpu", ()),
        GroupSpec("Data sync", "sync", ()),
    ),
}


@dataclass(frozen=True)
class GroupTotal:
    """One group's aggregate for one dispatch."""

    label: str
    kind: str
    ms: float | None
    component_count: int
    expected_component_count: int
    missing_components: tuple[str, ...]

    @property
    def is_complete(self) -> bool:
        return self.component_count == self.expected_component_count


#: Host buckets a mode's DECODER variant opens beyond its encoder set, keyed by
#: CSV execution_mode. Declared beside the encoder table rather than merged
#: into it, because the completeness check is per variant: `offload`'s encoder
#: path never opens `residual_add`, and a merged 6-bucket declaration would
#: report every encoder row incomplete over a bucket the graph does not have.
DECODER_EXTRA_HOST_COMPONENTS: dict[str, tuple[str, ...]] = {
    "offload": ("residual_add",),
}


def groups_for(
    execution_mode: str, variant: str = "encoder_bert"
) -> tuple[GroupSpec, ...]:
    """The taxonomy for a CSV execution_mode. Raises rather than returning ().

    ``variant`` widens the host group for a decoder run by the mode's declared
    extra buckets (``DECODER_EXTRA_HOST_COMPONENTS``); the encoder taxonomy is
    returned untouched for every other value.
    """
    try:
        specs = COMPONENT_GROUPS[execution_mode]
    except KeyError:
        raise ValueError(
            f"no component taxonomy for execution_mode {execution_mode!r}; "
            f"known are {sorted(COMPONENT_GROUPS)}. Note this keys on the CSV "
            "value, so `coarse` is `hybrid` (convention 7)."
        ) from None
    extra = (
        DECODER_EXTRA_HOST_COMPONENTS.get(execution_mode, ())
        if variant == "decoder_gpt2"
        else ()
    )
    if not extra:
        return specs
    return tuple(
        GroupSpec(s.label, s.kind, s.components + extra)
        if s.kind == "host_cpu"
        else s
        for s in specs
    )


def aggregate(extra: dict, execution_mode: str) -> tuple[GroupTotal, ...]:
    """One dispatch's ``extra`` -> one ``GroupTotal`` per group, in order.

    A group whose instrument reported nothing gets ``ms=None``, never 0.0 --
    the same rule ``run_mode`` applies to the schema's decomposition columns,
    and for the same reason: an unmeasured quantity that reads as a measured
    zero is the failure this whole tier is built to avoid.
    """
    host_buckets = extra.get("host_cpu_ms")
    totals = []
    # The variant is read off the dispatch's own extra (the offload decoder
    # reports it), so a decoder run is judged against its own bucket set.
    for spec in groups_for(execution_mode, extra.get("variant", "encoder_bert")):
        if spec.kind == "host_cpu":
            if host_buckets is None:
                ms, counted, missing = None, 0, spec.components
            else:
                # An EMPTY dict is a measured zero: the mode instruments host
                # compute and ran none.
                measured = {
                    name: float(value)
                    for name, value in host_buckets.items()
                    if name in spec.components
                }
                ms = float(sum(measured.values()))
                counted = len(measured)
                missing = tuple(n for n in spec.components if n not in measured)
        elif spec.kind == "device":
            value = extra.get("device_ms")
            # A mode TOTAL under a group label. Nothing is attributed, so every
            # expected component is missing and the count is 0 -- see the
            # module docstring.
            ms = None if value is None else float(value)
            counted, missing = 0, spec.components
        else:
            value = extra.get("sync_ms")
            ms = None if value is None else float(value)
            counted, missing = 0, spec.components
        totals.append(
            GroupTotal(
                label=spec.label,
                kind=spec.kind,
                ms=ms,
                component_count=counted,
                expected_component_count=len(spec.components),
                missing_components=tuple(missing),
            )
        )
    return tuple(totals)


def attributed_ms(totals: tuple[GroupTotal, ...]) -> float:
    """Milliseconds the groups account for. Unmeasured groups contribute 0."""
    return sum(t.ms for t in totals if t.ms is not None)


def render(
    totals: tuple[GroupTotal, ...],
    *,
    execution_mode: str,
    avg_latency_ms: float | None = None,
) -> str:
    """A text table plus the unattributed remainder, stated rather than implied."""
    lines = [f"## Component groups -- {execution_mode}\n"]
    lines.append("| group | kind | ms | components | complete |")
    lines.append("|---|---|---|---|---|")
    for total in totals:
        ms = "—" if total.ms is None else f"{total.ms:.3f}"
        lines.append(
            f"| {total.label} | `{total.kind}` | {ms} | "
            f"{total.component_count}/{total.expected_component_count} | "
            f"{'yes' if total.is_complete else 'no'} |"
        )

    attributed = attributed_ms(totals)
    lines.append(f"\nattributed: {attributed:.3f} ms")
    if avg_latency_ms is not None:
        remainder = avg_latency_ms - attributed
        lines.append(
            f"whole layer: {avg_latency_ms:.3f} ms\n"
            f"UNATTRIBUTED: {remainder:.3f} ms "
            f"({remainder / avg_latency_ms * 100.0:.1f}%) -- host overhead "
            "outside every instrumented region. Doc 03 records this dominating "
            "the per-operator modes; it is printed rather than left for a "
            "reader to notice the columns are short."
        )

    incomplete = [t for t in totals if not t.is_complete]
    if incomplete:
        lines.append("\nnot attributed to components:")
        for total in incomplete:
            lines.append(f"- {total.label}: {', '.join(total.missing_components)}")
        lines.append(
            "  These need the per-stage record_kernel/record_cpu calls doc 09 "
            "scopes (2026-08-08); the taxonomy above is what they will fill in."
        )
    return "\n".join(lines) + "\n"


def rows(
    totals: tuple[GroupTotal, ...],
    *,
    execution_mode: str,
    study_id: str = "adhoc",
    study_case_id: str | None = None,
    study_case_label: str | None = None,
    workload_variant: str | None = None,
    seq_len: int | None = None,
    failure_message: str = "",
) -> list[dict]:
    """Schema ``component`` rows. A group with no milliseconds is ``failed``."""
    out = []
    for total in totals:
        row = schema.empty_row("component")
        row.update(
            {
                "study_id": study_id,
                "study_case_id": study_case_id,
                "study_case_label": study_case_label,
                "workload_variant": workload_variant,
                "execution_mode": execution_mode,
                "seq_len": seq_len,
                "group_label": total.label,
                "group_kind": total.kind,
                "avg_latency_ms": total.ms,
                "component_count": total.component_count,
                "expected_component_count": total.expected_component_count,
                "missing_components_json": json.dumps(list(total.missing_components)),
                "is_complete": total.is_complete,
                "run_status": "passed" if total.ms is not None else "failed",
                "failure_message": (
                    ""
                    if total.ms is not None
                    else failure_message
                    or "the mode reported no milliseconds for this group's instrument"
                ),
            }
        )
        out.append(row)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", default="offload", help="the mode's CODE name")
    ap.add_argument("--out", default=None, help="component CSV to write")
    ap.add_argument("--seq", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--study-id", default="adhoc")
    ap.add_argument("--md", default=None, help="also write the text report here")
    args = ap.parse_args(argv)

    import run_mode  # noqa: E402  -- imports the toolchain; keep it lazy

    try:
        run_mode.require_turbo()
    except run_mode.TurboNotEnforced as exc:
        # Same fail-closed guard as run_mode: the device milliseconds this
        # aggregates are not comparable with any recorded number off Turbo.
        print(f"[component-groups] refused: {exc}")
        return 2

    csv_mode = schema.EXECUTION_MODE_CSV.get(args.mode, args.mode)
    spec = run_mode._spec_for(args.mode)
    # Three values since the family override (2026-08-12); this line unpacked
    # two until 2026-08-19, so the CLI crashed at every invocation since.
    shape, shape_key, _variant = run_mode._shape_for(spec, args.seq)

    try:
        prepared = spec["prepare"](shape)
        dispatch = prepared["dispatch"]
        for _ in range(args.warmup):
            dispatch(prepared["inputs"], run_mode._stage_stats)
        # The whole-layer clock, wrapped around the SAME dispatch the groups are
        # read from -- one iteration, not a distribution. It exists so the
        # report can state the unattributed remainder, which is the number this
        # table is really about; it is NOT a latency measurement and
        # `run_mode.py` remains the only thing that produces one.
        started = time.perf_counter()
        _outputs, extra = dispatch(prepared["inputs"], run_mode._stage_stats)
        whole_layer_ms = (time.perf_counter() - started) * 1000.0
    except Exception as e:
        print(f"[component-groups] {args.mode}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    totals = aggregate(extra, csv_mode)
    report = render(totals, execution_mode=csv_mode, avg_latency_ms=whole_layer_ms)
    print(report)
    if args.md:
        with open(args.md, "w", encoding="utf-8") as fh:
            fh.write(report)
        print(f"[component-groups] wrote {args.md}")
    if args.out:
        results_io.write_rows(
            args.out,
            rows(
                totals,
                execution_mode=csv_mode,
                study_id=args.study_id,
                study_case_id=shape_key,
                study_case_label=f"{args.mode} {shape_key}",
                workload_variant="encoder_bert",
                seq_len=shape.get("seq_len"),
            ),
            table="component",
        )
        print(f"[component-groups] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
