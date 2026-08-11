# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The case matrix: which model families and sequence lengths a study walks.

    python3 study/cases.py --list
    python3 study/cases.py --family baseline_768 --seq 1024

CONTRACT
    ``FAMILY_SPECS`` is the six-family matrix -- three encoder widths and three
    decoder widths -- each carrying its shape, its display label, and its
    position in the 2x3 reporting grid. ``SEQUENCE_LADDER`` is the nine-point
    ladder. ``case(family, seq)`` resolves one point into a ``Case`` whose
    ``Workload`` carries every shape column a results row needs, and
    ``iter_cases`` walks a filtered subset in a stable order.

    ``effective_gflops_per_sec`` turns a latency into a rate through a FLOP
    count derived from the SHAPE. It is the schema column of the same name, and
    it is what makes two modes at one shape comparable on something other than
    milliseconds.

WHY THE PLOT LABELS LIVE HERE
    iron splits this across ``end_to_end/cases.py`` (shapes) and
    ``plot_families.py`` (labels, order, grid position). Doc 09 records that the
    second module's "name is misleading -- it holds family definitions, not
    drawing code", and two files defining the same six families is exactly the
    drift porting convention 8 says to delete on the way in. So there is one
    table, and a text report and a future plot read the same rows.

WHY THE MODE VOCABULARY IS NOT RESTATED HERE
    Convention 7 confines the ``coarse``/``hybrid`` mapping to the schema module.
    ``canonical_execution_mode`` therefore VALIDATES against
    ``schema.EXECUTION_MODES`` and translates through
    ``schema.EXECUTION_MODE_CSV``; it defines neither. A second list of the
    modes here is how the two would end up disagreeing.

FOOTGUNS
    - **The FLOP count is EFFECTIVE, not executed.** It is the dense-layer
      arithmetic the shape implies -- six projection GEMMs, both attention
      matmuls, both FFN GEMMs -- and it is identical for every execution mode by
      construction. It deliberately counts no softmax, no normalization, no
      transpose, and it does not know that a mode ran attention on the host or
      that another paid for a banded chain. That is what makes it a fair
      denominator across modes, and it means the number is NOT a hardware
      utilization figure and must never be read as a fraction of peak.
    - **A family id is not a shape.** ``baseline_768`` fixes width, FFN width
      and head count; the sequence length comes from the ladder. Rows are
      identified by BOTH, which is why ``compare_roots`` keys on
      ``(study_case_id, execution_mode, seq_len)``.
    - **The ladder reaches 16384 and the hardware does not.** These are the
      points the matrix DEFINES; which of them a mode can build is a separate
      question that only a run answers. Nothing here filters by feasibility,
      deliberately -- a case that cannot build should produce a failed row, not
      vanish from the matrix.
    - ``gpt2_*`` families are declared and no mode is gated on them today. They
      are in the matrix because the matrix is the study's design; a decoder run
      needs the causal-mask path, which is a builder keyword argument here
      rather than an operator (convention 8).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import schema  # noqa: E402

#: The nine points, powers of two. Matches ``sweep/sweep_families.py``'s ladder
#: so a case and the GEMM shapes it needs are quoted at the same lengths.
SEQUENCE_LADDER: tuple[int, ...] = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)

WORKLOAD_VARIANTS: tuple[str, ...] = ("encoder_bert", "decoder_gpt2")

#: The reporting grid: encoder families on the top row, decoder on the bottom,
#: small to large left to right. Carried from iron's ``plot_families.py`` and
#: used by text reports today.
FAMILY_GRID_SHAPE: tuple[int, int] = (2, 3)
FAMILY_ROW_LABELS: tuple[str, str] = ("Encoder", "Decoder")


@dataclass(frozen=True)
class FamilySpec:
    """One model family: its shape, its name, and where it sits in the grid."""

    workload_variant: str
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    display_label: str
    short_label: str
    row_index: int
    col_index: int


FAMILY_SPECS: dict[str, FamilySpec] = {
    "tinybert_512": FamilySpec(
        workload_variant="encoder_bert",
        hidden_size=512,
        intermediate_size=2048,
        num_attention_heads=8,
        display_label="TinyBERT",
        short_label="B-S",
        row_index=0,
        col_index=0,
    ),
    "baseline_768": FamilySpec(
        workload_variant="encoder_bert",
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
        display_label="BERT-Base",
        short_label="B-M",
        row_index=0,
        col_index=1,
    ),
    "baseline_1024": FamilySpec(
        workload_variant="encoder_bert",
        hidden_size=1024,
        intermediate_size=4096,
        num_attention_heads=16,
        display_label="BERT-Large",
        short_label="B-L",
        row_index=0,
        col_index=2,
    ),
    "gpt2_512": FamilySpec(
        workload_variant="decoder_gpt2",
        hidden_size=512,
        intermediate_size=2048,
        num_attention_heads=8,
        display_label="GPT-2 512",
        short_label="G-S",
        row_index=1,
        col_index=0,
    ),
    "gpt2_small_768": FamilySpec(
        workload_variant="decoder_gpt2",
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
        display_label="GPT-2 Small",
        short_label="G-M",
        row_index=1,
        col_index=1,
    ),
    "gpt2_medium_1024": FamilySpec(
        workload_variant="decoder_gpt2",
        hidden_size=1024,
        intermediate_size=4096,
        num_attention_heads=16,
        display_label="GPT-2 Medium",
        short_label="G-L",
        row_index=1,
        col_index=2,
    ),
}

#: Declaration order IS reporting order, and the grid positions above agree with
#: it. Derived rather than restated so the two cannot drift.
FAMILY_IDS: tuple[str, ...] = tuple(FAMILY_SPECS)


@dataclass(frozen=True)
class Workload:
    """One resolved shape. Every field is a schema column."""

    workload_variant: str
    seq_len: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int

    @property
    def attention_head_size(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def effective_flop_count(self) -> int:
        return effective_flop_count(
            seq_len=self.seq_len,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_attention_heads=self.num_attention_heads,
        )


@dataclass(frozen=True)
class Case:
    """One point of the matrix: a family at a sequence length."""

    study_case_id: str
    study_case_label: str
    workload: Workload

    @property
    def workload_variant(self) -> str:
        return self.workload.workload_variant

    @property
    def seq_len(self) -> int:
        return self.workload.seq_len

    def shape_columns(self) -> dict[str, object]:
        """The shape half of a results row, so no runner restates it.

        Deliberately NOT a whole row: identity, timing and outcome belong to the
        thing that measured it.
        """
        return {
            "study_case_id": self.study_case_id,
            "study_case_label": self.study_case_label,
            "workload_variant": self.workload_variant,
            "seq_len": self.workload.seq_len,
            "hidden_size": self.workload.hidden_size,
            "intermediate_size": self.workload.intermediate_size,
            "num_attention_heads": self.workload.num_attention_heads,
            "attention_head_size": self.workload.attention_head_size,
        }


def canonical_workload_variant(value: str) -> str:
    if value not in WORKLOAD_VARIANTS:
        raise ValueError(
            f"unknown workload variant {value!r}; known are "
            f"{list(WORKLOAD_VARIANTS)}"
        )
    return value


def canonical_execution_mode(value: str) -> str:
    """The CSV ``execution_mode`` for ``value``, accepting a code name too.

    Accepts either side of convention 7's mapping and always returns the CSV
    side, because that is what a results row carries. The domain and the mapping
    are the schema module's; this only applies them.
    """
    if value in schema.EXECUTION_MODE_CSV:
        return schema.EXECUTION_MODE_CSV[value]
    if value in schema.EXECUTION_MODES:
        return value
    raise ValueError(
        f"unknown execution mode {value!r}; CSV values are "
        f"{list(schema.EXECUTION_MODES)} and code names are "
        f"{sorted(schema.EXECUTION_MODE_CSV)}"
    )


def effective_flop_count(
    *,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
) -> int:
    """Dense-layer FLOPs the shape implies. Identical across execution modes.

    Six projection GEMMs (Q, K, V and the output projection are 8 x S x H x H
    of multiply-adds, i.e. 8 S H^2 FLOPs, split here as qkv + out to match how
    the modes name them), both attention matmuls at 4 S^2 H, and the two FFN
    GEMMs at 4 S H I. Softmax, normalization and transposes are excluded --
    see the module docstring on why that is what makes it a fair denominator.
    """
    if num_attention_heads <= 0 or hidden_size % num_attention_heads:
        raise ValueError(
            f"hidden_size {hidden_size} is not divisible by "
            f"num_attention_heads {num_attention_heads}, so the head size the "
            "attention term needs is undefined"
        )
    qkv = 6 * seq_len * hidden_size * hidden_size
    attention = 4 * seq_len * seq_len * hidden_size
    out_proj = 2 * seq_len * hidden_size * hidden_size
    ffn = 4 * seq_len * hidden_size * intermediate_size
    return qkv + attention + out_proj + ffn


def effective_gflops_per_sec(
    workload: Workload, avg_latency_ms: float | None
) -> float | None:
    """The schema column. ``None`` for an unmeasured or nonsensical latency."""
    if avg_latency_ms is None or avg_latency_ms <= 0:
        return None
    return workload.effective_flop_count / (avg_latency_ms / 1000.0) / 1.0e9


def effective_gflops_per_sec_per_watt(
    gflops_per_sec: float | None, avg_power_w: float | None
) -> float | None:
    """The efficiency column. ``None`` unless both inputs are real and positive."""
    if gflops_per_sec is None or avg_power_w is None or avg_power_w <= 0:
        return None
    return gflops_per_sec / avg_power_w


def case(study_case_id: str, seq_len: int) -> Case:
    """Resolve one matrix point. Raises on an unknown family, never guesses."""
    try:
        spec = FAMILY_SPECS[study_case_id]
    except KeyError:
        raise ValueError(
            f"unknown family {study_case_id!r}; known are {list(FAMILY_IDS)}"
        ) from None
    return Case(
        study_case_id=study_case_id,
        study_case_label=f"{spec.display_label} @ {seq_len}",
        workload=Workload(
            workload_variant=spec.workload_variant,
            seq_len=seq_len,
            hidden_size=spec.hidden_size,
            intermediate_size=spec.intermediate_size,
            num_attention_heads=spec.num_attention_heads,
        ),
    )


def iter_cases(
    *,
    workload_variant: str = "all",
    family: str = "all",
    seq_len: int | str = "all",
) -> tuple[Case, ...]:
    """The filtered matrix, in declaration order then ladder order."""
    if family != "all":
        family_ids: tuple[str, ...] = (family,)
    else:
        family_ids = tuple(
            fid
            for fid in FAMILY_IDS
            if workload_variant == "all"
            or FAMILY_SPECS[fid].workload_variant
            == canonical_workload_variant(workload_variant)
        )
    lengths = SEQUENCE_LADDER if seq_len == "all" else (int(seq_len),)
    return tuple(case(fid, n) for fid in family_ids for n in lengths)


def ordered_family_ids(
    present: object = None, *, workload_variant: str | None = None
) -> tuple[str, ...]:
    """Declaration order, optionally narrowed to a variant and to what exists.

    ``present`` is any iterable of family ids -- typically the ones a results
    tree actually has -- so a report orders what it has rather than leaving gaps
    for families nobody measured.
    """
    seen = None if present is None else set(present)
    return tuple(
        fid
        for fid, spec in FAMILY_SPECS.items()
        if (workload_variant is None or spec.workload_variant == workload_variant)
        and (seen is None or fid in seen)
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family", default="all")
    ap.add_argument("--variant", default="all")
    ap.add_argument("--seq", default="all")
    ap.add_argument("--list", action="store_true", help="families only, no ladder")
    args = ap.parse_args(argv)

    if args.list:
        for fid, spec in FAMILY_SPECS.items():
            print(
                f"{fid:<18} {spec.display_label:<14} {spec.short_label:<4} "
                f"{spec.workload_variant:<13} H={spec.hidden_size:<5} "
                f"I={spec.intermediate_size:<5} heads={spec.num_attention_heads} "
                f"grid=({spec.row_index},{spec.col_index})"
            )
        return 0

    for c in iter_cases(
        workload_variant=args.variant, family=args.family, seq_len=args.seq
    ):
        print(
            f"{c.study_case_id:<18} seq {c.seq_len:<6} "
            f"{c.workload.effective_flop_count / 1e9:>10.2f} GFLOP  "
            f"{c.study_case_label}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
