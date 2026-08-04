# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The catalogue the registry sweep measures: which shapes, and which tile
configurations are worth trying for each.

CONTRACT
    ``shapes_for_family(family_id)`` returns the ordered list of
    ``ShapeSpec(role, seq, M, K, N, used_by)`` a family needs, and
    ``candidates_for_shape(shape)`` returns the ordered list of
    ``Candidate`` configurations to measure for one of them. Neither touches
    hardware, neither imports the AIR runtime, and neither knows how a
    measurement is taken -- ``sweep_measure.py`` owns that.

WHY THIS IS A SEPARATE MODULE FROM ``registry_sweep.py``
    The same seam ``opcheck.py`` / ``opcheck_specs.py`` already draws in this
    example: mechanism on one side, catalogue on the other. Adding a family or
    widening the candidate grid touches only this file; changing how a winner is
    chosen, resumed or written touches only the orchestrator. It also keeps both
    under porting convention 5's ~800-line cap.

THE FAMILIES
    Three transformer widths from the case matrix, each swept over the full
    9-point sequence ladder, each contributing four projection GEMMs:

        qkv_proj  (seq, h,   3h)      ffn_up    (seq, h,   4h)
        ffn_down  (seq, 4h,  h )      o_proj    (seq, h,   h )

    ``baseline_768`` is the family Phase D needs and the one this sub-phase
    registers. ``baseline_512`` and ``baseline_1024`` are the same tool run
    against a different ``--family`` -- no code change, just machine time.

WHY THE CANDIDATE GRID IS SHAPE-DERIVED AND NOT A CONSTANT TABLE
    The example's own divisibility asserts
    (``matrix_multiplication/bf16_in_bf16_out/run.py:63-66``) rule out most of a
    fixed grid at these shapes, and they rule out DIFFERENT parts of it per
    shape. ``N = 768`` cannot use the registry's usual ``tile_n = 128`` at
    ``herd_n = 4`` (``768 % 512 != 0``); ``seq = 64`` cannot use ``herd_m = 8``
    at either method's ``tile_m``. A hardcoded candidate list would therefore
    have to be per-shape anyway, and would silently go stale the moment a family
    id was added. Deriving it from the constraints keeps the two in step.

FOOTGUNS
    - ``tile_m`` IS NOT A FREE KNOB for the two high-precision methods. drain is
      32 and fused-cast is 64 (``llms/shared/builders/gemm_builder.py:21-26``),
      and ``_spec_with_tiles`` asserts the registry agrees. Emitting a row whose
      ``tile_m`` disagrees with its method would not mis-tune a model, it would
      abort it at build time. ``METHODS`` is the single place that mapping lives.
    - ``TILE_N_MAX`` is a MEASURED ceiling, not a derived one. ``tile_n = 192``
      satisfies every divisibility rule at ``N = 768`` and ``N = 2304`` and still
      fails to place ("Bank-aware allocation failed" then an L1 over-allocation).
      Candidates above the ceiling are not generated, because a placement failure
      costs a full aircc compile to discover.
    - ``herd_n`` is held at 4 wherever a legal ``tile_n`` exists for it. The
      registry already records that dropping to ``herd_n = 1`` at N=896 placed
      but then FAILED AT RUNTIME (``details/GEMM_bf16_in_bf16_out.json``, the
      2048x896x896 ``_note``) -- the fused-cast and drain paths assume the 8x4
      array. Narrowing ``tile_n`` is the supported way to divide an awkward N.
    - The order of ``candidates_for_shape`` is the tie-break order the
      orchestrator inherits, so it must be deterministic. It is: methods in
      ``METHODS`` order, then tile_n descending, then tile_k_l2 in preference
      order.
"""

from dataclasses import dataclass

# Sequence ladder from the case matrix. Nine points, powers of two.
SEQ_LADDER = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)

# family id -> hidden size. ffn_dim is 4*hidden throughout the case matrix.
FAMILY_HIDDEN = {
    "baseline_512": 512,
    "baseline_768": 768,
    "baseline_1024": 1024,
}

# role -> (K, N) as multiples of the hidden size. M is always the sequence length.
ROLE_KN_MULTIPLES = {
    "qkv_proj": (1, 3),
    "ffn_up": (1, 4),
    "ffn_down": (4, 1),
    "o_proj": (1, 1),
}
ROLES = ("qkv_proj", "ffn_up", "ffn_down", "o_proj")

# The three bf16-out GEMM methods the example harness exposes, and the two facts
# about each that the sweep cannot choose: the tile_m the method forces, and the
# registry precision tier it lands in.
#
# tier "high" = FP32 accumulate across the whole K reduction + one epilogue cast.
# tier "low"  = direct-codegen bf16, truncated once per L2 tile.
#
# `high_precision` and `method_arg` are what the example harness's CLI/builders
# want; the sweep never re-derives them.
METHODS = {
    "fused-cast": {"tile_m": 64, "tier": "high", "high_precision": True},
    "drain": {"tile_m": 32, "tier": "high", "high_precision": True},
    "direct": {"tile_m": 64, "tier": "low", "high_precision": False},
}

# Inner-accumulation chunk. Every registry row uses 32 and the detail page lists
# no alternative worth measuring; it stays a constant so the grid stays small.
TILE_K_L1 = 32

# Descending, because a wider N tile is the dominant performance knob (the detail
# page measures 128 beating 64 by ~1.3-1.4x). Capped by TILE_N_MAX below.
TILE_N_PREFERENCE = (128, 96, 64, 48, 32, 16, 8)

# Measured L1 ceiling -- see the footgun above. 128 is also the largest tile_n
# any existing registry row uses.
TILE_N_MAX = 128

# 256 first because it is what nearly every existing row settled on; 512 second
# because it wins on the wide-K shapes (2048x2048x2048, 4096x4096x4096).
TILE_K_L2_PREFERENCE = (256, 512, 128, 64)

# Full chip is 8 rows x 4 columns. Both lists are descending: more tiles is more
# parallelism, and a smaller herd is only used when divisibility forces it.
HERD_M_PREFERENCE = (8, 4, 2, 1)
HERD_N_PREFERENCE = (4, 2, 1)

# How many options of each knob to keep per method. Two apiece gives four
# candidates per method and twelve per shape, which is the grid this sub-phase
# ran. Raising it is a machine-time decision, not a correctness one.
DEFAULT_TILE_N_OPTIONS = 2
DEFAULT_TILE_K_L2_OPTIONS = 2


@dataclass(frozen=True)
class ShapeSpec:
    """One (M, K, N) the sweep must register, and where it came from."""

    family: str
    role: str
    seq: int
    M: int
    K: int
    N: int

    @property
    def key(self):
        return (self.M, self.K, self.N)

    @property
    def label(self):
        return f"{self.M}x{self.K}x{self.N}"

    @property
    def used_by(self):
        return f"transformer-layer study {self.family} {self.role} (seq={self.seq})"


@dataclass(frozen=True)
class Candidate:
    """One tile configuration to build, check and time for a shape."""

    method: str
    tile_m: int
    tile_k_l2: int
    tile_k_l1: int
    tile_n: int
    herd_m: int
    herd_n: int
    cast_tile_n: int

    @property
    def tier(self):
        return METHODS[self.method]["tier"]

    @property
    def high_precision(self):
        return METHODS[self.method]["high_precision"]

    @property
    def tile(self):
        """The registry's named tile dict, exactly as a JSON row stores it."""
        return {
            "tile_m": self.tile_m,
            "tile_k_l2": self.tile_k_l2,
            "tile_k_l1": self.tile_k_l1,
            "tile_n": self.tile_n,
        }

    @property
    def label(self):
        return (
            f"{self.method}/tm{self.tile_m}/tk2{self.tile_k_l2}"
            f"/tn{self.tile_n}/herd{self.herd_m}x{self.herd_n}"
        )

    def as_dict(self):
        """Flat dict used as the resume signature. Every field that changes what
        is built must appear here, or editing the grid would silently reuse a
        stale winner."""
        return {
            "method": self.method,
            "tile_m": self.tile_m,
            "tile_k_l2": self.tile_k_l2,
            "tile_k_l1": self.tile_k_l1,
            "tile_n": self.tile_n,
            "herd_m": self.herd_m,
            "herd_n": self.herd_n,
            "cast_tile_n": self.cast_tile_n,
        }


def shapes_for_family(family, seq_ladder=SEQ_LADDER, roles=ROLES):
    """Every (role, seq) shape one family contributes, in a stable order."""
    if family not in FAMILY_HIDDEN:
        raise KeyError(
            f"unknown family {family!r}; known families: {sorted(FAMILY_HIDDEN)}"
        )
    hidden = FAMILY_HIDDEN[family]
    shapes = []
    for role in roles:
        k_mult, n_mult = ROLE_KN_MULTIPLES[role]
        for seq in seq_ladder:
            shapes.append(
                ShapeSpec(
                    family=family,
                    role=role,
                    seq=seq,
                    M=seq,
                    K=hidden * k_mult,
                    N=hidden * n_mult,
                )
            )
    return shapes


def _herd_options(dim, tile, preference):
    """Herd sizes that tile ``dim`` exactly, largest first.

    The example asserts ``dim % (tile * herd) == 0``
    (``bf16_in_bf16_out/run.py:63,66``); anything else aborts before it compiles.
    """
    return [h for h in preference if tile * h <= dim and dim % (tile * h) == 0]


def _tile_n_options(n, limit):
    """(tile_n, herd_n) pairs for one N, best first, at most ``limit`` of them.

    herd_n = 4 is tried across every legal tile_n before herd_n = 2 is considered
    at all -- see the footgun on why a narrow herd is the wrong way to divide an
    awkward N.
    """
    options = []
    for herd_n in HERD_N_PREFERENCE:
        for tile_n in TILE_N_PREFERENCE:
            if tile_n > TILE_N_MAX:
                continue
            if tile_n * herd_n <= n and n % (tile_n * herd_n) == 0:
                options.append((tile_n, herd_n))
        if options:
            break
    return options[:limit]


def _tile_k_l2_options(k, limit):
    """tile_k_l2 values that divide K and are a whole number of L1 chunks."""
    options = [
        tk2
        for tk2 in TILE_K_L2_PREFERENCE
        if tk2 <= k and k % tk2 == 0 and tk2 % TILE_K_L1 == 0
    ]
    return options[:limit]


def _cast_tile_n(m, n, herd_x=8):
    """L1 tile for fused-cast's separate cast launch.

    ``_build_cast_module`` collapses the M x N output to 1D, gives each of
    ``herd_x`` tiles a contiguous chunk, and asserts the chunk is a whole number
    of ``cast_tile_n`` (``bf16_in_bf16_out/run.py:734-736``). The harness default
    of 2048 divides every chunk in the case matrix, but deriving it keeps a
    future family from tripping the assert after a full compile.
    """
    chunk = (m * n) // herd_x
    for tile in (2048, 1024, 512, 256, 128, 64, 32, 16, 8):
        if chunk % tile == 0:
            return tile
    return 8


def candidates_for_shape(
    shape,
    methods=tuple(METHODS),
    tile_n_options=DEFAULT_TILE_N_OPTIONS,
    tile_k_l2_options=DEFAULT_TILE_K_L2_OPTIONS,
):
    """Every configuration worth building for one shape, best-guess first.

    Returns ``[]`` for a shape no legal configuration exists for, which the
    orchestrator records as a placement failure rather than as an omission.
    """
    tn_options = _tile_n_options(shape.N, tile_n_options)
    tk2_options = _tile_k_l2_options(shape.K, tile_k_l2_options)
    cast_tile_n = _cast_tile_n(shape.M, shape.N)

    candidates = []
    for method in methods:
        tile_m = METHODS[method]["tile_m"]
        herd_m_options = _herd_options(shape.M, tile_m, HERD_M_PREFERENCE)
        if not herd_m_options:
            # M smaller than this method's forced tile_m. Nothing to try; the
            # other method's smaller tile_m may still fit.
            continue
        herd_m = herd_m_options[0]
        for tile_n, herd_n in tn_options:
            for tile_k_l2 in tk2_options:
                candidates.append(
                    Candidate(
                        method=method,
                        tile_m=tile_m,
                        tile_k_l2=tile_k_l2,
                        tile_k_l1=TILE_K_L1,
                        tile_n=tile_n,
                        herd_m=herd_m,
                        herd_n=herd_n,
                        cast_tile_n=cast_tile_n,
                    )
                )
    return candidates
