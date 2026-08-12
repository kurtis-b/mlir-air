# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The energy/latency reference table: ``(component, action, arguments) -> cost``.

    python3 study/balance_ert.py --seed-gemm-sweep sweep/results/baseline_768 \\
        --seed-routed-design <air_project-dir> --out results/ert.json

CONTRACT
    An ``Ert`` maps an ``ActionKey`` -- a component, an action, and the FULL
    argument tuple that action was invoked with -- to a ``Cost`` carrying
    nanoseconds, bytes, and for each of those a ``source`` naming how it was
    obtained. ``insert`` refuses a key whose required arguments are missing;
    ``lookup`` is EXACT and raises ``ErtMiss`` listing the near neighbours
    rather than returning a default. ``save``/``load`` round-trip it as JSON.

    This is Accelergy's ERT pattern (doc 40) with one deliberate difference:
    Accelergy's table holds energy per action estimated by a plugin, and ours
    holds MEASURED nanoseconds and COUNTED bytes with the artifact path beside
    each. Doc 44: "Ours holds measured nanoseconds and bytes, which makes our
    version stronger than Accelergy's rather than weaker." Doc 40 records why
    the plugin half does not transfer -- CACTI hard-asserts 22-180 nm and
    reaches a 4 nm-class node by a literal ``read_energy *= scale**0.5``
    extrapolation ~5.5x out of range, and plugin "accuracy" is a self-declared
    constant typed into source. Take the pattern, not the tool.

THE ARGUMENTS ARE THE POINT, AND ``REQUIRED_ARGUMENTS`` IS WHY
    Doc 44, in the sentence that specifies this module: "Actions must carry
    arguments: a ``dma_transfer`` is a function of ``(n_words, n_dims,
    stride)``, not a scalar -- given our BD-stride walls, a counter reporting
    'number of DMA transfers' has already destroyed the information we need."

    That is not a style preference here. Doc 23's L3-side offset rule and doc
    31's wall-4/wall-5 findings are all statements about descriptor SHAPE:
    ``sizes [8, 4, 8, 8]`` / ``strides [6144, 8, 768, 1]`` is a retile that
    uses all four BD dimensions and cannot take a fifth, and no scalar summary
    of it can say so. So ``REQUIRED_ARGUMENTS`` names, per (component, action),
    the arguments an entry MUST carry, and ``insert`` raises without them. A
    ``dma_transfer`` costed as one number is rejected at the point of insert
    rather than discovered later as a search that converged on a shape the
    compiler cannot lower.

    ``lookup`` is exact for the same reason. Two transfers agreeing on
    ``n_words`` and differing in ``strides`` are different objects to the BD
    allocator, and a lookup that fell back to the ``n_words`` match would erase
    exactly the difference the table exists to preserve. A miss names the near
    neighbours so the difference is visible.

WHAT ``source`` MEANS, AND WHY IT IS NOT A BOOLEAN
    The brief this module was built to says a modelled constant presented as
    measured is the failure this instrument exists to prevent, so the split is
    four-valued and stored per number rather than per entry:

    - ``measured``  -- a stopwatch on hardware produced it. ``provenance`` is
      the results artifact, and ``condition`` carries the NPU power mode,
      because doc 32 records a ~15-20x error between Turbo and Default and a
      latency without its pmode is not comparable with anything.
    - ``counted``   -- read exactly off a compiled artifact (bytes from
      ``sizes`` x element width). Not a measurement of the machine, not a
      model; it is arithmetic on a file, and it is reproducible without the
      device.
    - ``modelled``  -- computed from an assumption, which ``provenance`` must
      state. Nothing in the shipped seeds writes this.
    - ``absent``    -- nothing is known. The value MUST be ``None``, enforced
      in ``__post_init__``, so an unpriced action can never read as a free one.

    A caller wanting "how much of this table is real" calls ``by_source``.

FOOTGUNS
    - **There is no measured AIR-native shim bandwidth in this tree.** Doc 33
      settled that: the memcpy operator was DEFERRED, and iron's 67.9-70.9 GB/s
      is a cross-toolchain import that doc 33 marks as an order-of-magnitude
      statement rather than a measurement. So every ``shim_dma`` entry this
      module seeds carries ``bytes`` (counted) and ``ns=None`` (absent). That
      is the honest state and it is what makes ``balance.back_solve`` -- which
      asks what bandwidth a stall-free run WOULD have needed -- the useful
      question rather than the ratio against a ceiling.
    - **A seeded latency is pmode-conditional.** The sweep JSONs this reads
      carry no pmode field; the caller passes ``--condition``, and the default
      is the unknown marker, not Turbo. README: latencies recorded 2026-08-10
      are Default-conditional and pre-08-10 are Turbo-conditional.
    - **``insert`` refuses a conflicting duplicate.** Two runs of the same
      candidate produce two latencies, and silently keeping the last one is how
      a table starts disagreeing with the artifacts behind it. Pass
      ``replace=True`` deliberately, or key the two apart.
    - **JSON keys are tuples**, so ``save`` writes a list of records rather
      than an object keyed by the action. A dict keyed on a rendered tuple
      would make the argument set a string and re-introduce exactly the
      scalar-summary defect above.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

#: The file format version. Bumped when a RECORD's shape changes, so a table
#: written by an older run is rejected loudly rather than read as this shape.
ERT_FORMAT_VERSION = 1

#: How a number was obtained. See the module docstring; the ordering is
#: strongest-evidence first and ``report`` prints it in this order.
COST_SOURCES: tuple[str, ...] = ("measured", "counted", "modelled", "absent")

#: What a measurement records when the NPU power mode behind it is not known.
#: Never replaced by a guess: doc 32 measures a ~15-20x Turbo/Default error, so
#: an unconditioned latency is unconditioned on the axis that dominates it.
CONDITION_UNKNOWN = "npu_power_mode=unknown"

#: Per ``(component, action)``, the arguments an entry MUST carry. ``"*"`` as
#: the action matches any action of that component -- ``gemm``'s actions are
#: tiling METHODS (``direct`` / ``drain`` / ``fused-cast`` / ...) and a new one
#: must not be able to enter the table with a smaller argument set than its
#: siblings.
#:
#: The ``shim_dma`` row is doc 44's sentence, transcribed: a ``dma_transfer``
#: is a function of ``(n_words, n_dims, stride)``. ``element_bytes`` is here
#: too because ``n_words`` alone does not give bytes, and the back-solve needs
#: bytes.
REQUIRED_ARGUMENTS: dict[tuple[str, str], tuple[str, ...]] = {
    ("shim_dma", "dma_transfer"): (
        "n_words",
        "n_dims",
        "strides",
        "element_bytes",
    ),
    ("gemm", "*"): (
        "M",
        "K",
        "N",
        "tile_m",
        "tile_k_l2",
        "tile_k_l1",
        "tile_n",
        "herd_m",
        "herd_n",
    ),
}


class ErtMiss(KeyError):
    """A lookup found no exact entry. Carries the near neighbours it did find."""


def required_arguments(component: str, action: str) -> tuple[str, ...]:
    """The argument names ``(component, action)`` must carry, possibly empty."""
    if (component, action) in REQUIRED_ARGUMENTS:
        return REQUIRED_ARGUMENTS[(component, action)]
    return REQUIRED_ARGUMENTS.get((component, "*"), ())


def _freeze(value):
    """Make an argument value hashable and JSON-stable, preserving structure.

    A list becomes a tuple -- ``strides`` is a sequence and must stay one. It is
    deliberately NOT flattened to a string: a stringified stride list compares
    equal for two different dimension orders under some renderings, and the
    dimension ORDER is what makes a retile a retile (doc 23).
    """
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


@dataclass(frozen=True)
class ActionKey:
    """One priced thing: a component, an action, and the arguments it took."""

    component: str
    action: str
    arguments: tuple[tuple[str, object], ...]

    @staticmethod
    def of(component: str, action: str, **arguments) -> "ActionKey":
        """Build a key, freezing and sorting the arguments so order is not part of it."""
        frozen = tuple(sorted((k, _freeze(v)) for k, v in arguments.items()))
        return ActionKey(component=component, action=action, arguments=frozen)

    @property
    def argument_dict(self) -> dict[str, object]:
        return dict(self.arguments)

    def __str__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.arguments)
        return f"{self.component}.{self.action}({args})"


@dataclass(frozen=True)
class Cost:
    """What one action costs, with each number's provenance beside it.

    ``ns`` is the REPRESENTATIVE time and it is the MINIMUM over the samples,
    not the mean. Doc 23 §open item 1 settled that for this study -- "Compare
    minimums, not medians: the one-pass runs carried more host jitter, which
    flatters it on the median" -- and the min-to-min figures there agreed to 0.3
    points across a 12x shape range while the medians did not. ``ns_min`` /
    ``ns_max`` / ``ns_samples`` carry the spread beside it, because this project
    has published two wrong claims by comparing a fresh run against one recorded
    number, and an ERT holding a bare scalar per action would make that mistake
    unavoidable rather than merely easy.
    """

    ns: float | None = None
    ns_source: str = "absent"
    bytes: int | None = None
    bytes_source: str = "absent"
    provenance: str = ""
    condition: str = CONDITION_UNKNOWN
    ns_samples: int = 0
    ns_min: float | None = None
    ns_max: float | None = None

    def __post_init__(self) -> None:
        for name, value, source in (
            ("ns", self.ns, self.ns_source),
            ("bytes", self.bytes, self.bytes_source),
        ):
            if source not in COST_SOURCES:
                raise ValueError(
                    f"{name}_source={source!r} is not one of {list(COST_SOURCES)}"
                )
            if (value is None) != (source == "absent"):
                raise ValueError(
                    f"{name}={value!r} with {name}_source={source!r}: a value is "
                    "present exactly when its source is not 'absent'. An unpriced "
                    "action reading as a free one is the defect this pairing "
                    "exists to prevent."
                )
        if self.ns_source != "absent" and not self.provenance:
            raise ValueError(
                "a timed cost needs a provenance path: this project has been "
                "misrouted twice by a claim with no artifact behind it"
            )
        if self.ns is None:
            if self.ns_samples:
                raise ValueError("ns_samples without an ns is a count of nothing")
        else:
            # Default the spread to the single sample rather than leaving it
            # None: a reader must never have to guess whether an absent min
            # means "one sample" or "nobody recorded it".
            object.__setattr__(self, "ns_samples", max(1, self.ns_samples))
            if self.ns_min is None:
                object.__setattr__(self, "ns_min", self.ns)
            if self.ns_max is None:
                object.__setattr__(self, "ns_max", self.ns)
            if not self.ns_min <= self.ns <= self.ns_max:
                raise ValueError(
                    f"ns={self.ns!r} outside [{self.ns_min!r}, {self.ns_max!r}]"
                )

    @property
    def has_time(self) -> bool:
        return self.ns is not None

    @property
    def ns_spread(self) -> float | None:
        """``(max - min) / min`` over the samples, or ``None`` below two of them."""
        if self.ns_samples < 2 or not self.ns_min:
            return None
        return (self.ns_max - self.ns_min) / self.ns_min

    def observing(self, ns: float, *, provenance: str) -> "Cost":
        """This cost with one more measurement of the same action folded in.

        The representative moves to the new minimum; the spread widens. The
        provenance accumulates the file that produced the minimum, because that
        is the artifact a reader would go and check.
        """
        if self.ns is None:
            raise ValueError("cannot fold a sample into a cost with no ns")
        new_min = min(self.ns_min, ns)
        return Cost(
            ns=new_min,
            ns_source=self.ns_source,
            bytes=self.bytes,
            bytes_source=self.bytes_source,
            provenance=provenance if ns < self.ns_min else self.provenance,
            condition=self.condition,
            ns_samples=self.ns_samples + 1,
            ns_min=new_min,
            ns_max=max(self.ns_max, ns),
        )


@dataclass(frozen=True)
class Entry:
    """A row of the table."""

    key: ActionKey
    cost: Cost


class Ert:
    """The table. Insert-checked, lookup-exact, JSON-persistable."""

    def __init__(self, entries: dict[ActionKey, Cost] | None = None) -> None:
        self._entries: dict[ActionKey, Cost] = dict(entries or {})

    # -- building ---------------------------------------------------------

    def insert(self, key: ActionKey, cost: Cost, *, replace: bool = False) -> None:
        """Add an entry, refusing one whose required arguments are missing.

        The refusal is the module's whole point -- see the docstring on
        ``REQUIRED_ARGUMENTS``.
        """
        needed = required_arguments(key.component, key.action)
        present = set(key.argument_dict)
        if missing := [n for n in needed if n not in present]:
            raise ValueError(
                f"{key.component}.{key.action} needs arguments {list(needed)} and "
                f"is missing {missing}. Doc 44: a dma_transfer is a function of "
                "(n_words, n_dims, stride), not a scalar -- a counter reporting "
                "one number has already destroyed what the BD-stride walls need."
            )
        if not replace and key in self._entries and self._entries[key] != cost:
            raise ValueError(
                f"{key} is already priced differently ({self._entries[key]!r} vs "
                f"{cost!r}). Two runs of one candidate are two measurements: keep "
                "them apart in the key or pass replace=True deliberately."
            )
        self._entries[key] = cost

    def add(self, component: str, action: str, cost: Cost, **arguments) -> ActionKey:
        """``insert`` with the key built inline. Returns the key it made."""
        key = ActionKey.of(component, action, **arguments)
        self.insert(key, cost)
        return key

    def observe(self, key: ActionKey, ns: float, *, provenance: str) -> bool:
        """Fold a repeat measurement of an already-priced action in.

        Returns True if it merged into an existing entry, False if there was
        none and the caller should ``insert``. This is the path a sweep tree
        takes: 259 of the 1,467 passing sweep results measure an action another
        file already priced, and DROPPING them would leave the table asserting
        one number for an action this project has measured to vary by ~2.6%.
        """
        existing = self._entries.get(key)
        if existing is None:
            return False
        self._entries[key] = existing.observing(ns, provenance=provenance)
        return True

    # -- reading ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def entries(self) -> tuple[Entry, ...]:
        return tuple(Entry(k, v) for k, v in self._entries.items())

    def lookup(self, component: str, action: str, **arguments) -> Cost:
        """Exact lookup. Raises ``ErtMiss`` naming the near neighbours.

        No nearest-match fallback, ever. See the module docstring: two transfers
        agreeing on ``n_words`` and differing in ``strides`` are different
        objects to the BD allocator.
        """
        key = ActionKey.of(component, action, **arguments)
        try:
            return self._entries[key]
        except KeyError:
            pass
        near = [
            k
            for k in self._entries
            if k.component == component and k.action == action
        ]
        wanted = key.argument_dict
        differing = []
        for k in near[:8]:
            got = k.argument_dict
            diff = {
                n: (wanted.get(n), got.get(n))
                for n in sorted(set(wanted) | set(got))
                if wanted.get(n) != got.get(n)
            }
            differing.append(f"    {k}  differs in {diff}")
        raise ErtMiss(
            f"no entry for {key}. {len(near)} entr(ies) share its component and "
            "action; the first few differ as follows, and the difference is the "
            "information a nearest-match fallback would have erased:\n"
            + ("\n".join(differing) if differing else "    (none)")
        )

    def by_source(self) -> dict[str, int]:
        """How many entries carry each ``ns_source``. What ``report`` prints."""
        counts = {s: 0 for s in COST_SOURCES}
        for cost in self._entries.values():
            counts[cost.ns_source] += 1
        return counts

    def bytes_by_source(self) -> dict[str, int]:
        counts = {s: 0 for s in COST_SOURCES}
        for cost in self._entries.values():
            counts[cost.bytes_source] += 1
        return counts

    # -- persistence ------------------------------------------------------

    def to_json(self) -> dict:
        return {
            "ert_format_version": ERT_FORMAT_VERSION,
            "records": [
                {
                    "component": k.component,
                    "action": k.action,
                    "arguments": [[n, _unfreeze(v)] for n, v in k.arguments],
                    "ns": c.ns,
                    "ns_source": c.ns_source,
                    "ns_samples": c.ns_samples,
                    "ns_min": c.ns_min,
                    "ns_max": c.ns_max,
                    "bytes": c.bytes,
                    "bytes_source": c.bytes_source,
                    "provenance": c.provenance,
                    "condition": c.condition,
                }
                for k, c in self._entries.items()
            ],
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2) + "\n", encoding="utf-8")
        return path

    @staticmethod
    def from_json(blob: dict) -> "Ert":
        version = blob.get("ert_format_version")
        if version != ERT_FORMAT_VERSION:
            raise ValueError(
                f"ERT format v{version!r}, this module is v{ERT_FORMAT_VERSION}; "
                "a record whose shape changed must be rejected, not reinterpreted"
            )
        ert = Ert()
        for record in blob["records"]:
            key = ActionKey.of(
                record["component"],
                record["action"],
                **{n: v for n, v in record["arguments"]},
            )
            ert.insert(
                key,
                Cost(
                    ns=record["ns"],
                    ns_source=record["ns_source"],
                    ns_samples=record.get("ns_samples", 0),
                    ns_min=record.get("ns_min"),
                    ns_max=record.get("ns_max"),
                    bytes=record["bytes"],
                    bytes_source=record["bytes_source"],
                    provenance=record["provenance"],
                    condition=record["condition"],
                ),
            )
        return ert

    @staticmethod
    def load(path: str | Path) -> "Ert":
        return Ert.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def _unfreeze(value):
    """Tuples back to lists for JSON. Inverse of ``_freeze`` up to list/tuple."""
    if isinstance(value, tuple):
        return [_unfreeze(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Seeds. Each reads an artifact that ALREADY EXISTS in this tree; nothing here
# runs a measurement, and nothing here invents a constant.
# ---------------------------------------------------------------------------

#: The sweep JSON fields that become a gemm action's arguments. Named rather
#: than taken as "everything in the candidate dict", so a new candidate field
#: appearing upstream changes the KEY only when someone decides it should.
_GEMM_ARGUMENT_FIELDS = (
    "tile_m",
    "tile_k_l2",
    "tile_k_l1",
    "tile_n",
    "herd_m",
    "herd_n",
)


def seed_from_gemm_sweep(
    ert: Ert,
    paths: list[str],
    *,
    condition: str = CONDITION_UNKNOWN,
    skip_failed: bool = True,
) -> dict[str, int]:
    """Add one MEASURED gemm entry per sweep result JSON. Returns a reason count.

    The sweep tree (``sweep/results/baseline_{512,768,1024}/*.json``) is this
    study's largest body of measured device latency: one file per (shape,
    candidate) with ``latency_us`` beside the full tiling that produced it.
    That is an ERT row already -- component ``gemm``, action the tiling METHOD,
    arguments the shape and the six tile/herd factors -- and seeding from it
    costs no device time.

    The returned counts are named rather than a bare ``skipped`` total, because
    the reasons are not interchangeable:

    - ``added``    -- a new priced action.
    - ``merged``   -- a REPEAT measurement of an action already in the table,
      folded in through ``Ert.observe``. 259 of the 1,467 passing sweep results
      are these, and the pairs differ by ~2.6%. Reporting them as "skipped"
      would have hidden that the table's scalar is a representative rather than
      the value.
    - ``failed``   -- ``status`` is not ``passed``. A latency from a candidate
      that failed its numeric check measures the wrong thing.
    - ``incomplete`` -- the file does not carry the full argument set, so it
      cannot be keyed. A shorter key is the defect ``REQUIRED_ARGUMENTS``
      exists to refuse.
    - ``unreadable`` -- the file did not parse.
    """
    counts = {
        "added": 0,
        "merged": 0,
        "failed": 0,
        "incomplete": 0,
        "unreadable": 0,
    }
    for path in paths:
        try:
            blob = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            counts["unreadable"] += 1
            continue
        shape = blob.get("shape") or {}
        candidate = blob.get("candidate") or {}
        latency_us = blob.get("latency_us")
        if latency_us is None or (skip_failed and blob.get("status") != "passed"):
            counts["failed"] += 1
            continue
        method = candidate.get("method")
        if method is None or any(shape.get(d) is None for d in ("M", "K", "N")):
            counts["incomplete"] += 1
            continue
        arguments = {"M": shape["M"], "K": shape["K"], "N": shape["N"]}
        for name in _GEMM_ARGUMENT_FIELDS:
            arguments[name] = candidate.get(name)
        if any(v is None for v in arguments.values()):
            # An incomplete argument set is a refusal, not a shorter key --
            # see REQUIRED_ARGUMENTS.
            counts["incomplete"] += 1
            continue
        # `role` is recorded but is NOT an argument: it names which projection
        # this shape happened to serve, and two roles at one shape+tiling are
        # the same priced action. Keeping it out of the key is what lets the
        # table be reused across the layer.
        ns = float(latency_us) * 1000.0
        key = ActionKey.of("gemm", method, **arguments)
        if ert.observe(key, ns, provenance=str(path)):
            counts["merged"] += 1
            continue
        ert.insert(
            key,
            Cost(
                ns=ns,
                ns_source="measured",
                provenance=str(path),
                condition=condition,
            ),
        )
        counts["added"] += 1
    return counts


def seed_from_transfers(
    ert: Ert,
    transfers,
    *,
    provenance: str,
) -> int:
    """Add one COUNTED ``shim_dma.dma_transfer`` entry per distinct descriptor.

    ``transfers`` is ``balance.parse_transfers``' output. Each becomes an entry
    whose arguments are the BD's own shape -- ``n_words``, ``n_dims``,
    ``strides``, ``element_bytes`` -- and whose cost is ``bytes`` COUNTED and
    ``ns`` ABSENT.

    ``ns`` is absent on purpose and it is not an oversight: doc 33 records that
    the AIR-native bandwidth operator was DEFERRED, so this tree holds no
    measured shim byte rate to convert a byte count into a time. Writing one in
    from iron's 67.9-70.9 GB/s would be a cross-toolchain constant wearing a
    measurement's label, which doc 33 explicitly declines to do.
    """
    added = 0
    seen: set[ActionKey] = set()
    for transfer in transfers:
        if transfer.n_words is None:
            continue
        key = ActionKey.of(
            "shim_dma",
            "dma_transfer",
            n_words=transfer.n_words,
            n_dims=transfer.n_dims,
            strides=transfer.strides,
            element_bytes=transfer.element_bytes,
        )
        if key in seen:
            continue
        seen.add(key)
        ert.insert(
            key,
            Cost(
                bytes=transfer.n_words * transfer.element_bytes,
                bytes_source="counted",
                provenance=provenance,
            ),
            replace=True,
        )
        added += 1
    return added


def report(ert: Ert) -> str:
    """A text summary that states the measured/counted/modelled split up front."""
    ns_counts = ert.by_source()
    byte_counts = ert.bytes_by_source()
    lines = [
        f"## ERT -- {len(ert)} entries\n",
        "| number | measured | counted | modelled | absent |",
        "|---|---|---|---|---|",
        "| ns | {measured} | {counted} | {modelled} | {absent} |".format(**ns_counts),
        "| bytes | {measured} | {counted} | {modelled} | {absent} |".format(
            **byte_counts
        ),
        "",
        "`measured` = a stopwatch on hardware; `counted` = read exactly off a "
        "compiled artifact; `modelled` = computed from a stated assumption; "
        "`absent` = nothing known, value is None.",
    ]
    components: dict[tuple[str, str], int] = {}
    for entry in ert.entries():
        components[(entry.key.component, entry.key.action)] = (
            components.get((entry.key.component, entry.key.action), 0) + 1
        )
    lines.append("\n| component | action | entries |")
    lines.append("|---|---|---|")
    for (component, action), count in sorted(components.items()):
        lines.append(f"| `{component}` | `{action}` | {count} |")

    repeated = [e for e in ert.entries() if e.cost.ns_samples > 1]
    if repeated:
        spreads = sorted(
            (e.cost.ns_spread for e in repeated if e.cost.ns_spread is not None)
        )
        worst = spreads[-1] if spreads else 0.0
        median = spreads[len(spreads) // 2] if spreads else 0.0
        lines.append(
            f"\n{len(repeated)} action(s) were measured more than once; their "
            f"(max-min)/min spread has median {median:.1%} and worst "
            f"{worst:.1%}. `ns` is the MINIMUM over the samples (doc 23 §open "
            "item 1: compare minimums, not medians), and `ns_min`/`ns_max` "
            "carry the spread so no caller has to compare a fresh run against "
            "a single recorded number."
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--seed-gemm-sweep",
        action="append",
        default=[],
        metavar="DIR",
        help="a sweep results directory of *.json; repeatable",
    )
    ap.add_argument(
        "--seed-routed-design",
        action="append",
        default=[],
        metavar="PROJECT_DIR",
        help="an aircc project directory to count shim descriptors from; repeatable",
    )
    ap.add_argument(
        "--condition",
        default=CONDITION_UNKNOWN,
        help="the NPU power mode the seeded latencies were recorded under. "
        "NOT defaulted to turbo: an unconditioned latency is unconditioned on "
        "the axis doc 32 measures a ~15-20x error across.",
    )
    ap.add_argument("--out", default=None, help="write the table here as JSON")
    args = ap.parse_args(argv)

    ert = Ert()
    for directory in args.seed_gemm_sweep:
        paths = sorted(glob.glob(os.path.join(directory, "*.json")))
        counts = seed_from_gemm_sweep(ert, paths, condition=args.condition)
        print(
            f"[ert] {directory}: {len(paths)} files -> "
            + ", ".join(f"{v} {k}" for k, v in counts.items())
        )

    if args.seed_routed_design:
        import balance  # noqa: E402  -- only needed for this seed

        for project in args.seed_routed_design:
            design = balance.routed_design_path(project)
            text = Path(design).read_text(encoding="utf-8")
            transfers = balance.parse_transfers(text)
            added = seed_from_transfers(ert, transfers, provenance=str(design))
            print(
                f"[ert] {design}: {added} counted shim_dma descriptors from "
                f"{len(transfers)} transfers"
            )

    print(report(ert))
    if args.out:
        ert.save(args.out)
        print(f"[ert] wrote {args.out}")
    return 0 if len(ert) else 1


if __name__ == "__main__":
    sys.exit(main())
