# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Sample SoC power without root, and summarize it into the schema's power block.

    python3 study/power.py --probe            # what this host can measure
    python3 study/power.py --sample 2.0       # take a 2 s reading

CONTRACT
    ``open_monitor(backend)`` returns a context manager that samples power for
    the duration of the ``with`` block; ``monitor.stats()`` returns exactly
    ``schema``'s power columns. ``summarize(samples)`` is the statistics on
    their own, so a caller with samples from somewhere else gets the same
    outlier policy.

    ``backend="auto"`` picks the first available of ``rapl_package`` and
    ``amdgpu_ppt``, and falls back to ``none`` -- a monitor that yields nothing
    and reports an all-``None`` block -- rather than raising. A study that
    cannot measure power must still be able to measure latency.

WHAT THIS MEASURES, AND WHAT IT DOES NOT -- READ BEFORE QUOTING A NUMBER
    Doc 09 measured the options on this host and the load-bearing finding is
    that **no sensor on this platform measures the NPU**:

    - ``intel-rapl:0`` ``package-0`` is CPU PACKAGE energy. Readable unprivileged
      by differencing ``energy_uj``; measured 19.96 W over 2.00 s.
    - ``amdgpu``'s ``power1_average`` (label ``PPT``) is the GPU/SoC rail;
      measured 22.05 W.
    - The ``amdxdna`` driver exposes ``power_state`` -- a PM state -- and **no
      energy or power counter at all**.

    On an APU the NPU shares that SoC envelope, so neither number isolates it.
    A power comparison BETWEEN EXECUTION MODES therefore partly measures host
    CPU work, and the modes differ in exactly that. "Watts per token on the NPU"
    is not measurable here; "SoC watts while executing this mode" is, and that
    is what these columns mean.

WHY THE SAMPLING BACKEND IS REPLACED AND THE STATISTICS ARE PORTED
    iron's backend shells out to ``sudo -n turbostat --show PkgWatt``. On this
    host ``sudo -n`` fails -- a password is required -- so iron's backend cannot
    run unattended here AT ALL, and unprivileged ``turbostat --no-msr`` exits 0
    while emitting no samples, which is the worst failure shape available: a
    power column full of nothing behind a zero exit status.

    So doc 09's instruction is followed exactly: port the STATISTICS -- the
    modified-Z outlier filter, the percentile fallback, the probe-completeness
    policy -- and replace the sampling backend. Nothing here needs root, which
    is also what Phase G's unattended runner needs.

FOOTGUNS
    - **RAPL is a monotonic ENERGY counter, not a power reading.** Power is the
      difference over an interval, and the counter WRAPS at ``max_energy_range_uj``.
      The wrap is handled; a single reading is meaningless and there is no API
      here that returns one.
    - **Two samplers, two different rails.** Never mix a ``rapl_package`` row
      with an ``amdgpu_ppt`` row in one comparison. ``power_backend`` records
      which produced the block precisely so that mistake is visible, and it is
      the first thing to check before differencing two power numbers.
    - **The filtered and raw statistics are both persisted.** The filter runs
      only with >= 10 samples and >= 6 retained, and only when it actually
      reduces the spread; a reader can always tell what it did instead of
      trusting that it was reasonable. iron's policy, kept.
    - **The outlier filter has a blind spot, and the raw columns are how you
      see past it.** Both dispersion measures degenerate when most samples are
      identical: a quantized sampler drives the median absolute deviation to 0,
      and if more than three quarters of the samples share a value the
      interquartile fallback is 0 as well, so nothing is flagged and a lone
      spike survives into the filtered mean. Kept as iron has it -- the
      alternative is inventing a dispersion measure for a distribution that has
      none -- and pinned in ``test_power.py``. Compare ``max_power_w`` against
      ``avg_power_w`` rather than trusting ``power_outlier_filter_applied``.
    - **A short block yields few samples.** At the default 100 ms interval a
      50 ms timed region produces zero or one, and the summary of one sample has
      no standard deviation. ``probe_is_complete`` is the check a runner should
      make before recording the block rather than after.
    - **This does not subtract idle.** iron differences a quiescent baseline
      measured just before the run, which turns its numbers into "power above
      idle" and makes them incomparable with an absolute reading. The baseline
      is measured and REPORTED here (``quiescent_power_w``) and never
      subtracted: the columns are absolute SoC watts.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import schema  # noqa: E402

#: Backends in preference order for ``auto``. ``rapl_package`` first because it
#: is an energy counter -- differencing it is exact over the interval -- where
#: the hwmon reading is an instantaneous average the driver computes.
BACKENDS: tuple[str, ...] = ("rapl_package", "amdgpu_ppt", "none")

_RAPL_ROOT = Path("/sys/class/powercap")
_HWMON_ROOT = Path("/sys/class/hwmon")

#: iron's outlier policy, unchanged: modified Z over the median absolute
#: deviation, applied only where there is enough data for it to mean anything.
OUTLIER_MODIFIED_Z_THRESHOLD = 3.5
OUTLIER_MIN_SAMPLE_COUNT = 10
OUTLIER_MIN_RETAINED_COUNT = 6

DEFAULT_SAMPLE_INTERVAL_SEC = 0.1
#: Below this many samples a block is a reading, not a distribution.
MIN_USEFUL_SAMPLE_COUNT = 6


def empty_power_columns() -> dict[str, object]:
    """Every schema power column, all ``None``. The unmeasured block."""
    return {name: None for name in schema.POWER_FIELDNAMES}


# ---------------------------------------------------------------------------
# Sampling backends. Each is (name, is_available, read) where read() returns
# either watts directly or an energy reading to be differenced.
# ---------------------------------------------------------------------------


def rapl_package_zone() -> Path | None:
    """The ``package-0`` powercap zone, or ``None`` if unreadable.

    Located by NAME rather than by assuming ``intel-rapl:0``: the numbering is
    enumeration order, and a machine with a second package or a psys zone can
    put ``package-0`` somewhere else. Readability is TESTED, not inferred from
    the file existing -- the sub-zones exist and are root-only on this host.
    """
    if not _RAPL_ROOT.is_dir():
        return None
    for zone in sorted(_RAPL_ROOT.glob("intel-rapl:*")):
        name_file = zone / "name"
        energy_file = zone / "energy_uj"
        try:
            if name_file.read_text().strip() != "package-0":
                continue
            energy_file.read_text()
        except OSError:
            continue
        return zone
    return None


def amdgpu_ppt_input() -> Path | None:
    """The ``amdgpu`` hwmon ``power1_average``, or ``None`` if unreadable."""
    if not _HWMON_ROOT.is_dir():
        return None
    for hwmon in sorted(_HWMON_ROOT.glob("hwmon*")):
        try:
            if (hwmon / "name").read_text().strip() != "amdgpu":
                continue
            average = hwmon / "power1_average"
            average.read_text()
        except OSError:
            continue
        return average
    return None


def available_backends() -> tuple[str, ...]:
    """Which samplers this host can actually run, in preference order."""
    found = []
    if rapl_package_zone() is not None:
        found.append("rapl_package")
    if amdgpu_ppt_input() is not None:
        found.append("amdgpu_ppt")
    found.append("none")
    return tuple(found)


def resolve_backend(requested: str) -> str:
    """``auto`` -> the best available; anything else validated and returned."""
    if requested == "auto":
        return available_backends()[0]
    if requested not in BACKENDS:
        raise ValueError(
            f"unknown power backend {requested!r}; known are {list(BACKENDS)} "
            "plus 'auto'"
        )
    return requested


# ---------------------------------------------------------------------------
# Statistics. Ported from iron unchanged in policy; see the module docstring.
# ---------------------------------------------------------------------------


def _percentile(ordered: list[float], fraction: float) -> float | None:
    if not ordered:
        return None
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * float(fraction)
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return float(ordered[low])
    weight = position - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def detect_outliers(samples: list[float]) -> list[bool]:
    """Modified-Z outlier mask, with an IQR fallback for a zero MAD.

    The fallback matters: a sampler quantized to whole watts produces runs of
    identical values, the median absolute deviation is then exactly 0, and the
    modified Z is a division by zero for every sample. iron's shape, kept.
    """
    if len(samples) < 5:
        return [False] * len(samples)

    median = statistics.median(samples)
    mad = statistics.median([abs(s - median) for s in samples])
    if mad > 0:
        return [
            abs(0.6745 * (s - median) / mad) > OUTLIER_MODIFIED_Z_THRESHOLD
            for s in samples
        ]

    ordered = sorted(samples)
    q1 = _percentile(ordered, 0.25)
    q3 = _percentile(ordered, 0.75)
    if q1 is None or q3 is None or q3 - q1 <= 0:
        return [False] * len(samples)
    span = q3 - q1
    low, high = q1 - 1.5 * span, q3 + 1.5 * span
    return [s < low or s > high for s in samples]


def summarize(samples: list[float], *, backend: str | None = None) -> dict[str, object]:
    """The schema power columns for these samples. Empty input -> all ``None``.

    The filter is applied only when it removes something AND leaves enough
    behind AND actually reduces the standard deviation. That last condition is
    iron's and is worth keeping: a filter that widens the spread has found
    structure, not outliers, and dropping the samples would be editing the
    measurement.

    Takes no elapsed time: iron derives an ``energy_j`` from one, and schema v2
    has no energy column. Adding the argument "for later" would put a number in
    the API that nothing can persist.
    """
    columns = empty_power_columns()
    columns["power_backend"] = backend
    if not samples:
        return columns

    raw_mean = statistics.fmean(samples)
    raw_std = statistics.stdev(samples) if len(samples) >= 2 else 0.0

    retained = list(samples)
    mask = [False] * len(samples)
    applied = False
    if len(samples) >= OUTLIER_MIN_SAMPLE_COUNT:
        candidate_mask = detect_outliers(samples)
        candidate = [s for s, bad in zip(samples, candidate_mask) if not bad]
        if OUTLIER_MIN_RETAINED_COUNT <= len(candidate) < len(samples):
            candidate_std = statistics.stdev(candidate) if len(candidate) >= 2 else 0.0
            if candidate_std < raw_std:
                retained, mask, applied = candidate, candidate_mask, True

    mean = statistics.fmean(retained)
    columns.update(
        {
            "avg_power_w": mean,
            "min_power_w": min(retained),
            "max_power_w": max(retained),
            "power_sample_count": len(retained),
            "power_std_w": statistics.stdev(retained) if len(retained) >= 2 else 0.0,
            "raw_avg_power_w": raw_mean,
            "raw_min_power_w": min(samples),
            "raw_max_power_w": max(samples),
            "raw_power_sample_count": len(samples),
            "raw_power_std_w": raw_std,
            "power_outlier_sample_count": sum(1 for bad in mask if bad),
            "power_outlier_filter_applied": applied,
        }
    )
    return columns


def probe_is_complete(
    *,
    completed_runs: int,
    min_runs: int,
    elapsed_sec: float,
    min_duration_sec: float,
    sample_count: int,
    min_sample_count: int = MIN_USEFUL_SAMPLE_COUNT,
) -> bool:
    """Whether a power probe ran long enough to be worth recording.

    All three conditions, not any: enough iterations to be the workload, enough
    wall time for the sampler to fire, and enough samples for the statistics to
    be a distribution. iron's policy, and the reason a runner should extend a
    short case rather than record a one-sample mean.
    """
    return (
        completed_runs >= int(min_runs)
        and elapsed_sec >= float(min_duration_sec)
        and sample_count >= int(min_sample_count)
    )


def resolve_sample_interval(
    *,
    requested_sec: float,
    estimated_window_sec: float | None,
    min_sample_count: int = MIN_USEFUL_SAMPLE_COUNT,
    floor_sec: float = 0.01,
) -> float:
    """Shrink the interval so a short window still yields enough samples."""
    interval = float(requested_sec)
    if not estimated_window_sec or estimated_window_sec <= 0:
        return interval
    target = float(estimated_window_sec) / float(min_sample_count)
    return max(float(floor_sec), min(interval, target))


# ---------------------------------------------------------------------------
# The monitors.
# ---------------------------------------------------------------------------


class _NullMonitor:
    """Yields nothing and reports an unmeasured block. Never raises."""

    backend = "none"

    quiescent_power_w: float | None = None

    def sample_count(self) -> int:
        return 0

    def stats(self) -> dict[str, object]:
        return summarize([], backend="none")


class _SamplingMonitor:
    """Polls a reader on a background thread for the life of the block.

    A thread rather than a subprocess because both readers are a ``read()`` of
    a sysfs file -- microseconds, no fork -- so the sampling cost sits inside
    the timed region as a few microseconds of Python rather than as a process
    spawn. That is the whole reason this can sample a 50 ms window at all,
    which iron's ``turbostat`` subprocess cannot.
    """

    def __init__(self, backend: str, read_watts, *, interval_sec: float):
        self.backend = backend
        self._read = read_watts
        self._interval = max(0.001, float(interval_sec))
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.quiescent_power_w: float | None = None

    def __enter__(self):
        # One reading before the block so the report can say what idle looked
        # like. Reported, never subtracted -- see the module docstring.
        #
        # It is None for `rapl_package` BY CONSTRUCTION and that is not a bug:
        # one read of a monotonic energy counter is not a power reading, so the
        # first call establishes the baseline the loop then differences against.
        # An instantaneous backend (`amdgpu_ppt`) does report a value here. The
        # column is None rather than 0.0 for the same reason everything else in
        # this module is: an unmeasured quantity must not read as a measured
        # zero.
        self.quiescent_power_w = self._read()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._interval * 4.0))
        return False

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                watts = self._read()
            except OSError:
                # A counter that disappears mid-run (driver reload) ends
                # sampling; the samples already taken stay valid.
                return
            if watts is not None:
                self._samples.append(watts)
            self._stop.wait(self._interval)

    def sample_count(self) -> int:
        return len(self._samples)

    def stats(self) -> dict[str, object]:
        return summarize(list(self._samples), backend=self.backend)


def _rapl_reader(zone: Path):
    """A closure returning watts since its previous call, wrap-safe."""
    energy_file = zone / "energy_uj"
    try:
        wrap_uj = int((zone / "max_energy_range_uj").read_text().strip())
    except (OSError, ValueError):
        wrap_uj = None
    state = {"uj": None, "t": None}

    def read() -> float | None:
        now = time.perf_counter()
        uj = int(energy_file.read_text().strip())
        previous_uj, previous_t = state["uj"], state["t"]
        state["uj"], state["t"] = uj, now
        if previous_uj is None:
            # The first call establishes the baseline; there is no interval to
            # divide by yet, and returning 0.0 would put a fabricated sample in
            # the distribution.
            return None
        delta_uj = uj - previous_uj
        if delta_uj < 0 and wrap_uj:
            delta_uj += wrap_uj
        interval = now - previous_t
        if interval <= 0 or delta_uj < 0:
            return None
        return delta_uj / 1e6 / interval

    return read


def _hwmon_reader(path: Path):
    """A closure returning the driver's averaged watts. microwatts -> watts."""

    def read() -> float | None:
        raw = path.read_text().strip()
        try:
            return int(raw) / 1e6
        except ValueError:
            return None

    return read


@contextmanager
def open_monitor(
    backend: str = "auto", *, interval_sec: float = DEFAULT_SAMPLE_INTERVAL_SEC
):
    """Sample power for the block. Always yields a monitor, never raises."""
    resolved = resolve_backend(backend)
    if resolved == "rapl_package":
        zone = rapl_package_zone()
        if zone is not None:
            with _SamplingMonitor(
                "rapl_package", _rapl_reader(zone), interval_sec=interval_sec
            ) as monitor:
                yield monitor
            return
    elif resolved == "amdgpu_ppt":
        path = amdgpu_ppt_input()
        if path is not None:
            with _SamplingMonitor(
                "amdgpu_ppt", _hwmon_reader(path), interval_sec=interval_sec
            ) as monitor:
                yield monitor
            return
    yield _NullMonitor()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--sample", type=float, default=None, metavar="SECONDS")
    ap.add_argument("--interval", type=float, default=DEFAULT_SAMPLE_INTERVAL_SEC)
    ap.add_argument("--probe", action="store_true", help="report what is available")
    args = ap.parse_args(argv)

    if args.probe or args.sample is None:
        print(f"[power] rapl package-0 zone : {rapl_package_zone()}")
        print(f"[power] amdgpu PPT input   : {amdgpu_ppt_input()}")
        print(f"[power] available          : {list(available_backends())}")
        print(f"[power] auto resolves to   : {resolve_backend('auto')}")
        print(
            "[power] NOTE no sensor on this platform measures the NPU; these "
            "are SoC rails. See the module docstring."
        )
        if args.sample is None:
            return 0

    started = time.perf_counter()
    with open_monitor(args.backend, interval_sec=args.interval) as monitor:
        # A busy wait would change what is being measured; sleeping measures
        # idle, which is what a bare probe should report.
        time.sleep(args.sample)
        elapsed = time.perf_counter() - started
        columns = monitor.stats()
        quiescent = monitor.quiescent_power_w

    print(f"[power] backend {columns['power_backend']} over {elapsed:.2f} s")
    if quiescent is not None:
        print(f"[power]   quiescent (reported, never subtracted): {quiescent:.2f} W")
    if columns["avg_power_w"] is None:
        print("[power]   no samples")
        return 1
    print(
        f"[power]   avg {columns['avg_power_w']:.2f} W  "
        f"min {columns['min_power_w']:.2f}  max {columns['max_power_w']:.2f}  "
        f"std {columns['power_std_w']:.3f}  n {columns['power_sample_count']}"
        f"/{columns['raw_power_sample_count']}  "
        f"filtered {columns['power_outlier_filter_applied']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
