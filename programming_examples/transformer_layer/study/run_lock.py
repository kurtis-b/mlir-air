# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One writer per results file. NOT the device lock -- read the second section.

    with run_lock.hold(run_lock.lock_path_for("results/coarse.csv"),
                       study="sequence ladder"):
        ...

CONTRACT
    ``hold`` takes a non-blocking exclusive ``flock`` on a lock file beside the
    output, writes the holder's pid and study name into it so a refusal can name
    who is holding it, and releases on exit. A second holder does not queue --
    it raises ``StudyAlreadyRunning`` immediately, with the first holder's
    identity in the message.

    ``lock_path_for(output)`` is the naming rule: ``results/coarse.csv`` locks
    on ``results/coarse.csv.lock``. Deriving it from the output rather than
    naming it separately is what stops two runners writing one file while
    holding two different locks.

WHY NON-BLOCKING, AND WHY IT IS NOT THE DEVICE LOCK
    Two runs writing one results CSV is a mistake, not contention: the second
    would overwrite the first's rows, and waiting politely for the chance to do
    that is worse than refusing. So this fails fast and says who holds it.

    **The device is scheduled somewhere else entirely.** ``agents/scripts/devq.sh``
    is the queue for anything that dispatches, and it is the ONLY thing that
    should touch ``/tmp/mlir-air-npu.lock``. This lock is per OUTPUT FILE and
    holds no claim on the NPU. Two facts follow:

    - Holding this does not make a measurement safe to run. Wrap the whole
      invocation in ``devq.sh run --class measure -- ...`` as well.
    - Never point this at ``/tmp/mlir-air-npu.lock`` or ``/tmp/npu.lock``.
      Both are BSD ``flock(2)`` on inodes the queue and ``KernelCache`` already
      hold; taking either here would deadlock against the wrapper that launched
      the run. Doc 15 keeps those two inodes deliberately separate, and this is
      a third, unrelated one.

FOOTGUNS
    - **POSIX only.** ``fcntl.flock`` has no Windows equivalent; iron's study
      tier is POSIX-only for this reason and doc 09 says to state it, so it is
      stated here.
    - **The lock is on the open file description, not on the path.** Deleting
      the lock file while it is held does not release it, and a later run
      creating a fresh file at the same path will acquire a DIFFERENT lock and
      both will write. Do not clean lock files out of a live results tree.
    - **It protects the file, not the tree.** Two runs writing different CSVs
      into one results root each take their own lock and both proceed, which is
      intended -- a per-mode ladder does exactly that.
    - The holder record is truncated on release, so a stale file with empty
      contents means "released", not "held by an unknown process".
"""

from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class StudyAlreadyRunning(RuntimeError):
    """Another process holds the lock for this output. Carries its identity."""


def lock_path_for(output_path: str | Path) -> Path:
    """``results/coarse.csv`` -> ``results/coarse.csv.lock``.

    Appends rather than replacing the suffix, so ``a.csv`` and ``a.json`` in one
    directory do not share a lock and serialize against each other for no
    reason.
    """
    output_path = Path(output_path)
    return output_path.with_name(output_path.name + ".lock")


@contextmanager
def hold(lock_path: str | Path, *, study: str) -> Iterator[Path]:
    """Hold the lock for the duration of the block, or raise immediately."""
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            holder = handle.read().strip().replace("\n", " ")
            raise StudyAlreadyRunning(
                f"{study} is already running for {lock_path}"
                + (f" ({holder})" if holder else "")
                + ". This is a per-output-file lock, not the device queue: a "
                "second writer would overwrite the first's rows."
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\nstudy={study}\n")
        handle.flush()
        try:
            yield lock_path
        finally:
            # Truncate before release so a released lock file is empty rather
            # than naming a process that has moved on.
            handle.seek(0)
            handle.truncate()
            handle.flush()
