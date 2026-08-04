# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Runlist aggregation and the dispatch vector for `KernelCache`.

A dispatch sequence is a list of `DispatchStep`s. This module turns one into as few
XRT submissions as the hardware actually allows, and reports what it did as a
`DispatchVector` — the six-field record defined in
`docs/plans/transformer-layer-execution-studies/03-measurement-model.md`, which is
what distinguishes the four execution modes from each other. There is one
implementation of those counts and every mode calls it; a per-mode reimplementation
of "what counts as a submission" would make the comparison meaningless.

Buffer ownership, liveness and sync are `bo_pool.py`'s job; the rules both modules
implement are in
`docs/plans/transformer-layer-execution-studies/05b-phase-b-buffer-rules.md`.

Footguns:

- **Runs may only share a runlist when they share a hardware configuration.** On
  XRT 2.21.0 / NPU2 that means one artifact: a full ELF cannot be rebound into
  another artifact's `hw_context`, and `hw_context::add_config` takes exactly one
  config ELF. `plan_submissions` therefore splits the sequence at every artifact
  change. See `05a-phase-b-runlist-spike-result.md` for the measurements.

- **Never build a cross-configuration runlist "to see what happens".** It executes,
  raises nothing, times out on nothing, and returns wrong numbers for every entry
  except the one that supplied the context (05a §4). `RunlistSplitError` exists so
  that failure mode is unreachable rather than merely discouraged.

- **The two ABIs number their arguments differently.** Under the ELF ABI buffers
  start at index 0 and the instruction stream is inside the ELF. Under the xclbin
  ABI index 0 is the opcode, 1 the instruction BO, 2 the instruction length, and
  buffers start at 3. `_bind_args` is the only place that knows this; do not
  open-code either layout elsewhere.

- **A split sequence reports its real submission count.** `DispatchVector` never
  claims one submission for work that took several — that would collapse `runlist`
  into `offload` in the results, which is exactly the distinction the study exists
  to measure.
"""

import time
from dataclasses import dataclass, field

from shared.infra.bo_pool import DispatchStep  # re-exported for callers

__all__ = [
    "DispatchStep",
    "DispatchVector",
    "RunlistSplitError",
    "RunlistExecutionError",
    "Submission",
    "plan_submissions",
    "OPCODE_DPU",
    "N_XCLBIN_PREFIX_ARGS",
]

#: `ert_start_npu` opcode used by the xclbin ABI's generic MLIR_AIE kernel.
OPCODE_DPU = 3

#: opcode, instruction BO, instruction length — the xclbin ABI's fixed prefix.
N_XCLBIN_PREFIX_ARGS = 3


class RunlistSplitError(RuntimeError):
    """A single runlist was required but the sequence spans hardware configurations.

    Raised at build time, never worked around: see the module docstring and
    `05a-phase-b-runlist-spike-result.md` §4 for why emitting the runlist anyway
    is worse than failing.
    """


class RunlistExecutionError(RuntimeError):
    """A submission failed, attributed to the entry that failed (rule E1).

    Attributes:
        submission_index: which submission of the sequence failed.
        entry_index: index of the first non-completed entry within it.
        step: the `DispatchStep` that entry came from.
        state: the XRT command state reported for it, or None if unavailable.
    """

    def __init__(self, message, submission_index, entry_index, step, state):
        super().__init__(message)
        self.submission_index = submission_index
        self.entry_index = entry_index
        self.step = step
        self.state = state


@dataclass
class DispatchVector:
    """The six-field dispatch record from 03-measurement-model.md.

    Written definitions, because copying iron's column names does not define what
    they mean under AIR's timing and synchronization model:

    host_submissions: host->device work submissions for the sequence. One runlist
        counts as one however many entries it holds; one plain kernel call counts
        as one. A sequence split across artifact configurations counts each
        submission, so it can never be mistaken for an aggregated one.
    runlist_entries: total `xrt.run` objects submitted across those submissions.
        Equals the number of dispatch steps. `entries_per_submission` derives the
        per-submission figure the CSV wants.
    air_launches: `air.launch` operations in the compiled modules the sequence
        touched, summed per step. Recorded at compile time from the MLIR module,
        because the runtime artifact does not carry it.
    herd_launches: `air.herd` operations, same accounting.
    sync_boundaries: `bo.sync()` calls issued for the sequence, both directions.
        This is the number the dirty-bit discipline reduces, and the number that
        makes `offload` and `fused_elf` differ even at equal submission counts.
    bytes_transferred: bytes moved by those syncs.
    """

    host_submissions: int = 0
    runlist_entries: int = 0
    air_launches: int = 0
    herd_launches: int = 0
    sync_boundaries: int = 0
    bytes_transferred: int = 0
    submission_ms: float = 0.0
    sync_ms: float = 0.0
    per_submission_entries: tuple = ()

    def entries_per_submission(self):
        """Mean entries per submission; 0.0 for an empty sequence."""
        if not self.host_submissions:
            return 0.0
        return self.runlist_entries / self.host_submissions

    def as_row(self):
        """Flat dict for the study CSV. Keys are the schema's field names."""
        return {
            "host_submissions_per_layer": self.host_submissions,
            "runlist_entries_per_submission": self.entries_per_submission(),
            "air_launches_per_elf": self.air_launches,
            "herd_launches": self.herd_launches,
            "sync_boundaries": self.sync_boundaries,
            "bytes_transferred": self.bytes_transferred,
        }


@dataclass
class Submission:
    """One contiguous run of steps sharing an artifact, hence a hw_context."""

    artifact: str
    steps: tuple = ()
    first_index: int = 0

    def __len__(self):
        return len(self.steps)


def plan_submissions(steps, artifact_of, require_single=False):
    """Group a dispatch sequence into the submissions the hardware allows.

    Consecutive steps whose artifacts share a hardware configuration go into one
    submission. On this stack that means an identical artifact name: two
    separately-compiled artifacts cannot share a `hw_context`, measured in
    `05a-phase-b-runlist-spike-result.md`.

    Args:
        steps: ordered list of `DispatchStep`.
        artifact_of: callable step-kernel-name -> artifact identity. Two steps
            whose artifacts compare equal may share a runlist.
        require_single: raise `RunlistSplitError` instead of splitting. Callers
            that are measuring a "one submission per layer" mode pass True so a
            silent split cannot be recorded as an aggregated dispatch.

    Returns:
        list of `Submission`.

    Raises:
        RunlistSplitError: `require_single` and the sequence spans configurations.
    """
    subs = []
    for idx, step in enumerate(steps):
        art = artifact_of(step.kernel)
        if subs and subs[-1].artifact == art:
            subs[-1].steps = subs[-1].steps + (step,)
        else:
            subs.append(Submission(artifact=art, steps=(step,), first_index=idx))

    if require_single and len(subs) > 1:
        spanned = [s.artifact for s in subs]
        raise RunlistSplitError(
            f"a single runlist was required but the sequence spans "
            f"{len(set(spanned))} hardware configurations across {len(subs)} "
            f"submissions: {spanned}. Separately-compiled artifacts cannot share "
            f"an XRT hw_context on this stack — see "
            f"docs/plans/transformer-layer-execution-studies/"
            f"05a-phase-b-runlist-spike-result.md. Building the runlist anyway "
            f"executes without error and returns wrong numbers."
        )
    return subs


def _bind_args(xrt, run, bos, elf_abi, instr_bo=None, instr_len=0):
    """Set a run's buffer arguments under the ABI the artifact was built for.

    ELF: buffers from index 0; the instruction stream lives in the ELF.
    xclbin: opcode at 0, instruction BO at 1, instruction length at 2, buffers
    from 3. The two must never be conflated — an ELF run given the xclbin prefix
    writes buffer arguments three positions too high and reads uninitialised DDR.
    """
    if elf_abi:
        for i, bo in enumerate(bos):
            run.set_arg(i, bo)
        return
    if instr_bo is None:
        raise ValueError("xclbin ABI requires an instruction BO")
    run.set_arg(0, OPCODE_DPU)
    run.set_arg(1, instr_bo)
    run.set_arg(2, instr_len)
    for i, bo in enumerate(bos):
        run.set_arg(i + N_XCLBIN_PREFIX_ARGS, bo)


def _completed(xrt, state):
    return state == xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED


def submit(xrt, context, runs, steps, submission_index):
    """Execute one runlist and wait, timing only the submission itself (rule T1).

    A single-entry submission still goes through `xrt.runlist`, so the timed
    region means the same thing for every mode.

    Returns:
        Elapsed seconds for execute+wait.

    Raises:
        RunlistExecutionError: attributed to the first entry that did not
            complete (rule E1). `pyxrt.runlist` reports one aggregate state, so
            attribution comes from polling each `xrt.run.state()` afterwards.
    """
    runlist = xrt.runlist(context)
    for run in runs:
        runlist.add(run)

    t0 = time.perf_counter()
    try:
        runlist.execute()
        runlist.wait()
        elapsed = time.perf_counter() - t0
    except Exception as exc:  # attribute before re-raising
        elapsed = time.perf_counter() - t0
        idx, state = _first_failed(xrt, runs)
        raise RunlistExecutionError(
            f"runlist submission {submission_index} failed at entry {idx} "
            f"(kernel {steps[idx].kernel!r}, state {state}): {exc}",
            submission_index,
            idx,
            steps[idx],
            state,
        ) from exc

    idx, state = _first_failed(xrt, runs)
    if idx is not None:
        raise RunlistExecutionError(
            f"runlist submission {submission_index} entry {idx} "
            f"(kernel {steps[idx].kernel!r}) did not complete: {state}",
            submission_index,
            idx,
            steps[idx],
            state,
        )
    return elapsed


def _first_failed(xrt, runs):
    """(index, state) of the first entry that is not COMPLETED, else (None, None).

    Falls back to entry 0 when `state()` is unavailable, so a failure is always
    attributable to *some* entry rather than to the sequence as a whole.
    """
    for i, run in enumerate(runs):
        try:
            state = run.state()
        except Exception:
            return 0, None
        if not _completed(xrt, state):
            return i, state
    return None, None


@dataclass
class LaunchCounts:
    """Static per-artifact counts, taken from the MLIR module at compile time.

    The compiled artifact does not carry them, so `KernelCache.compile_and_cache`
    records them into the manifest and `DispatchVector` reads them back. Counting
    them at dispatch time is not possible; counting them by hand per mode is what
    03-measurement-model.md warns against.
    """

    air_launches: int = 0
    herd_launches: int = 0

    @staticmethod
    def from_module(mlir_module):
        text = str(mlir_module)
        return LaunchCounts(
            air_launches=text.count("air.launch"),
            herd_launches=text.count("air.herd "),
        )

    def as_dict(self):
        return {"air_launches": self.air_launches, "herd_launches": self.herd_launches}


def run_sequence(
    cache,
    steps,
    specs,
    backend_kwargs,
    arrays,
    host_writes=None,
    require_single_submission=False,
):
    """Execute a dispatch sequence with pooled BOs, dirty-bit sync and runlists.

    This is the multi-step counterpart to `KernelCache.load_and_run`. It derives
    what that method's `static_input_indices` / `intermediate_indices` /
    `shared_nonstatic` flags say by hand from the declared sequence instead:
    static buffers come from `BufferSpec.static`, intermediates are the buffers a
    step produces, and slot sharing falls out of the liveness analysis.

    Args:
        cache: the owning `KernelCache`.
        steps: ordered `DispatchStep`s. `step.kernel` names a cached artifact.
        specs: dict buffer name -> `BufferSpec`.
        backend_kwargs: per-artifact XRTBackend kwargs, or a dict of them keyed
            by artifact name when the artifacts differ.
        arrays: dict buffer name -> numpy array. Supplies the bytes for host
            writes and the dtype/element count for readback views — never
            `bo.size()`, which is the 4 KiB-rounded slot (rule O3).
        host_writes: buffer names the host writes before the sequence. Defaults
            to every buffer no step produces, plus every static buffer.
        require_single_submission: raise `RunlistSplitError` rather than split.

    Returns:
        (results, vector) where results is a dict of buffer name -> numpy view
        for every declared host output, and vector is a `DispatchVector`.

        The views are zero-copy into pool memory (rule H1): the next sequence
        overwrites them. Copy anything you keep.
    """
    import filelock
    import numpy as np
    import pyxrt as xrt
    from ml_dtypes import bfloat16

    from shared.infra.bo_pool import BoPool, plan_pool

    if not steps:
        return {}, DispatchVector()

    kernels = list(dict.fromkeys(s.kernel for s in steps))
    # `backend_kwargs` is either one preset shared by every artifact, or a dict
    # keyed by artifact name. Per-artifact wins only when it covers every kernel,
    # so a preset that happens to contain a matching key is not misread.
    per_artifact = all(
        k in backend_kwargs and isinstance(backend_kwargs[k], dict) for k in kernels
    )
    kwargs_for = (
        (lambda n: backend_kwargs[n]) if per_artifact else (lambda n: backend_kwargs)
    )

    for name in kernels:
        cache.ensure_loaded(name, kwargs_for(name))

    abis = {cache.artifacts[n].output_binary.endswith(".elf") for n in kernels}
    if len(abis) > 1:
        raise ValueError(
            "a dispatch sequence cannot mix the ELF and xclbin ABIs: their "
            "argument numbering differs (buffers from 0 vs from 3)"
        )
    elf_abi = abis.pop()

    plan = plan_pool(steps, specs, elf_abi)

    # A slot needs a representative (kernel, arg index) to pick its memory group
    # under the xclbin ABI (rule C2). Take it from the first step that names a
    # buffer assigned to that slot.
    slot_position = {}
    for step in steps:
        for i, bufname in enumerate(step.args):
            slot_position.setdefault(plan.slot_of[bufname], (step.kernel, i))

    produced = {n for s in steps for n in s.written_names()}
    if host_writes is None:
        host_writes = {n for n in specs if n not in produced or specs[n].static}

    def _alloc(nbytes, slot):
        backend, _ = cache._loaded[slot_position[slot][0]]
        if elf_abi:
            return xrt.ext.bo(backend.device, nbytes)
        arg_index = slot_position[slot][1]
        return xrt.bo(
            backend.device,
            nbytes,
            xrt.bo.host_only,
            backend.kernel.group_id(arg_index + N_XCLBIN_PREFIX_ARGS),
        )

    vector = DispatchVector(runlist_entries=len(steps))
    counted = {}

    def _to_device(bo):
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def _from_device(bo):
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

    pool = cache.pool_for(plan, _alloc, _to_device, _from_device)
    subs = plan_submissions(
        steps, lambda k: cache.artifacts[k].output_binary, require_single_submission
    )

    with filelock.FileLock("/tmp/npu.lock"):
        t_sync = time.perf_counter()
        for name in sorted(host_writes):
            src_arr = arrays[name]
            src = np.frombuffer(
                src_arr.view(np.int16) if src_arr.dtype == bfloat16 else src_arr,
                dtype=np.uint8,
            )
            bo = pool.bo_for(name, plan)
            np.copyto(
                np.frombuffer(bo.map(), dtype=np.uint8, count=len(src)),
                src,
                casting="no",
            )
            pool.mark_written_by_host(name)
        for name in sorted(set(specs) & (host_writes | produced)):
            if pool.sync_to_device_if_needed(name, plan, specs):
                vector.sync_boundaries += 1
                vector.bytes_transferred += specs[name].nbytes
        vector.sync_ms += (time.perf_counter() - t_sync) * 1000.0

        # Instruction BOs sync once per identity, not per call (rule D6).
        if not elf_abi:
            for name in kernels:
                backend, _ = cache._loaded[name]
                if backend.bo_instr is not None and cache._mark_instr_synced(name):
                    backend.bo_instr.sync(
                        xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
                    )
                    vector.sync_boundaries += 1

        for sub_idx, sub in enumerate(subs):
            backend, _ = cache._loaded[sub.steps[0].kernel]
            runs = []
            for step in sub.steps:
                bos = [pool.bo_for(n, plan) for n in step.args]
                run = xrt.run(backend.kernel)
                _bind_args(
                    xrt,
                    run,
                    bos,
                    elf_abi,
                    instr_bo=backend.bo_instr,
                    instr_len=(
                        len(backend.instr_v) if backend.instr_v is not None else 0
                    ),
                )
                runs.append(run)
                counts = cache.launch_counts.get(step.kernel)
                if counts is not None:
                    vector.air_launches += counts.get("air_launches", 0)
                    vector.herd_launches += counts.get("herd_launches", 0)

            elapsed = submit(xrt, backend.context, runs, sub.steps, sub_idx)
            vector.submission_ms += elapsed * 1000.0
            vector.host_submissions += 1
            counted[sub_idx] = len(sub.steps)
            for step in sub.steps:
                for name in step.written_names():
                    pool.mark_written_by_device(name)

        t_sync = time.perf_counter()
        results = {}
        for name, spec in specs.items():
            if not spec.host_output:
                continue
            pool.sync_from_device(name, plan)
            vector.sync_boundaries += 1
            vector.bytes_transferred += spec.nbytes
            ref = arrays[name]
            results[name] = np.frombuffer(
                pool.bo_for(name, plan).map(), dtype=ref.dtype, count=ref.size
            ).reshape(ref.shape)
        vector.sync_ms += (time.perf_counter() - t_sync) * 1000.0

    vector.per_submission_entries = tuple(counted[i] for i in sorted(counted))
    return results, vector
