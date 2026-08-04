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

- **Runs may only share a runlist when each one carries its own configuration.**
  Under the ELF ABI they do: every artifact gets its own `hw_context` from its own
  full ELF, and a runlist built on any one of those contexts executes runs drawn
  from all of them, correctly. Under the xclbin ABI they do not — the array
  configuration comes from the xclbin behind the context, not from the run — so
  entries there may only share a runlist when they share an artifact.
  `plan_submissions` splits on exactly that distinction. See
  `05a-phase-b-runlist-spike-result.md` for both measurements.

- **Never build a cross-artifact runlist under the xclbin ABI "to see what
  happens".** It executes, raises nothing, times out on nothing, and returns wrong
  numbers for every entry except the one that supplied the context (05a §4).
  `RunlistSplitError` exists so that failure mode is unreachable rather than
  merely discouraged.

- **Do not try to put two full ELFs into one `hw_context`.** That is what
  05-phase-b originally proposed and XRT rejects it three different ways (05a §§1-3).
  Aggregation does not need it: one context per ELF, one runlist across them.

- **The two ABIs number their arguments differently.** Under the ELF ABI buffers
  start at index 0 and the instruction stream is inside the ELF. Under the xclbin
  ABI index 0 is the opcode, 1 the instruction BO, 2 the instruction length, and
  buffers start at 3. `_bind_args` is the only place that knows this; do not
  open-code either layout elsewhere.

- **A split sequence reports its real submission count.** `DispatchVector` never
  claims one submission for work that took several — that would collapse `runlist`
  into `offload` in the results, which is exactly the distinction the study exists
  to measure.

- **An in-place buffer at one argument position must be declared.**
  `default_host_writes` uploads every buffer a step reads before any step writes
  it, which covers a plain input and the two-position in-place form rule A2 asks
  for. It cannot cover a buffer that appears only at a written position: nothing
  in `DispatchStep` distinguishes a read-modify-write there from a plain output.
  Name such a buffer in `host_writes` and its host bytes are uploaded and its
  slot pinned from before the sequence.
"""

import time
from dataclasses import dataclass, field

from shared.infra.bo_pool import DispatchStep, host_supplied_names

__all__ = [
    "DispatchStep",  # re-exported for callers
    "DispatchVector",
    "LaunchCounts",
    "RunlistSplitError",
    "RunlistExecutionError",
    "Submission",
    "default_host_writes",
    "instr_bo_nbytes",
    "launch_totals",
    "plan_submissions",
    "sync_instruction_bos",
    "OPCODE_DPU",
    "N_XCLBIN_PREFIX_ARGS",
    "INSTR_WORD_BYTES",
]

#: `ert_start_npu` opcode used by the xclbin ABI's generic MLIR_AIE kernel.
OPCODE_DPU = 3

#: opcode, instruction BO, instruction length — the xclbin ABI's fixed prefix.
N_XCLBIN_PREFIX_ARGS = 3

#: `XRTBackend` holds the instruction stream as `uint32` words.
INSTR_WORD_BYTES = 4


class RunlistSplitError(RuntimeError):
    """A single runlist was required but the sequence spans hardware configurations.

    Only reachable under the xclbin ABI, where the configuration comes from the
    xclbin behind the context rather than from the run. Raised at build time,
    never worked around: see the module docstring and
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
        touched, counted once per distinct ELF — 03-measurement-model.md defines
        the field as the launches *in the compiled module*, so an artifact two
        steps invoke contributes its count once, not twice. Recorded at compile
        time from the MLIR module, because the runtime artifact does not carry
        it.
    herd_launches: `air.herd` operations *executed*, which 03 defines as a count
        of launches rather than a property of a module, so this one does
        accumulate per step: two invocations of a two-herd artifact are four.
        The asymmetry with `air_launches` is 03's; `launch_totals` is the one
        place either is computed.
    sync_boundaries: `bo.sync()` calls issued for the sequence, both directions.
        This is the number the dirty-bit discipline reduces, and the number that
        makes `offload` and `fused_elf` differ even at equal submission counts.
    bytes_transferred: bytes moved by those syncs, instruction streams included.
        Under the xclbin ABI an artifact's instructions are a BO the host
        uploads, so they cross the boundary exactly as a weight's bytes do and
        are counted the same way; under the ELF ABI they are inside the ELF and
        cross nothing. Every sync that raises `sync_boundaries` adds its bytes
        here, so the two fields always describe the same set of transfers.
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
    """One runlist: a contiguous run of steps that may share a submission.

    `artifacts` holds the distinct artifacts its entries span, in first-use
    order. Under the ELF ABI that is often more than one — each entry's run comes
    from its own artifact's `hw_context` and carries its own configuration. Under
    the xclbin ABI it is always exactly one.
    """

    artifacts: tuple = ()
    steps: tuple = ()
    first_index: int = 0

    def __len__(self):
        return len(self.steps)

    @property
    def context_artifact(self):
        """The artifact whose `hw_context` the runlist object is built on.

        Any of the spanned contexts works — measured over every ordering and
        every choice of context in 05a §5 — so the first entry's is used, which
        makes the choice deterministic and the timing reproducible.
        """
        return self.artifacts[0]


def plan_submissions(steps, artifact_of, require_single=False, elf_abi=True):
    """Group a dispatch sequence into the submissions the hardware allows.

    Under the **ELF ABI** every step is aggregatable: each artifact is loaded
    into its own `hw_context` from its own full ELF, and one `xrt.runlist` built
    on any one of those contexts executes runs drawn from all of them. The whole
    sequence becomes one submission. 05a §5 measures that this is bit-identical
    to sequential dispatch in every ordering and measurably faster.

    Under the **xclbin ABI** it is not: the array configuration comes from the
    xclbin behind the context, so entries from another artifact execute against
    the wrong configuration and return wrong numbers with no error raised
    (05a §4). The sequence is therefore split at every artifact change.

    Args:
        steps: ordered list of `DispatchStep`.
        artifact_of: callable step-kernel-name -> artifact identity. Under the
            xclbin ABI two steps may share a runlist only if these compare equal.
        require_single: raise `RunlistSplitError` instead of splitting. Callers
            that are measuring a "one submission per layer" mode pass True so a
            silent split cannot be recorded as an aggregated dispatch.
        elf_abi: True when the artifacts are ELFs. Defaults True because that is
            the path the study's artifacts take; pass it explicitly rather than
            relying on the default when the ABI is not statically known.

    Returns:
        list of `Submission`.

    Raises:
        RunlistSplitError: `require_single` and the sequence spans configurations.
    """
    subs = []
    for idx, step in enumerate(steps):
        art = artifact_of(step.kernel)
        mergeable = subs and (elf_abi or subs[-1].artifacts == (art,))
        if mergeable:
            sub = subs[-1]
            sub.steps = sub.steps + (step,)
            if art not in sub.artifacts:
                sub.artifacts = sub.artifacts + (art,)
        else:
            subs.append(Submission(artifacts=(art,), steps=(step,), first_index=idx))

    if require_single and len(subs) > 1:
        spanned = [s.context_artifact for s in subs]
        raise RunlistSplitError(
            f"a single runlist was required but the sequence spans "
            f"{len(set(spanned))} hardware configurations across {len(subs)} "
            f"submissions: {spanned}. Under the xclbin ABI the array "
            f"configuration comes from the xclbin behind the hw_context, not "
            f"from the run, so entries from another artifact execute against "
            f"the wrong configuration — see docs/plans/"
            f"transformer-layer-execution-studies/"
            f"05a-phase-b-runlist-spike-result.md §4. Building the runlist "
            f"anyway executes without error and returns wrong numbers. Compile "
            f"these artifacts to ELF to aggregate them."
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


def instr_bo_nbytes(backend):
    """Bytes an artifact's instruction BO moves when it is synced to device.

    `XRTBackend` reads the instruction stream as a `uint32` array and sizes the
    BO at `len(instr_v) * 4` (`python/air/backend/xrt.py:647`), so that product —
    not `bo.size()`, which is the rounded allocation — is what actually crosses
    the boundary. Returns 0 under the ELF ABI, where there is no instruction BO
    because the stream is inside the ELF.
    """
    if backend.bo_instr is None or backend.instr_v is None:
        return 0
    return len(backend.instr_v) * INSTR_WORD_BYTES


def sync_instruction_bos(cache, kernels, sync, vector):
    """Upload each artifact's instruction stream once per identity (rule D6).

    xclbin ABI only; the caller skips this entirely under the ELF ABI. A
    sequence that invokes one artifact eight times uploads its instructions
    once, which is what `KernelCache._mark_instr_synced` is for.

    Both halves of the record are updated together. Counting the boundary but
    not the bytes under-reports every xclbin-backed row by the size of its
    instruction streams — for a small kernel that is larger than its
    activations, so the omission is not a rounding error in the study's transfer
    numbers.

    Args:
        cache: the owning `KernelCache`, for `_mark_instr_synced` and `_loaded`.
        kernels: kernel names the sequence dispatches, in first-use order.
        sync: callable bo -> None performing the host->device sync.
        vector: `DispatchVector` the boundaries and bytes are recorded on.
    """
    for name in kernels:
        backend, _ = cache._loaded[name]
        if backend.bo_instr is None or not cache._mark_instr_synced(name):
            continue
        sync(backend.bo_instr)
        vector.sync_boundaries += 1
        vector.bytes_transferred += instr_bo_nbytes(backend)


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
            complete (rule E1).

    Note: `runlist.wait()` succeeding is the only success signal available.
    A run submitted through a runlist is owned by it, and `xrt.run.state()` on
    such a run still reports `ERT_CMD_STATE_NEW` after a *successful* wait — the
    runlist does not write the per-run state back. So the per-run states are read
    only on the failure path, to say which entry to blame, and never as a
    post-success check. Treating `NEW` as a failure fails every healthy runlist.
    """
    runlist = xrt.runlist(context)
    for run in runs:
        runlist.add(run)

    t0 = time.perf_counter()
    try:
        runlist.execute()
        runlist.wait()
        return time.perf_counter() - t0
    except Exception as exc:  # attribute before re-raising
        idx, state = _first_failed(xrt, runs)
        raise RunlistExecutionError(
            f"runlist submission {submission_index} failed at entry {idx} "
            f"(kernel {steps[idx].kernel!r}, state {state}): {exc}",
            submission_index,
            idx,
            steps[idx],
            state,
        ) from exc


def _first_failed(xrt, runs):
    """(index, state) of the first entry to blame for a failed submission.

    Only meaningful after `runlist.wait()` has raised. `ERT_CMD_STATE_NEW` means
    the entry never started, which after a failure points at the first entry that
    did not run — usually the one after the entry that actually faulted, so both
    `NEW` and any other non-`COMPLETED` state count. Falls back to entry 0 when
    `state()` is unavailable, so a failure is always attributable to *some* entry
    rather than to the sequence as a whole.
    """
    for i, run in enumerate(runs):
        try:
            state = run.state()
        except Exception:
            return 0, None
        if not _completed(xrt, state):
            return i, state
    return 0, None


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


def launch_totals(steps, binary_of, counts_of):
    """(air_launches, herd_launches) for a sequence, per 03-measurement-model.md.

    The two are counted differently and that is deliberate.
    `air_launches_per_elf` is defined as the `air.launch` operations *in the
    compiled module*, so every distinct ELF the sequence touches contributes its
    count once however many entries invoke it — a real layer dispatches its gate
    and up projections through one artifact, and counting per entry would report
    a module with N launches as having 2N. `herd_launches` is defined as launches
    executed, so it accumulates per step.

    Both live here rather than in the dispatch loop so that the asymmetry is
    visible in one place and a later edit cannot quietly make them agree.

    Args:
        steps: ordered `DispatchStep`s.
        binary_of: kernel name -> compiled artifact path. Two cache entries may
            resolve to one ELF; that is one compiled module.
        counts_of: kernel name -> `LaunchCounts.as_dict()`, or None for an
            artifact loaded from a manifest written before the counts existed.
    """
    air = 0
    herd = 0
    seen = set()
    for step in steps:
        counts = counts_of(step.kernel) or {}
        herd += counts.get("herd_launches", 0)
        binary = binary_of(step.kernel)
        if binary not in seen:
            seen.add(binary)
            air += counts.get("air_launches", 0)
    return air, herd


def default_host_writes(steps, specs):
    """The buffers `run_sequence` writes from the host when not told which (D7).

    Every buffer some step reads before any step writes it — `host_supplied_names`
    decides that, and its docstring records the one case a `DispatchStep` cannot
    express — plus every static weight. A3 makes a static read-only, so it already
    qualifies; naming statics anyway keeps a missed weight upload, which is silent
    garbage rather than an error, from resting on a rule enforced elsewhere.

    A buffer whose only appearance is at a written position is **not** in here.
    That is a plain output in the overwhelming majority of cases, and a caller
    doing a single-position in-place update declares it by naming the buffer in
    `host_writes`.
    """
    return host_supplied_names(steps) | {
        n for s in steps for n in s.args if specs[n].static
    }


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
    step produces that no step reads first (D7), and slot sharing falls out of
    the liveness analysis.

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
            to `default_host_writes`: every buffer read before it is written,
            plus every static buffer. Declaring one by hand is how a caller
            expresses a single-position in-place buffer, which the step
            declaration cannot; a name no step dispatches is an error rather
            than a silent no-op, since it has no BO to be written into.
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

    # Only buffers the sequence actually dispatches. `specs` may carry more —
    # a caller reusing one spec table across several sequences — and a buffer no
    # step names has no slot in the plan (bo_pool: no position, hence no bank).
    dispatched = {n for s in steps for n in s.args}

    # Materialized before `plan_pool`, which may be handed it: `host_writes` is
    # any iterable, and a generator consumed there would read as empty here.
    if host_writes is not None:
        host_writes = set(host_writes)

    # The host-supplied set decides each buffer's live-range start (D7/L1), so
    # the plan has to know it: an in-place buffer treated as produced starts at
    # the step that writes it, and an earlier-dying buffer takes the slot its
    # host bytes are sitting in.
    plan = plan_pool(steps, specs, elf_abi, host_supplied=host_writes)

    # After `plan_pool`, which is what reports a step naming a buffer with no
    # spec — deriving the default first would turn that into a bare KeyError.
    if host_writes is None:
        host_writes = default_host_writes(steps, specs)
    else:
        undispatched = sorted(host_writes - dispatched)
        if undispatched:
            raise ValueError(
                f"host_writes names {undispatched}, which no step dispatches, so "
                f"they have no pool slot to be written into. Declare them on a "
                f"step or drop them from host_writes."
            )

    # One device for the whole pool (rule O2). Each XRTBackend.load() builds its
    # own `xrt.device(0)` wrapper, so without this a buffer shared between two
    # artifacts would be allocated against whichever wrapper happened to be seen
    # first. They all refer to one physical device, but pinning the choice is
    # what makes the pool's ownership statement true rather than incidental.
    pool_device = cache._loaded[steps[0].kernel][0].device

    def _alloc(nbytes, slot):
        if elf_abi:
            return xrt.ext.bo(pool_device, nbytes)
        # Under the xclbin ABI the memory group is a function of (kernel, arg
        # index) (rule C2), and `plan.slot_positions` holds every position this
        # slot is bound at. They normally agree; when the sequence forces one BO
        # across positions that disagree, refuse rather than bank it for the
        # first one and hand the rest a buffer in the wrong group.
        groups = {}
        for kernel_name, arg_index in plan.slot_positions[slot]:
            backend, _ = cache._loaded[kernel_name]
            groups[(kernel_name, arg_index)] = backend.kernel.group_id(
                arg_index + N_XCLBIN_PREFIX_ARGS
            )
        if len(set(groups.values())) > 1:
            raise ValueError(
                f"slot {slot} is bound at positions in different memory groups "
                f"({groups}); one BO cannot satisfy both. Give the buffer a "
                f"distinct identity per position, or compile to ELF where "
                f"`xrt.ext.bo` carries no group id (rule C2)."
            )
        return xrt.bo(
            pool_device, nbytes, xrt.bo.host_only, next(iter(groups.values()))
        )

    vector = DispatchVector(runlist_entries=len(steps))
    vector.air_launches, vector.herd_launches = launch_totals(
        steps,
        lambda k: cache.artifacts[k].output_binary,
        cache.launch_counts.get,
    )
    counted = {}

    def _to_device(bo):
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def _from_device(bo):
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

    # Before `pool_for`, so a sequence refused by `require_single_submission`
    # does not leave an empty pool behind under its signature.
    subs = plan_submissions(
        steps,
        lambda k: cache.artifacts[k].output_binary,
        require_single_submission,
        elf_abi=elf_abi,
    )
    pool = cache.pool_for(plan, _alloc, _to_device, _from_device)

    with filelock.FileLock("/tmp/npu.lock"):
        t_sync = time.perf_counter()
        # Host writes are the only host->device traffic the pool can have: a
        # buffer a step produces has no host bytes to send, and uploading its
        # slot before its producer runs would push the previous occupant's bytes
        # at a kernel that is about to overwrite them anyway. D5's dirty bit on
        # such a buffer is cleared by the device write (D4), not paid for with a
        # sync.
        for name in sorted(host_writes):
            spec = specs[name]
            # A static weight is written to its BO once and never again (S2).
            # Pools outlive a sequence, so on the second and later dispatches
            # this skips the copy as well as the sync — which is most of what
            # makes a repeated sequence cheap.
            if spec.static and pool.is_static_resident(name, plan):
                continue
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
            if pool.sync_to_device_if_needed(name, plan, specs):
                vector.sync_boundaries += 1
                vector.bytes_transferred += spec.nbytes
        vector.sync_ms += (time.perf_counter() - t_sync) * 1000.0

        # Instruction BOs sync once per identity, not per call (rule D6). Timed
        # into `sync_ms` like every other host->device sync: it is one, and the
        # first dispatch of a sequence is where its cost lands.
        if not elf_abi:
            t_sync = time.perf_counter()
            sync_instruction_bos(cache, kernels, _to_device, vector)
            vector.sync_ms += (time.perf_counter() - t_sync) * 1000.0

        for sub_idx, sub in enumerate(subs):
            runs = []
            for step in sub.steps:
                # Per step, not per submission: a submission may span artifacts
                # under the ELF ABI, and each entry has to be a run of *its own*
                # artifact's kernel. Taking the first step's kernel for all of
                # them would execute the wrong program with the right buffers.
                backend, _ = cache._loaded[step.kernel]
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

            ctx_backend, _ = cache._loaded[sub.steps[0].kernel]
            elapsed = submit(xrt, ctx_backend.context, runs, sub.steps, sub_idx)
            vector.submission_ms += elapsed * 1000.0
            vector.host_submissions += 1
            counted[sub_idx] = len(sub.steps)
            for step in sub.steps:
                for name in step.written_names():
                    pool.mark_written_by_device(name)

        t_sync = time.perf_counter()
        results = {}
        for name in sorted(dispatched):
            spec = specs[name]
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
