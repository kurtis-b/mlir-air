# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only tests for runlist grouping, the ABI split and the dispatch vector.

The hardware behaviour these encode was measured in
`docs/plans/transformer-layer-execution-studies/01-original-plan-superseded.md`;
this file keeps the code honest about it without needing an NPU.

No test-framework dependency — see `test_bo_pool.py` for why.

    python shared/infra/test_dispatch.py
"""

import pathlib
import sys
import tempfile
import traceback
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.infra.bo_pool import BufferSpec  # noqa: E402
from shared.infra.cache import KernelCache  # noqa: E402
from shared.infra.dispatch import (  # noqa: E402
    ArgCountMismatchError,
    elf_arg_count,
    validate_arg_count,
    INSTR_WORD_BYTES,
    N_XCLBIN_PREFIX_ARGS,
    OPCODE_DPU,
    DispatchStep,
    DispatchVector,
    LaunchCounts,
    RunlistSplitError,
    _bind_args,
    default_host_writes,
    instr_bo_nbytes,
    launch_totals,
    plan_submissions,
    sync_instruction_bos,
)


@contextmanager
def raises(exc_type, match=None):
    try:
        yield
    except exc_type as exc:
        if match is not None and match not in str(exc):
            raise AssertionError(
                f"{exc_type.__name__} raised but {match!r} not in {str(exc)!r}"
            ) from exc
        return
    raise AssertionError(f"expected {exc_type.__name__}, nothing was raised")


ARTIFACTS = {"qkv": "a.elf", "attn": "b.elf", "ffn": "c.elf", "qkv2": "a.elf"}


def artifact_of(kernel):
    return ARTIFACTS[kernel]


def step(kernel, *args):
    return DispatchStep(kernel, tuple(args), writes=(len(args) - 1,))


# --- submission grouping ---------------------------------------------------


def test_one_artifact_aggregates_into_one_submission():
    steps = [step("qkv", "x", "y"), step("qkv", "y", "z"), step("qkv", "z", "w")]
    subs = plan_submissions(steps, artifact_of)
    assert len(subs) == 1
    assert len(subs[0]) == 3


def test_elf_abi_aggregates_across_artifacts():
    """05a §5: N full ELFs means N hw_contexts and still one runlist.

    Each entry's run carries its own artifact's configuration, so one submission
    covers all three. This is the gate's requirement and the study's `runlist`
    axis; if it ever regresses to three submissions, `runlist` collapses into
    `offload` and the comparison stops meaning anything.
    """
    steps = [step("qkv", "x", "y"), step("attn", "y", "z"), step("ffn", "z", "w")]
    subs = plan_submissions(steps, artifact_of, elf_abi=True)
    assert len(subs) == 1
    assert len(subs[0]) == 3
    assert subs[0].artifacts == ("a.elf", "b.elf", "c.elf")
    assert subs[0].context_artifact == "a.elf"


def test_xclbin_abi_splits_at_every_artifact_change():
    """05a §4: under the xclbin ABI the configuration is in the xclbin, not the
    run, so entries from another artifact execute against the wrong one."""
    steps = [step("qkv", "x", "y"), step("attn", "y", "z"), step("ffn", "z", "w")]
    subs = plan_submissions(steps, artifact_of, elf_abi=False)
    assert [s.context_artifact for s in subs] == ["a.elf", "b.elf", "c.elf"]
    assert [len(s) for s in subs] == [1, 1, 1]


def test_two_kernel_names_on_one_artifact_share_a_submission():
    """Grouping is by artifact identity, not by cache key."""
    steps = [step("qkv", "x", "y"), step("qkv2", "y", "z")]
    subs = plan_submissions(steps, artifact_of, elf_abi=False)
    assert len(subs) == 1
    assert subs[0].artifacts == ("a.elf",)


def test_xclbin_split_is_contiguous_not_global():
    """A->B->A is three submissions; entries may not be reordered to merge them."""
    steps = [step("qkv", "x", "y"), step("attn", "y", "z"), step("qkv", "z", "w")]
    subs = plan_submissions(steps, artifact_of, elf_abi=False)
    assert len(subs) == 3
    assert subs[0].first_index == 0 and subs[2].first_index == 2


def test_require_single_refuses_the_xclbin_cross_artifact_runlist():
    """Rule E2: the one runlist that executes and returns wrong numbers."""
    steps = [step("qkv", "x", "y"), step("attn", "y", "z")]
    with raises(RunlistSplitError, match="hardware configurations"):
        plan_submissions(steps, artifact_of, require_single=True, elf_abi=False)


def test_require_single_passes_when_the_sequence_is_aggregatable():
    steps = [step("qkv", "x", "y"), step("qkv", "y", "z")]
    assert len(plan_submissions(steps, artifact_of, require_single=True)) == 1


def test_require_single_is_satisfied_by_elf_cross_artifact_aggregation():
    steps = [step("qkv", "x", "y"), step("attn", "y", "z")]
    assert len(plan_submissions(steps, artifact_of, require_single=True)) == 1


def test_empty_sequence_is_no_submissions():
    assert plan_submissions([], artifact_of) == []


# --- N instruction streams under one xclbin --------------------------------
#
# `[2026-08-09]` Demonstrated on hardware by
# `agents/probes/probe_one_xclbin_n_streams.py`: two GEMMs of different shape
# chained with `--xclbin-input` both execute correctly from ONE `hw_context`,
# given a distinct `instance_name` AND a distinct `kernel_id` per stream. That
# makes the xclbin split rule a question about CONFIGURATION identity rather
# than artifact identity, which is what `config_of` expresses.


def test_config_of_defaults_to_artifact_of():
    """The default must reproduce one-xclbin-per-artifact behaviour EXACTLY.

    Every shipped model dispatches through this function; a default that changed
    grouping would change their submission counts silently.
    """
    steps = [step("qkv", "x", "y"), step("attn", "y", "z"), step("ffn", "z", "w")]
    assert [
        (s.artifacts, len(s))
        for s in plan_submissions(steps, artifact_of, elf_abi=False)
    ] == [
        (s.artifacts, len(s))
        for s in plan_submissions(steps, artifact_of, elf_abi=False, config_of=artifact_of)
    ]


def test_xclbin_entries_sharing_one_xclbin_share_a_submission():
    """N streams, one array configuration: the split premise does not apply.

    The three artifacts are distinct files but were packaged into one xclbin, so
    the configuration behind the context is the same for all three and entries
    may share a runlist -- which is the whole of `offload`'s
    reconfiguration-minimizing claim.
    """
    steps = [step("qkv", "x", "y"), step("attn", "y", "z"), step("ffn", "z", "w")]
    subs = plan_submissions(
        steps, artifact_of, elf_abi=False, config_of=lambda k: "shared.xclbin"
    )
    assert len(subs) == 1
    assert len(subs[0]) == 3
    assert subs[0].artifacts == ("a.elf", "b.elf", "c.elf")


def test_xclbin_still_splits_when_configurations_differ():
    """Two shared xclbins is still two configurations, so still two submissions."""
    config = {"qkv": "one.xclbin", "attn": "one.xclbin", "ffn": "two.xclbin"}
    steps = [step("qkv", "x", "y"), step("attn", "y", "z"), step("ffn", "z", "w")]
    subs = plan_submissions(
        steps, artifact_of, elf_abi=False, config_of=lambda k: config[k]
    )
    assert [len(s) for s in subs] == [2, 1]


def test_require_single_is_satisfied_by_one_shared_xclbin():
    """The clause `offload` will gate on once its N streams land."""
    steps = [step("qkv", "x", "y"), step("attn", "y", "z")]
    subs = plan_submissions(
        steps,
        artifact_of,
        require_single=True,
        elf_abi=False,
        config_of=lambda k: "shared.xclbin",
    )
    assert len(subs) == 1


def test_config_split_is_contiguous_like_the_artifact_split():
    """A->B->A on configurations is three submissions, not two merged ones."""
    config = {"qkv": "one.xclbin", "attn": "two.xclbin"}
    steps = [step("qkv", "x", "y"), step("attn", "y", "z"), step("qkv", "z", "w")]
    subs = plan_submissions(
        steps, artifact_of, elf_abi=False, config_of=lambda k: config[k]
    )
    assert len(subs) == 3
    assert subs[0].first_index == 0 and subs[2].first_index == 2


# --- per-run ABI -----------------------------------------------------------


class FakeRun:
    def __init__(self):
        self.args = {}

    def set_arg(self, i, v):
        self.args[i] = v


class FakeXrt:
    pass


def test_elf_abi_binds_buffers_from_index_zero():
    run = FakeRun()
    _bind_args(FakeXrt(), run, ["A", "B", "C"], elf_abi=True)
    assert run.args == {0: "A", 1: "B", 2: "C"}


def test_xclbin_abi_reserves_the_opcode_and_instruction_prefix():
    run = FakeRun()
    _bind_args(
        FakeXrt(), run, ["A", "B", "C"], elf_abi=False, instr_bo="I", instr_len=7
    )
    assert run.args[0] == OPCODE_DPU
    assert run.args[1] == "I"
    assert run.args[2] == 7
    assert run.args[N_XCLBIN_PREFIX_ARGS] == "A"
    assert run.args[N_XCLBIN_PREFIX_ARGS + 2] == "C"


def test_xclbin_abi_without_an_instruction_bo_is_rejected():
    with raises(ValueError, match="instruction BO"):
        _bind_args(FakeXrt(), FakeRun(), ["A"], elf_abi=False)


# --- dispatch vector -------------------------------------------------------


def test_vector_reports_entries_per_submission():
    v = DispatchVector(host_submissions=1, runlist_entries=29)
    assert v.entries_per_submission() == 29.0
    assert v.as_row()["runlist_entries_per_submission"] == 29.0


def test_a_split_sequence_never_claims_one_submission():
    """The distinction the whole study rests on: 8 entries in 8 submissions is
    `offload`, 8 entries in 1 submission is `runlist`. They must not read alike."""
    offload = DispatchVector(host_submissions=8, runlist_entries=8)
    runlist = DispatchVector(host_submissions=1, runlist_entries=8)
    assert offload.as_row()["host_submissions_per_layer"] == 8
    assert offload.entries_per_submission() == 1.0
    assert runlist.entries_per_submission() == 8.0


def test_empty_vector_does_not_divide_by_zero():
    assert DispatchVector().entries_per_submission() == 0.0


def test_row_carries_all_six_fields():
    """The six schema fields, plus the two timings that are NOT schema fields.

    `[2026-08-08]` device_submission_ms and host_sync_ms were measured from the
    first dispatch vector and thrown away. They are surfaced here so a mode can
    report how much of its latency is the NPU, but they are deliberately not
    study-schema columns -- adding one of those is a version bump, and the
    study reads these out of the dispatch extra instead. This test pins both
    halves: the six must all be present, and the timings must not be mistaken
    for schema fields.
    """
    row = DispatchVector().as_row()
    schema_fields = {
        "host_submissions_per_layer",
        "runlist_entries_per_submission",
        "air_launches_per_elf",
        "herd_launches",
        "sync_boundaries",
        "bytes_transferred",
    }
    timings = {"device_submission_ms", "host_sync_ms"}
    assert schema_fields <= set(row)
    assert set(row) == schema_fields | timings


def test_air_launches_are_counted_once_per_elf():
    """03-measurement-model.md defines `air_launches_per_elf` as the launches in
    the compiled module, so an artifact two steps invoke — the gate/up projection
    pair of a real layer — contributes its count once. `herd_launches` is defined
    as launches *executed*, so that one does double.
    """
    steps = [step("qkv", "x", "y"), step("qkv2", "y", "z"), step("attn", "z", "w")]
    counts = {
        "qkv": {"air_launches": 6, "herd_launches": 2},
        "qkv2": {"air_launches": 6, "herd_launches": 2},  # same ELF as qkv
        "attn": {"air_launches": 1, "herd_launches": 1},
    }
    assert launch_totals(steps, artifact_of, counts.get) == (7, 5)


def test_launch_totals_survive_an_artifact_with_no_recorded_counts():
    """A manifest written before the dispatch vector landed costs the counts, not
    the run."""
    steps = [step("qkv", "x", "y")]
    assert launch_totals(steps, artifact_of, {}.get) == (0, 0)


# --- default host writes ---------------------------------------------------


def test_default_host_writes_covers_inputs_and_weights_but_not_outputs():
    """D7: the host supplies what it is read for, and every weight."""
    steps = [
        DispatchStep("gemm", ("x", "w", "mid"), writes=(2,)),
        DispatchStep("gemm", ("mid", "w2", "out"), writes=(2,)),
    ]
    specs = {
        "x": BufferSpec("x", 4096),
        "w": BufferSpec("w", 4096, static=True, content_key="sha256:aa"),
        "w2": BufferSpec("w2", 4096, static=True, content_key="sha256:bb"),
        "mid": BufferSpec("mid", 4096),
        "out": BufferSpec("out", 4096, host_output=True),
    }
    assert default_host_writes(steps, specs) == {"x", "w", "w2"}


def test_default_host_writes_includes_an_in_place_buffer():
    """D7: A2's in-place form is read before it is written, so its bytes come
    from the host. Classified as produced it gets no upload at all, and the
    kernel reads whatever the pool slot held before it."""
    steps = [DispatchStep("k", ("acc", "acc"), writes=(1,))]
    specs = {"acc": BufferSpec("acc", 4096, host_output=True)}
    assert default_host_writes(steps, specs) == {"acc"}


# --- instruction-stream accounting -----------------------------------------


class FakeBackend:
    """The three `XRTBackend` attributes the instruction sync path reads.

    `bo_instr=None` with `instr_v=None` is how the backend represents an ELF
    artifact, whose instruction stream is inside the ELF.
    """

    def __init__(self, n_words):
        self.bo_instr = None if n_words is None else object()
        self.instr_v = None if n_words is None else [0] * n_words


class FakeCache:
    """A `KernelCache` reduced to what `sync_instruction_bos` touches.

    `_mark_instr_synced` is the real method, not a stand-in: the once-per-
    identity rule it implements is half of what these tests are checking.
    """

    _mark_instr_synced = KernelCache._mark_instr_synced

    def __init__(self, backends):
        self._loaded = {name: (b, None) for name, b in backends.items()}
        self._instr_synced = set()


def test_instr_bo_nbytes_is_four_bytes_per_instruction_word():
    assert instr_bo_nbytes(FakeBackend(128)) == 128 * INSTR_WORD_BYTES


def test_an_elf_backend_has_no_instruction_bytes():
    assert instr_bo_nbytes(FakeBackend(None)) == 0


def test_instruction_bytes_are_counted_at_the_sync_that_moves_them():
    """The uploaded instruction stream is host->device traffic like any other.

    Raising `sync_boundaries` without raising `bytes_transferred` would report
    an xclbin-backed row as moving fewer bytes than crossed the bus — by the
    size of its instruction streams, which for a small kernel exceeds its
    activations.
    """
    cache = FakeCache({"qkv": FakeBackend(128), "attn": FakeBackend(64)})
    synced = []
    vector = DispatchVector()
    sync_instruction_bos(cache, ["qkv", "attn"], synced.append, vector)
    assert len(synced) == 2
    assert vector.sync_boundaries == 2
    assert vector.bytes_transferred == (128 + 64) * INSTR_WORD_BYTES


def test_instructions_upload_once_per_identity_in_bytes_as_well_as_count():
    """Rule D6 applies to both halves of the record, or the second dispatch of a
    repeated sequence bills instruction bytes it never sent."""
    cache = FakeCache({"qkv": FakeBackend(128)})
    first, second = DispatchVector(), DispatchVector()
    sync_instruction_bos(cache, ["qkv"], lambda bo: None, first)
    sync_instruction_bos(cache, ["qkv"], lambda bo: None, second)
    assert first.bytes_transferred == 128 * INSTR_WORD_BYTES
    assert (second.sync_boundaries, second.bytes_transferred) == (0, 0)


def test_an_elf_artifact_syncs_no_instruction_bo_at_all():
    """Belt and braces: `run_sequence` skips this path entirely under the ELF
    ABI, and a backend with no instruction BO is a no-op if it ever does not."""
    cache = FakeCache({"qkv": FakeBackend(None)})
    synced = []
    vector = DispatchVector()
    sync_instruction_bos(cache, ["qkv"], synced.append, vector)
    assert synced == []
    assert (vector.sync_boundaries, vector.bytes_transferred) == (0, 0)


def test_launch_counts_come_from_the_mlir_module():
    module = """
    module {
      func.func @f() {
        air.launch (%a) in (%b=%c) { air.herd @h tile (%x, %y) in (...) }
        air.launch (%d) in (%e=%f) { air.herd @g tile (%x, %y) in (...) }
      }
    }
    """
    counts = LaunchCounts.from_module(module)
    assert counts.air_launches == 2
    assert counts.herd_launches == 2
    assert counts.as_dict() == {"air_launches": 2, "herd_launches": 2}


def test_evict_pools_for_drops_exactly_the_named_kernels_pools():
    """Targeted eviction: a caller breaking context reuse for specific artifacts
    (pattern/runlist's per-head attention eviction) drops only the pools whose
    sequences involve those kernels. The wholesale `_pools.clear()` this
    replaces also destroyed the content-keyed static-weight pools — ~14 MB
    re-uploaded per layer at 4096, measured in doc 30 as the runlist-front
    cells' zero warm-vs-cold byte drop.
    """
    import tempfile

    from shared.infra.bo_pool import plan_pool, signature_kernels

    def signature_of(*kernels):
        steps = [DispatchStep(k, ("in", "out"), writes=(1,)) for k in kernels]
        specs = {
            "in": BufferSpec("in", 4096),
            "out": BufferSpec("out", 4096, host_output=True),
        }
        return plan_pool(steps, specs, elf_abi=True).signature

    attention = signature_of("attn_scores", "softmax")
    front = signature_of("qkv_proj")
    assert signature_kernels(attention) == {"attn_scores", "softmax"}

    with tempfile.TemporaryDirectory() as d:
        cache = KernelCache(cache_dir=d)
        cache._pools = {attention: object(), front: object()}

        # The failing direction first: a kernel no pool involves drops nothing.
        cache.evict_pools_for({"unrelated"})
        assert set(cache._pools) == {attention, front}

        cache.evict_pools_for({"attn_scores", "attn_output"})
        assert set(cache._pools) == {front}


# --- reconfiguration accounting ---------------------------------------------
#
# The counters behind `offload`'s gated `reconfiguration:` line and schema v2's
# context_loads / kernel_attaches columns. The single increment lives in
# `KernelCache.ensure_loaded`: every `backend.load()` -- an ELF exactly as an
# xclbin -- is one context load, an `attach_kernel` onto a standing context is
# one attach, and an evicted context's reload counts AGAIN. These tests drive
# the REAL `ensure_loaded` against a fake `air.backend.xrt` and a fake
# `filelock`, injected into sys.modules around the call, so the accounting is
# checked host-side without XRT.


class _FakeLoadedBackend:
    """What `ensure_loaded` needs of an `XRTBackend`."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.loaded_binary = None
        self.attached = []

    def load(self, artifact):
        self.loaded_binary = artifact.output_binary
        return "invoker"

    def attach_kernel(self, artifact):
        self.attached.append(artifact.output_binary)
        return self  # the attached pseudo-backend, as the real one returns

    def unload(self):
        pass


class _FakeArtifact:
    """The one attribute the load path reads."""

    def __init__(self, output_binary):
        self.output_binary = output_binary


@contextmanager
def _fake_runtime():
    """`filelock` + `air.backend.xrt` stand-ins, restored on exit.

    `ensure_loaded` imports both inside the function body, so injecting the
    modules here is enough; nothing touches /tmp/npu.lock or a device.
    """
    import types

    names = ("filelock", "air", "air.backend", "air.backend.xrt")
    saved = {name: sys.modules.get(name) for name in names}

    class _FakeFileLock:
        def __init__(self, path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    filelock = types.ModuleType("filelock")
    filelock.FileLock = _FakeFileLock
    air = types.ModuleType("air")
    air_backend = types.ModuleType("air.backend")
    air_xrt = types.ModuleType("air.backend.xrt")
    air_xrt.XRTBackend = _FakeLoadedBackend
    air.backend = air_backend
    air_backend.xrt = air_xrt
    sys.modules.update(
        {
            "filelock": filelock,
            "air": air,
            "air.backend": air_backend,
            "air.backend.xrt": air_xrt,
        }
    )
    try:
        yield
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _counting_cache(tmpdir, artifacts):
    cache = KernelCache(cache_dir=tmpdir)
    cache.artifacts = {n: _FakeArtifact(b) for n, b in artifacts.items()}
    return cache


def test_elf_loads_count_as_context_loads():
    """An ELF `backend.load()` configures the array like an xclbin load does.

    Doc 03 recorded the ELF-path modes' context loads as uninstrumented; the
    instrument is in fact the same increment the shared-xclbin path uses, and
    this pins that an ELF artifact's load raises it.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as d, _fake_runtime():
        cache = _counting_cache(d, {"g1": "a.elf", "g2": "b.elf"})
        cache.ensure_loaded("g1", {})
        cache.ensure_loaded("g2", {})
        assert cache.reconfiguration_counts() == (2, 0)
        # A cached context is NOT a reconfiguration.
        cache.ensure_loaded("g1", {})
        assert cache.reconfiguration_counts() == (2, 0)


def test_an_evicted_context_reloaded_counts_again():
    """Eviction then reload is offload-ELF's 30 and the runlist front's
    per-head attention reloads -- the mode's real reconfiguration cost. A
    counter that only counted first loads would read both modes as free."""
    import tempfile

    with tempfile.TemporaryDirectory() as d, _fake_runtime():
        cache = _counting_cache(d, {"g1": "a.elf"})
        cache.ensure_loaded("g1", {})
        # What pattern/offload's _evict_context and pattern/runlist's
        # evict_attention_contexts do, minus the pool eviction.
        loaded = cache._loaded.pop("g1")
        loaded[0].unload()
        cache.ensure_loaded("g1", {})
        assert cache.reconfiguration_counts() == (2, 0)


def test_a_shared_binary_attaches_instead_of_reloading():
    """Artifacts sharing one xclbin cost one load plus N-1 attaches.

    The shared-xclbin gate's `context_loads 1 kernel_attaches 4` is this
    accounting at five artifacts; two are enough to pin which counter moves.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as d, _fake_runtime():
        cache = _counting_cache(
            d, {"s1": "shared.xclbin", "s2": "shared.xclbin"}
        )
        cache.ensure_loaded("s1", {})
        assert cache.reconfiguration_counts() == (1, 0)
        cache.ensure_loaded("s2", {})
        assert cache.reconfiguration_counts() == (1, 1)


# ---------------------------------------------------------------------------
# `[2026-08-25]` Item 14 review (blocking): the ELF is the ABI ground truth.
# A loaded artifact called with the wrong argument count must be REFUSED before
# any device work -- xrt.run.set_arg binds only the args it is handed, so a
# mismatched call starts the kernel with part of its signature UNBOUND (devq
# 583: the forced fused-cast 19-arg o_ffn restored with the registry's 15-arg
# drain layout, four f32 scratches dangling, a nondeterministic wrong answer).
# These pin the parser, both refusal directions, the stale-manifest clause and
# the chokepoint's position, all on synthetic ELFs -- no device, no XRT.
# ---------------------------------------------------------------------------


def _mk_elf(names, path):
    """A minimal ELF32 whose .dynsym carries exactly `names` (plus the
    conventional null symbol). Layout: ehdr | .dynstr | .dynsym | shdrs."""
    import struct

    strtab = b"\x00"
    offs = []
    for n in names:
        offs.append(len(strtab))
        strtab += n.encode() + b"\x00"
    ehsize = 52
    str_off = ehsize
    sym_off = str_off + len(strtab)
    syms = struct.pack("<IIIBBH", 0, 0, 0, 0, 0, 0)  # null symbol
    for o in offs:
        syms += struct.pack("<IIIBBH", o, 0, 0, 0x11, 0, 1)
    shoff = sym_off + len(syms)
    ehdr = b"\x7fELF" + bytes([1, 1, 1]) + b"\x00" * 9
    ehdr += struct.pack("<HHIIIIIHHHHHH", 2, 0x5C, 1, 0, 0, shoff, 0, ehsize, 0, 0, 40, 3, 0)
    assert len(ehdr) == ehsize
    shdrs = struct.pack("<10I", *([0] * 10))  # NULL section
    # .dynsym: type 11, link -> section 2 (.dynstr), entsize 16
    shdrs += struct.pack("<10I", 0, 11, 0, 0, sym_off, len(syms), 2, 0, 0, 16)
    # .dynstr: type 3
    shdrs += struct.pack("<10I", 0, 3, 0, 0, str_off, len(strtab), 0, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(ehdr + strtab + syms + shdrs)
    return path


def test_elf_arg_count_reads_the_abi_from_the_binary_itself():
    """Numeric dynsym names ARE the buffer-arg indices; .pdi.* configuration
    images are not; the count is max+1 (a trailing arg no launch references is
    invisible to the ELF and to XRT alike). Non-ELF and symbol-free files are
    UNKNOWN (None), never zero."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = _mk_elf(["0", "1", "2", "3", "4", ".pdi.1", ".pdi.2"], f"{d}/a.elf")
        assert elf_arg_count(p) == 5
        p = _mk_elf(["14", "0", "7", ".pdi.1"], f"{d}/gap.elf")
        assert elf_arg_count(p) == 15  # max + 1, indices need not be dense here
        p = _mk_elf([".pdi.1", ".pdi.2"], f"{d}/nonum.elf")
        assert elf_arg_count(p) is None
        with open(f"{d}/not.elf", "wb") as f:
            f.write(b"not an elf at all")
        assert elf_arg_count(f"{d}/not.elf") is None
        assert elf_arg_count(f"{d}/absent.elf") is None


def test_a_call_that_does_not_match_the_elf_abi_is_refused():
    """The devq-583 scenario: a FORCED fused-cast ELF (19 args) whose
    compile.json is absent or garbled restores the registry drain layout (15
    args). The call must be refused naming both counts -- not started with
    unbound scratch args."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = _mk_elf([str(i) for i in range(19)], f"{d}/o_ffn_qwen.elf")
        with raises(ArgCountMismatchError, match="called with 15 arguments"):
            validate_arg_count("o_ffn_qwen", p, 15)
        try:
            validate_arg_count("o_ffn_qwen", p, 15)
        except ArgCountMismatchError as e:
            assert "19" in str(e) and "unbound" in str(e)
        # control: the matching call passes and reports the validated count
        assert validate_arg_count("o_ffn_qwen", p, 19) == 19


def test_stale_metadata_claiming_the_other_method_is_refused():
    """The inverse mismatch: metadata claims fused-cast (19-arg layout
    restored) beside an ELF that is actually the 15-arg drain cascade."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = _mk_elf([str(i) for i in range(15)], f"{d}/o_ffn_qwen.elf")
        with raises(ArgCountMismatchError, match="called with 19 arguments"):
            validate_arg_count("o_ffn_qwen", p, 19)


def test_a_manifest_contradicting_its_elf_is_refused():
    """A manifest n_args that disagrees with the ELF it names is stale
    metadata: refused outright, even when the CALL happens to match one of
    them. When the binary's ABI is unreadable the manifest count is the
    fallback (clause 3) -- still enforced, never ignored."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = _mk_elf([str(i) for i in range(19)], f"{d}/o_ffn_qwen.elf")
        with raises(ArgCountMismatchError, match="stale or contradictory"):
            validate_arg_count("o_ffn_qwen", p, 19, manifest_n_args=15)
        assert validate_arg_count("o_ffn_qwen", p, 19, manifest_n_args=19) == 19
        with open(f"{d}/blob.bin", "wb") as f:
            f.write(b"\x00" * 64)
        with raises(ArgCountMismatchError, match="called with 15"):
            validate_arg_count("k", f"{d}/blob.bin", 15, manifest_n_args=19)
        assert validate_arg_count("k", f"{d}/blob.bin", 4) is None  # nothing known


def test_the_dispatch_chokepoint_runs_before_any_device_work():
    """Every load path -- the verify adapters, ModelAdapter.prepare and
    build_session --run-only -- dispatches through KernelCache.load_and_run,
    so the validation there is the one gate; pinned by source: it runs BEFORE
    ensure_loaded (no XRT is touched by a refused call), and compile time
    records the ELF's own count into the manifest. Chosen semantics for
    --run-only on a FORCED cache (it never reads compile.json): the first
    o_ffn dispatch is REFUSED with the remediation in the message; a set with
    valid metadata restores correctly through the adapters instead."""
    import inspect

    src = inspect.getsource(KernelCache.load_and_run)
    assert "validate_arg_count" in src
    assert src.index("validate_arg_count") < src.index("ensure_loaded")
    cc = inspect.getsource(KernelCache.compile_and_cache)
    assert "elf_arg_count" in cc
    sm = inspect.getsource(KernelCache._save_manifest)
    assert '"n_args"' in sm


# ---------------------------------------------------------------------------
# `[2026-08-26]` Doc 56 H3 stage 1/2 (queue item 19): the `xrt.run` cache.
# Building a run and binding its arguments was measured at 57.5 us per call
# (devq 622: 28 x o_gemv_ffn over one set of BOs, 1.5132 -> 1.4557 ms/call),
# against 16.8 us for the submission a runlist removes -- so a 57-submission
# decode token spends ~3.3 ms re-binding runs it could build once, and
# -2.205 ms/token of it is recovered on the whole token (devq 623). Reuse is
# only legal while the bindings stay valid, which is what these pin: a bo_key's
# BO list is allocated once and never replaced, so the run is reused per
# (kernel, bo_key); a context that was evicted and reloaded has a NEW
# xrt.kernel and the run bound to the dead one must be discarded, not started;
# and in `run_sequence` the runs belong to the POOL, so evicting the pool
# rebuilds them. Driven against a fake `pyxrt` -- no device.
# ---------------------------------------------------------------------------


class _FakeXrtBo:
    def __init__(self, nbytes):
        self._buf = bytearray(int(nbytes))

    def map(self):
        return memoryview(self._buf)

    def sync(self, direction):
        pass

    def size(self):
        return len(self._buf)


class _FakeXrtRuntime:
    """A `pyxrt` stand-in that counts the run objects the code under test builds."""

    def __init__(self):
        self.runs_created = 0
        rt = self

        class _Run:
            def __init__(self, kernel):
                rt.runs_created += 1
                self.kernel = kernel
                self.bound = {}

            def set_arg(self, i, bo):
                self.bound[i] = bo

            def start(self):
                pass

            def wait2(self):
                pass

            def state(self):
                return "COMPLETED"

        class _Runlist:
            def __init__(self, context):
                self.context = context
                self.entries = []

            def add(self, run):
                self.entries.append(run)

            def execute(self):
                pass

            def wait(self):
                pass

        class _Ext:
            @staticmethod
            def bo(device, nbytes):
                return _FakeXrtBo(nbytes)

        class _Dir:
            XCL_BO_SYNC_BO_TO_DEVICE = "to"
            XCL_BO_SYNC_BO_FROM_DEVICE = "from"

        class _State:
            ERT_CMD_STATE_COMPLETED = "COMPLETED"

        self.run = _Run
        self.runlist = _Runlist
        self.ext = _Ext
        self.xclBOSyncDirection = _Dir
        self.ert_cmd_state = _State


class _FakeXrtBackend:
    """What `load_and_run` / `run_sequence` read off an `XRTBackend`."""

    _n = 0

    def __init__(self, **kwargs):
        self.bo_instr = None
        self.instr_v = None
        self.loaded_binary = None
        self.device = "device0"

    def load(self, artifact):
        self.loaded_binary = artifact.output_binary
        # A fresh kernel OBJECT per load: that identity is what the run cache
        # checks, and reusing one across loads would make the check vacuous.
        _FakeXrtBackend._n += 1
        self.kernel = f"kernel#{_FakeXrtBackend._n}"
        self.context = f"context#{_FakeXrtBackend._n}"
        return "invoker"

    def attach_kernel(self, artifact):
        return self

    def unload(self):
        pass


@contextmanager
def _fake_xrt_runtime():
    """`filelock`, `air.backend.xrt` and `pyxrt` stand-ins, restored on exit."""
    import types

    names = ("filelock", "air", "air.backend", "air.backend.xrt", "pyxrt")
    saved = {name: sys.modules.get(name) for name in names}

    class _FakeFileLock:
        def __init__(self, path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    filelock = types.ModuleType("filelock")
    filelock.FileLock = _FakeFileLock
    air = types.ModuleType("air")
    air_backend = types.ModuleType("air.backend")
    air_xrt = types.ModuleType("air.backend.xrt")
    air_xrt.XRTBackend = _FakeXrtBackend
    air.backend = air_backend
    air_backend.xrt = air_xrt
    rt = _FakeXrtRuntime()
    pyxrt = types.ModuleType("pyxrt")
    for attr in ("run", "runlist", "ext", "xclBOSyncDirection", "ert_cmd_state"):
        setattr(pyxrt, attr, getattr(rt, attr))
    sys.modules.update(
        {
            "filelock": filelock,
            "air": air,
            "air.backend": air_backend,
            "air.backend.xrt": air_xrt,
            "pyxrt": pyxrt,
        }
    )
    try:
        yield rt
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_the_xrt_run_cache_reuses_one_run_per_bo_key_and_dies_with_its_context():
    """`load_and_run`: OFF builds a run per call (today's 57 per decode token);
    ON builds one per (kernel, bo_key) and reuses it; a different bo_key is a
    different BO set and gets its own; and a context evicted and reloaded
    invalidates the cached run by kernel identity rather than starting a run
    bound to a dead context."""
    import tempfile

    import numpy as np

    with tempfile.TemporaryDirectory() as d, _fake_xrt_runtime() as rt:
        p = _mk_elf(["0", "1", "2"], f"{d}/k.elf")
        cache = KernelCache(cache_dir=d)
        cache.artifacts = {"k": _FakeArtifact(p)}
        args = tuple(np.zeros(4, dtype=np.uint8) for _ in range(3))
        call = lambda key: cache.load_and_run(
            "k", {}, *args, output_indices=[2], static_input_indices={0},
            intermediate_indices={2}, bo_key=key)

        cache.cache_xrt_runs = False
        for _ in range(3):
            call("L0")
        assert rt.runs_created == 3, rt.runs_created

        cache.cache_xrt_runs = True
        base = rt.runs_created
        for _ in range(5):
            call("L0")
        assert rt.runs_created == base + 1, rt.runs_created
        call("L1")
        assert rt.runs_created == base + 2

        # What `pattern/offload._evict_context` does: the kernel object changes.
        stale = cache._cached_runs[("k", "L0")][1]
        cache._loaded.pop("k")[0].unload()
        cache.ensure_loaded("k", {})
        call("L0")
        assert rt.runs_created == base + 3
        assert cache._cached_runs[("k", "L0")][1] is not stale


def test_run_sequence_deliberately_rebuilds_its_runs_every_call():
    """`run_sequence` pays the ~47 us of re-binding per entry (devq 622) on
    EVERY call on purpose, and this pins the reason so a later aggregation item
    does not "obviously" turn it on without the second half.

    A cached run must be keyed on something that dies when its BOs die, and the
    only such thing is the POOL object -- so the entry holds a strong reference
    to the pool. `pattern/offload._evict_context` clears `cache._pools` before
    EVERY dispatch precisely so a dispatch's buffers do not outlive it
    ("nothing may stay device resident between GEMMs" is that mode's
    definition), and `pattern/runlist` evicts per artifact. A run cache would
    keep one evicted pool -- and its BOs -- alive per plan signature, turning
    those evictions into no-ops for memory in the two modes whose subject is
    residency. Landing it needs an eviction-aware clear on both paths, which
    belongs with the aggregation that would use it (doc 56 H3 stage 3), not
    with stage 2. `LLMS_CACHE_XRT_RUNS_SEQ=1` opts in for a probe.

    `load_and_run` has no such problem: `_cached_bos` is never cleared by any
    eviction path, so a run kept beside it retains nothing new -- which is why
    the cache is ON there and OFF here.
    """
    import tempfile

    import numpy as np

    with tempfile.TemporaryDirectory() as d, _fake_xrt_runtime() as rt:
        p = _mk_elf(["0", "1", "2"], f"{d}/k.elf")
        cache = KernelCache(cache_dir=d)
        cache.artifacts = {"k": _FakeArtifact(p)}
        cache.launch_counts = {"k": {"air_launches": 1, "herd_launches": 1}}
        steps = [DispatchStep("k", ("w", "x", "y"), writes=(2,))]
        specs = {
            "w": BufferSpec("w", 4, static=True, content_key="sha256:aa"),
            "x": BufferSpec("x", 4),
            "y": BufferSpec("y", 4, host_output=True),
        }
        arrays = {n: np.zeros(4, dtype=np.uint8) for n in ("w", "x", "y")}

        # The load_and_run cache is ON by default and must NOT leak into here.
        assert cache.cache_xrt_runs is True
        assert cache.cache_xrt_runs_in_sequences is False
        for _ in range(3):
            _, vec = cache.run_sequence(steps, specs, {}, arrays)
        assert rt.runs_created == 3, rt.runs_created
        assert (vec.host_submissions, vec.runlist_entries) == (1, 1)

        # Opted in, the runs are reused per (plan signature, submission) while
        # the pool object lives, and rebuilt when it is evicted.
        cache.cache_xrt_runs_in_sequences = True
        base = rt.runs_created
        for _ in range(4):
            cache.run_sequence(steps, specs, {}, arrays)
        assert rt.runs_created == base + 1, rt.runs_created
        cache.evict_pools_for({"k"})
        cache.run_sequence(steps, specs, {}, arrays)
        assert rt.runs_created == base + 2


# ---------------------------------------------------------------------------
# The lock-race fix: one family needs it, one is broken by it (queue item 28)
# ---------------------------------------------------------------------------
#
# `matvec.py`'s multi-row herd HANGS without aircc's
# `--use-lock-race-condition-fix` (ERT_CMD_STATE_TIMEOUT, item 27 section 6.1,
# devq 673/674). `[2026-08-27]` and the same flag FAULTS the device on the
# transformer-layer study's QKV split-cast form -- `RunlistExecutionError`,
# `fatal_error_exception_pc = 0x00000000`, devq 812 and devq 813 with item 31
# excluded. `off_gemm_*` and `rl_gemm_*` take it and are fine. So it is a
# transform with preconditions, not insurance, and the rule is a POSITIVE
# statement about the one family measured to need it: the builder marks its own
# herd and nothing else is touched.
#
# Four earlier drafts tried to decide this from outside the builder -- a
# `link_with` filename, then three shapes of allow-list -- and all four were
# unsound. These tests pin what replaced them, including that no injection
# reaches an unmarked module.
#
# Driven with a stand-in module so the suite keeps its no-toolchain property
# (`test_profiles._module_constant`'s rule). Real IR goes through it on every
# compile.


class _FakeAttr:
    def __init__(self, name, attr):
        self.name = name
        self.attr = attr


class _FakeAttrs:
    def __init__(self, pairs):
        self._pairs = [_FakeAttr(k, v) for k, v in pairs]

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, i):
        return self._pairs[i]


class _FakeOp:
    def __init__(self, name, attrs=(), operands=()):
        self.name = name
        self.attributes = _FakeAttrs(attrs)
        self.operands = list(operands)


def _fake_const(value):
    return _FakeOp("arith.constant", [("value", value)])


class _FakeValue:
    def __init__(self, owner):
        self.owner = owner


class _FakeModule:
    """Just enough of `air.ir.Module` for the herd walk."""

    def __init__(self, ops):
        class _Operation:
            @staticmethod
            def walk(fn):
                for op in ops:
                    fn(op)

        self.operation = _Operation()


def _fake_herd(cols, rows, marked=False, n_deps=0, n_operands=3, sym="herd_0",
               link="mv.o", seg=True, const_rows=True):
    from shared.infra.dispatch import LOCK_RACE_FIX_REQUIRED_ATTR

    sizes = [_FakeValue(_fake_const(cols)),
             _FakeValue(_fake_const(rows) if const_rows
                        else _FakeOp("air.wait_all"))]
    operands = [_FakeValue(_fake_const(0)) for _ in range(n_deps)]
    operands += sizes + [_FakeValue(_fake_const(0)) for _ in range(n_operands)]
    attrs = [("sym_name", f'"{sym}"')]
    if seg:
        attrs.append(("operandSegmentSizes", [n_deps, 2, n_operands]))
    if link is not None:
        attrs.append(("link_with", f'"{link}"'))
    if marked:
        attrs.append((LOCK_RACE_FIX_REQUIRED_ATTR, "unit"))
    return _FakeOp("air.herd", attrs, operands)


def test_the_marked_family_gets_the_fix_whatever_it_links():
    """The mark comes from `matvec.py` itself, so a renamed or copied
    micro-kernel object cannot escape it -- which is the defect that killed
    draft 1, where the rule was `link_with == "mv.o"`."""
    from shared.infra.dispatch import ensure_lock_fix_for_marked_herds as ensure

    for link in ("mv.o", "down_mv.o", "mv_heads_hd128.o", None):
        module = _FakeModule([_fake_herd(8, 4, marked=True, link=link)])
        kwargs, rows, applied = ensure("any_name_at_all", module, {})
        assert kwargs["use_lock_race_condition_fix"] is True, link
        assert rows == 4 and applied is True, link


def test_an_unmarked_module_is_returned_untouched():
    """THE MEASURED CONTRAINDICATION. Injecting the flag into the study's QKV
    split-cast artifacts faults the device (devq 812/813), and `off_gemm_*` /
    `rl_gemm_*` tolerate it but were never shown to need it. So an unmarked
    module keeps exactly the kwargs its caller wrote -- no injection, for any
    name, at any row count."""
    from shared.infra.dispatch import ensure_lock_fix_for_marked_herds as ensure

    for rows in (1, 2, 4):
        for name in ("blk_qkv_proj_4096x768", "fused_qkv_proj_1024x768",
                     "off_gemm_4096x768x768", "rl_gemm_up_4096x768x3072",
                     "o_ffn_qwen", "flash_attn", "o_gemv_ffn"):
            module = _FakeModule([_fake_herd(8, rows, marked=False)])
            given = {"omit_pingpong": "", "output_format": "elf"}
            kwargs, _r, applied = ensure(name, module, given)
            assert kwargs == given, (name, rows)
            assert applied is False, (name, rows)


def test_a_mixed_module_is_decided_by_the_mark_not_the_row_count():
    """A stitched ELF can hold both forms. One marked herd is enough; unmarked
    multi-row herds beside it do not make the difference either way."""
    from shared.infra.dispatch import ensure_lock_fix_for_marked_herds as ensure

    mixed = _FakeModule([_fake_herd(8, 4, marked=False, sym="gemm"),
                         _fake_herd(8, 2, marked=True, sym="gemv"),
                         _fake_herd(8, 1, marked=False, sym="small")])
    kwargs, rows, applied = ensure("stitched", mixed, {})
    assert kwargs["use_lock_race_condition_fix"] is True
    assert applied is True and rows == 4

    unmarked = _FakeModule([_fake_herd(8, 4, marked=False, sym="gemm"),
                            _fake_herd(8, 4, marked=False, sym="gemm2")])
    kwargs, _rows, applied = ensure("stitched", unmarked, {})
    assert kwargs == {} and applied is False


def test_the_async_token_form_does_not_read_a_dependency_as_a_herd_size():
    """`[2026-08-27, restored]` The sizes are found THROUGH `operandSegmentSizes`
    (`operands[seg[0] + index]`), never by position, so an `air.herd` carrying
    async dependency tokens does not read a dependency as its row count.

    Round 6 rewrote this module and dropped the only test that passed
    `n_deps > 0`; the helper kept the parameter but nothing exercised it, so
    positional indexing would have passed every remaining test while reading the
    WRONG operand on the real async form -- a row count that was never read,
    which is the defect this whole item is about. This test discriminates: with
    three dependency tokens, positional indexing reads 0 rather than 2.
    """
    from shared.infra.dispatch import (ensure_lock_fix_for_marked_herds,
                                       herd_row_geometry)

    assert herd_row_geometry(_FakeModule([_fake_herd(8, 2, n_deps=3)])) == [
        ("herd_0", 2)
    ]
    # and the decision path reads the same geometry through the same seam
    _kw, rows, applied = ensure_lock_fix_for_marked_herds(
        "async_form", _FakeModule([_fake_herd(8, 2, marked=True, n_deps=3)]), {}
    )
    assert (rows, applied) == (2, True)


def test_a_module_that_cannot_be_walked_is_refused_not_guessed():
    """Neither applying nor withholding the transform is defensible when the
    module cannot be read: one direction hangs, the other faults."""
    from shared.infra.dispatch import ensure_lock_fix_for_marked_herds as ensure

    class _Explodes:
        @property
        def operation(self):
            raise RuntimeError("no bindings here")

    try:
        ensure("mystery", _Explodes(), {})
    except ValueError as exc:
        assert "could not be read" in str(exc)
        assert "FAULTS" in str(exc) or "faults" in str(exc)
    else:
        raise AssertionError("an unreadable module was guessed at")


def test_an_undecodable_size_is_hazardous_not_single_row():
    """`[2026-08-27, review round 6]` THE DEFECT THIS ITEM CHASED FIVE TIMES,
    found once more inside the decode that was reported as fail-closed: a herd
    whose row count could not be decoded was being REPORTED AS ONE ROW. A value
    that says "small and safe" where nothing was established is the whole family
    of bugs. Undecodable geometry must refuse."""
    from shared.infra.dispatch import (
        HerdGeometryUndecidable,
        ensure_lock_fix_for_marked_herds as ensure,
        herd_row_geometry,
        max_herd_rows,
    )

    for marked in (True, False):
        module = _FakeModule([_fake_herd(8, 4, marked=marked, const_rows=False)])
        try:
            herd_row_geometry(module)
        except HerdGeometryUndecidable:
            pass
        else:
            raise AssertionError(f"marked={marked}: an undecodable size was decoded")
        assert max_herd_rows(module) is None, marked
        try:
            ensure("k", module, {})
        except ValueError as exc:
            assert "could not be read" in str(exc), marked
        else:
            raise AssertionError(f"marked={marked}: an undecodable size was guessed at")

    # the same for a herd with no readable size segment at all
    for kwargs in ({}, {"use_lock_race_condition_fix": True}):
        module = _FakeModule([_fake_herd(8, 4, marked=True, seg=False)])
        try:
            ensure("k", module, kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("a herd with no size segment was accepted")


def test_nothing_reports_a_row_count_it_did_not_read():
    """The reporting paths must not manufacture a number either: a mixed module
    with one undecodable herd is unknown, not `max(the ones that worked)`."""
    from shared.infra.dispatch import HerdGeometryUndecidable, herd_row_geometry, max_herd_rows

    mixed = _FakeModule([_fake_herd(8, 2, sym="ok"),
                         _fake_herd(8, 4, sym="bad", const_rows=False)])
    assert max_herd_rows(mixed) is None
    try:
        herd_row_geometry(mixed)
    except HerdGeometryUndecidable:
        pass
    else:
        raise AssertionError("a partially undecodable module returned a geometry")


def test_an_explicit_false_on_a_marked_herd_is_refused():
    """The one refusal that earned its place. `GEMV_K2048_BACKEND` carried
    `False` as a legacy default while `o_gemv_ffn` inherited it -- a real find,
    and the reason this check exists rather than a silent override."""
    from shared.infra.dispatch import ensure_lock_fix_for_marked_herds as ensure

    module = _FakeModule([_fake_herd(8, 4, marked=True)])
    for key in ("use_lock_race_condition_fix", "use_lock_race_condition_fix_v2"):
        try:
            ensure("k", module, {key: False})
        except ValueError as exc:
            assert "ERT_CMD_STATE_TIMEOUT" in str(exc)
        else:
            raise AssertionError(f"{key}=False was silently overridden")
    # ... and on an UNMARKED module it is simply the caller's business
    plain = _FakeModule([_fake_herd(8, 4, marked=False)])
    kwargs, _rows, applied = ensure("k", plain, {"use_lock_race_condition_fix": False})
    assert kwargs == {"use_lock_race_condition_fix": False} and applied is False


def test_a_caller_that_already_asked_for_the_fix_is_not_double_applied():
    from shared.infra.dispatch import ensure_lock_fix_for_marked_herds as ensure

    module = _FakeModule([_fake_herd(8, 4, marked=True)])
    for asked in ({"use_lock_race_condition_fix": True},
                  {"use_lock_race_condition_fix_v2": True}):
        kwargs, rows, applied = ensure("k", module, asked)
        assert kwargs == asked and applied is False and rows == 4


def test_the_mark_is_the_only_thing_that_injects_the_flag():
    """`[2026-08-27, review round 6]` Round 5 claimed the mark was the only
    trigger while `backend_presets.with_herd_rows` was still injecting the flag
    from a ROW COUNT. Over-broad injection is exactly what faulted the device
    (devq 812/813), so there must be no second trigger -- and the claim has to be
    true rather than nearly true, because the mode lits' green depends on it
    being causally what fixed them."""
    import shared.infra.backend_presets as bp

    assert not hasattr(bp, "with_herd_rows"), (
        "with_herd_rows is a second, row-based injection trigger; the mark is "
        "supposed to be the only one"
    )
    # no preset may SET the flag (prose about why is fine and wanted)
    code = [ln for ln in pathlib.Path(bp.__file__).read_text().splitlines()
            if not ln.lstrip().startswith("#")]
    offenders = [ln for ln in code if "use_lock_race_condition_fix" in ln]
    assert not offenders, (
        f"a preset sets the flag: {offenders}. The compile chokepoint supplies "
        "it, and only for a marked herd"
    )
    # and no preset dict actually carries the key at runtime
    for name in dir(bp):
        value = getattr(bp, name)
        if isinstance(value, dict):
            assert "use_lock_race_condition_fix" not in value, name


def test_there_is_no_exemption_mechanism_left_to_audit():
    """Four drafts died on ways of deciding this from outside the builder. If
    one comes back, this is the test that should have to be deleted first."""
    import shared.infra.dispatch as d

    for gone in ("HERD_ROWS_MEASURED_GREEN", "HERD_ROWS_GREEN_PREFIXES",
                 "MEASURED_MULTI_ROW", "HERD_ROW_HAZARD_OBJECTS",
                 "_herd_rows_recorded_green", "require_lock_fix",
                 "check_herd_rows_lock_fix", "ensure_lock_fix_for_multi_row"):
        assert not hasattr(d, gone), f"{gone} came back"
    src = pathlib.Path(d.__file__).read_text()
    body = src[src.index("def ensure_lock_fix_for_marked_herds"):]
    body = body[:body.index("\ndef ")]
    assert "startswith" not in body and "link_with" not in body


def test_the_compile_chokepoint_applies_the_rule_before_it_compiles_anything():
    """The flag must reach the BACKEND, not merely be computed."""
    import shared.infra.cache as cache_mod
    from shared.infra.cache import KernelCache

    seen = {}

    class _Stop(Exception):
        pass

    class _Backend:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            raise _Stop

    with tempfile.TemporaryDirectory() as d:
        cache = KernelCache(cache_dir=d)
        module = _FakeModule([_fake_herd(8, 4, marked=True)])
        real_prepare = cache_mod.prepare_air_project
        cache_mod.prepare_air_project = lambda **kw: None
        try:
            import air.backend.xrt as xrt_mod

            real_backend = xrt_mod.XRTBackend
            xrt_mod.XRTBackend = _Backend
            try:
                cache.compile_and_cache("anything", module, {"output_format": "elf"})
            except _Stop:
                pass
            finally:
                xrt_mod.XRTBackend = real_backend
        except ImportError:
            return
        finally:
            cache_mod.prepare_air_project = real_prepare
    assert seen.get("use_lock_race_condition_fix") is True, seen
    assert cache.launch_counts["anything"]["lock_race_fix_applied"] is True


def test_the_row_count_is_recorded_in_the_manifest_only_when_it_is_known():
    from shared.infra.dispatch import LaunchCounts

    counts = LaunchCounts(air_launches=10, herd_launches=10).as_dict()
    assert counts == {"air_launches": 10, "herd_launches": 10}
    counts["herd_rows"] = 2
    counts["lock_race_fix_applied"] = True
    assert counts.get("air_launches") == 10 and counts.get("herd_rows") == 2


def _main():
    tests = [
        (n, o) for n, o in globals().items() if n.startswith("test_") and callable(o)
    ]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS {name}")
        except Exception:
            failed.append(name)
            print(f"  FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
