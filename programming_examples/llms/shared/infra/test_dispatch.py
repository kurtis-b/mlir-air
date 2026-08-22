# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only tests for runlist grouping, the ABI split and the dispatch vector.

The hardware behaviour these encode was measured in
`docs/plans/transformer-layer-execution-studies/01-original-plan-superseded.md`;
this file keeps the code honest about it without needing an NPU.

No test-framework dependency — see `test_bo_pool.py` for why.

    python shared/infra/test_dispatch.py
"""

import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.infra.bo_pool import BufferSpec  # noqa: E402
from shared.infra.cache import KernelCache  # noqa: E402
from shared.infra.dispatch import (  # noqa: E402
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
