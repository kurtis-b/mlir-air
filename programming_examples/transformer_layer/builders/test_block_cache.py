# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only checks on what the block's ELF cache will and will not reuse.

The block gate caches four ELFs so that its clean run and its fault-injected run
do not compile the layer twice. That saving is only safe if a cached ELF is
reused when it is the binary the current source would produce and NEVER
otherwise -- and the artifact names, which carry the shape and nothing else, do
not answer that question. The tiles and the method come from ``kernel_registry``
and the IR comes from builders this study edits, so four binaries whose names
match the shape can still be four binaries nobody would build today. Running
those while the results artifact records the freshly resolved specs is a passing
gate for an implementation that was never on the device.

So this file pins the rule ``builders/block_cache.py`` states, from both sides:
an unchanged configuration reuses all four, and each thing that can change the
ELF without changing its name -- a registry spec, the emitted MLIR, a device
kernel source, a recorded fingerprint that no longer matches -- forces a
recompile.

    python3 builders/test_block_cache.py

No hardware and no compilation: ``KernelCache`` and the external-object builds
are stubbed, so the only real work is building the four MLIR modules, about
0.1s. It is a ``test_*.py`` so ordinary pytest discovery finds it too (porting
convention 11).

FOOTGUNS
    - The stub below reuses on exactly the two facts the real path reuses on --
      the artifact is in ``cache.artifacts`` and its recorded fingerprint
      matches. It deliberately does NOT model compilation, so a test here can
      never observe a stale ELF directly; what it observes is the decision that
      would run one. Keep it that way: a stub that compiled would need hardware
      and would stop being a unit test of the decision.
    - ``_stub_kernel_objects`` is not an optimization. Un-stubbed, each pass
      runs peano over every device source -- minutes -- and writes the objects
      into whatever directory the test was started from, which for a test run
      from the example root is the source tree.
"""

import io
import json
import os
import sys
import tempfile
from contextlib import contextmanager, redirect_stdout

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE_ROOT = os.path.dirname(_HERE)
if _EXAMPLE_ROOT not in sys.path:
    sys.path.insert(0, _EXAMPLE_ROOT)

import builders.block as block  # noqa: E402
import builders.block_cache as block_cache  # noqa: E402
import opcheck_specs  # noqa: E402
from builders.block import (  # noqa: E402
    block_config,
    compile_block_artifacts,
)
from builders.block_cache import block_artifact_fingerprint  # noqa: E402

#: The operator whose SPECS row is the configuration these checks reason about.
GATE_OPERATOR = "block"


def _gate_shape(operator=GATE_OPERATOR):
    """The gate's own configuration, READ FROM the row the gate runs.

    Transcribing this dict was the defect: it agreed with ``opcheck_specs``
    until that row moved, and then it agreed with history instead -- and the
    checks below would have gone on reporting a caching decision for a shape
    nobody runs. ``opcheck_specs`` is imported rather than parsed because this
    module already builds the block's MLIR, so it already needs everything
    that import pulls in (unlike ``study/test_run_ladder.py``, which does not
    and derives the same claim by AST for that reason).
    """
    for row in opcheck_specs.SPECS:
        if row.get("operator") == operator:
            return dict(row["shape"])
    raise AssertionError(
        f"no {operator!r} row in opcheck_specs.SPECS -- the gate this module "
        f"checks the caching decision for no longer exists, so these checks "
        f"describe nothing. Known operators: "
        f"{sorted({r.get('operator') for r in opcheck_specs.SPECS})}"
    )


SHAPE = _gate_shape()


class FakeCache:
    """The four things ``compile_block_artifacts`` asks a ``KernelCache`` for.

    ``artifacts`` is populated only by ``load_manifest``, exactly as the real
    one is under ``run_only``: an entry exists because a previous process
    compiled it and recorded it, never because this process did.
    """

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.artifacts = {}
        self.manifest = None
        self.compiled = []
        self.objects = []

    def load_manifest(self):
        if self.manifest is None:
            return False
        self.artifacts = dict(self.manifest)
        return True

    def _save_manifest(self):
        self.manifest = dict(self.artifacts)

    def compile_and_cache(self, name, mlir_module, backend_kwargs):
        self.compiled.append(name)
        self.artifacts[name] = f"{name}.elf"


class FakeModule:
    """Stands in for an ``air.ir.Module`` where only its text matters."""

    def __init__(self, text):
        self.text = text

    def __str__(self):
        return self.text


@contextmanager
def _stub_kernel_objects(record):
    """Replace every artifact's external-object build with a recorder.

    See the module footgun: those calls are real peano invocations writing real
    objects into the working directory. What is under test is which artifacts
    the reuse rule decides to build, and the recorder answers that exactly.
    """
    real = block._ARTIFACT_BUILD
    block._ARTIFACT_BUILD = {
        key: (lambda cfg, key=key: record.append(key), build_module)
        for key, (_, build_module) in real.items()
    }
    try:
        yield
    finally:
        block._ARTIFACT_BUILD = real


def _run(cache, cfg, run_only=True):
    """One ``compile_block_artifacts`` pass, returning what it compiled.

    The builders narrate their progress on stdout; swallow it so a failing
    assertion is the only thing in the output.
    """
    cache.compiled = []
    cache.objects = []
    with _stub_kernel_objects(cache.objects), redirect_stdout(io.StringIO()):
        compile_block_artifacts(cache, cfg, run_only=run_only)
    # The external objects are built for exactly the artifacts that are compiled
    # and no others: a reused ELF already has its objects linked in, so building
    # one for it is a peano compile spent on nothing.
    #
    # Until Phase E1 it was worse than wasteful -- object names were minted from
    # the GEMM method alone, so building an object for a reused artifact could
    # OVERWRITE the one a later artifact in the same pass was relying on. That
    # hazard is gone (names are per (tile_m, tile_n) now) and this assertion
    # stays, because the wasted-compile reason is reason enough and because it is
    # what pins the pairing in _ARTIFACT_BUILD.
    assert sorted(cfg["artifacts"][key] for key in cache.objects) == sorted(
        cache.compiled
    )
    return sorted(cache.compiled)


def _fingerprint_file(cache):
    return os.path.join(cache.cache_dir, block_cache.FINGERPRINT_FILE)


def _recorded(cache):
    with open(_fingerprint_file(cache)) as handle:
        return json.load(handle)


def _fresh(tmp):
    """A cold cache and the resolved configuration for ``SHAPE``."""
    return FakeCache(tmp), block_config(**SHAPE)


def test_the_shape_under_test_is_the_gate_s_own_specs_row():
    """``SHAPE`` must BE the gate's row, not a copy that agreed with it once.

    Every check in this module reasons about the caching decision taken for the
    configuration the gate runs. A hand-written copy of that configuration
    holds until ``opcheck_specs`` moves and then describes a shape nobody
    runs -- while every check here goes on passing, because they are all
    self-consistent about the wrong number. This is the assertion that turns
    that drift red.
    """
    rows = [r for r in opcheck_specs.SPECS if r.get("operator") == GATE_OPERATOR]
    assert len(rows) == 1, (
        f"expected exactly one {GATE_OPERATOR!r} SPECS row, found {len(rows)}"
    )
    assert SHAPE == rows[0]["shape"], (
        f"the shape these checks run is {SHAPE}, but the {GATE_OPERATOR!r} gate "
        f"row is {rows[0]['shape']}. They must be the same object's value -- if "
        "SHAPE has been re-transcribed, derive it from the row again."
    )
    # Not vacuous: the row must actually carry the fields block_config needs.
    assert set(SHAPE) >= {"seq_len", "emb_dim", "ffn_dim", "num_heads", "head_dim"}
    assert all(isinstance(v, int) and v > 0 for v in SHAPE.values())


def test_a_cold_cache_compiles_every_artifact():
    with tempfile.TemporaryDirectory() as tmp:
        cache, cfg = _fresh(tmp)
        assert _run(cache, cfg) == sorted(cfg["artifacts"].values())
        assert sorted(_recorded(cache)) == sorted(cfg["artifacts"].values())


def test_an_unchanged_configuration_reuses_every_artifact():
    with tempfile.TemporaryDirectory() as tmp:
        cache, cfg = _fresh(tmp)
        _run(cache, cfg)
        assert _run(cache, cfg) == []


def test_a_manifest_without_fingerprints_is_not_trusted():
    """The pre-fingerprint cache layout, and the reason this file exists.

    A manifest holding all four names used to be the whole test. It is now a
    miss, because a manifest records what was built and not what it was built
    from.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cache, cfg = _fresh(tmp)
        _run(cache, cfg)
        os.remove(_fingerprint_file(cache))
        assert _run(cache, cfg) == sorted(cfg["artifacts"].values())


def test_a_stale_fingerprint_recompiles_only_its_own_artifact():
    """Per-artifact granularity: the four ELFs are independently linked."""
    with tempfile.TemporaryDirectory() as tmp:
        cache, cfg = _fresh(tmp)
        _run(cache, cfg)
        stale = cfg["artifacts"]["ffn"]
        recorded = _recorded(cache)
        recorded[stale] = "0" * 64
        with open(_fingerprint_file(cache), "w") as handle:
            json.dump(recorded, handle)
        assert _run(cache, cfg) == [stale]
        # And the freshly recorded fingerprint puts it back in the reuse set,
        # rather than leaving it permanently dirty.
        assert _run(cache, cfg) == []


def test_a_registry_spec_change_at_the_same_shape_is_a_miss():
    """The finding this file was written for.

    A re-swept registry row moves the tiles without moving the shape, so every
    artifact name is unchanged. ``tile_k_l1`` in particular reaches the ELF
    through ``compile_gemm_mm``'s ``-D`` flags rather than through the IR, which
    is why the resolved configuration is fingerprinted alongside the MLIR.

    All four rebuild, not just the FFN: the whole configuration goes into every
    artifact's digest. That is over-invalidation on purpose -- deciding which
    artifacts a given config field can reach is exactly the reasoning that gets
    a stale ELF accepted -- and it costs a compile rather than a wrong result.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cache, cfg = _fresh(tmp)
        _run(cache, cfg)
        cfg["ffn_up_spec"] = dict(cfg["ffn_up_spec"], tile_k_l1=64)
        assert cfg["artifacts"] == block_config(**SHAPE)["artifacts"]
        assert _run(cache, cfg) == sorted(cfg["artifacts"].values())


def test_a_device_kernel_edit_is_a_miss():
    """A ``.cc`` change moves the object the ELF already linked.

    Nothing else in the fingerprint sees it: the MLIR names its external
    objects, it does not carry them.
    """
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as srcs:
        cache, cfg = _fresh(tmp)
        kernel = os.path.join(srcs, "fake_kernel.cc")
        with open(kernel, "w") as handle:
            handle.write("void k() {}\n")
        real_dirs = block_cache._KERNEL_SOURCE_DIRS
        block_cache._KERNEL_SOURCE_DIRS = real_dirs + (srcs,)
        try:
            _run(cache, cfg)
            assert _run(cache, cfg) == []
            with open(kernel, "w") as handle:
                handle.write("void k() { /* one more instruction */ }\n")
            assert _run(cache, cfg) == sorted(cfg["artifacts"].values())
        finally:
            block_cache._KERNEL_SOURCE_DIRS = real_dirs


def test_the_emitted_mlir_is_part_of_the_fingerprint():
    """A builder edit that reaches the IR, without the shape moving."""
    cfg = block_config(**SHAPE)
    before = block_artifact_fingerprint(cfg, "ffn", FakeModule("module { }"))
    after = block_artifact_fingerprint(cfg, "ffn", FakeModule("module { // edit\n }"))
    assert before != after


def test_the_four_artifacts_do_not_share_a_fingerprint():
    """Otherwise one artifact's match would vouch for another's binary."""
    cfg = block_config(**SHAPE)
    module = FakeModule("module { }")
    digests = {
        key: block_artifact_fingerprint(cfg, key, module) for key in cfg["artifacts"]
    }
    assert len(set(digests.values())) == len(digests)


def test_another_shape_keeps_its_recorded_fingerprints():
    """The file is merged, not rewritten.

    A bring-up shape beside the gate point shares the cache directory; writing
    only the current configuration's four names would cost it a needless
    four-ELF recompile on its next run.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cache, cfg = _fresh(tmp)
        _run(cache, cfg)
        recorded = _recorded(cache)
        recorded["blk_qkv_proj_512x768"] = "f" * 64
        with open(_fingerprint_file(cache), "w") as handle:
            json.dump(recorded, handle)
        _run(cache, cfg)
        assert _recorded(cache)["blk_qkv_proj_512x768"] == "f" * 64


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"block cache tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
