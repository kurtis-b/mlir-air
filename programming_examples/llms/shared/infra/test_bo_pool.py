# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only tests for the BO liveness allocator and dirty-bit discipline.

No NPU and no XRT: `BoPool` takes its allocator as a callable, so these run in CI
alongside the compile-only transformer-layer suite. The rule identifiers name the
sections of
`docs/plans/transformer-layer-execution-studies/05b-phase-b-buffer-rules.md`.

Deliberately has **no test-framework dependency**. pytest is not installed in this
repository's sandbox venv — the suites run under lit — so these are plain functions
plus a `__main__` runner. pytest collects them unchanged where it does exist, which
is what convention rule 11 (standard `test_*.py` discovery) asks for.

    python shared/infra/test_bo_pool.py
"""

import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.infra.bo_pool import (  # noqa: E402
    BoPool,
    BufferSpec,
    DispatchStep,
    _bin_size,
    compute_live_ranges,
    content_key,
    host_supplied_names,
    plan_pool,
    plan_signature,
)


@contextmanager
def raises(exc_type, match=None):
    """Minimal stand-in for `pytest.raises`, so the file needs no framework."""
    try:
        yield
    except exc_type as exc:
        if match is not None and match not in str(exc):
            raise AssertionError(
                f"{exc_type.__name__} raised but {match!r} not in {str(exc)!r}"
            ) from exc
        return
    raise AssertionError(f"expected {exc_type.__name__}, nothing was raised")


class FakeBo:
    """Stands in for an `xrt.bo`: the pool only needs map/sync/size."""

    def __init__(self, nbytes, slot):
        self.nbytes = nbytes
        self.slot = slot
        self.to_device = 0
        self.from_device = 0

    def size(self):
        return self.nbytes


def make_pool():
    made = []

    def alloc(nbytes, slot):
        bo = FakeBo(nbytes, slot)
        made.append(bo)
        return bo

    return (
        BoPool(
            alloc,
            lambda bo: setattr(bo, "to_device", bo.to_device + 1),
            lambda bo: setattr(bo, "from_device", bo.from_device + 1),
        ),
        made,
    )


def spec(name, nbytes=4096, **kw):
    return BufferSpec(name=name, nbytes=nbytes, **kw)


# --- live ranges -----------------------------------------------------------


def test_host_input_is_live_from_before_the_sequence():
    """L1: a buffer no step writes starts at -1."""
    steps = [DispatchStep("k", ("x", "y"), writes=(1,))]
    specs = {"x": spec("x"), "y": spec("y")}
    live = compute_live_ranges(steps, specs)
    assert live["x"] == (-1, 0)
    assert live["y"] == (0, 0)


def test_host_output_stays_live_past_the_last_kernel():
    """L3: declaring a host output pins it to the end of the sequence."""
    steps = [
        DispatchStep("k", ("x", "y"), writes=(1,)),
        DispatchStep("k", ("y", "z"), writes=(1,)),
    ]
    specs = {"x": spec("x"), "y": spec("y", host_output=True), "z": spec("z")}
    live = compute_live_ranges(steps, specs)
    assert live["y"] == (0, len(steps))


def test_static_buffers_are_outside_the_liveness_analysis():
    """S3."""
    steps = [DispatchStep("k", ("w", "x"), writes=(1,))]
    specs = {"w": spec("w", static=True, content_key="sha256:aa"), "x": spec("x")}
    assert "w" not in compute_live_ranges(steps, specs)


def _in_place_steps(acc_args, acc_writes):
    """Two chained steps, then one that writes `acc`. `tmp` dies at step 1."""
    return [
        DispatchStep("k", ("in", "tmp"), writes=(1,)),
        DispatchStep("k", ("tmp", "sink"), writes=(1,)),
        DispatchStep("k", acc_args, writes=acc_writes),
    ]


def _slot_mates(plan, name):
    """Every other buffer assigned the same slot as `name`.

    A host-supplied buffer is live from -1, so it overlaps every buffer in the
    sequence and this is empty for one. Non-empty means something else writes
    the slot its host bytes are in before the kernel that reads them runs.
    """
    return sorted(
        n for n, s in plan.slot_of.items() if s == plan.slot_of[name] and n != name
    )


def test_an_in_place_buffer_is_live_from_before_the_sequence():
    """L1/D7: a buffer a step reads and writes is host-supplied, not produced.

    Declared A2's way: one identity at a read position and a write position of
    the same step. Started at the step that writes it instead, its live range
    begins after `in` and `sink` have died, they are handed the slot holding its
    host bytes, and the kernel reads what they left there instead.
    """
    steps = _in_place_steps(("acc", "acc"), (1,))
    specs = {n: spec(n) for n in ("in", "tmp", "sink", "acc")}
    assert host_supplied_names(steps) == {"in", "acc"}
    live = compute_live_ranges(steps, specs)
    assert live["acc"] == (-1, 2)
    assert live["tmp"] == (0, 1)
    assert _slot_mates(plan_pool(steps, specs, elf_abi=True), "acc") == []


def test_a_single_position_in_place_buffer_is_declared_by_the_caller():
    """D7's undecidable case, and the escape hatch for it.

    `DispatchStep("k", ("acc",), writes=(0,))` cannot say whether the kernel
    reads `acc` before writing it, and a plain output is much the commoner
    reading, so the derivation calls it produced. A caller that means in-place
    passes the buffer in `host_supplied` — `run_sequence`'s `host_writes` — and
    the plan then holds its slot from before the sequence.
    """
    steps = _in_place_steps(("acc",), (0,))
    specs = {n: spec(n) for n in ("in", "tmp", "sink", "acc")}
    assert "acc" not in host_supplied_names(steps)
    derived = plan_pool(steps, specs, elf_abi=True)
    assert derived.live["acc"] == (2, 2)
    assert _slot_mates(derived, "acc") == ["in", "sink"]

    declared = plan_pool(steps, specs, elf_abi=True, host_supplied={"in", "acc"})
    assert declared.live["acc"] == (-1, 2)
    assert _slot_mates(declared, "acc") == []
    # O5: the two assignments differ, so they must not share a pool.
    assert declared.signature != derived.signature


# --- overlapping live ranges ----------------------------------------------


def test_overlapping_live_ranges_do_not_share_a_slot():
    """L4/C4: three chained steps, the two intermediates overlap."""
    steps = [
        DispatchStep("k", ("in", "t0"), writes=(1,)),
        DispatchStep("k", ("t0", "t1"), writes=(1,)),
        DispatchStep("k", ("t0", "t1", "out"), writes=(2,)),
    ]
    specs = {n: spec(n) for n in ("in", "t0", "t1", "out")}
    specs["out"] = spec("out", host_output=True)
    plan = plan_pool(steps, specs, elf_abi=True)
    assert plan.slot_of["t0"] != plan.slot_of["t1"]


def test_disjoint_live_ranges_share_one_slot():
    """C4: t0 dies at step 0, so t1 may reuse its slot."""
    steps = [
        DispatchStep("k", ("in", "t0"), writes=(1,)),
        DispatchStep("k", ("t0", "t1"), writes=(1,)),
        DispatchStep("k", ("t1", "out"), writes=(1,)),
    ]
    specs = {n: spec(n) for n in ("in", "t0", "t1", "out")}
    specs["out"] = spec("out", host_output=True)
    plan = plan_pool(steps, specs, elf_abi=True)
    # t0 is live [0,1], t1 is live [1,2] -> they overlap at step 1 and must not share.
    assert plan.slot_of["t0"] != plan.slot_of["t1"]
    # `in` dies at step 0 and does not overlap t1, so it may back t1's successor.
    assert plan.n_slots() < len(specs)


def test_buffers_in_one_step_never_alias():
    """A1: same size, disjoint ranges on paper, but they meet in step 1."""
    steps = [
        DispatchStep("k", ("a", "b"), writes=(1,)),
        DispatchStep("k", ("b", "c"), writes=(1,)),
    ]
    specs = {n: spec(n) for n in ("a", "b", "c")}
    plan = plan_pool(steps, specs, elf_abi=True)
    assert plan.slot_of["b"] != plan.slot_of["c"]


# --- output reuse ----------------------------------------------------------


def test_declared_host_output_keeps_its_slot_to_itself():
    """H2: nothing may take a host output's slot, even after its last kernel use."""
    steps = [
        DispatchStep("k", ("in", "out"), writes=(1,)),
        DispatchStep("k", ("in", "scratch"), writes=(1,)),
    ]
    specs = {
        "in": spec("in"),
        "out": spec("out", host_output=True),
        "scratch": spec("scratch"),
    }
    plan = plan_pool(steps, specs, elf_abi=True)
    assert plan.slot_of["out"] != plan.slot_of["scratch"]


def test_undeclared_output_may_be_recycled():
    """The complement of H2: without host_output, the slot is fair game."""
    steps = [
        DispatchStep("k", ("in", "tmp"), writes=(1,)),
        DispatchStep("k", ("in", "scratch"), writes=(1,)),
    ]
    specs = {n: spec(n) for n in ("in", "tmp", "scratch")}
    plan = plan_pool(steps, specs, elf_abi=True)
    assert plan.slot_of["tmp"] == plan.slot_of["scratch"]


# --- bank / ABI compatibility ---------------------------------------------


def test_xclbin_abi_does_not_pool_across_argument_positions():
    """C2: the group id is a function of (kernel, arg index)."""
    steps = [
        DispatchStep("k1", ("a", "b"), writes=(1,)),
        DispatchStep("k2", ("c", "d"), writes=(1,)),
    ]
    specs = {n: spec(n) for n in ("a", "b", "c", "d")}
    elf = plan_pool(steps, specs, elf_abi=True)
    xcl = plan_pool(steps, specs, elf_abi=False)
    # b dies at step 0; under the ELF ABI d can reuse its slot, under xclbin it cannot
    # because d sits at (k2, 1) and b at (k1, 1).
    assert elf.n_slots() < xcl.n_slots()


def test_different_size_bins_never_share():
    """C3."""
    steps = [
        DispatchStep("k", ("small", "big"), writes=(1,)),
        DispatchStep("k", ("big", "small2"), writes=(1,)),
    ]
    specs = {
        "small": spec("small", 4096),
        "big": spec("big", 65536),
        "small2": spec("small2", 4096),
    }
    plan = plan_pool(steps, specs, elf_abi=True)
    assert plan.slot_of["small"] != plan.slot_of["big"]
    assert plan.bins[plan.slot_of["big"]] == 65536


def test_bin_size_rounds_up_to_a_page():
    assert _bin_size(1) == 4096
    assert _bin_size(4096) == 4096
    assert _bin_size(4097) == 8192


# --- static / content-keyed pool ------------------------------------------


def test_content_key_accepts_a_bfloat16_array():
    """The dtype every static weight in this stack actually has.

    `memoryview` refuses ml_dtypes' bfloat16 ('E' has no buffer-protocol format
    code), so a content key computed the obvious way raises on exactly the arrays
    it exists for. The key is over bytes, so the same bytes viewed as int16 key
    identically.
    """
    import numpy as np
    from ml_dtypes import bfloat16

    arr = np.arange(64, dtype=np.float32).reshape(8, 8).astype(bfloat16)
    key = content_key(arr)
    assert key.startswith("sha256:")
    assert key == content_key(arr.view(np.int16))
    assert key != content_key(np.zeros((8, 8), dtype=bfloat16))


def test_identical_static_content_shares_one_bo():
    """S1."""
    key = content_key(b"\x01" * 4096)
    steps = [
        DispatchStep("k", ("w_q", "x"), writes=(1,)),
        DispatchStep("k", ("w_k", "y"), writes=(1,)),
    ]
    specs = {
        "w_q": spec("w_q", static=True, content_key=key),
        "w_k": spec("w_k", static=True, content_key=key),
        "x": spec("x"),
        "y": spec("y"),
    }
    plan = plan_pool(steps, specs, elf_abi=True)
    assert plan.slot_of["w_q"] == plan.slot_of["w_k"]


def test_identical_static_content_at_different_xclbin_positions_does_not_share():
    """S5: content keying is subordinate to the bank rule (C2).

    Under the xclbin ABI the memory group comes from (kernel, arg index). Two
    identical weights bound at different positions must get different BOs, or
    the allocator banks one BO for whichever position it saw first and the other
    kernel reads its weights through the wrong group — no error, wrong numbers.
    """
    key = content_key(b"\x03" * 4096)
    steps = [
        DispatchStep("k1", ("w_q", "x"), writes=(1,)),
        DispatchStep("k2", ("y", "w_k"), writes=(0,)),
    ]
    specs = {
        "w_q": spec("w_q", static=True, content_key=key),
        "w_k": spec("w_k", static=True, content_key=key),
        "x": spec("x"),
        "y": spec("y"),
    }
    assert plan_pool(steps, specs, elf_abi=True).slot_of["w_q"] == (
        plan_pool(steps, specs, elf_abi=True).slot_of["w_k"]
    )
    xcl = plan_pool(steps, specs, elf_abi=False)
    assert xcl.slot_of["w_q"] != xcl.slot_of["w_k"]


def test_identical_static_content_at_one_xclbin_position_still_shares():
    """S1 survives S5: dedup is only given up where the bank differs."""
    key = content_key(b"\x04" * 4096)
    steps = [
        DispatchStep("k", ("w_q", "x"), writes=(1,)),
        DispatchStep("k", ("w_k", "y"), writes=(1,)),
    ]
    specs = {
        "w_q": spec("w_q", static=True, content_key=key),
        "w_k": spec("w_k", static=True, content_key=key),
        "x": spec("x"),
        "y": spec("y"),
    }
    plan = plan_pool(steps, specs, elf_abi=False)
    assert plan.slot_of["w_q"] == plan.slot_of["w_k"]


def test_each_static_slot_is_synced_even_at_one_content_key():
    """S2 is per BO, not per content key.

    The complement of the test above: once the bank rule has split one content
    key across two slots, both BOs still have to be written. Tracking "already
    synced" by content key would leave the second one holding whatever the
    allocator gave it.
    """
    key = content_key(b"\x05" * 4096)
    steps = [
        DispatchStep("k1", ("w_q", "x"), writes=(1,)),
        DispatchStep("k2", ("y", "w_k"), writes=(0,)),
    ]
    specs = {
        "w_q": spec("w_q", static=True, content_key=key),
        "w_k": spec("w_k", static=True, content_key=key),
        "x": spec("x"),
        "y": spec("y"),
    }
    plan = plan_pool(steps, specs, elf_abi=False)
    pool, _ = make_pool()
    assert pool.sync_to_device_if_needed("w_q", plan, specs) is True
    assert pool.sync_to_device_if_needed("w_k", plan, specs) is True
    assert pool.bo_for("w_q", plan).to_device == 1
    assert pool.bo_for("w_k", plan).to_device == 1


def test_a_slot_records_every_position_it_is_bound_at():
    """C2: the allocator reads `slot_positions` rather than guessing from the
    first use, so it can refuse a slot whose positions disagree on the bank."""
    steps = [
        DispatchStep("k1", ("in", "t"), writes=(1,)),
        DispatchStep("k2", ("t", "out"), writes=(1,)),
    ]
    specs = {n: spec(n) for n in ("in", "t", "out")}
    plan = plan_pool(steps, specs, elf_abi=False)
    assert plan.slot_positions[plan.slot_of["t"]] == (("k1", 1), ("k2", 0))


def test_writing_a_static_buffer_is_rejected():
    """A3."""
    steps = [DispatchStep("k", ("x", "w"), writes=(1,))]
    specs = {"x": spec("x"), "w": spec("w", static=True, content_key="sha256:aa")}
    with raises(ValueError, match="static"):
        plan_pool(steps, specs, elf_abi=True)


def test_missing_spec_is_rejected():
    steps = [DispatchStep("k", ("x", "y"), writes=(1,))]
    with raises(KeyError):
        plan_pool(steps, {"x": spec("x")}, elf_abi=True)


# --- dirty-bit discipline --------------------------------------------------


def test_only_dirty_buffers_sync_to_device():
    """D2: a second sync without an intervening host write is a no-op."""
    steps = [DispatchStep("k", ("x", "y"), writes=(1,))]
    specs = {"x": spec("x"), "y": spec("y")}
    plan = plan_pool(steps, specs, elf_abi=True)
    pool, _ = make_pool()
    pool.mark_written_by_host("x")
    assert pool.sync_to_device_if_needed("x", plan, specs) is True
    assert pool.sync_to_device_if_needed("x", plan, specs) is False
    assert pool.bo_for("x", plan).to_device == 1


def test_static_buffers_sync_exactly_once_per_content():
    """S2: two names, one content key, one sync."""
    key = content_key(b"\x02" * 4096)
    steps = [
        DispatchStep("k", ("w_q", "x"), writes=(1,)),
        DispatchStep("k", ("w_k", "y"), writes=(1,)),
    ]
    specs = {
        "w_q": spec("w_q", static=True, content_key=key),
        "w_k": spec("w_k", static=True, content_key=key),
        "x": spec("x"),
        "y": spec("y"),
    }
    plan = plan_pool(steps, specs, elf_abi=True)
    pool, _ = make_pool()
    assert pool.sync_to_device_if_needed("w_q", plan, specs) is True
    assert pool.sync_to_device_if_needed("w_k", plan, specs) is False
    assert pool.bo_for("w_q", plan).to_device == 1


def test_reassigning_a_slot_marks_the_new_occupant_dirty():
    """D5: the bytes in a recycled slot belong to the previous buffer."""
    steps = [
        DispatchStep("k", ("in", "tmp"), writes=(1,)),
        DispatchStep("k", ("in", "scratch"), writes=(1,)),
    ]
    specs = {n: spec(n) for n in ("in", "tmp", "scratch")}
    plan = plan_pool(steps, specs, elf_abi=True)
    assert plan.slot_of["tmp"] == plan.slot_of["scratch"]

    pool, _ = make_pool()
    pool.bo_for("tmp", plan)
    pool.mark_written_by_device("tmp")
    assert pool.is_dirty("tmp") is False
    pool.bo_for("scratch", plan)  # slot changes hands
    assert pool.is_dirty("scratch") is True


def test_device_write_clears_the_host_dirty_bit():
    """D4: a host->device sync after a kernel wrote the buffer would destroy it."""
    steps = [DispatchStep("k", ("x", "y"), writes=(1,))]
    specs = {"x": spec("x"), "y": spec("y")}
    plan = plan_pool(steps, specs, elf_abi=True)
    pool, _ = make_pool()
    pool.mark_written_by_host("y")
    pool.mark_written_by_device("y")
    assert pool.sync_to_device_if_needed("y", plan, specs) is False


def test_pool_footprint_is_smaller_than_one_bo_per_buffer():
    """The reason the allocator exists at all."""
    steps = [DispatchStep("k", (f"t{i}", f"t{i + 1}"), writes=(1,)) for i in range(8)]
    specs = {f"t{i}": spec(f"t{i}", 1 << 20) for i in range(9)}
    specs["t8"] = spec("t8", 1 << 20, host_output=True)
    plan = plan_pool(steps, specs, elf_abi=True)
    assert plan.footprint_bytes() < 9 * (1 << 20)


# --- pool identity (O5) ----------------------------------------------------


def _sig_steps():
    return [
        DispatchStep("k", ("in", "t0"), writes=(1,)),
        DispatchStep("k", ("t0", "out"), writes=(1,)),
    ]


def _sig_specs():
    s = {n: spec(n) for n in ("in", "t0", "out")}
    s["out"] = spec("out", host_output=True)
    return s


def test_the_same_sequence_has_the_same_plan_signature():
    """O5: this is what a pool is keyed on, so a re-run reuses its BOs.

    `id(plan)` would not: every dispatch builds a fresh plan, so an id key
    allocates a new pool and re-uploads every static weight on each call — and
    CPython recycles the freed plan's id, which can hand that pool to an
    unrelated sequence whose slots are sized and banked for other buffers.
    """
    first = plan_pool(_sig_steps(), _sig_specs(), elf_abi=True).signature
    second = plan_pool(_sig_steps(), _sig_specs(), elf_abi=True).signature
    assert first == second
    assert hash(first) == hash(second)


def test_the_plan_signature_separates_sequences_that_must_not_share_a_pool():
    base = plan_pool(_sig_steps(), _sig_specs(), elf_abi=True).signature

    resized = _sig_specs()
    resized["t0"] = spec("t0", 1 << 20)
    assert plan_pool(_sig_steps(), resized, elf_abi=True).signature != base

    other_abi = plan_pool(_sig_steps(), _sig_specs(), elf_abi=False).signature
    assert other_abi != base

    reordered = list(reversed(_sig_steps()))
    assert plan_pool(reordered, _sig_specs(), elf_abi=True).signature != base

    rekeyed = _sig_specs()
    rekeyed["in"] = spec("in", static=True, content_key="sha256:ff")
    assert plan_pool(_sig_steps(), rekeyed, elf_abi=True).signature != base


def test_a_spec_no_step_names_is_not_part_of_the_plan():
    """`specs` may carry more than the sequence dispatches — a caller reusing one
    spec table across sequences. Those entries get no slot: they have no position,
    hence no memory group. The signature ignores them for the same reason, so two
    such callers still share a pool.
    """
    extra = _sig_specs()
    extra["unused"] = spec("unused", 1 << 20)
    extra["unused_weight"] = spec("unused_weight", static=True, content_key="sha256:ab")
    plan = plan_pool(_sig_steps(), extra, elf_abi=False)
    assert "unused" not in plan.slot_of
    assert "unused_weight" not in plan.slot_of
    assert set(plan.slot_positions) == set(plan.bins)
    assert plan_signature(_sig_steps(), extra, True) == plan_signature(
        _sig_steps(), _sig_specs(), True
    )


def test_declaring_a_static_host_write_does_not_split_the_pool():
    """S3/O5: a static is outside the liveness analysis, so naming one in the
    host-supplied set cannot change the assignment — and must not change the
    signature either, or a caller that lists its weights in `host_writes` keys a
    second pool on the same sequence and re-uploads every one of them.
    """
    key = content_key(b"\x07" * 4096)
    steps = [DispatchStep("k", ("w", "x", "y"), writes=(2,))]
    specs = {
        "w": spec("w", static=True, content_key=key),
        "x": spec("x"),
        "y": spec("y"),
    }
    base = plan_pool(steps, specs, elf_abi=True).signature
    declared = plan_pool(steps, specs, elf_abi=True, host_supplied={"w", "x"})
    assert declared.signature == base


def test_a_reused_pool_does_not_resync_a_resident_static():
    """S2 across dispatches: the second run of a sequence skips the weight upload."""
    key = content_key(b"\x06" * 4096)
    steps = [DispatchStep("k", ("w", "x"), writes=(1,))]
    specs = {"w": spec("w", static=True, content_key=key), "x": spec("x")}
    pool, _ = make_pool()
    for _ in range(3):
        plan = plan_pool(steps, specs, elf_abi=True)  # a fresh plan each dispatch
        pool.sync_to_device_if_needed("w", plan, specs)
        assert pool.is_static_resident("w", plan) is True
    assert pool.bo_for("w", plan_pool(steps, specs, elf_abi=True)).to_device == 1


def test_slot_assignment_is_deterministic():
    """O4: the same sequence must always produce the same plan."""
    steps = [
        DispatchStep("k", ("in", "t0"), writes=(1,)),
        DispatchStep("k", ("t0", "t1"), writes=(1,)),
        DispatchStep("k", ("t1", "out"), writes=(1,)),
    ]
    specs = {n: spec(n) for n in ("in", "t0", "t1", "out")}
    first = plan_pool(steps, specs, elf_abi=True).slot_of
    for _ in range(5):
        assert plan_pool(steps, specs, elf_abi=True).slot_of == first


def _main():
    """Run every test_* in definition order and report. Exit 1 on any failure."""
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
