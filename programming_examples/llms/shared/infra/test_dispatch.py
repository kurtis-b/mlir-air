# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only tests for runlist grouping, the ABI split and the dispatch vector.

The hardware behaviour these encode was measured in
`docs/plans/transformer-layer-execution-studies/05a-phase-b-runlist-spike-result.md`;
this file keeps the code honest about it without needing an NPU.

No test-framework dependency — see `test_bo_pool.py` for why.

    python shared/infra/test_dispatch.py
"""

import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.infra.dispatch import (  # noqa: E402
    N_XCLBIN_PREFIX_ARGS,
    OPCODE_DPU,
    DispatchStep,
    DispatchVector,
    LaunchCounts,
    RunlistSplitError,
    _bind_args,
    plan_submissions,
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
    row = DispatchVector().as_row()
    assert set(row) == {
        "host_submissions_per_layer",
        "runlist_entries_per_submission",
        "air_launches_per_elf",
        "herd_launches",
        "sync_boundaries",
        "bytes_transferred",
    }


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
