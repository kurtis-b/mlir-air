# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The numerical check for every transformer-layer operator, and its negative control.

One entry point for all of Phase C. An operator is "validated" when this module
runs it on real NPU2 against an FP32 reference and the run passes; nothing else
in the phase counts as evidence.

CONTRACT (the driver calls this directly, not through the Makefile)

    python3 opcheck.py --list
        JSON array on stdout: [{"operator": ..., "shape_key": ...}, ...].
        Every (operator, shape) this sub-phase claims. Touches no hardware.

    python3 opcheck.py --operator <op> [--shape-key <k>]
        Run the check on hardware. Exit 0 if and only if it passed. Writes
        results/<operator>__<shape_key>.json.

    python3 opcheck.py --operator <op> [--shape-key <k>] --fault-inject input
        The negative control. MUST exit non-zero. Writes into results/fault/
        instead, so an injected run can never overwrite the verdict of a clean
        one.

    Omitting --shape-key runs every shape the operator claims, and the exit
    status is the conjunction.

    --expect-failure inverts the status of a --fault-inject run for callers that
    want the control's own verdict rather than the run's: exit 0 if and only if
    the comparison ran and rejected the perturbed run. It is additive; the three
    invocations above behave identically with or without it.

WHY --expect-failure LIVES HERE AND NOT IN THE CALLER
    "The process exited non-zero" is not evidence that the fault was detected. A
    missing PEANO_INSTALL_DIR, a kernel that fails to compile or link, and an NPU
    that never comes up all exit non-zero without ever comparing anything, so a
    caller that simply inverted the exit status would report a passing negative
    control for every one of them and let the operator gate advance on a check
    that was never exercised. The inversion belongs where the evidence is: this
    module holds the completed comparison's statistics, and only reports the
    control satisfied when those statistics show the comparison itself doing the
    rejecting.

WHY THE COMPARISON IS NOT WRITTEN HERE
    ``XRTRunner.run_test`` already is the gate this phase requires: ``np.isclose``
    over the FULL output, ``max_mismatch_percentage`` defaulting to zero, bf16
    upcast before comparing. ``_RecordingRunner`` below subclasses it only to
    copy out the error statistics on the way past -- the verdict is still
    ``XRTRunner._check_outputs``'s, delegated to with ``super()``. A second
    comparison written here would be a second thing to get wrong, and the
    registry's numbers would stop being comparable with every other kernel's.

THE ONE OPERATOR THAT OWNS ITS OWN DISPATCH
    ``run_test`` takes ONE ``air.ir.Module`` and runs it as one artifact, which
    is every operator in Phase C. The Phase D2 ``block`` is not: it is four ELFs
    dispatched as four ``KernelCache.run_sequence`` calls, because ``addnorm``
    caps at 104 rows at width 768 (three L3->L1 streams against two shim MM2S
    channels; see ``builders/addnorm.py``) and the layer's two normalization
    points therefore have to be row-blocked into 64 dispatches each. There is no
    single module to hand ``run_test``.

    So a spec may return a ``dispatch`` callable instead of a ``module``:

        dispatch(inputs, stage_stats) -> (actual_outputs, record_extra)

    Everything that makes this a check rather than a self-report is UNCHANGED by
    that seam. The reference is still computed in the catalogue from the clean
    inputs before any injection; the injection still perturbs the list handed to
    ``dispatch``, after the reference exists; the verdict is still
    ``_RecordingRunner._check_outputs`` at the same ``RTOL`` and the spec's
    ``atol`` with zero permitted mismatches; the results artifact is still
    written by ``_write_result``. What the seam replaces is only WHO CALLS THE
    HARDWARE. ``stage_stats`` is handed in rather than imported so a
    multi-artifact operator's per-boundary comparison is the same code as its
    verdict -- passing it as an argument is also what keeps the catalogue from
    importing this module and closing a cycle.

WHY THE NEGATIVE CONTROL EXISTS
    Every value in a results file is produced by code in this repository, so a
    driver that trusted ``passed`` would be trusting the thing under test. Fault
    injection is the one check that cannot be satisfied by making the test
    laxer: a reference compared against itself, a tolerance wide enough to
    swallow anything, and an ignored ``--fault-inject`` flag all still PASS
    under injection. The injection perturbs the array handed to the DEVICE,
    after the reference has been computed from the clean one -- perturbing the
    reference instead would satisfy the letter of the check and destroy its
    purpose.

WHERE THE OPERATORS THEMSELVES LIVE
    Two modules, not one, and this file is a third. Porting convention 5 caps a
    module at ~800 lines and the split has happened twice as the port grew:

        opcheck.py          what counts as evidence -- this file. The recording
                            runner, the injection, the results artifact, the
                            negative-control verdict, the CLI. Knows nothing
                            about what an operator is.
        opcheck_prepare.py  HOW each operator is built and fed. One
                            ``prepare_<operator>`` each. Every per-operator
                            footgun -- which element a fault may be injected
                            into, which external objects a shape builds, which
                            backend settings each needs -- is documented there,
                            next to the code it applies to.
        opcheck_specs.py    WHICH ``(operator, shape)`` the port claims, and at
                            what ``atol``. ``SPECS``, and the measurement behind
                            every tolerance in it.

    Adding a shape touches only the catalogue. Adding an operator touches the
    catalogue and the preparers. Changing what counts as evidence touches only
    this file. ``SPECS`` is still imported from one place, below.

FOOTGUNS
    - Results always land next to THIS file, wherever the process was started
      from. The external kernel objects the catalogue builds do not: they go to
      the current working directory, because that is where aiecc's
      ``link_with`` search looks. Run from a scratch directory.
    - This dispatches to the NPU. Serialize it on ``/tmp/mlir-air-npu.lock``
      -- a DIFFERENT inode from the ``/tmp/npu.lock`` ``XRTRunner`` takes
      internally, since both are BSD ``flock(2)`` and one inode self-deadlocks.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_PROJ_ROOT = _HERE.parent  # programming_examples/
for _p in (str(_PROJ_ROOT), str(_PROJ_ROOT / "llms"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from air.backend.xrt_runner import XRTRunner  # noqa: E402

from opcheck_specs import SPECS  # noqa: E402

# Held fixed across every kernel in the registry; `atol` is what moves, sized to
# the kernel's measured worst-case absolute error. See kernel_registry/README.md.
RTOL = 1.6e-2

RESULTS_DIR = _HERE / "results"
FAULT_RESULTS_DIR = RESULTS_DIR / "fault"

# Added to one element of one device input under --fault-inject. Two orders of
# magnitude above the tolerance band at these input scales, so a device that
# reads the perturbed buffer cannot round back into agreement.
FAULT_DELTA = 2.0


class _RecordingRunner(XRTRunner):
    """``XRTRunner`` that copies out the check's statistics as it passes.

    The verdict is unchanged: ``_check_outputs`` records and then returns
    ``super()._check_outputs(...)``. The recorded ``n_mismatch`` re-applies
    ``np.isclose`` with the same ``rtol``/``atol``, so it agrees with the
    verdict by construction rather than by assertion.

    ``atol_required`` is the smallest ``atol`` this run would have passed at:
    ``max(|a - e| - rtol * |e|)`` over every element, floored at zero. It is
    what makes an ``atol`` choice checkable. ``abs_err_max`` alone is not --
    it counts error on a large-magnitude element that ``rtol`` already covers,
    so an operator whose outputs span a wide dynamic range (causal attention,
    where early rows attend to a handful of keys) looks far closer to its
    tolerance than it is. The margin worth quoting is ``atol / atol_required``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats = None

    def _check_outputs(
        self,
        actual_outputs,
        expected_outputs,
        rtol=1e-3,
        atol=1e-8,
        max_mismatch_percentage=0,
        min_correlation=None,
    ):
        n_elements = 0
        n_mismatch = 0
        abs_err_sum = 0.0
        ref_abs_sum = 0.0
        rel_err_max = 0.0
        abs_err_max = 0.0
        atol_required = 0.0
        for actual, expected in zip(actual_outputs, expected_outputs):
            a = np.reshape(actual, expected.shape).astype(np.float64)
            e = np.asarray(expected).astype(np.float64)
            abs_err = np.abs(a - e)
            n_elements += e.size
            n_mismatch += int(np.count_nonzero(~np.isclose(a, e, rtol=rtol, atol=atol)))
            abs_err_sum += float(abs_err.sum())
            ref_abs_sum += float(np.abs(e).sum())
            rel_err_max = max(rel_err_max, float((abs_err / (np.abs(e) + 1e-30)).max()))
            abs_err_max = max(abs_err_max, float(abs_err.max()))
            atol_required = max(
                atol_required, float((abs_err - rtol * np.abs(e)).max())
            )
        self.stats = {
            "mean_rel_L1": abs_err_sum / (ref_abs_sum + 1e-30),
            "rel_err_max": rel_err_max,
            "abs_err_max": abs_err_max,
            "atol_required": max(atol_required, 0.0),
            "n_elements": n_elements,
            "n_mismatch": n_mismatch,
        }
        return super()._check_outputs(
            actual_outputs=actual_outputs,
            expected_outputs=expected_outputs,
            rtol=rtol,
            atol=atol,
            max_mismatch_percentage=max_mismatch_percentage,
            min_correlation=min_correlation,
        )


def stage_stats(actual, expected, rtol=RTOL, atol=0.0):
    """The verdict's own comparison, applied to one intermediate boundary.

    Handed to a spec's ``dispatch`` callable so that a multi-artifact operator's
    per-boundary evidence is produced by the SAME code as its verdict --
    ``_RecordingRunner._check_outputs``, hence ``np.isclose`` at the same
    ``rtol``, hence an ``n_mismatch`` that agrees with ``passed`` by construction
    rather than by assertion. A second comparison written for the intermediates
    could disagree with the one that decides the run, and the whole point of
    capturing them is to localize a disagreement.

    Returns the statistics dict ``_RecordingRunner`` records, plus ``passed``.

    A boundary whose reference is deliberately f32 (an operator's INTERIOR
    staging, which the reference does not round) is upcast rather than
    downcast: ``np.isclose`` on a bf16 actual against an f32 expected leaves the
    common type to numpy, and rounding the reference down to meet the device
    would hide exactly the staging error the check exists to measure.
    """
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    if expected.dtype == np.float32 and actual.dtype != np.float32:
        actual = actual.astype(np.float32)
    runner = _RecordingRunner()
    passed = bool(
        runner._check_outputs(
            actual_outputs=[actual],
            expected_outputs=[expected],
            rtol=rtol,
            atol=atol,
            max_mismatch_percentage=0,
        )
    )
    return dict(runner.stats, passed=passed)


def _inject(inputs, where, delta=FAULT_DELTA):
    """Perturb one element of one DEVICE input, in place on a copy.

    Called only after the reference has been computed from the clean inputs, so
    the device is now being asked to reproduce a value it was not given.
    """
    idx, position = where
    perturbed = list(inputs)
    buf = np.array(perturbed[idx], copy=True)
    before = float(buf[position])
    buf[position] = np.asarray(before + delta, dtype=buf.dtype)
    print(
        f"[fault-inject] input {idx} at {position}: {before} -> {float(buf[position])}"
    )
    perturbed[idx] = buf
    return perturbed


def _write_result(spec, stats, passed, fault_inject, extra=None):
    directory = FAULT_RESULTS_DIR if fault_inject else RESULTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{spec['operator']}__{spec['shape_key']}.json"
    record = {
        "operator": spec["operator"],
        "shape_key": spec["shape_key"],
        "shape": spec["shape"],
        "rtol": RTOL,
        "atol": spec["atol"],
        # The reference is computed in float32 from bf16-rounded inputs, in
        # every builder module. Not iron's bf16 reference, which agrees with a
        # bf16 device partly by being wrong in the same direction.
        "ref_dtype": "float32",
        "mean_rel_L1": stats["mean_rel_L1"],
        "rel_err_max": stats["rel_err_max"],
        "abs_err_max": stats["abs_err_max"],
        # The smallest atol this run would have passed at. `atol / atol_required`
        # is the real margin; see _RecordingRunner on why abs_err_max is not.
        "atol_required": stats["atol_required"],
        "n_elements": stats["n_elements"],
        "n_mismatch": stats["n_mismatch"],
        "passed": passed,
        "fault_injected": fault_inject,
    }
    # Operator-specific provenance (e.g. which GEMM method and tiles were
    # resolved, and whether they came from the registry). Merged after the
    # fixed fields so an operator cannot overwrite the verdict it is reporting.
    for key, value in (extra or {}).items():
        record.setdefault(key, value)
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"[opcheck] wrote {path}")
    return record


def run_spec(spec, fault_inject=None, verbose=False):
    """Run one (operator, shape) on hardware. Returns the results record written.

    Returning at all means the comparison ran: ``_write_result`` reads
    ``runner.stats``, which nothing but ``_RecordingRunner._check_outputs`` sets,
    so a run that dies before comparing raises rather than returning a verdict.
    That is what makes the record safe to reason about in
    ``_negative_control_verdict``.
    """
    label = f"{spec['operator']} [{spec['shape_key']}]"
    print(f"[opcheck] {label}: preparing")
    prepared = spec["prepare"](spec["shape"])

    inputs = prepared["inputs"]
    if fault_inject == "input":
        inputs = _inject(inputs, prepared["inject"])
    elif fault_inject is not None:
        raise ValueError(f"unknown fault-inject mode {fault_inject!r}")

    runner_kwargs = {
        "verbose": verbose,
        "omit_while_true_loop": False,
        "output_format": "xclbin",
        "instance_name": spec["operator"],
        "report_precision": True,
    }
    # Per-operator backend settings, e.g. the BD-ID recycling and the ELF output
    # format a multi-segment design needs. Nothing here can reach the
    # comparison: run_test takes the tolerances separately and _check_outputs is
    # _RecordingRunner's.
    runner_kwargs.update(prepared.get("runner_kwargs", {}))
    runner = _RecordingRunner(**runner_kwargs)
    extra = dict(prepared.get("record_extra") or {})

    dispatch = prepared.get("dispatch")
    if dispatch is not None:
        # A multi-artifact operator: it calls the hardware itself and hands back
        # the outputs to compare. See the module docstring on what this seam
        # does and does not change.
        actual_outputs, dispatch_extra = dispatch(inputs, stage_stats)
        extra.update(dispatch_extra or {})
        passed = bool(
            runner._check_outputs(
                actual_outputs=actual_outputs,
                expected_outputs=prepared["expected"],
                rtol=RTOL,
                atol=spec["atol"],
                max_mismatch_percentage=0,
            )
        )
        print("PASS!" if passed else "failed.")
        # A multi-boundary operator's verdict is the CONJUNCTION of its
        # end-to-end comparison and every boundary comparison it recorded. This
        # can only make the verdict stricter: `stages_passed` is absent for
        # every operator that records no stages, and an operator that records
        # them has already had each one checked by `stage_stats`, at the same
        # rtol and with zero permitted mismatches. Without it a layer that ends
        # in a LayerNorm could report `passed` while carrying a stage the
        # comparison rejected -- which is the exact failure the stages exist to
        # surface.
        if not extra.get("stages_passed", True):
            print("failed: a per-boundary comparison rejected the run.")
            passed = False
    else:
        return_code = runner.run_test(
            prepared["module"],
            inputs=inputs,
            expected_outputs=prepared["expected"],
            rtol=RTOL,
            atol=spec["atol"],
        )
        passed = return_code == 0

    # A spec may bound the run's AGGREGATE error on top of the element-wise
    # verdict: `mean_rel_L1_max`, conjoined here so it can only make the
    # verdict stricter. The element-wise check cannot enforce an aggregate
    # claim -- every element can sit inside rtol/atol while the mean relative
    # L1 error exceeds the figure the operator exists to beat (norm_tail's
    # resident-pipeline claim is exactly that shape of claim), so a spec that
    # states a ceiling has it checked by the same run that produced the
    # statistic, not by a reader of the results file.
    mean_rel_l1_max = spec.get("mean_rel_L1_max")
    if mean_rel_l1_max is not None:
        extra["mean_rel_L1_max"] = mean_rel_l1_max
        if runner.stats["mean_rel_L1"] > mean_rel_l1_max:
            print(
                f"failed: mean_rel_L1 {runner.stats['mean_rel_L1']:.3e} exceeds "
                f"the spec's ceiling {mean_rel_l1_max:.3e} -- the aggregate "
                "claim fails even where every element is inside rtol/atol"
            )
            passed = False

    record = _write_result(spec, runner.stats, passed, fault_inject, extra=extra)

    verdict = "PASS" if passed else "FAIL"
    print(f"[opcheck] {label}: {verdict}")
    if fault_inject:
        # Under injection FAIL is the desired outcome, and a PASS means the
        # check is not actually reading the device's inputs.
        control = "as required" if not passed else "NEGATIVE CONTROL DID NOT FAIL"
        print(
            f"[opcheck] {label}: fault-inject {fault_inject} -> {verdict} ({control})"
        )
    return record


def _negative_control_verdict(records, fault_inject):
    """Exit status for --expect-failure. 0 only on evidence the fault was caught.

    Every record here came from a completed comparison (see ``run_spec``), so
    reaching this function already rules out the setup failures that an
    exit-status inversion would misread as a passing control. What remains to
    check is that the comparison is what rejected the run: it must have been
    given the injected inputs, it must have returned FAIL, and it must have
    counted mismatching elements to do so. A FAIL with ``n_mismatch == 0`` would
    mean the verdict came from somewhere other than the tolerance check, which
    proves nothing about whether the check discriminates.
    """
    problems = []
    for record in records:
        label = f"{record['operator']} [{record['shape_key']}]"
        if record["fault_injected"] != fault_inject:
            problems.append(
                f"{label}: recorded fault_injected={record['fault_injected']!r}, "
                f"not {fault_inject!r} -- the injection did not reach the run"
            )
        if record["passed"]:
            problems.append(
                f"{label}: PASSED under injection, so the check is not reading "
                f"the buffer the device was actually given"
            )
        elif record["n_mismatch"] <= 0:
            problems.append(
                f"{label}: FAILED with n_mismatch={record['n_mismatch']}, so the "
                f"tolerance check is not what rejected it"
            )

    if problems:
        print("NEGATIVE CONTROL: FAIL")
        for problem in problems:
            print(f"  {problem}")
        return 1

    print("NEGATIVE CONTROL: PASS (injected run failed the comparison, as required)")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--list",
        action="store_true",
        help="print every claimed (operator, shape_key) as JSON; no hardware",
    )
    parser.add_argument("--operator", help="operator to check")
    parser.add_argument(
        "--shape-key", help="one shape of that operator; omit to run them all"
    )
    parser.add_argument(
        "--fault-inject",
        choices=["input"],
        default=None,
        help="negative control: perturb one device input after the reference "
        "is computed. The run MUST then fail.",
    )
    parser.add_argument(
        "--expect-failure",
        action="store_true",
        help="report the negative control's verdict instead of the run's: exit "
        "0 only if the comparison ran and rejected the injected run. Requires "
        "--fault-inject.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.list:
        print(
            json.dumps(
                [
                    {"operator": s["operator"], "shape_key": s["shape_key"]}
                    for s in SPECS
                ]
            )
        )
        return 0

    if not args.operator:
        parser.error("one of --list or --operator is required")

    if args.expect_failure and not args.fault_inject:
        parser.error("--expect-failure is only meaningful with --fault-inject")

    selected = [s for s in SPECS if s["operator"] == args.operator]
    if args.shape_key:
        selected = [s for s in selected if s["shape_key"] == args.shape_key]
    if not selected:
        parser.error(
            f"no such (operator, shape_key): {args.operator} / {args.shape_key}"
        )

    records = [
        run_spec(spec, fault_inject=args.fault_inject, verbose=args.verbose)
        for spec in selected
    ]

    if args.expect_failure:
        return _negative_control_verdict(records, args.fault_inject)
    return 0 if all(r["passed"] for r in records) else 1


if __name__ == "__main__":
    sys.exit(main())
