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

WHY THE COMPARISON IS NOT WRITTEN HERE
    ``XRTRunner.run_test`` already is the gate this phase requires: ``np.isclose``
    over the FULL output, ``max_mismatch_percentage`` defaulting to zero, bf16
    upcast before comparing. ``_RecordingRunner`` below subclasses it only to
    copy out the error statistics on the way past -- the verdict is still
    ``XRTRunner._check_outputs``'s, delegated to with ``super()``. A second
    comparison written here would be a second thing to get wrong, and the
    registry's numbers would stop being comparable with every other kernel's.

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

FOOTGUNS
    - The perturbed element is picked strictly BELOW the diagonal for
      ``causal_mask``. Above it the reference is ``-10000``, and
      ``rtol * 10000 = 160`` swallows any perturbation worth making -- the
      negative control would silently pass and prove nothing.
    - External kernel objects are written to the CURRENT WORKING DIRECTORY,
      because that is where aiecc's ``link_with`` search looks. Results, by
      contrast, always land next to this file. Run from a scratch directory.
    - ``encoder.o`` is built with the addnorm half only. Building both halves
      collides with ``addnorm_ffn.o`` on ``ffn_gelu_bf16`` and
      ``ffn_eltwise_add_bf16_vector``.
    - This dispatches to the NPU. Serialize it on ``/tmp/mlir-air-npu.lock``
      -- a DIFFERENT inode from the ``/tmp/npu.lock`` ``XRTRunner`` takes
      internally, since both are BSD ``flock(2)`` and one inode self-deadlocks.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_HERE = Path(__file__).resolve().parent
_PROJ_ROOT = _HERE.parent  # programming_examples/
for _p in (str(_PROJ_ROOT), str(_PROJ_ROOT / "llms"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from air.backend.xrt_runner import XRTRunner  # noqa: E402

import shared.infra.external_kernels as ek  # noqa: E402
from builders.addnorm import addnorm_reference, build_addnorm_module  # noqa: E402
from builders.elementwise_add import (  # noqa: E402
    build_elementwise_add_module,
    causal_mask_bias,
    elementwise_add_reference,
)
from builders.layer_norm import (  # noqa: E402
    build_layer_norm_module,
    layer_norm_reference,
)

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
        self.stats = {
            "mean_rel_L1": abs_err_sum / (ref_abs_sum + 1e-30),
            "rel_err_max": rel_err_max,
            "abs_err_max": abs_err_max,
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


# ---------------------------------------------------------------------------
# Per-operator preparation
#
# Each returns the four things a run needs: the built module, the device inputs,
# the FP32-derived expected outputs, and where to inject a fault. The reference
# is ALWAYS computed here, from the clean inputs, before any injection.
# ---------------------------------------------------------------------------


def _prepare_elementwise_add(shape, seed=0):
    rows, cols = shape["rows"], shape["cols"]
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((rows, cols)).astype(bfloat16)
    b = rng.standard_normal((rows, cols)).astype(bfloat16)
    return {
        "module": build_elementwise_add_module(rows, cols, bfloat16),
        "inputs": [a, b],
        "expected": [elementwise_add_reference(a, b)],
        "inject": (0, (rows - 1, 0)),
    }


def _prepare_causal_mask(shape, seed=1):
    seq = shape["rows"]
    rng = np.random.default_rng(seed)
    scores = rng.standard_normal((seq, seq)).astype(bfloat16)
    mask = causal_mask_bias(seq, bfloat16)
    return {
        # causal_mask=True changes no MLIR; it asserts the square shape and
        # records that `b` is the static mask rather than a residual.
        "module": build_elementwise_add_module(seq, seq, bfloat16, causal_mask=True),
        "inputs": [scores, mask],
        "expected": [elementwise_add_reference(scores, mask)],
        # Strictly below the diagonal: see the module footgun about rtol
        # swallowing a perturbation of a -10000 masked element.
        "inject": (0, (seq - 1, 0)),
    }


def _prepare_layer_norm(shape, seed=2):
    rows, cols = shape["rows"], shape["cols"]
    ek.compile_layer_norm()
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((rows, cols)).astype(bfloat16)
    return {
        "module": build_layer_norm_module(rows, cols, bfloat16),
        "inputs": [x],
        "expected": [layer_norm_reference(x)],
        "inject": (0, (rows - 1, 0)),
    }


def _prepare_addnorm(shape, seed=3):
    rows, cols = shape["rows"], shape["cols"]
    # addnorm half only -- the FFN half collides with addnorm_ffn.o.
    ek.compile_encoder(build_ffn=False, build_addnorm=True)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((rows, cols)).astype(bfloat16)
    residual = rng.standard_normal((rows, cols)).astype(bfloat16)
    # A trained LayerNorm gamma sits near 1; uniform(0.5, 1.5) is that range
    # without being exactly 1, which would hide a dropped weight multiply.
    weight = rng.uniform(0.5, 1.5, size=cols).astype(bfloat16)
    return {
        "module": build_addnorm_module(rows, cols, bfloat16),
        "inputs": [x, residual, weight],
        "expected": [addnorm_reference(x, residual, weight)],
        "inject": (0, (rows - 1, 0)),
    }


# Every (operator, shape) C1 claims. `atol` is the measured worst-case absolute
# error rounded up, per the kernel_registry methodology; `rtol` is fixed at RTOL
# for all of them.
SPECS = [
    {
        "operator": "elementwise_add",
        "shape_key": "512x512",
        "shape": {"rows": 512, "cols": 512},
        "atol": 5e-2,
        "prepare": _prepare_elementwise_add,
    },
    {
        "operator": "causal_mask",
        "shape_key": "512x512",
        "shape": {"rows": 512, "cols": 512},
        "atol": 5e-2,
        "prepare": _prepare_causal_mask,
    },
    {
        "operator": "layer_norm",
        "shape_key": "512x512",
        "shape": {"rows": 512, "cols": 512},
        "atol": 5e-2,
        "prepare": _prepare_layer_norm,
    },
    {
        # 64 rows, not 512: addnorm needs one kernel call per tile, which caps
        # rows at herd_x * (what fits L1). See builders/addnorm.py.
        "operator": "addnorm",
        "shape_key": "64x512",
        "shape": {"rows": 64, "cols": 512},
        "atol": 5e-2,
        "prepare": _prepare_addnorm,
    },
]


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


def _write_result(spec, stats, passed, fault_inject):
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
        "n_elements": stats["n_elements"],
        "n_mismatch": stats["n_mismatch"],
        "passed": passed,
        "fault_injected": fault_inject,
    }
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"[opcheck] wrote {path}")
    return record


def run_spec(spec, fault_inject=None, verbose=False):
    """Run one (operator, shape) on hardware. Returns True if it passed."""
    label = f"{spec['operator']} [{spec['shape_key']}]"
    print(f"[opcheck] {label}: preparing")
    prepared = spec["prepare"](spec["shape"])

    inputs = prepared["inputs"]
    if fault_inject == "input":
        inputs = _inject(inputs, prepared["inject"])
    elif fault_inject is not None:
        raise ValueError(f"unknown fault-inject mode {fault_inject!r}")

    runner = _RecordingRunner(
        verbose=verbose,
        omit_while_true_loop=False,
        output_format="xclbin",
        instance_name=spec["operator"],
        report_precision=True,
    )
    return_code = runner.run_test(
        prepared["module"],
        inputs=inputs,
        expected_outputs=prepared["expected"],
        rtol=RTOL,
        atol=spec["atol"],
    )
    passed = return_code == 0
    _write_result(spec, runner.stats, passed, fault_inject)

    verdict = "PASS" if passed else "FAIL"
    print(f"[opcheck] {label}: {verdict}")
    if fault_inject:
        # Under injection FAIL is the desired outcome, and a PASS means the
        # check is not actually reading the device's inputs.
        control = "as required" if not passed else "NEGATIVE CONTROL DID NOT FAIL"
        print(
            f"[opcheck] {label}: fault-inject {fault_inject} -> {verdict} ({control})"
        )
    return passed


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

    selected = [s for s in SPECS if s["operator"] == args.operator]
    if args.shape_key:
        selected = [s for s in selected if s["shape_key"] == args.shape_key]
    if not selected:
        parser.error(
            f"no such (operator, shape_key): {args.operator} / {args.shape_key}"
        )

    ok = True
    for spec in selected:
        ok = run_spec(spec, fault_inject=args.fault_inject, verbose=args.verbose) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
