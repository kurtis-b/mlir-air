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
    - encoder.cc is built TWICE here, to two objects, each with one half:
      ``encoder.o`` (addnorm half, for ``addnorm``) and ``encoder_ffn.o`` (FFN
      half, for ``ffn``). Building both halves into one object collides with
      ``addnorm_ffn.o`` on ``ffn_gelu_bf16`` and
      ``ffn_eltwise_add_bf16_vector``, and neither operator needs the other's
      half. ``compile_kernels.py`` checks that the split actually happened.
    - The GEMM-backed operators need extra backend settings that the C1
      operators do not -- BD-ID recycling and ELF output. They are declared per
      spec, so adding an operator that forgets them fails to place rather than
      quietly inheriting the wrong ones. See ``_GEMM_RUNNER_KWARGS``.
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
from builders.ffn import (  # noqa: E402
    build_ffn_module,
    ffn_device_inputs,
    ffn_gemm_specs,
    ffn_reference,
)
from builders.layer_norm import (  # noqa: E402
    build_layer_norm_module,
    layer_norm_reference,
)
from builders.qkv_proj import (  # noqa: E402
    build_qkv_proj_module,
    qkv_gemm_spec,
    qkv_proj_reference,
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


# ---------------------------------------------------------------------------
# GEMM-backed operators (C2)
#
# HOW THEIR DATA IS SCALED, AND WHY IT IS NOT A FREE CHOICE
#     The external GEMM's error is dominated by its bfp16 MMUL emulation, not by
#     the bf16 output rounding: the registry records mean_rel_L1 = 9.3e-3 for it
#     and these runs measure 9.7e-3, i.e. roughly 1% of the OUTPUT's own
#     magnitude. So the absolute error a run reports is proportional to how big
#     the output is, and quoting an `atol` only means something alongside the
#     scale it was measured at.
#
#     The scale used here is the one the registry's own GEMM sweep uses
#     (matrix_multiplication/bf16_in_bf16_out/run.py): operands ~ N(0, 1/sqrt(K))
#     for a reduction of depth K, which puts the product at 1/sqrt(K). Every
#     `atol` below is then the MEASURED worst-case absolute error at that scale
#     rounded up, ~2-3x, exactly as the registry's GEMM rows are sized -- and it
#     makes the mean_rel_L1 recorded here directly comparable with them.
#
#     The one departure is the FFN's up-projection, which is scaled to put its
#     output at unit variance instead. GeLU is only interesting on |x| ~ 1-3;
#     at 1/sqrt(K) the whole tensor would sit inside +/-0.15 where the
#     activation is indistinguishable from 0.5x, and the operator's own stage
#     would go untested.
#
# WHY THEIR RUNS NEED EXTRA BACKEND SETTINGS
#     `runtime_loop_tiling_sizes=[2,2]` for BD-ID recycling, and ELF output for
#     multi-segment designs. Both are placement/packaging concerns; neither
#     touches the comparison. See _GEMM_RUNNER_KWARGS.
# ---------------------------------------------------------------------------

# Backend settings both GEMM-backed operators need.
#   runtime_loop_tiling_sizes: BD-ID recycling. A fused-cast GEMM herd runs out
#     of buffer descriptors without it, which surfaces as a placement failure.
#   output_format "elf": these designs are MULTI-SEGMENT -- one aie.device per
#     air.launch, driven by an aiex.configure/aiex.run runtime sequence. The
#     xclbin path names a single instruction blob on the aircc command line, so
#     a second segment collides on it and aiecc stops with "produced duplicate
#     output path". ELF is the format the shipped multi-launch llama builders
#     use for exactly this reason, and it is not a relaxation of anything: the
#     comparison downstream is identical.
_GEMM_RUNNER_KWARGS = {
    "runtime_loop_tiling_sizes": [2, 2],
    "output_format": "elf",
}


def _registry_gemm_scale(k):
    """Operand scale the registry's GEMM sweep uses for a depth-``k`` reduction.

    Both operands at ``1/sqrt(k)`` put the product at ``1/sqrt(k)`` too. Copied
    from ``matrix_multiplication/bf16_in_bf16_out/run.py`` so the error figures
    recorded here sit on the same axis as the registry's GEMM rows.
    """
    return 1.0 / np.sqrt(k)


def _unit_output_scale(k):
    """Operand scale that puts a depth-``k`` product at unit variance.

    Both operands at ``k^-0.25``: the reduction multiplies the two scales ``k``
    times. Used only where a downstream stage needs activation-sized inputs --
    the FFN's GeLU.
    """
    return k**-0.25


# E[gelu(x)^2] for x ~ N(0, 1). Used to size the down-projection weight so its
# output lands where the registry's GEMM sweep would put a depth-`ffn_dim`
# reduction, rather than sqrt(1/0.42) ~ 1.5x above it.
_GELU_SECOND_MOMENT = 0.42


def _prepare_qkv_proj(shape, seed=4):
    seq_len, emb_dim = shape["seq_len"], shape["emb_dim"]
    n_total = 3 * emb_dim
    spec, source = qkv_gemm_spec(seq_len, emb_dim)
    ek.compile_gemm_mm(
        tile_m=spec["tile_m"],
        tile_n=spec["tile_n"],
        tile_k_l1=spec["tile_k_l1"],
        sym_suffix=spec["sym_suffix"],
        out_name=spec["obj"],
    )
    rng = np.random.default_rng(seed)
    scale = _registry_gemm_scale(emb_dim)
    x = (rng.standard_normal((seq_len, emb_dim)) * scale).astype(bfloat16)
    w = (rng.standard_normal((emb_dim, n_total)) * scale).astype(bfloat16)
    return {
        "module": build_qkv_proj_module(seq_len, emb_dim),
        # arg2 is the f32 C scratch: an input slot the device writes, not an
        # output. See builders/qkv_proj.py on why it precedes q/k/v.
        "inputs": [x, w, np.zeros((seq_len, n_total), dtype=np.float32)],
        "expected": list(qkv_proj_reference(x, w)),
        "inject": (0, (seq_len - 1, 0)),
        "runner_kwargs": _GEMM_RUNNER_KWARGS,
        "record_extra": {"gemm_spec_source": source, "gemm_spec": _spec_digest(spec)},
    }


def _prepare_ffn(shape, seed=5):
    seq_len, emb_dim = shape["seq_len"], shape["emb_dim"]
    ffn_dim = shape["ffn_dim"]
    up_spec, down_spec, source = ffn_gemm_specs(seq_len, emb_dim, ffn_dim)
    for spec in (up_spec, down_spec):
        ek.compile_gemm_mm(
            tile_m=spec["tile_m"],
            tile_n=spec["tile_n"],
            tile_k_l1=spec["tile_k_l1"],
            sym_suffix=spec["sym_suffix"],
            out_name=spec["obj"],
        )
    # encoder.cc's FFN half only. Building the addnorm half too would collide
    # with addnorm_ffn.o on ffn_gelu_bf16, and this ELF needs neither.
    ek.compile_encoder(build_ffn=True, build_addnorm=False, out_name="encoder_ffn.o")

    rng = np.random.default_rng(seed)
    # Up-projection at unit output variance so GeLU is exercised where it is
    # nonlinear; down-projection weight sized to land y where the registry's
    # sweep puts a depth-ffn_dim reduction. See the section header above.
    up_scale = _unit_output_scale(emb_dim)
    down_scale = _registry_gemm_scale(ffn_dim) / np.sqrt(ffn_dim * _GELU_SECOND_MOMENT)
    x = (rng.standard_normal((seq_len, emb_dim)) * up_scale).astype(bfloat16)
    w_up = (rng.standard_normal((emb_dim, ffn_dim)) * up_scale).astype(bfloat16)
    w_down = (rng.standard_normal((ffn_dim, emb_dim)) * down_scale).astype(bfloat16)
    return {
        "module": build_ffn_module(seq_len, emb_dim, ffn_dim),
        "inputs": ffn_device_inputs(x, w_up, w_down, up_spec, down_spec),
        "expected": [ffn_reference(x, w_up, w_down)],
        "inject": (0, (seq_len - 1, 0)),
        "runner_kwargs": _GEMM_RUNNER_KWARGS,
        "record_extra": {
            "gemm_spec_source": source,
            "gemm_spec_up": _spec_digest(up_spec),
            "gemm_spec_down": _spec_digest(down_spec),
        },
    }


def _spec_digest(spec):
    """The part of a GEMM spec worth recording: method and the four tiles.

    Written into the results artifact so a spec that came from ``gemm_spec_fn``
    rather than the registry is VISIBLE there. The phase allows that injection
    hook for an unmeasured shape precisely on the condition that the guess is
    not silent.
    """
    return {
        "method": spec["method"],
        "tile_m": spec["tile_m"],
        "tile_k_l2": spec["tile_k_l2"],
        "tile_k_l1": spec["tile_k_l1"],
        "tile_n": spec["tile_n"],
    }


# Every (operator, shape) C1 and C2 claim. `atol` is the measured worst-case
# absolute error rounded up, per the kernel_registry methodology; `rtol` is fixed
# at RTOL for all of them.
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
    {
        # The only registered projection shapes are (M, K, 3K) for K in
        # {1024, 2048}; 2048x1024x3072 is the smaller of the two. See the
        # structured report for the case-matrix shapes C4 still has to measure.
        "operator": "qkv_proj",
        "shape_key": "2048x1024",
        "shape": {"seq_len": 2048, "emb_dim": 1024},
        # Measured abs_err max 1.95e-3 over 6.3M elements at the registry GEMM
        # scale, mean_rel_L1 9.9e-3 -- which is the registry's own 9.3e-3 for
        # this GEMM, so the split-cast launches add nothing measurable. atol is
        # that worst case rounded up, 2.6x.
        "atol": 5e-3,
        "prepare": _prepare_qkv_proj,
    },
    {
        # The larger of the two registered (M, K, 3K) triples.
        "operator": "qkv_proj",
        "shape_key": "2048x2048",
        "shape": {"seq_len": 2048, "emb_dim": 2048},
        "atol": 5e-3,
        "prepare": _prepare_qkv_proj,
    },
    {
        "operator": "ffn",
        "shape_key": "2048x1024x3072",
        "shape": {"seq_len": 2048, "emb_dim": 1024, "ffn_dim": 3072},
        # Measured abs_err max 1.59e-3 over 2.1M elements, mean_rel_L1 1.6e-2 --
        # roughly 1.6x the single GEMM's, which is where the bf16 staging of h
        # and the activation's bf16 intermediates show up against an FP32
        # reference. atol is that worst case rounded up, 3.1x.
        "atol": 5e-3,
        "prepare": _prepare_ffn,
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
    return_code = runner.run_test(
        prepared["module"],
        inputs=inputs,
        expected_outputs=prepared["expected"],
        rtol=RTOL,
        atol=spec["atol"],
    )
    passed = return_code == 0
    record = _write_result(
        spec, runner.stats, passed, fault_inject, extra=prepared.get("record_extra")
    )

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
