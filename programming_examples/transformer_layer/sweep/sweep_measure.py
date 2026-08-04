# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One candidate configuration: build it, check it numerically, time it.

This is the process the sweep forks per candidate. It is also runnable by hand,
which is the only practical way to debug a candidate that will not place:

    python3 sweep_measure.py --payload payload.json --result result.json

    payload.json = {"shape": {...}, "candidate": {...}, "perf_iters": 20}

Exit status is 0 whenever a verdict was REACHED, including a verdict of "this
configuration does not work". A non-zero exit means the measurement itself broke
down and nothing about the candidate was learned -- the orchestrator records
those two outcomes differently on purpose.

WHAT "CHECK IT NUMERICALLY" MEANS HERE
    The same comparison Phase C1 gates every operator on:
    ``XRTRunner._check_outputs`` -- ``np.isclose`` over the FULL output with
    ``max_mismatch_percentage = 0`` -- reached through ``opcheck._RecordingRunner``,
    which subclasses it only to copy out the error statistics on the way past.
    The verdict is still ``XRTRunner``'s, delegated to with ``super()``.

    Importing a private name across modules is deliberate and is the lesser
    evil. ``opcheck.py``'s own docstring records why a second comparison must not
    be written: the registry's numbers stop being comparable with every other
    kernel's the moment two different checks produce them. The alternative to
    this import is exactly that second check.

WHERE THE TIMING COMES FROM, AND WHY IT IS NOT ``Profiler``
    ``XRTRunner``'s own ``n_perf_iters`` / ``perf_flops`` / ``last_latency_us``
    (``python/air/backend/xrt_runner.py``), which times ``run.start()`` +
    ``wait2()`` over N iterations after a warmup, inside the backend. That is
    not a second measurement mechanism standing beside the repository's -- it
    IS the repository's per-kernel one, and specifically it is the one every
    row already in this registry was measured with: the reproduction command
    the detail page publishes for those rows is
    ``make run ... PERF_ITERS=20``, which reaches the same code path through
    ``bf16_in_bf16_out/run.py:1251``.

    Porting convention 10 points measurement at ``Profiler`` and
    ``llms/bench/extract_perf.py``. Neither fits here, and using them would
    cost the property that matters. ``Profiler`` accumulates per-kernel and
    per-layer times across a multi-kernel inference run inside ``KernelCache``;
    this sweep dispatches one kernel per process through ``run_test``, which is
    also what performs the numerical check the verdict depends on.
    ``extract_perf.py`` scrapes whole-model metrics -- TTFT, tok/s, prompt_len
    -- for ``append_history.py`` and the published per-model charts; it has no
    field for a per-shape GEMM latency and nothing downstream of it would know
    what to do with 432 of them.

    The rule those two exist to enforce is that a ported measurement must not
    invent its own timing. This one does not: it uses the same timer, over the
    same iteration count, on the same inputs, at the same enforced power mode
    as the numbers it will be ranked against.

THE ACCEPTANCE TOLERANCE, AND WHY IT IS NOT A CONSTANT
    ``rtol`` is the canonical bf16 1.6e-2 at every shape and every tier, held
    fixed across the whole registry.

    ``atol`` is the registry's floor for elements too near zero for ``rtol`` to
    bite, and the registry's rule for it is "sized to the kernel's measured
    worst-case absolute error" with roughly 2.5x margin
    (``kernel_registry/details/GEMM_bf16_in_bf16_out.md``). The published
    constant, 1.5e-3, is that rule evaluated at ONE shape: 2048x8192x2048.

    It does not transfer, because the harness scales its inputs by ``1/sqrt(K)``
    (``bf16_in_bf16_out/run.py``), so the output magnitude -- and with it the
    absolute error of a fixed-relative-precision datapath -- goes as
    ``K^-1/2``. Holding ``atol`` constant across K therefore does not hold the
    check's strictness constant; it tightens it by ``sqrt(K_anchor / K)`` for
    every shape narrower than the anchor. At this family's ``K = 768`` that is a
    3.3x tightening, which rejects a datapath measurably in the same precision
    tier as the anchor.

    So the high tier's ``atol`` carries the anchor forward at CONSTANT
    STRICTNESS: ``1.5e-3 * sqrt(8192 / K)``. Measured on hardware, three points:

        K = 8192   atol_required 6.10e-4   vs 1.50e-3   margin 2.5x
        K = 3072   atol_required 8.55e-4   vs 2.45e-3   margin 2.9x
        K =  768   atol_required 1.73e-3   vs 4.90e-3   margin 2.8x

    -- i.e. the derived bound reproduces the registry's own design margin at
    every K, rather than drifting from 2.5x to 0.9x as a fixed constant does.

    The LOW tier keeps the published 4e-3 unscaled. Its error comes from
    per-L2-tile truncation, which accumulates over ``K / tile_k_l2`` truncations
    and so largely cancels the ``K^-1/2`` magnitude scaling: measured worst-case
    is 2.4e-3 at K=8192 and 2.9e-3 at K=768, a 1.2x spread rather than 3.3x.
    Scaling it would loosen the low tier well past the registry's stated 1.6x
    margin, so it is left alone -- deliberately the conservative choice, and a
    candidate that exceeds it simply contributes no ``low`` entry.

    Neither knob is a per-shape escape hatch: both are pure functions of the
    tier and K, every record carries ``atol_required`` and the achieved
    ``atol_margin``, and a run must ALSO land inside its tier's
    ``mean_rel_L1`` band (below) -- a scale-free check the example harness does
    not perform at all.

FOOTGUNS
    - ``mm.o`` is rebuilt per candidate into the CURRENT WORKING DIRECTORY,
      because that is where aiecc's ``link_with`` search looks, and because its
      ``-DDIM_M`` / ``-DDIM_N`` / ``-DDIM_K`` must match this candidate's
      ``tile_m`` / ``tile_n`` / ``tile_k_l1``. A stale ``mm.o`` from another
      candidate does not fail to link. Run each candidate in its own directory.
    - fused-cast is the only multi-launch module here, and it needs ELF output;
      the default xclbin path mis-runs it. drain and direct are single-launch and
      use xclbin. Getting this backwards produces wrong numbers, not an error.
    - Inputs are regenerated from seed 42 in the same order as the example
      harness, so a row here is reproducible by that harness's own documented
      command line. Changing the order changes every measured number.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_HERE = Path(__file__).resolve().parent
_EXAMPLE_DIR = _HERE.parent  # programming_examples/transformer_layer/
_PROJ_ROOT = _EXAMPLE_DIR.parent  # programming_examples/
for _p in (str(_PROJ_ROOT), str(_PROJ_ROOT / "llms"), str(_EXAMPLE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from opcheck import RTOL, _RecordingRunner  # noqa: E402

from sweep_families import METHODS  # noqa: E402

_MM_SRC = _PROJ_ROOT / "matrix_multiplication" / "bf16_in_bf16_out" / "mm_aie2p.cc"

# The published tolerance and the shape it was measured at. See the module
# docstring for why the pair travels together.
ATOL_ANCHOR_HIGH = 1.5e-3
K_ANCHOR = 8192
ATOL_LOW = 4e-3

# Upper bound on mean_rel_L1 for each precision tier, from the registry's own
# per-tier ranges (high 9.3-9.9e-3, low 1.0-1.9e-2) with headroom. This is the
# scale-free half of the acceptance check: a candidate that silently dropped to
# bf16 accumulation would still satisfy a loose atol at a narrow K, but it cannot
# satisfy the high tier's band.
TIER_MEAN_REL_L1_MAX = {"high": 1.2e-2, "low": 2.0e-2}


def acceptance_tolerances(tier, k):
    """(rtol, atol) this tier must be checked at, for a GEMM of depth ``k``."""
    if tier == "high":
        return RTOL, ATOL_ANCHOR_HIGH * math.sqrt(K_ANCHOR / k)
    if tier == "low":
        return RTOL, ATOL_LOW
    raise ValueError(f"unknown precision tier {tier!r}")


def compile_mm_object(tile_m, tile_n, tile_k_l1, out_path):
    """Build the external GEMM microkernel this candidate links.

    Mirrors ``bf16_in_bf16_out/Makefile``'s ``compile-kernel`` recipe. The
    ``DIM_*`` values are the candidate's L1 tile, so the object is only valid for
    the candidate that built it.
    """
    peano = os.environ.get("PEANO_INSTALL_DIR")
    if not peano:
        raise RuntimeError(
            "PEANO_INSTALL_DIR is unset; the external GEMM microkernel cannot "
            "be built and every high-precision candidate would fail to link."
        )
    aie_opt = shutil.which("aie-opt")
    if not aie_opt:
        raise RuntimeError("aie-opt is not on PATH; cannot locate the AIE headers")
    aieopt_dir = Path(aie_opt).resolve().parent.parent
    cmd = [
        f"{peano}/bin/clang++",
        "-O2",
        "-std=c++20",
        "--target=aie2p-none-unknown-elf",
        "-Wno-parentheses",
        "-Wno-attributes",
        "-Wno-macro-redefined",
        "-Wno-empty-body",
        "-DNDEBUG",
        "-I",
        str(aieopt_dir / "include"),
        "-D__AIE_API_AIE_ADF_HPP__",
        "-DBIT_WIDTH=8",
        "-c",
        str(_MM_SRC),
        "-o",
        str(out_path),
        f"-DDIM_M={tile_m}",
        f"-DDIM_N={tile_n}",
        f"-DDIM_K={tile_k_l1}",
        f"-DDIM_N_DIV_4={tile_n // 4}",
        f"-DDIM_M_DIV_4={tile_m // 4}",
        f"-DDIM_N_DIV_8={tile_n // 8}",
        f"-DDIM_M_DIV_8={tile_m // 8}",
        "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"mm.o build failed for tile_m={tile_m} tile_n={tile_n} "
            f"tile_k_l1={tile_k_l1}:\n{result.stderr[-2000:]}"
        )


def _inputs_and_reference(m, k, n):
    """The example harness's inputs, byte-for-byte.

    Seeded and generated in the same order as
    ``bf16_in_bf16_out/run.py``, so any row this sweep writes can be reproduced
    from that harness's documented command line. The reference is FP32 through
    the reduction and cast once to bf16, which is the registry's stated standard.
    """
    np.random.seed(42)
    scale = 1.0 / math.sqrt(k)
    input_a = (np.random.randn(m, k) * scale).astype(bfloat16)
    input_b = (np.random.randn(k, n) * scale).astype(bfloat16)
    reference = (input_a.astype(np.float32) @ input_b.astype(np.float32)).astype(
        bfloat16
    )
    return input_a, input_b, reference


def _build(shape, candidate):
    """(module, instance_name, inputs, output_format) for one candidate."""
    from matrix_multiplication.bf16_in_bf16_out.run import (
        build_module,
        build_module_gemm_cast,
        build_module_lowered,
    )

    m, k, n = shape["M"], shape["K"], shape["N"]
    tm = candidate["tile_m"]
    tk2 = candidate["tile_k_l2"]
    tk1 = candidate["tile_k_l1"]
    tn = candidate["tile_n"]
    hm, hn = candidate["herd_m"], candidate["herd_n"]
    method = candidate["method"]

    input_a, input_b, reference = _inputs_and_reference(m, k, n)

    if method == "fused-cast":
        module = build_module_gemm_cast(
            m,
            k,
            n,
            tm,
            tk2,
            tk1,
            tn,
            hm,
            hn,
            arch="aie2p",
            cast_tile_n=candidate["cast_tile_n"],
        )
        # A, B, and the f32 C scratch the GEMM launch drains into.
        inputs = [input_a, input_b, np.zeros((m, n), dtype=np.float32)]
        return module, "gemm_cast_bf16", inputs, [reference], "elf"

    if method == "drain":
        module = build_module(
            m,
            k,
            n,
            tm,
            tk2,
            tk1,
            tn,
            hm,
            hn,
            bfloat16,
            bfloat16,
            arch="aie2p",
            emit_external_call=True,
        )
        return module, "matmul_bf16", [input_a, input_b], [reference], "xclbin"

    if method == "direct":
        module = build_module_lowered(
            m, k, n, tm, tk2, tk1, tn, hm, hn, bfloat16, bfloat16, arch="aie2p"
        )
        return module, "matmul_bf16", [input_a, input_b], [reference], "xclbin"

    raise ValueError(f"unknown method {method!r}")


def measure_candidate(shape, candidate, perf_iters=20, verbose=False):
    """Build, check and time one candidate. Returns the record; never raises for
    a candidate that simply does not work."""
    m, k, n = shape["M"], shape["K"], shape["N"]
    tier = METHODS[candidate["method"]]["tier"]
    rtol, atol = acceptance_tolerances(tier, k)

    record = {
        "shape": dict(shape),
        "candidate": dict(candidate),
        "tier": tier,
        "rtol": rtol,
        "atol": atol,
        "perf_iters": perf_iters,
        "status": None,
        "gflops": None,
        "latency_us": None,
        "error": None,
    }

    try:
        if candidate["method"] in ("fused-cast", "drain"):
            compile_mm_object(
                candidate["tile_m"],
                candidate["tile_n"],
                candidate["tile_k_l1"],
                Path.cwd() / "mm.o",
            )
        module, instance, inputs, expected, output_format = _build(shape, candidate)
    except Exception as exc:  # noqa: BLE001 - any build failure is a verdict
        record["status"] = "failed_build"
        record["error"] = f"{type(exc).__name__}: {exc}"[:2000]
        return record

    runner = _RecordingRunner(
        verbose=verbose,
        omit_while_true_loop=False,
        runtime_loop_tiling_sizes=[2, 2],
        stack_size=2048,
        report_precision=True,
        n_perf_iters=perf_iters,
        perf_flops=(2.0 * m * k * n) if perf_iters > 0 else None,
        output_format=output_format,
        instance_name=instance,
    )

    try:
        return_code = runner.run_test(
            module, inputs=inputs, expected_outputs=expected, rtol=rtol, atol=atol
        )
    except Exception as exc:  # noqa: BLE001 - a placement failure is a verdict
        record["status"] = "failed_build"
        record["error"] = f"{type(exc).__name__}: {exc}"[:2000]
        return record

    if runner.stats is None:
        # run_test returned without the comparison ever running. Nothing was
        # learned about this candidate, so it must not be recorded as rejected.
        record["status"] = "failed_no_comparison"
        record["error"] = "run_test completed without reaching the output check"
        return record

    record.update(runner.stats)
    record["latency_us"] = runner.last_latency_us
    if runner.last_latency_us:
        record["gflops"] = (2.0 * m * k * n) / (runner.last_latency_us * 1e-6) / 1e9
    record["atol_margin"] = (
        atol / record["atol_required"] if record["atol_required"] > 0 else None
    )

    tier_max = TIER_MEAN_REL_L1_MAX[tier]
    if return_code != 0:
        record["status"] = "failed_precision"
        record["error"] = (
            f"{record['n_mismatch']}/{record['n_elements']} elements outside "
            f"rtol={rtol:.1e} atol={atol:.2e} (needs atol >= "
            f"{record['atol_required']:.2e})"
        )
    elif record["mean_rel_L1"] > tier_max:
        # Passed the element-wise check but landed outside its precision tier --
        # the datapath is not the one this tier's row would claim.
        record["status"] = "failed_tier"
        record["error"] = (
            f"mean_rel_L1={record['mean_rel_L1']:.3e} exceeds the {tier} tier "
            f"band of {tier_max:.1e}"
        )
    elif record["gflops"] is None:
        record["status"] = "failed_no_timing"
        record["error"] = "the run passed but reported no latency to rank it by"
    else:
        record["status"] = "passed"

    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--payload", required=True, help="JSON file: shape + candidate")
    parser.add_argument("--result", required=True, help="JSON file to write")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    payload = json.loads(Path(args.payload).read_text())
    try:
        record = measure_candidate(
            payload["shape"],
            payload["candidate"],
            perf_iters=payload.get("perf_iters", 20),
            verbose=args.verbose,
        )
    except Exception:  # noqa: BLE001
        # measure_candidate is meant to convert a bad candidate into a verdict,
        # so reaching here means the harness itself broke. Say so, and exit
        # non-zero so the orchestrator does not bank a verdict that was never
        # reached.
        traceback.print_exc()
        return 2

    Path(args.result).write_text(json.dumps(record, indent=2) + "\n")
    print(f"[sweep] {record['status']}: {record.get('error') or ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
