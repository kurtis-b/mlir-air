# Can an `offload` GEMM artifact execute twice in ONE `hw_context`?
#
# WHAT THIS ESTABLISHED `[2026-08-09]`, as a 2x2 factorial on real NPU2 --
# `q_proj` 1024x768x768, drain tile_n=96, herd 8x4, four executions of ONE artifact on
# ONE pair of inputs, with the CONTROL arm (evict between runs) run first:
#
#   ABI      tiling   reuse across 4 runs
#   elf      [2,2]    CORRUPT -- mean_rel_L1 vs run 1 = 3.8141e-01   (replicated 2/2)
#   elf      [1,1]    clean   -- bit-identical, 4/4
#   xclbin   [2,2]    clean   -- bit-identical, 4/4
#   xclbin   [1,1]    clean   -- bit-identical, 4/4
#   (control: elf [2,2] WITH eviction -- clean, bit-identical, 4/4)
#
# EXACTLY ONE CELL OF FOUR FAILS, AND IT IS THE ONE `offload` SHIPS. The control's
# reference error is 9.5676e-03, which is the `9.6e-3` `_evict_context` cites -- so this
# probe reproduces the documented measurement before contradicting its scope.
#
# THREE CORRECTIONS TO `_evict_context`'s ACCOUNT, which says the corruption belongs to
# "these runtime-tiled GEMM ELFs" as a class:
#
#   1. It is not the class. It is `output_format="elf"` AND `[2,2]` TOGETHER. Either knob
#      alone removes it. The eviction is a workaround for one cell, not a device law.
#   2. It does not accumulate. Runs 2, 3 and 4 are identical to EACH OTHER and differ
#      only from run 1, so it is a one-time state change after the first execution rather
#      than progressive decay. The original note's "from the SECOND execution onward" is
#      right and reads as though it might worsen; it does not.
#   3. `runtime_loop_tiling_sizes=[2,2]` now has TWO measured pathologies, not one. It
#      hangs `mha_out_proj` @4096 (`probe_backend_preset_hardware.py`), and it leaves
#      context-corrupting residue in a plain projection GEMM under the ELF ABI. Doc 26 §4
#      called the knob "inert" from a compile-only diff; this is the second independent
#      hardware refutation of that.
#
# WHAT IT UNBLOCKS. `offload`'s remaining half is N instruction streams under ONE xclbin,
# which requires the xclbin ABI -- and under that ABI reuse is clean AT THE MODE'S
# EXISTING `[2,2]` TILING. So the work needs no retune and is not blocked on a device
# defect: the `plan_submissions` split rule can be written against a context that is safe
# to share. Separately, the mode pays 30 context load/unload cycles per layer today
# (`_dispatch_gemm` evicts before every dispatch, 12 heads x 2 GEMMs + 6 projections),
# which is the MAXIMUM reconfiguration cost in the mode 03 defines as minimizing it, and
# is the leading candidate for the 120% intra-walk latency variance recorded in
# 27-common-ladder-result.md.
#
# WHY THIS EXISTS.  `pattern/offload/offload.py:465` (`_evict_context`) records, as a
# measurement rather than a precaution, that re-executing one of these runtime-tiled GEMM
# ELFs in a reused `hw_context` returns wrong numbers from the SECOND execution onward --
# mean_rel_L1 3.56e-1 against the same run's own 9.6e-3 on a fresh context, same inputs,
# uniformly across rows and columns, at roughly one third of the reduction lost.
# Stale-input and stale-output explanations were ruled out directly. So the mode reloads
# the context before EVERY dispatch: 30 times per layer at any sequence length.
#
# WHAT TURNS ON IT.  Two things, and they point opposite ways.
#
#   1. `offload` is defined by 03 as "reconfiguration MINIMIZED by dynamic partitioning",
#      and it currently pays a full context teardown+setup per dispatch, which is the
#      MAXIMUM reconfiguration cost available. The mode is not measuring what it is for.
#   2. The remaining half of the mode -- N instruction streams under ONE xclbin -- exists
#      precisely to share one array configuration across dispatches. That is the exact
#      pattern the eviction exists to prevent. The `plan_submissions` split rule is a
#      small edit and it is USELESS until this question is answered: landing it first
#      yields a mode that aggregates perfectly and returns 3.56e-1.
#
# So this probe is the gate on that work, and it is deliberately cheap.
#
# WHAT IT CHANGES, AND WHAT IT HOLDS FIXED.  One artifact, one shape, one pair of inputs,
# executed `--runs` times back to back. The ONLY variable is whether the context is
# evicted between executions, crossed with the two backend knobs that could plausibly own
# the residue: the ABI (`elf` vs `xclbin`) and `runtime_loop_tiling_sizes`. The `evict`
# arm is the CONTROL -- it is what the mode does today, so it must come back clean, and if
# it does not the probe is wrong rather than the mode.
#
# THE COMPARISON THAT MATTERS is run N against run 1, not against the reference. A
# reference comparison conflates the bf16 GEMM's own error with the residue; run-to-run
# divergence with identical inputs is the residue alone, and it is zero on a correct
# device.
#
# Each arm must run in its OWN PROCESS on an exclusive device -- doc 23 records a run
# where the same mode and shape passed alone and failed as a later rung of a shared
# process. Submit through `agents/scripts/devq.sh run --class measure`.
#
# Usage:  python3 probe_context_reuse.py <arm> [--seq N] [--runs N]
#   evict          elf  [2,2], evicting between runs -- THE CONTROL, must pass
#   reuse          elf  [2,2], reusing            -- reproduces the documented defect?
#   reuse-t11      elf  [1,1], reusing            -- is it the tiling?
#   reuse-xclbin   xclbin [2,2], reusing          -- is it the ABI? DECIDES TRACK C
#   reuse-xclbin-t11  xclbin [1,1], reusing       -- both knobs moved
import argparse
import os
import sys

import numpy as np

_TL = "/home/cj/mlir-air/programming_examples/transformer_layer"
_PROJ = "/home/cj/mlir-air/programming_examples"
for _p in (_PROJ, os.path.join(_PROJ, "llms"), _TL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ml_dtypes import bfloat16  # noqa: E402

from shared.infra.bo_pool import BufferSpec, DispatchStep  # noqa: E402
from shared.infra.cache import KernelCache, Profiler  # noqa: E402

from pattern.offload.offload import (  # noqa: E402
    _build_offload_module,
    offload_config,
)

#: arm -> (backend knob overrides, evict between runs). The knobs are overlaid on the
#: mode's own `backend_kwargs`, so anything not named here stays exactly as the shipped
#: mode builds it.
ARMS = {
    "evict": ({}, True),
    "reuse": ({}, False),
    "reuse-t11": ({"runtime_loop_tiling_sizes": [1, 1]}, False),
    "reuse-xclbin": ({"output_format": "xclbin"}, False),
    "reuse-xclbin-t11": (
        {"output_format": "xclbin", "runtime_loop_tiling_sizes": [1, 1]},
        False,
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=sorted(ARMS))
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument(
        "--op",
        default="q_proj",
        help="which of the mode's GEMMs to re-execute; default is a plain "
        "drain projection, the simplest artifact the mode owns",
    )
    args = ap.parse_args()
    overrides, evict = ARMS[args.arm]

    cfg = offload_config(args.seq, 768, 3072, 12, 64)
    key = cfg["gemms"][args.op]
    spec, (m, k, n) = cfg["specs"][key]
    artifact = cfg["artifacts"][key]

    # This arm's OWN cache directory, outside the repo. Two arms sharing one could trade
    # ELFs whose fingerprints happen to agree -- the same failure mode every mode's cache
    # docstring warns about -- and these arms differ precisely in backend kwargs.
    cache_dir = f"/home/cj/.claude/jobs/e75c34c9/tmp/ctxprobe/{args.arm}_{args.seq}"
    os.makedirs(cache_dir, exist_ok=True)

    kwargs = dict(cfg["backend_kwargs"][artifact])
    kwargs.update(overrides)

    print(f"[probe] arm      : {args.arm}  (evict_between_runs={evict})")
    print(f"[probe] artifact : {artifact}  {args.op} {m}x{k}x{n}")
    print(f"[probe] spec     : {spec['method']} tile_n={spec['tile_n']} "
          f"herd {spec['herd_m']}x{spec['herd_n']}")
    print(f"[probe] kwargs   : {kwargs}")
    print(f"[probe] runs     : {args.runs}")

    cache = KernelCache(cache_dir=cache_dir, verbose=False, profiler=Profiler(False))
    module = _build_offload_module(cfg, key)
    print(f"[probe] compiling {artifact} ...", flush=True)
    cache.compile_and_cache(artifact, module, kwargs)

    rng = np.random.default_rng(0)
    a = rng.standard_normal((m, k)).astype(np.float32).astype(bfloat16)
    b = (rng.standard_normal((k, n)) / np.sqrt(k)).astype(np.float32).astype(bfloat16)
    reference = a.astype(np.float32) @ b.astype(np.float32)

    if spec["needs_f32_scratch"]:
        arg_names, writes = ("a", "b", "c_f32", "c"), (2, 3)
        host_writes = {"a", "b", "c_f32"}
    else:
        arg_names, writes = ("a", "b", "c"), (2,)
        host_writes = {"a", "b"}

    outputs = []
    for i in range(args.runs):
        if evict and i:
            loaded = cache._loaded.pop(artifact, None)
            if loaded is not None:
                loaded[0].unload()
            cache._pools.clear()

        arrays = {"a": np.ascontiguousarray(a), "b": np.ascontiguousarray(b),
                  "c": np.zeros((m, n), dtype=bfloat16)}
        if spec["needs_f32_scratch"]:
            arrays["c_f32"] = np.zeros((m, n), dtype=np.float32)
        specs = {
            name: BufferSpec(name=name, nbytes=arr.size * arr.itemsize, static=False,
                             host_output=name == "c", content_key=None)
            for name, arr in arrays.items()
        }
        results, _ = cache.run_sequence(
            [DispatchStep(artifact, arg_names, writes=writes)],
            specs, {artifact: kwargs}, arrays, host_writes=host_writes,
        )
        outputs.append(np.array(results["c"], copy=True).astype(np.float32))

        rel = np.abs(outputs[-1] - reference).mean() / np.abs(reference).mean()
        # vs run 1 is the residue on its own: identical inputs, so a correct device
        # returns bit-identical output and this is exactly 0.
        drift = (
            np.abs(outputs[-1] - outputs[0]).mean() / np.abs(outputs[0]).mean()
            if i else 0.0
        )
        exact = np.array_equal(outputs[-1], outputs[0])
        print(f"[probe] run {i + 1}: mean_rel_L1_vs_ref={rel:.4e}  "
              f"mean_rel_L1_vs_run1={drift:.4e}  bit_identical_to_run1={exact}")

    drifts = [
        np.abs(o - outputs[0]).mean() / np.abs(outputs[0]).mean() for o in outputs[1:]
    ]
    worst = max(drifts) if drifts else 0.0
    stable = all(np.array_equal(o, outputs[0]) for o in outputs[1:])
    print(f"[probe] RESULT {args.arm}: stable_across_runs={stable} "
          f"worst_drift_vs_run1={worst:.4e} runs={args.runs}")
    # A drift at the 1e-1 scale is the documented corruption; 0 is a clean reuse.
    return 0 if stable else 1


if __name__ == "__main__":
    sys.exit(main())
