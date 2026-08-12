# Does an off-preset ELF actually RUN? `runtime_loop_tiling_sizes` is not inert.
#
# WHAT THIS ESTABLISHED `[2026-08-08]`, as a replicated 2x2 factorial on real NPU2 --
# `mha_out_proj` at 4096x768, twelve heads, non-causal:
#
#   tiling   ping-pong   n     result
#   [1,1]    OFF         2     PASS   0 / 3,145,728 mismatches, mean_rel_L1 5.3348e-02,
#   [1,1]    ON          1     PASS   atol_required 8.7061e-03 vs atol 2.5e-02 (2.87x).
#                                     All three runs are byte-identical on every statistic.
#   [2,2]    ON          2     ERT_CMD_STATE_TIMEOUT
#   [2,2]    OFF         1     ERT_CMD_STATE_TIMEOUT
#
# So the discriminating variable is `runtime_loop_tiling_sizes`, and `omit_pingpong` is
# IRRELEVANT at this shape -- ping-pong ON passes with statistics identical to ping-pong
# OFF, and [2,2] hangs either way.
#
# WHY THAT MATTERS. Doc 26 §4 concluded `runtime_loop_tiling_sizes` is "inert", from a
# compile-only spike: aircc's lowered `aie.air.mlir` is IDENTICAL between [1,1] and [2,2]
# (280 aie.dma_bd / 44 shim_dma_allocation / 628 aie.buffer / 424 aie.lock either way,
# with a raw diff of 98 lines of channel renumbering and nothing else), because
# `air-opt-shim-dma-bds` early-exits when there is no shim-level scf.for to tile. That
# observation is correct AND the conclusion drawn from it is wrong. The knob is inert in
# the IR that was diffed and it is decisive on hardware. Doc 26 flagged the caveat itself
# -- "compile-only refutes 'a placement failure at best', it does not refute 'wrong
# numbers at worst'" -- and this is that caveat coming due, with a third outcome neither
# branch predicted: it neither places badly nor computes wrongly, it hangs.
#
# CONSEQUENCE: the backend-settings conflict that `pattern/fused/fused.py`,
# `builders/mha_out_proj.py` and `builders/block.py` document is REAL. FlashAttention at
# 4096 requires [1,1], the wide GEMMs are built at [2,2], one ELF is one aircc
# invocation. Only the stated REASON needed correcting: it is the tiling sizes, not
# `omit_pingpong`, and it is now measured rather than asserted.
#
# WHAT MAKES THIS A TEST AND NOT A DEMO: it changes ONE thing. The operator, the shape,
# the golden reference, the rtol/atol and the element-wise zero-mismatch rule all come
# from the shipped `mha_out_proj [4096x768x12h]` SPECS row unmodified -- the same row the
# transformer-layer suite gates on today. Only `runner_kwargs` is swapped. The shipped
# preset re-run through this same harness is the control, and it PASSES, which is what
# makes the timeouts a property of the preset rather than of the probe.
#
# Each arm must run in its OWN PROCESS on an exclusive device -- submit through
# `agents/scripts/devq.sh submit --class measure`. Doc 23 records a run where the same
# mode and shape passed alone and failed as a later rung of a shared process; one process
# per measurement is what excludes that explanation here.
#
# RE-MEASURED `[2026-08-12]`, queue item 24. THE READING ABOVE SURVIVES; the MECHANISM
# PARAGRAPH ABOVE DOES NOT. Two things made the 3/3 worth re-running: wall 7 showed this
# class of composition hanging NONDETERMINISTICALLY (one ELF, identical inputs, fresh
# processes, PASS/TIMEOUT/PASS/TIMEOUT/PASS), which makes any 3/3 weak; and for R1 the two
# tiling settings are byte-identical through `.pdi`, i.e. provably inert THERE.
#
#   ARTIFACTS, at this row's own shape, `attn` against `t22pp` (tiling is the ONLY
#   difference), two independent compiles per arm -- devq 279-282. The knob is NOT inert
#   here. Byte-stable within an arm and different across arms:
#       npu_insts .../at_attn_seg_sequence.bin   402,448 -> 356,368 B   (-11.4%)
#       npu_insts .../op_matmul_seg_sequence.bin  88,080 ->  29,712 B   (2.96x smaller)
#       npu_insts .../main_mha_out_proj.bin     1,026,496 -> 922,048 B  (-10.2%)
#       at_attn_seg.pdi                          differs, same length
#   and the same three numbers are the ELF's `.ctrltext.{0,1,2}` section sizes.
#
#   AND THE CONTROL THAT MATTERS MORE: `aie.air.mlir` IS NOT BYTE-REPRODUCIBLE. Two
#   compiles of the SAME preset differ by 94 lines (`attn`) and 98 lines (`t22pp`), the
#   same order as the 98-line `[1,1]`-vs-`[2,2]` diff the paragraph above calls "channel
#   renumbering and nothing else". That diff was COMPILER NOISE, not the knob, so the
#   compile-only spike was reading a difference it could not have attributed either way.
#   `placed.air.mlir` is unstable the same way; `npu.air.mlir` is unstable in bytes but
#   STABLE in length (1,507,775 at [1,1] vs 1,306,762 at [2,2]). Settle inertness on the
#   `.bin`/`.pdi`/`.ctrltext` artifacts, never on an IR dump.
#
#   HARDWARE, 5 fresh processes per arm, interleaved -- devq 283-292:
#       attn  [1,1] pp OFF   PASS  PASS  PASS  PASS  PASS      5/5
#       t22pp [2,2] pp OFF   TIME  TIME  TIME  TIME  TIME      0/5, every one
#                                                              ERT_CMD_STATE_TIMEOUT,
#                                                              ctx_pc 0x28B060AD
#   No wrong answers and no compile failures in either arm. All five passes returned
#   BYTE-IDENTICAL statistics (0 / 3,145,728, mean_rel_L1 5.3348e-02, atol_required
#   8.7061e-03) -- and identical to the 2026-08-08 runs. Pooled with those: [1,1] 8/8
#   PASS, [2,2] 0/8. Fisher exact on the 10 new runs alone p = 0.0079; pooled p = 1.6e-4.
#
# So this is NOT wall 7's coin flip. Wall 7's signature is a MIXED arm plus, at 128, four
# DISTINCT wrong answers; here neither arm is mixed and the passing arm reproduces one
# answer to the bit. `runtime_loop_tiling_sizes` is load-bearing at this shape, and the
# reason is now visible in the artifacts instead of inferred from a hang.
#
# THE MECHANISM, corrected. `runtime_loop_tiling_sizes` -> aircc
# `--air-runtime-loop-tiling-sizes` -> `air-opt-shim-dma-bds` as `shim-dma-tile-sizes`.
# That pass FIRST lowers `air.launch` to `scf.for` (`AIRLaunchToScfForPattern`,
# `AIRDependencyScheduleOpt.cpp:8275`) and only THEN collects the shim-level `scf.for`
# band; the "no shim `scf.for`" early-exit is at `:8287-8291` and is therefore a statement
# about the IR AFTER that lowering, not about the design as written. `mha_out_proj` @4096 has a 2-D launch grid plus the projection launch, so the
# band is non-empty, the pass tiles it by `findLargestFactor(trip_count, 2)`, and the BD
# program shrinks -- most visibly 2.96x on the GEMM segment. The early-exit is real; it
# simply does not fire for THIS design. R1 is a design where it does fire, which is why
# the identical knob is inert there and decisive here. Do not carry either result across
# to another design without diffing that design's own `.bin`/`.pdi` artifacts first.
#
# Usage:  python3 probe_backend_preset_hardware.py [attn|gemm|t11nopp|t22pp] [--compile-only]
#           attn    = [1,1] pp OFF -- the SHIPPED preset, the control
#           gemm    = [2,2] pp ON  -- the GEMM preset
#           t11nopp = [1,1] pp ON  -- isolates ping-pong
#           t22pp   = [2,2] pp OFF -- isolates tiling
#
#         `--compile-only` runs aircc and prints a digest of every artifact aircc wrote
#         (`aie.air.mlir`, `npu.air.mlir`, `placed.air.mlir`, each `.pdi`, each
#         `npu_insts*.bin`, and the ELF's `.ctrltext`/`.pdi` section sizes) instead of
#         touching the device. It exists so the inertness question can be settled for a
#         NEW design without spending device time, and so `--class build` is the right
#         devq class for it. Run each arm in its own empty cwd and diff the two digests.
#
# THERE IS DELIBERATELY NO REPEAT COUNT. Repeats come from N separate devq jobs, because
# doc 23's rule is one process per device measurement and a loop inside this process would
# be exactly the structure that rule exists to forbid.
import os
import sys

_TL = "/home/cj/mlir-air/programming_examples/transformer_layer"
_PROJ = "/home/cj/mlir-air/programming_examples"
for _p in (_PROJ, os.path.join(_PROJ, "llms"), _TL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_ARGV = [a for a in sys.argv[1:] if a != "--compile-only"]
COMPILE_ONLY = "--compile-only" in sys.argv[1:]
PRESET = _ARGV[0] if _ARGV else "gemm"

# The two presets, verbatim from spike C's probe.py so this run is comparable to the
# compile-only record it is completing.
PRESETS = {
    # What pattern/offload and the 4096-row GEMM modules use.
    "gemm": {
        "runtime_loop_tiling_sizes": [2, 2],
        "output_format": "elf",
        "omit_while_true_loop": False,
    },
    # What mha_out_proj ships with today (MHA_OUT_PROJ_RUNNER_KWARGS).
    "attn": {
        "omit_pingpong": "all",
        "runtime_loop_tiling_sizes": [1, 1],
        "output_format": "elf",
        "omit_while_true_loop": False,
    },
    # The two one-variable-at-a-time arms. `gemm` differs from `attn` in BOTH
    # knobs at once, so it cannot say which one matters; these split it.
    #   t11nopp = attn's tiling, gemm's ping-pong  -> isolates PING-PONG
    #   t22pp   = gemm's tiling, attn's ping-pong  -> isolates TILING
    "t11nopp": {
        "runtime_loop_tiling_sizes": [1, 1],
        "output_format": "elf",
        "omit_while_true_loop": False,
    },
    "t22pp": {
        "omit_pingpong": "all",
        "runtime_loop_tiling_sizes": [2, 2],
        "output_format": "elf",
        "omit_while_true_loop": False,
    },
}

import opcheck  # noqa: E402
import opcheck_specs  # noqa: E402

ROW = next(
    s
    for s in opcheck_specs.SPECS
    if s["operator"] == "mha_out_proj" and s["shape_key"] == "4096x768x12h"
)

shipped = None
try:
    from builders.mha_out_proj import MHA_OUT_PROJ_RUNNER_KWARGS

    shipped = dict(MHA_OUT_PROJ_RUNNER_KWARGS)
except Exception as exc:  # pragma: no cover - reported, not swallowed
    print(f"[probe] could not read the shipped runner kwargs: {exc}")

print(f"[probe] row      : {ROW['operator']} [{ROW['shape_key']}] {ROW['shape']}")
print(f"[probe] atol     : {ROW['atol']}  (unmodified, from the shipped row)")
print(f"[probe] shipped  : {shipped}")
print(f"[probe] using    : {PRESET} -> {PRESETS[PRESET]}")

if PRESET == "attn" and shipped is not None and shipped != PRESETS["attn"]:
    # The control is only a control if it really is the shipped configuration.
    print(f"[probe] WARNING: 'attn' preset differs from the shipped kwargs {shipped}")


def _compile_only():
    """aircc this preset and digest every artifact it wrote. No device.

    Deliberately does NOT go through `ROW["prepare"]`: the oracle for this row is a
    4096x4096x12-head FP32 attention in numpy, which costs minutes and answers nothing
    a compile-only question asks. The module built here is the same
    `build_mha_out_proj_module` call `prepare_mha_out_proj` makes, with the row's own
    shape, so the artifacts belong to the row that gates the suite.

    What is printed is a digest per artifact plus its size. Inertness is then a
    `diff` of two of these, and the honest control is a SECOND compile of the SAME
    preset -- without it, a difference between two arms cannot be told apart from a
    compiler that is not byte-reproducible.
    """
    import hashlib
    import subprocess
    import time

    from builders.mha_out_proj import (
        build_mha_out_proj_module,
        compile_mha_out_proj_kernels,
        mha_out_proj_config,
    )
    from shared.infra.cache import prepare_air_project
    from air.backend.xrt import XRTBackend

    shape = ROW["shape"]
    attn_cfg, gemm_spec, _ = mha_out_proj_config(
        shape["seq_len"],
        shape["head_dim"],
        shape["num_heads"],
        num_kv_heads=shape.get("num_kv_heads"),
        causal=shape["causal"],
    )
    compile_mha_out_proj_kernels(attn_cfg, gemm_spec)
    module = build_mha_out_proj_module(
        shape["seq_len"],
        shape["head_dim"],
        shape["num_heads"],
        num_kv_heads=shape.get("num_kv_heads"),
        causal=shape["causal"],
    )
    text = str(module)
    with open(f"module_mha_{PRESET}.mlir", "w") as fh:
        fh.write(text)
    print(
        f"[probe] module   : {len(text.splitlines())} lines "
        f"md5={hashlib.md5(text.encode()).hexdigest()[:12]}",
        flush=True,
    )

    prepare_air_project(quant="bf16")
    kwargs = dict(PRESETS[PRESET])
    kwargs["instance_name"] = ROW["operator"]
    t0 = time.time()
    artifact = XRTBackend(**kwargs).compile(
        module, output_binary_name=f"probe_mha_{PRESET}"
    )
    print(f"[probe] aircc    : {time.time() - t0:.1f}s -> {artifact.output_binary}")

    print("[probe] ARTIFACTS (md5-12  bytes  name)")
    names = sorted(os.listdir("air_project"))
    for name in names:
        path = os.path.join("air_project", name)
        if not os.path.isfile(path):
            continue
        if not (
            name.endswith(".air.mlir")
            or name.endswith(".pdi")
            or name.endswith(".bin")
        ):
            continue
        with open(path, "rb") as fh:
            blob = fh.read()
        print(
            f"[artifact] {hashlib.md5(blob).hexdigest()[:12]}  {len(blob):>9}  {name}"
        )

    elf = artifact.output_binary
    if os.path.isfile(elf):
        with open(elf, "rb") as fh:
            blob = fh.read()
        print(
            f"[artifact] {hashlib.md5(blob).hexdigest()[:12]}  {len(blob):>9}  "
            f"{os.path.basename(elf)}"
        )
        try:
            out = subprocess.run(
                ["readelf", "-S", "-W", elf], capture_output=True, text=True
            ).stdout
            for line in out.splitlines():
                if ".ctrltext" in line or ".pdi" in line or ".ctrldata" in line:
                    print(f"[section] {' '.join(line.split())}")
        except FileNotFoundError:
            print("[probe] readelf not available; section sizes skipped")
    print(f"[probe] RESULT {PRESET}: compile-only OK")


if COMPILE_ONLY:
    _compile_only()
    sys.exit(0)


def prepare_with_preset(shape):
    prepared = ROW["prepare"](shape)
    prepared["runner_kwargs"] = dict(PRESETS[PRESET])
    return prepared


spec = dict(ROW, prepare=prepare_with_preset, shape_key=f"{ROW['shape_key']}_{PRESET}")
record = opcheck.run_spec(spec)

print(
    f"[probe] RESULT {PRESET}: passed={record['passed']} "
    f"n_mismatch={record['n_mismatch']}/{record['n_elements']} "
    f"mean_rel_L1={record['mean_rel_L1']:.4e} "
    f"atol_required={record['atol_required']:.4e} "
    f"atol={record['atol']:.4e} "
    f"margin={record['atol'] / record['atol_required']:.2f}x"
)
sys.exit(0 if record["passed"] else 1)
