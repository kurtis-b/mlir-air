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
# Usage:  python3 probe_backend_preset_hardware.py [attn|gemm|t11nopp|t22pp]
#           attn    = [1,1] pp OFF -- the SHIPPED preset, the control
#           gemm    = [2,2] pp ON  -- the GEMM preset
#           t11nopp = [1,1] pp ON  -- isolates ping-pong
#           t22pp   = [2,2] pp OFF -- isolates tiling
import os
import sys

_TL = "/home/cj/mlir-air/programming_examples/transformer_layer"
_PROJ = "/home/cj/mlir-air/programming_examples"
for _p in (_PROJ, os.path.join(_PROJ, "llms"), _TL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PRESET = sys.argv[1] if len(sys.argv) > 1 else "gemm"

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
