#!/usr/bin/env python3
"""Which builders actually need packet multiplexing?

Hypothesis: the shim feed-order miscompile only reaches designs with MORE THAN
TWO L3->L1 streams per column, because a column has two shim MM2S channels and
only then does air fall back to packet-multiplexing them onto one queue.

  addnorm      x + residual + weight = 3 streams  -> packet, and it miscompiles
  layer_norm   1 in                               -> no packet
  eltwise_add  2 in                               -> no packet

If that holds, a PIPELINED norm tail sidesteps the wall by construction: split
the streams across herds so no herd needs more than two.

Compile-only through air-opt, no NPU, no kernel objects.
"""
import os
import subprocess
import sys
import tempfile

_PE = "/home/cj/mlir-air/programming_examples"
for _p in (_PE, os.path.join(_PE, "llms"), os.path.join(_PE, "transformer_layer")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ROWS, COLS, HERD_X = 4096, 768, 8

PIPELINE = ("air-dependency,air-hoist-dma-in-accum-pattern,air-broadcast-detection,"
            "air-specialize-dma-broadcast,air-dma-to-channel,canonicalize,cse")


def lower(mod, tag):
    work = tempfile.mkdtemp(prefix=f"probe-pkt-{tag}-")
    src = os.path.join(work, "in.mlir")
    out = os.path.join(work, "out.mlir")
    open(src, "w").write(str(mod))
    r = subprocess.run(["air-opt", src, f"--pass-pipeline=builtin.module({PIPELINE})",
                        "-o", out], capture_output=True, text=True)
    if r.returncode != 0:
        return None, r.stderr[:400]
    return open(out).read(), None


def report(name, mod):
    text, err = lower(mod, name)
    if text is None:
        print(f"  {name:22s} lowering FAILED: {err.splitlines()[0][:90] if err else '?'}")
        return
    packet = text.count('"npu_dma_packet"')
    chans = text.count("air.channel @")
    print(f"  {name:22s} channel decls={chans:3d}   packet-typed={packet:3d}   "
          f"-> {'PACKET MULTIPLEXED' if packet else 'no packet path'}")


def main():
    from builders.addnorm import build_addnorm_module
    from builders.layer_norm import build_layer_norm_module
    from builders.elementwise_add import build_elementwise_add_module

    print(f"shape {ROWS}x{COLS}, herd_x={HERD_X}\n")

    # addnorm at its legal one-trip cap (104 rows), the shape the block dispatches.
    print("the operator that miscompiles at >1 trip on >1 column:")
    report("addnorm (3 streams)", build_addnorm_module(104, COLS, herd_x=HERD_X, pre_add=True))

    print("\nthe streamed builders `fused` uses instead, over all 4096 rows:")
    try:
        report("layer_norm (1 stream)", build_layer_norm_module(ROWS, COLS, herd_x=HERD_X))
    except Exception as e:  # noqa: BLE001
        print(f"  layer_norm build failed: {type(e).__name__}: {str(e)[:120]}")
    try:
        report("eltwise_add (2 streams)",
               build_elementwise_add_module(ROWS, COLS, herd_x=HERD_X))
    except Exception as e:  # noqa: BLE001
        print(f"  eltwise_add build failed: {type(e).__name__}: {str(e)[:120]}")


if __name__ == "__main__":
    sys.exit(main())
