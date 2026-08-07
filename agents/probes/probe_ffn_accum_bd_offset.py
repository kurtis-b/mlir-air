#!/usr/bin/env python3
"""The J7b wall: a per-iteration L2 read offset is DROPPED past the unroll limit.

WHAT THIS PROBE SHOWS
    Dumps the memtile MM2S BD chain that ``builders/ffn_accum.py`` produces, at
    whatever K-step counts you ask for. Every ``aie.dma_bd`` offset in the chain
    is printed in chain order. For the CURRENT builder every offset is 0 and that
    is correct: both operands advance on the L3 side, where the shim streams
    contiguously, so no BD ever needs a moving offset.

WHAT IT IS FOR
    The design this replaced staged A WHOLE in the memtile and put each K step's
    slice from an advancing offset -- the obvious economy, since one contiguous
    L3 copy then serves every step. That construction compiles, places, keeps
    zero packet-typed channels and hoists 4 -> 2 exactly like this one, and it is
    WRONG past two K steps. To see it, run this probe against the builder as it
    stood at commit e6cdd138:

        git show e6cdd138:programming_examples/transformer_layer/builders/ffn_accum.py

    and the chain reads (herd_x=1, tile_k=32, block 2048):

        ksteps=2   [0, 0, 2048, 0]   <- A's two BDs walk: 0, then 2048
        ksteps=4   [0, 0, 0,    0]   <- A's two BDs BOTH read 0

    At 2 trips the K loop fully unrolls, so each put carries its own literal
    offset and the 4-BD chain IS the whole computation. At 4 and beyond the loop
    stops unrolling, the chain becomes a cycle covering two steps, and the
    advancing offset is simply gone -- ``aie.dma_bd`` offsets are static and
    nothing rewrites the chain into a form that can walk.

    On hardware the core then stalls, the accumulator store never fires, and the
    output is returned byte-identical to what the host wrote: seed y with 1.0 and
    4096/4096 elements come back 1.0. At the spec shape (4 columns x 96 steps) it
    does not return at all -- ERT_CMD_STATE_TIMEOUT, which is how the phase's
    implement session first met it.

    Not ping-pong: ``--omit-pingpong all`` reproduces it identically.

    This is doc 16's H5 -- compile-time-only channel/BD indexing -- with a worse
    failure mode than H5 records. H5 says such a loop fully unrolls and exhausts
    locks, which at least fails loudly at compile time. Here it declines to
    unroll and silently emits a chain that repeats a stale offset forever.

ROOT CAUSE, located after this probe was written
    `mlir/lib/Conversion/AIRToAIESchedulingUtils.cpp`, `air::get1DOffset`:

        auto offset = mlir::getConstantIntValue(memcpy_offsets[i]);
        one_d_offset += *offset;        // no has_value() check

    `getConstantIntValue` returns nullopt exactly when the value is NOT a
    compile-time constant -- an induction variable, here. Dereferencing that is
    UB; the observed result is a silent 0. The BD-dim-layout construction at
    ~462 has the same unchecked deref, while the same file checks correctly at
    527 and 945. The only caller is AIRToAIEPass.cpp:6527.

    Confirmed in the pre-air-to-aie dump: the put reads `%arg4[%7]` where
    `%7 = affine.apply #map()[%arg6]` over the K loop's IV, while the
    accumulator fetch beside it is all literals -- which is why the C BDs were
    right and the A BD was not.

    Specced as phase H10, docs/plans/.../24-phase-h10-non-constant-bd-offsets.md.

NOT a test. Nothing runs it in CI.
"""

import argparse, glob, os, re, sys, tempfile

_PE = "/home/cj/mlir-air/programming_examples"
for _p in (_PE, os.path.join(_PE, "llms"), os.path.join(_PE, "transformer_layer")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from builders.ffn_accum import build_ffn_accum_module, compile_ffn_accum_kernel
from air.backend.xrt import XRTBackend


def memtile_bd_offsets(work):
    """Every dma_bd offset in the memtile MM2S chain of the last dump, in order."""
    dumps = sorted(glob.glob(os.path.join(work, "air_project/debug_ir/*.mlir")))
    if not dumps:
        return None
    src = open(dumps[-1]).read()
    m = re.search(r"aie\.memtile_dma\([^)]*\) \{(.*?)\n      aie\.end", src, re.S)
    if not m:
        return []
    return [int(x) for x in re.findall(r"aie\.dma_bd\([^)]*offset = (\d+)", m.group(1))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--emb", type=int, default=64)
    ap.add_argument("--herd-x", type=int, default=1)
    ap.add_argument("--tile-k", type=int, default=32)
    ap.add_argument("--ksteps", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--omit-pingpong", default="", choices=["", "L1", "L2", "all"])
    args = ap.parse_args()

    tile_n = args.emb // args.herd_x
    print(
        f"seq={args.seq} emb={args.emb} herd_x={args.herd_x} "
        f"tile_n={tile_n} tile_k={args.tile_k}"
        + (f"  omit_pingpong={args.omit_pingpong}" if args.omit_pingpong else "")
    )
    print(
        "A feed block = TILE_M*tile_k = 64*%d = %d elements\n"
        % (args.tile_k, 64 * args.tile_k)
    )

    for ksteps in args.ksteps:
        ffn_dim = ksteps * args.tile_k
        work = tempfile.mkdtemp(prefix=f"bdoff-k{ksteps}-")
        os.chdir(work)
        compile_ffn_accum_kernel(tile_k=args.tile_k, tile_n=tile_n)
        mod = build_ffn_accum_module(
            args.seq, ffn_dim, args.emb, herd_x=args.herd_x, tile_k=args.tile_k
        )
        be = XRTBackend(
            verbose=False,
            omit_while_true_loop=False,
            output_format="xclbin",
            instance_name="bdoff",
            debug_ir=True,
            omit_pingpong=args.omit_pingpong,
        )
        try:
            be.compile(mod)
        except Exception as e:
            print(f"  (compile raised {type(e).__name__}; dumps still written)")
        finally:
            try:
                be.unload()
            except Exception:
                pass
        offs = memtile_bd_offsets(work)
        print(f"ksteps={ksteps:3d} (ffn_dim={ffn_dim:5d})  memtile BD offsets: {offs}")
        if offs and len(set(offs)) == 1 and offs[0] == 0:
            print(
                "            all zero -- correct for the CURRENT builder: both "
                "operands advance on the L3 side"
            )
        else:
            print(
                "            offsets vary -- some BD is being read at a moving "
                "offset; check it walks for EVERY trip, not just the unrolled ones"
            )
        print()


if __name__ == "__main__":
    main()
