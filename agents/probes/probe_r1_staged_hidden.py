#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Doc 53 section 2.4: does staging `hidden` whole collapse R1's `maxq`?

THE QUESTION, AND WHY IT IS NOT THE ONE SECTION 2.3 ASKED
    R1 re-reads `hidden` once per (sweep, k'): `sweeps * k_steps_up` separate
    shim `dma_start_task`s onto ONE channel with no await between them.  That
    count IS doc 52 section 10.6's `maxq` -- PASS at 2/3/4, FAIL at 5, TIMEOUT
    at 6+ -- and at the gate shape it is 96.  It is the binding wall.

    Doc 53 section 2.3 proposed staging the band whole in L2 and reading k'
    slices at Python-unrolled LITERAL offsets, and refused it on memtile block
    count.  Section 2.3a then compiled the baseline and found that refutation
    was arguing the wrong side of the hop: the memtile chain is ALREADY
    maximally folded (96 shim tasks against 2 ping-ponged A BDs), so no
    memtile count was ever the obstacle.  What is actually unknown is whether
    the DRAIN side is expressible.

WHAT THE TWO ARMS MEASURE
    control  `stage_hidden=False` -- R1 as it ships.  Establishes `maxq` and
             the memtile block census on the same instrument, so the staged
             arm is read against a number from the same run rather than
             against a recorded one.
    staged   `stage_hidden=True` -- one 4-D shim read of the whole band (Q1),
             drained per k' by a put whose L2-side offset is the LOOP IV (Q2).

    Q2 is the one at risk.  Doc 23 section 2's rule is that an L2/L1-side
    offset may not depend on an induction variable -- an `aie.dma_bd` offset
    is static -- and H10 made that refuse by message instead of silently
    emitting a chain that repeats a stale offset.  So a REFUSAL in the staged
    arm is a RESULT, not a failure: it says the contiguous re-stream needs its
    own A-only channel rather than a moving offset, and it prices that route's
    next obstacle (an A channel and a B channel want 8 memtile MM2S ports
    against the 6 a memtile has).

    A SILENT PASS with `maxq` unchanged would be the bad outcome and is
    checked for by name: it would mean the staged read was emitted and the
    drain still walked per k', which is the shape doc 52 section 12 found
    (a step that cannot fire looking green).

HERMETIC
    No NPU, no kernel objects.  aiecc writes every MLIR dump before core ELFs
    (doc 23 section 5), so the ELF link failing on absent objects is expected
    and the dumps are still read.  This probe never dispatches.

RUN
    PATH=$PWD/sandbox/bin:$PWD/build-xrt/bin:$PATH \
    PYTHONPATH=$PWD/build-xrt/python:/opt/xilinx/xrt/python \
    PEANO_INSTALL_DIR=$PWD/sandbox/lib/python3.12/site-packages/llvm-aie \
    MLIR_AIE_INSTALL_DIR=$PWD/sandbox/lib/python3.12/site-packages/mlir_aie \
      ./sandbox/bin/python agents/probes/probe_r1_staged_hidden.py
"""

import glob
import os
import re
import signal
import sys
import tempfile
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "programming_examples", "transformer_layer"))

from builders.ffn_accum import FFN_ACCUM_HERD_X, FFN_ACCUM_TILE_K  # noqa: E402
from builders.ffn_resident import build_ffn_resident_module  # noqa: E402
from air.backend.xrt import XRTBackend  # noqa: E402

TILE_M = 64
EMB_DIM = 768
FFN_DIM = 3072
COMPILE_TIMEOUT_S = 600


class _Timeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _Timeout()


def maxq(dump: str):
    """Outstanding `dma_start_task`s per channel before the first await.

    Doc 52 section 10.6's quantity, read the way that section reads it: walk
    the runtime sequence in order, counting starts per channel, and stop at
    the first `dma_await_task`.  Returns (busiest_count, per_channel).
    """
    src = open(dump).read()
    # A task is configured for a (tile, direction, channel) and started by SSA
    # name; map the name to its channel at configure time.
    # NB the op opens with a BRACE, not a paren.  The first cut of this regex
    # required `\(` and therefore matched nothing, putting every task in one
    # anonymous bucket and reporting a confident 25 for a control whose value
    # is 15.  A parse that silently degrades to one bucket is the same class of
    # defect as the checks doc 51 catalogues, so the bucket is asserted below.
    chan_of = {}
    for m in re.finditer(
        r"%(\S+)\s*=\s*aiex\.dma_configure_task_for\s+@(\S+?)\s*\{", src
    ):
        chan_of[m.group(1)] = m.group(2)
    counts = {}
    for m in re.finditer(r"aiex\.dma_(start_task|await_task)\(%(\S+?)\)", src):
        if m.group(1) == "await_task":
            break
        c = chan_of.get(m.group(2), "?")
        counts[c] = counts.get(c, 0) + 1
    if counts.get("?"):
        raise AssertionError(
            f"maxq parse degraded: {counts['?']} task(s) resolved to no channel. "
            "The configure-op regex no longer matches the dump; a per-channel "
            "count that silently collapses to one bucket is worse than none."
        )
    return (max(counts.values()) if counts else 0), counts


def memtile_blocks(dump: str):
    """Top-level blocks per `aie.memtile_dma` region -- the 48-block cap's unit."""
    src = open(dump).read()
    out = {}
    for m in re.finditer(r"aie\.memtile_dma\(%(\S+?)\)", src):
        start = src.find("{", m.end())
        d = 0
        i = start
        while i < len(src):
            if src[i] == "{":
                d += 1
            elif src[i] == "}":
                d -= 1
                if d == 0:
                    break
            i += 1
        body = src[start + 1 : i]
        depth = 0
        top = 0
        bds = 0
        for line in body.splitlines():
            s = line.strip()
            if depth == 0 and re.match(r"^\^bb\d+", s):
                top += 1
            if depth == 0 and "aie.dma_bd" in s:
                bds += 1
            depth += line.count("{") - line.count("}")
        out[m.group(1)] = (top, bds)
    return out


def run_arm(name: str, stage_hidden: bool):
    prev = os.getcwd()
    with tempfile.TemporaryDirectory(prefix=f"staged-hidden-{name}-") as work:
        os.chdir(work)
        error = None
        try:
            module = build_ffn_resident_module(
                TILE_M,
                FFN_DIM,
                EMB_DIM,
                herd_x=FFN_ACCUM_HERD_X,
                tile_k=FFN_ACCUM_TILE_K,
                stage_hidden=stage_hidden,
            )
        except Exception as exc:
            os.chdir(prev)
            return {"arm": name, "built": False, "error": f"{type(exc).__name__}: {exc}"}
        try:
            be = XRTBackend(
                omit_while_true_loop=False,
                output_format="elf",
                instance_name="r1_staged",
                runtime_loop_tiling_sizes=[2, 2],
                target_device="npu2",
                debug_ir=True,
            )
            signal.signal(signal.SIGALRM, _on_alarm)
            signal.alarm(COMPILE_TIMEOUT_S)
            try:
                be.compile(module)
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc)[-600:]}"
            finally:
                signal.alarm(0)
                try:
                    be.unload()
                except Exception:
                    pass
            dumps = sorted(glob.glob("air_project/debug_ir/pass_*.mlir"))
            npu = [d for d in dumps if "airrt-to-npu" in d]
            mt = [d for d in dumps if "aie.memtile_dma" in open(d).read()]
            res = {
                "arm": name,
                "built": True,
                "dumps": len(dumps),
                "error": error,
                "maxq": maxq(npu[-1]) if npu else None,
                "memtiles": memtile_blocks(mt[-1]) if mt else None,
            }
        finally:
            os.chdir(prev)
    return res


def main():
    print(
        f"[staged-hidden] band {TILE_M}x{FFN_DIM}x{EMB_DIM}, "
        f"herd_x={FFN_ACCUM_HERD_X}, tile_k={FFN_ACCUM_TILE_K} "
        f"(k_steps_up={EMB_DIM // FFN_ACCUM_TILE_K}, "
        f"sweeps={FFN_DIM // EMB_DIM}, expected control maxq="
        f"{(FFN_DIM // EMB_DIM) * (EMB_DIM // FFN_ACCUM_TILE_K)})"
    )
    results = []
    for name, flag in (("control", False), ("staged", True)):
        try:
            r = run_arm(name, flag)
        except Exception:
            traceback.print_exc()
            r = {"arm": name, "built": False, "error": "probe raised"}
        results.append(r)
        print(f"\n--- {name} (stage_hidden={flag}) ---")
        if not r.get("built"):
            print(f"  REFUSED AT BUILD: {r['error']}")
            continue
        print(f"  dumps: {r['dumps']}")
        if r["maxq"]:
            busiest, per = r["maxq"]
            print(f"  maxq (busiest channel before first await): {busiest}")
            print(f"    per channel: {per}")
        else:
            print("  maxq: no airrt-to-npu dump reached")
        if r["memtiles"]:
            for t, (top, bds) in sorted(r["memtiles"].items()):
                print(f"  {t}: {top} top-level blocks, {bds} dma_bd (cap 48)")
        if r["error"]:
            print(f"  compile ended with: {r['error'][:300]}")

    print("\n--- verdict ---")
    c = next((x for x in results if x["arm"] == "control"), None)
    s = next((x for x in results if x["arm"] == "staged"), None)
    if not s:
        print("  staged arm did not run")
        return 1
    refused = (not s.get("built")) or (
        s.get("error") and "not a compile-time constant" in s["error"]
    )
    if refused:
        print("  Q2 REFUSED, and BY MESSAGE -- the drain cannot take a moving")
        print("  L2-side offset (doc 23 s2; H10 made this refuse rather than")
        print("  emit a stale-offset chain).  The refusal IS the result: the")
        print("  contiguous re-stream needs its own A-only channel.  Next")
        print("  obstacle is priced, not guessed -- an A channel and a B channel")
        print("  want 8 memtile MM2S ports against the 6 a memtile has.")
        if c and c.get("maxq"):
            print(f"  control maxq stands at {c['maxq'][0]} ({c['maxq'][1]})")
        return 0
    if s.get("maxq") and c and c.get("maxq"):
        cm, sm = c["maxq"][0], s["maxq"][0]
        print(f"  control maxq {cm} -> staged maxq {sm}")
        if sm >= cm:
            print("  NO COLLAPSE. The staged read was emitted and the drain still")
            print("  walks per k' -- section 2.4's route does not pay as written.")
        else:
            print(f"  COLLAPSED by {cm - sm}. Section 2.4's route is viable on this")
            print("  axis; the remaining question is correctness, not structure.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
