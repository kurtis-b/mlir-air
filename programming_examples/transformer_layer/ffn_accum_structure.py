# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The J7b structural check: the ring forms in the REAL pipeline, and stays formed.

CONTRACT
    Compiles ``build_ffn_accum_module`` at EVERY shape the catalogue claims
    for ``ffn_accum`` (read from ``opcheck_specs.SPECS``) through
    ``XRTBackend(debug_ir=True)`` -- the production aircc binary, with the
    same backend options the numeric arm's runner passes -- and reads the
    per-pass dumps aircc writes to ``air_project/debug_ir/``. A hand-built
    air-opt pass list is NOT evidence here: it answers "what does this pass
    do", not "what does the toolchain do", and the two diverged before
    (phase J7a; docs/plans/.../22 "at which altitude").

    Four checks, all read from the dumps:

    1. THE HOIST. In the dump written after
       ``air-hoist-dma-in-accum-pattern`` the K loop must carry 2
       data-movement ops against 4 in the dump immediately preceding it:
       the C fetch/store pair left, the A|B feed gets stayed. That 4 -> 2
       IS the phase: a loop that keeps its C round trip computes identical
       numbers one DDR trip per K step slower, invisible to every
       numerical arm.
    2. DISPATCH. ``ffn_matmul_bf16_bf16_up_proj`` -- the in-place
       accumulating entry point, named literally so builder drift cannot
       satisfy this by exporting something else -- must be called INSIDE
       that K loop. Three accumulate-into-C kernels have been exported
       since phase A and never dispatched; a numerically equivalent
       replacement would pass clause 1 and every numeric check while
       violating the phase. The K loop must ALSO dispatch the device-side
       zero ``ffn_zero_bf16_up_proj`` under an ``scf.if`` (the first-K-step
       guard): that call is what makes a stale, reused output BO safe, and
       no numeric arm can see it missing because ``XRTRunner.run_test``
       zero-fills output placeholders (review round 2).
    3. PERSISTENCE. In the LAST dump still carrying the herd (the end of
       the AIR-dialect pipeline, just before air-to-aie) the K loop must
       still hold exactly the two feed gets plus the accumulate call and
       the guarded zero, with one accumulator get and one put hoisted
       outside it. A pass running after the hoist that re-sinks or
       reintroduces the C round trip -- or drops the zero -- fails HERE
       even though clause 1 still reads 4 -> 2.
    4. ZERO PACKET-TYPED CHANNELS in EVERY dump, J7a's column rule. Three
       per-core input streams packet-multiplex EVERY input (measured; see
       the builder), and the packet path is silently wrong past one trip
       pre-H9 and BD-starved post-H9.

    Liveness, so no count can pass vacuously: the hoist dump must exist,
    the ``air-dma-to-channel`` dump must exist with no residual
    ``air.dma_memcpy_nd``, and it must still declare the A|B feed channel
    by name.

FOOTGUNS
    - The compile FAILS at the core-ELF link, deliberately: the scratch
      dir holds no kernel object. Every dump this file reads is written
      before aiecc runs, in under a second, with no NPU and no Peano. Do
      not "fix" the failure by compiling the kernel first; it only slows
      the check down.
    - The K-loop count for clause 1 reads the FIRST ``scf.for`` in the
      module -- the convention ``agents/probes/probe_accum_hoist.py`` and
      the driver's own clause-3 reading of the same dump share. The
      builder emits the feed loop after the herd precisely so that first
      loop IS the K loop.
"""

import glob
import os
import re
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJ_ROOT = _HERE.parent  # programming_examples/
for _p in (str(_PROJ_ROOT), str(_PROJ_ROOT / "llms"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from air.backend.xrt import XRTBackend  # noqa: E402

from builders.ffn_accum import (  # noqa: E402
    CHANNEL_FEED,
    FFN_ACCUM_HERD_X,
    FFN_ACCUM_MM_SYMBOL,
    FFN_ACCUM_TILE_K,
    build_ffn_accum_module,
)
from opcheck_specs import SPECS  # noqa: E402

# Spec clause 2 names this entry point. Stating it literally means a builder
# that drifts to some other symbol fails here, rather than quietly redefining
# what "the in-place kernel" means; the cross-check against the builder's
# exported constant catches the drift at its source.
REQUIRED_MM_SYMBOL = "ffn_matmul_bf16_bf16_up_proj"

# The device-side zero, guarded to the K loop's first iteration. Stated
# literally for the same reason as the symbol above: a builder that stops
# dispatching it reverts to relying on y entering zeroed, which every numeric
# arm conceals (XRTRunner zero-fills output placeholders).
REQUIRED_ZERO_SYMBOL = "ffn_zero_bf16_up_proj"

# The herd op as the dumps print it; brace-matched to scope clause 3.
_HERD_MARKER = "air.herd @ffn_accum_herd"

_HOIST_DUMP_SUFFIX = "_after_air-hoist-dma-in-accum-pattern.mlir"
_CHANNEL_DUMP_SUFFIX = "_after_air-dma-to-channel.mlir"

_MOVEMENT = re.compile(r"air\.dma_memcpy_nd|air\.channel\.(?:put|get)")


def _aircc_debug_dumps(module):
    """The production pipeline's per-pass dumps for ``module``.

    Compiles in a scratch dir through the same backend configuration the
    numeric arm's runner uses (opcheck.py plus ``_GEMM_RUNNER_KWARGS``:
    elf output, runtime loop tiling [2, 2]), with ``debug_ir=True`` and an
    explicit ``target_device`` so no xrt-smi query runs. Returns the
    ordered ``(dump name, dump text)`` list and the compile error string
    (the expected core-ELF link failure, or worse -- the caller decides by
    checking which dumps exist).
    """
    prev_cwd = os.getcwd()
    error = None
    with tempfile.TemporaryDirectory(prefix="ffn-accum-structure-") as work:
        os.chdir(work)
        try:
            backend = XRTBackend(
                omit_while_true_loop=False,
                output_format="elf",
                instance_name="ffn_accum",
                runtime_loop_tiling_sizes=[2, 2],
                target_device="npu2",
                debug_ir=True,
            )
            try:
                backend.compile(module)
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc)[-200:]}"
            finally:
                try:
                    backend.unload()
                except Exception:
                    pass
            dumps = [
                (os.path.basename(p), Path(p).read_text())
                for p in sorted(glob.glob("air_project/debug_ir/pass_*.mlir"))
            ]
        finally:
            os.chdir(prev_cwd)
    return dumps, error


def _braced_block(text, marker):
    """Lines from the first line containing ``marker`` through its close."""
    if text is None:
        return None
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if marker in line), None)
    if start is None:
        return None
    depth, body = 0, []
    for line in lines[start:]:
        body.append(line)
        depth += line.count("{") - line.count("}")
        if depth <= 0 and len(body) > 1:
            break
    return "\n".join(body)


def _first_loop(text):
    """The first ``scf.for`` body -- probe_accum_hoist's counting scope."""
    return _braced_block(text, "scf.for")


def _count_movement(text):
    return len(_MOVEMENT.findall(text or ""))


def check_shape(shape_key, seq_len, ffn_dim, emb_dim):
    """One shape's verdict. Returns the list of problems, empty on pass."""
    label = f"ffn_accum [{shape_key}]"
    module = build_ffn_accum_module(
        seq_len, ffn_dim, emb_dim, herd_x=FFN_ACCUM_HERD_X, tile_k=FFN_ACCUM_TILE_K
    )
    dumps, compile_error = _aircc_debug_dumps(module)

    names = [name for name, _ in dumps]
    hoist_idx = next(
        (i for i, n in enumerate(names) if n.endswith(_HOIST_DUMP_SUFFIX)), None
    )
    if not hoist_idx:  # missing, or first (nothing before it to diff against)
        return [
            f"{label}: aircc wrote {len(dumps)} debug dumps and none usable "
            "for air-hoist-dma-in-accum-pattern -- the compile stopped before "
            "the hoist pass or the pipeline changed; nothing structural is "
            f"measured ({compile_error or 'compile reported success'})"
        ]

    problems = []
    before = _count_movement(_first_loop(dumps[hoist_idx - 1][1]))
    after_loop = _first_loop(dumps[hoist_idx][1])
    after = _count_movement(after_loop)
    if before != 4 or after != 2:
        problems.append(
            f"{label}: K-loop data movement {before} -> {after} across "
            f"{names[hoist_idx]}, need 4 -> 2 -- the accumulator round trip "
            "did not hoist; the ring is not forming and every numerical check "
            "will still pass"
        )

    call_marker = f"call @{REQUIRED_MM_SYMBOL}("
    if FFN_ACCUM_MM_SYMBOL != REQUIRED_MM_SYMBOL:
        problems.append(
            f"{label}: the builder's entry point is {FFN_ACCUM_MM_SYMBOL!r}, "
            f"not the in-place accumulating {REQUIRED_MM_SYMBOL!r} the phase "
            "requires -- only the one-memref read-add-write kernel can match "
            "areSymmetricDmaOps"
        )
    if call_marker not in (after_loop or ""):
        problems.append(
            f"{label}: no call @{REQUIRED_MM_SYMBOL} inside the K loop of "
            f"{names[hoist_idx]} -- the in-place accumulating entry point is "
            "not what the loop dispatches, so the hoist count above measures "
            "some other computation"
        )
    zero_marker = f"call @{REQUIRED_ZERO_SYMBOL}("
    if zero_marker not in (after_loop or "") or "scf.if" not in (after_loop or ""):
        problems.append(
            f"{label}: no guarded call @{REQUIRED_ZERO_SYMBOL} inside the K "
            f"loop of {names[hoist_idx]} -- without the first-iteration "
            "device-side zero the design relies on y entering zeroed, which "
            "XRTRunner's zero-filled placeholders conceal: a reused stale BO "
            "returns y_old + a @ w and every numeric arm still passes"
        )

    last_air = next(
        ((name, text) for name, text in reversed(dumps) if _HERD_MARKER in text),
        None,
    )
    if last_air is None:
        problems.append(
            f"{label}: no dump carries {_HERD_MARKER} -- the herd vanished "
            "before air-to-aie, so persistence of the hoist is unmeasured"
        )
    else:
        herd_body = _braced_block(last_air[1], _HERD_MARKER)
        loop_body = _first_loop(herd_body)
        in_loop = _count_movement(loop_body)
        feed_gets = len(
            re.findall(
                r"air\.channel\.get[^\n]*@" + re.escape(CHANNEL_FEED), loop_body or ""
            )
        )
        outside = (herd_body or "").replace(loop_body or "", "")
        out_gets = len(re.findall(r"air\.channel\.get", outside))
        out_puts = len(re.findall(r"air\.channel\.put", outside))
        if (
            in_loop != 2
            or feed_gets != 2
            or out_gets != 1
            or out_puts != 1
            or call_marker not in (loop_body or "")
            or zero_marker not in (loop_body or "")
        ):
            problems.append(
                f"{label}: at {last_air[0]} (the AIR pipeline's last herd "
                f"dump) the K loop holds {in_loop} data-movement ops "
                f"({feed_gets} feed gets) with {out_gets} get / {out_puts} "
                "put hoisted outside; need exactly 2 feed gets plus the "
                "accumulate call and the guarded zero inside and 1 get + 1 "
                "put outside -- a pass after the hoist re-sank or "
                "reintroduced accumulator traffic, or dropped the zero"
            )

    channel_dump = next(
        ((name, text) for name, text in dumps if name.endswith(_CHANNEL_DUMP_SUFFIX)),
        None,
    )
    if channel_dump is None:
        problems.append(
            f"{label}: no air-dma-to-channel dump -- the channel lowering did "
            "not run, so the packet count proves nothing"
        )
    else:
        n_residual = channel_dump[1].count("air.dma_memcpy_nd")
        if n_residual:
            problems.append(
                f"{label}: {n_residual} air.dma_memcpy_nd left after "
                "air-dma-to-channel -- unconverted data movement; the packet "
                "count proves nothing"
            )
        if f"@{CHANNEL_FEED}" not in channel_dump[1]:
            problems.append(
                f"{label}: feed channel @{CHANNEL_FEED} missing from the "
                "channel-lowered dump -- this is not the A|B-fed accumulator "
                "design"
            )

    n_packet = sum(text.count("npu_dma_packet") for _, text in dumps)
    if n_packet:
        problems.append(
            f"{label}: {n_packet} packet-typed channel references across the "
            "aircc dumps -- a third L3-facing stream is back in the "
            "per-column budget; see builders/ffn_accum.py"
        )

    verdict = "FAIL" if problems else "PASS"
    print(
        f"[ffn-accum-structure] {label}: {verdict} (K-loop {before} -> {after}, "
        f"{n_packet} packet-typed channels)"
    )
    return problems


def main():
    shapes = [s for s in SPECS if s["operator"] == "ffn_accum"]
    if not shapes:
        print("[ffn-accum-structure] FAIL: no ffn_accum shapes in SPECS")
        return 1

    problems = []
    for spec in shapes:
        shape = spec["shape"]
        problems += check_shape(
            spec["shape_key"], shape["seq_len"], shape["ffn_dim"], shape["emb_dim"]
        )

    if problems:
        print("[ffn-accum-structure] FAIL")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print("[ffn-accum-structure] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
