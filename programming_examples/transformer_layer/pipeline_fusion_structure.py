# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The H8 gate: ``air-fuse-pipeline-launches`` REPRODUCES the hand-written module.

CONTRACT
    The deliverable of H8 is not "the pass emits a fused module". It is that
    the pass, given J7a's three stages as separate attributed launches,
    produces the module ``builders/norm_tail.py`` writes BY HAND -- the one
    that is gated, measured (``mean_rel_L1`` 3.620e-3) and known to route.

    This checks exactly that, at every shape the catalogue claims for
    ``norm_tail`` plus the two variants ``pattern/fused`` builds
    (``plane_major``, ``mirror_out``), and it checks it as BYTE EQUALITY of
    the printed modules rather than as a list of structural resemblances.

WHY BYTE EQUALITY, AND WHY THAT IS THE RIGHT COMPARISON
    A structural comparison has to name the things it compares -- channels,
    herds, segment count, flows, BD chains, use_lock counts, placements -- and
    is only as good as that list. Every such list this project has written has
    later turned out to miss something (doc 23 §5: a check that counted packet
    typing could not see an edge round-tripping through L3, which was the
    phase's whole claim).

    Byte equality of the printed module needs no list. It implies equality on
    every axis a structural check could name AND on every axis it could
    forget, because those axes are all functions of the module. And it implies
    the ROUTED designs are identical too, without compiling either: aircc is
    deterministic, so identical input gives identical output. The structural
    census below is therefore redundant TODAY -- it is kept because it becomes
    the load-bearing clause the moment this equality has to weaken to
    "equivalent", and because it is the clause that states the fused design
    meets the per-column shim budget rather than leaving that inferred.

    Both sides are printed by the SAME air-opt binary (the reference is
    round-tripped through it with no passes), so the comparison cannot pass or
    fail on printer differences.

WHY THE TWO ARRANGEMENTS SHARE THEIR STAGE BODIES
    ``build_norm_tail_module(stage_launches=True)`` emits the same three herd
    bodies as the default, differing only in launch/segment nesting -- see
    that builder. If the unfused form were a separate transcription, this gate
    would be comparing two pieces of Python and would pass or fail for reasons
    about them rather than about the pass.

    Verified separately and worth restating: the default arrangement's IR is
    BYTE-IDENTICAL to the pre-H8 builder's, so J7a's own gate is untouched by
    the restructure.

THE NEGATIVE CONTROL IS PART OF THE GATE, NOT A FOOTNOTE
    A fusion pass that silently declines is worse than one that does not
    exist: the stages stay in separate ``air.launch``es, each lowers to its
    own ``aie.device``, and the declared L1->L1 edges span devices. So clause
    3 requires the pass to REFUSE a malformed group -- and requires the
    refusal to be a diagnostic and a nonzero exit, not a quiet no-op.

WHAT THIS GATE DELIBERATELY DOES NOT CLAIM
    Nothing here runs on hardware, and nothing here says a composition the
    pass can newly express will work. R1 -- the one hand-built instance of
    J7a x2 + J7b -- is correct only at the fully degenerate rung and has two
    distinct hardware defects open (README queue items 21 and 23). The claim
    of this gate is bounded: the pass reproduces modules that are ALREADY
    known-good, and refuses malformed ones. Generating novel compositions is
    not claimed and is not gated.
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJ_ROOT = _HERE.parent  # programming_examples/
for _p in (str(_PROJ_ROOT), str(_PROJ_ROOT / "llms"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from builders.ffn_accum import build_ffn_accum_module  # noqa: E402
from builders.norm_tail import (  # noqa: E402
    NORM_TAIL_PIPELINE_GROUP,
    build_norm_tail_module,
)
from opcheck_specs import SPECS  # noqa: E402

PASS_FLAG = "--air-fuse-pipeline-launches"

# build_norm_tail_module's herd width, and the stage count it declares.
HERD_X = 8
N_STAGES = 3

# The census's budget, imported rather than restated.
from ffn_resident_structure import (  # noqa: E402
    SHIM_MM2S_PER_COLUMN,
    _device_blocks,
    _shim_mm2s_census,
)

_TILE_RE = re.compile(r"%(\S+) = aie\.tile\((\d+),\s*(\d+)\)")
_FLOW_RE = re.compile(
    r"aie\.flow\(%(\S+),\s*(\w+)\s*:\s*(\d+),\s*%(\S+),\s*(\w+)\s*:\s*(\d+)\)"
)


def _air_opt():
    """The air-opt this gate runs, resolved from PATH and reported.

    Provenance matters here more than usual: the pass is new, so an air-opt
    that predates it does not fail the comparison -- it fails to run at all,
    and a gate that treated that as "no difference found" would be vacuous.
    Clause 0 below turns that into an explicit failure.
    """
    from shutil import which

    return which("air-opt")


def _run_air_opt(src_text, args, workdir):
    """Run air-opt over ``src_text``; return (stdout_text, stderr, returncode)."""
    src = os.path.join(workdir, "in.mlir")
    Path(src).write_text(src_text)
    proc = subprocess.run(
        [_air_opt(), src] + list(args),
        capture_output=True,
        text=True,
    )
    return proc.stdout, proc.stderr, proc.returncode


def _counts(text):
    return {
        "launch": len(re.findall(r"\bair\.launch\b", text)),
        "segment": len(re.findall(r"\bair\.segment\b", text)),
        "herd": len(re.findall(r"\bair\.herd\b", text)),
        "pipeline_attr": len(re.findall(r"air\.pipeline_(?:group|stage)", text)),
        "staging_attr": len(re.findall(r"air\.staging", text)),
    }


def check_reproduction(label, ref_module, staged_module, n_stages, workdir):
    """Clause 1+2: the pass turns the staged form into the hand-written one."""
    problems = []
    ref_text, ref_err, ref_rc = _run_air_opt(str(ref_module), [], workdir)
    if ref_rc != 0:
        return [
            f"{label}: the hand-written module does not round-trip: {ref_err[-300:]}"
        ]
    staged_src = str(staged_module)
    out_text, err, rc = _run_air_opt(staged_src, [PASS_FLAG], workdir)
    if rc != 0:
        return [f"{label}: {PASS_FLAG} refused the well-formed group: {err[-400:]}"]

    # Liveness: the INPUT must really be several launches, or the comparison
    # is between the reference and itself.
    cin = _counts(staged_src)
    if cin["launch"] != n_stages or cin["segment"] != n_stages:
        problems.append(
            f"{label}: the staged input has {cin['launch']} air.launch and "
            f"{cin['segment']} air.segment, expected {n_stages} of each -- "
            "the reproduction gate would be comparing the reference to itself"
        )
    if cin["pipeline_attr"] != 2 * n_stages:
        problems.append(
            f"{label}: the staged input carries {cin['pipeline_attr']} pipeline "
            f"markers, expected {2 * n_stages} (group + stage per launch)"
        )

    # Liveness: the OUTPUT must be one launch, one segment, every herd, and no
    # surviving markers (they are consumed, as the ping-pong labels are).
    cout = _counts(out_text)
    if cout["launch"] != 1 or cout["segment"] != 1:
        problems.append(
            f"{label}: fused output has {cout['launch']} air.launch and "
            f"{cout['segment']} air.segment, expected 1 of each -- residency "
            "holds only within a segment, so this is the operation itself"
        )
    if cout["herd"] != n_stages:
        problems.append(
            f"{label}: fused output has {cout['herd']} air.herd, expected "
            f"{n_stages} -- a stage body was dropped or duplicated"
        )
    if cout["pipeline_attr"] or cout["staging_attr"]:
        problems.append(
            f"{label}: fused output still carries "
            f"{cout['pipeline_attr']} pipeline / {cout['staging_attr']} staging "
            "markers; they must be erased on consume"
        )

    # THE reproduction clause.
    if out_text != ref_text:
        ref_lines = ref_text.splitlines()
        out_lines = out_text.splitlines()
        first = next(
            (i for i, (a, b) in enumerate(zip(ref_lines, out_lines)) if a != b),
            min(len(ref_lines), len(out_lines)),
        )
        problems.append(
            f"{label}: pass output differs from the hand-written module at "
            f"line {first + 1}:\n"
            f"      hand-written: {ref_lines[first] if first < len(ref_lines) else '<eof>'}\n"
            f"      pass output : {out_lines[first] if first < len(out_lines) else '<eof>'}"
        )
    else:
        print(
            f"[pipeline-fusion] {label}: REPRODUCED "
            f"({cin['launch']} launches/{cin['segment']} segments in -> "
            f"1 launch/1 segment/{cout['herd']} herds out, "
            f"{len(ref_text)} bytes byte-identical to the hand-written module)"
        )
    return problems


def check_routed_design(label, fused_text, workdir):
    """Clause 4: the FUSED design routes inside the per-column shim budget.

    Redundant while clause 1 holds byte equality -- aircc is deterministic, so
    an identical module compiles identically. It is here because fusion is
    exactly the operation that makes this budget binding: stacking N stages
    into one segment ADDS their per-column L3-facing demand (doc 23), and a
    pass that ignored it would emit designs that compile and behave
    differently. Counted with ffn_resident_structure's census, not a third
    counter.
    """
    import air.ir  # noqa: E402
    from air.backend.xrt import XRTBackend  # noqa: E402

    problems = []
    prev = os.getcwd()
    dumps = []
    with tempfile.TemporaryDirectory(prefix="pipeline-fusion-") as work:
        os.chdir(work)
        try:
            with air.ir.Context(), air.ir.Location.unknown():
                module = air.ir.Module.parse(fused_text)
                backend = XRTBackend(
                    omit_while_true_loop=False,
                    output_format="xclbin",
                    instance_name="norm_tail",
                    target_device="npu2",
                    debug_ir=True,
                )
                try:
                    backend.compile(module)
                except Exception:
                    # Expected: no kernel objects here. aiecc writes every MLIR
                    # dump before it compiles core ELFs, so the routed form has
                    # already landed. Clause 4a below refuses a compile that
                    # died EARLIER than that.
                    pass
                finally:
                    try:
                        backend.unload()
                    except Exception:
                        pass
                import glob

                dumps = [
                    (os.path.basename(p), Path(p).read_text())
                    for p in sorted(glob.glob("air_project/debug_ir/pass_*.mlir"))
                ]
        finally:
            os.chdir(prev)

    if not dumps:
        return [f"{label}: aircc wrote no debug dumps -- nothing routed is measured"]
    final_name, final = dumps[-1]
    devices = _device_blocks(final)
    if not devices:
        return [
            f"{label}: no aie.device in {final_name} -- the compile stopped "
            "before routing, so the census below would prove nothing"
        ]
    # 4a. Fusion means ONE device. Several would mean the stages are
    # time-multiplexed after all, which is the thing being removed.
    if len(devices) != 1:
        problems.append(
            f"{label}: {len(devices)} aie.device in the routed design, expected 1 "
            "-- residency holds only within a segment, so more than one device "
            "means the stages are still time-multiplexed"
        )
    worst = 0
    for name, dev in devices:
        demand, _detail = _shim_mm2s_census(dev)
        over = {c: n for c, n in demand.items() if n > SHIM_MM2S_PER_COLUMN}
        worst = max([worst] + list(demand.values()))
        if over:
            problems.append(
                f"{label}: device {name} columns {sorted(over)} demand more than "
                f"{SHIM_MM2S_PER_COLUMN} shim MM2S ({over}) -- fusing stages ADDS "
                "their per-column L3-facing demand (doc 23)"
            )
    # 4b. The stage edges really are L1->L1 in the fused design.
    tiles = {m[0]: (int(m[1]), int(m[2])) for m in _TILE_RE.findall(final)}

    def kind(t):
        rc = tiles.get(t)
        return (
            None
            if rc is None
            else ("shim" if rc[1] == 0 else "memtile" if rc[1] == 1 else "core")
        )

    core_core = sum(
        1
        for s, _sb, _sp, d, _db, _dp in _FLOW_RE.findall(final)
        if kind(s) == "core" and kind(d) == "core"
    )
    expected = (N_STAGES - 1) * HERD_X
    if core_core != expected:
        problems.append(
            f"{label}: {core_core} core->core flows in the fused design, expected "
            f"{expected} -- the two inter-stage edges, one per column each. An "
            "edge that round-tripped through L3 would still pass every other "
            "clause here (doc 23 §5)"
        )
    if not problems:
        print(
            f"[pipeline-fusion] {label}: ROUTED "
            f"({len(devices)} aie.device, {core_core} L1->L1 flows, worst column "
            f"{worst}/{SHIM_MM2S_PER_COLUMN} shim MM2S)"
        )
    return problems


def check_refusal(label, module_text, expect_substr, workdir):
    """Clause 3: a malformed group is REFUSED, with a diagnostic."""
    out, err, rc = _run_air_opt(module_text, [PASS_FLAG], workdir)
    if rc == 0:
        return [
            f"{label}: the pass ACCEPTED a malformed group and produced output. "
            "Declining to fuse is not a safe fallback here -- the stages stay in "
            "separate aie.devices and the declared L1->L1 edges span them."
        ]
    if expect_substr not in err:
        return [
            f"{label}: refused, but the diagnostic does not mention "
            f"{expect_substr!r}; got: {err.strip()[-300:]}"
        ]
    print(f"[pipeline-fusion] {label}: REFUSED ({expect_substr!r})")
    return []


def _malformed_duplicate_stage(rows, cols):
    """The staged module with stage 2 relabelled as stage 1: a duplicate.

    Derived by rewriting the REAL builder's output rather than hand-writing a
    fixture, so the control cannot drift away from what the builder emits.
    """
    text = str(build_norm_tail_module(rows, cols, herd_x=HERD_X, stage_launches=True))
    return text.replace("air.pipeline_stage = 2 : i64", "air.pipeline_stage = 1 : i64")


def _malformed_dropped_stage(rows, cols):
    """The staged module with stage 1's markers stripped: a gap at index 1."""
    text = str(build_norm_tail_module(rows, cols, herd_x=HERD_X, stage_launches=True))
    return text.replace(
        f'attributes {{air.pipeline_group = "{NORM_TAIL_PIPELINE_GROUP}", '
        "air.pipeline_stage = 1 : i64}",
        "",
    )


def main():
    problems = []
    air_opt = _air_opt()
    print(f"[pipeline-fusion] air-opt: {air_opt}")

    with tempfile.TemporaryDirectory(prefix="pipeline-fusion-opt-") as workdir:
        # Clause 0: the binary on PATH must actually carry the pass. Without
        # this, every comparison below would fail identically and for the wrong
        # reason -- or, worse, a refusal clause would "pass" because air-opt
        # exits nonzero on an unknown flag.
        probe = subprocess.run([air_opt, "--help"], capture_output=True, text=True)
        if PASS_FLAG.lstrip("-") not in probe.stdout:
            print(
                f"[pipeline-fusion] FAIL: {air_opt} does not register "
                f"{PASS_FLAG}. Build and install the tree that carries "
                "AIRFusePipelineLaunches.cpp before running this gate."
            )
            return 1

        # --- Clause 1+2: reproduction, at every claimed shape and variant ---
        shapes = sorted(
            {
                (s["shape"]["rows"], s["shape"]["cols"])
                for s in SPECS
                if s["operator"] == "norm_tail"
            }
        )
        if not shapes:
            print("[pipeline-fusion] FAIL: no norm_tail shapes in SPECS")
            return 1

        variants = [({}, "")]
        # The two forms pattern/fused actually builds, so the gate covers the
        # production caller and not only the catalogue's default.
        variants.append(({"plane_major": True}, " plane_major"))
        variants.append(
            ({"plane_major": True, "mirror_out": True}, " plane_major+mirror_out")
        )

        fused_text_for_routing = None
        for rows, cols in shapes:
            for kwargs, suffix in variants:
                if kwargs.get("plane_major") and rows * cols > (1 << 20):
                    # The builder refuses this by construction (shim BD stride
                    # cap); skipping is correct and is not a silent pass.
                    print(
                        f"[pipeline-fusion] norm_tail {rows}x{cols}{suffix}: "
                        "SKIPPED (plane_major over the shim BD stride cap, "
                        "refused by the builder)"
                    )
                    continue
                label = f"norm_tail {rows}x{cols}{suffix}"
                ref = build_norm_tail_module(rows, cols, herd_x=HERD_X, **kwargs)
                staged = build_norm_tail_module(
                    rows, cols, herd_x=HERD_X, stage_launches=True, **kwargs
                )
                found = check_reproduction(label, ref, staged, N_STAGES, workdir)
                problems += found
                if not found and not suffix and fused_text_for_routing is None:
                    out, _e, _rc = _run_air_opt(str(staged), [PASS_FLAG], workdir)
                    fused_text_for_routing = out

        # --- Clause 1b: J7b, the one-stage group, must be an IDENTITY -------
        # Not a degenerate case worth skipping: it is the clause that says the
        # pass leaves the compiler-formed accumulator ring and the memtile A|B
        # feed untouched, and it is where the air.staging claim is checked.
        plain = str(build_ffn_accum_module())
        staged_accum = str(build_ffn_accum_module(pipeline_group="ffn_down"))
        plain_rt, _e, rc = _run_air_opt(plain, [], workdir)
        fused_accum, err, rc2 = _run_air_opt(staged_accum, [PASS_FLAG], workdir)
        if rc != 0 or rc2 != 0:
            problems.append(f"ffn_accum: air-opt failed ({err.strip()[-300:]})")
        elif "air.staging" not in staged_accum:
            problems.append(
                "ffn_accum: the staged module carries no air.staging, so the "
                "accum_in_place claim is not being checked at all"
            )
        elif fused_accum != plain_rt:
            problems.append(
                "ffn_accum: fusing the one-stage group is not an identity -- "
                "the pass altered a module it had nothing to co-locate"
            )
        else:
            print(
                "[pipeline-fusion] ffn_accum 1-stage group: IDENTITY "
                "(accum_in_place claim accepted, module byte-unchanged)"
            )

        # --- Clause 3: the negative controls, verified refusing -------------
        rows, cols = shapes[0]
        problems += check_refusal(
            "negative control: duplicate stage index",
            _malformed_duplicate_stage(rows, cols),
            "more than once",
            workdir,
        )
        problems += check_refusal(
            "negative control: gap in stage indices",
            _malformed_dropped_stage(rows, cols),
            "has no air.pipeline_stage = 1",
            workdir,
        )
        problems += check_refusal(
            "negative control: accum_in_place claim without the ring shape",
            str(
                build_norm_tail_module(rows, cols, herd_x=HERD_X, stage_launches=True)
            ).replace(
                "air.pipeline_stage = 0 : i64",
                'air.pipeline_stage = 0 : i64, air.staging = "accum_in_place"',
                1,
            ),
            'declares air.staging = "accum_in_place"',
            workdir,
        )

    # --- Clause 4: the fused design routes, inside the column budget --------
    if fused_text_for_routing is not None:
        problems += check_routed_design(
            f"norm_tail {shapes[0][0]}x{shapes[0][1]} fused",
            fused_text_for_routing,
            None,
        )
    else:
        problems.append(
            "no fused module reached the routing clause -- clause 4 measured " "nothing"
        )

    if problems:
        print()
        for p in problems:
            print(f"[pipeline-fusion] {p}")
        print("[pipeline-fusion] FAIL")
        return 1
    print("[pipeline-fusion] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
