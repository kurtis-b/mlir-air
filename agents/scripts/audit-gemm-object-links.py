# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Does every shipped model COMPILE the GEMM objects its modules LINK? No hardware.

WHY THIS EXISTS
    `llms/shared/builders/gemm_builder.py` mints the MLIR symbol suffix and the
    `mm_*.o` filename of every external GEMM, and ten shipped LLM deployments
    resolve their tile configurations through it. The cross-deployment regression
    rule says any change there re-runs `make verify` on all ten -- hours of
    hardware, and the worst possible place to discover that a module asks for an
    object nobody built.

    This is the same question asked in about a minute. It runs each model's
    `compile_all_kernels` with the two expensive things stubbed:

        external_kernels._compile_kernel   -> records (object name, -DDIM_* flags)
        KernelCache.compile_and_cache      -> scrapes `link_with = "..."` out of
                                              the module text and returns

    Every module still gets BUILT (about a second each), so what each one
    references is real. Nothing invokes peano or aiecc. It then reports two things:

      - an object a module links that nothing compiled  (unresolved symbol at
        link time -- loud, and hours in)
      - an object built at a DIM_N other than the one its name claims  (links
        cleanly and returns zeros for part of every output tile -- silent, and
        this is the one that cost Phase D2 a gate cycle)

    Phase E1 made both names functions of (tile_m, tile_n). This script found that
    `qwen25_0_5b` had been mis-linked all along: it resolves its O and Down GEMMs
    at the registry's tile_n=32, overrides tile_n to 128 because the narrow tile is
    numerically broken for the drain path, and under the old method-only naming
    still asked for `mm_m32.o` -- an object compiled at DIM_N=128. Correct by
    accident.

USAGE
    python3 agents/scripts/audit-gemm-object-links.py            # all ten models
    python3 agents/scripts/audit-gemm-object-links.py qwen25_0_5b

    Exit 0 only if every model agrees. No NPU, no lock, no XRT.

FOOTGUNS
    - Run it from a scratch directory. The stubs stop peano from writing objects,
      but a model's kernel-compile path still creates a KernelCache directory.
    - ONE MODEL PER PROCESS. Each model repoints `sys.path[0]` at its own
      directory and several of them share module names (`*_prefill`, `*_weights`),
      so auditing two in one interpreter reads the first one's modules. The
      no-argument form forks per model for exactly this reason.
    - A model's decode entry may raise on a missing `.o` -- the stub records
      compiles instead of performing them, so a decode path that reads an object
      back off disk fails. That is reported and not fatal: the prefill modules are
      where the external GEMMs are, and they have already been built by then.
"""

import re
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
_LLMS = _REPO / "programming_examples" / "llms"

MODELS = [
    "llama32_1b",
    "llama32_1b_int4",
    "llama32_3b",
    "smollm2_1_7b",
    "qwen25_0_5b",
    "qwen25_1_5b",
    "qwen25_3b",
    "qwen3_0_6b",
    "qwen3_1_7b",
    "qwen3_4b",
]

# verify_runner.py: "Production prefill kernels are tiled for seq_len=2048."
SEQ_LEN = 2048


def audit(model):
    """``(built, linked, problems)`` for one model. Import-heavy; one per process."""
    for p in (str(_LLMS), str(_LLMS / "verify"), str(_LLMS / model)):
        while p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)

    import shared.infra.cache as cache_mod
    import shared.infra.external_kernels as ek

    built = {}  # object name -> {"DIM_M": ..., "DIM_N": ..., "DIM_K": ...}
    linked = {}  # artifact name -> set of objects its MLIR names in `link_with`

    def stub_compile_kernel(src_path, output_name, extra_flags=None, force=False):
        dims = {}
        for flag in extra_flags or []:
            match = re.match(r"-D(DIM_[MNK])=(\d+)$", flag)
            if match:
                dims[match.group(1)] = int(match.group(2))
        built[output_name] = dims

    def stub_compile_and_cache(self, name, module, backend_kwargs=None, **kwargs):
        linked[name] = set(re.findall(r'link_with = "([^"]+)"', str(module)))
        self.artifacts[name] = name

    ek._compile_kernel = stub_compile_kernel
    cache_mod.KernelCache.compile_and_cache = stub_compile_and_cache
    # The manifest records per-artifact metadata the stub above does not produce,
    # and a loaded manifest would let a model skip compiling entirely.
    cache_mod.KernelCache._save_manifest = lambda self: None
    cache_mod.KernelCache.load_manifest = lambda self, *a, **k: False

    adapter = __import__(f"{model}.verify_adapter", fromlist=["*"])
    cache = cache_mod.KernelCache(f"audit_{model}_cache", verbose=False)
    adapter.compile_prefill_kernels(
        cache, adapter.build_config(), seq_len=SEQ_LEN, cpu_attn=False
    )
    try:
        adapter.compile_decode_kernels(cache, adapter.build_config())
    except Exception as exc:  # see the footgun on decode paths that read objects
        print(f"  note: decode entry raised {type(exc).__name__}: {exc}")

    problems = []
    for artifact, objects in sorted(linked.items()):
        for obj in sorted(o for o in objects if o.startswith("mm_m")):
            if obj not in built:
                problems.append(f"{artifact} links {obj}, which nothing compiled")
                continue
            claimed = int(re.search(r"n(\d+)\.o$", obj).group(1))
            actual = built[obj].get("DIM_N")
            if actual != claimed:
                problems.append(
                    f"{artifact} links {obj} but it was built at DIM_N={actual} "
                    f"-- it would link cleanly and compute the wrong tile width"
                )
    return sorted(built), linked, problems


def _report(model):
    built, linked, problems = audit(model)
    print(f"== {model} ==")
    print(f"  built:  {[b for b in built if b.startswith('mm_')]}")
    for artifact, objects in sorted(linked.items()):
        gemm = sorted(o for o in objects if o.startswith("mm_m"))
        if gemm:
            print(f"  links:  {artifact} -> {gemm}")
    if problems:
        print("  PROBLEMS:")
        for problem in problems:
            print(f"    - {problem}")
        return 1
    print("  OK: every linked GEMM object was built at the DIM_N it names")
    return 0


def main():
    if len(sys.argv) > 1:
        return _report(sys.argv[1])

    failed = []
    for model in MODELS:
        # One subprocess per model: several of them share module names, so two in
        # one interpreter would read the first one's modules.
        result = subprocess.run(
            [sys.executable, __file__, model], capture_output=True, text=True
        )
        verdict = "OK" if result.returncode == 0 else "MISMATCH"
        print(f"{model}: {verdict}")
        if result.returncode != 0:
            failed.append(model)
            print(result.stdout)
            print(result.stderr[-2000:])
    print(f"\n{len(MODELS) - len(failed)}/{len(MODELS)} models consistent")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
