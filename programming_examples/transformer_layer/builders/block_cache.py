# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""When a cached block ELF is the ELF the current source would build.

CONTRACT
    ``block_artifact_fingerprint(cfg, key, module)`` is a hex digest over
    everything that determines one of the four block artifacts' binaries.
    ``load_fingerprints(cache)`` and ``save_fingerprints(cache, ...)`` persist
    them beside ``KernelCache``'s own manifest, so
    ``block.py::compile_block_artifacts`` can reuse a cached ELF exactly when
    its fingerprint still matches and rebuild it otherwise.

WHY THE ARTIFACT NAME IS NOT ENOUGH, WHICH IS THE WHOLE REASON THIS EXISTS
    The block's artifact names carry the shape: ``blk_ffn_4096x768x3072``. That
    is enough to keep a manifest written for a different configuration from
    being read as this one, and it is enough for nothing else. The tiles and the
    method come from ``kernel_registry`` and the IR comes from builders this
    study edits, so four binaries whose names match the shape are also satisfied
    by four binaries built from a registry row that has since been re-swept, or
    from a builder that has since been fixed. Reusing those runs the gate
    against an implementation that is no longer here while the results artifact
    records the specs that were just resolved -- a PASS for code nobody tested,
    which is the one outcome the whole example exists to prevent.

WHAT THE FINGERPRINT COVERS, AND WHY EACH PART IS NEEDED
    - The BUILT MLIR, as text. The precise statement of what the builders
      produced, so any edit that reaches the IR is a miss -- including one in
      ``llms/shared``, which this study imports and does not own.
    - The RESOLVED CONFIGURATION. Redundant with the IR for most changes and
      deliberately kept anyway: a registry row's ``tile_k_l1`` also feeds
      ``compile_gemm_mm``'s ``-D`` flags, so a change there can move the external
      object without moving a character of the IR.
    - The DEVICE KERNEL SOURCES. The MLIR NAMES its external objects; it does
      not carry them, so a ``.cc`` edit changes what the ELF executes while
      leaving everything else identical.
    - The BACKEND KWARGS, which pick the ABI and the placement settings.

FOOTGUNS
    - IT CANNOT SEE THE TOOLCHAIN. peano, mlir-aie, aircc and the AIE API
      headers are version state shared with every other example in the
      repository, not this study's source. After a toolchain bump the cache is
      stale and nothing here will say so; ``make clean`` is what invalidates it.
    - The digest is only as good as its determinism. ``str(module)`` and the
      config JSON are both stable across processes -- if that ever stops being
      true the cache silently stops hitting, which costs compile time rather
      than correctness, and ``test_block_cache.py`` is where it would show.
    - ``matrix_multiplication/bf16_in_bf16_out`` is absent from the source
      dirs ON PURPOSE. Its ``run.py`` builds the drain / fused-cast GEMM
      modules -- covered by the IR hash -- but its ``.cc`` files are never
      compiled into any artifact here: every linked GEMM object comes from
      ``bf16_in_fp32_out/mm_aie2p.cc`` (``compile_gemm_mm``), a strict
      superset of the bf16_out copy, and that source IS hashed. An edit to
      the bf16_out copy alone therefore changes neither the binaries nor the
      fingerprint, and listing the directory anyway would turn such an edit
      into a rebuild that produces the identical ELF -- false assurance that
      the gate tracked a change it cannot see. What an edit like that CAN
      invalidate is the registry: the C4 sweep measured its rows against the
      bf16_out copy, so if the two copies drift, re-sweep -- new rows reach
      this fingerprint through the config and the IR.
    - It fingerprints the INPUTS to a compile, not the bytes it produced.
      ``KernelCache.load_manifest`` checks that each cached binary still exists;
      neither of them checks that it is still the binary that compile wrote. A
      run killed mid-copy heals itself on the next pass -- a binary is only
      rewritten after a fingerprint MISS, and the miss stands until a compile
      completes -- but a file replaced under the cache directory by hand does
      not, and nothing here would say so.
"""

import glob
import hashlib
import json
import os

_PROJ_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

#: Where the per-artifact fingerprints are recorded, beside ``KernelCache``'s
#: own ``manifest.json`` in the cache directory.
FINGERPRINT_FILE = "block_fingerprint.json"

#: Every directory holding a device kernel source one of the four ELFs can link.
#: Hashed WHOLE -- every ``.cc`` in each of them -- into every artifact's
#: fingerprint, rather than mapped per artifact.
#:
#: Coarse on purpose. The precise map (which ELF links which object, and which
#: further source each of those includes -- ``encoder.cc`` alone pulls in four)
#: is one more thing to keep in step with the builders, and the way it goes
#: wrong is that an edit lands in a source no entry names and the stale ELF is
#: accepted. Hashing the directories goes wrong the other way: editing
#: ``encoder.cc`` also rebuilds the addnorm ELF, which costs a compile and never
#: costs a wrong result.
#:
#: ``bf16_in_fp32_out``, not ``bf16_in_bf16_out``: the objects every GEMM
#: method links -- drain included -- compile from the fp32_out copy of
#: ``mm_aie2p.cc``. See the module footguns before "fixing" this.
_KERNEL_SOURCE_DIRS = (
    os.path.join(_PROJ_ROOT, "transformer_layer", "kernels"),
    os.path.join(_PROJ_ROOT, "matrix_multiplication", "bf16_in_fp32_out"),
    os.path.join(_PROJ_ROOT, "flash_attention", "kernel_fusion_based"),
)


def _device_source_digest():
    """One digest over every device kernel source in ``_KERNEL_SOURCE_DIRS``."""
    digest = hashlib.sha256()
    for directory in _KERNEL_SOURCE_DIRS:
        for path in sorted(glob.glob(os.path.join(directory, "*.cc"))):
            digest.update(os.path.relpath(path, _PROJ_ROOT).encode())
            with open(path, "rb") as handle:
                digest.update(handle.read())
    return digest.hexdigest()


def block_artifact_fingerprint(cfg, key, module):
    """Everything that determines one artifact's ELF, as a hex digest.

    ``cfg`` is ``block_config(...)``, ``key`` names one of its ``artifacts``,
    and ``module`` is what that artifact's builder just produced. See the module
    docstring for what this does and does not cover.
    """
    name = cfg["artifacts"][key]
    payload = {
        "artifact": name,
        "backend_kwargs": cfg["backend_kwargs"][name],
        "config": {k: v for k, v in cfg.items() if k != "backend_kwargs"},
        "mlir": str(module),
        "device_sources": _device_source_digest(),
    }
    # `default=repr` rather than a raise: a spec that grows a non-JSON field
    # should keep contributing to the digest, not take the check down.
    blob = json.dumps(payload, sort_keys=True, default=repr)
    return hashlib.sha256(blob.encode()).hexdigest()


def load_fingerprints(cache):
    """The fingerprints a previous compile recorded in ``cache``, or ``{}``.

    Any failure to read the file is an empty dict rather than an exception. A
    missing, truncated or hand-edited file means "nothing in here is known
    good", and the only safe response to that is to compile.
    """
    path = os.path.join(cache.cache_dir, FINGERPRINT_FILE)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as handle:
            recorded = json.load(handle)
    except (OSError, ValueError):
        return {}
    return recorded if isinstance(recorded, dict) else {}


def save_fingerprints(cache, fingerprints):
    """Record what is now in the cache, MERGED over what was already recorded.

    Merged because the cache holds every shape ever built into it -- a bring-up
    block beside the gate point, say -- and rewriting the file with only this
    configuration's names would drop the others' entries, turning their next run
    into a needless four-ELF recompile.
    """
    merged = dict(load_fingerprints(cache))
    merged.update(fingerprints)
    path = os.path.join(cache.cache_dir, FINGERPRINT_FILE)
    with open(path, "w") as handle:
        json.dump(merged, handle, indent=2, sort_keys=True)
