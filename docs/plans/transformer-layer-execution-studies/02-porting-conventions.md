# 02 — Porting Conventions

iron and MLIR-AIR have genuinely different house styles. A file-faithful port would import
iron's conventions into a repository that does not use them, and the result would be
permanently foreign — harder to review, harder to extend, and visibly grafted on.

**Every ported file is rewritten to MLIR-AIR's conventions as part of the port, not as a
follow-up.** This document is the checklist; use it at review time.

This refactoring is not free, and the plan's effort estimates assume it. The ~19,000-line
"ports by structure" tier in [09-phase-f-study-harness.md](01-original-plan-superseded.md) is
where most of it lands.

---

## 1. Operator model — the biggest single change

iron models operators as an `AIE*`-prefixed class hierarchy: `AIEOperatorBase` →
`AIEGEMM`, `AIEQKVProj`, `AIETransformerRunlist`, with a two-phase
`set_up_artifacts()` / `set_up_runtime()` contract and a paired `op.py` / `design.py` split.

MLIR-AIR models them as **plain module-level functions** — `build_<name>_module(...)` returning
an `air.ir.Module`. Classes are reserved for infrastructure: `KernelCache`, `Profiler`,
`FuncArg`, `KernelSlice`.

```python
# iron
class AIEQKVProj(AIEOperatorBase):
    def set_up_artifacts(self): ...
    def set_up_runtime(self): ...
    def forward(self, x): ...

# MLIR-AIR
def build_qkv_proj_module(seq_len, emb_dim, ...):
    """..."""
    return module
```

Collapse each pair into one builder function and let `KernelCache` own compile, cache and
dispatch. **Drop the `AIE` class prefix entirely** — no ported symbol should carry it.

## 2. Delete iron's runtime scaffolding rather than porting it

`compilation.py` (712 lines, artifact DAG), `AIEContext`, and the `AIEDeviceManager` singleton
with its `reset_runtime()` workarounds exist to solve problems that `KernelCache` + native
`aircc` + `XRTBackend` already solve.

The one genuinely valuable idea inside them — BO liveness pooling — is extracted into
`KernelCache` in [05-phase-b-runtime-seam.md](01-original-plan-superseded.md). The rest is dropped.

## 3. The `design.py`-loaded-by-file-path seam disappears

iron never imports `design.py`. It loads it by path through
`GenerateMLIRFromPythonCompilationRule`, passing `callback_args` / `callback_kwargs` — purely
because the artifact DAG needs a file-level dependency node.

With the DAG gone, builders are ordinary imported functions. Remove the indirection; do not
reproduce a plugin-loading mechanism that has no remaining purpose.

## 4. File layout

iron's `<op>/{op,design,reference,test}.py` quadruple becomes MLIR-AIR's shape:

```
programming_examples/<example>/
├── <name>.py            # builder + its numpy/torch reference as module-level functions
├── Makefile             # help / compile / run / profile / clean
├── run_npu2_*.lit       # target encoded in the filename
└── <kernel>.cc          # if the example owns a device kernel
```

The reference oracle lives in the same file as the builder — see
`programming_examples/weighted_rms_norm/weighted_rms_norm.py`, which holds both
`build_module()` and `rms_norm_reference()`. Architecture-orthogonal builders belong in
`programming_examples/llms/shared/builders/`.

## 5. Module size

MLIR-AIR's shared modules run 177–799 lines:

| Module | Lines |
|---|---|
| `shared/builders/gemm_builder.py` | 177 |
| `shared/infra/external_kernels.py` | 304 |
| `shared/infra/stitching.py` | 438 |
| `shared/infra/cache.py` | 624 |
| `shared/builders/o_ffn_multi.py` | 712 |
| `shared/builders/rms_qkv_qknorm_rope_multi.py` | 799 |

iron's exceed that by two to three times: `unattended_reboot.py` (2494),
`end_to_end/modes.py` (2336), `test_unattended_reboot.py` (1790), `mha_out_proj/design.py`
(1350), `ffn/design.py` (1096), `dynamic_gemm/design.py` (1009).

Split along seams that already exist inside them:

- **`unattended_reboot.py`** → job planning · state persistence · crontab hook · thermal gate ·
  TTM page-limit transitions · reboot orchestration · CLI.
- **`modes.py`** → one module per execution mode plus a thin registry, replacing the
  `_build_operator()` switch.

## 6. Licence headers

MLIR-AIR uses a two-line header:

```python
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
```

iron uses `SPDX-FileCopyrightText` with Apache-2.0, and enforces `reuse lint` through a
`REUSE.toml`.

- Files that are **genuine rewrites** take the MLIR-AIR MIT header.
- Files copied **substantially verbatim** retain their Apache-2.0 header. Both repositories are
  AMD-copyright and Apache-2.0 may live in an MIT project, but attribution must be preserved.
  Note the mixed licensing in the example README.
- **Do not port `REUSE.toml` or the `reuse lint` CI step.** MLIR-AIR does not use REUSE, and
  iron's own commit message records that tooling producing spurious failures.

## 7. Drop iron's dual naming

iron keeps `hybrid` as the internal module name while the paper label is "coarse runlist", and
carries a `pattern_label` column to bridge them.

The port has no internal back-compatibility obligation. **Name it `coarse`** in code,
directories and prose. Keep `hybrid` only as the CSV *value*, where diffing against iron result
trees needs it, and confine that mapping to one place in the schema module.

## 8. Delete redundancy on the way in

- `pattern/{offload,runlist,hybrid}/reference.py` are 8-line re-exports of the shared reference.
  Delete them; import the shared one.
- `addnorm_ffn.cc` (931) and `addnorm_ffn_addnorm.cc` (936) are near-duplicates differing by a
  trailing stage. Make it one source behind a `-D` flag, matching how MLIR-AIR already
  parameterizes kernels (`-DDIM_M` / `-DDIM_N` / `-DDIM_K`).
  *(`.cc` including other `.cc` is **not** a deviation — `mm_aie2p.cc` includes `zero.cc`.)*
- `causal_mask` has no device design at all: it is `elementwise_add` plus a precomputed static
  mask. It becomes a builder keyword argument, not an operator.

## 9. Configuration source of truth

iron hardcodes tuning candidates in `{hybrid,runlist,offload}_candidates.json`.

MLIR-AIR's convention is registry-driven. `programming_examples/kernel_registry/registry_lookup.py`
exists specifically so tile sizes are never hand-copied — its docstring records that hand-copying
"caused drift / stale-config bugs" — and it **raises** rather than guessing on an unmeasured
shape.

Feed the `block` sweep into `kernel_registry` and have builders read from it.

## 10. Reuse the existing measurement plumbing

iron's `conftest.py` regex-scrapes stdout through a `@pytest.mark.metrics` marker.

MLIR-AIR already has `Profiler` (`shared/infra/cache.py`) for in-process timing and
`llms/bench/extract_perf.py` for stdout scraping, feeding `append_history.py` → the
`perf-history` branch → published charts. Route ported measurement through those rather than
standing up a second, parallel mechanism.

## 11. Test discovery

iron's `pytest.ini` sets `python_files = test.py`, so its own `study/test_*.py` files are
silently **not** collected by directory — an inconsistency that hid real bugs.

Normalize on standard `test_*.py` discovery.

## 12. Formatting and docstrings

- Run `black` over all ported Python; `.github/workflows/lintAndFormat.yml` enforces it.
- Run clang-format / clang-tidy over ported C++.
- Match MLIR-AIR's docstring standard: modules carry a docstring stating the contract **and its
  footguns**. See `stitching.py` (SSA renaming, arg wiring, extern de-dup),
  `cache.py` (the zero-copy shared-buffer caveat), `registry_lookup.py` (why it raises), and
  `external_kernels.py::compile_gemm_mm` (why `sym_suffix` exists). This is more explicit than
  iron's style, and it is what makes the shared infrastructure safe to reuse.

---

## Review checklist

Before merging any ported file:

- [ ] No `AIE*`-prefixed operator class
- [ ] No `op.py` / `design.py` pair; one `build_*_module()` function
- [ ] Reference oracle in the same module as the builder
- [ ] Module under ~800 lines, or split with a stated seam
- [ ] Correct licence header for its disposition (rewrite → MIT; verbatim → Apache-2.0)
- [ ] No `REUSE.toml`, no `reuse lint` step
- [ ] `coarse`, not `hybrid`, outside the one CSV mapping
- [ ] Tile configuration read from `kernel_registry`, not hardcoded
- [ ] Measurement routed through `Profiler` / `extract_perf.py`
- [ ] Tests discoverable as `test_*.py`
- [ ] `black` clean; module docstring states the contract and its footguns
