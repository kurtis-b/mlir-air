Review this phase of a compiler-porting effort. Report only substantive problems; do not restate
the diff, do not praise it.

## What to review

Exactly this diff, in the repository at /home/cj/mlir-air:

```
git diff __PHASE_START_SHA__..HEAD
```

Start by running that command and `git log --oneline __PHASE_START_SHA__..HEAD`. Read the changed
files in full where the diff alone is not enough to judge them. Nothing outside this diff is in
scope, except where you need surrounding context to tell whether a change is correct.

Phase: __PHASE_ID__ — __PHASE_NAME__
Specification: `__PHASE_DOC__` (read it — it defines what this code is supposed to do)
Binding house style: `docs/plans/transformer-layer-execution-studies/02-porting-conventions.md`

The work ports code from the AMD IRON repository at /home/cj/iron (branch `devel`, commit
`1e014c1`) into MLIR-AIR. IRON uses the `aie.iron` Python API; MLIR-AIR uses the AIR dialect.
The port is a re-expression, never a file copy.

## What to examine, in priority order

**1. Weakened gates — report these in `weakened_gates`, and be suspicious.** This diff was
written by an autonomous agent that was told to make a gate pass. Did it weaken, delete, narrow,
skip, stub or `XFAIL` any test, assertion, tolerance, or check? Is any new test trivial — passing
without actually exercising the code it claims to cover? Phase A is especially exposed because it
creates the very lit suite that gates it: confirm the test genuinely compiles the ported kernels
rather than merely succeeding. Prefer a false positive here to silence.

**2. Correctness.** Concrete defects, with a specific failing input or condition. For kernel
work: wrong ABI, wrong symbol names, wrong tile or shape assumptions, wrong compile-time `-D`
macros, arguments that do not match the builder that calls them.

**3. Wrong target.** A known trap in this codebase: the shared LLM path compiles
`programming_examples/matrix_multiplication/bf16_in_fp32_out/mm_aie2p.cc`, **not** the bf16-output
variant. Extending the wrong file produces code that compiles and is never used. Check that every
file touched is the one actually on the path it claims to affect.

**4. Convention violations** from `02-porting-conventions.md`: `AIE`-prefixed operator classes,
`op.py`/`design.py` pairs, a ported `REUSE.toml`, modules materially over ~800 lines, missing
module docstrings, wrong licence header for the file's disposition, hardcoded tile sizes that
should come from `kernel_registry`.

**5. Shared-infrastructure risk.** Ten LLM deployments under `programming_examples/llms/` depend
on `llms/shared/`, `kernel_registry/`, and `matrix_multiplication/`. Does this diff change
behaviour any of them relies on?

## Output

Return the structured verdict. `verdict` is `pass` only when `blocking` and `weakened_gates` are
both empty. Cite concrete file paths. A finding you cannot tie to a specific file and a specific
failure mode belongs in `non_blocking`, not `blocking`.
