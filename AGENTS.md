# MLIR-AIR Agent Guide

The workflow — roles (Claude codes on a branch, Codex reviews advisory-only,
`agents/scripts/pr.sh land` merges through a script-enforced gate), git rules, the landing gate,
review protocol, integrity, enforcement — is `agents/WORKFLOW.md` (vendored and adapted from
kurtis-b/agent-standards @ 84484c2 via DAM-RS's landing-gate variant; plain files, not a
submodule — the standards repo is private and CI must clone nothing extra). This file adds only
mlir-air-specific rules. This fork is origin-only: no pushes or PRs to upstream Xilinx/mlir-air;
upstream is pulled into `main` by the operator.

## Repo-specific workflow rules

- PR size: ≤ 500 ADDED lines vs the merge-base with origin/main
  (`agents/scripts/check_pr_size.sh`; rename-aware; submodule bumps/lockfiles/declared generated
  files exempt, adjudicated by the human). Total churn above the advisory threshold must be
  acknowledged in the review.
- Refactor-before-add: every task states its preparatory refactor or `none` with a reason;
  default is atomic structure/behavior commits within one PR. Hardware-touching structure commits
  keep their gates (lit subset, device suite, verify) with a before/after baseline.
- Reuse-first: before writing a new function or file, search for the existing seam and name what
  is reused, or state the scope searched.
- Pre-edit validation plan: name the invariant or failing test, the baseline for structural
  changes, the cheapest check first, and the NPU/perf gates owed when hardware behavior may
  change. No performance claim without an artifact.
- Review findings are implemented only when they affect correctness or stated requirements;
  everything else is explicitly adjudicated (fix or reject with reason), never silently applied
  or dropped.
- Hardware gates run through the device scheduler, `agents/scripts/devq.sh` (FIFO broker for the
  single NPU: `run --class build|measure -- CMD` is the drop-in for a bare `flock`; `preflight`
  asks before dispatching; never `tee /dev/stderr` inside a job — the script's header explains
  each rule). Software gates on main: the lit subset (`build-xrt/mlir/test`) and the per-example
  verify targets.
- PRs are landed by `agents/scripts/pr.sh land` (see the workflow's Landing gate); the fork-side
  `.github/workflows/pr-size.yml` is a deliberate divergence of the upstream workflows dir.

## Task start

This repository keeps human-facing build, run, test, and AIR semantics documentation in `docs/`. The `agents/` directory is scripts-only: keep helper scripts in `agents/scripts/`, and keep generated local state under ignored `agents/.state/`.

Start each task by naming its profile:

- Ryzen/AIE/NPU
- GPU/ROCDL
- compiler development
- benchmarking
- testing
- docs

Before setup, build, or test commands, check the current state: branch, dirty worktree, sourced shell environment, dependency paths, build/install directories, and the tool versions actually on `PATH`. Prefer recovering shell state by sourcing the existing setup scripts over reconfiguring a build.

Use incremental rebuilds by default. If a build directory already exists, prefer a targeted `ninja` command in that directory, followed by the narrowest useful lit test, unit test, compile-only example, or benchmark dry run. Clean rebuilds, deleting build directories, and reverting user changes require explicit user intent.

Canonical docs:

- Build index: `docs/building.md`
- Ryzen/AIE/NPU setup: `docs/buildingRyzenLin.md`
- GPU/ROCDL setup: `docs/buildingGPU.md`
- Testing: `docs/testing.md`
- `aircc`: `docs/aircc.md`
- AIR semantics and backend mapping: `docs/AIRComputeModel.md`
- Async/dependency workflows: `docs/AIRAsyncConcurrency.md`
- GEMM/NPU pipeline reference: `docs/GEMMCaseStudy.md`
- Runtime/tracing: `docs/AIRRunner.md`, `docs/trace.md`

Repo-local skill guides:

- Claude skills (canonical): `.claude/skills/`
- Codex reviewer skills (findings-only manifest): `.codex/skills/`

For state checks and helper commands, use `agents/scripts/doctor.sh help`.
