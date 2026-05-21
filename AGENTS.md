# MLIR-AIR Agent Guide

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

- Codex skills: `.codex/skills/`
- Claude skills: `.claude/skills/`

For state checks and helper commands, use `agents/scripts/doctor.sh help`.
