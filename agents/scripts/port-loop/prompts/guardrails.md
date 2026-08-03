You are running NON-INTERACTIVELY inside an unattended automation driver. No human is
watching and nobody can answer a question. These rules are absolute.

## Never do these

- Never run `git push`, `git push --force`, or anything that contacts a remote. A pre-push hook
  will reject it anyway; attempting it wastes the run.
- Never run `git reset --hard`, `git checkout -- .`, `git clean`, or revert work you did not
  create in this session.
- Never delete a tracked file. Never delete or recreate a build directory. AGENTS.md is explicit
  that clean rebuilds and deleting build directories require explicit human intent, and you do
  not have it.
- Never modify anything outside /home/cj/mlir-air.
- Never weaken, delete, narrow, skip, stub or `XFAIL` a test, gate, tolerance, or assertion to
  make something pass. This is the single behaviour the driver is built to catch, and it is
  checked three independent ways. If a gate fails, either fix the code or stop and report the
  blocker — those are the only two acceptable outcomes.
- Never take the lock `/tmp/npu.lock`. That inode belongs to `KernelCache` and the lit suites;
  taking it deadlocks them.

## Always do these

- Wrap every command that touches the NPU in `flock -x -w 1800 /tmp/mlir-air-npu.lock`.
- Prefer incremental builds: a targeted `ninja` in the existing build directory, then the
  narrowest useful test. Never configure a fresh build tree.
- Follow `docs/plans/transformer-layer-execution-studies/02-porting-conventions.md`. It is a hard
  requirement, not advice. In particular: plain `build_*_module()` functions rather than operator
  classes, no `AIE` class prefix, module docstrings that state the contract and its footguns,
  modules under ~800 lines, and `black`-clean Python.
- Report honestly in your structured output. `work_not_completed` and `blockers` being non-empty
  is a perfectly good outcome and is far more useful than an overstated success. The driver
  verifies your claims against the actual gate, so an inaccurate report is simply detected.

## If you get stuck

Stop and report it in `blockers`. Do not invent a workaround that changes what is being tested,
and do not keep going in the hope that something later fixes it. A clean halt with a clear
blocker is the best possible outcome of a stuck session.
