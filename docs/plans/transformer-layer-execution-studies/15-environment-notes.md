# 15 — Environment notes

Read this before running a gate. On 2026-08-03 the toolchain on this machine was four layers
stale, and the failure mode was not a build error — it was **every NPU test reporting UNSUPPORTED
while the suite exited 0**. A hardware gate that runs zero hardware tests reports success.

## Two CMake flags that are lost on any clean rebuild

`utils/build-mlir-air-using-wheels.sh` sets neither. After any `rm -rf build-xrt`, re-apply both:

```bash
cmake -S . -B build-xrt \
  -DXRT_COREUTIL=/opt/xilinx/xrt/lib/libxrt_coreutil.so \
  -DENABLE_RUN_XRT_TESTS=ON
```

They survive an ordinary reconfigure (they are in the cache); only a wipe loses them.

**`XRT_COREUTIL`** — this machine has **two XRT installs**: `/opt/xilinx/xrt` and a system one
under `/usr/lib/x86_64-linux-gnu`. `FindXRT.cmake` derives the install root as the *parent of the
lib directory*, which for a multiarch path gives `/usr/lib`, so `XRT_BIN_DIR` becomes the
nonexistent `/usr/lib/bin`. lit probes `${config.xrt_bin_dir}/xrt-smi` to decide whether an NPU
exists, never finds it, and marks every NPU test UNSUPPORTED. `-DXRT_DIR=` cannot fix this —
`FindXRT` overwrites `XRT_DIR` from `find_library`. `XRT_COREUTIL` is the only lever.

**`ENABLE_RUN_XRT_TESTS`** — defaults `OFF`. Without it lit reports "xrt not found or xrt tests
disabled" and skips everything, again exiting 0.

Verify both took effect:

```bash
grep -E "xrt_bin_dir|enable_run_xrt_tests" build-xrt/programming_examples/lit.site.cfg.py
# expect: /opt/xilinx/xrt/bin  and  "ON"
lit -s build-xrt/programming_examples/flash_attention   # expect 9 passed / 2 unsupported
```

`port-loop.sh` preflight refuses to start a hardware phase when `config.xrt_bin_dir` has no
executable `xrt-smi`, and prints the fix. That guard exists because of this.

## Toolchain versions that must agree

| Component | Version | Pinned by |
|---|---|---|
| LLVM/MLIR | `23.0.0.2026071405+46fcb339` | `utils/clone-llvm.sh --get-wheel-version` |
| mlir-aie | `1.4.0` | commit `d3c5f870` |
| llvm-aie (Peano) | `21.0.0.2026051501+f4933ef7` | unpinned |
| AIR build + install | rebuilt 2026-08-04 into `build-xrt` / `install-xrt` | — |

Use **`install-xrt`**, not `install`: only `build-xrt` carries `XRT_LIB_DIR`.

Each layer masked the next when stale, in this order:

1. Installed `air` Python bindings predated `dynamic_src_offsets` (`ChannelPutOp.__init__()
   got an unexpected keyword argument`) — needed a rebuild.
2. The rebuild failed on `no member named 'ExpandModeAttr' in namespace 'xilinx::AIEX'` — mlir-aie
   was older than `d3c5f870` requires.
3. With mlir-aie 1.4.0 the build failed inside **its** headers on `no type named 'PropertyRef' in
   namespace 'mlir'` — LLVM/MLIR was four months behind the pin.

### Stale tablegen output after changing the MLIR wheel

CMake tracks `.td` → `.inc` dependencies but **not the `mlir-tblgen` binary**. Swapping the MLIR
wheel under an existing build tree leaves generated headers behind, producing errors that look
like source bugs (`unknown type name 'Properties'`, `no type named 'OpaqueProperties'`). Fix
without a full wipe:

```bash
find build-xrt/mlir/include -name "*.inc" -delete && ninja -C build-xrt install
```

### Python bindings need a clean configure

Extension-to-source wiring is baked into `build-xrt/python/CMakeFiles` at configure time. When
the July MLIR moved `IRCore.cpp` from `MLIRPythonExtension.Core` into
`MLIRPythonExtension.MLIRPythonSupport`, an incremental reconfigure could not re-wire it and
`air._mlir_libs._mlir` was missing `OnExplicitAction`. That one genuinely needs `rm -rf build-xrt`
and a fresh configure — then re-apply the two flags above.

## The transformer-layer suite now needs an NPU

`[2026-08-04]` `check-programming-examples-transformer-layer` was NPU-free through Phases A and B
— which is how Phase B's hardware claim came to be gated by nothing the driver ran. It is not
NPU-free any more: it carries the Phase B runlist gate and every Phase C operator gate.

`[2026-08-05]` **16 tests as of D2**, ten of them `REQUIRES: ryzen_ai_npu2`. D1 added no files but
new shapes to six existing tests; D2 added `run_npu2_block_peano.lit`, `run_reference_tests.lit`
and `run_block_cache_tests.lit`. The driver now refuses to accept a `needs_hardware` phase whose
gate log does not show all of them passing with nothing unsupported or xfailed — see
[14](14-the-port-loop-harness.md). Enrolment is path-based (`--filter "transformer_layer/"`), so a
new `.lit` anywhere under the example joins the suite with no CMake change, and the driver's count
follows automatically.

If you want the PR-safe subset, the individual make targets are still hardware-free:

```bash
cd programming_examples/transformer_layer
make compile          # Peano objects and their symbol lists
make seam-tests       # BO pooling and runlist aggregation rules
make registry-plan    # what the sweep would measure
make registry-writer-tests
make reference-tests  # the golden model's composition, host-only  [2026-08-05]
make block-cache-tests
```

Everything else in that directory dispatches, and every dispatching command belongs inside
`flock -x -w 1800 /tmp/mlir-air-npu.lock`.

## Generated artifacts land in the source directory

`make runlist-gate` and the sweep run with `programming_examples/transformer_layer/` as their
working directory, so `aircc` and `KernelCache` write their objects, `air.mlir`, `air.elf` and
`air_project/` straight into the source tree. Eleven of those were committed by mistake in
`bf69ed69`; the nine `.o` files were the only tracked object files in the whole repository. They
are untracked and `.gitignore`d now, but the *cause* is unchanged — anything new that runs from
that directory will leak artifacts there too, so check `git status` before committing.

`[2026-08-05]` It happened again, exactly as predicted: D2's block gate left a 6.3 MB
`block_cache/` of ELFs and `insts.bin` in the source tree, ignored by nothing and cleaned by
nothing. Both are fixed (`transformer_layer/.gitignore` and the `clean` target), but **a new
artifact directory is the default outcome of adding a `KernelCache`-backed gate, not an
exception.** If Phase E adds a cache per strategy, add each one to both places in the same commit.

## Known-stale, not yet fixed

- **mlir-aie 1.4.0 ships `aiecc`, not `aiecc.py`**, and lit still probes for `aiecc.py`
  (`Did not find aiecc.py in ...`). It did not block `flash_attention` or Phases A–C, but it may
  bite examples relying on that substitution.
- **Six pre-existing `check-programming-examples-peano` failures**, NPU1-only:
  `matrix_multiplication/{bf16,i16,i8}` pass `-mllvm -aie-disable-fold-imm`, which the installed
  llvm-aie rejects at option parsing. Toolchain drift, outside this plan's diff.

## Hardware facts

| | |
|---|---|
| NPU | `/dev/accel/accel0`, AMD Ryzen AI 9 HX 370 (NPU2 / Strix), firmware 1.1.2.64 |
| iGPU | Radeon 890M — `host_comparison` (Phase F) is possible on this machine |
| XRT | 2.21.0, `amdxdna` 2.21.0_20260514 |
| RAM | 31 GB — 3B/4B verify needs `verify_runner.py`'s subprocess split. **`[2026-08-20]` and still OOM-kills the session — see the oomd note below** |
| Concurrent `hw_context` ceiling | **32**; 33 fails with `DRM_IOCTL_AMDXDNA_CREATE_HWCTX err=-2` |
| Chassis | **laptop** — lid close suspends. Wrap long runs in `systemd-inhibit`. On battery, GNOME suspends after 15 min idle; on AC it does not. |
| Logout | Background runs survive (`KillUserProcesses=no`) |

## `[2026-08-20]` The big-model `verify` leg is killed by systemd-oomd, and it takes the session with it

What looked like two overnight laptop reboots on 2026-08-19/20 were **systemd-oomd cgroup kills**
at ~27 GB used of 30 (`uptime -s` still reads 2026-08-13): `llama32_3b`'s `make verify` crosses the
memory-pressure threshold during its HF-compare phase, and because the devq runner and the terminal
share one session scope, oomd kills the whole scope — the job reads `exit 137` with
`runner pid gone` (devq 416, 419, 423, 424) and every foreground shell dies with it.

Two mitigations were measured. Wrapping each model in its own `systemd-run --user --scope -p
MemoryHigh=… -p MemoryMax=…` contains the kill to that scope, but at `MemoryMax=18G` the 0.5B model's
`verify` itself gets `Terminated` (exit 143, devq 424) — the per-model ceiling has to sit between the
small models' real peak and the session threshold, and the 3B/4B peaks are above any ceiling that
leaves the session alive. So the operator **deferred `qwen25_3b`, `llama32_3b`, `qwen3_4b`** from the
regression leg rather than keep losing sessions: **8/11 is the standing leg** (devq 416 + 425, eight
models ≤ 1.7B, scope-wrapped at `MemoryHigh=16G MemoryMax=18G`, small-first). The three are
**deferred, not failed** — the last time they ran, 11/11 passed on the 28(a) compiler (devq 402) —
and they are owed one pass each, singly, in a window where nothing else holds memory.

## Locks

Two deliberately different inodes — do not unify them:

- `/tmp/mlir-air-npu.lock` — the agent/human convention, always `flock -x -w 1800`
- `/tmp/npu.lock` — taken internally by `KernelCache` and the lit suites. Taking it from a wrapper
  deadlocks them.

## Git

Work is on `exper/transformer-layer-execution-studies`, **not pushed**. A `pre-commit` hook is
installed with no `.pre-commit-config.yaml` on this branch, so commits need
`PRE_COMMIT_ALLOW_NO_CONFIG=1` — or run `pre-commit uninstall`.

## `[2026-08-11]` The wider lit tree carries three pre-existing failures — not study regressions

The full `build-xrt/programming_examples` lit tree (361 tests) is NOT a study gate; the standing
regression legs are the transformer-layer suite (`ninja check-programming-examples-transformer-layer`)
and the ten-model `make verify`. A whole-tree sweep on 2026-08-11 — both legs green the same
day — found three failures, reproducible 2/2, all outside every standing gate leg and all
predating that day's work (none of the three directories has a 2026-08 commit):

- `llms/llama32_1b_int4/multi_launch_builder/run_o_gemv_ffn_int4_fused_npu2_peano.lit` — kernel
  compile fails: `aie_api/adf/stream.hpp` includes `adf.h`, a Chess-only header, under Peano
  (the mlir-aie 1.4.0 include-layout family above). The int4 *model* itself verifies (10/10 run
  the same day includes it); only this sub-example's lit is red.
- `conv2d_14x14/run_npu2_makefile_peano.lit` — device run completes, `Output 0 does not meet
  expected output` (an NPU1-port example).
- `matrix_vector_multiplication/bf16/run_npu2_makefile_peano.lit` — the `profile` target dies
  with `std::runtime_error` (abort 134).

If a whole-tree sweep is ever promoted to a gate, these three need owners first; until then a
red whole-tree run with exactly these three is the known baseline, not a regression.

## `[2026-08-11]` Which toolchain tree a run actually tests

Two resolution paths coexist and they diverge whenever `mlir/` changes:

- **The lit suites test the BUILD tree**: `build-xrt/programming_examples/lit.site.cfg.py` puts
  `build-xrt/python` on PYTHONPATH (sandbox python, sandbox aiecc from `mlir_aie`'s wheel). A
  compiler fix gates through the suite as soon as `ninja -C build-xrt` has relinked
  `build-xrt/python/air/_mlir_libs/` — no install refresh needed. Under the build tree the
  backend resolves `aircc` package-relative (…`/bin/aircc` beside `python/`), so an ad-hoc run
  with `PYTHONPATH=build-xrt/python` also needs `build-xrt/bin` on PATH.
- **The probes and models test the INSTALL tree**: `agents/probes/*` default to
  `install-xrt/bin` / `install-xrt/python`, and the llms verify path resolves the installed
  backend. These see a compiler fix only after the operator's `ninja -C build-xrt install`.

A fix that is green in one path and untested in the other is exactly how "the gate passed" and
"the probe still crashes" can both be true on the same day.

~~**`[2026-08-12]` The two trees are currently four days apart**~~ **`[2026-08-12]` REFRESHED — the
two trees now agree, and the divergence above is closed.** `ninja -C build-xrt install` ran with no
compile or link steps (`build-xrt` was already current, so it was a copy);
`install-xrt/bin/air-opt`, `install-xrt/bin/aircc` and `install-xrt/python/air/_mlir_libs/_air*.so`
are all **2026-08-11 13:28**, matching `build-xrt`. Both of the resident tail's compiler fixes —
6a's fusion correction and 6b's shim-BD pacing — are now in every probe and model run as well as in
every lit suite. **Verified by artifact rather than by timestamp**:
`agents/probes/probe_fuse_channels_sibling_nests.py --nests 4 --tries 5` resolves
`install-xrt/bin/air-opt` directly (line 222) and reports `{'ok': 5}` / "does not reproduce" where
the pre-refresh binary was a deterministic 5/5 SEGV; its aircc leg succeeded off the same tree.

**The check itself is permanent and worth keeping**, because the trees diverge again the moment
`mlir/` changes: use `ls -l` on the two binaries, **never `cmp`**. The install step rewrites
RUNPATH, so the bytes always differ and a `cmp` difference proves nothing about staleness. And
prefer an artifact to a timestamp where one is cheap — a matching mtime proves a file was copied,
not that the copy behaves. Also: pairing ironenv's `aiecc`
with the build tree's air bindings produces `error: expected attribute value` parsing
`npu.air.mlir` — a version-mismatch artifact of the ad-hoc env, not a compiler bug; the suites'
sandbox aiecc is the referee.

### `[2026-08-11]` The four things a bare shell is missing, and why the first one costs the most

The lit config carries these; a hand-run shell does not. Collected from four failed submissions
during the Phase F device legs plus one probe re-run at merge review:

| missing | symptom |
|---|---|
| `sandbox/bin` on PATH (for `aiecc`) | `AirBackendError: aircc compilation failed:` with the real line, `Error: could not find aiecc in PATH`, **below the first line** |
| `PEANO_INSTALL_DIR` | aiecc drops off the peano path and dies at the per-core link edge (`chesslinked_{0}.ll`) |
| `MLIR_AIE_INSTALL_DIR` | `aie-opt`-on-PATH does not resolve to the wheel's include directory |
| `/opt/xilinx/xrt/python` on PYTHONPATH (for `pyxrt`) | **everything compiles**, the xclbin builds, and the FIRST DISPATCH raises `ModuleNotFoundError` — minutes in, looking like a runtime bug |

The full incantation, which is what every ad-hoc probe run should start from:

```
PATH=$PWD/sandbox/bin:$PWD/build-xrt/bin:$PATH \
PYTHONPATH=$PWD/build-xrt/python:/opt/xilinx/xrt/python \
PEANO_INSTALL_DIR=$PWD/sandbox/lib/python3.12/site-packages/llvm-aie \
MLIR_AIE_INSTALL_DIR=$PWD/sandbox/lib/python3.12/site-packages/mlir_aie \
  ./sandbox/bin/python agents/probes/<probe>.py
```

**The trap is the diagnostic, not the variable.** A probe that summarizes an exception by its
first line reports `aircc compilation failed:` and nothing else, which reads as *the design
refused* when it means *the toolchain was never found*. A green-looking probe suite and an
env-broken one are distinguishable only by reading the whole exception — so when an arm reports
zero dumps in **0.0 s**, suspect the environment before the IR. Seen twice in one day: once as
three failed devq submissions, once as a probe re-run that looked like a design regression and
was not.
