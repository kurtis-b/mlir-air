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
NPU-free any more: it carries the Phase B runlist gate and every Phase C operator gate, 13 tests
as of C4.

If you want the PR-safe subset, the individual make targets are still hardware-free:

```bash
cd programming_examples/transformer_layer
make compile        # Peano objects and their symbol lists
make seam-tests     # BO pooling and runlist aggregation rules
make registry-plan  # what the sweep would measure
make registry-writer-tests
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
| RAM | 31 GB — 3B/4B verify needs `verify_runner.py`'s subprocess split |
| Concurrent `hw_context` ceiling | **32**; 33 fails with `DRM_IOCTL_AMDXDNA_CREATE_HWCTX err=-2` |
| Chassis | **laptop** — lid close suspends. Wrap long runs in `systemd-inhibit`. On battery, GNOME suspends after 15 min idle; on AC it does not. |
| Logout | Background runs survive (`KillUserProcesses=no`) |

## Locks

Two deliberately different inodes — do not unify them:

- `/tmp/mlir-air-npu.lock` — the agent/human convention, always `flock -x -w 1800`
- `/tmp/npu.lock` — taken internally by `KernelCache` and the lit suites. Taking it from a wrapper
  deadlocks them.

## Git

Work is on `exper/transformer-layer-execution-studies`, **not pushed**. A `pre-commit` hook is
installed with no `.pre-commit-config.yaml` on this branch, so commits need
`PRE_COMMIT_ALLOW_NO_CONFIG=1` — or run `pre-commit uninstall`.
