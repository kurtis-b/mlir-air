# Build And Test Overlay

Canonical testing guidance is `docs/testing.md`; build entry points are indexed from `docs/building.md`.

## Focused Selection

Choose the smallest check that covers the change:

- One `ninja` target for build-system or compiler changes.
- One lit file or directory for pass and lowering changes.
- One compile-only programming example for backend pipeline changes.
- One smoke script from `agents/scripts/doctor.sh` for environment validation.

Broad suites such as all XRT tests or full programming examples are appropriate only when the user asks, a shared contract changed, or focused tests cannot cover the risk.

## Common Commands

```bash
ninja -C build check-air-mlir
ninja -C build check-air-python
ninja -C build check-air-cpp
lit -sv --time-tests mlir/test/Conversion/AIRToROCDL
lit -sv --time-tests test/xrt/01_air_to_npu
```

For GPU-only builds, use the GPU build directory and GPU lit paths. For Ryzen/AIE builds, keep hardware execution separate from compile-only validation.

## Dirty Worktree Safety

Before changing generated tests or expected output, check whether the file is already modified. Work with user changes in place; do not revert them unless explicitly requested.
