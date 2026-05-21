# Compiler Development Overlay

Use `docs/AIRComputeModel.md` for AIR semantics, `docs/AIRAsyncConcurrency.md` for async/dependency behavior, and backend-specific setup docs for lowering targets.

## Checklist

- Identify the dialect, pass, conversion, runtime, or driver surface being changed.
- Check the matching `CMakeLists.txt` before adding new files.
- Add or update focused lit tests near the pass or conversion being changed.
- Prefer structured MLIR parser/rewriter APIs over string handling.
- Keep GPU-only and AIE-only code behind the existing build options.
- Verify with the smallest relevant `ninja` target and lit file.

## Build Hints

- Pass or dialect changes usually need `ninja -C build air-opt`.
- Driver changes usually need `ninja -C build aircc`.
- Python binding changes usually need `ninja -C build check-air-python` or a focused Python test.
- GPU lowering changes should be checked against a GPU-enabled build.
- AIE/NPU lowering changes should be checked against a Ryzen/AIE build and a compile-only smoke before hardware execution.
