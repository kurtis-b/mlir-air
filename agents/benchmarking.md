# Benchmarking Overlay

Canonical runtime and trace docs are `docs/AIRRunner.md` and `docs/trace.md`; GEMM pipeline context is in `docs/GEMMCaseStudy.md`.

## Output Hygiene

Keep benchmark outputs, traces, logs, and temporary binaries out of commits unless the user asks to preserve them. Prefer `/tmp` or a clearly ignored artifact directory.

When reporting results, separate:

- Compile success or failure.
- Runtime execution success or hardware availability.
- Validation correctness.
- Timing, trace, or profiling numbers.

Do not mix compile-time smoke checks with performance claims.

## Incremental Benchmark Flow

1. Confirm the environment and build profile with `agents/scripts/doctor.sh env`.
2. Rebuild only the changed target or install tree needed for the benchmark.
3. Run a compile-only or validation pass first.
4. Run the benchmark with explicit problem sizes, target device, and trace settings.
5. Record enough provenance to reproduce the run: git head, dirty state, dependency heads, CMake options, and hardware/runtime state.
