# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernel compilation cache, profiling, and air_project utilities."""

import json
import shutil
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16


def prepare_air_project(quant: str = "bf16"):
    """Clean and prepare the air_project/ directory for a fresh compilation.

    aircc defaults to 'air_project/' as its working directory. Sequential
    compilations leave stale artifacts that corrupt subsequent kernels.
    This method wipes the directory, compiles all external C++ kernels from
    source, and copies them to air_project/.

    Args:
        quant: "bf16" or "awq". When "awq", also compiles + stages
            `mv_int4_bf16.o` so the int4 decode ELFs can link it.
    """
    air_proj = Path("air_project")
    if air_proj.exists():
        shutil.rmtree(air_proj)
    air_proj.mkdir(parents=True, exist_ok=True)

    # Compile external kernels from source (not stale .o copies)
    from shared.infra.external_kernels import compile_all_external_kernels

    compile_all_external_kernels(quant=quant)

    # Copy compiled .o files to air_project/ for aiecc to find. Must include
    # every external symbol referenced by `link_with` in the kernel modules:
    # - mv.o            : K=2048 GEMVs (rms_gemv_rope, o_gemv_ffn, lm_head_gemv)
    # - mv_bf16.o       : 2-tile matvec+add (o_gemv_ffn stages 1 and 3)
    # - rope.o          : RoPE (prefill + decode rms_*_rope)
    # - silu_and_mul.o  : SwiGLU (prefill o_ffn, decode o_gemv_ffn)
    # - attn.o          : flash attention (prefill, when --cpu-attn=False)
    # - attn_npu2.o     : flash attention NPU2 variant alias
    # - mv_int4_bf16.o  : int4-AWQ GEMV (decode int4 ELFs only, when quant=awq)
    obj_names = [
        "silu_and_mul.o",
        "rope.o",
        "attn.o",
        "attn_npu2.o",
        "mv.o",
        "mv_bf16.o",
        "attn_decode_npu2.o",
        # bare mm.o: legacy f32-out / single-variant ELFs that name their own
        # object. The SUFFIXED variants are globbed below rather than listed.
        "mm.o",
    ]
    if quant == "awq":
        obj_names.append("mv_int4_bf16.o")

    # External GEMM microkernel variants (prefill o_ffn / rms_gemms_rope), named
    # per (tile_m, tile_n) by `gemm_builder.gemm_variant_names` -- `mm_m32n128.o`,
    # `mm_m64n96.o`, and so on. GLOBBED, not enumerated: tile_n takes four values
    # across the registry, so there are up to eight of these and which ones exist
    # depends on which shapes the caller resolved. A fixed list was correct while
    # the name came from the method alone (two objects) and is a silent
    # link-failure waiting to happen now that it does not -- an ELF whose object
    # never reached air_project/ fails inside aiecc, several frames from the list
    # that omitted it.
    obj_names.extend(sorted(p.name for p in Path(".").glob("mm_m*.o")))
    for obj_name in obj_names:
        src = Path(obj_name)
        if src.exists():
            shutil.copy2(src, air_proj / obj_name)


class Profiler:
    """Tracks per-kernel and per-layer execution times."""

    def __init__(self, enabled=False):
        self.enabled = enabled
        self.compile_times = {}  # name -> seconds
        self.kernel_times = {}  # NPU XRT call: name -> list of seconds
        self.cpu_times = {}  # CPU op: name -> list of seconds
        self.layer_times = []  # list of (layer_idx, seconds)
        self.kernel_breakdowns = (
            {}
        )  # name -> list of {write_ms, kernel_ms, read_ms, ...}

    def record_compile(self, name, duration):
        if self.enabled:
            self.compile_times[name] = duration

    def record_kernel(self, name, duration):
        if self.enabled:
            self.kernel_times.setdefault(name, []).append(duration)

    def record_cpu(self, name, duration):
        """Record a CPU host-side operation's wall time. Use for things like
        embed lookup, KV-cache extract, CPU attention fallback, final RMSNorm
        — anything that is not an `xrt.run()` but consumes inference wall
        time. Reported in a separate section from NPU XRT calls so the two
        are easy to compare."""
        if self.enabled:
            self.cpu_times.setdefault(name, []).append(duration)

    def record_breakdown(
        self, name, write_ms, kernel_ms, read_ms, n_written, bytes_written, n_readback
    ):
        if self.enabled:
            self.kernel_breakdowns.setdefault(name, []).append(
                {
                    "write_ms": write_ms,
                    "kernel_ms": kernel_ms,
                    "read_ms": read_ms,
                    "n_written": n_written,
                    "bytes_written": bytes_written,
                    "n_readback": n_readback,
                }
            )

    def start_layer(self):
        if self.enabled:
            return time.perf_counter()
        return None

    def end_layer(self, layer_idx, t0):
        if self.enabled and t0 is not None:
            self.layer_times.append((layer_idx, time.perf_counter() - t0))

    def time_cpu(self, name):
        """Context manager: `with prof.time_cpu("embed_lookup"): ...`
        Records the elapsed wall time as a CPU op named `name`. Safe to
        use whether enabled or disabled (zero overhead when disabled)."""
        prof = self

        class _Ctx:
            def __enter__(self_inner):
                self_inner.t0 = time.perf_counter() if prof.enabled else None
                return self_inner

            def __exit__(self_inner, *exc):
                if self_inner.t0 is not None:
                    prof.record_cpu(name, time.perf_counter() - self_inner.t0)
                return False

        return _Ctx()

    def per_token_walls_ms(self, n_layers):
        """Sum every consecutive `n_layers` layer-time entries into one
        per-token wall (in ms). Returns [] if not enabled or no data.
        Used by the dataflow summary to expose decode slowdown trends."""
        if not self.enabled or not self.layer_times:
            return []
        if len(self.layer_times) % n_layers != 0:
            # Shouldn't happen in a clean run; bail rather than mis-bucket.
            return []
        out = []
        for tok_start in range(0, len(self.layer_times), n_layers):
            chunk = self.layer_times[tok_start : tok_start + n_layers]
            out.append(sum(t for _, t in chunk) * 1000.0)
        return out

    def report(self):
        if not self.enabled:
            return

        print(f"\n{'='*60}")
        print("PROFILING REPORT")
        print(f"{'='*60}")

        # Top-level phase summary: total wall time attributed to NPU XRT
        # calls vs CPU host ops vs the layer envelope. Sums won't add up
        # exactly (layer envelope is the wall budget; NPU + CPU are the
        # accounted-for parts inside it; remainder is python scheduling /
        # numpy view setup / loop overhead). Useful as a sanity check.
        if self.kernel_times or self.cpu_times or self.layer_times:
            npu_total_ms = sum(t * 1000 for v in self.kernel_times.values() for t in v)
            cpu_total_ms = sum(t * 1000 for v in self.cpu_times.values() for t in v)
            layer_total_ms = sum(t * 1000 for _, t in self.layer_times)
            npu_count = sum(len(v) for v in self.kernel_times.values())
            cpu_count = sum(len(v) for v in self.cpu_times.values())
            print(f"\n--- Wall-Time Attribution ---")
            if npu_count:
                print(
                    f"  NPU XRT calls         {npu_total_ms:9.2f}ms  ({npu_count} calls)"
                )
            if cpu_count:
                print(
                    f"  CPU host ops          {cpu_total_ms:9.2f}ms  ({cpu_count} calls)"
                )
            if self.layer_times:
                accounted = npu_total_ms + cpu_total_ms
                # CPU ops happen both inside and outside the layer envelope;
                # so layer_total_ms is the inside-layer wall budget, and the
                # remainder vs (NPU+CPU) inside layers is python overhead.
                print(
                    f"  Layer-loop wall       {layer_total_ms:9.2f}ms  "
                    f"({len(self.layer_times)} layer-invocations)"
                )

        if self.compile_times:
            print(f"\n--- Compilation Phase ---")
            total_compile = 0
            for name, t in sorted(self.compile_times.items()):
                print(f"  {name:40s} {t:8.1f}s")
                total_compile += t
            print(
                f"  {'Total compilation':40s} {total_compile:8.1f}s ({len(self.compile_times)} kernels)"
            )

        if self.layer_times:
            # Group by layer_idx. Prefill: each idx appears once -> one row per
            # layer. Decode: each idx appears once per token -> aggregate with
            # avg / min / max / count.
            from collections import defaultdict

            grouped = defaultdict(list)
            for idx, t in self.layer_times:
                grouped[idx].append(t * 1000.0)  # ms
            multi_invocation = any(len(v) > 1 for v in grouped.values())
            print(f"\n--- Per-Layer Execution ---")
            if multi_invocation:
                for idx in sorted(grouped):
                    ts = grouped[idx]
                    print(
                        f"  Layer {idx:3d}: avg={sum(ts)/len(ts):7.2f}ms  "
                        f"min={min(ts):7.2f}ms  max={max(ts):7.2f}ms  (x{len(ts)})"
                    )
            else:
                for idx in sorted(grouped):
                    print(f"  Layer {idx:3d}: {grouped[idx][0]:7.2f}ms")
            total_ms = sum(t * 1000.0 for _, t in self.layer_times)
            print(f"  {'Total layer-time':40s} {total_ms:8.2f}ms")

        if self.kernel_times:
            print(f"\n--- NPU XRT Call Breakdown (avg per invocation) ---")
            total_avg = 0
            for name, times in sorted(self.kernel_times.items()):
                times_ms = [t * 1000.0 for t in times]
                avg = sum(times_ms) / len(times_ms)
                total_avg += avg * len(times_ms)
                count = len(times_ms)
                print(
                    f"  {name:40s} avg={avg:7.2f}ms  "
                    f"min={min(times_ms):7.2f}ms  max={max(times_ms):7.2f}ms  (x{count})"
                )
            if self.layer_times:
                n_layers = len(self.layer_times)
                print(f"  {'Total kernel time':40s} {total_avg:8.2f}ms")
                print(
                    f"  {'Avg per layer (kernel time)':40s} {total_avg/n_layers:8.2f}ms"
                )

        if self.cpu_times:
            print(f"\n--- CPU Op Breakdown (avg per invocation) ---")
            total_cpu_ms = 0
            for name, times in sorted(self.cpu_times.items()):
                times_ms = [t * 1000.0 for t in times]
                avg = sum(times_ms) / len(times_ms)
                total_cpu_ms += avg * len(times_ms)
                count = len(times_ms)
                print(
                    f"  {name:40s} avg={avg:7.2f}ms  "
                    f"min={min(times_ms):7.2f}ms  max={max(times_ms):7.2f}ms  (x{count})"
                )
            print(f"  {'Total CPU op time':40s} {total_cpu_ms:8.2f}ms")

        if self.kernel_breakdowns:
            print(f"\n--- Fine-Grained NPU Breakdown (avg per invocation) ---")
            print(
                f"  Three-segment timing of each XRT call:\n"
                f"    BO Write = host→DDR memcpy of dynamic inputs (weights\n"
                f"               pre-loaded once via static_input_indices)\n"
                f"    NPU Run  = xrt.run.start() + wait() — actual NPU exec\n"
                f"    BO Read  = numpy view construction (zero-copy, ~0)"
            )
            print(
                f"  {'Kernel':20s} {'BO Write':>10s} {'NPU Run':>10s} {'BO Read':>10s} {'Total':>10s}  {'Written':>8s} {'Read':>6s}"
            )
            print(
                f"  {'\u2500'*20} {'\u2500'*10} {'\u2500'*10} {'\u2500'*10} {'\u2500'*10}  {'\u2500'*8} {'\u2500'*6}"
            )
            total_write = total_kernel = total_read = 0
            for name in sorted(self.kernel_breakdowns.keys()):
                entries = self.kernel_breakdowns[name]
                n = len(entries)
                avg_w = sum(e["write_ms"] for e in entries) / n
                avg_k = sum(e["kernel_ms"] for e in entries) / n
                avg_r = sum(e["read_ms"] for e in entries) / n
                avg_total = avg_w + avg_k + avg_r
                avg_bytes = sum(e["bytes_written"] for e in entries) / n
                avg_nw = sum(e["n_written"] for e in entries) / n
                avg_nr = sum(e["n_readback"] for e in entries) / n
                total_write += avg_w * n
                total_kernel += avg_k * n
                total_read += avg_r * n
                mb = avg_bytes / 1024 / 1024
                print(
                    f"  {name:20s} {avg_w:8.2f}ms {avg_k:8.2f}ms {avg_r:8.2f}ms {avg_total:8.2f}ms"
                    f"  {mb:6.1f}MB {avg_nr:4.0f}bo  (x{n})"
                )
            grand_total = total_write + total_kernel + total_read
            print(
                f"  {'\u2500'*20} {'\u2500'*10} {'\u2500'*10} {'\u2500'*10} {'\u2500'*10}"
            )
            print(
                f"  {'TOTAL':20s} {total_write:8.1f}ms {total_kernel:8.1f}ms {total_read:8.1f}ms {grand_total:8.1f}ms"
            )
            if grand_total > 0:
                print(
                    f"  {'%':20s} {total_write/grand_total*100:7.0f}%  {total_kernel/grand_total*100:7.0f}%  {total_read/grand_total*100:7.0f}%"
                )


class KernelCache:
    """Pre-compiles unique kernel binaries and caches them for reuse.

    The key insight: XRTCompileArtifact is a simple dataclass with
    (output_binary, kernel, insts) paths. backend.load(artifact) reads from
    these paths -- no compilation context needed. So we compile each unique
    kernel once, save the binary to a cache directory, and construct artifacts
    from saved paths at runtime.
    """

    # Manifest file stores artifact metadata for --run-only mode
    MANIFEST_FILE = "manifest.json"

    def __init__(self, cache_dir=None, verbose=False, profiler=None):
        if cache_dir is None:
            cache_dir = Path(__file__).resolve().parent / "kernel_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.profiler = profiler or Profiler()
        self.artifacts = {}  # name -> XRTCompileArtifact
        self.launch_counts = {}  # name -> {"air_launches":.., "herd_launches":..}
        self._loaded = {}  # name -> (backend, invoker) for XRT context reuse
        #: How many times the ARRAY was configured, and how many kernels were
        #: bound onto a configuration already standing. See
        #: `reconfiguration_counts`.
        self.context_loads = 0
        self.kernel_attaches = 0
        self._cached_bos = {}  # name -> list of xrt.bo for BO reuse
        self._pools = {}  # PoolPlan.signature -> BoPool, for run_sequence
        self._instr_synced = set()  # artifact names whose instruction BO is resident
        # Pool of BOs shared across bo_keys (see shared_nonstatic in
        # load_and_run). Keyed by (name, arg_index, size_bytes) so every
        # per-layer bo_key of a kernel reuses ONE transient buffer instead of
        # allocating its own — collapses the prefill activation-scratch
        # footprint from N_layers copies to 1.
        self._shared_bos = {}

    def _log(self, msg):
        if self.verbose:
            print(f"  [KernelCache] {msg}")

    def _save_manifest(self):
        """Save artifact metadata so --run-only can reconstruct artifacts."""
        manifest = {}
        for name, art in self.artifacts.items():
            manifest[name] = {
                "output_binary": str(art.output_binary),
                "kernel": art.kernel,
                "insts": str(art.insts) if art.insts else None,
                # air.launch / air.herd counts for the dispatch vector. They come
                # from the MLIR module, which --run-only does not have, so they
                # are persisted here rather than recomputed.
                "launches": self.launch_counts.get(name),
            }
        manifest_path = self.cache_dir / self.MANIFEST_FILE
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        self._log(f"Saved manifest with {len(manifest)} entries")

    def load_manifest(self):
        """Load artifact metadata from a previous compilation.

        Returns True if manifest was loaded successfully, False otherwise.
        """
        from air.backend.xrt import XRTCompileArtifact

        manifest_path = self.cache_dir / self.MANIFEST_FILE
        if not manifest_path.exists():
            return False

        with open(manifest_path) as f:
            manifest = json.load(f)

        for name, info in manifest.items():
            binary_path = info["output_binary"]
            if not Path(binary_path).exists():
                print(f"  WARNING: cached binary not found: {binary_path}")
                return False
            self.artifacts[name] = XRTCompileArtifact(
                binary_path, info["kernel"], info["insts"]
            )
            # Absent in manifests written before the dispatch vector landed; a
            # missing entry costs the launch counts, not the run.
            if info.get("launches"):
                self.launch_counts[name] = info["launches"]

        self._log(f"Loaded manifest with {len(self.artifacts)} entries")
        return True

    def compile_and_cache(
        self, name, mlir_module, backend_kwargs, output_binary_name="air"
    ):
        """Compile kernel and save binary to cache.

        Args:
            name: Unique name for this kernel config
            mlir_module: MLIR module to compile
            backend_kwargs: Dict of kwargs for XRTBackend constructor
            output_binary_name: Base name for output binary
        """
        from air.backend.xrt import XRTBackend
        from shared.infra.dispatch import LaunchCounts

        self.launch_counts[name] = LaunchCounts.from_module(mlir_module).as_dict()
        self._log(f"Compiling {name}...")
        # Int4 ELFs need the AWQ GEMV micro-kernel staged alongside the bf16
        # objects. Detect from kernel name so existing call sites don't need
        # an extra flag.
        quant = "awq" if "int4" in name else "bf16"
        prepare_air_project(quant=quant)
        backend = XRTBackend(**backend_kwargs)

        t0 = time.time()
        artifact = backend.compile(mlir_module, output_binary_name=output_binary_name)
        compile_time = time.time() - t0
        self.profiler.record_compile(name, compile_time)

        # Copy binary to cache with unique name
        src_binary = Path(artifact.output_binary)
        ext = src_binary.suffix  # .xclbin, .elf, or .txn
        cached_binary = self.cache_dir / f"{name}{ext}"
        shutil.copy2(src_binary, cached_binary)

        # Copy instructions file if present (xclbin mode)
        cached_insts = None
        if artifact.insts and Path(artifact.insts).exists():
            cached_insts = str(self.cache_dir / f"{name}.insts.bin")
            shutil.copy2(artifact.insts, cached_insts)

        from air.backend.xrt import XRTCompileArtifact

        self.artifacts[name] = XRTCompileArtifact(
            str(cached_binary), artifact.kernel, cached_insts
        )
        backend.unload()

        print(f"  Compiled {name}: {compile_time:.1f}s -> {cached_binary.name}")

    def compile_shared_xclbin(self, entries, output_binary_name="air"):
        """Compile several modules into ONE xclbin, N instruction streams.

        `[2026-08-09]` The build half of `offload`'s reconfiguration-minimizing
        mechanism. Each module is compiled in turn with `xclbin_input` pointing at
        the previous one's xclbin, so aircc packages every kernel into a single
        file; the LAST xclbin written is the one holding them all, and every
        artifact is recorded against it. The kernels then share one array
        configuration and differ only in their instruction stream.

        `compile_and_cache` cannot express this: it caches each compile's own
        output, and in a chain every output but the last is missing kernels.

        Args:
            entries: ordered list of `(name, mlir_module, backend_kwargs)`. Each
                `backend_kwargs` MUST carry `output_format="xclbin"` and a
                distinct `kernel_name`, `instance_name` AND `kernel_id` -- see the
                failure modes above and in `XRTBackend.attach_kernel`.
                `xclbin_input` is supplied here and must not be set by the caller.
            output_binary_name: base name aircc writes per compile.

        The compile ORDER is the chain order and is therefore load-bearing; it is
        taken from `entries` rather than from dict iteration so it is explicit.
        """
        from air.backend.xrt import XRTBackend, XRTCompileArtifact
        from shared.infra.dispatch import LaunchCounts

        if not entries:
            return
        # Three identifiers must be distinct per stream, and they fail three
        # different ways: a duplicate `kernel_name` makes xclbinutil REFUSE the
        # merge ("Kernel name already exists in the EMBEDDED_METADATA section"),
        # a duplicate `instance_name` makes the kernel lookup's SUBSTRING match
        # silently return the wrong program, and a duplicate `kernel_id` makes
        # the second kernel run against the first's array configuration. Only the
        # first is loud, so all three are checked here.
        seen_names, seen_instances, seen_ids = {}, {}, {}
        for name, _, kwargs in entries:
            if kwargs.get("output_format") != "xclbin":
                raise ValueError(
                    f"{name}: a shared xclbin needs output_format='xclbin', got "
                    f"{kwargs.get('output_format')!r}"
                )
            for field, seen in (
                ("kernel_name", seen_names),
                ("instance_name", seen_instances),
                ("kernel_id", seen_ids),
            ):
                value = kwargs.get(field)
                if not value:
                    raise ValueError(
                        f"{name}: a shared xclbin needs an explicit {field}. "
                        "Without it every stream takes the same default and the "
                        "second kernel runs against the first's configuration."
                    )
                if value in seen:
                    raise ValueError(
                        f"{name}: {field}={value!r} is already used by "
                        f"{seen[value]!r}. Streams sharing an xclbin must differ "
                        "in both; a collision fails silently at run time."
                    )
                seen[value] = name
            if kwargs.get("xclbin_input"):
                raise ValueError(
                    f"{name}: xclbin_input is supplied by compile_shared_xclbin; "
                    "setting it here would break the chain."
                )

        prepare_air_project(quant="bf16")
        chained, kernels = "", {}
        for i, (name, module, kwargs) in enumerate(entries):
            self.launch_counts[name] = LaunchCounts.from_module(module).as_dict()
            self._log(f"Compiling {name} into the shared xclbin...")
            backend = XRTBackend(**dict(kwargs, xclbin_input=chained))
            t0 = time.time()
            # Each link needs its OWN output name. aircc writes relative to cwd,
            # so reusing one base name makes link i+1's input and output the same
            # file and xclbinutil refuses: "The following output file is also used
            # for input". The chain is what makes these distinct compiles.
            artifact = backend.compile(
                module, output_binary_name=f"{output_binary_name}_shared_{i}"
            )
            self.profiler.record_compile(name, time.time() - t0)
            # aircc writes relative to cwd and every compile uses the same base
            # name, so the chain input must be an absolute path captured now.
            chained = str(Path(artifact.output_binary).resolve())
            cached_insts = None
            if artifact.insts and Path(artifact.insts).exists():
                cached_insts = str(self.cache_dir / f"{name}.insts.bin")
                shutil.copy2(artifact.insts, cached_insts)
            # The kernel is found in the xclbin by INSTANCE name, not by the
            # `MLIR_AIE` default `compile` records for xclbin output.
            kernels[name] = (kwargs["instance_name"], cached_insts)
            backend.unload()
            print(f"  Compiled {name} -> shared xclbin ({kwargs['instance_name']})")

        shared = self.cache_dir / f"{entries[-1][0]}_shared.xclbin"
        shutil.copy2(chained, shared)
        for name, (instance, insts) in kernels.items():
            self.artifacts[name] = XRTCompileArtifact(str(shared), instance, insts)
        print(
            f"  Shared xclbin: {shared.name} holding {len(kernels)} kernels "
            f"({', '.join(i for i, _ in kernels.values())})"
        )

    def ensure_loaded(self, name, backend_kwargs):
        """Load an artifact's XRT context once and cache it.

        Artifacts that share an `output_binary` share ONE context: the first
        loads it, the rest bind their own kernel and instruction BO onto the
        standing context via `XRTBackend.attach_kernel`. That is what makes N
        instruction streams under one xclbin cost one array configuration
        instead of N -- and with distinct artifacts it changes nothing, because
        no two of them compare equal.

        Lock path note: `/tmp/npu.lock` is intentionally a different inode from
        the `/tmp/mlir-air-npu.lock` file the project's outer `flock` convention
        uses. Both layers use BSD flock(2), so sharing one inode would make this
        inner lock self-deadlock against an outer
        `flock /tmp/mlir-air-npu.lock make run`. Keep them separate.
        """
        import filelock
        from air.backend.xrt import XRTBackend

        if name not in self.artifacts:
            raise RuntimeError(
                f"Kernel '{name}' not found in cache. "
                f"Available: {list(self.artifacts.keys())}"
            )
        if name not in self._loaded:
            binary = self.artifacts[name].output_binary
            shares_with = [
                other
                for other, art in self.artifacts.items()
                if other in self._loaded and art.output_binary == binary
            ]
            host = next(
                (
                    other
                    for other, (backend, _) in self._loaded.items()
                    if getattr(backend, "loaded_binary", None) == binary
                ),
                None,
            )
            # A stale `air` install is the reason this would be reachable, and it
            # must not pass quietly: `attach_kernel`/`loaded_binary` live in
            # `python/air/backend/xrt.py`, and the runtime imports from
            # `install-xrt/`. Without the check, an un-installed change degrades
            # to one context PER ARTIFACT while every printed line still claims
            # one xclbin -- the mode would report a reconfiguration reduction it
            # did not make.
            if shares_with and host is None:
                raise RuntimeError(
                    f"'{name}' shares an xclbin with {shares_with} but no loaded "
                    f"backend reports it. The installed air backend is missing "
                    f"`loaded_binary`/`attach_kernel` -- almost certainly a stale "
                    f"install: edits to python/air/backend/xrt.py need an install "
                    f"step before the runtime sees them. Sharing an xclbin without "
                    f"sharing the context would configure the array once per "
                    f"artifact while the mode claims once per layer."
                )
            if host is not None:
                backend = self._loaded[host][0]
                with filelock.FileLock("/tmp/npu.lock"):
                    attached = backend.attach_kernel(self.artifacts[name])
                self._loaded[name] = (attached, None)
                self.kernel_attaches += 1
                self._log(f"Attached {name} to {host}'s context (shared xclbin)")
            else:
                backend = XRTBackend(**backend_kwargs)
                with filelock.FileLock("/tmp/npu.lock"):
                    invoker = backend.load(self.artifacts[name])
                self._loaded[name] = (backend, invoker)
                self.context_loads += 1
                self._log(f"Loaded {name} (XRT context cached)")
        return self._loaded[name]

    def reconfiguration_counts(self):
        """`(context_loads, kernel_attaches)` since this cache was built.

        `[2026-08-09]` The observable behind `offload`'s reconfiguration claim,
        and it is deliberately NOT one of the six dispatch-vector fields: the
        vector is identical whether the mode loads one context or thirty,
        because it makes one `run_sequence` call per GEMM either way. What
        separates the two is how often the ARRAY is configured, which is this.

        A context load configures the array; an attach binds another kernel and
        its instruction stream onto a configuration already standing. Evicting a
        context (as the ELF path must, see `pattern/offload`'s
        `_evict_context`) makes the next use count as another load, which is
        what makes the two paths distinguishable here rather than only in prose.
        """
        return self.context_loads, self.kernel_attaches

    def pool_for(self, plan, alloc, sync_to_device, sync_from_device):
        """The `BoPool` backing one `PoolPlan`, created on first use.

        Keyed by `plan.signature` — the sequence's *value* identity — so a caller
        that re-runs the same sequence keeps its slots and its resident static
        weights. That is the point of the pool: the second and subsequent
        invocations skip both the allocation and the weight upload.

        Not keyed by `id(plan)`. Every `run_sequence` builds a fresh plan, so an
        id key would allocate a new pool and re-upload every weight on each call,
        and — because CPython recycles the id of the plan it just freed — could
        instead hand a stale pool to an unrelated sequence, whose slots are sized
        and banked for buffers that are no longer there.
        """
        from shared.infra.bo_pool import BoPool

        key = plan.signature
        if key not in self._pools:
            self._pools[key] = BoPool(alloc, sync_to_device, sync_from_device)
        return self._pools[key]

    def _mark_instr_synced(self, name):
        """True the first time an artifact's instruction BO needs syncing (D6)."""
        if name in self._instr_synced:
            return False
        self._instr_synced.add(name)
        return True

    def run_sequence(
        self,
        steps,
        specs,
        backend_kwargs,
        arrays,
        host_writes=None,
        require_single_submission=False,
    ):
        """Run a multi-step dispatch sequence as runlists, with BO pooling.

        Thin delegation to `shared.infra.dispatch.run_sequence`, which owns the
        aggregation rules and their footguns. Returns
        `(results, DispatchVector)`; the results are zero-copy views into pool
        memory and are overwritten by the next sequence.
        """
        from shared.infra.dispatch import run_sequence as _run_sequence

        return _run_sequence(
            self,
            steps,
            specs,
            backend_kwargs,
            arrays,
            host_writes=host_writes,
            require_single_submission=require_single_submission,
        )

    def load_and_run(
        self,
        name,
        backend_kwargs,
        *inputs,
        output_indices=None,
        static_input_indices=None,
        intermediate_indices=None,
        bo_key=None,
        shared_nonstatic=False,
    ):
        """Load cached kernel and execute with BO reuse.

        Three levels of caching to minimize per-invocation overhead:
        1. XRT context (device, xclbin, kernel) -- cached per kernel name
        2. Buffer Objects -- cached per kernel name, reused across calls
        3. Instruction BO sync -- done once on first call

        Args:
            name: Kernel name (must have been compiled first)
            backend_kwargs: Dict of kwargs for XRTBackend constructor
            *inputs: numpy arrays to pass to the kernel
            output_indices: Optional list of buffer indices to read back from
                device. If None, only the last buffer is read back (default).
                Use for multi-output kernels (e.g. attn_gemms: [2, 4, 6]).
            static_input_indices: Optional set of buffer indices that are static
                (e.g. weights, LUTs). On the first call for a given bo_key the BO is
                written; on subsequent calls the host->device sync is skipped because
                the kernel reads from the already-resident BO.
            intermediate_indices: Optional set of buffer indices that are
                intermediate (overwritten by kernel). Skips host->device sync.
            bo_key: Optional cache key for BO reuse. Calls sharing a bo_key reuse
                the same xrt.bo objects, which combined with static_input_indices
                enables write-once-read-many for weights. Default uses the kernel
                name (one BO set shared across all calls to that kernel).
            shared_nonstatic: When True, every arg index NOT in
                static_input_indices draws its BO from a process-wide pool
                (self._shared_bos) keyed by (name, arg_index, size) instead of
                the per-bo_key set. Static weight BOs stay per-layer (must
                persist); transient buffers (dynamic inputs, intermediates,
                outputs) are consumed within a single sequential kernel launch,
                so one shared buffer per arg suffices across all layers. Used by
                prefill to avoid holding N_layers copies of the seq_len-sized
                activation scratch.
                CONTRACT/FOOTGUN: returned outputs for shared (non-static) indices
                are zero-copy views into the shared BO and are OVERWRITTEN by the
                next load_and_run call that reuses that arg's shared buffer. Safe
                only for callers that consume each output before the next call
                (the prefill layer loop does). Any caller that PERSISTS outputs
                across calls (e.g. per-layer diagnosis capture) must copy them, or
                must not pass shared_nonstatic=True.

        Returns:
            Tuple of numpy arrays (all kernel outputs). With shared_nonstatic=True
            the non-static entries are views into shared BOs — see the footgun
            note above; copy them if you need to keep them past the next call.
        """
        import filelock
        import pyxrt as xrt

        # Level 1: Load backend on first call (XRT context reuse). See
        # ensure_loaded for why /tmp/npu.lock is a different inode from the
        # project's outer /tmp/mlir-air-npu.lock flock convention.
        backend, _ = self.ensure_loaded(name, backend_kwargs)

        # Level 2: Allocate BOs on first call, reuse on subsequent calls
        # bo_key allows separate BO sets for the same kernel (e.g., per-layer weights)
        _bo_key = bo_key if bo_key is not None else name
        sizes_in_bytes = [a.size * a.itemsize for a in inputs]
        is_elf = self.artifacts[name].output_binary.endswith(".elf")
        static_indices = set(static_input_indices or [])
        intermediate_set = set(intermediate_indices or [])

        shared_idx = (
            set(range(len(inputs))) - static_indices if shared_nonstatic else set()
        )

        def _alloc_bo(i, s):
            if is_elf:
                return xrt.ext.bo(backend.device, s)
            return xrt.bo(
                backend.device, s, xrt.bo.host_only, backend.kernel.group_id(i + 3)
            )

        first_call = _bo_key not in self._cached_bos
        if first_call:
            bos = []
            for i, s in enumerate(sizes_in_bytes):
                if i in shared_idx:
                    sk = (name, i, s)
                    bo = self._shared_bos.get(sk)
                    if bo is None:
                        bo = _alloc_bo(i, s)
                        self._shared_bos[sk] = bo
                    bos.append(bo)
                else:
                    bos.append(_alloc_bo(i, s))
            self._cached_bos[_bo_key] = bos
            # Sync instruction BO once (only needed for xclbin mode)
            if not is_elf and backend.bo_instr is not None:
                backend.bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            self._log(f"Allocated {len(bos)} BOs for {_bo_key}")

        bos = self._cached_bos[_bo_key]

        # Write input data to cached BOs.
        # Static inputs (e.g. weights) are written on first call only,
        # then skipped on subsequent calls since BO data is unchanged.
        t0 = time.time()
        with filelock.FileLock("/tmp/npu.lock"):
            # Phase 1: Write inputs using bo.map() (zero-copy into BO memory)
            t_write = time.perf_counter()
            n_written = 0
            bytes_written = 0
            for i, a in enumerate(inputs):
                if i in static_indices and not first_call:
                    continue  # Already written on first call
                if i in intermediate_set and not first_call:
                    continue  # Intermediate buffer, kernel overwrites it
                if a.dtype == bfloat16:
                    a = a.view(np.int16)
                mv = bos[i].map()
                src = np.frombuffer(a, dtype=np.uint8)
                dst = np.frombuffer(mv, dtype=np.uint8, count=len(src))
                np.copyto(dst, src, casting="no")
                bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
                n_written += 1
                bytes_written += len(src)
            t_write_ms = (time.perf_counter() - t_write) * 1000

            # Phase 2: Launch kernel
            t_kernel = time.perf_counter()
            if is_elf:
                run = xrt.run(backend.kernel)
                for i, bo in enumerate(bos):
                    run.set_arg(i, bo)
                run.start()
                run.wait2()
            else:
                h = backend.kernel(3, backend.bo_instr, len(backend.instr_v), *bos)
                h.wait()
            t_kernel_ms = (time.perf_counter() - t_kernel) * 1000

            # Phase 3: Read back output buffers using bo.map() (zero-copy view).
            t_read = time.perf_counter()
            if output_indices is None:
                readback_set = {len(inputs) - 1}
            else:
                readback_set = set(output_indices)
            for idx in readback_set:
                bos[idx].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            results = tuple(
                (
                    np.frombuffer(
                        bos[i].map(),
                        dtype=inputs[i].dtype,
                        count=inputs[i].size,
                    )  # Zero-copy view into BO memory (like IRON's read_buffer)
                    if i in readback_set
                    else np.empty(0, dtype=inputs[i].dtype)
                )
                for i, s in enumerate(sizes_in_bytes)
            )
            t_read_ms = (time.perf_counter() - t_read) * 1000

        duration = time.time() - t0
        self.profiler.record_kernel(name, duration)
        self.profiler.record_breakdown(
            name,
            t_write_ms,
            t_kernel_ms,
            t_read_ms,
            n_written,
            bytes_written,
            len(readback_set),
        )
        return results
