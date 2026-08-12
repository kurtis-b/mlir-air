# ./python/air/backend/xrt_backend.py -*- Python -*-
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import air.ir
import air.passmanager

from .abc import AirBackend, AirBackendError

import air.compiler.util

# Register the AIR dialect so air.ir.Context() can parse AIR ops.
# This was previously done as a side effect of importing aircc.main.
from air.dialects import air as _air_dialect  # noqa: F401

import numpy as np
import os
import glob
import shutil
import subprocess
import time

from air.tools import resolve_tool

from ml_dtypes import bfloat16

# Device name mappings aligned with mlir-aie (hostruntime.py, lit_config_helpers.py)
# Maps generation name to list of model strings that may appear in xrt-smi
NPU_MODELS = {
    "npu1": ["npu1", "Phoenix"],
    "npu2": ["npu4", "Strix", "npu5", "Strix Halo", "npu6", "Krackan"],
}


class XRTCompileArtifact:
    """A class encompassing information on the artifacts produced by compilation for the NPU/XRT"""

    def __init__(
        self,
        output_binary,
        kernel,
        insts,
    ):
        """
        Constructor for an XRTCompileArtifact

        Args:
            output_binary: output binary file name/path (.xclbin, .elf, or .txn)
            kernel: kernel name
            insts: instruction file name/path
        """
        self.output_binary = output_binary
        self.kernel = kernel
        self.insts = insts


class XRTBackend(AirBackend):
    """Main entry-point for the xrt based AIR backend."""

    def __init__(
        self,
        verbose: bool = False,
        target_device: str = None,
        omit_while_true_loop: bool = False,
        omit_pingpong: str = "",
        lower_linalg_to_func: str = None,
        air_loop_fusion: bool = False,
        runtime_loop_tiling_sizes: list[int] = [],
        omit_auto_broadcast: bool = False,
        channel_multiplexing: list[str] = [],
        use_lock_race_condition_fix: bool = False,
        use_lock_race_condition_fix_v2: bool = False,
        mimo_chain_lock: bool = False,
        coalesce_shim_dma: bool = False,
        trace_offset: int = 0,
        trace_size: int = 0,
        output_format: str = "xclbin",
        kernel_name: str = "",
        instance_name: str = "",
        kernel_id: str = "",
        xclbin_input: str = "",
        num_device_cols: int = 0,
        debug_ir: bool = False,
        bf16_emulation: bool = False,
        stack_size: int = 1024,
        n_perf_iters: int = 0,
        n_warmup_iters: int = 10,
    ):
        """Constructor for XRTBackend

        Args:
            verbose: verbose output
            target_device: specify target device explicitly ("npu1", "npu2", etc.). If None, will attempt auto-detection via xrt-smi. This parameter is useful when compiling without XRT installed.
            omit_while_true_loop: configure aircc to omit the while true loop it traditionally emits.
            omit_pingpong: configure aircc to omit the generation of ping-pong buffering for specific memory levels. Supported values: "", "L1", "L2", "all". Empty string means no omission (default).
            lower_linalg_to_func: configure aircc to lower linalg.generic to function calls, or loops.
            air_loop_fusion: configure aircc to add air-loop-fusion experimental pass.
            runtime_loop_tiling_sizes: tile sizes forwarded to aircc as --air-runtime-loop-tiling-sizes, which the shim DMA BD optimization pass (air-opt-shim-dma-bds) consumes as shim-dma-tile-sizes. Omit or pass an empty list to skip tiling.
            omit_auto_broadcast: configure aircc to omit the detection and lowering of broadcast data movements.
            channel_multiplexing: configure aircc to perform air channel multiplexing on specified memroy spaces.
            use_lock_race_condition_fix: configure aircc to enable a fix for lock race condition which protects against race condition.
            coalesce_shim_dma: configure aircc to coalesce consecutive contiguous shim DMA transfers on the same channel (marked air.preserve_shim_dma_order) into a single wide transfer, reducing host-issued DMA task triplets. Opt-in: only enable for feeds verified numerically equivalent when coalesced.
            trace_offset: configure aircc to stream out profiling traces at outputs, starting from the specified offset.
            trace_size: configure aircc to stream out profiling traces at outputs, with specified trace data size.
            output_format: configure aircc to produce output binary in to one of the following formats: [xclbin, txn, elf].
            kernel_name: configure aircc to package the kernel with the specified name.
            instance_name: configure aircc to package the kernel with specified instance name in xclbin metadata.
            kernel_id: configure aircc to package the kernel with specified kernel id in xclbin file.
            xclbin_input: configure aircc to package the kernel into an existing xclbin with specified xclbin file name.
            num_device_cols: number of device columns to confine the design within (0 means entire device, default).
                For npu1 (4 columns total): valid values are 0 (entire device), 1, 2, 3
                For npu2 (8 columns total): valid values are 0 (entire device), 1, 2, 3, 4, 5, 6, 7
            debug_ir: enable debug mode to emit IR after each individual pass for fine-grained inspection.
                IRs are saved to <tmpdir>/debug_ir/ with sequence numbers.
            bf16_emulation: emulate f32 vector arithmetic using bf16 operations.
            stack_size: stack size in bytes per AIE core (default: 1024). Increase when
                kernels have deep call chains (e.g., scalar fdiv needs ~1152 bytes).
            n_perf_iters: when > 0, the loaded invoker times the kernel over this many
                iterations (after n_warmup_iters warmup runs) and stores the average
                wall-clock latency in microseconds on self.last_latency_us. Only the
                kernel invocation + wait is timed (buffer sync is excluded). Default 0
                disables timing, preserving the original single-shot behavior.
            n_warmup_iters: warmup iterations excluded from timing when n_perf_iters > 0.
        """
        super().__init__()
        self.verbose = verbose
        self.target_device = target_device
        self.omit_while_true_loop = omit_while_true_loop
        # Support backward compatibility: convert True to "all", False to ""
        if isinstance(omit_pingpong, bool):
            self.omit_pingpong = "all" if omit_pingpong else ""
        else:
            self.omit_pingpong = omit_pingpong
        self.lower_linalg_to_func = lower_linalg_to_func
        self.air_loop_fusion = air_loop_fusion
        self.runtime_loop_tiling_sizes = runtime_loop_tiling_sizes
        self.omit_auto_broadcast = omit_auto_broadcast
        self.channel_multiplexing = channel_multiplexing
        if use_lock_race_condition_fix and use_lock_race_condition_fix_v2:
            raise AirBackendError(
                "use_lock_race_condition_fix and use_lock_race_condition_fix_v2 "
                "are mutually exclusive; enable at most one"
            )
        if mimo_chain_lock and not use_lock_race_condition_fix_v2:
            raise AirBackendError(
                "mimo_chain_lock only has an effect under "
                "use_lock_race_condition_fix_v2; enable both or neither"
            )
        self.use_lock_race_condition_fix = use_lock_race_condition_fix
        self.use_lock_race_condition_fix_v2 = use_lock_race_condition_fix_v2
        # FALSIFIER ARM, off by default. See doc 52 §8: it orders the writers
        # and is still unsound on the read side. Kept reachable only so the
        # measurement can be reproduced from the tree.
        self.mimo_chain_lock = mimo_chain_lock
        self.coalesce_shim_dma = coalesce_shim_dma
        self.trace_offset = trace_offset
        self.trace_size = trace_size
        self.currently_loaded = False
        #: The binary behind this backend's context once loaded; `attach_kernel`
        #: refuses to bind a kernel from any other one.
        self.loaded_binary = None
        self.output_format = output_format
        self.kernel_name = kernel_name
        self.instance_name = instance_name
        self.kernel_id = kernel_id
        self.xclbin_input = xclbin_input
        self.num_device_cols = num_device_cols
        self.debug_ir = debug_ir
        self.bf16_emulation = bf16_emulation
        if not isinstance(n_perf_iters, int) or n_perf_iters < 0:
            raise ValueError("`n_perf_iters` must be a non-negative integer")
        if not isinstance(n_warmup_iters, int) or n_warmup_iters < 0:
            raise ValueError("`n_warmup_iters` must be a non-negative integer")
        self.n_perf_iters = n_perf_iters
        self.n_warmup_iters = n_warmup_iters
        self.last_latency_us = None
        if not isinstance(stack_size, int) or stack_size <= 0:
            raise ValueError("`stack_size` must be a positive integer")
        self.stack_size = stack_size

    def __del__(self):
        self.unload()

    def compile(
        self,
        air_module: air.ir.Module,
        output_binary_name="air",
        kernel="MLIR_AIE",
        insts="air.insts.bin",
    ):
        """Compiles an AIR module for the NPU / XRT Runtime with aircc.

        The module is expected to be AIR dialect IR. The input IR is passed directly to aircc.

        Args:
            air_module: The MLIR module consisting of funcs in the AIR dialect.
            output_binary_name: base name for the output binary (without extension).
                Extension is determined by output_format: .xclbin, .elf, or .txn
            kernel: kernel name to use
            insts: instruction filename to use
        Returns:
            An XRTCompileArtifact object
        """
        if self.currently_loaded:
            raise AirBackendError(
                "Cannot use XRTBackend to compile while the artifact is currently loaded. Call unload() first."
            )

        # Determine target device: use explicit parameter if provided, otherwise auto-detect
        if self.target_device is not None:
            target_device = self.target_device
            if self.verbose:
                print(f"Using explicitly specified target device: {target_device}")
        else:
            # Try to auto-detect device via xrt-smi
            target_device = "npu1"  # Default fallback
            try:
                xrtsmi = shutil.which("xrt-smi") or "/opt/xilinx/xrt/bin/xrt-smi"
                result = subprocess.run(
                    [xrtsmi, "examine"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=10,
                )
                if result.returncode != 0:
                    if self.verbose:
                        print(
                            f"xrt-smi exited with code {result.returncode}, "
                            f"using default target device"
                        )
                        stderr = result.stderr.decode("utf-8").strip()
                        if stderr:
                            print(f"xrt-smi stderr: {stderr}")
                else:
                    output_lc = result.stdout.decode("utf-8").lower()
                    # Use case-insensitive substring matching against NPU_MODELS,
                    # aligned with mlir-aie's hostruntime.py approach.
                    detected = False
                    for version, keywords in NPU_MODELS.items():
                        if any(kw.lower() in output_lc for kw in keywords):
                            target_device = version
                            detected = True
                            if self.verbose:
                                print(f"Detected NPU device: {version}")
                            break
                    if not detected:
                        print(
                            f"WARNING: xrt-smi did not report a recognized NPU model. "
                            f"Supported: {dict(NPU_MODELS)}. "
                            f"Falling back to '{target_device}'."
                        )
            except Exception as e:
                if self.verbose:
                    print("Failed to run xrt-smi, using default target device")
                    print(e)

        # Validate output_format compatibility with target device.
        # Full ELF requires aiebu-asm aie2_config which targets npu2/AIE2P only.
        # PDI output works on all NPU generations.
        if self.output_format == "elf" and "npu1" in target_device:
            raise AirBackendError(
                f"output_format='elf' is not supported for {target_device} target. "
                "ELF output format is only supported on npu2 and later devices. "
                "Use output_format='pdi' for a raw PDI alongside the NPU instruction sequence."
            )

        # Apply user-specified device column configuration if provided
        if self.num_device_cols > 0:
            # Validate column count based on detected device
            max_cols = 4 if target_device == "npu1" else 8
            if self.num_device_cols > max_cols - 1:
                raise AirBackendError(
                    f"Invalid num_device_cols value: {self.num_device_cols}. "
                    f"For {target_device}, valid values are 0 (entire device) or 1-{max_cols-1}"
                )
            base_device = target_device
            target_device = f"{target_device}_{self.num_device_cols}col"
            if self.verbose:
                print(
                    f"Confining design to {self.num_device_cols} column(s) of {base_device} device: {target_device}"
                )

        import os, site, glob

        # Try to get peano package dir from environment variable, fallback to site-packages
        peano_package_dir = os.environ.get("PEANO_INSTALL_DIR", "")

        if peano_package_dir and os.path.isdir(peano_package_dir):
            print(
                "XRTBackend: llvm-aie package detected via PEANO_INSTALL_DIR:",
                peano_package_dir,
            )

        # Determine output file extension based on output_format
        if self.output_format == "elf":
            output_binary = f"{output_binary_name}.elf"
        elif self.output_format == "txn":
            output_binary = f"{output_binary_name}.txn"
        elif self.output_format == "pdi":
            output_binary = f"{output_binary_name}.pdi"
        else:  # xclbin (default)
            output_binary = f"{output_binary_name}.xclbin"

        with air.ir.Context():

            module_str = str(air_module)

            if self.verbose:
                print("AIR Module:")
                print(module_str)

            # A module with SEVERAL air.launch ops lowers to one aie.device per
            # launch plus a "main" orchestration device whose runtime sequence
            # configures and runs each launch in order. aiecc then produces one
            # instruction stream and one xclbin PER DEVICE, so a fixed -o/-i
            # name collides with itself: "edge 'air.insts.bin' produced
            # duplicate output path". The xclbin case threads {0} templates
            # through instead and repackages the main device's outputs into the
            # single-artifact contract below (_finalize_multi_launch_xclbin).
            # Counted the same way LaunchCounts.from_module does.
            n_launches = module_str.count("air.launch")
            multi_launch_xclbin = self.output_format == "xclbin" and n_launches > 1

            aircc_options = [
                "--device",
                target_device,
                "air.mlir",
            ]

            # Add output file options based on format
            if self.output_format == "elf":
                aircc_options += ["--elf-name", output_binary]
                # Note: ELF mode features (main device wrapper, load_pdi) are
                # automatically enabled by --output-format=elf in aircc
            elif self.output_format == "pdi":
                aircc_options += ["--pdi-name", output_binary]
                aircc_options += ["-i", insts]
            elif multi_launch_xclbin:
                # {0} is aiecc's per-device substitution: the device symbol for
                # the xclbin edge, "<device>_<sequence>" for the insts edge.
                # Scoped to the xclbin case ON PURPOSE: this else-branch also
                # serves txn, and dropping or templating -i unconditionally
                # would change the artifact contract for every txn caller.
                insts_base, insts_ext = os.path.splitext(insts)
                aircc_options += ["-o", f"{output_binary_name}_{{0}}.xclbin"]
                aircc_options += ["-i", f"{insts_base}.{{0}}{insts_ext}"]
                # Stale template outputs from a PREVIOUS multi-launch compile in
                # this directory would defeat the exactly-one globs below.
                for stale in glob.glob(
                    f"{insts_base}.*{insts_ext}"
                ) + glob.glob(f"{output_binary_name}_*.xclbin"):
                    os.remove(stale)
            else:
                aircc_options += ["-o", output_binary]
                aircc_options += ["-i", insts]

            for s in self.runtime_loop_tiling_sizes:
                aircc_options += [f"--air-runtime-loop-tiling-sizes={s}"]

            if self.verbose:
                aircc_options = aircc_options + ["-v"]

            if self.omit_while_true_loop:
                aircc_options += ["--omit-while-true-loop"]

            if self.omit_pingpong:
                # Handle both bool (True -> "all") and string ("L1", "L2", "all")
                pp_val = (
                    "all" if self.omit_pingpong is True else str(self.omit_pingpong)
                )
                aircc_options += [f"--omit-ping-pong-transform={pp_val}"]

            if self.lower_linalg_to_func:
                aircc_options += ["--lower-linalg-to-func"]
                aircc_options += [self.lower_linalg_to_func]

            if self.air_loop_fusion:
                aircc_options += ["--air-loop-fusion"]

            if self.omit_auto_broadcast:
                aircc_options += ["--omit-auto-broadcast"]

            if len(self.channel_multiplexing) != 0:
                for ch in self.channel_multiplexing:
                    aircc_options += [f"--air-channel-multiplexing={ch}"]

            if self.use_lock_race_condition_fix:
                aircc_options += ["--use-lock-race-condition-fix"]

            if self.use_lock_race_condition_fix_v2:
                aircc_options += ["--use-lock-race-condition-fix-v2"]

            if self.mimo_chain_lock:
                aircc_options += ["--mimo-chain-lock"]

            if self.coalesce_shim_dma:
                aircc_options += ["--coalesce-shim-dma"]

            if self.trace_size != 0:
                aircc_options += ["-trace-size"]
                aircc_options += [str(self.trace_size)]
                aircc_options += ["-trace-offset"]
                aircc_options += [str(self.trace_offset)]

            if self.output_format != "":
                aircc_options += ["--output-format"]
                aircc_options += [self.output_format]
            if self.kernel_name != "":
                aircc_options += ["--xclbin-kernel-name"]
                aircc_options += [self.kernel_name]
            if self.instance_name != "":
                aircc_options += ["--xclbin-instance-name"]
                aircc_options += [self.instance_name]
            if self.kernel_id != "":
                aircc_options += ["--xclbin-kernel-id"]
                aircc_options += [self.kernel_id]
            if self.xclbin_input != "":
                aircc_options += ["--xclbin-input"]
                aircc_options += [self.xclbin_input]
            if peano_package_dir != "":
                aircc_options += ["--peano"]
                aircc_options += [peano_package_dir]
                aircc_options += ["--no-xchesscc"]
                aircc_options += ["--no-xbridge"]
            else:
                aircc_options += ["--xchesscc"]
                aircc_options += ["--xbridge"]

            if self.debug_ir:
                aircc_options += ["--debug-ir"]

            if self.bf16_emulation:
                aircc_options += ["--bf16-emulation"]

            if self.stack_size != 1024:
                aircc_options += ["--stack-size", str(self.stack_size)]

            if self.verbose:
                print("Running aircc with options:", " ".join(aircc_options))

            # Write the in-memory module to the input file expected by aircc
            with open("air.mlir", "w") as f:
                f.write(module_str)

            # Invoke the C++ aircc binary. Prefer the tool bundled in the wheel.
            try:
                aircc_exe = resolve_tool("aircc")
            except RuntimeError as exc:
                raise AirBackendError(str(exc)) from exc
            result = subprocess.run(
                [str(aircc_exe)] + aircc_options,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                error_msg = result.stderr if result.stderr else result.stdout
                raise AirBackendError(f"aircc compilation failed:\n{error_msg}")

            if multi_launch_xclbin:
                self._finalize_multi_launch_xclbin(
                    output_binary_name, output_binary, insts
                )

        # For ELF mode, the kernel identifier is "main:instance_name"
        # This is used when loading the ELF via xrt.ext.kernel()
        if self.output_format == "elf" and self.instance_name != "":
            # Validate that instance_name matches a function in the module.
            # For ELF output, instance_name must match the
            # @FuncOp.from_py_func function name.
            import re

            func_names = re.findall(r"func\.func @(\w+)\(", module_str)
            if func_names and self.instance_name not in func_names:
                import warnings

                warnings.warn(
                    f"instance_name='{self.instance_name}' does not match any "
                    f"function in the module (available: {func_names}). "
                    f"For ELF output, instance_name must match the "
                    f"@FuncOp.from_py_func function name. "
                    f"Using the wrong name may cause ERT_CMD_STATE_TIMEOUT or a runtime deadlock.",
                    stacklevel=2,
                )
            elf_kernel = f"main:{self.instance_name}"
        else:
            elf_kernel = kernel

        return XRTCompileArtifact(output_binary, elf_kernel, insts)

    # The 16-byte header every NPU instruction stream starts with, and the
    # 16-byte load_pdi instruction whose first word is (pdi_id << 16) | 0x0008.
    # Observed layout of a multi-launch main stream, asserted below rather than
    # assumed: header, then one (load_pdi, device-sequence body) pair per
    # configure/run step, with each body byte-identical to that device's own
    # stream minus its header.
    _INSTS_HEADER_BYTES = 16
    _LOAD_PDI_OP_BYTES = 16
    _LOAD_PDI_OPCODE = 0x0008

    def _finalize_multi_launch_xclbin(
        self, output_binary_name, output_binary, insts, tmpdir="air_project"
    ):
        """Repackage aiecc's per-device outputs into the single-artifact contract.

        A multi-launch module reaches aiecc as N per-launch aie.device ops plus
        a "main" device whose runtime sequence inlines every launch's DMA
        program with an aiex.npu.load_pdi between them -- the same orchestration
        the ELF path packages (there, with --expand-load-pdis) into one ELF. On
        the xclbin path aiecc writes one xclbin and one instruction stream per
        device and NOTHING combines them: the main xclbin's AIE_PARTITION holds
        only the main device's (empty) PDI, while the main stream's load_pdi
        instructions reference the per-launch PDIs by an id no partition entry
        carries. This method finishes the packaging:

        1. picks the main device's xclbin and instruction stream;
        2. walks the main stream, matching each embedded device body to its own
           per-device stream to recover which load_pdi id names which device;
        3. renumbers those ids to values unique within the (possibly chained)
           partition -- aiecc numbers devices 1..N per compile, so two
           multi-launch kernels chained into one xclbin would otherwise collide,
           and single-launch chain links already occupy pdi_id 0x1;
        4. merges the per-launch PDIs into the main xclbin's AIE_PARTITION as
           kernel-less entries under the renumbered ids (the kernel keeps
           routing to the main PDI via dpu_kernel_ids; the others are reachable
           only through load_pdi);
        5. renames the finished pair to the names the caller asked for, so the
           XRTCompileArtifact contract (one binary, one kernel, one insts file)
           is unchanged and load()/attach_kernel()/the shared-xclbin chain need
           no knowledge that the module had several launches.

        Every layout assumption is asserted; a violation raises with the
        evidence rather than packaging a stream this method does not understand.

        NOTE on validation status: the compiled artifact executes its load_pdi
        instructions in the DPU stream (opcode 0x8) against partition-resident
        PDIs. That is aiecc's default lowering for multi-device xclbin modules,
        but this tree has not yet run such an artifact on hardware -- the
        hardware gate is a separate phase.
        """
        import glob
        import json
        import struct
        import tempfile
        import uuid

        insts_base, insts_ext = os.path.splitext(insts)

        main_xclbin = f"{output_binary_name}_main.xclbin"
        if not os.path.isfile(main_xclbin):
            raise AirBackendError(
                f"multi-launch xclbin packaging: expected aiecc to write "
                f"'{main_xclbin}' for the main orchestration device, found: "
                f"{sorted(glob.glob(f'{output_binary_name}_*.xclbin'))}"
            )

        stream_files = sorted(glob.glob(f"{insts_base}.*{insts_ext}"))

        def _middle(path):
            # "<insts_base>.<device>_<sequence><insts_ext>" -> "<device>_<sequence>"
            name = os.path.basename(path)
            prefix = os.path.basename(insts_base) + "."
            return name[len(prefix) : len(name) - len(insts_ext)]

        # Device names come from THIS run's per-device xclbins -- the compile
        # pre-cleans '{output_binary_name}_*.xclbin' before aircc, so unlike
        # aircc's tmpdir (which a chained build reuses across links, and whose
        # partition JSON names differ between the fresh and the xclbin_input
        # case) the set is exactly this module's devices. Match each
        # instruction stream to its device ({0} on the insts edge is
        # "<device>_<sequence>"), longest device name first so one name being
        # a prefix of another cannot mis-attribute a stream.
        device_names = sorted(
            (
                os.path.basename(p)[
                    len(f"{os.path.basename(output_binary_name)}_") : -len(".xclbin")
                ]
                for p in glob.glob(f"{output_binary_name}_*.xclbin")
            ),
            key=len,
            reverse=True,
        )
        streams_by_device = {}
        for f in stream_files:
            middle = _middle(f)
            device = next(
                (d for d in device_names if middle.startswith(d + "_")), None
            )
            if device is None:
                raise AirBackendError(
                    f"multi-launch xclbin packaging: instruction stream '{f}' "
                    f"matches no device in {sorted(device_names)}"
                )
            if device in streams_by_device:
                raise AirBackendError(
                    f"multi-launch xclbin packaging: device '{device}' has two "
                    f"instruction streams ({streams_by_device[device]!r} and "
                    f"{f!r}); one runtime sequence per device is assumed."
                )
            streams_by_device[device] = f
        if "main" not in streams_by_device:
            raise AirBackendError(
                f"multi-launch xclbin packaging: no instruction stream for the "
                f"main orchestration device among {stream_files}"
            )
        main_stream_file = streams_by_device.pop("main")
        device_stream_files = list(streams_by_device.values())
        bodies = {}
        for device, f in streams_by_device.items():
            with open(f, "rb") as fh:
                bodies[device] = fh.read()[self._INSTS_HEADER_BYTES :]

        # Walk the main stream: header, then (load_pdi, body) pairs. Recover
        # the load_pdi id for each device and record each op's byte offset so
        # the renumbering below patches exactly the words it understood.
        with open(main_stream_file, "rb") as fh:
            main_bytes = bytearray(fh.read())
        pos = self._INSTS_HEADER_BYTES
        device_ids = {}  # device -> id aiecc assigned
        op_offsets = []  # (byte offset of load_pdi word, device)
        while pos < len(main_bytes):
            (opword,) = struct.unpack_from("<I", main_bytes, pos)
            if opword & 0xFFFF != self._LOAD_PDI_OPCODE:
                raise AirBackendError(
                    f"multi-launch xclbin packaging: expected a load_pdi "
                    f"instruction at byte {pos} of '{main_stream_file}', got "
                    f"word 0x{opword:08x}. The stream layout this packaging "
                    f"understands is header + (load_pdi + device body)*."
                )
            body_start = pos + self._LOAD_PDI_OP_BYTES
            device = next(
                (
                    d
                    for d, b in bodies.items()
                    if main_bytes[body_start : body_start + len(b)] == b
                ),
                None,
            )
            if device is None:
                raise AirBackendError(
                    f"multi-launch xclbin packaging: no per-device stream "
                    f"matches the body at byte {body_start} of "
                    f"'{main_stream_file}' (devices: {sorted(bodies)})."
                )
            assigned = opword >> 16
            if device_ids.setdefault(device, assigned) != assigned:
                raise AirBackendError(
                    f"multi-launch xclbin packaging: device '{device}' is "
                    f"loaded under two different pdi ids "
                    f"({device_ids[device]} and {assigned})."
                )
            op_offsets.append((pos, device))
            pos = body_start + len(bodies[device])

        # Renumber. aiecc assigns 1..N per compile; chained into a shared
        # xclbin those collide with other links (single-launch links' PDIs sit
        # at pdi_id 0x1, a second multi-launch link would reuse 1..N). The
        # kernel id is distinct per chain link by construction (the shared
        # chain validates it), so it seeds ids unique across links.
        base = (int(self.kernel_id, 16) << 4) & 0xFFFF if self.kernel_id else 0x40
        new_ids = {
            device: base + k for k, device in enumerate(sorted(device_ids), start=1)
        }
        for offset, device in op_offsets:
            struct.pack_into(
                "<I",
                main_bytes,
                offset,
                (new_ids[device] << 16) | self._LOAD_PDI_OPCODE,
            )

        # Merge the per-launch PDIs into the main xclbin's partition under the
        # renumbered ids. Done via dump + add-replace-section so a chained
        # xclbin_input's already-merged PDIs survive untouched.
        xclbinutil = shutil.which("xclbinutil") or "/opt/xilinx/xrt/bin/xclbinutil"
        workdir = tempfile.mkdtemp(prefix="multi_launch_pack_", dir=tmpdir)
        dump_json = os.path.join(workdir, "partition.json")
        dump = subprocess.run(
            [
                xclbinutil,
                "--input",
                os.path.abspath(main_xclbin),
                "--dump-section",
                f"AIE_PARTITION:JSON:{os.path.basename(dump_json)}",
                "--force",
            ],
            cwd=workdir,
            capture_output=True,
            text=True,
        )
        if dump.returncode != 0:
            raise AirBackendError(
                f"multi-launch xclbin packaging: xclbinutil could not dump "
                f"AIE_PARTITION from '{main_xclbin}':\n{dump.stderr or dump.stdout}"
            )
        with open(dump_json) as fh:
            partition = json.load(fh)
        pdis = partition["aie_partition"]["PDIs"]
        existing = {
            int(group["pdi_id"], 16)
            for entry in pdis
            for group in entry["cdo_groups"]
        }
        for device, new_id in sorted(new_ids.items()):
            if new_id in existing:
                raise AirBackendError(
                    f"multi-launch xclbin packaging: renumbered pdi id "
                    f"0x{new_id:x} for device '{device}' already exists in the "
                    f"partition ({sorted(hex(i) for i in existing)}). Give this "
                    f"compile a different kernel_id."
                )
            pdi_file = os.path.abspath(os.path.join(tmpdir, f"{device}.pdi"))
            if not os.path.isfile(pdi_file):
                raise AirBackendError(
                    f"multi-launch xclbin packaging: no PDI for device "
                    f"'{device}' at '{pdi_file}'."
                )
            pdis.append(
                {
                    "uuid": str(uuid.uuid4()),
                    "file_name": pdi_file,
                    "cdo_groups": [
                        {
                            "name": "DPU",
                            "type": "PRIMARY",
                            "pdi_id": hex(new_id),
                            # No dpu_kernel_ids: nothing routes here at context
                            # creation; the PDI is reachable only through the
                            # stream's load_pdi. Claiming the kernel id would
                            # recreate the duplicate-kernel-id failure the
                            # shared-xclbin chain validates against.
                            "dpu_kernel_ids": [],
                            "pre_cdo_groups": ["0xC1"],
                        }
                    ],
                }
            )
        merged_json = os.path.join(workdir, "merged.json")
        with open(merged_json, "w") as fh:
            json.dump(partition, fh, indent=1)
        merge = subprocess.run(
            [
                xclbinutil,
                "--input",
                os.path.abspath(main_xclbin),
                "--add-replace-section",
                f"AIE_PARTITION:JSON:{os.path.basename(merged_json)}",
                "--force",
                "--output",
                os.path.abspath(output_binary),
            ],
            cwd=workdir,
            capture_output=True,
            text=True,
        )
        if merge.returncode != 0:
            raise AirBackendError(
                f"multi-launch xclbin packaging: xclbinutil could not merge the "
                f"per-launch PDIs into '{output_binary}':\n"
                f"{merge.stderr or merge.stdout}"
            )
        shutil.rmtree(workdir, ignore_errors=True)

        with open(insts, "wb") as fh:
            fh.write(main_bytes)

        # Drop the per-device intermediates so a later compile in this
        # directory (the shared chain reuses one cwd) cannot glob them up.
        for stale in device_stream_files + [main_stream_file]:
            os.remove(stale)
        for stale in glob.glob(f"{output_binary_name}_*.xclbin"):
            os.remove(stale)

        if self.verbose:
            print(
                f"multi-launch xclbin packaging: '{output_binary}' holds "
                f"{len(pdis)} PDIs; per-launch ids "
                f"{ {d: hex(i) for d, i in sorted(new_ids.items())} } "
                f"embedded in '{insts}'."
            )

    def compile_from_torch_mlir(
        self,
        imported_module,
        pipeline=None,
        verbose=False,
    ):
        import torch_mlir
        import torch_mlir.passmanager

        if type(imported_module) is torch_mlir.ir.Module:
            with imported_module.operation.context:
                pm = torch_mlir.passmanager.PassManager.parse(
                    "builtin.module(refback-mlprogram-bufferize)"
                )
                pm.run(imported_module.operation)

        with air.ir.Context():
            linalg_module = air.ir.Module.parse(str(imported_module))
            pm = air.passmanager.PassManager.parse(
                air.compiler.util.LINALG_TENSOR_TO_MEMREF_PIPELINE
            )
            if verbose:
                print(
                    "Running MLIR pass pipeline: ",
                    air.compiler.util.LINALG_TENSOR_TO_MEMREF_PIPELINE,
                )
            pm.run(linalg_module.operation)

            if verbose:
                print("Linalg Module:")
                print(linalg_module)

            DEFAULT_PIPELINE = (
                "builtin.module("
                + ",".join(
                    [
                        "buffer-results-to-out-params",
                        "air-linalg-codegen",
                        "air-par-to-herd{depth=-1}",
                        "air-par-to-launch{has-air-segment=true}",
                        "air-copy-to-dma",
                        "canonicalize",
                        "cse",
                    ]
                )
                + ")"
            )
            if pipeline is None:
                pipeline = DEFAULT_PIPELINE

            if callable(pipeline):
                air_module = pipeline(linalg_module)
            else:
                pm = air.passmanager.PassManager.parse(pipeline)
                pm.run(linalg_module.operation)
                air_module = linalg_module

            if verbose:
                print("Air Module:")
                print(air_module)

        return self.compile(air_module)

    def load(self, artifact: XRTCompileArtifact):
        """Load a compiled artifact into the air runtime.

        Args:
            artifact: The result of calling compile with XRTBackend on an MLIR-AIR module.
                Supports both xclbin and ELF formats.

        Returns: A callable that can be used to invoke the loaded module.
            The callable takes a list of numpy arrays. Each numpy array is
            assumed to be an input/output tensor. The callable also returns a
            list of numpy arrays, one for each tensor.
        """
        # Try to import pyxrt - it's only needed for load(), not compile()
        try:
            import pyxrt as xrt
        except ImportError:
            raise AirBackendError(
                "XRT runtime (pyxrt) is not available. "
                "The compile() method can generate artifacts without XRT, "
                "but load() requires XRT to be installed for hardware execution. "
                "To compile without XRT, use compile() and specify target_device parameter. "
                "Install XRT to use load() for hardware execution."
            )

        if self.currently_loaded:
            raise AirBackendError(
                "Cannot use XRTBackend to compile while the artifact is currently loaded. Call unload() first."
            )

        if not os.path.isfile(artifact.output_binary):
            raise AirBackendError(
                f"Cannot load XRTCompileArtifact because {artifact.output_binary} file does not exist"
            )

        # PDI artifacts are intended for alternative (non-XRT) runtimes and
        # cannot be loaded via this XRT-based load() path.
        if artifact.output_binary.endswith(".pdi"):
            raise AirBackendError(
                "output_format='pdi' produces artifacts for alternative runtimes "
                "and cannot be loaded via XRTBackend.load(). Pass the .pdi file "
                "and accompanying .insts.bin to your target runtime directly."
            )

        # Determine the loading mode based on file extension
        is_elf = artifact.output_binary.endswith(".elf")

        # Which binary this backend's context belongs to. `attach_kernel` checks
        # it: binding a kernel from a DIFFERENT xclbin onto this context would
        # execute it against the wrong array configuration.
        self.loaded_binary = artifact.output_binary

        # create the device
        self.device = xrt.device(0)

        if is_elf:
            # ELF loading path - uses experimental APIs
            # No instruction file needed for ELF (instructions embedded in ELF)
            try:
                self.elf = xrt.elf(artifact.output_binary)
                self.context = xrt.hw_context(self.device, self.elf)
                self.kernel = xrt.ext.kernel(self.context, artifact.kernel)
            except Exception as e:
                raise AirBackendError(
                    f"Failed to load ELF kernel for XRT from '{artifact.output_binary}' "
                    f"with kernel name '{artifact.kernel}'. "
                    "Ensure this file is a valid ELF binary compiled for the target device "
                    "and that it contains a kernel symbol matching the provided name."
                ) from e
            self.bo_instr = None  # Not needed for ELF
            self.instr_v = None

            def invoker(*args):
                sizes_in_bytes = [a.size * a.itemsize for a in args]
                # Use xrt.ext.bo for ELF mode (simpler, no group_id needed)
                bos = [xrt.ext.bo(self.device, s) for s in sizes_in_bytes]

                # Map each BO once and keep the mapped numpy array alive. All
                # host<->device data movement goes through these mapped arrays +
                # bo.sync(); bo.write()/bo.read() are avoided because they
                # misbehave under numpy 2.x with older pyxrt.
                bo_maps = [
                    np.frombuffer(bos[i].map(), dtype=np.uint8)
                    for i in range(len(args))
                ]

                for i, a in enumerate(args):
                    if a.dtype == bfloat16:
                        a = a.view(np.int16)
                    bo_maps[i][: sizes_in_bytes[i]] = np.frombuffer(
                        np.ascontiguousarray(a).tobytes(), dtype=np.uint8
                    )
                    bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

                # Use xrt.run for ELF mode
                run = xrt.run(self.kernel)
                for i, bo in enumerate(bos):
                    run.set_arg(i, bo)
                if self.n_perf_iters > 0:
                    # Time only run.start()+wait2(), averaged over n_perf_iters
                    # after n_warmup_iters warmup runs (buffer sync excluded).
                    for _ in range(self.n_warmup_iters):
                        run.start()
                        run.wait2()
                    t0 = time.perf_counter()
                    for _ in range(self.n_perf_iters):
                        run.start()
                        run.wait2()
                    t1 = time.perf_counter()
                    self.last_latency_us = (t1 - t0) / self.n_perf_iters * 1e6
                else:
                    run.start()
                    run.wait2()

                for i in range(len(args)):
                    bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
                return tuple(
                    [
                        np.frombuffer(bo_maps[i].tobytes(), dtype=args[i].dtype)
                        for i in range(len(args))
                    ]
                )

        else:
            # xclbin loading path - original implementation
            if not os.path.isfile(artifact.insts):
                raise AirBackendError(
                    f"Cannot load XRTCompileArtifact because {artifact.insts} insts file does not exist"
                )

            self.xclbin = xrt.xclbin(artifact.output_binary)
            self.device.register_xclbin(self.xclbin)
            self.context = xrt.hw_context(self.device, self.xclbin.get_uuid())

            # find and load the kernel
            kernels = self.xclbin.get_kernels()
            try:
                xkernel = [k for k in kernels if artifact.kernel in k.get_name()][0]
            except:
                raise AirBackendError(
                    f"Kernel '{artifact.kernel}' not found in '{artifact.output_binary}'"
                )
            self.kernel = xrt.kernel(self.context, xkernel.get_name())

            # load the instructions as a numpy array
            with open(artifact.insts, "rb") as f:
                instr_data = f.read()
                self.instr_v = np.frombuffer(instr_data, dtype=np.uint32).copy()

            self.bo_instr = xrt.bo(
                self.device,
                len(self.instr_v) * 4,
                xrt.bo.cacheable,
                self.kernel.group_id(1),
            )
            self.bo_instr.write(self.instr_v.tobytes(), 0)

            def invoker(*args):
                # limit arg length to 5
                if len(args) > 5:
                    raise ValueError("Too many arguments")
                sizes_in_bytes = [a.size * a.itemsize for a in args]
                bos = [
                    xrt.bo(
                        self.device, s, xrt.bo.host_only, self.kernel.group_id(i + 3)
                    )
                    for i, s in enumerate(sizes_in_bytes)
                ]

                # Map each host_only BO once and keep the mapped numpy array
                # alive. All host<->device data movement goes through these
                # mapped arrays + bo.sync(); bo.write()/bo.read() are avoided
                # because they misbehave under numpy 2.x with older pyxrt.
                # This mirrors mlir-aie's XRTTensor implementation.
                bo_maps = [
                    np.frombuffer(bos[i].map(), dtype=np.uint8)
                    for i in range(len(args))
                ]

                self.bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
                for i, a in enumerate(args):
                    if a.dtype == bfloat16:
                        # store bfloat16 in binary as int16
                        a = a.view(np.int16)
                    bo_maps[i][: sizes_in_bytes[i]] = np.frombuffer(
                        np.ascontiguousarray(a).tobytes(), dtype=np.uint8
                    )
                    bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

                if self.n_perf_iters > 0:
                    # Time only the kernel invocation + wait, averaged over
                    # n_perf_iters after n_warmup_iters warmup runs (buffer sync
                    # excluded — matches the C++ test-harness timing range).
                    for _ in range(self.n_warmup_iters):
                        self.kernel(3, self.bo_instr, len(self.instr_v), *bos).wait()
                    t0 = time.perf_counter()
                    for _ in range(self.n_perf_iters):
                        self.kernel(3, self.bo_instr, len(self.instr_v), *bos).wait()
                    t1 = time.perf_counter()
                    self.last_latency_us = (t1 - t0) / self.n_perf_iters * 1e6
                else:
                    h = self.kernel(3, self.bo_instr, len(self.instr_v), *bos)
                    h.wait()

                for i in range(len(args)):
                    bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
                return tuple(
                    [
                        np.frombuffer(bo_maps[i].tobytes(), dtype=args[i].dtype)
                        for i in range(len(args))
                    ]
                )

        self.currently_loaded = True
        return invoker

    def attach_kernel(self, artifact):
        """Bind ANOTHER kernel out of the xclbin this backend already loaded.

        `[2026-08-09]` This is the runtime half of "N instruction streams under
        one xclbin": several AIR kernels packaged into one xclbin by chaining
        `xclbin_input`, then executed from ONE `hw_context` -- the array is
        configured once and moving between kernels costs an instruction swap
        rather than a reconfiguration.

        `load()` builds a context per artifact. This reuses the context already
        standing and binds only what is per-kernel: the `xrt.kernel` and its own
        instruction BO. Returns a view exposing `kernel`, `bo_instr` and
        `instr_v`, which is the surface a dispatch loop reads -- so a caller can
        treat it exactly like a loaded backend without knowing they are shared.

        Two identifiers must be distinct per stream, and NEITHER is checked here
        because both fail at a distance:

        - `instance_name`, the kernel's name in the xclbin. The lookup below is a
          SUBSTRING match, so two kernels sharing a name silently return whichever
          appears first -- the wrong program with the right buffers.
        - `kernel_id`, which routes the kernel to its PDI (its array
          configuration) in the merged `AIE_PARTITION`. Every AIR compile
          defaults to `0x901`, so two PDIs both claiming it are indistinguishable
          to the runtime and the second kernel executes against the first's
          configuration. Measured: `ERT_CMD_STATE_TIMEOUT` at one shape and
          garbage at `mean_rel_L1` 1.41 with no error raised at another.

        Reproduce both with `agents/probes/probe_one_xclbin_n_streams.py`.

        Args:
            artifact: an `XRTCompileArtifact` whose `output_binary` is the SAME
                xclbin this backend loaded, and whose `kernel` names a different
                kernel inside it.

        Returns:
            An object with `kernel`, `bo_instr` and `instr_v`.
        """
        import pyxrt as xrt

        if self.context is None or self.xclbin is None:
            raise AirBackendError(
                "attach_kernel needs a backend with an xclbin already loaded; "
                "call load() on the first artifact of the shared xclbin first."
            )
        if artifact.output_binary != self.loaded_binary:
            raise AirBackendError(
                f"attach_kernel expects the SAME xclbin this backend loaded "
                f"({self.loaded_binary}), got {artifact.output_binary}. Sharing a "
                "context across two different xclbins would execute one kernel "
                "against the other's array configuration."
            )
        if not artifact.insts or not os.path.isfile(artifact.insts):
            raise AirBackendError(
                f"attach_kernel needs this kernel's own instruction stream; "
                f"{artifact.insts!r} is missing. Under one xclbin the kernels "
                "share a configuration and differ ONLY in their instructions."
            )

        matches = [
            k.get_name() for k in self.xclbin.get_kernels() if artifact.kernel in k.get_name()
        ]
        if not matches:
            raise AirBackendError(
                f"Kernel '{artifact.kernel}' not found in "
                f"'{artifact.output_binary}'. Present: "
                f"{[k.get_name() for k in self.xclbin.get_kernels()]}"
            )
        if len(matches) > 1:
            # The substring match cannot choose, and choosing wrong is silent.
            raise AirBackendError(
                f"Kernel name '{artifact.kernel}' matches {len(matches)} kernels "
                f"in '{artifact.output_binary}': {matches}. The lookup is a "
                "substring match, so an ambiguous name would silently select the "
                "wrong program. Give each stream a distinct instance_name."
            )

        kernel = xrt.kernel(self.context, matches[0])
        with open(artifact.insts, "rb") as f:
            instr_v = np.frombuffer(f.read(), dtype=np.uint32).copy()
        bo_instr = xrt.bo(
            self.device, len(instr_v) * 4, xrt.bo.cacheable, kernel.group_id(1)
        )
        bo_instr.write(instr_v.tobytes(), 0)

        class _AttachedKernel:
            """One stream's per-kernel state over a shared context.

            Overrides only what is PER KERNEL -- the `xrt.kernel` and its own
            instruction stream -- and delegates everything else to the backend
            that owns the context (`device`, `context`, `xclbin`, ...). Written as
            a proxy rather than a fixed attribute list so a caller reading some
            other backend attribute gets the host's value instead of an
            `AttributeError`; the surface a dispatch loop touches is not fully
            enumerable from here.
            """

            def __init__(self, host, kernel, bo_instr, instr_v):
                # Bypass __setattr__/__getattr__ recursion on the proxy target.
                object.__setattr__(self, "_host", host)
                object.__setattr__(self, "kernel", kernel)
                object.__setattr__(self, "bo_instr", bo_instr)
                object.__setattr__(self, "instr_v", instr_v)

            def __getattr__(self, item):
                # Only reached for attributes not set above.
                return getattr(object.__getattribute__(self, "_host"), item)

        return _AttachedKernel(self, kernel, bo_instr, instr_v)

    def compile_and_load(self, module):
        """
        Compile and load a module in one step.

        Args:
            air_module: The MLIR module consisting of funcs in the AIR dialect.

        Returns: A callable that can be used to invoke the loaded module.
            The callable takes a list of numpy arrays. Each numpy array is
            assumed to be an input/output tensor. The callable also returns a
            list of numpy arrays, one for each tensor.
        """
        c = self.compile(module)
        return self.load(c)

    def unload(self):
        """Unload any loaded module and shutdown the air runtime."""
        self.kernel = None
        self.context = None
        self.xclbin = None
        self.elf = None
        self.device = None
        self.bo_instr = None
        self.instr_v = None
        self.loaded_binary = None
        self.currently_loaded = False
