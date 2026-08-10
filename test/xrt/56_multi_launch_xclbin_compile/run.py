#!/usr/bin/env python3
# run.py -*- Python -*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""A TWO-LAUNCH module compiled to XCLBIN, compile-only.

The regression this pins: XRTBackend.compile's xclbin branch used to pass a
fixed instruction-file name (`-i air.insts.bin`) straight through to aiecc,
and a module with two `air.launch` ops lowers to two per-launch `aie.device`
ops plus a "main" orchestration device -- three instruction streams, one
output path, and aiecc refuses: "edge 'air.insts.bin' produced duplicate
output path". Only the ELF path packaged multi-launch modules, so the
shared-xclbin offload chain was silently bounded to single-launch shapes
(doc 29 "The 4096 wall") -- and no backend-level fixture existed to say so.

This test compiles the same two-launch module 47_multi_launch_pdi_reconfig
runs over the ELF path, but to xclbin, and then checks the finished artifact
against the single-artifact contract the backend promises:

  1. ONE xclbin and ONE instruction stream, under the caller's names;
  2. the xclbin's AIE_PARTITION holds the main PDI (owning the kernel id)
     plus one kernel-less PDI per launch;
  3. the instruction stream is header + (load_pdi + launch body) pairs whose
     renumbered pdi ids match the partition exactly.

Compile-only on purpose: it needs aircc/aiecc/peano/xclbinutil but no NPU
dispatch. Hardware execution of multi-launch xclbins is gated elsewhere.
"""

import json
import os
import struct
import subprocess
import sys

from air.backend.xrt import XRTBackend
from air.ir import Context, Location, Module

# The two-launch module from test 47 (add 2 over the first half, reconfigure,
# add 3 over the second half), verbatim except for comments.
AIR_MODULE = """
module {
  func.func @reconfigure_example(%arg0: memref<512xi32>, %arg1: memref<512xi32>) {
    %c8 = arith.constant 8 : index
    air.launch (%x) in (%sz=%c8) args(%input=%arg0, %output=%arg1) : memref<512xi32>, memref<512xi32> attributes {id = 1 : i32} {
      %c16_0 = arith.constant 16 : index
      %tile_offset = arith.muli %x, %c16_0 : index
      air.segment @add_two args(%seg_input=%input, %seg_output=%output, %offset=%tile_offset) : memref<512xi32>, memref<512xi32>, index attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 1 : i64, y_loc = 2 : i64, y_size = 1 : i64} {
        %c1_1 = arith.constant 1 : index
        air.herd @herd_add_two tile (%tx, %ty) in (%sx=%c1_1, %sy=%c1_1) args(%herd_input=%seg_input, %herd_output=%seg_output, %herd_offset=%offset) : memref<512xi32>, memref<512xi32>, index attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
          %c0_h = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c16_h = arith.constant 16 : index
          %c2_i32 = arith.constant 2 : i32
          %l1_in = memref.alloc() : memref<16xi32, 2>
          %l1_out = memref.alloc() : memref<16xi32, 2>
          air.dma_memcpy_nd (%l1_in[] [] [], %herd_input[%herd_offset] [%c16_h] [%c1_h]) : (memref<16xi32, 2>, memref<512xi32>)
          scf.for %i = %c0_h to %c16_h step %c1_h {
            %val = memref.load %l1_in[%i] : memref<16xi32, 2>
            %result = arith.addi %val, %c2_i32 : i32
            memref.store %result, %l1_out[%i] : memref<16xi32, 2>
          }
          air.dma_memcpy_nd (%herd_output[%herd_offset] [%c16_h] [%c1_h], %l1_out[] [] []) : (memref<512xi32>, memref<16xi32, 2>)
          memref.dealloc %l1_in : memref<16xi32, 2>
          memref.dealloc %l1_out : memref<16xi32, 2>
        }
      }
    }
    air.launch (%x) in (%sz=%c8) args(%input=%arg0, %output=%arg1) : memref<512xi32>, memref<512xi32> attributes {id = 4 : i32} {
      %c16_0 = arith.constant 16 : index
      %c128_0 = arith.constant 128 : index
      %iter_offset = arith.muli %x, %c16_0 : index
      %tile_offset = arith.addi %iter_offset, %c128_0 : index
      air.segment @add_three args(%seg_input=%input, %seg_output=%output, %offset=%tile_offset) : memref<512xi32>, memref<512xi32>, index attributes {id = 5 : i32, x_loc = 0 : i64, x_size = 1 : i64, y_loc = 2 : i64, y_size = 1 : i64} {
        %c1_1 = arith.constant 1 : index
        air.herd @herd_add_three tile (%tx, %ty) in (%sx=%c1_1, %sy=%c1_1) args(%herd_input=%seg_input, %herd_output=%seg_output, %herd_offset=%offset) : memref<512xi32>, memref<512xi32>, index attributes {id = 6 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
          %c0_h = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c16_h = arith.constant 16 : index
          %c3_i32 = arith.constant 3 : i32
          %l1_in = memref.alloc() : memref<16xi32, 2>
          %l1_out = memref.alloc() : memref<16xi32, 2>
          air.dma_memcpy_nd (%l1_in[] [] [], %herd_input[%herd_offset] [%c16_h] [%c1_h]) : (memref<16xi32, 2>, memref<512xi32>)
          scf.for %i = %c0_h to %c16_h step %c1_h {
            %val = memref.load %l1_in[%i] : memref<16xi32, 2>
            %result = arith.addi %val, %c3_i32 : i32
            memref.store %result, %l1_out[%i] : memref<16xi32, 2>
          }
          air.dma_memcpy_nd (%herd_output[%herd_offset] [%c16_h] [%c1_h], %l1_out[] [] []) : (memref<512xi32>, memref<16xi32, 2>)
          memref.dealloc %l1_in : memref<16xi32, 2>
          memref.dealloc %l1_out : memref<16xi32, 2>
        }
      }
    }
    return
  }
}
"""

KERNEL_ID = "0x777"
INSTS_HEADER_BYTES = 16
LOAD_PDI_OP_BYTES = 16
LOAD_PDI_OPCODE = 0x0008


def xclbinutil():
    import shutil

    return shutil.which("xclbinutil") or "/opt/xilinx/xrt/bin/xclbinutil"


def dump_partition(xclbin):
    out = "partition_dump.json"
    subprocess.run(
        [
            xclbinutil(),
            "--input",
            xclbin,
            "--dump-section",
            f"AIE_PARTITION:JSON:{out}",
            "--force",
        ],
        check=True,
        capture_output=True,
    )
    with open(out) as fh:
        return json.load(fh)


def stream_load_pdi_ids(insts):
    """The pdi ids of every load_pdi in the stream, walked in order.

    Walking requires knowing each body's length; the fixture only needs the
    ids, so it scans the two 16-byte-aligned candidates the packaging wrote:
    every op word shaped (id << 16) | 0x0008 whose id is one the partition
    carries. The count assertion below keeps this honest.
    """
    with open(insts, "rb") as fh:
        data = fh.read()
    ids = []
    for pos in range(0, len(data) - 3, 4):
        (word,) = struct.unpack_from("<I", data, pos)
        if word & 0xFFFF == LOAD_PDI_OPCODE and (word >> 16) > 0xFF:
            ids.append(word >> 16)
    return data, ids


def main():
    with Context(), Location.unknown():
        module = Module.parse(AIR_MODULE)
        backend = XRTBackend(
            target_device="npu2",
            output_format="xclbin",
            kernel_name="two_launch_fixture",
            instance_name="two_launch_fixture",
            kernel_id=KERNEL_ID,
            omit_while_true_loop=False,
            runtime_loop_tiling_sizes=[4, 4],
        )
        artifact = backend.compile(
            module, output_binary_name="two_launch", insts="two_launch.insts.bin"
        )

    # 1. Single-artifact contract under the caller's names.
    assert artifact.output_binary == "two_launch.xclbin", artifact.output_binary
    assert artifact.insts == "two_launch.insts.bin", artifact.insts
    assert os.path.isfile(artifact.output_binary), "xclbin missing"
    assert os.path.isfile(artifact.insts), "instruction stream missing"

    # 2. Partition: the main PDI owns the kernel id; one kernel-less PDI per
    #    launch, under ids renumbered off the kernel id.
    partition = dump_partition(artifact.output_binary)
    groups = [
        g
        for pdi in partition["aie_partition"]["PDIs"]
        for g in pdi["cdo_groups"]
    ]
    owning = [g for g in groups if g.get("dpu_kernel_ids")]
    kernel_less = [g for g in groups if not g.get("dpu_kernel_ids")]
    assert len(owning) == 1, f"exactly one PDI must own the kernel: {groups}"
    assert owning[0]["dpu_kernel_ids"] == [KERNEL_ID], owning[0]
    assert len(kernel_less) == 2, (
        f"two launches need two kernel-less PDIs, got {len(kernel_less)}: {groups}"
    )
    partition_ids = sorted(int(g["pdi_id"], 16) for g in kernel_less)
    base = (int(KERNEL_ID, 16) << 4) & 0xFFFF
    assert partition_ids == [base + 1, base + 2], (
        f"per-launch pdi ids must be renumbered off kernel_id "
        f"{KERNEL_ID}: expected {[hex(base + 1), hex(base + 2)]}, "
        f"got {[hex(i) for i in partition_ids]}"
    )

    # 3. Stream: exactly two load_pdi ops, ids matching the partition.
    data, stream_ids = stream_load_pdi_ids(artifact.insts)
    assert sorted(stream_ids) == partition_ids, (
        f"stream load_pdi ids {[hex(i) for i in stream_ids]} do not match the "
        f"partition's kernel-less pdi ids {[hex(i) for i in partition_ids]}"
    )
    assert len(stream_ids) == 2, stream_ids
    (total_len,) = struct.unpack_from("<I", data, 12)
    assert total_len == len(data), (
        f"stream header length {total_len} != file size {len(data)}"
    )

    print(
        f"PASS: two-launch xclbin compile -- {artifact.output_binary} holds "
        f"{len(groups)} PDIs, stream load_pdi ids "
        f"{[hex(i) for i in sorted(stream_ids)]}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
