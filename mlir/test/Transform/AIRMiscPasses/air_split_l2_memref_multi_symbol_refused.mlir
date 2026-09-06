//===- air_split_l2_memref_multi_symbol_refused.mlir ------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// A split offset map with more than one symbol must be REFUSED with a
// diagnostic, not asserted on. `infoEntryTy` stores the map without the
// operands of the apply it came from, which is sound only for a single symbol:
// substituting s0 with the split offset then leaves a constant. With two
// symbols -- what a multi-level loop nest over the split dimension produces,
// e.g. a herd with more than one row -- s1 is left bound to an operand nobody
// carried, and composing it used to build an invalid map and abort the
// compiler inside AffineMap::get (and, once that was bypassed, inside
// AffineApplyOp::fold).
//
// The input is the reduced module in Inputs/, captured from aircc's own
// pipeline; see its header for provenance.

// RUN: not air-opt %S/Inputs/split_l2_herd_rows2_pre_split.mlir --air-split-l2-memref="max-launch-channels-mm2s=16 max-launch-channels-s2mm=16 tiles-per-l2-tile=4" 2>&1 | FileCheck %s

// CHECK: error: 'air.channel.get' op air-split-l2-memref cannot split this access
// CHECK-SAME: its offset map ()[s0, s1] -> (s0 * 2 + s1) has 2 symbols
// CHECK-SAME: only one is supported
