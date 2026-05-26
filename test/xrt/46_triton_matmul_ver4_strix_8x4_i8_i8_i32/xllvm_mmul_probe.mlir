//===- xllvm_mmul_probe.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Diagnostic-only AIE2P raw MAC probe. This is intentionally not linked into
// the runtime path; it exists to compare direct XLLVM codegen against the
// production Peano/AIE API native microkernel.
//
//===----------------------------------------------------------------------===//

module {
  llvm.func @xllvm_mmul_probe(%a : !llvm.ptr, %b : !llvm.ptr, %c : !llvm.ptr) {
    %conf = llvm.mlir.constant(0 : i32) : i32
    %av = llvm.load %a {alignment = 64 : i64} : !llvm.ptr -> vector<16xi32>
    %bv = llvm.load %b {alignment = 64 : i64} : !llvm.ptr -> vector<32xi16>
    %cv = llvm.load %c {alignment = 64 : i64} : !llvm.ptr -> vector<32xi64>
    %r = "xllvm.intr.aie2p.I512.I512.ACC2048.mac.conf"(%av, %bv, %cv, %conf)
      : (vector<16xi32>, vector<32xi16>, vector<32xi64>, i32) -> vector<32xi64>
    llvm.store %r, %c {alignment = 64 : i64} : vector<32xi64>, !llvm.ptr
    llvm.return
  }
}
