//===- air_sync.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#map = affine_map<()[s0] -> (s0 * 128)>
module {
  llvm.func @mgpuInitI8I32(!llvm.ptr, !llvm.ptr, i64, i64, i64)
  llvm.func @mgpuCheckOutputI8I32(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64) -> i32
  llvm.func @mgpuBenchmarkReset()
  llvm.func @mgpuBenchmarkSetKernelProfiling(i32)
  llvm.func @mgpuHostTimeNs() -> i64
  llvm.func @mgpuBenchmarkRecordHostNs(i64)
  llvm.func @mgpuBenchmarkPrintI8I32(i64, i64, i64, i64, i64)

  func.func @main() {
    func.call @test_matmul() : () -> ()
    return
  }

  func.func @test_matmul() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %c5 = arith.constant 5 : index
    %c1024 = arith.constant 1024 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c10_i64 = arith.constant 10 : i64
    %c5_i64 = arith.constant 5 : i64
    %c256_i64 = arith.constant 256 : i64
    %alloc = memref.alloc() : memref<1024x1024xi8>
    %alloc_0 = memref.alloc() : memref<1024x1024xi8>
    %alloc_1 = memref.alloc() : memref<1024x1024xi32>
    %intptr = memref.extract_aligned_pointer_as_index %alloc : memref<1024x1024xi8> -> index
    %intptr_0 = memref.extract_aligned_pointer_as_index %alloc_0 : memref<1024x1024xi8> -> index
    %m64 = arith.index_cast %c1024 : index to i64
    %n64 = arith.index_cast %c1024 : index to i64
    %k64 = arith.index_cast %c1024 : index to i64
    %a_ptr_i64 = arith.index_cast %intptr : index to i64
    %b_ptr_i64 = arith.index_cast %intptr_0 : index to i64
    %a_ptr = llvm.inttoptr %a_ptr_i64 : i64 to !llvm.ptr
    %b_ptr = llvm.inttoptr %b_ptr_i64 : i64 to !llvm.ptr
    llvm.call @mgpuInitI8I32(%a_ptr, %b_ptr, %m64, %n64, %k64) : (!llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()

    %memref = gpu.alloc () : memref<1024x1024xi8>
    gpu.memcpy %memref, %alloc : memref<1024x1024xi8>, memref<1024x1024xi8>
    %memref_2 = gpu.alloc () : memref<1024x1024xi8>
    gpu.memcpy %memref_2, %alloc_0 : memref<1024x1024xi8>, memref<1024x1024xi8>
    %memref_3 = gpu.alloc () : memref<1024x1024xi32>

    scf.for %warmup = %c0 to %c10 step %c1 {
      func.call @forward(%memref, %memref_2, %memref_3) : (memref<1024x1024xi8>, memref<1024x1024xi8>, memref<1024x1024xi32>) -> ()
    }

    llvm.call @mgpuBenchmarkReset() : () -> ()
    llvm.call @mgpuBenchmarkSetKernelProfiling(%c1_i32) : (i32) -> ()
    scf.for %iter = %c0 to %c5 step %c1 {
      %start_ns = llvm.call @mgpuHostTimeNs() : () -> i64
      func.call @forward(%memref, %memref_2, %memref_3) : (memref<1024x1024xi8>, memref<1024x1024xi8>, memref<1024x1024xi32>) -> ()
      %end_ns = llvm.call @mgpuHostTimeNs() : () -> i64
      %elapsed_ns = arith.subi %end_ns, %start_ns : i64
      llvm.call @mgpuBenchmarkRecordHostNs(%elapsed_ns) : (i64) -> ()
    }
    llvm.call @mgpuBenchmarkSetKernelProfiling(%c0_i32) : (i32) -> ()
    llvm.call @mgpuBenchmarkPrintI8I32(%m64, %n64, %k64, %c10_i64, %c5_i64) : (i64, i64, i64, i64, i64) -> ()

    gpu.memcpy %alloc_1, %memref_3 : memref<1024x1024xi32>, memref<1024x1024xi32>
    %c_ptr_index = memref.extract_aligned_pointer_as_index %alloc_1 : memref<1024x1024xi32> -> index
    %c_ptr_i64 = arith.index_cast %c_ptr_index : index to i64
    %c_ptr = llvm.inttoptr %c_ptr_i64 : i64 to !llvm.ptr
    %mismatches = llvm.call @mgpuCheckOutputI8I32(%c_ptr, %a_ptr, %b_ptr, %m64, %n64, %k64, %c256_i64) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64) -> i32
    gpu.dealloc %memref : memref<1024x1024xi8>
    gpu.dealloc %memref_2 : memref<1024x1024xi8>
    gpu.dealloc %memref_3 : memref<1024x1024xi32>
    memref.dealloc %alloc : memref<1024x1024xi8>
    memref.dealloc %alloc_0 : memref<1024x1024xi8>
    memref.dealloc %alloc_1 : memref<1024x1024xi32>
    return
  }

  func.func @forward(%arg0: memref<1024x1024xi8>, %arg1: memref<1024x1024xi8>, %arg2: memref<1024x1024xi32>) {
    %c16_grid = arith.constant 16 : index
    %c32_grid = arith.constant 32 : index
    air.launch (%arg3, %arg4) in (%arg5=%c32_grid, %arg6=%c16_grid) args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg2) : memref<1024x1024xi8>, memref<1024x1024xi8>, memref<1024x1024xi32> attributes {air.gpu.int8_gemm_wmma} {
      air.segment @forward_0 args(%arg10=%arg3, %arg11=%arg4, %arg12=%arg7, %arg13=%arg8, %arg14=%arg9) : index, index, memref<1024x1024xi8>, memref<1024x1024xi8>, memref<1024x1024xi32> {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c8_s = arith.constant 8 : index
        %c16 = arith.constant 16 : index
        %c64 = arith.constant 64 : index
        %c128 = arith.constant 128 : index
        %c256 = arith.constant 256 : index
        %c1024 = arith.constant 1024 : index
        %c0_i32 = arith.constant 0 : i32

        %row_off = affine.apply #map()[%arg11]
        %col_off = affine.apply #map()[%arg10]

        %a_reg = memref.alloc() : memref<8xi8, 2>
        %b_reg = memref.alloc() : memref<8xi8, 2>
        %acc = memref.alloc() : memref<64xi32, 2>

        scf.for %i = %c0 to %c64 step %c1 {
          memref.store %c0_i32, %acc[%i] : memref<64xi32, 2>
        }

        scf.for %k = %c0 to %c1024 step %c8_s {
          %As = memref.alloc() : memref<128x8xi8, 1>
          %Bs = memref.alloc() : memref<8x128xi8, 1>
          air.dma_memcpy_nd (%As[] [] [], %arg12[%row_off, %k] [%c128, %c8_s] [%c1024, %c1]) : (memref<128x8xi8, 1>, memref<1024x1024xi8>)
          air.dma_memcpy_nd (%Bs[] [] [], %arg13[%k, %col_off] [%c8_s, %c128] [%c1024, %c1]) : (memref<8x128xi8, 1>, memref<1024x1024xi8>)
          gpu.barrier

          air.herd @herd_0 tile (%tx, %ty) in (%ntx=%c256, %nty=%c1) args(%hAs=%As, %hBs=%Bs, %ha=%a_reg, %hb=%b_reg, %hacc=%acc) : memref<128x8xi8, 1>, memref<8x128xi8, 1>, memref<8xi8, 2>, memref<8xi8, 2>, memref<64xi32, 2> {
            %c0_h = arith.constant 0 : index
            %c1_h = arith.constant 1 : index
            %c8_h = arith.constant 8 : index
            %c16_h = arith.constant 16 : index
            %tile_row_idx = arith.remsi %tx, %c16_h : index
            %tile_col_idx = arith.divsi %tx, %c16_h : index
            %row_start = arith.muli %tile_row_idx, %c8_h : index
            %col_start = arith.muli %tile_col_idx, %c8_h : index

            scf.for %kk = %c0_h to %c8_h step %c1_h {
              air.dma_memcpy_nd (%ha[] [] [], %hAs[%row_start, %kk] [%c8_h] [%c8_h]) : (memref<8xi8, 2>, memref<128x8xi8, 1>)
              air.dma_memcpy_nd (%hb[] [] [], %hBs[%kk, %col_start] [%c8_h] [%c1_h]) : (memref<8xi8, 2>, memref<8x128xi8, 1>)

              scf.for %yt = %c0_h to %c8_h step %c1_h {
                scf.for %xt = %c0_h to %c8_h step %c1_h {
                  %flat = arith.muli %yt, %c8_h : index
                  %idx = arith.addi %flat, %xt : index
                  %av_i8 = memref.load %ha[%yt] : memref<8xi8, 2>
                  %bv_i8 = memref.load %hb[%xt] : memref<8xi8, 2>
                  %av = arith.extsi %av_i8 : i8 to i32
                  %bv = arith.extsi %bv_i8 : i8 to i32
                  %cv = memref.load %hacc[%idx] : memref<64xi32, 2>
                  %prod = arith.muli %av, %bv : i32
                  %sum = arith.addi %cv, %prod : i32
                  memref.store %sum, %hacc[%idx] : memref<64xi32, 2>
                }
              }
            }
            gpu.barrier
            air.herd_terminator
          }
          memref.dealloc %As : memref<128x8xi8, 1>
          memref.dealloc %Bs : memref<8x128xi8, 1>
        }

        air.herd @writeback tile (%tx, %ty) in (%ntx=%c256, %nty=%c1) args(%wacc=%acc, %wC=%arg14, %wrow=%row_off, %wcol=%col_off) : memref<64xi32, 2>, memref<1024x1024xi32>, index, index {
          %c0_w = arith.constant 0 : index
          %c8_w = arith.constant 8 : index
          %c16_w = arith.constant 16 : index
          %c1024_w = arith.constant 1024 : index
          %c1_w = arith.constant 1 : index
          %tr = arith.remsi %tx, %c16_w : index
          %tc = arith.divsi %tx, %c16_w : index
          %dr = arith.muli %tr, %c8_w : index
          %dc = arith.muli %tc, %c8_w : index
          %dst_r = arith.addi %wrow, %dr : index
          %dst_c = arith.addi %wcol, %dc : index
          air.dma_memcpy_nd (%wC[%dst_r, %dst_c] [%c8_w, %c8_w] [%c1024_w, %c1_w], %wacc[] [] []) : (memref<1024x1024xi32>, memref<64xi32, 2>)
          air.herd_terminator
        }
        memref.dealloc %a_reg : memref<8xi8, 2>
        memref.dealloc %b_reg : memref<8xi8, 2>
        memref.dealloc %acc : memref<64xi32, 2>
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}
