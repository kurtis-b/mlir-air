#map = affine_map<()[s0] -> (s0 * 512)>
#map1 = affine_map<()[s0] -> (s0 * 256)>
#map2 = affine_map<()[s0] -> (s0 * 32)>
#map3 = affine_map<()[s0, s1, s2] -> (s0 + (s1 + s2) * 524288)>
module {
  func.func private @op_has_no_registered_library_name_m64(memref<1x1x4x8x8x8xbf16, 2 : i32>, memref<1x1x16x4x8x8xbf16, 2 : i32>, memref<1x1x16x8x8x8xf32, strided<[32768, 8192, 512, 64, 8, 1], offset: ?>, 2 : i32>) attributes {link_with = "mm_m64.o", llvm.emit_c_interface}
  func.func private @zero_f32_mn_m64(memref<1x1x16x8x8x8xf32, strided<[32768, 8192, 512, 64, 8, 1], offset: ?>, 2 : i32>) attributes {link_with = "mm_m64.o", llvm.emit_c_interface}
  func.func @gemm_cast_bf16(%arg0: memref<2048x8192xbf16>, %arg1: memref<8192x2048xbf16>, %arg2: memref<2048x2048xf32>, %arg3: memref<2048x2048xbf16>) {
    %c4 = arith.constant 4 : index
    %c4_0 = arith.constant 4 : index
    air.launch (%arg4, %arg5) in (%arg6=%c4, %arg7=%c4_0) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2) : memref<2048x8192xbf16>, memref<8192x2048xbf16>, memref<2048x2048xf32> {
      air.segment @gm_matmul_seg  args(%arg11=%arg4, %arg12=%arg5, %arg13=%arg8, %arg14=%arg9, %arg15=%arg10) : index, index, memref<2048x8192xbf16>, memref<8192x2048xbf16>, memref<2048x2048xf32> {
        %alloc = memref.alloc() : memref<8x1x64x256xbf16, 1 : i32>
        %alloc_1 = memref.alloc() : memref<1x4x256x128xbf16, 1 : i32>
        %alloc_2 = memref.alloc() : memref<8x4x64x128xf32, 1 : i32>
        %alloc_3 = memref.alloc() : memref<8x4x16x8x8x8xf32, 2 : i32>
        %0 = affine.apply #map()[%arg11]
        %1 = affine.apply #map()[%arg12]
        %c8 = arith.constant 8 : index
        %c4_4 = arith.constant 4 : index
        air.herd @gm_herd_0  tile (%arg16, %arg17) in (%arg18=%c8, %arg19=%c4_4) args(%arg20=%alloc_3) : memref<8x4x16x8x8x8xf32, 2 : i32> attributes {link_with = "mm_m64.o"} {
          %subview = memref.subview %arg20[%arg16, %arg17, 0, 0, 0, 0] [1, 1, 16, 8, 8, 8] [1, 1, 1, 1, 1, 1] : memref<8x4x16x8x8x8xf32, 2 : i32> to memref<1x1x16x8x8x8xf32, strided<[32768, 8192, 512, 64, 8, 1], offset: ?>, 2 : i32>
          func.call @zero_f32_mn_m64(%subview) : (memref<1x1x16x8x8x8xf32, strided<[32768, 8192, 512, 64, 8, 1], offset: ?>, 2 : i32>) -> ()
        }
        %c0 = arith.constant 0 : index
        %c32 = arith.constant 32 : index
        %c1 = arith.constant 1 : index
        scf.for %arg16 = %c0 to %c32 step %c1 {
          %2 = affine.apply #map1()[%arg16]
          air.dma_memcpy_nd (%alloc[] [] [], %arg13[0, 0, %0, %2] [8, 1, 64, 256] [524288, 256, 8192, 1]) : (memref<8x1x64x256xbf16, 1 : i32>, memref<2048x8192xbf16>)
          air.dma_memcpy_nd (%alloc_1[] [] [], %arg14[0, 0, %2, %1] [1, 4, 256, 128] [524288, 128, 2048, 1]) : (memref<1x4x256x128xbf16, 1 : i32>, memref<8192x2048xbf16>)
          %c8_7 = arith.constant 8 : index
          %c4_8 = arith.constant 4 : index
          air.herd @gm_herd_0  tile (%arg17, %arg18) in (%arg19=%c8_7, %arg20=%c4_8) args(%arg21=%alloc_3, %arg22=%alloc, %arg23=%alloc_1) : memref<8x4x16x8x8x8xf32, 2 : i32>, memref<8x1x64x256xbf16, 1 : i32>, memref<1x4x256x128xbf16, 1 : i32> attributes {link_with = "mm_m64.o"} {
            %alloc_9 = memref.alloc() : memref<1x1x4x8x8x8xbf16, 2 : i32>
            %alloc_10 = memref.alloc() : memref<1x1x16x4x8x8xbf16, 2 : i32>
            %c0_11 = arith.constant 0 : index
            %c8_12 = arith.constant 8 : index
            %c1_13 = arith.constant 1 : index
            scf.for %arg24 = %c0_11 to %c8_12 step %c1_13 {
              %3 = affine.apply #map2()[%arg24]
              air.dma_memcpy_nd (%alloc_9[] [] [], %arg22[%arg17, 0, 0, 0, 0, %3] [1, 1, 4, 8, 8, 8] [16384, 16384, 8, 2048, 256, 1]) : (memref<1x1x4x8x8x8xbf16, 2 : i32>, memref<8x1x64x256xbf16, 1 : i32>)
              air.dma_memcpy_nd (%alloc_10[] [] [], %arg23[0, %arg18, 0, 0, %3, 0] [1, 1, 16, 4, 8, 8] [131072, 32768, 8, 1024, 128, 1]) : (memref<1x1x16x4x8x8xbf16, 2 : i32>, memref<1x4x256x128xbf16, 1 : i32>)
              %subview = memref.subview %arg21[%arg17, %arg18, 0, 0, 0, 0] [1, 1, 16, 8, 8, 8] [1, 1, 1, 1, 1, 1] : memref<8x4x16x8x8x8xf32, 2 : i32> to memref<1x1x16x8x8x8xf32, strided<[32768, 8192, 512, 64, 8, 1], offset: ?>, 2 : i32>
              func.call @op_has_no_registered_library_name_m64(%alloc_9, %alloc_10, %subview) : (memref<1x1x4x8x8x8xbf16, 2 : i32>, memref<1x1x16x4x8x8xbf16, 2 : i32>, memref<1x1x16x8x8x8xf32, strided<[32768, 8192, 512, 64, 8, 1], offset: ?>, 2 : i32>) -> ()
            }
            memref.dealloc %alloc_9 : memref<1x1x4x8x8x8xbf16, 2 : i32>
            memref.dealloc %alloc_10 : memref<1x1x16x4x8x8xbf16, 2 : i32>
          }
        }
        %c8_5 = arith.constant 8 : index
        %c4_6 = arith.constant 4 : index
        air.herd @gm_herd_0  tile (%arg16, %arg17) in (%arg18=%c8_5, %arg19=%c4_6) args(%arg20=%alloc_3, %arg21=%alloc, %arg22=%alloc_1, %arg23=%alloc_2) : memref<8x4x16x8x8x8xf32, 2 : i32>, memref<8x1x64x256xbf16, 1 : i32>, memref<1x4x256x128xbf16, 1 : i32>, memref<8x4x64x128xf32, 1 : i32> {
          air.dma_memcpy_nd (%arg23[%arg16, %arg17, 0, 0] [1, 1, 64, 128] [32768, 8192, 128, 1], %arg20[%arg16, %arg17, 0, 0, 0, 0] [1, 1, 8, 8, 16, 8] [32768, 8192, 64, 8, 512, 1]) : (memref<8x4x64x128xf32, 1 : i32>, memref<8x4x16x8x8x8xf32, 2 : i32>)
        }
        air.dma_memcpy_nd (%arg15[%0, %1] [512, 512] [2048, 1], %alloc_2[0, 0, 0, 0] [8, 64, 4, 128] [32768, 128, 8192, 1]) : (memref<2048x2048xf32>, memref<8x4x64x128xf32, 1 : i32>)
        memref.dealloc %alloc : memref<8x1x64x256xbf16, 1 : i32>
        memref.dealloc %alloc_1 : memref<1x4x256x128xbf16, 1 : i32>
        memref.dealloc %alloc_2 : memref<8x4x64x128xf32, 1 : i32>
        memref.dealloc %alloc_3 : memref<8x4x16x8x8x8xf32, 2 : i32>
      }
    }
    air.launch () in () args(%arg4=%arg2, %arg5=%arg3) : memref<2048x2048xf32>, memref<2048x2048xbf16> {
      %collapse_shape = memref.collapse_shape %arg4 [[0, 1]] : memref<2048x2048xf32> into memref<4194304xf32>
      %collapse_shape_1 = memref.collapse_shape %arg5 [[0, 1]] : memref<2048x2048xbf16> into memref<4194304xbf16>
      air.segment @ct_cast_seg  args(%arg6=%collapse_shape, %arg7=%collapse_shape_1) : memref<4194304xf32>, memref<4194304xbf16> {
        %c8 = arith.constant 8 : index
        %c1 = arith.constant 1 : index
        air.herd @ct_herd_0  tile (%arg8, %arg9) in (%arg10=%c8, %arg11=%c1) args(%arg12=%arg6, %arg13=%arg7) : memref<4194304xf32>, memref<4194304xbf16> {
          %alloc = memref.alloc() : memref<2048xf32, 2 : i32>
          %alloc_2 = memref.alloc() : memref<2048xbf16, 2 : i32>
          %c0 = arith.constant 0 : index
          %cst = arith.constant 0.000000e+00 : f32
          %c0_3 = arith.constant 0 : index
          %c524288 = arith.constant 524288 : index
          %c2048 = arith.constant 2048 : index
          scf.for %arg14 = %c0_3 to %c524288 step %c2048 {
            %0 = affine.apply #map3()[%arg14, %arg8, %arg9]
            air.dma_memcpy_nd (%alloc[] [] [], %arg12[%0] [2048] [1]) : (memref<2048xf32, 2 : i32>, memref<4194304xf32>)
            %c0_4 = arith.constant 0 : index
            %c2048_5 = arith.constant 2048 : index
            %c16 = arith.constant 16 : index
            scf.for %arg15 = %c0_4 to %c2048_5 step %c16 {
              %subview = memref.subview %alloc[%arg15] [16] [1] : memref<2048xf32, 2 : i32> to memref<16xf32, strided<[1], offset: ?>, 2 : i32>
              %subview_6 = memref.subview %alloc_2[%arg15] [16] [1] : memref<2048xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
              %1 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>, 2 : i32>, vector<16xf32>
              %2 = arith.truncf %1 : vector<16xf32> to vector<16xbf16>
              vector.transfer_write %2, %subview_6[%c0] {in_bounds = [true]} : vector<16xbf16>, memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
            }
            air.dma_memcpy_nd (%arg13[%0] [2048] [1], %alloc_2[] [] []) : (memref<4194304xbf16>, memref<2048xbf16, 2 : i32>)
          }
          memref.dealloc %alloc : memref<2048xf32, 2 : i32>
          memref.dealloc %alloc_2 : memref<2048xbf16, 2 : i32>
        }
      }
    }
    return
  }
}
