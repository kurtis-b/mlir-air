#map = affine_map<()[s0, s1] -> (s0 + s1 * 8)>
module {
  func.func private @fused_add_layer_norm_1outs(memref<8x768xbf16, 2 : i32>, memref<8x768xbf16, 2 : i32>, memref<768xbf16, 2 : i32>, memref<8x768xbf16, 2 : i32>, i32, i32) attributes {link_with = "addnorm_pre_add.o", llvm.emit_c_interface}
  func.func @addnorm(%arg0: memref<64x768xbf16>, %arg1: memref<64x768xbf16>, %arg2: memref<768xbf16>, %arg3: memref<64x768xbf16>) {
    air.launch () in () args(%arg4=%arg0, %arg5=%arg1, %arg6=%arg2, %arg7=%arg3) : memref<64x768xbf16>, memref<64x768xbf16>, memref<768xbf16>, memref<64x768xbf16> {
      air.segment @addnorm_seg  args(%arg8=%arg4, %arg9=%arg5, %arg10=%arg6, %arg11=%arg7) : memref<64x768xbf16>, memref<64x768xbf16>, memref<768xbf16>, memref<64x768xbf16> {
        %c8 = arith.constant 8 : index
        %c1 = arith.constant 1 : index
        air.herd @addnorm_herd  tile (%arg12, %arg13) in (%arg14=%c8, %arg15=%c1) args(%arg16=%arg8, %arg17=%arg9, %arg18=%arg10, %arg19=%arg11) : memref<64x768xbf16>, memref<64x768xbf16>, memref<768xbf16>, memref<64x768xbf16> attributes {link_with = "addnorm_pre_add.o"} {
          %alloc = memref.alloc() : memref<8x768xbf16, 2 : i32>
          %alloc_0 = memref.alloc() : memref<8x768xbf16, 2 : i32>
          %alloc_1 = memref.alloc() : memref<768xbf16, 2 : i32>
          %alloc_2 = memref.alloc() : memref<8x768xbf16, 2 : i32>
          %c768_i32 = arith.constant 768 : i32
          %c8_i32 = arith.constant 8 : i32
          %c0 = arith.constant 0 : index
          %c8_3 = arith.constant 8 : index
          %c8_4 = arith.constant 8 : index
          scf.for %arg20 = %c0 to %c8_3 step %c8_4 {
            %0 = affine.apply #map()[%arg20, %arg12]
            air.dma_memcpy_nd (%alloc_1[] [] [], %arg18[0] [768] [1]) : (memref<768xbf16, 2 : i32>, memref<768xbf16>)
            air.dma_memcpy_nd (%alloc[] [] [], %arg16[%0, 0] [8, 768] [768, 1]) : (memref<8x768xbf16, 2 : i32>, memref<64x768xbf16>)
            air.dma_memcpy_nd (%alloc_0[] [] [], %arg17[%0, 0] [8, 768] [768, 1]) : (memref<8x768xbf16, 2 : i32>, memref<64x768xbf16>)
            func.call @fused_add_layer_norm_1outs(%alloc, %alloc_0, %alloc_1, %alloc_2, %c768_i32, %c8_i32) : (memref<8x768xbf16, 2 : i32>, memref<8x768xbf16, 2 : i32>, memref<768xbf16, 2 : i32>, memref<8x768xbf16, 2 : i32>, i32, i32) -> ()
            air.dma_memcpy_nd (%arg19[%0, 0] [8, 768] [768, 1], %alloc_2[] [] []) : (memref<64x768xbf16>, memref<8x768xbf16, 2 : i32>)
          }
          memref.dealloc %alloc : memref<8x768xbf16, 2 : i32>
          memref.dealloc %alloc_0 : memref<8x768xbf16, 2 : i32>
          memref.dealloc %alloc_1 : memref<768xbf16, 2 : i32>
          memref.dealloc %alloc_2 : memref<8x768xbf16, 2 : i32>
        }
      }
    }
    return
  }
}
