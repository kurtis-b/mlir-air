// Crash input for the split-L2 multi-row-herd abort (goal 4d, PR #83).
//
// NOT a lit test: `Inputs/` is excluded from the testsuite (mlir/test/lit.cfg.py:93),
// because running the pass below ABORTS air-opt. It is committed so the assertion, and
// any candidate fix for it, can be reproduced by anyone in about a second:
//
//   air-opt Inputs/split_l2_herd_rows2_pre_split.mlir \\
//     --air-split-l2-memref="max-launch-channels-mm2s=16 max-launch-channels-s2mm=16 tiles-per-l2-tile=4"
//
//   -> air-opt: mlir/lib/IR/MLIRContext.cpp:1251: AffineMap::get(...):
//      Assertion `willBeValidAffineMap(...)' failed.
//      AIRSplitL2MemrefForBufferConstraintPass::runOnOperation()
//        -> xilinx::tileChannelOpByFactor(...)
//
// Provenance: emitted by aircc's own 24-pass prefix (printed by `aircc -v`) up to
// air-split-l2-memref, from the matvec GEMV example built with a 2-row herd --
// branch feat/matvec-herd-rows, --m 2048 --k 8192 --tile-m 2 --m-input 1 --herd-m 4
// --herd-rows 2. The options above are the ones aircc passes at that stage.

#map = affine_map<()[s0] -> (s0 * 16)>
#map1 = affine_map<()[s0, s1, s2] -> (s0 * 4 + s1 * 2 + s2)>
#map2 = affine_map<()[s0, s1] -> (s0 * 4 + s1 * 2)>
#set = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 >= 0, -s1 + 1 >= 0)>
module {
  air.channel @channel_0 []
  air.channel @channel_1 [1, 1] {broadcast_shape = [4, 2]}
  air.channel @channel_2 [4, 2]
  air.channel @channel_3 [4, 2]
  air.channel @channel_4 []
  func.func private @matvec_vectorized_bf16_bf16(i32, i32, i32, memref<1x8192xbf16, 2 : i32>, memref<8192xbf16, 2 : i32>, memref<2xbf16, 2 : i32>) attributes {link_with = "mv.o", llvm.emit_c_interface}
  func.func private @linalg_fill_bf16(memref<2xbf16, 2 : i32>) attributes {link_with = "mv.o", llvm.emit_c_interface}
  func.func @matvec_bf16(%arg0: memref<2048x8192xbf16>, %arg1: memref<8192xbf16>, %arg2: memref<2048xbf16>) {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg3, %arg4) in (%arg5=%c128, %arg6=%c1) args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg2) : memref<2048x8192xbf16>, memref<8192xbf16>, memref<2048xbf16> attributes {id = 1 : i32} {
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c1_0 = arith.constant 1 : index
      %1 = affine.apply #map()[%arg3]
      %2 = air.channel.put async  @channel_0[] (%arg7[%1, 0] [16, 8192] [8192, 1]) {id = 1 : i32} : (memref<2048x8192xbf16>)
      %3 = air.wait_all async 
      %4 = scf.for %arg10 = %c0 to %c2 step %c1_0 iter_args(%arg11 = %3) -> (!air.async.token) {
        %7 = air.channel.put async [%arg11]  @channel_1[] (%arg8[] [] []) {id = 2 : i32} : (memref<8192xbf16>)
        scf.yield %7 : !air.async.token
      }
      %5 = air.channel.get async  @channel_4[] (%arg9[%1] [16] [1]) {id = 3 : i32} : (memref<2048xbf16>)
      %6 = air.segment @matvec_bf16_0 async  attributes {id = 2 : i32} {
        %c0_1 = arith.constant 0 : index
        %c1_2 = arith.constant 1 : index
        %c2_3 = arith.constant 2 : index
        %c4 = arith.constant 4 : index
        %async_token, %results = air.execute -> (memref<16x8192xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<16x8192xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<16x8192xbf16, 1 : i32>
        }
        %async_token_4, %results_5 = air.execute -> (memref<16xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<16xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<16xbf16, 1 : i32>
        }
        %7 = air.channel.get async [%async_token]  @channel_0[] (%results[] [] []) {id = 4 : i32} : (memref<16x8192xbf16, 1 : i32>)
        %8 = air.wait_all async [%async_token_4, %7] 
        %9 = scf.parallel (%arg10, %arg11) = (%c0_1, %c0_1) to (%c4, %c2_3) step (%c1_2, %c1_2) init (%8) -> !air.async.token {
          %13 = air.wait_all async [%async_token_4, %7] 
          %14 = scf.for %arg12 = %c0_1 to %c2_3 step %c1_2 iter_args(%arg13 = %13) -> (!air.async.token) {
            %15 = affine.apply #map1()[%arg10, %arg11, %arg12]
            %16 = air.channel.put async [%arg13]  @channel_2[%arg10, %arg11] (%results[%15, 0] [1, 8192] [8192, 1]) {id = 5 : i32} : (memref<16x8192xbf16, 1 : i32>)
            scf.yield %16 : !air.async.token
          }
          scf.reduce(%14 : !air.async.token) {
          ^bb0(%arg12: !air.async.token, %arg13: !air.async.token):
            %15 = air.wait_all async [%arg12, %arg13] 
            scf.reduce.return %15 : !air.async.token
          }
        }
        %10 = scf.parallel (%arg10, %arg11) = (%c0_1, %c0_1) to (%c4, %c2_3) step (%c1_2, %c1_2) init (%async_token_4) -> !air.async.token {
          %13 = affine.apply #map2()[%arg10, %arg11]
          %14 = air.channel.get async [%async_token_4]  @channel_3[%arg10, %arg11] (%results_5[%13] [2] [1]) {id = 6 : i32} : (memref<16xbf16, 1 : i32>)
          scf.reduce(%14 : !air.async.token) {
          ^bb0(%arg12: !air.async.token, %arg13: !air.async.token):
            %15 = air.wait_all async [%arg12, %arg13] 
            scf.reduce.return %15 : !air.async.token
          }
        }
        %11 = air.herd @herd_0 async [%async_token_4]  tile (%arg10, %arg11) in (%arg12=%c4, %arg13=%c2_3) attributes {id = 3 : i32, link_with = "mv.o"} {
          %c8192_i32 = arith.constant 8192 : i32
          %c1_i32 = arith.constant 1 : i32
          %c1_8 = arith.constant 1 : index
          %c2_9 = arith.constant 2 : index
          %c0_10 = arith.constant 0 : index
          %async_token_11, %results_12 = air.execute -> (memref<1x8192xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<1x8192xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<1x8192xbf16, 2 : i32>
          }
          %async_token_13, %results_14 = air.execute -> (memref<8192xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<8192xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<8192xbf16, 2 : i32>
          }
          %async_token_15, %results_16 = air.execute -> (memref<2xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<2xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<2xbf16, 2 : i32>
          }
          %async_token_17 = air.execute [%async_token_15] {
            func.call @linalg_fill_bf16(%results_16) : (memref<2xbf16, 2 : i32>) -> ()
          }
          %13 = air.wait_all async [%async_token_11, %async_token_13, %async_token_17] 
          %14 = scf.for %arg14 = %c0_10 to %c2_9 step %c1_8 iter_args(%arg15 = %13) -> (!air.async.token) {
            %16 = affine.if #set()[%arg10, %arg11] -> !air.async.token {
              %19 = air.channel.get async [%arg15]  @channel_1[%arg10, %arg11] (%results_14[] [] []) {id = 7 : i32} : (memref<8192xbf16, 2 : i32>)
              affine.yield %19 : !air.async.token
            } else {
              affine.yield %arg15 : !air.async.token
            }
            %17 = air.channel.get async [%arg15]  @channel_2[%arg10, %arg11] (%results_12[] [] []) {id = 8 : i32} : (memref<1x8192xbf16, 2 : i32>)
            %18 = arith.index_cast %arg14 : index to i32
            %async_token_21 = air.execute [%17, %16] {
              func.call @matvec_vectorized_bf16_bf16(%c1_i32, %c8192_i32, %18, %results_12, %results_14, %results_16) : (i32, i32, i32, memref<1x8192xbf16, 2 : i32>, memref<8192xbf16, 2 : i32>, memref<2xbf16, 2 : i32>) -> ()
            }
            scf.yield %async_token_21 : !air.async.token
          }
          %async_token_18 = air.execute [%14] {
            memref.dealloc %results_14 : memref<8192xbf16, 2 : i32>
          }
          %async_token_19 = air.execute [%14] {
            memref.dealloc %results_12 : memref<1x8192xbf16, 2 : i32>
          }
          %15 = air.channel.put async [%14]  @channel_3[%arg10, %arg11] (%results_16[] [] []) {id = 9 : i32} : (memref<2xbf16, 2 : i32>)
          %async_token_20 = air.execute [%15] {
            memref.dealloc %results_16 : memref<2xbf16, 2 : i32>
          }
        }
        %async_token_6 = air.execute [%9] {
          memref.dealloc %results : memref<16x8192xbf16, 1 : i32>
        }
        %12 = air.channel.put async [%11]  @channel_4[] (%results_5[] [] []) {id = 10 : i32} : (memref<16xbf16, 1 : i32>)
        %async_token_7 = air.execute [%12] {
          memref.dealloc %results_5 : memref<16xbf16, 1 : i32>
        }
      }
    }
    return
  }
}

