// RUN: triton-opt --triton-to-linalg --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @store_v2_no_mask
// CHECK-NOT: bufferization.materialize_in_destination
// CHECK: hfusion.store_v2
// CHECK-SAME: ins(
// CHECK-SAME: outs(
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @store_v2_no_mask(%dst_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %data = arith.constant dense<1.000000e+00> : tensor<8xf32>
    %offs = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %base = tt.splat %dst_ptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %ptr = tt.addptr %base, %offs : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    tt.store %ptr, %data : tensor<8x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @store_v2_boundary_check
// CHECK: tensor.extract_slice
// CHECK: memref.subview
// CHECK-NOT: bufferization.materialize_in_destination
// CHECK: hfusion.store_v2
// CHECK-SAME: ins(
// CHECK-SAME: outs(
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @store_v2_boundary_check(%dst_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %off0: i32, %off1: i32) attributes {noinline = false} {
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c8_i64 = arith.constant 8 : i64
    %data = arith.constant dense<1.000000e+00> : tensor<4x8xf32>
    %store_ptr = tt.make_tensor_ptr %dst_ptr, [%c64_i64, %c8_i64], [%c8_i64, %c1_i64], [%off0, %off1] {order = array<i32: 1, 0>} : <tensor<4x8xf32>>
    tt.store %store_ptr, %data {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<4x8xf32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @store_v2_continuous_mask
// CHECK: tensor.extract_slice
// CHECK: memref.subview
// CHECK-NOT: bufferization.materialize_in_destination
// CHECK: hfusion.store_v2
// CHECK-SAME: ins(
// CHECK-SAME: outs(
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @store_v2_continuous_mask(%dst_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %numel: i32) attributes {noinline = false} {
    %data = arith.constant dense<1.000000e+00> : tensor<8xf32>
    %offs = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %limit = tt.splat %numel : i32 -> tensor<8xi32>
    %mask = arith.cmpi slt, %offs, %limit : tensor<8xi32>
    %base = tt.splat %dst_ptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %ptr = tt.addptr %base, %offs : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    tt.store %ptr, %data, %mask : tensor<8x!tt.ptr<f32>>
    tt.return
  }
}
