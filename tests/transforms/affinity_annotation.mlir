// RUN: %iree_compile --compile-to=dispatch-creation --coralnpu-affinity-io-min-threshold-kb=1 --coralnpu-affinity-io-max-threshold-kb=20 %s | FileCheck %s

// CHECK-LABEL: @big_matmul
func.func @big_matmul(
    %arg0: tensor<32x32xf32>,
    %arg1: tensor<32x32xf32>,
    %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // 32x32xf32 = 4096 bytes per tensor, total 12 KB (1 KB <= 12 KB <= 20 KB).
  // CHECK: flow.dispatch.workgroups
  // CHECK: stream.affinity = #hal.device.affinity<@__device_1>
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<32x32xf32>, tensor<32x32xf32>)
                     outs(%arg2 : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// CHECK-LABEL: @huge_matmul
func.func @huge_matmul(
    %arg0: tensor<64x64xf32>,
    %arg1: tensor<64x64xf32>,
    %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> {
  // 64x64xf32 = 16384 bytes per tensor, total 48 KB > max threshold (20 KB).
  // CHECK: flow.dispatch.workgroups
  // CHECK-NOT: stream.affinity
  // CHECK: util.return
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<64x64xf32>, tensor<64x64xf32>)
                     outs(%arg2 : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// CHECK-LABEL: @small_matmul
func.func @small_matmul(
    %arg0: tensor<2x2xf32>,
    %arg1: tensor<2x2xf32>,
    %arg2: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // 2x2xf32 = 16 bytes per tensor, total 48 bytes < min threshold (1 KB).
  // Below threshold: left unannotated, falling back to stream.affinity.default.
  // CHECK: flow.dispatch.workgroups
  // CHECK-NOT: stream.affinity
  // CHECK: util.return
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<2x2xf32>, tensor<2x2xf32>)
                     outs(%arg2 : tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// CHECK-LABEL: @unsupported_i64
func.func @unsupported_i64(
    %arg0: tensor<32x32xi64>,
    %arg1: tensor<32x32xi64>,
    %arg2: tensor<32x32xi64>) -> tensor<32x32xi64> {
  // i64 is not supported on CoralNPU, assigned to host device.
  // CHECK: flow.dispatch.workgroups
  // CHECK: stream.affinity = #hal.device.affinity<@__device_0>
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<32x32xi64>, tensor<32x32xi64>)
                     outs(%arg2 : tensor<32x32xi64>) -> tensor<32x32xi64>
  return %0 : tensor<32x32xi64>
}

// CHECK-LABEL: @dynamic_shape
func.func @dynamic_shape(
    %arg0: tensor<?x?xf32>,
    %arg1: tensor<?x?xf32>,
    %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // Dynamic shapes are not supported on CoralNPU, assigned to host device.
  // CHECK: flow.dispatch.workgroups
  // CHECK: stream.affinity = #hal.device.affinity<@__device_0>
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: @elementwise_add
func.func @elementwise_add(
    %arg0: tensor<32x32xf32>,
    %arg1: tensor<32x32xf32>,
    %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // Elementwise linalg.generic is supported on CoralNPU.
  // CHECK: flow.dispatch.workgroups
  // CHECK: stream.affinity = #hal.device.affinity<@__device_1>
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<32x32xf32>, tensor<32x32xf32>)
    outs(%arg2 : tensor<32x32xf32>) {
  ^bb0(%in1: f32, %in2: f32, %out: f32):
    %add = arith.addf %in1, %in2 : f32
    linalg.yield %add : f32
  } -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}
