// RUN: %iree_compile --compile-to=dispatch-creation %s | FileCheck %s
// RUN: %iree_compile --compile-to=dispatch-creation --coralnpu-roofline-speedup-threshold=3 %s | FileCheck %s --check-prefix=CHECK-HIGH-SPEEDUP
// RUN: %iree_compile --compile-to=dispatch-creation --coralnpu-roofline-speedup-threshold=0 %s | FileCheck %s --check-prefix=CHECK-ZERO
// RUN: iree-compile --mlir-disable-threading --iree-hal-target-device=coralnpu --compile-to=dispatch-creation %s | FileCheck %s --check-prefix=CHECK-NO-HOST

// Speedups (T_cpu / T_npu) at the default constants are noted per case.
// Function arguments are on the host, so CoralNPU pays to copy them in.

// CHECK-LABEL: @large_matmul
// CHECK-HIGH-SPEEDUP-LABEL: @large_matmul
func.func @large_matmul(
    %arg0: tensor<128x128xf32>,
    %arg1: tensor<128x128xf32>,
    %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> {
  // Large matmul, ~4.0.
  // CHECK: flow.dispatch.workgroups
  // CHECK: stream.affinity = #hal.device.affinity<@__device_1>
  // CHECK-HIGH-SPEEDUP: flow.dispatch.workgroups
  // CHECK-HIGH-SPEEDUP: stream.affinity = #hal.device.affinity<@__device_1>
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2 : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// CHECK-LABEL: @int8_matmul
// CHECK-HIGH-SPEEDUP-LABEL: @int8_matmul
func.func @int8_matmul(
    %arg0: tensor<64x64xi8>,
    %arg1: tensor<64x64xi8>,
    %arg2: tensor<64x64xi8>) -> tensor<64x64xi8> {
  // 8-bit tier on both devices, ~2.1 (~8.2 if only CoralNPU got it).
  // CHECK: flow.dispatch.workgroups
  // CHECK: stream.affinity = #hal.device.affinity<@__device_1>
  // CHECK-HIGH-SPEEDUP: flow.dispatch.workgroups
  // CHECK-HIGH-SPEEDUP: stream.affinity = #hal.device.affinity<@__device_0>
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<64x64xi8>, tensor<64x64xi8>)
                     outs(%arg2 : tensor<64x64xi8>) -> tensor<64x64xi8>
  return %0 : tensor<64x64xi8>
}

// CHECK-LABEL: @bf16_matmul
// CHECK-HIGH-SPEEDUP-LABEL: @bf16_matmul
func.func @bf16_matmul(
    %arg0: tensor<32x32xbf16>,
    %arg1: tensor<32x32xbf16>,
    %arg2: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
  // bf16 runs at the f32 rate on the host, ~1.9 (~1.0 otherwise).
  // CHECK: flow.dispatch.workgroups
  // CHECK: stream.affinity = #hal.device.affinity<@__device_1>
  // CHECK-HIGH-SPEEDUP: flow.dispatch.workgroups
  // CHECK-HIGH-SPEEDUP: stream.affinity = #hal.device.affinity<@__device_0>
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<32x32xbf16>, tensor<32x32xbf16>)
                     outs(%arg2 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
  return %0 : tensor<32x32xbf16>
}

// CHECK-LABEL: @mixed_precision_matmul
// CHECK-HIGH-SPEEDUP-LABEL: @mixed_precision_matmul
func.func @mixed_precision_matmul(
    %arg0: tensor<32x32xbf16>,
    %arg1: tensor<32x32xbf16>) -> tensor<32x32xf32> {
  // Casts are not counted, ~2.4 (~4.9 if the two extf ops were).
  // CHECK: flow.dispatch.workgroups
  // CHECK: stream.affinity = #hal.device.affinity<@__device_1>
  // CHECK-HIGH-SPEEDUP: flow.dispatch.workgroups
  // CHECK-HIGH-SPEEDUP: stream.affinity = #hal.device.affinity<@__device_0>
  %empty = tensor.empty() : tensor<32x32xf32>
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<32x32xbf16>, tensor<32x32xbf16>)
                     outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// CHECK-LABEL: @chain
// CHECK-HIGH-SPEEDUP-LABEL: @chain
func.func @chain(
    %arg0: tensor<128x128xf32>,
    %arg1: tensor<128x128xf32>,
    %arg2: tensor<128x128xf32>,
    %arg3: tensor<64x256xf32>,
    %arg4: tensor<256x1xf32>,
    %arg5: tensor<64x1xf32>) -> (tensor<64x1xf32>, tensor<64x1xf32>) {
  // The same GEMV stays on CoralNPU when its input is produced there, even
  // through a flow.tensor.reshape (~8.8), but goes to the host when its input
  // is on the host (~0.4).
  // CHECK: flow.dispatch.workgroups
  // CHECK: stream.affinity = #hal.device.affinity<@__device_1>
  // CHECK: flow.tensor.reshape
  // CHECK: flow.dispatch.workgroups
  // CHECK: stream.affinity = #hal.device.affinity<@__device_1>
  // CHECK: flow.dispatch.workgroups
  // CHECK: stream.affinity = #hal.device.affinity<@__device_0>
  // CHECK-HIGH-SPEEDUP: flow.dispatch.workgroups
  // CHECK-HIGH-SPEEDUP: stream.affinity = #hal.device.affinity<@__device_1>
  // CHECK-HIGH-SPEEDUP: flow.dispatch.workgroups
  // CHECK-HIGH-SPEEDUP: stream.affinity = #hal.device.affinity<@__device_1>
  // CHECK-HIGH-SPEEDUP: flow.dispatch.workgroups
  // CHECK-HIGH-SPEEDUP: stream.affinity = #hal.device.affinity<@__device_0>
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2 : tensor<128x128xf32>) -> tensor<128x128xf32>
  %1 = tensor.collapse_shape %0 [[0, 1]] : tensor<128x128xf32> into tensor<16384xf32>
  %2 = tensor.expand_shape %1 [[0, 1]] output_shape [64, 256] : tensor<16384xf32> into tensor<64x256xf32>
  %3 = linalg.matmul ins(%2, %arg4 : tensor<64x256xf32>, tensor<256x1xf32>)
                     outs(%arg5 : tensor<64x1xf32>) -> tensor<64x1xf32>
  %4 = linalg.matmul ins(%arg3, %arg4 : tensor<64x256xf32>, tensor<256x1xf32>)
                     outs(%arg5 : tensor<64x1xf32>) -> tensor<64x1xf32>
  return %3, %4 : tensor<64x1xf32>, tensor<64x1xf32>
}

// CHECK-LABEL: @small_matmul
// CHECK-HIGH-SPEEDUP-LABEL: @small_matmul
// CHECK-ZERO-LABEL: @small_matmul
// CHECK-NO-HOST-LABEL: @small_matmul
func.func @small_matmul(
    %arg0: tensor<2x2xf32>,
    %arg1: tensor<2x2xf32>,
    %arg2: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // Launch- and copy-bound, ~0.02; CoralNPU at 0 or without a host (where
  // CoralNPU is the only device, @__device_0).
  // CHECK: flow.dispatch.workgroups
  // CHECK: stream.affinity = #hal.device.affinity<@__device_0>
  // CHECK-ZERO: stream.affinity = #hal.device.affinity<@__device_1>
  // CHECK-NO-HOST: stream.affinity = #hal.device.affinity<@__device_0>
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<2x2xf32>, tensor<2x2xf32>)
                     outs(%arg2 : tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// The extra labels bound the preceding checks.
// CHECK-LABEL: @unsupported_i64
// CHECK-ZERO-LABEL: @unsupported_i64
// CHECK-NO-HOST-LABEL: @unsupported_i64
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

// CHECK-LABEL: @sort
func.func @sort(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  // Non-linalg root (LinalgExt) is assigned to the host device.
  // CHECK: flow.dispatch.workgroups
  // CHECK: stream.affinity = #hal.device.affinity<@__device_0>
  %0 = iree_linalg_ext.sort dimension(0) outs(%arg0 : tensor<128xf32>) {
  ^bb0(%lhs: f32, %rhs: f32):
    %cmp = arith.cmpf olt, %lhs, %rhs : f32
    iree_linalg_ext.yield %cmp : i1
  } -> tensor<128xf32>
  return %0 : tensor<128xf32>
}

// CHECK-LABEL: @elementwise_add
func.func @elementwise_add(
    %arg0: tensor<32x32xf32>,
    %arg1: tensor<32x32xf32>,
    %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // Supported on CoralNPU, but memory- and copy-bound, ~0.5.
  // CHECK: flow.dispatch.workgroups
  // CHECK: stream.affinity = #hal.device.affinity<@__device_0>
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

util.global private @weight = #stream.parameter.named<"model"::"weight"> : tensor<64x256xf32>
util.global private @big_weight = #stream.parameter.named<"model"::"big_weight"> : tensor<512x4096xf32>

// CHECK-LABEL: @global_weight
func.func @global_weight(
    %arg0: tensor<256x1xf32>,
    %arg1: tensor<64x1xf32>) -> tensor<64x1xf32> {
  // The GEMV from @chain; weights are placed where used, so no copy, ~3.0.
  // CHECK: flow.dispatch.workgroups
  // CHECK: stream.affinity = #hal.device.affinity<@__device_1>
  %w = util.global.load @weight : tensor<64x256xf32>
  %0 = linalg.matmul ins(%w, %arg0 : tensor<64x256xf32>, tensor<256x1xf32>)
                     outs(%arg1 : tensor<64x1xf32>) -> tensor<64x1xf32>
  return %0 : tensor<64x1xf32>
}

// CHECK-LABEL: @spill
func.func @spill(
    %arg0: tensor<1x512xf32>,
    %arg1: tensor<1x4096xf32>) -> tensor<1x4096xf32> {
  // 8 MiB of weights spill past the 4 MiB window, ~0.9 (~3.9 without spill).
  // CHECK: flow.dispatch.workgroups
  // CHECK: stream.affinity = #hal.device.affinity<@__device_0>
  %w = util.global.load @big_weight : tensor<512x4096xf32>
  %0 = linalg.matmul ins(%arg0, %w : tensor<1x512xf32>, tensor<512x4096xf32>)
                     outs(%arg1 : tensor<1x4096xf32>) -> tensor<1x4096xf32>
  return %0 : tensor<1x4096xf32>
}
