// RUN: %iree_compile --compile-to=preprocessing %s | FileCheck %s

// Global preprocessing promotes even when CoralNPU is not a target device.
// RUN: iree-compile --mlir-disable-threading \
// RUN:   --iree-hal-target-device=local \
// RUN:   --iree-hal-local-target-device-backends=llvm-cpu \
// RUN:   --compile-to=preprocessing %s | FileCheck %s --check-prefix=CHECK-CPU

// CHECK-LABEL: util.func public @bf16_matmul
// CHECK-CPU-LABEL: util.func public @bf16_matmul
func.func @bf16_matmul(%a: tensor<128x256xbf16>,
                       %b: tensor<256x128xbf16>) -> tensor<128x128xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %init = tensor.empty() : tensor<128x128xbf16>
  %fill = linalg.fill ins(%cst : bf16)
      outs(%init : tensor<128x128xbf16>) -> tensor<128x128xbf16>

  // Accumulator fill is rebuilt directly in FP32.
  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<128x128xf32>
  // CHECK: %[[FILL:.+]] = linalg.fill
  // CHECK-SAME: outs(%[[EMPTY]] : tensor<128x128xf32>) -> tensor<128x128xf32>

  // Inputs stay BF16; original affinity carries over.
  // CHECK: linalg.matmul
  // CHECK-SAME: stream.affinity = #hal.device.affinity<@__device_1>
  // CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<128x256xbf16>, tensor<256x128xbf16>)
  // CHECK-SAME: outs(%[[FILL]] : tensor<128x128xf32>) -> tensor<128x128xf32>

  // CHECK-CPU: linalg.matmul
  // CHECK-CPU-SAME: ins(%{{.+}}, %{{.+}} : tensor<128x256xbf16>, tensor<256x128xbf16>)
  // CHECK-CPU-SAME: outs(%{{.+}} : tensor<128x128xf32>) -> tensor<128x128xf32>
  %0 = linalg.matmul ins(%a, %b : tensor<128x256xbf16>, tensor<256x128xbf16>)
      outs(%fill : tensor<128x128xbf16>) -> tensor<128x128xbf16>

  // No attrs before `ins`: unannotated truncation fuses into producer.
  // CHECK: linalg.copy ins(%{{.+}} : tensor<128x128xf32>)
  return %0 : tensor<128x128xbf16>
}

// Non-default inherent `indexing_maps` (transpose-B) must be preserved.
#map_lhs = affine_map<(d0, d1, d2) -> (d0, d2)>
#map_rhs_t = affine_map<(d0, d1, d2) -> (d1, d2)>
#map_out = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: util.func public @bf16_matmul_transpose_b
func.func @bf16_matmul_transpose_b(
    %a: tensor<128x256xbf16>,
    %b: tensor<128x256xbf16>) -> tensor<128x128xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %init = tensor.empty() : tensor<128x128xbf16>
  %fill = linalg.fill ins(%cst : bf16)
      outs(%init : tensor<128x128xbf16>) -> tensor<128x128xbf16>

  // CHECK: linalg.matmul
  // CHECK-SAME: indexing_maps = [{{.+}}, {{.+}}, {{.+}}]
  // CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<128x256xbf16>, tensor<128x256xbf16>)
  // CHECK-SAME: outs(%{{.+}} : tensor<128x128xf32>) -> tensor<128x128xf32>
  %0 = linalg.matmul
      indexing_maps = [#map_lhs, #map_rhs_t, #map_out]
      ins(%a, %b : tensor<128x256xbf16>, tensor<128x256xbf16>)
      outs(%fill : tensor<128x128xbf16>) -> tensor<128x128xbf16>
  return %0 : tensor<128x128xbf16>
}

// Non-fill BF16 accumulator is widened before and narrowed after.
// CHECK-LABEL: util.func public @bf16_matmul_live_init
func.func @bf16_matmul_live_init(
    %a: tensor<128x256xbf16>,
    %b: tensor<256x128xbf16>,
    %init: tensor<128x128xbf16>) -> tensor<128x128xbf16> {
  // CHECK: linalg.copy ins(%{{.+}} : tensor<128x128xbf16>)
  // CHECK: linalg.matmul
  // CHECK-SAME: outs(%{{.+}} : tensor<128x128xf32>) -> tensor<128x128xf32>
  // CHECK: linalg.copy ins(%{{.+}} : tensor<{{.+}}xf32>)
  %0 = linalg.matmul ins(%a, %b : tensor<128x256xbf16>, tensor<256x128xbf16>)
      outs(%init : tensor<128x128xbf16>) -> tensor<128x128xbf16>
  return %0 : tensor<128x128xbf16>
}

// The FP32 accumulator takes its dynamic sizes from the BF16 init.
// CHECK-LABEL: util.func public @bf16_matmul_dynamic
func.func @bf16_matmul_dynamic(%a: tensor<?x?xbf16>,
                               %b: tensor<?x?xbf16>) -> tensor<?x?xbf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : bf16
  %m = tensor.dim %a, %c0 : tensor<?x?xbf16>
  %n = tensor.dim %b, %c1 : tensor<?x?xbf16>
  %init = tensor.empty(%m, %n) : tensor<?x?xbf16>
  %fill = linalg.fill ins(%cst : bf16)
      outs(%init : tensor<?x?xbf16>) -> tensor<?x?xbf16>

  // CHECK: %[[EMPTY:.+]] = tensor.empty(%{{.+}}, %{{.+}}) : tensor<?x?xf32>
  // CHECK: %[[FILL:.+]] = linalg.fill
  // CHECK-SAME: outs(%[[EMPTY]] : tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK: linalg.matmul
  // CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<?x?xbf16>, tensor<?x?xbf16>)
  // CHECK-SAME: outs(%[[FILL]] : tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK: linalg.copy ins(%{{.+}} : tensor<{{.+}}xf32>)
  %0 = linalg.matmul ins(%a, %b : tensor<?x?xbf16>, tensor<?x?xbf16>)
      outs(%fill : tensor<?x?xbf16>) -> tensor<?x?xbf16>
  return %0 : tensor<?x?xbf16>
}

// CHECK-LABEL: util.func public @bf16_vecmat
// CHECK-CPU-LABEL: util.func public @bf16_vecmat
func.func @bf16_vecmat(%a: tensor<256xbf16>,
                       %b: tensor<256x128xbf16>) -> tensor<128xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %init = tensor.empty() : tensor<128xbf16>
  %fill = linalg.fill ins(%cst : bf16)
      outs(%init : tensor<128xbf16>) -> tensor<128xbf16>

  // CHECK: linalg.vecmat
  // CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<256xbf16>, tensor<256x128xbf16>)
  // CHECK-SAME: outs(%{{.+}} : tensor<128xf32>) -> tensor<128xf32>
  // CHECK: linalg.copy ins(%{{.+}} : tensor<{{.+}}xf32>)

  // CHECK-CPU: linalg.vecmat
  // CHECK-CPU-SAME: outs(%{{.+}} : tensor<128xf32>) -> tensor<128xf32>
  %0 = linalg.vecmat ins(%a, %b : tensor<256xbf16>, tensor<256x128xbf16>)
      outs(%fill : tensor<128xbf16>) -> tensor<128xbf16>
  return %0 : tensor<128xbf16>
}

// CHECK-LABEL: util.func public @bf16_conv_1d
func.func @bf16_conv_1d(%input: tensor<2x120x3xbf16>,
                        %filter: tensor<5x3x32xbf16>) -> tensor<2x116x32xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %init = tensor.empty() : tensor<2x116x32xbf16>
  %fill = linalg.fill ins(%cst : bf16)
      outs(%init : tensor<2x116x32xbf16>) -> tensor<2x116x32xbf16>

  // CHECK: linalg.conv_1d_nwc_wcf
  // CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<2x120x3xbf16>, tensor<5x3x32xbf16>)
  // CHECK-SAME: outs(%{{.+}} : tensor<2x116x32xf32>) -> tensor<2x116x32xf32>
  // CHECK: linalg.copy ins(%{{.+}} : tensor<{{.+}}xf32>)
  %0 = linalg.conv_1d_nwc_wcf {dilations = dense<1> : vector<1xi64>,
                               strides = dense<1> : vector<1xi64>}
      ins(%input, %filter : tensor<2x120x3xbf16>, tensor<5x3x32xbf16>)
      outs(%fill : tensor<2x116x32xbf16>) -> tensor<2x116x32xbf16>
  return %0 : tensor<2x116x32xbf16>
}

// CHECK-LABEL: util.func public @bf16_reduce
func.func @bf16_reduce(%input: tensor<120x256xbf16>) -> tensor<120xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %init = tensor.empty() : tensor<120xbf16>
  %fill = linalg.fill ins(%cst : bf16)
      outs(%init : tensor<120xbf16>) -> tensor<120xbf16>

  // CHECK: linalg.reduce
  // CHECK-SAME: ins(%{{.+}} : tensor<120x256xbf16>)
  // CHECK-SAME: outs(%{{.+}} : tensor<120xf32>)
  // CHECK: linalg.copy ins(%{{.+}} : tensor<{{.+}}xf32>)
  %0 = linalg.reduce ins(%input : tensor<120x256xbf16>)
      outs(%fill : tensor<120xbf16>) dimensions = [1]
    (%in: bf16, %out: bf16) {
      %add = arith.addf %in, %out : bf16
      linalg.yield %add : bf16
    }
  return %0 : tensor<120xbf16>
}

#map_red_in = affine_map<(d0, d1) -> (d0, d1)>
#map_red_out = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: util.func public @bf16_generic_reduction
func.func @bf16_generic_reduction(
    %input: tensor<120x256xbf16>) -> tensor<120xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %init = tensor.empty() : tensor<120xbf16>
  %fill = linalg.fill ins(%cst : bf16)
      outs(%init : tensor<120xbf16>) -> tensor<120xbf16>

  // CHECK: linalg.generic
  // CHECK-SAME: ins(%{{.+}} : tensor<120x256xbf16>)
  // CHECK-SAME: outs(%{{.+}} : tensor<120xf32>)
  // CHECK: linalg.copy ins(%{{.+}} : tensor<{{.+}}xf32>)
  %0 = linalg.generic {
      indexing_maps = [#map_red_in, #map_red_out],
      iterator_types = ["parallel", "reduction"]}
      ins(%input : tensor<120x256xbf16>) outs(%fill : tensor<120xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %add = arith.addf %in, %out : bf16
      linalg.yield %add : bf16
    } -> tensor<120xbf16>
  return %0 : tensor<120xbf16>
}

// Generic contraction: body cloned in FP32.
#map_bmm_lhs = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3)>
#map_bmm_rhs = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map_bmm_out = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// CHECK-LABEL: util.func public @bf16_generic_contraction
func.func @bf16_generic_contraction(
    %lhs: tensor<16x4x32xbf16>,
    %rhs: tensor<4x32x8xbf16>) -> tensor<4x16x8xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %init = tensor.empty() : tensor<4x16x8xbf16>
  %fill = linalg.fill ins(%cst : bf16)
      outs(%init : tensor<4x16x8xbf16>) -> tensor<4x16x8xbf16>

  // CHECK: linalg.generic
  // CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<16x4x32xbf16>, tensor<4x32x8xbf16>)
  // CHECK-SAME: outs(%{{.+}} : tensor<4x16x8xf32>)
  // CHECK: arith.mulf %{{.+}}, %{{.+}} : f32
  // CHECK: arith.addf %{{.+}}, %{{.+}} : f32
  // CHECK: linalg.copy
  // CHECK-SAME: -> tensor<4x16x8xbf16>
  %0 = linalg.generic {
      indexing_maps = [#map_bmm_lhs, #map_bmm_rhs, #map_bmm_out],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%lhs, %rhs : tensor<16x4x32xbf16>, tensor<4x32x8xbf16>)
      outs(%fill : tensor<4x16x8xbf16>) {
    ^bb0(%a: bf16, %b: bf16, %out: bf16):
      %mul = arith.mulf %a, %b : bf16
      %add = arith.addf %out, %mul : bf16
      linalg.yield %add : bf16
    } -> tensor<4x16x8xbf16>
  return %0 : tensor<4x16x8xbf16>
}

// Lowered `stablehlo.reduce_window` sum: f32 window, non-fill init.
#map_rw_in = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d3)>
#map_rw_win = affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>
#map_rw_out = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>

// CHECK-LABEL: util.func public @bf16_reduce_window_sum
func.func @bf16_reduce_window_sum(
    %input: tensor<1x16x16x4xbf16>,
    %init: tensor<1x8x8x4xbf16>) -> tensor<1x8x8x4xbf16> {
  %window = tensor.empty() : tensor<2x2xf32>
  // CHECK: linalg.generic
  // CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<1x16x16x4xbf16>, tensor<2x2xf32>)
  // CHECK-SAME: outs(%{{.+}} : tensor<1x8x8x4xf32>)
  // CHECK: ^bb0(%[[IN:.+]]: bf16, %{{.+}}: f32, %[[OUT:.+]]: f32):
  // CHECK: %[[EXT:.+]] = arith.extf %[[IN]] : bf16 to f32
  // CHECK: arith.addf %[[OUT]], %[[EXT]] : f32
  // CHECK: linalg.copy ins(%{{.+}} : tensor<1x8x8x4xf32>)
  %0 = linalg.generic {
      indexing_maps = [#map_rw_in, #map_rw_win, #map_rw_out],
      iterator_types = ["parallel", "parallel", "parallel", "parallel",
                        "reduction", "reduction"]}
      ins(%input, %window : tensor<1x16x16x4xbf16>, tensor<2x2xf32>)
      outs(%init : tensor<1x8x8x4xbf16>) {
    ^bb0(%in: bf16, %w: f32, %out: bf16):
      %add = arith.addf %out, %in : bf16
      linalg.yield %add : bf16
    } -> tensor<1x8x8x4xbf16>
  return %0 : tensor<1x8x8x4xbf16>
}

// CHECK-LABEL: util.func public @bf16_reduce_prod
func.func @bf16_reduce_prod(%input: tensor<64x128xbf16>) -> tensor<64xbf16> {
  %cst = arith.constant 1.000000e+00 : bf16
  %init = tensor.empty() : tensor<64xbf16>
  %fill = linalg.fill ins(%cst : bf16)
      outs(%init : tensor<64xbf16>) -> tensor<64xbf16>

  // CHECK: linalg.reduce
  // CHECK-SAME: outs(%{{.+}} : tensor<64xf32>)
  // CHECK: arith.mulf %{{.+}}, %{{.+}} : f32
  // CHECK: linalg.copy ins(%{{.+}} : tensor<64xf32>)
  %0 = linalg.reduce ins(%input : tensor<64x128xbf16>)
      outs(%fill : tensor<64xbf16>) dimensions = [1]
    (%in: bf16, %out: bf16) {
      %mul = arith.mulf %in, %out : bf16
      linalg.yield %mul : bf16
    }
  return %0 : tensor<64xbf16>
}

// Max-pooling (no arith.addf in body) is left in BF16.
// CHECK-LABEL: util.func public @bf16_pooling_max
func.func @bf16_pooling_max(%input: tensor<1x8x8x4xbf16>,
                            %window: tensor<2x2xbf16>) -> tensor<1x4x4x4xbf16> {
  %cst = arith.constant 0xFF80 : bf16
  %init = tensor.empty() : tensor<1x4x4x4xbf16>
  %fill = linalg.fill ins(%cst : bf16)
      outs(%init : tensor<1x4x4x4xbf16>) -> tensor<1x4x4x4xbf16>

  // CHECK: linalg.pooling_nhwc_max
  // CHECK-SAME: outs(%{{.+}} : tensor<1x4x4x4xbf16>) -> tensor<1x4x4x4xbf16>
  // CHECK-NOT: linalg.copy
  %0 = linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>,
                                strides = dense<2> : vector<2xi64>}
      ins(%input, %window : tensor<1x8x8x4xbf16>, tensor<2x2xbf16>)
      outs(%fill : tensor<1x4x4x4xbf16>) -> tensor<1x4x4x4xbf16>
  return %0 : tensor<1x4x4x4xbf16>
}

// Body captures an outer BF16 value: left in BF16.
// CHECK-LABEL: util.func public @bf16_reduction_captured_scale
func.func @bf16_reduction_captured_scale(%input: tensor<64x128xbf16>,
                                         %init: tensor<64xbf16>,
                                         %scale: bf16) -> tensor<64xbf16> {
  // CHECK: linalg.generic
  // CHECK-SAME: outs(%{{.+}} : tensor<64xbf16>)
  // CHECK-NOT: linalg.copy
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%input : tensor<64x128xbf16>) outs(%init : tensor<64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %mul = arith.mulf %in, %scale : bf16
      %add = arith.addf %out, %mul : bf16
      linalg.yield %add : bf16
    } -> tensor<64xbf16>
  return %0 : tensor<64xbf16>
}

// FP32 contractions are left untouched (keep last to scope CHECK-NOT).
// CHECK-LABEL: util.func public @f32_matmul
func.func @f32_matmul(%a: tensor<128x256xf32>,
                      %b: tensor<256x128xf32>) -> tensor<128x128xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<128x128xf32>
  %fill = linalg.fill ins(%cst : f32)
      outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>

  // CHECK: linalg.matmul
  // CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<128x256xf32>, tensor<256x128xf32>)
  // CHECK-SAME: outs(%{{.+}} : tensor<128x128xf32>) -> tensor<128x128xf32>
  // CHECK-NOT: linalg.copy
  %0 = linalg.matmul ins(%a, %b : tensor<128x256xf32>, tensor<256x128xf32>)
      outs(%fill : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}
