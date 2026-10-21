// RUN: iree-opt --pass-pipeline="builtin.module(func.func(coralnpu-restore-conv-input-channel,iree-codegen-insert-batch-dim-for-batchless-conv))" --split-input-file %s | FileCheck %s

// A convolution whose unit input channel was folded away: the input and
// filter get the unit dimension back and the generic gets an IC reduction
// loop.
// CHECK: #[[$IN:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
// CHECK: #[[$FILTER:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
// CHECK: #[[$OUT:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func.func @conv_2d_nhwc_hwcf_unit_ic
// CHECK: %[[X:.*]] = tensor.expand_shape %arg0 {{\[\[}}0], [1], [2, 3]] output_shape [1, 32, 32, 1]
// CHECK: %[[W:.*]] = tensor.expand_shape %arg1 {{\[\[}}0], [1, 2], [3]] output_shape [3, 3, 1, 8]
// CHECK: linalg.generic {indexing_maps = [#[[$IN]], #[[$FILTER]], #[[$OUT]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%[[X]], %[[W]] : tensor<1x32x32x1xf32>, tensor<3x3x1x8xf32>)
#in = affine_map<(n, oh, ow, oc, kh, kw) -> (n, oh + kh, ow + kw)>
#filter = affine_map<(n, oh, ow, oc, kh, kw) -> (kh, kw, oc)>
#out = affine_map<(n, oh, ow, oc, kh, kw) -> (n, oh, ow, oc)>
func.func @conv_2d_nhwc_hwcf_unit_ic(%x: tensor<1x32x32xf32>, %w: tensor<3x3x8xf32>, %init: tensor<1x30x30x8xf32>) -> tensor<1x30x30x8xf32> {
  %r = linalg.generic {indexing_maps = [#in, #filter, #out], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]}
      ins(%x, %w : tensor<1x32x32xf32>, tensor<3x3x8xf32>) outs(%init : tensor<1x30x30x8xf32>) {
  ^bb0(%a: f32, %b: f32, %acc: f32):
    %m = arith.mulf %a, %b : f32
    %s = arith.addf %acc, %m : f32
    linalg.yield %s : f32
  } -> tensor<1x30x30x8xf32>
  return %r : tensor<1x30x30x8xf32>
}

// -----

// With the batch dim also folded away (as in the first convolution of a
// single-channel model), the input channel is restored first and then the
// batch dim; the two reshapes of the input compose.
// CHECK-LABEL: func.func @batchless_conv_2d_nhwc_hwcf_unit_ic
// CHECK-DAG: %[[W:.*]] = tensor.expand_shape %arg1 {{\[\[}}0], [1, 2], [3]] output_shape [3, 3, 1, 8]
// CHECK-DAG: %[[X:.*]] = tensor.expand_shape %arg0 {{\[\[}}0, 1], [2, 3]] output_shape [1, 32, 32, 1]
// CHECK-DAG: %[[INIT:.*]] = tensor.expand_shape %arg2 {{\[\[}}0, 1], [2], [3]] output_shape [1, 30, 30, 8]
// CHECK: %[[CONV:.*]] = linalg.generic {{.*}} ins(%[[X]], %[[W]] : tensor<1x32x32x1xf32>, tensor<3x3x1x8xf32>) outs(%[[INIT]] : tensor<1x30x30x8xf32>)
// CHECK: tensor.collapse_shape %[[CONV]] {{\[\[}}0, 1], [2], [3]]
#in = affine_map<(oh, ow, oc, kh, kw) -> (oh + kh, ow + kw)>
#filter = affine_map<(oh, ow, oc, kh, kw) -> (kh, kw, oc)>
#out = affine_map<(oh, ow, oc, kh, kw) -> (oh, ow, oc)>
func.func @batchless_conv_2d_nhwc_hwcf_unit_ic(%x: tensor<32x32xf32>, %w: tensor<3x3x8xf32>, %init: tensor<30x30x8xf32>) -> tensor<30x30x8xf32> {
  %r = linalg.generic {indexing_maps = [#in, #filter, #out], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]}
      ins(%x, %w : tensor<32x32xf32>, tensor<3x3x8xf32>) outs(%init : tensor<30x30x8xf32>) {
  ^bb0(%a: f32, %b: f32, %acc: f32):
    %m = arith.mulf %a, %b : f32
    %s = arith.addf %acc, %m : f32
    linalg.yield %s : f32
  } -> tensor<30x30x8xf32>
  return %r : tensor<30x30x8xf32>
}

// -----

// A channel-first convolution (output channel outermost) is left alone.
// CHECK-LABEL: func.func @conv_2d_nchw_fchw_unit_ic
// CHECK: linalg.generic {{.*}} ins(%arg0, %arg1 : tensor<1x32x32xf32>, tensor<8x3x3xf32>)
#in = affine_map<(n, oc, oh, ow, kh, kw) -> (n, oh + kh, ow + kw)>
#filter = affine_map<(n, oc, oh, ow, kh, kw) -> (oc, kh, kw)>
#out = affine_map<(n, oc, oh, ow, kh, kw) -> (n, oc, oh, ow)>
func.func @conv_2d_nchw_fchw_unit_ic(%x: tensor<1x32x32xf32>, %w: tensor<8x3x3xf32>, %init: tensor<1x8x30x30xf32>) -> tensor<1x8x30x30xf32> {
  %r = linalg.generic {indexing_maps = [#in, #filter, #out], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]}
      ins(%x, %w : tensor<1x32x32xf32>, tensor<8x3x3xf32>) outs(%init : tensor<1x8x30x30xf32>) {
  ^bb0(%a: f32, %b: f32, %acc: f32):
    %m = arith.mulf %a, %b : f32
    %s = arith.addf %acc, %m : f32
    linalg.yield %s : f32
  } -> tensor<1x8x30x30xf32>
  return %r : tensor<1x8x30x30xf32>
}
