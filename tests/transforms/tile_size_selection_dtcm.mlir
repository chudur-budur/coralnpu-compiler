// RUN: iree-compile --mlir-disable-threading --iree-hal-target-device=coralnpu \
// RUN:   --compile-to=executable-configurations %s | FileCheck %s

// A 3x3 convolution cannot peel N, so N = 24 takes two 12-column Zvt tiles
// rather than a 16-column one and an 8-column remainder.
// CHECK-LABEL: func.func @conv_3x3_n24
// CHECK: linalg.generic
// CHECK-SAME: vector_common_parallel = [0, 1, 0, 12, 0, 0, 0]
func.func @conv_3x3_n24(%in: tensor<1x34x34x16xi8>, %w: tensor<3x3x16x24xi8>) -> tensor<1x32x32x24xi32> {
  %c0 = arith.constant 0 : i32
  %e = tensor.empty() : tensor<1x32x32x24xi32>
  %f = linalg.fill ins(%c0 : i32) outs(%e : tensor<1x32x32x24xi32>) -> tensor<1x32x32x24xi32>
  %r = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
       ins(%in, %w : tensor<1x34x34x16xi8>, tensor<3x3x16x24xi8>) outs(%f : tensor<1x32x32x24xi32>) -> tensor<1x32x32x24xi32>
  return %r : tensor<1x32x32x24xi32>
}

// N = 20 splits into neither 12- nor 8-column tiles, so the convolution keeps
// the RVV sizes.
// CHECK-LABEL: func.func @conv_3x3_n20
// CHECK: linalg.generic
// CHECK-SAME: vector_common_parallel = [0, 1, 4, 0, 0, 0, 0]
func.func @conv_3x3_n20(%in: tensor<1x34x34x16xi8>, %w: tensor<3x3x16x20xi8>) -> tensor<1x32x32x20xi32> {
  %c0 = arith.constant 0 : i32
  %e = tensor.empty() : tensor<1x32x32x20xi32>
  %f = linalg.fill ins(%c0 : i32) outs(%e : tensor<1x32x32x20xi32>) -> tensor<1x32x32x20xi32>
  %r = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
       ins(%in, %w : tensor<1x34x34x16xi8>, tensor<3x3x16x20xi8>) outs(%f : tensor<1x32x32x20xi32>) -> tensor<1x32x32x20xi32>
  return %r : tensor<1x32x32x20xi32>
}

// Softmax: the max reduction (non-root), the sum of exp(x - max) (root) and
// the division (a wider consumer) each get a config, with the shared
// reduction tile.
// CHECK-LABEL: func.func @softmax
// CHECK: linalg.generic
// CHECK-SAME: lowering_config = #iree_cpu.lowering_config<vector_common_parallel = [1, 0], vector_reduction = [0, 16]>
// CHECK: linalg.generic
// CHECK-SAME: distribution = [8, 0], vector_common_parallel = [1, 0], vector_reduction = [0, 16]
// CHECK: linalg.generic
// CHECK-SAME: lowering_config = #iree_cpu.lowering_config<vector_common_parallel = [1, 0], vector_inner_parallel = [0, 16]>
func.func @softmax(%x: tensor<16x256xf32>) -> tensor<16x256xf32> {
  %e = tensor.empty() : tensor<16x256xf32>
  %r = linalg.softmax dimension(1) ins(%x : tensor<16x256xf32>) outs(%e : tensor<16x256xf32>) -> tensor<16x256xf32>
  return %r : tensor<16x256xf32>
}

// 1x4 and 4x1 Zvt strips: N = 128 takes 64-column tiles, M = 128 64-row ones.
// CHECK-LABEL: func.func @matmul_16x128
// CHECK: linalg.matmul
// CHECK-SAME: vector_common_parallel = [0, 64, 0]
func.func @matmul_16x128(%a: tensor<16x128xf32>, %b: tensor<128x128xf32>) -> tensor<16x128xf32> {
  %c0 = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<16x128xf32>
  %f = linalg.fill ins(%c0 : f32) outs(%e : tensor<16x128xf32>) -> tensor<16x128xf32>
  %r = linalg.matmul ins(%a, %b : tensor<16x128xf32>, tensor<128x128xf32>) outs(%f : tensor<16x128xf32>) -> tensor<16x128xf32>
  return %r : tensor<16x128xf32>
}

// CHECK-LABEL: func.func @matmul_128x16
// CHECK: linalg.matmul
// CHECK-SAME: vector_common_parallel = [64, 0, 0]
func.func @matmul_128x16(%a: tensor<128x128xf32>, %b: tensor<128x16xf32>) -> tensor<128x16xf32> {
  %c0 = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<128x16xf32>
  %f = linalg.fill ins(%c0 : f32) outs(%e : tensor<128x16xf32>) -> tensor<128x16xf32>
  %r = linalg.matmul ins(%a, %b : tensor<128x128xf32>, tensor<128x16xf32>) outs(%f : tensor<128x16xf32>) -> tensor<128x16xf32>
  return %r : tensor<128x16xf32>
}

// With a partial N, M = 196 needs whole M tiles: 14 rows (196 = 14 x 14).
// CHECK-LABEL: func.func @matmul_m196_n8
// CHECK: linalg.matmul
// CHECK-SAME: vector_common_parallel = [14, 0, 0]
func.func @matmul_m196_n8(%a: tensor<196x64xi8>, %b: tensor<64x8xi8>) -> tensor<196x8xi32> {
  %c0 = arith.constant 0 : i32
  %e = tensor.empty() : tensor<196x8xi32>
  %f = linalg.fill ins(%c0 : i32) outs(%e : tensor<196x8xi32>) -> tensor<196x8xi32>
  %r = linalg.matmul ins(%a, %b : tensor<196x64xi8>, tensor<64x8xi8>) outs(%f : tensor<196x8xi32>) -> tensor<196x8xi32>
  return %r : tensor<196x8xi32>
}

// N = 1040 takes 64-column tiles and a peeled 16-column remainder. The 16 x N
// accumulator does not fit the DTCM, so N shrinks by whole 64-column tiles
// (256) rather than to an exact divisor (260) with a 4-column remainder.
// CHECK-LABEL: func.func @matmul_n1040
// CHECK-SAME: enable_loop_peeling
// CHECK: linalg.matmul
// CHECK-SAME: distribution = [0, 256, 0]
// CHECK-SAME: vector_common_parallel = [0, 64, 0]
func.func @matmul_n1040(%a: tensor<16x64xi8>, %b: tensor<64x1040xi8>) -> tensor<16x1040xi32> {
  %c0 = arith.constant 0 : i32
  %e = tensor.empty() : tensor<16x1040xi32>
  %f = linalg.fill ins(%c0 : i32) outs(%e : tensor<16x1040xi32>) -> tensor<16x1040xi32>
  %r = linalg.matmul ins(%a, %b : tensor<16x64xi8>, tensor<64x1040xi8>) outs(%f : tensor<16x1040xi32>) -> tensor<16x1040xi32>
  return %r : tensor<16x1040xi32>
}

// N = 36 takes 12-column tiles: 16-column ones would leave a 4-column
// remainder that the Zvt codegen cannot lower.
// CHECK-LABEL: func.func @matmul_n36
// CHECK: linalg.matmul
// CHECK-SAME: vector_common_parallel = [0, 12, 0]
func.func @matmul_n36(%a: tensor<32x64xi8>, %b: tensor<64x36xi8>) -> tensor<32x36xi32> {
  %c0 = arith.constant 0 : i32
  %e = tensor.empty() : tensor<32x36xi32>
  %f = linalg.fill ins(%c0 : i32) outs(%e : tensor<32x36xi32>) -> tensor<32x36xi32>
  %r = linalg.matmul ins(%a, %b : tensor<32x64xi8>, tensor<64x36xi8>) outs(%f : tensor<32x36xi32>) -> tensor<32x36xi32>
  return %r : tensor<32x36xi32>
}

// Swapped inputs (the first is K x N) with a partial N have no Zvt lowering,
// so the contraction keeps the RVV sizes.
// CHECK-LABEL: func.func @matmul_swapped_n12
// CHECK: linalg.generic
// CHECK-SAME: vector_common_parallel = [4, 4, 0]
func.func @matmul_swapped_n12(%a: tensor<64x12xi8>, %b: tensor<32x64xi8>) -> tensor<32x12xi32> {
  %c0 = arith.constant 0 : i32
  %e = tensor.empty() : tensor<32x12xi32>
  %f = linalg.fill ins(%c0 : i32) outs(%e : tensor<32x12xi32>) -> tensor<32x12xi32>
  %r = linalg.generic {indexing_maps = [affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (m, n)>],
                       iterator_types = ["parallel", "parallel", "reduction"]}
       ins(%a, %b : tensor<64x12xi8>, tensor<32x64xi8>) outs(%f : tensor<32x12xi32>) {
  ^bb0(%x: i8, %y: i8, %acc: i32):
    %xe = arith.extsi %x : i8 to i32
    %ye = arith.extsi %y : i8 to i32
    %p = arith.muli %xe, %ye : i32
    %s = arith.addi %acc, %p : i32
    linalg.yield %s : i32
  } -> tensor<32x12xi32>
  return %r : tensor<32x12xi32>
}

// K = 8192 does not fit the DTCM in full, but K is streamed rather than split.
// CHECK-LABEL: func.func @matmul_k8192
// CHECK: linalg.matmul
// CHECK-SAME: cache_reduction = [0, 0, 0]
// CHECK-SAME: vector_common_parallel = [0, 32, 0], vector_reduction = [0, 0, 1]
func.func @matmul_k8192(%a: tensor<32x8192xi8>, %b: tensor<8192x64xi8>) -> tensor<32x64xi32> {
  %c0 = arith.constant 0 : i32
  %e = tensor.empty() : tensor<32x64xi32>
  %f = linalg.fill ins(%c0 : i32) outs(%e : tensor<32x64xi32>) -> tensor<32x64xi32>
  %r = linalg.matmul ins(%a, %b : tensor<32x8192xi8>, tensor<8192x64xi8>) outs(%f : tensor<32x64xi32>) -> tensor<32x64xi32>
  return %r : tensor<32x64xi32>
}

// An elementwise body with few live values takes LMUL=8 tiles (8 x 4 f32);
// one with more takes LMUL=4.
// CHECK-LABEL: func.func @add_lmul8
// CHECK: linalg.generic
// CHECK-SAME: vector_common_parallel = [32]
func.func @add_lmul8(%a: tensor<16x1024xf32>, %b: tensor<16x1024xf32>) -> tensor<16x1024xf32> {
  %r = arith.addf %a, %b : tensor<16x1024xf32>
  return %r : tensor<16x1024xf32>
}

// CHECK-LABEL: func.func @heavy_body_lmul4
// CHECK: linalg.generic
// CHECK-SAME: vector_common_parallel = [16]
func.func @heavy_body_lmul4(%a: tensor<16x1024xf32>, %b: tensor<16x1024xf32>, %c: tensor<16x1024xf32>) -> tensor<16x1024xf32> {
  %e = tensor.empty() : tensor<16x1024xf32>
  %r = linalg.generic {indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>],
                       iterator_types = ["parallel", "parallel"]}
       ins(%a, %b, %c : tensor<16x1024xf32>, tensor<16x1024xf32>, tensor<16x1024xf32>) outs(%e : tensor<16x1024xf32>) {
  ^bb0(%x: f32, %y: f32, %z: f32, %o: f32):
    %0 = arith.mulf %x, %y : f32
    %1 = arith.mulf %x, %z : f32
    %2 = arith.addf %0, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<16x1024xf32>
  return %r : tensor<16x1024xf32>
}

// A transpose that moves the innermost dim takes 8x8 tiles.
// CHECK-LABEL: func.func @transpose
// CHECK: linalg.generic
// CHECK-SAME: vector_common_parallel = [8, 8]
func.func @transpose(%a: tensor<64x128xf32>) -> tensor<128x64xf32> {
  %e = tensor.empty() : tensor<128x64xf32>
  %r = linalg.transpose ins(%a : tensor<64x128xf32>) outs(%e : tensor<128x64xf32>) permutation = [1, 0]
  return %r : tensor<128x64xf32>
}

// A row reduction is vectorized along the reduction dim.
// CHECK-LABEL: func.func @row_sum
// CHECK: linalg.generic
// CHECK-SAME: vector_common_parallel = [1, 0], vector_reduction = [0, 16]
func.func @row_sum(%a: tensor<64x512xf32>) -> tensor<64xf32> {
  %c0 = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<64xf32>
  %f = linalg.fill ins(%c0 : f32) outs(%e : tensor<64xf32>) -> tensor<64xf32>
  %r = linalg.reduce ins(%a : tensor<64x512xf32>) outs(%f : tensor<64xf32>) dimensions = [1]
    (%x: f32, %acc: f32) {
      %s = arith.addf %x, %acc : f32
      linalg.yield %s : f32
    }
  return %r : tensor<64xf32>
}

// With K innermost in both inputs (N x K right-hand side) and no Zvt shape,
// the matmul vectorizes K.
// CHECK-LABEL: func.func @matmul_k_inner
// CHECK: linalg.matmul
// CHECK-SAME: vector_reduction = [0, 0, 4]
func.func @matmul_k_inner(%a: tensor<8x256xf32>, %b: tensor<4x256xf32>) -> tensor<8x4xf32> {
  %c0 = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<8x4xf32>
  %f = linalg.fill ins(%c0 : f32) outs(%e : tensor<8x4xf32>) -> tensor<8x4xf32>
  %r = linalg.matmul indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, k) -> (m, n)>]
       ins(%a, %b : tensor<8x256xf32>, tensor<4x256xf32>) outs(%f : tensor<8x4xf32>) -> tensor<8x4xf32>
  return %r : tensor<8x4xf32>
}
