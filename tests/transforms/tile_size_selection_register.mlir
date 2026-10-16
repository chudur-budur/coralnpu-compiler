// RUN: iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(coralnpu-tile-size-selection-register)))))" %s | FileCheck %s

#target = #hal.executable.target<"coralnpu", "coralnpu_elf", {
  cpu_features = "+m,+f,+zvl128b,+zve32f"
}>

// Softmax whose max reduction reads a product that nothing else reads. Root
// fusion does not reach that product, so it gets its own config; exp(x - max),
// a producer of the root sum only (the division recomputes it, as in the
// softmax IREE decomposes), is fused and gets none.
// CHECK-DAG: #[[INNER:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 0], vector_inner_parallel = [0, 16]>
// CHECK-DAG: #[[RED:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 0], vector_reduction = [0, 16]>
// CHECK: func.func @softmax_scaled_max
// CHECK: %[[MUL:.+]] = linalg.generic {{.*}} attrs = {lowering_config = #[[INNER]]}
// CHECK: %[[MAX:.+]] = linalg.generic {{.*}} ins(%[[MUL]] : {{.*}} attrs = {lowering_config = #[[RED]]}
// CHECK: %[[EXP:.+]] = linalg.generic {{.*}} ins(%arg0, %[[MAX]] : {{.*}}) outs(%{{.+}} : tensor<16x256xf32>) {
// CHECK: linalg.generic {{.*}} ins(%[[EXP]] : {{.*}} attrs = {compilation_info =
// CHECK: linalg.generic {{.*}} ins(%arg0, %[[MAX]], %{{.+}} : {{.*}} attrs = {lowering_config = #[[INNER]]}
hal.executable private @softmax_scaled_max {
  hal.executable.variant public @coralnpu_elf target(#target) {
    builtin.module {
      func.func @softmax_scaled_max(%x: tensor<16x256xf32>, %w: tensor<16x256xf32>) -> tensor<16x256xf32> {
        %cmin = arith.constant -3.40282347E+38 : f32
        %c0 = arith.constant 0.0 : f32
        %e2 = tensor.empty() : tensor<16x256xf32>
        %e1 = tensor.empty() : tensor<16xf32>
        %s = linalg.generic {indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>], iterator_types = ["parallel", "parallel"]}
             ins(%x, %w : tensor<16x256xf32>, tensor<16x256xf32>) outs(%e2 : tensor<16x256xf32>) {
        ^bb0(%a: f32, %b: f32, %o: f32):
          %m = arith.mulf %a, %b : f32
          linalg.yield %m : f32
        } -> tensor<16x256xf32>
        %fmax = linalg.fill ins(%cmin : f32) outs(%e1 : tensor<16xf32>) -> tensor<16xf32>
        %max = linalg.generic {indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i)>], iterator_types = ["parallel", "reduction"]}
             ins(%s : tensor<16x256xf32>) outs(%fmax : tensor<16xf32>) {
        ^bb0(%a: f32, %o: f32):
          %m = arith.maximumf %a, %o : f32
          linalg.yield %m : f32
        } -> tensor<16xf32>
        %ex = linalg.generic {indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i)>, affine_map<(i, j) -> (i, j)>], iterator_types = ["parallel", "parallel"]}
             ins(%x, %max : tensor<16x256xf32>, tensor<16xf32>) outs(%e2 : tensor<16x256xf32>) {
        ^bb0(%a: f32, %b: f32, %o: f32):
          %d = arith.subf %a, %b : f32
          %x2 = math.exp %d : f32
          linalg.yield %x2 : f32
        } -> tensor<16x256xf32>
        %fsum = linalg.fill ins(%c0 : f32) outs(%e1 : tensor<16xf32>) -> tensor<16xf32>
        %sum = linalg.generic {indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i)>], iterator_types = ["parallel", "reduction"]}
             ins(%ex : tensor<16x256xf32>) outs(%fsum : tensor<16xf32>) {
        ^bb0(%a: f32, %o: f32):
          %r = arith.addf %a, %o : f32
          linalg.yield %r : f32
        } -> tensor<16xf32>
        %div = linalg.generic {indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i)>, affine_map<(i, j) -> (i)>, affine_map<(i, j) -> (i, j)>], iterator_types = ["parallel", "parallel"]}
             ins(%x, %max, %sum : tensor<16x256xf32>, tensor<16xf32>, tensor<16xf32>) outs(%e2 : tensor<16x256xf32>) {
        ^bb0(%a: f32, %m: f32, %b: f32, %o: f32):
          %d = arith.subf %a, %m : f32
          %x2 = math.exp %d : f32
          %r = arith.divf %x2, %b : f32
          linalg.yield %r : f32
        } -> tensor<16x256xf32>
        return %div : tensor<16x256xf32>
      }
    }
  }
}
