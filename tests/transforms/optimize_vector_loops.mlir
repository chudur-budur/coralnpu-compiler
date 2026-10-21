// RUN: iree-opt --pass-pipeline="builtin.module(func.func(coralnpu-optimize-vector-loops))" %s | FileCheck %s
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(coralnpu-optimize-vector-loops{fast-math=false}))" %s | FileCheck %s --check-prefix=NOFAST

// A K-tiled dot product: the per-iteration horizontal add becomes a vector
// accumulator and a single reduction after the loop. The reduction's
// fast-math flags carry over. (39 iterations, so the loop is not unrolled.)
// CHECK-LABEL: func.func @dot_loop
// CHECK: %[[INIT:.*]] = vector.insert %{{.*}}, %{{.*}} [0] : f32 into vector<16xf32>
// CHECK: %[[LOOP:.*]] = scf.for %{{.*}} iter_args(%[[ACC:.*]] = %[[INIT]]) -> (vector<16xf32>) {
// CHECK:   %[[MUL:.*]] = arith.mulf
// CHECK-NOT: vector.reduction
// CHECK:   %[[SUM:.*]] = arith.addf %[[ACC]], %[[MUL]] fastmath<reassoc> : vector<16xf32>
// CHECK:   scf.yield %[[SUM]] : vector<16xf32>
// CHECK: %[[RED:.*]] = vector.reduction <add>, %[[LOOP]] fastmath<reassoc> : vector<16xf32> into f32
// CHECK: vector.broadcast %[[RED]] : f32 to vector<1xf32>
func.func @dot_loop(%a: memref<624xf32>, %b: memref<624xf32>, %init: vector<1xf32>) -> vector<1xf32> {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c624 = arith.constant 624 : index
  %r = scf.for %k = %c0 to %c624 step %c16 iter_args(%acc = %init) -> (vector<1xf32>) {
    %va = vector.load %a[%k] : memref<624xf32>, vector<16xf32>
    %vb = vector.load %b[%k] : memref<624xf32>, vector<16xf32>
    %s = vector.extract %acc[0] : f32 from vector<1xf32>
    %m = arith.mulf %va, %vb : vector<16xf32>
    %red = vector.reduction <add>, %m, %s fastmath<reassoc> : vector<16xf32> into f32
    %y = vector.broadcast %red : f32 to vector<1xf32>
    scf.yield %y : vector<1xf32>
  }
  return %r : vector<1xf32>
}

// A scalar accumulator used elsewhere in the body is left alone.
// CHECK-LABEL: func.func @acc_used_in_body
// CHECK: scf.for {{.*}} -> (f32)
func.func @acc_used_in_body(%a: memref<640xf32>, %init: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c640 = arith.constant 640 : index
  %r = scf.for %k = %c0 to %c640 step %c16 iter_args(%acc = %init) -> (f32) {
    %va = vector.load %a[%k] : memref<640xf32>, vector<16xf32>
    %red = vector.reduction <add>, %va, %acc fastmath<reassoc> : vector<16xf32> into f32
    memref.store %acc, %a[%k] : memref<640xf32>
    scf.yield %red : f32
  }
  return %r : f32
}

// Without the reassoc flag, a float sum of products of bf16 values is
// hoisted only with fast math (the default), as hoisting reorders the sum, and
// then gets the flag; a same-width f32 sum keeps its order, and an integer sum
// is always hoisted.
// NOFAST-LABEL: func.func @widened_and_same_width
// NOFAST: scf.for {{.*}} -> (f32)
// NOFAST:   vector.reduction <add>
// CHECK-LABEL: func.func @widened_and_same_width
// CHECK: scf.for {{.*}} -> (vector<16xf32>)
// CHECK-NOT: vector.reduction
// CHECK:   arith.addf {{.*}} fastmath<reassoc>
// CHECK: vector.reduction <add>, {{.*}} fastmath<reassoc>
// CHECK: scf.for {{.*}} -> (f32)
// CHECK:   vector.reduction <add>
// CHECK: scf.for {{.*}} -> (vector<16xi32>)
// CHECK-NOT: vector.reduction
// CHECK:   arith.addi
func.func @widened_and_same_width(%a: memref<640xbf16>, %f: memref<640xf32>, %b: memref<640xi32>, %init: f32, %iinit: i32) -> (f32, f32, i32) {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c640 = arith.constant 640 : index
  %rw = scf.for %k = %c0 to %c640 step %c16 iter_args(%acc = %init) -> (f32) {
    %va0 = vector.load %a[%k] : memref<640xbf16>, vector<16xbf16>
    %va = arith.extf %va0 : vector<16xbf16> to vector<16xf32>
    %m = arith.mulf %va, %va : vector<16xf32>
    %red = vector.reduction <add>, %m, %acc : vector<16xf32> into f32
    scf.yield %red : f32
  }
  %rs = scf.for %k = %c0 to %c640 step %c16 iter_args(%acc = %init) -> (f32) {
    %vf = vector.load %f[%k] : memref<640xf32>, vector<16xf32>
    %m = arith.mulf %vf, %vf : vector<16xf32>
    %red = vector.reduction <add>, %m, %acc : vector<16xf32> into f32
    scf.yield %red : f32
  }
  %ri = scf.for %k = %c0 to %c640 step %c16 iter_args(%acc = %iinit) -> (i32) {
    %vb = vector.load %b[%k] : memref<640xi32>, vector<16xi32>
    %red = vector.reduction <add>, %vb, %acc : vector<16xi32> into i32
    scf.yield %red : i32
  }
  return %rw, %rs, %ri : f32, f32, i32
}

// The row loop around a hoisted dot loop is unroll-and-jammed by 2, so the K
// loop carries two accumulators; K (40 iters) unrolls by 4.
// CHECK-LABEL: func.func @matvec_rows
// CHECK: scf.for %{{.*}} = %c0 to %c8 step %c2{{[_0-9]*}} {
// CHECK:   %[[LOOP:.*]]:2 = scf.for {{.*}} to %c640 step %c64 iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (vector<16xf32>, vector<16xf32>) {
// CHECK-COUNT-8: arith.addf
// CHECK-NOT: vector.reduction
// CHECK:     scf.yield
// CHECK-COUNT-2: vector.reduction <add>, %[[LOOP]]#
// CHECK-NOT: scf.for
func.func @matvec_rows(%w: memref<8x640xbf16>, %x: memref<640xbf16>, %y: memref<8xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c640 = arith.constant 640 : index
  %zero = arith.constant 0.0 : f32
  scf.for %i = %c0 to %c8 step %c1 {
    %r = scf.for %k = %c0 to %c640 step %c16 iter_args(%acc = %zero) -> (f32) {
      %vw0 = vector.load %w[%i, %k] : memref<8x640xbf16>, vector<16xbf16>
      %vx0 = vector.load %x[%k] : memref<640xbf16>, vector<16xbf16>
      %vw = arith.extf %vw0 : vector<16xbf16> to vector<16xf32>
      %vx = arith.extf %vx0 : vector<16xbf16> to vector<16xf32>
      %m = arith.mulf %vw, %vx : vector<16xf32>
      %red = vector.reduction <add>, %m, %acc fastmath<reassoc> : vector<16xf32> into f32
      scf.yield %red : f32
    }
    memref.store %r, %y[%i] : memref<8xf32>
  }
  return
}

// A row loop whose body also writes a buffer it reads, here through two
// subspans of its binding, is not jammed.
// CHECK-LABEL: func.func @rows_read_write_same_buffer
// CHECK: scf.for %{{.*}} = %c0 to %c8 step %c1 {
// CHECK:   scf.for {{.*}} -> (vector<16xf32>)
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
func.func @rows_read_write_same_buffer(%w: memref<8x640xf32>, %x: memref<640xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c640 = arith.constant 640 : index
  %zero = arith.constant 0.0 : f32
  %yr = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) offset(%c0) : memref<8xf32>
  %yw = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) offset(%c0) : memref<8xf32>
  scf.for %i = %c0 to %c8 step %c1 {
    %r = scf.for %k = %c0 to %c640 step %c16 iter_args(%acc = %zero) -> (f32) {
      %vw = vector.load %w[%i, %k] : memref<8x640xf32>, vector<16xf32>
      %vx = vector.load %x[%k] : memref<640xf32>, vector<16xf32>
      %m = arith.mulf %vw, %vx : vector<16xf32>
      %red = vector.reduction <add>, %m, %acc fastmath<reassoc> : vector<16xf32> into f32
      scf.yield %red : f32
    }
    %prev = memref.load %yr[%i] : memref<8xf32>
    %s = arith.addf %prev, %r : f32
    memref.store %s, %yw[%i] : memref<8xf32>
  }
  return
}

// A row loop that writes before its dot loop is not jammed: that would move
// row i + 1's clearing of %y[i + 1] ahead of row i's store of its result
// there.
// CHECK-LABEL: func.func @rows_write_before_dot_loop
// CHECK: scf.for %{{.*}} = %c0 to %c8 step %c1 {
// CHECK:   scf.for {{.*}} -> (vector<16xf32>)
func.func @rows_write_before_dot_loop(%w: memref<8x640xf32>, %x: memref<640xf32>, %y: memref<9xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c640 = arith.constant 640 : index
  %zero = arith.constant 0.0 : f32
  scf.for %i = %c0 to %c8 step %c1 {
    memref.store %zero, %y[%i] : memref<9xf32>
    %r = scf.for %k = %c0 to %c640 step %c16 iter_args(%acc = %zero) -> (f32) {
      %vw = vector.load %w[%i, %k] : memref<8x640xf32>, vector<16xf32>
      %vx = vector.load %x[%k] : memref<640xf32>, vector<16xf32>
      %m = arith.mulf %vw, %vx : vector<16xf32>
      %red = vector.reduction <add>, %m, %acc fastmath<reassoc> : vector<16xf32> into f32
      scf.yield %red : f32
    }
    %i1 = arith.addi %i, %c1 : index
    memref.store %r, %y[%i1] : memref<9xf32>
  }
  return
}

// Nested jam candidates: the inner row loop is jammed (by 2; the 1-element
// %xb load lowers to a scalar load and is not counted), and the outer loop,
// whose nested loops write, is left as is.
// CHECK-LABEL: func.func @nested_rows
// CHECK: scf.for %{{.*}} = %c0 to %c4 step %c1 {
// CHECK:   scf.for {{.*}} -> (vector<16xf32>)
// CHECK:   scf.for %{{.*}} = %c0 to %c8 step %c2{{[_0-9]*}} {
// CHECK:     scf.for {{.*}} -> (vector<32xf32>, vector<32xf32>)
func.func @nested_rows(%w: memref<4x8x640xbf16>, %x: memref<640xf32>, %xb: memref<640xbf16>, %y: memref<4x8xf32>, %z: memref<4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c640 = arith.constant 640 : index
  %zero = arith.constant 0.0 : f32
  scf.for %b = %c0 to %c4 step %c1 {
    %d = scf.for %k = %c0 to %c640 step %c16 iter_args(%acc = %zero) -> (f32) {
      %vx = vector.load %x[%k] : memref<640xf32>, vector<16xf32>
      %red = vector.reduction <add>, %vx, %acc fastmath<reassoc> : vector<16xf32> into f32
      scf.yield %red : f32
    }
    memref.store %d, %z[%b] : memref<4xf32>
    scf.for %i = %c0 to %c8 step %c1 {
      %r = scf.for %k = %c0 to %c640 step %c32 iter_args(%acc = %zero) -> (f32) {
        %vw0 = vector.load %w[%b, %i, %k] : memref<4x8x640xbf16>, vector<32xbf16>
        %vx0 = vector.load %xb[%k] : memref<640xbf16>, vector<1xbf16>
        %vxb = vector.broadcast %vx0 : vector<1xbf16> to vector<32xbf16>
        %vw = arith.extf %vw0 : vector<32xbf16> to vector<32xf32>
        %vx = arith.extf %vxb : vector<32xbf16> to vector<32xf32>
        %m = arith.mulf %vw, %vx : vector<32xf32>
        %red = vector.reduction <add>, %m, %acc fastmath<reassoc> : vector<32xf32> into f32
        scf.yield %red : f32
      }
      memref.store %r, %y[%b, %i] : memref<4x8xf32>
    }
  }
  return
}

// 3-channel conv (kh -> kw -> ic = 3) with one vector<32xf32> accumulator: the
// ic loop unrolls into kw (3 FMAs per kw step); kh and kw stay rolled.
// CHECK-LABEL: func.func @stem_conv_3x3x3_filter_loops
// CHECK: scf.for %{{.*}} = %c0 to %c3 step %c1 iter_args(%[[A:.*]] = %{{.*}}) -> (vector<32xf32>) {
// CHECK: scf.for %{{.*}} = %c0 to %c3 step %c1 iter_args(%{{.*}} = %[[A]]) -> (vector<32xf32>) {
// CHECK-NOT: scf.for
// CHECK-COUNT-3: vector.fma
// CHECK-NOT: vector.fma
// CHECK:   scf.yield
func.func @stem_conv_3x3x3_filter_loops(%x: memref<3x3x3xf32>, %w: memref<3x3x3x32xf32>, %init: vector<32xf32>) -> vector<32xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %r = scf.for %kh = %c0 to %c3 step %c1 iter_args(%a_kh = %init) -> (vector<32xf32>) {
    %r_kw = scf.for %kw = %c0 to %c3 step %c1 iter_args(%a_kw = %a_kh) -> (vector<32xf32>) {
      %r_ic = scf.for %ic = %c0 to %c3 step %c1 iter_args(%a_ic = %a_kw) -> (vector<32xf32>) {
        %sx = memref.load %x[%kh, %kw, %ic] : memref<3x3x3xf32>
        %bx = vector.broadcast %sx : f32 to vector<32xf32>
        %vw = vector.load %w[%kh, %kw, %ic, %c0] : memref<3x3x3x32xf32>, vector<32xf32>
        %fma = vector.fma %vw, %bx, %a_ic : vector<32xf32>
        scf.yield %fma : vector<32xf32>
      }
      scf.yield %r_ic : vector<32xf32>
    }
    scf.yield %r_kw : vector<32xf32>
  }
  return %r : vector<32xf32>
}

// A 2-D nest whose inner body computes a vector from %b[%j] alone: interchange
// makes %j the outer loop and hoists that vector DAG out of the %i loop.
// CHECK-LABEL: func.func @broadcast_nest
// CHECK: scf.for %[[J:.*]] = %c0 to %c64 step %c16 {
// CHECK:   %[[B:.*]] = vector.load %{{.*}}[%[[J]]]
// CHECK:   %[[E:.*]] = arith.mulf %[[B]], %[[B]]
// CHECK:   scf.for %[[I:.*]] = %c0 to %c8 step %c1 {
// CHECK:     %[[A:.*]] = vector.load %{{.*}}[%[[I]], %[[J]]]
// CHECK:     %[[R:.*]] = arith.mulf %[[A]], %[[E]]
// CHECK:     vector.store %[[R]], %{{.*}}[%[[I]], %[[J]]]
func.func @broadcast_nest(%a: memref<8x64xf32>, %b: memref<64xf32>, %c: memref<8x64xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c64 = arith.constant 64 : index
  scf.for %i = %c0 to %c8 step %c1 {
    scf.for %j = %c0 to %c64 step %c16 {
      %vb = vector.load %b[%j] : memref<64xf32>, vector<16xf32>
      %e = arith.mulf %vb, %vb : vector<16xf32>
      %va = vector.load %a[%i, %j] : memref<8x64xf32>, vector<16xf32>
      %r = arith.mulf %va, %e : vector<16xf32>
      vector.store %r, %c[%i, %j] : memref<8x64xf32>, vector<16xf32>
    }
  }
  return
}

// The store index mixes both IVs, so interchange could change which iteration
// writes a location last: the nest is left as is.
// CHECK-LABEL: func.func @broadcast_nest_mixed_index
// CHECK: scf.for %{{.*}} = %c0 to %c8 step %c1 {
// CHECK:   scf.for %{{.*}} = %c0 to %c64 step %c16 {
func.func @broadcast_nest_mixed_index(%a: memref<8x64xf32>, %b: memref<64xf32>, %c: memref<128xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c64 = arith.constant 64 : index
  scf.for %i = %c0 to %c8 step %c1 {
    scf.for %j = %c0 to %c64 step %c16 {
      %vb = vector.load %b[%j] : memref<64xf32>, vector<16xf32>
      %e = arith.mulf %vb, %vb : vector<16xf32>
      %va = vector.load %a[%i, %j] : memref<8x64xf32>, vector<16xf32>
      %r = arith.mulf %va, %e : vector<16xf32>
      %k = arith.addi %i, %j : index
      vector.store %r, %c[%k] : memref<128xf32>, vector<16xf32>
    }
  }
  return
}

// A second store that indexes by the IVs the other way around would change
// which iteration writes the elements off the diagonal last: the nest is left
// as is.
// CHECK-LABEL: func.func @broadcast_nest_two_stores
// CHECK: scf.for %{{.*}} = %c0 to %c8 step %c1 {
// CHECK:   scf.for %{{.*}} = %c0 to %c4 step %c1 {
func.func @broadcast_nest_two_stores(%a: memref<8x8xf32>, %b: memref<8xf32>, %c: memref<8x8xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    scf.for %j = %c0 to %c4 step %c1 {
      %vb = vector.load %b[%j] : memref<8xf32>, vector<1xf32>
      %e = arith.mulf %vb, %vb : vector<1xf32>
      %va = vector.load %a[%i, %j] : memref<8x8xf32>, vector<1xf32>
      %r = arith.mulf %va, %e : vector<1xf32>
      vector.store %r, %c[%i, %j] : memref<8x8xf32>, vector<1xf32>
      vector.store %e, %c[%j, %i] : memref<8x8xf32>, vector<1xf32>
    }
  }
  return
}
