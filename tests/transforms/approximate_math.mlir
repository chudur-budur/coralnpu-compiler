// RUN: iree-opt --pass-pipeline='builtin.module(func.func(coralnpu-approximate-math))' %s | FileCheck %s

// CHECK-LABEL: func.func @fast_tanh
// CHECK: %[[X:.*]] = arith.extf %arg0 : vector<16xbf16> to vector<16xf32>
// CHECK: %[[HI:.*]] = arith.minnumf %[[X]]
// CHECK: %[[XC:.*]] = arith.maxnumf %[[HI]]
// CHECK: arith.mulf %[[XC]], %[[XC]]
// CHECK: %[[DIV:.*]] = arith.divf %{{.*}}, %{{.*}} : vector<16xf32>
// CHECK: %[[RES:.*]] = arith.truncf %[[DIV]] : vector<16xf32> to vector<16xbf16>
// CHECK: return %[[RES]] : vector<16xbf16>
func.func @fast_tanh(%x: vector<16xbf16>) -> vector<16xbf16> {
  %y = math.tanh %x : vector<16xbf16>
  return %y : vector<16xbf16>
}

// CHECK-LABEL: func.func @fast_exp
// CHECK-DAG: %[[TWO:.*]] = arith.constant dense<2.000000e+00> : vector<16xf32>
// CHECK: %[[HI:.*]] = arith.minnumf %arg0
// CHECK: arith.maxnumf %[[HI]]
// CHECK: %[[SHL:.*]] = arith.shli %{{.*}}, %{{.*}} : vector<16xi32>
// CHECK: %[[POW2:.*]] = arith.bitcast %[[SHL]] : vector<16xi32> to vector<16xf32>
// CHECK: %[[SCALED:.*]] = arith.mulf %{{.*}}, %[[POW2]] : vector<16xf32>
// CHECK: %[[RES:.*]] = arith.mulf %[[SCALED]], %[[TWO]] : vector<16xf32>
// CHECK: return %[[RES]] : vector<16xf32>
func.func @fast_exp(%x: vector<16xf32>) -> vector<16xf32> {
  %y = math.exp %x : vector<16xf32>
  return %y : vector<16xf32>
}

// Scalar operands are handled too; f64 is left to the default lowering.
// CHECK-LABEL: func.func @fast_exp_scalar_and_f64
// CHECK: arith.minnumf %arg0, %{{.*}} : f32
// CHECK: math.exp %arg1 : f64
func.func @fast_exp_scalar_and_f64(%x: f32, %y: f64) -> (f32, f64) {
  %a = math.exp %x : f32
  %b = math.exp %y : f64
  return %a, %b : f32, f64
}

// CHECK-LABEL: func.func @fast_erf
// CHECK: %[[ABS:.*]] = math.absf %arg0 : vector<16xf32>
// CHECK: arith.minnumf %[[ABS]]
// CHECK: %[[INV:.*]] = arith.divf %{{.*}}, %{{.*}} : vector<16xf32>
// CHECK: %[[MAG:.*]] = arith.subf %{{.*}}, %[[INV]] : vector<16xf32>
// CHECK: %[[RES:.*]] = math.copysign %[[MAG]], %arg0 : vector<16xf32>
// CHECK: return %[[RES]] : vector<16xf32>
func.func @fast_erf(%x: vector<16xf32>) -> vector<16xf32> {
  %y = math.erf %x : vector<16xf32>
  return %y : vector<16xf32>
}

// StableHLO's expansion of chlo.erfc(x),
//   select(|x| < 1, 1 - x * T(x^2), select(x < 0, 2 - E, E)),
// with T(0) = 2/sqrt(pi) (here with T(x^2) * x) is replaced with 1 - erf(x),
// while the same selects with T(0) != 2/sqrt(pi) are left untouched.
// CHECK-LABEL: func.func @fast_erfc_expansion
// CHECK: %[[INNER:.*]] = arith.select
// CHECK-NOT: arith.select
// CHECK: %[[ABS:.*]] = math.absf %arg0 : vector<16xf32>
// CHECK: arith.minnumf %[[ABS]]
// CHECK: %[[ERF:.*]] = math.copysign %{{.*}}, %arg0 : vector<16xf32>
// CHECK: %[[RES:.*]] = arith.subf %{{.*}}, %[[ERF]] : vector<16xf32>
// CHECK: %[[KEEP:.*]] = arith.select %{{.*}}, %{{.*}}, %[[INNER]]
// CHECK-NOT: arith.select
// CHECK: return %[[RES]], %[[KEEP]] : vector<16xf32>, vector<16xf32>
func.func @fast_erfc_expansion(%x: vector<16xf32>, %e: vector<16xf32>) -> (vector<16xf32>, vector<16xf32>) {
  %one = arith.constant dense<1.0> : vector<16xf32>
  %two = arith.constant dense<2.0> : vector<16xf32>
  %zero = arith.constant dense<0.0> : vector<16xf32>
  %t0 = arith.constant dense<1.12837911> : vector<16xf32>
  %bad_t0 = arith.constant dense<1.5> : vector<16xf32>
  %abs = math.absf %x : vector<16xf32>
  %x2 = arith.mulf %x, %x : vector<16xf32>
  %neg = arith.cmpf olt, %x, %zero : vector<16xf32>
  %two_sub_e = arith.subf %two, %e : vector<16xf32>
  %erfc_ge1 = arith.select %neg, %two_sub_e, %e : vector<16xi1>, vector<16xf32>
  %t = arith.addf %x2, %t0 : vector<16xf32>
  %erf_lt1 = arith.mulf %t, %x : vector<16xf32>
  %erfc_lt1 = arith.subf %one, %erf_lt1 : vector<16xf32>
  %lt1 = arith.cmpf olt, %abs, %one : vector<16xf32>
  %r = arith.select %lt1, %erfc_lt1, %erfc_ge1 : vector<16xi1>, vector<16xf32>
  %bad_t = arith.addf %x2, %bad_t0 : vector<16xf32>
  %bad_erf = arith.mulf %x, %bad_t : vector<16xf32>
  %bad_lt1 = arith.subf %one, %bad_erf : vector<16xf32>
  %keep = arith.select %lt1, %bad_lt1, %erfc_ge1 : vector<16xi1>, vector<16xf32>
  return %r, %keep : vector<16xf32>, vector<16xf32>
}
