// RUN: not %iree_compile --coralnpu-dtcm-size-kb=0 %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERR-ZERO
// RUN: not %iree_compile --coralnpu-roofline-speedup-threshold=-1 %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERR-THRESHOLD
// RUN: not %iree_compile --coralnpu-roofline-speedup-threshold=0 --coralnpu-linker-script-path=/nonexistent/path.ld %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERR-LINK
// RUN: %iree_compile --coralnpu-dtcm-size-kb=1024 --coralnpu-roofline-speedup-threshold=0 --coralnpu-dump-affinity-profile-format=pretty --mlir-print-ir-after=iree-codegen-math-transform %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: %iree_compile --coralnpu-roofline-speedup-threshold=0 --coralnpu-fast-math=false --mlir-print-ir-after=iree-codegen-math-transform %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-NO-FAST-MATH

// CHECK-ERR-ZERO: coralnpu-dtcm-size-kb must be positive, got 0

// CHECK-ERR-THRESHOLD: coralnpu-roofline-speedup-threshold must be a finite, non-negative value, got -1

// CHECK-ERR-LINK: failed to serialize executables

// CHECK-SUCCESS: Execution Profile by Affinity:
// CHECK-SUCCESS:   Affinity: coralnpu
// CHECK-SUCCESS:     Dispatches: 1
// 214877.484 is the constant term of CoralNPUApproximateMath's tanh.
// CHECK-SUCCESS: 214877.484

// CHECK-NO-FAST-MATH: IR Dump After MathTransformPass
// CHECK-NO-FAST-MATH-NOT: 214877.484

module {
  func.func @main(%arg0: tensor<4x8xf32>, %arg1: tensor<8x4xf32>) -> tensor<4x4xf32> {
    %0 = "stablehlo.dot"(%arg0, %arg1) : (tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<4x4xf32>
    %1 = "stablehlo.tanh"(%0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }
}
