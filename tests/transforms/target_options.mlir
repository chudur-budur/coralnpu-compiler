// RUN: not %iree_compile --coralnpu-dtcm-size-kb=0 %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERR-ZERO
// RUN: not %iree_compile --coralnpu-roofline-speedup-threshold=-1 %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERR-THRESHOLD
// RUN: not %iree_compile --coralnpu-roofline-speedup-threshold=0 --coralnpu-linker-script-path=/nonexistent/path.ld %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERR-LINK
// RUN: %iree_compile --coralnpu-dtcm-size-kb=1024 --coralnpu-roofline-speedup-threshold=0 --coralnpu-dump-affinity-profile-format=pretty %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS

// CHECK-ERR-ZERO: coralnpu-dtcm-size-kb must be positive, got 0

// CHECK-ERR-THRESHOLD: coralnpu-roofline-speedup-threshold must be a finite, non-negative value, got -1

// CHECK-ERR-LINK: failed to serialize executables

// CHECK-SUCCESS: Execution Profile by Affinity:
// CHECK-SUCCESS:   Affinity: coralnpu
// CHECK-SUCCESS:     Dispatches: 1

module {
  func.func @main(%arg0: tensor<4x8xi32>, %arg1: tensor<8x4xi32>) -> tensor<4x4xi32> {
    %0 = "stablehlo.dot"(%arg0, %arg1) : (tensor<4x8xi32>, tensor<8x4xi32>) -> tensor<4x4xi32>
    return %0 : tensor<4x4xi32>
  }
}
