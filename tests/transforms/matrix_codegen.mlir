// RUN: iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(coralnpu-matrix-codegen)))))" %s | FileCheck %s
// RUN: iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(coralnpu-matrix-codegen)))))" %s | FileCheck %s --check-prefix=NOVSET

// Whole-file invariant: any vset* clears mtype and vtype.altfmt
// (RvvFrontEnd.sv), so the pass must emit none at all.
// NOVSET-NOT: vset

#target = #hal.executable.target<"coralnpu", "coralnpu-elf", {
  cpu_features = "+zvtbase"
}>

// Same backend, but the Zvt feature is absent.
#target_no_zvt = #hal.executable.target<"coralnpu", "coralnpu-elf", {
  cpu_features = "+v"
}>

// Non-CoralNPU backends must be left untouched even with +zvtbase.
#target_llvmcpu = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  cpu_features = "+zvtbase"
}>

// CHECK-LABEL: hal.executable private @matmul {
module {
  hal.executable private @matmul {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul(
        func.func @matmul(%arg0: memref<16x1xf32>, %arg1: memref<1x16xf32>, %arg2: memref<16x16xf32>) {
          %c0 = arith.constant 0 : index
          %cst_0 = arith.constant 0.0 : f32

          %lhs = vector.transfer_read %arg0[%c0, %c0], %cst_0 : memref<16x1xf32>, vector<16x1xf32>
          %rhs = vector.transfer_read %arg1[%c0, %c0], %cst_0 : memref<1x16xf32>, vector<1x16xf32>
          %acc = arith.constant dense<0.0> : vector<16x16xf32>

          // CHECK: llvm.inline_asm {{.*}}li t5, 16417{{.*}}li t6, 18{{.*}}msetmtype t5, t6{{.*}}msettn zero, t6{{.*}}vtzero mt0{{.*}}vtfmm.tvv mt0
          %res = vector.contract {
            indexing_maps = [
              affine_map<(d0, d1, d2) -> (d0, d2)>,
              affine_map<(d0, d1, d2) -> (d2, d1)>,
              affine_map<(d0, d1, d2) -> (d0, d1)>
            ],
            iterator_types = ["parallel", "parallel", "reduction"]
          } %lhs, %rhs, %acc : vector<16x1xf32>, vector<1x16xf32> into vector<16x16xf32>

          // Writeback drains mt0 row by row: vtmv.v.t + vse32.v, 16 times.
          // CHECK: llvm.inline_asm {{.*}}msetmtype t5, t6{{.*}}li t0, 0{{.*}}vtmv.v.t v0, t0{{.*}}vse32.v v0, (t4){{.*}}add t4, t4, t5{{.*}}addi t0, t0, 1{{.*}}bnez t6
          vector.transfer_write %res, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
          return
        }
      }
    }
  }

  // CHECK-LABEL: hal.executable private @matmul_loop {
  hal.executable private @matmul_loop {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_loop(
        func.func @matmul_loop(%arg0: memref<16x8xf32>, %arg1: memref<8x16xf32>, %arg2: memref<16x16xf32>) {
          // The init is read from %arg2, so the tiles are seeded from it.
          // CHECK: llvm.inline_asm {{.*}}li t5, 16417{{.*}}msetmtype t5, t6{{.*}}vle32.v v0, (t4){{.*}}vtmv.t.v t0, v0{{.*}}vlse32.v{{.*}}vtfmm.tvv mt0{{.*}}vse32.v
          %c0 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c1 = arith.constant 1 : index
          %cst_0 = arith.constant 0.0 : f32

          %acc_init = vector.transfer_read %arg2[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x16xf32>, vector<16x16xf32>
          %res = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %acc_init) -> (vector<16x16xf32>) {
            %lhs = vector.transfer_read %arg0[%c0, %iv], %cst_0 : memref<16x8xf32>, vector<16x1xf32>
            %rhs = vector.transfer_read %arg1[%iv, %c0], %cst_0 : memref<8x16xf32>, vector<1x16xf32>
            %next = vector.contract {
              indexing_maps = [
                affine_map<(d0, d1, d2) -> (d0, d2)>,
                affine_map<(d0, d1, d2) -> (d2, d1)>,
                affine_map<(d0, d1, d2) -> (d0, d1)>
              ],
              iterator_types = ["parallel", "parallel", "reduction"]
            } %lhs, %rhs, %acc : vector<16x1xf32>, vector<1x16xf32> into vector<16x16xf32>
            scf.yield %next : vector<16x16xf32>
          }
          vector.transfer_write %res, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
          return
        }
      }
    }
  }

  // Writeback via per-row vector.extract + vector.store. Every row is stored,
  // as the tile drain that replaces them writes all rows.
  // CHECK-LABEL: hal.executable private @matmul_loop_extract_store {
  hal.executable private @matmul_loop_extract_store {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_loop_extract_store(
        func.func @matmul_loop_extract_store(%arg0: memref<16x8xf32>, %arg1: memref<8x16xf32>, %arg2: memref<16x16xf32>) {
          // CHECK: llvm.inline_asm {{.*}}vtzero mt0{{.*}}vtfmm.tvv mt0{{.*}}vtmv.v.t v0, t0{{.*}}vse32.v
          // CHECK-NOT: vector.extract
          // CHECK-NOT: vector.store
          // CHECK-NOT: scf.for
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c2 = arith.constant 2 : index
          %c3 = arith.constant 3 : index
          %c4 = arith.constant 4 : index
          %c5 = arith.constant 5 : index
          %c6 = arith.constant 6 : index
          %c7 = arith.constant 7 : index
          %c8 = arith.constant 8 : index
          %c9 = arith.constant 9 : index
          %c10 = arith.constant 10 : index
          %c11 = arith.constant 11 : index
          %c12 = arith.constant 12 : index
          %c13 = arith.constant 13 : index
          %c14 = arith.constant 14 : index
          %c15 = arith.constant 15 : index
          %cst_0 = arith.constant 0.0 : f32

          %acc_init = arith.constant dense<0.0> : vector<16x16xf32>
          %res = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %acc_init) -> (vector<16x16xf32>) {
            %lhs = vector.transfer_read %arg0[%c0, %iv], %cst_0 : memref<16x8xf32>, vector<16x1xf32>
            %rhs = vector.transfer_read %arg1[%iv, %c0], %cst_0 : memref<8x16xf32>, vector<1x16xf32>
            %next = vector.contract {
              indexing_maps = [
                affine_map<(d0, d1, d2) -> (d0, d2)>,
                affine_map<(d0, d1, d2) -> (d2, d1)>,
                affine_map<(d0, d1, d2) -> (d0, d1)>
              ],
              iterator_types = ["parallel", "parallel", "reduction"]
            } %lhs, %rhs, %acc : vector<16x1xf32>, vector<1x16xf32> into vector<16x16xf32>
            scf.yield %next : vector<16x16xf32>
          }
          %row0 = vector.extract %res[0] : vector<16xf32> from vector<16x16xf32>
          vector.store %row0, %arg2[%c0, %c0] : memref<16x16xf32>, vector<16xf32>
          %row1 = vector.extract %res[1] : vector<16xf32> from vector<16x16xf32>
          vector.store %row1, %arg2[%c1, %c0] : memref<16x16xf32>, vector<16xf32>
          %row2 = vector.extract %res[2] : vector<16xf32> from vector<16x16xf32>
          vector.store %row2, %arg2[%c2, %c0] : memref<16x16xf32>, vector<16xf32>
          %row3 = vector.extract %res[3] : vector<16xf32> from vector<16x16xf32>
          vector.store %row3, %arg2[%c3, %c0] : memref<16x16xf32>, vector<16xf32>
          %row4 = vector.extract %res[4] : vector<16xf32> from vector<16x16xf32>
          vector.store %row4, %arg2[%c4, %c0] : memref<16x16xf32>, vector<16xf32>
          %row5 = vector.extract %res[5] : vector<16xf32> from vector<16x16xf32>
          vector.store %row5, %arg2[%c5, %c0] : memref<16x16xf32>, vector<16xf32>
          %row6 = vector.extract %res[6] : vector<16xf32> from vector<16x16xf32>
          vector.store %row6, %arg2[%c6, %c0] : memref<16x16xf32>, vector<16xf32>
          %row7 = vector.extract %res[7] : vector<16xf32> from vector<16x16xf32>
          vector.store %row7, %arg2[%c7, %c0] : memref<16x16xf32>, vector<16xf32>
          %row8 = vector.extract %res[8] : vector<16xf32> from vector<16x16xf32>
          vector.store %row8, %arg2[%c8, %c0] : memref<16x16xf32>, vector<16xf32>
          %row9 = vector.extract %res[9] : vector<16xf32> from vector<16x16xf32>
          vector.store %row9, %arg2[%c9, %c0] : memref<16x16xf32>, vector<16xf32>
          %row10 = vector.extract %res[10] : vector<16xf32> from vector<16x16xf32>
          vector.store %row10, %arg2[%c10, %c0] : memref<16x16xf32>, vector<16xf32>
          %row11 = vector.extract %res[11] : vector<16xf32> from vector<16x16xf32>
          vector.store %row11, %arg2[%c11, %c0] : memref<16x16xf32>, vector<16xf32>
          %row12 = vector.extract %res[12] : vector<16xf32> from vector<16x16xf32>
          vector.store %row12, %arg2[%c12, %c0] : memref<16x16xf32>, vector<16xf32>
          %row13 = vector.extract %res[13] : vector<16xf32> from vector<16x16xf32>
          vector.store %row13, %arg2[%c13, %c0] : memref<16x16xf32>, vector<16xf32>
          %row14 = vector.extract %res[14] : vector<16xf32> from vector<16x16xf32>
          vector.store %row14, %arg2[%c14, %c0] : memref<16x16xf32>, vector<16xf32>
          %row15 = vector.extract %res[15] : vector<16xf32> from vector<16x16xf32>
          vector.store %row15, %arg2[%c15, %c0] : memref<16x16xf32>, vector<16xf32>
          return
        }
      }
    }
  }

  // Dynamic trip count: the loop stays and the tiles are zeroed once before it.
  // CHECK-LABEL: hal.executable private @matmul_loop_unfused {
  hal.executable private @matmul_loop_unfused {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_loop_unfused(
        func.func @matmul_loop_unfused(%arg0: memref<16x8xf32>, %arg1: memref<8x16xf32>, %arg2: memref<16x16xf32>, %n: index) {
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %cst_0 = arith.constant 0.0 : f32

          %acc_init = arith.constant dense<0.0> : vector<16x16xf32>
          // CHECK: llvm.inline_asm {{.*}}msetmtype t5, t6{{.*}}vtzero mt0
          // CHECK: scf.for
          %res = scf.for %iv = %c0 to %n step %c1 iter_args(%acc = %acc_init) -> (vector<16x16xf32>) {
            %lhs = vector.transfer_read %arg0[%c0, %iv], %cst_0 : memref<16x8xf32>, vector<16x1xf32>
            %rhs = vector.transfer_read %arg1[%iv, %c0], %cst_0 : memref<8x16xf32>, vector<1x16xf32>
            // The multiply block re-programs mtype: a vset* may sit between it
            // and the vtzero block.
            // CHECK: llvm.inline_asm {{.*}}msetmtype t5, t6
            // CHECK-NOT: vtzero
            // CHECK-SAME: vtfmm.tvv mt0
            %next = vector.contract {
              indexing_maps = [
                affine_map<(d0, d1, d2) -> (d0, d2)>,
                affine_map<(d0, d1, d2) -> (d2, d1)>,
                affine_map<(d0, d1, d2) -> (d0, d1)>
              ],
              iterator_types = ["parallel", "parallel", "reduction"]
            } %lhs, %rhs, %acc : vector<16x1xf32>, vector<1x16xf32> into vector<16x16xf32>
            scf.yield %next : vector<16x16xf32>
          }
          // CHECK: llvm.inline_asm {{.*}}vtmv.v.t v0, t0{{.*}}vse32.v
          vector.transfer_write %res, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
          return
        }
      }
    }
  }

  // CHECK-LABEL: hal.executable private @matmul_i8 {
  hal.executable private @matmul_i8 {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_i8(
        func.func @matmul_i8(%arg0: memref<16x1xi8>, %arg1: memref<1x16xi8>, %arg2: memref<16x16xi32>) {
          %c0 = arith.constant 0 : index
          %cst_i8 = arith.constant 0 : i8

          %lhs = vector.transfer_read %arg0[%c0, %c0], %cst_i8 : memref<16x1xi8>, vector<16x1xi8>
          %rhs = vector.transfer_read %arg1[%c0, %c0], %cst_i8 : memref<1x16xi8>, vector<1x16xi8>
          %acc = arith.constant dense<0> : vector<16x16xi32>

          // vtzero runs under the f32 config; the i8 config sets altfmt (256)
          // to make operand B signed.
          // CHECK: llvm.inline_asm {{.*}}li t5, 16417{{.*}}li t6, 18{{.*}}msetmtype t5, t6{{.*}}vtzero mt0{{.*}}li t5, 16419{{.*}}li t6, 256{{.*}}msetmtype t5, t6{{.*}}vtmms.tvv mt0
          %res = vector.contract {
            indexing_maps = [
              affine_map<(d0, d1, d2) -> (d0, d2)>,
              affine_map<(d0, d1, d2) -> (d2, d1)>,
              affine_map<(d0, d1, d2) -> (d0, d1)>
            ],
            iterator_types = ["parallel", "parallel", "reduction"]
          } %lhs, %rhs, %acc : vector<16x1xi8>, vector<1x16xi8> into vector<16x16xi32>

          // Writeback drains mt0 row by row: vtmv.v.t + vse32.v, 16 times.
          // CHECK: llvm.inline_asm {{.*}}msetmtype t5, t6{{.*}}li t0, 0{{.*}}vtmv.v.t v0, t0{{.*}}vse32.v v0, (t4){{.*}}add t4, t4, t5{{.*}}addi t0, t0, 1{{.*}}bnez t6
          vector.transfer_write %res, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xi32>, memref<16x16xi32>
          return
        }
      }
    }
  }

  // CHECK-LABEL: hal.executable private @matmul_i8_loop {
  hal.executable private @matmul_i8_loop {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_i8_loop(
        func.func @matmul_i8_loop(%arg0: memref<16x8xi8>, %arg1: memref<8x16xi8>, %arg2: memref<16x16xi32>) {
          // CHECK: llvm.inline_asm {{.*}}li t5, 16417{{.*}}msetmtype t5, t6{{.*}}vtzero mt0{{.*}}li t5, 16419{{.*}}li t6, 256{{.*}}msetmtype t5, t6{{.*}}vlse8.v{{.*}}vle8.v{{.*}}vtmms.tvv mt0{{.*}}vse32.v
          %c0 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c1 = arith.constant 1 : index
          %cst_i8 = arith.constant 0 : i8

          %acc_init = arith.constant dense<0> : vector<16x16xi32>
          %res = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %acc_init) -> (vector<16x16xi32>) {
            %lhs = vector.transfer_read %arg0[%c0, %iv], %cst_i8 : memref<16x8xi8>, vector<16x1xi8>
            %rhs = vector.transfer_read %arg1[%iv, %c0], %cst_i8 : memref<8x16xi8>, vector<1x16xi8>
            %next = vector.contract {
              indexing_maps = [
                affine_map<(d0, d1, d2) -> (d0, d2)>,
                affine_map<(d0, d1, d2) -> (d2, d1)>,
                affine_map<(d0, d1, d2) -> (d0, d1)>
              ],
              iterator_types = ["parallel", "parallel", "reduction"]
            } %lhs, %rhs, %acc : vector<16x1xi8>, vector<1x16xi8> into vector<16x16xi32>
            scf.yield %next : vector<16x16xi32>
          }
          vector.transfer_write %res, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xi32>, memref<16x16xi32>
          return
        }
      }
    }
  }

  // CHECK-LABEL: hal.executable private @matmul_bf16 {
  hal.executable private @matmul_bf16 {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_bf16(
        func.func @matmul_bf16(%arg0: memref<16x1xbf16>, %arg1: memref<1x16xbf16>, %arg2: memref<16x16xf32>) {
          %c0 = arith.constant 0 : index
          %cst_bf16 = arith.constant 0.0 : bf16

          %lhs16 = vector.transfer_read %arg0[%c0, %c0], %cst_bf16 : memref<16x1xbf16>, vector<16x1xbf16>
          %rhs16 = vector.transfer_read %arg1[%c0, %c0], %cst_bf16 : memref<1x16xbf16>, vector<1x16xbf16>
          %lhs = arith.extf %lhs16 : vector<16x1xbf16> to vector<16x1xf32>
          %rhs = arith.extf %rhs16 : vector<1x16xbf16> to vector<1x16xf32>
          %acc = arith.constant dense<0.0> : vector<16x16xf32>

          // bf16 config: mtwiden=2 (16418), SEW16/LMUL2/altfmt (265). The asm
          // takes the unextended bf16 values in LMUL4 registers.
          // CHECK: llvm.inline_asm {{.*}}li t5, 16417{{.*}}li t6, 18{{.*}}msetmtype t5, t6{{.*}}vtzero mt0{{.*}}li t5, 16418{{.*}}li t6, 265{{.*}}msetmtype t5, t6{{.*}}vtfmm.alt.tvv mt0{{.*}}(vector<[16]xbf16>, vector<[16]xbf16>) -> ()
          %res = vector.contract {
            indexing_maps = [
              affine_map<(d0, d1, d2) -> (d0, d2)>,
              affine_map<(d0, d1, d2) -> (d2, d1)>,
              affine_map<(d0, d1, d2) -> (d0, d1)>
            ],
            iterator_types = ["parallel", "parallel", "reduction"]
          } %lhs, %rhs, %acc : vector<16x1xf32>, vector<1x16xf32> into vector<16x16xf32>

          vector.transfer_write %res, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
          return
        }
      }
    }
  }

  // CHECK-LABEL: hal.executable private @matmul_bf16_loop {
  hal.executable private @matmul_bf16_loop {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_bf16_loop(
        func.func @matmul_bf16_loop(%arg0: memref<16x8xbf16>, %arg1: memref<8x16xbf16>, %arg2: memref<16x16xf32>) {
          // CHECK: llvm.inline_asm {{.*}}li t5, 16417{{.*}}msetmtype t5, t6{{.*}}vtzero mt0{{.*}}li t5, 16418{{.*}}li t6, 265{{.*}}msetmtype t5, t6{{.*}}vlse16.v{{.*}}vle16.v{{.*}}vtfmm.alt.tvv mt0{{.*}}addi t0, t0, 2{{.*}}vse32.v
          %c0 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c1 = arith.constant 1 : index
          %cst_bf16 = arith.constant 0.0 : bf16

          %acc_init = arith.constant dense<0.0> : vector<16x16xf32>
          %res = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %acc_init) -> (vector<16x16xf32>) {
            %lhs = vector.transfer_read %arg0[%c0, %iv], %cst_bf16 : memref<16x8xbf16>, vector<16x1xbf16>
            %rhs = vector.transfer_read %arg1[%iv, %c0], %cst_bf16 : memref<8x16xbf16>, vector<1x16xbf16>
            %next = vector.contract {
              indexing_maps = [
                affine_map<(d0, d1, d2) -> (d0, d2)>,
                affine_map<(d0, d1, d2) -> (d2, d1)>,
                affine_map<(d0, d1, d2) -> (d0, d1)>
              ],
              iterator_types = ["parallel", "parallel", "reduction"]
            } %lhs, %rhs, %acc : vector<16x1xbf16>, vector<1x16xbf16> into vector<16x16xf32>
            scf.yield %next : vector<16x16xf32>
          }
          vector.transfer_write %res, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
          return
        }
      }
    }
  }

  // CHECK-LABEL: hal.executable private @matmul_i8_multitile {
  hal.executable private @matmul_i8_multitile {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_i8_multitile(
        func.func @matmul_i8_multitile(%arg0: memref<16x8xi8>, %arg1: memref<8x32xi8>, %arg2: memref<16x32xi32>) {
          // CHECK: llvm.inline_asm {{.*}}msetmtype t5, t6{{.*}}vtzero mt0{{.*}}vtzero mt4{{.*}}li t5, 16419{{.*}}msetmtype t5, t6{{.*}}vtmms.tvv mt0{{.*}}vtmms.tvv mt4{{.*}}lui t0, 0x20000{{.*}}vtmv.v.t
          %c0 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c1 = arith.constant 1 : index
          %cst_i8 = arith.constant 0 : i8

          %acc_init = arith.constant dense<0> : vector<16x32xi32>
          %res = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %acc_init) -> (vector<16x32xi32>) {
            %lhs = vector.transfer_read %arg0[%c0, %iv], %cst_i8 : memref<16x8xi8>, vector<16x1xi8>
            %rhs = vector.transfer_read %arg1[%iv, %c0], %cst_i8 : memref<8x32xi8>, vector<1x32xi8>
            %next = vector.contract {
              indexing_maps = [
                affine_map<(d0, d1, d2) -> (d0, d2)>,
                affine_map<(d0, d1, d2) -> (d2, d1)>,
                affine_map<(d0, d1, d2) -> (d0, d1)>
              ],
              iterator_types = ["parallel", "parallel", "reduction"]
            } %lhs, %rhs, %acc : vector<16x1xi8>, vector<1x32xi8> into vector<16x32xi32>
            scf.yield %next : vector<16x32xi32>
          }
          vector.transfer_write %res, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x32xi32>, memref<16x32xi32>
          return
        }
      }
    }
  }

  // CHECK-LABEL: hal.executable private @matmul_i8_transposed {
  hal.executable private @matmul_i8_transposed {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_i8_transposed(
        func.func @matmul_i8_transposed(%arg0: memref<8x16xi8>, %arg1: memref<8x32xi8>, %arg2: memref<16x32xi32>) {
          // CHECK: llvm.inline_asm {{.*}}vle8.v v4{{.*}}vle8.v v8{{.*}}vle8.v v12{{.*}}vtmms.tvv mt0{{.*}}vtmms.tvv mt4
          %c0 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c1 = arith.constant 1 : index
          %cst_0 = arith.constant 0 : i32
          %cst_i8 = arith.constant 0 : i8

          %acc_init = vector.transfer_read %arg2[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x32xi32>, vector<16x32xi32>
          %res = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %acc_init) -> (vector<16x32xi32>) {
            %lhs = vector.transfer_read %arg0[%iv, %c0], %cst_i8 : memref<8x16xi8>, vector<1x16xi8>
            %rhs = vector.transfer_read %arg1[%iv, %c0], %cst_i8 : memref<8x32xi8>, vector<1x32xi8>
            %next = vector.contract {
              indexing_maps = [
                affine_map<(d0, d1, d2) -> (d2, d0)>,
                affine_map<(d0, d1, d2) -> (d2, d1)>,
                affine_map<(d0, d1, d2) -> (d0, d1)>
              ],
              iterator_types = ["parallel", "parallel", "reduction"]
            } %lhs, %rhs, %acc : vector<1x16xi8>, vector<1x32xi8> into vector<16x32xi32>
            scf.yield %next : vector<16x32xi32>
          }
          vector.transfer_write %res, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x32xi32>, memref<16x32xi32>
          return
        }
      }
    }
  }

  // CHECK-LABEL: hal.executable private @matmul_2x2_multitile {
  hal.executable private @matmul_2x2_multitile {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_2x2_multitile(
        func.func @matmul_2x2_multitile(%arg0: memref<8x32xf32>, %arg1: memref<8x32xf32>, %arg2: memref<32x32xf32>) {
          // CHECK: llvm.inline_asm {{.*}}msetmtype t5, t6{{.*}}vtzero mt0{{.*}}vtzero mt4{{.*}}vtzero mt8{{.*}}vtzero mt12{{.*}}vtfmm.tvv mt0{{.*}}vtfmm.tvv mt4{{.*}}vtfmm.tvv mt8{{.*}}vtfmm.tvv mt12{{.*}}lui t0, 0x20000{{.*}}lui t0, 0x40000{{.*}}lui t0, 0x60000{{.*}}"{a0},{a1},{a2},{a3},{a4},{a5},~{t0},~{t1},~{t2},~{t3},~{t4},~{t5},~{t6},~{v0},~{v1},~{v2},~{v3},~{v4},{{.*}},~{v15},~{v24},~{v25},~{v26},~{v27},~{memory}"
          %c0 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c1 = arith.constant 1 : index
          %cst_0 = arith.constant 0.0 : f32

          %acc_init = arith.constant dense<0.0> : vector<32x32xf32>
          %res = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %acc_init) -> (vector<32x32xf32>) {
            %lhs = vector.transfer_read %arg0[%iv, %c0], %cst_0 : memref<8x32xf32>, vector<1x32xf32>
            %rhs = vector.transfer_read %arg1[%iv, %c0], %cst_0 : memref<8x32xf32>, vector<1x32xf32>
            %next = vector.contract {
              indexing_maps = [
                affine_map<(d0, d1, d2) -> (d2, d0)>,
                affine_map<(d0, d1, d2) -> (d2, d1)>,
                affine_map<(d0, d1, d2) -> (d0, d1)>
              ],
              iterator_types = ["parallel", "parallel", "reduction"]
            } %lhs, %rhs, %acc : vector<1x32xf32>, vector<1x32xf32> into vector<32x32xf32>
            scf.yield %next : vector<32x32xf32>
          }
          vector.transfer_write %res, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<32x32xf32>, memref<32x32xf32>
          return
        }
      }
    }
  }

  // CHECK-LABEL: hal.executable private @matmul_mixed_f32_i8 {
  hal.executable private @matmul_mixed_f32_i8 {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_mixed_f32_i8(
        func.func @matmul_mixed_f32_i8(%arg0: memref<16x1xf32>, %arg1: memref<1x16xf32>, %arg2: memref<16x16xf32>, %arg3: memref<16x1xi8>, %arg4: memref<1x16xi8>, %arg5: memref<16x16xi32>) {
          // Two element types in one function: each block carries its own
          // mtype/vtype, so no cross-chain bookkeeping is needed.
          // CHECK: llvm.inline_asm {{.*}}li t5, 16417{{.*}}li t6, 18{{.*}}msetmtype t5, t6{{.*}}vtzero mt0{{.*}}vtfmm.tvv mt0
          // The i8 acc is read from %arg5, so a separate block seeds its tiles.
          // CHECK: llvm.inline_asm {{.*}}vle32.v v0, (t4){{.*}}vtmv.t.v t0, v0
          // CHECK: llvm.inline_asm {{.*}}li t5, 16419{{.*}}li t6, 256{{.*}}msetmtype t5, t6{{.*}}vtmms.tvv mt0
          %c0 = arith.constant 0 : index
          %cst_0 = arith.constant 0.0 : f32
          %cst_i32 = arith.constant 0 : i32
          %cst_i8 = arith.constant 0 : i8

          %lhs0 = vector.transfer_read %arg0[%c0, %c0], %cst_0 : memref<16x1xf32>, vector<16x1xf32>
          %rhs0 = vector.transfer_read %arg1[%c0, %c0], %cst_0 : memref<1x16xf32>, vector<1x16xf32>
          %acc0 = arith.constant dense<0.0> : vector<16x16xf32>
          %res0 = vector.contract {
            indexing_maps = [
              affine_map<(d0, d1, d2) -> (d0, d2)>,
              affine_map<(d0, d1, d2) -> (d2, d1)>,
              affine_map<(d0, d1, d2) -> (d0, d1)>
            ],
            iterator_types = ["parallel", "parallel", "reduction"]
          } %lhs0, %rhs0, %acc0 : vector<16x1xf32>, vector<1x16xf32> into vector<16x16xf32>
          vector.transfer_write %res0, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>

          %lhs1 = vector.transfer_read %arg3[%c0, %c0], %cst_i8 : memref<16x1xi8>, vector<16x1xi8>
          %rhs1 = vector.transfer_read %arg4[%c0, %c0], %cst_i8 : memref<1x16xi8>, vector<1x16xi8>
          %acc1 = vector.transfer_read %arg5[%c0, %c0], %cst_i32 {in_bounds = [true, true]} : memref<16x16xi32>, vector<16x16xi32>
          %res1 = vector.contract {
            indexing_maps = [
              affine_map<(d0, d1, d2) -> (d0, d2)>,
              affine_map<(d0, d1, d2) -> (d2, d1)>,
              affine_map<(d0, d1, d2) -> (d0, d1)>
            ],
            iterator_types = ["parallel", "parallel", "reduction"]
          } %lhs1, %rhs1, %acc1 : vector<16x1xi8>, vector<1x16xi8> into vector<16x16xi32>
          vector.transfer_write %res1, %arg5[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xi32>, memref<16x16xi32>
          return
        }
      }
    }
  }

  // A CoralNPU target without +zvtbase must be left alone.
  // CHECK-LABEL: hal.executable private @matmul_no_zvtbase {
  hal.executable private @matmul_no_zvtbase {
    hal.executable.variant public @coralnpu_elf target(#target_no_zvt) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_no_zvtbase(
        func.func @matmul_no_zvtbase(%arg0: memref<16x1xf32>, %arg1: memref<1x16xf32>, %arg2: memref<16x16xf32>) {
          // CHECK-NOT: llvm.inline_asm
          // CHECK: vector.contract
          %c0 = arith.constant 0 : index
          %cst_0 = arith.constant 0.0 : f32
          %lhs = vector.transfer_read %arg0[%c0, %c0], %cst_0 : memref<16x1xf32>, vector<16x1xf32>
          %rhs = vector.transfer_read %arg1[%c0, %c0], %cst_0 : memref<1x16xf32>, vector<1x16xf32>
          %acc = vector.transfer_read %arg2[%c0, %c0], %cst_0 : memref<16x16xf32>, vector<16x16xf32>
          %res = vector.contract {
            indexing_maps = [
              affine_map<(d0, d1, d2) -> (d0, d2)>,
              affine_map<(d0, d1, d2) -> (d2, d1)>,
              affine_map<(d0, d1, d2) -> (d0, d1)>
            ],
            iterator_types = ["parallel", "parallel", "reduction"]
          } %lhs, %rhs, %acc : vector<16x1xf32>, vector<1x16xf32> into vector<16x16xf32>
          vector.transfer_write %res, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
          return
        }
      }
    }
  }

  // A non-CoralNPU backend must likewise be untouched.
  // CHECK-LABEL: hal.executable private @matmul_other_backend {
  hal.executable private @matmul_other_backend {
    hal.executable.variant public @embedded_elf target(#target_llvmcpu) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_other_backend(
        func.func @matmul_other_backend(%arg0: memref<16x1xf32>, %arg1: memref<1x16xf32>, %arg2: memref<16x16xf32>) {
          // CHECK-NOT: llvm.inline_asm
          // CHECK: vector.contract
          %c0 = arith.constant 0 : index
          %cst_0 = arith.constant 0.0 : f32
          %lhs = vector.transfer_read %arg0[%c0, %c0], %cst_0 : memref<16x1xf32>, vector<16x1xf32>
          %rhs = vector.transfer_read %arg1[%c0, %c0], %cst_0 : memref<1x16xf32>, vector<1x16xf32>
          %acc = vector.transfer_read %arg2[%c0, %c0], %cst_0 : memref<16x16xf32>, vector<16x16xf32>
          %res = vector.contract {
            indexing_maps = [
              affine_map<(d0, d1, d2) -> (d0, d2)>,
              affine_map<(d0, d1, d2) -> (d2, d1)>,
              affine_map<(d0, d1, d2) -> (d0, d1)>
            ],
            iterator_types = ["parallel", "parallel", "reduction"]
          } %lhs, %rhs, %acc : vector<16x1xf32>, vector<1x16xf32> into vector<16x16xf32>
          vector.transfer_write %res, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
          return
        }
      }
    }
  }

  // 8x8 is not a supported tile shape, so this takes the generic lowering.
  // CHECK-LABEL: hal.executable private @matmul_unsupported_shape {
  hal.executable private @matmul_unsupported_shape {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_unsupported_shape(
        func.func @matmul_unsupported_shape(%arg0: memref<8x1xf32>, %arg1: memref<1x8xf32>, %arg2: memref<8x8xf32>) {
          // CHECK-NOT: llvm.inline_asm
          // CHECK-NOT: vector.contract
          %c0 = arith.constant 0 : index
          %cst_0 = arith.constant 0.0 : f32
          %lhs = vector.transfer_read %arg0[%c0, %c0], %cst_0 : memref<8x1xf32>, vector<8x1xf32>
          %rhs = vector.transfer_read %arg1[%c0, %c0], %cst_0 : memref<1x8xf32>, vector<1x8xf32>
          %acc = vector.transfer_read %arg2[%c0, %c0], %cst_0 : memref<8x8xf32>, vector<8x8xf32>
          %res = vector.contract {
            indexing_maps = [
              affine_map<(d0, d1, d2) -> (d0, d2)>,
              affine_map<(d0, d1, d2) -> (d2, d1)>,
              affine_map<(d0, d1, d2) -> (d0, d1)>
            ],
            iterator_types = ["parallel", "parallel", "reduction"]
          } %lhs, %rhs, %acc : vector<8x1xf32>, vector<1x8xf32> into vector<8x8xf32>
          vector.transfer_write %res, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<8x8xf32>, memref<8x8xf32>
          return
        }
      }
    }
  }
}
