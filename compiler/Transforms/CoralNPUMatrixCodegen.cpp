// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <functional>

#include "compiler/Transforms/CoralNPUTileSizeSelectionUtils.h"
#include "compiler/Transforms/Passes.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

//===----------------------------------------------------------------------===//
// CoralNPU Matrix Codegen Pass
//
// Lowers 2D vector.contract to Zvt matrix instructions (msetmtype, vtzero,
// vtmms.tvv / vtfmm.tvv / vtfmm.alt.tvv, vtmv).
//
// Supports FP32->FP32, BF16->FP32 and INT8->INT32 in 16x16, 16x32, 32x16 and
// 32x32 tile shapes. A contraction yielded by an scf.for reduction (lb 0,
// step 1, constant trip count, k the row of B and the row or innermost index
// of A, no memory writes in the body) replaces the whole loop with one
// inline-asm K loop holding the accumulators in mt0..mt12; otherwise each
// iteration runs its own multiply block.
//
// The tiles start from vtzero for a zero `acc`, or are seeded row by row
// (vle32.v + vtmv.t.v) when `acc` is a row-major tile load; any other `acc`
// is left to the generic lowering.
//===----------------------------------------------------------------------===//

namespace mlir::coralnpu_compiler {

#define GEN_PASS_DEF_CORALNPUMATRIXCODEGEN
#include "compiler/Transforms/Passes.h.inc"

namespace {

// A memref and the indices of a tile's first element.
struct TileAddr {
  Value base;
  SmallVector<Value> indices;
  explicit operator bool() const { return base != nullptr; }
};

// A contraction, its destination tile, the enclosing K-reduction scf.for (if
// any), the ops the drain replaces and the seed address (null: zero).
struct AccumulatorChain {
  vector::ContractionOp contractOp;
  TileAddr c;
  scf::ForOp loop;
  SmallVector<Operation*> opsToErase;
  TileAddr init;
};

Value traceToSourceValue(Value val) {
  while (Operation* def = val.getDefiningOp()) {
    if (isa<arith::ExtFOp, arith::ExtSIOp, vector::BroadcastOp,
            vector::ShapeCastOp, vector::ExtractStridedSliceOp>(def)) {
      val = def->getOperand(0);
    } else if (auto fe = dyn_cast<vector::FromElementsOp>(def)) {
      auto ex = fe.getElements().front().getDefiningOp<vector::ExtractOp>();
      if (!ex) break;
      val = ex.getSource();
    } else {
      break;
    }
  }
  return val;
}

Type traceToSourceElementType(Value val) {
  return getElementTypeOrSelf(traceToSourceValue(val));
}

// Drops a widening ext, as the asm reads the narrow input elements.
Value stripExt(Value v) {
  Operation* d = v.getDefiningOp();
  return isa_and_nonnull<arith::ExtFOp, arith::ExtSIOp>(d) ? d->getOperand(0)
                                                           : v;
}

bool isSupportedMatrixContraction(vector::ContractionOp op) {
  VectorType lhsTy = op.getLhsType(), rhsTy = op.getRhsType();
  auto accTy = dyn_cast<VectorType>(op.getAccType());
  if (!accTy || accTy.getRank() != 2) return false;
  int64_t m = accTy.getDimSize(0), n = accTy.getDimSize(1);
  if ((m != 16 && m != 32) || (n != 16 && n != 32)) return false;
  if (lhsTy.getNumElements() != m || rhsTy.getNumElements() != n) return false;
  if (!isZvtElementTypes(traceToSourceElementType(op.getLhs()),
                         traceToSourceElementType(op.getRhs()),
                         accTy.getElementType()))
    return false;
  // The asm writes LHS-indexed rows to C rows, so C must be M x N.
  auto maps = op.getIndexingMapsArray();
  auto iters = op.getIteratorTypes().getValue();
  unsigned mDim = maps[2].getDimPosition(0), nDim = maps[2].getDimPosition(1);
  return op.getKind() == vector::CombiningKind::ADD &&
         vector::isParallelIterator(iters[mDim]) &&
         vector::isParallelIterator(iters[nDim]) &&
         maps[0].isFunctionOfDim(mDim) && maps[1].isFunctionOfDim(nDim);
}

void createInlineAsm(OpBuilder& b, Location loc, StringRef asmStr,
                     StringRef constraints, ValueRange operands) {
  LLVM::InlineAsmOp::create(
      b, loc, TypeRange{}, operands, b.getStringAttr(asmStr),
      b.getStringAttr(constraints), /*has_side_effects=*/b.getUnitAttr(),
      /*is_align_stack=*/nullptr, /*tail_call_kind=*/nullptr,
      /*convergent=*/nullptr, /*asm_dialect=*/nullptr,
      /*operand_attrs=*/nullptr);
}

// Bit size of the scalable type used for inline-asm "^vr" (vector register)
// operands: LMUL4, so operands are 4-register aligned as the RTL requires.
constexpr int64_t kScalableBits = 256;

Value castToScalable(OpBuilder& b, Location loc, Value val) {
  VectorType ty = cast<VectorType>(val.getType());
  Value flat = vector::ShapeCastOp::create(
      b, loc, VectorType::get({ty.getNumElements()}, ty.getElementType()), val);
  VectorType scalableTy =
      VectorType::get({kScalableBits / ty.getElementTypeBitWidth()},
                      ty.getElementType(), {true});
  Value zero =
      LLVM::ConstantOp::create(b, loc, b.getI64Type(), b.getI64IntegerAttr(0));
  return LLVM::CallIntrinsicOp::create(
             b, loc, static_cast<Type>(scalableTy),
             b.getStringAttr("llvm.vector.insert"),
             ValueRange{LLVM::PoisonOp::create(b, loc, scalableTy), flat, zero})
      .getResult(0);
}

// Returns the `tileIdx`-th contiguous 16-element slice of `val`.
Value extractSubVector(OpBuilder& b, Location loc, Value val, int64_t tileIdx) {
  auto vecTy = cast<VectorType>(val.getType());
  if (vecTy.getNumElements() == 16) return val;
  Value mat = vector::ShapeCastOp::create(
      b, loc,
      VectorType::get({vecTy.getNumElements() / 16, 16},
                      vecTy.getElementType()),
      val);
  return vector::ExtractOp::create(b, loc, mat, ArrayRef<int64_t>{tileIdx});
}

// Returns the byte address of `indices` in `memref` and its row stride in
// bytes. The innermost stride must be 1.
std::pair<Value, Value> extractPtrAndStride(OpBuilder& b, Location loc,
                                            Value memref, ValueRange indices,
                                            int64_t elemBytes) {
  auto s = memref::ExtractStridedMetadataOp::create(b, loc, memref);
  auto strides = s.getStrides();
  Value stride0 = strides[strides.size() - 2];
  Value base =
      memref::ExtractAlignedPointerAsIndexOp::create(b, loc, s.getBaseBuffer());
  Value bytes = arith::ConstantIndexOp::create(b, loc, elemBytes);
  auto add = [&](Value x, Value y) {
    return arith::AddIOp::create(b, loc, x, y);
  };
  auto mul = [&](Value x, Value y) {
    return arith::MulIOp::create(b, loc, x, y);
  };
  Value off = add(s.getOffset(), indices.back());
  for (auto [idx, stride] : llvm::zip(indices.drop_back(), strides))
    off = add(off, mul(idx, stride));
  Type i32 = b.getI32Type();
  return {arith::IndexCastOp::create(b, loc, i32, add(base, mul(off, bytes))),
          arith::IndexCastOp::create(b, loc, i32, mul(stride0, bytes))};
}

// Zvt accumulator tiles are always 32-bit (f32 or i32).
constexpr int64_t kAccElemBytes = 4;

// msetmtype operands (decode: third_party/coralnpu .../design/RvvFrontEnd.sv).
// mtype (rs1): tm = bits[23:10], tk = bits[7:5], mtwiden = bits[1:0]. Always
// TM=16, TK=1; mtwiden is 3 for i8 (TWIDEN=4), 2 for bf16 and 1 for f32.
constexpr int64_t kMtypeI8 = (16 << 10) | (1 << 5) | 3;
constexpr int64_t kMtypeBF16 = (16 << 10) | (1 << 5) | 2;
constexpr int64_t kMtypeF32 = (16 << 10) | (1 << 5) | 1;
// vtype (rs2): SEW = bits[5:3], LMUL = bits[2:0], altfmt = bit 8. altfmt makes
// the *second* operand signed INT8 / BF16; the opcode does so for the first
// (vtmms, vtfmm.alt). FP32 leaves it clear.
constexpr int64_t kVtypeI8 = (0 << 3) | 0 | (1 << 8);    // SEW8, LMUL1, altfmt
constexpr int64_t kVtypeBF16 = (1 << 3) | 1 | (1 << 8);  // SEW16, LMUL2, altfmt
constexpr int64_t kVtypeF32 = (2 << 3) | 2;              // SEW32, LMUL4

// Programs mtype/vtype, then TN=16 (msetmtype zeroes vl). Emitted per block, as
// LLVM may insert a vset* between blocks, which clears mtype and vtype.altfmt.
// Clobbers t5 and t6. A null `elTy` selects the 32-bit accumulator config.
std::string emitTileConfigAsm(Type elTy = {}) {
  bool isI8 = elTy && elTy.isInteger(8), isBF16 = elTy && elTy.isBF16();
  return llvm::formatv(
      "li t5, {0}\n\tli t6, {1}\n\tmsetmtype t5, t6\n\t"
      "li t6, 16\n\tmsettn zero, t6\n\t",
      isI8 ? kMtypeI8 : (isBF16 ? kMtypeBF16 : kMtypeF32),
      isI8 ? kVtypeI8 : (isBF16 ? kVtypeBF16 : kVtypeF32));
}

StringRef getMatmulOp(Type elTy) {
  if (elTy.isInteger(8)) return "vtmms.tvv";
  return elTy.isBF16() ? "vtfmm.alt.tvv" : "vtfmm.tvv";
}

// Drains accumulator tiles mt0..mt12 to memory via vtmv.v.t + vse32.v or, with
// `load`, seeds them from memory via vle32.v + vtmv.t.v. `ptrReg`/`strideReg`
// name the asm operands holding the pointer and stride. Clobbers t0, t1,
// t4-t6 and v0-v3.
std::string emitTileRowsAsm(int64_t numTilesM, int64_t numTilesN,
                            StringRef ptrReg, StringRef strideReg, bool load) {
  std::string asmStr = emitTileConfigAsm() +
                       llvm::formatv("mv t5, {0}\n\t{1}", strideReg,
                                     numTilesM > 1 ? "slli t1, t5, 4\n\t" : "")
                           .str();
  const char* lui[] = {"li t0, 0\n\t", "lui t0, 0x20000\n\t",
                       "lui t0, 0x40000\n\t", "lui t0, 0x60000\n\t"};
  StringRef move = load ? "vle32.v v0, (t4)\n\tvtmv.t.v t0, v0\n\t"
                        : "vtmv.v.t v0, t0\n\tvse32.v v0, (t4)\n\t";
  int label = 2;
  for (int64_t m = 0; m < numTilesM; ++m) {
    for (int64_t n = 0; n < numTilesN; ++n) {
      asmStr += m ? llvm::formatv("add t4, {0}, t1\n\t", ptrReg)
                  : llvm::formatv("mv t4, {0}\n\t", ptrReg);
      if (n)
        asmStr += llvm::formatv("addi t4, t4, {0}\n\t", n * 16 * kAccElemBytes);
      asmStr += llvm::formatv(
          "{0}li t6, 16\n\t{1}:\n\t{2}add t4, t4, t5\n\t"
          "addi t0, t0, 1\n\taddi t6, t6, -1\n\tbnez t6, {1}b\n\t",
          lui[m * 2 + n], label++, move);
    }
  }
  return asmStr;
}

// Clears the tiles (vtfmm/vtmms accumulate in place) under the 32-bit
// accumulator config. Clobbers t5 and t6.
std::string emitTileZeroAsm(int64_t numTilesM, int64_t numTilesN) {
  std::string asmStr = emitTileConfigAsm();
  for (int64_t m = 0; m < numTilesM; ++m)
    for (int64_t n = 0; n < numTilesN; ++n)
      asmStr += llvm::formatv("vtzero mt{0}\n\t", (m * 2 + n) * 4);
  return asmStr;
}

// `contigA`: k indexes the rows of A, so each K step loads a contiguous row;
// otherwise a strided column. B rows are always contiguous.
void emitInlineFusedGEMM(OpBuilder& b, Location loc, const TileAddr& lhs,
                         const TileAddr& rhs, const AccumulatorChain& chain,
                         int64_t numTilesM, int64_t numTilesN,
                         int64_t numKSteps, Type inputElTy, bool contigA) {
  int64_t inBytes = inputElTy.getIntOrFloatBitWidth() / 8;
  auto [ptrA, strideA] =
      extractPtrAndStride(b, loc, lhs.base, lhs.indices, inBytes);
  auto [ptrB, strideB] =
      extractPtrAndStride(b, loc, rhs.base, rhs.indices, inBytes);
  auto [ptrC, strideC] =
      extractPtrAndStride(b, loc, chain.c.base, chain.c.indices, kAccElemBytes);
  SmallVector<Value> operands = {ptrA, strideA, ptrB, strideB, ptrC, strideC};
  std::string constraints = "{a0},{a1},{a2},{a3},{a4},{a5},";

  // The tile init and narrow config precede the pointer setup: they use
  // t0/t1/t4-t6 as scratch, which the K loop then holds.
  std::string asmStr;
  if (chain.init) {
    auto [ptrI, strideI] = extractPtrAndStride(
        b, loc, chain.init.base, chain.init.indices, kAccElemBytes);
    operands.append({ptrI, strideI});
    constraints += "{a6},{a7},";
    asmStr = emitTileRowsAsm(numTilesM, numTilesN, "$6", "$7", /*load=*/true);
  } else {
    asmStr = emitTileZeroAsm(numTilesM, numTilesN);
  }
  if (!inputElTy.isF32()) asmStr += emitTileConfigAsm(inputElTy);
  asmStr += "mv t0, $0\n\tmv t1, $1\n\tmv t2, $2\n\tmv t3, $3\n\t";

  // A: pointer t0 (second M tile t5), stride t1. B: t2 (second N tile t4), t3.
  auto load = [&](bool contig, StringRef v, StringRef p) {
    if (contig)
      asmStr += llvm::formatv("vle{0}.v {1}, ({2})\n\t", inBytes * 8, v, p);
    else
      asmStr +=
          llvm::formatv("vlse{0}.v {1}, ({2}), t1\n\t", inBytes * 8, v, p);
  };
  auto stepA = [&](StringRef p) {
    if (contigA)
      asmStr += llvm::formatv("add {0}, {0}, t1\n\t", p);
    else
      asmStr += llvm::formatv("addi {0}, {0}, {1}\n\t", p, inBytes);
  };

  if (numTilesM > 1) {
    if (contigA)
      asmStr += llvm::formatv("addi t5, t0, {0}\n\t", 16 * inBytes);
    else
      asmStr += "slli t5, t1, 4\n\tadd t5, t0, t5\n\t";
  }
  if (numTilesN > 1)
    asmStr += llvm::formatv("addi t4, t2, {0}\n\t", 16 * inBytes);
  asmStr += llvm::formatv("li t6, {0}\n\t1:\n\t", numKSteps);
  load(contigA, "v4", "t0");
  if (numTilesM > 1) load(contigA, "v24", "t5");
  load(/*contig=*/true, "v8", "t2");
  if (numTilesN > 1) load(/*contig=*/true, "v12", "t4");
  for (int64_t m = 0; m < numTilesM; ++m)
    for (int64_t n = 0; n < numTilesN; ++n)
      asmStr +=
          llvm::formatv("{0} mt{1}, {2}, {3}\n\t", getMatmulOp(inputElTy),
                        (m * 2 + n) * 4, m ? "v24" : "v4", n ? "v12" : "v8");
  stepA("t0");
  if (numTilesM > 1) stepA("t5");
  asmStr += "add t2, t2, t3\n\t";
  if (numTilesN > 1) asmStr += "add t4, t4, t3\n\t";
  asmStr += "addi t6, t6, -1\n\tbnez t6, 1b\n\t";

  asmStr += emitTileRowsAsm(numTilesM, numTilesN, "$4", "$5", /*load=*/false);
  // A clobber names one register, so list each LMUL4 group in full.
  constraints +=
      "~{t0},~{t1},~{t2},~{t3},~{t4},~{t5},~{t6},~{v0},~{v1},~{v2},~{v3},"
      "~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},"
      "~{v14},~{v15},~{v24},~{v25},~{v26},~{v27},~{memory}";
  createInlineAsm(b, loc, asmStr, constraints, operands);
}

void lowerContractionToAsm(OpBuilder& b, vector::ContractionOp op,
                           int64_t numTilesM, int64_t numTilesN, Type inputElTy,
                           bool zeroTiles) {
  Location loc = op.getLoc();
  b.setInsertionPoint(op);
  SmallVector<Value> operands;
  Value lhs = stripExt(op.getLhs()), rhs = stripExt(op.getRhs());
  for (int64_t m = 0; m < numTilesM; ++m)
    operands.push_back(
        castToScalable(b, loc, extractSubVector(b, loc, lhs, m)));
  for (int64_t n = 0; n < numTilesN; ++n)
    operands.push_back(
        castToScalable(b, loc, extractSubVector(b, loc, rhs, n)));

  std::string constraints;
  for (size_t i = 0; i < operands.size(); ++i) constraints += "^vr,";
  constraints += "~{t5},~{t6}";

  // Program the config unless the vtzero block left the right (f32) one.
  std::string asmStr = zeroTiles ? emitTileZeroAsm(numTilesM, numTilesN) : "";
  if (!zeroTiles || !inputElTy.isF32()) asmStr += emitTileConfigAsm(inputElTy);
  for (int64_t m = 0; m < numTilesM; ++m)
    for (int64_t n = 0; n < numTilesN; ++n)
      asmStr +=
          llvm::formatv("{0} mt{1}, ${2}, ${3}\n\t", getMatmulOp(inputElTy),
                        (m * 2 + n) * 4, m, numTilesM + n);
  createInlineAsm(b, loc, asmStr, constraints, operands);
  // The drain asm writes the result from the tiles; zero is a placeholder.
  op.replaceAllUsesWith(
      arith::ConstantOp::create(b, loc, b.getZeroAttr(op.getType()))
          .getResult());
  op.erase();
}

// Returns the address of a row-major access: rank >= 2 with unit innermost
// stride, else null.
TileAddr getRowMajorAddr(Value base, ValueRange indices) {
  auto ty = dyn_cast<MemRefType>(base.getType());
  if (!ty || indices.size() < 2 || !ty.isLastDimUnitStride()) return {};
  return {base, llvm::to_vector(indices)};
}

// Returns the row-major address of a contraction operand's load, or null.
TileAddr extractBaseAndIndices(Value val) {
  val = traceToSourceValue(val);
  if (auto fe = val.getDefiningOp<vector::FromElementsOp>()) {
    Value e0 = fe.getElements().front();
    auto te = e0.getDefiningOp<vector::ToElementsOp>();
    if (!te || te.getResult(0) != e0) return {};
    val = te.getSource();
  }
  if (auto ld = val.getDefiningOp<vector::LoadOp>())
    return getRowMajorAddr(ld.getBase(), ld.getIndices());
  auto rd = val.getDefiningOp<vector::TransferReadOp>();
  if (!rd || rd.getMask() || !rd.getPermutationMap().isMinorIdentity())
    return {};
  return getRowMajorAddr(rd.getBase(), rd.getIndices());
}

// Returns 0 if `iv` is the innermost index of a row-major access, 1 if it is
// the row index, else -1.
int64_t getKPos(const TileAddr& a, Value iv) {
  ArrayRef<Value> idx = a.indices;
  if (!a || llvm::count(idx, iv) != 1) return -1;
  if (idx.back() == iv) return 0;
  return idx[idx.size() - 2] == iv ? 1 : -1;
}

// A transfer the asm can replace: unmasked, in bounds and row-major.
TileAddr getTransferAddr(VectorTransferOpInterface op) {
  if (op.getMask() || op.hasOutOfBoundsDim() ||
      !op.getPermutationMap().isMinorIdentity())
    return {};
  return getRowMajorAddr(op.getBase(), op.getIndices());
}

// Returns `idx - idx0` if it is a known constant.
std::optional<int64_t> getConstantDelta(Value idx, Value idx0) {
  if (idx == idx0) return 0;
  if (auto add = idx.getDefiningOp<arith::AddIOp>()) {
    if (add.getLhs() == idx0) return getConstantIntValue(add.getRhs());
    if (add.getRhs() == idx0) return getConstantIntValue(add.getLhs());
  }
  std::optional<int64_t> i = getConstantIntValue(idx);
  std::optional<int64_t> i0 = getConstantIntValue(idx0);
  if (i && i0) return *i - *i0;
  return std::nullopt;
}

// True if `a` addresses row `r` of the tile at `a0`.
bool isTileRow(const TileAddr& a, const TileAddr& a0, int64_t r) {
  ArrayRef<Value> idx = a.indices, idx0 = a0.indices;
  return a.base == a0.base && idx.drop_back(2) == idx0.drop_back(2) &&
         idx.back() == idx0.back() &&
         getConstantDelta(idx[idx.size() - 2], idx0[idx0.size() - 2]) == r;
}

// Returns the address of a row-major tile load: a 2-D transfer_read, or a
// from_elements of one vector.load per consecutive row. Null otherwise.
TileAddr getTileLoadAddress(Value v) {
  if (auto rd = v.getDefiningOp<vector::TransferReadOp>())
    return getTransferAddr(rd);
  auto fe = v.getDefiningOp<vector::FromElementsOp>();
  if (!fe) return {};
  auto ty = cast<VectorType>(v.getType());
  int64_t n = ty.getDimSize(1);
  TileAddr row0;
  for (int64_t r = 0; r < ty.getDimSize(0); ++r) {
    auto te = fe.getElements()[r * n].getDefiningOp<vector::ToElementsOp>();
    auto ld = te ? te.getSource().getDefiningOp<vector::LoadOp>() : nullptr;
    if (!ld || ld.getVectorType().getRank() != 1 ||
        !llvm::equal(fe.getElements().slice(r * n, n), te.getResults()))
      return {};
    TileAddr a = getRowMajorAddr(ld.getBase(), ld.getIndices());
    if (r == 0) row0 = a;
    if (!a || !isTileRow(a, row0, r)) return {};
  }
  return row0;
}

// The asm either zeroes the tiles or seeds them from a tile load, setting
// `addr` to its address. Returns false for any other init.
bool matchInit(Value init, TileAddr& addr) {
  addr = getTileLoadAddress(init);
  return addr || matchPattern(init, m_Zero()) ||
         matchPattern(init, m_AnyZeroFloat());
}

TileAddr getTileStoreAddress(Operation* op) {
  auto w = dyn_cast<vector::TransferWriteOp>(op);
  return w ? getTransferAddr(w) : TileAddr{};
}

SmallVector<AccumulatorChain> collectChains(mlir::FunctionOpInterface funcOp) {
  SmallVector<AccumulatorChain> chains;

  funcOp.walk([&](vector::ContractionOp contract) {
    if (!isSupportedMatrixContraction(contract) || !contract->hasOneUse())
      return;
    Operation* user = *contract->getUsers().begin();
    auto forOp = dyn_cast<scf::ForOp>(user->getParentOp());
    TileAddr init;
    if (!isa<scf::YieldOp>(user) || !forOp) {
      if (TileAddr c = getTileStoreAddress(user);
          c && matchInit(contract.getAcc(), init))
        chains.push_back({contract, std::move(c), nullptr, {user}, init});
      return;
    }
    // The loop must carry only this accumulator, read only by the contraction.
    if (forOp.getNumResults() != 1 ||
        contract.getAcc() != forOp.getRegionIterArgs()[0] ||
        !contract.getAcc().hasOneUse())
      return;

    // Every user of the result is replaced by the tile drain, which writes all
    // rows: either a lone transfer_write, or row extracts stored to one tile.
    Value res = forOp.getResult(0);
    SmallVector<Operation*> toErase;
    SmallVector<std::pair<int64_t, TileAddr>> rows;
    llvm::SmallBitVector stored(cast<VectorType>(res.getType()).getDimSize(0));
    TileAddr c;
    for (Operation* u : res.getUsers()) {
      auto ext = dyn_cast<vector::ExtractOp>(u);
      if (!ext) {
        c = getTileStoreAddress(u);
        if (!c || !res.hasOneUse()) return;
        toErase.push_back(u);
        stored.set();
        continue;
      }
      ArrayRef<int64_t> pos = ext.getStaticPosition();
      if (pos.size() != 1 || pos[0] < 0) return;
      toErase.push_back(ext);
      for (Operation* eu : ext.getResult().getUsers()) {
        auto st = dyn_cast<vector::StoreOp>(eu);
        TileAddr a =
            st ? getRowMajorAddr(st.getBase(), st.getIndices()) : TileAddr{};
        if (!a) return;
        toErase.push_back(st);
        stored.set(pos[0]);
        if (pos[0] == 0) c = a;
        rows.push_back({pos[0], std::move(a)});
      }
    }
    // The drain goes right after the loop, so the writes it replaces must be
    // in the loop's block.
    auto inLoopBlock = [&](Operation* op) {
      return op->getBlock() == forOp->getBlock();
    };
    auto isRow = [&](auto& r) { return isTileRow(r.second, c, r.first); };
    if (c && stored.all() && llvm::all_of(rows, isRow) &&
        llvm::all_of(toErase, inLoopBlock) &&
        matchInit(forOp.getInitArgs()[0], init))
      chains.push_back(
          {contract, std::move(c), forOp, std::move(toErase), init});
  });
  return chains;
}

struct CoralNPUMatrixCodegenPass
    : public impl::CoralNPUMatrixCodegenBase<CoralNPUMatrixCodegenPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    if (!hasZvtTargetFeature(funcOp)) return;

    OpBuilder builder(&getContext());
    SmallVector<AccumulatorChain> chains = collectChains(funcOp);

    for (auto& chain : chains) {
      auto contractOp = chain.contractOp;
      auto accTy = cast<VectorType>(contractOp.getType());
      int64_t numTilesM = accTy.getDimSize(0) / 16;
      int64_t numTilesN = accTy.getDimSize(1) / 16;
      Type inputElTy = traceToSourceElementType(contractOp.getLhs());
      Location loc = contractOp.getLoc();
      auto eraseChainOps = [&]() {
        for (Operation* op : llvm::reverse(chain.opsToErase)) op->erase();
      };

      if (scf::ForOp forOp = chain.loop) {
        // Moves `op`, if in or after `forOp`, before it, operands first.
        std::function<void(Operation*)> hoist = [&](Operation* op) {
          if (!op || (!forOp->isProperAncestor(op) &&
                      !(op->getBlock() == forOp->getBlock() &&
                        forOp->isBeforeInBlock(op)))) {
            return;
          }
          for (Value operand : op->getOperands())
            hoist(operand.getDefiningOp());
          op->moveBefore(forOp);
        };
        auto hoistAddr = [&](const TileAddr& t) {
          hoist(t.base.getDefiningOp());
          for (Value v : t.indices) hoist(v.getDefiningOp());
        };
        // C is hoisted even when unfused: the drain goes right after the loop.
        hoistAddr(chain.c);

        TileAddr lhs = extractBaseAndIndices(contractOp.getLhs());
        TileAddr rhs = extractBaseAndIndices(contractOp.getRhs());
        Value iv = forOp.getInductionVar();
        int64_t kA = getKPos(lhs, iv), kB = getKPos(rhs, iv);

        // The asm K loop starts at k = 0 and advances k by one per step.
        int64_t kTripCount =
            isConstantIntValue(forOp.getLowerBound(), 0) &&
                    isConstantIntValue(forOp.getStep(), 1)
                ? getConstantIntValue(forOp.getUpperBound()).value_or(0)
                : 0;
        // Fusion erases the loop body, so its ops must be read-only.
        auto onlyReads = [](Operation& o) {
          return isMemoryEffectFree(&o) ||
                 hasSingleEffect<MemoryEffects::Read>(&o);
        };

        if (kA >= 0 && kB == 1 && kTripCount > 0 &&
            llvm::all_of(forOp.getBody()->without_terminator(), onlyReads)) {
          builder.setInsertionPoint(forOp);
          Value c0 = arith::ConstantIndexOp::create(builder, loc, 0);
          for (TileAddr* t : {&lhs, &rhs}) {
            llvm::replace(t->indices, iv, c0);
            hoistAddr(*t);
          }
          emitInlineFusedGEMM(builder, loc, lhs, rhs, chain, numTilesM,
                              numTilesN, kTripCount, inputElTy,
                              /*contigA=*/kA == 1);
          eraseChainOps();
          forOp.erase();
          continue;
        }
      }
      // The unfused asm takes the operand values as inputElTy.
      if (getElementTypeOrSelf(stripExt(contractOp.getLhs())) != inputElTy ||
          getElementTypeOrSelf(stripExt(contractOp.getRhs())) != inputElTy)
        continue;
      // Seeds (`load`) or drains the tiles at `addr`.
      auto emitTileRows = [&](const TileAddr& addr, bool load) {
        auto [ptr, stride] = extractPtrAndStride(builder, loc, addr.base,
                                                 addr.indices, kAccElemBytes);
        createInlineAsm(
            builder, loc,
            emitTileRowsAsm(numTilesM, numTilesN, "$0", "$1", load),
            "{a0},{a1},~{t0},~{t1},~{t4},~{t5},~{t6},~{v0},~{v1},~{v2},"
            "~{v3},~{memory}",
            {ptr, stride});
      };
      // Seed the tiles, or zero them before a K loop, in a separate block;
      // otherwise the multiply block zeroes them.
      if (chain.loop || chain.init) {
        builder.setInsertionPoint(chain.loop ? chain.loop : contractOp);
        if (chain.init) {
          emitTileRows(chain.init, /*load=*/true);
        } else {
          createInlineAsm(builder, loc, emitTileZeroAsm(numTilesM, numTilesN),
                          "~{t5},~{t6}", {});
        }
      }
      lowerContractionToAsm(builder, contractOp, numTilesM, numTilesN,
                            inputElTy,
                            /*zeroTiles=*/!chain.loop && !chain.init);
      if (chain.loop)
        builder.setInsertionPointAfter(chain.loop);
      else
        builder.setInsertionPoint(chain.opsToErase.front());
      emitTileRows(chain.c, /*load=*/false);
      eraseChainOps();
    }

    // Lower the contractions we did not take with OuterProduct: 0010 makes
    // LLVMCPUVirtualVectorLowering skip them and ConvertToLLVM would use Dot.
    bool hasContract =
        funcOp
            .walk([](vector::ContractionOp) { return WalkResult::interrupt(); })
            .wasInterrupted();
    if (!hasContract) return;

    RewritePatternSet patterns(&getContext());
    vector::populateVectorToVectorCanonicalizationPatterns(patterns);
    vector::populateVectorContractLoweringPatterns(
        patterns, vector::VectorContractLowering::OuterProduct);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
      signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createCoralNPUMatrixCodegenPass() {
  return std::make_unique<CoralNPUMatrixCodegenPass>();
}

}  // namespace mlir::coralnpu_compiler
