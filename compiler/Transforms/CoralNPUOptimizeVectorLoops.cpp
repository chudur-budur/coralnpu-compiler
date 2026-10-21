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

#include "compiler/Transforms/CoralNPUTileSizeSelectionUtils.h"
#include "compiler/Transforms/Passes.h"

// IREE:
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"

// MLIR:
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// LLVM:
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

// Standard C/C++ headers
#include <optional>
#include <utility>

namespace mlir::coralnpu_compiler {

#define GEN_PASS_DEF_CORALNPUOPTIMIZEVECTORLOOPS
#include "compiler/Transforms/Passes.h.inc"

namespace {

// Maximum number of operations in a loop body after full unrolling or
// unroll-and-jam.
constexpr int64_t kMaxUnrolledOps = 96;
// Maximum number of operations in the body of an inner loop to unroll by 4 or
// 2.
constexpr size_t kMaxUnrolledBodyOps = 32;
// VLEN assumed without a +zvl<N>b target feature.
constexpr int64_t kDefaultVlenBits = 128;
// The 1/4, 3/4, and 1/2 `regFileBits` caps below reserve vector registers for
// non-accumulator loads and temporaries so unrolling does not cause spills.

// Total bits of `t` if it is a fixed-width VectorType, else 0.
int64_t getVectorBits(Type t) {
  auto vecTy = dyn_cast<VectorType>(t);
  return vecTy && !vecTy.isScalable()
             ? vecTy.getNumElements() * vecTy.getElementTypeBitWidth()
             : 0;
}

// Total bits of the vector results of `forOp`, or 0 if it has a non-vector
// result.
int64_t vectorResultBits(scf::ForOp forOp) {
  if (!llvm::all_of(forOp.getResultTypes(), getVectorBits)) return 0;
  return llvm::sum_of(llvm::map_range(forOp.getResultTypes(), getVectorBits));
}

// True if `forOp` contains a vector.contract.
bool containsContraction(scf::ForOp forOp) {
  return forOp
      ->walk([](vector::ContractionOp) { return WalkResult::interrupt(); })
      .wasInterrupted();
}

// True if `v` is a product in `body` of values widened from <= 16-bit types.
bool isWidenedExpression(Value v, Block *body) {
  Operation *def = v.getDefiningOp();
  if (!def || def->getBlock() != body) return false;
  if (isa<arith::ExtFOp, arith::SIToFPOp, arith::UIToFPOp>(def))
    return getElementTypeOrSelf(def->getOperand(0)).getIntOrFloatBitWidth() <=
           16;
  return isa<arith::MulFOp, arith::NegFOp, vector::BroadcastOp,
             vector::ShapeCastOp>(def) &&
         llvm::all_of(def->getOperands(),
                      [&](Value o) { return isWidenedExpression(o, body); });
}

// Replaces a loop-carried scalar updated by `vector.reduction <add>` with a
// vector accumulator reduced once after the loop. This reorders a float sum,
// so it needs the reassoc flag, or `fastMath` for products of values widened
// from <= 16-bit types.
struct HoistLoopReduction : OpRewritePattern<scf::ForOp> {
  HoistLoopReduction(MLIRContext *context, bool fastMath)
      : OpRewritePattern(context), fastMath(fastMath) {}

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    for (auto [idx, iterArg] : llvm::enumerate(forOp.getRegionIterArgs())) {
      Value yielded = yieldOp.getOperand(idx);
      auto bcast = yielded.getDefiningOp<vector::BroadcastOp>();
      auto red = (bcast ? bcast.getSource() : yielded)
                     .getDefiningOp<vector::ReductionOp>();
      if (!red || red.getKind() != vector::CombiningKind::ADD ||
          !red.getAcc() || red.getSourceVectorType().getRank() != 1 ||
          red.getSourceVectorType().isScalable() || !red->hasOneUse() ||
          (bcast && !bcast->hasOneUse()) || !iterArg.hasOneUse())
        continue;
      auto extract = red.getAcc().getDefiningOp<vector::ExtractOp>();
      if (extract ? !(extract.getSource() == iterArg && extract->hasOneUse() &&
                      extract.getStaticPosition() == ArrayRef<int64_t>{0} &&
                      !extract.getSourceVectorType().isScalable() &&
                      extract.getSourceVectorType().getNumElements() == 1)
                  : red.getAcc() != iterArg)
        continue;
      VectorType vecTy = red.getSourceVectorType();
      bool isFloat = isa<FloatType>(vecTy.getElementType());
      bool canReassoc =
          isa<IntegerType>(vecTy.getElementType()) ||
          arith::bitEnumContainsAll(red.getFastmath(),
                                    arith::FastMathFlags::reassoc) ||
          (fastMath && isWidenedExpression(red.getVector(), forOp.getBody()));
      if (!canReassoc) continue;
      Type accTy = iterArg.getType();
      Location loc = forOp.getLoc();
      Value init = forOp.getInitArgs()[idx];
      rewriter.setInsertionPoint(forOp);
      Value zero =
          arith::ConstantOp::create(rewriter, loc, rewriter.getZeroAttr(vecTy));
      Value scalarInit = extract ? vector::ExtractOp::create(
                                       rewriter, loc, init, /*position=*/0)
                                       .getResult()
                                 : init;
      Value vecInit = vector::InsertOp::create(rewriter, loc, scalarInit, zero,
                                               /*position=*/0);
      SmallVector<Value> inits(forOp.getInitArgs());
      inits[idx] = vecInit;
      auto newFor =
          scf::ForOp::create(rewriter, loc, forOp.getLowerBound(),
                             forOp.getUpperBound(), forOp.getStep(), inits,
                             /*bodyBuilder=*/nullptr, forOp.getUnsignedCmp());
      newFor->setDiscardableAttrs(forOp->getDiscardableAttrDictionary());
      Block *newBody = newFor.getBody();
      Value vecAcc = newBody->getArgument(idx + 1);
      rewriter.setInsertionPoint(yieldOp);
      // Hoisting already reorders the sum, so the vector sum and the final
      // reduction may reassociate further.
      arith::FastMathFlags fastmath = red.getFastmath();
      if (isFloat) fastmath = fastmath | arith::FastMathFlags::reassoc;
      Value sum = vector::makeArithReduction(
          rewriter, loc, vector::CombiningKind::ADD, vecAcc, red.getVector(),
          arith::FastMathFlagsAttr::get(getContext(), fastmath));
      rewriter.modifyOpInPlace(
          yieldOp, [&, idx = idx] { yieldOp.setOperand(idx, sum); });
      if (bcast) rewriter.eraseOp(bcast);
      rewriter.eraseOp(red);
      if (extract) rewriter.eraseOp(extract);
      rewriter.mergeBlocks(forOp.getBody(), newBody, newBody->getArguments());
      rewriter.setInsertionPointAfter(newFor);
      Value result =
          vector::ReductionOp::create(rewriter, loc, vector::CombiningKind::ADD,
                                      newFor.getResult(idx), fastmath);
      if (extract)
        result = vector::BroadcastOp::create(rewriter, loc, accTy, result);
      SmallVector<Value> results(newFor.getResults());
      results[idx] = result;
      rewriter.replaceOp(forOp, results);
      return success();
    }
    return failure();
  }

 private:
  bool fastMath;
};

// The buffer that `v` views, or null if its root may alias another value.
// Subspans of the same binding or globals of the same symbol map to one root;
// distinct bindings, allocs, or function arguments are assumed not to alias.
Value getBuffer(Value v, DenseMap<uint64_t, Value> &bindings,
                DenseMap<Attribute, Value> &globals) {
  while (auto view = v.getDefiningOp<ViewLikeOpInterface>())
    v = view.getViewSource();
  if (auto subspan = v.getDefiningOp<
                     iree_compiler::IREE::HAL::InterfaceBindingSubspanOp>())
    return bindings.try_emplace(subspan.getBinding().getZExtValue(), v)
        .first->second;
  if (auto global = v.getDefiningOp<memref::GetGlobalOp>())
    return globals.try_emplace(global.getNameAttr(), v).first->second;
  if (isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(v.getDefiningOp()))
    return v;
  auto arg = dyn_cast<BlockArgument>(v);
  return arg && isa_and_nonnull<FunctionOpInterface>(
                    arg.getOwner()->getParentOp())
             ? v
             : Value();
}

// True if `forOp` has only scf.for regions, known read/write memory effects,
// and no buffer both read and written.
bool iterationsAreIndependent(scf::ForOp forOp) {
  DenseSet<Value> read, written;
  DenseMap<uint64_t, Value> bindings;
  DenseMap<Attribute, Value> globals;
  WalkResult result = forOp.getBody()->walk([&](Operation *op) {
    if (op->getNumRegions() != 0)
      return isa<scf::ForOp>(op) ? WalkResult::advance()
                                 : WalkResult::interrupt();
    auto iface = dyn_cast<MemoryEffectOpInterface>(op);
    if (!iface) return WalkResult::interrupt();
    SmallVector<MemoryEffects::EffectInstance> effects;
    iface.getEffects(effects);
    for (auto &effect : effects) {
      Value v = effect.getValue();
      bool isRead = isa<MemoryEffects::Read>(effect.getEffect());
      if (!v || (!isRead && !isa<MemoryEffects::Write>(effect.getEffect())))
        return WalkResult::interrupt();
      v = getBuffer(v, bindings, globals);
      if (!v || (isRead ? written : read).contains(v))
        return WalkResult::interrupt();
      (isRead ? read : written).insert(v);
    }
    return WalkResult::advance();
  });
  return !result.wasInterrupted();
}

// True if unroll-and-jam of `outer` keeps the order of its writes. It
// interleaves the nested loops of adjacent iterations and moves the code
// before them ahead of the previous iteration's, so only the code after the
// last nested loop may write. Expects iterationsAreIndependent(outer), which
// rejects ops with unknown effects.
bool writesFollowNestedLoops(scf::ForOp outer) {
  bool beforeLastLoop = false;
  for (Operation &op : llvm::reverse(outer.getBody()->without_terminator())) {
    beforeLastLoop |= isa<scf::ForOp>(op);
    if (beforeLastLoop && op.walk([](Operation *nested) {
                              return hasEffect<MemoryEffects::Write>(nested)
                                         ? WalkResult::interrupt()
                                         : WalkResult::advance();
                            }).wasInterrupted())
      return false;
  }
  return true;
}

// Fully unrolls small innermost vector-accumulating loops inside another loop
// (e.g. convolution filter window loops).
void unrollFilterLoops(func::FuncOp funcOp, int64_t regFileBits,
                       int maxLoopUnrolling) {
  uint64_t maxTrip = maxLoopUnrolling > 0 ? std::min(4, maxLoopUnrolling) : 4;
  SmallVector<scf::ForOp> filterLoops;
  funcOp->walk([&](scf::ForOp forOp) {
    std::optional<APInt> trip = forOp.getStaticTripCount();
    int64_t accBits = vectorResultBits(forOp);
    if (trip && trip->uge(2) && trip->ule(maxTrip) &&
        isa<scf::ForOp>(forOp->getParentOp()) &&
        forOp.getOps<scf::ForOp>().empty() && accBits > 0 &&
        accBits <= regFileBits / 4 &&
        static_cast<int64_t>(trip->getZExtValue() *
                             forOp.getBody()->getOperations().size()) <=
            kMaxUnrolledOps &&
        !containsContraction(forOp))
      filterLoops.push_back(forOp);
  });
  for (scf::ForOp loop : filterLoops)
    (void)loopUnrollByFactor(loop, loop.getStaticTripCount()->getZExtValue());
}

// Unroll-and-jams loops enclosing vector-accumulating loops so adjacent
// iterations share loop overhead and invariant operands.
void unrollAndJamAccumulatingLoops(func::FuncOp funcOp, int64_t regFileBits,
                                   int maxLoopUnrolling) {
  SmallVector<scf::ForOp> outerLoops;
  funcOp->walk([&](scf::ForOp forOp) {
    std::optional<APInt> trip = forOp.getStaticTripCount();
    if (forOp.getNumResults() == 0 && trip && !trip->isZero() &&
        forOp.getConstantStep() &&
        forOp.getInductionVar().getType().isIndex() &&
        llvm::any_of(forOp.getOps<scf::ForOp>(), [](scf::ForOp inner) {
          return inner.getOps<scf::ForOp>().empty() &&
                 vectorResultBits(inner) != 0 && !containsContraction(inner);
        }))
      outerLoops.push_back(forOp);
  });
  for (scf::ForOp outer : outerLoops) {
    if (!iterationsAreIndependent(outer) || !writesFollowNestedLoops(outer))
      continue;
    DenseSet<Value> perCopy{outer.getInductionVar()};
    outer.getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (llvm::any_of(op->getOperands(),
                       [&](Value v) { return perCopy.contains(v); }))
        perCopy.insert(op->result_begin(), op->result_end());
    });
    int64_t perCopyBits = 0, sharedBits = 0;
    size_t maxInnerOps = 0;
    outer.getBody()->walk([&](scf::ForOp inner) {
      if (!inner.getOps<scf::ForOp>().empty()) return;
      perCopyBits += vectorResultBits(inner);
      maxInnerOps =
          std::max(maxInnerOps, inner.getBody()->getOperations().size());
      for (Operation &op : inner.getBody()->getOperations()) {
        if (!isa<vector::TransferReadOp, vector::LoadOp>(op)) continue;
        auto vecTy = cast<VectorType>(op.getResult(0).getType());
        if (vecTy.isScalable() || vecTy.getNumElements() == 1) continue;
        (perCopy.contains(op.getResult(0)) ? perCopyBits : sharedBits) +=
            getVectorBits(vecTy);
      }
    });
    uint64_t trip = outer.getStaticTripCount()->getZExtValue();
    for (uint64_t factor : {4, 2}) {
      if ((maxLoopUnrolling <= 0 ||
           factor <= static_cast<uint64_t>(maxLoopUnrolling)) &&
          trip % factor == 0 &&
          static_cast<int64_t>(factor * maxInnerOps) <= kMaxUnrolledOps &&
          static_cast<int64_t>(factor) * perCopyBits + sharedBits <=
              (regFileBits * 3) / 4) {
        (void)loopUnrollJamByFactor(outer, factor);
        break;
      }
    }
  }
}

// Unrolls small innermost vector-accumulating loops by 4 or 2.
void unrollAccumulatingInnerLoops(func::FuncOp funcOp, int64_t vlenBits,
                                  int64_t regFileBits, int maxLoopUnrolling) {
  bool hasZvfbfwma = hasZvtTargetFeature(funcOp) &&
                     hasCoralNPUTargetFeature(funcOp, "+zvfbfwma") &&
                     vlenBits == 128;
  auto v32f32 = VectorType::get({32}, Builder(funcOp).getF32Type());
  SmallVector<scf::ForOp> innerLoops;
  funcOp->walk([&](scf::ForOp forOp) {
    std::optional<APInt> trip = forOp.getStaticTripCount();
    int64_t accBits = vectorResultBits(forOp);
    if (!trip || trip->isZero() || trip->urem(2) != 0 || accBits == 0 ||
        accBits > regFileBits / 2 || !forOp.getOps<scf::ForOp>().empty())
      return;
    Block &body = *forOp.getBody();
    // Leave BF16 vecmat K loops to CoralNPUMatrixCodegen (vfwmaccbf16.vf,
    // VLEN 128 only); keep in sync with its matcher.
    if (hasZvfbfwma &&
        (forOp.getNumResults() == 1 || forOp.getNumResults() == 2) &&
        llvm::all_of(forOp.getResultTypes(),
                     [&](Type t) { return t == v32f32; }) &&
        isConstantIntValue(forOp.getStep(), 1) &&
        !body.getOps<vector::FMAOp>().empty() &&
        llvm::any_of(body.getOps<arith::ExtFOp>(), [](arith::ExtFOp e) {
          return getElementTypeOrSelf(e.getIn().getType()).isBF16();
        }))
      return;
    if (body.getOperations().size() > kMaxUnrolledBodyOps ||
        llvm::any_of(body.getOperations(), [](Operation &op) {
          return isa_and_nonnull<math::MathDialect>(op.getDialect()) ||
                 isa<arith::DivFOp, vector::ReductionOp, vector::ContractionOp>(
                     op);
        }))
      return;
    innerLoops.push_back(forOp);
  });
  for (scf::ForOp inner : innerLoops) {
    uint64_t trip = inner.getStaticTripCount()->getZExtValue();
    uint64_t factor = trip % 4 == 0 ? 4 : 2;
    if (maxLoopUnrolling > 0)
      factor = std::min<uint64_t>(factor, maxLoopUnrolling);
    if (factor > 1) (void)loopUnrollByFactor(inner, factor);
  }
}

// True if `ty` has positive, non-overlapping strides across all dimensions.
bool hasNonOverlappingLayout(MemRefType ty) {
  if (ty.getLayout().isIdentity()) return ty.getRank() > 0;
  SmallVector<int64_t> strides;
  int64_t offset;
  if (ty.getRank() == 0 || failed(ty.getStridesAndOffset(strides, offset)) ||
      strides.back() <= 0)
    return false;
  ArrayRef<int64_t> shape = ty.getShape();
  for (size_t d = 0; d + 1 < shape.size(); ++d) {
    if (shape[d + 1] <= 0 || strides[d] <= 0 || strides[d + 1] <= 0 ||
        strides[d] / shape[d + 1] < strides[d + 1])
      return false;
  }
  return true;
}

// Interchanges 2D elementwise nests to hoist outer-invariant vector sub-DAGs.
void interchangeBroadcastNests(func::FuncOp funcOp, int64_t regFileBits) {
  SmallVector<scf::ForOp> broadcastNests;
  funcOp->walk([&](scf::ForOp outer) {
    std::optional<APInt> outerTrip = outer.getStaticTripCount();
    auto innerOps = outer.getOps<scf::ForOp>();
    if (outer.getNumResults() != 0 || !llvm::hasSingleElement(innerOps) ||
        !outerTrip || outerTrip->ule(1))
      return;
    scf::ForOp inner = *innerOps.begin();
    std::optional<APInt> innerTrip = inner.getStaticTripCount();
    if (inner.getNumResults() == 0 && inner.getOps<scf::ForOp>().empty() &&
        innerTrip && !innerTrip->isZero() &&
        llvm::all_of(
            ValueRange{inner.getLowerBound(), inner.getUpperBound(),
                       inner.getStep()},
            [&](Value v) { return outer.isDefinedOutsideOfLoop(v); }) &&
        llvm::all_of(outer.getBody()->without_terminator(), [&](Operation &o) {
          return &o == inner ||
                 (o.isBeforeInBlock(inner) && isMemoryEffectFree(&o) &&
                  llvm::none_of(o.getResultTypes(), llvm::IsaPred<VectorType>));
        }))
      broadcastNests.push_back(outer);
  });
  for (scf::ForOp outer : broadcastNests) {
    if (!iterationsAreIndependent(outer)) continue;
    scf::ForOp inner = *outer.getOps<scf::ForOp>().begin();
    DenseSet<Value> dependsOnOuter{outer.getInductionVar()};
    int64_t hoistableVecOps = 0, hoistedBits = 0;
    for (Block *blk : {outer.getBody(), inner.getBody()}) {
      for (Operation &op : blk->without_terminator()) {
        if (&op == inner) break;
        if ((!isMemoryEffectFree(&op) &&
             !hasSingleEffect<MemoryEffects::Read>(&op)) ||
            llvm::any_of(op.getOperands(),
                         [&](Value v) { return dependsOnOuter.contains(v); })) {
          dependsOnOuter.insert(op.result_begin(), op.result_end());
        } else if (blk == inner.getBody() &&
                   llvm::any_of(op.getResultTypes(), getVectorBits)) {
          ++hoistableVecOps;
        }
      }
    }
    auto staysInInner = [&](Operation *op) {
      return op->getNumResults() == 0 ||
             llvm::any_of(op->getResults(),
                          [&](Value r) { return dependsOnOuter.contains(r); });
    };
    for (Operation &op : inner.getBody()->without_terminator()) {
      for (Value v : op.getResults()) {
        if (!dependsOnOuter.contains(v) &&
            llvm::any_of(v.getUsers(), staysInInner))
          hoistedBits += getVectorBits(v.getType());
      }
    }
    if (!hoistableVecOps || hoistedBits > regFileBits / 2) continue;
    DenseSet<Value> dependsOnInner{inner.getInductionVar()};
    for (Operation &op : inner.getBody()->without_terminator())
      if (llvm::any_of(op.getOperands(),
                       [&](Value v) { return dependsOnInner.contains(v); }))
        dependsOnInner.insert(op.result_begin(), op.result_end());
    // Interchange keeps each location's last write only for one store that
    // indexes a non-overlapping memref by both IVs, each index depending on at
    // most one IV: its writes then form a grid, last in both loop orders.
    int64_t numStores = 0;
    auto indexesBothApart = [&](Operation &op) {
      if (++numStores > 1) return false;
      Value base;
      ValueRange indices;
      if (auto w = dyn_cast<vector::TransferWriteOp>(op)) {
        if (w.getMask() || w.hasOutOfBoundsDim() ||
            !w.getPermutationMap().isMinorIdentity() ||
            w.getVectorType().isScalable())
          return false;
        base = w.getBase();
        indices = w.getIndices();
      } else if (auto w = dyn_cast<vector::StoreOp>(op)) {
        // vector.store may write past the end of a row: require that it stays
        // inside its row, so that each index still addresses one dim.
        VectorType vt = w.getVectorType();
        auto memTy = w.getMemRefType();
        if (vt.isScalable() || vt.getRank() != 1 || !memTy.getRank() ||
            !memTy.isLastDimUnitStride() ||
            ShapedType::isDynamic(memTy.getShape().back()))
          return false;
        int64_t vecLen = vt.getNumElements(), lastDim = memTy.getShape().back();
        Value lastIdx = w.getIndices().back();
        std::optional<int64_t> maxIdx = getConstantIntValue(lastIdx);
        if (maxIdx && *maxIdx < 0) return false;
        for (scf::ForOp loop : {outer, inner}) {
          if (lastIdx != loop.getInductionVar() || loop.getUnsignedCmp())
            continue;
          auto lb = getConstantIntValue(loop.getLowerBound()),
               ub = getConstantIntValue(loop.getUpperBound()),
               step = getConstantIntValue(loop.getStep());
          if (lb && ub && step && *lb >= 0 && *step > 0 && *ub > *lb)
            maxIdx = *lb + ((*ub - *lb - 1) / *step) * *step;
        }
        if (!maxIdx || *maxIdx > lastDim - vecLen) return false;
        base = w.getBase();
        indices = w.getIndices();
      } else if (auto w = dyn_cast<memref::StoreOp>(op)) {
        base = w.getMemRef();
        indices = w.getIndices();
      }
      auto memTy = base ? dyn_cast<MemRefType>(base.getType()) : MemRefType();
      if (!memTy || !hasNonOverlappingLayout(memTy) ||
          dependsOnOuter.contains(base) || dependsOnInner.contains(base))
        return false;
      bool outerIdx = false, innerIdx = false;
      for (Value v : indices) {
        bool o = dependsOnOuter.contains(v), i = dependsOnInner.contains(v);
        if (o && i) return false;
        outerIdx |= o;
        innerIdx |= i;
      }
      return outerIdx && innerIdx;
    };
    if (llvm::any_of(inner.getBody()->without_terminator(), [&](Operation &op) {
          return !isMemoryEffectFree(&op) &&
                 !hasSingleEffect<MemoryEffects::Read>(&op) &&
                 !indexesBothApart(op);
        }))
      continue;
    OpBuilder b(outer);
    auto newOuter = scf::ForOp::create(
        b, inner.getLoc(), inner.getLowerBound(), inner.getUpperBound(),
        inner.getStep(), /*initArgs=*/ValueRange{}, /*bodyBuilder=*/nullptr,
        inner.getUnsignedCmp());
    newOuter->setDiscardableAttrs(inner->getDiscardableAttrDictionary());
    b.setInsertionPointToStart(newOuter.getBody());
    auto newInner = scf::ForOp::create(
        b, outer.getLoc(), outer.getLowerBound(), outer.getUpperBound(),
        outer.getStep(), /*initArgs=*/ValueRange{}, /*bodyBuilder=*/nullptr,
        outer.getUnsignedCmp());
    newInner->setDiscardableAttrs(outer->getDiscardableAttrDictionary());
    Operation *newInnerTerm = newInner.getBody()->getTerminator();
    inner.getInductionVar().replaceAllUsesWith(newOuter.getInductionVar());
    outer.getInductionVar().replaceAllUsesWith(newInner.getInductionVar());
    for (Block *blk : {outer.getBody(), inner.getBody()}) {
      for (Operation &op :
           llvm::make_early_inc_range(blk->without_terminator())) {
        if (&op == inner) break;
        op.moveBefore(staysInInner(&op) ? newInnerTerm : newInner);
      }
    }
    outer.erase();
  }
}

struct CoralNPUOptimizeVectorLoopsPass final
    : impl::CoralNPUOptimizeVectorLoopsBase<CoralNPUOptimizeVectorLoopsPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    int64_t vlenBits =
        getVlenFromTargetFeatures(funcOp).value_or(kDefaultVlenBits);
    int64_t regFileBits = numVectorRegisters * vlenBits;

    RewritePatternSet patterns(&getContext());
    patterns.add<HoistLoopReduction>(&getContext(), fastMath);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
      return signalPassFailure();

    unrollFilterLoops(funcOp, regFileBits, maxLoopUnrolling);
    unrollAndJamAccumulatingLoops(funcOp, regFileBits, maxLoopUnrolling);
    unrollAccumulatingInnerLoops(funcOp, vlenBits, regFileBits,
                                 maxLoopUnrolling);
    interchangeBroadcastNests(funcOp, regFileBits);
  }
};

}  // namespace

std::unique_ptr<Pass> createCoralNPUOptimizeVectorLoopsPass() {
  return std::make_unique<CoralNPUOptimizeVectorLoopsPass>();
}

std::unique_ptr<Pass> createCoralNPUOptimizeVectorLoopsPass(
    CoralNPUOptimizeVectorLoopsOptions options) {
  return std::make_unique<CoralNPUOptimizeVectorLoopsPass>(options);
}

}  // namespace mlir::coralnpu_compiler
