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

#include "compiler/Transforms/DeviceUtils.h"
#include "compiler/Transforms/Passes.h"

// MLIR:
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// LLVM:
#include "llvm/Support/MathExtras.h"

namespace mlir::coralnpu_compiler {

#define GEN_PASS_DEF_CORALNPURAISEAPPLYSCALE
#include "compiler/Transforms/Passes.h.inc"

namespace {

/// Returns `v` without zero extensions.
Value stripExtUI(Value v) {
  while (auto ext = v.getDefiningOp<arith::ExtUIOp>()) v = ext.getIn();
  return v;
}

/// Splits `v = a + b` (arith.addi) into the operand `isA` accepts and the
/// other one, or returns null values.
std::pair<Value, Value> splitAdd(Value v, function_ref<bool(Value)> isA) {
  auto add = v.getDefiningOp<arith::AddIOp>();
  if (!add) return {};
  if (isA(add.getLhs())) return {add.getLhs(), add.getRhs()};
  if (isA(add.getRhs())) return {add.getRhs(), add.getLhs()};
  return {};
}

/// Returns the operand of `arith.extsi %operand : iN to iM` if N <= `bits`, or
/// null.
Value getSignExtended(Value v, unsigned bits) {
  auto ext = v.getDefiningOp<arith::ExtSIOp>();
  return ext && ext.getIn().getType().getIntOrFloatBitWidth() <= bits
             ? ext.getIn()
             : Value();
}

/// Raises the 64-bit arithmetic of TosaToArith's ApplyScaleGenericOpConverter,
/// as folded by the TOSA input conversion, back to tosa.apply_scale:
///   trunci(shrsi(extsi(value) * multiplier + round [+ dir], shift))
/// where round is (1 << shift) >> 1 and dir, for DOUBLE_ROUND and shift > 31,
/// is +-2^30 by the sign of value (a select on shift > 31 for a dynamic
/// shift). A shift of 32 shifts right logically after folding. DOUBLE_ROUND
/// with a constant shift <= 31 is the same as SINGLE_ROUND.
struct RaiseApplyScale : OpRewritePattern<arith::TruncIOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(arith::TruncIOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getType().isInteger(32) || !op.getIn().getType().isInteger(64))
      return failure();
    Operation *shr = op.getIn().getDefiningOp();
    if (!isa_and_present<arith::ShRSIOp, arith::ShRUIOp>(shr)) return failure();
    Value shift64 = shr->getOperand(1);
    std::optional<int64_t> constShift = getConstantIntValue(shift64);
    Value shift8 = stripExtUI(shift64);
    if (constShift ? *constShift < 1 || *constShift > 63
                   : !shift8.getType().isInteger(8))
      return failure();
    if (isa<arith::ShRUIOp>(shr) && constShift != 32) return failure();

    // The double-round sum dir + base: under a select on shift > 31, or alone
    // for a constant shift > 31.
    Value acc = shr->getOperand(0), dir, base = acc;
    auto isDir = [](Value v) {
      auto select = v.getDefiningOp<arith::SelectOp>();
      return select && isConstantIntValue(select.getTrueValue(), 1 << 30) &&
             isConstantIntValue(select.getFalseValue(), -(1 << 30));
    };
    if (auto select = acc.getDefiningOp<arith::SelectOp>();
        select && !constShift) {
      auto cmp = select.getCondition().getDefiningOp<arith::CmpIOp>();
      base = select.getFalseValue();
      Value other;
      std::tie(dir, other) = splitAdd(select.getTrueValue(), isDir);
      if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::sgt ||
          stripExtUI(cmp.getLhs()) != shift8 ||
          !isConstantIntValue(cmp.getRhs(), 31) || other != base)
        return failure();
    } else if (constShift && *constShift > 31) {
      if (auto split = splitAdd(acc, isDir); split.first)
        std::tie(dir, base) = split;
    }

    // base = extsi(value) * multiplier + round.
    auto [mulValue, round] = splitAdd(base, [](Value v) {
      return isa_and_present<arith::MulIOp>(v.getDefiningOp());
    });
    if (!mulValue) return failure();
    if (constShift ? !isConstantIntValue(round, int64_t{1} << (*constShift - 1))
                   : !isDynamicRound(round, shift64))
      return failure();
    auto mul = mulValue.getDefiningOp<arith::MulIOp>();
    // A value narrower than i32 (e.g. an i8 input with a zero point of 0) was
    // extended to i32 for tosa.apply_scale, then to i64.
    Value value = getSignExtended(mul.getLhs(), /*bits=*/32);
    Value multiplier = getSignExtended(mul.getRhs(), /*bits=*/32);
    std::optional<int64_t> constMultiplier = getConstantIntValue(mul.getRhs());
    if (!value ||
        (multiplier ? !multiplier.getType().isInteger(32)
                    : !(constMultiplier && llvm::isInt<32>(*constMultiplier))))
      return failure();
    // dir selects by the sign of value.
    if (dir) {
      auto cmp = dir.getDefiningOp<arith::SelectOp>()
                     .getCondition()
                     .getDefiningOp<arith::CmpIOp>();
      if (!cmp) return failure();
      Value cmpValue = cmp.getLhs();
      if (Value narrow = getSignExtended(cmpValue, /*bits=*/16))
        cmpValue = narrow;
      if (cmp.getPredicate() != arith::CmpIPredicate::sge ||
          cmpValue != value || !matchPattern(cmp.getRhs(), m_Zero()))
        return failure();
    }

    Location loc = op.getLoc();
    if (!value.getType().isInteger(32))
      value = arith::ExtSIOp::create(rewriter, loc, op.getType(), value);
    if (!multiplier)
      multiplier = arith::ConstantIntOp::create(rewriter, loc, *constMultiplier,
                                                /*width=*/32);
    if (constShift)
      shift8 = arith::ConstantIntOp::create(rewriter, loc, *constShift,
                                            /*width=*/8);
    rewriter.replaceOpWithNewOp<tosa::ApplyScaleOp>(
        op, op.getType(), value, multiplier, shift8,
        dir ? tosa::RoundingMode::DOUBLE_ROUND
            : tosa::RoundingMode::SINGLE_ROUND);
    return success();
  }

 private:
  /// True if `round` is (1 << shift64) >> 1.
  static bool isDynamicRound(Value round, Value shift64) {
    auto shr = round.getDefiningOp<arith::ShRUIOp>();
    auto shl = shr ? shr.getLhs().getDefiningOp<arith::ShLIOp>() : nullptr;
    return shl && shl.getRhs() == shift64 &&
           matchPattern(shl.getLhs(), m_One()) &&
           matchPattern(shr.getRhs(), m_One());
  }
};

/// True if dimension `dim` of `v` has dynamic size `size`.
bool hasDimSize(Value v, unsigned dim, Value size) {
  auto dimOp = size.getDefiningOp<tensor::DimOp>();
  bool validDimOp = dimOp && dimOp.getConstantIndex() == dim;
  while (true) {
    if (validDimOp && dimOp.getSource() == v) return true;
    if (auto empty = v.getDefiningOp<tensor::EmptyOp>())
      return dyn_cast<Value>(empty.getMixedSizes()[dim]) == size;
    if (auto expand = v.getDefiningOp<tensor::ExpandShapeOp>())
      return dyn_cast<Value>(expand.getMixedOutputShape()[dim]) == size;
    auto linalgOp = v.getDefiningOp<linalg::LinalgOp>();
    if (!linalgOp || linalgOp->getNumResults() != 1 ||
        linalgOp.getNumDpsInits() != 1)
      return false;
    v = linalgOp.getDpsInitOperand(0)->get();
  }
}

/// Folds TosaToLinalg's rank >= 3 `scf.if (dim == 1)` broadcast when the
/// `then` branch broadcasts a dimension of size `dim` to an output
/// `tensor.empty` of the same dynamic size `dim` (making `dim == 1` an
/// identity copy).
struct FoldIdentityBroadcastIf : OpRewritePattern<scf::IfOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(scf::IfOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1 || !op.elseBlock()) return failure();
    auto cmp = op.getCondition().getDefiningOp<arith::CmpIOp>();
    if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::eq)
      return failure();
    Value dimVal = isConstantIntValue(cmp.getRhs(), 1)   ? cmp.getLhs()
                   : isConstantIntValue(cmp.getLhs(), 1) ? cmp.getRhs()
                                                         : Value();
    if (!dimVal) return failure();

    Value src = op.elseYield().getOperand(0);
    auto generic =
        op.thenYield().getOperand(0).getDefiningOp<linalg::GenericOp>();
    if (!generic || generic.getNumDpsInputs() != 1 ||
        generic.getNumDpsInits() != 1 ||
        generic.getDpsInputOperand(0)->get() != src)
      return failure();

    auto empty =
        generic.getDpsInitOperand(0)->get().getDefiningOp<tensor::EmptyOp>();
    // Leave rank < 3 untouched so 2D multi-reduction TOSA softmax dispatches
    // do not fuse across broadcast guards past the 32 KiB stack limit.
    if (!empty || empty.getType().getRank() < 3 ||
        empty.getType() != src.getType())
      return failure();

    Block *body = generic.getBlock();
    if (!llvm::hasSingleElement(*body) ||
        body->getTerminator()->getOperands() !=
            ValueRange{body->getArgument(0)})
      return failure();

    SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
    if (!maps[1].isIdentity()) return failure();

    SmallVector<OpFoldResult> mixedSizes = empty.getMixedSizes();
    for (auto [i, expr] : llvm::enumerate(maps[0].getResults())) {
      if (expr == rewriter.getAffineDimExpr(i)) continue;
      if (expr != rewriter.getAffineConstantExpr(0) ||
          dyn_cast<Value>(mixedSizes[i]) != dimVal ||
          !hasDimSize(src, i, dimVal))
        return failure();
    }

    rewriter.replaceOp(op, src);
    return success();
  }
};

/// Folds `tensor.dim` of `tensor.expand_shape` to its dynamic output size
/// operand so `FoldIdentityBroadcastIf` can match `dim == 1`.
struct DimOfExpandShapeOp : OpRewritePattern<tensor::DimOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(tensor::DimOp op,
                                PatternRewriter &rewriter) const override {
    auto expand = op.getSource().getDefiningOp<tensor::ExpandShapeOp>();
    std::optional<int64_t> idx = op.getConstantIndex();
    if (!expand || !idx || *idx < 0 || *idx >= expand.getResultType().getRank())
      return failure();
    Value dim = dyn_cast<Value>(expand.getMixedOutputShape()[*idx]);
    if (!dim) return failure();
    rewriter.replaceOp(op, dim);
    return success();
  }
};

struct CoralNPURaiseApplyScalePass final
    : impl::CoralNPURaiseApplyScaleBase<CoralNPURaiseApplyScalePass> {
  void runOnOperation() override {
    // Modules targeting only host devices keep the upstream lowering.
    if (targetsOnlyHost(getOperation())) return;
    RewritePatternSet patterns(&getContext());
    patterns.add<RaiseApplyScale, FoldIdentityBroadcastIf, DimOfExpandShapeOp>(
        &getContext());
    tensor::DimOp::getCanonicalizationPatterns(patterns, &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<Pass> createCoralNPURaiseApplyScalePass() {
  return std::make_unique<CoralNPURaiseApplyScalePass>();
}

}  // namespace mlir::coralnpu_compiler
