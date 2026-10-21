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

#include "compiler/Transforms/Passes.h"

// MLIR:
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::coralnpu_compiler {

#define GEN_PASS_DEF_CORALNPURESTORECONVINPUTCHANNEL
#include "compiler/Transforms/Passes.h.inc"

namespace {

/// Detects a channel-last convolution whose unit input channel dimension was
/// stripped by IREE's unit dim folding, e.g. the first convolution of a model
/// with a single-channel image or audio input:
///   image (N, IH, IW) * filter (KH, KW, F) -> output (N, OH, OW, F).
bool isChannellessConv(linalg::GenericOp op) {
  if (!op.hasPureTensorSemantics() || op->hasAttr("lowering_config") ||
      op.getNumDpsInputs() != 2 || !linalg::isaConvolutionOpInterface(op) ||
      llvm::any_of(op->getOperands(), [](Value v) {
        auto rType = dyn_cast<RankedTensorType>(v.getType());
        return !rType || rType.getEncoding() != nullptr;
      }))
    return false;
  FailureOr<linalg::ConvolutionDimensions> maybeConvDims =
      linalg::inferConvolutionDims(op);
  if (failed(maybeConvDims) || !maybeConvDims->inputChannel.empty() ||
      !maybeConvDims->depth.empty() || maybeConvDims->outputChannel.size() != 1)
    return false;
  // The filter holds the reduction (filter window) dims followed by the
  // output channel, which is also the innermost output dim.
  AffineExpr outputChannel =
      getAffineDimExpr(maybeConvDims->outputChannel.front(), op.getContext());
  AffineMap filterMap = op.getMatchingIndexingMap(op.getDpsInputOperand(1));
  AffineMap outputMap = op.getMatchingIndexingMap(op.getDpsInitOperand(0));
  int64_t numReductions = op.getNumReductionLoops();
  return numReductions > 0 && filterMap.getNumResults() == numReductions + 1 &&
         filterMap.getResults().back() == outputChannel &&
         outputMap.getResults().back() == outputChannel;
}

/// Expands a tensor with a unit dimension at position `pos` > 0, grouped with
/// the dimension before it.
Value insertUnitDimToTensor(RewriterBase &rewriter, Location loc, Value tensor,
                            int64_t pos) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  SmallVector<int64_t> newShape(tensorType.getShape());
  newShape.insert(newShape.begin() + pos, 1);
  SmallVector<ReassociationIndices> reassoc;
  for (int64_t i = 0; i < tensorType.getRank(); ++i)
    reassoc.push_back(i == pos - 1 ? ReassociationIndices{i, i + 1}
                                   : ReassociationIndices{i < pos ? i : i + 1});
  auto newType = RankedTensorType::get(newShape, tensorType.getElementType());
  return tensor::ExpandShapeOp::create(rewriter, loc, newType, tensor, reassoc);
}

/// Restores the unit input channel dimension of a channel-less convolution
/// (see isChannellessConv) as the innermost image dim and the filter dim
/// before the output channel, which makes it a regular convolution again:
///   image (N, IH, IW, 1) * filter (KH, KW, 1, F) -> (N, OH, OW, F).
void insertUnitInputChannelDimension(RewriterBase &rewriter,
                                     linalg::GenericOp op) {
  Location loc = op.getLoc();
  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  Value newImage = insertUnitDimToTensor(
      rewriter, loc, op.getDpsInputOperand(0)->get(), maps[0].getNumResults());
  Value newFilter =
      insertUnitDimToTensor(rewriter, loc, op.getDpsInputOperand(1)->get(),
                            maps[1].getNumResults() - 1);

  // The input channel becomes the new last loop.
  unsigned numLoops = op.getNumLoops();
  AffineExpr inputChannel = getAffineDimExpr(numLoops, rewriter.getContext());
  for (AffineMap &m : maps) m = m.shiftDims(/*shift=*/1, /*offset=*/numLoops);
  maps[0] = maps[0].insertResult(inputChannel, maps[0].getNumResults());
  maps[1] = maps[1].insertResult(inputChannel, maps[1].getNumResults() - 1);
  SmallVector<utils::IteratorType> newIterTypes = op.getIteratorTypesArray();
  newIterTypes.push_back(utils::IteratorType::reduction);

  auto newConvOp = linalg::GenericOp::create(
      rewriter, loc, op.getResultTypes(), ValueRange{newImage, newFilter},
      op.getDpsInits(), maps, newIterTypes);
  newConvOp->setDiscardableAttrs(op->getDiscardableAttrDictionary());
  newConvOp->removeAttr("linalg.memoized_indexing_maps");
  rewriter.inlineRegionBefore(op.getRegion(), newConvOp.getRegion(),
                              newConvOp.getRegion().begin());
  rewriter.replaceOp(op, newConvOp.getResults());
}

struct CoralNPURestoreConvInputChannelPass final
    : impl::CoralNPURestoreConvInputChannelBase<
          CoralNPURestoreConvInputChannelPass> {
  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    getOperation()->walk([&](linalg::GenericOp op) {
      if (!isChannellessConv(op)) return;
      rewriter.setInsertionPoint(op);
      insertUnitInputChannelDimension(rewriter, op);
    });
  }
};

}  // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createCoralNPURestoreConvInputChannelPass() {
  return std::make_unique<CoralNPURestoreConvInputChannelPass>();
}

}  // namespace mlir::coralnpu_compiler
