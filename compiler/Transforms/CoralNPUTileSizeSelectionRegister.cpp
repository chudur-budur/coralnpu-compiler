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
//
//===----------------------------------------------------------------------===//
// CoralNPU Register Tile Size Selection Pass
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <memory>
#include <numeric>
#include <utility>

#include "compiler/Transforms/CoralNPUTileSizeSelectionAnalysis.h"
#include "compiler/Transforms/CoralNPUTileSizeSelectionUtils.h"
#include "compiler/Transforms/Passes.h"

// IREE headers
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUDialect.h"
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"

// MLIR headers
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"

// LLVM headers
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#define DEBUG_TYPE "coralnpu-tile-size-selection"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace mlir::coralnpu_compiler {

#define GEN_PASS_DEF_CORALNPUTILESIZESELECTIONREGISTER
#include "compiler/Transforms/Passes.h.inc"

namespace {

// Returns the largest exact divisor of `range` in [2, maxTile], 1 if there is
// none, or `maxTile` for a dynamic range.
int64_t pickTileDivisor(int64_t range, int64_t maxTile) {
  if (range <= 0) return maxTile;
  for (int64_t t = std::min(range, maxTile); t >= 2; --t) {
    if (range % t == 0) return t;
  }
  return 1;
}

bool hasTranscendentalBody(linalg::LinalgOp linalgOp) {
  return llvm::any_of(*linalgOp.getBlock(), [](Operation &op) {
    return isa<math::ExpOp, math::Exp2Op, math::LogOp, math::Log2Op,
               math::Log10Op, math::Log1pOp, math::TanhOp, math::ErfOp,
               math::PowFOp, math::SinOp, math::CosOp, math::TanOp,
               math::AsinOp, math::AcosOp, math::AtanOp, math::Atan2Op,
               math::SinhOp, math::CoshOp>(&op);
  });
}

bool hasTranscendentalInBlock(Operation *rootOp) {
  return llvm::any_of(rootOp->getBlock()->getOps<linalg::LinalgOp>(),
                      hasTranscendentalBody);
}

// Peak non-constant live values in `op`'s body (+2 at each transcendental op).
int64_t peakLiveValues(linalg::LinalgOp op) {
  Block &body = *op.getBlock();
  DenseMap<Value, Operation *> lastUse;
  for (Operation &o : body)
    for (Value v : o.getOperands()) lastUse[v] = &o;
  auto counted = [&](Value v) {
    Operation *def = v.getDefiningOp();
    return lastUse.contains(v) && v.getParentBlock() == &body &&
           !(def && def->hasTrait<OpTrait::ConstantLike>());
  };
  llvm::SmallDenseSet<Value> loaded;
  int64_t live = 0, peak = 0;
  for (Operation &o : body) {
    llvm::SmallDenseSet<Value> operands(o.operand_begin(), o.operand_end());
    for (Value v : operands)
      if (isa<BlockArgument>(v) && loaded.insert(v).second) live += counted(v);
    for (Value r : o.getResults()) live += counted(r);
    bool transcendental =
        isa<math::MathDialect>(o.getDialect()) &&
        !isa<math::AbsFOp, math::AbsIOp, math::FmaOp, math::CopySignOp>(o);
    peak = std::max(peak, live + (transcendental ? 2 : 0));
    for (Value v : operands) live -= counted(v) && lastUse.lookup(v) == &o;
  }
  return peak;
}

int64_t peakLiveValuesInBlock(Operation *rootOp) {
  int64_t peak = 0;
  for (auto linalgOp : rootOp->getBlock()->getOps<linalg::LinalgOp>())
    peak = std::max(peak, peakLiveValues(linalgOp));
  return peak;
}

// tiling loops: [N, OH, OW, OC, KH, KW, IC]
void setConv2DNhwcHwcfVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::CPU::LoweringPipeline &pipeline) {
  pipeline = IREE::CPU::LoweringPipeline::ConvTileAndDecomposeExpert;

  const auto &loops = analysis.parallelLoops;
  int64_t ocRange = analysis.staticLoopRanges[loops[0]];
  int64_t owRange = analysis.staticLoopRanges[loops[1]];
  auto dims = linalg::inferConvolutionDims(op);
  bool unitStrideW = succeeded(dims) && dims->strides.back() == 1;
  bool wideOw = unitStrideW && pickTileDivisor(owRange, 4) == 1 &&
                pickTileDivisor(owRange, 7) > 1 &&
                ocRange % (4 * vectorWidth) == 0;

  int64_t ocTile =
      (wideOw || hasTranscendentalInBlock(op) ? 4 : 8) * vectorWidth;
  int64_t owTile =
      unitStrideW && ocRange % vectorWidth == 0 ? (wideOw ? 7 : 4) : 1;

  vectorParallelSizes[loops[0]] = ocTile;
  vectorParallelSizes[loops[1]] = owTile;
}

// tiling loops: [N, OH, OW, C, KH, KW]
void setPoolingNhwcVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::CPU::LoweringPipeline &pipeline) {
  pipeline = IREE::CPU::LoweringPipeline::ConvTileAndDecomposeExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t owRange = analysis.staticLoopRanges[loops[1]];
  bool wideOw = pickTileDivisor(owRange, 4) == 1 &&
                pickTileDivisor(owRange, 7) > 1 &&
                analysis.staticLoopRanges[loops[0]] % (4 * vectorWidth) == 0;

  int64_t cTile = (wideOw ? 4 : 8) * vectorWidth;
  int64_t owTile = analysis.staticLoopRanges[loops[0]] % vectorWidth == 0
                       ? (wideOw ? 7 : 4)
                       : 1;

  vectorParallelSizes[loops[0]] = cTile;
  vectorParallelSizes[loops[1]] = owTile;
}

// tiling loops: [N, OH, OW, C, KH, KW]
void setDepthwiseConv2DNhwcHwcVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::CPU::LoweringPipeline &pipeline) {
  pipeline = IREE::CPU::LoweringPipeline::ConvTileAndDecomposeExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t cRange = analysis.staticLoopRanges[loops[0]];
  // Budget 1/4 of vector registers for accumulators across owTile x cTile.
  int64_t kAccRegs = std::max<int64_t>(1, analysis.numVectorRegisters / 4);
  int64_t cTile = 4 * vectorWidth;
  int64_t owTile = 1;
  if (cRange > 0 && cRange % vectorWidth == 0) {
    int64_t cRegs = llvm::divideCeil(std::min(cTile, cRange), vectorWidth);
    owTile = std::clamp<int64_t>(kAccRegs / cRegs, 1, 8);
  }

  vectorParallelSizes[loops[0]] = cTile;
  vectorParallelSizes[loops[1]] = owTile;
}

// tiling loops: [[N,] OW, OC, KW, IC]
void setConv1DNwcWcfVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::CPU::LoweringPipeline &pipeline) {
  setConv2DNhwcHwcfVectorSizes(op, vectorWidth, analysis, vectorParallelSizes,
                               pipeline);
}

// tiling loops: [[N,] OC, OW, IC, KW]
void setConv1DNcwFcwVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::CPU::LoweringPipeline &pipeline) {
  pipeline = IREE::CPU::LoweringPipeline::DoubleTilingExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t owTile = vectorWidth;
  int64_t ocTile =
      analysis.staticLoopRanges[loops[0]] % vectorWidth == 0 ? 4 : 1;

  vectorParallelSizes[loops[0]] = owTile;
  vectorParallelSizes[loops[1]] = ocTile;
}

// tiling loops: [[N,] OC, OH, OW, IC, KH, KW]
void setConv2DNchwFchwVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::CPU::LoweringPipeline &pipeline) {
  pipeline = IREE::CPU::LoweringPipeline::DoubleTilingExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t owTile = 4 * vectorWidth;

  vectorParallelSizes[loops[0]] = owTile;
}

// tiling loops: [[N,] OH, OW, OC, KH, KW, IC]
void setConv2DNhwcFhwcVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::CPU::LoweringPipeline &pipeline) {
  pipeline = IREE::CPU::LoweringPipeline::DoubleTilingExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t ocTile = 4 * vectorWidth;
  int64_t owTile =
      analysis.staticLoopRanges[loops[0]] % vectorWidth == 0 ? 4 : 1;

  vectorParallelSizes[loops[0]] = ocTile;
  vectorParallelSizes[loops[1]] = owTile;
}

// tiling loops: [[N,] OH, OW, G, OC, KH, KW, C]
void setConv2DNhwgcGfhwcVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::CPU::LoweringPipeline &pipeline) {
  pipeline = IREE::CPU::LoweringPipeline::ConvTileAndDecomposeExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t ocTile = 8 * vectorWidth;

  vectorParallelSizes[loops[0]] = ocTile;
}

// tiling loops: [[N,] G, OC, OH, OW, C, KH, KW]
void setConv2DNgchwGfchwVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::CPU::LoweringPipeline &pipeline) {
  pipeline = IREE::CPU::LoweringPipeline::DoubleTilingExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t owTile = 8 * vectorWidth;

  vectorParallelSizes[loops[0]] = owTile;
}

// tiling loops: [[N,] G, OC, OH, OW, C, KH, KW]
void setConv2DNgchwFgchwVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::CPU::LoweringPipeline &pipeline) {
  pipeline = IREE::CPU::LoweringPipeline::DoubleTilingExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t owTile = 8 * vectorWidth;

  vectorParallelSizes[loops[0]] = owTile;
}

// tiling loops: [OD, OH, OW, KD, KH, KW]
void setConv3DVectorSizes(linalg::LinalgOp op, int64_t vectorWidth,
                          const CoralNPUTileSizeSelectionAnalysis &analysis,
                          MutableArrayRef<int64_t> vectorParallelSizes,
                          IREE::CPU::LoweringPipeline &pipeline) {
  pipeline = IREE::CPU::LoweringPipeline::DoubleTilingExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t owTile = 8 * vectorWidth;

  vectorParallelSizes[loops[0]] = owTile;
}

// tiling loops: [[N,] OD, OH, OW, OC, KD, KH, KW, IC]
void setConv3DNdhwcDhwcfVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::CPU::LoweringPipeline &pipeline) {
  pipeline = IREE::CPU::LoweringPipeline::ConvTileAndDecomposeExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t ocTile = 4 * vectorWidth;
  int64_t owTile =
      analysis.staticLoopRanges[loops[0]] % vectorWidth == 0 ? 4 : 1;

  vectorParallelSizes[loops[0]] = ocTile;
  vectorParallelSizes[loops[1]] = owTile;
}

// tiling loops: [[N,] OC, OD, OH, OW, IC, KD, KH, KW]
void setConv3DNcdhwFcdhwVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::CPU::LoweringPipeline &pipeline) {
  pipeline = IREE::CPU::LoweringPipeline::DoubleTilingExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t ocTile = 4 * vectorWidth;
  int64_t owTile =
      analysis.staticLoopRanges[loops[0]] % vectorWidth == 0 ? 4 : 1;

  vectorParallelSizes[loops[0]] = ocTile;
  vectorParallelSizes[loops[1]] = owTile;
}

// tiling loops: [[N,] OD, OH, OW, C, KD, KH, KW]
void setDepthwiseConv3DNdhwcDhwcVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::CPU::LoweringPipeline &pipeline) {
  pipeline = IREE::CPU::LoweringPipeline::ConvTileAndDecomposeExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t cTile = 4 * vectorWidth;
  int64_t owTile = 2 * vectorWidth;

  vectorParallelSizes[loops[0]] = cTile;
  vectorParallelSizes[loops[1]] = owTile;
}

// tiling loops: [[N,] OD, OH, OW, C, M, KD, KH, KW]
void setDepthwiseConv3DNdhwcDhwcmVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::CPU::LoweringPipeline &pipeline) {
  pipeline = IREE::CPU::LoweringPipeline::DoubleTilingExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t mTile = 4 * vectorWidth;
  int64_t cTile = 2 * vectorWidth;

  vectorParallelSizes[loops[0]] = mTile;
  vectorParallelSizes[loops[1]] = cTile;
}

// tiling loops: [[N,] OD, OH, OW, C, KD, KH, KW]
void setPoolingNdhwcVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::CPU::LoweringPipeline &pipeline) {
  pipeline = IREE::CPU::LoweringPipeline::ConvTileAndDecomposeExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t cTile = vectorWidth;
  int64_t owTile = vectorWidth;

  vectorParallelSizes[loops[0]] = cTile;
  vectorParallelSizes[loops[1]] = owTile;
}

// tiling loops: [OW, KW]
void setConv1DVectorSizes(linalg::LinalgOp op, int64_t vectorWidth,
                          const CoralNPUTileSizeSelectionAnalysis &analysis,
                          MutableArrayRef<int64_t> vectorParallelSizes,
                          IREE::CPU::LoweringPipeline &pipeline) {
  pipeline = IREE::CPU::LoweringPipeline::DoubleTilingExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t owTile = 4 * vectorWidth;

  vectorParallelSizes[loops[0]] = owTile;
}

// tiling loops: [OH, OW, KH, KW]
void setConv2DVectorSizes(linalg::LinalgOp op, int64_t vectorWidth,
                          const CoralNPUTileSizeSelectionAnalysis &analysis,
                          MutableArrayRef<int64_t> vectorParallelSizes,
                          IREE::CPU::LoweringPipeline &pipeline) {
  pipeline = IREE::CPU::LoweringPipeline::DoubleTilingExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t owTile = 4 * vectorWidth;

  vectorParallelSizes[loops[0]] = owTile;
}

// tiling loops: [[N,] OW, C, KW]
void setDepthwiseConv1DNwcWcVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::CPU::LoweringPipeline &pipeline) {
  pipeline = IREE::CPU::LoweringPipeline::ConvTileAndDecomposeExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t cTile = 4 * vectorWidth;
  int64_t owTile =
      analysis.staticLoopRanges[loops[0]] % vectorWidth == 0 ? 2 : 1;

  vectorParallelSizes[loops[0]] = cTile;
  vectorParallelSizes[loops[1]] = owTile;
}

// tiling loops: [[N,] C, OW, KW]
void setDepthwiseConv1DNcwCwVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::CPU::LoweringPipeline &pipeline) {
  pipeline = IREE::CPU::LoweringPipeline::ConvTileAndDecomposeExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t owTile = 4 * vectorWidth;
  int64_t cTile =
      analysis.staticLoopRanges[loops[0]] % vectorWidth == 0 ? 2 : 1;

  vectorParallelSizes[loops[0]] = owTile;
  vectorParallelSizes[loops[1]] = cTile;
}

// tiling loops: [[N,] OW, C, M, KW]
void setDepthwiseConv1DNwcWcmVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::CPU::LoweringPipeline &pipeline) {
  pipeline = IREE::CPU::LoweringPipeline::DoubleTilingExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t mTile = 4 * vectorWidth;
  int64_t cTile =
      analysis.staticLoopRanges[loops[0]] % vectorWidth == 0 ? 2 : 1;

  vectorParallelSizes[loops[0]] = mTile;
  vectorParallelSizes[loops[1]] = cTile;
}

// tiling loops: [[N,] OH, OW, C, M, KH, KW]
void setDepthwiseConv2DNhwcHwcmVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::CPU::LoweringPipeline &pipeline) {
  pipeline = IREE::CPU::LoweringPipeline::DoubleTilingExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t mTile = 4 * vectorWidth;
  int64_t cTile =
      analysis.staticLoopRanges[loops[0]] % vectorWidth == 0 ? 2 : 1;

  vectorParallelSizes[loops[0]] = mTile;
  vectorParallelSizes[loops[1]] = cTile;
}

// tiling loops: [[N,] OW, C, KW]
void setPoolingNwcVectorSizes(linalg::LinalgOp op, int64_t vectorWidth,
                              const CoralNPUTileSizeSelectionAnalysis &analysis,
                              MutableArrayRef<int64_t> vectorParallelSizes,
                              IREE::CPU::LoweringPipeline &pipeline) {
  setPoolingNhwcVectorSizes(op, vectorWidth, analysis, vectorParallelSizes,
                            pipeline);
}

// tiling loops: [[N,] C, OW, KW]
void setPoolingNcwVectorSizes(linalg::LinalgOp op, int64_t vectorWidth,
                              const CoralNPUTileSizeSelectionAnalysis &analysis,
                              MutableArrayRef<int64_t> vectorParallelSizes,
                              IREE::CPU::LoweringPipeline &pipeline) {
  pipeline = IREE::CPU::LoweringPipeline::DoubleTilingExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t owTile = 4 * vectorWidth;

  vectorParallelSizes[loops[0]] = owTile;
}

// tiling loops: [[N,] C, OH, OW, KH, KW]
void setPoolingNchwVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::CPU::LoweringPipeline &pipeline) {
  pipeline = IREE::CPU::LoweringPipeline::DoubleTilingExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t owTile = 4 * vectorWidth;

  vectorParallelSizes[loops[0]] = owTile;
}

void setTransposeVectorSizes(linalg::LinalgOp op, int64_t vectorWidth,
                             const CoralNPUTileSizeSelectionAnalysis &analysis,
                             MutableArrayRef<int64_t> vectorParallelSizes,
                             IREE::CPU::LoweringPipeline &pipeline) {
  pipeline = IREE::CPU::LoweringPipeline::DoubleTilingExpert;

  // Tile a moved innermost dim as 8x8 so vector.transpose lowers to shuffles.
  auto maps = op.getIndexingMapsArray();
  unsigned in = maps[0].getDimPosition(maps[0].getNumResults() - 1);
  unsigned out = maps[1].getDimPosition(maps[1].getNumResults() - 1);
  if (in != out) {
    vectorParallelSizes[in] = vectorParallelSizes[out] = 8;
    return;
  }
  const auto &loops = analysis.parallelLoops;
  int64_t tile = 4 * vectorWidth;
  vectorParallelSizes[loops[0]] = tile;
}

void setBroadcastVectorSizes(linalg::LinalgOp op, int64_t vectorWidth,
                             const CoralNPUTileSizeSelectionAnalysis &analysis,
                             MutableArrayRef<int64_t> vectorParallelSizes,
                             IREE::CPU::LoweringPipeline &pipeline) {
  pipeline = IREE::CPU::LoweringPipeline::DoubleTilingExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t tile = analysis.staticLoopRanges[loops[0]] % vectorWidth == 0
                     ? 8 * vectorWidth
                     : 4 * vectorWidth;

  vectorParallelSizes[loops[0]] = tile;
}

void setReduceVectorSizes(linalg::LinalgOp op, int64_t vectorWidth,
                          const CoralNPUTileSizeSelectionAnalysis &analysis,
                          MutableArrayRef<int64_t> vectorParallelSizes,
                          MutableArrayRef<int64_t> vectorReductionSizes,
                          IREE::CPU::LoweringPipeline &pipeline) {
  pipeline = IREE::CPU::LoweringPipeline::DoubleTilingExpert;

  // Row reductions vectorize along the contiguous reduction dim.
  AffineMap inMap = op.getMatchingIndexingMap(op.getDpsInputOperand(0));
  bool rowReduction =
      inMap.getNumResults() > 0 &&
      llvm::is_contained(analysis.reductionLoops,
                         inMap.getDimPosition(inMap.getNumResults() - 1));
  if (!rowReduction && !analysis.parallelLoops.empty()) {
    const auto &loops = analysis.parallelLoops;

    int64_t tile = 4 * vectorWidth;
    if (analysis.staticLoopRanges[loops[0]] % vectorWidth == 0) {
      tile = 8 * vectorWidth;
    }

    vectorParallelSizes[loops[0]] = tile;
    return;
  }

  if (!analysis.reductionLoops.empty()) {
    const auto &loops = analysis.reductionLoops;
    Type inTy = getElementTypeOrSelf(op.getDpsInputOperand(0)->get());
    Type outTy = getElementTypeOrSelf(op.getDpsInitOperand(0)->get());
    if (inTy.isIntOrFloat() && outTy.isIntOrFloat() &&
        inTy.getIntOrFloatBitWidth() >= 8 &&
        outTy.getIntOrFloatBitWidth() > inTy.getIntOrFloatBitWidth()) {
      vectorWidth *=
          outTy.getIntOrFloatBitWidth() / inTy.getIntOrFloatBitWidth();
    }
    vectorReductionSizes[loops[0]] = 4 * vectorWidth;
  }
}

// tiling loops: [[B,] M, N, K]
void setMatmulVectorSizes(linalg::LinalgOp op, int64_t vectorWidth,
                          const CoralNPUTileSizeSelectionAnalysis &analysis,
                          MutableArrayRef<int64_t> vectorParallelSizes,
                          IREE::CPU::LoweringPipeline &pipeline) {
  pipeline = IREE::CPU::LoweringPipeline::DoubleTilingExpert;

  bool isMixedPrecision = false;
  if (op.getNumDpsInputs() >= 2 && op.getNumDpsInits() >= 1) {
    Type inType = op.getDpsInputOperand(0)->get().getType();
    Type outType = op.getDpsInitOperand(0)->get().getType();
    if (auto inShaped = dyn_cast<ShapedType>(inType)) {
      if (auto outShaped = dyn_cast<ShapedType>(outType)) {
        if (outShaped.getElementTypeBitWidth() >
            inShaped.getElementTypeBitWidth()) {
          isMixedPrecision = true;
        }
      }
    }
  }

  const auto &loops = analysis.parallelLoops;
  int64_t mTile = 4;
  int64_t nTile = std::min<int64_t>(16, 2 * vectorWidth);

  bool isBatch = loops.size() >= 3 ||
                 isa<linalg::BatchMatmulOp, linalg::QuantizedBatchMatmulOp,
                     linalg::BatchReduceMatmulOp>(op);

  if (isa<linalg::QuantizedMatmulOp, linalg::QuantizedBatchMatmulOp>(op) ||
      isMixedPrecision) {
    if (getElementTypeOrSelf(op.getDpsInputOperand(0)->get())
            .getIntOrFloatBitWidth() == 16) {
      mTile = 1;
      nTile = 8 * vectorWidth;
    } else {
      nTile = 8;
    }
  } else if (vectorWidth <= 4 && !isBatch) {
    mTile = pickTileDivisor(analysis.staticLoopRanges[loops[1]], 8);
    nTile = (mTile <= 4 ? 4 : 2) * vectorWidth;
  }

  vectorParallelSizes[loops[0]] = nTile;
  vectorParallelSizes[loops[1]] = mTile;
}

// tiling loops: [[B,] M, K]
void setMatvecVectorSizes(linalg::LinalgOp op, int64_t vectorWidth,
                          const CoralNPUTileSizeSelectionAnalysis &analysis,
                          MutableArrayRef<int64_t> vectorParallelSizes,
                          IREE::CPU::LoweringPipeline &pipeline) {
  pipeline = IREE::CPU::LoweringPipeline::DoubleTilingExpert;

  Type outTy = getElementTypeOrSelf(op.getDpsInitOperand(0)->get());
  int64_t mTile = (outTy.isF32() ? 8 : 4) * vectorWidth;

  const auto &loops = analysis.parallelLoops;

  vectorParallelSizes[loops[0]] = mTile;
}

// tiling loops: [[B,] N, K]
void setVecmatVectorSizes(linalg::LinalgOp op, int64_t vectorWidth,
                          const CoralNPUTileSizeSelectionAnalysis &analysis,
                          MutableArrayRef<int64_t> vectorParallelSizes,
                          IREE::CPU::LoweringPipeline &pipeline) {
  pipeline = IREE::CPU::LoweringPipeline::DoubleTilingExpert;

  Type outTy = getElementTypeOrSelf(op.getDpsInitOperand(0)->get());
  int64_t nTile = (outTy.isF32() ? 8 : 4) * vectorWidth;

  const auto &loops = analysis.parallelLoops;

  vectorParallelSizes[loops[0]] = nTile;
}

void setDotVectorSizes(linalg::LinalgOp op, int64_t vectorWidth,
                       const CoralNPUTileSizeSelectionAnalysis &analysis,
                       MutableArrayRef<int64_t> vectorParallelSizes,
                       IREE::CPU::LoweringPipeline &pipeline) {
  pipeline = IREE::CPU::LoweringPipeline::DoubleTilingExpert;
}

// tiling loops: [[B,] M1, N1, K1, M0, N0, K0]
void setMmt4DVectorSizes(linalg::LinalgOp op, int64_t vectorWidth,
                         const CoralNPUTileSizeSelectionAnalysis &analysis,
                         MutableArrayRef<int64_t> vectorParallelSizes,
                         IREE::CPU::LoweringPipeline &pipeline) {
  pipeline = IREE::CPU::LoweringPipeline::DoubleTilingExpert;
}

void setContractVectorSizes(linalg::LinalgOp op, int64_t vectorWidth,
                            const CoralNPUTileSizeSelectionAnalysis &analysis,
                            MutableArrayRef<int64_t> vectorParallelSizes,
                            IREE::CPU::LoweringPipeline &pipeline) {
  pipeline = IREE::CPU::LoweringPipeline::DoubleTilingExpert;

  auto dims = linalg::inferContractionDims(op);
  if (failed(dims)) return;

  if (!dims->m.empty() && !dims->n.empty() && !dims->k.empty() &&
      !(dims->m.size() == 2 && dims->n.size() == 2 && dims->k.size() == 2)) {
    return setMatmulVectorSizes(op, vectorWidth, analysis, vectorParallelSizes,
                                pipeline);
  }
  // The innermost m/n dim is vectorized.
  if (dims->batch.size() <= 1 && !dims->m.empty() && dims->n.empty() &&
      dims->k.size() == 1) {
    return setMatvecVectorSizes(op, vectorWidth, analysis, vectorParallelSizes,
                                pipeline);
  }
  if (dims->batch.size() <= 1 && dims->m.empty() && !dims->n.empty() &&
      dims->k.size() == 1) {
    return setVecmatVectorSizes(op, vectorWidth, analysis, vectorParallelSizes,
                                pipeline);
  }
  if (dims->batch.empty() && dims->m.empty() && dims->n.empty() &&
      dims->k.size() == 1) {
    return setDotVectorSizes(op, vectorWidth, analysis, vectorParallelSizes,
                             pipeline);
  }
  if (dims->batch.size() <= 1 && dims->m.size() == 2 && dims->n.size() == 2 &&
      dims->k.size() == 2) {
    return setMmt4DVectorSizes(op, vectorWidth, analysis, vectorParallelSizes,
                               pipeline);
  }
}

int64_t getElementwiseTileSize(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis) {
  size_t innerDim = analysis.parallelLoops[0];
  AffineExpr inner = getAffineDimExpr(innerDim, op.getContext());
  // Allow LMUL=8 tiles when peak live values fit in 3/4 of vector registers.
  bool wide =
      analysis.staticLoopRanges[innerDim] % (8 * vectorWidth) == 0 &&
      peakLiveValuesInBlock(op) * 8 <= (analysis.numVectorRegisters * 3) / 4 &&
      llvm::all_of(op->getOperandTypes(),
                   [&](Type t) {
                     Type el = getElementTypeOrSelf(t);
                     return el.isIntOrFloat() && el.getIntOrFloatBitWidth() <=
                                                     analysis.elemSizeBytes * 8;
                   }) &&
      llvm::all_of(op.getIndexingMapsArray(), [&](AffineMap m) {
        return m.isProjectedPermutation() &&
               (!llvm::is_contained(m.getResults(), inner) ||
                m.getResults().back() == inner);
      });
  return (wide ? 8 : 4) * vectorWidth;
}

void setElementwiseArithBinaryVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::CPU::LoweringPipeline &pipeline) {
  pipeline = IREE::CPU::LoweringPipeline::DoubleTilingExpert;

  if (!analysis.parallelLoops.empty()) {
    const auto &loops = analysis.parallelLoops;
    vectorParallelSizes[loops[0]] =
        getElementwiseTileSize(op, vectorWidth, analysis);
  }
}

void setElementwiseDivBinaryVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::CPU::LoweringPipeline &pipeline) {
  pipeline = IREE::CPU::LoweringPipeline::DoubleTilingExpert;

  if (!analysis.parallelLoops.empty()) {
    const auto &loops = analysis.parallelLoops;
    int64_t tile = 4 * vectorWidth;
    vectorParallelSizes[loops[0]] = tile;
  }
}

void setElementwiseUnaryArithVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::CPU::LoweringPipeline &pipeline) {
  pipeline = IREE::CPU::LoweringPipeline::DoubleTilingExpert;

  if (!analysis.parallelLoops.empty()) {
    const auto &loops = analysis.parallelLoops;
    vectorParallelSizes[loops[0]] =
        getElementwiseTileSize(op, vectorWidth, analysis);
  }
}

void setElementwiseRoundingVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::CPU::LoweringPipeline &pipeline) {
  pipeline = IREE::CPU::LoweringPipeline::DoubleTilingExpert;

  if (!analysis.parallelLoops.empty()) {
    const auto &loops = analysis.parallelLoops;
    int64_t tile = 4 * vectorWidth;
    vectorParallelSizes[loops[0]] = tile;
  }
}

void setElementwiseRootVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::CPU::LoweringPipeline &pipeline) {
  pipeline = IREE::CPU::LoweringPipeline::DoubleTilingExpert;

  if (!analysis.parallelLoops.empty()) {
    const auto &loops = analysis.parallelLoops;
    int64_t tile = 4 * vectorWidth;
    vectorParallelSizes[loops[0]] = tile;
  }
}

void setElementwiseTranscendentalVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::CPU::LoweringPipeline &pipeline) {
  pipeline = IREE::CPU::LoweringPipeline::DoubleTilingExpert;

  if (!analysis.parallelLoops.empty()) {
    const auto &loops = analysis.parallelLoops;
    int64_t tile = 4 * vectorWidth;
    vectorParallelSizes[loops[0]] = tile;
  }
}

void setMapVectorSizes(linalg::LinalgOp op, int64_t vectorWidth,
                       const CoralNPUTileSizeSelectionAnalysis &analysis,
                       MutableArrayRef<int64_t> vectorParallelSizes,
                       IREE::CPU::LoweringPipeline &pipeline) {
  pipeline = IREE::CPU::LoweringPipeline::DoubleTilingExpert;

  if (!analysis.parallelLoops.empty()) {
    const auto &loops = analysis.parallelLoops;
    vectorParallelSizes[loops[0]] =
        getElementwiseTileSize(op, vectorWidth, analysis);
  }
}

void setGenericVectorSizes(linalg::GenericOp genericOp, int64_t vectorWidth,
                           const CoralNPUTileSizeSelectionAnalysis &analysis,
                           MutableArrayRef<int64_t> vectorParallelSizes,
                           MutableArrayRef<int64_t> vectorReductionSizes,
                           IREE::CPU::LoweringPipeline &pipeline) {
  pipeline = IREE::CPU::LoweringPipeline::DoubleTilingExpert;

  if ((analysis.nonRootTilingOps.empty() || analysis.reductionLoops.empty()) &&
      !analysis.parallelLoops.empty()) {
    const auto &loops = analysis.parallelLoops;
    vectorParallelSizes[loops[0]] =
        getElementwiseTileSize(genericOp, vectorWidth, analysis);
    return;
  }

  if (!analysis.reductionLoops.empty()) {
    const auto &loops = analysis.reductionLoops;
    vectorReductionSizes[loops[0]] = 4 * vectorWidth;
  }
}

bool hasMultiplyAddBody(linalg::GenericOp genericOp) {
  Block *body = genericOp.getBody();
  if (!body) return false;
  bool hasMul = llvm::any_of(body->getOperations(), [](Operation &op) {
    return isa<arith::MulFOp, arith::MulIOp>(&op);
  });
  bool hasAdd = llvm::any_of(body->getOperations(), [](Operation &op) {
    return isa<arith::AddFOp, arith::AddIOp>(&op);
  });
  return hasMul && hasAdd;
}

bool isDepthwiseConv1DNcwCwGenericOp(linalg::GenericOp genericOp) {
  if (linalg::isaConvolutionOpOfType<linalg::DepthwiseConv1DNcwCwOp>(
          genericOp)) {
    return true;
  }
  if (genericOp.getNumDpsInputs() != 2 || genericOp.getNumDpsInits() != 1) {
    return false;
  }
  if (genericOp.getNumReductionLoops() != 1 ||
      genericOp.getNumParallelLoops() > 3) {
    return false;
  }
  if (!hasMultiplyAddBody(genericOp)) return false;

  AffineMap inMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(0));
  AffineMap filterMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(1));
  AffineMap outMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInitOperand(0));

  if (inMap.getNumResults() < 2 || filterMap.getNumResults() != 2) return false;
  bool hasAdd = llvm::any_of(inMap.getResults(), [](AffineExpr expr) {
    return expr.getKind() == AffineExprKind::Add;
  });
  if (!hasAdd) return false;

  if (outMap.getNumResults() == 2 && filterMap.getNumResults() == 2) {
    return outMap.getResult(0) == filterMap.getResult(0);
  }
  return false;
}

bool isDepthwiseConv1DNwcWcGenericOp(linalg::GenericOp genericOp) {
  if (linalg::isaConvolutionOpOfType<linalg::DepthwiseConv1DNwcWcOp>(
          genericOp)) {
    return true;
  }
  if (genericOp.getNumDpsInputs() != 2 || genericOp.getNumDpsInits() != 1) {
    return false;
  }
  if (genericOp.getNumReductionLoops() != 1 ||
      genericOp.getNumParallelLoops() > 3) {
    return false;
  }
  if (!hasMultiplyAddBody(genericOp)) return false;

  AffineMap inMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(0));
  AffineMap filterMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(1));
  AffineMap outMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInitOperand(0));

  if (inMap.getNumResults() < 2 || filterMap.getNumResults() != 2) return false;
  bool hasAdd = llvm::any_of(inMap.getResults(), [](AffineExpr expr) {
    return expr.getKind() == AffineExprKind::Add;
  });
  if (!hasAdd) return false;

  if (outMap.getNumResults() == 2 && filterMap.getNumResults() == 2) {
    return outMap.getResult(1) == filterMap.getResult(1);
  }
  return false;
}

bool isDepthwiseConv3DNdhwcDhwcmGenericOp(linalg::GenericOp genericOp) {
  if (linalg::isaConvolutionOpOfType<linalg::DepthwiseConv3DNdhwcDhwcmOp>(
          genericOp)) {
    return true;
  }
  if (genericOp.getNumDpsInputs() != 2 || genericOp.getNumDpsInits() != 1) {
    return false;
  }
  if (genericOp.getNumReductionLoops() != 3 ||
      genericOp.getNumParallelLoops() > 6) {
    return false;
  }
  if (!hasMultiplyAddBody(genericOp)) return false;

  AffineMap filterMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(1));
  AffineMap outMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInitOperand(0));
  if (filterMap.getNumResults() != 5 || outMap.getNumResults() < 5)
    return false;

  return outMap.getResult(outMap.getNumResults() - 1) ==
             filterMap.getResult(4) &&
         outMap.getResult(outMap.getNumResults() - 2) == filterMap.getResult(3);
}

bool isDepthwiseConv3DNdhwcDhwcGenericOp(linalg::GenericOp genericOp) {
  if (linalg::isaConvolutionOpOfType<linalg::DepthwiseConv3DNdhwcDhwcOp>(
          genericOp)) {
    return true;
  }
  if (genericOp.getNumDpsInputs() != 2 || genericOp.getNumDpsInits() != 1) {
    return false;
  }
  if (genericOp.getNumReductionLoops() != 3 ||
      genericOp.getNumParallelLoops() > 5) {
    return false;
  }
  if (!hasMultiplyAddBody(genericOp)) return false;

  AffineMap inMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(0));
  AffineMap filterMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(1));
  AffineMap outMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInitOperand(0));

  if (filterMap.getNumResults() != 4 || outMap.getNumResults() < 4) {
    return false;
  }
  bool hasAdd = llvm::any_of(inMap.getResults(), [](AffineExpr expr) {
    return expr.getKind() == AffineExprKind::Add;
  });
  if (!hasAdd) return false;

  return outMap.getResult(outMap.getNumResults() - 1) == filterMap.getResult(3);
}

bool isDepthwiseConv2DNhwcHwcmGenericOp(linalg::GenericOp genericOp) {
  if (linalg::isaConvolutionOpOfType<linalg::DepthwiseConv2DNhwcHwcmOp>(
          genericOp) ||
      linalg::isaConvolutionOpOfType<linalg::DepthwiseConv2DNhwcHwcmQOp>(
          genericOp)) {
    return true;
  }
  if (genericOp.getNumDpsInputs() != 2 || genericOp.getNumDpsInits() != 1) {
    return false;
  }
  if (genericOp.getNumReductionLoops() != 2 ||
      genericOp.getNumParallelLoops() > 5) {
    return false;
  }
  if (!hasMultiplyAddBody(genericOp)) return false;

  AffineMap inMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(0));
  AffineMap filterMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(1));
  AffineMap outMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInitOperand(0));

  if (filterMap.getNumResults() != 4 || outMap.getNumResults() < 4) {
    return false;
  }
  bool hasAdd = llvm::any_of(inMap.getResults(), [](AffineExpr expr) {
    return expr.getKind() == AffineExprKind::Add;
  });
  if (!hasAdd) return false;

  return outMap.getResult(outMap.getNumResults() - 1) ==
             filterMap.getResult(3) &&
         outMap.getResult(outMap.getNumResults() - 2) == filterMap.getResult(2);
}

bool isDepthwiseConv2DNhwcHwcGenericOp(linalg::GenericOp genericOp) {
  if (linalg::isaConvolutionOpOfType<linalg::DepthwiseConv2DNhwcHwcOp>(
          genericOp)) {
    return true;
  }
  if (genericOp.getNumDpsInputs() != 2 || genericOp.getNumDpsInits() != 1) {
    return false;
  }
  if (genericOp.getNumReductionLoops() != 2 ||
      genericOp.getNumParallelLoops() > 4) {
    return false;
  }
  if (!hasMultiplyAddBody(genericOp)) return false;

  AffineMap inMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(0));
  AffineMap filterMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(1));
  AffineMap outMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInitOperand(0));

  if (filterMap.getNumResults() != 3 || outMap.getNumResults() < 3) {
    return false;
  }
  bool hasAdd = llvm::any_of(inMap.getResults(), [](AffineExpr expr) {
    return expr.getKind() == AffineExprKind::Add;
  });
  if (!hasAdd) return false;

  return outMap.getResult(outMap.getNumResults() - 1) == filterMap.getResult(2);
}

bool isaPoolingNdhwcGenericOp(linalg::GenericOp genericOp) {
  if (linalg::isaConvolutionOpOfType<linalg::PoolingNdhwcSumOp>(genericOp) ||
      linalg::isaConvolutionOpOfType<linalg::PoolingNdhwcMaxOp>(genericOp) ||
      linalg::isaConvolutionOpOfType<linalg::PoolingNdhwcMinOp>(genericOp)) {
    return true;
  }
  if (genericOp.getNumDpsInputs() != 2 || genericOp.getNumDpsInits() != 1) {
    return false;
  }
  if (genericOp.getNumReductionLoops() != 3 ||
      genericOp.getNumParallelLoops() > 5) {
    return false;
  }

  AffineMap filterMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(1));
  AffineMap outMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInitOperand(0));
  return filterMap.getNumResults() == 3 && outMap.getNumResults() >= 4;
}

bool isaTransposeGenericOp(linalg::GenericOp genericOp) {
  if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1) {
    return false;
  }
  if (genericOp.getNumParallelLoops() != genericOp.getNumLoops()) {
    return false;
  }

  AffineMap inMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(0));
  AffineMap outMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInitOperand(0));
  if (!inMap.isPermutation() || !outMap.isPermutation() || inMap == outMap) {
    return false;
  }

  Block *body = genericOp.getBody();
  if (!body || body->empty()) return false;
  auto yieldOp = dyn_cast<linalg::YieldOp>(body->getTerminator());
  return yieldOp && yieldOp.getNumOperands() == 1 &&
         yieldOp.getOperand(0) == body->getArgument(0);
}

bool isaBroadcastGenericOp(linalg::GenericOp genericOp) {
  if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1) {
    return false;
  }
  if (genericOp.getNumParallelLoops() != genericOp.getNumLoops()) {
    return false;
  }

  AffineMap inMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(0));
  AffineMap outMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInitOperand(0));
  if (inMap.getNumResults() >= outMap.getNumResults()) {
    return false;
  }
  if (!inMap.isProjectedPermutation() || !outMap.isPermutation()) {
    return false;
  }

  Block *body = genericOp.getBody();
  if (!body || body->empty()) return false;
  auto yieldOp = dyn_cast<linalg::YieldOp>(body->getTerminator());
  return yieldOp && yieldOp.getNumOperands() == 1 &&
         yieldOp.getOperand(0) == body->getArgument(0);
}

bool isaReduceGenericOp(linalg::GenericOp genericOp) {
  if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1) {
    return false;
  }
  if (genericOp.getNumReductionLoops() < 1) {
    return false;
  }

  AffineMap inMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(0));
  AffineMap outMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInitOperand(0));
  if (outMap.getNumResults() >= inMap.getNumResults()) {
    return false;
  }
  return (inMap.isPermutation() || inMap.isProjectedPermutation()) &&
         (outMap.isPermutation() || outMap.isProjectedPermutation());
}

FailureOr<linalg::ContractionDimensions> getContractionDims(
    linalg::GenericOp genericOp) {
  if (linalg::isaContractionOpInterface(genericOp)) {
    return linalg::inferContractionDims(genericOp);
  }
  if (genericOp.getNumDpsInputs() < 2 || genericOp.getNumDpsInits() != 1 ||
      !hasMultiplyAddBody(genericOp)) {
    return failure();
  }
  AffineMap in0Map =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(0));
  AffineMap in1Map =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(1));
  AffineMap outMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInitOperand(0));
  if (!in0Map.isProjectedPermutation() || !in1Map.isProjectedPermutation() ||
      !outMap.isProjectedPermutation() ||
      outMap.getNumResults() != genericOp.getNumParallelLoops()) {
    return failure();
  }
  return linalg::inferContractionDims(
      ArrayRef<AffineMap>{in0Map, in1Map, outMap});
}

bool isBatchMatmulGenericOp(linalg::GenericOp genericOp) {
  if (genericOp.getNumParallelLoops() < 2 ||
      genericOp.getNumReductionLoops() < 1) {
    return false;
  }
  auto dims = getContractionDims(genericOp);
  if (failed(dims)) return false;
  return !dims->m.empty() && !dims->n.empty() && !dims->k.empty() &&
         !(dims->m.size() == 2 && dims->n.size() == 2 && dims->k.size() == 2);
}

bool isMatvecGenericOp(linalg::GenericOp genericOp) {
  if ((genericOp.getNumParallelLoops() != 1 &&
       genericOp.getNumParallelLoops() != 2) ||
      genericOp.getNumReductionLoops() != 1) {
    return false;
  }
  auto dims = getContractionDims(genericOp);
  if (failed(dims)) return false;
  return dims->batch.size() == genericOp.getNumParallelLoops() - 1 &&
         dims->m.size() == 1 && dims->n.empty() && dims->k.size() == 1;
}

bool isVecmatGenericOp(linalg::GenericOp genericOp) {
  if ((genericOp.getNumParallelLoops() != 1 &&
       genericOp.getNumParallelLoops() != 2) ||
      genericOp.getNumReductionLoops() != 1) {
    return false;
  }
  auto dims = getContractionDims(genericOp);
  if (failed(dims)) return false;
  return dims->batch.size() == genericOp.getNumParallelLoops() - 1 &&
         dims->m.empty() && dims->n.size() == 1 && dims->k.size() == 1;
}

bool isDotGenericOp(linalg::GenericOp genericOp) {
  if (genericOp.getNumParallelLoops() != 0 ||
      genericOp.getNumReductionLoops() != 1) {
    return false;
  }
  auto dims = getContractionDims(genericOp);
  if (failed(dims)) return false;
  return dims->batch.empty() && dims->m.empty() && dims->n.empty() &&
         dims->k.size() == 1;
}

bool isaElementwiseDivGenericOp(linalg::GenericOp genericOp) {
  if (genericOp.getNumDpsInputs() != 2 || genericOp.getNumDpsInits() != 1) {
    return false;
  }
  if (genericOp.getNumParallelLoops() != genericOp.getNumLoops()) {
    return false;
  }
  if (!linalg::isElementwise(genericOp)) {
    return false;
  }

  Block *body = genericOp.getBody();
  if (!body) return false;
  return llvm::any_of(body->getOperations(), [](Operation &op) {
    return isa<arith::DivFOp, arith::DivSIOp, arith::DivUIOp>(&op);
  });
}

bool isaElementwiseArithBinaryGenericOp(linalg::GenericOp genericOp) {
  if (genericOp.getNumDpsInputs() != 2 || genericOp.getNumDpsInits() != 1) {
    return false;
  }
  if (genericOp.getNumParallelLoops() != genericOp.getNumLoops()) {
    return false;
  }
  if (!linalg::isElementwise(genericOp)) {
    return false;
  }

  Block *body = genericOp.getBody();
  if (!body) return false;
  return llvm::any_of(body->getOperations(), [](Operation &op) {
    return isa<arith::AddFOp, arith::AddIOp, arith::SubFOp, arith::SubIOp,
               arith::MulFOp, arith::MulIOp, arith::MinimumFOp,
               arith::MinNumFOp, arith::MinSIOp, arith::MinUIOp,
               arith::MaximumFOp, arith::MaxNumFOp, arith::MaxSIOp,
               arith::MaxUIOp>(&op);
  });
}

bool isaElementwiseUnaryArithGenericOp(linalg::GenericOp genericOp) {
  if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1) {
    return false;
  }
  if (genericOp.getNumParallelLoops() != genericOp.getNumLoops()) {
    return false;
  }
  if (!linalg::isElementwise(genericOp)) {
    return false;
  }

  Block *body = genericOp.getBody();
  if (!body) return false;
  return llvm::any_of(body->getOperations(), [](Operation &op) {
    return isa<math::AbsFOp, math::AbsIOp, arith::NegFOp>(&op);
  });
}

bool isaElementwiseRoundingGenericOp(linalg::GenericOp genericOp) {
  if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1) {
    return false;
  }
  if (genericOp.getNumParallelLoops() != genericOp.getNumLoops()) {
    return false;
  }
  if (!linalg::isElementwise(genericOp)) {
    return false;
  }

  Block *body = genericOp.getBody();
  if (!body) return false;
  return llvm::any_of(body->getOperations(), [](Operation &op) {
    return isa<math::CeilOp, math::FloorOp, math::RoundOp, math::RoundEvenOp>(
        &op);
  });
}

bool isaElementwiseRootGenericOp(linalg::GenericOp genericOp) {
  if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1) {
    return false;
  }
  if (genericOp.getNumParallelLoops() != genericOp.getNumLoops()) {
    return false;
  }
  if (!linalg::isElementwise(genericOp)) {
    return false;
  }

  Block *body = genericOp.getBody();
  if (!body) return false;
  return llvm::any_of(body->getOperations(), [](Operation &op) {
    return isa<math::SqrtOp, math::RsqrtOp>(&op);
  });
}

bool isaElementwiseTranscendentalGenericOp(linalg::GenericOp genericOp) {
  if (genericOp.getNumDpsInputs() < 1 || genericOp.getNumDpsInits() != 1) {
    return false;
  }
  if (genericOp.getNumParallelLoops() != genericOp.getNumLoops()) {
    return false;
  }
  if (!linalg::isElementwise(genericOp)) {
    return false;
  }

  return hasTranscendentalBody(genericOp);
}

bool isaMapGenericOp(linalg::GenericOp genericOp) {
  if (genericOp.getNumParallelLoops() != genericOp.getNumLoops()) {
    return false;
  }
  if (!linalg::isElementwise(genericOp)) {
    return false;
  }
  return llvm::all_of(genericOp.getIndexingMapsArray(),
                      [](AffineMap map) { return map.isIdentity(); });
}

void setLinalgGenericVectorSizes(
    linalg::GenericOp genericOp, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    MutableArrayRef<int64_t> vectorReductionSizes,
    IREE::CPU::LoweringPipeline &pipeline) {
  if (isaTransposeGenericOp(genericOp)) {
    return setTransposeVectorSizes(genericOp, vectorWidth, analysis,
                                   vectorParallelSizes, pipeline);
  }

  if (isaBroadcastGenericOp(genericOp)) {
    return setBroadcastVectorSizes(genericOp, vectorWidth, analysis,
                                   vectorParallelSizes, pipeline);
  }

  if (isaReduceGenericOp(genericOp)) {
    return setReduceVectorSizes(genericOp, vectorWidth, analysis,
                                vectorParallelSizes, vectorReductionSizes,
                                pipeline);
  }

  if (linalg::isaConvolutionOpOfType<linalg::Conv2DNhwcHwcfOp>(genericOp)) {
    return setConv2DNhwcHwcfVectorSizes(genericOp, vectorWidth, analysis,
                                        vectorParallelSizes, pipeline);
  }

  if (linalg::isaConvolutionOpOfType<linalg::Conv2DNchwFchwOp>(genericOp) ||
      linalg::isaConvolutionOpOfType<linalg::Conv2DNchwFchwQOp>(genericOp)) {
    return setConv2DNchwFchwVectorSizes(genericOp, vectorWidth, analysis,
                                        vectorParallelSizes, pipeline);
  }

  if (linalg::isaConvolutionOpOfType<linalg::Conv2DNhwcFhwcOp>(genericOp) ||
      linalg::isaConvolutionOpOfType<linalg::Conv2DNhwcFhwcQOp>(genericOp)) {
    return setConv2DNhwcFhwcVectorSizes(genericOp, vectorWidth, analysis,
                                        vectorParallelSizes, pipeline);
  }

  if (linalg::isaConvolutionOpOfType<linalg::Conv1DNwcWcfOp>(genericOp)) {
    return setConv1DNwcWcfVectorSizes(genericOp, vectorWidth, analysis,
                                      vectorParallelSizes, pipeline);
  }

  if (linalg::isaConvolutionOpOfType<linalg::Conv1DNcwFcwOp>(genericOp)) {
    return setConv1DNcwFcwVectorSizes(genericOp, vectorWidth, analysis,
                                      vectorParallelSizes, pipeline);
  }

  if (linalg::isaConvolutionOpOfType<linalg::Conv1DOp>(genericOp)) {
    return setConv1DVectorSizes(genericOp, vectorWidth, analysis,
                                vectorParallelSizes, pipeline);
  }

  if (linalg::isaConvolutionOpOfType<linalg::Conv2DOp>(genericOp)) {
    return setConv2DVectorSizes(genericOp, vectorWidth, analysis,
                                vectorParallelSizes, pipeline);
  }

  if (linalg::isaConvolutionOpOfType<linalg::Conv3DOp>(genericOp)) {
    return setConv3DVectorSizes(genericOp, vectorWidth, analysis,
                                vectorParallelSizes, pipeline);
  }

  if (linalg::isaConvolutionOpOfType<linalg::Conv3DNdhwcDhwcfOp>(genericOp) ||
      linalg::isaConvolutionOpOfType<linalg::Conv3DNdhwcDhwcfQOp>(genericOp)) {
    return setConv3DNdhwcDhwcfVectorSizes(genericOp, vectorWidth, analysis,
                                          vectorParallelSizes, pipeline);
  }

  if (linalg::isaConvolutionOpOfType<linalg::Conv3DNcdhwFcdhwOp>(genericOp)) {
    return setConv3DNcdhwFcdhwVectorSizes(genericOp, vectorWidth, analysis,
                                          vectorParallelSizes, pipeline);
  }

  if (linalg::isaConvolutionOpOfType<linalg::Conv2DNhwgcGfhwcOp>(genericOp) ||
      linalg::isaConvolutionOpOfType<linalg::Conv2DNhwgcGfhwcQOp>(genericOp)) {
    return setConv2DNhwgcGfhwcVectorSizes(genericOp, vectorWidth, analysis,
                                          vectorParallelSizes, pipeline);
  }

  if (linalg::isaConvolutionOpOfType<linalg::Conv2DNgchwGfchwOp>(genericOp) ||
      linalg::isaConvolutionOpOfType<linalg::Conv2DNgchwGfchwQOp>(genericOp)) {
    return setConv2DNgchwGfchwVectorSizes(genericOp, vectorWidth, analysis,
                                          vectorParallelSizes, pipeline);
  }

  if (linalg::isaConvolutionOpOfType<linalg::Conv2DNgchwFgchwOp>(genericOp)) {
    return setConv2DNgchwFgchwVectorSizes(genericOp, vectorWidth, analysis,
                                          vectorParallelSizes, pipeline);
  }

  if (linalg::isaConvolutionOpOfType<linalg::DepthwiseConv1DNwcWcmOp>(
          genericOp)) {
    return setDepthwiseConv1DNwcWcmVectorSizes(genericOp, vectorWidth, analysis,
                                               vectorParallelSizes, pipeline);
  }

  if (linalg::isaConvolutionOpOfType<linalg::PoolingNhwcSumOp>(genericOp) ||
      linalg::isaConvolutionOpOfType<linalg::PoolingNhwcMaxOp>(genericOp) ||
      linalg::isaConvolutionOpOfType<linalg::PoolingNhwcMinOp>(genericOp) ||
      linalg::isaConvolutionOpOfType<linalg::PoolingNhwcMaxUnsignedOp>(
          genericOp) ||
      linalg::isaConvolutionOpOfType<linalg::PoolingNhwcMinUnsignedOp>(
          genericOp)) {
    return setPoolingNhwcVectorSizes(genericOp, vectorWidth, analysis,
                                     vectorParallelSizes, pipeline);
  }

  if (linalg::isaConvolutionOpOfType<linalg::PoolingNwcSumOp>(genericOp) ||
      linalg::isaConvolutionOpOfType<linalg::PoolingNwcMaxOp>(genericOp) ||
      linalg::isaConvolutionOpOfType<linalg::PoolingNwcMaxUnsignedOp>(
          genericOp) ||
      linalg::isaConvolutionOpOfType<linalg::PoolingNwcMinOp>(genericOp) ||
      linalg::isaConvolutionOpOfType<linalg::PoolingNwcMinUnsignedOp>(
          genericOp)) {
    return setPoolingNwcVectorSizes(genericOp, vectorWidth, analysis,
                                    vectorParallelSizes, pipeline);
  }

  if (linalg::isaConvolutionOpOfType<linalg::PoolingNcwSumOp>(genericOp) ||
      linalg::isaConvolutionOpOfType<linalg::PoolingNcwMaxOp>(genericOp)) {
    return setPoolingNcwVectorSizes(genericOp, vectorWidth, analysis,
                                    vectorParallelSizes, pipeline);
  }

  if (linalg::isaConvolutionOpOfType<linalg::PoolingNchwSumOp>(genericOp) ||
      linalg::isaConvolutionOpOfType<linalg::PoolingNchwMaxOp>(genericOp)) {
    return setPoolingNchwVectorSizes(genericOp, vectorWidth, analysis,
                                     vectorParallelSizes, pipeline);
  }

  if (isDepthwiseConv1DNwcWcGenericOp(genericOp)) {
    return setDepthwiseConv1DNwcWcVectorSizes(genericOp, vectorWidth, analysis,
                                              vectorParallelSizes, pipeline);
  }

  if (isDepthwiseConv1DNcwCwGenericOp(genericOp)) {
    return setDepthwiseConv1DNcwCwVectorSizes(genericOp, vectorWidth, analysis,
                                              vectorParallelSizes, pipeline);
  }

  if (isDepthwiseConv3DNdhwcDhwcmGenericOp(genericOp)) {
    return setDepthwiseConv3DNdhwcDhwcmVectorSizes(
        genericOp, vectorWidth, analysis, vectorParallelSizes, pipeline);
  }

  if (isDepthwiseConv3DNdhwcDhwcGenericOp(genericOp)) {
    return setDepthwiseConv3DNdhwcDhwcVectorSizes(
        genericOp, vectorWidth, analysis, vectorParallelSizes, pipeline);
  }

  if (isDepthwiseConv2DNhwcHwcmGenericOp(genericOp)) {
    return setDepthwiseConv2DNhwcHwcmVectorSizes(
        genericOp, vectorWidth, analysis, vectorParallelSizes, pipeline);
  }

  if (isDepthwiseConv2DNhwcHwcGenericOp(genericOp)) {
    return setDepthwiseConv2DNhwcHwcVectorSizes(
        genericOp, vectorWidth, analysis, vectorParallelSizes, pipeline);
  }

  if (isaPoolingNdhwcGenericOp(genericOp)) {
    return setPoolingNdhwcVectorSizes(genericOp, vectorWidth, analysis,
                                      vectorParallelSizes, pipeline);
  }

  if (isaElementwiseDivGenericOp(genericOp)) {
    return setElementwiseDivBinaryVectorSizes(genericOp, vectorWidth, analysis,
                                              vectorParallelSizes, pipeline);
  }

  if (isaElementwiseArithBinaryGenericOp(genericOp)) {
    return setElementwiseArithBinaryVectorSizes(
        genericOp, vectorWidth, analysis, vectorParallelSizes, pipeline);
  }

  if (isaElementwiseUnaryArithGenericOp(genericOp)) {
    return setElementwiseUnaryArithVectorSizes(genericOp, vectorWidth, analysis,
                                               vectorParallelSizes, pipeline);
  }

  if (isaElementwiseRoundingGenericOp(genericOp)) {
    return setElementwiseRoundingVectorSizes(genericOp, vectorWidth, analysis,
                                             vectorParallelSizes, pipeline);
  }

  if (isaElementwiseRootGenericOp(genericOp)) {
    return setElementwiseRootVectorSizes(genericOp, vectorWidth, analysis,
                                         vectorParallelSizes, pipeline);
  }
  if (isaElementwiseTranscendentalGenericOp(genericOp)) {
    return setElementwiseTranscendentalVectorSizes(
        genericOp, vectorWidth, analysis, vectorParallelSizes, pipeline);
  }

  if (isaMapGenericOp(genericOp)) {
    return setMapVectorSizes(genericOp, vectorWidth, analysis,
                             vectorParallelSizes, pipeline);
  }

  if (isBatchMatmulGenericOp(genericOp)) {
    return setMatmulVectorSizes(genericOp, vectorWidth, analysis,
                                vectorParallelSizes, pipeline);
  }

  if (isMatvecGenericOp(genericOp)) {
    return setMatvecVectorSizes(genericOp, vectorWidth, analysis,
                                vectorParallelSizes, pipeline);
  }

  if (isVecmatGenericOp(genericOp)) {
    return setVecmatVectorSizes(genericOp, vectorWidth, analysis,
                                vectorParallelSizes, pipeline);
  }

  if (isDotGenericOp(genericOp)) {
    return setDotVectorSizes(genericOp, vectorWidth, analysis,
                             vectorParallelSizes, pipeline);
  }

  return setGenericVectorSizes(genericOp, vectorWidth, analysis,
                               vectorParallelSizes, vectorReductionSizes,
                               pipeline);
}

void setLinalgOpVectorSizes(linalg::LinalgOp linalgOp, int64_t vectorWidth,
                            const CoralNPUTileSizeSelectionAnalysis &analysis,
                            MutableArrayRef<int64_t> vectorParallelSizes,
                            MutableArrayRef<int64_t> vectorReductionSizes,
                            IREE::CPU::LoweringPipeline &pipeline) {
  if (isa<linalg::Conv2DNhwcHwcfOp, linalg::Conv2DNhwcHwcfQOp>(linalgOp)) {
    return setConv2DNhwcHwcfVectorSizes(linalgOp, vectorWidth, analysis,
                                        vectorParallelSizes, pipeline);
  }

  if (isa<linalg::Conv2DNchwFchwOp, linalg::Conv2DNchwFchwQOp>(linalgOp)) {
    return setConv2DNchwFchwVectorSizes(linalgOp, vectorWidth, analysis,
                                        vectorParallelSizes, pipeline);
  }

  if (isa<linalg::Conv2DNhwcFhwcOp, linalg::Conv2DNhwcFhwcQOp>(linalgOp)) {
    return setConv2DNhwcFhwcVectorSizes(linalgOp, vectorWidth, analysis,
                                        vectorParallelSizes, pipeline);
  }

  if (isa<linalg::Conv1DNwcWcfOp>(linalgOp)) {
    return setConv1DNwcWcfVectorSizes(linalgOp, vectorWidth, analysis,
                                      vectorParallelSizes, pipeline);
  }

  if (isa<linalg::Conv1DNcwFcwOp>(linalgOp)) {
    return setConv1DNcwFcwVectorSizes(linalgOp, vectorWidth, analysis,
                                      vectorParallelSizes, pipeline);
  }

  if (isa<linalg::Conv1DOp>(linalgOp)) {
    return setConv1DVectorSizes(linalgOp, vectorWidth, analysis,
                                vectorParallelSizes, pipeline);
  }

  if (isa<linalg::Conv2DOp>(linalgOp)) {
    return setConv2DVectorSizes(linalgOp, vectorWidth, analysis,
                                vectorParallelSizes, pipeline);
  }

  if (isa<linalg::Conv3DOp>(linalgOp)) {
    return setConv3DVectorSizes(linalgOp, vectorWidth, analysis,
                                vectorParallelSizes, pipeline);
  }

  if (isa<linalg::Conv3DNdhwcDhwcfOp, linalg::Conv3DNdhwcDhwcfQOp>(linalgOp)) {
    return setConv3DNdhwcDhwcfVectorSizes(linalgOp, vectorWidth, analysis,
                                          vectorParallelSizes, pipeline);
  }

  if (isa<linalg::Conv3DNcdhwFcdhwOp>(linalgOp)) {
    return setConv3DNcdhwFcdhwVectorSizes(linalgOp, vectorWidth, analysis,
                                          vectorParallelSizes, pipeline);
  }

  if (isa<linalg::DepthwiseConv2DNhwcHwcOp>(linalgOp)) {
    return setDepthwiseConv2DNhwcHwcVectorSizes(linalgOp, vectorWidth, analysis,
                                                vectorParallelSizes, pipeline);
  }

  if (isa<linalg::DepthwiseConv1DNwcWcOp>(linalgOp)) {
    return setDepthwiseConv1DNwcWcVectorSizes(linalgOp, vectorWidth, analysis,
                                              vectorParallelSizes, pipeline);
  }

  if (isa<linalg::DepthwiseConv1DNcwCwOp>(linalgOp)) {
    return setDepthwiseConv1DNcwCwVectorSizes(linalgOp, vectorWidth, analysis,
                                              vectorParallelSizes, pipeline);
  }

  if (isa<linalg::DepthwiseConv1DNwcWcmOp>(linalgOp) ||
      linalg::isaConvolutionOpOfType<linalg::DepthwiseConv1DNwcWcmOp>(
          linalgOp)) {
    return setDepthwiseConv1DNwcWcmVectorSizes(linalgOp, vectorWidth, analysis,
                                               vectorParallelSizes, pipeline);
  }

  if (isa<linalg::PoolingNhwcSumOp, linalg::PoolingNhwcMaxOp,
          linalg::PoolingNhwcMinOp, linalg::PoolingNhwcMaxUnsignedOp,
          linalg::PoolingNhwcMinUnsignedOp>(linalgOp)) {
    return setPoolingNhwcVectorSizes(linalgOp, vectorWidth, analysis,
                                     vectorParallelSizes, pipeline);
  }

  if (isa<linalg::DepthwiseConv3DNdhwcDhwcOp>(linalgOp)) {
    return setDepthwiseConv3DNdhwcDhwcVectorSizes(
        linalgOp, vectorWidth, analysis, vectorParallelSizes, pipeline);
  }

  if (isa<linalg::DepthwiseConv3DNdhwcDhwcmOp>(linalgOp)) {
    return setDepthwiseConv3DNdhwcDhwcmVectorSizes(
        linalgOp, vectorWidth, analysis, vectorParallelSizes, pipeline);
  }

  if (isa<linalg::Conv2DNhwgcGfhwcOp, linalg::Conv2DNhwgcGfhwcQOp>(linalgOp)) {
    return setConv2DNhwgcGfhwcVectorSizes(linalgOp, vectorWidth, analysis,
                                          vectorParallelSizes, pipeline);
  }

  if (isa<linalg::Conv2DNgchwGfchwOp, linalg::Conv2DNgchwGfchwQOp>(linalgOp)) {
    return setConv2DNgchwGfchwVectorSizes(linalgOp, vectorWidth, analysis,
                                          vectorParallelSizes, pipeline);
  }

  if (isa<linalg::Conv2DNgchwFgchwOp>(linalgOp)) {
    return setConv2DNgchwFgchwVectorSizes(linalgOp, vectorWidth, analysis,
                                          vectorParallelSizes, pipeline);
  }

  if (isa<linalg::DepthwiseConv2DNhwcHwcmOp,
          linalg::DepthwiseConv2DNhwcHwcmQOp>(linalgOp)) {
    return setDepthwiseConv2DNhwcHwcmVectorSizes(
        linalgOp, vectorWidth, analysis, vectorParallelSizes, pipeline);
  }

  if (isa<linalg::PoolingNwcSumOp, linalg::PoolingNwcMaxOp,
          linalg::PoolingNwcMaxUnsignedOp, linalg::PoolingNwcMinOp,
          linalg::PoolingNwcMinUnsignedOp>(linalgOp) ||
      linalg::isaConvolutionOpOfType<linalg::PoolingNwcSumOp>(linalgOp) ||
      linalg::isaConvolutionOpOfType<linalg::PoolingNwcMaxOp>(linalgOp) ||
      linalg::isaConvolutionOpOfType<linalg::PoolingNwcMaxUnsignedOp>(
          linalgOp) ||
      linalg::isaConvolutionOpOfType<linalg::PoolingNwcMinOp>(linalgOp) ||
      linalg::isaConvolutionOpOfType<linalg::PoolingNwcMinUnsignedOp>(
          linalgOp)) {
    return setPoolingNwcVectorSizes(linalgOp, vectorWidth, analysis,
                                    vectorParallelSizes, pipeline);
  }

  if (isa<linalg::PoolingNcwSumOp, linalg::PoolingNcwMaxOp>(linalgOp) ||
      linalg::isaConvolutionOpOfType<linalg::PoolingNcwSumOp>(linalgOp) ||
      linalg::isaConvolutionOpOfType<linalg::PoolingNcwMaxOp>(linalgOp)) {
    return setPoolingNcwVectorSizes(linalgOp, vectorWidth, analysis,
                                    vectorParallelSizes, pipeline);
  }

  if (isa<linalg::PoolingNchwSumOp, linalg::PoolingNchwMaxOp>(linalgOp) ||
      linalg::isaConvolutionOpOfType<linalg::PoolingNchwSumOp>(linalgOp) ||
      linalg::isaConvolutionOpOfType<linalg::PoolingNchwMaxOp>(linalgOp)) {
    return setPoolingNchwVectorSizes(linalgOp, vectorWidth, analysis,
                                     vectorParallelSizes, pipeline);
  }

  if (isa<linalg::PoolingNdhwcSumOp, linalg::PoolingNdhwcMaxOp,
          linalg::PoolingNdhwcMinOp>(linalgOp)) {
    return setPoolingNdhwcVectorSizes(linalgOp, vectorWidth, analysis,
                                      vectorParallelSizes, pipeline);
  }

  if (isa<linalg::MatmulOp, linalg::QuantizedMatmulOp, linalg::BatchMatmulOp,
          linalg::QuantizedBatchMatmulOp, linalg::BatchReduceMatmulOp>(
          linalgOp)) {
    return setMatmulVectorSizes(linalgOp, vectorWidth, analysis,
                                vectorParallelSizes, pipeline);
  }

  if (isa<linalg::ContractOp>(linalgOp)) {
    return setContractVectorSizes(linalgOp, vectorWidth, analysis,
                                  vectorParallelSizes, pipeline);
  }

  if (isa<linalg::MatvecOp>(linalgOp)) {
    return setMatvecVectorSizes(linalgOp, vectorWidth, analysis,
                                vectorParallelSizes, pipeline);
  }

  if (isa<linalg::BatchMatvecOp>(linalgOp)) {
    return setMatvecVectorSizes(linalgOp, vectorWidth, analysis,
                                vectorParallelSizes, pipeline);
  }

  if (isa<linalg::VecmatOp>(linalgOp)) {
    return setVecmatVectorSizes(linalgOp, vectorWidth, analysis,
                                vectorParallelSizes, pipeline);
  }

  if (isa<linalg::BatchVecmatOp>(linalgOp)) {
    return setVecmatVectorSizes(linalgOp, vectorWidth, analysis,
                                vectorParallelSizes, pipeline);
  }

  if (isa<linalg::DotOp>(linalgOp)) {
    return setDotVectorSizes(linalgOp, vectorWidth, analysis,
                             vectorParallelSizes, pipeline);
  }

  if (isa<linalg::Mmt4DOp>(linalgOp)) {
    return setMmt4DVectorSizes(linalgOp, vectorWidth, analysis,
                               vectorParallelSizes, pipeline);
  }

  if (isa<linalg::BatchMmt4DOp>(linalgOp)) {
    return setMmt4DVectorSizes(linalgOp, vectorWidth, analysis,
                               vectorParallelSizes, pipeline);
  }

  if (isa<linalg::AddOp, linalg::SubOp, linalg::MulOp, linalg::MinOp,
          linalg::MaxOp, linalg::ElementwiseOp>(linalgOp)) {
    return setElementwiseArithBinaryVectorSizes(linalgOp, vectorWidth, analysis,
                                                vectorParallelSizes, pipeline);
  }

  if (isa<linalg::DivOp, linalg::DivUnsignedOp>(linalgOp)) {
    return setElementwiseDivBinaryVectorSizes(linalgOp, vectorWidth, analysis,
                                              vectorParallelSizes, pipeline);
  }

  if (isa<linalg::AbsOp, linalg::NegFOp, linalg::SquareOp, linalg::CopyOp>(
          linalgOp)) {
    return setElementwiseUnaryArithVectorSizes(linalgOp, vectorWidth, analysis,
                                               vectorParallelSizes, pipeline);
  }

  if (isa<linalg::CeilOp, linalg::FloorOp, linalg::RoundOp>(linalgOp)) {
    return setElementwiseRoundingVectorSizes(linalgOp, vectorWidth, analysis,
                                             vectorParallelSizes, pipeline);
  }

  if (isa<linalg::SqrtOp, linalg::RsqrtOp, linalg::ReciprocalOp>(linalgOp)) {
    return setElementwiseRootVectorSizes(linalgOp, vectorWidth, analysis,
                                         vectorParallelSizes, pipeline);
  }

  if (isa<linalg::ExpOp, linalg::LogOp, linalg::TanhOp, linalg::ErfOp,
          linalg::PowFOp>(linalgOp)) {
    return setElementwiseTranscendentalVectorSizes(
        linalgOp, vectorWidth, analysis, vectorParallelSizes, pipeline);
  }

  if (isa<linalg::MapOp>(linalgOp)) {
    return setMapVectorSizes(linalgOp, vectorWidth, analysis,
                             vectorParallelSizes, pipeline);
  }

  if (isa<linalg::TransposeOp>(linalgOp)) {
    return setTransposeVectorSizes(linalgOp, vectorWidth, analysis,
                                   vectorParallelSizes, pipeline);
  }

  if (isa<linalg::BroadcastOp>(linalgOp)) {
    return setBroadcastVectorSizes(linalgOp, vectorWidth, analysis,
                                   vectorParallelSizes, pipeline);
  }

  if (isa<linalg::ReduceOp>(linalgOp)) {
    return setReduceVectorSizes(linalgOp, vectorWidth, analysis,
                                vectorParallelSizes, vectorReductionSizes,
                                pipeline);
  }
}

// Snaps tiles to exact divisors of the range (or 0 when covering the range).
void capVectorSizes(const CoralNPUTileSizeSelectionAnalysis &analysis,
                    int64_t vectorWidth,
                    MutableArrayRef<int64_t> vectorParallelSizes,
                    MutableArrayRef<int64_t> vectorReductionSizes) {
  for (auto [i, range] : llvm::enumerate(analysis.staticLoopRanges)) {
    int64_t &tile = vectorParallelSizes[i];
    if (range != 0 && tile >= range) tile = 0;  // do not tile
    if (tile > 0) {
      bool isInnerParallel =
          !analysis.parallelLoops.empty() && i == analysis.parallelLoops[0];
      // Keep full tile on 1-parallel, 1-reduction ops when tail >= vectorWidth.
      if (isInnerParallel && analysis.parallelLoops.size() == 1 &&
          analysis.reductionLoops.size() == 1 && range > tile &&
          std::gcd(tile, range) < tile / 2 && range % tile >= vectorWidth) {
        continue;
      }
      int64_t g = std::gcd(tile, range);
      tile = (isInnerParallel && g >= 4) ? g : pickTileDivisor(range, tile);
    }
  }

  for (auto [tile, range] :
       llvm::zip_equal(vectorReductionSizes, analysis.staticLoopRanges)) {
    if (range != 0 && tile >= range) tile = 0;  // do not tile
    if (tile > 0) tile = pickTileDivisor(range, tile);
  }
}

void setVectorSizes(TilingInterface tilingOp, int64_t vectorWidth,
                    const CoralNPUTileSizeSelectionAnalysis &analysis,
                    MutableArrayRef<int64_t> vectorParallelSizes,
                    MutableArrayRef<int64_t> vectorReductionSizes,
                    IREE::CPU::LoweringPipeline &pipeline) {
  // set very strict default/fallback values
  pipeline = IREE::CPU::LoweringPipeline::DoubleTilingExpert;

  for (size_t vecIdx : analysis.parallelLoops) {
    vectorParallelSizes[vecIdx] = 1;
  }

  for (size_t vecIdx : analysis.reductionLoops) {
    vectorReductionSizes[vecIdx] = 1;
  }

  // try to specialize the values
  if (auto genericOp = dyn_cast<linalg::GenericOp>(tilingOp.getOperation())) {
    setLinalgGenericVectorSizes(genericOp, vectorWidth, analysis,
                                vectorParallelSizes, vectorReductionSizes,
                                pipeline);
  } else if (auto linalgOp =
                 dyn_cast<linalg::LinalgOp>(tilingOp.getOperation())) {
    setLinalgOpVectorSizes(linalgOp, vectorWidth, analysis, vectorParallelSizes,
                           vectorReductionSizes, pipeline);
  }

  // Vectorize K reduction when innermost in both inputs.
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(tilingOp.getOperation());
      linalgOp && linalgOp.getNumDpsInputs() >= 2 &&
      linalgOp.hasOnlyProjectedPermutations() &&
      !analysis.reductionLoops.empty() && !isZvtMatrixContraction(linalgOp)) {
    const auto &loops = analysis.parallelLoops;
    size_t k = analysis.reductionLoops[0];
    AffineExpr kExpr = getAffineDimExpr(k, linalgOp.getContext());
    auto maps = linalgOp.getIndexingMapsArray();
    if ((loops.empty() || vectorParallelSizes[loops[0]] > 1) &&
        std::gcd(vectorWidth, analysis.staticLoopRanges[k]) >= 4 &&
        maps[0].getResults().take_back() == ArrayRef(kExpr) &&
        maps[1].getResults().take_back() == ArrayRef(kExpr)) {
      auto eff = [&](size_t i) {
        return std::gcd(vectorParallelSizes[i], analysis.staticLoopRanges[i]);
      };
      Type inTy = getElementTypeOrSelf(linalgOp.getDpsInputOperand(0)->get());
      Type outTy = getElementTypeOrSelf(linalgOp.getDpsInitOperand(0)->get());
      bool isWidened =
          inTy.isIntOrFloat() && outTy.isIntOrFloat() &&
          inTy.getIntOrFloatBitWidth() < outTy.getIntOrFloatBitWidth();
      bool wideMN =
          loops.size() >= 2 && eff(loops[0]) >= 4 && eff(loops[1]) >= 2;
      if (!wideMN)
        for (size_t i : loops) vectorParallelSizes[i] = 1;
      int64_t redScale = !wideMN && isWidened ? 8 : (wideMN ? 1 : 4);
      vectorReductionSizes[k] = redScale * vectorWidth;
    }
  }

  capVectorSizes(analysis, vectorWidth, vectorParallelSizes,
                 vectorReductionSizes);
}

struct CoralNPUTileSizeSelectionRegisterPass
    : public impl::CoralNPUTileSizeSelectionRegisterBase<
          CoralNPUTileSizeSelectionRegisterPass> {
  using CoralNPUTileSizeSelectionRegisterBase::
      CoralNPUTileSizeSelectionRegisterBase;

  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();

    auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
    if (!targetAttr || targetAttr.getBackend().getValue() != "coralnpu") {
      return;
    }

    LLVM_DEBUG(llvm::dbgs()
               << "Running CoralNPUTileSizeSelectionRegisterPass on function: "
               << funcOp.getName() << "\n");

    if (numVectorRegisters <= 0) {
      funcOp.emitError("num-vector-registers must be positive, got ")
          << numVectorRegisters;
      signalPassFailure();
      return;
    }
    if (vectorRegisterWidthBits < 0) {
      funcOp.emitError("vector-register-width-bits must be non-negative, got ")
          << vectorRegisterWidthBits;
      signalPassFailure();
      return;
    }

    auto &analysis = getAnalysis<CoralNPUTileSizeSelectionAnalysis>();
    if (failed(analysis.status)) {
      return;
    }
    analysis.numVectorRegisters = numVectorRegisters;

    auto tilingOp = analysis.rootTilingOp;

    // Resolve VLEN and compute vector width
    int64_t vlenBits = vectorRegisterWidthBits;
    if (vlenBits == 0) {
      vlenBits = getVlenFromTargetFeatures(funcOp).value_or(128);
    }
    int64_t vectorWidth = (vlenBits / 8) / analysis.elemSizeBytes;
    assert(vectorWidth > 0);

    int64_t numLoops = analysis.staticLoopRanges.size();
    SmallVector<int64_t> vectorParallelSizes(numLoops, 0);
    SmallVector<int64_t> vectorReductionSizes(numLoops, 0);

    IREE::CPU::LoweringPipeline pipeline = IREE::CPU::LoweringPipeline::Default;

    setVectorSizes(tilingOp, vectorWidth, analysis, vectorParallelSizes,
                   vectorReductionSizes, pipeline);

    auto addTilingLevel = [&](SmallVectorImpl<NamedAttribute> &items,
                              IREE::CPU::TilingLevel level,
                              ArrayRef<int64_t> sizes) {
      if (llvm::any_of(sizes, [](int64_t s) { return s != 0; })) {
        items.emplace_back(
            StringAttr::get(context, IREE::CPU::getTilingLevelName(level)),
            getTilingLevelAttr(context, sizes));
      }
    };

    // Clamp the shared reduction tile to the smallest per-op inner tile.
    for (TilingInterface nonRootOp : analysis.nonRootTilingOps) {
      CoralNPUTileSizeSelectionAnalysis opAnalysis = analysis;
      classifyLoops(nonRootOp, opAnalysis.parallelLoops,
                    opAnalysis.reductionLoops);
      bool isParallel = opAnalysis.reductionLoops.empty();
      Type elTy = getElementTypeOrSelf(nonRootOp->getResult(0));
      int64_t opVectorWidth = vectorWidth;
      if (elTy.isIntOrFloat() && elTy.getIntOrFloatBitWidth() >= 8) {
        opVectorWidth = vlenBits / elTy.getIntOrFloatBitWidth();
        opAnalysis.elemSizeBytes = elTy.getIntOrFloatBitWidth() / 8;
      }
      SmallVector<int64_t> opVecParallel(numLoops, 0);
      SmallVector<int64_t> opVecReduction(numLoops, 0);
      IREE::CPU::LoweringPipeline opPipeline =
          IREE::CPU::LoweringPipeline::Default;
      setVectorSizes(nonRootOp, opVectorWidth, opAnalysis, opVecParallel,
                     opVecReduction, opPipeline);
      size_t red = analysis.reductionLoops[0];  // The only reduction loop.
      int64_t opTile = isParallel ? opVecParallel[red] : opVecReduction[red];
      if (opTile > 0 && (vectorReductionSizes[red] == 0 ||
                         opTile < vectorReductionSizes[red]))
        vectorReductionSizes[red] = opTile;
    }

    // Pure parallel producers of rootTilingOp (directly or through other such
    // producers) are fused during reduction tiling and do not need a separate
    // lowering_config.
    llvm::SmallDenseSet<Operation *> fusedProducers;
    for (TilingInterface nonRootOp : llvm::reverse(analysis.nonRootTilingOps)) {
      if (!llvm::is_contained(nonRootOp.getLoopIteratorTypes(),
                              utils::IteratorType::reduction) &&
          llvm::any_of(nonRootOp->getUsers(), [&](Operation *user) {
            return user == tilingOp.getOperation() ||
                   fusedProducers.contains(user);
          })) {
        fusedProducers.insert(nonRootOp.getOperation());
      }
    }

    for (TilingInterface nonRootOp : analysis.nonRootTilingOps) {
      if (fusedProducers.contains(nonRootOp.getOperation())) continue;
      bool isParallel = !llvm::is_contained(nonRootOp.getLoopIteratorTypes(),
                                            utils::IteratorType::reduction);
      SmallVector<NamedAttribute> opConfigItems;
      addTilingLevel(opConfigItems,
                     IREE::CPU::TilingLevel::VectorCommonParallelTiles,
                     vectorParallelSizes);
      addTilingLevel(opConfigItems,
                     isParallel
                         ? IREE::CPU::TilingLevel::VectorInnerParallelTiles
                         : IREE::CPU::TilingLevel::VectorReductionTiles,
                     vectorReductionSizes);
      if (!opConfigItems.empty()) {
        setLoweringConfig(nonRootOp, IREE::CPU::LoweringConfigAttr::get(
                                         context, opConfigItems));
      }
    }

    SmallVector<NamedAttribute> configItems;
    addTilingLevel(configItems,
                   IREE::CPU::TilingLevel::VectorCommonParallelTiles,
                   vectorParallelSizes);
    addTilingLevel(configItems, IREE::CPU::TilingLevel::VectorReductionTiles,
                   vectorReductionSizes);

    auto loweringConfig =
        IREE::CPU::LoweringConfigAttr::get(context, configItems);

    auto translationInfo = IREE::Codegen::TranslationInfoAttr::get(
        context, IREE::CPU::PipelineAttr::get(context, pipeline),
        SymbolRefAttr(), /*workgroupSize=*/{}, /*subgroupSize=*/0,
        /*configuration=*/DictionaryAttr());

    auto compilationInfo = IREE::Codegen::CompilationInfoAttr::get(
        context, loweringConfig, translationInfo);

    setCompilationInfo(tilingOp, compilationInfo);

    markAnalysesPreserved<CoralNPUTileSizeSelectionAnalysis>();
  }
};

}  // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createCoralNPUTileSizeSelectionRegisterPass() {
  return std::make_unique<CoralNPUTileSizeSelectionRegisterPass>();
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createCoralNPUTileSizeSelectionRegisterPass(
    CoralNPUTileSizeSelectionRegisterOptions options) {
  return std::make_unique<CoralNPUTileSizeSelectionRegisterPass>(
      std::move(options));
}

}  // namespace mlir::coralnpu_compiler
