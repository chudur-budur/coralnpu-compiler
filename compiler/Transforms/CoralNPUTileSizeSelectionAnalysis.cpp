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

#include "compiler/Transforms/CoralNPUTileSizeSelectionAnalysis.h"

#include <algorithm>

#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace mlir::coralnpu_compiler {

void classifyLoops(TilingInterface tilingOp,
                   SmallVectorImpl<size_t> &parallelLoops,
                   SmallVectorImpl<size_t> &reductionLoops) {
  auto iterTypes = tilingOp.getLoopIteratorTypes();

  parallelLoops.clear();
  reductionLoops.clear();
  for (auto [index, iterType] : llvm::enumerate(iterTypes)) {
    switch (iterType) {
      case utils::IteratorType::parallel:
        parallelLoops.push_back(index);
        break;
      case utils::IteratorType::reduction:
        reductionLoops.push_back(index);
        break;
    }
  }
  std::reverse(parallelLoops.begin(), parallelLoops.end());
  std::reverse(reductionLoops.begin(), reductionLoops.end());
}

namespace {

// Helper to get static loop ranges from TilingInterface
SmallVector<int64_t> getStaticLoopRanges(TilingInterface op) {
  OpBuilder builder(op);
  SmallVector<Range> loopRange = op.getIterationDomain(builder);
  return llvm::map_to_vector(loopRange, [](Range r) -> int64_t {
    std::optional<int64_t> intVal = getConstantIntValue(r.size);
    return intVal ? intVal.value() : ShapedType::kDynamic;
  });
}

}  // namespace

CoralNPUTileSizeSelectionAnalysis::CoralNPUTileSizeSelectionAnalysis(
    Operation *op) {
  auto funcOp = dyn_cast<FunctionOpInterface>(op);
  if (!funcOp) return;

  SmallVector<Operation *> computeOps = getComputeOps(funcOp);
  auto rootOpOr = getRootOperation(computeOps);
  if (failed(rootOpOr) || !rootOpOr.value()) return;

  rootTilingOp = dyn_cast<TilingInterface>(*rootOpOr);
  if (!rootTilingOp) {
    rootOpOr.value()->emitWarning(
        "root operation is not a TilingInterface; skipping tile size "
        "selection");
    return;
  }

  // Element size
  if (rootTilingOp->getNumResults() == 0) return;
  if (auto type = dyn_cast<ShapedType>(rootTilingOp->getResult(0).getType())) {
    auto elemType = type.getElementType();
    if (!elemType.isIntOrFloat()) {
      rootTilingOp->emitWarning("only integer and float types are supported");
      return;
    }
    auto elemBitWidth = elemType.getIntOrFloatBitWidth();
    if (elemBitWidth < 8) {
      rootTilingOp->emitWarning(
          "sub-byte types (e.g. i1, i4) are not supported");
      return;
    }
    elemSizeBytes = elemBitWidth / 8;
  } else {
    return;
  }

  // Static loop ranges
  staticLoopRanges = getStaticLoopRanges(rootTilingOp);
  if (llvm::is_contained(staticLoopRanges, ShapedType::kDynamic)) {
    rootTilingOp->emitWarning("dynamic shapes are not supported");
    return;
  }

  // Loop classification
  classifyLoops(rootTilingOp, parallelLoops, reductionLoops);

  // CoralNPUTileSizeSelection relies on producer/consumer fusion along
  // rootTilingOp's parallel loops to tile other compute ops in the dispatch.
  // A non-root compute op with reduction loops or more parallel loops than
  // rootTilingOp (e.g. decomposed linalg.softmax) is supported if both are
  // same-block elementwise ops or float reductions with the same loop ranges
  // and rootTilingOp's single innermost reduction loop (it gets its own
  // lowering_config unless it is a pure parallel producer of rootTilingOp);
  // otherwise skip custom tile size selection and fall back to
  // LLVMCPUSelectLoweringStrategyPass (MultiLoweringConfigGenerator).
  auto isSupportedOp = [](Operation *op) {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp || op->getNumResults() != 1 ||
        !linalgOp.hasOnlyProjectedPermutations() ||
        linalg::isaContractionOpInterface(linalgOp)) {
      return false;
    }
    if (linalgOp.getNumReductionLoops() == 0) return true;
    if (!getElementTypeOrSelf(op->getResult(0)).isFloat()) return false;
    if (linalgOp.getNumDpsInputs() < 2) return true;
    auto maps = linalgOp.getIndexingMapsArray();
    auto dims = linalg::inferContractionDims(
        ArrayRef<AffineMap>{maps[0], maps[1], maps.back()});
    return failed(dims) || dims->k.empty();
  };
  for (Operation *computeOp : computeOps) {
    if (computeOp == rootTilingOp) continue;
    auto tilingOp = dyn_cast<TilingInterface>(computeOp);
    if (!tilingOp) continue;
    SmallVector<size_t> opParallelLoops;
    SmallVector<size_t> opReductionLoops;
    classifyLoops(tilingOp, opParallelLoops, opReductionLoops);
    if (opReductionLoops.empty() &&
        opParallelLoops.size() <= parallelLoops.size()) {
      continue;
    }
    if (reductionLoops != ArrayRef<size_t>{staticLoopRanges.size() - 1} ||
        computeOp->getBlock() != rootTilingOp->getBlock() ||
        !isSupportedOp(rootTilingOp) || !isSupportedOp(tilingOp) ||
        getStaticLoopRanges(tilingOp) != staticLoopRanges ||
        (!opReductionLoops.empty() && opReductionLoops != reductionLoops)) {
      rootTilingOp->emitWarning(
          "non-root compute operation requires separate tiling configuration; "
          "skipping tile size selection");
      return;
    }
    nonRootTilingOps.push_back(tilingOp);
  }

  status = success();
}

}  // namespace mlir::coralnpu_compiler
