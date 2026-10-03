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

// IREE:
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"

// MLIR:
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Matchers.h"

// LLVM:
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace mlir::coralnpu_compiler {

FailureOr<int64_t> getVlenFromTargetFeatures(FunctionOpInterface funcOp) {
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
  if (!targetAttr) return failure();

  auto config = targetAttr.getConfiguration();
  if (!config) return failure();

  auto attr = config.getAs<StringAttr>("cpu_features");
  if (!attr) return failure();

  llvm::StringRef cpuFeatures = attr.getValue();
  size_t pos = cpuFeatures.find("+zvl");
  if (pos == llvm::StringRef::npos) return failure();

  llvm::StringRef suffix = cpuFeatures.substr(pos + 4);
  size_t endPos = suffix.find("b");
  if (endPos == llvm::StringRef::npos) return failure();

  llvm::StringRef vlenStr = suffix.substr(0, endPos);
  int64_t parsedVlen = 0;
  if (vlenStr.getAsInteger(10, parsedVlen)) return failure();

  return parsedVlen;
}

Attribute getTilingLevelAttr(MLIRContext *context, ArrayRef<int64_t> sizes) {
  SmallVector<bool> scalableFlags(sizes.size(), false);
  return IREE::Codegen::LoweringConfigTilingLevelAttr::get(
      context, sizes, /*tileInterchange=*/{}, scalableFlags);
}

bool hasZvtTargetFeature(Operation *op) {
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(op);
  return targetAttr && targetAttr.getBackend().getValue() == "coralnpu" &&
         hasFeature(targetAttr.getConfiguration(), "+zvtbase");
}

bool isZvtElementTypes(Type lhs, Type rhs, Type acc) {
  return lhs == rhs && ((acc.isF32() && (lhs.isF32() || lhs.isBF16())) ||
                        (acc.isInteger(32) && lhs.isInteger(8)));
}

// Keep in sync with isZvtMatrixContraction in iree-v3.11.0-0010-*.patch
// (looser; only suppresses generic contract lowering) and
// isSupportedMatrixContraction in CoralNPUMatrixCodegen.cpp (emits the asm).
bool isZvtMatrixContraction(Operation *op) {
  if (!hasZvtTargetFeature(op)) return false;

  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp || !linalg::isaContractionOpInterface(linalgOp)) return false;
  // The Zvt asm drains tiles straight to memory; with a fused consumer the
  // contraction stays on RVV, where Zvt tile sizes only grow the code.
  if (llvm::any_of(linalgOp->getUsers(), llvm::IsaPred<linalg::LinalgOp>))
    return false;

  auto in0Shaped =
      dyn_cast<ShapedType>(linalgOp.getDpsInputOperand(0)->get().getType());
  auto in1Shaped =
      dyn_cast<ShapedType>(linalgOp.getDpsInputOperand(1)->get().getType());
  Value init = linalgOp.getDpsInitOperand(0)->get();
  auto initShaped = cast<ShapedType>(init.getType());
  if (!in0Shaped || !in1Shaped || !initShaped.hasStaticShape()) return false;

  if (!isZvtElementTypes(in0Shaped.getElementType(), in1Shaped.getElementType(),
                         initShaped.getElementType()))
    return false;
  // The asm zeroes the tiles or seeds them from memory, so any other init
  // (e.g. a broadcast bias) stays on RVV.
  if (auto fill = init.getDefiningOp<linalg::FillOp>())
    init = fill.getInputs()[0];
  if (!matchPattern(init, m_Zero()) && !matchPattern(init, m_AnyZeroFloat()) &&
      !isa_and_nonnull<IREE::Codegen::LoadFromBufferOp,
                       IREE::TensorExt::DispatchTensorLoadOp>(
          init.getDefiningOp()))
    return false;

  if (isa<linalg::Mmt4DOp>(linalgOp)) {
    return in0Shaped.hasStaticShape() && initShaped.getDimSize(2) == 16 &&
           initShaped.getDimSize(3) == 16 && in0Shaped.getDimSize(3) == 1;
  }

  if (linalgOp.getNumParallelLoops() != 2 ||
      linalgOp.getNumReductionLoops() != 1 || initShaped.getRank() != 2) {
    return false;
  }
  // A may be transposed; B must be K x N and C M x N.
  SmallVector<unsigned> reductionDims;
  linalgOp.getReductionDims(reductionDims);
  auto maps = linalgOp.getIndexingMapsArray();
  AffineExpr mDim = maps[2].getResult(0), nDim = maps[2].getResult(1);
  AffineExpr kDim = getAffineDimExpr(reductionDims[0], op->getContext());
  using Exprs = ArrayRef<AffineExpr>;
  if ((maps[0].getResults() != Exprs{mDim, kDim} &&
       maps[0].getResults() != Exprs{kDim, mDim}) ||
      maps[1].getResults() != Exprs{kDim, nDim}) {
    return false;
  }
  int64_t m = initShaped.getDimSize(0), n = initShaped.getDimSize(1);
  return m >= 16 && m % 16 == 0 && n >= 16 && n % 16 == 0;
}

}  // namespace mlir::coralnpu_compiler
