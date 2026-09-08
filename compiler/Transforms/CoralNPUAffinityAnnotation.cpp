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

#include <algorithm>
#include <cmath>

#include "compiler/Target/Utils.h"
#include "compiler/Transforms/DeviceUtils.h"
#include "compiler/Transforms/Passes.h"

// IREE
#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"

// MLIR
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Pass/Pass.h"

// LLVM
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace mlir::coralnpu_compiler {

#define GEN_PASS_DEF_CORALNPUAFFINITYANNOTATION
#include "compiler/Transforms/Passes.h.inc"

namespace {

bool isSupportedType(
    Type type, const IREE::HAL::TargetBackend::SupportedTypes &supportedTypes) {
  // Shaped types must have static shapes and a supported element type.
  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    return shapedType.hasStaticShape() &&
           supportedTypes.supportsElementType(shapedType.getElementType());
  }

  return supportedTypes.supportsScalarType(type);
}

bool isSupportedOperandAndResultTypes(
    Operation *op,
    const IREE::HAL::TargetBackend::SupportedTypes &supportedTypes) {
  auto isSupported = [&](Type t) { return isSupportedType(t, supportedTypes); };
  return llvm::all_of(op->getOperandTypes(), isSupported) &&
         llvm::all_of(op->getResultTypes(), isSupported);
}

FailureOr<uint64_t> estimateBytesForType(Type type) {
  // NB: pay attention to sub-byte types (i.e. divide by 8 only at the very
  // end).

  Type elementType = getElementTypeOrSelf(type);
  uint64_t bits = 0;
  // Determine the bitwidth of the (element) type.
  if (auto indexType = llvm::dyn_cast<IndexType>(elementType)) {
    bits = indexType.kInternalStorageBitWidth;
  } else if (elementType.isIntOrFloat()) {
    bits = elementType.getIntOrFloatBitWidth();
  } else {
    return failure();
  }

  // Compute bit size for shaped types from element count and bitwidth.
  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    // Dynamic shapes have unbounded size.
    if (!shapedType.hasStaticShape()) return failure();

    bits *= shapedType.getNumElements();
  }

  return llvm::divideCeil(bits, 8);
}

FailureOr<uint64_t> estimateIOBytes(
    IREE::Flow::DispatchWorkgroupsOp workgroupsOp) {
  uint64_t totalBytes = 0;

  // Accumulate the byte size of all dispatch input arguments.
  for (Value arg : workgroupsOp.getArguments()) {
    auto bytes = estimateBytesForType(arg.getType());
    if (failed(bytes)) return failure();
    totalBytes += *bytes;
  }

  // Accumulate the byte size of all dispatch results.
  for (Value result : workgroupsOp.getResults()) {
    auto bytes = estimateBytesForType(result.getType());
    if (failed(bytes)) return failure();
    totalBytes += *bytes;
  }

  return totalBytes;
}

// Iterations x non-cast body ops (at least one); casts are skipped so mixed
// precision is not inflated.
double estimateComputeOps(linalg::LinalgOp linalgOp) {
  double ops = 1.0;
  for (int64_t range : linalgOp.getStaticLoopRanges()) {
    if (range <= 0) return 0.0;  // Dynamic or empty.
    ops *= range;
  }
  int64_t opsPerIter =
      llvm::count_if(linalgOp.getBlock()->without_terminator(),
                     [](Operation &op) { return !isa<CastOpInterface>(&op); });
  return ops * std::max<int64_t>(1, opsPerIter);
}

// Greedy roofline placement for the unified-memory SoC: CoralNPU if
// T_cpu / T_npu >= threshold; inputs from the other device cross m_axi.
// Constants: see README.md.
// TODO: calibrate against profiled dispatches; charge result copies.
bool preferCoralNPU(IREE::Flow::DispatchWorkgroupsOp workgroupsOp,
                    linalg::LinalgOp rootOp, ArrayRef<Operation *> computeOps,
                    const DenseSet<Value> &npuValues, double speedupThreshold) {
  constexpr double kNpuGflops = 128.0;
  constexpr double kNpuWindowGBps = 16.0;
  constexpr double kNpuSpillGBps = 2.0;  // Bytes beyond the window.
  constexpr double kNpuWindowBytes = 4 * 1024 * 1024;  // EXTMEM window.
  constexpr double kNpuLaunchNs = 800.0;
  constexpr double kCpuGflops = 8.0;
  constexpr double kCpuGBps = 4.0;
  // Host <-> CoralNPU over m_axi (4-byte beats, 3.2 GB/s peak at 800 MHz).
  constexpr double kCopyGBps = 2.0;

  FailureOr<uint64_t> bytes = estimateIOBytes(workgroupsOp);
  if (failed(bytes)) return false;
  // Inputs not produced on CoralNPU (e.g. function arguments) are on the host,
  // except immutable weights and constants, which are placed where used.
  double npuCopyBytes = 0.0, cpuCopyBytes = 0.0;
  for (Value arg : workgroupsOp.getArguments()) {
    if (!isa<ShapedType>(arg.getType())) continue;
    double argBytes = *estimateBytesForType(arg.getType());  // Checked above.
    while (auto reshape = arg.getDefiningOp<IREE::Flow::TensorReshapeOp>())
      arg = reshape.getSource();
    Operation *producer = arg.getDefiningOp();
    auto load =
        dyn_cast_if_present<IREE::Util::GlobalLoadOpInterface>(producer);
    if ((load && load.isGlobalImmutable()) ||
        (producer && producer->hasTrait<OpTrait::ConstantLike>()))
      continue;
    (npuValues.contains(arg) ? cpuCopyBytes : npuCopyBytes) += argBytes;
  }
  double ops = 0.0;
  for (Operation *op : computeOps) {
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op))
      ops += estimateComputeOps(linalgOp);
  }

  // The narrowest root input sets the tier: 16-bit runs 2x and 8-bit 4x
  // faster; the host has no BF16 extension, so bf16 runs at the f32 rate there.
  unsigned bitWidth = 32;
  bool hasBF16Input = false;
  for (Value input : rootOp.getDpsInputs()) {
    Type type = getElementTypeOrSelf(input.getType());
    if (type.isIntOrFloat())
      bitWidth = std::min(bitWidth, type.getIntOrFloatBitWidth());
    hasBF16Input |= type.isBF16();
  }
  double npuScale = bitWidth <= 8 ? 4.0 : (bitWidth <= 16 ? 2.0 : 1.0);
  double cpuScale = hasBF16Input ? 1.0 : npuScale;

  // Latencies in ns.
  double npuMemNs =
      std::min<double>(*bytes, kNpuWindowBytes) / kNpuWindowGBps +
      std::max<double>(*bytes - kNpuWindowBytes, 0) / kNpuSpillGBps;
  double npuNs = kNpuLaunchNs + npuCopyBytes / kCopyGBps +
                 std::max(ops / npuScale / kNpuGflops, npuMemNs);
  double cpuNs = cpuCopyBytes / kCopyGBps +
                 std::max(ops / cpuScale / kCpuGflops, *bytes / kCpuGBps);
  return cpuNs >= speedupThreshold * npuNs;
}

struct CoralNPUAffinityAnnotationPass
    : public impl::CoralNPUAffinityAnnotationBase<
          CoralNPUAffinityAnnotationPass> {
  using CoralNPUAffinityAnnotationBase::CoralNPUAffinityAnnotationBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *context = &getContext();

    if (!std::isfinite(rooflineSpeedupThreshold) ||
        rooflineSpeedupThreshold < 0.0) {
      moduleOp.emitError(
          "roofline-speedup-threshold must be a finite, non-negative value");
      return signalPassFailure();
    }

    FailureOr<DeviceAffinities> affinities = lookupDeviceAffinities(moduleOp);
    if (failed(affinities)) {
      return signalPassFailure();
    }

    // Nothing to annotate if no CoralNPU device is targeted.
    if (!affinities->coralnpu) return;

    IREE::HAL::TargetBackend::SupportedTypes supportedTypes =
        getCoralNPUSupportedTypes(context);

    // Results of dispatches placed on CoralNPU so far, in program order.
    DenseSet<Value> npuValues;
    moduleOp.walk([&](IREE::Flow::DispatchWorkgroupsOp workgroupsOp) {
      // If op already has affinity, don't change it
      if (Attribute affinity = workgroupsOp->getAttr("stream.affinity")) {
        if (affinity == affinities->coralnpu)
          npuValues.insert(workgroupsOp->result_begin(),
                           workgroupsOp->result_end());
        return;
      }

      SmallVector<Operation *> computeOps;
      workgroupsOp.getWorkgroupBody().walk([&](Operation *op) {
        // Collect all compute operations in the dispatch body.
        if (iree_compiler::isComputeOp(op)) {
          computeOps.push_back(op);
        }
      });

      FailureOr<Operation *> rootOp = getRootOperation(computeOps);
      bool canRunOnNPU =
          succeeded(rootOp) && *rootOp && isa<linalg::LinalgOp>(*rootOp) &&
          isSupportedOperandAndResultTypes(*rootOp, supportedTypes);

      // Route eligible dispatches by the roofline model, or fall back to host.
      if (canRunOnNPU) {
        // Warn if any non-root compute operation has unsupported types.
        if (llvm::any_of(computeOps, [&](Operation *op) {
              return op != *rootOp &&
                     !isSupportedOperandAndResultTypes(op, supportedTypes);
            })) {
          workgroupsOp.emitWarning(
              "non-root linalg operation has types not supported by CoralNPU "
              "in a dispatch eligible for CoralNPU");
        }

        if (!affinities->host ||
            preferCoralNPU(workgroupsOp, cast<linalg::LinalgOp>(*rootOp),
                           computeOps, npuValues, rooflineSpeedupThreshold)) {
          workgroupsOp->setAttr("stream.affinity", affinities->coralnpu);
          npuValues.insert(workgroupsOp->result_begin(),
                           workgroupsOp->result_end());
          return;
        }
      }
      if (affinities->host) {
        workgroupsOp->setAttr("stream.affinity", affinities->host);
      }
    });
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
createCoralNPUAffinityAnnotationPass() {
  return std::make_unique<CoralNPUAffinityAnnotationPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createCoralNPUAffinityAnnotationPass(
    CoralNPUAffinityAnnotationOptions options) {
  return std::make_unique<CoralNPUAffinityAnnotationPass>(std::move(options));
}

}  // namespace mlir::coralnpu_compiler
