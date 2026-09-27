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

#include <memory>

#include "compiler/Transforms/DeviceUtils.h"
#include "compiler/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir::iree_compiler;

namespace mlir::coralnpu_compiler {

#define GEN_PASS_DEF_CORALNPUMATERIALIZEDEVICETOPOLOGY
#include "compiler/Transforms/Passes.h.inc"

namespace {

struct CoralNPUMaterializeDeviceTopologyPass
    : public impl::CoralNPUMaterializeDeviceTopologyBase<
          CoralNPUMaterializeDeviceTopologyPass> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *context = &getContext();

    FailureOr<DeviceAffinities> affinities = lookupDeviceAffinities(moduleOp);
    if (failed(affinities)) {
      return signalPassFailure();
    }

    // Nothing to materialize if no CoralNPU or host device is targeted.
    if (!affinities->coralnpu || !affinities->host) return;

    // Attach bidirectional unified-memory topology links between host and NPU.
    if (!moduleOp->hasAttr("stream.topology")) {
      auto host = affinities->host.getDevice(),
           npu = affinities->coralnpu.getDevice();
      IREE::HAL::DeviceLinkAttr links[] = {
          IREE::HAL::DeviceLinkAttr::get(context, host, npu,
                                         /*unified_memory=*/true,
                                         /*transparent_access=*/true, nullptr),
          IREE::HAL::DeviceLinkAttr::get(context, npu, host,
                                         /*unified_memory=*/true,
                                         /*transparent_access=*/true, nullptr)};
      moduleOp->setAttr("stream.topology",
                        IREE::HAL::DeviceTopologyAttr::get(context, links));
    }

    // Set the host device as the default stream affinity if none is specified.
    if (!moduleOp->hasAttr("stream.affinity.default")) {
      moduleOp->setAttr("stream.affinity.default", affinities->host);
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
createCoralNPUMaterializeDeviceTopologyPass() {
  return std::make_unique<CoralNPUMaterializeDeviceTopologyPass>();
}

}  // namespace mlir::coralnpu_compiler
