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

// IREE:
#include "iree/compiler/Dialect/HAL/Analysis/DeviceAnalysis.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"

// MLIR:
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace mlir::coralnpu_compiler {

FailureOr<DeviceAffinities> lookupDeviceAffinities(ModuleOp moduleOp) {
  MLIRContext *context = moduleOp.getContext();

  IREE::HAL::DeviceAnalysis deviceAnalysis(moduleOp);
  if (failed(deviceAnalysis.run())) {
    return failure();
  }

  DeviceAffinities affinities;

  // Scan all device globals to find the CoralNPU and host device affinities.
  for (auto globalOp : deviceAnalysis.getDeviceGlobals()) {
    // Stop once both CoralNPU and host affinities have been found.
    if (affinities.coralnpu && affinities.host) break;
    auto deviceSet = deviceAnalysis.lookupDeviceTargets(globalOp);
    // Skip globals without resolved device targets.
    if (!deviceSet) continue;
    // Inspect each target configuration associated with this device global.
    for (auto targetAttr : deviceSet->getValues()) {
      auto deviceAffinity = IREE::HAL::DeviceAffinityAttr::get(
          context, SymbolRefAttr::get(globalOp.getGlobalName()),
          /*queue_mask=*/-1ll);
      // Record the CoralNPU device affinity, or fallback to host.
      if (targetAttr.getDeviceID().getValue() == "coralnpu") {
        // Keep the first matching CoralNPU device global.
        if (!affinities.coralnpu) affinities.coralnpu = deviceAffinity;
      } else if (!affinities.host) {
        // Keep the first non-CoralNPU device as the host fallback.
        affinities.host = deviceAffinity;
      }
    }
  }

  // Prefer the module's existing default affinity as the host device.
  if (auto defaultAffinity = dyn_cast_if_present<IREE::HAL::DeviceAffinityAttr>(
          IREE::Stream::AffinityAttr::lookupOrDefault(moduleOp))) {
    // Only use the default affinity as host if it is not the CoralNPU itself.
    if (defaultAffinity != affinities.coralnpu) {
      affinities.host = defaultAffinity;
    }
  }

  return affinities;
}

}  // namespace mlir::coralnpu_compiler
