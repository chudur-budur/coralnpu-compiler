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

#ifndef COMPILER_TRANSFORMS_DEVICEUTILS_H_
#define COMPILER_TRANSFORMS_DEVICEUTILS_H_

// IREE:
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"

// MLIR:
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::coralnpu_compiler {

struct DeviceAffinities {
  iree_compiler::IREE::HAL::DeviceAffinityAttr coralnpu;
  iree_compiler::IREE::HAL::DeviceAffinityAttr host;
};

// Runs HAL DeviceAnalysis on `moduleOp` and resolves the CoralNPU and host
// device affinities from the module's device globals and default stream
// affinity. Returns failure if DeviceAnalysis fails.
FailureOr<DeviceAffinities> lookupDeviceAffinities(ModuleOp moduleOp);

}  // namespace mlir::coralnpu_compiler

#endif  // COMPILER_TRANSFORMS_DEVICEUTILS_H_
