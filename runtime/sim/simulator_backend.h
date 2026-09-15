/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef RUNTIME_SIM_SIMULATOR_BACKEND_H_
#define RUNTIME_SIM_SIMULATOR_BACKEND_H_

#include "runtime/driver/coralnpu_exec_backend.h"
#include "runtime/sim/simulator_api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Constructs an iree_hal_coralnpu_exec_backend_t wrapping the given simulator
// factory function. If |factory| is NULL, returns a zero-initialized backend.
iree_hal_coralnpu_exec_backend_t iree_hal_coralnpu_simulator_backend_make(
    coralnpu_simulator_create_fn_t factory);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // RUNTIME_SIM_SIMULATOR_BACKEND_H_
