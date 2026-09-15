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

#ifndef RUNTIME_DRIVER_REGISTRATION_DRIVER_MODULE_H_
#define RUNTIME_DRIVER_REGISTRATION_DRIVER_MODULE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "runtime/driver/coralnpu_exec_backend.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

IREE_API_EXPORT iree_status_t
iree_hal_coralnpu_driver_module_register(iree_hal_driver_registry_t *registry);

// Overrides the execution backend that drivers created by this module run on;
// without an override the default simulator is used. Intended for tools that
// let the user pick a simulator. Must be called before any CoralNPU device is
// created.
IREE_API_EXPORT void iree_hal_coralnpu_driver_module_set_exec_backend(
    const iree_hal_coralnpu_exec_backend_t *exec_backend);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // RUNTIME_DRIVER_REGISTRATION_DRIVER_MODULE_H_
