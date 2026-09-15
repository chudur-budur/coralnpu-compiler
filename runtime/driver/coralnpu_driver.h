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

#ifndef RUNTIME_DRIVER_CORALNPU_DRIVER_H_
#define RUNTIME_DRIVER_CORALNPU_DRIVER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "runtime/driver/coralnpu_device.h"
#include "runtime/driver/coralnpu_exec_backend.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a new CoralNPU driver exposing a single device.
// |exec_backend| is required and supplies the execution backend vtable
// used by created devices.
iree_status_t iree_hal_coralnpu_driver_create(
    iree_string_view_t identifier,
    const iree_hal_coralnpu_device_params_t* default_params,
    const iree_hal_coralnpu_exec_backend_t* exec_backend,
    iree_hal_allocator_t* device_allocator, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // RUNTIME_DRIVER_CORALNPU_DRIVER_H_
