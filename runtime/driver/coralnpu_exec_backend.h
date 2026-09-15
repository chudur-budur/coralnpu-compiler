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

#ifndef RUNTIME_DRIVER_CORALNPU_EXEC_BACKEND_H_
#define RUNTIME_DRIVER_CORALNPU_EXEC_BACKEND_H_

#include <stdbool.h>

#include "iree/base/api.h"
#include "iree/hal/local/executable_library.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Vtable for a CoralNPU execution backend (functional simulator, RTL
// simulator, FPGA, or hardware). It runs dispatches of an executable the
// device has already loaded.
typedef struct iree_hal_coralnpu_exec_backend_t {
  void *self;
  // Issues an inline dispatch on the execution backend.
  iree_status_t (*dispatch)(
      void *self, iree_const_byte_span_t dispatch_image,
      const iree_hal_executable_dispatch_state_v0_t *dispatch_state,
      const bool *binding_writeable, iree_host_size_t ordinal,
      iree_host_size_t local_memory_size);
} iree_hal_coralnpu_exec_backend_t;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // RUNTIME_DRIVER_CORALNPU_EXEC_BACKEND_H_
