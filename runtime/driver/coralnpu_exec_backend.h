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
#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/local/executable_library.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Vtable defining operations for a CoralNPU execution backend (e.g. functional
// simulator, RTL simulator, FPGA, or physical hardware). Distinct from
// iree_hal_executable_loader_t, which devices also take: a loader turns
// executable bytes into an iree_hal_executable_t, while a backend runs the
// dispatches within one.
typedef struct iree_hal_coralnpu_exec_backend_t {
  void *self;
  // Creates a per-device execution backend context.
  iree_status_t (*create)(void *self, iree_allocator_t host_allocator,
                          void **out_context);
  // Destroys a per-device execution backend context.
  void (*destroy)(void *self, iree_allocator_t host_allocator, void *context);
  // Issues an inline dispatch on the execution backend context.
  iree_status_t (*dispatch)(
      void *self, void *context, iree_const_byte_span_t dispatch_image,
      const iree_hal_executable_dispatch_state_v0_t *dispatch_state,
      const bool *binding_writeable, iree_host_size_t ordinal,
      iree_byte_span_t local_memory);
} iree_hal_coralnpu_exec_backend_t;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // RUNTIME_DRIVER_CORALNPU_EXEC_BACKEND_H_
