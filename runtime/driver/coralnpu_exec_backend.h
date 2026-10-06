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
#include "runtime/metrics/dispatch_stats.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Vtable for a CoralNPU execution backend (functional simulator, RTL
// simulator, FPGA, or hardware). It runs dispatches of an executable the
// device has already loaded.
typedef struct iree_hal_coralnpu_exec_backend_t {
  void *self;
  // Name of the simulator or device the backend runs dispatches on (e.g.
  // "mpact"), reported in the statistics. Must outlive the backend.
  iree_string_view_t name;
  // Issues an inline dispatch on the execution backend. |out_stats| is NULL if
  // the caller collects no statistics, and the backend then skips its
  // measurements. Otherwise it is zero-initialized by the caller, and the
  // backend sets the fields it collects, also when the dispatch fails after it
  // started running.
  iree_status_t (*dispatch)(
      void *self, iree_const_byte_span_t dispatch_image,
      const iree_hal_executable_dispatch_state_v0_t *dispatch_state,
      const bool *binding_writeable, iree_host_size_t ordinal,
      iree_host_size_t local_memory_size,
      iree_hal_coralnpu_dispatch_stats_t *out_stats);
} iree_hal_coralnpu_exec_backend_t;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // RUNTIME_DRIVER_CORALNPU_EXEC_BACKEND_H_
