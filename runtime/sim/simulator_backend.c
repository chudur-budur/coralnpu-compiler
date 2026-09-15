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

#include "runtime/sim/simulator_backend.h"

#include "runtime/sim/simulator_inline.h"

// Each dispatch runs on a fresh simulator.
static iree_status_t iree_hal_coralnpu_simulator_backend_dispatch(
    void *self, iree_const_byte_span_t dispatch_image,
    const iree_hal_executable_dispatch_state_v0_t *dispatch_state,
    const bool *binding_writeable, iree_host_size_t ordinal,
    iree_host_size_t local_memory_size) {
  coralnpu_simulator_t *sim = ((coralnpu_simulator_create_fn_t)self)();
  if (!sim) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "failed to initialize CoralNPU device backend "
                            "(hardware device not connected or unavailable)");
  }
  iree_status_t status = iree_hal_simulator_issue_dispatch_inline(
      sim, dispatch_image, dispatch_state, binding_writeable, ordinal,
      local_memory_size);
  coralnpu_simulator_destroy(sim);
  return status;
}

iree_hal_coralnpu_exec_backend_t iree_hal_coralnpu_simulator_backend_make(
    coralnpu_simulator_create_fn_t factory) {
  iree_hal_coralnpu_exec_backend_t backend = {
      .self = (void *)factory,
      .dispatch = iree_hal_coralnpu_simulator_backend_dispatch,
  };
  return backend;
}
