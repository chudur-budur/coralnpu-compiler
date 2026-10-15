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

#include "iree/base/threading/call_once.h"
#include "iree/base/threading/mutex.h"
#include "runtime/sim/simulator_inline.h"

// Serializes dispatches, which share process-wide state: the MPACT simulator
// and the read-only binding and ELF layout caches.
static iree_once_flag iree_hal_coralnpu_simulator_mutex_once =
    IREE_ONCE_FLAG_INIT;
static iree_slim_mutex_t iree_hal_coralnpu_simulator_mutex;

static void iree_hal_coralnpu_simulator_mutex_initialize(void) {
  iree_slim_mutex_initialize(&iree_hal_coralnpu_simulator_mutex);
}

// MPACT reuses one simulator; other backends get a fresh one per dispatch.
static iree_status_t iree_hal_coralnpu_simulator_backend_dispatch_locked(
    void *self, iree_const_byte_span_t dispatch_image,
    const iree_hal_executable_dispatch_state_v0_t *dispatch_state,
    const bool *binding_writeable, iree_host_size_t ordinal,
    iree_host_size_t local_memory_size,
    iree_hal_coralnpu_dispatch_stats_t *out_stats) {
  coralnpu_simulator_create_fn_t factory = (coralnpu_simulator_create_fn_t)self;
  static coralnpu_simulator_t *cached_mpact_sim = NULL;
  const bool reuse_sim = (factory == coralnpu_simulator_mpact_create);
  coralnpu_simulator_t *sim =
      reuse_sim && cached_mpact_sim ? cached_mpact_sim : factory();
  // The FPGA factory returns NULL when no device is connected.
  if (!sim) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "failed to initialize CoralNPU device backend "
                            "(hardware device not connected or unavailable)");
  }
  if (reuse_sim) cached_mpact_sim = sim;
  iree_status_t status = iree_hal_simulator_issue_dispatch_inline(
      sim, dispatch_image, dispatch_state, binding_writeable, ordinal,
      local_memory_size, out_stats);
  if (reuse_sim && iree_status_is_ok(status)) return status;
  if (reuse_sim) cached_mpact_sim = NULL;
  iree_hal_simulator_inline_forget(sim);
  coralnpu_simulator_destroy(sim);
  return status;
}

static iree_status_t iree_hal_coralnpu_simulator_backend_dispatch(
    void *self, iree_const_byte_span_t dispatch_image,
    const iree_hal_executable_dispatch_state_v0_t *dispatch_state,
    const bool *binding_writeable, iree_host_size_t ordinal,
    iree_host_size_t local_memory_size,
    iree_hal_coralnpu_dispatch_stats_t *out_stats) {
  iree_call_once(&iree_hal_coralnpu_simulator_mutex_once,
                 iree_hal_coralnpu_simulator_mutex_initialize);
  iree_slim_mutex_lock(&iree_hal_coralnpu_simulator_mutex);
  iree_status_t status = iree_hal_coralnpu_simulator_backend_dispatch_locked(
      self, dispatch_image, dispatch_state, binding_writeable, ordinal,
      local_memory_size, out_stats);
  iree_slim_mutex_unlock(&iree_hal_coralnpu_simulator_mutex);
  return status;
}

iree_hal_coralnpu_exec_backend_t iree_hal_coralnpu_simulator_backend_make(
    coralnpu_simulator_create_fn_t factory, iree_string_view_t name) {
  iree_hal_coralnpu_exec_backend_t backend = {
      .self = (void *)factory,
      .name = name,
      .dispatch = iree_hal_coralnpu_simulator_backend_dispatch,
  };
  return backend;
}
