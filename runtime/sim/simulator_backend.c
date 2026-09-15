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

typedef struct iree_hal_coralnpu_simulator_context_t {
  coralnpu_simulator_create_fn_t factory;
  coralnpu_simulator_t *simulator;
} iree_hal_coralnpu_simulator_context_t;

static iree_status_t iree_hal_coralnpu_simulator_backend_create(
    void *self, iree_allocator_t host_allocator, void **out_context) {
  *out_context = NULL;
  coralnpu_simulator_create_fn_t factory = (coralnpu_simulator_create_fn_t)self;
  if (!factory) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "simulator factory is NULL");
  }
  coralnpu_simulator_t *sim = factory();
  if (!sim) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "failed to initialize CoralNPU device backend "
                            "(hardware device not connected or unavailable)");
  }
  iree_hal_coralnpu_simulator_context_t *ctx = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*ctx), (void **)&ctx);
  if (!iree_status_is_ok(status)) {
    coralnpu_simulator_destroy(sim);
    return status;
  }
  ctx->factory = factory;
  ctx->simulator = sim;
  *out_context = ctx;
  return iree_ok_status();
}

static void iree_hal_coralnpu_simulator_backend_destroy(
    void *self, iree_allocator_t host_allocator, void *context) {
  (void)self;
  if (!context) return;
  iree_hal_coralnpu_simulator_context_t *ctx =
      (iree_hal_coralnpu_simulator_context_t *)context;
  if (ctx->simulator) {
    coralnpu_simulator_destroy(ctx->simulator);
  }
  iree_allocator_free(host_allocator, ctx);
}

static iree_status_t iree_hal_coralnpu_simulator_backend_dispatch(
    void *self, void *context, iree_const_byte_span_t dispatch_image,
    const iree_hal_executable_dispatch_state_v0_t *dispatch_state,
    const bool *binding_writeable, iree_host_size_t ordinal,
    iree_byte_span_t local_memory) {
  (void)self;
  if (!context) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "device backend unavailable for dispatch");
  }
  iree_hal_coralnpu_simulator_context_t *ctx =
      (iree_hal_coralnpu_simulator_context_t *)context;
  if (ctx->factory) {
    if (ctx->simulator) {
      coralnpu_simulator_destroy(ctx->simulator);
      ctx->simulator = NULL;
    }
    ctx->simulator = ctx->factory();
  }
  if (!ctx->simulator) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "device backend unavailable for dispatch");
  }
  return iree_hal_simulator_issue_dispatch_inline(
      ctx->simulator, dispatch_image, dispatch_state, binding_writeable,
      ordinal, local_memory);
}

iree_hal_coralnpu_exec_backend_t iree_hal_coralnpu_simulator_backend_make(
    coralnpu_simulator_create_fn_t factory) {
  if (!factory) {
    iree_hal_coralnpu_exec_backend_t empty = {0};
    return empty;
  }
  iree_hal_coralnpu_exec_backend_t backend = {
      .self = (void *)factory,
      .create = iree_hal_coralnpu_simulator_backend_create,
      .destroy = iree_hal_coralnpu_simulator_backend_destroy,
      .dispatch = iree_hal_coralnpu_simulator_backend_dispatch,
  };
  return backend;
}
