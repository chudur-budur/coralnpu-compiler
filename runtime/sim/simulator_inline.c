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

#include "runtime/sim/simulator_inline.h"

#ifdef CORALNPU_SIMULATOR_PROFILE
#include <stdio.h>
#endif  // CORALNPU_SIMULATOR_PROFILE

#include <inttypes.h>

#include "crt/coralnpu_dispatch.h"
#include "iree/base/api.h"
#include "runtime/sim/simulator_api.h"
#include "runtime/sim/simulator_elf_loader.h"

// Bump-allocates |size| bytes aligned to |alignment| below |limit|.
static iree_status_t iree_hal_coralnpu_allocate(uint32_t* cursor,
                                                uint64_t limit,
                                                uint32_t alignment, size_t size,
                                                const char* region,
                                                uint32_t* out_address) {
  uint64_t aligned =
      ((uint64_t)*cursor + alignment - 1u) & ~((uint64_t)alignment - 1u);
  uint64_t allocation_end = aligned + size;

  if (allocation_end < aligned || allocation_end > limit) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "dispatch data exceeds the %s", region);
  }

  *out_address = (uint32_t)aligned;
  *cursor = (uint32_t)allocation_end;
  return iree_ok_status();
}

iree_status_t iree_hal_simulator_issue_dispatch_inline(
    coralnpu_simulator_t* sim, iree_const_byte_span_t dispatch_image,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const bool* binding_writeable, iree_host_size_t ordinal,
    iree_host_size_t local_memory_size) {
  IREE_ASSERT_ARGUMENT(dispatch_state);
  IREE_ASSERT_LE(dispatch_state->binding_count,
                 IREE_HAL_EXECUTABLE_MAX_BINDING_COUNT);

  iree_hal_coralnpu_simulator_elf_layout_t elf_layout;

  IREE_RETURN_IF_ERROR(iree_hal_coralnpu_simulator_load_elf_with_layout(
      sim, dispatch_image, &elf_layout));

  coralnpu_dispatch_request_t request = {0};

  if (elf_layout.dispatch_request_size != sizeof(request)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "CoralNPU dispatch ABI size mismatch: ELF=%" PRIu32
                            " runtime=%zu",
                            elf_layout.dispatch_request_size, sizeof(request));
  }

  request.magic = CORALNPU_DISPATCH_MAGIC;
  request.version = CORALNPU_DISPATCH_VERSION;
  request.status = CORALNPU_DISPATCH_STATUS_READY;

  request.workgroup_size_x = dispatch_state->workgroup_size_x;
  request.workgroup_size_y = dispatch_state->workgroup_size_y;
  request.workgroup_size_z = dispatch_state->workgroup_size_z;

  request.workgroup_count_x = dispatch_state->workgroup_count_x;
  request.workgroup_count_y = dispatch_state->workgroup_count_y;
  request.workgroup_count_z = dispatch_state->workgroup_count_z;

  request.max_concurrency = dispatch_state->max_concurrency;
  request.push_constant_count = dispatch_state->constant_count;
  request.binding_count = dispatch_state->binding_count;
  request.ordinal = (uint32_t)ordinal;

  uint32_t heap_cursor = elf_layout.heap_start_addr;

  const size_t constants_size =
      (size_t)dispatch_state->constant_count * sizeof(uint32_t);
  if (dispatch_state->constant_count != 0) {
    IREE_RETURN_IF_ERROR(iree_hal_coralnpu_allocate(
        &heap_cursor, elf_layout.heap_end_addr, 4, constants_size,
        "firmware heap", &request.push_constants_addr));
    coralnpu_simulator_write_mem(sim, request.push_constants_addr,
                                 dispatch_state->constants, constants_size);
  }

  const size_t binding_table_size =
      (size_t)dispatch_state->binding_count * sizeof(uint32_t);
  if (dispatch_state->binding_count != 0) {
    IREE_RETURN_IF_ERROR(iree_hal_coralnpu_allocate(
        &heap_cursor, elf_layout.heap_end_addr, 4, binding_table_size,
        "firmware heap", &request.binding_ptrs_addr));

    IREE_RETURN_IF_ERROR(iree_hal_coralnpu_allocate(
        &heap_cursor, elf_layout.heap_end_addr, 4, binding_table_size,
        "firmware heap", &request.binding_lengths_addr));
  }

  if (local_memory_size != 0) {
    IREE_RETURN_IF_ERROR(iree_hal_coralnpu_allocate(
        &heap_cursor, elf_layout.heap_end_addr, 64, local_memory_size,
        "firmware heap", &request.local_memory_addr));

    iree_hal_coralnpu_simulator_zero_mem(sim, request.local_memory_addr,
                                         local_memory_size);
  }

  // Bindings are staged in the 1 GB simulated DDR region at 0x80000000.
  uint32_t ddr_cursor = 0x80000000u;
  uint32_t binding_addresses[IREE_HAL_EXECUTABLE_MAX_BINDING_COUNT] = {0};
  uint32_t binding_sizes[IREE_HAL_EXECUTABLE_MAX_BINDING_COUNT] = {0};

  for (uint32_t i = 0; i < dispatch_state->binding_count; ++i) {
    void* binding_ptr = dispatch_state->binding_ptrs[i];
    size_t binding_length = dispatch_state->binding_lengths[i];

    uint32_t binding_address = 0;
    for (uint32_t j = 0; j < i; ++j) {
      // Reuse a slot only if it covers this binding.
      if (dispatch_state->binding_ptrs[j] == binding_ptr &&
          dispatch_state->binding_lengths[j] >= binding_length) {
        binding_address = binding_addresses[j];
        break;
      }
    }

    if (binding_address == 0) {
      IREE_RETURN_IF_ERROR(iree_hal_coralnpu_allocate(
          &ddr_cursor, 0xC0000000u, 64, binding_length, "DDR region",
          &binding_address));
      if (binding_length != 0) {
        coralnpu_simulator_write_mem(sim, binding_address, binding_ptr,
                                     binding_length);
      }
    }

    binding_addresses[i] = binding_address;
    binding_sizes[i] = (uint32_t)binding_length;
  }

  if (dispatch_state->binding_count != 0) {
    coralnpu_simulator_write_mem(sim, request.binding_ptrs_addr,
                                 binding_addresses, binding_table_size);
    coralnpu_simulator_write_mem(sim, request.binding_lengths_addr,
                                 binding_sizes, binding_table_size);
  }

  coralnpu_simulator_write_mem(sim, elf_layout.dispatch_request_addr, &request,
                               sizeof(request));

#ifdef CORALNPU_SIMULATOR_PROFILE
  const uint64_t cycle_start = coralnpu_simulator_get_cycle_count(sim);
#endif  // CORALNPU_SIMULATOR_PROFILE

  if (!coralnpu_simulator_run(sim, elf_layout.start_pc)) {
    return iree_make_status(IREE_STATUS_INTERNAL, "CoralNPU core did not halt");
  }

#ifdef CORALNPU_SIMULATOR_PROFILE
  const uint64_t cycle_end = coralnpu_simulator_get_cycle_count(sim);
  fprintf(stderr,
          "[CoralNPU simulator] execution returned: %" PRIu64
          " cycles (total=%" PRIu64 ")\n",
          cycle_end - cycle_start, cycle_end);
  fflush(stderr);
#endif  // CORALNPU_SIMULATOR_PROFILE

  coralnpu_simulator_read_mem(sim, elf_layout.dispatch_request_addr, &request,
                              sizeof(request));

  if (request.magic != CORALNPU_DISPATCH_MAGIC ||
      request.version != CORALNPU_DISPATCH_VERSION) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "firmware corrupted the CoralNPU dispatch request");
  }

  if (request.status != CORALNPU_DISPATCH_STATUS_COMPLETE) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "firmware dispatch did not complete: status=%u return_code=%" PRId32,
        (unsigned)request.status, request.return_code);
  }

  for (uint32_t i = 0; i < dispatch_state->binding_count; ++i) {
    if (!binding_writeable[i] || binding_sizes[i] == 0) {
      continue;
    }
    // Skip if an earlier writeable binding already read back this range.
    bool already_read = false;
    for (uint32_t j = 0; j < i; ++j) {
      if (binding_writeable[j] && binding_sizes[j] >= binding_sizes[i] &&
          binding_addresses[j] == binding_addresses[i]) {
        already_read = true;
        break;
      }
    }
    if (!already_read) {
      coralnpu_simulator_read_mem(sim, binding_addresses[i],
                                  dispatch_state->binding_ptrs[i],
                                  binding_sizes[i]);
    }
  }

  return iree_ok_status();
}
