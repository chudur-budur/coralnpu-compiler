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

#ifndef RUNTIME_DRIVER_CORALNPU_EXECUTABLE_H_
#define RUNTIME_DRIVER_CORALNPU_EXECUTABLE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Returns true if |executable_format| is supported by the CoralNPU driver.
static inline bool iree_hal_coralnpu_is_executable_format(
    iree_string_view_t executable_format) {
  return iree_string_view_equal(executable_format,
                                IREE_SV("embedded-elf-riscv_32"));
}

// Creates an executable containing a CoralNPU dispatch image.
iree_status_t iree_hal_coralnpu_executable_create(
    const iree_hal_executable_params_t *executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t **out_executable);

// Returns true if |base_executable| is a CoralNPU executable.
bool iree_hal_coralnpu_executable_isa(iree_hal_executable_t *base_executable);

// Returns the dispatch image byte span stored in |base_executable|.
iree_const_byte_span_t iree_hal_coralnpu_executable_dispatch_image(
    iree_hal_executable_t *base_executable);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // RUNTIME_DRIVER_CORALNPU_EXECUTABLE_H_
