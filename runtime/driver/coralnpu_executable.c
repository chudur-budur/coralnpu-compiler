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

#include "runtime/driver/coralnpu_executable.h"

#include <string.h>

// The CoralNPU dispatch image currently holds exactly one export.
#define IREE_HAL_CORALNPU_EXECUTABLE_EXPORT_COUNT 1

typedef struct iree_hal_coralnpu_executable_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_const_byte_span_t dispatch_image;
} iree_hal_coralnpu_executable_t;

static const iree_hal_executable_vtable_t iree_hal_coralnpu_executable_vtable;

static iree_hal_coralnpu_executable_t *iree_hal_coralnpu_executable_cast(
    iree_hal_executable_t *base_executable) {
  IREE_HAL_ASSERT_TYPE(base_executable, &iree_hal_coralnpu_executable_vtable);
  return (iree_hal_coralnpu_executable_t *)base_executable;
}

bool iree_hal_coralnpu_executable_isa(iree_hal_executable_t *base_executable) {
  return iree_hal_resource_is(base_executable,
                              &iree_hal_coralnpu_executable_vtable);
}

iree_const_byte_span_t iree_hal_coralnpu_executable_dispatch_image(
    iree_hal_executable_t *base_executable) {
  iree_hal_coralnpu_executable_t *executable =
      iree_hal_coralnpu_executable_cast(base_executable);
  return executable->dispatch_image;
}

static void iree_hal_coralnpu_executable_destroy(
    iree_hal_executable_t *base_executable) {
  iree_hal_coralnpu_executable_t *executable =
      iree_hal_coralnpu_executable_cast(base_executable);
  iree_allocator_t host_allocator = executable->host_allocator;
  iree_allocator_free(host_allocator, executable);
}

static iree_host_size_t iree_hal_coralnpu_executable_export_count(
    iree_hal_executable_t *base_executable) {
  return IREE_HAL_CORALNPU_EXECUTABLE_EXPORT_COUNT;
}

static iree_status_t iree_hal_coralnpu_executable_export_info(
    iree_hal_executable_t *base_executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_hal_executable_export_info_t *out_info) {
  if (!out_info) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "out_info is null");
  }
  if (export_ordinal >= IREE_HAL_CORALNPU_EXECUTABLE_EXPORT_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "export ordinal out of range");
  }

  memset(out_info, 0, sizeof(*out_info));
  return iree_ok_status();
}

static iree_status_t iree_hal_coralnpu_executable_export_parameters(
    iree_hal_executable_t *base_executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_host_size_t capacity,
    iree_hal_executable_export_parameter_t *out_parameters) {
  if (export_ordinal >= IREE_HAL_CORALNPU_EXECUTABLE_EXPORT_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "export ordinal out of range");
  }
  if (capacity > 0 && !out_parameters) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_parameters is null");
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_coralnpu_executable_lookup_export_by_name(
    iree_hal_executable_t *base_executable, iree_string_view_t name,
    iree_hal_executable_export_ordinal_t *out_export_ordinal) {
  return iree_make_status(IREE_STATUS_NOT_FOUND, "export lookup unsupported");
}

static const iree_hal_executable_vtable_t iree_hal_coralnpu_executable_vtable =
    {
        .destroy = iree_hal_coralnpu_executable_destroy,
        .export_count = iree_hal_coralnpu_executable_export_count,
        .export_info = iree_hal_coralnpu_executable_export_info,
        .export_parameters = iree_hal_coralnpu_executable_export_parameters,
        .lookup_export_by_name =
            iree_hal_coralnpu_executable_lookup_export_by_name,
};

iree_status_t iree_hal_coralnpu_executable_create(
    const iree_hal_executable_params_t *executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t **out_executable) {
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;

  if (!executable_params->executable_data.data ||
      executable_params->executable_data.data_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "empty executable_data");
  }

  const iree_host_size_t image_size =
      executable_params->executable_data.data_length;
  iree_hal_coralnpu_executable_t *executable = NULL;
  iree_host_size_t total_size = 0;
  iree_host_size_t image_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(*executable), &total_size,
      IREE_STRUCT_FIELD_ALIGNED(image_size, uint8_t, 1, &image_offset)));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
      host_allocator, total_size, (void **)&executable));

  iree_hal_resource_initialize(&iree_hal_coralnpu_executable_vtable,
                               &executable->resource);
  executable->host_allocator = host_allocator;

  uint8_t *image_storage = (uint8_t *)executable + image_offset;
  memcpy(image_storage, executable_params->executable_data.data, image_size);

  executable->dispatch_image =
      iree_make_const_byte_span(image_storage, image_size);

  *out_executable = (iree_hal_executable_t *)executable;
  return iree_ok_status();
}
