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

#include "runtime/driver/coralnpu_driver.h"

#include <inttypes.h>
#include <stddef.h>
#include <string.h>

// The driver exposes a single device; IREE reserves 0 for the default device.
#define IREE_HAL_CORALNPU_DEVICE_ID 1

typedef struct iree_hal_coralnpu_driver_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_hal_allocator_t* device_allocator;

  iree_string_view_t identifier;
  iree_hal_coralnpu_device_params_t default_params;
  iree_hal_coralnpu_exec_backend_t exec_backend;
} iree_hal_coralnpu_driver_t;

static const iree_hal_driver_vtable_t iree_hal_coralnpu_driver_vtable;

static iree_hal_coralnpu_driver_t* iree_hal_coralnpu_driver_cast(
    iree_hal_driver_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_coralnpu_driver_vtable);
  return (iree_hal_coralnpu_driver_t*)base_value;
}

iree_status_t iree_hal_coralnpu_driver_create(
    iree_string_view_t identifier,
    const iree_hal_coralnpu_device_params_t* default_params,
    const iree_hal_coralnpu_exec_backend_t* exec_backend,
    iree_hal_allocator_t* device_allocator, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(default_params);
  IREE_ASSERT_ARGUMENT(exec_backend);
  IREE_ASSERT_ARGUMENT(device_allocator);
  IREE_ASSERT_ARGUMENT(out_driver);
  *out_driver = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_coralnpu_driver_t* driver = NULL;
  iree_host_size_t total_size = 0;
  iree_host_size_t identifier_offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(sizeof(*driver), &total_size,
                             IREE_STRUCT_FIELD_ALIGNED(identifier.size, char, 1,
                                                       &identifier_offset)));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&driver));
  memset(driver, 0, total_size);
  iree_hal_resource_initialize(&iree_hal_coralnpu_driver_vtable,
                               &driver->resource);
  driver->host_allocator = host_allocator;
  driver->device_allocator = device_allocator;
  driver->exec_backend = *exec_backend;
  iree_hal_allocator_retain(device_allocator);

  iree_string_view_append_to_buffer(identifier, &driver->identifier,
                                    (char*)driver + identifier_offset);
  memcpy(&driver->default_params, default_params,
         sizeof(driver->default_params));

  *out_driver = (iree_hal_driver_t*)driver;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_coralnpu_driver_destroy(iree_hal_driver_t* base_driver) {
  iree_hal_coralnpu_driver_t* driver =
      iree_hal_coralnpu_driver_cast(base_driver);
  iree_allocator_t host_allocator = driver->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_allocator_release(driver->device_allocator);

  iree_allocator_free(host_allocator, driver);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_coralnpu_driver_query_available_devices(
    iree_hal_driver_t* base_driver, iree_allocator_t host_allocator,
    iree_host_size_t* out_device_info_count,
    iree_hal_device_info_t** out_device_infos) {
  iree_hal_coralnpu_driver_t* driver =
      iree_hal_coralnpu_driver_cast(base_driver);

  const iree_hal_device_info_t device_info = {
      .device_id = IREE_HAL_CORALNPU_DEVICE_ID,
      .path = iree_string_view_empty(),
      .name = driver->identifier,
  };

  *out_device_info_count = 1;
  return iree_allocator_clone(
      host_allocator,
      iree_make_const_byte_span(&device_info, sizeof(device_info)),
      (void**)out_device_infos);
}

static iree_status_t iree_hal_coralnpu_driver_dump_device_info(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_string_builder_t* builder) {
  // TODO: dump detailed device info (features, simulator configuration).
  return iree_ok_status();
}

static iree_status_t iree_hal_coralnpu_driver_create_device_by_id(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_coralnpu_driver_t* driver =
      iree_hal_coralnpu_driver_cast(base_driver);

  if (device_id != IREE_HAL_DEVICE_ID_DEFAULT &&
      device_id != IREE_HAL_CORALNPU_DEVICE_ID) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "device_id %" PRIuPTR " not found", device_id);
  }

  return iree_hal_coralnpu_device_create(
      driver->identifier, &driver->default_params, &driver->exec_backend,
      driver->device_allocator, host_allocator, out_device);
}

static iree_status_t iree_hal_coralnpu_driver_create_device_by_path(
    iree_hal_driver_t* base_driver, iree_string_view_t driver_name,
    iree_string_view_t device_path, iree_host_size_t param_count,
    const iree_string_pair_t* params, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  if (!iree_string_view_is_empty(device_path) &&
      !iree_string_view_equal(device_path, IREE_SV("default"))) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "device path '%.*s' not found under driver '%.*s'",
                            (int)device_path.size, device_path.data,
                            (int)driver_name.size, driver_name.data);
  }
  return iree_hal_coralnpu_driver_create_device_by_id(
      base_driver, IREE_HAL_DEVICE_ID_DEFAULT, param_count, params,
      host_allocator, out_device);
}

static const iree_hal_driver_vtable_t iree_hal_coralnpu_driver_vtable = {
    .destroy = iree_hal_coralnpu_driver_destroy,
    .query_available_devices = iree_hal_coralnpu_driver_query_available_devices,
    .dump_device_info = iree_hal_coralnpu_driver_dump_device_info,
    .create_device_by_id = iree_hal_coralnpu_driver_create_device_by_id,
    .create_device_by_path = iree_hal_coralnpu_driver_create_device_by_path,
};
