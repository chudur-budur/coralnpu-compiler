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

#include "pjrt_plugin/client.h"

#include <cinttypes>
#include <utility>

#include "iree/async/frontier_tracker.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/local_sync/registration/driver_module.h"
#include "runtime/driver/registration/driver_module.h"

namespace iree::pjrt::coralnpu {
namespace {

// Sub-driver names and exposed device infos, indexed by device id.
const iree_string_view_t kDriverNames[] = {
    IREE_SVL("local-sync"),
    IREE_SVL("coralnpu"),
};
const iree_hal_device_info_t kDeviceInfos[] = {
    {
        .device_id = 0,
        .path = IREE_SVL("cpu"),
        .name = IREE_SVL("cpu"),
    },
    {
        .device_id = 1,
        .path = IREE_SVL("coralnpu"),
        .name = IREE_SVL("coralnpu"),
    },
};
constexpr iree_host_size_t kDeviceCount = IREE_ARRAYSIZE(kDeviceInfos);
static_assert(IREE_ARRAYSIZE(kDriverNames) == kDeviceCount);

struct iree_hal_composite_driver_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_hal_driver_t* drivers[kDeviceCount];
  // Created with all devices on first use; lives as long as the driver.
  iree_hal_device_group_t* device_group;
};

extern const iree_hal_driver_vtable_t iree_hal_composite_driver_vtable;

iree_hal_composite_driver_t* iree_hal_composite_driver_cast(
    iree_hal_driver_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_composite_driver_vtable);
  return (iree_hal_composite_driver_t*)base_value;
}

void iree_hal_composite_driver_destroy(iree_hal_driver_t* base_driver) {
  iree_hal_composite_driver_t* driver =
      iree_hal_composite_driver_cast(base_driver);
  iree_allocator_t host_allocator = driver->host_allocator;

  iree_hal_device_group_release(driver->device_group);
  for (iree_hal_driver_t* sub_driver : driver->drivers) {
    iree_hal_driver_release(sub_driver);
  }

  iree_allocator_free(host_allocator, driver);
}

iree_status_t iree_hal_composite_driver_query_available_devices(
    iree_hal_driver_t* base_driver, iree_allocator_t host_allocator,
    iree_host_size_t* out_device_info_count,
    iree_hal_device_info_t** out_device_infos) {
  *out_device_info_count = kDeviceCount;
  return iree_allocator_clone(
      host_allocator,
      iree_make_const_byte_span(kDeviceInfos, sizeof(kDeviceInfos)),
      (void**)out_device_infos);
}

iree_status_t iree_hal_composite_driver_dump_device_info(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_string_builder_t* builder) {
  iree_hal_composite_driver_t* driver =
      iree_hal_composite_driver_cast(base_driver);
  if (device_id >= kDeviceCount) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "device_id %" PRIuPTR " not found", device_id);
  }
  return iree_hal_driver_dump_device_info(driver->drivers[device_id],
                                          IREE_HAL_DEVICE_ID_DEFAULT, builder);
}

iree_status_t iree_hal_composite_driver_create_device_by_id(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_composite_driver_t* driver =
      iree_hal_composite_driver_cast(base_driver);
  if (device_id >= kDeviceCount) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "device_id %" PRIuPTR " not found", device_id);
  }
  // First call creates and groups all devices with these params.
  if (!driver->device_group) {
    iree_async_frontier_tracker_t* frontier_tracker = nullptr;
    IREE_RETURN_IF_ERROR(iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), host_allocator,
        &frontier_tracker));
    iree_hal_device_group_builder_t builder;
    iree_hal_device_group_builder_initialize(&builder, frontier_tracker);
    iree_async_frontier_tracker_release(frontier_tracker);

    iree_hal_device_t* devices[kDeviceCount] = {};
    iree_status_t status = iree_ok_status();
    for (iree_host_size_t i = 0; i < kDeviceCount; ++i) {
      status = iree_hal_driver_create_device_by_id(
          driver->drivers[i], IREE_HAL_DEVICE_ID_DEFAULT, param_count, params,
          create_params, host_allocator, &devices[i]);
      if (iree_status_is_ok(status)) {
        status = iree_hal_device_group_builder_add_device(&builder, devices[i]);
      }
      if (!iree_status_is_ok(status)) break;
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_group_builder_finalize(&builder, host_allocator,
                                                      &driver->device_group);
    }
    iree_hal_device_group_builder_deinitialize(&builder);
    for (iree_hal_device_t* device : devices) {
      iree_hal_device_release(device);
    }
    IREE_RETURN_IF_ERROR(status);
  }

  iree_hal_device_t* device =
      iree_hal_device_group_device_at(driver->device_group, device_id);
  iree_hal_device_retain(device);
  *out_device = device;
  return iree_ok_status();
}

iree_status_t iree_hal_composite_driver_create_device_by_path(
    iree_hal_driver_t* base_driver, iree_string_view_t driver_name,
    iree_string_view_t device_path, iree_host_size_t param_count,
    const iree_string_pair_t* params,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  // Accept the device path or its numeric id.
  uint32_t device_id = 0;
  if (!iree_string_view_atoi_uint32(device_path, &device_id)) {
    device_id = kDeviceCount;
    for (iree_host_size_t i = 0; i < kDeviceCount; ++i) {
      if (iree_string_view_equal(device_path, kDeviceInfos[i].path)) {
        device_id = i;
        break;
      }
    }
  }
  if (device_id >= kDeviceCount) {
    return iree_make_status(IREE_STATUS_NOT_FOUND, "device_path %.*s not found",
                            (int)device_path.size, device_path.data);
  }
  return iree_hal_composite_driver_create_device_by_id(
      base_driver, device_id, param_count, params, create_params,
      host_allocator, out_device);
}

const iree_hal_driver_vtable_t iree_hal_composite_driver_vtable = {
    .destroy = iree_hal_composite_driver_destroy,
    .query_available_devices =
        iree_hal_composite_driver_query_available_devices,
    .dump_device_info = iree_hal_composite_driver_dump_device_info,
    .create_device_by_id = iree_hal_composite_driver_create_device_by_id,
    .create_device_by_path = iree_hal_composite_driver_create_device_by_path,
};

}  // namespace

CoralNPUClientInstance::CoralNPUClientInstance(
    std::unique_ptr<Platform> platform)
    : ClientInstance(std::move(platform)) {
  cached_platform_name_ = "iree_coralnpu";
}

iree_status_t CoralNPUClientInstance::CreateDriver(
    iree_hal_driver_t** out_driver) {
  IREE_RETURN_IF_ERROR(
      iree_hal_local_sync_driver_module_register(driver_registry_));
  IREE_RETURN_IF_ERROR(
      iree_hal_coralnpu_driver_module_register(driver_registry_));

  iree_hal_composite_driver_t* driver = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator_, sizeof(*driver), (void**)&driver));
  iree_hal_resource_initialize(&iree_hal_composite_driver_vtable,
                               &driver->resource);
  driver->host_allocator = host_allocator_;

  for (iree_host_size_t i = 0; i < kDeviceCount; ++i) {
    iree_status_t status = iree_hal_driver_registry_try_create(
        driver_registry_, kDriverNames[i], host_allocator_,
        &driver->drivers[i]);
    if (!iree_status_is_ok(status)) {
      iree_hal_driver_release((iree_hal_driver_t*)driver);
      return status;
    }
  }

  composite_driver_ = (iree_hal_driver_t*)driver;
  *out_driver = composite_driver_;
  return iree_ok_status();
}

iree_status_t CoralNPUClientInstance::PopulateVMModules(
    std::vector<iree::vm::ref<iree_vm_module_t>>& modules,
    iree_hal_device_t* hal_device,
    iree::vm::ref<iree_vm_module_t>& main_module) {
  // |hal_device| came from create_device_by_id, so the group exists.
  modules.push_back({});
  IREE_RETURN_IF_ERROR(iree_hal_module_create(
      vm_instance(), iree_hal_module_device_policy_default(),
      iree_hal_composite_driver_cast(composite_driver_)->device_group,
      IREE_HAL_MODULE_FLAG_NONE, iree_hal_module_debug_sink_stdio(stderr),
      host_allocator_, &modules.back()));
  modules.push_back(main_module);
  return iree_ok_status();
}

bool CoralNPUClientInstance::SetCompilerFlags(
    CompilerJob* compiler_job, const xla::CompileOptionsProto& options) {
  int target_dev = -1;
  const auto& build_options = options.executable_build_options();
  if (build_options.has_device_assignment() &&
      build_options.device_assignment().computation_devices_size() > 0 &&
      build_options.device_assignment()
              .computation_devices(0)
              .replica_device_ids_size() > 0) {
    target_dev = build_options.device_assignment()
                     .computation_devices(0)
                     .replica_device_ids(0);
  }

  bool is_uncommitted =
      build_options.allow_spmd_sharding_propagation_to_parameters_size() > 0 &&
      build_options.allow_spmd_sharding_propagation_to_parameters(0);

  auto set_cpu_flags = [&]() {
    return compiler_job->SetFlag("--iree-hal-target-device=local") &&
           compiler_job->SetFlag(
               "--iree-hal-local-target-device-backends=llvm-cpu") &&
           compiler_job->SetFlag("--iree-llvmcpu-target-cpu=host");
  };

  auto set_coralnpu_flags = [&]() {
    return compiler_job->SetFlag("--iree-hal-target-device=coralnpu");
  };

  // A committed single device compiles for that device only.
  if (!is_uncommitted && target_dev == 0) return set_cpu_flags();
  if (!is_uncommitted && target_dev == 1) return set_coralnpu_flags();

  // Multi-device: target both so CoralNPUAffinityAnnotation partitions ops
  // between CPU and CoralNPU.
  return set_cpu_flags() && set_coralnpu_flags();
}

bool CoralNPUClientInstance::SetDefaultCompilerFlags(
    CompilerJob* compiler_job) {
  return SetCompilerFlags(compiler_job, xla::CompileOptionsProto());
}

}  // namespace iree::pjrt::coralnpu
