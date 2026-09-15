# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

iree_add_all_subdirs()

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  iree_register_external_hal_driver(
    NAME
      coralnpu
    DRIVER_TARGET
      coralnpu_runtime::driver::registration
    REGISTER_FN
      iree_hal_coralnpu_driver_module_register
  )
endif()

iree_cc_library(
  NAME
    coralnpu
  HDRS
    "coralnpu_command_buffer.h"
    "coralnpu_device.h"
    "coralnpu_driver.h"
    "coralnpu_event.h"
    "coralnpu_exec_backend.h"
    "coralnpu_executable.h"
    "coralnpu_executable_cache.h"
  SRCS
    "coralnpu_command_buffer.c"
    "coralnpu_device.c"
    "coralnpu_driver.c"
    "coralnpu_event.c"
    "coralnpu_executable.c"
    "coralnpu_executable_cache.c"
  DEPS
    iree::base
    iree::base::internal::arena
    iree::hal
    iree::hal::drivers::local_sync::sync_driver
    iree::hal::local::executable_library
    iree::hal::utils::deferred_command_buffer
    iree::hal::utils::file_transfer
    iree::hal::utils::files
    iree::hal::utils::queue_emulation
  INCLUDES
    "${PROJECT_SOURCE_DIR}"
  PUBLIC
)
