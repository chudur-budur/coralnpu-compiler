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

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  # Registers the driver with IREE and selects the execution backend it runs
  # on. The simulator dependency is confined to this target so that the HAL
  # driver in coralnpu_runtime::driver::coralnpu stays simulator-unaware.
  iree_cc_library(
    NAME
      registration
    HDRS
      "driver_module.h"
    SRCS
      "driver_module.c"
    DEPS
      coralnpu_runtime::driver::coralnpu
      coralnpu_runtime::sim::simulation
      coralnpu_runtime::sim::simulator_mpact
      iree::base
      iree::base::internal::dynamic_library
      iree::base::tooling::flags
      iree::hal
    DEFINES
      "IREE_HAVE_HAL_CORALNPU_DRIVER_MODULE=1"
    PUBLIC
  )
endif()

# BAZEL_TO_CMAKE_PRESERVES_ALL_CONTENT_BELOW_THIS_LINE ###
