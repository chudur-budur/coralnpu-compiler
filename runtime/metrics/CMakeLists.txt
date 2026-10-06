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

# Per-dispatch execution statistics: the record the execution backends
# (runtime/sim) collect, the observer interface through which the HAL driver
# (runtime/driver) passes it on, and its consumers. Depends on neither the
# driver nor the simulators.

# The statistics of a single dispatch.
iree_cc_library(
  NAME
    dispatch_stats
  HDRS
    "dispatch_stats.h"
  INCLUDES
    "${CMAKE_SOURCE_DIR}"
  PUBLIC
)

# The interface between a HAL device and the consumer of its statistics.
iree_cc_library(
  NAME
    dispatch_observer
  HDRS
    "dispatch_observer.h"
  INCLUDES
    "${CMAKE_SOURCE_DIR}"
  DEPS
    ::dispatch_stats
    iree::base
  PUBLIC
)

# Writes the statistics as JSON Lines or as a summary table
# (--coralnpu_stats_format).
iree_cc_library(
  NAME
    stats_writer
  HDRS
    "stats_writer.h"
  SRCS
    "stats_writer.c"
  INCLUDES
    "${CMAKE_SOURCE_DIR}"
  DEPS
    ::dispatch_observer
    ::dispatch_stats
    iree::base
    iree::base::threading
  PUBLIC
)
