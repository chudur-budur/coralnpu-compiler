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

// Collects the statistics of a dispatch run by
// iree_hal_simulator_issue_dispatch_inline (simulator_inline.h). The run calls
// the hooks below at fixed points, which keeps the measurements out of it; the
// hooks do nothing when the caller collects no statistics.

#ifndef RUNTIME_SIM_SIMULATOR_STATS_H_
#define RUNTIME_SIM_SIMULATOR_STATS_H_

#include <stdint.h>

#include "runtime/metrics/dispatch_stats.h"
#include "runtime/sim/simulator_api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_coralnpu_stats_collector_t {
  coralnpu_simulator_t *sim;
  // Where the statistics go; NULL if the caller collects none.
  iree_hal_coralnpu_dispatch_stats_t *stats;
  // The simulator's cycle count when the run started.
  uint64_t cycle_start;
} iree_hal_coralnpu_stats_collector_t;

// Initializes |out_collector| for a dispatch on |sim| that fills |stats|,
// which may be NULL.
void iree_hal_coralnpu_stats_collector_initialize(
    coralnpu_simulator_t *sim, iree_hal_coralnpu_dispatch_stats_t *stats,
    iree_hal_coralnpu_stats_collector_t *out_collector);

// Called right before the program runs.
void iree_hal_coralnpu_stats_collector_begin_run(
    iree_hal_coralnpu_stats_collector_t *collector);

// Called right after the program ran, also when it failed.
void iree_hal_coralnpu_stats_collector_end_run(
    iree_hal_coralnpu_stats_collector_t *collector);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // RUNTIME_SIM_SIMULATOR_STATS_H_
