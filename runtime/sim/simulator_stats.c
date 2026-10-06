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

#include "runtime/sim/simulator_stats.h"

#include <string.h>

void iree_hal_coralnpu_stats_collector_initialize(
    coralnpu_simulator_t *sim, iree_hal_coralnpu_dispatch_stats_t *stats,
    iree_hal_coralnpu_stats_collector_t *out_collector) {
  memset(out_collector, 0, sizeof(*out_collector));
  out_collector->sim = sim;
  out_collector->stats = stats;
}

void iree_hal_coralnpu_stats_collector_begin_run(
    iree_hal_coralnpu_stats_collector_t *collector) {
  if (!collector->stats) return;
  collector->cycle_start = coralnpu_simulator_get_cycle_count(collector->sim);
}

void iree_hal_coralnpu_stats_collector_end_run(
    iree_hal_coralnpu_stats_collector_t *collector) {
  iree_hal_coralnpu_dispatch_stats_t *stats = collector->stats;
  if (!stats) return;
  const uint64_t cycle_end = coralnpu_simulator_get_cycle_count(collector->sim);
  // Simulators without a cycle counter always report 0.
  if (cycle_end > collector->cycle_start) {
    stats->cycles_total = cycle_end - collector->cycle_start;
    stats->valid_fields |= IREE_HAL_CORALNPU_DISPATCH_STATS_FIELD_CYCLES_TOTAL;
  }
}
