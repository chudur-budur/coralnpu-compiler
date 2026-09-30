// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Chained-descriptor DMA test.
//
// Because DMA descriptors have no stride field, gathering a 2D tile from a
// row-major DDR buffer requires one chained descriptor per row. This test
// copies ROWS rows (ROW_BYTES each, spaced SRC_PITCH bytes apart in DDR) into
// a contiguous DTCM buffer.
//
// On success, the test writes RESULT_TAG_CHAIN to `dma_test_result[0]`. On
// failure, the low 16 bits hold the failing step number, and
// `dma_test_result[1]` (`aux=`) holds the hardware error code if
// `coralnpu_dma_wait_done()` fails.

#include <stdint.h>

#include "crt/coralnpu_dma.h"
#include "tests/dma/dma_test_common.h"

#define ROWS 8u
#define ROW_BYTES 64u  // 16 u32 words per row (multiple of 16-byte beat)
#define ROW_WORDS (ROW_BYTES / 4u)
#define SRC_PITCH 256u  // byte stride between source rows in DDR
#define SRC_WORDS ((SRC_PITCH * ROWS) / 4u)
#define MAX_BEAT_BYTES (1u << CORALNPU_DMA_MAX_WIDTH_LOG2)
#define GUARD_WORDS (MAX_BEAT_BYTES / 4u)  // one max-width beat past the tile
#define GUARD_VALUE 0xDEADBEEFu

static uint32_t dtcm_tile[ROWS * ROW_WORDS + GUARD_WORDS]
    __attribute__((aligned(64)));
static coralnpu_dma_descriptor_t chain[ROWS];

_Static_assert(ROW_BYTES % MAX_BEAT_BYTES == 0u,
               "ROW_BYTES must be a multiple of the max beat width");
_Static_assert(SRC_PITCH % MAX_BEAT_BYTES == 0u,
               "SRC_PITCH must be a multiple of the max beat width");
_Static_assert(DDR_SRC % MAX_BEAT_BYTES == 0u,
               "DDR_SRC must be aligned to the max beat width");
_Static_assert(__alignof__(dtcm_tile) >= MAX_BEAT_BYTES,
               "dtcm_tile must be aligned to the max beat width");

static volatile uint32_t *const ddr_src = (volatile uint32_t *)DDR_SRC;

static int run(void) {
  for (uint32_t i = 0; i < SRC_WORDS; ++i) ddr_src[i] = i;
  for (uint32_t i = 0; i < ROWS * ROW_WORDS + GUARD_WORDS; ++i) {
    dtcm_tile[i] = GUARD_VALUE;
  }

  // One descriptor per row. start() does not validate per-descriptor beat
  // width (only copy_async() does); the static asserts above guarantee every
  // row is beat-aligned at both ends for the max width.
  for (uint32_t r = 0; r < ROWS; ++r) {
    chain[r].src_addr = DDR_SRC + r * SRC_PITCH;
    chain[r].dst_addr = (uint32_t)(uintptr_t)&dtcm_tile[r * ROW_WORDS];
    chain[r].len_flags =
        coralnpu_dma_make_len_flags(ROW_BYTES, CORALNPU_DMA_MAX_WIDTH_LOG2, 0u);
    chain[r].next_desc =
        (r + 1u < ROWS) ? (uint32_t)(uintptr_t)&chain[r + 1u] : 0u;
    chain[r].poll_addr = 0u;
    chain[r].poll_mask = 0u;
    chain[r].poll_value = 0u;
    chain[r].reserved = 0u;
  }

  if (coralnpu_dma_start(&chain[0]) != CORALNPU_DMA_OK) return 1;
  if (coralnpu_dma_wait_done() != CORALNPU_DMA_OK) {
    dma_test_result[1] = coralnpu_dma_error_code();
    return 2;
  }

  // Verify that all rows were gathered contiguously into dtcm_tile.
  for (uint32_t r = 0; r < ROWS; ++r) {
    for (uint32_t c = 0; c < ROW_WORDS; ++c) {
      if (dtcm_tile[r * ROW_WORDS + c] != (r * SRC_PITCH) / 4u + c) return 3;
    }
  }

  // The chain must not write past the last row.
  for (uint32_t i = 0; i < GUARD_WORDS; ++i) {
    if (dtcm_tile[ROWS * ROW_WORDS + i] != GUARD_VALUE) return 4;
  }

  return 0;
}

int main(void) {
  *dma_test_result = RESULT_TAG_CHAIN | (uint32_t)run();
  return 0;
}
