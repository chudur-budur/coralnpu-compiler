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

// Positive DMA test: a transfer in each direction, the busy guard, and the
// shape checks. The runner expects RESULT_TAG_RUNTIME with a zero low half;
// a nonzero low half is the step number that failed.

#include <stdint.h>

#include "crt/coralnpu_dma.h"
#include "tests/dma/dma_test_common.h"

#define TEST_SIZE 64  // 256 bytes: a multiple of the 16-byte beat

static uint32_t dtcm_buf[TEST_SIZE] __attribute__((aligned(64)));
static coralnpu_dma_descriptor_t desc;

static volatile uint32_t *const ddr_src = (volatile uint32_t *)DDR_SRC;
static volatile uint32_t *const ddr_dst = (volatile uint32_t *)DDR_DST;

// #define DEBUG_LOG_ADDR 0x80002000u
// volatile uint32_t *debug_log = (volatile uint32_t *)DEBUG_LOG_ADDR;

static int run(void) {
  for (int i = 0; i < TEST_SIZE; ++i) {
    ddr_src[i] = 0xA5A50000u + (uint32_t)i;
    dtcm_buf[i] = 0;
    ddr_dst[i] = 0;
  }

  // DDR -> DTCM.
  if (coralnpu_dma_copy_async((uint32_t)(uintptr_t)dtcm_buf, DDR_SRC,
                              sizeof(dtcm_buf), &desc) != CORALNPU_DMA_OK) {
    return 1;
  }
  if (coralnpu_dma_wait_done() != CORALNPU_DMA_OK) return 2;
  for (int i = 0; i < TEST_SIZE; ++i) {
    if (dtcm_buf[i] != 0xA5A50000u + (uint32_t)i) return 3;
    dtcm_buf[i] += 1;
  }

  // DTCM -> DDR.
  if (coralnpu_dma_copy_async(DDR_DST, (uint32_t)(uintptr_t)dtcm_buf,
                              sizeof(dtcm_buf), &desc) != CORALNPU_DMA_OK) {
    return 4;
  }
  if (coralnpu_dma_wait_done() != CORALNPU_DMA_OK) return 5;
  for (int i = 0; i < TEST_SIZE; ++i) {
    if (ddr_dst[i] != 0xA5A50001u + (uint32_t)i) return 6;
  }

  // A start issued while a chain is outstanding must be refused.
  if (coralnpu_dma_copy_async((uint32_t)(uintptr_t)dtcm_buf, DDR_SRC,
                              sizeof(dtcm_buf), &desc) != CORALNPU_DMA_OK) {
    return 7;
  }
  // dma_test_result[1] = *(volatile uint32_t *)CORALNPU_DMA_STATUS;
  // dma_test_result[2] = dtcm_buf[0];
  // dma_test_result[3] = desc.len_flags;
  if (coralnpu_dma_start(&desc) != CORALNPU_DMA_EBUSY) return 8;
  if (coralnpu_dma_wait_done() != CORALNPU_DMA_OK) return 9;

  // A zero length must be refused.
  if (coralnpu_dma_copy_async(DDR_DST, DDR_SRC, 0, &desc) !=
      CORALNPU_DMA_EINVAL) {
    return 10;
  }

  // A length no accepted beat divides must be refused rather than degraded to
  // a 1-byte beat: 6 is a multiple of 2 but not of 4, so it fails the
  // CORALNPU_DMA_MIN_WIDTH_LOG2 floor. This is the shape that would underflow
  // the engine's remaining counter if the width were hardcoded.
  if (coralnpu_dma_copy_async(DDR_DST, DDR_SRC, 6, &desc) !=
      CORALNPU_DMA_EINVAL) {
    return 11;
  }

  // Likewise a misaligned address, even with a well-shaped length.
  if (coralnpu_dma_copy_async(DDR_DST + 2u, DDR_SRC, 64, &desc) !=
      CORALNPU_DMA_EINVAL) {
    return 12;
  }

  // A wait with nothing outstanding is a no-op, not an error.
  if (coralnpu_dma_wait_done() != CORALNPU_DMA_OK) return 13;

  return 0;
}

int main(void) {
  *dma_test_result = RESULT_TAG_RUNTIME | (uint32_t)run();
  return 0;
}
