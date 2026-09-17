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

// CoralNPU DMA driver.
//
// Under CORALNPU_DMA_ISSUE_DEFERRED (the default, used in simulation) the
// asynchrony of the engine is modelled here rather than in the simulator: the
// MPACT emulation completes a whole chain synchronously inside the CTRL write,
// which would make a missing coralnpu_dma_wait_done() unobservable. So
// coralnpu_dma_start() only records the chain, and coralnpu_dma_wait_done()
// spins out a modelled transfer latency and *then* writes DESC_ADDR / CTRL.
//
// Between start and wait the destination holds entirely stale data, whereas on
// hardware it would hold partially transferred data. Deferred issue is the more
// pessimistic model for that hazard, and only that one: nothing reads the
// source until wait_done() writes CTRL, so overwriting the source buffer or
// mutating the descriptor between the two calls is silently accepted here while
// racing the engine on hardware.
//
// Under CORALNPU_DMA_ISSUE_IMMEDIATE (real hardware) start() writes the CSRs
// directly. The API and the generated call sequence are identical in both.

#include "crt/coralnpu_dma.h"

#define REG32(addr) (*(volatile uint32_t *)(uintptr_t)(addr))

// Upper bound on STATUS polls in coralnpu_dma_wait_done().
#ifndef CORALNPU_DMA_POLL_LIMIT
#define CORALNPU_DMA_POLL_LIMIT 1000000
#endif  // CORALNPU_DMA_POLL_LIMIT

// Modelled costs, derived from the engine's FSM: two 128-bit descriptor
// fetches per descriptor, then one read and one write per beat with a single
// outstanding transaction. Overridable at build time so later phases can
// sensitivity-test their numbers; these defaults are a starting point, not a
// calibrated model.
#ifndef CORALNPU_DMA_FETCH_CYCLES
#define CORALNPU_DMA_FETCH_CYCLES 4u
#endif  // CORALNPU_DMA_FETCH_CYCLES
#ifndef CORALNPU_DMA_BEAT_CYCLES
#define CORALNPU_DMA_BEAT_CYCLES 4u
#endif  // CORALNPU_DMA_BEAT_CYCLES

// Longest chain the driver will walk. A chain longer than this means either a
// cycle in the next_desc pointers or a chain too long to account for; either
// way it is rejected rather than truncated, because the driver hands the chain
// to the engine afterwards and the engine has no guard of its own.
// MPACT's RunDma() walks it with a bare `while (desc_addr != 0)`, so a cycle
// reaching the engine hangs the simulator process, not just the guest.
// TODO: Do extensive test with chains.
#define CORALNPU_DMA_MAX_CHAIN 4096u

static struct {
  const coralnpu_dma_descriptor_t *pending;  // deferred issue only
  uint32_t pending_cycles;                   // deferred issue only
  uint32_t seq;
  uint32_t bytes;
} g_dma;

// Appends one record to the trace region (see coralnpu_dma.h). The runner
// zeroes the count word before the run and reads the region back afterwards.
static void coralnpu_dma_event(uint32_t phase, uint32_t bytes) {
  #if CORALNPU_DMA_TRACE_CAPACITY > 0
    uint32_t n = REG32(CORALNPU_DMA_TRACE_ADDR);
    if (n >= CORALNPU_DMA_TRACE_CAPACITY) return;
    coralnpu_dma_trace_t *rec =
        (coralnpu_dma_trace_t *)(uintptr_t)(CORALNPU_DMA_TRACE_ADDR + 4u +
                                            n * sizeof(coralnpu_dma_trace_t));
    rec->seq = g_dma.seq;
    rec->phase = phase;
    rec->bytes = bytes;
    rec->stamp = coralnpu_dma_stamp();
    REG32(CORALNPU_DMA_TRACE_ADDR) = n + 1u;
  #else
    (void)phase;
    (void)bytes;
  #endif
}

// Walks the chain once, writing the total payload bytes to |out_bytes| and the
// modelled transfer cost to |out_cycles|. One walk rather than two so the trace
// and the latency model can never disagree about what the chain holds.
//
// Returns CORALNPU_DMA_OK or CORALNPU_DMA_EINVAL if the chain exceeds
// CORALNPU_DMA_MAX_CHAIN, in which case neither output is written. Rejecting
// rather than truncating: a truncated walk under-counts both the traced byte
// total and the modelled latency, and the chain is not supposed to reach the engine.
static int coralnpu_dma_walk_chain(const coralnpu_dma_descriptor_t *d,
                                    uint32_t *out_bytes, uint32_t *out_cycles) {
  if(d == 0) return CORALNPU_DMA_EINVAL;
  uint32_t bytes = 0;
  uint32_t cycles = 0;
  uint32_t guard = 0;
  while (d) {
    if (++guard > CORALNPU_DMA_MAX_CHAIN) return CORALNPU_DMA_EINVAL;
    uint32_t len = d->len_flags & CORALNPU_DMA_LEN_MASK;
    uint32_t beat = 1u << ((d->len_flags >> CORALNPU_DMA_WIDTH_SHIFT) &
                           CORALNPU_DMA_WIDTH_MASK);
    bytes += len;
    cycles += 2u * CORALNPU_DMA_FETCH_CYCLES +
              (len / beat) * CORALNPU_DMA_BEAT_CYCLES;
    d = (const coralnpu_dma_descriptor_t *)(uintptr_t)d->next_desc;
  }
  if (out_bytes != 0) *out_bytes = bytes;
  if (out_cycles != 0) *out_cycles = cycles;
  return CORALNPU_DMA_OK;
}

#if CORALNPU_DMA_ISSUE_MODE == CORALNPU_DMA_ISSUE_DEFERRED
// Deterministic delay that needs no CSR support.
static void coralnpu_dma_spin(uint32_t cycles) {
  for (volatile uint32_t i = 0; i < cycles; ++i) { }
}
#endif  // CORALNPU_DMA_ISSUE_MODE

static void coralnpu_dma_issue(const coralnpu_dma_descriptor_t *desc) {
  REG32(CORALNPU_DMA_DESC_ADDR) = (uint32_t)(uintptr_t)desc;
  REG32(CORALNPU_DMA_CTRL) = CORALNPU_DMA_CTRL_ENABLE | CORALNPU_DMA_CTRL_START;
}

int coralnpu_dma_busy(void) {
#if CORALNPU_DMA_ISSUE_MODE == CORALNPU_DMA_ISSUE_DEFERRED
  return g_dma.pending != 0;
#else
  return (REG32(CORALNPU_DMA_STATUS) & CORALNPU_DMA_STATUS_BUSY) != 0;
#endif  // CORALNPU_DMA_ISSUE_MODE
}

uint32_t coralnpu_dma_error_code(void) {
  uint32_t status = REG32(CORALNPU_DMA_STATUS);
  return (status >> CORALNPU_DMA_STATUS_ERROR_CODE_SHIFT) &
         CORALNPU_DMA_STATUS_ERROR_CODE_MASK;
}

int coralnpu_dma_start(const coralnpu_dma_descriptor_t *desc) {
  // The engine clears CTRL.start only from idle, so a start written while busy
  // stays latched and re-fires when the current chain completes; the caller's
  // wait would then observe the first transfer is done while a second one runs
  // unpaired. Refuse in both modes.
  if (coralnpu_dma_busy()) return CORALNPU_DMA_EBUSY;

  // Validate the chain before anything is published or recorded: a cyclic or
  // over-long chain must not reach the engine, and must not burn a sequence
  // number or emit a START record on its way to being refused.
  uint32_t bytes = 0;
  uint32_t modelled_cycles = 0;
  int status = coralnpu_dma_walk_chain(desc, &bytes, &modelled_cycles);
  if (status != CORALNPU_DMA_OK) return status;

  // Publish the descriptor stores before the engine can fetch them.
  coralnpu_dma_fence();

  ++g_dma.seq;
  g_dma.bytes = bytes;
  coralnpu_dma_event(CORALNPU_DMA_EVENT_START, g_dma.bytes);

#if CORALNPU_DMA_ISSUE_MODE == CORALNPU_DMA_ISSUE_DEFERRED
  // Record only. No CSR write, so no data moves until the wait -- this is what
  // makes a missing wait observable.
  g_dma.pending = desc;
  g_dma.pending_cycles = modelled_cycles;
#else
  (void)modelled_cycles;  // nothing to model when the engine starts for real
  coralnpu_dma_issue(desc);
#endif  // CORALNPU_DMA_ISSUE_MODE
  return CORALNPU_DMA_OK;
}

int coralnpu_dma_wait_done(void) {
#if CORALNPU_DMA_ISSUE_MODE == CORALNPU_DMA_ISSUE_DEFERRED
  if (g_dma.pending == 0) return CORALNPU_DMA_OK;  // nothing outstanding

  // Pay the modelled transfer latency, then issue for real.
  coralnpu_dma_spin(g_dma.pending_cycles);
  coralnpu_dma_fence();
  coralnpu_dma_issue(g_dma.pending);
  g_dma.pending = 0;
#endif  // CORALNPU_DMA_ISSUE_MODE

  for (int i = 0; i < CORALNPU_DMA_POLL_LIMIT; ++i) {
    uint32_t status = REG32(CORALNPU_DMA_STATUS);
    if (status & CORALNPU_DMA_STATUS_DONE) {
      // Make the engine's writes visible before the caller reads them.
      coralnpu_dma_fence();
      coralnpu_dma_event(CORALNPU_DMA_EVENT_DONE, g_dma.bytes);
      return (status & CORALNPU_DMA_STATUS_ERROR) ? CORALNPU_DMA_EHARDWARE
                                                  : CORALNPU_DMA_OK;
    }
  }
  return CORALNPU_DMA_ETIMEOUT;
}

void coralnpu_dma_abort(void) {
#if CORALNPU_DMA_ISSUE_MODE == CORALNPU_DMA_ISSUE_DEFERRED
  g_dma.pending = 0;  // never issued; nothing in the engine to abort
#endif  // CORALNPU_DMA_ISSUE_MODE
  REG32(CORALNPU_DMA_CTRL) = CORALNPU_DMA_CTRL_ABORT;
  coralnpu_dma_fence();
}

int coralnpu_dma_copy_async(uint32_t dst, uint32_t src, uint32_t size_bytes,
                            coralnpu_dma_descriptor_t *desc) {
  if (size_bytes == 0u || size_bytes > CORALNPU_DMA_LEN_MASK) {
    return CORALNPU_DMA_EINVAL;
  }
  // Checked here as well as in start(): |desc| may be the descriptor of the
  // chain still in flight, and a chain must not be modified while the engine
  // can fetch it.
  if (coralnpu_dma_busy()) return CORALNPU_DMA_EBUSY;

  // The beat size must divide the length exactly or the engine's 24-bit
  // remaining counter underflows. The emulation ignores xfer_width entirely,
  // so this check is the ONLY thing standing between a bad descriptor and a
  // runaway transfer on real hardware; it must never be compiled out.
  // choose_width guarantees divisibility and alignment for any width it
  // returns, and refuses rather than falling back to a 1-byte beat.
  uint32_t width_log2 = coralnpu_dma_choose_width(size_bytes, src, dst);
  if (width_log2 == CORALNPU_DMA_WIDTH_INVALID) {
    return CORALNPU_DMA_EINVAL;
  }

  desc->src_addr = src;
  desc->dst_addr = dst;
  desc->len_flags = coralnpu_dma_make_len_flags(size_bytes, width_log2, 0u);
  desc->next_desc = 0u;
  desc->poll_addr = 0u;
  desc->poll_mask = 0u;
  desc->poll_value = 0u;
  desc->reserved = 0u;

  return coralnpu_dma_start(desc);
}
