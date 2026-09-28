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

#include "hw_sim/coralnpu_simulator.h"
#include "sw/coralnpu_sim/spike_simulator.h"

namespace {

class CoralNPUSpikeSimulator final : public CoralNPUSimulator {
 public:
  // TCM (2 MB, fits the highmem layout), EXTMEM (4 MB), and DDR (1 GB).
  CoralNPUSpikeSimulator()
      : sim_({.memory_regions = {
                  {0x00000000, 0x00200000},
                  {0x20000000, 0x00400000},
                  {0x80000000, 0x40000000},
              }}) {}

  void ReadMem(uint32_t addr, size_t size, char* data) override {
    sim_.ReadMemory(addr, data, size);
  }

  const CoralNPUMailbox& ReadMailbox(void) override { return mailbox_; }

  void WriteMem(uint32_t addr, size_t size, const char* data) override {
    sim_.WriteMemory(addr, data, size);
  }

  void WriteMailbox(const CoralNPUMailbox& mailbox) override {
    mailbox_ = mailbox;
  }

  void Run(uint32_t start_addr) override {
    accumulated_cycles_ += sim_.GetCycleCount();
    sim_.Reset();
    sim_.WriteRegister("pc", start_addr);
  }

  // Runs to halt, ignoring the budget, like the MPACT backend.
  bool WaitForTermination(int /*timeout*/) override {
    while (sim_.Step(100000) > 0) {
    }
    return sim_.IsHalted();
  }

  uint64_t GetCycleCount() const override {
    return accumulated_cycles_ + sim_.GetCycleCount();
  }

 private:
  coralnpu::sim::SpikeSimulator sim_;
  CoralNPUMailbox mailbox_{};
  uint64_t accumulated_cycles_ = 0;
};

}  // namespace

extern "C" __attribute__((visibility("default"))) CoralNPUSimulator*
coralnpu_simulator_spike_create(void) {
  return new CoralNPUSpikeSimulator();
}
