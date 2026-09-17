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

// Host-side runner for the standalone CoralNPU DMA tests.
//
// Loads a bare-metal riscv32 ELF into the MPACT simulator, runs it to
// completion, then reads back a result word and the DMA event trace that the
// guest wrote to DDR. The guest cannot return an exit status itself, so the
// verdict is decided here from the result word.
//
// The runtime's ELF loader (runtime/sim/simulator_elf_loader.c) is not used
// because it requires the `coralnpu_dispatch_request` mailbox symbol, which
// only dispatch firmware linked against libcoralnpu_iree.a defines. These
// tests link the CRT alone, so a minimal PT_LOAD loader lives here instead.
//
// Usage:
//   dma_sim_runner <elf> --expect <hex>
//   dma_sim_runner <elf> --expect-nonzero-low16

#include <elf.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "crt/coralnpu_dma.h"
#include "runtime/sim/simulator_api.h"
#include "tests/dma/dma_test_common.h"

extern "C" coralnpu_simulator_t* coralnpu_simulator_mpact_create(void);

namespace {

// Must match RESULT_ADDR in the guest tests. The word after it is an
// auxiliary value some tests report (printed as `aux=`).
constexpr uint32_t kResultAddr = RESULT_ADDR;

std::vector<uint8_t> ReadFile(const char* path) {
  FILE* f = std::fopen(path, "rb");
  if (f == nullptr) {
    std::perror(path);
    std::exit(2);
  }
  std::fseek(f, 0, SEEK_END);
  long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> buf(static_cast<size_t>(n));
  if (std::fread(buf.data(), 1, buf.size(), f) != buf.size()) {
    std::fprintf(stderr, "short read: %s\n", path);
    std::exit(2);
  }
  std::fclose(f);
  return buf;
}

// Copies every PT_LOAD segment into simulator memory and returns the entry
// point. Only the ELF32 little-endian RISC-V layout is accepted.
bool LoadElf(coralnpu_simulator_t* sim, const std::vector<uint8_t>& image,
             uint32_t* out_entry) {
  if (image.size() < sizeof(Elf32_Ehdr)) {
    std::fprintf(stderr, "ELF image too small\n");
    return false;
  }
  Elf32_Ehdr ehdr;
  std::memcpy(&ehdr, image.data(), sizeof(ehdr));
  if (std::memcmp(ehdr.e_ident, ELFMAG, SELFMAG) != 0 ||
      ehdr.e_ident[EI_CLASS] != ELFCLASS32 ||
      ehdr.e_ident[EI_DATA] != ELFDATA2LSB || ehdr.e_machine != EM_RISCV ||
      ehdr.e_phentsize != sizeof(Elf32_Phdr)) {
    std::fprintf(stderr, "not a riscv32 ELF\n");
    return false;
  }
  uint64_t phdr_end = static_cast<uint64_t>(ehdr.e_phoff) +
                      static_cast<uint64_t>(ehdr.e_phnum) * sizeof(Elf32_Phdr);
  if (phdr_end > image.size()) {
    std::fprintf(stderr, "ELF program headers out of bounds\n");
    return false;
  }

  bool entry_loaded = false;
  for (unsigned i = 0; i < ehdr.e_phnum; ++i) {
    Elf32_Phdr phdr;
    std::memcpy(&phdr, image.data() + ehdr.e_phoff + i * sizeof(Elf32_Phdr),
                sizeof(phdr));
    if (phdr.p_type != PT_LOAD || phdr.p_memsz == 0) continue;
    if (phdr.p_memsz < phdr.p_filesz ||
        static_cast<uint64_t>(phdr.p_offset) + phdr.p_filesz > image.size()) {
      std::fprintf(stderr, "ELF PT_LOAD %u is malformed\n", i);
      return false;
    }
    if (phdr.p_filesz != 0) {
      coralnpu_simulator_write_mem(sim, phdr.p_paddr,
                                   image.data() + phdr.p_offset, phdr.p_filesz);
    }
    if (phdr.p_memsz > phdr.p_filesz) {
      std::vector<uint8_t> zeros(phdr.p_memsz - phdr.p_filesz, 0);
      coralnpu_simulator_write_mem(sim, phdr.p_paddr + phdr.p_filesz,
                                   zeros.data(), zeros.size());
    }
    if ((phdr.p_flags & PF_X) != 0 && ehdr.e_entry >= phdr.p_paddr &&
        ehdr.e_entry < phdr.p_paddr + phdr.p_memsz) {
      entry_loaded = true;
    }
  }
  if (!entry_loaded) {
    std::fprintf(stderr, "ELF entry 0x%08x is not in an executable segment\n",
                 ehdr.e_entry);
    return false;
  }
  *out_entry = ehdr.e_entry;
  return true;
}

void DumpTrace(coralnpu_simulator_t* sim) {
  uint32_t count = 0;
  coralnpu_simulator_read_mem(sim, CORALNPU_DMA_TRACE_ADDR, &count,
                              sizeof(count));
  if (count > CORALNPU_DMA_TRACE_CAPACITY) count = CORALNPU_DMA_TRACE_CAPACITY;
  std::vector<coralnpu_dma_trace_t> records(count);
  if (count != 0) {
    coralnpu_simulator_read_mem(sim, CORALNPU_DMA_TRACE_ADDR + 4u,
                                records.data(),
                                count * sizeof(coralnpu_dma_trace_t));
  }
  std::fprintf(stderr, "dma trace: %u record(s)\n", count);
  for (const auto& rec : records) {
    const char* phase = rec.phase == CORALNPU_DMA_EVENT_START  ? "start"
                        : rec.phase == CORALNPU_DMA_EVENT_DONE ? "done"
                                                               : "?";
    std::fprintf(stderr, "  seq=%u %-5s bytes=%u stamp=%u\n", rec.seq, phase,
                 rec.bytes, rec.stamp);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: %s <elf> (--expect <hex> | --expect-nonzero-low16)\n",
                 argv[0]);
    return 2;
  }
  const std::string mode = argv[2];
  uint32_t expect = 0;
  if (mode == "--expect") {
    if (argc < 4) {
      std::fprintf(stderr, "--expect requires a value\n");
      return 2;
    }
    expect = static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 16));
  } else if (mode != "--expect-nonzero-low16") {
    std::fprintf(stderr, "unknown mode: %s\n", mode.c_str());
    return 2;
  }

  std::vector<uint8_t> image = ReadFile(argv[1]);

  coralnpu_simulator_t* sim = coralnpu_simulator_mpact_create();
  if (sim == nullptr) {
    std::fprintf(stderr, "failed to create MPACT simulator\n");
    return 2;
  }

  uint32_t entry = 0;
  if (!LoadElf(sim, image, &entry)) {
    coralnpu_simulator_destroy(sim);
    return 2;
  }

  // Clear the result words and the trace count so nothing stale is read back.
  const uint32_t zeros[2] = {0, 0};
  coralnpu_simulator_write_mem(sim, kResultAddr, zeros, sizeof(zeros));
  const uint32_t zero = 0;
  coralnpu_simulator_write_mem(sim, CORALNPU_DMA_TRACE_ADDR, &zero,
                               sizeof(zero));

  coralnpu_simulator_run(sim, entry);

  uint32_t result = 0;
  coralnpu_simulator_read_mem(sim, kResultAddr, &result, sizeof(result));
  uint32_t aux = 0;
  coralnpu_simulator_read_mem(sim, kResultAddr + 4u, &aux, sizeof(aux));
  const uint64_t cycles = coralnpu_simulator_get_cycle_count(sim);
  std::fprintf(stderr, "result=0x%08x aux=%u cycles=%llu\n", result, aux,
               static_cast<unsigned long long>(cycles));

  // uint32_t dbg[3] = {0};
  // coralnpu_simulator_read_mem(sim, kResultAddr + 4u, dbg, sizeof(dbg));
  // std::fprintf(stderr, "STATUS=0x%08x dtcm_buf[0]=0x%08x len_flags=0x%08x\n",
  //           dbg[0], dbg[1], dbg[2]);
  DumpTrace(sim);
  coralnpu_simulator_destroy(sim);

  if (mode == "--expect-nonzero-low16") {
    // The guest reports a count (or a flag) in the low half; zero is failure.
    // For dma_runtime_nowait_test a zero means the transfer completed without
    // a wait, i.e. deferred issue is not in effect.
    if ((result & 0xFFFFu) == 0u) {
      std::fprintf(stderr, "FAIL: result low half is zero\n");
      return 1;
    }
    return 0;
  }
  if (result != expect) {
    std::fprintf(stderr, "FAIL: expected 0x%08x\n", expect);
    return 1;
  }
  return 0;
}
