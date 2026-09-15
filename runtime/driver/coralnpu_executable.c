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

#include "runtime/driver/coralnpu_executable.h"

#include <inttypes.h>
#include <stdint.h>
#include <string.h>

#include "iree/hal/local/elf/elf_types.h"
#include "iree/hal/local/executable_library.h"

typedef struct iree_hal_coralnpu_elf32_library_header_t {
  uint32_t version;
  uint32_t name;
  uint32_t features;
  uint32_t sanitizer;
} iree_hal_coralnpu_elf32_library_header_t;

typedef struct iree_hal_coralnpu_elf32_import_table_v0_t {
  uint32_t count;
  uint32_t symbols;
} iree_hal_coralnpu_elf32_import_table_v0_t;

typedef struct iree_hal_coralnpu_elf32_export_table_v0_t {
  uint32_t count;
  uint32_t ptrs;
  uint32_t attrs;
  uint32_t params;
  uint32_t occupancy;
  uint32_t names;
  uint32_t tags;
  uint32_t parameter_names;
  uint32_t source_locations;
  uint32_t stage_locations;
} iree_hal_coralnpu_elf32_export_table_v0_t;

typedef struct iree_hal_coralnpu_elf32_constant_table_v0_t {
  uint32_t count;
} iree_hal_coralnpu_elf32_constant_table_v0_t;

typedef struct iree_hal_coralnpu_elf32_source_file_table_v0_t {
  uint32_t count;
  uint32_t files;
} iree_hal_coralnpu_elf32_source_file_table_v0_t;

typedef struct iree_hal_coralnpu_elf32_library_v0_t {
  uint32_t header;
  iree_hal_coralnpu_elf32_import_table_v0_t imports;
  iree_hal_coralnpu_elf32_export_table_v0_t exports;
  iree_hal_coralnpu_elf32_constant_table_v0_t constants;
  iree_hal_coralnpu_elf32_source_file_table_v0_t sources;
} iree_hal_coralnpu_elf32_library_v0_t;

typedef struct iree_hal_coralnpu_executable_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_const_byte_span_t dispatch_image;
  iree_host_size_t function_count;
  const iree_hal_executable_dispatch_attrs_v0_t *dispatch_attrs;
  iree_string_view_t *function_names;
} iree_hal_coralnpu_executable_t;

static const iree_hal_executable_vtable_t iree_hal_coralnpu_executable_vtable;

static iree_hal_coralnpu_executable_t *iree_hal_coralnpu_executable_cast(
    iree_hal_executable_t *base_executable) {
  IREE_HAL_ASSERT_TYPE(base_executable, &iree_hal_coralnpu_executable_vtable);
  return (iree_hal_coralnpu_executable_t *)base_executable;
}

static iree_status_t iree_hal_coralnpu_executable_validate_elf32(
    iree_const_byte_span_t elf_image) {
  if (elf_image.data_length < sizeof(iree_elf32_ehdr_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dispatch image is smaller than ELF32 header");
  }

  const uint8_t *ident = elf_image.data;
  if (memcmp(ident, "\177ELF", 4) != 0 ||
      ident[IREE_ELF_EI_CLASS] != IREE_ELF_ELFCLASS32 ||
      ident[IREE_ELF_EI_DATA] != IREE_ELF_ELFDATA2LSB) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dispatch image is not little-endian ELF32");
  }

  const iree_elf32_ehdr_t *ehdr = (const iree_elf32_ehdr_t *)elf_image.data;
  if (ehdr->e_machine != 243) {  // EM_RISCV
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "ELF is not RISC-V");
  }
  if (ehdr->e_phentsize != sizeof(iree_elf32_phdr_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unexpected ELF program header size");
  }
  uint64_t program_table_end =
      (uint64_t)ehdr->e_phoff +
      (uint64_t)ehdr->e_phnum * sizeof(iree_elf32_phdr_t);
  if (program_table_end > elf_image.data_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ELF program header table is out of bounds");
  }
  if (ehdr->e_shentsize != sizeof(iree_elf32_shdr_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unexpected ELF section header size");
  }
  uint64_t section_table_end =
      (uint64_t)ehdr->e_shoff +
      (uint64_t)ehdr->e_shnum * sizeof(iree_elf32_shdr_t);
  if (section_table_end > elf_image.data_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ELF section header table is out of bounds");
  }

  for (uint16_t i = 0; i < ehdr->e_phnum; ++i) {
    const iree_elf32_phdr_t *phdr =
        (const iree_elf32_phdr_t *)(elf_image.data + ehdr->e_phoff +
                                    i * sizeof(iree_elf32_phdr_t));
    if (phdr->p_type != IREE_ELF_PT_LOAD) {
      continue;
    }
    uint64_t segment_file_end = (uint64_t)phdr->p_offset + phdr->p_filesz;
    if (segment_file_end > elf_image.data_length) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "ELF PT_LOAD data is out of bounds");
    }
    if (phdr->p_memsz < phdr->p_filesz) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "ELF PT_LOAD memsz is smaller than filesz");
    }
  }

  return iree_ok_status();
}

static bool iree_hal_coralnpu_executable_string_equals(const char *string_table,
                                                       size_t string_table_size,
                                                       uint32_t string_offset,
                                                       const char *expected) {
  if (string_offset >= string_table_size) {
    return false;
  }
  const char *string = string_table + string_offset;
  size_t remaining = string_table_size - string_offset;
  return memchr(string, '\0', remaining) != NULL &&
         strcmp(string, expected) == 0;
}

iree_status_t iree_hal_coralnpu_executable_find_symbol(
    iree_const_byte_span_t elf_image, const char *symbol_name,
    uint32_t *out_address, uint32_t *out_size) {
  const uint8_t *data = elf_image.data;
  const iree_elf32_ehdr_t *ehdr = (const iree_elf32_ehdr_t *)data;
  const iree_elf32_shdr_t *section_headers =
      (const iree_elf32_shdr_t *)(data + ehdr->e_shoff);

  for (uint16_t i = 0; i < ehdr->e_shnum; ++i) {
    const iree_elf32_shdr_t *symbol_section = &section_headers[i];
    if (symbol_section->sh_type != IREE_ELF_SHT_SYMTAB) {
      continue;
    }
    if (symbol_section->sh_entsize != sizeof(iree_elf32_sym_t) ||
        symbol_section->sh_link >= ehdr->e_shnum) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid ELF symbol table");
    }

    const iree_elf32_shdr_t *string_section =
        &section_headers[symbol_section->sh_link];
    uint64_t symbol_table_end =
        (uint64_t)symbol_section->sh_offset + symbol_section->sh_size;
    uint64_t string_table_end =
        (uint64_t)string_section->sh_offset + string_section->sh_size;
    if (symbol_table_end > elf_image.data_length ||
        string_table_end > elf_image.data_length) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "ELF symbol table is out of bounds");
    }

    const iree_elf32_sym_t *symbols =
        (const iree_elf32_sym_t *)(data + symbol_section->sh_offset);
    size_t symbol_count = symbol_section->sh_size / sizeof(iree_elf32_sym_t);
    const char *string_table = (const char *)(data + string_section->sh_offset);

    for (size_t j = 0; j < symbol_count; ++j) {
      const iree_elf32_sym_t *symbol = &symbols[j];
      if (symbol->st_shndx == IREE_ELF_SHN_UNDEF) {
        continue;
      }
      if (!iree_hal_coralnpu_executable_string_equals(
              string_table, string_section->sh_size, symbol->st_name,
              symbol_name)) {
        continue;
      }
      *out_address = symbol->st_value;
      if (out_size) {
        *out_size = symbol->st_size;
      }
      return iree_ok_status();
    }
  }

  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "ELF symbol `%s` was not found", symbol_name);
}

// Returns the PT_LOAD segment whose file data contains |address|, or NULL.
static const iree_elf32_phdr_t *iree_hal_coralnpu_executable_find_segment(
    iree_const_byte_span_t elf_image, uint32_t address) {
  const uint8_t *data = elf_image.data;
  const iree_elf32_ehdr_t *ehdr = (const iree_elf32_ehdr_t *)data;
  for (uint16_t i = 0; i < ehdr->e_phnum; ++i) {
    const iree_elf32_phdr_t *phdr =
        (const iree_elf32_phdr_t *)(data + ehdr->e_phoff +
                                    i * sizeof(iree_elf32_phdr_t));
    if (phdr->p_type == IREE_ELF_PT_LOAD && address >= phdr->p_vaddr &&
        (uint64_t)address < (uint64_t)phdr->p_vaddr + phdr->p_filesz) {
      return phdr;
    }
  }
  return NULL;
}

static iree_status_t iree_hal_coralnpu_executable_resolve_ptr(
    iree_const_byte_span_t elf_image, uint32_t address, size_t size,
    const void **out_ptr) {
  const iree_elf32_phdr_t *phdr =
      iree_hal_coralnpu_executable_find_segment(elf_image, address);
  if (!phdr || size > phdr->p_filesz - (address - phdr->p_vaddr)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ELF address 0x%08" PRIx32
                            " (size %zu) is outside PT_LOAD file data",
                            address, size);
  }
  *out_ptr = elf_image.data + phdr->p_offset + (address - phdr->p_vaddr);
  return iree_ok_status();
}

static iree_status_t iree_hal_coralnpu_executable_resolve_string(
    iree_const_byte_span_t elf_image, uint32_t address,
    iree_string_view_t *out_string) {
  const iree_elf32_phdr_t *phdr =
      iree_hal_coralnpu_executable_find_segment(elf_image, address);
  if (!phdr) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ELF string address 0x%08" PRIx32
                            " is outside PT_LOAD file data",
                            address);
  }
  size_t offset = address - phdr->p_vaddr;
  const char *str = (const char *)elf_image.data + phdr->p_offset + offset;
  const char *nul = (const char *)memchr(str, '\0', phdr->p_filesz - offset);
  if (!nul) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unterminated string at ELF address 0x%08" PRIx32,
                            address);
  }
  *out_string = iree_make_string_view(str, (iree_host_size_t)(nul - str));
  return iree_ok_status();
}

bool iree_hal_coralnpu_executable_isa(iree_hal_executable_t *base_executable) {
  return iree_hal_resource_is(base_executable,
                              &iree_hal_coralnpu_executable_vtable);
}

iree_const_byte_span_t iree_hal_coralnpu_executable_dispatch_image(
    iree_hal_executable_t *base_executable) {
  iree_hal_coralnpu_executable_t *executable =
      iree_hal_coralnpu_executable_cast(base_executable);
  return executable->dispatch_image;
}

const iree_hal_executable_dispatch_attrs_v0_t *
iree_hal_coralnpu_executable_dispatch_attrs(
    iree_hal_executable_t *base_executable, uint32_t ordinal) {
  iree_hal_coralnpu_executable_t *executable =
      iree_hal_coralnpu_executable_cast(base_executable);
  return &executable->dispatch_attrs[ordinal];
}

static void iree_hal_coralnpu_executable_destroy(
    iree_hal_executable_t *base_executable) {
  iree_hal_coralnpu_executable_t *executable =
      iree_hal_coralnpu_executable_cast(base_executable);
  iree_allocator_t host_allocator = executable->host_allocator;
  iree_allocator_free(host_allocator, executable);
}

static iree_host_size_t iree_hal_coralnpu_executable_function_count(
    iree_hal_executable_t *base_executable) {
  iree_hal_coralnpu_executable_t *executable =
      iree_hal_coralnpu_executable_cast(base_executable);
  return executable->function_count;
}

static iree_status_t iree_hal_coralnpu_executable_function_info(
    iree_hal_executable_t *base_executable,
    iree_hal_executable_function_t function,
    iree_hal_executable_function_info_t *out_info) {
  iree_hal_coralnpu_executable_t *executable =
      iree_hal_coralnpu_executable_cast(base_executable);
  memset(out_info, 0, sizeof(*out_info));
  if (!iree_hal_executable_function_is_index_in_range(
          function, executable->function_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "function ordinal out of range");
  }

  const uint32_t ordinal = iree_hal_executable_function_index(function);
  out_info->name = executable->function_names[ordinal];
  const iree_hal_executable_dispatch_attrs_v0_t *attrs =
      &executable->dispatch_attrs[ordinal];
  if (iree_any_bit_set(attrs->flags,
                       IREE_HAL_EXECUTABLE_DISPATCH_FLAG_V0_SEQUENTIAL)) {
    out_info->flags |= IREE_HAL_EXECUTABLE_FUNCTION_FLAG_SEQUENTIAL;
  }
  if (iree_any_bit_set(
          attrs->flags,
          IREE_HAL_EXECUTABLE_DISPATCH_FLAG_V0_WORKGROUP_SIZE_DYNAMIC)) {
    out_info->flags |= IREE_HAL_EXECUTABLE_FUNCTION_FLAG_WORKGROUP_SIZE_DYNAMIC;
  }
  out_info->constant_count = attrs->constant_count;
  out_info->binding_count = attrs->binding_count;
  out_info->parameter_count = attrs->parameter_count;
  out_info->workgroup_size[0] = attrs->workgroup_size_x;
  out_info->workgroup_size[1] = attrs->workgroup_size_y;
  out_info->workgroup_size[2] = attrs->workgroup_size_z;

  return iree_ok_status();
}

static iree_status_t iree_hal_coralnpu_executable_function_parameters(
    iree_hal_executable_t *base_executable,
    iree_hal_executable_function_t function, iree_host_size_t capacity,
    iree_hal_executable_function_parameter_t *out_parameters) {
  iree_hal_coralnpu_executable_t *executable =
      iree_hal_coralnpu_executable_cast(base_executable);
  if (!iree_hal_executable_function_is_index_in_range(
          function, executable->function_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "function ordinal out of range");
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_coralnpu_executable_lookup_function_by_name(
    iree_hal_executable_t *base_executable, iree_string_view_t name,
    iree_hal_executable_function_t *out_function) {
  iree_hal_coralnpu_executable_t *executable =
      iree_hal_coralnpu_executable_cast(base_executable);
  IREE_ASSERT_ARGUMENT(out_function);
  *out_function = iree_hal_executable_function_invalid();

  for (iree_host_size_t i = 0; i < executable->function_count; ++i) {
    if (iree_string_view_equal(executable->function_names[i], name)) {
      *out_function = iree_hal_executable_function_from_index((uint32_t)i);
      return iree_ok_status();
    }
  }

  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "function '%.*s' not found in executable",
                          (int)name.size, name.data);
}

static iree_status_t iree_hal_coralnpu_executable_lookup_global_by_name(
    iree_hal_executable_t *base_executable, iree_string_view_t name,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_buffer_t **out_buffer) {
  *out_buffer = NULL;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "executable global lookup not implemented");
}

static const iree_hal_executable_vtable_t iree_hal_coralnpu_executable_vtable =
    {
        .destroy = iree_hal_coralnpu_executable_destroy,
        .function_count = iree_hal_coralnpu_executable_function_count,
        .function_info = iree_hal_coralnpu_executable_function_info,
        .function_parameters = iree_hal_coralnpu_executable_function_parameters,
        .lookup_function_by_name =
            iree_hal_coralnpu_executable_lookup_function_by_name,
        .lookup_global_by_name =
            iree_hal_coralnpu_executable_lookup_global_by_name,
};

iree_status_t iree_hal_coralnpu_executable_create(
    const iree_hal_executable_params_t *executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t **out_executable) {
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;

  IREE_RETURN_IF_ERROR(iree_hal_coralnpu_executable_validate_elf32(
      executable_params->executable_data));

  uint32_t library_addr = 0;
  IREE_RETURN_IF_ERROR(iree_hal_coralnpu_executable_find_symbol(
      executable_params->executable_data,
      "iree_hal_executable_library_query_v0", &library_addr, NULL));

  const iree_hal_coralnpu_elf32_library_v0_t *library = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_coralnpu_executable_resolve_ptr(
      executable_params->executable_data, library_addr, sizeof(*library),
      (const void **)&library));

  const iree_hal_coralnpu_elf32_library_header_t *header = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_coralnpu_executable_resolve_ptr(
      executable_params->executable_data, library->header, sizeof(*header),
      (const void **)&header));
  if (header->version != IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "executable library version %u does not match runtime version %u",
        header->version, IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST);
  }

  const iree_host_size_t function_count = library->exports.count;
  if (function_count > 0 &&
      (library->exports.names == 0 || library->exports.attrs == 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "executable exports must provide names and dispatch attributes");
  }

  const iree_host_size_t image_size =
      executable_params->executable_data.data_length;
  iree_hal_coralnpu_executable_t *executable = NULL;
  iree_host_size_t total_size = 0;
  iree_host_size_t function_names_offset = 0;
  iree_host_size_t image_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(*executable), &total_size,
      IREE_STRUCT_FIELD_ALIGNED(function_count, iree_string_view_t,
                                iree_alignof(iree_string_view_t),
                                &function_names_offset),
      IREE_STRUCT_FIELD_ALIGNED(image_size, uint8_t, iree_max_align_t,
                                &image_offset)));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
      host_allocator, total_size, (void **)&executable));

  iree_hal_resource_initialize(&iree_hal_coralnpu_executable_vtable,
                               &executable->resource);
  executable->host_allocator = host_allocator;
  executable->function_count = function_count;
  executable->dispatch_attrs = NULL;
  executable->function_names =
      (iree_string_view_t *)((uint8_t *)executable + function_names_offset);

  uint8_t *image_storage = (uint8_t *)executable + image_offset;
  memcpy(image_storage, executable_params->executable_data.data, image_size);
  executable->dispatch_image =
      iree_make_const_byte_span(image_storage, image_size);

  iree_status_t status = iree_ok_status();
  if (function_count > 0) {
    status = iree_hal_coralnpu_executable_resolve_ptr(
        executable->dispatch_image, library->exports.attrs,
        function_count * sizeof(iree_hal_executable_dispatch_attrs_v0_t),
        (const void **)&executable->dispatch_attrs);
  }

  const uint32_t *name_addrs = NULL;
  if (iree_status_is_ok(status) && function_count > 0) {
    status = iree_hal_coralnpu_executable_resolve_ptr(
        executable->dispatch_image, library->exports.names,
        function_count * sizeof(uint32_t), (const void **)&name_addrs);
  }

  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < function_count;
       ++i) {
    status = iree_hal_coralnpu_executable_resolve_string(
        executable->dispatch_image, name_addrs[i],
        &executable->function_names[i]);
    if (iree_status_is_ok(status) &&
        iree_string_view_is_empty(executable->function_names[i])) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "executable export %" PRIhsz " missing function name", i);
    }
  }

  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, executable);
    return status;
  }

  *out_executable = (iree_hal_executable_t *)executable;
  return iree_ok_status();
}
