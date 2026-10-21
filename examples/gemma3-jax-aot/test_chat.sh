#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Exit immediately on error (including in a pipeline), or when accessing an
# unset variable
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SCRIPT_DIR}"

main() {
  # Kill background compiles if the script exits early.
  trap 'kill $(jobs -p) 2>/dev/null || true' EXIT

  echo "=== Phase 1: Generating StableHLO MLIR ==="
  echo '| inputs:'
  echo '|   gm.nn.Gemma3_270M()'
  echo '|   gemma3_util.get_abstract_params()'
  echo '| outputs:'
  echo '|   gemma3_270m.mlir (16-token step)'
  echo '|   gemma3_270m_prefill.mlir (32-token prefill step)'
  echo '|   gemma3_270m_decode.mlir (1-token decode step)'
  bazel build --config=dev //examples/gemma3-jax-aot:export_gemma
  "${ROOT_DIR}/bazel-bin/examples/gemma3-jax-aot/export_gemma"

  local cpu_only=false
  for arg in "$@"; do
    if [[ "${arg}" == "--cpu" ]]; then
      cpu_only=true
      break
    fi
  done

  if [[ "${cpu_only}" == "true" ]]; then
    echo
    echo "=== Phase 2: (building IREE and) Compiling to CPU VMFB ==="
    echo '| input:'
    echo '|   gemma3_270m.mlir'
    echo '| output:'
    echo '|   gemma3_270m_cpu.vmfb (CPU)'

    bazel build --config=dev @iree_core//tools:iree-compile

    local -a compile_options=()
    compile_options+=('--iree-hal-target-device=local')
    compile_options+=('--iree-hal-local-target-device-backends=llvm-cpu')
    compile_options+=('--iree-llvmcpu-target-cpu=host')

    echo "Compiling to CPU..."
    "${ROOT_DIR}/bazel-bin/external/iree_core+/tools/iree-compile" \
      "${compile_options[@]}" \
      "${PWD}/gemma3_270m.mlir" \
      -o "${PWD}/gemma3_270m_cpu.vmfb"
  else
    echo
    echo "=== Phase 2: (building IREE and) Compiling to CoralNPU + CPU VMFB ==="
    echo '| inputs:'
    echo '|   gemma3_270m.mlir'
    echo '|   gemma3_270m_prefill.mlir'
    echo '|   gemma3_270m_decode.mlir'
    echo '| outputs:'
    echo '|   gemma3_270m.vmfb (CoralNPU + CPU)'
    echo '|   gemma3_270m_prefill.vmfb (CoralNPU + CPU)'
    echo '|   gemma3_270m_decode.vmfb (CoralNPU + CPU)'

    bazel build --config=dev \
      @iree_core//tools:iree-compile \
      //crt:coralnpu_tcm_highmem_ld

    local -a compile_options=()
    compile_options+=('--iree-hal-target-device=local')
    compile_options+=('--iree-hal-local-target-device-backends=llvm-cpu')
    compile_options+=('--iree-llvmcpu-target-cpu=host')
    compile_options+=('--iree-hal-target-device=coralnpu')
    compile_options+=('--coralnpu-dump-affinity-profile-format=pretty')
    # Must match the DTCM size of //crt:coralnpu_tcm_highmem_ld (1 MiB).
    compile_options+=('--coralnpu-dtcm-size-kb=1024')
    local ld_path="${ROOT_DIR}/bazel-bin/crt/coralnpu_tcm_highmem.ld"
    compile_options+=("--coralnpu-linker-script-path=${ld_path}")

    echo "Compiling to CoralNPU + CPU..."
    local -a pids=()
    for stem in gemma3_270m gemma3_270m_prefill gemma3_270m_decode; do
      "${ROOT_DIR}/bazel-bin/external/iree_core+/tools/iree-compile" \
        "${compile_options[@]}" \
        "${PWD}/${stem}.mlir" \
        -o "${PWD}/${stem}.vmfb" &
      pids+=("$!")
    done
    # Wait per PID: a bare wait returns 0 even if a compile failed, and
    # `jobs -p` omits a compile that already exited.
    for pid in "${pids[@]}"; do wait "${pid}"; done
  fi

  echo
  echo "=== Phase 3: Build IREE runtime and python bindings ==="
  bazel build --config=dev \
    @iree_core//runtime/bindings/python:runtime \
    //examples/gemma3-jax-aot:chat

  echo
  local -a extra_chat_args=()
  if [[ "${cpu_only}" == "true" ]]; then
    echo "=== Phase 4: Running on CPU ==="
    echo '| input:'
    echo '|   gemma3_270m_cpu.vmfb (CPU)'
  else
    echo "=== Phase 4: Running on CoralNPU + CPU ==="
    echo '| inputs:'
    echo '|   gemma3_270m.vmfb (CoralNPU + CPU)'
    echo '|   gemma3_270m_prefill.vmfb (CoralNPU + CPU)'
    echo '|   gemma3_270m_decode.vmfb (CoralNPU + CPU)'
    extra_chat_args=(
      --prefill_vmfb_path "${PWD}/gemma3_270m_prefill.vmfb"
      --decode_vmfb_path "${PWD}/gemma3_270m_decode.vmfb"
    )
  fi

  "${ROOT_DIR}/bazel-bin/examples/gemma3-jax-aot/chat" \
    ${extra_chat_args[@]+"${extra_chat_args[@]}"} "$@" <<EOF
What is the capital of France?
And the second largest city?
exit
EOF

  echo
  echo "=== DONE ==="
}

main "$@"
