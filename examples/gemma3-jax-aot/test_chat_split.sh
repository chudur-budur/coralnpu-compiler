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
  echo "=== Phase 1: Generating StableHLO MLIR ==="
  echo '| inputs:'
  echo '|   gm.nn.Gemma3_270M()'
  echo '|   gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_270M_IT)'
  echo '| outputs:'
  echo '|   gemma3_270m_part1.mlir'
  echo '|   gemma3_270m_part2.mlir'
  echo '|   gemma3_270m_part3.mlir'
  bazel build --config=dev //examples/gemma3-jax-aot:export_gemma
  "${ROOT_DIR}/bazel-bin/examples/gemma3-jax-aot/export_gemma" \
    export_gemma_split

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
    echo '| inputs:'
    echo '|   gemma3_270m_part1.mlir'
    echo '|   gemma3_270m_part2.mlir'
    echo '|   gemma3_270m_part3.mlir'
    echo '| outputs:'
    echo '|   gemma3_270m_cpu_part1.vmfb (CPU)'
    echo '|   gemma3_270m_cpu_part2.vmfb (CPU)'
    echo '|   gemma3_270m_cpu_part3.vmfb (CPU)'

    bazel build --config=dev @iree_core//tools:iree-compile

    local -a compile_options=()
    compile_options+=('--iree-hal-target-device=local')
    compile_options+=('--iree-hal-local-target-device-backends=llvm-cpu')
    compile_options+=('--iree-llvmcpu-target-cpu=host')

    echo "Compiling Part 1 (Layers 0..8) to CPU..."
    "${ROOT_DIR}/bazel-bin/external/iree_core+/tools/iree-compile" \
      "${compile_options[@]}" \
      "${PWD}/gemma3_270m_part1.mlir" \
      -o "${PWD}/gemma3_270m_cpu_part1.vmfb" &

    echo "Compiling Part 2 (Layers 9..17 + Final Norm) to CPU..."
    "${ROOT_DIR}/bazel-bin/external/iree_core+/tools/iree-compile" \
      "${compile_options[@]}" \
      "${PWD}/gemma3_270m_part2.mlir" \
      -o "${PWD}/gemma3_270m_cpu_part2.vmfb" &

    echo "Compiling Part 3 (Logits Decode) to CPU..."
    "${ROOT_DIR}/bazel-bin/external/iree_core+/tools/iree-compile" \
      "${compile_options[@]}" \
      "${PWD}/gemma3_270m_part3.mlir" \
      -o "${PWD}/gemma3_270m_cpu_part3.vmfb" &

    echo "Waiting for background compilation jobs to finish..."
    wait
  else
    echo
    echo "=== Phase 2: (building IREE and) Compiling to CoralNPU + CPU VMFB ==="
    echo '| inputs:'
    echo '|   gemma3_270m_part1.mlir'
    echo '|   gemma3_270m_part2.mlir'
    echo '|   gemma3_270m_part3.mlir'
    echo '| outputs:'
    echo '|   gemma3_270m_part1.vmfb (CoralNPU + CPU)'
    echo '|   gemma3_270m_part2.vmfb (CoralNPU + CPU)'
    echo '|   gemma3_270m_part3.vmfb (CoralNPU + CPU)'

    bazel build --config=dev \
      @iree_core//tools:iree-compile \
      //crt:coralnpu_tcm_highmem_ld

    local -a compile_options=()
    compile_options+=('--iree-hal-target-device=local')
    compile_options+=('--iree-hal-local-target-device-backends=llvm-cpu')
    compile_options+=('--iree-llvmcpu-target-cpu=host')
    compile_options+=('--iree-hal-target-device=coralnpu')
    compile_options+=('--coralnpu-dump-affinity-profile-format=pretty')
    local ld_path="${ROOT_DIR}/bazel-bin/crt/coralnpu_tcm_highmem.ld"
    compile_options+=("--coralnpu-linker-script-path=${ld_path}")

    echo "Compiling Part 1 (Layers 0..8) to CoralNPU + CPU..."
    "${ROOT_DIR}/bazel-bin/external/iree_core+/tools/iree-compile" \
      "${compile_options[@]}" \
      "${PWD}/gemma3_270m_part1.mlir" \
      -o "${PWD}/gemma3_270m_part1.vmfb" &

    echo "Compiling Part 2 (Layers 9..17 + Final Norm) to CoralNPU + CPU..."
    "${ROOT_DIR}/bazel-bin/external/iree_core+/tools/iree-compile" \
      "${compile_options[@]}" \
      "${PWD}/gemma3_270m_part2.mlir" \
      -o "${PWD}/gemma3_270m_part2.vmfb" &

    echo "Compiling Part 3 (Logits Decode) to CoralNPU + CPU..."
    "${ROOT_DIR}/bazel-bin/external/iree_core+/tools/iree-compile" \
      "${compile_options[@]}" \
      "${PWD}/gemma3_270m_part3.mlir" \
      -o "${PWD}/gemma3_270m_part3.vmfb" &

    echo "Waiting for background compilation jobs to finish..."
    wait
  fi

  echo
  echo "=== Phase 3: Build IREE runtime and python bindings ==="
  bazel build --config=dev \
    @iree_core//runtime/bindings/python:runtime \
    //examples/gemma3-jax-aot:chat

  echo
  if [[ "${cpu_only}" == "true" ]]; then
    echo "=== Phase 4: Running on CPU ==="
    echo '| inputs:'
    echo '|   gemma3_270m_cpu_part1.vmfb (CPU)'
    echo '|   gemma3_270m_cpu_part2.vmfb (CPU)'
    echo '|   gemma3_270m_cpu_part3.vmfb (CPU)'

    (
      export LD_LIBRARY_PATH="${ROOT_DIR}/runtime/sim"
      "${ROOT_DIR}/bazel-bin/examples/gemma3-jax-aot/chat" chat_split "$@" <<EOF
What is the capital of France?
What is the second largest city?
exit
EOF
    )
  else
    echo "=== Phase 4: Running on CoralNPU + CPU ==="
    echo '| inputs:'
    echo '|   gemma3_270m_part1.vmfb (CoralNPU + CPU)'
    echo '|   gemma3_270m_part2.vmfb (CoralNPU + CPU)'
    echo '|   gemma3_270m_part3.vmfb (CoralNPU + CPU)'

    (
      export LD_LIBRARY_PATH="${ROOT_DIR}/runtime/sim"
      "${ROOT_DIR}/bazel-bin/examples/gemma3-jax-aot/chat" chat_split "$@" <<EOF
France capital is
exit
EOF
    )
  fi

  echo
  echo "=== DONE ==="
}

main "$@"
