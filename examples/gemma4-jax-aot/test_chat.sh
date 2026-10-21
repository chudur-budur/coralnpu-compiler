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

  local cpu_only=false
  for arg in "$@"; do
    if [[ "${arg}" == "--cpu" ]]; then
      cpu_only=true
      break
    fi
  done

  echo "=== Phase 1: Generating StableHLO MLIR ==="
  echo '| inputs:'
  echo '|   gm.nn.Gemma4_E2B(text_only=True)'
  echo '| outputs:'
  echo '|   gemma4_e2b.mlir (16-token step)'
  echo '|   gemma4_e2b_prefill.mlir (32-token prefill step)'
  echo '|   gemma4_e2b_decode.mlir (1-token decode step)'
  bazel run --config=dev //examples/gemma4-jax-aot:export_gemma -- \
    --output_path "${PWD}/gemma4_e2b.mlir"

  local mode_desc="CoralNPU + CPU"
  local -a stems=(gemma4_e2b gemma4_e2b_prefill gemma4_e2b_decode)
  local -a vmfbs=("${stems[@]/%/.vmfb}")
  local -a extra_chat_args=(
    --prefill_vmfb_path "${PWD}/gemma4_e2b_prefill.vmfb"
    --decode_vmfb_path "${PWD}/gemma4_e2b_decode.vmfb"
  )
  local -a bazel_targets=(
    "@iree_core//tools:iree-compile"
    "//crt:coralnpu_tcm_highmem_ld"
  )

  if [[ "${cpu_only}" == "true" ]]; then
    mode_desc="CPU"
    stems=(gemma4_e2b)
    vmfbs=(gemma4_e2b_cpu.vmfb)
    extra_chat_args=()
    bazel_targets=("@iree_core//tools:iree-compile")
  fi

  echo
  echo "=== Phase 2: (building IREE and) Compiling to ${mode_desc} VMFB ==="
  echo '| inputs:'
  for stem in "${stems[@]}"; do echo "|   ${stem}.mlir"; done
  echo '| outputs:'
  for vmfb in "${vmfbs[@]}"; do echo "|   ${vmfb} (${mode_desc})"; done

  bazel build --config=dev "${bazel_targets[@]}"

  local -a compile_options=()
  compile_options+=('--iree-hal-target-device=local')
  compile_options+=('--iree-hal-local-target-device-backends=llvm-cpu')
  compile_options+=('--iree-llvmcpu-target-cpu=host')

  if [[ "${cpu_only}" != "true" ]]; then
    local ld_script="${ROOT_DIR}/bazel-bin/crt/coralnpu_tcm_highmem.ld"
    compile_options+=('--iree-hal-target-device=coralnpu')
    compile_options+=('--coralnpu-dump-affinity-profile-format=pretty')
    compile_options+=('--coralnpu-dtcm-size-kb=1024')
    compile_options+=("--coralnpu-linker-script-path=${ld_script}")
  fi

  echo "Compiling to ${mode_desc}..."
  local -a pids=()
  for i in "${!stems[@]}"; do
    "${ROOT_DIR}/bazel-bin/external/iree_core+/tools/iree-compile" \
      "${compile_options[@]}" \
      "${PWD}/${stems[i]}.mlir" \
      -o "${PWD}/${vmfbs[i]}" &
    pids+=("$!")
  done
  # Wait per PID: a bare wait returns 0 even if a compile failed, and
  # `jobs -p` omits a compile that already exited.
  for pid in "${pids[@]}"; do wait "${pid}"; done

  echo
  echo "=== Phase 3: Build IREE runtime and python bindings ==="
  bazel build --config=dev \
    @iree_core//runtime/bindings/python:runtime \
    //examples/gemma4-jax-aot:chat

  echo
  echo "=== Phase 4: Running on ${mode_desc} ==="
  echo '| inputs:'
  for vmfb in "${vmfbs[@]}"; do echo "|   ${vmfb} (${mode_desc})"; done

  "${ROOT_DIR}/bazel-bin/examples/gemma4-jax-aot/chat" \
    ${extra_chat_args[@]+"${extra_chat_args[@]}"} "$@" <<EOF
What is the capital of France?
And the second largest city?
exit
EOF

  echo
  echo "=== DONE ==="
}

main "$@"
