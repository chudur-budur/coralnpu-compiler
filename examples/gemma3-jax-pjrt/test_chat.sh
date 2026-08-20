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

main() {
  local cpu_only=false
  for arg in "$@"; do
    if [[ "${arg}" == "--cpu" ]]; then
      cpu_only=true
      break
    fi
  done

  echo "=== Phase 1: Building Targets ==="
  bazel build -c opt \
    @iree_core//lib:libIREECompiler.so \
    //pjrt_plugin:iree_pjrt_coralnpu_dylib \
    //compiler/tools:coralnpu-compile \
    //crt:coralnpu_tcm_highmem_ld

  # Link CRT and toolchain outputs so libIREECompiler.so in external/iree_core+/lib can resolve relative paths
  mkdir -p "${ROOT_DIR}/bazel-bin/external"
  ln -sf ../crt ../toolchain_rv32 "${ROOT_DIR}/bazel-bin/external/"

  echo
  export IREE_PJRT_COMPILER_LIB_PATH="${ROOT_DIR}/bazel-bin/external/iree_core+/lib/libIREECompiler.so"
  export PJRT_NAMES_AND_LIBRARY_PATHS="coralnpu_plugin:${ROOT_DIR}/bazel-bin/pjrt_plugin/libiree_pjrt_coralnpu_dylib.so"
  export IREE_PJRT_LOG_LEVEL=ERROR
  export ENABLE_PJRT_COMPATIBILITY=1
  export IREE_PJRT_CACHE_DIR="${ROOT_DIR}/.iree_pjrt_cache"
  export PYTHONUNBUFFERED=1

  local python="${PYTHON:-}"
  if [[ -z "${python}" ]]; then
    [[ -x "${ROOT_DIR}/venv/bin/python" ]] && python="${ROOT_DIR}/venv/bin/python" || python="python3"
  fi

  if ! "${python}" -c "import jax, gemma" &>/dev/null; then
    echo "Error: Required Python packages (jax, gemma) not found in current Python environment (${python})."
    echo "Please activate your virtual environment or run: pip install -r ${ROOT_DIR}/requirements_lock.txt"
    exit 1
  fi

  if [[ "${cpu_only}" == "true" ]]; then
    echo "=== Phase 2: Running on CPU ==="
    "${python}" "${SCRIPT_DIR}/chat.py" "$@" <<EOF
What is the capital of France?
What is the second largest city?
exit
EOF
  else
    echo "=== Phase 2: Running Multi-Device (CPU + CoralNPU) ==="
    export IREE_PJRT_IREE_COMPILER_OPTIONS="--coralnpu-linker-script-path=${ROOT_DIR}/bazel-bin/external/crt/coralnpu_tcm_highmem.ld"
    "${python}" "${SCRIPT_DIR}/chat.py" "$@" <<EOF
France capital is
exit
EOF
  fi

  echo
  echo "=== DONE ==="
}

main "$@"
