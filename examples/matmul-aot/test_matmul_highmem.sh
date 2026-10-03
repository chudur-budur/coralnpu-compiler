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

# Runs test_matmul.sh with 1 MB ITCM/DTCM on MPACT (default N=128).

# Exit immediately on error, or when accessing an unset variable
set -euo pipefail

HIGHMEM=true N="${N:-128}" exec "$(dirname "${BASH_SOURCE[0]}")/test_matmul.sh" "$@"
