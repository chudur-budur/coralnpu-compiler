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
"""Interactive chat client for Gemma 3 270M."""

from __future__ import annotations

from gemma3_util import (
    STOP_TOKENS,
    chat_loop,
    init_caches,
    load_and_flatten_params,
)
from gemma_common import make_chat_parser
from gemma_iree_runtime import (
    load_step_funcs,
    make_aot_run_turn,
    to_device_arrays,
)


def chat(
    cpu: bool = False,
    max_new_tokens: int = 128,
    full_history: bool = True,
    vmfb_path: str | None = None,
    decode_vmfb_path: str | None = None,
    prefill_vmfb_path: str | None = None,
):
  """Runs interactive chat for Gemma 3 270M."""
  steps, device, _ = load_step_funcs(
      "gemma3_270m",
      __file__,
      cpu=cpu,
      vmfb_path=vmfb_path,
      decode_vmfb_path=decode_vmfb_path,
      prefill_vmfb_path=prefill_vmfb_path,
  )
  flat_params = to_device_arrays(device, load_and_flatten_params())
  run_turn = make_aot_run_turn(steps, flat_params, init_caches(), STOP_TOKENS)
  chat_loop(run_turn, full_history=full_history, max_new_tokens=max_new_tokens)


def main():
  parser = make_chat_parser("Gemma 3 270M Chat", vmfb_stem="gemma3_270m")
  chat(**vars(parser.parse_args()))


if __name__ == "__main__":
  main()
