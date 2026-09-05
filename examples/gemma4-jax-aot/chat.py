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
"""Interactive chat client for Gemma 4 E2B (CPU and CoralNPU)."""

from __future__ import annotations

import argparse
from gemma4_util import (
    chat_loop,
    create_cpu_context,
    create_multidevice_context,
    generate,
    init_caches,
    load_and_flatten_params,
    load_kept_indices,
    make_mask,
    resolve_vmfb_path,
)
import numpy as np


def init_iree_func(
    cpu: bool = False,
    vmfb_path: str | None = None,
):
  """Loads Gemma 4 E2B VMFB module and returns the main step function."""
  primary = "gemma4_e2b_cpu.vmfb" if cpu else "gemma4_e2b.vmfb"
  fallback = "gemma4_e2b_coralnpu.vmfb" if not cpu else None
  path = resolve_vmfb_path(vmfb_path, primary, fallback)

  ctx = create_cpu_context([path]) if cpu else create_multidevice_context(path)
  try:
    return ctx.modules.jit_step_fn.main
  except AttributeError:
    return getattr(ctx.modules, list(ctx.modules.keys())[-1]).main


def run_inference(
    main_func,
    flat_params_np: list[np.ndarray],
    kept_indices: list[int],
    tokens_list: list[int],
    max_new_tokens: int = 1,
) -> list[int]:
  """Runs inference for Gemma 4 E2B with KV cache."""
  cache = init_caches(cache_length=128)

  def step_fn(tok_id: int, pos_idx: int) -> int:
    nonlocal cache
    tok = np.array([[tok_id]], dtype=np.int32)
    pos = np.array([[pos_idx]], dtype=np.int32)
    mask = make_mask(pos_idx + 1)
    all_args = flat_params_np + [tok, pos] + cache + [mask]
    call_args = [all_args[i] for i in kept_indices]
    res = main_func(*call_args)
    cache = list(res[1:])
    return int(np.asarray(res[0]).reshape(-1)[0])

  return generate(step_fn, tokens_list, max_new_tokens=max_new_tokens)


def chat(
    cpu: bool = False,
    max_new_tokens: int | None = None,
    full_history: bool | None = None,
    dummy_weights: bool = False,
    vmfb_path: str | None = None,
):
  """Runs interactive chat for Gemma 4 E2B."""
  if max_new_tokens is None:
    max_new_tokens = 128 if cpu else 1
  if full_history is None:
    full_history = True if cpu else False

  primary = "gemma4_e2b_cpu.vmfb" if cpu else "gemma4_e2b.vmfb"
  fallback = "gemma4_e2b_coralnpu.vmfb" if not cpu else None
  resolved_path = resolve_vmfb_path(vmfb_path, primary, fallback)
  main_func = init_iree_func(cpu=cpu, vmfb_path=resolved_path)
  kept_indices = load_kept_indices(resolved_path)
  flat_params = load_and_flatten_params(use_dummy=dummy_weights)

  def run_turn(tokens_list: list[int], max_tokens: int) -> list[int]:
    return run_inference(
        main_func,
        flat_params,
        kept_indices,
        tokens_list,
        max_new_tokens=max_tokens,
    )

  chat_loop(run_turn, full_history=full_history, max_new_tokens=max_new_tokens)


def main():
  # Options shared by the top-level parser and every subcommand.
  #
  # Defaults are SUPPRESS rather than real values so that an option parsed by
  # the top-level parser is not silently reset by the subparser. argparse
  # copies *every* key of the subparser's namespace over the top-level one, so
  # with concrete defaults "chat.py --cpu chat" would come back as cpu=False.
  # With SUPPRESS, unset options are absent from the namespace entirely and the
  # real defaults come from chat()'s own signature, which must stay in sync.
  common = argparse.ArgumentParser(add_help=False)
  common.add_argument(
      "--cpu",
      action="store_true",
      default=argparse.SUPPRESS,
      help="Run on CPU only (default: CPU + CoralNPU).",
  )
  common.add_argument(
      "--max_new_tokens",
      type=int,
      default=argparse.SUPPRESS,
      help=("Maximum new tokens to generate per turn (default: 1 for CoralNPU, "
            "128 for CPU). Note: token generation is slow in the CoralNPU "
            "simulator."),
  )
  common.add_argument(
      "--full-history",
      "--format_chat",
      dest="full_history",
      action=argparse.BooleanOptionalAction,
      default=argparse.SUPPRESS,
      help=(
          "Whether to retain full conversation history formatted with Gemma "
          "chat turn tokens (<|turn>user...<turn|>). Default is "
          "True for CPU, False for CoralNPU to keep prompt length minimal for "
          "fast simulation."),
  )
  common.add_argument(
      "--dummy_weights",
      action="store_true",
      default=argparse.SUPPRESS,
      help="Use synthetic zero weights for fast testing without GCS download.",
  )
  common.add_argument(
      "--vmfb_path",
      type=str,
      default=argparse.SUPPRESS,
      help=("Path to VMFB (default: ./gemma4_e2b.vmfb or"
            " ./gemma4_e2b_cpu.vmfb)."),
  )

  # This example only has the one model, so the subcommand is optional and
  # "chat" is implied. It is still accepted, to mirror the Gemma 3 example.
  parser = argparse.ArgumentParser(
      description="Gemma 4 E2B Interactive Chat (CoralNPU + CPU)",
      parents=[common],
  )
  parser.set_defaults(func=chat)
  subparsers = parser.add_subparsers()

  full_parser = subparsers.add_parser(
      "chat",
      aliases=["chat_gemma4"],
      parents=[common],
      help="Chat with the Gemma 4 E2B model.",
  )
  full_parser.set_defaults(func=chat)

  kwargs = vars(parser.parse_args())
  func = kwargs.pop("func")
  func(**kwargs)


if __name__ == "__main__":
  main()
