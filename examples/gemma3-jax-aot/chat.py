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
"""Interactive chat client for Gemma 3 270M (unified and 3-part split)."""

from __future__ import annotations

import argparse
from gemma3_util import (
    chat_loop,
    create_cpu_context,
    create_multidevice_context,
    generate,
    init_caches,
    init_split_caches,
    load_and_flatten_params,
    load_and_flatten_split_params,
    make_mask,
    resolve_vmfb_path,
)
import numpy as np


def init_iree_func(
    cpu: bool = False,
    vmfb_path: str | None = None,
):
  """Loads full Gemma 3 270M VMFB module and returns the main step function."""
  primary = "gemma3_270m_cpu.vmfb" if cpu else "gemma3_270m.vmfb"
  fallback = "gemma3_270m_coralnpu.vmfb" if not cpu else None
  path = resolve_vmfb_path(vmfb_path, primary, fallback)

  ctx = create_cpu_context([path]) if cpu else create_multidevice_context(path)
  try:
    return ctx.modules.jit_step_fn.main
  except AttributeError:
    return getattr(ctx.modules, list(ctx.modules.keys())[-1]).main


def init_iree_funcs(
    cpu: bool = False,
    part1_path: str | None = None,
    part2_path: str | None = None,
    part3_path: str | None = None,
):
  """Loads 3-part split Gemma 3 270M VMFB modules."""
  s = "_cpu" if cpu else ""
  p1 = resolve_vmfb_path(part1_path, f"gemma3_270m{s}_part1.vmfb")
  p2 = resolve_vmfb_path(part2_path, f"gemma3_270m{s}_part2.vmfb")
  p3 = resolve_vmfb_path(part3_path, f"gemma3_270m{s}_part3.vmfb")

  if cpu:
    ctx = create_cpu_context([p1, p2, p3])
    return (
        ctx.modules.jit_p1_dec.main,
        ctx.modules.jit_p2_dec.main,
        ctx.modules.jit_p3_dec.main,
    )

  ctx1 = create_multidevice_context(p1)
  ctx2 = create_multidevice_context(p2)
  ctx3 = create_multidevice_context(p3)
  return (
      ctx1.modules.jit_p1_dec.main,
      ctx2.modules.jit_p2_dec.main,
      ctx3.modules.jit_p3_dec.main,
  )


def run_inference(
    main_func,
    flat_params_np: list[np.ndarray],
    tokens_list: list[int],
    max_new_tokens: int = 128,
) -> list[int]:
  """Runs inference for full Gemma 3 270M with KV cache."""
  cache = init_caches()

  def step_fn(tok_id: int, pos_idx: int) -> int:
    nonlocal cache
    tok = np.array([[tok_id]], dtype=np.int32)
    pos = np.array([[pos_idx]], dtype=np.int32)
    mask = make_mask(pos_idx + 1)
    res = main_func(*(flat_params_np + [tok, pos] + cache + [mask]))
    cache = list(res[1:])
    return int(np.asarray(res[0])[0, 0])

  return generate(step_fn, tokens_list, max_new_tokens=max_new_tokens)


def run_split_inference(
    m1_dec,
    m2_dec,
    m3_dec,
    p1_np: list[np.ndarray],
    p2_np: list[np.ndarray],
    p3_np: list[np.ndarray],
    tokens_list: list[int],
    max_new_tokens: int = 128,
) -> list[int]:
  """Runs inference for 3-part split Gemma 3 270M with KV cache."""
  c1, c2 = init_split_caches()

  def step_fn(tok_id: int, pos_idx: int) -> int:
    nonlocal c1, c2
    tok = np.array([[tok_id]], dtype=np.int32)
    pos = np.array([[pos_idx]], dtype=np.int32)
    mask = make_mask(pos_idx + 1)
    res1 = m1_dec(*(p1_np + [tok, pos] + c1 + [mask]))
    c1 = list(res1[1:])
    res2 = m2_dec(*(p2_np + [res1[0], pos] + c2 + [mask]))
    c2 = list(res2[1:])
    res3 = m3_dec(*(p3_np + [res2[0]]))
    return int(np.asarray(res3)[0, 0])

  return generate(step_fn, tokens_list, max_new_tokens=max_new_tokens)


def chat(
    cpu: bool = False,
    max_new_tokens: int | None = None,
    full_history: bool | None = None,
    vmfb_path: str | None = None,
):
  """Runs interactive chat for the unified full Gemma 3 270M model."""
  if max_new_tokens is None:
    max_new_tokens = 128 if cpu else 1
  if full_history is None:
    full_history = True if cpu else False

  main_func = init_iree_func(cpu=cpu, vmfb_path=vmfb_path)
  flat_params = load_and_flatten_params()

  def run_turn(tokens_list, max_tokens):
    return run_inference(main_func,
                         flat_params,
                         tokens_list,
                         max_new_tokens=max_tokens)

  chat_loop(run_turn, full_history=full_history, max_new_tokens=max_new_tokens)


def chat_split(
    cpu: bool = False,
    max_new_tokens: int | None = None,
    full_history: bool | None = None,
    part1_vmfb_path: str | None = None,
    part2_vmfb_path: str | None = None,
    part3_vmfb_path: str | None = None,
):
  """Runs interactive chat for the 3-part split Gemma 3 270M model."""
  if max_new_tokens is None:
    max_new_tokens = 128 if cpu else 1
  if full_history is None:
    full_history = True if cpu else False

  m1, m2, m3 = init_iree_funcs(
      cpu=cpu,
      part1_path=part1_vmfb_path,
      part2_path=part2_vmfb_path,
      part3_path=part3_vmfb_path,
  )
  p1, p2, p3 = load_and_flatten_split_params()

  def run_turn(tokens_list, max_tokens):
    return run_split_inference(m1,
                               m2,
                               m3,
                               p1,
                               p2,
                               p3,
                               tokens_list,
                               max_new_tokens=max_tokens)

  chat_loop(run_turn, full_history=full_history, max_new_tokens=max_new_tokens)


def main():
  # Options shared by all subcommands.
  common = argparse.ArgumentParser(add_help=False)
  common.add_argument(
      "--cpu",
      action="store_true",
      default=False,
      help="Run on CPU only (default: CPU + CoralNPU).",
  )
  common.add_argument(
      "--max_new_tokens",
      type=int,
      default=None,
      help=("Maximum new tokens to generate per turn (default: 1 for CoralNPU, "
            "128 for CPU)."),
  )
  common.add_argument(
      "--full-history",
      action=argparse.BooleanOptionalAction,
      default=None,
      help=("Whether to retain full conversation history formatted with Gemma "
            "chat turn tokens. Default is True for CPU, False for CoralNPU."),
  )

  parser = argparse.ArgumentParser(
      description="Gemma 3 270M Chat (Unified or Split)")
  subparsers = parser.add_subparsers(required=True)

  full_parser = subparsers.add_parser(
      "chat",
      parents=[common],
      help="Chat with the unified model.",
  )
  full_parser.add_argument("--vmfb_path",
                           default=None,
                           help="Path to full VMFB.")
  full_parser.set_defaults(func=chat)

  split_parser = subparsers.add_parser(
      "chat_split",
      parents=[common],
      help="Chat with the 3-part split model.",
  )
  split_parser.add_argument("--part1_vmfb_path",
                            default=None,
                            help="Path to Part 1 VMFB.")
  split_parser.add_argument("--part2_vmfb_path",
                            default=None,
                            help="Path to Part 2 VMFB.")
  split_parser.add_argument("--part3_vmfb_path",
                            default=None,
                            help="Path to Part 3 VMFB.")
  split_parser.set_defaults(func=chat_split)

  kwargs = vars(parser.parse_args())
  func = kwargs.pop("func")
  func(**kwargs)


if __name__ == "__main__":
  main()
