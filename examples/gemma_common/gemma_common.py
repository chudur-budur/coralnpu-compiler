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
"""Helpers shared by the Gemma 3 and Gemma 4 JAX examples."""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
import typing
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np

CACHE_LEN = 128
SEQ_LEN = 16
PREFILL_SEQ_LEN = 32
DECODE_SEQ_LEN = 1
STEP_LENS = (SEQ_LEN, PREFILL_SEQ_LEN, DECODE_SEQ_LEN)


def make_fast_einsum(base_cls: type[nn.Module]) -> type[nn.Module]:
  """Returns an `Einsum` subclass that lowers K x N weights to 2D dots."""

  class FastEinsum(base_cls):
    """Lowers projections with `transpose_params` weights as 2D matmuls."""

    @nn.compact
    def __call__(self, eqn: str,
                 x: jax.Array) -> jax.Array | tuple[jax.Array, ...]:
      if eqn == "BTD,NDH->BTNH":
        n, d, h = self.shape
        param_shape = (d, n, h)
      elif eqn == "BSD,CKDH->CBSKH":
        c, k, d, h = self.shape
        param_shape = (c, d, k, h)
      elif eqn == "...F,NHF->...NH":
        n, h, f = self.shape
        param_shape = (f, n, h)
      else:
        param_shape = self.shape
      w = self.param(self.weight_name, self.initializer, param_shape,
                     self.dtype)
      if isinstance(w, dict) and "w" in w:  # Shared-scope parameter.
        w = w["w"]
      if getattr(self, "w_scale", None) is not None:  # Gemma 4 only.
        w = w * self.w_scale
      if eqn in ("BTD,NDH->BTNH", "...F,NHF->...NH", "...td,dnp->...tnp"):
        return jnp.dot(x, w.reshape(param_shape[0],
                                    -1)).reshape(*x.shape[:-1],
                                                 *param_shape[1:])
      if eqn == "BSD,CKDH->CBSKH":
        w_2d = w.reshape(c, d, k * h)
        return tuple(
            jnp.dot(x, w_2d[i]).reshape(*x.shape[:-1], k, h) for i in range(c))
      if eqn == "BTNH,NHD->BTD":
        return jnp.dot(x.reshape(*x.shape[:-2], -1),
                       w.reshape(-1, self.shape[2]))
      return jnp.einsum(eqn, x, w)

  return FastEinsum


def transpose_params(tree: typing.Any) -> typing.Any:
  """Reorders q/kv/gating weight axes into `make_fast_einsum` layouts."""
  if not isinstance(tree, dict):
    return tree

  def moved(a, src, dst):
    return np.ascontiguousarray(np.moveaxis(np.asarray(a), src, dst))

  out = {}
  for k, v in tree.items():
    if k == "q_einsum":  # (N, D, H) -> (D, N, H)
      out[k] = jax.tree.map(lambda a: moved(a, 0, 1), v)
    elif k == "kv_einsum":  # (C, K, D, H) -> (C, D, K, H)
      out[k] = jax.tree.map(lambda a: moved(a, 1, 2), v)
    elif k == "gating_einsum":  # (2, H, F) -> (F, 2, H)
      out[k] = jax.tree.map(lambda a: moved(a, -1, 0), v)
    else:
      out[k] = transpose_params(v)
  return out


def step_inputs(chunk: list[int], pos_idx: int,
                seq_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Returns padded tokens, positions and causal mask of a `seq_len` step."""
  tok = np.zeros((1, seq_len), dtype=np.int32)
  tok[0, :len(chunk)] = chunk
  pos = np.arange(pos_idx, pos_idx + seq_len, dtype=np.int32)[None, :]
  mask = (np.arange(CACHE_LEN) <= pos[:, :, None]).astype(np.int8)
  return tok, pos, mask


def make_run_turn(
    step_fn: typing.Callable[[list[int], int, int], int],
    stop_tokens: typing.Collection[int],
    step_lens: typing.Collection[int],
) -> typing.Callable[[list[int], int], list[int]]:
  """Returns `run_turn(tokens_list, max_tokens)` for one chat session."""
  cached_tokens: list[int] = []

  def run_turn(tokens_list: list[int], max_tokens: int) -> list[int]:
    return generate(step_fn, tokens_list, stop_tokens, max_tokens, step_lens,
                    cached_tokens)

  return run_turn


def export_step_mlirs(
    step_fn: typing.Callable[..., typing.Any],
    params: typing.Any,
    cache: typing.Any,
    output_path: str,
    cache_length: int = CACHE_LEN,
) -> list[int]:
  """Lowers the 3 step lengths to StableHLO MLIR files; returns kept indices."""
  if cache_length < PREFILL_SEQ_LEN:
    raise ValueError(
        f"cache_length ({cache_length}) must be >= {PREFILL_SEQ_LEN}.")
  os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
  stem = os.path.splitext(output_path)[0]
  jax.config.update("jax_use_shardy_partitioner", False)
  # Print the MLIR without source locations.
  jax.config.update("jax_include_debug_info_in_dumps", False)
  fn = getattr(step_fn, "__wrapped__", step_fn)
  kept_indices = None
  for seq_len, path, name in (
      (SEQ_LEN, output_path, "step_fn"),
      (PREFILL_SEQ_LEN, f"{stem}_prefill.mlir", "prefill_fn"),
      (DECODE_SEQ_LEN, f"{stem}_decode.mlir", "decode_fn"),
  ):
    # JAX names the module after the function: distinct names let one context
    # load the VMFBs of all step lengths.
    def named_fn(*args):
      return fn(*args)

    named_fn.__name__ = name
    tok = jax.ShapeDtypeStruct((1, seq_len), jnp.int32)
    mask = jax.ShapeDtypeStruct((1, seq_len, cache_length), jnp.int8)
    print(f"Lowering {seq_len}-token step to StableHLO ({path})...")
    t0 = time.time()
    exported = jax.export.export(jax.jit(named_fn))(params, tok, tok, cache,
                                                    mask)
    # JAX drops unused arguments; return the kept subset for make_aot_run_turn.
    kept = list(exported.module_kept_var_idx)
    if kept_indices not in (None, kept):
      raise ValueError(f"The {seq_len}-token step keeps other arguments.")
    kept_indices = kept
    with open(path, "w") as f:
      f.write(exported.mlir_module())
    print(f"{seq_len}-token step MLIR exported in {time.time() - t0:.2f}s")
  return kept_indices


def generate(
    step_fn: typing.Callable[[list[int], int, int], int],
    tokens_list: list[int],
    stop_tokens: typing.Collection[int],
    max_new_tokens: int,
    step_lens: typing.Collection[int],
    cached_tokens: list[int],
) -> list[int]:
  """Autoregressively generates tokens using step_fn(chunk, pos_idx, seq_len).

  Prefill resumes after the `SEQ_LEN`-aligned prompt prefix already in
  `cached_tokens` (the KV cache contents, updated in place), in chunks of up
  to `max(step_lens)` tokens.
  """
  decode_len, prefill_len = min(step_lens), max(step_lens)
  prompt_len = len(tokens_list)
  start_pos = 0
  for a, b in zip(cached_tokens, tokens_list[:-1]):
    if a != b:
      break
    start_pos += 1
  # Align chunk start to SEQ_LEN so fixed-size steps fit in the cache.
  start_pos = (start_pos // SEQ_LEN) * SEQ_LEN
  reused = f" (reusing {start_pos} cached tokens)" if start_pos else ""
  print(f"[Debug] Prefilling {prompt_len - start_pos} tokens{reused}...")

  pos = start_pos
  while pos < prompt_len:
    # A step writes seq_len cache slots: use the prefill step only if it fits.
    chunk_len = (prefill_len if prompt_len - pos > SEQ_LEN and
                 pos <= CACHE_LEN - prefill_len else SEQ_LEN)
    chunk = tokens_list[pos:pos + chunk_len]
    seq_len = min(s for s in step_lens if s >= len(chunk))
    step_tok = step_fn(chunk, pos, seq_len)
    cached_tokens[pos:] = chunk
    pos += len(chunk)

  generated = [step_tok]
  if step_tok in stop_tokens:
    return generated

  curr_len = prompt_len
  curr_tok = step_tok
  # Stop once the decode step's cache slots would run past the cache.
  while curr_len <= CACHE_LEN - decode_len and len(generated) < max_new_tokens:
    cached_tokens[curr_len:] = [curr_tok]
    curr_tok = step_fn([curr_tok], curr_len, decode_len)
    generated.append(curr_tok)
    if curr_tok in stop_tokens:
      break
    curr_len += 1

  if len(generated) < max_new_tokens and generated[-1] not in stop_tokens:
    print(f"[Info] Reached KV cache limit ({CACHE_LEN} tokens).")
  return generated


def decode_response(tokenizer: typing.Any, output_tokens: list[int],
                    stop_tokens: typing.Collection[int]) -> str:
  """Decodes token IDs into text up to the first stop token."""
  return tokenizer.decode([
      int(t) for t in itertools.takewhile(lambda t: t not in stop_tokens,
                                          output_tokens)
  ])


def chat_loop(
    run_turn_fn: typing.Callable[[list[int], int], list[int]],
    tokenizer: typing.Any,
    format_prompt: typing.Callable[[str, str, bool], str],
    end_of_turn: str,
    stop_tokens: typing.Collection[int],
    full_history: bool,
    max_new_tokens: int,
) -> None:
  """Interactive multi-turn chat loop over `run_turn_fn(tokens, max_tokens)`."""
  print("\nInteractive Chat started. Type 'exit' or 'quit' to end.")

  history = ""
  turn = 1
  while True:
    print(f"\n--- Turn {turn} ---\n")

    try:
      user_input = input("User: ").strip()
    except (EOFError, KeyboardInterrupt):
      user_input = "exit"
    if not sys.stdin.isatty():
      print(user_input)

    if not user_input or user_input.lower() in ("exit", "quit"):
      print("Done")
      break

    prompt = format_prompt(history, user_input, full_history)
    tokens_list = [int(t) for t in tokenizer.encode(prompt, add_bos=True)]
    print(f"[Debug] Prompt length: {len(tokens_list)} tokens")

    if len(tokens_list) > CACHE_LEN:
      print("ERROR: History exceeds limit.")
      break
    if 0 in tokens_list:  # Padded step slots hold token 0.
      print("ERROR: Prompt contains the pad token.")
      break

    t0 = time.time()
    output_tokens = run_turn_fn(tokens_list, max_new_tokens)
    t1 = time.time()

    model_response = decode_response(tokenizer, output_tokens, stop_tokens)
    print(f"Model: {model_response}")
    print(f"[Inference took {t1 - t0:.2f}s]")

    if full_history:
      history = prompt + model_response + end_of_turn
    else:
      history = f"{prompt} {model_response}".strip()

    turn += 1


def make_chat_parser(
    description: str,
    *,
    vmfb_stem: str | None = None,
    default_max_new_tokens: int | None = 128,
    max_new_tokens_help: str = "default: 128",
    accept_format_chat: bool = False,
) -> argparse.ArgumentParser:
  """Builds the shared CLI parser; `vmfb_stem` adds the AOT VMFB flags."""
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument(
      "--cpu",
      action="store_true",
      default=False,
      help="Run on CPU only (default: CPU + CoralNPU).",
  )
  parser.add_argument(
      "--max_new_tokens",
      type=int,
      default=default_max_new_tokens,
      help=f"Maximum new tokens to generate per turn ({max_new_tokens_help}).",
  )
  history_flags = (("--full-history",
                    "--format_chat") if accept_format_chat else
                   ("--full-history",))
  parser.add_argument(
      *history_flags,
      dest="full_history",
      action=argparse.BooleanOptionalAction,
      default=True,
      help=("Whether to retain full conversation history formatted with Gemma "
            "chat turn tokens (default: True)."),
  )
  if vmfb_stem:
    parser.add_argument(
        "--vmfb_path",
        default=None,
        help=(f"Path to the {SEQ_LEN}-token step VMFB (default: "
              f"{vmfb_stem}.vmfb, or {vmfb_stem}_cpu.vmfb with --cpu)."),
    )
    parser.add_argument(
        "--decode_vmfb_path",
        default=None,
        help=f"Path to the {DECODE_SEQ_LEN}-token decode VMFB "
        f"(default: decode with the {SEQ_LEN}-token step).",
    )
    parser.add_argument(
        "--prefill_vmfb_path",
        default=None,
        help=f"Path to the {PREFILL_SEQ_LEN}-token prefill VMFB "
        f"(default: prefill with the {SEQ_LEN}-token step).",
    )
  return parser
