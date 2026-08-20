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
"""Interactive chat client for Gemma 3 270M via the CoralNPU PJRT plugin."""

import argparse
import sys
import time
import typing
from gemma import gm
import gemma.gm.math._positional_embeddings as _pe
import gemma.gm.nn._modules as _mod
import jax
import jax.numpy as jnp
import numpy as np


# Real-time RoPE computation for CoralNPU.
# Replaces complex64 arithmetic with direct real float32 vector math
# to bypass unsupported complex types on CoralNPU.
def _fast_rope(
    inputs: jax.Array,
    positions: jax.Array,
    *,
    base_frequency: float = 1000000.0,
    scale_factor: float = 1.0,
    rope_proportion: float = 1.0,
) -> jax.Array:
  """Applies RoPE using real float32 arithmetic instead of complex numbers.

  Args:
    inputs: Array of shape [B, L, N, H].
    positions: Array of shape [B, L].
    base_frequency: Base frequency used to compute rotations.
    scale_factor: The scale factor used for positional interpolation.
    rope_proportion: The proportion of the head dimension to apply RoPE to (1.0
      for Gemma 3-270M). Unrotated dimensions are padded with inf timescale so
      that sin=0, cos=1.
  """
  head_dim = inputs.shape[-1]
  rope_angles = int(rope_proportion * head_dim // 2)
  nope_angles = head_dim // 2 - rope_angles

  fraction = 2.0 * jnp.arange(0, rope_angles, dtype=jnp.float32) / head_dim
  timescale = base_frequency**fraction
  if nope_angles > 0:
    timescale = jnp.pad(
        timescale,
        (0, nope_angles),
        mode="constant",
        constant_values=(0, jnp.inf),
    )

  pos = positions.astype(jnp.float32)
  sinusoid_inp = (pos[:, :, None] / timescale[None, None, :]) / scale_factor

  cos = jnp.expand_dims(jnp.cos(sinusoid_inp), -2)
  sin = jnp.expand_dims(jnp.sin(sinusoid_inp), -2)

  inputs_f32 = inputs.astype(jnp.float32)
  first_half, second_half = jnp.split(inputs_f32, 2, axis=-1)
  first_part = first_half * cos - second_half * sin
  second_part = second_half * cos + first_half * sin
  out = jnp.concatenate([first_part, second_part], axis=-1)
  return out.astype(inputs.dtype)


def patch_fast_rope():
  """Monkey-patches Gemma RoPE and attention mask for CoralNPU."""
  _pe.apply_rope = _fast_rope
  _mod.apply_rope = _fast_rope
  _mod.K_MASK = -1e4


# Apply patch upon module import
patch_fast_rope()

# Special tokens indicating termination
STOP_TOKENS = (
    0,
    gm.text.Gemma3Tokenizer.special_tokens.EOS,
    gm.text.Gemma3Tokenizer.special_tokens.END_OF_TURN,
)


class GemmaKV(gm.nn.Gemma3_270M):

  def apply_step(
      self,
      tokens: jax.Array,
      positions: jax.Array,
      cache_list: list[typing.Any],
      attention_mask: jax.Array,
  ) -> tuple[jax.Array, list[typing.Any]]:
    x = self.embedder.encode(tokens)
    mask = attention_mask.astype(jnp.bool_)
    new_cache = []
    for i in range(18):
      c, x = self.blocks[i](
          x,
          positions,
          cache_list[i],
          mask,
      )
      new_cache.append(c)
    x = self.final_norm(x)
    logits = self.embedder.decode(x.astype(jnp.bfloat16))
    next_token = jnp.argmax(logits, axis=-1).astype(jnp.int32)
    return next_token, new_cache


def load_params(device=None):
  """Loads Gemma 3 270M IT parameters, optionally placing on device."""
  checkpoint_path = gm.ckpts.CheckpointPath.GEMMA3_270M_IT

  print("Loading params...")
  params = gm.ckpts.load_params(checkpoint_path)
  if device is not None:
    params = jax.tree.map(lambda x: jax.device_put(x, device), params)
  print("Params loaded.")
  return params


def format_prompt(history: str,
                  user_input: str,
                  full_history: bool = True) -> str:
  """Formats user input and conversation history into a Gemma prompt."""
  if full_history:
    return (f"{history}<start_of_turn>user\n{user_input}<end_of_turn>\n"
            "<start_of_turn>model\n")
  return f"{history} {user_input}".strip()


def make_step_fn(model, params):
  """Creates a JIT-compiled single-token step function for the model."""
  full_params = {"params": params}

  @jax.jit
  def step_fn(curr_tok, pos, cache_in, mask):
    return model.apply(
        full_params,
        curr_tok,
        pos,
        cache_in,
        mask,
        method=model.apply_step,
    )

  return step_fn


def _device_array(val, dtype, device=None) -> jax.Array:
  """Creates a JAX array, optionally placing it on device."""
  arr = jnp.array(val, dtype=dtype)
  return jax.device_put(arr, device) if device is not None else arr


def _make_mask(length: int, max_length: int = 128, device=None) -> jax.Array:
  """Creates a causal attention mask [1, 1, max_length] of int8."""
  mask = np.zeros((1, 1, max_length), dtype=np.int8)
  mask[0, 0, :length] = 1
  arr = jnp.array(mask)
  return jax.device_put(arr, device) if device is not None else arr


def run_inference(model,
                  step_fn,
                  tokens_list: list[int],
                  device=None,
                  max_new_tokens: int = 1) -> list[int]:
  """Runs inference for Gemma 3 270M with KV cache.

  Note: In CoralNPU simulation, each autoregressive decode step is slow.
  max_new_tokens defaults to 1 for CoralNPU so that automated testing completes
  in reasonable time, while allowing multi-token generation on CPU or when
  configured.
  """
  dict_cache = model.init_cache(batch_size=1,
                                dtype=jnp.bfloat16,
                                cache_length=128)
  cache = [dict_cache[f"layer_{i}"] for i in range(18)]
  if device is not None:
    cache = jax.tree.map(lambda c: jax.device_put(c, device), cache)

  prompt_len = len(tokens_list)
  if prompt_len == 0:
    raise ValueError("tokens_list must contain at least one token.")
  print(f"[Debug] Prefilling {prompt_len} tokens...")

  # 1. Prefill
  for step in range(prompt_len):
    curr_tok = _device_array([[tokens_list[step]]], jnp.int32, device)
    pos = _device_array([[step]], jnp.int32, device)
    mask = _make_mask(step + 1, device=device)
    pred_tok, cache = step_fn(curr_tok, pos, cache, mask)

  first_tok = int(pred_tok[0, 0])
  generated = [first_tok]

  if first_tok in STOP_TOKENS:
    return generated

  # 2. Decode
  curr_len = prompt_len
  curr_tok = _device_array([[first_tok]], jnp.int32, device)

  while curr_len < 128 and len(generated) < max_new_tokens:
    pos = _device_array([[curr_len]], jnp.int32, device)
    mask = _make_mask(curr_len + 1, device=device)

    pred_tok, cache = step_fn(curr_tok, pos, cache, mask)
    next_tok = int(pred_tok[0, 0])
    generated.append(next_tok)

    if next_tok in STOP_TOKENS:
      break

    curr_tok = _device_array([[next_tok]], jnp.int32, device)
    curr_len += 1

  return generated


def decode_response(tokenizer, output_tokens: list[int]) -> str:
  """Decodes token IDs into text up to EOS or END_OF_TURN special tokens."""
  response_tokens = []
  for t in output_tokens:
    if t in STOP_TOKENS:
      break
    response_tokens.append(int(t))
  return tokenizer.decode(response_tokens)


def chat(
    cpu: bool = False,
    max_new_tokens: int | None = None,
    full_history: bool | None = None,
):
  """Runs interactive chat for the Gemma 3 270M model."""
  if max_new_tokens is None:
    max_new_tokens = 128 if cpu else 1
  if full_history is None:
    full_history = True if cpu else False

  print(f"jax version={jax.__version__}")
  jax.config.update("jax_platforms", "coralnpu_plugin")
  jax.config.update("jax_use_shardy_partitioner", False)

  devices = jax.devices("coralnpu_plugin")
  print(f"Devices found: {devices}")
  cpu_dev = devices[0]
  print(f"Using CPU device (device 0): {cpu_dev}")
  if not cpu:
    npu_dev = devices[1]
    print(f"Using CoralNPU device (device 1): {npu_dev}")

  model = GemmaKV()
  target_device = cpu_dev if cpu else None
  params = load_params(device=target_device)
  step_fn = make_step_fn(model, params)

  tokenizer = gm.text.Gemma3Tokenizer()

  print("\nInteractive Chat started. Type 'exit' or 'quit' to end.")

  history = ""
  turn = 1
  while True:
    print(f"\n--- Turn {turn} ---\n")

    try:
      user_input = input("User: ").strip()
      if not sys.stdin.isatty():
        print(user_input)
    except (EOFError, KeyboardInterrupt):
      user_input = "exit"
      if not sys.stdin.isatty():
        print(user_input)

    if not user_input or user_input.lower() in ("exit", "quit"):
      print("Done")
      break

    prompt = format_prompt(history, user_input, full_history=full_history)

    # Tokenize
    tokens_list = tokenizer.encode(prompt, add_bos=True)
    print(f"[Debug] Prompt length: {len(tokens_list)} tokens")

    if len(tokens_list) > 128:
      print("ERROR: History exceeds limit.")
      break

    print("Model is thinking...")
    t0 = time.time()
    output_tokens = run_inference(
        model,
        step_fn,
        tokens_list,
        device=target_device,
        max_new_tokens=max_new_tokens,
    )
    elapsed = time.time() - t0
    print(f"[Debug] Generated in {elapsed:.2f}s")

    response_text = decode_response(tokenizer, output_tokens)
    print(f"Model: {response_text}")

    if full_history:
      history = prompt + response_text + "<end_of_turn>\n"
    else:
      history = f"{prompt} {response_text}".strip()

    turn += 1


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
            "128 for CPU). Note: token generation is slow in the CoralNPU "
            "simulator."),
  )
  common.add_argument(
      "--full-history",
      action=argparse.BooleanOptionalAction,
      default=None,
      help=(
          "Whether to retain full conversation history formatted with Gemma "
          "chat turn tokens (<start_of_turn>user...<end_of_turn>). Default is "
          "True for CPU, False for CoralNPU to keep prompt length minimal for "
          "fast simulation."),
  )

  # This example only has the unsplit model, so the subcommand is optional and
  # "chat" is implied. It is still accepted, to mirror the AOT example's CLI.
  parser = argparse.ArgumentParser(
      description="Gemma 3 270M PJRT Chat (CoralNPU + CPU)",
      parents=[common],
  )
  parser.set_defaults(func=chat)
  subparsers = parser.add_subparsers()

  full_parser = subparsers.add_parser(
      "chat",
      parents=[common],
      help="Chat with the Gemma 3 270M model.",
  )
  full_parser.set_defaults(func=chat)

  kwargs = vars(parser.parse_args())
  func = kwargs.pop("func")
  func(**kwargs)


if __name__ == "__main__":
  main()
