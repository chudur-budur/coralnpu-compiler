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
"""Common utilities, models, and runtime helpers for Gemma 3 examples."""

from __future__ import annotations

import os
import sys
import time
import typing
from gemma import gm
import gemma.gm.math._positional_embeddings as _pe
import gemma.gm.nn._modules as _mod
import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Fast RoPE & Attention Monkey-Patching for CoralNPU
# ---------------------------------------------------------------------------


def fast_rope(
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
  _pe.apply_rope = fast_rope
  _mod.apply_rope = fast_rope
  _mod.K_MASK = -1e4


# Apply patch upon module import
patch_fast_rope()

# Special tokens indicating termination
STOP_TOKENS = (
    0,
    gm.text.Gemma3Tokenizer.special_tokens.EOS,
    gm.text.Gemma3Tokenizer.special_tokens.END_OF_TURN,
)

# ---------------------------------------------------------------------------
# Gemma 3 Model Definition with Full & Split KV Cache Inference
# ---------------------------------------------------------------------------


class GemmaKV(gm.nn.Gemma3_270M):
  """Gemma 3 270M wrapper with unified and 3-part split KV cache inference."""

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
      c, x = self.blocks[i](x, positions, cache_list[i], mask)
      new_cache.append(c)
    x = self.final_norm(x)
    logits = self.embedder.decode(x.astype(jnp.bfloat16))
    next_token = jnp.argmax(logits, axis=-1).astype(jnp.int32)
    return next_token, new_cache

  def apply_part1(
      self,
      tokens: jax.Array,
      positions: jax.Array,
      cache_list: list[typing.Any],
      attention_mask: jax.Array,
  ) -> tuple[jax.Array, list[typing.Any]]:
    x = self.embedder.encode(tokens)
    mask = attention_mask.astype(jnp.bool_)
    new_cache = []
    for i in range(9):
      c, x = self.blocks[i](x, positions, cache_list[i], mask)
      new_cache.append(c)
    return x.astype(jnp.float32), new_cache

  def apply_part2(
      self,
      x_in: jax.Array,
      positions: jax.Array,
      cache_list: list[typing.Any],
      attention_mask: jax.Array,
  ) -> tuple[jax.Array, list[typing.Any]]:
    x = x_in.astype(jnp.bfloat16)
    mask = attention_mask.astype(jnp.bool_)
    new_cache = []
    for i in range(9, 18):
      c, x = self.blocks[i](x, positions, cache_list[i - 9], mask)
      new_cache.append(c)
    x = self.final_norm(x)
    return x.astype(jnp.float32), new_cache

  def apply_part3(self, x_in: jax.Array) -> jax.Array:
    logits = self.embedder.decode(x_in.astype(jnp.bfloat16))
    return jnp.argmax(logits, axis=-1).astype(jnp.int32)


SplitGemmaKV = GemmaKV  # Backward compatibility alias

# ---------------------------------------------------------------------------
# Parameters Loading and Partitioning
# ---------------------------------------------------------------------------


def load_params(device=None):
  """Loads Gemma 3 270M IT parameters, optionally placing on device."""
  checkpoint_path = gm.ckpts.CheckpointPath.GEMMA3_270M_IT
  print("Loading params...")
  params = gm.ckpts.load_params(checkpoint_path)
  if device is not None:
    params = jax.tree.map(lambda x: jax.device_put(x, device), params)
  print("Params loaded.")
  return params


def load_and_flatten_params(params=None) -> list[np.ndarray]:
  """Loads and flattens parameters into numpy arrays for full model."""
  if params is None:
    params = load_params()
  print("Flattening params...")
  flat_params, _ = jax.tree_util.tree_flatten(params)
  return [np.asarray(p) for p in flat_params]


def load_and_flatten_split_params(
    params=None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
  """Loads and extracts parameters for 3-part split execution."""
  if params is None:
    params = load_params()
  print("Extracting Part 1, Part 2, Part 3 params...")
  flat_params, _ = jax.tree_util.tree_flatten(params)

  p1_np = [np.asarray(flat_params[0])] + [
      np.asarray(p)
      for l in range(9)
      for p in jax.tree_util.tree_leaves(params[f"layer_{l}"])
  ]

  # Layer 9 appears after layer 17 due to alphabetical key sorting in JAX.
  p2_layers = [10, 11, 12, 13, 14, 15, 16, 17, 9]
  p2_np = [np.asarray(flat_params[1])] + [
      np.asarray(p)
      for l in p2_layers
      for p in jax.tree_util.tree_leaves(params[f"layer_{l}"])
  ]

  p3_np = [np.asarray(flat_params[0])]
  return p1_np, p2_np, p3_np


# ---------------------------------------------------------------------------
# KV Cache and Mask Helpers
# ---------------------------------------------------------------------------


def init_cache_raw(cache_length: int = 128):
  """Initializes raw 18-layer Gemma 3 cache dictionary."""
  dummy_model = gm.nn.Gemma3_270M()
  return dummy_model.init_cache(batch_size=1,
                                dtype=jnp.bfloat16,
                                cache_length=cache_length)


def init_cache_layers(cache_length: int = 128) -> list[typing.Any]:
  """Initializes raw 18-layer Gemma 3 cache list."""
  dict_cache = init_cache_raw(cache_length=cache_length)
  return [dict_cache[f"layer_{i}"] for i in range(18)]


def init_split_cache_layers(
    cache_length: int = 128,) -> tuple[list[typing.Any], list[typing.Any]]:
  """Initializes split (Part 1: 0..8, Part 2: 9..17) cache lists."""
  layers = init_cache_layers(cache_length=cache_length)
  return layers[:9], layers[9:]


def _flatten_cache(cache_layers) -> list[np.ndarray]:
  flat, _ = jax.tree_util.tree_flatten(cache_layers)
  return [np.asarray(c) for c in flat]


def init_caches(cache_length: int = 128) -> list[np.ndarray]:
  """Initializes flattened numpy arrays for full 18-layer KV cache."""
  return _flatten_cache(init_cache_layers(cache_length=cache_length))


def init_split_caches(
    cache_length: int = 128,) -> tuple[list[np.ndarray], list[np.ndarray]]:
  """Initializes flattened numpy arrays for split Parts 1 & 2 KV cache."""
  c1, c2 = init_split_cache_layers(cache_length=cache_length)
  return _flatten_cache(c1), _flatten_cache(c2)


def make_mask(length: int, max_length: int = 128) -> np.ndarray:
  """Creates a causal attention mask [1, 1, max_length] of int8."""
  mask = np.zeros((1, 1, max_length), dtype=np.int8)
  mask[0, 0, :length] = 1
  return mask


# ---------------------------------------------------------------------------
# Generation, Prompt Formatting, and Chat Loop
# ---------------------------------------------------------------------------


def generate(
    step_fn: typing.Callable[[int, int], int],
    tokens_list: list[int],
    max_new_tokens: int = 128,
) -> list[int]:
  """Autoregressively generates tokens using step_fn(token_id, pos_index)."""
  prompt_len = len(tokens_list)
  print(f"[Debug] Prefilling {prompt_len} tokens...")

  # 1. Prefill
  for step in range(prompt_len):
    step_tok = step_fn(tokens_list[step], step)

  # 2. Decode
  generated = [step_tok]
  if step_tok in STOP_TOKENS:
    return generated

  curr_len = prompt_len
  curr_tok = step_tok
  while curr_len < 128 and len(generated) < max_new_tokens:
    curr_tok = step_fn(curr_tok, curr_len)
    generated.append(curr_tok)
    if curr_tok in STOP_TOKENS:
      break
    curr_len += 1

  return generated


def format_prompt(history: str,
                  user_input: str,
                  full_history: bool = True) -> str:
  """Formats user input and conversation history into a Gemma prompt."""
  if full_history:
    return (f"{history}<start_of_turn>user\n{user_input}<end_of_turn>\n"
            "<start_of_turn>model\n")
  return f"{history} {user_input}".strip()


def decode_response(tokenizer, output_tokens: list[int]) -> str:
  """Decodes token IDs into text up to EOS or END_OF_TURN special tokens."""
  response_tokens = []
  for t in output_tokens:
    if t in STOP_TOKENS:
      break
    response_tokens.append(int(t))
  return tokenizer.decode(response_tokens)


def chat_loop(
    run_turn_fn: typing.Callable[[list[int], int], list[int]],
    full_history: bool = True,
    max_new_tokens: int = 128,
):
  """Shared interactive multi-turn chat loop."""
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
    tokens_list = tokenizer.encode(prompt, add_bos=True)
    print(f"[Debug] Prompt length: {len(tokens_list)} tokens")

    if len(tokens_list) > 128:
      print("ERROR: History exceeds limit.")
      break

    t0 = time.time()
    output_tokens = run_turn_fn(tokens_list, max_new_tokens)
    t1 = time.time()

    model_response = decode_response(tokenizer, output_tokens)
    print(f"Model: {model_response}")
    print(f"[Inference took {t1 - t0:.2f}s]")

    if full_history:
      history = prompt + model_response + "<end_of_turn>\n"
    else:
      history = f"{prompt} {model_response}".strip()

    turn += 1


# ---------------------------------------------------------------------------
# IREE Runtime and VMFB Helpers
# ---------------------------------------------------------------------------


class MultiDeviceConfig:
  """Config wrapper for multi-device (CPU + CoralNPU) execution."""

  def __init__(self, device, vm_instance, default_vm_modules):
    self.device = device
    self.vm_instance = vm_instance
    self.default_vm_modules = default_vm_modules


def load_vmfb(vm_instance, vmfb_path: str):
  """Loads a VMFB module using mmap, falling back to from_flatbuffer."""
  import iree.runtime as ireert

  try:
    return ireert.VmModule.mmap(vm_instance, vmfb_path)
  except Exception as e:
    print(f"mmap failed: {e}. Trying from_flatbuffer...")
    with open(vmfb_path, "rb") as f:
      return ireert.VmModule.from_flatbuffer(vm_instance, f.read())


def resolve_vmfb_path(
    vmfb_path: str | None,
    default_name: str,
    fallback_name: str | None = None,
) -> str:
  """Resolves path to a VMFB file searching local, examples, and runfiles."""
  if vmfb_path is not None:
    if not os.path.exists(vmfb_path) or os.path.getsize(vmfb_path) == 0:
      raise FileNotFoundError(
          f"Specified VMFB file '{vmfb_path}' does not exist or is empty.")
    return os.path.abspath(vmfb_path)

  names = [default_name]
  if fallback_name:
    names.append(fallback_name)

  candidates = []
  build_working_dir = os.environ.get("BUILD_WORKING_DIRECTORY")
  for name in names:
    candidates.extend([
        os.path.join(".", name),
        os.path.join("examples", "gemma3-jax-aot", name),
        os.path.join(os.path.dirname(__file__), name),
    ])
    if build_working_dir:
      candidates.extend([
          os.path.join(build_working_dir, name),
          os.path.join(build_working_dir, "examples", "gemma3-jax-aot", name),
      ])

  for c in candidates:
    if os.path.exists(c) and os.path.getsize(c) > 0:
      return os.path.abspath(c)

  searched = "\n  ".join(candidates)
  raise FileNotFoundError(
      f"Could not find valid VMFB file '{default_name}'.\n"
      f"Searched locations:\n  {searched}\n"
      "Please run compilation script first or specify VMFB path.")


def create_cpu_context(vmfb_paths: list[str]):
  """Creates an IREE local-sync SystemContext with loaded VMFBs."""
  import iree.runtime as ireert

  config = ireert.Config("local-sync")
  ctx = ireert.SystemContext(config=config)
  for path in vmfb_paths:
    print(f"Loading {path} (local-sync)...")
    ctx.add_vm_module(load_vmfb(config.vm_instance, path))
  return ctx


def create_multidevice_context(vmfb_path: str):
  """Creates a multi-device (CPU + CoralNPU) SystemContext with loaded VMFB."""
  import iree.runtime as ireert

  instance = ireert.VmInstance()
  cpu_device = ireert.get_device("local-sync")
  npu_device = ireert.get_device("coralnpu")
  hal_module = ireert.create_hal_module(instance,
                                        devices=[cpu_device, npu_device])
  config = MultiDeviceConfig(cpu_device, instance, (hal_module,))
  ctx = ireert.SystemContext(config=config)
  print(f"Loading {vmfb_path} (CPU + CoralNPU)...")
  ctx.add_vm_module(load_vmfb(instance, vmfb_path))
  return ctx
