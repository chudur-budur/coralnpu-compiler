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
"""Common utilities and models for Gemma 3 examples."""

from __future__ import annotations

import typing
from gemma import gm
import gemma.gm.math._positional_embeddings as _pe
import gemma.gm.nn._layers as _layers
import gemma.gm.nn._modules as _mod
import gemma_common as common
import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Fast RoPE, Einsum & Attention Monkey-Patching for CoralNPU
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
  """Monkey-patches Gemma RoPE, attention mask, and Einsum for CoralNPU."""
  _pe.apply_rope = fast_rope
  _mod.apply_rope = fast_rope
  _mod.K_MASK = -1e4
  _layers.Einsum = common.make_fast_einsum(_layers.Einsum)


# Apply patch upon module import
patch_fast_rope()

# Special tokens indicating termination
STOP_TOKENS = (
    0,
    gm.text.Gemma3Tokenizer.special_tokens.EOS,
    gm.text.Gemma3Tokenizer.special_tokens.END_OF_TURN,
)

# ---------------------------------------------------------------------------
# Gemma 3 Model Definition with KV Cache Inference
# ---------------------------------------------------------------------------


class GemmaKV(gm.nn.Gemma3_270M):
  """Gemma 3 270M wrapper with KV cache inference."""

  def apply_step(
      self,
      tokens: jax.Array,
      positions: jax.Array,
      cache_list: list[typing.Any],
      attention_mask: jax.Array,
  ) -> tuple[jax.Array, list[typing.Any]]:
    """Returns the token after the last non-pad token, and the new caches."""
    x = self.embedder.encode(tokens)
    mask = attention_mask.astype(jnp.bool_)
    new_cache = []
    for i, block in enumerate(self.blocks):
      c, x = block(x, positions, cache_list[i], mask)
      new_cache.append(c)
    x = self.final_norm(x)
    # Decode only the last non-pad token.
    last = jnp.sum(tokens != 0, axis=-1) - 1
    x = x[jnp.arange(x.shape[0]), last]
    logits = self.embedder.decode(x.astype(jnp.bfloat16))
    next_token = jnp.argmax(logits, axis=-1).astype(jnp.int32)
    return next_token, new_cache


# ---------------------------------------------------------------------------
# Parameters Loading
# ---------------------------------------------------------------------------


def get_abstract_params() -> dict[str, typing.Any]:
  """Derives abstract parameter shapes/dtypes without downloading weights."""
  tok = jnp.zeros((1, 1), dtype=jnp.int32)
  return jax.eval_shape(GemmaKV().init, jax.random.PRNGKey(0), tok)["params"]


def load_params(device=None):
  """Loads transposed Gemma 3 270M IT params, optionally placing on device."""
  checkpoint_path = gm.ckpts.CheckpointPath.GEMMA3_270M_IT
  print("Loading params...")
  params = common.transpose_params(gm.ckpts.load_params(checkpoint_path))
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


# ---------------------------------------------------------------------------
# KV Cache Helpers
# ---------------------------------------------------------------------------


def init_cache_raw(cache_length: int = common.CACHE_LEN):
  """Initializes raw Gemma 3 cache dictionary."""
  dummy_model = gm.nn.Gemma3_270M()
  return dummy_model.init_cache(batch_size=1,
                                dtype=jnp.bfloat16,
                                cache_length=cache_length)


def init_cache_layers(cache_length: int = common.CACHE_LEN) -> list[typing.Any]:
  """Initializes raw Gemma 3 cache list."""
  dict_cache = init_cache_raw(cache_length=cache_length)
  return [dict_cache[f"layer_{i}"] for i in range(len(dict_cache))]


def init_caches(cache_length: int = common.CACHE_LEN) -> list[np.ndarray]:
  """Initializes flattened numpy arrays for the KV cache."""
  flat, _ = jax.tree_util.tree_flatten(init_cache_layers(cache_length))
  return [np.asarray(c) for c in flat]


# ---------------------------------------------------------------------------
# Prompt Formatting and Chat Loop
# ---------------------------------------------------------------------------


def format_prompt(history: str,
                  user_input: str,
                  full_history: bool = True) -> str:
  """Formats user input and conversation history into a Gemma prompt."""
  if full_history:
    return (f"{history}<start_of_turn>user\n{user_input}<end_of_turn>\n"
            "<start_of_turn>model\n")
  return f"{history} {user_input}".strip()


def chat_loop(
    run_turn_fn: typing.Callable[[list[int], int], list[int]],
    full_history: bool = True,
    max_new_tokens: int = 128,
):
  """Interactive multi-turn chat loop for Gemma 3."""
  common.chat_loop(run_turn_fn, gm.text.Gemma3Tokenizer(), format_prompt,
                   "<end_of_turn>\n", STOP_TOKENS, full_history, max_new_tokens)
