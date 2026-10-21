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
"""Common utilities and model helpers for Gemma 4 E2B AOT."""

from __future__ import annotations

import json
import os
import typing
from gemma import gm
import gemma.gm.nn.gemma4._layers as _layers
import gemma_common as common
import jax
import jax.numpy as jnp
import numpy as np

# Projections read pre-transposed K x N weights (see `transpose_params`).
_layers.Einsum = common.make_fast_einsum(_layers.Einsum)

# ---------------------------------------------------------------------------
# Tokens and Prompt Formatting Helpers
# ---------------------------------------------------------------------------

STOP_TOKENS = (
    0,
    gm.text.Gemma4Tokenizer.special_tokens.EOS,
    gm.text.Gemma4Tokenizer.special_tokens.END_OF_TURN,
)


def format_prompt(
    history: str,
    user_input: str,
    full_history: bool = True,
) -> str:
  """Formats user input and history with Gemma 4 turn tokens (<|turn>)."""
  user_input = user_input.replace("<start_of_turn>",
                                  "<|turn>").replace("<end_of_turn>", "<turn|>")
  history = history.replace("<start_of_turn>",
                            "<|turn>").replace("<end_of_turn>", "<turn|>")
  if full_history:
    return f"{history}<|turn>user\n{user_input}<turn|>\n<|turn>model\n"
  return f"{history} {user_input}".strip()


# ---------------------------------------------------------------------------
# Chat Loop
# ---------------------------------------------------------------------------


def chat_loop(
    run_turn_fn: typing.Callable[[list[int], int], list[int]],
    full_history: bool = True,
    max_new_tokens: int = 128,
):
  """Interactive multi-turn chat loop for Gemma 4."""
  common.chat_loop(run_turn_fn, gm.text.Gemma4Tokenizer(), format_prompt,
                   "<turn|>\n", STOP_TOKENS, full_history, max_new_tokens)


# ---------------------------------------------------------------------------
# Model Step Function and Export Helpers
# ---------------------------------------------------------------------------


def get_abstract_inputs(cache_length: int = common.CACHE_LEN):
  """Derives abstract parameters and inputs for Gemma 4 E2B text model."""
  model = gm.nn.Gemma4_E2B(text_only=True)
  cache = model.init_cache(
      batch_size=1,
      dtype=jnp.bfloat16,
      cache_length=cache_length,
  )
  dummy_token = jnp.zeros((1, common.SEQ_LEN), dtype=jnp.int32)
  pos = jnp.zeros((1, common.SEQ_LEN), dtype=jnp.int32)
  # The mask crosses the module boundary as int8 and is cast to bool inside.
  mask = jnp.zeros((1, common.SEQ_LEN, cache_length), dtype=jnp.int8)
  rng = jax.random.PRNGKey(0)
  abstract_params = jax.eval_shape(
      model.init,
      rng,
      dummy_token,
      positions=pos,
      cache=cache,
      attention_mask=mask.astype(jnp.bool_),
  )
  return model, abstract_params, dummy_token, pos, cache, mask


def make_step_fn(model):
  """Creates JIT-compiled step function for Gemma 4 E2B."""

  @jax.jit
  def step_fn(p, token, pos, cache, mask):
    out = model.apply(
        p,
        token,
        positions=pos,
        cache=cache,
        attention_mask=mask.astype(jnp.bool_),
        return_last_only=True,
    )
    next_token = jnp.argmax(out.logits, axis=-1).astype(jnp.int32)
    return next_token, out.cache

  return step_fn


def filter_to_template(tree, template):
  """Filters parameter dictionary to match abstract template structure."""
  if isinstance(template, dict):
    return {
        k: filter_to_template(tree[k], template[k])
        for k in template
        if k in tree
    }
  return tree


# ---------------------------------------------------------------------------
# Parameters Loading and KV Cache
# ---------------------------------------------------------------------------


def load_and_flatten_params(
    use_dummy: bool = False,
    checkpoint_path: str | None = None,
) -> list[np.ndarray]:
  """Loads, transposes and flattens Gemma 4 E2B parameters into numpy arrays."""
  _, abstract_params, _, _, _, _ = get_abstract_inputs()
  if use_dummy:
    print(
        "Generating synthetic dummy weights for testing (bypassing download)..."
    )
    flat_shapes, _ = jax.tree_util.tree_flatten(abstract_params)
    return [np.zeros(leaf.shape, dtype=jnp.bfloat16) for leaf in flat_shapes]

  if not checkpoint_path:
    checkpoint_path = gm.ckpts.CheckpointPath.GEMMA4_E2B_IT
  print(f"Loading params from {checkpoint_path}...")
  params = gm.ckpts.load_params(checkpoint_path, text_only=True)
  print("Params loaded.")

  params_dict = params.get("params", params) if isinstance(params,
                                                           dict) else params
  filtered = common.transpose_params(
      filter_to_template(params_dict, abstract_params["params"]))
  flat_params, _ = jax.tree_util.tree_flatten({"params": filtered})
  return [
      np.asarray(
          leaf.astype(jnp.bfloat16) if leaf.dtype != jnp.bfloat16 else leaf)
      for leaf in flat_params
  ]


def init_caches(cache_length: int = common.CACHE_LEN) -> list[np.ndarray]:
  """Initializes flattened numpy arrays for Gemma 4 E2B KV cache."""
  dummy_model = gm.nn.Gemma4_E2B(text_only=True)
  dict_cache = dummy_model.init_cache(
      batch_size=1,
      dtype=jnp.bfloat16,
      cache_length=cache_length,
  )
  flat_cache, _ = jax.tree_util.tree_flatten(dict_cache)
  return [np.asarray(c) for c in flat_cache]


def load_kept_indices(vmfb_path: str) -> list[int]:
  """Loads or derives kept variable indices after DCE."""
  vmfb_dir = os.path.dirname(os.path.abspath(vmfb_path))
  base_name = os.path.splitext(os.path.basename(vmfb_path))[0].replace(
      "_cpu", "")
  json_name = f"{base_name}_kept_indices.json"
  search_dirs = [
      vmfb_dir,
      ".",
      "examples/gemma4-jax-aot",
      os.path.dirname(__file__),
  ]
  build_working_dir = os.environ.get("BUILD_WORKING_DIRECTORY")
  if build_working_dir:
    search_dirs.extend([
        build_working_dir,
        os.path.join(build_working_dir, "examples", "gemma4-jax-aot"),
    ])

  for d in search_dirs:
    for name in (json_name, "gemma4_e2b_kept_indices.json"):
      path = os.path.join(d, name)
      if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "r") as f:
          return json.load(f)

  print("Deriving kept variable indices from abstract step function...")
  jax.config.update("jax_use_shardy_partitioner", False)
  model, abstract_params, token, pos, cache, mask = get_abstract_inputs()
  exported = jax.export.export(make_step_fn(model))(abstract_params, token, pos,
                                                    cache, mask)
  return list(exported.module_kept_var_idx)
