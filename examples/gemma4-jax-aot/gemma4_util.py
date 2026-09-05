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
"""Common utilities, model, and runtime helpers for Gemma 4 E2B AOT."""

from __future__ import annotations

import json
import os
import sys
import time
import typing
from gemma import gm
import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Tokens, Mask, and Prompt Formatting Helpers
# ---------------------------------------------------------------------------

STOP_TOKENS = (
    0,
    gm.text.Gemma4Tokenizer.special_tokens.EOS,
    gm.text.Gemma4Tokenizer.special_tokens.END_OF_TURN,
)


def make_mask(
    length: int,
    max_length: int = 128,
    dtype: np.typing.DTypeLike = bool,
) -> np.ndarray:
  """Creates a causal attention boolean mask [1, 1, max_length]."""
  mask = np.zeros((1, 1, max_length), dtype=dtype)
  mask[0, 0, :length] = True
  return mask


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


def decode_response(tokenizer, output_tokens: list[int]) -> str:
  """Decodes token IDs into text up to EOS or END_OF_TURN special tokens."""
  response_tokens = []
  for t in output_tokens:
    if t in STOP_TOKENS:
      break
    response_tokens.append(int(t))
  return tokenizer.decode(response_tokens)


# ---------------------------------------------------------------------------
# Generation and Chat Loop
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


def chat_loop(
    run_turn_fn: typing.Callable[[list[int], int], list[int]],
    full_history: bool = True,
    max_new_tokens: int = 128,
):
  """Shared interactive multi-turn chat loop for Gemma 4."""
  tokenizer = gm.text.Gemma4Tokenizer()
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
    tokens_list = [int(t) for t in tokenizer.encode(prompt, add_bos=True)]
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
      history = f"{prompt}{model_response}<turn|>\n"
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
        os.path.join("examples", "gemma4-jax-aot", name),
        os.path.join(os.path.dirname(__file__), name),
    ])
    if build_working_dir:
      candidates.extend([
          os.path.join(build_working_dir, name),
          os.path.join(build_working_dir, "examples", "gemma4-jax-aot", name),
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


# ---------------------------------------------------------------------------
# Model Step Function and Export Helpers
# ---------------------------------------------------------------------------


def get_abstract_inputs(cache_length: int = 128):
  """Derives abstract parameters and inputs for Gemma 4 E2B text model."""
  model = gm.nn.Gemma4_E2B(text_only=True)
  cache = model.init_cache(
      batch_size=1,
      dtype=jnp.bfloat16,
      cache_length=cache_length,
  )
  dummy_token = jnp.zeros((1, 1), dtype=jnp.int32)
  pos = jnp.zeros((1, 1), dtype=jnp.int32)
  mask = jnp.zeros((1, 1, cache_length), dtype=jnp.bool_)
  rng = jax.random.PRNGKey(0)
  abstract_params = jax.eval_shape(
      model.init,
      rng,
      dummy_token,
      positions=pos,
      cache=cache,
      attention_mask=mask,
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
        attention_mask=mask,
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
  """Loads and flattens Gemma 4 E2B parameters into numpy arrays."""
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
  filtered = filter_to_template(params_dict, abstract_params["params"])
  flat_params, _ = jax.tree_util.tree_flatten({"params": filtered})
  return [
      np.asarray(
          leaf.astype(jnp.bfloat16) if leaf.dtype != jnp.bfloat16 else leaf)
      for leaf in flat_params
  ]


def init_caches(cache_length: int = 128) -> list[np.ndarray]:
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
  step_fn = make_step_fn(model)
  lowered = step_fn.lower(abstract_params, token, pos, cache, mask)
  return sorted(lowered._lowering.compile_args["kept_var_idx"])
