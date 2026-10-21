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

import concurrent.futures
import time
import typing
import gemma3_util as g3
import gemma_common as common
import jax
import numpy as np


def compile_steps(model: g3.GemmaKV, params: typing.Any, cache: typing.Any,
                  device: jax.Device | None) -> dict[int, typing.Any]:
  """JIT-compiles the step function concurrently for each STEP_LENS length."""

  @jax.jit
  def step_fn(p, token, pos, cache_in, mask):
    return model.apply(p, token, pos, cache_in, mask, method=model.apply_step)

  lowered = {}
  for seq_len in common.STEP_LENS:
    tok, pos, mask = jax.device_put(common.step_inputs([], 0, seq_len), device)
    lowered[seq_len] = step_fn.lower(params, tok, pos, cache, mask)
  print(f"Compiling the {', '.join(map(str, common.STEP_LENS))}-token steps...")
  t0 = time.time()
  with concurrent.futures.ThreadPoolExecutor(len(lowered)) as ex:
    compiled = dict(
        zip(lowered, ex.map(lambda low: low.compile(), lowered.values())))
  print(f"Compiled in {time.time() - t0:.2f}s")
  return compiled


def chat(
    cpu: bool = False,
    max_new_tokens: int = 128,
    full_history: bool = True,
):
  """Runs interactive chat for the Gemma 3 270M model."""
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

  model = g3.GemmaKV()
  target_device = cpu_dev if cpu else None
  # Upload a host copy once to avoid cross-device replication.
  params = jax.tree.map(np.asarray, g3.load_params())
  params = jax.device_put({"params": params}, target_device)
  # Fresh buffer per leaf: in-place updates must not share one zero buffer.
  cache = jax.tree.map(
      lambda c: jax.device_put(np.zeros(c.shape, dtype=c.dtype), target_device),
      jax.eval_shape(g3.init_cache_layers),
  )
  steps = compile_steps(model, params, cache, target_device)

  def step_fn(chunk: list[int], pos_idx: int, seq_len: int) -> int:
    nonlocal cache
    tok, pos, mask = jax.device_put(common.step_inputs(chunk, pos_idx, seq_len),
                                    target_device)
    # Write this chunk at pos_idx; padded steps advance end_index too far.
    for layer in cache:
      layer["end_index"] = jax.device_put(np.array([pos_idx], dtype=np.int32),
                                          target_device)
    next_tok, cache = steps[seq_len](params, tok, pos, cache, mask)
    return int(np.asarray(next_tok)[0])

  run_turn = common.make_run_turn(step_fn, g3.STOP_TOKENS, steps)
  g3.chat_loop(run_turn,
               full_history=full_history,
               max_new_tokens=max_new_tokens)


def main():
  parser = common.make_chat_parser("Gemma 3 270M PJRT Chat (CoralNPU + CPU)")
  chat(**vars(parser.parse_args()))


if __name__ == "__main__":
  main()
