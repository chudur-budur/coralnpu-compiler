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
"""Exporters for Gemma 3 270M (full and 3-part split) to StableHLO MLIR."""

import os
import sys
import time
from gemma3_util import (
    GemmaKV,
    init_cache_layers,
    init_split_cache_layers,
    load_params,
)
import jax
import jax.numpy as jnp


def _save_mlir(path: str, lowered, desc: str, t0: float):
  with open(path, "w") as f:
    f.write(str(lowered.compiler_ir(dialect="stablehlo")))
  print(f"{desc} exported in {time.time() - t0:.2f}s")


def export_gemma(params=None, output_path: str = "gemma3_270m.mlir"):
  """Lowers and exports the full Gemma 3 270M model to StableHLO MLIR."""
  model = GemmaKV()
  if params is None:
    params = load_params()

  cache_init = init_cache_layers()
  dummy_token = jnp.zeros((1, 1), dtype=jnp.int32)
  pos = jnp.zeros((1, 1), dtype=jnp.int32)
  mask = jnp.zeros((1, 1, 128), dtype=jnp.int8)
  full_params = {"params": params}

  @jax.jit
  def step_fn(p, token, pos, cache, mask):
    return model.apply(p, token, pos, cache, mask, method=model.apply_step)

  print(f"Lowering full model to StableHLO ({output_path})...")
  t0 = time.time()
  lowered = step_fn.lower(full_params, dummy_token, pos, cache_init, mask)
  _save_mlir(output_path, lowered, "Full model MLIR", t0)


def export_gemma_split(params=None, out_dir: str = "./"):
  """Lowers and exports 3-part split Gemma 3 270M to StableHLO MLIR."""
  model = GemmaKV()
  if params is None:
    params = load_params()

  os.makedirs(out_dir, exist_ok=True)
  c1_init, c2_init = init_split_cache_layers()

  dummy_token = jnp.zeros((1, 1), dtype=jnp.int32)
  pos = jnp.zeros((1, 1), dtype=jnp.int32)
  mask = jnp.zeros((1, 1, 128), dtype=jnp.int8)
  dummy_x = jnp.zeros((1, 1, model.config.embed_dim), dtype=jnp.float32)
  full_params = {"params": params}

  @jax.jit
  def p1_dec(p, token, pos, cache, mask):
    return model.apply(p, token, pos, cache, mask, method=model.apply_part1)

  @jax.jit
  def p2_dec(p, x, pos, cache, mask):
    return model.apply(p, x, pos, cache, mask, method=model.apply_part2)

  @jax.jit
  def p3_dec(p, x):
    return model.apply(p, x, method=model.apply_part3)

  print("Lowering Part 1...")
  t0 = time.time()
  l_p1 = p1_dec.lower(full_params, dummy_token, pos, c1_init, mask)
  _save_mlir(os.path.join(out_dir, "gemma3_270m_part1.mlir"), l_p1,
             "Part 1 MLIR", t0)

  print("Lowering Part 2...")
  t0 = time.time()
  l_p2 = p2_dec.lower(full_params, dummy_x, pos, c2_init, mask)
  _save_mlir(os.path.join(out_dir, "gemma3_270m_part2.mlir"), l_p2,
             "Part 2 MLIR", t0)

  print("Lowering Part 3...")
  t0 = time.time()
  l_p3 = p3_dec.lower(full_params, dummy_x)
  _save_mlir(os.path.join(out_dir, "gemma3_270m_part3.mlir"), l_p3,
             "Part 3 MLIR", t0)


def main():
  jax.config.update("jax_use_shardy_partitioner", False)
  func_name = sys.argv[1] if len(sys.argv) > 1 else "export_gemma"
  valid_funcs = (
      "export_gemma",
      "export_full",
      "export_gemma_split",
      "export_split",
      "all",
  )
  if func_name not in valid_funcs:
    raise ValueError(f"Unknown function '{func_name}'. "
                     "Expected 'export_gemma' or 'export_gemma_split'.")

  params = load_params()
  if func_name in ("export_gemma", "export_full", "all"):
    export_gemma(params)
  if func_name in ("export_gemma_split", "export_split", "all"):
    export_gemma_split(params)


if __name__ == "__main__":
  main()
