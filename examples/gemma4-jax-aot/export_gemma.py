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
"""Exporter for Gemma 4 E2B to StableHLO MLIR."""

import argparse
import json
import os
import time
from gemma4_util import get_abstract_inputs, make_step_fn
import jax


def export_gemma(
    output_path: str = "gemma4_e2b.mlir",
    cache_length: int = 128,
):
  """Lowers and exports Gemma 4 E2B to StableHLO MLIR and saves kept indices."""
  jax.config.update("jax_use_shardy_partitioner", False)
  print("Deriving model parameter shapes...")
  model, abstract_params, token, pos, cache, mask = get_abstract_inputs(
      cache_length=cache_length)
  num_leaves = len(jax.tree_util.tree_leaves(abstract_params))
  print(f"Abstract shapes derived ({num_leaves} leaf tensors).")

  step_fn = make_step_fn(model)

  print("Lowering to StableHLO...")
  t0 = time.time()
  lowered = step_fn.lower(abstract_params, token, pos, cache, mask)
  ir_str = str(lowered.compiler_ir(dialect="stablehlo"))
  # JAX drops arguments that the lowered computation does not actually use, so
  # the compiled function takes a subset of the arguments passed to lower().
  # Persist that subset here so chat.py can select the matching arguments at
  # call time. This reads a private JAX attribute and may need updating on a
  # JAX upgrade; there is no public API exposing kept_var_idx.
  kept_indices = sorted(lowered._lowering.compile_args["kept_var_idx"])
  print(f"Lowered to StableHLO in {time.time() - t0:.2f}s.")

  os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
  with open(output_path, "w") as f:
    f.write(ir_str)
  print(f"MLIR exported to {output_path}")

  kept_indices_path = os.path.splitext(output_path)[0] + "_kept_indices.json"
  with open(kept_indices_path, "w") as f:
    json.dump(kept_indices, f)
  print(f"Kept indices ({len(kept_indices)}) saved to {kept_indices_path}")


def main():
  parser = argparse.ArgumentParser(
      description="Export Gemma 4 E2B to StableHLO MLIR")
  parser.add_argument(
      "--cache_length",
      type=int,
      default=128,
      help="KV cache length (default: 128)",
  )
  parser.add_argument(
      "--output_path",
      type=str,
      default="./gemma4_e2b.mlir",
      help="Output StableHLO MLIR file path (default: ./gemma4_e2b.mlir)",
  )
  parsed_args = parser.parse_args()
  export_gemma(
      output_path=parsed_args.output_path,
      cache_length=parsed_args.cache_length,
  )


if __name__ == "__main__":
  main()
