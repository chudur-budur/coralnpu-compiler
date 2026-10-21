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
from gemma4_util import get_abstract_inputs, make_step_fn
from gemma_common import CACHE_LEN, SEQ_LEN, export_step_mlirs
import jax


def export_gemma(
    output_path: str = "gemma4_e2b.mlir",
    cache_length: int = CACHE_LEN,
):
  """Exports Gemma 4 E2B steps to StableHLO MLIR and saves kept indices."""
  print("Deriving model parameter shapes...")
  model, abstract_params, _, _, cache, _ = get_abstract_inputs(
      cache_length=cache_length)
  num_leaves = len(jax.tree_util.tree_leaves(abstract_params))
  print(f"Abstract shapes derived ({num_leaves} leaf tensors).")

  kept_indices = export_step_mlirs(make_step_fn(model), abstract_params, cache,
                                   output_path, cache_length)

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
      default=CACHE_LEN,
      help=f"KV cache length (default: {CACHE_LEN})",
  )
  parser.add_argument(
      "--output_path",
      type=str,
      default="./gemma4_e2b.mlir",
      help=(f"Path of the {SEQ_LEN}-token step MLIR; the prefill/decode MLIRs "
            "and kept indices are written next to it (default: "
            "./gemma4_e2b.mlir)"),
  )
  parsed_args = parser.parse_args()
  export_gemma(
      output_path=parsed_args.output_path,
      cache_length=parsed_args.cache_length,
  )


if __name__ == "__main__":
  main()
