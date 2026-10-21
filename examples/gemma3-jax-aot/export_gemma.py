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
"""Exporter for Gemma 3 270M to StableHLO MLIR."""

import argparse
from gemma3_util import GemmaKV, get_abstract_params, init_cache_layers
from gemma_common import export_step_mlirs
import jax


def export_gemma(output_path: str = "gemma3_270m.mlir"):
  """Exports the Gemma 3 270M 16-, 32- and 1-token steps to StableHLO MLIR."""
  model = GemmaKV()

  @jax.jit
  def step_fn(p, token, pos, cache, mask):
    return model.apply(p, token, pos, cache, mask, method=model.apply_step)

  export_step_mlirs(step_fn, {"params": get_abstract_params()},
                    init_cache_layers(), output_path)


def main():
  argparse.ArgumentParser(
      description="Export Gemma 3 270M to StableHLO MLIR").parse_args()
  export_gemma()


if __name__ == "__main__":
  main()
