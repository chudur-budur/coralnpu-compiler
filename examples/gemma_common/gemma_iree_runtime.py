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
"""IREE runtime helpers shared by the Gemma 3 and Gemma 4 AOT examples."""

from __future__ import annotations

import os
import types
import typing
import gemma_common as common
import iree.runtime as ireert
import numpy as np

# A function of a loaded VMFB module.
_VmFunc = typing.Callable[..., typing.Any]


def make_aot_run_turn(
    steps: dict[int, _VmFunc],
    flat_params: list[typing.Any],
    cache: list[typing.Any],
    stop_tokens: typing.Collection[int],
    kept_indices: list[int] | None = None,
) -> typing.Callable[[list[int], int], list[int]]:
  """Returns an AOT `run_turn` whose steps update `cache` in place."""

  def step_fn(chunk: list[int], pos_idx: int, seq_len: int) -> int:
    tok, pos, mask = common.step_inputs(chunk, pos_idx, seq_len)
    # Point each layer's end_index (the 1D leaves) at pos_idx; DeviceArray.shape
    # is a list (`[1]`), ndarray.shape is `(1,)`.
    end_index = np.array([pos_idx], dtype=np.int32)
    cache[:] = [end_index if tuple(c.shape) == (1,) else c for c in cache]
    args = flat_params + [tok, pos] + cache + [mask]
    if kept_indices is not None:
      args = [args[i] for i in kept_indices]
    res = steps[seq_len](*args)
    cache[:] = list(res[1:])
    return int(np.asarray(res[0]).reshape(-1)[0])

  return common.make_run_turn(step_fn, stop_tokens, steps)


def load_vmfb(vm_instance: ireert.VmInstance,
              vmfb_path: str) -> ireert.VmModule:
  """Loads a VMFB module using mmap, falling back to from_flatbuffer."""
  try:
    return ireert.VmModule.mmap(vm_instance, vmfb_path)
  except Exception as e:
    print(f"mmap failed: {e}. Trying from_flatbuffer...")
    with open(vmfb_path, "rb") as f:
      return ireert.VmModule.from_flatbuffer(vm_instance, f.read())


def to_device_arrays(device: ireert.HalDevice,
                     arrays: list[np.ndarray]) -> list[ireert.DeviceArray]:
  """Uploads immutable arrays (weights) once for the life of the process."""
  usage = (int(ireert.BufferUsage.DEFAULT) |
           int(ireert.BufferUsage.SHARING_IMMUTABLE) |
           int(ireert.BufferUsage.SHARING_REPLICATE))
  return [ireert.asdevicearray(device, a, allowed_usage=usage) for a in arrays]


def resolve_vmfb_path(
    vmfb_path: str | None,
    default_name: str = "",
    fallback_name: str | None = None,
    example_dir: str = "",
    script_dir: str = "",
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
        os.path.join(example_dir, name),
        os.path.join(script_dir, name),
    ])
    if build_working_dir:
      candidates.extend([
          os.path.join(build_working_dir, name),
          os.path.join(build_working_dir, example_dir, name),
      ])

  for c in candidates:
    if os.path.exists(c) and os.path.getsize(c) > 0:
      return os.path.abspath(c)

  searched = "\n  ".join(candidates)
  raise FileNotFoundError(
      f"Could not find valid VMFB file '{default_name}'.\n"
      f"Searched locations:\n  {searched}\n"
      "Please run compilation script first or specify VMFB path.")


def _create_context(cpu: bool) -> ireert.SystemContext:
  """Creates a local-sync or multi-device (CPU + CoralNPU) SystemContext."""
  if cpu:
    return ireert.SystemContext(config=ireert.Config("local-sync"))
  instance = ireert.VmInstance()
  cpu_device = ireert.get_device("local-sync")
  npu_device = ireert.get_device("coralnpu")
  hal = ireert.create_hal_module(instance, devices=[cpu_device, npu_device])
  config = types.SimpleNamespace(device=cpu_device,
                                 vm_instance=instance,
                                 default_vm_modules=(hal,))
  return ireert.SystemContext(config=config)


def load_step_funcs(
    stem: str,
    script_file: str,
    cpu: bool = False,
    vmfb_path: str | None = None,
    decode_vmfb_path: str | None = None,
    prefill_vmfb_path: str | None = None,
) -> tuple[dict[int, _VmFunc], ireert.HalDevice, str]:
  """Loads step VMFBs; returns (steps_by_seq_len, device, main_vmfb_path)."""
  script_dir = os.path.dirname(os.path.abspath(script_file))
  example_dir = os.path.join("examples", os.path.basename(script_dir))
  primary = f"{stem}_cpu.vmfb" if cpu else f"{stem}.vmfb"
  fallback = None if cpu else f"{stem}_coralnpu.vmfb"
  main_path = resolve_vmfb_path(vmfb_path, primary, fallback, example_dir,
                                script_dir)
  step_paths = [(common.SEQ_LEN, main_path)]
  for seq_len, p in ((common.DECODE_SEQ_LEN, decode_vmfb_path),
                     (common.PREFILL_SEQ_LEN, prefill_vmfb_path)):
    if p is not None:
      step_paths.append((seq_len, resolve_vmfb_path(p)))
  ctx = _create_context(cpu)
  mode = "local-sync" if cpu else "CPU + CoralNPU"
  steps = {}
  for seq_len, path in step_paths:
    print(f"Loading {path} ({mode})...")
    vm_mod = load_vmfb(ctx.config.vm_instance, path)
    ctx.add_vm_module(vm_mod)
    steps[seq_len] = ctx.modules[vm_mod.name].main
  return steps, ctx.config.device, main_path
