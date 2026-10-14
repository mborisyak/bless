import os

import jax
import numpy as np

__all__ = [
  'save', 'load', 'load_metadata',
  'latest_version', 'latest_free_version'
]

def save(path, controls=None, parameters=None, timestamps=None, observations=None, **metadata):
  flat_dict = {}
  if controls is not None:
    for f in controls._fields:
      flat_dict[f'controls.{f}'] = np.array(getattr(controls, f))

  if parameters is not None:
    for f in parameters._fields:
      flat_dict[f'parameters.{f}'] = np.array(getattr(parameters, f))

  if timestamps is not None:
    for f in timestamps._fields:
      flat_dict[f'timestamps.{f}'] = np.array(getattr(timestamps, f))

  if observations is not None:
    for f in observations._fields:
      flat_dict[f'observations.{f}'] = np.array(getattr(observations, f))

  for f in metadata:
    flat_dict[f'metadata.{f}'] = np.array(metadata[f])

  return np.savez(path, **flat_dict, allow_pickle=False)

def extract(flat_dict, prefix):
  n = len(prefix)
  return {
    name[n:]: flat_dict[name]
    for name in flat_dict
    if name.startswith(prefix)
  }

def load(path, model, device: jax.Device | None=None):
  flat_dict = np.load(path)

  controls = extract(flat_dict, 'controls.')
  controls = None if len(controls) == 0 else model.ControlParameters(**controls)

  parameters = extract(flat_dict, 'parameters.')
  parameters = None if len(parameters) == 0 else model.Parameters(**parameters)
  timestamps = extract(flat_dict, 'timestamps.')
  timestamps = None if len(timestamps) == 0 else model.Observables(**timestamps)

  observations = extract(flat_dict, 'observations.')
  observations = None if len(observations) == 0 else model.Observables(**observations)

  loaded = controls, parameters, timestamps, observations

  return jax.tree.map(
    lambda x: jax.device_put(x, device=device),
    loaded
  )

def load_metadata(path):
  flat_dict = np.load(path)

  return extract(flat_dict, 'metadata.')

def max_version_index(output):
  if not os.path.isdir(output):
    raise ValueError(f'{output} is not a directory')

  max_index = -1
  for item in os.listdir(output):
    if item.startswith('v'):
      try:
        index = int(item[1:])
      except ValueError:
        continue

      if index > max_index:
        max_index = index

  return max_index

def latest_version(output, version: int | None=None):
  if version is None:
    max_index = max_version_index(output)
    return os.path.join(output, f'v{max_index}')

  else:
    path = os.path.join(output, f'v{version}')
    if not os.path.exists(path):
      raise ValueError(f"{path} doesn't exist")

    elif not os.path.isdir(path):
      raise ValueError(f"{path} is not a directory")

    else:
      return path

def latest_free_version(output, version: int | None=None):
  if version is None:
    max_index = max_version_index(output)
    return os.path.join(output, f'v{max_index + 1}')

  else:
    path = os.path.join(output, f'v{version}')
    if os.path.exists(path):
      raise ValueError(f"{path} exists")
    else:
      return path