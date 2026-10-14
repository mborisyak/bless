from typing import Any, TypeVar
import math
import re

import jax

__all__ = [
  'split',
  'extract',
  'optimizer',
  'expand',
  'expand_template',
  'expand_library',
  'models',
  'get_device'
]

NAME = '__name__'
GROUND_TRUTH = 'ground-truth'

def split(config: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
  """
  Python objects (models, optimizers etc) are configured in the following way in YAML:
      <name of the model / method / optimizer>:
        <arguments>

  This function checks that config follows this format and returns the name and the arguments.

  :param config: configuration for an object;
  :param library: a dictionary of named objects;
  :return: name of the object, arguments.
  """
  if len(config) == 1:
    name, = config.keys()
    arguments = config[name]
    return name, arguments
  else:
    raise ValueError(
      f'config entry must contain a dictionary with exactly one field (name of the object), '
      f'got {", ".join(config.keys())}'
    )

T = TypeVar('T')
def extract(config: dict[str, dict[str, Any]], library: dict[str, T]) -> tuple[T, dict[str, Any]]:
  name, arguments = split(config)
  if name not in library:
    raise ValueError(f'{name} does not appear to be a valid object')

  return library[name], arguments


def optimizer(config: dict[str, Any]):
  import optax

  name, arguments = split(config)

  return getattr(optax, name)(**arguments)

def expansion_size(config):
  import math
  values = config['values']

  return math.prod([len(vals) for _, vals in values.items()])

def get_strides(sizes):
  n = len(sizes)
  strides = []
  acc = 1

  for i in range(n):
    acc *= sizes[i]
    strides.append(acc)

  return strides

def get_value(key: str, values):
  tokens = key.split('.')
  value = values

  for token in tokens:
    if token.isdigit():
      index = int(token)
      try:
        value = value[index]
      except Exception as e:
        raise ValueError(f'{value} does not have key {token} (full key: {key})') from e

    else:
      if token in value:
        value = value[token]
      else:
        raise ValueError(f'{value} does not have key {token} (full key: {key})')

  return value


def expand_template(template, values):
  if isinstance(template, dict):
    concrete = {}

    for k, v in template.items():
      if k.startswith('$'):
        k_expanded = get_value(k[1:], values)
      else:
        k_expanded = k

      concrete[k_expanded] = expand_template(template[k], values)

    return concrete

  elif isinstance(template, list):
    return [expand_template(temp, values) for temp in template]

  elif isinstance(template, str):
    if template.startswith('$'):
      return get_value(template[1:], values)
    else:
      return template

  else:
    return template


def expand(template, values, name, postfix=None):
  normalized_values = {}
  normalized_names = {}
  for arg, vs in values.items():
    if isinstance(vs, (tuple, list)):
      normalized_values[arg] = vs
      normalized_names[arg] = [str(i) for i, _ in enumerate(vs)]

    elif isinstance(vs, dict):
      normalized_names[arg], normalized_values[arg] = zip(*[
        (k, v) for k, v in vs.items()
      ])

    else:
      raise ValueError('values must be either list of dict')

  total = math.prod([len(vs) for _, vs in normalized_values.items()])
  expanded = {}

  for i in range(total):
    i_ = i
    instance_config = {}
    value_names = []

    for arg, vs in normalized_values.items():
      j = i_ % len(vs)
      i_ = i_ // len(vs)
      instance_config[arg] = vs[j]
      value_names.append(str(normalized_names[arg][j]))

    if postfix is not None:
      instance_postfix = postfix.format(**{
        arg: name for arg, name in zip(normalized_values.keys(), value_names)
      })
    else:
      instance_postfix = '-'.join(value_names)

    if len(instance_postfix) > 0:
      instance_name = f'{name}-{instance_postfix}'
    else:
      instance_name = name

    ### may occur if there is nothing to expand
    if instance_name in expanded:
      raise ValueError(f'name collision {instance_name}, please, check template _postfix')

    expanded[instance_name] = expand_template(template, instance_config)

  return expanded

TEMPLATE = '__template__'
VALUES = '__values__'
POSTFIX = '__postfix__'

def expand_library(library):
  """
  Enumerates the concrete configurations of a library.

  An entry holding `__template__` and `__values__` becomes one entry per combination of the values,
  named `<entry>-<value names joined by '-'>` unless `__postfix__` gives a format for the names.
  Every other entry, including `__name__`, is passed through unchanged.

  :param library: library as loaded from YAML;
  :return: the library with every template replaced by its instances.
  """
  expanded = {}

  for key, config in library.items():
    if isinstance(config, dict) and TEMPLATE in config:
      instances = expand(config[TEMPLATE], config[VALUES], key, config.get(POSTFIX, None))

      for name, instance in instances.items():
        if name in expanded:
          raise ValueError(f'name collision during expansion: {name}')
        expanded[name] = instance

    else:
      expanded[key] = config

  return expanded

def models(library, patterns=None):
  """
  The fitted entries of an expanded library: every entry but `__name__` and `ground-truth`, restricted to the names
  matching one of the regular expressions `patterns` when given.
  """
  names = [k for k in library if k != NAME and k != GROUND_TRUTH]
  if patterns is None:
    return names
  return [n for n in names if any(re.fullmatch(p, n) for p in patterns)]

def get_job_name(values):
  import hashlib, base64, json

  sha = hashlib.sha256()
  sha.update(json.dumps(values).encode('utf-8'))
  return base64.b32encode(sha.digest()).decode('utf-8').replace('=', '_')

def get_device(device_spec: jax.Device | str | None=None):
  if device_spec is None:
    device, *_ = jax.devices()
    return device

  if isinstance(device_spec, jax.Device):
    return device_spec

  if ':' in device_spec:
    device_name, device_index = device_spec.split(':')
    device_index = int(device_index)
  else:
    device_name = device_spec
    device_index = 0

  return jax.devices(device_name)[device_index]