from typing import Sequence

import math
import jax

__all__ = [
  'concrete',
  'batch_shape',
  'align',
  'broadcast_left',
  'broadcast_right',
  'flat_shape',
  'broadcast'
]


def concrete(*shape):
  """
  Removes None from the `shape`
  :param shape: shape with potential Nones
  :return: shape without Nones
  """
  return tuple(s for s in shape if s is not None)

def batch_shape(shape, feature_shape):
  fdim = len(feature_shape)
  bdim = len(shape) - fdim
  return shape[:bdim]

def contract(shape: Sequence[int | None]) -> Sequence[int | None]:
  i = 0
  for i in range(len(shape) - 1, -1, -1):
    if shape[i] is not None:
      break

  return shape[:i + 1]

def _align(shape1: Sequence[int | None], shape2: Sequence[int | None]):
  n1, n2 = len(shape1), len(shape2)
  dim = max(n1, n2)

  _shape1 = (*shape1, *(None for _ in range(dim - n1)))
  _shape2 = (*shape2, *(None for _ in range(dim - n2)))

  common = []

  for s1, s2 in zip(_shape1, _shape2):
    if s1 is None:
      common.append(s2)
    elif s2 is None:
      common.append(s1)
    elif s1 == s2:
      common.append(s1)
    else:
      raise ValueError(f'shapes {shape1} and {shape2} cannot be aligned.')

  return tuple(common)

def align(shape: Sequence[int | None], condition):
  if isinstance(condition, (float, int)):
    return shape
  elif condition is None:
    return shape
  else:
    return _align(shape, condition.shape)

def broadcast_left(array, target):
  if isinstance(target, jax.Array):
    target_ndim = target.ndim
  elif isinstance(target, tuple):
    target_ndim = len(target)
  elif isinstance(target, int):
    target_ndim = target
  else:
    raise ValueError()

  broadcast = (
    *(slice(None, None, None) for _ in range(array.ndim)),
    *(None for _ in range(target_ndim - array.ndim)),
  )

  return array[broadcast]

def broadcast_right(array, target):
  if isinstance(target, jax.Array):
    target_ndim = target.ndim
  elif isinstance(target, tuple):
    target_ndim = len(target)
  elif isinstance(target, int):
    target_ndim = target
  else:
    raise ValueError()

  broadcast = (
    *(None for _ in range(target_ndim - array.ndim)),
    *(slice(None, None, None) for _ in range(array.ndim)),
  )

  return array[broadcast]

def flat_shape(shapes, axes=()):
  is_shape = lambda shape: isinstance(shape, (tuple, )) and all(isinstance(s, int) for s in shape)
  shapes_flat = jax.tree.leaves(shapes, is_leaf=is_shape)

  b_shape, = set([tuple(shape[i] for i in axes) for shape in shapes_flat])
  concatenated = sum(
    math.prod(shape[i] for i in range(len(shape)) if i not in axes)
    for shape in shapes_flat
  )

  return *b_shape, concatenated

class broadcast(object):
  def __init__(self, tree):
    self.tree = tree

  def __getitem__(self, item):
    return jax.tree.map(lambda x: x[item], self.tree)