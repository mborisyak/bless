from typing import TypeVar, Iterable

import math
import functools

import jax
import jax.numpy as jnp
import jax.scipy as jsp

__all__ = [
  'logsumexp',
  'log1psumexp',
  'LOG_10',

  'monod',
  'inh_monod',

  'gprod',
  'gsum',
  'tree_sum',
  'invert_axes',
  'bsum',

  'uniform_to_normal',
  'normal_to_uniform',
  'exponential_to_normal',
  'normal_to_exponential',

  'stable_logit'
]

LOG_10 = math.log(10.0)

def softmax(*xs):
  if len(xs) == 0:
    raise ValueError('Need at least one array!')
  elif len(xs) == 1:
    x, =  xs
    return (jnp.ones_like(x), )

  else:
    if len(xs) == 2:
      M = jnp.maximum(*xs)
    else:
      M = jnp.max(jnp.stack(xs, axis=-1), axis=-1)

    exp_xs_ = [jnp.exp(x - M) for x in xs]
    Z = sum(exp_xs_)

    return [exp_x_ / Z for exp_x_ in exp_xs_]

def logsumexp(*xs):
  if len(xs) == 0:
    raise ValueError('Need at least one array!')
  elif len(xs) == 1:
    return xs

  else:
    x1, x2, *rest = xs
    M = jnp.maximum(x1, x2)
    for x in rest:
      M = jnp.maximum(M, x)

    return jnp.log(sum(jnp.exp(x - M) for x in xs)) + M

def log1psumexp(*xs):
  if len(xs) == 0:
    raise ValueError('Need at least one array!')
  elif len(xs) == 1:
    x, = xs
    return jnp.where(x > 0, x - jnp.log1p(jnp.exp(-x)), jnp.log1p(jnp.exp(x)))

  else:
    x1, x2, *rest = xs
    M = jnp.maximum(x1, x2)
    for x in rest:
      M = jnp.maximum(M, x)

    M = jnp.maximum(M, 0.0)

    return jnp.log(sum(jnp.exp(x - M) for x in xs) + jnp.exp(-M)) + M

def clip_max(x, m):
    return m - jax.nn.relu(m - x)

def clip_min(x, m):
  return jax.nn.relu(x - m) + m

@functools.partial(jax.custom_jvp, nondiff_argnums=(2, ))
def monod(x, K, grad_clip=1.0e+2):
  """
  Returns x / (x + K) but defines custom clipped gradients.
  :param x: x
  :param K: K
  :param grad_clip: upper bound for the gradient, for both d monod / dx and d monod / dK
  :return: x / (x + K)
  """
  return x / (x + K)

@monod.defjvp
def monod_jvp(grad_clip, primals, tangents):
    x, K = primals
    x_dot, K_dot = tangents
    f = 1 / (x + K)
    a = jnp.square(f)

    primal_out = x * f
    tangent_out = clip_max(K * a, grad_clip) * x_dot - clip_max(x * a, grad_clip) * K_dot

    return primal_out, tangent_out

def inh_monod(x, K, grad_clip=1.0e+2):
  """
  'Inhibition' monod = monod(K, x).

  :param x: x
  :param K: K
  :param grad_clip: upper bound for the gradient, for both d monod / dx and d monod / dK
  :return: K / (x + K)
  """
  return monod(K, x, grad_clip=grad_clip)

T = TypeVar('T')

def gprod(terms: Iterable[T], initial: T | None=None) -> T:
  """
  Product that does not confuse type checker.
  """
  import functools
  if initial is None:
    return functools.reduce(lambda a, b: a * b, terms)
  else:
    return functools.reduce(lambda a, b: a * b, terms, initial)

def gsum(terms: Iterable[T]) -> T:
  """
  Sum that does not confuse type checker.
  """
  import functools
  return functools.reduce(lambda a, b: a + b, terms)

def tree_sum(tree):
  return gsum(jax.tree.leaves(tree))

def invert_axes(x, axes):
  axes = tuple(x.ndim + axis if axis < 0 else axis for axis in axes)

  if axes is None:
    return tuple(range(len(x.shape)))
  else:
    return tuple(dim for dim in range(len(x.shape)) if dim not in axes)

def bsum(x, keep=None):
  axes = invert_axes(x, keep)

  if len(axes) == 0:
    return x
  else:
    return jnp.sum(x, axis=axes)

INV_SQRT_2 = math.sqrt(0.5)
SQRT_2 = math.sqrt(2.0)
LOG_2 = math.log(2.0)
INV_SQRT3 = math.sqrt(1.0 / 3.0)

def uniform_to_normal(xs):
  """
  A transformation that converts a uniformly distributed variable U[-1, 1] into N(0, 1).
  """
  ### uniform = 1 - jnp.exp(-x / 2)
  ### normal = erfinv(2 * uniform - 1) * sqrt(2)
  if isinstance(xs, (float, int)):
    dtype = float
  else:
    dtype = xs.dtype

  eps = jnp.finfo(dtype).eps
  us = jnp.clip(xs, eps - 1, 1 - eps)

  return jsp.special.erfinv(us) * SQRT_2

def normal_to_uniform(xs):
  """
  A transformation that converts a normally distributed variable N(0, 1) into U[-1, 1].
  """
  us = jsp.special.erf(xs * INV_SQRT_2)
  return us

def exponential_to_normal(xs):
  """
  Exp(1) to N(0, 1)
  """
  ### uniform = 1 - jnp.exp(-x / 2)
  ### normal = erfinv(2 * uniform - 1) * sqrt(2)
  eps = jnp.finfo(xs.dtype).eps

  us = 1 - 2 * jnp.exp(-xs)
  us = jnp.clip(us, -1 + eps, 1 - eps)

  return jsp.special.erfinv(us) * SQRT_2

def normal_to_exponential(xs):
  eps = jnp.finfo(xs.dtype).eps

  us = jsp.special.erf(xs * INV_SQRT_2)
  us = jnp.clip(us, -1 + eps, 1 - eps)

  return LOG_2 - jnp.log1p(-us)

def stable_logit(xs):
  eps = jnp.finfo(xs.dtype).eps
  us = jnp.clip(xs, eps, 1 - eps)
  return jsp.special.logit(us)