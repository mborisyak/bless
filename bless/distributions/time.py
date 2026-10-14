from typing import Self

import jax
import jax.numpy as jnp

from .conditional import ConditionalDistribution, CondDistr
from .. import utils

__all__ = [
  'grid',
  'grid_uniform',
]

class grid(ConditionalDistribution):
  """
  Returns a regular grid.
  """
  __slots__ = ('n', 'origin', 'horizon')

  @classmethod
  def full_parse(cls: type[Self], n: int, horizon: float, origin: float = 0.0) -> tuple[Self, jax.Array | None]:
    dt = (horizon - origin) / n
    g = jnp.arange(1, n + 1) * dt - dt / 2 + origin
    return cls(n, origin=origin, horizon=horizon), g

  @classmethod
  def parse(cls, n: int, horizon: float, origin: float = 0.0) -> 'grid':
    return cls(n, horizon=horizon, origin=origin)

  def encode(self, x, condition):
    *shape, _ = x.shape
    return jnp.zeros(shape=(*shape, 0), dtype=x.dtype)

  def decode(self, x_encoded, condition):
    *shape, _ = x_encoded.shape
    broadcast = (*(None for _ in shape), slice(None, None, None),)
    return jnp.broadcast_to(condition[broadcast], (*shape, self.n,))

  def __init__(self, n: int, origin: float, horizon: float):
    self.origin = origin
    self.horizon = horizon
    self.n = n

    super().__init__(shape=(n, ))

  def sample(self, key, condition, *, shape: tuple[int, ...]=(), dtype=float):
    broadcast = (*(None for _ in shape), slice(None, None, None),)
    condition = jnp.asarray(condition, dtype=dtype)
    return jnp.broadcast_to(condition[broadcast], (*shape, self.n,))

  def scale(self, x, condition):
    return (x - self.origin) / (self.horizon - self.origin)

  def descale(self, x_scaled, condition):
    return x_scaled * (self.horizon - self.origin) + self.origin

class grid_uniform(ConditionalDistribution):
  """
  A regular grid perturbed by uniform noise:
      t_k = i * dt + u_i, i = 1, 2, ..., n
      u_i ~ U[-delta, delta];
      dt = T / (n + 1)
  """

  __slots__ = ('delta', )

  @classmethod
  def full_parse(cls, delta: float, n: int, horizon: float, origin: float=0.0):
    assert 0 <= delta <= 1, 'invalid bounds'
    dt = (horizon - origin) / n
    condition = jnp.arange(1, n + 1, dtype=float) * dt - dt / 2 + origin
    return cls(delta, origin, horizon), condition

  @classmethod
  def parse(cls, delta: float) -> 'grid_uniform':
    raise ValueError('grid-uniform requires a grid condition; use it as a timestamps distribution, not as timestamps-noise.')

  def __init__(self, delta: float, origin: float, horizon: float):
    self.delta = delta
    self.origin = origin
    self.horizon = horizon

    super().__init__()

  def encode(self, x, condition):
    w = self.delta * (self.horizon - self.origin) / condition.shape[-1] / 2
    eps = (x - condition) / w
    return utils.math.uniform_to_normal(eps)

  def decode(self, x_encoded, condition):
    w = self.delta * (self.horizon - self.origin) / condition.shape[-1] / 2
    delta = utils.math.normal_to_uniform(x_encoded) * w
    return condition + delta

  def sample(self, key, condition, *, shape: tuple[int, ...]=(), dtype=float):
    br = (*(None for _ in shape), *(slice(None, None, None) for _ in condition.shape))
    eps = jax.random.uniform(key, shape=(*shape, *condition.shape, *self.shape), minval=-1, maxval=1, dtype=dtype)
    *_, n = condition.shape
    dt = (self.horizon - self.origin) / n
    return condition[br] + self.delta * eps * dt / 2

  def scale(self, x, condition):
    return (x - self.origin) / (self.horizon - self.origin)

  def descale(self, x_scaled, condition):
    return x_scaled * (self.horizon - self.origin) + self.origin
