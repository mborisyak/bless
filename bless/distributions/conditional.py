from typing import TypeVar, TypeAlias, Self
from collections.abc import Sequence
import math

import jax
import jax.numpy as jnp

from .common import Distribution
from .. import utils

__all__ = [
  'ConditionalDistribution',

  'const',
  'uniform',
  'log_uniform',
  'normal',
  'glorot_normal',
  'exponential',
  'log_normal',
  'logit_normal',
  'biased_log_normal',
  'normal_positive_definite',
  'normal_negative_definite',
  'normal_positive_definite_bound',
  'normal_negative_definite_bound',
  'normal_skew_symmetric',
  'identity'
]

CondDistr = TypeVar('CondDistr', bound='ConditionalDistribution')
Shape: TypeAlias = Sequence[int]

class ConditionalDistribution(Distribution):
  __slots__ = ('shape', )

  @classmethod
  def full_parse(cls: type[Self], *args, **kwargs) -> tuple[Self, jax.Array | None]:
    """
    Parses a fully specified distribution, including condition.
    Returns likelihood-like distribution and the condition.
    """
    raise NotImplementedError()

  @classmethod
  def parse(cls: type[Self], *args, **kwargs) -> Self:
    """
    Parses a likelihood-like distribution, excluding condition.
    """
    raise NotImplementedError()

  def __init__(self, shape: Shape=()):
    self.shape = shape

  def sample(self, key, condition, *, shape: Shape=(), dtype=float):
    raise NotImplementedError()

  def get_shape(self, condition) -> Shape:
    key = jax.random.PRNGKey(0)
    shaped = jax.eval_shape(self.sample, key, condition, shape=())
    return jax.tree.map(jnp.shape, shaped)

  def get_shape_encoded(self, condition, shapes=None) -> Shape:
    key = jax.random.PRNGKey(0)
    shaped = jax.eval_shape(
      lambda k, c, s: self.encode(self.sample(k, c, shape=s), c),
      key, condition, () if shapes is None else shapes
    )
    return jax.tree.map(jnp.shape, shaped)

  def scale(self, x, condition):
    """
    Brings the variable to approx. [-1, 1] scale.
    :param x: the value;
    :param condition: distribution's condition;
    :return: scaled variable.
    """
    return self.encode(x, condition)

  def descale(self, x_scaled, condition):
    """
    Inverse of scale.
    :param x_scaled: scaled variable
    :param condition: condition
    :return: the variable in the original scale.
    """
    return self.decode(x_scaled, condition)

  def encode(self, x, condition):
    """
    Transformation that makes the random variable distributed normally with zero mean and unit variance.

    :param x: samples from the original distribution;
    :param condition: condition of the distribution.
    :return: encoded values.
    """
    raise NotImplementedError()

  def decode(self, x_encoded, condition):
    """
    Transformations a normally distributed (zero mean, unit variance) variable into the target distribution.

    :param x_encoded: samples from N(0, 1);
    :param condition: condition of the original distribution.
    :return: decoded values.
    """
    raise NotImplementedError()

  def encoded_log_p(self, x_encoded, axes: tuple[int, ...]=()):
    """
    Computes log P_y(encode(X)). Please, note that this function does not return the original log P_x!
    Instead, it computes:
    log P_y(y) = log P_x(x) + log | d decode(y) / d y |,
    where y = encode(x), i.e., probability density in the transformed space.

    By convention, this function almost always returns log probability density of the standard normal distribution
    as most of the priors encode variables in such a way that the resulting distribution is normal.

    :param x_encoded: encoded value;
    :param axes: batch axes;
    :return: log P_y(y).
    """
    ### by default, normal log_p as, by convention, encode transforms variable into N(0, 1)
    Z = -0.5 * jnp.log(2 * jnp.pi)

    if axes is None or len(axes) == 0:
      n = 1
    else:
      n = math.prod(x_encoded.shape[axis] for axis in utils.math.invert_axes(x_encoded, axes))

    return n * Z + self.encoded_un_log_p(x_encoded, axes=axes)

  def encoded_un_log_p(self, x_encoded, axes: tuple[int, ...]=()):
    """
    Unnormalized version of `encoded_log_p` potentially stripped of constants.

    Please, note a difference between this method and that of `Likelihood`:
    while two methods represent practically the same notion, namely, `log_p` simplified for optimization,
    in the likelihood it is assumed that `condition` is a variable (e.g., a gradient is taken w.r.t. it),
    while `Prior` treats `x` as the optimization target.
    """
    return -0.5 * utils.math.bsum(jnp.square(x_encoded), keep=axes)

  def log_p(self, x, condition, axes: tuple[int, ...]=()):
    """
    Computes likelihood `P(x | condition)` where `condition` is typically mean of the distribution.

    :param x: point at which density is computed;
    :param condition: condition of the distribution, typically, mean of the distribution;
    :param axes: batch axes, i.e., axes excluded from summation;
    :return: log P(x | condition).
    """
    raise NotImplementedError()

  def un_log_p(self, x, condition, axes: tuple[int, ...]=()):
    """
    Biased versions of log_p potentially stripped of constants.
    The function is optimized for minimal computations.

    :param x: point at which density is computed;
    :param condition: condition of the distribution, typically, mean of the distribution;
    :param axes: batch axes, i.e., axes excluded from summation;
    :return: log P(x | condition) + const.
    """
    raise NotImplementedError()

  def loss(self, x, condition, axes: tuple[int, ...]=()):
    """
    Biased versions of negative log p such that minimum of the function w.r.t. the condition is 0:
        min_c loss(x, c) = 0

    :param x: point at which density is computed;
    :param condition: condition of the distribution, typically, mean of the distribution;
    :param axes: batch axes, i.e., axes excluded from summation;
    :return: -log P(x | condition) + const.
    """
    raise NotImplementedError()

  def __call__(self, x_encoded, axes: tuple[int, ...]=()):
    return self.encoded_un_log_p(x_encoded, axes)

  def sample(self, key, condition, *, shape: tuple[int, ...]=(), dtype=float):
    eps = jax.random.normal(key, shape=(*utils.shape.align(shape, condition), *self.shape), dtype=dtype)
    return self.decode(eps, condition)

  def __str__(self):
    parameters = ', '.join(
      f'{k}={getattr(self, k)}'
      for k in self.__slots__
    )
    return f'{self.__class__.__name__}({parameters})'

  def __repr__(self):
    return str(self)

class const(ConditionalDistribution):
  """
  A pseudo-distribution representing a single value. Useful for fixing value of a parameter.

  Encoding always returns zeros, decoding --- the specified constant.
  Probability functions return 0.
  """
  __slots__ = ('shape', )

  @classmethod
  def full_parse(cls, value: float, shape: Shape=()):
    return cls(shape=shape), value

  @classmethod
  def parse(cls, shape: Shape=()):
    return cls(shape=shape)

  def __init__(self, shape: Shape=()):
    super().__init__(shape=shape)

  def encode(self, x, condition):
    return jnp.zeros(shape=(*x.shape, 0), dtype=x.dtype)

  def decode(self, x_encoded, condition):
    *shape, zero_dim = x_encoded.shape
    assert zero_dim == 0
    return jnp.zeros(shape=shape) + condition

  def scale(self, x, condition):
    return jnp.zeros(shape=(*x.shape, 0), dtype=x.dtype)

  def descale(self, x_scaled, condition):
    *shape, zero_dim = x_scaled.shape
    assert zero_dim == 0
    return jnp.zeros(shape=shape) + condition

  def log_p(self, x, condition, axes: tuple[int, ...] = ()):
    return jnp.zeros(
      shape=tuple(s for i, s in enumerate(x.shape) if i in axes),
      dtype=x.dtype
    )

  def un_log_p(self, x, condition, axes: tuple[int, ...] = ()):
    return self.log_p(x, condition, axes=axes)

  def loss(self, x, condition, axes: tuple[int, ...]=()):
    return self.log_p(x, condition, axes=axes)

  def encoded_log_p(self, x_encoded, axes: tuple[int, ...]=()):
    return jnp.zeros(
      shape=tuple(s for i, s in enumerate(x_encoded.shape) if i in axes),
      dtype=x_encoded.dtype
    )

  def encoded_un_log_p(self, x_encoded, axes: tuple[int, ...]=()):
    return self.encoded_log_p(x_encoded, axes)

  def sample(self, key, condition, *, shape: tuple[int, ...] = (), dtype=float):
    return jnp.ones(shape=(*utils.shape.align(shape, condition), *self.shape), dtype=dtype) * condition

class uniform(ConditionalDistribution):
  """
  Represents U[condition - delta, condition + delta].
  """
  __slots__ = ('delta', 'shape')

  @classmethod
  def full_parse(cls, lower: float, upper: float, shape: Shape=()):
    delta = 0.5 * (upper - lower)
    center = 0.5 * (upper + lower)

    return cls(delta, shape=shape), center

  @classmethod
  def parse(cls, delta: float, shape: Shape=()):
    return cls(delta, shape=shape)

  def __init__(self, delta, shape: Shape=()):
    self.delta = delta
    super().__init__(shape=shape)

  def encode(self, x, condition):
    us = (x - condition) / self.delta
    return utils.math.uniform_to_normal(us)

  def decode(self, x_encoded, condition):
    us = utils.math.normal_to_uniform(x_encoded)
    return us * self.delta + condition

  def log_p(self, x, condition, axes: tuple[int, ...]=()):
    return utils.math.bsum(
      jnp.where(
        jnp.abs(x - condition) < self.delta,
        jnp.zeros_like(x) - jnp.log(2 * self.delta),
        -jnp.inf * jnp.ones_like(x),
      ),
      keep=axes
    )

  def un_log_p(self, x, condition, axes: tuple[int, ...]=()):
    return self.log_p(x, condition, axes=axes)

  def loss(self, x, condition, axes: tuple[int, ...]=()):
    return utils.math.bsum(
      jnp.where(
        jnp.abs(x - condition) < self.delta,
        jnp.zeros_like(x) / (2 * self.delta),
        -jnp.inf * jnp.ones_like(x),
      ),
      keep=axes
    )

  def sample(self, key, condition, *, shape: tuple[int, ...]=(), dtype=float):
    eps = jax.random.uniform(
      key, minval=-1, maxval=1,
      shape=(*utils.shape.align(shape, condition), *self.shape), dtype=dtype
    )
    return self.delta * eps + condition

  def scale(self, x, condition):
    return (x - condition) / self.delta

  def descale(self, x_scaled, condition):
    return x_scaled * self.delta + condition

class logit_normal(ConditionalDistribution):
  """
  Represents U[condition - delta, condition + delta].
  """
  __slots__ = ('delta', 'shape', 'mu', 'sigma')

  @classmethod
  def full_parse(cls, lower: float, upper: float, mu: float=0.0, sigma: float=1.0, shape: Shape=()):
    assert sigma > 0.0, 'sigma must be positive'

    delta = 0.5 * (upper - lower)
    center = 0.5 * (upper + lower)

    return cls(delta, mu, sigma, shape=shape), center

  @classmethod
  def parse(cls, delta: float, mu: float=0.0, sigma: float=1.0, shape: Shape=()):
    return cls(delta, mu=mu, sigma=sigma, shape=shape)

  def __init__(self, delta, mu, sigma, shape: Shape=()):
    self.delta = delta
    self.mu = mu
    self.sigma = sigma
    super().__init__(shape=shape)

  def encode(self, x, condition):
    lower, upper = condition - self.delta, condition + self.delta

    us = 0.5 * (x - lower) / self.delta
    unbounded = utils.math.stable_logit(us)
    return (unbounded - self.mu) / self.sigma

  def decode(self, x_encoded, condition):
    import jax.scipy as jsp

    logit = x_encoded * self.sigma + self.mu
    us = jsp.special.expit(logit)

    lower = condition - self.delta
    return 2 * us * self.delta + lower

  def log_p(self, x, condition, axes: tuple[int, ...]=()):
    raise NotImplementedError()

  def un_log_p(self, x, condition, axes: tuple[int, ...]=()):
    return self.log_p(x, condition, axes=axes)

  def loss(self, x, condition, axes: tuple[int, ...]=()):
    raise NotImplementedError()

  def sample(self, key, condition, *, shape: tuple[int, ...]=(), dtype=float):
    import jax.scipy as jsp

    eps = jax.random.normal(key, shape=(*utils.shape.align(shape, condition), *self.shape), dtype=dtype)
    logit = eps * self.sigma + self.mu
    u = jsp.special.expit(logit)

    lower = condition - self.delta
    return 2 * u * self.delta + lower

  def scale(self, x, condition):
    return (x - condition) / self.delta

  def descale(self, x_scaled, condition):
    return x_scaled * self.delta + condition

class log_uniform(ConditionalDistribution):
  """
  log X ~ U[condition - delta, condition + delta].
  """
  __slots__ = ('delta_log', 'shape')

  @classmethod
  def full_parse(cls, lower: float, upper: float, shape: Shape=()):
    assert lower < upper, 'the lower bound must be lower than the upper one'

    lower_log = jnp.log(lower)
    upper_log = jnp.log(upper)

    delta_log = 0.5 * (upper_log - lower_log)
    mean_log = 0.5 * (upper_log + lower_log)

    return cls(delta_log, shape=shape), mean_log

  @classmethod
  def parse(cls, delta_log: float, shape: Shape=()):
    return cls(delta_log, shape)

  def __init__(self, delta_log, shape: Shape=()):
    self.delta_log = delta_log
    super().__init__(shape=shape)

  def encode(self, x, condition):
    us = (jnp.log(x) - condition) / self.delta_log
    return utils.math.uniform_to_normal(us)

  def decode(self, x_encoded, condition):
    us = utils.math.normal_to_uniform(x_encoded)
    return jnp.exp(us * self.delta_log + condition)

  def _inside(self, x, condition):
    log_x = jnp.log(x)
    ### the rounding of log x, condition and delta_log at the bounds
    slack = jnp.finfo(log_x.dtype).eps * (jnp.abs(log_x) + jnp.abs(condition) + self.delta_log)
    return jnp.abs(log_x - condition) <= self.delta_log + slack

  def log_p(self, x, condition, axes: tuple[int, ...]=()):
    """
    log of the density `1 / (x (log upper - log lower))` on `[lower, upper]`, `-inf` outside.
    """
    log_x = jnp.log(x)
    return utils.math.bsum(
      jnp.where(
        self._inside(x, condition),
        -log_x - jnp.log(2.0 * self.delta_log),
        -jnp.inf * jnp.ones_like(x),
      ),
      keep=axes
    )

  def un_log_p(self, x, condition, axes: tuple[int, ...]=()):
    """`log_p` without its normalisation constant: `-log x` inside the support."""
    return utils.math.bsum(
      jnp.where(self._inside(x, condition), -jnp.log(x), -jnp.inf * jnp.ones_like(x)),
      keep=axes
    )

  def loss(self, x, condition, axes: tuple[int, ...]=()):
    """
    `-log p` shifted to a minimum of 0 over the support: `log(x / lower)` inside, `+inf` outside.
    """
    lower_log = condition - self.delta_log
    return utils.math.bsum(
      jnp.where(self._inside(x, condition), jnp.log(x) - lower_log, jnp.inf * jnp.ones_like(x)),
      keep=axes
    )

  def scale(self, x, condition):
    lower = jnp.exp(condition - self.delta_log)
    upper = jnp.exp(condition + self.delta_log)
    center, delta = (upper + lower) / 2,(upper - lower) / 2
    return (x - center) / delta

  def descale(self, x_scaled, condition):
    lower = jnp.exp(condition - self.delta_log)
    upper = jnp.exp(condition + self.delta_log)
    center, delta = (upper + lower) / 2, (upper - lower) / 2
    return x_scaled * delta + center

  def sample(self, key, condition, *, shape: tuple[int, ...]=(), dtype=float):
    eps = jax.random.uniform(
      key, minval=-1, maxval=1,
      shape=(*utils.shape.align(shape, condition), *self.shape), dtype=dtype
    )

    return jnp.exp(self.delta_log * eps + condition)

class normal(ConditionalDistribution):
  __slots__ = ('sigma', 'shape')

  @classmethod
  def full_parse(cls, mean: float, sigma: float, shape: Shape=()):
    return cls(sigma, shape=shape), mean

  @classmethod
  def parse(cls, sigma: float, shape: Shape=()):
    return cls(sigma, shape=shape)

  def __init__(self, sigma: float, shape: Shape=()):
    assert sigma > 0, 'sigma of a normal distribution must be positive'

    self.sigma = sigma
    super().__init__(shape=shape)

  def encode(self, x, condition):
    return (x - condition) / self.sigma

  def decode(self, x_encoded, condition):
    return x_encoded * self.sigma + condition

  def log_p(self, x, condition, axes: tuple[int, ...] = ()):
    Z = -0.5 * jnp.log(2 * jnp.pi) - jnp.log(self.sigma)
    square_error = jnp.square(x - condition)

    return utils.math.bsum(
      Z - 0.5 * square_error / (self.sigma * self.sigma),
      keep=axes
    )

  def un_log_p(self, x, condition, axes: tuple[int, ...]=()):
    square_error = utils.math.bsum(jnp.square(x - condition), keep=axes)
    return -0.5 * square_error / (self.sigma * self.sigma)

  def loss(self, x, condition, axes: tuple[int, ...]=()):
    square_error = utils.math.bsum(jnp.square(x - condition), keep=axes)
    return 0.5 * square_error / (self.sigma * self.sigma)

  def sample(self, key, condition, *, shape: tuple[int, ...] = (), dtype=float):
    eps = jax.random.normal(key, shape=(*utils.shape.align(shape, condition), *self.shape), dtype=dtype)
    return self.sigma * eps + condition

  def scale(self, x, condition):
    return (x - condition) / self.sigma

  def descale(self, x_scaled, condition):
    return (x_scaled - condition) / self.sigma

class glorot_normal(ConditionalDistribution):
  __slots__ = ('axis', 'shape')

  @classmethod
  def full_parse(cls, gain: float=1.0, axis: int=-1, shape: Shape = ()):
    return cls(gain, axis=axis, shape=shape), None

  @classmethod
  def parse(cls, gain: float=1.0, axis: int=-1, shape: Shape = ()):
    return cls(gain, shape=shape, axis=axis)

  def get_sigma(self, n):
    return self.gain / math.sqrt(n)

  def __init__(self, gain: float=1.0, axis: int=-1, shape: Shape = ()):
    """
    Axis corresponds to inputs.
    """
    assert gain >= 0, 'gain of a glorot-normal distribution must be positive'

    self.gain = gain
    self.axis = axis

    super().__init__(shape=shape)

  def encode(self, x, condition):
    return x

  def decode(self, x_encoded, condition):
    return x_encoded

  def log_p(self, x, condition, axes: tuple[int, ...] = ()):
    sigma = self.get_sigma(x.shape[self.axis])

    Z = -0.5 * jnp.log(2 * jnp.pi) - jnp.log(sigma)
    square_error = jnp.square(x)

    return utils.math.bsum(
      Z - 0.5 * square_error / (sigma * sigma),
      keep=axes
    )

  def un_log_p(self, x, condition, axes: tuple[int, ...] = ()):
    sigma = self.get_sigma(x.shape[self.axis])
    square_error = utils.math.bsum(jnp.square(x - condition), keep=axes)
    return -0.5 * square_error / (sigma * sigma)

  def loss(self, x, condition, axes: tuple[int, ...] = ()):
    sigma = self.get_sigma(x.shape[self.axis])
    square_error = utils.math.bsum(jnp.square(x - condition), keep=axes)
    return 0.5 * square_error / (sigma * sigma)

  def sample(self, key, condition, *, shape: tuple[int, ...] = (), dtype=float):
    sigma = self.get_sigma(shape[self.axis])

    eps = jax.random.normal(key, shape=(*shape, *self.shape), dtype=dtype)
    return sigma * eps

  def scale(self, x, condition):
    shape = x.shape
    sigma = self.get_sigma(shape[self.axis])
    return x / sigma

  def descale(self, x_scaled, condition):
    shape = x_scaled.shape
    sigma = self.get_sigma(shape[self.axis])
    return x_scaled * sigma

class exponential(ConditionalDistribution):
  __slots__ = ('shape', )

  @classmethod
  def full_parse(cls, beta: float, shape: Shape=()):
    return cls(shape=shape), beta

  @classmethod
  def parse(cls, shape: Shape=()):
    return cls(shape)

  def encode(self, x, condition):
    return utils.math.exponential_to_normal(x / condition)

  def decode(self, x_encoded, condition):
    return utils.math.normal_to_exponential(x_encoded) * condition

  def log_p(self, x, condition, axes: tuple[int, ...]=()):
    return utils.math.bsum(
      -x / condition - jnp.log(condition),
      keep=axes
    )

  def loss(self, x, condition, axes: tuple[int, ...]=()):
    ### min log c + c / x = log x + 1
    return utils.math.bsum(
      -x / condition - jnp.log(condition) - jnp.log(x) - 1,
      keep=axes
    )

  def un_log_p(self, x, condition, axes: tuple[int, ...]=()):
    return self.log_p(x, condition, axes=axes)

  def sample(self, key, condition, *, shape: tuple[int, ...]=(), dtype=float):
    return condition * jax.random.exponential(
      key, shape=(*utils.shape.align(shape, condition), *self.shape), dtype=dtype
    )

  def scale(self, x, condition):
    return x / condition

  def descale(self, x_scaled, condition):
    return x_scaled * condition

class log_normal(ConditionalDistribution):
  __slots__ = ('shape', 'sigma')

  @classmethod
  def parse(cls, sigma: float, shape: Shape=()):
    return cls(sigma, shape=shape)

  @classmethod
  def full_parse(cls, median: float | None=None, sigma: float | None=None, mean: float | None=None, mode: float | None=None, shape: Shape=()):
    import math

    assert len([x for x in (median, mode, mean) if x is not None]) == 1, \
      'Only one of the following arguments must be specified: median, mean, mode'

    assert all(x is None or x > 0 for x in (median, mode, mean)), \
      'all provided values must be positive'

    assert (sigma is not None) and (sigma > 0), 'sigma must be positive'

    if median is not None:
      mean = median * math.exp(0.5 * sigma * sigma)

    elif mean is not None:
      pass

    elif mode is not None:
      mean = mode * math.exp(-sigma * sigma)

    else:
      raise ValueError('At least one of the following arguments must be specified: median, mean, mode')

    return log_normal(sigma, shape=shape), mean

  def __init__(self, sigma: float, shape: Shape=()):
    assert sigma > 0, 'sigma must be positive'
    self.sigma = sigma
    super().__init__(shape=shape)

  def encode(self, x, condition):
    mu = jnp.log(condition) - 0.5 * self.sigma * self.sigma
    return (jnp.log(x) - mu) / self.sigma

  def decode(self, x_encoded, condition):
    return condition * jnp.exp(self.sigma * (x_encoded - 0.5 * self.sigma))

  def log_p(self, x, condition, axes: tuple[int, ...]=()):
    log_x = jnp.log(x)
    Z = -0.5 * jnp.log(2 * jnp.pi) - jnp.log(self.sigma) - log_x
    mu = jnp.log(condition) - 0.5 * self.sigma * self.sigma
    square_error = jnp.square(log_x - mu) / (self.sigma * self.sigma)

    return utils.math.bsum(Z - 0.5 * square_error, keep=axes)

  def un_log_p(self, x, condition, axes: tuple[int, ...]=()):
    log_x = jnp.log(x)
    ### log X in the normalization term is omitted as it is a constant w.r.t. condition.
    mu = jnp.log(condition) - 0.5 * jnp.square(self.sigma)
    square_error = utils.math.bsum(jnp.square(log_x - mu), keep=axes)
    return -0.5 * square_error / (self.sigma * self.sigma)

  def loss(self, x, condition, axes: tuple[int, ...]=()):
    return self.un_log_p(x, condition, axes=axes)

  def sample(self, key, condition, *, shape: tuple[int, ...] = (), dtype=float):
    eps = jax.random.normal(key, shape=(*utils.shape.align(shape, condition), *self.shape), dtype=dtype)
    return jnp.exp(eps * self.sigma - 0.5 * self.sigma * self.sigma) * condition

  def scale(self, x, condition):
    mean = condition
    std = jnp.sqrt(jnp.exp(self.sigma ** 2) - 1) * mean

    return (x - mean) / std

  def descale(self, x_scaled, condition):
    mean = condition
    std = jnp.sqrt(jnp.exp(self.sigma ** 2) - 1) * mean

    return (x_scaled - mean) / std

class biased_log_normal(ConditionalDistribution):
  __slots__ = ('sigma', 'shape', 'bias')

  @classmethod
  def parse(cls, sigma: float, bias: float=0.0, shape: Shape = ()):
    return cls(sigma, bias=bias, shape=shape)

  @classmethod
  def full_parse(cls, median: float, sigma: float = None, bias: float=0.0, shape: Shape = ()):
    import math
    assert (sigma is not None) and (sigma > 0), 'sigma must be positive'
    mean = median * math.exp(0.5 * sigma * sigma)
    return log_normal(sigma, bias=bias, shape=shape), mean

  def __init__(self, sigma: float, bias: float, shape: Shape = ()):
    assert sigma > 0, 'sigma must be positive'
    self.sigma = sigma
    self.bias = bias
    super().__init__(shape=shape)

  def encode(self, x, condition):
    mu = jnp.log(condition + self.bias) - 0.5 * self.sigma * self.sigma
    return (jnp.log(x) - mu) / self.sigma

  def decode(self, x_encoded, condition):
    return (condition + self.bias) * jnp.exp(self.sigma * (x_encoded - 0.5 * self.sigma))

  def log_p(self, x, condition, axes: tuple[int, ...] = ()):
    log_x = jnp.log(x)
    Z = -0.5 * jnp.log(2 * jnp.pi) - jnp.log(self.sigma) - log_x
    mu = jnp.log(condition + self.bias) - 0.5 * self.sigma * self.sigma
    square_error = jnp.square(log_x - mu) / (self.sigma * self.sigma)

    return utils.math.bsum(Z - 0.5 * square_error, keep=axes)

  def un_log_p(self, x, condition, axes: tuple[int, ...] = ()):
    log_x = jnp.log(x)
    ### log X in the normalization term is omitted as it is a constant w.r.t. condition.
    mu = jnp.log(condition + self.bias) - 0.5 * jnp.square(self.sigma)
    square_error = utils.math.bsum(jnp.square(log_x - mu), keep=axes)
    return -0.5 * square_error / (self.sigma * self.sigma)

  def loss(self, x, condition, axes: tuple[int, ...] = ()):
    return self.un_log_p(x, condition, axes=axes)

  def sample(self, key, condition, *, shape: tuple[int, ...] = (), dtype=float):
    eps = jax.random.normal(key, shape=(*utils.shape.align(shape, condition), *self.shape), dtype=dtype)
    return jnp.exp(eps * self.sigma - 0.5 * self.sigma * self.sigma) * (condition + self.bias)

class normal_positive_definite(normal):
  """
  A normal distribution that samples only positive definite matrices.

  Encoding/decoding, log_p and un_log_p are exactly the same as in the normal distribution, i.e.,
  matrices are treated as a collection of independent normally distributed elements without taking 'definiteness'
  into account. Sampling, however, always returns positive definite matrices, but it *does not* draw from
  a normal distribution conditioned on positive 'definiteness' (especially when condition != 0),
  instead, it draws from normal distribution and projects the random matrix via polar decomposition:
  - A ~ N(0, 1);
  - U, S, V = svd(A)
  - result = V @ S @ V.T
  """
  def sample(self, key, condition, *, shape: tuple[int, ...] = (), dtype=float):
    assert len(self.shape) > 1, \
      f'Sampling from a positive definite the distribution is defined only for 2D+ tensors, got {shape}'

    A = self.sigma * jax.random.normal(
      key, shape=(*utils.shape.align(shape, condition), *self.shape,), dtype=dtype
    ) / math.sqrt(min(self.shape)) + condition

    _, S, Vt = jnp.linalg.svd(A)

    A = jnp.matmul(
      jnp.swapaxes(Vt, -1, -2) * S[..., None, :],
      Vt
    )
    return A

class normal_skew_symmetric(normal):
  """
  A normal distribution that samples only positive definite matrices.

  Encoding/decoding, log_p and un_log_p are exactly the same as in the normal distribution, i.e.,
  matrices are treated as a collection of independent normally distributed elements without taking 'definiteness'
  into account. Sampling, however, always returns positive definite matrices, but it *does not* draw from
  a normal distribution conditioned on positive 'definiteness' (especially when condition != 0),
  instead, it draws from normal distribution and projects the random matrix via polar decomposition:
  - A ~ N(0, 1);
  - U, S, V = svd(A)
  - result = V @ S @ V.T
  """
  def sample(self, key, condition, *, shape: tuple[int, ...] = (), dtype=float):
    shape = (*utils.shape.align(shape, condition), *self.shape)
    assert len(shape) > 1, 'Sampling from a positive definite the distribution is defined only for 2D+ tensors'
    assert shape[-1] == shape[-2], f'Normal Skew Symmetric can be only square got {shape[-2:]}'
    *_, n = shape

    A = self.sigma * jax.random.normal(key, shape=shape, dtype=dtype) / math.sqrt(n) + condition
    return 0.5 * (A - jnp.swapaxes(A, -1, -2))

class normal_negative_definite(normal_positive_definite):
  def sample(self, key, condition, *, shape: tuple[int, ...] = (), dtype=float):
    return -super().sample(key, condition, shape=shape, dtype=dtype)


class normal_positive_definite_bound(normal):
  __slots__ = ('sigma', 'singular_min', 'singular_max')

  @classmethod
  def full_parse(
      cls, mean: float, sigma: float,
      singular_min: float | None=None, singular_max: float | None=None,
      shape: Shape=()
  ):
    return cls(sigma, singular_min, singular_max, shape=shape), mean

  @classmethod
  def parse(
      cls, sigma: float,
      singular_min: float | None=None, singular_max: float | None=None,
      shape: Shape=()
  ):
    return cls(sigma, singular_min, singular_max, shape=shape)

  def __init__(self, sigma: float, singular_min: float | None, singular_max: float | None, shape: Shape=()):
    self.singular_min = singular_min
    self.singular_max = singular_max

    super().__init__(sigma, shape=shape)

  def sample(self, key, condition, *, shape: tuple[int, ...] = (), dtype=float):
    shape = (*utils.shape.align(shape, condition), *self.shape,)
    A = self.sigma * jax.random.normal(
      key, shape=shape, dtype=dtype
    ) / math.sqrt(min(shape[-2:])) + condition

    _, S, Vt = jnp.linalg.svd(A)
    S = jnp.clip(S, self.singular_min, self.singular_max)

    A = jnp.matmul(
      jnp.swapaxes(Vt, -1, -2) * S[..., None, :],
      Vt
    )
    return A

class normal_negative_definite_bound(normal_positive_definite_bound):
  def sample(self, key, condition, *, shape: tuple[int, ...] = (), dtype=float):
    return -super().sample(key, condition, shape=shape, dtype=dtype)

class identity(ConditionalDistribution):
  __slots__ = ('shape', )

  @classmethod
  def full_parse(cls, shape: Shape=()):
    return cls(shape=shape), None

  @classmethod
  def parse(cls, shape: Shape=()):
    return cls(shape=shape)

  def encode(self, x, condition):
    return x

  def decode(self, x_encoded, condition):
    return x_encoded

  def log_p(self, x, condition, axes: tuple[int, ...] = ()):
    Z = -0.5 * jnp.log(2 * jnp.pi)
    square_error = jnp.square(x)

    return Z - utils.math.bsum(0.5 * square_error, keep=axes)

  def un_log_p(self, x, condition, axes: tuple[int, ...]=()):
    square_error = utils.math.bsum(jnp.square(x), keep=axes)
    return -0.5 * square_error

  def loss(self, x, condition, axes: tuple[int, ...]=()):
    square_error = utils.math.bsum(jnp.square(x), keep=axes)
    return 0.5 * square_error

  def sample(self, key, condition, *, shape: tuple[int, ...] = (), dtype=float):
    shape = (*utils.shape.align(shape, condition), *self.shape)
    br = (
      *(slice(None, None, None) for _ in range(condition.ndim)),
      *(None for _ in range(len(shape) - condition.ndim))
    )
    return condition[br]
