import jax
import jax.numpy as jnp

__all__ = [
  'softmax_continuous',
  'pulses',
]

def softmax_continuous(t, points, values, sharpness: float):
  Ws = jax.nn.softmax(-sharpness * jnp.square(t - points))
  return jnp.sum(Ws * values)

def pulses(timestamps, points, values, zero: float=0.0):
  index = jnp.searchsorted(timestamps, points)
  *_, n_t = timestamps.shape
  *batch, _ = points.shape

  x = jnp.zeros(shape=(*batch, n_t), dtype=points.dtype) + zero
  return jax.vmap(
    jax.vmap(
      lambda a, i, v: a.at[i].set(v),
      in_axes=(0, 0, 0), out_axes=-1
    ),
    in_axes=(0, 0, 0), out_axes=-2
  )(x, index, values)
