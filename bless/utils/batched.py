from builtins import range as _range

import jax
import jax.numpy as jnp

__all__ = [
  'number_of_batches',
  'range'
]

def get_size(*arrays, axis: int):
  if any(x.ndim < axis + 1 for x in arrays):
    raise ValueError(f'Some arrays have lower dimensionality than implies by axis {axis}.')

  sizes = set([x.shape[axis] for x in arrays])

  if len(sizes) > 1:
    raise ValueError(f'Shapes of the inputs arrays are inconsistent across axis {axis}.')

  size, = sizes
  return size

def number_of_batches(size: int, batch: int, subbatch: bool=True):
  if subbatch:
    return size // batch + (0 if size % batch == 0  else 1)
  else:
    return size // batch

class range(object):
  def __init__(self, size: int, batch: int, subbatch: bool=True):
    self.size = size
    self.batch = batch

    self.n_full_batches = self.size // batch
    self.remainder = self.size % batch
    self.subbatch = subbatch

  def __len__(self):
    if self.subbatch and self.remainder > 0:
      return self.n_full_batches + 1
    else:
      return self.n_full_batches

  def __iter__(self):
    for i in _range(self.n_full_batches):
      index = slice(i * self.batch, (i + 1) * self.batch)
      yield index

    if self.remainder > 0 and self.subbatch:
      index = slice(self.size - self.remainder, self.size)
      yield index

class sequence(object):
  def __init__(self, size: int, batch: int, subbatch: bool=True):
    self.size = size
    self.batch = batch

    self.n_full_batches = self.size // batch
    self.remainder = self.size % batch
    self.subbatch = subbatch

  def __len__(self):
    if self.subbatch and self.remainder > 0:
      return self.n_full_batches + 1
    else:
      return self.n_full_batches

  def __iter__(self):
    for i in range(self.n_full_batches):
      index = slice(i * self.batch, (i + 1) * self.batch)
      yield index

    if self.remainder > 0 and self.subbatch:
      index = slice(self.size - self.remainder, self.size)
      yield index

def evaluate(f, batch: int):
  def _eval(*args, **kwargs):
    n, = set([x.shape[0] for x in jax.tree.leaves(args)])
    effective_batch = min(n, batch)

    n_full_batches = n // effective_batch
    full_batches_size = n_full_batches * effective_batch
    remainder = n - full_batches_size

    def batched(xs):
      xs_full_batches = jnp.reshape(
        xs[:full_batches_size],
        shape=(n_full_batches, effective_batch, *xs.shape[1:])
      )

      return xs_full_batches

    batched_args = jax.tree.map(batched, args)
    results_shaped = jax.eval_shape(f, *args, **kwargs)

    ### if necessary, allocating +1 batch
    n_batches = n_full_batches + (0 if remainder == 0 else 1)
    full_results_batched = jax.tree.map(
      lambda shaped: jnp.zeros(shape=(n_batches, effective_batch, *shaped.shape[1:]), dtype=shaped.dtype),
      results_shaped
    )

    def eval_batched(i, results):
      args_batch = jax.tree.map(lambda x: x[i], batched_args)

      return jax.tree.map(
        lambda r, rb: r.at[i].set(rb),
        results, f(*args_batch, **kwargs)
      )

    full_results_batched = jax.lax.fori_loop(0, n_full_batches, eval_batched, init_val=full_results_batched)
    full_results = jax.tree.map(
      lambda x: jnp.reshape(x, shape=(x.shape[0] * x.shape[1], *x.shape[2:]))[:n],
      full_results_batched
    )

    if remainder > 0:
      ### the overlap is computed twice, generally faster than recompiling for smaller batch
      last_batch = jax.tree.map(lambda x: x[n - effective_batch:], args)
      full_results = jax.tree.map(
        lambda r, rb: r.at[n - effective_batch:].set(rb),
        full_results, f(*last_batch, **kwargs)
      )

    return full_results

  return _eval