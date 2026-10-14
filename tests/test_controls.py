import bless

import os
import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

def test_pulse(seed, plot_root):
  rng = jax.random.PRNGKey(seed)

  n_grid = 1024
  grid = jnp.linspace(0, 1, num=n_grid)

  n_b, n_e, n_t = 3, 2, 5
  rng, key_ts, key_xs = jax.random.split(rng, num=3)
  ts = jax.random.uniform(key_ts, shape=(n_b, n_e, n_t), minval=0.0, maxval=1.0)
  xs = jax.random.normal(key_xs, shape=(n_b, n_e, n_t))

  ps = bless.utils.controls.pulses(grid, ts, xs)

  assert ps.shape == (n_grid, n_b, n_e)
  assert jnp.sum(ps != 0.0) == n_b * n_e * n_t

  dt = (grid[-1] - grid[0]) / (n_grid - 1)

  for i in range(n_b):
    for j in range(n_e):
      for k in range(n_t):
        index = (ts[i, j, k] / dt + 1).astype(jnp.int32)
        assert ps[index, i, j] == xs[i, j, k]

  fig = plt.figure(figsize=(2 * n_e, 2 * n_b))
  axes = fig.subplots(n_b, n_e)
  for i in range(n_b):
    for j in range(n_e):
      axes[i, j].plot(grid, ps[:, i, j])
      axes[i, j].scatter(ts[i, j], xs[i, j])

  fig.savefig(os.path.join(plot_root, 'pulses.png'))
  plt.close(fig)
