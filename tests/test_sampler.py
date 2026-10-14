import os

import jax
import jax.numpy as jnp
import numpy as np

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

import bless

def test_encoding():
  xs = jnp.linspace(-10, 10, num=101)
  d = bless.distributions.log_normal(0.5)
  ys = d.decode(xs, 0.5)

  print(ys)
  print(d.encode(ys, 0.5))

def test_log_normal(seed, plot_root):
  rng = jax.random.PRNGKey(seed)

  import matplotlib.pyplot as plt
  fig = plt.figure(figsize=(9, 9))
  axes = fig.subplots()

  d, cond = bless.distributions.log_normal.full_parse(median=2.0, sigma=0.1)
  print(cond)
  xs = d.sample(rng, cond, shape=(32 * 1024, ))

  axes.hist(xs, bins=np.linspace(0, 4, num=101), histtype='step', label=f'mean={np.mean(xs)}')
  plt.legend()
  fig.savefig(os.path.join(plot_root, 'log-normal.png'))
  plt.close(fig)

def test_sigmoid(seed, plot_root):
  import matplotlib.pyplot as plt

  rng = jax.random.PRNGKey(seed)

  n = 32 * 1024

  distr, condition = bless.distributions.logit_normal.full_parse(0.0, 100.0, mu=-1.5, sigma=1.5)
  xs = distr.sample(rng, condition, shape=(n, ))
  encoded = distr.encode(xs, condition)
  decoded = distr.decode(encoded, condition)

  assert np.allclose(xs, decoded)

  fig = plt.figure()
  axes = fig.subplots(1, 2)
  axes[0].hist(xs, bins=np.linspace(condition - distr.delta, condition + distr.delta, num=129), histtype='step')
  _, bins, _ = axes[1].hist(encoded, bins=100, histtype='step')
  dx = np.mean(np.diff(bins))
  p = 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * np.square(bins))
  axes[1].plot(bins, p * dx * n)

  fig.savefig(os.path.join(plot_root, 'sigmoid.png'))
  plt.close(fig)

def test_grid(seed, plot_root):
  rng = jax.random.PRNGKey(seed)

  for n in range(1, 10):
    distribution, condition = bless.distributions.time.grid.full_parse(n=n, origin=1.0, horizon=2.0)
    print(condition)

  distribution, condition = bless.distributions.time.grid_uniform.full_parse(n=10, origin=1.0, horizon=2.0, delta=1.0)
  ts = distribution.sample(rng, condition, shape=(1024 * 1024, ))

  plt.figure(figsize=(12, 9))
  plt.hist(ts, bins=256, histtype='step')
  plt.savefig(os.path.join(plot_root, 'ts.png'))
  plt.close()