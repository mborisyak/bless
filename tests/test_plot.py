import bless

import os
import numpy as np

import bokeh.plotting as plt
import bokeh.palettes as palettes

def test_losses(seed, plot_root):
  rng = np.random.default_rng(seed)

  n, m, k = 32, 1024, 3
  ts = np.arange(32)
  losses = [
    {
      'losses': np.exp(-0.1 * ts[:, None] + 0.5 * rng.exponential(size=(n, m))),
      'label': 'MSE',
      'title': 'training'
    },
    {
      'losses': np.exp(-0.1 * ts[:, None] + 0.5 * rng.exponential(size=(n, m))),
      'label': 'MSE',
      'title': 'val'
    }
  ]

  ts = np.sort(rng.uniform(size=(5, 1, 24)))
  params = rng.uniform(size=(5, ))[:, None, None]
  grid = np.linspace(0, 1, num=128)

  fs = {'A': lambda t: np.sqrt(t + params), 'B': lambda t: np.exp(-params * t), 'C': lambda t: params - t,}

  samples = {
    'observations': {
      k: f(ts) + 0.1 * rng.normal(size=ts.shape)
      for k, f in fs.items()
    },
    'predictions': {
      k: f(grid[None, None, :])
      for k, f in fs.items()
    },
    'observation_ts': ts,
    'prediction_ts': grid
  }

  bless.utils.plot.losses(os.path.join(plot_root, 'losses.html'), losses, samples=samples).join()