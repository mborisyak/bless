import os
import numpy as np
import jax

import bless
import matplotlib.pyplot as plt

def test_log_normal(seed, plot_root):
  d = bless.distributions.log_normal(sigma=0.1)
  xs = d.sample(jax.random.PRNGKey(seed), 1.0, shape=(32 * 1024, ))

  _ = plt.hist(xs, bins=100, label=f'std: {np.std(xs):.2f}')
  plt.legend()
  plt.savefig(os.path.join(plot_root, 'log-normal.png'))
  plt.close()