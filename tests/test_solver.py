import os
import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

import bless

def test_solver(seed, plot_root):
  rng = jax.random.PRNGKey(seed)
  mm = bless.mm.Monod()

  solver = bless.mm.RKC2(mm, dt=1.0e-2, T=1.0)

  sampler = bless.distributions.Sampler.from_profile(mm, 'monod.common', 'regulatory.controls', 'regulatory.likelihood')

  n_b, n_e = 5, 3

  system = sampler.sample(rng, batch=n_b, experiments=n_e)

  states = solver.solve_batch(system.controls, system.parameters)
  observables = solver.observe_batch(system.controls, system.parameters, system.timestamps)
  _, algebraics = solver.solve_algebraics_batch(system.controls, system.parameters)

  print('States', jax.tree.map(jnp.shape, states))
  print('Obs', jax.tree.map(jnp.shape, observables))
  print('Alg', jax.tree.map(jnp.shape, algebraics))

  n_s, n_obs, n_alg = len(states), len(observables), len(algebraics)

  fig = plt.figure(figsize=(4 * n_b, 4 * (n_s + n_obs + n_alg)))
  axes = fig.subplots(n_s + n_obs + n_alg, n_b, squeeze=False)

  for i in range(n_b):
    for j, field in enumerate(states._fields):
      for k in range(n_e):
        axes[j, i].plot(solver.grid, getattr(states, field)[i, k], color=plt.cm.tab10(k))
        axes[j, i].set_title(field)

    for j, field in enumerate(observables._fields):
      for k in range(n_e):
        axes[j + n_s, i].scatter(
          getattr(system.timestamps, field)[i, k], getattr(observables, field)[i, k],
          color=plt.cm.tab10(k)
        )
      axes[j + n_s, i].set_title(field)

    for j, field in enumerate(algebraics._fields):
      for k in range(n_e):
        axes[j, i].plot(solver.grid, getattr(algebraics, field)[i, k], color=plt.cm.tab10(k))
        axes[j, i].set_title(field)

  fig.tight_layout()
  fig.savefig(os.path.join(plot_root, 'monod.png'))
  plt.close(fig)
