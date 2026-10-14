import os
import threading

import numpy as np
import jax
import jax.numpy as jnp

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


import bless

### serialises the plot threads
_plot_lock = threading.Lock()

def _plot_predictions(path, observations, trajectories, kwargs):
  """
  Observations: (ts, observations)
  trajectories: dicts with timestamps, trajectory and label keys
  kwargs: name -> (ts, trajectory)
  """
  all_obs = []
  if observations is not None:
    ts, obs = observations
    all_obs.append(obs)

  trajectories = [
    *trajectories,
    *[
      { 'timestamps': ts, 'trajectory': xs, 'label': None if name.startswith('_') else name }
      for name, (ts, xs) in kwargs.items()
    ]
  ]

  for traj in trajectories:
    all_obs.append(traj['trajectory'])

  fields = set([f for xs in all_obs for f in xs._fields])
  n_obs = len(fields)

  (n_b, n_e), = set([x.shape[:2] for xs in all_obs for x in xs])

  fig = plt.Figure(figsize=(n_obs * 4, n_b * n_e * 4))
  axes = fig.subplots(n_b * n_e, n_obs, squeeze=False)

  for k, field in enumerate(fields):
    y_min, y_max = bless.utils.plot.get_range(*[getattr(xs, field) for xs in all_obs if hasattr(xs, field)])

    for i in range(n_b):
      for j in range(n_e):
        ax = axes[i * n_e + j, k]

        ax.set_title(f'{field}[{i}, {j}]')
        if observations is not None:
          ts, obs = observations
          if hasattr(obs, field):
            ax.scatter(getattr(ts, field)[i, j], getattr(obs, field)[i, j], color='black')

        for l, traj in enumerate(trajectories):
          traj = traj.copy()
          ts, xs = traj.pop('timestamps'), traj.pop('trajectory')
          color = traj.pop('color', plt.cm.tab10(l))
          if hasattr(xs, field):
            ax.plot(ts, getattr(xs, field)[i, j], color=color, **traj)

        ax.set_ylim([y_min, y_max])
        ax.legend()

  fig.tight_layout()
  fig.savefig(path)
  plt.close(fig)

def plot_predictions(path, observations=None, *trajectories, **kwargs):
  def target(*args):
    with _plot_lock:
      _plot_predictions(*args)

  threading.Thread(
    target=target, args=(path, observations, trajectories, kwargs),
    daemon=False
  ).start()

def continuous_controls(model, control_parameters, grid):
  """
  The controls signal of every `(batch, experiment)` design on a shared time grid, `(batch, experiment, n_t)` per field.
  """
  over_grid = jax.vmap(lambda t, cp: model.controls(t, cp), in_axes=(0, None), out_axes=-1)
  over_experiments = jax.vmap(over_grid, in_axes=(None, 0), out_axes=0)
  over_batch = jax.vmap(over_experiments, in_axes=(None, 0), out_axes=0)

  return over_batch(grid, control_parameters)

def states_on_grid(solver, control_parameters, parameters, grid):
  """
  The state of every `(batch, experiment)` trajectory interpolated onto a shared time grid.
  """
  return jax.vmap(
    jax.vmap(lambda cs, ps: solver.select(solver.solve(cs, ps), grid), in_axes=(0, None), out_axes=0),
    in_axes=(0, 0), out_axes=0
  )(control_parameters, parameters)

def score(sampler, observations, predicted):
  """
  Mean negative log-likelihood per observation point over the scored channels, per `(batch, experiment)`.
  Channels may carry different numbers of points.
  """
  total = sampler.loss(observations, predicted, batch_axes=(0, 1))
  n_points = sum(getattr(observations, f).shape[-1] for f in sampler.likelihood._fields)

  return total / n_points

def _plot_losses(losses, path):
  n_batch, n_steps = losses.shape

  fig = plt.Figure(figsize=(4, 4 * n_batch))
  axes = fig.subplots(n_batch, 1, squeeze=False)
  for i in range(n_batch):
    ax = axes[i, 0]
    ax.plot(losses[i])
    ax.set_yscale('log')
  fig.tight_layout()
  fig.savefig(path)
  plt.close(fig)

def plot_losses(losses, path):
  def target(*args):
    with _plot_lock:
      _plot_losses(*args)

  threading.Thread(target=target, args=(losses, path), daemon=False).start()

SPLITS = ('train', 'validation', 'test')

def seed_root(root: str, seed: int):
  """Where seed `seed` keeps its datasets and fits."""
  return os.path.join(root, f'seed-{seed}')

def split_experiments(tree, n_train: int, n_validation: int):
  """`tree` split along the experiment axis into its train, validation and test experiments."""
  bounds = (0, n_train, n_train + n_validation, None)
  return tuple(jax.tree.map(lambda x: x[:, lo:hi], tree) for lo, hi in zip(bounds[:-1], bounds[1:]))

def generate(seed: int, root: str='data/', **config):
  """The ground-truth dataset of seed `seed` (the sampling seed shifted by `seed`) under `seed_root(root, seed)`."""
  device = bless.utils.config.get_device(config.get('device', None))

  with jax.default_device(device):
    rng = jax.random.PRNGKey(config['sampling']['seed'] + seed)

    library = bless.utils.config.expand_library(config['library'])
    library_root = os.path.join(seed_root(root, seed), library[bless.utils.config.NAME])
    truth = library[bless.utils.config.GROUND_TRUTH]

    model = bless.mm.from_config(truth['model'])
    sampler = bless.distributions.Sampler.from_profile(model, *truth['profiles'])
    solver = bless.mm.solver.from_config(model, truth['solver'])

    n, n_batch = config['sampling']['samples'], config['sampling']['batch']
    n_exp = sum(config['sampling']['experiments'][split] for split in SPLITS)

    rng, key_sample = jax.random.split(rng, num=2)
    samples = sampler.sample(key_sample, batch=n, experiments=n_exp)

    observations = []

    observe = jax.jit(solver.observe_batch)

    for indx in bless.utils.batched.range(n, n_batch, subbatch=True):
      rng, key_observations = jax.random.split(rng, num=2)
      sample_batch = jax.tree.map(lambda x: x[indx], samples)
      obs = observe(sample_batch.controls, sample_batch.parameters, sample_batch.timestamps)
      obs = sampler.sample_observations(key_observations, obs)
      observations.append(obs)

    observations = jax.tree.map(
      lambda *xs: jnp.concatenate(xs, axis=0),
      *observations
    )

    os.makedirs(library_root, exist_ok=True)
    bless.utils.io.save(
      os.path.join(library_root, 'ground-truth.npz'),
      controls=samples.controls,
      parameters=samples.parameters,
      timestamps=samples.timestamps,
      observations=observations
    )

    n_plot = config['sampling']['plot']
    if n_plot > 0:
      os.makedirs(os.path.join(library_root, 'ground-truth'), exist_ok=True)

      _, key_display = jax.random.split(rng, num=2)
      display_index = jnp.sort(jax.random.choice(key_display, n, shape=(n_plot, ), replace=False))
      display_grid = jnp.linspace(solver.grid[0], solver.grid[-1], num=128)

      for i in display_index:
        display_samples, display_observations = jax.tree.map(lambda x: x[i:(i + 1)], (samples, observations))

        display_trajectories = states_on_grid(solver, display_samples.controls, display_samples.parameters, display_grid)
        display_controls = continuous_controls(model, display_samples.controls, display_grid)

        plot_predictions(
          os.path.join(library_root, 'ground-truth', f'sample-{i}.png'),
          (display_samples.timestamps, display_observations),
          {'label': 'ground-truth', 'timestamps': display_grid, 'trajectory': display_trajectories, 'color': 'black'},
          {'label': 'controls', 'timestamps': display_grid, 'trajectory': display_controls, 'color': 'black'}
        )

def fit(seeds, root: str='data/', exact: bool=False, progress: bool=True, **config):
  """
  Fits the library entry `config['model']` to the datasets of every seed of `seeds` in one optimisation, the seeds'
  systems stacked into one batch. Seed `s` reads its dataset under `seed_root(root, s)`, draws its initial parameters
  from the fitting seed shifted by `s` and writes, under the model's directory there, `<task>.npz` (the fit, its
  predictions, the per-experiment scores of every split and of every restart), `<task>-restarts.npz` (the parameters
  of every restart), the loss curves and the sample plots. A seed with a `<task>.npz` is skipped. The task is `mle`,
  the observations, or with `exact` `approximate`, the ground truth's noiseless observables on its grid.
  """
  device = bless.utils.config.get_device(config.get('device', None))
  task = 'approximate' if exact else 'mle'
  n_train, n_validation = config['sampling']['experiments']['train'], config['sampling']['experiments']['validation']
  n_plot = config['fitting']['plot']

  library = bless.utils.config.expand_library(config['library'])
  model_name = config['model']
  entry, truth = library[model_name], library[bless.utils.config.GROUND_TRUTH]

  def library_root(seed):
    return os.path.join(seed_root(root, seed), library[bless.utils.config.NAME])

  pending = [s for s in seeds if not os.path.exists(os.path.join(library_root(s), model_name, f'{task}.npz'))]
  for s in seeds:
    if s not in pending:
      print(f'Seed {s}: fitted, skipped')

  if len(pending) == 0:
    return

  with jax.default_device(device):
    gt_model = bless.mm.from_config(truth['model'])
    gt_solver = bless.mm.solver.from_config(gt_model, truth['solver'])
    observe_truth = jax.jit(gt_solver.observables_batch)

    model = bless.mm.from_config(entry['model'])
    ### steps per loop iteration of the solver scans; `fitting.unroll` in config/fit.yaml
    solver = bless.mm.solver.from_config(model, entry['solver'], remat=True, unroll=config['fitting'].get('unroll', 1))
    sampler = bless.distributions.Sampler.from_profile(model, *entry['profiles'], skip_unknown=True)
    predict = jax.jit(solver.observe_batch)

    estimator = bless.fit.gradient.MaximumLikelihood(
      solver, sampler, optimizer=bless.utils.config.optimizer(config['fitting']['optimizer']),
      n_steps=config['fitting']['iterations'], n_substeps=config['fitting']['substeps'],
      n_multistart=config['fitting']['multistart'],
      c_reg=entry.get('regularization', None),
      grad_clip=config['fitting'].get('gradient_clipping', None)
    )

    ### per seed: its dataset, the (controls, observations, timestamps) of every split, its number of systems, the
    ### key of its sample plots and its initial parameters
    datasets, splits, n_systems, keys, initial = {}, {}, {}, {}, []
    for s in pending:
      datasets[s] = bless.utils.io.load(os.path.join(library_root(s), 'ground-truth.npz'), gt_model, device=device)
      controls, gt_parameters, timestamps, observations = datasets[s]
      n_systems[s], = {x.shape[0] for x in jax.tree.leaves(controls)}

      cs = split_experiments(controls, n_train, n_validation)
      if exact:
        obs = tuple(observe_truth(c, gt_parameters) for c in cs)
        ### the observables sit on the ground truth's grid; the fitted model interpolates onto it
        ts = (gt_solver.grid, ) * len(SPLITS)
      else:
        obs = split_experiments(observations, n_train, n_validation)
        ts = split_experiments(timestamps, n_train, n_validation)
      splits[s] = tuple(zip(cs, obs, ts))

      keys[s], key_initial = jax.random.split(jax.random.PRNGKey(config['fitting']['seed'] + s), num=2)
      initial.append(estimator.initial(key_initial, n_systems[s]))

    print(f'Fitting seeds {", ".join(str(s) for s in pending)}')
    concatenate = lambda axis: (lambda *xs: jnp.concatenate(xs, axis=axis))
    trains = [splits[s][0] for s in pending]
    cs_train, obs_train = (jax.tree.map(concatenate(0), *(t[k] for t in trains)) for k in (0, 1))
    ### the exact task's timestamps are the shared grid
    ts_train = trains[0][2] if exact else jax.tree.map(concatenate(0), *(t[2] for t in trains))

    losses, estimated, restarts = estimator.fit(
      None, cs_train, obs_train, timestamps=ts_train, progress=progress, initial=jax.tree.map(concatenate(1), *initial)
    )

    offset = 0
    for s in pending:
      part = slice(offset, offset + n_systems[s])
      offset += n_systems[s]

      controls, gt_parameters, timestamps, observations = datasets[s]
      fitted = jax.tree.map(lambda x: x[part], estimated)
      restart_parameters = jax.tree.map(lambda x: x[:, part], restarts.parameters)
      n_restarts = restarts.losses.shape[0]

      predictions = [predict(c, fitted, t) for c, _, t in splits[s]]
      metadata = {
        split: np.asarray(score(sampler, o, p)) for split, (_, o, _), p in zip(SPLITS, splits[s], predictions)
      }
      for split, (c, o, t) in zip(SPLITS, splits[s]):
        metadata[f'restarts.{split}'] = np.stack([
          np.asarray(score(sampler, o, predict(c, jax.tree.map(lambda x: x[r], restart_parameters), t)))
          for r in range(n_restarts)
        ], axis=0)
      metadata['restarts.losses'] = np.asarray(restarts.losses[:, part])
      metadata['restarts.selected'] = np.asarray(restarts.selected[part])

      model_root = os.path.join(library_root(s), model_name)
      os.makedirs(model_root, exist_ok=True)
      plot_losses(losses[part], os.path.join(model_root, f'losses-{task}.png'))
      bless.utils.io.save(os.path.join(model_root, f'{task}-restarts.npz'), parameters=restart_parameters)
      bless.utils.io.save(
        os.path.join(model_root, f'{task}.npz'),
        controls=controls, parameters=fitted, timestamps=None,
        observations=jax.tree.map(concatenate(1), *predictions), **metadata
      )
      print(f'Seed {s}: written')

      if n_plot > 0:
        _, key_display = jax.random.split(keys[s], num=2)
        display_index = jnp.sort(jax.random.choice(key_display, n_systems[s], shape=(n_plot, ), replace=False))
        display_grid = jnp.linspace(gt_solver.grid[0], gt_solver.grid[-1], num=128)

        for i in display_index:
          display_controls, display_parameters, display_fitted, display_timestamps, display_observations = jax.tree.map(
            lambda x: x[i:(i + 1)], (controls, gt_parameters, fitted, timestamps, observations)
          )

          plot_predictions(
            os.path.join(model_root, f'sample-{task}-{i}.png'),
            None if exact else (display_timestamps, display_observations),
            {
              'label': 'ground-truth', 'timestamps': display_grid, 'color': 'black',
              'trajectory': gt_solver.observe_batch(display_controls, display_parameters, display_grid),
            },
            {
              'label': model_name, 'timestamps': display_grid, 'color': plt.cm.tab10(0),
              'trajectory': predict(display_controls, display_fitted, display_grid),
            },
            {
              'label': 'controls', 'timestamps': display_grid, 'color': 'black',
              'trajectory': continuous_controls(gt_model, display_controls, display_grid),
            },
          )

def available(**config):
  """The fitted entries of `config['library']`."""
  for model_name in bless.utils.config.models(bless.utils.config.expand_library(config['library'])):
    print(model_name)

if __name__ == '__main__':
  ### gearup is imported for the command line only
  import gearup

  gearup.gearup(
    generate=generate,
    fit=fit,
    available=available,
  ).with_config('config/fit.yaml')()
