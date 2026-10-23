"""
The E. coli figures: cultivations of the ground truth `ecoli2024` under the designs of the `ecoli-monod` library, and
the Monod base with its correctors fitted to them. Written to `<out>`: `ground-truth.pdf` (the truth's channels and
the feed per experiment), `fits.pdf` (truth, observations and every fitted model on the base's channels, training and
held-out experiments), `losses.pdf` (training loss per model). Printed: the NLL per observation point of every model
on the training and the held-out experiments.

    PYTHONPATH=$PWD python examples/ecoli.py --out imgs --device cuda
"""
import argparse
import os

import numpy as np
import jax
import optax
import yaml

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

import bless

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

LIBRARY = os.path.join(ROOT, 'config', 'library', 'ecoli-monod.yaml')
### the library entries shown, pattern -> label
MODELS = {
  'baseline': 'Monod base',
  'sequential-[0-9].*': 'Sequential',
  'residual-[0-9].*': 'Residual',
  'bless-[0-9].*': 'BLESS',
  'lin-bless-[0-9].*': 'linear BLESS',
}
CHANNELS = {'X': 'biomass, g/L', 'S': 'glucose, g/L', 'A': 'acetate, g/L', 'DOTm': 'dissolved oxygen, fraction'}
### adamaxw of the campaign protocol
LEARNING_RATE = 1.0e-3
B1, B2 = 0.1, 0.99
WEIGHT_DECAY = 1.0e-4
PLOT_POINTS = 480

def entries(library):
  """(label, library entry) of the shown models."""
  shown = []
  for pattern, label in MODELS.items():
    name, = bless.utils.config.models(library, [pattern])
    shown.append((label, library[name]))
  return shown

def feed(model, controls, grid):
  """The dilution rate of every (batch, experiment) design on the grid."""
  over_grid = jax.vmap(lambda t, c: model.controls(t, c).f, in_axes=(0, None))
  return jax.vmap(jax.vmap(over_grid, in_axes=(None, 0)), in_axes=(None, 0))(grid, controls)

def nll_per_point(sampler, observations, predicted):
  """Mean negative log-likelihood per observation point of the sampler's channels, per (batch, experiment)."""
  n_points = sum(getattr(observations, f).shape[-1] for f in sampler.likelihood._fields)
  return sampler.loss(observations, predicted, batch_axes=(0, 1)) / n_points

def fit(key, entry, controls, observations, timestamps, args):
  """The model of a library entry fitted to the experiments; (solver, sampler, losses, parameters)."""
  model = bless.mm.from_config(entry['model'])
  solver = bless.mm.solver.from_config(model, entry['solver'], remat=True)
  sampler = bless.distributions.Sampler.from_profile(model, *entry['profiles'], skip_unknown=True)
  estimator = bless.fit.gradient.MaximumLikelihood(
    solver, sampler, optimizer=optax.adamaxw(LEARNING_RATE, b1=B1, b2=B2, weight_decay=WEIGHT_DECAY),
    n_steps=args.iterations, n_substeps=args.substeps, n_multistart=args.multistart,
    c_reg=entry.get('regularization', None),
  )
  losses, estimated, _ = estimator.fit(key, controls, observations, timestamps=timestamps, progress=False)
  return solver, sampler, np.asarray(losses), estimated

def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--out', default='imgs')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--train', type=int, default=8, help='training experiments of the system')
  parser.add_argument('--held-out', type=int, default=4, dest='held_out', help='experiments shown but not fitted')
  parser.add_argument('--iterations', type=int, default=48)
  parser.add_argument('--substeps', type=int, default=128)
  parser.add_argument('--multistart', type=int, default=1)
  parser.add_argument('--device', default='cpu')
  args = parser.parse_args()

  with open(LIBRARY) as f:
    library = bless.utils.config.expand_library(yaml.safe_load(f))
  truth = library[bless.utils.config.GROUND_TRUTH]
  n_train, n_all = args.train, args.train + args.held_out
  splits = {'train': lambda tree: jax.tree.map(lambda x: x[:, :n_train], tree)}
  if args.held_out > 0:
    splits['held-out'] = lambda tree: jax.tree.map(lambda x: x[:, n_train:], tree)

  device = bless.utils.config.get_device(args.device)
  with jax.default_device(device):
    gt_model = bless.mm.from_config(truth['model'])
    gt_solver = bless.mm.solver.from_config(gt_model, truth['solver'])
    gt_sampler = bless.distributions.Sampler.from_profile(gt_model, *truth['profiles'])

    key_system, key_noise, key_fit = jax.random.split(jax.random.PRNGKey(args.seed), 3)
    system = gt_sampler.sample(key_system, batch=1, experiments=n_all)
    observe_truth = jax.jit(gt_solver.observe_batch)
    observations = gt_sampler.sample_observations(
      key_noise, observe_truth(system.controls, system.parameters, system.timestamps)
    )

    grid = gt_solver.grid[::max(1, len(gt_solver.grid) // PLOT_POINTS)]
    truth_curves = jax.tree.map(np.asarray, observe_truth(system.controls, system.parameters, grid))
    dilution = np.asarray(feed(gt_model, system.controls, grid))

    fits = {}
    for label, entry in entries(library):
      key_fit, key = jax.random.split(key_fit)
      print(f'{label}: fitting')
      solver, sampler, losses, estimated = fit(
        key, entry, splits['train'](system.controls), splits['train'](observations), splits['train'](system.timestamps), args
      )
      predict = jax.jit(solver.observe_batch)
      nll = {
        split: float(np.mean(nll_per_point(sampler, select(observations), predict(select(system.controls), estimated, select(system.timestamps)))))
        for split, select in splits.items()
      }
      curves = jax.tree.map(np.asarray, predict(system.controls, estimated, grid))
      fits[label] = (losses, curves, nll)
      print(f'{label}: NLL per point ' + ', '.join(f'{split} {v:.3f}' for split, v in nll.items()))

  os.makedirs(args.out, exist_ok=True)
  grid = np.asarray(grid)
  timestamps = jax.tree.map(np.asarray, system.timestamps)
  observations = jax.tree.map(np.asarray, observations)
  title = lambda j: f'experiment {j}' + ('' if j < n_train else ' (held out)')

  ### the truth: every channel and the feed per experiment
  n_rows = len(CHANNELS) + 1
  fig, axes = plt.subplots(n_rows, n_all, figsize=(3.2 * n_all, 2.2 * n_rows), sharex=True, squeeze=False)
  for j in range(n_all):
    for i, (f, unit) in enumerate(CHANNELS.items()):
      axes[i, j].plot(grid, getattr(truth_curves, f)[0, j], color='black')
      axes[i, j].scatter(getattr(timestamps, f)[0, j], getattr(observations, f)[0, j], s=8, color='black', alpha=0.6)
      axes[i, 0].set_ylabel(unit)
    axes[-1, j].plot(grid, dilution[0, j], color='tab:purple')
    axes[-1, j].set_xlabel('t, h')
    axes[0, j].set_title(title(j), fontsize=9)
  axes[-1, 0].set_ylabel('dilution rate f, 1/h')
  fig.tight_layout()
  fig.savefig(os.path.join(args.out, 'ground-truth.pdf'), bbox_inches='tight')
  plt.close(fig)

  ### the fitted models against the truth on the base's channels
  channels = next(iter(fits.values()))[1]._fields
  fig, axes = plt.subplots(len(channels), n_all, figsize=(3.2 * n_all, 2.4 * len(channels)), sharex=True, squeeze=False)
  for j in range(n_all):
    for i, f in enumerate(channels):
      ax = axes[i, j]
      ax.plot(grid, getattr(truth_curves, f)[0, j], color='black', label='ground truth' if i == j == 0 else None)
      ax.scatter(getattr(timestamps, f)[0, j], getattr(observations, f)[0, j], s=8, color='black', alpha=0.6)
      for k, (label, (_, curves, _)) in enumerate(fits.items()):
        ax.plot(grid, getattr(curves, f)[0, j], color=plt.cm.tab10(k), label=label if i == j == 0 else None)
      axes[i, 0].set_ylabel(CHANNELS.get(f, f))
    axes[-1, j].set_xlabel('t, h')
    axes[0, j].set_title(title(j), fontsize=9)
  fig.legend(loc='upper center', ncol=len(fits) + 1, bbox_to_anchor=(0.5, 1.04))
  fig.tight_layout()
  fig.savefig(os.path.join(args.out, 'fits.pdf'), bbox_inches='tight')
  plt.close(fig)

  ### the training losses
  fig, ax = plt.subplots(figsize=(6, 4))
  for k, (label, (losses, _, _)) in enumerate(fits.items()):
    ax.plot(losses.mean(axis=0), color=plt.cm.tab10(k), label=label)
  ax.set_xlabel('iteration')
  ax.set_ylabel('training loss')
  ax.set_yscale('log')
  ax.legend()
  fig.tight_layout()
  fig.savefig(os.path.join(args.out, 'losses.pdf'), bbox_inches='tight')
  plt.close(fig)
  print('figures in', os.path.abspath(args.out))

if __name__ == '__main__':
  main()
