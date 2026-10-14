"""
The fit pipeline end to end on tiny problems: dataset generation, both fitting tasks, scoring, plots.
"""
import os
import copy
import glob
import threading
import importlib.util

import yaml
import numpy as np
import jax
import jax.numpy as jnp

import bless

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)

def load_script(name):
  spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, 'scripts', f'{name}.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module

def tiny_config(library: str, horizon: float | None=None):
  with open(os.path.join(ROOT, 'config', 'fit.yaml')) as f:
    config = yaml.safe_load(f)
  with open(os.path.join(ROOT, 'config', 'library', f'{library}.yaml')) as f:
    config['library'] = bless.utils.config.expand_library(yaml.safe_load(f))

  config['device'] = 'cpu'
  config['sampling'].update(samples=2, batch=2, plot=1, experiments={'train': 2, 'validation': 1, 'test': 1})
  config['fitting'].update(iterations=2, substeps=2, multistart=2, plot=1)

  if horizon is not None:
    library = config['library']
    for name in [bless.utils.config.GROUND_TRUTH, *bless.utils.config.models(library)]:
      for solver in library[name]['solver'].values():
        solver['T'] = horizon

  return config

def join_plots():
  for thread in threading.enumerate():
    if thread is not threading.main_thread():
      thread.join()

def run_pipeline(tmp_path, library, models, horizon=None):
  fit = load_script('fit')
  root = str(tmp_path) + '/'
  config = tiny_config(library, horizon=horizon)

  fit.generate(seed=0, root=root, **copy.deepcopy(config))
  data = os.path.join(fit.seed_root(root, 0), library)
  assert os.path.exists(os.path.join(data, 'ground-truth.npz'))

  for model in models:
    for exact in (False, True):
      c = copy.deepcopy(config); c['model'] = model
      fit.fit([0], root=root, exact=exact, progress=False, **c)

      task = 'approximate' if exact else 'mle'
      metadata = bless.utils.io.load_metadata(os.path.join(data, model, f'{task}.npz'))
      for split in ('train', 'validation', 'test'):
        assert metadata[split].shape == (2, {'train': 2, 'validation': 1, 'test': 1}[split])
        assert np.all(np.isfinite(metadata[split])), (model, task, split)

      ### every restart is kept and scored; the selected one is the saved fit
      selected = metadata['restarts.selected']
      assert metadata['restarts.losses'].shape == (2, 2, 2)
      for split in ('train', 'validation', 'test'):
        assert metadata[f'restarts.{split}'].shape == (2, *metadata[split].shape)
        chosen = metadata[f'restarts.{split}'][selected, np.arange(len(selected))]
        assert np.allclose(chosen, metadata[split], rtol=1.0e-5), (model, task, split)

  ### the plotted sample is drawn at random
  join_plots()
  assert glob.glob(os.path.join(data, 'ground-truth', 'sample-*.png'))
  assert glob.glob(os.path.join(data, models[-1], 'sample-mle-*.png'))
  assert glob.glob(os.path.join(data, models[-1], 'sample-approximate-*.png'))

def test_pipeline_regulatory(tmp_path):
  run_pipeline(tmp_path, 'regulatory', [
    'baseline', 'bless-4w12-reg3', 'residual-103-reg3', 'sequential-residual-44-reg3'
  ])

def test_seeds_fitted_together_match_seeds_fitted_alone(tmp_path):
  fit = load_script('fit')
  seeds = [3, 11]
  model = 'bless-4w12-reg3'
  config = tiny_config('regulatory')
  config['sampling']['plot'] = 0
  config['fitting']['plot'] = 0
  config['model'] = model

  alone, together = str(tmp_path / 'alone'), str(tmp_path / 'together')
  for s in seeds:
    for root in (alone, together):
      fit.generate(seed=s, root=root, **copy.deepcopy(config))
    fit.fit([s], root=alone, progress=False, **copy.deepcopy(config))

  fit.fit(seeds, root=together, progress=False, **copy.deepcopy(config))
  join_plots()

  for s in seeds:
    path = os.path.join('regulatory', model)
    for name in ('mle.npz', 'mle-restarts.npz', 'losses-mle.png'):
      assert os.path.exists(os.path.join(fit.seed_root(together, s), path, name)), (s, name)

    a = bless.utils.io.load_metadata(os.path.join(fit.seed_root(alone, s), path, 'mle.npz'))
    b = bless.utils.io.load_metadata(os.path.join(fit.seed_root(together, s), path, 'mle.npz'))
    for key in ('test', 'restarts.test'):
      assert np.array_equal(np.isfinite(a[key]), np.isfinite(b[key])), (s, key)
      finite = np.isfinite(a[key])
      assert np.allclose(a[key][finite], b[key][finite], rtol=1.0e-4), (s, key)

  ### a fitted seed is not fitted again
  mtime = os.path.getmtime(os.path.join(fit.seed_root(together, seeds[0]), path, 'mle.npz'))
  fit.fit(seeds, root=together, progress=False, **copy.deepcopy(config))
  assert os.path.getmtime(os.path.join(fit.seed_root(together, seeds[0]), path, 'mle.npz')) == mtime

def test_pipeline_ecoli_mixed_channel_lengths(tmp_path):
  ### the reduced base observes X, S at 24 points and DOTm at 120
  run_pipeline(tmp_path, 'ecoli', ['baseline', 'bless-4w12-reg3'], horizon=0.5)

def test_euler_matches_rkc2_on_a_controlled_model(rng):
  model = bless.mm.EColi2024Reduced()
  sampler = bless.distributions.Sampler.from_profile(
    model, 'ecoli2024reduced.realistic', 'ecoli2024.controls', 'ecoli2024.likelihood', skip_unknown=True
  )
  system = sampler.sample(rng, batch=2, experiments=3)

  euler = bless.mm.Euler(model, dt=1.0e-3, T=0.5)
  rkc2 = bless.mm.RKC2(model, dt=1.0e-2, T=0.5, stages=4)

  cs = jax.tree.map(lambda x: x[0, 0], system.controls)
  ps = jax.tree.map(lambda x: x[0], system.parameters)

  final_euler = jax.tree.map(lambda x: x[-1], euler.solve(cs, ps))
  final_rkc2 = jax.tree.map(lambda x: x[-1], rkc2.solve(cs, ps))

  for field in ('X', 'S', 'DOTm'):
    a, b = getattr(final_euler, field), getattr(final_rkc2, field)
    assert np.allclose(a, b, rtol=1.0e-2, atol=1.0e-3), (field, a, b)

  grid = jnp.linspace(0.05, 0.5, num=10)
  observed = euler.observe_batch(system.controls, system.parameters, grid)
  assert observed.X.shape == (2, 3, 10)

  observed = euler.observe_batch(system.controls, system.parameters, system.timestamps)
  assert observed.X.shape == system.timestamps.X.shape

def test_reverse_mode_gradient_matches_forward_mode(rng):
  base = bless.mm.Monod()
  solver = bless.mm.RKC2(base, dt=1.0e-2, T=1.0, stages=4, remat=True)
  sampler = bless.distributions.Sampler.from_profile(base, 'monod.common', 'regulatory.controls', 'regulatory.likelihood', skip_unknown=True)
  estimator = bless.fit.gradient.MaximumLikelihood(solver, sampler, n_multistart=2, c_reg=1.0e-3)

  truth = bless.mm.RegulatoryMonod()
  truth_solver = bless.mm.RKC2(truth, dt=1.0e-2, T=1.0, stages=4)
  truth_sampler = bless.distributions.Sampler.from_profile(truth, 'regulatory.common', 'regulatory.controls', 'regulatory.likelihood')

  key_system, key_obs, key_init = jax.random.split(rng, 3)
  system = truth_sampler.sample(key_system, batch=2, experiments=3)
  observations = truth_sampler.sample_observations(key_obs, truth_solver.observe_batch(system.controls, system.parameters, system.timestamps))
  parameters = estimator.sample_multi(key_init, 2, 2)

  grad, loss = estimator.grad_loss(observations, system.controls, parameters, system.timestamps)

  ### the forward-mode formulation: one Jacobian per trajectory, summed over experiments
  forward, _ = jax.vmap(
    jax.vmap(
      jax.vmap(jax.jacfwd(estimator.loss_fn, argnums=2, has_aux=True), in_axes=(0, 0, None, 0), out_axes=0),
      in_axes=(0, 0, 0, 0), out_axes=0
    ),
    in_axes=(None, None, 0, None), out_axes=0
  )(observations, system.controls, parameters, system.timestamps)
  forward = jax.tree.map(lambda g: jnp.sum(g, axis=2), forward)

  assert loss.shape == (2, 2, 3)
  for field in grad._fields:
    a, b = getattr(grad, field), getattr(forward, field)
    assert a.shape == b.shape, field
    assert np.allclose(a, b, rtol=1.0e-3, atol=1.0e-4), (field, np.abs(a - b).max())

def hybrid_at_a_point(rng, name, args, profiles, controls, t=0.5):
  model = bless.mm.from_config({name: args})
  sampler = bless.distributions.Sampler.from_profile(model, *profiles, controls, skip_unknown=True)
  key_p, key_c = jax.random.split(rng)
  parameters = jax.tree.map(lambda x: x[0], sampler.sample_parameters(key_p, batch=1))
  cp = jax.tree.map(lambda x: x[0, 0], sampler.sample_controls(key_c, batch=1, experiments=1))
  internal = model.internal_parameters(parameters)
  state = model.initial_state(cp, internal)
  if 'z' in state._fields:
    state = state._replace(z=jnp.ones(args['latent']))
  cs = model.controls(jnp.float32(t), cp)
  return model, internal, state, cs

def test_single_term_residual_is_a_source_or_a_first_order_sink(rng):
  cases = [
    ('blessed-monod', {'profiles': ['monod.common'], 'units': [8], 'latent': 3}, ['bless.common'], 'regulatory.controls'),
    ('residual-ecoli-monod', {'profiles': ['ecoli-monod.realistic'], 'units': [8]}, ['residual.common'], 'ecoli2024.controls'),
  ]
  for name, args, profiles, controls in cases:
    model, internal, state, cs = hybrid_at_a_point(rng, name, args, profiles, controls)
    rates = model.residual_rates(state, internal)
    assert set(rates) == set(model.coupled_states), name

    alg = model.algebraics(state, internal, cs)
    base = model.base.derivatives_algebraics(model.get_base_state(state), alg.base_parameters, cs, alg.base_algebraics)
    total = model.derivatives_algebraics(state, internal, cs, alg)
    for i, k in enumerate(model.coupled_states):
      x = getattr(state, k)
      source, gain = rates[k]
      expected = source - gain * x
      assert np.allclose(getattr(total, k) - getattr(base, k), expected, rtol=1.0e-5, atol=1.0e-6), (name, k)

def test_oxygen_equilibrium_solves_the_balance_with_the_residual():
  from bless.mm.ecoli_reduced_2024 import oxygen_equilibrium
  rng = np.random.default_rng(0)
  K, DOT_lim, K_o = rng.uniform(100.0, 1500.0, 64), rng.uniform(0.9, 1.1, 64), rng.uniform(1.0e-4, 0.1, 64)
  k_O = rng.uniform(0.0, 3000.0, 64)
  r = rng.normal(0.0, 100.0, 64)
  source, gain = np.maximum(r, 0.0), np.maximum(-r, 0.0)

  D = np.asarray(oxygen_equilibrium(K, DOT_lim, K_o, k_O, source, gain))
  assert np.all(D > 0.0)
  balance = K * (DOT_lim - D) - k_O * D / (K_o + D) + np.where(r > 0, r, r * D)
  ### float32: relative to the transfer scale
  assert np.allclose(balance / (K * DOT_lim), 0.0, atol=1.0e-6)

  ### no residual: the model's original expression, and the general one agrees with it
  base = np.asarray(oxygen_equilibrium(K, DOT_lim, K_o, k_O))
  assert np.allclose(np.asarray(oxygen_equilibrium(K, DOT_lim, K_o, k_O, 0.0 * K, 0.0 * K)), base, rtol=1.0e-4)

def test_oxygen_residual_enters_the_balance(rng):
  from bless.mm.ecoli2024 import oxygen_equilibrium
  for name, base_profile in (('blessed-ecoli', 'ecoli2024reduced.realistic'), ('residual-ecoli-oxidative', 'ecoli2024oxidative.realistic')):
    args = {'profiles': [base_profile], 'units': [8]}
    if name.startswith('blessed'): args['latent'] = 3
    model, internal, state, cs = hybrid_at_a_point(rng, name, args, ['bless.common' if name.startswith('blessed') else 'residual.common'], 'ecoli2024.controls', t=5.0)

    alg = model.algebraics(state, internal, cs)
    source, gain = alg.residuals['DOTa']
    base_state, p = model.get_base_state(state), alg.base_parameters
    plain = model.base.algebraics(base_state, p, cs)
    expected = oxygen_equilibrium(p.K_La + cs.f, p.DOT_lim, p.K_o, plain.k_O, source, gain)
    assert np.allclose(np.asarray(alg.base_algebraics.DOT_eq), np.asarray(expected), rtol=1.0e-6), name
    ### a source raises the equilibrium, a sink lowers it
    sign = np.sign(np.asarray(source) - np.asarray(gain))
    assert np.all(sign * (np.asarray(alg.base_algebraics.DOT_eq) - np.asarray(plain.DOT_eq)) >= 0.0), name

    ### the derivative path (`analytical=False`): the plain balance plus the residual rate
    base = model.base.derivatives_algebraics(base_state, p, cs, alg.base_algebraics)
    total = model.derivatives_algebraics(state, internal, cs, alg)
    assert np.allclose(float(total.DOTa), float(base.DOTa + source - gain * state.DOTa), rtol=1.0e-5, atol=1.0e-6), name

