"""
The single-head BLESS variants: `SequentialBLESS` keeps the parameter head only, `ResidualBLESS` the residual head only.
"""
import numpy as np
import jax
import jax.numpy as jnp

import bless
from bless.mm.solver import RKC2

SPEC = {'profiles': ['ecoli2024oxidative.realistic'], 'latent': 4, 'units': [12]}

def _build(key):
  model = bless.mm.from_config({key: dict(SPEC)})
  sampler = bless.distributions.Sampler.from_profile(model, 'bless.common', 'ecoli2024.controls', 'ecoli2024.likelihood', skip_unknown=True)
  k1, k2 = jax.random.split(jax.random.PRNGKey(0))
  params = jax.tree.map(lambda x: x[0], sampler.sample_parameters(k1, batch=1))
  controls = jax.tree.map(lambda x: x[0, 0], sampler.sample_controls(k2, batch=1, experiments=1))
  return model, params, controls

def test_heads_and_parameters():
  full, _, _ = _build('blessed-ecoli-oxidative')
  seq, _, _ = _build('sequential-blessed-ecoli-oxidative')
  res, _, _ = _build('residual-blessed-ecoli-oxidative')
  names = lambda m: set(m.extension_parameters())
  wp = {n for n in names(full) if n.startswith(('wp_', 'Wp_', 'bp_'))}
  wr = {n for n in names(full) if n.startswith(('wr_', 'Wr_', 'br_'))}
  assert wp and wr
  assert names(seq) == names(full) - wr
  assert names(res) == names(full) - wp

def test_residual_variant_leaves_parameters_and_sequential_has_no_residual():
  res, params, controls = _build('residual-blessed-ecoli-oxidative')
  p = res.internal_parameters(params)
  state = res.initial_state(controls, p)
  coupled = res.get_coupled_parameters(state, p)
  for k, v in coupled.items():
    assert np.array_equal(np.asarray(v), np.asarray(getattr(p, k))), k
  assert set(res.residual_rates(state, p)) == set(res.coupled_states)

  seq, params, controls = _build('sequential-blessed-ecoli-oxidative')
  p = seq.internal_parameters(params)
  state = seq.initial_state(controls, p)
  assert seq.residual_rates(state, p) == {}
  assert any(not np.array_equal(np.asarray(v), np.asarray(getattr(p, k))) for k, v in seq.get_coupled_parameters(state, p).items())

def test_variants_integrate():
  for key in ('sequential-blessed-ecoli-oxidative', 'residual-blessed-ecoli-oxidative'):
    model, params, controls = _build(key)
    obs = RKC2(model, dt=0.025, T=6.0, stages=4).observe(controls, params, jnp.linspace(0.25, 6.0, 24))
    for f in obs._fields:
      assert np.isfinite(np.asarray(getattr(obs, f))).all(), (key, f)
