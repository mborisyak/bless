"""
RKC2 advances the states the model integrates analytically (`analytical_step` not None) inside the stages and keeps the
last stage's value; the other states follow the Chebyshev recurrence.
"""
import numpy as np
import jax
import jax.numpy as jnp

import bless
from bless.mm.solver import RKC2

def _ecoli():
  model = bless.mm.from_config({'ecoli2024reduced': {}})
  sampler = bless.distributions.Sampler.from_profile(
    model, 'ecoli2024reduced.realistic', 'ecoli2024.controls', 'ecoli2024.likelihood', skip_unknown=True
  )
  key = jax.random.PRNGKey(0)
  params = jax.tree.map(lambda x: x[0], sampler.sample_parameters(key, batch=1))
  ### a fast-transfer, fast-probe system: the regime where an explicit recurrence of DOTa and DOTm swings
  params = params._replace(K_La=jnp.asarray(900.0), K_p=jnp.asarray(500.0), q_ox_max=jnp.asarray(0.65), K_s_ox=jnp.asarray(0.05))
  controls = jax.tree.map(lambda x: x[0, 0], sampler.sample_controls(key, batch=1, experiments=1))
  controls = controls._replace(X=jnp.asarray(0.3), S=jnp.asarray(5.0))
  return model, params, controls

def _stage_min_max(solver, model, params, controls, field, analytical):
  p = model.internal_parameters(params)
  s0 = model.initial_state(controls, p)
  def step(s, t):
    combined, _ = solver.step(s, t, p, controls, analytical=analytical)
    ws, _, _, _ = solver.stages(s, t, p, controls, analytical=analytical)
    vals = jnp.stack([getattr(w, field) for w in ws])
    return combined, (jnp.min(vals), jnp.max(vals))
  _, (lo, hi) = jax.lax.scan(step, s0, solver.grid)
  return float(jnp.min(lo)), float(jnp.max(hi))

def test_no_analytical_states_means_plain_recurrence():
  """A model whose `analytical_step` is None everywhere integrates identically with and without `analytical`."""
  model = bless.mm.from_config({'monod': {}})
  params = model.Parameters(q=jnp.asarray(2.0), K=jnp.asarray(0.3))
  controls = model.ControlParameters(A=jnp.asarray(2.0))
  solver = RKC2(model, dt=0.01, T=1.0, stages=4)
  a = solver.solve(controls, params, analytical=True)
  b = solver.solve(controls, params, analytical=False)
  for x, y in zip(a, b):
    assert np.array_equal(np.asarray(x), np.asarray(y))

def test_analytical_states_stay_physical_inside_the_stages():
  """At K_La dt = 36 the recurrence swings DOTa out of [0, DOT_lim]; the analytical stages never leave it."""
  model, params, controls = _ecoli()
  solver = RKC2(model, dt=0.04, T=24.0, stages=4)
  lo_a, hi_a = _stage_min_max(solver, model, params, controls, 'DOTa', analytical=True)
  lo_r, hi_r = _stage_min_max(solver, model, params, controls, 'DOTa', analytical=False)
  assert -1e-6 <= lo_a and hi_a <= 1.2, (lo_a, hi_a)
  assert not (-0.1 <= lo_r and hi_r <= 1.5), 'the plain recurrence did not swing in this test case; adjust the case'

def test_coarse_step_tracks_a_fine_reference():
  model, params, controls = _ecoli()
  grid = jnp.linspace(0.25, 24.0, 96)
  ref = RKC2(model, dt=0.001, T=24.0, stages=4).observe(controls, params, grid)
  obs = RKC2(model, dt=0.04, T=24.0, stages=4).observe(controls, params, grid)
  for f, tol in (('X', 0.05), ('S', 0.1), ('DOTm', 0.02)):
    d = float(jnp.max(jnp.abs(getattr(obs, f) - getattr(ref, f))))
    assert np.isfinite(d) and d < tol, (f, d)

def test_gradient_through_the_analytical_states_is_kept():
  """The gradient of the observables with respect to K_La and K_p is finite and non-zero (no freezing)."""
  model, params, controls = _ecoli()
  solver = RKC2(model, dt=0.04, T=6.0, stages=4)
  grid = jnp.linspace(0.25, 6.0, 24)
  def loss(k_la, k_p):
    obs = solver.observe(controls, params._replace(K_La=k_la, K_p=k_p), grid)
    return jnp.sum(jnp.square(obs.DOTm)) + jnp.sum(jnp.square(obs.X))
  g = jax.grad(loss, argnums=(0, 1))(jnp.asarray(900.0), jnp.asarray(500.0))
  assert all(np.isfinite(float(x)) for x in g)
  assert abs(float(g[0])) > 0 and abs(float(g[1])) > 0
