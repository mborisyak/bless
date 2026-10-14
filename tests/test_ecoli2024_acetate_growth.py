"""
Re-assimilated acetate yields biomass in the ground-truth E. coli model (Anane 2017, eq. 13: + q_sA Y_xa).
"""
import jax
import jax.numpy as jnp

import bless

def _setup():
  model = bless.mm.from_config({'ecoli2024': {}})
  sampler = bless.distributions.Sampler.from_profile(model, 'ecoli2024.realistic', 'ecoli2024.controls', 'ecoli2024.likelihood')
  params = jax.tree.map(lambda x: x[0], sampler.sample_parameters(jax.random.PRNGKey(1), batch=1))
  ### glucose exhausted, acetate present, oxygen saturated: only the acetate route runs
  state = model.State(X=jnp.asarray(1.0), S=jnp.asarray(0.0), A=jnp.asarray(1.0), DOTa=jnp.asarray(1.0), DOTm=jnp.asarray(1.0))
  controls = model.Controls(S_f=jnp.asarray(0.0), f=jnp.asarray(0.0))
  return model, params, state, controls

def test_acetate_consumption_yields_biomass():
  model, params, state, controls = _setup()
  p = model.internal_parameters(params)

  def growth(y_x_a):
    d = model.derivatives(state, p._replace(Y_x_a=jnp.asarray(y_x_a)), controls)
    return float(d.X / state.X + p.m_0), float(-d.A / state.X)

  mu_0, q_a = growth(0.0)
  assert q_a > 0.0, 'acetate is not consumed in the test state'
  assert abs(mu_0) < 1e-6, 'growth without an acetate yield should be maintenance only'
  for y in (0.25, 0.5, 0.75):
    mu, _ = growth(y)
    assert abs(mu - y * q_a) < 1e-6 * max(1.0, q_a), (y, mu, q_a)
