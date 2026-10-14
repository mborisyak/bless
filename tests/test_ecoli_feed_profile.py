"""
The truth, every base and every hybrid integrate the same feed from a design: one shared softmax sharpness.
"""
import numpy as np
import jax
import jax.numpy as jnp

import bless
from bless.mm.ecoli2024 import CONTROL_SHARPNESS

def test_feed_profiles_agree():
  truth = bless.mm.from_config({'ecoli2024': {}})
  sampler = bless.distributions.Sampler.from_profile(truth, 'ecoli2024.realistic', 'ecoli2024.controls', 'ecoli2024.likelihood')
  design = jax.tree.map(lambda x: x[0, 0], sampler.sample_controls(jax.random.PRNGKey(3), batch=1, experiments=1))
  t = jnp.linspace(0.0, 24.0, 481)
  f_truth = np.asarray(jax.vmap(lambda tt: truth.controls(tt, design).f)(t))
  for name in ('ecoli2024reduced', 'ecoli2024oxidative', 'ecoli-monod'):
    base = bless.mm.from_config({name: {}})
    assert base.control_sharpness == CONTROL_SHARPNESS == truth.control_sharpness
    f_base = np.asarray(jax.vmap(lambda tt: base.controls(tt, design).f)(t))
    assert np.array_equal(f_base, f_truth), name
  ### a hybrid builds its base through `Base.from_config(**kwargs)` with no sharpness in the kwargs
  for name, base_profile in (('blessed-ecoli', 'ecoli2024reduced.realistic'), ('sequential-ecoli-oxidative', 'ecoli2024oxidative.realistic'), ('residual-ecoli-monod', 'ecoli-monod.realistic')):
    hybrid = bless.mm.from_config({name: {'profiles': [base_profile], 'latent': 4, 'units': [12]}} if 'blessed' in name else {name: {'profiles': [base_profile], 'units': [12]}})
    assert hybrid.base.control_sharpness == CONTROL_SHARPNESS, name
    f_hybrid = np.asarray(jax.vmap(lambda tt: hybrid.controls(tt, design).f)(t))
    assert np.array_equal(f_hybrid, f_truth), name
