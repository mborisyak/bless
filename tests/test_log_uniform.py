import os

import numpy as np
import jax
import jax.numpy as jnp
from scipy import stats
import yaml

import bless
from bless.distributions.conditional import log_uniform

HERE = os.path.dirname(os.path.abspath(__file__))

LOWER, UPPER = 0.02, 30.0

def test_log_uniform_parse():
  d, c = log_uniform.full_parse(LOWER, UPPER)
  assert np.isclose(float(d.delta_log), 0.5 * (np.log(UPPER) - np.log(LOWER)))
  assert np.isclose(float(c), 0.5 * (np.log(UPPER) + np.log(LOWER)))

def test_log_uniform_sample_encode_decode(seed):
  d, c = log_uniform.full_parse(LOWER, UPPER)
  xs = d.sample(jax.random.PRNGKey(seed), c, shape=(64 * 1024, ))
  xs_np = np.asarray(xs)
  ### samples live on the support and log x is uniform there
  assert xs_np.min() >= LOWER * (1 - 1e-5) and xs_np.max() <= UPPER * (1 + 1e-5)
  p = stats.kstest(np.log(xs_np), stats.uniform(np.log(LOWER), np.log(UPPER) - np.log(LOWER)).cdf).pvalue
  assert p > 1e-3, f'log x is not uniform (KS p = {p})'
  ### the encoding is standard normal, the round trip exact
  enc = d.encode(xs, c)
  enc_np = np.asarray(enc)
  assert abs(enc_np.mean()) < 0.02 and abs(enc_np.std() - 1.0) < 0.02
  assert stats.kstest(enc_np, 'norm').pvalue > 1e-3
  dec = np.asarray(d.decode(enc, c))
  assert np.max(np.abs(dec - xs_np) / xs_np) < 1e-4
  ### saturation at the bounds with finite gradients
  z = jnp.array([-30.0, -6.0, 0.0, 6.0, 30.0])
  dec_z = np.asarray(d.decode(z, c))
  assert np.isclose(dec_z[0], LOWER) and np.isclose(dec_z[-1], UPPER)
  assert np.isclose(dec_z[2], np.sqrt(LOWER * UPPER), rtol=1e-5)
  grads = np.asarray(jax.vmap(jax.grad(lambda e: d.decode(e, c)))(z))
  assert np.isfinite(grads).all() and grads[2] > 0

def test_log_uniform_density():
  d, c = log_uniform.full_parse(LOWER, UPPER)
  xs = jnp.array([0.01, LOWER, 0.1, 1.0, 10.0, UPPER, 40.0])
  inside = (np.asarray(xs) >= LOWER) & (np.asarray(xs) <= UPPER)
  expected = np.where(inside, -np.log(np.asarray(xs)) - np.log(np.log(UPPER / LOWER)), -np.inf)
  got = np.asarray(d.log_p(xs, c, axes=(0, )))
  assert np.allclose(got[inside], expected[inside], rtol=1e-5) and np.all(np.isneginf(got[~inside]))
  ### the density integrates to one over the support
  grid = np.linspace(LOWER, UPPER, 400001)
  density = np.exp(np.asarray(d.log_p(jnp.asarray(grid), c, axes=(0, ))))
  assert abs(np.trapezoid(density, grid) - 1.0) < 1e-3
  ### un_log_p drops the constant, loss is -log p shifted to a minimum of 0 at the lower bound
  un = np.asarray(d.un_log_p(xs, c, axes=(0, )))
  assert np.allclose(un[inside], -np.log(np.asarray(xs))[inside])
  loss = np.asarray(d.loss(xs, c, axes=(0, )))
  assert np.allclose(loss[inside], np.log(np.asarray(xs)[inside] / LOWER)) and np.all(np.isposinf(loss[~inside]))
  assert loss[1] == 0.0

def test_log_uniform_in_sampler_profile():
  """`logU: [lower, upper]` in a profile decodes a hybrid's corrected base parameters into the box."""
  with open(os.path.join(os.path.dirname(HERE), 'config', 'library', 'regulatory.yaml')) as f:
    library = bless.utils.config.expand_library(yaml.safe_load(f))
  m = bless.mm.from_config(library['bless-4w12-reg3']['model'])
  for f, prior, cond in zip(m.base.Parameters._fields, m.priors, m.conditions):
    assert type(prior).__name__ == 'log_uniform', f
    lo, hi = float(prior.decode(jnp.array(-40.0), cond)), float(prior.decode(jnp.array(40.0), cond))
    expected = {'q': (0.02, 30.0), 'K': (0.0333, 30.0)}[f]
    assert np.isclose(lo, expected[0], rtol=1e-4) and np.isclose(hi, expected[1], rtol=1e-4)
