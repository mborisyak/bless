"""
The oxygen channels' exact stage step and the secant relaxation rate.
"""
import numpy as np
import jax
import jax.numpy as jnp
from scipy.linalg import expm

from bless.mm.ecoli2024 import oxygen_step, oxygen_relaxation_rate, oxygen_equilibrium

CASES = [   # (rate, K_p, dt): rate >> K_p, rate << K_p, rate == K_p, extreme rate, rate ~ K_p, probe faster than the transfer
  (1.0e4, 100.0, 0.02), (1.0, 300.0, 0.02), (300.0, 300.0, 0.02), (1.0e6, 500.0, 0.005), (100.0 + 1.0e-6, 100.0, 0.02),
  (2.0, 500.0, 0.02), (500.0, 5000.0, 0.02), (50.0, 50.0 + 1.0e-3, 0.0092),
]

def _reference(rate, K_p, dt, DOTa, DOTm, DOT_eq):
  M = np.array([[-rate, 0.0], [K_p, -K_p]])
  y = expm(M * dt) @ np.array([DOTa - DOT_eq, DOTm - DOT_eq])
  return y + DOT_eq

def test_step_matches_the_matrix_exponential():
  DOTa, DOTm, DOT_eq = 0.9, 0.3, 0.5
  for rate, K_p, dt in CASES:
    a, m = (float(x) for x in oxygen_step(dt, jnp.float32(DOTa), jnp.float32(DOTm), jnp.float32(DOT_eq), jnp.float32(rate), jnp.float32(K_p)))
    ra, rm = _reference(rate, K_p, dt, DOTa, DOTm, DOT_eq)
    assert abs(a - ra) < 2e-6 and abs(m - rm) < 2e-6, (rate, K_p, dt, a, ra, m, rm)

def test_step_is_finite_with_finite_gradients_at_extremes():
  f = lambda rate, K_p, dt: jnp.sum(jnp.stack(oxygen_step(dt, 0.9, 0.3, 0.5, rate, K_p)))
  for rate, K_p, dt in CASES + [(1.0e7, 500.0, 0.02), (1.0e-3, 500.0, 0.5), (500.0, 500.0, 0.5)]:
    v, g = jax.value_and_grad(f, argnums=(0, 1))(jnp.float32(rate), jnp.float32(K_p), jnp.float32(dt))
    assert np.isfinite(float(v)) and all(np.isfinite(float(x)) for x in g), (rate, K_p, dt, v, g)

def test_secant_rate_reproduces_the_balance_and_its_tangent():
  rng = np.random.default_rng(0)
  for _ in range(200):
    K, L, K_o, k_O = rng.uniform(100, 450), rng.uniform(0.9, 1.1), 10 ** rng.uniform(-3, -1), 10 ** rng.uniform(0, 3.5)
    eq = float(oxygen_equilibrium(K, L, K_o, k_O))
    D = rng.uniform(0.0, 1.1)
    rate = float(oxygen_relaxation_rate(K, L, K_o, k_O, D, eq))
    g = K * (L - D) - k_O * D / (K_o + D)
    assert rate > 0
    if abs(eq - D) > 1e-3:
      assert abs(rate * (eq - D) - g) <= 1e-4 * max(abs(g), 1.0), (rate, eq, D, g)
    ### at the equilibrium the tangent, and the gain adds to it
    tangent = K + k_O * K_o / (K_o + eq) ** 2
    assert abs(float(oxygen_relaxation_rate(K, L, K_o, k_O, eq, eq)) - tangent) <= 1e-3 * tangent
    assert abs(float(oxygen_relaxation_rate(K, L, K_o, k_O, eq, eq, source=0.0, gain=7.0)) - (tangent + 7.0)) <= 1e-3 * tangent
