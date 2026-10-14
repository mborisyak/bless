import math

import numpy as np

import jax
import jax.numpy as jnp

from ..common import MechanisticModel
from .common import Solver

__all__ = ['RKC2', ]

class RKC2(Solver):
  """
  Second-order Runge-Kutta-Chebyshev with `stages` stages. Controls are evaluated at the stage times.

  A state for which `model.analytical_step` returns a value is advanced by that formula inside every stage, over
  `(c_j - c_{j-1}) dt` from the previous stage with its algebraics and controls; a state for which it returns None
  follows the Chebyshev recurrence; a state with neither an analytical step nor a derivative is carried through
  unchanged. `analytical=False` solves every state by the recurrence.
  """
  @staticmethod
  def _coefficients(s, eps=2.0 / 13.0):
    ### Chebyshev coefficients.
    w0 = 1.0 + eps / (s * s)
    th = np.arccosh(w0)

    T = lambda j: np.cosh(j * th)
    Tp = lambda j: j * np.sinh(j * th) / np.sinh(th)
    Tpp = lambda j: j * (
      j * np.cosh(j * th) * np.sinh(th) - np.sinh(j * th) * np.cosh(th)
    ) / np.sinh(th) ** 3

    w1 = Tp(s) / Tpp(s)

    b = np.zeros(s + 1)
    b[2:] = [Tpp(j) / Tp(j) ** 2 for j in range(2, s + 1)]
    b[0] = b[1] = b[2] if s >= 2 else 1.0
    a = np.array([1.0 - b[j] * T(j) for j in range(s + 1)])

    c = np.array([0.0] + [w1 * b[j] * Tp(j) for j in range(1, s + 1)])
    c[s] = 1.0

    mu = np.zeros(s + 1)
    nu = np.zeros(s + 1)
    mu_t = np.zeros(s + 1)
    gam_t = np.zeros(s + 1)

    mu_t[1] = b[1] * w1
    for j in range(2, s + 1):
      mu[j] = 2.0 * b[j] * w0 / b[j - 1]
      nu[j] = -b[j] / b[j - 2]
      mu_t[j] = 2.0 * b[j] * w1 / b[j - 1]
      gam_t[j] = -a[j - 1] * mu_t[j]

    return jnp.array(mu), jnp.array(nu), jnp.array(mu_t), jnp.array(gam_t), jnp.array(c)

  def __init__(self, model: MechanisticModel, dt: float, T: float, stages: int=5, remat: bool=False, unroll: int=1):
    self.model = model
    self.dt = dt
    self.T = T
    self.s = stages
    self.remat = remat
    self.unroll = unroll

    self.n = int(math.ceil(T / dt))
    self.grid = jnp.linspace(dt, T, num=self.n)

    self.mu, self.nu, self.mu_t, self.gam_t, self.c = self._coefficients(self.s)

  def _stage(self, t, j, control_parameters):
    """Controls at stage `j` of the step that ends at `t`."""
    return self.model.controls(t + (self.c[j] - 1.0) * self.dt, control_parameters)

  def stages(self, state, t, params, control_parameters, analytical: bool=True):
    """
    The stage values `[w_0, ..., w_s]` of the step ending at `t`, with the algebraics, derivatives and
    controls of the first stage.
    """
    model = self.model

    def analytic(dt_stage, w, cs, algebraics):
      """The analytical states after `dt_stage` from stage `w`, None per state without a formula."""
      if not analytical:
        return model.State(*(None for _ in model.State._fields))
      return model.analytical_step(dt_stage, w, params, cs, algebraics)

    cs_0 = self._stage(t, 0, control_parameters)

    algebraics = model.algebraics(state, params, cs_0)
    ds_dt = model.derivatives_algebraics(state, params, cs_0, algebraics)

    ### w_1 = w_0 + mu~_1 dt F(w_0)
    ws = [state, jax.tree.map(
      lambda x, dx_dt, x_a: (
        x_a if x_a is not None else
        x if dx_dt is None else
        x + self.mu_t[1] * self.dt * dx_dt
      ),
      state, ds_dt, analytic(self.c[1] * self.dt, state, cs_0, algebraics)
    )]

    for j in range(2, self.s + 1):
      ### the right-hand side and the analytical substep of stage j are evaluated at stage j - 1
      cs_prev = self._stage(t, j - 1, control_parameters)

      algebraics_prev = model.algebraics(ws[j - 1], params, cs_prev)
      dw_dt = model.derivatives_algebraics(ws[j - 1], params, cs_prev, algebraics_prev)

      ### w_j = (1 - mu_j - nu_j) w_0 + mu_j w_{j-1} + nu_j w_{j-2} + mu~_j dt F(w_{j-1}) + gam~_j dt F(w_0)
      ws.append(jax.tree.map(
        lambda x, x_1, x_2, dx_dt, dx_dt_0, x_a, j=j: (
          x_a if x_a is not None else
          x if dx_dt is None else
          (1.0 - self.mu[j] - self.nu[j]) * x + self.mu[j] * x_1 + self.nu[j] * x_2
          + self.mu_t[j] * self.dt * dx_dt + self.gam_t[j] * self.dt * dx_dt_0
        ),
        state, ws[j - 1], ws[j - 2], dw_dt, ds_dt, analytic((self.c[j] - self.c[j - 1]) * self.dt, ws[j - 1], cs_prev, algebraics_prev)
      ))

    return ws, algebraics, ds_dt, cs_0

  def finalize(self, w, params, cs, algebraics):
    """
    The non-ODE tail of a step: `clip` and `pulse`.
    """
    clipped = self.model.clip(w, params, algebraics)

    return jax.tree.map(
      lambda x, x_: x if x_ is None else x_,
      clipped,
      self.model.pulse(w, params, cs, algebraics)
    )

  def step(self, state, t, params, control_parameters, analytical: bool):
    """
    One RKC2 step ending at `t`.

    :param analytical: whether the states with a `model.analytical_step` are integrated by it;
      otherwise every state follows the recurrence.
    :return: `(next_state, (controls, algebraics, derivatives))`, the second element being what an
      `output_f` might ask for (controls of the last stage, algebraics and derivatives of the first).
    """
    ws, algebraics, ds_dt, cs_0 = self.stages(state, t, params, control_parameters, analytical=analytical)
    cs = self._stage(t, self.s, control_parameters)

    combined = self.finalize(ws[self.s], params, cs, algebraics)

    ### `ds_dt` is the model's full right-hand side at the first stage, including the analytical states' ODE rates
    return combined, (cs, algebraics, ds_dt)

  def _scan(self, control_parameters, parameters, output_f, analytical: bool):
    params = self.model.internal_parameters(parameters)
    state_0 = self.model.initial_state(control_parameters, params)

    def step(s, t):
      combined, (cs, algebraics, ds_dt) = self.step(s, t, params, control_parameters, analytical)
      return combined, output_f(combined, params, cs, algebraics, ds_dt)

    _, states = jax.lax.scan(self._step(step), state_0, self.grid, unroll=self.unroll)

    return states
