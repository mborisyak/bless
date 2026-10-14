import math

import jax
import jax.numpy as jnp

from ..common import MechanisticModel
from .common import Solver

__all__ = ['Euler', ]

class Euler(Solver):
  """
  Explicit Euler. Controls are evaluated at the start of every step, pulses at its end.
  """
  def __init__(self, model: MechanisticModel, dt: float, T: float, remat: bool=False, unroll: int=1):
    self.model = model
    self.dt = dt
    self.T = T
    self.remat = remat
    self.unroll = unroll

    self.n = int(math.ceil(T / dt))
    self.grid = jnp.linspace(dt, T, num=self.n)

  def _scan(self, control_parameters, parameters, output_f, analytical: bool):
    params = self.model.internal_parameters(parameters)
    state_0 = self.model.initial_state(control_parameters, params)

    def step(s, t):
      ### `t` is the end of the step
      cs_0 = self.model.controls(t - self.dt, control_parameters)
      cs = self.model.controls(t, control_parameters)

      algebraics = self.model.algebraics(s, params, cs_0)
      ds_dt = self.model.derivatives_algebraics(s, params, cs_0, algebraics)

      w = jax.tree.map(
        lambda x, dx_dt: x if dx_dt is None else x + dx_dt * self.dt,
        s, ds_dt
      )

      if analytical:
        s_analytic = self.model.analytical_step(self.dt, s, params, cs_0, algebraics)

        w = jax.tree.map(
          lambda x, x_analytic: x if x_analytic is None else x_analytic,
          w, s_analytic
        )

      clipped = self.model.clip(w, params, algebraics)

      combined = jax.tree.map(
        lambda x, x_: x if x_ is None else x_,
        clipped,
        self.model.pulse(w, params, cs, algebraics)
      )

      return combined, output_f(combined, params, cs, algebraics, ds_dt)

    _, states = jax.lax.scan(self._step(step), state_0, self.grid, unroll=self.unroll)

    return states
