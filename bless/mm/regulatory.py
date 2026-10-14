from collections import namedtuple

import jax
import jax.numpy as jnp

from .common import MechanisticModel

__all__ = [
  'RegulatoryMonod',
]


State = namedtuple('State', ['A', 'E', 'P'])
Observables = namedtuple('Observables', ['A', ])

Parameters = namedtuple('Parameters', [
  'k_A', 'K_A', 'q_E', 'K_E', 'k_E_deg',
])


ControlParameters = namedtuple('ControlParameters', ['A', ])
Controls = namedtuple('Controls', [])
Algebraics = namedtuple('Algebraics', ['A_E_monod'])

class RegulatoryMonod(MechanisticModel):
  State = State
  Observables = Observables

  Parameters = Parameters
  InternalParameters = Parameters

  Controls = Controls
  ControlParameters = ControlParameters
  Algebraics = Algebraics

  def initial_state(self, controls, parameters):
    return State(
      A=controls.A,
      E=jnp.zeros_like(controls.A),
      P=jnp.zeros_like(controls.A),
    )

  def observe(self, state, parameters, algebraics):
    return Observables(A=state.A, )

  def controls(self, timestamps, control_parameters):
    return Controls()

  def algebraics(self, state, parameters, controls):
    return Algebraics(
      A_E_monod=parameters.q_E * state.A / (state.A + parameters.K_E)
    )

  def derivatives_algebraics(self, state, parameters, controls, algebraics):
    A_E_monod = algebraics.A_E_monod

    rate = parameters.k_A * state.E * state.A / (state.A + parameters.K_A)
    dE_dt = A_E_monod - parameters.k_E_deg * state.E

    return State(A=-rate, P=+rate, E=+dE_dt)

  def analytical_step(self, dt: float, state, parameters, controls, algebraics):
    A_E_monod = algebraics.A_E_monod
    alpha = parameters.k_E_deg
    E_eq = A_E_monod / alpha

    return State(
      A=None,
      P=None,
      E=E_eq + (state.E - E_eq) * jnp.exp(-alpha * dt)
    )

  def clip(self, state, parameters, algebraics) -> State:
    return State(
      A=jnp.maximum(state.A, 0.0),
      E=jnp.maximum(state.E, 0.0),
      P=jnp.maximum(state.P, 0.0)
    )