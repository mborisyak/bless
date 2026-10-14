from collections import namedtuple

import jax
import jax.numpy as jnp

from .common import MechanisticModel

__all__ = [
  'Monod',
]

State = namedtuple('State', ['A', 'P'])
Observables = namedtuple('Observables', ['A', ])
Parameters = namedtuple('Parameters', ['q', 'K'])
Algebraics = namedtuple('Algebraics', [])

ControlParameters = namedtuple('ControlParameters', ['A', ])
Controls = namedtuple('Controls', [])

class Monod(MechanisticModel):
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
      P=jnp.zeros_like(controls.A),
    )

  def observe(self, state, parameters, algebraics):
    return Observables(A=state.A, )

  def controls(self, timestamps, control_parameters):
    return Controls()

  def algebraics(self, state, parameters, controls):
    return Algebraics()

  def derivatives(self, state, parameters, controls):
    from .. import utils

    rate = parameters.q * utils.math.monod(state.A, parameters.K)

    return State(A=-rate, P=+rate)

  def analytical_step(self, dt: float, state, parameters, controls, algebraics):
    return State(A=None, P=None)

  def clip(self, state, parameters, algebraics) -> State:
    return State(A=jnp.maximum(state.A, 0.0), P=state.P)
