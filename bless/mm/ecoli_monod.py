from collections import namedtuple

import jax
import jax.numpy as jnp

from .common import MechanisticModel
from .ecoli2024 import CONTROL_SHARPNESS
from .. utils.math import monod as _monod

__all__ = [
  'EColiMonod',
  'Parameters', 'InternalParameters',
  'State', 'Observables',
  'ControlParameters', 'Controls'
]

Parameters = namedtuple(
  'Parameters', [
    'q_s_max', 'K_s', 'Y_s', 'm_0'
  ]
)

InternalParameters = Parameters

State = namedtuple(
  'State', [
    'X',     # biomass, g/L
    'S',     # glucose, g/L
  ]
)

Observables = namedtuple(
  'Observables', [
    'X',     # biomass, g/L
    'S',     # glucose, g/L
  ]
)

Controls = namedtuple(
  'Controls', [
    'f',     # flow, L/hour
    'S_f',   # feed concentration, g/L
  ]
)

ControlParameters = namedtuple(
  'ControlParameters', [
    'X', 'S',
    'f',     # flow, L/hour
    'S_f',   # feed concentration, g/L
    'ts',
  ]
)

Algebraics = namedtuple(
  'Algebraics', []
)

class EColiMonod(MechanisticModel):
  """
  Simplified E. coli model based on EColi2024 but with only aerobic growth.

  Key simplifications:
  - Only oxidative glucose utilization (no anaerobic pathway)
  - No overflow metabolism (no acetate production)
  - No acetate consumption
  - No oxygen at all (no DOT state, no oxygen limitation)
  - Fed-batch operation with flow control

  Model equations:
  - q_s = q_s_max * monod(S, K_s)
  - mu = q_s * Y_s - m_0
  - dX/dt = (mu - f) * X
  - dS/dt = -q_s * X + (S_f - S) * f
  """
  State = State
  Observables = Observables

  Parameters = Parameters
  InternalParameters = Parameters

  Controls = Controls
  ControlParameters = ControlParameters
  Algebraics = Algebraics

  MONOD_GRAD_CLIP = 1.0e+2

  def __init__(self, control_sharpness: float=CONTROL_SHARPNESS, monod_grad_clip=1.0e+2):
    self.control_sharpness = control_sharpness
    self.MONOD_GRAD_CLIP = monod_grad_clip
    super().__init__()

  def initial_state(self, controls: ControlParameters, parameters: Parameters) -> State:
    return State(X=controls.X, S=controls.S,)

  def observe(self, state: State, parameters: Parameters, algebraics: Algebraics) -> Observables:
    return Observables(X=state.X, S=state.S,)

  def controls(self, timestamps: jax.Array, control_parameters: ControlParameters) -> Controls:
    from .. import utils
    flow = utils.controls.softmax_continuous(
      timestamps, control_parameters.ts, control_parameters.f, sharpness=self.control_sharpness
    )
    concentration = control_parameters.S_f * jnp.ones_like(flow)

    return Controls(S_f=concentration, f=flow)

  def algebraics(self, state: State, parameters: Parameters, controls: Controls) -> Algebraics:
    return Algebraics()

  def derivatives_algebraics(self, state: State, parameters: Parameters, controls: Controls, algebraics: Algebraics) -> State:
    monod = lambda x, K: _monod(x, K, grad_clip=self.MONOD_GRAD_CLIP)

    q_s = parameters.q_s_max * monod(state.S, parameters.K_s)

    # Growth rate
    mu = q_s * parameters.Y_s - parameters.m_0

    dilution = controls.f

    # ODE system
    return State(
      X=(mu - dilution) * state.X,
      S=-q_s * state.X + (controls.S_f - state.S) * dilution
    )

  def analytical_step(self, dt: float, state: State, parameters: Parameters, controls: Controls, algebraics: Algebraics) -> State:
    return State(X=None, S=None,)

  def clip(self, state: State, parameters: Parameters, algebraics: Algebraics) -> State:
    return State(X=jnp.maximum(state.X, 0.0), S=jnp.maximum(state.S, 0.0))
