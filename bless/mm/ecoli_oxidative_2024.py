from collections import namedtuple

import jax
import jax.numpy as jnp

from .common import MechanisticModel
from .ecoli2024 import CONTROL_SHARPNESS, oxygen_yield, oxygen_equilibrium, oxygen_relaxation_rate, oxygen_step
from .. utils.math import monod as _monod

__all__ = [
  'EColi2024Oxidative',
  'Parameters', 'InternalParameters',
  'State', 'Observables',
  'ControlParameters', 'Controls'
]

### the parameters of EColi2024Reduced without the non-oxidative route
Parameters = namedtuple(
  'Parameters', [
    'q_ox_max',
    'K_s_ox',
    'K_o',
    'Y_x_ox',
    'm_0',
    'DOT_lim', 'H', 'K_La', 'K_p',
  ]
)

### the parameters plus the derived oxygen yield of the oxidative route (see ecoli2024.oxygen_yield)
InternalParameters = namedtuple('InternalParameters', [*Parameters._fields, 'Y_o_ox'])

State = namedtuple(
  'State', [
    'X', # biomass, g/L
    'S', # glucose, g/L
    'DOTa', # dissolved oxygen, fraction of air saturation
    'DOTm', # oxygen measured, fraction, 0 min, 1 max;
  ]
)

### state without DOTa
Observables = namedtuple(
  'Observables', [
    'X', # biomass, g/L
    'S', # glucose, g/L
    'DOTm', # oxygen measured, fraction, 0 min, 1 max;
  ]
)

Controls = namedtuple(
  'Controls', [
    'f', # dilution rate F/V, 1/h (constant volume)
    'S_f', # feed concentration, g/L,
  ]
)

ControlParameters = namedtuple(
  'ControlParameters', [
    'X', 'S',

    'f', # dilution rate F/V, 1/h (constant volume)
    'S_f', # feed concentration, g/L,
    'ts',
  ]
)

Algebraics = namedtuple(
  'Algebraics', [
    'DOT_eq', # the equilibrium of the oxygen balance at the current X, S and feed
    'DOT_rate', # the rate at which DOTa relaxes to it
    'DOT_balance', # the balance at the state, the rate of DOTa for `analytical=False`
    'k_O', # the oxygen demand at full oxygen, DOT fraction per hour
    'k_s'
  ]
)

class EColi2024Oxidative(MechanisticModel):
  """
  `EColi2024Reduced` without the anaerobic route `q_an`: glucose is consumed only through the oxygen-limited `q_ox`.
  """
  State = State
  Observables = Observables

  Parameters = Parameters
  InternalParameters = InternalParameters

  Controls = Controls
  ControlParameters = ControlParameters
  Algebraics = Algebraics

  MONOD_GRAD_CLIP = 1.0e+2

  def __init__(self, control_sharpness: float=CONTROL_SHARPNESS, monod_grad_clip=1.0e+2):
    self.control_sharpness = control_sharpness
    self.MONOD_GRAD_CLIP = monod_grad_clip
    super().__init__()

  def internal_parameters(self, parameters: Parameters) -> InternalParameters:
    return InternalParameters(**parameters._asdict(), Y_o_ox=oxygen_yield(parameters.Y_x_ox))

  def initial_state(self, controls: ControlParameters, parameters: InternalParameters) -> State:
    return State(
      X=controls.X,
      S=controls.S,
      DOTa=jnp.ones_like(controls.X),
      DOTm=jnp.ones_like(controls.X),
    )

  def observe(self, state: State, parameters: InternalParameters, algebraics: Algebraics) -> Observables:
    return Observables(
      X=state.X,
      S=state.S,
      DOTm=state.DOTm
    )

  def controls(self, timestamps: jax.Array, control_parameters: ControlParameters) -> Controls:
    from .. import utils
    flow = utils.controls.softmax_continuous(
      timestamps, control_parameters.ts, control_parameters.f, sharpness=self.control_sharpness
    )
    concentration = control_parameters.S_f * jnp.ones_like(flow)

    return Controls(S_f=concentration, f=flow)

  def algebraics(self, state: State, parameters: InternalParameters, controls: Controls) -> Algebraics:
    monod = lambda x, K: _monod(x, K, grad_clip=self.MONOD_GRAD_CLIP)

    k_s = parameters.q_ox_max * monod(state.S, parameters.K_s_ox)

    k_O = (k_s * parameters.Y_o_ox) * state.X * parameters.H
    K_La = parameters.K_La
    K_o = parameters.K_o
    DOT_lim = parameters.DOT_lim

    K = K_La + controls.f

    DOT_eq = oxygen_equilibrium(K, DOT_lim, K_o, k_O)
    DOT_rate = oxygen_relaxation_rate(K, DOT_lim, K_o, k_O, state.DOTa, DOT_eq)

    DOT_balance = K * (DOT_lim - state.DOTa) - k_O * state.DOTa / (K_o + state.DOTa)
    return Algebraics(
      DOT_eq=DOT_eq,
      DOT_rate=DOT_rate,
      DOT_balance=DOT_balance,
      k_O=k_O,
      k_s=k_s,
    )

  def derivatives_algebraics(self, state: State, parameters: InternalParameters, controls: Controls, algebraics: Algebraics) -> State:
    monod = lambda x, K: _monod(x, K, grad_clip=self.MONOD_GRAD_CLIP)

    k_s = algebraics.k_s

    DOT_monod = monod(state.DOTa, parameters.K_o)
    q_s = k_s * DOT_monod

    mu = q_s * parameters.Y_x_ox - parameters.m_0

    dilution = controls.f

    # ODE system
    return State(
      X=(mu - dilution) * state.X,
      S=-q_s * state.X + (controls.S_f - state.S) * dilution,
      ### the oxygen channels; `analytical_step` integrates them exactly, these rates serve `analytical=False`
      DOTa=algebraics.DOT_balance,
      DOTm=parameters.K_p * (state.DOTa - state.DOTm),
    )

  def analytical_step(self, dt: float, state: State, parameters: InternalParameters, controls: Controls, algebraics: Algebraics) -> State:
    DOTa, DOTm = oxygen_step(dt, state.DOTa, state.DOTm, algebraics.DOT_eq, algebraics.DOT_rate, parameters.K_p)
    return State(X=None, S=None, DOTa=DOTa, DOTm=DOTm)

  def clip(self, state: State, parameters: InternalParameters, algebraics: Algebraics) -> State:
    return State(
      X=jnp.maximum(state.X, 0.0),
      S=jnp.maximum(state.S, 0.0),
      DOTm=jnp.clip(state.DOTm, 0.0, parameters.DOT_lim),
      DOTa=jnp.clip(state.DOTa, 0.0, parameters.DOT_lim),
    )
