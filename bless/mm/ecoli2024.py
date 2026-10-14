from collections import namedtuple

import jax
import jax.numpy as jnp

from .common import MechanisticModel
from .. utils.math import monod as _monod

__all__ = [
  'EColi2024', 'CONTROL_SHARPNESS', 'oxygen_yield', 'O2_PER_SUBSTRATE', 'BIOMASS_ELECTRON_FACTOR',
  'oxygen_equilibrium', 'oxygen_relaxation_rate', 'oxygen_step',
  'Parameters', 'InternalParameters',
  'State', 'Observables',
  'ControlParameters', 'Controls'
]

### width of the softmax blend of the feed phases, 1 / sqrt(CONTROL_SHARPNESS) hours; shared by the truth and every
### base model
CONTROL_SHARPNESS = 0.5

### electron balance of the oxygen yields: g O2 per g of glucose or acetate completely oxidised, and the electron
### content of biomass relative to the substrate's
O2_PER_SUBSTRATE = 1.067
BIOMASS_ELECTRON_FACTOR = 1.2

def oxygen_yield(Y_x):
  """Oxygen respired per gram of substrate through a route with biomass yield `Y_x` (electron balance)."""
  return O2_PER_SUBSTRATE * (1.0 - BIOMASS_ELECTRON_FACTOR * Y_x)

def oxygen_equilibrium(K, DOT_lim, K_o, k_O, source=None, gain=None):
  """
  The equilibrium of dissolved oxygen: the positive root of

    K (DOT_lim - DOT) - k_O DOT / (K_o + DOT) + source - gain DOT = 0,

  transfer, Monod uptake and an optional extra rate `source - gain DOT` (`source`, `gain` >= 0).
  """
  if source is None and gain is None:
    D = (DOT_lim - K_o - k_O / K)
    return (D + jnp.sqrt(jnp.square(D) + 4 * DOT_lim * K_o)) / 2

  K_total = K + gain
  B = (K * (DOT_lim - K_o) - k_O + source - gain * K_o) / K_total
  C = K_o * (K * DOT_lim + source) / K_total

  ### the positive root of DOT^2 - B DOT - C = 0, in the form without cancellation at B < 0
  root = jnp.sqrt(jnp.square(B) + 4 * C)
  return jnp.where(B >= 0, (B + root) / 2, 2 * C / (root - B))

def oxygen_relaxation_rate(K, DOT_lim, K_o, k_O, DOT, DOT_eq, source=None, gain=None):
  """
  The rate at which dissolved oxygen relaxes to its equilibrium, the secant of the balance

    rate = g(DOT) / (DOT_eq - DOT),   g(DOT) = K (DOT_lim - DOT) - k_O DOT / (K_o + DOT) + source - gain DOT,

  replaced by the tangent K + gain + k_O K_o / (K_o + DOT)^2 where the secant's cancellation error
  eps DOT_lim / |DOT_eq - DOT| exceeds the tangent's curvature error |DOT_eq - DOT| / (K_o + DOT).
  """
  g = K * (DOT_lim - DOT) - k_O * DOT / (K_o + DOT)
  tangent = K + k_O * K_o / jnp.square(K_o + DOT)
  if source is not None:
    g = g + source
  if gain is not None:
    g = g - gain * DOT
    tangent = tangent + gain
  delta = DOT_eq - DOT
  near = jnp.abs(delta) < jnp.sqrt(jnp.finfo(jnp.result_type(delta)).eps * DOT_lim * (K_o + DOT))
  return jnp.where(near, tangent, g / jnp.where(near, 1.0, delta))

def oxygen_step(dt, DOTa, DOTm, DOT_eq, rate, K_p):
  """
  The exact step of the oxygen channels over `dt` with `DOT_eq` and the rates frozen,

    d DOTa / dt = rate (DOT_eq - DOTa),   d DOTm / dt = K_p (DOTa - DOTm),

  the factor (e^{-rate dt} - e^{-K_p dt}) / (K_p - rate) by the series e^{-K_p dt} (1 + x / 2 + x^2 / 6) in
  x = (K_p - rate) dt where the direct form's cancellation error eps / |x| exceeds the series' truncation |x|^3 / 24.
  Returns `(DOTa, DOTm)` after `dt`.
  """
  e_a = jnp.exp(-rate * dt)
  e_m = jnp.exp(-K_p * dt)
  x = (K_p - rate) * dt
  small = jnp.abs(x) < (24 * jnp.finfo(jnp.result_type(x)).eps) ** 0.25
  probe = K_p * dt * jnp.where(small, e_m * (1.0 + x / 2 + jnp.square(x) / 6), (e_a - e_m) / jnp.where(small, 1.0, x))
  return (
    DOT_eq + (DOTa - DOT_eq) * e_a,
    DOT_eq + (DOTm - DOT_eq) * e_m + (DOTa - DOT_eq) * probe,
  )

### kinetic constants named after Anane et al. 2017; one rate and one biomass yield per route, the oxygen yields
### derived (`oxygen_yield`); H is the inverse oxygen solubility, K_La the mass transfer coefficient
Parameters = namedtuple(
  'Parameters', [
    'q_ox_max', 'q_an_max', 'q_of_max', 'q_a_max',
    'K_s_ox', 'Ki_a_ox',
    'K_s_an', 'ki_ox_an', 'Ki_a_an',
    'K_s_of', 'ki_ox_of', 'Ki_a_of',
    'K_a', 'Ki_s_a',
    'K_o',
    'Y_x_ox', 'Y_x_an', 'Y_x_of', 'Y_a_of', 'Y_x_a',
    'm_0',
    'DOT_lim', 'H', 'K_La', 'K_p',
  ]
)

### the parameters plus the derived oxygen yields
InternalParameters = namedtuple('InternalParameters', [*Parameters._fields, 'Y_o_ox', 'Y_o_a'])

State = namedtuple(
  'State', [
    'X', # biomass, g/L
    'S', # glucose, g/L
    'A', # acetate, g/L
    'DOTa', # dissolved oxygen, fraction of air saturation
    'DOTm', # oxygen measured, fraction, 0 min, 1 max;
  ]
)

### state without DOTa
Observables = namedtuple(
  'Observables', [
    'X', # biomass, g/L
    'S', # glucose, g/L
    'A', # acetate, g/L
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
    'DOT_eq', # the equilibrium of the oxygen balance at the current X, S, A and feed
    'DOT_rate', # the rate at which DOTa relaxes to it
    'DOT_balance', # the balance at the state, the rate of DOTa for `analytical=False`
    'k_O', # the oxygen demand at full oxygen, DOT fraction per hour
    'k_s', 'k_a'
  ]
)

class EColi2024(MechanisticModel):
  """
  E. coli on glucose with overflow metabolism and acetate cycling; kinetic constants named after
    Anane E, Neubauer P, Cruz Bournazou MN. Modelling overflow metabolism in Escherichia coli by acetate cycling.
    Biochemical Engineering Journal 2017;125:23-30.
  Glucose uptake through an oxidative route (q_ox, limited by the dissolved oxygen DOTa), an anaerobic route (q_an) and
  an overflow route (q_of, producing acetate); acetate re-assimilated (q_a) and respired; one biomass yield per route,
  growth = sum of route x yield - maintenance. `DOTa` relaxes to the equilibrium of its transfer / uptake balance at the
  balance's secant rate; the kinetics read `DOTa`; a first-order probe gives `DOTm`.
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
    return InternalParameters(
      **parameters._asdict(),
      Y_o_ox=oxygen_yield(parameters.Y_x_ox),
      Y_o_a=oxygen_yield(parameters.Y_x_a),
    )

  def initial_state(self, controls: ControlParameters, parameters: InternalParameters) -> State:
    return State(
      X=controls.X,
      S=controls.S,
      A=jnp.zeros_like(controls.X),
      DOTa=jnp.ones_like(controls.X),
      DOTm=jnp.ones_like(controls.X),
    )

  def observe(self, state: State, parameters: InternalParameters, algebraics: Algebraics) -> Observables:
    return Observables(
      X=state.X,
      S=state.S,
      A=state.A,
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

    k_s = parameters.q_ox_max * monod(parameters.Ki_a_ox, state.A) * monod(state.S, parameters.K_s_ox)
    k_a = parameters.q_a_max * monod(parameters.Ki_s_a, state.S) * monod(state.A, parameters.K_a)

    k_O = (k_s * parameters.Y_o_ox + k_a * parameters.Y_o_a) * state.X * parameters.H
    K_o = parameters.K_o
    K = parameters.K_La + controls.f

    DOT_eq = oxygen_equilibrium(K, parameters.DOT_lim, K_o, k_O)
    DOT_rate = oxygen_relaxation_rate(K, parameters.DOT_lim, K_o, k_O, state.DOTa, DOT_eq)

    DOT_balance = K * (parameters.DOT_lim - state.DOTa) - k_O * state.DOTa / (K_o + state.DOTa)
    return Algebraics(DOT_eq=DOT_eq, DOT_rate=DOT_rate, DOT_balance=DOT_balance, k_O=k_O, k_s=k_s, k_a=k_a)

  def derivatives_algebraics(self, state: State, parameters: InternalParameters, controls: Controls, algebraics: Algebraics) -> State:
    monod = lambda x, K: _monod(x, K, grad_clip=self.MONOD_GRAD_CLIP)

    k_s, k_a = algebraics.k_s, algebraics.k_a

    DOT_monod = monod(state.DOTa, parameters.K_o)
    q_ox = k_s * DOT_monod
    q_a = k_a * DOT_monod
    q_an = parameters.q_an_max * monod(state.S, parameters.K_s_an) * \
           monod(parameters.Ki_a_an, state.A) * monod(parameters.ki_ox_an * parameters.q_ox_max, q_ox)
    q_of = parameters.q_of_max * monod(state.S, parameters.K_s_of) * \
           monod(parameters.Ki_a_of, state.A) * monod(parameters.ki_ox_of * parameters.q_ox_max, q_ox)

    q_s = q_ox + q_an + q_of
    p_a = q_of * parameters.Y_a_of

    ### growth on glucose through the three routes and on re-assimilated acetate
    mu = q_ox * parameters.Y_x_ox + q_an * parameters.Y_x_an + q_of * parameters.Y_x_of + q_a * parameters.Y_x_a - parameters.m_0

    dilution = controls.f
    # ODE system
    return State(
      X=(mu - dilution) * state.X,
      S=-q_s * state.X + (controls.S_f - state.S) * dilution,
      A=(p_a - q_a) * state.X - dilution * state.A,
      ### the oxygen channels; `analytical_step` integrates them exactly, these rates serve `analytical=False`
      DOTa=algebraics.DOT_balance,
      DOTm=parameters.K_p * (state.DOTa - state.DOTm),
    )

  def analytical_step(self, dt: float, state: State, parameters: InternalParameters, controls: Controls, algebraics: Algebraics) -> State:
    DOTa, DOTm = oxygen_step(dt, state.DOTa, state.DOTm, algebraics.DOT_eq, algebraics.DOT_rate, parameters.K_p)
    return State(X=None, S=None, A=None, DOTa=DOTa, DOTm=DOTm)

  def clip(self, state: State, parameters: InternalParameters, algebraics: Algebraics) -> State:
    return State(
      X=jnp.maximum(state.X, 0.0),
      A=jnp.maximum(state.A, 0.0),
      S=jnp.maximum(state.S, 0.0),
      DOTa=jnp.clip(state.DOTa, 0.0, parameters.DOT_lim),
      DOTm=jnp.clip(state.DOTm, 0.0, parameters.DOT_lim),
    )
