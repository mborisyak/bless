from collections import namedtuple

import jax
import jax.numpy as jnp

from .common import MechanisticModel

__all__ = [
  'Chain'
]

State = namedtuple('State', ['A', 'B', 'P'])
Observables = namedtuple('Observables', ['A', ])
Parameters = namedtuple('Parameters', [
  'log_k_A_E1__AE1', 'log_k_AE1__A_E1', 'log_k_AE1__BE1', 'log_k_BE1__B_E1', 'log_k_B_E1__BE1',
  'log_k_B_E2__BE2', 'log_k_BE2__B_E2', 'log_k_BE2__PE2', 'log_k_PE2__P_E2',
  'E1_total', 'E2_total'
])
InternalParameters = namedtuple('InternalParameters', [
  'k_A_E1__AE1', 'k_AE1__A_E1', 'k_AE1__BE1', 'k_BE1__B_E1', 'k_B_E1__BE1',
  'k_B_E2__BE2', 'k_BE2__B_E2', 'k_BE2__PE2', 'k_PE2__P_E2',
  'E1_total', 'E2_total'
])

ControlParameters = namedtuple('ControlParameters', ['A', ])
Controls = namedtuple('Controls', [])
Algebraics = namedtuple('Algebraics', [
  'E1', 'AE1', 'BE1', 'E2', 'BE2', 'PE2'
])

class Chain(MechanisticModel):
  State = State
  Observables = Observables
  Parameters = Parameters
  InternalParameters = InternalParameters
  ControlParameters = ControlParameters
  Controls = Controls
  Algebraics = Algebraics

  def controls(self, timestamps, control_parameters):
    return Controls()

  def initial_state(self, controls, parameters):
    return self.State(
      A=controls.A,
      B=jnp.zeros_like(controls.A),
      P=jnp.zeros_like(controls.A),
    )

  def internal_parameters(self, parameters):
    return self.InternalParameters(
      k_A_E1__AE1=jnp.exp(parameters.log_k_A_E1__AE1),
      k_AE1__A_E1=jnp.exp(parameters.log_k_AE1__A_E1),
      k_AE1__BE1=jnp.exp(parameters.log_k_AE1__BE1),
      k_BE1__B_E1=jnp.exp(parameters.log_k_BE1__B_E1),
      k_B_E1__BE1=jnp.exp(parameters.log_k_B_E1__BE1),
      k_B_E2__BE2=jnp.exp(parameters.log_k_B_E2__BE2),
      k_BE2__B_E2=jnp.exp(parameters.log_k_BE2__B_E2),
      k_BE2__PE2=jnp.exp(parameters.log_k_BE2__PE2),
      k_PE2__P_E2=jnp.exp(parameters.log_k_PE2__P_E2),
      E1_total=parameters.E1_total,
      E2_total=parameters.E2_total,
    )

  def algebraics(self, state, parameters, controls):
    A, B, P = state.A, state.B, state.P

    E1_total = parameters.E1_total
    E2_total = parameters.E2_total

    k_A_E1__AE1 = parameters.k_A_E1__AE1
    k_AE1__A_E1 = parameters.k_AE1__A_E1
    k_AE1__BE1 = parameters.k_AE1__BE1
    k_BE1__B_E1 = parameters.k_BE1__B_E1
    k_B_E1__BE1 = parameters.k_B_E1__BE1
    k_B_E2__BE2 = parameters.k_B_E2__BE2
    k_BE2__B_E2 = parameters.k_BE2__B_E2
    k_BE2__PE2 = parameters.k_BE2__PE2
    k_PE2__P_E2 = parameters.k_PE2__P_E2

    ### QSS solution is obtained by sympy
    AE1 = A * E1_total * k_A_E1__AE1 * k_BE1__B_E1 / (
        A * k_AE1__BE1 * k_A_E1__AE1 + A * k_A_E1__AE1 * k_BE1__B_E1 + B * k_AE1__A_E1 * k_B_E1__BE1 + B * k_AE1__BE1 * k_B_E1__BE1 + k_AE1__A_E1 * k_BE1__B_E1 + k_AE1__BE1 * k_BE1__B_E1)
    BE1 = E1_total * (A * k_AE1__BE1 * k_A_E1__AE1 + B * k_AE1__A_E1 * k_B_E1__BE1 + B * k_AE1__BE1 * k_B_E1__BE1) / (
        A * k_AE1__BE1 * k_A_E1__AE1 + A * k_A_E1__AE1 * k_BE1__B_E1 + B * k_AE1__A_E1 * k_B_E1__BE1 + B * k_AE1__BE1 * k_B_E1__BE1 + k_AE1__A_E1 * k_BE1__B_E1 + k_AE1__BE1 * k_BE1__B_E1)
    BE2 = B * E2_total * k_B_E2__BE2 * k_PE2__P_E2 / (
        B * k_BE2__PE2 * k_B_E2__BE2 + B * k_B_E2__BE2 * k_PE2__P_E2 + k_BE2__B_E2 * k_PE2__P_E2 + k_BE2__PE2 * k_PE2__P_E2)
    PE2 = B * E2_total * k_BE2__PE2 * k_B_E2__BE2 / (
        B * k_BE2__PE2 * k_B_E2__BE2 + B * k_B_E2__BE2 * k_PE2__P_E2 + k_BE2__B_E2 * k_PE2__P_E2 + k_BE2__PE2 * k_PE2__P_E2)

    E1 = E1_total * k_BE1__B_E1 * (k_AE1__A_E1 + k_AE1__BE1) / (
        A * k_AE1__BE1 * k_A_E1__AE1 + A * k_A_E1__AE1 * k_BE1__B_E1 + B * k_AE1__A_E1 * k_B_E1__BE1 + B * k_AE1__BE1 * k_B_E1__BE1 + k_AE1__A_E1 * k_BE1__B_E1 + k_AE1__BE1 * k_BE1__B_E1)
    E2 = E2_total * k_PE2__P_E2 * (k_BE2__B_E2 + k_BE2__PE2) / (
        B * k_BE2__PE2 * k_B_E2__BE2 + B * k_B_E2__BE2 * k_PE2__P_E2 + k_BE2__B_E2 * k_PE2__P_E2 + k_BE2__PE2 * k_PE2__P_E2)

    return self.Algebraics(
      AE1=AE1, BE1=BE1, BE2=BE2, PE2=PE2, E1=E1, E2=E2
    )

  def derivatives_algebraics(self, state, parameters, controls, algebraics):
    E1, AE1, BE1, E2, BE2, PE2 = algebraics

    k_A_E1__AE1 = parameters.k_A_E1__AE1
    k_AE1__A_E1 = parameters.k_AE1__A_E1
    k_BE1__B_E1 = parameters.k_BE1__B_E1
    k_B_E1__BE1 = parameters.k_B_E1__BE1
    k_B_E2__BE2 = parameters.k_B_E2__BE2
    k_BE2__B_E2 = parameters.k_BE2__B_E2
    k_PE2__P_E2 = parameters.k_PE2__P_E2

    return self.State(
      A=-k_A_E1__AE1 * state.A * E1 + k_AE1__A_E1 * AE1,
      B=-state.B * (k_B_E1__BE1 * E1 + k_B_E2__BE2 * E2) + k_BE1__B_E1 * BE1 + k_BE2__B_E2 * BE2,
      P=+k_PE2__P_E2 * PE2,
    )

  def observe(self, state, parameters, algebraics):
    return self.Observables(A=state.A)

  def clip(self, state, parameters, algebraics):
    return self.State(
      A=jnp.clip(state.A, 0.0),
      B=jnp.clip(state.B, 0.0),
      P=jnp.clip(state.P, 0.0),
    )