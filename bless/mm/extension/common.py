from typing import ClassVar
from collections import namedtuple
from collections.abc import Sequence, Mapping

import jax
import jax.numpy as jnp

from ..common import MechanisticModel

__all__ = [
  'HybridExtension',
]

HybridAlgebraics = namedtuple(
  'HybridAlgebraics', ['extension_algebraics', 'base_algebraics', 'base_parameters', 'residuals']
)

class HybridExtension(MechanisticModel):
  Base: ClassVar[type[MechanisticModel]]
  base: ClassVar[MechanisticModel]

  ### what the corrector reads and corrects; the controls go to the base model untouched
  coupled_states: ClassVar[Sequence[str]]
  coupled_parameters: ClassVar[Sequence[str]]

  def __init_subclass__(cls,):
    if getattr(cls, 'Base', None) is not None:
      missing_states = [s for s in cls.coupled_states if s not in cls.Base.State._fields]
      assert len(missing_states) == 0, f'coupled states {missing_states} are not in the base state'

      missing_parameters = [s for s in cls.coupled_parameters if s not in cls.Base.Parameters._fields]
      assert len(missing_parameters) == 0, f'coupled parameters {missing_parameters} are not in the base parameters'

      ### the hybrid holds the base's Parameters; the base's InternalParameters are derived from the corrected
      ### parameters at every evaluation in `get_base_internal_parameters`
      cls.Parameters = namedtuple('Parameters', [
        *cls.Base.Parameters._fields, *cls.extension_parameters()
      ])
      cls.InternalParameters = namedtuple('InternalParameters', [
        *cls.Base.Parameters._fields, *cls.extension_internal_parameters()
      ])
      cls.State = namedtuple('State', [*cls.Base.State._fields, *cls.extension_states()])

      cls.ControlParameters = cls.Base.ControlParameters
      cls.Controls = cls.Base.Controls
      cls.Algebraics = HybridAlgebraics
      cls.Observables = cls.Base.Observables

  def __init__(self, priors, conditions):
    self.priors = priors
    self.conditions = conditions

  @classmethod
  def extension_states(cls) -> Sequence[str]:
    raise NotImplementedError()

  @classmethod
  def extension_parameters(cls) -> Sequence[str]:
    raise NotImplementedError()

  @classmethod
  def extension_internal_parameters(cls) -> Sequence[str]:
    return cls.extension_parameters()

  def get_coupled_parameters(self, state, parameters):
    """
    Returns encoded coupled parameters.
    """
    raise NotImplementedError()

  def parameter_shapes(self):
    raise NotImplementedError()

  def extension_state_shapes(self):
    raise NotImplementedError()

  def extension_initial_state(self, parameters, control_parameters) -> Mapping[str, jax.Array]:
    initial = {}
    for k in self.extension_states():
      x = getattr(parameters, f'{k}_initial')
      x_ = jnp.broadcast_to(x, shape=x.shape)
      initial[k] = x_

    return initial

  def extension_algebraics(self, state, parameters):
    raise NotImplementedError()

  def residual_rates(self, state, parameters) -> Mapping[str, tuple[jax.Array, jax.Array]]:
    """
    The residual corrections of the coupled states, `{state: (source, gain)}` for a rate `source - gain x`; empty for
    an extension without a residual path. An integrated state's correction enters its derivative, a quasi-steady
    state's its balance.
    """
    raise NotImplementedError()

  def total_derivatives(self, base_derivatives, extension_derivatives):
    total = {}

    for k in self.base.State._fields:
      base_dx_dt = getattr(base_derivatives, k)

      if extension_derivatives.get(k, None) is None:
        total[k] = base_dx_dt
      else:
        total[k] = base_dx_dt + extension_derivatives[k]

    return self.base.State(**total)

  def extension_dynamics(self, state, parameters, algebraics, base_derivatives):
    raise NotImplementedError()

  def extension_dynamics_analytical(self, dt, state, parameters, algebraics):
    return {k: None for k in self.extension_states()}

  def extension_clip(self, state, parameters, algebraics):
    return {k: getattr(state, k) for k in self.extension_states()}

  ### extension defaults and helper methods

  def residual_terms(self, state, rates):
    """The derivative corrections `source - gain x` of `rates`."""
    return {k: source - gain * getattr(state, k) for k, (source, gain) in rates.items()}

  def get_base_state(self, state):
    return self.base.State(**{
      k: getattr(state, k)
      for k in self.base.State._fields
    })

  def get_base_internal_parameters(self, state, parameters):
    coupled = self.get_coupled_parameters(state, parameters)
    rest = {
      k: getattr(parameters, k)
      for k in self.base.Parameters._fields
      if k not in coupled
    }

    parameters = self.decode_base_parameters(
      self.base.Parameters(**coupled, **rest)
    )

    internal = self.base.internal_parameters(parameters)
    return internal

  def encode_base_parameters(self, base_parameters):
    return self.base.Parameters(*(
      prior.encode(param, condition)
      for prior, param, condition in zip(self.priors, base_parameters, self.conditions)
    ))

  def decode_base_parameters(self, base_parameters):
    return self.base.Parameters(*(
      prior.decode(param, condition)
      for prior, param, condition in zip(self.priors, base_parameters, self.conditions)
    ))

  ### model methods

  def state_shapes(self):
    return self.State(**self.base.state_shapes()._asdict(), **self.extension_state_shapes())

  def observables_shapes(self):
    return self.base.observables_shapes()

  def controls_shapes(self):
    return self.ControlParameters(**self.base.controls_shapes()._asdict())

  def initial_state(self, controls, parameters):
    extension_initial_state = self.extension_initial_state(parameters, controls)
    # state0 = self.State(**extension_initial_state, **{f: None for f in self.base.State._fields})
    # base_parameters = self.get_base_internal_parameters(state0, parameters)

    base_initial_state = self.base.initial_state(controls, None)

    return self.State(
      **base_initial_state._asdict(),
      **extension_initial_state
    )

  def observe(self, state, parameters, algebraics):
    base_state = self.get_base_state(state)
    base_parameters = algebraics.base_parameters
    base_algebraics = algebraics.base_algebraics

    return self.base.observe(base_state, base_parameters, base_algebraics)

  def controls(self, timestamps, control_parameters):
    return self.base.controls(timestamps, control_parameters)

  def algebraics(self, state, parameters, controls):
    base_parameters = self.get_base_internal_parameters(state, parameters)
    base_state = self.get_base_state(state)

    extension_algebraics = self.extension_algebraics(state, parameters)

    residuals = self.residual_rates(state, parameters)
    base_algebraics = self.base.algebraics(base_state, base_parameters, controls)

    return HybridAlgebraics(
      extension_algebraics=extension_algebraics,
      base_algebraics=base_algebraics,
      base_parameters=base_parameters,
      residuals=residuals,
    )

  def derivatives_algebraics(self, state, parameters, controls, algebraics):
    base_state = self.get_base_state(state)
    base_parameters = algebraics.base_parameters
    base_algebraics = algebraics.base_algebraics

    base_derivatives = self.base.derivatives_algebraics(base_state, base_parameters, controls, base_algebraics)
    extension_derivatives = self.residual_terms(state, algebraics.residuals)
    total_derivatives = self.total_derivatives(base_derivatives, extension_derivatives)

    latent_derivatives = self.extension_dynamics(
      state, parameters, algebraics.extension_algebraics, base_derivatives
    )

    return self.State(
      **total_derivatives._asdict(),
      **latent_derivatives
    )

  def analytical_step(self, dt: float, state, parameters, controls, algebraics):
    base_state = self.get_base_state(state)
    base_parameters = algebraics.base_parameters
    base_algebraics = algebraics.base_algebraics

    base_analytical = self.base.analytical_step(dt, base_state, base_parameters, controls, base_algebraics)
    extension_analytical = self.extension_dynamics_analytical(
      dt, state, parameters, algebraics.extension_algebraics
    )

    return self.State(
      **base_analytical._asdict(),
      **extension_analytical
    )

  def clip(self, state, parameters, algebraics):
    base_state = self.get_base_state(state)
    base_parameters = algebraics.base_parameters
    base_algebraics = algebraics.base_algebraics

    return self.State(
      **self.base.clip(base_state, base_parameters, base_algebraics)._asdict(),
      **self.extension_clip(state, parameters, algebraics)
    )

  def pulse(self, state, parameters, controls, algebraics):
    base_state = self.get_base_state(state)
    base_parameters = algebraics.base_parameters
    base_algebraics = algebraics.base_algebraics

    return self.State(
      **self.base.pulse(base_state, base_parameters, controls, base_algebraics)._asdict(),
      **{k: getattr(state, k) for k in self.extension_states()}
    )

  def regularization(self, parameters):
    return None
