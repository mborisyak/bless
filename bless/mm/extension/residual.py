from typing import Mapping, Sequence

import jax
import jax.numpy as jnp

from .common import HybridExtension

__all__ = [
  'Residual',
  'source_sink',
]

def source_sink(r, states):
  """
  The single-term residual `r` (one column per state) as `{state: (relu(r), relu(-r))}`: a source where positive and
  a first-order sink where negative, the rate being `source - gain x`.
  """
  return {k: (jax.nn.relu(r[..., i]), jax.nn.relu(-r[..., i])) for i, k in enumerate(states)}

class Residual(HybridExtension):
  ### one hidden layer of `units[0]` then a projection to the outputs
  layers: int = 2

  @classmethod
  def extension_states(cls) -> Sequence[str]:
    return ()

  @classmethod
  def extension_parameters(cls, ) -> Sequence[str]:
    return (
      'b_0',
      *[f'w_0_state_{k}' for k in cls.coupled_states],
      *[f'b_{i}' for i in range(1, cls.layers)],
      *[f'W_{i}' for i in range(1, cls.layers)],
    )

  @classmethod
  def from_config(cls, profiles, units: Sequence[int], root=None, library=None, **kwargs):
    from ... import distributions
    condition, distribution, _, _ = distributions.load(cls.Base, *profiles, root=root, library=library)
    return cls(distribution.parameters, condition.parameters, units=units, **kwargs)

  def __init__(self, priors, conditions, units: Sequence[int], **kwargs):
    self.units = units
    assert len(units) == self.layers - 1, f'{units} vs {self.layers} layers'

    self.base = self.Base.from_config(**kwargs)

    super().__init__(priors, conditions)

  def get_coupled_parameters(self, state, parameters) -> Mapping[str, jax.Array]:
    return {}

  def initial_state(self, controls, parameters):
    extension_initial_state = self.extension_initial_state(parameters, controls)
    base_parameters = self.base.internal_parameters(self.decode_base_parameters(
      self.base.Parameters(*(getattr(parameters, f) for f in self.base.Parameters._fields))
    ))

    base_initial_state = self.base.initial_state(controls, base_parameters)

    return self.State(
      **base_initial_state._asdict(),
      **extension_initial_state
    )

  def parameter_shapes(self):
    units = self.units
    n_out = len(self.coupled_states)

    n_first_hidden, *rest = (*units, n_out)

    return self.Parameters(
      **{f'w_0_state_{k}': (n_first_hidden, ) for k in self.coupled_states},
      b_0=(n_first_hidden, ),
      **{
          f'W_{i}': (n_in, n_out)
          for i, n_in, n_out in zip(range(1, self.layers), units, rest)
      },
      **{
        f'b_{i}': (n_out, )
        for i, n_out in zip(range(1, self.layers), rest)
      },
      **self.base.parameter_shapes()._asdict()
    )

  def extension_state_shapes(self):
    return {}

  def extension_algebraics(self, state, parameters):
    return ()

  def extension_dynamics(self, state, parameters, algebraics, base_derivatives):
    return {}

  def residual_network(self, state, parameters):
    hidden = sum(
      getattr(state, f)[..., None] * getattr(parameters, f'w_0_state_{f}')
      for f in self.coupled_states
    ) + parameters.b_0

    for i in range(1, self.layers):
      W, b = getattr(parameters, f'W_{i}'), getattr(parameters, f'b_{i}')
      hidden = jnp.squeeze(
        jax.nn.celu(hidden)[..., None, :] @ W,
        axis=-2
      ) + b

    return hidden

  def residual_rates(self, state, parameters):
    return source_sink(self.residual_network(state, parameters), self.coupled_states)

  def regularization(self, parameters):
    extension_parameters = [
      getattr(parameters, param)
      for param in self.extension_parameters()
    ]

    return sum(jnp.sum(jnp.square(p)) for p in extension_parameters)
