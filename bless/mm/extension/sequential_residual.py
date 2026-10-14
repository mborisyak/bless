from typing import Mapping, Sequence

import jax
import jax.numpy as jnp

from .common import HybridExtension
from .residual import source_sink

__all__ = [
  'SequentialResidual',
]

class SequentialResidual(HybridExtension):
  """
  BLESS without the latent state: a parameter head as `Sequential` and a residual head as `Residual`, both reading
  the coupled states only, one hidden layer of `units[0]` each.
  """
  layers: int = 2

  @classmethod
  def extension_states(cls) -> Sequence[str]:
    return ()

  @classmethod
  def extension_parameters(cls, ) -> Sequence[str]:
    return (
      *[f'wp_0_state_{k}' for k in cls.coupled_states],
      *[f'bp_{i}' for i in range(cls.layers)],
      *[f'Wp_{i}' for i in range(1, cls.layers)],

      *[f'wr_0_state_{k}' for k in cls.coupled_states],
      *[f'br_{i}' for i in range(cls.layers)],
      *[f'Wr_{i}' for i in range(1, cls.layers)],
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

  def head(self, state, parameters, name: str):
    hidden = sum(
      getattr(state, f)[..., None] * getattr(parameters, f'w{name}_0_state_{f}')
      for f in self.coupled_states
    ) + getattr(parameters, f'b{name}_0')

    for i in range(1, self.layers):
      W, b = getattr(parameters, f'W{name}_{i}'), getattr(parameters, f'b{name}_{i}')
      hidden = jnp.squeeze(
        jax.nn.celu(hidden)[..., None, :] @ W,
        axis=-2
      ) + b

    return hidden

  def get_coupled_parameters(self, state, parameters) -> Mapping[str, jax.Array]:
    delta = self.head(state, parameters, 'p')
    return {k: getattr(parameters, k) + delta[..., i] for i, k in enumerate(self.coupled_parameters)}

  def residual_rates(self, state, parameters):
    return source_sink(self.head(state, parameters, 'r'), self.coupled_states)

  def initial_state(self, controls, parameters):
    base_parameters = self.base.internal_parameters(self.decode_base_parameters(
      self.base.Parameters(*(getattr(parameters, f) for f in self.base.Parameters._fields))
    ))
    base_initial_state = self.base.initial_state(controls, base_parameters)
    return self.State(**base_initial_state._asdict(),)

  def parameter_shapes(self):
    units = self.units

    def shapes(name, n_out):
      n_first_hidden, *rest = (*units, n_out)
      return {
        **{f'w{name}_0_state_{k}': (n_first_hidden, ) for k in self.coupled_states},
        f'b{name}_0': (n_first_hidden, ),
        **{f'W{name}_{i}': (n_in, n_out) for i, n_in, n_out in zip(range(1, self.layers), units, rest)},
        **{f'b{name}_{i}': (n_out, ) for i, n_out in zip(range(1, self.layers), rest)},
      }

    return self.Parameters(
      **shapes('p', len(self.coupled_parameters)),
      **shapes('r', len(self.coupled_states)),
      **self.base.parameter_shapes()._asdict()
    )

  def extension_state_shapes(self):
    return {}

  def extension_algebraics(self, state, parameters):
    return ()

  def extension_dynamics(self, state, parameters, algebraics, base_derivatives):
    return {}

  def regularization(self, parameters):
    extension_parameters = [
      getattr(parameters, param)
      for param in self.extension_parameters()
    ]

    return sum(jnp.sum(jnp.square(p)) for p in extension_parameters)
