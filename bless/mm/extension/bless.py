from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp

from .common import HybridExtension
from .residual import source_sink

__all__ = [
  'LinBLESS', 'BLESS',
  'SequentialBLESS', 'ResidualBLESS',
]

class LinBLESS(HybridExtension):
  @classmethod
  def extension_states(cls) -> Sequence[str]:
    return ('z', )

  @classmethod
  def extension_parameters(cls, ) -> Sequence[str]:
    return (
      'L_z_z',
      'b_z',
      'B_z_r',
      'A_z_p',
      'z_initial',
      *[f'w_state_{k}_z' for k in cls.coupled_states],

      *[f'a_state_{k}_p' for k in cls.coupled_states],
      *[f'b_state_{k}_r' for k in cls.coupled_states],
    )

  @classmethod
  def extension_internal_parameters(cls, ) -> Sequence[str]:
    return (
      'A_z_z',
      'B_z_r',
      'A_z_p',
      'b_z',
      'z_initial',
      *[f'w_state_{k}_z' for k in cls.coupled_states],

      *[f'a_state_{k}_p' for k in cls.coupled_states],
      *[f'b_state_{k}_r' for k in cls.coupled_states],
    )

  def internal_parameters(self, parameters):
    ### A_z_z: the strictly lower triangle of L_z_z, its diagonal (the eigenvalues) -min_eigen - softplus(diag L_z_z)
    L = parameters.L_z_z
    *_, n = L.shape
    diagonal = -self.min_eigen - jax.nn.softplus(jnp.diagonal(L, axis1=-2, axis2=-1))
    A_z_z = jnp.tril(L, k=-1) + diagonal[..., None, :] * jnp.eye(n)

    return self.InternalParameters(
      A_z_z=A_z_z,
      **{k: getattr(parameters, k) for k in parameters._fields if k not in ('L_z_z', )}
    )

  @classmethod
  def from_config(cls, profiles, latent: int, min_eigen: float, root=None, library=None, **kwargs):
    from ... import distributions
    condition, distribution, _, _ = distributions.load(cls.Base, *profiles, root=root, library=library)
    return cls(distribution.parameters, condition.parameters, latent=latent, min_eigen=min_eigen, **kwargs)

  def __init__(self, priors, conditions, latent: int, min_eigen: float, **kwargs):
    self.latent = latent
    self.min_eigen = min_eigen
    self.base = self.Base.from_config(**kwargs)

    super().__init__(priors, conditions)

  def get_coupled_parameters(self, state, parameters) -> Mapping[str, jax.Array]:
    z = state.z
    A_z_p = parameters.A_z_p

    delta = jnp.squeeze(z[..., None, :] @ A_z_p, axis=-2) + sum(
      getattr(state, field)[..., None] * getattr(parameters, f'a_state_{field}_p')
      for field in self.coupled_states
    )

    return {k: getattr(parameters, k) + delta[..., i] for i, k in enumerate(self.coupled_parameters)}

  def parameter_shapes(self):
    n_z = self.latent
    n_p = len(self.coupled_parameters)
    n_r = len(self.coupled_states)

    return self.Parameters(
      L_z_z=(n_z, n_z), A_z_p=(n_z, n_p), B_z_r=(n_z, n_r), b_z=(n_z, ),
      **{f'w_state_{k}_z': (n_z,) for k in self.coupled_states},

      **{f'a_state_{k}_p': (n_p,) for k in self.coupled_states},
      **{f'b_state_{k}_r': (n_r,) for k in self.coupled_states},

      # **{f'w_d{k}_z': (n_z,) for k in self.coupled_states},
      z_initial=(n_z, ),
      **self.base.parameter_shapes()._asdict()
    )

  def extension_state_shapes(self):
    return {
      'z': (self.latent, )
    }

  def extension_algebraics(self, state, parameters):
    return ()

  def residual_rates(self, state, parameters):
    z = state.z

    r = jnp.squeeze(z[..., None, :] @ parameters.B_z_r, axis=-2) + sum(
      getattr(state, field)[..., None] * getattr(parameters, f'b_state_{field}_r')
      for field in self.coupled_states
    )
    return source_sink(r, self.coupled_states)

  def extension_dynamics(self, state, parameters, algebraics, base_derivatives):
    ### batch, experiments, latent
    z = state.z

    ### batch, 1, latent, latent
    A = parameters.A_z_z
    b = parameters.b_z

    z_z = jnp.squeeze(z[..., None, :] @ A, axis=-2)
    z_coupled = sum(
      getattr(state, field)[..., None] * getattr(parameters, f'w_state_{field}_z')
      for field in self.coupled_states
    )

    dz_dt = z_z + z_coupled + b

    return {'z': dz_dt}

  def regularization(self, parameters):
    extension_parameters = [
      getattr(parameters, param)
      for param in self.extension_parameters()
    ]

    return sum(jnp.sum(jnp.square(p)) for p in extension_parameters)

class BLESS(HybridExtension):
  """
  A latent state `z` driven by a network of `z` and the coupled states, read by two heads: the parameter head `p`
  corrects the coupled parameters (the Sequential component), the residual head `r` adds residual rates to the coupled
  states (the Residual component). `SequentialBLESS` and `ResidualBLESS` keep one head each.
  """
  layers: int = 2

  @classmethod
  def extension_states(cls) -> Sequence[str]:
    return ('z', )

  @classmethod
  def latent_parameters(cls) -> Sequence[str]:
    return (
      'z_initial',
      *[f'wz_0_state_{k}' for k in cls.coupled_states],
      *[f'bz_{i}' for i in range(cls.layers)],
      *[f'Wz_{i}' for i in range(cls.layers)],
    )

  @classmethod
  def head_parameters(cls, name: str) -> Sequence[str]:
    return (
      *[f'w{name}_0_state_{k}' for k in cls.coupled_states],
      *[f'b{name}_{i}' for i in range(cls.layers)],
      *[f'W{name}_{i}' for i in range(cls.layers)],
    )

  @classmethod
  def extension_parameters(cls, ) -> Sequence[str]:
    return (*cls.latent_parameters(), *cls.head_parameters('p'), *cls.head_parameters('r'))

  @classmethod
  def extension_internal_parameters(cls, ) -> Sequence[str]:
    return cls.extension_parameters()

  @classmethod
  def from_config(cls, profiles, units: Sequence[int], root=None, library=None, **kwargs):
    from ... import distributions
    condition, distribution, _, _ = distributions.load(cls.Base, *profiles, root=root, library=library)
    return cls(distribution.parameters, condition.parameters, units=units, **kwargs)

  def __init__(self, priors, conditions, latent: int, units: Sequence[int], **kwargs):
    self.latent = latent
    self.units = units

    self.base = self.Base.from_config(**kwargs)

    super().__init__(priors, conditions)

  def head(self, state, parameters, name: str):
    """Head `name` read from the latent state and the coupled states."""
    hidden = state.z @ getattr(parameters, f'W{name}_0') + sum(
      getattr(state, f)[..., None] * getattr(parameters, f'w{name}_0_state_{f}')
      for f in self.coupled_states
    ) + getattr(parameters, f'b{name}_0')

    for i in range(1, self.layers):
      W, b = getattr(parameters, f'W{name}_{i}'), getattr(parameters, f'b{name}_{i}')
      hidden = jax.nn.celu(hidden) @ W + b

    return hidden

  def get_coupled_parameters(self, state, parameters) -> Mapping[str, jax.Array]:
    delta = self.head(state, parameters, 'p')
    return {k: getattr(parameters, k) + delta[..., i] for i, k in enumerate(self.coupled_parameters)}

  def residual_rates(self, state, parameters):
    return source_sink(self.head(state, parameters, 'r'), self.coupled_states)

  def latent_shapes(self):
    n_z = self.latent
    n_first_hidden, *rest = (*self.units, n_z)
    return {
      'z_initial': (n_z, ),
      **{f'wz_0_state_{k}': (n_first_hidden, ) for k in self.coupled_states},
      'Wz_0': (n_z, n_first_hidden), 'bz_0': (n_first_hidden, ),
      **{f'Wz_{i}': (n_in, n_out) for i, n_in, n_out in zip(range(1, self.layers), self.units, rest)},
      **{f'bz_{i}': (n_out, ) for i, n_out in zip(range(1, self.layers), rest)},
    }

  def head_shapes(self, name: str, n_out: int):
    n_first_hidden, *rest = (*self.units, n_out)
    return {
      **{f'w{name}_0_state_{k}': (n_first_hidden, ) for k in self.coupled_states},
      f'W{name}_0': (self.latent, n_first_hidden), f'b{name}_0': (n_first_hidden, ),
      **{f'W{name}_{i}': (n_in, n_out) for i, n_in, n_out in zip(range(1, self.layers), self.units, rest)},
      **{f'b{name}_{i}': (n_out, ) for i, n_out in zip(range(1, self.layers), rest)},
    }

  def parameter_shapes(self):
    return self.Parameters(
      **self.latent_shapes(),
      **self.head_shapes('p', len(self.coupled_parameters)),
      **self.head_shapes('r', len(self.coupled_states)),
      **self.base.parameter_shapes()._asdict(),
    )

  def extension_state_shapes(self):
    return {'z': (self.latent, )}

  def extension_algebraics(self, state, parameters):
    return ()

  def latent_network(self, state, parameters):
    """
    The latent network `f(z, coupled states)`, the drive of the latent dynamics.
    """
    z = state.z
    W_0 = parameters.Wz_0
    b_0 = parameters.bz_0
    hidden_z = z @ W_0

    hidden = sum(
      getattr(state, f)[..., None] * getattr(parameters, f'wz_0_state_{f}')
      for f in self.coupled_states
    ) + hidden_z + b_0

    for i in range(1, self.layers):
      W, b = getattr(parameters, f'Wz_{i}'), getattr(parameters, f'bz_{i}')
      hidden = jax.nn.celu(hidden) @ W + b

    return hidden

  def extension_dynamics(self, state, parameters, algebraics, base_derivatives):
    z = state.z
    dz_dt = self.latent_network(state, parameters) - z * jnp.abs(z)

    #z_reg = z * jnp.exp(jnp.square(z))

    return {'z': dz_dt}

  def regularization(self, parameters):
    extension_parameters = [
      getattr(parameters, param)
      for param in self.extension_parameters()
    ]

    return sum(jnp.sum(jnp.square(p)) for p in extension_parameters)

class SequentialBLESS(BLESS):
  """
  BLESS with the parameter head alone: the latent state and the coupled states correct the coupled parameters;
  no residual rates.
  """
  @classmethod
  def extension_parameters(cls, ) -> Sequence[str]:
    return (*cls.latent_parameters(), *cls.head_parameters('p'))

  def residual_rates(self, state, parameters):
    return {}

  def parameter_shapes(self):
    return self.Parameters(
      **self.latent_shapes(),
      **self.head_shapes('p', len(self.coupled_parameters)),
      **self.base.parameter_shapes()._asdict(),
    )

class ResidualBLESS(BLESS):
  """
  BLESS with the residual head alone: the latent state and the coupled states drive residual rates of the coupled
  states; the coupled parameters stay the base's.
  """
  @classmethod
  def extension_parameters(cls, ) -> Sequence[str]:
    return (*cls.latent_parameters(), *cls.head_parameters('r'))

  def get_coupled_parameters(self, state, parameters) -> Mapping[str, jax.Array]:
    return {k: getattr(parameters, k) for k in self.coupled_parameters}

  def parameter_shapes(self):
    return self.Parameters(
      **self.latent_shapes(),
      **self.head_shapes('r', len(self.coupled_states)),
      **self.base.parameter_shapes()._asdict(),
    )
