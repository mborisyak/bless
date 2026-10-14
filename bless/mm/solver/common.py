import jax
import jax.numpy as jnp

from ..common import MechanisticModel

__all__ = [
  'select',
  'Solver',
]

def select(dt, grid, states, timestamps):
  """
  Linearly interpolates a trajectory with time on axis 0 onto `timestamps`.

  `timestamps` is either a bare array, applied to every channel, or a structure matching `states`
  with one array per channel.
  """
  def _select(xs, ts):
    tau = ts / dt - 1
    index = jnp.floor(tau).astype(jnp.int32)

    left = jnp.clip(index, 0, grid.shape[0] - 1)
    right = jnp.clip(index + 1, 0, grid.shape[0] - 1)
    r = tau - index

    left_f = xs[left]
    right_f = xs[right]

    return (1 - r) * left_f + r * right_f

  if isinstance(timestamps, jax.Array):
    return type(states)(*(_select(x, timestamps) for x in states))
  else:
    return type(states)(**{
      f: _select(getattr(states, f), getattr(timestamps, f))
      for f in states._fields
    })

class Solver:
  """
  Integrates a `MechanisticModel` over a fixed time grid.

  Every method computes ONE trajectory: `control_parameters` and `parameters` carry no batch axes and
  the result puts time on axis 0. Subclasses implement `_scan`.

  The `*_batch` helpers vmap over `(batch, experiment)`: parameters vary along `batch` and are shared
  across experiments, control parameters vary along both.
  """
  model: MechanisticModel
  dt: float
  T: float
  grid: jax.Array
  ### recompute the step in the backward pass instead of storing its intermediates
  remat: bool = False
  ### steps per iteration of the scan's loop (`jax.lax.scan(..., unroll=)`)
  unroll: int = 1

  def _step(self, step):
    return jax.checkpoint(step) if self.remat else step

  def _scan(self, control_parameters, parameters, output_f, analytical: bool):
    """
    Runs the integration, calling `output_f(state, parameters, controls, algebraics, derivatives)` after
    every step and stacking its results along axis 0.
    """
    raise NotImplementedError()

  def solve(self, control_parameters, parameters, analytical: bool=True):
    return self._scan(
      control_parameters, parameters,
      lambda state, params, controls, algebraics, derivatives: state,
      analytical=analytical
    )

  def observables(self, control_parameters, parameters):
    return self._scan(
      control_parameters, parameters,
      lambda state, params, controls, algebraics, derivatives: self.model.observe(state, params, algebraics),
      analytical=True
    )

  def observe(self, control_parameters, parameters, timestamps):
    return self.select(self.observables(control_parameters, parameters), timestamps)

  def solve_algebraics(self, control_parameters, parameters):
    return self._scan(
      control_parameters, parameters,
      lambda state, params, controls, algebraics, derivatives: (state, algebraics),
      analytical=True
    )

  def solve_derivatives(self, control_parameters, parameters):
    return self._scan(
      control_parameters, parameters,
      lambda state, params, controls, algebraics, derivatives: (state, derivatives),
      analytical=True
    )

  def select(self, states, timestamps):
    return select(self.dt, self.grid, states, timestamps)

  ### batched over (batch, experiment)

  @staticmethod
  def _over_batch(f, *in_axes):
    return jax.vmap(jax.vmap(f, in_axes=in_axes, out_axes=0), in_axes=tuple(0 for _ in in_axes), out_axes=0)

  def solve_batch(self, control_parameters, parameters, analytical: bool=True):
    return jax.vmap(
      jax.vmap(lambda cs, ps: self.solve(cs, ps, analytical=analytical), in_axes=(0, None), out_axes=0),
      in_axes=(0, 0), out_axes=0
    )(control_parameters, parameters)

  def observables_batch(self, control_parameters, parameters):
    return jax.vmap(
      jax.vmap(self.observables, in_axes=(0, None), out_axes=0),
      in_axes=(0, 0), out_axes=0
    )(control_parameters, parameters)

  def observe_batch(self, control_parameters, parameters, timestamps):
    """
    `observe` over `(batch, experiment)`. A bare one-dimensional `timestamps` is shared by every
    trajectory; otherwise timestamps vary along both axes like the control parameters.
    """
    shared = isinstance(timestamps, jax.Array) and timestamps.ndim == 1

    return jax.vmap(
      jax.vmap(self.observe, in_axes=(0, None, None if shared else 0), out_axes=0),
      in_axes=(0, 0, None if shared else 0), out_axes=0
    )(control_parameters, parameters, timestamps)

  def solve_algebraics_batch(self, control_parameters, parameters):
    return jax.vmap(
      jax.vmap(self.solve_algebraics, in_axes=(0, None), out_axes=0),
      in_axes=(0, 0), out_axes=0
    )(control_parameters, parameters)

  def solve_derivatives_batch(self, control_parameters, parameters):
    return jax.vmap(
      jax.vmap(self.solve_derivatives, in_axes=(0, None), out_axes=0),
      in_axes=(0, 0), out_axes=0
    )(control_parameters, parameters)
