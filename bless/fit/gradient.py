from collections.abc import Sequence
from typing import NamedTuple
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

import optax

from ..mm.solver import Solver
from ..distributions import Sampler
from .. import utils

default_optimizer = optax.adamaxw(learning_rate=1.0e-3, b1=0.1, b2=0.99, weight_decay=1.0e-4)

__all__ = [
  'MaximumLikelihood', 'Restarts'
]

class Restarts(NamedTuple):
  """
  Every multistart restart of a fit, not only the selected one: `losses` (multistart, batch, steps), decoded
  `parameters` with a leading multistart axis, and the index of the `selected` restart per batch element.
  """
  losses: np.ndarray
  parameters: tuple
  selected: np.ndarray

class MaximumLikelihood:
  def __init__(
      self, solver: Solver, sampler: Sampler, optimizer=default_optimizer,
      n_steps: int=8, n_substeps: int=128, n_multistart: int=1,
      grad_clip: float | None=None, c_reg: float | None=None, check_nans: bool=True,
      reg_states: Sequence[str]=('z', ), c_reg_states: float | None=None
  ):
    @jax.jit
    def solve(controls, parameters, timestamps):
      decoded = sampler.decode_parameters(parameters)

      states, algebraics = solver.solve_algebraics(controls, decoded)
      observables = solver.model.observe(states, decoded, algebraics)

      if timestamps is None:
        return states, observables
      else:
        observations = solver.select(observables, timestamps)
        return states, observations

    @jax.jit
    def solve_algebraics(controls, parameters):
      decoded = sampler.decode_parameters(parameters)
      return solver.solve_algebraics(controls, decoded)

    @jax.jit
    def regularization(parameters):
      reg_f = getattr(self.solver.model, 'regularization', None)

      if reg_f is not None:
        return reg_f(parameters)
      else:
        return sum(jnp.mean(jnp.sum(jnp.square(x), axis=range(1, x.ndim))) for x in jax.tree.leaves(parameters))

    @jax.jit
    def regularization_states(states):
      return sum(
        jnp.mean(jnp.square(getattr(states, f)))
        for f in reg_states
      )

    @jax.jit
    def loss_fn(observations, controls, parameters, timestamps):
      states, predicted = solve(controls, parameters, timestamps)
      nll = sampler.loss(observations, predicted, batch_axes=())
      ls = nll

      if c_reg is not None:
        ls = ls + c_reg * regularization(parameters)

      if len(reg_states) > 0 and c_reg_states is not None:
        ls = ls + c_reg_states * regularization_states(states, )

      return ls, nll

    @jax.jit
    def grad_loss(observations, controls, parameters, timestamps):
      """
      Gradient of every parameter set (outer: multistart, middle: batch) with respect to the sum of its
      experiments' losses; returns the gradient and the loss per experiment.
      The experiments `vmap` sits outside `grad`; the per-experiment gradients are summed.
      """
      grid = isinstance(timestamps, jax.Array)

      def summed_over_experiments(obs, cs, ps, ts):
        grads, nll = jax.vmap(
          jax.grad(loss_fn, argnums=2, has_aux=True),
          in_axes=(0, 0, None, None if grid else 0), out_axes=0
        )(obs, cs, ps, ts)
        return jax.tree.map(lambda g: jnp.sum(g, axis=0), grads), nll

      grad, loss = jax.vmap(
        jax.vmap(summed_over_experiments, in_axes=(0, 0, 0, None if grid else 0), out_axes=0),
        in_axes=(None, None, 0, None), out_axes=0
      )(observations, controls, parameters, timestamps)

      return grad, loss

    self.grad_loss = grad_loss

    @jax.jit
    def step(observations, controls, parameters, timestamps, state):
      ### outer: multistart
      ### middle: batch
      ### inner: experiments
      grad, ls = grad_loss(observations, controls, parameters, timestamps)

      print('Parameters:', jax.tree.map(jnp.shape, parameters))
      print('Grad:', jax.tree.map(jnp.shape, grad))

      if grad_clip is not None:
        grad = jax.tree.map(lambda x: jnp.clip(x, -grad_clip, grad_clip), grad)

      updates, state_updated = optimizer.update(grad, state, params=parameters)
      parameters_updated = optax.apply_updates(parameters, updates)

      return ls, parameters_updated, state_updated

    @jax.jit
    def optimize(observations, controls, parameters, timestamps, state):
      def scan_step(carry, _):
        enc_ps, s = carry
        ls, upd_enc_ps, upd_s = step(observations, controls, enc_ps, timestamps, s)

        return (upd_enc_ps, upd_s), ls

      (final_ps, final_state), losses = jax.lax.scan(
        scan_step,
        init=(parameters, state),
        xs=None, length=n_substeps
      )

      return jnp.mean(losses, axis=0), final_ps, final_state

    @partial(jax.jit, static_argnames=('n_b', ))
    def sample(key, n_b):
      parameters = self.sampler.sample_parameters(key, batch=n_b)
      return self.sampler.encode_parameters(parameters)

    @partial(jax.jit, static_argnames=('n_b', 'n_multi'))
    def sample_multi(key, n_b, n_multi):
      return jax.vmap(lambda k: sample(k, n_b), in_axes=(0, ), out_axes=0)(
        jax.random.split(key, num=n_multi)
      )

    init_multi = jax.vmap(optimizer.init, in_axes=(0, ), out_axes=0)

    self.solve = solve
    self.solve_algebraics = solve_algebraics

    self.regularization = regularization
    self.loss_fn = loss_fn
    self.step = step

    self.optimize = optimize
    self.sample = sample
    self.sample_multi = sample_multi

    self.init_multi = init_multi

    self.solver = solver
    self.sampler = sampler
    self.optimizer = optimizer

    self.n_steps = n_steps
    self.n_substeps = n_substeps
    self.n_multistart = n_multistart
    self.grad_clip = grad_clip
    self.check_nans = check_nans
    self.c_reg = c_reg

  def initial(self, key, n_batch: int):
    """
    The encoded initial parameters `_fit` draws from `key`, multistart on axis 0 and batch on axis 1.
    """
    key, key_initial = jax.random.split(key, num=2)
    return self.sample_multi(key_initial, n_batch, self.n_multistart)

  def _fit(self, key, controls, observations, timestamps, progress=True, initial=None):
    from tqdm import tqdm
    (n_batch, _), = { obs.shape[:2] for obs in observations }

    parameters_encoded = self.initial(key, n_batch) if initial is None else initial
    state = self.optimizer.init(parameters_encoded)

    if progress is True:
      pb = tqdm
    elif progress is False:
      pb = lambda x, *args, **kwargs: utils.progress.SimpleProgress(x, *args, **kwargs)
    else:
      pb = progress

    losses = np.ndarray(shape=(self.n_multistart, n_batch, self.n_steps), dtype=np.float32)

    for i in pb(range(self.n_steps), leave=False):
      ls, parameters_encoded, state = self.optimize(
        observations, controls, parameters_encoded, timestamps, state
      )
      ### `optimize` returns the loss per experiment; the objective is their sum
      losses[:, :, i] = jnp.sum(ls, axis=-1)

    ### the best restart among those with a finite final loss
    final = np.where(np.isfinite(losses[:, :, -1]), losses[:, :, -1], np.inf)
    best = np.argmin(final, axis=0)

    restarts = Restarts(
      losses=losses,
      parameters=jax.vmap(self.sampler.decode_parameters)(parameters_encoded),
      selected=best,
    )

    index = np.arange(n_batch)
    parameters_encoded = jax.tree.map(
      lambda x: x[best, index],
      parameters_encoded
    )
    losses = losses[best, index]

    decoded = self.sampler.decode_parameters(parameters_encoded)

    return losses, decoded, restarts

  def fit(self, key, controls, observations, timestamps, progress=True, initial=None):
    """
    `initial`: encoded initial parameters as `initial(key, n_batch)` gives them; drawn from `key` when None.
    """
    return self._fit(key, controls, observations, timestamps, progress=progress, initial=initial)
