import jax
import jax.numpy as jnp

from ..mm import MechanisticModel, System
from .. import utils

__all__ = [
  'Sampler',
]

class Sampler:
  @classmethod
  def from_profile(
    cls, model: MechanisticModel, spec: str, *specs: str,
    root: str | None=None,
    warn: bool=True, skip_unknown: bool=False, library=None
  ):
    from .parse import load

    condition, distribution, noise, likelihood = load(
      model, spec, *specs, root=root,
      warn=warn, skip_unknown=skip_unknown, library=library
    )

    return cls(model, condition, distribution, noise, likelihood)

  def __init__(
    self, model: MechanisticModel, condition: System, distribution: System, noise: System, likelihood
  ):
    self.model = model
    self.condition = condition
    self.distribution = distribution
    self.noise = noise

    self.likelihood = likelihood

    self.shapes = self._get_shapes()
    self.encoded_shapes = self._get_encoded_shapes()
    self.scaled_shapes = self._get_scaled_shapes()

  @staticmethod
  def _get_shape(ds, cs, ms):
    return type(cs)(*(d.get_shape_encoded(c, shapes=m) for d, c, m in zip(ds, cs, ms)))

  def get_shapes(self):
    time_shapes = self.model.Observables(*[() for _ in self.model.Observables._fields])

    return System(
      controls=self._get_shape(self.distribution.controls, self.condition.controls, self.model.controls_shapes()),
      parameters=self._get_shape(self.distribution.parameters, self.condition.parameters, self.model.parameter_shapes()),
      timestamps=self._get_shape(self.distribution.timestamps, self.condition.timestamps, time_shapes),
    )

  def get_flat_shapes(self):
    from .. import utils
    shapes = self.get_shapes()

    return System(
      controls=utils.shape.flat_shape(shapes.controls),
      parameters=utils.shape.flat_shape(shapes.parameters),
      timestamps=utils.shape.flat_shape(shapes.timestamps, axes=()),
    )

  @staticmethod
  def sample_from(key: jax.Array, distribution, condition, shapes):
    clazz = type(condition)
    return clazz(*(
      d.sample(k, c, shape=shape)
      for k, d, c, shape in zip(
        jax.random.split(key, num=len(distribution)), distribution, condition, shapes
      )
    ))

  @staticmethod
  def _encode(distribution, value, condition):
    clazz = type(condition)
    return clazz(*(
      d.encode(x, c)
      for d, x, c in zip(distribution, value, condition)
    ))

  @staticmethod
  def _decode(distribution, encoded, condition):
    return jax.tree.map(
      lambda d, enc, c: d.decode(enc, c),
      distribution, encoded, condition
    )

  @staticmethod
  def _scale(distribution, value, condition):
    return jax.tree.map(
      lambda d, x, c: d.scale(x, c),
      distribution, value, condition
    )

  @staticmethod
  def _descale(distribution, scaled, condition):
    return jax.tree.map(
      lambda d, x, c: d.descale(x, c),
      distribution, scaled, condition
    )

  def sample_controls(self, key: jax.Array, batch: int | None, experiments: int | None):
    if self.distribution.controls is None:
      return None

    else:
      control_shapes = [
        (*utils.shape.concrete(batch, experiments), *model_shape)
        for model_shape in self.model.controls_shapes()
      ]
      return self.sample_from(key, self.distribution.controls, self.condition.controls, shapes=control_shapes)

  def sample_parameters(self, key: jax.Array, batch: int | None):
    if self.distribution.parameters is None:
      return None

    else:
      parameter_shapes = [
        (*utils.shape.concrete(batch, ), *model_shape)
        for model_shape in self.model.parameter_shapes()
      ]
      return self.sample_from(key, self.distribution.parameters, self.condition.parameters, shapes=parameter_shapes)

  def sample_timestamps(self, key: jax.Array, batch: int | None, experiments: int | None):
    if self.distribution.timestamps is None:
      return None

    else:
      timestamp_shapes = [
        utils.shape.concrete(batch, experiments)
        for _ in self.model.Observables._fields
      ]
      return self.sample_from(key, self.distribution.timestamps, self.condition.timestamps, shapes=timestamp_shapes)

  def sample(self, key: jax.Array, batch: int | None=None, experiments: int | None=None):
    key_parameters, key_controls, key_timestamps = jax.random.split(key, num=3)

    controls = self.sample_controls(key_controls, batch, experiments)
    parameters = self.sample_parameters(key_parameters, batch)
    timestamps = self.sample_timestamps(key_timestamps, batch, experiments)

    return System(controls, parameters, timestamps)

  def encode_controls(self, controls):
    return self._encode(self.distribution.controls, controls, self.condition.controls)

  def encode_parameters(self, parameters):
    return self._encode(self.distribution.parameters, parameters, self.condition.parameters)

  def encode_timestamps(self, timestamps):
    return self._encode(self.distribution.timestamps, timestamps, self.condition.timestamps)

  def encode(self, value: System):
    return System(
      controls=self.encode_controls(value.controls),
      parameters=self.encode_parameters(value.parameters),
      timestamps=self.encode_timestamps(value.timestamps),
    )

  def decode_controls(self, controls):
    return self._decode(self.distribution.controls, controls, self.condition.controls)

  def decode_parameters(self, parameters):
    return self._decode(self.distribution.parameters, parameters, self.condition.parameters)

  def decode_timestamps(self, timestamps):
    return self._decode(self.distribution.timestamps, timestamps, self.condition.timestamps)

  def decode(self, encoded: System):
    return System(
      controls=self.decode_controls(encoded.controls),
      parameters=self.decode_parameters(encoded.parameters),
      timestamps=self.decode_timestamps(encoded.timestamps),
    )

  def scale_controls(self, controls):
    return self._scale(self.distribution.controls, controls, self.condition.controls)

  def scale_parameters(self, parameters):
    return self._scale(self.distribution.parameters, parameters, self.condition.parameters)

  def scale_timestamps(self, timestamps):
    return self._scale(self.distribution.timestamps, timestamps, self.condition.timestamps)

  def descale_controls(self, controls):
    return self._descale(self.distribution.controls, controls, self.condition.controls)

  def descale_parameters(self, parameters):
    return self._descale(self.distribution.parameters, parameters, self.condition.parameters)

  def descale_timestamps(self, timestamps):
    return self._descale(self.distribution.timestamps, timestamps, self.condition.timestamps)

  def scale(self, value: System):
    return System(
      controls=self.scale_controls(value.controls),
      parameters=self.scale_parameters(value.parameters),
      timestamps=self.scale_timestamps(value.timestamps),
    )

  def perturb_controls(self, key: jax.Array, controls):
    return self.sample_from(key, self.noise.controls, controls, shapes=[() for _ in controls])

  def perturb_parameters(self, key: jax.Array, parameters):
    return self.sample_from(key, self.noise.parameters, parameters, shapes=[() for _ in parameters])

  def perturb_timestamps(self, key: jax.Array, timestamps):
    return self.sample_from(key, self.noise.timestamps, timestamps, shapes=[() for _ in timestamps])

  def perturb(self, key: jax.Array, value: System):
    key_parameters, key_controls, key_timestamps = jax.random.split(key, num=3)

    controls = value.controls if self.noise.controls is None else self.perturb_controls(key_controls, value.controls)
    parameters = value.parameters if self.noise.parameters is None else self.perturb_parameters(key_parameters, value.parameters)
    timestamps = value.timestamps if self.noise.timestamps is None else self.perturb_timestamps(key_timestamps, value.timestamps)

    return System(controls, parameters, timestamps)

  ### likelihood

  def log_p(self, observations, predicted, batch_axes=(0,)):
    return utils.math.tree_sum(
      jax.tree.map(
        lambda p, m, pred: p.log_p(m, pred, axes=batch_axes),
        self.likelihood,
        observations,
        predicted
      )
    )

  def loss(self, observations, predicted, batch_axes=(0,)):
    return sum(
      getattr(self.likelihood, f).loss(getattr(observations, f), getattr(predicted, f), axes=batch_axes)
      for f in self.likelihood._fields
    )

  def sample_observations(self, key: jax.Array, observables):
    return self.model.Observables(*(
      l.sample(k, obs)
      for k, l, obs in zip(
        jax.random.split(key, num=len(observables)),
        self.likelihood,
        observables
      )
    ))

  def _get_shapes(self):
    key = jax.eval_shape(jax.random.PRNGKey, 0)
    shapes = jax.eval_shape(lambda k: self.sample(k, 1, 1), key)
    drop_batch = lambda n: lambda x: jax.ShapeDtypeStruct(shape=x.shape[n:], dtype=x.dtype)

    return System(
      controls=jax.tree.map(drop_batch(2), shapes.controls),
      parameters=jax.tree.map(drop_batch(1), shapes.parameters),
      timestamps=jax.tree.map(drop_batch(2), shapes.timestamps),
    )

  def _get_encoded_shapes(self):
    key = jax.eval_shape(jax.random.PRNGKey, 0)
    shapes = jax.eval_shape(lambda k: self.encode(self.sample(k, 1, 1)), key)

    drop_batch = lambda n: lambda x: jax.ShapeDtypeStruct(shape=x.shape[n:], dtype=x.dtype)

    return System(
      controls=jax.tree.map(drop_batch(2), shapes.controls),
      parameters=jax.tree.map(drop_batch(1), shapes.parameters),
      timestamps=jax.tree.map(drop_batch(2), shapes.timestamps),
    )

  def _get_scaled_shapes(self):
    key = jax.eval_shape(jax.random.PRNGKey, 0)
    shapes = jax.eval_shape(lambda k: self.scale(self.sample(k, 1, 1)), key)

    drop_batch = lambda n: lambda x: jax.ShapeDtypeStruct(shape=x.shape[n:], dtype=x.dtype)

    return System(
      controls=jax.tree.map(drop_batch(2), shapes.controls),
      parameters=jax.tree.map(drop_batch(1), shapes.parameters),
      timestamps=jax.tree.map(drop_batch(2), shapes.timestamps),
    )
