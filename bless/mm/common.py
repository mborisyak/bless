from collections import namedtuple
from collections.abc import Sequence

from typing import Protocol, ClassVar

__all__ = [
  'MechanisticModel',
  'NamedTuple',

  'System'
]


class NamedTuple(Protocol):
  _fields: ClassVar[Sequence[str]]

  def __init__(self, *args, **kwargs): ...
  def __getitem__(self, *args, **kwargs): ...

  def __len__(self) -> int: ...
  def __iter__(self): ...

System = namedtuple('System', ['controls', 'parameters', 'timestamps'])

class MechanisticModel:
  """
  An interface for mechanistic models.

  This class is intended to be used as a collection of methods, thus, an instance of such a class should only hold
  hyperparameters, i.e., model should be stateless: state, pulses, controls and parameters are to be managed externally.

  The model is meant to be agnostic of ODE solvers, thus, provides all methods necessary for simulation.
  """

  ### types below are intended to be namedtuple subclasses

  ### parameters of the model
  Parameters: type[NamedTuple]

  ### parameters of the model
  InternalParameters: type[NamedTuple]

  ### state on which model is defined
  ### might contain latent states
  ### to make RHS time-dependent, one can introduce time as a hidden state
  State: type[NamedTuple]

  ### internal constants
  Algebraics: type[tuple]

  ### returns variables that are observed
  ### does not have to be necessarily the subset of states
  ### might also include combinations of states like X * Y
  Observables: type[NamedTuple]

  ### pulse based controls
  Controls: type[NamedTuple]

  ### continuous controls
  ControlParameters: type[NamedTuple]

  @classmethod
  def from_config(cls, *args, **kwargs):
    return cls(*args, **kwargs)

  def parameter_shapes(self):
    return self.Parameters(*(
      () for _ in self.Parameters._fields
    ))

  def parameter_size(self) -> int:
    return len(self.Parameters._fields)

  def state_shapes(self):
    return self.State(*(
      () for _ in self.State._fields
    ))

  def state_size(self) -> int:
    return len(self.State._fields)

  def observables_shapes(self):
    return self.Observables(*(
      () for _ in self.Observables._fields
    ))

  def observables_size(self):
    return len(self.Observables._fields)

  def controls_shapes(self):
    return self.ControlParameters(*(
      () for _ in self.ControlParameters._fields
    ))

  def internal_parameters(self, parameters: 'Parameters') -> 'InternalParameters':
    return self.InternalParameters(*parameters)

  def initial_state(self, controls: 'ControlParameters', parameters: 'InternalParameters') -> 'State':
    """
    Converts an instance of InitialState into initial State.

    :param controls: conditions of the experiment;
    :param parameters: parameters of the system;
    :return: initial state.
    """
    raise NotImplementedError()

  def observe(self, state, parameters, algebraics):
    """
    Converts internal state into observables.

    :param state: state of the model;
    :param parameters: model parameters;
    :param algebraics: intermediate values return by algebraics
    :return: observables.
    """
    raise NotImplementedError()

  def controls(self, timestamps, control_parameters):
    """
    Expands controls into the timegrid.

    The resulting structure must have arrays with size: `(n_t, *)` where `n_t` is the size of `timestamps`.

    :param timestamps: timegrid, an array of shape `(n_t, )`
    :param control_parameters: parameters of the controls, arbitrary shape,
    :return: expanded controls.
    """
    raise NotImplementedError()

  def algebraics(self, state, parameters, controls):
    """
    Returns intermediate values for use in methods `derivatives_algebraics` and `analytical_step`.

    :param state: current state of the system;
    :param parameters: parameters of the system;
    :param controls: controls;
    :return: user-defined values for later use.
    """
    return ()

  def analytical_step(self, dt: float, state, parameters, controls, algebraics):
    """
    Returns an (approximate) analytically integrated solution when available:
        S_0 + int^{t + dt}_{t} (d S / d t) dt,
    where S_0 --- initial state.

    This method allows to override solver's step if a better analytical approximation exists.
    This is an optional method, if no approximation is available for a channel then None should be returned.
    In this case, solver performs the step according to the integration method.

    :param dt: size of the step;
    :param state: current state of the system;
    :param parameters: (internal) parameters of the system;
    :param controls: controls;
    :param algebraics: intermediate values returned by `algebraics` method.
    :return: (approximate) analytical solution.
    """
    return self.State(*(None for _ in self.State._fields))

  def derivatives(self, state, parameters, controls):
    """
    Returns d S / d t where S --- state, t --- time.

    :param state: current model state;
    :param parameters: model parameters;
    :param controls: controls;
    :return: the same structure as State holding derivatives.
    """
    algebraics = self.algebraics(state, parameters, controls)
    return self.derivatives_algebraics(state, parameters, controls, algebraics)

  def derivatives_algebraics(self, state, parameters, controls, algebraics):
    """
    Returns d S / d t where S --- state, t --- time.

    This method should be overridden in the models.

    :param state: current model state;
    :param parameters: model parameters;
    :param algebraics: values returned by algebraics method;
    :param controls: controls;
    :return: the same structure as State holding derivatives.
    """
    ### one of the methods, derivatives or derivatives_algebraics, must be overriden
    return self.derivatives(state, parameters, controls)

  def pulse(self, state, parameters, controls, algebraics):
    """
    Performs instantaneous state change, e.g., pulse feeding.
    """
    return self.State(*(None for _ in self.State._fields))

  def clip(self, state, parameters, algebraics):
    """
    Applies bounds to the state. For example, some of the state variables might have a physical meaning only when
    they are positive, however, due to errors might become negative during integration. In such a case, `clip` function
    might set these variables to 0.

    This function is intended to be applied after the integration step to remove possible numerical artefacts.

    :param state: current state;
    :parameter parameters: parameters of the system;
    :parameter algebraics: intermediate model-defined values returned by `algebraics` method;
    :return: clipped state.
    """
    return state
