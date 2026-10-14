from typing import Union, Type, TypeVar, Callable

import math

import re

from ..mm import MechanisticModel, NamedTuple

from .common import Distribution
from .conditional import ConditionalDistribution

import os, sys

__all__ = [
  'load_profile',

  'parse',
  'parse_with_condition',

  'combine', 'match',

  'load'
]

PathLike = bytes | str | os.PathLike[str]

Args = Union[
  list[Union[float, list[float]]],
  dict[str, Union[float, list[float]]]
]

Dist = TypeVar('Dist', bound=Distribution)
Condition = TypeVar('Condition')

### this file is <root>/extension/distributions/parse.py
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

MGM_DATA = 'mgm_data'

LIKELIHOOD = 'likelihood'

PARAMETERS = 'parameters'
CONTROLS = 'controls'
TIMESTAMPS = 'timestamps'

PARAMETER_NOISE = 'parameter-noise'
CONTROL_NOISE = 'control-noise'
TIMESTAMPS_NOISE = 'timestamps-noise'

def get_available(root, path: PathLike, ext='.yaml'):
  if root is None:
    return []

  directory, _ = os.path.split(path)
  target = os.path.join(root, directory)

  if not os.path.exists(target):
    return []

  if not os.path.isdir(target):
    return []

  return [
    item[:-len(ext)]
    for item in os.listdir(target)
    if item.endswith(ext)
  ]


def get_distribution_location(spec: str):
  *tokens, distribution = spec.split('.')
  return os.path.join(*tokens, f'{distribution}.yaml')

def load_profile(spec: str, *, root: str | None=None):
  import yaml

  path = get_distribution_location(spec)

  if root is not None:
    try:
      with open(os.path.join(root, path), 'r') as f:
        return root, yaml.load(f, Loader=yaml.Loader)
    except FileNotFoundError:
      raise FileNotFoundError(
        f"{spec}: "
        f"tried the following locations: {root}. "
        f"Available distributions for the model are: "
        f"{', '.join(get_available(root, path))}"
      )

  ### first, try relative to the current file:
  ### <root>/mgm/distributions/parse.py
  root_package = os.path.join(PACKAGE_ROOT, 'config', 'mm')
  try:
    full_path = os.path.join(root_package, path)
    actual_root = os.path.dirname(full_path)
    with open(full_path, 'r') as f:
      return actual_root, yaml.load(f, Loader=yaml.Loader)

  except FileNotFoundError:
    pass

  ### if package is installed and data is stored in the root of the current python env
  root_sys = os.path.join(sys.prefix, MGM_DATA, 'models')
  try:
    full_path = os.path.join(root_sys, path)
    actual_root = os.path.dirname(full_path)
    with open(full_path, 'r') as f:
      return actual_root, yaml.load(f, Loader=yaml.Loader)

  except FileNotFoundError:
    pass

  raise FileNotFoundError(
    f"{spec}: "
    f"tried the following locations: {root_package}, {root_sys}. "
    f"Available distributions for the model are: "
    f"{', '.join(get_available(root_package, path) + get_available(root_sys, path))}"
  )

T = TypeVar('T')
Tuple = TypeVar('Tuple', bound=NamedTuple)

def apply(
  description: dict[str, Args],
  library: dict[str, Callable[[Args, ...], T]],
) -> T:

  if len(description) > 1:
    raise ValueError(
      f'More than one distribution was specified for a single parameter: {description}!'
    )

  distribution_name, = description

  if distribution_name not in library:
    raise ValueError(f'Unknown distribution {distribution_name}')

  distribution = library[distribution_name]
  arguments = description[distribution_name]

  try:
    if isinstance(arguments, list):
      return distribution(*arguments)
    elif isinstance(arguments, dict):
      return distribution(**arguments)
    else:
      raise ValueError("Distribution's arguments must be either a list of arguments or a dictionary.")
  except TypeError as e:
    import inspect
    raise TypeError(
      f'arguments ({arguments}) don\'t match distribution signature: {inspect.signature(distribution)}'
    ) from e

def parse_with_condition(
  description: dict[str, dict[str, Args]],
  library: dict[str, Type[ConditionalDistribution]],
) -> tuple[dict[str, ConditionalDistribution], dict[str, Condition]]:
  parsing = {
    dist: library[dist].full_parse
    for dist in library
  }

  prior: dict[str, ConditionalDistribution] = dict()
  condition: dict[str, Condition] = dict()

  for name in description:
    prior[name], condition[name] = apply(description[name], parsing)

  return condition, prior

def parse(
  description: dict[str, dict[str, Args]],
  library: dict[str, Type[Dist]],
) -> dict[str, ConditionalDistribution]:
  parsing = {
    dist: library[dist].parse
    for dist in library
  }

  prior = dict()
  for name in description:
    prior[name] = apply(description[name], parsing)

  return prior

def combine(
  spec: str, *additional_specs : str,
  root: None | PathLike=None,
  warn: bool=True,
):
  """
  Loads and brings to the standard form parameters according to the provided specifications (`spec1` and `specs`).

  Prior can be specified in the following ways:
  - preferred, `<model name>.<prior name>`: for parameters supplied with the package;
  - path to the prior: for custom parameters.

  If several specifications are provided, parameters are merged.

  Different types of specifications can be combined, e.g.:
      combined_prior = parse_priors('ecoli2017.w3110', 'config/parameters/my_bioreactor.yaml')

  :param spec: the first prior specification;
  :param additional_specs: optional consequent profiles to combine with the first one;
  :param warn: controls if a warning should be issued when one prior overrides any keys from another;
  :param root: path to the configuration directory.
  :return: combined distribution.
  """
  specs = (spec, *additional_specs)
  combined_prior = dict()

  for spec in specs:
    _, description = load_profile(spec, root=root)

    if warn:
      for group in combined_prior:
        if group not in description:
          continue
        overrides = [p for p in description[group] if p in combined_prior[group]]

        if len(overrides) > 0:
          import warnings
          warnings.warn(
            f'{group} from the spec {spec} overrides {group} '
            f'from the previous specs: {", ".join(overrides)}'
          )

    for group in combined_prior:
      if group in description:
        combined_prior[group].update(description[group])

    for group in description:
      if group in combined_prior:
        combined_prior[group].update(description[group])
      else:
        combined_prior[group] = description[group]

  return combined_prior

def make_reg_exp(expr: str):
  star_expr = re.compile(r'([^*]+|[*]{1,2})')
  tokens = re.findall(star_expr, expr)
  reg_expr = list()

  for token in tokens:
    if token == '*':
      reg_expr.append(r'([A-Za-z0-9]*)')
    elif token == '**':
      reg_expr.append(r'([A-Za-z0-9_]*)')
    else:
      reg_expr.append(f'({token})')

  complexity = len(reg_expr)
  return re.compile(''.join(reg_expr)), complexity

def match(fields, definition: dict, skip_unknown: bool=False):
  regular_expressions = {
    expr: make_reg_exp(expr)
    for expr in definition
  }

  matching = {
    expr: set([k for k in fields if re.fullmatch(reg_exp, k)])
    for expr, (reg_exp, _) in regular_expressions.items()
  }

  most_specific_matching = dict()

  for field in fields:
    matching_expressions = [
      (expr, len(matching[expr]), regular_expressions[expr][1])
      for expr in matching
      if field in matching[expr]
    ]

    best_matches, minimal_specificity, minimal_complexity = list(), len(fields) + 1, math.inf
    for expr, specificity, complexity in matching_expressions:
      if specificity < minimal_specificity:
        best_matches = [expr]
        minimal_specificity = specificity
        minimal_complexity = complexity

      elif specificity == minimal_specificity and complexity < minimal_complexity:
        best_matches = [expr]
        minimal_specificity = specificity
        minimal_complexity = complexity

      elif specificity == minimal_specificity and complexity == minimal_complexity:
        best_matches.append(expr)

    if len(best_matches) > 1:
      raise ValueError(
        'Field {field} matches several patterns with the same specificity ({spec}, {comp}): {patterns}'.format(
          field=field,
          spec=minimal_specificity,
          comp=minimal_complexity,
          patterns=', '.join(best_matches)
        )
      )

    if len(best_matches) == 0:
      raise ValueError(
        'Field {field} does not match any patterns'.format(
          field=field,
        )
      )

    best_match, = best_matches
    most_specific_matching[field] = definition[best_match]

  if not skip_unknown:
    non_matching = [expr for expr in matching if len(matching[expr]) == 0]
    if len(non_matching) > 0:
      raise ValueError(
        'The following patterns do not match any fields: {patterns} ({fields}). '
        'To ignore this exception set skip_unknown=True.'.format(
          patterns=', '.join(f'\"{expr}\"' for expr in non_matching),
          fields=','.join(fields)
        )
      )

  return most_specific_matching

def expand(
    clazz: Type[NamedTuple], description=None, *,
    library: dict[str, Type[ConditionalDistribution]],
    skip_unknown: bool=False
):
  if description is None:
    return None

  matched = match(clazz._fields, description, skip_unknown=skip_unknown)
  distributions = clazz(**parse(matched, library=library))
  return distributions

def expand_with_condition(
    clazz: Type[NamedTuple], description=None, *,
    library: dict[str, Type[ConditionalDistribution]],
    skip_unknown: bool=False
):
  if description is None:
    return None, None

  matched = match(clazz._fields, description, skip_unknown=skip_unknown)
  condition, distribution = parse_with_condition(matched, library=library)
  distribution = clazz(**distribution)
  condition = clazz(**condition)

  return condition, distribution

def get_distribution(
    clazz, description,
    key, key_noise,
    library: dict[str, Type[ConditionalDistribution]],
    skip_unknown: bool=False
):
  condition, distribution = expand_with_condition(
    clazz, description.get(key, None),
    library=library, skip_unknown=skip_unknown
  )

  if key_noise in description:
    noise = expand(clazz, description[key_noise], library=library, skip_unknown=skip_unknown)
  else:
    noise = None

  return condition, distribution, noise

def load(
    model: MechanisticModel, spec: str, *additional_specs: str,
    root: str | None=None,
    warn: bool=True, skip_unknown: bool=False,
    library: dict[str, Type[ConditionalDistribution]] | None=None
):
  from ..mm import System

  if library is None:
    from . import __std_library__
    library = __std_library__

  description = combine(spec, *additional_specs, root=root, warn=warn)

  likelihood = expand(
    model.Observables, description.get(LIKELIHOOD, None),
    library=library, skip_unknown=skip_unknown
  )
  parameters_condition, parameters, parameters_noise = get_distribution(
    model.Parameters, description, PARAMETERS, PARAMETER_NOISE,
    library=library, skip_unknown=skip_unknown
  )
  controls_condition, controls, controls_noise = get_distribution(
    model.ControlParameters, description, CONTROLS, CONTROL_NOISE,
    library=library, skip_unknown=skip_unknown
  )
  timestamps_condition, timestamps, timestamps_noise = get_distribution(
    model.Observables, description, TIMESTAMPS, TIMESTAMPS_NOISE,
    library=library, skip_unknown=skip_unknown
  )

  condition = System(controls_condition, parameters_condition, timestamps_condition)
  distribution = System(controls, parameters, timestamps)
  noise = System(controls_noise, parameters_noise, timestamps_noise)

  return condition, distribution, noise, likelihood
