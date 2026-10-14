from .common import Solver, select
from .euler import Euler
from .rkc2 import RKC2

from ... import utils

__all__ = ['Solver', 'select', 'Euler', 'RKC2', 'from_config']

__solvers__ = {
  'euler': Euler,
  'rkc2': RKC2
}

def from_config(model, config, **overrides):
  solver, arguments = utils.config.extract(config, library=__solvers__)
  return solver(model, **{**arguments, **overrides})
