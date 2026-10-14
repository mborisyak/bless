from .. import Monod
from ..extension import LinBLESS, BLESS
from ..extension import SequentialBLESS, ResidualBLESS
from ..extension import Sequential, Residual
from ..extension import SequentialResidual

__all__ = [
  'LinBLESSedMonod',
  'BLESSedMonod',
  'SequentialBLESSedMonod',
  'ResidualBLESSedMonod',
  'SequentialMonod',
  'SequentialResidualMonod',
  'ResidualMonod',
]

class LinBLESSedMonod(LinBLESS):
  Base = Monod

  coupled_states = ['A', ]
  coupled_parameters = ['q', 'K']

class BLESSedMonod(BLESS):
  Base = Monod

  coupled_states = ['A', ]
  coupled_parameters = ['q', 'K']

class SequentialMonod(Sequential):
  Base = Monod

  coupled_states = ['A', ]
  coupled_parameters = ['q', 'K']

class ResidualMonod(Residual):
  Base = Monod

  coupled_states = ['A', ]
  coupled_parameters = ['q', 'K']

class SequentialResidualMonod(SequentialResidual):
  Base = Monod

  coupled_states = ['A', ]
  coupled_parameters = ['q', 'K']

class SequentialBLESSedMonod(SequentialBLESS):
  Base = Monod

  coupled_states = ['A', ]
  coupled_parameters = ['q', 'K']

class ResidualBLESSedMonod(ResidualBLESS):
  Base = Monod

  coupled_states = ['A', ]
  coupled_parameters = ['q', 'K']
