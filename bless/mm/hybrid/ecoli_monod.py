from .. import EColiMonod

from ..extension import LinBLESS, BLESS
from ..extension import SequentialBLESS, ResidualBLESS
from ..extension import Sequential, Residual
from ..extension import SequentialResidual

__all__ = [
  'LinBLESSedEColiMonod',
  'BLESSedEColiMonod',
  'SequentialBLESSedEColiMonod',
  'ResidualBLESSedEColiMonod',
  'SequentialEColiMonod',
  'SequentialResidualEColiMonod',
  'ResidualEColiMonod',
]

ECOLI_MONOD_COUPLED_STATES = ['X', 'S']
ECOLI_MONOD_COUPLED_PARAMETERS = ['q_s_max', 'K_s', 'Y_s', 'm_0']

class LinBLESSedEColiMonod(LinBLESS):
  Base = EColiMonod

  coupled_states = ECOLI_MONOD_COUPLED_STATES
  coupled_parameters = ECOLI_MONOD_COUPLED_PARAMETERS

class BLESSedEColiMonod(BLESS):
  Base = EColiMonod

  coupled_states = ECOLI_MONOD_COUPLED_STATES
  coupled_parameters = ECOLI_MONOD_COUPLED_PARAMETERS

class SequentialEColiMonod(Sequential):
  Base = EColiMonod

  coupled_states = ECOLI_MONOD_COUPLED_STATES
  coupled_parameters = ECOLI_MONOD_COUPLED_PARAMETERS

class ResidualEColiMonod(Residual):
  Base = EColiMonod

  coupled_states = ECOLI_MONOD_COUPLED_STATES
  coupled_parameters = []

class SequentialResidualEColiMonod(SequentialResidual):
  Base = EColiMonod

  coupled_states = ECOLI_MONOD_COUPLED_STATES
  coupled_parameters = ECOLI_MONOD_COUPLED_PARAMETERS

class SequentialBLESSedEColiMonod(SequentialBLESS):
  Base = EColiMonod

  coupled_states = ECOLI_MONOD_COUPLED_STATES
  coupled_parameters = ECOLI_MONOD_COUPLED_PARAMETERS

class ResidualBLESSedEColiMonod(ResidualBLESS):
  Base = EColiMonod

  coupled_states = ECOLI_MONOD_COUPLED_STATES
  coupled_parameters = []
