from .. import EColi2024Oxidative

from ..extension import LinBLESS, BLESS
from ..extension import SequentialBLESS, ResidualBLESS
from ..extension import Sequential, Residual
from ..extension import SequentialResidual
from .ecoli_common import OxygenResidual

__all__ = [
  'LinBLESSedEColi2024Oxidative',
  'BLESSedEColi2024Oxidative',
  'SequentialBLESSedEColi2024Oxidative',
  'ResidualBLESSedEColi2024Oxidative',
  'SequentialEColi2024Oxidative',
  'SequentialResidualEColi2024Oxidative',
  'ResidualEColi2024Oxidative',
]

ECOLI_OXIDATIVE_COUPLED_STATES = ['X', 'S', 'DOTa']
ECOLI_OXIDATIVE_COUPLED_PARAMETERS = [
  'q_ox_max',
  'K_s_ox',
  'K_o',
  'Y_x_ox',
  'm_0',
]

class LinBLESSedEColi2024Oxidative(OxygenResidual, LinBLESS):
  Base = EColi2024Oxidative

  coupled_states = ECOLI_OXIDATIVE_COUPLED_STATES
  coupled_parameters = ECOLI_OXIDATIVE_COUPLED_PARAMETERS

class BLESSedEColi2024Oxidative(OxygenResidual, BLESS):
  Base = EColi2024Oxidative

  coupled_states = ECOLI_OXIDATIVE_COUPLED_STATES
  coupled_parameters = ECOLI_OXIDATIVE_COUPLED_PARAMETERS

class SequentialEColi2024Oxidative(Sequential):
  Base = EColi2024Oxidative

  coupled_states = ECOLI_OXIDATIVE_COUPLED_STATES
  coupled_parameters = ECOLI_OXIDATIVE_COUPLED_PARAMETERS

class ResidualEColi2024Oxidative(OxygenResidual, Residual):
  Base = EColi2024Oxidative

  coupled_states = ECOLI_OXIDATIVE_COUPLED_STATES
  coupled_parameters = []

class SequentialResidualEColi2024Oxidative(OxygenResidual, SequentialResidual):
  Base = EColi2024Oxidative

  coupled_states = ECOLI_OXIDATIVE_COUPLED_STATES
  coupled_parameters = ECOLI_OXIDATIVE_COUPLED_PARAMETERS

class SequentialBLESSedEColi2024Oxidative(SequentialBLESS):
  Base = EColi2024Oxidative

  coupled_states = ECOLI_OXIDATIVE_COUPLED_STATES
  coupled_parameters = ECOLI_OXIDATIVE_COUPLED_PARAMETERS

class ResidualBLESSedEColi2024Oxidative(OxygenResidual, ResidualBLESS):
  Base = EColi2024Oxidative

  coupled_states = ECOLI_OXIDATIVE_COUPLED_STATES
  coupled_parameters = []
