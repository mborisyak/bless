from .. import EColi2024Reduced

from ..extension import LinBLESS, BLESS
from ..extension import SequentialBLESS, ResidualBLESS
from ..extension import Sequential, Residual
from ..extension import SequentialResidual

from .ecoli_common import OxygenResidual

__all__ = [
  'LinBLESSedEColi2024Reduced',
  'BLESSedEColi2024Reduced',
  'SequentialBLESSedEColi2024Reduced',
  'ResidualBLESSedEColi2024Reduced',
  'SequentialEColi2024Reduced',
  'SequentialResidualEColi2024Reduced',
  'ResidualEColi2024Reduced',
]

ECOLI_COUPLED_STATES = ['X', 'S', 'DOTa']
ECOLI_COUPLED_PARAMETERS = [
  'q_ox_max', 'q_an_max',
  'K_s_ox',
  'K_s_an', 'ki_ox_an',
  'K_o',
  'Y_x_ox', 'Y_x_an',
  'm_0',
]

class LinBLESSedEColi2024Reduced(OxygenResidual, LinBLESS):
  Base = EColi2024Reduced

  coupled_states = ECOLI_COUPLED_STATES
  coupled_parameters = ECOLI_COUPLED_PARAMETERS

class BLESSedEColi2024Reduced(OxygenResidual, BLESS):
  Base = EColi2024Reduced

  coupled_states = ECOLI_COUPLED_STATES
  coupled_parameters = ECOLI_COUPLED_PARAMETERS

class SequentialEColi2024Reduced(Sequential):
  Base = EColi2024Reduced

  coupled_states = ECOLI_COUPLED_STATES
  coupled_parameters = ECOLI_COUPLED_PARAMETERS

class ResidualEColi2024Reduced(OxygenResidual, Residual):
  Base = EColi2024Reduced

  coupled_states = ECOLI_COUPLED_STATES
  coupled_parameters = []

class SequentialResidualEColi2024Reduced(OxygenResidual, SequentialResidual):
  Base = EColi2024Reduced

  coupled_states = ECOLI_COUPLED_STATES
  coupled_parameters = ECOLI_COUPLED_PARAMETERS

class SequentialBLESSedEColi2024Reduced(SequentialBLESS):
  Base = EColi2024Reduced

  coupled_states = ECOLI_COUPLED_STATES
  coupled_parameters = ECOLI_COUPLED_PARAMETERS

class ResidualBLESSedEColi2024Reduced(OxygenResidual, ResidualBLESS):
  Base = EColi2024Reduced

  coupled_states = ECOLI_COUPLED_STATES
  coupled_parameters = []
