from .common import *

from . import solver
from .solver import Euler, RKC2

from .ecoli2024 import EColi2024
from .ecoli_reduced_2024 import EColi2024Reduced
from .ecoli_oxidative_2024 import EColi2024Oxidative
from .ecoli_monod import EColiMonod
from .monod import Monod
from .chain import Chain
from .regulatory import RegulatoryMonod

from . import extension
from . import hybrid

from .. import utils

__models__: dict[str, type[common.MechanisticModel]] = {
  'ecoli2024': EColi2024,
  'ecoli2024reduced': EColi2024Reduced,
  'ecoli2024oxidative': EColi2024Oxidative,
  'ecoli-monod': EColiMonod,
  'monod': Monod,
  'chain': Chain,
  'regulatory': RegulatoryMonod,

  'lin-blessed-monod': hybrid.LinBLESSedMonod,
  'blessed-monod': hybrid.BLESSedMonod,
  'sequential-blessed-monod': hybrid.SequentialBLESSedMonod,
  'residual-blessed-monod': hybrid.ResidualBLESSedMonod,
  'sequential-monod': hybrid.SequentialMonod,
  'sequential-residual-monod': hybrid.SequentialResidualMonod,
  'residual-monod': hybrid.ResidualMonod,

  'lin-blessed-ecoli': hybrid.LinBLESSedEColi2024Reduced,
  'blessed-ecoli': hybrid.BLESSedEColi2024Reduced,
  'sequential-blessed-ecoli': hybrid.SequentialBLESSedEColi2024Reduced,
  'residual-blessed-ecoli': hybrid.ResidualBLESSedEColi2024Reduced,
  'sequential-ecoli': hybrid.SequentialEColi2024Reduced,
  'sequential-residual-ecoli': hybrid.SequentialResidualEColi2024Reduced,
  'residual-ecoli': hybrid.ResidualEColi2024Reduced,

  'lin-blessed-ecoli-oxidative': hybrid.LinBLESSedEColi2024Oxidative,
  'blessed-ecoli-oxidative': hybrid.BLESSedEColi2024Oxidative,
  'sequential-blessed-ecoli-oxidative': hybrid.SequentialBLESSedEColi2024Oxidative,
  'residual-blessed-ecoli-oxidative': hybrid.ResidualBLESSedEColi2024Oxidative,
  'sequential-ecoli-oxidative': hybrid.SequentialEColi2024Oxidative,
  'sequential-residual-ecoli-oxidative': hybrid.SequentialResidualEColi2024Oxidative,
  'residual-ecoli-oxidative': hybrid.ResidualEColi2024Oxidative,

  'lin-blessed-ecoli-monod': hybrid.LinBLESSedEColiMonod,
  'blessed-ecoli-monod': hybrid.BLESSedEColiMonod,
  'sequential-blessed-ecoli-monod': hybrid.SequentialBLESSedEColiMonod,
  'residual-blessed-ecoli-monod': hybrid.ResidualBLESSedEColiMonod,
  'sequential-ecoli-monod': hybrid.SequentialEColiMonod,
  'sequential-residual-ecoli-monod': hybrid.SequentialResidualEColiMonod,
  'residual-ecoli-monod': hybrid.ResidualEColiMonod,
}

def from_config(config):
  model, arguments = utils.config.extract(config, library=__models__)
  return model.from_config(**arguments)