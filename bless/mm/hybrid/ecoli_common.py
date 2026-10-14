from ..extension import HybridExtension
from ..ecoli2024 import oxygen_equilibrium, oxygen_relaxation_rate

__all__ = [
  'OxygenResidual',
]

class OxygenResidual(HybridExtension):
  """
  The residual rate of `DOTa` taken into the oxygen balance: the equilibrium and the relaxation rate of the base's
  algebraics become those of the balance with the extra rate `source - gain DOTa`.
  """
  def algebraics(self, state, parameters, controls):
    algebraics = super().algebraics(state, parameters, controls)
    base, p = algebraics.base_algebraics, algebraics.base_parameters
    source, gain = algebraics.residuals['DOTa']
    K = p.K_La + controls.f
    DOT_eq = oxygen_equilibrium(K, p.DOT_lim, p.K_o, base.k_O, source, gain)
    DOT_rate = oxygen_relaxation_rate(K, p.DOT_lim, p.K_o, base.k_O, state.DOTa, DOT_eq, source, gain)
    return algebraics._replace(base_algebraics=base._replace(DOT_eq=DOT_eq, DOT_rate=DOT_rate))
