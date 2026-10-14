# E. coli 2024 macrokinetic growth model

`EColi2024` (`bless/mm/ecoli2024.py`) is the ground truth of the E. coli libraries. Its kinetic constants are named
after the acetate-cycling overflow model of Anane et al. 2017 (Biochem Eng J 125:23-30); the route structure is the
package's own. Glucose is taken up through an oxidative route (`q_ox`, limited by dissolved oxygen), an anaerobic
route (`q_an`) and an overflow route (`q_of`, producing acetate); every route is inhibited by acetate, the latter two
also by the oxidative flux. Acetate is re-assimilated (`q_a`, repressed by glucose) and respired. Every route has a
biomass yield; the respiring routes have an oxygen yield derived from it. Dissolved oxygen relaxes to the equilibrium
of its balance of transfer and uptake, and a first-order probe measures it. The reactor is a constant-volume
fed-batch: the dilution rate `f` (one value per feed phase, blended between the phases by the softmax of
`CONTROL_SHARPNESS`) and the feed glucose `S_f` are the controls.

## Variables

State: `X` biomass, g/L; `S` glucose, g/L; `A` acetate, g/L; `DOTa` dissolved oxygen, fraction of air saturation;
`DOTm` measured dissolved oxygen, fraction of air saturation. Observed: `X`, `S`, `A`, `DOTm`. Initial state: `X` and
`S` from the design, `A = 0`, `DOTa = DOTm = 1`.

## ODE system

With `monod(x, K) = x / (x + K)`:

```
k_s  = q_ox_max monod(Ki_a_ox, A) monod(S, K_s_ox)
k_a  = q_a_max  monod(Ki_s_a, S)  monod(A, K_a)
q_ox = k_s monod(DOTa, K_o)
q_a  = k_a monod(DOTa, K_o)
q_an = q_an_max monod(S, K_s_an) monod(Ki_a_an, A) monod(ki_ox_an q_ox_max, q_ox)
q_of = q_of_max monod(S, K_s_of) monod(Ki_a_of, A) monod(ki_ox_of q_ox_max, q_ox)
q_s  = q_ox + q_an + q_of
p_a  = q_of Y_a_of
mu   = q_ox Y_x_ox + q_an Y_x_an + q_of Y_x_of + q_a Y_x_a - m_0

dX / dt    = (mu - f) X
dS / dt    = -q_s X + (S_f - S) f
dA / dt    = (p_a - q_a) X - f A
dDOTa / dt = lambda (DOT_eq - DOTa)
dDOTm / dt = K_p (DOTa - DOTm)
```

Oxygen balance: `k_O = (k_s Y_o_ox + k_a Y_o_a) X H` is the oxygen demand at full oxygen (DOT fraction per hour) and
`K = K_La + f` the transfer rate; `DOT_eq` is the positive root of the balance
`g(DOT) = K (DOT_lim - DOT) - k_O DOT / (K_o + DOT)` (`oxygen_equilibrium`) and `lambda = g(DOTa) / (DOT_eq - DOTa)`
its secant rate (the tangent `K + k_O K_o / (K_o + DOTa)^2` within 1e-4 of the equilibrium; `oxygen_relaxation_rate`).
The oxygen yields are `Y_o = 1.067 (1 - 1.2 Y_x)` for `Y_x_ox` and `Y_x_a` (`oxygen_yield`). Inside the solver's stages
the two oxygen channels are advanced exactly with `DOT_eq` and `lambda` frozen (`oxygen_step`, `analytical_step`).
After every step `X`, `S`, `A` are clipped at zero and `DOTa`, `DOTm` to `[0, DOT_lim]`.

## Parameters

- `q_ox_max`, `q_an_max`, `q_of_max` --- maximal glucose uptake of the routes, g/g/h; `q_a_max` --- maximal acetate
  uptake, g/g/h;
- `K_s_ox`, `K_s_an`, `K_s_of` --- glucose half-saturation, g/L; `K_a` --- acetate half-saturation, g/L;
- `Ki_a_ox`, `Ki_a_an`, `Ki_a_of` --- acetate inhibition of glucose uptake, g/L; `Ki_s_a` --- glucose inhibition of
  acetate uptake, g/L;
- `ki_ox_an`, `ki_ox_of` --- inhibition of the anaerobic and overflow routes by the oxidative flux, dimensionless;
- `K_o` --- oxygen half-saturation, fraction of air saturation;
- `Y_x_ox`, `Y_x_an`, `Y_x_of` --- biomass per glucose of the routes, g/g; `Y_x_a` --- biomass per acetate, g/g;
  `Y_a_of` --- acetate per glucose through the overflow route, g/g;
- `m_0` --- maintenance, g/g/h;
- `H` --- inverse oxygen solubility, L/g; `K_La` --- volumetric oxygen transfer, 1/h; `K_p` --- probe response
  constant, 1/h; `DOT_lim` --- saturation level of the probe signal, fraction.

The prior `realistic.yaml` is written by hand with a literature source per range; the base priors of
`ecoli2024reduced`, `ecoli2024oxidative` and `ecoli-monod` are derived from it.
