# E. coli 2024 reduced model

`EColi2024Reduced` (`bless/mm/ecoli_reduced_2024.py`) is the base of the `ecoli` library against the truth `EColi2024`
(`config/mm/ecoli2024/README.md`). Its kinetic constants are named after the overflow model of Anane et al. 2017
(Biochem Eng J 125:23-30); the route structure is the package's own. It keeps the truth's oxidative route (`q_ox`,
limited by dissolved oxygen), its anaerobic route (`q_an`, inhibited by the oxidative flux) and the oxygen balance
with the probe, and has neither the overflow route nor acetate. The reactor is a constant-volume fed-batch: the
dilution rate `f` (one value per feed phase, blended by the softmax of `CONTROL_SHARPNESS`) and the feed glucose `S_f`
are the controls.

## Variables

State: `X` biomass, g/L; `S` glucose, g/L; `DOTa` dissolved oxygen, fraction of air saturation; `DOTm` measured
dissolved oxygen, fraction of air saturation. Observed: `X`, `S`, `DOTm`. Initial state: `X` and `S` from the design,
`DOTa = DOTm = 1`.

## ODE system

With `monod(x, K) = x / (x + K)`:

```
k_s  = q_ox_max monod(S, K_s_ox)
q_ox = k_s monod(DOTa, K_o)
q_an = q_an_max monod(S, K_s_an) monod(ki_ox_an q_ox_max, q_ox)
q_s  = q_ox + q_an
mu   = q_ox Y_x_ox + q_an Y_x_an - m_0

dX / dt    = (mu - f) X
dS / dt    = -q_s X + (S_f - S) f
dDOTa / dt = lambda (DOT_eq - DOTa)
dDOTm / dt = K_p (DOTa - DOTm)
```

Oxygen balance: `k_O = k_s Y_o_ox X H` is the oxygen demand at full oxygen (DOT fraction per hour) and `K = K_La + f`
the transfer rate; `DOT_eq` is the positive root of the balance `g(DOT) = K (DOT_lim - DOT) - k_O DOT / (K_o + DOT)`
and `lambda = g(DOTa) / (DOT_eq - DOTa)` its secant rate (the tangent within 1e-4 of the equilibrium); the oxygen
yield is `Y_o_ox = 1.067 (1 - 1.2 Y_x_ox)`. These are the functions `oxygen_equilibrium`, `oxygen_relaxation_rate`,
`oxygen_yield` and `oxygen_step` of `bless.mm.ecoli2024`; inside the solver's stages the two oxygen channels are
advanced exactly with `DOT_eq` and `lambda` frozen. After every step `X`, `S` are clipped at zero and `DOTa`, `DOTm` to
`[0, DOT_lim]`.

## Parameters

- `q_ox_max`, `q_an_max` --- maximal glucose uptake of the two routes, g/g/h;
- `K_s_ox`, `K_s_an` --- glucose half-saturation, g/L;
- `ki_ox_an` --- inhibition of the anaerobic route by the oxidative flux, dimensionless;
- `K_o` --- oxygen half-saturation, fraction of air saturation;
- `Y_x_ox`, `Y_x_an` --- biomass per glucose of the two routes, g/g;
- `m_0` --- maintenance, g/g/h;
- `H` --- inverse oxygen solubility, L/g; `K_La` --- volumetric oxygen transfer, 1/h; `K_p` --- probe response
  constant, 1/h; `DOT_lim` --- saturation level of the probe signal, fraction.

The prior `realistic.yaml` is derived from the truth's `config/mm/ecoli2024/realistic.yaml`: the truth's bounds at
encoded +-2.5 in the base's prior, the yields widened.
