# E. coli Monod base

`EColiMonod` (`bless/mm/ecoli_monod.py`) is the base of the `ecoli-monod` library against the truth `EColi2024`
(`config/mm/ecoli2024/README.md`): one glucose route with Monod kinetics, a biomass yield and a maintenance term; no
acetate and no oxygen (no dissolved-oxygen state, no oxygen limitation). The reactor is a constant-volume fed-batch:
the dilution rate `f` (one value per feed phase, blended by the softmax of `CONTROL_SHARPNESS`) and the feed glucose
`S_f` are the controls.

## Variables

State: `X` biomass, g/L; `S` glucose, g/L; both observed. Initial state: `X` and `S` from the design.

## ODE system

With `monod(x, K) = x / (x + K)`:

```
q_s = q_s_max monod(S, K_s)
mu  = q_s Y_s - m_0

dX / dt = (mu - f) X
dS / dt = -q_s X + (S_f - S) f
```

After every step `X` and `S` are clipped at zero.

## Parameters

- `q_s_max` --- maximal glucose uptake, g/g/h;
- `K_s` --- glucose half-saturation, g/L;
- `Y_s` --- biomass per glucose, g/g;
- `m_0` --- maintenance, g/g/h.

The prior `realistic.yaml` is derived from the truth's `config/mm/ecoli2024/realistic.yaml`: `K_s`, `m_0` from
`K_s_ox`, `m_0` at encoded +-2.5, `Y_s` from `Y_x_ox` widened, `q_s_max` from `q_ox_max` with the single-route
ceiling and a floor of 0.1.
