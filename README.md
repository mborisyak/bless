# bless

Hybrid mechanistic models in JAX: a mechanistic base model extended by neural corrections, fitted by maximum
likelihood on simulated experiments of a richer ground-truth model. BLESS carries a latent state that corrects the
base's parameters and adds residual rates; Sequential and Residual are the memoryless correctors.

## Layout

- `bless/mm`: the mechanistic models (`ecoli2024` with its reduced, oxidative and Monod bases; `monod`, `regulatory`,
  `chain`), the solvers (`rkc2`, `euler`), the extension classes (`bless/mm/extension`), the hybrids per base
  (`bless/mm/hybrid`) and the registry `bless.mm.__models__`
- `bless/distributions`: priors, designs and likelihoods from the profiles under `config/mm/<model>/`
- `bless/fit`: gradient-based maximum likelihood with restarts
- `config/library/<library>.yaml`: one comparison (a ground truth, a baseline, the fitted models); `config/fit.yaml`:
  the fitting protocol; `config/snakemake.yaml`: the campaign
- `scripts`: the fits (`fit.py`) and the results table (`results_table.py`); `tests`; `examples`

## Running

The package is used from the working tree:

    export PYTHONPATH=$PWD

The campaign, a dataset per (library, seed) and every model fitted for all seeds in one process:

    snakemake all -c11 --resources gpu=1

Results: `scripts/results_table.py`. One dataset and one fit by hand, with `config/fit.yaml` as the base config:

    python scripts/fit.py generate library=ecoli seed=0 root=data/campaign
    python scripts/fit.py fit library=ecoli model=baseline 'seeds=[0]' root=data/campaign

Tests: `pytest tests`.
