"""
The seeded campaign: one dataset per (library, seed); every model of a library fitted for all seeds in one process
(`scripts/fit.py fit`: the seeds' systems in one batch).

    snakemake all --cores 11 --resources gpu=1                    # local executor
    snakemake all --executor slurm --jobs 6 --resources gpu=1     # SLURM

Scope comes from `config/snakemake.yaml`: the libraries, the seeds and the model patterns, matched against the
expanded library entries. The protocol lives in `config/fit.yaml`.
"""
import sys

import yaml

import bless

configfile: 'config/snakemake.yaml'

LIBRARIES = 'config/library'
### the interpreter running snakemake
PYTHON = sys.executable
ENV = 'PYTHONPATH=$PWD XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1'

ROOT = config['root']
SEEDS = config['seeds']
DEVICE = config['device']
SHARDS = config['shards']
ACCOUNT = config['account']

def models_of(library):
  with open(f'{LIBRARIES}/{library}.yaml') as f:
    return bless.utils.config.models(bless.utils.config.expand_library(yaml.safe_load(f)), config['models'])

targets = [
  f'{ROOT}/seed-{seed}/{library}/{model}/mle.npz'
  for library in config['libraries']
  for model in models_of(library)
  for seed in SEEDS
]

rule all:
  input: targets

rule dataset:
  output: f'{ROOT}/seed-{{seed}}/{{library}}/ground-truth.npz'
  resources:
    cpus_per_task=1,
    mem_mb=2500,
    runtime=45,
    slurm_account=ACCOUNT,
  shell:
    f'{ENV} {PYTHON} scripts/fit.py generate '
    'library={wildcards.library} seed={wildcards.seed} '
    f'root={ROOT} sampling.plot=0'

### one process per (library, model): every seed's dataset in, every seed's fit out; seeds already fitted are skipped
rule fit:
  input: expand(f'{ROOT}/seed-{{seed}}/{{{{library}}}}/ground-truth.npz', seed=SEEDS)
  output: expand(f'{ROOT}/seed-{{seed}}/{{{{library}}}}/{{{{model}}}}/mle.npz', seed=SEEDS)
  resources:
    cpus_per_task=1,
    mem_mb=3000,
    runtime=1440,
    slurm_account=ACCOUNT,
    slurm_extra=f"'--gres=shard:{SHARDS}'" if DEVICE == 'cuda' else '',
    ### one unit of `gpu` per fit; `--resources gpu=1` runs the fits on one card one at a time
    gpu=1 if DEVICE == 'cuda' else 0,
  shell:
    f'{ENV} {PYTHON} scripts/fit.py fit '
    'library={wildcards.library} model={wildcards.model} '
    f"'seeds=[{','.join(str(s) for s in SEEDS)}]' root={ROOT} device={DEVICE} progress=false fitting.plot=0"
