#!/bin/bash
### One campaign fit on a CERN HTCondor GPU slot: the Snakefile's fit rule, `scripts/fit.py fit` for library $1 and
### model $2 over every seed of config/snakemake.yaml, in the LCG_110_cuda view plus blessvenv. Run from the AFS
### checkout; writes under its data/campaign.
set -uo pipefail

### condor does not set CUDA_VISIBLE_DEVICES; the assigned device, a MIG UUID on a partitioned card, comes in
### _CONDOR_AssignedGPUs
export CUDA_VISIBLE_DEVICES=${_CONDOR_AssignedGPUs:-}

D=/afs/cern.ch/work/m/maborisy
V=/cvmfs/sft.cern.ch/lcg/views/LCG_110_cuda/x86_64-el9-gcc13-opt

### setup.sh reads unbound variables and sets PYTHONHOME, so the venv's packages go on PYTHONPATH, after the repo
set +u
source $V/setup.sh
set -u
export PYTHONPATH=$D/bless-campaign:$D/blessvenv/lib/python3.13/site-packages:${PYTHONPATH:-}
### the view's cuDNN 9.3 is older than the 9.8 its jax was built with, which leaves XLA without DNN support: blessvenv's
### cuDNN 9.24 goes first on the library path
export LD_LIBRARY_PATH=$D/blessvenv/lib/python3.13/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
### no XLA command buffers (CUDA graphs), which fail to capture cuBLAS on H100s
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "host $(hostname), assigned ${_CONDOR_AssignedGPUs:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -2

cd $D/bless-campaign || exit 1
PYTHON=$D/blessvenv/bin/python
SEEDS=$($PYTHON -c "import yaml; print(','.join(str(s) for s in yaml.safe_load(open('config/snakemake.yaml'))['seeds']))")
$PYTHON -c 'import jax; print("jax", jax.__version__, jax.devices())'
$PYTHON scripts/fit.py fit library=$1 model=$2 "seeds=[$SEEDS]" root=data/campaign device=cuda progress=false fitting.plot=0
