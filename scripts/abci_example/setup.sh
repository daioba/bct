#!/bin/bash
#PBS -P gxx00000
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=1:00:00
#PBS -j oe
# One-time setup on ABCI: pull the NGC PyTorch container and build a venv on
# top of it.  Do not pip-install torch or numpy: the container ships builds
# matched to its CUDA / NCCL.
#
#   qsub -P <your group> scripts/abci_example/setup.sh
set -euo pipefail
cd "${PBS_O_WORKDIR}"
source scripts/abci_example/env.sh
mkdir -p "$(dirname "${SIF}")"
[ -f "${SIF}" ] || singularity pull "${SIF}" docker://nvcr.io/nvidia/pytorch:25.02-py3
IMG="${PBS_LOCALDIR:-/local}/ngc-pytorch-25.02"
singularity build --sandbox "${IMG}" "${SIF}"
singularity exec --nv --bind /groups:/groups --bind /home:/home "${IMG}" bash -c "
  set -e
  [ -d ${VENV} ] || python -m venv --system-site-packages ${VENV}
  source ${VENV}/bin/activate
  cd ${BCT_REPO}
  pip install -r requirements.txt
  python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
"
