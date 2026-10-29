#!/bin/bash
# Run a command inside the NGC PyTorch container with the venv activated.
# Must be called on a compute node (inside a job).
#
#   scripts/abci_example/run.sh bash scripts/bct/eval_gsm8k.sh
#
# On ABCI the .sif image cannot be executed directly; it is unpacked into a
# sandbox on the node-local disk once per job.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/env.sh"
IMG="${PBS_LOCALDIR:-/local}/ngc-pytorch-25.02"
if [ ! -d "${IMG}" ]; then
  singularity build --sandbox "${IMG}.tmp.$$" "${SIF}" >&2
  mv -T "${IMG}.tmp.$$" "${IMG}" 2>/dev/null || rm -rf "${IMG}.tmp.$$"
fi
exec singularity exec --nv --bind /groups:/groups --bind /home:/home "${IMG}" \
  bash -c "source ${HERE}/env.sh && source ${VENV}/bin/activate && cd ${BCT_REPO} && $(printf '%q ' "$@")"
