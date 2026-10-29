# Example environment for running this repository on ABCI 3.0.
# Edit the group and the paths below; everything else is derived from them.
export ABCI_GROUP="${ABCI_GROUP:-gxx00000}"
export WORK="${WORK:-/groups/${ABCI_GROUP}/${USER}/bct}"
export BCT_REPO="${BCT_REPO:-${HOME}/bct}"

export SIF="${WORK}/sif/pytorch-25.02-py3.sif"   # NGC PyTorch container
export VENV="${WORK}/venv"                         # venv on top of it
export CACHE_DIR="${WORK}/data_cache"              # data.cache_dir
export HF_HOME="${WORK}/hf"
export CKPT="${WORK}/checkpoints"                  # downloaded checkpoints
export OUT="${WORK}/outputs"                       # evaluation outputs
mkdir -p "${WORK}" "${CACHE_DIR}" "${HF_HOME}" "${CKPT}" "${OUT}"
