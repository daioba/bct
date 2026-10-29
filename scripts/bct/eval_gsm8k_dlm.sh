#!/bin/bash
# Standard temperature scaling on the discrete diffusion baselines (MDLM, Duo)
# on GSM8K, using the released TinyGSM checkpoints of S-FLM.
#
#   MODEL=mdlm|duo T=0.4 SEED=1 \
#   CKPT_PATH=checkpoints/s-flm/tinygsm/mdlm.ckpt \
#   OUTPUT_DIR=outputs/gsm8k_dlm/mdlm/T0.4_seed1 \
#   bash scripts/bct/eval_gsm8k_dlm.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL="${MODEL:?mdlm or duo}"
CKPT_PATH="${CKPT_PATH:?set CKPT_PATH}"
T="${T:-1.0}"
SEED="${SEED:-1}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR}"
export STEPS="${STEPS:-1024}"
export DEVICES="${DEVICES:-8}"
export NUM_NODES="${NUM_NODES:-1}"

case "${MODEL}" in
  mdlm|duo) ;;
  *) echo "ERROR: unknown MODEL '${MODEL}'" >&2; exit 1 ;;
esac

if [ -f "${OUTPUT_DIR}/results.json" ]; then
  echo "${OUTPUT_DIR}/results.json exists; not overwriting." >&2
  exit 1
fi
mkdir -p "${OUTPUT_DIR}"

cd "${REPO_ROOT}"
# The GSM8K test set ships with the repository (data/gsm8k_test.json).
CKPT_PATH="${CKPT_PATH}" OUTPUT_DIR="${OUTPUT_DIR}" \
EXTRA_ARGS="sampler.temperature=${T} seed=${SEED} data.data_path=${REPO_ROOT}/data/gsm8k_test.json" \
  bash "scripts/sample/tinygsm/${MODEL}.sh"
