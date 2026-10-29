#!/bin/bash
# Evaluate an S-FLM checkpoint on GSM8K with standard tempering or BCT.
#
#   METHOD   standard | bcte | bctp | bctl | top1
#              standard  standard posterior tempering, softmax(l / T)
#              bcte      BCT-E (training-free)
#              bctp      BCT-P (training-free)
#              bctl      BCT-L, for a TFT checkpoint; LAMBDA is the strength
#                        lambda (1.0 = full correction)
#              top1      the Top-1 velocity of S-FLM (temperature-invariant)
#   TFT      1 for a TFT checkpoint (required by bctl)
#   T        sampling temperature (T = 1 is the untempered model)
#
# At T = 1 every BCT correction is the identity; the paper uses METHOD=standard
# T=1.0 for the T = 1 entry of every method.
#
# Example (released S-FLM, BCT-P at T = 0.4):
#   CKPT_PATH=checkpoints/s-flm/tinygsm/sfm/sphere_arch_truncated_adaptive_no_renorm.ckpt \
#   BACKBONE=sphere-arch METHOD=bctp T=0.4 SEED=1 \
#   OUTPUT_DIR=outputs/gsm8k/sarch/released/bctp_T0.4_seed1 \
#   bash scripts/bct/eval_gsm8k.sh
#
# Set EVAL_SET=tinygsm_heldout and HELDOUT_DIR=<dir written by
# prepare_tinygsm_heldout.py> to evaluate on the TinyGSM held-out problems used
# to select LAMBDA instead of on GSM8K.
#
# Results are written to OUTPUT_DIR/results.json.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CKPT_PATH="${CKPT_PATH:?set CKPT_PATH}"
BACKBONE="${BACKBONE:?sphere-arch or sphere-dit}"
METHOD="${METHOD:?standard, bcte, bctp or bctl}"
T="${T:-1.0}"
LAMBDA="${LAMBDA:-1.0}"
SEED="${SEED:-1}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR}"
EVAL_SET="${EVAL_SET:-gsm8k}"
export STEPS="${STEPS:-1024}"
export DEVICES="${DEVICES:-8}"
export NUM_NODES="${NUM_NODES:-1}"
export VELOCITY=exact
export TOPK_VELOCITY=-1
if [ "${METHOD}" = "top1" ]; then export VELOCITY=sample TOPK_VELOCITY=1; fi

case "${BACKBONE}" in
  sphere-arch) SAMPLE_SCRIPT=scripts/sample/tinygsm/sfm_sphere_arch_truncated_adaptive_no_renorm.sh ;;
  sphere-dit)  SAMPLE_SCRIPT=scripts/sample/tinygsm/sfm_sphere_dit_truncated_adaptive_no_renorm.sh ;;
  *) echo "ERROR: unknown BACKBONE '${BACKBONE}'" >&2; exit 1 ;;
esac

# TFT=1 for a TFT checkpoint: it carries the temperature-conditioning
# parameters, which must be loaded for every method (also at T = 1).
if [ "${TFT:-0}" = "1" ]; then TEMP_COND=True; else TEMP_COND=False; fi
if [ "${METHOD}" = "bctl" ] && [ "${TEMP_COND}" != "True" ]; then
  echo "ERROR: METHOD=bctl needs a TFT checkpoint (set TFT=1)" >&2; exit 1
fi

case "${METHOD}" in
  standard) ARGS="sampler.temperature=${T} sampler.bct.mode=none sampler.bct.T=1.0" ;;
  top1)     ARGS="sampler.temperature=1.0 sampler.bct.mode=none sampler.bct.T=1.0" ;;
  bcte)     ARGS="sampler.temperature=1.0 sampler.bct.mode=endpoint sampler.bct.T=${T}" ;;
  bctp)     ARGS="sampler.temperature=${T} sampler.bct.mode=path sampler.bct.T=${T}" ;;
  bctl)     ARGS="sampler.temperature=1.0 sampler.bct.mode=trained_path sampler.bct.T=${T} sampler.bct.correction_strength=${LAMBDA}" ;;
  *) echo "ERROR: unknown METHOD '${METHOD}'" >&2; exit 1 ;;
esac

case "${EVAL_SET}" in
  gsm8k) ;;
  tinygsm_heldout)
    # The held-out problems are stored in the same format as the GSM8K test
    # cache, so evaluating on them only changes the cache directory.
    export CACHE_DIR="${HELDOUT_DIR:?set HELDOUT_DIR}" ;;
  *) echo "ERROR: unknown EVAL_SET '${EVAL_SET}'" >&2; exit 1 ;;
esac

if [ -f "${OUTPUT_DIR}/results.json" ]; then
  echo "${OUTPUT_DIR}/results.json exists; not overwriting." >&2
  exit 1
fi
mkdir -p "${OUTPUT_DIR}"

cd "${REPO_ROOT}"
# The GSM8K test set ships with the repository (data/gsm8k_test.json).
CKPT_PATH="${CKPT_PATH}" OUTPUT_DIR="${OUTPUT_DIR}" \
EXTRA_ARGS="${ARGS} model.temp_conditioning=${TEMP_COND} seed=${SEED} data.data_path=${REPO_ROOT}/data/gsm8k_test.json" \
  bash "${SAMPLE_SCRIPT}"
