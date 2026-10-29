#!/bin/bash
# Evaluate an S-FLM Sudoku-Hard checkpoint with standard tempering or BCT.
#
#   METHOD   standard | bcte | bctp | bctl   (see scripts/bct/eval_gsm8k.sh)
#   TFT      1 for a TFT checkpoint (required by bctl)
#   EVAL_SET test     the official Sudoku-Hard test set (2,000 puzzles)
#            heldout  2,000 held-out puzzles from the official generator with
#                     a different seed, used to select LAMBDA
#
# Example (TFT checkpoint, BCT-L with lambda = 1.0 at T = 0.1):
#   CKPT_PATH=checkpoints/bct/sudoku_hard/sfm/sfm_truncated_adaptive_tft.ckpt \
#   TFT=1 METHOD=bctl LAMBDA=1.0 T=0.1 SEED=1 \
#   OUTPUT_DIR=outputs/sudoku_hard/tft/bctl_T0.1_l1.0_seed1 \
#   bash scripts/bct/eval_sudoku_hard.sh
#
# Results are written to OUTPUT_DIR/results.json.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CKPT_PATH="${CKPT_PATH:?set CKPT_PATH}"
METHOD="${METHOD:?standard, bcte, bctp or bctl}"
T="${T:-1.0}"
LAMBDA="${LAMBDA:-1.0}"
SEED="${SEED:-1}"
EVAL_SET="${EVAL_SET:-test}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR}"
export STEPS="${STEPS:-180}"
export DEVICES=1
export NUM_NODES=1
export DIFFICULTY=hard
export VELOCITY=exact
export TOPK_VELOCITY=-1

# TFT=1 for a TFT checkpoint: it carries the temperature-conditioning
# parameters, which must be loaded for every method (also at T = 1).
if [ "${TFT:-0}" = "1" ]; then TEMP_COND=True; else TEMP_COND=False; fi
if [ "${METHOD}" = "bctl" ] && [ "${TEMP_COND}" != "True" ]; then
  echo "ERROR: METHOD=bctl needs a TFT checkpoint (set TFT=1)" >&2; exit 1
fi

case "${METHOD}" in
  standard) ARGS="sampler.temperature=${T}" ;;
  bcte)     ARGS="sampler.bct.mode=endpoint sampler.bct.T=${T}" ;;
  bctp)     ARGS="sampler.temperature=${T} sampler.bct.mode=path sampler.bct.T=${T}" ;;
  bctl)     ARGS="sampler.temperature=1.0 sampler.bct.mode=trained_path sampler.bct.T=${T} sampler.bct.correction_strength=${LAMBDA}" ;;
  *) echo "ERROR: unknown METHOD '${METHOD}'" >&2; exit 1 ;;
esac
ARGS="${ARGS} model.temp_conditioning=${TEMP_COND}"

case "${EVAL_SET}" in
  test) ;;
  heldout) ARGS="${ARGS} data.num_train=0 data.num_valid=2000 data.data_seed=1000000" ;;
  *) echo "ERROR: unknown EVAL_SET '${EVAL_SET}'" >&2; exit 1 ;;
esac
# The default seed of the config is 1.
if [ "${SEED}" != "1" ]; then ARGS="${ARGS} seed=${SEED}"; fi
# Worker processes that generate the puzzles on the first run (a single worker
# takes hours).  The generated puzzles do not depend on the number of workers.
ARGS="${ARGS} data.sudoku_num_workers=${SUDOKU_WORKERS:-16}"

if [ -f "${OUTPUT_DIR}/results.json" ]; then
  echo "${OUTPUT_DIR}/results.json exists; not overwriting." >&2
  exit 1
fi
mkdir -p "${OUTPUT_DIR}"

cd "${REPO_ROOT}"
CKPT_PATH="${CKPT_PATH}" OUTPUT_DIR="${OUTPUT_DIR}" EXTRA_ARGS="${ARGS}" \
  bash scripts/sample/sudoku/sfm_truncated_adaptive.sh
