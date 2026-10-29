#!/bin/bash
# Sudoku-Hard S-FLM training with the Sudoku recipe of S-FLM, on one GPU.
#
#   ARM=base  the S-FLM model: 20,000 steps from scratch, with the arguments of
#             scripts/train/sudoku/sfm_truncated_adaptive.sh (DIFFICULTY=hard)
#   ARM=cont  the base checkpoint (CKPT_PATH) + ADD_STEPS updates, S-FLM objective
#   ARM=tft   the base checkpoint (CKPT_PATH) + ADD_STEPS updates, TFT
#             (see scripts/bct/train_tinygsm.sh)
#
#   ARM=base OUTPUT_DIR=outputs/train/sudoku_hard bash scripts/bct/train_sudoku_hard.sh
#   ARM=tft CKPT_PATH=outputs/train/sudoku_hard/checkpoints/<...>-20000.ckpt \
#   OUTPUT_DIR=outputs/train/sudoku_hard_tft bash scripts/bct/train_sudoku_hard.sh
#
# SUDOKU_WORKERS (default 16) processes generate the puzzles on the first run
# (a single worker takes hours); the puzzles do not depend on their number.
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ARM="${ARM:?base, cont or tft}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
ADD_STEPS="${ADD_STEPS:-2000}"
S0="${S0:-20000}"
NUM_NODES=1
DEVICES=1

case "${ARM}" in
  base) TEMP_COND=False; ARM_ARGS="" ;;
  cont) TEMP_COND=False; ARM_ARGS="" ;;
  tft)  TEMP_COND=True
        ARM_ARGS="algo.bct_train_T_dist=mixture algo.bct_train_p_one=0.2 algo.bct_train_p_low=0.1 algo.bct_train_T_low=0.1 algo.bct_train_beta_min=1.25 algo.bct_train_beta_max=5.0" ;;
  *) echo "ERROR: unknown ARM '${ARM}'" >&2; exit 1 ;;
esac

GLOBAL_BS=256
BUF_SIZE=$((50 * GLOBAL_BS))
if [ "${ARM}" = "base" ]; then
  # The released recipe: 20,000 steps, a checkpoint every 5,000 steps.
  MAX_STEPS=20000; CKPT_EVERY=5000; RESUME_ARGS=""
else
  CKPT_PATH="${CKPT_PATH:?Sudoku-Hard S-FLM checkpoint to continue}"
  MAX_STEPS=$((S0 + ADD_STEPS)); CKPT_EVERY="${ADD_STEPS}"
  RESUME_ARGS="checkpointing.resume_from_ckpt=true checkpointing.resume_ckpt_path=${CKPT_PATH}"
fi

if [ -d "${OUTPUT_DIR}/checkpoints" ]; then
  echo "${OUTPUT_DIR}/checkpoints exists; not overwriting." >&2
  exit 1
fi
mkdir -p "${OUTPUT_DIR}"

cd "${REPO_ROOT}"
# Same arguments as scripts/train/sudoku/sfm_truncated_adaptive.sh, except for
# the continuation (max_steps, checkpoint interval, resume, TFT) and the number
# of puzzle-generation workers.
python -u -m main \
    data=sudoku \
    data.cache_dir="${CACHE_DIR}" \
    data.difficulty=hard \
    model=tiny-sphere-dit \
    algo=sfm \
    algo.invert_time_convention=false \
    noise=log-linear-adaptive \
    noise.alpha_max=0.093 \
    noise.adaptive_refit_every=50 \
    noise.adaptive_buffer_size=${BUF_SIZE} \
    noise.adaptive_ema=0.9 \
    noise.adaptive_uniform_mix=1e-3 \
    loader.global_batch_size=${GLOBAL_BS} \
    loader.batch_size=256 \
    loader.eval_batch_size=256 \
    loader.num_workers=8 \
    eval.generate_samples=False \
    trainer.num_nodes="${NUM_NODES}" \
    trainer.devices="${DEVICES}" \
    trainer.val_check_interval=20_000 \
    trainer.limit_val_batches=0 \
    trainer.max_steps="${MAX_STEPS}" \
    callbacks.checkpoint_every_n_steps.every_n_train_steps="${CKPT_EVERY}" \
    hydra.run.dir="${OUTPUT_DIR}" \
    data.sudoku_num_workers="${SUDOKU_WORKERS:-16}" \
    model.temp_conditioning="${TEMP_COND}" \
    ${ARM_ARGS} \
    ${RESUME_ARGS}
