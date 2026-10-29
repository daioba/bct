#!/bin/bash
# Continue a released TinyGSM S-FLM checkpoint for ADD_STEPS optimizer updates.
#
#   ARM=cont  Cont.: the original S-FLM objective (matched-budget control)
#   ARM=tft   TFT: tempered-flow training with temperature conditioning
#
# The released checkpoint is resumed with its optimizer, LR scheduler, EMA,
# adaptive noise schedule and global step, so both arms start from the same
# state and differ only in the training objective.
#
# Example:
#   BACKBONE=sphere-arch ARM=tft \
#   CKPT_PATH=checkpoints/s-flm/tinygsm/sfm/sphere_arch_truncated_adaptive_no_renorm.ckpt \
#   OUTPUT_DIR=outputs/train/tinygsm_sarch_tft \
#   bash scripts/bct/train_tinygsm.sh
#
# The checkpoint used in the paper is OUTPUT_DIR/checkpoints/*-252000.ckpt
# (the released models are at step 250000).  CKPT_EVERY=500 additionally
# saves the 250500-step checkpoint of the held-out lambda table.
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BACKBONE="${BACKBONE:?sphere-arch or sphere-dit}"
ARM="${ARM:?cont or tft}"
CKPT_PATH="${CKPT_PATH:?released S-FLM checkpoint to continue}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
ADD_STEPS="${ADD_STEPS:-2000}"
CKPT_EVERY="${CKPT_EVERY:-${ADD_STEPS}}"
S0="${S0:-250000}"   # global step of the released checkpoints
DEVICES="${DEVICES:-8}"
NUM_NODES="${NUM_NODES:-1}"
SEED="${SEED:-1}"

case "${BACKBONE}" in
  sphere-arch)
    MODEL_ARGS="model=small-sphere-arch model.length=512 model.normalize_input_embed=False"
    RENORM=True ;;
  sphere-dit)
    MODEL_ARGS="model=small-sphere-dit model.init=ngpt model.length=512"
    RENORM=False ;;
  *) echo "ERROR: unknown BACKBONE '${BACKBONE}'" >&2; exit 1 ;;
esac

# TFT: per sequence, T = 1 with probability 0.2, T = 0.1 with probability 0.1,
# and otherwise beta = 1/T ~ U(1.25, 5).
case "${ARM}" in
  cont) TEMP_COND=False; ARM_ARGS="" ;;
  tft)  TEMP_COND=True
        ARM_ARGS="algo.bct_train_T_dist=mixture algo.bct_train_p_one=0.2 algo.bct_train_p_low=0.1 algo.bct_train_T_low=0.1 algo.bct_train_beta_min=1.25 algo.bct_train_beta_max=5.0" ;;
  *) echo "ERROR: unknown ARM '${ARM}'" >&2; exit 1 ;;
esac

GLOBAL_BS=512
BUF_SIZE=$((50 * GLOBAL_BS))
MAX_STEPS=$((S0 + ADD_STEPS))

if [ -d "${OUTPUT_DIR}/checkpoints" ]; then
  echo "${OUTPUT_DIR}/checkpoints exists; not overwriting." >&2
  exit 1
fi
mkdir -p "${OUTPUT_DIR}"

cd "${REPO_ROOT}"
python -u -c "
import sys, torch
gs = torch.load('${CKPT_PATH}', map_location='cpu', weights_only=False)['global_step']
if gs != ${S0}:
  sys.exit(f'checkpoint global_step is {gs}, expected S0=${S0}')
"

python -u -m main \
    data=tiny-gsm \
    data.cache_dir="${CACHE_DIR}" \
    data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M \
    data.wrap=False \
    data.train_on_prompt=False \
    data.train_on_pad=True \
    data.filter_too_long=True \
    ${MODEL_ARGS} \
    model.temp_conditioning="${TEMP_COND}" \
    algo=sfm \
    algo.renormalize_weights="${RENORM}" \
    algo.invert_time_convention=false \
    ${ARM_ARGS} \
    noise=log-linear-adaptive \
    noise.alpha_max=0.121 \
    noise.adaptive_plot_profile=false \
    noise.adaptive_refit_every=50 \
    noise.adaptive_buffer_size=${BUF_SIZE} \
    noise.adaptive_ema=0.9 \
    noise.adaptive_uniform_mix=1e-3 \
    loader.global_batch_size=${GLOBAL_BS} \
    loader.num_workers=16 \
    training.finetune_path='' \
    checkpointing.resume_from_ckpt=true \
    checkpointing.resume_ckpt_path="${CKPT_PATH}" \
    eval.generate_samples=False \
    seed="${SEED}" \
    trainer.num_nodes="${NUM_NODES}" \
    trainer.devices="${DEVICES}" \
    trainer.val_check_interval=10000 \
    trainer.max_steps="${MAX_STEPS}" \
    trainer.limit_val_batches=500 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps="${CKPT_EVERY}" \
    hydra.run.dir="${OUTPUT_DIR}"
