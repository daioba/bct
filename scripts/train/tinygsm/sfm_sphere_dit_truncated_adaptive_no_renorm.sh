#!/bin/bash

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/tinygsm/sfm_sphere_dit_truncated_adaptive_no_renorm}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-8}"

GLOBAL_BS=512
BUF_SIZE=$((50 * GLOBAL_BS))

cd "${REPO_ROOT}"

python -u -m main \
    data=tiny-gsm \
    data.cache_dir="${CACHE_DIR}" \
    data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M \
    data.wrap=False \
    data.train_on_prompt=False \
    data.train_on_pad=True \
    data.filter_too_long=True \
    model=small-sphere-dit \
    model.init=ngpt \
    model.length=512 \
    algo=sfm \
    algo.renormalize_weights=False \
    algo.invert_time_convention=false \
    noise=log-linear-adaptive \
    noise.alpha_max=0.121 \
    noise.adaptive_plot_profile=true \
    noise.adaptive_refit_every=50 \
    noise.adaptive_buffer_size=${BUF_SIZE} \
    noise.adaptive_ema=0.9 \
    noise.adaptive_uniform_mix=1e-3 \
    loader.global_batch_size=${GLOBAL_BS} \
    loader.batch_size=64 \
    loader.eval_batch_size=64 \
    loader.num_workers=16 \
    eval.generate_samples=False \
    trainer.num_nodes="${NUM_NODES}" \
    trainer.devices="${DEVICES}" \
    trainer.val_check_interval=10_000 \
    trainer.max_steps=250_000 \
    trainer.limit_val_batches=500 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=10_000 \
    hydra.run.dir="${OUTPUT_DIR}"
