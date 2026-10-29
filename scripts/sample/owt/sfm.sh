#!/bin/bash

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CKPT_PATH="${CKPT_PATH:-${REPO_ROOT}/checkpoints/owt/sfm.ckpt}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/eval_runs/owt/sfm}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-1}"
STEPS="${STEPS:-32}"
NUM_SAMPLE_BATCHES="${NUM_SAMPLE_BATCHES:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
VELOCITY="${VELOCITY:-exact}"        # sample / exact
TOPK_VELOCITY="${TOPK_VELOCITY:--1}"   # 1 = top-1, -1 = full vocab (no top-k)

GLOBAL_BS=512
BUF_SIZE=$((50 * GLOBAL_BS))

cd "${REPO_ROOT}"

python -u -m main \
    mode=sample_eval \
    eval.checkpoint_path="${CKPT_PATH}" \
    eval.strict_loading=false \
    eval.compute_generative_perplexity=True \
    data=openwebtext-split \
    data.cache_dir="${CACHE_DIR}" \
    model=small-sphere-dit \
    model.init=ngpt \
    algo=sfm \
    algo.renormalize_weights=False \
    algo.invert_time_convention=false \
    noise=log-linear-adaptive \
    noise.alpha_max=0.121 \
    noise.adaptive_refit_every=50 \
    noise.adaptive_buffer_size=${BUF_SIZE} \
    noise.adaptive_ema=0.9 \
    noise.adaptive_uniform_mix=1e-3 \
    sampler=sfm \
    sampler.noise_removal=greedy \
    sampler.velocity="${VELOCITY}" \
    sampler.top_k_velocity="${TOPK_VELOCITY}" \
    sampler.steps="${STEPS}" \
    sampler.num_sample_batches="${NUM_SAMPLE_BATCHES}" \
    loader.eval_batch_size="${EVAL_BATCH_SIZE}" \
    loader.num_workers=4 \
    trainer.num_nodes="${NUM_NODES}" \
    trainer.devices="${DEVICES}" \
    +wandb.offline=True \
    hydra.run.dir="${OUTPUT_DIR}"
