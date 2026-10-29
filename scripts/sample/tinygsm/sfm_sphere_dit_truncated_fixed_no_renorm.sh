#!/bin/bash

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CKPT_PATH="${CKPT_PATH:-${REPO_ROOT}/checkpoints/tinygsm/sfm/sphere_dit_truncated_fixed_no_renorm.ckpt}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/eval_runs/tinygsm/sfm_sphere_dit_truncated_fixed_no_renorm}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-1}"
STEPS="${STEPS:-32}"
VELOCITY="${VELOCITY:-exact}"        # sample / exact
TOPK_VELOCITY="${TOPK_VELOCITY:--1}"   # 1 = top-1, -1 = full vocab (no top-k)

cd "${REPO_ROOT}"

python -u -m main \
    mode=gsm8k_eval \
    eval.checkpoint_path="${CKPT_PATH}" \
    eval.strict_loading=false \
    data=gsm8k-test \
    data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M \
    data.cache_dir="${CACHE_DIR}" \
    model=small-sphere-dit \
    model.init=ngpt \
    model.length=512 \
    algo=sfm \
    algo.renormalize_weights=False \
    noise=log-linear \
    noise.alpha_min=0.869 \
    noise.alpha_max=null \
    sampler=sfm \
    sampler.noise_removal=greedy \
    sampler.velocity="${VELOCITY}" \
    sampler.top_k_velocity="${TOPK_VELOCITY}" \
    sampler.steps="${STEPS}" \
    loader.eval_batch_size=32 \
    loader.num_workers=4 \
    trainer.num_nodes="${NUM_NODES}" \
    trainer.devices="${DEVICES}" \
    gsm8k.output_dir="${OUTPUT_DIR}" \
    +wandb.offline=True \
    hydra.run.dir="${OUTPUT_DIR}"
